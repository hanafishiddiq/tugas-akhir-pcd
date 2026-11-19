"""Single-image inference for a hash-bound classical camera/class assembly."""
import argparse,json
from pathlib import Path
from lokasi_aset import lokasi_aset
from time import perf_counter
import cv2,joblib,numpy as np
from threadpoolctl import threadpool_limits
from modern_inference import ModernModel
from classifier_v9 import extra
from classifier_v18 import letterbox
from consensus_v20 import consensus_predictions
from localize import transformed,adjust_boxes
from localize_v4 import geometry_score
from vision import CLASSES,canonical,window_proposals,rgb_read,save_json,sha,summarize,render

class AssembledModel:
    def __init__(self,runs,assembly):
        self.runs=Path(runs);self.assembly_path=Path(assembly);self.cfg=json.loads(self.assembly_path.read_text());self.base=ModernModel(self.runs,self.runs/'localization_20261117_v16/MODEL_SELECTION.json');self.rois=self.base.rois
        roi_path=lokasi_aset(self.cfg['roi_config']);assert sha(roi_path)==self.cfg['roi_config_sha256'];self.roi=json.loads(roi_path.read_text());assert self.cfg['lock_sha256']==self.roi['lock_sha256']==self.base.provenance['lock_sha256']
        self.provenance=dict(self.base.provenance,assembly_sha256=sha(self.assembly_path),roi_sha256=sha(roi_path));self.forest=None;self.camera_reg={};self.classifiers={};self.custom_quality={}
        required={v['branch'] for c in self.cfg['selected'].values() for v in c.values()};specs=[self.cfg['specs'][k] for k in required]
        if any(s['quality'] in ('forest','mean') for s in specs):
            p=self.runs/'localization_20261117_v22';c=json.loads((p/'MODEL.json').read_text());assert sha(p/'fitted.joblib')==c['model_sha256'];self.forest=joblib.load(p/'fitted.joblib');self.provenance['forest_sha256']=c['model_sha256']
        for s in specs:
            path=s.get('quality_model')
            if not path or path in self.custom_quality:continue
            p=lokasi_aset(path);c=json.loads((p/'MODEL.json').read_text());assert sha(p/'fitted.joblib')==c['model_sha256'];self.custom_quality[path]=joblib.load(p/'fitted.joblib');self.provenance['custom_quality_'+p.name]=c['model_sha256']
        if any(s['regression']=='camera' for s in specs):
            p=self.runs/'localization_20261117_v23';c=json.loads((p/'MODEL.json').read_text())
            for camera,info in c['cameras'].items():
                path=p/'models'/camera/'weights.npz';assert sha(path)==info['weights_sha256'];self.camera_reg[camera]=dict(np.load(path));self.provenance[camera+'_regression_sha256']=info['weights_sha256']
        for s in specs:
            name=s['classifier']
            if name in ('v2','v14') or name in self.classifiers:continue
            if name=='v18':
                p=self.runs/'localization_20261117_v18';which=json.loads((p/'CLASSIFIER_TRIALS.json').read_text())['selected']['name'];p=p/'models'/which
            elif s.get('classifier_model'):p=lokasi_aset(s['classifier_model'])
            else:raise ValueError('Unknown classifier '+name)
            c=json.loads((p/'MODEL.json').read_text());assert sha(p/'fitted.joblib')==c['model_sha256'];self.classifiers[name]=(joblib.load(p/'fitted.joblib'),c);self.provenance[name+'_sha256']=c['model_sha256']
        self.model_id=canonical(self.provenance)[:20]

    def raw(self,rgb,camera):
        spec=self.rois[camera];assert (rgb.shape[1],rgb.shape[0])==(spec['width'],spec['height']),'Framing dimensions differ'
        base=self.base;ext=base.ext;transform=base.transform;prep=ext.prepare(rgb,camera);b=window_proposals(spec);x=np.concatenate([transformed(ext.batch(prep,b[o:o+1024],camera),transform['pca'],transform['scaler']) for o in range(0,len(b),1024)]);q=np.clip(base.quality.predict(x),0,1);adj=adjust_boxes(b,transform['reg'].predict(x),spec['width'],spec['height']);keep=q>=.3;b=b[keep];q=q[keep];adj=adj[keep]
        required={v[camera if cls=='motor' else 'global']['branch'] for cls,v in self.cfg['selected'].items()};out={};box_cache={};quality_cache={};classifier_cache={}
        if not len(b):return {n:dict(adjusted=adj,quality=q,geometry_adjusted=np.empty((0,4)),class_adjusted=np.empty(0,int)) for n in required}
        original_hog=None
        for name in required:
            s=self.cfg['specs'][name];reg=s['regression']
            if reg not in box_cache:
                if reg=='base':boxes=adj
                else:
                    if original_hog is None:original_hog=ext.hog.batch(prep[0],b)
                    weights=base.reg if reg=='wide_global' else self.camera_reg[camera] if reg=='camera' else None
                    if weights is None:raise ValueError('Unknown box regression')
                    boxes=adjust_boxes(b,original_hog@weights['coef'].T+weights['intercept'],spec['width'],spec['height'])
                box_cache[reg]=dict(boxes=boxes,hog=ext.hog.batch(prep[0],boxes),geometry=geometry_score(boxes,camera,self.rois,transform['geometry']))
            cache=box_cache[reg];boxes=cache['boxes'];feat=cache['hog'];ckey=(reg,s['classifier'])
            if ckey not in classifier_cache:
                if s['classifier']=='v2':
                    margins=feat@base.crop.weights['classifier_coef'].T+base.crop.weights['classifier_intercept'];labels=np.array([CLASSES.index(base.crop.config['class_order'][j]) for j in margins.argmax(1)])
                else:
                    if 'stats' not in cache:cache['stats']=np.array([extra(rgb,box) for box in boxes])
                    if s['classifier']=='v14':model=base.newcrop;cfg=base.newcfg
                    else:model,cfg=self.classifiers[s['classifier']]
                    family=cfg.get('family','standard')
                    if family in ('combined','letterbox'):
                        lb=np.array([letterbox(ext.hog,prep[0],box) for box in boxes]);xx=np.c_[feat,lb,cache['stats']] if family=='combined' else np.c_[lb,cache['stats']]
                    else:xx=np.c_[feat,cache['stats']*cfg.get('extra_scale',1.)]
                    margins=model.decision_function(xx);labels=np.array([CLASSES.index(c) if c in CLASSES else -1 for c in model.predict(xx)])
                classifier_cache[ckey]=(labels,margins)
            labels,margins=classifier_cache[ckey];kind=s['quality'];qkey=(reg,kind,s.get('quality_model',''),s.get('probability_class',''))
            if qkey not in quality_cache:
                if kind=='base':scores=q
                elif kind=='joint_v2':
                    top=np.sort(margins,axis=1)[:,-2:];scores=q/(1+np.exp(-np.clip(top[:,1]-top[:,0],0,50)))
                else:
                    qs=[]
                    for off in range(0,len(boxes),1024):
                        full=ext.batch(prep,boxes[off:off+1024],camera)
                        if kind=='rescored':qs.extend(np.clip(base.quality.predict(transformed(full,transform['pca'],transform['scaler'])),0,1))
                        elif kind in ('forest','mean'):
                            quality_model=self.custom_quality[s['quality_model']] if s.get('quality_model') else self.forest
                            qs.extend(np.clip(quality_model.predict(full),0,1))
                        elif kind in ('class_probability','class_probability_times_base'):
                            quality_model=self.custom_quality[s['quality_model']];ci=list(quality_model.classes_).index(s['probability_class'])
                            qs.extend(quality_model.predict_proba(full)[:,ci])
                        else:raise ValueError('Unknown quality branch')
                    scores=np.asarray(qs)
                    if kind=='mean':scores=(scores+q)/2
                    elif kind=='class_probability_times_base':scores=scores*q
                quality_cache[qkey]=scores
            scores=quality_cache[qkey]
            if s.get('multiply_top_margin'):
                top=np.sort(margins,axis=1)[:,-2:]
                scores=scores/(1+np.exp(-np.clip(top[:,1]-top[:,0],0,50)))
            if s.get('probability_class'):labels=np.full(len(boxes),CLASSES.index(s['probability_class']),int)
            if s.get('class_score'):
                target=s['class_score']
                classes=base.crop.config['class_order'] if s['classifier']=='v2' else (base.newcrop.classes_ if s['classifier']=='v14' else self.classifiers[s['classifier']][0].classes_)
                ci=list(classes).index(target);gap=margins[:,ci]-np.max(np.delete(margins,ci,axis=1),axis=1)
                scores=scores/(1+np.exp(-np.clip(gap,-50,50)));labels=np.full(len(boxes),CLASSES.index(target),int)
            out[name]=dict(adjusted=boxes,quality=scores,class_adjusted=labels,geometry_adjusted=cache['geometry'])
        return out

    def predict(self,rgb,camera):
        start=perf_counter();raw=self.raw(rgb,camera);ps=[]
        for cls,options in self.cfg['selected'].items():
            c=options[camera if cls=='motor' else 'global'];ps.extend(consensus_predictions(raw[c['branch']],camera,self.rois,cls,c,self.roi,c['vote_iou'],c['nested']))
        if self.cfg.get('postprocess'):
            from consistency_v31 import filter_predictions
            ps=filter_predictions(ps,**self.cfg['postprocess'])
        if self.cfg.get('cross_class'):
            from cross_class_v34 import cross_filter
            ps=cross_filter(ps,camera,self.cfg['selected'],**self.cfg['cross_class'])
        return dict(status='ok',camera_id=camera,model_id=self.model_id,model_provenance=self.provenance,predictions=ps,summary=summarize(ps,self.rois[camera]),elapsed_seconds=perf_counter()-start,score_semantics='Uncalibrated localization/class ranking score, not probability')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--assembly',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--camera',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        m=AssembledModel(a.runs,a.assembly);rgb=rgb_read(a.image);result=m.predict(rgb,a.camera);result['image_sha256']=sha(a.image);save_json(a.output.with_suffix('.json'),result);render(rgb,result,m.rois[a.camera],a.output.with_suffix('.jpg'),title='Classical CCTV PoC | development model',score_label='kualitas gabungan, bukan probabilitas');print(json.dumps(result['summary']))
