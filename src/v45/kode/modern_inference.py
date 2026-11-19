"""Single-image inference for the selected classical class branches; no labels."""
import argparse,json
from pathlib import Path
from time import perf_counter
import cv2,joblib,numpy as np
from threadpoolctl import threadpool_limits
from classifier_v9 import extra
from calibrate_v7 import class_predictions
from inference import Model
from localize import Extractor,transformed,adjust_boxes
from localize_v4 import model_load,geometry_score
from vision import CLASSES,canonical,window_proposals,save_json,sha,rgb_read,summarize,render,membership

class ModernModel:
    def __init__(self,runs,selection,roi_config=None,consensus=None):
        self.runs=Path(runs);self.selection_path=Path(selection);self.selection=json.loads(self.selection_path.read_text());v4=self.runs/'localization_20261117_v4';v5=self.runs/'localization_20261117_v5';v8=self.runs/'localization_20261117_v8/models/100';v14=self.runs/'localization_20261117_v14'
        self.transform=model_load(v4/'models/linear');cfg=json.loads((v4/'models/linear/MODEL.json').read_text());assert sha(v4/'data/backgrounds.npz')==cfg['background_sha256'];self.rois=cfg['rois'];self.ext=Extractor(dict(np.load(v4/'data/backgrounds.npz')))
        qc=json.loads((v5/'quality/MODEL.json').read_text());assert sha(v5/'quality/fitted.joblib')==qc['model_sha256'];self.quality=joblib.load(v5/'quality/fitted.joblib');self.crop=Model(self.runs/'pilot_20261117_v2/models/initial')
        rc=json.loads((v8/'MODEL.json').read_text());assert sha(v8/'weights.npz')==rc['weights_sha256'];self.reg=dict(np.load(v8/'weights.npz'))
        chosen=json.loads((v14/'CLASSIFIER_TRIALS.json').read_text())['selected']['name'];p=v14/'models'/chosen;self.newcfg=json.loads((p/'MODEL.json').read_text());assert sha(p/'fitted.joblib')==self.newcfg['model_sha256'];self.newcrop=joblib.load(p/'fitted.joblib')
        allowed={'localization_20261117_v7','localization_20261117_v8/calibrated_100','localization_20261117_v13/calibrated','localization_20261117_v14/calibrated'}
        if not {p['branch'] for p in self.selection['classes'].values()}<=allowed:raise ValueError('Selected branch not yet implemented; refuse silent substitution')
        self.roi_override=None
        self.provenance=dict(selection_sha256=sha(self.selection_path),transform_sha256=cfg['model_sha256'],quality_sha256=qc['model_sha256'],crop_sha256=self.crop.config['weights_sha256'],wide_regression_sha256=rc['weights_sha256'],new_classifier_sha256=self.newcfg['model_sha256'],background_sha256=cfg['background_sha256'],lock_sha256=self.selection['lock_sha256'])
        if roi_config:
            self.roi_override=json.loads(Path(roi_config).read_text());assert self.roi_override['lock_sha256']==self.selection['lock_sha256'];assert sha(Path(self.roi_override['contact_model'])/'MODEL.json')==self.roi_override['contact_model_sha256'];self.provenance['roi_config_sha256']=sha(roi_config)
        self.consensus=None
        if consensus:
            self.consensus=json.loads(Path(consensus).read_text());assert self.roi_override is not None;assert self.consensus['model_selection_sha256']==sha(self.selection_path);assert self.consensus['roi_config_sha256']==sha(roi_config);self.provenance['consensus_sha256']=sha(consensus)
        self.model_id=canonical(self.provenance)[:20]

    def raw(self,rgb,camera):
        spec=self.rois[camera]
        if (rgb.shape[1],rgb.shape[0])!=(spec['width'],spec['height']):raise ValueError('Camera dimensions differ; framing must be verified before use')
        b=window_proposals(spec);prepared=self.ext.prepare(rgb,camera);x=np.concatenate([transformed(self.ext.batch(prepared,b[o:o+1024],camera),self.transform['pca'],self.transform['scaler']) for o in range(0,len(b),1024)])
        q=np.clip(self.quality.predict(x),0,1);adj=adjust_boxes(b,self.transform['reg'].predict(x),spec['width'],spec['height']);ids=np.flatnonzero(q>=.3);b=b[ids];adj=adj[ids];q=q[ids]
        if not len(b):return {name:dict(adjusted=adj,quality=q,geometry_adjusted=np.empty((0,4)),class_adjusted=np.array([],int)) for name in {x['branch'] for x in self.selection['classes'].values()}}
        feat=self.ext.hog.batch(prepared[0],adj);margins=feat@self.crop.weights['classifier_coef'].T+self.crop.weights['classifier_intercept'];labels=np.array([CLASSES.index(self.crop.config['class_order'][j]) for j in margins.argmax(1)]);geo=geometry_score(adj,camera,self.rois,self.transform['geometry']);base=dict(adjusted=adj,quality=q,geometry_adjusted=geo,class_adjusted=labels);raw={}
        for name in {x['branch'] for x in self.selection['classes'].values()}:
            rr=dict(base)
            if name=='localization_20261117_v13/calibrated':
                top=np.sort(margins,axis=1)[:,-2:];rr['quality']=q/(1+np.exp(-np.clip(top[:,1]-top[:,0],0,50)))
            elif name=='localization_20261117_v8/calibrated_100':
                old=self.ext.hog.batch(prepared[0],b);wide=adjust_boxes(b,old@self.reg['coef'].T+self.reg['intercept'],spec['width'],spec['height']);qs=[];ls=[]
                for off in range(0,len(wide),1024):
                    full=self.ext.batch(prepared,wide[off:off+1024],camera);qs.extend(np.clip(self.quality.predict(transformed(full,self.transform['pca'],self.transform['scaler'])),0,1));cm=full[:,:1764]@self.crop.weights['classifier_coef'].T+self.crop.weights['classifier_intercept'];ls.extend(CLASSES.index(self.crop.config['class_order'][j]) for j in cm.argmax(1))
                rr=dict(adjusted=wide,quality=np.array(qs),geometry_adjusted=geometry_score(wide,camera,self.rois,self.transform['geometry']),class_adjusted=np.array(ls))
            elif name=='localization_20261117_v14/calibrated':
                stats=np.array([extra(rgb,box) for box in adj]);xx=np.c_[feat,stats*self.newcfg['extra_scale']];rr['class_adjusted']=np.array([CLASSES.index(c) for c in self.newcrop.predict(xx)])
            raw[name]=rr
        return raw

    def predict(self,rgb,camera):
        t=perf_counter();raw=self.raw(rgb,camera);ps=[]
        for cls,cfg in self.selection['classes'].items():
            p=cfg['global_params'] if self.selection['mode']=='global' else cfg['per_camera'][camera]
            if self.consensus:
                from consensus_v20 import consensus_predictions
                c=self.consensus['classes'][cls];ps.extend(consensus_predictions(raw[cfg['branch']],camera,self.rois,cls,p,self.roi_override,c['vote_iou'],c['nested']))
            else:ps.extend(class_predictions(raw[cfg['branch']],camera,self.rois,cls,p['threshold'],p['nms'],p['prior']))
        if self.roi_override:
            for p in ps:
                label,point,dist=membership(p['bbox_xyxy'],p['class_name'],camera,self.rois,self.roi_override['rules'][camera],self.roi_override['offsets']);p.update(roi_membership=label,contact_xy=point)
        return dict(status='ok',model_id=self.model_id,model_provenance=self.provenance,camera_id=camera,predictions=ps,summary=summarize(ps,self.rois[camera]),elapsed_seconds=perf_counter()-t,score_semantics='Uncalibrated localization/class ranking score, not probability')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--roi-config',type=Path);p.add_argument('--consensus',type=Path);p.add_argument('--image',type=Path,required=True);p.add_argument('--camera',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        model=ModernModel(a.runs,a.selection,a.roi_config,a.consensus);rgb=rgb_read(a.image);result=model.predict(rgb,a.camera);result['image_sha256']=sha(a.image);save_json(a.output.with_suffix('.json'),result);render(rgb,result,model.rois[a.camera],a.output.with_suffix('.jpg'),title='Classical CCTV PoC | validation development model',score_label='kualitas gabungan, bukan probabilitas');print(json.dumps(result['summary'],ensure_ascii=False))
