"""TRAIN-only classical residual box correction, selected on grouped TRAIN."""
import argparse,copy,json
from collections import Counter
from pathlib import Path
from lokasi_aset import lokasi_aset
from time import perf_counter
import cv2,joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from data_access import PilotData
from experiment import add_f1,source_snapshot
from localize import Extractor,adjust_boxes
from assembled_inference import AssembledModel
from vision import membership,overlaps,render,save_json,sha,summarize

def geometry(boxes,camera,rois):
    w,h=rois[camera]['width'],rois[camera]['height'];b=np.asarray(boxes);wh=b[:,2:]-b[:,:2]
    return np.c_[b/np.array([w,h,w,h]),wh/np.array([w,h]),np.log(wh[:,0]/wh[:,1]),np.tile([float(camera==c) for c in sorted(rois)],(len(b),1))].astype(np.float32)

def object_weights(rows):
    counts=Counter(r['object_id'] for r in rows)
    return np.array([len(rows)/(len(counts)*counts[r['object_id']]) for r in rows])

def correction(boxes,delta,width,height,strength=1.):
    delta=np.asarray(delta).copy()*strength
    delta[:,:2]=np.clip(delta[:,:2],-.25,.25);delta[:,2:]=np.clip(delta[:,2:],-.4,.4)
    return adjust_boxes(np.asarray(boxes),delta,width,height)

def train(d,runs,out):
    base=runs/'localization_20261117_v8';cfg=json.loads((base/'models/100/MODEL.json').read_text())
    assert sha(base/'data/training.npz')==cfg['training_sha256'];assert sha(base/'data/SAMPLES.json')==cfg['samples_sha256']
    old=json.loads((base/'data/SAMPLES.json').read_text());ids=[j for j,r in enumerate(old) if r['source_class']=='motor'];rows=[old[j] for j in ids];z=np.load(base/'data/training.npz');y=z['y'][ids]
    ext=Extractor(dict(np.load(runs/'localization_20261117_v4/data/backgrounds.npz')));x=np.empty((len(rows),1950),np.float32);xridge=np.empty((len(rows),1774),np.float32)
    for i in sorted({r['image_id'] for r in rows}):
        f=d.frames[i];loc=[j for j,r in enumerate(rows) if r['image_id']==i];assert d.groups[i]['split']=='train'
        for j in loc:
            r=rows[j];assert r['split']=='train' and r['source_sha256']==f['source_sha256'];obj=next(o for o in f['objects'] if o['id']==r['object_id']);assert d.adapter.accepted(obj) and obj['bbox_xyxy']==r['target_bbox'] and obj['class_name']=='motor'
        b=np.array([rows[j]['candidate_bbox'] for j in loc]);rgb=d.read(f,'train','v42_residual_features');features=ext.batch(ext.prepare(rgb,f['camera_id']),b,f['camera_id']);x[loc]=features
        assert np.allclose(features[:,:1764],z['x'][np.array(ids)[loc]],atol=1e-6)
        xridge[loc]=np.c_[features[:,:1764],geometry(b,f['camera_id'],d.rois)]
    np.savez_compressed(out/'TRAIN_FEATURES.npz',full=x,ridge=xridge,y=y);save_json(out/'TRAIN_MANIFEST.json',rows);save_json(out/'PIXEL_ACCESS_fit.json',d.access)
    groups=np.array([d.groups[r['image_id']]['component_id'] for r in rows]);fitids,checkids=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=20261118).split(x,groups=groups))
    assert not set(groups[fitids])&set(groups[checkids]);w=object_weights(rows);vw=object_weights([rows[j] for j in checkids]);targets=np.array([rows[j]['target_bbox'] for j in checkids]);boxes=np.array([rows[j]['candidate_bbox'] for j in checkids])
    def measure(delta):
        ious=[]
        for j,(b,t,de) in enumerate(zip(boxes,targets,delta)):
            f=d.frames[rows[checkids[j]]['image_id']];bb=correction(b[None],de[None],f['width'],f['height'])[0];ious.append(overlaps(t,bb[None])[0])
        return dict(mean_iou=float(np.average(ious,weights=vw)),iou50=float(np.average(np.array(ious)>=.5,weights=vw)))
    factories={'ridge':lambda:Ridge(alpha=100),'forest':lambda:ExtraTreesRegressor(n_estimators=128,max_features=.33,min_samples_leaf=3,max_depth=24,random_state=20261117,n_jobs=4)}
    features={'ridge':xridge,'forest':x};trials={'identity':measure(np.zeros((len(checkids),4)))};best='identity'
    for name,factory in factories.items():
        start=perf_counter();m=factory().fit(features[name][fitids],y[fitids],sample_weight=object_weights([rows[j] for j in fitids]));trials[name]=dict(**measure(m.predict(features[name][checkids])),fit_seconds=perf_counter()-start)
        print(json.dumps(dict(inner_trial=name,**trials[name])),flush=True)
        if trials[name]['mean_iou']>trials[best]['mean_iou']:best=name
    save_json(out/'INNER_TRAIN_SELECTION.json',dict(trials=trials,selected=best,fit_groups=sorted(set(groups[fitids])),check_groups=sorted(set(groups[checkids])),selection_metric='object_weighted_mean_iou',seed=20261118))
    if best=='identity':return False
    start=perf_counter();m=factories[best]().fit(features[best],y,sample_weight=w);joblib.dump(m,out/'fitted.joblib',compress=3)
    save_json(out/'MODEL.json',dict(family=best,parameters=m.get_params(),model_sha256=sha(out/'fitted.joblib'),training_features_sha256=sha(out/'TRAIN_FEATURES.npz'),training_manifest_sha256=sha(out/'TRAIN_MANIFEST.json'),parent_training_sha256=cfg['training_sha256'],parent_manifest_sha256=cfg['samples_sha256'],background_path=str((runs/'localization_20261117_v4/data/backgrounds.npz').resolve()),background_sha256=sha(runs/'localization_20261117_v4/data/backgrounds.npz'),lock_sha256=sha(d.package/'SPLIT_LOCK.json'),rows=len(rows),objects=len({r['object_id'] for r in rows}),fit_seconds=perf_counter()-start,classes=['motor']))
    print(json.dumps(dict(fitted=best,seconds=perf_counter()-start)),flush=True);return True

class ResidualModel:
    def __init__(self,run,rois):
        self.run=Path(run);self.cfg=json.loads((self.run/'MODEL.json').read_text());assert sha(self.run/'fitted.joblib')==self.cfg['model_sha256'];self.model=joblib.load(self.run/'fitted.joblib');p=lokasi_aset(self.cfg['background_path']);assert sha(p)==self.cfg['background_sha256'];self.ext=Extractor(dict(np.load(p)));self.rois=rois
    def deltas(self,rgb,camera,predictions):
        selected=[j for j,p in enumerate(predictions) if p['class_name']=='motor'];out=np.zeros((len(predictions),4))
        if not selected:return out
        b=np.array([predictions[j]['bbox_xyxy'] for j in selected]);features=self.ext.batch(self.ext.prepare(rgb,camera),b,camera)
        if self.cfg['family']=='ridge':features=np.c_[features[:,:1764],geometry(b,camera,self.rois)]
        out[selected]=self.model.predict(features);return out
    def apply(self,predictions,delta,camera,roi,strength):
        ps=copy.deepcopy(predictions);spec=self.rois[camera]
        for j,p in enumerate(ps):
            if p['class_name']!='motor':continue
            b=correction(np.array([p['bbox_xyxy']]),delta[j:j+1],spec['width'],spec['height'],strength)[0].tolist();member,point,_=membership(b,p['class_name'],camera,self.rois,roi['rules'][camera],roi['offsets']);p.update(bbox_xyxy=b,roi_membership=member,contact_xy=point)
        return ps

def evaluate(d,runs,out):
    parent=runs/'localization_20261117_v35';cfg=json.loads((parent/'ASSEMBLY.json').read_text());roi=json.loads(Path(cfg['roi_config']).read_text());pred=json.loads((parent/'evaluation/VALIDATION_PREDICTIONS.json').read_text());baseline=json.loads((parent/'evaluation/VALIDATION_METRICS.json').read_text());model=ResidualModel(out,d.rois);frames=d.selected('validation');deltas={}
    for f in frames:deltas[f['image_id']]=model.deltas(d.read(f,'validation','v42_score'),f['camera_id'],pred[f['image_id']]['predictions'])
    np.savez_compressed(out/'VALIDATION_DELTAS.npz',**deltas);trials=[];best=baseline;bestpred=pred;chosen=None
    for strength in (.5,1.,1.5):
        predictions={}
        for f in frames:
            i=f['image_id'];ps=model.apply(pred[i]['predictions'],deltas[i],f['camera_id'],roi,strength);predictions[i]=dict(status='ok',predictions=ps,summary=summarize(ps,d.rois[f['camera_id']]))
        report=add_f1(d.evaluator.evaluate_holdout(frames,d.masks,d.groups,predictions,subset='validation'));s=report['summary'];bs=baseline['summary'];eligible=s['macro_f1']>bs['macro_f1'] and all(s['detection_by_class'][c]['f1']>=bs['detection_by_class'][c]['f1'] for c in bs['detection_by_class']) and s['mae_full_only']['total']<=1 and s['mae_full_only']['coverage_pp']<=5
        trial=dict(strength=strength,macro_f1=s['macro_f1'],mae=s['mae_full_only'],by_class=s['detection_by_class'],eligible=eligible);trials.append(trial);path=out/f'strength_{strength:g}';save_json(path/'VALIDATION_METRICS.json',report);save_json(path/'VALIDATION_PREDICTIONS.json',predictions);print(json.dumps(trial),flush=True)
        if eligible and s['macro_f1']>best['summary']['macro_f1']:chosen=strength;best=report;bestpred=predictions
    best.update(test_evaluated=False,selection_split='validation');save_json(out/'evaluation/VALIDATION_METRICS.json',best);save_json(out/'evaluation/VALIDATION_PREDICTIONS.json',bestpred);save_json(out/'evaluation/TRIALS.json',dict(trials=trials,selected=chosen,baseline_parent=str(parent.resolve())))
    save_json(out/'POSTPROCESS.json',dict(parent_assembly=str((parent/'ASSEMBLY.json').resolve()),parent_sha256=sha(parent/'ASSEMBLY.json'),residual_model=str(out.resolve()),model_sha256=sha(out/'fitted.joblib'),strength=chosen,lock_sha256=sha(d.package/'SPLIT_LOCK.json')))
    for f in frames:render(d.read(f,'validation','v42_overlay'),bestpred[f['image_id']],d.rois[f['camera_id']],out/'evaluation/overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | residual refinement {out.name} | development",score_label='skor gabungan, bukan probabilitas')
    save_json(out/'PIXEL_ACCESS.json',d.access)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1);a.out.mkdir(parents=True,exist_ok=False);source_snapshot(a.out,'all');d=PilotData(a.root)
    with threadpool_limits(limits=1):
        if train(d,a.runs,a.out):evaluate(d,a.runs,a.out)
