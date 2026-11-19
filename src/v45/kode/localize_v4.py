"""Hard localization mining confined to known TRAIN boxes, protocol overlay v3."""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC,SVC
from threadpoolctl import threadpool_limits
from data_access import PilotData
from experiment import add_f1,source_snapshot
from localize import Extractor,adjust_boxes,iou_matrix,transformed,weights
from vision import CLASSES,canonical,membership,nms,overlaps,render,save_json,sha,summarize,window_proposals

SEED=20261117

def eligible(frame,boxes,data):
    if data.groups[frame['image_id']]['split']!='train':raise ValueError('Mining requires train')
    targets=[o['bbox_xyxy'] for o in frame['objects'] if data.adapter.accepted(o)]
    x,y,u,v=boxes.T;inside_known=np.zeros(len(boxes),bool)
    for a,b,c,d in targets:inside_known|=(x>=a)&(y>=b)&(u<=c)&(v<=d)
    inside_scope=np.zeros(len(boxes),bool)
    if frame['metrics']['count_reference_available'] and data.groups[frame['image_id']]['sample_role']=='queue_or_control':
        m=data.masks[frame['camera_id']];i=np.pad(m.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
        inside_scope=(i[v,u]-i[y,u]-i[v,x]+i[y,x])==((u-x)*(v-y))
    ok=inside_known|inside_scope
    for o in frame['objects']:
        if data.adapter.accepted(o) or (o['class_name']=='other' and o['roi_membership']=='inside' and not o['ignore_reason']):continue
        a,b,c,d=o['bbox_xyxy'];ok&=~((x<c+8)&(u>a-8)&(y<d+8)&(v>b-8))
    for zone in data.adapter.adapt(frame,data.masks[frame['camera_id']]).neutral:
        a,b,c,d=zone['bbox_xyxy'];ok&=~((x<c+8)&(u>a-8)&(y<d+8)&(v>b-8))
    mx=iou_matrix(boxes,targets).max(axis=1,initial=0)
    return ok&(mx<.35),mx,inside_known

def model_load(path):
    config=json.loads((path/'MODEL.json').read_text())
    if sha(path/'fitted.joblib')!=config['model_sha256']:raise ValueError('Artifact hash mismatch')
    return joblib.load(path/'fitted.joblib')

def prepare(data,base,out):
    out.mkdir(parents=True,exist_ok=False)
    baseline=model_load(base/'models/linear');bg=dict(np.load(base/'data/backgrounds.npz'));ext=Extractor(bg)
    z=np.load(base/'data/training.npz');xx=[transformed(z['x'],baseline['pca'],baseline['scaler'])];yy=[z['y']];dd=[z['delta']]
    rows=json.loads((base/'data/SAMPLES.json').read_text());existing={(r['image_id'],tuple(r['bbox_xyxy'])) for r in rows};added=[]
    grid={c:window_proposals(r) for c,r in data.rois.items()}
    for n,f in enumerate(data.selected('train'),1):
        camera=f['camera_id'];boxes=grid[camera];ok,mx,known=eligible(f,boxes,data)
        ids=[i for i in np.flatnonzero(ok) if (f['image_id'],tuple(boxes[i])) not in existing]
        if not ids:continue
        ids=np.asarray(ids);b=boxes[ids];rgb=data.read(f,'train','mine_known_box_fragments');prep=ext.prepare(rgb,camera)
        feats=[];margins=[]
        for off in range(0,len(b),1024):
            x=transformed(ext.batch(prep,b[off:off+1024],camera),baseline['pca'],baseline['scaler']);scores=baseline['estimator'].decision_function(x)
            feats.append(x);margins.append((scores[:,1:]-scores[:,:1]).max(1))
        x=np.concatenate(feats);scores=np.concatenate(margins);order=np.argsort(-scores,kind='stable')[:192]
        xx.append(x[order]);yy.append(np.zeros(len(order),int));dd.append(np.zeros((len(order),4)))
        for j in order:
            i=int(ids[j]);record=dict(image_id=f['image_id'],object_id=None,camera_id=camera,source_image=f['source_image'],source_sha256=f['source_sha256'],split='train',encounter_id=data.groups[f['image_id']]['component_id'],bbox_xyxy=boxes[i].tolist(),class_name='background',kind='known_fragment' if known[i] else 'mined_full_scope',iou=float(mx[i]),mining_score=float(scores[j]))
            rows.append(record);added.append(record)
        print(json.dumps(dict(mined=n,image=f['image_id'],eligible=len(ids),added=len(order))),flush=True)
    np.savez_compressed(out/'training_projected.npz',x=np.concatenate(xx),y=np.concatenate(yy),delta=np.concatenate(dd))
    joblib.dump(dict(pca=baseline['pca'],scaler=baseline['scaler']),out/'transforms.joblib',compress=3)
    np.savez_compressed(out/'backgrounds.npz',**bg)
    save_json(out/'SAMPLES.json',rows);save_json(out/'MINED.json',added);save_json(out/'PIXEL_ACCESS.json',data.access)
    save_json(out/'PREPARATION.json',dict(protocol='localization_v3',parent=str(base.resolve()),counts=dict(Counter(r['class_name'] for r in rows)),added=dict(Counter(r['kind'] for r in added)),pca_fit_source='v3 TRAIN only, frozen transform reused',lock_sha256=sha(data.package/'SPLIT_LOCK.json'),hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}))

def geom_features(boxes,w,h):
    x=(boxes[:,0]+boxes[:,2])/(2*w);y=boxes[:,3]/h
    return np.c_[x,y,x*y,x*x,y*y]

def geometry_fit(data):
    records=[]
    for f in data.selected('train'):
        for p in data.positives(f,'train'):records.append(dict(camera=f['camera_id'],cls=p['class_name'],box=p['bbox_xyxy'],w=f['width'],h=f['height']))
    out={}
    for cam in ['*',*data.rois]:
        for cls in CLASSES:
            rr=[r for r in records if r['cls']==cls and (cam=='*' or r['camera']==cam)]
            if not rr:continue
            x=np.concatenate([geom_features(np.array([r['box']]),r['w'],r['h']) for r in rr]);y=np.array([np.log((np.array(r['box'][2:])-r['box'][:2])/np.array([r['w'],r['h']])) for r in rr])
            model=Ridge(alpha=1.).fit(x,y);res=y-model.predict(x);sigma=np.maximum(np.std(res,axis=0),.3 if len(rr)>10 else .6)
            out[cam+':'+cls]=dict(model=model,sigma=sigma,n=len(rr))
    return out

def geometry_score(boxes,camera,rois,geometry):
    r=rois[camera];x=geom_features(boxes,r['width'],r['height']);y=np.log((boxes[:,2:]-boxes[:,:2])/np.array([r['width'],r['height']]))
    scores=[]
    for cls in CLASSES:
        g=geometry.get(camera+':'+cls,geometry['*:'+cls]);res=(y-g['model'].predict(x))/g['sigma'];scores.append(np.exp(-.5*np.sum(res*res,axis=1)))
    return np.asarray(scores).T

def fit(data,out):
    path=out/'data';prep=json.loads((path/'PREPARATION.json').read_text())
    for name,digest in prep['hashes'].items():
        if sha(path/name)!=digest:raise ValueError('Changed training file '+name)
    z=np.load(path/'training_projected.npz');x,y=z['x'],z['y'];rows=json.loads((path/'SAMPLES.json').read_text())
    if any(data.groups[r['image_id']]['split']!='train' for r in rows):raise ValueError('Nontrain fit')
    wt=weights(rows);transform=joblib.load(path/'transforms.joblib');reg=Ridge(alpha=100).fit(x[y>0],z['delta'][y>0],sample_weight=wt[y>0]);geometry=geometry_fit(data)
    models=[('linear',LinearSVC(C=.1,dual='auto',max_iter=30000,random_state=SEED)),('boost',HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,l2_regularization=10,early_stopping=False,random_state=SEED))]
    for name,model in models:
        t=perf_counter();model.fit(x,y,sample_weight=wt);p=out/'models'/name;p.mkdir(parents=True,exist_ok=False)
        joblib.dump(dict(**transform,reg=reg,geometry=geometry,estimator=model),p/'fitted.joblib',compress=3)
        save_json(p/'MODEL.json',dict(name=name,protocol='localization_v3',model_sha256=sha(p/'fitted.joblib'),training_sha256=sha(path/'training_projected.npz'),background_sha256=sha(path/'backgrounds.npz'),lock_sha256=sha(data.package/'SPLIT_LOCK.json'),seconds=perf_counter()-t,rois=data.rois))
        print(json.dumps(dict(fit=name,seconds=perf_counter()-t)),flush=True)

def postprocess(raw,camera,rois,threshold,nms_iou,regression,prior):
    boxes=raw['adjusted'] if regression else raw['boxes'];s=raw['scores'].copy()
    if prior:s=s+np.log(np.maximum(raw['geometry_adjusted'] if regression else raw['geometry'],1e-8))*.5
    cls=s.argmax(1);score=s.max(1);ids=np.flatnonzero(score>=threshold);ids=ids[nms(boxes[ids],score[ids],nms_iou)]
    pred=[]
    for j,i in enumerate(ids):
        b=boxes[i].tolist();c=CLASSES[cls[i]];label,point,dist=membership(b,c,camera,rois,'bottom',{})
        pred.append(dict(id=f'p{j:04d}',bbox_xyxy=b,class_name=c,confidence=float(score[i]),roi_membership=label,contact_xy=point))
    return dict(status='ok',predictions=pred,summary=summarize(pred,rois[camera]))

def evaluate(data,out):
    ev=out/'evaluation';ev.mkdir(exist_ok=False);frames=data.selected('validation');extract=Extractor(dict(np.load(out/'data/backgrounds.npz')))
    models={n:model_load(out/'models'/n) for n in ('linear','boost')};grid={c:window_proposals(r) for c,r in data.rois.items()};raw={n:{} for n in models}
    (ev/'projected').mkdir()
    for n in models:(ev/'raw'/n).mkdir(parents=True)
    for n,f in enumerate(frames,1):
        start=perf_counter();c=f['camera_id'];boxes=grid[c];rgb=data.read(f,'validation','v4_full_image_prediction');prep=extract.prepare(rgb,c);first=models['linear']
        x=np.concatenate([transformed(extract.batch(prep,boxes[off:off+1024],c),first['pca'],first['scaler']) for off in range(0,len(boxes),1024)])
        np.savez_compressed(ev/'projected'/(f['image_id']+'.npz'),x=x)
        delta=first['reg'].predict(x);adj=adjust_boxes(boxes,delta,f['width'],f['height']);geo=geometry_score(boxes,c,data.rois,first['geometry']);geoa=geometry_score(adj,c,data.rois,first['geometry'])
        for name,m in models.items():
            if name=='boost':p=m['estimator'].predict_proba(x);s=np.log(np.maximum(p[:,1:],1e-9))-np.log(np.maximum(p[:,:1],1e-9))
            else:ds=m['estimator'].decision_function(x);s=ds[:,1:]-ds[:,:1]
            rr=dict(boxes=boxes,adjusted=adj,scores=s,geometry=geo,geometry_adjusted=geoa);raw[name][f['image_id']]=rr
            np.savez_compressed(ev/'raw'/name/(f['image_id']+'.npz'),**rr)
        print(json.dumps(dict(scored=n,image=f['image_id'],seconds=perf_counter()-start)),flush=True)
        save_json(ev/'PIXEL_ACCESS.json',data.access)
    evaluate_cached(data,out,raw)

def evaluate_cached(data,out,raw=None):
    ev=out/'evaluation';frames=data.selected('validation')
    if raw is None:raw={p.name:{f['image_id']:dict(np.load(p/(f['image_id']+'.npz'))) for f in frames} for p in (ev/'raw').iterdir() if p.is_dir()}
    trials=[];best=None
    for name in raw:
        for threshold in (-1.,0.,.5,1.,2.):
            for nms_iou in (.3,.5):
                for regression in (False,True):
                    for prior in (False,True):
                        predictions={f['image_id']:postprocess(raw[name][f['image_id']],f['camera_id'],data.rois,threshold,nms_iou,regression,prior) for f in frames}
                        report=add_f1(data.evaluator.evaluate_holdout(frames,data.masks,data.groups,predictions,subset='validation'));s=report['summary']
                        trial=dict(model=name,threshold=threshold,nms=nms_iou,regression=regression,prior=prior,macro_f1=s['macro_f1'],micro_f1=s['micro_f1'],mae=s['mae_full_only'],by_class=s['detection_by_class']);trials.append(trial)
                        if best is None or (trial['macro_f1'],-trial['mae']['total'])>(best['macro_f1'],-best['mae']['total']):best,best_result,best_report=trial,predictions,report
        print(json.dumps(dict(evaluated=name,best_f1=best['macro_f1'],best_model=best['model'])),flush=True)
    save_json(ev/'TRIALS.json',dict(trials=trials,selected=best));best_report.update(test_evaluated=False,selection_split='validation')
    save_json(ev/'VALIDATION_METRICS.json',best_report);save_json(ev/'VALIDATION_PREDICTIONS.json',best_result)
    for f in frames:render(data.read(f,'validation','v4_prediction_overlay'),best_result[f['image_id']],data.rois[f['camera_id']],ev/'overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | v4 {best['model']} | validation")
    save_json(ev/'PIXEL_ACCESS.json',data.access);print(json.dumps(best),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',default='all',choices=['prepare','fit','evaluate','cached','all']);a=ap.parse_args()
    cv2.setNumThreads(1);data=PilotData(a.root);a.out.mkdir(parents=True,exist_ok=True);source_snapshot(a.out,a.phase)
    with threadpool_limits(limits=1):
        if a.phase in ('all','prepare'):prepare(data,a.base,a.out/'data')
        if a.phase in ('all','fit'):fit(data,a.out)
        if a.phase in ('all','evaluate'):evaluate(data,a.out)
        if a.phase=='cached':evaluate_cached(data,a.out)
    save_json(a.out/f'ACCESS_{a.phase}.json',data.access)

if __name__=='__main__':main()
