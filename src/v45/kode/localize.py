"""Train-only localization negatives and fixed-camera classical detection v3.

Frozen package/evaluator remain unchanged. This module uses the explicitly
versioned training overlay in ../Perbaikan_20261117/RENCANA.md.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from threadpoolctl import threadpool_limits

from data_access import PilotData
from experiment import add_f1, source_snapshot
from vision import CLASSES, Features, canonical, membership, nms, overlaps, render, roi_mask, save_json, sha, summarize, window_proposals

SEED=20261117
CONFIG=dict(protocol='localization_v2',positive_iou=.60,negative_iou=.20,
            positives_per_object=12,negatives_per_full_frame=256,pca=128,
            background='median_empty_train',feature_version=1)

def iou_matrix(boxes, targets):
    return np.stack([overlaps(np.asarray(t),boxes) for t in targets],axis=1) if len(targets) else np.zeros((len(boxes),0))

def allowed_negatives(frame, boxes, data):
    group=data.groups[frame['image_id']]
    if group['split']!='train' or group['sample_role']!='queue_or_control' or not frame['metrics']['count_reference_available']:
        return np.zeros(len(boxes),bool),np.zeros(len(boxes))
    mask=data.masks[frame['camera_id']];integ=np.pad(mask.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
    x,y,u,v=boxes.T;allowed=(integ[v,u]-integ[y,u]-integ[v,x]+integ[y,x])==((u-x)*(v-y))
    labels=data.adapter.adapt(frame,mask)
    # Outside/unknown/context objects can never be converted into negatives.
    blocked=[o['bbox_xyxy'] for o in frame['objects'] if not data.adapter.accepted(o) and not (o['class_name']=='other' and o['roi_membership']=='inside' and not o['ignore_reason'])]
    blocked += [r['bbox_xyxy'] for r in labels.neutral]
    for a,b,c,d in blocked: allowed &= ~((x<c+8)&(u>a-8)&(y<d+8)&(v>b-8))
    targets=[o['bbox_xyxy'] for o in frame['objects'] if data.adapter.accepted(o)]
    mx=iou_matrix(boxes,targets).max(axis=1,initial=0)
    return allowed&(mx<CONFIG['negative_iou']),mx

def backgrounds(data, out):
    images={};manifest={}
    for camera in data.rois:
        frames=[f for f in data.selected('train') if f['camera_id']==camera and f['metrics']['count_reference_available'] and sum(f['metrics']['counts_reference'].values())==0]
        if not frames:raise ValueError('No training background for '+camera)
        images[camera]=np.median(np.stack([data.read(f,'train','fit_background_median') for f in frames]),axis=0).astype(np.uint8)
        manifest[camera]=[dict(image_id=f['image_id'],source_image=f['source_image'],sha256=f['source_sha256'],split='train') for f in frames]
    np.savez_compressed(out/'backgrounds.npz',**images);save_json(out/'BACKGROUND_SOURCES.json',manifest)
    return images

class Extractor:
    """Train and inference share all transforms. No annotation access here."""
    def __init__(self,background):self.hog=Features();self.background=background;self.cameras=sorted(background)
    def prepare(self,rgb,camera):
        gray=self.hog.prepare(rgb);bg=self.background[camera]
        # Global exposure offset is robustly inferred from the input itself.
        residual=rgb.astype(np.float32)-bg.astype(np.float32)
        offset=np.median(residual[::8,::8],axis=(0,1))
        diff=np.max(np.abs(residual-offset),axis=2)/255
        lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)/255
        return gray,diff,lab
    def batch(self,prepared,boxes,camera):
        gray,diff,lab=prepared;h,w=gray.shape;extra=[]
        for x,y,u,v in boxes:
            patch=diff[y:v,x:u]
            e=np.r_[cv2.resize(patch,(8,8),interpolation=cv2.INTER_AREA).ravel(),
                cv2.resize((patch>.08).astype(np.float32),(8,8),interpolation=cv2.INTER_AREA).ravel(),
                cv2.resize(lab[y:v,x:u],(4,4),interpolation=cv2.INTER_AREA).ravel(),
                x/w,y/h,u/w,v/h,(u-x)/w,(v-y)/h,np.log((u-x)/(v-y)),
                [float(camera==c) for c in self.cameras]]
            extra.append(e)
        return np.c_[self.hog.batch(gray,boxes),np.asarray(extra,np.float32)].astype(np.float32)

def prepare(data,out):
    out.mkdir(parents=True,exist_ok=False);bg=backgrounds(data,out);extract=Extractor(bg)
    grid={c:window_proposals(r) for c,r in data.rois.items()};rows=[];all_x=[];all_y=[];deltas=[]
    for num,f in enumerate(data.selected('train'),1):
        camera=f['camera_id'];boxes=grid[camera];pos=data.positives(f,'train');selected=[];records=[];targets=[]
        for p in pos:
            b=np.asarray(p['bbox_xyxy']);ov=overlaps(b,boxes);ids=np.flatnonzero(ov>=CONFIG['positive_iou']);ids=ids[np.argsort(-ov[ids],kind='stable')[:CONFIG['positives_per_object']]]
            candidates=np.unique(np.concatenate([b[None,:],boxes[ids]]),axis=0)
            for c in candidates:
                selected.append(c);targets.append(b);records.append(dict(p,bbox_xyxy=c.tolist(),parent_bbox=b.tolist(),kind='positive',iou=float(overlaps(b,c[None,:])[0])))
        allowed,ov=allowed_negatives(f,boxes,data);ids=np.flatnonzero(allowed)
        rng=np.random.default_rng(int(canonical(f['image_id'])[:8],16))
        # Preserve both pure road and the localization negatives that old training excluded.
        chosen=[]
        for pool in (ids[ov[ids]>0],ids[ov[ids]==0]):
            k=min(len(pool),CONFIG['negatives_per_full_frame']//2)
            if k:chosen.extend(rng.choice(pool,k,replace=False).tolist())
        for i in chosen:
            b=boxes[i];selected.append(b);targets.append(b)
            records.append(dict(image_id=f['image_id'],object_id=None,camera_id=camera,source_image=f['source_image'],source_sha256=f['source_sha256'],split='train',
                encounter_id=data.groups[f['image_id']]['component_id'],bbox_xyxy=b.tolist(),class_name='background',kind='localization_negative' if ov[i]>0 else 'background',iou=float(ov[i])))
        if not selected:continue
        selected=np.asarray(selected,np.int32);targets=np.asarray(targets,np.float32)
        rgb=data.read(f,'train','localization_training_features');x=extract.batch(extract.prepare(rgb,camera),selected,camera)
        wh=selected[:,2:]-selected[:,:2];targetwh=targets[:,2:]-targets[:,:2]
        delta=np.c_[((targets[:,:2]+targets[:,2:])-(selected[:,:2]+selected[:,2:]))/(2*wh),np.log(targetwh/wh)]
        all_x.append(x);deltas.append(delta);rows.extend(records);all_y.extend([CLASSES.index(r['class_name'])+1 if r['kind']=='positive' else 0 for r in records])
        print(json.dumps(dict(prepare=num,image=f['image_id'],samples=len(x))),flush=True)
    x=np.concatenate(all_x);y=np.asarray(all_y);delta=np.concatenate(deltas)
    np.savez_compressed(out/'training.npz',x=x,y=y,delta=delta);save_json(out/'SAMPLES.json',rows)
    save_json(out/'PREPARATION.json',dict(config=CONFIG,lock_sha256=sha(data.package/'SPLIT_LOCK.json'),counts=dict(Counter(r['class_name'] for r in rows)),types=dict(Counter(r['kind'] for r in rows)),training_sha256=sha(out/'training.npz'),manifest_sha256=sha(out/'SAMPLES.json'),background_sha256=sha(out/'backgrounds.npz')))
    save_json(out/'PIXEL_ACCESS.json',data.access)

def weights(rows):
    groups=Counter((r['class_name'],r.get('object_id') or r['image_id']) for r in rows)
    classes=Counter(c for c,i in groups)
    # Background gets 40% mass; each foreground class 15%.
    return np.asarray([len(rows)*(.4 if r['class_name']=='background' else .15)/(classes[r['class_name']]*groups[r['class_name'],r.get('object_id') or r['image_id']]) for r in rows])

def transformed(x,pca,scaler):
    return np.c_[pca.transform(x[:,:1764]),scaler.transform(x[:,1764:])*.5].astype(np.float32)

def fit(data,out):
    prep=json.loads((out/'data/PREPARATION.json').read_text())
    for file,key in [('training.npz','training_sha256'),('SAMPLES.json','manifest_sha256'),('backgrounds.npz','background_sha256')]:
        if sha(out/'data'/file)!=prep[key]:raise ValueError('Training cache changed')
    rows=json.loads((out/'data/SAMPLES.json').read_text());z=np.load(out/'data/training.npz');x,y=z['x'],z['y'];wt=weights(rows)
    if any(r['split']!='train' or data.groups[r['image_id']]['split']!='train' for r in rows):raise ValueError('Nontrain fitting')
    pca=PCA(n_components=CONFIG['pca'],whiten=True,svd_solver='randomized',random_state=SEED).fit(x[:,:1764]);scaler=StandardScaler().fit(x[:,1764:]);xx=transformed(x,pca,scaler)
    reg=Ridge(alpha=100).fit(xx[y>0],z['delta'][y>0],sample_weight=wt[y>0])
    out.joinpath('models').mkdir(exist_ok=True)
    for name,model in [('linear',LinearSVC(C=.1,dual='auto',max_iter=30000,random_state=SEED)),('rbf',SVC(C=10,gamma='scale',decision_function_shape='ovr',cache_size=512)),('boost',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=10,early_stopping=False,random_state=SEED))]:
        start=perf_counter();model.fit(xx,y,sample_weight=wt);path=out/'models'/name;path.mkdir(exist_ok=False)
        joblib.dump(dict(pca=pca,scaler=scaler,reg=reg,estimator=model),path/'fitted.joblib',compress=3)
        save_json(path/'MODEL.json',dict(name=name,config=CONFIG,model_sha256=sha(path/'fitted.joblib'),training_sha256=prep['training_sha256'],background_sha256=prep['background_sha256'],class_order=model.classes_.tolist(),rois=data.rois,score_semantics='foreground vs background margin' if name!='boost' else 'foreground probability',fit_seconds=perf_counter()-start,pca_explained=float(pca.explained_variance_ratio_.sum()),lock_sha256=sha(data.package/'SPLIT_LOCK.json')))
        print(json.dumps(dict(fit=name,seconds=perf_counter()-start,samples=len(x))),flush=True)

def raw_scores(fitted,x,name):
    model=fitted['estimator'];xx=transformed(x,fitted['pca'],fitted['scaler'])
    if name=='boost':scores=model.predict_proba(xx)[:,1:]
    else:
        ds=model.decision_function(xx);scores=ds[:,1:]-ds[:,:1]
    delta=fitted['reg'].predict(xx)
    return scores.astype(np.float32),delta.astype(np.float32)

def adjust_boxes(boxes,delta,width,height):
    wh=boxes[:,2:]-boxes[:,:2];center=(boxes[:,:2]+boxes[:,2:])/2
    center=center+np.clip(delta[:,:2],-.5,.5)*wh;size=wh*np.exp(np.clip(delta[:,2:],-.5,.5))
    b=np.rint(np.c_[center-size/2,center+size/2]).astype(np.int32)
    b[:,[0,2]]=np.clip(b[:,[0,2]],0,width);b[:,[1,3]]=np.clip(b[:,[1,3]],0,height)
    b[:,2]=np.maximum(b[:,2],b[:,0]+1);b[:,3]=np.maximum(b[:,3],b[:,1]+1)
    return b

def predict(raw,rois,camera,threshold,nms_iou,regression):
    boxes=adjust_boxes(raw['boxes'],raw['delta'],rois[camera]['width'],rois[camera]['height']) if regression else raw['boxes']
    cls=np.argmax(raw['scores'],axis=1);score=raw['scores'].max(axis=1);ids=np.flatnonzero(score>=threshold)
    keep=nms(boxes[ids],score[ids],nms_iou);pred=[]
    for j,k in enumerate(ids[keep]):
        b=boxes[k].tolist();c=CLASSES[cls[k]];label,point,dist=membership(b,c,camera,rois,'bottom',{})
        pred.append(dict(id=f'p{j:04d}',bbox_xyxy=b,class_name=c,confidence=float(score[k]),roi_membership=label,contact_xy=point))
    return dict(status='ok',predictions=pred,summary=summarize(pred,rois[camera]))

def evaluate(data,out):
    target=out/'evaluation';target.mkdir(exist_ok=False);frames=data.selected('validation');grid={c:window_proposals(r) for c,r in data.rois.items()}
    background=dict(np.load(out/'data/backgrounds.npz'));extract=Extractor(background)
    models={};raw={name:{} for name in ('linear','rbf','boost')}
    for name in raw:
        cfg=json.loads((out/'models'/name/'MODEL.json').read_text());path=out/'models'/name/'fitted.joblib'
        if sha(path)!=cfg['model_sha256']:raise ValueError('Untrusted fitted artifact')
        models[name]=joblib.load(path) # Only locally generated and hash-verified artifacts.
        (target/'raw'/name).mkdir(parents=True)
    for n,f in enumerate(frames,1):
        start=perf_counter();camera=f['camera_id'];b=grid[camera];rgb=data.read(f,'validation','v3_full_image_prediction');prepared=extract.prepare(rgb,camera)
        accum={name:[[],[]] for name in models}
        for offset in range(0,len(b),1024):
            x=extract.batch(prepared,b[offset:offset+1024],camera)
            for name,model in models.items():
                s,d=raw_scores(model,x,name);accum[name][0].append(s);accum[name][1].append(d)
        for name in models:
            result=dict(boxes=b,scores=np.concatenate(accum[name][0]),delta=np.concatenate(accum[name][1]));raw[name][f['image_id']]=result
            np.savez_compressed(target/'raw'/name/(f['image_id']+'.npz'),**result)
        print(json.dumps(dict(scored=n,image=f['image_id'],seconds=perf_counter()-start)),flush=True)
        save_json(target/'PIXEL_ACCESS.json',data.access)
    trials=[];best=None;best_result=None;best_report=None
    for name in models:
        for threshold in ((.2,.4,.6,.8) if name=='boost' else (-1.,0.,.5,1.)):
            for nms_iou in (.3,.5):
                for regression in (False,True):
                    predictions={f['image_id']:predict(raw[name][f['image_id']],data.rois,f['camera_id'],threshold,nms_iou,regression) for f in frames}
                    report=add_f1(data.evaluator.evaluate_holdout(frames,data.masks,data.groups,predictions,subset='validation'));s=report['summary']
                    trial=dict(model=name,threshold=threshold,nms=nms_iou,regression=regression,macro_f1=s['macro_f1'],micro_f1=s['micro_f1'],mae=s['mae_full_only'],by_class=s['detection_by_class']);trials.append(trial)
                    if best is None or (trial['macro_f1'],-trial['mae']['total'])>(best['macro_f1'],-best['mae']['total']):best,best_result,best_report=trial,predictions,report
        print(json.dumps(dict(evaluated=name,best_macro_f1=best['macro_f1'],best_model=best['model'])),flush=True)
    save_json(target/'TRIALS.json',dict(trials=trials,selected=best));save_json(target/'VALIDATION_METRICS.json',best_report);save_json(target/'VALIDATION_PREDICTIONS.json',best_result)
    save_json(target/'GATES.json',dict(targets=dict(macro_f1=.85,class_f1=.75,count_mae=1.,coverage_mae=5.),passed=dict(macro_f1=best['macro_f1']>=.85,class_f1=all(v['f1']>=.75 for v in best['by_class'].values()),count_mae=best['mae']['total']<=1,coverage_mae=best['mae']['coverage_pp']<=5),test_evaluated=False))
    for f in frames:
        rgb=data.read(f,'validation','v3_prediction_overlay');render(rgb,best_result[f['image_id']],data.rois[f['camera_id']],target/'overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | v3 {best['model']} | validation")
    save_json(target/'PIXEL_ACCESS.json',data.access)
    print(json.dumps(best),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['prepare','fit','evaluate','all'],default='all');args=ap.parse_args()
    cv2.setNumThreads(1);data=PilotData(args.root);args.out.mkdir(parents=True,exist_ok=True);source_snapshot(args.out,args.phase)
    with threadpool_limits(limits=1):
        if args.phase in ('prepare','all'):prepare(data,args.out/'data')
        if args.phase in ('fit','all'):fit(data,args.out)
        if args.phase in ('evaluate','all'):evaluate(data,args.out)
    save_json(args.out/f'ACCESS_{args.phase}.json',data.access)

if __name__=='__main__':main()
