"""Aspect-preserving HOG features for rare vehicle crop classification."""
import argparse,json
from pathlib import Path
import cv2,joblib,numpy as np
from sklearn.metrics import classification_report
from threadpoolctl import threadpool_limits
from data_access import PilotData
from experiment import source_snapshot
from training import balanced_encounter_weights,fit_linear
from classifier_v9 import extra
from calibrate_v7 import main as calibrate
from vision import Features,CLASSES,rgb_read,sha,save_json

def letterbox(hog,gray,box,flip=False):
    x,y,u,v=box;w,h=u-x,v-y;scale=64/max(w,h);nw,nh=max(1,round(w*scale)),max(1,round(h*scale));tile=np.full((64,64),128,np.uint8);a,b=(64-nw)//2,(64-nh)//2;tile[b:b+nh,a:a+nw]=cv2.resize(gray[y:v,x:u],(nw,nh),interpolation=cv2.INTER_AREA)
    return hog.hog.compute(np.ascontiguousarray(tile[:,::-1] if flip else tile)).ravel()

def run(d,parent,base,out):
    out.mkdir(parents=True,exist_ok=False);source_snapshot(out,'all');cfg=json.loads(next((parent/'models').glob('*/MODEL.json')).read_text());assert sha(parent/'FEATURES.npz')==cfg['features_sha256'];assert sha(parent/'TRAIN_MANIFEST.json')==cfg['train_manifest_sha256'];trainrows=json.loads((parent/'TRAIN_MANIFEST.json').read_text());old=np.load(parent/'FEATURES.npz');valrows=json.loads((parent.parent/'pilot_20261117_v2/data/CROPS.json').read_text())['validation_positive'];arrays={};hog=Features();access=[]
    for split,rows in [('train',trainrows),('validation',valrows)]:
        boxes=np.empty((len(rows),1764),np.float32)
        for image_id in sorted({r['image_id'] for r in rows}):
            ids=[i for i,r in enumerate(rows) if r['image_id']==image_id];r=rows[ids[0]];assert r['split']==split
            if image_id in d.groups:assert d.groups[image_id]['split']==split
            else:assert split=='train' and r['independent_review_passed'] and r['identity_eligible']
            assert sha(d.root/r['source_image'])==r['source_sha256'];rgb=rgb_read(d.root/r['source_image']);g=hog.prepare(rgb);access.append(dict(image_id=image_id,split=split,source_sha256=r['source_sha256']))
            for i in ids:
                a=rows[i].get('augmentation','original');flip=a=='flip' or a.endswith('_flip');boxes[i]=letterbox(hog,g,rows[i]['bbox_xyxy'],flip)
        arrays[split]=dict(letterbox=np.c_[boxes,old[split][:,1764:]],combined=np.c_[old[split][:,:1764],boxes,old[split][:,1764:]])
    save_json(out/'PIXEL_ACCESS_fit.json',access);y=[r['class_name'] for r in trainrows];truth=[r['class_name'] for r in valrows];wt=balanced_encounter_weights(trainrows);trials=[];best=None
    for family in ['letterbox','combined']:
        for c in (.1,1.):
            m=fit_linear(arrays['train'][family],y,wt,c);report=classification_report(truth,m.predict(arrays['validation'][family]),labels=list(CLASSES),output_dict=True,zero_division=0);name=f'{family}_c{c}';p=out/'models'/name;p.mkdir(parents=True,exist_ok=False);joblib.dump(m,p/'fitted.joblib',compress=3);row=dict(name=name,family=family,C=c,macro_f1=report['macro avg']['f1-score'],macro_recall=report['macro avg']['recall'],report=report);trials.append(row)
            save_json(p/'MODEL.json',dict(family=family,C=c,model_sha256=sha(p/'fitted.joblib'),parent=str(parent.resolve()),train_manifest_sha256=cfg['train_manifest_sha256'],features_sha256=cfg['features_sha256'],supplement_sha256=cfg['supplement_sha256'],lock_sha256=sha(d.package/'SPLIT_LOCK.json')))
            if best is None or (row['macro_f1'],row['macro_recall'])>(best['macro_f1'],best['macro_recall']):best=row
            print(json.dumps(row),flush=True)
    save_json(out/'CLASSIFIER_TRIALS.json',dict(trials=trials,selected=best,test_evaluated=False))
    previous=json.loads((parent/'CLASSIFIER_TRIALS.json').read_text())['selected']['macro_f1']
    if best['macro_f1']<=previous:save_json(out/'STATUS.json',dict(status='stopped_crop_did_not_improve',previous=previous,test_evaluated=False));return
    p=out/'models'/best['name'];cfg=json.loads((p/'MODEL.json').read_text());assert sha(p/'fitted.joblib')==cfg['model_sha256'];m=joblib.load(p/'fitted.joblib');ev=out/'candidates/evaluation/raw';ev.mkdir(parents=True)
    for n,f in enumerate(d.selected('validation'),1):
        i=f['image_id'];src=base/'evaluation/raw'/(i+'.npz');rr=dict(np.load(src));rgb=d.read(f,'validation','v18_classifier');g=hog.prepare(rgb);labels=[]
        for off in range(0,len(rr['adjusted']),512):
            b=rr['adjusted'][off:off+512];lb=np.array([letterbox(hog,g,x) for x in b]);ex=np.array([extra(rgb,x) for x in b]);x=np.c_[lb,ex] if best['family']=='letterbox' else np.c_[hog.batch(g,b),lb,ex];labels.extend(CLASSES.index(c) for c in m.predict(x))
        rr['class_adjusted']=np.asarray(labels);np.savez_compressed(ev/(i+'.npz'),**rr);save_json(ev/(i+'.json'),dict(source_sha256=f['source_sha256'],parent_raw_sha256=sha(src),model_sha256=cfg['model_sha256'],output_sha256=sha(ev/(i+'.npz'))));print(json.dumps(dict(scored=n,image=i)),flush=True)
    save_json(out/'candidates/PARENT.json',dict(parent=str(base.resolve()),classifier=str(p.resolve()),lock_sha256=sha(d.package/'SPLIT_LOCK.json')));save_json(out/'candidates/evaluation/PIXEL_ACCESS.json',d.access);calibrate(d,out/'candidates',out/'calibrated')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    with threadpool_limits(limits=1):run(PilotData(a.root),a.parent,a.base,a.out)
