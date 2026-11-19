"""Versioned TRAIN overlay for explicit other/fragment rejection."""
import argparse,json
from pathlib import Path
import cv2,joblib,numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import classification_report
from data_access import PilotData
from experiment import source_snapshot
from classifier_v18 import letterbox
from classifier_v9 import extra
from training import balanced_encounter_weights,fit_linear
from calibrate_v7 import main as calibrate
from vision import Features,CLASSES,sha,save_json,rgb_read

def features(hog,rgb,rows):
    g=hog.prepare(rgb);xx=[]
    for r in rows:
        b=r['bbox_xyxy'];a=r.get('augmentation','original');flip=a=='flip' or a.endswith('_flip');x,y,u,v=b;tile=cv2.resize(g[y:v,x:u],(64,64),interpolation=cv2.INTER_AREA)
        standard=hog.hog.compute(np.ascontiguousarray(tile[:,::-1] if flip else tile)).ravel();xx.append(np.r_[standard,letterbox(hog,g,b,flip),extra(rgb,b)])
    return np.asarray(xx,np.float32)

def run(d,runs,out):
    out.mkdir(parents=True,exist_ok=False);source_snapshot(out,'all');parent=runs/'localization_20261117_v14';base=runs/'localization_20261117_v5';rows=json.loads((parent/'TRAIN_MANIFEST.json').read_text());original=len(rows);other=[];negative=[];mining=runs/'localization_20261117_v4/data/SAMPLES.json';mined=json.loads(mining.read_text());prep=json.loads((mining.parent/'PREPARATION.json').read_text());assert sha(mining)==prep['hashes']['SAMPLES.json']
    for f in d.selected('train'):
        group=d.groups[f['image_id']]
        for o in f['objects']:
            if o['class_name']=='other' and o['roi_membership']=='inside' and not o.get('ignore_reason'):
                for aug in ['original','flip']:
                    other.append(dict(image_id=f['image_id'],object_id=o['id'],source_image=f['source_image'],source_sha256=f['source_sha256'],split='train',bbox_xyxy=o['bbox_xyxy'],class_name='other',camera_id=f['camera_id'],encounter_id=group['component_id'],augmentation=aug,kind='known_other'))
        for is_fragment,maximum in [(True,24),(False,12)]:
            pool=[r for r in mined if r['image_id']==f['image_id'] and r['class_name']=='background' and (r['kind']=='known_fragment')==is_fragment]
            for r in pool[:maximum]:negative.append(dict(r,augmentation='original'))
    rows+=other+negative;save_json(out/'TRAIN_MANIFEST.json',rows);hog=Features();x=np.empty((len(rows),3588),np.float32);access=[]
    for image_id in sorted({r['image_id'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['image_id']==image_id];r=rows[ids[0]];assert r['split']=='train'
        if image_id in d.groups:assert d.groups[image_id]['split']=='train'
        else:assert r['identity_eligible'] and r['independent_review_passed']
        assert sha(d.root/r['source_image'])==r['source_sha256'];rgb=rgb_read(d.root/r['source_image']);x[ids]=features(hog,rgb,[rows[i] for i in ids]);access.append(dict(image_id=image_id,split='train',source_sha256=r['source_sha256']))
    np.savez_compressed(out/'TRAIN_FEATURES.npz',x=x);save_json(out/'PIXEL_ACCESS_fit.json',access);models={}
    for name,n in [('other',original+len(other)),('other_fragments',len(rows))]:
        m=fit_linear(x[:n],[r['class_name'] for r in rows[:n]],balanced_encounter_weights(rows[:n]),.1);p=out/'models'/name;p.mkdir(parents=True);joblib.dump(m,p/'fitted.joblib',compress=3);models[name]=m
        save_json(p/'MODEL.json',dict(model_sha256=sha(p/'fitted.joblib'),training_manifest_sha256=sha(out/'TRAIN_MANIFEST.json'),training_features_sha256=sha(out/'TRAIN_FEATURES.npz'),mined_manifest_sha256=sha(mining),parent_manifest_sha256=sha(parent/'TRAIN_MANIFEST.json'),rows=n,original_rows=original,other_rows=len(other),fragment_rows=n-original-len(other),C=.1,class_order=m.classes_.tolist(),lock_sha256=sha(d.package/'SPLIT_LOCK.json')));print(json.dumps(dict(fit=name,rows=n)),flush=True)
    valrows=json.loads((runs/'pilot_20261117_v2/data/CROPS.json').read_text())['validation_positive'];vx=[]
    for r in valrows:
        rgb=d.read(d.frames[r['image_id']],'validation','v21_crop_diagnostic');vx.append(features(hog,rgb,[r])[0])
    crop={n:classification_report([r['class_name'] for r in valrows],m.predict(vx),labels=list(CLASSES),output_dict=True,zero_division=0) for n,m in models.items()};save_json(out/'CROP_DIAGNOSTIC.json',crop);print(json.dumps({n:r['macro avg'] for n,r in crop.items()}),flush=True)
    for n,f in enumerate(d.selected('validation'),1):
        i=f['image_id'];src=base/'evaluation/raw'/(i+'.npz');rr=dict(np.load(src));rgb=d.read(f,'validation','v21_reject_classes');xx=features(hog,rgb,[dict(bbox_xyxy=b) for b in rr['adjusted']])
        for name,m in models.items():
            labels=m.predict(xx);out_raw=dict(rr,class_adjusted=np.array([CLASSES.index(c) if c in CLASSES else -1 for c in labels]));ev=out/name/'candidates/evaluation/raw';ev.mkdir(parents=True,exist_ok=True);np.savez_compressed(ev/(i+'.npz'),**out_raw);save_json(ev/(i+'.json'),dict(source_sha256=f['source_sha256'],parent_raw_sha256=sha(src),model_sha256=sha(out/'models'/name/'fitted.joblib'),output_sha256=sha(ev/(i+'.npz'))))
        print(json.dumps(dict(scored=n,image=i)),flush=True)
    for name in models:
        save_json(out/name/'candidates/PARENT.json',dict(parent=str(base.resolve()),classifier=str((out/'models'/name).resolve()),lock_sha256=sha(d.package/'SPLIT_LOCK.json')));calibrate(d,out/name/'candidates',out/name/'calibrated')
    save_json(out/'PIXEL_ACCESS.json',d.access)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    with threadpool_limits(limits=1):run(PilotData(a.root),a.runs,a.out)
