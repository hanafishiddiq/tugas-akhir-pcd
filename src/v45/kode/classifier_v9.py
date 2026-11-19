"""RBF crop classifier, full-image validation uses unchanged v5 candidate boxes."""
import argparse,json
from pathlib import Path
from time import perf_counter
import cv2,joblib,numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from data_access import PilotData
from experiment import source_snapshot
from training import balanced_encounter_weights
from calibrate_v7 import main as calibrate
from vision import Features,CLASSES,save_json,sha

def extra(rgb,box):
    x,y,u,v=box;tile=rgb[y:v,x:u]
    hist=np.concatenate([cv2.calcHist([tile],[c],None,[16],[0,256]).ravel().astype(np.float64)/((u-x)*(v-y)) for c in range(3)])
    gray=cv2.resize(cv2.cvtColor(tile,cv2.COLOR_RGB2GRAY),(64,64),interpolation=cv2.INTER_AREA);center=gray[1:-1,1:-1];bits=np.stack([gray[dy:dy+62,dx:dx+62]>=center for dx,dy in [(0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,1)]])
    transitions=np.count_nonzero(bits!=np.roll(bits,1,axis=0),axis=0);bins=np.where(transitions<=2,bits.sum(0),9);lbp=np.bincount(bins.ravel(),minlength=10)/bins.size;ratio=np.log((u-x)/(v-y))
    return np.r_[np.sqrt(hist),np.sqrt(lbp),ratio,ratio*ratio/3].astype(np.float32)

def fit(data,crops,out):
    out.mkdir(parents=True,exist_ok=True);source_snapshot(out,'fit');prep=json.loads((crops/'data/PREPARATION.json').read_text())
    if sha(crops/'data/features.npz')!=prep['feature_cache_sha256'] or sha(crops/'data/CROPS.json')!=prep['crop_manifest_sha256']:raise ValueError('Parent classifier cache changed')
    rows=json.loads((crops/'data/CROPS.json').read_text());z=np.load(crops/'data/features.npz');arrays={};frames={f['image_id']:f for f in [*data.frames.values(),*data.aux]}
    for split in ['train','validation']:
        records=rows[split+'_positive'];ex=np.empty((len(records),60),np.float32)
        for i in sorted({r['image_id'] for r in records}):
            f=frames[i];rgb=data.read(f,split,'v9_classification_extra_features');ids=[n for n,r in enumerate(records) if r['image_id']==i]
            for n in ids:ex[n]=extra(rgb,records[n]['bbox_xyxy'])
        arrays[split]=np.c_[z[split+'_positive'],ex]
    training_rows=rows['train_positive']
    if any(r['split']!='train' or data.groups[r['image_id']]['split']!='train' for r in training_rows):raise ValueError('Nontrain fitting')
    y=np.array([r['class_name'] for r in training_rows]);truth=[r['class_name'] for r in rows['validation_positive']];wt=balanced_encounter_weights(training_rows);trials=[];best=None
    for features,dim in [('hog',1764),('hog_color_texture_shape',1824)]:
        for c in (1.,10.,100.):
            t=perf_counter();model=SVC(C=c,gamma='scale',kernel='rbf',cache_size=512,random_state=20261117).fit(arrays['train'][:,:dim],y,sample_weight=wt)
            pred=model.predict(arrays['validation'][:,:dim]);report=classification_report(truth,pred,labels=list(CLASSES),output_dict=True,zero_division=0);name=features+'_'+str(int(c));p=out/'models'/name;p.mkdir(parents=True,exist_ok=False);joblib.dump(model,p/'fitted.joblib',compress=3)
            cfg=dict(features=features,dimension=dim,C=c,model_sha256=sha(p/'fitted.joblib'),source_crops=str(crops.resolve()),parent_features_sha256=prep['feature_cache_sha256'],parent_manifest_sha256=prep['crop_manifest_sha256'],lock_sha256=sha(data.package/'SPLIT_LOCK.json'),fit_and_eval_seconds=perf_counter()-t,validation_report=report)
            save_json(p/'MODEL.json',cfg);trial=dict(name=name,macro_f1=report['macro avg']['f1-score'],macro_recall=report['macro avg']['recall'],report=report,seconds=perf_counter()-t);trials.append(trial)
            if best is None or (trial['macro_f1'],trial['macro_recall'])>(best['macro_f1'],best['macro_recall']):best=trial
            print(json.dumps(dict(name=name,macro_f1=trial['macro_f1'],truck=report['truk'],seconds=perf_counter()-t)),flush=True)
    save_json(out/'CLASSIFIER_TRIALS.json',dict(trials=trials,selected=best,test_evaluated=False));save_json(out/'PIXEL_ACCESS_fit.json',data.access)

def score(data,base,out):
    source_snapshot(out,'score');name=json.loads((out/'CLASSIFIER_TRIALS.json').read_text())['selected']['name'];p=out/'models'/name;cfg=json.loads((p/'MODEL.json').read_text());assert sha(p/'fitted.joblib')==cfg['model_sha256'];model=joblib.load(p/'fitted.joblib');features=Features();ev=out/'candidates/evaluation/raw';ev.mkdir(parents=True,exist_ok=False)
    for n,f in enumerate(data.selected('validation'),1):
        start=perf_counter();i=f['image_id'];source=base/'evaluation/raw'/(i+'.npz');rr=dict(np.load(source));b=rr['adjusted'];rgb=data.read(f,'validation','v9_candidate_classification');gray=features.prepare(rgb);labels=[]
        for off in range(0,len(b),512):
            boxes=b[off:off+512];x=features.batch(gray,boxes)
            if cfg['dimension']>1764:x=np.c_[x,np.array([extra(rgb,box) for box in boxes])]
            labels.extend(CLASSES.index(c) for c in model.predict(x))
        rr['class_adjusted']=np.array(labels);np.savez_compressed(ev/(i+'.npz'),**rr)
        save_json(ev/(i+'.json'),dict(image_sha256=f['source_sha256'],parent_raw_sha256=sha(source),classifier_sha256=cfg['model_sha256'],output_sha256=sha(ev/(i+'.npz'))))
        print(json.dumps(dict(scored=n,image=i,candidates=len(b),seconds=perf_counter()-start)),flush=True)
    save_json(out/'candidates/PARENT.json',dict(parent=str(base.resolve()),classifier=str(p.resolve()),lock_sha256=sha(data.package/'SPLIT_LOCK.json')));save_json(out/'candidates/evaluation/PIXEL_ACCESS.json',data.access)
    calibrate(data,out/'candidates',out/'calibrated');save_json(out/'PIXEL_ACCESS_score.json',data.access)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--crops',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['fit','score','all'],default='all');a=ap.parse_args();d=PilotData(a.root);cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        if a.phase in ('fit','all'):fit(d,a.crops,a.out)
        if a.phase in ('score','all'):score(d,a.base,a.out)
