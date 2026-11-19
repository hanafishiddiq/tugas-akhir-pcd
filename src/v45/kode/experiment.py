"""Reproducible train/validation pilot. Test prediction is intentionally unavailable."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from data_access import PilotData,prepare
from inference import Model
from training import train,SEED
from vision import CLASSES,canonical,overlaps,render,save_json,sha,window_proposals

HERE=Path(__file__).resolve().parent

def source_snapshot(output,phase):
    target=output/'source'/phase;target.mkdir(parents=True,exist_ok=True)
    files={}
    sources=[*HERE.glob('*.py'),HERE/'pyproject.toml',HERE/'uv.lock']
    files={s.name:sha(s) for s in sources if s.is_file()}
    receipt=target/'CHECKSUMS.json'
    if receipt.exists() and json.loads(receipt.read_text())!=files:
        raise ValueError('Code changed after this phase started; use a new experiment directory')
    for source in sources:
        if source.is_file():shutil.copy2(source,target/source.name);files[source.name]=sha(source)
    save_json(target/'CHECKSUMS.json',files)
    return files

def add_f1(report):
    if not report['evaluation_run_valid']:raise ValueError(report['errors'])
    for s in [report['summary'],*report['by_camera'].values(),*report['by_reference_scope'].values()]:
        tp=fp=fn=0
        for row in s['detection_by_class'].values():
            a,b,c=row['tp'],row['fp'],row['fn'];row['f1']=2*a/(2*a+b+c) if 2*a+b+c else 0.;tp+=a;fp+=b;fn+=c
        s['macro_f1']=float(np.mean([r['f1'] for r in s['detection_by_class'].values()]))
        s['micro_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return report

def candidate_recall(data,grid):
    rows=[];by_camera={c:window_proposals(r,grid) for c,r in data.rois.items()}
    for frame in data.selected('validation'):
        boxes=by_camera[frame['camera_id']]
        for obj in frame['objects']:
            if not data.adapter.accepted(obj):continue
            best=float(overlaps(np.asarray(obj['bbox_xyxy']),boxes).max(initial=0.))
            rows.append(dict(image_id=frame['image_id'],object_id=obj['id'],class_name=obj['class_name'],camera_id=frame['camera_id'],best_iou=best,covered=best>=.5))
    def summary(items):
        return {c:dict(objects=sum(x['class_name']==c for x in items),covered=sum(x['class_name']==c and x['covered'] for x in items),
                       recall=sum(x['class_name']==c and x['covered'] for x in items)/sum(x['class_name']==c for x in items) if any(x['class_name']==c for x in items) else None) for c in CLASSES}
    return dict(by_class=summary(rows),by_camera={c:summary([r for r in rows if r['camera_id']==c]) for c in data.rois},objects=rows,
                grid_counts={c:len(b) for c,b in by_camera.items()},uses_gt_to_generate_proposals=False)

def score_validation(data,model,directory):
    directory.mkdir(parents=True,exist_ok=True);output={}
    for n,frame in enumerate(data.selected('validation'),1):
        image_id=frame['image_id'];file=directory/(image_id+'.npz');meta=directory/(image_id+'.json')
        key=canonical(dict(weights=model.config['weights_sha256'],features=model.config['features'],grid=model.config['grid'],
            image=frame['source_sha256'],vision=sha(HERE/'vision.py'),inference=sha(HERE/'inference.py')))
        if file.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if record['cache_key']!=key or sha(file)!=record['npz_sha256']:raise ValueError('Stale candidate cache '+image_id)
            with np.load(file,allow_pickle=False) as z:scored={k:z[k].copy() for k in z.files}
            scored.update({k:record[k] for k in ('seconds','candidate_count','minimum_stored_score')})
        else:
            rgb=data.read(frame,'validation','full_image_prediction_without_annotations')
            scored=model.score(rgb,frame['camera_id'],minimum_score=-1.)
            np.savez_compressed(file,**{k:scored[k] for k in ('boxes','detector_scores','class_scores')})
            save_json(meta,dict(cache_key=key,npz_sha256=sha(file),**{k:scored[k] for k in ('seconds','candidate_count','minimum_stored_score')}))
            save_json(directory/'PIXEL_ACCESS.json',data.access)
        output[image_id]=scored
        print(json.dumps({'model':model.config['tag'],'validation_done':n,'validation_total':61,'image':image_id,'candidates':scored['candidate_count'],'seconds':round(scored['seconds'],3)}),flush=True)
    return output

def bootstrap_components(report,data,repeats=1000):
    component=defaultdict(lambda:np.zeros((4,3),float))
    for item in report['per_image']:
        frame=data.frames[item['image_id']];group=data.groups[item['image_id']]['component_id'];counts=component[group]
        for det in item['detections']:
            if det['outcome'] in ('tp','fp'):counts[CLASSES.index(det['class_name']),0 if det['outcome']=='tp' else 1]+=1
        objects={o['id']:o for o in frame['objects']}
        for oid in item['false_negative_object_ids']:counts[CLASSES.index(objects[oid]['class_name']),2]+=1
    matrix=np.stack(list(component.values()));rng=np.random.default_rng(SEED);values=[]
    for _ in range(repeats):
        tp,fp,fn=matrix[rng.integers(0,len(matrix),len(matrix))].sum(0).T
        values.append(np.divide(2*tp,2*tp+fp+fn,out=np.zeros(4),where=(2*tp+fp+fn)>0))
    values=np.asarray(values)
    return {'unit':'locked component; diagnostic resampling of validation only','components':len(matrix),'repeats':repeats,
            'macro_f1_percentile_95':np.percentile(values.mean(1),[2.5,97.5]).tolist(),
            'class_f1_percentile_95':{c:np.percentile(values[:,i],[2.5,97.5]).tolist() for i,c in enumerate(CLASSES)},
            'limitation':'Small rare-class support and parameter selection on this validation set limit inference; not an independent test confidence interval.'}

def evaluate(data,output,model_sources=None,thresholds=(-1.,-.5,0.,.5,1.,2.)):
    directory=output/'evaluation';directory.mkdir(exist_ok=True);trials=[];all_results={};raw_by_model={}
    frames=data.selected('validation')
    baseline={f['image_id']:{'status':'ok','predictions':[]} for f in frames}
    null=add_f1(data.evaluator.evaluate_holdout(frames,data.masks,data.groups,baseline,subset='validation'))
    save_json(directory/'EMPTY_DETECTOR_BASELINE.json',null)
    model_sources=model_sources or [(n,output/'models'/n,directory/'raw_candidates'/n) for n in ('initial','hard_negative')]
    model_paths={n:path for n,path,cache in model_sources};complexity={n:i for i,(n,path,cache) in enumerate(model_sources)}
    first=Model(model_sources[0][1]);save_json(directory/'CANDIDATE_RECALL.json',candidate_recall(data,first.config['grid']))
    for name,model_path,cache_directory in model_sources:
        model=Model(model_path);raw=score_validation(data,model,cache_directory);raw_by_model[name]=raw
        for threshold in thresholds:
            for nms_iou in (.3,.5):
                results={f['image_id']:model.postprocess(raw[f['image_id']],f['camera_id'],threshold=threshold,nms_iou=nms_iou) for f in frames}
                measured=add_f1(data.evaluator.evaluate_holdout(frames,data.masks,data.groups,results,subset='validation'))
                trial=dict(model=name,threshold=threshold,nms_iou=nms_iou,roi_rule=model.config['selected']['roi_rule'],
                           macro_f1=measured['summary']['macro_f1'],micro_f1=measured['summary']['micro_f1'],
                           mae=measured['summary']['mae_full_only'],by_class=measured['summary']['detection_by_class'])
                trials.append(trial);key=(name,threshold,nms_iou);all_results[key]=(measured,results)
                print(json.dumps({'validation_trial':len(trials),'model':name,'threshold':threshold,'nms':nms_iou,'macro_f1':trial['macro_f1'],'count_mae':trial['mae']['total']}),flush=True)
    best=max(trials,key=lambda t:(t['macro_f1'],-t['mae']['total'],-complexity[t['model']],-t['threshold']))
    key=(best['model'],best['threshold'],best['nms_iou']);measured,results=all_results[key]
    save_json(directory/'THRESHOLD_TRIALS.json',{'selection':'validation macro F1; tie lower count MAE, simpler model, lower threshold','trials':trials,'selected':best})
    selected=output/'selected_model'
    if selected.exists():raise FileExistsError(selected)
    shutil.copytree(model_paths[best['model']],selected)
    cfg=json.loads((selected/'MODEL.json').read_text());cfg['selected'].update(detector_threshold=best['threshold'],nms_iou=best['nms_iou']);cfg['model_id']=canonical({k:v for k,v in cfg.items() if k!='model_id'})[:20];save_json(selected/'MODEL.json',cfg)
    for result in results.values():result['model_id']=cfg['model_id']
    measured['selection_split']='validation';measured['test_evaluated']=False
    measured['timing']={'scope':'native decoding and model load excluded; feature extraction plus scoring included; single native call','median_seconds':float(np.median([r['inference_seconds'] for r in results.values()])),
                        'p95_seconds':float(np.percentile([r['inference_seconds'] for r in results.values()],95))}
    measured['bootstrap']=bootstrap_components(measured,data)
    measured['minimum_utility_checks']={'tp_in_each_class':all(x['tp']>0 for x in measured['summary']['detection_by_class'].values()),
        'count_mae_better_than_empty':measured['summary']['mae_full_only']['total']<null['summary']['mae_full_only']['total'],
        'coverage_mae_better_than_empty':measured['summary']['mae_full_only']['coverage_pp']<null['summary']['mae_full_only']['coverage_pp']}
    measured['minimum_utility_passed']=all(measured['minimum_utility_checks'].values())
    save_json(directory/'VALIDATION_METRICS.json',measured);save_json(directory/'VALIDATION_PREDICTIONS.json',results)
    examples=[]
    for frame in frames:
        image_id=frame['image_id'];rgb=data.read(frame,'validation','render_final_model_prediction');result=results[image_id]
        path=directory/'overlays'/f'{image_id}.jpg';render(rgb,result,data.rois[frame['camera_id']],path,title=f'{image_id} | {frame["camera_id"]} | VALIDATION prediction')
        detail=next(x for x in measured['per_image'] if x['image_id']==image_id)
        examples.append(dict(image_id=image_id,camera=frame['camera_id'],full_count=detail['full_count_frame'],
                             fp=sum(x['outcome']=='fp' for x in detail['detections']),fn=len(detail['false_negative_object_ids']),
                             counts=result['summary']['count_by_class'],count_errors=detail['count_absolute_errors']))
    save_json(directory/'ERROR_ANALYSIS.json',dict(images=examples,fp_reasons=dict(Counter(x.get('reason','matched') for row in measured['per_image'] for x in row['detections'] if x['outcome']=='fp')),
        most_errors=sorted(examples,key=lambda x:x['fp']+x['fn'],reverse=True)[:12]))
    cards=''.join(f'<article><h2>{html.escape(e["image_id"])} | {html.escape(e["camera"])}</h2><a href="overlays/{e["image_id"]}.jpg"><img loading="lazy" src="overlays/{e["image_id"]}.jpg"></a><p>FP {e["fp"]}; FN {e["fn"]}; acuan lengkap {e["full_count"]}</p></article>' for e in examples)
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Bantul HOG-SVM validation</title><style>body{font:16px sans-serif;margin:24px;background:#f6f6f6}article{background:white;padding:12px;margin:20px 0}img{max-width:100%}</style><h1>Prediksi HOG-SVM pada validation</h1><p>Kotak adalah hasil model. Test belum dievaluasi.</p>'+cards)
    save_json(directory/'PIXEL_ACCESS.json',data.access)
    return measured

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--phase',choices=('prepare','fit','evaluate','all'),default='all');args=ap.parse_args()
    cv2.setNumThreads(1);output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);data=PilotData(args.root)
    source=source_snapshot(output,args.phase);start=perf_counter()
    save_json(output/'STATUS.json',dict(status='running',phase=args.phase,started_utc=datetime.now(timezone.utc).isoformat(),dataset_lock_sha256=sha(data.package/'SPLIT_LOCK.json'),source=source,
                                      machine=platform.platform(),python=platform.python_version(),opencv=cv2.__version__))
    try:
        with threadpool_limits(limits=1):
            if args.phase in ('prepare','all'):
                arrays,rows,prep=prepare(data,output/'data');prep['feature_code_sha256']=sha(HERE/'vision.py');save_json(output/'data/PREPARATION.json',prep)
            if args.phase in ('fit','all'):
                prep=json.loads((output/'data/PREPARATION.json').read_text())
                if prep['feature_cache_sha256']!=sha(output/'data/features.npz') or prep['crop_manifest_sha256']!=sha(output/'data/CROPS.json') or prep['feature_code_sha256']!=sha(HERE/'vision.py') or prep['split_lock_sha256']!=sha(data.package/'SPLIT_LOCK.json'):
                    raise ValueError('Prepared data cache no longer matches source or split')
                with np.load(output/'data/features.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
                rows=json.loads((output/'data/CROPS.json').read_text());train(data,arrays,rows,output/'models')
            if args.phase in ('evaluate','all'):evaluate(data,output)
    except Exception as e:
        save_json(output/'STATUS.json',dict(status='failed',phase=args.phase,error=repr(e),source=source));raise
    save_json(output/'STATUS.json',dict(status='complete',phase=args.phase,elapsed_seconds=perf_counter()-start,source=source,test_evaluated=False))
    save_json(output/f'ACCESS_{args.phase}.json',data.access)

if __name__=='__main__':main()
