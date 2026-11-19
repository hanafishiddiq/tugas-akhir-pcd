"""Validation selection of independent class thresholds and suppression."""
import argparse,json
from pathlib import Path
import numpy as np
from data_access import PilotData
from experiment import add_f1,source_snapshot
from vision import CLASSES,nms,membership,save_json,summarize,render,sha

def class_predictions(raw,camera,rois,cls,threshold,overlap,prior):
    label=CLASSES.index(cls);ids=np.flatnonzero(raw['class_adjusted']==label);boxes=raw['adjusted'][ids];scores=raw['quality'][ids].copy()
    if prior:scores*=np.sqrt(raw['geometry_adjusted'][ids,label])
    eligible=np.flatnonzero(scores>=threshold);ids2=eligible[nms(boxes[eligible],scores[eligible],overlap)]
    predictions=[]
    for j,i in enumerate(ids2):
        box=boxes[i].tolist();member,point,dist=membership(box,cls,camera,rois,'bottom',{})
        predictions.append(dict(id=f'{cls}_{j:04d}',bbox_xyxy=box,class_name=cls,confidence=float(scores[i]),roi_membership=member,contact_xy=point))
    return predictions

def main(data,base,out):
    out.mkdir(parents=True,exist_ok=True);source_snapshot(out,'calibration');ev=out/'evaluation';ev.mkdir(exist_ok=False);frames=data.selected('validation')
    raw={f['image_id']:dict(np.load(base/'evaluation/raw'/(f['image_id']+'.npz'))) for f in frames}
    trials=[];chosen={};selected_preds={}
    for cls in CLASSES:
        best=None
        for threshold in (.3,.4,.5,.6,.7,.8,.9):
            for overlap in (.25,.4,.55):
                for prior in (False,True):
                    preds={f['image_id']:dict(status='ok',predictions=class_predictions(raw[f['image_id']],f['camera_id'],data.rois,cls,threshold,overlap,prior)) for f in frames}
                    report=add_f1(data.evaluator.evaluate_holdout(frames,data.masks,data.groups,preds,subset='validation'));row=report['summary']['detection_by_class'][cls]
                    trial=dict(class_name=cls,threshold=threshold,nms=overlap,prior=prior,**{k:row[k] for k in ('tp','fp','fn','f1','precision','recall')});trials.append(trial)
                    if best is None or (trial['f1'],trial['precision'] or 0,threshold)>(best['f1'],best['precision'] or 0,best['threshold']):best=trial;best_preds=preds
        chosen[cls]=best;selected_preds[cls]=best_preds;print(json.dumps(best),flush=True)
    results={}
    for f in frames:
        ps=[p for c in CLASSES for p in selected_preds[c][f['image_id']]['predictions']];results[f['image_id']]=dict(status='ok',predictions=ps,summary=summarize(ps,data.rois[f['camera_id']]))
    report=add_f1(data.evaluator.evaluate_holdout(frames,data.masks,data.groups,results,subset='validation'));report.update(test_evaluated=False,selection_split='validation');s=report['summary']
    save_json(ev/'TRIALS.json',dict(trials=trials,selected=dict(model='v5_class_specific',classes=chosen,macro_f1=s['macro_f1'],micro_f1=s['micro_f1'],mae=s['mae_full_only'],by_class=s['detection_by_class'])));save_json(ev/'VALIDATION_METRICS.json',report);save_json(ev/'VALIDATION_PREDICTIONS.json',results)
    save_json(out/'PARENT.json',dict(parent=str(base.resolve()),raw_hashes={p.name:sha(p) for p in (base/'evaluation/raw').glob('*.npz')},lock_sha256=sha(data.package/'SPLIT_LOCK.json')))
    for f in frames:render(data.read(f,'validation','v7_prediction_overlay'),results[f['image_id']],data.rois[f['camera_id']],ev/'overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | class specific thresholds | validation")
    save_json(ev/'PIXEL_ACCESS.json',data.access);print(json.dumps(dict(macro_f1=s['macro_f1'],mae=s['mae_full_only'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(PilotData(a.root),a.base,a.out)
