"""Geometric consistency among independent predicted vehicles; no labels at inference."""
import argparse,copy,json,itertools
from pathlib import Path
import numpy as np
from data_access import PilotData
from experiment import source_snapshot,add_f1
from consensus_v20 import consensus_predictions
from vision import save_json,sha,summarize,render

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersection(a,b):return [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
def union_area(boxes):
    boxes=[b for b in boxes if area(b)>0];xs=sorted({x for b in boxes for x in (b[0],b[2])});total=0
    for x,u in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in boxes if b[0]<=x and b[2]>=u);length=0;end=None
        for y,v in intervals:
            length+=v-y if end is None else max(0,v-max(y,end));end=v if end is None else max(end,v)
        total+=(u-x)*length
    return total

def filter_predictions(predictions,rare_nested=False,composite_car_cover=None):
    dropped=set();rare=[(i,p) for i,p in enumerate(predictions) if p['class_name'] in ('bus','truk')]
    if rare_nested:
        for i,p in rare:
            a=p['bbox_xyxy'];aa=area(a)
            for j,q in rare:
                if i==j:continue
                b=q['bbox_xyxy']
                if aa<=.7*area(b) and area(intersection(a,b))>=.9*aa:dropped.add(i);break
    if composite_car_cover is not None:
        cars=[p['bbox_xyxy'] for p in predictions if p['class_name']=='mobil']
        for i,p in rare:
            b=p['bbox_xyxy'];inside=[intersection(b,c) for c in cars if area(intersection(b,c))>=.8*area(c)]
            if len(inside)>=2 and union_area(inside)>=composite_car_cover*area(b):dropped.add(i)
    return [p for i,p in enumerate(predictions) if i not in dropped]

def run(d,parent,out):
    out.mkdir(exist_ok=False,parents=True);source_snapshot(out,'all');assembly=json.loads((parent/'ASSEMBLY.json').read_text());roi=json.loads(Path(assembly['roi_config']).read_text());previous=json.loads((parent/'evaluation/VALIDATION_PREDICTIONS.json').read_text());frames=d.selected('validation');raw={};precomputed={};trials=[]
    for cls,thresholds in [('bus',(.65,.675,.70)),('truk',(.55,.575,.60,.625))]:
        c=assembly['selected'][cls]['global'];p=Path(assembly['specs'][c['branch']]['raw_parent']);raw[cls]={f['image_id']:dict(np.load(p/'evaluation/raw'/(f['image_id']+'.npz'))) for f in frames}
        for threshold in thresholds:
            cfg=dict(c,threshold=threshold);precomputed[cls,threshold]={f['image_id']:consensus_predictions(raw[cls][f['image_id']],f['camera_id'],d.rois,cls,cfg,roi,c['vote_iou'],c['nested']) for f in frames}
    prior_report=json.loads((parent/'evaluation/VALIDATION_METRICS.json').read_text());s=prior_report['summary'];best=(s['macro_f1'],-s['mae_full_only']['total'],-s['mae_full_only']['coverage_pp'],1);chosen=None;best_pred=previous;best_report=prior_report
    for bt,tt,nested,cover in itertools.product((.65,.675,.70),(.55,.575,.60,.625),(False,True),(None,.5,.65,.8)):
        pred={}
        for f in frames:
            i=f['image_id'];ps=[p for p in previous[i]['predictions'] if p['class_name'] not in ('bus','truk')]+precomputed['bus',bt][i]+precomputed['truk',tt][i];ps=filter_predictions(ps,nested,cover);pred[i]=dict(status='ok',predictions=ps,summary=summarize(ps,d.rois[f['camera_id']]))
        report=add_f1(d.evaluator.evaluate_holdout(frames,d.masks,d.groups,pred,subset='validation'));s=report['summary'];trial=dict(bus_threshold=bt,truk_threshold=tt,rare_nested=nested,composite_car_cover=cover,macro_f1=s['macro_f1'],mae=s['mae_full_only'],by_class=s['detection_by_class']);trials.append(trial);key=(s['macro_f1'],-s['mae_full_only']['total'],-s['mae_full_only']['coverage_pp'],0)
        if key>best:best=key;chosen=trial;best_pred=pred;best_report=report
    if chosen:
        assembly['selected']['bus']['global']['threshold']=chosen['bus_threshold'];assembly['selected']['truk']['global']['threshold']=chosen['truk_threshold'];assembly['postprocess']={k:chosen[k] for k in ('rare_nested','composite_car_cover')}
    assembly['parent']=str(parent.resolve());save_json(out/'ASSEMBLY.json',assembly);ev=out/'evaluation';ev.mkdir();best_report.update(test_evaluated=False,selection_split='validation');save_json(ev/'VALIDATION_METRICS.json',best_report);save_json(ev/'VALIDATION_PREDICTIONS.json',best_pred);save_json(ev/'TRIALS.json',dict(trials=trials,selected=chosen))
    for f in frames:render(d.read(f,'validation','v31_consistency_overlay'),best_pred[f['image_id']],d.rois[f['camera_id']],ev/'overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | geometric consistency | development",score_label='skor gabungan, bukan probabilitas')
    save_json(ev/'PIXEL_ACCESS.json',d.access);print(json.dumps(dict(selected={k:chosen[k] for k in ('bus_threshold','truk_threshold','rare_nested','composite_car_cover')} if chosen else None,macro_f1=best_report['summary']['macro_f1'],by_class={c:{k:v[k] for k in ('tp','fp','fn','f1')} for c,v in best_report['summary']['detection_by_class'].items()},mae=best_report['summary']['mae_full_only'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(PilotData(a.root),a.parent,a.out)
