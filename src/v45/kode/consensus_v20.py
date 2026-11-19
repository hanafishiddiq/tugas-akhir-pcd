"""Bounded candidate box voting using selected fixed classical branches."""
import argparse,json
from pathlib import Path
import numpy as np
from data_access import PilotData
from experiment import add_f1,source_snapshot
from vision import CLASSES,overlaps,nms,membership,summarize,render,save_json,sha

def consensus_predictions(raw,camera,rois,cls,params,roi_config,vote_iou=None,nested=False):
    label=CLASSES.index(cls);ids=np.flatnonzero(raw['class_adjusted']==label)
    boxes=raw['adjusted'][ids];scores=raw['quality'][ids].copy()
    if params['prior']:scores*=np.sqrt(raw['geometry_adjusted'][ids,label])
    support=scores>=params['threshold']-params.get('vote_pool_delta',0.)
    support_boxes=boxes[support];support_scores=scores[support]
    mask=scores>=params['threshold'];boxes=boxes[mask];scores=scores[mask]
    selected=nms(boxes,scores,params['nms']);bb=[];ss=[]
    for idx in selected:
        b=boxes[idx].copy()
        if vote_iou is not None:
            candidates=np.flatnonzero(overlaps(b,support_boxes)>=vote_iou)
            candidates=candidates[np.argsort(-support_scores[candidates],kind='stable')[:params.get('vote_topk',8)]]
            b=np.rint(np.average(support_boxes[candidates],axis=0,weights=support_scores[candidates]**2)).astype(int)
        bb.append(b);ss.append(scores[idx])
    ps=[]
    for j,(b,s) in enumerate(zip(bb,ss)):
        if nested:
            area=np.prod(b[2:]-b[:2]);drop=False
            for k,other in enumerate(bb):
                if j==k:continue
                other_area=np.prod(other[2:]-other[:2]);intersection=np.maximum(0,np.minimum(b[2:],other[2:])-np.maximum(b[:2],other[:2])).prod()
                if area<=.7*other_area and intersection/area>=.9 and ss[k]>=s-.1:drop=True;break
            if drop:continue
        member,point,dist=membership(b.tolist(),cls,camera,rois,roi_config['rules'][camera],roi_config['offsets'])
        ps.append(dict(id=f'{cls}_{j:04d}',bbox_xyxy=b.tolist(),class_name=cls,confidence=float(s),roi_membership=member,contact_xy=point))
    return ps

def run(d,base,roi_path,out):
    out.mkdir(parents=True,exist_ok=False);source_snapshot(out,'consensus');cfg=json.loads((base/'MODEL_SELECTION.json').read_text());roi=json.loads(roi_path.read_text());frames=d.selected('validation');choices=[dict(vote_iou=None,nested=False)]+[dict(vote_iou=x,nested=n) for n in (False,True) for x in (.5,.65,.8)];trials=[];chosen={};selected={};rawhash={}
    for c,branch in cfg['classes'].items():
        parent=Path(branch['raw_parent']);raw={f['image_id']:dict(np.load(parent/'evaluation/raw'/(f['image_id']+'.npz'))) for f in frames};rawhash[c]={f['image_id']:sha(parent/'evaluation/raw'/(f['image_id']+'.npz')) for f in frames};best=None
        for i,choice in enumerate(choices):
            pred={}
            for f in frames:
                p=branch['global_params'] if cfg['mode']=='global' else branch['per_camera'][f['camera_id']]
                ps=consensus_predictions(raw[f['image_id']],f['camera_id'],d.rois,c,p,roi,**choice);pred[f['image_id']]=dict(status='ok',predictions=ps)
            report=add_f1(d.evaluator.evaluate_holdout(frames,d.masks,d.groups,pred,subset='validation'));s=report['summary']['detection_by_class'][c];trial=dict(class_name=c,**choice,**{k:s[k] for k in ('tp','fp','fn','f1','precision','recall')});trials.append(trial);key=(s['f1'],s['precision'] or 0,-i)
            if best is None or key>best:best=key;chosen[c]=trial;selected[c]=pred
        print(json.dumps(chosen[c]),flush=True)
    pred={}
    for f in frames:
        ps=[p for c in CLASSES for p in selected[c][f['image_id']]['predictions']];pred[f['image_id']]=dict(status='ok',predictions=ps,summary=summarize(ps,d.rois[f['camera_id']]))
    report=add_f1(d.evaluator.evaluate_holdout(frames,d.masks,d.groups,pred,subset='validation'));report.update(test_evaluated=False,selection_split='validation');ev=out/'evaluation';ev.mkdir();save_json(ev/'TRIALS.json',dict(trials=trials,selected=chosen));save_json(ev/'VALIDATION_METRICS.json',report);save_json(ev/'VALIDATION_PREDICTIONS.json',pred);save_json(out/'PARENT.json',dict(parent=str(base.resolve()),model_selection_sha256=sha(base/'MODEL_SELECTION.json'),roi_config=str(roi_path.resolve()),roi_config_sha256=sha(roi_path),raw_hashes=rawhash,lock_sha256=sha(d.package/'SPLIT_LOCK.json'),classes=chosen))
    for f in frames:render(d.read(f,'validation','v20_consensus_overlay'),pred[f['image_id']],d.rois[f['camera_id']],ev/'overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | consensus boxes | development validation",score_label='kualitas gabungan, bukan probabilitas')
    save_json(ev/'PIXEL_ACCESS.json',d.access);print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--roi-config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(PilotData(a.root),a.base,a.roi_config,a.out)
