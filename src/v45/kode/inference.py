"""Inference depends only on image, camera ROI and the exported numeric model."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from vision import CLASSES, Features, canonical, membership, nms, render, rgb_read, save_json, sha, summarize, window_proposals

class Model:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.config=json.loads((self.directory/'MODEL.json').read_text())
        if sha(self.directory/'weights.npz')!=self.config['weights_sha256']:
            raise ValueError('Model weight checksum mismatch')
        with np.load(self.directory/'weights.npz',allow_pickle=False) as z:
            self.weights={k:z[k].copy() for k in z.files}
        self.features=Features(self.config['features'])
        required={'classifier_coef','classifier_intercept','detector_coef','detector_intercept'}
        if set(self.weights)!=required or not all(np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError('Invalid numeric artifact')
        if self.weights['classifier_coef'].shape!=(4,self.features.dimension) or self.weights['detector_coef'].shape!=(1,self.features.dimension):
            raise ValueError('Unexpected model dimensions')
        if self.weights['classifier_intercept'].shape!=(4,) or self.weights['detector_intercept'].shape!=(1,) or set(self.config['class_order'])!=set(CLASSES):
            raise ValueError('Invalid intercept or class order')

    def score(self, rgb, camera, *, minimum_score=-1.0, grid=None):
        if camera not in self.config['rois']:
            raise ValueError('Camera ROI is not registered')
        spec=self.config['rois'][camera]
        if rgb.shape[:2]!=(spec['height'],spec['width']):
            raise ValueError('Native image size differs from locked camera ROI')
        start=perf_counter();gray=self.features.prepare(rgb)
        boxes=window_proposals(spec,grid or self.config['grid']);chunks=[];det=[];cls=[]
        for offset in range(0,len(boxes),512):
            batch=boxes[offset:offset+512];x=self.features.batch(gray,batch)
            vehicle=(x@self.weights['detector_coef'].T+self.weights['detector_intercept']).reshape(-1)
            keep=vehicle>=minimum_score
            if keep.any():
                chunks.append(batch[keep]);det.append(vehicle[keep])
                cls.append(x[keep]@self.weights['classifier_coef'].T+self.weights['classifier_intercept'])
        return {'boxes':np.concatenate(chunks) if chunks else np.empty((0,4),np.int32),
                'detector_scores':np.concatenate(det) if det else np.empty(0),
                'class_scores':np.concatenate(cls) if cls else np.empty((0,4)),
                'candidate_count':len(boxes),'seconds':perf_counter()-start,
                'minimum_stored_score':minimum_score}

    def postprocess(self, scored, camera, *, threshold=None, nms_iou=None, roi_rule=None):
        cfg=self.config['selected'];threshold=cfg['detector_threshold'] if threshold is None else threshold
        nms_iou=cfg['nms_iou'] if nms_iou is None else nms_iou;rule=roi_rule or cfg['roi_rule']
        if threshold<scored['minimum_stored_score']:
            raise ValueError('Threshold below available scored candidates')
        yes=scored['detector_scores']>=threshold;boxes=scored['boxes'][yes];scores=scored['detector_scores'][yes];classes=scored['class_scores'][yes]
        kept=nms(boxes,scores,nms_iou);predictions=[]
        for k,idx in enumerate(kept):
            box=[int(x) for x in boxes[idx]];class_id=int(classes[idx].argmax());name=self.config['class_order'][class_id]
            where,point,distance=membership(box,name,camera,self.config['rois'],rule,self.config['contact_offsets'])
            predictions.append(dict(id=f'p{k:04}',class_name=name,bbox_xyxy=box,confidence=float(scores[idx]),
                class_margin=float(classes[idx,class_id]),roi_membership=where,contact_xy=point,boundary_distance_px=distance))
        return {'status':'ok','predictions':predictions,'summary':summarize(predictions,self.config['rois'][camera]),
                'candidate_count':scored['candidate_count'],'inference_seconds':scored['seconds'],
                'model_id':self.config['model_id'],'parameters':dict(detector_threshold=threshold,nms_iou=nms_iou,roi_rule=rule)}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--image',type=Path,required=True);ap.add_argument('--camera',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();cv2.setNumThreads(1)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);model=Model(args.model);rgb=rgb_read(args.image)
    scored=model.score(rgb,args.camera,minimum_score=min(-1.,model.config['selected']['detector_threshold']))
    result=model.postprocess(scored,args.camera);result['image_sha256']=sha(args.image)
    save_json(args.output/'prediction.json',result);render(rgb,result,model.config['rois'][args.camera],args.output/'prediction.jpg')
    print(json.dumps(result['summary'],ensure_ascii=False))

if __name__=='__main__':main()
