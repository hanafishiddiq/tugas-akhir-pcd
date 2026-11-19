"""Hash-checked, split-restricted access to the frozen pilot dataset."""
from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from vision import CLASSES, Features, canonical, rgb_read, roi_mask, save_json, sha, window_proposals

class PilotData:
    def __init__(self, root):
        self.root=Path(root).resolve();self.package=self.root/'Data/Eksperimen/Bantul_Pilot_v1'
        self.lock=json.loads((self.package/'SPLIT_LOCK.json').read_text())
        if not self.lock['locked']:raise ValueError('Dataset split is not locked')
        for key,path in self.lock['paths'].items():
            if sha(self.root/path)!=self.lock['input_sha256'][key]:raise ValueError('Frozen input changed: '+key)
        sys.path.insert(0,str(self.package/'code'))
        self.adapter=importlib.import_module('dataset');self.evaluator=importlib.import_module('evaluate')
        if Path(self.adapter.__file__).resolve()!=self.package/'code/dataset.py':raise RuntimeError('Wrong dataset adapter imported')
        for name,digest in self.lock['code_sha256'].items():
            path=self.package/'code'/name
            if not path.exists() and name=='annotation_reference_core':path=self.root/'Data/Anotasi/Bantul_222_v1/code/core.py'
            if not path.is_file() or sha(path)!=digest:raise ValueError('Frozen helper changed or missing: '+name)
        self.groups=json.loads((self.package/'GROUPS.json').read_text())['images']
        if {i:g['split'] for i,g in self.groups.items()}!=self.lock['assignments']:raise ValueError('Split mismatch')
        self.frames={x['image_id']:x for x in json.loads((self.package/'ANNOTATIONS.json').read_text()) if self.groups[x['image_id']]['split'] in ('train','validation')}
        self.aux=json.loads((self.package/'AUXILIARY_MANIFEST.json').read_text())
        self.rois=json.loads((self.package/'ROIS.json').read_text());self.masks={c:roi_mask(r) for c,r in self.rois.items()}
        for c,mask in self.masks.items():
            actual=np.array(Image.open(self.root/f'Data/ROI/Bantul_v1/masks/{c}_road.png'))>0
            if not np.array_equal(mask,actual):raise ValueError('Raster ROI differs from frozen mask: '+c)
        self.access=[];self.verified=set()

    def read(self, frame, subset, purpose):
        image_id=frame['image_id']
        if subset not in ('train','validation') or self.groups[image_id]['split']!=subset:
            raise ValueError('Pixel access outside the authorized experiment split: '+image_id)
        p=self.root/frame['source_image'];digest=frame['source_sha256']
        if str(p) not in self.verified:
            if sha(p)!=digest:raise ValueError('Source image checksum changed: '+image_id)
            self.verified.add(str(p))
        self.access.append(dict(image_id=image_id,split=subset,purpose=purpose,source_image=frame['source_image'],sha256=digest))
        rgb=rgb_read(p)
        if rgb.shape[:2]!=(frame['height'],frame['width']):raise ValueError('Source dimensions changed')
        return rgb

    def selected(self, subset):
        if subset not in ('train','validation'):raise ValueError('Test is reserved for a later stage')
        return [f for f in self.frames.values() if self.groups[f['image_id']]['split']==subset]

    def positives(self, frame, subset):
        entry=self.groups[frame['image_id']]
        if subset=='train':return self.adapter.train_positives(frame,self.masks[frame['camera_id']],entry)
        if subset!='validation' or entry['split']!=subset:raise ValueError('Invalid crop evaluation subset')
        return [dict(image_id=frame['image_id'],object_id=o['id'],source_image=frame['source_image'],source_sha256=frame['source_sha256'],
                     class_name=o['class_name'],bbox_xyxy=o['bbox_xyxy'],camera_id=frame['camera_id'],split=subset,
                     encounter_id=entry['object_encounter_ids'][o['id']],component_id=entry['component_id'],sample_role=entry['sample_role'])
                for o in frame['objects'] if self.adapter.accepted(o)]

    def safe_negatives(self, frame, candidates):
        """Vectorized candidate prefilter, then confirm with frozen adapter."""
        entry=self.groups[frame['image_id']]
        if entry['split']!='train' or entry['sample_role']!='queue_or_control' or not frame['metrics']['count_reference_available']:
            return []
        m=self.masks[frame['camera_id']];integ=np.pad(m.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
        x1,y1,x2,y2=candidates.T
        permitted=(integ[y2,x2]-integ[y1,x2]-integ[y2,x1]+integ[y1,x1])==((x2-x1)*(y2-y1))
        labels=self.adapter.adapt(frame,m)
        for a,b,c,d in labels.exclusion_boxes:
            permitted &= ~((x1<c+8)&(x2>a-8)&(y1<d+8)&(y2>b-8))
        allowed=candidates[permitted]
        rng=np.random.default_rng(int(canonical(frame['image_id'])[:8],16))
        if len(allowed)>400:allowed=allowed[rng.choice(len(allowed),400,replace=False)]
        rows=self.adapter.negative_manifest(frame,m,entry,[[int(v) for v in b] for b in allowed])
        if len(rows)!=len(allowed):raise ValueError('Negative sampling disagrees with frozen validator')
        for i,row in enumerate(rows):row.update(camera_id=frame['camera_id'],initial=i<64,encounter_id=entry['component_id'],class_name='background')
        return rows

def prepare(data, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);features=Features()
    arrays={};records={};grid_by_camera={c:window_proposals(r) for c,r in data.rois.items()}
    for subset in ('train','validation'):
        x=[];rows=[]
        for frame in data.selected(subset):
            positives=data.positives(frame,subset)
            if not positives:continue
            rgb=data.read(frame,subset,'crop_features');gray=features.prepare(rgb)
            for row in positives:
                row['camera_id']=frame['camera_id'];box=row['bbox_xyxy'];x.append(features.crop(gray,box));rows.append(row)
                if subset=='train':
                    a,b,c,d=box;path=directory/'train_crops'/f"{row['object_id']}__{row['class_name']}.png"
                    path.parent.mkdir(exist_ok=True);Image.fromarray(rgb[b:d,a:c]).save(path)
                    row['exported_crop']=str(path.relative_to(directory));row['exported_crop_sha256']=sha(path)
        if subset=='train':
            for frame in data.aux:
                rgb=data.read(frame,'train','auxiliary_classifier_crop');gray=features.prepare(rgb)
                for row in data.adapter.auxiliary_positive_manifest(frame,data.groups[frame['image_id']]):
                    x.append(features.crop(gray,row['bbox_xyxy']));rows.append(row)
                    a,b,c,d=row['bbox_xyxy'];path=directory/'train_crops'/f"{row['object_id']}__{row['class_name']}.png"
                    Image.fromarray(rgb[b:d,a:c]).save(path);row['exported_crop']=str(path.relative_to(directory));row['exported_crop_sha256']=sha(path)
        arrays[subset+'_positive']=np.asarray(x,np.float32);records[subset+'_positive']=rows
    nx=[];nr=[]
    for frame in data.selected('train'):
        rows=data.safe_negatives(frame,grid_by_camera[frame['camera_id']])
        if not rows:continue
        rgb=data.read(frame,'train','background_pool');gray=features.prepare(rgb)
        nx.extend(features.batch(gray,[x['bbox_xyxy'] for x in rows]));nr.extend(rows)
    arrays['negative_pool']=np.asarray(nx,np.float32);records['negative_pool']=nr
    np.savez_compressed(directory/'features.npz',**arrays);save_json(directory/'CROPS.json',records)
    summary=dict(split_lock_sha256=sha(data.package/'SPLIT_LOCK.json'),features=features.config,
        positive_counts={split:{c:sum(x['class_name']==c for x in records[split+'_positive']) for c in CLASSES} for split in ('train','validation')},
        negative_pool=len(nr),initial_negatives=sum(x['initial'] for x in nr),test_pixels_opened=0,
        feature_cache_sha256=sha(directory/'features.npz'),crop_manifest_sha256=sha(directory/'CROPS.json'),
        grid_counts={c:len(b) for c,b in grid_by_camera.items()})
    save_json(directory/'PREPARATION.json',summary);save_json(directory/'PIXEL_ACCESS.json',data.access)
    return arrays,records,summary

def contact_offsets(data):
    values=defaultdict(list)
    for f in data.selected('train'):
        for o in f['objects']:
            if o['class_name'] not in CLASSES or not o['road_contact_xy'] or o['roi_membership'] not in ('inside','outside') or o['ignore_reason']:
                continue
            x1,y1,x2,y2=o['bbox_xyxy'];px,py=o['road_contact_xy'];offset=[(px-x1)/(x2-x1),(py-y1)/(y2-y1)]
            if not (0<=offset[0]<=1 and 0<=offset[1]<=1):continue
            values[f["camera_id"]+':'+o['class_name']].append(offset);values['*:'+o['class_name']].append(offset)
    return {key:np.median(v,axis=0).tolist() for key,v in values.items() if len(v)>=5}
