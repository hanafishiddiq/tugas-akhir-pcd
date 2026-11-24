"""Shared pixel transforms and prediction geometry; no annotation dependencies."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASSES = ('mobil', 'motor', 'bus', 'truk')
FEATURE = dict(size=64, bins=9, cell=8, block=16, stride=8, gamma=True, color='RGB_to_gray')
GRID = dict(heights_720=[40,56,80,112,160,224,320,448], aspects=[.5,.75,1.1,1.6,2.3], stride_fraction=.25, min_stride=8)
_MASK_CACHE = {}

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def save_json(path, x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)

def rgb_read(path):
    with Image.open(path) as f:
        f.load()
        return np.array(f.convert('RGB'))

def valid_box(box, width, height):
    return (len(box)==4 and all(isinstance(x,(int,np.integer)) for x in box)
            and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height)

class Features:
    def __init__(self, config=None):
        self.config=dict(FEATURE if config is None else config)
        if self.config!=FEATURE:
            raise ValueError('Unsupported feature transform')
        self.hog=cv2.HOGDescriptor(_winSize=(64,64),_blockSize=(16,16),
            _blockStride=(8,8),_cellSize=(8,8),_nbins=9,_gammaCorrection=True)
        self.dimension=self.hog.getDescriptorSize()

    def prepare(self, rgb):
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
            raise ValueError('Expected RGB uint8 image')
        return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

    def crop(self, gray, box):
        if not valid_box(box,gray.shape[1],gray.shape[0]):
            raise ValueError('Invalid native half-open bbox')
        x1,y1,x2,y2=box
        tile=cv2.resize(gray[y1:y2,x1:x2],(64,64),interpolation=cv2.INTER_AREA)
        result=self.hog.compute(tile).reshape(-1)
        if result.size!=self.dimension or not np.isfinite(result).all():
            raise ValueError('Invalid HOG feature')
        return result

    def batch(self, gray, boxes):
        if len(boxes)==0:
            return np.empty((0,self.dimension),dtype=np.float32)
        return np.stack([self.crop(gray,b) for b in boxes]).astype(np.float32,copy=False)

def roi_mask(spec):
    """Rasterize at pixel centers and preserve boundary inclusion."""
    key=canonical({k:spec[k] for k in ('polygon_xy','width','height')})
    if key in _MASK_CACHE:return _MASK_CACHE[key]
    poly=spec['polygon_xy'];w,h=spec['width'],spec['height']
    # Match the source reference scanline rule exactly, including edge semantics.
    yy,xx=np.mgrid[:h,:w];x=xx+.5;y=yy+.5
    inside=np.zeros((h,w),bool);on_edge=np.zeros((h,w),bool)
    for a,b in zip(poly,poly[1:]+poly[:1]):
        x1,y1=a;x2,y2=b
        cross=(x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        on_edge |= (np.abs(cross)<1e-8)&(x>=min(x1,x2))&(x<=max(x1,x2))&(y>=min(y1,y2))&(y<=max(y1,y2))
        if y1!=y2:
            inside ^= ((y1>y)!=(y2>y)) & (x<(x2-x1)*(y-y1)/(y2-y1)+x1)
    result=inside|on_edge;result.setflags(write=False);_MASK_CACHE[key]=result
    return result

def window_proposals(spec, config=None):
    """Fixed grid uses camera geometry only, never object annotations."""
    config=dict(GRID if config is None else config)
    w,h=spec['width'],spec['height'];mask=roi_mask(spec)
    integral=np.pad(mask.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)))
    poly=np.asarray(spec['polygon_xy']);xmin,ymin=poly.min(0);xmax,ymax=poly.max(0)
    parts=[]
    for base in config['heights_720']:
        bh=round(base*h/720)
        for aspect in config['aspects']:
            bw=round(bh*aspect)
            if bw>w or bh>h:continue
            sx=max(config['min_stride'],round(bw*config['stride_fraction']))
            sy=max(config['min_stride'],round(bh*config['stride_fraction']))
            xs=np.arange(max(0,int(xmin)-bw),min(w-bw,int(xmax))+1,sx,dtype=np.int32)
            ys=np.arange(max(0,int(ymin)-bh),min(h-bh,int(ymax))+1,sy,dtype=np.int32)
            xx,yy=np.meshgrid(xs,ys);x=xx.ravel();y=yy.ravel()
            area=integral[y+bh,x+bw]-integral[y,x+bw]-integral[y+bh,x]+integral[y,x]
            keep=area>=max(8,.02*bw*bh)
            parts.append(np.stack([x[keep],y[keep],x[keep]+bw,y[keep]+bh],axis=1))
    return np.unique(np.concatenate(parts),axis=0).astype(np.int32)

def overlaps(box, boxes):
    boxes=np.asarray(boxes,dtype=float)
    if boxes.size==0:return np.empty(0)
    area=np.maximum(0,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2])).prod(1)
    union=(box[2]-box[0])*(box[3]-box[1])+(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])-area
    return area/np.maximum(union,1)

def nms(boxes, scores, threshold, max_output=100):
    if len(boxes)==0:return []
    order=np.argsort(-np.asarray(scores),kind='stable');kept=[]
    while len(order) and len(kept)<max_output:
        idx=int(order[0]);kept.append(idx)
        order=order[1:][overlaps(np.asarray(boxes[idx]),np.asarray(boxes)[order[1:]])<=threshold]
    return kept

def membership(box, class_name, camera, rois, rule, offsets):
    x1,y1,x2,y2=box
    if rule=='center':fx,fy=.5,.5
    elif rule=='bottom':fx,fy=.5,1.
    elif rule=='learned_contact':
        fx,fy=offsets.get(camera+':'+class_name,offsets.get('*:'+class_name,[.5,1.]))
    else:raise ValueError('Unknown membership rule')
    point=[float(x1+fx*(x2-x1)),float(y1+fy*(y2-y1))]
    distance=cv2.pointPolygonTest(np.asarray(rois[camera]['polygon_xy'],np.float32),tuple(point),True)
    label='inside' if distance>3 else 'outside' if distance < -3 else 'uncertain'
    return label,point,float(distance)

def summarize(predictions, spec):
    mask=roi_mask(spec);union=np.zeros_like(mask);counts={c:0 for c in CLASSES}
    for p in predictions:
        if p['roi_membership']!='inside':continue
        counts[p['class_name']]+=1;x1,y1,x2,y2=p['bbox_xyxy'];union[y1:y2,x1:x2]=True
    covered=int((union&mask).sum());area=int(mask.sum())
    return dict(count_by_class=counts,vehicle_count=sum(counts.values()),roi_area_px=area,
                covered_area_px=covered,coverage_percent=100*covered/area)

def render(rgb, result, spec, path, title='HOG + SVM prediction', score_label='margin SVM'):
    image=Image.fromarray(rgb);d=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=16)
    d.line([tuple(p) for p in spec['polygon_xy']]+[tuple(spec['polygon_xy'][0])],fill='#00d4e8',width=3)
    for p in result['predictions']:
        color={'inside':'#18ed66','outside':'#a8a8a8','uncertain':'#ffad18'}[p['roi_membership']]
        x1,y1,x2,y2=p['bbox_xyxy'];d.rectangle((x1,y1,x2-1,y2-1),outline=color,width=2)
        label=f"{p['class_name']} {p['confidence']:.2f} {p['roi_membership']}"
        bounds=d.textbbox((x1,max(0,y1-20)),label,font=font);d.rectangle(bounds,fill='black');d.text((x1,max(0,y1-20)),label,fill=color,font=font)
        px,py=p['contact_xy'];d.ellipse((px-3,py-3,px+3,py+3),fill=color)
    footer=Image.new('RGB',(image.width,image.height+74),'#111827');footer.paste(image,(0,0));d=ImageDraw.Draw(footer)
    s=result['summary'];d.text((10,image.height+5),title,fill='white',font=font)
    d.text((10,image.height+28),f"Prediksi {s['count_by_class']} | cakupan {s['coverage_percent']:.2f}%",fill='white',font=font)
    d.text((10,image.height+51),f'Kotak adalah prediksi model. Skor = {score_label}. Cyan = ROI.',fill='white',font=font)
    Path(path).parent.mkdir(parents=True,exist_ok=True);footer.save(path,quality=90)
