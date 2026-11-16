import hashlib,json,math
import numpy as np
TARGETS=('mobil','motor','bus','truk')
CLASSES=TARGETS+('other','unknown')
def annotation_hash(record):
 return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def bbox_valid(box,width,height):
 return isinstance(box,list) and len(box)==4 and all(type(v) is int for v in box) and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height
def boundary_distance(point,poly):
 p=np.asarray(point,float);best=float('inf')
 for a,b in zip(poly,poly[1:]+poly[:1]):
  a=np.asarray(a,float);d=np.asarray(b,float)-a;t=np.clip(np.dot(p-a,d)/np.dot(d,d),0,1);best=min(best,float(np.linalg.norm(p-(a+t*d))))
 return best
def point_inside(point,poly):
 x,y=point;inside=False
 if boundary_distance(point,poly)<1e-8:return True
 for a,b in zip(poly,poly[1:]+poly[:1]):
  if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
 return inside
def union_area(mask,boxes):
 union=np.zeros(mask.shape,bool)
 for x1,y1,x2,y2 in boxes:union[y1:y2,x1:x2]=True
 return int(np.count_nonzero(union&mask))
def metrics(objects,regions,mask):
 accepted=[o for o in objects if o['class_name'] in TARGETS and o['roi_membership']=='inside' and o['role']=='target_approach' and o['confidence']!='low' and not o['ignore_reason']]
 accepted_ids={o['id'] for o in accepted}
 pending=[o for o in objects if o['class_name'] in TARGETS+('unknown',) and o['roi_membership']!='outside' and o['id'] not in accepted_ids]
 # Projected roof/helmet pixels can lie outside the road while the hidden base
 # is inside it. Physical relevance must never be inferred from raster overlap.
 relevant_regions=[r for r in regions if r.get('roi_relevance','possible_target_in_roi')=='possible_target_in_roi']
 counts={c:sum(o['class_name']==c for o in accepted) for c in TARGETS};area=union_area(mask,[o['bbox_xyxy'] for o in accepted]);den=int(mask.sum());complete=not pending and not relevant_regions
 upper={c:counts[c]+sum(o['class_name'] in (c,'unknown') for o in pending) for c in TARGETS}
 envelope_area=union_area(mask,[o['bbox_xyxy'] for o in accepted+pending]+[r['bbox_xyxy'] for r in relevant_regions])
 # An ambiguous helmet/fragment is not a complete vehicle bbox. Its localized
 # region cannot bound the missing vehicle's rectangle, so no numeric upper.
 return {'accepted_object_ids':[o['id'] for o in accepted],'pending_object_ids':[o['id'] for o in pending],'count_reference_available':complete,'counts_reference':counts if complete else None,'total_count_reference':len(accepted) if complete else None,'coverage_reference_percent':100*area/den if complete else None,'counts_resolved_lower_bound':counts,'total_count_lower_bound':len(accepted),'count_upper_bound_by_class':None if relevant_regions else upper,'total_count_upper_bound':None if relevant_regions else len(accepted)+len(pending),'covered_area_lower_px':area,'covered_area_upper_px':None if relevant_regions else envelope_area,'observed_envelope_area_px':envelope_area,'roi_area_px':den,'coverage_lower_percent':100*area/den,'coverage_upper_percent':None if relevant_regions else 100*envelope_area/den,'coverage_observed_envelope_percent':100*envelope_area/den,'upper_count_semantics':'Per-class bounds are marginal, not simultaneously additive; object instances across photos are not unique vehicles.','coverage_upper_semantics':'Union of recorded identifiable object bboxes; null when a relevant ignore region cannot bound a vehicle extent. Observed rectangle envelope is diagnostic only.','incomplete_region_count':len(relevant_regions)}
def schema_errors(r,m):
 errors=[];w,h=m['width'],m['height'];required=['image_id','source_image','camera_id','native_reviewed','self_overlay_reviewed','annotation_scope','review_notes','visual_evidence','objects','ignore_regions','reviewer','source_legacy_annotation']
 for k in required:
  if k not in r:errors.append('missing '+k)
 if errors:return errors
 for k in ('image_id','source_image','camera_id'):
  if r[k]!=m[k]:errors.append('input mismatch '+k)
 if r['native_reviewed'] is not True or r['self_overlay_reviewed'] is not True:errors.append('visual self-review incomplete')
 ids=[];boxes=[]
 for o in r['objects']:
  need=['id','bbox_xyxy','class_name','subtype','confidence','role','occlusion','truncated','roi_membership','membership_reason','road_contact_xy','road_contact_note','crosses_roi_boundary','classification_note','ignore_reason']
  missing=[k for k in need if k not in o]
  if missing:errors.append(str(o.get('id'))+' missing '+','.join(missing));continue
  oid=o['id'];ids.append(oid);boxes.append(tuple(o['bbox_xyxy']))
  if not oid.startswith(r['image_id']+'-'):errors.append(oid+' ID prefix')
  if not bbox_valid(o['bbox_xyxy'],w,h):errors.append(oid+' invalid bbox')
  for k,values in [('class_name',CLASSES),('confidence',('high','medium','low')),('role',('target_approach','context')),('occlusion',('none','partial','heavy')),('roi_membership',('inside','outside','boundary_review','unknown'))]:
   if o[k] not in values:errors.append(oid+' invalid '+k)
  for k in ('subtype','membership_reason','road_contact_note','classification_note'):
   if not isinstance(o[k],str) or not o[k].strip():errors.append(oid+' empty '+k)
  if type(o['truncated']) is not bool:errors.append(oid+' truncated must be bool')
  if o['crosses_roi_boundary'] not in [True,False,'unknown']:errors.append(oid+' invalid boundary flag')
  p=o['road_contact_xy']
  if p is None and 'presisi tidak dicatat' in o['road_contact_note']:errors.append(oid+' null contact needs actual visibility/ambiguity reason, not omitted precision')
  if p is not None and (not isinstance(p,list) or len(p)!=2 or not all(isinstance(t,(float,int)) and math.isfinite(t) for t in p) or not 0<=p[0]<w or not 0<=p[1]<h):errors.append(oid+' invalid contact point')
  if (o['class_name']=='unknown' or o['confidence']=='low' or o['roi_membership'] in ['unknown','boundary_review']) and not o['ignore_reason']:errors.append(oid+' uncertainty needs ignore_reason')
 if len(set(ids))!=len(ids):errors.append('duplicate object IDs')
 if len(set(boxes))!=len(boxes):errors.append('identical bboxes need correction/review')
 for region in r['ignore_regions']:
  if not bbox_valid(region.get('bbox_xyxy'),w,h) or not region.get('reason'):errors.append('invalid ignore region')
  if region.get('roi_relevance') not in ['possible_target_in_roi','outside_target_roi']:errors.append('ignore region needs explicit physical roi_relevance')
 return errors
