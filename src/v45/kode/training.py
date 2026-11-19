"""Train-only fitting, explicit validation selection and pickle-free export."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC

from data_access import contact_offsets
from vision import CLASSES, FEATURE, GRID, canonical, membership, save_json, sha

SEED=20261117

def balanced_encounter_weights(rows):
    if any(r['split']!='train' for r in rows):raise ValueError('Fitting weights received non-train data')
    groups=Counter((r['class_name'],r['encounter_id']) for r in rows)
    per_class=Counter(c for c,e in groups)
    return np.asarray([len(rows)/(len(per_class)*per_class[r['class_name']]*groups[r['class_name'],r['encounter_id']]) for r in rows])

def fit_linear(x,y,weights,c):
    if len(x)!=len(y) or len(y)!=len(weights) or not np.isfinite(x).all():raise ValueError('Invalid training arrays')
    model=LinearSVC(C=c,dual='auto',random_state=SEED,max_iter=30000,tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y,sample_weight=weights)
    return model

def crop_metrics(truth,prediction,rows):
    report=classification_report(truth,prediction,labels=list(CLASSES),output_dict=True,zero_division=0)
    by_camera={}
    for c in sorted({r['camera_id'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['camera_id']==c]
        by_camera[c]=classification_report(np.asarray(truth)[ids],np.asarray(prediction)[ids],labels=list(CLASSES),output_dict=True,zero_division=0)
    return dict(class_order=list(CLASSES),report=report,confusion=confusion_matrix(truth,prediction,labels=list(CLASSES)).tolist(),by_camera=by_camera)

def roi_diagnostics(data, offsets):
    all_rows=[];scores={}
    for rule in ('center','bottom','learned_contact'):
        rows=[]
        for f in data.selected('validation'):
            for o in f['objects']:
                if o['class_name'] not in CLASSES or o['roi_membership'] not in ('inside','outside') or o['ignore_reason']:continue
                label,point,distance=membership(o['bbox_xyxy'],o['class_name'],f['camera_id'],data.rois,rule,offsets)
                rows.append(dict(image_id=f['image_id'],camera_id=f['camera_id'],object_id=o['id'],class_name=o['class_name'],truth=o['roi_membership'],prediction=label,point=point,distance=distance))
        recalls={k:sum(r['prediction']==k and r['truth']==k for r in rows)/sum(r['truth']==k for r in rows) for k in ('inside','outside') if any(r['truth']==k for r in rows)}
        scores[rule]=dict(oracle_boxes=True,objects=len(rows),balanced_accuracy=float(np.mean(list(recalls.values()))),recall=recalls,
            uncertain=sum(r['prediction']=='uncertain' for r in rows),confusion={t:{p:sum(r['truth']==t and r['prediction']==p for r in rows) for p in ('inside','outside','uncertain')} for t in ('inside','outside')})
        all_rows.extend(dict(rule=rule,**r) for r in rows)
    chosen=max(scores,key=lambda k:(scores[k]['balanced_accuracy'],scores[k]['recall'].get('inside',0),k=='bottom'))
    return {'chosen':chosen,'scores':scores,'objects':all_rows,'limitation':'Diagnostic uses reference boxes/classes; full-image prediction is evaluated separately.'}

def export_model(directory, classifier, detector, data, offsets, chosen_roi, tag, training_info):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/'weights.npz',classifier_coef=classifier.coef_.astype(np.float32),classifier_intercept=classifier.intercept_.astype(np.float32),
                        detector_coef=detector.coef_.astype(np.float32),detector_intercept=detector.intercept_.astype(np.float32))
    cfg=dict(model_id='',tag=tag,weights_sha256=sha(directory/'weights.npz'),features=FEATURE,grid=GRID,
        class_order=classifier.classes_.tolist(),score_semantics='Uncalibrated binary SVM margin; class is argmax of four OVR class margins.',
        rois=data.rois,contact_offsets=offsets,selected=dict(detector_threshold=0.,nms_iou=.3,roi_rule=chosen_roi),
        dataset_lock_sha256=sha(data.package/'SPLIT_LOCK.json'),seed=SEED,sklearn_version=sklearn.__version__,training=training_info)
    cfg['model_id']=canonical({k:v for k,v in cfg.items() if k!='model_id'})[:20];save_json(directory/'MODEL.json',cfg)
    return cfg

def train(data, arrays, rows, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    xp=arrays['train_positive'];vp=arrays['validation_positive'];yp=np.asarray([r['class_name'] for r in rows['train_positive']]);yv=np.asarray([r['class_name'] for r in rows['validation_positive']])
    weights=balanced_encounter_weights(rows['train_positive']);trials=[];models={}
    majority=Counter(yp).most_common(1)[0][0]
    for c in (.01,.1,1.,10.):
        model=fit_linear(xp,yp,weights,c);models[c]=model
        metrics=crop_metrics(yv,model.predict(vp),rows['validation_positive'])
        trial=dict(c=c,iterations=int(np.max(model.n_iter_)),**metrics);trials.append(trial)
        print(json.dumps({'crop_C':c,'macro_f1':metrics['report']['macro avg']['f1-score']},ensure_ascii=False),flush=True)
    best=max(trials,key=lambda t:(t['report']['macro avg']['f1-score'],-t['c']));classifier=models[best['c']]
    crop_report=dict(selected_C=best['c'],selected=best,trials=trials,majority_baseline=crop_metrics(yv,np.full(len(yv),majority),rows['validation_positive']),
                     validation_predictions=[dict(object_id=r['object_id'],image_id=r['image_id'],truth=r['class_name'],prediction=str(pred)) for r,pred in zip(rows['validation_positive'],classifier.predict(vp))])
    save_json(directory/'CROP_EVALUATION.json',crop_report)
    offsets=contact_offsets(data);roi=roi_diagnostics(data,offsets);save_json(directory/'ROI_DIAGNOSTIC.json',roi)
    use=np.asarray([r['sample_role']!='auxiliary_classification_train_only' for r in rows['train_positive']]);core_x=xp[use];core_rows=[r for r,k in zip(rows['train_positive'],use) if k]
    pool=arrays['negative_pool'];initial=np.asarray([r['initial'] for r in rows['negative_pool']]);initial_ids=np.flatnonzero(initial)
    def detector_fit(ids):
        nx=pool[ids];n=len(core_x)+len(nx);pw=balanced_encounter_weights(core_rows)*n/(2*len(core_rows))
        bg_rows=[rows['negative_pool'][int(i)]|{'split':'train'} for i in ids];nw=balanced_encounter_weights(bg_rows)*n/(2*len(bg_rows))
        return fit_linear(np.concatenate([core_x,nx]),np.r_[np.ones(len(core_x),int),np.zeros(len(nx),int)],np.r_[pw,nw],1.)
    first=detector_fit(initial_ids);scores=first.decision_function(pool);extra=[]
    for image_id in sorted({r['image_id'] for r in rows['negative_pool']}):
        candidates=[i for i,r in enumerate(rows['negative_pool']) if r['image_id']==image_id and not initial[i] and scores[i]>0]
        extra.extend(sorted(candidates,key=lambda i:-scores[i])[:64])
    all_ids=np.unique(np.r_[initial_ids,np.asarray(extra,int)]);second=detector_fit(all_ids)
    info=dict(classifier_C=best['c'],classifier_train_counts=dict(Counter(yp)),classifier_encounters={c:len({r['encounter_id'] for r in rows['train_positive'] if r['class_name']==c}) for c in CLASSES},
              detector_C=1.,detector_core_positive_counts=dict(Counter(r['class_name'] for r in core_rows)),auxiliary_used_by_detector=False,negative_pool=len(pool),initial_negatives=len(initial_ids),hard_negatives_added=len(extra),weighting='equal class/encounter total for positives; equal component total for background; binary sides balanced')
    save_json(directory/'HARD_NEGATIVES.json',dict(selected=[rows['negative_pool'][int(i)]|{'initial_detector_margin':float(scores[i])} for i in extra],training=info))
    exported={}
    for name,detector in [('initial',first),('hard_negative',second)]:
        cfg=export_model(directory/name,classifier,detector,data,offsets,roi['chosen'],name,info)
        # Verify public numeric export against the fitted estimators before full-image inference.
        from inference import Model
        loaded=Model(directory/name);x=np.concatenate([vp[:64],pool[:64]])
        assert np.allclose(x@loaded.weights['classifier_coef'].T+loaded.weights['classifier_intercept'],classifier.decision_function(x),atol=2e-4)
        assert np.allclose((x@loaded.weights['detector_coef'].T+loaded.weights['detector_intercept']).ravel(),detector.decision_function(x),atol=2e-4)
        exported[name]=cfg
    save_json(directory/'TRAINING.json',dict(**info,export_parity_passed=True,roi_rule=roi['chosen']))
    return crop_report,roi,exported
