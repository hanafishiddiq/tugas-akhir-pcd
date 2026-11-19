"""Conservative TRAIN-fitted whole-vehicle verification after consensus."""
import argparse,copy,json
from pathlib import Path
import cv2,joblib,numpy as np
from threadpoolctl import threadpool_limits
from data_access import PilotData
from experiment import add_f1,source_snapshot
from reject_v21 import features
from vision import Features,sha,save_json,summarize,render

class Verifier:
    def __init__(self,path):
        self.path=Path(path);self.cfg=json.loads((self.path/'MODEL.json').read_text());assert sha(self.path/'fitted.joblib')==self.cfg['model_sha256'];self.model=joblib.load(self.path/'fitted.joblib');self.hog=Features();self.classes=list(self.model.classes_)
    def margins(self,rgb,predictions):
        if not predictions:return np.empty(0)
        x=features(self.hog,rgb,predictions);s=self.model.decision_function(x);neg=np.max(s[:,[self.classes.index(c) for c in ('other','background')]],axis=1)
        return np.array([neg[j]-s[j,self.classes.index(p['class_name'])] for j,p in enumerate(predictions)])
    @staticmethod
    def apply(predictions,margins,classes,threshold):
        return [p for p,m in zip(predictions,margins) if p['class_name'] not in classes or m<threshold]

def run(d,runs,out):
    out.mkdir(parents=True,exist_ok=False);source_snapshot(out,'all');parent=runs/'localization_20261117_v35';modelpath=runs/'localization_20261117_v21/models/other_fragments';v=Verifier(modelpath);train=modelpath.parent.parent;cfg=v.cfg
    assert sha(train/'TRAIN_MANIFEST.json')==cfg['training_manifest_sha256'];assert sha(train/'TRAIN_FEATURES.npz')==cfg['training_features_sha256'];rows=json.loads((train/'TRAIN_MANIFEST.json').read_text())
    for r in rows:
        assert r['split']=='train'
        if r['image_id'] in d.groups:assert d.groups[r['image_id']]['split']=='train'
        else:assert r['identity_eligible'] and r['independent_review_passed']
    save_json(out/'TRAIN_LINEAGE.json',dict(model_path=str(modelpath.resolve()),model_sha256=cfg['model_sha256'],manifest_sha256=cfg['training_manifest_sha256'],features_sha256=cfg['training_features_sha256'],rows=len(rows),test_used=False))
    frames=d.selected('validation');pred=json.loads((parent/'evaluation/VALIDATION_PREDICTIONS.json').read_text());baseline=json.loads((parent/'evaluation/VALIDATION_METRICS.json').read_text());margins={}
    for f in frames:margins[f['image_id']]=v.margins(d.read(f,'validation','v43_final_box_verification'),pred[f['image_id']]['predictions'])
    np.savez_compressed(out/'MARGINS.npz',**margins);best=baseline;bestpred=pred;chosen=None;trials=[]
    for classes in [('motor',),('mobil',),('mobil','motor')]:
        for threshold in (0.,.5,1.,2.):
            predictions={}
            for f in frames:
                i=f['image_id'];ps=v.apply(pred[i]['predictions'],margins[i],classes,threshold);predictions[i]=dict(status='ok',predictions=ps,summary=summarize(ps,d.rois[f['camera_id']]))
            report=add_f1(d.evaluator.evaluate_holdout(frames,d.masks,d.groups,predictions,subset='validation'));s=report['summary'];bs=baseline['summary'];eligible=s['macro_f1']>bs['macro_f1'] and all(s['detection_by_class'][c]['f1']>=bs['detection_by_class'][c]['f1'] for c in bs['detection_by_class']) and s['mae_full_only']['total']<=1 and s['mae_full_only']['coverage_pp']<=5
            trial=dict(classes=list(classes),threshold=threshold,macro_f1=s['macro_f1'],mae=s['mae_full_only'],by_class={c:{k:v[k] for k in ('tp','fp','fn','f1')} for c,v in s['detection_by_class'].items()},eligible=eligible);trials.append(trial);print(json.dumps(trial),flush=True)
            if eligible and s['macro_f1']>best['summary']['macro_f1']:best=report;bestpred=predictions;chosen=trial
    best.update(test_evaluated=False,selection_split='validation');save_json(out/'evaluation/VALIDATION_METRICS.json',best);save_json(out/'evaluation/VALIDATION_PREDICTIONS.json',bestpred);save_json(out/'evaluation/TRIALS.json',dict(trials=trials,selected=chosen))
    save_json(out/'POSTPROCESS.json',dict(parent_assembly=str((parent/'ASSEMBLY.json').resolve()),parent_sha256=sha(parent/'ASSEMBLY.json'),verifier_model=str(modelpath.resolve()),verifier_model_sha256=cfg['model_sha256'],selected=chosen,lock_sha256=sha(d.package/'SPLIT_LOCK.json')))
    for f in frames:render(d.read(f,'validation','v43_overlay'),bestpred[f['image_id']],d.rois[f['camera_id']],out/'evaluation/overlays'/(f['image_id']+'.jpg'),title=f"{f['image_id']} | final-box verifier | development",score_label='skor gabungan, bukan probabilitas')
    save_json(out/'PIXEL_ACCESS.json',d.access)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    with threadpool_limits(limits=1):run(PilotData(a.root),a.runs,a.out)
