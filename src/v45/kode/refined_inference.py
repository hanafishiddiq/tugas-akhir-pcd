"""Image-only classical inference with versioned post-consensus refinements."""
import argparse,copy,json
from pathlib import Path
from lokasi_aset import lokasi_aset
from time import perf_counter
import cv2
from threadpoolctl import threadpool_limits
from assembled_inference import AssembledModel
from residual_v42 import ResidualModel
from verifier_v43 import Verifier
from vision import canonical,rgb_read,render,save_json,sha,summarize

class RefinedModel:
    def __init__(self,runs,pipeline):
        self.pipeline_path=Path(pipeline);self.cfg=json.loads(self.pipeline_path.read_text());parent=lokasi_aset(self.cfg['parent_assembly']);assert sha(parent)==self.cfg['parent_sha256'];self.base=AssembledModel(runs,parent);self.rois=self.base.rois;self.ops=[]
        assert self.cfg['lock_sha256']==self.base.cfg['lock_sha256']
        for spec in self.cfg['operations']:
            path=lokasi_aset(spec['model_path']);assert sha(path/'fitted.joblib')==spec['model_sha256']
            if spec['kind']=='residual':obj=ResidualModel(path,self.rois)
            elif spec['kind']=='verifier':obj=Verifier(path)
            else:raise ValueError('Unknown postprocessor: '+spec['kind'])
            self.ops.append((spec,obj))
        self.provenance=dict(self.base.provenance,pipeline_sha256=sha(self.pipeline_path));self.model_id=canonical(self.provenance)[:20]
    def predict(self,rgb,camera,return_base=False):
        start=perf_counter();result=self.base.predict(rgb,camera);baseline=copy.deepcopy(result) if return_base else None;ps=result['predictions']
        for spec,obj in self.ops:
            if spec['kind']=='residual':ps=obj.apply(ps,obj.deltas(rgb,camera,ps),camera,self.base.roi,spec['strength'])
            else:ps=obj.apply(ps,obj.margins(rgb,ps),spec['classes'],spec['threshold'])
        result.update(predictions=ps,summary=summarize(ps,self.rois[camera]),model_id=self.model_id,model_provenance=self.provenance,elapsed_seconds=perf_counter()-start)
        return (baseline,result) if return_base else result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--pipeline',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--camera',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        model=RefinedModel(a.runs,a.pipeline);result=model.predict(rgb_read(a.image),a.camera);result['image_sha256']=sha(a.image);save_json(a.output.with_suffix('.json'),result);render(rgb_read(a.image),result,model.rois[a.camera],a.output.with_suffix('.jpg'),title='Classical CCTV PoC | refined development model',score_label='skor gabungan, bukan probabilitas');print(json.dumps(result['summary']))
