"""Current-frame FP32-I/O CUDA worker without historical benchmark imports."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from multiprocessing import shared_memory

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'


def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name in ['inputs.json','targets.json','detections.jsonl','geometry.jsonl'] or path.is_relative_to((ROOT.parent/'output/raw_only_project/evaluation_data').resolve()):
            raise RuntimeError('RAW worker cannot read dataset cache/truth: '+str(path))


def main(name,weights,expected_sha256,cache_dir,body_association=False,frame_rate=25.,center_ensemble_weights=None,center_ensemble_sha256=None):
    cache=Path(cache_dir).resolve();cache.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(cache/'ultralytics')
    os.environ['CUDA_CACHE_PATH']=str(cache/'cuda')
    sys.addaudithook(guard);began=time.perf_counter()
    model=Path(weights).resolve()
    assert hashlib.sha256(model.read_bytes()).hexdigest()==expected_sha256
    import cv2,numpy as np,torch
    torch.set_num_threads(2)
    from event_drone_forecast import detector_backend as backend_module
    with contextlib.redirect_stdout(sys.stderr):
        backend=backend_module.Backend('graph',model_path=model)
        secondary=None
        if center_ensemble_weights is not None:
            path=Path(center_ensemble_weights).resolve()
            assert center_ensemble_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest()==center_ensemble_sha256
            secondary=backend_module.Backend('graph',model_path=path)
        associator=None
        if body_association:
            from body_association import BodyAssociation
            associator=BodyAssociation(frame_rate)
    shared=shared_memory.SharedMemory(name=name,create=False)
    image=np.ndarray((720,1280,3),np.uint8,buffer=shared.buf);image.setflags(write=False);counter=0
    try:
        print(json.dumps(dict(ready=True,weight_sha256=expected_sha256,precision=32,
            precision_scope='FP32 input/output; internal arithmetic follows the hashed ONNX graph',
            internal_precision_hint=backend.session.get_modelmeta().custom_metadata_map.get('edf.internal_precision','not_declared'),
            cold_start_s=time.perf_counter()-began,provider_options=backend.provider_options,
            center_ensemble_weight_sha256=center_ensemble_sha256,
            body_association=associator.provenance if associator else None)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('stop'):break
            assert request['id']==counter;counter+=1
            frame_sha=hashlib.sha256(image.tobytes()).hexdigest() if request.get('verify_frame') else None
            tick=time.perf_counter();raw=backend.run(backend_module.preprocess(image));assert np.isfinite(raw).all()
            b=backend_module.boxes(raw.copy() if associator is not None else raw)
            elapsed=(time.perf_counter()-tick)*1000
            boxes=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in b]
            extra={}
            if secondary is not None:
                second_start=time.perf_counter()
                second=backend_module.boxes(secondary.run(backend_module.preprocess(image)))
                extra.update(ensemble_boxes=[dict(bbox=row[:4].tolist(),confidence=float(row[4]),class_id=int(row[5])) for row in second],
                    ensemble_inference_ms=(time.perf_counter()-second_start)*1000)
            if request.get('refine_roi') and not request.get('warmup'):
                from live_roi_refine import refine
                ref_start=time.perf_counter()
                if boxes:
                    reference=max(boxes,key=lambda row:row['confidence'])
                    refined,audit=refine(backend,backend_module,image,reference['bbox'])
                else:refined,audit=None,dict(reason='no_current_full_frame_detection')
                extra.update(refined_body=refined,refinement=audit,refinement_ms=(time.perf_counter()-ref_start)*1000)
            if associator is not None and not request.get('warmup'):
                # NMS mutates xywh, so each confidence pass gets its own copy.
                candidate=backend_module.boxes(raw.copy(),confidence=.1)
                high=candidate[candidate[:,4]>=.25]
                if high.shape!=b.shape or not np.allclose(high,b,atol=1e-5,rtol=0):
                    raise RuntimeError('High-score NMS baseline changed')
                selected,audit=associator.update(candidate,float(request['t']))
                extra=dict(candidates=[dict(bbox=r[:4].tolist(),confidence=float(r[4]),class_id=int(r[5])) for r in candidate],
                           associated_boxes=selected,association=audit,high_score_invariant=True)
            print(json.dumps(dict(id=request['id'],boxes=boxes,inference_ms=elapsed,
                frame_sha256=frame_sha,**extra),allow_nan=False),flush=True)
    finally:
        image=None;shared.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-memory',required=True)
    p.add_argument('--precision',type=int,choices=[32],default=32)
    p.add_argument('--weights',required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--cache-dir',required=True)
    p.add_argument('--body-association',action='store_true')
    p.add_argument('--frame-rate',type=float,default=25.)
    p.add_argument('--center-ensemble-weights');p.add_argument('--center-ensemble-sha256')
    a=p.parse_args()
    main(a.shared_memory,a.weights,a.expected_sha256,a.cache_dir,a.body_association,a.frame_rate,a.center_ensemble_weights,a.center_ensemble_sha256)
