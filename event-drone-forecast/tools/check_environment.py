"""Environment/model-contract check; does not run inference or open a recording."""
from pathlib import Path
import argparse,json,hashlib,importlib,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['base','sdk','gpu']);p.add_argument('--weights');p.add_argument('--sha256');a=p.parse_args()
    result=dict(mode=a.mode,python=sys.version.split()[0],ok=True,inference_run=False,RAW_read=False)
    try:
        names={'base':['numpy','scipy','cv2'],'sdk':['metavision_sdk_base','metavision_sdk_core','metavision_sdk_cv','metavision_core.event_io'],
            'gpu':['numpy','torch','ultralytics','onnxruntime']}[a.mode]
        result['imports']={n:getattr(importlib.import_module(n),'__version__','available') for n in names}
        if a.mode=='gpu':
            import torch,onnxruntime as ort
            result['cuda_available']=torch.cuda.is_available();result['ort_providers']=ort.get_available_providers()
            if not result['cuda_available'] or 'CUDAExecutionProvider' not in result['ort_providers']:raise ValueError('CUDA required by the current GPU backend')
            model=Path(a.weights or '')
            if not model.is_file():raise ValueError('Weights not bundled; provide a compatible model')
            actual=hashlib.sha256(model.read_bytes()).hexdigest();result['model_sha256']=actual
            if actual!=a.sha256:raise ValueError('Model SHA256 mismatch')
            session=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
            result['input_shapes']=[x.shape for x in session.get_inputs()];result['output_shapes']=[x.shape for x in session.get_outputs()]
            if result['input_shapes']!=[[1,3,384,640]] or result['output_shapes']!=[[1,5,5040]]:raise ValueError('Incompatible ONNX layout: see docs/MODEL_CARD.md')
            result['model_contract_valid']=True
    except Exception as exc:result.update(ok=False,error=type(exc).__name__+': '+str(exc))
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['ok'] else 1


if __name__=='__main__':raise SystemExit(main())
