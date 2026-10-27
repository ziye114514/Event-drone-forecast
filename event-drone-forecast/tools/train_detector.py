"""Train/export your own single-class compatible detector; no training data bundled."""
from pathlib import Path
import argparse,hashlib,json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--base',default='yolo11n.pt');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='0');a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory')
    from ultralytics import YOLO
    model=YOLO(a.base)
    model.train(data=str(a.data.resolve()),epochs=a.epochs,imgsz=640,batch=8,device=a.device,
        project=str(a.output.resolve()),name='detector',exist_ok=False,seed=27)
    best=a.output.resolve()/'detector/weights/best.pt';trained=YOLO(str(best))
    if len(trained.names)!=1:raise ValueError('The model contract requires exactly one drone class')
    path=Path(trained.export(format='onnx',imgsz=[384,640],batch=1,dynamic=False,nms=False,half=False,simplify=False,opset=17))
    import onnxruntime as ort
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    shapes=dict(input=[x.shape for x in session.get_inputs()],output=[x.shape for x in session.get_outputs()])
    if shapes!=dict(input=[[1,3,384,640]],output=[[1,5,5040]]):raise ValueError('Exported model is incompatible: '+str(shapes))
    result=dict(model_name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shapes=shapes,
        base_model=a.base,classes=trained.names,license_note='Follow Ultralytics and base-weight/data terms; export does not remove model licensing obligations')
    (a.output.resolve()/'model_metadata.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
