"""Export native ON/OFF detector images; never invent bounding-box annotations."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=0.)
    p.add_argument('--duration',type=float,default=10.);p.add_argument('--fps',type=int,choices=[5,10,25,50],default=10);a=p.parse_args()
    if a.output.exists():p.error('Choose a new output directory')
    import cv2,numpy as np
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    from live_decode import native_batches
    if a.start<0 or a.duration<=0:p.error('Positive duration/nonnegative start required')
    step=20000;start=np.floor(a.start*1e6/step)*step/1e6
    noise=ActivityNoiseFilterAlgorithm(1280,720,10000);buf=noise.get_empty_output_buffer();image=np.empty((720,1280,3),np.uint8)
    out=a.output.resolve();images=out/'images';images.mkdir(parents=True);records=[]
    for i,(t,e) in enumerate(native_batches(str(a.raw.resolve()),start,a.duration,False,step)):
        noise.process_events(e,buf);clean=buf.numpy()
        if i%(50//a.fps):continue
        BaseFrameGenerationAlgorithm.generate_frame(clean,image);name=f'{i:07d}.png';cv2.imwrite(str(images/name),image)
        records.append(dict(image=name,t=t,warmup=i==0))
    (out/'frames.json').write_text(json.dumps(dict(native_ON_OFF=True,intensity_reconstruction=False,
        sensor_shape=[720,1280],window_us=20000,annotations_created=False,records=records),indent=2),'utf-8')
    (out/'ANNOTATE.md').write_text('Images contain native ON/OFF features. Create real single-class YOLO bounding-box labels before training; empty/automatic placeholder labels are not generated. Split train/val/test by recording.\n','utf-8')
    print('Exported',len(records),'ON/OFF images; no labels created.')


if __name__=='__main__':main()
