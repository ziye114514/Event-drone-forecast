"""Public source launcher. Help/dry-run never import SDK/CUDA or read RAW contents."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
REFERENCE='ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'


def configuration(value):
    path=Path(value).expanduser().resolve() if value else ROOT/'configs/local.json'
    if value and not path.is_file():raise ValueError('Configuration not found: '+str(path))
    cfg=json.loads(path.read_text('utf-8')) if path.exists() else {}
    for key in ('sdk_python','gpu_python','detector_weights','cache_dir'):
        v=cfg.get(key)
        if v and (key not in ('sdk_python','gpu_python') or '/' in v or '\\' in v):
            p=Path(v).expanduser();cfg[key]=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    return cfg


def executable(value):
    path=Path(shutil.which(str(value)) or str(value)).expanduser().resolve()
    if not path.is_file():raise ValueError('Python executable not found: '+str(value))
    return str(path)


def settings(a):
    cfg=configuration(a.config)
    def choose(key,env,default):return getattr(a,key,None) or os.environ.get(env) or cfg.get(key) or default
    return dict(sdk_python=executable(choose('sdk_python','EDF_SDK_PYTHON',sys.executable)),
        gpu_python=executable(choose('gpu_python','EDF_GPU_PYTHON',sys.executable)),
        detector_weights=str(Path(choose('detector_weights','EDF_DETECTOR_WEIGHTS',ROOT/'weights/best.onnx')).expanduser().resolve()),
        detector_sha256=choose('detector_sha256','EDF_DETECTOR_SHA256',REFERENCE),
        cache_dir=str(Path(choose('cache_dir','EDF_CACHE_DIR',ROOT/'.cache')).expanduser().resolve()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Synthetic five-circle/phase/force controls; no RAW, weights, SDK or GPU')
    demo.add_argument('--output',type=Path,default=ROOT/'outputs/synthetic_demo')
    doctor=sub.add_parser('doctor',help='Check base dependencies, or SDK/GPU/model contract')
    doctor.add_argument('--base-only',action='store_true')
    predict=sub.add_parser('predict',aliases=['structure-fast'],help='Current physical-only paced RAW source replay')
    predict.add_argument('--raw',required=True,type=Path);predict.add_argument('--output',required=True,type=Path)
    predict.add_argument('--start',type=float,default=0.);predict.add_argument('--duration',type=float,default=2.)
    predict.add_argument('--hz',type=float,default=30.);predict.add_argument('--dry-run',action='store_true')
    predict.add_argument('--preview',action='store_true');predict.add_argument('--kinematic-control',action='store_true')
    predict.add_argument('--legacy-structure',action='store_true');predict.add_argument('--no-partial-arc-information',action='store_true')
    predict.add_argument('--delayed-disturbance-control',action='store_true');predict.add_argument('--no-force-gap-hold',action='store_true')
    predict.add_argument('--no-frame-verification',action='store_true')
    predict.add_argument('--profile',choices=('baseline','candidate'),default='baseline',help='Baseline preserves alpha.1 launch options; candidate enables the documented alpha.2 joint physics configuration')
    predict.add_argument('--sdk-avx2',action='store_true',help='Optional local SDK-header adapter; requires an explicit local build, AVX2 CPU and compatible SDK license')
    predict.add_argument('--current-nn-on-loss',action='store_true',help='Use current NN if current image tracking is lost; used in the train2 development run')
    for cmd in (doctor,predict):
        cmd.add_argument('--config');cmd.add_argument('--sdk-python');cmd.add_argument('--gpu-python')
        cmd.add_argument('--detector-weights');cmd.add_argument('--detector-sha256');cmd.add_argument('--cache-dir')
    a=p.parse_args(argv)
    try:
        if a.command=='demo':
            return subprocess.call([sys.executable,'-X','utf8',str(ROOT/'tools/demo_synthetic.py'),'--output',str(a.output.resolve())],cwd=ROOT)
        if a.command=='doctor' and a.base_only:
            return subprocess.call([sys.executable,str(ROOT/'tools/check_environment.py'),'base'],cwd=ROOT)
        cfg=settings(a)
        if a.command=='doctor':
            results=[]
            for mode,python in [('sdk',cfg['sdk_python']),('gpu',cfg['gpu_python'])]:
                cmd=[python,str(ROOT/'tools/check_environment.py'),mode]
                if mode=='gpu':cmd+=['--weights',cfg['detector_weights'],'--sha256',cfg['detector_sha256']]
                results.append(subprocess.call(cmd,cwd=ROOT))
            return int(any(results))
        raw=a.raw.expanduser().resolve();out=a.output.expanduser().resolve()
        if not raw.is_file():raise ValueError('RAW not found: '+str(raw))
        if out.exists():raise ValueError('Choose a new output directory: '+str(out))
        if a.start<0 or a.duration<=0 or not 1<=a.hz<=50:raise ValueError('Require start>=0, duration>0, 1<=hz<=50')
        model=Path(cfg['detector_weights'])
        if not model.is_file():raise ValueError('Compatible weights are not bundled. See docs/MODEL_CARD.md: '+str(model))
        actual=hashlib.sha256(model.read_bytes()).hexdigest()
        if actual!=cfg['detector_sha256']:raise ValueError('Weight SHA256 mismatch; use the exact hash of your compatible model')
        command=[cfg['sdk_python'],'-X','utf8',str(ROOT/'run_fast_structure.py'),
            '--raw',str(raw),'--output',str(out),'--start',str(a.start),'--duration',str(a.duration),'--hz',str(a.hz),
            '--gpu-python',cfg['gpu_python'],'--detector-weights',str(model),'--detector-sha256',actual,
            '--cache-dir',cfg['cache_dir'],'--async-nn','--paced']
        if not a.no_frame_verification:command+=['--verify-frames']
        for flag in ('preview','legacy_structure','no_partial_arc_information','delayed_disturbance_control','no_force_gap_hold'):
            if getattr(a,flag):command+=['--'+flag.replace('_','-')]
        if a.kinematic_control:command+=['--with-kinematic-control']
        if a.profile=='candidate':command+=['--target-candidate','--rigid-motion','--tangent-disturbance','--structure-guided-nn']
        if a.sdk_avx2:command+=['--sdk-avx2']
        if a.current_nn_on_loss:command+=['--current-nn-on-loss']
        if a.dry_run:
            print(json.dumps(dict(command=command,RAW_contents_read=False,output_created=False,GPU_started=False),ensure_ascii=False,indent=2));return 0
        return subprocess.call(command,cwd=ROOT)
    except (ValueError,OSError,json.JSONDecodeError) as exc:
        p.error(str(exc))


if __name__=='__main__':raise SystemExit(main())
