"""Run the included real algorithm controls without camera/SDK/CUDA/weights."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'tools')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('Use a new output directory')
    out.mkdir(parents=True)
    from check_three_continuation import run as structure
    from check_shared_phase_fft import run as phase
    from check_force_gap_hold import run as force
    structure(out/'structure');phase(out/'phase');force(out/'force')
    result=dict(synthetic_only=True,real_flight_accuracy_proven=False,proprietary_SDK_or_GPU_used=False,
        structure=json.loads((out/'structure/continuation_checks.json').read_text('utf-8')),
        phase=json.loads((out/'phase/controls.json').read_text('utf-8')),
        force=json.loads((out/'force/force_gap_controls.json').read_text('utf-8')))
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print('Synthetic controls passed:',out/'summary.json')


if __name__=='__main__':main()
