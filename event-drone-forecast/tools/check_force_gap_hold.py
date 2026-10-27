"""A held total acceleration has a finite age and never counts as current thrust."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fast_force_response import FastForceResponse


def run(output):
    m=FastForceResponse();c=dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.])
    m.step(0.,c,None,[0.,-1.],.8)
    _,a=m.step(.033333,c,None,[.5,-1.],.8)
    acceleration=a['acceleration_px_s2'];last=m.last_valid
    for t in [.066666,.1,.133333]:
        _,info=m.step(t,c,None,None,0.)
        assert info['short_gap_acceleration_held'] and not info['active']
        assert info['acceleration_px_s2']==acceleration and m.last_valid==last
    _,expired=m.step(.166666,c,None,None,0.)
    assert not expired['short_gap_acceleration_held'] and expired['acceleration_px_s2']==[0.,0.]
    _,relocked=m.step(.2,c,None,[.5,-1.],.8)
    assert relocked['active'] and not relocked['short_gap_acceleration_held']
    assert not m.rows and not m.learned
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    result=dict(short_gap_hold_ttl_s=.12,held_does_not_refresh_age_or_count_as_current_thrust=True,
        timeout_falls_back_without_new_physical_measurement=True,relock_supported=True,
        missing_positions_do_not_train_gain_or_disturbance=True,synthetic_only=True)
    (out/'force_gap_controls.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
