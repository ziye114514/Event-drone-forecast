"""Counterfactual guards for reanchor velocity and unmeasured held amplitude."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared_image_motion import SharedImageMotion
from fast_force_response import FastForceResponse


def observation(change=None,t=.033333):
    m=SharedImageMotion();old=dict(detected=True,low_confidence=True,confidence=.3,bbox=[100.,100.,300.,200.],center=[200.,150.],nn_observed_t=0.)
    c=dict(position=[230.,150.],velocity=[1000.,0.],acceleration=[0.,0.]);P=np.eye(2)*1e6
    m.step(0.,c,P,old,dict(reason='no_previous_ROI'))
    current=dict(old,low_confidence=False,confidence=.8,bbox=[130.,100.,330.,200.],center=[230.,150.],
        nn_observed_t=.02,measurement_source='current_KLT_from_delayed_NN')
    if change:current.update(change)
    track=dict(reason='current_image_translation',points=12,delta=[10.,0.],displacement_covariance_px2=np.eye(2).tolist())
    return m.step(t,c,P,current,track),c


def main(output):
    (new,info),old=observation();assert info['observation']['anchor_upgrade']
    assert np.allclose(new['velocity'],[10/.033333,0.]) and new['position']==old['position']
    cases=[dict(low_confidence=True,confidence=.4),dict(nn_observed_t=0.),dict(bbox=[800.,100.,1000.,200.],center=[900.,150.])]
    for change in cases:assert not observation(change)[0][1]['velocity']['applied']
    assert not observation(t=.1)[0][1]['velocity']['applied']
    def expansion(weak=False,stamp=.02,large=False):
        state=SharedImageMotion();old=dict(detected=True,low_confidence=weak,confidence=.3 if weak else .6,
            bbox=[100.,100.,180.,160.],center=[140.,130.],nn_observed_t=0.)
        c=dict(position=[200.,130.],velocity=[1000.,0.],acceleration=[0.,0.]);P=np.eye(2)*1e6
        state.step(0.,c,P,old,dict(reason='no_previous_ROI'))
        current=dict(old,low_confidence=False,confidence=.8,bbox=[110.,98.,290.,162.],center=[200.,130.],
            nn_observed_t=stamp,measurement_source='current_KLT_from_delayed_NN')
        if large:current.update(bbox=[1.,1.,600.,600.],center=[300.5,300.5])
        return state.step(.033333,c,P,current,dict(reason='current_image_translation',points=12,delta=[10.,0.],displacement_covariance_px2=np.eye(2).tolist()))
    c,info=expansion();assert info['observation']['visibility_reanchor'] and np.allclose(c['velocity'],[10/.033333,0.])
    assert not expansion(weak=True)[1]['velocity']['applied']
    assert not expansion(stamp=0.)[1]['velocity']['applied']
    assert not expansion(large=True)[1]['velocity']['applied']
    m=FastForceResponse();c=dict(position=[0.,0.],velocity=[0.,0.],acceleration=[0.,0.])
    m.step(0.,c,[0.,0.],np.array([0.,-1.]),.8)
    for t in [.033333,.066666,.099999]:
        m.input_learning_current=False;m.step(t,c,[0.,0.],np.array([.1,-.99]),.35)
        assert m.last_valid==0. and m.inputs[-1][1] is None
    _,last=m.step(.133332,c,[0.,0.],None,0.)
    assert not last['short_gap_acceleration_held'] and not m.learned
    result=dict(strong_current_anchor_uses_track_velocity=True,position_unchanged=True,
        weak_current_and_old_NN_and_far_anchor_rejected=True,long_motion_gap_rejected=True,
        contained_strong_partial_box_reanchor_uses_measured_velocity=True,
        containment_does_not_override_weak_old_box_or_unbounded_box_growth=True,
        held_amplitude_not_input_integral_or_gain_evidence=True,held_amplitude_does_not_refresh_expiry=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':main(sys.argv[1])
