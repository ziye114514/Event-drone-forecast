"""One fixed mechanism test: visibility transitions and lumped force recovery."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from five_circle_model import FiveCircleModel
from lumped_disturbance import LumpedDisturbance,forecasts


def sequence():
    layout=np.array([[-50,-35],[50,-35],[50,35],[-50,35]],float)
    for i in range(60):
        t=i*.02;theta=.15*np.sin(i*.3);A=np.array([[1.,.1*np.sin(i*.3)],[0.,.7+.08*np.cos(i*.3)]])
        points=layout@A.T+[640,360]
        n=4 if i<15 or i>=45 else (3 if i<30 else (2 if i<40 else 0))
        rotors=[]
        for j in range(n):
            if n==2:
                point=np.array([580+120*j,360.]);major,minor=40.,3.
            else:point=points[j];major,minor=24.,24.*A[1,1]
            rotors.append(dict(cx=float(point[0]),cy=float(point[1]),major=major,minor=minor,
                angle=0.,ratio=minor/major,mode='line' if n==2 else 'ellipse',
                current_observation=True,inferred_shape=False,visual_track_id=j))
        geometry=dict(observed_rotors=rotors,state='four' if n==4 else 'insufficient',confidence=.8,
            axis_xy=[0.,-1.],two_axis=dict(valid=n==2,normal_candidates=[[0.,1.],[0.,-1.]]))
        yield t,dict(detected=True,center=[640.,360.],body_track_id=1),geometry


def run(output):
    out=Path(output);out.mkdir(exist_ok=False,parents=True)
    arms={i:FiveCircleModel(i) for i in (1,2,3)};rows=[]
    for t,b,g in sequence():rows.append(dict(t=t,results={str(i):a.step(t,b,g) for i,a in arms.items()}))
    three=[r['results']['3'] for r in rows if r['results']['3']['observed_count']==3]
    assert all(r['mode']=='three' and len(r['inferred_model_ids'])==1 for r in three)
    assert max(np.linalg.norm(np.asarray(r['current_geometric_center'])-[640,360]) for r in three)<1e-3
    side=[r['results']['3'] for r in rows if r['results']['3']['observed_count']==2]
    assert all(r['mode']=='edge_on' and r['current_geometric_center'] is None and r['axis_unsigned'] for r in side)
    assert rows[45]['results']['3']['mode']=='four' and rows[45]['results']['3']['recent_model_reused']
    assert rows[46]['results']['3']['mode']=='four'
    assert all(not r['results']['3']['inferred_rotor_frequencies'] for r in rows)
    empty=[r['results']['3'] for r in rows if r['results']['3']['observed_count']==0]
    assert all(not r['joint_center_is_current_measurement'] for r in empty)
    model=FiveCircleModel(3);t,b,g=next(sequence());model.step(t,b,g)
    try:model.step(t,b,g);raise AssertionError('noncausal cutoff accepted')
    except ValueError:pass
    # Pixel units here are replaced by metric units with an explicit noise
    # setting. This checks the estimator mechanism, not the event-camera model.
    rng=np.random.default_rng(42);observer=LumpedDisturbance(.005,.5)
    p=np.zeros(2);v=np.zeros(2);d=np.array([.6,-9.81]);u=np.array([0.,9.81]);force_rows=[]
    for i in range(301):
        t=i*.02
        if i:p+=.02*v+.5*.02**2*(u+d);v+=.02*(u+d)
        c,audit=observer.step(t,p+rng.normal(0,.005,2),u)
        if i>200:force_rows.append(dict(t=t,estimated=audit['disturbance_acceleration'],truth=d.tolist(),
            future_error=float(np.linalg.norm(np.asarray(forecasts(c)['0.4'])-(p+.4*v+.5*.4**2*(u+d))))))
    force_mae=float(np.mean([np.linalg.norm(np.asarray(r['estimated'])-d) for r in force_rows]))
    assert force_mae<.25,force_mae
    result=dict(structure_transition_controls_passed=True,synthetic_rows=len(rows),
        synthetic_disturbance_mean_vector_error_m_s2=force_mae,
        synthetic_prediction_mean04s_m=float(np.mean([r['future_error'] for r in force_rows])),
        actual_RAW=False,independent_real_force_truth=False,parameters_swept=False)
    (out/'controls.json').write_text(json.dumps(result,indent=2),'utf-8')
    (out/'synthetic_structure.json').write_text(json.dumps(rows),'utf-8')
    (out/'synthetic_disturbance.json').write_text(json.dumps(force_rows),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
