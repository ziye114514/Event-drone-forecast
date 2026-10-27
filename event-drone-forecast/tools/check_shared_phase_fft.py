"""Known spatial rotation must recover its blade phase; random timing must not."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from shared_phase_fft import SharedPhaseFFT


def run(output):
    centers=np.array([[300,230],[500,230],[500,400],[300,400]],float)
    rotors=[dict(cx=p[0],cy=p[1],major=120.,minor=40.,angle=0.,ratio=1/3,
        mode='ellipse',current_observation=True) for p in centers]
    body=dict(detected=True,bbox=[220,190,580,440])
    structure=dict(mode='four',model_centers=centers.tolist(),observed_model_ids=[0,1,2,3],geometry_epoch=1)
    dtype=[('x','u2'),('y','u2'),('p','u2'),('t','i8')]
    models={name:SharedPhaseFFT() for name in ['rotating','random']};last={};rng=np.random.default_rng(27)
    for tick in range(1,22):
        t=tick/30;ts=np.arange(round((tick-1)*1e6/30),round(t*1e6),100)
        for name,model in models.items():
            parts=[]
            for c in centers:
                for blade in range(2):
                    for p in range(2):
                        angle=2*np.pi*50*ts/1e6+blade*np.pi+p*.04 if name=='rotating' else rng.uniform(-np.pi,np.pi,len(ts))
                        e=np.zeros(len(ts),dtype=dtype);e['t']=ts;e['p']=p
                        e['x']=np.rint(c[0]+60*np.cos(angle));e['y']=np.rint(c[1]+20*np.sin(angle));parts.append(e)
            e=np.sort(np.concatenate(parts),order='t');last[name]=model.step(e,t,body,structure,rotors,[0,0])
    positive=last['rotating'];negative=last['random']
    assert positive['qualified_count']==4,positive
    assert max(abs(q['frequency_hz']-100.) for q in positive['rotors'])<1.5,positive
    assert negative['qualified_count']==0,negative
    three=dict(structure,mode='three',observed_model_ids=[0,1,2])
    z=models['rotating'].step(e,.734,body,three,rotors[:3],[0,0])
    assert z['rotors'][3]['reason']=='hidden_slot_not_measured'
    flat=dict(structure,mode='axis_only')
    z=models['rotating'].step(e,.768,body,flat,rotors,[0,0])
    assert not z['rotors'] and z['qualified_count']==0
    # Shared plane wobble around a fixed body centre must not change the known
    # rotational phase. This checks geometry decoupling, not flight dynamics.
    wobble=SharedPhaseFFT();center=np.array([400.,315.]);last_wobble=None
    for tick in range(1,22):
        t=tick/30;ts=np.arange(round((tick-1)*1e6/30),round(t*1e6),100)
        a=.12*np.sin(2*np.pi*2*t);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        A=R@np.diag([1.,1+.15*np.sin(2*np.pi*1.5*t)])
        hubs=(centers-center)@A.T+center;shape=A@np.diag([60.**2,20.**2])@A.T
        values,vectors=np.linalg.eigh(shape);axes=2*np.sqrt(values[::-1])
        angle=float(np.rad2deg(np.arctan2(vectors[1,1],vectors[0,1])))
        current=[dict(rotors[i],cx=h[0],cy=h[1],major=axes[0],minor=axes[1],angle=angle,ratio=axes[1]/axes[0]) for i,h in enumerate(hubs)]
        parts=[]
        for c in centers:
            for blade in range(2):
                for p in range(2):
                    theta=2*np.pi*50*ts/1e6+blade*np.pi+p*.04
                    base=c+np.c_[60*np.cos(theta),20*np.sin(theta)]
                    xy=(base-center)@A.T+center
                    ev=np.zeros(len(ts),dtype=dtype);ev['t']=ts;ev['p']=p;ev['x']=np.rint(xy[:,0]);ev['y']=np.rint(xy[:,1]);parts.append(ev)
        ev=np.sort(np.concatenate(parts),order='t')
        last_wobble=wobble.step(ev,t,dict(body,bbox=[190,140,610,490]),dict(structure,model_centers=hubs.tolist()),current,[0,0])
    assert last_wobble['qualified_count']==4,last_wobble
    assert max(abs(q['frequency_hz']-100) for q in last_wobble['rotors'])<1.5,last_wobble
    flicker=SharedPhaseFFT();last_flicker=None
    for tick in range(1,22):
        t=tick/30;start=(tick-1)/30;parts=[]
        flashes=np.arange(np.ceil(start*100)/100,t,1/100)
        for c in centers:
            # Nonuniform STATIC angular support. A bright sector dominates,
            # making temporal peaks/coherence alone an insufficient guard.
            angles=np.r_[np.zeros(24),np.arange(12)*2*np.pi/12]
            for flash in flashes:
                for p in range(2):
                    ts=np.full(len(angles),round(flash*1e6)+100+p*100)
                    keep=(ts>=round(start*1e6))&(ts<round(t*1e6));theta=angles[keep];ts=ts[keep]
                    ev=np.zeros(len(ts),dtype=dtype);ev['t']=ts;ev['p']=p
                    ev['x']=np.rint(c[0]+60*np.cos(theta));ev['y']=np.rint(c[1]+20*np.sin(theta));parts.append(ev)
        ev=np.sort(np.concatenate(parts),order='t')
        last_flicker=flicker.step(ev,t,body,structure,rotors,[0,0])
    assert last_flicker['qualified_count']==0,last_flicker
    old_accept=sum(q.get('phase_concentration',0)>=.5 and q.get('peak_to_noise',0)>=8
        and q.get('current_sectors',0)>=9 for q in last_flicker['rotors'])
    assert old_accept==4,last_flicker
    result=dict(synthetic_only=True,mechanical_truth_available_for_real_RAW=False,
        two_blade_phase_truth_hz=100.,qualified_positive=positive['qualified_count'],
        positive_frequencies=[q['frequency_hz'] for q in positive['rotors']],
        fixed_body_shared_plane_wobble_phase_frequencies=[q['frequency_hz'] for q in last_wobble['rotors']],
        wobble_test_is_geometry_only_not_force_equilibrium=True,
        static_common_flicker_with_dominant_bright_sector_rejected=True,
        old_nondirectional_gates_false_accept_count=int(old_accept),
        flicker_directionality=[q.get('signed_power_directionality') for q in last_flicker['rotors']],
        random_timing_rejected=True,hidden_slot_not_filled=True,flat_plane_not_rectified=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'controls.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
