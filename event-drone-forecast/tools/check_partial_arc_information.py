"""Normal constraints must add current information without extra rotor detections."""
from pathlib import Path
import sys,json,copy
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from check_three_continuation import initialized
from partial_arc_information import update


def run(out):
    m,b,g=initialized();g['observed_rotors']=g['observed_rotors'][:2]
    m.step(.08,b,g);initial=m._hubs().copy();last=m.last_informative
    offset=np.array([540,260]);mask=np.zeros((200,200),np.uint8)
    truth=initial.copy();truth[2:,0]+=2.
    for j,(p,S) in enumerate(zip(truth,m.shapes)):
        values,vectors=np.linalg.eigh(S)
        cv2.ellipse(mask,tuple(np.rint(p-offset).astype(int)),
            tuple(np.rint(np.sqrt(values[::-1])).astype(int)),
            float(np.rad2deg(np.arctan2(vectors[1,1],vectors[0,1]))),0,360,255,-1)
    old=copy.deepcopy(m)
    result=update(m,.08,b,mask,offset,g['observed_rotors'])
    assert result['updated'],result
    before=float(np.linalg.norm(initial[2:]-truth[2:]));after=float(np.linalg.norm(m._hubs()[2:]-truth[2:]))
    assert after<before,(before,after)
    assert m.last_informative==last and result['new_detected_rotors']==0
    assert np.linalg.eigvalsh(m.P).min()>0
    for name,test in [('empty',np.zeros_like(mask)),('solid',np.full_like(mask,255))]:
        model=copy.deepcopy(old);state=model.beta.copy();r=update(model,.08,b,test,offset,g['observed_rotors'])
        assert not r['updated'] and np.array_equal(model.beta,state),(name,r)
    model=copy.deepcopy(old)
    assert not update(model,.08,b,mask,offset,g['observed_rotors'][:1])['updated']
    assert not update(model,.3,b,mask,offset,g['observed_rotors'])['updated']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(synthetic_only=True,normal_constraints=result,before_hidden_hub_error_px=before,
        after_hidden_hub_error_px=after,no_current_rotors_invented=True,no_freshness_refresh=True,
        empty_and_solid_rejected=True,one_anchor_and_expired_template_rejected=True)
    (out/'partial_arc_controls.json').write_text(json.dumps(result,indent=2),'utf-8')
    print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
