"""A template alone or sparse clutter must not produce measured rotor centres."""
from pathlib import Path
import sys,json,copy
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from continuous_five_circle import ContinuousFiveCircle
from shared_arc_recovery import recover
from check_five_circle_v3 import sequence


def run(out):
    seq=list(sequence());model=ContinuousFiveCircle()
    for row in seq[:4]:model.step(*row)
    t,b,g=seq[4];b=dict(b,confidence=.8);initial=copy.deepcopy(g['observed_rotors'][:2])
    offset=np.array([540,260]);empty=np.zeros((200,200),np.uint8)
    result,info=recover(model,t,b,empty,offset,initial)
    assert info['accepted']==0 and len(result)==2
    mask=empty.copy()
    for r in g['observed_rotors']:
        center=np.array([r['cx'],r['cy']])-offset
        cv2.ellipse(mask,tuple(np.rint(center).astype(int)),(round(r['major']/2),round(r['minor']/2)),r['angle'],0,360,255,-1)
    recovered,positive=recover(model,t,b,mask,offset,initial)
    assert positive['accepted']>=1 and len(recovered)>=3,positive
    for detail in positive['details']:assert detail['current_area_union_iou']>=.72 and detail['coverage']>=.48
    rng=np.random.default_rng(148)
    clutter=np.uint8(rng.random(empty.shape)<.18)*255
    rejected,negative=recover(model,t,b,clutter,offset,initial)
    assert negative['accepted']==0 and len(rejected)==2,negative
    # Dense backgrounds have plenty of events but no isolated rotor boundary.
    dense=np.full_like(empty,255)
    _,solid=recover(model,t,b,dense,offset,initial)
    assert solid['accepted']==0,solid
    line=empty.copy();cv2.line(line,(0,100),(199,100),255,3)
    _,flat=recover(model,t,b,line,offset,initial)
    assert flat['accepted']==0,flat
    # A body/arm attached to a genuine disc can reduce union IoU. It must
    # still retain a current curved boundary and a relatively empty ring.
    attached=mask.copy()
    for r in g['observed_rotors'][2:]:
        x,y=np.rint(np.array([r['cx'],r['cy']])-offset).astype(int)
        cv2.rectangle(attached,(x+7,y-6),(x+30,y+6),255,-1)
    _,attachment=recover(model,t,b,attached,offset,initial)
    assert attachment['accepted']>=1,attachment
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(controlled_only=True,no_template_only_measurements=True,sparse_clutter_rejected=True,
        solid_foreground_rejected=True,flat_line_rejected=True,
        attached_body_current_arc_recovered=attachment['accepted'],
        attachment_details=attachment['details'],
        supported_current_ellipse_recovered=positive['accepted'],positive_details=positive['details'])
    (out/'arc_controls.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
