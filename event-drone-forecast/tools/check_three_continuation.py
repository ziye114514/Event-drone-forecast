"""Three current centres may refresh guidance; rank-deficient/feedback-only data may not."""
from pathlib import Path
import sys,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from continuous_five_circle import ContinuousFiveCircle
from check_five_circle_v3 import sequence


def initialized():
    seq=list(sequence());m=ContinuousFiveCircle()
    for row in seq[:4]:m.step(*row)
    _,b,g=seq[4];b=dict(b,confidence=.8);g=copy.deepcopy(g)
    g.update(state='insufficient',confidence=0.);g['observed_rotors']=g['observed_rotors'][:3]
    return m,b,g


def run(out):
    m,b,g=initialized();g['observed_rotors'][2]['shared_arc_recovered']=True
    refs=[]
    for i in range(25):
        t=.08+i*.02;r=m.step(t,b,g);ref=m.guidance_reference(t+.001,.12)
        assert r['mode']=='three' and r['continuous'].get('guidance_reference_refreshed')
        refs.append(ref)
    assert t-m.last_full>.12 and refs[-1]['eligible'],refs[-1]
    last=m.last_informative
    two=copy.deepcopy(g);two['observed_rotors']=two['observed_rotors'][:2]
    for i in range(1,9):
        z=m.step(t+i*.02,b,two);m.observe_axis(z,[0.,1.],.8,t+i*.02)
    assert m.last_informative==last and not m.guidance_reference(t+.16,.12)['eligible']
    feedback,b2,g2=initialized()
    for rr in g2['observed_rotors']:rr['shared_arc_recovered']=True
    for i in range(12):feedback.step(.08+i*.02,b2,g2)
    assert feedback.last_informative==feedback.last_full
    assert not feedback.guidance_reference(.32,.12)['eligible']
    m,b,g=initialized();m.P*=10000.
    assert m.guidance_reference(m.t+.001,.12)['reason']=='hub_uncertainty_too_large'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(three_refreshes_beyond_full_expiry=True,two_and_axis_do_not_refresh=True,
        all_guided_centres_do_not_refresh=True,uncertain_hidden_hub_rejected=True,
        interval_unchanged_s=.12,synthetic_only=True,last_qualified_reference=refs[-1])
    (out/'continuation_checks.json').write_text(json.dumps(result,indent=2),'utf-8');print(json.dumps(result))


if __name__=='__main__':run(sys.argv[1])
