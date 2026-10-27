"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""
import numpy as np
import frequency_features as native79
from event_drone_forecast.geometry import _line

PARAMS=dict(native_configuration='strict7',frequency_range_hz=[60.,1200.],minimum_body_confidence=.45,
    selection='Unchanged79 current20ms unique-pixel votes,1.02 log bins,>=16 pixels,mode fraction>=.35,ring-density<.5 ROI density.',
    axis='Unchanged geometry._line: extent>=12px, width/extent<=.14, extent<=sensor_width/2. Align latest pixel timestamps to current time with current body-CV velocity.',
    interpretation='Periodic-support unsigned image line/normal candidate; not rotor identity, calibrated 3D attitude, mechanical RPM or thrust. Not used by predictor140.')

class NativeAxis:
    def __init__(self,w,h):self.w=w;self.h=h;self.algo=native79.engine(w,h,'strict7');self.output=self.algo.get_empty_output_buffer();self.last_t=None
    def feed(self,e,t):
        assert self.last_t is None or t>self.last_t;self.last_t=t
        assert not len(e) or e['t'].max()<round(t*1e6)
        return native79.checked_process(self.algo,e,self.output,self.w,self.h)
    def measure(self,q,body,t,velocity,snapshot_t=None):
        # Paced replay can measure an immutable q/t packet while its producer
        # continues feeding the SDK engine. This branch never reads that engine.
        assert (self.last_t if snapshot_t is None else snapshot_t)==t
        return measure_current(q,body,t,velocity,self.w,self.h)

def measure_current(q,body,t,velocity,w,h):
        """Stateless measurement of an already computed current SDK packet.

        Shares the same algorithm as NativeAxis.measure without a second SDK
        engine or copying events; caller owns q/t and preserves its cutoff.
        """
        bbox=body.get('bbox') if body.get('detected') and body.get('confidence',0)>=PARAMS['minimum_body_confidence'] else None
        summary=native79.summarize(q,bbox,t,w,h);summary.update(line=None,axis_valid=False,mechanical_RPM_estimated=False,thrust_estimated=False,forecast_uses_this_axis=False)
        points=np.empty((0,2),float)
        if not summary['qualified']:return summary,points
        q=q[(q['t']>=round(t*1e6)-20000)&(q['frequency']>=60)&(q['frequency']<=1200)]
        pixel=q['y'].astype(np.int64)*w+q['x'];_,idx=np.unique(pixel[::-1],return_index=True);q=q[len(q)-1-idx]
        xy=np.c_[q['x'],q['y']];b=np.asarray(bbox);inside=np.all((xy>=b[:2])&(xy<b[2:]),axis=1)
        matching=np.abs(np.log(q['frequency']/summary['frequency_hz']))<=np.log(1.02)*1.5;q=q[inside&matching]
        age=t-q['t']/1e6;points=np.c_[q['x'],q['y']]+age[:,None]*np.asarray(velocity)
        if np.any(points<0) or np.any(points>=[w,h]):summary['axis_reason']='aligned_support_outside_sensor';return summary,points
        line=_line(points,w*.5) if len(points)>=16 else None
        if line is not None:
            summary['line']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in line.items()};summary['axis_valid']=True;summary['axis_reason']='current_periodic_support_line'
            n=np.asarray(line['normal']);summary['normal_axis_candidates_xy']=[n.tolist(),(-n).tolist()]
        else:summary['axis_reason']='periodic_support_not_one_thin_line'
        summary['selected_pixel_count']=len(points)
        return summary,points
