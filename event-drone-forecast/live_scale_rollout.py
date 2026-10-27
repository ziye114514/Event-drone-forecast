"""Conditional image-scale correction; isolated from the default forecaster.

With fixed apparent shape and constant depth velocity, inverse image scale is
linear. Its rate k=-q'/q contributes .5*k*a*h^3/(1-k*h) beyond the existing
second-order displacement. Keep the current p/v/a and curved rollout intact.
Width/height agreement is only a compatibility check, not a proof of depth.
"""
from collections import deque
import numpy as np
from scipy.stats import theilslopes
from live_motion import LiveMotion,predictions

DESIGN=dict(window_s=.35,minimum_points=6,minimum_span_s=.16,
    slope_confidence=.95,max_current_size_residual_px=12.,maximum_abs_k_times_h=.5,
    current_state_and_validity_unchanged=True,constant_apparent_shape_assumed=True,
    slope_interval_is_not_calibrated_depth_confidence=True,
    parameters_swept=False,default_promoted=False)


def estimate_scale(history,t):
    if len(history)<6:return None,dict(reason='few_size_observations')
    x=np.array([r[0]-t for r in history]);size=np.array([r[1] for r in history])
    if np.ptp(x)<.16-1e-8:return None,dict(reason='short_size_span')
    rate=[];bounds=[];residuals=[]
    for dim in range(2):
        y=size[-1,dim]/size[:,dim]
        fit=theilslopes(y,x,alpha=.95,method='joint')
        if not np.isfinite(fit).all() or fit.intercept<=0:return None,dict(reason='invalid_inverse_size_fit')
        error=float(abs(fit.intercept-1)*size[-1,dim]);residuals.append(error)
        if error>12:return None,dict(reason='current_size_not_supported_by_history')
        rate.append(float(-fit.slope/fit.intercept))
        bounds.append([-float(fit.high_slope/fit.intercept),-float(fit.low_slope/fit.intercept)])
    low=max(b[0] for b in bounds);high=min(b[1] for b in bounds)
    diagnostic=dict(width_height_rates=rate,rate_intervals=bounds,current_size_residuals_px=residuals)
    if low>high:diagnostic['reason']='width_height_disagree';return None,diagnostic
    if low<=0<=high:diagnostic['reason']='common_scale_rate_not_signed';return None,diagnostic
    precision=1/np.maximum(np.ptp(bounds,axis=1)**2,1e-12)
    k=float(np.clip(np.average(rate,weights=precision),low,high))
    diagnostic.update(reason='compatible_signed_inverse_size_rate',common_interval=[low,high],rate=k)
    return k,diagnostic


class ScaleRolloutMotion(LiveMotion):
    def __init__(self,tracker_hz,noise_model='legacy'):
        super().__init__(tracker_hz,noise_model);self.sizes=deque()

    def step(self,t,body):
        c,valid,d=super().step(t,body)
        b=np.asarray(body.get('bbox',[]),float)
        good=bool(body.get('detected') and not body.get('low_confidence',False) and b.shape==(4,)
            and np.isfinite(b).all() and np.all(b[:2]>0) and np.all(b[2:]<[1279,719]) and np.all(b[2:]-b[:2]>=12))
        if good:self.sizes.append((t,b[2:]-b[:2]))
        else:self.sizes.clear()
        while self.sizes and t-self.sizes[0][0]>.35+1e-8:self.sizes.popleft()
        k=None;info=dict(reason='missing_current_or_history')
        if good and d['coverage']>=.9 and d['maximum_gap_s']<=2*self.period+1e-8:
            k,info=estimate_scale(self.sizes,t)
        if c is not None:c['conditional_scale_rate']=k
        d.update(scale_rollout=info)
        return c,valid,d


def scale_predictions(c,age=0.):
    preferred,cv=predictions(c,age)
    k=c.get('conditional_scale_rate') if c is not None else None
    if k is None:return preferred,cv
    a=np.asarray(c['acceleration'])
    for h in (.1,.2,.4):
        H=h+age
        if abs(k*H)>.5:continue
        correction=max(.5,1-.9*H*H)*.5*k*a*(H**3)/(1-k*H)
        preferred[str(h)]=(np.asarray(preferred[str(h)])+correction).tolist()
    return preferred,cv
