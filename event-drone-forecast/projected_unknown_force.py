"""Partial unknown-force feedback perpendicular to the observed thrust axis.

In the weak2D model a=T*n+d, a tangent annihilates the unknown thrust amplitude:
tangent@a=tangent@d. Update that one observable component, even when the past
period window is incomplete. Normal force remains unidentified. This is a
geometric projection, not metric gravity/wind measurement or a new ESO claim.
"""
from collections import deque
import numpy as np
from flow_force_balance import FlowBalancedForce
from axis_amplitude_bridge import current_axis_qualified


class ProjectedResidualForce(FlowBalancedForce):
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.axes=deque();self.axis_motion=None

    def observe_axis_motion(self,structure,motion):self.axis_motion=(structure,motion)

    def step(self,t,c,observed_position,u,quality=0.):
        previous=self.last_t;incoming=self.axis_motion;self.axis_motion=None
        predictions,info=super().step(t,c,observed_position,u,quality)
        diagnostic=dict(updated=False,reason='no_current_axis_and_motion',normal_unknown_force_not_identified=True)
        if incoming is None:info['projected_unknown_force']=diagnostic;return predictions,info
        structure,motion=incoming;n=np.asarray(structure.get('axis_xy') if structure.get('axis_xy') is not None else [],float)
        current=bool(current_axis_qualified(structure) and n.shape==(2,) and np.isfinite(n).all() and np.linalg.norm(n)>1e-8)
        if current:n=n/np.linalg.norm(n);self.axes.append((t,n.copy()))
        while self.axes and t-self.axes[0][0]>.35+1e-8:self.axes.popleft()
        acceleration=(motion or {}).get('acceleration') or {}
        ready=bool(current and acceleration.get('applied') and self.d is not None and len(self.axes)>=6 and self.axes[-1][0]-self.axes[0][0]>=.16-1e-8)
        if ready and not info['flow_force_balance'].get('updated'):
            times=np.array([r[0] for r in self.axes]);normals=np.array([r[1] for r in self.axes])
            spread=float(np.rad2deg(np.arccos(np.clip(abs(normals@n),0.,1.))).max())
            confidence=float((structure.get('axis_support') or {}).get('confidence',structure.get('geometric_weight',.45)))
            # Same5deg/confidence angular scale as ContinuousFiveCircle's
            # existing current-axis observation. No label-selected angle sweep.
            limit=5./max(.45,min(1.,confidence))
            coherent=bool(np.max(np.diff(times))<=.08+1e-8 and spread<=limit)
            diagnostic.update(axis_spread_deg=spread,axis_gate_deg=limit,axis_rows=len(times),
                reason='axis_changes_too_much_for_past_flow_derivative')
            if coherent:
                tangent=np.array([-n[1],n[0]]);observed=np.asarray(acceleration['shrunk_acceleration'])
                # The current flow slope already has a0.35s robust window.
                # Do not add another EWMA when it is more precise than the
                # matured-position residual. Scalar minimum-volume CI with
                # unknown correlation chooses the more precise endpoint.
                se=np.asarray(acceleration['standard_error_proxy'])
                flow_variance=float(np.sum(tangent*tangent*se*se))
                # Existing6px position prior and0.2s residual maturity. Rows
                # overlap, so their empirical scatter is NOT divided by N.
                floor=(np.sqrt(2.)*6./(.5*.2*.2))**2
                baseline_variance=floor
                if self.rows:
                    residuals=np.array([r[2]-self.gain*r[1] for r in self.rows])
                    baseline_variance+=float(np.mean(((residuals-self.d)@tangent)**2))
                diagnostic.update(flow_tangent_variance_proxy=flow_variance,baseline_tangent_variance_proxy=baseline_variance,
                    covariance_intersection='minimum_volume_scalar_endpoint',confidence_is_not_calibrated=True)
                if not np.isfinite(flow_variance) or flow_variance>=baseline_variance:
                    diagnostic['reason']='matured_tangent_estimate_more_precise'
                    info['projected_unknown_force']=diagnostic
                    return predictions,info
                error=float(tangent@(observed-self.d));correction=np.clip(error,-3000.,3000.)*tangent
                self.d+=correction;info['disturbance_px_s2']=self.d.tolist()
                applied=bool(info['active'] or info['short_gap_acceleration_held'])
                if applied:
                    a=self.gain*np.asarray(u)+self.d if info['active'] else np.asarray(info['acceleration_px_s2'])+correction
                    norm=np.linalg.norm(a)
                    if norm>3000.:a*=3000./norm
                    self.last_acceleration=a.copy();info['acceleration_px_s2']=a.tolist()
                    p,v=np.asarray(c['position']),np.asarray(c['velocity']);tau=self.correlation_s
                    predictions={str(h):(p+h*v+tau*(h+tau*np.expm1(-h/tau))*a).tolist() for h in (.1,.2,.4)}
                diagnostic.update(updated=True,reason='current_axis_tangent_unknown_force_feedback',
                    correction_px_s2=correction.tolist(),normal_component_of_correction=float(n@correction),
                    output_updated=applied,partial_information_rank=1,full_same_flow_balance_not_applied=True,
                    no_second_smoothing_after_qualified_flow_derivative=True,metric_force_estimated=False)
        elif info['flow_force_balance'].get('updated'):diagnostic['reason']='same_flow_already_used_by_full_balance'
        info['projected_unknown_force']=diagnostic
        return predictions,info
