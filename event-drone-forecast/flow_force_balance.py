"""Causal lumped-force residual from already measured event-image velocity.

Subtract the known input's velocity integral over the SAME observation times,
then estimate the residual slope. Do not subtract today's input from a past
window's average acceleration. Gain remains the existing slow learner.
"""
from collections import deque
import numpy as np
from responsive_force import DampedResponsiveForce
from live_flow_acceleration import fit_derivative


def input_integrals(velocities,inputs):
    if len(velocities)<6:return None,dict(reason='few_current_flow_velocities')
    stamps=np.array([r[0] for r in velocities]);tt=np.array([r[0] for r in inputs])
    lo=int(np.searchsorted(tt,stamps[0],side='right'))-1
    hi=int(np.searchsorted(tt,stamps[-1],side='left'))+1
    if lo<0 or hi>len(inputs):return None,dict(reason='known_input_does_not_cover_velocity_window')
    data=list(inputs)[lo:hi];tt=np.array([r[0] for r in data])
    if any(r[1] is None for r in data) or len(tt)<2 or np.max(np.diff(tt))>.07:
        return None,dict(reason='unobserved_input_in_flow_balance_window')
    uu=np.array([r[1] for r in data]);intervals=np.diff(tt)
    integral=np.vstack((np.zeros(2),np.cumsum(.5*(uu[:-1]+uu[1:])*intervals[:,None],axis=0)))
    indices=np.searchsorted(tt,stamps,side='right')-1;indices=np.minimum(indices,len(tt)-2)
    dt=stamps-tt[indices];slopes=(uu[indices+1]-uu[indices])/intervals[indices,None]
    known=integral[indices]+dt[:,None]*uu[indices]+.5*dt[:,None]**2*slopes
    return known,dict(reason='same_window_integrated_input_removed',velocity_rows=len(velocities),
        input_rows=len(data),span_s=float(np.ptp(stamps)),
        current_input_not_subtracted_from_past_average=True)


def residual_history(velocities,inputs,gain):
    known,info=input_integrals(velocities,inputs)
    if known is None:return None,info
    return [(r[0],r[1]-gain*k,*r[2:]) for r,k in zip(velocities,known)],dict(info,known_force_gain=gain)


class FlowBalancedForce(DampedResponsiveForce):
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.flow_velocities=deque();self.balance_inputs=deque();self.flow_observation=None

    def observe_flow(self,observation,qualified_total_acceleration):
        self.flow_observation=(observation,qualified_total_acceleration)

    def step(self,t,c,observed_position,u,quality=0.):
        previous=self.last_t;incoming=self.flow_observation;self.flow_observation=None
        predictions,info=super().step(t,c,observed_position,u,quality)
        self.balance_inputs.append((t,None if not info['active'] or not self.input_learning_current else np.asarray(u,float)))
        obs,qualified=(None,False) if incoming is None else incoming
        if obs is not None and obs.get('valid'):
            R=np.asarray(obs['velocity_covariance']);W=np.linalg.inv(np.linalg.cholesky(R))
            self.flow_velocities.append((obs['midpoint_t'],np.asarray(obs['velocity']),R,W))
        while self.balance_inputs and t-self.balance_inputs[0][0]>.5:self.balance_inputs.popleft()
        while self.flow_velocities and t-self.flow_velocities[0][0]>.35+1e-8:self.flow_velocities.popleft()
        diagnostic=dict(updated=False,reason='no_current_qualified_total_acceleration')
        if qualified and info['active'] and self.d is not None:
            rows,diagnostic=residual_history(self.flow_velocities,self.balance_inputs,self.gain)
            if rows is not None:
                target,fit=fit_derivative(rows,t);diagnostic.update(fit=fit)
                if target is not None:
                    dt=0. if previous is None else min(.12,t-previous)
                    self.d+=(1-np.exp(-dt/.1))*np.clip(target-self.d,-3000.,3000.)
                    a=self.gain*np.asarray(u)+self.d
                    magnitude=np.linalg.norm(a)
                    if magnitude>3000.:a*=3000./magnitude
                    self.last_acceleration=a.copy();info['acceleration_px_s2']=a.tolist()
                    info['disturbance_px_s2']=self.d.tolist()
                    p,v=np.asarray(c['position']),np.asarray(c['velocity']);tau=self.correlation_s
                    predictions={str(h):(p+h*v+tau*(h+tau*np.expm1(-h/tau))*a).tolist() for h in (.1,.2,.4)}
                    diagnostic.update(updated=True,reason='same_window_flow_residual_disturbance',
                        correction_time_scale_s=.1,estimated_disturbance_px_s2=target.tolist())
        info['flow_force_balance']=diagnostic
        return predictions,info
