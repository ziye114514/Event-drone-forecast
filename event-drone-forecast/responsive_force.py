"""Fast recent lumped-residual correction with the existing slow gain learner.

Reuses the causal0.2s integrated-input rows from FastForceResponse. Only the
disturbance time scale changes; body p/v and event input are unchanged. This
does not turn image periodicity into calibrated thrust or infer hidden RPM.
"""
import numpy as np
from collections import deque
from fast_force_response import FastForceResponse


class ResponsiveForce(FastForceResponse):
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)

    def step(self,t,c,observed_position,u,quality=0.):
        previous_t=self.last_t
        predictions,info=super().step(t,c,observed_position,u,quality)
        dt=0. if previous_t is None or t-previous_t>.5 else t-previous_t
        update=bool(c is not None and observed_position is not None and self.rows and
            t-self.rows[-1][0]<=.12 and self.d is not None)
        info['recent_disturbance_updated']=False
        if update:
            #0.2s is the existing causal integration horizon, not a label-tuned
            # best window.0.1s is the existing position smoothing time scale.
            age=np.array([t-r[0] for r in self.rows]);weights=np.exp(-age/.2);weights/=weights.sum()
            U=np.array([r[1] for r in self.rows]);R=np.array([r[2] for r in self.rows])
            target=np.sum(weights[:,None]*(R-self.gain*U),axis=0)
            self.d+=(1-np.exp(-dt/.1))*np.clip(target-self.d,-3000.,3000.)
            info.update(recent_disturbance_updated=True,residual_time_scale_s=.2,correction_time_scale_s=.1,
                correction_uses_only_matured_positions=True)
        active=info['active'];held=info['short_gap_acceleration_held']
        if active:
            a=self.gain*np.asarray(u)+self.d
            norm=np.linalg.norm(a)
            if norm>3000.:a*=3000./norm
            self.last_acceleration=a.copy()
        elif held:a=self.last_acceleration.copy()
        else:a=np.asarray(c['acceleration']) if c is not None else np.zeros(2)
        if c is not None:
            p,v=np.asarray(c['position']),np.asarray(c['velocity'])
            predictions={str(h):(p+h*v+.5*h*h*a).tolist() for h in (.1,.2,.4)}
        info.update(acceleration_px_s2=a.tolist(),disturbance_px_s2=None if self.d is None else self.d.tolist(),
            disturbance_estimator='recent_exponential_integrated_position_residual')
        return predictions,info


class DampedResponsiveForce(ResponsiveForce):
    """Singer conditional-mean rollout of the estimated TOTAL acceleration.

    The flight command beyond the current instant is unobserved. A fixed0.35s
    correlation prior (already used by this project's jerk model) avoids
    asserting that noisy current acceleration persists unchanged for0.4s.
    Gravity itself is NOT modeled as decaying. This is a total net-acceleration
    prior, not a new measured drag force or a learned physical constant.
    Formula reference: Stone Soup Singer transition matrix; independently coded.
    """
    correlation_s=.35

    def step(self,t,c,observed_position,u,quality=0.):
        predictions,info=super().step(t,c,observed_position,u,quality)
        if c is not None:
            p,v=np.asarray(c['position']),np.asarray(c['velocity']);a=np.asarray(info['acceleration_px_s2'])
            tau=self.correlation_s
            predictions={str(h):(p+h*v+tau*(h+tau*np.expm1(-h/tau))*a).tolist() for h in (.1,.2,.4)}
        info.update(future_net_acceleration_prior='Singer_zero_mean_conditional_expectation',
            acceleration_correlation_s=self.correlation_s,future_commands_observed=False)
        return predictions,info


class CalibratedResponsiveForce(DampedResponsiveForce):
    """Causal0.4s response calibration from matured OWN body observations.

    Calibrates the image forecast response, not mass/true thrust. The current
    force estimator is preserved; a scalar conservative rollout reliability
    is learned with the existing6px position-noise prior and0.35s time scale.
    This is one physical rollout plus calibration, not another kinematic filter.
    """
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.calibration_pending=deque();self.calibration_rows=deque();self.calibration_previous=None
        self.calibration_origin=None;self.response_reliability=1.

    def step(self,t,c,observed_position,u,quality=0.):
        predictions,info=super().step(t,c,observed_position,u,quality)
        updated=False
        if observed_position is not None:
            z=np.asarray(observed_position,float)
            while self.calibration_pending and self.calibration_pending[0]['target']<=t:
                record=self.calibration_pending.popleft();target=record['target']
                previous=self.calibration_previous
                if previous is None or not previous[0]<=target<=t or t-previous[0]>.15:continue
                alpha=(target-previous[0])/max(t-previous[0],1e-9)
                observed=(1-alpha)*previous[1]+alpha*z
                delta=record['delta'];residual=observed-record['base']
                if delta@delta>=36.:
                    self.calibration_rows.append((t,delta,residual));updated=True
            self.calibration_previous=(t,z.copy())
        while self.calibration_rows and t-self.calibration_rows[0][0]>3.:self.calibration_rows.popleft()
        if updated:
            energy=36.;correlation=36.  #6px prior, reliability centered at1.
            for stamp,delta,residual in self.calibration_rows:
                weight=np.exp(-(t-stamp)/.35)
                energy+=weight*float(delta@delta);correlation+=weight*float(delta@residual)
            self.response_reliability=float(np.clip(correlation/energy,0.,1.))
        if c is not None:
            p,v=np.asarray(c['position']),np.asarray(c['velocity'])
            uncalibrated=predictions
            if observed_position is not None and (self.calibration_origin is None or t-self.calibration_origin>=.099):
                h=.4;base=p+h*v;delta=np.asarray(uncalibrated[str(h)])-base
                self.calibration_pending.append(dict(target=t+h,base=base.copy(),delta=delta.copy()))
                self.calibration_origin=t
            predictions={str(h):(p+h*v+self.response_reliability*(np.asarray(uncalibrated[str(h)])-p-h*v)).tolist() for h in (.1,.2,.4)}
        info.update(rollout_response_reliability=self.response_reliability,response_calibration_updated=updated,
            response_calibration_matured_rows=len(self.calibration_rows),response_calibration_target_s=.4,
            calibration_reference='matured own position observations, never annotation or future unavailable data',
            calibration_is_not_true_force_identification=True)
        return predictions,info


class InnovationResidualForce(FastForceResponse):
    """Unknown acceleration follows current position-filter velocity innovation.

    Same observed p/v state; no second trajectory filter. The integral0.2s
    rows still identify the slow gain. Short-term d is corrected from the
    change in observed velocity, removing the extra integration-maturity lag.
    """
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.velocity_reference=None

    def step(self,t,c,observed_position,u,quality=0.):
        predictions,info=super().step(t,c,observed_position,u,quality)
        updated=False
        current_good=bool(c is not None and observed_position is not None and info['active'])
        previous=self.velocity_reference
        if current_good:
            v=np.asarray(c['velocity']);current_u=np.asarray(u)
            if previous is not None and 0.<t-previous[0]<=.12:
                dt=t-previous[0]
                acceleration_observation=(v-previous[1])/dt
                # The input integrated over this already observed interval.
                midpoint_u=.5*(previous[2]+current_u)
                target=acceleration_observation-self.gain*midpoint_u
                self.d+=(1-np.exp(-dt/.1))*np.clip(target-self.d,-3000.,3000.)
                updated=True
            self.velocity_reference=(t,v.copy(),current_u.copy())
        elif previous is not None and t-previous[0]>.12:self.velocity_reference=None
        if info['active']:
            a=self.gain*np.asarray(u)+self.d
            norm=np.linalg.norm(a)
            if norm>3000.:a*=3000./norm
            self.last_acceleration=a.copy()
        elif info['short_gap_acceleration_held']:a=self.last_acceleration.copy()
        else:a=np.asarray(c['acceleration']) if c is not None else np.zeros(2)
        if c is not None:
            p,v=np.asarray(c['position']),np.asarray(c['velocity'])
            predictions={str(h):(p+h*v+.5*h*h*a).tolist() for h in (.1,.2,.4)}
        info.update(acceleration_px_s2=a.tolist(),disturbance_px_s2=None if self.d is None else self.d.tolist(),
            current_velocity_residual_updated=updated,residual_smoothing_s=.1,
            disturbance_estimator='current_position_filter_velocity_innovation_minus_interval_force',
            current_velocity_is_not_independent_force_truth=True)
        return predictions,info


class WindowResidualForce(FastForceResponse):
    """Fit unknown net acceleration after integrating the observed force input.

    Jointly eliminate nuisance initial position/velocity rather than treating
    the lagging KF velocity as exact. A0.35s bounded window uses the existing
    model time scale. Only d updates; the trusted p/v state stays unchanged.
    """
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        super().__init__(default_gain,early_disturbance,hold_acceleration_gaps,gain_signal_to_noise)
        self.fit_observations=deque()

    def step(self,t,c,observed_position,u,quality=0.):
        predictions,info=super().step(t,c,observed_position,u,quality)
        updated=False;fit_diagnostic={}
        if observed_position is not None and info['active']:
            self.fit_observations.append((t,np.asarray(observed_position,float),np.asarray(u,float)))
        while self.fit_observations and t-self.fit_observations[0][0]>.35:self.fit_observations.popleft()
        if info['active'] and len(self.fit_observations)>=7:
            tt=np.array([r[0] for r in self.fit_observations]);span=tt[-1]-tt[0]
            if span>=.2 and np.max(np.diff(tt))<=.07:
                zz=np.array([r[1] for r in self.fit_observations]);uu=np.array([r[2] for r in self.fit_observations])
                integral=np.zeros_like(uu);velocity=np.zeros(2)
                for i,dt in enumerate(np.diff(tt),1):
                    integral[i]=integral[i-1]+dt*velocity+dt*dt*(uu[i-1]/3+uu[i]/6)
                    velocity+=.5*dt*(uu[i-1]+uu[i])
                scaled=(tt-tt[-1])/span;A=np.c_[np.ones(len(tt)),scaled,.5*scaled**2]
                residual_positions=zz-self.gain*integral
                beta=np.linalg.lstsq(A,residual_positions,rcond=None)[0]
                residual=np.linalg.norm(A@beta-residual_positions,axis=1)
                weights=np.minimum(1.,6./np.maximum(residual,1e-9))
                beta=np.linalg.lstsq(A*np.sqrt(weights)[:,None],residual_positions*np.sqrt(weights)[:,None],rcond=None)[0]
                rms=float(np.sqrt(np.mean((A@beta-residual_positions)**2)))
                if rms<=12.:
                    self.d=beta[2]/(span*span);updated=True
                fit_diagnostic=dict(samples=len(tt),span_s=float(span),rms_px=rms,
                    initial_position_velocity_are_nuisance_fit_only=True,input_double_integral_used=True)
        if info['active']:
            a=self.gain*np.asarray(u)+self.d
            norm=np.linalg.norm(a)
            if norm>3000.:a*=3000./norm
            self.last_acceleration=a.copy()
        elif info['short_gap_acceleration_held']:a=self.last_acceleration.copy()
        else:a=np.asarray(c['acceleration']) if c is not None else np.zeros(2)
        if c is not None:
            p,v=np.asarray(c['position']),np.asarray(c['velocity'])
            predictions={str(h):(p+h*v+.5*h*h*a).tolist() for h in (.1,.2,.4)}
        info.update(acceleration_px_s2=a.tolist(),disturbance_px_s2=None if self.d is None else self.d.tolist(),
            local_force_balance_updated=updated,local_force_balance=fit_diagnostic,
            disturbance_estimator='local_integrated_force_balance_with_free_initial_position_velocity',
            nuisance_velocity_is_not_used_as_a_parallel_position_forecaster=True)
        return predictions,info
