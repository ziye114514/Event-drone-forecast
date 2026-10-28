"""Slow identifiable effective-gain learning, immediate current force response.

Image acceleration = k * relative_frequency_squared * unsigned_pose_prior + d.
Matured0.2s displacement supervises k,d; current input is used immediately.
This does not identify Newton thrust, mass, wind or a signed3D normal.
"""
from collections import deque
import numpy as np


class FastForceResponse:
    def __init__(self,default_gain=200.,early_disturbance=True,hold_acceleration_gaps=True,gain_signal_to_noise=0.):
        self.gain=float(default_gain);self.d=None;self.last_t=None;self.last_valid=None
        self.early_disturbance=early_disturbance
        self.hold_acceleration_gaps=hold_acceleration_gaps;self.last_acceleration=None
        self.gain_signal_to_noise=float(gain_signal_to_noise)
        self.inputs=deque();self.pending=deque();self.rows=deque();self.last_origin=None
        self.learned=False;self.gain_sigma=None;self.frozen_reason='not_enough_matured_positions'
        self.gain_learning_enabled=True
        self.input_learning_current=True

    def step(self,t,c,observed_position,u,quality=0.):
        dt=0. if self.last_t is None else t-self.last_t
        if dt<0:raise ValueError('Strict causal timestamps required')
        if self.last_t is not None and dt>.5:self.__init__(self.gain,self.early_disturbance,self.hold_acceleration_gaps,self.gain_signal_to_noise);dt=0.
        self.last_t=t
        good=u is not None and np.isfinite(u).all() and quality>=.35
        self.inputs.append((t,None if not good or not self.input_learning_current else np.asarray(u,float)))
        # Only positions observed at/after maturity supervise earlier predictions.
        if c is not None and observed_position is not None:
            p=np.asarray(observed_position,float)
            while self.pending and self.pending[0][0]+.2<=t+1e-8:
                t0,p0,v0=self.pending.popleft();H=t-t0
                data=[(s,U) for s,U in self.inputs if t0-1e-8<=s<=t+1e-8]
                if H>.25 or len(data)<3 or any(U is None for _,U in data):continue
                tt=np.array([s for s,_ in data]);UU=np.array([U for _,U in data])
                if tt[0]>t0+.04 or np.max(np.diff(tt))>.07:continue
                integrand=(t-tt)[:,None]*UU
                weighted=np.sum(.5*(integrand[1:]+integrand[:-1])*np.diff(tt)[:,None],axis=0)/(.5*H*H)
                residual=(p-p0-H*v0)/(.5*H*H)
                self.rows.append((t,weighted,residual))
            if self.last_origin is None or t-self.last_origin>=.099:
                self.pending.append((t,np.asarray(c['position']),np.asarray(c['velocity'])))
                self.last_origin=t
        while self.rows and t-self.rows[0][0]>3.:self.rows.popleft()
        while self.inputs and t-self.inputs[0][0]>.5:self.inputs.popleft()
        info=dict(matured_rows=len(self.rows),gain_learned=False,disturbance_updated=False)
        if self.rows:
            U=np.array([r[1] for r in self.rows]);R=np.array([r[2] for r in self.rows])
            mean_u=U.mean(0);mean_r=R.mean(0)
        if len(self.rows)>=8:
            du=U-mean_u;dr=R-mean_r
            energy=float(np.sum(du*du));excitation=float(np.sqrt(energy/len(U)))
            info['excitation_rms']=excitation
            self.frozen_reason='insufficient_input_excitation'
            if excitation>=.08 and self.gain_learning_enabled:
                k=float(np.sum(du*dr)/max(energy,1e-9));res=dr-k*du
                sigma=float(np.sqrt(np.sum(res*res)/max(1,2*len(U)-3)/max(energy,1e-9)))
                self.gain_sigma=sigma
                self.frozen_reason='gain_sign_or_uncertainty'
                info.update(candidate_gain_px_s2=k,candidate_gain_sigma_proxy=sigma,
                    gain_signal_to_noise=float(k/max(sigma,1e-9)),
                    gain_signal_to_noise_minimum=self.gain_signal_to_noise,
                    uncertainty_is_uncalibrated_overlapping_row_proxy=True)
                if 0.<k<5000. and sigma<max(k,200.) and k>=self.gain_signal_to_noise*sigma:
                    # Slow parameter update; keep mean acceleration continuous.
                    alpha=1-np.exp(-dt/1.);new_gain=self.gain+alpha*(k-self.gain)
                    if self.d is not None:self.d-=(new_gain-self.gain)*mean_u
                    self.gain=new_gain;self.learned=True;self.frozen_reason=None
                    info['gain_learned']=True
        # The effective gain needs excitation and multiple rows. With that gain
        # held fixed, the lumped disturbance does NOT need the same8-row gate.
        # Its first correction still waits for a matured0.2s displacement.
        if self.d is not None and len(self.rows)>=(1 if self.early_disturbance else 8):
            target=mean_r-self.gain*mean_u
            self.d+=(1-np.exp(-dt/1.))*np.clip(target-self.d,-3000.,3000.)
            info['disturbance_updated']=True
        a0=np.asarray(c['acceleration'],float) if c is not None else np.zeros(2)
        active=bool(good and c is not None)
        held=False
        if active:
            # Reacquisition after a gap establishes a new image/proxy gauge.
            if self.d is None or self.last_valid is None or t-self.last_valid>.12:
                self.d=a0-self.gain*np.asarray(u)
            if self.input_learning_current:self.last_valid=t
            a=self.gain*np.asarray(u)+self.d
            norm=np.linalg.norm(a)
            if norm>3000:a*=3000/norm
            self.last_acceleration=a.copy()
        else:
            held=bool(self.hold_acceleration_gaps and self.last_acceleration is not None
                and self.last_valid is not None and t-self.last_valid<=.12+1e-9)
            a=self.last_acceleration.copy() if held else a0
        pred={}
        if c is not None:
            p,v=np.asarray(c['position']),np.asarray(c['velocity'])
            pred={str(H):(p+H*v+.5*H*H*a).tolist() for H in (.1,.2,.4)}
        info.update(active=active,effective_gain_px_s2=self.gain,gain_ever_learned=self.learned,
            gain_sigma=self.gain_sigma,frozen_reason=self.frozen_reason,
            disturbance_px_s2=None if self.d is None else self.d.tolist(),
            acceleration_px_s2=a.tolist(),input_u=None if not good else np.asarray(u).tolist(),
            current_input_used_without_waiting_for_current_future_position=active,
            short_gap_acceleration_held=held,
            acceleration_source='current_force_input' if active else 'past_total_acceleration_hold' if held else 'position_state_default',
            last_current_force_age_s=None if self.last_valid is None else t-self.last_valid,
            held_input_not_a_current_thrust_measurement=True,
            direction_sign='image_up_prior',metric_force_measured=False)
        return pred,info
