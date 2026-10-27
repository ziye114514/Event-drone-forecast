"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""
import numpy as np
from event_drone_forecast.position._filterpy import predict,update

PARAMS=dict(reset_gap_s=.5,position_sigma_px=6.,acceleration_cap=3000.,
    jerk_sigma=6400./np.sqrt(.35),minimum_observations=6,minimum_span_s=.16,
    innovation_chi2_99=9.21034037197618,
    uncertainty_is_proxy=True,force_in_newtons=False,wind_or_mass_identified=False)

class LumpedDisturbance:
    def __init__(self,measurement_sigma=6.,jerk_sigma=None):
        self.sigma=float(measurement_sigma);self.jerk_sigma=PARAMS['jerk_sigma'] if jerk_sigma is None else float(jerk_sigma)
        self.x=None;self.P=None;self.last_t=None;self.seen=None;self.started=None
        self.observations=0;self.previous_u=np.zeros(2);self.input_valid=False;self.identity=None

    def step(self,t,position,known_acceleration=None,identity=None):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing cutoffs required')
        z=None if position is None else np.asarray(position,float)
        u=np.zeros(2) if known_acceleration is None else np.asarray(known_acceleration,float)
        if u.shape!=(2,) or not np.isfinite(u).all():raise ValueError('Finite2D input required')
        if z is not None and (z.shape!=(2,) or not np.isfinite(z).all()):raise ValueError('Finite2D position required')
        valid=known_acceleration is not None
        reset=(identity is not None and self.identity is not None and identity!=self.identity)
        reset=reset or (self.seen is not None and t-self.seen>.5)
        if reset:
            self.x=None;self.P=None;self.observations=0;self.started=None
        if identity is not None:self.identity=identity
        accepted=False;nis=None
        if self.x is None:
            if z is not None:
                self.x=np.r_[z,np.zeros(4)];self.P=np.diag([self.sigma**2]*2+[1e4]*2+[1e6]*2)
                self.started=t;self.seen=t;self.observations=1;accepted=True
        else:
            dt=t-self.last_t;I=np.eye(2);F=np.eye(6)
            F[:2,2:4]=dt*I;F[:2,4:6]=.5*dt*dt*I;F[2:4,4:6]=dt*I
            B=np.r_[.5*dt*dt*I,dt*I,np.zeros((2,2))]
            Q1=np.array([[dt**5/20,dt**4/8,dt**3/6],
                         [dt**4/8,dt**3/3,dt**2/2],[dt**3/6,dt**2/2,dt]])
            self.x,self.P=predict(self.x,self.P,F=F,Q=self.jerk_sigma**2*np.kron(Q1,I))
            self.x+=B@self.previous_u
            H=np.c_[I,np.zeros((2,4))];R=I*self.sigma**2
            if z is not None:
                innovation=z-H@self.x;S=H@self.P@H.T+R
                nis=float(innovation@np.linalg.solve(S,innovation))
                if nis<=PARAMS['innovation_chi2_99']:
                    self.x,self.P=update(self.x,self.P,z,R=R,H=H)
                    self.seen=t;self.observations+=1;accepted=True
            if valid!=self.input_valid:
                # Availability changes are changes of decomposition, not a
                # sudden observed acceleration. Keep total acceleration fixed.
                self.x[4:6]-=u-self.previous_u
            self.x[4:6]=np.clip(self.x[4:6],-3000.,3000.)
        self.last_t=t;self.previous_u=u.copy();self.input_valid=valid
        ready=bool(self.x is not None and self.observations>=6 and t-self.started>=.16
                   and self.seen is not None and t-self.seen<=.16)
        coefficient=None if self.x is None else dict(position=self.x[:2].tolist(),velocity=self.x[2:4].tolist(),
            acceleration=np.clip(u+self.x[4:6],-3000.,3000.).tolist())
        diagnostic=dict(ready=ready,accepted_position=accepted,innovation_nis=nis,
            disturbance_acceleration=None if self.x is None else self.x[4:6].tolist(),
            input_acceleration=u.tolist(),input_available=valid,
            disturbance_includes_unobserved_thrust=not valid,observations=self.observations,
            force_calibrated=False,uncertainty_is_proxy=True)
        return coefficient,diagnostic

def forecasts(coefficient):
    if coefficient is None:return {}
    p,v,a=(np.asarray(coefficient[k]) for k in ['position','velocity','acceleration'])
    return {str(h):(p+h*v+.5*h*h*a).tolist() for h in (.1,.2,.4)}
