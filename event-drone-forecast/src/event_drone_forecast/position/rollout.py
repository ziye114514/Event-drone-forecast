"""Curved rollout with the SAME current position, velocity and acceleration.

Normal acceleration sets instantaneous turn rate; tangential acceleration sets
speed change. This changes higher-order rollout only, not the measured state.
Unreliable near-zero speed or >90-degree extrapolation retains the baseline.
"""
import numpy as np
PARAMS=dict(minimum_speed_px_s=30.,maximum_extrapolated_angle_rad=float(np.pi/2),
    same_damping_alpha=.9,no_fitted_parameters=True,default_promoted=False)
def forecast(c,h):
    if c is None:return None
    p=np.asarray(c['position'],float);v=np.asarray(c['velocity'],float);a=np.asarray(c['acceleration'],float)
    squared=float(v@v);d=v*h+.5*a*h*h
    if squared>=PARAMS['minimum_speed_px_s']**2:
        omega=float((v[0]*a[1]-v[1]*a[0])/squared);theta=omega*h
        if abs(theta)<=PARAMS['maximum_extrapolated_angle_rad']:
            sinc=float(np.sinc(theta/np.pi));half=float(np.sinc(theta/(2*np.pi)))
            cosc=.5*theta*half*half;c1=sinc-.5*half*half
            c2=theta/3-theta**3/30+theta**5/840 if abs(theta)<.001 else (sinc-np.cos(theta))/theta
            parallel=v*float(a@v)/squared
            Jv=np.array([-v[1],v[0]]);Ja=np.array([-parallel[1],parallel[0]])
            d=h*(sinc*v+cosc*Jv)+h*h*(c1*parallel+c2*Ja)
    return (p+max(.5,1-PARAMS['same_damping_alpha']*h*h)*d).tolist()
