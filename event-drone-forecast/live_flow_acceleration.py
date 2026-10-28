"""Causal derivative of qualified raw image-flow velocity, with uncertainty.

Uses the project's existing Huber least-squares conventions: .35s history,
six points/.16s span, normalized Huber8.07/6, and the same3000px/s² cap.
No resampling or centered Savitzky-Golay pass, since cutoffs are irregular and
future samples must not be used. Covariances remain engineering proxies.
"""
from collections import deque
import copy
import numpy as np
from scipy.optimize import least_squares
from live_local_flow import box
from live_motion import cap

DESIGN=dict(window_s=.35,minimum_points=6,minimum_span_s=.16,maximum_gap_s=.08,
    huber_standardized_delta=8.07/6.,max_nfev=30,minimum_robust_inlier_fraction=.9,
    maximum_median_standardized_residual=1.,position_residual_px=6.,position_inlier_px=8.07,
    acceleration_cap_px_s2=3000.,policies=['velocity-only','position-consistent'],
    current_position_velocity_and_validity_unchanged=True,raw_midpoint_velocity_before_acceleration_compensation=True,
    covariance_is_uncalibrated_proxy=True,parameters_swept=False,default_promoted=False)


def fit_derivative(history,t):
    if len(history)<6:return None,dict(reason='few_flow_velocities',n=len(history))
    times=np.array([r[0] for r in history]);span=float(np.ptp(times));gap=float(np.max(np.diff(times)))
    info=dict(n=len(times),span_s=span,maximum_gap_s=gap)
    if span<.16-1e-8:return None,dict(info,reason='short_flow_velocity_span')
    if gap>.08+1e-8:return None,dict(info,reason='flow_velocity_gap')
    velocities=np.array([r[1] for r in history])
    tau=(times-t)/span;design=np.column_stack((np.ones(len(times)),tau))
    jacobian=np.kron(design,np.eye(2)).reshape(len(times),2,4)
    # Accepted observations are immutable; reuse their original whitening.
    # Three-element histories remain supported for callers outside this class.
    whiten=np.array([r[3] if len(r)>3 else np.linalg.inv(np.linalg.cholesky(r[2])) for r in history])
    J=(whiten@jacobian).reshape(-1,4);target=(whiten@velocities[:,:,None]).ravel()
    initial=np.linalg.lstsq(J,target,rcond=None)[0]
    initial_residual=J@initial-target
    # Huber is exactly quadratic here. If the existing solver's default gtol
    # is already satisfied, it returns this same initial WLS solution at nfev1.
    if np.all(abs(initial_residual)<=8.07/6.) and np.max(abs(J.T@initial_residual))<1e-8:
        parameters=initial;nfev=1
    else:
        fit=least_squares(lambda p:J@p-target,initial,jac=lambda p:J.copy(),loss='huber',f_scale=8.07/6.,max_nfev=30)
        if not fit.success:return None,dict(info,reason='flow_derivative_solver_failed')
        parameters=fit.x;nfev=fit.nfev
    residual=J@parameters-target;inliers=float(np.mean(abs(residual)<=8.07/6.));median=float(np.median(abs(residual)))
    info.update(median_standardized_residual=median,robust_inlier_fraction=inliers,nfev=nfev)
    if inliers<.9 or median>1.:return None,dict(info,reason='flow_derivative_residual_not_supported')
    weights=np.minimum(1.,(8.07/6.)/np.maximum(abs(residual),1e-12))
    sigma=max(1.,1.4826*float(np.median(abs(residual-np.median(residual)))))
    covariance=sigma*sigma*np.linalg.pinv(J.T@(weights[:,None]*J))
    acceleration=parameters[2:]/span;se=np.sqrt(np.maximum(0.,np.diag(covariance)[2:]))/span
    shrink=np.maximum(0.,1.-se*se/(acceleration*acceleration+1e-12));revised=cap(acceleration*shrink)
    info.update(acceleration=acceleration.tolist(),standard_error_proxy=se.tolist(),shrunk_acceleration=revised.tolist(),
        fitted_current_velocity=parameters[:2].tolist(),reason='qualified_velocity_derivative')
    if np.linalg.norm(revised)<=1e-8:return None,dict(info,reason='flow_derivative_signal_not_supported')
    return revised,info


class FlowAcceleration:
    def __init__(self):self.velocities=deque();self.positions=deque();self.last_t=None;self.identity=None

    def step(self,t,coefficient,body,flow,policy='position-consistent'):
        if policy not in DESIGN['policies']:raise ValueError('Unknown flow acceleration policy')
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing flow acceleration cutoffs required')
        self.last_t=t
        identity=(body.get('body_track_epoch'),body.get('body_track_id')) if body.get('body_track_id') is not None else None
        if identity is not None and self.identity is not None and identity!=self.identity:self.velocities.clear();self.positions.clear()
        if identity is not None:self.identity=identity
        current=box(body)
        if current is not None:self.positions.append((t,np.asarray(body['center'],float)))
        if current is not None and flow.get('valid'):
            midpoint=flow['midpoint_t']
            if midpoint>t+1e-8:raise ValueError('Future flow velocity supplied')
            covariance=np.asarray(flow['velocity_covariance'],float)
            whitener=np.linalg.inv(np.linalg.cholesky(covariance))
            self.velocities.append((midpoint,np.asarray(flow['velocity'],float),covariance,whitener))
        while self.velocities and t-self.velocities[0][0]>.35+1e-8:self.velocities.popleft()
        while self.positions and t-self.positions[0][0]>.35+1e-8:self.positions.popleft()
        if coefficient is None or current is None or not flow.get('valid'):return coefficient,dict(applied=False,reason='no_current_qualified_flow')
        acceleration,info=fit_derivative(self.velocities,t)
        if acceleration is None:return coefficient,dict(info,applied=False)
        if policy=='position-consistent':
            if len(self.positions)<6:return coefficient,dict(info,applied=False,reason='few_position_consistency_points')
            times=np.array([r[0]-t for r in self.positions]);p=np.array([r[1] for r in self.positions])
            if np.ptp(times)<.16-1e-8:return coefficient,dict(info,applied=False,reason='short_position_consistency_span')
            dynamics=times[:,None]*np.asarray(coefficient['velocity'])+.5*times[:,None]**2*acceleration
            origin=np.median(p-dynamics,axis=0);residual=p-origin-dynamics
            median=float(np.median(abs(residual)));inliers=float(np.mean(abs(residual)<=8.07))
            info.update(position_median_residual_px=median,position_inlier_fraction=inliers)
            if median>6. or inliers<.9:return coefficient,dict(info,applied=False,reason='flow_derivative_disagrees_with_position_history')
        revised=copy.deepcopy(coefficient);revised['acceleration']=acceleration.tolist()
        return revised,dict(info,applied=True,policy=policy,reason='qualified_flow_acceleration')
