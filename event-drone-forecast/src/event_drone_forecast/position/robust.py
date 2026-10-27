"""Existing23 robust acceleration math; frozen parameters, no history imports."""
import numpy as np
from scipy.optimize import least_squares
PARAMS={'min_real_observations': 5,
 'max_coast_s': 0.16,
 'reset_gap_s': 0.5,
 'horizons': [0.1, 0.2, 0.4],
 'acceleration_cap_px_s2': 3000.0,
 'minimum_fit_span_s': 0.16,
 'minimum_fit_points': 6,
 'huber_delta_px': 8.07,
 'noise_floor_px': 6.0,
 'max_nfev': 30}

def cap(acceleration):
    a = np.array(acceleration, float)
    norm = np.linalg.norm(a)
    return a * min(1.0, PARAMS['acceleration_cap_px_s2'] / max(norm, 1e-09))

def robust_acceleration(history):
    if len(history) < PARAMS['minimum_fit_points']:
        return (np.zeros(2), np.full(2, np.inf), dict(valid=False, reason='few_points'))
    times = np.array([t for t, p in history])
    points = np.array([p for t, p in history])
    span = np.ptp(times)
    if span < PARAMS['minimum_fit_span_s'] - 1e-08:
        return (np.zeros(2), np.full(2, np.inf), dict(valid=False, reason='short_time_support'))
    tau = (times - times.mean()) / span
    design = np.column_stack([np.ones(len(times)), tau, 0.5 * tau * tau])
    initial = np.linalg.lstsq(design, points, rcond=None)[0]
    jacobian = np.kron(design, np.eye(2))
    fit = least_squares(lambda x: (design @ x.reshape(3, 2) - points).ravel(), initial.ravel(), jac=lambda x: jacobian.copy(), loss='huber', f_scale=PARAMS['huber_delta_px'], max_nfev=PARAMS['max_nfev'])
    if not fit.success:
        return (np.zeros(2), np.full(2, np.inf), dict(valid=False, reason='solver_not_converged', nfev=fit.nfev))
    coefficients = fit.x.reshape(3, 2)
    residual = points - design @ coefficients
    variance = []
    for dim in range(2):
        r = residual[:, dim]
        weights = np.minimum(1.0, PARAMS['huber_delta_px'] / np.maximum(abs(r), 1e-12))
        sigma = max(PARAMS['noise_floor_px'], 1.4826 * float(np.median(abs(r - np.median(r)))))
        covariance = sigma * sigma * np.linalg.pinv(design.T @ (weights[:, None] * design))
        variance.append(max(0.0, float(covariance[2, 2])) / span ** 4)
    acceleration = coefficients[2] / span ** 2
    standard_error = np.sqrt(variance)
    return (acceleration, standard_error, dict(valid=True, reason='huber_fit', span_s=float(span), n=len(times), nfev=fit.nfev, acceleration=acceleration.tolist(), standard_error=standard_error.tolist(), median_abs_residual_px=float(np.median(abs(residual))), robust_inlier_fraction=float(np.mean(abs(residual) <= PARAMS['huber_delta_px']))))
