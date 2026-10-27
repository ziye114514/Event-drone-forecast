"""Current live motion state; mechanically extracted, same numerical rules."""
from collections import deque
import copy,math
import numpy as np
from .kalman import MeasurementKF
from .robust import robust_acceleration,cap
from .rollout import forecast as curved_forecast

class LiveMotion:

    def __init__(self, tracker_hz, noise_model='legacy'):
        if noise_model not in ('legacy', 'continuous'):
            raise ValueError('Unknown noise model')
        self.noise_model = noise_model
        self.period = 1.0 / tracker_hz
        self.kf = MeasurementKF(noise_model)
        self.identity = None
        self.history = deque()
        self.support = deque()
        self.seen = None
        self.observations = 0

    def step(self, t, body):
        identity = (body.get('body_track_epoch', 0), body.get('body_track_id')) if body.get('body_track_id') is not None else None
        if identity is not None and self.identity is not None and (identity != self.identity):
            self.__init__(1.0 / self.period, self.noise_model)
        if identity is not None:
            self.identity = identity
        if self.seen is not None and t - self.seen > 0.5:
            hz = 1.0 / self.period
            self.__init__(hz, self.noise_model)
        detected = bool(body.get('detected'))
        strong = detected and (not body.get('low_confidence', False))
        box = np.asarray(body.get('bbox', []), float)
        complete = bool(box.shape == (4,) and np.isfinite(box).all() and np.all(box[:2] > 0) and np.all(box[2:] < [1279, 719]))
        self.support.append((t, strong, complete))
        if detected:
            self.seen = t
            self.observations += 1
            if strong:
                self.history.append((t, np.asarray(body['center'], float)))
        while self.history and t - self.history[0][0] > 0.35 + 1e-08:
            self.history.popleft()
        while self.support and t - self.support[0][0] > 0.35 + 1e-08:
            self.support.popleft()
        measurement_scale = min(4.0, 0.25 / max(float(body.get('confidence', 0.25)), 0.1)) if detected and (not strong) else 1.0
        c = self.kf.step(t, body['center'] if detected else None, measurement_scale)
        a, se, d = robust_acceleration(self.history)
        expected = math.floor(0.35 / self.period + 1e-08) + 1
        times = [s for s, seen, _ in self.support if seen]
        coverage = min(1.0, len(times) / expected)
        span = times[-1] - times[0] if times else 0.0
        gap = max(np.diff(times), default=0.0)
        eligible = bool(detected and coverage >= 0.9 and (span >= 0.35 - self.period - 1e-08) and (gap <= 2 * self.period + 1e-08) and all((ok for _, seen, ok in self.support if seen)) and d.get('valid') and (d['median_abs_residual_px'] <= 6.0) and (d['robust_inlier_fraction'] >= 0.9))
        if c is not None:
            c = copy.deepcopy(c)
            shrink = np.maximum(0.0, 1.0 - se * se / (a * a + 1e-12)) if eligible else np.zeros(2)
            c['acceleration'] = cap(a * shrink).tolist()
        valid = bool(c is not None and self.observations >= 5 and (self.seen is not None) and (t - self.seen <= 0.16 + 1e-08))
        return (c, valid, dict(acceleration_eligible=eligible, coverage=coverage, span_s=span, maximum_gap_s=float(gap), fit=d, observations=self.observations, measurement_noise_scale=measurement_scale, strong_observation=strong, geometry_used=False, thrust_estimated=False))

def predictions(c, age=0.0):
    if c is None:
        return ({}, {})
    preferred = {str(h): curved_forecast(c, age + h) for h in (0.1, 0.2, 0.4)}
    cv = {str(h): (np.asarray(c['position']) + (age + h) * np.asarray(c['velocity'])).tolist() for h in (0.1, 0.2, 0.4)}
    return (preferred, cv)
