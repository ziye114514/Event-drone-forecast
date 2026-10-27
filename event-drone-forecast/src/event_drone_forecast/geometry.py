"""Bounded, deterministic ellipse/partial-arc/line measurements.

These are image-space observations, not calibrated attitude estimates. A line
constrains an axis only; an extrapolated arc centre is explicitly uncertain.
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def ellipse_points(parameters, phases):
    cx, cy, a, b, angle = parameters
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return np.column_stack((a*np.cos(phases), b*np.sin(phases))) @ rotation.T + [cx, cy]


def _fit(points):
    if len(points) < 6:
        return None
    try:
        (cx, cy), (a, b), angle = cv2.fitEllipseAMS(np.float32(points).reshape(-1, 1, 2))
    except cv2.error:
        return None
    if a < b:
        a, b, angle = b, a, angle + 90.
    p = np.array([cx, cy, a/2, b/2, np.deg2rad(angle) % np.pi])
    return p if np.isfinite(p).all() and b > .1 else None


def _distance(points, p):
    cx, cy, a, b, angle = p
    delta = points - [cx, cy]
    x = delta[:, 0]*np.cos(angle) + delta[:, 1]*np.sin(angle)
    y = -delta[:, 0]*np.sin(angle) + delta[:, 1]*np.cos(angle)
    gradient = 2*np.sqrt((x/a**2)**2 + (y/b**2)**2)
    return ((x/a)**2 + (y/b)**2 - 1) / np.maximum(gradient, 1e-6)


def _coverage(points, p):
    delta = points - p[:2]
    x = (delta[:, 0]*np.cos(p[4])+delta[:, 1]*np.sin(p[4]))/p[2]
    y = (-delta[:, 0]*np.sin(p[4])+delta[:, 1]*np.cos(p[4]))/p[3]
    phases = np.sort(np.arctan2(y, x) % (2*np.pi))
    if len(phases) < 3:
        return 0.
    return float(1-np.max(np.diff(np.r_[phases, phases[0]+2*np.pi]))/(2*np.pi))


def _uncertainty(points, p):
    # Local covariance with a correlation inflation factor for adjacent pixels.
    # This is a diagnostic approximation, not a calibrated confidence interval.
    if len(points) > 96:
        points = points[np.linspace(0, len(points)-1, 96).astype(int)]
    residual = _distance(points, p)
    sigma = max(.8, 1.4826*np.median(abs(residual-np.median(residual))))
    steps = np.array([0.05, 0.05, .05, .05, .001])
    # The same ten central-difference evaluations, batched over parameters.
    # No analytical approximation or cached past uncertainty is introduced.
    parameters=np.repeat(p[None],10,axis=0)
    parameters[::2]+=np.diag(steps);parameters[1::2]-=np.diag(steps)
    delta=points[None]-parameters[:,None,:2]
    cosine=np.cos(parameters[:,4,None]);sine=np.sin(parameters[:,4,None])
    x=delta[:,:,0]*cosine+delta[:,:,1]*sine
    y=-delta[:,:,0]*sine+delta[:,:,1]*cosine
    a=parameters[:,2,None];b=parameters[:,3,None]
    gradient=2*np.sqrt((x/a**2)**2+(y/b**2)**2)
    distance=((x/a)**2+(y/b)**2-1)/np.maximum(gradient,1e-6)
    jacobian=((distance[::2]-distance[1::2])/(2*steps[:,None])).T.copy()
    covariance = np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-10)*sigma**2*4
    singular = np.linalg.svd(jacobian*np.array([p[2], p[2], p[2], p[2], 1.]), compute_uv=False)
    condition = float(singular[0]/max(singular[-1], 1e-9))
    center_sigma = float(np.sqrt(max(np.linalg.eigvalsh(covariance[:2, :2])[-1], .25)))
    angle_sigma = float(np.rad2deg(np.sqrt(max(covariance[4, 4], 0))))
    return min(center_sigma, 500.), min(angle_sigma, 90.), condition


def _curved_points(points):
    if len(points) < 16:
        return points
    smooth = gaussian_filter1d(points.astype(float), 1.5, axis=0, mode='wrap')
    stride = max(3, min(8, len(points)//45))
    before, after = smooth-np.roll(smooth, stride, axis=0), np.roll(smooth, -stride, axis=0)-smooth
    cross = before[:, 0]*after[:, 1]-before[:, 1]*after[:, 0]
    turn = np.arctan2(cross, (before*after).sum(axis=1))
    sign = np.sign(np.sum(turn)) or 1
    keep = (turn*sign > .035) & (turn*sign < .7)
    return points[keep] if keep.sum() >= 12 else points


def _valid(p, points, max_major, prior):
    if p is None or p[2] < 5 or p[3] < .7 or p[3]/p[2] < .055 or 2*p[2] > max_major:
        return False
    span = max(np.ptp(points, axis=0).max(), 1.)
    if 2*p[2] > span*2.5 or np.linalg.norm(p[:2]-np.mean(points, axis=0)) > span:
        return False
    if prior is not None:
        if not .55 < 2*p[2]/max(prior['major'], 1.) < 1.7:
            return False
        if np.linalg.norm(p[:2]-prior['center']) > max(18., prior['major']*.65):
            return False
    return True


def _line(points, max_major):
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    axis = vectors[:, -1]
    along, across = (points-center) @ axis, (points-center) @ vectors[:, 0]
    major = float(np.ptp(along))
    low,high=np.percentile(across,[5,95])
    width = max(1., float(high-low))
    if major < 12 or major > max_major or width/major > .14:
        return None
    center = center+axis*(along.max()+along.min())/2
    angle = float(np.arctan2(axis[1], axis[0]) % np.pi)
    angle_sigma = max(1., float(np.rad2deg(width/max(major, 1.))))
    normal = np.array([-np.sin(angle), np.cos(angle)])
    return dict(center=center, major=major, minor=width, angle=float(np.rad2deg(angle)),
                ratio=width/major, normal=normal, mode='line', visible_arc_fraction=None,
                support_fraction=1., fit_residual_px=float(np.median(abs(across))),
                center_sigma_px=max(2., width), axis_angle_sigma_deg=angle_sigma,
                fit_condition=None, geometry_quality=float(np.clip(1-width/major*4, .25, .7)),
                shape_quality=.7, inferred_shape=False, axis_only=True,
                center_covariance_px2=(np.eye(2)*max(2., width)**2).tolist())


def fit_rotor_geometry(contour, max_major, prior=None, allow_arc=True):
    points = np.asarray(contour, float).reshape(-1, 2)
    if len(points) < 6:
        return None
    # Thin contours are observable as an axis even if a conic is singular.
    line = _line(points, max_major)
    if line is not None:
        return line
    p = _fit(points)
    valid = _valid(p, points, max_major, None)
    area = abs(cv2.contourArea(np.float32(points).reshape(-1, 1, 2)))
    fill = area/max(np.pi*p[2]*p[3], 1.) if valid else 0.
    initial_error = float(np.percentile(abs(_distance(points, p)), 85)) if valid else np.inf
    # Well-supported complete shapes retain a fast path. Partial contours include
    # occlusion cuts; robust hypotheses use curved samples rather than those cuts.
    fast = valid and .80 < fill < 1.3 and initial_error < max(1.3, .025*p[2]*2)
    candidates = [p] if valid else []
    evaluation_points = points
    if allow_arc and not fast:
        curved = _curved_points(points)
        evaluation_points = curved
        pool = curved[np.linspace(0, len(curved)-1, min(len(curved), 100)).astype(int)]
        candidate = _fit(pool)
        if _valid(candidate, points, max_major, prior):
            candidates.append(candidate)
        rng = np.random.default_rng(419)  # per-call deterministic; no future state
        if len(pool) >= 12:
            for _ in range(20):
                candidate = _fit(pool[rng.choice(len(pool), 7, replace=False)])
                if _valid(candidate, points, max_major, prior):
                    candidates.append(candidate)
    if not candidates:
        return None
    best = None
    for candidate in candidates:
        tolerance = max(1.4, .024*candidate[2]*2)
        distances = abs(_distance(evaluation_points, candidate))
        inliers = distances < tolerance
        if inliers.sum() < 10:
            continue
        coverage = _coverage(evaluation_points[inliers], candidate)
        if coverage < .25:
            continue
        support = float(inliers.mean())
        median = float(np.median(distances[inliers]))
        score = support + .25*coverage - .08*median/tolerance
        if prior is not None:
            score -= .08*np.linalg.norm(candidate[:2]-prior['center'])/max(prior['major'], 1.)
        if best is None or score > best[0]:
            best = (score, candidate, inliers, coverage, support, median)
    if best is None:
        return None
    _, p, inliers, coverage, support, median = best
    center_sigma, angle_sigma, condition = _uncertainty(evaluation_points[inliers], p)
    boundary_support=float(np.mean(abs(_distance(points,p))<max(1.4,.048*p[2])))
    mode = 'ellipse' if coverage >= .82 and support >= .88 and boundary_support>=.88 else 'arc'
    if mode == 'arc':
        # Explicit penalty for unobserved boundary and unmodelled occluder edges.
        center_sigma = max(center_sigma, (1-coverage)*p[2]*.15)
    if center_sigma > min(30., max(8., p[2]*.35)) or (condition > 1e6 and mode=='arc'):
        return None
    geometry_quality = float(np.clip(coverage*np.sqrt(support)*np.exp(-angle_sigma/20), .05, 1.))
    if p[3]/p[2] > .9:
        # A near-circle has a poorly observable in-plane axis.
        geometry_quality *= max(.05, (1-p[3]/p[2])/.1)
        angle_sigma=max(angle_sigma,45.*(p[3]/p[2]-.9)/.1)
    normal = np.array([-np.sin(p[4]), np.cos(p[4])])*np.sqrt(max(0., 1-(p[3]/p[2])**2))
    return dict(center=p[:2], major=float(2*p[2]), minor=float(2*p[3]), angle=float(np.rad2deg(p[4])),
                ratio=float(p[3]/p[2]), normal=normal, mode=mode,
                visible_arc_fraction=coverage, support_fraction=boundary_support,
                fit_residual_px=median, center_sigma_px=center_sigma,
                axis_angle_sigma_deg=angle_sigma, fit_condition=condition,
                geometry_quality=geometry_quality,
                shape_quality=float(np.clip(np.sqrt(support)*np.exp(-median/max(2.,p[2]*.04)),.3,1.)),
                inferred_shape=mode == 'arc', axis_only=False,
                center_covariance_px2=(np.eye(2)*center_sigma**2).tolist())
