"""Equivalent-compute geometry paths; old functions remain available.
Own project geometry/curvature definitions reused with safe early-outs and
lower-overhead percentile dispatch. No detection threshold or sample changed.
"""
import numpy as np
import cv2
from itertools import combinations
from event_drone_forecast.geometry import _fit,_distance,_coverage,_uncertainty,_curved_points,_valid
from curvature_split import PARAMS,notches,pair_evidence

def percentile_pair_linear(values):
    # Same NumPy default linear interpolation, including its upper-half lerp.
    q=np.array([.05,.95]);virtual=(len(values)-1)*q
    lower=np.floor(virtual).astype(int);upper=np.ceil(virtual).astype(int)
    partition=np.partition(values,np.unique(np.concatenate((lower,upper))))
    a,b=partition[lower],partition[upper];fraction=virtual-lower
    difference=b-a
    return np.where(fraction>=.5,b-difference*(1-fraction),a+difference*fraction)

def line_geometry_fast(points, max_major):
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    axis = vectors[:, -1]
    along, across = (points-center) @ axis, (points-center) @ vectors[:, 0]
    major = float(np.ptp(along))
    low,high=percentile_pair_linear(across)
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

def fit_rotor_geometry_fast(contour, max_major, prior=None, allow_arc=True, minimum_major=0.,uncertainty_fn=None,line_fn=None):
    points = np.asarray(contour, float).reshape(-1, 2)
    if len(points) < 6:
        return None
    # Thin contours are observable as an axis even if a conic is singular.
    line = (line_geometry_fast if line_fn is None else line_fn)(points, max_major)
    if line is not None:
        return line if line['major']>=minimum_major else None
    p = _fit(points)
    valid = _valid(p, points, max_major, None)
    if not allow_arc and (not valid or 2*p[2]<minimum_major):
        # With a single full-conic hypothesis, size cannot change downstream.
        # The existing largest/2.5 filter would discard this candidate anyway.
        # Avoid residual, coverage and SVD uncertainty for that doomed result.
        return None
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
    center_sigma, angle_sigma, condition = (_uncertainty if uncertainty_fn is None else uncertainty_fn)(evaluation_points[inliers], p)
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

def fit_arc_fast(points, parent_extent):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 22: return None
    # Event masks also contain blade gaps and airframe edges inside a path.
    # Reuse the project's deterministic ellipse hypothesis / residual approach,
    # retaining only actual supported points, never a fabricated closing chord.
    def hypotheses():
        yield _fit(points)
        rng = np.random.default_rng(147)
        for _ in range(PARAMS['random_hypotheses_per_arc']):
            yield _fit(points[np.sort(rng.choice(len(points), 7, replace=False))])
    best = None
    for p in hypotheses():
        if p is None: continue
        if not (10 <= 2*p[2] <= parent_extent*1.6 and p[3] >= 2.5 and p[3]/p[2] >= .12): continue
        inliers = abs(_distance(points, p)) <= 1.8
        if inliers.sum() < max(22, .60*len(points)): continue
        refined = _fit(points[inliers])
        if refined is None: continue
        p = refined; distances = _distance(points, p); inliers = abs(distances) <= 1.8; supported = points[inliers]
        if len(supported) < max(22, .60*len(points)): continue
        if not (10 <= 2*p[2] <= parent_extent*1.6 and p[3] >= 2.5 and p[3]/p[2] >= .12): continue
        rms = float(np.sqrt(np.mean(distances[inliers]**2))); coverage = _coverage(supported, p)
        if coverage < PARAMS['minimum_arc_coverage'] or rms > PARAMS['maximum_arc_rms_px']: continue
        score = inliers.mean()+.2*coverage-.1*rms
        if best is None or score > best[0]: best = score, p, supported, rms, coverage, float(inliers.mean())
        if inliers.mean() >= .90 and coverage >= .72 and rms < .8: break
    if best is None: return None
    _, p, supported, rms, coverage, support = best
    sigma, angle_sigma, condition = _uncertainty(supported, p)
    if sigma > PARAMS['maximum_center_sigma_px'] or condition > 1500: return None
    return dict(parameters=p, points=supported, rms=rms, coverage=coverage,
                sigma=sigma, angle_sigma=angle_sigma, condition=condition, support_fraction=support)

def split_contour_fast(contour):
    p = np.asarray(contour, np.int32).reshape(-1, 2)
    concave = notches(contour)
    diag = dict(accepted=False, notches=concave, pairs_tried=0, reason='no_supported_notch_pair')
    if len(concave) < 2: return None, diag
    extent = max(np.ptp(p, axis=0)); best = None
    # Every successful pair_evidence call needs the SAME parent's fit/error.
    single = _fit(p)
    if single is None or float(np.sqrt(np.mean(_distance(p.astype(float), single)**2))) < PARAMS['minimum_single_fit_rms_px']:
        return None,dict(diag,reason='parent_single_fit_invariant_rejected_without_child_fits')
    for left, right in combinations(concave, 2):
        i, j = sorted([left['index'], right['index']]); trim = 3
        # These are two OPEN observed paths. Do not add a cut line or join ends.
        segments = [p[i+trim:j-trim+1], p[np.arange(j+trim, i+len(p)-trim+1) % len(p)]]
        if min(map(len, segments)) < max(22, .22*len(p)): continue
        diag['pairs_tried'] += 1
        if diag['pairs_tried'] > PARAMS['maximum_notch_pair_trials']: break
        first_arc = fit_arc_fast(segments[0], extent)
        if first_arc is None: continue
        second_arc = fit_arc_fast(segments[1], extent)
        arcs = [first_arc, second_arc]
        if any(arc is None for arc in arcs): continue
        evidence = pair_evidence(p, arcs)
        if evidence is None: continue
        score = evidence['union_iou']+.2*evidence['error_reduction']-.05*sum(r['rms'] for r in arcs)
        if best is None or score > best[0]: best = (score, arcs, evidence, [left, right])
        # First passing high-depth pair: bounded online work, not an exhaustive
        # decomposition of arbitrary clusters. Rejected cases retain146.
        break
    if best is None: return None, diag
    _, arcs, evidence, cuts = best
    diag.update(accepted=True, reason='paired_concavities_and_two_supported_original_arcs',
                evidence=evidence, cuts=cuts, no_cut_chord=True,
                centers=[r['parameters'][:2].tolist() for r in arcs],
                coverages=[r['coverage'] for r in arcs], rms=[r['rms'] for r in arcs],
                center_sigma_px=[r['sigma'] for r in arcs])
    return arcs, diag
