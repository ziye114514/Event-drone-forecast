"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""
from itertools import combinations
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from event_drone_forecast.geometry import _fit,_distance,_coverage,_uncertainty

PARAMS = dict(minimum_defect_depth_px=2.0, minimum_concave_turn_rad=.12,
              minimum_arc_coverage=.48, maximum_arc_rms_px=1.6,
              maximum_center_sigma_px=3., minimum_union_iou=.78,
              minimum_boundary_error_reduction=.35, minimum_center_separation=.30,
              maximum_size_ratio=2., maximum_pair_angle_deg=20., max_notches=8,
              maximum_notch_pair_trials=6, random_hypotheses_per_arc=16,
              minimum_single_fit_rms_px=1.5, threshold=4.,
              source='Current threshold-4 contour only; split ends trimmed; artificial closing chord NEVER fitted.',
              units='sensor pixels / current20ms ON/OFF pair counts',
              output='Two fitted partial ellipses, not two complete measured rims or verified motor IDs.',
              pairing='Both children of a split must survive grouping together; otherwise restore original parent candidates.',
              default_promoted=False)

def notches(contour):
    p = np.asarray(contour, np.int32).reshape(-1, 2)
    if len(p) < 35: return []
    hull = cv2.convexHull(p.reshape(-1, 1, 2), returnPoints=False)
    if hull is None or len(hull) < 4: return []
    try: defects = cv2.convexityDefects(p.reshape(-1, 1, 2), hull)
    except cv2.error: return []
    if defects is None: return []
    turns = []
    # CHAIN_APPROX_NONE gives approximately one-pixel arclength sampling.
    orientation = np.sign(cv2.contourArea(p.reshape(-1, 1, 2), oriented=True)) or 1.
    for sigma, stride in ((1.2, 4), (2., 6)):
        smooth = gaussian_filter1d(p.astype(float), sigma, axis=0, mode='wrap')
        before = smooth-np.roll(smooth, stride, axis=0)
        after = np.roll(smooth, -stride, axis=0)-smooth
        turns.append(orientation*np.arctan2(before[:, 0]*after[:, 1]-before[:, 1]*after[:, 0],
                                          np.sum(before*after, axis=1)))
    found = []
    for start, end, far, depth in np.asarray(defects).reshape(-1, 4):
        depth = float(depth)/256.
        if depth < PARAMS['minimum_defect_depth_px']: continue
        # Defect and negative signed turn must agree at BOTH smoothing scales.
        nearby = (np.arange(far-3, far+4) % len(p)).astype(int)
        index = int(nearby[np.argmin(turns[1][nearby])])
        if any(turn[index] > -PARAMS['minimum_concave_turn_rad'] for turn in turns): continue
        if any(min(abs(index-r['index']), len(p)-abs(index-r['index'])) < 8 for r in found): continue
        found.append(dict(index=index, point=p[index].tolist(), depth=depth,
                          turn=[float(turn[index]) for turn in turns]))
    return sorted(found, key=lambda r: r['depth'], reverse=True)[:PARAMS['max_notches']]

def fit_arc(points, parent_extent):
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
        p = refined; inliers = abs(_distance(points, p)) <= 1.8; supported = points[inliers]
        if len(supported) < max(22, .60*len(points)): continue
        if not (10 <= 2*p[2] <= parent_extent*1.6 and p[3] >= 2.5 and p[3]/p[2] >= .12): continue
        rms = float(np.sqrt(np.mean(_distance(supported, p)**2))); coverage = _coverage(supported, p)
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

def pair_evidence(contour, arcs):
    p = np.asarray(contour, float).reshape(-1, 2)
    a, b = [r['parameters'] for r in arcs]
    separation = float(np.linalg.norm(a[:2]-b[:2]))
    if separation < PARAMS['minimum_center_separation']*max(2*a[2], 2*b[2]): return None
    if max(a[2], b[2])/min(a[2], b[2]) > PARAMS['maximum_size_ratio']: return None
    angle_gap = np.rad2deg(np.arccos(np.clip(abs(np.cos(a[4]-b[4])), 0, 1)))
    if angle_gap > PARAMS['maximum_pair_angle_deg']: return None
    if max(a[3]/a[2], b[3]/b[2])/min(a[3]/a[2], b[3]/b[2]) > 2.: return None
    # A lower residual alone can reward fitting small unrelated arcs. Compare
    # the entire parent's support with the UNION of independently fitted disks.
    bounds = []
    for q in (a, b):
        dx = np.hypot(q[2]*np.cos(q[4]), q[3]*np.sin(q[4]))
        dy = np.hypot(q[2]*np.sin(q[4]), q[3]*np.cos(q[4]))
        bounds.extend([q[:2]-[dx, dy], q[:2]+[dx, dy]])
    all_bounds = np.r_[p, bounds]
    lo = np.floor(np.min(all_bounds, axis=0)-4).astype(int)
    hi = np.ceil(np.max(all_bounds, axis=0)+5).astype(int)
    if max(hi-lo) > 700: return None
    shape = tuple((hi-lo)[::-1]); actual = np.zeros(shape, np.uint8); union = np.zeros_like(actual)
    cv2.fillPoly(actual, [np.rint(p-lo).astype(np.int32)], 1)
    for q in (a, b):
        cv2.ellipse(union, tuple(np.rint(q[:2]-lo).astype(int)),
                    tuple(np.maximum(1, np.rint(q[2:4]).astype(int))), float(np.rad2deg(q[4])), 0, 360, 1, -1)
    iou = float(np.sum((actual>0)&(union>0))/max(1, np.sum((actual>0)|(union>0))))
    if iou < PARAMS['minimum_union_iou']: return None
    single = _fit(p)
    if single is None: return None
    one_error = float(np.sqrt(np.mean(_distance(p, single)**2)))
    two_error = float(np.sqrt(np.mean(np.minimum(abs(_distance(p, a)), abs(_distance(p, b)))**2)))
    gain = 1-two_error/max(one_error, 1e-9)
    if one_error < PARAMS['minimum_single_fit_rms_px'] or gain < PARAMS['minimum_boundary_error_reduction']: return None
    return dict(union_iou=iou, one_rms_px=one_error, two_rms_px=two_error,
                error_reduction=gain, separation_px=separation, angle_gap_deg=float(angle_gap))

def split_contour(contour):
    p = np.asarray(contour, np.int32).reshape(-1, 2)
    concave = notches(contour)
    diag = dict(accepted=False, notches=concave, pairs_tried=0, reason='no_supported_notch_pair')
    if len(concave) < 2: return None, diag
    extent = max(np.ptp(p, axis=0)); best = None
    for left, right in combinations(concave, 2):
        i, j = sorted([left['index'], right['index']]); trim = 3
        # These are two OPEN observed paths. Do not add a cut line or join ends.
        segments = [p[i+trim:j-trim+1], p[np.arange(j+trim, i+len(p)-trim+1) % len(p)]]
        if min(map(len, segments)) < max(22, .22*len(p)): continue
        diag['pairs_tried'] += 1
        if diag['pairs_tried'] > PARAMS['maximum_notch_pair_trials']: break
        arcs = [fit_arc(segment, extent) for segment in segments]
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
