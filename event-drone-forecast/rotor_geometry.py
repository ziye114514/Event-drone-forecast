"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""
from itertools import permutations
import numpy as np
import cv2

PARAMS = dict(step_us=20000, threshold_units='blurred min(N_ON,N_OFF) per 20 ms',
              threshold_default=4., thin_fallback_threshold=2., split_threshold_factor=2.,
              three_max_ratio_spread=1.6, adaptive_percentile=70., model_expiry_s=.3,
              side_max_ratio=.16, side_axis_tolerance_deg=15.,
              frequency_min_pixels=8, frequency_mode_fraction=.35,
              minimum_assumed_blades=2, missing_rate_max_age_s=.12,
              missing_rate_fallback='equal speeds of hidden and visible rotors, explicitly hypothetical',
              geometry_reuse='110 merge-safe components, 61 line-first geometry, 124 mask provenance',
              frequency_reuse='79 installed SDK FrequencyAlgorithm strict7; original events before spatial suppression',
              one_or_zero='allowed observation failure, baseline forecast retained',
              physical_units='relative squared blade-passage-frequency proxy; not calibrated Newtons/RPM')

def centres(rotors):
    return np.array([[r['cx'], r['cy']] for r in rotors], float).reshape(-1, 2)

def unit(x):
    x = np.asarray(x, float)
    return x / max(np.linalg.norm(x), 1e-12)

def metric(rotors):
    """Inverse projected-circle metric, up to unknown common scale."""
    matrices = []
    for r in rotors:
        angle = np.deg2rad(r['angle']); c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        matrices.append(rotation @ np.diag([1., 1. / max(r['ratio'], .08)**2]) @ rotation.T)
    return np.mean(matrices, axis=0)

def ellipse_tensor(r):
    angle = np.deg2rad(r['angle']); c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag([(r['major']/2)**2, (r['minor']/2)**2]) @ rotation.T

def three_affine_hypotheses(rotors):
    """Infer a missing rectangular-layout hub using ellipse-whitened angles.

    Three unlabelled points alone admit three parallelograms. Coplanar equal
    circular disks supply an affine metric, which can resolve the right corner.
    This is not an exact projective reconstruction and may remain ambiguous.
    """
    p = centres(rotors)
    if len(p) != 3 or any(r.get('mode') != 'ellipse' or r.get('inferred_shape')
                           or r['ratio'] < .16 for r in rotors):
        return dict(valid=False, reason='three_need_three_nonflat_measured_ellipses', hypotheses=[])
    singular = np.linalg.svd(p - p.mean(0), compute_uv=False)
    if singular[1] / max(singular[0], 1e-9) < .08:
        return dict(valid=False, reason='three_near_collinear', hypotheses=[])
    G = metric(rotors)
    sizes = np.array([r['major'] for r in rotors])
    if sizes.max() / sizes.min() > 1.6:
        return dict(valid=False, reason='unequal_projected_disk_sizes', hypotheses=[])
    ratios = np.array([r['ratio'] for r in rotors])
    if ratios.max()/ratios.min() > PARAMS['three_max_ratio_spread']:
        return dict(valid=False, reason='three_shapes_not_one_affine_plane_possible_merged_disk', hypotheses=[])
    hypotheses = []
    for corner in range(3):
        j, k = [i for i in range(3) if i != corner]
        a, b = p[j] - p[corner], p[k] - p[corner]
        cosine = abs(float(a @ G @ b)) / max(np.sqrt((a @ G @ a) * (b @ G @ b)), 1e-9)
        inferred = p[j] + p[k] - p[corner]
        hypotheses.append(dict(corner=corner, orthogonality_error=float(cosine),
                               inferred_center=inferred.tolist()))
    hypotheses.sort(key=lambda q: q['orthogonality_error'])
    best = hypotheses[0]
    margin = hypotheses[1]['orthogonality_error'] - best['orthogonality_error']
    valid = best['orthogonality_error'] <= .25 and margin >= .20
    return dict(valid=valid, reason='weak_perspective_rectangle' if valid else 'three_assignment_ambiguous',
                inferred_center=best['inferred_center'], margin=float(margin),
                hypotheses=hypotheses, inferred_is_observation=False)

def two_axis(rotors):
    p = centres(rotors)
    if len(p) != 2:
        return dict(valid=False, reason='need_two')
    delta = p[1] - p[0]; distance = float(np.linalg.norm(delta))
    direction = unit(delta)
    if distance < max(12., .35 * np.median([r['major'] for r in rotors])):
        return dict(valid=False, reason='two_centres_too_close')
    alignment = [abs(float(direction @ np.array([np.cos(np.deg2rad(r['angle'])),
                                                np.sin(np.deg2rad(r['angle']))]))) for r in rotors]
    flat = all(r['ratio'] <= PARAMS['side_max_ratio'] or r.get('mode') == 'line' for r in rotors)
    valid = flat and min(alignment) >= np.cos(np.deg2rad(PARAMS['side_axis_tolerance_deg']))
    normal = np.array([-direction[1], direction[0]])
    return dict(valid=bool(valid), reason='two_flat_aligned_components' if valid else 'two_is_not_proof_of_side_view',
                center=p.mean(0).tolist(), tangent=direction.tolist(),
                normal_candidates=[normal.tolist(), (-normal).tolist()],
                alignment=[float(a) for a in alignment], separation_px=distance,
                full_3d_plane_observed=False, physical_motor_identity_resolved=False)

def plane_axis(rotors):
    """Unsigned weak-perspective normal, with an explicit image-up sign prior."""
    normals = []
    for r in rotors:
        a = np.deg2rad(r['angle'])
        n = np.array([-np.sin(a), np.cos(a)])
        if n[1] > 0: n = -n
        normals.append(n * np.sqrt(max(0., 1. - r['ratio']**2)))
    vector = np.mean(normals, axis=0)
    agreement = np.linalg.norm(np.mean([unit(v) for v in normals], axis=0))
    return vector, float(agreement)

class BranchGeometry:
    def __init__(self):
        self.hubs = None; self.body = None; self.pose_t = None
        self.full_t = None; self.last_rates = {}; self.shapes = None

    def step(self, t, body, rotors):
        n = len(rotors); p = centres(rotors)
        if self.pose_t is not None and t - self.pose_t > PARAMS['model_expiry_s']:
            self.__init__()
        out = dict(observed_count=n, state='insufficient', model_centers=None,
                   observed_ids=list(range(n)), inferred_ids=[], confidence=0.,
                   axis_xy=None, sign_prior=None, reason='zero_or_one_observation',
                   observed_rotors=rotors, full_3d_pose=False)
        if n not in (2, 3, 4): return out
        if n == 2:
            line = two_axis(rotors); out['two_axis'] = line
            if not line['valid']:
                out.update(state='two_unresolved', reason=line['reason']); return out
            normal = np.array(line['normal_candidates'][0])
            if normal[1] > 0: normal = -normal
            out.update(state='two_side', reason=line['reason'], axis_xy=normal.tolist(),
                       sign_prior='image_up_assumption', confidence=.35,
                       model_centers=p.tolist(), observed_ids=[0, 1],
                       total_thrust_not_identifiable_from_two=True)
            return out
        if any(r.get('inferred_shape') or r.get('fov_clipped') or r.get('mode') != 'ellipse' for r in rotors):
            out.update(state=f'{n}_unresolved', reason='not_independent_complete_disks'); return out
        normal, agreement = plane_axis(rotors)
        if agreement < .85:
            out.update(state=f'{n}_unresolved', reason='ellipse_axes_disagree'); return out
        if n == 4:
            order = np.argsort(np.arctan2(*(p - p.mean(0))[:, ::-1].T))
            ordered = p[order]
            diagonal_error = np.linalg.norm(ordered[0] + ordered[2] - ordered[1] - ordered[3])
            singular = np.linalg.svd(p-p.mean(0), compute_uv=False)
            # Perspective does not preserve parallelograms. Four distinct hubs
            # may legitimately form a trapezoid (the reviewed C frame does).
            # Reject collapsed/concave constellations rather than imposing an
            # affine parallelogram on the image of a 3-D rectangle.
            if len(cv2.convexHull(p.astype(np.float32))) != 4 or singular[1]/max(singular[0], 1e-9) < .08:
                out.update(state='four_unresolved', reason='four_centres_collapsed_or_nonconvex'); return out
            a, b, c, d = np.c_[ordered, np.ones(4)]
            center_h = np.cross(np.cross(a, c), np.cross(b, d))
            projective_center = (center_h[:2]/center_h[2]).tolist()
            ids = np.arange(4)
            if self.hubs is not None and body.get('detected') and self.body is not None:
                expected = self.hubs + np.array(body['center']) - self.body
                alternatives = [(np.linalg.norm(p - expected[list(ids)], axis=1).mean(), ids)
                                for ids in permutations(range(4))]
                _, ids = min(alternatives, key=lambda q: q[0]); ids = np.array(ids)
                hubs = np.zeros((4, 2)); hubs[ids] = p
            else: hubs = p.copy()
            out.update(state='four', reason='four_measured_convex_coplanar_hub_hypothesis', observed_ids=ids.tolist(),
                       model_centers=hubs.tolist(), confidence=.8, axis_xy=normal.tolist(),
                       sign_prior='image_up_assumption', diagonal_affine_error_px=float(diagonal_error),
                       projected_diagonal_intersection=projective_center,
                       image_ellipse_centres_approximate_physical_hubs=True)
            self.full_t = t
            shapes = np.zeros((4, 2, 2))
            for index, rotor_id in enumerate(ids): shapes[rotor_id] = ellipse_tensor(rotors[index])
        else:
            completed = None
            if self.hubs is not None and self.pose_t is not None and body.get('detected') and self.body is not None:
                expected = self.hubs + np.array(body['center']) - self.body
                options = [(float(np.mean(np.linalg.norm(p - expected[list(ids)], axis=1)**2)), ids)
                           for ids in permutations(range(4), 3)]
                options.sort(key=lambda q: q[0]); cost, ids = options[0]
                A = cv2.getAffineTransform(expected[list(ids)].astype(np.float32), p.astype(np.float32))
                singular = np.linalg.svd(A[:, :2], compute_uv=False)
                good = (np.sqrt(cost) <= max(8., .6 * np.median([r['major'] for r in rotors]))
                        and np.linalg.det(A[:, :2]) > 0 and singular.min() > .6 and singular.max() < 1.6
                        and options[1][0] - cost > 4.)
                if good and self.shapes is not None:
                    predicted_shapes = np.array([A[:, :2] @ shape @ A[:, :2].T for shape in self.shapes])
                    axis_ratios = []
                    for observed, rotor_id in zip(rotors, ids):
                        predicted_axes = np.sqrt(np.maximum(np.linalg.eigvalsh(predicted_shapes[rotor_id]), 1e-9))*2
                        axis_ratios.append(np.array([observed['minor'], observed['major']])/predicted_axes)
                    if np.min(axis_ratios) < .65 or np.max(axis_ratios) > 1.5:
                        good = False; out['previous_model_shape_rejected'] = True
                if good:
                    completed = np.c_[expected, np.ones(4)] @ A.T
                    missing = list(set(range(4)) - set(ids))
                    out.update(observed_ids=list(ids), inferred_ids=missing,
                               completion='recent_model_affine_transport', confidence=.6)
                    shapes = predicted_shapes if self.shapes is not None else np.repeat(np.mean([ellipse_tensor(r) for r in rotors], axis=0)[None], 4, axis=0)
                    for observed, rotor_id in zip(rotors, ids): shapes[rotor_id] = ellipse_tensor(observed)
            if completed is None:
                inferred = three_affine_hypotheses(rotors); out['three_hypotheses'] = inferred
                if not inferred['valid']:
                    out.update(state='three_unresolved', reason=inferred['reason']); return out
                completed = np.vstack([p, inferred['inferred_center']])
                out.update(observed_ids=[0, 1, 2], inferred_ids=[3],
                           completion='three_ellipse_metric_weak_perspective', confidence=.45)
                shapes = np.array([ellipse_tensor(r) for r in rotors]+[np.mean([ellipse_tensor(r) for r in rotors], axis=0)])
            out.update(state='three_completed', reason='position_completed_frequency_not_observed',
                       model_centers=completed.tolist(), axis_xy=normal.tolist(),
                       sign_prior='image_up_assumption', inferred_center_is_measurement=False)
            hubs = completed
        self.hubs = hubs.copy(); self.pose_t = t
        self.shapes = shapes.copy()
        self.body = np.array(body['center']) if body.get('detected') else None
        return out

def rotor_rates(q, rotors, t, velocity, w, unique_pixels=False):
    """Unique current pixel votes in independently measured rotor regions."""
    if len(q):
        if not unique_pixels:
            pixel = q['y'].astype(np.int64) * w + q['x']
            _, index = np.unique(pixel[::-1], return_index=True)
            q = q[len(q) - 1 - index]
        q = q[q['t'] >= round(t * 1e6) - 20000]
    xy = np.c_[q['x'], q['y']].astype(float)
    if len(q): xy += (t - q['t'] / 1e6)[:, None] * np.asarray(velocity)
    membership = []
    for r in rotors:
        d = xy - [r['cx'], r['cy']]; a = np.deg2rad(r['angle'])
        u = d[:, 0]*np.cos(a) + d[:, 1]*np.sin(a)
        v = -d[:, 0]*np.sin(a) + d[:, 1]*np.cos(a)
        membership.append((u / max(r['major']*.55, 1))**2 +
                          (v / max(r['minor']*.65, 2))**2 <= 1)
    ambiguous = np.sum(membership, axis=0) > 1 if membership else np.zeros(len(q), bool)
    result = []
    for inside in membership:
        f = q['frequency'][inside & ~ambiguous].astype(float)
        summary = dict(valid=False, pixels=len(f), frequency_hz=None,
                       mechanical_rpm_measured=False, overlapping_pixel_votes_excluded=int(np.sum(inside & ambiguous)))
        if len(f) >= PARAMS['frequency_min_pixels']:
            bins = np.floor(np.log(f / 60.) / np.log(1.04)).astype(int)
            hist = np.bincount(np.maximum(bins, 0)); k = int(np.argmax(hist))
            chosen = abs(bins - k) <= 1; fraction = float(chosen.mean())
            hz = float(np.median(f[chosen]))
            summary.update(valid=fraction >= PARAMS['frequency_mode_fraction'],
                           frequency_hz=hz, mode_fraction=fraction,
                           conditional_rpm_if_two_blades=hz*30.,
                           blade_count_assumed=2, harmonic_identity_unverified=True)
        result.append(summary)
    return result
