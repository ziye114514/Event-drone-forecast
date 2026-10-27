"""Five-circle image structure with current-evidence visibility transitions.

Strategy 3 is the user's structural/visibility mechanism, NOT detector revision
accumulated_v3_consistency. Existing 146/147 geometry supplies the observations;
missing model points never become observations or measured rotor frequencies.
"""
from itertools import permutations
import copy
import cv2
import numpy as np

PARAMS = dict(relock_observations=2, model_coast_s=.40, identity_gap_s=.50,
    assignment_major_gate=.60, assignment_min_gate_px=8., affine_min_scale=.60,
    affine_max_scale=1.60, assignment_margin_squared_px=4.,
    inherited='146/147 current arcs and two-axis checks;149 bounded coasting/relock',
    coordinates='sensor pixels; fifth circle is geometric centre, not measured mass centre',
    inferred_points_are_observations=False, full_3d_pose=False)


def collinear_axis(rotors):
    """Current3+ components may support an axis without resolving motor IDs.

    Reuse the existing .16 flatness,15degree alignment,.08 collapse and12px
    span conventions. Identity-ambiguous CURRENT regions may vote for an axis,
    but never for a labelled hub or a missing rotor frequency.
    """
    current=[r for r in rotors if r.get('current_observation',True) and not r.get('fov_clipped')]
    if len(current)<3:return dict(valid=False,reason='need_three_current_axis_regions')
    p=np.array([[r['cx'],r['cy']] for r in current]);center=p.mean(0)
    _,singular,V=np.linalg.svd(p-center,full_matrices=False);direction=V[0]
    ratio=float(singular[1]/max(singular[0],1e-9))
    alignment=[abs(float(direction@np.array([np.cos(np.deg2rad(r['angle'])),np.sin(np.deg2rad(r['angle']))]))) for r in current]
    flat=float(np.median([r['ratio'] for r in current]))
    extent=float(np.ptp((p-center)@direction))
    valid=ratio<.08 and flat<=.16 and min(alignment)>=np.cos(np.deg2rad(15.)) and extent>=12.
    n=np.array([-direction[1],direction[0]])
    return dict(valid=bool(valid),reason='current_flat_collinear_regions' if valid else 'regions_not_flat_collinear_aligned',
        center=center.tolist(),tangent=direction.tolist(),normal_candidates=[n.tolist(),(-n).tolist()],
        component_count=len(current),centre_rank_ratio=ratio,median_flatness=flat,
        alignment=alignment,span_px=extent,physical_motor_identity_resolved=False)


def diagonal_center(points):
    p=np.asarray(points,float)
    order=np.argsort(np.arctan2(*(p-p.mean(0))[:,::-1].T))
    a,b,c,d=np.c_[p[order],np.ones(4)]
    q=np.cross(np.cross(a,c),np.cross(b,d))
    if abs(q[2])<1e-9:return None
    return q[:2]/q[2]


def ellipse_tensors(rotors):
    result=[]
    for r in rotors:
        angle=np.deg2rad(r['angle']);c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,-s],[s,c]])
        result.append(R@np.diag([(r['major']/2)**2,(r['minor']/2)**2])@R.T)
    return np.asarray(result)


class FiveCircleModel:
    """1 independent centroid; 2 complete rigid only; 3 visibility switching.

    Three-point completion uses the established recent-model affine transport
    with bounded singular values and shape checks, not an arbitrary perspective
    three-point solution. Two groups only supply an unsigned image axis.
    """
    def __init__(self,strategy=3):
        if strategy not in (1,2,3):raise ValueError('strategy must be1,2,3')
        self.strategy=strategy;self.last_t=None;self.last_body_t=None
        self.identity=None;self.epoch=0;self._clear()

    def _clear(self):
        self.hubs=None;self.shapes=None;self.anchor=None;self.seen=None
        self.pending=None;self.pending_count=0;self.mode='uninitialized'

    def _expected(self,body_center):
        delta=body_center-self.anchor if body_center is not None and self.anchor is not None else 0.
        return None if self.hubs is None else self.hubs+delta

    def _three(self,points,rotors,expected):
        if expected is None:return None
        costs=sorted((float(np.mean(np.sum((points-expected[list(ids)])**2,axis=1))),ids)
                     for ids in permutations(range(4),3))
        cost,ids=costs[0];gate=max(8.,.60*np.median([r['major'] for r in rotors]))
        if np.sqrt(cost)>gate or costs[1][0]-cost<=4.:return None
        A=cv2.getAffineTransform(expected[list(ids)].astype('float32'),points.astype('float32'))
        scales=np.linalg.svd(A[:,:2],compute_uv=False)
        if np.linalg.det(A[:,:2])<=0 or scales.min()<.6 or scales.max()>1.6:return None
        # Same shape-transport check as146; no hidden ellipse fitted as evidence.
        transformed=np.array([A[:,:2]@S@A[:,:2].T for S in self.shapes])
        ratios=[]
        for r,i in zip(rotors,ids):
            axes=2*np.sqrt(np.maximum(np.linalg.eigvalsh(transformed[i]),1e-9))
            ratios.extend(np.array([r['minor'],r['major']])/axes)
        if min(ratios)<.65 or max(ratios)>1.5:return None
        hubs=np.c_[expected,np.ones(4)]@A.T
        for r,i in zip(ellipse_tensors(rotors),ids):transformed[i]=r
        return hubs,transformed,list(ids)

    def step(self,t,body,geometry):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Strictly increasing cutoffs required')
        self.last_t=t
        detected=bool(body.get('detected') and not body.get('low_confidence'))
        center=np.asarray(body['center'],float) if detected else None
        identity=(body.get('body_track_epoch'),body.get('body_track_id'))
        changed=self.identity is not None and identity[1] is not None and identity!=self.identity
        if changed or (detected and self.last_body_t is not None and t-self.last_body_t>.5):
            self._clear();self.epoch+=1
        if identity[1] is not None:self.identity=identity
        if detected:self.last_body_t=t
        rotors=[r for r in geometry.get('observed_rotors',[]) if r.get('current_observation',True)
                and not r.get('association_ambiguous') and r.get('association_state')!='relock_pending']
        n=len(rotors);points=np.array([[r['cx'],r['cy']] for r in rotors],float).reshape(-1,2)
        before=self.mode;expected=self._expected(center)
        out=dict(strategy=self.strategy,observed_count=n,geometry_epoch=self.epoch,
            current_region_count=sum(r.get('current_observation',True) for r in geometry.get('observed_rotors',[])),
            observed_centers=points.tolist(),current_geometric_center=None,
            center_source=None,model_centers=None,observed_model_ids=[],inferred_model_ids=[],
            model_points_are_measurements=False,full_3d_pose=False,axis_xy=None,
            axis_unsigned=True,geometric_weight=0.,reason='no_supported_current_structure',
            inferred_rotor_frequencies=[],model_age_s=None if self.seen is None else t-self.seen)
        if self.strategy==1:
            self.mode='independent' if n else 'insufficient'
            if n:
                out.update(current_geometric_center=points.mean(0).tolist(),
                    center_source='visible_rotor_centroid_control',geometric_weight=1.)
        else:
            independent=[r for r in rotors if not r.get('fov_clipped') and
                (r.get('mode')=='ellipse' or r.get('curvature_validated') or r.get('arc_center_qualified')) and
                (not r.get('inferred_shape') or r.get('curvature_validated') or r.get('arc_center_qualified'))]
            full=(n==4 and len(independent)==4 and geometry.get('state')=='four'
                  and geometry.get('confidence',0.)>0)
            if full:
                order=np.argsort(np.arctan2(*(points-points.mean(0))[:,::-1].T))
                candidate=points[order];shapes=ellipse_tensors(rotors)[order]
                reference=expected if expected is not None else (None if self.pending is None else self.pending[0])
                if reference is not None:
                    choices=sorted((float(np.mean(np.linalg.norm(candidate[list(ids)]-reference,axis=1))),ids)
                                   for ids in permutations(range(4)))
                    cost,ids=choices[0];candidate=candidate[list(ids)];shapes=shapes[list(ids)]
                else:cost=0.
                gate=max(8.,.6*np.median([r['major'] for r in rotors]))
                # A recovered/full constellation is confirmed twice; degrade
                # immediately, never wait for hidden points to become visible.
                recent_compatible=(self.hubs is not None and self.seen is not None and t-self.seen<=.4 and cost<=gate)
                if recent_compatible and self.shapes is not None:
                    ratios=np.sqrt(np.linalg.eigvalsh(shapes)/np.maximum(np.linalg.eigvalsh(self.shapes),1e-9))
                    recent_compatible=bool(ratios.min()>=.65 and ratios.max()<=1.5)
                need_confirmation=not recent_compatible
                out['recent_model_reused']=recent_compatible and before not in ('four','three')
                if need_confirmation:
                    compatible=False
                    if self.pending is not None:
                        prev,prev_body=self.pending
                        delta=center-prev_body if center is not None and prev_body is not None else 0.
                        best=min(float(np.mean(np.linalg.norm(candidate[list(ids)]-prev-delta,axis=1)))
                                 for ids in permutations(range(4)))
                        compatible=best<=gate
                    self.pending_count=self.pending_count+1 if compatible else 1
                    self.pending=(candidate.copy(),None if center is None else center.copy())
                if not need_confirmation or self.pending_count>=2:
                    joint=diagonal_center(candidate)
                    if joint is not None:
                        self.hubs=candidate;self.shapes=shapes;self.anchor=center;self.seen=t
                        self.pending=None;self.pending_count=0;self.mode='four'
                        out.update(current_geometric_center=joint.tolist(),center_source='four_diagonal_intersection',
                            model_centers=candidate.tolist(),observed_model_ids=list(range(4)),
                            axis_xy=geometry.get('axis_xy'),geometric_weight=float(geometry['confidence']),
                            reason='four_current_hubs_confirmed')
                    else:self.mode='unresolved'
                else:
                    self.mode='relock_pending'
                    # The common geometric centre of four CURRENT hubs does
                    # not require persistent motor labels. Keep the template
                    # unconfirmed, but expose this lower-confidence measurement.
                    joint=diagonal_center(candidate)
                    if joint is not None:
                        out.update(current_geometric_center=joint.tolist(),
                            center_source='current_four_center_template_unconfirmed',
                            model_centers=candidate.tolist(),observed_model_ids=list(range(4)),
                            geometric_weight=.4,reason='four_current_center_without_template_lock')
            elif self.strategy==3 and n==3 and len(independent)==3 and min(r['ratio'] for r in rotors)>.16 and self.seen is not None and t-self.seen<=.4:
                self.pending=None;self.pending_count=0
                completed=self._three(points,rotors,expected)
                if completed is not None:
                    hubs,shapes,ids=completed;joint=diagonal_center(hubs)
                    if joint is not None:
                        self.hubs=hubs;self.shapes=shapes;self.anchor=center;self.seen=t;self.mode='three'
                        out.update(current_geometric_center=joint.tolist(),center_source='three_recent_model_affine',
                            model_centers=hubs.tolist(),observed_model_ids=ids,
                            inferred_model_ids=sorted(set(range(4))-set(ids)),
                            axis_xy=geometry.get('axis_xy'),geometric_weight=.6,
                            reason='three_current_hubs_hidden_hub_is_prediction')
                    else:self.mode='unresolved'
                else:self.mode='three_unresolved'
            else:
                self.pending=None;self.pending_count=0
                line=geometry.get('two_axis',{})
                if self.strategy==3 and not line.get('valid'):
                    line=collinear_axis(geometry.get('observed_rotors',[]))
                if self.strategy==3 and line.get('valid'):
                    self.mode='edge_on'
                    out.update(axis_xy=line['normal_candidates'][0],geometric_weight=.35,
                        reason='current_flat_aligned_axis_without_motor_identity',axis_support=line)
                else:self.mode='coasting' if self.seen is not None and t-self.seen<=.4 else 'insufficient'
            if out['model_centers'] is None and expected is not None and self.seen is not None and t-self.seen<=.4:
                out['model_centers']=expected.tolist();out['inferred_model_ids']=list(range(4))
        out.update(mode=self.mode,previous_mode=before,mode_changed=self.mode!=before,
            model_age_s=None if self.seen is None else t-self.seen,
            joint_center_is_current_measurement=out['current_geometric_center'] is not None,
            body_fallback=center.tolist() if center is not None else None,
            body_fallback_is_geometric_measurement=False)
        if out['current_geometric_center'] is not None:
            out['five_circles']=dict(body=out['current_geometric_center'],rotors=out['model_centers'])
        else:out['five_circles']=dict(body=None,rotors=out['model_centers'])
        return out
