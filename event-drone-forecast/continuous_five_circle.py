"""Shared affine image template, partial-information updates and bounded relock.

Uses existing four-disc initialization, assignment/scale bounds and conic gates.
Posterior covariance is a heuristic information measure, not calibrated truth.
"""
from itertools import permutations
import copy
import numpy as np
from five_circle_model import FiveCircleModel,diagonal_center,ellipse_tensors


class ContinuousFiveCircle:
    def __init__(self,current_shape_feedback=True):
        self.core=FiveCircleModel(3);self.template=None;self.beta=None;self.P=None
        self.body=None;self.t=None;self.last_full=None;self.epoch=None
        self.last_update=None
        self.full_observations=0
        self.last_informative=None;self.informative_source=None
        self.informative_independent_centres=0
        self.current_shape_feedback=current_shape_feedback

    def _seed(self,t,body,g):
        hubs=np.asarray(g['model_centers']);center=diagonal_center(hubs)
        scale=max(1.,np.sqrt(np.mean(np.sum((hubs-center)**2,axis=1))))
        self.template=(hubs-center)/scale
        # Template coordinates stay fixed until a new seed. Reuse the exact
        # same design matrix instead of rebuilding it in every hub query.
        self.design=np.column_stack((np.ones(4),self.template))
        self.beta=np.array([center[0],scale,0.,center[1],0.,scale])
        self.P=np.diag([36.,16.,16.,36.,16.,16.])
        self.body=np.asarray(body['center']);self.t=t;self.last_full=t;self.last_update=t
        self.shapes=self.core.shapes.copy()
        self.last_informative=t;self.informative_source='current_four'
        self.informative_independent_centres=4

    def _hubs(self,beta=None):
        b=self.beta if beta is None else beta;D=self.design
        return np.column_stack((D@b[:3],D@b[3:]))

    def guidance_reference(self,t,max_age):
        """A current rank6 observation can refresh guidance, not axis/coasting.

        Propagate uncertainty to ALL template hubs, including the hidden one.
        Reuse148's .15-major/4px centre gate as the uncertainty ceiling.
        """
        if self.beta is None or self.last_informative is None:
            return dict(eligible=False,reason='no_informative_template')
        age=t-self.last_informative
        if age>max_age+1e-9:
            return dict(eligible=False,reason='informative_observation_expired',reference_age_s=age)
        D=self.design;H=np.zeros((4,2,6))
        H[:,0,:3]=D;H[:,1,3:]=D
        predicted=self.P+np.diag([36.,225.,225.,36.,225.,225.])*max(0.,t-self.t)
        covariance=H@predicted@H.transpose(0,2,1)
        sigma=np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:,-1],0))
        major=2*np.sqrt(np.maximum(np.linalg.eigvalsh(self.shapes)[:,-1],1e-9))
        limit=np.maximum(4.,.15*major);ratio=float(np.max(sigma/limit))
        valid=bool(np.isfinite(ratio) and ratio<=1.)
        return dict(eligible=valid,reason='recent_rank6_and_bounded_hubs' if valid else 'hub_uncertainty_too_large',
            reference_age_s=age,reference_source=self.informative_source,
            independent_current_centres=self.informative_independent_centres,
            hub_sigma_proxy_px=sigma.tolist(),hub_gate_px=limit.tolist(),maximum_uncertainty_gate_ratio=ratio)

    def step(self,t,body,geometry):
        old=self.core.step(t,body,geometry);g=copy.deepcopy(old)
        strong=bool(body.get('detected') and not body.get('low_confidence'))
        epoch=g['geometry_epoch']
        if self.epoch is not None and epoch!=self.epoch:
            self.template=None;self.beta=None;self.P=None;self.last_full=None
            self.full_observations=0
            self.last_informative=None;self.informative_source=None
            self.informative_independent_centres=0
        self.epoch=epoch
        if g['mode']=='four' and strong and self.beta is None:
            self._seed(t,body,g)
            self.full_observations+=1
            g['continuous']=dict(updated=True,measurement_rank=6,source='current_four',center_estimate_is_prior_conditioned=False)
            return g
        if self.beta is None or not strong:
            g['continuous']=dict(updated=False,reason='no_confirmed_template_or_strong_body',measurement_rank=0)
            return g
        dt=t-self.t;current_body=np.asarray(body['center']);delta=current_body-self.body
        self.beta[[0,3]]+=delta;self.body=current_body;self.t=t
        # Body remains tracked, but hidden shape directions grow uncertain.
        self.P+=np.diag([36.,225.,225.,36.,225.,225.])*max(dt,0.)
        if g['mode']=='four':
            self.full_observations+=1
            D=self.design;H=np.zeros((8,6));H[::2,:3]=D;H[1::2,3:]=D
            z=np.asarray(g['model_centers']).ravel();R=np.eye(8)*9.
            S=H@self.P@H.T+R;K=np.linalg.solve(S,H@self.P).T
            self.beta+=K@(z-H@self.beta);J=np.eye(6)-K@H;self.P=J@self.P@J.T+K@R@K.T
            self.shapes=self.core.shapes.copy();self.last_full=t;self.last_update=t
            self.last_informative=t;self.informative_source='current_four'
            self.informative_independent_centres=4
            g.update(current_geometric_center=self.beta[[0,3]].tolist(),model_centers=self._hubs().tolist(),
                center_source='shared_affine_four_information',current_geometric_center_is_prior_conditioned=True)
            self.core.hubs=self._hubs();self.core.anchor=current_body.copy()
            g['continuous']=dict(updated=True,measurement_rank=6,source='current_four_same_template',
                center_estimate_is_prior_conditioned=True,template_retained=True)
            g['five_circles']=dict(body=g['current_geometric_center'],rotors=g['model_centers'])
            return g
        rotors=[r for r in geometry.get('observed_rotors',[]) if r.get('current_observation',True)
            and not r.get('fov_clipped') and not r.get('association_ambiguous') and
            (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
        info=dict(updated=False,reason='current_partial_geometry_unqualified',measurement_rank=0,
            last_full_age_s=t-self.last_full,template_retained=True)
        # Two nonflat centres constrain only four of six affine image quantities.
        # Thin groups supply an axis elsewhere; never fit a full plane to a line.
        if len(rotors) in (2,3) and min(r['ratio'] for r in rotors)>.16 and t-self.last_update<=2.:
            p=np.array([[r['cx'],r['cy']] for r in rotors]);expected=self._hubs()
            assignments=np.array(list(permutations(range(4),len(p))))
            costs=np.mean(np.sum((p[None]-expected[assignments])**2,axis=2),axis=1)
            order=np.argsort(costs);ids=assignments[order[0]];margin=float(costs[order[1]]-costs[order[0]])
            gate=max(8.,.6*np.median([r['major'] for r in rotors]))
            if np.sqrt(costs[order[0]])<=gate and margin>4.:
                D=self.design[ids]
                H=np.zeros((2*len(ids),6));H[::2,:3]=D;H[1::2,3:]=D
                z=p.ravel();variance=np.repeat([max(2.,r.get('center_sigma_px',3.))**2 for r in rotors],2)
                S=H@self.P@H.T+np.diag(variance);innovation=z-H@self.beta
                nis=float(innovation@np.linalg.solve(S,innovation))
                info.update(assignment=ids.tolist(),assignment_margin_px2=margin,innovation_nis=nis,
                    measurement_rank=int(np.linalg.matrix_rank(H)))
                if nis<=16.812: #99% chi-square for at most six scalar constraints
                    K=np.linalg.solve(S,H@self.P).T;candidate=self.beta+K@innovation
                    A=np.array([self.beta[1:3],self.beta[4:6]])
                    B=np.array([candidate[1:3],candidate[4:6]])
                    transform=B@np.linalg.inv(A);scales=np.linalg.svd(transform,compute_uv=False)
                    good=np.linalg.det(transform)>0 and scales.min()>=.6 and scales.max()<=1.6
                    warped=np.array([transform@shape@transform.T for shape in self.shapes])
                    ratios=[]
                    for r,i in zip(rotors,ids):
                        axes=2*np.sqrt(np.maximum(np.linalg.eigvalsh(warped[i]),1e-9))
                        ratios.extend(np.array([r['minor'],r['major']])/axes)
                    good=bool(good and min(ratios)>=.65 and max(ratios)<=1.5)
                    if good:
                        self.beta=candidate;J=np.eye(6)-K@H
                        self.P=J@self.P@J.T+(K*variance)@K.T
                        self.shapes=warped;self.last_update=t
                        # Match the existing FiveCircleModel._three behavior:
                        # accepted CURRENT conics replace transported shapes.
                        # This does not add information to unobserved hubs or
                        # turn a two-centre update into a rank6 observation.
                        if self.current_shape_feedback:
                            measured_shapes=ellipse_tensors(rotors)
                            # CoHelix's slow ellipse observer uses a0.2s time
                            # constant to avoid short fits moving geometry.
                            # Apply the same fixed lag to SPD image tensors.
                            alpha=1-np.exp(-max(dt,0.)/.2)
                            for shape,rotor_id in zip(measured_shapes,ids):
                                self.shapes[rotor_id]=(1-alpha)*self.shapes[rotor_id]+alpha*shape
                        g.update(mode='three' if len(ids)==3 else 'partial_two',model_centers=self._hubs().tolist(),
                            current_geometric_center=self.beta[[0,3]].tolist() if len(ids)==3 else None,
                            center_source='shared_affine_partial_information' if len(ids)==3 else None,
                            joint_center_is_current_measurement=len(ids)==3,
                            observed_model_ids=ids.tolist(),inferred_model_ids=sorted(set(range(4))-set(ids)),
                            geometric_weight=.35 if len(ids)==3 else 0.,
                            reason='current_centres_update_observable_shared_state',
                            current_geometric_center_is_prior_conditioned=True)
                        self.core.hubs=self._hubs();self.core.shapes=self.shapes.copy();self.core.anchor=current_body.copy()
                        # A partial measurement is not a new full calibration.
                        self.core.seen=t;self.core.mode=g['mode']
                        info.update(updated=True,reason='bounded_partial_information_update',
                            center_estimate_is_prior_conditioned=True,hidden_frequencies_added=0,
                            current_conic_shapes_updated=len(ids) if self.current_shape_feedback else 0)
                        info['current_conic_shape_feedback']='ema_tau_0.2s' if self.current_shape_feedback else 'transport_only'
                        independent=sum(not r.get('shared_arc_recovered',False) for r in rotors)
                        # At least two current anchors must be independent of
                        # template-guided search. A line or two-point update
                        # cannot keep unobserved shape directions "fresh".
                        if len(ids)==3 and independent>=2 and np.linalg.matrix_rank(H)==6:
                            self.last_informative=t;self.informative_source='three_with_current_independent_anchors'
                            self.informative_independent_centres=independent
                            info['guidance_reference_refreshed']=True
                    else:info['reason']='partial_shape_transport_rejected'
                else:info['reason']='partial_innovation_rejected'
            else:info['reason']='partial_assignment_ambiguous_or_far'
        if not info['updated']:
            # Retain diagnostic template through axis-only stages. No current
            # centre measurement is fabricated, so centre CI cannot consume it.
            if t-self.last_update<=2.:
                g['model_centers']=self._hubs().tolist();g['inferred_model_ids']=list(range(4))
        sigma=float(np.sqrt(max(np.linalg.eigvalsh(self.P[np.ix_([0,3],[0,3])]))))
        info['center_sigma_proxy_px']=sigma
        info['unobserved_template_points_are_current_measurements']=False
        g.update(model_conditioned_center=self.beta[[0,3]].tolist(),
            model_conditioned_center_covariance=self.P[np.ix_([0,3],[0,3])].tolist(),
            model_conditioned_center_supported_by_current_partial_centres=bool(info['updated']),
            model_conditioned_center_is_current_measurement=False)
        if info['updated'] and g['mode']=='partial_two':g['geometric_weight']=.35
        g['continuous']=info;g['five_circles']=dict(body=g['current_geometric_center'],rotors=g['model_centers'])
        return g

    def observe_axis(self,g,normal,confidence,t):
        """One current line-angle constraint; hidden lengths remain uncertain."""
        if self.beta is None or self.t!=t or confidence<.45:return g
        n=np.asarray(normal,float);target=float(np.arctan2(-n[0],n[1]))
        C=self.template.T@self.template/4
        def angle(b):
            A=np.array([b[1:3],b[4:6]]);S=A@C@A.T
            return .5*np.arctan2(2*S[0,1],S[0,0]-S[1,1])
        wrap=lambda a:(a+np.pi/2)%np.pi-np.pi/2
        before=angle(self.beta);H=np.zeros(6)
        for i in (1,2,4,5):
            plus=self.beta.copy();minus=self.beta.copy();plus[i]+=.01;minus[i]-=.01
            H[i]=wrap(angle(plus)-angle(minus))/.02
        residual=wrap(target-before);variance=np.deg2rad(5./max(confidence,.45))**2
        S=float(H@self.P@H+variance);accepted=abs(residual)<=np.pi/4 and residual*residual/S<=6.635
        info=dict(accepted=False,measurement_rank=1,innovation_deg=float(np.rad2deg(residual)),
            current_line_does_not_measure_hidden_hubs=True,unobserved_lengths_not_reset=True)
        if accepted:
            K=self.P@H/S;candidate=self.beta+K*residual
            A=np.array([self.beta[1:3],self.beta[4:6]]);B=np.array([candidate[1:3],candidate[4:6]])
            T=B@np.linalg.inv(A);scales=np.linalg.svd(T,compute_uv=False)
            if np.linalg.det(T)>0 and scales.min()>=.6 and scales.max()<=1.6:
                self.beta=candidate;J=np.eye(6)-np.outer(K,H)
                self.P=J@self.P@J.T+np.outer(K,K)*variance
                self.shapes=np.array([T@shape@T.T for shape in self.shapes]);self.last_update=t
                self.core.hubs=self._hubs();self.core.shapes=self.shapes.copy();self.core.anchor=self.body.copy()
                g['model_centers']=self._hubs().tolist();g['inferred_model_ids']=list(range(4))
                g['five_circles']=dict(body=None,rotors=g['model_centers'])
                info.update(accepted=True,posterior_axis_residual_deg=float(np.rad2deg(wrap(target-angle(self.beta)))))
        g['continuous_axis']=info
        return g
