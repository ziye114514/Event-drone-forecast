"""Reuse already-computed current KLT displacement; no new optical-flow pass.

Uses existing velocity covariance intersection and current-flow derivative.
The shared position KF and its eligibility stay unchanged. Flow-derived net
acceleration is unobserved total force when rotor periodicity is unavailable.
"""
import copy
import numpy as np
from live_local_flow import box,velocity_coefficient
from live_flow_acceleration import FlowAcceleration


class SharedImageMotion:
    def __init__(self,reanchor_velocity=True):self.previous=None;self.derivative=FlowAcceleration();self.reanchor_velocity=reanchor_velocity

    def step(self,t,coefficient,velocity_covariance,body,tracking):
        prior=self.previous;self.previous=(t,copy.deepcopy(body))
        observation=dict(valid=False,reason='no_qualified_current_shared_translation')
        current_box=box(body)
        old_box=None if prior is None else box(prior[1]);upgrade=False
        new_anchor=bool(prior is not None and current_box is not None and body.get('confidence',0)>=.45 and
            body.get('measurement_source') in ('current_NN','current_KLT_from_delayed_NN') and
            body.get('nn_observed_t',t)!=prior[1].get('nn_observed_t'))
        if (self.reanchor_velocity and prior is not None and old_box is None and current_box is not None and
            prior[1].get('low_confidence') and prior[1].get('confidence',0)>=.25 and
            new_anchor):
            old_box=box(dict(prior[1],low_confidence=False));upgrade=old_box is not None
        if (prior is not None and current_box is not None and old_box is not None and
            tracking.get('reason')=='current_image_translation' and tracking.get('points',0)>=8 and
            0.<t-prior[0]<=.08000001):
            dt=t-prior[0];delta=np.asarray(tracking['delta'],float)
            covariance=np.asarray(tracking.get('displacement_covariance_px2',np.eye(2)*4.),float)
            similarity=tracking.get('similarity_motion') or {}
            if similarity.get('valid'):
                delta=np.asarray(similarity['displacement'],float)
                covariance=np.asarray(similarity['displacement_covariance_px2'],float)
            actual=np.asarray(body['center'])-np.asarray(prior[1]['center']);mismatch=delta-actual
            consistency=float(mismatch@np.linalg.solve(covariance+np.eye(2)*72.,mismatch))
            translated=old_box+np.tile(delta,2)
            intersection=np.prod(np.maximum(0.,np.minimum(translated[2:],current_box[2:])-np.maximum(translated[:2],current_box[:2])))
            union=np.prod(translated[2:]-translated[:2])+np.prod(current_box[2:]-current_box[:2])-intersection
            overlap=float(intersection/max(union,1e-9))
            areas=[float(np.prod(translated[2:]-translated[:2])),float(np.prod(current_box[2:]-current_box[:2]))]
            containment=float(intersection/max(min(areas),1e-9));area_ratio=max(areas)/max(min(areas),1e-9)
            # A reliable partial box can grow when more of the same body is
            # visible. Containment of its measured track region is stronger
            # evidence here than IoU, which penalizes the newly exposed area.
            # Do not give this extra exception to an old weak semantic box.
            contained_strong=bool(not prior[1].get('low_confidence') and prior[1].get('confidence',0)>=.45 and
                containment>=.8 and area_ratio<=4.)
            anchor_compatible=bool(self.reanchor_velocity and new_anchor and (overlap>=.5 or contained_strong) and
                np.linalg.eigvalsh(covariance).max()<=36.)
            # A newly strong NN anchor corrects the position gauge. Its center
            # shift is not a velocity measurement. The already measured tracks
            # can still constrain velocity if both current boxes overlap.
            if (not upgrade and consistency<=5.991464547107979) or anchor_compatible:
                observation=dict(valid=True,reason='reused_current_bidirectional_translation',
                    velocity=(delta/dt).tolist(),velocity_covariance=(covariance/(dt*dt)).tolist(),
                    displacement=delta.tolist(),midpoint_t=.5*(t+prior[0]),box_displacement_chi2=consistency,
                    original_tracks_reused=True,not_independent_of_body_localization=True,
                    covariance_source='observed_full_KLT_scatter' if 'displacement_covariance_px2' in tracking else 'legacy2px_proxy_for_saved_records_only')
                if similarity.get('valid'):observation.update(covariance_source='full_similarity_residual_scatter_plus_localization_floor',
                    current_similarity_reused=True,evaluation_center_source=similarity.get('evaluation_center_source'))
                if anchor_compatible and (upgrade or consistency>5.991464547107979):observation.update(
                    reason='current_strong_NN_anchor_upgrade_with_current_tracks' if upgrade else 'current_strong_NN_visibility_reanchor_with_current_tracks',
                    anchor_upgrade=upgrade,visibility_reanchor=not upgrade,current_projected_box_IoU=overlap,
                    current_projected_box_containment=containment,current_box_area_ratio=area_ratio,
                    position_gauge_correction_not_treated_as_velocity=mismatch.tolist(),
                    previous_weak_box_not_promoted_to_strong_detection=True)
        base_acceleration=np.zeros(2) if coefficient is None else np.asarray(coefficient['acceleration'])
        value,velocity_info=velocity_coefficient(coefficient,velocity_covariance,observation,t)
        value,acceleration_info=self.derivative.step(t,value,body,observation,'position-consistent')
        correction=np.zeros(2)
        if velocity_info.get('applied') and acceleration_info.get('applied'):
            # The first CI used its prior acceleration to align midpoint flow.
            # Replace only that half-step term using the newly qualified slope;
            # do not fuse the same flow observation twice.
            weight=velocity_info['baseline_weight'];P=np.asarray(velocity_covariance);R=np.asarray(observation['velocity_covariance'])
            invP,invR=np.linalg.inv(P),np.linalg.inv(R);C=np.linalg.inv(weight*invP+(1-weight)*invR)
            correction=C@((1-weight)*invR)@((t-observation['midpoint_t'])*(np.asarray(value['acceleration'])-base_acceleration))
            value=dict(value,velocity=(np.asarray(value['velocity'])+correction).tolist())
        return value,dict(observation=observation,velocity=velocity_info,acceleration=acceleration_info,
            midpoint_alignment_correction_px_s=correction.tolist(),new_flow_passes=0,new_position_filters=0,
            position_and_validity_unchanged=True,acceleration_is_not_measured_rotor_thrust=True)
