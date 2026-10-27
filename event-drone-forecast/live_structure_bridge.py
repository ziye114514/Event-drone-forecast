"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""
import copy,numpy as np
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_measurement_kf import MeasurementKF

PARAMS=dict(tracker_hz=25.,source_publication_hz=5.,body_sigma_px=6.,geometry_sigma_floor_px=6.,
    position_ci_weight=.5,innovation_chi2_99=9.21034037197618,
    no_geometry_on_missing_body=True,geometry_uncertainty_is_proxy=True,
    world_force_estimated=False,wall_clock_realtime_verified=False)

class QualifiedCenter:
    def __init__(self):self.epoch=None;self.offset=None;self.offset_variance=None
    def step(self,body,g,rotors):
        if self.epoch!=g['geometry_epoch']:
            self.epoch=g['geometry_epoch'];self.offset=None;self.offset_variance=None
        result=copy.deepcopy(body);info=dict(applied=False,reason='no_current_qualified_geometry',noise_scale=1.)
        if not body.get('detected') or body.get('low_confidence'):return result,info
        current_four=(g['mode']=='relock_pending' and g.get('center_source')=='current_four_center_template_unconfirmed')
        partial=(g['mode']=='partial_two' and g.get('model_conditioned_center_supported_by_current_partial_centres',False))
        covariance=np.asarray(g.get('model_conditioned_center_covariance',np.eye(2)*1e9),float)
        if partial and (covariance.shape!=(2,2) or not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).max()>36.):
            return result,dict(info,reason='partial_center_uncertainty_too_large')
        if not partial and ((g['mode'] not in ('four','three') and not current_four) or not g.get('joint_center_is_current_measurement')):return result,info
        p=np.asarray(body['center'],float);z=np.asarray(g['model_conditioned_center'] if partial else g['current_geometric_center'],float)
        sigmas=[float(r.get('center_sigma_px',1.)) for r in rotors if r.get('current_observation',True)]
        fit_var=sum(s*s for s in sigmas)/max(len(sigmas)**2,1)
        local_variance=36.+fit_var
        if self.offset is None:
            if g['mode']!='four':return result,dict(info,reason='geometry_gauge_not_initialized')
            self.offset=z-p;self.offset_variance=36.+local_variance
        R=np.eye(2)*(local_variance+self.offset_variance)/max(g['geometric_weight'],.05)
        if partial:R+=covariance/max(g['geometric_weight'],.05)
        P=np.eye(2)*36.;geometry_point=z-self.offset;residual=geometry_point-p
        nis=float(residual@np.linalg.solve(P+R,residual))
        info.update(offset=self.offset.tolist(),geometry_point=geometry_point.tolist(),innovation_nis=nis,
            geometry_covariance_proxy=R.tolist(),current_body_position_unchanged=p.tolist())
        if nis>PARAMS['innovation_chi2_99']:return result,dict(info,reason='geometry_body_disagreement')
        # Fixed symmetric covariance intersection, as in the project's existing
        # symmetric-CI comparison; no independent-image-noise assumption.
        invP,invR=np.linalg.inv(P),np.linalg.inv(R)
        C=np.linalg.inv(.5*invP+.5*invR);point=C@(.5*invP@p+.5*invR@geometry_point)
        result['center']=point.tolist();result['position_measurement_source']='qualified_structure_CI'
        info.update(applied=True,reason='current_geometry_covariance_intersection',
            fused_position=point.tolist(),correction_px=(point-p).tolist(),
            noise_scale=float(np.sqrt(np.trace(C)/72.)),covariance_proxy=C.tolist())
        if partial:info.update(reason='prior_conditioned_partial_center_CI',center_is_direct_current_measurement=False)
        return result,info

class NoiseKF(MeasurementKF):
    def __init__(self,noise_model='legacy'):super().__init__(noise_model);self.noise_scale=1.
    def step(self,t,z,measurement_scale=1.):
        return super().step(t,z,measurement_scale*self.noise_scale)

class GeometryMotion(ScaleRolloutMotion):
    def __init__(self,tracker_hz=25.,noise_model='legacy'):
        super().__init__(tracker_hz,noise_model);self.kf=NoiseKF(noise_model)
