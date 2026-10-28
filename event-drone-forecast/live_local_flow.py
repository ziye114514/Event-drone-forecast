"""Quality-controlled local displacement; native ON/OFF support, no reconstruction.

Reuse: installed OpenCV pyramidal LK + partial-affine RANSAC, and installed
Metavision sparse full-motion flow. These are established algorithms, not new
networks. Current detector boxes select an ROI; future boxes never participate.
"""
import numpy as np

DESIGN=dict(lk=dict(max_corners=80,quality_level=.01,min_distance=4,window=21,max_level=3,
    maximum_gap_s=.08000001,forward_backward_px=1.5,minimum_tracks=8,ransac_px=2.,minimum_inlier_fraction=.6),
    sdk=dict(preset='FastObjects',minimum_distinct_current_clusters=3),
    fusion=dict(method='Covariance intersection on velocity only; unknown shared-image correlation',
        localization_noise_floor_px=1.,box_consistency_chi2_95=5.991464547107979),
    parameters_swept=False,current_position_acceleration_validity_unchanged=True,
    grayscale_reconstruction=False,thrust_estimated=False,default_promoted=False)


def event_support(events):
    image=np.zeros((720,1280),np.uint8)
    image[events['y'],events['x']]=255
    return image


def frame_support(image,background_blue):
    # Official SDK default palette: ON=255, OFF=200, background=52 in B.
    # Both polarities are active; this is a binary event support, not intensity.
    return (image[:,:,0]!=background_blue).astype(np.uint8)*255


def box(body):
    b=np.asarray(body.get('bbox',[]),float)
    if not body.get('detected') or body.get('low_confidence',False) or b.shape!=(4,) or not np.isfinite(b).all():return None
    if np.any(b[:2]<=0) or np.any(b[2:]>=[1279,719]) or np.any(b[2:]-b[:2]<12):return None
    return b


def reused_similarity(points,current,fb,center):
    """Reuse existing accepted LK pairs; evaluate a similarity at body center.

    Same installed OpenCV RANSAC/gates as FeatureFlow. Residual scatter is not
    divided by track count: a common correlated error must not become precise
    simply because many blade corners support the same motion.
    """
    import cv2
    points=np.asarray(points,float);current=np.asarray(current,float);fb=np.asarray(fb,float)
    if len(points)<8:return dict(valid=False,reason='few_current_track_pairs')
    affine,inliers=cv2.estimateAffinePartial2D(points,current,method=cv2.RANSAC,ransacReprojThreshold=2.,
        maxIters=1000,confidence=.99,refineIters=10)
    if affine is None:return dict(valid=False,reason='current_similarity_fit_failed')
    keep=inliers.ravel().astype(bool);fraction=float(keep.mean())
    if keep.sum()<8 or fraction<.6:return dict(valid=False,reason='few_current_similarity_inliers')
    residual=current[keep]-(points[keep]@affine[:,:2].T+affine[:,2])
    covariance=np.cov(residual.T)+np.eye(2)*max(1.,float(np.median(fb[keep]**2)))
    center=np.asarray(center,float);delta=affine[:,:2]@center+affine[:,2]-center
    return dict(valid=True,reason='current_similarity_from_existing_track_pairs',displacement=delta.tolist(),
        displacement_covariance_px2=covariance.tolist(),affine=affine.tolist(),evaluation_center=center.tolist(),
        inliers=int(keep.sum()),inlier_fraction=fraction,rotation_scale_removed_from_translation_scatter=True,
        covariance_divided_by_point_count=False,uncertainty_is_proxy=True,new_LK_passes=0)


class FeatureFlow:
    def __init__(self,crop=False,padding=192,spatial_diagnostics=False):
        self.previous=None;self.crop=crop;self.padding=padding;self.spatial_diagnostics=spatial_diagnostics

    def step(self,t,support,body):
        import cv2
        current=box(body);prior=self.previous
        self.previous=(t,support.copy(),body) if current is not None else None
        info=dict(valid=False,reason='no_current_or_previous_box')
        if current is None or prior is None:return info
        previous_t,previous,oldbody=prior;old=box(oldbody);dt=t-previous_t
        if oldbody.get('body_track_id') is not None and body.get('body_track_id') is not None:
            if (oldbody.get('body_track_epoch'),oldbody['body_track_id'])!=(body.get('body_track_epoch'),body['body_track_id']):
                return dict(valid=False,reason='identity_changed')
        if not 0<dt<=.08000001:return dict(valid=False,reason='gap',dt=dt)
        origin=np.zeros(2,dtype=int)
        if self.crop:
            # Align to 2**3 to retain grid phase. The tight112px variant covers
            # coarse LK radius80 + interpolation8 + pyramid Gaussian14 + base
            # blur1 =103px. Initial192px padding remains available for reference.
            origin=np.maximum(0,np.floor((np.minimum(old[:2],current[:2])-self.padding)/8)*8).astype(int)
            edge=np.minimum([1280,720],np.ceil((np.maximum(old[2:],current[2:])+self.padding)/8)*8).astype(int)
            previous=previous[origin[1]:edge[1],origin[0]:edge[0]]
            support=support[origin[1]:edge[1],origin[0]:edge[0]]
        mask=np.zeros_like(previous);lo=np.ceil(old[:2]+2-origin).astype(int);hi=np.floor(old[2:]-2-origin).astype(int)
        mask[lo[1]:hi[1],lo[0]:hi[0]]=255
        a=cv2.GaussianBlur(previous,(3,3),0);b=cv2.GaussianBlur(support,(3,3),0)
        points=cv2.goodFeaturesToTrack(a,maxCorners=80,qualityLevel=.01,minDistance=4,mask=mask,blockSize=3)
        count=0 if points is None else len(points);info.update(reason='few_corners',corners=count,dt=dt)
        if count<8:return info
        options=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
        q,forward,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,**options)
        back,backward,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,**options)
        p=points.reshape(-1,2);q=q.reshape(-1,2);back=back.reshape(-1,2)
        fb=np.linalg.norm(back-p,axis=1);p=p+origin;q=q+origin
        good=forward.ravel().astype(bool)&backward.ravel().astype(bool)&np.isfinite(q).all(axis=1)&(fb<=1.5)
        good&=(q[:,0]>=current[0])&(q[:,0]<=current[2])&(q[:,1]>=current[1])&(q[:,1]<=current[3])
        p,q,fb=p[good],q[good],fb[good];info.update(reason='few_bidirectional_tracks',bidirectional_tracks=len(p))
        if len(p)<8:return info
        affine,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.,maxIters=1000,confidence=.99,refineIters=10)
        if affine is None:return dict(info,reason='affine_fit_failed')
        inside=inliers.ravel().astype(bool);fraction=float(inside.mean());p,q,fb=p[inside],q[inside],fb[inside]
        info.update(reason='few_affine_inliers',inliers=len(p),inlier_fraction=fraction)
        if len(p)<8 or fraction<.6:return info
        center=np.asarray(oldbody['center'],float)
        displacement=affine[:,:2]@center+affine[:,2]-center
        residual=q-(p@affine[:,:2].T+affine[:,2]);variance=max(1.,float(np.mean(residual*residual)+np.median(fb*fb)))
        # Linear similarity parameters [a,b,tx,ty], covariance at the old center.
        jac=np.zeros((2*len(p),4));jac[::2,0]=p[:,0];jac[::2,1]=-p[:,1];jac[::2,2]=1
        jac[1::2,0]=p[:,1];jac[1::2,1]=p[:,0];jac[1::2,3]=1
        target=np.array([[center[0],-center[1],1,0],[center[1],center[0],0,1]])
        covariance=target@np.linalg.pinv(jac.T@jac)@target.T*variance+np.eye(2)
        bbox_displacement=np.asarray(body['center'])-center
        mismatch=displacement-bbox_displacement
        consistency=float(mismatch@np.linalg.solve(covariance+np.eye(2)*72.,mismatch))
        info.update(reason='disagrees_with_current_box',box_displacement_chi2=consistency)
        if consistency>5.991464547107979:return info
        info.update(valid=True,reason='qualified_bidirectional_similarity_flow',velocity=(displacement/dt).tolist(),
            velocity_covariance=(covariance/(dt*dt)).tolist(),midpoint_t=(t+previous_t)/2,
            displacement=displacement.tolist(),affine=affine.tolist(),rms_residual_px=float(np.sqrt(np.mean(residual*residual))))
        if self.spatial_diagnostics:
            from live_flow_spatial import support as spatial_support
            info['spatial_support']=spatial_support(p,q,center,dt)
        return info


def summarize_sparse(flow,t,body):
    b=box(body)
    if b is None or not len(flow):return dict(valid=False,reason='no_current_box_or_sparse_flow',clusters=0)
    select=(flow['center_x']>=b[0])&(flow['center_x']<=b[2])&(flow['center_y']>=b[1])&(flow['center_y']<=b[3])
    select&=(flow['t']>=round((t-.02)*1e6))&(flow['t']<round(t*1e6))
    flow=flow[select]
    if not len(flow):return dict(valid=False,reason='no_current_sparse_cluster',clusters=0)
    # One vote per cluster, regardless of rotor event rate.
    _,reverse=np.unique(flow['id'][::-1],return_index=True);latest=flow[len(flow)-1-reverse]
    velocities=np.c_[latest['vx'],latest['vy']].astype(float);velocities=velocities[np.isfinite(velocities).all(axis=1)]
    result=dict(valid=False,reason='few_distinct_sparse_clusters',clusters=len(velocities),raw_flow_records=len(flow))
    if len(velocities)<3:return result
    center=np.median(velocities,axis=0);distance=np.linalg.norm(velocities-center,axis=1)
    mad=np.median(abs(distance-np.median(distance)));keep=distance<=np.median(distance)+3*1.4826*max(mad,1.)
    values=velocities[keep];result.update(inliers=len(values))
    if len(values)<3:return result
    covariance=np.cov(values.T)/len(values)+np.eye(2)*(1/.02)**2
    result.update(valid=True,reason='qualified_sparse_cluster_consensus',velocity=np.median(values,axis=0).tolist(),
        velocity_covariance=covariance.tolist(),midpoint_t=t-.01)
    return result


def velocity_coefficient(coefficient,baseline_covariance,observation,t,strategy='minimum-volume'):
    """Isolated rollout change; state/eligibility remain exactly the baseline's.

    Covariance intersection avoids treating correlated image estimates as
    independent. Its weight minimizes log determinant, never trajectory error.
    Covariances are engineering uncertainty proxies, not calibrated confidence.
    """
    if coefficient is None or not observation.get('valid'):return coefficient,dict(applied=False,reason=observation.get('reason'))
    import copy
    P=np.asarray(baseline_covariance,float);R=np.asarray(observation['velocity_covariance'],float)
    invP,invR=np.linalg.inv(P),np.linalg.inv(R)
    if strategy=='symmetric':
        # Established default omega=.5, e.g. Stone Soup ChernoffUpdater.
        # This weights precision matrices, not a 50/50 average of velocities.
        weight=.5
    elif strategy=='minimum-volume':
        # If one precision dominates the other in PSD order, every interior
        # information matrix is bounded by that endpoint. Its log determinant
        # cannot be larger; the same minimum-volume endpoint is known exactly.
        # Otherwise retain the original scalar optimizer, without approximation.
        difference=invR-invP
        eigenvalues=np.linalg.eigvalsh((difference+difference.T)*.5)
        if eigenvalues[0]>=0.:weight=0.
        elif eigenvalues[-1]<=0.:weight=1.
        else:
            from scipy.optimize import minimize_scalar
            def objective(w):return -np.linalg.slogdet(w*invP+(1-w)*invR)[1]
            optimal=minimize_scalar(objective,bounds=(0.,1.),method='bounded')
            weight=min([0.,1.,float(optimal.x)],key=objective)
    else:raise ValueError('Unknown velocity CI strategy')
    if weight>=1.-1e-8:return coefficient,dict(applied=False,reason='baseline_velocity_more_precise',baseline_weight=weight)
    incoming=np.asarray(observation['velocity'])+(t-observation['midpoint_t'])*np.asarray(coefficient['acceleration'])
    covariance=np.linalg.inv(weight*invP+(1-weight)*invR)
    value=covariance@(weight*invP@np.asarray(coefficient['velocity'])+(1-weight)*invR@incoming)
    result=copy.deepcopy(coefficient);result['velocity']=value.tolist()
    return result,dict(applied=True,reason='velocity_covariance_intersection' if strategy=='minimum-volume' else 'velocity_covariance_intersection_symmetric',baseline_weight=weight,
        before=coefficient['velocity'],flow_current=incoming.tolist(),after=value.tolist(),uncertainty_is_proxy=True)
