"""Current rim constraints on the shared template, never extra detected rotors.

Inspired by model-contour normal-distance tracking (ViSP/VVS). This NumPy
implementation is original; no ViSP code is included. A short arc is allowed
to constrain only its observable centre directions, with prior conic shape.
It cannot initialize/recalibrate geometry or refresh the guidance clock.
"""
import cv2
import numpy as np
from event_drone_forecast.geometry import _distance, _coverage


def update(model, t, body, raw_mask, offset, rotors):
    audit=dict(updated=False, source='current_normal_distance_with_prior_shape',
        new_detected_rotors=0, refreshes_guidance=False, shape_is_prior_conditioned=True)
    if not hasattr(model,'guidance_reference') or model.beta is None:
        return audit
    reference=model.guidance_reference(t,.12)
    if not reference['eligible']:
        return dict(audit,reason='no_recent_qualified_template')
    qualified=[r for r in rotors if not r.get('fov_clipped') and
        not r.get('shared_arc_recovered') and
        (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
    if len(qualified)<2:
        return dict(audit,reason='fewer_than_two_independent_current_anchors')
    contours,_=cv2.findContours(raw_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    points=[c.reshape(-1,2) for c in contours if len(c)>=10]
    if not points:return dict(audit,reason='no_current_contour')
    points=np.unique(np.concatenate(points),axis=0)+offset
    hubs=model._hubs();seeds=[]
    for center,S in zip(hubs,model.shapes):
        values,vectors=np.linalg.eigh(S)
        seeds.append(np.r_[center,np.sqrt(values[::-1]),np.arctan2(vectors[1,1],vectors[0,1])])
    distances=np.abs(np.array([_distance(points,s) for s in seeds]))
    order=np.argsort(distances,axis=0);best=order[0]
    margin=np.take_along_axis(distances,order[1:2],axis=0)[0]-distances.min(0)
    occupied=set()
    for r in qualified:
        d=np.linalg.norm(hubs-[r['cx'],r['cy']],axis=1);j=int(d.argmin())
        if d[j]<=max(4.,.15*r['major']):occupied.add(j)
    # Filled CURRENT contours are area evidence, not projected template pixels.
    area=np.zeros_like(raw_mask);cv2.drawContours(area,contours,-1,255,cv2.FILLED)
    D=np.c_[np.ones(4),model.template];rows=[];residuals=[];variances=[];details=[]
    for j,seed in enumerate(seeds):
        if j in occupied:continue
        chosen=points[(best==j)&(distances[j]<=2.5)&(margin>=.7)]
        if len(chosen)<22:continue
        disk=np.zeros_like(raw_mask);outer=np.zeros_like(raw_mask)
        center=tuple(np.rint(seed[:2]-offset).astype(int));axes=np.maximum(1,np.rint(seed[2:4]).astype(int))
        cv2.ellipse(disk,center,tuple(axes),float(np.rad2deg(seed[4])),0,360,1,-1)
        cv2.ellipse(outer,center,tuple(np.maximum(1,np.rint(seed[2:4]*1.25).astype(int))),float(np.rad2deg(seed[4])),0,360,1,-1)
        ring=(outer>0)&(disk==0)
        inside=float(np.sum((area>0)&(disk>0))/max(1,disk.sum()))
        outside=float(np.sum((area>0)&ring)/max(1,ring.sum()))
        if inside<.72 or ring.sum()<.25*disk.sum() or outside>.5*inside:continue
        S_inv=np.linalg.inv(model.shapes[j]);delta=chosen-hubs[j]
        gradient=delta@S_inv
        length=np.linalg.norm(gradient,axis=1)
        normal=gradient/np.maximum(length[:,None],1e-9)
        # First-order signed distance to the fixed-shape conic.
        residual=(np.einsum('ij,ij->i',delta,gradient)-1)/(2*np.maximum(length,1e-9))
        weight=np.minimum(1.,1.4/np.maximum(np.abs(residual),1e-9))
        information=(normal.T*weight)@normal/weight.sum()
        rhs=normal.T@(weight*residual)/weight.sum()
        values,vectors=np.linalg.eigh(information);rank=0
        H=np.zeros((2,6));H[0,:3]=D[j];H[1,3:]=D[j]
        for k in range(2):
            # Weak directions remain unobserved instead of inventing a centre.
            if values[k]<.05:continue
            rows.append(vectors[:,k]@H)
            residuals.append(float(vectors[:,k]@rhs/values[k]))
            # Samples on a contour are correlated: use MEAN information and
            # a3px common error floor, never variance divided by pixel count.
            variances.append(9./values[k]);rank+=1
        if rank:details.append(dict(seed=j,pixels=len(chosen),
            arc_coverage=float(_coverage(chosen,seed)),observable_directions=rank,
            normal_information_eigenvalues=values.tolist(),
            current_interior_coverage=inside,current_ring_density=outside))
    if not rows:return dict(audit,reason='no_qualified_current_normal_constraints')
    H=np.asarray(rows);z=np.asarray(residuals);variance=np.asarray(variances)
    S=H@model.P@H.T+np.diag(variance)
    nis=float(z@np.linalg.solve(S,z));K=np.linalg.solve(S,H@model.P).T
    change=K@z
    if nis>13.277 or np.linalg.norm(change[[0,3]])>4.:
        return dict(audit,reason='normal_innovation_rejected',innovation_nis=nis,details=details)
    candidate=model.beta+change
    A=np.array([model.beta[1:3],model.beta[4:6]])
    B=np.array([candidate[1:3],candidate[4:6]])
    transform=B@np.linalg.inv(A);scales=np.linalg.svd(transform,compute_uv=False)
    if np.linalg.det(transform)<=0 or scales.min()<.85 or scales.max()>1.15:
        return dict(audit,reason='normal_shape_transport_rejected',details=details)
    model.beta=candidate;J=np.eye(6)-K@H
    model.P=J@model.P@J.T+(K*variance)@K.T
    model.shapes=np.array([transform@s@transform.T for s in model.shapes])
    model.core.hubs=model._hubs();model.core.shapes=model.shapes.copy()
    return dict(audit,updated=True,measurement_rank=int(np.linalg.matrix_rank(H)),
        innovation_nis=nis,center_correction_px=change[[0,3]].tolist(),details=details)
