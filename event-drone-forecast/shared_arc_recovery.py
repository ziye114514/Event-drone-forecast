"""Shared model guides CURRENT supported arcs; no prior-only rotor output.

Reuse148's fixed distance/coverage/compatibility gates and147's arc fitter.
Points come from the current threshold mask BEFORE closing, no cut chords.
"""
from pathlib import Path
import sys
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src')]
from arc_parameters import PARAMS
from curvature_split import fit_arc
from event_drone_forecast.geometry import _distance,_coverage


def recover(model,t,body,raw_mask,offset,rotors,full_only=False,area_mask=None,contrast_area=True):
    audit=dict(attempted=False,accepted=0,no_prior_only_output=True,no_closing_chord=True)
    if (not hasattr(model,'beta') or model.beta is None or
        getattr(model,'full_observations',0)<PARAMS['min_prior_observations']):return rotors,audit
    if full_only:
        audit['guidance_reference']=dict(eligible=t-model.last_full<=PARAMS['max_prior_age_s'],
            reference_source='current_four_only',reference_age_s=t-model.last_full)
    else:audit['guidance_reference']=model.guidance_reference(t,PARAMS['max_prior_age_s'])
    if not audit['guidance_reference']['eligible']:return rotors,audit
    covariance=model.P[np.ix_([0,3],[0,3])]
    if np.linalg.eigvalsh(covariance).max()>36.:return rotors,dict(audit,reason='uncertain_template')
    hubs=model._hubs()+np.asarray(body['center'])-model.body
    seeds=[]
    for center,S in zip(hubs,model.shapes):
        values,vectors=np.linalg.eigh(S);major,minor=2*np.sqrt(np.maximum(values[::-1],1e-9))
        angle=np.arctan2(vectors[1,1],vectors[0,1])%np.pi
        seeds.append(np.r_[center,major/2,minor/2,angle])
    # Preserve only independently qualified current candidates in seed slots.
    slots={}
    def compatible(p,s):
        return (np.linalg.norm(p[:2]-s[:2])<=max(4.,2*s[2]*PARAMS['prior_center_gate_major']) and
            .75<p[2]/s[2]<1.3 and .6<p[3]/s[3]<1.5)
    for r in rotors:
        if r.get('fov_clipped') or not (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated')):continue
        p=np.array([r['cx'],r['cy'],r['major']/2,r['minor']/2,np.deg2rad(r['angle'])])
        ids=[i for i,s in enumerate(seeds) if compatible(p,s)]
        if len(ids)==1:
            i=ids[0]
            if i not in slots or r.get('geometry_quality',0)>slots[i].get('geometry_quality',0):slots[i]=r
    if len(slots)==4:return rotors,audit
    contours,_=cv2.findContours(raw_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    area_contours=contours
    audit['area_source']='strong_count_threshold'
    if area_mask is not None:
        area_contours,_=cv2.findContours(area_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        audit['area_source']='current_binary_bipolar_occupancy'
    area_support=np.zeros_like(raw_mask)
    cv2.drawContours(area_support,area_contours,-1,255,cv2.FILLED)
    sets=[np.asarray(c,float).reshape(-1,2)+offset for c in contours if len(c)>=10]
    if not sets:return rotors,audit
    points=np.unique(np.concatenate(sets),axis=0)
    distances=np.array([abs(_distance(points,s)) for s in seeds])
    order=np.argsort(distances,axis=0);best=order[0];margin=np.take_along_axis(distances,order[1:2],axis=0)[0]-np.min(distances,axis=0)
    details=[];new=0;rejections=[]
    for i,seed in enumerate(seeds):
        if i in slots:continue
        keep=(best==i)&(distances[i]<=PARAMS['arc_distance_px'])&(margin>=PARAMS['assignment_distance_margin_px'])
        chosen=points[keep]
        coverage=_coverage(chosen,seed) if len(chosen)>=3 else 0.
        if len(chosen)<22 or coverage<PARAMS['minimum_current_arc_coverage']:
            rejections.append(dict(seed=i,reason='insufficient_current_strong_arc',pixels=len(chosen),coverage=coverage));continue
        audit['attempted']=True
        arc=fit_arc(chosen,max(2*seed[2],np.ptp(chosen,axis=0).max()))
        if arc is None:
            rejections.append(dict(seed=i,reason='current_arc_fit_failed',pixels=len(chosen),coverage=coverage));continue
        if not compatible(arc['parameters'],seed):
            rejections.append(dict(seed=i,reason='fitted_arc_incompatible_with_template',pixels=len(chosen),coverage=coverage));continue
        p=arc['parameters'];center=p[:2];dx=np.hypot(p[2]*np.cos(p[4]),p[3]*np.sin(p[4]));dy=np.hypot(p[2]*np.sin(p[4]),p[3]*np.cos(p[4]))
        if min(center-[dx,dy])<0 or np.any(center+[dx,dy]>=[1280,720]):continue
        # Reuse148's current-area union gate. Near-prior points alone can fit
        # an ellipse even in clutter; the actual contour area must support it.
        lo=np.maximum(0,np.floor(center-offset-[dx,dy]-4).astype(int))
        hi=np.minimum(raw_mask.shape[::-1],np.ceil(center-offset+[dx,dy]+5).astype(int))
        actual=area_support[lo[1]:hi[1],lo[0]:hi[0]]>0;ellipse=np.zeros(actual.shape,np.uint8)
        cv2.ellipse(ellipse,tuple(np.rint(center-offset-lo).astype(int)),tuple(np.maximum(1,np.rint(p[2:4]).astype(int))),float(np.rad2deg(p[4])),0,360,1,-1)
        union_iou=float(np.sum(actual&(ellipse>0))/max(1,np.sum(actual|(ellipse>0))))
        foreground_fraction=float(np.sum(actual&(ellipse>0))/max(1,np.sum(ellipse>0)))
        disk=np.zeros(raw_mask.shape,np.uint8);outer=np.zeros_like(disk)
        local_center=tuple(np.rint(center-offset).astype(int))
        cv2.ellipse(disk,local_center,tuple(np.maximum(1,np.rint(p[2:4]).astype(int))),float(np.rad2deg(p[4])),0,360,1,-1)
        cv2.ellipse(outer,local_center,tuple(np.maximum(1,np.rint(p[2:4]*1.25).astype(int))),float(np.rad2deg(p[4])),0,360,1,-1)
        ring=(outer>0)&(disk==0);ring_pixels=int(ring.sum())
        ring_density=float(np.sum((area_support>0)&ring)/max(1,ring_pixels))
        # Attached airframe pixels can lower union IoU even when the current
        # disc interior is well supported. Accept this separate path only with
        # >=.72 current interior coverage AND a sparse25% surrounding ring.
        contrast_ok=(foreground_fraction>=PARAMS['minimum_fragment_union_iou'] and
            ring_pixels>=.25*np.sum(disk) and ring_density<=.5*foreground_fraction)
        area_path='union_iou' if union_iou>=PARAMS['minimum_fragment_union_iou'] else 'foreground_and_ring_contrast'
        if union_iou<PARAMS['minimum_fragment_union_iou'] and not(contrast_area and contrast_ok):
            rejections.append(dict(seed=i,reason='current_area_support_failed',area_union_iou=union_iou,
                pixels=len(chosen),coverage=arc['coverage'],rms=arc['rms'],
                current_interior_coverage=foreground_fraction,current_ring_density=ring_density));continue
        slots[i]=dict(cx=float(p[0]),cy=float(p[1]),major=float(2*p[2]),minor=float(2*p[3]),
            ratio=float(p[3]/p[2]),angle=float(np.rad2deg(p[4])),mode='arc',inferred_shape=True,
            arc_center_qualified=True,current_observation=True,fov_clipped=False,center_sigma_px=arc['sigma'],
            fit_residual_px=arc['rms'],visible_arc_fraction=arc['coverage'],support_fraction=arc['support_fraction'],
            fit_condition=arc['condition'],axis_angle_sigma_deg=arc['angle_sigma'],geometry_quality=.6,
            shared_arc_recovered=True,source='current_preclosing_threshold_contour_supported_arc')
        new+=1;details.append(dict(seed=i,pixels=len(chosen),coverage=arc['coverage'],rms=arc['rms'],current_area_union_iou=union_iou,
            current_interior_coverage=foreground_fraction,current_ring_density=ring_density,area_evidence_path=area_path))
    audit['rejections']=rejections
    if new and len(slots)>=3:
        audit.update(accepted=new,current_supported_slots=len(slots),details=details)
        return [slots[i] for i in sorted(slots)],audit
    return rotors,audit
