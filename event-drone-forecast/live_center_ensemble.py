"""Equal current-center ensemble, keep original box and measurement noise.

The two related detectors are correlated: do not halve the covariance. The
secondary model cannot initialize, extend a gap, change identity or add a box.
Only matched current measurements contribute. This is a standard ensemble,
not a new physical estimator or evidence of true center-of-mass recovery.
"""
import copy
import numpy as np
DESIGN=dict(weight_each=.5,minimum_secondary_confidence=.45,minimum_IoU=.5,
    maximum_center_shift_short_side_fraction=.25,original_covariance_preserved=True,
    original_detection_and_box_preserved=True,parameter_sweep=False,default_promoted=False)

def blend_current(body,secondary_boxes):
    output=copy.deepcopy(body)
    info=dict(used=False,reason='no_original_current_detection')
    if not body.get('detected'):return output,info
    b=np.asarray(body['bbox'],float);size=b[2:]-b[:2];center=np.asarray(body['center'],float)
    best=None;best_iou=0.
    for candidate in secondary_boxes:
        if candidate.get('confidence',0)<.45:continue
        c=np.asarray(candidate['bbox'],float)
        intersection=np.prod(np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])))
        union=np.prod(size)+np.prod(c[2:]-c[:2])-intersection;iou=float(intersection/max(union,1e-12))
        delta=np.linalg.norm((c[:2]+c[2:])/2-center)
        if iou>=.5 and delta<=.25*min(size) and iou>best_iou:best=c;best_iou=iou
    if best is None:info['reason']='no_matched_secondary_current_box';return output,info
    other=(best[:2]+best[2:])/2;output['center']=((center+other)/2).tolist()
    output['measurement_source']='equal_current_detector_centers'
    info.update(used=True,reason='two_matched_current_centers',IoU=best_iou,
                original_center=center.tolist(),secondary_center=other.tolist(),same_measurement_noise=True)
    return output,info
