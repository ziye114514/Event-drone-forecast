"""One current-frame crop refinement using the existing fixed-size detector.

Related established idea: SAHI (https://github.com/obss/sahi). This adapter is
one ROI proposal, not an implementation/reproduction of SAHI tiled inference.
"""
import cv2
import numpy as np

DESIGN=dict(context_factor=2.,maximum_zoom=4.,minimum_IoU=.5,
    maximum_center_shift_short_side_fraction=.25,minimum_confidence=.45,
    parameters_swept=False,grayscale_reconstruction=False)

def crop_transform(box):
    b=np.asarray(box,float);size=b[2:]-b[:2]
    if np.any(b[:2]<=0) or np.any(b[2:]>=[1279,719]):return None
    width=min(1280.,max(320.,2*max(size[0],size[1]*1280/720)))
    height=width*720/1280;center=(b[:2]+b[2:])/2
    origin=np.clip(center-np.array([width,height])/2,[0,0],[1280-width,720-height])
    scale=1280/width
    matrix=np.array([[scale,0,-origin[0]*scale],[0,scale,-origin[1]*scale]],float)
    return origin,scale,matrix

def select_refinement(original,boxes,origin,scale):
    b=np.asarray(original,float);best=None;best_iou=0.;size=b[2:]-b[:2];center=(b[:2]+b[2:])/2
    for row in boxes:
        candidate=row.copy();candidate[:4]=candidate[:4]/scale+np.tile(origin,2)
        c=candidate[:4];intersection=np.prod(np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])))
        union=np.prod(size)+np.prod(np.maximum(0,c[2:]-c[:2]))-intersection
        iou=intersection/max(union,1e-12);delta=np.linalg.norm((c[:2]+c[2:])/2-center)
        if candidate[4]<.45 or iou<.5 or delta>.25*min(size):continue
        if iou>best_iou:best,best_iou=candidate,float(iou)
    return best,best_iou

def refine(backend,backend_module,image,box):
    transform=crop_transform(box)
    if transform is None:return None,dict(reason='clipped_original_box')
    origin,scale,matrix=transform
    crop=cv2.warpAffine(image,matrix,(1280,720),flags=cv2.INTER_LINEAR)
    raw=backend.run(backend_module.preprocess(crop));boxes=backend_module.boxes(raw)
    best,iou=select_refinement(box,boxes,origin,scale)
    audit=dict(reason='matched_current_crop' if best is not None else 'no_matching_crop_detection',
               origin=origin.tolist(),scale=float(scale),candidate_count=len(boxes),selected_IoU=iou)
    if best is None:return None,audit
    return dict(bbox=best[:4].tolist(),confidence=float(best[4]),class_id=int(best[5])),audit
