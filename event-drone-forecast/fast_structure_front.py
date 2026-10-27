"""ROI-only current ON/OFF geometry; reuse conic, curvature and five-circle code.

No historical multi-frontend controls, whole-frame masks, or virtual rotor pixels.
Counts retain a20ms physical window independent of the requested update rate.
"""
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src')]
from event_drone_forecast.geometry import fit_rotor_geometry,_fit,_valid,_distance,_coverage
from rotor_geometry import BranchGeometry,rotor_rates,plane_axis
from curvature_split import split_contour
from body_axis import body_measurement
from five_circle_model import FiveCircleModel
from continuous_five_circle import ContinuousFiveCircle
from shared_arc_recovery import recover as recover_shared_arcs
from partial_arc_information import update as update_partial_arcs


def roi_bounds(box,w=1280,h=720,padding=.15):
    b=np.asarray(box,float);pad=np.maximum((b[2:]-b[:2])*padding,12.)
    return np.r_[np.maximum(0,np.floor(b[:2]-pad)),np.minimum([w,h],np.ceil(b[2:]+pad))].astype(int)


def roi_events(events,bounds):
    x,y,X,Y=bounds
    return events[(events['x']>=x)&(events['x']<X)&(events['y']>=y)&(events['y']<Y)]


class NativeROI:
    """Installed SDK C++ rectangle filter; same original coordinates/order."""
    def __init__(self):
        from metavision_sdk_core import RoiFilterAlgorithm
        self.algorithm=RoiFilterAlgorithm
        self.output=RoiFilterAlgorithm(0,0,1279,719,False).get_empty_output_buffer()
    def select(self,events,bounds):
        x,y,X,Y=map(int,bounds)
        a=self.algorithm(x,y,X-1,Y-1,False);a.process_events(events,self.output)
        return self.output.numpy()


def count_pair(events,bounds,t,selector=None):
    x,y,X,Y=bounds;w,h=X-x,Y-y
    # Timestamp-ordered suffix is a view, not a structured-array boolean copy.
    begin=int(np.searchsorted(events['t'],round(t*1e6)-20000))
    e=events[begin:]
    e=roi_events(e,bounds) if selector is None else selector.select(e,bounds)
    index=(e['y'].astype(np.int32)-y)*w+(e['x'].astype(np.int32)-x)
    # One joint polarity/pixel histogram avoids two boolean gathers of index.
    # Both returned planes remain contiguous and retain exact integer counts.
    joint=np.bincount(index+e['p'].astype(np.int32)*(w*h),minlength=2*w*h)
    counts=joint.reshape(2,h,w).astype(np.float32)
    return [counts[1],counts[0]]


def record(g,offset):
    g=dict(g);p=np.asarray(g.pop('center'))+offset
    return dict(g,cx=float(p[0]),cy=float(p[1]),current_observation=True,
        fov_clipped=bool(p[0]-g['major']/2<0 or p[0]+g['major']/2>=1280 or
                         p[1]-g['minor']/2<0 or p[1]+g['minor']/2>=720))


class FastStructure:
    def __init__(self,reuse_uncertainty=False,continuous=True,shared_arc_recovery=True,guide_full_only=False,bipolar_guide_area=False,contrast_guide_area=True,current_shape_feedback=True,partial_arc_information=True):
        self.geometry=BranchGeometry();self.model=ContinuousFiveCircle(current_shape_feedback) if continuous else FiveCircleModel(3)
        self.tick=0;self.selector=NativeROI()
        self.previous=[];self.previous_body=None;self.previous_t=None
        self.reuse_uncertainty=reuse_uncertainty
        self.shared_arc_recovery=shared_arc_recovery
        self.guide_full_only=guide_full_only
        self.bipolar_guide_area=bipolar_guide_area
        self.contrast_guide_area=contrast_guide_area
        self.partial_arc_information=partial_arc_information

    def quick_conic(self,contour,offset,body,t,max_major):
        """Current AMS conic; reuse conservative uncertainty for at most2ticks.

        Full SVD uncertainty is refreshed at10Hz. Shape/centre/residual are still
        measured NOW. This approximates uncertainty, not observed hub positions.
        """
        if not self.reuse_uncertainty or self.tick%3==1 or not self.previous or t-self.previous_t>.10:return None
        points=np.asarray(contour,float).reshape(-1,2);p=_fit(points)
        if not _valid(p,points,max_major,None):return None
        delta=np.asarray(body['center'])-self.previous_body
        prior=min(self.previous,key=lambda r:np.linalg.norm(p[:2]+offset-[r['cx'],r['cy']]-delta))
        if prior.get('fit_condition') is None or prior.get('mode')=='line':return None
        size=np.array([2*p[2]/prior['major'],2*p[3]/prior['minor']])
        angle=abs((np.rad2deg(p[4])-prior['angle']+90)%180-90)
        error=np.linalg.norm(p[:2]+offset-[prior['cx'],prior['cy']]-delta)
        if size.min()<.85 or size.max()>1.15 or angle>5 or error>max(8.,.25*prior['major']):return None
        distance=abs(_distance(points,p));keep=distance<max(1.4,.048*p[2])
        if keep.sum()<10:return None
        coverage=_coverage(points[keep],p);support=float(keep.mean());median=float(np.median(distance[keep]))
        if coverage<.48 or support<.6:return None
        g=dict(prior);g.pop('cx');g.pop('cy')
        sigma=max(prior['center_sigma_px']*1.05,median,(1-coverage)*p[2]*.15)
        g.update(center=p[:2],major=float(2*p[2]),minor=float(2*p[3]),angle=float(np.rad2deg(p[4])),
            ratio=float(p[3]/p[2]),normal=np.array([-np.sin(p[4]),np.cos(p[4])])*np.sqrt(max(0.,1-(p[3]/p[2])**2)),
            mode='ellipse' if coverage>=.82 and support>=.88 else 'arc',
            inferred_shape=coverage<.82 or support<.88,visible_arc_fraction=coverage,support_fraction=support,
            fit_residual_px=median,center_sigma_px=sigma,center_covariance_px2=(np.eye(2)*sigma*sigma).tolist(),
            uncertainty_source='recent_full_fit_inflated_current_shape_and_residual_checked',
            uncertainty_approximate=True)
        return g

    def step(self,events,t,body,q,velocity,frequency_pixels_unique=False):
        self.tick+=1
        empty=dict(state='insufficient',observed_count=0,observed_rotors=[],confidence=0.)
        if not body.get('detected'):
            return self.model.step(t,body,empty),[],dict(reason='no_current_body'),None
        bounds=roi_bounds(body['bbox']);x,y,X,Y=bounds
        on,off=count_pair(events,bounds,t,self.selector);density=np.minimum(on,off)
        # Same stage6 thresholds and side qualification, on the small native ROI.
        local_box=np.asarray(body['bbox'])-np.r_[bounds[:2],bounds[:2]]
        side={}
        # Reuse neighbor10 exactly: spatially align each polarity BEFORE min.
        # min(blur(ON),blur(OFF)) differs materially from blur(min(ON,OFF)).
        feature=np.minimum(cv2.GaussianBlur(on,(3,3),.8),cv2.GaussianBlur(off,(3,3),.8))
        feature[(on+off)<2]=0
        blurred=cv2.GaussianBlur(feature,(3,3),.8)
        raw_mask=np.uint8(blurred>=4.)*255
        # Existing morphology: remove isolated gaps without fitting their edges.
        mask=cv2.morphologyEx(raw_mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours=sorted([c for c in contours if len(c)>=16 and cv2.contourArea(c)>=20],key=cv2.contourArea,reverse=True)[:12]
        rotors=[];split_count=0
        # Degenerate side views need an axis, not repeated ill-conditioned conics.
        # A thin whole-body projection alone cannot exclude four visible discs.
        # Rebuild the complete template at10Hz even in the cheap side branch.
        # Cheap complete/qualified-arc fits each current tick. The body-axis LSD
        # is a fallback AFTER checking for four discs, never a preemptive route.
        if contours:
            for c in contours[:6]:
                # Fast current complete conics first. Expensive partial-arc
                # proposals are confined to the validated curvature path below.
                max_major=min(500.,1.5*max(X-x,Y-y))
                g=self.quick_conic(c,bounds[:2],body,t,max_major)
                if g is None:g=fit_rotor_geometry(c,max_major=max_major,allow_arc=False)
                if g is not None:
                    g['arc_center_qualified']=bool(g['mode']=='arc' and g.get('visible_arc_fraction',0)>=.48 and
                        g.get('support_fraction',0)>=.6 and g.get('fit_condition',1e9)<=1500 and
                        g.get('center_sigma_px',1e9)<=3 and g.get('axis_angle_sigma_deg',90)<=15)
                    rotors.append(record(g,bounds[:2]))
            if rotors:
                # Existing merge110 size-ratio convention removes tiny body
                # details; independent discs still have to pass joint geometry.
                largest=max(r['major'] for r in rotors)
                rotors=[r for r in rotors if r['major']>=largest/2.5]
            # Original concavity/arc validation runs at10Hz, never on every side tick.
            if len(rotors)!=4 and self.tick%15==1:
                for c in contours[:1]:
                    arcs,_=split_contour(c)
                    if arcs is None:continue
                    replacements=[]
                    for arc in arcs:
                        p=arc['parameters'];replacements.append(dict(cx=float(p[0]+x),cy=float(p[1]+y),
                            major=float(2*p[2]),minor=float(2*p[3]),ratio=float(p[3]/p[2]),angle=float(np.rad2deg(p[4])),
                            mode='arc',inferred_shape=True,curvature_validated=True,current_observation=True,
                            fov_clipped=False,center_sigma_px=arc['sigma'],fit_residual_px=arc['rms'],geometry_quality=.6))
                    # Replace only the original component whose contour contains its fit centre.
                    rotors=[r for r in rotors if cv2.pointPolygonTest(c,(r['cx']-x,r['cy']-y),False)<0]+replacements
                    split_count+=1
        arc_recovery=dict(attempted=False,accepted=0)
        if self.shared_arc_recovery and self.tick%3==1 and body.get('confidence',0)>=.45:
            area_mask=None
            if self.bipolar_guide_area:
                occupancy=((on>0)&(off>0)).astype(np.float32)
                area_mask=np.uint8(cv2.GaussianBlur(occupancy,(3,3),.8)>=.5)*255
            rotors,arc_recovery=recover_shared_arcs(self.model,t,body,raw_mask,bounds[:2],rotors,
                full_only=self.guide_full_only,area_mask=area_mask,contrast_area=self.contrast_guide_area)
        geom=self.geometry.step(t,body,rotors)
        # Curvature validates actual open arcs; permit the existing four-point model.
        qualified=[r for r in rotors if r.get('mode')=='ellipse' or r.get('curvature_validated') or r.get('arc_center_qualified')]
        if len(rotors) in (3,4) and len(qualified)==len(rotors) and geom['state'] not in ('four','three_completed'):
            adapted=[dict(r,mode='ellipse',inferred_shape=False) for r in rotors]
            geom=self.geometry.step(t,body,adapted);geom['observed_rotors']=rotors
        structure=self.model.step(t,body,geom)
        normal_constraints=dict(updated=False,reason='disabled')
        if self.partial_arc_information:
            normal_constraints=update_partial_arcs(self.model,t,body,raw_mask,bounds[:2],rotors)
            if normal_constraints['updated']:
                structure['model_centers']=self.model._hubs().tolist()
                structure['model_conditioned_center']=self.model.beta[[0,3]].tolist()
                structure['model_conditioned_center_covariance']=self.model.P[np.ix_([0,3],[0,3])].tolist()
                structure['five_circles']['rotors']=structure['model_centers']
                if structure['current_geometric_center'] is not None:
                    structure['current_geometric_center']=self.model.beta[[0,3]].tolist()
                    structure['five_circles']['body']=structure['current_geometric_center']
                structure['continuous_normal_constraints']=normal_constraints
        visible_plane=False
        if structure.get('axis_xy') is None and len(qualified)>=2 and np.median([r['ratio'] for r in qualified])>.16:
            normal,agreement=plane_axis(qualified)
            if agreement>=.85:
                visible_plane=True
                structure.update(axis_xy=normal.tolist(),geometric_weight=.35,
                    pose_source='current_qualified_visible_conics',pose_without_complete_five_circle=True)
        if structure['current_geometric_center'] is None and not visible_plane:
            side=body_measurement(density,local_box)
        if side.get('valid') and side.get('confidence',0)>=.45 and structure['current_geometric_center'] is None:
            n=np.asarray(side['normal_axis_xy'])
            # Whole-body elongation is an unsigned axis, not proof of exactly
            # two physical motors. Keep this unresolved mode distinct.
            flat=bool(qualified and np.median([r['ratio'] for r in qualified])<=.16)
            mode='edge_on' if flat else 'axis_only'
            if hasattr(self.model,'core'):self.model.core.mode=mode
            else:self.model.mode=mode
            structure.update(mode=mode,axis_xy=n.tolist(),geometric_weight=.35,
                reason='stage6_current_ROI_count_axis',axis_support=dict(side,center=(np.asarray(side['center'])+[x,y]).tolist()))
            if hasattr(self.model,'observe_axis'):
                structure=self.model.observe_axis(structure,n,side['confidence'],t)
        rates=rotor_rates(q,rotors,t,velocity,1280,unique_pixels=frequency_pixels_unique) if rotors else []
        self.previous=rotors;self.previous_body=np.asarray(body['center']);self.previous_t=t
        diag=dict(roi=bounds.tolist(),current_window_us=20000,threshold=4.,contours=len(contours),
            curvature_splits=split_count,shared_arc_recovery=arc_recovery,
            partial_arc_information=normal_constraints,
            side=side,observed_regions=len(rotors),rotors=rotors,roi_pixels=int((X-x)*(Y-y)))
        return structure,rates,diag,(density,bounds)
