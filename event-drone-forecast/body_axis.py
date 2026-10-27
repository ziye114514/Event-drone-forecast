"""Current runtime definitions extracted without arithmetic changes; see SOURCE_PROVENANCE.json."""
import cv2,numpy as np

PARAMS=dict(density_thresholds=[1.4,3.,6.],close_kernel=7,min_support=70,
    min_extent_px=22,max_extent_px=950,side_ratio_max=.38,
    multiscale_max_deg=12.,line_pca_max_deg=15.,track_gate_fraction=.8,
    coast_s=.3,reacquire_max_axis_change_deg=65.,body_confidence_min=.45)

def angle_distance(a,b):return float(abs((a-b+np.pi/2)%np.pi-np.pi/2))

def principal(points,weights=None):
    points=np.asarray(points,float)
    if len(points)<15:return None
    center=points.mean(0) if weights is None else np.average(points,axis=0,weights=weights)
    d=points-center;cov=d.T@d/max(1,len(d)) if weights is None else (d*weights[:,None]).T@d/max(weights.sum(),1e-9)
    values,vectors=np.linalg.eigh(cov);u=vectors[:,-1]
    angle=float(np.arctan2(u[1],u[0])%np.pi)
    return dict(center=center,angle=angle,ratio=float(np.sqrt(max(values[0],0)/max(values[1],1e-9))))

def body_measurement(density,bbox):
    h,w=density.shape;b=np.array(bbox,float)
    lo=np.maximum(np.floor(b[:2]).astype(int),0);hi=np.minimum(np.ceil(b[2:]).astype(int),[w-1,h-1])
    if np.any(hi-lo<12):return dict(valid=False,reason='small_roi')
    crop=density[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
    measurements=[];pointsets=[]
    for threshold in PARAMS['density_thresholds']:
        yy,xx=np.nonzero(crop>=threshold)
        if len(xx)<PARAMS['min_support']:continue
        points=np.c_[xx,yy].astype(float)
        if len(points)>5000:points=points[::int(np.ceil(len(points)/5000))]
        m=principal(points);measurements.append(m);pointsets.append(points)
    if len(measurements)<2:return dict(valid=False,reason='insufficient_multiscale_support')
    base=measurements[0];angle=base['angle'];spread=max(np.degrees(angle_distance(angle,m['angle'])) for m in measurements)
    points=pointsets[0];p=points+lo
    low=np.quantile(p,.015,axis=0);high=np.quantile(p,.985,axis=0);extent=float(max(high-low))
    if extent<PARAMS['min_extent_px']:return dict(valid=False,reason='small_extent')
    # OpenCV LSD is an actual line detector. Lines are image support, not the
    # four virtual spokes of the disk model.
    gray=np.uint8(np.clip(np.log1p(crop)/4,0,1)*255)
    lines=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    usable=[]
    if lines is not None:
        for line in lines.reshape(-1,4):
            d=line[2:]-line[:2];length=float(np.linalg.norm(d))
            if length<max(8,extent*.08):continue
            a=float(np.arctan2(d[1],d[0])%np.pi)
            if np.degrees(angle_distance(a,angle))<25:usable.append((a,length))
    line_angle=None;agreement=None
    if len(usable)>=2:
        aa=np.array([v[0] for v in usable]);weights=np.array([v[1] for v in usable])**2
        z=np.sum(weights*np.exp(2j*aa));line_angle=float((np.angle(z)/2)%np.pi)
        agreement=float(abs(z)/weights.sum())
    disagreement=float(np.degrees(angle_distance(line_angle,angle))) if line_angle is not None else None
    thin=base['ratio']<PARAMS['side_ratio_max']
    valid=bool(thin and spread<=PARAMS['multiscale_max_deg'] and
               (line_angle is None or disagreement<=PARAMS['line_pca_max_deg']))
    if valid and line_angle is not None and agreement>.7:
        z=np.exp(2j*angle)+np.exp(2j*line_angle);angle=float((np.angle(z)/2)%np.pi)
    confidence=float(np.clip((1-base['ratio'])*np.exp(-spread/15)*min(1,len(points)/250),0,1))
    axis=np.array([-np.sin(angle),np.cos(angle)])
    return dict(valid=valid,reason='side_body_axis' if valid else 'thick_or_inconsistent',
        center=((low+high)/2).tolist(),bbox=np.r_[low,high].tolist(),
        angle_rad=angle,normal_axis_xy=axis.tolist(),ratio=base['ratio'],
        multiscale_spread_deg=spread,line_count=len(usable),line_pca_disagreement_deg=disagreement,
        confidence=confidence,event_support=len(points),extent_px=extent,
        camera_model='Weak/orthographic image-axis approximation; no calibrated 3D normal.',
        sign_observed=False,thrust_magnitude_observed=False)
