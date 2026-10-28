"""Equivalent fused CPU line qualification and10 conic central differences."""
import time
import numpy as np
from native_counts_numba import njit
from event_drone_forecast.geometry import _distance


@njit(cache=True,nogil=True)
def line_statistics(points,max_major):
    n=len(points);mean=np.zeros(2)
    for i in range(n):
        mean[0]+=points[i,0];mean[1]+=points[i,1]
    mean/=n;covariance=np.zeros((2,2))
    for i in range(n):
        x=points[i,0]-mean[0];y=points[i,1]-mean[1]
        covariance[0,0]+=x*x;covariance[0,1]+=x*y;covariance[1,1]+=y*y
    covariance[1,0]=covariance[0,1];covariance/=n-1
    values,vectors=np.linalg.eigh(covariance)
    along=np.empty(n);across=np.empty(n)
    for i in range(n):
        x=points[i,0]-mean[0];y=points[i,1]-mean[1]
        along[i]=x*vectors[0,1]+y*vectors[1,1]
        across[i]=x*vectors[0,0]+y*vectors[1,0]
    major=along.max()-along.min();ordered=np.sort(across);bounds=np.empty(2)
    for j,fraction in enumerate((.05,.95)):
        location=(n-1)*fraction;lo=int(np.floor(location));hi=int(np.ceil(location));f=location-lo
        a=ordered[lo];b=ordered[hi];difference=b-a
        bounds[j]=b-difference*(1-f) if f>=.5 else a+difference*f
    width=max(1.,bounds[1]-bounds[0])
    if major<12. or major>max_major or width/major>.14:return False,mean,major,width,0.,0.
    center=mean+vectors[:,1]*(along.max()+along.min())/2
    angle=np.arctan2(vectors[1,1],vectors[0,1])%np.pi
    return True,center,major,width,angle,np.median(np.abs(across))


class JITLine:
    def __init__(self):
        start=time.perf_counter();points=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.],[4.,0.],[5.,0.]])
        line_statistics(points,500.);points.setflags(write=False);line_statistics(points,500.)
        self.provenance=dict(warm_ms=(time.perf_counter()-start)*1000,
            method='same_sample_covariance_eigh_projections_and_linear5_95percentiles',fastmath=False,
            thresholds_and_all_contour_points_unchanged=True,numerical_roundoff_not_bitwise_equivalence=True)

    def __call__(self,points,max_major):
        valid,center,major,width,angle,residual=line_statistics(points,float(max_major))
        if not valid:return None
        sigma=max(2.,width);angle_sigma=max(1.,float(np.rad2deg(width/max(major,1.))))
        return dict(center=center,major=major,minor=width,angle=float(np.rad2deg(angle)),ratio=width/major,
            normal=np.array([-np.sin(angle),np.cos(angle)]),mode='line',visible_arc_fraction=None,
            support_fraction=1.,fit_residual_px=residual,center_sigma_px=sigma,axis_angle_sigma_deg=angle_sigma,
            fit_condition=None,geometry_quality=float(np.clip(1-width/major*4,.25,.7)),shape_quality=.7,
            inferred_shape=False,axis_only=True,center_covariance_px2=(np.eye(2)*sigma**2).tolist())


@njit(cache=True,nogil=True)
def central_jacobian(points,parameters,cosine,sine,steps):
    distances=np.empty((10,len(points)),np.float64)
    for k in range(10):
        cx,cy,a,b=parameters[k,0],parameters[k,1],parameters[k,2],parameters[k,3]
        a2=a*a;b2=b*b
        for i in range(len(points)):
            dx=points[i,0]-cx;dy=points[i,1]-cy
            x=dx*cosine[k]+dy*sine[k];y=-dx*sine[k]+dy*cosine[k]
            xa=x/a2;yb=y/b2;gradient=2*np.sqrt(xa*xa+yb*yb)
            X=x/a;Y=y/b
            distances[k,i]=(X*X+Y*Y-1)/max(gradient,1e-6)
    result=np.empty((len(points),5),np.float64)
    for i in range(len(points)):
        for k in range(5):result[i,k]=(distances[2*k,i]-distances[2*k+1,i])/(2*steps[k])
    return result


class JITUncertainty:
    def __init__(self):
        begin=time.perf_counter();steps=np.array([.05,.05,.05,.05,.001])
        p=np.array([0.,0.,10.,4.,.2]);parameters=np.repeat(p[None],10,axis=0)
        parameters[::2]+=np.diag(steps);parameters[1::2]-=np.diag(steps)
        central_jacobian(np.empty((0,2)),parameters,np.cos(parameters[:,4]),np.sin(parameters[:,4]),steps)
        self.provenance=dict(jacobian_kernel_warm_ms=(time.perf_counter()-begin)*1000,
            method='same10central_difference_points_steps_and_NumPy_trigonometry',fastmath=False,
            covariance_and_SVD_definitions_unchanged=True)

    def __call__(self,points,p):
        if len(points)>96:points=points[np.linspace(0,len(points)-1,96).astype(int)]
        residual=_distance(points,p);sigma=max(.8,1.4826*np.median(abs(residual-np.median(residual))))
        steps=np.array([.05,.05,.05,.05,.001]);parameters=np.repeat(p[None],10,axis=0)
        parameters[::2]+=np.diag(steps);parameters[1::2]-=np.diag(steps)
        jacobian=central_jacobian(points,parameters,np.cos(parameters[:,4]),np.sin(parameters[:,4]),steps)
        covariance=np.linalg.pinv(jacobian.T@jacobian,rcond=1e-10)*sigma**2*4
        singular=np.linalg.svd(jacobian*np.array([p[2],p[2],p[2],p[2],1.]),compute_uv=False)
        condition=float(singular[0]/max(singular[-1],1e-9))
        center_sigma=float(np.sqrt(max(np.linalg.eigvalsh(covariance[:2,:2])[-1],.25)))
        angle_sigma=float(np.rad2deg(np.sqrt(max(covariance[4,4],0))))
        return min(center_sigma,500.),min(angle_sigma,90.),condition
