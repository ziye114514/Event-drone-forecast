"""Batched helical-phase evidence after shared five-circle affine rectification.

Experimental observation only. Frequency is blade-phase frequency, not verified
mechanical RPM. Phase/time consistency follows HelixTrack's measurement idea;
shared geometry follows the existing model/CoHelix idea. No author code copied.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class SharedPhaseFFT:
    window_us=160000;bin_us=100;capacity=4096

    def __init__(self):
        self.reference=None;self.shapes=None;self.epoch=None;self.last_informative=None
        self.started=None;self.last_t=None
        self.ids=np.full(self.capacity,-1,np.int64)
        self.stream=np.zeros((4,2,self.capacity),complex)
        self.weight=np.zeros((4,2,self.capacity),float)

    def step(self,e,t,body,structure,rotors,velocity):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing cutoffs required')
        epoch=structure.get('geometry_epoch',0)
        if self.epoch is not None and epoch!=self.epoch:self.__init__()
        self.epoch=epoch;self.last_t=t
        empty=dict(initialized=self.reference is not None,rotors=[],qualified_count=0,
            mechanical_RPM_measured=False,force_measured=False,new_hidden_frequencies=0)
        if not body.get('detected'):return dict(empty,reason='no_current_body')
        hubs=structure.get('model_centers')
        if hubs is None or len(hubs)!=4:return dict(empty,reason='no_shared_template')
        hubs=np.asarray(hubs,float);mode=structure['mode']
        if self.reference is None:
            qualified=[r for r in rotors if not r.get('fov_clipped') and
                (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated'))]
            if mode!='four' or len(qualified)<4:return dict(empty,reason='awaiting_current_four_calibration')
            centers=np.array([[r['cx'],r['cy']] for r in qualified])
            cost=np.linalg.norm(hubs[:,None]-centers[None],axis=2);ii,jj=linear_sum_assignment(cost)
            if len(ii)!=4 or any(cost[i,j]>max(4.,.15*qualified[j]['major']) for i,j in zip(ii,jj)):
                return dict(empty,reason='ambiguous_initial_conics')
            ordered=[qualified[j] for j in jj]
            if min(r['ratio'] for r in ordered)<=.16:return dict(empty,reason='initial_plane_too_flat')
            self.reference=hubs.copy();self.shapes=[]
            for r in ordered:
                a=np.deg2rad(r['angle']);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                self.shapes.append((np.array([r['cx'],r['cy']]),R,np.array([r['major']/2,r['minor']/2])))
            self.started=t
        if mode in ('four','three'):
            independent=sum(not r.get('shared_arc_recovered',False) for r in rotors)
            if independent>=2:self.last_informative=t
        if (mode not in ('four','three','partial_two') or self.last_informative is None
                or t-self.last_informative>.12+1e-9):
            return dict(empty,initialized=True,reason='no_current_observable_plane')
        D=np.c_[hubs,np.ones(4)];T=np.linalg.lstsq(D,self.reference,rcond=None)[0]
        singular=np.linalg.svd(T[:2],compute_uv=False)
        if singular.min()<.1 or singular.max()>10 or singular.max()/singular.min()>20:
            return dict(empty,initialized=True,reason='ill_conditioned_shared_rectification')
        observed=set(structure.get('observed_model_ids',[]))
        if not observed:return dict(empty,initialized=True,reason='no_current_observed_model_slots')
        box=np.asarray(body['bbox'],float);pad=np.maximum((box[2:]-box[:2])*.35,12.)
        keep=(e['x']>=box[0]-pad[0])&(e['x']<box[2]+pad[0])&(e['y']>=box[1]-pad[1])&(e['y']<box[3]+pad[1])
        e=e[keep]
        if not len(e):return dict(empty,initialized=True,reason='no_current_ROI_events')
        xy=np.c_[e['x'],e['y']].astype(float)+(t-e['t']/1e6)[:,None]*np.asarray(velocity)
        aligned=xy@T[:2]+T[2]
        normalized=[(aligned-center)@R/axes for center,R,axes in self.shapes]
        radius=np.array([np.linalg.norm(z,axis=1) for z in normalized])
        distances=np.abs(radius-1.);best=np.argmin(distances,axis=0)
        # For four values, the second minimum needs pairwise minima/maxima,
        # not one tiny strided sort per event. Tied distances still have0margin.
        low01=np.minimum(distances[0],distances[1]);high01=np.maximum(distances[0],distances[1])
        low23=np.minimum(distances[2],distances[3]);high23=np.maximum(distances[2],distances[3])
        second=np.minimum(np.minimum(high01,high23),np.maximum(low01,low23))
        margin=second-np.minimum(low01,low23)
        absolute=e['t']//self.bin_us;unique=np.unique(absolute);slots=unique%self.capacity
        changed=slots[self.ids[slots]!=unique]
        self.stream[:,:,changed]=0;self.weight[:,:,changed]=0;self.ids[slots]=unique
        current_sectors={};counts={}
        for i in observed:
            selected=(best==i)&(margin>=.1)&(radius[i]>=.2)&(radius[i]<=1.5)
            r=radius[i,selected];z=normalized[i][selected]
            theta=np.arctan2(z[:,1],z[:,0]);w=np.exp(-.5*((r-1)/.15)**2)
            index=e['p'][selected].astype(np.int64)*self.capacity+absolute[selected]%self.capacity
            signal=w*np.exp(2j*theta)
            self.stream[i]+= (np.bincount(index,weights=signal.real,minlength=2*self.capacity)+
                1j*np.bincount(index,weights=signal.imag,minlength=2*self.capacity)).reshape(2,self.capacity)
            self.weight[i]+=np.bincount(index,weights=w,minlength=2*self.capacity).reshape(2,self.capacity)
            informative=w>=.1
            current_sectors[i]=len(np.unique(np.floor((theta[informative]+np.pi)*12/(2*np.pi)).astype(int)%12))
            counts[i]=int(informative.sum())
        end=int(np.ceil(t*1e6/self.bin_us));n=self.window_us//self.bin_us
        bins=np.arange(end-n,end);slots=bins%self.capacity;valid=self.ids[slots]==bins
        frequency=np.fft.fftfreq(n,self.bin_us/1e6);allowed=(abs(frequency)>=60)&(abs(frequency)<=1200)
        candidates=np.flatnonzero(allowed);window=np.hanning(n);result=[]
        for i in range(4):
            q=dict(model_id=i,qualified=False,current_slot_observed=i in observed,frequency_hz=None)
            if i not in observed:
                result.append(dict(q,reason='hidden_slot_not_measured'));continue
            signal=self.stream[i][:,slots]*valid;weight=self.weight[i][:,slots]*valid
            total=float(weight.sum());sectors=current_sectors.get(i,0)
            if t-self.started<.6*self.window_us/1e6 or total<100 or counts.get(i,0)<32:
                result.append(dict(q,reason='warming_or_sparse',current_events=counts.get(i,0)));continue
            power=np.sum(abs(np.fft.fft((signal-signal.mean(1,keepdims=True))*window,axis=1))**2,axis=0)
            k=int(candidates[np.argmax(power[candidates])]);noise=max(float(np.median(power[candidates])),1e-12)
            logp=np.log(np.maximum(power[[(k-1)%n,k,(k+1)%n]],1e-12));den=logp[0]-2*logp[1]+logp[2]
            shift=0. if den>=-1e-12 else float(np.clip(.5*(logp[0]-logp[2])/den,-.5,.5))
            f=float(frequency[k]+shift*1e6/self.window_us)
            times=(bins+.5)*self.bin_us/1e6
            concentration=float(np.abs(np.sum(signal*np.exp(-2j*np.pi*f*times),axis=1)).sum()/max(total,1e-9))
            # Static spatial structure under a common flicker has a real
            # temporal envelope: its two signed-frequency powers are equal.
            # A rotating spatial moment should favor one direction. Require
            # a fixed3:1 signed-power contrast, not just a temporal peak.
            mirror=float(power[(-k)%n]);directionality=float((power[k]-mirror)/max(power[k]+mirror,1e-12))
            snr=float(power[k]/noise);qualified=bool(snr>=8 and concentration>=.5 and sectors>=9 and directionality>=.5)
            result.append(dict(q,qualified=qualified,reason='current_spatial_phase_supported' if qualified else 'weak_spatial_phase',
                frequency_hz=abs(f) if qualified else None,candidate_phase_hz=abs(f),signed_phase_hz=f,
                phase_concentration=concentration,peak_to_noise=snr,current_sectors=sectors,
                signed_power_directionality=directionality,
                current_events=counts.get(i,0),blade_count_assumed=2,harmonic_identity_unverified=True))
        return dict(empty,initialized=True,reason='current_shared_rectified_phase_FFT',rotors=result,
            qualified_count=sum(q['qualified'] for q in result),ROI_events=len(e),
            shared_affine_condition=float(singular.max()/singular.min()),
            window_us=self.window_us,bin_us=self.bin_us,geometry_source='current_model_not_future_truth')
