"""Visibility-aware relative blade-frequency input, not a Newton thrust sensor.

Only accepted current conics are associated to five-circle model slots. A
partial set estimates a COMMON scale of the previously observed per-slot f^2;
hidden frequencies are never output as measurements. Axis-only views use the
qualified whole-body period, never a changing average of unidentified blobs.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def force_input(structure,relative,coherent):
    if relative is None or not coherent or structure.get('axis_xy') is None:
        return None,0.
    n=np.asarray(structure['axis_xy'],float)
    if n.shape!=(2,) or not np.isfinite(n).all() or not .2<=relative<=5.:
        return None,0.
    if n[1]>0:n=-n
    return relative*n,max(.35,float(structure.get('geometric_weight',0.)))


class LegacyPeriodProxy:
    """Exact previous selector retained for matched-input controls."""
    def __init__(self):self.reference=None;self.previous=None
    def step(self,t,structure,rotors,rates,pooled):
        fs=[r['frequency_hz'] for r in rates if r.get('valid')]
        f=float(np.sqrt(np.mean(np.square(fs)))) if len(fs)>=2 else pooled.get('frequency_hz') if pooled.get('qualified') else None
        source=('four_current_squared_frequencies' if len(fs)==4 and structure['mode']=='four' else
            'visible_frequency_squared_mean_hidden_equal_mean_hypothesis' if len(fs)>=2 else
            'pooled_flicker_representative_speed_hypothesis')
        coherent=False;relative=None
        if f is not None:
            coherent=self.previous is not None and t-self.previous[0]<.12 and abs(np.log(f/self.previous[1]))<.08
            self.previous=(t,f)
            if coherent and structure.get('axis_xy') is not None:
                if self.reference is None:self.reference=f
                relative=(f/self.reference)**2
        u,quality=force_input(structure,relative,coherent)
        return u,quality,dict(source=source,frequency_hz=f,relative_f2=relative,
            coherent=bool(coherent),reason='current_coherent_relative_frequency_and_pose' if u is not None else 'no_current_qualified_frequency')


class VisibilityPeriodProxy:
    gap_s=.12
    def __init__(self):
        self.epoch=None;self.last_t=None;self.baseline=None;self.model_anchor=1.
        self.previous_slots={};self.previous_pool=None;self.pool_reference=None
        self.previous_output=None;self.previous_source=None

    def _slots(self,structure,rotors,rates):
        info=dict(mapped_model_ids=[],assignments=[],rejected_reason=None)
        hubs=structure.get('model_centers');observed=structure.get('observed_model_ids',[])
        if structure.get('mode') not in ('four','three','partial_two') or hubs is None or not observed:
            return {},dict(info,rejected_reason='no_current_identified_nonflat_geometry')
        hubs=np.asarray(hubs,float)
        if hubs.shape!=(4,2) or not np.isfinite(hubs).all():return {},dict(info,rejected_reason='invalid_template')
        observed=sorted(set(int(i) for i in observed if 0<=int(i)<4))
        candidates=[]
        for j,(r,rate) in enumerate(zip(rotors,rates)):
            good=(r.get('current_observation',True) and not r.get('fov_clipped') and
                not r.get('association_ambiguous') and r.get('association_state')!='relock_pending' and
                (r.get('mode')=='ellipse' or r.get('arc_center_qualified') or r.get('curvature_validated')) and
                r.get('ratio',0.)>.16 and rate.get('valid'))
            f=rate.get('frequency_hz')
            if good and f is not None and np.isfinite(f) and 60<=f<=1200:candidates.append(j)
        if len(candidates)<2:return {},dict(info,rejected_reason='fewer_than_two_current_slot_periods')
        centers=np.array([[rotors[j]['cx'],rotors[j]['cy']] for j in candidates])
        distance=np.linalg.norm(hubs[observed,None]-centers[None],axis=2)
        ii,jj=linear_sum_assignment(distance);samples={}
        for a,b in zip(ii,jj):
            j=candidates[b];r=rotors[j];slot=observed[a]
            gate=max(4.,.15*float(r['major']))
            others=np.delete(np.linalg.norm(hubs-centers[b],axis=1),slot)
            margin=float(others.min()-distance[a,b])
            if distance[a,b]>gate or margin<2.:continue
            samples[slot]=float(rates[j]['frequency_hz'])
            info['assignments'].append(dict(model_id=slot,region_index=j,distance_px=float(distance[a,b]),margin_px=margin))
        info['mapped_model_ids']=sorted(samples)
        return samples,info

    def step(self,t,structure,rotors,rates,pooled):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Increasing causal timestamps required')
        epoch=structure.get('geometry_epoch',0)
        if (self.epoch is not None and epoch!=self.epoch) or (self.last_t is not None and t-self.last_t>.5):self.__init__()
        self.last_t=t;self.epoch=epoch
        samples,mapping=self._slots(structure,rotors,rates)
        stable={}
        for i,f in samples.items():
            old=self.previous_slots.get(i)
            if old is not None and t-old[0]<self.gap_s and abs(np.log(f/old[1]))<.08:stable[i]=f
            self.previous_slots[i]=(t,f)
        pf=pooled.get('frequency_hz') if pooled.get('qualified') else None
        pool_good=False;pool_relative=None
        if pf is not None and np.isfinite(pf) and 60<=pf<=1200:
            old=self.previous_pool
            pool_good=old is not None and t-old[0]<self.gap_s and abs(np.log(pf/old[1]))<.08
            self.previous_pool=(t,float(pf))
            if pool_good:
                if self.pool_reference is None:self.pool_reference=float(pf*pf)
                pool_relative=float(pf*pf/self.pool_reference)
        initialized=False
        if self.baseline is None and structure.get('mode')=='four' and len(stable)==4:
            self.baseline={i:f*f for i,f in stable.items()}
            self.model_anchor=pool_relative if pool_relative is not None else 1.
            initialized=True
        relative=None;source='unavailable'
        if self.baseline is not None and len(stable)>=2:
            ids=sorted(stable)
            relative=float(self.model_anchor*sum(stable[i]**2 for i in ids)/sum(self.baseline[i] for i in ids))
            source='four_slot_f2_sum' if len(ids)==4 else 'partial_slot_common_scale_hypothesis'
            # Calibrate the pooled gauge only at a simultaneous identified
            # observation. Missing motors are not filled with made-up rates.
            if pool_good and relative>0:self.pool_reference=float(pf*pf/relative)
        elif pool_good:
            relative=pool_relative;source='pooled_axis_separate_gauge'
        coherent=False
        if relative is not None and np.isfinite(relative) and relative>0:
            old=self.previous_output
            coherent=old is not None and t-old[0]<self.gap_s and abs(.5*np.log(relative/old[1]))<.08
            self.previous_output=(t,relative)
        u,quality=force_input(structure,relative,coherent)
        switched=self.previous_source is not None and source!=self.previous_source
        if relative is not None:self.previous_source=source
        info=dict(source=source,frequency_hz=None,relative_f2=relative,coherent=bool(coherent),
            reason='current_visibility_conditioned_period_and_pose' if u is not None else 'warming_missing_or_incoherent_period_or_axis',
            slot_baseline_initialized=initialized,calibrated_model_slots=[] if self.baseline is None else sorted(self.baseline),
            current_model_ids=sorted(stable),mapping=mapping,source_changed=switched,
            full_sum_observed=source=='four_slot_f2_sum',hidden_frequencies_measured=0,
            partial_sum_requires_common_scale=source=='partial_slot_common_scale_hypothesis',
            differential_hidden_rotor_changes_observable=False,mechanical_RPM_measured=False,
            pooled_gauge_cross_calibrated=bool(self.baseline is not None and len(stable)>=2 and pool_good))
        return u,quality,info


class ConservativeSlotPeriodProxy:
    """Correct identified visibility changes; keep the old unmodelled side route.

The separate pooled-side estimator remains an explicit experimental control:
its first real pilot worsened mean forecast error. This class keeps the actual
slot-normalization fix without silently deploying that unrelated side change.
"""
    def __init__(self):
        self.legacy=LegacyPeriodProxy();self.slots=VisibilityPeriodProxy();self.anchor=None

    def step(self,t,structure,rotors,rates,pooled):
        old_u,old_q,old=self.legacy.step(t,structure,rotors,rates,pooled)
        u,q,new=self.slots.step(t,structure,rotors,rates,pooled)
        if not new['calibrated_model_slots']:self.anchor=None
        identified=new['source'] in ('four_slot_f2_sum','partial_slot_common_scale_hypothesis')
        if identified:
            if self.anchor is None:
                if old['relative_f2'] is None:return old_u,old_q,dict(old,slot_normalization_applied=False)
                self.anchor=old['relative_f2']/new['relative_f2']
            relative=new['relative_f2']*self.anchor
            u,q=force_input(structure,relative,new['coherent'])
            return u,q,dict(new,relative_f2=relative,slot_normalization_applied=True,
                old_reference_gauge_preserved=True)
        if new['calibrated_model_slots'] and len(new['mapping']['mapped_model_ids'])>=2:
            # A returning slot first needs temporal consistency. Do not fill
            # its one-frame warm-up with the old composition-biased average.
            return None,0.,dict(new,source='identified_slots_relock_warmup',relative_f2=None,
                slot_normalization_applied=False,reason='awaiting_current_slot_temporal_confirmation')
        return old_u,old_q,dict(old,slot_normalization_applied=False,
            preserved_unidentified_side_control=True,slot_mapping=new['mapping'])
