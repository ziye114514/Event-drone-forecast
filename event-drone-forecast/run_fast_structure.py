"""Minimal mechanism3 RAW pilot:30Hz observations,10Hz NN, current force response.

Thirty means fresh source updates, not repeated forecasts or video playback.
The offline capacity and paced deadline results are reported separately.
"""
from pathlib import Path
import argparse
import gc
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory
# Bound tiny matrix BLAS work before importing NumPy/SciPy. Large default BLAS
# thread teams contend with CUDA preprocessing and introduce latency spikes.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src')]
from live_decode import native_batches
from forecast_runtime import resolve
from fast_structure_front import FastStructure,roi_bounds,NativeROI,rotor_rates
from fast_force_response import FastForceResponse
from visibility_period import LegacyPeriodProxy,VisibilityPeriodProxy,ConservativeSlotPeriodProxy
from live_frequency_pixels import latest_pixels
from live_scale_rollout import ScaleRolloutMotion,scale_predictions
from live_structure_bridge import QualifiedCenter,GeometryMotion
import frequency_features as native79


def js(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())


def receive(p):
    for _ in range(64):
        s=p.stdout.readline()
        if not s:raise RuntimeError('Worker exited; inspect gpu_worker.log')
        try:return json.loads(s)
        except json.JSONDecodeError:pass
    raise RuntimeError('GPU protocol failure')


def track(previous,current,body,rigid_motion=False,center_hint=None):
    """Current-image translation only; no inertial extrapolation called detection."""
    if previous is None or body is None or not body.get('detected'):return None,dict(reason='no_previous_ROI')
    box=np.asarray(body['bbox']);x,y,X,Y=roi_bounds(box,padding=.35)
    if X-x<12 or Y-y<12:return None,dict(reason='previous_ROI_outside_sensor_or_too_small')
    a=previous[y:Y,x:X];b=current[y:Y,x:X]
    mask=np.zeros(a.shape,np.uint8);lo=np.maximum(np.floor(box[:2]).astype(int)-[x,y],0)
    hi=np.minimum(np.ceil(box[2:]).astype(int)-[x,y],[X-x,Y-y]);mask[lo[1]:hi[1],lo[0]:hi[0]]=255
    points=cv2.goodFeaturesToTrack(a,60,.01,4,mask=mask)
    if points is None or len(points)<6:return None,dict(reason='too_few_current_features')
    q,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(15,15),maxLevel=2)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(15,15),maxLevel=2)
    d=(q-points).reshape(-1,2);fb=np.linalg.norm((back-points).reshape(-1,2),axis=1)
    keep=ok.ravel().astype(bool)&ok2.ravel().astype(bool)&(fb<=1.5)&np.isfinite(d).all(axis=1)
    if keep.sum()<6:return None,dict(reason='failed_forward_backward_support',points=int(keep.sum()))
    delta=np.median(d[keep],axis=0);scatter=np.linalg.norm(d-delta,axis=1)
    keep&=scatter<=max(2.,3.*float(np.median(scatter[keep])))
    if keep.sum()<6:return None,dict(reason='incoherent_translation')
    delta=np.median(d[keep],axis=0)
    if np.linalg.norm(delta)>.4*max(box[2:]-box[:2]):return None,dict(reason='translation_too_large')
    shifted=box+np.tile(delta,2)
    if shifted[0]>=current.shape[1] or shifted[1]>=current.shape[0] or shifted[2]<=0 or shifted[3]<=0:
        return None,dict(reason='current_translated_ROI_outside_sensor')
    result=dict(body,bbox=(box+np.tile(delta,2)).tolist(),center=(np.asarray(body['center'])+delta).tolist(),
        measurement_source='current_KLT_translation',detected=True)
    covariance=np.cov(d[keep].T)+np.eye(2)*max(1.,float(np.median(fb[keep]**2)))
    info=dict(reason='current_image_translation',points=int(keep.sum()),delta=delta.tolist(),
        displacement_covariance_px2=covariance.tolist(),covariance_uses_full_track_scatter_without_independence_gain=True)
    if rigid_motion:
        from live_local_flow import reused_similarity
        start=time.perf_counter();origin=np.array([x,y])
        info['similarity_motion']=reused_similarity(points.reshape(-1,2)[keep]+origin,q.reshape(-1,2)[keep]+origin,fb[keep],
            body['center'] if center_hint is None else center_hint)
        info['similarity_motion']['CPU_ms']=(time.perf_counter()-start)*1000
        info['similarity_motion']['evaluation_center_source']='previous_qualified_five_circle_center' if center_hint is not None else 'previous_body_box_center'
    return result,info


class PhysicalPositionState:
    """Only p/v needed by mechanics, with no parallel acceleration/scale fit."""
    def __init__(self):
        from live_structure_bridge import NoiseKF
        self.kf=NoiseKF();self.seen=None;self.observations=0
    def step(self,t,body):
        detected=bool(body.get('detected'))
        if self.seen is not None and t-self.seen>.5:self.__init__()
        if detected:self.seen=t;self.observations+=1
        c=self.kf.step(t,body.get('center') if detected else None)
        valid=bool(c is not None and self.observations>=5 and self.seen is not None and t-self.seen<=.16)
        return c,valid,dict(position_velocity_initial_state_only=True,parallel_kinematic_acceleration=False)


def run(args):
    pipeline_started=time.perf_counter()
    raw=Path(args.raw).resolve();out=Path(args.output).resolve()
    # Private workspace guards are not part of the public source export.
    if out.exists():raise FileExistsError(out)
    if args.duration<=0 or args.hz<=0:raise ValueError('Positive duration/hz required')
    runtime=resolve(args.gpu_python,args.detector_weights,args.detector_sha256,args.cache_dir)
    from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
    from metavision_sdk_core import BaseFrameGenerationAlgorithm
    cv2.setNumThreads(2)
    out.mkdir(parents=True);snapshot=out/'source_snapshot';snapshot.mkdir()
    sources={}
    for f in ['arc_parameters.py', 'axis_amplitude_bridge.py', 'body_association.py', 'body_axis.py', 'continuous_five_circle.py', 'curvature_split.py', 'fast_force_response.py', 'fast_geometry_compute.py', 'fast_raw_reader.py', 'fast_structure_front.py', 'five_circle_model.py', 'flow_force_balance.py', 'forecast.py', 'forecast_runtime.py', 'frequency_features.py', 'live_center_ensemble.py', 'live_covariance.py', 'live_decode.py', 'live_flow_acceleration.py', 'live_frequency_pixels.py', 'live_local_flow.py', 'live_measurement_kf.py', 'live_motion.py', 'live_pixel_axis.py', 'live_roi_refine.py', 'live_scale_rollout.py', 'live_structure_bridge.py', 'live_worker.py', 'lumped_disturbance.py', 'native/sdk_activity_avx2.cpp', 'native_axis.py', 'native_counts_numba.py', 'native_geometry_numba.py', 'native_latest_frequency.py', 'native_render.py', 'partial_arc_information.py', 'period_process.py', 'projected_unknown_force.py', 'responsive_force.py', 'rotor_geometry.py', 'run_fast_structure.py', 'sdk_activity_backend.py', 'shared_arc_recovery.py', 'shared_image_motion.py', 'shared_phase_fft.py', 'src/event_drone_forecast/__init__.py', 'src/event_drone_forecast/_vendor/__init__.py', 'src/event_drone_forecast/_vendor/filterpy_linear_upstream/helpers.py', 'src/event_drone_forecast/_vendor/filterpy_linear_upstream/kalman_filter.py', 'src/event_drone_forecast/_vendor/filterpy_process_upstream/discretization.py', 'src/event_drone_forecast/detector_backend.py', 'src/event_drone_forecast/geometry.py', 'src/event_drone_forecast/position/__init__.py', 'src/event_drone_forecast/position/_filterpy.py', 'src/event_drone_forecast/position/covariance.py', 'src/event_drone_forecast/position/kalman.py', 'src/event_drone_forecast/position/motion.py', 'src/event_drone_forecast/position/robust.py', 'src/event_drone_forecast/position/rollout.py', 'src/event_drone_forecast/raw_io.py', 'visibility_period.py']:
        content=(ROOT/f).read_bytes();p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        sources[f]=hashlib.sha256(content).hexdigest()
    stats=raw.stat()
    if args.sdk_avx2 and not args.prefilter_decode:
        from sdk_activity_backend import SDKActivityAVX2
        noise=SDKActivityAVX2(1280,720,10000)
    else:noise=ActivityNoiseFilterAlgorithm(1280,720,10000)
    clean_buffer=noise.get_empty_output_buffer()
    from live_pixel_axis import SDKFixedPixelAxis
    periodic=SDKFixedPixelAxis(1280,720);period_roi=NativeROI()
    frequency_latest=None
    if args.jit_counts and not args.process_period:
        from native_latest_frequency import LatestFrequency
        frequency_latest=LatestFrequency(1280,720,periodic.output.numpy().dtype)
    structure=FastStructure(args.reuse_conic_uncertainty,not args.legacy_structure,not args.no_shared_arc_recovery,args.guide_full_only,args.bipolar_guide_area and not args.strict_guide_area,not args.strict_guide_area,not args.no_current_shape_feedback,args.partial_arc_information,optimized_compute=args.optimized_compute,side_axis_budget=args.side_axis_budget,native_histogram=args.native_histogram,jit_counts=args.jit_counts,jit_conics=args.jit_conics);motion=GeometryMotion(args.hz) if args.with_kinematic_control else PhysicalPositionState()
    image_motion=None
    if args.reuse_image_motion:
        from shared_image_motion import SharedImageMotion
        image_motion=SharedImageMotion(reanchor_velocity=not args.legacy_motion_anchor_gate)
    amplitude_bridge=None
    if args.axis_amplitude_bridge:
        from axis_amplitude_bridge import AxisAmplitudeBridge
        amplitude_bridge=AxisAmplitudeBridge()
    force_class=FastForceResponse
    if args.responsive_force:
        from responsive_force import DampedResponsiveForce
        force_class=DampedResponsiveForce
    if args.flow_force_balance:
        from flow_force_balance import FlowBalancedForce
        force_class=FlowBalancedForce
    if args.tangent_disturbance:
        from projected_unknown_force import ProjectedResidualForce
        force_class=ProjectedResidualForce
    centering=QualifiedCenter();force=force_class(early_disturbance=not args.delayed_disturbance_control,
        hold_acceleration_gaps=not args.no_force_gap_hold and not args.delayed_disturbance_control,
        gain_signal_to_noise=0. if args.legacy_gain_quality or args.delayed_disturbance_control else 2.)
    period_selector=(VisibilityPeriodProxy() if getattr(args,'pooled_axis_period',False) else
        ConservativeSlotPeriodProxy() if getattr(args,'visibility_period',False) else LegacyPeriodProxy())
    memory=gpu=log=video=None;records=[];counter=0;nn_count=0;previous=None;body=None;last_nn=None
    pool=ThreadPoolExecutor(max_workers=1);reader_pool=ThreadPoolExecutor(max_workers=1)
    period_pool=ThreadPoolExecutor(max_workers=1)
    raw_process=None;period_process=None;read_future=None;pending=None;pending_gray=None;pending_t=None;pending_hash=None;pending_reference=None
    coeff=None;first_decode=True;roi_memory=None
    protocol=dict(entry='run_fast_structure.py',strategy=3,target_fps=args.hz,nn_stride=args.nn_stride,
        native_step_us=round(1e6/args.hz),geometry_window_us=20000,paced=args.paced,async_nn=args.async_nn,
        physical_only=not args.with_kinematic_control,periodic_sampling='Existing SDKFixedPixelAxis seed141000 fixed25% pixels; ALL native times at each retained pixel',
        period_selector=type(period_selector).__name__,optimized_compute=args.optimized_compute,
        prefilter_in_decoder=args.prefilter_decode,
        parallel_period_geometry=args.parallel_period,
        adaptive_period_execution=args.adaptive_period,
        separate_period_process=args.process_period,
        side_axis_budget=args.side_axis_budget,force_model=type(force).__name__,
        goal_profile=args.goal_profile,
        histogram_backend='Numba_fused_ROI_count' if args.jit_counts else 'OpenCV_float32_exact_integer_counts' if args.native_histogram else 'NumPy_int64_then_float32',
        count_kernel_provenance=None if structure.count_current is None else structure.count_current.provenance,
        conic_kernel_provenance=None if structure.uncertainty_current is None else structure.uncertainty_current.provenance,
        latest_frequency_kernel_provenance=None if frequency_latest is None else frequency_latest.provenance,
        frame_hash_verification=args.verify_frames,reuse_current_image_motion=args.reuse_image_motion,
        same_window_flow_force_balance=args.flow_force_balance,
        current_axis_with_recent_amplitude=args.axis_amplitude_bridge,
        current_strong_reanchor_reuses_velocity=not args.legacy_motion_anchor_gate,
        reuse_rigid_current_track_pairs=args.rigid_motion,
        partial_tangent_unknown_force=args.tangent_disturbance,
        structure_guided_NN_cadence=args.structure_guided_nn,
        startup_clock_scope='run entry, includes optional JIT import/compile and module initialization',
        current_box_association=args.body_association,
        current_NN_when_KLT_unavailable=args.current_nn_on_loss,roi_recovery_after_current_full_frame_miss=args.roi_recovery,
        decode_prefetch=not args.serial_decode,decode_queue_capacity=2 if not args.thread_decode and not args.serial_decode else 1,drop_source_frames=False,
        decoder_overflow_policy='bounded queued immutable packet; preserve all native events',
        raw_read_chunk=args.raw_read_chunk,
        defer_cyclic_GC_during_finite_stream=args.defer_cyclic_gc,
        local_SDK_AVX2_filter=args.sdk_avx2,
        conic_uncertainty_reuse=args.reuse_conic_uncertainty,
        continuous_shared_geometry=not args.legacy_structure,
        shared_current_arc_recovery=not args.no_shared_arc_recovery,
        guide_reference_policy='full_only' if args.guide_full_only else 'rank6_current_anchors_and_hub_uncertainty',
        guide_area_source='existing124_binary_bipolar_occupancy' if args.bipolar_guide_area and not args.strict_guide_area else 'strong_count_threshold',
        guide_area_evidence='union_iou_only' if args.strict_guide_area else 'union_iou_or_current_coverage_and_ring_contrast',
        current_conic_shape_feedback=not args.no_current_shape_feedback,
        partial_arc_information=args.partial_arc_information,
        disturbance_learning='first_matured_0.2s_row' if not args.delayed_disturbance_control else 'historical_8row_gate',
        force_reference_gauge='adjusted_position_measurement' if not args.delayed_disturbance_control else 'historical_raw_body_center',
        force_gap_hold_s=.12 if not args.no_force_gap_hold and not args.delayed_disturbance_control else 0.,
        gain_signal_to_noise_minimum=0. if args.legacy_gain_quality or args.delayed_disturbance_control else 2.,
        geometry_input='reuse_current_expanded_frequency_ROI; all native times inside geometry ROI retained',
        separate_decoder_process=not args.thread_decode and not args.serial_decode,
        raw=str(raw),raw_size=stats.st_size,raw_mtime_ns=stats.st_mtime_ns,start=args.start,duration=args.duration,
        sources=sources,runtime=runtime,input='RAW and fixed detector only; no labels or output cache read',
        input_force='relative flicker squared times image normal; sign uses image-up prior',
        limits=['2D weak projection','no true RPM, metric force or external pose truth',
                'short pilot; no claim of maneuver accuracy before scoring',
                'KLT translations distinguishable from fresh NN boxes','no default v7 replacement'])
    step_us=round(1e6/args.hz)
    # Identical source alignment in both compute paths.
    aligned_start=(round(args.start*1e6)//step_us)*step_us/1e6
    aligned_duration=(round(args.duration*1e6)//step_us)*step_us/1e6
    protocol.update(actual_iterator_start=aligned_start,actual_duration=aligned_duration)
    iterator=iter(native_batches(raw,aligned_start,aligned_duration,False,step_us,read_chunk=args.raw_read_chunk))
    def read_owned():
        before=time.perf_counter();packet=next(iterator,None)
        if packet is None:return None
        stamp,e=packet
        if not e.flags.c_contiguous:raise RuntimeError('Expected contiguous native event batch')
        owned=e.view(np.uint8).copy().view(e.dtype);owned.setflags(write=False)
        return stamp,owned,(time.perf_counter()-before)*1000
    def start_reader():
        nonlocal raw_process,period_process,read_future
        protocol['decoder_started_after_s']=time.perf_counter()-pipeline_started
        if not args.thread_decode and not args.serial_decode:
            from fast_raw_reader import ProcessRAW
            raw_process=ProcessRAW(raw,aligned_start,aligned_duration,step_us,filter_events=args.prefilter_decode,read_chunk=args.raw_read_chunk,avx2_filter=args.sdk_avx2)
        elif not args.serial_decode:read_future=reader_pool.submit(read_owned)
        if args.process_period:
            from period_process import PeriodProcess
            period_process=PeriodProcess(jit=args.jit_counts)
    def measure_period(selected,t):
        if period_process is not None:return period_process.compute(selected,t)
        started=time.perf_counter();q=periodic.feed(selected,t)
        q=q[q['t']>=round(t*1e6)-20000];pooled_input=q
        in_range=bool(np.all((q['frequency']>=60.)&(q['frequency']<=1200.)))
        q=latest_pixels(q,1280) if frequency_latest is None else frequency_latest(q)
        if in_range:pooled_input=q
        return q,pooled_input,in_range,(time.perf_counter()-started)*1000
    gc_was_enabled=gc.isenabled();gc_events=[];gc_started={};gc_callback=None
    try:
        if args.optimized_compute:start_reader()
        memory=shared_memory.SharedMemory(create=True,size=720*1280*3)
        shared_image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf);shared_image[:]=52
        image=np.empty((720,1280,3),np.uint8)
        log=(out/'gpu_worker.log').open('x',encoding='utf-8')
        worker_args=[runtime['gpu_python'],str(ROOT/'live_worker.py'),'--shared-memory',memory.name,
            '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],
            '--cache-dir',runtime['cache_dir']]
        if args.body_association:worker_args+=['--body-association','--frame-rate',str(args.hz/args.nn_stride)]
        gpu=subprocess.Popen(worker_args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        protocol['gpu_ready']=receive(gpu)
        for _ in range(3):
            gpu.stdin.write(js(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();receive(gpu);counter+=1
        protocol['gpu_ready_after_s']=time.perf_counter()-pipeline_started
        if period_process is not None:protocol['period_worker_provenance']=period_process.receive_ready()
        (out/'protocol.json').write_text(js(protocol),'utf-8')
        if args.preview:
            video=cv2.VideoWriter(str(out/'predictions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.hz,(1280,720))
            if not video.isOpened():raise RuntimeError('Cannot open video writer')
        if not args.optimized_compute:start_reader()
        if args.defer_cyclic_gc:gc.collect();gc.disable()
        if args.audit_gc or args.defer_cyclic_gc:
            def gc_callback(phase,info):
                generation=info['generation']
                if phase=='start':gc_started[generation]=time.perf_counter()
                elif generation in gc_started:
                    gc_events.append(dict(generation=generation,completed_after_s=time.perf_counter()-pipeline_started,
                        duration_ms=(time.perf_counter()-gc_started.pop(generation))*1000,collected=info['collected']))
            gc.callbacks.append(gc_callback)
        epoch=None;last_state_completed_wall=None
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream:
            while True:
                read_start=time.perf_counter()
                if raw_process is not None:packet=raw_process.read()
                elif args.serial_decode:packet=read_owned()
                else:
                    packet=read_future.result()
                    if packet is not None:read_future=reader_pool.submit(read_owned)
                if packet is None:break
                t,events,producer_decode_ms=packet
                decoded=time.perf_counter();decode_ms=(decoded-read_start)*1000
                if epoch is None:
                    epoch=decoded-(t-args.start);protocol['initial_seek_decode_s']=decode_ms/1000
                    protocol['first_source_ready_after_s']=decoded-pipeline_started
                if args.paced:
                    remaining=epoch+(t-args.start)-time.perf_counter()
                    if remaining>0:time.sleep(remaining)
                tick=time.perf_counter();cost={}
                a=tick
                if args.prefilter_decode:clean=raw_process.filtered_events
                else:noise.process_events(events,clean_buffer);clean=clean_buffer.numpy()
                last=clean[int(np.searchsorted(clean['t'],round(t*1e6)-20000)):]
                image_current=False;render_during_localize_ms=0.
                if args.optimized_compute:
                    from native_render import occupancy_frame
                    gray=occupancy_frame(last) if structure.count_current is None else structure.count_current.occupancy(last)
                else:
                    BaseFrameGenerationAlgorithm.generate_frame(last,image);image_current=True
                    gray=np.uint8(image[:,:,0]!=52)*255
                cost['filter_render_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter();center_hint=None
                if args.rigid_motion and getattr(structure.model,'beta',None) is not None:
                    model=structure.model
                    if t-model.last_update<=.12 and np.linalg.eigvalsh(model.P[np.ix_([0,3],[0,3])]).max()<=36.:
                        center_hint=model.beta[[0,3]]
                tracked,tracking=track(previous,gray,body,rigid_motion=args.rigid_motion,center_hint=center_hint)
                due=len(records)%args.nn_stride==0 or tracked is None or last_nn is None or t-last_nn>.3
                schedule=None
                if args.structure_guided_nn:
                    from axis_amplitude_bridge import current_axis_qualified
                    from live_local_flow import box as qualified_box
                    prior_structure=records[-1]['structure'] if records else {}
                    mandatory=tracked is None or last_nn is None or t-last_nn>.3
                    covariance=np.asarray(tracking.get('displacement_covariance_px2',np.eye(2)*1e9))
                    eligible=bool(not mandatory and qualified_box(tracked) is not None and tracking.get('points',0)>=12 and
                        np.linalg.eigvalsh(covariance).max()<=36. and current_axis_qualified(prior_structure))
                    original_due=due;interval=min(.3,2*args.nn_stride/args.hz)
                    if eligible:
                        requested_at=pending_t if pending_t is not None else last_nn
                        due=bool(t-requested_at>=interval-.5/args.hz)
                    schedule=dict(eligible=eligible,previous_geometry_guidance_only=True,
                        current_KLT_support_checked=True,target_request_interval_s=interval if eligible else args.nn_stride/args.hz,
                        deferred_original_due=bool(original_due and not due),mandatory_refresh=mandatory,
                        last_request_t=pending_t,current_geometry_not_yet_measured_at_this_scheduling_step=True)
                fresh_nn=False;nn_answer_age=None;association_diag=None;recovery_diag=None
                use_async=bool(args.async_nn and not(args.current_nn_on_loss and tracked is None))
                discarded_reply=False
                if not use_async and pending is not None:
                    stale=pending.result();pending=None;discarded_reply=True
                    if args.verify_frames and pending_hash!=stale['frame_sha256']:raise RuntimeError('Recovery worker frame mismatch')
                    pending_gray=None;pending_reference=None
                hint=None
                if args.roi_recovery:
                    if tracked is not None and not tracked.get('low_confidence'):
                        hint=tracked['bbox']
                    elif roi_memory is not None and t-roi_memory[0]<=.3:
                        dt_roi=t-roi_memory[0]
                        delta=np.zeros(2) if coeff is None else np.asarray(coeff['velocity'])*dt_roi
                        hint=(roi_memory[1]+np.r_[delta,delta]).tolist()
                request_extra=dict(recover_roi=args.roi_recovery,roi_hint=hint)
                if use_async:
                    body=tracked if tracked is not None else dict(detected=False)
                    if pending is not None and pending.done():
                        answer=pending.result();pending=None;nn_answer_age=t-pending_t
                        if args.verify_frames and pending_hash!=answer['frame_sha256']:raise RuntimeError('Async worker frame mismatch')
                        boxes=answer.get('associated_boxes',[]) if args.body_association else answer['boxes']
                        association_diag=answer.get('association')
                        recovery_diag=answer.get('roi_recovery')
                        if boxes:
                            best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
                            measured=dict(best,detected=True,center=((b[:2]+b[2:])/2).tolist(),
                                measurement_source='NN_at_earlier_time',low_confidence=best['confidence']<.45)
                            # Reuse the already measured current KLT trajectory
                            # rather than extracting/tracking the same frame twice.
                            if tracked is not None and pending_reference is not None and nn_answer_age<=.12:
                                delta=np.asarray(tracked['center'])-pending_reference
                                projected=dict(measured,bbox=(b+np.r_[delta,delta]).tolist(),
                                    center=(np.asarray(measured['center'])+delta).tolist())
                                transfer=dict(reason='reused_current_KLT_translation_since_NN_request',delta=delta.tolist())
                            else:projected,transfer=track(pending_gray,gray,measured)
                            if projected is not None and nn_answer_age<=.16:
                                body=dict(projected,measurement_source='current_KLT_from_delayed_NN',nn_observed_t=pending_t)
                                last_nn=pending_t;fresh_nn=True;tracking['delayed_NN_transfer']=transfer
                        pending_gray=None
                    if pending is None and due:
                        if not image_current:
                            render_start=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(last,image)
                            render_during_localize_ms+=(time.perf_counter()-render_start)*1000;image_current=True
                        np.copyto(shared_image,image)
                        pending_hash=hashlib.sha256(image.tobytes()).hexdigest() if args.verify_frames else None
                        gpu.stdin.write(js(dict(id=counter,t=t,verify_frame=args.verify_frames,**request_extra))+'\n');gpu.stdin.flush()
                        counter+=1;nn_count+=1;pending_t=t;pending_gray=gray.copy()
                        pending_reference=np.asarray(body['center']).copy() if body.get('detected') else None
                        pending=pool.submit(receive,gpu)
                elif due:
                    if not image_current:
                        render_start=time.perf_counter();BaseFrameGenerationAlgorithm.generate_frame(last,image)
                        render_during_localize_ms+=(time.perf_counter()-render_start)*1000;image_current=True
                    np.copyto(shared_image,image)
                    gpu.stdin.write(js(dict(id=counter,t=t,verify_frame=args.verify_frames,**request_extra))+'\n');gpu.stdin.flush()
                    answer=receive(gpu);counter+=1;nn_count+=1;fresh_nn=True
                    if args.verify_frames and hashlib.sha256(image.tobytes()).hexdigest()!=answer['frame_sha256']:
                        raise RuntimeError('Worker frame mismatch')
                    boxes=answer.get('associated_boxes',[]) if args.body_association else answer['boxes']
                    association_diag=answer.get('association')
                    recovery_diag=answer.get('roi_recovery')
                    if boxes:
                        best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
                        body=dict(best,detected=True,center=((b[:2]+b[2:])/2).tolist(),
                            measurement_source='current_NN_zoom_recovery' if best.get('roi_recovered') else 'current_NN',low_confidence=best['confidence']<.45)
                        last_nn=t
                    elif tracked is not None and last_nn is not None and t-last_nn<=.16:body=tracked
                    else:body=dict(detected=False)
                else:body=tracked
                cost['localize_ms']=(time.perf_counter()-a)*1000-render_during_localize_ms
                cost['filter_render_ms']+=render_during_localize_ms
                if body.get('detected') and not body.get('low_confidence'):
                    roi_memory=(t,np.asarray(body['bbox']).copy())
                previous=gray
                a=time.perf_counter()
                # Frequency uses ALL original timestamps in the selected expanded ROI,
                # including the13ms not rendered in the20ms shape window.
                selected=period_roi.select(events,roi_bounds(body['bbox'],padding=.35)) if body.get('detected') else events[:0]
                select_ms=(time.perf_counter()-a)*1000
                velocity=[0.,0.] if coeff is None else coeff['velocity']
                overlap_period=args.parallel_period and (not args.adaptive_period or not structure.last_side_qualified)
                if overlap_period:
                    frequency_future=period_pool.submit(measure_period,selected,t)
                    a=time.perf_counter();s,_,geom,density=structure.step(selected,t,body,None,
                        velocity,frequency_pixels_unique=True,defer_rates=True)
                    shape_ms=(time.perf_counter()-a)*1000
                    wait_started=time.perf_counter()
                    q,pooled_input,all_frequencies_in_range,engine_ms=frequency_future.result()
                    cost['period_wait_ms']=(time.perf_counter()-wait_started)*1000
                    a=time.perf_counter()
                    rates=rotor_rates(q,geom.get('rotors',[]),t,velocity,1280,unique_pixels=True) if geom.get('rotors') else []
                    cost['structure_ms']=shape_ms+(time.perf_counter()-a)*1000
                    cost['period_engine_ms']=select_ms+engine_ms
                else:
                    q,pooled_input,all_frequencies_in_range,engine_ms=measure_period(selected,t)
                    cost['period_engine_ms']=select_ms+engine_ms;cost['period_wait_ms']=0.
                    a=time.perf_counter();s,rates,geom,density=structure.step(selected,t,body,q,
                        velocity,frequency_pixels_unique=True)
                    cost['structure_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter()
                # Fixed geometry/body gauge from existing bridge, without another filter.
                adjusted,center_diag=centering.step(body,s,geom.get('rotors',[]))
                motion.kf.noise_scale=center_diag.get('noise_scale',1.)
                coeff,valid,motion_diag=motion.step(t,adjusted)
                image_motion_diag=None
                if image_motion is not None:
                    coeff,image_motion_diag=image_motion.step(t,coeff,motion.kf.P[2:4,2:4] if coeff else np.eye(2),adjusted,tracking)
                base,cv=scale_predictions(coeff) if coeff is not None and args.with_kinematic_control else ({},{})
                pooled=native79.summarize(pooled_input,body.get('bbox'),t,1280,720,unique_pixels=all_frequencies_in_range)
                u,quality,period_diag=period_selector.step(t,s,geom.get('rotors',[]),rates,pooled)
                amplitude_diag=None
                if amplitude_bridge is not None:
                    u,quality,amplitude_diag=amplitude_bridge.step(t,s,u,quality,period_diag)
                    force.input_learning_current=amplitude_diag['learning_input_current']
                f=period_diag['frequency_hz'];proxy_source=period_diag['source'];period_reason=period_diag['reason']
                force_reference=body if args.delayed_disturbance_control else adjusted
                position=force_reference.get('center') if force_reference.get('detected') else None
                if args.flow_force_balance:
                    force.observe_flow(image_motion_diag['observation'],image_motion_diag['acceleration'].get('applied',False))
                if args.tangent_disturbance:force.observe_axis_motion(s,image_motion_diag)
                physical,force_diag=force.step(t,coeff,position,u,quality)
                if image_motion_diag is not None:
                    inferred_acceleration=bool(image_motion_diag['acceleration'].get('applied'))
                    force_diag['event_motion_net_acceleration_available']=inferred_acceleration
                    if inferred_acceleration and not force_diag['active'] and not force_diag['short_gap_acceleration_held']:
                        force_diag['acceleration_source']='qualified_event_flow_total_acceleration'
                cost['motion_force_ms']=(time.perf_counter()-a)*1000
                if period_process is not None:cost['period_worker_engine_ms']=period_process.last_worker_engine_ms
                if raw_process is not None:
                    cost.update(producer_native_decode_ms=raw_process.last_native_decode_ms,producer_activity_filter_ms=raw_process.last_activity_filter_ms)
                done=time.perf_counter();compute_ms=(done-tick)*1000
                row=dict(t=t,body=body,structure=s,geometry=geom,rates=rates,period=pooled,
                    period_reason=period_reason,period_proxy_source=proxy_source,
                    squared_frequency_proxy=None if f is None else f*f,
                    relative_frequency_squared_proxy=period_diag['relative_f2'],period_selection=period_diag,
                    coefficient=coeff,valid=valid,baseline=base,cv=cv,physical=physical,
                    force=force_diag,center_bridge=center_diag,tracking=tracking,motion=motion_diag,
                    image_motion=image_motion_diag,
                    axis_amplitude_bridge=amplitude_diag,
                    period_execution='overlapped_current_process' if overlap_period and period_process is not None else 'overlapped_current_thread' if overlap_period else 'current_serial',
                    nn_structure_schedule=schedule,
                    force_observed_position=position,
                    nn_fresh=fresh_nn,nn_answer_age_s=nn_answer_age,nn_pending=pending is not None,
                    association=association_diag,
                    nn_execution_mode='asynchronous' if use_async else 'current_synchronous',
                    stale_NN_reply_discarded_for_current_recovery=discarded_reply,nn_roi_recovery=recovery_diag,
                    raw_events=len(events),frequency_input_events=len(selected),
                    decoder_packet_overflow=False if raw_process is None else raw_process.packet_overflow,
                    geometry_input_events=len(selected),
                    max_event_t=None if not len(events) else int(events['t'].max()),
                    cost=dict(cost,decode_ms=decode_ms,producer_decode_ms=producer_decode_ms,compute_ms=compute_ms,
                        service_ms=compute_ms+(0. if first_decode else decode_ms),
                        completion_lag_ms=(done-epoch-(t-args.start))*1000),first_decode=first_decode)
                first_decode=False
                if len(records)==0:protocol['first_result_after_s']=done-pipeline_started
                a=time.perf_counter();stream.write(js(row)+'\n')
                if video is not None:
                    if not image_current:BaseFrameGenerationAlgorithm.generate_frame(last,image)
                    display=image.copy()
                    if body.get('detected'):
                        b=np.rint(body['bbox']).astype(int);cv2.rectangle(display,tuple(b[:2]),tuple(b[2:]),(0,220,0),1)
                    for r in s['observed_centers']:cv2.circle(display,tuple(np.rint(r).astype(int)),3,(0,0,255),-1)
                    for pred,color in [(base,(255,140,0)),(physical,(0,220,255))]:
                        if valid and pred:cv2.polylines(display,[np.rint([coeff['position']]+[pred[str(H)] for H in (.1,.2,.4)]).astype(np.int32)],False,color,2)
                    axis=s.get('axis_xy')
                    if body.get('detected') and axis is not None:
                        p=np.asarray(body['center']);cv2.arrowedLine(display,tuple(np.rint(p).astype(int)),tuple(np.rint(p+60*np.asarray(axis)).astype(int)),(0,0,255),2)
                    cv2.putText(display,f'{t:.3f}s {s["mode"]} NN={due} force={force_diag["active"]} {compute_ms:.1f}ms',(15,28),0,.65,(0,180,255),1)
                    video.write(display)
                row['cost']['output_ms']=(time.perf_counter()-a)*1000
                row['cost']['total_service_ms']=row['cost']['service_ms']+row['cost']['output_ms']
                records.append(row)
                last_state_completed_wall=time.perf_counter()
        end=time.perf_counter()
        if gc_callback is not None:gc.callbacks.remove(gc_callback);gc_callback=None
        if args.defer_cyclic_gc and gc_was_enabled:gc.enable()
        # Output-write cost is known only after its prediction line is written.
        # Export the completed per-tick costs separately, after the timed run;
        # do not pretend those late fields exist in the original JSONL line.
        (out/'timings.jsonl').write_text('\n'.join(js(dict(t=r['t'],**r['cost'])) for r in records)+'\n','utf-8')
        samples=records[1:];cost_keys=['decode_ms','producer_decode_ms','compute_ms','service_ms','total_service_ms',
            'filter_render_ms','localize_ms','period_engine_ms','period_wait_ms','structure_ms','motion_force_ms','output_ms','completion_lag_ms']
        timing={key:dict(median=float(np.median([r['cost'][key] for r in samples])),
            p95=float(np.percentile([r['cost'][key] for r in samples],95)),mean=float(np.mean([r['cost'][key] for r in samples]))) for key in cost_keys} if samples else {}
        observed=sum(r['body'].get('detected',False) for r in records)
        service=np.array([r['cost']['total_service_ms'] for r in samples])
        summary=dict(frames=len(records),fresh_image_observations=observed,nn_frames=nn_count,
            valid_forecasts=sum(r['valid'] for r in records),modes=dict(Counter(r['structure']['mode'] for r in records)),
            current_force_inputs=sum(r['force']['active'] and not (r.get('axis_amplitude_bridge') or {}).get('applied',False) for r in records),
            recent_amplitude_current_axis_inputs=sum((r.get('axis_amplitude_bridge') or {}).get('applied',False) for r in records),
            gain_learning_updates=sum(r['force']['gain_learned'] for r in records),
            final_gain=force.gain,timing_ms=timing,
            steady_capacity_fps=None if not len(service) else float(1000/service.mean()),
            service_within_target_fraction=None if not len(service) else float(np.mean(service<=1000/args.hz)),
            paced=args.paced,paced_fresh_observation_rate=None if not args.paced else observed/max(last_state_completed_wall-epoch,1e-9),
            paced_observation_rate_including_eof=None if not args.paced else observed/max(end-epoch,1e-9),
            eof_wait_s=0. if last_state_completed_wall is None else max(0.,end-last_state_completed_wall),
            paced_deadline_fraction=None if not args.paced else float(np.mean([r['cost']['completion_lag_ms']<=1000/args.hz for r in samples])),
            initial_seek_decode_s=protocol['initial_seek_decode_s'],raw_unchanged=raw.stat().st_mtime_ns==stats.st_mtime_ns and raw.stat().st_size==stats.st_size,
            first_result_after_s=protocol['first_result_after_s'],
            decoder_overflow_packets=0 if raw_process is None else raw_process.overflow_packets,
            decoder_max_packet_events=None if raw_process is None else raw_process.max_packet_events,
            startup_note='initial_seek_decode_s is consumer blocking wait only; first_result_after_s includes overlapped model/decoder startup',
            target_achieved=False,accuracy='not scored yet; annotations never read by this runner')
        summary['period_worker_overflow_packets']=0 if period_process is None else period_process.overflow
        summary['structure_guided_NN_deferred_ticks']=sum((r.get('nn_structure_schedule') or {}).get('deferred_original_due',False) for r in records)
        summary['cyclic_GC']=dict(deferred=args.defer_cyclic_gc,events=gc_events,
            audited=args.audit_gc or args.defer_cyclic_gc,
            measured_stream_scope_only=True,reference_count_reclamation_remains_enabled=True,
            original_enabled_state_restored=gc.isenabled()==gc_was_enabled)
        protocol['actual_activity_filter']=getattr(noise,'provenance',None) if raw_process is None or not args.prefilter_decode else raw_process.filter_provenance
        (out/'summary.json').write_text(js(summary),'utf-8')
        protocol['finished']=True;(out/'protocol.json').write_text(js(protocol),'utf-8')
        print(js(summary),flush=True)
    finally:
        if gc_callback is not None:gc.callbacks.remove(gc_callback)
        if gc_was_enabled:gc.enable()
        elif not gc_was_enabled:gc.disable()
        if video is not None:video.release()
        if pending is not None:
            try:pending.result(timeout=15)
            except Exception:pass
        pool.shutdown(wait=True)
        period_pool.shutdown(wait=True)
        reader_pool.shutdown(wait=True)
        if period_process is not None:period_process.close()
        if raw_process is not None:raw_process.close()
        if gpu is not None:
            if gpu.poll() is None:
                try:gpu.stdin.write(js(dict(stop=True))+'\n');gpu.stdin.flush();gpu.wait(timeout=15)
                except Exception:gpu.terminate();gpu.wait(timeout=10)
            gpu.stdin.close();gpu.stdout.close()
        if log is not None:log.close()
        if memory is not None:image=None;shared_image=None;memory.close();memory.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',type=float,default=110.);p.add_argument('--duration',type=float,default=3.)
    p.add_argument('--hz',type=float,default=30.);p.add_argument('--nn-stride',type=int,default=3)
    p.add_argument('--gpu-python');p.add_argument('--detector-weights');p.add_argument('--detector-sha256');p.add_argument('--cache-dir');p.add_argument('--paced',action='store_true')
    p.add_argument('--async-nn',action='store_true')
    p.add_argument('--optimized-compute',action='store_true',help='Equivalent geometry work reduction, lazy native color render and overlapped decoder/model startup')
    p.add_argument('--reference-compute',dest='optimized_compute',action='store_false',help='Retain the original geometry/render/startup compute path as a control')
    p.set_defaults(optimized_compute=True)
    p.add_argument('--prefilter-decode',action='store_true',help='Overlap unchanged SDK activity filter with geometry via two additional bounded shared buffers')
    p.add_argument('--sdk-avx2',action='store_true',help='Optional locally built AVX2 instantiation of unchanged installed SDK activity filter; SDK license/local binary required; implies JIT dependency resolver')
    p.add_argument('--parallel-period',action='store_true',help='Overlap independent current geometry and SDK period calculations; rates join before force prediction')
    p.add_argument('--process-period',action='store_true',help='Run the unchanged SDK period observer in a bounded separate CPU process; implies parallel period')
    p.add_argument('--adaptive-period',action='store_true',help='With parallel period, keep cheap qualified side-axis ticks serial and overlap more expensive conic discovery; computations join at same source tick')
    p.add_argument('--side-axis-budget',action='store_true',help='Current qualified side axis each tick; full conic rediscovery every3ticks, with rotor count unresolved between')
    p.add_argument('--responsive-force',action='store_true',help='Recent causal disturbance correction and Singer conditional mean net-acceleration rollout')
    p.add_argument('--goal-profile',action='store_true',help='Candidate50px/30FPS profile: asyncNN + prefilter decode + side axis budget + responsive force; not a performance guarantee')
    p.add_argument('--target-candidate',action='store_true',help='20261028 candidate: goal profile, optional native kernels, shared motion/reanchor correction, same-window force balance, axis/amplitude bridge and adaptive period scheduling; finite-stream GC deferred')
    p.add_argument('--body-association',action='store_true',help='Reuse installed ByteTrack identity association; return only current NMS boxes, including matched low-score boxes')
    p.add_argument('--native-histogram',action='store_true',help='Exact-count C++ histogram directly in float32, with large-count fallback; no event sampling or saturation')
    p.add_argument('--jit-counts',action='store_true',help='Optional fused native CPU ROI/count kernel; requires compatible NumPy/Numba and times startup compilation')
    p.add_argument('--current-nn-on-loss',action='store_true',help='Use current synchronous NN when current KLT support fails; retain async when current image tracking succeeds')
    p.add_argument('--roi-recovery',action='store_true',help='Reuse current-frame zoom detector after full-frame miss, guided only by a recent current box/past velocity; implies current NN on loss')
    p.add_argument('--jit-conics',action='store_true',help='Fuse the same finite-difference conic Jacobian; implies optional JIT counts')
    p.add_argument('--reuse-image-motion',action='store_true',help='Reuse existing current KLT displacement for shared-state velocity/net-acceleration correction; no extra image flow pass or position KF')
    p.add_argument('--flow-force-balance',action='store_true',help='Estimate lumped disturbance from same-window velocity minus known input integral; implies shared image motion and responsive force')
    p.add_argument('--axis-amplitude-bridge',action='store_true',help='Current observed axis may use <=0.12s recent period amplitude; held amplitude never teaches gain')
    p.add_argument('--legacy-motion-anchor-gate',action='store_true',help='Control: reject motion from a weak previous box even when a current strong NN reanchor overlaps the measured tracks')
    p.add_argument('--rigid-motion',action='store_true',help='Experimental: fit existing current LK pairs to a similarity at the body/five-circle center; boxes stay original; implies reuse image motion')
    p.add_argument('--tangent-disturbance',action='store_true',help='Experimental: qualified current motion updates only the unknown-force component perpendicular to current axis; implies flow-force balance')
    p.add_argument('--structure-guided-nn',action='store_true',help='When past geometry and current strong KLT agree, double the NN interval up to0.3s; current image/geometry/period updates keep full source rate')
    p.add_argument('--with-kinematic-control',action='store_true',help='Research control only; default physical-only runs no parallel traditional forecast')
    period_options=p.add_mutually_exclusive_group()
    period_options.add_argument('--visibility-period',action='store_true',help='Per-model-slot f^2 normalization; keeps original unmodelled side control')
    period_options.add_argument('--pooled-axis-period',action='store_true',help='Experimental distinct pooled-side gauge; first side mean-error pilot regressed')
    p.add_argument('--delayed-disturbance-control',action='store_true',help='Control: hold lumped disturbance until the historical8-row gain-learning gate')
    p.add_argument('--no-force-gap-hold',action='store_true',help='Control: immediately drop force acceleration on every missing period/axis observation')
    p.add_argument('--legacy-gain-quality',action='store_true',help='Control: historical gain-uncertainty gate without the additional2sigma-proxy requirement')
    p.add_argument('--strict-gain-quality',dest='legacy_gain_quality',action='store_false',help='Experimental2sigma-proxy gain gate; current pilot predictions unchanged')
    p.set_defaults(legacy_gain_quality=True)
    p.add_argument('--serial-decode',action='store_true',help='Disable bounded one-packet native decoder prefetch')
    p.add_argument('--raw-read-chunk',type=int,default=100000,help='SDK RAW decode read chunk; source time windows and every native event remain unchanged')
    p.add_argument('--audit-gc',action='store_true',help='Record cyclic-GC pauses in the timed main-process stream')
    p.add_argument('--defer-cyclic-gc',action='store_true',help='Collect before this finite replay and defer cyclic-GC until cleanup; Python reference-count reclamation remains active')
    p.add_argument('--thread-decode',action='store_true',help='Use thread rather than separate decoder process')
    p.add_argument('--reuse-conic-uncertainty',action='store_true',help='Experimental conservative covariance reuse; no demonstrated speed benefit yet')
    p.add_argument('--legacy-structure',action='store_true',help='Frozen original four/three/axis model instead of continuous shared-state extension')
    p.add_argument('--no-shared-arc-recovery',action='store_true',help='Disable shared-template guidance of current genuine contour arcs')
    p.add_argument('--guide-full-only',action='store_true',help='Retain original guidance expiry since last full quad; control arm')
    p.add_argument('--strict-guide-area',action='store_true',help='Use strong count mask also for area evidence; original control')
    p.add_argument('--bipolar-guide-area',action='store_true',help='Experimental124 occupancy area; negative short pilot, disabled by default')
    p.add_argument('--no-current-shape-feedback',action='store_true',help='Control: retain transported conic shapes after partial updates')
    p.add_argument('--partial-arc-information',action='store_true',help='Experimental: current short arcs constrain only observable template directions; no extra detected rotors')
    p.add_argument('--no-partial-arc-information',dest='partial_arc_information',action='store_false',help='Control: disable partial normal-distance constraints')
    p.set_defaults(partial_arc_information=True)
    p.add_argument('--preview',action='store_true');p.add_argument('--verify-frames',action='store_true')
    args=p.parse_args()
    if args.target_candidate:
        args.goal_profile=True;args.jit_conics=True;args.flow_force_balance=True
        args.axis_amplitude_bridge=True;args.defer_cyclic_gc=True
        args.parallel_period=True;args.adaptive_period=not args.process_period
    if args.goal_profile:
        args.async_nn=True;args.prefilter_decode=True;args.side_axis_budget=True;args.responsive_force=True
    if args.roi_recovery:args.current_nn_on_loss=True
    if args.jit_conics:args.jit_counts=True
    if args.sdk_avx2:args.jit_counts=True
    if args.rigid_motion:args.reuse_image_motion=True
    if args.tangent_disturbance:args.flow_force_balance=True
    if args.flow_force_balance:args.reuse_image_motion=True;args.responsive_force=True
    if args.process_period:args.parallel_period=True
    if args.nn_stride<1:p.error('nn-stride must be positive')
    if args.raw_read_chunk<1:p.error('raw-read-chunk must be positive')
    if args.jit_counts and args.native_histogram:p.error('Choose only one optional count backend')
    if args.jit_conics and not args.optimized_compute:p.error('--jit-conics requires optimized compute')
    if args.roi_recovery and args.body_association:p.error('ROI recovery currently uses its own strict current-box matching; do not combine with body association')
    if args.structure_guided_nn and args.body_association:p.error('Structure-guided variable NN cadence is not validated with the fixed-rate body-association adapter')
    if args.prefilter_decode and (args.thread_decode or args.serial_decode):p.error('--prefilter-decode requires separate decoder process')
    run(args)


if __name__=='__main__':main()
