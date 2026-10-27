"""Minimal mechanism3 RAW pilot:30Hz observations,10Hz NN, current force response.

Thirty means fresh source updates, not repeated forecasts or video playback.
The offline capacity and paced deadline results are reported separately.
"""
from pathlib import Path
import argparse
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
from fast_structure_front import FastStructure,roi_bounds,NativeROI
from fast_force_response import FastForceResponse
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


def track(previous,current,body):
    """Current-image translation only; no inertial extrapolation called detection."""
    if previous is None or body is None or not body.get('detected'):return None,dict(reason='no_previous_ROI')
    box=np.asarray(body['bbox']);x,y,X,Y=roi_bounds(box,padding=.35)
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
    result=dict(body,bbox=(box+np.r_[delta,delta]).tolist(),center=(np.asarray(body['center'])+delta).tolist(),
        measurement_source='current_KLT_translation',detected=True)
    return result,dict(reason='current_image_translation',points=int(keep.sum()),delta=delta.tolist())


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
    for f in ['arc_parameters.py', 'body_association.py', 'body_axis.py', 'continuous_five_circle.py', 'curvature_split.py', 'fast_force_response.py', 'fast_raw_reader.py', 'fast_structure_front.py', 'five_circle_model.py', 'forecast.py', 'forecast_runtime.py', 'frequency_features.py', 'live_center_ensemble.py', 'live_covariance.py', 'live_decode.py', 'live_frequency_pixels.py', 'live_measurement_kf.py', 'live_motion.py', 'live_pixel_axis.py', 'live_roi_refine.py', 'live_scale_rollout.py', 'live_structure_bridge.py', 'live_worker.py', 'lumped_disturbance.py', 'native_axis.py', 'partial_arc_information.py', 'rotor_geometry.py', 'run_fast_structure.py', 'shared_arc_recovery.py', 'shared_phase_fft.py', 'src/event_drone_forecast/__init__.py', 'src/event_drone_forecast/_vendor/__init__.py', 'src/event_drone_forecast/_vendor/filterpy_linear_upstream/helpers.py', 'src/event_drone_forecast/_vendor/filterpy_linear_upstream/kalman_filter.py', 'src/event_drone_forecast/_vendor/filterpy_process_upstream/discretization.py', 'src/event_drone_forecast/detector_backend.py', 'src/event_drone_forecast/geometry.py', 'src/event_drone_forecast/position/__init__.py', 'src/event_drone_forecast/position/_filterpy.py', 'src/event_drone_forecast/position/covariance.py', 'src/event_drone_forecast/position/kalman.py', 'src/event_drone_forecast/position/motion.py', 'src/event_drone_forecast/position/robust.py', 'src/event_drone_forecast/position/rollout.py', 'src/event_drone_forecast/raw_io.py']:
        content=(ROOT/f).read_bytes();p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
        sources[f]=hashlib.sha256(content).hexdigest()
    stats=raw.stat();noise=ActivityNoiseFilterAlgorithm(1280,720,10000);clean_buffer=noise.get_empty_output_buffer()
    from live_pixel_axis import SDKFixedPixelAxis
    periodic=SDKFixedPixelAxis(1280,720);period_roi=NativeROI()
    structure=FastStructure(args.reuse_conic_uncertainty,not args.legacy_structure,not args.no_shared_arc_recovery,args.guide_full_only,args.bipolar_guide_area and not args.strict_guide_area,not args.strict_guide_area,not args.no_current_shape_feedback,args.partial_arc_information);motion=GeometryMotion(args.hz) if args.with_kinematic_control else PhysicalPositionState()
    centering=QualifiedCenter();force=FastForceResponse(early_disturbance=not args.delayed_disturbance_control,
        hold_acceleration_gaps=not args.no_force_gap_hold and not args.delayed_disturbance_control,
        gain_signal_to_noise=0. if args.legacy_gain_quality or args.delayed_disturbance_control else 2.)
    memory=gpu=log=video=None;records=[];counter=0;nn_count=0;previous=None;body=None;last_nn=None
    pool=ThreadPoolExecutor(max_workers=1);reader_pool=ThreadPoolExecutor(max_workers=1)
    raw_process=None;read_future=None;pending=None;pending_gray=None;pending_t=None;pending_hash=None;pending_reference=None
    reference_frequency=None;previous_frequency=None;coeff=None;first_decode=True
    protocol=dict(entry='run_fast_structure.py',strategy=3,target_fps=args.hz,nn_stride=args.nn_stride,
        native_step_us=round(1e6/args.hz),geometry_window_us=20000,paced=args.paced,async_nn=args.async_nn,
        physical_only=not args.with_kinematic_control,periodic_sampling='Existing SDKFixedPixelAxis seed141000 fixed25% pixels; ALL native times at each retained pixel',
        decode_prefetch=not args.serial_decode,decode_queue_capacity=1,drop_source_frames=False,
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
    try:
        memory=shared_memory.SharedMemory(create=True,size=720*1280*3)
        shared_image=np.ndarray((720,1280,3),np.uint8,buffer=memory.buf);shared_image[:]=52
        image=np.empty((720,1280,3),np.uint8)
        log=(out/'gpu_worker.log').open('x',encoding='utf-8')
        gpu=subprocess.Popen([runtime['gpu_python'],str(ROOT/'live_worker.py'),'--shared-memory',memory.name,
            '--weights',runtime['detector_weights'],'--expected-sha256',runtime['detector_sha256'],
            '--cache-dir',runtime['cache_dir']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
            text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        protocol['gpu_ready']=receive(gpu)
        for _ in range(3):
            gpu.stdin.write(js(dict(id=counter,warmup=True))+'\n');gpu.stdin.flush();receive(gpu);counter+=1
        (out/'protocol.json').write_text(js(protocol),'utf-8')
        if args.preview:
            video=cv2.VideoWriter(str(out/'predictions.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.hz,(1280,720))
            if not video.isOpened():raise RuntimeError('Cannot open video writer')
        step_us=round(1e6/args.hz)
        # SDK requires start_ts to be a multiple of delta_t. Record the actual
        # alignment (at30Hz it differs from110s by only1.1ms), never retime events.
        aligned_start=(round(args.start*1e6)//step_us)*step_us/1e6
        aligned_duration=(round(args.duration*1e6)//step_us)*step_us/1e6
        protocol.update(actual_iterator_start=aligned_start,actual_duration=aligned_duration)
        iterator=iter(native_batches(raw,aligned_start,aligned_duration,False,step_us))
        def read_owned():
            before=time.perf_counter();packet=next(iterator,None)
            if packet is None:return None
            stamp,e=packet
            # Reader may reuse its ring. One bulk byte copy gives the consumer
            # immutable owned storage; avoid per-field structured-array copying.
            if not e.flags.c_contiguous:raise RuntimeError('Expected contiguous native event batch')
            owned=e.view(np.uint8).copy().view(e.dtype);owned.setflags(write=False)
            return stamp,owned,(time.perf_counter()-before)*1000
        if not args.thread_decode and not args.serial_decode:
            from fast_raw_reader import ProcessRAW
            raw_process=ProcessRAW(raw,aligned_start,aligned_duration,step_us)
        elif not args.serial_decode:read_future=reader_pool.submit(read_owned)
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
                if args.paced:
                    remaining=epoch+(t-args.start)-time.perf_counter()
                    if remaining>0:time.sleep(remaining)
                tick=time.perf_counter();cost={}
                a=tick;noise.process_events(events,clean_buffer);clean=clean_buffer.numpy()
                last=clean[int(np.searchsorted(clean['t'],round(t*1e6)-20000)):]
                BaseFrameGenerationAlgorithm.generate_frame(last,image)
                gray=np.uint8(image[:,:,0]!=52)*255;cost['filter_render_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter();tracked,tracking=track(previous,gray,body)
                due=len(records)%args.nn_stride==0 or tracked is None or last_nn is None or t-last_nn>.3
                fresh_nn=False;nn_answer_age=None
                if args.async_nn:
                    body=tracked if tracked is not None else dict(detected=False)
                    if pending is not None and pending.done():
                        answer=pending.result();pending=None;nn_answer_age=t-pending_t
                        if args.verify_frames and pending_hash!=answer['frame_sha256']:raise RuntimeError('Async worker frame mismatch')
                        if answer['boxes']:
                            best=max(answer['boxes'],key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
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
                        np.copyto(shared_image,image)
                        pending_hash=hashlib.sha256(image.tobytes()).hexdigest() if args.verify_frames else None
                        gpu.stdin.write(js(dict(id=counter,t=t,verify_frame=args.verify_frames))+'\n');gpu.stdin.flush()
                        counter+=1;nn_count+=1;pending_t=t;pending_gray=gray.copy()
                        pending_reference=np.asarray(body['center']).copy() if body.get('detected') else None
                        pending=pool.submit(receive,gpu)
                elif due:
                    np.copyto(shared_image,image)
                    gpu.stdin.write(js(dict(id=counter,t=t,verify_frame=args.verify_frames))+'\n');gpu.stdin.flush()
                    answer=receive(gpu);counter+=1;nn_count+=1;fresh_nn=True
                    if args.verify_frames and hashlib.sha256(image.tobytes()).hexdigest()!=answer['frame_sha256']:
                        raise RuntimeError('Worker frame mismatch')
                    boxes=answer['boxes']
                    if boxes:
                        best=max(boxes,key=lambda b:b['confidence']);b=np.asarray(best['bbox'])
                        body=dict(best,detected=True,center=((b[:2]+b[2:])/2).tolist(),
                            measurement_source='current_NN',low_confidence=best['confidence']<.45)
                        last_nn=t
                    elif tracked is not None and last_nn is not None and t-last_nn<=.16:body=tracked
                    else:body=dict(detected=False)
                else:body=tracked
                cost['localize_ms']=(time.perf_counter()-a)*1000
                previous=gray
                a=time.perf_counter()
                # Frequency uses ALL original timestamps in the selected expanded ROI,
                # including the13ms not rendered in the20ms shape window.
                selected=period_roi.select(events,roi_bounds(body['bbox'],padding=.35)) if body.get('detected') else events[:0]
                q=periodic.feed(selected,t)
                q=q[q['t']>=round(t*1e6)-20000]
                pooled_input=q
                all_frequencies_in_range=bool(np.all((q['frequency']>=60.)&(q['frequency']<=1200.)))
                q=latest_pixels(q,1280)
                if all_frequencies_in_range:pooled_input=q
                cost['period_engine_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter();s,rates,geom,density=structure.step(selected,t,body,q,
                    [0.,0.] if coeff is None else coeff['velocity'],frequency_pixels_unique=True)
                cost['structure_ms']=(time.perf_counter()-a)*1000
                a=time.perf_counter()
                # Fixed geometry/body gauge from existing bridge, without another filter.
                adjusted,center_diag=centering.step(body,s,geom.get('rotors',[]))
                motion.kf.noise_scale=center_diag.get('noise_scale',1.)
                coeff,valid,motion_diag=motion.step(t,adjusted)
                base,cv=scale_predictions(coeff) if coeff is not None and args.with_kinematic_control else ({},{})
                pooled=native79.summarize(pooled_input,body.get('bbox'),t,1280,720,unique_pixels=all_frequencies_in_range)
                # Prefer per-region current estimates. Side views pool unique pixels;
                # no temporary blob ID is interpreted as one physical motor.
                fs=[r['frequency_hz'] for r in rates if r.get('valid')]
                f=float(np.sqrt(np.mean(np.square(fs)))) if len(fs)>=2 else pooled.get('frequency_hz') if pooled.get('qualified') else None
                proxy_source=('four_current_squared_frequencies' if len(fs)==4 and s['mode']=='four' else
                    'visible_frequency_squared_mean_hidden_equal_mean_hypothesis' if len(fs)>=2 else
                    'pooled_flicker_representative_speed_hypothesis')
                u=None;quality=0.;period_reason='no_current_qualified_frequency'
                if f is not None:
                    coherent=previous_frequency is not None and t-previous_frequency[0]<.12 and abs(np.log(f/previous_frequency[1]))<.08
                    previous_frequency=(t,f)
                    if coherent and s.get('axis_xy') is not None:
                        if reference_frequency is None:reference_frequency=f
                        n=np.asarray(s['axis_xy'],float)
                        if n[1]>0:n=-n
                        relative=(f/reference_frequency)**2
                        if .2<=relative<=5.:
                            u=relative*n;quality=max(.35,float(s.get('geometric_weight',0.)))
                            period_reason='current_coherent_relative_frequency_and_pose'
                force_reference=body if args.delayed_disturbance_control else adjusted
                position=force_reference.get('center') if force_reference.get('detected') else None
                physical,force_diag=force.step(t,coeff,position,u,quality)
                cost['motion_force_ms']=(time.perf_counter()-a)*1000
                done=time.perf_counter();compute_ms=(done-tick)*1000
                row=dict(t=t,body=body,structure=s,geometry=geom,rates=rates,period=pooled,
                    period_reason=period_reason,period_proxy_source=proxy_source,
                    squared_frequency_proxy=None if f is None else f*f,
                    coefficient=coeff,valid=valid,baseline=base,cv=cv,physical=physical,
                    force=force_diag,center_bridge=center_diag,tracking=tracking,motion=motion_diag,
                    force_observed_position=position,
                    nn_fresh=fresh_nn,nn_answer_age_s=nn_answer_age,nn_pending=pending is not None,
                    raw_events=len(events),frequency_input_events=len(selected),
                    geometry_input_events=len(selected),
                    max_event_t=None if not len(events) else int(events['t'].max()),
                    cost=dict(cost,decode_ms=decode_ms,producer_decode_ms=producer_decode_ms,compute_ms=compute_ms,
                        service_ms=compute_ms+(0. if first_decode else decode_ms),
                        completion_lag_ms=(done-epoch-(t-args.start))*1000),first_decode=first_decode)
                first_decode=False
                a=time.perf_counter();stream.write(js(row)+'\n')
                if video is not None:
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
        samples=records[1:];cost_keys=['decode_ms','producer_decode_ms','compute_ms','service_ms','total_service_ms',
            'filter_render_ms','localize_ms','period_engine_ms','structure_ms','motion_force_ms','output_ms','completion_lag_ms']
        timing={key:dict(median=float(np.median([r['cost'][key] for r in samples])),
            p95=float(np.percentile([r['cost'][key] for r in samples],95)),mean=float(np.mean([r['cost'][key] for r in samples]))) for key in cost_keys} if samples else {}
        observed=sum(r['body'].get('detected',False) for r in records)
        service=np.array([r['cost']['total_service_ms'] for r in samples])
        summary=dict(frames=len(records),fresh_image_observations=observed,nn_frames=nn_count,
            valid_forecasts=sum(r['valid'] for r in records),modes=dict(Counter(r['structure']['mode'] for r in records)),
            current_force_inputs=sum(r['force']['active'] for r in records),gain_learning_updates=sum(r['force']['gain_learned'] for r in records),
            final_gain=force.gain,timing_ms=timing,
            steady_capacity_fps=None if not len(service) else float(1000/service.mean()),
            service_within_target_fraction=None if not len(service) else float(np.mean(service<=1000/args.hz)),
            paced=args.paced,paced_fresh_observation_rate=None if not args.paced else observed/max(last_state_completed_wall-epoch,1e-9),
            paced_observation_rate_including_eof=None if not args.paced else observed/max(end-epoch,1e-9),
            eof_wait_s=0. if last_state_completed_wall is None else max(0.,end-last_state_completed_wall),
            paced_deadline_fraction=None if not args.paced else float(np.mean([r['cost']['completion_lag_ms']<=1000/args.hz for r in samples])),
            initial_seek_decode_s=protocol['initial_seek_decode_s'],raw_unchanged=raw.stat().st_mtime_ns==stats.st_mtime_ns and raw.stat().st_size==stats.st_size,
            target_achieved=False,accuracy='not scored yet; annotations never read by this runner')
        (out/'summary.json').write_text(js(summary),'utf-8')
        protocol['finished']=True;(out/'protocol.json').write_text(js(protocol),'utf-8')
        print(js(summary),flush=True)
    finally:
        if video is not None:video.release()
        if pending is not None:
            try:pending.result(timeout=15)
            except Exception:pass
        pool.shutdown(wait=True)
        reader_pool.shutdown(wait=True)
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
    p.add_argument('--with-kinematic-control',action='store_true',help='Research control only; default physical-only runs no parallel traditional forecast')
    p.add_argument('--delayed-disturbance-control',action='store_true',help='Control: hold lumped disturbance until the historical8-row gain-learning gate')
    p.add_argument('--no-force-gap-hold',action='store_true',help='Control: immediately drop force acceleration on every missing period/axis observation')
    p.add_argument('--legacy-gain-quality',action='store_true',help='Control: historical gain-uncertainty gate without the additional2sigma-proxy requirement')
    p.add_argument('--strict-gain-quality',dest='legacy_gain_quality',action='store_false',help='Experimental2sigma-proxy gain gate; current pilot predictions unchanged')
    p.set_defaults(legacy_gain_quality=True)
    p.add_argument('--serial-decode',action='store_true',help='Disable bounded one-packet native decoder prefetch')
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
    if args.nn_stride<1:p.error('nn-stride must be positive')
    run(args)


if __name__=='__main__':main()
