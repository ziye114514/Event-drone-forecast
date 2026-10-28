"""One bounded current-period task in a separate CPU process.

Same SDK fixed-pixel observer and exact latest-per-pixel reducer. No pipelined
future inputs, stale period reuse, temporal event sampling or source drops.
"""
from multiprocessing import shared_memory
import multiprocessing as mp
import time,traceback


def produce(name,output_name,capacity,requests,answers,jit):
    import numpy as np
    from metavision_sdk_base import EventCD
    from live_pixel_axis import SDKFixedPixelAxis
    from live_frequency_pixels import latest_pixels
    memory=shared_memory.SharedMemory(name=name,create=False);output_memory=shared_memory.SharedMemory(name=output_name,create=False);dtype=np.dtype(EventCD)
    try:
        observer=SDKFixedPixelAxis(1280,720);latest=None
        if jit:
            from native_latest_frequency import LatestFrequency
            latest=LatestFrequency(1280,720,observer.output.numpy().dtype)
        answers.put(dict(ready=True,latest_kernel=None if latest is None else latest.provenance))
        expected=0
        while True:
            request=requests.get()
            if request is None:break
            if request['id']!=expected:raise RuntimeError('Period task order changed')
            expected+=1;t=request['t'];start=time.perf_counter()
            e=np.frombuffer(request['overflow_bytes'],dtype=dtype) if 'overflow_bytes' in request else np.ndarray((request['n'],),dtype,buffer=memory.buf)
            e.setflags(write=False)
            q=observer.feed(e,t);q=q[q['t']>=round(t*1e6)-20000]
            allowed=bool(np.all((q['frequency']>=60.)&(q['frequency']<=1200.)))
            pooled=q
            q=latest_pixels(q,1280) if latest is None else latest(q)
            if allowed:pooled=q
            target=np.ndarray(q.shape,q.dtype,buffer=output_memory.buf);np.copyto(target,q);target=None
            answers.put(dict(id=request['id'],q_count=len(q),pooled=None if allowed else pooled.tobytes(),
                dtype=q.dtype,in_range=allowed,engine_ms=(time.perf_counter()-start)*1000,
                total_events=observer.total_events,selected_events=observer.selected_events))
            e=None
    except BaseException:answers.put(dict(error=traceback.format_exc()))
    finally:memory.close();output_memory.close()


class PeriodProcess:
    def __init__(self,jit=False):
        import numpy as np
        from metavision_sdk_base import EventCD
        self.np=np;self.dtype=np.dtype(EventCD);self.capacity=2000000*self.dtype.itemsize
        self.memory=shared_memory.SharedMemory(create=True,size=self.capacity)
        self.output_memory=shared_memory.SharedMemory(create=True,size=1280*720*24)
        context=mp.get_context('spawn');self.requests=context.Queue(maxsize=1);self.answers=context.Queue(maxsize=1)
        self.process=context.Process(target=produce,args=(self.memory.name,self.output_memory.name,self.capacity,self.requests,self.answers,jit),daemon=True)
        self.counter=0;self.provenance=None;self.overflow=0;self.total_events=0;self.selected_events=0
        self.last_worker_engine_ms=0.;self.process.start()

    def receive_ready(self):
        if self.provenance is None:
            answer=self.answers.get(timeout=120)
            if 'error' in answer:raise RuntimeError(answer['error'])
            if not answer.get('ready'):raise RuntimeError('Period process startup protocol')
            self.provenance=dict(answer,request_capacity=1,answer_capacity=1,native_capacity_bytes=self.capacity,
                overflow_policy='bounded immutable bytes; never truncate',separate_CPU_process=True,
                latest_frequency_output_shared_memory_bytes=self.output_memory.size,
                transport='shared_native_input_and_latest_frequency_output; bounded_queue_headers_only')
        return self.provenance

    def compute(self,e,t):
        self.receive_ready();begin=time.perf_counter()
        request=dict(id=self.counter,t=t,n=len(e));self.counter+=1
        if e.nbytes>self.capacity:
            request['overflow_bytes']=e.tobytes();self.overflow+=1
        else:
            target=self.np.ndarray((len(e),),self.dtype,buffer=self.memory.buf)
            self.np.copyto(target,e);target=None
        self.requests.put(request);answer=self.answers.get(timeout=120)
        if 'error' in answer:raise RuntimeError(answer['error'])
        if answer['id']!=request['id']:raise RuntimeError('Period result order changed')
        q=self.np.ndarray((answer['q_count'],),answer['dtype'],buffer=self.output_memory.buf).copy()
        pooled=q if answer['pooled'] is None else self.np.frombuffer(answer['pooled'],dtype=answer['dtype'])
        self.last_worker_engine_ms=answer['engine_ms'];self.total_events=answer['total_events'];self.selected_events=answer['selected_events']
        return q,pooled,answer['in_range'],(time.perf_counter()-begin)*1000

    def close(self):
        if self.process.is_alive():
            try:self.requests.put(None,timeout=1)
            except Exception:pass
        self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        for q in [self.requests,self.answers]:q.cancel_join_thread();q.close()
        self.memory.close();self.memory.unlink()
        self.output_memory.close();self.output_memory.unlink()
