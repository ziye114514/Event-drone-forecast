"""Bounded two-slot RAW decoder process; native bytes, no frame dropping."""
from multiprocessing import shared_memory
import multiprocessing as mp
import time
import traceback


def produce(raw,start,duration,step_us,names,capacity,ready,free):
    import numpy as np
    from live_decode import native_batches
    buffers=[shared_memory.SharedMemory(name=n,create=False) for n in names]
    try:
        iterator=iter(native_batches(raw,start,duration,False,step_us))
        while True:
            slot=free.get();begin=time.perf_counter();packet=next(iterator,None)
            if packet is None:ready.put(None);break
            t,e=packet
            if not e.flags.c_contiguous or e.nbytes>capacity:raise RuntimeError('Native packet exceeds fixed shared slot')
            target=np.ndarray((e.nbytes,),np.uint8,buffer=buffers[slot].buf)
            np.copyto(target,e.view(np.uint8));target=None
            ready.put(dict(t=t,n=len(e),slot=slot,decode_ms=(time.perf_counter()-begin)*1000))
    except BaseException:
        ready.put(dict(error=traceback.format_exc()))
    finally:
        for b in buffers:b.close()


class ProcessRAW:
    def __init__(self,raw,start,duration,step_us):
        import numpy as np
        from metavision_sdk_base import EventCD
        self.np=np;self.dtype=np.dtype(EventCD);self.capacity=2000000*self.dtype.itemsize
        self.buffers=[shared_memory.SharedMemory(create=True,size=self.capacity) for _ in range(2)]
        context=mp.get_context('spawn');self.ready=context.Queue(maxsize=2);self.free=context.Queue(maxsize=2)
        for i in range(2):self.free.put(i)
        self.previous=None;self.view=None
        self.process=context.Process(target=produce,args=(str(raw),start,duration,step_us,
            [b.name for b in self.buffers],self.capacity,self.ready,self.free),daemon=True)
        self.process.start()
    def read(self):
        self.view=None
        if self.previous is not None:self.free.put(self.previous);self.previous=None
        packet=self.ready.get(timeout=120)
        if packet is None:return None
        if 'error' in packet:raise RuntimeError(packet['error'])
        self.previous=packet['slot']
        self.view=self.np.ndarray((packet['n'],),self.dtype,buffer=self.buffers[self.previous].buf)
        self.view.setflags(write=False)
        return packet['t'],self.view,packet['decode_ms']
    def close(self):
        self.view=None
        if self.previous is not None:
            try:self.free.put_nowait(self.previous)
            except Exception:pass
            self.previous=None
        self.process.join(timeout=3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        for q in (self.ready,self.free):q.cancel_join_thread();q.close()
        for b in self.buffers:b.close();b.unlink()
