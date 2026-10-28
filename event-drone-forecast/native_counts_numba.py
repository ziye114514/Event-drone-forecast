"""Optional fused CPU ROI/count pass, exact same native20ms ON/OFF counts.

Numba is an external optional dependency. No event timestamp/polarity/pixel is
sampled or saturated, and its startup compilation is explicitly timed.
"""
import os,sys,time
from pathlib import Path
import numpy as np

extra=os.environ.get('EDF_NUMBA_PATH')
if extra and Path(extra).is_dir():sys.path.append(str(Path(extra).resolve()))
try:
    import numba
    from numba import njit
except ImportError as error:
    raise RuntimeError('Optional --jit-counts requires compatible numba/llvmlite; see requirements-counts.txt or EDF_NUMBA_PATH') from error


@njit(cache=True,nogil=True)
def histogram(events,x,y,X,Y):
    result=np.zeros((2,Y-y,X-x),np.float32)
    for i in range(len(events)):
        ex=int(events[i]['x']);ey=int(events[i]['y'])
        if x<=ex<X and y<=ey<Y:
            polarity=int(events[i]['p'])
            if polarity<0 or polarity>1:raise ValueError('Expected binary native polarity')
            result[polarity,ey-y,ex-x]+=1.
    return result


@njit(cache=True,nogil=True)
def occupancy(events,width,height):
    image=np.zeros((height,width),np.uint8)
    for i in range(len(events)):image[int(events[i]['y']),int(events[i]['x'])]=255
    return image


class NumbaCounter:
    def __init__(self):
        from metavision_sdk_base import EventCD
        begin=time.perf_counter();empty=np.empty(0,dtype=np.dtype(EventCD))
        # The decoder's borrowed packets are readonly, but an SDK ROI/filter
        # may return writable views. Warm both signatures before paced input.
        # Otherwise the first real source tick can compile a second kernel
        # and create hundreds of milliseconds of artificial initial backlog.
        histogram(empty,0,0,1,1);occupancy(empty,1,1)
        empty.setflags(write=False);histogram(empty,0,0,1,1);occupancy(empty,1,1)
        self.provenance=dict(backend='Numba fused native ROI and float32 count',numba_version=numba.__version__,
            numpy_version=np.__version__,kernel_warm_compile_load_ms=(time.perf_counter()-begin)*1000,
            no_sampling=True,all_native_20ms_events=True,no_fastmath=True,large_count_int64_fallback=True,
            writable_and_readonly_signatures_warmed_before_pacing=True,occupancy_is_identical_binary_event_support=True)

    @staticmethod
    def occupancy(events,width=1280,height=720):return occupancy(events,width,height)

    def __call__(self,events,bounds,t,selector=None):
        begin=int(np.searchsorted(events['t'],round(t*1e6)-20000));e=events[begin:]
        if len(e)>=2**24:
            # Repeated float32 increments cease to be exact at2**24. Preserve
            # original integer-then-cast semantics for such exceptional packets.
            from fast_structure_front import count_pair
            return count_pair(events,bounds,t,selector,cpp_histogram=False)
        x,y,X,Y=map(int,bounds);counts=histogram(e,x,y,X,Y)
        return [counts[1],counts[0]]
