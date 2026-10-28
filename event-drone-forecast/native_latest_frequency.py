"""Exact latest-per-pixel order without sorting repeated frequency records."""
import time
import numpy as np
from native_counts_numba import njit


@njit(cache=True,nogil=True)
def latest_indices(q,width,indices,touched):
    count=0
    for i in range(len(q)-1,-1,-1):
        pixel=int(q[i]['y'])*width+int(q[i]['x'])
        if indices[pixel]<0:
            indices[pixel]=i;touched[count]=pixel;count+=1
    ordered=np.sort(touched[:count]);result=np.empty(count,np.int64)
    for i in range(count):
        pixel=ordered[i];result[i]=indices[pixel];indices[pixel]=-1
    return result


class LatestFrequency:
    def __init__(self,width,height,dtype):
        self.width=width;self.indices=np.full(width*height,-1,np.int64);self.touched=np.empty(width*height,np.int64)
        begin=time.perf_counter();empty=np.empty(0,dtype=dtype)
        latest_indices(empty,width,self.indices,self.touched)
        empty.setflags(write=False);latest_indices(empty,width,self.indices,self.touched)
        self.provenance=dict(method='reverse native scan; sort distinct pixel IDs only; identical latest-record and pixel ordering',
            warm_ms=(time.perf_counter()-begin)*1000,bounded_index_storage_bytes=self.indices.nbytes+self.touched.nbytes)

    def __call__(self,q):
        if not len(q):return q
        return q[latest_indices(q,self.width,self.indices,self.touched)]
