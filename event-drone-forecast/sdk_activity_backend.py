"""Optional private AVX2 instantiation of the user's installed SDK filter.

The adapter invokes unmodified vendor headers; no vendor source or binary is
shipped. A compatible local SDK license and local build are prerequisites.
"""
from pathlib import Path
import os,sys,hashlib
import numpy as np


class Output:
    def __init__(self,dtype):self.data=np.empty(0,dtype)
    def numpy(self):return self.data


class SDKActivityAVX2:
    def __init__(self,width,height,threshold):
        # The optional Numba resolver makes the already isolated llvmlite
        # available. Check host support before loading any AVX2 machine code.
        from native_counts_numba import numba
        from llvmlite.binding import get_host_cpu_features
        if not get_host_cpu_features().get('avx2'):raise RuntimeError('Optional SDK AVX2 adapter needs an AVX2 host')
        folder=Path(os.environ.get('EDF_SDK_ACTIVITY_PATH',Path(__file__).resolve().parent.parent/'.cache/sdk_activity_avx2')).resolve()
        sys.path.insert(0,str(folder))
        try:import _edf_sdk_activity as module
        except ImportError as error:raise RuntimeError('Build the local SDK adapter with tools/build_sdk_activity.py; no binary is included') from error
        if not module.compiled_AVX2:raise RuntimeError('Local SDK adapter compiler target is not AVX2')
        from metavision_sdk_base import EventCD
        self.dtype=np.dtype(EventCD);self.backend=module.ActivityAdapter(width,height,threshold)
        path=Path(module.__file__)
        self.provenance=dict(backend='installed_SDK_header_local_AVX2_instantiation',binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_SDK_threshold_us=threshold,SDK_algorithm_unmodified=True,host_AVX2=True,
            vendor_source_and_binary_not_redistributed=True,SDK_license_required=True)

    def get_empty_output_buffer(self):return Output(self.dtype)
    def process_events(self,events,output):
        if events.dtype!=self.dtype:raise ValueError('Expected native EventCD dtype')
        output.data=self.backend.process(events)
