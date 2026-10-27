"""Fixed pixel subset: preserve all temporal events at selected native pixels.

This reduces work only in the optional periodic observer. It never samples
events in time, rescales coordinates or changes the full detector input.
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from native_axis import NativeAxis


class FixedPixelAxis(NativeAxis):
    def __init__(self,w,h):
        super().__init__(w,h)
        self.mask=np.random.default_rng(141000).integers(0,4,size=(h,w),dtype=np.uint8)==0
        self.total_events=self.selected_events=0

    def feed(self,e,t):
        self.total_events+=len(e)
        selected=e[self.mask[e['y'],e['x']]]
        self.selected_events+=len(selected)
        return self._feed_selected(selected,t)

    def _feed_selected(self,selected,t):
        if len(selected):return NativeAxis.feed(self,selected,t)
        assert self.last_t is None or t>self.last_t
        self.last_t=t
        # An empty selected stream provides no new observation. Do not expose
        # previous output-buffer contents as a current periodic event.
        return self.output.numpy()[:0].copy()


class SDKFixedPixelAxis(FixedPixelAxis):
    def __init__(self,w,h):
        super().__init__(w,h)
        from metavision_sdk_core import RoiMaskAlgorithm
        self.float_mask=self.mask.astype(np.float64)
        self.pixel_filter=RoiMaskAlgorithm(self.float_mask)
        self.pixel_buffer=self.pixel_filter.get_empty_output_buffer()

    def feed(self,e,t):
        self.total_events+=len(e)
        self.pixel_filter.process_events(e,self.pixel_buffer)
        selected=self.pixel_buffer.numpy()
        self.selected_events+=len(selected)
        return self._feed_selected(selected,t)
