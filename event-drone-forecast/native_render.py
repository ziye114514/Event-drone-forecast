"""Exact occupancy used by KLT; render native BGR only for NN/preview."""
import numpy as np


def occupancy_frame(events,width=1280,height=720):
    result=np.zeros((height,width),np.uint8)
    result[events['y'],events['x']]=255
    return result
