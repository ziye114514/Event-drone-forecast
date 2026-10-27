"""One latest current frequency per pixel, shared by geometry and body pooling."""
import numpy as np


def latest_pixels(q,width):
    if not len(q):return q
    pixels=q['y'].astype(np.int64)*width+q['x']
    _,index=np.unique(pixels[::-1],return_index=True)
    return q[len(q)-1-index]
