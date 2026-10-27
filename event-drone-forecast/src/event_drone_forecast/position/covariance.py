"""Pinned FilterPy continuous-time process noise; no tuned new spectral power."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
from scipy.linalg import block_diag

from .kalman import SimpleKF,predict,update,coefficient,PARAMS
VENDOR=Path(__file__).resolve().parents[1]/'_vendor/filterpy_process_upstream'
snapshot=json.loads((VENDOR/'UPSTREAM_SNAPSHOT.json').read_text(encoding='utf-8'))
for row in snapshot['files']:
    assert hashlib.sha256((VENDOR/row['path']).read_bytes()).hexdigest()==row['sha256']
tree=ast.parse((VENDOR/'discretization.py').read_text(encoding='utf-8'))
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
           and node.name in ('Q_continuous_white_noise','order_by_derivative')]
assert len(functions)==2
namespace=dict(array=np.array,zeros=np.zeros,eye=np.eye,block_diag=block_diag)
exec(compile(ast.Module(body=functions,type_ignores=[]),str(VENDOR/'discretization.py'),'exec'),namespace)
author_Q=namespace['Q_continuous_white_noise']
REFERENCE_DT=.02
SPECTRAL_DENSITY=PARAMS['sigma_acceleration']**2*REFERENCE_DT


def process_noise(dt):
    if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite dt required')
    return author_Q(2,dt=dt,spectral_density=SPECTRAL_DENSITY,block_size=2,order_by_dim=False)


class ContinuousKF(SimpleKF):
    def step(self,t,z):
        if self.seen is not None and t-self.seen>PARAMS['reset_body_gap_s']:self.__init__()
        if self.x is None:
            if z is None:return None
            self.x=np.r_[z,[0.,0.]];self.P=np.eye(4)*PARAMS['initial_covariance'];self.t=t;self.seen=t
            return coefficient(self.x)
        dt=t-self.t
        if dt<=0:raise ValueError('Strictly increasing observation time required')
        self.t=t;F=np.eye(4);F[:2,2:4]=np.eye(2)*dt
        self.x,self.P=predict(self.x,self.P,F=F,Q=process_noise(dt))
        if z is not None:
            self.seen=t
            self.x,self.P=update(self.x,self.P,z,R=np.eye(2)*36.,H=np.c_[np.eye(2),np.zeros((2,2))])
        return coefficient(self.x)
