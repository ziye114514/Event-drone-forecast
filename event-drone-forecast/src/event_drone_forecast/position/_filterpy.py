"""Execute pinned unmodified MIT FilterPy author functions, independent of research paths."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
VENDOR=Path(__file__).resolve().parents[1]/'_vendor/filterpy_linear_upstream'
snapshot=json.loads((VENDOR/'UPSTREAM_SNAPSHOT.json').read_text('utf-8'))
for row in snapshot['files']:
    data=(VENDOR/row['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1']
# Execute the actual author function ASTs, without rewriting the arithmetic or
# importing unrelated stats/display dependencies from an installed package.
namespace=dict(np=np,dot=np.dot,zeros=np.zeros,eye=np.eye,isscalar=np.isscalar,shape=np.shape,linalg=np.linalg)
for file,names in [('helpers.py',['reshape_z']),('kalman_filter.py',['predict','update'])]:
    tree=ast.parse((VENDOR/file).read_text('utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(VENDOR/file),'exec'),namespace)
predict=namespace['predict'];update=namespace['update']

