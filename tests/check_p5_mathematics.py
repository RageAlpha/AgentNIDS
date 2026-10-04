import json
from pathlib import Path
import cupy as cp
import numpy as np
from nids.replay.mmd import FrozenMMD,squared_distance

x=cp.asarray([[1e12,2.],[1e12+1,3.],[1e12+4,8.]],dtype=cp.float64)
d=squared_distance(x,x)
np.testing.assert_allclose(cp.asnumpy(d),((cp.asnumpy(x)[:,None]-cp.asnumpy(x)[None,:])**2).sum(2),rtol=0,atol=0)
m=object.__new__(FrozenMMD);m.tile=2;m.sigma=2.3
expected=cp.exp(-d/(2*m.sigma**2)).sum()
np.testing.assert_allclose(float(m.kernel_sum(x,x,True)),float(expected),rtol=1e-14)
y=x[:2].copy();cross=squared_distance(x,y)
np.testing.assert_allclose(float(m.kernel_sum(x,y)),float(cp.exp(-cross/(2*m.sigma**2)).sum()),rtol=1e-14)
root=Path('outputs/round1_v2/P5');root.mkdir(parents=True,exist_ok=True)
(root/'mathematics_validation.json').write_text(json.dumps(dict(status='PASS',device='cuda:0',direct_distances_large_nearby_vectors=True,symmetric_tiled_kernel_with_short_tile=True,cross_kernel_sum=True),indent=2)+'\n')
print('PASS: P5 exact distances and tiled biased RBF kernel sums')
