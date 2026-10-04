import tempfile
from pathlib import Path
import cupy as cp
import numpy as np
from nids.coordination.validation import check_visibility
from nids.coordination.selection import bounds,choose
from nids.coordination.replay import trajectory
from nids.coordination.evidence import post_features

constant=lambda p:dict(status='CONSTANT_SINGLE_LABEL',constant=p)
fitted={family:{f'{i}_{j}':constant(.2+.1*j) for i in range(3) for j in range(3) if i!=j} for family in ['corrective','hard','uncertainty']}
fitted.update(post=constant(.8),soft_moe=constant(.6),static_weights=[.3,.6,.8])
print(check_visibility(fitted,[1,2,3]))
class V:
 def __init__(self,n):
  self.n=n;self.s=cp.ones(n)*.2;self.mask=cp.zeros((n,3),dtype=bool);self.p=cp.zeros((n,3));self.h=cp.zeros((n,3));self.u=cp.zeros((n,3));self.r=cp.zeros((n,3));self.reads=[]
 def query(self,rows,j):
  assert not bool(self.mask[rows,j].any());self.mask[rows,j]=True;self.p[rows,j]=.1+.3*j;self.h[rows,j]=.9;self.u[rows,j]=.2;self.r[rows,j]=.9;self.reads.append((j,len(rows)))
 def current(self,rows,current):return cp.column_stack((self.h[rows,current],self.u[rows,current],self.s[rows],self.p[rows,current]>=.5))
class D:
 n=4;domain='CIC'
 def accessor(self,a,b):return V(b-a)
with tempfile.TemporaryDirectory() as t:
 for kind in ['coordinator','learned','uncertainty','hard','cheapest','random','fixed','best','uniform','static','confidence','soft']:
  path=Path(t)/f'{kind}.npy';trajectory(D(),fitted,dict(normalized_cost=[1,2,3],overhead_normalized={kind:[0.,0.,0.]},shift_normalized={'CIC':0.}),dict(selected=0,order=[0,1,2]),kind,path)
  if kind=='coordinator':
   limited=Path(t)/'limited.npy';trajectory(D(),fitted,dict(normalized_cost=[1,2,3],overhead_normalized={kind:[0.,0.,0.]},shift_normalized={'CIC':0.}),dict(selected=0,order=[0,1,2]),kind,limited,cap=3,tau_stop=.5)
   assert (np.load(limited)['query'][:,1:]==-1).all()
  x=np.load(path);assert np.isfinite(x['fused']).all() and (x['calls']<=3).all()
  if kind in ['hard','best']:assert (x['calls']==1).all()
  for row in x['query']:
   observed=row[row>=0];assert len(observed)==len(set(observed))
# Exact fully enumerated two-block resampling, including empty accepted replicates.
mult=cp.asarray([[2,0],[1,1],[0,2]]);sizes=cp.asarray([100,100]);a=cp.asarray([100,0]);e=cp.asarray([0,0]);b=bounds([(a,e),(a,e)],[(mult,sizes),(mult,sizes)],[sizes,sizes]);assert b['upper95']>.8 and b['risk']==0
print('PASS: visibility, no repeated queries, hard MoE terminal state, weighted risk and empty-resample upper bound')
