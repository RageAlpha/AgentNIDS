import json
from pathlib import Path
import cupy as cp
import torch
import numpy as np
import pyarrow.parquet as pq
from cupyx.scipy.special import xlogy
from nids.reliability.round1_data import PRIMARY
from nids.reliability.cre import predict_cre

POST_FEATURES=['mask_RF','mask_XGB','mask_MLP','p_RF','p_XGB','p_MLP','r_RF','r_XGB','r_MLP','fused_probability','max_observed_probability_difference','s']
GATE_FEATURES=POST_FEATURES[:9]+['s']


_GPU_PRE={}


def primary_pre(x,m):
    key=id(m)
    if key not in _GPU_PRE:
        model=m['models']['hus'];_GPU_PRE[key]=(m,cp.asarray(m['anchor']),cp.asarray(m['mean_offset']),cp.asarray(m['scale']),torch.tensor(model['coefficients'],device='cuda',dtype=torch.float64),model['intercept'])
    _,anchor,offset,scale,beta,intercept=_GPU_PRE[key]
    z=cp.ascontiguousarray((x-anchor-offset)/scale)
    with torch.inference_mode():return cp.from_dlpack(torch.sigmoid(torch.from_dlpack(z)@beta+intercept))


def entropy(p):return -(xlogy(p,p)+xlogy(1-p,1-p))/cp.log(2.)


def fuse(mask,p,r):
    weight=mask*r;denom=weight.sum(1);weighted=(weight*p).sum(1)
    return cp.where(denom>0,weighted/cp.where(denom>0,denom,1),(mask*p).sum(1)/mask.sum(1))


def post_features(mask,p,r,s,probability=None):
    p=cp.where(mask,p,0);r=cp.where(mask,r,0)
    f=fuse(mask,p,r) if probability is None else probability
    difference=cp.where(mask,p,-cp.inf).max(1)-cp.where(mask,p,cp.inf).min(1)
    return cp.column_stack((mask,p,r,f,difference,s)).astype(cp.float64)


class Development:
    def __init__(self,c,prepared,fold,domain,partition):
        assert domain in c['folds'][fold]['development']
        root=Path(c['project_root']);part=prepared.partition(domain,'development',partition)
        windows=pq.read_table(root/c['outputs']['shift']/'10k'/f'{domain}_windows.parquet').to_pylist()
        windows=sorted([r for r in windows if r['scheme']=='development' and r['partition']==partition],key=lambda r:r['rank_start'])
        self.a=part.rank_start;self.b=windows[-1]['rank_end_exclusive'];self.n=self.b-self.a;self.domain=domain;self.partition=partition;self.fold=fold
        self.windows=windows;assert self.n==len(windows)*10000
        self.shift=np.repeat(np.array([r['shift'] for r in windows]),10000)
        self.cache=[np.load(root/c['outputs']['base_predictions']/domain/f'{m}.npy',mmap_mode='r') for m in PRIMARY]
        self.label_path=root/c['outputs']['feature_store']/f'{domain}_label.npy'
        if partition in ['R_cal','R_select']:
            self.pre_cache=[np.load(root/c['outputs']['round1']/'P4'/fold/'scores'/domain/partition/f'{m}.npy',mmap_mode='r') for m in PRIMARY]
        self.pre=[json.loads((root/c['outputs']['round1']/'P4'/fold/'models'/f'{m}.json').read_text()) for m in PRIMARY]
        self.audit=dict(domain=domain,partition=partition,rows=self.n,rank_start=self.a,rank_end_exclusive=self.b,windows=len(windows),tail=part.rank_end_exclusive-self.b,initialization_rows=sum(r['initialization'] for r in windows)*10000)
    def labels(self):return cp.asarray(np.load(self.label_path,mmap_mode='r')[self.a:self.b])
    def training_arrays(self):
        s=cp.asarray(self.shift);p=cp.column_stack([cp.asarray(v['probability'][self.a:self.b]) for v in self.cache]);u=cp.column_stack([cp.asarray(v['u'][self.a:self.b]) for v in self.cache]);h=cp.column_stack([cp.asarray(v['h'][self.a:self.b]) for v in self.cache]);r=cp.column_stack([primary_pre(cp.column_stack((h[:,i],u[:,i],s)),self.pre[i]) for i in range(3)])
        return dict(p=p,u=u,h=h,r=r,s=s,y=self.labels())
    def accessor(self,start,stop):return VisibleEvidence(self,start,stop)


class VisibleEvidence:
    """Candidate cache bytes are obtained only by the query operation."""
    def __init__(self,data,start,stop):
        self.data=data;self.start=start;self.n=stop-start;self.s=cp.asarray(data.shift[start:stop]);self.mask=cp.zeros((self.n,3),dtype=bool)
        self.p=cp.zeros((self.n,3));self.u=cp.zeros((self.n,3));self.h=cp.zeros((self.n,3));self.r=cp.zeros((self.n,3));self.reads=[]
    def query(self,rows,detector):
        assert not bool(self.mask[rows,detector].any())
        rank=cp.asnumpy(rows)+self.start+self.data.a
        cached=self.data.cache[detector][rank]
        self.p[rows,detector]=cp.asarray(cached['probability']);self.u[rows,detector]=cp.asarray(cached['u']);self.h[rows,detector]=cp.asarray(cached['h'])
        self.r[rows,detector]=cp.asarray(self.data.pre_cache[detector]['hus'][rank-self.data.a])
        self.mask[rows,detector]=True;self.reads.append((detector,len(rows)))
    def current(self,rows,current):
        assert bool(self.mask[rows,current].all())
        return cp.column_stack((self.h[rows,current],self.u[rows,current],self.s[rows],self.p[rows,current]>=.5))
