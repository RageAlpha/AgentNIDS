from pathlib import Path
import time
import cupy as cp
import numpy as np
from nids.data.preprocessing import SourcePreprocessor
from .predictions import write_json

# Direct coordinate differences preserve distances for large, nearly equal vectors.
DISTANCE = cp.RawKernel(r'''
extern "C" __global__ void squared_distance(const double* x,const double* y,double* out,int nx,int ny,int d) {
 int k=blockDim.x*blockIdx.x+threadIdx.x;
 if(k<nx*ny){int i=k/ny,j=k%ny;double s=0;
 for(int c=0;c<d;c++){double z=x[i*d+c]-y[j*d+c];s+=z*z;}out[k]=s;}
}
''','squared_distance')


def squared_distance(x,y):
    out=cp.empty((len(x),len(y)),dtype=cp.float64)
    DISTANCE(((out.size+255)//256,),(256,),(x,y,out,len(x),len(y),x.shape[1]))
    return out


class FrozenMMD:
    def __init__(self,config,prepared,output):
        started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
        self.tile=config['batches']['mmd_tile_rows']
        root=Path(config['project_root'])
        with np.load(root/config['outputs']['source']/'preprocessing/preprocessing_stats.npz') as stats:
            self.preprocessor=SourcePreprocessor(stats['fill'],stats['mean'],stats['scale'])
        source=prepared.partition('UNSW','source','S_train')
        # Uniform sampling without replacement using a registered NumPy PCG64 generator.
        ranks=np.random.default_rng(config['seed']).choice(source.rows,config['shift']['mmd']['source_reference_rows'],replace=False)+source.rank_start
        values=np.load(root/config['outputs']['feature_store']/'UNSW_X.npy',mmap_mode='r')
        self.reference=self.preprocessor.transform(values[ranks],True).astype(cp.float64)
        d2=cp.empty((len(ranks),len(ranks)),dtype=cp.float64)
        for a in range(0,len(ranks),self.tile):
            for b in range(0,len(ranks),self.tile):
                d2[a:a+self.tile,b:b+self.tile]=squared_distance(self.reference[a:a+self.tile],self.reference[b:b+self.tile])
        positive=d2[cp.triu_indices(len(ranks),1)];positive=positive[positive>0]
        self.sigma=float(cp.median(cp.sqrt(positive))) if len(positive) else config['shift']['mmd']['all_identical_bandwidth']
        self.source_term=float(cp.exp(-d2/(2*self.sigma**2)).mean())
        np.save(output/'source_reference_ranks.npy',ranks)
        np.save(output/'source_reference_standardized.npy',cp.asnumpy(self.reference))
        self.manifest=dict(status='COMPLETE',seed=config['seed'],sampling='numpy.default_rng.PCG64.choice_without_replacement',
            source_partition='S_train',source_rank_start=source.rank_start,source_rank_end_exclusive=source.rank_end_exclusive,
            reference_rows=len(ranks),sigma=self.sigma,source_kernel_mean=self.source_term,estimator='biased_MMD_squared',
            tile_rows=self.tile,device='cuda:0',seconds=time.perf_counter()-started)
        write_json(output/'reference_manifest.json',self.manifest)

    def kernel_sum(self,x,y,symmetric=False):
        total=cp.zeros((),dtype=cp.float64)
        for a in range(0,len(x),self.tile):
            for b in range(a if symmetric else 0,len(y),self.tile):
                d2=squared_distance(x[a:a+self.tile],y[b:b+self.tile])
                total+=cp.exp(-d2/(2*self.sigma**2)).sum()*(2 if symmetric and b!=a else 1)
        return total

    def batch(self,values,categorical):
        out=[]
        for window in values:
            x=self.preprocessor.transform(window,True).astype(cp.float64)
            value=self.kernel_sum(x,x,True)/len(x)**2+self.source_term-2*self.kernel_sum(x,self.reference)/(len(x)*len(self.reference))
            assert bool(cp.isfinite(value)) and float(value)>=-1e-12 and float(value)<=2+1e-12
            out.append(cp.maximum(value,0))
        return cp.stack(out).reshape(-1,1)
