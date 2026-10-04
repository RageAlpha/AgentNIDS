from pathlib import Path
import cupy as cp
import numpy as np
import pyarrow.parquet as pq
from .round1_data import PRIMARY,CONTROL


def sensitivity_loader(window_size,shift_root):
    def load(config,prepared,fold,domain,partition,for_fit=False):
        root=Path(config['project_root']);p=prepared.partition(domain,'development',partition)
        assert domain in config['folds'][fold]['development']
        rows=pq.read_table(Path(shift_root)/f'{domain}_windows.parquet').to_pylist()
        windows=sorted([r for r in rows if r['scheme']=='development' and r['partition']==partition],key=lambda r:r['rank_start'])
        a=p.rank_start;b=windows[-1]['rank_end_exclusive'];n=b-a
        assert n==len(windows)*window_size
        for i,r in enumerate(windows):assert r['rank_start']==a+i*window_size and r['rank_end_exclusive']==a+(i+1)*window_size
        shift=cp.repeat(cp.asarray([r['shift'] for r in windows]),window_size)
        labels=cp.asarray(np.load(root/config['outputs']['feature_store']/f'{domain}_label.npy',mmap_mode='r')[a:b])
        cached={m:np.load(root/config['outputs']['base_predictions']/domain/f'{m}.npy',mmap_mode='r')[a:b] for m in (PRIMARY if for_fit else PRIMARY+CONTROL)}
        x={m:cp.stack((cp.asarray(cached[m]['h']),cp.asarray(cached[m]['u']),shift),axis=1) for m in PRIMARY}
        for z in x.values():assert bool(cp.isfinite(z).all())
        audit=dict(domain=domain,partition=partition,rows=n,windows=len(windows),window_size=window_size,rank_start=a,rank_end_exclusive=b,
            excluded_tail=p.rank_end_exclusive-b,initialization_rows=sum(r['initialization'] for r in windows)*window_size,
            benign=int((labels==0).sum()),attack=int((labels==1).sum()),identity='prediction_and_P1_labels_indexed_by_same_domain_rank',shift_file=str(Path(shift_root)/f'{domain}_windows.parquet'))
        result=dict(audit=audit,x=x,labels=labels,errors=cp.stack([cp.asarray(cached[m]['predicted_class'])!=labels for m in PRIMARY],axis=1))
        if not for_fit:
            result['control_errors']=cp.stack([cp.asarray(cached[m]['predicted_class'])!=labels for m in CONTROL],axis=1)
            result['probability']={m:cp.asarray(cached[m]['probability']) for m in PRIMARY+CONTROL}
            for m,pred in result['probability'].items():assert bool(cp.all((pred>=.5)==cp.asarray(cached[m]['predicted_class'])))
        print(f'{fold}/{domain}/{partition}/{window_size}: rows={n:,}, tail={audit["excluded_tail"]}',flush=True)
        return result
    return load
