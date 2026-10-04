from __future__ import annotations

from pathlib import Path
import cupy as cp
import numpy as np
import pyarrow.parquet as pq

PRIMARY=['rf_Traffic','xgb_Protocol','mlp_Behavior']
CONTROL=['rf_All49','xgb_All49','mlp_All49']
METHODS=['prior','raw_h','u','hu','hus','one_minus_u']
SCORE_DTYPE=np.dtype([(m,'<f8') for m in METHODS])


def load_partition(config,prepared,fold,domain,partition,for_fit=False):
    project=Path(config['project_root'])
    assert domain in config['folds'][fold]['development'] and domain!=config['folds'][fold]['heldout']
    p=prepared.partition(domain,'development',partition)
    windows=[r for r in prepared.windows if r['domain']==domain and r['scheme']=='development' and r['partition']==partition and int(r['window_size'])==10000]
    windows=sorted(windows,key=lambda r:int(r['rank_start']))
    end=int(windows[-1]['rank_end_exclusive']);n=end-p.rank_start
    assert n==len(windows)*10000
    filename=project/config['outputs']['round1']/'P3/development'/f'fold={fold}'/f'domain={domain}'/f'partition={partition}'/'records.parquet'
    columns=['rank','original_row','complete_window','initialization','shift']
    for mid in PRIMARY:
        columns += [mid+'__h',mid+'__u',mid+'__predicted_class']
    if not for_fit:
        columns += [m+'__probability' for m in PRIMARY+CONTROL]+[m+'__predicted_class' for m in CONTROL]
    values={name:np.empty(n,dtype=np.uint8 if name.endswith('__predicted_class') else np.float64) for name in columns if name not in ['rank','original_row','complete_window','initialization']}
    order=prepared.sorted_rows(domain)
    cursor=p.rank_start;initialized=0
    for batch in pq.ParquetFile(filename).iter_batches(batch_size=config['batches']['inference_rows'],columns=columns):
        selection=batch.column('complete_window').to_numpy(zero_copy_only=False)
        rank=batch.column('rank').to_numpy()[selection]
        if len(rank)==0:
            continue
        assert np.array_equal(rank,np.arange(cursor,cursor+len(rank)))
        assert np.array_equal(batch.column('original_row').to_numpy()[selection],order[cursor:cursor+len(rank)])
        initialized+=int(batch.column('initialization').to_numpy(zero_copy_only=False)[selection].sum())
        for name in values:
            values[name][cursor-p.rank_start:cursor-p.rank_start+len(rank)]=batch.column(name).to_numpy(zero_copy_only=False)[selection]
        cursor+=len(rank)
    assert cursor==end
    labels=cp.asarray(np.load(project/config['outputs']['feature_store']/f'{domain}_label.npy',mmap_mode='r')[p.rank_start:end])
    shift=cp.asarray(values['shift'])
    result=dict(domain=domain,partition=partition,rows=n,windows=len(windows),rank_start=p.rank_start,rank_end_exclusive=end,
        excluded_tail=p.rank_end_exclusive-end,initialization_rows=initialized,
        benign=int((labels==0).sum()),attack=int((labels==1).sum()))
    x={m:cp.stack((cp.asarray(values[m+'__h']),cp.asarray(values[m+'__u']),shift),axis=1) for m in PRIMARY}
    errors=cp.stack([cp.asarray(values[m+'__predicted_class'])!=labels for m in PRIMARY],axis=1)
    output=dict(audit=result,x=x,labels=labels,errors=errors)
    if not for_fit:
        output['control_errors']=cp.stack([cp.asarray(values[m+'__predicted_class'])!=labels for m in CONTROL],axis=1)
        output['probability']={m:cp.asarray(values[m+'__probability']) for m in PRIMARY+CONTROL}
        for i,m in enumerate(PRIMARY+CONTROL):
            assert bool(cp.all((output['probability'][m]>=.5)==cp.asarray(values[m+'__predicted_class'])))
    print(f'{fold}/{domain}/{partition}: {n:,} complete-window rows, init={initialized}, tail excluded={result["excluded_tail"]}',flush=True)
    return output
