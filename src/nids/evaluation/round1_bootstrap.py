from __future__ import annotations

import gc
import time
from pathlib import Path
import cupy as cp
import numpy as np
from .round1_metrics import BlockRanking,harmonic_numbers
from nids.reliability.round1_data import PRIMARY,METHODS
from nids.replay.predictions import write_json


def block_pattern_counts(errors,block,blocks):
    code=errors[:,0].astype(cp.int64)+2*errors[:,1]+4*errors[:,2]
    return cp.bincount(block*8+code,minlength=blocks*8).reshape(blocks,8).astype(cp.float64)


def sampled_headroom(counts):
    total=counts.sum()
    errors=cp.stack([counts[cp.asarray([k for k in range(8) if k & (1<<i)])].sum() for i in range(3)])
    return (errors.min()-counts[7])/total


def domain_bootstrap(config,data,score_folder,output,rng,methods=METHODS):
    started=time.perf_counter()
    n=data['audit']['rows'];domain=data['audit']['domain']
    window_size=data['audit'].get('window_size',10000)
    block_rows=config['bootstrap']['block_windows']*window_size
    block=cp.arange(n,dtype=cp.int64)//block_rows
    blocks=int(block[-1])+1
    reps=config['bootstrap']['replicates']
    samples=rng.randint(0,blocks,size=(reps,blocks),dtype=cp.int32)
    offset=cp.arange(reps,dtype=cp.int64)[:,None]*blocks
    multiplicities=cp.bincount((samples+offset).ravel(),minlength=reps*blocks).reshape(reps,blocks)
    assert bool(cp.all(multiplicities.sum(axis=1)==blocks))
    np.save(output/f'{domain}_sampled_blocks.npy',cp.asnumpy(samples))
    np.save(output/f'{domain}_block_multiplicity.npy',cp.asnumpy(multiplicities))
    patterns=block_pattern_counts(data['errors'],block,blocks)
    controls=block_pattern_counts(data['control_errors'],block,blocks)
    boot_g0=cp.empty((reps,2),dtype=cp.float64)
    for rep in range(reps):
        boot_g0[rep,0]=sampled_headroom(multiplicities[rep]@patterns)
        boot_g0[rep,1]=sampled_headroom(multiplicities[rep]@controls)
    harmonic=harmonic_numbers(blocks*block_rows)
    boot_auc=np.empty((reps,len(PRIMARY),len(methods)),dtype=np.float64)
    ranking_metadata=[]
    for i,mid in enumerate(PRIMARY):
        saved=np.load(score_folder/f'{mid}.npy',mmap_mode='r')
        correct=(~data['errors'][:,i]).astype(cp.uint8)
        for j,method in enumerate(methods):
            t=time.perf_counter()
            ranking=BlockRanking(cp.asarray(saved[method]),correct,block,blocks)
            result=cp.empty(reps,dtype=cp.float64)
            for rep in range(reps):
                result[rep]=ranking.aurc(multiplicities[rep],harmonic)
            assert bool(cp.isfinite(result).all())
            boot_auc[:,i,j]=cp.asnumpy(result)
            elapsed=time.perf_counter()-t
            ranking_metadata.append(dict(model=mid,method=method,tie_groups=ranking.groups,group_block_cells=ranking.entries,seconds=elapsed))
            print(f'{domain} bootstrap {mid}/{method}: {reps} paired replicates, groups={ranking.groups:,}, cells={ranking.entries:,}, {elapsed:.2f}s',flush=True)
            del ranking,result
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
    np.save(output/f'{domain}_bootstrap_AURC.npy',boot_auc)
    np.save(output/f'{domain}_bootstrap_G0.npy',cp.asnumpy(boot_g0))
    record=dict(domain=domain,rows=n,windows=n//window_size,window_size=window_size,blocks=blocks,last_block_rows=n-(blocks-1)*block_rows,
        replicates=reps,model_order=PRIMARY,method_order=methods,seconds=time.perf_counter()-started,
        ranks_preserved=True,shift='original_record_score_preserved',paired_samples='same_multiplicity_for_all_models_methods_and_G0',
        G0_best_single='recomputed_inside_each_replicate',ranking=ranking_metadata)
    write_json(output/f'{domain}_bootstrap_manifest.json',record)
    return cp.asnumpy(boot_g0),boot_auc,record


def percentile_ci(values):
    return cp.asnumpy(cp.percentile(cp.asarray(values),cp.asarray([2.5,97.5]))).tolist()


def gate(effect,ci,threshold,estimable):
    if estimable and effect>=threshold and ci[0]>0:
        return 'PASS'
    if estimable and ci[1]<threshold:
        return 'BELOW_MINIMUM_EFFECT'
    return 'INCONCLUSIVE'
