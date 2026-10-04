"""Paired target AURC differences using unchanged P7 2000 window-block resamples."""
from run_b2 import *

def main():
    point=read(OUT/'point_validation.json')
    assert point['status']=='POINT_ESTIMATES_COMPLETE'
    if (OUT/'paired_aurc_comparisons.csv').exists(): raise RuntimeError('Bootstrap output already exists')
    scopes0={(r['domain'],r['partition']):{k:int(v) if k in ['rank_start','rank_end_exclusive','rows','windows','excluded_tail','benign','attack','initialization_rows'] else v for k,v in r.items()} for r in rows(OUT/'evaluation_scope.csv')}
    directions={(r['fold'],r['model'],r['output']):r for r in rows(OUT/'entropy_direction_audit.csv') if r['partition']=='T_heldout'}
    metrics={(r['fold'],r['model'],r['output'],r['method']):r for r in rows(OUT/'development_target_metrics.csv') if r['partition']=='T_heldout' and r['stratum']=='all'}
    comparisons=[];started=time.time()
    for fold,info in CFG['folds'].items():
        d=info['heldout'];scope=scopes0[d,'T_heldout'];n=scope['rows']
        block_rows=CFG['bootstrap']['block_windows']*10000
        blocks=(n+block_rows-1)//block_rows;reps=CFG['bootstrap']['replicates']
        path=ROOT/f'outputs/outer_v2/P7/{fold}/round1/bootstrap/{d}_block_multiplicity.npy'
        multiplicity=cp.asarray(np.load(path,mmap_mode='r'))
        assert multiplicity.shape==(reps,blocks) and bool((multiplicity.sum(axis=1)==blocks).all())
        metadata=dict(domain=d,fold=fold,rows=n,blocks=blocks,block_rows=block_rows,last_block_rows=n-(blocks-1)*block_rows,
            replicates=reps,multiplicity_source=str(path),minimum_blocks=CFG['bootstrap']['minimum_units_per_evaluation_domain'],
            paired='same saved P7 multiplicity for both scores and all six output/model conditions in this domain',
            scope='exploratory conditional on current domain and fitted models; not future-domain guarantee')
        write(OUT/'bootstrap'/f'{d}_manifest.json',metadata)
        for m in MODELS:
            for o in ['Raw','Platt']:
                fit=read(OUT/'models'/f'{fold}__{m}__{o}.json')
                key=dict(fold=fold,domain=d,model=m,output=o)
                if fit['status']!='COMPLETE':
                    comparisons.append(dict(**key,status=fit['status'],delta_fu_minus_fixed=None,CI95_low=None,CI95_high=None))
                    save('paired_aurc_comparisons.csv',comparisons);continue
                audit=directions[fold,m,o]
                fa=float(metrics[fold,m,o,'fu']['AURC']);ba=float(metrics[fold,m,o,'one_minus_u']['AURC']);delta=fa-ba
                if blocks<CFG['bootstrap']['minimum_units_per_evaluation_domain']:
                    comparisons.append(dict(**key,status='NOT_ESTIMABLE_INSUFFICIENT_BLOCKS',delta_fu_minus_fixed=delta,CI95_low=None,CI95_high=None))
                    save('paired_aurc_comparisons.csv',comparisons);continue
                if audit['exact_same_order_and_record_tie_partition']=='True':
                    assert abs(delta)<1e-12
                    comparisons.append(dict(**key,status='EXACT_ZERO_IDENTICAL_ORDER_AND_TIES',AURC_fixed=ba,AURC_fu=fa,
                        delta_fu_minus_fixed=0.,point_float_residual=delta,CI95_low=0.,CI95_high=0.,replicates=0,
                        blocks=blocks,last_block_rows=metadata['last_block_rows'],multiplicity_source=str(path),
                        reason='all-record adjacent boundary audit proves same ordered record groups; every paired resample difference is exactly zero'))
                    save('paired_aurc_comparisons.csv',comparisons)
                    print(f'{fold}/{m}/{o}: exact same order/ties; bootstrap difference identically zero',flush=True)
                    continue
                print(f'BOOTSTRAP {fold}/{d}/{m}/{o}: delta={delta:.9f}',flush=True)
                u,pred,y=load_data(d,m,o,scope);correct=pred==y
                base=1-u;fu=predict(u,fit)
                block=cp.arange(n,dtype=cp.int64)//block_rows
                h=harmonic_numbers(blocks*block_rows)
                values={}
                for method,score in [('one_minus_u',base),('fu',fu)]:
                    t=time.time();ranking=BlockRanking(score,correct,block,blocks)
                    ones=cp.ones(blocks,dtype=cp.int64)
                    full=float(ranking.aurc(ones,h))
                    assert abs(full-float(metrics[fold,m,o,method]['AURC']))<1e-12
                    result=cp.empty(reps,dtype=cp.float64)
                    for rep in range(reps):
                        result[rep]=ranking.aurc(multiplicity[rep],h)
                        if (rep+1)%200==0: print(f'  {method}: {rep+1}/{reps}, {time.time()-t:.1f}s',flush=True)
                    assert bool(cp.isfinite(result).all())
                    values[method]=cp.asnumpy(result)
                    print(f'  {method}: groups={ranking.groups:,}, group-block cells={ranking.entries:,}',flush=True)
                    del ranking,result;release()
                difference=values['fu']-values['one_minus_u']
                ci=percentile_ci(difference)
                np.savez_compressed(OUT/'bootstrap'/f'{fold}__{m}__{o}_paired_AURC.npz',**values,difference=difference)
                comparisons.append(dict(**key,status='PAIRED_2000_PERCENTILE_CI',AURC_fixed=ba,AURC_fu=fa,
                    delta_fu_minus_fixed=delta,point_float_residual=0.,CI95_low=ci[0],CI95_high=ci[1],replicates=reps,
                    blocks=blocks,last_block_rows=metadata['last_block_rows'],multiplicity_source=str(path),
                    reason='same P7 record blocks resampled for both scores; negative delta favors fu; exploratory interval'))
                save('paired_aurc_comparisons.csv',comparisons)
                print(f'  CI95={ci}',flush=True)
                del u,pred,y,correct,base,fu,block,h,values;release()
        del multiplicity;release()
    assert len(comparisons)==18
    before=read(OUT/'upstream_inventory.json');assert before==inventory()
    write(OUT/'bootstrap_validation.json',dict(status='COMPLETE',comparisons=len(comparisons),seconds=time.time()-started,
        exact_zero_cases=sum(r['status']=='EXACT_ZERO_IDENTICAL_ORDER_AND_TIES' for r in comparisons),
        paired_2000_cases=sum(r['status']=='PAIRED_2000_PERCENTILE_CI' for r in comparisons),
        all_blocks_supported=True,paired_multiplicities_checked=True,original_scores_and_records_preserved=True,
        upstream_size_mtime_unchanged=True))
    print('B2 paired comparisons complete',flush=True)

if __name__=='__main__': main()
