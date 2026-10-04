"""Phase 2: reads frozen thresholds; computes C0/C1 aggregates and paired bootstrap.
No threshold selection function is imported or called. No calibration resampling.
"""
from common import *

def category_rows(k,method,split,status,threshold,book,counts,support,segment=None):
    result=[]
    for code,name in book.items():
        ci=int(code); n=int(support[ci]); c=counts[ci] if counts is not None else None
        truth=0 if name=='Benign' else 1
        if c is not None:
            assert int(c.sum())==n
            assert (c[2:].sum()==0 if truth==0 else c[:2].sum()==0)
        result.append(dict(**k,method=method,split=split,segment=segment,status=status,threshold=threshold,
            original_attack_code=ci,original_attack_category=name,binary_label=truth,category_support=n,
            **{v:int(c[i]) if c is not None else None for i,v in enumerate(COUNTS)},
            attack_recall=div(int(c[3]),n) if c is not None and truth==1 else None,
            benign_FPR=div(int(c[1]),n) if c is not None and truth==0 else None,
            category_correct_rate=div(int(c[0]+c[3]),n) if c is not None else None,
            undefined_reason=UNSUPPORTED if c is None else 'ZERO_CATEGORY_SUPPORT' if not n else ''))
    return result

def main():
    check_frozen()
    assert not (OUT/'evaluation_access.json').exists(), 'Evaluation already started; do not silently overwrite'
    protocol=read(OUT/'protocol.json'); frozen=rows(OUT/'classification_thresholds.csv')
    lookup={(r['domain'],r['model'],r['output']):r for r in frozen}
    sample=rows(PRE/'calibration_sample_manifest.csv')
    calibration=rows(OUT/'calibration_metrics_frozen.csv')
    metric_rows=[]; cats=[]; temporals=[]; temporal_cats=[]; paired=[]; gaps=[]; checks=[]; bootstrap_summary=[]
    (OUT/'bootstrap').mkdir(exist_ok=True); (OUT/'window_aggregates').mkdir(exist_ok=True)
    matrices={}
    for d,(_,end,nw,nb,na) in DOMAINS.items():
        blocks=(nw+9)//10; assert blocks>=20
        seed=protocol['bootstrap_execution']['seeds'][d]
        rng=np.random.Generator(np.random.PCG64(seed))
        indices=rng.integers(0,blocks,size=(2000,blocks),dtype=np.int32)
        mult=np.zeros((2000,blocks),dtype=np.int64)
        for i in range(2000): mult[i]=np.bincount(indices[i],minlength=blocks)
        assert (mult.sum(axis=1)==blocks).all()
        np.save(OUT/f'bootstrap/{d}_sampled_blocks.npy',indices)
        np.save(OUT/f'bootstrap/{d}_block_multiplicity.npy',mult)
        blockrefs=[dict(domain=d,block=i,window_index_start=10*i,window_index_end_exclusive=min(nw,10*(i+1)),
            rank_start=100000+100000*i,rank_end_exclusive=min(end,100000+100000*(i+1)),
            rows=min(end,100000+100000*(i+1))-(100000+100000*i)) for i in range(blocks)]
        save(OUT/f'bootstrap/{d}_block_scope.csv',blockrefs)
        bmeta=dict(domain=d,seed=seed,blocks=blocks,last_block_rows=blockrefs[-1]['rows'],replicates=2000,
            evaluation_rank_interval=[100000,end],generator=protocol['bootstrap_execution']['generator'],
            interval='95% percentile; linear quantile; finite replicates only; counts disclosed',
            shared_across_all_methods=True,new_matrix=True,old_P7_matrix_loaded=False,
            sampled_blocks_sha256=sha(OUT/f'bootstrap/{d}_sampled_blocks.npy'),
            block_multiplicity_sha256=sha(OUT/f'bootstrap/{d}_block_multiplicity.npy'))
        write(OUT/f'bootstrap/{d}_manifest.json',bmeta); matrices[d]=mult
    check_frozen()
    access=dict(first_evaluation_label_access_after_utc=now(),threshold_freeze_sha256=sha(OUT/'threshold_freeze.json'),
        threshold_freeze_at_utc=read(OUT/'threshold_freeze.json')['frozen_at_utc'],domains=[])
    assert access['first_evaluation_label_access_after_utc']>access['threshold_freeze_at_utc']
    write(OUT/'evaluation_access.json',access)
    for d,(_,end,nw,nb,na) in DOMAINS.items():
        log(f'{d}: opening evaluation labels only after all thresholds committed.')
        labels=np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
        codes=np.load(ROOT/f'outputs/feature_store_v2/{d}_attack_code.npy',mmap_mode='r')
        book=read(ROOT/f'outputs/feature_store_v2/{d}_manifest.json')['attack_codebook']; nc=len(book)
        assert sorted(map(int,book))==list(range(nc))
        rec=[r for r in sample if r['domain']==d]
        ranks=np.array([int(r['availability_rank']) for r in rec]); cy=np.array([int(r['binary_label']) for r in rec],dtype=np.uint8)
        cc=np.array([int(r['original_attack_code']) for r in rec]); csupport=np.bincount(cc,minlength=nc)
        splits=np.r_[0,np.cumsum(SEGMENTS[d])]; mult=matrices[d]
        for m in MODELS:
            cached=cache(d,m)
            for output,field in FIELDS.items():
                k=key(d,m,output); r=lookup[d,m,output]; t1=None if r['threshold']=='NA' else np.float64(r['threshold'])
                assert t1 is None or float(t1).hex()==r['threshold_hex']
                thresholds=[np.float64(.5),t1]; wc=np.zeros((2,nw,nc,4),dtype=np.int64)
                cp=np.array(cached[field][ranks],dtype=np.float64)
                assert hashlib.sha256(cp.tobytes()).hexdigest()==r['calibration_probability_sha256']
                # Contiguous chunks are aligned to the original complete 10k windows.
                direct=np.zeros((2,4),dtype=np.int64)
                for lo in range(100000,end,200000):
                    hi=min(end,lo+200000); n=hi-lo; first=(lo-100000)//10000; wn=n//10000
                    y=np.array(labels[lo:hi],dtype=np.int64); c=np.array(codes[lo:hi],dtype=np.int64)
                    p=np.array(cached[field][lo:hi],dtype=np.float64)
                    assert np.isin(y,[0,1]).all() and ((c>=0)&(c<nc)).all()
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    catwin=(np.repeat(np.arange(wn),10000)*nc+c)*4+y*2
                    for mi,t in enumerate(thresholds):
                        if t is None: continue
                        pred=(p>=t).astype(np.int64)
                        counts=np.bincount(catwin+pred,minlength=wn*nc*4).reshape(wn,nc,4)
                        assert (counts.sum(axis=(1,2))==10000).all()
                        wc[mi,first:first+wn]=counts
                        direct[mi]+=np.bincount(y*2+pred,minlength=4)
                valid_methods=1 if t1 is None else 2
                for mi in range(valid_methods): assert np.array_equal(wc[mi].sum(axis=(0,1)),direct[mi])
                if t1 is not None: assert np.array_equal(wc[0].sum(axis=2),wc[1].sum(axis=2))
                np.savez_compressed(OUT/f'window_aggregates/{d}__{m}__{output}.npz',
                    counts=wc[:valid_methods],methods=np.array(['C0-class','C1-class'][:valid_methods]),
                    confusion_order=np.array(COUNTS),category_codes=np.arange(nc),evaluation_rank_start=np.int64(100000),
                    window_rows=np.int64(10000),thresholds=np.array(thresholds[:valid_methods],dtype=np.float64))
                esupport=wc[0].sum(axis=(0,2)); overall={}; cal={}; boot={}
                for mi,method in enumerate(['C0-class','C1-class']):
                    t=thresholds[mi]; status='COMPLETE' if mi==0 else r['status']; available=t is not None
                    calcount=confusion(cy,cp,t) if available else None
                    cal[mi]=metrics(calcount); overall[mi]=metrics(direct[mi] if available else None)
                    for split,n,b,a,vals in [('calibration',400,nb,na,cal[mi]),('evaluation',end-100000,int(direct[0,0]+direct[0,1]),int(direct[0,2]+direct[0,3]),overall[mi])]:
                        metric_rows.append(dict(**k,method=method,split=split,status=status,threshold=float(t) if available else None,
                            rows=n,accepted_rows=n if available else None,benign=b,attack=a,**vals))
                    calcat=np.bincount(cc*4+cy.astype(np.int64)*2+(cp>=t),minlength=nc*4).reshape(nc,4) if available else None
                    cats+=category_rows(k,method,'calibration',status,t,book,calcat,csupport)
                    cats+=category_rows(k,method,'evaluation',status,t,book,wc[mi].sum(axis=0) if available else None,esupport)
                    for si,(a,b) in enumerate(zip(splits[:-1],splits[1:])):
                        c=wc[mi,a:b].sum(axis=(0,1)) if available else None
                        baseline=wc[0,a:b].sum(axis=(0,1))
                        temporals.append(dict(**k,method=method,split='evaluation',segment=si+1,status=status,threshold=t,
                            rank_start=100000+int(a)*10000,rank_end_exclusive=100000+int(b)*10000,
                            windows=int(b-a),rows=int(b-a)*10000,accepted_rows=int(b-a)*10000 if available else None,
                            benign=int(baseline[:2].sum()),attack=int(baseline[2:].sum()),**metrics(c)))
                        temporal_cats+=category_rows(k,method,'evaluation',status,t,book,
                            wc[mi,a:b].sum(axis=0) if available else None,wc[0,a:b].sum(axis=(0,2)),segment=si+1)
                    if available:
                        windows=wc[mi].sum(axis=1)
                        blocks=np.add.reduceat(windows,np.arange(0,nw,10),axis=0)
                        assert np.array_equal(blocks.sum(axis=0),direct[mi])
                        bc=mult@blocks; assert (bc.sum(axis=1)>0).all()
                        boot[mi]={**{v:bc[:,i].astype(np.float64) for i,v in enumerate(COUNTS)},**rate_arrays(bc)}
                        for metric,vals in boot[mi].items():
                            finite=vals[np.isfinite(vals)]
                            ci=np.percentile(finite,[2.5,97.5],method='linear') if len(finite) else [None,None]
                            bootstrap_summary.append(dict(**k,method=method,metric=metric,estimate=overall[mi][metric],
                                ci_low=ci[0],ci_high=ci[1],valid_replicates=len(finite),total_replicates=2000))
                    else: boot[mi]={v:np.full(2000,np.nan) for v in METRICS}
                    for metric in METRICS:
                        x,z=cal[mi][metric],overall[mi][metric]
                        gaps.append(dict(**k,method=method,status=status,metric=metric,calibration_value=x,evaluation_value=z,
                            evaluation_minus_calibration=z-x if x is not None and z is not None else None,
                            units='count; denominators differ' if metric in COUNTS else 'absolute rate; evaluation minus calibration',
                            calibration_rows=400,evaluation_rows=end-100000))
                deltas={}
                for metric in METRICS:
                    vals=boot[1][metric]-boot[0][metric]; finite=vals[np.isfinite(vals)]; deltas[metric]=vals
                    ci=np.percentile(finite,[2.5,97.5],method='linear') if len(finite) else [None,None]
                    for split,points in [('calibration',cal),('evaluation',overall)]:
                        c0,c1=points[0][metric],points[1][metric]
                        paired.append(dict(**k,split=split,contrast='C1-class minus C0-class',status=r['status'],metric=metric,
                            C0=c0,C1=c1,difference=c1-c0 if c1 is not None and c0 is not None else None,
                            ci_low=ci[0] if split=='evaluation' else None,ci_high=ci[1] if split=='evaluation' else None,
                            valid_replicates=len(finite) if split=='evaluation' else None,
                            total_replicates=2000 if split=='evaluation' else None,
                            units='count' if metric in COUNTS else 'absolute rate (0-1)',
                            interval_note='95% paired block percentile; finite replicates only' if split=='evaluation' else 'descriptive calibration difference; no calibration resampling'))
                np.savez_compressed(OUT/f'bootstrap/{d}__{m}__{output}_replicates.npz',
                    **{f'C0_{v}':x for v,x in boot[0].items()},**{f'C1_{v}':x for v,x in boot[1].items()},
                    **{f'delta_{v}':x for v,x in deltas.items()})
                # Independent frozen calibration table equality, including NA behavior.
                for mi,method in enumerate(['C0-class','C1-class']):
                    old=next(x for x in calibration if (x['domain'],x['model'],x['output'],x['method'])==(d,m,output,method))
                    for metric in METRICS:
                        assert (old[metric]=='NA' and cal[mi][metric] is None) or (old[metric]!='NA' and float(old[metric])==cal[mi][metric])
                checks.append(dict(**k,evaluation_rows=end-100000,C0_all_accepted=True,
                    C1_all_accepted=True if t1 is not None else None,window_category_overall_counts_match=True,
                    independent_chunk_overall_counts_match=True,calibration_matches_frozen_table=True,
                    C1_status=r['status'],C1_threshold_unchanged=True))
                log(f'{d}/{m}/{output}: evaluated C0'+('/C1' if t1 is not None else '; C1 NA')+'; aggregates/bootstrap saved.')
        access['domains'].append(dict(domain=d,rank_start=100000,rank_end_exclusive=end,unique_evaluation_labels=end-100000,
            unsampled_startup_labels_read=0,completed_at_utc=now()))
        write(OUT/'evaluation_access.json',access)
    save(OUT/'classification_metrics.csv',metric_rows)
    save(OUT/'attack_category_metrics.csv',cats)
    save(OUT/'temporal_metrics.csv',temporals)
    save(OUT/'temporal_attack_category_metrics.csv',temporal_cats)
    save(OUT/'paired_comparisons.csv',paired)
    save(OUT/'calibration_evaluation_gap.csv',gaps)
    save(OUT/'bootstrap_intervals.csv',bootstrap_summary)
    write(OUT/'evaluation_validation.json',checks)
    check_frozen()
    write(OUT/'evaluation_complete.json',dict(completed_at_utc=now(),actual_C0_conditions=36,actual_C1_conditions=24,
        unsupported_C1_conditions=12,total_unique_evaluation_records=64260000,windows=6426,C2='NOT_RUN',C0_select='NOT_RUN'))
    log('C0/C1 evaluation and 2000 paired block replicates complete; thresholds unchanged.')

if __name__=='__main__': main()
