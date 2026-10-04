from common import *
import cupy as cp
engine=score_engine()
METHODS=['C0-select','C2-select','C0-class']
BMET=['coverage','effective_attack_detection','wrong_accepted_attack_fraction','selective_risk']

def rate(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
def bootrates(c,bg):
    total=bg.sum(axis=1); attack=bg[:,2:].sum(axis=1)
    return dict(coverage=rate(c.sum(axis=1),total),effective_attack_detection=rate(c[:,3],attack),
        wrong_accepted_attack_fraction=rate(c[:,2],attack),selective_risk=rate(c[:,1]+c[:,2],c.sum(axis=1)))
def main():
    check_frozen(); assert not (OUT/'evaluation_access.json').exists()
    start=now(); assert start>read(OUT/'threshold_freeze.json')['frozen_at_utc']
    write(OUT/'evaluation_access.json',dict(started_at_utc=start,threshold_freeze_sha256=sha(OUT/'threshold_freeze.json'),
        target_label_ranges={d:[100000,v[1]] for d,v in DOMAINS.items()},unsampled_startup_labels_read=0))
    t0={(r['domain'],r['model'],r['output'],r['score']):r for r in rows(OUT/'C0_select_thresholds.csv')}
    t2={(r['domain'],r['model'],r['output'],r['score']):r for r in rows(OUT/'C2_acceptance_thresholds.csv')}
    old={(r['domain'],r['model'],r['output']):r for r in rows(C1/'classification_metrics.csv') if r['method']=='C0-class' and r['split']=='evaluation'}
    allmetrics=[]; temporal=[]; cats=[]; tcats=[]; pairs=[]; gaps=[]; checks=[]
    cal=rows(OUT/'target_calibration_metrics.csv')
    (OUT/'window_aggregates').mkdir(exist_ok=True); (OUT/'bootstrap_replicates').mkdir(exist_ok=True)
    for d,(_,end,nw,nb,na) in DOMAINS.items():
        ycache=np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
        ccache=np.load(ROOT/f'outputs/feature_store_v2/{d}_attack_code.npy',mmap_mode='r')
        book=read(ROOT/f'outputs/feature_store_v2/{d}_manifest.json')['attack_codebook']; nc=len(book)
        bounds=np.r_[0,np.cumsum(SEGMENTS[d])]; mult=np.load(C1/f'bootstrap/{d}_block_multiplicity.npy')
        for m in MODELS:
            cached=cache(d,m)
            for o,field in FIELDS.items():
                fit=read(B2/f'models/{DOMAINS[d][0]}__{m}__{o}.json')
                counts=np.zeros((2,3,nw,nc,4),dtype=np.int64)
                thresholds={s:[float(t0[d,m,o,s]['threshold']),None if t2[d,m,o,s]['threshold']=='NA' else float(t2[d,m,o,s]['threshold']),ACCEPT_ALL] for s in SCORES}
                for a in range(100000,end,200000):
                    b=min(end,a+200000); wn=(b-a)//10000; first=(a-100000)//10000
                    p=np.array(cached[field][a:b],dtype=np.float64); y=np.array(ycache[a:b],dtype=np.int64); c=np.array(ccache[a:b],dtype=np.int64)
                    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and np.isin(y,[0,1]).all()
                    pu=cp.asarray(p); u=engine.entropy(pu); gpu={'one_minus_u':1-u,'fu':engine.predict(u,fit)}
                    code=(np.repeat(np.arange(wn),10000)*nc+c)*4+y*2+(p>=.5)
                    for si,s in enumerate(SCORES):
                        score=cp.asnumpy(gpu[s]); assert np.isfinite(score).all() and ((score>=0)&(score<=1)).all()
                        for mi,t in enumerate(thresholds[s]):
                            if t is None: continue
                            mask=score>=np.float64(t)
                            cnt=np.bincount(code[mask],minlength=wn*nc*4).reshape(wn,nc,4)
                            assert cnt.sum()==mask.sum()
                            counts[si,mi,first:first+wn]=cnt
                assert np.array_equal(counts[0,2],counts[1,2])
                assert np.array_equal(counts[0,2].sum(axis=(0,1)),[int(old[d,m,o][v]) for v in COUNTS])
                for si,s in enumerate(SCORES):
                    k=dict(**key(d,m,o),score=s); stem=f'{d}__{m}__{o}__{s}'; wc=counts[si]
                    available=[True,thresholds[s][1] is not None,True]
                    statuses=[t0[d,m,o,s]['status'],t2[d,m,o,s]['status'],'COMPLETE']
                    np.savez_compressed(OUT/f'window_aggregates/{stem}.npz',counts=wc,methods=np.array(METHODS),
                        method_available=np.array(available),category_codes=np.arange(nc),confusion_order=np.array(COUNTS),
                        classification_threshold=.5,evaluation_rank_start=100000,window_size=10000)
                    bg=wc[2].sum(axis=(0,1)); totals=(bg[:2].sum(),bg[2:].sum())
                    bgblock=np.add.reduceat(wc[2].sum(axis=1),np.arange(0,nw,10)); bootbg=mult@bgblock
                    points=[]; br=[]
                    for mi,method in enumerate(METHODS):
                        c=wc[mi].sum(axis=(0,1)) if available[mi] else None
                        vals=selective(c,totals); points.append(vals)
                        allmetrics.append(dict(**k,method=method,source='target_evaluation',status=statuses[mi],threshold=thresholds[s][mi],classification_threshold=.5,**vals))
                        def catrows(segment,a,b):
                            support=wc[2,a:b].sum(axis=(0,2)); accepted=wc[mi,a:b].sum(axis=0) if available[mi] else None
                            result=[]
                            for code,name in book.items():
                                ci=int(code); n=int(support[ci]); tot=(n,0) if name=='Benign' else (0,n)
                                co=accepted[ci] if accepted is not None else None
                                if co is not None: assert (co[2:].sum()==0 if name=='Benign' else co[:2].sum()==0)
                                result.append(dict(**k,method=method,segment=segment,status=statuses[mi],original_attack_code=ci,
                                    original_attack_category=name,category_support=n,**selective(co,tot)))
                            return result
                        cats+=catrows('overall',0,nw)
                        for seg,(a,b) in enumerate(zip(bounds[:-1],bounds[1:]),1):
                            base=wc[2,a:b].sum(axis=(0,1)); ct=wc[mi,a:b].sum(axis=(0,1)) if available[mi] else None
                            temporal.append(dict(**k,method=method,segment=seg,status=statuses[mi],threshold=thresholds[s][mi],
                                rank_start=100000+int(a)*10000,rank_end_exclusive=100000+int(b)*10000,windows=int(b-a),
                                **selective(ct,(base[:2].sum(),base[2:].sum()))))
                            tcats+=catrows(seg,a,b)
                        if mi<2:
                            if available[mi]:
                                block=np.add.reduceat(wc[mi].sum(axis=1),np.arange(0,nw,10)); br.append(bootrates(mult@block,bootbg))
                            else: br.append({v:np.full(2000,np.nan) for v in BMET})
                            calrow=next(x for x in cal if all(x[v]==k[v] for v in k) and x['method']==method)
                            for metric in ['coverage','selective_risk','effective_attack_detection','wrong_accepted_attack_fraction','attack_rejection']:
                                cv=None if calrow[metric]=='NA' else float(calrow[metric]); ev=vals[metric]
                                gaps.append(dict(**k,method=method,selection_source='development R_select' if mi==0 else 'target400',
                                    comparison_source='same target400 diagnostic versus evaluation',metric=metric,calibration_value=cv,evaluation_value=ev,
                                    evaluation_minus_calibration=ev-cv if ev is not None and cv is not None else None))
                    saved={}
                    for metric in BMET:
                        delta=br[1][metric]-br[0][metric]; finite=delta[np.isfinite(delta)]
                        ci=np.percentile(finite,[2.5,97.5],method='linear') if len(finite) else [None,None]
                        a,b=points[0][metric],points[1][metric]
                        pairs.append(dict(**k,contrast='C2-select minus C0-select',metric=metric,C0=a,C2=b,
                            difference=b-a if a is not None and b is not None else None,ci_low=ci[0],ci_high=ci[1],
                            valid_replicates=len(finite),total_replicates=2000,C0_coverage=points[0]['coverage'],C2_coverage=points[1]['coverage'],
                            C2_status=statuses[1],interval='95% percentile linear; finite paired replicates; not equal coverage'))
                        saved.update({f'C0_{metric}':br[0][metric],f'C2_{metric}':br[1][metric],f'delta_{metric}':delta})
                    np.savez_compressed(OUT/f'bootstrap_replicates/{stem}.npz',**saved)
                    checks.append(dict(**k,background_matches_C1_C0=True,classification_threshold=.5,
                        background_all_accepted=True,unsupported_is_not_full_reject=not available[1],status_C2=statuses[1]))
                    log(f'{stem}: evaluation complete; C0 accepted={points[0]["accepted_rows"]}, C2 accepted={points[1]["accepted_rows"]}.')
    save(OUT/'selective_metrics.csv',allmetrics); save(OUT/'attack_category_metrics.csv',cats)
    save(OUT/'temporal_metrics.csv',temporal); save(OUT/'temporal_attack_category_metrics.csv',tcats)
    save(OUT/'paired_comparisons.csv',pairs); save(OUT/'calibration_evaluation_gap.csv',gaps)
    write(OUT/'evaluation_validation.json',checks)
    check_frozen(); write(OUT/'evaluation_complete.json',dict(completed_at_utc=now(),unique_rows=64260000,windows=6426,C0_configs=36,C2_supported_configs=24))
    log('Evaluation complete, frozen thresholds unchanged.')

if __name__=='__main__': main()
