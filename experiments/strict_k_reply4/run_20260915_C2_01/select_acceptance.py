from common import *
import cupy as cp
import gc
engine=score_engine()

def scores(p,fit):
    u=engine.entropy(p)
    assert bool(cp.isfinite(u).all()) and bool(((u>=0)&(u<=1)).all())
    return {'one_minus_u':1-u,'fu':engine.predict(u,fit)}

def kscore(d,m,o,s): return dict(**key(d,m,o),score=s)

def main():
    assert not (OUT/'threshold_freeze.json').exists()
    protocol=read(OUT/'protocol.json')
    for p,h in read(OUT/'protocol_freeze.json')['files'].items(): assert sha(p)==h
    definition=read(OUT/'score_definition_and_assets.json')
    assert sha(B2/'run_b2.py')==definition['B2_code_sha256']
    for a in definition['assets']: assert sha(a['path'])==a['sha256']
    (OUT/'C0_candidate_audit').mkdir(exist_ok=True)
    endpoint=cp.asarray([0.,.5,1.],dtype=cp.float64)
    assert np.array_equal(cp.asnumpy(engine.entropy(endpoint)),[0,1,0])
    sample=rows(PRE/'calibration_sample_manifest.csv'); devscope=rows(PRE/'development_R_select_scope.csv')
    c2rows=[]; c2threshold=[]; c0threshold=[]; summaries=[]; devmetrics=[]; calmetrics=[]; statusrows=[]
    for d,(fold,end,nw,nb,na) in DOMAINS.items():
        sr=[r for r in sample if r['domain']==d]; rank=np.array([int(r['availability_rank']) for r in sr])
        y=np.array([int(r['binary_label']) for r in sr],dtype=np.int64)
        devs=[r for r in devscope if r['fold']==fold]
        for m in MODELS:
            for o,field in FIELDS.items():
                fit=read(B2/f'models/{fold}__{m}__{o}.json')
                p=cp.asarray(cache(d,m)[field][rank]); ss=scores(p,fit); pred=cp.asnumpy(p)>=.5
                bins=2*y+pred; totals=(nb,na)
                chosen_c2={}
                for s in SCORES:
                    k=kscore(d,m,o,s); cpu=cp.asnumpy(ss[s]); supported=min(nb,na)>=20
                    selected=None; t=None; status=UNSUPPORTED
                    if supported:
                        levels=np.unique(np.r_[cpu,ACCEPT_ALL,REJECT_ALL]); candidates=[]
                        assert np.all((cpu>=0)&(cpu<=1))
                        for a in levels:
                            mask=cpu>=np.float64(a); c=np.bincount(bins[mask],minlength=4); v=selective(c,totals)
                            candidates.append(dict(**k,threshold=float(a),threshold_hex=float(a).hex(),**v,
                                feasible=bool(v['accepted_rows']>=40 and 20*v['accepted_errors']<=v['accepted_rows']),selected=False))
                        feasible=[r for r in candidates if r['feasible']]
                        if feasible:
                            # Equal accepted count implies equal risk denominator; exact integer tie comparison.
                            selected=min(feasible,key=lambda r:(-r['accepted_rows'],r['accepted_errors'],-r['threshold']))
                            t=selected['threshold']; status='SELECTED'
                        else:
                            t=REJECT_ALL; status=NOFEAS; selected=next(r for r in candidates if r['threshold']==t)
                        selected['selected']=True; c2rows+=candidates
                        assert sum(cpu>=ACCEPT_ALL)==400 and sum(cpu>=REJECT_ALL)==0
                        assert np.array_equal(cpu>=t,cpu>=float(repr(t)))
                        chosen_c2[s]=np.bincount(bins[cpu>=t],minlength=4)
                    else: chosen_c2[s]=None
                    c2threshold.append(dict(**k,status=status,threshold=t,threshold_hex=float(t).hex() if t is not None else None,
                        calibration_rows=400,calibration_benign=nb,calibration_attack=na,classification_threshold=.5,
                        source='fixed target 400',score_sha256=hashlib.sha256(cpu.tobytes()).hexdigest()))
                    calmetrics.append(dict(**k,method='C2-select',source='target_calibration_400',status=status,threshold=t,
                        **selective(chosen_c2[s],totals)))
                del ss,p
                # Only development R_select labels are accessed here, never this fold's target evaluation labels.
                devdata=[]
                for r in devs:
                    dd=r['domain']; a,b=int(r['rank_start']),int(r['rank_end_exclusive'])
                    yp=cp.asarray(np.load(ROOT/f'outputs/feature_store_v2/{dd}_label.npy',mmap_mode='r')[a:b],dtype=cp.int64)
                    pp=cp.asarray(cache(dd,m)[field][a:b]); score=scores(pp,fit)
                    assert bool(cp.isfinite(pp).all()) and bool(((pp>=0)&(pp<=1)).all())
                    code=yp*2+(pp>=.5)
                    # Numerical partition check: direct frozen formulas on a smaller batch must preserve scores.
                    small=scores(pp[:1024],fit)
                    assert all(bool(cp.array_equal(small[s],score[s][:1024])) for s in SCORES)
                    devdata.append(dict(domain=dd,N=b-a,totals=(int((yp==0).sum()),int((yp==1).sum())),score=score,code=code))
                    del yp,pp,small
                n1,n2=[r['N'] for r in devdata]; product=n1*n2
                assert 40*product<np.iinfo(np.int64).max
                for s in SCORES:
                    k=kscore(d,m,o,s); stem=f'{fold}__{m}__{o}__{s}'
                    allscore=cp.concatenate([r['score'][s] for r in devdata])
                    allcode=cp.concatenate([r['code']+i*4 for i,r in enumerate(devdata)])
                    order=cp.argsort(-allscore); sortedscore=allscore[order]; code=allcode[order]
                    ends=cp.r_[cp.flatnonzero(sortedscore[:-1]!=sortedscore[1:]),len(sortedscore)-1]
                    levels=cp.asnumpy(sortedscore[ends]); g=len(levels)
                    cumulative=np.empty((g,8),dtype=np.int64)
                    for i in range(8): cumulative[:,i]=cp.asnumpy(cp.cumsum(code==i,dtype=cp.int64)[ends])
                    assert (np.diff(levels)<0).all() and np.all((levels>=0)&(levels<=1))
                    assert cumulative[-1].sum()==n1+n2
                    A1=cumulative[:,:4].sum(axis=1); A2=cumulative[:,4:].sum(axis=1)
                    E1=cumulative[:,1]+cumulative[:,2]; E2=cumulative[:,5]+cumulative[:,6]
                    ca=A1*n2+A2*n1; ce=E1*n2+E2*n1
                    feasible=(5*ca>=product)&(20*ce<=ca)
                    ids=np.flatnonzero(feasible); idx=int(ids[-1]) if len(ids) else None
                    t=float(levels[idx]) if idx is not None else REJECT_ALL
                    status='SELECTED' if idx is not None else NOFEAS
                    assert np.all(np.diff(ca)>0)
                    # Distinct groups strictly increase CA. Therefore the last feasible group uniquely
                    # maximizes coverage. The added full-accept boundary has identical counts to min score,
                    # and loses the higher-threshold tie break. Full reject cannot satisfy coverage.
                    chunks=[]
                    for a in range(0,g,65536):
                        b=min(g,a+65536); good=np.flatnonzero(feasible[a:b])+a
                        h=hashlib.sha256(levels[a:b].tobytes()+cumulative[a:b].tobytes()).hexdigest()
                        chunks.append(dict(group_start=a,group_end_exclusive=b,first_threshold=float(levels[a]),last_threshold=float(levels[b-1]),
                            feasible_count=len(good),last_feasible_group=int(good[-1]) if len(good) else None,
                            last_feasible_CA=int(ca[good[-1]]) if len(good) else None,sha256_float64_levels_then_int64_cumulative_counts=h))
                    save(OUT/f'C0_candidate_audit/{stem}_chunks.csv',chunks)
                    starts=np.flatnonzero(feasible & ~np.r_[False,feasible[:-1]])
                    stops=np.flatnonzero(feasible & ~np.r_[feasible[1:],False])
                    runs=[dict(first_group=int(a),last_group=int(b),highest_threshold=float(levels[a]),lowest_threshold=float(levels[b]),
                        first_CA=int(ca[a]),last_CA=int(ca[b]),first_CE=int(ce[a]),last_CE=int(ce[b])) for a,b in zip(starts,stops)]
                    point=cumulative[idx] if idx is not None else np.zeros(8,dtype=np.int64)
                    direct=[]
                    for i,r in enumerate(devdata):
                        mask=r['score'][s]>=np.float64(t)
                        count=cp.asnumpy(cp.bincount(r['code'][mask],minlength=4))
                        assert np.array_equal(count,point[i*4:(i+1)*4])
                        assert bool(cp.array_equal(mask,r['score'][s]>=float(repr(t))))
                        direct.append(count)
                        devmetrics.append(dict(**k,method='C0-select',source='development_R_select',development_domain=r['domain'],
                            aggregation='per_domain',threshold=t,status=status,**selective(count,r['totals'])))
                    selected_ca=int(point[:4].sum())*n2+int(point[4:].sum())*n1
                    selected_ce=int(point[1]+point[2])*n2+int(point[5]+point[6])*n1
                    audit=dict(**k,domain_order=[r['domain'] for r in devdata],N1=n1,N2=n2,unique_score_groups=g,candidate_count=g+2,
                        feasible_group_count=len(ids),full_accept_candidate_feasible=bool(feasible[-1]),feasible_runs=runs,
                        full_reject_feasible=False,selected_group=idx,threshold=t,status=status,selected_cumulative_confusion=point.tolist(),
                        selected_CA=selected_ca,selected_CE=selected_ce,coverage_denominator=2*product,
                        maximum_cross_product=40*product,int64_overflow_excluded=True,
                        full_group_coverage='every exact score group scanned, none sampled or quantized',
                        tie_evidence='strictly increasing weighted acceptance over nonempty groups; min-score threshold beats -1 duplicate',
                        selected_direct_masks_match=True,serialization_masks_match=True,chunk_audit=str(OUT/f'C0_candidate_audit/{stem}_chunks.csv'))
                    write(OUT/f'C0_candidate_audit/{stem}.json',audit)
                    summaries.append({x:audit[x] for x in ['domain','fold','model','output','score','N1','N2','unique_score_groups','candidate_count','feasible_group_count','threshold','status','selected_CA','selected_CE','coverage_denominator']})
                    c0threshold.append(dict(**k,status=status,threshold=t,threshold_hex=t.hex(),classification_threshold=.5,
                        source='equal-domain complete R_select',N1=n1,N2=n2,coverage=selected_ca/(2*product),
                        selective_risk=div(selected_ce,selected_ca)))
                    devmetrics.append(dict(**k,method='C0-select',source='development_R_select',development_domain='two_domains',
                        aggregation='equal_domain_coverage_and_risk; raw counts remain per-domain rows above',threshold=t,status=status,
                        rows=n1+n2,coverage=selected_ca/(2*product),selective_risk=div(selected_ce,selected_ca),
                        accepted_rows=int(point.sum()),accepted_errors=int(point[[1,2,5,6]].sum()),
                        weighted_accept_numerator=selected_ca,weighted_error_numerator=selected_ce,coverage_denominator=2*product))
                    # C0 on target calibration is a diagnostic only, never used for its threshold selection.
                    targetp=cp.asarray(cache(d,m)[field][rank]); targetscore=cp.asnumpy(scores(targetp,fit)[s])
                    c=np.bincount(bins[targetscore>=t],minlength=4)
                    calmetrics.append(dict(**k,method='C0-select',source='target_calibration_400_diagnostic_only',status=status,threshold=t,**selective(c,totals)))
                    ct=next(r for r in c2threshold if all(r[v]==k[v] for v in k))
                    statusrows.append(dict(**k,C0_status=status,C2_status=ct['status'],C0_threshold=t,C2_threshold=ct['threshold'],classification_threshold=.5))
                    log(f'{d}/{m}/{o}/{s}: C0 {g+2:,} candidates, {status}; C2 {ct["status"]}.')
                    del allscore,allcode,order,sortedscore,code,ends,cumulative,ca,ce,A1,A2,E1,E2,targetp
                del devdata; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    assert len(c2threshold)==len(c0threshold)==36
    save(OUT/'C2_acceptance_candidates.csv',c2rows); save(OUT/'C2_acceptance_thresholds.csv',c2threshold)
    save(OUT/'C0_select_thresholds.csv',c0threshold); save(OUT/'C0_select_candidate_summary.csv',summaries)
    save(OUT/'development_selection_metrics.csv',devmetrics); save(OUT/'target_calibration_metrics.csv',calmetrics)
    save(OUT/'configuration_status.csv',statusrows)
    frozen=[p for p in OUT.iterdir() if p.suffix=='.csv']+list((OUT/'C0_candidate_audit').iterdir())
    write(OUT/'threshold_freeze.json',dict(frozen_at_utc=now(),status='ALL_C0_C2_THRESHOLDS_FROZEN_BEFORE_THIS_RUN_TARGET_EVALUATION',
        files={str(p):sha(p) for p in frozen},target_evaluation_labels_read=False,
        note='development R_select labels have cross-fold historical target roles; no own-fold target performance used to select'))
    log('All 36 C0 and 24 C2 decisions + 12 unsupported records frozen. Stop selection phase.')

if __name__=='__main__': main()
