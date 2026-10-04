"""Phase 1: freeze protocol; select on saved 400 labels; commit all thresholds.
Does not open any evaluation label array before the threshold freeze.
"""
from common import *
import copy
import pyarrow.parquet as pq
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import checked_windows

def main():
    assert not (OUT/'protocol.json').exists(), 'Never overwrite protocol or reselect thresholds'
    write(OUT/'upstream_inventory_before.json',inventory())
    assert sha(PLAN)==PLAN_SHA
    PLAN.read_text()
    draft=read(PRE/'protocol_draft.json')
    assert draft['source_plan_sha256']==PLAN_SHA
    commitment=read(PRE/'rank_commitment.json')
    assert all(sha(PRE/n)==h for n,h in commitment['sha256'].items())
    assert read(PRE/'validation.json')['status']=='PASS_C_PREFLIGHT_COMPLETE'
    assert read(PRE/'deliverable_validation.json')['status']=='PASS'
    sample=rows(PRE/'calibration_sample_manifest.csv')
    assert len(sample)==1200 and len({r['record_id'] for r in sample})==1200
    prepared=PreparedData(ROOT/'outputs/round1_data_prep_v1')
    references=[]; window_refs=[]; segment_refs=[]; identity_audits=[]
    historical=read(PRE/'upstream_inventory_after.json')
    for d,(fold,end,nw,nb,na) in DOMAINS.items():
        part,ws=checked_windows(prepared,d,pq.read_table(ROOT/f'outputs/shift_v2/10k/{d}_windows.parquet').to_pylist())
        assert part.rank_start==0 and ws[9]['rank_end_exclusive']==100000 and ws[-1]['rank_end_exclusive']==end
        assert len(ws[10:])==nw and SEGMENTS[d]==[nw//4+(i<nw%4) for i in range(4)]
        for p in [prepared.root/f'{d}_sorted_original_rows.npy',prepared.root/f'{d}_sorted_end_ms.npy',ROOT/f'outputs/outer_v2/P7/{fold}/identities.parquet']:
            s=p.stat(); assert (s.st_size,s.st_mtime_ns)==(historical[str(p)]['bytes'],historical[str(p)]['mtime_ns'])
        rec=[r for r in sample if r['domain']==d]; ranks=np.array([int(r['availability_rank']) for r in rec])
        assert np.array_equal(ranks,np.load(PRE/f'{d}_sorted_ranks.npy'))
        assert len(np.unique(ranks))==400 and ranks.min()>=0 and ranks.max()<100000
        order=prepared.sorted_rows(d); originals=np.array([int(r['original_row']) for r in rec])
        assert np.array_equal(order[ranks],originals)
        assert all(r['record_id']==f'{d}:{r["original_row"]}' for r in rec)
        assert np.array_equal(prepared.sorted_end_ms(d)[ranks],[int(r['availability_ms']) for r in rec])
        # Explicit current disjointness scan over identities only, no labels.
        overlap=sum(int(np.isin(order[a:min(end,a+1000000)],originals).sum()) for a in range(100000,end,1000000))
        assert overlap==0
        y=np.array([int(r['binary_label']) for r in rec],dtype=np.uint8)
        assert (int((y==0).sum()),int((y==1).sum()))==(nb,na)
        references.append(dict(domain=d,manifest_path=str(PRE/'calibration_sample_manifest.csv'),manifest_sha256=sha(PRE/'calibration_sample_manifest.csv'),
            sorted_ranks_path=str(PRE/f'{d}_sorted_ranks.npy'),sorted_ranks_sha256=sha(PRE/f'{d}_sorted_ranks.npy'),rows=400,benign=nb,attack=na))
        identity_audits.append(dict(domain=d,calibration_evaluation_intersection=overlap,
            proof='current evaluation original-row scan against saved 400 identities; original P7 ledger unchanged since full preflight audit'))
        for i,w in enumerate(ws[10:]):
            window_refs.append(dict(domain=d,evaluation_window_index=i,window_id=w['window_id'],rank_start=w['rank_start'],
                rank_end_exclusive=w['rank_end_exclusive'],first_available_ms=w['first_available_ms'],last_available_ms=w['last_available_ms']))
        start=0
        for i,count in enumerate(SEGMENTS[d]):
            segment_refs.append(dict(domain=d,segment=i+1,window_index_start=start,window_index_end_exclusive=start+count,
                windows=count,rank_start=100000+start*10000,rank_end_exclusive=100000+(start+count)*10000,rows=count*10000))
            start+=count
    save(OUT/'calibration_identity_references.csv',references)
    save(OUT/'evaluation_window_references.csv',window_refs)
    save(OUT/'temporal_scope.csv',segment_refs)
    save(OUT/'evaluation_scope.csv',rows(PRE/'evaluation_scope.csv'))
    write(OUT/'identity_validation.json',identity_audits)
    protocol=copy.deepcopy(draft)
    protocol['history_preflight']=dict(status=draft['status'],prohibited_this_run=protocol.pop('prohibited_this_run'),
        explanation='These prohibitions applied only to completed C-preflight, not this authorized C1 run.')
    protocol.update(version='strict-k-reply4-C1-v1.0',status='FROZEN_FOR_C1_CLASS_AND_C0_CLASS',frozen_at_utc=now(),
        source_plan_sha256=PLAN_SHA,source_plan_version='2026-09-14 design draft, exact SHA256 verified',
        preflight_source=str(PRE),preflight_review='user reviewed materials and explicitly authorized C1/C0-class on 2026-09-15',
        preflight_file_sha256={str(p):sha(p) for p in PRE.iterdir() if p.is_file()},
        current_scope=['C1 calibration threshold selection','C0-class','calibration and fixed evaluation classification metrics','paired block bootstrap'],
        current_prohibitions=['C2','C0-select','resampling calibration','extra target calibration labels','balancing classes','expanding startup','detector/Platt/f(u) retraining','conformal','coordination','open-set','Strict-K sensitivity'],
        bootstrap_execution=dict(replicates=2000,block_windows=10,window_size=10000,include_final_short_block=True,
            minimum_blocks=20,confidence_level=0.95,interval='percentile',percentiles=[2.5,97.5],quantile_method='linear',
            generator='numpy.random.Generator(numpy.random.PCG64(seed))',seeds={'ToN':20260915,'CIC':20260916,'BoT':20260917},
            samples='independently for each domain draw B block indices with replacement for each replicate; save fresh matrix on shifted evaluation scope',
            paired='same domain matrix for every model, output and C0/C1',backend='CPU NumPy int64 confusion aggregation and float64 ratios; same block/percentile convention as project',
            undefined='exclude undefined replicates for each metric, report effective replicate count; no imputing zero',
            intervals_scope='overall evaluation paired differences only; temporal/calibration are descriptive; conditional on fixed 400-label draw and already unblinded domain'),
        temporal_segments=SEGMENTS,implementation_dtype='float64 probabilities and thresholds; int64 counts',
        unsupported_CIC='12 C1 rows retained with null threshold/metrics/differences; C0 computed normally',
        timestamp_boundary='ToN/CIC ranks 99999 and 100000 share availability time; identities disjoint, no strict clock separation claimed')
    protocol['C1']['status']='AUTHORIZED_NOT_YET_SELECTED'
    protocol['C0_class']['status']='AUTHORIZED_NOT_YET_EVALUATED'
    protocol['sampling']['current_stage_action']='reuse saved ranks and manifest only; do not call calibration RNG'
    protocol['C1']['support_sufficient_configurations']=24
    protocol['C1']['max_calibration_FP']={'ToN':17,'BoT':2}
    sentinel=np.nextafter(np.float64(1),np.float64(np.inf))
    assert sentinel>1 and float(str(sentinel))>1 and not (np.array([1.],dtype=np.float64)>=sentinel).any()
    protocol['C1']['sentinel']=float(sentinel)
    assert protocol['C2']==draft['C2'] and protocol['C0_select']==draft['C0_select']
    write(OUT/'protocol.json',protocol)
    write(OUT/'protocol_freeze.json',dict(frozen_at_utc=now(),evaluation_labels_read=False,
        files={str(OUT/'protocol.json'):sha(OUT/'protocol.json'),str(PLAN):sha(PLAN)},
        source_draft_sha256=sha(PRE/'protocol_draft.json'),script_sha256={p.name:sha(p) for p in OUT.glob('*.py')}))
    log('Protocol frozen. Plan/rank commitments verified, evaluation identities disjoint. Selecting only on fixed 400 labels.')
    candidates=[]; thresholds=[]; statuses=[]; calibration=[]; audits=[]
    for d,(_,end,nw,nb,na) in DOMAINS.items():
        rec=[r for r in sample if r['domain']==d]
        ranks=np.array([int(r['availability_rank']) for r in rec])
        y=np.array([int(r['binary_label']) for r in rec],dtype=np.uint8)
        # Only gather these exact 400 labels, never slice the startup or evaluation.
        arr=np.load(ROOT/f'outputs/feature_store_v2/{d}_label.npy',mmap_mode='r')
        assert np.array_equal(arr[ranks],y); del arr
        for m in MODELS:
            cached=cache(d,m)
            for output,field in FIELDS.items():
                p=np.array(cached[field][ranks],dtype=np.float64)
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                k=key(d,m,output); base=confusion(y,p,.5)
                calibration.append(dict(**k,method='C0-class',split='calibration',status='COMPLETE',threshold=.5,rows=400,accepted_rows=400,benign=nb,attack=na,**metrics(base)))
                status=UNSUPPORTED if min(nb,na)<20 else 'SELECTED'
                best=None; count=0
                if status!=UNSUPPORTED:
                    ts=np.unique(np.concatenate([p,np.array([0.,.5,sentinel],dtype=np.float64)]))
                    assert np.array_equal(np.unique(p),ts[np.isin(ts,p)]) and 0 in ts and .5 in ts and sentinel in ts
                    # Exact sorted-prefix counts with left insertion implement p>=t, whole ties.
                    idx=np.argsort(p,kind='stable'); sortedp=p[idx]
                    prefix=np.r_[0,np.cumsum(y[idx],dtype=np.int64)]
                    local=[]
                    for t in ts:
                        cut=int(np.searchsorted(sortedp,t,side='left')); fn=int(prefix[cut]); tn=cut-fn
                        c=np.array([tn,nb-tn,fn,na-fn],dtype=np.int64)
                        assert np.array_equal(c,confusion(y,p,t))
                        local.append(dict(**k,threshold=float(t),threshold_hex=float(t).hex(),**metrics(c),
                            feasible=bool(20*c[1]<=nb),selected=False,is_t_half=bool(t==.5),is_sentinel=bool(t==sentinel)))
                    feasible=[r for r in local if r['feasible']]
                    best=min(feasible,key=lambda r:(-r['TP'],r['FP'],abs(r['threshold']-.5),-r['threshold']))
                    best['selected']=True
                    if best['TP']==0: status=DEGENERATE
                    candidates.extend(local); count=len(local)
                    assert len({r['TN']+r['FN'] for r in local})==len(np.unique(p))+1
                    assert (nb//20)=={'ToN':17,'BoT':2}[d]
                    audits.append(dict(**k,candidates=count,unique_probabilities=len(np.unique(p)),
                        all_realizable_partitions_covered=True,all_candidates_direct_count_verified=True,
                        whole_ties=True,sentinel_strictly_above_one_in_computation_dtype=True,selected_exact_lexicographic_optimum=True))
                threshold=best['threshold'] if best else None
                thresholds.append(dict(**k,status=status,threshold=threshold,threshold_hex=float(threshold).hex() if threshold is not None else None,
                    candidates=count,calibration_rows=400,calibration_benign=nb,calibration_attack=na,max_allowed_FP=nb//20,
                    calibration_FP=best['FP'] if best else None,calibration_TP=best['TP'] if best else None,
                    calibration_probability_sha256=hashlib.sha256(p.tobytes()).hexdigest(),sentinel=float(sentinel),dtype='float64'))
                statuses.append(dict(**k,C0_class_status='READY_FOR_EVALUATION',C1_class_status=status,
                    label_support_sufficient=status!=UNSUPPORTED,calibration_benign=nb,calibration_attack=na,
                    C1_threshold=threshold,C2_status='NOT_RUN',C0_select_status='NOT_RUN'))
                calibration.append(dict(**k,method='C1-class',split='calibration',status=status,threshold=threshold,rows=400,
                    accepted_rows=400 if best else None,benign=nb,attack=na,**metrics(confusion(y,p,threshold) if best else None)))
        log(f'{d}: calibration phase saved in memory; 12 configurations, support={nb}/{na}.')
    assert sum(r['threshold'] is not None for r in thresholds)==24
    save(OUT/'classification_candidates.csv',candidates)
    save(OUT/'classification_thresholds.csv',thresholds)
    save(OUT/'configuration_status.csv',statuses)
    save(OUT/'calibration_metrics_frozen.csv',calibration)
    write(OUT/'candidate_validation.json',audits)
    committed=['classification_candidates.csv','classification_thresholds.csv','configuration_status.csv','calibration_metrics_frozen.csv','candidate_validation.json']
    write(OUT/'threshold_freeze.json',dict(frozen_at_utc=now(),status='ALL_24_C1_THRESHOLDS_AND_12_UNSUPPORTED_RECORDS_FROZEN_BEFORE_EVALUATION_LABEL_ACCESS',
        evaluation_labels_read=False,calibration_labels_per_domain=400,thresholds_selected=24,unsupported=12,
        files={str(OUT/n):sha(OUT/n) for n in committed},protocol_sha256=sha(OUT/'protocol.json')))
    log(f'All thresholds frozen; degenerate configurations={sum(r["status"]==DEGENERATE for r in thresholds)}. No evaluation labels read.')

if __name__=='__main__': main()
