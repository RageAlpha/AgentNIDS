"""Fixed 1-entropy selection, no fitting, no threshold selection, no per-flow outputs."""
from pathlib import Path
import csv
import json
import sys
import gc
import time
from datetime import datetime, timezone
import numpy as np
import cupy as cp
from cupyx.scipy.special import xlogy
import pyarrow.parquet as pq

ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
A=ROOT/'outputs/post_p7_diagnostics/stage_A/run_20260914_A1'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from nids.evaluation.round1_metrics import ranking_groups, grouped_aurc, harmonic_numbers
from nids.data.partitions import PreparedData
from nids.evaluation.outer_data import checked_windows

PCTS=[1,5,10,20,40,60,80,100]
CHUNK=262144
def read(path): return json.loads(Path(path).read_text())
def csvread(path): return list(csv.DictReader(Path(path).open()))
def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        w.writerows([{k:('NA' if v is None else v) for k,v in r.items()} for r in rows])
def ratio(n,d): return n/d if d else None
def entropy(p): return -(xlogy(p,p)+xlogy(1-p,1-p))/cp.log(2.)

def evaluate(domain,m,output,scope,y,code,book,cat_totals,binary_by_code,stagea):
    n=len(y);benign=int((y==0).sum());attack=int(y.sum())
    a,b=int(scope['rank_start']),int(scope['rank_end_exclusive'])
    cachepath=ROOT/f'outputs/prediction_cache_v2/{domain}/{m}.npy'
    cache=np.load(cachepath,mmap_mode='r')
    manifest=read(cachepath.with_name(m+'_manifest.json'))
    train=read(ROOT/f'outputs/source_v2/models/{m}/training_manifest.json')
    assert manifest['model_id']==m and manifest['domain']==domain
    assert manifest['rows']==len(cache)
    assert manifest['feature_indices']==train['feature_indices']
    field='raw_probability' if output=='Raw' else 'probability'
    score=cp.empty(n,dtype=cp.float64);pred=cp.empty(n,dtype=cp.bool_)
    pmin,pmax,umin,umax=1.,0.,1.,0.
    endpoints=[0,0];cached_u_error=0.
    for lo in range(0,n,CHUNK):
        hi=min(n,lo+CHUNK)
        p=cp.asarray(cache[field][a+lo:a+hi]);u=entropy(p)
        assert bool(cp.isfinite(p).all() & cp.isfinite(u).all())
        assert bool(((p>=0)&(p<=1)).all())
        assert bool(((u>=-2e-15)&(u<=1+2e-15)).all())
        assert bool((u[(p==0)|(p==1)]==0).all())
        endpoints[0]+=int((p==0).sum());endpoints[1]+=int((p==1).sum())
        pmin=min(pmin,float(p.min()));pmax=max(pmax,float(p.max()))
        umin=min(umin,float(u.min()));umax=max(umax,float(u.max()))
        score[lo:hi]=1-u;pred[lo:hi]=p>=.5
        if output=='Platt':
            cached_u_error=max(cached_u_error,float(cp.max(cp.abs(u-cp.asarray(cache['u'][a+lo:a+hi])))))
            assert bool((pred[lo:hi]==cp.asarray(cache['predicted_class'][a+lo:a+hi])).all())
    # Score is the only sort key; truth is aggregated after score-only groups exist.
    order,starts,count,correct=ranking_groups(score,pred==y)
    errors=(count-correct).astype(cp.int64)
    cum=cp.cumsum(count,dtype=cp.int64);ce=cp.cumsum(errors,dtype=cp.int64)
    group_scores=score[order[starts]]
    assert bool((group_scores[:-1]>group_scores[1:]).all())
    assert int(cum[-1])==n
    ca=cp.cumsum(cp.add.reduceat(y[order].astype(cp.int64),starts),dtype=cp.int64)
    ct=cp.cumsum(cp.add.reduceat((y[order]&pred[order]).astype(cp.int64),starts),dtype=cp.int64)
    aurc=float(grouped_aurc(count,errors,harmonic_numbers(n)))
    assert 0<=aurc<=1
    full_error=int(ce[-1]);full_fn=int(ca[-1]-ct[-1]);full_tp=int(ct[-1])
    full_fp=full_error-full_fn;full_tn=benign-full_fp
    assert [full_tn,full_fp,full_fn,full_tp]==[int(stagea[k]) for k in ['TN','FP','FN','TP']]

    # Compact display grid. Expected curves permit fractional membership only as a
    # label-independent expectation within a tie; no such actions are materialized.
    grid_k=cp.maximum(cp.arange(1,1001,dtype=cp.int64)*n//1000,1)
    g=cp.searchsorted(cum,grid_k,side='left')
    before=cum-count;ebefore=ce-errors
    expected_error=ebefore[g]+(grid_k-before[g])*errors[g]/count[g]
    idx=cp.searchsorted(cum,grid_k,side='right')-1
    available=idx>=0;safe=cp.maximum(idx,0)
    det_count=cp.where(available,cum[safe],0)
    det_error=cp.where(available,ce[safe],0)
    det_risk=cp.where(available,det_error/cp.maximum(det_count,1),cp.nan)
    np.savez_compressed(OUT/'risk_coverage_curves'/f'{domain}__{m}__{output}.npz',
        requested_coverage=np.arange(1,1001)/1000,
        expected_prefix_count=cp.asnumpy(grid_k),expected_coverage=cp.asnumpy(grid_k/n),
        expected_error=cp.asnumpy(expected_error),expected_risk=cp.asnumpy(expected_error/grid_k),
        deterministic_count=cp.asnumpy(det_count),deterministic_error=cp.asnumpy(det_error),
        deterministic_coverage=cp.asnumpy(det_count/n),deterministic_risk=cp.asnumpy(det_risk))
    key=dict(domain=domain,model=m,feature_view=train['view'],role='primary_candidate' if train['view']=='All49' else 'split_view_control',output=output)
    ops=[];cats=[];common=[]
    previous=cp.zeros(n,dtype=cp.bool_);previous_cut=float('inf')
    encoded=code.astype(cp.int32)*2+pred.astype(cp.int32)
    for pct in PCTS:
        cap=n*pct//100
        j=int(cp.searchsorted(cum,cap,side='right'))-1
        cutoff=float(group_scores[j]) if j>=0 else None
        accept=score>=cutoff if j>=0 else cp.zeros(n,dtype=cp.bool_)
        accepted=int(accept.sum())
        assert accepted==(int(cum[j]) if j>=0 else 0) and accepted<=cap
        assert j==len(cum)-1 or int(cum[j+1])>cap
        assert not bool((previous & ~accept).any())
        assert (cutoff if cutoff is not None else float('inf'))<=previous_cut
        previous=accept;previous_cut=cutoff if cutoff is not None else float('inf')
        # One global acceptance set, then class/category counts, never per-class ranking.
        tn=int((accept & (y==0) & ~pred).sum());fp=int((accept & (y==0) & pred).sum())
        fn=int((accept & (y==1) & ~pred).sum());tp=int((accept & (y==1) & pred).sum())
        assert tn+fp+fn+tp==accepted
        assert fp+fn==(int(ce[j]) if j>=0 else 0)
        if pct==100:
            assert [tn,fp,fn,tp]==[int(stagea[k]) for k in ['TN','FP','FN','TP']]
        op=dict(**key,requested_coverage=pct/100,capacity_count=cap,accepted_count=accepted,actual_coverage=accepted/n,
            score_cutoff=cutoff,score_cutoff_rule='s >= cutoff; whole exact-score groups',
            accepted_errors=fp+fn,selective_risk=ratio(fp+fn,accepted),
            TN=tn,FP=fp,FN=fn,TP=tp,benign_total=benign,benign_accepted=tn+fp,benign_coverage=(tn+fp)/benign,
            accepted_benign_fpr=ratio(fp,tn+fp),attack_total=attack,attack_accepted=fn+tp,attack_coverage=(fn+tp)/attack,
            accepted_attack_fnr=ratio(fn,fn+tp),correct_accepted_attack_count=tp,effective_attack_detection=tp/attack,
            wrong_accepted_attack_count=fn,wrong_accepted_attack_fraction=fn/attack,
            attack_abstained=attack-fn-tp,attack_abstention_fraction=(attack-fn-tp)/attack,
            attack_fraction_in_accepted=ratio(fn+tp,accepted),attack_fraction_in_population=attack/n,
            undefined_reason='no whole-score-group prefix within capacity' if accepted==0 else None)
        ops.append(op)
        bins=cp.asnumpy(cp.bincount(encoded,weights=accept.astype(cp.int64),minlength=2*len(book))).astype(np.int64).reshape(-1,2)
        sumtp=sumfn=0
        for k,name in book.items():
            c=int(k);total=int(cat_totals[c])
            if binary_by_code[c]!=1: continue
            wrong,right=map(int,bins[c]);acc=wrong+right;abstained=total-acc
            assert 0<=acc<=total
            sumtp+=right;sumfn+=wrong
            cats.append(dict(**key,requested_coverage=pct/100,actual_overall_coverage=accepted/n,
                original_attack_category=name,category_total=total,accepted_count=acc,category_coverage=ratio(acc,total),
                correct_accepted_attack_count=right,effective_category_detection=ratio(right,total),
                wrong_accepted_benign_count=wrong,wrong_accepted_fraction=ratio(wrong,total),
                abstained_count=abstained,abstained_fraction=ratio(abstained,total),
                accepted_category_fnr=ratio(wrong,acc),undefined_reason='zero category support' if total==0 else ('no accepted category records' if acc==0 else None)))
        assert sumtp==tp and sumfn==fn
        gi=int(cp.searchsorted(cum,cap,side='left'))
        ee=float(ebefore[gi]+(cap-before[gi])*errors[gi]/count[gi])
        common.append(dict(**key,requested_coverage=pct/100,prefix_count=cap,common_coverage=cap/n,
            expected_risk=ee/cap,interpretation='expected uniform ordering within exact tie; not deterministic operating point',
            actual_whole_group_coverage=accepted/n,actual_whole_group_risk=ratio(fp+fn,accepted)))

    # Enumerate every connected run of attainable group endpoints meeting the
    # descriptive 5% / 10% criterion. No cutoff or winning model is selected.
    risk=ce/cum
    qualifies=(cum*10>=n)&(ce*20<=cum)
    runstarts=cp.flatnonzero(qualifies & cp.concatenate((cp.asarray([True]),~qualifies[:-1])))
    runends=cp.flatnonzero(qualifies & cp.concatenate((~qualifies[1:],cp.asarray([True]))))
    regions=[]
    for z,(lo,hi) in enumerate(zip(cp.asnumpy(runstarts),cp.asnumpy(runends))):
        r=dict(**key,region=z+1,attainable_endpoints=int(hi-lo+1),minimum_risk=float(risk[lo:hi+1].min()),
            maximum_risk=float(risk[lo:hi+1].max()),diagnostic_only=True)
        for label,j in [('start',int(lo)),('end',int(hi))]:
            nc,na,nt,ne=map(int,[cum[j],ca[j],ct[j],ce[j]])
            nf=na-nt;nb=nc-na;nfp=ne-nf
            r.update({f'{label}_coverage':nc/n,f'{label}_risk':ne/nc,
                f'{label}_attack_coverage':na/attack,f'{label}_effective_attack_detection':nt/attack,
                f'{label}_wrong_accepted_attack_fraction':nf/attack,
                f'{label}_benign_coverage':nb/benign,f'{label}_accepted_benign_fpr':ratio(nfp,nb),
                f'{label}_attack_fraction_in_accepted':na/nc})
        regions.append(r)
    historical=None
    if domain!='UNSW' and m in ['rf_Traffic','xgb_Protocol','mlp_Behavior'] and output=='Platt':
        fold=scope['fold']
        historical=read(ROOT/f'outputs/outer_v2/P7/{fold}/round1/metrics.json')['CRE'][m]['all']['one_minus_u']['AURC']
        assert abs(aurc-historical)<1e-12
    overall=dict(**key,rows=n,benign=benign,attack=attack,AURC=aurc,full_accept_error_count=full_error,
        full_accept_error_rate=full_error/n,AURC_minus_full_error=aurc-full_error/n,
        exact_score_groups=len(count),largest_tie_group=int(count.max()),top_tie_group_count=int(count[0]),
        minimum_nonzero_coverage=int(count[0])/n,attainable_5pct_risk_10pct_coverage_regions=len(regions),
        legacy_P7_one_minus_u_AURC=historical,legacy_AURC_difference=aurc-historical if historical is not None else None)
    audit=dict(**key,probability_min=pmin,probability_max=pmax,entropy_min=umin,entropy_max=umax,
        score_min=float(score.min()),score_max=float(score.max()),p_zero_count=endpoints[0],p_one_count=endpoints[1],
        endpoints_entropy_zero=True,extra_probability_clipping=False,Platt_cached_entropy_max_error=cached_u_error if output=='Platt' else None,
        classification_matches_A=True,accepted_count_balance=True,accepted_error_balance=True,category_attack_balance=True,
        nested_acceptance=True,whole_ties_maximal_prefix=True,label_independent_selection=True)
    return overall,ops,cats,common,regions,audit

def main():
    if (OUT/'overall_selective_metrics.csv').exists():
        raise RuntimeError('Run directory already contains results: create a new run directory.')
    started=time.time()
    cfg=read(ROOT/'configs/execution_v2.json')
    scopes=csvread(A/'evaluation_scope.csv');stagea=csvread(A/'detector_raw_platt_metrics.csv')
    amap={(r['domain'],r['model'],r['output']):r for r in stagea}
    acats=csvread(A/'attack_category_recall.csv')
    prep=PreparedData(ROOT/'outputs/round1_data_prep_v1')
    save('evaluation_scope.csv',scopes)
    overall=[];ops=[];cats=[];common=[];regions=[];checks=[]
    endpoint=cp.asnumpy(entropy(cp.asarray([0.,.5,1.],dtype=cp.float64)))
    assert np.array_equal(endpoint,[0.,1.,0.])
    for scope in scopes:
        domain=scope['domain'];a=int(scope['rank_start']);b=int(scope['rank_end_exclusive'])
        if domain=='UNSW':
            part=prep.partition(domain,'source','S_source_test')
            assert (a,b)==(part.rank_start,part.rank_end_exclusive)
        else:
            part,windows=checked_windows(prep,domain,pq.read_table(ROOT/f'outputs/shift_v2/10k/{domain}_windows.parquet').to_pylist())
            assert (a,b)==(part.rank_start,windows[-1]['rank_end_exclusive'])
            assert len(windows)==int(scope['complete_windows'])
            assert part.rank_end_exclusive-b==int(scope['excluded_tail'])
            # Reuse A's full identity audit; verify the same immutable P7 ledger length.
            assert pq.ParquetFile(ROOT/f'outputs/outer_v2/P7/{scope["fold"]}/identities.parquet').metadata.num_rows==b-a
        store=ROOT/'outputs/feature_store_v2'
        y=cp.asarray(np.load(store/f'{domain}_label.npy',mmap_mode='r')[a:b])
        code=cp.asarray(np.load(store/f'{domain}_attack_code.npy',mmap_mode='r')[a:b])
        book=read(store/f'{domain}_manifest.json')['attack_codebook']
        cat_totals=cp.asnumpy(cp.bincount(code.astype(cp.int32),minlength=len(book)))
        binary_by_code={}
        for k,name in book.items():
            ar=next(r for r in acats if r['domain']==domain and r['original_category']==name)
            binary_by_code[int(k)]=int(ar['binary_label'])
            assert int(cat_totals[int(k)])==int(ar['rows'])
            assert bool((y[code==int(k)]==int(ar['binary_label'])).all())
        assert int(y.sum())==int(scope['attack']) and len(y)==int(scope['rows'])
        assert sum(int(cat_totals[k]) for k,v in binary_by_code.items() if v==1)==int(y.sum())
        for m in cfg['models']['schedule']:
            for output in ['Raw','Platt']:
                print(f'{domain}/{m}/{output}: {b-a:,} records',flush=True)
                result=evaluate(domain,m,output,scope,y,code,book,cat_totals,binary_by_code,amap[domain,m,output])
                ov,op,cat,co,reg,ch=result
                overall.append(ov);ops.extend(op);cats.extend(cat);common.extend(co);regions.extend(reg);checks.append(ch)
                print(f'  AURC={ov["AURC"]:.9f}; full error={ov["full_accept_error_rate"]:.9f}; top tie={ov["top_tie_group_count"]:,}',flush=True)
                save('overall_selective_metrics.csv',overall);save('operating_points.csv',ops)
                save('attack_category_selective_metrics.csv',cats);save('common_coverage_expected_risks.csv',common)
                if regions: save('diagnostic_regions.csv',regions)
                gc.collect();cp.get_default_memory_pool().free_all_blocks()
        del y,code
    assert len(overall)==48 and len(ops)==384
    validation=dict(status='PASS_B1_COMPUTATION',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        stage_A_directory=str(A),conditions=len(overall),operating_points=len(ops),attack_category_rows=len(cats),
        source_rows=235424,outer_rows=64560000,outer_windows=6456,excluded_outer_tail=9597,
        endpoint_entropy=endpoint.tolist(),score_definition='1-normalized_binary_entropy; high score first; no extra clipping',
        AURC_implementation='nids.evaluation.round1_metrics.ranking_groups/grouped_aurc/harmonic_numbers',
        AURC_ties='expected risk over every nonempty prefix; uniform random order within exact score ties',
        operating_point_ties='entire exact-score groups; largest prefix <= floor(requested_coverage*N); no label or row-order tie split',
        full_coverage_stage_A_confusion_match=all(c['classification_matches_A'] for c in checks),
        window_scope_check=True,record_identity='same A rank intervals; checked_windows revalidated; A full P7 record-id audit reused',
        checks=checks,gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),cupy=cp.__version__,numpy=np.__version__,
        inference_performed=False,fitting_performed=False,classification_threshold=.5,bootstrap_performed=False,
        elapsed_seconds=time.time()-started)
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='checks'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
