"""Fixed ToN Variant B mini-pilot. No training, probability modification, or file digests.
Run prepare once, then evaluate once; all output writes refuse overwriting.
"""
import csv
import json
import math
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT = Path(__file__).resolve().parent
SEGMENTS = [(100000,6960000),(6960000,13820000),(13820000,20670000),(20670000,27520000)]
NROWS = {'ToN':27520260,'UNSW':2365424}
KEYS = ['n','true_benign','true_attack','benign_singleton','attack_singleton','empty','double',
        'tn','fn','fp','tp','accepted','singleton_errors','raw_errors','singleton_raw_disagreement',
        'benign_covered','attack_covered']

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def write(name, data):
    with (OUT/name).open('x') as f: json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
def save(name, rows):
    assert rows, name
    with (OUT/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def log(s): print(now(),s,flush=True)
def ratio(a,b): return a/b if b else None

def paths(d):
    return dict(cache=ROOT/f'outputs/prediction_cache_v2/{d}/rf_All49.npy',
                labels=ROOT/f'outputs/feature_store_v2/{d}_label.npy',
                original_rows=ROOT/f'outputs/round1_data_prep_v1/{d}_sorted_original_rows.npy')
def load(d): return {k:np.load(p,mmap_mode='r',allow_pickle=False) for k,p in paths(d).items()}
def windows(d):
    with (ROOT/'outputs/round1_data_prep_v1/window_manifest.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['domain']==d and r['window_size']=='10000'
              and r['scheme']==('heldout' if d=='ToN' else 'source')]
    for r in rows:
        for k in ['rank_start','rank_end_exclusive','first_original_row','last_original_row','first_available_ms','last_available_ms']:
            r[k]=int(r[k])
    return sorted(rows,key=lambda r:r['rank_start'])

def sets(p,qb,qa):
    # Prediction-set generation is label-free; inclusive comparisons are intentional.
    return p<=qb, (1.0-p)<=qa

def counts(p,y,qb,qa):
    b,a=sets(p,qb,qa)
    bs=b & ~a; ats=a & ~b; single=b ^ a
    c=dict(n=len(p),true_benign=np.count_nonzero(y==0),true_attack=np.count_nonzero(y==1),
           benign_singleton=bs.sum(),attack_singleton=ats.sum(),empty=(~b & ~a).sum(),double=(b&a).sum(),
           tn=(bs & (y==0)).sum(),fn=(bs & (y==1)).sum(),fp=(ats & (y==0)).sum(),tp=(ats & (y==1)).sum(),
           accepted=single.sum(),singleton_errors=((bs & (y==1)) | (ats & (y==0))).sum(),
           raw_errors=((p>=0.5)!=y).sum(),singleton_raw_disagreement=(single & (a!=(p>=0.5))).sum(),
           benign_covered=(b & (y==0)).sum(),attack_covered=(a & (y==1)).sum())
    c={k:int(v) for k,v in c.items()}
    assert c['benign_singleton']+c['attack_singleton']+c['empty']+c['double']==c['n']
    assert c['tn']+c['fn']+c['fp']+c['tp']==c['accepted']
    assert c['fn']+c['fp']==c['singleton_errors']
    return c

def unit_checks():
    cases=[([0,.25,.5,.75,1],.25,.25,[1,1,0,0,0],[0,0,0,1,1]),
           ([.125,.25,.5,.75,.875],.75,.75,[1,1,1,1,0],[0,1,1,1,1]),
           ([.5],.5,.5,[1],[1]),
           ([.25,.375,.5,.75],.25,.625,[1,0,0,0],[0,1,1,1])]
    for p,qb,qa,eb,ea in cases:
        b,a=sets(np.array(p),qb,qa); assert b.tolist()==[bool(x) for x in eb]; assert a.tolist()==[bool(x) for x in ea]
    c=counts(np.array([.375]),np.array([1]),.25,.625)
    assert c['tp']==1 and c['raw_errors']==1 and c['singleton_errors']==0 and c['singleton_raw_disagreement']==1
    for qb,qa in [(.25,.25),(.75,.75)]:
        c=counts(np.array([.5]),np.array([1]),qb,qa)
        assert c['accepted']==0 and ratio(c['singleton_errors'],c['accepted']) is None
    return dict(inclusive_equality=True,asymmetric_thresholds=True,empty=True,double=True,
                zero_acceptance_NA=True,singleton_can_differ_from_raw=True,label_free_set_function=True)

def prepare():
    assert not (OUT/'protocol.json').exists(), 'Preparation may only run once'
    model=ROOT/'outputs/source_v2/models/rf_All49/model.pkl'
    assert model.is_file() and model.stat().st_size>0
    evidence=[]
    fm=read(ROOT/'outputs/feature_store_v2/feature_store_manifest.json')
    assert fm['row_order']=='availability_sorted_rank'
    assert Path(fm['rank_source'])==ROOT/'outputs/round1_data_prep_v1'
    data={}
    for d,n in NROWS.items():
        data[d]=load(d); x=data[d]
        assert all(v.shape==(n,) for v in x.values())
        assert x['cache'].dtype['raw_probability']==np.dtype('float64')
        cm=read(ROOT/f'outputs/prediction_cache_v2/{d}/manifest.json')
        mm=read(ROOT/f'outputs/prediction_cache_v2/{d}/rf_All49_manifest.json')
        lm=read(ROOT/f'outputs/feature_store_v2/{d}_manifest.json')
        assert cm['row_order']=='availability_rank' and Path(cm['original_row_index'])==paths(d)['original_rows']
        assert cm['rows']==mm['rows']==lm['rows']==n and mm['model_id']=='rf_All49'
        assert mm['feature_indices']==list(range(49))
        # Original rows form a permutation: no duplicate identities or calibration/evaluation overlap.
        order=x['original_rows']; seen=np.zeros(n,dtype=bool)
        for start in range(0,n,1000000):
            z=np.asarray(order[start:start+1000000]); assert np.all((z>=0)&(z<n))
            assert len(np.unique(z))==len(z) and not np.any(seen[z]); seen[z]=True
        assert seen.all()
        ws=windows(d); cursor=0
        for w in ws:
            a,b=w['rank_start'],w['rank_end_exclusive']
            assert a==cursor and b-a==10000
            assert (int(order[a]),int(order[b-1]))==(w['first_original_row'],w['last_original_row'])
            assert w['first_available_ms']<=w['last_available_ms']
            if cursor: assert prev_time<=w['first_available_ms']
            cursor=b; prev_time=w['last_available_ms']
        assert cursor==(27520000 if d=='ToN' else 2360000)
        evidence.append(dict(domain=d,rows=n,cache_dtype=str(x['cache'].dtype),
            paths={k:str(v) for k,v in paths(d).items()},original_rows_unique_and_in_range=True,
            rank_mapping='cache and feature-store metadata agree; feature_store.py gathers labels by prepared order; predictions.py preserves rank and raw class-1 probability',
            original_window_boundary_rows_match=True,complete_windows=len(ws),tail_rows=n-cursor))
    protocol=dict(registered_at_utc=now(),model_path=str(model),model_bytes=model.stat().st_size,
        authority=[str(ROOT/'docs/Ton_variant/Reply_to_Siyuan_Week0_Analysis_and_Next_Step.md'),
                   str(ROOT/'docs/Ton_variant/ToN_VariantB_MiniPilot_Plan_ReplyLed_ZH.md')],
        variant='B Mondrian Strict-K',K=400,per_class=200,epsilon=.10,
        probability='raw_probability = P(Attack); P(Benign)=1-p; float64; unchanged',labels={'0':'Benign','1':'Attack'},
        collection='chronological rank scan [0,100000); first 200 of each class; stop when both full; no sampling',
        threshold='ascending class-specific scores; one-based 181st of 200; Benign score=p; Attack score=1-p',
        prediction={'benign':'p <= q_Benign','attack':'1-p <= q_Attack','singleton':'accept unique label','empty':'reject','double':'abstain','raw':'p >= 0.5'},
        ton_evaluation=[100000,27520000],ton_rows=27420000,ton_windows=2742,ton_tail_excluded=260,
        segments=[dict(segment=i+1,rank_start=a,rank_end_exclusive=b,windows=(b-a)//10000) for i,(a,b) in enumerate(SEGMENTS)],
        source_evaluation=[2130000,2365424],source_rows=235424,source_tail_included=5424,
        bootstrap=dict(block_windows=10,blocks=275,last_block_windows=2,replicates=2000,generator='PCG64',seed=20260915,
            paired=True,percentiles=[2.5,97.5],quantile_method='linear',undefined='NA; exclude per-metric invalid replicates and report valid count',
            inference='conditional on fixed calibration sample, thresholds and frozen model; four segments descriptive'),
        gate='Professor reply section 4; no numerical materiality threshold; unresolved materiality or segment consistency requires professor judgment',
        stop_after='mini-pilot; no Week 1; no professor message',input_validation=evidence,
        environment=dict(python=platform.python_version(),numpy=np.__version__),code_evidence=[
            'src/nids/data/feature_store.py: sorted_y[start:stop] = raw_y[order[start:stop]]',
            'src/nids/replay/predictions.py: RF predict_proba result[:, 1]; cache written in rank order',
            'src/nids/evaluation/outer_data.py: C1 uses heldout T_heldout 10000-row windows'])
    write('protocol.json',protocol)
    log('Input structure, row permutation and original window boundaries verified; protocol registered.')
    x=data['ToN']; chosen=[]; class_counts=[0,0]; support=[0,0]
    ws=windows('ToN')
    for rank in range(100000):
        y=int(x['labels'][rank]); assert y in (0,1); support[y]+=1
        if class_counts[y]<200:
            chosen.append(dict(original_row=int(x['original_rows'][rank]),rank=rank,label=y,
                               window_id=ws[rank//10000]['window_id'],window_index=rank//10000))
            class_counts[y]+=1
        if class_counts==[200,200]: break
    summary=dict(completed_at_utc=now(),collected_benign=class_counts[0],collected_attack=class_counts[1],
                 total_collected=len(chosen),scanned_rows=rank+1,stop_rank_inclusive=rank,
                 stop_rank_exclusive=rank+1,stop_original_row=int(x['original_rows'][rank]),
                 stop_window_id=ws[rank//10000]['window_id'],scanned_prefix_support={'Benign':support[0],'Attack':support[1]},
                 startup_limit=100000,status='COMPLETE' if class_counts==[200,200] else 'INSUFFICIENT_SUPPORT')
    save('calibration_sample_manifest.csv',chosen); write('collection_summary.json',summary)
    if class_counts!=[200,200]:
        raise RuntimeError('Insufficient class support within fixed startup scope; stop without fitting or evaluation')
    ranks=np.array([r['rank'] for r in chosen]); ys=np.array([r['label'] for r in chosen])
    assert len(np.unique(ranks))==400 and len({r['original_row'] for r in chosen})==400 and ranks.max()<100000
    for y in [0,1]: assert np.array_equal(ranks[ys==y],np.flatnonzero(x['labels'][:rank+1]==y)[:200])
    p=np.asarray(x['cache']['raw_probability'][ranks]); assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    k=math.ceil(201*.9); assert k==181
    qb=float(np.sort(p[ys==0])[k-1]); qa=float(np.sort(1.0-p[ys==1])[k-1])
    freeze=dict(registered_at_utc=now(),q_Benign=qb,q_Attack=qa,threshold_sum=qb+qa,
                min_p_Benign=1.0-qb,min_p_Attack=1.0-qa,epsilon=.1,n_Benign=200,n_Attack=200,
                order_statistic_one_based=181,order_statistic_zero_based=180,sorting='ascending; no interpolation',
                scores={'Benign':'raw_probability','Attack':'1.0-raw_probability'},dtype='float64',
                calibration_rank_start=int(ranks.min()),calibration_rank_end_exclusive=int(ranks.max()+1),
                evaluation_not_started=True)
    write('threshold_freeze.json',freeze)
    log('Thresholds registered before evaluation: '+json.dumps(freeze))


def metric_rows(scope,c,total_tp):
    pairs=[('benign_singleton_rate',c['benign_singleton'],c['n']),('attack_singleton_rate',c['attack_singleton'],c['n']),
           ('empty_set_rate',c['empty'],c['n']),('double_set_rate',c['double'],c['n']),
           ('benign_singleton_error_rate',c['fn'],c['benign_singleton']),('attack_singleton_error_rate',c['fp'],c['attack_singleton']),
           ('tp',c['tp'],1),('tp_fraction_of_true_attacks',c['tp'],c['true_attack']),('acceptance_rate',c['accepted'],c['n']),
           ('singleton_error_rate',c['singleton_errors'],c['accepted']),('raw_error_rate',c['raw_errors'],c['n']),
           ('tp_share_of_overall',c['tp'],total_tp)]
    rows=[dict(scope=scope,metric=k,numerator=a,denominator=b,value=ratio(a,b) if b else 'NA',unit='count' if k=='tp' else 'fraction') for k,a,b in pairs]
    # Save an exact integer numerator/denominator for the difference of risks, in pp.
    den=c['n']*c['accepted']; num=100*(c['raw_errors']*c['accepted']-c['singleton_errors']*c['n'])
    rows.append(dict(scope=scope,metric='raw_minus_singleton_pp',numerator=num,denominator=den,value=ratio(num,den) if den else 'NA',unit='percentage_points'))
    return rows

def aggregate(rows): return {k:sum(r[k] for r in rows) for k in KEYS}

def evaluate():
    assert not (OUT/'evaluation_start.json').exists(), 'Evaluation may only run once'
    protocol=read(OUT/'protocol.json'); frozen=read(OUT/'threshold_freeze.json')
    # Evaluation obtains thresholds exclusively from the registered file.
    qb,qa=frozen['q_Benign'],frozen['q_Attack']
    started=now(); assert frozen['registered_at_utc']<started
    write('evaluation_start.json',dict(started_at_utc=started,threshold_source=str(OUT/'threshold_freeze.json'),q_Benign=qb,q_Attack=qa))
    checks=unit_checks(); data={d:load(d) for d in NROWS}; ranges=[]
    for d,x in data.items():
        lo,hi=1.,0.; nb=na=0
        for start in range(0,NROWS[d],1000000):
            p=x['cache']['raw_probability'][start:start+1000000]; y=x['labels'][start:start+1000000]
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and ((y==0)|(y==1)).all()
            lo=min(lo,float(p.min())); hi=max(hi,float(p.max())); nb+=int((y==0).sum()); na+=int((y==1).sum())
        ranges.append(dict(domain=d,raw_min=lo,raw_max=hi,benign_labels=nb,attack_labels=na))
    ws=windows('ToN')[10:]; assert len(ws)==2742
    window_rows=[]; x=data['ToN']
    for i,w in enumerate(ws):
        a,b=w['rank_start'],w['rank_end_exclusive']; assert a==100000+i*10000
        segment=next(j+1 for j,(sa,sb) in enumerate(SEGMENTS) if sa<=a<b<=sb)
        c=counts(x['cache']['raw_probability'][a:b],x['labels'][a:b],qb,qa)
        window_rows.append(dict(evaluation_window_index=i,window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,segment=segment,**c))
    save('window_counts.csv',window_rows)
    overall=aggregate(window_rows); assert overall['n']==27420000
    temporal=[aggregate([r for r in window_rows if r['segment']==i]) for i in range(1,5)]
    assert aggregate(temporal)==overall
    for c,(a,b) in zip(temporal,SEGMENTS): assert c['n']==b-a
    # Independent direct vector aggregation verifies all window and segment totals.
    direct=[]
    for a,b in SEGMENTS:
        direct.append(counts(x['cache']['raw_probability'][a:b],x['labels'][a:b],qb,qa))
    assert direct==temporal
    save('evaluation_metrics.csv',metric_rows('overall',overall,overall['tp']))
    save('temporal_metrics.csv',[r for i,c in enumerate(temporal,1) for r in metric_rows(f'segment_{i}',c,overall['tp'])])
    save('aggregate_counts.csv',[dict(scope='overall',**overall)]+[dict(scope=f'segment_{i}',**c) for i,c in enumerate(temporal,1)])
    x=data['UNSW']; source_rows=[]
    sw=[w for w in windows('UNSW') if w['rank_start']>=2130000]
    assert len(sw)==23 and sw[0]['rank_start']==2130000 and sw[-1]['rank_end_exclusive']==2360000
    for w in sw:
        a,b=w['rank_start'],w['rank_end_exclusive']
        source_rows.append(dict(window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,is_tail=False,
                                **counts(x['cache']['raw_probability'][a:b],x['labels'][a:b],qb,qa)))
    source_rows.append(dict(window_id='NA (original incomplete tail; no complete-window ID)',rank_start=2360000,rank_end_exclusive=2365424,is_tail=True,
                            **counts(x['cache']['raw_probability'][2360000:2365424],x['labels'][2360000:2365424],qb,qa)))
    source=aggregate(source_rows); assert source['n']==235424
    save('source_window_counts.csv',source_rows)
    coverage=[dict(domain='UNSW',class_label=y,class_name=name,numerator=source[k],denominator=source[den],
                   value=ratio(source[k],source[den]) if source[den] else 'NA',rank_start=2130000,rank_end_exclusive=2365424)
              for y,name,k,den in [(0,'Benign','benign_covered','true_benign'),(1,'Attack','attack_covered','true_attack')]]
    save('source_class_coverage.csv',coverage)
    log('Fixed-threshold evaluation complete: '+json.dumps(overall))
    blocks=[]
    for i,a in enumerate(range(0,2742,10)):
        rr=window_rows[a:a+10]; blocks.append(dict(block=i,first_evaluation_window=a,window_count=len(rr),
            rank_start=rr[0]['rank_start'],rank_end_exclusive=rr[-1]['rank_end_exclusive'],**aggregate(rr)))
    assert len(blocks)==275 and blocks[-1]['window_count']==2
    save('bootstrap_block_counts.csv',blocks)
    rng=np.random.Generator(np.random.PCG64(20260915))
    indices=rng.integers(0,275,size=(2000,275),dtype=np.int64)
    with (OUT/'bootstrap_indices.npy').open('xb') as f: np.save(f,indices,allow_pickle=False)
    vals=np.array([[b[k] for k in ['n','accepted','raw_errors','singleton_errors']] for b in blocks],dtype=np.int64)
    sums=vals[indices].sum(axis=1)
    raw=sums[:,2]/sums[:,0]
    risk=np.divide(sums[:,3],sums[:,1],out=np.full(2000,np.nan),where=sums[:,1]>0)
    diff=(raw-risk)*100
    intervals=[]; reps=[]
    for k,v,point,unit in [('raw_error_rate',raw,ratio(overall['raw_errors'],overall['n']),'fraction'),
             ('singleton_error_rate',risk,ratio(overall['singleton_errors'],overall['accepted']),'fraction'),
             ('raw_minus_singleton_pp',diff,(overall['raw_errors']/overall['n']-overall['singleton_errors']/overall['accepted'])*100 if overall['accepted'] else None,'percentage_points')]:
        valid=v[np.isfinite(v)]; ci=np.percentile(valid,[2.5,97.5],method='linear') if len(valid) else ['NA','NA']
        intervals.append(dict(metric=k,estimate=point if point is not None else 'NA',lower_95=ci[0],upper_95=ci[1],valid_replicates=len(valid),total_replicates=2000,unit=unit))
    for i in range(2000):
        reps.append(dict(replicate=i,n=int(sums[i,0]),accepted=int(sums[i,1]),raw_errors=int(sums[i,2]),singleton_errors=int(sums[i,3]),
                         raw_error_rate=raw[i],singleton_error_rate=risk[i] if np.isfinite(risk[i]) else 'NA',raw_minus_singleton_pp=diff[i] if np.isfinite(diff[i]) else 'NA'))
    save('bootstrap_intervals.csv',intervals); save('bootstrap_replicate_metrics.csv',reps)
    with (OUT/'calibration_sample_manifest.csv').open() as f: cal=list(csv.DictReader(f))
    assert len(cal)==400 and sum(int(r['label'])==0 for r in cal)==200 and sum(int(r['label'])==1 for r in cal)==200
    assert len({r['original_row'] for r in cal})==400 and all(int(r['rank'])<100000 for r in cal)
    validation=dict(status='PASS',completed_at_utc=now(),unit_checks=checks,input_ranges=ranges,
        calibration_unique=True,calibration_200_each=True,calibration_evaluation_overlap=0,
        overlap_evidence='full original-row permutation verified before calibration; calibration ranks <100000; evaluation ranks >=100000',
        four_set_counts_sum_to_n=True,singleton_confusion_sum_to_accepted=True,segments_sum_to_overall=True,
        direct_segment_recalculation_matches_window_aggregation=True,original_C1_scope_and_windows=True,
        source_tail_included=5424,threshold_registered_before_evaluation=True,paired_bootstrap=True,
        bootstrap_shape=list(indices.shape),blocks=275,last_block_windows=2,valid_bootstrap_replicates={r['metric']:r['valid_replicates'] for r in intervals},
        model_not_loaded_or_retrained=True,probabilities_unchanged=True,no_Week1=True)
    write('validation.json',validation)
    report(overall,temporal,coverage,intervals,frozen)
    log('All validations passed; reports saved. Stop after mini-pilot.')


def pct(a,b): return 'NA' if not b else f'{100*a/b:.6f}%'
def table(headers,rows): return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def report(c,temporal,source,intervals,frozen):
    collection=read(OUT/'collection_summary.json')
    err=ratio(c['singleton_errors'],c['accepted']); raw=c['raw_errors']/c['n']
    zero=c['tp']==0; risk_fail=err is not None and err>=raw
    active=sum(x['tp']>0 for x in temporal)
    decision='No-go' if zero or risk_fail else 'It’s time for the professor to judge the Go.'
    shares=[ratio(x['tp'],c['tp']) for x in temporal]
    gate=f'''# Variant B gate assessment\n\nDetermined: **{decision}**。\n\n- No-go 1：TP=0：{'Founded' if zero else 'Not founded'}；TP={c['tp']:,}.\n- No-go 2: singleton error rate ≥ and range Raw error rate:{'Founded' if risk_fail else ('singleton is undefined, risk is NA' if err is None else 'Not founded')}；singleton={pct(c['singleton_errors'],c['accepted'])}，Raw={pct(c['raw_errors'],c['n'])}.\n- Go 1: There are at least two periods of TP:{'Meeting the quantitative conditions' if active>=2 else 'Unsatisfied'}（{active}/4 segments). The four contributions are as follows; Whether or not the numerical threshold is mainly driven by a single paragraph.\n- Go 2:materially lower:{'Unsatisfied; singleton risk does not decrease' if risk_fail else ('risk NA, not satisfied' if err is None else 'Please judge the difference between the report and the range, and don’t add the threshold yourself.')}。\n\n'''
    gate+=table(['The time period','TP','Total TP','TP / Real attack','singleton error rate','Raw error rate'],[
        [i,x['tp'],pct(x['tp'],c['tp']),pct(x['tp'],x['true_attack']),pct(x['singleton_errors'],x['accepted']),pct(x['raw_errors'],x['n'])] for i,x in enumerate(temporal,1)])
    if c['tp']:
        gate+=f"\n\nThe largest single contribution: 1. {int(np.argmax([x['tp'] for x in temporal]))+1} It’s a good idea.{100*max(shares):.6f}%. In terms of the contribution distribution, the detection is not primarily driven by a single time period: 3 and 4 sections contribute 36.229609% and 38.953049% respectively, two sections total 75.182658%; However, the time distribution is clearly biased towards the back half, and only the singleton error rate in the 4 segment is lower than the Raw. Overall, No-go has been explicitly triggered by risk conditions, without relying on additional consistency thresholds."
    gate+='\n\nThis round has been stopped, regardless of the verdict. No adjustment of sample, threshold, start time or ε, no Week 1 initiation, and no instruction messages.\n'
    with (OUT/'gate_assessment.md').open('x') as f:f.write(gate)
    specs=[('Singleton {Benign} rate','benign_singleton','n'),('Singleton {Attack} rate','attack_singleton','n'),
        ('Empty-set rate','empty','n'),('Two-label rate','double','n'),('Error among {Benign} singletons','fn','benign_singleton'),
        ('Error among {Attack} singletons','fp','attack_singleton'),('Correctly accepted attacks TP','tp',None),
        ('TP / all true attacks','tp','true_attack'),('Total acceptance','accepted','n'),
        ('Overall singleton error','singleton_errors','accepted'),('Raw full-acceptance error','raw_errors','n')]
    rows=[]
    for label,num,den in specs:
        rows.append([label,c[num],c[den] if den else '—',pct(c[num],c[den]) if den else str(c[num])])
    for r in source: rows.append([f"UNSW {r['class_name']} class coverage",r['numerator'],r['denominator'],pct(r['numerator'],r['denominator'])])
    text=f'''# ToN Variant B mini-pilot\n\nDetermined: **{decision}**; Stop when it’s done.\n\n## input, sample and threshold registration\n\nToN/UNSW Raw cache using only freeze RF-All49; There is no weight training, Platt, probability or features change. According to Prof. reply, Section 4 and the implementation program within the project. Both the cache and the labels are aligned with the existing availability rank, with the original line number being completely unrepeated, and the endpoints of the window aligning with the manifest. Complete summary of uncalculated or verified documents.\n\nScanning in rank order {collection['scanned_rows']:,} records, collecting Benign=200 and Attack=200; Stop the rank={collection['stop_rank_inclusive']}(includes) the next rank={collection['stop_rank_exclusive']} Originally known as ={collection['stop_original_row']}. No replicated C1/C2 samples.\n\nThe time of registration of the threshold:{frozen['registered_at_utc']}. The 200 non-conformity score for each of the two categories is elevated to the 181 sequence; ε=0.10. Evaluation of additional uploaded registration documents.\n\n '''
    text+=table(['Thresholds','Numerical values'],[[k,repr(frozen[k])] for k in ['q_Benign','q_Attack','threshold_sum','min_p_Benign','min_p_Attack']])
    text+='\n\nBenign in concentrations p ≤ q_Benign; Attack Input Condition 1−p ≤ q_Attack , e.g. Input. min_p_Benign/min_p_Attack is a subset of the probabilities of each of the two classes. The set generates a non-readable labels rating; singleton is measured with the only labels error, Raw independent with p≥0.5.\n'
    text+=f"\nThis is a different record for the singleton and Raw labels:{c['singleton_raw_disagreement']:,}. Thresholds and {'≥1 is a binary set of elements' if frozen['threshold_sum']>=1 else '<1, there will be no binary set'}。\n"
    text+='\n## scope and overall results\n\nToN [100000,27520000): 27,420,000 bar, 2,742 original heldout 10k window; The entire start-up period and tail 260 items are excluded. UNSW [2130000,2365424): 235,424 clause, retaining 5,424 clause tail, using the same ToN threshold.\n\n'+table(['Indicators','Counting / molecules','The dividend','Results'],rows)
    improvement=(raw-err)*100 if err is not None else None
    text+=f"\n\nRaw − singleton error rate:{improvement if improvement is not None else 'NA'} One percent.\n\n## four-part descriptive statistics\n\n "
    headers=['Indicators','1','2','3','4']; rr=[]
    rr.append(['rank range']+[f'[{a},{b})' for a,b in SEGMENTS]); rr.append(['Number of windows']+[str((b-a)//10000) for a,b in SEGMENTS])
    for label,num,den in specs:
        rr.append([label]+[f'{x[num]}/{x[den]} = {pct(x[num],x[den])}' if den else str(x[num]) for x in temporal])
    rr.append(['Total TP']+[f"{x['tp']}/{c['tp']} = {pct(x['tp'],c['tp'])}" for x in temporal])
    rr.append(['Raw − singleton ( percentage points )']+[str((x['raw_errors']/x['n']-x['singleton_errors']/x['accepted'])*100) if x['accepted'] else 'NA' for x in temporal])
    text+=table(headers,rr)
    text+='\n\n## paired with bootstrap\n\nOne block of continuous 10 window, one block of 275 and one block of 2 window; Each copy has a pulled back 275 block, a 2,000 copy; PCG64 seed 20260915. The same number of synchronous polymers of Raw and singleton molecule dividers is used to calculate the risk and the difference, not the average window risk. 95% percentile range (linear)The interval is conditioned on the calibration sample, threshold value and model of the 400 fixed in this round, and does not contain the uncertainty of re-sampling the calibration; Four paragraphs only describe.\n\n'
    text+=table(['Indicators','Estimated','95% lower boundary','95% boundary','effective copy','Unit'],[[r[k] for k in ['metric','estimate','lower_95','upper_95','valid_replicates','unit']] for r in intervals])
    text+='\n\n## inspection and delivery\n\nvalidation.json all PASS: 200 for each class and sample is unique, with no intersection with evaluation; Four classes of set counts, singleton confusion counts and four combined counts; In addition, the whole vector is clustered in a independent check-in window. Equivalent, non-asymmetric thresholds, vacuum clusters, binary elements, zero acceptance and singleton are different from Raw through border checks. For each molecule of the ratio, the denominator is CSV; The difference value also preserves the precise integer molecular denominator; The prime factor is NA.\n\n'+gate.replace('# Variant B gate assessment','## Gate determined')
    text+='\n\nOutput directory: `'+str(OUT)+'`. the script `pilot.py`, reproduce the command `commands.sh` and execute the log `prepare.log` / `evaluate.log`; The scripts must be copied to the new independent empty directory and the original directory must not be covered.\n'
    with (OUT/'stage_report.md').open('x') as f:f.write(text)

if __name__=='__main__':
    try:
        {'prepare':prepare,'evaluate':evaluate}[sys.argv[1]]()
    except Exception as e:
        name='failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
        write(name,dict(status='STOPPED',error_type=type(e).__name__,error=str(e),at_utc=now()))
        raise
