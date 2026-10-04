"""User-authorized exploratory evaluation after Outcome B; no calibration or training.
Pure counting/statistical helpers adapted from the historical pilot in a new copy.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
OUT=Path(__file__).resolve().parent
SEGMENTS=[(100000,6960000),(6960000,13820000),(13820000,20670000),(20670000,27520000)]
NROWS={'ToN':27520260,'UNSW':2365424}
KEYS=['n','true_benign','true_attack','benign_singleton','attack_singleton','empty','double','tn','fn','fp','tp','accepted','singleton_errors','raw_errors','singleton_raw_disagreement','benign_covered','attack_covered']
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

def pct(a,b): return 'NA' if not b else f'{100*a/b:.6f}%'

def table(headers,rows): return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def evaluate_counts(data,qb,qa):
    ws=windows('ToN')[10:]; assert len(ws)==2742
    window_rows=[]; x=data['ToN']
    for i,w in enumerate(ws):
        a,b=w['rank_start'],w['rank_end_exclusive']; assert a==100000+i*10000
        segment=next(j+1 for j,(sa,sb) in enumerate(SEGMENTS) if sa<=a<b<=sb)
        c=counts(np.asarray(x['cache']['raw_probability'][a:b],dtype=np.float64),x['labels'][a:b],qb,qa)
        window_rows.append(dict(evaluation_window_index=i,window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,segment=segment,first_original_row=int(x['original_rows'][a]),last_original_row=int(x['original_rows'][b-1]),**c))
    save('window_counts.csv',window_rows)
    overall=aggregate(window_rows); assert overall['n']==27420000
    temporal=[aggregate([r for r in window_rows if r['segment']==i]) for i in range(1,5)]
    assert aggregate(temporal)==overall
    for c,(a,b) in zip(temporal,SEGMENTS): assert c['n']==b-a
    # Independent direct vector aggregation verifies all window and segment totals.
    direct=[]
    for a,b in SEGMENTS:
        direct.append(counts(np.asarray(x['cache']['raw_probability'][a:b],dtype=np.float64),x['labels'][a:b],qb,qa))
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
                                **counts(np.asarray(x['cache']['raw_probability'][a:b],dtype=np.float64),x['labels'][a:b],qb,qa)))
    source_rows.append(dict(window_id='NA (original incomplete tail; no complete-window ID)',rank_start=2360000,rank_end_exclusive=2365424,is_tail=True,
                            **counts(np.asarray(x['cache']['raw_probability'][2360000:2365424],dtype=np.float64),x['labels'][2360000:2365424],qb,qa)))
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
    return overall,temporal,coverage,intervals

def main():
    base=ROOT/'outputs/post_p7_diagnostics/xgboost_variant_b_ton_pilot'
    candidates=[]
    for f in base.glob('run_*/threshold_freeze.json'):
        p=read(f.parent/'protocol.json'); t=read(f)
        if p['model_identity']['model_id']=='xgb_All49' and t['q_Benign']==0.009039638563990593 and t['q_Attack']==0.9996102542208973 and p['inputs']['cache']==str(ROOT/'outputs/prediction_cache_v2/ToN/xgb_All49.npy'):
            candidates.append(f.parent)
    assert len(candidates)==1, str(candidates)
    cal=candidates[0]
    names=['protocol.json','calibration_sample_manifest.csv','calibration_scores.csv','threshold_freeze.json','threshold_report.md','validation.json']
    snapshots={name:(cal/name).read_bytes() for name in names}
    original_protocol=read(cal/'protocol.json'); frozen=read(cal/'threshold_freeze.json')
    assert read(cal/'validation.json')['status']=='PASS'
    qb,qa=frozen['q_Benign'],frozen['q_Attack']
    assert qa>=.90 and 'Outcome B' in frozen['gate_branch']
    assert original_protocol['labels']=={'0':'Benign','1':'Attack'}
    samples=list(csv.DictReader(snapshots['calibration_sample_manifest.csv'].decode().splitlines()))
    scores=list(csv.DictReader(snapshots['calibration_scores.csv'].decode().splitlines()))
    assert len(samples)==len(scores)==400
    assert sum(r['label']=='0' for r in samples)==sum(r['label']=='1' for r in samples)==200
    assert len({r['original_row'] for r in samples})==400
    assert all(0<=int(r['rank'])<100000 for r in samples)
    assert all(all(s[k]==r[k] for k in r) for s,r in zip(scores,samples))
    cfg=read(ROOT/'configs/execution_v2.json')
    training_path=Path(original_protocol['inputs']['model_record']); training=read(training_path)
    assert training['model_id']=='xgb_All49' and training['status']=='COMPLETE'
    assert training['parameters']==cfg['models']['xgb']
    assert training['parameters']['objective']=='binary:logistic'
    artifact=training_path.parent/training['artifact']
    assert artifact.stat().st_size==training['artifact_size_bytes']
    fm=read(ROOT/cfg['outputs']['feature_store']/'feature_store_manifest.json')
    assert fm['row_order']=='availability_sorted_rank'
    assert Path(fm['rank_source'])==Path(cfg['prepared_run'])
    data={}; input_paths={}; input_checks=[]
    for domain,n in NROWS.items():
        folder=ROOT/cfg['outputs']['base_predictions']/domain
        mm=read(folder/'xgb_All49_manifest.json'); cm=read(folder/'manifest.json')
        lm=read(ROOT/cfg['outputs']['feature_store']/f'{domain}_manifest.json')
        paths=dict(cache=ROOT/mm['file'],labels=ROOT/cfg['outputs']['feature_store']/lm['files']['binary_label'],original_rows=Path(cfg['prepared_run'])/f'{domain}_sorted_original_rows.npy')
        assert mm['model_id']=='xgb_All49' and mm['domain']==cm['domain']==domain
        assert mm in cm['models'] and mm['rows']==cm['rows']==lm['rows']==n
        assert mm['feature_indices']==training['feature_indices']==list(range(49))
        assert mm['feature_names']==training['feature_names']==fm['feature_order']
        assert cm['row_order']=='availability_rank' and Path(cm['original_row_index'])==paths['original_rows']
        assert paths['cache'].stat().st_size==mm['bytes']
        x={k:np.load(p,mmap_mode='r',allow_pickle=False) for k,p in paths.items()}
        assert all(v.shape==(n,) for v in x.values())
        assert x['cache'].dtype['raw_probability']==np.dtype('float64')
        # Verify the rank->original-row permutation without changing order.
        seen=np.zeros(n,dtype=bool)
        for start in range(0,n,1000000):
            rows=x['original_rows'][start:min(n,start+1000000)]
            assert ((rows>=0)&(rows<n)).all()
            assert len(np.unique(rows))==len(rows) and not seen[rows].any()
            seen[rows]=True
        assert seen.all()
        del seen
        for w in windows(domain):
            assert int(x['original_rows'][w['rank_start']])==w['first_original_row']
            assert int(x['original_rows'][w['rank_end_exclusive']-1])==w['last_original_row']
        start,end=(100000,27520000) if domain=='ToN' else (2130000,2365424)
        low,high=1.,0.
        for a in range(start,end,1000000):
            b=min(a+1000000,end)
            p=np.asarray(x['cache']['raw_probability'][a:b],dtype=np.float64); y=x['labels'][a:b]
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            assert ((y==0)|(y==1)).all()
            low=min(low,float(p.min()));high=max(high,float(p.max()))
        input_paths[domain]={k:str(v) for k,v in paths.items()}
        input_paths[domain]['cache_manifest']=str(folder/'xgb_All49_manifest.json')
        input_checks.append(dict(domain=domain,rows=n,raw_min=low,raw_max=high,rank_permutation_and_window_boundaries=True))
        data[domain]=x
    for r in samples:
        rank=int(r['rank'])
        assert int(data['ToN']['original_rows'][rank])==int(r['original_row'])
        assert int(data['ToN']['labels'][rank])==int(r['label'])
    tests=unit_checks()
    # Exact frozen comparison boundaries and adjacent float64 values.
    bp=np.array([np.nextafter(qb,-np.inf),qb,np.nextafter(qb,np.inf)])
    assert sets(bp,qb,qa)[0].tolist()==[True,True,False]
    ap=np.array([np.nextafter(1-qa,-np.inf),1-qa,np.nextafter(1-qa,np.inf)])
    assert np.array_equal(sets(ap,qb,qa)[1],np.array([(1-float(v))<=qa for v in ap]))
    assert sets(np.array([1-qa]),qb,qa)[1][0]
    protocol=dict(started_at_utc=now(),classification='Exploratory evaluation after threshold gate stop',
        deviation='User explicitly authorized evaluation despite q_Attack >= 0.90. This deviates from the latest professor letter threshold stop rule; it is not evaluation following protocol passage.',
        original_gate='Outcome B; unchanged',authority_letter=str(ROOT/'docs/XGBOOST_TON/Reply_to_Siyuan_VariantB_Analysis_and_Next_Step.md'),
        calibration_directory=str(cal),threshold_source=str(cal/'threshold_freeze.json'),q_Benign=qb,q_Attack=qa,
        model_record=str(training_path),model_artifact=str(artifact),inputs=input_paths,
        probability='raw_probability=P(Attack); float64; unchanged',labels={'0':'Benign','1':'Attack'},
        epsilon=.1,startup=[0,100000],ton_evaluation=[100000,27520000],ton_rows=27420000,ton_windows=2742,ton_tail_excluded=260,
        segments=[dict(segment=i,rank_start=a,rank_end_exclusive=b,windows=(b-a)//10000) for i,(a,b) in enumerate(SEGMENTS,1)],
        source_evaluation=[2130000,2365424],source_rows=235424,source_tail_included=5424,
        prediction={'Benign':'p <= q_Benign','Attack':'1-p <= q_Attack','singleton':'accept unique label, regardless of Raw label','empty':'reject','double':'abstain','raw':'p>=0.5'},
        bootstrap={'block_windows':10,'blocks':275,'last_block_windows':2,'replicates':2000,'generator':'PCG64','seed':20260915,'paired':True,'risk':'sum(errors)/sum(records)','percentiles':[2.5,97.5],'method':'linear','undefined':'NA; report valid replicates','inference':'conditional on fixed calibration samples, thresholds, and frozen model'},
        interpretation='Original gate unchanged; no automatic Outcome A or Phase 2; no numerical materially-lower threshold; segments descriptive',
        lineage_evidence=['predictions.py loads configured source model and stores binary logistic output in raw_probability before probability transformation','cache and feature-store manifests specify availability rank; feature_store.py gathers labels and features by the same original-row order'],
        stop_after='Requested exploratory evaluation only; no further experiment; no professor message')
    write('protocol.json',protocol)
    log('Input and threshold checks passed; evaluating fixed scopes.')
    c,temporal,source,intervals=evaluate_counts(data,qb,qa)
    assert all((cal/name).read_bytes()==value for name,value in snapshots.items())
    assert read(cal/'threshold_freeze.json')['q_Benign'].hex()==qb.hex()
    assert read(cal/'threshold_freeze.json')['q_Attack'].hex()==qa.hex()
    matrix=np.load(OUT/'bootstrap_indices.npy',allow_pickle=False)
    assert matrix.shape==(2000,275) and ((matrix>=0)&(matrix<275)).all()
    write('validation.json',dict(status='PASS',completed_at_utc=now(),missing_items=[],input_checks=input_checks,unit_checks=tests,
        frozen_thresholds_loaded_exactly=True,original_calibration_files_unchanged=True,no_threshold_recalculation=True,
        startup_evaluation_overlap=0,calibration_identities_and_labels_match=True,
        four_sets_sum_to_n=True,singleton_confusion_sum_to_accepted=True,segments_sum_to_overall=True,
        direct_segment_counts_match_window_counts=True,ton_windows=2742,segment_windows=[686,686,685,685],
        source_tail_included=5424,paired_bootstrap=True,bootstrap_shape=list(matrix.shape),bootstrap_valid_replicates={r['metric']:r['valid_replicates'] for r in intervals},
        model_loaded_or_trained=False,phase_2_started=False,original_outcome='Outcome B'))
    risk=ratio(c['singleton_errors'],c['accepted']); raw=ratio(c['raw_errors'],c['n'])
    fail=c['tp']==0 or (risk is not None and risk>=raw)
    conclusion=('Clear No-go: TP=0.' if c['tp']==0 else 'Clear No-go: The singleton error rate is greater than or equal to the Raw error rate in the same range, not reaching materially lower.') if fail else 'does not trigger the above specified No-go conditions; materially lower and the time distribution must be judged by the professor, not automatically by Go.'
    rows=metric_rows('overall',c,c['tp'])
    report='# XGBoost-All49 / ToN Variant B Exploratory evaluation after shutdown of gate control\n\n'
    report+='This time, the user has explicitly authorized and deviated from the devalue stop rule of the latest professor’s letter, not the original agreement’s approval assessment. The original door control q_Attack>=0.90, Outcome B remained unchanged.\n\n'
    report+=f' freeze Threshold: q_Benign={qb!r}，q_Attack={qa!r}; From {cal}/threshold_freeze.json loaded, not recalculated.\n\n '
    report+=f' Exploratory testing contrasts with the original assessment criteria:{conclusion}\n\n'
    report+=table(['Overall indicators','The molecule','The dividend','Example or numerical value'],[[r['metric'],r['numerator'],r['denominator'],r['value']] for r in rows])
    report+='\n\n'+table(['The time period','TP','TP as a whole','TP/ real attack','singleton error rate','Raw error rate','Raw−singleton(pp)'],[[i,t['tp'],pct(t['tp'],c['tp']),pct(t['tp'],t['true_attack']),pct(t['singleton_errors'],t['accepted']),pct(t['raw_errors'],t['n']),100*(t['raw_errors']/t['n']-t['singleton_errors']/t['accepted']) if t['accepted'] else 'NA'] for i,t in enumerate(temporal,1)])
    active=sum(t['tp']>0 for t in temporal)
    report+=f'\n\nIn the meantime, I’m going to have to make a decision.{active}/4. The largest single contribution:{pct(max(t["tp"] for t in temporal),c["tp"])}. The threshold for non-aluminium is set by a single-section drive valve; For the full four indicators and molecular proportions see temporal_metrics.csv.\n\n '
    report+=table(['risk indicator','Score estimates','95% lower boundary','95% boundary','effective copy','Unit'],[[r['metric'],r['estimate'],r['lower_95'],r['upper_95'],r['valid_replicates'],r['unit']] for r in intervals])
    report+='\n\nThe range is based on adjacent 10 window blocks, 275 block (last block 2 window), 2000 paired with bootstrap and PCG64 seed 20260915; Each time the total record number is calculated by the total error number /. The range is conditioned on fixed calibration samples, thresholds, and models, and does not include calibration uncertainty.\n\n'
    report+=table(['UNSW category covered','The molecule','The dividend','Examples'],[[r['class_name'],r['numerator'],r['denominator'],r['value']] for r in source])
    report+='\n\nThe source category coverage contains a set of two elements, not equal to the singleton accuracy rate. It is necessary to inspect all passes without missing items; The output of the zero subset is NA. ToN excludes startup and 260 clause tail and UNSW retains 5424 clause tail.\n\n'
    report+='The original calibration directory and historical products are unchanged. Outcome A is not automatically re-determined, Phase 2 is not entered, no additional experiments are performed, and no messages are sent to professors. Stop when it’s done.\n'
    with (OUT/'evaluation_report.md').open('x') as f:f.write(report)
    log('All checks PASS. '+conclusion+' Original Outcome B unchanged. Stop.')

if __name__=='__main__':
    main()
