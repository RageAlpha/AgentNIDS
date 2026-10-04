from pathlib import Path
import csv,json,sys,platform
from datetime import datetime,timezone
import numpy as np
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run'); OUT=Path(__file__).resolve().parent
BASE=ROOT/'outputs/post_p7_diagnostics'
RF=BASE/'variant_b_ton_pilot/run_20260916T085242Z'
XC=BASE/'xgboost_variant_b_ton_pilot/run_20260918T070417Z'
XE=BASE/'xgboost_variant_b_ton_exploratory_eval/run_20260918T071352Z'
SEG=[(100000,6960000),(6960000,13820000),(13820000,20670000),(20670000,27520000)]
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def js(name,obj):
 with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def save(name,data):
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
CHECK=[]
def check(name,ok,detail=''):
 CHECK.append(dict(check=name,passed=bool(ok),detail=detail)); assert ok,(name,detail)
def sets(p,qb,qa):
 b=p<=qb;a=(1.0-p)<=qa
 return b,a,b&~a,a&~b,~b&~a,b&a
def counts(p,y,qb,qa):
 p=np.asarray(p,dtype=np.float64);b,a,bs,ats,e,d=sets(p,qb,qa);y=np.asarray(y);z=y==0;t=y==1
 c=dict(n=len(y),true_benign=int(z.sum()),true_attack=int(t.sum()),benign_singleton=int(bs.sum()),attack_singleton=int(ats.sum()),empty=int(e.sum()),double=int(d.sum()),tn=int((bs&z).sum()),fn=int((bs&t).sum()),fp=int((ats&z).sum()),tp=int((ats&t).sum()),accepted=int((bs|ats).sum()),singleton_errors=int(((bs&t)|(ats&z)).sum()),raw_errors=int(((p>=.5)!=y).sum()),singleton_raw_disagreement=int(((bs&(p>=.5))|(ats&(p<.5))).sum()),benign_covered=int((b&z).sum()),attack_covered=int((a&t).sum()))
 assert c['benign_singleton']+c['attack_singleton']+c['empty']+c['double']==c['n']
 assert c['tn']+c['fn']+c['fp']+c['tp']==c['accepted']
 return c
PAIRS={'benign_coverage':('benign_covered','true_benign'),'attack_coverage':('attack_covered','true_attack'),'singleton_error_rate':('singleton_errors','accepted'),'acceptance_rate':('accepted','n'),'benign_singleton_error_rate':('fn','benign_singleton'),'attack_singleton_error_rate':('fp','attack_singleton'),'tp':('tp',None),'tp_fraction_of_true_attacks':('tp','true_attack'),'benign_singleton_rate':('benign_singleton','n'),'attack_singleton_rate':('attack_singleton','n'),'empty_set_rate':('empty','n'),'double_set_rate':('double','n'),'raw_error_rate':('raw_errors','n')}
def metrics(c):
 out={}
 for m,(a,b) in PAIRS.items():
  num=c[a];den=c[b] if b else 1;out[m]=(num,den,num/den if den else None,'count' if b is None else 'fraction')
 num=100*(c['raw_errors']*c['accepted']-c['singleton_errors']*c['n']);den=c['n']*c['accepted']
 out['raw_minus_singleton_pp']=(num,den,num/den if den else None,'percentage_points')
 return out

def main():
 check('equalities_asymmetric_empty_double',all(np.array_equal(x,y) for x,y in zip(sets(np.array([0.,.25,.5,.75,1.]),.25,.5)[2:],[[1,1,0,0,0],[0,0,1,1,1],[0,0,0,0,0],[0,0,0,0,0]])))
 check('empty_boundary',sets(np.array([.25,.5,.75]),.25,.25)[4].tolist()==[False,True,False])
 check('double_boundary',sets(np.array([.25,.5,.75]),.75,.75)[5].tolist()==[True,True,True])
 check('zero_acceptance_NA',metrics(counts(np.array([.5]),np.array([1]),.25,.25))['singleton_error_rate'][2] is None)
 check('singleton_can_differ_from_raw',counts(np.array([.25]),np.array([1]),.1,.75)['tp']==1)
 rp=read(RF/'protocol.json');xp=read(XC/'protocol.json');ep=read(XE/'protocol.json')
 check('historical_protocol_linkage',ep['threshold_source']==str(XC/'threshold_freeze.json') and xp['inputs']['original_calibration_manifest']==str(RF/'calibration_sample_manifest.csv') and rp['ton_evaluation']==ep['ton_evaluation']==[100000,27520000])
 early=rows(RF/'calibration_sample_manifest.csv');check('same_historical_sample',early==rows(XC/'calibration_sample_manifest.csv'))
 y=np.load(ROOT/'outputs/feature_store_v2/ToN_label.npy',mmap_mode='r'); orig=np.load(ROOT/'outputs/round1_data_prep_v1/ToN_sorted_original_rows.npy',mmap_mode='r')
 check('row_count',len(y)==len(orig)==27520260)
 check('binary_labels',np.all((y==0)|(y==1)))
 # Full identity permutation verifies disjoint original-row identities, not just rank ranges.
 check('original_row_permutation',np.array_equal(np.sort(orig),np.arange(len(orig))))
 wm=[r for r in rows(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv') if r['domain']=='ToN' and r['window_size']=='10000' and r['partition']=='T_heldout']
 wm.sort(key=lambda r:int(r['rank_start']));check('manifest_windows',len(wm)==2752)
 for i,w in enumerate(wm):
  assert int(w['rank_start'])==i*10000 and int(w['rank_end_exclusive'])==(i+1)*10000
  assert int(w['first_original_row'])==orig[i*10000] and int(w['last_original_row'])==orig[(i+1)*10000-1]
 check('all_window_original_endpoints',True)
 check('startup_before_evaluation',int(wm[9]['last_available_ms'])<=int(wm[10]['first_available_ms']))
 er=np.array([int(r['rank']) for r in early]);check('early_identity',len(np.unique(er))==400 and all(int(r['original_row'])==orig[int(r['rank'])] and int(r['label'])==y[int(r['rank'])] and r['window_id']==wm[int(r['rank'])//10000]['window_id'] for r in early))
 check('early_class_counts',np.bincount(y[er],minlength=2).tolist()==[200,200] and er.max()==4019)
 fs=read(ROOT/'outputs/feature_store_v2/feature_store_manifest.json');check('feature_rank_source',fs['row_order']=='availability_sorted_rank' and fs['rank_source']==str(ROOT/'outputs/round1_data_prep_v1'))
 inputs={};caches={}; thresholds={}
 for model,folder in [('rf_All49',RF),('xgb_All49',XC)]:
  cm=read(ROOT/f'outputs/prediction_cache_v2/ToN/{model}_manifest.json');tm=read(ROOT/f'outputs/source_v2/models/{model}/training_manifest.json')
  artifact=ROOT/f'outputs/source_v2/models/{model}'/tm['artifact'];cachepath=ROOT/f'outputs/prediction_cache_v2/ToN/{model}.npy'
  check(model+'_identity',cm['model_id']==tm['model_id']==model and cm['feature_indices']==tm['feature_indices']==list(range(49)) and cm['feature_names']==tm['feature_names']==fs['feature_order'] and artifact.stat().st_size==tm['artifact_size_bytes'] and cachepath.stat().st_size==cm['bytes'])
  caches[model]=np.load(cachepath,mmap_mode='r');check(model+'_dtype_shape',caches[model].shape==y.shape and caches[model].dtype['raw_probability']==np.dtype('float64'))
  thresholds[model]={'Early-400':read(folder/'threshold_freeze.json')}
  inputs[model]=dict(cache=str(cachepath),cache_manifest=str(ROOT/f'outputs/prediction_cache_v2/ToN/{model}_manifest.json'),model_record=str(ROOT/f'outputs/source_v2/models/{model}/training_manifest.json'),model_artifact=str(artifact),model_identity=tm,early_threshold_source=str(folder/'threshold_freeze.json'))
 protocol=dict(registered_at_utc=now(),classification='User-authorized exploratory sampling comparison after Outcome B; historical gates unchanged',inputs=inputs,early_manifest=str(RF/'calibration_sample_manifest.csv'),historical_evaluations=dict(rf=str(RF),xgb=str(XE)),labels={'0':'Benign','1':'Attack'},probability='raw_probability=P(Attack); float64; unchanged',startup=[0,100000],strata=[[i*25000,(i+1)*25000] for i in range(4)],sampling='one Generator(PCG64(20260918)); T1..T4, label 0 then 1; sorted candidate ranks; choice(size=50,replace=False); final sort by rank; only once',epsilon=.1,K=400,per_class=200,threshold='class scores p / 1-p; ascending item 181, index 180; no interpolation',evaluation=[100000,27520000],evaluation_periods=SEG,excluded_tail=260,prediction=ep['prediction'],bootstrap=ep['bootstrap'],new_threshold_gate='none; continue regardless of q_Attack',scope='ToN RF-All49 and XGBoost-All49 only; no retraining, probability changes, sweeps, rolling calibration, other experiments, hashes, messages or Phase 2',environment=dict(python=platform.python_version(),numpy=np.__version__),lineage_evidence=['feature_store.py: sorted_y[start:stop]=raw_y[order[start:stop]]','predictions.py: RF result[:,1]; XGB binary:logistic inplace_predict; raw stored before transforms in identical feature rank order'])
 js('protocol.json',protocol)
 support=[dict(startup_segment=f'T{i+1}',rank_start=i*25000,rank_end_exclusive=(i+1)*25000,label=k,class_name=['Benign','Attack'][k],support=int((y[i*25000:(i+1)*25000]==k).sum()),required=50) for i in range(4) for k in range(2)]
 save('startup_support.csv',support)
 if any(r['support']<50 for r in support):js('status.json',dict(status='STOP_SUPPORT'));return
 rng=np.random.Generator(np.random.PCG64(20260918));picked=[]
 for i in range(4):
  for k in range(2):
   candidates=np.sort(np.flatnonzero(y[i*25000:(i+1)*25000]==k)+i*25000)
   picked.extend(rng.choice(candidates,size=50,replace=False).tolist())
 ranks=np.sort(picked);check('new_unique_400',len(np.unique(ranks))==400)
 samples=[dict(rank=int(r),original_row=int(orig[r]),label=int(y[r]),window_id=wm[r//10000]['window_id'],startup_segment=f'T{r//25000+1}') for r in ranks]
 check('each_stratum_class_50',all(sum(r['startup_segment']==f'T{i+1}' and r['label']==k for r in samples)==50 for i in range(4) for k in range(2)))
 check('calibration_evaluation_disjoint',ranks.max()<100000 and er.max()<100000,'Full original-row permutation verified')
 save('calibration_sample_manifest.csv',samples);save('early_sample_manifest.csv',early)
 overlap=int(np.intersect1d(ranks,er).size)
 js('label_use.json',dict(calibration_labels_per_scheme=400,startup_label_records_read_for_support_and_candidates=100000,early_historical_scanned_labels=4020,overlap_with_early=overlap,label_return_timestamp_record_found=False,assumption='Assuming the full startup labels offline experiment is available; The availability of the current is not the return time of labels and the availability of the actual deployment of labels has not been verified; The two schemes only have the same pH-value for the fit sample, and the labels derived from cost cannot be called the same.',startup_last_available_ms=wm[9]['last_available_ms'],evaluation_first_available_ms=wm[10]['first_available_ms'],evaluation_labels_not_used_for_sampling_or_tuning=True))
 for model,cache in caches.items():
  p=np.asarray(cache['raw_probability'][ranks],dtype=np.float64);score=np.where(y[ranks]==0,p,1.-p)
  check(model+'_calibration_probability_valid',np.isfinite(p).all() and np.all((p>=0)&(p<=1)))
  q=[float(np.sort(score[y[ranks]==k])[180]) for k in range(2)]
  for k in range(2):check(model+f'_order_statistic_{k}',q[k]==sorted(score[y[ranks]==k])[180])
  save(model+'_calibration_scores.csv',[dict(**s,raw_probability=float(v),true_class_score=float(t)) for s,v,t in zip(samples,p,score)])
  thresholds[model]['Temporal-Stratified-400']=dict(q_Benign=q[0],q_Attack=q[1],threshold_sum=sum(q),one_minus_q_Benign=1-q[0],one_minus_q_Attack=1-q[1],frozen_at_utc=now(),sample_manifest=str(OUT/'calibration_sample_manifest.csv'),per_class=200,order_statistic_one_based=181,order_statistic_zero_based=180,precision='float64',evaluation_not_started=True)
  js(model+'_threshold_freeze.json',thresholds[model])
 js('freeze_complete.json',dict(frozen_at_utc=now(),shared_sample_manifest=str(OUT/'calibration_sample_manifest.csv'),overlap=overlap))
 print('FROZEN',json.dumps(thresholds),flush=True)
 evaluate(caches,y,wm,thresholds)


def evaluate(caches,y,wm,thresholds):
 js('evaluation_start.json',dict(started_at_utc=now()))
 allwindow=[];metricrows=[];aggregates={};arrays={};keys=None
 for model,cache in caches.items():
  for scheme,q in thresholds[model].items():
   wr=[]
   for i,w in enumerate(wm[10:]):
    a=int(w['rank_start']);b=int(w['rank_end_exclusive']);p=np.asarray(cache['raw_probability'][a:b],dtype=np.float64)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    c=counts(p,y[a:b],q['q_Benign'],q['q_Attack']);period=next(j+1 for j,(lo,hi) in enumerate(SEG) if lo<=a<hi)
    wr.append(dict(model=model,scheme=scheme,evaluation_window_index=i,window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,period=f'E{period}',**c))
   keys=list(c);allwindow.extend(wr);arr=np.array([[r[k] for k in keys] for r in wr],dtype=np.int64);arrays[model,scheme]=arr
   for period in ['overall','E1','E2','E3','E4']:
    chosen=wr if period=='overall' else [r for r in wr if r['period']==period]
    agg={k:sum(r[k] for r in chosen) for k in keys};aggregates[model,scheme,period]=agg
    for m,(num,den,val,unit) in metrics(agg).items():metricrows.append(dict(model=model,scheme=scheme,period=period,metric=m,numerator=num,denominator=den,value=val if val is not None else 'NA',unit=unit))
   check(model+scheme+'_period_sums',all(sum(aggregates[model,scheme,f'E{i}'][k] for i in range(1,5))==aggregates[model,scheme,'overall'][k] for k in keys))
   check(model+scheme+'_count_invariants',len(wr)==2742 and aggregates[model,scheme,'overall']['n']==27420000)
   if scheme=='Early-400':
    hist=RF if model=='rf_All49' else XE;old=rows(hist/'window_counts.csv')
    check(model+'_all_historical_window_counts',len(old)==len(wr) and all(all(int(o[k])==r[k] for k in keys) for o,r in zip(old,wr)))
    for filename in ['evaluation_metrics.csv','temporal_metrics.csv']:
     for h in rows(hist/filename):
      per='overall' if h['scope']=='overall' else 'E'+h['scope'].split('_')[-1]
      if h['metric']=='tp_share_of_overall':num=aggregates[model,scheme,per]['tp'];den=aggregates[model,scheme,'overall']['tp'];val=num/den
      else:num,den,val,_=metrics(aggregates[model,scheme,per])[h['metric']]
      assert num==int(h['numerator']) and den==int(h['denominator']) and np.isclose(val,float(h['value']),rtol=0,atol=1e-12),(model,filename,h)
    check(model+'_historical_metrics_exact_counts',True)
   print('EVALUATED',model,scheme,flush=True)
 save('window_counts.csv',allwindow);save('metrics.csv',metricrows)
 idx=np.load(RF/'bootstrap_indices.npy',allow_pickle=False)
 reference=np.random.Generator(np.random.PCG64(20260915)).integers(0,275,size=(2000,275),dtype=np.int64)
 check('historical_bootstrap_matrix_exact',np.array_equal(idx,reference) and idx.shape==(2000,275))
 with (OUT/'bootstrap_indices.npy').open('xb') as f:np.save(f,idx,allow_pickle=False)
 js('bootstrap_manifest.json',dict(source=str(RF/'bootstrap_indices.npy'),verified_against_fixed_PCG64=True,shape=list(idx.shape),seed=20260915,block_windows=10,blocks=275,last_block_windows=2,shared_across_all_four_operations=True,percentiles=[2.5,97.5],conditional_on='fixed models and this fixed calibration sample; excludes calibration resampling uncertainty'))
 breps={};blockrows=[];intervals=[];rep_rows=[]
 for (model,scheme),arr in arrays.items():
  blocks=np.add.reduceat(arr,np.arange(0,len(arr),10),axis=0)
  for i,vals in enumerate(blocks):blockrows.append(dict(model=model,scheme=scheme,block=i,windows=10 if i<274 else 2,**dict(zip(keys,map(int,vals)))))
  summed=blocks[idx].sum(axis=1);d={k:summed[:,j].astype(np.float64) for j,k in enumerate(keys)};rr={}
  for m,(a,b) in PAIRS.items():
   den=d[b] if b else np.ones(2000);rr[m]=np.divide(d[a],den,out=np.full(2000,np.nan),where=den!=0)
  rr['raw_minus_singleton_pp']=100*(rr['raw_error_rate']-rr['singleton_error_rate']);breps[model,scheme]=rr
  for m,v in rr.items():
   good=v[np.isfinite(v)];ci=np.percentile(good,[2.5,97.5]) if len(good) else [None,None]
   intervals.append(dict(model=model,comparison=scheme,metric=m,estimate=metrics(aggregates[model,scheme,'overall'])[m][2],low_95=ci[0],high_95=ci[1],valid_replicates=len(good),unit='count' if m=='tp' else ('percentage_points' if m=='raw_minus_singleton_pp' else 'fraction')))
  for i in range(2000):rep_rows.append(dict(model=model,comparison=scheme,replicate=i,**{m:float(v[i]) if np.isfinite(v[i]) else 'NA' for m,v in rr.items()}))
 for model in caches:
  for m in PAIRS.keys()|{'raw_minus_singleton_pp'}:
   v=breps[model,'Temporal-Stratified-400'][m]-breps[model,'Early-400'][m];scale=1 if m in ['tp','raw_minus_singleton_pp'] else 100;v*=scale
   good=v[np.isfinite(v)];ci=np.percentile(good,[2.5,97.5]) if len(good) else [None,None]
   e=metrics(aggregates[model,'Early-400','overall'])[m][2];s=metrics(aggregates[model,'Temporal-Stratified-400','overall'])[m][2]
   intervals.append(dict(model=model,comparison='Stratified-minus-Early',metric=m,estimate=(s-e)*scale if e is not None and s is not None else None,low_95=ci[0],high_95=ci[1],valid_replicates=len(good),unit='count' if m=='tp' else 'percentage_points'))
 save('bootstrap_block_counts.csv',blockrows);save('bootstrap_replicate_metrics.csv',rep_rows);save('bootstrap_intervals.csv',intervals)
 check('freeze_before_evaluation',read(OUT/'freeze_complete.json')['frozen_at_utc']<read(OUT/'evaluation_start.json')['started_at_utc'])
 js('validation.json',dict(status='PASS',checks=CHECK,completed_at_utc=now(),historical_files_written=False,hashes_computed=False))
 js('aggregate_counts.json',[dict(model=m,scheme=s,period=p,**c) for (m,s,p),c in aggregates.items()])
 report(aggregates,thresholds,intervals)
 print('COMPLETE',OUT,flush=True)

def report(agg,thresholds,intervals):
 labels={'benign_coverage':'Benign category coverage','attack_coverage':'Attack category coverage','singleton_error_rate':'singleton error rate','acceptance_rate':'Acceptance rate','benign_singleton_error_rate':'{Benign} error rate','attack_singleton_error_rate':'{Attack} error rate','tp':'Accepting the TP attack correctly','tp_fraction_of_true_attacks':'TP / all real Attack','benign_singleton_rate':'{Benign} Number of / examples','attack_singleton_rate':'{Attack} Number of / examples','empty_set_rate':'Number of vacuums / sample','double_set_rate':'Number of binary elements / ratio','raw_error_rate':'Raw Full acceptance error rate','raw_minus_singleton_pp':'Raw − singleton (percentage points)'}
 text='# Temporal-Stratified-400 ToN independent calibration comparison\n\nOriginal Outcome B, original gate control and historical records preserved; The experiment is a new user-authorized exploration. 27,420,000 rating / 2,742 window, excluding the entire startup and tail 260 rating. All areas are closed to the left and open to the right\n\n'
 text+='Assuming the complete startup labels is available offline experiment: without finding the labels return time record, it is not possible to verify the availability of the real labels deployment with flow time available. Each set of fit uses 400 labels; The new scheme establishes a candidate pool to read 100,000 labels, Early historical scanning 4,020 labels and labels to obtain cost can’t be called the same. New and old samples naturally overlap '+str(read(OUT/'label_use.json')['overlap_with_early'])+' It’s not.\n\n'
 text+='Fixed T1 pumps T4 for [0,25000), [25000,50000), [50000,75000) and [75000,100000) respectively, with 50 pumps for each class; Just take one PCG64(20260918). The two models share the same manifest. The threshold value before the new evaluation is freeze, 200 points of all categories are upgraded to 181 items, with no inserts. Early manifest and thresholds directly repeated.\n\n'
 text+='## Startup support\n\n| Paragraph |Benign|Attack|\n|---|---:|---:|\n'
 sup=rows(OUT/'startup_support.csv')
 for i in range(4):text+=f"|T{i+1}|{sup[2*i]['support']}|{sup[2*i+1]['support']}|\n"
 for model in thresholds:
  text+=f'\n## {model}\n\n| threshold |Early-400|Temporal-Stratified-400| threshold |\n|---|---:|---:|---:|\n '
  e=thresholds[model]['Early-400'];s=thresholds[model]['Temporal-Stratified-400']
  for name,a,b in [('q_Benign',e['q_Benign'],s['q_Benign']),('q_Attack',e['q_Attack'],s['q_Attack']),('Thresholds and',e['q_Benign']+e['q_Attack'],s['threshold_sum']),('1-q_Benign',1-e['q_Benign'],1-s['q_Benign']),('1-q_Attack',1-e['q_Attack'],1-s['q_Attack'])]:text+=f'|{name}|{a:.17g}|{b:.17g}|{b-a:+.17g}|\n'
  text+='\nFor example, the unit %The difference is a percentage point (TP is the number of items). The number of sets is listed simultaneously.\n\n|Metric|Early-400|Temporal-Stratified-400| difference |\n|---|---:|---:|---:|\n'
  em=metrics(agg[model,'Early-400','overall']);sm=metrics(agg[model,'Temporal-Stratified-400','overall'])
  for m,label in labels.items():
   ev=em[m][2];sv=sm[m][2];scale=1 if m in ['tp','raw_minus_singleton_pp'] else 100
   def fmt(v):return 'NA' if v is None else (f'{int(v):,}' if m=='tp' else f'{v*scale:.6f}')
   a=fmt(ev);b=fmt(sv)
   if m in ['benign_singleton_rate','attack_singleton_rate','empty_set_rate','double_set_rate']:a=f'{em[m][0]:,} / {a}%';b=f'{sm[m][0]:,} / {b}%'
   delta='NA' if ev is None or sv is None else f'{(sv-ev)*scale:+.6f}'
   text+=f'|{label}|{a}|{b}|{delta}|\n'
  text+='\nThe total paired bootstrap (95% percentile)Difference = Stratified − Early ):\n\n| indicator |Early 95% CI|Stratified 95% CI| paired difference 95% CI (percentage point) | effective duplicate E/S/ difference |\n|---|---|---|---|---|\n'
  for m in ['benign_coverage','attack_coverage','singleton_error_rate','acceptance_rate','tp_fraction_of_true_attacks']:
   rr=[next(r for r in intervals if r['model']==model and r['metric']==m and r['comparison']==comp) for comp in ['Early-400','Temporal-Stratified-400','Stratified-minus-Early']]
   cis=[f"[{r['low_95']*(100 if i<2 else 1):.4f}, {r['high_95']*(100 if i<2 else 1):.4f}]" if r['valid_replicates'] else 'NA' for i,r in enumerate(rr)]
   text+=f"|{labels[m]}|{'|'.join(cis)}|{' / '.join(str(r['valid_replicates']) for r in rr)}|\n"
  for m in ['benign_coverage','attack_coverage']:
   text+=f'\n### {labels[m]} Changes over time\n\n|Evaluation Period|Early Coverage|Stratified Coverage| Difference (percentage point) |\n|---|---:|---:|---:|\n '
   vals=[]
   for i,(lo,hi) in enumerate(SEG,1):
    ev=metrics(agg[model,'Early-400',f'E{i}'])[m][2]*100;sv=metrics(agg[model,'Temporal-Stratified-400',f'E{i}'])[m][2]*100;vals.append((ev,sv));text+=f'|E{i} [{lo},{hi})|{ev:.6f}%|{sv:.6f}%|{sv-ev:+.6f}|\n'
   v=np.array(vals);text+=f'\n4 extreme coverage (descriptive): Early {np.ptp(v[:,0]):.6f} pp → Stratified {np.ptp(v[:,1]):.6f} pp; Mean absolute deviation from 90% across four equally weighted segments, descriptive only:{np.mean(abs(v[:,0]-90)):.6f} → {np.mean(abs(v[:,1]-90)):.6f} pp。\n'
  text+='\nreliability and attack acceptance (%):\n\n|Period|Early risk |Stratified risk |Early acceptance rate |Stratified acceptance rate |Early TP/Attack|Stratified TP/Attack|Raw risk |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
  for i in range(1,5):
   a=metrics(agg[model,'Early-400',f'E{i}']);b=metrics(agg[model,'Temporal-Stratified-400',f'E{i}']);v=[a['singleton_error_rate'][2],b['singleton_error_rate'][2],a['acceptance_rate'][2],b['acceptance_rate'][2],a['tp_fraction_of_true_attacks'][2],b['tp_fraction_of_true_attacks'][2],a['raw_error_rate'][2]];text+=f"|E{i}|"+'|'.join('NA' if x is None else f'{100*x:.6f}' for x in v)+'|\n'
 text+='\n## definition, statistical conditions and verification\n\nCategory coverage contains a set of binary elements, not equal to acceptance rate or accuracy rate. Only labels is acceptable and permitted to differ from Raw according to the p<=q_Benign and 1-p<=q_Attack equations; Raw independent to p>=0.5. Empty cluster rejection, binary element abstention; An unknown attack cannot be directly explained as an unknown attack. The prime factor is NA. 90% covers reference for nominal categories only, without any unconditional warranty or addition of lines under drift.\n\nReuse and verify the original RF matrix, continuous 10 window block, 275 block, end block 2 window, PCG64(20260915) , 2,000; The same matrix is used for two models of two schemes, and the ratio is calculated by multiplying the pre-sum counts. The interval conditions for this fixed sample with the freeze model, without the uncertainty of re-extracting the 400 line. The section describes only, without adding any significant sexual barriers.\n\nvalidation.json all checked through: line identity sequence and window endpoints, historical sample / protocol, model identity /49 features /raw fields, for each category 50 items, sample only / two models with manifest / evaluate non-intersecting, threshold 181 items, equivalent / non-asymmetrical / vacuum / binary elements / zero acceptance, set and confusion counts, four combinations, all historical window counts with total / points historical indicators, freeze early in the evaluation, bootstrap matrix is completely consistent. The historical probability is read at float64, unchanged.\n\nA fixed random sampling provides only evidence of the operation point, cannot prove the underrepresentation of the original sample time from the threshold changes, nor can it be promoted as a general conclusion of the sampling mechanism. No change in Outcome or B and no follow-up experiments.\n\nAll the overall / fractional indicators and molecular fractions see metrics.csv; For all indicators and variances of CI and effective multiples see bootstrap_intervals.csv; The number of windows, protocols, samples, scores, labels descriptions, code, commands and logs are kept in the same directory.\n'
 (OUT/'report.md').write_text(text)
if __name__=='__main__':
 try:main()
 except Exception as e:
  js('failure.json',dict(time=now(),error=repr(e),checks=CHECK));raise
