from common import *
from implementation import prediction_sets,restore_q
assert not (OUT/'evaluation_complete.json').exists(),'evaluation already complete'
pf=read(OUT/'protocol_freeze.json'); assert sha(OUT/'protocol.json')==pf['protocol_sha256']
qf=read(OUT/'q_freeze.json');q=restore_q(qf['q_serialized']);qhash=sha(OUT/'q_freeze.json')
assert qf['protocol_sha256']==pf['protocol_sha256']
protocol=read(OUT/'protocol.json');start=now();assert qf['frozen_at_utc']<start
write('evaluation_start.json',dict(started_at_utc=start,q_frozen_at_utc=qf['frozen_at_utc'],q_sha256=qhash,threshold_selection_called=False))
windows=csvread(ROOT/'outputs/round1_data_prep_v1/window_manifest.csv')
overall=[];category=[];temporal=[];temporal_categories=[];window_rows=[];setrows=[];strata=[];summaries={};schema={};checks={}
for domain,lo,hi,scheme,part in [('UNSW',2130000,2365424,'source','S_source_test'),('ToN',0,27520000,'heldout','T_heldout')]:
 inp=protocol['inputs'][domain]; cache=np.load(inp['probability_cache'],mmap_mode='r'); labels=np.load(inp['labels'],mmap_mode='r');cats=np.load(inp['attack_codes'],mmap_mode='r')
 assert cache.dtype['raw_probability']==np.float64
 ww=[w for w in windows if w['domain']==domain and w['scheme']==scheme and w['partition']==part and w['window_size']=='10000']
 intervals=[(int(w['rank_start']),int(w['rank_end_exclusive']),w['window_id']) for w in ww]
 if domain=='UNSW':intervals.append((2360000,2365424,None)) # retained original partition tail; not a new window
 assert intervals[0][0]==lo and intervals[-1][1]==hi and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
 K=len(inp['attack_codebook']); counts=np.zeros((len(intervals),K,2,2,4),dtype=np.int64)
 raw_direct=np.zeros(4,dtype=np.int64);local_checks=0;prob_rows=0
 for wi,(a,b,wid) in enumerate(intervals):
  p=np.asarray(cache['raw_probability'][a:b])
  # prediction_sets validates finite/unit interval/exact float32 origin, without alteration.
  c=prediction_sets(p,q); code=c[:,0].astype(np.int64)+2*c[:,1].astype(np.int64)
  # Labels are accessed only after the corresponding sets exist.
  y=np.asarray(labels[a:b]);cat=np.asarray(cats[a:b]);raw=(p>=.5).astype(np.int64)
  assert np.isin(y,[0,1]).all() and ((cat>=0)&(cat<K)).all()
  assert all(np.all(y[cat==int(k)]==(0 if name=='Benign' else 1)) for k,name in inp['attack_codebook'].items())
  singleton=(code==1)|(code==2)
  assert np.array_equal((code[singleton]==2).astype(int),raw[singleton])
  if q<.5:assert not (code==3).any()
  elif q>.5:assert not (code==0).any()
  else:assert np.array_equal(code==3,p==.5) and not (code==0).any()
  packed=(((cat.astype(np.int64)*2+y)*2+raw)*4+code)
  counts[wi]=np.bincount(packed,minlength=K*16).reshape(K,2,2,4)
  raw_direct+=np.bincount(y*2+raw,minlength=4)
  assert int(counts[wi].sum())==b-a
  # Direct per-sample checks at fixed deterministic ranks of every window and tail.
  ix=np.unique(np.linspace(0,len(p)-1,min(17,len(p)),dtype=int))
  oracle=np.array([[float(p[j])<=q,1.0-float(p[j])<=q] for j in ix]);assert np.array_equal(c[ix],oracle)
  local_checks+=len(ix);prob_rows+=len(p)
  wr=dict(domain=domain,window_index=wi,window_id=wid,rank_start=a,rank_end_exclusive=b,complete_window=wid is not None,**summarize(counts[wi].sum(0)))
  window_rows.append(wr)
  if wi%500==0:print(domain,'windows',wi,'/',len(intervals),flush=True)
 total=counts.sum(axis=(0,1));r=dict(domain=domain,**summarize(total));overall.append(r);summaries[domain]=r
 assert [r['raw_'+x] for x in ['TN','FP','FN','TP']]==raw_direct.tolist()
 assert r['rows']==hi-lo
 for yy,name in [(0,'Benign'),(1,'Attack')]:
  t=np.zeros_like(total);t[yy]=total[yy];strata.append(dict(domain=domain,true_class=name,**summarize(t)))
 for k,name in inp['attack_codebook'].items():category.append(dict(domain=domain,category_code=int(k),category=name,**summarize(counts[:,int(k)].sum(0))))
 for code,name in enumerate(['{}','{Benign}','{Attack}','{Benign,Attack}']):
  n=int(total[:,:,code].sum());e=int(total[0,1,code]+total[1,0,code]);setrows.append(dict(domain=domain,set_code=code,prediction_set=name,records=n,raw_correct=n-e,raw_errors=e,raw_error_rate=e/n if n else None,raw_error_rate_numerator=e,raw_error_rate_denominator=n))
 if domain=='ToN':
  for seg in range(4):
   a,b=seg*688,(seg+1)*688;t=counts[a:b].sum(axis=(0,1))
   temporal.append(dict(domain=domain,segment=seg+1,first_window_index=a,last_window_index_exclusive=b,rank_start=a*10000,rank_end_exclusive=b*10000,**summarize(t)))
   for k,name in inp['attack_codebook'].items():temporal_categories.append(dict(domain=domain,segment=seg+1,category_code=int(k),category=name,**summarize(counts[a:b,int(k)].sum(0))))
 np.savez_compressed(OUT/f'{domain}_window_counts.npz',counts=counts)
 schema[domain]=dict(shape=list(counts.shape),axes=['original_window_or_retained_source_tail','original_category_code','true_label','raw_pred','set_code'],set_codes={'0':'{}','1':'{Benign}','2':'{Attack}','3':'{Benign,Attack}'},taxonomy=inp['attack_codebook'],source_tail_is_window=False)
 checks[domain]=dict(rows_numerical_contract_checked=prob_rows,vector_scalar_fixed_samples=local_checks,raw_independent_counts=raw_direct.tolist(),labels_used_after_sets=True)
# Historical comparator from Stage A, same full scopes and Raw only.
history=[]
for r in csvread(ROOT/'outputs/post_p7_diagnostics/stage_A/run_20260914_A1/detector_raw_platt_metrics.csv'):
 if r['domain'] in summaries and r['model']=='rf_All49' and r['output']=='Raw':
  cur=summaries[r['domain']]
  for col in ['TN','FP','FN','TP']:
   assert cur['raw_'+col]==int(r[col]);history.append(dict(domain=r['domain'],field=col,stage_A=int(r[col]),pilot_raw=cur['raw_'+col],match=True))
assert len(history)==8
csvwrite('historical_raw_comparison.csv',history)
csvwrite('overall_metrics.csv',overall)
setkeys=['empty_count','benign_singleton_count','attack_singleton_count','multi_label_count','singleton_rate','acceptance_rate','multi_label_rate','empty_set_rate','mean_set_size','prediction_set_coverage','benign_prediction_set_coverage','attack_prediction_set_coverage']
rejkeys=['raw_errors','raw_correct','rejected','accepted','rejected_raw_errors','accepted_raw_errors','rejected_raw_correct','rejection_rate','raw_error_given_rejected','raw_error_given_accepted','rejected_given_raw_error','rejected_given_raw_correct','raw_error_rate_gap','rejection_rate_error_correct_gap']
selkeys=['TN','FP','FN','TP','raw_TN','raw_FP','raw_FN','raw_TP','selective_risk','benign_acceptance_rate','attack_acceptance_rate','accepted_benign_FPR','accepted_attack_miss_rate','effective_attack_detection','wrong_accepted_attack_fraction','attack_abstention_rate','raw_error_rate','raw_attack_recall','raw_benign_FPR']
csvwrite('prediction_set_metrics.csv',[table_fields(r,setkeys) for r in overall]);csvwrite('rejection_diagnostics.csv',[table_fields(r,rejkeys) for r in overall]);csvwrite('selective_metrics.csv',[table_fields(r,selkeys) for r in overall]);csvwrite('set_type_error_counts.csv',setrows)
csvwrite('attack_category_metrics.csv',category);csvwrite('class_metrics.csv',strata);csvwrite('temporal_metrics.csv',temporal);csvwrite('temporal_attack_category_metrics.csv',temporal_categories);csvwrite('window_metrics.csv',window_rows)
write('aggregation_schema.json',schema);write('overall_metrics.json',summaries)
write('evaluation_complete.json',dict(status='PASS',started_at_utc=start,completed_at_utc=now(),q_sha256=qhash,protocol_sha256=pf['protocol_sha256'],checks=checks,q_unchanged=sha(OUT/'q_freeze.json')==qhash,historical_Raw_counts_match=True,code_hashes={n:sha(OUT/n) for n in ['evaluate.py','implementation.py','common.py']},source_target_rejection_gap_descriptive=summaries['ToN']['rejection_rate']-summaries['UNSW']['rejection_rate']))
assert sha(OUT/'q_freeze.json')==qhash
print(json.dumps(summaries,indent=2))
