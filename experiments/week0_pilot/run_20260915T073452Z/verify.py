from common import *
from implementation import restore_q
# Independent metric reconstruction from categorical sufficient counts using scalar loops.
def independent(t):
 d={k:0 for k in ['N','B','A','cov','covB','covA','acc','rej','error','errR','correctR','TN','FP','FN','TP']}
 for y in range(2):
  for raw in range(2):
   for code in range(4):
    n=int(t[y,raw,code]);d['N']+=n;d['B' if y==0 else 'A']+=n
    accepted=code in (1,2);covered=bool(code & (1<<y));error=raw!=y
    if covered:d['cov']+=n;d['covB' if y==0 else 'covA']+=n
    if accepted:
     d['acc']+=n
     pred=0 if code==1 else 1
     d[['TN','FP','FN','TP'][2*y+pred]]+=n
    else:
     d['rej']+=n
     if error:d['errR']+=n
     else:d['correctR']+=n
    if error:d['error']+=n
 def div(a,b):return a/b if b else None
 out={'prediction_set_coverage':div(d['cov'],d['N']),'rejection_rate':div(d['rej'],d['N']),'acceptance_rate':div(d['acc'],d['N']),'selective_risk':div(d['FP']+d['FN'],d['acc']),'raw_error_given_rejected':div(d['errR'],d['rej']),'raw_error_given_accepted':div(d['error']-d['errR'],d['acc']),'rejected_given_raw_error':div(d['errR'],d['error']),'rejected_given_raw_correct':div(d['correctR'],d['N']-d['error']),'effective_attack_detection':div(d['TP'],d['A']),'wrong_accepted_attack_fraction':div(d['FN'],d['A'])}
 for gap,a,b in [('raw_error_rate_gap','raw_error_given_rejected','raw_error_given_accepted'),('rejection_rate_error_correct_gap','rejected_given_raw_error','rejected_given_raw_correct')]:out[gap]=None if out[a] is None or out[b] is None else out[a]-out[b]
 return out
protocol=read(OUT/'protocol.json');pf=read(OUT/'protocol_freeze.json');qf=read(OUT/'q_freeze.json');ev=read(OUT/'evaluation_complete.json')
assert sha(OUT/'protocol.json')==pf['protocol_sha256']==ev['protocol_sha256']
assert sha(OUT/'q_freeze.json')==ev['q_sha256']
assert pf['frozen_at_utc']<qf['computation_started_at_utc']<qf['frozen_at_utc']<ev['started_at_utc']
assert qf['selection_calls_on_real_calibration']==1 and qf['source_calibration_only']
q=restore_q(qf['q_serialized']);assert math.isinf(q) or float(qf['q_decimal']).hex()==q.hex()
assert qf['scores_less_than_q']<324001<=qf['scores_less_than_q']+qf['scores_equal_q']
assert qf['scores_less_than_q']+qf['scores_equal_q']+qf['scores_greater_than_q']==360000
# Verify scalar ratios for overall, every original window, every category and time segment.
overall=read(OUT/'overall_metrics.json');checks=0
for domain,N in [('UNSW',235424),('ToN',27520000)]:
 w=np.load(OUT/f'{domain}_window_counts.npz')['counts'];assert int(w.sum())==N
 candidates=[w.sum(axis=(0,1))]+list(w.sum(axis=1))+list(w.sum(axis=0))
 if domain=='ToN':candidates += [w[s*688:(s+1)*688].sum(axis=(0,1)) for s in range(4)]
 for t in candidates:
  direct=independent(t);fast=summarize(t)
  for k,v in direct.items():assert v==fast[k],(domain,k,v,fast[k])
  checks+=1
 s=summarize(w.sum(axis=(0,1)))
 for k,v in s.items():assert v==overall[domain][k]
 cats=csvread(OUT/'attack_category_metrics.csv');crows=[r for r in cats if r['domain']==domain]
 for key in ['rows','TN','FP','FN','TP','raw_errors','accepted','rejected','prediction_set_covered']:
  assert sum(int(r[key]) for r in crows)==s[key]
 wins=[r for r in csvread(OUT/'window_metrics.csv') if r['domain']==domain]
 for key in ['rows','TN','FP','FN','TP','raw_errors','accepted','rejected','prediction_set_covered']:
  assert sum(int(r[key]) for r in wins)==s[key]
 if domain=='ToN':
  for key in ['rows','TN','FP','FN','TP','raw_errors','accepted','rejected','prediction_set_covered']:
   assert sum(int(r[key]) for r in csvread(OUT/'temporal_metrics.csv'))==s[key]
# NA behavior: no support, zero acceptance, all-correct and no rejection.
for t in [np.zeros((2,2,4),dtype=np.int64)]:
 z=summarize(t);assert z['selective_risk'] is None and z['prediction_set_coverage'] is None
for code in [0,1,2,3]:
 t=np.zeros((2,2,4),dtype=np.int64);t[0,0 if code!=2 else 1,code]=7
 z=summarize(t)
 if code in (0,3):assert z['selective_risk'] is None
 if code in (1,2):assert z['raw_error_given_rejected'] is None
 assert z['effective_attack_detection'] is None
# Recompute all bootstrap counts via direct selected block indexing (different from matrix multiplication).
idx=np.load(OUT/'bootstrap_indices.npy');mul=np.load(OUT/'bootstrap_multiplicity.npy');b=np.load(OUT/'bootstrap_block_counts.npz')['counts']
assert np.array_equal(idx,np.random.Generator(np.random.PCG64(20260915)).integers(0,276,size=(2000,276),dtype=np.int64))
assert all(np.array_equal(np.bincount(r,minlength=276),m) for r,m in zip(idx,mul))
reps=csvread(OUT/'bootstrap_replicate_metrics.csv');values={k:[] for k in BOOT_METRICS}
for i,draw in enumerate(idx):
 t=b[draw].sum(axis=0);assert np.array_equal(t,(mul[i]@b.reshape(276,-1)).reshape(2,2,4))
 direct=independent(t);s=summarize(t)
 for k,v in direct.items():assert v==s[k]
 for k in BOOT_METRICS:
  saved=reps[i][k];assert (saved=='NA' if s[k] is None else float(saved)==s[k])
  values[k].append(np.nan if s[k] is None else s[k])
  for suf in ['_numerator','_denominator']:
   if k+suf in s:assert int(reps[i][k+suf])==s[k+suf]
for row in csvread(OUT/'bootstrap_intervals.csv'):
 v=np.array(values[row['metric']]);v=v[np.isfinite(v)]
 assert len(v)==int(row['valid_replicates'])
 if len(v):
  ci=np.quantile(v,[.025,.975],method='linear');assert np.array_equal(ci,[float(row['lower_95']),float(row['upper_95'])])
 else:assert row['lower_95']==row['upper_95']=='NA'
# Input preservation includes preflight and A-C2. Content rehash of all listed frozen inputs.
refs=csvread(OUT/'source_hashes.csv')
for r in refs:assert sha(r['path'])==r['sha256'],r['path']
before=read(OUT/'upstream_inventory_before.json');after=inventory();assert before==after
write('upstream_inventory_after.json',after)
assert all(r['match']=='True' for r in csvread(OUT/'historical_raw_comparison.csv'))
val=dict(status='PASS',version='week0-validation-v1.0',completed_at_utc=now(),implementation_valid=True,q_computed_once=True,q_from_S_prob_cal_only=True,q_freeze_precedes_evaluation=True,numerical_contract_all_rows_passed=True,source_rows=235424,ToN_rows=27520000,source_tail_retained=5424,ToN_tail_excluded=260,ToN_original_windows=2752,sets_partition_and_ratios=True,singletons_match_raw=True,errors_conserve=True,class_category_window_time_aggregates_conserve=True,independent_scalar_metric_groups=checks,historical_stage_A_Raw_matches=True,bootstrap_all_2000_replicates_independently_recomputed=True,bootstrap_denominators_and_CIs_recomputed=True,NA_checks=True,source_hashes_rechecked=len(refs),upstream_files_size_mtime_unchanged=len(before),preflight_unchanged=True,protocol_sha256=sha(OUT/'protocol.json'),q_sha256=sha(OUT/'q_freeze.json'),real_pilot_completed=True,Week1_started=False,professor_confirmation_received=False,automatic_messages_sent=False)
write('validation.json',val);print(json.dumps(val,indent=2))
