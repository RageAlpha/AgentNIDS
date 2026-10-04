from pathlib import Path
import csv,json,numpy as np
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
ROOT=Path('/home/karri/projects/NIDSProject_Professor_Run')
def load(n):return json.loads((P/n).read_text())
def rows(n):return list(csv.DictReader((P/n).open()))
checks=[]
def ck(name,condition):
 assert condition,name
 checks.append(dict(check=name,passed=True))
sample=rows('calibration_sample_manifest.csv');rank=np.array([int(r['rank']) for r in sample]);labels=np.array([int(r['label']) for r in sample]);ck('final_manifest_sorted',np.all(np.diff(rank)>0))
for model in ['rf_All49','xgb_All49']:
 score=rows(model+'_calibration_scores.csv');q=load(model+'_threshold_freeze.json')['Temporal-Stratified-400'];cache=np.load(ROOT/f'outputs/prediction_cache_v2/ToN/{model}.npy',mmap_mode='r');p=cache['raw_probability'][rank].astype(np.float64)
 ck(model+'_saved_shared_identity',all(all(r[k]==s[k] for k in s) for r,s in zip(score,sample)))
 ck(model+'_saved_raw_probabilities_exact',np.array_equal(p,np.array([float(r['raw_probability']) for r in score])))
 ss=np.where(labels==0,p,1-p);ck(model+'_saved_scores_exact',np.array_equal(ss,np.array([float(r['true_class_score']) for r in score])))
 ck(model+'_saved_thresholds_exact',all(np.sort(ss[labels==k])[180]==q[name] for k,name in enumerate(['q_Benign','q_Attack'])))
ck('all_metric_valid_replicates',all(int(r['valid_replicates'])==2000 for r in rows('bootstrap_intervals.csv')))
met=rows('metrics.csv');lookup={(r['model'],r['scheme'],r['period'],r['metric']):r for r in met}
def v(model,scheme,metric,period='overall'):return float(lookup[model,scheme,period,metric]['value'])
E='Early-400';S='Temporal-Stratified-400'
def pv(model,scheme,metric,period='overall'):return 100*v(model,scheme,metric,period)
text='\n## answers to six research questions\n\n'
text+='1. ** threshold change **: q_Benign rise of RF, q_Attack decrease; The q_Benign of XGBoost is exactly the same, with a slight decrease in q_Attack. For details, see the threshold table; Both new q_Attack values remain greater than 0.90; continue the authorized fixed evaluation.\n\n'
text+='2. ** nominal coverage with time stability ** : RF covers both classes more closely to 90%, but cannot be said to improve time stability overall. The four phases of the Benign decrease, while the Attack decrease increases. E2 Attack covers from '+f"{pv('rf_All49',E,'attack_coverage','E2'):.4f}% decreases to {pv('rf_All49',S,'attack_coverage','E2'):.4f}%"+'E4 Benign is still significantly lower than 90%. XGBoost coverage of Benign is completely unchanged; The overall coverage of Attack is slightly lower, further away from 90%, and E4 is still about 70%. The 90% deviation is a descriptive summary, not a new threshold.\n\n'
text+='3. ** risk and acceptance rate **:'
for m in ['rf_All49','xgb_All49']:
 text+=f"{m} singleton error rate {pv(m,E,'singleton_error_rate'):.4f}%→{pv(m,S,'singleton_error_rate'):.4f}%, acceptance rate {pv(m,E,'acceptance_rate'):.4f}%→{pv(m,S,'acceptance_rate'):.4f}%。"
text+='the overall cost of acceptance of the rate decrease; However, the decrease in the RF E1 acceptance rate does not extend the overall rise to all periods.\n\n'
text+=f" 4. ** correctly responds to the ** attack: RF TP {int(v('rf_All49',E,'tp')):,}→{int(v('rf_All49',S,'tp')):,} TP/ is all real Attack. {pv('rf_All49',E,'tp_fraction_of_true_attacks'):.4f}%→{pv('rf_All49',S,'tp_fraction_of_true_attacks'):.4f}%; XGBoost TP and the ratio remained unchanged. The RF cost reflects the correct acceptance of the attack rather than the overall acceptance of the number.\n\n "
text+='5. **Temporal or local:** RF risks decrease in all four segments, but gains vary. E1 TP/Attack decreases from '+f"{pv('rf_All49',E,'tp_fraction_of_true_attacks','E1'):.4f}% to {pv('rf_All49',S,'tp_fraction_of_true_attacks','E1'):.4f}%"+'; the other three also decrease. XGBoost E1–E3 risk decreases but E4 rises slightly; Benign coverage is unchanged and Attack coverage falls in every segment, so improvement is not temporally consistent.\n\n'
text+='6. ** is still inferior to Raw in relation to **:'
for m in ['rf_All49','xgb_All49']:
 row=next(r for r in rows('bootstrap_intervals.csv') if r['model']==m and r['comparison']==S and r['metric']=='raw_minus_singleton_pp')
 text+=f"{m} The new singleton risk is higher than the Raw model.{-v(m,S,'raw_minus_singleton_pp'):.6f} One percentage point; Raw − singleton with a 95% interval of [{float(row['low_95']):.4f}, {float(row['high_95']):.4f}] percentage points."
text+='Overall, the two points are estimated to still have a risk degradation, but the range is between zero and it cannot be asserted that the overall difference has been accurately determined; RF E1 E3 risk is still higher than Raw, XGBoost E1 and E4 are still higher than Raw.\n\n'
text+='This result supports an improvement in the acceptability of risk at the fixed operating point RF, while exposing the loss of the correct acceptance of the attack; It is not supported by the general conclusion that the two models, the two classes, are more stable and reliable at any given time. The original Outcome B was retained and stopped after completion.\n'
# Full per-period table, every requested metric; counts and denominators exposed.
for model in ['rf_All49','xgb_All49']:
 for period in ['E1','E2','E3','E4']:
  text+=f'\nAppendix ## {model} / {period}\n\n|Metric|Early-400 (molecular / molecule); The value is |Temporal-Stratified-400 (molecule / molecule); Value) | Difference |\n|---|---:|---:|---:|\n '
  for r in met:
   if (r['model'],r['scheme'],r['period'])!=(model,E,period):continue
   s=lookup[model,S,period,r['metric']];scale=100 if r['unit']=='fraction' else 1;unit='%' if scale==100 else ('pp' if r['unit']=='percentage_points' else 'records')
   def fmt(x):return f"{x['numerator']}/{x['denominator']}；"+('NA' if x['value']=='NA' else f"{float(x['value'])*scale:.6f}{unit}")
   diff='NA' if 'NA' in [r['value'],s['value']] else f"{(float(s['value'])-float(r['value']))*scale:+.6f}"+('records' if r['unit']=='count' else 'pp')
   text+=f"|{r['metric']}|{fmt(r)}|{fmt(s)}|{diff}|\n"
with (P/'report.md').open('a') as f:f.write(text)
with (P/'delivery_validation.json').open('x') as f:json.dump(dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),checks=checks,original_validation_checks=len(load('validation.json')['checks'])),f,indent=2)
with (P/'output_inventory.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['absolute_path','bytes']);w.writeheader()
 for p in sorted(P.iterdir()):
  if p.is_file() and p.name!='output_inventory.csv':w.writerow(dict(absolute_path=str(p),bytes=p.stat().st_size))
print('PASS:',len(checks),'delivery checks;',len(load('validation.json')['checks']),'execution checks; no calibration resampling or threshold refitting')
