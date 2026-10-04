from pathlib import Path
import csv,json
import numpy as np
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run');T=R/'outputs/post_p7_diagnostics/temporal_stratified_400_ton/run_20260918T131742Z'
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
p=json.loads((O/'protocol.json').read_text());checks=[]
def check(name,ok):
 assert ok,name
 checks.append(dict(check=name,passed=True))
points=rows(O/'raw_confidence_risk_coverage_points.csv');raw=rows(O/'raw_metrics.csv');lookup={(r['model'],r['period'],r['metric']):r for r in raw};history=rows(T/'window_counts.csv');context=rows(O/'existing_conformal_summary.csv');bins=rows(O/'confidence_bin_metrics.csv');Y=np.load(R/'outputs/feature_store_v2/ToN_label.npy',mmap_mode='r')
for model in p['models']:
 cache=np.load(R/f'outputs/prediction_cache_v2/ToN/{model}.npy',mmap_mode='r')
 # Whole-tie selections checked directly on cache chunks, independently of unique-value sorting implementation.
 for period,(lo,hi) in p['evaluation_periods'].items():
  targets=[r for r in points if r['model']==model and r['period']==period];acc=np.zeros((8,7),dtype=np.int64)
  for a in range(lo,hi,100000):
   prob=np.asarray(cache['raw_probability'][a:min(a+100000,hi)],dtype=np.float64);y=Y[a:min(a+100000,hi)];confidence=np.where(prob>=.5,prob,1-prob);pred=prob>=.5
   for i,t in enumerate(targets):
    q=float(t['diagnostic_confidence_boundary']);sel=confidence>=q;eq=confidence==q
    acc[i]+=np.array([sel.sum(),eq.sum(),((pred!=y)&sel).sum(),((~pred)&(y==0)&sel).sum(),(pred&(y==1)&sel).sum(),(pred&(y==0)&sel).sum(),((~pred)&(y==1)&sel).sum()],dtype=np.int64)
  for t,c in zip(targets,acc):
   check(model+period+t['target_coverage']+'_direct_whole_tie_selection',all(int(t[k])==int(v) for k,v in zip(['accepted','tie_group_n','errors','accepted_tn','accepted_tp','accepted_fp','accepted_fn'],c)) and c[0]>=float(t['target_coverage'])*(hi-lo) and c[0]-c[1]<float(t['target_coverage'])*(hi-lo))
  rr=[r for r in bins if r['model']==model and r['period']==period];allbin=[r for r in rr if r['prediction_subset']=='all'];check(model+period+'_bin_raw_totals',sum(int(r['n']) for r in allbin)==hi-lo and sum(int(r['errors']) for r in allbin)==int(lookup[model,period,'raw_error_rate']['numerator']))
  for b in p['confidence_bins']:
   a=next(r for r in rr if r['prediction_subset']=='all' and r['bin']==b);parts=[r for r in rr if r['prediction_subset']!='all' and r['bin']==b]
   check(model+period+b+'_predicted_partition_counts',all(int(a[k])==sum(int(r[k]) for r in parts) for k in ['n','errors','true_benign','true_attack','fp','fn']))
  for scheme in ['Early-400','Temporal-Stratified-400']:
   wr=[r for r in history if r['model']==model and r['scheme']==scheme and (period=='overall' or r['period']==period)];fields=['n','true_benign','true_attack','benign_singleton','attack_singleton','empty','double','tn','fn','fp','tp','accepted','singleton_errors','raw_errors','benign_covered','attack_covered'];c={k:sum(int(r[k]) for r in wr) for k in fields}
   mapping={'benign_coverage':('benign_covered','true_benign'),'attack_coverage':('attack_covered','true_attack'),'singleton_error_rate':('singleton_errors','accepted'),'acceptance_rate':('accepted','n'),'benign_singleton_error_rate':('fn','benign_singleton'),'attack_singleton_error_rate':('fp','attack_singleton'),'tp_fraction_of_true_attacks':('tp','true_attack'),'benign_singleton_rate':('benign_singleton','n'),'attack_singleton_rate':('attack_singleton','n'),'empty_set_rate':('empty','n'),'double_set_rate':('double','n'),'raw_error_rate':('raw_errors','n')}
   ctx=[r for r in context if r['model']==model and r['scheme']==scheme and r['period']==period]
   for r in ctx:
    if r['metric']=='tp':num,den=c['tp'],1
    elif r['metric']=='raw_minus_singleton_pp':num=100*(c['raw_errors']*c['accepted']-c['singleton_errors']*c['n']);den=c['n']*c['accepted']
    else:ka,kb=mapping[r['metric']];num,den=c[ka],c[kb]
    check(model+period+scheme+r['metric']+'_context_saved_counts',num==int(r['numerator']) and den==int(r['denominator']) and abs(num/den-float(r['value']))<1e-12)
 print('AUDITED',model,flush=True)
# Independent Python scalar toy checks, no production helper reused.
for prob,label,expected in [(.5,0,True),(.5,1,False),(0.,0,False),(1.,0,True)]:check('scalar_raw_boundary_'+str((prob,label)),((1 if prob>=.5 else 0)!=label)==expected)
check('zero_support_bins_NA',all(all(r[k]=='NA' for k in ['bin_fraction','error_rate','mean_confidence','accuracy','confidence_minus_accuracy']) for r in bins if int(r['n'])==0))
check('all_figures_present',len(list((O/'figures').glob('*.png')))==20 and len(list((O/'figures').glob('*.pdf')))==20)
result=dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),checks=checks,method='independent direct cache comparisons for every target and period, no grouping function reused; historic context reaggregated from saved window counts')
with (O/'delivery_validation.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('PASS',len(checks),'independent delivery checks',flush=True)
