"""Independent reference from joint histograms of original CSV labels and probabilities.
No import, execution or function reuse from the original pilot. Percentiles implemented
by sorted-order linear interpolation; bootstrap aggregates through block multiplicities.
"""
import csv,json,math,sys,re
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path('/home/karri/projects/NIDSProject_Professor_Run');O=Path(__file__).resolve().parent
OLD=R/'outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z'
SEG=[(100000,6960000),(6960000,13820000),(13820000,20670000),(20670000,27520000)]
def js(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def write(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def save(n,r):
 if not r:r=[dict(status='NO_DIFFERENCES')]
 with (O/n).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
DIFF=[];COMPARISONS=[]
def compare(context,expected,actual):
 mismatches=0;examined=0
 for key,e in expected.items():
  a=actual.get(key,'MISSING');examined+=1
  if isinstance(e,(int,np.integer)):ok=str(int(e))==str(a)
  elif isinstance(e,(float,np.floating)):
   try:ok=float(e)==float(a)
   except (ValueError,TypeError):ok=False
  else:ok=str(e)==str(a)
  if not ok:
   mismatches+=1;DIFF.append(dict(context=context,field=key,reference=str(e),original=str(a)))
 COMPARISONS.append(dict(context=context,checked_fields=examined,differences=mismatches))
 return mismatches

def decode(h):
 # h indexes set-membership bits, raw CSV class, Raw 0.5 prediction.
 cube=np.asarray(h,dtype=np.int64).reshape(4,2,2)
 by_set=cube.sum(axis=(1,2)); by_truth=cube.sum(axis=(0,2))
 confusion=cube[[1,2]].sum(axis=2)
 tn,fn,fp,tp=map(int,confusion.ravel())
 return dict(n=int(cube.sum()),true_benign=int(by_truth[0]),true_attack=int(by_truth[1]),
  benign_singleton=int(by_set[1]),attack_singleton=int(by_set[2]),empty=int(by_set[0]),double=int(by_set[3]),
  tn=tn,fn=fn,fp=fp,tp=tp,accepted=tn+fn+fp+tp,singleton_errors=fn+fp,
  raw_errors=int(cube[:,0,1].sum()+cube[:,1,0].sum()),
  singleton_raw_disagreement=int(cube[1,:,1].sum()+cube[2,:,0].sum()),
  benign_covered=int(cube[[1,3],0,:].sum()),attack_covered=int(cube[[2,3],1,:].sum()))

def ratio(n,d):return n/d if d else 'NA'
def references(scope,c,tp):
 values={'benign_singleton_rate':(c['tn']+c['fn'],c['n']),'attack_singleton_rate':(c['tp']+c['fp'],c['n']),
  'empty_set_rate':(c['empty'],c['n']),'double_set_rate':(c['double'],c['n']),
  'benign_singleton_error_rate':(c['fn'],c['tn']+c['fn']),'attack_singleton_error_rate':(c['fp'],c['tp']+c['fp']),
  'tp':(c['tp'],1),'tp_fraction_of_true_attacks':(c['tp'],c['true_attack']),
  'acceptance_rate':(c['accepted'],c['n']),'singleton_error_rate':(c['fn']+c['fp'],c['accepted']),
  'raw_error_rate':(c['raw_errors'],c['n']),'tp_share_of_overall':(c['tp'],tp),
  'raw_minus_singleton_pp':(100*(c['raw_errors']*c['accepted']-(c['fn']+c['fp'])*c['n']),c['n']*c['accepted'])}
 return [dict(scope=scope,metric=k,numerator=n,denominator=d,value=ratio(n,d),unit='count' if k=='tp' else ('percentage_points' if k=='raw_minus_singleton_pp' else 'fraction')) for k,(n,d) in values.items()]

def scalar_membership(p,qb,qa):
 candidates=[]
 for label in (0,1):
  score=float(p) if label==0 else 1.-float(p)
  if score<=[qb,qa][label]:candidates.append(label)
 return candidates

def threshold_checks():
 cal=rows(O/'calibration_reference.csv');frozen=js(OLD/'threshold_freeze.json');report=[];result={};bands=[]
 for label,name in [(0,'Benign'),(1,'Attack')]:
  cl=[r for r in cal if int(r['true_label'])==label];assert len(cl)==200
  ordered=sorted(cl,key=lambda r:float(r['nonconformity_score']))
  q=float(ordered[180]['nonconformity_score']);result['q_'+name]=q
  # Independently evaluate 1-P(true class) directly from cache probability.
  for r in cl:
   p=float(r['original_raw_probability']);score=1.-(1.-p) if label==0 else 1.-p
   compare(f'calibration_score/rank{r["rank"]}',{'nonconformity_score':score},r)
  for i in range(178,183):bands.append(dict(class_name=name,one_based_position=i+1,score=ordered[i]['nonconformity_score'],rank=ordered[i]['rank'],original_row=ordered[i]['original_row']))
  scores=[float(r['nonconformity_score']) for r in cl]
  fresh=sorted(float(r['reinferred_score']) for r in cl if r['reinferred_score']!='NA')
  report.append(dict(class_name=name,n=200,order_statistic=181,cache_q=q,registered_q=frozen['q_'+name],exact_match=q==frozen['q_'+name],
   below=sum(s<q for s in scores),equal=sum(s==q for s in scores),above=sum(s>q for s in scores),
   reinferred_q=fresh[180] if len(fresh)==200 else 'NOT_VERIFIED',reinferred_q_matches_cache=(fresh[180]==q) if len(fresh)==200 else 'NOT_VERIFIED'))
 result['threshold_sum']=result['q_Benign']+result['q_Attack'];result['min_p_Benign']=1.-result['q_Benign'];result['min_p_Attack']=1.-result['q_Attack']
 compare('five_frozen_threshold_values',result,frozen)
 save('threshold_order_neighborhood.csv',bands)
 write('threshold_comparison.json',dict(status='PASS' if all(result[k]==frozen[k] for k in result) else 'FAIL',values=result,classes=report,
  original_thresholds_used_for_all_replay=True,reinferred_thresholds_for_diagnostic_only=True))
 probes=[]
 for qb,qa in [(result['q_Benign'],result['q_Attack']),(.25,.25),(.75,.75),(.25,.625),(.5,.5)]:
  for p in sorted(set([0.,1.,.5,.375,qb,1.-qa,np.nextafter(qb,0.),np.nextafter(qb,1.),np.nextafter(1.-qa,0.),np.nextafter(1.-qa,1.)])):
   labels=scalar_membership(p,qb,qa)
   probes.append(dict(p=p,q_Benign=qb,q_Attack=qa,prediction_set=str(labels),cardinality=len(labels),
    singleton_prediction=labels[0] if len(labels)==1 else 'NA',raw_prediction=int(p>=.5)))
 assert scalar_membership(.25,.25,.25)==[0]
 assert scalar_membership(.75,.25,.25)==[1]
 assert scalar_membership(.5,.25,.25)==[]
 assert scalar_membership(.5,.75,.75)==[0,1]
 assert scalar_membership(.375,.25,.625)==[1]
 assert ratio(0,0)=='NA'
 save('boundary_reference.csv',probes)
 return result

def main():
 threshold=threshold_checks();H=np.load(O/'reference_histograms.npz')
 hist=H['ToN_cache'];src=H['UNSW_cache'];assert hist.shape==(2742,16) and src.shape==(24,16)
 wc=[];source_windows=[]
 manifest=rows(R/'outputs/round1_data_prep_v1/window_manifest.csv')
 lookup={int(r['rank_start']):r for r in manifest if r['domain']=='ToN' and r['scheme']=='heldout' and r['window_size']=='10000'}
 for i,h in enumerate(hist):
  a=100000+i*10000;b=a+10000;segment=1+next(j for j,(aa,bb) in enumerate(SEG) if aa<=a<bb)
  wc.append(dict(evaluation_window_index=i,window_id=lookup[a]['window_id'],rank_start=a,rank_end_exclusive=b,segment=segment,**decode(h)))
 oldwc=rows(OLD/'window_counts.csv');assert len(oldwc)==len(wc)
 for ref,old in zip(wc,oldwc):compare('window/'+str(ref['evaluation_window_index']),ref,old)
 save('reference_window_counts.csv',wc)
 sums=[hist.sum(axis=0)]+[hist[(a-100000)//10000:(b-100000)//10000].sum(axis=0) for a,b in SEG]
 scopes=['overall']+[f'segment_{i}' for i in range(1,5)]
 total=[dict(scope=s,**decode(h)) for s,h in zip(scopes,sums)]
 for ref,old in zip(total,rows(OLD/'aggregate_counts.csv')):compare('aggregate/'+ref['scope'],ref,old)
 save('reference_aggregate_counts.csv',total)
 metrics=[r for c in total for r in references(c['scope'],c,total[0]['tp'])]
 oldmetrics=rows(OLD/'evaluation_metrics.csv')+rows(OLD/'temporal_metrics.csv')
 assert len(metrics)==len(oldmetrics)==65
 for ref,old in zip(metrics,oldmetrics):compare('metric/'+ref['scope']+'/'+ref['metric'],ref,old)
 save('reference_metrics.csv',metrics)
 sl={int(r['rank_start']):r for r in manifest if r['domain']=='UNSW' and r['scheme']=='source' and r['window_size']=='10000'}
 for i,h in enumerate(src):
  a=2130000+i*10000;b=min(a+10000,2365424)
  source_windows.append(dict(window_id=sl[a]['window_id'] if i<23 else 'NA (original incomplete tail; no complete-window ID)',rank_start=a,rank_end_exclusive=b,is_tail=str(i==23),**decode(h)))
 oldsource=rows(OLD/'source_window_counts.csv');assert len(oldsource)==24
 for ref,old in zip(source_windows,oldsource):compare('source_window/'+str(ref['rank_start']),ref,old)
 save('reference_source_window_counts.csv',source_windows)
 sc=decode(src.sum(axis=0));coverage=[]
 for y,name in [(0,'Benign'),(1,'Attack')]:
  num=sc['benign_covered' if y==0 else 'attack_covered'];den=sc['true_benign' if y==0 else 'true_attack']
  coverage.append(dict(domain='UNSW',class_label=y,class_name=name,numerator=num,denominator=den,value=ratio(num,den),rank_start=2130000,rank_end_exclusive=2365424))
 for ref,old in zip(coverage,rows(OLD/'source_class_coverage.csv')):compare('source_coverage/'+ref['class_name'],ref,old)
 save('reference_source_class_coverage.csv',coverage)
 impact=[]
 for domain in ['ToN','UNSW']:
  a=H[domain+'_cache'];b=H[domain+'_reinferred'];ca=decode(a.sum(axis=0));cb=decode(b.sum(axis=0))
  impact.append(dict(domain=domain,window_joint_histogram_differences=int(np.count_nonzero(a!=b)),
   count_differences={k:cb[k]-ca[k] for k in ca},status='PASS' if np.array_equal(a,b) else 'FAIL',
   reinference_status=js(O/'environment.json')['status']))
 write('reinference_metric_impact.json',impact)
 blockhist=np.add.reduceat(hist,np.arange(0,2742,10),axis=0);assert blockhist.shape==(275,16)
 blocks=[]
 for i,h in enumerate(blockhist):
  a=i*10;stop=min(a+10,2742)
  blocks.append(dict(block=i,first_evaluation_window=a,window_count=stop-a,rank_start=100000+a*10000,rank_end_exclusive=100000+stop*10000,**decode(h)))
 for ref,old in zip(blocks,rows(OLD/'bootstrap_block_counts.csv')):compare('bootstrap_block/'+str(ref['block']),ref,old)
 save('reference_block_counts.csv',blocks)
 index=np.load(OLD/'bootstrap_indices.npy',allow_pickle=False)
 rng=np.random.Generator(np.random.PCG64(20260915));regenerated=rng.integers(low=0,high=275,size=(2000,275),dtype=np.int64)
 assert index.shape==(2000,275)
 matrix_diff=int(np.count_nonzero(index!=regenerated))
 # Count multiplicities, then matrix multiply integer joint-state block counts.
 multiplicity=np.array([np.bincount(draw,minlength=275) for draw in regenerated],dtype=np.int64)
 bh=multiplicity@blockhist
 replicas=[]
 for i,h in enumerate(bh):
  c=decode(h);raw=ratio(c['raw_errors'],c['n']);risk=ratio(c['singleton_errors'],c['accepted'])
  delta=(raw-risk)*100 if risk!='NA' else 'NA'
  replicas.append(dict(replicate=i,n=c['n'],accepted=c['accepted'],raw_errors=c['raw_errors'],singleton_errors=c['singleton_errors'],
   raw_error_rate=raw,singleton_error_rate=risk,raw_minus_singleton_pp=delta))
 for ref,old in zip(replicas,rows(OLD/'bootstrap_replicate_metrics.csv')):compare('bootstrap_replicate/'+str(ref['replicate']),ref,old)
 save('reference_bootstrap_replicates.csv',replicas)
 def percentile(v,q):
  a=sorted(v);position=(len(a)-1)*q;lo=math.floor(position);hi=math.ceil(position);f=position-lo
  # NumPy linear interpolation chooses the nearer endpoint for numerical stability.
  return a[lo]+(a[hi]-a[lo])*f if f<.5 else a[hi]-(a[hi]-a[lo])*(1-f)
 point={r['metric']:r['value'] for r in metrics if r['scope']=='overall'};intervals=[]
 # Original interval point difference uses the floating risk subtraction expression.
 point['raw_minus_singleton_pp']=(point['raw_error_rate']-point['singleton_error_rate'])*100
 for metric in ['raw_error_rate','singleton_error_rate','raw_minus_singleton_pp']:
  v=[r[metric] for r in replicas if r[metric]!='NA']
  intervals.append(dict(metric=metric,estimate=point[metric],lower_95=percentile(v,.025),upper_95=percentile(v,.975),
   valid_replicates=len(v),total_replicates=2000,unit='percentage_points' if metric.endswith('_pp') else 'fraction'))
 for ref,old in zip(intervals,rows(OLD/'bootstrap_intervals.csv')):compare('bootstrap_interval/'+ref['metric'],ref,old)
 save('reference_bootstrap_intervals.csv',intervals)
 bcomp=[dict(item='sampling_matrix',comparisons=index.size,differences=matrix_diff,note='regenerated PCG64(20260915); exact element comparison'),
  dict(item='blocks',comparisons=275,differences=sum(r['differences'] for r in COMPARISONS if r['context'].startswith('bootstrap_block/')),note='joint histograms reduced over adjacent ten windows; last two'),
  dict(item='replicates',comparisons=2000,differences=sum(r['differences'] for r in COMPARISONS if r['context'].startswith('bootstrap_replicate/')),note='integer block multiplicities @ reference histograms; paired ratios'),
  dict(item='percentile_intervals',comparisons=3,differences=sum(r['differences'] for r in COMPARISONS if r['context'].startswith('bootstrap_interval/')),note='independent sorted linear interpolation; per-replicate differences')]
 save('bootstrap_comparison.csv',bcomp)
 c1=[r for r in rows(R/'outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/temporal_scope.csv') if r['domain']=='ToN']
 for i,((a,b),old) in enumerate(zip(SEG,c1),1):compare('C1_segment/'+str(i),dict(rank_start=a,rank_end_exclusive=b,windows=(b-a)//10000),old)
 sample=js(O/'sample_alignment.json');original_sample=js(OLD/'collection_summary.json')
 compare('collection',dict(scanned_rows=sample['scan_rows'],stop_rank_inclusive=sample['stop_rank'],total_collected=400,collected_benign=sample['counts'][0],collected_attack=sample['counts'][1]),original_sample)
 compare('prefix_support',dict(Benign=sample['prefix_benign'],Attack=sample['prefix_attack']),original_sample['scanned_prefix_support'])
 previous=[r for r in rows(R/'outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/calibration_sample_manifest.csv') if r['domain']=='ToN']
 cal=rows(O/'calibration_reference.csv');oldset={int(r['original_row']) for r in previous};newset={int(r['original_row']) for r in cal}
 write('historical_sample_comparison.json',dict(old_rows=len(oldset),current_rows=len(newset),intersection=len(oldset&newset),same_set=oldset==newset,
  interpretation='Incidental shared rows do not imply sample reuse. Independently reconstructed chronological first200-per-class exactly matches current manifest; original preparation code reads no C1/C2 sample.'))
 report_check(total,coverage,intervals,threshold,sample)
 save('metric_differences.csv',DIFF)
 save('comparison_summary.csv',COMPARISONS)
 write('reference_complete.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),checked_fields=sum(c['checked_fields'] for c in COMPARISONS),
  differences=len(DIFF),sampling_matrix_differences=matrix_diff,overall=total[0],temporal=total[1:],source=sc,intervals=intervals,
  zero_denominator_NA_checked=True,prediction_set_boundary_checks=True))
 print('Reference comparison complete:',len(DIFF),'field differences;',matrix_diff,'sampling matrix differences',flush=True)


def report_check(total,coverage,intervals,threshold,sample):
 stage=(OLD/'stage_report.md').read_text();gate=(OLD/'gate_assessment.md').read_text();c=total[0];seg=total[1:]
 specs=[('Singleton {Benign} rate','benign_singleton','n'),('Singleton {Attack} rate','attack_singleton','n'),('Empty-set rate','empty','n'),('Two-label rate','double','n'),
  ('Error among {Benign} singletons','fn','benign_singleton'),('Error among {Attack} singletons','fp','attack_singleton'),('Correctly accepted attacks TP','tp',None),
  ('TP / all true attacks','tp','true_attack'),('Total acceptance','accepted','n'),('Overall singleton error','singleton_errors','accepted'),('Raw full-acceptance error','raw_errors','n')]
 def pct(a,b):return f'{100*a/b:.6f}%' if b else 'NA'
 check=[]
 def contains(context,line,text):
  ok=line in text;check.append(dict(context=context,expected=line,present=ok));compare('report/'+context,{'present':str(True)},{'present':str(ok)})
 for k,v in threshold.items():contains('threshold/'+k,f'| {k} | {repr(v)} |',stage)
 for label,num,den in specs:
  line=f"| {label} | {c[num]} | {c[den] if den else '—'} | {pct(c[num],c[den]) if den else str(c[num])} |"
  contains('overall/'+label,line,stage)
  cells=[f'{x[num]}/{x[den]} = {pct(x[num],x[den])}' if den else str(x[num]) for x in seg]
  contains('segments/'+label,'| '+' | '.join([label]+cells)+' |',stage)
 for r in coverage:contains('source/'+r['class_name'],f"| UNSW {r['class_name']} class coverage | {r['numerator']} | {r['denominator']} | {pct(r['numerator'],r['denominator'])} |",stage)
 contains('segment_ranges','| rank range | '+' | '.join(f'[{a},{b})' for a,b in SEG)+' |',stage)
 contains('segment_windows','| Number of windows | '+' | '.join(str((b-a)//10000) for a,b in SEG)+' |',stage)
 contains('segment_shares','| for total TP | '+' | '.join(f"{x['tp']}/{c['tp']} = {pct(x['tp'],c['tp'])}" for x in seg)+' |',stage)
 contains('segment_difference','| Raw − singleton ( percentage points ) | '+' | '.join(str((x['raw_errors']/x['n']-x['singleton_errors']/x['accepted'])*100) if x['accepted'] else 'NA' for x in seg)+' |',stage)
 for r in intervals:contains('interval/'+r['metric'],'| '+' | '.join(str(r[k]) for k in ['metric','estimate','lower_95','upper_95','valid_replicates','unit'])+' |',stage)
 for i,x in enumerate(seg,1):
  line='| '+' | '.join([str(i),str(x['tp']),pct(x['tp'],c['tp']),pct(x['tp'],x['true_attack']),pct(x['singleton_errors'],x['accepted']),pct(x['raw_errors'],x['n'])])+' |'
  contains('gate_segment/'+str(i),line,gate);contains('stage_gate_segment/'+str(i),line,stage)
 for label,value in [('TP',c['tp']),('disagreement',c['singleton_raw_disagreement']),('scan',sample['scan_rows'])]:contains('text/'+label,f'{value:,}',stage)
 contains('stop_rank',f" Stop the rank={sample['stop_rank']}(includes)",stage)
 risk=c['singleton_errors']/c['accepted'];raw=c['raw_errors']/c['n'];d=(raw-risk)*100
 contains('overall_risk_difference',f' Raw − singleton error rate:{d} One percent.',stage)
 active=[i+1 for i,x in enumerate(seg) if x['tp']>0]
 better=[i+1 for i,x in enumerate(seg) if x['accepted'] and x['singleton_errors']/x['accepted']<x['raw_errors']/x['n']]
 peak=1+int(np.argmax([x['tp'] for x in seg]));last_share=100*sum(x['tp'] for x in seg[2:])/c['tp']
 contains('hardcoded_segment3',pct(seg[2]['tp'],c['tp']),gate)
 contains('hardcoded_segment4',pct(seg[3]['tp'],c['tp']),gate)
 contains('hardcoded_last_two',f'{last_share:.6f}%',gate)
 contains('active_segments',f'（{len(active)}/4 (Figure 1)',gate)
 assert active==[1,2,3,4] and better==[4]
 decision='No-go' if c['tp']==0 or risk>=raw else 'Teacher judgement'
 contains('decision',f'**{decision}**',gate)
 contains('gate_raw_risk',pct(c['raw_errors'],c['n']),gate);contains('gate_singleton_risk',pct(c['singleton_errors'],c['accepted']),gate)
 assert threshold['threshold_sum']>1 and c['empty']==0
 save('report_numeric_comparison.csv',check)
 dynamic=f'''# independent Dynamic results text\n\nIt is generated by the combined computational dynamics of the original CSV labels and independent set states, without calling the original report().\n\nTP={c['tp']:,}; There is a TP time period ={active}. The contributions are: {', '.join(pct(x['tp'],c['tp']) for x in seg)}; The maximum contribution is 1.{peak} Paragraphs, the last two paragraphs together {last_share:.6f}%. The detection of distribution biases in the later half of the spectrum does not result from a single period of time.\n\nsingleton risk ={pct(c['singleton_errors'],c['accepted'])}; Raw risk ={pct(c['raw_errors'],c['n'])}；Raw−singleton={d:.12f} One percent. The time period for risk to be less than Raw is {better}. singleton differs from Raw predictions {c['singleton_raw_disagreement']:,} It’s not.\n\nq and ={threshold['threshold_sum']} It’s not a vacuum cluster, it’s a vacuum cluster ={c['empty']} The two-element set ={c['double']:,}.\n\nThe TP=0 condition is {c['tp']==0}; The singleton≥Raw condition is {risk>=raw}; It was re-judged:{decision}. The materially lower value threshold is not defined.\n\nThe original scripts call the same counts function in whole and by window, only to verify aggregation consistency; This time, the independent state coding, the original CSV labels and the combined counting are used. The original report’s deadline text is established for this data, but should not be used as a generalizable reporting logic.\n '''
 (O/'dynamic_reference_report.md').write_text(dynamic)

if __name__=='__main__':main()
