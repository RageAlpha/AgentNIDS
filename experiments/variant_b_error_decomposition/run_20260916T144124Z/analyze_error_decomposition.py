"""Arithmetic error decomposition on the original fixed Variant B acceptance set.
Read-only historical inputs. No fitted parameters or new acceptance strategy.
"""
import csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path('/home/karri/projects/NIDSProject_Professor_Run');O=Path(__file__).resolve().parent
OLD=R/'outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z'
SEG=[(100000,6960000),(6960000,13820000),(13820000,20670000),(20670000,27520000)]
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def save(name,rr):
 with (O/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def write(name,data):
 with (O/name).open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
def inv():return {str(p.relative_to(OLD)):[p.stat().st_size,p.stat().st_mtime_ns] for p in OLD.rglob('*') if p.is_file()}
def div(n,d):return n/d if d else 'NA'
def extract(h,truth=None,pred=None):
 # h: [set code 0 empty/1 B/2 A/3 both, true class, Raw class].
 ts=[0,1] if truth is None else [truth];ps=[0,1] if pred is None else [pred]
 t=dict(both_correct=0,H=0,C=0,both_wrong=0,new_false_positive=0,new_false_negative=0,corrected_false_positive=0,corrected_false_negative=0)
 for y in ts:
  for b in ps:
   for raw in (0,1):
    n=int(h[b+1,y,raw]);er=raw!=y;eb=b!=y
    key='both_wrong' if er and eb else ('H' if eb else ('C' if er else 'both_correct'))
    t[key]+=n
    if key=='H':t['new_false_positive' if y==0 else 'new_false_negative']+=n
    if key=='C':t['corrected_false_positive' if y==0 else 'corrected_false_negative']+=n
 t.update(n_A=sum(t[k] for k in ['both_correct','H','C','both_wrong']),raw_errors_A=t['C']+t['both_wrong'],B_errors_A=t['H']+t['both_wrong'],changed_predictions=t['H']+t['C'],net_added_errors=t['H']-t['C'])
 assert t['B_errors_A']-t['raw_errors_A']==t['net_added_errors']
 return t

def measure(h,truth=None):
 t=extract(h,truth);ts=[0,1] if truth is None else [truth]
 n=sum(int(h[:,y,:].sum()) for y in ts);er=sum(int(h[:,y,1-y].sum()) for y in ts)
 rejected=n-t['n_A'];errej=er-t['raw_errors_A']
 pairs={'R_raw_all':(er,n),'R_raw_A':(t['raw_errors_A'],t['n_A']),'R_B_A':(t['B_errors_A'],t['n_A']),'R_raw_rejected':(errej,rejected)}
 # Integer numerator/denominator for each percentage-point contrast.
 pairs.update(D_selection=(100*(t['raw_errors_A']*n-er*t['n_A']),t['n_A']*n),
  D_relabel=(100*t['net_added_errors'],t['n_A']),D_total=(100*(t['B_errors_A']*n-er*t['n_A']),t['n_A']*n))
 assert pairs['D_selection'][0]+pairs['D_relabel'][0]*n==pairs['D_total'][0]
 return t,pairs

def metric_rows(scope,h,truth=None):
 t,p=measure(h,truth)
 return [dict(scope=scope,true_class='all' if truth is None else ['Benign','Attack'][truth],metric=k,numerator=n,denominator=d,value=div(n,d),unit='percentage_points' if k.startswith('D_') else 'fraction') for k,(n,d) in p.items()]
def pct(n,d):return 'NA' if not d else f'{100*n/d:.6f}%'
def md(headers,rr):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
def main():
 before=inv();freeze=read(OLD/'threshold_freeze.json');qb,qa=freeze['q_Benign'],freeze['q_Attack']
 p=np.load(R/'outputs/prediction_cache_v2/ToN/rf_All49.npy',mmap_mode='r')['raw_probability']
 y=np.load(R/'outputs/feature_store_v2/ToN_label.npy',mmap_mode='r')
 oldw=rows(OLD/'window_counts.csv');assert len(oldw)==2742
 hist=np.zeros((2742,4,2,2),dtype=np.int64);wout=[];checks=0
 for i,w in enumerate(oldw):
  a=100000+i*10000;b=a+10000;assert int(w['rank_start'])==a and int(w['rank_end_exclusive'])==b
  pp=p[a:b];yy=y[a:b];assert np.isfinite(pp).all() and np.all((yy==0)|(yy==1))
  # Labels never participate in set membership or acceptance.
  membership=(pp<=qb).astype(np.int64)+2*((1.-pp)<=qa)
  raw=(pp>=.5).astype(np.int64)
  hist[i]=np.bincount(membership*4+yy.astype(np.int64)*2+raw,minlength=16).reshape(4,2,2)
  h=hist[i];t=extract(h);n=int(h.sum());rawerr=int(h[:,0,1].sum()+h[:,1,0].sum())
  expected=dict(n=n,true_benign=int(h[:,0,:].sum()),true_attack=int(h[:,1,:].sum()),
   benign_singleton=int(h[1].sum()),attack_singleton=int(h[2].sum()),empty=int(h[0].sum()),double=int(h[3].sum()),
   tn=int(h[1,0,:].sum()),fn=int(h[1,1,:].sum()),fp=int(h[2,0,:].sum()),tp=int(h[2,1,:].sum()),
   accepted=t['n_A'],singleton_errors=t['B_errors_A'],raw_errors=rawerr,singleton_raw_disagreement=t['changed_predictions'],
   benign_covered=int(h[[1,3],0,:].sum()),attack_covered=int(h[[2,3],1,:].sum()))
  for key,v in expected.items():assert int(w[key])==v,(i,key,v,w[key]);checks+=1
  segment=1+next(j for j,(aa,bb) in enumerate(SEG) if aa<=a<bb)
  assert int(w['segment'])==segment
  wout.append(dict(window_index=i,window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,segment=segment,
   n=n,n_rejected=n-t['n_A'],raw_errors_all=rawerr,raw_errors_rejected=rawerr-t['raw_errors_A'],**t))
 save('window_decomposition_counts.csv',wout)
 scoped=[('overall',hist.sum(axis=0))]+[(f'segment_{i}',hist[(a-100000)//10000:(b-100000)//10000].sum(axis=0)) for i,(a,b) in enumerate(SEG,1)]
 assert np.array_equal(sum(h for _,h in scoped[1:]),scoped[0][1])
 transitions=[];classes=[];risks=[]
 for scope,h in scoped:
  risks.extend(metric_rows(scope,h))
  for truth in (None,0,1):
   for pred in (None,0,1):
    transitions.append(dict(scope=scope,true_class='all' if truth is None else ['Benign','Attack'][truth],singleton_prediction='all' if pred is None else ['Benign','Attack'][pred],**extract(h,truth,pred)))
  for truth in (0,1):classes.extend(metric_rows(scope,h,truth))
 save('error_transition_counts.csv',transitions);save('class_decomposition.csv',classes)
 save('risk_decomposition.csv',[r for r in risks if r['scope']=='overall']);save('temporal_decomposition.csv',[r for r in risks if r['scope']!='overall'])
 total=extract(scoped[0][1]);_,pairs=measure(scoped[0][1])
 assert total['n_A']==13131642 and total['B_errors_A']==2918563 and pairs['R_raw_all'][0]==4361537
 assert total['changed_predictions']==2052584==total['H']+total['C']
 assert total['B_errors_A']-total['raw_errors_A']==total['H']-total['C']
 assert div(*pairs['D_total'])== -float(next(r for r in rows(OLD/'evaluation_metrics.csv') if r['metric']=='raw_minus_singleton_pp')['value'])
 # Reuse original block definitions and indices without creating an RNG.
 ob=rows(OLD/'bootstrap_block_counts.csv');assert len(ob)==275
 bh=[];bout=[]
 for i,b in enumerate(ob):
  lo=int(b['first_evaluation_window']);count=int(b['window_count']);assert lo==10*i and count==min(10,2742-lo)
  hh=hist[lo:lo+count].sum(axis=0);bh.append(hh)
  tt=extract(hh);nn=int(hh.sum());err=int(hh[:,0,1].sum()+hh[:,1,0].sum())
  assert int(b['rank_start'])==100000+lo*10000 and int(b['rank_end_exclusive'])==100000+(lo+count)*10000
  assert [nn,tt['n_A'],tt['B_errors_A'],err]==[int(b[k]) for k in ['n','accepted','singleton_errors','raw_errors']]
  bout.append(dict(block=i,window_start=lo,window_count=count,n=nn,raw_errors_all=err,**tt))
 save('block_decomposition_counts.csv',bout)
 idx=np.load(OLD/'bootstrap_indices.npy',allow_pickle=False);assert idx.shape==(2000,275) and idx.min()>=0 and idx.max()<275
 reps=[]
 for j,draw in enumerate(idx):
  hh=np.asarray(bh)[draw].sum(axis=0);t,pr=measure(hh)
  rr=dict(replicate=j,n=int(hh.sum()),n_A=t['n_A'],raw_errors_all=pr['R_raw_all'][0],raw_errors_A=t['raw_errors_A'],B_errors_A=t['B_errors_A'],H=t['H'],C=t['C'])
  for k in ['D_selection','D_relabel','D_total']:rr[k]=div(*pr[k])
  if rr['n_A']:assert abs(rr['D_selection']+rr['D_relabel']-rr['D_total'])<1e-12
  reps.append(rr)
 save('bootstrap_replicates.csv',reps)
 intervals=[]
 for k in ['D_selection','D_relabel','D_total']:
  values=[r[k] for r in reps if r[k]!='NA'];lo,hi=np.percentile(values,[2.5,97.5],method='linear') if values else ('NA','NA')
  intervals.append(dict(metric=k,estimate=div(*pairs[k]),lower_95=lo,upper_95=hi,valid_replicates=len(values),total_replicates=2000,unit='percentage_points'))
 save('bootstrap_intervals.csv',intervals)
 write('analysis_protocol.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),threshold_source=str(OLD/'threshold_freeze.json'),q_Benign=qb,q_Attack=qa,
  evaluation=[100000,27520000],segments=SEG,bootstrap_indices_source=str(OLD/'bootstrap_indices.npy'),bootstrap_block_source=str(OLD/'bootstrap_block_counts.csv'),
  new_sampling_plan=False,acceptance='original singleton set only',rejected_B_prediction='undefined; never imputed',
  sign='positive = risk increase; negative = risk decrease',interpretation='descriptive arithmetic; no causal identification',
  class_denominators='within same true class and scope for all/accepted/rejected; singleton prediction strata only used for transition counts'))
 assert before==inv()
 validation=dict(status='PASS',completed_at_utc=datetime.now(timezone.utc).isoformat(),original_window_fields_compared=checks,
  original_windows_matched=2742,original_acceptance_rule_unchanged=True,all_original_set_and_confusion_counts_match=True,
  n_A=total['n_A'],B_errors_A=total['B_errors_A'],raw_errors_all=pairs['R_raw_all'][0],raw_errors_A=total['raw_errors_A'],
  H=total['H'],C=total['C'],H_plus_C=total['changed_predictions'],net_added_errors=total['net_added_errors'],
  exact_integer_decomposition_identity_all_scopes_classes_and_replicates=True,segments_sum_to_overall=True,
  bootstrap_matrix_reused=True,blocks=275,last_block_windows=2,replicates=2000,new_RNG_not_used=True,
  no_B_label_assigned_to_rejected=True,old_files_size_mtime_unchanged=True,zero_denominator='NA',
  reference_total_pp=div(*pairs['D_total']))
 write('validation.json',validation)
 report(scoped,intervals,transitions)
 print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
 print('Intervals:',json.dumps(intervals),flush=True)

def report(scoped,intervals,transitions):
 overall=scoped[0][1];t,p=measure(overall)
 text='# ToN Variant B: Error decomposition of screening and reassessing\n\nOnly analyzing the original fixed threshold and singleton accepts the set A. No other acceptance strategies or modified forecasts have been implemented, and no record of Variant B labels has been refused. an increase in the error rate of positive signals; The following is an arithmetic breakdown of the current data, not a proof of the causal mechanism.\n\n'
 text+=md(['Scope','N','N_A','Raw is all wrong.','Raw error rate at A','Variant B at A error rate','Refusal to collect Raw error rate','Select changes in pp','Changes in pp','Total change in pp'],[
  [scope,int(h.sum()),extract(h)['n_A']]+[pct(*measure(h)[1][k]) for k in ['R_raw_all','R_raw_A','R_B_A','R_raw_rejected']]+[f'{div(*measure(h)[1][k]):.6f}' if measure(h)[1][k][1] else 'NA' for k in ['D_selection','D_relabel','D_total']] for scope,h in scoped])
 text+='\n\n## accepts the error transfer of the set\n\n'+md(['Scope','Raw is correct→ B is correct','Raw correct→ B incorrect H','Raw Error→ B correctly C','Raw error→ B error','Net increase in H - C'],[[s]+[extract(h)[k] for k in ['both_correct','H','C','both_wrong','net_added_errors']] for s,h in scoped])
 text+='\n\nThe overall 2×2 table:\n\n'+md(['','Variant B is correct.','Variant B error'],[['Raw is correct.',t['both_correct'],t['H']],['Raw error',t['C'],t['both_wrong']]])
 text+='\n\n## real category decomposition\n\nAll of the categories, accepting and rejecting subcategories, are restricted to the same real category and cannot be directly added to the percentage of class conditions as a whole.\n\n'+md(['Scope','The real category','Category N','Category N_A','Raw all risk','Raw at A risk','B at A risk','Refusing Raw risk','Select pp','Changing the pp','Total change in pp','Added H error','Correction of C error'],[
  [s,['Benign','Attack'][y],measure(h,y)[1]['R_raw_all'][1],extract(h,y)['n_A']]+[pct(*measure(h,y)[1][k]) for k in ['R_raw_all','R_raw_A','R_B_A','R_raw_rejected']]+[f'{div(*measure(h,y)[1][k]):.6f}' if measure(h,y)[1][k][1] else 'NA' for k in ['D_selection','D_relabel','D_total']]+[extract(h,y)['H'],extract(h,y)['C']] for s,h in scoped for y in (0,1)])
 text+='\n\nOverall, the real category with singleton prediction is divided into:\n\n'+md(['The real category','singleton forecast','N_A','Both are correct.','H','C','Both are wrong.'],[[['Benign','Attack'][y],['Benign','Attack'][b]]+[extract(overall,y,b)[k] for k in ['n_A','both_correct','H','C','both_wrong']] for y in (0,1) for b in (0,1)])
 text+=f"\n\nNew misreported ={t['new_false_positive']:,} New leaked ={t['new_false_negative']:,} correcting the = error {t['corrected_false_positive']:,} Correction of = leak {t['corrected_false_negative']:,}. For each segment, the real category, the singleton category and the total transition counts for the cross-sectional layers see error_transition_counts.csv; For each proportional molecule /, see risk_decomposition.csv, temporal_decomposition.csv, and class_decomposition.csv.\n "
 text+='\n## paired with bootstrap\n\nDirectly load the original 2000×275 sampling matrix and the original 275 block definition, the final block 2 window; No new sampling scheme has been established. Each copy simultaneously summarizes all errors, counting the set Raw errors and the set Variant B errors, and calculates the three difference values. The 95% percentile interval uses linear inputs; Four paragraphs only describe. There is no alternative threshold for experimentation with 0 or not.\n\n'+md(['Changes','Estimated pp','95% lower bound pp','95% upper bound pp','effective copy'],[[r[k] for k in ['metric','estimate','lower_95','upper_95','valid_replicates']] for r in intervals])
 text+='\n\n## directly answers\n\n'
 selection=div(*p['D_selection']);relabel=div(*p['D_relabel'])
 text+=f" 1. ** Does the current accepted set yield a more reliable record of Raw? ** {'yes' if selection<0 else 'no'}. Raw overall risk {pct(*p['R_raw_all'])} A in Raw risk {pct(*p['R_raw_A'])}; The screening changes are:{selection:+.6f} One percent. Refusing to set Raw to risk {pct(*p['R_raw_rejected'])}. This is a descriptive comparison on a fixed A, not a deployment of a new strategy.\n\n "
 text+=f" 2. ** Changes in net impact: ** H={t['H']:,}，C={t['C']:,}，{'Net increase' if t['net_added_errors']>=0 else 'Net correction'}{abs(t['net_added_errors']):,} It was a mistake. Changes in judgments {relabel:+.6f} One percent. H+C={t['changed_predictions']:,} It is also true that two different records in a set are predicted.\n\n "
 signs={k:[i for i,(_,h) in enumerate(scoped[1:],1) if div(*measure(h)[1][k])<0] for k in ['D_selection','D_relabel','D_total']}
 text+=f" 3. **Are the four segments and both classes consistent?** Segments where screening reduces risk:{signs['D_selection']} This is the first time that the risk decrease has been {signs['D_relabel']} The total risk decrease is:{signs['D_total']}。"
 for y in (0,1):
  _,q=measure(overall,y)
  text+=f" The truth.{['Benign','Attack'][y]} The total change in the pH of the filtered / to / is {div(*q['D_selection']):+.6f}/{div(*q['D_relabel']):+.6f}/{div(*q['D_total']):+.6f} One percent."
 text+='Therefore, it is not possible to extend the overall decomposition direction directly to all time periods or to both types of records.\n\n'
 if selection<0 and relabel>0:
  priority='Priority is given to studying the labels rule changes. The Raw risk in the current A is lower than the overall Raw risk, and the screening overall shows a decrease in risk, while re-judging the net increase in error and offsetting this improvement. Selection still needs to be combined with segmentation and class differentiation studies, so it cannot be determined that it already has a stable generalization capability.'
 elif selection>0 and relabel>0:
  priority='Both parts are worth studying; Overall increase in risk can be prioritized'+('labels Changes to the Rules' if relabel>selection else 'Selection rules')+'While the other part of the time is reserved for class heterogeneity analysis.'
 else:priority='Subsequent studies shall be conducted to determine the heterogeneity of the / class of symbols, widths and sections in combination with the two sections; This decomposition does not constitute a causal attribution or a new strategy to prove effective.'
 text+='4. ** What is the priority study? ** '+priority+' This is a recommendation for the direction of subsequent diagnosis, not the conclusion of a new experiment; This time, no improvements have been made automatically.\n'
 text+=f"\n## verification and delivery\n\nOne by one, the set of the original 2742 window is confused with the Raw count, accepting the set rules and the counting is perfectly consistent. N_A={t['n_A']:,} Variant B accepts the error ={t['B_errors_A']:,} Raw is all wrong =.{p['R_raw_all'][0]:,}. For each range, each real category and each replica, check D_total=D_selection+D_relabel with an integer constant, the sum of the four counts is equal to the total. The original was read-only and no document summary was calculated.\n\nCatalogue: `{O}`. Analyze the scripts analyze.py, commands.sh and analysis.log to keep the same directory; bootstrap_replicates.csv keeps count of each copy. There is no large-scale gradual trajectory. The original No-go is unchanged, and this time it stops and does not enter Week1.\n "
 (O/'decomposition_report.md').write_text(text)
if __name__=='__main__':main()
