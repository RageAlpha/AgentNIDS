"""User-authorized RF/BoT supplemental evaluation, original frozen thresholds."""
import sys,json
from pathlib import Path
import numpy as np
import input_helpers as h
from set_logic_checks import counts,unit_checks
OLD=h.ROOT/'outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z'
OUT=Path(__file__).resolve().parent

def aggregate(rows):return {k:sum(r[k] for r in rows) for k in ['n','true_benign','true_attack','benign_singleton','attack_singleton','empty','double','tn','fn','fp','tp','accepted','singleton_errors','raw_errors','singleton_raw_disagreement','benign_covered','attack_covered']}
def metrics(scope,c,total_tp):
 pairs=[('true_benign_fraction','true_benign','n'),('true_attack_fraction','true_attack','n'),('acceptance_rate','accepted','n'),('singleton_error_rate','singleton_errors','accepted'),('raw_error_rate','raw_errors','n'),('benign_singleton_rate','benign_singleton','n'),('attack_singleton_rate','attack_singleton','n'),('empty_set_rate','empty','n'),('double_set_rate','double','n'),('benign_singleton_error_rate','fn','benign_singleton'),('attack_singleton_error_rate','fp','attack_singleton'),('tp_fraction_of_true_attacks','tp','true_attack')]
 rows=[dict(scope=scope,metric=name,numerator=c[a],denominator=c[b],value=c[a]/c[b] if c[b] else 'NA',unit='fraction') for name,a,b in pairs]
 for k in ['n','true_benign','true_attack','tp']:rows.append(dict(scope=scope,metric=k,numerator=c[k],denominator=1,value=c[k],unit='count'))
 rows.append(dict(scope=scope,metric='tp_share_of_overall',numerator=c['tp'],denominator=total_tp,value=c['tp']/total_tp if total_tp else 'NA',unit='fraction'))
 num=100*(c['raw_errors']*c['accepted']-c['singleton_errors']*c['n']);den=c['n']*c['accepted']
 rows.append(dict(scope=scope,metric='raw_minus_singleton_pp',numerator=num,denominator=den,value=num/den if den else 'NA',unit='percentage_points'))
 return rows

def main():
 docs=[h.ROOT/'docs/OutcomeB_2Expirements'/x for x in ['Reply_to_Siyuan_OutcomeB_BeginWriting.md','OutcomeB_Two_Experiments_Plan_ZH.md']]
 for p in docs:assert len(p.read_text())>1000
 oldprotocol=h.read(OLD/'protocol.json');gates=h.csvread(OLD/'gate_decisions.csv');collection=h.read(OLD/'bot/collection_summary.json')
 sample=h.csvread(OLD/'bot/calibration_sample_manifest.csv');scores=h.csvread(OLD/'bot/rf_All49/calibration_scores.csv');tp=OLD/'bot/rf_All49/threshold_freeze.json';f=h.read(tp)
 qb,qa=f['q_Benign'],f['q_Attack']
 assert qb==0.6932222247123718 and qa==0.8865555301308632
 assert f['model']=='rf_All49' and f['domain']=='BoT' and f['gate']=='The Professor’s Judgment' and .8<=qa<.9
 assert next(r for r in gates if r['model']=='rf_All49')['gate']=='The Professor’s Judgment'
 assert collection['collected_per_class']==[200,200] and collection['actual_labels_scanned_for_collection']==14864
 print(h.now(),'Validating frozen model, rank mapping, windows and original sample.',flush=True)
 y,o,ws,cache,iv=h.validate_domain('BoT',['rf_All49']);cache=cache['rf_All49']
 h.check_sample(sample,y,o,ws)
 assert [{k:r[k] for k in sample[0]} for r in scores]==sample
 ranks=np.array([int(r['rank']) for r in sample]);p=cache['raw_probability'][ranks];ys=np.array([int(r['label']) for r in sample])
 assert np.array_equal(p,np.array([float(r['raw_probability']) for r in scores]))
 assert np.array_equal(np.where(ys==0,p,1-p),np.array([float(r['true_class_score']) for r in scores]))
 assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
 pm=h.read(Path(oldprotocol['preprocessing_manifest']));assert pm['fit_domain']=='UNSW' and pm['fit_partition']=='S_train'
 assert Path(oldprotocol['preprocessing_stats']).is_file()
 checks=unit_checks()
 h.write('authorization.json',dict(recorded_at_utc=h.now(),authorization='Complementary full user authorized evaluation',source='The user is instructed to complete RF-All49 / BoT Variant B evaluation.',scope='Only frozen RF-All49 / BoT, fixed prior thresholds, startup excluded, fixed four segments',original_gate='Undefined [0.80,0.90) range, for the professor to decide',automatic_gate_pass=False,professor_approval_claimed=False,other_model_domain_branches='remain threshold-stopped'))
 h.write('protocol.json',dict(created_at_utc=h.now(),authorization='Complementary full user authorized evaluation',authorization_record=str(OUT/'authorization.json'),original_gate='Undefined [0.80,0.90) range, not original gate control automatic release',professor_approved=False,authority=[str(p) for p in docs],original_protocol=str(OLD/'protocol.json'),original_gate_file=str(OLD/'gate_decisions.csv'),threshold_source=str(tp),sample_source=str(OLD/'bot/calibration_sample_manifest.csv'),scores_source=str(OLD/'bot/rf_All49/calibration_scores.csv'),q_Benign=qb,q_Attack=qa,epsilon=.1,K=400,preprocessing_manifest=oldprotocol['preprocessing_manifest'],preprocessing_stats=oldprotocol['preprocessing_stats'],input_validation=iv,rank_convention='zero-based half-open; original row is zero-based data-row index',evaluation=[100000,16930000],evaluation_rows=16830000,evaluation_windows=1683,segments=h.SEGS['BoT'],excluded_startup=[0,100000],excluded_tail=[16930000,16933808],probability='raw_probability=P(Attack); P(Benign)=1-p; unchanged float64',labels={'0':'Benign','1':'Attack'},prediction={'Benign':'p<=q_Benign','Attack':'1-p<=q_Attack','singleton':'accept unique label independently of Raw','double':'abstain','empty':'reject','Raw':'p>=0.5'},metrics='counts; acceptance, accepted error, Raw full-accept error, Raw-minus-accepted percentage points; four sets; each singleton error; TP, TP/true Attack, segment TP/overall TP; exact numerators and denominators, zero denominator NA',no_bootstrap=True,no_source_evaluation=True,no_recalibration=True,no_file_digests=True))
 h.write('threshold_used.json',dict(source=str(tp),loaded_at_utc=h.now(),original_frozen_record=f,q_Benign=qb,q_Attack=qa,recalibrated=False))
 windows=ws[10:];assert len(windows)==1683 and int(windows[-1]['rank_end_exclusive'])==16930000
 window_rows=[]
 for i,w in enumerate(windows):
  a,b=int(w['rank_start']),int(w['rank_end_exclusive']);assert a==100000+i*10000 and b==a+10000
  p=cache['raw_probability'][a:b];labels=y[a:b]
  assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and ((labels==0)|(labels==1)).all()
  c=counts(p,labels,qb,qa)
  assert c['empty']==0 # Checked after actual label-free set construction, never forced.
  seg=next(j+1 for j,(lo,hi) in enumerate(zip(h.SEGS['BoT'][:-1],h.SEGS['BoT'][1:])) if lo<=a<b<=hi)
  window_rows.append(dict(window_id=w['window_id'],rank_start=a,rank_end_exclusive=b,segment=seg,**c))
 h.save('window_counts.csv',window_rows)
 overall=aggregate(window_rows);segments=[aggregate([r for r in window_rows if r['segment']==i]) for i in range(1,5)]
 assert overall['n']==16830000 and aggregate(segments)==overall
 for c,(a,b) in zip(segments,zip(h.SEGS['BoT'][:-1],h.SEGS['BoT'][1:])):
  assert c['n']==b-a and c==counts(cache['raw_probability'][a:b],y[a:b],qb,qa)
 h.save('aggregate_counts.csv',[dict(scope=s,**c) for s,c in zip(['overall','segment_1','segment_2','segment_3','segment_4'],[overall]+segments)])
 mr=metrics('overall',overall,overall['tp']);tr=[r for i,c in enumerate(segments,1) for r in metrics('segment_'+str(i),c,overall['tp'])]
 h.save('evaluation_metrics.csv',mr);h.save('temporal_metrics.csv',tr)
 h.write('validation.json',dict(status='PASS',completed_at_utc=h.now(),unit_checks=checks,input_checks=iv,original_sample_and_saved_scores_match=True,thresholds_loaded_from_original_not_refitted=True,full_precision_thresholds=True,actual_evaluation_rows=overall['n'],actual_evaluation_windows=len(windows),startup_and_tail_excluded=True,calibration_evaluation_identity_overlap=0,sets_sum_to_n=True,singleton_confusion_sum_to_accepted=True,segments_sum_to_overall=True,direct_segment_counts_match_window_aggregation=True,raw_and_singleton_errors_computed_separately=True,empty_count_computed=overall['empty'],empty_count_matches_threshold_sum_above_one=True,all_evaluated_probabilities_finite_and_in_unit_interval=True,zero_denominator='NA',bootstrap_run=False,source_test_run=False,other_models_run=False,historical_artifacts_modified=False,file_digests_computed=False))
 def pct(a,b):return 'NA' if not b else f'{100*a/b:.6f}%'
 def pair(a,b):return f'{a}/{b} ({pct(a,b)})'
 report='# RF-All49 / BoT User Licensed Additional Full Review\n\nThis time with explicit authorization from the user; The original gate control remains undefined for the [0.80,0.90) range, is not automatically released, and is not claimed to have been approved by the professor. The / domain of other models keeps the threshold stopping.\n\n'
 report+=f' From `{tp}` Direct Loading: q_Benign={qb!r}，q_Attack={qa!r}; No re-sampling or calibration.\n\n '
 report+='[100000,16930000) , 16830000 items, 1683 windows; Excluding the entire start-up period and 3808 tail. Raw calculated by p≥0.5 independent; singleton uses the only labels of the set.\n\n'
 report+='| range | N / Benign / Attack | acceptor /N | error singleton/ acceptor | Raw error /N | Raw − acceptor (pp) | TP/ all Attack | TP/ total TP |\n|---|---|---|---|---|---|---|---|\n'
 diffs=[]
 for name,c in zip(['Overall','Paragraph 1','Paragraph 2','Paragraph 3','Paragraph 4'],[overall]+segments):
  diff=100*(c['raw_errors']/c['n']-c['singleton_errors']/c['accepted']) if c['accepted'] else None;diffs.append(diff)
  report+='| '+' | '.join([name,f"{c['n']} / {c['true_benign']} / {c['true_attack']}",pair(c['accepted'],c['n']),pair(c['singleton_errors'],c['accepted']),pair(c['raw_errors'],c['n']),'NA' if diff is None else f'{diff:.6f}',pair(c['tp'],c['true_attack']),pair(c['tp'],overall['tp'])])+' |\n'
 report+='\n| range | Benign singleton/N | Attack singleton/N | exhaust /N | double element /N | Benign singleton error rate | Attack singleton error rate |\n|---|---|---|---|---|---|---|\n'
 for name,c in zip(['Overall','Paragraph 1','Paragraph 2','Paragraph 3','Paragraph 4'],[overall]+segments):report+='| '+' | '.join([name]+[pair(c[k],c['n']) for k in ['benign_singleton','attack_singleton','empty','double']]+[pair(c['fn'],c['benign_singleton']),pair(c['fp'],c['attack_singleton'])])+' |\n'
 improved=[str(i) for i,v in enumerate(diffs[1:],1) if v is not None and v>0]
 report+=f"\nOverall Raw - acceptance error rate ={diffs[0]} One percentage point; The time frame for improvement in descriptive error rates:{', '.join(improved) or 'none'}. Whether all four sections have improved:{'yes' if len(improved)==4 else 'no'}。\n"
 report+=f"\nOverall acceptance rate {pct(overall['accepted'],overall['n'])} The two-element abstention ratio {pct(overall['double'],overall['n'])} It’s the right answer to the TP= attack {overall['tp']} And that’s the real Attack.{pct(overall['tp'],overall['true_attack'])}. Changes in the error rate must be interpreted along with these acceptance, abstention and attack detection results.\n "
 report+='\nNo interim estimates have been made and no statistical significance is declared; The original reply did not define the new passing standards after the evaluation of this range, so it did not add Go/No-go or automatically declare Outcome A.\n\nAll checks are approved: line identity, original sample and caching consistent; The full range corresponds to the manifest of the C1/ window; Four sets, singleton confused with the total of four points; The whole independent counts are clustered in the check window. The vacuum concentrate is actually calculated as 0, in line with threshold values and >1. Synthetic checks cover equality, non-asymmetric thresholds, zero denominators, and singleton different from Raw. For example, the molecular quotient is CSV and the zero quotient is NA.\n\nUnmodified historical product, unreported, applied Platt, modified probability, executed source test or bootstrap, other models or experiments. After completion of the evaluation, stop and submit user reviews; There is no substitute for a professor.\n\nOutput directory: `'+str(OUT)+'`\n'
 h.txt('evaluation_report.md',report)
 print(report,flush=True)
if __name__=='__main__':main()
