from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent
def rows(n):
 with (P/n).open() as f:return list(csv.DictReader(f))
def pct(x,d=4):return 'NA' if x=='NA' else f'{float(x)*100:.{d}f}%'
def num(x):return f'{int(float(x)):,}'
def table(headers,rr):return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rr)+'\n'
raw=rows('raw_metrics.csv');bins=rows('confidence_bin_metrics.csv');ci=rows('confidence_bin_intervals.csv');dist=rows('score_distribution_summary.csv');cdf=rows('score_threshold_cdf.csv');points=rows('raw_confidence_risk_coverage_points.csv');high=rows('high_confidence_summary.csv');context=rows('existing_conformal_summary.csv');inputs=json.loads((P/'input_records.json').read_text());protocol=json.loads((P/'protocol.json').read_text())
models=protocol['models'];schemes=['Early-400','Temporal-Stratified-400'];periods=list(protocol['evaluation_periods'])
rlookup={(r['model'],r['period'],r['metric']):r for r in raw};blookup={(r['model'],r['period'],r['prediction_subset'],r['bin']):r for r in bins};clookup={(r['model'],r['prediction_subset'],r['bin']):r for r in ci};hlookup={(r['model'],r['period'],r['prediction_subset']):r for r in high};plookup={(r['model'],r['period'],float(r['target_coverage'])):r for r in points}
def rv(m,p,k):return float(rlookup[m,p,k]['value'])
t='# ToN Raw Misrepresentation and calibration percentage distribution diagnosis\n\n'
t+='** has been completed; The original Outcome B remains unchanged.** is a descriptive diagnosis of freeze after RF-All49 and XGBoost-All49 already have a Raw probability, with no resampling, resampling of calibration, training, probability change, threshold selection or new acceptance strategies. labels rating is only used for post-concussion diagnosis.\n\n'
t+='## Core Explanation: Answers to 6 Questions\n\n'
t+='Does the ### 1. Raw confidence contain the wrong risk ranking information?\n\n'
t+='There is local information, but it is not stable, and there is an overall single-chord error in the risk sequence. RF predictions are concentrated in Attack, with error rates at [0.5,0.6), [0.6,0.7), and [0.7,0.8), respectively. '+', '.join(pct(blookup['rf_All49','overall','predicted_Attack',x]['error_rate']) for x in ['[0.5,0.6)','[0.6,0.7)','[0.7,0.8)'])+'And then it went up to [0.8,0.9). '+pct(blookup['rf_All49','overall','predicted_Attack','[0.8,0.9)']['error_rate'])+'. XGBoost predicts a clearer decline in Attack from low confidence to [0.9,0.95), followed by a reversal. It is not possible to generalize that the probability of addiction is completely worthless, nor can it be assumed that higher confidence is always safer.\n\n'
for m in models:
 r1=plookup[m,'overall',.01];r10=plookup[m,'overall',.1];t+=f"{m} Objective 1% Acceptance {pct(r1['actual_coverage'])} risk {pct(r1['risk'])}; Target 10% Acceptance {pct(r10['actual_coverage'])} risk {pct(r10['risk'])}; Accepted risk {rv(m,'overall','raw_error_rate')*100:.4f}%. These are fixed diagnosis points, none of which are selected or recommended.\n\n "
t+='### 2. Where is the focus of high confidence mistakes?\n\n'
t+='The following table summarizes the last two [0.99,1.0] boxes, not the new deployment thresholds. It is necessary to look at the number, the denominator and the error rate at the same time.\n\n'
for m in models:
 a=hlookup[m,'overall','all'];b=hlookup[m,'overall','predicted_Benign'];att=hlookup[m,'overall','predicted_Attack'];t+=f"{m} High confidence record {num(a['n'])} It’s wrong.{num(a['errors'])} The error rate.{pct(a['error_rate'])} And that’s the whole Raw error of the model.{pct(a['fraction_of_all_raw_errors'])}; It’s the first time that a person has ever seen a Benign error.{num(b['errors'])} The article is based on a misconception.{100*int(b['errors'])/int(a['errors']):.4f}%。"
 if m=='rf_All49':t+='RF in this range has no predicted Attack record; The maximum number of errors is at E3, but the E4 error rate is higher, and the two cannot be confused.\n\n'
 else:t+='Predicting Benign high confidence error at E4 maximum and risk maximum, E2; The E1 high confidence error rate of predicting Attack is high, but the support is small and cannot be used to replace overall conclusions.\n\n'
t+='Is the order of ### 3. different from the numerical value of calibration?\n\n'
t+='The two are distinctly different. XGBoost predicts that Attack has local ranking information from low confidence to [0.9,0.95), while many high confidence boxes have confidence values much higher than the actual accuracy. The low confidence boxes in the RF segment underestimated the actual accuracy. High confidence numbers should not be interpreted directly as verified accuracy rates.\n\n'
for m,b in [('rf_All49','[0.7,0.8)'),('xgb_All49','[0.999,1]')]:
 r=blookup[m,'overall','all',b];t+=f"{m} {b}：n={num(r['n'])} Average confidence={pct(r['mean_confidence'])} The actual accuracy rate is ={pct(r['accuracy'])} The difference between ={float(r['confidence_minus_accuracy'])*100:+.4f} One percent.\n\n "
t+='What is the difference between ### 4. calibration and the actual class score of the evaluation?\n\n'
t+='It’s a two-part analysis. RF calibration Attack score median is about 0.82 0.83 , E2/E3/E4 median is significantly lower, but E2 high score tail is still heavy, and E2’s 90% score exceeds two freeze Attack thresholds. The two sets of calibration Attack median for XGBoost are approximately 0.9984 and E2/E3/E4 for median are much lower; However, approximately thirty percent of Attack records in E4 lie in the extreme tail near 1; the frozen threshold covers only about 70%. The phenomenon can occur at the same time as the phenomenon of oxidation and the phenomenon of extreme tail.\n\n'
t+='On the Benign side, the median of both E4 models is lower than Early, but the 90%/99% tail is significantly heavier. The Benign of the time layered sample is higher than the tail and heavier than the Early. The XGBoost and Benign freeze thresholds are exactly the same, indicating that only one threshold will miss the distribution difference. For each category, only 200 and calibration samples, especially 99% and tail estimates are unstable, and it has not been proven that a single sample difference can be used as an overall mechanism.\n\n'
t+='### 5. Does the aggregate distribution hide local changes?\n\n'
t+='Yes, I did. The overall XGBoost target 10% of Raw risk is about 25%, E4 corresponds to risk is about 83%, E1 is about 4%; The overall goal of RF is 10%, risk is about 13%, E2 is about 31%, and E1 is less than 1%. Complete homogenization also significantly exceeds the target ratio: RF E4 targets 1% , 5% , 10% all fall to the same group boundary, effectively accepting about 13.63% and cannot be forced to record the nominal percentage within the intercept group.\n\n'
t+='XGBoost Early Attack covers about 86% overall, but E1 is about 99%, E4 is about 71%; RF Early Benign as a whole is about 85%, E3 is about 95%, E4 is about 79%. These local variations can be seen in the ECDF, segmentation table and the already accepted results below. confidence=max(p,1-p) also mixes two predictive categories; Much of the highest confidence area is predicting Benign, and low risk cannot automatically equal the correct response to more attacks.\n\n'
t+='### What explanations does 6. support and which cannot be excluded?\n\n'
t+='Supported evidence: Raw confidence/error relationships vary by class and time; numeric confidence differs from actual accuracy; calibration/evaluation differ in class-conditional score distributions, tails and point masses at thresholds. Calibration timing, finite-sample uncertainty, traffic composition, within-class changes and frozen-model errors across traffic regions may all contribute. This diagnosis cannot identify separate causal contributions or force a choice between base-model failure and calibration failure.\n\n'
t+='without indicating evidence of spatial or causal interference, representation failure cannot be misrepresented by high confidence; Different factors do not necessarily mean that Weighted Conformal is applicable, and proximity does not necessarily mean that Contrastive should be used; A poor operating point does not negate all selective prediction. There is no equivalent source domain rating and no claim that quantization source domain to target domain performance has decreased.\n\n'
t+='Variant B can change Raw to labels, which receives risk not equal to Raw to risk on the same receiver set; It is not possible to attribute all the reasons to the confidence ranking with Variant B risk higher than Raw. There is no additional / re-determination filter.\n\n'
t+='## protocol and subset\n\nconfidence without labels; alpha_true requires labels and is used only for calibration and post-diagnosis. p for Raw P(Attack) and float64; p>=0.5 predicts Attack, alpha_true=p (Benign) or 1-p (Attack).\n\n'
t+=table(['Evaluation Period','rank range, left closed right open','Record count'],[[k,f'[{v[0]},{v[1]})',f'{v[1]-v[0]:,}'] for k,v in protocol['evaluation_periods'].items()])
t+='Excluding the entire startup and the final 260 points. The subset of the distribution box ratio is the same time range, the same predicted subset of the categories; The empty box ratio is / mean / and the error rate is NA. The / coefficient of all proportions is maintained at CSV. The curve sums the two classes by default to record all real classes in the same range as a subset, i.e. the curve correctly accepts the / of the class for all the curves of the class; Alternatively, it accepts the recall of the class of internal responses to avoid confusion. All curves keep the Raw prediction of the labels and the whole group is included in the same confidence record.\n\n'
t+='The distribution of fractions is determined by using linear as a description, rather than replacing the 181 rule of freeze. ECDF for each group is unified; The compression graph of the fixed grid shows an accurate CDF calculation at each horizontal coordinate of the graph; The curve graph is compressed to a maximum of about the boundary of the complete group 3000 and retains all fixed target points, with the target points, fractions and thresholds of CDF in CSV accurately calculated.\n\n'
t+='Reuse and verify RF and XGBoost by element to explore the same bootstrap matrix that runs in time scales: 275 blocks, each 10 window, end block 2 window, 2000 times, 95% percentile (linear). The risk in the box after each copy of the sum of the molecules is divided. Overall, the non-empty box effective duplicates of each predicted subset are 2000; RF predicts Attack to be 0 in the final two cases. paragraphs only describe; Narrow spaces cannot solve labels / time change conditions or model mechanism identification.\n\n'
t+='## Raw classification performance\n'
for m in models:
 t+=f'\n### {m}\n'
 rr=[]
 for per in periods:
  vals=[m,per,num(rv(m,per,'record_count')),f"{rv(m,per,'true_benign_fraction')*100:.4f}% / {rv(m,per,'true_attack_fraction')*100:.4f}%"]+[num(rv(m,per,k)) for k in ['tp','tn','fp','fn']]+[f'{rv(m,per,k)*100:.4f}%' for k in ['raw_error_rate','accuracy','benign_recall','attack_recall','balanced_accuracy']];rr.append(vals)
 t+=table(['Model','Period','N','Benign / Attack','TP','TN','FP','FN','The error rate','Accuracy rate','Benign recall','Attack recall','Balanced accuracy'],rr)
t+='## Overall Error in the Fidelity Discount Box\n'
for m in models:
 t+=f'\n![{m} Box ](figures/{m}_confidence_bins.png)\n'
 for sub in ['all','predicted_Benign','predicted_Attack']:
  t+=f'\n### {m} / {sub}\n';rr=[]
  for r in bins:
   if (r['model'],r['period'],r['prediction_subset'])!=(m,'overall',sub):continue
   c=clookup[m,sub,r['bin']];rr.append([r['bin'],num(r['n']),pct(r['bin_fraction']),num(r['errors']),pct(r['error_rate']),f"[{pct(c['low_95'])}, {pct(c['high_95'])}]",c['valid_replicates'],pct(r['mean_confidence']),pct(r['accuracy']),'NA' if r['confidence_minus_accuracy']=='NA' else f"{100*float(r['confidence_minus_accuracy']):+.4f}",num(r['true_benign']),num(r['true_attack']),num(r['fp']),num(r['fn'])])
  t+=table(['Box','N','Subset share','The number of errors','The error rate','95% CI','effective copy','Average conf','Accuracy rate','conf - accuracy rate pp','The real B','The real A','FP','FN'],rr)
t+='Small supporting example: RF predicts Attack [0.95,0.99) only as a 501 line; XGBoost predicts that Attack [0.999,1] is only 1185 and the range is wide. It is not possible to extend the extreme point estimates of these boxes to the whole.\n\n'
t+='## High confidence error: [0.99,1.0] fixed two boxes summarized\n'
t+=table(['Models','Period','Predicting subset','N','The number of errors','The error rate','Subset records','Total Raw error during the same period'],[[r['model'],r['period'],r['prediction_subset'],num(r['n']),num(r['errors']),pct(r['error_rate']),pct(r['fraction_of_subset']),pct(r['fraction_of_all_raw_errors'])] for r in high])
t+='Distribution of ## class condition scores\n'
t+=table(['Models','The real class','Group','N','min','q10','q25','q50','q75','q90','q95','q99','max'],[[r['model'],r['true_class'],r['group'],num(r['n'])]+[f"{float(r[k]):.10g}" for k in ['minimum','q10','q25','q50','q75','q90','q95','q99','maximum']] for r in dist])
for m in models:
 t+=f'\n### {m} ECDF is unified\n '
 for cls,suffix in [('Benign','0_1'),('Attack','0_1'),('Attack','0.9_1'),('Attack','0.99_1')]:t+=f'\n![{m} true {cls} ECDF {suffix}](figures/{m}_{cls}_ecdf_{suffix}.png)\n'
t+='\n## freeze threshold CDF and point mass\n\n<= contains the equivalent point quality, consistent with the historical correspondence category covering each verification.\n'
t+=table(['Models','The real class','Group','Depreciation schemes','q','N','<q Numbers / Example','=q Numbers / Example','>q Numbers / Example','<=q Numbers / Example'],[[r['model'],r['true_class'],r['group'],r['threshold_scheme'],f"{float(r['q']):.17g}",num(r['n'])]+[f"{num(r[k+'_count'])} / {pct(r[k+'_fraction'])}" for k in ['less','equal','greater','less_equal']] for r in cdf])
t+='The threshold equation cannot be ignored: for example, XGBoost Benign has a total mass of about 2.82% points at freeze q; RF Early Benign calibration q has a 11% point mass. The same q does not represent the same distribution, and the insertion value 90% does not always represent the 181 of the freeze.\n\n'
t+='## Raw confidence risk Acceptance Rate: Fixed Diagnostic Point\n'
for m in models:
 t+=f'\n### {m}\n'
 for per in periods:
  t+=f'\n![{m} {per} Raw curve](figures/{m}_{per}_raw_risk_coverage.png)\n'
  rr=[r for r in points if r['model']==m and r['period']==per]
  t+=table(['Period','Goal Acceptance','Practical Acceptance','Acceptance of numbers','Group size','The number of errors','risk','Benign withdraws / all real B','Attack withdraws / all real A','Accepting TP correctly','TP/ all A'],[[per,pct(r['target_coverage']),pct(r['actual_coverage']),num(r['accepted']),num(r['tie_group_n']),num(r['errors']),pct(r['risk']),pct(r['benign_recall_all_true']),pct(r['attack_recall_all_true']),num(r['accepted_tp']),pct(r['tp_fraction_all_attacks'])] for r in rr])
t+='## has both Variant and B operating points: just as above\n\nDirectly read existing indicators of time-scale controls and reconfirm them with the count of the history window; There are no new strategies.\n'
ctx={(r['model'],r['scheme'],r['period'],r['metric']):r for r in context}
for m in models:
 t+=f'\n### {m} freeze threshold\n '
 t+=table(['Programs','q_Benign','q_Attack','q and','1−q_Benign','1−q_Attack'],[[s]+[f'{v:.17g}' for v in [inputs[m]['frozen_thresholds'][s]['q_Benign'],inputs[m]['frozen_thresholds'][s]['q_Attack'],inputs[m]['frozen_thresholds'][s]['q_Benign']+inputs[m]['frozen_thresholds'][s]['q_Attack'],1-inputs[m]['frozen_thresholds'][s]['q_Benign'],1-inputs[m]['frozen_thresholds'][s]['q_Attack']]] for s in schemes])
 keys=['benign_coverage','attack_coverage','benign_singleton_rate','attack_singleton_rate','empty_set_rate','double_set_rate','singleton_error_rate','acceptance_rate','tp','tp_fraction_of_true_attacks'];rr=[]
 for s in schemes:
  for per in periods:rr.append([s,per]+[num(ctx[m,s,per,k]['value']) if k=='tp' else pct(ctx[m,s,per,k]['value']) for k in keys])
 t+=table(['Programs','Period','B coverage','A coverage','{B}','{A}','The Air Force','The two elements','Accept the risk','Acceptance rate','TP','TP/ all A'],rr)
t+='## subcategory box full annex (descriptive)\n'
for m in models:
 for per in periods[1:]:
  t+=f'\n### {m} / {per}\n'
  rr=[r for r in bins if r['model']==m and r['period']==per]
  t+=table(['Predicting subset','Box','N','Proportion of subsets','The number of errors','The error rate','Average confidence','Accuracy rate','conf - accuracy rate pp','The real B','The real A','FP','FN'],[[r['prediction_subset'],r['bin'],num(r['n']),pct(r['bin_fraction']),num(r['errors']),pct(r['error_rate']),pct(r['mean_confidence']),pct(r['accuracy']),'NA' if r['confidence_minus_accuracy']=='NA' else f"{100*float(r['confidence_minus_accuracy']):+.4f}",num(r['true_benign']),num(r['true_attack']),num(r['fp']),num(r['fn'])] for r in rr])
v=json.loads((P/'validation.json').read_text());a=json.loads((P/'delivery_validation.json').read_text())
t+=f"\n## verification, restrictions and products\n\nPerforming checks {len(v['checks'])} Adopted; independent delivery inspection {len(a['checks'])} Adopted. independent verifies the acceptance, iso-grouping, error, and confusion of all 80 target points directly to the original caching block according to confidence>= diagnostic boundary counts; There is no reusable sequence classification function. In addition, historical window-by-window counting is used to verify the entire sub-section of the context.\n\n "
t+='The number of Raw errors per window is historically consistent; 40 rating category / program / time combination CDF coverage consistent with history; In addition, the report also highlights the impact of the new policy on the environment. The curve 100% risk is equal to Raw; Empty boxes and zero splitters are approved. The matrix is read from history, with no new random sampling.\n\nNon-nuclear testing: not re-run model reasoning, caching identity according to model / features / caching manifest, source code and historical counting; Unresolved original CSV, labels and row sequences based on features construct logic, row sequences, window boundaries and save sample verification; Unverified labels return time; source domain controls are not running. No hashes counted. The graph has checked the representative box graph and the local CDF graph of Attack.\n\n'
t+='This time it’s just post-diagnosis, no modification of the original Outcome B and no guarantee of future data performance. No new deployment thresholds, models or calibration methods have been proposed or implemented, and are discontinued upon completion.\n\n'
t+=f' Definite output directory: `{P}`. All CSVs contain exact counts; save figures as PNG/PDF and keep source, commands, logs, and the full path inventory in the same directory.\n '
(P/'diagnostic_report.md').write_text(t)
print('REPORT',P/'diagnostic_report.md','bytes',len(t.encode()))
