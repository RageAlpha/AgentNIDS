"""Assemble evidence-based review; never modify original artifacts."""
import csv,json,ast
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent;R=Path('/home/karri/projects/NIDSProject_Professor_Run')
OLD=R/'outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z'
def js(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def put(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def save(n,rr):
 with (O/n).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def md(h,rr):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
def pct(a,b):return 'NA' if not b else f'{100*a/b:.6f}%'
u=js(O/'upstream_complete.json');ref=js(O/'reference_complete.json');sample=js(O/'sample_alignment.json');th=js(O/'threshold_comparison.json')
fp=rows(O/'feature_probability_comparison.csv');align=rows(O/'data_alignment_comparison.csv');boot=rows(O/'bootstrap_comparison.csv')
env=js(O/'environment.json');original_protocol=js(OLD/'protocol.json');frozen=js(OLD/'threshold_freeze.json');start=js(OLD/'evaluation_start.json')
cal=rows(O/'calibration_reference.csv');original_collection=js(OLD/'collection_summary.json')
assert original_collection['stop_original_row']==int(max(cal,key=lambda r:int(r['rank']))['original_row'])
assert original_collection['stop_rank_exclusive']==sample['scan_rows']
assert original_protocol['K']==400 and original_protocol['per_class']==200 and original_protocol['epsilon']==.1
assert original_protocol['ton_evaluation']==[100000,27520000] and original_protocol['source_evaluation']==[2130000,2365424]
assert original_protocol['ton_rows']==27420000 and original_protocol['ton_windows']==2742
assert original_protocol['ton_tail_excluded']==260 and original_protocol['source_tail_included']==5424
assert original_protocol['bootstrap']['seed']==20260915 and original_protocol['bootstrap']['replicates']==2000
assert original_protocol['registered_at_utc']<frozen['registered_at_utc']<start['started_at_utc']
assert all(start[k]==frozen[k] for k in ['q_Benign','q_Attack'])
assert frozen['registered_at_utc'] in (OLD/'prepare.log').read_text()
# Static check of historical set generation: no label argument or external label reads.
source=(OLD/'pilot.py').read_text();tree=ast.parse(source)
functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
setfun=functions['sets'];assert [x.arg for x in setfun.args.args]==['p','qb','qa']
assert {n.id for n in ast.walk(setfun) if isinstance(n,ast.Name)}<={'p','qb','qa'}
prep_source=ast.get_source_segment(source,functions['prepare'])
assert not any(token in prep_source for token in ['strict_k_reply4/', 'Cpreflight/', 'run_20260915_C1', 'run_20260915_C2'])
assert "x['cache']['raw_probability'][ranks]" in prep_source
lineage=[
 ('outputs/round1_data_prep_v1/prepare_data.py','original CSV data rows are zero-based; Label int is preserved; availability FLOW_END_MILLISECONDS; stable argsort implies original-row tie break'),
 ('outputs/round1_data_prep_v1/round1_config.json','replay.sort_keys and availability rule; source and target window policies'),
 ('outputs/round1_data_prep_v1/run_config.json','historical preparation executable and recorded settings'),
 ('configs/execution_v2.json','original CUDA environment, S_train fill rules, 131072 inference limit, detector float32, RF parameters'),
 ('configs/feature_views.csv','49-feature column order; All49 retains all 49'),
 ('src/nids/data/feature_store.py','CSV feature float64; empty/nonfinite to NaN; float32 store; gather by prepared original-row permutation; direct binary Label'),
 ('src/nids/data/preprocessing.py','S_train numeric finite median; categorical mode smallest tie; saved fill; tree input float64 impute then float32, no scaling'),
 ('src/nids/p2.py','UNSW S_train fitting boundary and frozen source preprocessing applied to original numeric representation'),
 ('src/nids/models/base_models.py','RF-All49 standardized=False; cuML RandomForestClassifier output_type=cupy; pickle persistence; direct predict_proba'),
 ('src/nids/replay/predictions.py','Raw probability is predict_proba[:,1] cast float64; saved rank-preserving cache; Platt in separate fields'),
 ('outputs/source_v2/models/rf_All49/training_manifest.json','frozen RF settings, features and training rank provenance'),
 ('outputs/source_v2/preprocessing/preprocessing_manifest.json','saved S_train fit scope [0,1420000), feature order and fill statistics provenance'),
 ('src/nids/evaluation/outer_data.py','C1 heldout 10k window identity contract'),
 ('outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/temporal_scope.csv','ToN four segments 686/686/685/685 windows')]
# Read complete source/config files; collect review-relevant line references, no execution.
evidence=[]
for path,meaning in lineage:
 text=(R/path).read_text();evidence.append(dict(path=str(R/path),lines=len(text.splitlines()),reviewed_rule=meaning))
save('code_and_config_evidence.csv',evidence)
compliance=[]
def comply(item,basis,evidence,status='Compliance'):
 compliance.append(dict(item=item,authority=basis,status=status,evidence=evidence))
comply('freeze RF-All49, Raw Probability, not training','reply §4 introductory paragraph / Step 2','original model.pkl loaded; n_features_in_=49; classes_=[0,1]; full inference bit-identical; historical pilot has no training or model inference calls')
comply('Initial 200 phase Benign and 200 phase Attack','reply §4 Step 1','raw CSV labels plus independently verified two-key chronology recover all 400 manifest rows exactly; scan 4020, stop rank4019')
comply('Unused or replenished C1/C2 samples','reply §4 Step 1','historical preparation source has no C1/C2 sample input; raw chronology reproduces current sample; old/current intersection=1 is incidental, sample sets differ')
comply('181 for each class','reply §4 Step 2','Python independent sorted order; both cached and reinferred thresholds exactly match; positions179–183 exported')
comply('Includes categories with equality','reply §2 inequalities; explicit numerical rule in execution plan §4.1 / user request','scalar candidate enumeration checks equal and adjacent values; independent bit-state replay matches all windows')
comply('Thresholds registered earlier than evaluation','reply §4 Step 2','protocol 09:04:40.217328Z < freeze 09:04:40.319330Z < evaluation 09:04:53.163333Z; logs agree; retrospective artifact evidence, not external trusted timestamp')
comply('Excluding the entire launch period','reply §4 Step 3; concrete ranks from execution plan/user','reference replays [100000,27520000); all startup excluded despite collection ending at4020; original260 tail excluded')
comply('Four phases with C1','reply §4 Step 3; C1 temporal_scope.csv','exact C1 rank boundaries; 686/686/685/685 windows; not old Week0 688×4')
comply('Full output reply 10 indicators','reply §4 Step 3','eight ToN metrics overall/four segments plus two UNSW class coverages; CSV numerators denominators and Markdown units verified')
comply('Raw in the same range','reply §4 Step 4; p>=0.5 convention from execution plan/user','independent raw baseline on 27420000 rows; separate singleton unique-label error; labels do not enter set generation')
comply('No-go Depends and Stops','reply §4 Step 4; user stop-after-pilot instruction','TP>0 in all segments, but singleton risk≥Raw; explicit No-go; no subsequent Week1 execution in this review')
comply('275 blocks, 2000 paired copies, PCG64 seeds','execution plan §4.3 / user; NOT separately specified in professor reply','last block2 windows; 550000 matrix cells identical; multiplicity-based independent aggregation and percentile computation match')
save('reply_compliance.csv',compliance)
# No post-hoc tolerance. All numerical equality checks are exact.
assert ref['differences']==0 and ref['sampling_matrix_differences']==0
assert sample['status']=='PASS' and th['status']=='PASS'
assert all(int(r['differences'])==0 for r in boot)
assert all(int(r[k])==0 for r in align for k in ['label_mismatches','label_family_mismatches','invalid_labels','independently_sorted_rank_mismatches','raw_timestamp_mismatches','window_boundary_mismatches','duplicate_original_rows','missing_original_rows'])
assert all(int(r[k])==0 for r in fp for k in ['feature_mismatch_rows','feature_mismatch_elements','feature_bit_mismatch_elements','imputed_input_mismatch_elements','probability_unequal_rows','prediction_set_changed_rows','raw_prediction_changed_rows'])
assert [int(r['probability_compared_rows']) for r in fp]==[400,27420000,235424]
assert env['status']=='PASS'
oldnow={str(p.relative_to(OLD)):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in OLD.rglob('*') if p.is_file()}
assert oldnow==js(O/'old_inventory_before.json')
put('old_inventory_after.json',oldnow)
layers=[
 ('Implementation of the Protocol','PASS','The reply is consistent; Specific rank, numerical comparison rules and bootstrap settings are attributed to the execution program / user instructions, rather than the erroneous teaching of the original specifications.'),
 ('Sampling and identity','PASS','The original CSV double-key sequence, early 200+200, 4020 scan stop, 400 identity and the window are all consistent; Unweighted / Leaks and Assessment Exchange'),
 ('Primary labels and features','PASS','Both domains are primitive CSV labels and sequential algebra; Specify the three ranges 49 features for the elements and locations to be consistent; The original freeze refill applies consistent'),
 ('Probability of the original RF','PASS','CUDA environment compatible with the same pickle, 49 features /classes_[0,1]; The full 27655824 is reconsidered and is consistent with the Raw float64 caching'),
 ('Two thresholds','PASS','Each class of independent is ordered by the 181 item, with five registered values completely consistent; The calibration is also consistent.'),
 ('Collections and indicators','PASS','independent state / true value /Raw predicted joint rectangle graph; 2742 window, four sections, overall, source 23 complete windows and tail are consistent'),
 ('bootstrap','PASS','275 block and end block 2 window; the 550000 index; 2000 coupled duplicates; The three 95% ranges are all exactly the same.'),
 ('Report with No-go','PASS','The original report values, unit proportions and deadline text are applicable to the data; independent dynamics is determined to be No-go')]
notes=[
 'In the original report, the entire vector of the independent polymerase cannot be considered as a validation of the independent algorithm: the same counts function only checks the path of the polymerase. The independent combined-state algorithm supplements this evidence without affecting the original values.',
 'The original report() writes the contribution of paragraph 3/4 and the reduction of paragraph risk of paragraph 4 only; This data is completely correct. If future is duplicated, it should be generated from the current counting dynamics. This time the historical script or report was not modified.',
 'Registration time verification based on existing protocols, freeze documents, evaluation of documents and logs, and does not claim to have an external credible timeline or document summary certification.',
 'Verification using the original freeze S_train filler and the original model; No models were retrained, preprocessing was not refitted, and stepwise execution of model training was not audited. This limit does not affect the comparison of the original CSV → freeze preprocessing → freeze RF → caching quantity required for this requirement.',
 'This time in the original CSV sequence of blocks, once not exceeding the upper limit of the original 131072 clause, the block composition is different from the original rank batch. The total output remains consistent, with no numerical difference.']
validation=dict(classification='C',conclusion='Within the scope of the full declaration, the results are consistent and the No-go remains valid.',completed_at_utc=datetime.now(timezone.utc).isoformat(),
 layers=[dict(layer=a,status=b,evidence=c) for a,b,c in layers],material_errors=0,not_verified_requested_layers=[],
 comparison_fields=ref['checked_fields'],field_differences=0,bootstrap_matrix_cells=550000,bootstrap_matrix_differences=0,
 full_reinference_rows=sum(int(r['probability_compared_rows']) for r in fp),feature_elements=sum(int(r['feature_elements']) for r in fp),
 probability_bit_equal_rows=sum(int(r['probability_bit_equal_rows']) for r in fp),probability_max_absolute_difference=0.,changed_prediction_sets=0,
 original_files_unchanged_by_size_mtime=True,checks_not_file_digests=True,review_only=True,notes=notes)
put('review_validation.json',validation)
c=ref['overall'];risk=c['singleton_errors']/c['accepted'];raw=c['raw_errors']/c['n'];delta=(raw-risk)*100
report=f'''# ToN Variant B mini-pilot independent Full review\n\n**C. results are consistent within the scope of the full declaration and No-go remains valid.** has not detected material errors affecting the results; Eight layers are PASS and the full RF reassessment required for this is complete, with no caching or sampling replacement.\n\nIt’s the first time I’ve seen this.{OLD}`.\nChecklist: `{O}`.\n\n## classification conclusions\n\n '''+md(['Nuclear testing','Conclusions','Evidence'],layers)
report+='\n\n## method, independent sex and source\n\nprepare/evaluate that does not execute the original pilot.py, does not import or call its sets, counts, metric_rows, aggregate, report. In the original CSV flow analysis, availability is represented by FLOW_END_MILLISECONDS, and the same time series breaks the ties sequence by starting from zero with the CSV data line. This time, two independent lexsort keys, each rank checking the original mapping, and verifying that all the original lines occurred once. labels directly reads the original CSV Label (0=Benign , 1=Attack) and verifies the Benign relationship of the Attack text category; There is no caching labels to prove itself.\n\nThe set of independent is encoded as an empty /Benign/Attack/ binary four-state, and is then combined with the original CSV real value and the Raw classification to form a 16 integrated rectangle. The real calibration value is used only to replicate the established 400 points; Assessment values do not participate in set generation. Windows-by-window counts are derived from the Unified Table, which is different from the original counts. All the original CSV results have been read in sequence; For more details see comparison_summary.csv and metric_differences.csv.\n\nPhase-by-phase protocol verification see reply_compliance.csv and use of phosphorus in accordance with / with insufficient evidence of deviation from / in phosphorus class III; All of the projects listed here are in compliance. Specific rank, p>=0.5 and bootstrap parameters come from execution schemes or user agreements, without pretending to teach the original. Source code and configuration sources see code_and_config_evidence.csv.\n'
report+='\n## Original identity, labels and windows\n\n'+md(['domain','CSV records','labels difference','independent ranking differences','Differences in timing','Repeat / Leakage','Differences in window terminals'],[[r['domain'],r['csv_rows'],r['label_mismatches'],r['independently_sorted_rank_mismatches'],r['raw_timestamp_mismatches'],r['duplicate_original_rows']+'/'+r['missing_original_rows'],r['window_boundary_mismatches']] for r in align])
report+=f'''\n\nInitial labels support: Benign={sample['support_100k'][0]:,}、Attack={sample['support_100k'][1]:,}. independent The initial 200+200 corresponds to the rank, original_row, labels, window_id and window_index of the original manifest. Scanning {sample['scan_rows']:,} Finally, rank={sample['stop_rank']} Originally known as ={original_collection['stop_original_row']}; Scanning the forehead Benign={sample['prefix_benign']:,}、Attack={sample['prefix_attack']:,}. 400 is the only one, all of which are in the start-up phase, and does not interfere with the evaluation.\n\nThis wheel sample is different from the historical C1/C2 set of 400 points, with only 1 points naturally intersecting; The original code was not read in the older sample, and the independent prime time sequence demonstrates that this round follows the earliest 200/ class rules. It’s not the same thing as recycling.\n\nToN evaluates [100000,27520000) , 27,420,000 items, 2,742 complete windows; The previous 100,000 and tail 260 excluded. The actual C1 manifest has been compared in four sections: 686, 686, 685, 685 windows. UNSW source tests [2130000,2365424), 235,424, including 23 complete windows and 5,424 tail. Complete sorting and comparison of labels covers the entire range, in order to verify the overall temporal order and identity; features / reasoning only covers three ranges of requests.\n '''
report+='\n## Primary features → freeze RF → Raw Probability\n\nAccording to the configs/feature_views.csv sequence, read 49 from CSV as float64; the empty and non-finite values values are unified as NaN; Converting float32 to reconstruct the original feature to store. After a step-by-step comparison, float64 applies S_train in the original preprocessing_stats.npz and converts float32 into RF input. Keep the original units of value, with no additional units of conversion, standardization, softmax, clip or round. RF-All49 is not standardized. The reasoning input comes from reconstructing the original features, not using feature store as the reasoning input.\n\n'+md(['Scope','Record count','features number of elements','features Difference between / elements','Probability is consistent','The probability is inconsistent','Maximum absolute difference','Collective change'],[[r['domain']+'/'+r['scope'],r['compared_rows'],r['feature_elements'],r['feature_mismatch_rows']+'/'+r['feature_mismatch_elements'],r['probability_bit_equal_rows'],r['probability_unequal_rows'],r['max_abs_probability_difference'],r['prediction_set_changed_rows']] for r in fp])
report+=f"\n\nTotal {validation['full_reinference_rows']:,} The article reiterates,{validation['feature_elements']:,} A features element is perfectly consistent. freeze model classes_={env['classes']}，n_features_in_={env['n_features_in']}; P(Attack) comes from predict_proba[:,1], float32 output directly rises to float64 and represents perfect consistency with caching. The environment:{env['device']}、cuML {env['cuml']}、CuPy {env['cupy']}、NumPy {env['numpy']}、CUDA runtime {env['runtime_version']}. Batch and model parameters are preserved in environment.json / inference_batches.csv. The maximum difference is 0 and the tolerance is not set or released; The re-thinking of the combined counts per window is the same, without changing any indicator or No-go.\n "
report+='\n## thresholds and boundaries\n\n'+md(['Projects','independent value'],list(th['values'].items()))
report+='\n\n'+md(['Categories','181','Less than the threshold','equals devaluation','Higher than the threshold','Re-reasoning for consistent thresholds'],[[r['class_name'],r['cache_q'],r['below'],r['equal'],r['above'],r['reinferred_q_matches_cache']] for r in th['classes']])
report+='\n\ncalibration_reference.csv preserves the original labels, the original caching probability, the re-reasoning probability and the corresponding score of the 400 line; threshold_order_neighborhood.csv preserves 179 and 183 for each class. The original cached independent sequence is fully consistent with the five freeze registration values; The new reasoning threshold is for diagnostic purposes only, without replacing the original threshold. Equivalent and adjacent floating point values, non-symmetric thresholds, vacuum clusters, binary elements and zero acceptance NA are all subject to reference checks. Thresholds were recorded earlier than the 12.844 rating. The two threshold values are >1 and the condition interval covers the entire [0,1] and the test vacuum set is 0 and corresponds to the mathematical relationship.\n'
report+='\n## independent indicator and No-go\n\n'+md(['Overall indicators','Molecule / coefficient or count','Results'],[
 ['Benign singleton',f"{c['benign_singleton']}/{c['n']}",pct(c['benign_singleton'],c['n'])],
 ['Attack singleton',f"{c['attack_singleton']}/{c['n']}",pct(c['attack_singleton'],c['n'])],
 ['The Air Force',f"{c['empty']}/{c['n']}",pct(c['empty'],c['n'])],['The two elements',f"{c['double']}/{c['n']}",pct(c['double'],c['n'])],
 ['Benign singleton error rate',f"{c['fn']}/{c['benign_singleton']}",pct(c['fn'],c['benign_singleton'])],
 ['Attack singleton error rate',f"{c['fp']}/{c['attack_singleton']}",pct(c['fp'],c['attack_singleton'])],
 ['TP',str(c['tp']),str(c['tp'])],['TP/ is all real Attack.',f"{c['tp']}/{c['true_attack']}",pct(c['tp'],c['true_attack'])],
 *[[f" Source testing {r['class_name']} Coverage ",f"{r['numerator']}/{r['denominator']}",pct(int(r['numerator']),int(r['denominator']))] for r in rows(O/'reference_source_class_coverage.csv')],
 ['singleton overall error rate',f"{c['singleton_errors']}/{c['accepted']}",pct(c['singleton_errors'],c['accepted'])],
 ['Raw Acceptance of errors',f"{c['raw_errors']}/{c['n']}",pct(c['raw_errors'],c['n'])]])
report+='\n\n'+md(['Section 1','TP','Total TP','TP/ This paragraph is real Attack','singleton error rate','Raw error rate'],[[i,x['tp'],pct(x['tp'],c['tp']),pct(x['tp'],x['true_attack']),pct(x['singleton_errors'],x['accepted']),pct(x['raw_errors'],x['n'])] for i,x in enumerate(ref['temporal'],1)])
report+=f"\n\nAll four sections have a TP; singleton risk is less than Raw in only 4 segment. singleton differs from Raw predictions {c['singleton_raw_disagreement']:,} The only labels miscalculation for singleton is that Raw labels is not miscalculated. The total number of controls for each window, each section and each source test.{ref['checked_fields']:,} A numerical / text field comparison with 0 difference. All proportional divisors and percentage units of / are identical; The reference value of the zero denominator is defined as NA. For full reference, see reference_metrics.csv and reference_source_class_coverage.csv.\n "
report+='\n## paired with bootstrap\n\nindependent builds a block of 275 and a block of 2 in a continuous 10 window. PCG64(20260915) re-generates the 2000×275 matrix, which is completely consistent with the original 550,000 element. independent constructs the average number of times for each draw, summing the combined state counts by the integer matrix times, and then calculating the total sample of / for each duplicate total error and the acceptance quantity of / for each error; It is different from the original indexing and realization. Each replicate difference subtracts the two risks within that replicate. Do not average window risks or subtract endpoints of independent intervals. The independent sequence and the linear insertion value reconstruct the percentile interval.\n\n'+md(['Indicators','Estimated','95% lower boundary','95% boundary','effective copy'],[[r['metric'],r['estimate'],r['lower_95'],r['upper_95'],r['valid_replicates']] for r in ref['intervals']])
report+=f"\n\nThe two risk units in the table are fraction; The unit of variance is percentage points. Raw−singleton={delta:.12f} A percentage point and a negative value indicates that singleton is worse. All block counts, 2,000 duplicates, / molecule / risk / differential values, and the three ranges are exactly the same. the / threshold is determined on the basis of the original model and the fixed calibration sample; The calibration set is not re-sampled.\n "
report+='\n## report on wording, correction of recommendations and border checks\n\n'+ '\n\n'.join(f'{i}. {note}' for i,note in enumerate(notes,1))
report+='\n\nThe original stage_report and gate_assessment tables, molecules, subunits, units, fractional contributions and No-go texts are all dynamically verified by reference; report_numeric_comparison.csv lists verification items, dynamic_reference_report.md is all generated based on this round of reference counts. There is no need to modify the results of the experiment or to repeat the formal experiment. If the future replicates the original script, it is recommended that the deadline text be removed only in the new version, and that the same function be checked as a polymerase consistency check box. This time, the historical product was preserved unchanged.\n'
report+=f"\n## finally decided and stopped\n\nNo-go Condition 1 (TP=0): Not established, TP={c['tp']:,}. No-go condition 2 (singleton overall error rate ≥Raw ): established,{100*risk:.6f}% ≥ {100*raw:.6f}%. Therefore, **No-go is still **, not dependent on the addition of materially to lower or cross-sectional consistency thresholds.\n\nUnfilled document summary calculates / verification, training, modification, re-selection of samples, threshold modification, scope expansion, other models of / target domain or Week1. The size of the original running directory file is fully consistent with the time before and after the modification; This inspection is only for the reading of border evidence. Reviewing the code, reference aggregations, first differences (this time to zero), commands and logs are all in this directory. There is no preservation of the full features, probability or trajectory of the flow. Stop when it’s done.\n "
(O/'review_report.md').write_text(report)
print(json.dumps(validation,ensure_ascii=False,indent=2))
