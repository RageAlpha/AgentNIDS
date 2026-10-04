from audit_common import *
P=read(PILOT/'protocol.json');F=read(PILOT/'q_freeze.json');q=float.fromhex(F['q_serialized']['value'])
replay=read(OUT/'cache_replay_validation.json');csvval=read(OUT/'csv_reconstruction_validation.json');infer=read(OUT/'inference_validation.json');numeric=read(OUT/'numerical_validation.json');stats=read(OUT/'statistics_validation.json')
assert all(x['status']=='PASS' for x in [replay,csvval,infer,numeric,stats])
# Additional identity and sampling coverage assertions, no model/threshold fitting.
order=np.load(P['inputs']['UNSW']['original_rows'],mmap_mode='r')
train=order[:1420000];cal=order[1420000:1780000];test=order[2130000:2365424]
assert not len(np.intersect1d(train,cal)) and not len(np.intersect1d(train,test)) and not len(np.intersect1d(cal,test))
features=read(ROOT/'outputs/source_v2/models/rf_All49/training_manifest.json')['feature_names'];fm=read(ROOT/'outputs/feature_store_v2/feature_store_manifest.json')
assert len(set(features))==49 and features==fm['feature_order']==read(ROOT/'outputs/source_v2/preprocessing/preprocessing_manifest.json')['feature_order']
samp=rows(OUT/'frozen_sample_manifest.csv');lookup={(r['domain'],int(r['rank'])) for r in samp}
for d,lo,full in [('UNSW',2130000,23),('ToN',0,2752)]:
 for w in range(full):
  assert all((d,int(r)) in lookup for r in np.linspace(lo+10000*w,lo+10000*w+9999,17,dtype=np.int64))
assert all(('UNSW',int(r)) in lookup for r in np.linspace(1420000,1779999,3601,dtype=np.int64))
assert read(OUT/'sample_freeze.json')['ToN_accepted_included']==731
write('information_isolation_validation.json',dict(status='PASS',S_train_S_prob_cal_intersection=0,S_train_S_source_test_intersection=0,S_prob_cal_S_source_test_intersection=0,all_49_features_unique=True,feature_order_matches=True,forbidden_fields_absent=True,preprocessing_fill_all49_matches_S_train_recalculation=True,recorded_model_selection='No S_prob_cal-driven Raw RF selection path found; model saved before Platt fit',unrecorded_human_selection='UNKNOWN',full_fit_retrained=False,required_sampling_coverage=True))
# Every original frozen input and every old pilot/preflight output retained.
refs=rows(OUT/'artifact_differences.csv')
for r in refs:
 r['actual_sha256']=sha(r['path']);r['match']=r['actual_sha256']==r['expected_sha256']
assert all(r['match'] for r in refs)
table('artifact_differences.csv',refs)
assert inventory()==read(OUT/'upstream_inventory_before.json')
write('upstream_inventory_after.json',inventory())
# Include exact provenance for every code/data source actually used, including verification-only auxiliary stats.
paths={r['path'] for r in refs}
for p in [ROOT/'outputs/feature_store_v2/UNSW_categorical.npy',ROOT/'outputs/round1_data_prep_v1/prepare_data.py',ROOT/'src/nids/data/feature_store.py',ROOT/'src/nids/data/preprocessing.py',ROOT/'src/nids/replay/predictions.py',ROOT/'src/nids/models/base_models.py',ROOT/'src/nids/p2.py',ROOT/'configs/execution_v2.json']:
 paths.add(str(p))
rs=[]
for p in sorted(paths):
 pp=Path(p);version='Exact content snapshot; see SHA256'
 if pp.suffix=='.json':
  try:
   ob=read(pp)
   if isinstance(ob,dict):version=str(ob.get('version',ob.get('completed_at_utc',version)))
  except Exception:pass
 rs.append(dict(path=p,version=version,sha256=sha(pp),bytes=pp.stat().st_size,mtime_ns=pp.stat().st_mtime_ns))
table('source_hashes.csv',rs)
write('read_access_ledger.json',dict(unit='Logical array slices/passes, not operating-system page reads. Whole-file hashing reads bytes without probability interpretation.',hash_checks='Initial and final SHA256 for all recorded pilot/preflight inputs and outputs; ToN CSV/X audit-start fingerprints added',replay=dict(calibration_probability_rows=360000,evaluation_probability_rows=27755424,evaluation_windows_or_tail=2776,legacy_expression_uses_same_loaded_p=True),sample_selection=dict(boundary_passes_per_partition=4,boundary_partition_rows=[360000,235424,27520000],source_Attack_directed_passes=2,original_row_entries_frozen=53813),csv_reconstruction=dict(raw_CSV_passes_per_domain=1,raw_csv_rows={'UNSW':2365424,'ToN':27520260},label_attack_rows_checked=28115424,end_time_identity_rows=29885684,sampled_feature_rows=53813),distribution=dict(probability_passes_per_partition=1,rows=[360000,235424,27520000],labels_read_same_ranges=True),inference=dict(frozen_model_loads=1,samples={'UNSW':5544,'ToN':48269},actual_batch_sizes={'UNSW':[5544],'ToN':[48269]},original_P3_batch_limit=131072,cache_comparison_rows=53813),fill_verification=dict(S_train_rows=1420000,numeric_features=37,categorical_features=12,RF_standardization=False),bootstrap='2000 index-only reconstruction passes over saved aggregate blocks; no new per-flow evaluation'))
(OUT/'producer_code_review.md').write_text('''# Original CSV → RF Caching production chain review

## identity and sequencing
`outputs/round1_data_prep_v1/prepare_data.py:139` read FLOW_END_MILLISECONDS/Label/Attack in the order of the original CSV;`:199` executes np.argsort(ends, kind="stable"), with the original order being zero base original_row; At the same time as the end of time, maintain the original order. identity is domain:original_row and cannot be used as a continuous CSV interval at the top of the window.
This audit reads the end time from the original CSV in a stepwise manner, using the original line→ rank versus mapping as the total quantity compared to sorted_end_ms, and verifies the complete alignment of mapping, the non-decreasing end time and the simultaneous increase in original_row. So it’s not just checking the timing of the two caches. The target end 260 only participates in the original time mapping completeness check and does not enter the pilot performance repositioning.
After `prepare_data.py:215` the partition is covered by a fixed split_ranges with the rules for generating the original window; The existing source partition and the complete ToN window remain unchanged. The original calibration / source test / training identity intersection is expressed as 0.

## CSV to feature store
`src/nids/data/feature_store.py:44` declares a training view of column 49, read from column CSV by column name, rather than by column CSV position intercept; Label/Attack is read separately, and port / protocols are retained in units of prime values. Clearly exclude IP, Label, Attack, start time and rank/ID auxiliary fields.
`:56` starts with the initial sequence float32 features, CSV starts with float64 analysis, unlimited conversion of NaN, and then float32; The category auxiliary matrix is float64 and is used for S_train category filler statistics.
`:106` in the same sorted_original_rows batch as gather, features, labels and attack code. Each batch of [start:stop] uses order with the same slice, stop=min(start+batch_rows,N), and the tail batch is not lost; It’s not the only time-lapse connection. Category encoded with the original Attack string dictionary mapping. This time, the reverse Charan CSV strings are reversed to freeze codebook, and the full quantity is compared without caching the labels.
audit rebuild the CSV float64 → Unlimited NaN → float32 temporary value of the 53,813 item sample by element, including missing state and bit mode, and replicate the preprocessing input; 2,636,837 elements are all consistent. It is not claimed that all feature store elements have been reconstructed.

## preprocessing and Model Training
`src/nids/data/preprocessing.py:49` limits UNSW S_train[0,1420000) statistics: the numerical value of finite median; The 12 category field uses the original float64 auxiliary matrix mode, with ties being the least; The whole 0 is missing. This time, independent recalculated the 49 values from the freeze S_train data, which are equivalent to npz per /hex value; It’s just a test, not a replacement for preprocessing. mean / standard deviation is not used in RF and has not been standardized.
`SourcePreprocessor.transform` raises the float32 of feature store to float64, unlimited filling of NaN, and finally float32 output. Direct CSV-to-RF reconstruction therefore preserves the initial P1 float32 quantization step, rather than silently bypassing it with original CSV float64 values.
`src/nids/p2.py:154` only transform S_train training;`:160` calls model fit to be saved; It was then that the S_prob_cal and fit Platt were reconciled.`base_models.py:111` RF is fixed to parameters, random_state, output_type="cupy", fit to pickle. Existing configuration of /manifest/ code without calibration feedback to the Raw RF modeling call path; Those who haven’t been preserved are still UNKNOWN for experimentation, and there is no code to prove that they don’t exist.

## P3 Written and batched
`src/nids/replay/predictions.py:75` takes [start:end] in succession according to the rank domain, with the initial batch limiting to 131072, before 49 in the preprocessing and contiguous columns; Raw takes the same RF predict_proba[:,1] and raises the float64. The clip/logit of `gpu_evidence` only constitutes an additional Platt probability, not covering Raw. Write cache[name][start:end] with absolute rank, the last batch with min and flush; The local rank without deviation is written as the global starting point.
Structural caching linear ID is a substantive audit limitation: a uniform shape or overall confusion cannot independently prove consistency. This time, the full-length CSV labels / category / time mapping, window-by-window volume repositioning and the 53,813 line CSV → model probability closure is reinforced. The sample contains each complete windows 17 point, all 731 reception records, the adjacent record of the original 131072 batch boundary and the directional boundary; Cached production was gradually replicated in these samples, but not rethinking all 27.52M targets.

## Implementation Conditions and Scope of Evidence
Recharging the original cuML26.8.0 RF, classes=[0,1], 49 inputs, CUDA float32 contiguous predict_proba; audit source / target is actually one line of 5544/48269, which is lower than the original 131072 batch limit. It is consistent with phase-by-phase caching, so there is no need to introduce the error margin or change the batch size positioning. There is no Platt, softmax, retrain, target calibration, or any other model.
''')
(OUT/'documentation_errata.md').write_text('''# document error (only in the audit directory, the original is unchanged)

1. Pilot numerical_contract.md copied the historical preflight contract, including scores/q not yet computed. Read that as **status at preflight**. Formal q_freeze.json and evaluation_start/complete show later computation/evaluation completed; the old sentence does not establish nonexecution. This independent reconstruction confirms the same q/results.
2. The audit source and model reasoning of PASS limit coverage. The original cache does not have the old flow prediction cluster trajectory, so the clock follows the old / and the new consistent thumb to the full probability flow independent reproduces the old expression that has been reviewed; The preserved old results were validated by 444160 in individual windows / category / labels /Raw/ set of tensor arrays and 282006 in CSV fields, not falsifying the reading of non-existent old phased-out files.
3. RF predict_proba has two layers with float32 as its original output, and audit finds that it is not always accurate as 1. The protocol has been fixed using the attack columns p and 1-p; The original Benign column difference is diagnostic only and is not used to rephase calibration, replace the prediction set, or change the gate.
''')
status_layers={
'algorithm':dict(status='PASS',scope='independent full calibration order statistic, exact k/q/score hash/ties, direct inclusive set definitions; no old helpers'),
'cache_replay':dict(status='PASS',scope='360000 calibration,235424 source test,27520000 ToN; all saved tensor cells and relevant CSV fields'),
'label_identity':dict(status='PASS',scope='28115424 original CSV labels+Attack strings;29885684 full-domain end-time mappings and permutation/stable order'),
'feature_reconstruction':dict(status='PASS',scope='SAMPLED:53813 rows,2636837 elements, rawCSV stage values and filled RF input bit match;49 S_train fill statistics match'),
'original_model_reinference':dict(status='PASS',scope='SAMPLED_INFERENCE:53813 rows,[:,1] bit exact,0 q-set changes; native two columns recorded'),
'numerical_boundaries':dict(status='PASS',scope='12042 exact Fraction comparisons plus adjacent float32 boundary cases; distribution counts over full required ranges'),
'statistics':dict(status='PASS',scope='all2000 original replicates,blocks,multiplicity,ratios,NA and within-replicate difference CIs independently reproduced; time-series adequacy not established'),
'protocol_consistency':dict(status='PASS',scope='old q/epsilon/model/ranges/gate and every recorded old artifact unchanged; no Week1')}
notverified=['All feature-store elements outside the 53813 reconstructed samples','Full-cache model re-inference beyond sampled rows','Undocumented human model-selection/training history','Stationarity/exchangeability and adequacy of10-window block length for future inference']
write('audit_validation.json',dict(version='week0-independent-audit-validation-v1.0',completed_at_utc=now(),conclusion='No problems found but still unverified links',layers=status_layers,not_verified=[dict(status='NOT_VERIFIED',item=x) for x in notverified],implementation_or_data_failures=[],native_probability_columns_diagnostic='Benign native column differs from1-p at float32 level; required Attack column is bit identical to cache; no protocol change',prior_NO_GO_EVIDENCE_supported=True,gate_scope='rf_All49/ToN/VariantA/Strict0/epsilon1/10 and existing frozen q only',old_q_hex=F['q_serialized']['value'],q_verification_only=True,old_artifact_hash_comparisons=len(refs),upstream_unchanged=True,Week1_started=False,professor_message_sent=False))
# Plain report generated from validated counts; no automatic gate selection algorithm.
cv=rows(OUT/'class_coverage_rejection.csv');diag=rows(OUT/'probability_score_diagnostics.csv')
def percent(x):return f'{float(x)*100:.6f}%'
def mdtable(h,rs):return '| '+' | '.join(h)+' |\n|'+ '|'.join(['---']*len(h))+'|\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rs)
layer_table=mdtable(['Levels','Status','Actual coverage'],[[k,v['status'],v['scope']] for k,v in status_layers.items()])
class_table=mdtable(['Partition','The real class','N','The real labels concentration rate','Raw is correct.','Raw correct but rejected'],[[r['partition'],r['true_class'],r['rows'],percent(r['coverage']),r['raw_correct'],r['raw_correct_rejected']] for r in cv])
quant_table=mdtable(['partition / class','p median','p Minimum / Maximum','p is the only value','p=0/1 number of items'],[[r['partition']+'/'+r['true_class'],r['p50'],r['min']+' / '+r['max'],r['unique_values'],r['zero_count']+' / '+r['one_count']] for r in diag if r['field']=='attack_probability'])
report=f'''# Week 0 independent code and upstream data audit

** Conclusion: No problems have been found, but there are still unverified links. The current NO_GO_EVIDENCE is still supported, limited to the current freeze RF-All49 / ToN / Variant A / Strict-0 / epsilon=0.1 conditions.**

Complete caching repositioning and verification of undetected differences with the original CSV labels / category / identity; After reconstructing features from 53,813 original CSV records and rerunning inference, attack probabilities match the cache bit for bit. There were no problems in achieving / data requiring isolation repair, modification of q or running back on old results. A re-modelling of all features elements and all records has not yet been reasoned, nor can it be proven that unrecorded people do not exist as selective models.

audit directory: `{OUT}`. Original pilot: `{PILOT}`; Original preflight: `{PREF}`. All old files are read-only and the original model /49 column / preprocessing / /epsilon/q/gate is unchanged; There is no script that will be written in the old directory.

## 1. classification determined

{layer_table}

NOT_VERIFIED is also specified: full feature-store element-by-element reconstruction, full cache model rethinking, unpreserved artificial history, sequence interchangeability and block-length applicability. PASS is interpreted according to the scope of the table and is not expanded to the entire project.

## 2. independent Probability→ Results Replaced

There is no call or import of the old scores/threshold/prediction_sets/summarize. Reference to the use of integers k=((n+1)*9+9)//10=324001 (one set), np.partition sequence statistics instead of old sequence functions; Calculate the q={q!r}，hex=`{q.hex()}` is the same as freeze in phases.
calibration percentage SHA256 `{F['score_sha256']}` is consistent; Less than / equals / and higher than q and 287840/38470/33690 respectively. 313934/325600 and 12376/34400 are the same for each class of calibration input.
The 235424 and ToN27520000 rows of the source test are recalculated, and the two candidates are strictly determined to be p<=q and 1.0-p<=q; p is always a Attack probability, with no softmax, clip, round or Platt.
The full 27755424 appraisal of probability flows is consistent with the old comparison path of local re-expression; The old voltage 444160 characteristics, the corresponding CSV282006 numerical / fractional fields are consistent, including / by window and / by time, rather than just nuclear overall confusion.
The old experiment did not preserve the sequential prediction cluster trajectory, and this audit as a real differential sequence expresses the replication of the old voltage of the cyanide and the cyanide conserved by the cyanide. There is no difference, replay_differences.csv is an empty table; The first difference is not fictional. The caching repositioning itself does not prove that the production of caching is correct, and the next layer is another closed loop.

## 3. Original CSV identity, labels, category with features

Source CSV The end time of the entire 2365424 line, ToN the entire 27520260 line is aligned with the order coordinates of the freeze through the original line→ rank mapping; mapping is the complete unique sequence, ending time non-decreasing, with the same time the original line number increases. Full 29885684 time mapping is correct. source training / calibration / source test identity two two two non-intersecting; Windows and tail rules are not reconstructed.
From the original CSV, read the Label and Attack strings, with calibration 360000, source test 235424, target 27520000 lines correspond to line-by-line caching, total 28115424 lines, labels difference 0, attack class difference 0. labels proves that it does not come from another co-source cache. ToN ends with 260 only at the time of nucleation mapping and does not enter performance repositioning.
The sample is {read(OUT/'sample_freeze.json')['frozen_at_utc']} The freeze re-thinking started at {infer['inference_started_at_utc']}. For each full-source / target evaluation window, 17 is fixed at the same distance as rank, 17 at the end of the source, and calibration 3601 at the same point; Also include all 731 accepted ToN records, near-boundary cases, source Attack coverage/rejection, and original inference-batch boundaries. It’s all about the 5544, the ToN48269, and the 53813. The seed 20260915; Targeted samples are used only for audit and are not used as independent performance estimates.
Rebuild the 49 column name from the original CSV by training manifest and retain the original unit and P1 for float64 analysis→ Unlimited NaN → float32 storage, then fill in float64 value→ float32 RF input. The / missing / bit pattern matches the 2636837 elements, and the input is consistent after filling. The 49 column is the only one with no Label/Attack/IP/ time /rank/ID is mixed. The result of each row is feature_comparison.csv.
All 49 values are recalculated from freeze S_train[0,1420000) independent: 37 numerical values finite median, 12 category auxiliary float64 mode (minimum value above ties) are the same as freeze npz. Just verify that fit / replaces preprocessing without replacing it. Code /manifest shows that RF is only recorded with S_train fit and saved first, then S_prob_cal with Platt and indicator; The Raw model has not been detected and the path of re-selection has not been recorded.

## 4. Original RF Reassessment of the closed loop

The original cuML26.8.0 model is reloaded, with classes_=[0,1] input, 49 input; CUDA contiguous float32 samples rebuilt directly to the original CSV predict_proba. The actual 5544/48269 batch of the source / target, with the maximum batch remaining 131072; There is no caching probability of participating in model input.
53813 paragraph [:,1] is equal to raw_probability in phases, the error sample 0, the maximum absolute error and all specified error scores are 0; The set changes 0 at a fixed q. Not used, not expanded or changed backend due to inconsistency. The conclusion is only **sampled inference PASS** and does not claim to be consistent with the full-size model.
The two columns of native output both retain decimal /hex. p_Benign_native and 1-p_Attack are not always equal in degrees, and the maximum difference between the source / targets is 1.30385160446167e-7 and 1.4156103134155273e-7, respectively; The two columns have the same maximum value as the deviation from 1. The attack column itself accurately restores cache. This is the diagnosis of two layers of the original float32, without replacing another in the protocol, without changing the q, without conducting alternative performance experiments.
The probability of a sample attack is unchanged and thus does not trigger an expansion to production batch positioning or total quantity reasoning; The total feature/model chain outputs maintain NOT_VERIFIED.

## 5. Why q is small, rejection rate extreme

{quant_table}

The complete fixed point min/1/5/10/50/90/95/99/max, the only value with 0/1 numbers, is probability_score_diagnostics.csv for both p and real category scores.
The source calibration Benign is 325600/360000 and p median is about 2.8641e-8; q is well located in the large ties group of 38470 class Benign. calibration Attack p median 0.9999995827674866 appears to be close to 1, but its median 4.172325134277344e-7 score is still greater than q. This is the Raw difference between lithium and lithium through the extreme single-element threshold.
Under a fixed contract, Benign is added to the aggregate demand for p<=3.145372886592668e-7; Attack must meet 1.0-p<=q directly, corresponding to the boundary approximation {1.0-q!r}（hex `{(1.0-q).hex()}`). The maximum p of any ToN class is only 0.989444375038147, so the minimum Attack candidate score 0.010555624961853027 is still much larger than q and no single element of Attack is present. Only the 731 reaches the Benign side threshold. When typical p lies between the two extreme thresholds, neither label enters the prediction set, naturally producing almost entirely empty sets.
The source probability is highly concentrated near the 0/1 of float32 and has a large ties, but not the newly made audit probability is substituted for the / intersection. ToN with 1 clause p=0 without p=1; It’s not a hypothesis that all the probabilities of a target are caused by the endpoint. 12042 for a precise reference comparison of Fraction plus adjacent float32 boundary tests with no definite difference, see boundary_counts.csv/numerical_validation.json. Approximately 1e-7 disagreement between the two native columns cannot explain the large gap between ToN minimum Attack candidate scores near 1e-2 and q; The path of addition numbers has not been changed accordingly.

{class_table}

The Raw classification of source calibration 34400 Attack is completely correct, but 22024 is rejected; The source test 20016 Raw correct Attack of 12959 was rejected; another 1 original Raw error Attack was also rejected). ToN The correct Raw and Attack7606499 are all rejected and cannot be considered as an attack detection.
The total calibration concentration of 90.641667%, the total 91.309297% of the source test, coexists with a severe imbalance in the two categories. Source Attack covers only 35.98%/35.26% and target Attack covers 0. The standard VariantA is marginal calibration and does not provide a commitment for each 90% class. Mixed distributions and imbalances reveal verifiable overweight structures; It is not possible to claim that the only causal function with a defined distribution shift or category ratio is based solely on these numerical values.

## 6. statistical review and applicable boundaries

The original 2752 window is blocked in the order of each 10 window, and the end of the 276 block is the 2 window; The ID/ range of the original window is consistent. PCG64(20260915) reconstructed 2000×276 index is consistent with shared multiplicity. Using the independent repositioning meter, the whole block is summed directly, and the whole ratio, the zero factor NA, the effective duplicate and the 95% linear percentile interval, including the two values of the duplicate difference, are exactly the same.
The source test only tested 3 blocks, which were not originally given the source bootstrap, and did not temporarily change the block length or deletion. The performance of the target time period is not uniform, and whether the fixed 10 window block adequately describes the unproven dependence; The correct calculation does not equate to the equilibrium / convertible validity. The range is still conditioned on freeze RF/q and observation sequence, with no source calibration re-sampling uncertainty, and no assurance of future domain. The TP range [0,0] is only because of all the current TP=0 blocks, which does not mean that the probability of detecting future must be 0. Only the 731 clause accepts an effect on the extrusion accuracy of the receiving subset, but does not make sense of the observed fact that the entire 27520000 clause is almost completely rejected.

## 7. protocol, documentary errors and conclusions

The original SHA256 and size/mtime files of the old model, cache, labels, configuration, preflight and pilot remain unchanged. Old scripts that are not running will be written to the old directory. audit is only validated by re-calculating q using independent; There are no new threshold choices, epsilon adjustments, retraining, Platt, target calibration, Mondrian or other models.
Real q in pilot numerical_contract.md uncalculated phosphorus is pre-set historical text copied and cannot be considered as proof that formal pilot has not been executed. This time, the documentation_errata.md was saved incorrectly and the original file was unchanged.
** The current NO_GO_EVIDENCE is still supported by ** : ToN rejects 99.99734375%, accepts only 731 clause, 616 misses Attack , TP=0 , rejects Raw error rate 15.9252% instead of receiving 84.2681%; It is also important to keep in mind that this is not the case in the United States. The negative contrast of the source / target with the time period is not due to the fact that the code / labels / sample features was not discovered or due to a re-thinking error.
This conclusion is limited to a specific scope of verification. have not been fully rethinked, reconstructed all feature store elements, verified unrecorded artificial history or time sequence statistical assumptions; Therefore, to summarize the unidentified but still unverified linkage, not unlimitedly excluding all upstream issues.
If there is a subsequent difference in the / data from the specimen, the proof of domain/rank/original_row/ features and p hex should be retained, the isolated positioning involves the production batch, the rebuilt version is repaired and only re-run in the new catalogue; This time, there is no problem that needs to be fixed.

** audit stopped after completion, Week1 didn’t start and didn’t send a message to the professor. Whether further progress is still required is reported and obtained with a confirmation.**
'''
(OUT/'audit_report.md').write_text(report)
print('Audit report and layered validation saved. No code/data mismatch found; unverified full inference/history explicitly retained.')
