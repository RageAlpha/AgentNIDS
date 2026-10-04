import json, hashlib, csv
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent; R=Path('/home/karri/projects/NIDSProject_Professor_Run')
def save(n,v): (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def md(n,s): (O/n).write_text(s.strip()+'\n')
metrics={
'prediction_set_coverage':'sum(1[y in C])/N; overall and separately true Benign/Attack',
'acceptance_rate':'sum(1[|C|=1])/N',
'singleton_rate':'acceptance_rate (same quantity, explicit alias)',
'multi_label_rate':'sum(1[|C|=2])/N',
'empty_set_rate':'sum(1[|C|=0])/N',
'mean_set_size':'sum(|C|)/N',
'raw_pred':'1[p>=0.5]',
'raw_error':'raw_pred != y',
'four_set_counts':'For {}, {Benign}, {Attack}, {Benign,Attack}: N, Raw correct count, Raw error count, Raw error/N_set',
'rejection_rate':'P(R)=N_R/N, R={|C|!=1}',
'raw_error_given_rejected':'E_R/N_R',
'raw_error_given_accepted':'E_A/N_A',
'rejected_given_raw_error':'E_R/E_all',
'rejected_given_raw_correct':'(N_R-E_R)/(N-E_all)',
'accepted_confusion':'TN/FP/FN/TP using unique label of singleton as accepted prediction',
'selective_risk':'(FP+FN)/(TN+FP+FN+TP)',
'accepted_benign_FPR':'FP/(TN+FP)',
'accepted_attack_miss_rate':'FN/(TP+FN)',
'effective_attack_detection':'TP/N_all_Attack',
'wrong_accepted_attack_fraction':'FN/N_all_Attack',
'attack_abstention_rate':'(N_all_Attack-TP-FN)/N_all_Attack',
'benign_acceptance_rate':'(TN+FP)/N_all_Benign',
'attack_acceptance_rate':'(TP+FN)/N_all_Attack',
'correct_record_loss':'N(rejected AND raw_correct)/N(raw_correct)',
'stratification':'overall, true Benign/Attack, frozen original attack_code taxonomy (including ToN scanning), and four chronological segments',
'zero_denominators':'JSON null + NA reason; CSV NA; counts 0 when appropriate, rates never zero merely because denominator is zero'
}
p={
'version':'week0-feasibility-preflight-draft-v1.0','run_id':O.name,'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'DRAFT_REVIEW_REQUIRED_NOT_AUTHORIZED_TO_RUN_PILOT',
'provenance':{
'roadmap_explicit':{'path':str(R/'docs/Research_Roadmap_Phase2_Conformal_Coordination.md'),'sections':['3.1','3.2','4.2','4.5','7 pre-registration','9 Week 0'],'rules':['existing RF-All49','ToN','Variant A Strict-0 source calibration','primary epsilon 0.10','prediction sets with <=','pilot before Week 1; professor informed and pilot confirmation required']},
'existing_project':['frozen exact partitions and availability-rank identities','P1 float32 features and S_train-only fill','P2 cuML RF with classes [0,1]','P3 raw_probability attack column, float64 cache','P7 original complete ToN windows and excluded tail','A/B1 full source test including 5424 tail; raw_pred >=0.5','2000 bootstrap replicates, 10 windows/block and final short block'],
'current_user_required_clarifications':['RF predict_proba directly; roadmap softmax wording not applied to RF','finite order statistic, rational epsilon, no interpolation','full P7 ToN scope without C1/C2 startup exclusion','synthetic only algorithm tests; q reserved for pilot','no new target probability analysis','separate set coverage and acceptance','all requested metrics and four gate states'],
'new_proposed_execution_choices':['CPU NumPy float64 conformal arithmetic on exact P3 float64 cache','main pilot descriptive statistics; no confidence intervals proposed in this draft','human evidence review, no invented minimum effect size','no C1/C2 auxiliary subrange in initial pilot']},
'model_id':'rf_All49','model_path':str(R/'outputs/source_v2/models/rf_All49/model.pkl'),'target':'ToN','variant':'A','strict_k':0,'epsilon':{'numerator':1,'denominator':10},
'calibration':{'domain':'UNSW','partition':'S_prob_cal','rank_start':1420000,'rank_end_exclusive':1780000,'rows':360000,'benign':325600,'attack':34400,'all_records':True,'balance_or_resample':False,'probability_field':'raw_probability','k_one_based':324001,'k_zero_based':324000,'q':None,'q_status':'NOT_COMPUTED'},
'evaluation':{'source':{'rank_start':2130000,'rank_end_exclusive':2365424,'rows':235424,'retained_tail':5424},'ToN':{'rank_start':0,'rank_end_exclusive':27520000,'rows':27520000,'windows':2752,'excluded_tail':260,'startup_exclusion':0,'window_reference':'evaluation_window_references.csv'},'comparison_subrange':None},
'algorithm':{'p':'raw_probability; p_Benign=1-p, p_Attack=p','score':'where(y==0,p,1-p)','k':'ceil((n+1)*9/10), implemented as ((n+1)*9+9)//10; one-based','q':'sort(scores)[k-1] if k<=n else +infinity','prediction_set':'Benign iff p<=q; Attack iff 1-p<=q','ties':'inclusive <=; entire equal-score group, deterministic','quantile_interpolation':None,'extra_softmax':False,'extra_clipping':False,'rounding':False,'dtype':'float64; preserve existing float32 inference rounded values stored exactly as float64','serialization':'float64.hex/fromhex; +infinity as {kind:positive_infinity}; no JSON nonfinite numeric literals','label_boundary':'test set generation function accepts only p and q, never y'},
'metrics':metrics,
'time':{'window_counts':[688,688,688,688],'rank_intervals':[[0,6880000],[6880000,13760000],[13760000,20640000],[20640000,27520000]],'threshold':'same frozen source q in all segments','retune':False},
'statistics':{'selected_draft':'DESCRIPTIVE_ONLY','confidence_intervals':False,'seed_for_selected_draft':None,'no_bootstrap_matrix_generated':True,'optional_amendment_before_pilot':{'replicates':2000,'block_windows':10,'keep_final_short_block':True,'min_blocks':20,'target_blocks':276,'target_final_block_windows':2,'source_blocks_if_required':3,'source_reason':'23 complete windows plus existing 5424-row tail; consecutive groups: 10 windows,10 windows,3 windows+tail; below minimum 20, therefore no source CI','target_rng':'numpy.random.Generator(PCG64(20260915))','sampling':'draw B block indices with replacement B times per replicate, keep whole blocks; pool counts and recompute denominators','CI':'95% percentile, NumPy quantile(method=linear) for bootstrap estimates ONLY, never for conformal q; exclude undefined replicates and report valid count','sharing':'same ToN matrix for all metrics/methods at identical scope; no C1 shrunken-scope matrix','target_minus_source':'descriptive difference; no source/target independent or paired CI claimed','per_segment_and_category':'descriptive only in this draft'}},
'gate':'gate_definition.md','open_protocol_choices':['approve complete scope and P3 raw float64 convention','approve descriptive-only pilot or amend ToN-only bootstrap before execution','approve qualitative evidence review; minimum effect remains unspecified','missing separate A-C2 synthesis: confirm existing reports suffice or provide synced location'],
'pilot_sequence_after_approval':['create a new pilot run directory and freeze reviewed protocol and source hashes','revalidate required input hashes, schema, identity and original environment','compute real calibration scores and q once using all S_prob_cal; freeze q hex, k, epsilon and calibration hashes before evaluation','evaluate same frozen q on complete source test and ToN; produce aggregate metrics, original-window summaries and four segments only','apply evidence review categories; report to user; professor report/confirmation before any Week 1'],
'forbidden':['retrain or replace model','fit Platt','use Platt probability/predicted_class/f(u)/C1/C2 thresholds','target calibration or Mondrian','epsilon/model/scope selection after target results','Week 1','message professor automatically'],
'boundary':'Previously unblinded P7/A/B1/B2/C1/C2 domains; exploratory follow-up, not first target contact or independent outer validation.'}
save('pilot_protocol_draft.json',p)
md('metric_definitions.md','# Pilot indicator and subdivision (only registered this time)\n\n'+'\n'.join('- **'+k+'**: '+v for k,v in metrics.items())+'\n\nEach class and the original attack class derive from the same global prediction set, without the actual class changing the accepted action. scanning retains its original name and code. Raw error decomposition and the predicted mixed matrix of the single element are retained, respectively, without replacing each other.')
md('gate_definition.md','''# Week 0 gate Draft: Audit Rules, Not the Conclusions of this Experiment

## Source and boundary
Roadmap § 9 requires target domain to indicate more multi-label and / or empty sets, the error focusing on the uncertain / refusal to record.
Aluminum meaningfully more aluminum and aluminum errors concentrated aluminum have no original value thresholds; 99%+ singleton is an example of a failure, not the only determining factor.
The following four state and indicator breakdowns are the executive refinement of this user request; roadmap does not have these state names or numerical rules.
The recommended minimum impact thresholds are not sufficiently met, so they remain under review and do not automatically fill in the easily passed figures. Directions are only indicative of directions, and do not alone prove useful.

## A. IMPLEMENTATION_INVALID
The model entity, class mapping, freeze identity / isolation, data matching or algorithm test failure. Repair and re-evaluation; It’s not the no-go mechanism that uses programming errors.
When the environment is unable to verify, mark the technical blockage without first claiming that the mechanism has failed. An abnormality in the source coverage can trigger an investigation, but 90% ± 2% is not the only one to perform the inspection, and >92% does not automatically err; The discrete scores and the whole group ties can make coverage conservative and time non-exchangeable.

## B. NO_GO_EVIDENCE
Achieving effective, but lacking any meaningful set changes or misunderstandings, or low risk, largely comes at the expense of almost all attack detection.
Complete report acceptance rate, correct record loss, Benign error report, Attack detection and abstention, scanning and time breakdown, not extended to Week 1.

## C. POTENTIALLY_PROMISING_REVIEW_REQUIRED
There is evidence to distinguish, along with acceptance rates, effective attack detection, good reporting and timing performance audits.
It does not automatically start Week 1 and does not automatically start GO. It is not possible to pass a separate judgment by refusing to capture the number of errors.

## D. INCONCLUSIVE
If the denominator is insufficient, the interval is too wide (if it is decided in advance to do the interval) or different evidence contradicts it. The descriptive results cannot support the stability judgement and cannot be passed by default.

## audit indicator ((fixed direction, minimum scope to be audited)
1. P_target(R) − P_source(R) and shows changes in multi-label/empty respectively.
2. P_target(raw_error|R)−P_target(raw_error|not R).
3. P_target(R|raw_error)−P_target(R|raw_correct).
4. acceptance_rate, TP/, all Attack, FN/, all Attack, Attack, abstention, Benign FPR, and correctly recorded rejection rate.
5. is the same as q under four performance and primary attack categories, with emphasis on scanning; Reported support quantity.

q<0.5 is more than labels with 0 structural; q>0.5 is structured as 0; At q=0.5, p=.5 gives a two-element set; all other probabilities give a singleton. It’s not a failure to code with structural zeros.
epsilon=.10 is not a coverage budget and is not equal to C2’s 5% selective risk; 5% risk /10% acceptance rate gate without inheriting C2.
It is not possible to change the epsilon, model, range based on known target results, and the empty set is not named Unknown Attack.
Reporting to the professor after failure; You can’t enter Week 1 until you get pilot confirmation. This time, we’re not sending messages.
''')
md('information_boundary.md','''# Information boundaries and historical use

## confirmed this time
- S_train=[0,1420000): detector fit and preprocessing are the only statistical sources. P1 limited to median / mode fill in the freeze value; RF is not standardized.
- S_prob_cal=[1420000,1780000): historical fit Platt and record Raw/Platt calibration indicator; This time only validate the 360000 clause identity, labels and Raw probability effective, and future for Variant A calibration. Historical Platt does not automatically make Raw calibration unusable, this time it is not called.
- S_competence=[1780000,2130000): Historical Platt Conditional Capability Table Source, not participating in this round of calibration or evaluation.
- S_source_test=[2130000,2365424): Has been evaluated by P2 and A/B1 and this round retains the full 235424 clause.

## Raw Model Training / Select the strength of the evidence at the boundary
configs/execution_v2.json is the same as a copy of freeze; freeze-record-v2.1 states that parameters is unchanged and originally executed freeze earlier than the basic model results.
src/nids/p2.py:train_source_model first calls train_model to save the model and then to reason on S_prob_cal/S_competence/S_source_test; Subsequently, the fit_platt was recorded with the indicator, with no feedback to the path chosen by the RF training or parameters.
RF of src/nids/models/base_models.py fixed parameters fit receives only S_train; src/nids/data/preprocessing.py just cuts S_train to fit statistics.
manifest training, P2 verification and reporting consistent with the code. The S_prob_cal/ selection / modeling record in synchronous docs/configs/src/ training manifest has been retrieved and no current Raw RF record has been found to model selection or adjustment using S_prob_cal.
This belongs to the existing record and code path audit, which cannot prove that the unpreserved person did not exist for experimentation. The executing person shall confirm prior to pilot the selection or modification of a Raw RF without synchronous material; If necessary, the isolation conclusions should be reviewed.
This time, the RF-All49 is fixed from roadmap and user requirements, not based on the new ToN result selection model.

## identity response
All partition sources use freeze sorted_original_rows in full array, identity as domain:original_row, with the original line zero base and no header; The rank interval is non-overlapping and the only identity, with calibration intersecting with the training / source test as 0.
360000 calibration labels and original CSV are verified by original_row mapping in this phase; Both Raw and labels use the same P1/P3 and availability rank indexes.
The P3 array does not have a sequential ID field: the alignment is based on a continuous rank entry that generates the code, the full range of shape, manifest and the original identity mapping, without pretending that ID exists in the cache, nor does it claim to rethink the entire probability.

## is unblinding
P7 and its comprehensive conclusions The / Professor’s report has revealed the results of target classification, misconstruction and coordination; A, B1, B2, C1, C2 are also diagnosed after working in these areas.
ToN in F1 is heldout, and development domain in other LODO foldings. C1/C2 fixed 400 has been used to select the target threshold; The Strict-0 wheel does not use these thresholds or labels calibration.
Prior to the full ToN recovery, the 100000 clause is the Strict-0 range prescribed in this round and cannot erase its historical labels exposure. This is not the first time that the target data has been accessed, nor is it a new independent external verification.

## Access Restricted
The read-only source calibration Raw checks for limitations and scope, without generating any real non-conformity scores or q. The ToN cache only checks the NPY head, size and SHA256 byte abstract, and does not decode the probability value; Only verify identity, time, window and number, without calculating new target sets or performance.
The model is only loaded, with no retraining, replacement, softmax or new reasoning caching. The full probability cache is not copied; There is no outflow trajectory.

## interchangeability and follow-up issues
The source data for partition time has also not been shown to be interchangeable; It’s not guaranteed with independent samples. Variant There is no formal guarantee that A targets across the data set will be covered; bootstrap (if chosen) is the conditional distance between the freeze model and the observed time sequence.
The existing 400 clauses: ToN 343/57, CIC 400/0, BoT 41/359 do not meet all 200 classes. This restriction does not impede Strict-0 Week 0, but subsequent Mondrian will require further sampling and time isolation.
Roadmap§3.2/3.4 requires200/class,while§4.6 describes BoT fallback below100; support100–199 leaves an interpretive conflict needing later clarification. No Mondrian here.
Multi-detector intersection does not automatically inherit individual coverage; The vacuum cluster is only validated to reject the signal and cannot be directly called Unknown Attack. Coordination rules, use of K target labels, and whether the previously used 400 labels support a subsequent conformal-calibration-only claim require separate review.
''')
md('commands.sh','''#!/usr/bin/env bash
# Working directory: /home/karri/projects/NIDSProject_Professor_Run
# Exact execution commands. GPU command was executed with approved sandbox escalation.
set -euo pipefail
TASK_OUT=/home/karri/projects/NIDSProject_Professor_Run/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z
TASK_PY=${NIDS_PYTHON:-python3}
# Initial sandbox attempt retained in model_load.log; successful GPU attempt in model_load_gpu.log.
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/load_model_check.py" > "$TASK_OUT/model_load.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/load_model_check.py" > "$TASK_OUT/model_load_gpu.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/run_preflight.py" > "$TASK_OUT/run.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/write_protocol.py" > "$TASK_OUT/protocol.log" 2>&1
# Do not blindly replay into this reviewed run. For a rerun copy scripts to a new run directory first.
''')
print('Draft protocol, metrics, gates and boundaries written')
