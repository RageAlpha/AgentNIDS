import json,csv,hashlib,re,platform,sys
from pathlib import Path
from datetime import datetime,timezone
O=Path(__file__).resolve().parent; R=Path('/home/karri/projects/NIDSProject_Professor_Run')
def save(n,v): (O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def csvwrite(n,rs):
 with (O/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
audit=list(csv.DictReader((O/'source_hashes.csv').open()))
extras=['outputs/feature_store_v2/stage_report.md','outputs/source_v2/P2/stage_report.md','outputs/round1_v2/P0_P1_stage_results_v2/P1_preprocessing_and_jsd_manifest.json','outputs/round1_v2/P0_P1_stage_results_v2/P1_artifact_validation.json','outputs/round1_v2/P0_P1_stage_results_v2/P1_feature_layer_report.md','datasets/NF-UNSW-NB15-v3.csv','scripts/run_p2.sh']
for name in extras:
 p=R/name
 if p.exists():
  version='unversioned; exact SHA256 snapshot'
  if p.suffix in ['.md','.json','.sh']:
   t=p.read_text();version=' | '.join(t.splitlines()[:4])[:500]
  audit.append(dict(path=str(p),version=version,sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,role='existing_project'))
csvwrite('source_hashes.csv',audit)
# Source history search corpus and hits, with no claim about unsaved human decisions.
hits=[]; searched=[]
for base in ['docs','configs','src','outputs/source_v2/P2','outputs/source_v2/models','logs']:
 for p in (R/base).rglob('*'):
  if p.is_file() and p.suffix in ['.md','.json','.py','.log','.txt']:
   searched.append(str(p)); t=p.read_text(errors='replace')
   for i,line in enumerate(t.splitlines(),1):
    if re.search(r'S_prob_cal|hyperparameter| model selects | to modify |tuning',line,re.I): hits.append(dict(path=str(p),line=i,text=line[:1500]))
save('history_selection_search.json',dict(corpus=searched,hits=hits,interpretation='P2 saved RF before source calibration use. No recorded source-calibration-driven RF selection found in inspected material; undocumented trials cannot be excluded.'))
# Recheck all protocol and asset hashes at end, not just modification timestamps.
for r in audit:
 if r['sha256']: assert sha(Path(r['path']))==r['sha256'],r['path']
checks=json.loads((O/'validation_checks.json').read_text()); assert checks['status']=='PASS'
p=json.loads((O/'pilot_protocol_draft.json').read_text())
p['algorithm']['input_contract']='Only RF-native float32 probabilities losslessly stored in float64 are admitted; validate exact float32 roundtrip without modifying probabilities. Nonconforming values stop execution. Entire source calibration verified. Future pilot validates target chunks before scores/sets.'
p['provenance']['new_proposed_execution_choices'].append('Explicit float32-origin input validation to bound binary64 complement arithmetic; review required')
p['open_protocol_choices'].append('approve explicit RF float32-origin validation; incompatible probabilities halt without rounding')
save('pilot_protocol_draft.json',p)
(O/'numerical_contract.md').write_text('''# numerical agreement with first test findings

This implementation only serves freeze cuML RF: the float32 class probability was originally inferred, and the P3 was originally converted to float64. The calibration Raw value of the entire 360000 line verifies that the float32 return is completely equal; There’s no data left. The target value was not read, and pilot verified the contract block by block, failing to stop and not automatically convert.

For the first time, extend the test to an arbitrary binary64 input, and at p=nextafter(0.5,0) the 1-p of float64 is transferred to 0.5; A direct comparison under q=0.5 gives a binary element. This demonstrates that the assumption of an arbitrary binary64 sub-mathematical structure is not unconditionally the same as the inference algorithm. Failure logs retain run_attempt1.log.

Modification is not a probability interruption or override: scores and prediction_sets input verification p is equal to its float32 lossless recovery value; It’s not enough to be satisfied. The critical test uses float32 values adjacent to 0.5 on both sides; Another test for the antigen binary64 was explicitly rejected. The / probability accuracy needs to be reviewed by a new algorithm, which cannot be quietly followed.

For fixed RF inputs, the float32 gap near 0.5 is much larger than the float64 calculation gap, and the test covers q<0.5, =0.5, and >0.5 structures. q itself is still stored in the float64 sequence statistic, not quantization to float32. All comparisons contain equalizations, and candidate Attack is calculated directly from 1.0-p<=q, without replacing the inequality with different pathways.
The main epsilon is calculated using the integers k=((n+1)*9+9)//10; n=360000 is k=324001, 324000 is zero. Only k has been calculated, the real score and q have not been calculated. The limit threshold float.hex/fromhex is accurately restored and the limit threshold is kind=positive_infinity.
''')
(O/'preflight_report.md').write_text('''# Week 0 Feasibility Pilot Pre-set inspection

** Conclusion: freeze RF-All49 + ToN + Variant A / Strict-0 technical preconditions have been verified; The draft agreement is still pending review and has not yet been authorized to implement pilot.**
The real q, ToN prediction set and pilot performance have not been calculated. After completion, stop and wait for the user to review and determine the execution protocol.

This directory: `'''+str(O)+'''`.
Version: week0-feasibility-preflight-draft-v1.0; The actual source path, version / time and SHA256 see source_hashes.csv; See validation.json for machine checks. All new materials are in this directory only, and do not cover A, C2, original configuration, model, data, or freeze results.

## 1, source verification and source level
- has been found and read the docs subroutine Research_Roadmap_Phase2_Conformal_Coordination.md (September 2026 , Approved direction final version); The original text is unchanged.
- has read freeze execution_v2 configuration, config_snapshot and freeze-record-v2.1, execution-plan-v2.2-p7, source split/window/tail/fold manifest, P1/P2/P3 training /, manifest prediction and report.
- has read A asset_and_scope.md/model_asset_audit.csv, B1/B2 report and / range definition, Strict-K preflight, C1/C2 report and protocol.
- integrated backdrop uses Project_Core_Conclusion_After_P7.md, Professor_Progress_Report_After_P7.md and Followup. Synchronous independent synchronous A synchronous C2 synthesis synthesis document: this is a source material gap, without summarizing the P7 after an old synthesis called C2; It is still necessary to confirm that the existing material is sufficient or that it has been synchronized.
- is a single roadmap column in source_hashes.csv, the others being existing projects; pilot_protocol_draft.json’s provenance also distinguishes between the refinement of user requirements and this new recommendation. The old protocol excluded SHA/smoke test, which was explicitly requested by the user to be hashed and synthesized, without modifying the old protocol.
- `.git` is currently unreadable as a Git warehouse, not a fictional version of commit; The exact version is in the current SHA256 file, and there is an explicit version and a mtime record.

## 2 has been approved.
### model and probability
The original `outputs/source_v2/models/rf_All49/model.pkl` exists, 5,684,186 bytes, training version base-model-v2.0, model ID rf_All49, cuML RandomForestClassifier.
The original agentnids environment successfully loaded: Python3.11.16/cuML26.8.0/CuPy14.2.0/NumPy2.4.6 , CUDA runtime12090 , driver13020; classes_=[0,1],n_features_in_=49.
The sandbox CUDA fails to access the first time and retains the model_load.log; model_load_gpu.log was successful after accessing GPU. No replacement or retraining model.
49 features name, order, indices corresponds to training, P1, preprocessing, P3, see feature_contract.json. RF input is float32; First, the non-finite values and freeze fill values are processed with float64, then the float32 is converted, not standardized. The filler value is from S_train only; The numerical value finite is median, the category mode is the minimum value broken by ties, the total absence of 0.
RF predict_proba[:,1] is directly Attack probability, not additional softmax. P3 raw_probability without Platt; probability is the probability after Platt; predicted_class is the historical Platt>=0.5 result. pilot uses only raw_probability and does not call f(u) or C1/C2 thresholds.

### source partition (actual value of freeze, not weighted in proportion)
| partition | rank Semi-open range | N | Benign | Attack |
|---|---|---:|---:|---:|
| S_train | [0,1420000) | 1420000 | 1377436 | 42564 |
| S_prob_cal | [1420000,1780000) | 360000 | 325600 | 34400 |
| S_competence | [1780000,2130000) | 350000 | 319288 | 30712 |
| S_source_test | [2130000,2365424) | 235424 | 215407 | 20017 |

Source original_row mapping is the only complete sequence and follows the end-time / initial stable sequence, with calibration and training, source testing identity intersecting 0. All 360000 calibration labels are verified in phases by CSV mapping; The Raw probability is limited and at [0,1], the float32 is elevated to float64 with no loss.
The Raw cache already has full source domain data, no additional calculations needed. The original cache does not have an embedded ID, Raw/ labels alignment based on the same rank generating contract with the entire manifest domain, without claiming to verify all probabilities in this rethink.
The training code is fit and preserves RF and then makes the source calibration reasoning, Platt fit and indicator records. No S_prob_cal has been found to be used to adjust the current Raw RF record. Evidence and historical limitations can be found in information_boundary.md and history_selection_search.json.

### scope and time
The source tests the complete [2130000,2365424) with the end of the 5424 remaining; ToN for P7 complete [0,27520000), 27520000 items, 2752 original 10k windows, and the tail 260 items continue to be excluded.
freeze manifest, P7 data_audit does not conflict with the expected; The original heldout split contains the 260 clause and the window evaluation excludes tail. The two levels are different and are not in conflict.
ToN all 27520000 lines P7 rank/original_row/record_id/availability_ms/window_id have been verified; The complete element mapping and time stability sequence are passed. Windows not reconstructed; Keep initialized windows, without inheriting the C1/C2 prior to 100000 exclusion.
Four 688/688/688/688 windows, all using the same q source. The complete ToN cannot be directly used with the C1 2742 window bootstrap matrix.

### algorithms and synthetic testing
a limited sample k using an accurate rational number; n=360000; epsilon=1/10 at k=324001 (one group); 324000 array index. No real calibration or q score has yet been calculated.
Sequence the k instead of the linear insert; k>n is +infinity. Contains an equal number, a whole group ties, no random breakdown, no new intersections; q is stored with hex or a clearly defined infinite marking.
The synthetic test covers three structures: p=0/.5/1, p=q and p=1-q, q, repeated large ties groups, k=1/n/>n, sequencing, four set counts, labels, replacement of non-affecting sets, and quantization as defined by the sample.
The first test exposed an arbitrary binary64 input to the additive input problem, ultimately achieving an explicitly limited freeze RF float32 source probability and verifying the non-silent input. Specific counter-cases, grounds for revision and scope see numerical_contract.md; In the end, implementation_check.json is PASS.
q<.5 times labels is structural for 0 and q>.5 times 0; When q=.5 is p=.5 binary element, the remaining single element (within the explicit RF input contract)These zeros do not constitute failure.

## 3 still needs to be verified.
Re-verify this protocol and the necessary model / / identity hash before - pilot is officially launched; If environmental changes are implemented, the nuclear GPU is loaded.
The current - code / log does not show calibration driven Raw selection, but those who did not save it were unable to prove its non-existence by attempting to document it. The person responsible must confirm that the S_prob_cal selection model / is modified without any other material.
- did not read the ToN probability value this time, so its limited / value domain /float32 source test was not renewed; pilot is verified block by block when it is read, fails to stop and is not automatically corrected.
- real q calculation and preservation restored checks, formal evaluation of the polymer preservation checks left to the audited pilot, cannot be passed with this synthesized PASS to pretend that pilot has been passed.

## 4, pilot protocol selection required to be determined prior to
1. audits complete source testing of /P7 ToN main range, P3 raw float64 and obvious float32 source contracts.
2. The initial recommendation for pilot is that all results are descriptive and not generate CI. Reason: Source testing only 23 complete windows +5424 tail, only 3 blocks per 10 window block, not meeting the project at least 20 blocks; tail cannot be deleted for statistics. If a ToN range is required, a replicable scheme must be included in the draft approved prior to implementation: 2000 blocks, 10 window blocks, 276 endings, 2 windows, PCG64 seeds, 20260915, 95% percentile), different indicator-sharing matrices; It does not generate source CI or pretend to be a target source for CI.
3. meaningfully more/errors concentrated has no original numerical threshold and must be determined beforehand using evidence-based joint audits or otherwise substantiated minimum effects. This time it’s not a number, it’s not an automatic inheritance of C2 5%/10%.
4. The independent A–C2 integrated summary is missing; confirm adequate sources. Original stage reports/P7 summary currently disclose background.

##-5 does not block Week 0 but affects subsequent factors
The existing 400 clauses ToN343/57, CIC400/0 and BoT41/359 do not meet the Mondrian conditions of roadmap for each 200 class. The total 400 cannot be considered satisfactory; The current Strict-0 does not use these targeted calibration samples.
Subsequent review of the Mondrian sampling / time drift, exposure to both the 400 clause used for C1/C2, the roadmap BoT100 and the general 200 threshold formulations, and the issue of non-automatic inheritance of multiple detector intersections.

## 6, indicators, judging and stopping
All indicators and zero factors are defined in metric_definitions.md/pilot_protocol_draft.json. prediction_set_coverage and acceptance_rate respectively; At the same time, record the Raw error refusal to distinguish, correctly record the loss, accept the confusion matrix, both the class and the original attack class (scanning) and time stability.
gate_definition.md contains IMPLEMENTATION_INVALID, NO_GO_EVIDENCE, POTENTIALLY_PROMISING_REVIEW_REQUIRED, and INCONCLUSIVE; This time, the mechanism does not give gate results.
The source coverage of 90% ± 2% is not the only verification, and >92% does not automatically error. epsilon=.10 is not C2 selective risk; The vacuum is not directly named Unknown Attack.
P7/A/B1/B2/C1/C2 is already unblinding; This is an exploratory follow-up, not a independent external verification. Time sequence interchangeability has not been proven and is not guaranteed to cover independent samples.
** real q, ToN prediction set, pilot performance is not calculated; No complete pilot, Week1 or any other new experiment has been performed. Wait for the Week0 to be executed until the user audit determines the pilot protocol; The pilot failure is reported to the professor before pilot confirmation does not enter Week1. This is the first time that a widower has sent a message.**
''')
# Preserve original preflight inventory and ensure the final snapshot has not changed.
before=json.loads((O/'upstream_inventory_before.json').read_text())
changed=[]
for rel,st in before.items():
 pth=R/rel
 if not pth.exists() or [pth.stat().st_size,pth.stat().st_mtime_ns]!=st:changed.append(rel)
assert not changed
val=dict(version='week0-preflight-validation-v1.0',status='PASS_WITH_REVIEW_ITEMS',technical_conditions='READY_FOR_REVIEWED_PILOT',pilot_authorized=False,checks=checks,source_hash_count=len(audit),source_hashes_rechecked=True,upstream_unchanged=True,missing_materials=['Separate A-C2 integrated summary not found'],open_protocol_items=p['open_protocol_choices'],real_q_computed=False,ToN_prediction_sets_computed=False,pilot_performance_computed=False,Week1_started=False,report_path=str(O/'preflight_report.md'))
save('validation.json',val)
with (O/'commands.sh').open('a') as f:f.write('PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/finalize.py" > "$TASK_OUT/finalize.log" 2>&1\n')
art=[]
for f in sorted(O.iterdir()):
 if f.is_file() and f.name not in ['deliverable_hashes.csv','finalize.log']:art.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
csvwrite('deliverable_hashes.csv',art)
print(json.dumps(val,ensure_ascii=False,indent=2))
