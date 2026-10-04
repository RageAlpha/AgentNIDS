# Two additional experiments after Outcome B: execution design

Status: design only; no new samples,thresholds or evaluation at this document date. Date:2026-09-20.

## 1. Authority and scope

Follow section4 and Reporting order of `C:/Users/ysy/Desktop/Reply_to_Siyuan_OutcomeB_BeginWriting.md`. Consulted designs,results/review records in the historical tasks Understanding Variant B Results (01a0ad01-887f-7802-998f-bac0ea5c3140) and NO_GO_EVIDENCE (01a0a53d-671a-71c2-8455-6107c55ef5cd),and checked local RF pilot/C1 frozen protocols.

This round designs only two experiments:

| Experiment | Model/target | Calibration | Purpose |
|---|---|---|---|
| 1 | MLP-All49→ToN | Same400 records already used by RF/XGBoost | Check whether the third architecture triggers the same threshold gate |
| 2 | RF-All49,XGBoost-All49,MLP-All49→BoT | Newly collected earliest200 Benign+200 Attack,shared by all models | Check second-domain thresholds and decide evaluation per model |

The latest reply explicitly adds these experiments,overriding former prohibitions on MLP/BoT and immediate writing after Outcome B for this work. It neither restarts full Week1 nor reruns RF/XGBoost ToN.

Shared fixed settings:frozen UNSW-NB15-trained models,All49,original preprocessing,Raw probabilities,ε=0.10,K=400/domain with200/class,original chronological order and complete10,000-record windows. Historical instructions exclude hash/SHA256 calculation/verification. Keep implementation steps without expanding calibration theory.

No temporal stratification,probability scaling,threshold search,rolling calibration,different K/ε,retraining,coordination,CIC or confidence diagnostics. Existing diagnostics are background,not additional tasks.

## 2. Input preparation

Execution project:`/home/karri/projects/NIDSProject`;all relative paths below refer to it. Local Windows copies check design without establishing WSL asset access.

| Asset | Location |
|---|---|
| Original RF pilot | `outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z/` |
| Original ToN400 manifest | Its `calibration_sample_manifest.csv` |
| Original RF protocol/implementation | Its `protocol.json`,`threshold_freeze.json`,`pilot.py` |
| C1 scope/segments | `outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/protocol.json` |
| Labels | `outputs/feature_store_v2/{ToN,BoT,UNSW}_label.npy` |
| Rank-to-original-row mapping | `outputs/round1_data_prep_v1/{ToN,BoT,UNSW}_sorted_original_rows.npy` |
| Windows | `outputs/round1_data_prep_v1/window_manifest.csv` |
| Models/probabilities | Locate exact All49 assets in existing `outputs/source_v2/models/` and `outputs/prediction_cache_v2/`;do not guess MLP/XGBoost names |

Check raw_probability=P(Attack),0=Benign,1=Attack. Do not use probability/predicted_class instead. Retain original precision/use existing float64 calculation,without normalization,rounding or Platt.

Check whether caches/labels are already availability-rank sorted,to avoid sorting twice. Record original-row indexing convention;sorted ranks are not original CSV rows. Probabilities must be finite within[0,1].

Index existing Raw caches directly. If absent,infer only the fixed400 calibration records using the same frozen model/preprocessing. Do not infer subsequent evaluation records or performance during threshold calculation. Report missing identity/preprocessing/alignment evidence without substitute assets.

## 3. Experiment1: MLP Variant B on ToN

### 3.1 Reuse the original calibration manifest

1. Read original RF400 manifest;check identity with XGBoost pilot samples.
2. Check200/class and uniqueness;retain original_row,rank,label,window_id,window_index and existing time fields.
3. Historical samples are earliest200+200 in startup,ranks0–4,019. **4,020 is scanned-prefix length;400 is actual calibration size.** Startup stays[0,100000).
4. Read MLP-All49 Raw probabilities at these fixed ranks,without resampling,recollection or temporal-stratified replacement. Recover missing/conflicting original artifacts rather than create substitutes.

### 3.2 Threshold calculation,freeze and first report

Apply latest reply section4 experiment1 Step1:sort200 true-class nonconformity scores/class and take the181st (zero-based180),without interpolated quantiles.

```text
p = MLP Raw P(Attack)
Benign score = p
Attack score = 1-p
q_Benign = sort(Benign scores)[180]
q_Attack = sort(Attack scores)[180]
```

Save400 probabilities/scores,exact thresholds and UTC freeze time. Submit this table before BoT collection or any full evaluation:

| Required value | MLP/ToN |
|---|---|
| q̂_Benign | Pending |
| q̂_Attack | Pending |
| q̂_Benign+q̂_Attack | Pending |
| Attack-inclusion probability lower bound:1−q̂_Attack | Pending |
| Benign-inclusion probability lower bound:1−q̂_Benign | Pending |

Include sample-reuse checks,gate status and full evaluation not yet run. RF/XGBoost values cannot stand in for MLP.

## 4. Experiment2: BoT Variant B for three models

### 4.1 New shared BoT calibration sample

1. Fix the first10 complete BoT windows,[0,100000),and scan true labels chronologically.
2. Retain earliest200 Benign/200 Attack,stopping when both complete;each record enters once.
3. Do not reuse ToN/old random C1 samples or choose by probabilities. All models share the same BoT400.
4. Save exact ranks,original rows,labels/windows;record each class’s200th position,stop rank,labels scanned and actual sample size.
5. Reply’s historical checks:first100,000 contain9,071 Benign/90,929 Attack;200th Benign rank14,863. Confirm actual positions;zero-based scan includes14,864 records. Stop at the larger class200th rank,not an expected value.
6. Investigate discrepancies. If either class lacks200 within startup,stop BoT calibration without larger windows,supplementary samples or reduced quotas.

### 4.2 Three-model thresholds and second report

Read RF-All49,XGBoost-All49,MLP-All49 Raw probabilities for shared400;each independently uses the181st-score rule. Save model-specific400 probabilities,scores/thresholds without averaging.

**Submit all three threshold tables before any full BoT evaluation.**

| Required value | RF/BoT | XGBoost/BoT | MLP/BoT |
|---|---|---|---|
| q̂_Benign | Pending | Pending | Pending |
| q̂_Attack | Pending | Pending | Pending |
| q̂_Benign+q̂_Attack | Pending | Pending | Pending |
| Attack-inclusion probability lower bound:1−q̂_Attack | Pending | Pending | Pending |
| Benign-inclusion probability lower bound:1−q̂_Benign | Pending | Pending | Pending |

Report sample/scanning counts,three independent gate statuses and eligible model list.

## 5. Gates and reporting order

Apply the same rule independently to MLP/ToN and each BoT model using unrounded thresholds:

| q̂_Attack | Action |
|---|---|
| ≥0.90 | Record Outcome B threshold stop;do not run full evaluation or auxiliary source test |
| <0.80 | After required threshold reporting,run full Variant B evaluation |
| 0.80≤q̂_Attack<0.90 | Undefined in authority;report pending professor decision;do not evaluate |

0.80 is undefined;0.90 stops. Do not assign the middle interval to Outcome B/evaluation. One stopped/pending model does not prevent other independently specified steps;one passing BoT model cannot admit all.

Fixed order:

1. Submit MLP/ToN five thresholds/gate status.
2. Collect BoT and submit all three five-value tables/individual gates.
3. Evaluate only eligible models. If MLP/ToN qualifies,finish/report it first;then eligible BoT RF,XGBoost,MLP in that order. This is implementation order,not another scientific criterion.
4. Report each completed evaluation immediately,rather than waiting for all.
5. After both experiments/all defined branches are completed/reported,await professor review before manuscript writing.

A saved local report does not mean professor receipt/review. Show tables to the user and deliver transferable reports,recording actual status without inventing messages/review. Explicit<0.80 branches require no new approval gate in the authority;undefined intervals require a decision. No email/messages without explicit user instructions.

## 6. Full evaluation only for eligible models

### 6.1 Frozen ranges and segments

Use zero-based,left-closed/right-open intervals. Exclude entire startup,not only400 samples/scanned prefix.

| Domain/scope | Rank interval | Complete windows |
|---|---|---:|
| ToN overall | [100000,27520000) | 2,742 |
| ToN segment1 | [100000,6960000) | 686 |
| ToN segment2 | [6960000,13820000) | 686 |
| ToN segment3 | [13820000,20670000) | 685 |
| ToN segment4 | [20670000,27520000) | 685 |
| BoT overall | [100000,16930000) | 1,683 |
| BoT segment1 | [100000,4310000) | 421 |
| BoT segment2 | [4310000,8520000) | 421 |
| BoT segment3 | [8520000,12730000) | 421 |
| BoT segment4 | [12730000,16930000) | 420 |

ToN evaluates27,420,000 and excludes260 tail records;BoT evaluates16,830,000 and excludes3,808. BoT boundaries use C1/preflight evaluation_scope,applying the reply’s four-segment structure without copying ToN numbers. Check actual windows beforehand;do not segment by performance.

### 6.2 Sets and baselines

Use each model/domain’s own frozen thresholds:

```text
Benign inclusion: p <= q_Benign
Attack inclusion: 1-p <= q_Attack
Benign only: accept,predict0
Attack only: accept,predict1
Empty set:reject;two-element set:abstain
Raw baseline:p >= 0.5 predicts1,otherwise0
```

Singleton labels may differ from Raw. Count errors using the singleton’s unique label,without a Raw-agreement condition. Inclusion lower bounds are not singleton-acceptance bounds. Generate sets from probabilities/thresholds only;truth is for scoring.

Each Raw baseline uses that model,target and same post-startup range. RF/ToN15.91% cannot replace MLP/BoT baselines.

### 6.3 Metrics

For each eligible model report overall/four segments,with all numerators/denominators and NA for zero denominators.

| Metric | Definition |
|---|---|
| Acceptance | Singletons/evaluation records |
| Accepted error | Wrong singletons/singletons |
| Raw all-accept error | Raw errors/same-range records |
| Error improvement,percentage points | 100×(Raw error−accepted error) |
| Four set counts/fractions | {Benign},{Attack},empty,two-element;denominator=evaluation size |
| Benign singleton error | True Attack/Benign singletons |
| Attack singleton error | True Benign/Attack singletons |
| Correctly accepted attacks TP/fraction | TP;TP/all true Attack in range |
| Segment TP contribution | Segment TP/overall TP |

The first four metrics/segment breakdown satisfy new BoT requirements;others retain RF pilot counts for MLP/ToN comparability. No new scan curves,ablations or risk-decomposition experiments.

If MLP/ToN qualifies,retain RF source-test auxiliary evaluation:UNSW[2130000,2365424),235,424 records including5,424 tail. Apply same ToN thresholds and report true-class inclusion coverage for Benign/Attack,including two-element sets. Do not refit source thresholds. BoT authority requires no source test;do not add one.

### 6.4 Retained statistics/checks

MLP/ToN retains RF protocol:10-window blocks,2,000 paired replicates,PCG64 seed20260915,275 blocks with2-window tail. Reuse RF sampling matrix if ranges match. Compute replicate Raw/accepted risks/difference from summed counts,95% percentiles/valid counts;segments are descriptive. BoT reports overall/four segments without additional unrequested bootstrap/significance tasks.

Check400 unique/200+200,calibration/evaluation separation,shared domain manifests,correct181st score,equality/unrounded gates,four sets sum=N,singleton confusion sums=accepts,segments sum=overall. Check asymmetry,0.80/0.90 boundaries,empty/two-element sets,zero acceptance.

When reusing pilot scripts,generate reporting from actual results;remove hardcoded historical No-go/only-segment4-improves statements. Edit new scripts only,preserving historical artifacts.

Reply threshold gates determine whether evaluation runs;<0.80 does not prove improvement. Report new outcome patterns truthfully without modifying thresholds,automatically declaring Outcome A,restarting Week1 or adding experiments.

## 7. Outputs and endpoint

Suggested separate execution directory:

`outputs/post_p7_diagnostics/outcome_b_two_experiments/run_<UTC_timestamp>/`

| Path/file | Contents |
|---|---|
| `protocol.json` | Authority,fixed rules,actual paths,range,report order |
| `ton_mlp/calibration_sample_manifest.csv` | Copy of original400 identities |
| `ton_mlp/calibration_scores.csv` | 400 MLP Raw probabilities/true-class scores |
| `ton_mlp/threshold_freeze.json`,`threshold_report.md` | First five-value report/freeze time |
| `bot/calibration_sample_manifest.csv`,`collection_summary.json` | New400 identities/scanning cost |
| `bot/{model}/calibration_scores.csv`,`threshold_freeze.json` | Separate probabilities/scores/thresholds for three models |
| `bot/threshold_report.md` | Complete three-model table before any BoT evaluation |
| `gate_decisions.csv` | Exact thresholds,branches,reasons/report status for four model/domain cases |
| `evaluation_metrics.csv`,`temporal_metrics.csv`,`evaluation_report.md` per eligible model | Overall/segments/same-model Raw comparison |
| `source_class_coverage.csv`,`bootstrap_intervals.csv`,sampling matrix if MLP/ToN eligible | Original full RF auxiliary protocol |
| `validation.json`,scripts,commands/logs | Required calculation/protocol checks |
| `completion_report.md` | Completed/stopped/pending branches and review items |

Do not duplicate probability caches or generate large per-flow trajectories. Gated models are marked full evaluation not run,without guessed risk/accuracy. If all four thresholds≥0.90,both experiments can finish at thresholds without full evaluation;undefined branches remain pending.

## 8. Traceability to latest reply

| Reply requirement | Implementation here |
|---|---|
| Reuse ToN400 for MLP,no retraining/Platt | §2,§3.1 |
| 181st score/class,five values reported first | §3.2,§4.2 |
| MLP≥0.90 stops;<0.80 uses full RF range | §5,§6 |
| BoT first10 full windows,earliest200+200,exact indices | §4.1 |
| BoT shared samples,independent thresholds | §4.2 |
| Per-model BoT gates;full acceptance/error/Raw/four segments | §5,§6.1–6.3 |
| MLP thresholds,all BoT thresholds,then eligible evaluation | §5 |
| Report each completed evaluation immediately | §5 |
| Writing only after both experiments/review | §5,§7 |

Sources checked:latest reply;local `C:/Users/ysy/Desktop/agent-nids/variant_b_ton_pilot/run_20260916T085242Z/protocol.json` and original samples;`C:/Users/ysy/Desktop/agent-nids/strictK/run_20260915_C1_01/protocol.json`;`C:/Users/ysy/Desktop/agent-nids/strictK/run_20260914_Cpreflight_01/preflight/evaluation_scope.csv`;historical ToN checklist. Existing records were read,without rerunning their validation.
