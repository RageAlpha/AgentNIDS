# ToN label-correction pilot: frozen E1 versus probability relabeling and local-only rules

Date2026-09-29. Status: execution design; laboratory data have not been run locally. Based on user-provided calibration results, without claiming independent reconstruction. The previous experiment is saved separately but C is reused, so this is not independent statistical validation.

## 1. Decision for this round

Proceed to the next ToN pilot, not full evaluation. E1 already produced13 correct attacks on C and both candidate branches are feasible. Freeze E1 now; do not continue changing distance, k, class-decision thresholds or strong-consistency gates on C.

E1 obtains final labels from local majority votes, while RF supplies acceptance features. It is local classification/selection supported by a target-label reference library, extending beyond pure reliability gating of original RF predictions. Record this scope change without attributing success beforehand to RF or CP.

## 2. Inputs and new output location

Laboratory project root: `/home/karri/projects/NIDSProject`.

V1 run：`outputs/post_p7_diagnostics/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr/`。

Action-correction calibration run: `outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/`.

Read its protocol, sample roles, original T grid, per-C records and selected_rules.json. Read corresponding400 RF probabilities/labels from existing assets only if actual fields are missing. Retain200R/200C, source preprocessing,48-dimensional mixed distance, equal-weight k=15 and neighbor tie rules.

Write new code/results under `outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_<UTC>/`. All old runs are read-only. Do not overwrite, resample, retrain, hash, checksum or inventory the whole project.

Reuse V1 saved ToN pilot ranks:274200 records,100 in each original10k window. Segment boundaries remain[100000,6960000,13820000,20670000,27520000], with68600,68600,68500,68500 samples. These labels have already been analyzed; this is repeated evaluation during continued development, not a new blind test.

## 3. Freeze E1 without another search

Let p be original RF Attack probability, s=max(p,1-p), and q the true-Attack proportion among15 neighbors.

E1 label h_L=1[q>=0.5]. With k=15, q cannot equal0.5.

Fixed final actions:

- h_L=A: accept all as A. Original selected A rule HIGH_T000_U000 (s>=0.5,u=0) always holds.
- h_L=B: accept B only when s<=0.8577222228050232 and q=0; otherwise abstain. This is LOW_T009_U003. Reuse the exact saved threshold and comparison precision.

q=0 means all15 neighbors are B, without guaranteeing correctness. C counts—B accepted52/errors2 and A accepted13/errors0—are implementation checks only.

If saved pilot evidence includes V1 local_support, reconstruct q using original RF predictions: q=local_support for A, q=1-local_support for B. Do not directly treat old support as support for corrected labels. If genuinely absent, recompute the same neighborhood only at existing ToN pilot indices; do not change k or distance.

## 4. Method matrix

| ID | Label/acceptance rule | Label budget and use |
|---|---|---|
| RAW | Original RF p>=0.5; accept all | Historical classification baseline |
| OLD_G2 | Original V1 fixed G2 labels/actions | Historical gating reference |
| E1_FIXED | Frozen correction/gating above | R200 library; C200 already selected rules |
| LOCAL_ALL | h_L; accept all | R200 |
| LOCAL_GATE | h_L; gate using support for final label only | R200; C200 selection |
| P_TAIL_C | RF probability only; normal/reversed relabeling and abstention | C200 selection |
| P_TAIL_400 | Same family as preceding row | All400 selection; label-efficiency control |
| P1D_GATE | Labeled neighbors in RF-probability space produce labels, then gating | Same R200/C200 |

P_TAIL_C, P_TAIL_400 and P1D_GATE are predefined primary controls; do not select only the weakest after execution. LOCAL_GATE checks whether RF acceptance features are necessary. P1D_GATE adds a cheap baseline without more labels, model training or feature-space tuning.

E2/E3 do not enter this new search. Old C results are historical context; knowledge that E1 performed better must not change any method’s thresholds.

## 5. Exact probability-relabeling baseline definitions

### 5.1 P_TAIL_C and P_TAIL_400

On the allowed selection set, consider NORMAL:r=p and REVERSE:r=1-p separately.

Action rules:

    r <= b: accept B
    r >= a: accept A
    b < r < a: abstain

Require b<a with no overlapping actions. b=-1 disables B; a=2 disables A. For each direction, enumerate all b<a from distinct observed selection-set r values plus-1 and2. Do not derive grid quantiles from pilot probabilities. Boundaries include equality and tied values remain together.

This family permits reversal of probability direction, class-boundary selection and rejection intervals, directly testing an unsuitable original0.5 boundary as an alternative explanation. It neither retrains RF nor covers every possible probability function.

Each nonempty enabled branch requires at least10 selection accepts and empirical action risk<=5%. OFF is allowed but does not satisfy that class’s target. Maximize U_bal=.5*(TN/N_B+TP/N_A), then min(UB,UA), then minimize FP+FN, prefer NORMAL, then ascending sorted b-index and a-index. b=2 or a=-1 cannot yield their respective meaningful actions in valid b<a pairs, but implement the full definition without manually supplying labels.

Use OFF when no enabled branch is feasible; both OFF gives NO_FEASIBLE_POLICY. Partial closure is allowed with PARTIAL_ACTION_POLICY recorded.

Prefix/suffix cumulative counts can evaluate candidates without repeated record scans. Save valid candidate counts by direction, selected boundaries, selection counts and a traceable candidate table. Select C and400 versions separately; never mix grids or labels.

### 5.2 P1D_GATE

Use the same R200 as E1. Distance is scalar abs(p_x-p_i), with equal-weight k=15 and ties ordered by reference availability_rank. q_1D is neighbor Attack proportion; label h_1D=1[q_1D>=0.5].

Reuse original G2 candidates: HIGH/LOW of original s, original12-point T, support q_1D or1-q_1D for h_1D, u∈{0,.5,.75,1}, and OFF. Select on C with unchanged empirical risk, minimum counts, objective and ties. The main distinction from E1 is neighborhood information: one RF-output dimension versus48 original features.

This baseline addresses part of the explanation that only a nonmonotonic probability/target-label relationship is needed. It cannot exclude every probability-only model. Do not add probability MLPs, trees or hyperparameter searches.

## 6. LOCAL_GATE: check RF’s incremental contribution

Fix labels to h_L. For each final predicted class allow only local support>=u, u∈{0,.5,.75,1}, or OFF, without s.

Select on C using the same risk<=5%, accepts>=10 and utility criterion. Prefer smaller u, then stable ID on ties. This family embeds in the earlier E1 candidates because HIGH s>=0.5 is unrestricted, making it a natural ablation of RF gating information.

LOCAL_ALL describes local classification and rejection cost. Comparing low accepted risk directly against full-acceptance risk cannot establish a fair winner.

## 7. Data access and execution order

1. Read this plan; create a new directory; load original R/C, frozen E1 and necessary400 RF probabilities; record input paths.
2. Select LOCAL_GATE, P_TAIL_C, P_TAIL_400 and P1D_GATE once on C or the declared400. Save rules, timestamps and calibration counts. Do not reselect E1.
3. Check action boundaries, OFF, reversal, risk denominators and ties with synthetic arrays. Confirm E1 on C gives52/2 and13/0. No hashes.
4. Load existing ToN pilot indices/evidence; recompute missing q only if needed and calculate new1D neighborhoods. Read/use corresponding pilot truth for scoring only after every action policy is frozen.
5. Evaluate all fixed methods, report, decide and stop. Infeasible baseline calibration does not skip E1 evaluation; unfavorable evaluation does not justify returning to C to change rules.

Existing evidence may contain labels alongside features. Reading non-label columns does not make the data previously unseen. Separate fitting/selection from evaluation functions; the former must not receive subsequent labels. Record prior researcher exposure to results.

## 8. Evaluation and criteria

Reuse original TN/FN/FP/TP, RB/RA, UB/UA and U_bal definitions. Report overall, four-segment and per-window counts/denominators for every method. A practical two-class operating point still requires RB/RA<=5%, UB/UA>=10%, each class accepting at least100 records across at least10 windows. Zero denominators are NA; report weak segment support separately.

Report two levels of E1 changes:

- Classification: LOCAL_ALL versus RAW B→A and A→B corrections/harm over the full pilot, including unaccepted records.
- Final actions: E1_FIXED risk/utility/abstention versus every control. An abstained attack is not a detected attack.

Add a predefined explanatory table for E1 predicted A: q groups8/15–11/15 and12/15–15/15, with sample/error/segment counts. Describe transfer of weak/strong neighbor majorities; do not use the table to reselect a0.75 gate.

Reuse paired blocks of10 consecutive10k windows, retaining the short final block;1000 replicates, PCG64(20261001),2.5/97.5 percentiles. Share replicates for E1, the three probability controls and LOCAL_GATE; compute ΔRB, ΔRA, ΔUB, ΔUA, ΔU_bal. Discard zero-denominator replicates, display valid rates and use NA below90%. No additional block-length/seed search.

Using the same existing resampling indices, also report descriptive overall/four-segment E1 RB/RA intervals. Utility-difference intervals cannot replace risk intervals. Do not call them certification or add interval upper bounds as new gates. E1’s13 error-free A accepts on C do not guarantee low target risk.

### Predefined outcome branches

| Outcome | Decision |
|---|---|
| E1 misses the two-class operating point | CALIBRATION_FEASIBILITY_DID_NOT_TRANSFER; stop full evaluation; distinguish risk failure, insufficient utility and inadequate support |
| E1 meets targets, none of the three probability controls does | LOCAL_CORRECTION_CANDIDATE; recommend full evaluation with unchanged rules, without excluding every probability method |
| E1 and one or more probability controls meet targets | LOCAL_INCREMENT_ESTABLISHED_AT_PILOT only if ΔU_bal>=1 percentage point and95% lower bound>0 against every qualifying control; otherwise superiority is unestablished and report all trade-offs |
| E1 and LOCAL_GATE meet targets but E1 misses the same utility-increment criterion | NO_ESTABLISHED_RF_GATE_GAIN; consider simpler local classification/selection without presenting it as RF-repair innovation |
| E1 meets targets while LOCAL_GATE does not, or E1 exceeds a qualifying LOCAL_GATE by that increment | RF_SELECTION_CANDIDATE; possible RF contribution in selection, not classification |
| A probability baseline meets targets but E1 does not | PROBABILITY_CORRECTION_PREFERRED; prefer simple probability adaptation |
| Overall targets met but<3/4 segments meet them | TEMPORALLY_FRAGILE; discuss feedback only after full evaluation |
| Good calibration, but new pilot errors offset corrections | Current local correction failed to transfer; do not retry this round with a higher/lower q threshold |

Do not infer equivalence from post-hoc approximate similarity. Missing the increment only leaves practical superiority unestablished. Preserve trade-offs when a method has lower risk and lower utility.

## 9. Review and stopping

This plan repairs two comparisons: probability controls can change labels, and local-only rules test whether RF still contributes. E1 remains frozen, candidate families are limited, labels/representations are unchanged, and parameters/outcome criteria align with the previous pilot.

Remaining limits: C has been repeatedly reused, pilot results are exposed, local classification may fit startup attack families,13 correct A records provide little evidence, and time intervals do not guarantee future behavior. This round informs continuation; it cannot establish independent confirmation, statistical certification or novelty.

The original first command authorized execution through this ToN pilot report without a further baseline-freeze approval. Genuine asset/semantic conflicts stop execution. BoT, full S3, feedback, other models, retuning k/distance/0.75, hashes and whole-project audits are outside that historical scope.

## 10. Delivery

`config/protocol.json`, `calibration/baseline_candidates.csv`, `calibration/frozen_baselines.json`, `tables/comparison.csv`, `tables/segment_metrics.csv`, `tables/window_counts.csv`, `tables/correction_harm.csv`, `tables/attack_support_groups.csv`, `tables/paired_intervals.csv`, `tables/E1_risk_intervals.csv`, `reports/summary_ZH.md`, `reports/decision.json`, `checks/basic_checks.json`, and actual code/commands/logs.

Adapt fields/paths through existing laboratory manifests. Lack of local access does not verify their current laboratory state.
