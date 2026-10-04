# P7 implementation specification v1.0

Date: 2026-09-08. Configuration: `configs/p7_v1.json`. This implementation record follows inspection of P4/P5/P6 development results and the Item3 support audit; it is not retrospective preregistration. The request covered overall-plan status revisions, preservation of the previous version, a change explanation, and the P7 specification and code. Formal outer evaluation was outside that implementation request.

## 1. Research scope and frozen inputs

P7 evaluates the original Round1/2 methods on domains excluded from fitting and selection in the corresponding fold. It fits no new models, scans no thresholds, and does not reselect the initial detector, cascade order, or budget mixture.

| Component | Input basis | Implementation |
|---|---|---|
| Round1 primary condition | `configs/execution_v2.json`, its P4 copy, and per-fold P4 CRE models | 10k JSD linear CRE; prior/raw_h/u/hu/hus/1-u for three detectors; six base detectors and two G0 groups |
| Round2 | `outputs/round2_v2/P6/round2_frozen.json`, fitted F1/F2 models, and frozen_policies | All 96 operating points, all frozen budget entries, and fixed-cascade selections |
| F1 | Development CIC/BoT; outer ToN | Round1 + frozen Round2 |
| F2 | Development CIC/ToN; outer BoT | Round1 + frozen Round2; abstain as specified when no final threshold exists |
| F3 | Development BoT/ToN; outer CIC | Round1; record NOT_ELIGIBLE for Round2, with no new fallback |
| P5 | Development sensitivity analyses completed | Preserve existing results; this P7 version does not add outer sensitivity analyses |
| Item3/P8 | item3-scenarios-v1.0 | Do not execute or refit; do not claim Unknown isolation using old models |

Disclose all three folds. Round2 covers only F1/F2 and cannot establish coordination effectiveness across all three domains. The post-P7 summary may guide resource allocation; method changes based on it require a separate exploratory version. Existing Item3 descriptive-outer conclusions and cross-fold exposure boundaries remain applicable.

## 2. Data and causal identities

Use existing `heldout/T_heldout` roles and complete 10k windows. Retain initialization windows, exclude tails, and disclose excluded counts. These are the original Round1/2 window rules; Item3 filtering and new tail rules do not apply.

`evaluation/outer_data.py` verifies the original window_manifest, rank continuity, original row indices, availability times, completed predecessors within the same stream, and cached JSD for each window. The first outer window has shift=0; subsequent shift equals JSD from the previous completed window of that outer stream, not a development-role predecessor. Associate each decision with record_id, rank, window_id, previous_window_id, and shift using its fold-specific identities.parquet position.

Prediction objects expose neither a labels method nor label_path; only evaluation functions read labels. Existing all-domain base caches from source models may be reused. Frozen fold-specific P4 CRE models generate outer CRE scores for the first time. Precomputing caches does not grant a policy access to all evidence at query time; actual querying continues to use VisibleEvidence.

## 3. Two-phase execution

1. Read-only verification of P4/P6 frozen configurations, model provenance, the F3 branch, 96 policies, budget files, and input metadata; verify current execution_v2 matches both stage copies.
2. Save identities and Round1 scores for all three folds. For F1/F2, save every frozen Round2 decision and random choices and selected-path decisions for all feasible budget mixtures. Prediction does not read outer labels.
3. Verify unchanged input sizes/mtime and write `predictions_complete.json`. Only after predictions for all three folds finish may status become EVALUATING and outer labels be read for the first time.
4. Evaluate each fold, save paired resampling and metrics, aggregate Round1 with equal domain weighting, and report per-fold Round2 results and budget comparisons.
5. Verify inputs again and write the P7 manifest and report.

The P7 output directory must not already exist, preventing overwrites and undocumented reruns. On failure, preserve stage status, existing artifacts, and failure.json; do not restart automatically. Reraise the original error after handling it, without automatic device fallback. A failure during evaluation means available outer results are exposed. Register the impact and recomputation scope of any repair; do not delete the directory to simulate a first run.

## 4. Executing frozen policies

Reuse conditional querying, routing, fusion, and Post-CRE in `coordination/replay.py`. The new record_sink only writes chunked results to compact decision files; original P6 calls without that argument retain their behavior. Saving all operating points does not change the information available to any individual policy.

Use each point's original tau_stop, tau_final, and cap. Early stopping uses inclusive `>=`; a null tau_stop disables early stopping, and a null tau_final makes all remaining terminal records abstain. F2 coordinator thresholds are all null: preserve actual query costs and full abstention without skipping the branch or inventing thresholds. Hard MoE may reuse existing expert outputs in terminal slots; count only detectors actually queried.

The compact ledger stores stage, accepted, reason, probability, reliability, calls, cost, three-step query identities, and actual queried probabilities. Reason codes remain 0 for early acceptance, 1 for final acceptance, 2 for below-final-threshold, and 3 for no feasible final threshold. Unexecuted queries use -1 and zero probability. Verify per record that no queries follow stopping and calls equals the number of actual queries.

Report all fixed permutations separately. Also report the development-selected permutation from `fixed_cascade_selection.json` separately for each cap. Do not use outer performance to choose another best permutation.

## 5. Costs and budgets

Load the original P6 cost_model, preserving measured detector, Pre-CRE, routing, fusion/Post-CRE component costs and the normalization reference. Add amortized existing P3 JSD time for each outer domain using the P6 formula. Do not retrain, remeasure, or adjust the budget reference using outer data.

These are cost estimates from measured components, not newly measured end-to-end deployment latency. Cache-reading speed is not detector execution latency.

All budget_matched entries retain cap_a, cap_b, lambda_a, and development budgets. Random selection uses the P6 formula `seed + int(calls_budget*4)*10 + domain_index`, with one full random vector per domain and shared seeds across methods; count only the selected path's calls/cost. Ordinary routing retains the P6 seed and chunk size 262144.

Report frozen budgets, outer expectations, and realized random quantities; mark whether outer calls/cost exceed the budget. Development feasibility does not ensure outer feasibility. Do not reselect mixtures to complete comparisons. Preserve NO_FEASIBLE_MATCHED_OPERATING_POINT for infeasible entries, without fabricated random policies.

## 6. Metrics and statistics

Round1 reuses the original mathematical implementations: binary detector metrics, complementarity, CRE correctness AUROC/AUPRC, AURC, Brier/NLL/ECE, and risk-coverage. Use 1-u only for ranking. Report overall, Benign, and Attack results; provide per-fold and equal-three-domain G0/G1 aggregates and paired per-model AURC differences and intervals.

Form blocks of 10 consecutive complete windows per domain, retaining the final short block. Run 2000 bootstrap replicates with independent sampling by domain and shared within-domain multiplicity across Round1/2 methods. Preserve original shifts and scores. Recompute the best single model for G0 in every replicate. Weight three domains equally for Round1 and additionally weight the three main detectors equally for G1. Apply original gates without revisiting P6 eligibility.

Round2 reports coverage, accepted error mass, selective risk, calls, component cost, Benign/Attack-stratified coverage/risk/abstention, and a 2×3 truth–action matrix. Report terminal reliability-ranking/probability metrics, actual G2 consultations, correction hits, correction opportunities, and consultation harm. The oracle reads caches and labels only after decisions are frozen.

Save block statistics and paired bootstrap results for fixed Round2 operating points and budget mixtures. At zero acceptance, point risk=null. If risk resamples include zero acceptance, set CI=null and disclose their count; do not assign zero risk or remove those resamples to obtain a conditional interval. Insufficient blocks do not provide estimable interval evidence.

At the same frozen budget, report coordinator-minus-other-feasible-method difference intervals. These are paired, pointwise descriptive intervals, not newly introduced multiplicity-adjusted superiority tests. Check both methods' realized outer budgets before interpreting risk–coverage–cost tradeoffs. Lower risk with lower coverage does not automatically establish a winner. If no feasible pair exists, report no comparative conclusion.

## 7. Outputs and storage

```text
outputs/outer_v2/P7/
  p7_rules.json, execution_v2.json, upstream_inventory.json
  status.json, predictions_complete.json, P7_manifest.json, stage_report.md
  round1_aggregate.json, outer_G0.npy, outer_AURC.npy, outer_G1.npy
  F*/identities.parquet, data_audit.json
  F*/round1/scores/*.npy, metrics.json, round1_aggregate.json, bootstrap/*
  F1/round2/ and F2/round2/
    frozen_policies.json, fixed_cascade_selection.json, budget_matched.json
    prediction_manifest.json, cost_model_applied.json
    policies/*/cap*.npy, cap*.bootstrap.npy, cap*.blocks.npy
    mixture_*.npy, mixture_*_choose_a.npy, mixture_*.bootstrap.npy, mixture_*.blocks.npy
    policy_metrics.json, selected_fixed_cascade.json
    budget_metrics.json, paired_budget_comparisons.json
  F3/round2_branch.json
```

Per-record ledgers for all operating points and random mixtures require substantial storage. Before execution, estimate usage from record counts, operating points, and feasible mixtures, requiring an additional 10% margin. Stop if space is insufficient; do not delete earlier artifacts. Actual size depends on feasible mixtures; do not report estimates as measured execution usage.

## 8. Entry points and validation

Read-only frozen-artifact verification, without outer predictions or outer-label reads:

```bash
PYTHONPATH=src /home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage preflight-p7
```

Code checks using only synthetic/exposed development data and outer identity metadata:

```bash
/home/karri/miniconda3/envs/agentnids/bin/python tests/check_p7_protocol.py
```

Formal outer-evaluation command, not executed during this implementation request:

```bash
bash scripts/run_p7.sh
```

See the [P7 implementation check](<../../NIDSProject_Professor_History_20261004/docs/P7_Implementation_Check.md>). Passing checks establishes contract compliance on inspected paths; it does not establish unseen-domain effectiveness, full-scale execution time, or sufficient storage.
