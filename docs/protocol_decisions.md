# Protocol registration v2

This file registers shared methods, data boundaries, and result-exposure status at P0. Actual registration time and runtime versions are recorded in `outputs/round1_v2/P0/runtime.json` and configuration snapshots.

## Data identities and roles

The implementation directly uses sorting indices, partitions, groups, windows, and predecessor relations from `outputs/round1_data_prep_v1`. The four CSVs remain unchanged. UNSW is the sole source domain; base-model preprocessing and the JSD source reference are fitted only on `S_train`. CIC, BoT, and ToN do not participate in these source fits.

The three outer folds are fixed as F1(CIC+BoT→ToN), F2(CIC+ToN→BoT), and F3(BoT+ToN→CIC). No outer-performance results are generated before P7.

## Registered shared methods

Training, calibration, CRE, windows, JSD, metrics, G0/G1, bootstrap, and Round 2 candidate rules use `configs/execution_v2.json`. The 49 input features and Traffic17, Protocol17, Behavior15 groups use `configs/feature_views.csv`. Open-set families use `configs/taxonomy.json`.

### Adoption of G0/G1 gates

At 2026-09-05T16:01:28Z, before any base-model, CRE, G0/G1, or outer-model results existed, G0=0.01 and G1=0.001 were formally adopted and frozen.

G0=0.01 requires at least 1 percentage point of recoverable error headroom as the minimum practical opportunity for cooperation. G1=0.001 requires a minimum absolute improvement on the 0–1 AURC scale, excluding numerical differences without practical reliability benefit. These are practical gates for this study, not universal statistical-significance thresholds.

### Fixed Round 2 rules and development-selected values

Round 2 candidate methods, eligibility conditions, initial-detector selection, corrective inputs/targets, fusion rules, Post-CRE fitting partitions, risk constraints, minimum coverage, candidate threshold grid, maximum calls, and latency-measurement procedure are fixed.

Per-fold initial detectors, `tau_stop`, and `tau_final` values have not yet been produced at this registration point. Fixed algorithms select them from the corresponding development partitions and write separate fold records at P6. Development data determine only these prescribed selections; they do not change selection rules.

P1 treats empty fields, positive infinity, and negative infinity as missing. Numeric features use finite `S_train` medians; JSD categorical columns use `S_train` modes, breaking ties by the smallest value. Entirely missing source-training columns use 0. Standardization means and population standard deviations are fitted only on imputed `S_train`; constant columns use scale=1.

Numeric JSD columns use 10-quantile boundaries from `S_train`, collapse duplicate boundaries, and extend outer intervals to negative/positive infinity. Categorical vocabularies come only from `S_train`; new target values map to OTHER. Per-feature JSD uses log2, averaged equally across 49 columns.

## Device rules

Execute sequentially on one GPU. RF, XGBoost, MLP, Platt, CRE, and GPU-supported numerical work use CUDA:0. CSV decoding, mmap, manifest joins, and documentation use CPU. Runtime code does not automatically fall back to CPU.

## Result-exposure registration

Existing project material contains only field, time, class, partition, and window statistics, with no base-model, CRE, G0/G1, or outer-model results. P0/P1 do not evaluate models. Exposure to previous model results outside this project was not supplied with the handoff and must be registered by the user before P7.

Configuration exposure fields directly record both facts: no model results in the current workspace through P1; outside-project history absent from the handoff. The user's outside-project exposure declaration is a P7 prerequisite and does not prevent P2–P6 development execution under fixed rules.

## Freeze-record revision

Freeze record v2.1 standardized status descriptions at 2026-09-05T16:01:28Z. G0/G1 changed from proposed to adopted and frozen. Round 2 distinguishes fixed selection rules from development-selected values not yet produced. Result exposure became structured facts. No parameter values, data membership, preprocessing statistics, JSD references, or P1 caches changed.

## Stage boundaries

P0 completes environment, configuration, and protocol registration. P1 completes four-domain feature caches, full anomaly counts, source preprocessing, and JSD references. P2 begins training the six base detectors.

## P6 coordination implementation registration

At 2026-09-06T14:41:48.449083+00:00, `configs/round2.json` was frozen before P6 fitting and comparisons. Existing execution v2.1 rules continue unchanged. F1/F2 enter full development; F3 is registered as ineligible. The main condition remains 10k JSD linear CRE.

Completed rules: Post-CRE has 12 inputs comprising mask3, p3, r3, fused probability, maximum observed probability disagreement, and shift. Missing slots use 0; total weights of 0 produce uniform fusion; classification threshold is 0.5. Corrective and Post models use CUDA logistic regression with the existing solver. Post training equally weights 7 subsets and two domains. The current detector is the most recently queried one; selection ties follow RF→XGB→MLP.

Risk upper bounds use the one-sided 95th percentile from 2000 paired 10-window block-bootstrap replicates, sampled independently by domain. Divide equal-domain error mass by accepted mass; empty-acceptance resamples have risk 1. Early stopping maximizes early accepted coverage among all records; the final threshold maximizes coverage among remaining terminal records. Both require risk upper bound≤0.05 and coverage≥0.10. Break ties by lower risk, then higher threshold. Disable early stopping if infeasible; abstain at termination if no final threshold is feasible. Intervals are pointwise development-selection evidence.

Comparators include 6 fixed cascade orders, a learned cascade with cost-free corrective modeling, random/cheapest/current-u-only routing, best single model, uniform/static/confidence fusion, and soft/hard MoE. Soft MoE uses all queried evidence; hard MoE queries the initial detector before selecting one expert and counts that initial cost. Ensemble and soft-MoE confidence scores serve only ranking and thresholding; querying uses Post-CRE. All methods select cap1/2/3 under the same risk rules. Budget matching constrains calls and measured normalized cost, allowing preregistered random mixtures of two operating points and explicitly reporting expectations. Detailed permissions, candidates, selection, and infeasible branches appear in round2.json.

Single-flow costs use 500 fixed R_cal positions per domain, 30 warmups, and 1000 measurements. They include host input, source preprocessing, feature views, detector, Platt/h/u, and synchronized output. CRE/routing/Post and amortized JSD are additional; cache-reading speed and 1000-record batch throughput are reported separately. P4/P5 development results are known; outer results remain blinded.

Before the first P6 fit, charging details were completed: budget/cost tables include measured single-flow query-chain medians, per-method stepwise Pre-CRE/routing/fusion/Post-CRE overhead, and per-flow amortized P3 JSD. The shared reference combines three detector costs, maximum full-method overhead, and shared shift amortization; composite values are labeled estimates from measured components. Full fusion and soft/hard MoE use final acceptance thresholds only; best single uses cap1, and hard MoE uses cap1/2.

## P6 completion registration

At 2026-09-07T01:31:16.537882+00:00, all 20 models in each of F1/F2 converged, and selections completed for 17 methods and 48 operating points per fold. F3 remains NOT_ELIGIBLE. Selected values and infeasibility states are frozen in outputs/round2_v2/P6/round2_frozen.json and both per-fold frozen_policies.json files. F1 coordinator cap3 thresholds are 0.71/0.94. F2 has no feasible early or final acceptance threshold, so early stopping is disabled and terminal decisions abstain under preregistered rules. P4/P5 conclusions are preserved; P7 and outer unblinding have not occurred at this registration point.

Final status descriptions distinguish DISABLED_BY_CAP, DISABLED_BY_METHOD, and DISABLED_NO_FEASIBLE_THRESHOLD; this only refines status reporting. A threshold-detail synchronization error was repaired by fully recomputing 96 candidate groups with the original selection module; frozen thresholds, metrics, and per-record decisions agree. Validation passed for 192 conditional replays, 12 identity files totaling 84670000 records, and 779 upstream file registrations. Reported runtime is the recorded lower bound of 7720.060 seconds; sampled GPU-memory peak is 4941.5 MiB. Timing gaps and recovery are documented in the stage report and manifests.
