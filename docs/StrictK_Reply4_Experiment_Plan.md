# Reply4 Strict-K threshold-calibration experiment plan

Status: design draft, 2026-09-14. This is a separate exploratory branch after A, B1, and B2; it has not been executed at this plan date. Values below are proposed fixed choices for this round, not requirements quoted from reply4 or an existing preregistration.

## 1. Research question and boundaries

With detectors frozen, can 400 target-domain labels improve the classification boundary or help select which predictions to accept?

Study two operations separately: C1 changes the attack-classification threshold and accepts every record; C2 fixes classification at 0.5 and changes the acceptance threshold. The first round does not adjust both jointly, retrain detectors/Platt/f(u), or execute conformal prediction, coordination, or open-set experiments.

Preserve P7, A, B1, and B2. All three current target domains have already been inspected. Excluding calibration records prevents direct fitting/evaluation overlap in this round but cannot restore a new unseen-domain validation. New independent-domain validation requires a subsequent design.

## 2. Experimental conditions

| Condition | Classification rule | Acceptance rule | Label source and role |
|---|---|---|---|
| C0-class | p≥0.5 | Accept all | No target labels from this fold |
| C1-class | p≥t_K | Accept all | K target labels select the classification threshold |
| C0-select | p≥0.5 | s≥a_dev | R_select in two development domains selects acceptance; no target labels from this fold |
| C2-select | p≥0.5 | s≥a_K | The same K target labels select acceptance |

Compare C1 against C0-class for classification thresholds and C2 against C0-select for acceptance thresholds. C1 and C2 have different acceptance rates; do not declare a winner from one aggregate error rate. Results without abstention also provide context for C2.

Classification uses six models: RF-Traffic17, RF-All49, XGBoost-Protocol17, XGBoost-All49, MLP-Behavior15, and MLP-All49. Retain each domain and Raw/Platt version, giving 36 C1 configurations. Feature-view variants are prespecified controls and cannot be removed based on target outcomes.

Acceptance uses only the three All49 models, each with fixed B1 1-u and the fold-specific frozen B2 f(u), separating Raw/Platt. There are 36 C2 configurations across three domains. Do not fit additional f(u) models for feature-view variants.

## 3. Separating target calibration from evaluation

The primary budget is K=400; do not scan multiple K values in the first round.

Use one fixed startup period: the earliest 10 complete 10k windows in each domain's P7 T_heldout, availability rank [0,100000). Sample 400 records uniformly without replacement using PCG64 and seed 20260914. Apply the same algorithm independently to each domain's ranks. Sample before reading selected labels, using ranks only, without alerts, true classes, or attack families.

This simulates accumulating 100000 flows at a new site, randomly labeling 400, then deploying thresholds. The added cost is waiting for 10 windows; calculate and report elapsed time from timestamp fields. This is neither the first 400 incoming flows nor review of 400 alerts.

Random startup sampling reduces concentration from using the earliest 400 consecutive flows. It does not guarantee class balance or exchangeability over time. After viewing labels, do not move the startup period, resample a seed, or supplement classes.

All methods in a domain share the same 400 annotations. Charge 400 labels per simulated deployment, without charging again per model or method. Only these 400 labels may enter target-threshold selection; other startup labels cannot be used for fitting or selection.

Evaluate all methods on the same subsequent interval:

| Domain | Calibration candidate startup interval | Subsequent evaluation interval | Evaluation records |
|---|---|---|---:|
| ToN | [0,100000) | [100000,27520000) | 27420000 |
| CIC | [0,100000) | [100000,20110000) | 20010000 |
| BoT | [0,100000) | [100000,16930000) | 16830000 |

Exclude the entire startup period, including 99600 unlabeled records. Preserve original window boundaries and excluded tails; do not rebuild windows. Subsequent evaluation totals 64260000 records in 6426 complete windows. Recompute C0 on this scope; do not copy A/B1/B2 metrics covering the full target domain.

Use each fold's development domains under the original LODO roles; B2 f(u) comes only from its two development domains. Reuse of data roles across folds does not constitute additional independent samples.

## 4. Small-sample support

Minimum primary support is Benign≥20 and Attack≥20 among the 400 selected records. The value 20 is an operational minimum for this experiment, not a theorem establishing risk guarantees or statistical sufficiency. A 5% false-positive rate on 20 benign records corresponds to one false positive; report counts alongside rates.

If support is insufficient, mark TARGET_CLASS_SUPPORT_INSUFFICIENT. Do not acquire extra labels or expand startup. C1 and C2 produce no new deployable thresholds. Retain C0 results and the support shortfall.

Insufficient support does not establish model failure, and an unavailable policy must not be reported as zero risk. Do not automatically switch to balanced sampling. In particular, lack of benign labels in naturally sampled BoT records is a practical limitation of K=400.

If too few configurations are comparable, report support first and discuss another sampling design separately. Do not change execution to 200 labels per class.

## 5. C1 classification-threshold selection

Maximize empirical Attack recall subject to empirical Benign FPR≤5% on target calibration.

Candidates are distinct probabilities from the 400 records, plus 0, 0.5, and an all-Benign sentinel greater than 1. Always classify with p≥t and treat probability ties as whole groups. Candidates cover every realizable calibration partition. Generate Raw/Platt candidates separately; a 0.01 grid alone could miss useful boundaries near 0.

Among FPR-feasible candidates, select in this order:

1. Maximize Attack recall;
2. Break ties by minimizing FPR;
3. Then choose the threshold closest to 0.5;
4. Then choose the larger threshold.

Retain calibration metrics at t=0.5 in the same comparison table. Compare integer false-positive/sample-count ratios to avoid floating-point boundary errors.

If the best feasible recall is 0, mark NO_NONTRIVIAL_CLASSIFICATION_SOLUTION. Preserve the degenerate operating point for diagnosis. Do not call it success or silently replace C1 with 0.5.

After selecting the threshold, accept all records. Do not call an acceptance model or refit a correctness model. Save candidate summaries and the final threshold before reading subsequent evaluation labels.

The primary 5% constraint is benign FPR, distinct from the C2 5% selective-risk constraint.

## 6. C2 acceptance-threshold selection

Fix classification at p≥0.5. Preserve B1/B2 scores: s=1-u or the fold's frozen f(u), with higher scores preferred. Score inputs and classification use the same Raw/Platt version.

Candidates are distinct scores from the 400 calibration records, the accept-all boundary, and an abstain-all sentinel. Allow only whole-group prefixes s≥a. Do not reverse ranking, set separate thresholds by predicted class, or use true classes to decide acceptance.

Maximize acceptance among candidates with calibration selective risk≤5% and acceptance≥10% (at least 40 records). Break ties by lower empirical risk, then higher threshold. Overall calibration support remains ≥20 for each class. Do not change the objective because few records from a class are accepted.

If none is feasible, mark NO_FEASIBLE_ACCEPTANCE_THRESHOLD and save an abstain-all fallback. Subsequent coverage=0, risk=NA, and effective attack detection=0. Full abstention does not establish successful safe deployment.

This is empirical threshold selection, without a finite-sample risk guarantee. Searching and checking many thresholds on the same data creates selection optimism. A conventional binomial interval around the chosen point does not establish a 95% risk guarantee; temporal dependence also matters.

This stage tests transfer of the specified limited-label strategies. It does not introduce conformal, learn-then-test, or another risk-control algorithm.

## 7. C0-select control without target labels

For the same 36 score configurations as C2, use all R_select records from both development domains and select a_dev under the same empirical 5% risk and 10% acceptance objectives. This is an explicit new baseline, not a claim that B1/B2 already had deployment thresholds.

Weight domains equally. Let c_d be each domain's accepted fraction and e_d its accepted-error fraction relative to all records. Overall coverage=(c_1+c_2)/2 and risk=(e_1+e_2)/(c_1+c_2). Do not directly average the two domain-specific selective risks.

Candidates are distinct scores from both development domains and accept-all/abstain-all sentinels; use C2 tie-breaking. Do not select scores, models, or Raw/Platt versions from target outcomes.

Abstain entirely if infeasible. Development sample sizes differ substantially from K and must be disclosed. This compares calibration sources and available budgets, not an equal-sample-size causal experiment.

## 8. Evaluation metrics

### Classification branch

Report calibration and subsequent TN/FP/FN/TP, FPR, Recall, Balanced accuracy, Accuracy, and recall by original attack category. The primary comparison is C1-C0-class.

Changing classification thresholds leaves attack probabilities unchanged, so AUROC, AP, and attack-probability Brier/ECE should be unchanged. Use these as consistency checks, not improvement endpoints.

### Acceptance branch

Report actual acceptance, selective risk, acceptance for both true classes, FPR among accepted Benign, false-negative rate among accepted Attack, correctly accepted Attack/all Attack, Attack incorrectly accepted as Benign/all Attack, and attack abstention. The primary comparison is C2-C0-select.

Retain original attack-category breakdowns, especially ToN scanning, BoT Reconnaissance, and CIC FTP-BruteForce/SlowHTTPTest. Zero support is NA; zero-acceptance risk is NA.

Acceptance thresholds leave scores and classifications unchanged, so full AURC should be unchanged. Compare selected operating points; do not treat AURC change as the target.

### Temporal stability

Split subsequent windows into four consecutive, nonoverlapping segments of approximately equal length, giving earlier segments one extra window when necessary. Report the same metrics per segment, retaining thresholds without recalibration from segment outcomes.

### Statistics

Use 10 consecutive complete windows per block, 2000 paired block-bootstrap replicates, retain the final short block, and require at least 20 blocks. Generate and save a new resampling matrix for this subsequent interval, shared by all methods in a domain. Do not reuse old P7 matrices that include startup.

Classification focuses on difference intervals for FPR, Recall, and BA. Acceptance reports paired differences in coverage, effective attack detection, and incorrectly accepted attack fractions. Report selective-risk differences only when both sides have nonzero acceptance, always alongside coverage; do not call them improvements at matched coverage. If a class or accepted set is empty in a resample, use NA and disclose valid replicate counts, not 0.

Intervals are conditional on one 400-record calibration sample, frozen models, and current evaluation domains. They do not cover calibration-sampling uncertainty or all future domains. Use one prespecified seed without outcome-driven resampling. Repeated calibration samples and K sensitivity require separate decisions.

## 9. Conclusion criteria

For C1, first check subsequent FPR≤5%, then whether attack recall improves over 0.5. If both FPR and recall fall, report the tradeoff. Do not choose a success narrative solely from BA/Accuracy. Meeting 5% is empirical on these data.

For C2, first check subsequent risk≤5% and coverage≥10%, then benign false positives, effective attack detection, and class-specific misses. Do not introduce an outcome-dependent minimum attack-detection rate. Without a prespecified operational detection requirement, report actual detection and do not infer deployability from the two aggregate gates alone.

Do not conflate the two 5% criteria, generalize the best ToN model to all domains, or omit insufficient-support/infeasible/full-abstention configurations from summaries. This stage supplies no single overall success score.

## 10. Execution order and delivery

First run C-preflight: verify paths, window scope, calibration-rank sampling, and label support without subsequent performance. Produce a support report and protocol configuration. Check support of 20 per class within 400 labels, actual startup waiting time, identity overlap, and evaluation counts.

Review preflight results, freeze the protocol, then run C1. After C1 review, run C2 and C0-select. Reuse the original 400 records and evaluation interval; do not change C2 from C1 results. Retain unsupported configurations as specified.

Suggested independent output directory: outputs/post_p7_diagnostics/strict_k_reply4/<run_id>/, containing:

- protocol.json and protocol-revision/freeze times, recording known A/B1/B2 outcomes;
- calibration_sample_manifest.csv、calibration_support.csv、evaluation_scope.csv；
- C1 classification candidates/thresholds and C0-select/C2 acceptance candidates/thresholds;
- calibration_evaluation_gap.csv；
- classification_metrics.csv、selective_metrics.csv、attack_category_metrics.csv；
- temporal_metrics.csv, paired_comparisons.csv, and resampling matrices;
- stage_report.md, validation records, scripts, commands, and logs.

Save only necessary calibration identities and aggregates, without duplicating full caches or adding large per-flow traces. Check accept-all results, category aggregates, calibration/evaluation separation, threshold-fitting information boundaries, and unchanged upstream files.

Summarize A, B1, B2, and Strict-K, then stop to decide the roadmap. Do not automatically proceed to joint thresholds, new CRE, conformal prediction, or coordination.

## 11. Statistical background

Risk control during threshold search requires separate methods and assumptions; empirical compliance does not automatically guarantee population risk. See Geifman and El-Yaniv on selective classification and Angelopoulos et al.'s Learn then Test framework. These references explain statistical boundaries; their algorithms are not implemented here.

- https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
- https://arxiv.org/abs/2110.01052
