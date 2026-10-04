# Core project conclusions after P7

Date: 2026-09-08. Status: CORE_AFFIRMATIVE_CLAIMS_NOT_SUPPORTED_BY_OUTER_EVALUATION. Basis: official `p7-outer-v1.0` results and artifact validation. This report was written after outer-result unblinding; it is not a preregistration and does not amend the P4/P6 or Item3 freeze history.

## 1. Overall assessment

Engineering execution and experimental procedures completed successfully: P7 evaluated 65,560,000 outer records in complete windows across three Round1 folds, 96 frozen Round2 operating points for F1/F2, and 306 budget entries. Artifact validation passed, and the original inputs and frozen selections were preserved.

The central positive scientific claims in the original plan were not supported by outer evaluation. Current evidence supports recoverable, complementary detector errors in some target domains. It does not support a stable improvement in cross-domain conditional reliability from causal shift, or superiority of explicit dynamic coordination over simple cascades or MoE under frozen risk and cost constraints. The paper or project summary should therefore present an informative negative result and withdraw the full positive Conditional Reliability-Guided Coordination claim.

## 2. Round1 outer gates

G0 is the best single-detector error rate minus the rate at which all three detectors are wrong. G1 is `AURC(f(h,u))-AURC(f(h,u,s))`; a positive value means that adding shift improves reliability ranking.

| Scope | G0 and 95% CI | G0 conclusion | G1 and 95% CI | G1 conclusion |
|---|---|---|---|---|
| Equal weighting across three domains | 0.121381 [0.108481, 0.135089] | PASS | -0.054990 [-0.060981, -0.048426] | BELOW_MINIMUM_EFFECT |
| F1→ToN | 0.241858 [0.219732, 0.267022] | PASS | -0.166935 [-0.184630, -0.147221] | BELOW_MINIMUM_EFFECT |
| F2→BoT | 0.004240 [0.001650, 0.007355] | BELOW_MINIMUM_EFFECT | -0.000138 [-0.000390, 0.000117] | BELOW_MINIMUM_EFFECT |
| F3→CIC | 0.118044 [0.087209, 0.151466] | PASS | 0.002103 [0.000905, 0.003345] | PASS |

Complementarity depends strongly on the target domain. ToN and CIC have substantial oracle headroom, whereas BoT's 0.424 percentage points fall below the preregistered minimum effect of 1 percentage point. Passing the equal-domain G0 gate establishes aggregate correction opportunities; it does not establish that cooperation is needed in every unseen domain.

The aggregate G1 result is negative. Equal-domain differences are negative for all three detectors: RF=-0.020223, XGB=-0.093887, and MLP=-0.050859; all three 95% intervals lie below 0. The positive CIC result does not offset the substantial degradation on ToN. Section 38 of the frozen protocol requires withdrawing the positive conditional-reliability claim. Passing P4 development gates cannot replace outer evidence.

## 3. Round2 outer results

Frozen F1 policies consistently violate the 5% selective-risk target on ToN. None of the 36 fixed operating points with nonzero coverage achieves outer risk≤0.05. All 121 budget entries feasible during development also exceed 0.05 on the outer domain.

| ToN operating point | Coverage | Selective risk and 95% CI | Mean calls | Normalized component cost |
|---|---:|---|---:|---:|
| Coordinator cap1 | 86.35% | 63.24% [59.34%, 66.89%] | 1.000 | 5.316 |
| Coordinator cap2 | 92.01% | 64.28% [60.62%, 67.70%] | 1.136 | 6.040 |
| Coordinator cap3 | 92.47% | 64.02% [60.30%, 67.52%] | 1.292 | 6.663 |
| Development-selected fixed cascade fixed_102 cap3 | 100.00% | 51.60% | 1.513 | 4.874 |
| Lowest risk among fixed points: fixed_120 cap3 | 35.87% | 42.66% | 2.794 | 8.082 |

Relative to the development-selected fixed_102 cap3 cascade, Coordinator cap3 has approximately 7.53 percentage points less coverage, 12.41 percentage points more risk, and higher cost. Paired intervals at the same frozen budget, for calls budget≥1.75, give a coordinator-minus-fixed_102 risk difference of [9.52, 15.23] percentage points at 95% confidence, a coverage difference of [-9.19, -5.98] percentage points, and a cost difference of [1.41, 2.18]. These results do not support incremental value from explicit coordination.

Consultation diagnostics show that locating correction opportunities does not establish final-system reliability. Coordinator cap3 captures 94.75% of queried oracle opportunities at step two, but correction yield per consultation is only 19.40%, consultation harm is 37.72%, and step-three correction yield is only 3.15%. A high opportunity-hit rate cannot replace final risk–coverage–cost evidence.

F2 already lacked a feasible final acceptance threshold during development. All 48 frozen BoT outer operating points retain coverage=0 and undefined risk. F3 does not execute Round2 under its frozen eligibility decision. No independent outer fold provides both acceptable risk and evidence that coordination outperforms strong baselines.

## 4. Conclusions to retain or withdraw

Retain:

- Detector errors show substantial complementarity on average across three domains, with BoT as an exception; correction opportunities are domain-dependent.
- Development-domain G0/G1 and coordination diagnostics remain valid intermediate results and illustrate how reliance on development proxy domains can produce misleading optimism.
- Outer results provide useful negative evidence: marginal JSD shift inputs, source-conditioned competence, and the current binary reliability model do not transfer consistently to unseen domains.
- Strict freezes, failure branches, full abstention, and ineligible folds were preserved. The experiments support research on failure modes of cross-domain selective prediction.

Withdraw:

- The positive claim that adding shift consistently improves cross-domain conditional reliability.
- Any safety or reliability claim that the coordinator meets a 5% risk target in unseen domains.
- The claim that explicit agent-style coordination outperforms, or must replace, fixed cascades, MoE, or single models.
- Interpreting development G0/G1 passes, oracle headroom, or G2 opportunity hits as final deployment benefits.
- Using unexecuted Item3 open-set results to rescue the original Round1/2 claims.

## 5. Project status and next steps

Record project status as `VALID_COMPLETED_NEGATIVE_CORE_RESULT`. Engineering execution completed; the original method's affirmative hypotheses failed. The immediate next step is P9 core summaries, figures, and paper narrative, reframing the question as when cross-domain conditional reliability and selective coordination fail, and why development gates cannot ensure safety in unseen domains.

Item3-v1.0 remains frozen. O3 remains not estimable. Even if O1/O2/O4 are later executed, they are separate descriptive extensions under controlled families and cannot alter this report's confirmatory outer conclusions for the original method. Given failed reliability and coordination transfer, immediate investment in the full P8 coordination chain is not recommended. If open-set material is required, first perform low-cost data-identity, scenario-filtering, source-prototype, and threshold-computability steps; make a separate decision before large-scale scenario refitting.

Any improvement designed after these outer results—such as replacing shift statistics, recalibrating reliability, domain-robust training, or new thresholds—is exploratory after result exposure. It requires new validation domains or a new independent time period. Modifying the method on the same ToN/BoT/CIC outer data cannot produce new confirmatory evidence.

## 6. Evidence locations

- Official P7 report: `outputs/outer_v2/P7/stage_report.md`
- Three-domain Round1 aggregate: `outputs/outer_v2/P7/round1_aggregate.json`
- Per-fold Round1 results: `outputs/outer_v2/P7/F*/round1/metrics.json`
- F1/F2 policy and budget results: `outputs/outer_v2/P7/F*/round2/policy_metrics.json` and `budget_metrics.json`
- Paired budget comparisons: `outputs/outer_v2/P7/F*/round2/paired_budget_comparisons.json`
- Validation record: `outputs/outer_v2/P7/Artifact_Validation.json`
- Flat tables for review: `outputs/outer_v2/P7/core_fold_results.csv`, `core_policy_results.csv`, and `core_budget_results.csv`

Statistical intervals are conditional on the frozen models and these three evaluation domains; they do not quantify uncertainty across all future domains. Paired budget comparisons are pointwise descriptive intervals, with no newly introduced multiplicity-adjusted superiority gate.

English delivery note: the historical narrative total of65,560,000 below is retained as originally reported. The [professor briefing](Professor_Progress_Report_After_P7.md) corrects the sum to64,560,000 complete-window records from fold artifacts, without changing frozen metrics.
