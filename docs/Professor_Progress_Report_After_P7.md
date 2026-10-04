# Research Progress After P7 — Professor Briefing

2026-09-08 | Status: `VALID_COMPLETED_NEGATIVE_CORE_RESULT`

**Summary.** Source detectors, development reliability and consultation models, and frozen three-fold outer evaluation are complete; artifact review passed. Detector errors offer complementary opportunities, but the primary shift-augmented reliability estimator deteriorates overall. ToN coordination incurs approximately 63%–64% selective risk; BoT abstains entirely; CIC was ineligible for full coordination. The frozen method’s core positive claims lack outer support. This does not reject conditional reliability or multi-agent research generally. Complete P9 and bounded diagnostics before deciding on independently validated improvements; defer P8.

## 1. Intended goal and actual experiment

The **long-term goal** is reliability-driven collaborative NIDS, followed by Known/Unknown/Abstain decisions. **RQ1:** can unseen-domain prediction correctness be estimated better than by confidence or fixed historical competence? Current tests cover detector complementarity, conditional reliability estimation (CRE), consultation and baselines—not a complete autonomous agent or open-set system.

| Executed pipeline | Data roles and operations |
|---|---|
| Source training | NF-UNSW-NB15-v3; chronological/group partitions for training, probability calibration, competence, and source testing, nominally 60/15/15/10. Fit preprocessing, detectors, Platt calibration, and historical competence `h`. |
| Development fitting and selection | F1: CIC+BoT; F2: CIC+ToN; F3: BoT+ToN. Within each domain, chronological `R_fit/R_cal/R_select=50/25/25`: fit CRE/consultation models, choose the initial detector, then thresholds and budget policies; equal domain weights. |
| Freeze | Fix P4 primary conditions and P6 models, detector choices, stop/accept thresholds, infeasible branches, and budget mixtures. P7 performs no retraining or reselection. |
| Held-out evaluation | F1→NF-ToN-IoT-v3; F2→NF-BoT-IoT-v3; F3→NF-CICIDS2018-v3. Each target is excluded from its own fold’s fitting/selection, although it serves as development data in other folds. |
| Aggregate | Round1 in all three folds; Round2 only in F1/F2; equal domain weights. Predictions and policy decisions were completed before outer labels were read for scoring. |

Sources: `configs/execution_v2.json`, `docs/protocol_decisions.md`, P7 `predictions_complete.json`. Strict-0 excludes outer training/adaptation labels, allowing labeled development proxies and causal unlabeled target statistics.

RF uses Traffic 17 features, XGBoost Protocol 17, MLP Behavior 15; all-49 controls were also trained. Inputs exclude IP and absolute timestamps; time orders records. Logistic `f(h,u,s)` predicts binary correctness: `h` is source prediction-class competence; `u` is normalized entropy of source-calibrated probabilities; `s` averages 49 marginal JSDs from the preceding complete window against the source. Unqueried outputs are unavailable; queried probabilities, reliability, masks and disagreement enter fusion/Post-CRE.

Round2 uses correction-opportunity/cost routing, reliability-weighted fusion, stopping and abstention. Controls: single detector, six fixed cascades, learned cascade, soft/hard MoE and fusion/routing ablations. Development thresholds use a 0.01 grid, **one-sided 95% risk upper bound≤5%, coverage≥10% within the relevant stage**—not an unseen-domain guarantee. Call budgets are 1–3 in 0.25 steps, alongside cost limits. Costs sum measured component medians, including CRE/routing/fusion/shift, normalized by median detector-query cost; deployment end-to-end latency was not measured.

Windows contain 10,000 completed-flow records. Percentile 95% CIs use 2,000 paired bootstrap replicates, ten windows/block, retaining the final shorter block; at least 20 blocks/domain. Methods share draws; domains are independently resampled. G0 recomputes the best single model per replicate. CIs condition on frozen models and these domains.

**Stage status:** P0 environment/registration, P1 data/features, P2 source models, P3 replay, P4 development primary experiments, P5 development sensitivities, P6 frozen coordination, and P7 evaluation/review are complete. P7 **does include all-49 detector classification and complementarity**. What remains unextended is P5 reliability sensitivity: 5k/50k JSD, 10k MMD, nonlinear CRE. P8 has only frozen Item3 rules, without scenario fitting/execution; O3 full coordination is not estimable. P9 has a core conclusion and this briefing; manuscript materials remain unfinished.

## 2. What changed from the Proposal

The earliest available proposal is already **Revised**. The original and some cited revisions were not found, limiting reconstruction of the original agent design. Only traceable changes are attributed; historical documents remain intact.

| Item | Proposal baseline that can be traced | Actual change or narrowing | Reason and implication |
|---|---|---|---|
| System scope | Collaborative NIDS vision; earliest available Revised already uses staged controlled tests | Current work ends at detector/CRE/consultation outer evaluation | Test reliability and incremental value first; neither validate nor reject complete agent capabilities. |
| Feature heterogeneity | Revised explicitly describes original all-49 access for each model | Disjoint 17/17/15 primary views; all-49 controls retained | Already a Revised design, not a P7 innovation. P7 includes all-49 classification/complementarity, without a new all-49 CRE/coordination chain. |
| Reliability information | Revised already includes `h/u/s`, Pre/Post-CRE and post-query disagreement | Implement paired `hu/hus`, entropy-only and historical baselines, enforcing queried-information access | Isolate shift contribution and prevent free peer evidence; conditional reliability itself is not newly added. |
| Shift/statistics | Causal JSD primary and MMD sensitivity already specified | Freeze source bins and 10k primary windows; run development window/MMD/nonlinear sensitivities | Operationalize the design; sensitivities cannot replace a failed primary condition. |
| Coordination/freeze | Cost-aware consultation, thresholds, budgets, cascade/MoE controls already present | P6 registers six directed joint-correction models, Post-CRE, accounting and selection; P7 replays frozen choices | Make comparisons auditable; development selects, outer evaluation cannot optimize choices. |
| F2/F3 branches | Gates and infeasible/inconclusive branches already planned | F2 has no feasible acceptance threshold; F3 development G1 is inconclusive | Outcomes, not post-hoc fold exclusions. F3 outer G1 passing cannot reactivate coordination. |
| Item3/P8 | Original Round3 anticipated O1–O4 and a single-Known pseudo-unknown step | Support audit makes single-Known leave-family-out validation inapplicable; freezes prototype/distance/source quantile rules; O3 non-estimable | Unknown must be quarantined before all fitting, requiring separate refits; limited support cannot validate broad open-set recognition. |
| Overall plan | P7→P8→P9 | Before unblinding: P7→core summary→conditional P8; after unblinding: negative conclusion and recommendation to defer P8 | Change priorities and claims, not old algorithms, thresholds or outcomes. |

**Timeline, UTC:** Sep 5 16:01 common-rule freeze; Sep 6 14:41 P6 rules registered before fitting, with P4/P5 development results known; Sep 7 01:31 P6 choices frozen; Sep 7 23:57 Item3 frozen after support audit; Sep 8 02:52 plan-status revision and 02:59 P7 implementation registration; 03:35 P7 completed. Item1/2 review notes synchronized Proposal/Protocol documents postdate P6 fitting: configuration and freeze snapshots, not synchronization dates, substantiate prior rules. Post-P7 reporting is interpretation, not preregistration. [S1–S3]

## 3. Key outer results

### Table A. Complementarity and incremental shift value

G0 is best-single error minus the three-detector joint-error rate: oracle headroom, minimum useful effect 0.01. G1 is `AURC(hu)−AURC(hus)`: positive means better ranking with shift, minimum effect 0.001; lower AURC is better. With sufficient blocks, **PASS requires estimate≥minimum and CI lower bound>0; an upper bound below the minimum gives BELOW; otherwise INCONCLUSIVE**.

| Target / aggregate | G0 [95% CI] | Gate | G1 [95% CI] | Gate |
|---|---:|---|---:|---|
| F1 ToN | 0.241858 [0.219732, 0.267022] | PASS | −0.166935 [−0.184630, −0.147221] | BELOW |
| F2 BoT | 0.004240 [0.001650, 0.007355] | BELOW | −0.000138 [−0.000390, 0.000117] | BELOW |
| F3 CIC | 0.118044 [0.087209, 0.151466] | PASS | 0.002103 [0.000905, 0.003345] | PASS |
| Equal-domain aggregate | 0.121381 [0.108481, 0.135089] | PASS | −0.054990 [−0.060981, −0.048426] | BELOW |

Sources: P7 `round1_aggregate.json`, `core_fold_results.csv`; `G0/G1.estimate, CI95, status` [S4]. BELOW abbreviates `BELOW_MINIMUM_EFFECT`; G1 also weights detectors equally. Aggregate G0 represents **12.14 percentage points (pp)** of oracle error headroom, not realized coordination gain. BoT G1 crosses zero but its upper bound remains below the useful-effect threshold. Development F1/F2 passed both gates; F3 G1 was inconclusive. Frozen eligibility remains unchanged.

### Table B. RQ1: ranking and probability quality

| Correctness score | AURC ↓ | Brier ↓ | NLL ↓ |
|---|---:|---:|---:|
| Development correctness constant `prior` | 0.503094 | 0.376997 | 0.966068 |
| Raw entropy confidence `1−u` | 0.512020 | — | — |
| Source historical competence `raw_h` | 0.484563 | 0.492207 | 3.719138 |
| Entropy-only logistic `f(u)` | 0.475369 | 0.323282 | 0.894401 |
| No-shift CRE `f(h,u)` | 0.513259 | 0.358287 | 1.008243 |
| Primary CRE `f(h,u,s)` | 0.568249 | 0.457561 | 1.563058 |

Source: P7 `F*/round1/metrics.json → CRE.<detector>.all.<method>.{AURC,Brier,NLL}` [S4]. Arithmetic means across three domains×three detectors; **not pooled metrics or new significance tests**. Ranking-only `1−u` has no registered Brier/NLL. Complete paired CIs for this nine-cell summary were not separately saved.

**Separate two conclusions:** adding shift worsens `hu` overall; `hu` itself has no aggregate AURC advantage over confidence/history. Yet `hu` improves Brier over `raw_h`, and entropy-only `f(u)` has better aggregate point estimates on all three metrics than `hu/hus`. “All CRE is ineffective” is therefore unwarranted. Ranking and probability calibration differ; current evidence does not support stable incremental value from the more complex inputs. Individual domain/model improvements do not establish general transfer.

**`f(u)` ranking audit.** Logistic coefficients are sign-unconstrained: five of nine frozen models (F1/F2 RF, all MLP) rank high entropy first, reversing `1−u`. Four negative-coefficient models have outer AURC differences below 10⁻¹². Saved tie-group counts match in all nine cases. Reconstruction on 180,000 deterministic sampled records gives maximum error 2.22×10⁻¹⁶ and no monotonicity violations beyond tolerance. The principal difference is a **development-learned ordering reversal**, not outer reselection, probability calibration alone, or an identified ranking bug. F3 MLP improves overall AURC 0.480104→0.300143, yet worsens within Attack: 0.903544→0.920916. Aggregate gains do not establish within-class gains. [Detailed audit](<../../NIDSProject_Professor_History_20261004/docs/Professor_Progress_Report_After_P7_Followup_EN.md>)

| Outer domain | Benign / Attack records | Attack share | Primary RF false-positive rate | Primary XGB attack recall |
|---|---:|---:|---:|---:|
| ToN | 16,791,954 / 10,728,046 | 38.98% | 76.66% | 0.02% |
| BoT | 51,434 / 16,878,566 | 99.70% | 84.02% | 0.79% |
| CIC | 17,509,097 / 2,600,903 | 12.93% | 87.60% | 0.45% |

Source: P7 `F*/round1/metrics.json → audit, detectors.all.<model>.{FP,TN,recall}`. FPR conditions on Benign; recall on Attack. BoT RF achieves 98.96% overall accuracy despite 84.02% FPR; CIC XGB achieves 86.75% accuracy despite 0.45% attack recall. ToN all-49 detector accuracies are 83.61%–85.62%, versus 39.30%–64.15% for primary views: investigate information lost through view restriction, not CRE alone. Different class proportions are observed; pure label shift is not established.

**Development versus outer evaluation:** development is `R_select`, equally weighted across two domains and three detectors; outer averages three detectors. Metrics are first computed within each domain. This is descriptive, not a paired transfer-gain test; original P4 gates and Table A retain their CIs/statuses.

| Fold | `f(u)` AURC development→outer | `hu` AURC development→outer | `hus` AURC development→outer | G1 development→outer |
|---|---:|---:|---:|---:|
| F1 | 0.494774→0.404954 | 0.472406→0.391788 | 0.460481→0.558723 | +0.011925→−0.166935 |
| F2 | 0.463030→0.607589 | 0.424568→0.677413 | 0.381375→0.677551 | +0.043193→−0.000138 |
| F3 | 0.581690→0.413564 | 0.563598→0.470577 | 0.560669→0.468474 | +0.002929→+0.002103 |

Sources: `outputs/round1_v2/P4/F*/CRE_metrics.csv` (`partition=R_select,stratum=all`) and outer `CRE`; exported to `docs/p7_report_followup/development_outer_reliability.csv`. Class/base-error differences prevent interpreting a cross-population AURC reduction directly as better generalization. F1 coordinator cap3 moves from development **3.89% risk/97.38% coverage** to outer **64.02%/92.47%**; the selected cascade moves from **3.92%/100% to 51.60%/100%**. These are frozen-policy transfer outcomes, not reselected comparisons.

### Table C. Round2 risk–coverage–cost

Selective risk is the error proportion among accepted predictions; coverage is the accepted fraction of all evaluated records. Caps limit detector calls. These are frozen cap points, not outer-optimal deployment selections.

| Domain / frozen method | Coverage | Selective risk [95% CI] | Mean calls | Normalized cost |
|---|---:|---:|---:|---:|
| ToN coordinator cap1 | 86.35% | 63.24% [59.34%, 66.89%] | 1.000 | 5.316 |
| ToN coordinator cap2 | 92.01% | 64.28% [60.62%, 67.70%] | 1.136 | 6.040 |
| ToN coordinator cap3 | 92.47% | 64.02% [60.30%, 67.52%] | 1.292 | 6.663 |
| ToN development-selected `fixed_102` cap3 | 100.00% | 51.60% [47.14%, 55.81%] | 1.513 | 4.874 |
| ToN hard MoE cap2 | 95.65% | 65.33% [61.65%, 68.82%] | 1.195 | 4.399 |
| ToN soft MoE cap3 | 71.74% | 65.96% [61.56%, 69.98%] | 3.000 | 8.522 |
| BoT coordinator cap3 (all 48 points abstain) | 0.00% | Undefined | 3.000 | 14.457 |
| CIC full coordination | Not run | Ineligible under frozen rules | — | — |

Sources: P7 `F*/round2/policy_metrics.json → <method>/capN.{coverage,risk,CI95.risk,mean_calls,normalized_cost}`; cascade identity in `selected_fixed_cascade.json` [S5].

At frozen **call budget 1.75 and cost ceiling 8.147**, coordinator cap3, selected cascade cap3, and hard MoE cap2 are already-selected points; all satisfy both realized outer budgets. Soft MoE cap3 is a separate full-query control, outside this budget comparison. Against the selected cascade, coordination increases risk by **12.41 pp** (paired 95% CI 9.52–15.23), reduces coverage by **7.53 pp**, and adds 1.789 cost units. Fewer calls do not imply lower total cost. The cascade’s own 51.60% risk also greatly exceeds 5%; it is not a successful deployment solution. All 36 nonzero-coverage ToN fixed points and 121 development-feasible budget entries exceed 5% outer risk. BoT has no feasible budget selection.

ToN cap3 second-query correction yield/harm are **19.40%/37.72%**; third-query **3.15%/11.56%**, both per consultation. Harm means current-correct/peer-wrong, not necessarily a final fusion error or causal explanation. Opportunity alone does not establish net benefit. [S5: `G2`]

**Scale and review:** Actual total: **64,560,000** complete-window records—ToN 27,520,000; BoT 16,930,000; CIC 20,110,000; 6,456 windows—not independent samples. The requested/previously summarized **65,560,000** is an addition error, exceeding fold artifacts by 1,000,000. This report corrects it from `audit.rows/windows`, without changing fold metrics or frozen results. P7 excludes 9,597 incomplete-tail records. There are 96 frozen policy points and 306 budget entries. Completion: Sep 8 03:35 UTC; 1,578.25 seconds (26.30 minutes, excluding training). RTX A5000 peak sampled memory: 6,565.5 MiB. Filesystem allocation: **292.3 GiB = 313.9 GB**. Prior artifact review passed for ledger shapes, blocks, truth–action matrices, intervals and aggregation, confirming unchanged upstream artifacts/P6 selections. The newly detected narrative total error illustrates that PASS is not an error-free guarantee. [S6]

## 4. Boundaries and next decision

**Supported:** complementary errors in ToN/CIC; development success does not guarantee unseen-domain risk or benefit; a valid, auditable negative result for the frozen method. **Unsupported:** stable shift gains, 5% unseen-domain coordination risk, and incremental coordination superiority over strong baselines. **Unresolved:** conditional reliability generally, complete multi-agent systems, outer P5 sensitivities, Known/Unknown decisions, and improved methods on new domains.

Observed failures include ToN ranking/probability deterioration, many accepted errors, BoT abstention and consultation harm. Loss of relevant structure in marginal shift, non-transferable development relationships, and fusion/threshold mismatch remain **hypotheses**, not established explanations.

| Route | Question | Minimum additional experiment | Data/resources | Continue/stop criterion to register beforehand |
|---|---|---|---|---|
| Retain core direction; bounded optimization | Can simpler reliability and net-benefit routing transfer? | Diagnose ranking/calibration by domain/class; use existing `u/hu` controls; at most one shift/calibration change and one harm-aware routing or fusion/stopping ablation | Exposed caches for exploration; fresh independent domain, frozen fit/selection data and component accounting for confirmation; no broad algorithm search | Require predefined AURC gain and useful coverage; retain risk≤5%, coverage≥10%, with matched-budget cascade/MoE comparison; otherwise stop expanding coordination |
| Narrow to reliability/selective prediction | Is there value without explicit coordination? | Focus on confidence/history/`f(u)` and cascades; separate ranking from calibration and document domain failures | Cached exploratory analysis, statistical reporting and independent validation; no full agent system required first | Useful coverage at predefined risk, or reproducible diagnostic value; better Brier alone is not deployment improvement |
| Conclude negatively and pivot | Is further investment justified? | Complete P9, necessary diagnostics and reproducibility index; no additional model search | Incremental analysis/writing effort; previous GPU/storage spending is sunk cost | No independent validation data, or no improvement signal in one bounded exploratory round: finish the negative-result study and move to a supported question |

**Priority: adopt the second route for the narrative, with the third as the stopping rule.** Ordering, class-stratified and descriptive development–outer diagnostics are now complete. Finish P9 and use this evidence to decide whether to add just one testable ablation. Examine view-related information loss and within-class errors before choosing ranking/calibration or consultation net benefit. Route one requires secured independent data. No-shift models, MMD, nonlinear CRE, Platt, fusion, stopping/abstention and correction models already exist; reusing them is not a new contribution. Existing routing predicts “current wrong and peer right”; an extension should test **net benefit**. Few-label target adaptation changes Strict-0 and needs a separate label budget and validation design.

Exposed P7 data support exploration. Confirmatory claims for new methods need **new independent data**; unseen-domain transfer specifically needs a new domain, while a later interval in an already-seen domain tests temporal transfer only. Preserve P4/P6/P7 and Item3 history; no retraining, reselection or P8 execution was performed for this report. Item3 is a restricted family extension: O3 BoT `R_cal=933, R_select=1,545` cannot form complete 10k windows; O1/O2 selection sets contain only Benign; O4 development Known support comes only from BoT. Existing binary risk bounds cannot validate Unknown semantics. [S3]

## 5. Audit index, gaps, and decisions for the professor

Paths are relative to the project root; `P7/` abbreviates `outputs/outer_v2/P7/`.

| Ref. | Source and field mapping |
|---|---|
| S1 Design | `archive/round1_handoff_v1/Revised Technical Proposal.md` (§2–6, §9, Item3 section); adjacent `Detailed Experimental Protocol and Pre-Implementation Specification.md`; `archive/execution_plan_v2_original/Execution_Plan_ZH.md`. Original full proposal and some cited revisions are missing. |
| S2 Revisions/plan | `docs/protocol_decisions.md`; `outputs/round2_v2/P6/Item12_Implementation_Audit.md`; `docs/Plan_Revision_Diff.md`; `archive/execution_plan_before_p7_20260908T025251Z/`; current `docs/execution_plan_v2.md`. |
| S3 Item3 | `item3_freeze_20260907T235725177310Z/Item3_Freeze_Report.md`, adjacent `before/`, `diffs/`, `snapshots/after/`; `item3_data_support_audit_20260908/Item3_Data_Support_Audit.md`; `item3_protocol_revision_draft_20260908/Item3_Feasibility_Check.md`; `configs/item3_scenarios_v1.json`. |
| S4 Tables A/B | `P7/round1_aggregate.json` and `P7/F*/round1/round1_aggregate.json → gates.G0/G1`; `P7/F*/round1/metrics.json → CRE.<detector>.all`; `P7/core_fold_results.csv → G0,G0_CI,G0_status,G1,G1_CI,G1_status`. Table B keys: `prior,one_minus_u,raw_h,u,hu,hus`; nine-cell mean, then rounding. Development: `outputs/round1_v2/P4/F*/gate_report.json`. |
| S5 Table C/pairs | `P7/F*/round2/policy_metrics.json`, `selected_fixed_cascade.json → cap3.selected_method`; `budget_metrics.json → frozen.{method,calls_budget,cost_budget,selection},outer`; `paired_budget_comparisons.json → baseline,calls_budget,CI95`. Consultation: `G2[].corrective_yield,consultation_harm`. |
| S6 Completion/review | `P7/P7_manifest.json → status,seconds,round2,gpu_memory`; `P7/Artifact_Validation.json`; `P7/F*/round1/metrics.json → audit`; `docs/P7_PreExecution_Review.md`. Current `du -s -B1 outputs/outer_v2/P7`: 313,863,049,216 allocated bytes. |

**Reconciliations and gaps:** The plan retains an earlier “do not run formal outer evaluation” note; its latest status table/P7 manifest establish completion. Earlier “293 GB” conflates GiB/GB. Removing positive reliability claims is limited to the frozen method, not all CRE. This follow-up corrects the previous report: all-49 outer classification/complementarity was executed. The original Proposal, some revisions and a separate external-result-exposure declaration remain missing. Full paired intervals for Table B’s summary, P5 reliability-sensitivity outer results, F3 Round2, P8, new-domain validation and deployment end-to-end latency remain unavailable. New stratified/development–outer comparisons are descriptive, without new CIs; see the [supplement and machine-readable sources](<../../NIDSProject_Professor_History_20261004/docs/Professor_Progress_Report_After_P7_Followup_EN.md>).

**Decisions requested:**

1. Should the current paper focus on failures and boundaries of cross-domain reliability/selective prediction, while full P8 remains deferred?
2. Can a new independent domain and one bounded improvement round be resourced? If not, should P9 close the study with its negative result?
3. If continuing, should Strict-0 remain the requirement, or should few-label target adaptation become a separate question with an explicitly changed setting?
