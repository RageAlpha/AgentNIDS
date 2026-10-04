# Research Roadmap — Phase 2
## Multi-Detector Conformal Coordination for Cross-Domain NIDS under Distribution Shift

**Prepared for:** Siyuan Yu  
**Date:** September 2026  
**Status:** Approved direction — final version, do not revise further

---

## 1. What Changed and Why

### 1.1 What Phase 1 established

Phase 1 (Progress Report 1) produced four concrete, rigorously established findings.
These are not failures to be hidden — they are scientific results that motivate Phase 2.

| Finding | Evidence |
|---|---|
| Complementary errors exist across three heterogeneous detectors | G0 PASS on ToN (24.2 pp) and CIC (11.8 pp) using all-49-feature models |
| Feature partitioning imposes a severe base-detection cost | ToN accuracy 39–64% (partitioned) vs 83–85% (all-49) |
| Window-level JSD shift signal hurts reliability ranking, it does not help | G1 BELOW on aggregate (−0.055); adding shift worsens AURC on ToN by −0.167 |
| Trained reliability meta-learners do not generalize from proxy to held-out domains | Development risk 3.89% → outer risk 64.02% on ToN; second-query harm 37.72% > correction 19.40% |

Additionally: entropy-only f(u) consistently outperformed every more complex
CRE variant (f(h,u), f(h,u,s)) across AURC, Brier, and NLL. The simplest
reliability signal is the most transferable one.

### 1.2 What is retired and why

The following components are retired. Their data is documented as negative findings
in the paper — not discarded.

| Retired component | Reason |
|---|---|
| Feature-diverse 17/17/15 evidence partition | Catastrophic base-detection cost with no coordination benefit |
| Window-level JSD shift signal s(x) | Shown to hurt reliability ranking; G1 BELOW aggregate |
| Logistic CRE f(h,u,s) as trained meta-learner | Trained policy does not generalize across domains |
| Pair-specific corrective models g_{j\|i} | Meta-generalization failure; harm rate exceeds correction rate |
| Frozen development-selected thresholds | 3.89% → 64% transfer collapse is structurally unfixable without target feedback |

### 1.3 What is kept

| Kept component | Reason |
|---|---|
| All-49-feature RF, XGBoost, MLP detectors | Demonstrably better base detection; G0 complementarity preserved |
| LODO three-fold evaluation protocol | Correct cross-domain evaluation design; keep exactly as-is |
| G0 complementarity analysis | Still motivates multi-detector design over single detector |
| All comparison baselines | Single detectors, MoE, cascades, voting — all still needed |
| Negative findings from Phase 1 | Become Section 3 of the paper |

### 1.4 The core insight motivating Phase 2

Every component that failed in Phase 1 shared one property: it required a
**learned policy trained on proxy domains to generalize to a held-out domain**.
The CRE, the corrective models, and the frozen thresholds all depend on this
generalization. The cross-domain distribution gap is large enough to break it.

Every component that worked — entropy, base detector probabilities, complementarity —
required **no such generalization**. They operate on the model's own output at
inference time.

Phase 2 is built on this principle: **replace every learned policy with a mechanism
that derives reliability directly from the detectors' own outputs at inference
time, without requiring domain-specific training.**

That mechanism is conformal prediction.

---

## 2. The New Research Direction

### 2.1 Title (working)

> **ConFID: Conformal Multi-Detector Coordination for Reliable Cross-Domain
> Network Intrusion Detection**

*(Title is flexible — finalize during writing.)*

### 2.2 Core research question

> Can a multi-detector system that coordinates through conformal prediction set
> agreement provide reliable selective detection — with formal coverage properties —
> under cross-domain distribution shift, where trained coordination policies fail?

### 2.3 The central mechanism in plain terms

Each detector is augmented with a **conformal calibration step** using the source
calibration partition. Instead of producing a single class label, each detector
produces a **prediction set** — the set of classes whose nonconformity scores fall
within the calibrated threshold at a pre-specified coverage level.

A prediction set can be:
- `{Benign}` — the detector's nonconformity scores favor benign only
- `{Attack}` — the detector's nonconformity scores favor attack only
- `{Benign, Attack}` — both classes pass the threshold; the detector is uncertain
- `{}` — empty; no class passes the threshold — a **conformal rejection / OOD signal**

The coordinator makes decisions based on the **agreement structure** of the three
detectors' prediction sets. This rule is deterministic — it contains no learned
parameters and nothing that needs to generalize across domains.

**Important:** Standard split conformal prediction guarantees marginal coverage
only when calibration and test data are exchangeable (drawn from the same distribution).
Under cross-dataset shift this assumption is violated. Variant A (Strict-0, source
calibration) therefore provides no formal coverage guarantee on the target domain —
it is a heuristic baseline. Variant B (Strict-K, target Mondrian calibration)
provides the standard class-conditional conformal coverage guarantee under the
exchangeability assumption between the K target calibration flows and the evaluation
distribution. Because the K flows are the chronologically earliest available target
data and evaluation is performed on later flows, temporal drift within the target
domain can violate this assumption. Empirical coverage under Variant B is therefore
measured, not assumed — it is the quantity tested in H0.
The coordinated intersection of three prediction sets does **not** automatically
inherit the individual detectors' coverage guarantees; the coverage of the
coordinated system is an empirical question studied in E3.

### 2.4 Why conformal prediction is the right answer here

| Problem in Phase 1 | How conformal coordination addresses it |
|---|---|
| Trained CRE thresholds collapse under shift | Calibrated nonconformity thresholds replace learned thresholds; no policy to transfer |
| JSD shift signal uninformative | No shift signal needed; uncertainty is read directly from prediction set size |
| Meta-policy doesn't generalize | Coordination rule is deterministic — no learned policy, no generalization required |
| High confidence wrong predictions | Calibrated sets expand under uncertainty rather than maintaining false confidence |
| Unknown Attack definition was informal | Empty prediction set is a **testable** conformal rejection / OOD signal (H3) |
| Abstain definition was informal | Multi-label sets and inter-detector disagreement provide a principled Abstain condition |

### 2.5 Positioning relative to closest existing work

| Paper | What they do | Gap this work fills |
|---|---|---|
| CRISP (Moccardi et al. 2026) | Single model, conformal selective prediction | No multi-detector coordination; no cross-domain evaluation |
| ICDF (Wang et al. 2022) | Multi-detector, IVAP calibration, confidence-filtered voting | No conformal sets; no coverage guarantees; no cross-domain evaluation |
| Talpini et al. (2023) | Single model, UQ with Mahalanobis OOD | No coordination; no cross-domain transfer |
| AMoE-IDS (Mjahed et al. 2026) | Adaptive MoE gating | Trained policy; no coverage guarantees; no Unknown/Abstain |

**The gap:** No existing paper combines multi-detector coordination through
conformal set agreement with cross-domain evaluation and formal Unknown/Abstain
decisions. This is the contribution.

---

## 3. Technical Specification

### 3.1 Base detectors

Use the existing all-49-feature detectors from Phase 1.

| Detector | Model | Features | Training data |
|---|---|---|---|
| D1 | Random Forest | All 49 NF-v3 features | S_train (60% of UNSW-NB15-v3) |
| D2 | XGBoost | All 49 NF-v3 features | S_train (60% of UNSW-NB15-v3) |
| D3 | MLP | All 49 NF-v3 features | S_train (60% of UNSW-NB15-v3) |

No retraining is needed if Phase 1 all-49 models are already saved.
Verify that models were saved with all 49 features before proceeding.

### 3.2 Conformal calibration

Replace Platt scaling with **split conformal prediction** calibrated on S_prob_cal
(the existing 15% source calibration partition). Two variants:

**Variant A — Standard split conformal (Strict-0, no target labels):**
Nonconformity score for a flow x with true class y:

```
α(x, y) = 1 − p̂_y(x)
```

where p̂_y(x) is the detector's softmax probability for class y (raw, before
Platt scaling — Platt is no longer needed).

The conformal threshold for coverage level 1−ε is the ⌈(n+1)(1−ε)⌉/n quantile
of the calibration nonconformity scores, where n = |S_prob_cal|.

The prediction set for a test flow x is:

```
C(x) = { y : α(x, y) ≤ q̂ }
```

**Variant B — Mondrian (class-conditional) conformal (Strict-K):**
Compute separate thresholds per class using only calibration flows of that class.
This requires at least 200 calibration flows per class. Use target-domain
calibration flows (K labeled flows from the target, not from source).

```
q̂_y = quantile of { α(x_i, y_i) : y_i = y, i ∈ S_K_cal }
C(x) = { y : α(x, y) ≤ q̂_y }
```

**Primary condition:** Variant B with Strict-K (K = 400 labeled target flows total,
200 per class, used only for conformal calibration — nothing else is retrained).

**Coverage guarantee under Variant B:** The standard Mondrian conformal guarantee
holds under the exchangeability assumption between the K calibration flows and the
evaluation flows. Because the K flows are the chronologically earliest available
target data and evaluation windows are later, temporal drift within the target
domain may violate exchangeability. Treat per-detector empirical coverage under
Mondrian calibration as a quantity to be measured (verified in E0, tested in H0),
not as a guaranteed property.

**Secondary condition:** Variant A with Strict-0. Run this as a sensitivity
analysis to show what happens when no target labels are available.

**Coverage level:** Primary ε = 0.10 (90% marginal coverage).
Sensitivity: ε ∈ {0.05, 0.10, 0.20}.

**Important:** Conformal calibration is performed within each LODO fold.
For fold F1 (held-out: ToN), the K labeled flows come from ToN — but they
are used only for threshold calibration, not for any retraining or model selection.

### 3.3 The coordination decision rule

This rule is deterministic. It contains no learned parameters.
Pre-register this rule exactly as written below before running any evaluation.

Let C1(x), C2(x), C3(x) be the prediction sets of D1, D2, D3 for flow x.

**Step 1 — Compute the intersection:**

```
I(x) = C1(x) ∩ C2(x) ∩ C3(x)
```

**Step 2 — Apply the decision rule:**

Evaluate conditions in the priority order shown. Apply the first matching row.

| Priority | Condition | Decision | Rationale |
|---|---|---|---|
| 1 | \|{i : C_i(x) = {}}\| ≥ 2 | **Conformal Rejection (Unknown Attack candidate)** | Majority of detectors find no plausible class — strong OOD signal |
| 2 | I(x) = {} AND all C_i(x) non-empty | **Abstain** | Detectors actively disagree; intersection is empty |
| 3 | I(x) = {Benign, Attack} | **Abstain** | Collective uncertainty across all detectors |
| 4 | Exactly 1 C_i(x) = {}; other two agree on singleton s | **Abstain** | One detector rejects all classes but majority agree — treat as uncertain, not unknown |
| 5 | I(x) = {Benign} | **Accept: Benign** | All non-empty sets agree on benign |
| 6 | I(x) = {Attack} | **Accept: Known Attack** | All non-empty sets agree on attack |
| 7 | All other cases | **Abstain** | Default when no clear agreement |

**Critical design note — coverage does not compose:**
Each individual detector has a finite-sample marginal coverage guarantee at level
1−ε under calibration assumptions. The intersection of three prediction sets does
NOT automatically achieve 1−ε coverage. By a union bound, coverage of the
intersection may be as low as 1−3ε. The paper must not claim the coordinated
system achieves 90% coverage by virtue of using conformal prediction.
The risk-coverage behavior of the coordinated system is measured empirically in E3.

**Note on the Conformal Rejection signal:**
An empty prediction set (row 1 above: ≥2 detectors) means the calibrated nonconformity
scores for every known class exceed their thresholds. This is a testable OOD signal,
not a definitional Unknown Attack label. Whether it is useful for identifying
genuinely unseen attack families is the empirical question in H3 and the empty-set
analysis in E3. Do not call it "Unknown Attack" in the mechanism description —
call it "Conformal Rejection" until the empirical results establish its value.

### 3.4 The Strict-K target calibration budget

**K is defined as total labeled target flows, split equally across classes.**

The primary setting is **K = 400 total (200 per class: Benign and Attack).**

These flows are:
- Drawn from the held-out target domain (chronologically first available)
- Used only for Mondrian conformal threshold calibration
- Not used for: detector retraining, model selection, hyperparameter tuning,
  policy fitting, or any other purpose
- Excluded from the evaluation windows

**Calibration variant by K value:**

| K (total) | Per class | Calibration variant used |
|---|---|---|
| 0 | 0 | Variant A — standard split conformal on source S_prob_cal |
| 100 | ~50 | Variant A — too few for stable Mondrian quantiles |
| 200 | ~100 | Variant A — below minimum; use source calibration |
| 400 | ~200 | Variant B — Mondrian (primary setting) |
| 800 | ~400 | Variant B — Mondrian |

The minimum per-class count for stable Mondrian quantile estimation is 200.
For K < 400, fall back to Variant A (source-calibrated standard split conformal).
This fallback must be applied consistently across E6 — do not use Mondrian
with fewer than 200 samples per class.

**Operational justification:** A security analyst reviewing 400 alerts in a new
deployment environment is a realistic and conservative assumption. The K sensitivity
in E6 quantifies how performance changes as this budget varies — the central
practical finding is: how much target supervision is actually necessary?

**How K is selected:** K = 400 is fixed a priori. Do not select K by optimizing
held-out performance. K is pre-registered before any held-out evaluation.

### 3.5 What replaces the shift signal

Nothing replaces it directly. The shift signal is not used.

Prediction set size implicitly carries uncertainty information:
- A singleton set means the detector is confident
- A multi-label set means the detector is uncertain
- An empty set means no class satisfies the conformal criterion — the nonconformity score for every known class exceeds the calibrated threshold at the given coverage level

This is all the uncertainty information the coordinator needs.
No external shift signal is computed, stored, or used.

---

## 4. Experimental Protocol

### 4.1 Datasets — unchanged

| Dataset | Role |
|---|---|
| NF-UNSW-NB15-v3 | Source (detector training and source conformal calibration) |
| NF-CICIDS2018-v3 | Target (F3 held-out; F1, F2 development) |
| NF-BoT-IoT-v3 | Target (F2 held-out; F1, F3 development) |
| NF-ToN-IoT-v3 | Target (F1 held-out; F2, F3 development) |

IP addresses and absolute timestamps excluded from all detector inputs.
Timestamps retained for chronological ordering only.

### 4.2 Source partition — unchanged

| Partition | Proportion | Purpose |
|---|---|---|
| S_train | 60% | Detector fitting |
| S_prob_cal | 15% | Source conformal calibration (Variant A, Strict-0) |
| S_competence | 15% | Retained for ablation comparison (Phase 1 CRE baseline) |
| S_source_test | 10% | Source-domain sanity evaluation |

### 4.3 LODO folds — unchanged

| Fold | Development domains | Held-out domain |
|---|---|---|
| F1 | CIC + BoT | ToN |
| F2 | CIC + ToN | BoT |
| F3 | BoT + ToN | CIC |

### 4.4 Target calibration protocol (Strict-K)

Within each fold, the held-out target provides K = 400 labeled flows for
Mondrian conformal calibration only. These flows are the chronologically
earliest available from the held-out domain. They are never used for
model selection, threshold grid search, or any other optimization.

For the Strict-0 sensitivity: the source S_prob_cal partition is used for
standard split conformal calibration. No target flows are used.

### 4.5 Evaluation windows

Window size: 10,000 flows (unchanged from Phase 1).
Grouped bootstrap over windows: 2,000 bootstrap replicates, 10 windows per
block, minimum 20 blocks per domain.

### 4.6 BoT-IoT special handling

BoT has 99.70% attack share. Class-conditional Mondrian calibration requires
~200 benign flows — BoT may not provide enough benign flows for stable calibration.

**Protocol for BoT:**
1. Attempt Mondrian calibration with available benign flows.
2. If fewer than 100 benign flows are available in K, report BoT under
   Variant A (Strict-0, standard split conformal) only.
3. Retain BoT in the analysis — its behavior under extreme class imbalance
   is a finding worth reporting, not a reason to exclude it.

---

## 5. Experiments

### Experiment E0 — Source-domain sanity check

Verify that each detector's conformal prediction sets achieve the target
coverage (1−ε = 90%) on S_source_test. If coverage is substantially off,
investigate the calibration implementation before proceeding.

**Expected result:** Coverage ≈ 90% ± 2% for each detector on source test.

### Experiment E1 — Prediction set behaviour under shift

For each held-out target domain and each detector, report:
- Fraction of flows receiving singleton prediction sets (confident predictions)
- Fraction receiving multi-label sets (uncertain)
- Fraction receiving empty sets (Conformal Rejection signal)
- Actual marginal coverage under Strict-K calibration

This characterizes how prediction sets change under domain shift.
It is a diagnostic experiment — run it first before any coordination evaluation.

**What to look for:** Under shift, you expect more multi-label sets and more
empty sets than on the source domain. Verify this is true. If the fraction
of empty sets is near zero on all targets, the Unknown Attack signal will be weak.

### Experiment E2 — Single-detector conformal selective prediction (baseline)

For each detector individually, treat singleton prediction sets as accepted
predictions and multi-label or empty sets as abstentions. Evaluate:
- Coverage (fraction accepted)
- Selective risk (error rate among accepted)
- Risk-coverage curve (AURC)
- Empty-set rate (Unknown Attack signal strength)

This is the single-model conformal baseline. It is CRISP-equivalent.
Compare D1, D2, D3 individually.

### Experiment E3 — Multi-detector conformal coordination (main result)

Apply the coordination decision rule from Section 3.3.
Evaluate across all three LODO folds under both Strict-K and Strict-0:

**Primary metrics:**
- Coverage by decision type (Benign, Known Attack, Abstain, Unknown Attack)
- Selective risk among accepted predictions [95% CI grouped bootstrap]
- Risk-coverage curve (AURC) — compare to single-detector conformal baseline
- Unknown Attack rate on flows from held-out attack families

**Secondary metrics:**
- Mean detector calls per flow (coordination cost)
- Empty-set rate as Unknown Attack signal — precision and recall against
  held-out attack families (from the frozen open-set taxonomy in Phase 1)

**Comparison targets:**
Compare against all baselines from Phase 1 (retained) plus:
- Single-detector conformal (E2 above)
- ICDF-style confidence-filtered voting (IVAP replaced by conformal, majority vote)
- CRISP-equivalent single-model conformal

### Experiment E4 — Ablation of coordination rule components

Test the following rule variants to isolate the contribution of each component:

| Variant | What changes |
|---|---|
| Full rule (Section 3.3) | Baseline |
| Intersection-only, no empty-set handling | Remove Unknown Attack signal |
| Majority vote on singletons (ignore multi-label) | Replace intersection rule |
| Single detector with conformal (D1 only) | Remove coordination entirely |
| Phase 1 coordinator (negative result baseline) | Compare new vs old |

This ablation directly answers: what does multi-detector coordination add
over single-model conformal, and what does conformal calibration add over
the Phase 1 coordinator?

### Experiment E5 — Erroneous consensus under shift

This was planned in Phase 1 but not run. Execute it now.

Identify windows where all three detectors produce the same singleton prediction
(consensus). Partition by shift level (use the Phase 1 JSD signal here —
not as a reliability predictor, but as a diagnostic stratification variable).

Compare:

- Error rate among consensus flows in high-shift windows
- Error rate among consensus flows in low-shift windows

Report ECAR (Erroneous Consensus Acceptance Rate) and CCPR (Correct Consensus
Preservation Rate) under the full coordination rule.

Under conformal coordination, erroneous consensus requires all three detectors
to be simultaneously confidently wrong — meaning all three produce singleton
sets that agree on the wrong class. Under shift, conformal sets should widen,
reducing this rate. Verify empirically.

### Experiment E6 — Strict-K label budget sensitivity

Vary K ∈ {0, 400, 800} and report how coverage, selective risk, and
conformal rejection rate change. K is total labeled target flows.

Calibration variant by K value (consistent with Section 3.4):
- K = 0: Variant A (standard split conformal on source S_prob_cal)
- K = 400 and K = 800: Variant B (Mondrian on target flows)

Report for each K: selective risk, coverage, empty-set rate, and AURC.
The central practical finding is whether Mondrian calibration (K = 400) improves
over source-calibrated conformal (K = 0), and whether doubling the budget
(K = 800) provides additional gains over K = 400 — together these test H4.
This tells a deploying organization the minimum labeled calibration budget needed.

**Rationale for K ∈ {0, 400, 800}:** K = 100 and K = 200 both fall below the
200-per-class minimum for stable Mondrian quantile estimation and therefore use
Variant A (source calibration), identical to K = 0. Including them would add three
data points that test nothing different — the method does not change. The meaningful
transition is at K = 400 (first Mondrian-calibrated condition); K = 800 tests
whether doubling the budget yields diminishing returns.

---

## 6. Comparison Baselines

Run all Phase 1 baselines unchanged (single detectors, fixed cascades, learned
cascade, soft/hard MoE). Add the following new baselines:

| Baseline | Description | Purpose |
|---|---|---|
| Best single detector (conformal) | D_best with split conformal, singleton = accept | CRISP-equivalent; isolates coordination contribution |
| Confidence-filtered voting | Majority vote among detectors above confidence threshold | ICDF-equivalent; isolates conformal contribution |
| Phase 1 coordinator (frozen) | From Phase 1 with development thresholds | Direct negative result comparison |
| Oracle coordinator | Uses true labels to choose best detector per flow | Upper bound on coordination gain |

All comparisons are budget-matched: report mean detector calls per flow and
normalized cost alongside every result.

---

## 7. Pre-Registration Checklist

Complete all items in this checklist **before running any held-out evaluation**.
Record each item with a timestamp. The pilot (Week 0) must pass before this
checklist is relevant.

**Phase: Pilot gate**
- [ ] Pilot executed on RF + ToN under Strict-0
- [ ] Pilot go criterion met: target shows meaningfully higher multi-label/empty-set rate than source, with errors concentrated in uncertain/rejected cases
- [ ] Professor informed of pilot result before proceeding

**Phase: Pre-registration (complete before touching held-out data)**
- [ ] Nonconformity score formula `α(x,y) = 1 − p̂_y(x)` written and code reviewed
- [ ] Coverage level ε = 0.10 fixed; sensitivity values {0.05, 0.10, 0.20} fixed
- [ ] Coordination decision rule (Section 3.3, priority table) written exactly as above; code reviewed
- [ ] K definition confirmed: K = total flows; Mondrian only for K ≥ 400 (≥200 per class); Variant A for K < 400
- [ ] Primary K = 400 (200 per class); Strict-0 (K = 0) as secondary; K sensitivity at {0, 400, 800}
- [ ] Variant fallback rule confirmed in code: K < 400 automatically uses source S_prob_cal calibration
- [ ] BoT fallback confirmed: if benign count in K < 100, use Variant A only — verify benign count in data before running
- [ ] Minimum effect sizes for E3 primary metrics defined and written down
- [ ] ECAR threshold (HighAgreement = all 3 singleton, same class) and HighShift threshold (JSD above 75th percentile of development distribution) written down for E5
- [ ] Held-out target excluded from its own fold's calibration — verified in code with a data-path audit
- [ ] Open-set attack family taxonomy from Phase 1 (Availability Flooding / Reconnaissance) confirmed unchanged
- [ ] Individual coverage guarantee claim is scoped to per-detector, not to the coordinated intersection — confirmed in paper draft language before submission

---

## 8. Paper Structure

The paper tells a complete story: Phase 1 established what fails and why;
Phase 2 establishes what works and why it works.

### Section 1 — Introduction
- Cross-domain NIDS reliability is a real operational problem
- Existing multi-agent and multi-detector approaches assume learned policies generalize
- We show this assumption fails, formally and empirically
- We propose conformal coordination as the theoretically grounded alternative

### Section 2 — Related Work
- Use the Phase 1 literature review table directly
- Position against CRISP, ICDF, AMoE, Talpini
- The six-condition gap from the literature review still holds — update it to include
  the conformal coordination condition

### Section 3 — Why Trained Meta-Learners Fail: Evidence from Phase 1
This section is the negative contribution. It is rigorous and precise.
- Phase 1 experimental setup (brief — reader does not need all protocol details)
- G0 result: complementarity exists (positive)
- G1 result: shift signal hurts ranking — why (uniform OOD, no differential signal)
- CRE meta-learner: development vs outer risk (3.89% → 64%) — why (policy generalization gap)
- Second-query harm > correction — why (corrective models fit to proxy domains)
- Conclusion: any mechanism requiring learned policy transfer across domains will fail
  under extreme cross-dataset shift

### Section 4 — Conformal Coordination: Framework and Theory
- Background: split conformal and Mondrian conformal prediction (brief)
- The three-detector prediction set mechanism
- The coordination decision rule (Section 3.3 of this roadmap)
- Coverage guarantee under Mondrian calibration (Strict-K)
- Why erroneous consensus is structurally harder under conformal coordination
- Why Unknown Attack and Abstain emerge naturally from the framework

### Section 5 — Experimental Setup
- Datasets, LODO protocol, Strict-K and Strict-0 conditions
- Baselines
- Metrics
- Pre-registration statement

### Section 6 — Results
- E1: Prediction set behavior under shift (diagnostic)
- E2: Single-detector conformal (baseline)
- E3: Multi-detector conformal coordination (main result)
- E4: Ablation
- E5: Erroneous consensus
- E6: Strict-K sensitivity

### Section 7 — Discussion
- What the minimum K budget means operationally
- Where conformal coordination fails (BoT edge case, extreme class imbalance)
- Limitations: exchangeability assumption, computational cost
- Connection to the original multi-agent vision: what was gained, what was simplified

### Section 8 — Conclusion

---

## 9. Timeline

### Week 0 — Feasibility Pilot (1–2 days, go/no-go gate)

Before any full implementation, run this cheap check on **one detector, one target domain**:

1. Load the existing all-49 RF model from Phase 1.
2. Compute split conformal calibration on S_prob_cal (Variant A, Strict-0).
3. Run prediction sets on NF-ToN-IoT-v3 evaluation windows.
4. Report: singleton rate, multi-label rate, empty-set rate, source coverage vs target coverage.

**Go criterion:** The target domain must show meaningfully more multi-label and/or
empty sets than the source domain. Errors must be concentrated in the uncertain/rejected
cases rather than distributed uniformly. If this differentiation is absent —
if the target produces 99%+ singleton sets with similar error distribution to source —
the conformal mechanism is not providing useful uncertainty signal and the direction
must be reconsidered before committing further time.

**No-go action:** If the pilot fails, report findings to professor before proceeding.
Do not proceed to Week 1 without pilot confirmation.

---

### Main implementation and evaluation

| Week | Work |
|---|---|
| 1 | Implement split conformal calibration for all three detectors. Verify source-domain coverage (E0). Implement Mondrian calibration and Strict-K procedure. |
| 2 | Run E1 (prediction set diagnostic) on all three held-out targets under both Strict-0 and Strict-K. Confirm H0. |
| 3 | Implement coordination decision rule (Section 3.3 exactly as written). Complete pre-registration checklist. Run E2 (single-detector conformal baseline). |
| 4 | Run E3 (main coordination result) under Strict-K (primary) and Strict-0 (secondary). Collect metrics and confidence intervals. |
| 5 | Run E4 (ablation), E5 (erroneous consensus), E6 (K sensitivity at {0, 400, 800}). |
| 6 | Full analysis: where coordination helps vs hurts, BoT behavior, conformal rejection signal quality, H2 and H3 verdict. |
| 7–8 | Write Sections 3 (Phase 1 negative findings) and 4 (conformal coordination method). |
| 9–10 | Write Sections 5 and 6. Generate all figures and tables. |
| 11–12 | Write Sections 1, 2, 7, 8. Internal revision. Submit draft to professor. |

**Total: approximately 12 weeks to submission-ready draft, plus 1–2 days for the pilot.**

---

## 10. What to Do If Results Are Negative Again

The fail-informative structure is retained. If the coordination fails under
conformal calibration as well, the response is determined by which component fails.

**If E2 (single-detector conformal) already fails to provide useful low-risk coverage:**
The problem is the base detectors — not coordination or calibration. In this case:
stop here, write a paper that precisely documents what level of base detector
performance is necessary for any selective prediction system to function under
cross-domain shift. This is a scientifically valid and publishable boundary-finding result.

**If E2 succeeds but E3 does not improve over E2:**
Multi-detector coordination adds no value over single-model conformal.
The contribution becomes: CRISP-equivalent single-model conformal works
under cross-domain shift; multi-detector coordination does not additionally help.
Document why (prediction set disagreement may be too rare to provide coverage gain).
This is also publishable, as it definitively answers whether coordination is necessary.

**If E3 succeeds on ToN and CIC but BoT fails:**
Report ToN and CIC as primary results. Treat BoT as a case study in the limits
of conformal coordination under extreme class imbalance. Document the minimum
benign prevalence needed for reliable calibration. This is a practical finding.

**In no case should results be adjusted post-hoc to appear more positive than they are.**
The paper's credibility depends on honest reporting. A clean negative result with
rigorous methodology is publishable. A manipulated positive result is not worth
submitting.

---

## 11. Key Scientific Claims to Establish (Pre-Stated Hypotheses)

State these before running held-out evaluation. They are tested, not post-hoc rationalized.

**H0 — Calibration validity (gate hypothesis):**
Under Strict-K (K = 400), each individual detector's Mondrian conformal prediction
sets achieve empirical class-conditional coverage within 5 percentage points of the
nominal level (90%) on at least two of three held-out targets.
If H0 fails, the calibration implementation must be inspected before proceeding.
The coordinated system's coverage is not asserted — only the individual detectors'.

**H1 — Coordination improves on the Phase 1 trained coordinator:**
Multi-detector conformal coordination achieves lower selective risk at equal
coverage than the Phase 1 trained coordinator on ToN and CIC.
This is a low bar — Phase 1 coordinator had 64% outer risk. It establishes that
conformal coordination does not make things worse.

**H2 — Coordination improves on single-detector conformal:**
Multi-detector conformal coordination achieves lower selective risk at equal
empirical coverage than the best single-detector conformal baseline (E2)
on at least two of three held-out targets.
This is the critical hypothesis. If H2 fails, the multi-detector coordination
claim fails. Single-model conformal (CRISP-equivalent) would be sufficient.
Be prepared for this outcome and document it honestly.

**H3 — Conformal rejection signal identifies unseen attack families:**
Among flows receiving a Conformal Rejection decision (≥2 detectors with empty
prediction sets), the prevalence of attack families absent from source training
is significantly higher than among flows receiving an Accept decision.
This is empirically tested, not definitionally assumed.
If H3 fails, the conformal rejection signal has no Unknown Attack value.
Report this and remove the Unknown Attack claim from the paper.

**H4 — Label efficiency:**
Mondrian calibration (K = 400) achieves lower selective risk at equal empirical
coverage than source-calibrated conformal (K = 0), with diminishing returns when
doubling the budget to K = 800. The magnitude of the K = 0 → K = 400 improvement
provides a practical deployment guideline for the minimum labeled calibration
budget needed at a new site.

H2 is the central hypothesis. H0 is the gate. H3 and H4 are secondary contributions.

---

## 12. What Not to Do

- **Do not reintroduce the window-level JSD shift signal.** The evidence against
  it is clear and reproducible. It is retired.

- **Do not reintroduce the learned corrective models g_{j\|i}.** The meta-generalization
  failure is the core negative finding. Reintroducing learned policies would
  repeat the Phase 1 failure.

- **Do not optimize K on the held-out target.** K = 400 is pre-registered. Do not
  increase it after seeing results. Report the K sensitivity separately as E6.

- **Do not exclude BoT from the paper** because it is difficult. Treat its behavior
  as a finding — extreme class imbalance imposes a floor on what conformal calibration
  can achieve. Document this limit clearly.

- **Do not add open-set experiments (O1–O4 from Phase 1) until E3 is complete
  and shows meaningful results.** The Unknown Attack signal from empty prediction
  sets is already available from E1. Use that first. Full open-set evaluation
  with held-out attack families can be added in the final analysis pass if
  E3 results are strong enough to justify it.

- **Do not change the coordination decision rule mid-experiment** based on
  intermediate results. The rule in Section 3.3 is pre-registered. If it
  underperforms, document that and analyze why — do not adjust and re-run.

---

## 13. Resources and References

**Conformal prediction — foundational:**
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.*
- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: a gentle introduction. *Foundations and Trends in Machine Learning.*

**Adaptive conformal inference under shift:**
- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021.*

**Mondrian conformal prediction:**
- Venn predictors / class-conditional conformal — see Angelopoulos & Bates survey above.

**Closest related NIDS work (directly relevant for positioning):**
- Moccardi et al. (2026). CRISP: Conformal Resilient Intrusion Detection with Selective Prediction.
- Wang et al. (2022). ICDF: Intrusion Collaborative Detection Framework Based on Confidence.
- Talpini et al. (2023). Enhancing Trustworthiness in ML-Based NIDS with UQ.

**Implementation:**
- `nonconformist` (Python library for conformal prediction)
- `MAPIE` (Model Agnostic Prediction Interval Estimator — supports classification conformal)
- Both are compatible with scikit-learn models (RF, XGBoost wrapper, MLP)

---

*This roadmap supersedes the Revised Technical Proposal for Phase 2 implementation.
The Phase 1 protocol, results, and negative findings remain valid and are incorporated
into the paper as Section 3.*
