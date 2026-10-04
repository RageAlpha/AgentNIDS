from pathlib import Path
import json,csv
O=Path(__file__).resolve().parent
v=json.loads((O/'audit_validation.json').read_text()); s=json.loads((O/'sample_coverage_verification.json').read_text()); r=json.loads((O/'roadmap_version_comparison.json').read_text())
text=f'''# Week 0 roadmap Conformity with upstream source audit

## Conclusions and Scope

** Overall Conclusion: B. has not found any substantive problems, but there is a clear evidence gap.**

No implementation or upstream data errors have been found to alter the current Week 0 results within the scope of the read execution protocol, code, freeze assets and independent audit coverage. The explicit Week 0 requirement of roadmap at execution is executed; Distinguish phrases such as RF softmax probability and underspecified settings as explained below. The original Reply4 has not been found and cannot be fully historical source verified. External disclosure statements and confirmation of specific versions by professors have not been found outside the project.** The current NO_GO_EVIDENCE is still supported, with only rf_All49 / ToN / Variant A / Strict-0 / epsilon=1/10 / having both freeze and q.**

This time, only read historical material, calculate file hashes and verify that the audit record has already been verified; There is no re-training, adjustment, re-reasoning, re-calculation of threshold selection or re-run A or C2. Week1 hasn’t started, no message has been sent to the professor. There were no anomalies in the numerical value that needed to be isolated to repair the /. The next step is to supplement the research decisions and the necessary original documentation without continuing to expand the calculation due to negative results.

## 1. Source, version and strength of evidence

- is the official pilot: `/home/karri/projects/NIDSProject_Professor_Run/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z`.
- preflight:`/home/karri/projects/NIDSProject_Professor_Run/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z`.
- independent audit: `/home/karri/projects/NIDSProject_Professor_Run/outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z`53,813 reconsideration, bit mismatch=0 set change=0 identification, 2026-09-15T12:11:31.272366+00:00 completion time based on the report and actual verification records; It is not just a directory name order.
- Current roadmap SHA256 : `{r['current_sha256']}` is consistent with the preflight, pilot, independent, audit registration values. Therefore, no differences between the current version of Week0 and the execution version have been found, and subsequent revisions need not be traced back to a violation.
- read and calculate byte byte this time {v['current_hashed_assets']} SHA256 of the source document; The historical manifest {v['frozen_hash_comparisons']} This is not the same as a comparison, there is no difference. path, original version fields, mtime and SHA256 see `source_hashes.csv`; Time stamps are only used as an aid and do not bear any proof of source.
- is the original Reply4 **MISSING**.`docs/StrictK_Reply4_Experiment_Plan.md` states that its own numerical value is not the original requirement of Reply4 and cannot replace the original response.
- Independent Windows A–C2 summary is **MISSING/not synchronized**;expected `/mnt/c/Users/ysy/Desktop/literature/StrictK_A_B1_B2_C1_C2_Integrated_Summary_20260915.md`. Stage code/protocols/validation/reports were read directly. The summary is background,not a computational blocker.
- Roadmap Approved direction is document status,not an independent record of professor approval of that SHA256 version. Approved version,pilot-report receipt and confirmation are **NOT_VERIFIED**. User authorization is not professor confirmation.

Evidence layering: ** raw product direct verification ** for current bytes, identity manifest, freeze q, sample tables and indicators;** execution log ** is used for the scope and time of execution;** static code ** is used for data flow and writing paths;** post-reported ** only as an auxiliary explanation; In the absence of the original record, ** cannot be verified as **. The current consensus is that hash cannot prove the complete history of the production era, and static code cannot prove all unrecorded human behavior.

## 2. roadmap Originally verified in stages

Complete extract by item, line number, protocol / code location and handling comments see `roadmap_compliance_matrix.csv`, total {sum(v['roadmap_matrix_classifications'].values())} Classified statistics: `{v['roadmap_matrix_classifications']}`. No DEVIATION found to conflict with the explicit Week0 requirement; This does not mean that all historical requirements have been externally certified.

| Facts | Determine the source of | and the boundary of |
|---|---|---|
| RF-All49 ToN S_prob_cal Variant A / Strict-0 | EXPLICIT_MATCH | roadmap § 9 clearThe same freeze RF, unread target 400 labels calibration |
| Raw without using Platt | EXPLICIT_MATCH | § 3.2 to specify before Platt; Read raw_probability |
| RF softmax probability | AMBIGUOUS | RF No further softmax layers are required; The user specifies predict_proba[:,1] without additional softmax |
| p_Benign=1-p  float32-origin/float64 calculation | IMPLEMENTATION_DETAIL | has audited user numeric value contracts; The roadmap does not promise bitwise complementarity of the two native RF columns |
| 60% /15% source partition | IMPLEMENTATION_DETAIL | roadmap unchanged; Succeeding the original 10k grouping takes the exact intervals, without re-interpolating the | ratio.
| epsilon=.10,score=1-p_y,≤ | EXPLICIT_MATCH | §3.2; Restricted sample integer sequence, uninterrupted value, isolated whole group, hex preserved for refinement of |
| full source tail, ToN full 2752 window, no startup excluding | IMPLEMENTATION_DETAIL | original P7 window and user-defined range; Can’t inherit C1/C2 decrease 100000 |
| 2000 times /10 window block | EXPLICIT_MATCH | § 4.5; seed, linear CI, NA and the shared matrix for finishing |.
| source only 3 blocks are not CI, category and time descriptive | IMPLEMENTATION_DETAIL | does not meet at least 20 blocks; According to user-defined statistics, no block length shortening or removal of |
| coverage name | AMBIGUOUS | roadmap E2 uses fraction accepted, while § 9 requires source/target labels coverage; This wheel is named after each other, with no mixed |
| combined with go/no-go | EXPLICIT_MATCH | more empty only partly satisfies the direction and is also useful for misunderstanding; Minimum effect threshold |
| 99%+ singleton , 90%±2% , C2 5%/10% | is not the only gate | examples of / background checks, the latter of which are not Week0; High coverage does not automatically achieve failure |
| Mondrian 200 per class, three-model coordination, later checklist | Not applicable to Week0 | Section 7 explicitly puts the pilot before the later checklist; Existing 400 samples are insufficient for each 200 class and do not impede Strict-0 |.
| Reporting and confirmation | NOT_VERIFIED / stopped | Report files have been generated; Send / confirmation cannot be launched from the report’s presence without automatically entering Week1 |

The GPU constraint of the historical implementation protocol must be separated from the NumPy float64 statistical realization authorized in this round: the original RF training / sampling reasoning remains in the original cuML environment; This round of hashing and statistical evidence checks were completed at CPU. The refinement of the audited implementation should not be described as a stealthy regression or a roadmap that explicitly specifies a database of values.

## 3. Dependence Chain

```mermaid
flowchart TD
 A[ primitive UNSW / ToN CSV] --> B[ stable availability sequence with original_row]
 B --> C[ freeze partition and original window ]
 A --> D[ in the original 49 column generates feature store]
 B --> D
 C --> E[S_train Fill in the ]
 D --> E
 E --> F[ Fixed configuration training RF-All49]
 F --> G[P3 raw_probability Caching ]
 D --> G
 E --> G
 G --> H[S_prob_cal and freeze q]
 H --> I[ source test and ToN set and window count ]
 G --> I
 C --> I
 I --> J[ToN bootstrap and joint gate]
 G --> K[A–C2 historical diagnostic branch]
 K -. phase A and range Raw calculation .-> I
```

`asset_lineage.csv` gives the code on each side, manifest, output, hash connection and coverage limits. B2 f(u) , C1 classification threshold , C2 acceptance threshold does not pass through the q or prediction set.

### data identity, features and training

1. is the original CSV in the stable order of end/availability, and original_row is the CSV zero-base line identity, with no time loop to connect the repeating keys. independent audit Full quantity verification 29,885,684 sequence / replacement identity, source calibration, source test and ToN combined with the original 28,115,424 sequence no difference Label/Attack.** raw products directly verify + independent audit perform ** proof.
2. source training `[0,1420000)` calibration `[1420000,1780000)` competence `[1780000,2130000)` `[2130000,2365424)` testTraining / calibration / Test the original line identity without intersection. identity differs from no different content, no shared host or statistical independent; There are no confirmed leaks, and there is no full body to test the evidence.** is the primary identity product; Depending on independent sex NOT_VERIFIED**.
3. 49 columns in the fixed order of feature_views and manifest; Unmixed Label, Attack, identity line ID, IP address or absolute sequencing time (the DNS_QUERY_ID in the original 49 column is already a protocol field, not CSV line ID). CSV float64, non-finite values converted NaN, store float32; Fixed S_train values median/ category mode filled in (ties minus), RF non-standardised and no cross-domain unit conversion. The 49 refill is consistent, and each sample element is consistent.** static code + raw product / sampling performs proof of **.
4. `src/nids/p2.py` /`models/base_models.py` to RF with S_train X/y fit only; Fixed 300 trees, depth16, leaf2, sqrt, n_bins128, bootstrap, seed20260905, and RF have no eval_set or current automatic pattern selection cycles. After saving the model, the calibration prediction and the fit historical Platt are generated.** static code + training_manifest + model entity **; independent original training stdout has not been found, and the history of human selection models has not been verified.
5. model SHA256 is `456d3341dfd464d2251b80d7d2a0d4453fe7cf22ad7b449b42933b985b2cb40e` and is connected to current, preflight, pilot, audit records and StrictK model checkpoints. Source calibration history for Platt does not automatically render Raw insoluble; No record of S_prob_cal being found for the current Raw RF fit / adjustment cannot be expressed as proof that it has never been tested in humans.
6. P3 is predicted in successive 131072 batches and the original availability sequence, with `predict_proba[:,1]` remaining intact in float64 and Platt other fields, which are excluded from the evaluation range after the original end paragraph is written. Lineless caching ID is the source of risk: it must be based on the freeze position mapping, independent original CSV labels and sample reasoning proof, and cannot be proven with the same caching proof.

### historical selection independent gender gap

`configs/execution_v2.json:388–389` maintains `external_history=NOT_PROVIDED_IN_HANDOFF` and `USER_DECLARATION_REQUIRED_BEFORE_P7`. This targeted search did not find a separate statement;`docs/Professor_Progress_Report_After_P7.md:148` also clearly records the same gap.** is a gap in historical research with clear documentary evidence of effective sexuality, not a confirmation of violation of the Selective Model Prohibition, nor a detection of numerical contamination.**

P7, A C2 target results have been observed, and RF+ToN’s Week0 is already explored on the unblinding data. This fact has been stated in this round of agreements. It is not possible to introduce new external independent confirmation from the negative results of the same data or its intervals, nor can it be asserted that all historical models chose independent due to the absence of a non-conforming log.

## 4. A–C2 implications

For more details see `historical_stage_impact.csv`.

- **A**: generates its own aggregation and model audit, supplementing two MLP source reasoning to explain historical numerical differences; It’s hard to say that no new product has been created. Week0 reads the same range as A and calculates RF Raw as a consistency check without using its results to generate a q/ probability.
- **B1**: generate entropy/u scores and diagnoses; Week0 does not read its scores as a set calculation.
- **B2**: indeed fit 18 is a f(u) asset, outputing its own models directory located in B2; It’s not a change in freeze to RF. The wheel has no f(u).
- **Strict-K preflight / C1 / C2**: create 400 labels calibration manifest and classify / acceptance threshold; The scope of reduction used is different from the current one. This wheel does not read these thresholds, does not use its bootstrap reduction, and does not inherit 5% risk /10% acceptance rate gate.

**A class changes the amount of asset pollution: not detected. B historical information use: The target has been confirmed by unblinding and the selection history is unknown. Historical problems with no direct numerical dependence on C class: MLP differences, f(u) fit, C1/C2 ranges and risk settings do not constitute the RF Week0 overrun justification.**

It’s based on the actual read-write code, model checkpoints, manifest and hashes, not just the file time. B1/B2 early protection mainly protects size/mtime, C1/C2 also contains small files SHA and cannot be extended to guarantee the full content chain of large caches from the original production. All current registered hashes are consistent in providing strong evidence of later checkpoints that cannot be replaced by non-existent earlier hashes.

What does ## 5. 53,813 independent audit prove?

This time, re-read the freeze sample and step-by-step reasoning for CSV, verify with the `(domain,rank,original_row)` set, and compare independent to save hex as a double-precision bit sequence. There is no need to rethink GPU again.

The - manifest hash corresponds to the pre-reasoning sample_freeze and the freeze is earlier than the actual reasoning. The only identity is 53,813.
- UNSW 5,544,ToN 48,269; Each full source test 23 window and ToN2752 window have 17 specifications for rank and the end 5424 source tail also extracts 17; The other source calibration is 3601 homogeneous.
- ToN Number of samples for four periods: `{s['ToN_four_segment_sample_counts']}`; The original production batch coverage and indexation see `sample_coverage_verification.json`, rather than using a single audit batch reasoning to mistakenly describe the history of complete reproduction of each batch.
- all 731 items ToN reception records are included; It also contains near-thresholds, source Attack covering / rejection, original 131072 production batch boundary-directed samples. Targeted samples are not independent performance estimates.
- CSV reconstructed 49 features: 53,813×49=2,636,837 element; The feature store value and the actual filling of the RF inputs are the same. This conclusion is drawn from the completed / log and the comparative evidence for each column of the reconstructed code; This time, CSV features was not reproduced.
- freeze RF cuML26.8.0 , classes=[0,1] , predict_proba actual execution;[:,1] is consistent, maximum error 0, and total change 0. The code uses bit comparisons rather than permissions, and the reasoning table hex is also 0 for this comparison.
- is the natural [:,0] and 1-[:,1] at a maximum of about 1.42e-7; Two sets and not total phases are equal to 1, which is a recorded diagnosis without changing the protocol. The user continues to use 1-p according to the freeze contract, and cannot silently replace the native Benign column.

Full downstream replication covers 360000 calibration, 235424 source testing, 27520000ToN and 444160 with no difference between the voltage units and the 282006 CSV fields. The old pilot does not have a step-by-step collection file, the previous audit is a step-by-step independent re-expression of the old formula and verification of the / category space in the saved window; I can’t tell you the difference between the old trace that doesn’t exist.

** boundary: total labels identity, all of which is cached until collective repositioning has been verified; features reconstruct and model rethink only sampling 53813 points; Other features elements, full cache reconsideration, unrecorded human history of NOT_VERIFIED.**

## 6. q, extreme rejection and mechanism explained

freeze q=`3.145372886592668e-07` , hex=`0x1.51bb5a0000000p-22`Source calibration n=360000 , integer k=324001 , less than / is equal to / and higher than q respectively 287840/38470/33690; The SHA factor corresponds to the independent statistical revision. calibration covers 90.641667% , Benign96.417076% and Attack35.976744% as a whole. In general, it is not possible to cover up the differences between classes.

The maximum probability of a Raw attack on ToN is 0.989444375038147, and the minimum candidate Attack is non-conformity score `1-p_max=0.010555624961853027`, which is strictly greater than q. Therefore, ** currently observes that ToN has no record of meeting the Attack input condition **, which is a formula that is directly derived from the verified probability range, rather than the bootstrap inference. multi-label is 0 when q<0.5 is structural and cannot be considered a failure.

| indicator | source test | ToN |
|---|---:|---:|
| total recorded | 235424 | 27520000 |
| Acceptance Rate | 91.309297% | 0.00265625% (731 article) |
| rejection rate | 8.690703% | 99.99734375% |
| True labels prediction set covers | 91.309297% | 0.000417878% (115 items) |
| receives TN / FP / FN / TP | 207907 / 0 / 0 / 7057 | 115 / 0 / 616 / 0 |
| receives selective risk | 0 | 84.268126% |

In the source Attack 20017 clause, Raw correctly identifies the 20016 clause, but with 12959 clause Raw correct Attack is rejected.`p>=0.5` is only a Raw judgment, far from the extreme Attack single-element threshold of reaching `1-p<=q`. Refusing is not the same as finding out.

ToN refuses to record Raw with an error rate of about 15.9252% and accepts a record of 84.2681%; The former is subtracted from the latter by about -68.3429 percentage points and the 95% conditional block by [-72.4442,-63.7533] percentage points. The error rate decreases the correct rejection rate by about -0.013557 percentage points and the range is negative. Denying almost all mistakes at the same time as losing almost all the correct records, the mechanism of declaring almost all mistakes cannot be used to capture almost all the mistakes. The total Raw error rate is about 15.9270%, which is different from accepting risk or real labels coverage.

scanning has a 1,358,977 point, and the 3 point is mistaken for Benign, and effective Attack detects 0. The four sections accept 20/265/1/445 and TP as 0; FN is 0/213/1/402. The first 20 clause accepts Benign to lead to risk=0 and does not constitute a solid stage of success. Subsequent phases selective risk highlight changes, limited acceptance of support and time-switched interpretation.

** directly observes the fact that **: source calibration is Benign dominant, the probability score is highly discrete and there is a large ties, source Attack covers much less than the total; The ToN probability range cannot reach the Attack singleton conditions.** reasonably describes **: freeze source thresholds and target probability distributions do not match; overall calibration cannot provide a class-by-class assurance of cross-domain.** unconfirmed causal **: exactly which features, training bias or domain factors led to extreme probability and transfer failure; It is not possible to declare causal as defined by a category imbalance or domain deviation formula.

## 7. gate and statistical boundaries

roadmap § 9 at the same time requires more multi/empty targets and errors focused on uncertain records. Although there are currently very strong empty variations, there are no available errors in rejecting the distinction; effective Attack is detected as 0 and the receptors are mainly leaked. NO_GO_EVIDENCE relies on these co-directional evidence, adding no subsequent minimum effect numbers, and requiring only 99%+singleton to produce no-go. Negative results limited the current freeze configuration, not denying all conservation methods, subsequent target calibration or the entire direction of the study; Don’t try that this time.

2000 ToN share duplicates, 276 blocks, 2 windows, NA and linear percentile are all reproduced by independent statistically by audit. This time, it verified its code with existing hashing evidence, without duplicating the matrix. The source is only the 3 block and retains the tail, not the source CI; The objective of the refusal is only descriptive. 4 and taxonomy are the same description.

The range is conditioned by the current model, freeze q and the observed target sequence, with no weighted draw of calibration uncertainty. The time dependence, instability, and sufficient 10 window length has not been demonstrated. No future domain covering / risk warranty is provided. The probability of detecting [0,0] in the TP range is not future but 0. The 731 rule accepts samples with little restriction on the accuracy of admission to the performance, but the entire 2752 window almost completely rejects the observation that effective Attack detects as 0 is still meaningful configuration-grade negative evidence.

Old numerical_contract.md wording that real q was uncomputed describes preflight. Formal q_freeze/evaluation logs show subsequent execution. The independent audit saved documentation_errata.md;old files are unchanged,and preliminary wording does not prove nonexecution.

## 8. Six final answers and decision to stop

1. **Does execution follow the applicable roadmap and frozen protocol?** Verified substantive Week0 execution is consistent; Clear requests, user authorization to fine-tune and original misunderstandings have been split. The original version of the professor is not validated.
2. ** detects an upstream error in the code / that changes the result? ** not detected; The full quantity of identity labels with downstream weighting and the original features / reasoning sample of 53813 have been tested. It can’t be extended to the full proof of re-thinking.
3. ** Is there a effective / selection problem being studied? ** is a known target unblinding and the protocol has declared exploration; There is a clear evidence gap between original external exposure statements and human-selected models. There is no proof of violation of the prohibition on selection rules and no claim for independent confirmation results.
4. What ** is confirmed, sampled, unknown? ** See sections 5 and `unresolved_items.csv`; Primary product / log / static code / Post-reporting is not confused with unverifiability.
5. **NO_GO is still in place? ** is, limiting this configuration and range of observations; Thresholds and probability ranges result in no Attack input, refusal to lose almost all correct records and higher acceptance of error rates, without the useful distinction that roadmap expects.
6. ** Expanded check / isolation run? ** does not require numerical run. A limited range of searches, results, and stop conditions has been made available due to explicit external exposure. See `expanded_checks.json`. Supplementing the original documentation can enhance historical authentication; There is no exception to the non-complete re-run of A by the C2 and no modification by the gate.

## Documents and Protection

The main documents: `roadmap_compliance_matrix.csv`, `asset_lineage.csv`, `historical_stage_impact.csv`, `unresolved_items.csv`, `audit_validation.json`. Supporting evidence: version comparisons, sample coverage, mechanism algorithms, code-line extracts, full source SHA, differential tables, environment, commands, logs, and before and after manifest.

size/mtime manifest is unchanged before and after this upstream;{v['end_rehashed_small_asset_count']} A source that does not exceed 10MB is again checked for SHA at the end, with no difference. Large inputs have been compared byte byte with historical SHA and checked before and after metadata, without claiming full hashing at the end. All new materials are only in this directory and historical products are not covered. After ** is completed, stop and wait for the research decision.**
'''
(O/'audit_report.md').write_text(text)
