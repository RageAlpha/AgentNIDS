# Experiment plan: class-specific acceptance rules and local label support

Version v1; 2026-09-29; controlled exploration. This plan takes precedence over conflicting sampling, model, stopping-threshold and hashing requirements in earlier plans. It does not continue the original Variant B or establish CP guarantees anew.

## 1. Research questions and authorized scope

Q1: With RF outputs fixed, does local label support provide additional error-identification information within the same predicted class, similar confidence and the same time segment?

Q2: Can this information improve acceptance decisions for both classes while retaining useful benign releases and attack alerts?

This round uses only RF-All49 Raw on ToN/BoT. Actions either accept the original label or abstain; the classification threshold is p_Attack≥0.5, including equality in Attack. No relabeling, retraining, Platt calibration, new CP, collaboration, feedback, XGB, tree leaves or representation learning. G3 distance gating is also excluded; neighborhood distance is saved only as a support descriptor.

S1 preparation and freezing → S2 fixed 1% evaluation → decision and stop. S3 full evaluation requires the subsequent second user instruction; a scientific recommendation to continue does not authorize a new execution branch.

## 2. Verified laboratory paths

The project root `/home/karri/projects/NIDSProject` comes from historical commands.sh; current remote laboratory access is not assumed.

| Asset | Path relative to the project root |
|---|---|
| Original random 400 manifest | outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/calibration_sample_manifest.csv |
| Labels | outputs/feature_store_v2/{domain}_label.npy |
| RF probabilities | raw_probability field in outputs/prediction_cache_v2/{domain}/rf_All49.npy |
| Cache metadata | outputs/prediction_cache_v2/{domain}/rf_All49_manifest.json |
| Original row indices | outputs/round1_data_prep_v1/{domain}_sorted_original_rows.npy |
| Windows | outputs/round1_data_prep_v1/window_manifest.csv |
| Feature metadata | outputs/feature_store_v2/feature_store_manifest.json and domain manifests |
| Source preprocessing | outputs/source_v2/preprocessing/preprocessing_manifest.json and its referenced statistics files |
| Original model metadata | outputs/source_v2/models/rf_All49/training_manifest.json |

Determine exact feature-array filenames from the actual manifest and existing loader. Use rank-aligned values after source imputation and before standardization; apply the frozen source fill to raw storage when necessary. Inverting standardized arrays using source statistics requires explicit evidence. Do not guess fields, sort again or apply preprocessing twice.

One necessary check of fields, shapes and sample-row alignment is allowed. Search the relevant asset directories first for missing locations. Stop and record a blocker for an affected domain only when identity/semantics cannot be confirmed. Other disks, retraining and full historical workflow audits are outside scope.

## 3. Labels, split and evaluation range

Strictly reuse the original natural random 400 per domain, not the former 200B+200A manifest. Original sampling used PCG64(20260914).choice(100000,400,replace=False), selecting indices before inspecting labels. If the manifest is missing but rank arrays remain, reconstruct it from those arrays. If both are absent, stop that domain rather than silently resample.

Sort each domain’s 400 records by availability_rank, reinitialize PCG64(20260929) per domain, and assign the first200 entries of permutation(400) to reference R and the remaining200 to rule-selection C. Perform this split once, without label stratification, seed changes or supplementary samples.

Deterministic local reconstruction from the original manifest, without model fitting or reading new evaluation labels:

| Domain | B/A in400 | B/A in R | B/A in C |
|---|---|---|---|
| ToN | 343/57 | 177/23 | 166/34 |
| BoT | 41/359 | 25/175 | 16/184 |

These counts check identity/splitting and must not change sampling. Investigate sorting/manifest differences if laboratory counts differ; do not force the data to match these counts.

Each true class requires at least5 records in R and10 in C; otherwise mark LABEL_SUPPORT_LIMITED. If C contains fewer than10 records of a predicted class, that action branch cannot select a reliable acceptance rule; the other branch may continue. Report predicted-class and true-class support separately.

All400 target labels count toward the budget: R supplies neighbors and C selects rules; Q400 may use all400 labels. Subsequent ground truth is read only for offline scoring and never feeds neighborhood construction, scaling, candidates or rule selection. Earlier research has exposed these domains, so this round is not a new blind test.

Exclude the entire startup interval [0,100000):

| Domain | Subsequent range, left closed/right open | Four segment boundaries | Full records | S2 sampled records |
|---|---|---|---:|---:|
| ToN | [100000,27520000) | 100000,6960000,13820000,20670000,27520000 | 27420000 | 274200 |
| BoT | [100000,16930000) | 100000,4310000,8520000,12730000,16930000 | 16830000 | 168300 |

Retain complete10k windows and the existing tail exclusion. S2 uniformly selects100 records without replacement in every subsequent10k window. Reinitialize independent generators per domain: ToN seed=20260930, BoT seed=20260931. In window order, call choice(10000,100,replace=False), add rank_start and sort within each window. Index selection reads neither labels nor probabilities; save and freeze the manifest without resampling for class support.

Total442500 records. All methods share the S2 sample within each domain. S2 metrics estimate performance on an equal-probability1% sample per window. Absolute error counts are marked sampled; multiplying by100 does not make them measured full counts. S3 includes S2 records and expands coverage rather than providing independent replication.

## 4. Fixed local evidence

The base RF keeps All49. Only neighborhood distance excludes the transaction identifier DNS_QUERY_ID, with equal weights for the remaining48 features. This distance design is specified beforehand; it neither changes RF inputs nor selects features from subsequent errors.

Use exact match/mismatch0/1 distance for11 categorical features: L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO, TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS, ICMP_TYPE, ICMP_IPV4_TYPE, DNS_QUERY_TYPE and FTP_COMMAND_RET_CODE.

For the remaining37 numeric features, use positive scale_j from frozen source preprocessing:

    delta_j(x,r) = min(abs(x_j-r_j)/scale_j, 10)/10

Total distance is the mean of48 terms. This fixed mixed-feature distance clips numeric contributions to limit heavy-tail dominance; it is not a validated semantic distance. Equal categorical mismatch weights and source scales may also fail, a limitation that must be retained.

k=15 with equal weights. The200 records in R are the only neighbor library. Sort by (distance, reference availability_rank) to resolve ties and take15 neighbors. Neither C nor evaluation labels serve as neighbors. Full per-record sorting is unnecessary if an equivalent stable rule resolves ties at the15th distance.

    local_support(x) = mean(y_neighbor == raw_prediction(x))

Also save mean15-neighbor distance, local support and confidence s=max(p,1-p). local_support describes label support, not a probability estimate or neighborhood-model accuracy.

Calculate and compare distances in float64, preferably in batches. CPU/GPU are allowed; check equivalent implementation on a small unlabeled subsequent sample first. Do not silently substitute approximate neighbors, a different distance, k or precision for speed. Batch size may follow available memory without an additional audit round.

## 5. Candidate gates and controls

For s in each original predicted class of C, use quantile levels [0,.25,.5,.75,1], method='linear', add fixed0.5 and1.0 and deduplicate into a shared domain threshold grid T. All C-based methods use this T; ignore quantiles of missing predicted classes.

All comparisons retain original labels: HIGH means s≥t and LOW means s≤t. Handle tied scores together rather than randomly splitting threshold ties. Each class also allows OFF, abstaining on the entire class.

| ID | Rule family | Labels used for selection |
|---|---|---|
| RAW | Accept all | None |
| G0 | Shared t across both classes, HIGH; each class may be ON/OFF | C |
| G1H | Independent t per class, HIGH or OFF | C |
| G1F | Independent t per class, HIGH/LOW/OFF | C |
| G2 | G1F plus local_support≥u, u∈{0,.5,.75,1} | C; neighbor labels from R |
| Q400 | Same family as G1F; grid built by the same rules from all400 | All400 |
| CONST_A / CONST_B / ABSTAIN | Predict Attack everywhere / Benign everywhere / abstain everywhere | None |

G2 at u=0 exactly contains G1F. Primary evidence compares G2 with G1F using the same C, and with Q400 for label efficiency. G0→G1H isolates threshold flexibility; G1H→G1F isolates direction flexibility. Comparing G2 only with G1H cannot establish the value of local information.

Include original Variant B only as historical context, noting its different sampling and possible relabeling. Do not rerun it or mix it into the primary comparison table.

## 6. Fixed empirical selection rules

The following is an engineering operating point for the new trial, rather than a former professor instruction, statistical guarantee or deployment safety standard: each enabled action branch requires calibration empirical error≤5% and at least10 accepted records.10 only avoids extremely small denominators; even10 records with no errors cannot certify5% risk. BoT C contains only16 true B records, so requiring20 per branch would make B structurally infeasible at the5% target. This round therefore fixes10 uniformly and does not adjust it after results.

Define TN=(D=B,Y=B), FN=(D=B,Y=A), FP=(D=A,Y=B), TP=(D=A,Y=A). RB=FN/(TN+FN), RA=FP/(TP+FP). Check each enabled branch separately. OFF has undefined risk and contributes0 utility; disabling a branch does not mean meeting its risk target.

Among feasible candidate pairs maximize U_bal=.5*(TN/N_B+TP/N_A), using all true-class counts in the relevant selection set as denominators. Then, in order: maximize min(U_B,U_A); minimize total errors across both branches; prefer lower u_B then lower u_A (OFF uses0;methods without u also use0); prefer B direction HIGH then A direction HIGH (OFF last); use the predefined stable candidate-ID lexicographic order. IDs combine direction, ascending-threshold index and u index with zero padding; save the order definition. Mathematically equivalent per-branch optimization may reduce enumeration cost but must retain tie rules.

G0 enumerates one shared t and ON/OFF combinations; other methods enumerate per-class candidate pairs. If only OFF is feasible for a class, retain the other class and mark PARTIAL_ACTION_POLICY. Both OFF gives NO_FEASIBLE_POLICY. Rejecting all B cannot conceal target failure.

Save every candidate’s counts, selected rules, selection support and freeze time before opening S2 labels. Do not use S2 to select HIGH/LOW direction, k, u, thresholds or replacement rules. This round has one split, one k and one distance.

## 7. Evaluation and local-information diagnosis

### 7.1 Actual decisions of fixed policies

All methods output overall, four-segment and per-window counts. Report at least RB, RA, UB=TN/N_B, UA=TP/N_A, U_bal, acceptance rate, acceptance by original predicted class, abstention within true class, incorrect releases FN, false alerts FP and every denominator. Undefined values use JSON null or CSV NA, not0.

The S2 useful operating point requires overall RB≤5%, RA≤5%, UB≥10%, UA≥10%, with at least100 accepted samples for each action class spanning at least10 distinct windows. Any unmet requirement prevents JOINT_TARGET_MET; inadequate denominators take priority as EVAL_SUPPORT_LIMITED. Apply the same definition separately to all four segments and display weak segments. These are empirical evaluations, not risk certification.

Overall acceptance alone cannot establish superiority. Report the complete four-dimensional trade-off of fixed policies. This round does not use evaluation labels to search for optimal thresholds at matched acceptance.

### 7.2 Auxiliary mechanism diagnosis for Q1

Stratify by four segments × original predicted class × fixed confidence bins with boundaries [0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0], left closed/right open, including1 in the final bin. Compare local_support≤0.5 (low) with≥0.75 (high); exclude intermediate records from this comparison but report their count.

For each stratum save both groups’ sample, error and window counts and error_low-error_high. A valid stratum requires each group≥30 records spanning≥5 windows. Save all strata and label invalid ones; do not display only favorable strata.

Within each original predicted class, aggregate valid-stratum differences with w=min(n_low,n_high), avoiding direct comparison of mixed confidence/time populations. Freeze these original weights and resample only numerator/denominator counts for uncertainty. A replicate is invalid if any originally valid stratum has a zero group denominator. Valid rate<90% gives CI=NA. This conservative NA indicates insufficient evidence rather than ineffectiveness.

Mark a predicted class LOCAL_SIGNAL_POSITIVE when its weighted difference is≥2 percentage points, the95% block-bootstrap lower bound is>0 and same-direction valid strata occur in at least two time segments. Do not extend this to both classes. Mark NO_MATERIAL_SIGNAL_AT_THIS_RESOLUTION only with adequate support and a95% upper bound<2 percentage points. Otherwise use INCONCLUSIVE, without forcing success/failure.

This is a stratified association, not a strict conditional-independence test, and uncontrolled features may still affect it. Mechanism diagnosis does not refit the current policy.

### 7.3 Uncertainty

Compute paired temporal block bootstrap only for frozen primary policy differences (G2-G1F, G2-Q400) and the local signals above. Within each domain use consecutive blocks of10 complete windows, retaining the final block;1000 replicates, PCG64(20261001), sharing block indices across methods within the domain and replicate. Use2.5/97.5 percentiles. Recompute risk/utility ratios from replicate counts rather than averaging window percentages. Discard zero-denominator replicates, report valid rate and omit CIs below90%.

This describes comparative uncertainty on the current time series, excluding calibration sampling, model seeds and arbitrary future drift. Do not increase replicates or change block length/seeds because intervals are unfavorable.

## 8. Execution steps and outputs

S1: read required assets → reuse400 and fixed R/C → load source preprocessing and48-dimensional distance → compute C neighborhoods → select all policies → save protocol and rules. Resolve semantic issues using existing project manifests/loaders where possible; ordinary path adaptation does not require a pause. Stop and record genuinely conflicting semantics for the affected domain.

S2: freeze100 indices per window → batch local support and all frozen actions → read corresponding labels for offline counts → produce the tables, intervals and02 decision → stop. An infeasible G2 does not end Q1 diagnosis: absence of a gating operating point does not establish absence of local information. Do not fabricate results for a blocked domain; the other may continue.

S3: only after a subsequent instruction, stream the predefined full range using exactly the same R, distance, thresholds and actions. No refitting or method changes after pilot. Retain at least RAW, G1F, G2 and Q400; G0/G1H may also be counted cheaply from frozen rules. Reuse this protocol’s statistics/decisions and disclose that full evaluation includes pilot samples.

Output structure:

    run_<UTC>/
      src/run_pilot.py, required adaptation modules
      config/protocol_v1.json, resolved_assets.json, metric_definitions.json
      calibration/sample_roles.csv, support.csv, candidates.csv, frozen_rules.json
      pilot/evaluation_sample_ranks.npy, prediction_evidence.parquet or chunked npz
      tables/comparison.csv, segment_metrics.csv, window_counts.csv, local_signal.csv
      tables/paired_intervals.csv
      reports/summary_ZH.md, decision.json
      checks/basic_checks.json
      logs/commands.sh, execution.log

Sample evidence includes at least domain/rank/segment/window/y/raw_p/raw_pred/local_support/mean_neighbor_distance and each method’s action. Do not print/export unrelated personal information.

## 9. Minimal correctness checks, without hashes

Check only uniqueness/disjointness of400 and R/C; rank bounds and sample alignment; valid probabilities; class meanings and feature order; k neighbors from R; unchanged original labels; threshold equality; NA for zero denominators; consistent window/segment/overall counts; S2/S3 row counts.

Use small synthetic arrays to verify RB/RA denominators, OFF, ties, G2 u=0 equivalence to G1F and detection of misalignment. Then perform one small performance/memory check without subsequent labels before S2. These are required statistical-code checks, not additional audit projects.

This historical protocol excludes hashlib, SHA/MD5, file digests/checksums, whole-disk/project inventory and exhaustive per-file checks of old models/caches. In particular, do not execute inventory/check_frozen from old strictK common.py. Reuse only data-reading logic; write new modules separately and keep old files read-only. Resume checks use configuration fields, completed chunk IDs and counts, without content hashes.
