# NIDSProject overall execution plan v2.2

> Item3 revision: item3-scenarios-v1.0; actual freeze UTC:2026-09-07T23:57:41.942021+00:00; configuration:configs/item3_scenarios_v1.json. Only O1–O4 are revised; earlier stage history is preserved.

Original date:2026-09-05; current status revision:2026-09-08. Execution environment: project WSL. Original engineering design/stage definitions are retained below. Early recommended parameters now have actual frozen settings in configs/execution_v2.json, P6 snapshots and independent Item3 configuration. Section1 and the P7 specification govern current progress/next steps.

## 1. Current progress and execution conventions

Status revision:execution-plan-v2.2-p7, dated2026-09-08. This updates project status and priorities after inspecting P4–P6 development results and Item3 data support; it is not retrospective preregistration before model fitting. Original plan bytes remain in the locations listed in the accompanying Plan_Revision_Diff.md.

Main question: under which cross-domain conditions does conditional reliability improve selective prediction, and can dynamic consultation add value beyond simple cascades or MoE? Interpret reliability ranking, probability calibration, coordination feasibility, matched-budget gains and open-set semantic correctness separately.

| Stage | Actual status | Next handling |
|---|---|---|
| P0–P3 | Environment, features, source models and development replay complete; base prediction caches cover all domains | Preserve artifacts for P7 reuse |
| P4 | F1/F2 G0/G1 PASS; F3 G0 PASS, G1 INCONCLUSIVE | P7 reports Round1 in three folds; outer results do not reselect frozen P6 eligibility |
| P5 | Development sensitivities complete | Preserve all; sensitivity cannot replace the primary condition; P7-v1 excludes outer expansion |
| P6 | FROZEN_FOR_P7; F1 feasible, F2 main coordination has no feasible threshold, F3 lacks full coordination | P7 preserves all frozen methods/infeasible branches; no retraining/reselection |
| P7 | Formal outer evaluation complete and artifacts reviewed; G0 PASS, G1 BELOW_MINIMUM_EFFECT; no successful Round2 outer fold | Preserve negative/insufficient results; no reselection/retraining |
| Item3/P8 | item3-scenarios-v1.0 documents/configuration frozen; scenario models not fitted | No full O3 coordination; first check independent support for O1/O2/O4 |
| P9 | Core conclusion available; complete summary, figures and manuscript unfinished | Complete a formal negative-core-result summary; separate P8 results/unexecuted status |

P7-v1 evaluated original P4 primary conditions and frozen P6 methods. Equal-domain G0=0.121381 PASS; G1=-0.054990 with95% CI entirely below0 gives BELOW_MINIMUM_EFFECT. ToN F1 coordination risk is about63%–64%, far above5% and no better than the fixed cascade; all F2 frozen policies have coverage=0; F3 is ineligible for Round2. Core positive reliability/coordination claims lack support; see [core conclusion](<Project_Core_Conclusion_After_P7.md>). P8 is a restricted-family extension, without requiring success for the core study. Adjusting Item3-v1.0 after P7 cannot provide independent confirmation; register new methods/family designs separately as exploration with data exposure.

This historical delivery covers plan-status revision, P7 specification and code, without executing formal outer evaluation or changing execution_v2, round2, taxonomy, Item3 configuration or P4–P6 frozen results. See [P7 specification](<P7_Implementation_Spec.md>) for P7 implementation/evidence/commands; the revision comparison is a referenced artifact not included.

Historical execution conventions:

- Track versions with configuration snapshots, ordinary version identifiers, run time, dependency versions, file sizes and record counts. Hashes, SHA and smoke tests were outside this implementation scope.
- Use explicit data contracts and direct algorithms; generic exception swallowing, automatic device fallback, redundant compatibility layers and defensive programming were outside this implementation scope.
- Use GPU training/inference for supported models; necessary CPU-only algorithms use CPU with explicit configuration records.
- Repair the root cause in the module responsible for the logic, then recompute affected downstream artifacts.
- Documents, logs and stage reports use direct affirmative descriptions.

Retain17/17/15 feature groups, source domain and LODO design. Observed empty fields/infinities are real data semantics handled by shared preprocessing.

## 2. Hardware, software and computation

Execution root: `/home/karri/projects/NIDSProject`.

Python: `/home/karri/miniconda3/envs/agentnids/bin/python`, user-reported version3.11.16.

Hardware:1 NVIDIA RTX A5000,24564 MiB (about24 GiB); user-reported driver595.95. The CUDA13.2 shown by nvidia-smi describes driver capability; record the actual framework CUDA runtime separately.

| Task | Implementation | Device |
|---|---|---|
| Primary/full-feature RF | Native `cuml.ensemble.RandomForestClassifier` | CUDA |
| Primary/full-feature XGBoost and nonlinear CRE | `tree_method="hist", device="cuda:0"`, CUDA arrays | CUDA |
| Primary/full-feature MLP | PyTorch | CUDA |
| Platt, linear CRE, corrective/Post-CRE | Shared PyTorch logistic+LBFGS | CUDA |
| Imputation/scaling, entropy, JSD, MMD, batched distance/metrics/resampling | CuPy/PyTorch | CUDA |
| CSV decoding, mmap I/O, files, manifest joins, logs and plot layout | PyArrow, NumPy, standard library and plotting libraries | CPU |

cuML RF uses quantile splits. Primary RF and all49 RF share implementation/parameters for consistent controls. Official documentation: [cuML RF](https://docs.nvidia.com/cuml/latest/api/generated/cuml.ensemble.RandomForestClassifier/), [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/).

Initially record Ubuntu, driver visibility, torch/cuml/cupy/xgboost versions and runtimes in WSL. Choose compatible dependencies based on existing agentnids and pin installed versions; consult [RAPIDS platform support](https://docs.rapids.ai/platform-support/). Produce environment/dependency records.

Run sequentially on one GPU, with a separate process per training task. Hand over after exit. Initial resource settings:

- CSV chunks:262144 rows; prediction batches:131072 rows.
- MLP batch:8192; float32; one model on GPU.
- CRE low-dimensional inputs/parameters use float64; full R_fit matrices may reside on GPU, with chunked closure accumulation of the full objective/gradient.
- JSD batches:32 windows; MMD kernel tiles:1024×1024.
- Resampling:1 replicate at a time, reusing sorted/block indices.
- Plan arrays/models/temporary tensors within about18 GiB; record actual peak in the first formal task log.

Batch sizes are engineering settings. Diagnose memory issues through lifetimes, duplicate arrays and chunks while preserving configured model parameters, sample identities and statistical definitions. Estimate runtime after the first full formal task.

## 3. Planned project directory

```text
/home/karri/projects/NIDSProject/
├── datasets/                         # four original CSVs and feature definitions
├── configs/
│   ├── execution_v2.json             # supplied with plan; frozen after WSL registration
│   ├── feature_views.csv             # fixed49-row feature list
│   ├── taxonomy.json                 # label mapping and O1–O4 families
│   ├── round2.json                   # rules registered at Round2 implementation
│   └── round3.json                   # rules registered at Round3 implementation
├── docs/
│   ├── execution_plan_v2.md
│   ├── data_preparation_report.md
│   └── protocol_decisions.md         # parameter rationale, freeze and exposure
├── src/nids/
│   ├── __init__.py
│   ├── run.py                       # unified entry; explicit stage
│   ├── data/
│   │   ├── feature_store.py          # CSV → original-order mmap → sorted mmap
│   │   ├── partitions.py            # rank/original-row/fold/window joins
│   │   └── preprocessing.py         # shared imputation/scaling/source reference
│   ├── models/
│   │   ├── rf.py
│   │   ├── xgb.py
│   │   ├── mlp.py
│   │   ├── logistic.py              # shared Platt/CRE/corrective solver
│   │   └── calibration.py           # Platt input transform and competence
│   ├── replay/
│   │   ├── shift.py                 # JSD/MMD source reference and windows
│   │   ├── engine.py                # completed-flow state progression
│   │   └── ledger.py                # fixed columns, partitioned predictions
│   ├── reliability/
│   │   ├── cre.py                   # paired per-fold/model fitting
│   │   └── gates.py                 # G0/G1 and three-state decisions
│   ├── evaluation/
│   │   ├── metrics.py               # GPU metrics and tie definitions
│   │   ├── bootstrap.py             # paired GPU window-block resampling
│   │   └── reporting.py             # CSV/Markdown/scientific figures
│   ├── coordination/                # created at Round2
│   │   ├── policy.py
│   │   ├── fusion.py
│   │   ├── baselines.py
│   │   └── latency.py
│   └── openset/                     # planned for Round3
│       ├── quarantine.py
│       ├── class_match.py
│       └── decisions.py
├── scripts/
│   ├── run_round1.sh                # sequential formal-stage calls
│   ├── run_round2.sh
│   └── run_round3.sh
├── outputs/
│   ├── round1_data_prep_v1/          # completed manifests/sorted indices
│   ├── feature_store_v2/            # per-domain X.npy/label.npy/attack_code.npy
│   ├── source_v2/                   # preprocessing,6 models, Platt,h
│   ├── prediction_cache_v2/         # reusable base predictions by domain
│   ├── shift_v2/                    # shifts by domain/role/window/metric
│   ├── round1_v2/
│   │   ├── config_snapshot/
│   │   ├── F1/                     # cre/、development/、heldout/
│   │   ├── F2/
│   │   ├── F3/
│   │   └── summary/
│   ├── round2_v2/                   # per-fold policies/thresholds/ledger/metrics
│   └── round3_v2/                   # separate O1/O2/O3/O4 artifacts
├── logs/                            # per-stage/model logs
├── requirements-runtime.txt         # actual dependency versions
└── pyproject.toml                   # nids package and src layout
```

Move existing preparation implementations into their responsible modules while retaining manifests as inputs. Use a unified entry for new code and archive old handoffs. Each stage emits stage_report.md with input versions, scope, device, time, peak memory, artifacts and actual status.

## 4. Data and model implementation

### 4.1 Formal feature caches

Scan each original CSV for49 features, binary labels and attack-subclass codes. Convert blanks to NaN and±Inf to missing; output full-table column counts. Earlier numeric inspection covered one10k window; this formal conversion covers all records.

Write temporary original-order mmap, then gather in chunks using DOMAIN_sorted_original_rows.npy into sorted features. Final row r corresponds to sorted-index entry r. Temporary storage is scoped to the current domain and cleared after delivery of files explicitly created by that stage.

split_manifest.csv ranges [rank_start, rank_end_exclusive) are half-open sorted ranks. partitions.py maps them to sorted mmap, reused by training/windows/ledgers. Keep features and labels separate.

All49 float32 columns require about13.12 GB (12.22 GiB), plus labels, indices, temporary files and predictions. Process domains sequentially with CPU mmap chunks. Retain exact high-cardinality discrete values in auxiliary column caches for JSD category encoding.

### 4.2 Shared source preprocessing

Fit imputation only on S_train: numeric valid-value medians and source categorical modes, choosing the lowest code on mode ties. Entirely missing source columns use predefined0 and are logged.

Tree models use imputed raw17/49 columns. MLP uses imputation plus S_train mean/standard-deviation scaling, with scale=1 for constant columns. Retain numeric encodings and fixed49-dimensional first-version representation; JSD vocabulary is separate.

GPU computes medians/scaling/histograms/transforms. Save preprocessing_manifest.json/statistics arrays and use transform() for subsequent source partitions and targets.

### 4.3 Six base models

The following parameters were recommended for freezing before inspecting results, to define a runnable first version without claiming optimality.

| Model | Primary view/control | Fixed parameters |
|---|---|---|
| cuML RF | Traffic17 / All49 | 300 trees，depth16，min_leaf2，sqrt，128 bins，bootstrap=True，max_samples=1，n_streams=1 |
| CUDA XGBoost | Protocol17 / All49 | binary:logistic，400 rounds，depth6，eta0.05，subsample0.8，colsample_bytree0.8，lambda1，min_child_weight1，max_bin256 |
| CUDA MLP | Behavior15 / All49 | input→128→64→1，ReLU，Adam，lr0.001，weight_decay0，batch8192，30 epochs，BCEWithLogitsLoss |

Shared seed20260905; shuffle S_train batches each MLP epoch. Retain natural S_train class proportions and fixed epochs. Source probability calibration, competence and evaluation partitions keep distinct roles.

Run RF17,RF49,XGB17,XGB49,MLP15,MLP49 sequentially, training full S_train. Share model artifacts across three folds and all Round1 window sensitivities.

Source-calibrated Attack probability p predicts Attack when p>=0.5. Historical competence for predicted class k is h[i,k]=(correct_count[i,k]+1)/(predicted_count[i,k]+2), counted only on S_competence. Binary-entropy uncertainty is u=-(p*log(p)+(1-p)*log(1-p))/log(2), taking endpoint terms by limit0. CRE labels indicate calibrated prediction correctness.

### 4.4 Shared GPU-logistic mathematical contract

Platt takes logit of base raw probability clipped at epsilon=1e-7 and predicts binary Attack labels. Fit six independent calibrators on S_prob_cal. Recommended unpenalized logistic shares the CRE solver but registers its own objective.

Fit f(u),f(h,u),f(h,u,s) separately per detector. The last two share training records, domain weights, standardization, solver and regularization for the primary comparison.

For two development domains with N1,N2 rows and total N, each domain-d row has w=N/(2*Nd), total W=N. CRE objective:

```text
J(beta,b) = sum_i w_i * BCEWithLogits(z_i @ beta + b, correct_i) / W
            + ||beta||² / (2*C*W)
C = 1
```

Save intercept separately; regularize beta. Fit weighted means/population variances only on fold R_fit, with scale=1 for zero variance.

PyTorch CUDA LBFGS uses float64,max_iter=2000,tolerance_grad=1e-4,tolerance_change=1e-12,line_search_fn="strong_wolfe",history_size=20 and initial parameters0. Each closure traverses the same full R_fit, accumulating the same objective/gradient in chunks before adding regularization once. Log iterations, final objective and gradient norm; diagnose convergence in this solver.

Source Platt uses the same BCE with regularization0; inputs/statistics come from S_prob_cal. Keep the small-parameter task on GPU as requested.

## 5. Causal replay, shift and prediction ledgers

Decide only after a complete flow is available, proxied offline by FLOW_END_MILLISECONDS. Equal end times form one availability batch, progressed by original row. Conclusions concern completed-flow replay; exporter delay is a deployment extension.

Replay domains independently. Fix current shift from the manifest’s preceding completed full window before the current window’s first record. Initialize s=0 and log its flag. Explicitly allowed unlabeled completed context may cross adjacent partitions; retain original short-tail counts.

JSD reference is imputed S_train. Numeric features use10 quantile bins, merged duplicate edges and infinite outer edges. Source valid categorical values plus fixed OTHER define vocabularies.

Fixed JSD categorical columns: L4_SRC_PORT,L4_DST_PORT,PROTOCOL,L7_PROTO,TCP_FLAGS,CLIENT_TCP_FLAGS,SERVER_TCP_FLAGS,ICMP_TYPE,ICMP_IPV4_TYPE,DNS_QUERY_ID,DNS_QUERY_TYPE,FTP_COMMAND_RET_CODE. Others use numeric bins. Histogram definitions are independent of the three evidence views.

Per-feature JSD uses log2 and limit0 for 0*log(0/q); average49 columns into s. Compute window JSD then map current s by predecessor ID. Development/heldout layouts use their respective manifest roles.

Precompute base predictions in batches because detectors/source calibrators are fixed. Replay decisions read only acquired_evidence for queried models. Pre-CRE receives h,u,s; policy inputs contain only acquired evidence.

Partition Parquet ledgers by fold/domain/stage: record_id,rank,window_id,previous_window_id,model probabilities/predictions,u,h,s,CRE. Round2 adds queried_mask,query_action,fusion probability,Post-CRE,final_decision,calls,latency. Evaluation joins labels after predictions are fixed.

## 6. Formal stages

| Stage | Implementation/run | Deliverables | Completion condition |
|---|---|---|---|
| P0 environment/configuration | Register WSL GPU/runtime, directories, manifests and training/metric/taxonomy/threshold rules | runtime,configuration snapshots,protocol_decisions | GPU dependencies and initial rules explicit |
| P1 features | Convert four domains, full missingness, S_train preprocessing/JSD reference | feature_store,preprocessing,source_reference | Model inputs/source statistics saved |
| P2 detectors | Train six full GPU models sequentially; source Platt,h,evaluation | models,calibrators,competence_table,source_metrics | Six models/source calibration complete |
| P3 predictions/replay | GPU domain predictions,10k JSD,fold development records | base cache,shift,development ledger | Development inputs/shifts aligned |
| P4 Round1 primary | F1→F2→F3 paired CRE fitting,development G0/G1,paired block bootstrap | CRE,R_cal diagnostics,R_select metrics,gate_report | PASS/BELOW_MINIMUM_EFFECT/INCONCLUSIVE per fold |
| P5 Round1 sensitivity | 5k/50k JSD,10k MMD,10k nonlinear CRE,all49 controls | sensitivity_summary | Complete separate primary/sensitivity reporting |
| P6 Round2 development | Eligible folds fit corrective/Post-CRE,select thresholds,compare MoE/cascades | policies,thresholds,latency,development G2 | All fold selection rules/artifacts frozen |
| P7 final outer evaluation | Original P4 primary in three folds; F1/F2 all original frozen P6 points/budgets; predict all folds before scoring | outputs/outer_v2/P7,identity manifests,metrics,paired intervals | Disclose F2 infeasibility,F3 limits,outer budget violations; no new threshold selection |
| P8 Round3 | Item3 filtering,source prototypes/95% quantiles,fixed Attack threshold,conditional reliability; no single-Known pseudo-unknown | separate artifacts,applicable3×4 matrices,all scenario statuses | No full O3 coordination; verify new targets/threshold support for O1/O2/O4 |
| P9 summary | Primary,negative/insufficient,sensitivity,time/cost | overall_report,CSV,manuscript figures | Claims match supported evidence |

P4 development R_select determines next-stage decisions. Three folds share candidates/selection procedure, with all rules frozen before outer results. P7 follows Round2 development.

Report G0/G1 in all folds. If only some enter full Round2, disclose actual coverage and registered entry conditions alongside other folds’ conclusions. Three-domain overall claims require all three folds.

Domains serve development/heldout roles across folds, so inspecting cross-fold development then changing shared methods affects independence. Register common methods/parameters now; subsequent selections follow predefined fold procedures. Research changes form separate exploratory versions needing independent confirmation.

Current exposure includes P4/P5/P6 development,Item1/2 audits,Item3 support and outer class counts; it is no longer merely a data-preparation report. This revision does not backdate protocol_decisions.md. No new P7 performance/Item3 scenario predictions were computed at this revision.

## 7. Round1 conclusion rules

For domain d, G0 is:

```text
headroom_d = min_i(error_rate_i,d) - all_three_wrong_rate_d
G0 = mean_over_development_domains(headroom_d)
```

Also report disagreement,double-fault,error correlation,recoverable errors,three-model detection and all49 controls. Stratify Benign/Attack, highlighting BoT’s extreme Attack proportion. G0’s best single is a retrospective upper-bound reference; development selection chooses deployment’s initial detector.

Compute AURC(f(h,u))-AURC(f(h,u,s)) per model, then average equally across three primary detectors and two development domains for G1.

Compare constant correctness prior,raw h,logistic f(u),f(h,u),f(h,u,s). The prior uses fold R_fit equal-domain weights.1-u is a ranking diagnostic; Brier/NLL primary probability comparisons use fitted models. R_cal provides diagnostics; primary Round1 CRE directly outputs fitted probabilities.

Metrics: correctness AUROC/AUPRC,Brier,NLL,ECE in15 equal-width bins,risk-coverage,AURC. Float64 NLL clips at1e-15. AURC averages risks over all nonempty acceptance prefixes, using expected within-tie random ordering rather than file order.

Recommended minimum effects: G0=0.01 (1 percentage point recoverable errors),G1=0.001 (absolute AURC gain on0–1 scale). These practical study thresholds require measurable benefit; they are not universal statistical-significance standards.

Use consecutive10-window blocks per domain,retaining the last short block;2000 paired percentile replicates,95% CI. Methods share sampled records; domains resample independently before equal-weight aggregation. Recompute G0’s best single per replicate and retain original causal shift. Intervals describe block-resampling uncertainty conditional on frozen models/current domains.

| Status | Criterion | Next step |
|---|---|---|
| PASS | Estimate≥minimum effect,95% lower bound>0,at least20 blocks in each relevant domain | Both gates PASS permits full Round2 |
| BELOW_MINIMUM_EFFECT | 95% upper bound<minimum effect with estimability conditions met | Report effect below practical threshold |
| INCONCLUSIVE | Otherwise,or inadequate blocks | Preserve estimates/limits; later work is exploration |

10k R_select blocks:CIC51,BoT43,ToN69,meeting counts; counts only establish estimability.50k gives10,9,14,so report descriptively. Sparse source-test blocks limit source evaluation,not fitting on complete predefined source partitions.

GPU resampling caches score order and uses block multiplicities per row. Accumulate duplicate-prefix risks/ties by exact expanded-sample definitions and reuse arrays in place,avoiding a replicate×all-record matrix.

## 8. Sensitivity reuse and scale

- 5k,10k,50k share partitions,six base models,Platt,h and source JSD edges; each layout rebuilds s and fits corresponding CRE.
- All49 detectors are fixed controls;report their all49 results alongside the10k JSD primary comparison.
- Nonlinear sensitivity pairs CUDA XGBoost f(h,u)/f(h,u,s),recommended200 rounds,depth3,eta0.05 with other fixed tree settings; independent of the linear primary gate.
- MMD:10k windows,source-scaled49 dimensions,fixed-seed uniform2048-record S_train reference. RBF bandwidth is median positive pairwise reference distance. kernel=`exp(-distance²/(2*sigma²))`; use biased MMD² including diagonal and register sigma. Identical source reference values give sigma=1.
- Compare the preceding complete window against fixed source reference,chunk10k kernel sums and reuse source-source terms. This separate stage has approximately quadratic within-window kernel cost,reported separately.

Sensitivities answer predefined questions;10k JSD primary determines main gates.

## 9. Round2 code/run boundaries

Document synchronization (2026-09-07): Items1/2 align with revised Proposal§4.2 and protocol§15–16. Initial-detector/corrective-model definitions are retained while mathematical/data boundaries are completed. This document date does not replace actual registration before fitting. Preserve backups/frozen artifacts and confirm actual registration against WSL configuration/protocol_decisions.md.

Implement Round2 only with the corresponding gates. First-version candidates were recommended for P0 registration,with fold models/thresholds selected on development at P6.

1. **Initial detector (Item1):** use each eligible fold’s own two development R_cal complete-window records. Compute primary-detector error in each domain using frozen source-calibrated classification,then choose the lowest two-domain arithmetic mean. Ties:RF-Traffic,XGBoost-Protocol,MLP-Behavior. i0=argmin_i [error_i(domain1)+error_i(domain2)]/2,without row-count pooling. One initial detector serves all fold records/windows,independent of current shift/unqueried outputs. Register criterion,partitions/ties before development;freeze identity before outer evaluation.
2. **Corrective models (Item2):** six directed pairs/fold train on its two labeled development R_fit complete windows,including initialization but excluding incomplete tails. Target z=current_wrong & peer_correct,inputs h_i,u_i,s,yhat_i;use all eligible records,not only current errors. Domain weights N/(2*N_d);freeze source detectors,Platt,h/causal shift. Use§4.4 CUDA logistic,C=1,weighted standardization/frozen solver fitted only on R_fit. Single-label pairs use Laplace constant(sum(w*z)+1)/(sum(w)+2),reporting counts. Offline candidate predictions/labels may construct targets;before queries,inputs contain only currently acquired evidence,never candidate current outputs. R_cal diagnoses/selects initial detector;R_select selects registered policies/thresholds. Strict-0 forbids outer labels/adaptation samples while allowing labeled development proxies.
3. **Query policy:** maximize g(j|i)/normalized_cost_j among unqueried models. For the third query,i is the most recently acquired detector;query the sole remaining candidate. At most three detectors/record.
4. **Fusion/Post-CRE:** initially recommend Pre-CRE-weighted mean of acquired probabilities. Post-CRE receives three queried masks,three p/r groups (missing slots0),fused probability,maximum disagreement and s;fit fusion correctness on R_fit. Train all7 nonempty queried subsets with equal subset/domain weights;register input order in round2.json.
5. **Stopping/final acceptance:** recommend alpha_stop=0.05,minimum_coverage=0.10,thresholds0–1 step0.01,bootstrap2000. On R_select,maximize coverage subject to risk upper bound;resolve ties by fixed risk/threshold order. No feasible early threshold means complete queries. Final acceptance uses the same risk target with independently selected threshold;no feasible threshold means Abstain.
6. **Baselines:** Proposal controls include best single,uniform/static/confidence weighting,uncertainty routing,soft/hard MoE,fixed/learned cascades,random/cheapest. Register evidence permissions,training partitions/costs in round2.json before implementation;GPU-capable gates/fusion use CUDA.

Fixed Post-CRE slots preserve acquired-evidence completeness. Replay builds selector inputs progressively;offline labels join through a separate interface.

G2 first reports actual correction yield,calls/cost of selected detectors on consultable records,then coverage–risk–cost curves. g/c measures correction opportunity per cost;results establish actual fusion benefit.

Measure real model-call latency:single-flow decisions are primary query costs;batch throughput is separate. Recommend30 warmups/1000 measured single calls/model using fixed fold R_cal samples. Synchronize GPU timing and report mean,median,p95. All methods share batching/transfer conventions;end-to-end accounting includes preprocessing,amortized shift,CRE,routing/fusion. Cache replay is functional;costs come from real execution.

Claims of collaboration require actual superiority over strong baselines in performance/risk/cost. Similar MoE/cascade results require cost ranges/method differences. Outer evaluation validates development selection;development risk bounds are pointwise.

## 10. Round3 plan and dedicated artifacts

P0 fixes Proposal family mapping;P8 implements O1–O4. Quarantine Unknown source/development records before any fitting. Refit preprocessing,shift reference,six detectors,Platt,h,CRE,policies/class matching within scenario directories.

O1:BoT,Known Recon/Unknown Flood;O2 reverses these;O3/O4 use the same exchange on ToN. Fit only Benign+specified Known,excluding/counting other attacks. Outer primary retains Benign,Known,Unknown. Form filtered windows inside original partitions per Item3;source statistics use all retained rows,development complete10k windows,outer tails retain predictions without updating complete-window statistics.

class_match.py follows formal Item3/configs/item3_scenarios_v1.json:49 original-feature block distance,source Known prototypes,S_competence fixed95% quantile,Attack threshold0.5. Single-Known scenarios omit leave-one-known-family-out. O1–O3 R_select has no Known;O2 ToN R_cal can diagnose future Attack recall but cannot replace selection evidence. O1/O2 selection is Benign-conditional;O4 development Attack diagnostics rely on BoT. O3’s empty BoT R_cal/R_select full windows prevent full coordination. O1/O2/O4 need new-target/threshold-support checks.

Actions:Benign,Known,Unknown,Abstain. Unknown requires Attack evidence and low matching to all Known families;Abstain is a reliability action.

Report Unknown AUROC/AUPRC,precision/recall/F1,error destinations,3×4 truth-action matrix and coverage/risk across all retained three-class records. Error-consensus analysis compares high-agreement/high-shift with high-agreement/low-shift,retaining correct high-shift agreement controls;report ECAR/CCPR.

Register Round3 outer-label/family exposure alongside prior Round1/2 exposure. Distinguish exploration from independent validation on reused domains.

## 11. Delivery order and command interface

Historical P0–P6 runs are complete/preserved. Current sequence:P7 specification/code check→explicit P7 outer evaluation→core P9 summary→P8 conditional on resources/scenario support. P0–P5 commands below describe historical interfaces,not current rerun instructions.

Commands below are historical interface contracts;src/nids/run.py determines callable stages. preflight-p7/outer-p7 are current additions;see the P7 specification.

```bash
cd /home/karri/projects/NIDSProject
export PYTHONPATH=/home/karri/projects/NIDSProject/src

/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage register
/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage features
/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage train-source
/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage calibrate-source
/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage replay-development
/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage round1-primary
/home/karri/miniconda3/envs/agentnids/bin/python -m nids.run --config configs/execution_v2.json --stage round1-sensitivity
```

Actual P6 entry is coordination-p6;P7 entries are read-only preflight-p7 and formal outer-p7. round3/report remain unimplemented;old planned names are not executable stages.

Formal checks inspect actual outputs:input/manifest identities,predecessor availability,paired CRE records/weights,data roles,objectives/gradients,prediction/metric counts. Save them with stage reports. Original test/wrapper requirements follow historical user execution conventions.

Repair examples:rank mapping in partitions.py,NaN/Inf in preprocessing.py,premature shift visibility in engine.py,regularization objective in logistic.py. Log downstream scope and restart at the affected node.

Immediate next work:complete core P9 summary,scientific figures/manuscript as a valid completed negative-core study. Defer full P8 coordination. If an open-set extension is required,first inspect identities/filtering/source-prototype estimability,then decide from independent resources. Do not rerun P0–P7,reselect P6 thresholds or rewrite Item3-v1.0 to improve conclusions.

<!-- ITEM3_FROZEN_RULES_BEGIN -->
# Item3 formal scenario rules: item3-scenarios-v1.0

Status:FROZEN_BEFORE_ITEM3_SCENARIO_FITTING_AFTER_DATA_SUPPORT_AUDIT. Actual freeze UTC:2026-09-07T23:57:41.942021+00:00.

Configuration:`configs/item3_scenarios_v1.json`. These formal O1–O4 rules revise corresponding Round3 clauses while preserving P4–P6/freeze history. Registration follows inspection of development support/outer class counts,not retrospective preregistration. Freezing does not authorize training/P7/P8 execution.

## A. Scenario scope [frozen]

| Scenario | Source | Development | Outer | Fitted Known | Specified Unknown |
|---|---|---|---|---|---|
| O1 | UNSW | CIC、ToN | BoT | Reconnaissance_Probing | Availability_Flooding |
| O2 | UNSW | CIC、ToN | BoT | Availability_Flooding | Reconnaissance_Probing |
| O3 | UNSW | CIC、BoT | ToN | Reconnaissance_Probing | Availability_Flooding |
| O4 | UNSW | CIC、BoT | ToN | Availability_Flooding | Reconnaissance_Probing |

Fit strictly Benign and specified Known. Quarantine Unknown before all fitting;exclude other attacks from the entire main scenario. Other is neither Benign nor Unknown and contributes to no vocabulary,imputation,scaling,models,prototypes,h,thresholds/policies. Follow taxonomy/native labels without creating separate Known families from DoS/DDoS sublabels.

Source S_train trains detectors/scenario preprocessing/reference/prototypes;S_prob_cal fits Platt;S_competence estimates h/Known distance thresholds;S_source_test only evaluates frozen models. Filter all source partitions first. Register S_competence’s dual role;do not select representations/distances/quantiles there. Its coverage is source calibration,not independent testing.

Two development R_fit domains fit CRE,corrective/Post-CRE;R_cal selects initial detector/diagnoses;R_select selects registered policies/reliability thresholds. Keep equal domain weights N/(2N_d),even when Known is absent. Outer records/labels cannot supplement development support.

Quarantined source/development Unknown remains raw data,without an additional tuning/validation channel. Main outer evaluation retains only Benign,Known,specified Unknown;exclude Other from metrics/replay and disclose counts/retention. This taxonomy-filtered controlled scenario does not represent full native traffic. Labels are used only for predefined data construction,not inference.

BoT O1/O2 each exclude1,615 Theft records;ToN O3/O4 each exclude5,024,357 Other records. Other partition counts are in candidate_partition_counts.csv;do not adjust them by scenario outcomes.

## B. Retained rows, windows and insufficient support [frozen]

Preserve domain,stream,partition,record_id/availability order. Filter/form sequential windows only within original partitions,without borrowing rows across boundaries.

Source supervised fitting,Platt,h/prototype-distance samples use all retained rows,including tails shorter than10k. JSD reference uses all allowed S_train. Development CRE/selection use filtered complete10k windows,including initialization but excluding incomplete tails. Shift uses the prior completed full window,initial0;cross-partition context requires existing explicit permission. Tails are never full predecessors.

Future outer replay forms windows on the predefined retained three-class stream. Tail records still receive predictions/3×4 counts,using the preceding completed full window without updating full-window statistics. Initialize0 if absent and flag it. Separate full-window/block metrics from all-row metrics. This tail rule is frozen here.

If a development partition has0 complete windows,mark the two-domain task NOT_ESTIMABLE_EMPTY_DOMAIN. Do not assign the other domain weight1 or describe this as no threshold passing. O3 already has this blocker and does not execute full outer coordination. Feasible source tasks do not remove it. Any substitute branch requires a separately proposed/frozen rule.

## C. Features, prototypes and distance [frozen]

Retain49 features and17/17/15 views. Prototypes use49 original-feature blocks:37 numeric/12 nominal per current shift list. Model views remain unchanged.

1. Convert numeric±Inf to missing;impute valid medians from allowed scenario S_train (Benign+Known). Entirely missing columns use0 and are logged. Standardize to z with the same population mean/standard deviation,scale=1 for constants. Other partitions only transform. Do not reuse old full-S_train prototype statistics.
2. Nominal values use P1 float64 auxiliary originals;build vocabulary only from allowed scenario S_train. Reserve distinct MISSING/OTHER symbols;nonfinite values→MISSING,valid unseen values→OTHER. No rare-category merging,target vocabulary expansion or ordinal Euclidean encoding.
3. Represent each categorical original feature with its own one-hot block,without per-column variance standardization. Compute high-cardinality blocks implicitly if desired;dense matrices are unnecessary. This is class-match representation only. Detectors retain numeric encodings/imputation but refit source fills per scenario;do not mix representations.
4. Known prototypes use only corresponding S_train:mean z vector μ_K and category-frequency vectors π_Kj. Benign contributes shared preprocessing only,not Known centers. Do not fit Unknown/Other prototypes.
5. Fixed squared distance:

   D_K(x) = [Σ_(j∈numeric)(z_j−μ_Kj)² + 0.5 Σ_(j∈categorical)||e_j(x)−π_Kj||²] / 49。

   Each original-feature block has weight1/49;categorical0.5 gives distinct one-hot points contribution1. Categorical distance is(1−2π_Kj[value]+Σπ_Kj²)/2,preventing automatic weighting by block width. Numeric squares may exceed1,so equal weights do not imply equal empirical contributions. This is a new fixed rule,not derived from mean JSD.
6. Do not fit PCA,embeddings,covariance/distance weights or select features by development results. All distance inputs/outputs must be finite;repair the responsible module rather than discarding anomalous records for support.

JSD remains causal shift over49 marginals;refit bins/vocabularies on allowed scenario S_train. Save separate OTHER/prototype encoding manifests rather than reusing old vocabularies and claiming quarantine.

## D. Source distance quantile and minimum support [frozen]

Fix q=0.95 without a grid. After S_train fixes μ,π/preprocessing,sort all scenario S_competence Known distances D_(1)…D_(n),take k=ceil((n+1)q),t_K=D_(k). D_K≤t_K matches,D_K>t_K does not;include equality,no interpolation. k>n is non-estimable,not an infinite threshold. This order statistic does not guarantee distribution-free coverage with temporal dependence/cross-domain shift.

Prototype minimum:100 Known records spanning≥20 original source10k groups. Threshold minimum:1,000 Known distances spanning≥20 original source10k groups. Groups describe temporal coverage,not independence. These operational limits lack established statistical sufficiency and are distinct from G0/G1 gates.

Current support:Recon S_train=5,386/46 groups,S_competence=4,520/35;Flood=1,739/46 and1,551/35. Both pass candidate counts. Threshold order indices k=4,295/1,475;distance/threshold values remain uncomputed.

Insufficient n/group coverage/finiteness gives PROTOTYPE_NOT_ESTIMABLE or MATCH_THRESHOLD_NOT_ESTIMABLE. Do not lower limits,borrow R_fit/outer rows or classify everything Unknown when a prototype is missing. Stop the four-action scheme;future separately authorized source binary metrics are not a complete open-set experiment.

Single-Known scenarios use NOT_APPLICABLE_SINGLE_KNOWN_FAMILY. Remove mandatory leave-one-known-family-out here,without substituting native sublabels or specified outer Unknown. Future multi-family extensions need independent rules/confirmation,not post-hoc attribution to this main experiment.

## E. Three thresholds and four actions [frozen]

### E1. Class-matching threshold t_K

From S_competence Known distances per section D. It concerns source Known distance distribution,not Attack probability,unknown-detection guarantees or reliability probability.

### E2. Attack-evidence threshold τ_A

All O scenarios fix τ_A=0.5 for new Platt/fused probability a(x),matching binary correctness’s0.5 boundary. Specify uniformly before fitting;even O4 with development Known does not optimize this threshold.

Do not maximize Attack F1/Youden/open-set thresholds on R_select lacking Known. O1–O3 cannot assess Known recall there. O2 ToN R_cal supports future recall diagnostics after new predictions,but does not replace R_select. No scenario selects τ_A with quarantined Unknown. Fixed0.5 defines actions without guaranteeing calibrated fusion or good Known/Unknown recall.

### E3. Early stopping and final acceptance/abstention τ_stop,τ_final

Apply P6 candidates to new populations:equal-domain R_select,grid0…1 step0.01,stop/final risk limit0.05,minimum coverage0.10,2,000 paired replicates in10-complete-window blocks,retaining final short blocks;≥20 blocks/domain. Share draws/retain causal shift. One-sided95% percentile upper bounds are pointwise development evidence,not simultaneous post-selection or outer safety guarantees.

Early coverage counts early accepts/all records;final coverage counts final accepts/remaining terminal population. Ties:max coverage,min empirical risk,max threshold. No early threshold disables early stopping;no final threshold abstains on the terminal population. Zero-accept risk is undefined;original bootstrap treats zero-accept replicate risk as1. Empty domains/inadequate blocks are non-estimable,not failed candidate scans.

O1/O2 thresholds are BENIGN_CONDITIONAL_SELECTION only:support for false alarms/Benign correctness,without validation of misses/open-set class errors. O3 lacks two-domain selection;do not invent thresholds. O4 supports overall two-domain selection with Attack evidence from BoT;report by domain.

Future R_fit predictions must verify correctness labels. Single-label primary P4 CRE remains NOT_FITTABLE_SINGLE_CORRECTNESS_CLASS,without an automatic constant substitute completing the chain. Six corrective models/Post-CRE retain P6 Laplace constants,but label CONSTANT_SINGLE_LABEL and absent empirical positive/negative support,without claiming learned correction. Empty h predicted classes likewise have prior support0. Register every scenario-specific application.

### E4. Action order and risk interpretation

When prerequisites/thresholds are estimable,acquire evidence under the new frozen policy. After passing early/final reliability,output Benign if a(x)<0.5;otherwise Known if D_K≤t_K,Unknown if D_K>t_K. Failed reliability gives Abstain. Missing matching thresholds make the entire open-set scheme non-estimable,not Unknown.

Pre/Post-CRE predicts binary Benign/Attack correctness. Passing τ_final does not imply Known/Unknown semantic correctness≥0.95. Separately report binary selective risk,three-class semantic errors,3×4 truth-action matrix and abstention. Measure prototype cost separately in future scenarios without rewriting old P6 costs.

## F. Non-estimable branches and claims [frozen]

O3 full coordination remains NOT_ESTIMABLE,without outer main comparison. Do not transfer O1 thresholds,substitute source for two-domain policy thresholds or resplit. O1/O2 only support Benign-conditional selection;outer attacks are first evaluated after freezing. O4 has Attack validation in only one development domain.

Supported discussion covers controlled-family source prototypes,source Known matching coverage,applicable Benign-conditional selection and one frozen descriptive outer open-set evaluation. Development risk bounds cannot validate unknown recognition,all-family coverage or transfer of old P4 G0/G1/P6 eligibility to filtered populations.

List every O scenario,including unrun,non-estimable and negative outcomes. Do not report only runnable/favorable ones.

Canonical configuration stores the same A–F text plus explicit parameters,roles,exclusions/failure branches. Full O1/O2/O4 coordination still requires new CRE/corrective targets,solver/threshold support;O3 does not begin full coordination.
<!-- ITEM3_FROZEN_RULES_END -->
