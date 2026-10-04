# Round 1 data preparation report

Status: DATA_PREPARATION_COMPLETE_NOT_EXPERIMENT_FROZEN; experiment configuration remains DRAFT_NOT_FROZEN.
Execution: 2026-09-05T07:24:27.186813+00:00 to 2026-09-05T07:27:41.872749+00:00.

Scanned 66,935,021 records. No models were trained and no outer-model results were inspected.

## Implementation and partition rules

Use NumPy and the standard library in the existing agentnids environment. Keep only times and binary labels per chunk; original CSVs remain read-only.
Record identity is (dataset, original filename, zero-based data-row index). Stable-sort by (FLOW_END_MILLISECONDS, original row index); index NPY files support mmap.
No actual capture/session columns or additional metadata are available. Treat each domain as one replay stream. Consecutive 10,000 records form indivisible temporal groups; retain the final short group.
Source boundaries use cumulative 60%/75%/90% of all groups, rounded to the nearest integer with exact halves upward. Target development uses cumulative 50%/75% of complete 10k groups under the same rounding; short tails belong to R_select.
Assign partitions consecutively in temporal order without class-driven changes. The 5k/10k/50k schemes reuse group assignments and restart windows within partitions.
Retain partition membership for tails shorter than a complete window, but exclude them from complete-window analyses at the relevant size.
A predecessor is the previous completed full window in the same scheme and stream. Label-free context may cross adjacent partitions; skip excluded tails.
Equal end times mean availability in the same batch. Predecessor completion may equal the current first-record availability time. Stable row order within a batch does not imply a strict earlier-time relation.
Bootstrap blocks contain 10 consecutive complete windows within each partition. Retain the final short block; no resampling is executed at this stage.

## Partition classes and counts

| Dataset | Role | Partition | Records | Benign | Attack | Actual fraction |
|---|---|---|---:|---:|---:|---:|
| UNSW | source | S_train | 1,420,000 | 1,377,436 | 42,564 | 60.031521% |
| UNSW | source | S_prob_cal | 360,000 | 325,600 | 34,400 | 15.219259% |
| UNSW | source | S_competence | 350,000 | 319,288 | 30,712 | 14.796502% |
| UNSW | source | S_source_test | 235,424 | 215,407 | 20,017 | 9.952719% |
| CIC | development | R_fit | 10,060,000 | 7,857,490 | 2,202,510 | 50.011113% |
| CIC | development | R_cal | 5,020,000 | 4,969,059 | 50,941 | 24.955844% |
| CIC | development | R_select | 5,035,529 | 4,688,077 | 347,452 | 25.033043% |
| CIC | heldout | T_heldout | 20,115,529 | 17,514,626 | 2,600,903 | 100.000000% |
| BoT | development | R_fit | 8,470,000 | 49,511 | 8,420,489 | 50.018283% |
| BoT | development | R_cal | 4,230,000 | 933 | 4,229,067 | 24.979615% |
| BoT | development | R_select | 4,233,808 | 1,545 | 4,232,263 | 25.002102% |
| BoT | heldout | T_heldout | 16,933,808 | 51,989 | 16,881,819 | 100.000000% |
| ToN | development | R_fit | 13,760,000 | 10,495,665 | 3,264,335 | 49.999528% |
| ToN | development | R_cal | 6,880,000 | 3,687,746 | 3,192,254 | 24.999764% |
| ToN | development | R_select | 6,880,260 | 2,608,803 | 4,271,457 | 25.000709% |
| ToN | heldout | T_heldout | 27,520,260 | 16,792,214 | 10,728,046 | 100.000000% |

See split_manifest.csv for partition millisecond ranges and data_audit.json for original time quality, attack-subcategory counts, and maximum within-domain time gaps.

## Complete windows, excluded tails, and resampling units

| Dataset | Role | Partition | Flows per window | Complete windows | Excluded records | Blocks | Median window span (seconds) |
|---|---|---|---:|---:|---:|---:|---:|
| UNSW | source | S_train | 5000 | 284 | 0 | 29 | 183.98399999999998 |
| UNSW | source | S_prob_cal | 5000 | 72 | 0 | 8 | 170.539 |
| UNSW | source | S_competence | 5000 | 70 | 0 | 7 | 168.09050000000002 |
| UNSW | source | S_source_test | 5000 | 47 | 424 | 5 | 165.173 |
| UNSW | source | S_train | 10000 | 142 | 0 | 15 | 379.091 |
| UNSW | source | S_prob_cal | 10000 | 36 | 0 | 4 | 340.211 |
| UNSW | source | S_competence | 10000 | 35 | 0 | 4 | 335.681 |
| UNSW | source | S_source_test | 10000 | 23 | 5424 | 3 | 331.849 |
| UNSW | source | S_train | 50000 | 28 | 20000 | 3 | 1962.44 |
| UNSW | source | S_prob_cal | 50000 | 7 | 10000 | 1 | 1783.208 |
| UNSW | source | S_competence | 50000 | 7 | 0 | 1 | 1730.509 |
| UNSW | source | S_source_test | 50000 | 4 | 35424 | 1 | 1751.1370000000002 |
| CIC | development | R_fit | 5000 | 2012 | 0 | 202 | 86.6815 |
| CIC | development | R_cal | 5000 | 1004 | 0 | 101 | 81.0535 |
| CIC | development | R_select | 5000 | 1007 | 529 | 101 | 76.673 |
| CIC | development | R_fit | 10000 | 1006 | 0 | 101 | 175.1415 |
| CIC | development | R_cal | 10000 | 502 | 0 | 51 | 162.122 |
| CIC | development | R_select | 10000 | 503 | 5529 | 51 | 153.194 |
| CIC | development | R_fit | 50000 | 201 | 10000 | 21 | 896.885 |
| CIC | development | R_cal | 50000 | 100 | 20000 | 10 | 806.36 |
| CIC | development | R_select | 50000 | 100 | 35529 | 10 | 764.266 |
| CIC | heldout | T_heldout | 5000 | 4023 | 529 | 403 | 81.042 |
| CIC | heldout | T_heldout | 10000 | 2011 | 5529 | 202 | 162.641 |
| CIC | heldout | T_heldout | 50000 | 402 | 15529 | 41 | 818.004 |
| BoT | development | R_fit | 5000 | 1694 | 0 | 170 | 0.6865000000000001 |
| BoT | development | R_cal | 5000 | 846 | 0 | 85 | 0.3745 |
| BoT | development | R_select | 5000 | 846 | 3808 | 85 | 0.336 |
| BoT | development | R_fit | 10000 | 847 | 0 | 85 | 1.667 |
| BoT | development | R_cal | 10000 | 423 | 0 | 43 | 0.891 |
| BoT | development | R_select | 10000 | 423 | 3808 | 43 | 0.998 |
| BoT | development | R_fit | 50000 | 169 | 20000 | 17 | 35.308 |
| BoT | development | R_cal | 50000 | 84 | 30000 | 9 | 27.032 |
| BoT | development | R_select | 50000 | 84 | 33808 | 9 | 37.0065 |
| BoT | heldout | T_heldout | 5000 | 3386 | 3808 | 339 | 0.511 |
| BoT | heldout | T_heldout | 10000 | 1693 | 3808 | 170 | 1.287 |
| BoT | heldout | T_heldout | 50000 | 338 | 33808 | 34 | 29.9465 |
| ToN | development | R_fit | 5000 | 2752 | 0 | 276 | 8.3855 |
| ToN | development | R_cal | 5000 | 1376 | 0 | 138 | 21.811 |
| ToN | development | R_select | 5000 | 1376 | 260 | 138 | 43.9945 |
| ToN | development | R_fit | 10000 | 1376 | 0 | 138 | 17.563499999999998 |
| ToN | development | R_cal | 10000 | 688 | 0 | 69 | 44.9995 |
| ToN | development | R_select | 10000 | 688 | 260 | 69 | 86.78450000000001 |
| ToN | development | R_fit | 50000 | 275 | 10000 | 28 | 115.549 |
| ToN | development | R_cal | 50000 | 137 | 30000 | 14 | 221.257 |
| ToN | development | R_select | 50000 | 137 | 30260 | 14 | 432.873 |
| ToN | heldout | T_heldout | 5000 | 5504 | 260 | 551 | 20.008499999999998 |
| ToN | heldout | T_heldout | 10000 | 2752 | 260 | 276 | 42.035 |
| ToN | heldout | T_heldout | 50000 | 550 | 20260 | 55 | 213.8555 |

## Validation and limitations

Passed: full local header/time/label consistency; sorting indices are exact permutations of original rows; each record belongs to exactly one partition per role; zero partition intersections; indivisible groups; conserved window/tail coverage; complete available predecessors; resampling blocks cover complete windows without crossing partitions.
Passed: synthetic long streams, equal-time boundaries, short tails, cross-partition predecessors, empty data, and small-group rounding. Feature groups remain 17/17/15 with 49 unique features; IP, absolute time, labels, and identities are excluded.
Partitions missing a required binary class: none.
Combinations with fewer than 20 resampling units (a gate only for relevant evaluation partitions): UNSW/source/S_prob_cal/5000, UNSW/source/S_competence/5000, UNSW/source/S_source_test/5000, UNSW/source/S_train/10000, UNSW/source/S_prob_cal/10000, UNSW/source/S_competence/10000, UNSW/source/S_source_test/10000, UNSW/source/S_train/50000, UNSW/source/S_prob_cal/50000, UNSW/source/S_competence/50000, UNSW/source/S_source_test/50000, CIC/development/R_cal/50000, CIC/development/R_select/50000, BoT/development/R_fit/50000, BoT/development/R_cal/50000, BoT/development/R_select/50000, ToN/development/R_cal/50000, ToN/development/R_select/50000.
No capture/session isolation evidence exists. Temporal groups and consecutive-window blocks do not guarantee statistical independence. Large time gaps are not treated as independent captures. Export/transmission delay was not simulated.
Development and heldout manifests are alternative roles for the same target domain across folds; their reuse is intentional. Within a fold, use only the role specified by fold_manifest.json; do not combine them for training.
Class counts support preparation audits only. Real-model future-feature perturbation, held-out-label perturbation, and paired CRE fitting have not been validated. Static predecessor checks do not establish those properties.
Configuration is not frozen. Training parameters, later sensitivity settings, gate rationale, prior outer-result exposure, and other freeze items still require registration. Missing training libraries are recorded in run_config.json; no dependencies were installed during this preparation.

## Artifacts and rerunning

Execution command:

```bash
/home/karri/miniconda3/envs/agentnids/bin/python /home/karri/projects/NIDSProject/round1_handoff_v1/round1_setup/prepare_data.py --data-root /home/karri/projects/NIDSProject/datasets --output /home/karri/projects/NIDSProject/outputs/round1_data_prep_v1
```

split_manifest.csv records half-open rank intervals; group_manifest.csv records group identities; window_manifest.csv records complete windows, predecessors, and initialization; tail_manifest.csv records exclusions; bootstrap_manifest.csv records resampling blocks.
Reconstruct member row indices with DOMAIN_sorted_original_rows.npy[rank_start:rank_end_exclusive], then extract features from original CSVs. A window's first/last original indices do not imply a contiguous interval in the original file.
Save source, configuration, and the original feature-list copy with each run. Record ordinary versions, times, sizes, modification times, and row counts; no hashing is performed.

## Next small-scale entry point (no training executed)

Run prepare_data.py --self-test, then use F1 development domains in this directory's fold_manifest.json and complete 10k windows in window_manifest.csv for small-scale reading/replay.
Training scripts had not yet been implemented at this preparation date. Register training configuration/dependencies before implementing S_train preprocessing/models, source calibration, development replay, and real perturbation checks. Preparation success does not authorize formal training or outer unblinding.

## On-disk validation and read-only small-scale entry point

Validation time: 2026-09-05T07:29:03.164165+00:00.
Reopened and validated 42,070 windows, 4,230 resampling blocks, and 13,155 group assignments: all passed.

Partition UTC availability ranges, including tail records:

| Dataset | Role | Partition | Earliest UTC | Latest UTC |
|---|---|---|---|---|
| UNSW | source | S_train | 2015-01-22T11:50:14.036000+00:00 | 2015-02-18T03:20:46.673000+00:00 |
| UNSW | source | S_prob_cal | 2015-02-18T03:20:46.771000+00:00 | 2015-02-18T06:53:19.270000+00:00 |
| UNSW | source | S_competence | 2015-02-18T06:53:19.277000+00:00 | 2015-02-18T10:15:46.739000+00:00 |
| UNSW | source | S_source_test | 2015-02-18T10:15:46.740000+00:00 | 2015-02-18T12:29:25.011000+00:00 |
| CIC | development | R_fit | 2018-02-14T12:28:07.705000+00:00 | 2018-02-21T21:03:56.013000+00:00 |
| CIC | development | R_cal | 2018-02-21T21:03:56.013000+00:00 | 2018-02-28T16:26:28.558000+00:00 |
| CIC | development | R_select | 2018-02-28T16:26:28.625000+00:00 | 2018-03-02T20:06:24.780000+00:00 |
| CIC | heldout | T_heldout | 2018-02-14T12:28:07.705000+00:00 | 2018-03-02T20:06:24.780000+00:00 |
| BoT | development | R_fit | 2018-05-15T00:29:11.932000+00:00 | 2018-06-04T04:30:33.139000+00:00 |
| BoT | development | R_cal | 2018-06-04T04:30:33.179000+00:00 | 2018-06-04T07:25:31.820000+00:00 |
| BoT | development | R_select | 2018-06-04T07:25:31.820000+00:00 | 2018-06-19T04:17:14.508000+00:00 |
| BoT | heldout | T_heldout | 2018-05-15T00:29:11.932000+00:00 | 2018-06-19T04:17:14.508000+00:00 |
| ToN | development | R_fit | 2019-04-23T14:09:50.618000+00:00 | 2019-04-25T20:46:15.634000+00:00 |
| ToN | development | R_cal | 2019-04-25T20:46:15.634000+00:00 | 2019-04-26T18:14:58.766000+00:00 |
| ToN | development | R_select | 2019-04-26T18:14:58.768000+00:00 | 2019-04-29T14:45:55.861000+00:00 |
| ToN | heldout | T_heldout | 2019-04-23T14:09:50.618000+00:00 | 2019-04-29T14:45:55.861000+00:00 |

Read-only smoke read passed: UNSW:source:10000:S_train:0, 10,000 source records, with group-array shapes 10000×17, 10000×17, and 10000×15.
This window has Benign=9426, Attack=574, and 512 nonfinite feature values. This checks one window's feature reading, not full-table feature quality.
Empty-field counts: {"SRC_TO_DST_SECOND_BYTES": 73}. After initial direct conversion encountered empty fields, the reader was changed to preserve NaN. No records were removed and no imputation values were fitted. Subsequently fit preprocessing only on S_train, then transform other partitions.
Infinity counts: {"SRC_TO_DST_SECOND_BYTES": 183, "DST_TO_SRC_SECOND_BYTES": 256}. This stage preserves original values. Apply registered nonfinite-value handling before training; this read PASS does not establish that features can be passed directly to models.

Rerunnable small-scale data entry point, without training:

```bash
/home/karri/miniconda3/envs/agentnids/bin/python /home/karri/projects/NIDSProject/round1_handoff_v1/round1_setup/verify_prepared_data.py --run-dir /home/karri/projects/NIDSProject/outputs/round1_data_prep_v1 --smoke-source-window
```

A full prepare_data.py rerun requires a new --output directory; the script refuses to overwrite an existing run.
