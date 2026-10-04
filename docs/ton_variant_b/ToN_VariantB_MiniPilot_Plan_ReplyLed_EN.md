# ToN Variant B mini-pilot execution checklist

## 1. Scope and inputs

Run under `/home/karri/projects/NIDSProject`, using frozen RF-All49 and existing Raw probability caches. Execute only the ToN Variant B mini-pilot.

Authority: Reply section4 (referenced artifact not included). Apply its Step2 directly for thresholds; this checklist does not repeat calibration instructions.

| Input | Project-relative path |
|---|---|
| RF-All49 model | `outputs/source_v2/models/rf_All49/model.pkl` |
| ToN probability cache | `outputs/prediction_cache_v2/ToN/rf_All49.npy` |
| Source probability cache | `outputs/prediction_cache_v2/UNSW/rf_All49.npy` |
| Labels | `outputs/feature_store_v2/{ToN,UNSW}_label.npy` |
| Original row indices | `outputs/round1_data_prep_v1/{ToN,UNSW}_sorted_original_rows.npy` |
| Window manifest | `outputs/round1_data_prep_v1/window_manifest.csv` |

Use raw_probability as Attack probability; labels0=Benign and1=Attack. Retain row order, features and windows.

No hashes/SHA256, retraining, Platt calibration, additional models/domains or Week1 execution. Save all new results separately and preserve old artifacts.

## 2. Execution order

1. Read the reply/assets and record input paths, class meanings and evaluation range.
2. Chronologically scan ToN’s first10 complete windows, collecting the earliest200 Benign and200 Attack records. Stop when both are complete; save original row, rank, label, window and actual scanned count.
3. Do not reuse/rebalance C1/C2 samples. If either class has fewer than200 among100,000 records, report support and stop without enlarging startup.
4. Apply reply section4 Step2; save its five required values and registration time before evaluation. Do not change samples or thresholds after results.
5. Generate Variant B prediction sets over the specified ToN range and source test using the same threshold pair.
6. Compute overall/four-segment results and same-range Raw all-accept baseline; finish statistics/go-no-go reporting and stop.

## 3. Evaluation range

All rank intervals are left closed/right open; retain original numbering.

| Use | Rank range | Count |
|---|---|---:|
| ToN startup, entirely excluded | [0,100000) | 100,000 records |
| Overall ToN evaluation | [100000,27520000) | 27,420,000 records;2,742 windows |
| Segment1 | [100000,6960000) | 686 windows |
| Segment2 | [6960000,13820000) | 686 windows |
| Segment3 | [13820000,20670000) | 685 windows |
| Segment4 | [20670000,27520000) | 685 windows |
| Source test | [2130000,2365424) | 235,424 records |

Continue excluding ToN’s260-record tail and retaining the original5,424-record source-test tail. Segments follow C1, not old Week0’s688 windows×4. Completing samples early does not allow earlier evaluation.

## 4. Prediction and metrics

### 4.1 Generate sets

Let `p=raw_probability` and read saved `q_B` and `q_A`:

```python
benign_in_set = p <= q_B
attack_in_set = (1.0 - p) <= q_A
```

- Benign only: accept and predict Benign.
- Attack only: accept and predict Attack.
- Neither: empty set, reject.
- Both: two-element set, abstain.

Set construction does not read evaluation labels. Compare the singleton label against truth for accepted errors; independently classify the Raw baseline at `p >= 0.5`.

When reusing old code, remove the assertion that singleton labels must equal Raw classifications. Raw errors on an accepted subset are not Variant B error counts. Change only this run’s script copy.

### 4.2 Required metrics

Report the first8 metrics below overall and for four segments, saving counts, denominators and ratios:

| Metric | Definition |
|---|---|
| `{Benign}` singleton fraction | These sets / evaluation records |
| `{Attack}` singleton fraction | These sets / evaluation records |
| Empty-set fraction | Empty sets / evaluation records |
| Two-element-set fraction | Two-element sets / evaluation records |
| `{Benign}` singleton error | True Attack within these sets / these sets |
| `{Attack}` singleton error | True Benign within these sets / these sets |
| Correctly accepted attacks | TP: true Attack with set`{Attack}` |
| Correctly accepted attack fraction | TP / all true Attack in the same range |
| Source-test Benign class coverage | True source Benign whose set contains Benign / true source Benign |
| Source-test Attack class coverage | True source Attack whose set contains Attack / true source Attack |

Report source metrics separately, using this round’s ToN thresholds without refitting source thresholds.

For go/no-go also report:

- Total acceptance: all singletons / evaluation records.
- Overall singleton error: wrong singletons / all singletons.
- Same-range Raw all-accept error: Raw errors / evaluation records.
- Error improvement: Raw error minus singleton error, in percentage points.
- Each segment’s TP, attack-detection fraction and share of total TP.

Compute Raw after excluding startup over exactly the same range; do not copy old Week0 full-range error. Undefined ratios are NA, not0.

### 4.3 Statistics and checks

- Follow the roadmap overall:10 windows/block,2,000 paired block-bootstrap replicates;275 blocks, with2 windows in the last.
- Use PCG64 seed20260915 and save this range’s sampling matrix. Each replicate calculates Raw risk, singleton risk and their difference; report95% percentile intervals and valid counts.
- Four segments use descriptive statistics without new significance gates.
- Check200 samples/class, unique original rows and disjoint calibration/evaluation.
- Check four set counts sum to evaluation size, segment counts sum to overall, and singleton confusion counts sum to accepts.
- Check asymmetric thresholds, equality, empty/two-element sets and zero acceptance. Singleton labels may differ from Raw.

## 5. Decision and stopping

Document evidence for each reply criterion:

**Go: both required**

1. Nonzero correctly accepted attacks in at least2 segments, without the outcome being mainly driven by one segment; display all four segments for judgment.
2. Overall singleton error is materially lower than same-range Raw all-accept error.

**No-go: either sufficient**

1. Total correctly accepted attacks=0.
2. Overall singleton error equals/exceeds Raw all-accept error.

With full abstention, risk is NA but TP=0, so the first no-go criterion applies. TP in only1 segment does not establish Go. If material improvement or cross-segment consistency is unclear, report values for professor judgment; do not invent a numerical threshold.

After No-go, report and stop without changing samples, thresholds or startup. After passing, report readiness for Week1; this script does not launch it automatically.

## 6. Outputs

Create a new directory:

`outputs/post_p7_diagnostics/variant_b_ton_pilot/run_<UTC_timestamp>/`

Save:

- `protocol.json`: paths, fixed rules, range, four segments and registration time.
- `calibration_sample_manifest.csv`: row, rank, label and window for400 records.
- `collection_summary.json`: class counts, stopping position and scanned rows; explain insufficient support.
- `threshold_freeze.json`: reply’s five values, computation settings and registration time.
- `evaluation_metrics.csv`: overall ToN metrics, Raw baseline and differences.
- `temporal_metrics.csv`: corresponding four-segment metrics.
- `source_class_coverage.csv`: source-test coverage for both classes.
- `bootstrap_intervals.csv`, sampling matrix and necessary window aggregates.
- `validation.json`, `stage_report.md`, `gate_assessment.md`, scripts, commands and logs.

Do not duplicate full caches, generate large per-flow trajectories or output file hashes.
