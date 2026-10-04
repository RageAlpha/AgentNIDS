# CIC / BoT Start-up category support quantity checks

According to Professor reply 5 section. Only the existing rank [0,100000) was estimated, not using the old 400 sample or the full range category ratio. ToN Variant B No-go remains unchanged, this time not representing permission to continue the full Phase2.

| domain | Total start-up period | Benign | Benign ratio | Attack | Attack ratio | 200+200 | Subsequent qualification by reply |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIC | 100000 | 100000 | 100.000% | 0 | 0.000% | False | The current 10 window does not meet the Mondrian quantitative requirements; Variant A only in reply if the field continues. |
| BoT | 100000 | 9071 | 9.071% | 90929 | 90.929% | True | meet the quantitative conditions of the 200+200 category; Quantitative only, not authorized to operate Mondrian or Week1 |

## Location of 200

Both rank and original_row start with 0; The scanning bar number starts at 1. The original number does not contain the CSV headings. NA indicates a lack of 200 during the fixed start-up period.

| domain | Categories | Actual support | Article 200 rank | Original CSV line | The original window_id | Number of scans |
| --- | --- | --- | --- | --- | --- | --- |
| CIC | Benign | 100000 | 199 | 277 | CIC:heldout:10000:T_heldout:0 | 200 |
| CIC | Attack | 0 | NA | NA | NA | NA |
| BoT | Benign | 9071 | 14863 | 16563349 | BoT:heldout:10000:T_heldout:1 | 14864 |
| BoT | Attack | 90929 | 226 | 16771984 | BoT:heldout:10000:T_heldout:0 | 227 |

| domain | The first 200+200 scan required to align |
| --- | --- |
| CIC | NA |
| BoT | 14864 |

This time only records the number with the 200 position, and no new calibration samples have been registered. Even before reaching the 200, complete statistics of all 100,000 items are still available.

## Original 10 window count

| domain | window_id | rank_start | rank_end_exclusive | Total number | Benign | Attack |
| --- | --- | --- | --- | --- | --- | --- |
| CIC | CIC:heldout:10000:T_heldout:0 | 0 | 10000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:1 | 10000 | 20000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:2 | 20000 | 30000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:3 | 30000 | 40000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:4 | 40000 | 50000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:5 | 50000 | 60000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:6 | 60000 | 70000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:7 | 70000 | 80000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:8 | 80000 | 90000 | 10000 | 10000 | 0 |
| CIC | CIC:heldout:10000:T_heldout:9 | 90000 | 100000 | 10000 | 10000 | 0 |
| BoT | BoT:heldout:10000:T_heldout:0 | 0 | 10000 | 10000 | 155 | 9845 |
| BoT | BoT:heldout:10000:T_heldout:1 | 10000 | 20000 | 10000 | 100 | 9900 |
| BoT | BoT:heldout:10000:T_heldout:2 | 20000 | 30000 | 10000 | 3288 | 6712 |
| BoT | BoT:heldout:10000:T_heldout:3 | 30000 | 40000 | 10000 | 101 | 9899 |
| BoT | BoT:heldout:10000:T_heldout:4 | 40000 | 50000 | 10000 | 101 | 9899 |
| BoT | BoT:heldout:10000:T_heldout:5 | 50000 | 60000 | 10000 | 70 | 9930 |
| BoT | BoT:heldout:10000:T_heldout:6 | 60000 | 70000 | 10000 | 147 | 9853 |
| BoT | BoT:heldout:10000:T_heldout:7 | 70000 | 80000 | 10000 | 65 | 9935 |
| BoT | BoT:heldout:10000:T_heldout:8 | 80000 | 90000 | 10000 | 767 | 9233 |
| BoT | BoT:heldout:10000:T_heldout:9 | 90000 | 100000 | 10000 | 4277 | 5723 |

## In order, identity and original labels verification

Strict-K pre-examination is used only to confirm the range and window of the start-up period; Both the old 400 random sample and the old 20/ class thresholds are not used in this inspection. The 10k window with the original heldout/T_heldout is used, with the previous 10 window continuously covering the [0,100000). label.npy has been ranked according to availability rank and read [:100000] directly, without repeating the original mapping line. availability is FLOW_END_MILLISECONDS with the same time line sequence along the original CSV data line; Windows are not rearranged or expanded.

- CIC: check the 100,000 item, 10 is complete windows; Two-way counting, window-by-window and overall, window endpoints are consistent with historical pre-checks, labels has no illegal / missing value, and identity is the only start-up period. Original CSV independent labels =PASS verification, only analyzing the original correspondence record for the 100,000 start-up period; labels difference =0, original end time and rank identity mapping difference =0.
- BoT: check the 100,000 item, 10 is complete windows; Two-way counting, window-by-window and overall, window endpoints are consistent with historical pre-checks, labels has no illegal / missing value, and identity is the only start-up period. Original CSV independent labels =PASS verification, only analyzing the original correspondence record for the 100,000 start-up period; labels difference =0, original end time and rank identity mapping difference =0.

The original CSV uses a sequence byte scan, skips the non-target line and does not parse its labels; Therefore, BoT passes through a larger CSV threshold, but no labels fields are statistically recorded or read for subsequent evaluation. Select the number of rows and finish time mapping simultaneously. labels is read directly from the original CSV Label, 0=Benign, 1=Attack, and checks whether the Benign is crossed with the Attack field. Prior to Article 200, both Articles 199 and 200 containing them have been validated; 200 notation NA does not exist.

## Input and Border


CIC：
- labels: `/home/karri/projects/NIDSProject/outputs/feature_store_v2/CIC_label.npy`
- mapping: `/home/karri/projects/NIDSProject/outputs/round1_data_prep_v1/CIC_sorted_original_rows.npy`
- times: `/home/karri/projects/NIDSProject/outputs/round1_data_prep_v1/CIC_sorted_end_ms.npy`
- raw_csv: `/home/karri/projects/NIDSProject/datasets/NF-CICIDS2018-v3.csv`
- domain_manifest: `/home/karri/projects/NIDSProject/outputs/feature_store_v2/CIC_manifest.json`

BoT：
- labels: `/home/karri/projects/NIDSProject/outputs/feature_store_v2/BoT_label.npy`
- mapping: `/home/karri/projects/NIDSProject/outputs/round1_data_prep_v1/BoT_sorted_original_rows.npy`
- times: `/home/karri/projects/NIDSProject/outputs/round1_data_prep_v1/BoT_sorted_end_ms.npy`
- raw_csv: `/home/karri/projects/NIDSProject/datasets/NF-BoT-IoT-v3.csv`
- domain_manifest: `/home/karri/projects/NIDSProject/outputs/feature_store_v2/BoT_manifest.json`

The window manifest: `/home/karri/projects/NIDSProject/outputs/round1_data_prep_v1/window_manifest.csv`. Pre-examination range: `/home/karri/projects/NIDSProject/outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/evaluation_scope.csv`; Protocol: `/home/karri/projects/NIDSProject/outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/protocol_draft.json`Sorting specification: `outputs/round1_data_prep_v1/round1_config.json`; labels generated: `src/nids/data/feature_store.py` is read as raw_y and sorted_y is logically written as order. All CSV results record absolute input paths, rank range and original line definition.

the number of categories is not sufficient to be calibration with time representation, is not equal to Mondrian effective and is not authorized to start Week1; When insufficient Variant A only is a conditional subsequent branch given by the professor, this time it does not run Variant A. No model or probability reading, no fit threshold, no modification of historical documents, no hash or SHA256 computational verification. When it is finished, stop and do not send messages to the professor.
