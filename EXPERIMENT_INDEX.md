# Follow-up experiment index

For the main P0–P7 study, use the [code guide](CODE_GUIDE.md) and [run guide](docs/RUN_GUIDE.md). For follow-up work, choose a question below and open its local README for code, inputs and primary results. The 29 runs are separate branches, not a required execution sequence.

Strict-K C1/C2 use 400 target labels to select ordinary thresholds; Variant B uses conformal prediction sets. Follow-up results are exploratory.

## Classification and ordinary threshold calibration

| ID | Question / local guide |
|---|---|
| E01 | [A: raw versus Platt classification diagnostics](experiments/stage_A/run_20260914_A1/README.md) |
| E02 | [B1: fixed entropy score for selective prediction](experiments/stage_B1/run_20260914_B1_01/README.md) |
| E03 | [B2: All49 entropy-only correctness model](experiments/stage_B2/run_20260914_B2_01/README.md) |
| E04 | [Strict-K C: 400-label target calibration preflight](experiments/strict_k_reply4/run_20260914_Cpreflight_01/README.md) |
| E05 | [Strict-K C1 versus C0: classification-threshold comparison](experiments/strict_k_reply4/run_20260915_C1_01/README.md) |
| E06 | [Strict-K C2 versus C0: acceptance-threshold comparison](experiments/strict_k_reply4/run_20260915_C2_01/README.md) |

## Conformal feasibility and failure analysis

| ID | Question / local guide |
|---|---|
| E07 | [Week 0 conformal feasibility preflight](experiments/week0_preflight/run_20260915T070731Z/README.md) |
| E08 | [Week 0 conformal feasibility pilot](experiments/week0_pilot/run_20260915T073452Z/README.md) |
| E09 | [Week 0 source-data and code audit](experiments/week0_audit/run_20260915T075634Z/README.md) |
| E10 | [Week 0 protocol and upstream lineage audit](experiments/week0_protocol_lineage_audit/run_20260915T125805Z/README.md) |
| E11 | [Week 0 probability and threshold anomaly analysis](experiments/week0_root_cause/run_20260916T035045Z/README.md) |

## Variant B, class support and temporal calibration

| ID | Question / local guide |
|---|---|
| E12 | [ToN Variant B mini pilot](experiments/variant_b_ton_pilot/run_20260916T085242Z/README.md) |
| E13 | [Independent review of ToN Variant B mini pilot](experiments/variant_b_ton_review/run_20260916T142121Z/README.md) |
| E14 | [ToN Variant B: selection versus relabeling errors](experiments/variant_b_error_decomposition/run_20260916T144124Z/README.md) |
| E15 | [CIC/BoT start-window class-support check](experiments/cic_bot_startup_support/run_20260917T014551Z/README.md) |
| E16 | [Independent review of CIC/BoT start-window support](experiments/cic_bot_startup_support_review/run_20260917T015900Z/README.md) |
| E17 | [ToN XGBoost-All49 Variant B threshold pilot](experiments/xgboost_variant_b_ton_pilot/run_20260918T070417Z/README.md) |
| E18 | [ToN XGBoost-All49 Variant B exploratory evaluation](experiments/xgboost_variant_b_ton_exploratory_eval/run_20260918T071352Z/README.md) |
| E19 | [ToN temporal-stratified 400-label calibration comparison](experiments/temporal_stratified_400_ton/run_20260918T131742Z/README.md) |
| E20 | [ToN raw errors versus confidence and calibrated scores](experiments/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/README.md) |
| E21 | [Outcome B: BoT three-detector and ToN MLP threshold studies](experiments/outcome_b_two_experiments/run_20260920T065121Z/README.md) |
| E22 | [BoT RF-All49 Variant B full evaluation](experiments/rf_bot_variant_b_full_evaluation/run_20260920T112946Z/README.md) |

## Local evidence and label correction

| ID | Question / local guide |
|---|---|
| E23 | [ToN/BoT local-evidence reliability pilot](experiments/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr/README.md) |
| E24 | [ToN local label-correction calibration feasibility](experiments/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/README.md) |
| E25 | [ToN label-correction pilot on reused evaluation data](experiments/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim/README.md) |

## Label refresh, memory and representation learning

| ID | Question / local guide |
|---|---|
| E26 | [ToN label recency: local vote versus small RF](experiments/ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd/README.md) |
| E27 | [ToN second target-label refresh at T2](experiments/ton_second_refresh_pilot_v1/run_20260929T122552Z_d10acc/README.md) |
| E28 | [ToN fixed-capacity old/recent memory pilot](experiments/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/README.md) |
| E29 | [ToN source-trained representation objectives: BCE, FCE, FSC](experiments/ton_representation_pilot_v1/run_20260929T164356Z_0a709e/README.md) |

Full file mappings are in [package_layout.json](configs/package_layout.json); execution gaps are in [known limitations](docs/KNOWN_LIMITATIONS.md).
