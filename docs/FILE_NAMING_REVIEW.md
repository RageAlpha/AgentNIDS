# File naming review and English delivery

Date: 2026-10-04. This review covers the professor code package.

The two language versions of the professor briefing are consolidated into one canonical English document.

## Changes

| Previous review path | Current review path |
|---|---|
| `docs/Professor_Progress_Report_After_P7_EN.md` | `docs/Professor_Progress_Report_After_P7.md` |
| `experiments/outcome_b_two_experiments/run_20260920T065121Z/experiment.py` | `experiments/outcome_b_two_experiments/run_20260920T065121Z/run_outcome_b_experiments.py` |
| `experiments/outcome_b_two_experiments/run_20260920T065121Z/finalize.py` | `experiments/outcome_b_two_experiments/run_20260920T065121Z/finalize_outcome_b_run.py` |
| `experiments/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/audit.py` | `experiments/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/validate_confidence_diagnostics.py` |
| `experiments/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/report.py` | `experiments/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/build_confidence_diagnostics_report.py` |
| `experiments/stage_B1/run_20260914_B1_01/report_b1.py` | `experiments/stage_B1/run_20260914_B1_01/build_entropy_selection_report.py` |
| `experiments/stage_B2/run_20260914_B2_01/report_b2.py` | `experiments/stage_B2/run_20260914_B2_01/build_correctness_model_report.py` |
| `experiments/strict_k_reply4/run_20260915_C2_01/prepare.py` | `experiments/strict_k_reply4/run_20260915_C2_01/prepare_acceptance_calibration.py` |
| `experiments/temporal_stratified_400_ton/run_20260918T131742Z/experiment.py` | `experiments/temporal_stratified_400_ton/run_20260918T131742Z/run_temporal_calibration.py` |
| `experiments/variant_b_error_decomposition/run_20260916T144124Z/analyze.py` | `experiments/variant_b_error_decomposition/run_20260916T144124Z/analyze_error_decomposition.py` |
| `experiments/variant_b_error_decomposition/run_20260916T144124Z/finalize.py` | `experiments/variant_b_error_decomposition/run_20260916T144124Z/finalize_error_decomposition.py` |
| `experiments/variant_b_ton_pilot/run_20260916T085242Z/pilot.py` | `experiments/variant_b_ton_pilot/run_20260916T085242Z/run_variant_b_pilot.py` |
| `experiments/week0_pilot/run_20260915T073452Z/calibrate.py` | `experiments/week0_pilot/run_20260915T073452Z/calibrate_conformal_threshold.py` |
| `experiments/week0_pilot/run_20260915T073452Z/prepare.py` | `experiments/week0_pilot/run_20260915T073452Z/prepare_pilot_inputs.py` |
| `experiments/week0_pilot/run_20260915T073452Z/report.py` | `experiments/week0_pilot/run_20260915T073452Z/build_pilot_report.py` |
| `experiments/week0_preflight/run_20260915T070731Z/finalize.py` | `experiments/week0_preflight/run_20260915T070731Z/finalize_preflight.py` |
| `experiments/week0_preflight/run_20260915T070731Z/write_protocol.py` | `experiments/week0_preflight/run_20260915T070731Z/register_pilot_protocol.py` |
| `experiments/week0_protocol_lineage_audit/run_20260915T125805Z/audit.py` | `experiments/week0_protocol_lineage_audit/run_20260915T125805Z/audit_protocol_lineage.py` |
| `experiments/week0_protocol_lineage_audit/run_20260915T125805Z/finalize.py` | `experiments/week0_protocol_lineage_audit/run_20260915T125805Z/finalize_protocol_audit.py` |
| `experiments/week0_protocol_lineage_audit/run_20260915T125805Z/write_report.py` | `experiments/week0_protocol_lineage_audit/run_20260915T125805Z/build_protocol_audit_report.py` |
| `experiments/week0_root_cause/run_20260916T035045Z/finish.py` | `experiments/week0_root_cause/run_20260916T035045Z/finalize_root_cause_analysis.py` |
| `experiments/xgboost_variant_b_ton_pilot/run_20260918T070417Z/calibrate.py` | `experiments/xgboost_variant_b_ton_pilot/run_20260918T070417Z/calibrate_variant_b_thresholds.py` |
| `docs/OutcomeB_2Expirements/OutcomeB_Two_Experiments_Plan_ZH.md` | `docs/outcome_b_two_experiments/OutcomeB_Two_Experiments_Plan_EN.md` |
| `docs/Reliability_LocalEvidence_Pilot_20260929/01_Experiment_Plan_ZH.md` | `docs/Reliability_LocalEvidence_Pilot_20260929/01_Experiment_Plan_EN.md` |
| `docs/ToN_ActionCorrection_Pilot_20260929/01_Plan_and_Audit_ZH.md` | `docs/ToN_ActionCorrection_Pilot_20260929/01_Plan_and_Audit_EN.md` |
| `docs/Ton_variant/ToN_VariantB_MiniPilot_Plan_ReplyLed_ZH.md` | `docs/ton_variant_b/ToN_VariantB_MiniPilot_Plan_ReplyLed_EN.md` |

## Naming conventions

- `run_*`: execute an experiment or pilot.
- `prepare_*`: assemble inputs or configuration.
- `calibrate_*`: calculate and freeze calibration thresholds.
- `build_*_report`: produce an experiment report.
- `validate_*`, `check_*`, `audit_*`: verify results, inputs or protocol lineage.
- `finalize_*`: complete a run and write its final delivery/validation records.

The main `p2.py` through `p7.py` names follow the registered stage identifiers. Their responsibilities appear in CODE_GUIDE.md and EXPERIMENT_INDEX.md. Timestamped run directories and research identifiers preserve experiment identity.

## Execution and provenance

Canonical source files have clearer names for review. `configs/package_layout.json` maps them back to their original execution paths. `scripts/workspace.py` stages those original names into a fresh workspace, so saved commands and imports continue to use the expected layout. New modules are copied separately without creating duplicate aliases for mapped files.

The package documentation, source comments, report templates, runtime messages, and descriptive configuration/input text have been converted to English. Scientific numeric configuration and existing binary inputs are preserved. Historical source/text identity checks refer to original files; translations change file bytes. Use fresh registration for new runs and the history companion for exact historical audits. Existing frozen hash values were not rewritten.

Full GPU experiments are not claimed to have been rerun. The compact CPU example and structural checks are documented separately.
