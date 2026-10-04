# E21: Outcome B: BoT three-detector and ToN MLP threshold studies

## Code

Execution files: [run_outcome_b_experiments.py](<run_outcome_b_experiments.py>) → [finalize_outcome_b_run.py](<finalize_outcome_b_run.py>).

Additional helpers/checks: [set_logic_checks.py](<set_logic_checks.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E21 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z>). Static upstream references: [E05](<../../strict_k_reply4/run_20260915_C1_01/README.md>), [E12](<../../variant_b_ton_pilot/run_20260916T085242Z/README.md>), [E17](<../../xgboost_variant_b_ton_pilot/run_20260918T070417Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [completion_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z/completion_report.md>)
- [bot/threshold_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z/bot/threshold_report.md>)
- [ton_mlp/threshold_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z/ton_mlp/threshold_report.md>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
