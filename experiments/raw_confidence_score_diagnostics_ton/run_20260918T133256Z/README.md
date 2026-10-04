# E20: ToN raw errors versus confidence and calibrated scores

## Code

Execution files: [diagnose.py](<diagnose.py>) → [validate_confidence_diagnostics.py](<validate_confidence_diagnostics.py>) → [build_confidence_diagnostics_report.py](<build_confidence_diagnostics_report.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E20 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z>). Static upstream references: [E19](<../../temporal_stratified_400_ton/run_20260918T131742Z/README.md>), [E12](<../../variant_b_ton_pilot/run_20260916T085242Z/README.md>), [E18](<../../xgboost_variant_b_ton_exploratory_eval/run_20260918T071352Z/README.md>), [E17](<../../xgboost_variant_b_ton_pilot/run_20260918T070417Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [diagnostic_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/diagnostic_report.md>)
- [raw_metrics.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/raw_metrics.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
