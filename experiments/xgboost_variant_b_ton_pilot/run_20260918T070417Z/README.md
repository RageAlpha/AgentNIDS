# E17: ToN XGBoost-All49 Variant B threshold pilot

## Code

Execution files: [calibrate_variant_b_thresholds.py](<calibrate_variant_b_thresholds.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E17 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/xgboost_variant_b_ton_pilot/run_20260918T070417Z>). Static upstream references: [E12](<../../variant_b_ton_pilot/run_20260916T085242Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [threshold_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/xgboost_variant_b_ton_pilot/run_20260918T070417Z/threshold_report.md>)
- [threshold_freeze.json](<../../../inputs/outputs/post_p7_diagnostics/xgboost_variant_b_ton_pilot/run_20260918T070417Z/threshold_freeze.json>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/xgboost_variant_b_ton_pilot/run_20260918T070417Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
