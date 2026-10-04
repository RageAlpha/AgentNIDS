# E24: ToN local label-correction calibration feasibility

## Code

Execution files: [run_calibration.py](<src/run_calibration.py>).

See [commands.sh](<logs/commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python -m pip install -r requirements-example.txt
python scripts/workspace.py example
```

This is the included CPU route.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [reports/summary_ZH.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/reports/summary_ZH.md>)
- [calibration/gate_feasibility.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/calibration/gate_feasibility.csv>)
- [calibration/selected_rules.json](<../../../inputs/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/calibration/selected_rules.json>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
