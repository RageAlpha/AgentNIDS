# E08: Week 0 conformal feasibility pilot

## Code

Execution files: [prepare_pilot_inputs.py](<prepare_pilot_inputs.py>) → [calibrate_conformal_threshold.py](<calibrate_conformal_threshold.py>) → [evaluate.py](<evaluate.py>) → [bootstrap.py](<bootstrap.py>) → [verify.py](<verify.py>) → [calibration_audit.py](<calibration_audit.py>) → [build_pilot_report.py](<build_pilot_report.py>) → [check_deliverables.py](<check_deliverables.py>).

Additional helpers/checks: [common.py](<common.py>), [implementation.py](<implementation.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E08 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z>). Static upstream references: [E01](<../../stage_A/run_20260914_A1/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [stage_Week0_pilot_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z/stage_Week0_pilot_report.md>)
- [overall_metrics.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z/overall_metrics.csv>)
- [gate_assessment.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z/gate_assessment.md>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
