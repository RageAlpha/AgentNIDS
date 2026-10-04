# E01: A: raw versus Platt classification diagnostics

## Code

Execution files: [run_stage_a.py](<run_stage_a.py>) → [audit_source_and_clipping.py](<audit_source_and_clipping.py>) → [plot_stage_a.py](<plot_stage_a.py>) → [build_report.py](<build_report.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E01 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/stage_A/run_20260914_A1>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [stage_A_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/stage_A/run_20260914_A1/stage_A_report.md>)
- [detector_raw_platt_metrics.csv](<../../../inputs/outputs/post_p7_diagnostics/stage_A/run_20260914_A1/detector_raw_platt_metrics.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/stage_A/run_20260914_A1>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
