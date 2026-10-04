# E02: B1: fixed entropy score for selective prediction

## Code

Execution files: [run_b1.py](<run_b1.py>) → [plot_b1.py](<plot_b1.py>) → [build_entropy_selection_report.py](<build_entropy_selection_report.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E02 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01>). Static upstream references: [E01](<../../stage_A/run_20260914_A1/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [stage_B1_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/stage_B1_report.md>)
- [overall_selective_metrics.csv](<../../../inputs/outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/overall_selective_metrics.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
