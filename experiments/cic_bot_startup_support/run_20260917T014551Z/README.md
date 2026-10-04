# E15: CIC/BoT start-window class-support check

## Code

Execution files: [check_support.py](<check_support.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E15 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/cic_bot_startup_support/run_20260917T014551Z>). Static upstream references: [E04](<../../strict_k_reply4/run_20260914_Cpreflight_01/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [startup_support_report.md](<../../../inputs/outputs/post_p7_diagnostics/cic_bot_startup_support/run_20260917T014551Z/startup_support_report.md>)
- [startup_class_support.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/cic_bot_startup_support/run_20260917T014551Z/startup_class_support.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/cic_bot_startup_support/run_20260917T014551Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
