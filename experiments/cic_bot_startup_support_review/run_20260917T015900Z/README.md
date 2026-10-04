# E16: Independent review of CIC/BoT start-window support

## Code

Execution files: [review.py](<review.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E16 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/cic_bot_startup_support_review/run_20260917T015900Z>). Static upstream references: [E15](<../../cic_bot_startup_support/run_20260917T014551Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [review_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/cic_bot_startup_support_review/run_20260917T015900Z/review_report.md>)
- [differences.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/cic_bot_startup_support_review/run_20260917T015900Z/differences.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/cic_bot_startup_support_review/run_20260917T015900Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
