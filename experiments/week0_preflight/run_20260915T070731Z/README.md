# E07: Week 0 conformal feasibility preflight

## Code

Execution files: [load_model_check.py](<load_model_check.py>) → [run_preflight.py](<run_preflight.py>) → [register_pilot_protocol.py](<register_pilot_protocol.py>) → [finalize_preflight.py](<finalize_preflight.py>) → [verify_deliverables.py](<verify_deliverables.py>).

Additional helpers/checks: [implementation.py](<implementation.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E07 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [preflight_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z/preflight_report.md>)
- [calibration_data_check.json](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z/calibration_data_check.json>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
