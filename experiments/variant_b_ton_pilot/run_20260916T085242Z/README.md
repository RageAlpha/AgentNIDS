# E12: ToN Variant B mini pilot

## Code

Execution files: [run_variant_b_pilot.py](<run_variant_b_pilot.py>) → [check_delivery.py](<check_delivery.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E12 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [stage_report.md](<../../../inputs/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z/stage_report.md>)
- [evaluation_metrics.csv](<../../../inputs/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z/evaluation_metrics.csv>)
- [gate_assessment.md](<../../../inputs/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z/gate_assessment.md>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
