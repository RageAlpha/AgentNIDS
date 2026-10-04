# E25: ToN label-correction pilot on reused evaluation data

## Code

Execution files: [run_experiment.py](<src/run_experiment.py>).

Additional helpers/checks: [baselines.py](<src/baselines.py>), [calibration_gates.py](<src/calibration_gates.py>), [fixed_metrics.py](<src/fixed_metrics.py>).

See [commands.sh](<logs/commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E25 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim>). Static upstream references: [E23](<../../reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [reports/summary_ZH.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim/reports/summary_ZH.md>)
- [tables/comparison.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim/tables/comparison.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
