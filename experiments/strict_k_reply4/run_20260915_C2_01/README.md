# E06: Strict-K C2 versus C0: acceptance-threshold comparison

## Code

Execution files: [prepare_acceptance_calibration.py](<prepare_acceptance_calibration.py>) → [select_acceptance.py](<select_acceptance.py>) → [evaluate.py](<evaluate.py>) → [verify_candidates.py](<verify_candidates.py>) → [report_and_validate.py](<report_and_validate.py>).

Additional helpers/checks: [common.py](<common.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E06 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01>). Static upstream references: [E04](<../run_20260914_Cpreflight_01/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [stage_C2_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/stage_C2_report.md>)
- [selective_metrics.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/selective_metrics.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
