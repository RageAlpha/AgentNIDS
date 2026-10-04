# E09: Week 0 source-data and code audit

## Code

Execution files: [prepare_audit.py](<prepare_audit.py>) → [replay.py](<replay.py>) → [csv_reconstruction.py](<csv_reconstruction.py>) → [distribution_and_numerics.py](<distribution_and_numerics.py>) → [inference_audit.py](<inference_audit.py>) → [statistical_audit.py](<statistical_audit.py>) → [finalize_audit.py](<finalize_audit.py>) → [check_deliverables.py](<check_deliverables.py>).

Additional helpers/checks: [audit_common.py](<audit_common.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E09 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [audit_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z/audit_report.md>)
- [audit_validation.json](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z/audit_validation.json>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
