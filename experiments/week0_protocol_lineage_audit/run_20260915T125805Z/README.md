# E10: Week 0 protocol and upstream lineage audit

## Code

Execution files: [audit_protocol_lineage.py](<audit_protocol_lineage.py>) → [build_protocol_audit_report.py](<build_protocol_audit_report.py>) → [finalize_protocol_audit.py](<finalize_protocol_audit.py>) → [verify_delivery.py](<verify_delivery.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E10 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z>). Static upstream references: [E08](<../../week0_pilot/run_20260915T073452Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [audit_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/audit_report.md>)
- [audit_validation.json](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/audit_validation.json>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
