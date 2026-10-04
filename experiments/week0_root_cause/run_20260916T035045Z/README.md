# E11: Week 0 probability and threshold anomaly analysis

## Code

Execution files: [investigate.py](<investigate.py>) → [inspect_forest.py](<inspect_forest.py>) → [followup.py](<followup.py>) → [representatives.py](<representatives.py>) → [finalize_root_cause_analysis.py](<finalize_root_cause_analysis.py>) → [check_delivery.py](<check_delivery.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E11 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

No run-specific compact assets are retained. Static upstream references: [E09](<../../week0_audit/run_20260915T075634Z/README.md>), [E08](<../../week0_pilot/run_20260915T073452Z/README.md>), [E10](<../../week0_protocol_lineage_audit/run_20260915T125805Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [root_cause_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_root_cause/run_20260916T035045Z/root_cause_report.md>)
- [validation.json](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_root_cause/run_20260916T035045Z/validation.json>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/week0_root_cause/run_20260916T035045Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
