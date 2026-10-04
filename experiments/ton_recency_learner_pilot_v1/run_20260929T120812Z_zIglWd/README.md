# E26: ToN label recency: local vote versus small RF

## Code

Execution files: [run_pilot.py](<src/run_pilot.py>).

Additional helpers/checks: [fixed_core.py](<src/fixed_core.py>).

See [commands.sh](<logs/commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E26 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd>). Static upstream references: [E23](<../../reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr/README.md>), [E24](<../../ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [reports/RESULTS_ZH.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd/reports/RESULTS_ZH.md>)
- [tables/policy_comparison.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd/tables/policy_comparison.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
