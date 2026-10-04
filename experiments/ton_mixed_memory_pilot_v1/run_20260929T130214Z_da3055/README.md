# E28: ToN fixed-capacity old/recent memory pilot

## Code

Execution files: [run_pilot.py](<src/run_pilot.py>).

Additional helpers/checks: [fixed_core.py](<src/fixed_core.py>), [frozen_functions.py](<src/frozen_functions.py>).

See [commands.sh](<logs/commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E28 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055>). Static upstream references: [E23](<../../reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr/README.md>), [E26](<../../ton_recency_learner_pilot_v1/run_20260929T120812Z_zIglWd/README.md>), [E27](<../../ton_second_refresh_pilot_v1/run_20260929T122552Z_d10acc/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [reports/RESULTS_ZH.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/reports/RESULTS_ZH.md>)
- [tables/policy_comparison.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/tables/policy_comparison.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
