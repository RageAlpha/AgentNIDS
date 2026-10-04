# E14: ToN Variant B: selection versus relabeling errors

## Code

Execution files: [analyze_error_decomposition.py](<analyze_error_decomposition.py>) → [finalize_error_decomposition.py](<finalize_error_decomposition.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E14 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

[Retained parameters and cross-run assets](<../../../inputs/outputs/post_p7_diagnostics/variant_b_error_decomposition/run_20260916T144124Z>). Static upstream references: [E12](<../../variant_b_ton_pilot/run_20260916T085242Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [decomposition_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_error_decomposition/run_20260916T144124Z/decomposition_report.md>)
- [risk_decomposition.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_error_decomposition/run_20260916T144124Z/risk_decomposition.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_error_decomposition/run_20260916T144124Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
