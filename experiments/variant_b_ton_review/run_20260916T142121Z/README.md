# E13: Independent review of ToN Variant B mini pilot

## Code

Execution files: [upstream_review.py](<upstream_review.py>) → [reference_review.py](<reference_review.py>) → [finalize_review.py](<finalize_review.py>).

See [commands.sh](<commands.sh>) for the exact sequence, arguments, phases and logs.

## Run

From the package root, after reading the [run guide](<../../../docs/RUN_GUIDE.md>):

```bash
python scripts/workspace.py prepare --experiment E13 --workspace /absolute/path/to/new-work --with-history
```

Preparation stages files. Supply upstream assets and the run configuration, then follow the recorded steps in that workspace.

## Inputs

No run-specific compact assets are retained. Static upstream references: [E04](<../../strict_k_reply4/run_20260914_Cpreflight_01/README.md>), [E05](<../../strict_k_reply4/run_20260915_C1_01/README.md>), [E12](<../../variant_b_ton_pilot/run_20260916T085242Z/README.md>). Additional data/models may be required; see [known limitations](<../../../docs/KNOWN_LIMITATIONS.md>).

## Results to read first

- [review_report.md](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_ton_review/run_20260916T142121Z/review_report.md>)
- [comparison_summary.csv](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_ton_review/run_20260916T142121Z/comparison_summary.csv>)

Links open saved evidence; fresh runs write the listed relative names under their run folder. Other saved outputs: [history folder](<../../../../NIDSProject_Professor_History_20261004/outputs/post_p7_diagnostics/variant_b_ton_review/run_20260916T142121Z>). Full file mappings: [package_layout.json](<../../../configs/package_layout.json>).
