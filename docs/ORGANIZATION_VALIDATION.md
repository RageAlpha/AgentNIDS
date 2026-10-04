# Organization validation — 2026-10-04

> Historical check record for the organization snapshot before English translation and naming changes. Byte-equality claims below refer to that snapshot. See [current delivery validation](ENGLISH_DELIVERY_VALIDATION.md) for the revised package.

The checks below were executed during this reorganization on Windows with Python 3.12.7, NumPy 1.26.4 and pandas 2.2.2. No full GPU experiment was started.

| Check | Observed result |
|---|---|
| Original research code preservation | All 194 original Python/Shell files are present and byte-equal to the pre-organization run copy. |
| Retained input preservation | All 302 selected compact inputs are byte-equal to their originals. |
| Python syntax | 157 Python files parsed successfully, including the new workspace tool. |
| JSON readability | All JSON files in the review package parsed successfully. |
| Fresh main workspace | Created with rebased project/data/output paths. Scientific execution-config fields were structurally equal after excluding directory/interpreter fields. |
| Existing-path protection | A second preparation at an existing workspace path was rejected. |
| Historical-state separation | E29 workspace with other-run history contained its source but no own complete.json or actions_complete.json; another run's historical report was copied. |
| Real compact calibration | PASS without the history companion or raw CSVs. E1-local accepted 52 benign candidates with 2 errors and 13 attack candidates with 0 errors. |

The real example used the original algorithm and four retained upstream calibration files. It verifies organization/path compatibility for that route. Reused calibration labels are not an independent outer test or a future-risk guarantee.

The research Shell sources were preserved byte-for-byte; no new bash syntax check was performed on this Windows host. All 29 experiments, GPU imports/training, full P0–P7 reconstruction and complete dynamic dependency closure remain unverified here. Missing large assets and two confirmed pre-existing missing document dependencies are documented in KNOWN_LIMITATIONS.md.

Historical validation claims remain in the separate snapshot and are not relabeled as tests executed during this reorganization. Runtime roots are changed only in generated working copies; saved protocol metadata is not silently rewritten.
