# English delivery and naming validation

Date:2026-10-04. Checks performed on the revised professor code package on Windows.

| Check | Result |
|---|---|
| Source completeness | All194 original research Python/Shell sources retain manifest destinations. |
| Python syntax/structure | All157 Python files parse. Original algorithms,expressions and numeric constants are structurally unchanged. The workspace helper adds a filter preventing duplicate aliases for renamed sources. |
| Configuration/data | All159 JSONs and61 CSVs parse. Scientific numeric values,keys and nontranslated cells are preserved;delivery-manifest fields are updated for new names. |
| Existing binary inputs | All86 retained binary inputs remain byte-equal. |
| English coverage | No literal Chinese in readable text or filenames;decoded JSON and Python text strings are also checked. |
| Naming | 21 Python sources and4 documents renamed;2 document folders clarified. Two briefing language versions consolidated into1 English briefing. |
| Links | All internal/companion document links resolve when the history package is next to the code package. |
| Legacy execution layout | Fresh E14 workspace contains all194 mapped research sources under their original execution names,without renamed duplicate aliases. |
| Main workspace | Preparation succeeds;scientific execution configuration equals canonical settings after excluding path/interpreter fields. |
| Real CPU calibration example | PASS. E1-local:B accepted52/errors2;A accepted13/errors0. No raw CSVs,history companion or GPU needed for this example. |

The checks establish source/layout compatibility and this compact execution route. Full P0–P7 GPU execution and all29 exploration runs have not been rerun. Shell syntax was not rechecked on this Windows host. Full experiments still require the assets/dependencies listed in EXTERNAL_ASSETS.md and KNOWN_LIMITATIONS.md.

The example reuses200R/200C target labels;it is an implementation check rather than independent scientific confirmation. Frozen numeric results are unchanged. The existing professional English briefing is the canonical summary and identifies the historical record-total correction.

Historical byte/hash identity records refer to original files in the history/backup snapshots. English translations change source/document bytes;frozen hashes are preserved as historical provenance. Register new experiments using the revised implementation,or use original files for exact historical identity audits.
