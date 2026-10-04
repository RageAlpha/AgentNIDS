# External assets

The included CPU example needs only its packaged inputs and `requirements-example.txt`. Full experiments need the following assets.

## Original data

Place these four NetFlow v3 exports, about 13 GB combined, in your data directory:

- `NF-UNSW-NB15-v3.csv`
- `NF-CICIDS2018-v3.csv`
- `NF-BoT-IoT-v3.csv`
- `NF-ToN-IoT-v3.csv`

Also copy the supplied [NetFlow_v3_Features.csv](../datasets/NetFlow_v3_Features.csv) dictionary there. Obtain the same exports used by the study from the project owner/provider.

## Generated assets

Paths below refer to the prepared execution workspace. Follow the [run guide](RUN_GUIDE.md) to rebuild them in order.

| Missing asset | Rebuild with | Needed by |
|---|---|---|
| Sorted row/time arrays and split/window manifests | `prepare_data.py` | P0–P7 and downstream experiments |
| `outputs/feature_store_v2/` full feature/label arrays | P1 | P2 onward |
| `outputs/source_v2/` detector weights and calibrators | P2 | P3 onward |
| `outputs/prediction_cache_v2/` and full `outputs/shift_v2/` | P3 | Reliability and diagnostics |
| Reliability, coordination and outer-test arrays | P4–P7 | Later stages and downstream studies |
| Post-P7 model checkpoints and detailed caches | Selected experiment, after rebuilding its upstream inputs | That experiment's execution or historical checks |

The package retains compact parameters, identities and selected cross-run inputs in `inputs/`; the history archive holds reports and process records. Neither is a complete model/cache store. Main outputs can occupy hundreds of GB.

For a historical replay, request the exact fitted artifacts from the owner. For a fresh build, use new workspaces and configured paths. Dependencies are recorded in `requirements-runtime.txt`; remaining gaps are in [known limitations](KNOWN_LIMITATIONS.md).
