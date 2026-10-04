# Known execution limits

- **Validation scope:** the CPU calibration example was rerun successfully. Full GPU P0–P7 execution and all 29 follow-up runs were not rerun during packaging.
- **Assets:** raw CSVs and large model/cache stores are omitted; rebuild or supply them as described in [external assets](EXTERNAL_ASSETS.md). Static dependency analysis does not establish complete runtime coverage.
- **Historical replay:** workspace preparation restores layout, not a validated resume state. The selected run's old outputs are excluded. Some report/check scripts require earlier tables or complete saved protocols.
- **Representation checkpoints:** nine historical `complete.json` records lack matching `final.pt` files. Rebuild those models in fresh workspaces.
- **Variant B checker:** its historical `evaluate.log` was already absent.
- **Source identity:** archived source hashes describe the original files. Translated/renamed delivery files require fresh registration for new runs; see [English validation](ENGLISH_DELIVERY_VALIDATION.md).

## Missing original documents

These paths refer to the original execution layout. Obtain the originals for the affected routes.

| Original path | Affected route |
|---|---|
| `docs/Ton_variant/Reply_to_Siyuan_Week0_Analysis_and_Next_Step.md` | Direct read by CIC/BoT startup-support code |
| `docs/OutcomeB_2Expirements/Reply_to_Siyuan_OutcomeB_BeginWriting.md` | Direct read by RF/BoT full evaluation |
| `docs/XGBOOST_TON/Reply_to_Siyuan_VariantB_Analysis_and_Next_Step.md` | Authority reference in XGBoost protocol construction |
