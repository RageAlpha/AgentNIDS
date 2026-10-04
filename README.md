# AgentNIDS

Cross-dataset network intrusion detection: the P0–P7 study and 29 follow-up experiment runs. UNSW is the source domain; CIC, BoT and ToN are target domains.

## Start here

1. [Code guide](CODE_GUIDE.md): project flow and the main files to review.
2. [Run guide](docs/RUN_GUIDE.md): commands and the results each command produces.
3. [Experiment index](EXPERIMENT_INDEX.md): choose a follow-up research question.

## First run

From this folder:

```bash
python -m pip install -r requirements-example.txt
python scripts/workspace.py example
```

This tested CPU example saves calibration results in a new `runs/` workspace. It needs no raw CSVs or GPU.

Main code is in `src/nids/`; follow-up code is in `experiments/`. P7 did not support the central positive hypotheses; later studies are exploratory. Full runs require [external assets](docs/EXTERNAL_ASSETS.md). Detailed documents are [reference material](docs/DOCUMENT_ROLES.md).

Historical results are supplied separately as `NIDSProject_Professor_History_20261004`. Place that folder beside the cloned `AgentNIDS` folder to use the local history links in the guides; the archive is outside this GitHub repository.
