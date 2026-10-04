# Code guide

## Main study

The pipeline converts NetFlow CSVs to 49 features, trains six source-domain detectors, estimates their reliability, coordinates their decisions and evaluates held-out target domains.

UNSW supplies source training data. The outer folds hold out ToN (F1), BoT (F2) and CIC (F3); the other two target domains supply development data.

Read the following files in order. The stage modules show orchestration; the linked implementation modules show the algorithms.

| Step | Files to review | Main question |
|---|---|---|
| Entry and data | [run.py](src/nids/run.py), [partitions.py](src/nids/data/partitions.py) | How are stages, splits and window identities defined? |
| P2: detection | [p2.py](src/nids/p2.py), [base_models.py](src/nids/models/base_models.py) | How are RF, XGBoost and MLP trained and calibrated? |
| P3: replay | [p3.py](src/nids/p3.py), [replay/](src/nids/replay/) | How are predictions and causal shift computed? |
| P4: reliability | [p4.py](src/nids/p4.py), [cre.py](src/nids/reliability/cre.py) | Does shift improve reliability ranking? |
| P6: coordination | [p6.py](src/nids/p6.py), [selection.py](src/nids/coordination/selection.py) | How are actions, thresholds and budgets selected? |
| P7: evaluation | [p7.py](src/nids/p7.py), [check_p7_protocol.py](tests/check_p7_protocol.py) | Are frozen development decisions reused on the outer test? |

P5 sensitivity analyses are in [p5.py](src/nids/p5.py). Prioritize data isolation, calibration, risk/coverage definitions and policy selection when suggesting improvements.

## Follow-up studies and execution

Choose a question in the [experiment index](EXPERIMENT_INDEX.md), then read that run's short README. It links the execution steps, supporting modules, inputs and primary results.

Edit source in this package, then create a fresh workspace using the [run guide](docs/RUN_GUIDE.md). Tests run in that working layout with `src` on `PYTHONPATH`; some require GPU libraries or full-run assets. `inputs/` holds supporting data and frozen settings.

For scientific context, consult the [core conclusion](docs/Project_Core_Conclusion_After_P7.md) and [protocol decisions](docs/protocol_decisions.md).
