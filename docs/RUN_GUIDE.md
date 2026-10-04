# Run commands and results

## 1. Included CPU example

Run from the package root on Windows or Linux:

```bash
python -m pip install -r requirements-example.txt
python scripts/workspace.py example
```

This calls [run_calibration.py](../experiments/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/src/run_calibration.py) on saved ToN reference/calibration records. No raw CSVs, GPU or history archive are needed.

The command prints `workspace`, `status` and `log`. Let `W` be that workspace, normally a new `runs/example_<time>/` folder. Read `W/EXAMPLE_VALIDATION.json` first. The remaining results are under:

```text
W/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/
```

| File within that result folder | Contents |
|---|---|
| `calibration/gate_feasibility.csv` | Accepted counts, errors, risk and utility |
| `calibration/selected_rules.json` | Selected thresholds and local-support gates |
| `calibration/correction_harm.csv` | Helpful and harmful label changes |
| `reports/summary_ZH.md` | English summary; the original output filename is retained |
| `logs/execution.log` | Console output and errors |

Expected check: `status=PASS`; `E1-local` accepts 52 benign candidates with 2 errors and 13 attack candidates with 0 errors. These are reused calibration records, not independent outer-test performance.

To choose a location, add `--workspace /absolute/path/to/new-example`; the directory must not already exist.

## 2. Main P0–P7 study

Use Linux/WSL and the compatible GPU environment recorded in `requirements-runtime.txt`. Supply the four [original CSVs and feature dictionary](EXTERNAL_ASSETS.md). Activate that environment, replace these paths and use a new `WORK_DIR`:

```bash
CODE_DIR="/path/to/AgentNIDS"
WORK_DIR="/path/to/new-nids-main"
DATA_DIR="/path/to/netflow-data"

cd "$CODE_DIR"
python scripts/workspace.py prepare --experiment main --workspace "$WORK_DIR" --data-root "$DATA_DIR"
cd "$WORK_DIR"
export NIDS_PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src"
python outputs/round1_data_prep_v1/prepare_data.py --self-test
python outputs/round1_data_prep_v1/prepare_data.py --data-root "$DATA_DIR" --output "$PWD/outputs/prepared_new"
python scripts/configure_paths.py --data-root "$DATA_DIR" --prepared-run "$PWD/outputs/prepared_new" --python "$NIDS_PYTHON"
```

Preparation writes sorted arrays and split/window manifests to `outputs/prepared_new/`; inspect `data_preparation_report.md` and `validation.json` there.

Run the following commands **in order from `WORK_DIR`**, after each preceding stage completes. Result paths below are relative to it.

| Stage / command | Produces | Results to read first |
|---|---|---|
| P0–P1: `bash scripts/run_round1.sh` | Environment record and feature store | `outputs/round1_v2/P0/stage_report.md`; `outputs/feature_store_v2/stage_report.md` |
| P2: `bash scripts/run_p2.sh` | Six detectors, calibrators and source scores | `outputs/source_v2/P2/stage_report.md`; `source_metrics.csv` in the same folder |
| P3: `bash scripts/run_p3.sh` | Predictions, causal shift and replay | `outputs/round1_v2/P3/stage_report.md`; `P3_validation.json` |
| P4: `bash scripts/run_p4.sh` | Reliability models and G0/G1 gates | `outputs/round1_v2/P4/stage_report.md`; `F*/CRE_metrics.csv`; `F*/gate_report.json` |
| P5: `bash scripts/run_p5.sh` | Sensitivity comparisons | `outputs/round1_v2/P5/stage_report.md`; `sensitivity_summary.json` |
| P6: `bash scripts/run_p6.sh` | Frozen coordination policies and budgets | `outputs/round2_v2/P6/stage_report.md`; `F*/frozen_policies.json`; `F*/budget_matched.json` |
| P7: `bash scripts/run_p7.sh` | Held-out risk, coverage and cost | `outputs/outer_v2/P7/stage_report.md`; `round1_aggregate.json`; `F*/round2/policy_metrics.json` where executed |

For P3–P7, shortened paths refer to that stage's folder. `F*` denotes fold subdirectories. `run_round1.sh` runs only P0–P1; P7 Round2 is defined for F1/F2. A metadata-only check before P7 is:

```bash
python -m nids.run --config configs/execution_v2.json --stage preflight-p7
```

Read each stage report before its metrics. A completed run can fail a scientific gate. Full GPU execution was not rerun during packaging; see [known limitations](KNOWN_LIMITATIONS.md).

## 3. Follow-up experiments

Choose E01–E29 in the [index](../EXPERIMENT_INDEX.md). They are separate research branches.

```bash
python scripts/workspace.py list
python scripts/workspace.py prepare --experiment E01 --workspace /absolute/path/to/new-experiment --with-history
```

`prepare` stages files and writes `WORKSPACE_PREPARATION.json`; it does not execute the experiment. `--with-history` requires the companion archive and supplies other runs' metadata. Use `--history /path/to/archive` if it is elsewhere.

Supply required assets, create the selected run's configuration/preregistration and follow its recorded command sequence in the prepared workspace. Old outputs of that selected run are excluded. The tool restores original execution filenames; package launchers in `tools/legacy/scripts/` become `scripts/` there. See each local README for code, inputs and primary result files. E24 can use the CPU example above.
