#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
python ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z/experiment.py ton
python ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z/experiment.py bot
python ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/outcome_b_two_experiments/run_20260920T065121Z/finalize.py
