#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Executed in ${NIDS_PROJECT_ROOT}. Existing output files must not be overwritten.
python outputs/post_p7_diagnostics/rf_bot_variant_b_full_evaluation/run_20260920T112946Z/evaluate.py > outputs/post_p7_diagnostics/rf_bot_variant_b_full_evaluation/run_20260920T112946Z/execution.log 2>&1
