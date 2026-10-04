#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
${NIDS_PYTHON} outputs/post_p7_diagnostics/xgboost_variant_b_ton_exploratory_eval/run_20260918T071352Z/evaluate.py > outputs/post_p7_diagnostics/xgboost_variant_b_ton_exploratory_eval/run_20260918T071352Z/execution.log 2>&1
