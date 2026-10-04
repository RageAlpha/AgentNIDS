#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/select_c1.py 2>&1 | tee outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/selection.log
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/evaluate_c1.py 2>&1 | tee outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/evaluation.log
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/report_and_validate.py 2>&1 | tee outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C1_01/report_validation.log
