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
# CPU preparation; independent CUDA processes require the approved GPU execution context.
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/prepare.py > outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/prepare.log 2>&1
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/select_acceptance.py > outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/selection.log 2>&1
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/evaluate.py > outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/evaluation.log 2>&1
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/verify_candidates.py > outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/candidate_validation.log 2>&1
${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/report_and_validate.py > outputs/post_p7_diagnostics/strict_k_reply4/run_20260915_C2_01/report_validation.log 2>&1
