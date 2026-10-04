#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
REVIEW=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/variant_b_ton_review/run_20260916T142121Z
PYTHON=${NIDS_PYTHON}
export PYTHONDONTWRITEBYTECODE=1
# Execute with access to the original CUDA GPU. Only review output is written.
"$PYTHON" -u "$REVIEW/upstream_review.py" 2>&1 | tee "$REVIEW/upstream_review.log"
"$PYTHON" -u "$REVIEW/reference_review.py" 2>&1 | tee "$REVIEW/reference_review.log"
"$PYTHON" -u "$REVIEW/finalize_review.py" 2>&1 | tee "$REVIEW/finalize_review.log"
