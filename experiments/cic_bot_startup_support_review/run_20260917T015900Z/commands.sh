#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
REVIEW=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/cic_bot_startup_support_review/run_20260917T015900Z
PYTHONDONTWRITEBYTECODE=1 ${NIDS_PYTHON} -u "$REVIEW/review.py" 2>&1 | tee "$REVIEW/review.log"
