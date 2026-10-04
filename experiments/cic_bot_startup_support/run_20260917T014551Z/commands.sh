#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
RUN=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/cic_bot_startup_support/run_20260917T014551Z
PYTHONDONTWRITEBYTECODE=1 ${NIDS_PYTHON} -u "$RUN/check_support.py" 2>&1 | tee "$RUN/analysis.log"
