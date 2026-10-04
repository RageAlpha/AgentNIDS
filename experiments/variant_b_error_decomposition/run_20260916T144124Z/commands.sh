#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
ANALYSIS=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/variant_b_error_decomposition/run_20260916T144124Z
PYTHONDONTWRITEBYTECODE=1 ${NIDS_PYTHON} -u "$ANALYSIS/analyze.py" 2>&1 | tee "$ANALYSIS/analysis.log"
PYTHONDONTWRITEBYTECODE=1 ${NIDS_PYTHON} -u "$ANALYSIS/finalize.py" 2>&1 | tee "$ANALYSIS/finalize.log"
