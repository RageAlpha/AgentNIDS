#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
PYTHONPATH=/tmp/nids_diagnostic_plotdeps MPLCONFIGDIR=/tmp/nids_diagnostic_mplconfig python -u outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/diagnose.py > outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/execution.log 2>&1
python -u outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/audit.py > outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/audit.log 2>&1
python outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/report.py > outputs/post_p7_diagnostics/raw_confidence_score_diagnostics_ton/run_20260918T133256Z/report.log 2>&1
