#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Run from project root; all writes stay in the new pilot directory.
set -euo pipefail
TASK_OUT=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/week0_pilot/run_20260915T073452Z
TASK_PY=${NIDS_PYTHON}
export PYTHONDONTWRITEBYTECODE=1
"$TASK_PY" "$TASK_OUT/prepare.py" > "$TASK_OUT/prepare.log" 2>&1
"$TASK_PY" "$TASK_OUT/calibrate.py" > "$TASK_OUT/calibrate.log" 2>&1
"$TASK_PY" "$TASK_OUT/evaluate.py" > "$TASK_OUT/evaluate.log" 2>&1
"$TASK_PY" "$TASK_OUT/bootstrap.py" > "$TASK_OUT/bootstrap.log" 2>&1
"$TASK_PY" "$TASK_OUT/verify.py" > "$TASK_OUT/verify.log" 2>&1
"$TASK_PY" "$TASK_OUT/calibration_audit.py" > "$TASK_OUT/calibration_audit.log" 2>&1
# report.py is run after inspecting aggregate results; never changes protocol or q.
"$TASK_PY" "$TASK_OUT/report.py" > "$TASK_OUT/report.log" 2>&1
"$TASK_PY" "$TASK_OUT/check_deliverables.py" > "$TASK_OUT/deliverable_check.log" 2>&1
