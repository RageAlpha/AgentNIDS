#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Actual stage commands. All audit scripts derive OUT from their own directory.
# For reproducibility copy scripts into a NEW audit directory, then change TASK_OUT.
# Never execute old pilot/preflight scripts that write into old directories.
set -euo pipefail
TASK_OUT=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/week0_audit/run_20260915T075634Z
TASK_PY=${NIDS_PYTHON}
export PYTHONDONTWRITEBYTECODE=1
"$TASK_PY" "$TASK_OUT/prepare_audit.py" > "$TASK_OUT/prepare.log" 2>&1
"$TASK_PY" "$TASK_OUT/replay.py" > "$TASK_OUT/replay.log" 2>&1
"$TASK_PY" "$TASK_OUT/csv_reconstruction.py" > "$TASK_OUT/csv_reconstruction.log" 2>&1
"$TASK_PY" "$TASK_OUT/distribution_and_numerics.py" > "$TASK_OUT/distribution.log" 2>&1
# inference_audit.py was executed with approved GPU sandbox escalation, same original environment.
"$TASK_PY" "$TASK_OUT/inference_audit.py" > "$TASK_OUT/inference.log" 2>&1
"$TASK_PY" "$TASK_OUT/statistical_audit.py" > "$TASK_OUT/statistical.log" 2>&1
# First report-only syntax error retained in finalize_attempt1.log; corrected script below.
"$TASK_PY" "$TASK_OUT/finalize_audit.py" > "$TASK_OUT/finalize.log" 2>&1
"$TASK_PY" "$TASK_OUT/check_deliverables.py" > "$TASK_OUT/deliverable_check.log" 2>&1
