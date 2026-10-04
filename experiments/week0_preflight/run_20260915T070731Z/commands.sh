#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Working directory: ${NIDS_PROJECT_ROOT}
# Exact execution commands. GPU command was executed with approved sandbox escalation.
set -euo pipefail
TASK_OUT=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/week0_preflight/run_20260915T070731Z
TASK_PY=${NIDS_PYTHON}
# Initial sandbox attempt retained in model_load.log; successful GPU attempt in model_load_gpu.log.
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/load_model_check.py" > "$TASK_OUT/model_load.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/load_model_check.py" > "$TASK_OUT/model_load_gpu.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/run_preflight.py" > "$TASK_OUT/run.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/write_protocol.py" > "$TASK_OUT/protocol.log" 2>&1
# Do not blindly replay into this reviewed run. For a rerun copy scripts to a new run directory first.
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/finalize.py" > "$TASK_OUT/finalize.log" 2>&1
PYTHONDONTWRITEBYTECODE=1 "$TASK_PY" "$TASK_OUT/verify_deliverables.py" > "$TASK_OUT/deliverable_validation.log" 2>&1
