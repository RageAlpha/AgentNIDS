#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
TASK_OUT=outputs/post_p7_diagnostics/week0_root_cause/run_20260916T035045Z
export PYTHONDONTWRITEBYTECODE=1
python "$TASK_OUT/investigate.py" > "$TASK_OUT/investigate.log" 2>&1
${NIDS_PYTHON} "$TASK_OUT/inspect_forest.py" > "$TASK_OUT/inspect_forest.log" 2>&1
python "$TASK_OUT/followup.py" > "$TASK_OUT/followup.log" 2>&1
python "$TASK_OUT/representatives.py" > "$TASK_OUT/representatives.log" 2>&1
python "$TASK_OUT/finish.py" > "$TASK_OUT/finish.log" 2>&1
# Earlier exploratory API probe: model.get_json is unavailable in this cuML version;
# used model.as_treelite().dump_as_json only for read-only tree inspection.
# No converted model was used for official inference or pilot results.
python "$TASK_OUT/check_delivery.py" > "$TASK_OUT/delivery.log" 2>&1
