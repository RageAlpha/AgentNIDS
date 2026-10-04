#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
RUN_DIR=${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/variant_b_ton_pilot/run_20260916T085242Z
python -u "$RUN_DIR/pilot.py" prepare 2>&1 | tee "$RUN_DIR/prepare.log"
python -u "$RUN_DIR/pilot.py" evaluate 2>&1 | tee "$RUN_DIR/evaluate.log"
# Final read-back verification; does not rerun the experiment.
python -u "$RUN_DIR/check_delivery.py" 2>&1 | tee "$RUN_DIR/delivery_check.log"
