#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa/src/run_calibration.py --source-run ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa
