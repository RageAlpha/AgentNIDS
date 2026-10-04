#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim/src/run_experiment.py --project-root ${NIDS_PROJECT_ROOT} --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim --stage prepare
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim/src/run_experiment.py --project-root ${NIDS_PROJECT_ROOT} --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim --stage evaluate
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim/src/run_experiment.py --project-root ${NIDS_PROJECT_ROOT} --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_action_correction_pilot_v1/run_20260929T110628Z_vkrXim --stage report
