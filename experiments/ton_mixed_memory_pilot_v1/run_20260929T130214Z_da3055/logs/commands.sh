#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/src/run_pilot.py --stage prepare --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/src/run_pilot.py --stage prepare --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/src/run_pilot.py --stage evaluate --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/src/run_pilot.py --stage prepare --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055
${NIDS_PYTHON} ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055/src/run_pilot.py --stage evaluate --run-dir ${NIDS_PROJECT_ROOT}/outputs/post_p7_diagnostics/ton_mixed_memory_pilot_v1/run_20260929T130214Z_da3055
