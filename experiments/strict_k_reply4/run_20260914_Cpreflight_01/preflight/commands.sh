#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
PYTHONDONTWRITEBYTECODE=1 ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/run_preflight.py 2>&1 | tee outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/run.log
PYTHONDONTWRITEBYTECODE=1 ${NIDS_PYTHON} outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/verify_deliverables.py 2>&1 | tee outputs/post_p7_diagnostics/strict_k_reply4/run_20260914_Cpreflight_01/preflight/deliverable_validation.log
