#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
python -u outputs/post_p7_diagnostics/temporal_stratified_400_ton/run_20260918T131742Z/experiment.py > outputs/post_p7_diagnostics/temporal_stratified_400_ton/run_20260918T131742Z/execution.log 2>&1
python outputs/post_p7_diagnostics/temporal_stratified_400_ton/run_20260918T131742Z/check_delivery.py > outputs/post_p7_diagnostics/temporal_stratified_400_ton/run_20260918T131742Z/delivery.log 2>&1
