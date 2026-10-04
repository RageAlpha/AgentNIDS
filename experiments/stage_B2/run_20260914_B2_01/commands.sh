#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Actual commands used for this completed run (project working directory):
# ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/run_b2.py > outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/run.log 2>&1
# ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/bootstrap_b2.py > outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/bootstrap.log 2>&1
# MPLCONFIGDIR=/tmp/nids_b2_matplotlib ${NIDS_PYTHON} outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/plot_b2.py > outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/plot.log 2>&1
# ${NIDS_PYTHON} outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/report_b2.py > outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01/report.log 2>&1
# GPU steps ran with approved GPU access. Plotting/reporting used CPU Python.
# To reproduce, invoke: bash commands.sh <NEW_run_id>. Never overwrite this run.
set -euo pipefail
cd ${NIDS_PROJECT_ROOT}
if [[ $# -ne 1 || ! "$1" =~ ^run_[A-Za-z0-9_]+$ ]]; then
    echo 'Usage: bash commands.sh <NEW_run_id>'; exit 2
fi
b2_origin=outputs/post_p7_diagnostics/stage_B2/run_20260914_B2_01
b2_out="outputs/post_p7_diagnostics/stage_B2/$1"
if [[ -e "$b2_out" ]]; then echo 'Output already exists; refusing to overwrite'; exit 2; fi
mkdir "$b2_out"
mkdir "$b2_out/models" "$b2_out/curves" "$b2_out/figures" "$b2_out/bootstrap"
cp "$b2_origin/run_b2.py" "$b2_origin/bootstrap_b2.py" "$b2_origin/plot_b2.py" "$b2_origin/report_b2.py" "$b2_origin/commands.sh" "$b2_out/"
${NIDS_PYTHON} -u "$b2_out/run_b2.py" > "$b2_out/run.log" 2>&1
${NIDS_PYTHON} -u "$b2_out/bootstrap_b2.py" > "$b2_out/bootstrap.log" 2>&1
MPLCONFIGDIR=/tmp/nids_b2_matplotlib ${NIDS_PYTHON} "$b2_out/plot_b2.py" > "$b2_out/plot.log" 2>&1
${NIDS_PYTHON} "$b2_out/report_b2.py" > "$b2_out/report.log" 2>&1
