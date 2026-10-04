#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Optional reproducibility entry point. Always creates a new run; never overwrites B1/A/P7.
set -euo pipefail
PROJECT=${NIDS_PROJECT_ROOT}
TEMPLATE="$PROJECT/outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01"
RUN=$(mktemp -d "$PROJECT/outputs/post_p7_diagnostics/stage_B1/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
cp "$TEMPLATE/run_b1.py" "$TEMPLATE/plot_b1.py" "$TEMPLATE/report_b1.py" "$RUN/"
mkdir "$RUN/figures" "$RUN/risk_coverage_curves"
cd "$PROJECT"
${NIDS_PYTHON} -u "$RUN/run_b1.py" > "$RUN/run.log" 2>&1
MPLCONFIGDIR=/tmp/nids_b1_matplotlib ${NIDS_PYTHON} "$RUN/plot_b1.py" > "$RUN/plot.log" 2>&1
${NIDS_PYTHON} "$RUN/report_b1.py" > "$RUN/report.log" 2>&1
printf '%s\n' "$RUN"

# Actual commands executed for this reviewed run, from PROJECT:
# ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/run_b1.py > outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/run.log 2>&1
# MPLCONFIGDIR=/tmp/nids_b1_matplotlib ${NIDS_PYTHON} outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/plot_b1.py > outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/plot.log 2>&1
# ${NIDS_PYTHON} outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/report_b1.py > outputs/post_p7_diagnostics/stage_B1/run_20260914_B1_01/report.log 2>&1
# Following review, report-only corrections were regenerated with the same command and >> report.log.
# CUDA uses the existing agentnids environment; plotting uses existing matplotlib in base Python.
