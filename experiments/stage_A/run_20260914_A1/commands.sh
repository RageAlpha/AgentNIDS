#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Re-input: Always create a new running directory that does not cover current results or freeze products.
set -euo pipefail
PROJECT=${NIDS_PROJECT_ROOT}
TEMPLATE="$PROJECT/outputs/post_p7_diagnostics/stage_A/run_20260914_A1"
RUN=$(mktemp -d "$PROJECT/outputs/post_p7_diagnostics/stage_A/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")
cp "$TEMPLATE/run_stage_a.py" "$TEMPLATE/audit_source_and_clipping.py" "$TEMPLATE/plot_stage_a.py" "$TEMPLATE/build_report.py" "$RUN/"
mkdir -p "$RUN/figures"
cd "$PROJECT"
${NIDS_PYTHON} -u "$RUN/run_stage_a.py" > "$RUN/run.log" 2>&1
${NIDS_PYTHON} -u "$RUN/audit_source_and_clipping.py" > "$RUN/source_audit.log" 2>&1
MPLCONFIGDIR=/tmp/nids_stage_a_matplotlib ${NIDS_PYTHON} "$RUN/plot_stage_a.py" > "$RUN/plot.log" 2>&1
${NIDS_PYTHON} "$RUN/build_report.py" > "$RUN/report.log" 2>&1
printf '%s\n' "$RUN"

# The actual execution of the project is recorded in the root directory:
# ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/stage_A/run_20260914_A1/run_stage_a.py > outputs/post_p7_diagnostics/stage_A/run_20260914_A1/run.log 2>&1
# For the first time, import failed at matplotlib, and no indicator has been calculated. The diagram is then demoted to plot_stage_a.py.
# ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/stage_A/run_20260914_A1/run_stage_a.py >> outputs/post_p7_diagnostics/stage_A/run_20260914_A1/run.log 2>&1
# ${NIDS_PYTHON} outputs/post_p7_diagnostics/stage_A/run_20260914_A1/plot_stage_a.py > outputs/post_p7_diagnostics/stage_A/run_20260914_A1/plot.log 2>&1
# ${NIDS_PYTHON} -u outputs/post_p7_diagnostics/stage_A/run_20260914_A1/audit_source_and_clipping.py > outputs/post_p7_diagnostics/stage_A/run_20260914_A1/source_audit.log 2>&1
# ${NIDS_PYTHON} outputs/post_p7_diagnostics/stage_A/run_20260914_A1/build_report.py > outputs/post_p7_diagnostics/stage_A/run_20260914_A1/report.log 2>&1
# The original rules read only checks: xgboost/sklearn.py:1859,1865; nvforest/_forest_inference.py:96.
