#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Run from ${NIDS_PROJECT_ROOT}. This audit only writes beside its scripts.
python outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/audit.py > outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/audit.log 2>&1
python outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/write_report.py > outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/report.log 2>&1
python outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/finalize.py > outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/finalize.log 2>&1
python outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/write_report.py > outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/report.log 2>&1
python outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/verify_delivery.py > outputs/post_p7_diagnostics/week0_protocol_lineage_audit/run_20260915T125805Z/delivery.log 2>&1
