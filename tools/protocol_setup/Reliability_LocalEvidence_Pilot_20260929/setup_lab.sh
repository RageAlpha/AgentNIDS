#!/usr/bin/env bash
# Professor run copy: path-adapted recorded commands. Review input and output scope before replay.
NIDS_PROJECT_ROOT="${NIDS_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
NIDS_PYTHON="${NIDS_PYTHON:-python3}"
cd "$NIDS_PROJECT_ROOT"
# Create a fresh result directory only. No experiment, dependency install, or digest.
set -euo pipefail
PROJECT_ROOT="${NIDS_PROJECT_ROOT}"
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$PROJECT_ROOT/outputs" ]]; then
  printf '%s\n' "Project outputs directory not found: $PROJECT_ROOT/outputs" >&2
  exit 1
fi
if [[ ! -f "$BUNDLE_DIR/protocol_v1.json" ]]; then
  printf '%s\n' "Missing protocol_v1.json next to setup_lab.sh" >&2
  exit 1
fi
RUN_BASE="$PROJECT_ROOT/outputs/post_p7_diagnostics/reliability_local_evidence_v1"
mkdir -p -- "$RUN_BASE"
RUN_DIR="$(mktemp -d "$RUN_BASE/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
mkdir -p -- "$RUN_DIR/src" "$RUN_DIR/config" "$RUN_DIR/calibration" "$RUN_DIR/pilot" "$RUN_DIR/tables" "$RUN_DIR/reports" "$RUN_DIR/checks" "$RUN_DIR/logs"
cp -- "$BUNDLE_DIR/protocol_v1.json" "$RUN_DIR/config/protocol_v1.json"
printf '%s\n' "$RUN_DIR" > "$BUNDLE_DIR/lab_run_path.txt"
printf 'Created result directory: %s\n' "$RUN_DIR"
printf '%s\n' "Next: paste Instruction 1 from 04_VSCode_Codex_Prompts_ZH.md into VSCode Codex."
