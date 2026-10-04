#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${NIDS_PYTHON:-python3}"
BASE="$PROJECT/outputs/post_p7_diagnostics"
SOURCE="$BASE/reliability_local_evidence_v1/run_20260929T074336Z_8Aw2gr"
SCRIPT_RUN="$BASE/ton_local_action_correction_v1/run_20260929T105356Z_w5yzUa"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/ton_local_action_correction_v1/run_professor_check_$STAMP"
n=1
while [[ -e "$RUN" ]]; do RUN="$BASE/ton_local_action_correction_v1/run_professor_check_${STAMP}_$n"; n=$((n+1)); done
mkdir -p "$RUN"/{src,config,calibration,reports,logs}
cp "$SCRIPT_RUN/src/run_calibration.py" "$RUN/src/run_calibration.py"
if [[ -f "$SCRIPT_RUN/config/06_Pilot_Interpretation_and_Next_Experiment_ZH.md" ]]; then
  cp "$SCRIPT_RUN/config/06_Pilot_Interpretation_and_Next_Experiment_ZH.md" "$RUN/config/"
fi
printf '%q ' "$PYTHON" "$RUN/src/run_calibration.py" --source-run "$SOURCE" --run-dir "$RUN" > "$RUN/logs/commands.sh"
printf '\n' >> "$RUN/logs/commands.sh"
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$RUN/src/run_calibration.py" --source-run "$SOURCE" --run-dir "$RUN" | tee "$RUN/logs/execution.log"
printf 'RUN_DIR=%s\n' "$RUN"
