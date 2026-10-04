#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${NIDS_PYTHON:-python3}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage train-model --model-id rf_Traffic
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage train-model --model-id rf_All49
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage train-model --model-id xgb_Protocol
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage train-model --model-id xgb_All49
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage train-model --model-id mlp_Behavior
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage train-model --model-id mlp_All49
exec "$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage finalize-p2
