#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${NIDS_PYTHON:-python3}"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
"$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage register
exec "$PYTHON" -u -m nids.run --config "$PROJECT/configs/execution_v2.json" --stage features
