#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONUTF8=1
exec "$PYTHON" -u experiments/run_grouped_reproduction.py all "$@"
