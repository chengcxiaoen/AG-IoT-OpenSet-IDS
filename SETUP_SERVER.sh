#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'
command -v g++ >/dev/null || { echo 'OpenMax LibMR requires g++. Install build-essential, then rerun.' >&2; exit 1; }
"$PYTHON" -m pip install -r requirements.txt
"$PYTHON" -m pip install --no-build-isolation libmr==0.1.9
"$PYTHON" -B -m unittest discover -s tests -p 'test_supplement_*.py' -v
echo 'SETUP OK. See docs/REPRODUCIBILITY.md before running the full protocol.'
