#!/usr/bin/env bash
set -euo pipefail

# One-command benchmark runner: test + collect + compare
# Example:
#   bash scripts/run_benchmark_compare.sh --gpu 1 --iterations 201 --parallel-nums 5,8,10

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_multi/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[error] No python interpreter found. Set PYTHON_BIN=/path/to/python" >&2
    exit 1
  fi
fi

exec "$PYTHON_BIN" scripts/benchmark_compare.py --python "$PYTHON_BIN" "$@"
