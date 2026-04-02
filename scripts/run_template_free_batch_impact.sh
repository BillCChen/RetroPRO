#!/usr/bin/env bash
set -euo pipefail

# One-command runner for template_free batch impact benchmark.
# Example:
#   bash scripts/run_template_free_batch_impact.sh --gpu 1 --iterations 201 --parallel-num 8

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

exec "$PYTHON_BIN" scripts/benchmark_template_free_batch_impact.py --python "$PYTHON_BIN" "$@"
