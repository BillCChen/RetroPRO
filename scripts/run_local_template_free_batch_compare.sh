#!/usr/bin/env bash
set -euo pipefail

# Local quick benchmark for template_free batch impact.
# Defaults are tuned for workstation validation (not final production benchmark).
# Example:
#   bash scripts/run_local_template_free_batch_compare.sh --parallel-num 6 --max-targets 64

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv_template_free/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv_multi/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv_multi/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[error] no python interpreter found; set PYTHON_BIN=/path/to/python" >&2
    exit 1
  fi
fi

exec "$PYTHON_BIN" scripts/benchmark_template_free_batch_impact.py \
  --python "$PYTHON_BIN" \
  --seed "${SEED:-42}" \
  --gpu "${GPU:-0}" \
  --iterations "${ITERATIONS:-80}" \
  --expansion-topk "${EXPANSION_TOPK:-8}" \
  --parallel-num "${PARALLEL_NUM:-4}" \
  --repeats "${REPEATS:-1}" \
  --max-targets "${MAX_TARGETS:-32}" \
  --allow-fail-runs \
  "$@"
