#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_template_baseline_smoke_bjmu.sh \
    [--cluster-root /home/liuzm/liuzm_chenqx/lustre1] \
    [--template-root /abs/path/test_synthetic_route_planning] \
    [--desp-python-bin /abs/path/conda_env/desp/bin/python] \
    [--pdvn-python-bin /abs/path/conda_env/pdvn/bin/python] \
    [--dataset pistachio_hard_100|uspto190] \
    [--gpu 0] [--run-group all|desp|pdvn] \
    [--limit 1] [--iterations 1] [--retro-topk 1] \
    [--timestamp YYYYMMDD_HHMMSS] [--dry-run]

BJMU cluster smoke runner for DESP/PDVN template baselines.
It does not require .venv symlinks. The DESP and PDVN Python interpreters are
passed as explicit conda-prefix paths and patched into the copied launcher.
USAGE
}

CLUSTER_ROOT="${CLUSTER_ROOT:-/home/liuzm/liuzm_chenqx/lustre1}"
TEMPLATE_ROOT="${TEMPLATE_ROOT:-}"
DESP_PYTHON_BIN="${DESP_PYTHON_BIN:-}"
PDVN_PYTHON_BIN="${PDVN_PYTHON_BIN:-}"
DATASET="${DATASET:-pistachio_hard_100}"
GPU_ID="${GPU_ID:-0}"
RUN_GROUP="${RUN_GROUP:-all}"
LIMIT="${LIMIT:-1}"
ITERATION_LIMIT="${ITERATION_LIMIT:-1}"
RETRO_TOPK="${RETRO_TOPK:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --cluster-root)
      CLUSTER_ROOT="$2"
      shift 2
      ;;
    --template-root)
      TEMPLATE_ROOT="$2"
      shift 2
      ;;
    --desp-python-bin)
      DESP_PYTHON_BIN="$2"
      shift 2
      ;;
    --pdvn-python-bin)
      PDVN_PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --gpu|--template-gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --run-group)
      RUN_GROUP="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --iterations)
      ITERATION_LIMIT="$2"
      shift 2
      ;;
    --retro-topk)
      RETRO_TOPK="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$TEMPLATE_ROOT" ]]; then
  TEMPLATE_ROOT="$CLUSTER_ROOT/test_synthetic_route_planning"
fi
if [[ -z "$DESP_PYTHON_BIN" ]]; then
  DESP_PYTHON_BIN="$CLUSTER_ROOT/conda_env/desp/bin/python"
fi
if [[ -z "$PDVN_PYTHON_BIN" ]]; then
  PDVN_PYTHON_BIN="$CLUSTER_ROOT/conda_env/pdvn/bin/python"
fi

case "$DATASET" in
  pistachio_hard_100)
    DESP_SOURCE="$TEMPLATE_ROOT/desp/desp/data/pistachio_hard_targets.txt"
    ;;
  uspto190)
    DESP_SOURCE="$TEMPLATE_ROOT/desp/desp/data/uspto_190_targets.txt"
    ;;
  *)
    echo "[error] unsupported dataset: $DATASET" >&2
    exit 1
    ;;
esac

case "$RUN_GROUP" in
  all|desp|pdvn) ;;
  *)
    echo "[error] --run-group must be one of: all, desp, pdvn" >&2
    exit 1
    ;;
esac

if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [[ "$LIMIT" -lt 1 ]]; then
  echo "[error] --limit must be a positive integer" >&2
  exit 1
fi

LAUNCHER="$TEMPLATE_ROOT/artifacts/inference_scripts/launch_pistachio_template_runs.sh"
PREPARE_PDVN="$TEMPLATE_ROOT/artifacts/inference_scripts/prepare_pdvn_pistachio_input.py"

for required in "$LAUNCHER" "$PREPARE_PDVN" "$DESP_SOURCE" "$PDVN_PYTHON_BIN" "$DESP_PYTHON_BIN"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] required path not found: $required" >&2
    exit 1
  fi
done

if [[ ! -x "$PDVN_PYTHON_BIN" ]]; then
  echo "[error] PDVN python is not executable: $PDVN_PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$DESP_PYTHON_BIN" ]]; then
  echo "[error] DESP python is not executable: $DESP_PYTHON_BIN" >&2
  exit 1
fi

SMOKE_ROOT="$TEMPLATE_ROOT/artifacts/smoke/template_baseline_smoke_bjmu_${DATASET}_${TIMESTAMP}"
INPUT_DIR="$SMOKE_ROOT/inputs"
mkdir -p "$INPUT_DIR"

DESP_TEST_PATH="$INPUT_DIR/${DATASET}_limit${LIMIT}.txt"
PDVN_TEST_ROUTES="$INPUT_DIR/${DATASET}_limit${LIMIT}.pkl"
PDVN_INPUT_META="$INPUT_DIR/${DATASET}_limit${LIMIT}.metadata.json"
PATCHED_LAUNCHER="$SMOKE_ROOT/launch_template_baselines_${DATASET}_bjmu.sh"

head -n "$LIMIT" "$DESP_SOURCE" > "$DESP_TEST_PATH"
"$PDVN_PYTHON_BIN" "$PREPARE_PDVN" \
  --source "$DESP_TEST_PATH" \
  --output-pkl "$PDVN_TEST_ROUTES" \
  --metadata "$PDVN_INPUT_META" > "$SMOKE_ROOT/prepare_pdvn_input.log" 2>&1

sed \
  -e "s|^ROOT=.*|ROOT=\"$TEMPLATE_ROOT\"|" \
  -e "s|\\\$ROOT/PDVN/.venv/bin/python|$PDVN_PYTHON_BIN|g" \
  -e "s|\\\$ROOT/desp/.venv/bin/python|$DESP_PYTHON_BIN|g" \
  "$LAUNCHER" > "$PATCHED_LAUNCHER"
PDVN_PYTHONPATH="$TEMPLATE_ROOT/PDVN:$TEMPLATE_ROOT/PDVN/retro_star/packages/rdchiral:$TEMPLATE_ROOT/PDVN/retro_star/packages/mlp_retrosyn"
python3 - "$PATCHED_LAUNCHER" "$PDVN_PYTHONPATH" <<'PY_PATCHED_LAUNCHER'
from pathlib import Path
import sys

launcher = Path(sys.argv[1])
pdvn_pythonpath = sys.argv[2]
text = launcher.read_text()
text = text.replace(
    "cd '$ROOT/PDVN/retro_star';",
    f"export PYTHONPATH='{pdvn_pythonpath}':${{PYTHONPATH:-}}; cd '$ROOT/PDVN/retro_star';",
)
launcher.write_text(text)
PY_PATCHED_LAUNCHER
chmod +x "$PATCHED_LAUNCHER"

RUN_SUFFIX="smoke_bjmu_${DATASET}_limit${LIMIT}_retrotopk${RETRO_TOPK}_iter${ITERATION_LIMIT}"

cat <<INFO
smoke_root=$SMOKE_ROOT
dataset=$DATASET
run_group=$RUN_GROUP
gpu=$GPU_ID
limit=$LIMIT
iterations=$ITERATION_LIMIT
retro_topk=$RETRO_TOPK
desp_python=$DESP_PYTHON_BIN
pdvn_python=$PDVN_PYTHON_BIN
desp_test_path=$DESP_TEST_PATH
pdvn_test_routes=$PDVN_TEST_ROUTES
patched_launcher=$PATCHED_LAUNCHER
INFO

RUN_COMMAND=(
  env
  "RUN_TS=$TIMESTAMP"
  "GPU_ID=$GPU_ID"
  "DATASET=$DATASET"
  "RUN_SUFFIX=$RUN_SUFFIX"
  "RETRO_TOPK=$RETRO_TOPK"
  "ITERATION_LIMIT=$ITERATION_LIMIT"
  "RUN_GROUP=$RUN_GROUP"
  "DESP_TEST_PATH=$DESP_TEST_PATH"
  "PDVN_TEST_ROUTES=$PDVN_TEST_ROUTES"
  bash "$PATCHED_LAUNCHER"
)

printf "command="
printf "%q " "${RUN_COMMAND[@]}"
printf "\n"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${RUN_COMMAND[@]}"
