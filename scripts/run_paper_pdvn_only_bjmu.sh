#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_paper_pdvn_only_bjmu.sh \
    --dataset <pistachio_hard_100|uspto190> \
    --run-suffix <paper_6methods_template_suffix> \
    --timestamp <YYYYMMDD_HHMMSS> \
    [--cluster-root /home/liuzm/liuzm_chenqx/lustre1] \
    [--template-root /abs/path/test_synthetic_route_planning] \
    [--pdvn-python-bin /abs/path/conda_env/pdvn/bin/python] \
    [--gpu 0] [--iterations 1000] [--retro-topk 8] \
    [--pdvn-source-txt /abs/path/targets.txt] \
    [--pdvn-test-routes /abs/path/routes.pkl] \
    [--dry-run]

Runs only the PDVN template baseline group for an existing paper 6-method run.
Use the same --run-suffix and --timestamp as the failed template baseline run
to write PDVN outputs into that run root without recomputing DESP.
USAGE
}

CLUSTER_ROOT="${CLUSTER_ROOT:-/home/liuzm/liuzm_chenqx/lustre1}"
TEMPLATE_ROOT="${TEMPLATE_ROOT:-}"
PDVN_PYTHON_BIN="${PDVN_PYTHON_BIN:-}"
DATASET="${DATASET:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
TIMESTAMP="${TIMESTAMP:-}"
GPU_ID="${GPU_ID:-0}"
ITERATION_LIMIT="${ITERATION_LIMIT:-1000}"
RETRO_TOPK="${RETRO_TOPK:-8}"
PDVN_SOURCE_TXT="${PDVN_SOURCE_TXT:-}"
PDVN_TEST_ROUTES="${PDVN_TEST_ROUTES:-}"
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
    --pdvn-python-bin)
      PDVN_PYTHON_BIN="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --run-suffix)
      RUN_SUFFIX="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --gpu|--template-gpu)
      GPU_ID="$2"
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
    --pdvn-source-txt)
      PDVN_SOURCE_TXT="$2"
      shift 2
      ;;
    --pdvn-test-routes)
      PDVN_TEST_ROUTES="$2"
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
if [[ -z "$PDVN_PYTHON_BIN" ]]; then
  PDVN_PYTHON_BIN="$CLUSTER_ROOT/conda_env/pdvn/bin/python"
fi

if [[ -z "$DATASET" || -z "$RUN_SUFFIX" || -z "$TIMESTAMP" ]]; then
  echo "[error] --dataset, --run-suffix, and --timestamp are required" >&2
  usage
  exit 1
fi

case "$DATASET" in
  pistachio_hard_100)
    if [[ -z "$PDVN_SOURCE_TXT" && -z "$PDVN_TEST_ROUTES" ]]; then
      PDVN_SOURCE_TXT="$TEMPLATE_ROOT/PDVN/retro_star/dataset/pistachio_hard_targets.txt"
    fi
    ;;
  uspto190)
    if [[ -z "$PDVN_SOURCE_TXT" && -z "$PDVN_TEST_ROUTES" ]]; then
      PDVN_TEST_ROUTES="$TEMPLATE_ROOT/PDVN/retro_star/dataset/uspto190.pkl"
    fi
    ;;
  *)
    echo "[error] unsupported dataset: $DATASET" >&2
    exit 1
    ;;
esac

LAUNCHER="$TEMPLATE_ROOT/artifacts/inference_scripts/launch_pistachio_template_runs.sh"
if [[ ! -f "$LAUNCHER" ]]; then
  echo "[error] template launcher not found: $LAUNCHER" >&2
  exit 1
fi
if [[ ! -x "$PDVN_PYTHON_BIN" ]]; then
  echo "[error] PDVN python bin not executable: $PDVN_PYTHON_BIN" >&2
  exit 1
fi
if [[ -n "$PDVN_SOURCE_TXT" && ! -f "$PDVN_SOURCE_TXT" ]]; then
  echo "[error] PDVN source txt not found: $PDVN_SOURCE_TXT" >&2
  exit 1
fi
if [[ -n "$PDVN_TEST_ROUTES" && ! -f "$PDVN_TEST_ROUTES" ]]; then
  echo "[error] PDVN test routes not found: $PDVN_TEST_ROUTES" >&2
  exit 1
fi

PATCH_ROOT="$TEMPLATE_ROOT/artifacts/smoke/pdvn_only_launcher_${DATASET}_${TIMESTAMP}"
mkdir -p "$PATCH_ROOT"
PATCHED_LAUNCHER="$PATCH_ROOT/launch_pdvn_only_${DATASET}.sh"

sed "s|^ROOT=.*|ROOT=\"$TEMPLATE_ROOT\"|" "$LAUNCHER" > "$PATCHED_LAUNCHER"
PDVN_PYTHONPATH="$TEMPLATE_ROOT/PDVN:$TEMPLATE_ROOT/PDVN/retro_star/packages/rdchiral:$TEMPLATE_ROOT/PDVN/retro_star/packages/mlp_retrosyn"
python3 - "$PATCHED_LAUNCHER" "$PDVN_PYTHON_BIN" "$PDVN_PYTHONPATH" <<'PY_PATCHED_LAUNCHER'
from pathlib import Path
import sys

launcher = Path(sys.argv[1])
pdvn_python = sys.argv[2]
pdvn_pythonpath = sys.argv[3]
text = launcher.read_text()
text = text.replace("$ROOT/PDVN/.venv/bin/python", pdvn_python)
text = text.replace(
    "cd '$ROOT/PDVN/retro_star';",
    f"export PYTHONPATH='{pdvn_pythonpath}':${{PYTHONPATH:-}}; cd '$ROOT/PDVN/retro_star';",
)
launcher.write_text(text)
PY_PATCHED_LAUNCHER
chmod +x "$PATCHED_LAUNCHER"

RUN_ROOT="$TEMPLATE_ROOT/artifacts/inference_results/${DATASET}__template_based__${RUN_SUFFIX}__${TIMESTAMP}"
COMMAND=(
  env
  "RUN_TS=$TIMESTAMP"
  "GPU_ID=$GPU_ID"
  "DATASET=$DATASET"
  "RUN_SUFFIX=$RUN_SUFFIX"
  "RETRO_TOPK=$RETRO_TOPK"
  "ITERATION_LIMIT=$ITERATION_LIMIT"
  "RUN_GROUP=pdvn"
)

if [[ -n "$PDVN_SOURCE_TXT" ]]; then
  COMMAND+=("PDVN_SOURCE_TXT=$PDVN_SOURCE_TXT")
fi
if [[ -n "$PDVN_TEST_ROUTES" ]]; then
  COMMAND+=("PDVN_TEST_ROUTES=$PDVN_TEST_ROUTES")
fi

COMMAND+=(bash "$PATCHED_LAUNCHER")

cat <<INFO
run_root=$RUN_ROOT
dataset=$DATASET
run_suffix=$RUN_SUFFIX
timestamp=$TIMESTAMP
gpu=$GPU_ID
iterations=$ITERATION_LIMIT
retro_topk=$RETRO_TOPK
pdvn_python=$PDVN_PYTHON_BIN
pdvn_pythonpath=$PDVN_PYTHONPATH
pdvn_source_txt=$PDVN_SOURCE_TXT
pdvn_test_routes=$PDVN_TEST_ROUTES
patched_launcher=$PATCHED_LAUNCHER
INFO

printf "command="
printf "%q " "${COMMAND[@]}"
printf "\n"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${COMMAND[@]}"
