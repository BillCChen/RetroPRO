#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_retropro_css_dict_ablation_dataset.sh \
    --dataset <pth_hard|uspto190> \
    --dataset-label <pistachio_hard|uspto190> \
    --app-dir /abs/path/RetroPRO/retro_star \
    --output-root /abs/path/results_root \
    [--python-bin /abs/path/env/bin/python] \
    [--gpu 0] [--timestamp YYYYMMDD_HHMMSS]

Runs one dataset's RetroPRO template-free CSS/DICT 2x2 ablation matrix:
  retropro_css0_dict0_topk8
  retropro_css1_dict0_topk8
  retropro_css0_dict1_topk8
  retropro_css1_dict1_topk8

Expansion data collection is enabled for every run.

Output folder:
  OUTPUT_ROOT/DATASET_LABEL_css_dict_ablation_topk8_iter1000_TIMESTAMP/

Optional environment overrides:
  SEED=42
  ITERATIONS=1000
  EXPANSION_TOPK=8
  RD_LIST='[(7,0),(3,0)]'
  PYTHONPATH_EXTRA=/extra/path
USAGE
}

APP_DIR=""
PYTHON_BIN="${PYTHON_BIN:-}"
OUTPUT_ROOT=""
DATASET=""
DATASET_LABEL=""
GPU="${GPU:-0}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dataset-label)
      DATASET_LABEL="$2"
      shift 2
      ;;
    --app-dir)
      APP_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
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

if [[ -z "$DATASET" || -z "$DATASET_LABEL" || -z "$APP_DIR" || -z "$OUTPUT_ROOT" ]]; then
  echo "[error] --dataset, --dataset-label, --app-dir, and --output-root are required" >&2
  usage
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

case "$OUTPUT_ROOT" in
  /*)
    ;;
  *)
    OUTPUT_ROOT="$(pwd)/$OUTPUT_ROOT"
    ;;
esac

case "$DATASET" in
  pth_hard|uspto190)
    ;;
  *)
    echo "[error] unsupported dataset: $DATASET" >&2
    exit 1
    ;;
esac

if [[ ! -d "$APP_DIR" ]]; then
  echo "[error] app dir not found: $APP_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python bin not executable: $PYTHON_BIN" >&2
  exit 1
fi

SEED="${SEED:-42}"
ITERATIONS="${ITERATIONS:-1000}"
EXPANSION_TOPK="${EXPANSION_TOPK:-8}"
RD_LIST="${RD_LIST:-[(7,0),(3,0)]}"

cd "$APP_DIR"

export PYTHONPATH="$APP_DIR:$APP_DIR/packages/rdchiral:$APP_DIR/packages/mlp_retrosyn${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}${PYTHONPATH:+:$PYTHONPATH}"

dataset_root="$OUTPUT_ROOT/${DATASET_LABEL}_css_dict_ablation_topk${EXPANSION_TOPK}_iter${ITERATIONS}_${TIMESTAMP}"
mkdir -p "$dataset_root"

run_one() {
  local css_flag="$1"
  local dict_flag="$2"

  local method_label="retropro_css${css_flag}_dict${dict_flag}_topk${EXPANSION_TOPK}"
  local result_dir="$dataset_root/$method_label"
  local log_path="$dataset_root/${method_label}.log"

  local args=(
    retro_plan.py
    --seed "$SEED"
    --use_value_fn
    --expansion_topk "$EXPANSION_TOPK"
    --one_step_type template_free
    --iterations "$ITERATIONS"
    --gpu "$GPU"
    --test_routes "$DATASET"
    --collect_expansion_data
    --result_folder "$result_dir"
    --viz_dir "$result_dir/viz"
  )

  if [[ "$css_flag" == "1" ]]; then
    args+=(--CSS --RD_list "$RD_LIST")
  fi

  if [[ "$dict_flag" == "1" ]]; then
    args+=(--DICT)
  fi

  echo "START ${DATASET_LABEL} ${method_label} $(date --iso-8601=seconds)" | tee -a "$log_path"
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log_path"
  echo "END ${DATASET_LABEL} ${method_label} $(date --iso-8601=seconds)" | tee -a "$log_path"
}

run_one 0 0
run_one 1 0
run_one 0 1
run_one 1 1
