#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_retropro_radius_sweep_dataset.sh \
    --dataset <pth_hard|uspto190> \
    --dataset-label <pistachio_hard|uspto190> \
    --app-dir /abs/path/RetroPRO/retro_star \
    --output-root /abs/path/results/data_collection \
    [--python-bin /abs/path/env/bin/python] \
    [--gpu 0] [--timestamp YYYYMMDD_HHMMSS] \
    [--radius-group odd|even|all] \
    [--radii 1,3,5] \
    [--collect-expansion-data]

Runs RetroPRO radius sweep experiments with fixed parameters:
  one_step_type=template_free
  CSS=on
  DICT=on
  expansion_topk=8
  iterations=1000
  seed=42
  RD_list=[(R,0)] for each selected radius

Environment overrides:
  SEED=42
  ITERATIONS=1000
  EXPANSION_TOPK=8
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
RADIUS_GROUP="${RADIUS_GROUP:-all}"
CUSTOM_RADII=""
COLLECT_EXPANSION_DATA=0

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
    --radius-group)
      RADIUS_GROUP="$2"
      shift 2
      ;;
    --radii)
      CUSTOM_RADII="$2"
      shift 2
      ;;
    --collect-expansion-data)
      COLLECT_EXPANSION_DATA=1
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

if [[ -z "$DATASET" || -z "$DATASET_LABEL" || -z "$APP_DIR" || -z "$OUTPUT_ROOT" ]]; then
  echo "[error] --dataset, --dataset-label, --app-dir, and --output-root are required" >&2
  usage
  exit 1
fi

case "$DATASET" in
  pth_hard|uspto190) ;;
  *)
    echo "[error] unsupported dataset: $DATASET" >&2
    exit 1
    ;;
esac

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$(pwd)/$OUTPUT_ROOT" ;;
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

select_radii() {
  if [[ -n "$CUSTOM_RADII" ]]; then
    echo "$CUSTOM_RADII" | tr ',' ' '
    return
  fi

  case "$RADIUS_GROUP" in
    odd)
      echo "1 3 5 7 9"
      ;;
    even)
      echo "2 4 6 8 10"
      ;;
    all)
      echo "1 2 3 4 5 6 7 8 9 10"
      ;;
    *)
      echo "[error] --radius-group must be odd, even, or all" >&2
      exit 1
      ;;
  esac
}

read -r -a RADII <<< "$(select_radii)"

if [[ "${#RADII[@]}" -eq 0 ]]; then
  echo "[error] no radii selected" >&2
  exit 1
fi

for radius in "${RADII[@]}"; do
  if [[ ! "$radius" =~ ^[0-9]+$ ]]; then
    echo "[error] invalid radius: $radius" >&2
    exit 1
  fi
done

dataset_root="$OUTPUT_ROOT/retropro_radius_${DATASET_LABEL}_css_dict_topk${EXPANSION_TOPK}_iter${ITERATIONS}_${RADIUS_GROUP}_gpu${GPU}_${TIMESTAMP}"
mkdir -p "$dataset_root"

manifest_path="$dataset_root/run_manifest.tsv"
printf "dataset\tradius\tRD_list\tgpu\titerations\texpansion_topk\tresult_dir\tlog_path\n" > "$manifest_path"

cd "$APP_DIR"

export PYTHONPATH="$APP_DIR:$APP_DIR/packages/rdchiral:$APP_DIR/packages/mlp_retrosyn${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}${PYTHONPATH:+:$PYTHONPATH}"

for radius in "${RADII[@]}"; do
  rd_list="[(${radius},0)]"
  result_dir="$dataset_root/R${radius}_D0"
  log_path="$dataset_root/R${radius}_D0.log"

  args=(
    retro_plan.py
    --seed "$SEED"
    --use_value_fn
    --expansion_topk "$EXPANSION_TOPK"
    --one_step_type template_free
    --CSS
    --RD_list "$rd_list"
    --DICT
    --iterations "$ITERATIONS"
    --gpu "$GPU"
    --test_routes "$DATASET"
    --result_folder "$result_dir"
    --viz_dir "$result_dir/viz"
  )

  if [[ "$COLLECT_EXPANSION_DATA" == "1" ]]; then
    args+=(--collect_expansion_data)
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$DATASET_LABEL" "$radius" "$rd_list" "$GPU" "$ITERATIONS" "$EXPANSION_TOPK" "$result_dir" "$log_path" >> "$manifest_path"

  echo "START ${DATASET_LABEL} radius=${radius} RD_list=${rd_list} $(date --iso-8601=seconds)" | tee -a "$log_path"
  "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log_path"
  echo "END ${DATASET_LABEL} radius=${radius} RD_list=${rd_list} $(date --iso-8601=seconds)" | tee -a "$log_path"
done

echo "Run root: $dataset_root"
