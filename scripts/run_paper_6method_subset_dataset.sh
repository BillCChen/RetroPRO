#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_paper_6method_subset_dataset.sh \
    --dataset <pth_hard|uspto190> \
    --dataset-label <pistachio_hard|uspto190> \
    --template-dataset <pistachio_hard_100|uspto190> \
    --retropro-app-dir /abs/path/RetroPRO/retro_star \
    --output-root /abs/path/results/data_collection \
    [--python-bin /abs/path/env/bin/python] \
    [--gpu 0] [--template-gpu 0] [--timestamp YYYYMMDD_HHMMSS] \
    [--template-root /abs/path/test_synthetic_route_planning] \
    [--template-launcher /abs/path/launch_pistachio_template_runs.sh] \
    [--desp-python-bin /abs/path/conda_env/desp/bin/python] \
    [--pdvn-python-bin /abs/path/conda_env/pdvn/bin/python] \
    [--run-template-baselines]

Runs the RetroPRO part of the paper 6-method subset:
  template_free_css_dict_topk8
  template_free_no_css_dict_topk8

When --run-template-baselines is supplied, also launches the template-based
baseline group from test_synthetic_route_planning:
  DESP F2E, DESP F2F, PDVN Retro, PDVN MCTS

Environment overrides:
  SEED=42
  RETROPRO_ITERATIONS=1000
  TEMPLATE_ITERATIONS=<dataset wrapper default>
  EXPANSION_TOPK=8
  RD_LIST='[(7,0),(3,0)]'
  PYTHONPATH_EXTRA=/extra/path
  TEMPLATE_LAUNCHER=/abs/path/launch_pistachio_template_runs.sh
  DESP_PYTHON_BIN=/abs/path/conda_env/desp/bin/python
  PDVN_PYTHON_BIN=/abs/path/conda_env/pdvn/bin/python
USAGE
}

DATASET=""
DATASET_LABEL=""
TEMPLATE_DATASET=""
RETROPRO_APP_DIR=""
OUTPUT_ROOT=""
PYTHON_BIN="${PYTHON_BIN:-}"
GPU="${GPU:-0}"
TEMPLATE_GPU="${TEMPLATE_GPU:-}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TEMPLATE_ROOT="${TEMPLATE_ROOT:-}"
TEMPLATE_LAUNCHER="${TEMPLATE_LAUNCHER:-}"
DESP_PYTHON_BIN="${DESP_PYTHON_BIN:-}"
PDVN_PYTHON_BIN="${PDVN_PYTHON_BIN:-}"
RUN_TEMPLATE_BASELINES=0
DESP_TEST_PATH="${DESP_TEST_PATH:-}"
PDVN_SOURCE_TXT="${PDVN_SOURCE_TXT:-}"
PDVN_TEST_ROUTES="${PDVN_TEST_ROUTES:-}"

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
    --template-dataset)
      TEMPLATE_DATASET="$2"
      shift 2
      ;;
    --retropro-app-dir|--app-dir)
      RETROPRO_APP_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --template-gpu)
      TEMPLATE_GPU="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --template-root)
      TEMPLATE_ROOT="$2"
      shift 2
      ;;
    --template-launcher)
      TEMPLATE_LAUNCHER="$2"
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
    --desp-test-path)
      DESP_TEST_PATH="$2"
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
    --run-template-baselines)
      RUN_TEMPLATE_BASELINES=1
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

if [[ -z "$DATASET" || -z "$DATASET_LABEL" || -z "$TEMPLATE_DATASET" || -z "$RETROPRO_APP_DIR" || -z "$OUTPUT_ROOT" ]]; then
  echo "[error] --dataset, --dataset-label, --template-dataset, --retropro-app-dir, and --output-root are required" >&2
  usage
  exit 1
fi

case "$DATASET" in
  pth_hard|uspto190) ;;
  *)
    echo "[error] unsupported RetroPRO dataset: $DATASET" >&2
    exit 1
    ;;
esac

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi

if [[ -z "$TEMPLATE_GPU" ]]; then
  TEMPLATE_GPU="$GPU"
fi

case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$(pwd)/$OUTPUT_ROOT" ;;
esac

if [[ ! -d "$RETROPRO_APP_DIR" ]]; then
  echo "[error] RetroPRO app dir not found: $RETROPRO_APP_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python bin not executable: $PYTHON_BIN" >&2
  exit 1
fi

SEED="${SEED:-42}"
RETROPRO_ITERATIONS="${RETROPRO_ITERATIONS:-1000}"
TEMPLATE_ITERATIONS="${TEMPLATE_ITERATIONS:-1000}"
EXPANSION_TOPK="${EXPANSION_TOPK:-8}"
RD_LIST="${RD_LIST:-[(7,0),(3,0)]}"

run_root="$OUTPUT_ROOT/${DATASET_LABEL}_paper_6methods_topk${EXPANSION_TOPK}_retropro_iter${RETROPRO_ITERATIONS}_template_iter${TEMPLATE_ITERATIONS}_${TIMESTAMP}"
mkdir -p "$run_root"

write_manifest() {
  cat > "$run_root/run_manifest.tsv" <<MANIFEST
method	engine	dataset	iterations	output
template_free_css_dict_topk${EXPANSION_TOPK}	RetroPRO	${DATASET}	${RETROPRO_ITERATIONS}	${run_root}/template_free_css_dict_topk${EXPANSION_TOPK}
template_free_no_css_dict_topk${EXPANSION_TOPK}	RetroPRO	${DATASET}	${RETROPRO_ITERATIONS}	${run_root}/template_free_no_css_dict_topk${EXPANSION_TOPK}
topk${EXPANSION_TOPK}_desp_f2e	DESP	${TEMPLATE_DATASET}	${TEMPLATE_ITERATIONS}	${TEMPLATE_ROOT:-not_requested}
topk${EXPANSION_TOPK}_desp_f2f	DESP	${TEMPLATE_DATASET}	${TEMPLATE_ITERATIONS}	${TEMPLATE_ROOT:-not_requested}
topk${EXPANSION_TOPK}_pdvn_retro	PDVN	${TEMPLATE_DATASET}	${TEMPLATE_ITERATIONS}	${TEMPLATE_ROOT:-not_requested}
topk${EXPANSION_TOPK}_pdvn_mcts	PDVN	${TEMPLATE_DATASET}	${TEMPLATE_ITERATIONS}	${TEMPLATE_ROOT:-not_requested}
MANIFEST
}

run_retropro_one() {
  local method_label="$1"
  local use_css="$2"

  local result_dir="$run_root/$method_label"
  local log_path="$run_root/${method_label}.log"

  local args=(
    retro_plan.py
    --seed "$SEED"
    --use_value_fn
    --expansion_topk "$EXPANSION_TOPK"
    --one_step_type template_free
    --DICT
    --iterations "$RETROPRO_ITERATIONS"
    --gpu "$GPU"
    --test_routes "$DATASET"
    --collect_expansion_data
    --result_folder "$result_dir"
    --viz_dir "$result_dir/viz"
  )

  if [[ "$use_css" == "1" ]]; then
    args+=(--CSS --RD_list "$RD_LIST")
  fi

  (
    cd "$RETROPRO_APP_DIR"
    export PYTHONPATH="$RETROPRO_APP_DIR:$RETROPRO_APP_DIR/packages/rdchiral:$RETROPRO_APP_DIR/packages/mlp_retrosyn${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}${PYTHONPATH:+:$PYTHONPATH}"
    echo "START ${method_label} $(date --iso-8601=seconds)" | tee -a "$log_path"
    "$PYTHON_BIN" "${args[@]}" 2>&1 | tee -a "$log_path"
    echo "END ${method_label} $(date --iso-8601=seconds)" | tee -a "$log_path"
  )
}

run_template_baselines() {
  if [[ -z "$TEMPLATE_ROOT" ]]; then
    echo "[error] --template-root is required when --run-template-baselines is used" >&2
    exit 1
  fi

  local launcher="$TEMPLATE_LAUNCHER"
  if [[ -z "$launcher" ]]; then
    launcher="$TEMPLATE_ROOT/remote_jobs/launch_pistachio_template_runs.sh"
  fi
  if [[ ! -f "$launcher" && -f "$TEMPLATE_ROOT/artifacts/inference_scripts/launch_pistachio_template_runs.sh" ]]; then
    launcher="$TEMPLATE_ROOT/artifacts/inference_scripts/launch_pistachio_template_runs.sh"
  fi
  if [[ ! -f "$launcher" ]]; then
    echo "[error] template launcher not found: $launcher" >&2
    exit 1
  fi

  local template_parent
  template_parent="$(cd "$TEMPLATE_ROOT/.." && pwd)"
  local desp_python_bin="$DESP_PYTHON_BIN"
  local pdvn_python_bin="$PDVN_PYTHON_BIN"
  if [[ -z "$desp_python_bin" && -x "$template_parent/conda_env/desp/bin/python" ]]; then
    desp_python_bin="$template_parent/conda_env/desp/bin/python"
  fi
  if [[ -z "$pdvn_python_bin" && -x "$template_parent/conda_env/pdvn/bin/python" ]]; then
    pdvn_python_bin="$template_parent/conda_env/pdvn/bin/python"
  fi
  if [[ -n "$desp_python_bin" && ! -x "$desp_python_bin" ]]; then
    echo "[error] DESP python bin not executable: $desp_python_bin" >&2
    exit 1
  fi
  if [[ -n "$pdvn_python_bin" && ! -x "$pdvn_python_bin" ]]; then
    echo "[error] PDVN python bin not executable: $pdvn_python_bin" >&2
    exit 1
  fi

  local patched_launcher="$run_root/launch_template_baselines_${DATASET_LABEL}.sh"
  sed "s|^ROOT=.*|ROOT=\"$TEMPLATE_ROOT\"|" "$launcher" > "$patched_launcher"
  python3 - "$patched_launcher" "$TEMPLATE_ROOT" "$desp_python_bin" "$pdvn_python_bin" <<'PY_PATCHED_LAUNCHER'
from pathlib import Path
import sys

launcher = Path(sys.argv[1])
template_root = sys.argv[2]
desp_python = sys.argv[3]
pdvn_python = sys.argv[4]
text = launcher.read_text()
if desp_python:
    text = text.replace("$ROOT/desp/.venv/bin/python", desp_python)
if pdvn_python:
    text = text.replace("$ROOT/PDVN/.venv/bin/python", pdvn_python)
pdvn_pythonpath = ":".join(
    [
        f"{template_root}/PDVN",
        f"{template_root}/PDVN/retro_star/packages/rdchiral",
        f"{template_root}/PDVN/retro_star/packages/mlp_retrosyn",
    ]
)
text = text.replace(
    "cd '$ROOT/PDVN/retro_star';",
    f"export PYTHONPATH='{pdvn_pythonpath}':${{PYTHONPATH:-}}; cd '$ROOT/PDVN/retro_star';",
)
launcher.write_text(text)
PY_PATCHED_LAUNCHER
  chmod +x "$patched_launcher"

  local run_suffix="${DATASET_LABEL}_paper_6methods_retrotopk${EXPANSION_TOPK}_template_iter${TEMPLATE_ITERATIONS}"
  local env_args=(
    "RUN_TS=$TIMESTAMP"
    "GPU_ID=$TEMPLATE_GPU"
    "DATASET=$TEMPLATE_DATASET"
    "RUN_SUFFIX=$run_suffix"
    "RETRO_TOPK=$EXPANSION_TOPK"
    "ITERATION_LIMIT=$TEMPLATE_ITERATIONS"
    "RUN_GROUP=all"
  )

  if [[ -n "$DESP_TEST_PATH" ]]; then
    env_args+=("DESP_TEST_PATH=$DESP_TEST_PATH")
  fi
  if [[ -n "$PDVN_SOURCE_TXT" ]]; then
    env_args+=("PDVN_SOURCE_TXT=$PDVN_SOURCE_TXT")
  fi
  if [[ -n "$PDVN_TEST_ROUTES" ]]; then
    env_args+=("PDVN_TEST_ROUTES=$PDVN_TEST_ROUTES")
  fi

  echo "START template baselines $(date --iso-8601=seconds)" | tee -a "$run_root/template_baselines.log"
  env "${env_args[@]}" bash "$patched_launcher" 2>&1 | tee -a "$run_root/template_baselines.log"
  echo "END template baselines $(date --iso-8601=seconds)" | tee -a "$run_root/template_baselines.log"
}

write_manifest
run_retropro_one "template_free_css_dict_topk${EXPANSION_TOPK}" 1
run_retropro_one "template_free_no_css_dict_topk${EXPANSION_TOPK}" 0

if [[ "$RUN_TEMPLATE_BASELINES" == "1" ]]; then
  run_template_baselines
else
  echo "Template baselines skipped. Use --run-template-baselines --template-root /path/to/test_synthetic_route_planning to launch DESP/PDVN."
fi

echo "Run root: $run_root"
