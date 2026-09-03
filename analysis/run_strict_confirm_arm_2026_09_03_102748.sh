#!/usr/bin/env bash
# One strict-confirmation GPU arm on BJMU (inside a Slurm gpu job).
# Usage: run_strict_confirm_arm.sh <mode> <test_routes> <seed> <result_dir>
#   mode: nonstrict | strict   test_routes: pth_hard | uspto190
set -euo pipefail

MODE="$1"
TEST_ROUTES="$2"
SEED="$3"
RESULT_DIR="$4"
ITERATIONS="${5:-1000}"
ROUTE_LIMIT="${6:-0}"

WT="${RETROPRO_WT:-/lustre1/liuzm/liuzm_chenqx/RetroPRO_strict_confirm}"
APP="$WT/retro_star"
PY=/home/liuzm/liuzm_chenqx/.conda/envs/retropro/bin/python

case "$MODE" in
  nonstrict) ;;
  strict) ;;
  cached_strict) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
case "$TEST_ROUTES" in
  pth_hard|uspto190) ;;
  *) echo "unknown test_routes: $TEST_ROUTES" >&2; exit 2 ;;
esac

if [[ -d "$RESULT_DIR" ]] && find "$RESULT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Result directory is not empty: $RESULT_DIR" >&2
  exit 1
fi
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd)"

if ! git -C "$WT" diff --quiet || ! git -C "$WT" diff --cached --quiet; then
  echo "Tracked files are dirty; refuse to start an unauditable run." >&2
  exit 1
fi
code_commit="$(git -C "$WT" rev-parse HEAD)"
started_at="$(date --iso-8601=seconds)"
printf '{\n  "as_of": "%s",\n  "code_commit": "%s",\n  "mode": "%s",\n  "test_routes": "%s",\n  "seed": %s,\n  "iterations": %s,\n  "route_limit": %s,\n  "execution_mode": "serial",\n  "strict_topk": %s,\n  "effective_cache": %s\n}\n' \
  "$started_at" "$code_commit" "$MODE" "$TEST_ROUTES" "$SEED" "$ITERATIONS" "$ROUTE_LIMIT" \
  "$([ "$MODE" = strict ] && echo 1 || echo 0)" \
  "$([ "$MODE" = strict ] && echo 1 || echo 0)" \
  >"$RESULT_DIR/run_manifest.json"

export PYTHONPATH="$APP:$APP/packages/rdchiral:$APP/packages/mlp_retrosyn"
export PYTHONHASHSEED=0
export TP_FREE_RETRO_BATCH_SIZE=512
export TP_FREE_FORWARD_BATCH_SIZE=512
export TP_FREE_MAPPER_BATCH_SIZE=256
export TP_FREE_DICT_DUMP_ON_EXIT=1
export TP_FREE_DICT_DUMP_DIR="$RESULT_DIR"
export TP_FREE_FRAGMENT_YIELD_LOG="$RESULT_DIR/fragment_yield.jsonl"
export TP_FREE_RETRO_CANDIDATE_LOG="$RESULT_DIR/retro_candidates.jsonl"
if [[ "$MODE" == "strict" ]]; then
  export TP_FREE_CSS_STRICT_TOPK=1
  export TP_FREE_EFFECTIVE_CACHE=1
fi
if [[ "$MODE" == "cached_strict" ]]; then
  export TP_FREE_CSS_SAMPLER=cached_strict
  export TP_FREE_CACHED_STRICT_EXPLORATION=2
fi

route_args=()
if [[ "$ROUTE_LIMIT" -gt 0 ]]; then
  route_args=(--route_limit "$ROUTE_LIMIT")
fi

cd "$APP"
{
  echo "START $MODE $TEST_ROUTES seed=$SEED $(date --iso-8601=seconds) commit=$code_commit"
  echo "node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
  "$PY" -c "import mlp_retrosyn,os,torch;print('mlp_retrosyn:',os.path.dirname(mlp_retrosyn.__file__));print('cuda:',torch.cuda.is_available())"
} >"$RESULT_DIR/stdout.log" 2>"$RESULT_DIR/stderr.log"

"$PY" retro_plan.py \
  --seed "$SEED" --use_value_fn --expansion_topk 8 \
  --one_step_type template_free --CSS --RD_list "[(7,0),(3,0)]" --DICT \
  --iterations "$ITERATIONS" --gpu 0 \
  --test_routes "$TEST_ROUTES" \
  --result_folder "$RESULT_DIR" \
  --collect_expansion_data \
  "${route_args[@]}" \
  >>"$RESULT_DIR/stdout.log" 2>>"$RESULT_DIR/stderr.log"

echo "END $MODE $TEST_ROUTES seed=$SEED $(date --iso-8601=seconds)" >>"$RESULT_DIR/stdout.log"
