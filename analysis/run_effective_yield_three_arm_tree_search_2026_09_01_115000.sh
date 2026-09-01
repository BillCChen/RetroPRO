#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 ROUTE_LIMIT ITERATIONS OUTPUT_ROOT" >&2
  exit 2
fi

route_limit="$1"
iterations="$2"
output_root="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chenqixuan/miniconda3/envs/unirxn2/bin/python"

mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
cd "$repo_root/retro_star"

export PYTHONHASHSEED=0
export TP_FREE_CSS_STRICT_TOPK=1
export TP_FREE_EFFECTIVE_CACHE=1
export TP_FREE_YIELD_MAX_TRIPLES=256
export TP_FREE_DICT_DUMP_ON_EXIT=1

run_arm() {
  local arm="$1"
  local sampler="$2"
  local rd_list="$3"
  local guardrail_slots="$4"
  local arm_dir="$output_root/$arm"

  mkdir -p "$arm_dir"
  TP_FREE_CSS_SAMPLER="$sampler" \
  TP_FREE_YIELD_GUARDRAIL_SLOTS="$guardrail_slots" \
  TP_FREE_FRAGMENT_YIELD_LOG="$arm_dir/fragment_yield.jsonl" \
  "$python_bin" retro_plan.py \
    --gpu 1 \
    --seed 42 \
    --use_value_fn \
    --expansion_topk 8 \
    --iterations "$iterations" \
    --one_step_type template_free \
    --CSS \
    --RD_list "$rd_list" \
    --DICT \
    --test_routes pth_hard \
    --route_limit "$route_limit" \
    --collect_expansion_data \
    --result_folder "$arm_dir" \
    >"$arm_dir/stdout.log" 2>"$arm_dir/stderr.log"
}

run_arm production random "[(7,0),(3,0)]" 4
run_arm hybrid yield8_hybrid "[(3,0)]" 4
run_arm retained6 yield8_hybrid "[(3,0)]" 6
