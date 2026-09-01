#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$1"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
python_bin="/home/chenqixuan/miniconda3/envs/unirxn2/bin/python"

cd "$repo_root/retro_star"
export PYTHONHASHSEED=0
export TP_FREE_CSS_STRICT_TOPK=1
export TP_FREE_EFFECTIVE_CACHE=1
export TP_FREE_YIELD_MAX_TRIPLES=256
export TP_FREE_YIELD_GUARDRAIL_SLOTS=6
export TP_FREE_DICT_DUMP_ON_EXIT=1
export TP_FREE_CSS_SAMPLER=yield8_hybrid
export TP_FREE_FRAGMENT_YIELD_LOG="$output_dir/fragment_yield.jsonl"

"$python_bin" retro_plan.py \
  --gpu 1 \
  --seed 42 \
  --use_value_fn \
  --expansion_topk 8 \
  --iterations 50 \
  --one_step_type template_free \
  --CSS \
  --RD_list "[(3,0)]" \
  --DICT \
  --test_routes pth_hard \
  --route_limit 10 \
  --collect_expansion_data \
  --result_folder "$output_dir" \
  >"$output_dir/stdout.log" 2>"$output_dir/stderr.log"
