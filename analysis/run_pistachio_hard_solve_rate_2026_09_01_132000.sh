#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
  echo "Usage: $0 OUTPUT_DIR [ITERATIONS] [ROUTE_LIMIT]" >&2
  exit 2
fi

output_dir="$1"
iterations="${2:-1000}"
route_limit="${3:-0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chenqixuan/miniconda3/envs/unirxn2/bin/python"

if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Output directory is not empty: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

cd "$repo_root/retro_star"
if ! git -C "$repo_root" diff --quiet || ! git -C "$repo_root" diff --cached --quiet; then
  echo "Tracked files are dirty; refuse to start an unauditable run." >&2
  exit 1
fi
code_commit="$(git -C "$repo_root" rev-parse HEAD)"
started_at="$(date --iso-8601=seconds)"
printf '{\n  "as_of": "%s",\n  "code_commit": "%s",\n  "iterations": %s,\n  "route_limit": %s,\n  "sampler": "yield8_hybrid",\n  "guardrail_slots": 6\n}\n' \
  "$started_at" "$code_commit" "$iterations" "$route_limit" \
  >"$output_dir/run_manifest_2026_09_01_132000.json"

export PYTHONHASHSEED=0
export TP_FREE_CSS_SAMPLER=yield8_hybrid
export TP_FREE_YIELD_GUARDRAIL_SLOTS=6
export TP_FREE_CSS_STRICT_TOPK=1
export TP_FREE_EFFECTIVE_CACHE=1
export TP_FREE_YIELD_MAX_TRIPLES=256
export TP_FREE_RETRO_BATCH_SIZE=128
export TP_FREE_FORWARD_BATCH_SIZE=128
export TP_FREE_MAPPER_BATCH_SIZE=128
export TP_FREE_DICT_DUMP_ON_EXIT=1
export TP_FREE_FRAGMENT_YIELD_LOG="$output_dir/fragment_yield.jsonl"

route_args=()
if [[ "$route_limit" -gt 0 ]]; then
  route_args=(--route_limit "$route_limit")
fi

"$python_bin" retro_plan.py \
  --gpu 1 \
  --seed 42 \
  --use_value_fn \
  --expansion_topk 8 \
  --iterations "$iterations" \
  --one_step_type template_free \
  --CSS \
  --RD_list "[(3,0)]" \
  --DICT \
  --test_routes pth_hard \
  "${route_args[@]}" \
  --multi_pool \
  --parallel_num 16 \
  --collect_expansion_data \
  --result_folder "$output_dir" \
  >"$output_dir/stdout.log" 2>"$output_dir/stderr.log"
