#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 4 ]]; then
  echo "Usage: $0 OUTPUT_DIR SAMPLER [ITERATIONS] [ROUTE_LIMIT]" >&2
  echo "SAMPLER: anchor8 | random" >&2
  exit 2
fi

output_dir="$1"
sampler="$2"
iterations="${3:-1000}"
route_limit="${4:-0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chenqixuan/miniconda3/envs/unirxn2/bin/python"

case "$sampler" in
  anchor8) rd_list="[(3,0)]" ;;
  random)  rd_list="[(7,0),(3,0)]" ;;
  *) echo "unknown sampler: $sampler" >&2; exit 2 ;;
esac

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
printf '{\n  "as_of": "%s",\n  "code_commit": "%s",\n  "iterations": %s,\n  "route_limit": %s,\n  "execution_mode": "serial",\n  "sampler": "%s",\n  "rd_list": "%s",\n  "strict_topk": 1,\n  "effective_cache": 1,\n  "anchor_max_replace": 2,\n  "anchor_margin": 1\n}\n' \
  "$started_at" "$code_commit" "$iterations" "$route_limit" "$sampler" "$rd_list" \
  >"$output_dir/run_manifest.json"

export PYTHONHASHSEED=0
export TP_FREE_CSS_SAMPLER="$sampler"
export TP_FREE_CSS_STRICT_TOPK=1
export TP_FREE_EFFECTIVE_CACHE=1
export TP_FREE_RETRO_BATCH_SIZE=512
export TP_FREE_FORWARD_BATCH_SIZE=512
export TP_FREE_MAPPER_BATCH_SIZE=256
export TP_FREE_DICT_DUMP_ON_EXIT=1
export TP_FREE_FRAGMENT_YIELD_LOG="$output_dir/fragment_yield.jsonl"
export TP_FREE_RETRO_CANDIDATE_LOG="$output_dir/retro_candidates.jsonl"
if [[ "$sampler" == "anchor8" ]]; then
  export TP_FREE_ANCHOR_MAX_REPLACE="${TP_FREE_ANCHOR_MAX_REPLACE:-2}"
  export TP_FREE_ANCHOR_MARGIN="${TP_FREE_ANCHOR_MARGIN:-1}"
  export TP_FREE_ANCHOR_SCORING="${TP_FREE_ANCHOR_SCORING:-consensus}"
  export TP_FREE_ANCHOR_MULTIVIEW="${TP_FREE_ANCHOR_MULTIVIEW:-1}"
fi

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
  --RD_list "$rd_list" \
  --DICT \
  --test_routes pth_hard \
  "${route_args[@]}" \
  --collect_expansion_data \
  --result_folder "$output_dir" \
  >"$output_dir/stdout.log" 2>"$output_dir/stderr.log"
