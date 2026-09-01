#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "Usage: $0 ROUTE_LIMIT ITERATIONS OUTPUT_ROOT [DUMP_DICT]" >&2
  exit 2
fi

route_limit="$1"
iterations="$2"
output_root="$3"
dump_dict="${4:-1}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/chenqixuan/miniconda3/envs/unirxn2/bin/python"

mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
cd "$repo_root/retro_star"

export PYTHONHASHSEED=0
export TP_FREE_CSS_STRICT_TOPK=1
export TP_FREE_EFFECTIVE_CACHE=1
export TP_FREE_YIELD_MAX_TRIPLES=256
export TP_FREE_DICT_DUMP_ON_EXIT="$dump_dict"

run_arm() {
  local arm="$1"
  local sampler="$2"
  local rd_list="$3"
  local arm_dir="$output_root/$arm"

  mkdir -p "$arm_dir"
  TP_FREE_CSS_SAMPLER="$sampler" \
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

run_arm production random "[(7,0),(3,0)]"
run_arm hybrid yield8_hybrid "[(3,0)]"

"$python_bin" - "$output_root" "$route_limit" "$iterations" <<'PY'
import hashlib
import json
import pickle
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_arm(arm_dir):
    plan_path = arm_dir / "plan.pkl"
    with plan_path.open("rb") as handle:
        plan = pickle.load(handle)

    candidates_path = arm_dir / "expansion_data" / "reaction_candidates.jsonl"
    node_path = arm_dir / "expansion_data" / "node_expansions.jsonl"
    fragment_path = arm_dir / "fragment_yield.jsonl"

    candidate_rows = 0
    candidates_with_sources = 0
    source_fragment_links = 0
    if candidates_path.exists():
        with candidates_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                candidate_rows += 1
                sources = row.get("source_fragments") or []
                if sources:
                    candidates_with_sources += 1
                    source_fragment_links += len(sources)

    node_rows = sum(1 for _ in node_path.open("r", encoding="utf-8")) if node_path.exists() else 0
    fragment_rows = sum(1 for _ in fragment_path.open("r", encoding="utf-8")) if fragment_path.exists() else 0
    successes = sum(bool(value) for value in plan.get("succ", []))
    completed = sum(value is not None for value in plan.get("succ", []))
    iterations_used = [value for value in plan.get("iter", []) if value is not None]

    return {
        "completed_targets": completed,
        "successful_targets": successes,
        "iterations_used": iterations_used,
        "node_rows": node_rows,
        "candidate_rows": candidate_rows,
        "candidates_with_source_fragments": candidates_with_sources,
        "source_fragment_links": source_fragment_links,
        "fragment_telemetry_rows": fragment_rows,
        "plan_sha256": sha256(plan_path),
        "candidate_log_sha256": sha256(candidates_path) if candidates_path.exists() else None,
        "node_log_sha256": sha256(node_path) if node_path.exists() else None,
        "fragment_log_sha256": sha256(fragment_path) if fragment_path.exists() else None,
    }


def main():
    output_root = Path(sys.argv[1])
    summary = {
        "route_limit": int(sys.argv[2]),
        "iterations": int(sys.argv[3]),
        "arms": {
            arm: summarize_arm(output_root / arm)
            for arm in ("production", "hybrid")
        },
    }
    summary_path = output_root / "tree_search_summary_2026_09_01_105300.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
PY
