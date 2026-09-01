"""Verify deterministic equivalence of two tree-search runs."""

import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-limit", type=int, default=0)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reaction_key(row):
    return "%s>>%s" % (
        row["expanded_mol"],
        ".".join(sorted(row.get("reactants") or [])),
    )


def route_signature(route):
    if route is None:
        return None
    steps = []
    for node_id, children in enumerate(route.children):
        if not children:
            continue
        steps.append(
            "%s>>%s"
            % (
                route.mols[node_id],
                ".".join(sorted(route.mols[child_id] for child_id in children)),
            )
        )
    return sorted(steps)


def load_run(run_dir, target_limit=0):
    with (run_dir / "plan.pkl").open("rb") as handle:
        plan = pickle.load(handle)
    candidates = load_jsonl(
        run_dir / "expansion_data" / "reaction_candidates.jsonl"
    )
    nodes = load_jsonl(run_dir / "expansion_data" / "node_expansions.jsonl")
    fragments = load_jsonl(run_dir / "fragment_yield.jsonl")
    if target_limit > 0:
        candidates = [
            row for row in candidates if int(row["target_id"]) < target_limit
        ]
        nodes = [row for row in nodes if int(row["target_id"]) < target_limit]
        fragments = [
            row for row in fragments if int(row["task_id"]) < target_limit
        ]
        plan = {
            key: value[:target_limit] if isinstance(value, list) else value
            for key, value in plan.items()
        }

    candidate_sets = defaultdict(set)
    candidate_sequences = defaultdict(list)
    for row in candidates:
        target_id = int(row["target_id"])
        key = reaction_key(row)
        candidate_sets[target_id].add(key)
        candidate_sequences[target_id].append(
            (int(row["iteration"]), key, tuple(row.get("source_fragments") or []))
        )

    node_sequences = defaultdict(list)
    for row in nodes:
        target_id = int(row["target_id"])
        node_sequences[target_id].append(
            (
                int(row["iteration"]),
                row["expanded_mol"],
                int(row["num_model_candidates"]),
                int(row["num_valid_candidates"]),
            )
        )

    fragment_multisets = defaultdict(Counter)
    for row in fragments:
        normalized = dict(row)
        normalized.pop("timestamp_epoch", None)
        target_id = int(normalized.pop("task_id"))
        fragment_multisets[target_id][
            json.dumps(normalized, sort_keys=True)
        ] += 1

    return {
        "success": [bool(value) for value in plan["succ"]],
        "iterations": list(plan["iter"]),
        "route_signatures": [route_signature(route) for route in plan["routes"]],
        "candidate_sets": dict(candidate_sets),
        "candidate_sequences": dict(candidate_sequences),
        "node_sequences": dict(node_sequences),
        "fragment_multisets": dict(fragment_multisets),
        "input_sha256": {
            "plan": sha256(run_dir / "plan.pkl"),
            "reaction_candidates": sha256(
                run_dir / "expansion_data" / "reaction_candidates.jsonl"
            ),
            "node_expansions": sha256(
                run_dir / "expansion_data" / "node_expansions.jsonl"
            ),
            "fragment_yield": sha256(run_dir / "fragment_yield.jsonl"),
        },
    }


def compare_runs(run_a, run_b):
    target_ids = sorted(set(run_a["candidate_sets"]) | set(run_b["candidate_sets"]))
    per_target = []
    for target_id in target_ids:
        set_a = run_a["candidate_sets"].get(target_id, set())
        set_b = run_b["candidate_sets"].get(target_id, set())
        union = set_a | set_b
        per_target.append(
            {
                "target_id": target_id,
                "reaction_set_equal": set_a == set_b,
                "reaction_set_jaccard": (
                    len(set_a & set_b) / len(union) if union else 1.0
                ),
                "candidate_sequence_equal": (
                    run_a["candidate_sequences"].get(target_id, [])
                    == run_b["candidate_sequences"].get(target_id, [])
                ),
                "node_sequence_equal": (
                    run_a["node_sequences"].get(target_id, [])
                    == run_b["node_sequences"].get(target_id, [])
                ),
                "fragment_multiset_equal": (
                    run_a["fragment_multisets"].get(target_id, Counter())
                    == run_b["fragment_multisets"].get(target_id, Counter())
                ),
            }
        )

    return {
        "success_equal": run_a["success"] == run_b["success"],
        "iterations_equal": run_a["iterations"] == run_b["iterations"],
        "route_signatures_equal": (
            run_a["route_signatures"] == run_b["route_signatures"]
        ),
        "all_reaction_sets_equal": all(
            row["reaction_set_equal"] for row in per_target
        ),
        "all_candidate_sequences_equal": all(
            row["candidate_sequence_equal"] for row in per_target
        ),
        "all_node_sequences_equal": all(
            row["node_sequence_equal"] for row in per_target
        ),
        "all_fragment_multisets_equal": all(
            row["fragment_multiset_equal"] for row in per_target
        ),
        "per_target": per_target,
    }


def json_ready_run(run):
    return {"input_sha256": run["input_sha256"]}


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    retro_star_root = repo_root / "retro_star"
    if str(retro_star_root) not in sys.path:
        sys.path.insert(0, str(retro_star_root))

    run_a = load_run(args.run_a, args.target_limit)
    run_b = load_run(args.run_b, args.target_limit)
    report = {
        "schema_version": "1.0.0",
        "as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_a": {"path": str(args.run_a.resolve()), **json_ready_run(run_a)},
        "run_b": {"path": str(args.run_b.resolve()), **json_ready_run(run_b)},
        "target_limit": args.target_limit,
        "comparison": compare_runs(run_a, run_b),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))
    print("output=%s" % args.output)
    print("sha256=%s" % sha256(args.output))


if __name__ == "__main__":
    main()
