"""Benchmark strict-cardinality CSS selection across Triple pool caps."""

import argparse
import hashlib
import json
import pickle
import statistics
import time
from datetime import datetime
from pathlib import Path

from rdkit import RDLogger

from mlp_retrosyn.css_effective_yield import select_effective_yield_fragments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--caps", default="32,64,128,256")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-targets", type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_targets(path, max_targets=0):
    with path.open("rb") as handle:
        entries = pickle.load(handle)
    targets = []
    for entry in entries:
        reaction = entry[0] if isinstance(entry, (list, tuple)) else entry
        if not isinstance(reaction, str) or ">>" not in reaction:
            raise ValueError("route entry does not contain a first-step reaction")
        targets.append(reaction.split(">>", 1)[0])
    if max_targets > 0:
        targets = targets[:max_targets]
    return targets


def percentile(values, probability):
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * probability)), len(ordered) - 1)
    return ordered[index]


def summarize(rows):
    times = [row["selection_seconds"] for row in rows]
    candidates = [row["candidate_count"] for row in rows]
    return {
        "target_count": len(rows),
        "strict_topk_count": sum(not row["capacity_limited"] for row in rows),
        "capacity_limited_count": sum(row["capacity_limited"] for row in rows),
        "selection_seconds": {
            "total": sum(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "p90": percentile(times, 0.90),
            "p99": percentile(times, 0.99),
            "max": max(times),
        },
        "candidate_count": {
            "mean": statistics.mean(candidates),
            "median": statistics.median(candidates),
            "p90": percentile(candidates, 0.90),
            "max": max(candidates),
        },
    }


def main():
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    route_path = Path(args.routes).resolve()
    output_path = Path(args.output).resolve()
    targets = load_targets(route_path, args.max_targets)
    caps = [int(value) for value in args.caps.split(",")]
    by_cap = {}
    per_target = []
    for cap in caps:
        rows = []
        for target_id, target in enumerate(targets):
            started = time.perf_counter()
            selected, metadata = select_effective_yield_fragments(
                target,
                topk=args.topk,
                known_reactions=lambda _fragment: set(),
                exploration_slots=2,
                max_triples=cap,
                seed=args.seed,
            )
            row = {
                "cap": cap,
                "target_id": target_id,
                "candidate_count": metadata["candidate_count"],
                "selected_count": len(selected),
                "capacity_limited": metadata["capacity_limited"],
                "selection_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            per_target.append(row)
        by_cap[str(cap)] = summarize(rows)

    now = datetime.now().astimezone()
    report = {
        "as_of": now.isoformat(),
        "analysis_version": "1.0.0",
        "input": {
            "routes": str(route_path),
            "routes_sha256": sha256_file(route_path),
            "target_count": len(targets),
        },
        "protocol": {
            "caps": caps,
            "topk": args.topk,
            "seed": args.seed,
            "known_reaction_callback": "empty; structural exploration path only",
        },
        "by_cap": by_cap,
        "per_target": per_target,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(output_path), "by_cap": by_cap}, sort_keys=True))


if __name__ == "__main__":
    main()
