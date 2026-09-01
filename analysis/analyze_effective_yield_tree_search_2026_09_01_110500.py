"""Analyze matched production and yield8_hybrid tree-search runs."""

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--baseline-arm", default="production")
    parser.add_argument("--challenger-arm", default="hybrid")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def reaction_key(row):
    reactants = ".".join(sorted(row.get("reactants") or []))
    return "%s>>%s" % (row.get("expanded_mol", ""), reactants)


def route_step_keys(route):
    if route is None:
        return []
    steps = []
    for node_id, children in enumerate(route.children):
        if not children:
            continue
        reactants = ".".join(sorted(route.mols[child_id] for child_id in children))
        steps.append(
            {
                "route_node_id": node_id,
                "product": route.mols[node_id],
                "reactants": reactants,
                "reaction_key": "%s>>%s" % (route.mols[node_id], reactants),
            }
        )
    return steps


def paired_bootstrap(values, samples, seed):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    chunk_size = max(1, min(samples, 20000))
    means = []
    remaining = samples
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, array.size, size=(current, array.size))
        means.append(array[indices].mean(axis=1))
        remaining -= current
    distribution = np.concatenate(means)
    return {
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
    }


def summarize_arm(arm_dir):
    expansion_dir = arm_dir / "expansion_data"
    metadata_path = expansion_dir / "metadata.json"
    node_path = expansion_dir / "node_expansions.jsonl"
    candidate_path = expansion_dir / "reaction_candidates.jsonl"
    fragment_path = arm_dir / "fragment_yield.jsonl"
    plan_path = arm_dir / "plan.pkl"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_count = int(metadata["target_count"])
    targets = list(metadata["targets"])

    with plan_path.open("rb") as handle:
        plan = pickle.load(handle)

    per_target = [
        {
            "target_id": target_id,
            "target": targets[target_id],
            "expansions": 0,
            "candidate_rows": 0,
            "unique_reactions": set(),
            "unique_expected_addable_reactions": set(),
            "candidates_with_sources": 0,
            "source_fragment_links": 0,
            "fragment_rows": 0,
            "selected_fragments": 0,
            "actual_augmentations": 0,
            "retro_raw": 0,
            "retro_valid": 0,
            "forward_consistent": 0,
            "mapped": 0,
            "templates": 0,
            "fragment_reaction_credits": 0,
        }
        for target_id in range(target_count)
    ]
    family_selection_counts = defaultdict(int)
    reaction_observations = [dict() for _ in range(target_count)]

    for row in load_jsonl(node_path):
        per_target[int(row["target_id"])]["expansions"] += 1

    for row in load_jsonl(candidate_path):
        target_id = int(row["target_id"])
        target = per_target[target_id]
        key = reaction_key(row)
        target["candidate_rows"] += 1
        target["unique_reactions"].add(key)
        if row.get("expected_added_to_tree"):
            target["unique_expected_addable_reactions"].add(key)
        sources = row.get("source_fragments") or []
        if sources:
            target["candidates_with_sources"] += 1
            target["source_fragment_links"] += len(sources)
        observation = reaction_observations[target_id].setdefault(
            key,
            {
                "first_iteration": int(row["iteration"]),
                "source_fragments": set(),
            },
        )
        observation["first_iteration"] = min(
            observation["first_iteration"], int(row["iteration"])
        )
        observation["source_fragments"].update(sources)

    for row in load_jsonl(fragment_path):
        task_id = row.get("task_id")
        if task_id is None:
            raise ValueError("fragment telemetry is missing task_id: %s" % fragment_path)
        target = per_target[int(task_id)]
        target["fragment_rows"] += 1
        target["selected_fragments"] += 1
        target["actual_augmentations"] += int(row["augmentation_count"])
        target["retro_raw"] += int(row["retro_raw_count"])
        target["retro_valid"] += int(row["retro_valid_count"])
        target["forward_consistent"] += int(row["forward_consistent_count"])
        target["mapped"] += int(row["mapped_count"])
        target["templates"] += int(row["template_extracted_count"])
        target["fragment_reaction_credits"] += int(
            row["full_product_unique_reaction_count"]
        )
        for family in row.get("families") or []:
            family_selection_counts[str(family)] += 1

    cumulative_times = plan.get("cumulated_time") or [None] * target_count
    previous = 0.0
    reaction_sets = []
    expected_addable_reaction_sets = []
    for target_id, row in enumerate(per_target):
        cumulative = cumulative_times[target_id]
        if cumulative is None:
            duration = None
        else:
            duration = float(cumulative) - previous
            previous = float(cumulative)
        row["duration_seconds"] = duration
        row["success"] = bool(plan["succ"][target_id])
        row["iterations_used"] = int(plan["iter"][target_id])
        row["route_cost"] = plan["route_costs"][target_id]
        row["route_length"] = plan["route_lens"][target_id]
        reaction_sets.append(set(row["unique_reactions"]))
        expected_addable_reaction_sets.append(
            set(row["unique_expected_addable_reactions"])
        )
        row["unique_reaction_count"] = len(row.pop("unique_reactions"))
        row["unique_expected_addable_reaction_count"] = len(
            row.pop("unique_expected_addable_reactions")
        )

    totals = defaultdict(float)
    integer_metrics = (
        "expansions",
        "candidate_rows",
        "unique_reaction_count",
        "unique_expected_addable_reaction_count",
        "candidates_with_sources",
        "source_fragment_links",
        "fragment_rows",
        "selected_fragments",
        "actual_augmentations",
        "retro_raw",
        "retro_valid",
        "forward_consistent",
        "mapped",
        "templates",
        "fragment_reaction_credits",
    )
    for row in per_target:
        for metric in integer_metrics:
            totals[metric] += row[metric]

    total_duration = sum(
        row["duration_seconds"] or 0.0 for row in per_target
    )
    total_valid = totals["retro_valid"]
    total_augmentations = totals["actual_augmentations"]
    unique_reactions = totals["unique_reaction_count"]
    totals.update(
        {
            "completed_targets": sum(plan_value is not None for plan_value in plan["succ"]),
            "successful_targets": sum(bool(value) for value in plan["succ"]),
            "total_duration_seconds": total_duration,
            "unique_reactions_per_actual_augmentation": (
                unique_reactions / total_augmentations if total_augmentations else None
            ),
            "forward_consistent_per_valid_retro": (
                totals["forward_consistent"] / total_valid if total_valid else None
            ),
            "unique_reactions_per_second": (
                unique_reactions / total_duration if total_duration else None
            ),
            "unique_reactions_per_expansion": (
                unique_reactions / totals["expansions"]
                if totals["expansions"]
                else None
            ),
            "expected_addable_unique_reactions_per_expansion": (
                totals["unique_expected_addable_reaction_count"]
                / totals["expansions"]
                if totals["expansions"]
                else None
            ),
            "unique_reaction_fraction_of_candidate_rows": (
                unique_reactions / totals["candidate_rows"]
                if totals["candidate_rows"]
                else None
            ),
            "mean_successful_route_cost": (
                float(
                    np.mean(
                        [
                            row["route_cost"]
                            for row in per_target
                            if row["route_cost"] is not None
                        ]
                    )
                )
                if any(row["route_cost"] is not None for row in per_target)
                else None
            ),
            "mean_successful_route_length": (
                float(
                    np.mean(
                        [
                            row["route_length"]
                            for row in per_target
                            if row["route_length"] is not None
                        ]
                    )
                )
                if any(row["route_length"] is not None for row in per_target)
                else None
            ),
            "success_by_iteration": {
                str(limit): sum(
                    row["success"] and row["iterations_used"] <= limit
                    for row in per_target
                )
                for limit in (10, 25, 50, 100)
            },
            "family_selection_counts": dict(
                sorted(family_selection_counts.items())
            ),
        }
    )
    totals = {
        key: int(value) if key in integer_metrics else value
        for key, value in totals.items()
    }

    return {
        "run_params": metadata["run_params"],
        "totals": totals,
        "per_target": per_target,
        "input_sha256": {
            "plan": sha256(plan_path),
            "metadata": sha256(metadata_path),
            "node_expansions": sha256(node_path),
            "reaction_candidates": sha256(candidate_path),
            "fragment_yield": sha256(fragment_path),
        },
        "_reaction_sets": reaction_sets,
        "_expected_addable_reaction_sets": expected_addable_reaction_sets,
        "_reaction_observations": reaction_observations,
        "_route_steps": [route_step_keys(route) for route in plan["routes"]],
    }


def compare_arms(
    baseline,
    challenger,
    bootstrap,
    seed,
    baseline_name,
    challenger_name,
):
    prod_rows = baseline["per_target"]
    hybrid_rows = challenger["per_target"]
    if len(prod_rows) != len(hybrid_rows):
        raise ValueError("arm target counts differ")
    if [row["target"] for row in prod_rows] != [row["target"] for row in hybrid_rows]:
        raise ValueError("arm target orders differ")

    metrics = (
        "success",
        "iterations_used",
        "duration_seconds",
        "expansions",
        "candidate_rows",
        "unique_reaction_count",
        "unique_expected_addable_reaction_count",
        "actual_augmentations",
        "forward_consistent",
        "templates",
    )
    paired = {}
    for offset, metric in enumerate(metrics):
        differences = [
            float(hybrid_row[metric]) - float(prod_row[metric])
            for prod_row, hybrid_row in zip(prod_rows, hybrid_rows)
        ]
        stats = paired_bootstrap(differences, bootstrap, seed + offset)
        stats.update(
            {
                "wins": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "losses": sum(value < 0 for value in differences),
                "per_target_differences": differences,
            }
        )
        paired[metric] = stats

    prod_success = [bool(row["success"]) for row in prod_rows]
    hybrid_success = [bool(row["success"]) for row in hybrid_rows]
    success_pairs = {
        "both_success": sum(a and b for a, b in zip(prod_success, hybrid_success)),
        "baseline_only": sum(a and not b for a, b in zip(prod_success, hybrid_success)),
        "challenger_only": sum(b and not a for a, b in zip(prod_success, hybrid_success)),
        "both_fail": sum(not a and not b for a, b in zip(prod_success, hybrid_success)),
    }
    reaction_overlap = []
    for target_id, (prod_set, hybrid_set) in enumerate(
        zip(baseline["_reaction_sets"], challenger["_reaction_sets"])
    ):
        intersection = len(prod_set & hybrid_set)
        union = len(prod_set | hybrid_set)
        reaction_overlap.append(
            {
                "target_id": target_id,
                "intersection": intersection,
                "baseline_only": len(prod_set - hybrid_set),
                "challenger_only": len(hybrid_set - prod_set),
                "union": union,
                "jaccard": intersection / union if union else None,
                "baseline_reaction_retention": (
                    intersection / len(prod_set) if prod_set else None
                ),
            }
        )
    total_intersection = sum(row["intersection"] for row in reaction_overlap)
    total_union = sum(row["union"] for row in reaction_overlap)
    total_production = sum(
        len(value) for value in baseline["_reaction_sets"]
    )
    overlap_summary = {
        "per_target": reaction_overlap,
        "micro_intersection": total_intersection,
        "micro_union": total_union,
        "micro_jaccard": total_intersection / total_union if total_union else None,
        "micro_baseline_reaction_retention": (
            total_intersection / total_production if total_production else None
        ),
        "mean_target_jaccard": float(
            np.mean(
                [
                    row["jaccard"]
                    for row in reaction_overlap
                    if row["jaccard"] is not None
                ]
            )
        ),
        "mean_target_baseline_reaction_retention": float(
            np.mean(
                [
                    row["baseline_reaction_retention"]
                    for row in reaction_overlap
                    if row["baseline_reaction_retention"] is not None
                ]
            )
        ),
    }
    route_transfer = []
    arm_map = {baseline_name: baseline, challenger_name: challenger}
    for source_arm, source_data in arm_map.items():
        for target_id, steps in enumerate(source_data["_route_steps"]):
            if not steps:
                continue
            transfer = {
                "source_arm": source_arm,
                "target_id": target_id,
                "route_length": len(steps),
                "observers": {},
            }
            for observer_arm, observer_data in arm_map.items():
                observations = observer_data["_reaction_observations"][target_id]
                step_observations = []
                for step in steps:
                    observed = observations.get(step["reaction_key"])
                    step_observations.append(
                        {
                            "route_node_id": step["route_node_id"],
                            "product": step["product"],
                            "reactants": step["reactants"],
                            "observed": observed is not None,
                            "first_iteration": (
                                observed["first_iteration"] if observed else None
                            ),
                            "source_fragments": (
                                sorted(observed["source_fragments"])
                                if observed
                                else []
                            ),
                        }
                    )
                recovered = sum(row["observed"] for row in step_observations)
                transfer["observers"][observer_arm] = {
                    "recovered_steps": recovered,
                    "all_steps_recovered": recovered == len(steps),
                    "steps": step_observations,
                }
            route_transfer.append(transfer)
    return {
        "arm_roles": {
            "baseline": baseline_name,
            "challenger": challenger_name,
        },
        "paired_challenger_minus_baseline": paired,
        "success_pairs": success_pairs,
        "reaction_overlap": overlap_summary,
        "route_transfer": route_transfer,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    retro_star_root = repo_root / "retro_star"
    if str(retro_star_root) not in sys.path:
        sys.path.insert(0, str(retro_star_root))

    arm_names = (args.baseline_arm, args.challenger_arm)
    if args.baseline_arm == args.challenger_arm:
        raise ValueError("baseline and challenger arms must differ")
    arms = {
        arm: summarize_arm(args.run_root / arm)
        for arm in arm_names
    }
    comparison = compare_arms(
        arms[args.baseline_arm],
        arms[args.challenger_arm],
        args.bootstrap,
        args.seed,
        args.baseline_arm,
        args.challenger_arm,
    )
    for arm_data in arms.values():
        arm_data.pop("_reaction_sets")
        arm_data.pop("_expected_addable_reaction_sets")
        arm_data.pop("_reaction_observations")
        arm_data.pop("_route_steps")
    report = {
        "schema_version": "1.0.0",
        "as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(args.run_root.resolve()),
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "arms": arms,
        "comparison": comparison,
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
