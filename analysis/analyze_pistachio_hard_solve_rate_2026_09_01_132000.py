"""Compare matched-budget Pistachio-hard solve-rate plans."""

import argparse
import hashlib
import json
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


MATCHED_KEYS = (
    "test_routes",
    "route_limit",
    "starting_molecules",
    "one_step_type",
    "retro_topk",
    "forward_topk",
    "expansion_topk",
    "iterations",
    "multi_pool",
    "parallel_num",
    "seed",
    "use_value_fn",
    "value_model",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [center - radius, center + radius]


def paired_bootstrap(values, samples, seed):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    chunks = []
    remaining = samples
    while remaining:
        current = min(remaining, 20000)
        indices = rng.integers(0, array.size, size=(current, array.size))
        chunks.append(array[indices].mean(axis=1))
        remaining -= current
    distribution = np.concatenate(chunks)
    return {
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
    }


def exact_mcnemar_p(baseline_only, candidate_only):
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2.0 ** discordant)
    return min(1.0, 2.0 * probability)


def summarize_plan(plan):
    success = [bool(value) for value in plan["succ"]]
    iterations = list(plan["iter"])
    if any(value is None for value in plan["succ"]):
        raise ValueError("plan contains unfinished success entries")
    if any(value is None for value in iterations):
        raise ValueError("plan contains unfinished iteration entries")
    total = len(success)
    solved = sum(success)
    thresholds = sorted(set([50, 100, 500, 1000, max(iterations)]))
    total_elapsed = max(
        [float(value) for value in plan.get("cumulated_time", []) if value is not None]
        or [0.0]
    )
    return {
        "target_count": total,
        "solved": solved,
        "solve_rate": solved / total if total else None,
        "solve_rate_wilson95": wilson_interval(solved, total),
        "success_by_iteration": {
            str(limit): sum(
                solved_flag and iteration <= limit
                for solved_flag, iteration in zip(success, iterations)
            )
            for limit in thresholds
        },
        "mean_iterations_all_targets": float(np.mean(iterations)),
        "median_iterations_all_targets": float(np.median(iterations)),
        "mean_iterations_solved": (
            float(np.mean([value for value, flag in zip(iterations, success) if flag]))
            if solved
            else None
        ),
        "total_elapsed_seconds": total_elapsed,
        "successful_route_cost_mean": (
            float(
                np.mean(
                    [
                        value
                        for value, flag in zip(plan["route_costs"], success)
                        if flag and value is not None
                    ]
                )
            )
            if solved
            else None
        ),
        "successful_route_length_mean": (
            float(
                np.mean(
                    [
                        value
                        for value, flag in zip(plan["route_lens"], success)
                        if flag and value is not None
                    ]
                )
            )
            if solved
            else None
        ),
    }


def compare_contracts(baseline, candidate):
    baseline_params = baseline.get("inference_run_params") or {}
    candidate_params = candidate.get("inference_run_params") or {}
    comparison = {
        key: {
            "baseline": baseline_params.get(key),
            "candidate": candidate_params.get(key),
            "equal": baseline_params.get(key) == candidate_params.get(key),
        }
        for key in MATCHED_KEYS
    }
    mismatches = [key for key, row in comparison.items() if not row["equal"]]
    if mismatches:
        raise ValueError("run contracts differ: %s" % ",".join(mismatches))
    return {
        "matched_fields": comparison,
        "baseline_rd_list": baseline_params.get("RD_list_parsed"),
        "candidate_rd_list": candidate_params.get("RD_list_parsed"),
        "baseline_tp_free_env": baseline_params.get("tp_free_env", {}),
        "candidate_tp_free_env": candidate_params.get("tp_free_env", {}),
    }


def compare_plans(baseline, candidate, bootstrap, seed):
    baseline_success = [bool(value) for value in baseline["succ"]]
    candidate_success = [bool(value) for value in candidate["succ"]]
    if len(baseline_success) != len(candidate_success):
        raise ValueError("target counts differ")
    differences = [
        float(candidate_flag) - float(baseline_flag)
        for baseline_flag, candidate_flag in zip(
            baseline_success, candidate_success
        )
    ]
    baseline_only = sum(
        baseline_flag and not candidate_flag
        for baseline_flag, candidate_flag in zip(
            baseline_success, candidate_success
        )
    )
    candidate_only = sum(
        candidate_flag and not baseline_flag
        for baseline_flag, candidate_flag in zip(
            baseline_success, candidate_success
        )
    )
    both_success_ids = [
        target_id
        for target_id, (baseline_flag, candidate_flag) in enumerate(
            zip(baseline_success, candidate_success)
        )
        if baseline_flag and candidate_flag
    ]

    paired_metrics = {}
    for offset, (name, baseline_values, candidate_values) in enumerate(
        (
            ("iterations", baseline["iter"], candidate["iter"]),
            ("route_cost", baseline["route_costs"], candidate["route_costs"]),
            ("route_length", baseline["route_lens"], candidate["route_lens"]),
        )
    ):
        values = [
            float(candidate_values[target_id])
            - float(baseline_values[target_id])
            for target_id in both_success_ids
        ]
        stats = paired_bootstrap(values, bootstrap, seed + offset + 1)
        stats["target_count"] = len(values)
        stats["wins_lower_is_better"] = sum(value < 0 for value in values)
        stats["ties"] = sum(value == 0 for value in values)
        stats["losses_lower_is_better"] = sum(value > 0 for value in values)
        paired_metrics[name] = stats

    success_stats = paired_bootstrap(differences, bootstrap, seed)
    success_stats.update(
        {
            "both_success": len(both_success_ids),
            "baseline_only": baseline_only,
            "candidate_only": candidate_only,
            "both_fail": sum(
                not baseline_flag and not candidate_flag
                for baseline_flag, candidate_flag in zip(
                    baseline_success, candidate_success
                )
            ),
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                baseline_only, candidate_only
            ),
        }
    )
    return {
        "success_rate_candidate_minus_baseline": success_stats,
        "both_success_paired_metrics_candidate_minus_baseline": paired_metrics,
        "per_target": [
            {
                "target_id": target_id,
                "baseline_success": baseline_success[target_id],
                "candidate_success": candidate_success[target_id],
                "baseline_iteration": int(baseline["iter"][target_id]),
                "candidate_iteration": int(candidate["iter"][target_id]),
                "baseline_route_cost": baseline["route_costs"][target_id],
                "candidate_route_cost": candidate["route_costs"][target_id],
                "baseline_route_length": baseline["route_lens"][target_id],
                "candidate_route_length": candidate["route_lens"][target_id],
            }
            for target_id in range(len(baseline_success))
        ],
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    retro_star_root = repo_root / "retro_star"
    if str(retro_star_root) not in sys.path:
        sys.path.insert(0, str(retro_star_root))

    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    baseline = load_plan(baseline_path)
    candidate = load_plan(candidate_path)
    contract = compare_contracts(baseline, candidate)
    report = {
        "schema_version": "1.0.0",
        "as_of": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol": {
            "bootstrap_samples": args.bootstrap,
            "bootstrap_seed": args.seed,
            "interpretation_boundary": (
                "Matched-budget Pistachio-hard comparison; only sampler and "
                "its declared TP_FREE configuration may differ."
            ),
        },
        "inputs": {
            "baseline": {
                "path": str(baseline_path),
                "sha256": sha256(baseline_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256(candidate_path),
            },
        },
        "contract": contract,
        "baseline": summarize_plan(baseline),
        "candidate": summarize_plan(candidate),
        "comparison": compare_plans(
            baseline, candidate, args.bootstrap, args.seed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "baseline": report["baseline"],
        "candidate": report["candidate"],
        "success_comparison": report["comparison"][
            "success_rate_candidate_minus_baseline"
        ],
    }, indent=2, sort_keys=True))
    print("output=%s" % args.output)
    print("sha256=%s" % sha256(args.output))


if __name__ == "__main__":
    main()
