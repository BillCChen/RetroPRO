"""Compare strict-eight CSS samplers with a shared frozen one-step model."""

import argparse
import ast
import hashlib
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from mlp_retrosyn.tp_free_inference import TP_free_Model


ARMS = {
    "random_prod": {"sampler": "random", "rd_list": [(7, 0), (3, 0)]},
    "random_r3": {"sampler": "random", "rd_list": [(3, 0)]},
    "paircov": {"sampler": "paircov", "rd_list": [(3, 0)]},
    "triplecov": {"sampler": "triplecov", "rd_list": [(3, 0)]},
    "yield8_legacy": {
        "sampler": "yield8",
        "rd_list": [(3, 0)],
        "profile": "legacy",
    },
    "yield8": {
        "sampler": "yield8",
        "rd_list": [(3, 0)],
        "profile": "balanced",
    },
    "yield8_hybrid": {
        "sampler": "yield8_hybrid",
        "rd_list": [(3, 0)],
        "profile": "balanced",
        "guardrail_slots": 4,
    },
    "yield8_hybrid6": {
        "sampler": "yield8_hybrid",
        "rd_list": [(3, 0)],
        "profile": "balanced",
        "guardrail_slots": 6,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--retro-model", required=True)
    parser.add_argument("--forward-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dict-snapshot", default="")
    parser.add_argument("--arms", default=",".join(ARMS))
    return parser.parse_args()


def load_targets(path, limit):
    targets = []
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            entries = pickle.load(handle)
        for entry in entries:
            reaction = entry[0] if isinstance(entry, (list, tuple)) else entry
            targets.append(reaction.split(">>", 1)[0])
    else:
        with path.open() as handle:
            for raw in handle:
                value = raw.strip()
                if not value:
                    continue
                parsed = ast.literal_eval(value)
                target = parsed[0] if isinstance(parsed, (list, tuple)) else parsed
                targets.append(target.split(">", 1)[0])
    if limit > 0:
        targets = targets[:limit]
    return targets


def load_telemetry(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_telemetry(rows):
    return {
        "selected_fragment_count": (
            rows[0]["selected_fragment_count"] if rows else 0
        ),
        "augmentation_count": sum(row["augmentation_count"] for row in rows),
        "retro_raw_count": sum(row["retro_raw_count"] for row in rows),
        "retro_valid_count": sum(row["retro_valid_count"] for row in rows),
        "forward_consistent_count": sum(
            row["forward_consistent_count"] for row in rows
        ),
        "mapped_count": sum(row["mapped_count"] for row in rows),
        "template_extracted_count": sum(
            row["template_extracted_count"] for row in rows
        ),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    os.environ["TP_FREE_CSS_STRICT_TOPK"] = "1"
    os.environ["TP_FREE_YIELD_MAX_TRIPLES"] = "256"
    targets = load_targets(Path(args.routes).resolve(), args.targets)
    invalid = [target for target in targets if Chem.MolFromSmiles(target) is None]
    if invalid:
        raise ValueError("unparseable targets: %d" % len(invalid))
    output_dir.mkdir(parents=True, exist_ok=False)
    dict_snapshot = Path(args.dict_snapshot).resolve() if args.dict_snapshot else None
    model = TP_free_Model(
        str(Path(args.retro_model).resolve()),
        3,
        str(Path(args.forward_model).resolve()),
        1,
        CCS=True,
        RD_list=[(3, 0)],
        DICT=dict_snapshot is not None,
        device=0,
    )
    if dict_snapshot is not None:
        with dict_snapshot.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("rules"), dict):
            payload = payload["rules"]
        if not isinstance(payload, dict):
            raise TypeError("DICT snapshot must contain a dictionary")
        model._dict_ref = payload
        model.DICT = payload
        model.renew_DICT = lambda *_args, **_kwargs: 0

    selected_arm_names = [value for value in args.arms.split(",") if value]
    unknown_arms = sorted(set(selected_arm_names) - set(ARMS))
    if unknown_arms:
        raise ValueError("unknown arms: %s" % ",".join(unknown_arms))
    selected_arms = {name: ARMS[name] for name in selected_arm_names}
    results = []
    telemetry_paths = {}
    for arm, config in selected_arms.items():
        os.environ["TP_FREE_CSS_SAMPLER"] = config["sampler"]
        os.environ["TP_FREE_YIELD_EXPLORATION_PROFILE"] = config.get(
            "profile", "balanced"
        )
        os.environ["TP_FREE_YIELD_GUARDRAIL_SLOTS"] = str(
            config.get("guardrail_slots", 4)
        )
        telemetry_pattern = output_dir / (arm + "_fragment_yield_{pid}.jsonl")
        os.environ["TP_FREE_FRAGMENT_YIELD_LOG"] = str(telemetry_pattern)
        telemetry_paths[arm] = Path(str(telemetry_pattern).format(pid=os.getpid()))
        model.RD_list = config["rd_list"]
        for target_id, target in enumerate(targets):
            task_seed = args.seed + target_id
            random.seed(task_seed)
            np.random.seed(task_seed)
            torch.manual_seed(task_seed)
            torch.cuda.manual_seed_all(task_seed)
            started = time.perf_counter()
            output = model.run(target, topk=args.topk, task_id=target_id)
            results.append({
                "arm": arm,
                "target_id": target_id,
                "target": target,
                "elapsed_seconds": time.perf_counter() - started,
                "output_is_none": output is None,
                "full_product_unique_reaction_count": (
                    len(output["reactants"]) if output is not None else 0
                ),
            })

    telemetry = {}
    for arm, path in telemetry_paths.items():
        rows = load_telemetry(path)
        telemetry[arm] = {
            str(target_id): aggregate_telemetry(
                [row for row in rows if row["task_id"] == target_id]
            )
            for target_id in range(len(targets))
        }

    by_arm = {}
    for arm in selected_arms:
        arm_rows = [row for row in results if row["arm"] == arm]
        arm_telemetry = list(telemetry[arm].values())
        reaction_counts = [
            row["full_product_unique_reaction_count"] for row in arm_rows
        ]
        by_arm[arm] = {
            "target_count": len(arm_rows),
            "reaction_count_sum": sum(reaction_counts),
            "reaction_count_mean": sum(reaction_counts) / len(reaction_counts),
            "elapsed_seconds_sum": sum(row["elapsed_seconds"] for row in arm_rows),
            "selected_fragment_count_sum": sum(
                row["selected_fragment_count"] for row in arm_telemetry
            ),
            "augmentation_count_sum": sum(
                row["augmentation_count"] for row in arm_telemetry
            ),
            "forward_consistent_count_sum": sum(
                row["forward_consistent_count"] for row in arm_telemetry
            ),
            "template_extracted_count_sum": sum(
                row["template_extracted_count"] for row in arm_telemetry
            ),
        }

    now = datetime.now().astimezone()
    report = {
        "as_of": now.isoformat(),
        "analysis_version": "1.0.0",
        "protocol": {
            "arms": selected_arms,
            "topk": args.topk,
            "strict_topk": True,
            "retro_topk": 3,
            "forward_topk": 1,
            "dict": dict_snapshot is not None,
            "dict_snapshot": str(dict_snapshot) if dict_snapshot else None,
            "dict_snapshot_sha256": (
                sha256_file(dict_snapshot) if dict_snapshot else None
            ),
            "dict_updates": "frozen" if dict_snapshot else "disabled",
            "effective_cache": os.getenv("TP_FREE_EFFECTIVE_CACHE", "0") == "1",
            "max_triples": 256,
            "seed": args.seed,
            "python_hash_seed": os.getenv("PYTHONHASHSEED"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_device": torch.cuda.get_device_name(0),
        },
        "by_arm": by_arm,
        "per_arm_target": results,
        "telemetry": telemetry,
        "interpretation_boundary": (
            "%d-target GPU one-step comparison with %s DICT; verifies "
            "effective-reaction yield and accounting, not route-search success."
            % (
                len(targets),
                "a frozen cross-dataset" if dict_snapshot else "an empty",
            )
        ),
    }
    stamp = now.strftime("%Y_%m_%d_%H%M%S")
    report_path = output_dir / ("effective_yield_gpu_comparison_%s.json" % stamp)
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"report": str(report_path), "by_arm": by_arm}, sort_keys=True))


if __name__ == "__main__":
    main()
