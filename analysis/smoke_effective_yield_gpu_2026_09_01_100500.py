"""Run a bounded GPU smoke for the EffectiveReactionYield@8 pipeline."""

import argparse
import json
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mlp_retrosyn.tp_free_inference import TP_free_Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--retro-model", required=True)
    parser.add_argument("--forward-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_targets(path, limit):
    with path.open("rb") as handle:
        entries = pickle.load(handle)
    targets = []
    for entry in entries[:limit]:
        reaction = entry[0] if isinstance(entry, (list, tuple)) else entry
        targets.append(reaction.split(">>", 1)[0])
    return targets


def read_telemetry(path):
    rows = []
    if path.exists():
        with path.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    return rows


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    telemetry_pattern = output_dir / "fragment_yield_{pid}.jsonl"
    os.environ["TP_FREE_CSS_SAMPLER"] = "yield8"
    os.environ["TP_FREE_FRAGMENT_YIELD_LOG"] = str(telemetry_pattern)
    os.environ["TP_FREE_YIELD_MAX_TRIPLES"] = "256"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = TP_free_Model(
        str(Path(args.retro_model).resolve()),
        3,
        str(Path(args.forward_model).resolve()),
        1,
        CCS=True,
        RD_list=[(3, 0)],
        DICT=False,
        device=0,
    )
    targets = load_targets(Path(args.routes).resolve(), args.targets)
    results = []
    for target_id, target in enumerate(targets):
        started = time.perf_counter()
        output = model.run(target, topk=args.topk, task_id=target_id)
        elapsed = time.perf_counter() - started
        results.append({
            "target_id": target_id,
            "target": target,
            "elapsed_seconds": elapsed,
            "output_is_none": output is None,
            "full_product_unique_reaction_count": (
                len(output["reactants"]) if output is not None else 0
            ),
            "reaction_source_fragment_counts": (
                [len(sources) for sources in output.get("fragment_sources", [])]
                if output is not None else []
            ),
        })

    telemetry_path = Path(str(telemetry_pattern).format(pid=os.getpid()))
    telemetry = read_telemetry(telemetry_path)
    telemetry_by_target = {}
    for target_id in range(len(targets)):
        rows = [row for row in telemetry if row["task_id"] == target_id]
        telemetry_by_target[str(target_id)] = {
            "fragment_rows": len(rows),
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

    now = datetime.now().astimezone()
    report = {
        "as_of": now.isoformat(),
        "analysis_version": "1.0.0",
        "protocol": {
            "sampler": "yield8",
            "topk": args.topk,
            "retro_topk": 3,
            "forward_topk": 1,
            "dict": False,
            "max_triples": 256,
            "seed": args.seed,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_device": torch.cuda.get_device_name(0),
        },
        "results": results,
        "telemetry_by_target": telemetry_by_target,
        "telemetry_file": str(telemetry_path),
    }
    report_path = output_dir / "effective_yield_gpu_smoke_summary_2026_09_01_100500.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"report": str(report_path), "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
