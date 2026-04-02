#!/usr/bin/env python3
"""Run Retro* serial vs multi-pool benchmarks with test, collection, and comparison.

Usage (from repo root):
  python scripts/benchmark_compare.py \
    --python python \
    --gpu 0 \
    --seed 42 \
    --iterations 201 \
    --expansion-topk 8 \
    --one-step-type template_based \
    --test-routes uspto190 \
    --parallel-nums 4,8,12

Template-free (CSS/DICT) example:
  python scripts/benchmark_compare.py \
    --python python \
    --one-step-type template_free \
    --test-routes uspto190 \
    --parallel-nums 5,8 \
    --extra-args "--CSS --RD_list '[(7,2),(3,0)]' --DICT"
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunRecord:
    case: str
    mode: str
    parallel_num: Optional[int]
    repeat: int
    return_code: int
    wall_time_sec: float
    planner_time_sec: Optional[float]
    num_targets: int
    succ_count: int
    succ_rate: float
    avg_iter: Optional[float]
    avg_route_len: Optional[float]
    avg_route_cost: Optional[float]
    avg_final_nodes: Optional[float]
    result_folder: str
    plan_pkl: Optional[str]
    log_file: str
    command: str


ROUTE_FILE_MAP = {
    "uspto190": "dataset/routes_possible_test_hard.pkl",
    "pth_hard": "dataset/pistachio_hard_targets.txt",
    "pth_reach": "dataset/pistachio_reachable_targets.txt",
    "8XIK_olorofim": "dataset/8XIK_olorofim.txt",
    "8XIK_NCI": "/home/chenqixuan/retro_star/retro_star/dataset/8XIK_NVI_PAI.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retro* benchmark compare runner")
    parser.add_argument("--python", default="python", help="Python executable in target environment")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=201)
    parser.add_argument("--expansion-topk", type=int, default=8)
    parser.add_argument("--one-step-type", default="template_based", choices=["template_based", "template_free"])
    parser.add_argument("--test-routes", default="uspto190")
    parser.add_argument("--parallel-nums", default="4,8,12", help="Comma-separated pool widths")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--use-value-fn", action="store_true", default=True)
    parser.add_argument("--no-use-value-fn", action="store_false", dest="use_value_fn")
    parser.add_argument("--viz", action="store_true", help="Enable viz outputs during benchmark")
    parser.add_argument("--starting-molecules", default="dataset/origin_dict.csv")
    parser.add_argument("--extra-args", default="", help="Extra args appended to retro_plan.py")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--strict-preflight", action="store_true")
    parser.add_argument("--strict-gpu", action="store_true", help="Fail if requested GPU is unavailable instead of auto-fallback to CPU")
    parser.add_argument("--allow-fail-runs", action="store_true", help="Do not fail process when a benchmark case returns non-zero")
    parser.add_argument("--report-root", default="benchmark_reports", help="Relative to repo root")
    return parser.parse_args()


def parse_parallel_nums(value: str) -> List[int]:
    nums = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        nums.append(int(piece))
    unique = sorted(set(nums))
    if not unique:
        raise ValueError("parallel-nums cannot be empty")
    return unique


def ensure_import_path(repo_root: Path) -> None:
    retro_dir = repo_root / "retro_star"
    if str(retro_dir) not in sys.path:
        sys.path.insert(0, str(retro_dir))


def run_cmd(cmd: List[str], cwd: Path, log_file: Path) -> Tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed


def tail_text(file_path: Path, lines: int = 40) -> str:
    if not file_path.exists():
        return ""
    text = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(text[-lines:])


def detect_cuda_available(python_bin: str, cwd: Path) -> Optional[bool]:
    cmd = [python_bin, "-c", "import torch; print(1 if torch.cuda.is_available() else 0)"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if out not in {"0", "1"}:
        return None
    return out == "1"


def maybe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def collect_plan_metrics(repo_root: Path, plan_pkl: Path) -> Dict[str, Optional[float]]:
    ensure_import_path(repo_root)

    if not plan_pkl.exists():
        return {
            "planner_time_sec": None,
            "num_targets": 0,
            "succ_count": 0,
            "succ_rate": 0.0,
            "avg_iter": None,
            "avg_route_len": None,
            "avg_route_cost": None,
            "avg_final_nodes": None,
        }

    with open(plan_pkl, "rb") as f:
        result = pickle.load(f)

    succ_flags = result.get("succ", []) or []
    num_targets = len(succ_flags)
    succ_count = int(sum(1 for x in succ_flags if bool(x)))
    succ_rate = float(succ_count / num_targets) if num_targets > 0 else 0.0

    iters = [float(x) for x in (result.get("iter", []) or []) if x is not None]
    route_lens = [float(x) for x in (result.get("route_lens", []) or []) if x is not None]
    route_costs = [float(x) for x in (result.get("route_costs", []) or []) if x is not None]
    final_nodes = [float(x) for x in (result.get("final_node", []) or []) if x is not None]
    cum_times = [float(x) for x in (result.get("cumulated_time", []) or []) if x is not None]

    return {
        "planner_time_sec": (max(cum_times) if cum_times else None),
        "num_targets": num_targets,
        "succ_count": succ_count,
        "succ_rate": succ_rate,
        "avg_iter": maybe_mean(iters),
        "avg_route_len": maybe_mean(route_lens),
        "avg_route_cost": maybe_mean(route_costs),
        "avg_final_nodes": maybe_mean(final_nodes),
    }


def check_preflight(args: argparse.Namespace, repo_root: Path) -> List[str]:
    retro_dir = repo_root / "retro_star"
    missing = []

    route_path = ROUTE_FILE_MAP.get(args.test_routes)
    if route_path:
        p = (retro_dir / route_path) if not os.path.isabs(route_path) else Path(route_path)
        if not p.exists():
            missing.append(str(p))

    start_path = retro_dir / args.starting_molecules
    if not start_path.exists():
        missing.append(str(start_path))

    if args.one_step_type == "template_based":
        for rel in ["one_step_model/template_rules_1.dat", "one_step_model/saved_rollout_state_1_2048.ckpt"]:
            p = retro_dir / rel
            if not p.exists():
                missing.append(str(p))
    else:
        for rel in ["one_step_model/USPTO_full_PtoR.pt", "one_step_model/USPTO-MIT_RtoP_mixed.pt"]:
            p = retro_dir / rel
            if not p.exists():
                missing.append(str(p))

    if args.use_value_fn:
        p = retro_dir / "saved_models/best_epoch_final_4.pt"
        if not p.exists():
            missing.append(str(p))

    return missing


def run_pre_tests(args: argparse.Namespace, repo_root: Path, report_dir: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if args.skip_tests:
        out["scheduler_smoke"] = {"status": "skipped", "detail": "--skip-tests"}
        return out

    tests = [
        {
            "name": "scheduler_smoke",
            "cmd": [args.python, "tests/scheduler_smoke_test.py"],
            "cwd": repo_root,
        },
        {
            "name": "tp_free_batch_smoke",
            "cmd": [args.python, "tests/tp_free_batch_smoke_test.py"],
            "cwd": repo_root,
        },
        {
            "name": "py_compile",
            "cmd": [
                args.python,
                "-m",
                "py_compile",
                "retro_star/retro_plan.py",
                "retro_star/common/parse_args.py",
                "retro_star/alg/molstar_parallel.py",
                "retro_star/alg/molstar_task.py",
                "retro_star/packages/mlp_retrosyn/mlp_retrosyn/tp_free_inference.py",
                "retro_star/packages/mlp_retrosyn/mlp_retrosyn/tp_free_tools.py",
                "scripts/benchmark_compare.py",
            ],
            "cwd": repo_root,
        },
    ]

    for test in tests:
        log_file = report_dir / "logs" / f"test_{test['name']}.log"
        rc, elapsed = run_cmd(test["cmd"], test["cwd"], log_file)
        if rc == 0:
            out[test["name"]] = {
                "status": "pass",
                "detail": f"elapsed={elapsed:.2f}s",
                "log": str(log_file),
            }
        else:
            out[test["name"]] = {
                "status": "fail",
                "detail": f"elapsed={elapsed:.2f}s",
                "log": str(log_file),
                "tail": tail_text(log_file, 40),
            }
    return out


def build_base_plan_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        args.python,
        "retro_plan.py",
        "--seed",
        str(args.seed),
        "--gpu",
        str(args.gpu),
        "--expansion_topk",
        str(args.expansion_topk),
        "--iterations",
        str(args.iterations),
        "--one_step_type",
        args.one_step_type,
        "--test_routes",
        args.test_routes,
        "--starting_molecules",
        args.starting_molecules,
    ]
    if args.use_value_fn:
        cmd.append("--use_value_fn")
    if args.viz:
        cmd.append("--viz")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def run_benchmarks(args: argparse.Namespace, repo_root: Path, report_dir: Path) -> List[RunRecord]:
    retro_dir = repo_root / "retro_star"
    bench_run_id = report_dir.name
    results: List[RunRecord] = []

    parallel_nums = parse_parallel_nums(args.parallel_nums)
    cases = [("serial", None)] + [(f"parallel_{n}", n) for n in parallel_nums]

    for repeat in range(1, args.repeats + 1):
        for case_name, parallel_num in cases:
            mode = "serial" if parallel_num is None else "parallel"
            cmd = build_base_plan_cmd(args)
            result_folder_rel = f"results/benchmarks/{bench_run_id}/{case_name}/repeat_{repeat}"
            viz_dir_rel = f"{result_folder_rel}/viz"

            if parallel_num is not None:
                cmd.extend(["--multi_pool", "--parallel_num", str(parallel_num)])

            cmd.extend(["--result_folder", result_folder_rel, "--viz_dir", viz_dir_rel])

            log_file = report_dir / "logs" / f"run_{case_name}_repeat_{repeat}.log"
            rc, elapsed = run_cmd(cmd, retro_dir, log_file)

            plan_pkl = retro_dir / result_folder_rel / "plan.pkl"
            metrics = collect_plan_metrics(repo_root, plan_pkl)

            record = RunRecord(
                case=case_name,
                mode=mode,
                parallel_num=parallel_num,
                repeat=repeat,
                return_code=rc,
                wall_time_sec=float(elapsed),
                planner_time_sec=metrics["planner_time_sec"],
                num_targets=int(metrics["num_targets"]),
                succ_count=int(metrics["succ_count"]),
                succ_rate=float(metrics["succ_rate"]),
                avg_iter=metrics["avg_iter"],
                avg_route_len=metrics["avg_route_len"],
                avg_route_cost=metrics["avg_route_cost"],
                avg_final_nodes=metrics["avg_final_nodes"],
                result_folder=result_folder_rel,
                plan_pkl=(str(plan_pkl) if plan_pkl.exists() else None),
                log_file=str(log_file),
                command=" ".join(shlex.quote(x) for x in cmd),
            )
            results.append(record)

            print(
                f"[run] case={case_name} repeat={repeat} rc={rc} "
                f"wall={elapsed:.2f}s succ={record.succ_count}/{record.num_targets}"
            )
            if rc != 0:
                print("[run] tail of failed log:\n" + tail_text(log_file, 30))

    return results


def aggregate_records(records: List[RunRecord]) -> Dict[str, Dict[str, Optional[float]]]:
    grouped: Dict[str, List[RunRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.case, []).append(rec)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for case, recs in grouped.items():
        wall = [r.wall_time_sec for r in recs]
        planner = [r.planner_time_sec for r in recs if r.planner_time_sec is not None]
        succ_count = [r.succ_count for r in recs]
        succ_rate = [r.succ_rate for r in recs]
        avg_iter = [r.avg_iter for r in recs if r.avg_iter is not None]
        avg_final_nodes = [r.avg_final_nodes for r in recs if r.avg_final_nodes is not None]
        num_targets = [r.num_targets for r in recs]
        rc_ok = sum(1 for r in recs if r.return_code == 0)

        out[case] = {
            "runs": len(recs),
            "run_success_count": rc_ok,
            "num_targets": int(statistics.median(num_targets)) if num_targets else 0,
            "wall_time_sec_mean": maybe_mean(wall),
            "wall_time_sec_min": float(min(wall)) if wall else None,
            "planner_time_sec_mean": maybe_mean(planner),
            "succ_count_mean": maybe_mean([float(x) for x in succ_count]),
            "succ_rate_mean": maybe_mean(succ_rate),
            "avg_iter_mean": maybe_mean(avg_iter),
            "avg_final_nodes_mean": maybe_mean(avg_final_nodes),
        }

    serial = out.get("serial")
    if serial and serial.get("wall_time_sec_mean"):
        base = float(serial["wall_time_sec_mean"])
        for case, stats in out.items():
            if stats.get("wall_time_sec_mean"):
                stats["speedup_vs_serial"] = base / float(stats["wall_time_sec_mean"])
            else:
                stats["speedup_vs_serial"] = None
            if stats.get("succ_count_mean") is not None and serial.get("succ_count_mean") is not None:
                stats["succ_count_delta_vs_serial"] = float(stats["succ_count_mean"]) - float(serial["succ_count_mean"])
            else:
                stats["succ_count_delta_vs_serial"] = None

    return out


def write_reports(
    args: argparse.Namespace,
    report_dir: Path,
    pre_tests: Dict[str, Dict[str, str]],
    records: List[RunRecord],
    summary: Dict[str, Dict[str, Optional[float]]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_json = {
        "args": vars(args),
        "pre_tests": pre_tests,
        "records": [asdict(r) for r in records],
        "summary": summary,
    }

    raw_json_path = report_dir / "benchmark_results.json"
    raw_json_path.write_text(json.dumps(raw_json, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = []
    md_lines.append("# Benchmark Comparison Report")
    md_lines.append("")
    md_lines.append("## Run Config")
    md_lines.append("")
    md_lines.append("```text")
    for k, v in vars(args).items():
        md_lines.append(f"{k}: {v}")
    md_lines.append("```")
    md_lines.append("")

    md_lines.append("## Pre-tests")
    md_lines.append("")
    for name, info in pre_tests.items():
        md_lines.append(f"- `{name}`: {info.get('status')} ({info.get('detail')})")
    md_lines.append("")

    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append("| Case | Runs | RC Pass | Wall Mean (s) | Planner Mean (s) | Speedup vs Serial | Succ Mean | Succ Rate Mean | Avg Iter | Avg Final Nodes |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for case in sorted(summary.keys(), key=lambda x: (x != "serial", x)):
        s = summary[case]
        md_lines.append(
            "| {case} | {runs} | {rc_ok} | {wall} | {planner} | {speedup} | {succ} | {succ_rate} | {avg_iter} | {avg_nodes} |".format(
                case=case,
                runs=s.get("runs"),
                rc_ok=s.get("run_success_count"),
                wall=f"{s.get('wall_time_sec_mean', 0):.3f}" if s.get("wall_time_sec_mean") is not None else "NA",
                planner=f"{s.get('planner_time_sec_mean', 0):.3f}" if s.get("planner_time_sec_mean") is not None else "NA",
                speedup=f"{s.get('speedup_vs_serial', 0):.3f}x" if s.get("speedup_vs_serial") is not None else "NA",
                succ=f"{s.get('succ_count_mean', 0):.3f}" if s.get("succ_count_mean") is not None else "NA",
                succ_rate=f"{s.get('succ_rate_mean', 0):.4f}" if s.get("succ_rate_mean") is not None else "NA",
                avg_iter=f"{s.get('avg_iter_mean', 0):.3f}" if s.get("avg_iter_mean") is not None else "NA",
                avg_nodes=f"{s.get('avg_final_nodes_mean', 0):.3f}" if s.get("avg_final_nodes_mean") is not None else "NA",
            )
        )

    md_lines.append("")
    md_lines.append("## Per-run Logs")
    md_lines.append("")
    for rec in records:
        md_lines.append(
            f"- `{rec.case}` repeat={rec.repeat}, rc={rec.return_code}, wall={rec.wall_time_sec:.3f}s, log=`{rec.log_file}`"
        )

    md_path = report_dir / "benchmark_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.gpu >= 0:
        cuda_ok = detect_cuda_available(args.python, repo_root)
        if cuda_ok is False:
            if args.strict_gpu:
                print(f"[gpu] requested gpu={args.gpu}, but CUDA is unavailable in {args.python}.")
                return 5
            print(f"[gpu] requested gpu={args.gpu}, but CUDA is unavailable in {args.python}; fallback to cpu (--gpu -1).")
            args.gpu = -1
        elif cuda_ok is None:
            print("[gpu] unable to detect CUDA availability; keep user-provided --gpu as-is.")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    report_dir = repo_root / args.report_root / timestamp

    missing = check_preflight(args, repo_root)
    if missing:
        print("[preflight] Missing required files:")
        for p in missing:
            print(f"  - {p}")
        if args.strict_preflight:
            print("[preflight] strict mode enabled; abort.")
            return 2
        print("[preflight] continue anyway (strict mode disabled).")

    pre_tests = run_pre_tests(args, repo_root, report_dir)
    for name, info in pre_tests.items():
        print(f"[test] {name}: {info.get('status')} ({info.get('detail')})")

    if any(info.get("status") == "fail" for info in pre_tests.values()):
        print("[test] one or more pre-tests failed; aborting benchmark.")
        write_reports(args, report_dir, pre_tests, [], {})
        return 3

    records = run_benchmarks(args, repo_root, report_dir)
    summary = aggregate_records(records)
    write_reports(args, report_dir, pre_tests, records, summary)

    print("[done] report directory:", report_dir)
    print("[done] summary file:", report_dir / "benchmark_report.md")
    print("[done] raw json:", report_dir / "benchmark_results.json")

    # Best case hint
    best_case = None
    best_time = None
    for case, s in summary.items():
        t = s.get("wall_time_sec_mean")
        if t is None:
            continue
        if best_time is None or t < best_time:
            best_time = float(t)
            best_case = case
    if best_case is not None:
        print(f"[done] fastest case: {best_case} ({best_time:.3f}s)")

    failed_runs = [r for r in records if r.return_code != 0]
    if failed_runs and not args.allow_fail_runs:
        print(f"[done] benchmark has {len(failed_runs)} failed run(s); use --allow-fail-runs to ignore.")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
