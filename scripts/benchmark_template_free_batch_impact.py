#!/usr/bin/env python3
"""Benchmark template_free batch impact: serial vs parallel(no-batch) vs parallel(batch)."""

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


ROUTE_FILE_MAP = {
    "uspto190": "dataset/routes_possible_test_hard.pkl",
    "pth_hard": "dataset/pistachio_hard_targets.txt",
    "pth_reach": "dataset/pistachio_reachable_targets.txt",
    "8XIK_olorofim": "dataset/8XIK_olorofim.txt",
    "8XIK_NCI": "/home/chenqixuan/retro_star/retro_star/dataset/8XIK_NVI_PAI.txt",
}


@dataclass
class RunRecord:
    case: str
    repeat: int
    return_code: int
    wall_time_sec: float
    planner_time_sec: Optional[float]
    num_targets: int
    succ_count: int
    succ_rate: float
    avg_iter: Optional[float]
    avg_final_nodes: Optional[float]
    result_folder: str
    plan_pkl: Optional[str]
    log_file: str
    command: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Template-free batch impact benchmark")
    p.add_argument("--python", default="python")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iterations", type=int, default=201)
    p.add_argument("--expansion-topk", type=int, default=8)
    p.add_argument("--test-routes", default="uspto190")
    p.add_argument("--starting-molecules", default="dataset/origin_dict.csv")
    p.add_argument("--parallel-num", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--rd-list", default="[(7,2),(3,0)]")
    p.add_argument("--use-css", action="store_true", default=True)
    p.add_argument("--no-use-css", action="store_false", dest="use_css")
    p.add_argument("--use-dict", action="store_true", default=True)
    p.add_argument("--no-use-dict", action="store_false", dest="use_dict")
    p.add_argument("--use-value-fn", action="store_true", default=True)
    p.add_argument("--no-use-value-fn", action="store_false", dest="use_value_fn")
    p.add_argument("--viz", action="store_true")
    p.add_argument("--extra-args", default="")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--strict-preflight", action="store_true")
    p.add_argument("--allow-fail-runs", action="store_true")
    p.add_argument("--report-root", default="benchmark_reports/template_free_batch")
    return p.parse_args()


def run_cmd(cmd: List[str], cwd: Path, log_file: Path, env_patch: Optional[Dict[str, str]] = None) -> Tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_patch:
        env.update(env_patch)
    start = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, time.perf_counter() - start


def tail_text(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def maybe_mean(xs: List[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def ensure_import_path(repo_root: Path) -> None:
    retro_dir = repo_root / "retro_star"
    if str(retro_dir) not in sys.path:
        sys.path.insert(0, str(retro_dir))


def collect_plan_metrics(repo_root: Path, plan_pkl: Path) -> Dict[str, Optional[float]]:
    ensure_import_path(repo_root)
    if not plan_pkl.exists():
        return {
            "planner_time_sec": None,
            "num_targets": 0,
            "succ_count": 0,
            "succ_rate": 0.0,
            "avg_iter": None,
            "avg_final_nodes": None,
        }

    with open(plan_pkl, "rb") as f:
        result = pickle.load(f)

    succ_flags = result.get("succ", []) or []
    num_targets = len(succ_flags)
    succ_count = int(sum(1 for x in succ_flags if bool(x)))
    succ_rate = float(succ_count / num_targets) if num_targets > 0 else 0.0

    iters = [float(x) for x in (result.get("iter", []) or []) if x is not None]
    final_nodes = [float(x) for x in (result.get("final_node", []) or []) if x is not None]
    cum_times = [float(x) for x in (result.get("cumulated_time", []) or []) if x is not None]

    return {
        "planner_time_sec": (max(cum_times) if cum_times else None),
        "num_targets": num_targets,
        "succ_count": succ_count,
        "succ_rate": succ_rate,
        "avg_iter": maybe_mean(iters),
        "avg_final_nodes": maybe_mean(final_nodes),
    }


def detect_cuda_available(python_bin: str, cwd: Path) -> Optional[bool]:
    cmd = [python_bin, "-c", "import torch; print(1 if torch.cuda.is_available() else 0)"]
    try:
        p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        return None
    if p.returncode != 0:
        return None
    out = (p.stdout or "").strip()
    if out not in {"0", "1"}:
        return None
    return out == "1"


def check_preflight(args: argparse.Namespace, repo_root: Path) -> List[str]:
    retro_dir = repo_root / "retro_star"
    miss: List[str] = []

    route_path = ROUTE_FILE_MAP.get(args.test_routes)
    if route_path:
        rp = (retro_dir / route_path) if not os.path.isabs(route_path) else Path(route_path)
        if not rp.exists():
            miss.append(str(rp))

    for rel in [
        args.starting_molecules,
        "one_step_model/USPTO_full_PtoR.pt",
        "one_step_model/USPTO-MIT_RtoP_mixed.pt",
    ]:
        p = retro_dir / rel
        if not p.exists():
            miss.append(str(p))

    if args.use_value_fn:
        p = retro_dir / "saved_models/best_epoch_final_4.pt"
        if not p.exists():
            miss.append(str(p))

    return miss


def run_pre_tests(args: argparse.Namespace, repo_root: Path, report_dir: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if args.skip_tests:
        out["tp_free_batch_smoke"] = {"status": "skipped", "detail": "--skip-tests"}
        return out

    tests = [
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
                "retro_star/common/prepare_utils.py",
                "retro_star/alg/molstar_parallel.py",
                "retro_star/packages/mlp_retrosyn/mlp_retrosyn/tp_free_inference.py",
                "scripts/benchmark_template_free_batch_impact.py",
            ],
            "cwd": repo_root,
        },
    ]

    for t in tests:
        logf = report_dir / "logs" / f"test_{t['name']}.log"
        rc, elapsed = run_cmd(t["cmd"], t["cwd"], logf)
        if rc == 0:
            out[t["name"]] = {"status": "pass", "detail": f"elapsed={elapsed:.2f}s", "log": str(logf)}
        else:
            out[t["name"]] = {
                "status": "fail",
                "detail": f"elapsed={elapsed:.2f}s",
                "log": str(logf),
                "tail": tail_text(logf, 40),
            }
    return out


def build_base_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        args.python,
        "retro_plan.py",
        "--seed", str(args.seed),
        "--gpu", str(args.gpu),
        "--one_step_type", "template_free",
        "--expansion_topk", str(args.expansion_topk),
        "--iterations", str(args.iterations),
        "--test_routes", args.test_routes,
        "--starting_molecules", args.starting_molecules,
        "--RD_list", args.rd_list,
    ]
    if args.use_css:
        cmd.append("--CSS")
    if args.use_dict:
        cmd.append("--DICT")
    if args.use_value_fn:
        cmd.append("--use_value_fn")
    if args.viz:
        cmd.append("--viz")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def run_bench(args: argparse.Namespace, repo_root: Path, report_dir: Path) -> List[RunRecord]:
    retro_dir = repo_root / "retro_star"
    run_id = report_dir.name
    results: List[RunRecord] = []

    cases = [
        ("serial", None, None),
        ("parallel_no_batch", args.parallel_num, {"RETROPRO_DISABLE_RUN_BATCH": "1"}),
        ("parallel_batch", args.parallel_num, {"RETROPRO_DISABLE_RUN_BATCH": "0"}),
    ]

    for rep in range(1, args.repeats + 1):
        for case_name, pool_size, env_patch in cases:
            cmd = build_base_cmd(args)
            result_folder_rel = f"results/benchmarks_template_free_batch/{run_id}/{case_name}/repeat_{rep}"
            viz_dir_rel = f"{result_folder_rel}/viz"

            if pool_size is not None:
                cmd.extend(["--multi_pool", "--parallel_num", str(pool_size)])
            cmd.extend(["--result_folder", result_folder_rel, "--viz_dir", viz_dir_rel])

            logf = report_dir / "logs" / f"run_{case_name}_repeat_{rep}.log"
            rc, wall = run_cmd(cmd, retro_dir, logf, env_patch=env_patch)

            plan_pkl = retro_dir / result_folder_rel / "plan.pkl"
            m = collect_plan_metrics(repo_root, plan_pkl)
            rec = RunRecord(
                case=case_name,
                repeat=rep,
                return_code=rc,
                wall_time_sec=float(wall),
                planner_time_sec=m["planner_time_sec"],
                num_targets=int(m["num_targets"]),
                succ_count=int(m["succ_count"]),
                succ_rate=float(m["succ_rate"]),
                avg_iter=m["avg_iter"],
                avg_final_nodes=m["avg_final_nodes"],
                result_folder=result_folder_rel,
                plan_pkl=str(plan_pkl) if plan_pkl.exists() else None,
                log_file=str(logf),
                command=" ".join(shlex.quote(x) for x in cmd),
            )
            results.append(rec)
            print(f"[run] case={case_name} repeat={rep} rc={rc} wall={wall:.2f}s succ={rec.succ_count}/{rec.num_targets}")
            if rc != 0:
                print("[run] tail of failed log:\n" + tail_text(logf, 30))

    return results


def aggregate(records: List[RunRecord]) -> Dict[str, Dict[str, Optional[float]]]:
    grouped: Dict[str, List[RunRecord]] = {}
    for r in records:
        grouped.setdefault(r.case, []).append(r)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for case, recs in grouped.items():
        wall = [x.wall_time_sec for x in recs]
        planner = [x.planner_time_sec for x in recs if x.planner_time_sec is not None]
        succ = [x.succ_count for x in recs]
        succ_rate = [x.succ_rate for x in recs]
        avg_iter = [x.avg_iter for x in recs if x.avg_iter is not None]
        avg_nodes = [x.avg_final_nodes for x in recs if x.avg_final_nodes is not None]
        out[case] = {
            "runs": len(recs),
            "run_success_count": sum(1 for x in recs if x.return_code == 0),
            "wall_time_sec_mean": maybe_mean(wall),
            "planner_time_sec_mean": maybe_mean(planner),
            "succ_count_mean": maybe_mean([float(x) for x in succ]),
            "succ_rate_mean": maybe_mean(succ_rate),
            "avg_iter_mean": maybe_mean(avg_iter),
            "avg_final_nodes_mean": maybe_mean(avg_nodes),
        }

    serial = out.get("serial", {})
    par_nb = out.get("parallel_no_batch", {})
    par_b = out.get("parallel_batch", {})

    if serial.get("wall_time_sec_mean"):
        base = float(serial["wall_time_sec_mean"])
        for s in out.values():
            s["speedup_vs_serial"] = (base / float(s["wall_time_sec_mean"])) if s.get("wall_time_sec_mean") else None

    if par_nb.get("wall_time_sec_mean") and par_b.get("wall_time_sec_mean"):
        nb = float(par_nb["wall_time_sec_mean"])
        b = float(par_b["wall_time_sec_mean"])
        out.setdefault("comparison", {})["batch_speedup_vs_no_batch"] = nb / b if b > 0 else None
        out["comparison"]["batch_wall_time_delta_sec"] = nb - b
        if par_nb.get("succ_count_mean") is not None and par_b.get("succ_count_mean") is not None:
            out["comparison"]["batch_succ_count_delta"] = float(par_b["succ_count_mean"]) - float(par_nb["succ_count_mean"])

    return out


def write_reports(args: argparse.Namespace, report_dir: Path, pre_tests: Dict[str, Dict[str, str]], records: List[RunRecord], summary: Dict[str, Dict[str, Optional[float]]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "args": vars(args),
        "pre_tests": pre_tests,
        "records": [asdict(r) for r in records],
        "summary": summary,
    }
    (report_dir / "benchmark_results.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Template-Free Batch Impact Report")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append("```text")
    for k, v in vars(args).items():
        lines.append(f"{k}: {v}")
    lines.append("```")
    lines.append("")

    lines.append("## Pre-tests")
    lines.append("")
    for n, info in pre_tests.items():
        lines.append(f"- `{n}`: {info.get('status')} ({info.get('detail')})")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Runs | RC Pass | Wall Mean (s) | Planner Mean (s) | Speedup vs Serial | Succ Mean | Succ Rate Mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for case in ["serial", "parallel_no_batch", "parallel_batch"]:
        s = summary.get(case, {})
        lines.append(
            "| {case} | {runs} | {rc_ok} | {wall} | {planner} | {speedup} | {succ} | {succ_rate} |".format(
                case=case,
                runs=s.get("runs", 0),
                rc_ok=s.get("run_success_count", 0),
                wall=f"{s.get('wall_time_sec_mean', 0):.3f}" if s.get("wall_time_sec_mean") is not None else "NA",
                planner=f"{s.get('planner_time_sec_mean', 0):.3f}" if s.get("planner_time_sec_mean") is not None else "NA",
                speedup=f"{s.get('speedup_vs_serial', 0):.3f}x" if s.get("speedup_vs_serial") is not None else "NA",
                succ=f"{s.get('succ_count_mean', 0):.3f}" if s.get("succ_count_mean") is not None else "NA",
                succ_rate=f"{s.get('succ_rate_mean', 0):.4f}" if s.get("succ_rate_mean") is not None else "NA",
            )
        )

    comp = summary.get("comparison", {})
    lines.append("")
    lines.append("## Batch Effect")
    lines.append("")
    lines.append(f"- `batch_speedup_vs_no_batch`: {comp.get('batch_speedup_vs_no_batch')}")
    lines.append(f"- `batch_wall_time_delta_sec`: {comp.get('batch_wall_time_delta_sec')}")
    lines.append(f"- `batch_succ_count_delta`: {comp.get('batch_succ_count_delta')}")

    lines.append("")
    lines.append("## Per-run Logs")
    lines.append("")
    for r in records:
        lines.append(f"- `{r.case}` repeat={r.repeat}, rc={r.return_code}, wall={r.wall_time_sec:.3f}s, log=`{r.log_file}`")

    (report_dir / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    cuda_ok = detect_cuda_available(args.python, repo_root)
    if args.gpu >= 0 and cuda_ok is False:
        print(f"[gpu] requested gpu={args.gpu}, but CUDA is unavailable in {args.python}; fallback to cpu (--gpu -1).")
        args.gpu = -1

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    report_dir = repo_root / args.report_root / timestamp

    missing = check_preflight(args, repo_root)
    if missing:
        print("[preflight] Missing required files:")
        for m in missing:
            print(f"  - {m}")
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

    records = run_bench(args, repo_root, report_dir)
    summary = aggregate(records)
    write_reports(args, report_dir, pre_tests, records, summary)

    print("[done] report directory:", report_dir)
    print("[done] summary file:", report_dir / "benchmark_report.md")
    print("[done] raw json:", report_dir / "benchmark_results.json")

    fails = [r for r in records if r.return_code != 0]
    if fails and not args.allow_fail_runs:
        print(f"[done] benchmark has {len(fails)} failed run(s); use --allow-fail-runs to ignore.")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
