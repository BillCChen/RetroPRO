# Multi-Molecule Parallel Retrosynthesis Plan

## 1. Goal

Build a robust multi-molecule parallel planning path for `retro_plan.py` with a fixed-width inference pool:

- Keep `N` active molecules (recommended 5-10) in flight.
- Perform one-step expansion inference in batches across active molecules.
- Dispatch batched outputs back to each molecule task.
- When one task finishes (success/fail/max-iter), immediately refill from pending targets.
- Continue until all targets are completed.

Primary objective: increase effective GPU utilization and reduce end-to-end wall time caused by small single-item inference calls and repeated overhead.

## 2. Current Pain Points

1. Single-molecule search runs independently, causing fragmented and small inference calls.
2. One-step model calls are effectively single-sample; GPU has low occupancy.
3. Repeated per-molecule control-flow overhead and serial scheduling reduce throughput.
4. Local dev constraints:
   - no guaranteed CUDA
   - incomplete dataset/model files locally
   - some paths are server-only absolute paths

## 3. Target Architecture

### 3.1 Core Concepts

- **Task**: one target molecule planning state machine (search tree + iteration state + done status).
- **Pool Scheduler**: maintains fixed number of active tasks.
- **Batch Expander**: receives a list of frontier molecules from active tasks, performs batched one-step inference, and returns aligned results.

### 3.2 Layered Design

1. **Task layer (`MolStarTask`)**
   - Encapsulate step-wise search for one molecule.
   - Expose:
     - `ready_for_expansion()`
     - `next_frontier_smiles()`
     - `apply_expansion(result)`
     - `is_done()`
     - `finalize_message()`

2. **Scheduler layer (`parallel planner`)**
   - Initialize first `pool_width` tasks.
   - Loop:
     - collect expandable frontier smiles from active tasks
     - batch infer all smiles
     - route each result to corresponding task
     - finalize done tasks and refill pool
   - Persist incremental progress to `plan.pkl` just like legacy path.

3. **Model interface layer**
   - Add `run_batch(smiles_list, topk)` to one-step models where available.
   - Fallback adaptor: if model lacks `run_batch`, call `run` in a loop and return aligned list.

4. **Compatibility layer**
   - Keep legacy serial path intact.
   - Add explicit switch to enable multi-pool mode.

## 4. Data/Control Flow (Single Scheduler Tick)

1. Scheduler asks each active task for one frontier molecule.
2. Build `batch_smiles: List[str]` and `batch_meta: List[(task_id, node_ref)]`.
3. Call `one_step.run_batch(batch_smiles, topk=expansion_topk)`.
4. For each batched output:
   - sanitize/validate reactants
   - transform score -> cost (`-log(score)`)
   - call task expansion/backprop.
5. Update done states; collect final route info.
6. Refill pool from pending targets.

## 5. Proposed Code Changes

### 5.1 New/Updated modules

1. `retro_star/alg/molstar_task.py` (new)
   - Step-wise planner task abstraction.
2. `retro_star/alg/molstar_parallel.py` (new)
   - Pool scheduler over multiple tasks.
3. `retro_star/packages/mlp_retrosyn/mlp_retrosyn/mlp_inference.py` (update)
   - Add batched MLP forward + per-sample template application path.
4. `retro_star/retro_plan.py` (update)
   - Add mode switch and integrate parallel scheduler.
5. `retro_star/common/parse_args.py` (update)
   - Add explicit flag for multi-pool mode and pool config semantics.
6. Optional: `retro_star/common/prepare_utils.py` (light update)
   - Utility wrapper for batch-capable expansion function.

### 5.2 Runtime switches

- `--multi_pool` (bool, default false)
- `--parallel_num` (pool width, reused existing arg)
- `--parallel_expansions` (reserved for future >1 expansion per task per tick; phase-1 use 1)

## 6. Debugging Strategy Under Local Constraints

Because local machine may not have CUDA, full datasets, or full model files, we enforce a two-track validation strategy.

### 6.1 Track A: Deterministic local functional tests (CPU, synthetic)

- Add lightweight fake one-step model class with:
  - `run(smiles, topk)`
  - `run_batch(smiles_list, topk)`
- Add tiny synthetic starting molecules and target list.
- Validate scheduler semantics:
  - fixed-width pool maintained
  - task refill works
  - completion order independent from input order
  - result aggregation shape matches legacy fields

### 6.2 Track B: Interface compatibility checks

- Ensure serial path still works with existing arguments.
- Ensure missing dataset/model files fail fast with clear logs.
- Ensure parallel mode degrades gracefully to sequential `run` if `run_batch` unavailable.

### 6.3 Optional local tooling in venv

Create dedicated local env for rapid iteration and tests:

- `pytest` for scheduler unit tests
- `numpy` for deterministic mock scoring

No dependency on CUDA-required packages for core scheduler tests.

## 7. Performance Measurement Plan

### 7.1 Metrics

1. Throughput: molecules finished per hour.
2. Mean target latency.
3. One-step model call stats:
   - average batch size
   - batch call count
4. GPU utilization (cloud run): average and p95.

### 7.2 A/B protocol

- Baseline: serial legacy run with same seed/config.
- Variant: `--multi_pool --parallel_num {5,8,10}`.
- Keep all other parameters fixed (`expansion_topk`, `iterations`, test set).

## 8. Risks and Mitigations

1. **Search behavior drift** (parallel scheduling may alter global completion order)
   - Mitigation: each task internal expansion logic remains equivalent to serial per-step behavior.
2. **CPU bottleneck from template application**
   - Mitigation: batch NN forward first; profile and optionally parallelize template application later.
3. **Incomplete local data**
   - Mitigation: mock-based test harness + clear runtime checks.
4. **Memory pressure with large pools**
   - Mitigation: start default pool width at moderate value, expose CLI control.

## 9. Development Checklist

1. Add task abstraction for step-wise planning.
2. Add pool scheduler with refill behavior.
3. Add `run_batch` for template-based one-step model.
4. Integrate into `retro_plan.py` behind `--multi_pool`.
5. Add synthetic unit tests for scheduler and fallback behavior.
6. Add brief usage docs and migration notes.

## 10. Phase-1 Definition of Done

1. Parallel mode can run end-to-end with same output structure (`plan.pkl` keys unchanged).
2. Serial mode remains functional.
3. Local synthetic tests pass without CUDA/datasets.
4. Cloud run can toggle between serial and parallel via CLI.

## 11. Benchmark Script (Test + Collect + Compare)

新增脚本：

1. `scripts/benchmark_compare.py`
2. `scripts/run_benchmark_compare.sh`

能力说明：

1. 先执行预测试（`tests/scheduler_smoke_test.py` + `py_compile`）。
2. 自动跑 A/B（serial + `parallel_num` 列表）并收集每次运行日志。
3. 汇总 `plan.pkl` 指标并输出对比报告（Markdown + JSON）。
4. 当传入 GPU 但本地无 CUDA 时，自动回退到 CPU（可通过 `--strict-gpu` 禁止）。

最小运行示例（仓库根目录）：

```bash
bash scripts/run_benchmark_compare.sh \
  --seed 42 \
  --iterations 201 \
  --expansion-topk 8 \
  --one-step-type template_based \
  --test-routes uspto190 \
  --parallel-nums 5,8,10 \
  --repeats 1
```

产物路径：

1. 报告目录：`benchmark_reports/<timestamp>/`
2. 汇总报告：`benchmark_report.md`
3. 原始结构化数据：`benchmark_results.json`
4. 每次运行日志：`logs/run_*.log`

## 12. Template-Free Batch Acceleration (CSS/DICT)

已实现 `TP_free_Model.run_batch(...)`，用于 `template_free --CSS --DICT` 路径：

1. 对同一调度轮中的多个分子统一构造增强输入。
2. 合并为单次 retro 模型批量推理（不再按分子逐条调用）。
3. 合并为单次 forward 模型批量推理。
4. 将 batched 输出按 owner molecule 回填，再分别做模板抽取与树扩展。

实现细节：

1. `run(x)` 保持兼容，内部委托到 `run_batch([x])`。
2. 支持通过环境变量调节吞吐相关 batch size：
   - `TP_FREE_RETRO_BATCH_SIZE`（默认 128）
   - `TP_FREE_FORWARD_BATCH_SIZE`（默认 256）
   - `TP_FREE_MAPPER_BATCH_SIZE`（默认 64）
3. 修复/增强了 DICT 相关逻辑中的 key 对齐与重复写入问题。
4. OpenNMT 与 PyTorch 版本兼容由环境侧保证，不在仓库代码中强制打补丁。

## 13. Template-Free Batch Impact Benchmark Script

新增脚本：

1. `scripts/benchmark_template_free_batch_impact.py`
2. `scripts/run_template_free_batch_impact.sh`

测试目标：

1. `serial`：单分子基线
2. `parallel_no_batch`：多分子池 + 强制关闭 `run_batch`
3. `parallel_batch`：多分子池 + 启用 `run_batch`

核心对比指标：

1. `batch_speedup_vs_no_batch`
2. `batch_wall_time_delta_sec`
3. `batch_succ_count_delta`

运行示例（服务器）：

```bash
bash scripts/run_template_free_batch_impact.sh \
  --python "$(which python)" \
  --gpu 1 \
  --seed 42 \
  --iterations 201 \
  --expansion-topk 8 \
  --test-routes uspto190 \
  --parallel-num 8 \
  --repeats 1
```
