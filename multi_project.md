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

