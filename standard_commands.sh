#!/usr/bin/env bash
# =============================================================================
# RetroPRO - Standard Command Reference
# =============================================================================
# All commands assume:  cd retro_star/  (for retro_plan.py direct calls)
#                  or   cd <repo_root>/ (for scripts/)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Benchmark Script (one-click test via run_template_free_batch_impact.sh)
# ─────────────────────────────────────────────────────────────────────────────

# 1a. Quick smoke run (50 iters, 190 molecules, pool=64, single GPU)
bash scripts/run_template_free_batch_impact.sh \
  --gpu 0 \
  --parallel-num 64 \
  --iterations 50 \
  --max-targets 190

# 1b. Full benchmark (201 iters) — matches the original evaluation protocol
bash scripts/run_template_free_batch_impact.sh --gpu 0 --parallel-num 64 --iterations 101 --max-targets 190

# 1c. Fine-grained batch-size sweep via env vars (override defaults at runtime)
TP_FREE_RETRO_BATCH_SIZE=256 TP_FREE_FORWARD_BATCH_SIZE=256 TP_FREE_MAPPER_BATCH_SIZE=128 bash scripts/run_template_free_batch_impact.sh --gpu 0 --parallel-num 32 --iterations 50 --max-targets 190

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — retro_plan.py: Template-Free Direct Calls
# ─────────────────────────────────────────────────────────────────────────────
# All commands below must be run from:  cd retro_star/

# 2a. Serial baseline (legacy, one molecule at a time)
python retro_plan.py \
  --seed 42 --gpu 0 \
  --one_step_type template_free \
  --expansion_topk 8 \
  --iterations 201 \
  --CSS --DICT \
  --RD_list "[(7,2),(3,0)]" \
  --use_value_fn \
  --test_routes uspto190

# 2b. Single-GPU parallel pool — recommended for L40 (pool=64 saturates GPU batch)
python retro_plan.py \
  --seed 42 --gpu 0 \
  --one_step_type template_free \
  --expansion_topk 8 \
  --iterations 201 \
  --CSS --DICT \
  --RD_list "[(7,2),(3,0)]" \
  --use_value_fn \
  --multi_pool --parallel_num 64 \
  --test_routes uspto190

# 2c. Single-GPU parallel pool, no DICT (faster per-call, no template cache)
python retro_plan.py \
  --seed 42 --gpu 0 \
  --one_step_type template_free \
  --expansion_topk 8 \
  --iterations 201 \
  --CSS \
  --use_value_fn \
  --multi_pool --parallel_num 64 \
  --test_routes uspto190

# 2d. Multi-GPU data parallel — 4 cards, shared DICT, pool=64 per card
#     (requires --gpu_list; --gpu is ignored when gpu_list is set)
python retro_plan.py \
  --seed 42 --gpu 0 \
  --gpu_list "1,2,3" \
  --one_step_type template_free \
  --expansion_topk 8 \
  --iterations 50 \
  --CSS --DICT \
  --RD_list "[(7,2),(3,0)]" \
  --use_value_fn \
  --parallel_num 64 \
  --test_routes uspto190

# 2e. Multi-GPU, 2 cards
python retro_plan.py --seed 42 --gpu 0 --gpu_list "0,1,2,3" --one_step_type template_free --expansion_topk 8 --iterations 101 --CSS --DICT --RD_list "[(9,0),(5,0)]" --use_value_fn --parallel_num 64 --test_routes uspto190


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — retro_plan.py: Template-Based Direct Calls
# ─────────────────────────────────────────────────────────────────────────────

# 3a. Serial template-based
python retro_plan.py \
  --seed 42 --gpu 0 \
  --one_step_type template_based \
  --expansion_topk 50 \
  --iterations 201 \
  --use_value_fn \
  --test_routes uspto190

# 3b. Parallel pool template-based
python retro_plan.py \
  --seed 42 --gpu 0 \
  --one_step_type template_based \
  --expansion_topk 50 \
  --iterations 201 \
  --use_value_fn \
  --multi_pool --parallel_num 64 \
  --test_routes uspto190

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Env-var Tuning Reference
# ─────────────────────────────────────────────────────────────────────────────
# Override model batch sizes without changing source code.
# Defaults (post-fix): RETRO=512, FORWARD=512, MAPPER=256
#
# Example — conservative settings if VRAM is limited:
#   TP_FREE_RETRO_BATCH_SIZE=128
#   TP_FREE_FORWARD_BATCH_SIZE=256
#   TP_FREE_MAPPER_BATCH_SIZE=64
#
# Example — aggressive settings for A100/H100 (80 GB VRAM):
#   TP_FREE_RETRO_BATCH_SIZE=1024
#   TP_FREE_FORWARD_BATCH_SIZE=1024
#   TP_FREE_MAPPER_BATCH_SIZE=512
#
# Force per-item expansion (no run_batch) for debugging:
#   RETROPRO_DISABLE_RUN_BATCH=1

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Web Server
# ─────────────────────────────────────────────────────────────────────────────
# Run from:  cd retro_star/

RETROTMP_HOST=127.0.0.1 \
RETROTMP_PORT=18000 \
RETROTMP_STARTING_MOLS_PATH="$(pwd)/dataset/origin_dict.csv" \
python -m uvicorn main:app --host 127.0.0.1 --port 18000
