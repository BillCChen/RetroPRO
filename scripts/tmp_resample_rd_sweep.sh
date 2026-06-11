#!/usr/bin/env bash
# Temporary helper to sweep RD_list values in the resampling tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/retro_star"

RUN_STAMP="$(date '+%m%d_%H%M%S')"
RUN_ROOT="results/pth_hard/rd_sweep_${RUN_STAMP}"
WRAPPER_LOG="${RUN_ROOT}/wrapper.log"

mkdir -p "${RUN_ROOT}"

RD_LISTS=(
  "[(9,0),(3,0)]"
  "[(8,0),(3,0)]"
  "[(7,0),(3,0)]"
  "[(6,0),(3,0)]"
  "[(5,0),(3,0)]"
  "[(4,0),(3,0)]"
  "[(9,0),(2,0)]"
  "[(8,0),(2,0)]"
  "[(7,0),(2,0)]"
  "[(6,0),(2,0)]"
  "[(5,0),(2,0)]"
  "[(4,0),(2,0)]"

)

echo "[info] cwd: $(pwd)"
echo "[info] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
echo "[info] starting RD sweep at $(date '+%F %T')"
echo "[info] wrapper log: ${WRAPPER_LOG}"

for rd in "${RD_LISTS[@]}"; do
  rd_tag="$(echo "${rd}" | tr -d '[]() ' | tr ',' '_' )"
  result_dir="${RUN_ROOT}/RD_${rd_tag}"

  echo
  echo "[info] =============================================="
  echo "[info] running RD_list=${rd}"
  echo "[info] result_folder=${result_dir}"
  echo "[info] start time: $(date '+%F %T')"

  python retro_plan.py --seed 42 --use_value_fn --viz --gpu 1 \
    --expansion_topk 8 --iterations 101 \
    --one_step_type template_free --CSS --RD_list "${rd}" --DICT \
    --test_routes pth_hard \
    --collect_expansion_data \
    --route_limit 101 \
    --result_folder "${result_dir}" 2>&1 | tee -a "${WRAPPER_LOG}"

  echo "[info] finished RD_list=${rd} at $(date '+%F %T')"
done

echo
echo "[info] all RD_list runs finished at $(date '+%F %T')"
