#!/usr/bin/env bash
# 快捷运行入口；完整参数说明见同目录 z-command.md
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/retro_star"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate unirxn

export TP_FREE_DICT_DUMP_ON_EXIT="${TP_FREE_DICT_DUMP_ON_EXIT:-1}"

python retro_plan.py --seed 42 --use_value_fn --viz --gpu 0 \
  --expansion_topk 8 --iterations 51 \
  --one_step_type template_free --CSS --RD_list "[(7,2)]" --DICT \
  --test_routes pth_hard \
  --route_limit 10
