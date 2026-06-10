#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEMPLATE_ITERATIONS="${TEMPLATE_ITERATIONS:-1000}"

exec bash "$SCRIPT_DIR/run_paper_6method_subset_dataset.sh" \
  --dataset pth_hard \
  --dataset-label pistachio_hard \
  --template-dataset pistachio_hard_100 \
  "$@"
