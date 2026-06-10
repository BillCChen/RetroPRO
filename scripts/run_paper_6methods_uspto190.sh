#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEMPLATE_ITERATIONS="${TEMPLATE_ITERATIONS:-500}"

exec bash "$SCRIPT_DIR/run_paper_6method_subset_dataset.sh" \
  --dataset uspto190 \
  --dataset-label uspto190 \
  --template-dataset uspto190 \
  "$@"
