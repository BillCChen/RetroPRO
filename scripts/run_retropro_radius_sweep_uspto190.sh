#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/run_retropro_radius_sweep_dataset.sh" \
  --dataset uspto190 \
  --dataset-label uspto190 \
  "$@"
