#!/usr/bin/env bash
set -euo pipefail

# Setup environment for template-free (CSS/DICT) inference.
# Linux example:
#   ENV_NAME=retrofree PYTHON_VER=3.9 bash scripts/setup_template_free_env.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-retrofree}"
PYTHON_VER="${PYTHON_VER:-3.9}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

install_deps() {
  if [[ -n "$PIP_INDEX_URL" ]]; then
    python -m pip install --upgrade pip -i "$PIP_INDEX_URL"
    python -m pip install "torch==2.6.*" -i "$PIP_INDEX_URL"
    python -m pip install "rdkit==2023.9.1" -i "$PIP_INDEX_URL"
    python -m pip install "rxnmapper" -i "$PIP_INDEX_URL"
    python -m pip install "numpy==1.26.4" -i "$PIP_INDEX_URL"
    python -m pip install "OpenNMT-py==2.2.0" -i "$PIP_INDEX_URL"
  else
    python -m pip install --upgrade pip
    python -m pip install "torch==2.6.*"
    python -m pip install "rdkit==2023.9.1"
    python -m pip install "rxnmapper"
    python -m pip install "numpy==1.26.4"
    python -m pip install "OpenNMT-py==2.2.0"
  fi

  python -m pip install -e "$ROOT_DIR/retro_star/packages/rdchiral"
  python -m pip install -e "$ROOT_DIR/retro_star/packages/mlp_retrosyn"
  python -m pip install -e "$ROOT_DIR"
}

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VER"
  conda activate "$ENV_NAME"
  install_deps
  echo "[done] conda env ready: $ENV_NAME"
  echo "[note] PyTorch 2.6 + OpenNMT 2.2 torch.load compatibility is auto-patched in tp_free_tools.py"
else
  echo "[warn] conda not found, fallback to local venv .venv_template_free"
  cd "$ROOT_DIR"
  python3 -m venv .venv_template_free
  # shellcheck source=/dev/null
  source .venv_template_free/bin/activate
  install_deps
  echo "[done] venv ready: $ROOT_DIR/.venv_template_free"
  echo "[note] PyTorch 2.6 + OpenNMT 2.2 torch.load compatibility is auto-patched in tp_free_tools.py"
fi
