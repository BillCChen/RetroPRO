#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "[error] run as root: sudo bash $0"
  exit 1
fi

if ! command -v ufw >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ufw
fi

ufw --force disable
ufw --force reset

ufw default deny incoming
ufw default allow outgoing

ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP with nginx auth'

# Explicitly deny application ports from outside
ufw deny 8000/tcp comment 'Block direct FastAPI 8000'
ufw deny 18100/tcp comment 'Block internal FastAPI 18100'

ufw --force enable
ufw status verbose

echo "[warn] 80 is globally open. Enable nginx basic auth immediately."
