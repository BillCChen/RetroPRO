#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "[error] run as root: sudo bash $0 <CIDR...>"
  exit 1
fi

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <CAMPUS_CIDR1> [CAMPUS_CIDR2 ...]"
  echo "Example: $0 10.0.0.0/8 172.16.0.0/12"
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

# Only allow HTTP from campus CIDRs
for cidr in "$@"; do
  ufw allow from "$cidr" to any port 80 proto tcp comment 'RetroPRO campus HTTP'
done

# Explicitly deny application ports from outside
ufw deny 8000/tcp comment 'Block direct FastAPI 8000'
ufw deny 18100/tcp comment 'Block internal FastAPI 18100'

ufw --force enable
ufw status verbose
