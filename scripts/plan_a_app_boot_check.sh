#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/plan_a_app_boot_check.sh \
    --app-dir /abs/path/RetroPRO/retro_star \
    --python-bin /abs/path/venv/bin/python \
    --starting-mols /abs/path/RetroPRO/retro_star/dataset/origin_dict.csv
USAGE
}

APP_DIR=""
PYTHON_BIN=""
STARTING_MOLS=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --app-dir)
      APP_DIR="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --starting-mols)
      STARTING_MOLS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${APP_DIR}" || -z "${PYTHON_BIN}" || -z "${STARTING_MOLS}" ]]; then
  echo "[error] missing required args"
  usage
  exit 1
fi

if [[ ! -d "${APP_DIR}" ]]; then
  echo "[error] app dir not found: ${APP_DIR}"
  exit 1
fi

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "[error] python bin not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${STARTING_MOLS}" ]]; then
  echo "[error] starting molecules file not found: ${STARTING_MOLS}"
  exit 1
fi

cd "${APP_DIR}"
REPO_DIR="$(cd "${APP_DIR}/.." && pwd)"

export RETROTMP_HOST=127.0.0.1
export RETROTMP_PORT=18000
export RETROTMP_STARTING_MOLS_PATH="${STARTING_MOLS}"

LOG_FILE="${APP_DIR}/plan_a_boot_check.log"

cleanup() {
  if [[ -n "${APP_PID:-}" ]] && kill -0 "${APP_PID}" 2>/dev/null; then
    kill "${APP_PID}" 2>/dev/null || true
    wait "${APP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[step] starting uvicorn for local boot check"
"${PYTHON_BIN}" -m uvicorn main:app --host 127.0.0.1 --port 18000 >"${LOG_FILE}" 2>&1 &
APP_PID="$!"

echo "[step] waiting for service readiness"
for i in $(seq 1 30); do
  if curl -fsS -I http://127.0.0.1:18000/ >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if [[ "$i" -eq 30 ]]; then
    echo "[error] service failed to become ready"
    echo "[info] last log lines:"
    tail -n 120 "${LOG_FILE}" || true
    exit 1
  fi
done

echo "[step] running smoke checks"
APP_URL="http://127.0.0.1:18000" EDGE_URL="http://127.0.0.1:18000" bash "${REPO_DIR}/scripts/plan_a_smoke_test.sh"

echo "[done] app boot check passed"
