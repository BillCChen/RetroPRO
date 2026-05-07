#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
mkdir -p "${RUNTIME_DIR}"

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

APP_DIR="${APP_DIR:-${REPO_ROOT}/retro_star}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[error] python/python3 not found in PATH; set PYTHON_BIN explicitly"
  exit 1
fi

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-}"
if [[ -z "${CLOUDFLARED_BIN}" ]]; then
  CLOUDFLARED_BIN="$(command -v cloudflared || true)"
fi
if [[ -z "${CLOUDFLARED_BIN}" && -x "${HOME}/bin/cloudflared" ]]; then
  CLOUDFLARED_BIN="${HOME}/bin/cloudflared"
fi
if [[ -z "${CLOUDFLARED_BIN}" ]]; then
  echo "[error] cloudflared not found; set CLOUDFLARED_BIN or install cloudflared"
  exit 1
fi

HOST="${RETROTMP_HOST:-0.0.0.0}"
PORT="${RETROTMP_PORT:-18100}"
STARTING_MOLS="${RETROTMP_STARTING_MOLS_PATH:-${APP_DIR}/dataset/origin_dict.csv}"

AUTH_ENABLED="${RETROTMP_BASIC_AUTH_ENABLED:-true}"
AUTH_USER="${RETROTMP_BASIC_AUTH_USER:-retropro}"
AUTH_PASSWORD="${RETROTMP_BASIC_AUTH_PASSWORD:-retropro2026}"
AUTH_PASSWORD_SHA256="${RETROTMP_BASIC_AUTH_PASSWORD_SHA256:-}"

RATE_LIMIT_ENABLED="${RETROTMP_PREDICT_RATE_LIMIT_ENABLED:-true}"
RATE_LIMIT_REQUESTS="${RETROTMP_PREDICT_RATE_LIMIT_REQUESTS:-6}"
RATE_LIMIT_WINDOW_SEC="${RETROTMP_PREDICT_RATE_LIMIT_WINDOW_SEC:-60}"

if [[ ! -d "${APP_DIR}" ]]; then
  echo "[error] APP_DIR not found: ${APP_DIR}"
  exit 1
fi
if [[ ! -f "${STARTING_MOLS}" ]]; then
  echo "[error] starting molecules file not found: ${STARTING_MOLS}"
  exit 1
fi

if is_true "${AUTH_ENABLED}"; then
  if [[ -z "${AUTH_USER}" ]]; then
    echo "[error] RETROTMP_BASIC_AUTH_USER is empty while auth is enabled"
    exit 1
  fi
  if [[ -z "${AUTH_PASSWORD}" && -z "${AUTH_PASSWORD_SHA256}" ]]; then
    echo "[error] auth is enabled but no password is configured"
    echo "[info] set RETROTMP_BASIC_AUTH_PASSWORD or RETROTMP_BASIC_AUTH_PASSWORD_SHA256"
    exit 1
  fi
fi

UVICORN_PID_FILE="${RUNTIME_DIR}/uvicorn_${PORT}.pid"
CLOUDFLARED_PID_FILE="${RUNTIME_DIR}/cloudflared_${PORT}.pid"
UVICORN_LOG_FILE="${RUNTIME_DIR}/uvicorn_${PORT}.log"
CLOUDFLARED_LOG_FILE="${RUNTIME_DIR}/cloudflared_${PORT}.log"

# Ensure clean state to avoid stale processes affecting startup.
RETROTMP_PORT="${PORT}" bash "${SCRIPT_DIR}/clean_web.sh" --quiet --keep-logs || true

: > "${UVICORN_LOG_FILE}"
: > "${CLOUDFLARED_LOG_FILE}"

echo "[step] starting uvicorn on ${HOST}:${PORT}"
(
  cd "${APP_DIR}"
  nohup env \
    RETROTMP_STARTING_MOLS_PATH="${STARTING_MOLS}" \
    RETROTMP_BASIC_AUTH_ENABLED="${AUTH_ENABLED}" \
    RETROTMP_BASIC_AUTH_USER="${AUTH_USER}" \
    RETROTMP_BASIC_AUTH_PASSWORD="${AUTH_PASSWORD}" \
    RETROTMP_BASIC_AUTH_PASSWORD_SHA256="${AUTH_PASSWORD_SHA256}" \
    RETROTMP_PREDICT_RATE_LIMIT_ENABLED="${RATE_LIMIT_ENABLED}" \
    RETROTMP_PREDICT_RATE_LIMIT_REQUESTS="${RATE_LIMIT_REQUESTS}" \
    RETROTMP_PREDICT_RATE_LIMIT_WINDOW_SEC="${RATE_LIMIT_WINDOW_SEC}" \
    "${PYTHON_BIN}" -m uvicorn main:app --host "${HOST}" --port "${PORT}" \
    </dev/null >>"${UVICORN_LOG_FILE}" 2>&1 &
  echo $! > "${UVICORN_PID_FILE}"
)

UVICORN_PID="$(cat "${UVICORN_PID_FILE}")"

for _ in $(seq 1 90); do
  if ! kill -0 "${UVICORN_PID}" 2>/dev/null; then
    echo "[error] uvicorn exited unexpectedly"
    tail -n 120 "${UVICORN_LOG_FILE}" || true
    exit 1
  fi
  code="$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)"
  if [[ "${code}" == "200" || "${code}" == "401" ]]; then
    break
  fi
  sleep 1
done

if [[ "${code:-}" != "200" && "${code:-}" != "401" ]]; then
  echo "[error] uvicorn readiness check timed out"
  tail -n 120 "${UVICORN_LOG_FILE}" || true
  exit 1
fi

echo "[step] starting cloudflared quick tunnel"
nohup "${CLOUDFLARED_BIN}" tunnel --url "http://127.0.0.1:${PORT}" \
  </dev/null >>"${CLOUDFLARED_LOG_FILE}" 2>&1 &
echo $! > "${CLOUDFLARED_PID_FILE}"

CLOUDFLARED_PID="$(cat "${CLOUDFLARED_PID_FILE}")"
TUNNEL_URL=""
for _ in $(seq 1 90); do
  if ! kill -0 "${CLOUDFLARED_PID}" 2>/dev/null; then
    echo "[error] cloudflared exited unexpectedly"
    tail -n 120 "${CLOUDFLARED_LOG_FILE}" || true
    exit 1
  fi
  TUNNEL_URL="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "${CLOUDFLARED_LOG_FILE}" | tail -1 || true)"
  if [[ -n "${TUNNEL_URL}" ]]; then
    break
  fi
  sleep 1
done

if [[ -z "${TUNNEL_URL}" ]]; then
  echo "[error] did not find tunnel URL from cloudflared logs"
  tail -n 120 "${CLOUDFLARED_LOG_FILE}" || true
  exit 1
fi

echo
echo "[ok] Web service started"
echo "[info] local URL: http://127.0.0.1:${PORT}/"
echo "[info] public URL: ${TUNNEL_URL}"

if is_true "${AUTH_ENABLED}"; then
  echo "[info] basic auth user: ${AUTH_USER}"
  if [[ -n "${AUTH_PASSWORD_SHA256}" ]]; then
    echo "[info] basic auth password: (not printed; using SHA256 env value)"
  else
    echo "[info] basic auth password: ${AUTH_PASSWORD}"
  fi
fi

echo "[info] uvicorn pid: ${UVICORN_PID}"
echo "[info] cloudflared pid: ${CLOUDFLARED_PID}"
echo "[info] uvicorn log: ${UVICORN_LOG_FILE}"
echo "[info] cloudflared log: ${CLOUDFLARED_LOG_FILE}"
echo "[info] cleanup command: bash ${SCRIPT_DIR}/clean_web.sh"
