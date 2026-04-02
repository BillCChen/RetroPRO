#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${SCRIPT_DIR}/runtime"
PORT="${RETROTMP_PORT:-18000}"
QUIET="false"
KEEP_LOGS="false"

for arg in "$@"; do
  case "${arg}" in
    --quiet) QUIET="true" ;;
    --keep-logs) KEEP_LOGS="true" ;;
    *) ;;
  esac
done

log() {
  if [[ "${QUIET}" != "true" ]]; then
    echo "$@"
  fi
}

kill_pid_file() {
  local file="$1"
  local name="$2"
  if [[ ! -f "${file}" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "${file}" 2>/dev/null || true)"
  if [[ -n "${pid}" && "${pid}" =~ ^[0-9]+$ ]]; then
    if kill -0 "${pid}" 2>/dev/null; then
      log "[step] stopping ${name} pid=${pid}"
      kill "${pid}" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "${pid}" 2>/dev/null; then
          break
        fi
        sleep 0.2
      done
      if kill -0 "${pid}" 2>/dev/null; then
        log "[warn] force killing ${name} pid=${pid}"
        kill -9 "${pid}" 2>/dev/null || true
      fi
    fi
  fi
  rm -f "${file}"
}

mkdir -p "${RUNTIME_DIR}"
UVICORN_PID_FILE="${RUNTIME_DIR}/uvicorn_${PORT}.pid"
CLOUDFLARED_PID_FILE="${RUNTIME_DIR}/cloudflared_${PORT}.pid"
UVICORN_LOG_FILE="${RUNTIME_DIR}/uvicorn_${PORT}.log"
CLOUDFLARED_LOG_FILE="${RUNTIME_DIR}/cloudflared_${PORT}.log"

kill_pid_file "${CLOUDFLARED_PID_FILE}" "cloudflared"
kill_pid_file "${UVICORN_PID_FILE}" "uvicorn"

# Fallback cleanup for unexpected leftovers
if command -v pgrep >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    log "[step] stopping leftover cloudflared pid=${pid}"
    kill "${pid}" 2>/dev/null || true
  done < <(pgrep -f "cloudflared tunnel --url http://127.0.0.1:${PORT}" || true)

  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    local_cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${local_cmd}" == *"uvicorn main:app"* && "${local_cmd}" == *"--port ${PORT}"* ]]; then
      log "[step] stopping leftover uvicorn pid=${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  done < <(pgrep -f "uvicorn main:app" || true)
fi

if command -v lsof >/dev/null 2>&1; then
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    local_cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    if [[ "${local_cmd}" == *"uvicorn main:app"* ]]; then
      log "[step] stopping listening uvicorn pid=${pid} on port ${PORT}"
      kill "${pid}" 2>/dev/null || true
    fi
  done < <(lsof -t -nP -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)
fi

if [[ "${KEEP_LOGS}" != "true" ]]; then
  rm -f "${UVICORN_LOG_FILE}" "${CLOUDFLARED_LOG_FILE}"
fi

log "[ok] cleanup finished for port ${PORT}"
