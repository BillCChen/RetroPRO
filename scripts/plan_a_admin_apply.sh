#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  sudo bash scripts/plan_a_admin_apply.sh \
    --repo-dir /abs/path/RetroPRO \
    --app-user <app_user> \
    --app-dir /abs/path/RetroPRO/retro_star \
    --starting-mols /abs/path/RetroPRO/retro_star/dataset/origin_dict.csv \
    [--campus-cidr 10.0.0.0/8 --campus-cidr 172.16.0.0/12 ...] \
    [--basic-auth-user <name> --basic-auth-pass <pass>] \
    [--skip-ufw]

Optional:
  --python-bin /abs/path/venv/bin/python
  If omitted, script will use `python` from current PATH.

Notes:
  1) Provide at least one --campus-cidr OR basic auth credentials.
  2) If no campus CIDR is provided, basic auth is mandatory.
USAGE
}

if [[ "${EUID}" -ne 0 ]]; then
  echo "[error] this script must run with sudo/root"
  exit 1
fi

REPO_DIR=""
APP_USER=""
APP_DIR=""
PYTHON_BIN=""
STARTING_MOLS=""
SKIP_UFW="false"
BASIC_AUTH_USER=""
BASIC_AUTH_PASS=""

declare -a CAMPUS_CIDRS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --app-user)
      APP_USER="$2"
      shift 2
      ;;
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
    --campus-cidr)
      CAMPUS_CIDRS+=("$2")
      shift 2
      ;;
    --basic-auth-user)
      BASIC_AUTH_USER="$2"
      shift 2
      ;;
    --basic-auth-pass)
      BASIC_AUTH_PASS="$2"
      shift 2
      ;;
    --skip-ufw)
      SKIP_UFW="true"
      shift 1
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

if [[ -z "${REPO_DIR}" || -z "${APP_USER}" || -z "${APP_DIR}" || -z "${STARTING_MOLS}" ]]; then
  echo "[error] missing required arguments"
  usage
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[error] repo dir not found: ${REPO_DIR}"
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

if [[ "${#CAMPUS_CIDRS[@]}" -eq 0 ]]; then
  if [[ -z "${BASIC_AUTH_USER}" || -z "${BASIC_AUTH_PASS}" ]]; then
    echo "[error] no campus CIDR provided; basic auth credentials are required"
    exit 1
  fi
fi

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      echo "[error] command failed after ${attempts} attempts: $*"
      return 1
    fi
    local sleep_sec=$((n * 3))
    echo "[warn] attempt ${n} failed, retrying in ${sleep_sec}s: $*"
    sleep "${sleep_sec}"
    ((n++))
  done
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\\&/]/\\&/g'
}

echo "[step] install required packages with retry"
retry 5 apt-get update
retry 5 apt-get install -y nginx ufw
if [[ -n "${BASIC_AUTH_USER}" && -n "${BASIC_AUTH_PASS}" ]]; then
  retry 5 apt-get install -y apache2-utils
fi

echo "[step] install systemd unit"
SERVICE_SRC="${REPO_DIR}/deploy/plan_a_dual_account/systemd/retropro-plan-a.service"
ENV_SRC="${REPO_DIR}/deploy/plan_a_dual_account/systemd/retropro-plan-a.env.example"

if [[ ! -f "${SERVICE_SRC}" || ! -f "${ENV_SRC}" ]]; then
  echo "[error] deployment templates not found under ${REPO_DIR}/deploy/plan_a_dual_account/systemd"
  exit 1
fi

cp "${SERVICE_SRC}" /etc/systemd/system/retropro-plan-a.service

APP_USER_ESCAPED="$(escape_sed "${APP_USER}")"
APP_DIR_ESCAPED="$(escape_sed "${APP_DIR}")"

sed -i "s/APP_USER_PLACEHOLDER/${APP_USER_ESCAPED}/g" /etc/systemd/system/retropro-plan-a.service
sed -i "s/APP_DIR_PLACEHOLDER/${APP_DIR_ESCAPED}/g" /etc/systemd/system/retropro-plan-a.service

cp "${ENV_SRC}" /etc/retropro-plan-a.env

cat > /etc/retropro-plan-a.env <<ENV
REPO_DIR=${REPO_DIR}
APP_DIR=${APP_DIR}
PYTHON_BIN=${PYTHON_BIN}
RETROTMP_HOST=127.0.0.1
RETROTMP_PORT=18000
RETROTMP_STARTING_MOLS_PATH=${STARTING_MOLS}
ENV

systemctl daemon-reload
systemctl enable --now retropro-plan-a
systemctl restart retropro-plan-a

if ! systemctl is-active --quiet retropro-plan-a; then
  echo "[error] retropro-plan-a failed to start"
  systemctl status retropro-plan-a --no-pager || true
  journalctl -u retropro-plan-a -n 100 --no-pager || true
  exit 1
fi

echo "[step] install nginx config"
NGINX_SRC="${REPO_DIR}/deploy/plan_a_dual_account/nginx/retropro_plan_a.conf"
ALLOWLIST_EXAMPLE="${REPO_DIR}/deploy/plan_a_dual_account/nginx/retropro-campus-allowlist.conf.example"

if [[ ! -f "${NGINX_SRC}" ]]; then
  echo "[error] nginx template not found: ${NGINX_SRC}"
  exit 1
fi

cp "${NGINX_SRC}" /etc/nginx/sites-available/retropro_plan_a
ln -sf /etc/nginx/sites-available/retropro_plan_a /etc/nginx/sites-enabled/retropro_plan_a
rm -f /etc/nginx/sites-enabled/default

if [[ "${#CAMPUS_CIDRS[@]}" -gt 0 ]]; then
  {
    echo "# Auto-generated by plan_a_admin_apply.sh"
    for cidr in "${CAMPUS_CIDRS[@]}"; do
      echo "allow ${cidr};"
    done
    echo "deny all;"
  } > /etc/nginx/snippets/retropro-campus-allowlist.conf

  # Keep include active, keep auth lines commented
  sed -i 's@^[[:space:]]*auth_basic @    # auth_basic @' /etc/nginx/sites-available/retropro_plan_a || true
  sed -i 's@^[[:space:]]*auth_basic_user_file @    # auth_basic_user_file @' /etc/nginx/sites-available/retropro_plan_a || true
  sed -i 's@^[[:space:]]*#[[:space:]]*include /etc/nginx/snippets/retropro-campus-allowlist.conf;@    include /etc/nginx/snippets/retropro-campus-allowlist.conf;@' /etc/nginx/sites-available/retropro_plan_a || true
else
  cp "${ALLOWLIST_EXAMPLE}" /etc/nginx/snippets/retropro-campus-allowlist.conf

  # Disable allowlist include
  sed -i 's@^[[:space:]]*include /etc/nginx/snippets/retropro-campus-allowlist.conf;@    # include /etc/nginx/snippets/retropro-campus-allowlist.conf;@' /etc/nginx/sites-available/retropro_plan_a

  # Enable basic auth lines
  sed -i 's@^[[:space:]]*#[[:space:]]*auth_basic @    auth_basic @' /etc/nginx/sites-available/retropro_plan_a
  sed -i 's@^[[:space:]]*#[[:space:]]*auth_basic_user_file @    auth_basic_user_file @' /etc/nginx/sites-available/retropro_plan_a

  printf '%s\n' "${BASIC_AUTH_PASS}" | htpasswd -i -c /etc/nginx/.htpasswd_retropro "${BASIC_AUTH_USER}"
fi

nginx -t
systemctl enable --now nginx
systemctl reload nginx

echo "[step] apply firewall"
if [[ "${SKIP_UFW}" == "true" ]]; then
  echo "[warn] skipped ufw changes by --skip-ufw"
else
  if [[ "${#CAMPUS_CIDRS[@]}" -gt 0 ]]; then
    bash "${REPO_DIR}/deploy/plan_a_dual_account/ufw/apply_firewall_campus_only.sh" "${CAMPUS_CIDRS[@]}"
  else
    bash "${REPO_DIR}/deploy/plan_a_dual_account/ufw/apply_firewall_temp_public80.sh"
  fi
fi

echo "[done] deployment applied"
echo "[check] systemd: systemctl status retropro-plan-a --no-pager"
echo "[check] nginx:   systemctl status nginx --no-pager"
echo "[check] local:   curl -I http://127.0.0.1:18000/ && curl -I http://127.0.0.1/"
