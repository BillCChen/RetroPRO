# Plan A 双账号协作部署指南（Ubuntu/Debian）

本目录用于部署：`retro_star/main.py + retro_star/static/index.html`。

部署分工：

1. 应用账号（无 sudo）：准备 Python 环境、验证本机服务。
2. 管理员账号（有 sudo）：配置 `systemd`、`nginx`、`ufw`。

推荐优先使用仓库根目录下的自动化脚本（支持重复执行，适合网络波动时重试）：

1. 应用账号自检脚本：`scripts/plan_a_app_boot_check.sh`
2. 管理员一键部署脚本：`scripts/plan_a_admin_apply.sh`

## 1. 变量约定

先在两边都确认以下变量（按你的真实路径替换）：

```bash
export APP_USER="your_app_user"
export REPO_DIR="/path/to/RetroPRO"
export APP_DIR="$REPO_DIR/retro_star"
export PYTHON_BIN="/path/to/venv/bin/python"
export STARTING_MOLS="$APP_DIR/dataset/origin_dict.csv"
```

## 2. 应用账号步骤（无 sudo）

1. 进入项目并安装依赖（按你现有环境流程执行）。
2. 先做本机启动验证：

```bash
cd "$APP_DIR"
RETROTMP_HOST=127.0.0.1 RETROTMP_PORT=18100 \
RETROTMP_STARTING_MOLS_PATH="$STARTING_MOLS" \
"$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 18100
```

3. 另开终端本机检查：

```bash
curl -I http://127.0.0.1:18100/
curl -sS -X POST http://127.0.0.1:18100/api/preview-smiles \
  -H 'Content-Type: application/json' -d '{"smiles":"CCOCC"}'
```

可替代地，直接使用自动化自检脚本：

```bash
bash "$REPO_DIR/scripts/plan_a_app_boot_check.sh" \
  --app-dir "$APP_DIR" \
  --starting-mols "$STARTING_MOLS"
```

说明：若你已激活正确 conda 环境，可省略 `--python-bin`；否则建议显式传入。

## 3. 管理员账号步骤（有 sudo）

可替代地，直接使用管理员一键脚本（可重复执行）：

```bash
sudo bash "$REPO_DIR/scripts/plan_a_admin_apply.sh" \
  --repo-dir "$REPO_DIR" \
  --app-user "$APP_USER" \
  --app-dir "$APP_DIR" \
  --starting-mols "$STARTING_MOLS" \
  --campus-cidr 10.0.0.0/8 \
  --campus-cidr 172.16.0.0/12
```

若校园网网段暂不明确，可改用 basic auth 临时方案：

```bash
sudo bash "$REPO_DIR/scripts/plan_a_admin_apply.sh" \
  --repo-dir "$REPO_DIR" \
  --app-user "$APP_USER" \
  --app-dir "$APP_DIR" \
  --starting-mols "$STARTING_MOLS" \
  --basic-auth-user retropro \
  --basic-auth-pass 'CHANGE_ME_STRONG_PASSWORD'
```

说明：管理员脚本同样支持省略 `--python-bin`，会自动使用当前 `PATH` 中的 `python`。

## 3.1 systemd

1. 复制模板：

```bash
sudo cp "$REPO_DIR/deploy/plan_a_dual_account/systemd/retropro-plan-a.service" /etc/systemd/system/
sudo cp "$REPO_DIR/deploy/plan_a_dual_account/systemd/retropro-plan-a.env.example" /etc/retropro-plan-a.env
```

2. 编辑 `/etc/systemd/system/retropro-plan-a.service`：
- 把 `User=`、`Group=` 改成应用账号。
- 把 `WorkingDirectory=` 改成你的真实 `retro_star` 绝对路径。

3. 编辑 `/etc/retropro-plan-a.env`：
- 填好 `REPO_DIR`、`APP_DIR`、`PYTHON_BIN`、`RETROTMP_STARTING_MOLS_PATH`。

4. 启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now retropro-plan-a
sudo systemctl status retropro-plan-a --no-pager
```

## 3.2 nginx

1. 安装并启用：

```bash
sudo apt-get update
sudo apt-get install -y nginx
```

2. 复制配置：

```bash
sudo cp "$REPO_DIR/deploy/plan_a_dual_account/nginx/retropro_plan_a.conf" /etc/nginx/sites-available/retropro_plan_a
sudo ln -sf /etc/nginx/sites-available/retropro_plan_a /etc/nginx/sites-enabled/retropro_plan_a
sudo rm -f /etc/nginx/sites-enabled/default
```

3. 校园网白名单（推荐）：
- 复制示例：

```bash
sudo cp "$REPO_DIR/deploy/plan_a_dual_account/nginx/retropro-campus-allowlist.conf.example" /etc/nginx/snippets/retropro-campus-allowlist.conf
```

- 按网管给出的校园网网段编辑 `allow` 列表。

4. 临时兜底（网段暂不明确时）：
- 启用 basic auth：

```bash
sudo apt-get install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd_retropro your_username
```

- 打开 `retropro_plan_a.conf` 中 `auth_basic` 两行注释。

5. 重载：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 3.3 防火墙（ufw）

严格校园网白名单模式（推荐）：

```bash
sudo bash "$REPO_DIR/deploy/plan_a_dual_account/ufw/apply_firewall_campus_only.sh" <CIDR1> <CIDR2> ...
```

临时模式（未知网段 + basic auth）：

```bash
sudo bash "$REPO_DIR/deploy/plan_a_dual_account/ufw/apply_firewall_temp_public80.sh"
```

## 4. 验证清单

1. 服务器本机：

```bash
curl -I http://127.0.0.1:18100/
curl -I http://127.0.0.1/
```

2. 校园网另一台机器：
- 访问 `http://<服务器校园网IP>/`。
- 提交一次任务，验证 `/api/predict` -> `/api/result/{task_id}` -> 下载 JSON/HTML。

3. 安全验证：
- 端口 `8000/18100` 外部不可达。
- 非白名单来源访问 `80` 被拒绝（或需要 basic auth）。

## 5. 仅在直连失败时启用 FRP 兜底

触发条件：同校园网机器对服务器 IP `ping` 不通或路由不可达。

模板在：

- `frp/frpc.toml.template`
