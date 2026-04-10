# web_scripts

两个一键脚本：

1. `start_web.sh`
- 一键清理旧残留后启动：
  - `uvicorn`（默认 `0.0.0.0:18000`）
  - `cloudflared quick tunnel`
- 在终端打印可访问公网 URL（`trycloudflare`）
- 默认启用 Basic Auth
  - 默认用户名：`retropro`
  - 默认密码：`retropro2026`
  - 可通过环境变量覆盖

2. `clean_web.sh`
- 一键停止本项目相关 `uvicorn/cloudflared` 进程
- 清理 PID 文件（可选保留日志）

## 快速使用

```bash
bash web_scripts/start_web.sh
```

停止并清理：

```bash
bash web_scripts/clean_web.sh
```

## 常用环境变量

- `PYTHON_BIN`：指定 Python 可执行文件
- `CLOUDFLARED_BIN`：指定 cloudflared 路径
- `RETROTMP_PORT`：默认 `18000`
- `RETROTMP_STARTING_MOLS_PATH`：起始分子文件路径
- `RETROTMP_BASIC_AUTH_ENABLED`：`true/false`
- `RETROTMP_BASIC_AUTH_USER`：默认 `retropro`
- `RETROTMP_BASIC_AUTH_PASSWORD`：明文密码
- `RETROTMP_BASIC_AUTH_PASSWORD_SHA256`：SHA256 哈希（与明文二选一）
- `RETROTMP_PREDICT_RATE_LIMIT_ENABLED`：默认 `true`
- `RETROTMP_PREDICT_RATE_LIMIT_REQUESTS`：默认 `6`
- `RETROTMP_PREDICT_RATE_LIMIT_WINDOW_SEC`：默认 `60`
