# 无 sudo 快速公网访问通用手册（readme_web）

本文档是一个与具体业务代码解耦的通用方案，目标是：
- 在只有普通账号（无 sudo）时，快速把本地 Web 服务发布到公网可访问
- 提供最小安全加固（认证 + 限流）
- 方便后续任何项目直接复用

## 1. 适用场景

满足以下条件即可：
- Linux 服务器可联网
- 你有普通用户 shell 权限
- 本地服务已经能在 `127.0.0.1:<PORT>` 正常访问
- 可使用 `cloudflared`（PATH 中存在，或放在 `~/bin/cloudflared`）

## 2. 方案总览

最小链路：
1. 应用进程（例如 `uvicorn` / `gunicorn` / `node`）
2. `cloudflared tunnel --url http://127.0.0.1:<PORT>` 创建公网入口
3. 应用层防护：Basic Auth + 接口限流

优点：
- 不依赖 Nginx/systemd/防火墙规则，部署很快

限制：
- Quick Tunnel 域名通常是临时的，重启后可能变化
- 不适合高可靠生产长期 SLA

## 3. 一次性准备

### 3.1 激活运行环境

```bash
conda activate <your_env>
cd <your_project_root>
```

### 3.2 安装基础依赖（按项目需要）

```bash
python -m pip install uvicorn fastapi
```

### 3.3 准备 cloudflared

```bash
which cloudflared
```

如果没有：
- 放置二进制到 `~/bin/cloudflared`
- 并执行 `chmod +x ~/bin/cloudflared`

## 4. 启动前自检（建议）

先确认应用本地可用：

```bash
# 例：以 FastAPI + uvicorn 为例
cd <app_dir>
python -m uvicorn <module>:<app_object> --host 127.0.0.1 --port 18100
```

另开终端验证：

```bash
curl --noproxy '*' -i http://127.0.0.1:18100/
```

说明：
- 若返回 `200`，服务就绪
- 若返回 `401`（你开启了认证），也表示服务就绪
- 若返回 `405` 且 `allow: GET`，通常只是你用了 `HEAD` 方法，不是服务挂掉

## 5. 快速启动公网访问（无 sudo）

## 5.1 推荐环境变量模板

```bash
export WEB_HOST=127.0.0.1
export WEB_PORT=18100

# 认证（建议开启）
export WEB_BASIC_AUTH_ENABLED=true
export WEB_BASIC_AUTH_USER=webuser
export WEB_BASIC_AUTH_PASSWORD='请替换为强密码'

# 限流（建议开启）
export WEB_RATE_LIMIT_ENABLED=true
export WEB_RATE_LIMIT_REQUESTS=6
export WEB_RATE_LIMIT_WINDOW_SEC=60
```

## 5.2 启动应用 + 启动 tunnel（最小可复制版）

```bash
# 1) 启动应用（按你的项目实际命令替换）
nohup python -m uvicorn <module>:<app_object> --host 127.0.0.1 --port 18100 \
  > ./web_runtime/app.log 2>&1 &
echo $! > ./web_runtime/app.pid

# 2) 启动 cloudflared
nohup cloudflared tunnel --url http://127.0.0.1:18100 \
  > ./web_runtime/cloudflared.log 2>&1 &
echo $! > ./web_runtime/cloudflared.pid

# 3) 读取公网地址
grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' ./web_runtime/cloudflared.log | tail -1
```

拿到 URL 后，可在手机 4G/5G 网络直接测试访问。

## 6. 一键清理（建议固化脚本）

核心动作：
1. 杀应用进程（PID 文件）
2. 杀 cloudflared 进程（PID 文件）
3. 清理 PID 残留
4. 可选保留日志

最小命令：

```bash
kill "$(cat ./web_runtime/cloudflared.pid)" 2>/dev/null || true
kill "$(cat ./web_runtime/app.pid)" 2>/dev/null || true
rm -f ./web_runtime/cloudflared.pid ./web_runtime/app.pid
```

## 7. 最小安全基线（长期开放必做）

1. Basic Auth
- 建议至少要求用户名/密码

2. 密码不要长期明文
- 推荐存 SHA256（由应用侧比对哈希）

示例：

```bash
WEB_PASS='你的强密码'
WEB_PASS_SHA256="$(python - <<'PY' "$WEB_PASS"
import hashlib, sys
print(hashlib.sha256(sys.argv[1].encode()).hexdigest())
PY
)"
unset WEB_PASS
```

3. 限流
- 对高成本接口（如预测、生成）做 IP 级限流

4. 监听地址
- 建议应用仅监听 `127.0.0.1`
- 对公网统一走 tunnel，不直接开放 `0.0.0.0`

## 8. 常见问题排查

### 8.1 `No module named uvicorn`

```bash
python -m pip install uvicorn
```

### 8.2 进程在但 `curl` 失败

先排查代理变量：

```bash
env | grep -i proxy
curl --noproxy '*' -i http://127.0.0.1:18100/
```

### 8.3 `ExecutableNotFound: dot`

说明你开启了可视化功能，但系统缺少 Graphviz 可执行程序 `dot`。
- 临时：关闭可视化参数
- 需要可视化：安装 Graphviz binary（不仅是 Python graphviz 包）

### 8.4 Tunnel 地址不通或失效

Quick Tunnel 可能因网络抖动断开：
- 重启 `cloudflared`
- 重新读取新 URL

### 8.5 本地健康但外网访问不了

重点检查：
- cloudflared 是否仍在运行
- 是否拿到了新的 `trycloudflare.com` URL
- 应用是否误退出（看 `app.log`）

## 9. 迁移到新项目时只改三处

1. 应用启动命令
- 例如从 `uvicorn` 换成 `gunicorn` / `node` / `java -jar`

2. 健康检查 URL
- 根路径 `/` 或 `/health`

3. 应用配置变量
- 鉴权、限流、模型路径等变量名改为新项目约定

## 10. 建议的脚本结构（可复制）

```text
web_scripts/
  start_web.sh   # 启动应用 + tunnel + 输出公网 URL
  clean_web.sh   # 杀进程 + 清理 pid
  runtime/
    app.pid
    cloudflared.pid
    app.log
    cloudflared.log
```

建议 `start_web.sh` 具备：
- 启动前先执行 `clean_web.sh`
- readiness 检查（本地 HTTP 状态）
- 自动解析并打印公网 URL

## 11. 日常操作模板

启动：

```bash
cd <your_project_root>
conda activate <your_env>
bash web_scripts/start_web.sh
```

停止：

```bash
cd <your_project_root>
bash web_scripts/clean_web.sh
```

---

一句话总结：
**无 sudo 场景下，最实用的通用公网方案是“本地服务 + cloudflared tunnel + 应用层认证限流 + 一键清理脚本”。**
