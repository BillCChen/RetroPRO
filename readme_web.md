# RetroPRO Web 现状说明（readme_web）

本文档基于当前仓库代码梳理 `RetroPRO` 的可视化前端网站实现情况，重点说明：

1. 项目里已经有哪些 Web 端能力
2. 前后端对应关系
3. 如何启动
4. 当前已知问题与改进建议

## 1. 项目当前的 Web 架构概览

当前仓库中实际上有 **两套并行的 Web 方案**：

1. **方案 A：单次预测 + 结果可视化（同步接口）**
   - 前端页面：`retro_star/static/index.html`
   - 后端服务：`retro_star/main.py`
   - 特点：直接调用预测接口，返回 task_id 后读取结果，可下载 JSON/HTML 路径图。

2. **方案 B：任务队列式提交（异步任务 + worker）**
   - 前端页面：`retro_star/frontend/index.html`
   - 后端服务：`retro_star/backend/main.py`
   - 任务执行器：`retro_star/backend/worker.py`
   - 特点：前端提交任务到数据库，worker 轮询 pending 任务并调用 `run.sh` 计算。

> 结论：仓库里不是“一个前端 + 一个后端”，而是两套页面/接口并存。

---

## 2. 目录与文件对应关系

### 2.1 前端页面

- `retro_star/static/index.html`
  - 与 `retro_star/main.py` 的 API 对齐
  - 主要用于“直接预测 + 展示结果 + 路径图下载/内嵌”

- `retro_star/static/index_copy.html`
  - `static/index.html` 的副本版本（内容高度相似）

- `retro_star/frontend/index.html`
  - 与 `retro_star/backend/main.py` 的 API 对齐
  - 主要用于“任务提交/轮询状态/结果下载”

### 2.2 后端服务

- `retro_star/main.py`
  - FastAPI
  - 面向 `static/index.html`
  - 提供 `/api/predict`、`/api/result/{task_id}`、`/api/download_html/{task_id}`、`/api/preview-smiles` 等

- `retro_star/backend/main.py`
  - FastAPI
  - 面向 `frontend/index.html`
  - 提供 `/api/submit`、`/api/status/{task_id}`、`/api/download/{task_id}` 等

- `retro_star/backend/worker.py`
  - 后台 worker，轮询数据库中的 `pending` 任务
  - 调用 `retro_star/run.sh` 执行计算并回写任务状态

### 2.3 数据与运行产物

- `retro_star/database/tasks.db`（运行时生成）
- `retro_star/database/tasks.csv`（运行时生成）
- `retro_star/results/`（结果与映射文件）
- `retro_star/logs/worker.log`（worker 日志，运行时生成）

---

## 3. 已有页面功能（按方案）

## 3.1 方案 A（`static/index.html` + `main.py`）

### 已实现

1. SMILES 输入与参数配置
   - iterations / expansion_topk / use_value_fn / one_step_type / CCS / radius / file_prefix
2. 分子预览
   - 调用 `/api/preview-smiles` 返回 base64 PNG 预览
3. 提交预测
   - 调用 `/api/predict`
4. 结果读取
   - 调用 `/api/result/{task_id}`
5. 结果下载
   - JSON 下载：`/api/download/{task_id}/json`
   - HTML 路径图：`/api/download_html/{task_id}`
6. 路径图内嵌
   - 前端通过 iframe `srcdoc` 内嵌 HTML 可视化

### 补充说明

- `main.py` 中含有路径图 HTML 生成逻辑（把反应字符串转为图并生成可交互页面）。

## 3.2 方案 B（`frontend/index.html` + `backend/main.py` + `worker.py`）

### 已实现

1. 任务提交表单（参数更全，含用户信息）
2. 任务提交接口 `/api/submit`
3. 任务状态轮询 `/api/status/{task_id}`
4. 已完成任务下载 `/api/download/{task_id}?file_type=json`
5. SQLite + CSV 双写存储任务参数/状态
6. worker 后台轮询并执行 `run.sh`

### 补充说明

- `backend/main.py` 负责“排队与状态管理”。
- 真正计算由 `worker.py -> run.sh` 触发。
- 当前 `run.sh` 是占位脚本，输出的是模拟结果 JSON。

---

## 4. API 对照（两套不可混用）

### 4.1 方案 A API（`retro_star/main.py`）

- `POST /api/predict`
- `GET /api/result/{task_id}`
- `GET /api/download_html/{task_id}`
- `GET /api/download/{task_id}/{format_type}`
- `POST /api/preview-smiles`
- `GET /`（返回 `static/index.html`）

### 4.2 方案 B API（`retro_star/backend/main.py`）

- `POST /api/submit`
- `GET /api/status/{task_id}`
- `GET /api/download/{task_id}?file_type=json`
- `GET /api/statistics`
- `GET /api/health`
- `GET /`（返回 `frontend/index.html`）

---

## 5. 快速启动方式

## 5.1 启动方案 A（直接预测页面）

```bash
cd retro_star
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

打开：`http://127.0.0.1:8000`

## 5.2 启动方案 B（任务队列页面）

终端 1：

```bash
cd retro_star
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

终端 2：

```bash
cd retro_star
python backend/worker.py
```

打开：`http://127.0.0.1:8000`

---

## 6. 当前代码状态与已知问题

1. **双方案并存但未统一**
   - `static/index.html` 与 `frontend/index.html` 使用不同接口体系。

2. **`main.py` 存在硬编码数据路径**
   - `prepare_starting_molecules('/home/chenqixuan/retro_star/retro_star/dataset/origin_dict.csv')`
   - 在非该机器路径下可能导致启动失败。

3. **方案 B 的计算脚本是占位实现**
   - `retro_star/run.sh` 目前写明“模拟计算结果”，并非真实逆合成主流程。

4. **方案 A 前端存在部分结果字段使用不一致风险**
   - 页面中对 `data.route` 与 `data.result` 的使用存在混合写法，需按真实返回结构再统一。

5. **缺少专门的 Web 文档与启动脚本编排**
   - 当前 README 主要介绍算法训练/推理，未系统覆盖 Web 双方案。

---

## 7. 建议的下一步（面向收敛）

1. 先选定一个主方案（建议二选一）：
   - 要“在线任务平台形态”就以 `backend/main.py + worker.py + frontend/index.html` 为主；
   - 要“研究演示快速预测形态”就以 `main.py + static/index.html` 为主。

2. 移除硬编码路径，统一为配置项或相对路径。

3. 统一返回数据结构与前端渲染字段（尤其 route/result 字段）。

4. 增加 Web 启动说明（可在主 README 增加“Web 模式”章节）。

5. 若采用任务队列方案，将 `run.sh` 替换为真实计算入口（或直接调用 Python 主流程）。

---

## 8. 一句话总结

当前 `RetroPRO` 的可视化网站部分已经完成了两套可运行雏形（同步预测页 + 异步任务页），但仍处在“并行演进”状态，尚未收敛为统一、生产可维护的一套 Web 方案。
