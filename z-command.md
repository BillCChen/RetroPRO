# Retro 规划命令与参数说明

本文档说明如何使用 [`retro_star/retro_plan.py`](retro_star/retro_plan.py) 做逆合成路线规划，重点包括 **template-free + DICT** 时的环境变量、输出文件位置及常用示例。

## 运行前准备

1. **工作目录**：在仓库中进入 `retro_star` 子目录（与 `retro_plan.py` 同级），所有相对路径（数据集、`one_step_model/` 等）均相对该目录。
2. **Conda 环境**：示例使用 `unirxn`；若路径不同，请自行修改 `conda activate` 与 `conda.sh` 位置。
3. **GPU**：`--gpu 0` 会设置 `CUDA_VISIBLE_DEVICES`（见 [`parse_args.py`](retro_star/common/parse_args.py)）。

```bash
cd /path/to/retro_star/retro_star
source "$HOME/miniconda3/etc/profile.d/conda.sh"   # 或 anaconda3
conda activate unirxn
```

---

## 输出目录与主要文件

| 机制 | 说明 |
|------|------|
| `--result_folder` | 未指定时自动生成，形如 `results/<test_routes>/<test_routes>_plan_<one_step_type>_iter<iterations>_topk<expansion_topk>_<时间戳>/`。 |
| `log.txt` | 与 `plan.pkl` 同目录。 |
| `plan.pkl` | 规划结果；在 `template_free` 且 `--DICT` 时含 `dict_cache_report`、`inference_run_params` 等。 |
| `args.yaml` | 本次运行完整 argparse 快照。 |
| DICT 规则快照 | 默认与 `plan.pkl` **同目录**（见下文「DICT 与环境变量」）。 |

---

## 常用命令行参数（节选）

更全列表见 [`retro_star/common/parse_args.py`](retro_star/common/parse_args.py)。

| 参数 | 含义 |
|------|------|
| `--seed` | 随机种子。 |
| `--use_value_fn` | 启用价值网络。 |
| `--viz` | 可视化（需有效 `--viz_dir`，默认在 `result_folder/viz`）。 |
| `--gpu` | 单卡 GPU 编号；与 `--gpu_list` 互斥场景见脚本内逻辑。 |
| `--expansion_topk` | 每步扩展保留的候选数（规划器传给一步模型）。 |
| `--iterations` | 每个目标分子最大搜索迭代步数。 |
| `--one_step_type` | `template_free`（R-SMILES）或 `template_based`（MLP）。 |
| `--CSS` | template-free 下启用子结构采样相关路径（与 `--RD_list` 配合）。 |
| `--RD_list` | 字符串形式，如 `"[(7,2)]"`，表示子结构采样半径等（会 `literal_eval`）。 |
| `--DICT` | 启用反应字典缓存（template-free）。 |
| `--test_routes` | 数据集键名或路径，如 `pth_hard`、`uspto190`，或自定义 `.pkl`/文本。 |
| `--route_limit` | **大于 0** 时只取 `test_routes` 解析结果中的前 N 个目标（例如 `pth_hard` 共 100 个时先跑 10 个试跑）。 |
| `--multi_pool` | 多目标并行池调度（与串行「legacy serial」相对）。 |
| `--parallel_num` | 并行池宽度（与 `multi_pool` 等配合）。 |
| `--result_folder` / `--viz_dir` | 显式指定输出目录（可选）。 |

**template-free 专用（一步模型权重）**

| 参数 | 含义 |
|------|------|
| `--retro_model_path` | 逆合成（P→R）模型。 |
| `--forward_model_path` | 正向（R→P）模型。 |
| `--retro_topk` | 一步逆合成 beam/条数上限相关。 |
| `--forward_topk` | 一步正向 top-k。 |

---

## DICT 与 `TP_FREE_*` 环境变量

未设置 **`TP_FREE_DICT_DUMP_DIR`** 时，程序会将周期/默认落盘目录设为 **本次 `result_folder` 的绝对路径**，与 `log.txt`、`plan.pkl` 同级。

**为何看不到 `tp_free_DICT_*.pkl`？**

- **`tp_free_DICT_final_*.pkl`**：规划全部结束后写入（默认开启，见上表 `TP_FREE_DICT_DUMP_ON_EXIT`）。请在本次运行的 **`result_folder`** 下查找（形如 `results/pth_hard/pth_hard_plan_..._0420_1534/`），与 `plan.pkl` 同级。
- **`tp_free_DICT_20xx_...pkl`（文件名里带时间、非 `final`）**：仅在 **`TP_FREE_DICT_DUMP_EVERY` 设为大于 0** 时，按「字典更新次数」周期性写出；默认 **`TP_FREE_DICT_DUMP_EVERY=0`**，因此**不会**产生这类文件。
- **统计与参数**：即使不保存上述 pkl，**`plan.pkl` 里的 `dict_cache_report` / `inference_run_params`** 已包含 DICT 分析所需信息。

| 环境变量 | 说明 |
|----------|------|
| `TP_FREE_DICT_DUMP_DIR` | 周期写入 `tp_free_DICT_*.pkl` 的目录；不设置则等于本次 `result_folder`。 |
| `TP_FREE_DICT_DUMP_EVERY` | 每累计多少次字典**更新**写一次周期文件；`0` 表示不按更新次数写周期文件（默认 `0`）。 |
| `TP_FREE_DICT_DUMP_WITH_META` | 设为 `1` 时，周期/快照 pickle 可为 `{"rules": ..., "stats": ...}`，否则主要为规则 dict。 |
| `TP_FREE_DICT_DUMP_ON_EXIT` | **默认 `1`**：规划结束写 **`tp_free_DICT_final_<时间戳>.pkl`**（与 `plan.pkl` 同目录）。设为 `0` 可关闭。 |
| `TP_FREE_DICT_STATS_TOPK` | `get_dict_cache_report()` 中命中分布 Top-K 条目数（默认如 `500`）。 |
| `TP_FREE_RETRO_BATCH_SIZE` / `TP_FREE_FORWARD_BATCH_SIZE` / `TP_FREE_MAPPER_BATCH_SIZE` | 一步模型与 mapper 的 batch 大小。 |

实现见 [`tp_free_inference.py`](retro_star/packages/mlp_retrosyn/mlp_retrosyn/tp_free_inference.py) 与 [`retro_plan.py`](retro_star/retro_plan.py) 中的 `_ensure_tp_free_dict_dump_dir`、`_dump_tp_free_dict_on_exit` 等。

---

## 推荐示例：pth_hard + template-free + DICT（先跑 10 个分子）

与 `plan.pkl` 同目录写出 DICT 快照时，可显式打开结束落盘：

```bash
export TP_FREE_DICT_DUMP_ON_EXIT=1   # 可选，脚本里常默认已开启

python retro_plan.py --seed 42 --use_value_fn --viz --gpu 0 \
  --expansion_topk 8 --iterations 51 \
  --one_step_type template_free --CSS --RD_list "[(7,2)]" --DICT \
  --test_routes pth_hard \
  --route_limit 10
```

- **`--route_limit 10`**：只规划前 10 个目标；全量 100 个时去掉该参数或改为 `100`。
- 结果目录由 `parse_args` 自动生成，内含 `plan.pkl`、`log.txt`、`args.yaml` 及可选 `tp_free_DICT_final_*.pkl`。

---

## 其它示例

**Template-based（MLP 模板）**

```bash
python retro_plan.py --seed 42 --use_value_fn \
  --expansion_topk 8 --iterations 101 \
  --one_step_type template_based --viz --gpu 0 \
  --test_routes uspto190
```

**Template-free，无 CSS/RD/DICT（基线）**

```bash
python retro_plan.py --seed 42 --use_value_fn --viz --gpu 0 \
  --expansion_topk 8 --iterations 101 \
  --one_step_type template_free \
  --test_routes pth_hard
```

**批量脚本（项目内）**

```bash
bash scripts/run_template_free_batch_impact.sh --gpu 0 --seed 42 --iterations 10 \
  --expansion-topk 8 --max-targets 10 --test-routes uspto190 \
  --parallel-num 4 --repeats 1 --rd-list "[(7,2),(3,0)]"
```

---

## 结果分析

- 使用仓库根目录 [`DICT_cache_analysis.ipynb`](DICT_cache_analysis.ipynb) 读取 `plan.pkl` 中的 `dict_cache_report` 与 `inference_run_params`（与本次 `RD_list`、topk、`test_routes` 等对应）。

---

## 一键脚本（可选）

若需保留可执行入口，可使用同目录下的 [`z-command.sh`](z-command.sh)（内容与上表一致，仅封装 `cd` 与 `conda activate`）。**以本文档为准**；脚本变更时请同步更新本文件。
