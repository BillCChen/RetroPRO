# RetroPRO

基于 Retro* 思路的逆合成规划项目代码。

## 1. 环境搭建

### 1) 克隆仓库

```bash
git clone <your-repo-url> RetroPRO
cd RetroPRO
```

### 2) 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate retro_star_env
```

## 2. 数据准备

下载并解压以下资源（building block molecules、pretrained models、可选测试数据）：

- [retro_data.zip](https://www.dropbox.com/s/ar9cupb18hv96gj/retro_data.zip?dl=0)

将解压后的 `dataset/`、`one_step_model/`、`saved_models/` 放到 `retro_star/` 目录下。

## 3. 安装依赖与库

```bash
pip install -e retro_star/packages/mlp_retrosyn
pip install -e retro_star/packages/rdchiral
pip install -e .
```

## 4. 运行与训练

### 1) 运行规划

```bash
cd retro_star
python retro_plan.py --use_value_fn
```

不使用 value function 时，移除 `--use_value_fn` 参数。

### 2) 训练 value function

```bash
python train.py
```

## 5. 示例代码

参考 `example.py`：

```python
from retro_star.api import RSPlanner

planner = RSPlanner(
    gpu=-1,
    use_value_fn=True,
    iterations=100,
    expansion_topk=50
)

result = planner.plan('CCCC[C@@H](C(=O)N1CCC[C@H]1C(=O)O)[C@@H](F)C(=O)OC')
```
