# Multimodal Recommender

[![CI](https://github.com/LazzyXP/Multimodal-Recommender-System/actions/workflows/ci.yml/badge.svg)](https://github.com/LazzyXP/Multimodal-Recommender-System/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/multimodal-recommender)](https://pypi.org/project/multimodal-recommender/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**用统一 API 完成推荐模型训练、验证选模、排名融合和批量推荐。**

面向交互数据与物品多模态特征的 AutoML 推荐工具，工作流受 AutoGluon 启发。
支持从简单基线开始，再按数据条件加入内容模型和可选 PyTorch 图模型。

[快速开始](#快速开始) · [使用指南](docs/usage.md) · [模型目录](docs/model-catalog.md) · [安装说明](docs/installation.md) · [能力审计](docs/AUDIT.md)

> **项目阶段：实验性。** 已实现训练与部署 API，但尚未完成生产规模验收或论文指标复现。
> 本 README 的功能说明对应源码开发版本 **0.1.2**；PyPI 已发布版本请以包页面为准。

## 能做什么

| 环节 | 已实现能力 |
| :--- | :--- |
| 数据接入 | DataFrame、CSV、TSV、TXT、JSONL、Parquet；文本转 Parquet 与缓存复用 |
| 模型训练 | 交互基线、物品多模态模型、可选图模型；统一 `fit()` 与模型注册 |
| 自动选择 | 验证集主指标选模、轻量 HPO、学习 RRF 融合权重、最佳模型重训 |
| 评测报告 | Recall、NDCG、MRR 等排名指标、覆盖率、leaderboard 与报告导出 |
| 批量部署 | 默认最佳模型推荐、分批输出 Parquet、保存加载、历史索引打包 |

## 安装

Python **3.11+**。安装已发布包：

```bash
python -m pip install multimodal-recommender
```

要使用本文的最新开发接口，从源码安装：

```bash
python -m pip install "git+https://github.com/LazzyXP/Multimodal-Recommender-System.git@main"
# 可选图模型运行时
python -m pip install "multimodal-recommender[torch] @ git+https://github.com/LazzyXP/Multimodal-Recommender-System.git@main"
```

复现实验时，将 `@main` 换成具体提交 SHA，或固定 PyPI 版本号。
CPU、CUDA、MPS 和镜像配置见 [安装说明](docs/installation.md)。

## 快速开始

下面的例子自带数据，可直接运行，无需下载数据集：

```python
import pandas as pd
from mmrec import MultiModalRecommender

interactions = pd.DataFrame({
    "user_id": ["u1"] * 4 + ["u2"] * 4 + ["u3"] * 4,
    "item_id": ["a", "b", "c", "d", "b", "c", "e", "f", "a", "d", "e", "f"],
    "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
})

recommender = MultiModalRecommender(eval_metric="ndcg@3")
recommender.fit(interactions, models=["Popularity", "ItemCF"])

print(recommender.leaderboard())
print(recommender.recommend(users=["u1"], k=2).data)

recommender.save("artifacts/demo")
restored = MultiModalRecommender.load("artifacts/demo")
```

`recommend()` 默认使用验证集选出的最佳模型；传 `models="all"` 可对比各候选与融合结果。
推荐结果包含 `user_id`、`item_id`、`rank`、`score` 和 `model`。
不同模型的 score 不保证同尺度，融合使用排名。

## 加入物品多模态特征

在上例基础上提供物品表与模态声明：

```python
items = pd.DataFrame({
    "item_id": ["a", "b", "c", "d", "e", "f"],
    "title": ["运动跑鞋", "轻量跑鞋", "登山背包", "旅行背包", "无线耳机", "运动耳机"],
    "category": ["鞋", "鞋", "包", "包", "耳机", "耳机"],
})

recommender.fit(
    interactions,
    items=items,
    modalities={"item": {"text": ["title"], "categorical": ["category"]}},
    models=["Popularity", "ItemCF", "MultiModalItemKNN"],
)
print(recommender.recommend(users=["u1"], k=2).data)
```

文本使用轻量哈希编码；图片需提供预计算数值向量。
`models="auto"` 根据 preset、物品模态与可选运行时选择候选。

| 模型族 | 模型 | 额外依赖 |
| :--- | :--- | :--- |
| 交互基线 | Popularity、ItemCF、BPRMF | 无 |
| 多模态物品 | MultiModalItemKNN、MultiModalLateFusionKNN、VBPR | FAISS 可选 |
| 图模型 | LightGCN、MMGCN、LATTICE、BM3、FREEDOM、MGCN、DRAGON、LGMRec | PyTorch |
| 融合 | RankFusion | 无 |

图模型是论文机制的简化实现，具体差异与来源见 [模型目录](docs/model-catalog.md)。

## 使用前需要了解

- **评测**：验证集参与选模，外部测试集不参与重训；候选分数不代表重训后模型的实测分数。
- **融合**：权重学习与融合选择共享一次验证集，尚未实现多折 OOF/bagging。
- **规模**：流式交互入口会采样训练部分模型；物品特征仍整体驻留内存，不能据此宣称全量大规模训练。
- **冷启动**：`MultiModalItemKNN` 可用与物品同维度的用户 embedding 为无历史用户生成内容推荐；其他模型仍主要依赖交互历史。
- **检索**：FAISS 默认使用 `IndexFlatIP` 精确索引，也可通过 `ann_backend="hnsw"` 启用 HNSW 近似检索；未安装 FAISS 时自动回退 NumPy。
- **模型文件**：仅加载可信来源的 pickle。SHA-256 用于发现意外损坏，不提供来源认证；保存使用 generation 目录和原子 `.CURRENT` 指针，并保留旧版本回退。长期部署还应配置 generation 清理策略。

详细参数、时间预算、切分方式、大数据输出和完整配置见 [使用指南](docs/usage.md)。
仍需完成的工程与实验验证见 [能力审计](docs/AUDIT.md)。

## 开发与测试

```bash
git clone https://github.com/LazzyXP/Multimodal-Recommender-System.git
cd Multimodal-Recommender-System
uv sync --group dev --frozen
uv run pytest
uv run ruff check .
uv build
```

`benchmarks/` 用于内部测量方法效果、训练耗时与资源占用，不作为独立发布产品。默认生成合成数据，
也可对真实交互表运行：

```bash
python benchmarks/benchmark.py --input data/interactions.parquet --seeds 7,8,9
```

输出 JSON 包含每个 seed 的数据规模、模型 leaderboard、吞吐、峰值 RSS、Python 和平台信息，
以及多 seed 的均值/标准差，便于固定环境对比。
发布包的构建、安装与验证流程见 [发布说明](docs/releasing.md)。

发现问题可提交 [Issue](https://github.com/LazzyXP/Multimodal-Recommender-System/issues)，
附上版本、运行环境、最小数据示例和错误日志，便于复现。
