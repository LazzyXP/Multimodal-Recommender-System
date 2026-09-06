# Multimodal Recommender

一个使用 `uv` 管理、采用 AutoML 工作流的多模态推荐框架。用户提供交互数据、用户特征、
物品特征和少量配置，框架在统一协议下训练多个模型，并输出每个模型及融合模型的推荐结果和评测报告。

当前版本是一个可运行的 MVP，包含：

- CSV、TSV、TXT、JSONL、Parquet 和 pandas DataFrame 输入
- 文本数据流式转换为 ZSTD Parquet，并按源文件指纹复用缓存
- 用户级 leave-one-out 时间切分
- `Popularity`、`ItemCF`、`BPRMF`、早期/晚期多模态 KNN 与 `VBPR`
- 可选 PyTorch 图模型：`LightGCN`、`MMGCN`、`LATTICE`、`BM3`、`FREEDOM`、`MGCN`、`DRAGON`、`LGMRec`（各自实现论文核心机制，见 [model catalog](docs/model-catalog.md)）
- 学到的堆叠集成：非负逻辑回归 meta-learner 在 holdout 上学出融合权重，输出 `RankFusion`
- Recall、NDCG、MRR、Hit Rate、MAP、Precision、AUC 与 catalog coverage
- 多模型推荐结果、leaderboard、报告导出和模型保存加载
- 自定义模型注册接口
- 文本、类别、数值与预计算图片 embedding 的多模态物品编码
- 用户/物品多模态 Schema 校验与持久化

`Popularity` 和 `ItemCF` 只消费交互数据；`MultiModalItemKNN` 会把用户历史物品的多模态向量
聚合成用户画像，并支持尚未产生交互但已经存在特征的冷启动物品。更重的编码器和 Two Tower
模型会复用相同的 API 与评测协议。

`models="auto"` 默认运行多个互补候选。`medium_quality` 在具备两种以上物品模态时会自动运行
`Popularity + ItemCF + BPRMF + MultiModalItemKNN + MultiModalLateFusionKNN + VBPR`。安装
`[torch]` extra 后，自动模式还会加入 `LightGCN`、`MMGCN`、`LATTICE`、`BM3`、`FREEDOM`、
`MGCN`、`DRAGON` 和 `LGMRec`，最后生成 `RankFusion`。完整模型分层和论文来源见
[model catalog](docs/model-catalog.md)；对标 AutoGluon 的验收自审见
[audit](docs/AUDIT.md)。

## 安装

**发布状态：尚未发布到 PyPI。** 当前不能使用 `pip install multimodal-recommender`
从 PyPI 安装；此前文档把计划中的发布方式写成已可用，这是错误的。
目前需要 Python 3.11+ 和 Git，可直接从 GitHub 安装：

```bash
python -m pip install "git+https://github.com/LazzyXP/Multimodal-Recommender-System.git@main"
# 可选 PyTorch 图模型
python -m pip install "multimodal-recommender[torch] @ git+https://github.com/LazzyXP/Multimodal-Recommender-System.git@main"
```

`main` 会更新；需要复现时，将 `@main` 换成具体提交 SHA。
也可以克隆后安装：

```bash
git clone https://github.com/LazzyXP/Multimodal-Recommender-System.git
cd Multimodal-Recommender-System
python -m pip install .
# 开发环境（需先安装 uv）
uv sync --group dev
```

首次 PyPI 发布还需要维护者配置账号和 Trusted Publisher，具体步骤见
[发布说明](docs/releasing.md)。GitHub Release 或构建产物不等于已经发布到 PyPI。

The core wheel has no platform-specific native dependency beyond NumPy, pandas and
PyArrow. The optional graph layer uses the official PyTorch wheel for the current
Python/OS/accelerator combination; CPU works on macOS, Linux and Windows, while
CUDA and Apple MPS are selected automatically when available. Set `device="cpu"`,
`device="cuda"` or `device="mps"` on an explicit graph model to override selection.
The torch extra is constrained below 2.7 for CUDA 12.x hosts; newer torch wheels may
require CUDA 13 and report CUDA as unavailable on older NVIDIA drivers.
For a complete platform and China-mirror installation matrix, see
[installation.md](docs/installation.md). CUDA is for NVIDIA Linux/Windows hosts;
macOS uses MPS or CPU.

## 数据导入与转换

CSV、TSV、TXT 和 JSONL 只是导入格式。框架不会直接用这些文本文件训练，而是先通过 PyArrow
分批读取并转换为带字典编码和统计信息的 ZSTD Parquet。默认缓存目录是 `.mmrec/cache`，缓存键
包含源文件路径、大小和修改时间，因此未发生变化的数据不会重复转换。

可以提前显式转换：

```bash
mmrec convert data/interactions.csv data/interactions.parquet
mmrec convert data/interactions.txt data/interactions.parquet --delimiter $'\t'
```

也可以在 Python 中转换：

```python
from mmrec import convert_to_parquet

path = convert_to_parquet("data/interactions.csv", compression="zstd")
```

## 大数据执行模式

`execution_mode="auto"` 会根据 Parquet 行数自动选择内存或流式执行。默认超过 200 万条交互时：

- 使用 Arrow Dataset 分批扫描，只读取用户、物品、时间和标签列
- 扫描全量数据并在 catalog 预算内统计物品频次；`Popularity` 被选中重训时使用该统计
- `ItemCF` 和 `MultiModalItemKNN` 使用按用户稳定哈希得到的有界样本
- `fit_summary()` 的 `training_scope` 明确标识 `full` 或 `sampled`
- 推荐时从原始 Parquet 精确读取当前用户历史，排除已交互物品

```python
recommender = MultiModalRecommender(
    execution_mode="auto",
    max_in_memory_interactions=2_000_000,
    sample_interactions=1_000_000,
    scan_batch_size=262_144,
    inference_batch_size=10_000,
    max_inference_score_mb=64,
    history_partitions=256,
    max_catalog_items=2_000_000,
    max_sample_history_per_user=200,
)
recommender.fit("data/interactions.parquet", models="auto")
```

大规模离线推荐应直接分批写 Parquet，避免在内存中构造完整结果表。用户参数可以是 ID 列表、
DataFrame、CSV 或 Parquet 用户表：

```python
recommender.recommend_to_parquet(
    users="data/target_users.parquet",
    path="artifacts/recommendations.parquet",
    k=100,
    batch_size=10_000,
)
```

流式模型默认引用缓存中的历史分桶索引，以避免复制大型数据。需要把模型部署到另一台机器时，
显式把索引打进模型目录：

```python
recommender.save("artifacts/model", include_history_index=True)
deployed = MultiModalRecommender.load("artifacts/model")
```

当前流式模式保证数据扫描和结果输出内存有界。ItemCF 仍是样本模型，不适合对全量高活跃用户
构建无界共现矩阵；生产级全量召回应使用后续的分批 Two Tower 与 FAISS/HNSW 索引。
当物品基数超过 `max_catalog_items` 时，框架保留重频候选集并在数据摘要中设置
`catalog_truncated=True`，避免 catalog 状态无限增长。

## 自动选模与部署

默认通过验证集主指标选择最佳单模型或 `RankFusion`，保存在 `model_best`。
`recommend()`、`predict()` 和 `recommend_to_parquet()` 默认使用最佳模型；
需要比较所有候选时显式传入 `models="all"`（这是相对于早期 MVP 的默认行为变化）。

```python
recommender = MultiModalRecommender(eval_metric="ndcg@20")
recommender.fit(
    interactions=train,
    validation_data=validation,
    test_data=test,
    items=items,
    time_limit=3600,
    refit_full="best",
)
print(recommender.model_best)
print(recommender.leaderboard())  # score_val 用于选模，is_best 标记选中的模型
recommendations = recommender.recommend(users=["u1"], k=20)
recommender.save("artifacts/best", best_only=True)
```

训练分两个阶段：先完成候选训练、验证与评测，再对最佳模型重训；如果融合胜出，
重训其全部基础模型。`refit_full=False` 禁用重训，`refit_full="all"` 重训所有成功候选。
`fit_summary().models` 记录候选训练时间、`chosen_config`、`refit_status` 和重训耗时。
重训失败或预算不足时保留原候选；融合依赖未全部重训成功时整体保留原版本。
checkpoint 分别保存候选和已完成的重训阶段，可在追加预算后复用。

显式 `validation_data` 用于 HPO、融合权重与模型选择；显式 `test_data` 只用于评测，
不会进入训练或全量重训。只传验证集而不传测试集时，leaderboard 是验证集结果，
报告的 `score_split` 会明确标记。完全不传时，仍自动进行 train/val/test 三路划分。
候选验证和测试指标均来自仅用 train 训练的版本，**不是重训后模型的实测分数**。
未传外部测试集时，全量重训会使用全部输入交互（包含内部划出的 holdout）；
传入外部验证集时还会合并该验证集，但始终排除外部测试集。

`split="global_temporal"` 按全局时间窗口留出末尾约 20% 的事件，再从剩余部分留出
验证窗口；相同时间戳不会跨分区。显式分区使用该选项时，要求 train、validation、test
时间严格递增。原 `temporal` 仍表示按用户 leave-one-out，不保证全局时间隔离。
显式分区中重复的交互事件会被拒绝；无时间戳时以用户—物品对判定重叠。
历史物品默认从推荐结果排除，重复消费场景需自行设计与该策略一致的评测数据。
流式模式的自动划分和候选评测仍基于用户样本，并非全流时间窗口评测。

可用 `set_model_best("ItemCF")` 手动指定已训练模型。`save(best_only=True)` 仅保存选中的
模型；选中融合时保留全部依赖。`predict()` 对融合模型返回基于每个用户传入候选集合
计算的 RRF 分数，分数随候选集合变化，不是概率。

推荐阶段默认把每个模型的临时 score matrix 限制在 64 MiB；可通过
`max_inference_score_mb` 调整。降低该值可避免大 catalog 或 GPU 参数占用导致 OOM，
代价是更小的用户批次和较低吞吐。

当前融合权重和融合模型选择共享单次验证集，验证分数可能乐观；独立测试集只做最终报告。
验证集没有可用分数时发出 warning，并回退到第一个成功候选，不会使用测试分数选模。

## 训练预算、并行与数据划分

`fit()` 支持 AutoGluon 风格的训练控制：

```python
recommender.fit(
    interactions,
    models="auto",
    presets="best_quality",  # 决定候选集与每个模型的训练强度
    time_limit=3600,         # 总 wall-clock 预算（秒）
    num_workers=4,           # 并行训练非 torch 模型
    split="temporal",        # temporal / random / cold_start / global_temporal
    hyperparameter_tune=True,  # TPE 式采样 + successive-halving 超参搜索
    n_trials=8,
    checkpoint_dir="artifacts/checkpoint",  # 断点续训
)
```

- `time_limit`：从数据准备开始计时的总 wall-clock 预算。内置可训练模型会在每个训练阶段使用
  剩余预算并在边界优雅早停；预算耗尽后剩余模型标记为 `skipped`，已成功的模型仍会生成报告。
  任意自定义模型的硬中断仍由模型工厂自行负责。
- `num_workers`：用线程池并行训练非 torch 模型；torch 图模型始终串行训练，避免在 CPU/CUDA/MPS
  上并发导致的内存超限与稳定性问题（会发出 warning）。
- 图模型使用 `device="auto"` 时遇到 CUDA out-of-memory 会清理显存并重试 CPU；显式
  `device="cuda"` 不会静默降级，失败会保留在 fit summary 中。
- `split`：`temporal`（按时间 leave-one-out，默认）、`random`（随机 leave-one-out）、
  `cold_start`（随机保留一部分用户整体作为冷启动测试集）。
- `presets` 现在真正影响训练强度：`fast_training < medium_quality < best_quality` 逐级提高
  `BPRMF`/`VBPR` 的 epoch、因子维度和训练样本上限。
- `hyperparameter_tune`：对 `BPRMF`/`VBPR`/Torch 图模型做「简化 TPE 采样 + successive-halving」
  搜索——先用 2 个 epoch 粗筛保留较优的一半，再对幸存配置跑满 epoch 精筛。
- `checkpoint_dir`：每个模型完成后落盘 `checkpoint.pkl`；包含交互、用户和物品表的内容指纹，
  用相同数据和参数重跑时自动跳过已完成的候选与重训阶段，只补训剩余（尤其适合配合 `time_limit` 加预算续跑）。

评测指标扩展为 `recall@k`、`ndcg@k`、`mrr@k`、`hit_rate@k`、`map@k`、`precision@k` 与
列表截断的 `auc@k`。`leaderboard()` 额外给出每个模型的 `train_time_s`、`num_params`、
`size_bytes`、`status` 与 `early_stopped`；`save()` 写入的 `metadata.json` 现在包含依赖版本、
交互 schema hash 和逐模型统计，便于复现与审计。

安装 `[faiss]` extra 后，`MultiModalItemKNN` 可设置 `use_ann=True` 使用 FAISS 近似最近邻
检索以支撑超大物品 catalog；未安装 FAISS 时自动回退到精确向量内积。

## 快速开始

```python
import pandas as pd

from mmrec import MultiModalRecommender

interactions = pd.read_parquet("interactions.parquet")
users = pd.read_parquet("users.parquet")
items = pd.read_parquet("items.parquet")

recommender = MultiModalRecommender(
    user_id="user_id",
    item_id="item_id",
    timestamp="timestamp",
    label="clicked",
    eval_metrics=["recall@10", "ndcg@10", "mrr@10"],
    cache_dir="artifacts/cache",
)

recommender.fit(
    interactions=interactions,
    users=users,
    items=items,
    modalities={
        "user": {
            "categorical": ["country", "device"],
            "numerical": ["age"],
        },
        "item": {
            "categorical": ["category", "brand"],
            "numerical": ["price"],
            "text": ["title", "description"],
            # 当前 MVP 接收预计算图片向量；原始图片编码器将在后续模型包中提供。
            "image": ["image_embedding"],
        },
    },
    # auto 会在检测到 item modalities 后加入 MultiModalItemKNN。
    models="auto",
    presets="medium_quality",
    model_configs={"LightGCN": {"device": "auto", "layers": 2}},
)

# 默认仅返回自动选中的最佳模型；models="all" 返回各候选和融合结果。
recommendations = recommender.recommend(users=["u1", "u2"], k=20)
recommendations.to_parquet("artifacts/recommendations.parquet")

print(recommender.leaderboard())
recommender.evaluation_report.export("artifacts/report")
recommender.save("artifacts/model")
```

推荐结果使用统一长表：

| user_id | item_id | rank | score | model |
|---|---|---:|---:|---|
| u1 | i17 | 1 | 0.932 | ItemCF |
| u1 | i08 | 2 | 0.881 | ItemCF |
| u1 | i17 | 1 | 0.033 | RankFusion |

不同模型的原始 `score` 不保证在同一尺度上。融合使用排名而非直接相加原始分数。

## 数据契约

交互表至少包含用户和物品 ID。默认还要求 `timestamp`；不具备时间信息时可在构造函数中设置
`timestamp=None`。正负反馈数据可指定 `label`，当前版本只将 `label > 0` 的记录作为隐式正反馈。

用户表的用户 ID 必须唯一，物品表的物品 ID 必须唯一。`modalities` 中声明的每一列必须存在于
对应特征表。

## 开发

```bash
uv run pytest
uv run ruff check .
uv run python examples/quickstart.py
uv run python benchmarks/streaming_smoke.py --rows 1000000
uv build
```

构建结果位于 `dist/`。`Package and publish` 工作流会验证源码包和 wheel 的安装、
训练、推荐及保存加载；仅在发布版本号匹配的 GitHub Release 时尝试上传 PyPI。
首次发布配置与安装验收步骤见 [发布说明](docs/releasing.md)。
