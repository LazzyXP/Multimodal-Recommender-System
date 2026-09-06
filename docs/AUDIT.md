# 验收自审报告（AutoGluon 对齐）

> 面向推荐领域的 AutoGluon 风格实现自审。结论先行：**当前版本已经具备统一 API、三路
> 评测、简化 HPO、模型融合和流式数据入口，但仍是实验性 MVP；距离 AutoGluon 的生产级
> 调度、OOF 集成、资源治理和预训练多模态还有明显差距。**

## 自动选模工作流更新

## README 审核补充（0.1.2 源码）

测试通过、wheel 可构建不等于生产验收完成。以下问题应优先于继续添加模型处理：

| 优先级 | 当前证据 | 后续验收要求 |
|---|---|---|
| 已完成 | `predictor.save()` 的多文件一致性 | generation 目录完整写入后原子替换 `.CURRENT` 指针；兼容旧根目录 artifact，并保留上一代回退和写锁 |
| 已完成 | CI 显式导入 torch 后运行图模型测试 | CPU smoke 会打印 torch 版本/CUDA 状态，缺 torch 时在测试前失败 |
| 已完成 | benchmark 支持真实数据和多随机种子 | JSON 记录 Recall/NDCG、耗时、峰值 RSS、平台和每次运行的均值/标准差 |
| 中 | `MultiModalItemKNN` 支持 `IndexFlatIP`、HNSW、IVF 和 IVF-PQ | 已覆盖 `ann_topk`/后端参数校验、已见物品过滤和批量/单用户结果一致性；`benchmarks/benchmark_retrieval.py` 可测大目录延迟/内存，真实召回率仍待服务器数据 |

当前 FAISS 默认路径是精确的 `IndexFlatIP`，`ann_backend="hnsw"`/`"ivf"`/`"ivfpq"` 才是近似或压缩 ANN；不同模型的 embedding 不同，也不能单独证明论文实现正确。
服务器环境可用于后续 GPU 和真实数据 benchmark，但本地验收不把服务器可用性当作已完成证据。
本地合成目录 benchmark（1000 items / 50 users / 2 seeds）显示 HNSW、IVF、IVF-PQ 的
`recall_at_20_vs_exact` 分别约为 0.902、0.677、0.523；这些数字只用于证明测量链路，
不代表真实业务数据的最终质量。
截至本次核验，GitHub 最新 Published Release 仍为 `v0.1.0`，PyPI 最新版本也仍为 `0.1.0`；
`v0.1.2` tag 已推送，但尚未创建 Published Release，因此不能把 0.1.2 视为已发布。

### 自动选模工作流记录

已补充验证集主指标选模、默认最佳模型推荐、显式验证/测试集、全局时间窗口切分、
最佳模型及依赖重训、分阶段 checkpoint，以及仅导出最佳模型。外部测试集始终排除在重训之外。
候选的测试分数来自仅用 train 训练的版本；保存的模型可能已经重训，报告不宣称该分数属于
重训后的模型。未重训候选仍可通过 `models="all"` 对比。

以下历史验收记录中的测试数量和性能数据为当时结果；新增工作流回归见
`tests/test_automl_workflow.py`。融合仍共享单次 validation 学权重和选模型，并非 OOF。

## 一、对照 AutoGluon 的验收清单

| AutoGluon 支柱 | 状态 | 证据 / 位置 |
|---|---|---|
| 统一 API + leaderboard | ✅ | `MultiModalRecommender.fit/recommend/leaderboard/predict/evaluate/save/load` |
| presets + 时间预算 | ⚠️ 部分 | 内置模型使用总 deadline 和阶段早停；自定义模型不可强制中断，训练前后仍有调度开销 |
| 模型动物园（多样性） | ✅（简化实现） | 14 个模型；8 个 torch 图模型各自实现论文的**核心机制**（`mmrec/models/graph/`，独立文件），但为简化实现、非逐行复现 |
| 自动集成（stacking） | ⚠️ 简化 | 单次 validation 上学习 RRF 权重；没有多折 OOF、bagging 或模型级 stacking |
| 超参搜索（HPO） | ⚠️ 简化 | 离散 TPE 式采样 + 两阶段筛选；没有完整 trial 调度、资源分配、并行搜索和恢复 |
| 并行训练 | ✅ | `num_workers` 线程池；torch 模型串行（避免并发不稳定） |
| 大数据路径 | ⚠️ 部分 | 交互扫描和部分图操作分块；特征矩阵、相似度块和图边仍可能成为瓶颈 |
| 可复现 / artifact | ✅ | `metadata.json` 含依赖版本 + schema hash + 逐模型统计；`checkpoint_dir` 断点续训 |
| 评测完整性 | ⚠️ | 推荐指标和三路切分齐全，但仍是单次 holdout，不是 AutoGluon 的 OOF/bagging 评测 |

## 二、关键验证证据

- **测试**：84 个用例全部通过（模型选择、流式、存储、torch 模型、指标数值正确性、
  TPE、集成、HPO、checkpoint、manifest）。`ruff` 全绿。
- **模型保真度**：8 个 torch 模型两两 item embedding **全部不同（28/28 对）**，证明
  不是「共享基类 + 开关」的别名，而是独立的图构建/传播/损失。
- **性能**：向量化推荐 fast path，500 用户 × 5 万 item 的 top-20 推荐约 0.44s；
  `benchmarks/benchmark.py` 提供端到端 timing/throughput 证据。
- **打包**：核心依赖仅 numpy/pandas/pyarrow，torch/faiss 全在 extra；wheel 68 KB，
  sdist 无缓存/虚拟环境泄漏。

## 三、本轮（验收前）完成

1. **8 个 torch 模型核心机制重构**（删除共享基类 + variant 的旧实现）：
   LightGCN（D^{-1/2} 对称归一化）、MMGCN（模态图 + 注意力）、LATTICE（学习 latent
   图 + 对比）、BM3（EMA bootstrap 对比）、FREEDOM（冻结图 + 去噪）、MGCN（多视图
   对比 + 行为引导）、DRAGON（user-user + item-item 同构图）、LGMRec（local+global）。
   每个实现论文的核心机制，但是**简化实现，非逐行复现**。
2. **meta-learner 堆叠**：非负逻辑回归在 validation 上监督学习 RRF 权重；最终指标使用独立
   test 切分，但仍不是多折 OOF stacking。
3. **简化 TPE HPO**：`hpo.TpeSearch` 无外部依赖的 TPE 式加权采样 + successive-halving。
4. **FAISS 检索**：`[faiss]` extra + `MultiModalItemKNN(use_ann=True)` 默认使用 `IndexFlatIP`，
   也支持 HNSW、IVF 和 IVF-PQ；缺依赖时自动回退 NumPy 精确内积。
5. **修复默认学习率导致的图模型欠拟合**：`lr 1e-3 → 1e-2`、`epochs 20 → 50`（修复前 BPR
   损失停在 log2≈0.693 附近不降，修复后 0.684 → 0.287，已验证梯度正常流动）。

## 四、当前剩余差距

**（1）论文指标的数值对齐。** 8 个模型实现的是各论文的核心机制，是简化
实现，**不是逐行复现**；即便拿到官方数据集，也需要进一步对齐论文的确切架构/损失/
超参后才能复现其报告的 Recall/NDCG。因此「数值对齐」同时是**实现保真度缺口 + 实验
验证缺口**，二者都要在 GPU 机器上用官方数据完成，本沙箱无法闭环。

**（2）AutoML 工程能力：**
- HPO 是离散空间的轻量 TPE 式搜索，缺少完整调度器、资源隔离、trial 恢复和搜索日志；
- stacking 只使用单次 validation，缺少多折 OOF、bagging、置信区间和防泄漏的层级评估；
- `time_limit` 对内置模型是 best-effort deadline，对自定义模型不能硬中断；
- checkpoint 已使用表内容指纹，但仍以 pickle 为主，缺少跨版本 schema migration 和安全 artifact 格式。
- generation 默认保留最近 3 代，可通过 `save(max_generations=...)` 调整；旧代清理发生在写锁内，`load()` 会等待当前写入完成。跨进程已打开旧文件的长时间读取仍需由部署方避免与清理并发。

**（3）多模态与规模：**
- item 特征仍是整体进内存的 dense 矩阵，尚无 mmap、分片编码或完整 ANN 训练路径；
- 图模型的相似度计算已动态限块，但仍可能产生 `chunk x item_count` 临时矩阵；
- `MultiModalItemKNN` 已消费与物品同维度的用户 embedding，为无历史用户提供内容冷启动；其他模型仍主要依赖交互历史；
- 图模型实现是论文核心机制的简化版本，尚未在官方数据集上逐模型调优和复现。

## 五、官方数据集对齐的复现路径

```bash
# 1. 下载官方数据集（MMRec 数据层提供，本项目不复制其 GPL 源码）
#    Amazon-Kindle / Baby / 微视频 / Yelp → interactions + item 特征表

# 2. 用相同协议训练并对齐
from mmrec import MultiModalRecommender
recommender = MultiModalRecommender(eval_metrics=["recall@20", "ndcg@20"])
recommender.fit(
    interactions_parquet,
    items=items_parquet,
    modalities={"item": {"text": ["title", "desc"], "image": ["image_emb"]}},
    models=["LightGCN", "MMGCN", "LATTICE", "BM3", "FREEDOM", "MGCN", "DRAGON", "LGMRec"],
    presets="best_quality",
    time_limit=3600 * 8,
    checkpoint_dir="artifacts/checkpoint",
)
recommender.evaluation_report.export("artifacts/report")
```

对齐时要固定：层数/嵌入维/学习率/温度/SSL 系数/负采样数/随机种子，并在报告中逐项
记录（`save()` 的 `metadata.json` 已含 schema hash 与依赖版本，可直接复用）。
