# Model catalog

The automatic candidate set is organized by model family rather than paper count. Every candidate
must implement the same split, seen-item filtering, recommendation result, and evaluation contracts.

## Built in

| Model | Family | Modalities | Automatic use |
|---|---|---|---|
| Popularity | baseline | none | every preset |
| ItemCF | collaborative neighborhood | none | every preset |
| BPRMF | pairwise latent factor | none | medium and best |
| MultiModalItemKNN | early feature fusion | item features | every multimodal preset |
| MultiModalLateFusionKNN | score-level modality fusion | 2+ item modalities | medium and best |
| VBPR | pairwise collaborative-content ranking | item features | medium and best |
| RankFusion | reciprocal-rank ensemble | model outputs | whenever 2+ models succeed |

The built-in VBPR generalizes the visual feature term to the declared item feature vector while
retaining the pairwise collaborative and content-factor objective from the original paper.

## Torch model layer

The optional `torch` model package provides paper-specific implementations. Each model has its own
graph construction, message passing and loss, sharing only a generic training harness (edge
normalization, sparse propagation, BPR/InfoNCE primitives, device management and persistence):

| Model | Paper mechanism |
|---|---|
| LightGCN | symmetric D^{-1/2} normalization, K-layer propagation, uniform layer mean, BPR |
| MMGCN | per-modality item-item graphs + learned attention fusion across modality views |
| LATTICE | learned latent item-item structure contrasted with the feature graph (InfoNCE) |
| BM3 | online encoder + EMA target encoder, bootstrap contrastive + BPR |
| FREEDOM | frozen modality graphs + degree-sensitive denoising of the interaction graph |
| MGCN | collaborative vs multimodal view contrastive with interaction-co-occurrence guidance |
| DRAGON | user-user (co-occurrence) + item-item (feature) homogeneous graphs + contrastive |
| LGMRec | local bipartite embeddings combined with global semantic-graph embeddings |

They live behind a `torch` extra so the core CSV/Parquet conversion and lightweight models do not
require a large deep-learning runtime. The implementation uses `device="auto"` with CPU, CUDA and
Apple MPS selection, and item-item graphs are built in row chunks so an N x N similarity matrix is
never materialized.

> **Fidelity note.** Each model above implements the paper's *distinct core mechanism* (graph
> construction, propagation and loss) independently, but is a simplified implementation, not a
> line-by-line reproduction. Reproducing the paper's reported Recall/NDCG still requires the
> official datasets plus hyperparameter tuning against those numbers.

## Sources and licensing

- [MMRec toolbox](https://github.com/enoche/MMRec)
- [MMRec paper](https://arxiv.org/abs/2302.03497)
- [Multimodal recommender systems survey](https://arxiv.org/abs/2302.04473)
- [VBPR](https://arxiv.org/abs/1510.01784)

MMRec is GPL-3.0. This Apache-2.0 package uses it as a model catalog and benchmarking reference only;
its source code is not copied into this project.
