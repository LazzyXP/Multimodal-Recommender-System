from __future__ import annotations

import pandas as pd
import pytest

from mmrec import MultiModalRecommender
from mmrec.config import ColumnConfig
from mmrec.models import MultiModalItemKNNModel, torch_available


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.DataFrame(
        {
            "user_id": [
                "u1",
                "u1",
                "u1",
                "u2",
                "u2",
                "u2",
                "u3",
                "u3",
                "u3",
                "u4",
                "u4",
                "u4",
            ],
            "item_id": ["a", "b", "c", "a", "c", "d", "b", "d", "e", "a", "e", "f"],
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
        }
    )
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e", "f", "cold"],
            "category": ["book", "book", "film", "film", "music", "music", "book"],
            "price": [10, 11, 20, 22, 8, 9, 12],
            "title": ["alpha", "beta", "cinema", "drama", "song", "album", "alpha guide"],
            "image_embedding": [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.5, 0.5],
                [0.4, 0.6],
                [0.95, 0.05],
            ],
        }
    )
    return interactions, items


def test_multimodal_ann_topk_must_be_positive() -> None:
    with pytest.raises(ValueError, match="ann_topk must be positive"):
        MultiModalItemKNNModel(ColumnConfig(), use_ann=True, ann_topk=0)


def test_multimodal_ann_backend_validation() -> None:
    with pytest.raises(ValueError, match="ann_backend"):
        MultiModalItemKNNModel(ColumnConfig(), use_ann=True, ann_backend="pq")
    model = MultiModalItemKNNModel(
        ColumnConfig(), use_ann=True, ann_backend="hnsw", ann_hnsw_m=16, ann_ef_search=32
    )
    assert (model.ann_backend, model.ann_hnsw_m, model.ann_ef_search) == ("hnsw", 16, 32)
    ivf = MultiModalItemKNNModel(
        ColumnConfig(), use_ann=True, ann_backend="ivf", ann_nlist=4, ann_nprobe=2
    )
    assert (ivf.ann_nlist, ivf.ann_nprobe) == (4, 2)
    ivfpq = MultiModalItemKNNModel(
        ColumnConfig(), use_ann=True, ann_backend="ivfpq", ann_pq_m=4, ann_pq_nbits=8
    )
    assert (ivfpq.ann_pq_m, ivfpq.ann_pq_nbits) == (4, 8)


def test_model_catalog_exposes_families_and_references() -> None:
    predictor = MultiModalRecommender()
    catalog = predictor.model_catalog().set_index("name")
    assert {"BPRMF", "MultiModalItemKNN", "MultiModalLateFusionKNN", "VBPR"} <= set(
        catalog.index
    )
    assert catalog.loc["VBPR", "requires_item_modalities"]
    assert catalog.loc["VBPR", "reference"] == "https://arxiv.org/abs/1510.01784"


def test_medium_auto_runs_multiple_multimodal_models(tmp_path) -> None:
    interactions, items = _data()
    predictor = MultiModalRecommender(
        eval_metrics=["recall@3", "ndcg@3"],
        cache_dir=tmp_path / "cache",
    ).fit(
        interactions,
        items=items,
        modalities={
            "item": {
                "categorical": ["category"],
                "numerical": ["price"],
                "text": ["title"],
                "image": ["image_embedding"],
            }
        },
        models="auto",
        presets="medium_quality",
    )

    expected = {
        "Popularity",
        "ItemCF",
        "BPRMF",
        "MultiModalItemKNN",
        "MultiModalLateFusionKNN",
        "VBPR",
    }
    if torch_available():
        expected |= {
            "LightGCN",
            "MMGCN",
            "LATTICE",
            "BM3",
            "FREEDOM",
            "MGCN",
            "DRAGON",
            "LGMRec",
        }
    assert set(predictor.models) == expected
    assert set(predictor.fit_summary().models["status"]) == {"succeeded"}
    assert (predictor.leaderboard()["coverage"] <= 1.0).all()
    result = predictor.recommend(users=["u1"], k=3, models="all")
    assert set(result.models) == expected | {"RankFusion"}
    assert "cold" in set(result.data["item_id"])


def test_fast_preset_keeps_multimodal_candidate_set_small(tmp_path) -> None:
    interactions, items = _data()
    predictor = MultiModalRecommender(cache_dir=tmp_path / "cache").fit(
        interactions,
        items=items,
        modalities={"item": {"text": ["title"], "image": ["image_embedding"]}},
        models="auto",
        presets="fast_training",
    )
    assert set(predictor.models) == {"Popularity", "ItemCF", "MultiModalItemKNN"}
