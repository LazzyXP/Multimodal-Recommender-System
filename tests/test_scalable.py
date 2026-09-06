from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from mmrec import MultiModalRecommender


def _large_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamp = pd.Timestamp("2026-01-01")
    for user_index in range(20):
        for item_offset in range(4):
            rows.append(
                {
                    "user_id": f"u{user_index}",
                    "item_id": f"i{(user_index + item_offset) % 12}",
                    "timestamp": timestamp,
                }
            )
            timestamp += pd.Timedelta(minutes=1)
    rows.append({"user_id": "tail-user", "item_id": "tail-item", "timestamp": timestamp})
    return pd.DataFrame(rows)


def test_streaming_mode_uses_full_popularity_and_sampled_itemcf(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    _large_fixture().to_parquet(source, index=False)
    predictor = MultiModalRecommender(
        eval_metrics=["recall@3"],
        cache_dir=tmp_path / "cache",
        execution_mode="auto",
        max_in_memory_interactions=20,
        sample_interactions=24,
        scan_batch_size=10,
    )
    predictor.fit(source, models=["Popularity", "ItemCF"])

    summary = predictor.fit_summary()
    scopes = summary.models.set_index("model")["training_scope"].to_dict()
    assert summary.dataset_summary["execution_mode"] == "streaming"
    assert summary.dataset_summary["interactions"] == 81
    assert summary.dataset_summary["sampled_interactions"] <= 24
    assert scopes == {"Popularity": "full", "ItemCF": "sampled"}
    assert "tail-item" in predictor.models["Popularity"].scores


def test_streaming_recommend_uses_exact_seen_history(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    _large_fixture().to_parquet(source, index=False)
    predictor = MultiModalRecommender(
        eval_metrics=["recall@3"],
        cache_dir=tmp_path / "cache",
        execution_mode="streaming",
        sample_interactions=20,
        scan_batch_size=8,
    ).fit(source, models=["Popularity", "ItemCF"])

    result = predictor.recommend(users=["u0"], k=5)
    assert not {"i0", "i1", "i2", "i3"} & set(result.data["item_id"])
    with pytest.raises(ValueError, match="users must be provided"):
        predictor.recommend(k=5)


def test_recommend_to_parquet_accepts_user_file(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    users = tmp_path / "users.csv"
    output = tmp_path / "recommendations.parquet"
    _large_fixture().to_parquet(source, index=False)
    pd.DataFrame({"user_id": ["u0", "u1", "u2"]}).to_csv(users, index=False)

    predictor = MultiModalRecommender(
        eval_metrics=["recall@3"],
        cache_dir=tmp_path / "cache",
        execution_mode="streaming",
        sample_interactions=30,
        inference_batch_size=2,
    ).fit(source, models=["Popularity", "ItemCF"])
    predictor.recommend_to_parquet(users, output, k=3, batch_size=2, models="all")

    recommendations = pd.read_parquet(output)
    assert set(recommendations["user_id"]) == {"u0", "u1", "u2"}
    assert set(recommendations["model"]) == {"Popularity", "ItemCF", "RankFusion"}


def test_recommend_to_parquet_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    _large_fixture().to_parquet(source, index=False)
    predictor = MultiModalRecommender(cache_dir=tmp_path / "cache").fit(
        source, models="Popularity"
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        predictor.recommend_to_parquet(["u0"], tmp_path / "out.parquet", batch_size=0)


def test_streaming_model_can_bundle_history_index(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    _large_fixture().to_parquet(source, index=False)
    predictor = MultiModalRecommender(
        eval_metrics=["recall@3"],
        cache_dir=tmp_path / "cache",
        execution_mode="streaming",
        sample_interactions=20,
    ).fit(source, models="Popularity")
    original_index = Path(predictor.history_index or "")
    model_path = predictor.save(tmp_path / "model", include_history_index=True)

    shutil.rmtree(original_index)
    source.unlink()
    loaded = MultiModalRecommender.load(model_path)
    result = loaded.recommend(users=["u0"], k=3)
    assert not {"i0", "i1", "i2", "i3"} & set(result.data["item_id"])


def test_streaming_catalog_budget_is_reported(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    _large_fixture().to_parquet(source, index=False)
    predictor = MultiModalRecommender(
        eval_metrics=["recall@3"],
        cache_dir=tmp_path / "cache",
        execution_mode="streaming",
        sample_interactions=20,
        max_catalog_items=5,
    ).fit(source, models="Popularity")

    assert predictor.fit_summary().dataset_summary["catalog_truncated"] is True
    assert len(predictor.models["Popularity"].items) == 5
