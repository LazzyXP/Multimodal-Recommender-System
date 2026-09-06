from __future__ import annotations

import pandas as pd
import pytest

from mmrec import MultiModalRecommender


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
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
            "item_id": ["a", "b", "c", "a", "b", "d", "a", "c", "d", "b", "c", "e"],
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
        }
    )


@pytest.fixture
def fitted(interactions: pd.DataFrame) -> MultiModalRecommender:
    predictor = MultiModalRecommender(eval_metrics=["recall@3", "ndcg@3", "mrr@3"])
    return predictor.fit(interactions, models=["Popularity", "ItemCF"])


def test_fit_produces_multimodel_report(fitted: MultiModalRecommender) -> None:
    leaderboard = fitted.leaderboard()
    assert set(leaderboard["model"]) == {"Popularity", "ItemCF", "RankFusion"}
    assert {"recall@3", "ndcg@3", "mrr@3", "coverage"} <= set(leaderboard.columns)
    assert set(fitted.fit_summary().models["status"]) == {"succeeded"}
    assert not fitted.evaluation_report.leaderboard_data.empty


def test_recommend_returns_each_model_and_fusion(fitted: MultiModalRecommender) -> None:
    result = fitted.recommend(users=["u1"], k=3, models="all")
    assert set(result.models) == {"Popularity", "ItemCF", "RankFusion"}
    seen = {"a", "b", "c"}
    assert not set(result.data["item_id"]) & seen
    assert (result.data.groupby("model")["rank"].max() <= 3).all()


def test_external_evaluation(fitted: MultiModalRecommender) -> None:
    test = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "item_id": ["d", "e"],
            "timestamp": pd.to_datetime(["2026-02-01", "2026-02-01"]),
        }
    )
    report = fitted.evaluate(test)
    assert set(report.leaderboard()["model"]) == {"Popularity", "ItemCF", "RankFusion"}


def test_save_and_load(fitted: MultiModalRecommender, tmp_path) -> None:
    model_path = fitted.save(tmp_path / "model")
    loaded = MultiModalRecommender.load(model_path)
    expected = fitted.recommend(users=["u1"], k=2).data
    actual = loaded.recommend(users=["u1"], k=2).data
    pd.testing.assert_frame_equal(expected, actual)


def test_validates_declared_modalities(interactions: pd.DataFrame) -> None:
    items = pd.DataFrame({"item_id": ["a", "b", "c", "d", "e"], "title": ["a"] * 5})
    predictor = MultiModalRecommender()
    predictor.fit(
        interactions,
        items=items,
        modalities={"item": {"text": ["title"]}},
        models="Popularity",
    )
    assert predictor.fit_summary().dataset_summary["declared_modalities"] == {
        "item": {"text": ["title"]}
    }


def test_rejects_non_numeric_labels(interactions: pd.DataFrame) -> None:
    invalid = interactions.assign(label="positive")
    predictor = MultiModalRecommender(label="label")
    with pytest.raises(ValueError, match="must contain numeric values"):
        predictor.fit(invalid, models="Popularity")


def test_duplicate_dataframe_index_does_not_corrupt_holdout(interactions: pd.DataFrame) -> None:
    interactions.index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models="Popularity")
    assert predictor.fit_summary().dataset_summary["test_interactions"] == 4


def test_auto_adds_multimodal_model(interactions: pd.DataFrame) -> None:
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e", "cold"],
            "category": ["book", "book", "movie", "movie", "music", "book"],
            "price": [10, 12, 20, 18, 8, 11],
            "title": ["alpha", "beta", "cinema", "drama", "song", "alpha guide"],
            "image_embedding": [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.5, 0.5],
                [0.95, 0.05],
            ],
        }
    )
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(
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
    )
    assert "MultiModalItemKNN" in predictor.models
    recommendations = predictor.recommend(
        users=["u1"], k=5, models="MultiModalItemKNN", include_ensemble=False
    ).data
    assert "cold" in set(recommendations["item_id"])
