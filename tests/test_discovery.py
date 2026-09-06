from __future__ import annotations

import pandas as pd

from mmrec import MultiModalRecommender


def test_fit_dataset_discovers_schema_and_trains(tmp_path) -> None:
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
            "movie_id": ["m1", "m2", "m3", "m1", "m3", "m4", "m2", "m3", "m4"],
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="h"),
            "rating": [5, 4, 5, 3, 4, 5, 4, 5, 4],
        }
    )
    items = pd.DataFrame(
        {
            "movie_id": ["m1", "m2", "m3", "m4"],
            "genre": ["action", "drama", "action", "drama"],
            "year": [2000, 2001, 2002, 2003],
            "title": [
                "an explosive action adventure",
                "a quiet family drama story",
                "the action blockbuster sequel",
                "the heartfelt drama masterpiece",
            ],
        }
    )
    interactions.to_csv(tmp_path / "interactions.csv", index=False)
    items.to_csv(tmp_path / "items.csv", index=False)

    recommender = MultiModalRecommender(eval_metrics=["recall@3"])
    recommender.fit_dataset(tmp_path, models=["Popularity", "ItemCF"])

    assert recommender.columns.user_id == "user_id"
    assert recommender.columns.item_id == "movie_id"
    assert recommender.columns.label == "rating"
    assert recommender.modalities["item"]["categorical"] == ["genre"]
    assert recommender.modalities["item"]["numerical"] == ["year"]
    assert recommender.modalities["item"]["text"] == ["title"]
    assert set(recommender.models) == {"Popularity", "ItemCF"}
