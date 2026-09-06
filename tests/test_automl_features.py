from __future__ import annotations

import json

import pandas as pd
import pytest

from mmrec import MultiModalRecommender
from mmrec.config import preset_intensity


@pytest.fixture
def interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3", "u4", "u4", "u4"],
            "item_id": ["a", "b", "c", "a", "b", "d", "a", "c", "d", "b", "c", "e"],
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
        }
    )


def test_preset_intensity_is_monotonic() -> None:
    fast = preset_intensity("fast_training")["BPRMF"]["epochs"]
    medium = preset_intensity("medium_quality")["BPRMF"]["epochs"]
    best = preset_intensity("best_quality")["BPRMF"]["epochs"]
    assert fast < medium < best


@pytest.mark.parametrize("split", ["temporal", "random", "cold_start"])
def test_split_strategies_fit(interactions: pd.DataFrame, split: str) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models="Popularity", split=split)
    assert predictor.fit_summary().dataset_summary["split"] == split


def test_new_metrics_appear_in_leaderboard(interactions: pd.DataFrame) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3", "map@3", "precision@3", "auc@3"])
    predictor.fit(interactions, models=["Popularity", "ItemCF"])
    columns = set(predictor.leaderboard().columns)
    assert {"map@3", "precision@3", "auc@3"} <= columns


def test_leaderboard_contains_statistics(interactions: pd.DataFrame) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models=["Popularity", "ItemCF"])
    columns = set(predictor.leaderboard().columns)
    assert {"train_time_s", "num_params", "size_bytes", "status"} <= columns
    trained = predictor.leaderboard()[
        predictor.leaderboard()["model"].isin(["Popularity", "ItemCF"])
    ]
    assert trained["num_params"].notna().all()
    assert (trained["num_params"] >= 0).all()


def test_parallel_training_matches_serial_models(interactions: pd.DataFrame) -> None:
    serial = MultiModalRecommender(eval_metrics=["recall@3"])
    serial.fit(interactions, models=["Popularity", "ItemCF"], num_workers=1)
    parallel = MultiModalRecommender(eval_metrics=["recall@3"])
    parallel.fit(interactions, models=["Popularity", "ItemCF"], num_workers=2)
    assert set(serial.models) == set(parallel.models) == {"Popularity", "ItemCF"}


def test_time_limit_is_recorded(interactions: pd.DataFrame) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models=["Popularity", "ItemCF"], time_limit=60.0)
    assert predictor.fit_summary().dataset_summary["time_limit"] == 60.0
    assert "early_stopped" in predictor.fit_summary().models.columns


def test_save_metadata_manifest(interactions: pd.DataFrame, tmp_path) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models=["Popularity", "ItemCF"])
    path = predictor.save(tmp_path / "model")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["package_version"]
    assert metadata["dependencies"]["numpy"] is not None
    assert metadata["dependencies"]["pandas"] is not None
    assert metadata["schema_hash"]
    assert set(metadata["model_statistics"]) == {"Popularity", "ItemCF"}


def test_ensemble_weights_are_learned(interactions: pd.DataFrame) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models=["Popularity", "ItemCF"])
    assert set(predictor.ensemble_weights) == {"Popularity", "ItemCF"}
    assert all(weight > 0 for weight in predictor.ensemble_weights.values())


def test_hyperparameter_tune_fits(interactions: pd.DataFrame) -> None:
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(interactions, models=["BPRMF"], hyperparameter_tune=True, n_trials=2)
    assert "BPRMF" in predictor.models
    assert predictor.fit_summary().models["status"].tolist() == ["succeeded"]


def test_ann_falls_back_to_exact_when_faiss_missing(interactions: pd.DataFrame) -> None:
    items = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "category": ["x", "x", "y", "y", "z"],
            "price": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    predictor = MultiModalRecommender(eval_metrics=["recall@3"])
    predictor.fit(
        interactions,
        items=items,
        modalities={"item": {"categorical": ["category"], "numerical": ["price"]}},
        models="MultiModalItemKNN",
        model_configs={"MultiModalItemKNN": {"use_ann": True}},
    )
    recommendations = predictor.recommend(
        users=["u1"], k=3, include_ensemble=False
    ).data
    assert not recommendations.empty


def test_checkpoint_resume_reuses_models(interactions: pd.DataFrame, tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    first = MultiModalRecommender(eval_metrics=["recall@3"])
    first.fit(interactions, models=["Popularity", "ItemCF"], checkpoint_dir=checkpoint)
    first_times = first.fit_summary().models.set_index("model")["train_time_s"].to_dict()

    second = MultiModalRecommender(eval_metrics=["recall@3"])
    second.fit(interactions, models=["Popularity", "ItemCF"], checkpoint_dir=checkpoint)
    second_times = second.fit_summary().models.set_index("model")["train_time_s"].to_dict()
    assert first_times == second_times
