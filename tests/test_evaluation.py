from __future__ import annotations

import math

import pandas as pd
import pytest

from mmrec.config import ColumnConfig
from mmrec.evaluation import evaluate_recommendations


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["a", "c", "d"],
            "rank": [1, 2, 3],
            "score": [0.9, 0.4, 0.2],
            "model": ["M", "M", "M"],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame({"user_id": ["u1", "u1"], "item_id": ["a", "b"]})


def test_metric_values_are_correct() -> None:
    columns = ColumnConfig()
    metrics = ["recall@3", "ndcg@3", "mrr@3", "hit_rate@3", "map@3", "precision@3", "auc@3"]
    leaderboard, details = evaluate_recommendations(
        _frame(), _truth(), columns, metrics, catalog_size=10
    )
    row = details["M"]
    assert row["recall@3"] == pytest.approx(0.5)
    assert row["precision@3"] == pytest.approx(1 / 3)
    assert row["hit_rate@3"] == pytest.approx(1.0)
    assert row["mrr@3"] == pytest.approx(1.0)
    assert row["map@3"] == pytest.approx(0.5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert row["ndcg@3"] == pytest.approx(1 / ideal)
    assert row["auc@3"] == pytest.approx(1.0)
    assert set(leaderboard.columns) >= set(metrics) | {"coverage", "model"}
