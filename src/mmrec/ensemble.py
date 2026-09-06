"""Rank-based model fusion with a learned stacking meta-learner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmrec.config import ColumnConfig


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def reciprocal_rank_fusion(
    recommendations: pd.DataFrame,
    user_column: str,
    item_column: str,
    k: int,
    weights: dict[str, float] | None = None,
    rank_constant: int = 60,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations.copy()

    frame = recommendations.copy()
    frame["_weight"] = frame["model"].map(weights or {}).fillna(1.0)
    frame["_rrf"] = frame["_weight"] / (rank_constant + frame["rank"])
    fused = (
        frame.groupby([user_column, item_column], as_index=False, sort=False)["_rrf"]
        .sum()
        .rename(columns={"_rrf": "score"})
    )
    fused["rank"] = fused.groupby(user_column)["score"].rank(method="first", ascending=False)
    fused = fused[fused["rank"] <= k].copy()
    fused["rank"] = fused["rank"].astype(int)
    fused["model"] = "RankFusion"
    return fused[[user_column, item_column, "rank", "score", "model"]].sort_values(
        [user_column, "rank"], kind="stable"
    )


def fit_ensemble_weights(
    recommendations: pd.DataFrame,
    truth: pd.DataFrame,
    columns: ColumnConfig,
    k: int,
    random_state: int = 42,
    rank_constant: int = 60,
    max_users: int = 5000,
) -> dict[str, float]:
    """Learn per-model fusion weights with a non-negative logistic stacking learner.

    Each (user, item) candidate is represented by one feature per model --
    its reciprocal rank ``1 / (rank_constant + rank)``, or zero when the model
    did not retrieve it -- and a logistic regression meta-learner is fit against
    holdout relevance labels. The resulting coefficients are clipped to be
    non-negative and normalized so the uniform baseline remains recoverable.
    """
    user_col = columns.user_id
    item_col = columns.item_id
    model_names = recommendations["model"].drop_duplicates().tolist()
    if len(model_names) < 2 or truth.empty:
        return {name: 1.0 for name in model_names}

    rng = np.random.default_rng(random_state)
    truth_users = truth[user_col].drop_duplicates().tolist()
    if len(truth_users) > max_users:
        sampled = set(rng.choice(truth_users, size=max_users, replace=False).tolist())
        truth = truth[truth[user_col].isin(sampled)]
        recommendations = recommendations[recommendations[user_col].isin(sampled)]

    frame = recommendations.copy()
    frame["_feat"] = 1.0 / (rank_constant + frame["rank"])
    pivot = frame.pivot_table(
        index=[user_col, item_col],
        columns="model",
        values="_feat",
        aggfunc="max",
        fill_value=0.0,
    )
    truth_by_user = {
        user_id: set(group[item_col].tolist())
        for user_id, group in truth.groupby(user_col, sort=False)
    }
    rows = list(pivot.index)
    labels = np.asarray(
        [
            1.0 if rows[index][1] in truth_by_user.get(rows[index][0], set()) else 0.0
            for index in range(len(rows))
        ],
        dtype=np.float64,
    )
    features = pivot[model_names].to_numpy(dtype=np.float64)
    design = np.hstack([features, np.ones((features.shape[0], 1))])
    coefficients = _logistic_fit(design, labels, rng)
    weights = np.maximum(coefficients[:-1], 0.0)
    if weights.sum() == 0:
        weights = np.ones(len(model_names))
    weights = weights / weights.sum()
    return {name: float(weight) for name, weight in zip(model_names, weights, strict=True)}


def _logistic_fit(
    design: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    epochs: int = 200,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
) -> np.ndarray:
    count, dim = design.shape
    weights = np.zeros(dim, dtype=np.float64)
    for _ in range(epochs):
        probabilities = _sigmoid(design @ weights)
        gradient = design.T @ (probabilities - labels) / count + l2 * weights
        weights -= learning_rate * gradient
    del rng
    return weights
