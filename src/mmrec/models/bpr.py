"""Bayesian Personalized Ranking matrix factorization baseline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from mmrec.data import DatasetBundle
from mmrec.models.base import BaseRecommendationModel


class BPRMFModel(BaseRecommendationModel):
    name = "BPRMF"

    def __init__(
        self,
        columns,
        factors: int = 32,
        epochs: int = 5,
        learning_rate: float = 0.03,
        regularization: float = 1e-4,
        max_training_pairs: int = 100_000,
        random_state: int = 42,
        time_limit: float | None = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(columns)
        self.factors = factors
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_training_pairs = max_training_pairs
        self.random_state = random_state
        self.time_limit = time_limit
        self.batch_size = batch_size
        self.early_stopped = False

    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> BPRMFModel:
        self._capture_catalog(interactions, dataset)
        user_col = self.columns.user_id
        item_col = self.columns.item_id
        self.user_ids = interactions[user_col].drop_duplicates().tolist()
        self.user_indices = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.item_indices = {item_id: index for index, item_id in enumerate(self.items)}
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.05, (len(self.user_ids), self.factors)).astype(
            np.float32
        )
        self.item_factors = rng.normal(0, 0.05, (len(self.items), self.factors)).astype(np.float32)
        self.item_bias = np.zeros(len(self.items), dtype=np.float32)

        # Vectorized index mapping: pandas .map + numpy column stack avoids a
        # per-row Python loop (and the pointer-chasing of dict lookups) here.
        user_indices = interactions[user_col].map(self.user_indices).to_numpy(dtype=np.int64)
        item_indices = interactions[item_col].map(self.item_indices)
        valid = item_indices.notna().to_numpy()
        pairs = np.column_stack([user_indices[valid], item_indices[valid].to_numpy(dtype=np.int64)])
        if len(pairs) > self.max_training_pairs:
            pairs = pairs[rng.choice(len(pairs), self.max_training_pairs, replace=False)]
        positives = {
            self.user_indices[user_id]: {
                self.item_indices[item_id]
                for item_id in group[item_col]
                if item_id in self.item_indices
            }
            for user_id, group in interactions.groupby(user_col, sort=False)
        }
        self._optimize(pairs, positives, rng)

        counts = interactions[item_col].value_counts()
        maximum = float(counts.max()) if not counts.empty else 1.0
        self.popularity = {item_id: float(count / maximum) for item_id, count in counts.items()}
        return self

    def score_all_items(self, user_id: Any, history: set[Any]) -> np.ndarray | None:
        del history
        user_index = self.user_indices.get(user_id)
        if user_index is None:
            return None
        return self.item_factors @ self.user_factors[user_index] + self.item_bias

    def score_all_users(
        self, users: list[Any], histories: dict[Any, set[Any]]
    ) -> np.ndarray | None:
        del histories
        indices = []
        for user_id in users:
            user_index = self.user_indices.get(user_id)
            if user_index is None:
                return None
            indices.append(user_index)
        user_indices = np.asarray(indices, dtype=np.int64)
        return (self.user_factors[user_indices] @ self.item_factors.T + self.item_bias).astype(
            np.float32
        )

    def score_items(self, user_id: Any, item_ids: list[Any]) -> dict[Any, float]:
        user_index = self.user_indices.get(user_id)
        if user_index is None:
            return {item_id: self.popularity.get(item_id, 0.0) for item_id in item_ids}
        known = [
            (item_id, self.item_indices[item_id])
            for item_id in item_ids
            if item_id in self.item_indices
        ]
        if not known:
            return {item_id: 0.0 for item_id in item_ids}
        indices = np.asarray([index for _, index in known])
        scores = (
            self.item_factors[indices] @ self.user_factors[user_index] + self.item_bias[indices]
        )
        known_scores = dict(zip((item_id for item_id, _ in known), scores.tolist(), strict=True))
        return {item_id: float(known_scores.get(item_id, 0.0)) for item_id in item_ids}

    def _optimize(
        self,
        pairs: np.ndarray,
        positives: dict[int, set[int]],
        rng: np.random.Generator,
    ) -> None:
        if len(self.items) < 2:
            return
        deadline = None if self.time_limit is None else perf_counter() + self.time_limit
        n_items = len(self.items)
        rate = self.learning_rate
        penalty = self.regularization
        for _ in range(self.epochs):
            if deadline is not None and perf_counter() >= deadline:
                self.early_stopped = True
                break
            order = rng.permutation(len(pairs))
            for start in range(0, len(order), self.batch_size):
                if deadline is not None and perf_counter() >= deadline:
                    self.early_stopped = True
                    break
                batch = order[start : start + self.batch_size]
                users = pairs[batch, 0]
                positive_indices = pairs[batch, 1]
                negative_indices = self._sample_negatives(users, positives, n_items, rng)

                # Mini-batch BPR: same objective, vectorized over the batch.
                user_vector = self.user_factors[users]
                positive_vector = self.item_factors[positive_indices]
                negative_vector = self.item_factors[negative_indices]
                difference = (
                    self.item_bias[positive_indices]
                    - self.item_bias[negative_indices]
                    + (user_vector * (positive_vector - negative_vector)).sum(axis=1)
                )
                gradient = 1.0 / (1.0 + np.exp(np.clip(difference, -30, 30)))

                user_update = rate * (
                    gradient[:, None] * (positive_vector - negative_vector) - penalty * user_vector
                )
                positive_update = rate * (
                    gradient[:, None] * user_vector - penalty * positive_vector
                )
                negative_update = rate * (
                    -gradient[:, None] * user_vector - penalty * negative_vector
                )
                np.add.at(self.user_factors, users, user_update)
                np.add.at(self.item_factors, positive_indices, positive_update)
                np.add.at(self.item_factors, negative_indices, negative_update)
                np.add.at(
                    self.item_bias,
                    positive_indices,
                    rate * (gradient - penalty * self.item_bias[positive_indices]),
                )
                np.add.at(
                    self.item_bias,
                    negative_indices,
                    rate * (-gradient - penalty * self.item_bias[negative_indices]),
                )

    def _sample_negatives(
        self,
        users: np.ndarray,
        positives: dict[int, set[int]],
        n_items: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        negatives = rng.integers(0, n_items, size=len(users), dtype=np.int64)
        for index in range(len(users)):
            user_index = int(users[index])
            if len(positives.get(user_index, ())) >= n_items:
                negatives[index] = 0
                continue
            while negatives[index] in positives[user_index]:
                negatives[index] = int(rng.integers(0, n_items))
        return negatives
