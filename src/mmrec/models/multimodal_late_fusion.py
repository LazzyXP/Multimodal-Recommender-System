"""Late-fusion content recommender with independent modality profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmrec.data import DatasetBundle
from mmrec.features import encode_item_feature_blocks
from mmrec.models.base import BaseRecommendationModel


class MultiModalLateFusionKNNModel(BaseRecommendationModel):
    name = "MultiModalLateFusionKNN"

    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> MultiModalLateFusionKNNModel:
        if dataset is None or dataset.items is None:
            raise ValueError("MultiModalLateFusionKNN requires an items feature table.")
        modalities = dataset.modalities.get("item", {})
        self._capture_catalog(interactions, dataset)
        item_ids, blocks = encode_item_feature_blocks(
            dataset.items,
            self.columns.item_id,
            modalities,
        )
        if len(blocks) < 2:
            raise ValueError("MultiModalLateFusionKNN requires at least two item modalities.")
        self.item_indices = {item_id: index for index, item_id in enumerate(item_ids)}
        self.feature_blocks = blocks
        counts = interactions[self.columns.item_id].value_counts()
        maximum = float(counts.max()) if not counts.empty else 1.0
        self.popularity = {item_id: float(count / maximum) for item_id, count in counts.items()}
        return self

    def score_all_items(self, user_id: Any, history: set[Any]) -> np.ndarray | None:
        del user_id
        history_indices = [
            self.item_indices[item_id]
            for item_id in history
            if item_id in self.item_indices
        ]
        if not history_indices:
            return None
        total = np.zeros(len(self.items), dtype=np.float32)
        active_blocks = 0
        for matrix in self.feature_blocks.values():
            profile = matrix[history_indices].mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm == 0:
                continue
            total += (matrix @ (profile / norm)).astype(np.float32)
            active_blocks += 1
        if not active_blocks:
            return None
        return total / active_blocks

    def score_all_users(
        self, users: list[Any], histories: dict[Any, set[Any]]
    ) -> np.ndarray | None:
        """Batched content scoring: per-block profile matrix + BLAS matmul."""
        batch = len(users)
        total = np.zeros((batch, len(self.items)), dtype=np.float32)
        active_blocks = 0
        for matrix in self.feature_blocks.values():
            profiles = np.zeros((batch, matrix.shape[1]), dtype=np.float32)
            valid = np.zeros(batch, dtype=bool)
            for index, user_id in enumerate(users):
                history_indices = [
                    self.item_indices[item_id]
                    for item_id in histories.get(user_id, ())
                    if item_id in self.item_indices
                ]
                if not history_indices:
                    continue
                profile = matrix[history_indices].mean(axis=0)
                norm = np.linalg.norm(profile)
                if norm == 0:
                    continue
                profiles[index] = profile / norm
                valid[index] = True
            if not valid.all():
                return None
            total += (profiles @ matrix.T).astype(np.float32)
            active_blocks += 1
        if not active_blocks:
            return None
        return total / active_blocks

    def score_items(self, user_id: Any, item_ids: list[Any]) -> dict[Any, float]:
        return self.score_items_with_history(
            user_id,
            item_ids,
            self.seen_by_user.get(user_id, set()),
        )

    def score_items_with_history(
        self,
        user_id: Any,
        item_ids: list[Any],
        history: set[Any],
    ) -> dict[Any, float]:
        del user_id
        history_indices = [
            self.item_indices[item_id]
            for item_id in history
            if item_id in self.item_indices
        ]
        if not history_indices:
            return {item_id: self.popularity.get(item_id, 0.0) for item_id in item_ids}
        known = [
            (item_id, self.item_indices[item_id])
            for item_id in item_ids
            if item_id in self.item_indices
        ]
        if not known:
            return {item_id: 0.0 for item_id in item_ids}
        candidate_indices = np.asarray([index for _, index in known])
        total = np.zeros(len(known), dtype=np.float32)
        active_blocks = 0
        for matrix in self.feature_blocks.values():
            profile = matrix[history_indices].mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm == 0:
                continue
            total += (matrix[candidate_indices] @ (profile / norm)).astype(np.float32)
            active_blocks += 1
        if active_blocks:
            total /= active_blocks
        known_scores = dict(zip((item_id for item_id, _ in known), total.tolist(), strict=True))
        return {item_id: float(known_scores.get(item_id, 0.0)) for item_id in item_ids}
