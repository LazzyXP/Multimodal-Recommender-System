"""Content-based item recommender backed by multimodal feature vectors.

Scoring defaults to an exact vectorized inner product (BLAS), which is optimal
for exact recall. When ``use_ann=True`` and the optional ``faiss-cpu`` package
is installed, an approximate nearest-neighbour index is built instead so the
catalog can scale well beyond what an exact dense pass supports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmrec.data import DatasetBundle
from mmrec.features import encode_item_features
from mmrec.models.base import BaseRecommendationModel


class MultiModalItemKNNModel(BaseRecommendationModel):
    name = "MultiModalItemKNN"

    def __init__(self, columns, use_ann: bool = False, ann_topk: int = 100) -> None:
        super().__init__(columns)
        self.use_ann = use_ann
        self.ann_topk = ann_topk
        self._ann_index: Any = None

    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> MultiModalItemKNNModel:
        if dataset is None or dataset.items is None:
            raise ValueError("MultiModalItemKNN requires an items feature table.")
        item_modalities = dataset.modalities.get("item", {})
        if not item_modalities:
            raise ValueError("MultiModalItemKNN requires declared item modalities.")

        self._capture_catalog(interactions, dataset)
        item_ids, matrix = encode_item_features(
            dataset.items,
            self.columns.item_id,
            item_modalities,
        )
        self.feature_matrix = matrix
        self.item_indices = {item_id: index for index, item_id in enumerate(item_ids)}
        self.item_vectors = {item_id: matrix[index] for index, item_id in enumerate(item_ids)}
        counts = interactions[self.columns.item_id].value_counts()
        maximum = float(counts.max()) if not counts.empty else 1.0
        self.popularity = {item_id: float(count / maximum) for item_id, count in counts.items()}
        self._ann_index = self._build_ann_index() if self.use_ann else None
        return self

    def _build_ann_index(self) -> Any:
        try:
            import faiss  # type: ignore[import-not-found]

            index = faiss.IndexFlatIP(self.feature_matrix.shape[1])
            index.add(self.feature_matrix.astype(np.float32))
            return index
        except ImportError:
            return None

    def score_all_items(self, user_id: Any, history: set[Any]) -> np.ndarray | None:
        del user_id
        history_vectors = [
            self.item_vectors[item_id]
            for item_id in history
            if item_id in self.item_vectors
        ]
        if not history_vectors:
            return None
        profile = np.mean(history_vectors, axis=0)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return None
        profile = profile / norm

        if self._ann_index is not None:
            scores = np.zeros(len(self.items), dtype=np.float32)
            values, indices = self._ann_index.search(
                profile.astype(np.float32)[None, :], self.ann_topk
            )
            for position, index in enumerate(indices[0]):
                if index >= 0:
                    scores[index] = values[0][position]
            return scores
        return self.feature_matrix @ profile

    def score_all_users(
        self, users: list[Any], histories: dict[Any, set[Any]]
    ) -> np.ndarray | None:
        """Batched content scoring: one profile matrix, one BLAS matmul."""
        profiles = []
        for user_id in users:
            history_vectors = [
                self.item_vectors[item_id]
                for item_id in histories.get(user_id, ())
                if item_id in self.item_vectors
            ]
            if not history_vectors:
                return None
            profile = np.mean(history_vectors, axis=0)
            norm = np.linalg.norm(profile)
            if norm == 0:
                return None
            profiles.append(profile / norm)
        matrix = np.stack(profiles)  # (B, D)
        return (matrix @ self.feature_matrix.T).astype(np.float32)  # (B, N)

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
        history_vectors = [
            self.item_vectors[item_id]
            for item_id in history
            if item_id in self.item_vectors
        ]
        if not history_vectors:
            return {item: self.popularity.get(item, 0.0) for item in item_ids}

        profile = np.mean(history_vectors, axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm:
            profile = profile / profile_norm
        known_items = [item_id for item_id in item_ids if item_id in self.item_indices]
        known_indices = [self.item_indices[item_id] for item_id in known_items]
        similarities = self.feature_matrix[known_indices] @ profile
        scores = dict(zip(known_items, similarities.tolist(), strict=True))
        return {item_id: float(scores.get(item_id, 0.0)) for item_id in item_ids}
