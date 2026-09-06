"""Global popularity recommendation baseline."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from mmrec.data import DatasetBundle
from mmrec.models.base import BaseRecommendationModel


class PopularityModel(BaseRecommendationModel):
    name = "Popularity"

    def __init__(self, columns, time_limit: float | None = None) -> None:
        super().__init__(columns)
        self.time_limit = time_limit
        self.early_stopped = False

    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> PopularityModel:
        deadline = None if self.time_limit is None else perf_counter() + self.time_limit
        self._capture_catalog(interactions, dataset)
        if deadline is not None and perf_counter() >= deadline:
            self.early_stopped = True
            raise TimeoutError("Popularity training exceeded its time budget.")
        counts = interactions[self.columns.item_id].value_counts()
        maximum = float(counts.max()) if not counts.empty else 1.0
        self.scores = {item_id: float(count / maximum) for item_id, count in counts.items()}
        self.ranked_items = sorted(
            self.items,
            key=lambda item_id: (-self.scores.get(item_id, 0.0), str(item_id)),
        )
        return self

    def fit_from_counts(
        self,
        item_counts: dict[Any, int],
        interactions: pd.DataFrame,
        dataset: DatasetBundle,
    ) -> PopularityModel:
        deadline = None if self.time_limit is None else perf_counter() + self.time_limit
        self._capture_catalog(interactions, dataset)
        if deadline is not None and perf_counter() >= deadline:
            self.early_stopped = True
            raise TimeoutError("Popularity training exceeded its time budget.")
        maximum = float(max(item_counts.values(), default=1))
        self.scores = {
            item_id: float(count / maximum) for item_id, count in item_counts.items()
        }
        self.items = list(item_counts)
        # Rebuild the index so it stays aligned with the full-catalog ``items``
        # (the catalog here spans the whole stream, not just the sample).
        self.item_to_index = {item_id: index for index, item_id in enumerate(self.items)}
        self.ranked_items = sorted(
            self.items,
            key=lambda item_id: (-self.scores.get(item_id, 0.0), str(item_id)),
        )
        return self

    def score_items(self, user_id: Any, item_ids: list[Any]) -> dict[Any, float]:
        del user_id
        return {item_id: self.scores.get(item_id, 0.0) for item_id in item_ids}

    def score_all_users(
        self, users: list[Any], histories: dict[Any, set[Any]]
    ) -> np.ndarray | None:
        del histories
        # Popularity scores are user-independent: broadcast one row per user.
        score_array = np.asarray(
            [self.scores.get(item_id, 0.0) for item_id in self.items], dtype=np.float32
        )
        return np.tile(score_array, (len(users), 1))
