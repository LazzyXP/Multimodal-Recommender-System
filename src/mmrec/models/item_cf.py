"""Cosine-normalized item-item collaborative filtering."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from mmrec.data import DatasetBundle
from mmrec.models.base import BaseRecommendationModel


class ItemCFModel(BaseRecommendationModel):
    name = "ItemCF"

    def __init__(
        self,
        columns,
        max_neighbors: int = 200,
        max_history_per_user: int = 200,
        time_limit: float | None = None,
    ) -> None:
        super().__init__(columns)
        self.max_neighbors = max_neighbors
        self.max_history_per_user = max_history_per_user
        self.time_limit = time_limit
        self.early_stopped = False

    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> ItemCFModel:
        deadline = None if self.time_limit is None else perf_counter() + self.time_limit
        self._capture_catalog(interactions, dataset)
        item_col = self.columns.item_id
        user_col = self.columns.user_id
        item_counts: Counter[Any] = Counter()
        # Co-occurrence uses integer item codes (fast dict hashing) and a
        # defaultdict so a Counter is only allocated on first access, not on
        # every ``setdefault`` call. Dedup keeps ``dict.fromkeys`` (C speed).
        cooccurrence: defaultdict[int, Counter[int]] = defaultdict(Counter)

        ordered = interactions
        if self.columns.timestamp:
            ordered = interactions.sort_values(self.columns.timestamp, kind="stable")
        for _, group in ordered.groupby(user_col, sort=False):
            if deadline is not None and perf_counter() >= deadline:
                self.early_stopped = True
                raise TimeoutError("ItemCF training exceeded its time budget.")
            user_items = list(dict.fromkeys(group[item_col].tolist()))[
                -self.max_history_per_user :
            ]
            if not user_items:
                continue
            item_counts.update(user_items)
            if len(user_items) < 2:
                continue
            codes = np.fromiter(
                (self.item_to_index[item_id] for item_id in user_items),
                dtype=np.int64,
                count=len(user_items),
            )
            left, right = np.triu_indices(len(codes), k=1)
            for left_code, right_code in zip(codes[left], codes[right], strict=True):
                left_key = int(left_code)
                right_key = int(right_code)
                cooccurrence[left_key][right_key] += 1
                cooccurrence[right_key][left_key] += 1

        self.similarities: dict[Any, dict[Any, float]] = {}
        for left, neighbors in cooccurrence.items():
            if deadline is not None and perf_counter() >= deadline:
                self.early_stopped = True
                raise TimeoutError("ItemCF training exceeded its time budget.")
            left_id = self.items[left]
            similarities = {
                self.items[right]: count
                / sqrt(item_counts[left_id] * item_counts[self.items[right]])
                for right, count in neighbors.items()
            }
            self.similarities[left_id] = dict(
                sorted(similarities.items(), key=lambda pair: pair[1], reverse=True)[
                    : self.max_neighbors
                ]
            )
        maximum = max(item_counts.values(), default=1)
        self.popularity = {item: count / maximum for item, count in item_counts.items()}
        self.popularity_ranked = sorted(
            self.items,
            key=lambda item: (-self.popularity.get(item, 0.0), str(item)),
        )
        return self

    def score_items(self, user_id: Any, item_ids: list[Any]) -> dict[Any, float]:
        history = self.seen_by_user.get(user_id, set())
        if not history:
            return {item: float(self.popularity.get(item, 0.0)) for item in item_ids}

        scores: dict[Any, float] = {}
        for candidate in item_ids:
            collaborative_score = sum(
                self.similarities.get(history_item, {}).get(candidate, 0.0)
                for history_item in history
            )
            # A tiny popularity prior makes sparse ties deterministic and useful.
            scores[candidate] = collaborative_score + 1e-6 * self.popularity.get(candidate, 0.0)
        return scores

    def recommend(
        self,
        users: list[Any],
        k: int,
        exclude_seen: bool = True,
        seen_override: dict[Any, set[Any]] | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        user_col = self.columns.user_id
        item_col = self.columns.item_id
        for user_id in users:
            histories = seen_override if seen_override is not None else self.seen_by_user
            seen = histories.get(user_id, set()) if exclude_seen else set()
            candidate_scores: Counter[Any] = Counter()
            for history_item in histories.get(user_id, set()):
                candidate_scores.update(self.similarities.get(history_item, {}))
            for item_id in seen:
                candidate_scores.pop(item_id, None)

            ranked_pairs = sorted(
                candidate_scores.items(),
                key=lambda pair: (-pair[1], str(pair[0])),
            )[:k]
            selected = {item_id for item_id, _ in ranked_pairs}
            if len(ranked_pairs) < k:
                for item_id in self.popularity_ranked:
                    if item_id in seen or item_id in selected:
                        continue
                    ranked_pairs.append((item_id, 1e-6 * self.popularity.get(item_id, 0.0)))
                    selected.add(item_id)
                    if len(ranked_pairs) == k:
                        break
            rows.extend(
                {
                    user_col: user_id,
                    item_col: item_id,
                    "rank": rank,
                    "score": float(score),
                    "model": self.name,
                }
                for rank, (item_id, score) in enumerate(ranked_pairs, start=1)
            )
        return pd.DataFrame(rows, columns=[user_col, item_col, "rank", "score", "model"])
