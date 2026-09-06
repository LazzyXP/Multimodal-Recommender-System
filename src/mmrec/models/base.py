"""Base contract shared by recommendation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from mmrec.config import ColumnConfig
from mmrec.data import DatasetBundle


def _top_k_items(
    items: list[Any],
    scores: Any,
    item_to_index: dict[Any, int],
    seen: set[Any],
    k: int,
) -> tuple[list[Any], list[float]]:
    """Return the top-k non-seen items and their scores using vectorized NumPy ranking."""
    score_array = np.asarray(scores, dtype=np.float32)
    score_array = np.where(np.isnan(score_array), -np.inf, score_array)
    if seen:
        score_array = score_array.copy()
        for item_id in seen:
            index = item_to_index.get(item_id)
            if index is not None:
                score_array[index] = -np.inf
    count = len(score_array)
    if count == 0:
        return [], []
    if k >= count:
        order = np.argsort(-score_array, kind="stable")
    else:
        order = np.argpartition(-score_array, k - 1)[:k]
        order = order[np.argsort(-score_array[order], kind="stable")]

    result_items: list[Any] = []
    result_scores: list[float] = []
    for index in order:
        score = float(score_array[index])
        if np.isneginf(score):
            continue
        result_items.append(items[index])
        result_scores.append(score)
        if len(result_items) >= k:
            break
    return result_items, result_scores


class BaseRecommendationModel(ABC):
    name: str
    # Upper bound for a batched (users x catalog) score matrix. The predictor
    # can override this per run; keeping a conservative default prevents a
    # recommendation call from competing with model tensors for GPU memory.
    max_score_bytes: int = 64 * 1024 * 1024

    def __init__(self, columns: ColumnConfig) -> None:
        self.columns = columns
        self.items: list[Any] = []
        self.item_to_index: dict[Any, int] = {}
        self.seen_by_user: dict[Any, set[Any]] = {}

    @abstractmethod
    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> BaseRecommendationModel:
        """Fit the model and return itself."""

    @abstractmethod
    def score_items(self, user_id: Any, item_ids: list[Any]) -> dict[Any, float]:
        """Return a score for each requested item."""

    def score_all_items(self, user_id: Any, history: set[Any]) -> np.ndarray | None:
        """Return a score for every catalog item in ``self.items`` order.

        Models that can score the full catalog vectorized override this method.
        ``history`` is the user's interaction history (possibly the exact
        streaming override) and is used by content-based models to build a
        profile. Returning ``None`` falls back to the per-item ``score_items``
        path.
        """
        del user_id, history
        return None

    def score_all_users(
        self, users: list[Any], histories: dict[Any, set[Any]]
    ) -> np.ndarray | None:
        """Return a ``(n_users, n_items)`` score matrix, or ``None`` if unsupported.

        Latent-factor models override this so recommendation can score every user
        in one BLAS matmul plus one batched top-k, instead of one full-catalog
        pass per user. Content-based models use ``histories`` to build a profile
        per user. The row-major layout keeps each user's scores contiguous in
        memory, which is cache-friendly for the subsequent top-k.
        """
        del users, histories
        return None

    def score_items_with_history(
        self,
        user_id: Any,
        item_ids: list[Any],
        history: set[Any],
    ) -> dict[Any, float]:
        del history
        return self.score_items(user_id, item_ids)

    def recommend(
        self,
        users: list[Any],
        k: int,
        exclude_seen: bool = True,
        seen_override: dict[Any, set[Any]] | None = None,
    ) -> pd.DataFrame:
        histories = seen_override if seen_override is not None else self.seen_by_user

        # Batched scoring runs in user chunks so the (n_users, n_items) score
        # matrix stays memory-bounded on large catalogs (~256 MiB per chunk).
        chunk = max(1, self.max_score_bytes // max(1, len(self.items) * 4))
        if users and self.score_all_users(users[:1], histories) is not None:
            frames: list[pd.DataFrame] = []
            for start in range(0, len(users), chunk):
                chunk_users = users[start : start + chunk]
                scores = self.score_all_users(chunk_users, histories)
                if scores is None:
                    break
                frames.append(
                    self._recommend_batch(chunk_users, k, exclude_seen, histories, scores)
                )
            else:
                return pd.concat(frames, ignore_index=True)

        rows: list[dict[str, Any]] = []
        user_col = self.columns.user_id
        item_col = self.columns.item_id
        for user_id in users:
            seen = histories.get(user_id, set()) if exclude_seen else set()
            all_scores = self.score_all_items(user_id, histories.get(user_id, set()))
            if all_scores is not None:
                ranked_items, ranked_scores = _top_k_items(
                    self.items, all_scores, self.item_to_index, seen, k
                )
                rows.extend(
                    {
                        user_col: user_id,
                        item_col: item_id,
                        "rank": rank,
                        "score": score,
                        "model": self.name,
                    }
                    for rank, (item_id, score) in enumerate(
                        zip(ranked_items, ranked_scores, strict=True), start=1
                    )
                )
                continue

            candidates = [item_id for item_id in self.items if item_id not in seen]
            scores = self.score_items_with_history(
                user_id,
                candidates,
                histories.get(user_id, set()),
            )
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))[:k]
            rows.extend(
                {
                    user_col: user_id,
                    item_col: item_id,
                    "rank": rank,
                    "score": float(score),
                    "model": self.name,
                }
                for rank, (item_id, score) in enumerate(ranked, start=1)
            )
        return pd.DataFrame(rows, columns=[user_col, item_col, "rank", "score", "model"])

    def _recommend_batch(
        self,
        users: list[Any],
        k: int,
        exclude_seen: bool,
        histories: dict[Any, set[Any]],
        scores: np.ndarray,
    ) -> pd.DataFrame:
        """Batched top-k over a ``(n_users, n_items)`` score matrix."""
        score_array = np.asarray(scores, dtype=np.float32)
        n_users, n_items = score_array.shape
        masked = np.zeros(n_users, dtype=np.int64)
        if exclude_seen:
            for row, user_id in enumerate(users):
                for item_id in histories.get(user_id, ()):
                    index = self.item_to_index.get(item_id)
                    if index is not None:
                        score_array[row, index] = -np.inf
                        masked[row] += 1
        limit = min(k, n_items)
        if limit <= 0:
            return pd.DataFrame(
                columns=[
                    self.columns.user_id,
                    self.columns.item_id,
                    "rank",
                    "score",
                    "model",
                ]
            )
        order = np.argpartition(-score_array, limit - 1, axis=1)[:, :limit]
        top = np.take_along_axis(score_array, order, axis=1)
        order = np.take_along_axis(order, np.argsort(-top, axis=1, kind="stable"), axis=1)

        rows: list[dict[str, Any]] = []
        for row, user_id in enumerate(users):
            # Seens are masked to -inf and sort last, so only the first
            # (n_items - masked) positions are finite; skip the rest without
            # a per-element isinf/isnan call.
            valid = min(limit, n_items - int(masked[row]))
            for position in range(valid):
                index = int(order[row, position])
                rows.append(
                    {
                        self.columns.user_id: user_id,
                        self.columns.item_id: self.items[index],
                        "rank": position + 1,
                        "score": float(score_array[row, index]),
                        "model": self.name,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[self.columns.user_id, self.columns.item_id, "rank", "score", "model"],
        )

    def _capture_catalog(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> None:
        user_col = self.columns.user_id
        item_col = self.columns.item_id
        if dataset is not None and dataset.items is not None:
            self.items = dataset.items[item_col].drop_duplicates().tolist()
        else:
            self.items = interactions[item_col].drop_duplicates().tolist()
        self.item_to_index = {item_id: index for index, item_id in enumerate(self.items)}
        self.seen_by_user = {
            user_id: set(group[item_col].tolist())
            for user_id, group in interactions.groupby(user_col, sort=False)
        }
