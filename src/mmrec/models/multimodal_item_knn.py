"""Content-based item recommender backed by multimodal feature vectors.

Scoring defaults to an exact vectorized inner product (BLAS), which is optimal
for exact recall. When ``use_ann=True`` and the optional ``faiss-cpu`` package
is installed, ``ann_backend="flat"`` uses exact IndexFlatIP and
``ann_backend="hnsw"`` enables approximate HNSW retrieval and
``ann_backend="ivf"`` enables trained IVF retrieval; ``"ivfpq"`` adds
product-quantized compression for large catalogs.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from mmrec.data import DatasetBundle
from mmrec.features import encode_item_features
from mmrec.models.base import BaseRecommendationModel


class MultiModalItemKNNModel(BaseRecommendationModel):
    name = "MultiModalItemKNN"

    def __init__(
        self,
        columns,
        use_ann: bool = False,
        ann_topk: int = 100,
        ann_backend: str = "flat",
        ann_hnsw_m: int = 32,
        ann_ef_search: int = 64,
        ann_nlist: int = 64,
        ann_nprobe: int = 8,
        ann_pq_m: int = 8,
        ann_pq_nbits: int = 8,
    ) -> None:
        super().__init__(columns)
        if ann_topk <= 0:
            raise ValueError("ann_topk must be positive.")
        if ann_backend not in {"flat", "hnsw", "ivf", "ivfpq"}:
            raise ValueError("ann_backend must be 'flat', 'hnsw', 'ivf', or 'ivfpq'.")
        if ann_hnsw_m <= 0:
            raise ValueError("ann_hnsw_m must be positive.")
        if ann_ef_search <= 0:
            raise ValueError("ann_ef_search must be positive.")
        if ann_nlist <= 0:
            raise ValueError("ann_nlist must be positive.")
        if ann_nprobe <= 0:
            raise ValueError("ann_nprobe must be positive.")
        if ann_pq_m <= 0:
            raise ValueError("ann_pq_m must be positive.")
        if ann_pq_nbits <= 0:
            raise ValueError("ann_pq_nbits must be positive.")
        self.use_ann = use_ann
        self.ann_topk = ann_topk
        self.ann_backend = ann_backend
        self.ann_hnsw_m = ann_hnsw_m
        self.ann_ef_search = ann_ef_search
        self.ann_nlist = ann_nlist
        self.ann_nprobe = ann_nprobe
        self.ann_pq_m = ann_pq_m
        self.ann_pq_nbits = ann_pq_nbits
        self.effective_ann_pq_nbits = ann_pq_nbits
        self._ann_index: Any = None
        self.user_vectors: dict[Any, np.ndarray] = {}

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
        user_modalities = dataset.modalities.get("user", {})
        if user_modalities and dataset.users is not None:
            user_ids, user_matrix = encode_item_features(
                dataset.users,
                self.columns.user_id,
                user_modalities,
            )
            if user_matrix.shape[1] != matrix.shape[1]:
                raise ValueError(
                    "user and item modality encodings must have the same feature dimension "
                    "for MultiModalItemKNN cold-start scoring."
                )
            self.user_vectors = {
                user_id: user_matrix[index] for index, user_id in enumerate(user_ids)
            }
        counts = interactions[self.columns.item_id].value_counts()
        maximum = float(counts.max()) if not counts.empty else 1.0
        self.popularity = {item_id: float(count / maximum) for item_id, count in counts.items()}
        self._ann_index = self._build_ann_index() if self.use_ann else None
        return self

    def _build_ann_index(self) -> Any:
        try:
            import faiss  # type: ignore[import-not-found]

            vectors = self.feature_matrix.astype(np.float32)
            if self.ann_backend == "hnsw":
                index = faiss.IndexHNSWFlat(
                    self.feature_matrix.shape[1], self.ann_hnsw_m, faiss.METRIC_INNER_PRODUCT
                )
                index.hnsw.efSearch = max(self.ann_ef_search, self.ann_topk)
            elif self.ann_backend in {"ivf", "ivfpq"}:
                nlist = min(self.ann_nlist, len(vectors))
                quantizer = faiss.IndexFlatIP(self.feature_matrix.shape[1])
                if self.ann_backend == "ivfpq":
                    if self.feature_matrix.shape[1] % self.ann_pq_m != 0:
                        raise ValueError(
                            "feature dimension must be divisible by ann_pq_m for ivfpq."
                        )
                    effective_nbits = min(self.ann_pq_nbits, max(1, int(np.log2(len(vectors)))))
                    self.effective_ann_pq_nbits = effective_nbits
                    if effective_nbits < self.ann_pq_nbits:
                        warnings.warn(
                            f"ivfpq reduced ann_pq_nbits from {self.ann_pq_nbits} to "
                            f"{effective_nbits} because the catalog has only {len(vectors)} "
                            "training vectors.",
                            UserWarning,
                            stacklevel=2,
                        )
                    index = faiss.IndexIVFPQ(
                        quantizer,
                        self.feature_matrix.shape[1],
                        nlist,
                        self.ann_pq_m,
                        effective_nbits,
                        faiss.METRIC_INNER_PRODUCT,
                    )
                else:
                    index = faiss.IndexIVFFlat(
                        quantizer, self.feature_matrix.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
                    )
                index.train(vectors)
                index.nprobe = min(self.ann_nprobe, nlist)
            else:
                index = faiss.IndexFlatIP(self.feature_matrix.shape[1])
            index.add(vectors)
            return index
        except ImportError:
            warnings.warn(
                f"use_ann=True with ann_backend={self.ann_backend!r} requested but faiss-cpu "
                "is unavailable; falling back to "
                "NumPy exact inner-product scoring.",
                UserWarning,
                stacklevel=2,
            )
            return None

    def score_all_items(self, user_id: Any, history: set[Any]) -> np.ndarray | None:
        history_vectors = [
            self.item_vectors[item_id]
            for item_id in history
            if item_id in self.item_vectors
        ]
        if history_vectors:
            profile_vectors = history_vectors
        elif user_id in self.user_vectors:
            profile_vectors = [self.user_vectors[user_id]]
        else:
            return None
        profile = np.mean(profile_vectors, axis=0)
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
        """Batch profiles through FAISS when enabled, otherwise use one BLAS matmul."""
        profiles = []
        for user_id in users:
            history_vectors = [
                self.item_vectors[item_id]
                for item_id in histories.get(user_id, ())
                if item_id in self.item_vectors
            ]
            if history_vectors:
                profile_vectors = history_vectors
            elif user_id in self.user_vectors:
                profile_vectors = [self.user_vectors[user_id]]
            else:
                return None
            profile = np.mean(profile_vectors, axis=0)
            norm = np.linalg.norm(profile)
            if norm == 0:
                return None
            profiles.append(profile / norm)
        if self._ann_index is not None:
            scores = np.zeros((len(profiles), len(self.items)), dtype=np.float32)
            values, indices = self._ann_index.search(
                np.asarray(profiles, dtype=np.float32), self.ann_topk
            )
            valid = indices >= 0
            rows = np.broadcast_to(np.arange(len(profiles))[:, None], indices.shape)
            scores[rows[valid], indices[valid]] = values[valid]
            return scores
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
        history_vectors = [
            self.item_vectors[item_id]
            for item_id in history
            if item_id in self.item_vectors
        ]
        if history_vectors:
            profile_vectors = history_vectors
        elif user_id in self.user_vectors:
            profile_vectors = [self.user_vectors[user_id]]
        else:
            return {item: self.popularity.get(item, 0.0) for item in item_ids}

        profile = np.mean(profile_vectors, axis=0)
        profile_norm = np.linalg.norm(profile)
        if profile_norm:
            profile = profile / profile_norm
        known_items = [item_id for item_id in item_ids if item_id in self.item_indices]
        known_indices = [self.item_indices[item_id] for item_id in known_items]
        similarities = self.feature_matrix[known_indices] @ profile
        scores = dict(zip(known_items, similarities.tolist(), strict=True))
        return {item_id: float(scores.get(item_id, 0.0)) for item_id in item_ids}
