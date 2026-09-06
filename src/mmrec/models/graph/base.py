"""Shared training harness for paper-specific graph recommenders.

Each concrete model implements ``_build`` (parameters and graph structures),
``_forward`` (embedding propagation) and ``_training_loss`` (BPR plus any
paper-specific self-supervised term). Everything else -- data preparation,
batching, device management, early stopping, persistence and evaluation -- is
shared and identical across models, so the differences between papers live
exactly where the papers differ.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from mmrec.data import DatasetBundle
from mmrec.features import encode_item_feature_blocks
from mmrec.models.base import BaseRecommendationModel
from mmrec.models.graph import common


def torch_available() -> bool:
    """Return whether the optional PyTorch runtime can be imported."""
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


class GraphModelBase(BaseRecommendationModel):
    """Base class for the torch graph recommendation models."""

    requires_modalities = False

    def __init__(
        self,
        columns,
        factors: int = 64,
        layers: int = 2,
        epochs: int = 50,
        learning_rate: float = 1e-2,
        regularization: float = 1e-4,
        batch_size: int = 4096,
        temperature: float = 0.5,
        ssl_weight: float = 0.1,
        random_state: int = 42,
        device: str = "auto",
        time_limit: float | None = None,
    ) -> None:
        super().__init__(columns)
        self.factors = factors
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.temperature = temperature
        self.ssl_weight = ssl_weight
        self.random_state = random_state
        self.device_name = device
        self.time_limit = time_limit
        self.early_stopped = False

    # -- public contract ----------------------------------------------------

    def fit(
        self,
        interactions: pd.DataFrame,
        dataset: DatasetBundle | None = None,
    ) -> GraphModelBase:
        if self.requires_modalities and (
            dataset is None or dataset.items is None or not dataset.modalities.get("item", {})
        ):
            raise ValueError(f"{self.name} requires declared item modalities.")

        self._capture_catalog(interactions, dataset)
        self._prepare(interactions, dataset)
        self._device = self._resolve_device()
        self._build()

        torch.manual_seed(self.random_state)
        parameters = self._parameters()
        # L2 regularization is applied explicitly in the loss (matching the
        # papers' BPR objective) rather than via Adam weight decay.
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        rng = torch.Generator(device="cpu").manual_seed(self.random_state)

        edge_users = self._edge_users
        deadline = None if self.time_limit is None else perf_counter() + self.time_limit

        batch_size = max(1, min(self.batch_size, edge_users.size(0)))
        for _ in range(self.epochs):
            if deadline is not None and perf_counter() >= deadline:
                self.early_stopped = True
                break
            order = torch.randperm(edge_users.size(0), generator=rng)
            for start in range(0, edge_users.size(0), batch_size):
                if deadline is not None and perf_counter() >= deadline:
                    self.early_stopped = True
                    break
                batch = order[start : start + batch_size]
                user_repr, item_repr = self._forward()
                users = edge_users[batch]
                positives_batch = self._edge_items[batch]
                negatives = common.sample_negatives(
                    users.cpu(), self.n_items, self._positive_mask, rng
                )
                loss = self._training_loss(users, positives_batch, negatives, user_repr, item_repr)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
                self._after_step()
            if self.early_stopped:
                break

        with torch.no_grad():
            user_repr, item_repr = self._forward()
            self._user_repr = user_repr.detach().cpu().numpy()
            self._item_repr = item_repr.detach().cpu().numpy()

        counts = interactions[self.columns.item_id].value_counts()
        maximum = float(counts.max()) if not counts.empty else 1.0
        self.popularity = {item_id: float(count / maximum) for item_id, count in counts.items()}
        return self

    def score_all_items(self, user_id: Any, history: set[Any]) -> np.ndarray | None:
        del history
        user_index = self.user_indices.get(user_id)
        if user_index is None:
            return None
        return self._item_repr @ self._user_repr[user_index]

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
        return (self._user_repr[user_indices] @ self._item_repr.T).astype(np.float32)

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
        indices = np.asarray([index for _, index in known], dtype=np.int64)
        scores = self._item_repr[indices] @ self._user_repr[user_index]
        values = dict(zip((item_id for item_id, _ in known), scores.tolist(), strict=True))
        return {item_id: float(values.get(item_id, 0.0)) for item_id in item_ids}

    def __getstate__(self) -> dict[str, Any]:
        """Persist portable inference arrays instead of device-bound tensors."""
        state = self.__dict__.copy()
        for key in list(state):
            if key.startswith("_") and key not in {"_user_repr", "_item_repr"}:
                state.pop(key, None)
        return state

    # -- subclass hooks -----------------------------------------------------

    def _build(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _after_step(self) -> None:
        """Optional per-optimizer-step hook (e.g. momentum/EMA updates)."""

    def _parameters(self) -> list[torch.Tensor]:
        collected: list[torch.Tensor] = []
        for key, value in self.__dict__.items():
            if key.startswith("_target"):  # momentum/EMA modules are updated separately
                continue
            collected.extend(self._iter_tensors(value))
        unique: list[torch.Tensor] = []
        seen: set[int] = set()
        for parameter in collected:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique.append(parameter)
        return unique

    def _iter_tensors(self, value: Any) -> list[torch.Tensor]:
        if isinstance(value, torch.nn.Module):
            return list(value.parameters())
        if isinstance(value, torch.nn.Parameter):
            return [value]
        if isinstance(value, torch.Tensor) and value.requires_grad:
            return [value]
        if isinstance(value, (list, tuple)):
            result: list[torch.Tensor] = []
            for item in value:
                result.extend(self._iter_tensors(item))
            return result
        if isinstance(value, dict):
            result = []
            for item in value.values():
                result.extend(self._iter_tensors(item))
            return result
        return []

    # -- shared data preparation --------------------------------------------

    def _prepare(self, interactions: pd.DataFrame, dataset: DatasetBundle | None) -> None:
        self.user_ids = interactions[self.columns.user_id].drop_duplicates().tolist()
        self.user_indices = {user_id: i for i, user_id in enumerate(self.user_ids)}
        self.item_indices = self.item_to_index
        self.n_users = len(self.user_ids)
        self.n_items = len(self.items)

        item_ids = interactions[self.columns.item_id]
        mask = item_ids.isin(self.item_indices).to_numpy()
        user_array = (
            interactions[self.columns.user_id][mask]
            .map(self.user_indices)
            .to_numpy(dtype=np.int64, copy=True)
        )
        item_array = item_ids[mask].map(self.item_indices).to_numpy(dtype=np.int64, copy=True)
        self._edge_users = torch.as_tensor(user_array, dtype=torch.long)
        self._edge_items = torch.as_tensor(item_array, dtype=torch.long)
        self._user_degree = (
            torch.bincount(self._edge_users, minlength=self.n_users).clamp_min(1).float()
        )
        self._item_degree = (
            torch.bincount(self._edge_items, minlength=self.n_items).clamp_min(1).float()
        )
        self._ui_weight = (
            self._user_degree[self._edge_users] * self._item_degree[self._edge_items]
        ).pow(-0.5)
        self._positive_mask = common.build_positive_mask(
            self._edge_users, self._edge_items, self.n_users, self.n_items
        )

        self._feature_blocks: dict[str, torch.Tensor] = {}
        if dataset is not None and dataset.items is not None and dataset.modalities.get("item", {}):
            _, blocks = encode_item_feature_blocks(
                dataset.items, self.columns.item_id, dataset.modalities["item"]
            )
            item_ids_ordered = dataset.items[self.columns.item_id].tolist()
            for name, matrix in blocks.items():
                by_item = {item_id: matrix[i] for i, item_id in enumerate(item_ids_ordered)}
                stacked = np.stack(
                    [by_item.get(item_id, np.zeros(matrix.shape[1])) for item_id in self.items]
                ).astype(np.float32)
                self._feature_blocks[name] = torch.as_tensor(stacked)

    def _resolve_device(self) -> torch.device:
        requested = self.device_name.lower()
        if requested != "auto":
            if requested.startswith("cuda") and not torch.cuda.is_available():
                torch_version = getattr(torch, "__version__", "unknown")
                compiled_cuda = getattr(torch.version, "cuda", None) or "none"
                raise ValueError(
                    "CUDA was requested but is not available. "
                    f"Installed torch={torch_version} (CUDA build={compiled_cuda}). "
                    "Install a wheel compatible with the host driver, or use device='cpu'."
                )
            if requested == "mps" and not getattr(torch.backends, "mps", None).is_available():
                raise ValueError("MPS was requested but is not available.")
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _embedding(self, count: int) -> torch.nn.Embedding:
        embedding = torch.nn.Embedding(count, self.factors, device=self._device)
        torch.nn.init.xavier_uniform_(embedding.weight)
        return embedding

    def _on_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device)

    def _bpr(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        bpr = common.bpr_loss(user_repr, item_repr, users, positives, negatives)
        return bpr + self._regularization_loss()

    def _regularization_loss(self) -> torch.Tensor:
        """Per-sample L2 on the 0-th layer embeddings (LightGCN Eq. 7).

        Uses the *mean* per-row squared norm, not the full-matrix Frobenius
        norm, so the term stays at the paper's intended scale and does not
        dominate the BPR loss (which collapses embeddings to zero).
        """
        total: torch.Tensor | None = None
        for name in (
            "_user_embedding",
            "_item_embedding",
            "_global_item_embedding",
            "_latent_item",
        ):
            embedding = getattr(self, name, None)
            if embedding is not None and hasattr(embedding, "weight"):
                norm = embedding.weight.norm(dim=1).pow(2).mean()
                total = norm if total is None else total + norm
        if total is None:
            return torch.zeros((), device=self._device)
        return self.regularization * total
