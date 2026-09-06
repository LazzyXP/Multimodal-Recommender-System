"""FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures.

Zhou, Shen, et al. (ACM MM 2023), https://arxiv.org/abs/2211.06957.

Faithful core: the modality item-item graphs are frozen (built once, never
trained), and the user-item interaction graph is denoised by a degree-sensitive
edge-pruning criterion before LightGCN-style propagation and BPR training.
"""

from __future__ import annotations

import torch

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class FREEDOMModel(GraphModelBase):
    name = "FREEDOM"
    requires_modalities = True

    def __init__(self, columns, knn_k: int = 10, denoise_quantile: float = 0.95, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.knn_k = knn_k
        self.denoise_quantile = denoise_quantile

    def _build(self) -> None:
        self._user_embedding = self._embedding(self.n_users)
        self._item_embedding = self._embedding(self.n_items)

        # Denoise the user-item graph with a degree-sensitive criterion: drop
        # edges whose user or item endpoint is a degree hub (above the quantile).
        item_threshold = torch.quantile(
            self._item_degree, self.denoise_quantile, interpolation="nearest"
        )
        user_threshold = torch.quantile(
            self._user_degree, self.denoise_quantile, interpolation="nearest"
        )
        keep = (self._item_degree[self._edge_items] <= item_threshold) & (
            self._user_degree[self._edge_users] <= user_threshold
        )
        edge_users = self._edge_users[keep]
        edge_items = self._edge_items[keep]
        ui_weight = (self._user_degree[edge_users] * self._item_degree[edge_items]).pow(-0.5)
        self._edge_users = self._on_device(edge_users)
        self._edge_items = self._on_device(edge_items)
        self._ui_weight = self._on_device(ui_weight)

        self._frozen_graphs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for block in self._feature_blocks.values():
            features = self._on_device(block)
            src, dst = common.knn_graph(features, self.knn_k)
            weight = common.symmetric_edge_weight(src, dst, self.n_items)
            self._frozen_graphs.append((src, dst, weight))

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self._user_embedding.weight]
        item_layers = [self._item_embedding.weight]
        for _ in range(self.layers):
            next_users = common.propagate(
                item_layers[-1], self._edge_items, self._edge_users, self._ui_weight, self.n_users
            )
            next_items = common.propagate(
                user_layers[-1], self._edge_users, self._edge_items, self._ui_weight, self.n_items
            )
            for src, dst, weight in self._frozen_graphs:
                next_items = next_items + common.propagate(
                    item_layers[-1], src, dst, weight, self.n_items
                )
            user_layers.append(next_users)
            item_layers.append(next_items)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(
            dim=0
        )

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        return self._bpr(users, positives, negatives, user_repr, item_repr)
