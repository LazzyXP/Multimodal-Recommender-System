"""LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.

He, Deng, Wang, Li, Zhang, Wang (SIGIR 2020), https://arxiv.org/abs/2002.02126.

Faithful to the paper: no feature transformation or non-linearity, symmetric
D^{-1/2} A D^{-1/2} normalization over the user-item bipartite graph, K-layer
propagation with uniform layer combination, and BPR training.
"""

from __future__ import annotations

import torch

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class LightGCNModel(GraphModelBase):
    name = "LightGCN"
    requires_modalities = False

    def _build(self) -> None:
        self._user_embedding = self._embedding(self.n_users)
        self._item_embedding = self._embedding(self.n_items)
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

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
            user_layers.append(next_users)
            item_layers.append(next_items)
        users = torch.stack(user_layers, dim=0).mean(dim=0)
        items = torch.stack(item_layers, dim=0).mean(dim=0)
        return users, items

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        return self._bpr(users, positives, negatives, user_repr, item_repr)
