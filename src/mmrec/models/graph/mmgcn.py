"""MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation.

Wei, Cheng, Zhu, Nie (ACM MM 2019), https://arxiv.org/abs/2002.10872.

Faithful to the paper: for every modality a separate item-item graph is built
from that modality's feature similarities; each modality maintains its own
user and item embeddings (item embeddings are a projection of the modality
features), propagates over the user-item graph plus its own item-item graph,
and the per-modality user/item representations are fused with a learned
attention before BPR training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class MMGCNModel(GraphModelBase):
    name = "MMGCN"
    requires_modalities = True

    def __init__(self, columns, knn_k: int = 10, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.knn_k = knn_k

    def _build(self) -> None:
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

        self._modality_names = sorted(self._feature_blocks)
        # Per-modality user embeddings and item feature projections.
        self._user_embeddings = nn.ModuleList(
            [self._embedding(self.n_users) for _ in self._modality_names]
        )
        self._item_projections = nn.ModuleList(
            [
                nn.Linear(self._feature_blocks[name].size(1), self.factors, bias=False).to(
                    self._device
                )
                for name in self._modality_names
            ]
        )
        for projection in self._item_projections:
            nn.init.xavier_uniform_(projection.weight)

        # Per-modality item-item graphs built from that modality's features.
        self._modality_graphs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for name in self._modality_names:
            features = self._on_device(self._feature_blocks[name])
            src, dst = common.knn_graph(features, self.knn_k)
            weight = common.symmetric_edge_weight(src, dst, self.n_items)
            self._modality_graphs.append((src, dst, weight))

        self._user_preference = nn.Parameter(torch.empty(self.factors, device=self._device))
        self._item_preference = nn.Parameter(torch.empty(self.factors, device=self._device))
        nn.init.xavier_uniform_(self._user_preference.unsqueeze(0))
        nn.init.xavier_uniform_(self._item_preference.unsqueeze(0))

    def _modality_forward(
        self,
        user_embedding: nn.Embedding,
        item_projection: nn.Linear,
        features: torch.Tensor,
        graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst, weight = graph
        item_init = item_projection(features)
        user_layers = [user_embedding.weight]
        item_layers = [item_init]
        for _ in range(self.layers):
            next_users = common.propagate(
                item_layers[-1], self._edge_items, self._edge_users, self._ui_weight, self.n_users
            )
            next_items = common.propagate(
                user_layers[-1], self._edge_users, self._edge_items, self._ui_weight, self.n_items
            )
            next_items = next_items + common.propagate(
                item_layers[-1], src, dst, weight, self.n_items
            )
            user_layers.append(next_users)
            item_layers.append(next_items)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(
            dim=0
        )

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_views: list[torch.Tensor] = []
        item_views: list[torch.Tensor] = []
        for index, name in enumerate(self._modality_names):
            user_view, item_view = self._modality_forward(
                self._user_embeddings[index],
                self._item_projections[index],
                self._on_device(self._feature_blocks[name]),
                self._modality_graphs[index],
            )
            user_views.append(user_view)
            item_views.append(item_view)
        users = self._attention_fuse(user_views, self._user_preference)
        items = self._attention_fuse(item_views, self._item_preference)
        return users, items

    def _attention_fuse(self, views: list[torch.Tensor], preference: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(views, dim=0)  # (M, N, D)
        scores = torch.einsum("mnd,d->mn", stacked, preference)
        weights = torch.softmax(scores / (self.factors**0.5), dim=0)
        return (stacked * weights.unsqueeze(-1)).sum(dim=0)

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        return self._bpr(users, positives, negatives, user_repr, item_repr)
