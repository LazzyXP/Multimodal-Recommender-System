"""LATTICE: Mining Latent Structures for Multimedia Recommendation.

Zhang, Zhu, Liu, et al. (ACM MM 2021), https://arxiv.org/abs/2104.09036.

Faithful core: item representations are a projection of the multimodal
features; the original item-item graph is built from these projected features,
while a second, *latent* item-item graph is learned from a separate trainable
item embedding and sparsified to its top-k neighbors. Two GCN views are
propagated and aligned with a contrastive self-supervised term, plus BPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class LATTICEModel(GraphModelBase):
    name = "LATTICE"
    requires_modalities = True

    def __init__(self, columns, knn_k: int = 10, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.knn_k = knn_k

    def _build(self) -> None:
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

        self._user_embedding = self._embedding(self.n_users)
        feature = torch.cat(
            [self._on_device(block) for block in self._feature_blocks.values()], dim=1
        )
        self._feature = feature
        self._feature_projection = nn.Linear(feature.size(1), self.factors, bias=False).to(
            self._device
        )
        nn.init.xavier_uniform_(self._feature_projection.weight)
        # A separate trainable item embedding learns the latent item-item graph.
        self._latent_item = self._embedding(self.n_items)

        projected = self._item_init()
        src, dst = common.knn_graph(projected, self.knn_k)
        self._orig_src, self._orig_dst = src, dst
        self._orig_weight = common.symmetric_edge_weight(src, dst, self.n_items)

    def _item_init(self) -> torch.Tensor:
        return F.normalize(self._feature_projection(self._feature), dim=1)

    def _latent_graph(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = F.normalize(self._latent_item.weight, dim=1)
        src, dst = common.knn_graph(latent, self.knn_k)
        weight = common.symmetric_edge_weight(src, dst, self.n_items)
        return src, dst, weight

    def _propagate(
        self,
        item_src: torch.Tensor,
        item_dst: torch.Tensor,
        item_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self._user_embedding.weight]
        item_layers = [self._item_init()]
        for _ in range(self.layers):
            next_users = common.propagate(
                item_layers[-1], self._edge_items, self._edge_users, self._ui_weight, self.n_users
            )
            next_items = common.propagate(
                user_layers[-1], self._edge_users, self._edge_items, self._ui_weight, self.n_items
            ) + common.propagate(item_layers[-1], item_src, item_dst, item_weight, self.n_items)
            user_layers.append(next_users)
            item_layers.append(next_items)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(
            dim=0
        )

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        view1_user, view1_item = self._propagate(self._orig_src, self._orig_dst, self._orig_weight)
        latent_src, latent_dst, latent_weight = self._latent_graph()
        view2_user, view2_item = self._propagate(latent_src, latent_dst, latent_weight)
        self._view1_user, self._view1_item = view1_user, view1_item
        self._view2_user, self._view2_item = view2_user, view2_item
        return (view1_user + view2_user) / 2, (view1_item + view2_item) / 2

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        bpr = self._bpr(users, positives, negatives, user_repr, item_repr)
        ssl = common.info_nce(self._view1_item, self._view2_item, self.temperature)
        ssl = ssl + common.info_nce(self._view1_user, self._view2_user, self.temperature)
        return bpr + self.ssl_weight * ssl
