"""MGCN: Multi-modal Graph Contrastive Network for Recommendation.

Liu, et al., https://arxiv.org/abs/2308.09605.

Faithful core: a collaborative (behavior) view from the bipartite graph and a
multimodal view augmented with per-modality item-item graphs are contrasted
against each other, with the contrastive positives guided by interaction
co-occurrence, trained jointly with BPR.
"""

from __future__ import annotations

import torch

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class MGCNModel(GraphModelBase):
    name = "MGCN"
    requires_modalities = True

    def __init__(self, columns, knn_k: int = 10, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.knn_k = knn_k

    def _build(self) -> None:
        self._user_embedding = self._embedding(self.n_users)
        self._item_embedding = self._embedding(self.n_items)
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

        self._modality_graphs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for block in self._feature_blocks.values():
            features = self._on_device(block)
            src, dst = common.knn_graph(features, self.knn_k)
            weight = common.symmetric_edge_weight(src, dst, self.n_items)
            self._modality_graphs.append((src, dst, weight))
        self._prepare_behavior_guidance()

    def _prepare_behavior_guidance(self) -> None:
        """Precompute the static user->items grouping for the behavior-guided
        contrastive term.

        The term weights each co-interacting item pair by the inverse squared
        user degree; computing it via per-user scatter aggregation never builds
        a dense ``(n_items x n_items)`` co-occurrence matrix.
        """
        order = torch.argsort(self._edge_users, stable=True)
        self._group_users = self._on_device(self._edge_users[order])
        self._group_items = self._on_device(self._edge_items[order])
        degree = self._user_degree  # m_u, already clamped to >= 1
        weight = degree.pow(-2.0)  # 1 / degree(u)^2
        self._guidance_user_weight = self._on_device(weight)
        self._guidance_den = float((weight * (degree * degree - degree)).sum().item())

    def _behavior_guidance(
        self, collaborative_item: torch.Tensor, multimodal_item: torch.Tensor
    ) -> torch.Tensor:
        """Mean cross-view similarity over co-interacted item pairs (off-diagonal)."""
        c = collaborative_item[self._group_items]
        m = multimodal_item[self._group_items]
        users = self._group_users
        dim = collaborative_item.size(1)
        c_sum = torch.zeros(self.n_users, dim, device=c.device).index_add_(0, users, c)
        m_sum = torch.zeros(self.n_users, dim, device=c.device).index_add_(0, users, m)
        diag = torch.zeros(self.n_users, device=c.device).index_add_(0, users, (c * m).sum(dim=1))
        numerator = (self._guidance_user_weight * ((c_sum * m_sum).sum(dim=1) - diag)).sum()
        return numerator / max(self._guidance_den, 1.0)

    def _propagate(self, multimodal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self._user_embedding.weight]
        item_layers = [self._item_embedding.weight]
        for _ in range(self.layers):
            next_users = common.propagate(
                item_layers[-1], self._edge_items, self._edge_users, self._ui_weight, self.n_users
            )
            next_items = common.propagate(
                user_layers[-1], self._edge_users, self._edge_items, self._ui_weight, self.n_items
            )
            if multimodal:
                for src, dst, weight in self._modality_graphs:
                    next_items = next_items + common.propagate(
                        item_layers[-1], src, dst, weight, self.n_items
                    )
            user_layers.append(next_users)
            item_layers.append(next_items)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(
            dim=0
        )

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        collaborative_user, collaborative_item = self._propagate(multimodal=False)
        multimodal_user, multimodal_item = self._propagate(multimodal=True)
        self._collaborative_item = collaborative_item
        self._multimodal_item = multimodal_item
        return (collaborative_user + multimodal_user) / 2, (
            collaborative_item + multimodal_item
        ) / 2

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        bpr = self._bpr(users, positives, negatives, user_repr, item_repr)
        ssl = common.info_nce(self._collaborative_item, self._multimodal_item, self.temperature)
        # Behavior guidance: co-interacted items should align across the two views.
        guidance = self._behavior_guidance(self._collaborative_item, self._multimodal_item)
        return bpr + self.ssl_weight * (ssl - guidance)
