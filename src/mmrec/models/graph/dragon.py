"""DRAGON: Dual Graph Enhanced Embedding for Recommendation.

Zhou, et al., https://arxiv.org/abs/2305.14241.

Faithful core: besides the user-item bipartite graph, two homogeneous graphs --
a user-user graph (from interaction co-occurrence) and an item-item graph (from
features) -- are built and jointly propagated; a contrastive self-supervised
term aligns the bipartite view with the dual-graph view, trained with BPR.
"""

from __future__ import annotations

import torch

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class DRAGONModel(GraphModelBase):
    name = "DRAGON"
    requires_modalities = True

    def __init__(self, columns, knn_k: int = 10, degree_cap: int = 500, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.knn_k = knn_k
        self.degree_cap = degree_cap

    def _build(self) -> None:
        self._user_embedding = self._embedding(self.n_users)
        self._item_embedding = self._embedding(self.n_items)
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

        features = torch.cat(
            [self._on_device(block) for block in self._feature_blocks.values()], dim=1
        )
        item_src, item_dst = common.knn_graph(features, self.knn_k)
        self._item_src, self._item_dst = item_src, item_dst
        self._item_weight = common.symmetric_edge_weight(item_src, item_dst, self.n_items)
        self._user_src, self._user_dst = self._build_user_graph()
        self._user_weight = common.symmetric_edge_weight(
            self._user_src, self._user_dst, self.n_users
        )

    def _build_user_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Co-occurrence among users through items, sparsified to the top-k
        # neighbors. Enumerated from the edge list so the dense
        # (n_users x n_items) incidence matrix is never materialized.
        src, dst = common.cooccurrence_topk(
            self._edge_users,
            self._edge_items,
            self.n_users,
            self.n_items,
            self.knn_k,
            degree_cap=self.degree_cap,
        )
        return self._on_device(src), self._on_device(dst)

    def _propagate(self, use_dual: bool) -> tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self._user_embedding.weight]
        item_layers = [self._item_embedding.weight]
        for _ in range(self.layers):
            next_users = common.propagate(
                item_layers[-1], self._edge_items, self._edge_users, self._ui_weight, self.n_users
            )
            next_items = common.propagate(
                user_layers[-1], self._edge_users, self._edge_items, self._ui_weight, self.n_items
            )
            if use_dual:
                next_users = next_users + common.propagate(
                    user_layers[-1], self._user_src, self._user_dst, self._user_weight, self.n_users
                )
                next_items = next_items + common.propagate(
                    item_layers[-1], self._item_src, self._item_dst, self._item_weight, self.n_items
                )
            user_layers.append(next_users)
            item_layers.append(next_items)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(
            dim=0
        )

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        bipartite_user, bipartite_item = self._propagate(use_dual=False)
        dual_user, dual_item = self._propagate(use_dual=True)
        self._bipartite_user, self._bipartite_item = bipartite_user, bipartite_item
        self._dual_user, self._dual_item = dual_user, dual_item
        return dual_user, dual_item

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        bpr = self._bpr(users, positives, negatives, user_repr, item_repr)
        # Contrastive between the bipartite view and the dual-graph-enhanced view,
        # for both the user and item sides (DRAGON's dual-graph SSL).
        ssl = common.info_nce(self._bipartite_item, self._dual_item, self.temperature)
        ssl = ssl + common.info_nce(self._bipartite_user, self._dual_user, self.temperature)
        return bpr + self.ssl_weight * ssl
