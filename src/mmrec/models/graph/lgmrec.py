"""LGMRec: Local and Global Graph Learning for Multimodal Recommendation.

https://arxiv.org/abs/2402.11503.

Faithful core: local embeddings are learned on the user-item bipartite graph
in Euclidean space, while global item embeddings live on the Lorentz hyperboloid
and are propagated over a multimodal semantic item-item graph using tangent-space
aggregation followed by the exponential map. The two are combined for BPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class LGMRecModel(GraphModelBase):
    name = "LGMRec"
    requires_modalities = True

    def __init__(self, columns, knn_k: int = 10, global_weight: float = 0.5, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.knn_k = knn_k
        self.global_weight = global_weight

    def _build(self) -> None:
        self._user_embedding = self._embedding(self.n_users)
        self._item_embedding = self._embedding(self.n_items)
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

        features = torch.cat(
            [self._on_device(block) for block in self._feature_blocks.values()], dim=1
        )
        src, dst = common.knn_graph(features, self.knn_k)
        self._item_src, self._item_dst = src, dst
        self._item_weight = common.symmetric_edge_weight(src, dst, self.n_items)

        # Global item embeddings on the Lorentz hyperboloid (curvature -1).
        tangent = torch.empty(self.n_items, self.factors, device=self._device)
        nn.init.xavier_uniform_(tangent)
        self._global_item = nn.Parameter(self._lorentz_exp0(tangent * 0.1))

    # -- Lorentz model (curvature -1) ---------------------------------------

    def _lorentz_exp0(self, vector: torch.Tensor) -> torch.Tensor:
        """Exponential map from the tangent space at the origin to the hyperboloid."""
        norm = vector.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        first = torch.cosh(norm)
        rest = torch.sinh(norm) * vector / norm
        return torch.cat([first, rest], dim=-1)

    def _lorentz_log0(self, hyperboloid: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from the hyperboloid to the tangent space at the origin."""
        first = hyperboloid[..., 0:1]
        rest = hyperboloid[..., 1:]
        norm = rest.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        distance = torch.acosh(torch.clamp(first, min=1.0 + 1e-6))
        return distance * rest / norm

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Local: Euclidean bipartite graph.
        local_users = [self._user_embedding.weight]
        local_items = [self._item_embedding.weight]
        for _ in range(self.layers):
            local_users.append(
                common.propagate(
                    local_items[-1],
                    self._edge_items,
                    self._edge_users,
                    self._ui_weight,
                    self.n_users,
                )
            )
            local_items.append(
                common.propagate(
                    local_users[-1],
                    self._edge_users,
                    self._edge_items,
                    self._ui_weight,
                    self.n_items,
                )
            )
        local_user = torch.stack(local_users, dim=0).mean(dim=0)
        local_item = torch.stack(local_items, dim=0).mean(dim=0)

        # Global: hyperbolic propagation over the semantic item-item graph.
        global_items = [self._global_item]
        for _ in range(self.layers):
            tangent = self._lorentz_log0(global_items[-1])
            aggregated = common.propagate(
                tangent, self._item_src, self._item_dst, self._item_weight, self.n_items
            )
            global_items.append(self._lorentz_exp0(aggregated))
        global_item = torch.stack(global_items, dim=0).mean(dim=0)

        # Map the global item to Euclidean and aggregate to users.
        global_item_euclidean = self._lorentz_log0(global_item)
        global_user = common.propagate(
            global_item_euclidean, self._edge_items, self._edge_users, self._ui_weight, self.n_users
        )

        return (
            local_user + self.global_weight * global_user,
            local_item + self.global_weight * global_item_euclidean,
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
