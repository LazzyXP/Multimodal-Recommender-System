"""BM3: Bootstrap Latent Representations for Multi-modal Recommendation.

Zhou, Zhou, et al. (WWW 2023), https://arxiv.org/abs/2207.05929.

Faithful core: a per-modality online encoder and an exponentially-moving-averaged
target encoder; a prediction head maps the online representation to the target's,
and a bootstrap contrastive term (InfoNCE between the online prediction and the
stopped-gradient target) is optimized together with BPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mmrec.models.graph import common
from mmrec.models.graph.base import GraphModelBase


class BM3Model(GraphModelBase):
    name = "BM3"
    requires_modalities = True

    def __init__(self, columns, momentum: float = 0.99, **kwargs) -> None:
        super().__init__(columns, **kwargs)
        self.momentum = momentum

    def _build(self) -> None:
        self._user_embedding = self._embedding(self.n_users)
        self._item_embedding = self._embedding(self.n_items)
        self._edge_users = self._on_device(self._edge_users)
        self._edge_items = self._on_device(self._edge_items)
        self._ui_weight = self._on_device(self._ui_weight)

        self._modality_names = sorted(self._feature_blocks)
        self._features = [
            self._on_device(self._feature_blocks[name]) for name in self._modality_names
        ]
        # Per-modality online projections and EMA target projections.
        self._online_projections = nn.ModuleList(
            [
                nn.Linear(features.size(1), self.factors, bias=False).to(self._device)
                for features in self._features
            ]
        )
        self._target_projections = nn.ModuleList(
            [
                nn.Linear(features.size(1), self.factors, bias=False).to(self._device)
                for features in self._features
            ]
        )
        for online, target in zip(self._online_projections, self._target_projections, strict=True):
            nn.init.xavier_uniform_(online.weight)
            target.load_state_dict(online.state_dict())
        # Bootstrap prediction head: online predicts the target representation.
        self._predictor = nn.Sequential(
            nn.Linear(self.factors, self.factors),
            nn.ReLU(),
            nn.Linear(self.factors, self.factors),
        ).to(self._device)

    def _fuse(self, projections: nn.ModuleList) -> torch.Tensor:
        content = sum(
            projection(features)
            for projection, features in zip(projections, self._features, strict=True)
        )
        return self._item_embedding.weight + content

    def _propagate(self, item_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        user_layers = [self._user_embedding.weight]
        item_layers = [item_repr]
        for _ in range(self.layers):
            next_users = common.propagate(
                item_layers[-1], self._edge_items, self._edge_users, self._ui_weight, self.n_users
            )
            next_items = common.propagate(
                user_layers[-1], self._edge_users, self._edge_items, self._ui_weight, self.n_items
            )
            user_layers.append(next_users)
            item_layers.append(next_items)
        return torch.stack(user_layers, dim=0).mean(dim=0), torch.stack(item_layers, dim=0).mean(
            dim=0
        )

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        online_item = self._fuse(self._online_projections)
        target_item = self._fuse(self._target_projections).detach()
        users, items = self._propagate(online_item)
        self._online_item = items
        self._target_item = target_item
        return users, items

    def _training_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        user_repr: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        bpr = self._bpr(users, positives, negatives, user_repr, item_repr)
        prediction = self._predictor(self._online_item)
        ssl = common.info_nce(prediction, self._target_item, self.temperature)
        return bpr + self.ssl_weight * ssl

    def _after_step(self) -> None:
        with torch.no_grad():
            for target, source in zip(
                self._target_projections, self._online_projections, strict=True
            ):
                for target_param, source_param in zip(
                    target.parameters(), source.parameters(), strict=True
                ):
                    target_param.mul_(self.momentum).add_(
                        source_param.detach(), alpha=1 - self.momentum
                    )
