"""Registry for built-in and third-party recommendation models."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from mmrec.config import ColumnConfig
from mmrec.models import (
    BM3Model,
    BPRMFModel,
    DRAGONModel,
    FREEDOMModel,
    ItemCFModel,
    LATTICEModel,
    LGMRecModel,
    LightGCNModel,
    MGCNModel,
    MMGCNModel,
    MultiModalItemKNNModel,
    MultiModalLateFusionKNNModel,
    PopularityModel,
    VBPRModel,
)
from mmrec.models.base import BaseRecommendationModel

ModelFactory = Callable[..., BaseRecommendationModel]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    family: str
    requires_item_modalities: bool
    training: str
    reference: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {
            "Popularity": PopularityModel,
            "ItemCF": ItemCFModel,
            "BPRMF": BPRMFModel,
            "MultiModalItemKNN": MultiModalItemKNNModel,
            "MultiModalLateFusionKNN": MultiModalLateFusionKNNModel,
            "VBPR": VBPRModel,
            "LightGCN": LightGCNModel,
            "MMGCN": MMGCNModel,
            "LATTICE": LATTICEModel,
            "BM3": BM3Model,
            "FREEDOM": FREEDOMModel,
            "MGCN": MGCNModel,
            "DRAGON": DRAGONModel,
            "LGMRec": LGMRecModel,
        }
        self._specs: dict[str, ModelSpec] = {
            "Popularity": ModelSpec("Popularity", "baseline", False, "full"),
            "ItemCF": ModelSpec("ItemCF", "collaborative", False, "bounded"),
            "BPRMF": ModelSpec(
                "BPRMF",
                "latent-factor",
                False,
                "bounded",
                "https://arxiv.org/abs/1205.2618",
            ),
            "MultiModalItemKNN": ModelSpec(
                "MultiModalItemKNN", "multimodal-content", True, "dense"
            ),
            "MultiModalLateFusionKNN": ModelSpec(
                "MultiModalLateFusionKNN", "multimodal-late-fusion", True, "dense"
            ),
            "VBPR": ModelSpec(
                "VBPR",
                "multimodal-ranking",
                True,
                "bounded",
                "https://arxiv.org/abs/1510.01784",
            ),
            "LightGCN": ModelSpec(
                "LightGCN",
                "graph-collaborative",
                False,
                "torch",
                "https://arxiv.org/abs/2002.02126",
            ),
            "MMGCN": ModelSpec(
                "MMGCN", "multimodal-graph", True, "torch", "https://arxiv.org/abs/2002.10872"
            ),
            "LATTICE": ModelSpec(
                "LATTICE", "multimodal-graph", True, "torch", "https://arxiv.org/abs/2104.09036"
            ),
            "BM3": ModelSpec(
                "BM3",
                "self-supervised-multimodal",
                True,
                "torch",
                "https://arxiv.org/abs/2207.05929",
            ),
            "FREEDOM": ModelSpec(
                "FREEDOM", "multimodal-graph", True, "torch", "https://arxiv.org/abs/2211.06957"
            ),
            "MGCN": ModelSpec(
                "MGCN", "multimodal-graph", True, "torch", "https://arxiv.org/abs/2308.09605"
            ),
            "DRAGON": ModelSpec(
                "DRAGON", "multimodal-graph", True, "torch", "https://arxiv.org/abs/2305.14241"
            ),
            "LGMRec": ModelSpec(
                "LGMRec", "multimodal-graph", True, "torch", "https://arxiv.org/abs/2402.11503"
            ),
        }

    def register(self, name: str, factory: ModelFactory, overwrite: bool = False) -> None:
        if name in self._factories and not overwrite:
            raise ValueError(f"Model {name!r} is already registered.")
        self._factories[name] = factory
        self._specs[name] = ModelSpec(name, "custom", False, "custom")

    def create(
        self,
        name: str,
        columns: ColumnConfig,
        **kwargs: Any,
    ) -> BaseRecommendationModel:
        try:
            return self._factories[name](columns, **kwargs)
        except KeyError as exc:
            available = ", ".join(sorted(self._factories))
            raise ValueError(f"Unknown model {name!r}. Available models: {available}") from exc

    def available(self) -> list[str]:
        return sorted(self._factories)

    def accepts_parameter(self, name: str, parameter: str) -> bool:
        """Return whether the factory for `name` accepts `parameter` in its constructor."""
        factory = self._factories[name]
        try:
            signature = inspect.signature(factory)
        except (TypeError, ValueError):
            return False
        if parameter in signature.parameters:
            return True
        # A ``**kwargs`` parameter (used by the graph-model subclasses) accepts
        # anything, so parameters such as ``time_limit`` flow through to the base.
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )

    def requires_torch(self, name: str) -> bool:
        """Return whether a model runs on the optional PyTorch runtime."""
        spec = self._specs.get(name)
        return spec is not None and spec.training == "torch"

    def catalog(self) -> list[dict[str, Any]]:
        return [self._specs[name].to_dict() for name in sorted(self._specs)]
