"""Paper-specific torch graph recommenders."""


def torch_available() -> bool:
    """Probe the optional runtime before importing any graph implementation."""
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


class _MissingTorchModel:
    def __init__(self, columns, **kwargs):
        raise ImportError(
            "Graph models require the optional PyTorch runtime. "
            "Install it with: python -m pip install torch"
        )


if torch_available():
    from mmrec.models.graph.bm3 import BM3Model
    from mmrec.models.graph.dragon import DRAGONModel
    from mmrec.models.graph.freedom import FREEDOMModel
    from mmrec.models.graph.lattice import LATTICEModel
    from mmrec.models.graph.lgmrec import LGMRecModel
    from mmrec.models.graph.lightgcn import LightGCNModel
    from mmrec.models.graph.mgcn import MGCNModel
    from mmrec.models.graph.mmgcn import MMGCNModel
else:
    BM3Model = DRAGONModel = FREEDOMModel = LATTICEModel = _MissingTorchModel
    LGMRecModel = LightGCNModel = MGCNModel = MMGCNModel = _MissingTorchModel


__all__ = [
    "BM3Model",
    "DRAGONModel",
    "FREEDOMModel",
    "LATTICEModel",
    "LGMRecModel",
    "LightGCNModel",
    "MGCNModel",
    "MMGCNModel",
    "torch_available",
]
