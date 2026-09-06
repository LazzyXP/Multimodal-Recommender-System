"""Built-in recommendation models."""

from mmrec.models.bpr import BPRMFModel
from mmrec.models.graph import (
    BM3Model,
    DRAGONModel,
    FREEDOMModel,
    LATTICEModel,
    LGMRecModel,
    LightGCNModel,
    MGCNModel,
    MMGCNModel,
    torch_available,
)
from mmrec.models.item_cf import ItemCFModel
from mmrec.models.multimodal_item_knn import MultiModalItemKNNModel
from mmrec.models.multimodal_late_fusion import MultiModalLateFusionKNNModel
from mmrec.models.popularity import PopularityModel
from mmrec.models.vbpr import VBPRModel

__all__ = [
    "BPRMFModel",
    "BM3Model",
    "DRAGONModel",
    "FREEDOMModel",
    "ItemCFModel",
    "LATTICEModel",
    "LGMRecModel",
    "LightGCNModel",
    "MGCNModel",
    "MMGCNModel",
    "MultiModalItemKNNModel",
    "MultiModalLateFusionKNNModel",
    "PopularityModel",
    "torch_available",
    "VBPRModel",
]
