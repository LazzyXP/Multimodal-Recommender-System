"""Paper-specific torch graph recommenders."""

from mmrec.models.graph.base import torch_available
from mmrec.models.graph.bm3 import BM3Model
from mmrec.models.graph.dragon import DRAGONModel
from mmrec.models.graph.freedom import FREEDOMModel
from mmrec.models.graph.lattice import LATTICEModel
from mmrec.models.graph.lgmrec import LGMRecModel
from mmrec.models.graph.lightgcn import LightGCNModel
from mmrec.models.graph.mgcn import MGCNModel
from mmrec.models.graph.mmgcn import MMGCNModel

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
