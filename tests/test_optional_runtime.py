"""Test the core import path even when the developer has torch installed."""

import subprocess
import sys
from pathlib import Path

import mmrec


def test_core_runs_with_torch_import_blocked(tmp_path):
    code = """
import sys
import importlib.abc
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch deliberately unavailable", name="torch")
sys.meta_path.insert(0, BlockTorch())
sys.path.insert(0, sys.argv[1])
import pandas as pd
from mmrec import MultiModalRecommender
from mmrec.models import torch_available, LightGCNModel
from mmrec.config import ColumnConfig
assert not torch_available()
data = pd.DataFrame({"user_id": ["u"] * 3, "item_id": ["a", "b", "c"]})
model = MultiModalRecommender(timestamp=None).fit(data, models="auto")
assert "LightGCN" in model.available_models()
assert "LightGCN" not in model.models
assert not model.recommend(["new"]).data.empty
restored = MultiModalRecommender.load(model.save("saved"))
assert restored.model_best == model.model_best
try:
    LightGCNModel(ColumnConfig())
except ImportError as error:
    assert "python -m pip install torch" in str(error)
else:
    raise AssertionError("Explicit torch model must explain the missing dependency")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(mmrec.__file__).parent.parent)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
