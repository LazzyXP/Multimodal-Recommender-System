import numpy as np
import pandas as pd
import pytest

from mmrec.features import align_item_feature_matrix, encode_item_feature_blocks


def test_embedding_missing_values_are_zero_filled():
    items = pd.DataFrame({"item_id": ["a", "b", "c"], "image": [[1.0, 0.0], None, [0.0, 2.0]]})
    _, blocks = encode_item_feature_blocks(items, "item_id", {"image": ["image"]})
    assert blocks["embedding:image"].shape == (3, 2)
    np.testing.assert_allclose(blocks["embedding:image"][1], [0.0, 0.0])


def test_embedding_all_missing_values_fail_with_clear_error():
    items = pd.DataFrame({"item_id": ["a", "b"], "image": [None, None]})
    with pytest.raises(ValueError, match="non-empty numeric embeddings"):
        encode_item_feature_blocks(items, "item_id", {"image": ["image"]})


def test_align_item_feature_matrix_uses_float32_and_catalog_order():
    aligned = align_item_feature_matrix(
        ["c", "missing", "a"], ["a", "c"], np.array([[1, 2], [3, 4]], dtype=float)
    )
    assert aligned.dtype == np.float32
    np.testing.assert_allclose(aligned, [[3, 4], [0, 0], [1, 2]])
