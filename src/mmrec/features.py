"""Lightweight multimodal feature encoding for item representations."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def encode_item_features(
    items: pd.DataFrame,
    item_column: str,
    modalities: dict[str, Any],
    text_dimensions: int = 128,
) -> tuple[list[Any], np.ndarray]:
    """Encode supported modality columns into normalized item vectors."""
    item_ids, named_blocks = encode_item_feature_blocks(
        items,
        item_column,
        modalities,
        text_dimensions=text_dimensions,
    )
    matrix = _row_normalize(np.concatenate(list(named_blocks.values()), axis=1))
    return item_ids, matrix


def encode_item_feature_blocks(
    items: pd.DataFrame,
    item_column: str,
    modalities: dict[str, Any],
    text_dimensions: int = 128,
) -> tuple[list[Any], dict[str, np.ndarray]]:
    """Encode each modality into an independently normalized feature block."""
    blocks: dict[str, np.ndarray] = {}

    categorical = _as_columns(modalities.get("categorical", []))
    if categorical:
        values = items[categorical].fillna("__missing__").astype(str)
        encoded = pd.get_dummies(values, columns=categorical, dtype=float).to_numpy()
        blocks["categorical"] = _row_normalize(encoded)

    numerical = _as_columns(modalities.get("numerical", []))
    if numerical:
        numeric = items[numerical].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median()).fillna(0.0)
        standard_deviation = numeric.std(ddof=0).replace(0, 1.0)
        encoded = ((numeric - numeric.mean()) / standard_deviation).to_numpy(dtype=float)
        blocks["numerical"] = _row_normalize(encoded)

    text = _as_columns(modalities.get("text", []))
    if text:
        blocks["text"] = _encode_text(items[text], text_dimensions)

    embedding_columns = [
        *_as_columns(modalities.get("embedding", [])),
        *_as_columns(modalities.get("image", [])),
    ]
    for column in embedding_columns:
        blocks[f"embedding:{column}"] = _encode_embedding_column(items[column], column)

    if not blocks:
        raise ValueError(
            "MultiModalItemKNN requires at least one item modality: categorical, numerical, "
            "text, embedding, or image."
        )

    return items[item_column].tolist(), blocks


def _encode_text(frame: pd.DataFrame, dimensions: int) -> np.ndarray:
    matrix = np.zeros((len(frame), dimensions), dtype=float)
    for row_index, values in enumerate(frame.fillna("").astype(str).itertuples(index=False)):
        text = " ".join(values).lower()
        for token in TOKEN_PATTERN.findall(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % dimensions
            matrix[row_index, bucket] += 1.0
    return _row_normalize(np.log1p(matrix))


def _encode_embedding_column(series: pd.Series, column: str) -> np.ndarray:
    vectors: list[np.ndarray] = []
    expected_size: int | None = None
    for value in series:
        if isinstance(value, str):
            raise ValueError(
                f"{column!r} contains paths or strings. The MVP expects precomputed numeric "
                "embeddings for image/embedding modalities."
            )
        vector = np.asarray(value, dtype=float).reshape(-1)
        if expected_size is None:
            expected_size = vector.size
        if vector.size != expected_size:
            raise ValueError(f"{column!r} embeddings must all have the same dimension.")
        vectors.append(vector)
    if not vectors or expected_size == 0:
        raise ValueError(f"{column!r} must contain non-empty numeric embeddings.")
    return _row_normalize(np.stack(vectors))


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=float), where=norms != 0)


def _as_columns(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)
