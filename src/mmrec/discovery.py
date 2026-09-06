"""Turnkey dataset discovery: infer files, columns and modalities from a directory.

This is the "give me a data directory, I do the rest" layer. It scans a
directory for interaction/user/item tables, guesses the ID / timestamp / label
columns by name, and infers item modalities (categorical, numerical, text,
embedding) from column names and dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet

from mmrec.storage import PARQUET_SUFFIXES, convert_to_parquet

USER_ID_CANDIDATES = ("user_id", "userid", "user", "uid", "user_id_str")
ITEM_ID_CANDIDATES = (
    "item_id",
    "itemid",
    "item",
    "iid",
    "movie_id",
    "movieid",
    "product_id",
    "productid",
    "book_id",
    "bookid",
)
TIMESTAMP_CANDIDATES = ("timestamp", "time", "ts", "datetime", "date", "created_at")
LABEL_CANDIDATES = ("rating", "label", "score", "click", "clicked", "purchase", "purchased")

TABLE_PATTERNS = {
    "interactions": (
        "interactions",
        "interaction",
        "ratings",
        "rating",
        "events",
        "clicks",
        "train",
    ),
    "users": ("users", "user"),
    "items": ("items", "item", "movies", "products", "books"),
}

EMBEDDING_NAME_HINTS = ("embedding", "vector", "feature", "emb", "vec")


@dataclass
class DatasetDiscovery:
    interactions: str
    users: str | None = None
    items: str | None = None
    user_id: str = "user_id"
    item_id: str = "item_id"
    timestamp: str | None = "timestamp"
    label: str | None = None
    modalities: dict[str, Any] = field(default_factory=dict)


def discover_dataset(directory: str | Path) -> DatasetDiscovery:
    """Scan a directory and infer the dataset layout and schema."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    tables = _discover_tables(root)
    if "interactions" not in tables:
        raise ValueError(
            "No interaction table found. Expected a file whose name contains "
            f"one of {TABLE_PATTERNS['interactions']} in {root}."
        )
    interactions = _materialize(tables["interactions"])
    columns = _column_names(interactions)

    user_id = _infer_column(columns, USER_ID_CANDIDATES, required=True, table="interactions")
    item_id = _infer_column(columns, ITEM_ID_CANDIDATES, required=True, table="interactions")
    timestamp = _infer_column(columns, TIMESTAMP_CANDIDATES, required=False)
    label = _infer_column(columns, LABEL_CANDIDATES, required=False)

    discovery = DatasetDiscovery(
        interactions=interactions,
        users=_materialize(tables["users"]) if "users" in tables else None,
        items=_materialize(tables["items"]) if "items" in tables else None,
        user_id=user_id,
        item_id=item_id,
        timestamp=timestamp,
        label=label,
    )

    if discovery.items is not None:
        item_columns = _column_names(discovery.items)
        discovery.modalities = _infer_item_modalities(discovery.items, item_id, item_columns)
    return discovery


def _discover_tables(root: Path) -> dict[str, Path]:
    tables: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem.lower()
        for kind, patterns in TABLE_PATTERNS.items():
            if kind in tables:
                continue
            if any(pattern in stem for pattern in patterns):
                tables[kind] = path
                break
    return tables


def _materialize(path: Path) -> str:
    if path.suffix.lower() in PARQUET_SUFFIXES:
        return str(path)
    return str(convert_to_parquet(path))


def _column_names(parquet_path: str) -> list[str]:
    return parquet.read_schema(parquet_path).names


def _infer_column(
    columns: list[str],
    candidates: tuple[str, ...],
    required: bool,
    table: str = "",
) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    if required:
        raise ValueError(
            f"Cannot infer the required column among {candidates} in {table} table. "
            f"Available columns: {columns}. Specify it explicitly instead."
        )
    return None


def _infer_item_modalities(
    items_path: str,
    item_id: str,
    columns: list[str],
) -> dict[str, Any]:
    dataset = arrow_dataset.dataset(items_path, format="parquet")
    frame = dataset.to_table(columns=columns).to_pandas()

    categorical: list[str] = []
    numerical: list[str] = []
    text: list[str] = []
    embedding: list[str] = []

    for column in columns:
        if column == item_id:
            continue
        series = frame[column]
        lowered = column.lower()
        if any(hint in lowered for hint in EMBEDDING_NAME_HINTS) or _looks_like_embedding(series):
            embedding.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numerical.append(column)
        elif _looks_like_text(series):
            text.append(column)
        else:
            categorical.append(column)

    modalities: dict[str, Any] = {"item": {}}
    if categorical:
        modalities["item"]["categorical"] = categorical
    if numerical:
        modalities["item"]["numerical"] = numerical
    if text:
        modalities["item"]["text"] = text
    if embedding:
        modalities["item"]["embedding"] = embedding
    return modalities if modalities["item"] else {}


def _looks_like_text(series: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    sampled = series.dropna().astype(str).head(500)
    if sampled.empty:
        return False
    average_length = sampled.str.len().mean()
    cardinality = sampled.nunique()
    # Text columns have long values; categorical keys are short and few.
    return average_length > 12 or cardinality > 200


def _looks_like_embedding(series: pd.Series) -> bool:
    if not pd.api.types.is_object_dtype(series):
        return False
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    return all(isinstance(value, (list, tuple)) for value in sample)
