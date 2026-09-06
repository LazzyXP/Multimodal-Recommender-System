"""Input loading, validation, and holdout splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mmrec.config import ColumnConfig
from mmrec.storage import ParquetCache

TableInput = pd.DataFrame | str | Path


@dataclass(slots=True)
class DatasetBundle:
    interactions: pd.DataFrame
    users: pd.DataFrame | None
    items: pd.DataFrame | None
    modalities: dict[str, Any]
    sources: dict[str, str | None]


def load_table(
    value: TableInput | None,
    name: str,
    cache: ParquetCache,
) -> tuple[pd.DataFrame | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, pd.DataFrame):
        materialized = cache.materialize_frame(value, name)
        return pd.read_parquet(materialized.parquet), str(materialized.parquet)

    materialized = cache.materialize(value, name)
    return pd.read_parquet(materialized.parquet), str(materialized.parquet)


def prepare_dataset(
    interactions: TableInput,
    users: TableInput | None,
    items: TableInput | None,
    columns: ColumnConfig,
    modalities: dict[str, Any] | None,
    cache_dir: str | Path = ".mmrec/cache",
) -> DatasetBundle:
    cache = ParquetCache(cache_dir)
    interaction_frame, interaction_source = load_table(interactions, "interactions", cache)
    if interaction_frame is None:  # Defensive guard for callers passing a null input.
        raise ValueError("interactions is required.")
    user_frame, user_source = load_table(users, "users", cache)
    item_frame, item_source = load_table(items, "items", cache)

    _require_columns(interaction_frame, columns.required_interaction_columns(), "interactions")
    if interaction_frame.empty:
        raise ValueError("interactions must contain at least one row.")
    if interaction_frame[[columns.user_id, columns.item_id]].isna().any().any():
        raise ValueError("user and item identifiers cannot contain missing values.")

    # Internal splitting uses row indices, so normalize user-provided duplicate indices.
    interaction_frame = interaction_frame.reset_index(drop=True)
    if columns.timestamp:
        interaction_frame[columns.timestamp] = pd.to_datetime(
            interaction_frame[columns.timestamp], errors="raise", utc=True
        )
    if columns.label:
        labels = pd.to_numeric(interaction_frame[columns.label], errors="coerce")
        if labels.isna().any():
            raise ValueError(f"interactions.{columns.label} must contain numeric values.")
        interaction_frame = interaction_frame[labels > 0].copy()
        if interaction_frame.empty:
            raise ValueError("No positive interactions remain after applying the label column.")

    if user_frame is not None:
        _require_columns(user_frame, [columns.user_id], "users")
        _require_unique(user_frame, columns.user_id, "users")
    if item_frame is not None:
        _require_columns(item_frame, [columns.item_id], "items")
        _require_unique(item_frame, columns.item_id, "items")

    normalized_modalities = modalities or {}
    _validate_modalities(normalized_modalities, user_frame, item_frame)
    return DatasetBundle(
        interaction_frame,
        user_frame,
        item_frame,
        normalized_modalities,
        {"interactions": interaction_source, "users": user_source, "items": item_source},
    )


def leave_one_out_split(
    interactions: pd.DataFrame,
    columns: ColumnConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each eligible user's latest interaction for evaluation."""
    user_col = columns.user_id
    counts = interactions.groupby(user_col, sort=False)[user_col].transform("size")
    eligible = interactions[counts >= 2]
    if eligible.empty:
        return interactions.copy(), interactions.iloc[0:0].copy()

    if columns.timestamp:
        ordered = eligible.sort_values([user_col, columns.timestamp], kind="stable")
    else:
        ordered = eligible.sort_index()
    test_indices = ordered.groupby(user_col, sort=False).tail(1).index
    test = interactions.loc[test_indices].copy()
    train = interactions.drop(index=test_indices).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def random_leave_one_out_split(
    interactions: pd.DataFrame,
    columns: ColumnConfig,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one uniformly random interaction per eligible user."""
    user_col = columns.user_id
    counts = interactions.groupby(user_col, sort=False)[user_col].transform("size")
    eligible = interactions[counts >= 2]
    if eligible.empty:
        return interactions.copy(), interactions.iloc[0:0].copy()

    rng = np.random.default_rng(random_state)
    test_indices: list[Any] = []
    for _, group in eligible.groupby(user_col, sort=False):
        test_indices.append(group.index[int(rng.integers(len(group)))])
    test = interactions.loc[test_indices].copy()
    train = interactions.drop(index=test_indices).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cold_start_split(
    interactions: pd.DataFrame,
    columns: ColumnConfig,
    cold_user_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out all interactions of a random sample of users (cold-start users)."""
    user_col = columns.user_id
    users = interactions[user_col].drop_duplicates().tolist()
    rng = np.random.default_rng(random_state)
    n_cold = max(1, int(round(len(users) * cold_user_frac)))
    # Keep at least one user in the training partition when possible.
    n_cold = min(n_cold, len(users) - 1) if len(users) > 1 else 0
    cold_users = set(rng.choice(users, size=n_cold, replace=False).tolist()) if n_cold else set()
    is_cold = interactions[user_col].isin(cold_users)
    holdout = interactions[is_cold].copy()
    train = interactions[~is_cold].copy()
    return train.reset_index(drop=True), holdout.reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns: list[str], table: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, column: str, table: str) -> None:
    if frame[column].duplicated().any():
        raise ValueError(f"{table}.{column} must be unique.")


def _validate_modalities(
    modalities: dict[str, Any],
    users: pd.DataFrame | None,
    items: pd.DataFrame | None,
) -> None:
    for entity, frame in (("user", users), ("item", items)):
        config = modalities.get(entity, {})
        if config and frame is None:
            raise ValueError(
                f"Modalities were declared for {entity}, but no {entity}s table was given."
            )
        for modality, configured_columns in config.items():
            if modality not in {"categorical", "numerical", "text", "embedding", "image"}:
                raise ValueError(
                    f"Unsupported {entity}.{modality} modality. "
                    "Choose categorical, numerical, text, embedding, or image."
                )
            names = (
                [configured_columns] if isinstance(configured_columns, str) else configured_columns
            )
            if not isinstance(names, (list, tuple)):
                raise TypeError(f"{entity}.{modality} modality columns must be a string or list.")
            _require_columns(frame, list(names), f"{entity}s")  # type: ignore[arg-type]


def global_temporal_split(
    interactions: pd.DataFrame,
    columns: ColumnConfig,
    holdout_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest time window without splitting equal timestamps."""
    if columns.timestamp is None:
        raise ValueError("global_temporal requires a timestamp column.")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one.")
    times = interactions[columns.timestamp]
    if times.isna().any() or times.nunique() < 2:
        raise ValueError("global_temporal requires at least two distinct, non-null timestamps.")
    ordered = times.sort_values(kind="stable")
    position = min(len(ordered) - 1, max(1, int(len(ordered) * (1 - holdout_fraction))))
    cutoff = ordered.iloc[position]
    if cutoff == ordered.iloc[0]:
        cutoff = ordered[ordered > cutoff].iloc[0]
    mask = times >= cutoff
    return (interactions[~mask].reset_index(drop=True), interactions[mask].reset_index(drop=True))
