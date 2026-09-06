"""Bounded-memory utilities for large interaction datasets."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as arrow_dataset
import pyarrow.parquet as parquet

from mmrec.config import ColumnConfig


@dataclass(slots=True)
class InteractionScanResult:
    interactions: int
    item_counts: Counter[Any]
    sample: pd.DataFrame
    catalog_truncated: bool


def parquet_row_count(path: str | Path) -> int:
    return parquet.ParquetFile(path).metadata.num_rows


def parquet_size_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size


def scan_and_sample_interactions(
    path: str | Path,
    columns: ColumnConfig,
    sample_rows: int,
    batch_size: int,
    max_catalog_items: int,
    max_history_per_user: int,
) -> InteractionScanResult:
    """Compute exact item counts while retaining a stable user-level sample."""
    dataset = arrow_dataset.dataset(path, format="parquet")
    required = columns.required_interaction_columns()
    missing = sorted(set(required) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"interactions is missing required columns: {missing}")

    estimated_rows = max(parquet_row_count(path), 1)
    sampling_rate = min(1.0, sample_rows / estimated_rows)
    hash_limit = max(1, int(sampling_rate * np.iinfo(np.uint64).max))
    item_counts: Counter[Any] = Counter()
    samples: list[pd.DataFrame] = []
    positive_rows = 0
    catalog_truncated = False
    sampled_rows = 0
    sampled_user_counts: Counter[Any] = Counter()

    scanner = dataset.scanner(columns=required, batch_size=batch_size, use_threads=True)
    for record_batch in scanner.to_batches():
        frame = record_batch.to_pandas()
        if columns.label:
            labels = pd.to_numeric(frame[columns.label], errors="coerce")
            if labels.isna().any():
                raise ValueError(f"interactions.{columns.label} must contain numeric values.")
            frame = frame[labels > 0]
        if frame.empty:
            continue
        if frame[[columns.user_id, columns.item_id]].isna().any().any():
            raise ValueError("user and item identifiers cannot contain missing values.")
        positive_rows += len(frame)
        item_counts.update(frame[columns.item_id].tolist())
        if len(item_counts) > max_catalog_items * 2:
            item_counts = Counter(dict(item_counts.most_common(max_catalog_items)))
            catalog_truncated = True

        hashes = pd.util.hash_pandas_object(
            frame[columns.user_id], index=False, hash_key="0123456789123456"
        ).to_numpy(dtype=np.uint64)
        selected = frame.loc[hashes <= hash_limit]
        if not selected.empty and sampled_rows < sample_rows:
            for user_id, group in selected.groupby(columns.user_id, sort=False):
                remaining_user = max_history_per_user - sampled_user_counts[user_id]
                remaining_global = sample_rows - sampled_rows
                take = min(len(group), remaining_user, remaining_global)
                if take <= 0:
                    continue
                samples.append(group.iloc[:take])
                sampled_user_counts[user_id] += take
                sampled_rows += take
                if sampled_rows >= sample_rows:
                    break

    if positive_rows == 0:
        raise ValueError("interactions must contain at least one positive row.")
    sample = pd.concat(samples, ignore_index=True) if samples else _fallback_sample(
        dataset, required, columns, min(sample_rows, positive_rows), batch_size
    )
    if columns.timestamp:
        sample[columns.timestamp] = pd.to_datetime(
            sample[columns.timestamp], errors="raise", utc=True
        )
    if len(item_counts) > max_catalog_items:
        item_counts = Counter(dict(item_counts.most_common(max_catalog_items)))
        catalog_truncated = True
    return InteractionScanResult(
        positive_rows,
        item_counts,
        sample.reset_index(drop=True),
        catalog_truncated,
    )


def seen_items_for_users(
    path: str | Path,
    columns: ColumnConfig,
    users: list[Any],
    batch_size: int,
) -> dict[Any, set[Any]]:
    """Read exact histories for only the users in the current inference batch."""
    if not users:
        return {}
    dataset = arrow_dataset.dataset(path, format="parquet")
    user_array = pa.array(users, type=dataset.schema.field(columns.user_id).type)
    expression = pc.field(columns.user_id).isin(user_array)
    selected_columns = [columns.user_id, columns.item_id]
    if columns.label:
        selected_columns.append(columns.label)
        expression = expression & (pc.field(columns.label) > 0)

    histories: dict[Any, set[Any]] = {user: set() for user in users}
    scanner = dataset.scanner(
        columns=selected_columns,
        filter=expression,
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        for user_id, group in frame.groupby(columns.user_id, sort=False):
            histories.setdefault(user_id, set()).update(group[columns.item_id].tolist())
    return histories


def build_history_index(
    path: str | Path,
    output_root: str | Path,
    columns: ColumnConfig,
    partitions: int,
    batch_size: int,
) -> Path:
    """Build a reusable, hash-partitioned user-item history dataset."""
    source = Path(path).resolve()
    stat = source.stat()
    fingerprint = hashlib.sha256(
        f"v2|{source}|{stat.st_size}|{stat.st_mtime_ns}|{partitions}".encode()
    ).hexdigest()[:16]
    target = Path(output_root) / f"history-{fingerprint}"
    success = target / "_SUCCESS"
    if success.exists():
        return target.resolve()

    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    dataset = arrow_dataset.dataset(source, format="parquet")
    selected_columns = [columns.user_id, columns.item_id]
    if columns.label:
        selected_columns.append(columns.label)
    partitioning = arrow_dataset.partitioning(
        pa.schema([("bucket", pa.int32())]),
        flavor="hive",
    )
    output_schema = pa.schema(
        [
            dataset.schema.field(columns.user_id),
            dataset.schema.field(columns.item_id),
            pa.field("bucket", pa.int32()),
        ]
    )

    def indexed_batches():
        scanner = dataset.scanner(
            columns=selected_columns,
            batch_size=batch_size,
            use_threads=True,
        )
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if columns.label:
                frame = frame[frame[columns.label] > 0]
            if frame.empty:
                continue
            hashes = pd.util.hash_pandas_object(
                frame[columns.user_id], index=False, hash_key="0123456789123456"
            ).to_numpy(dtype=np.uint64)
            frame = frame[[columns.user_id, columns.item_id]].copy()
            frame["bucket"] = (hashes % partitions).astype(np.int32)
            yield pa.RecordBatch.from_pandas(
                frame,
                schema=output_schema,
                preserve_index=False,
            )

    try:
        reader = pa.RecordBatchReader.from_batches(output_schema, indexed_batches())
        arrow_dataset.write_dataset(
            reader,
            temporary,
            format="parquet",
            partitioning=partitioning,
            basename_template="part-{i}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            file_options=arrow_dataset.ParquetFileFormat().make_write_options(
                compression="zstd",
                use_dictionary=True,
            ),
            max_open_files=64,
            max_rows_per_file=1_000_000,
            min_rows_per_group=64_000,
            max_rows_per_group=262_144,
        )
        (temporary / "_SUCCESS").touch()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target.resolve()


def seen_items_from_index(
    path: str | Path,
    columns: ColumnConfig,
    users: list[Any],
    partitions: int,
    batch_size: int,
) -> dict[Any, set[Any]]:
    if not users:
        return {}
    dataset = arrow_dataset.dataset(path, format="parquet", partitioning="hive")
    user_array = pa.array(users, type=dataset.schema.field(columns.user_id).type)
    hashes = pd.util.hash_pandas_object(
        pd.Series(users), index=False, hash_key="0123456789123456"
    ).to_numpy(dtype=np.uint64)
    buckets = sorted(set((hashes % partitions).astype(np.int32).tolist()))
    expression = arrow_dataset.field("bucket").isin(buckets) & arrow_dataset.field(
        columns.user_id
    ).isin(user_array)
    histories: dict[Any, set[Any]] = {user: set() for user in users}
    for batch in dataset.scanner(
        columns=[columns.user_id, columns.item_id],
        filter=expression,
        batch_size=batch_size,
        use_threads=True,
    ).to_batches():
        frame = batch.to_pandas()
        for user_id, group in frame.groupby(columns.user_id, sort=False):
            histories.setdefault(user_id, set()).update(group[columns.item_id].tolist())
    return histories


def iter_column_batches(
    path: str | Path,
    column: str,
    batch_size: int,
):
    dataset = arrow_dataset.dataset(path, format="parquet")
    if column not in dataset.schema.names:
        raise ValueError(f"users table is missing required column: {column}")
    for batch in dataset.scanner(
        columns=[column], batch_size=batch_size, use_threads=True
    ).to_batches():
        yield batch.column(0).to_pylist()


def _fallback_sample(
    dataset: arrow_dataset.Dataset,
    required: list[str],
    columns: ColumnConfig,
    sample_rows: int,
    batch_size: int,
) -> pd.DataFrame:
    batches: list[pd.DataFrame] = []
    remaining = sample_rows
    for batch in dataset.scanner(
        columns=required, batch_size=batch_size, use_threads=True
    ).to_batches():
        frame = batch.to_pandas()
        if columns.label:
            frame = frame[frame[columns.label] > 0]
        if frame.empty:
            continue
        batches.append(frame.iloc[:remaining])
        remaining -= min(remaining, len(frame))
        if remaining <= 0:
            break
    return pd.concat(batches, ignore_index=True)
