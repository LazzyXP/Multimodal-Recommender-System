"""Columnar data conversion and cache management."""

from __future__ import annotations

import csv as stdlib_csv
import hashlib
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.json as arrow_json
import pyarrow.parquet as parquet

DELIMITED_SUFFIXES = {".csv", ".tsv", ".txt"}
JSONL_SUFFIXES = {".jsonl", ".ndjson"}
PARQUET_SUFFIXES = {".parquet", ".pq"}


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Deterministic content hash of a DataFrame, for cache reuse."""
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(column.encode("utf-8"))
        digest.update(str(frame[column].dtype).encode("utf-8"))
    digest.update(str(frame.shape).encode("utf-8"))
    for column in frame.columns:
        series = frame[column]
        try:
            column_hash = pd.util.hash_pandas_object(series, index=False)
            digest.update(column_hash.to_numpy(dtype="uint64").tobytes())
        except (TypeError, ValueError):
            # Object columns (e.g. list embeddings) are unhashable; hash the
            # deterministic repr of each element. Raw ``.tobytes()`` on an
            # object array would encode pointer addresses, not contents, so two
            # equal DataFrames would get different fingerprints.
            for value in series:
                digest.update(repr(value).encode("utf-8", errors="replace"))
                digest.update(b"\x00")
    return digest.hexdigest()[:16]


def file_fingerprint(path: str | Path) -> str:
    """Return a content identity for a persisted table.

    Checkpoints must not be reused after a table is replaced in place. The
    metadata checks below are useful diagnostics, while the byte digest makes
    same-size/same-schema replacements unambiguous.
    """
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    digest = hashlib.sha256(
        f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode()
    )
    if source.suffix.lower() in PARQUET_SUFFIXES:
        schema = parquet.read_schema(source)
        digest.update("|".join(f"{field.name}:{field.type}" for field in schema).encode("utf-8"))
        metadata = parquet.read_metadata(source)
        digest.update(str(metadata.num_rows).encode("ascii"))
        digest.update(str(metadata.num_row_groups).encode("ascii"))
    with source.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class MaterializedData:
    source: Path
    parquet: Path
    converted: bool


class ParquetCache:
    def __init__(self, root: str | Path = ".mmrec/cache") -> None:
        self.root = Path(root)

    def materialize(self, source: str | Path, table_name: str) -> MaterializedData:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"{table_name} table does not exist: {source_path}")
        if source_path.suffix.lower() in PARQUET_SUFFIXES:
            return MaterializedData(source_path, source_path, converted=False)

        stat = source_path.stat()
        fingerprint = hashlib.sha256(
            f"{source_path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        ).hexdigest()[:16]
        output = self.root / f"{table_name}-{source_path.stem}-{fingerprint}.parquet"
        if output.exists():
            return MaterializedData(source_path, output.resolve(), converted=True)

        convert_to_parquet(source_path, output, overwrite=False)
        return MaterializedData(source_path, output.resolve(), converted=True)

    def materialize_frame(self, frame: pd.DataFrame, table_name: str) -> MaterializedData:
        self.root.mkdir(parents=True, exist_ok=True)
        # Content-hash fingerprint so identical DataFrames reuse one cache file
        # instead of accumulating a new file per call.
        fingerprint = frame_fingerprint(frame)
        output = (self.root / f"{table_name}-memory-{fingerprint}.parquet").resolve()
        if output.exists():
            return MaterializedData(output, output, converted=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            parquet.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                temporary,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        return MaterializedData(output, output, converted=True)


def convert_to_parquet(
    source: str | Path,
    output: str | Path | None = None,
    *,
    delimiter: str | None = None,
    compression: str = "zstd",
    overwrite: bool = False,
    block_size: int = 64 * 1024 * 1024,
) -> Path:
    """Stream a supported text dataset into a compressed Parquet file."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input data does not exist: {source_path}")

    suffix = source_path.suffix.lower()
    supported = DELIMITED_SUFFIXES | JSONL_SUFFIXES | PARQUET_SUFFIXES
    if suffix not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported input format {suffix!r}. Supported formats: {choices}")

    target = (
        Path(output).expanduser().resolve()
        if output is not None
        else source_path.with_suffix(".parquet")
    )
    if target == source_path:
        return source_path
    if target.exists() and not overwrite:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        if suffix in PARQUET_SUFFIXES:
            shutil.copy2(source_path, temporary)
        else:
            reader = _open_streaming_reader(source_path, delimiter, block_size)
            with parquet.ParquetWriter(
                temporary,
                reader.schema,
                compression=compression,
                use_dictionary=True,
                write_statistics=True,
            ) as writer:
                for batch in reader:
                    writer.write_batch(batch)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _open_streaming_reader(source: Path, delimiter: str | None, block_size: int):
    suffix = source.suffix.lower()
    if suffix in DELIMITED_SUFFIXES:
        actual_delimiter = delimiter or _detect_delimiter(source)
        if len(actual_delimiter) != 1:
            raise ValueError("delimiter must contain exactly one character.")
        return arrow_csv.open_csv(
            source,
            read_options=arrow_csv.ReadOptions(block_size=block_size, use_threads=True),
            parse_options=arrow_csv.ParseOptions(delimiter=actual_delimiter),
        )
    return arrow_json.open_json(
        source,
        read_options=arrow_json.ReadOptions(block_size=block_size, use_threads=True),
    )


def _detect_delimiter(source: Path) -> str:
    if source.suffix.lower() == ".csv":
        return ","
    if source.suffix.lower() == ".tsv":
        return "\t"
    with source.open(encoding="utf-8", errors="replace") as handle:
        sample = handle.read(8192)
    try:
        return stdlib_csv.Sniffer().sniff(sample, delimiters=",\t|;").delimiter
    except stdlib_csv.Error as exc:
        raise ValueError(
            f"Could not detect a delimiter for {source}; pass delimiter explicitly."
        ) from exc
