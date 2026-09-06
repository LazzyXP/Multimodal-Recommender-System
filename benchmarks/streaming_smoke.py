"""Generate synthetic interactions and exercise the bounded-memory execution path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

from mmrec import MultiModalRecommender


def generate(path: Path, rows: int, users: int, items: int, batch_size: int) -> None:
    writer: parquet.ParquetWriter | None = None
    try:
        for start in range(0, rows, batch_size):
            stop = min(start + batch_size, rows)
            values = np.arange(start, stop, dtype=np.int64)
            frame = pd.DataFrame(
                {
                    "user_id": values % users,
                    "item_id": (values * 17 + values // 7) % items,
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.to_timedelta(values, unit="s"),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = parquet.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--users", type=int, default=100_000)
    parser.add_argument("--items", type=int, default=50_000)
    parser.add_argument("--sample-rows", type=int, default=200_000)
    parser.add_argument("--workdir", type=Path, default=Path("artifacts/benchmark"))
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    interactions = args.workdir / "interactions.parquet"

    started = perf_counter()
    generate(interactions, args.rows, args.users, args.items, batch_size=250_000)
    generated_seconds = perf_counter() - started

    recommender = MultiModalRecommender(
        execution_mode="streaming",
        sample_interactions=args.sample_rows,
        cache_dir=args.workdir / "cache",
        history_partitions=64,
        eval_metrics=["recall@20", "ndcg@20"],
    )
    started = perf_counter()
    recommender.fit(interactions, models=["Popularity", "ItemCF"])
    fit_seconds = perf_counter() - started

    target_users = list(range(min(args.users, 1_000)))
    started = perf_counter()
    output = recommender.recommend_to_parquet(
        target_users,
        args.workdir / "recommendations.parquet",
        k=20,
        batch_size=500,
    )
    recommend_seconds = perf_counter() - started
    print(
        json.dumps(
            {
                "rows": args.rows,
                "generated_seconds": round(generated_seconds, 3),
                "fit_seconds": round(fit_seconds, 3),
                "recommend_1000_users_seconds": round(recommend_seconds, 3),
                "recommendation_rows": parquet.ParquetFile(output).metadata.num_rows,
                "recommendations_bytes": output.stat().st_size,
                "summary": recommender.fit_summary().dataset_summary,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
