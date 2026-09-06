"""End-to-end AutoML benchmark producing timing/throughput evidence.

Usage:
    python benchmarks/benchmark.py --users 5000 --items 20000 --rows 200000
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from mmrec import MultiModalRecommender


def peak_rss_mb() -> float | None:
    """Return process peak RSS when the host exposes it (best-effort)."""
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        return round(value / (1024 * 1024 if platform.system() == "Darwin" else 1024), 1)
    except (ImportError, AttributeError):
        return None


def generate(users: int, items: int, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    user_ids = rng.integers(0, users, rows)
    item_ids = rng.integers(0, items, rows)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in user_ids],
            "item_id": [f"i{i}" for i in item_ids],
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s"),
        }
    )


def run_once(args: argparse.Namespace, seed: int) -> dict[str, object]:
    if args.input:
        interactions = (
            pd.read_parquet(args.input)
            if args.input.suffix.lower() in {".parquet", ".pq"}
            else pd.read_csv(args.input)
        )
        missing = sorted({"user_id", "item_id"} - set(interactions.columns))
        if missing:
            raise ValueError(f"Benchmark input is missing required columns: {missing}")
        source = str(args.input)
        timestamp = "timestamp" if "timestamp" in interactions else None
    else:
        interactions = generate(args.users, args.items, args.rows, seed)
        source = "synthetic"
        timestamp = "timestamp"

    with tempfile.TemporaryDirectory() as cache:
        recommender = MultiModalRecommender(
            eval_metrics=["recall@20", "ndcg@20", "map@20"],
            cache_dir=cache,
            timestamp=timestamp,
        )
        started = perf_counter()
        recommender.fit(
            interactions,
            models=["Popularity", "ItemCF", "BPRMF"],
            presets="fast_training",
            num_workers=3,
        )
        fit_seconds = perf_counter() - started

        started = perf_counter()
        target = [f"u{i}" for i in range(1000)]
        result = recommender.recommend(target, k=20)
        recommend_seconds = perf_counter() - started

        return {
                    "rows": len(interactions),
                    "users": int(interactions["user_id"].nunique()),
                    "items": int(interactions["item_id"].nunique()),
                    "source": source,
                    "seed": seed,
                    "fit_seconds": round(fit_seconds, 3),
                    "recommend_1000_users_seconds": round(recommend_seconds, 3),
                    "recommend_throughput_users_per_sec": round(1000 / recommend_seconds, 1),
                    "recommendation_rows": len(result.data),
                    "peak_rss_mb": peak_rss_mb(),
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "models": sorted(recommender.models),
                    "leaderboard": recommender.leaderboard().to_dict(orient="records"),
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=5_000)
    parser.add_argument("--items", type=int, default=20_000)
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--input", type=Path, help="Real interactions CSV or Parquet file.")
    parser.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated random seeds; multiple values produce aggregate statistics.",
    )
    args = parser.parse_args()
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated list of integers.") from exc
    if not seeds:
        raise ValueError("--seeds must contain at least one integer.")
    results = [run_once(args, seed) for seed in seeds]
    aggregate: dict[str, object] = {}
    if len(results) > 1:
        metric_keys = (
            "fit_seconds",
            "recommend_1000_users_seconds",
            "recommend_throughput_users_per_sec",
            "peak_rss_mb",
        )
        for key in metric_keys:
            values = [float(result[key]) for result in results if result[key] is not None]
            if values:
                aggregate[f"{key}_mean"] = round(float(np.mean(values)), 3)
                aggregate[f"{key}_std"] = round(float(np.std(values)), 3)
    print(json.dumps({"runs": results, "aggregate": aggregate}, indent=2, default=str))


if __name__ == "__main__":
    main()
