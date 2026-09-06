"""End-to-end AutoML benchmark producing timing/throughput evidence.

Usage:
    python benchmarks/benchmark.py --users 5000 --items 20000 --rows 200000
"""

from __future__ import annotations

import argparse
import json
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

from mmrec import MultiModalRecommender


def generate(users: int, items: int, rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    user_ids = rng.integers(0, users, rows)
    item_ids = rng.integers(0, items, rows)
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in user_ids],
            "item_id": [f"i{i}" for i in item_ids],
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="s"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=5_000)
    parser.add_argument("--items", type=int, default=20_000)
    parser.add_argument("--rows", type=int, default=200_000)
    args = parser.parse_args()

    interactions = generate(args.users, args.items, args.rows)

    with tempfile.TemporaryDirectory() as cache:
        recommender = MultiModalRecommender(
            eval_metrics=["recall@20", "ndcg@20", "map@20"],
            cache_dir=cache,
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

        print(
            json.dumps(
                {
                    "rows": args.rows,
                    "users": args.users,
                    "items": args.items,
                    "fit_seconds": round(fit_seconds, 3),
                    "recommend_1000_users_seconds": round(recommend_seconds, 3),
                    "recommend_throughput_users_per_sec": round(1000 / recommend_seconds, 1),
                    "recommendation_rows": len(result.data),
                    "models": sorted(recommender.models),
                    "leaderboard": recommender.leaderboard().to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
