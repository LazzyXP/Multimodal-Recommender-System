"""Internal model benchmark for comparing quality, latency, and GPU memory.

This is an experiment tool, not part of the package publishing workflow. Run it
on a machine with the optional torch runtime, for example::

    python benchmarks/gpu_benchmark.py --models LightGCN --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mmrec import MultiModalRecommender


def make_interactions(users: int, items: int, history: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        (f"u{user}", f"i{item}", pd.Timestamp("2026-01-01") + pd.Timedelta(seconds=index))
        for index, (user, item) in enumerate(
            
                (user, int(item))
                for user in range(users)
                for item in rng.choice(items, history, replace=False)
            
        )
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])


def gpu_memory() -> int | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated() / 1024**2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--items", type=int, default=2_000)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        default="LightGCN,MMGCN,BM3",
        help="Comma-separated model names; use Popularity,ItemCF,BPRMF for core baselines.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    interactions = make_interactions(args.users, args.items, args.history, args.seed)
    target_users = [f"u{index}" for index in range(min(100, args.users))]
    rows: list[dict[str, object]] = []
    for name in [item.strip() for item in args.models.split(",") if item.strip()]:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
        except ImportError:
            torch = None
        started = time.perf_counter()
        try:
            predictor = MultiModalRecommender(
                eval_metrics=["recall@20", "ndcg@20"],
                random_state=args.seed,
                cache_dir=".mmrec/benchmark-cache",
            )
            config = {"epochs": args.epochs, "factors": args.factors, "device": args.device}
            predictor.fit(interactions, models=name, model_configs={name: config})
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            result = predictor.recommend(target_users, k=args.k)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            rows.append(
                {
                    "model": name,
                    "status": "succeeded",
                    "fit_seconds": round(fit_seconds, 4),
                    "recommend_seconds": round(time.perf_counter() - started, 4),
                    "gpu_memory_mb": gpu_memory(),
                    "recommendation_rows": len(result.data),
                    "leaderboard": predictor.leaderboard().to_dict(orient="records"),
                }
            )
        except Exception as exc:
            rows.append(
                {"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )

    payload = {
        "seed": args.seed,
        "users": args.users,
        "items": args.items,
        "history": args.history,
        "device": args.device,
        "results": rows,
    }
    rendered = json.dumps(payload, indent=2, default=str)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
