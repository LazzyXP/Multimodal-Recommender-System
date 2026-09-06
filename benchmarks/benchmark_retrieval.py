"""Benchmark multimodal catalog retrieval backends.

Example:
    python benchmarks/benchmark_retrieval.py --items 100000 --users 1000 \
        --backend flat,hnsw,ivf,ivfpq
"""

from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from mmrec.config import ColumnConfig
from mmrec.data import DatasetBundle
from mmrec.models.multimodal_item_knn import MultiModalItemKNNModel


def peak_rss_mb() -> float | None:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(value / (1024 * 1024 if platform.system() == "Darwin" else 1024), 1)
    except (ImportError, AttributeError):
        return None


def run_once(items: int, users: int, dimensions: int, seed: int, backend: str) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    item_ids = [f"i{i}" for i in range(items)]
    user_ids = [f"u{i}" for i in range(users)]
    vectors = rng.normal(size=(items, dimensions)).astype(np.float32)
    user_history = rng.integers(0, items, size=(users, 5))
    interactions = pd.DataFrame(
        {
            "user_id": np.repeat(user_ids, 5),
            "item_id": [item_ids[index] for index in user_history.reshape(-1)],
        }
    )
    item_frame = pd.DataFrame({"item_id": item_ids, "embedding": list(vectors)})
    dataset = DatasetBundle(
        interactions=interactions,
        users=None,
        items=item_frame,
        modalities={"item": {"embedding": "embedding"}},
        sources={"interactions": None, "users": None, "items": None},
    )
    model = MultiModalItemKNNModel(
        ColumnConfig(),
        use_ann=backend != "flat",
        ann_backend=backend,
        ann_topk=20,
        ann_nlist=64,
        ann_nprobe=8,
        ann_pq_m=max(1, dimensions // 8),
    )
    started = perf_counter()
    model.fit(interactions, dataset)
    fit_seconds = perf_counter() - started
    effective_backend = backend if model._ann_index is not None else "numpy"
    started = perf_counter()
    result = model.recommend(user_ids, k=20)
    recommend_seconds = perf_counter() - started
    exact_recalls: list[float] = []
    for user_id in user_ids:
        history = set(interactions.loc[interactions["user_id"] == user_id, "item_id"])
        profile = np.mean(
            [model.item_vectors[item_id] for item_id in history], axis=0
        )
        norm = np.linalg.norm(profile)
        if norm == 0:
            continue
        exact_scores = model.feature_matrix @ (profile / norm)
        for item_id in history:
            index = model.item_indices.get(item_id)
            if index is not None:
                exact_scores[index] = -np.inf
        limit = min(20, len(model.items))
        exact_indices = np.argpartition(-exact_scores, limit - 1)[:limit]
        exact_items = {model.items[index] for index in exact_indices}
        observed = set(result.loc[result["user_id"] == user_id, "item_id"])
        exact_recalls.append(len(exact_items & observed) / limit)
    return {
        "backend": backend,
        "effective_backend": effective_backend,
        "items": items,
        "users": users,
        "dimensions": dimensions,
        "seed": seed,
        "fit_seconds": round(fit_seconds, 3),
        "recommend_seconds": round(recommend_seconds, 3),
        "users_per_second": round(users / recommend_seconds, 1),
        "recommendation_rows": len(result),
        "recall_at_20_vs_exact": round(float(np.mean(exact_recalls)), 4),
        "peak_rss_mb": peak_rss_mb(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=20_000)
    parser.add_argument("--users", type=int, default=1_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument(
        "--seeds", default="0", help="Comma-separated random seeds for repeated runs."
    )
    parser.add_argument("--backend", default="flat,hnsw,ivf,ivfpq")
    args = parser.parse_args()
    if min(args.items, args.users, args.dimensions) <= 0:
        raise ValueError("items, users, and dimensions must be positive")
    backends = [value.strip() for value in args.backend.split(",") if value.strip()]
    try:
        seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("seeds must be a comma-separated list of integers") from exc
    if not seeds:
        raise ValueError("seeds must contain at least one integer")
    allowed = {"flat", "hnsw", "ivf", "ivfpq"}
    unknown = sorted(set(backends) - allowed)
    if not backends or unknown:
        raise ValueError(f"backend must contain only {sorted(allowed)}")
    runs = [
        run_once(args.items, args.users, args.dimensions, seed, backend)
        for seed in seeds
        for backend in backends
    ]
    aggregate: dict[str, dict[str, float]] = {}
    for backend in backends:
        backend_runs = [run for run in runs if run["backend"] == backend]
        metrics = ("fit_seconds", "recommend_seconds", "users_per_second", "recall_at_20_vs_exact")
        for key in metrics:
            values = np.asarray([float(run[key]) for run in backend_runs])
            aggregate.setdefault(backend, {})[f"{key}_mean"] = round(float(values.mean()), 4)
            aggregate[backend][f"{key}_std"] = round(float(values.std()), 4)
    print(json.dumps({"runs": runs, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
