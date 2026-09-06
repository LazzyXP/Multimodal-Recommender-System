"""Run the paper-aligned evaluation protocol on a dataset directory.

Standard multimodal-recommendation protocol: per-user leave-one-out temporal
split, top-k recommendation, Recall@20 / NDCG@20 / MRR@20. Point it at a data
directory and it trains, evaluates and exports the leaderboard -- this is the
metric evidence used to judge whether the models reproduce paper results.

Usage:
    python benchmarks/evaluate.py /path/to/dataset --time-limit 3600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmrec import MultiModalRecommender


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="dataset directory (interactions + items)")
    parser.add_argument("--time-limit", type=float, default=3600)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--models", default="auto")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=Path("artifacts/report"))
    args = parser.parse_args()

    models = args.models
    if models != "auto" and "," in models:
        models = [name.strip() for name in models.split(",")]

    recommender = MultiModalRecommender(
        eval_metrics=["recall@20", "ndcg@20", "mrr@20", "hit_rate@20", "map@20"],
    )
    recommender.fit(
        args.dataset,
        models=models,
        presets=args.presets,
        time_limit=args.time_limit,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
    )
    print(recommender)
    if args.export is not None:
        recommender.evaluation_report.export(args.export)
        print(json.dumps({"report_exported": str(args.export.resolve())}))


if __name__ == "__main__":
    main()
