"""Shared recommendation evaluation protocol and ranking metrics."""

from __future__ import annotations

from math import log2
from typing import Any

import pandas as pd

from mmrec.config import ColumnConfig


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    truth: pd.DataFrame,
    columns: ColumnConfig,
    metrics: list[str],
    catalog_size: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    requested = [_parse_metric(metric) for metric in metrics]
    user_col = columns.user_id
    item_col = columns.item_id
    truth_by_user = {
        user_id: set(group[item_col].tolist())
        for user_id, group in truth.groupby(user_col, sort=False)
    }
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, float]] = {}

    for model_name, model_frame in recommendations.groupby("model", sort=False):
        model_metrics: dict[str, float] = {}
        predictions = {
            user_id: group.sort_values("rank")[item_col].tolist()
            for user_id, group in model_frame.groupby(user_col, sort=False)
        }
        for original, metric_name, cutoff in requested:
            values = [
                _metric_for_user(
                    metric_name,
                    predictions.get(user_id, [])[:cutoff],
                    relevant,
                )
                for user_id, relevant in truth_by_user.items()
            ]
            model_metrics[original] = sum(values) / len(values) if values else float("nan")

        recommended_items = set(model_frame[item_col].tolist())
        model_metrics["coverage"] = len(recommended_items) / catalog_size if catalog_size else 0.0
        details[model_name] = model_metrics
        rows.append({"model": model_name, **model_metrics})

    leaderboard = pd.DataFrame(rows)
    primary_metric = metrics[0]
    if not leaderboard.empty and primary_metric in leaderboard:
        leaderboard = leaderboard.sort_values(primary_metric, ascending=False, kind="stable")
    return leaderboard.reset_index(drop=True), details


def max_metric_cutoff(metrics: list[str]) -> int:
    return max(_parse_metric(metric)[2] for metric in metrics)


def _parse_metric(metric: str) -> tuple[str, str, int]:
    normalized = metric.strip().lower()
    try:
        name, cutoff_text = normalized.split("@", maxsplit=1)
        cutoff = int(cutoff_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Metric {metric!r} must use the form name@k, for example ndcg@10."
        ) from exc
    aliases = {"hitrate": "hit_rate", "hit": "hit_rate"}
    name = aliases.get(name, name)
    if name not in {"recall", "ndcg", "mrr", "hit_rate", "map", "precision", "auc"}:
        raise ValueError(f"Unsupported metric {metric!r}.")
    if cutoff <= 0:
        raise ValueError("Metric cutoff must be positive.")
    return normalized, name, cutoff


def _metric_for_user(name: str, predictions: list[Any], relevant: set[Any]) -> float:
    if not relevant:
        return 0.0
    hits = [1 if item in relevant else 0 for item in predictions]
    if name == "recall":
        return sum(hits) / len(relevant)
    if name == "hit_rate":
        return float(any(hits))
    if name == "precision":
        return sum(hits) / len(predictions) if predictions else 0.0
    if name == "mrr":
        return next((1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)
    if name == "ndcg":
        dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        ideal_hits = min(len(relevant), len(predictions))
        ideal = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / ideal if ideal else 0.0
    if name == "map":
        hits_so_far = 0
        precision_sum = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                hits_so_far += 1
                precision_sum += hits_so_far / rank
        return precision_sum / len(relevant) if relevant else 0.0
    if name == "auc":
        # List-truncated AUC over the returned ranked list.
        if not predictions:
            return 0.0
        positives = sum(hits)
        negatives = len(predictions) - positives
        if positives == 0:
            return 0.0
        if negatives == 0:
            return 1.0
        negatives_above = 0
        correct = 0
        for is_relevant in hits:
            if is_relevant:
                correct += negatives - negatives_above
            else:
                negatives_above += 1
        return correct / (positives * negatives)
    raise AssertionError(f"Unhandled metric {name}")
