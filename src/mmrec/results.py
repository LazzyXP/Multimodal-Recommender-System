"""Structured return objects for fitting, recommendation, and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class FitResult:
    models: pd.DataFrame
    dataset_summary: dict[str, Any]

    def leaderboard(self) -> pd.DataFrame:
        return self.models.copy()


@dataclass(slots=True)
class RecommendationResult:
    data: pd.DataFrame

    @property
    def models(self) -> list[str]:
        return self.data["model"].drop_duplicates().tolist()

    def for_model(self, model: str) -> pd.DataFrame:
        return self.data[self.data["model"] == model].reset_index(drop=True).copy()

    def __getitem__(self, model: str) -> pd.DataFrame:
        return self.for_model(model)

    def to_parquet(self, path: str | Path) -> None:
        self.data.to_parquet(path, index=False)

    def to_csv(self, path: str | Path) -> None:
        self.data.to_csv(path, index=False)


@dataclass(slots=True)
class EvaluationReport:
    leaderboard_data: pd.DataFrame
    details: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def leaderboard(self) -> pd.DataFrame:
        return self.leaderboard_data.copy()

    def metrics(self, model: str) -> dict[str, float]:
        try:
            return self.details[model].copy()
        except KeyError as exc:
            raise KeyError(f"No evaluation metrics found for model {model!r}.") from exc

    def export(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.leaderboard_data.to_csv(target / "leaderboard.csv", index=False)
        (target / "metrics.json").write_text(
            json.dumps(self.details, indent=2, sort_keys=True), encoding="utf-8"
        )
        (target / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return target
