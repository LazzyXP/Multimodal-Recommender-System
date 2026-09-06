"""Configuration objects used by the public API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ColumnConfig:
    user_id: str = "user_id"
    item_id: str = "item_id"
    timestamp: str | None = "timestamp"
    label: str | None = None

    def required_interaction_columns(self) -> list[str]:
        columns = [self.user_id, self.item_id]
        if self.timestamp:
            columns.append(self.timestamp)
        if self.label:
            columns.append(self.label)
        return columns


@dataclass(slots=True)
class RunConfig:
    columns: ColumnConfig
    eval_metrics: list[str] = field(default_factory=lambda: ["recall@10", "ndcg@10"])
    modalities: dict[str, Any] = field(default_factory=dict)
    preset: str = "medium_quality"
    random_state: int = 42
    execution_mode: str = "auto"
    model_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    max_inference_score_mb: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, list[str]] = {
    "fast_training": ["Popularity", "ItemCF"],
    "medium_quality": ["Popularity", "ItemCF", "BPRMF"],
    "best_quality": ["Popularity", "ItemCF", "BPRMF"],
    "multimodal_high_quality": ["Popularity", "ItemCF", "BPRMF"],
}

# Per-preset training intensity applied under user `model_configs`. These give the
# presets genuinely different behaviour even when they share the same candidate set.
PRESET_INTENSITY: dict[str, dict[str, dict[str, Any]]] = {
    "fast_training": {
        "BPRMF": {"epochs": 2, "max_training_pairs": 20_000},
        "VBPR": {"epochs": 1, "max_training_pairs": 20_000},
    },
    "medium_quality": {
        "BPRMF": {"epochs": 5, "max_training_pairs": 100_000},
        "VBPR": {"epochs": 3, "max_training_pairs": 50_000},
    },
    "best_quality": {
        "BPRMF": {"epochs": 15, "factors": 64, "max_training_pairs": 200_000},
        "VBPR": {"epochs": 8, "id_factors": 48, "max_training_pairs": 100_000},
    },
    "multimodal_high_quality": {
        "BPRMF": {"epochs": 10, "factors": 64, "max_training_pairs": 150_000},
        "VBPR": {"epochs": 6, "id_factors": 48, "max_training_pairs": 100_000},
    },
}

SPLIT_STRATEGIES = ("temporal", "random", "cold_start", "global_temporal")

# Small per-model hyperparameter search spaces used by ``hyperparameter_tune``.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "BPRMF": {
        "factors": [16, 32, 64],
        "learning_rate": [0.01, 0.03, 0.1],
    },
    "VBPR": {
        "id_factors": [16, 24, 48],
        "learning_rate": [0.01, 0.02, 0.05],
    },
}

# Shared space applied to the optional torch graph family. ``epochs`` is kept out
# of the space: it is the training budget, chosen by the preset and the coarse ->
# full successive halving, not a TPE-modelled hyperparameter (observing a suggested
# ``epochs`` against a coarse ``epochs=2`` evaluation would contaminate the history).
TORCH_SEARCH_SPACE: dict[str, list[Any]] = {
    "factors": [16, 32, 64],
    "layers": [1, 2, 3],
}


def preset_intensity(preset: str) -> dict[str, dict[str, Any]]:
    """Return default per-model training configuration for a preset."""
    try:
        intensity = PRESET_INTENSITY[preset]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {choices}") from exc
    return {model: dict(config) for model, config in intensity.items()}


@dataclass(slots=True)
class ExecutionConfig:
    mode: str = "auto"
    max_in_memory_interactions: int = 2_000_000
    sample_interactions: int = 1_000_000
    scan_batch_size: int = 262_144
    inference_batch_size: int = 10_000
    history_partitions: int = 256
    max_catalog_items: int = 2_000_000
    max_sample_history_per_user: int = 200

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "in_memory", "streaming"}:
            raise ValueError("execution_mode must be auto, in_memory, or streaming.")
        for name in (
            "max_in_memory_interactions",
            "sample_interactions",
            "scan_batch_size",
            "inference_batch_size",
            "history_partitions",
            "max_catalog_items",
            "max_sample_history_per_user",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")


def resolve_models(models: str | list[str] | None, preset: str) -> list[str]:
    if models is None or models == "auto":
        try:
            return PRESETS[preset].copy()
        except KeyError as exc:
            choices = ", ".join(sorted(PRESETS))
            raise ValueError(f"Unknown preset {preset!r}. Available presets: {choices}") from exc
    if isinstance(models, str):
        return [models]
    if not models:
        raise ValueError("At least one model must be requested.")
    return list(dict.fromkeys(models))
