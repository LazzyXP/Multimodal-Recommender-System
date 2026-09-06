"""Public API for mmrec."""

from mmrec.predictor import MultiModalRecommender
from mmrec.results import EvaluationReport, FitResult, RecommendationResult
from mmrec.storage import convert_to_parquet

__all__ = [
    "EvaluationReport",
    "FitResult",
    "MultiModalRecommender",
    "RecommendationResult",
    "convert_to_parquet",
]

__version__ = "0.1.0"
