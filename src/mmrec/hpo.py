"""Dependency-free Tree-structured Parzen Estimator for discrete search spaces.

The standard TPE maintains two Parzen (kernel density) estimators: ``l`` over
the best-scoring observations and ``g`` over the rest. Candidates are drawn from
``l`` and ranked by the density ratio ``l(x) / g(x)``, which favors values that
are common among good trials and rare among bad ones. For a discrete space the
Parzen estimator is a smoothed categorical, so the implementation is exact
rather than an approximation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


class TpeSearch:
    """Sequential TPE sampler over a discrete search space."""

    def __init__(
        self,
        space: dict[str, list[Any]],
        random_state: int = 42,
        gamma: float = 0.25,
        warmup: int = 3,
        candidates: int = 24,
    ) -> None:
        self.space = space
        self.gamma = gamma
        self.warmup = warmup
        self.candidates = candidates
        self.rng = np.random.default_rng(random_state)
        self._history: list[tuple[dict[str, Any], float]] = []

    def suggest(self) -> dict[str, Any]:
        if len(self._history) < self.warmup:
            return self._random_config()
        return self._tpe_config()

    def observe(self, config: dict[str, Any], score: float) -> None:
        self._history.append((dict(config), float(score)))

    def _random_config(self) -> dict[str, Any]:
        return {
            parameter: choices[int(self.rng.integers(len(choices)))]
            for parameter, choices in self.space.items()
        }

    def _tpe_config(self) -> dict[str, Any]:
        scores = np.asarray([score for _, score in self._history], dtype=np.float64)
        threshold = float(np.quantile(scores, 1.0 - self.gamma))
        good = [config for config, score in self._history if score >= threshold]
        bad = [config for config, score in self._history if score < threshold]
        if not good:
            good = [config for config, _ in self._history]
        if not bad:
            bad = good
        config: dict[str, Any] = {}
        for parameter, choices in self.space.items():
            config[parameter] = self._sample_parameter(parameter, choices, good, bad)
        return config

    def _sample_parameter(
        self,
        parameter: str,
        choices: list[Any],
        good: list[dict[str, Any]],
        bad: list[dict[str, Any]],
    ) -> Any:
        good_counts = Counter(config[parameter] for config in good)
        bad_counts = Counter(config[parameter] for config in bad)
        # Smoothed categorical Parzen estimators over the good and bad splits.
        alpha = 1.0
        num_good = len(good) + alpha * len(choices)
        num_bad = len(bad) + alpha * len(choices)
        density_good = np.asarray(
            [(good_counts.get(choice, 0) + alpha) / num_good for choice in choices],
            dtype=np.float64,
        )
        density_bad = np.asarray(
            [(bad_counts.get(choice, 0) + alpha) / num_bad for choice in choices],
            dtype=np.float64,
        )
        # Draw candidates from l and keep the one maximizing l(x) / g(x).
        indices = self.rng.choice(len(choices), size=self.candidates, p=density_good)
        ratios = density_good[indices] / density_bad[indices]
        return choices[int(indices[np.argmax(ratios)])]
