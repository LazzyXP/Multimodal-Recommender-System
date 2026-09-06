"""AutoML-style public entry point for multimodel recommendation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as parquet

from mmrec.config import (
    SEARCH_SPACES,
    SPLIT_STRATEGIES,
    TORCH_SEARCH_SPACE,
    ColumnConfig,
    ExecutionConfig,
    RunConfig,
    preset_intensity,
    resolve_models,
)
from mmrec.data import (
    DatasetBundle,
    TableInput,
    cold_start_split,
    global_temporal_split,
    leave_one_out_split,
    prepare_dataset,
    random_leave_one_out_split,
)
from mmrec.ensemble import fit_ensemble_weights, reciprocal_rank_fusion
from mmrec.evaluation import evaluate_recommendations, max_metric_cutoff
from mmrec.hpo import TpeSearch
from mmrec.models import torch_available
from mmrec.models.base import BaseRecommendationModel
from mmrec.registry import ModelFactory, ModelRegistry
from mmrec.results import EvaluationReport, FitResult, RecommendationResult
from mmrec.scalable import (
    build_history_index,
    iter_column_batches,
    parquet_row_count,
    parquet_size_bytes,
    scan_and_sample_interactions,
    seen_items_for_users,
    seen_items_from_index,
)
from mmrec.storage import ParquetCache, file_fingerprint, frame_fingerprint


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MultiModalRecommender:
    """Train, compare, and fuse multiple recommendation models.

    The initial built-in models consume interaction data. User and item modality
    declarations are validated and persisted so multimodal models can plug into
    the same training and evaluation protocol.
    """

    def __init__(
        self,
        user_id: str = "user_id",
        item_id: str = "item_id",
        timestamp: str | None = "timestamp",
        label: str | None = None,
        eval_metrics: list[str] | None = None,
        random_state: int = 42,
        cache_dir: str | Path = ".mmrec/cache",
        execution_mode: str = "auto",
        max_in_memory_interactions: int = 2_000_000,
        sample_interactions: int = 1_000_000,
        scan_batch_size: int = 262_144,
        inference_batch_size: int = 10_000,
        history_partitions: int = 256,
        max_catalog_items: int = 2_000_000,
        max_sample_history_per_user: int = 200,
        eval_metric: str | None = None,
        max_inference_score_mb: int = 64,
    ) -> None:
        self.columns = ColumnConfig(user_id, item_id, timestamp, label)
        self.eval_metrics = eval_metrics or ["recall@10", "ndcg@10", "mrr@10"]
        self.eval_metric = eval_metric or self.eval_metrics[0]
        self.eval_metrics = list(dict.fromkeys([self.eval_metric, *self.eval_metrics]))
        self.model_best: str | None = None
        self.validation_report: EvaluationReport | None = None
        self._deployment_histories: dict[Any, set[Any]] | None = None
        self._validation_histories: dict[Any, set[Any]] = {}
        # Validate metric syntax at construction time.
        max_metric_cutoff(self.eval_metrics)
        self.random_state = random_state
        if max_inference_score_mb <= 0:
            raise ValueError("max_inference_score_mb must be positive.")
        self.max_inference_score_mb = max_inference_score_mb
        self.cache_dir = Path(cache_dir)
        self.execution = ExecutionConfig(
            mode=execution_mode,
            max_in_memory_interactions=max_in_memory_interactions,
            sample_interactions=sample_interactions,
            scan_batch_size=scan_batch_size,
            inference_batch_size=inference_batch_size,
            history_partitions=history_partitions,
            max_catalog_items=max_catalog_items,
            max_sample_history_per_user=max_sample_history_per_user,
        )
        self.registry = ModelRegistry()
        self.models: dict[str, BaseRecommendationModel] = {}
        self.training_users: list[Any] = []
        self.catalog_size = 0
        self.data_sources: dict[str, str | None] = {}
        self.modalities: dict[str, Any] = {}
        self.run_config: RunConfig | None = None
        self.fit_result: FitResult | None = None
        self.evaluation_report: EvaluationReport | None = None
        self.ensemble_weights: dict[str, float] = {}
        self.is_streaming = False
        self.history_index: str | None = None

    def register_model(
        self,
        name: str,
        factory: ModelFactory,
        overwrite: bool = False,
    ) -> MultiModalRecommender:
        self.registry.register(name, factory, overwrite=overwrite)
        return self

    def available_models(self) -> list[str]:
        return self.registry.available()

    def model_catalog(self) -> pd.DataFrame:
        return pd.DataFrame(self.registry.catalog())

    def __repr__(self) -> str:
        if not self.models:
            return "MultiModalRecommender (not fitted)"
        leaderboard = self.leaderboard()
        return (
            f"MultiModalRecommender (fitted: {len(self.models)} models)\n{leaderboard.to_string()}"
        )

    def fit(
        self,
        interactions: TableInput,
        users: TableInput | None = None,
        items: TableInput | None = None,
        modalities: dict[str, Any] | None = None,
        models: str | list[str] | None = "auto",
        presets: str = "medium_quality",
        model_configs: dict[str, dict[str, Any]] | None = None,
        time_limit: float | None = None,
        num_workers: int = 1,
        split: str = "temporal",
        hyperparameter_tune: bool = False,
        n_trials: int = 4,
        checkpoint_dir: str | Path | None = None,
        validation_data: TableInput | None = None,
        test_data: TableInput | None = None,
        refit_full: bool | str = "best",
    ) -> MultiModalRecommender:
        if refit_full not in (False, True, "best", "all"):
            raise ValueError("refit_full must be False, True, best, or all.")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive when provided.")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1.")
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if split not in SPLIT_STRATEGIES:
            choices = ", ".join(SPLIT_STRATEGIES)
            raise ValueError(f"Unknown split strategy {split!r}. Available strategies: {choices}")
        fit_deadline = None if time_limit is None else perf_counter() + time_limit

        # Turnkey entry: a directory is auto-discovered (files, columns, modalities).
        if isinstance(interactions, (str, Path)):
            interactions_path = Path(interactions)
            if interactions_path.is_dir():
                from mmrec.discovery import discover_dataset

                discovery = discover_dataset(interactions_path)
                self.columns = ColumnConfig(
                    user_id=discovery.user_id,
                    item_id=discovery.item_id,
                    timestamp=discovery.timestamp,
                    label=discovery.label,
                )
                interactions = discovery.interactions
                users = users if users is not None else discovery.users
                items = items if items is not None else discovery.items
                modalities = (
                    modalities if modalities is not None else (discovery.modalities or None)
                )

        cache = ParquetCache(self.cache_dir)
        interaction_materialized = (
            cache.materialize_frame(interactions, "interactions")
            if isinstance(interactions, pd.DataFrame)
            else cache.materialize(interactions, "interactions")
        )
        if fit_deadline is not None and perf_counter() >= fit_deadline:
            raise TimeoutError("Data preparation exceeded the requested time_limit.")
        interaction_rows = parquet_row_count(interaction_materialized.parquet)
        self.is_streaming = self.execution.mode == "streaming" or (
            self.execution.mode == "auto"
            and interaction_rows > self.execution.max_in_memory_interactions
        )

        full_item_counts: dict[Any, int] | None = None
        catalog_truncated = False
        fit_interactions: TableInput = interaction_materialized.parquet
        if self.is_streaming:
            scan_result = scan_and_sample_interactions(
                interaction_materialized.parquet,
                self.columns,
                self.execution.sample_interactions,
                self.execution.scan_batch_size,
                self.execution.max_catalog_items,
                self.execution.max_sample_history_per_user,
            )
            fit_interactions = scan_result.sample
            full_item_counts = dict(scan_result.item_counts)
            interaction_rows = scan_result.interactions
            catalog_truncated = scan_result.catalog_truncated
            self.history_index = str(
                build_history_index(
                    interaction_materialized.parquet,
                    self.cache_dir,
                    self.columns,
                    self.execution.history_partitions,
                    self.execution.scan_batch_size,
                )
            )

        dataset = prepare_dataset(
            fit_interactions,
            users,
            items,
            self.columns,
            modalities,
            cache_dir=self.cache_dir,
        )
        if fit_deadline is not None and perf_counter() >= fit_deadline:
            raise TimeoutError("Data preparation exceeded the requested time_limit.")
        model_names = resolve_models(models, presets)
        if models is None or models == "auto":
            model_names = self._add_automatic_multimodal_models(
                model_names,
                dataset,
                presets,
            )
        preset_configs = preset_intensity(presets)
        merged_configs = {
            name: {**preset_configs.get(name, {}), **(model_configs or {}).get(name, {})}
            for name in model_names
        }
        for name in model_names:
            if self.registry.accepts_parameter(name, "random_state"):
                merged_configs[name].setdefault("random_state", self.random_state)

        # External test data never enters training, HPO, fusion or refitting.
        def holdout(value: TableInput) -> pd.DataFrame:
            return prepare_dataset(
                value, None, None, self.columns, None, cache_dir=self.cache_dir
            ).interactions

        if validation_data is not None:
            train = dataset.interactions
            val = holdout(validation_data)
            test = holdout(test_data) if test_data is not None else train.iloc[0:0].copy()
            train_val = pd.concat([train, val], ignore_index=True)
        else:
            if test_data is not None:
                train_val, test = dataset.interactions, holdout(test_data)
            else:
                train_val, test = self._split_interactions(dataset.interactions, split)
            train, val = self._split_interactions(train_val, split)
        if split == "global_temporal":
            if self.columns.timestamp is None:
                raise ValueError("global_temporal requires a timestamp column.")
            partitions = [part for part in (train, val, test) if not part.empty]
            if any(part[self.columns.timestamp].isna().any() for part in partitions):
                raise ValueError("global_temporal requires non-null timestamps.")
            for earlier, later in zip(partitions, partitions[1:], strict=False):
                if earlier[self.columns.timestamp].max() >= later[self.columns.timestamp].min():
                    raise ValueError("global_temporal requires strictly ordered time windows.")
        if validation_data is not None or test_data is not None:
            keys = [self.columns.user_id, self.columns.item_id]
            if self.columns.timestamp:
                keys.append(self.columns.timestamp)
            for left, right in ((train, val), (train, test), (val, test)):
                if not left[keys].merge(right[keys], on=keys, how="inner").empty:
                    raise ValueError(
                        "Training, validation and test data contain overlapping events."
                    )
        # Refitting consumes input interactions and validation, never an external test set.
        deployment_interactions = (
            pd.concat([dataset.interactions, val], ignore_index=True)
            if validation_data is not None
            else dataset.interactions
        )
        if self.is_streaming and validation_data is not None:
            # Merge external validation into the bounded full-stream popularity counts.
            for item, count in val[self.columns.item_id].value_counts().items():
                if (
                    item in full_item_counts
                    or len(full_item_counts) < self.execution.max_catalog_items
                ):
                    full_item_counts[item] = full_item_counts.get(item, 0) + int(count)
                else:
                    catalog_truncated = True
        self._deployment_histories = None
        self._validation_histories = (
            {
                user: set(group[self.columns.item_id])
                for user, group in val.groupby(self.columns.user_id, sort=False)
            }
            if validation_data is not None
            else {}
        )
        if not self.is_streaming:
            self.history_index = None
            self._deployment_histories = {
                user: set(group[self.columns.item_id])
                for user, group in deployment_interactions.groupby(self.columns.user_id, sort=False)
            }
        deployment_dataset = DatasetBundle(
            deployment_interactions,
            dataset.users,
            dataset.items,
            dataset.modalities,
            dataset.sources,
        )
        evaluation_items = dataset.items
        if evaluation_items is None:
            evaluation_items = dataset.interactions[[self.columns.item_id]].drop_duplicates(
                ignore_index=True
            )
        evaluation_dataset = DatasetBundle(
            interactions=train,
            users=dataset.users,
            items=evaluation_items,
            modalities=dataset.modalities,
            sources=dataset.sources,
        )
        evaluation_k = max_metric_cutoff(self.eval_metrics)
        val_users = val[self.columns.user_id].drop_duplicates().tolist()
        test_users = test[self.columns.user_id].drop_duplicates().tolist()

        results: dict[
            str,
            tuple[
                pd.DataFrame | None,
                pd.DataFrame | None,
                BaseRecommendationModel | None,
                dict[str, Any],
            ],
        ] = {}
        jobs = list(model_names)

        checkpoint_validation = {
            "workflow_version": 2,
            "execution": asdict(self.execution),
            "columns": self.columns,
            "eval_metrics": self.eval_metrics,
            "preset": presets,
            "random_state": self.random_state,
            "hyperparameter_tune": hyperparameter_tune,
            "n_trials": n_trials,
            "model_configs": {
                name: {key: value for key, value in config.items() if key != "time_limit"}
                for name, config in merged_configs.items()
            },
            "split": split,
            "interactions": self._interaction_identity(interactions, interaction_materialized),
            "interaction_rows": interaction_rows,
            "modalities": dataset.modalities,
        }
        if checkpoint_dir is not None:
            # Table identities are only hashed when a checkpoint may be written or
            # read; hashing large multimodal item tables is otherwise wasted work.
            checkpoint_validation["validation_data"] = self._table_identity(val)
            checkpoint_validation["test_data"] = self._table_identity(test)
            checkpoint_validation["users"] = self._table_identity(dataset.users)
            checkpoint_validation["items"] = self._table_identity(dataset.items)
        if checkpoint_dir is not None:
            stored_validation, stored_results = self._load_checkpoint(checkpoint_dir)
            if stored_validation == checkpoint_validation and stored_results:
                done = {
                    name
                    for name, entry in stored_results.items()
                    if name in set(jobs) and entry[2] is not None
                }
                results.update({name: stored_results[name] for name in done})
                jobs = [name for name in jobs if name not in done]

        def budget_exhausted() -> bool:
            return fit_deadline is not None and perf_counter() >= fit_deadline

        def train_serial(name: str) -> None:
            if budget_exhausted():
                results[name] = (None, None, None, self._skip_row(name))
                return
            results[name] = self._train_one_model(
                name,
                train,
                evaluation_dataset,
                val,
                val_users,
                test_users,
                evaluation_k,
                full_item_counts,
                merged_configs,
                hyperparameter_tune,
                n_trials,
                self.eval_metrics[0],
                fit_deadline,
            )
            # Checkpoint after every model so an interrupted run keeps completed work.
            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, results, checkpoint_validation)

        if num_workers <= 1 or len(jobs) <= 1:
            for name in jobs:
                train_serial(name)
        else:
            torch_jobs = [name for name in jobs if self.registry.requires_torch(name)]
            torch_set = set(torch_jobs)
            non_torch_jobs = [name for name in jobs if name not in torch_set]
            if torch_jobs:
                warnings.warn(
                    "PyTorch models are trained serially to avoid oversubscribing "
                    "memory; non-torch models still run in parallel.",
                    stacklevel=2,
                )
            # Complete lightweight candidates first, then train torch models serially.
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        self._train_one_model,
                        name,
                        train,
                        evaluation_dataset,
                        val,
                        val_users,
                        test_users,
                        evaluation_k,
                        full_item_counts,
                        merged_configs,
                        hyperparameter_tune,
                        n_trials,
                        self.eval_metrics[0],
                        fit_deadline,
                    ): name
                    for name in non_torch_jobs
                }
                for future in as_completed(futures):
                    name = futures[future]
                    results[name] = future.result()
                    # Checkpoint after every completed model, matching the serial path.
                    if checkpoint_dir is not None:
                        self._save_checkpoint(checkpoint_dir, results, checkpoint_validation)

            for name in torch_jobs:
                train_serial(name)

        final_models: dict[str, BaseRecommendationModel] = {}
        fit_rows: list[dict[str, Any]] = []
        val_recommendations: list[pd.DataFrame] = []
        test_recommendations: list[pd.DataFrame] = []
        for name in model_names:
            val_recs, test_recs, model, row = results.get(
                name, (None, None, None, self._skip_row(name))
            )
            fit_rows.append(dict(row))
            if model is not None:
                final_models[name] = model
            if val_recs is not None:
                val_recommendations.append(val_recs)
            if test_recs is not None:
                test_recommendations.append(test_recs)

        if checkpoint_dir is not None and final_models:
            self._save_checkpoint(checkpoint_dir, results, checkpoint_validation)

        if not final_models:
            statuses = {row.get("status") for row in fit_rows}
            if statuses <= {"skipped"}:
                raise RuntimeError(
                    "Time budget was exhausted before any model could train; increase time_limit."
                )
            failures = "; ".join(row["error"] or "unknown error" for row in fit_rows)
            raise RuntimeError(f"All requested models failed to train: {failures}")

        self.models = final_models
        self.training_users = (
            []
            if self.is_streaming
            else deployment_interactions[self.columns.user_id].drop_duplicates().tolist()
        )
        self.catalog_size = (
            len(dataset.items)
            if dataset.items is not None
            else len(full_item_counts or next(iter(final_models.values())).items)
        )
        self.data_sources = dataset.sources.copy()
        self.data_sources["interactions"] = str(interaction_materialized.parquet)
        self.modalities = dataset.modalities
        combined_val = self._combine_frames(val_recommendations)
        self.ensemble_weights = (
            fit_ensemble_weights(
                combined_val,
                val,
                self.columns,
                evaluation_k,
                random_state=self.random_state,
            )
            if len(val_recommendations) >= 2 and not val.empty
            else {}
        )
        self.validation_report = self._build_report(
            val_recommendations,
            val,
            self.catalog_size,
            {"protocol": "validation selection; fusion weights fitted on this holdout"},
            weights=self.ensemble_weights,
        )
        selection = self.validation_report.leaderboard()
        if selection.empty:
            self.model_best = next(iter(self.models))
            warnings.warn(
                "No usable validation scores; selecting the first successful model.",
                stacklevel=2,
            )
        else:
            self.model_best = str(
                selection.sort_values(self.eval_metric, ascending=False, kind="stable").iloc[0][
                    "model"
                ]
            )
        selected_dependencies = (
            list(self.models) if self.model_best == "RankFusion" else [self.model_best]
        )
        refit_names = (
            list(self.models)
            if refit_full in (True, "all")
            else selected_dependencies
            if refit_full == "best"
            else []
        )
        # Stage all replacements, so a failed fusion dependency leaves the original
        # ensemble usable instead of deploying a mixture of refitted and old models.
        replacements = {}
        refit_results = {}
        refit_checkpoint = None if checkpoint_dir is None else Path(checkpoint_dir) / "refit"
        refit_validation = {
            **checkpoint_validation,
            "chosen_configs": {row["model"]: row.get("chosen_config") for row in fit_rows},
            "validation_predictions": (
                self._frame_hash(combined_val) if checkpoint_dir is not None else None
            ),
        }
        if refit_checkpoint is not None:
            stored_validation, stored_refits = self._load_checkpoint(refit_checkpoint)
            if stored_validation == refit_validation:
                refit_results = stored_refits
        rows_by_name = {row["model"]: row for row in fit_rows}
        for name in refit_names:
            row = rows_by_name[name]
            if name in refit_results:
                replacements[name], row["refit_time_s"] = refit_results[name]
                row["refit_status"] = "succeeded"
                continue
            if budget_exhausted():
                row["refit_status"] = "skipped_budget"
                continue
            started = perf_counter()
            try:
                replacements[name] = self._fit_model_for_evaluation(
                    name,
                    self._budgeted_config(name, row["chosen_config"], fit_deadline),
                    deployment_interactions,
                    deployment_dataset,
                    full_item_counts,
                    use_full_counts=True,
                )
                row["refit_status"] = "succeeded"
            except Exception as exc:
                row["refit_status"] = "failed"
                row["refit_error"] = f"{type(exc).__name__}: {exc}"
            row["refit_time_s"] = perf_counter() - started
            if name in replacements and refit_checkpoint is not None:
                refit_results[name] = (replacements[name], row["refit_time_s"])
                self._save_checkpoint(refit_checkpoint, refit_results, refit_validation)
        if self.model_best == "RankFusion" and not all(
            name in replacements for name in selected_dependencies
        ):
            for name in replacements:
                rows_by_name[name]["refit_status"] = "discarded_incomplete_ensemble"
            replacements = {}
        self.models.update(replacements)
        for name, model in replacements.items():
            rows_by_name[name].update(self._model_statistics(model))
            rows_by_name[name]["early_stopped"] = bool(getattr(model, "early_stopped", False))
            rows_by_name[name]["training_scope"] = (
                "full" if not self.is_streaming or name == "Popularity" else "sampled"
            )
        self.run_config = RunConfig(
            columns=self.columns,
            eval_metrics=self.eval_metrics,
            modalities=self.modalities,
            preset=presets,
            random_state=self.random_state,
            execution_mode="streaming" if self.is_streaming else "in_memory",
            model_configs={
                row["model"]: row.get("chosen_config", merged_configs[row["model"]])
                for row in fit_rows
            },
            ensemble_weights=self.ensemble_weights,
            max_inference_score_mb=self.max_inference_score_mb,
        )

        dataset_summary = {
            "interactions": interaction_rows,
            "sampled_interactions": len(dataset.interactions),
            "users": (
                None if self.is_streaming else dataset.interactions[self.columns.user_id].nunique()
            ),
            "items": len(full_item_counts) if full_item_counts is not None else self.catalog_size,
            "val_interactions": len(val),
            "test_interactions": len(test),
            "declared_modalities": dataset.modalities,
            "sources": self.data_sources,
            "execution_mode": "streaming" if self.is_streaming else "in_memory",
            "source_size_bytes": parquet_size_bytes(interaction_materialized.parquet),
            "history_index": self.history_index,
            "catalog_truncated": catalog_truncated,
            "split": split,
            "num_workers": num_workers,
            "time_limit": time_limit,
            "model_best": self.model_best,
            "eval_metric": self.eval_metric,
            "refit_full": refit_full,
            "external_test": test_data is not None,
            "max_inference_score_mb": self.max_inference_score_mb,
        }
        protocol = {
            "global_temporal": "global chronological holdout",
            "temporal": "per-user leave-one-out (temporal)",
            "random": "per-user leave-one-out (random)",
            "cold_start": "cold-start user holdout",
        }[split]
        self.evaluation_report = self._build_report(
            test_recommendations if not test.empty else val_recommendations,
            test if not test.empty else val,
            catalog_size=self.catalog_size,
            metadata={
                "protocol": protocol,
                "score_split": "test" if not test.empty else "validation",
                "evaluated_training_scope": "train",
                **dataset_summary,
            },
            weights=self.ensemble_weights,
        )

        fit_frame = pd.DataFrame(fit_rows)
        leaderboard = self.evaluation_report.leaderboard_data
        if not leaderboard.empty:
            val_scores = (
                selection[["model", self.eval_metric]].rename(
                    columns={self.eval_metric: "score_val"}
                )
                if not selection.empty
                else pd.DataFrame(columns=["model", "score_val"])
            )
            leaderboard = leaderboard.merge(val_scores, on="model", how="left")
            leaderboard["is_best"] = leaderboard["model"] == self.model_best
            stats_frame = fit_frame[
                [
                    "model",
                    "status",
                    "train_time_s",
                    "training_scope",
                    "early_stopped",
                    "num_params",
                    "size_bytes",
                    "refit_status",
                ]
            ]
            self.evaluation_report.leaderboard_data = leaderboard.merge(
                stats_frame, on="model", how="left"
            )
        self.fit_result = FitResult(fit_frame, dataset_summary)
        return self

    def fit_dataset(
        self,
        directory: str | Path,
        models: str | list[str] | None = "auto",
        presets: str = "medium_quality",
        **fit_kwargs: Any,
    ) -> MultiModalRecommender:
        """Turnkey fit: discover files, columns and modalities from a directory."""
        return self.fit(
            interactions=directory,
            models=models,
            presets=presets,
            **fit_kwargs,
        )

    def _split_interactions(
        self,
        interactions: pd.DataFrame,
        split: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if split == "global_temporal":
            return global_temporal_split(interactions, self.columns)
        if split == "temporal":
            return leave_one_out_split(interactions, self.columns)
        if split == "random":
            return random_leave_one_out_split(interactions, self.columns, self.random_state)
        if split == "cold_start":
            return cold_start_split(interactions, self.columns, random_state=self.random_state)
        raise ValueError(f"Unknown split strategy: {split}")

    def _fit_model_for_evaluation(
        self,
        model_name: str,
        config: dict[str, Any],
        interactions: pd.DataFrame,
        dataset: DatasetBundle,
        full_item_counts: dict[Any, int] | None,
        use_full_counts: bool,
    ) -> BaseRecommendationModel:
        """Create and fit a model.

        ``use_full_counts`` selects the streaming deployment path (counts from
        the whole stream). It is only enabled for the final deployed model: the
        candidates are fit only on ``train`` so held-out metrics stay independent
        of full-stream interaction counts (full-stream
        counts would leak the val/test interactions into the rankings).
        """
        model = self.registry.create(model_name, self.columns, **config)
        model.max_score_bytes = self.max_inference_score_mb * 1024 * 1024
        if use_full_counts and model_name == "Popularity" and full_item_counts is not None:
            model.fit_from_counts(  # type: ignore[attr-defined]
                full_item_counts, interactions, dataset
            )
            return model
        try:
            model.fit(interactions, dataset)
        except RuntimeError as exc:
            # ``auto`` is allowed to degrade to CPU when a graph model exceeds
            # available GPU memory. An explicit CUDA request remains strict so
            # deployment errors are never hidden.
            is_oom = "out of memory" in str(exc).lower()
            requested_device = str(config.get("device", "auto")).lower()
            if (
                not is_oom
                or not self.registry.requires_torch(model_name)
                or requested_device != "auto"
            ):
                raise
            warnings.warn(
                f"{model_name} ran out of GPU memory; retrying on CPU because device='auto'.",
                stacklevel=2,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass
            fallback_config = {**config, "device": "cpu"}
            model = self.registry.create(model_name, self.columns, **fallback_config)
            model.max_score_bytes = self.max_inference_score_mb * 1024 * 1024
            model.fit(interactions, dataset)
        if use_full_counts and full_item_counts is not None and model_name == "ItemCF":
            maximum = max(full_item_counts.values(), default=1)
            model.items = list(full_item_counts)
            model.popularity = {  # type: ignore[attr-defined]
                item_id: count / maximum for item_id, count in full_item_counts.items()
            }
            model.popularity_ranked = sorted(  # type: ignore[attr-defined]
                full_item_counts,
                key=lambda item_id: (-full_item_counts[item_id], str(item_id)),
            )
        return model

    def _budgeted_config(
        self,
        model_name: str,
        config: dict[str, Any],
        fit_deadline: float | None,
    ) -> dict[str, Any]:
        """Return ``config`` with a fresh ``time_limit`` equal to the remaining budget."""
        if fit_deadline is None or not self.registry.accepts_parameter(model_name, "time_limit"):
            return config
        remaining = max(0.0, fit_deadline - perf_counter())
        return {**config, "time_limit": remaining}

    def _train_one_model(
        self,
        model_name: str,
        train: pd.DataFrame,
        evaluation_dataset: DatasetBundle,
        val: pd.DataFrame,
        val_users: list[Any],
        test_users: list[Any],
        evaluation_k: int,
        full_item_counts: dict[Any, int] | None,
        model_configs: dict[str, dict[str, Any]],
        hyperparameter_tune: bool = False,
        n_trials: int = 4,
        primary_metric: str = "recall@10",
        fit_deadline: float | None = None,
    ) -> tuple[
        pd.DataFrame | None, pd.DataFrame | None, BaseRecommendationModel | None, dict[str, Any]
    ]:
        started = perf_counter()
        try:
            base_config = model_configs.get(model_name, {})
            chosen_config = self._tune_config(
                model_name,
                base_config,
                train,
                evaluation_dataset,
                val,
                val_users,
                evaluation_k,
                primary_metric,
                hyperparameter_tune,
                n_trials,
                fit_deadline,
            )

            candidate = self._fit_model_for_evaluation(
                model_name,
                self._budgeted_config(model_name, chosen_config, fit_deadline),
                train,
                evaluation_dataset,
                full_item_counts,
                use_full_counts=False,
            )
            val_recommendations = (
                candidate.recommend(val_users, evaluation_k, exclude_seen=True)
                if not val.empty
                else None
            )
            test_recommendations = (
                candidate.recommend(test_users, evaluation_k, exclude_seen=True)
                if test_users
                else None
            )
            final_model = candidate
            row = {
                "model": model_name,
                "status": "succeeded",
                "train_time_s": perf_counter() - started,
                "training_scope": ("train" if not self.is_streaming else "sampled"),
                "error": None,
                "chosen_config": chosen_config,
                "actual_device": getattr(final_model, "device_name", None),
                "refit_status": "not_requested",
                "early_stopped": bool(getattr(final_model, "early_stopped", False)),
                **self._model_statistics(final_model),
            }
            return val_recommendations, test_recommendations, final_model, row
        except Exception as exc:  # Models are isolated so one failure does not discard a run.
            if isinstance(exc, TimeoutError) or (
                fit_deadline is not None and perf_counter() >= fit_deadline
            ):
                return None, None, None, self._skip_row(model_name)
            row = {
                "model": model_name,
                "status": "failed",
                "train_time_s": perf_counter() - started,
                "training_scope": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "early_stopped": False,
                "actual_device": None,
                "num_params": 0,
                "size_bytes": 0,
            }
            return None, None, None, row

    def _tune_config(
        self,
        model_name: str,
        base_config: dict[str, Any],
        train: pd.DataFrame,
        evaluation_dataset: DatasetBundle,
        val: pd.DataFrame,
        val_users: list[Any],
        evaluation_k: int,
        primary_metric: str,
        hyperparameter_tune: bool,
        n_trials: int,
        fit_deadline: float | None = None,
    ) -> dict[str, Any]:
        """Select a model configuration via TPE-style guided successive halving.

        A sequential TPE sampler proposes each candidate from the good-vs-bad
        history accumulated so far, evaluates it cheaply, observes the score, and
        repeats; the better half then advances to full-budget training.
        """
        if not hyperparameter_tune or val.empty:
            return base_config
        space = SEARCH_SPACES.get(model_name)
        if space is None and self.registry.requires_torch(model_name):
            space = TORCH_SEARCH_SPACE
        if not space:
            return base_config

        def over_budget() -> bool:
            return fit_deadline is not None and perf_counter() >= fit_deadline

        catalog = len(evaluation_dataset.items) if evaluation_dataset.items is not None else 1

        def evaluate(config: dict[str, Any], coarse_epochs: int | None) -> float | None:
            trial = dict(config)
            if coarse_epochs is not None:
                trial["epochs"] = coarse_epochs
            return self._evaluate_config(
                model_name,
                trial,
                train,
                evaluation_dataset,
                val,
                val_users,
                evaluation_k,
                primary_metric,
                catalog,
                fit_deadline,
            )

        sampler = TpeSearch(space, random_state=self.random_state)
        coarse_results: list[tuple[dict[str, Any], float]] = []
        # Evaluate the base config separately (it may lack search-space keys, so
        # it is not fed to the TPE), then loop suggest -> evaluate -> observe so
        # the TPE history actually guides each subsequent candidate.
        if not over_budget():
            base_score = evaluate(base_config, 2)
            if base_score is not None:
                coarse_results.append((base_config, base_score))
        for _ in range(n_trials):
            if over_budget():
                break
            config = {**base_config, **sampler.suggest()}
            score = evaluate(config, 2)
            if score is not None:
                sampler.observe(config, score)
                coarse_results.append((config, score))
        if not coarse_results:
            return base_config
        coarse_results.sort(key=lambda item: item[1], reverse=True)
        survivors = [
            config for config, _ in coarse_results[: max(1, (len(coarse_results) + 1) // 2)]
        ]

        best_config = base_config
        best_score = float("-inf")
        for config in survivors:
            if over_budget():
                break
            score = evaluate(config, None)
            if score is not None and score > best_score:
                best_score = score
                best_config = config
        return best_config

    def _evaluate_config(
        self,
        model_name: str,
        config: dict[str, Any],
        train: pd.DataFrame,
        evaluation_dataset: DatasetBundle,
        holdout: pd.DataFrame,
        test_users: list[Any],
        evaluation_k: int,
        primary_metric: str,
        catalog: int,
        fit_deadline: float | None = None,
    ) -> float | None:
        try:
            trial_config = self._budgeted_config(model_name, config, fit_deadline)
            model = self.registry.create(model_name, self.columns, **trial_config).fit(
                train, evaluation_dataset
            )
            recommendations = model.recommend(test_users, evaluation_k, exclude_seen=True)
        except Exception:
            return None
        _, details = evaluate_recommendations(
            recommendations, holdout, self.columns, [primary_metric], catalog
        )
        if not details:
            return None
        values = [value for key, value in next(iter(details.values())).items() if key != "coverage"]
        return float(values[0]) if values else None

    def _skip_row(self, model_name: str) -> dict[str, Any]:
        return {
            "model": model_name,
            "status": "skipped",
            "train_time_s": 0.0,
            "training_scope": "skipped",
            "error": "time budget exhausted",
            "early_stopped": False,
            "actual_device": None,
            "num_params": 0,
            "size_bytes": 0,
        }

    def _interaction_identity(self, interactions: TableInput, materialized: Any) -> str:
        if isinstance(interactions, pd.DataFrame):
            return "frame:" + self._frame_hash(interactions)
        return "file:" + file_fingerprint(materialized.parquet)

    def _frame_hash(self, frame: pd.DataFrame) -> str:
        return frame_fingerprint(frame)

    def _table_identity(self, table: Any) -> str | None:
        if table is None:
            return None
        if isinstance(table, pd.DataFrame):
            return "frame:" + self._frame_hash(table)
        return "file:" + file_fingerprint(table)

    def _save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        results: dict[str, Any],
        validation: dict[str, Any],
    ) -> None:
        target = Path(checkpoint_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = {"validation": validation, "results": results}
        temporary = target / f".checkpoint.{uuid.uuid4().hex}.tmp"
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(target / "checkpoint.pkl")

    def _load_checkpoint(
        self,
        checkpoint_dir: str | Path,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        path = Path(checkpoint_dir) / "checkpoint.pkl"
        if not path.exists():
            return None, None
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - trusted local checkpoint
        return payload.get("validation"), payload.get("results")

    def _model_statistics(self, model: BaseRecommendationModel) -> dict[str, int]:
        num_params = self._count_parameters(model.__dict__)
        size_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        return {"num_params": num_params, "size_bytes": size_bytes}

    def _count_parameters(self, value: Any) -> int:
        if isinstance(value, np.ndarray):
            return int(value.size)
        if hasattr(value, "numel") and callable(value.numel):  # torch tensors
            return int(value.numel())
        if hasattr(value, "parameters") and callable(value.parameters):  # torch nn.Module
            return sum(
                int(parameter.numel())
                for parameter in value.parameters()
                if hasattr(parameter, "numel") and callable(parameter.numel)
            )
        if isinstance(value, dict):
            return sum(self._count_parameters(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._count_parameters(v) for v in value)
        return 0

    def _package_version(self) -> str:
        try:
            from importlib.metadata import version

            return version("multimodal-recommender")
        except Exception:  # pragma: no cover - only when running from an uninstalled source tree
            return "unknown"

    def _dependency_versions(self) -> dict[str, str | None]:
        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:  # pragma: no cover - Python 3.8 fallback
            return {}
        versions: dict[str, str | None] = {}
        for name in ("numpy", "pandas", "pyarrow", "torch"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                versions[name] = None
            except Exception:  # pragma: no cover - malformed distributions
                versions[name] = None
        return versions

    def _interaction_schema_hash(self) -> str | None:
        source = self.data_sources.get("interactions")
        if not source or source == "memory" or not Path(source).exists():
            return None
        try:
            schema = parquet.read_schema(source)
            canonical = "|".join(f"{field.name}:{field.type}" for field in schema)
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        except Exception:  # pragma: no cover - unreadable schema should not break save()
            return None

    def _model_manifest_statistics(self) -> dict[str, dict[str, Any]]:
        statistics: dict[str, dict[str, Any]] = {}
        frame = None if self.fit_result is None else self.fit_result.models
        for name in self.models:
            statistics[name] = {}
            if frame is None or frame.empty or "model" not in frame:
                continue
            matches = frame[frame["model"] == name]
            if matches.empty:
                continue
            row = matches.iloc[0]
            statistics[name] = {
                "status": row.get("status"),
                "train_time_s": row.get("train_time_s"),
                "num_params": row.get("num_params"),
                "size_bytes": row.get("size_bytes"),
            }
        return statistics

    def recommend(
        self,
        users: list[Any] | None = None,
        k: int = 10,
        models: str | list[str] = "best",
        include_ensemble: bool = True,
        exclude_seen: bool = True,
    ) -> RecommendationResult:
        self._require_fitted()
        if k <= 0:
            raise ValueError("k must be positive.")
        requested = self.model_best if models == "best" else models
        fusion_only = requested == "RankFusion"
        selected = self._select_fitted_models("all" if fusion_only else requested)
        if users is None:
            if self.is_streaming:
                raise ValueError(
                    "users must be provided in streaming mode; use recommend_to_parquet() "
                    "for large batches."
                )
            users = self.training_users.copy()

        seen_override = self._load_exact_histories(users) if exclude_seen else None

        frames = [
            self.models[name].recommend(
                users,
                k,
                exclude_seen=exclude_seen,
                seen_override=seen_override,
            )
            for name in selected
        ]
        combined = self._combine_frames(frames)
        if (include_ensemble or fusion_only) and len(frames) >= 2:
            combined = pd.concat(
                [
                    combined,
                    reciprocal_rank_fusion(
                        combined,
                        self.columns.user_id,
                        self.columns.item_id,
                        k,
                        weights=self.ensemble_weights,
                    ),
                ],
                ignore_index=True,
            )
        if fusion_only:
            combined = combined[combined["model"] == "RankFusion"]
        combined = combined.sort_values(
            ["model", self.columns.user_id, "rank"], kind="stable"
        ).reset_index(drop=True)
        return RecommendationResult(combined)

    def recommend_to_parquet(
        self,
        users: TableInput | list[Any],
        path: str | Path,
        k: int = 10,
        models: str | list[str] = "best",
        include_ensemble: bool = True,
        exclude_seen: bool = True,
        batch_size: int | None = None,
    ) -> Path:
        """Generate recommendations in bounded batches and stream them to Parquet."""
        self._require_fitted()
        if k <= 0:
            raise ValueError("k must be positive.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        writer: parquet.ParquetWriter | None = None
        size = self.execution.inference_batch_size if batch_size is None else batch_size
        if size <= 0:
            raise ValueError("batch_size must be positive.")
        batches = self._iter_inference_users(users, size)
        wrote_rows = False
        try:
            for user_batch in batches:
                result = self.recommend(
                    users=user_batch,
                    k=k,
                    models=models,
                    include_ensemble=include_ensemble,
                    exclude_seen=exclude_seen,
                ).data
                if result.empty:
                    continue
                table = pa.Table.from_pandas(result, preserve_index=False)
                if writer is None:
                    writer = parquet.ParquetWriter(
                        temporary,
                        table.schema,
                        compression="zstd",
                    )
                writer.write_table(table)
                wrote_rows = True
            if not wrote_rows:
                raise ValueError("No recommendations were generated for the supplied users.")
            if writer is not None:
                writer.close()
                writer = None
            temporary.replace(output)
        finally:
            if writer is not None:
                writer.close()
            temporary.unlink(missing_ok=True)
        return output

    def predict(self, pairs: TableInput, models: str | list[str] = "best") -> pd.DataFrame:
        """Score pairs; RankFusion ranks each user's supplied candidate set.

        Fusion scores are reciprocal-rank scores, not calibrated probabilities.
        Their values depend on the supplied candidate set.
        """
        self._require_fitted()
        frame = pairs.copy() if isinstance(pairs, pd.DataFrame) else self._load_pairs(pairs)
        required = {self.columns.user_id, self.columns.item_id}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"pairs is missing required columns: {missing}")

        requested = self.model_best if models == "best" else models
        fusion_only = requested == "RankFusion"
        rows: list[dict[str, Any]] = []
        selected = self._select_fitted_models("all" if fusion_only else requested)
        for model_name in selected:
            model = self.models[model_name]
            for user_id, group in frame.groupby(self.columns.user_id, sort=False):
                item_ids = group[self.columns.item_id].tolist()
                scores = model.score_items(user_id, item_ids)
                rows.extend(
                    {
                        self.columns.user_id: user_id,
                        self.columns.item_id: item_id,
                        "score": float(scores.get(item_id, 0.0)),
                        "model": model_name,
                    }
                    for item_id in item_ids
                )
        scored = pd.DataFrame(
            rows, columns=[self.columns.user_id, self.columns.item_id, "score", "model"]
        )
        if fusion_only and not scored.empty:
            unique = scored.drop_duplicates([self.columns.user_id, self.columns.item_id, "model"])
            unique = unique.copy()
            unique["rank"] = unique.groupby([self.columns.user_id, "model"])["score"].rank(
                method="first", ascending=False
            )
            fused = reciprocal_rank_fusion(
                unique,
                self.columns.user_id,
                self.columns.item_id,
                len(unique),
                weights=self.ensemble_weights,
            )
            keys = [self.columns.user_id, self.columns.item_id]
            return frame[keys].merge(fused.drop(columns="rank"), on=keys, how="left")
        return scored

    def evaluate(
        self,
        test_data: TableInput,
        models: str | list[str] = "all",
    ) -> EvaluationReport:
        self._require_fitted()
        test_bundle = prepare_dataset(
            test_data,
            None,
            None,
            self.columns,
            None,
            cache_dir=self.cache_dir,
        )
        users = test_bundle.interactions[self.columns.user_id].drop_duplicates().tolist()
        recommendations = self.recommend(
            users=users,
            k=max_metric_cutoff(self.eval_metrics),
            models=models,
            include_ensemble=True,
            exclude_seen=True,
        )
        leaderboard, details = evaluate_recommendations(
            recommendations.data,
            test_bundle.interactions,
            self.columns,
            self.eval_metrics,
            catalog_size=self.catalog_size,
        )
        return EvaluationReport(
            leaderboard,
            details,
            metadata={
                "protocol": "external test set",
                "test_interactions": len(test_bundle.interactions),
            },
        )

    def leaderboard(self) -> pd.DataFrame:
        self._require_fitted()
        assert self.evaluation_report is not None
        return self.evaluation_report.leaderboard()

    def fit_summary(self) -> FitResult:
        self._require_fitted()
        assert self.fit_result is not None
        return self.fit_result

    def save(
        self,
        path: str | Path,
        include_history_index: bool = False,
        best_only: bool = False,
        max_generations: int = 3,
    ) -> Path:
        """Persist an artifact while serializing concurrent writers per directory."""
        if max_generations <= 0:
            raise ValueError("max_generations must be positive.")
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        lock = target / ".save.lock"
        deadline = perf_counter() + 30.0
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                    handle.write(str(os.getpid()))
                break
            except FileExistsError:
                try:
                    owner = int(lock.read_text(encoding="ascii"))
                    os.kill(owner, 0)
                except (OSError, ValueError):
                    lock.unlink(missing_ok=True)
                    continue
                if perf_counter() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting to save model into {target}"
                    ) from None
                sleep(0.05)
        try:
            result = self._save_unlocked(path, include_history_index, best_only)
            self._publish_generation(result, max_generations)
            return result
        finally:
            lock.unlink(missing_ok=True)

    @staticmethod
    def _publish_generation(target: Path, max_generations: int) -> None:
        """Publish a complete artifact directory by atomically swapping a pointer."""
        generation_name = f".generation-{uuid.uuid4().hex}"
        generation = target / generation_name
        generation.mkdir()
        shutil.copy2(target / "recommender.pkl", generation / "recommender.pkl")
        shutil.copy2(target / "metadata.json", generation / "metadata.json")
        for history_index in target.glob("history-index-*"):
            if history_index.is_dir():
                shutil.copytree(history_index, generation / history_index.name)
        pointer = target / ".CURRENT"
        temporary = target / f".CURRENT.{uuid.uuid4().hex}.tmp"
        temporary.write_text(generation_name, encoding="utf-8")
        temporary.replace(pointer)
        generations = sorted(
            (candidate for candidate in target.glob(".generation-*") if candidate.is_dir()),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        )
        for obsolete in generations[max_generations:]:
            shutil.rmtree(obsolete)

    def _save_unlocked(
        self,
        path: str | Path,
        include_history_index: bool = False,
        best_only: bool = False,
    ) -> Path:
        self._require_fitted()
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        original_history_index = self.history_index
        bundled_history_index: str | None = None
        artifact_path = target / "recommender.pkl"
        metadata_path = target / "metadata.json"
        previous_artifact = target / ".recommender.pkl.previous"
        previous_metadata = target / ".metadata.json.previous"
        temporary_artifact = target / f".recommender.pkl.{uuid.uuid4().hex}.tmp"
        try:
            if artifact_path.exists() and metadata_path.exists():
                shutil.copy2(artifact_path, previous_artifact)
                shutil.copy2(metadata_path, previous_metadata)
            if include_history_index and self.history_index:
                source_index = Path(self.history_index)
                if not source_index.exists():
                    raise FileNotFoundError(f"History index does not exist: {source_index}")
                bundled_history_index = f"history-index-{uuid.uuid4().hex[:12]}"
                shutil.copytree(source_index, target / bundled_history_index)
                self.history_index = bundled_history_index
            with temporary_artifact.open("wb") as handle:
                artifact = copy.copy(self)
                if best_only and self.model_best != "RankFusion":
                    artifact.models = {self.model_best: self.models[self.model_best]}
                pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_artifact.replace(artifact_path)
        finally:
            self.history_index = original_history_index
            temporary_artifact.unlink(missing_ok=True)
        metadata = {
            "format_version": 2,
            "artifact_sha256": _sha256_file(artifact_path),
            "model_best": self.model_best,
            "eval_metric": self.eval_metric,
            "package_version": self._package_version(),
            "models": list(artifact.models),
            "config": self.run_config.to_dict() if self.run_config else {},
            "model_statistics": artifact._model_manifest_statistics(),
            "dependencies": self._dependency_versions(),
            "schema_hash": self._interaction_schema_hash(),
            "history_index_bundled": bundled_history_index is not None,
            "external_history_index": (
                original_history_index if self.is_streaming and not bundled_history_index else None
            ),
        }
        temporary_metadata = target / f".metadata.json.{uuid.uuid4().hex}.tmp"
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)
        return target

    @classmethod
    def load(cls, path: str | Path) -> MultiModalRecommender:
        root = Path(path)
        lock = root / ".save.lock"
        deadline = perf_counter() + 30.0
        while lock.exists():
            try:
                owner = int(lock.read_text(encoding="ascii"))
                os.kill(owner, 0)
            except (OSError, ValueError):
                lock.unlink(missing_ok=True)
                continue
            if perf_counter() >= deadline:
                raise TimeoutError(f"Timed out waiting to load model from {root}")
            sleep(0.05)
        candidates: list[tuple[Path, Path]] = []
        pointer = root / ".CURRENT"
        if pointer.exists():
            generation_name = pointer.read_text(encoding="utf-8").strip()
            generation = (root / generation_name).resolve()
            if generation.parent == root.resolve() and generation.name.startswith(".generation-"):
                candidates.append((generation / "recommender.pkl", generation / "metadata.json"))
        candidates.append((root / "recommender.pkl", root / "metadata.json"))
        previous = (root / ".recommender.pkl.previous", root / ".metadata.json.previous")
        if all(candidate.exists() for candidate in previous):
            candidates.append(previous)
        failures: list[Exception] = []
        for source, metadata_path in candidates:
            if not source.exists():
                continue
            try:
                metadata: dict[str, Any] | None = None
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata must be a JSON object")
                    if metadata.get("format_version", 1) > 2:
                        raise ValueError(
                            f"Unsupported model metadata format: {metadata['format_version']}"
                        )
                    expected_hash = metadata.get("artifact_sha256")
                    if expected_hash and expected_hash != _sha256_file(source):
                        raise ValueError("Saved recommender artifact failed its integrity check.")
                with source.open("rb") as handle:
                    value = pickle.load(handle)  # noqa: S301 - trusted local artifact.
                if not isinstance(value, cls):
                    raise TypeError(f"The saved object is not a {cls.__name__}.")
                if metadata is not None:
                    declared_models = metadata.get("models")
                    if declared_models is not None and set(declared_models) != set(value.models):
                        raise ValueError(
                            "Saved model metadata does not match the artifact model list."
                        )
                    declared_best = metadata.get("model_best")
                    if declared_best is not None and declared_best != value.model_best:
                        raise ValueError(
                            "Saved model metadata does not match the selected best model."
                        )
                if not hasattr(value, "model_best"):
                    value.model_best = next(iter(value.models), None)
                    value.eval_metric = value.eval_metrics[0]
                    value.validation_report = None
                    value._deployment_histories = None
                    value._validation_histories = {}
                if value.history_index and not Path(value.history_index).is_absolute():
                    value.history_index = str((source.parent / value.history_index).resolve())
                return value
            except (OSError, ValueError, TypeError, pickle.PickleError, EOFError) as exc:
                failures.append(exc)
        if not candidates or not failures:
            raise FileNotFoundError(f"Saved recommender not found: {root / 'recommender.pkl'}")
        raise ValueError(str(failures[-1])) from failures[-1]

    def _build_report(
        self,
        recommendation_frames: list[pd.DataFrame],
        truth: pd.DataFrame,
        catalog_size: int,
        metadata: dict[str, Any],
        weights: dict[str, float] | None = None,
    ) -> EvaluationReport:
        if truth.empty or not recommendation_frames:
            return EvaluationReport(pd.DataFrame(), metadata=metadata)
        combined = self._combine_frames(recommendation_frames)
        if len(recommendation_frames) >= 2:
            fused = reciprocal_rank_fusion(
                combined,
                self.columns.user_id,
                self.columns.item_id,
                max_metric_cutoff(self.eval_metrics),
                weights=weights,
            )
            combined = pd.concat([combined, fused], ignore_index=True)
        leaderboard, details = evaluate_recommendations(
            combined,
            truth,
            self.columns,
            self.eval_metrics,
            catalog_size,
        )
        return EvaluationReport(leaderboard, details, metadata)

    def set_model_best(self, model: str) -> MultiModalRecommender:
        self._require_fitted()
        if model == "RankFusion":
            if len(self.models) < 2:
                raise ValueError("RankFusion requires at least two fitted models.")
        elif model not in self.models:
            raise ValueError(f"Model is not fitted: {model}")
        self.model_best = model
        if self.fit_result is not None:
            self.fit_result.dataset_summary["model_best"] = model
        if self.evaluation_report is not None:
            self.evaluation_report.metadata["model_best"] = model
        if self.evaluation_report is not None:
            board = self.evaluation_report.leaderboard_data
            if not board.empty:
                board["is_best"] = board["model"] == model
        return self

    def _select_fitted_models(self, models: str | list[str]) -> list[str]:
        if models == "all":
            return list(self.models)
        selected = [models] if isinstance(models, str) else list(models)
        missing = sorted(set(selected) - set(self.models))
        if missing:
            raise ValueError(f"Models are not fitted: {missing}")
        return selected

    def _add_automatic_multimodal_models(
        self,
        model_names: list[str],
        dataset: DatasetBundle,
        preset: str,
    ) -> list[str]:
        if preset != "fast_training" and torch_available():
            model_names = list(dict.fromkeys([*model_names, "LightGCN"]))
        if dataset.items is None:
            return model_names
        modalities = dataset.modalities.get("item", {})
        if not modalities:
            return model_names
        blocks = 0
        blocks += int(bool(modalities.get("categorical")))
        blocks += int(bool(modalities.get("numerical")))
        blocks += int(bool(modalities.get("text")))
        for name in ("embedding", "image"):
            value = modalities.get(name, [])
            blocks += 1 if isinstance(value, str) else len(value)

        additions = ["MultiModalItemKNN"]
        if preset != "fast_training":
            if blocks >= 2:
                additions.append("MultiModalLateFusionKNN")
            additions.append("VBPR")
            # Graph models are optional so the core wheel stays lightweight.
            if torch_available():
                additions.extend(
                    [
                        "MMGCN",
                        "LATTICE",
                        "BM3",
                        "FREEDOM",
                        "MGCN",
                        "DRAGON",
                        "LGMRec",
                    ]
                )
        return list(dict.fromkeys([*model_names, *additions]))

    def _combine_frames(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        columns = [self.columns.user_id, self.columns.item_id, "rank", "score", "model"]
        nonempty = [frame for frame in frames if not frame.empty]
        return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=columns)

    def _load_pairs(self, pairs: str | Path) -> pd.DataFrame:
        materialized = ParquetCache(self.cache_dir).materialize(pairs, "pairs")
        return pd.read_parquet(materialized.parquet)

    def _load_exact_histories(self, users: list[Any]) -> dict[Any, set[Any]] | None:
        if self._deployment_histories is not None:
            return {user: self._deployment_histories.get(user, set()) for user in users}
        if self.history_index:
            index_path = Path(self.history_index)
            if not index_path.exists():
                raise FileNotFoundError(
                    f"Streaming history index is unavailable: {index_path}. Refit the model "
                    "or restore its cache directory."
                )
            histories = seen_items_from_index(
                index_path,
                self.columns,
                users,
                self.execution.history_partitions,
                self.execution.scan_batch_size,
            )
            for user in users:
                histories.setdefault(user, set()).update(
                    self._validation_histories.get(user, set())
                )
            return histories
        source = self.data_sources.get("interactions")
        if not source or source == "memory" or not Path(source).exists():
            if self.is_streaming:
                raise FileNotFoundError(
                    "The interaction Parquet used by this streaming model is unavailable. "
                    "Restore the cache or save with include_history_index=True."
                )
            return None
        return seen_items_for_users(
            source,
            self.columns,
            users,
            self.execution.scan_batch_size,
        )

    def _iter_inference_users(
        self,
        users: TableInput | list[Any],
        batch_size: int,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if isinstance(users, list):
            if not users:
                raise ValueError("users must contain at least one user identifier.")
            for start in range(0, len(users), batch_size):
                yield users[start : start + batch_size]
            return

        cache = ParquetCache(self.cache_dir)
        materialized = (
            cache.materialize_frame(users, "recommend-users")
            if isinstance(users, pd.DataFrame)
            else cache.materialize(users, "recommend-users")
        )
        yield from iter_column_batches(
            materialized.parquet,
            self.columns.user_id,
            batch_size,
        )

    def _require_fitted(self) -> None:
        if not self.models:
            raise RuntimeError("Call fit() before requesting predictions or reports.")
