from __future__ import annotations

import json

import pandas as pd
import pytest

from mmrec import MultiModalRecommender
from mmrec.config import ColumnConfig
from mmrec.data import global_temporal_split
from mmrec.models.base import BaseRecommendationModel


class OrderedModel(BaseRecommendationModel):
    calls: list[tuple[str, tuple[str, ...]]] = []

    def __init__(self, columns, variant="A", fail_refit=False):
        super().__init__(columns)
        self.name = variant
        self.fail_refit = fail_refit

    def fit(self, interactions, dataset=None):
        items = tuple(interactions[self.columns.item_id])
        self.calls.append((self.name, items))
        if self.fail_refit and "b" in items:
            raise RuntimeError("refit failed")
        self._capture_catalog(interactions, dataset)
        return self

    def score_items(self, user_id, item_ids):
        preferred = "b" if self.name == "A" else "c"
        return {item: float(item == preferred) for item in item_ids}


def tables():
    def frame(item, day):
        return pd.DataFrame(
            {
                "user_id": ["u"],
                "item_id": [item],
                "timestamp": [pd.Timestamp(f"2026-01-{day:02d}")],
            }
        )

    return frame("a", 1), frame("b", 2), frame("c", 3)


def predictor(tmp_path):
    result = MultiModalRecommender(eval_metric="recall@1", cache_dir=tmp_path / "cache")
    result.register_model("A", OrderedModel).register_model("B", OrderedModel)
    return result


def fit_kwargs():
    train, val, test = tables()
    return dict(
        interactions=train,
        validation_data=val,
        test_data=test,
        items=pd.DataFrame({"item_id": ["a", "b", "c", "d"]}),
        models=["A", "B"],
        model_configs={"B": {"variant": "B"}},
    )


def test_best_uses_validation_not_test_and_refits_after_comparison(tmp_path):
    OrderedModel.calls = []
    result = predictor(tmp_path).fit(**fit_kwargs())
    assert result.model_best == "A"
    assert result.leaderboard().iloc[0]["model"] == "B"
    assert OrderedModel.calls == [("A", ("a",)), ("B", ("a",)), ("A", ("a", "b"))]
    assert result.recommend(["new"], k=1).models == ["A"]
    assert set(result.recommend(["new"], k=1, models="all").models) == {"A", "B", "RankFusion"}
    assert result.leaderboard().set_index("model").loc["A", "score_val"] == 1
    assert result.predict(pd.DataFrame({"user_id": ["new"], "item_id": ["b"]}))[
        "model"
    ].tolist() == ["A"]


def test_no_refit_and_all_refit(tmp_path):
    for policy, expected_count in ((False, 2), ("all", 4)):
        OrderedModel.calls = []
        result = predictor(tmp_path).fit(**fit_kwargs(), refit_full=policy)
        assert len(OrderedModel.calls) == expected_count
        assert all("c" not in items for _, items in OrderedModel.calls)
        assert result.model_best == "A"


def test_failed_refit_retains_candidate(tmp_path):
    kwargs = fit_kwargs()
    kwargs["model_configs"]["A"] = {"fail_refit": True}
    result = predictor(tmp_path).fit(**kwargs)
    assert result.fit_summary().models.set_index("model").loc["A", "refit_status"] == "failed"
    assert result.models["A"].seen_by_user["u"] == {"a"}
    assert result.recommend(["new"], k=1).models == ["A"]


def test_budget_expiry_during_refit_keeps_candidate(tmp_path, monkeypatch):
    import mmrec.predictor as module

    clock = [0.0]
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    result = predictor(tmp_path)
    original = result._fit_model_for_evaluation

    def fit(*args, **kwargs):
        if kwargs.get("use_full_counts"):
            raise TimeoutError("budget exhausted")
        candidate = original(*args, **kwargs)
        if args[0] == "B":
            clock[0] = 10.0
        return candidate

    monkeypatch.setattr(result, "_fit_model_for_evaluation", fit)
    result.fit(**fit_kwargs(), time_limit=5)
    assert result.model_best == "A"
    assert (
        result.fit_summary().models.set_index("model").loc["A", "refit_status"] == "skipped_budget"
    )
    assert result.recommend(["new"], k=1).models == ["A"]


def test_external_test_changes_do_not_change_selected_model(tmp_path):
    kwargs = fit_kwargs()
    first = predictor(tmp_path).fit(**kwargs)
    kwargs["test_data"] = kwargs["test_data"].assign(item_id="d")
    second = predictor(tmp_path).fit(**kwargs)
    assert first.model_best == second.model_best
    assert first.ensemble_weights == second.ensemble_weights
    pd.testing.assert_frame_equal(first.recommend(["new"]).data, second.recommend(["new"]).data)


def test_explicit_validation_without_test_reports_validation(tmp_path):
    kwargs = fit_kwargs()
    kwargs.pop("test_data")
    result = predictor(tmp_path).fit(**kwargs)
    assert result.evaluation_report.metadata["score_split"] == "validation"
    assert result.model_best == "A"


def test_overlapping_external_events_rejected(tmp_path):
    kwargs = fit_kwargs()
    kwargs["test_data"] = kwargs["validation_data"]
    with pytest.raises(ValueError, match="overlapping events"):
        predictor(tmp_path).fit(**kwargs)


def test_global_time_split_keeps_ties_and_strict_order():
    data = pd.DataFrame(
        {
            "user_id": ["u", "v"] * 5,
            "item_id": list("abcdefghij"),
            "timestamp": pd.to_datetime(["2026-01-01"] * 8 + ["2026-01-02"] * 2),
        }
    )
    train, test = global_temporal_split(data, ColumnConfig())
    assert len(train) == 8
    assert train.timestamp.max() < test.timestamp.min()
    with pytest.raises(ValueError, match="timestamp"):
        global_temporal_split(data, ColumnConfig(timestamp=None))


def test_global_time_fit_and_external_time_validation(tmp_path):
    data = pd.DataFrame(
        {
            "user_id": ["u", "v"] * 10,
            "item_id": list("abcdefghij") * 2,
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="h"),
        }
    )
    result = MultiModalRecommender(cache_dir=tmp_path).fit(
        data,
        models="Popularity",
        split="global_temporal",
    )
    assert result.fit_summary().dataset_summary["test_interactions"] == 4
    kwargs = fit_kwargs()
    kwargs["validation_data"] = kwargs["validation_data"].assign(
        timestamp=pd.Timestamp("2025-01-01")
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        predictor(tmp_path).fit(**kwargs, split="global_temporal")


def test_best_export_and_fusion_selection_survive_save(tmp_path):
    result = predictor(tmp_path).fit(**fit_kwargs())
    path = result.save(tmp_path / "best", best_only=True)
    loaded = MultiModalRecommender.load(path)
    assert list(loaded.models) == ["A"]
    assert len(result.models) == 2
    assert json.loads((path / "metadata.json").read_text())["model_best"] == "A"
    pd.testing.assert_frame_equal(result.recommend(["u"]).data, loaded.recommend(["u"]).data)
    result.set_model_best("RankFusion")
    loaded = MultiModalRecommender.load(result.save(tmp_path / "fusion", best_only=True))
    assert set(loaded.models) == {"A", "B"}
    assert loaded.recommend(["new"], k=2).models == ["RankFusion"]
    pairs = pd.DataFrame({"user_id": ["new", "new", "new"], "item_id": ["b", "c", "b"]})
    scored = loaded.predict(pairs)
    assert len(scored) == 3
    assert set(scored.model) == {"RankFusion"}
    assert scored.score.iloc[0] == scored.score.iloc[2]


def test_checkpoint_reuses_candidates_but_invalidates_changed_validation(tmp_path):
    kwargs = fit_kwargs()
    kwargs["checkpoint_dir"] = tmp_path / "checkpoint"
    predictor(tmp_path).fit(**kwargs, refit_full=False)
    OrderedModel.calls = []
    predictor(tmp_path).fit(**kwargs, refit_full=False)
    assert OrderedModel.calls == []
    kwargs["validation_data"] = kwargs["validation_data"].assign(item_id="d")
    predictor(tmp_path).fit(**kwargs, refit_full=False)
    assert len(OrderedModel.calls) == 2


class FusionModel(OrderedModel):
    def score_items(self, user_id, item_ids):
        order = {
            ("A", "u1"): "bcd",
            ("B", "u1"): "dbc",
            ("A", "u2"): "dcb",
            ("B", "u2"): "cbd",
        }.get((self.name, user_id), "bcd")
        return {item: float(3 - order.index(item)) if item in order else 0.0 for item in item_ids}


def test_fusion_winner_refits_all_dependencies_atomically(tmp_path):
    train = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["a", "a"]})
    val = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["b", "c"]})
    result = MultiModalRecommender(
        timestamp=None,
        eval_metrics=["recall@1", "recall@2"],
        cache_dir=tmp_path,
    )
    result.register_model("A", FusionModel).register_model("B", FusionModel)
    result.fit(
        train,
        validation_data=val,
        items=pd.DataFrame({"item_id": list("abcd")}),
        models=["A", "B"],
        model_configs={"B": {"variant": "B", "fail_refit": True}},
    )
    assert result.model_best == "RankFusion"
    assert all(model.seen_by_user["u1"] == {"a"} for model in result.models.values())
    statuses = result.fit_summary().models.set_index("model").refit_status.to_dict()
    assert statuses == {"A": "discarded_incomplete_ensemble", "B": "failed"}
    assert result.recommend(["u1"], k=2).models == ["RankFusion"]


def test_refit_checkpoint_reuses_completed_refit(tmp_path):
    kwargs = fit_kwargs()
    kwargs["checkpoint_dir"] = tmp_path / "checkpoint"
    first = predictor(tmp_path).fit(**kwargs)
    OrderedModel.calls = []
    second = predictor(tmp_path).fit(**kwargs)
    assert OrderedModel.calls == []
    pd.testing.assert_frame_equal(first.recommend(["new"]).data, second.recommend(["new"]).data)


def test_streaming_external_validation_is_refit_but_test_is_not(tmp_path):
    train, val, test = tables()
    result = MultiModalRecommender(
        execution_mode="streaming",
        cache_dir=tmp_path / "cache",
        eval_metric="recall@1",
    ).fit(
        train,
        validation_data=val,
        test_data=test,
        models="Popularity",
        items=pd.DataFrame({"item_id": list("abcd")}),
    )
    assert set(result.models["Popularity"].scores) == {"a", "b"}
    assert result._load_exact_histories(["u"]) == {"u": {"a", "b"}}
    path = result.save(tmp_path / "model", include_history_index=True)
    loaded = MultiModalRecommender.load(path)
    assert loaded._load_exact_histories(["u"]) == {"u": {"a", "b"}}
