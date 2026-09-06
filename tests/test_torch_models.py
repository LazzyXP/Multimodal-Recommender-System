from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mmrec import MultiModalRecommender
from mmrec.config import ColumnConfig
from mmrec.models import LightGCNModel, torch_available
from mmrec.models.graph import common


def test_torch_models_are_catalogued_without_importing_torch() -> None:
    predictor = MultiModalRecommender()
    catalog = predictor.model_catalog().set_index("name")
    expected = {"LightGCN", "MMGCN", "LATTICE", "BM3", "FREEDOM", "MGCN", "DRAGON", "LGMRec"}
    assert expected <= set(catalog.index)
    assert set(catalog.loc[sorted(expected), "training"]) == {"torch"}


def test_torch_requirement_is_explicit_when_runtime_is_missing(tmp_path) -> None:
    if torch_available():
        return
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["a", "b", "a", "c"],
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
        }
    )
    try:
        MultiModalRecommender(cache_dir=tmp_path / "cache").fit(interactions, models="LightGCN")
    except RuntimeError as exc:
        assert "No module named 'torch'" in str(exc)
    else:
        raise AssertionError("LightGCN should require the optional torch dependency")


def test_graph_models_live_in_distinct_modules() -> None:
    from mmrec.models import (
        BM3Model,
        DRAGONModel,
        FREEDOMModel,
        LATTICEModel,
        LGMRecModel,
        LightGCNModel,
        MGCNModel,
        MMGCNModel,
    )

    models = [
        LightGCNModel,
        MMGCNModel,
        LATTICEModel,
        BM3Model,
        FREEDOMModel,
        MGCNModel,
        DRAGONModel,
        LGMRecModel,
    ]
    assert len({model.__name__ for model in models}) == 8
    # Each model lives in its own module rather than sharing a single class.
    assert len({model.__module__ for model in models}) == 8


def test_graph_model_actually_learns() -> None:
    """Guard against the undertraining regression: BPR loss must decrease."""
    if not torch_available():
        return
    rng = np.random.default_rng(1)
    n_users, n_items, n_cats = 80, 160, 5
    item_cat = rng.integers(0, n_cats, n_items)
    user_cat = rng.integers(0, n_cats, n_users)
    rows: list[tuple[str, str]] = []
    for user in range(n_users):
        liked = np.where(item_cat == user_cat[user])[0]
        for _ in range(8):
            item = liked[rng.integers(len(liked))] if rng.random() < 0.8 else rng.integers(n_items)
            rows.append((f"u{user}", f"i{item}"))
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id"])
    interactions["timestamp"] = pd.date_range("2026-01-01", periods=len(interactions), freq="s")

    columns = ColumnConfig()
    model = LightGCNModel(columns, factors=32, layers=2, device="cpu")
    model._capture_catalog(interactions, None)
    model._prepare(interactions, None)
    model._device = torch.device("cpu")
    model._build()
    optimizer = torch.optim.Adam(
        model._parameters(), lr=model.learning_rate, weight_decay=model.regularization
    )
    generator = torch.Generator(device="cpu").manual_seed(0)
    losses: list[float] = []
    for _ in range(10):
        user_repr, item_repr = model._forward()
        order = torch.randperm(model._edge_users.size(0), generator=generator)
        for start in range(0, model._edge_users.size(0), model.batch_size):
            batch = order[start : start + model.batch_size]
            users = model._edge_users[batch]
            pos = model._edge_items[batch]
            neg = common.sample_negatives(
                users.cpu(), model.n_items, model._positive_mask, generator
            )
            loss = model._training_loss(users, pos, neg, user_repr, item_repr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.05


def test_model_configs_are_forwarded_to_registered_factories() -> None:
    predictor = MultiModalRecommender()
    captured: dict[str, object] = {}

    def factory(columns, **kwargs):
        captured.update(kwargs)
        return predictor.registry.create("Popularity", columns)

    predictor.register_model("ConfiguredPopularity", factory)
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "item_id": ["a", "b"],
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h"),
        }
    )
    predictor.fit(
        interactions,
        models="ConfiguredPopularity",
        model_configs={"ConfiguredPopularity": {"device": "cpu"}},
    )
    assert captured["device"] == "cpu"
    # The factory uses ``**kwargs``, so it also accepts the auto-injected seed.
    assert captured.get("random_state") == predictor.random_state


def test_info_nce_chunked_matches_dense() -> None:
    torch.manual_seed(0)
    first = torch.randn(4096, 8)
    second = torch.randn(4096, 8)
    chunked = common.info_nce(first, second, temperature=0.5, chunk=512)
    dense = common.info_nce(first, second, temperature=0.5, chunk=8192)
    assert abs(chunked.item() - dense.item()) < 1e-4


def test_cooccurrence_topk_bounds_degree_and_excludes_self() -> None:
    left = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=torch.long)
    right = torch.tensor([0, 1, 0, 2, 3, 1, 2, 0, 1, 3], dtype=torch.long)
    src, dst = common.cooccurrence_topk(left, right, num_left=4, num_right=4, k=2)
    assert src.numel() > 0
    assert (src != dst).all()
    counts = torch.bincount(src, minlength=4)
    assert bool((counts <= 2).all())


def test_cooccurrence_topk_sums_shared_item_contributions() -> None:
    # Users 0 and 1 share two degree-2 items (1/4 + 1/4 = 0.5); user 2 shares
    # only one degree-2 item with user 0 (1/4). User 1 must outrank user 2.
    left = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    right = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    src, dst = common.cooccurrence_topk(left, right, num_left=3, num_right=3, k=2)
    user0_neighbors = dst[src == 0].tolist()
    assert user0_neighbors[0] == 1


def test_cooccurrence_topk_excludes_self_with_duplicate_edges() -> None:
    # Duplicate (left, right) edges must not produce self-loops: node 0 appears
    # twice under right-node 0, but (0, 0) is still excluded.
    left = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    right = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    src, dst = common.cooccurrence_topk(left, right, num_left=3, num_right=1, k=3)
    assert src.numel() > 0
    assert (src != dst).all()


def test_negative_sampler_resolves_collisions_and_rejects_full_catalog() -> None:
    edges_users = torch.tensor([0, 0], dtype=torch.long)
    edges_items = torch.tensor([0, 1], dtype=torch.long)
    mask = common.build_positive_mask(edges_users, edges_items, 1, 3)
    generator = torch.Generator(device="cpu").manual_seed(0)
    negative = common.sample_negatives(torch.tensor([0]), 3, mask, generator)
    assert int(negative[0]) == 2

    full_mask = common.build_positive_mask(
        torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]), 1, 3
    )
    with pytest.raises(ValueError, match="full catalog"):
        common.sample_negatives(torch.tensor([0]), 3, full_mask, generator)
