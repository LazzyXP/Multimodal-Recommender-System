from __future__ import annotations

import pytest

from mmrec.hpo import TpeSearch


def test_tpe_rejects_empty_search_space() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        TpeSearch({})
    with pytest.raises(ValueError, match="choices cannot be empty"):
        TpeSearch({"lr": []})
    with pytest.raises(ValueError, match="gamma"):
        TpeSearch({"lr": [0.01]}, gamma=0)
    with pytest.raises(ValueError, match="warmup"):
        TpeSearch({"lr": [0.01]}, warmup=-1)
    with pytest.raises(ValueError, match="candidates"):
        TpeSearch({"lr": [0.01]}, candidates=0)


def test_tpe_warmup_returns_random_configs() -> None:
    space = {"factors": [16, 32, 64], "epochs": [3, 5, 10]}
    sampler = TpeSearch(space, random_state=0)
    for _ in range(3):
        config = sampler.suggest()
        assert config["factors"] in space["factors"]
        assert config["epochs"] in space["epochs"]


def test_tpe_biases_toward_good_values() -> None:
    space = {"lr": [0.01, 0.03, 0.1]}
    sampler = TpeSearch(space, random_state=1, warmup=3)
    # Seed the history so 0.03 is always the good choice.
    for _ in range(4):
        sampler.observe({"lr": 0.03}, 0.9)
        sampler.observe({"lr": 0.01}, 0.1)
        sampler.observe({"lr": 0.1}, 0.1)
    picks = [sampler.suggest()["lr"] for _ in range(100)]
    assert picks.count(0.03) > picks.count(0.01)
    assert picks.count(0.03) > picks.count(0.1)


def test_tpe_suggest_covers_full_space_without_history() -> None:
    space = {"layers": [1, 2, 3], "factors": [16, 32, 64]}
    sampler = TpeSearch(space, random_state=2, warmup=3)
    seen_layers = {sampler.suggest()["layers"] for _ in range(20)}
    assert seen_layers == set(space["layers"])


def test_tpe_deduplicates_until_search_space_is_exhausted() -> None:
    sampler = TpeSearch({"a": [1, 2], "b": ["x", "y"]}, random_state=3, warmup=1)
    configs = []
    for _ in range(4):
        config = sampler.suggest()
        configs.append(tuple(sorted(config.items())))
        sampler.observe(config, 1.0)
    assert len(set(configs)) == 4
