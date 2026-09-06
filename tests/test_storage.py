from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmrec import MultiModalRecommender, convert_to_parquet
from mmrec.cli import main
from mmrec.storage import file_fingerprint


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "item_id": ["i1", "i2", "i1", "i3"],
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="h"),
        }
    )


def test_converts_csv_and_tab_delimited_txt(tmp_path: Path) -> None:
    frame = _interactions()
    csv_source = tmp_path / "interactions.csv"
    txt_source = tmp_path / "interactions.txt"
    frame.to_csv(csv_source, index=False)
    frame.to_csv(txt_source, index=False, sep="\t")

    csv_parquet = convert_to_parquet(csv_source)
    txt_parquet = convert_to_parquet(txt_source)

    pd.testing.assert_frame_equal(pd.read_parquet(csv_parquet), frame, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_parquet(txt_parquet), frame, check_dtype=False)


def test_csv_fit_uses_reusable_parquet_cache(tmp_path: Path) -> None:
    source = tmp_path / "interactions.csv"
    _interactions().to_csv(source, index=False)
    cache = tmp_path / "cache"

    first = MultiModalRecommender(eval_metrics=["recall@2"], cache_dir=cache)
    first.fit(source, models=["Popularity", "ItemCF"])
    first_source = first.fit_summary().dataset_summary["sources"]["interactions"]

    second = MultiModalRecommender(eval_metrics=["recall@2"], cache_dir=cache)
    second.fit(source, models="Popularity")
    second_source = second.fit_summary().dataset_summary["sources"]["interactions"]

    assert first_source == second_source
    assert first_source.endswith(".parquet")
    assert Path(first_source).exists()


def test_saved_model_does_not_depend_on_original_csv(tmp_path: Path) -> None:
    source = tmp_path / "interactions.csv"
    _interactions().to_csv(source, index=False)
    predictor = MultiModalRecommender(eval_metrics=["recall@2"], cache_dir=tmp_path / "cache")
    predictor.fit(source, models="Popularity")
    model_path = predictor.save(tmp_path / "model")
    cached_source = Path(predictor.data_sources["interactions"] or "")

    source.unlink()
    cached_source.unlink()
    loaded = MultiModalRecommender.load(model_path)
    assert not loaded.recommend(users=["u1"], k=1).data.empty


def test_file_fingerprint_changes_when_parquet_contents_change(tmp_path: Path) -> None:
    source = tmp_path / "interactions.parquet"
    first = _interactions()
    first.to_parquet(source, index=False)
    original = file_fingerprint(source)
    changed = first.copy()
    changed.loc[0, "item_id"] = "replacement"
    changed.to_parquet(source, index=False)
    assert file_fingerprint(source) != original


def test_convert_cli(tmp_path: Path, capsys) -> None:
    source = tmp_path / "interactions.csv"
    output = tmp_path / "converted.parquet"
    _interactions().to_csv(source, index=False)

    assert main(["convert", str(source), str(output)]) == 0
    assert output.exists()
    assert str(output) in capsys.readouterr().out
