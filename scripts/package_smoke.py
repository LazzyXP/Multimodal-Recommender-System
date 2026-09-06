"""Exercise an installed distribution from outside the source checkout."""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mmrec import MultiModalRecommender


def main() -> None:
    data = pd.DataFrame(
        {
            "user_id": ["u1"] * 3 + ["u2"] * 3 + ["u3"] * 3,
            "item_id": ["a", "b", "c", "a", "c", "d", "b", "d", "e"],
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="h"),
        }
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        predictor = MultiModalRecommender(cache_dir=root / "cache").fit(
            data,
            models=["Popularity", "ItemCF"],
        )
        expected = predictor.recommend(["u1"], k=2).data
        assert not expected.empty
        assert predictor.model_best is not None
        restored = MultiModalRecommender.load(predictor.save(root / "model"))
        pd.testing.assert_frame_equal(expected, restored.recommend(["u1"], k=2).data)
    print(f"Installed multimodal-recommender {version('multimodal-recommender')}: smoke passed")


if __name__ == "__main__":
    main()
