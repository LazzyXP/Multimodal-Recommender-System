"""Run a complete multimodel recommendation experiment."""

import pandas as pd

from mmrec import MultiModalRecommender

interactions = pd.DataFrame(
    {
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3", "u3"],
        "item_id": ["i1", "i2", "i3", "i1", "i3", "i4", "i2", "i3", "i4"],
        "timestamp": pd.date_range("2026-01-01", periods=9, freq="h"),
    }
)

recommender = MultiModalRecommender(eval_metrics=["recall@3", "ndcg@3", "mrr@3"])
recommender.fit(interactions, models=["Popularity", "ItemCF"])

print(recommender.leaderboard())
print(recommender.recommend(users=["u1", "u2"], k=3).data)
