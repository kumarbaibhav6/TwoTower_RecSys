import numpy as np
import pandas as pd

from src import baselines


def test_repeat_purchase_orders_by_recency_dedups_and_fills(monkeypatch):
    history = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(["2020-09-01", "2020-09-03", "2020-09-05", "2020-09-02"]),
            "customer_idx": np.array([1, 1, 1, 2], dtype=np.int32),
            "article_id": np.array([10, 20, 10, 30], dtype=np.int32),
        }
    )
    monkeypatch.setattr(baselines, "load_transactions", lambda **kwargs: history)

    preds = baselines.repeat_purchase(
        users=np.array([1, 2, 3]), history_end=pd.Timestamp("2020-09-08"), k=4,
        fallback=np.array([10, 99, 98, 97]), sample_frac=None,
    )

    assert preds[1] == [10, 20, 99, 98]  # 10 re-bought most recently, appears once; fill skips 10
    assert preds[2] == [30, 10, 99, 98]
    assert preds[3] == [10, 99, 98, 97]  # no history: pure popularity
