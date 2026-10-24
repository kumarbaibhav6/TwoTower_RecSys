import numpy as np
import pandas as pd
import pytest
import torch

from src import features
from src.dataset import FeatureStore
from src.evaluate import WeekEvaluator
from src.model import ModelConfig, TwoTowerModel
from src.split import Week

WEEK = Week("val", pd.Timestamp("2020-09-09"), pd.Timestamp("2020-09-15"))


@pytest.fixture
def store(tmp_path, monkeypatch):
    customers = pd.DataFrame({
        "customer_idx": np.arange(4, dtype=np.int32), "age": np.full(4, 30, dtype=np.float32),
        "club_member_status": ["ACTIVE"] * 4, "fashion_news_frequency": ["NONE"] * 4,
    })
    n_items = 120
    articles = pd.DataFrame({
        "article_id": np.arange(1000, 1000 + n_items, dtype=np.int32),
        "prod_name": ["p"] * n_items, "detail_desc": ["d"] * n_items,
        "product_group_name": ["Top"] * n_items, "colour_group_name": ["Black"] * n_items,
        "product_type_name": ["T-shirt"] * n_items, "index_group_name": ["Ladieswear"] * n_items,
    })
    tx = pd.DataFrame({
        "t_dat": pd.to_datetime(["2020-09-01", "2020-09-05", "2020-09-10", "2020-09-12", "2020-09-12",
                                 "2020-09-03", "2020-09-20"]),
        "customer_idx": np.array([0, 0, 0, 0, 2, 1, 3], dtype=np.int32),
        # customer 0: history {1000, 1001}, buys 1002 and 1119 in the week (neither bought in training)
        # customer 2: no history, buys 1001 in the week; customer 1: only history; customer 3: after the week
        "article_id": np.array([1000, 1001, 1002, 1119, 1001, 1003, 1004], dtype=np.int32),
    })
    tables = {"customers": customers, "articles": articles, "transactions": tx}
    monkeypatch.setattr(features, "load", lambda name, sample_frac=None, columns=None: tables[name])
    monkeypatch.setattr(features, "TRAIN_END", pd.Timestamp("2020-09-08"))
    features.build(None, tmp_path, with_text=False)
    return FeatureStore(tmp_path)


def test_week_users_history_pointers_and_ground_truth(store):
    ev = WeekEvaluator(store, WEEK)
    assert ev.users.tolist() == [0, 2]
    row = {a: i for i, a in enumerate(store.items["article_id"])}
    assert ev.actual[0] == {row[1002], row[1119]}
    assert ev.actual[2] == {row[1001]}
    s = store.seq
    hist0 = s["seq_item"][ev.user_start[0]:ev.hist_end[0]].tolist()
    assert hist0 == [row[1000], row[1001]]  # only purchases before the week
    assert ev.no_history.tolist() == [False, True]
    assert ev.new_items[row[1119]] and not ev.new_items[row[1001]]


def test_evaluate_runs_end_to_end_and_reports_slices(store):
    ev = WeekEvaluator(store, WEEK)
    model = TwoTowerModel(ModelConfig.from_meta(store.meta, use_text=False))
    result = ev.evaluate(model, torch.device("cpu"))
    assert result["users"] == 2
    # 120 items, top-100 retrieval: recall@100 must be between 0 and 1 and every slice present.
    assert 0.0 <= result["recall@100"] <= 1.0
    assert result["no_history_users"] == 1
    assert result["new_item_purchases"] == 2  # 1002 and 1119 both first appear in the week
    assert set(result) >= {"map@12", "recall@12", "no_history_map@12", "new_item_recall@100"}
