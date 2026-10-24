import numpy as np
import pandas as pd
import pytest

from src import features
from src.dataset import FeatureStore, TwoTowerDataset, make_loader
from src.features import PAD_ID, RARE_ID


@pytest.fixture
def store(tmp_path, monkeypatch):
    customers = pd.DataFrame(
        {
            "customer_idx": np.array([0, 1, 2], dtype=np.int32),
            "age": np.array([23, np.nan, 70], dtype=np.float32),
            "club_member_status": ["ACTIVE", None, "LEFT CLUB"],
            "fashion_news_frequency": ["None", "NONE", "Regularly"],
        }
    )
    articles = pd.DataFrame(
        {
            "article_id": np.array([100, 200, 300, 400], dtype=np.int32),
            "prod_name": ["a", "b", "c", "d"],
            "detail_desc": ["x", "y", None, "z"],
            "product_group_name": ["Top", "Top", "Trousers", "Socks"],
            "colour_group_name": ["Black", "White", "Black", "Blue"],
            "product_type_name": ["T-shirt", "T-shirt", "Jeans", "Socks"],
            "index_group_name": ["Ladieswear", "Divided", "Ladieswear", "Menswear"],
        }
    )
    tx = pd.DataFrame(
        {
            "t_dat": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-05", "2020-01-05", "2020-01-05", "2020-02-01",
                 "2020-01-03", "2020-09-20"]
            ),
            "customer_idx": np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int32),
            # customer 0 buys 100 twice on 2020-01-05 (duplicate) and 400 after the training window.
            "article_id": np.array([100, 200, 300, 100, 100, 200, 100, 400], dtype=np.int32),
        }
    )
    tables = {"customers": customers, "articles": articles, "transactions": tx}
    monkeypatch.setattr(features, "load", lambda name, sample_frac=None, columns=None: tables[name])
    # Tiny data: every item/user counts as frequent enough for its own id.
    monkeypatch.setattr(features, "MIN_ITEM_PURCHASES", 0)
    monkeypatch.setattr(features, "MIN_USER_PURCHASES", 0)
    features.build(None, tmp_path, with_text=False)
    return FeatureStore(tmp_path)


def item_vocab(store, article_ids):
    rows = [int(np.flatnonzero(store.items["article_id"] == a)[0]) for a in article_ids]
    return store.items["item_vocab"][rows].tolist()


def test_sequence_is_sorted_and_deduplicated(store):
    s = store.seq
    assert len(s["seq_item"]) == 7  # 8 transactions minus one same-day duplicate
    keys = list(zip(s["seq_customer"], s["seq_day"], s["seq_item"]))
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_history_excludes_same_day_basket_and_future(store):
    ds = TwoTowerDataset(store)
    assert len(ds) == 6  # the 2020-09-20 purchase is outside the training window
    batch = ds.__getitems__(list(range(len(ds))))
    s = store.seq
    for i, row in enumerate(ds.rows):
        hist = batch["user"]["hist_ids"][i][batch["user"]["hist_mask"][i]].tolist()
        # Expected: this customer's items from strictly earlier days, in time order.
        earlier = (s["seq_customer"] == s["seq_customer"][row]) & (s["seq_day"] < s["seq_day"][row])
        assert hist == store.items["item_vocab"][s["seq_item"][earlier]].tolist()

    # Concretely, customer 0's 2020-02-01 purchase sees the 01-01 and 01-05 baskets, not itself.
    feb = int(np.flatnonzero(s["seq_day"][ds.rows] == features.day_number(["2020-02-01"])[0])[0])
    hist = batch["user"]["hist_ids"][feb][batch["user"]["hist_mask"][feb]].tolist()
    assert hist == item_vocab(store, [100, 200, 100, 300])
    assert batch["user"]["numeric"][feb, 0].item() == pytest.approx(np.log1p(4))  # 4 prior purchases
    assert batch["user"]["numeric"][feb, 1].item() == pytest.approx(np.log1p(27))  # days since 2020-01-05


def test_first_purchase_has_empty_padded_history(store):
    ds = TwoTowerDataset(store)
    batch = ds.__getitems__([0])
    assert not batch["user"]["hist_mask"][0].any()
    assert (batch["user"]["hist_ids"][0] == PAD_ID).all()
    assert batch["user"]["numeric"][0].tolist() == [0.0, 0.0, 0.0]


def test_encodings_reserve_zero_for_unknown(store):
    assert store.users["age_bucket"].tolist() == [2, 0, 11]  # 23 -> bucket 2, NaN -> 0, 70 -> last
    assert store.users["club_status"][1] == 0
    # 'None' and 'NONE' are the same category.
    assert store.users["news_frequency"][0] == store.users["news_frequency"][1] != 0
    assert (store.items["item_vocab"] >= RARE_ID).all()


def test_log_q_matches_training_frequencies(store):
    counts = store.items["train_count"]
    assert counts.sum() == 6
    np.testing.assert_allclose(np.exp(store.item_log_q[counts > 0]), counts[counts > 0] / 6, rtol=1e-6)


def test_loader_yields_full_batches(store):
    loader = make_loader(TwoTowerDataset(store), batch_size=4, num_workers=0)
    batches = list(loader)
    assert len(batches) == 1  # drop_last: 6 examples -> one batch of 4
    assert batches[0]["user"]["hist_ids"].shape == (4, features.HIST_LEN)
    assert batches[0]["item"]["item_vocab"].shape == (4,)
