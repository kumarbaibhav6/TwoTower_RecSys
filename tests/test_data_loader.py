import numpy as np
import pandas as pd
import pytest

from src import data_loader


@pytest.fixture
def raw_dir(tmp_path):
    pd.DataFrame(
        {
            "customer_id": ["aaa", "bbb", "ccc", "ddd"],
            "FN": [1.0, None, 1.0, None],
            "Active": [1.0, None, None, 1.0],
            "club_member_status": ["ACTIVE", "ACTIVE", "PRE-CREATE", None],
            "fashion_news_frequency": ["Regularly", "NONE", None, "NONE"],
            "age": [25, 40, None, 33],
            "postal_code": ["p1", "p2", "p1", "p3"],
        }
    ).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame(
        {
            "article_id": ["0108775015", "0108775044"],
            "product_code": [108775, 108775],
            "prod_name": ["Strap top", "Strap top"],
            "colour_group_name": ["Black", "White"],
            "detail_desc": ["Jersey top", "Jersey top"],
        }
    ).to_csv(tmp_path / "articles.csv", index=False)
    pd.DataFrame(
        {
            "t_dat": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"],
            "customer_id": ["ccc", "aaa", "ccc", "ddd", "ccc"],
            "article_id": ["0108775015", "0108775044", "0108775044", "0108775015", "0108775015"],
            "price": [0.01, 0.02, 0.03, 0.04, 0.05],
            "sales_channel_id": [1, 2, 2, 1, 2],
        }
    ).to_csv(tmp_path / "transactions_train.csv", index=False)
    return tmp_path


@pytest.fixture(autouse=True)
def processed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", tmp_path / "processed")


def test_full_conversion_maps_customers_and_compacts_dtypes(raw_dir):
    data_loader.convert(raw_dir)
    tx = data_loader.load("transactions")
    customers = data_loader.load("customers")

    id_by_idx = dict(zip(customers["customer_idx"], customers["customer_id"]))
    assert [id_by_idx[i] for i in tx["customer_idx"]] == ["ccc", "aaa", "ccc", "ddd", "ccc"]
    assert tx["article_id"].tolist() == [108775015, 108775044, 108775044, 108775015, 108775015]
    assert tx["customer_idx"].dtype == np.int32
    assert tx["article_id"].dtype == np.int32
    assert tx["price"].dtype == np.float32
    assert tx["sales_channel_id"].dtype == np.int8
    assert tx["t_dat"].min() == pd.Timestamp("2020-01-01")


def test_sample_keeps_whole_customer_histories(raw_dir):
    data_loader.convert(raw_dir)
    full = data_loader.load("transactions")
    data_loader.convert(raw_dir, sample_frac=0.5)
    sample = data_loader.load("transactions", sample_frac=0.5)
    sampled_customers = data_loader.load("customers", sample_frac=0.5)

    assert len(sampled_customers) == 2
    for idx in sample["customer_idx"].unique():
        assert (sample["customer_idx"] == idx).sum() == (full["customer_idx"] == idx).sum()
    assert set(sample["customer_idx"]) <= set(sampled_customers["customer_idx"])


def test_unknown_customer_in_transactions_raises(raw_dir):
    tx = pd.read_csv(raw_dir / "transactions_train.csv", dtype=str)
    tx.loc[0, "customer_id"] = "zzz"
    tx.to_csv(raw_dir / "transactions_train.csv", index=False)
    with pytest.raises(ValueError, match="missing from customers.csv"):
        data_loader.convert(raw_dir)
