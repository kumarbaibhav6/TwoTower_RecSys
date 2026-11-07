import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.evaluate import WeekEvaluator
from src.export_embeddings import write_bundle
from src.model import ModelConfig, TwoTowerModel
from src.serving import ServingBundle, augmented_queries
from test_evaluate import WEEK, store  # noqa: F401  (shared fixture: 4 customers, 120 items)

ALPHA, REPEAT = 0.5, 1


@pytest.fixture
def bundle_dir(store, tmp_path):  # noqa: F811
    processed = tmp_path / "processed"
    processed.mkdir()
    n_items = len(store.items["article_id"])
    pd.DataFrame({
        "article_id": store.items["article_id"], "prod_name": [f"item {i}" for i in range(n_items)],
        "product_type_name": "T-shirt", "product_group_name": "Top", "colour_group_name": "Black",
        "index_group_name": "Ladieswear", "detail_desc": "d",
    }).to_parquet(processed / "articles.parquet")
    pd.DataFrame({"customer_idx": np.arange(4, dtype=np.int32), "customer_id": [f"c{i}" for i in range(4)]}) \
        .to_parquet(processed / "customers.parquet")
    torch.manual_seed(0)
    cfg = ModelConfig.from_meta(store.meta, use_text=False)
    model = TwoTowerModel(cfg).eval()
    week = WeekEvaluator(store, WEEK)
    write_bundle(store, model, cfg.to_dict(), week, processed, tmp_path / "bundle", run_name="test",
                 repeat=REPEAT, alpha=ALPHA)
    return tmp_path / "bundle", model, week


def test_bundle_matches_offline_evaluation(bundle_dir):
    path, model, week = bundle_dir
    bundle = ServingBundle(path)
    cpu = torch.device("cpu")
    np.testing.assert_allclose(bundle.user_vectors(week.users), week.user_vectors(model, cpu).numpy(), atol=1e-6)
    served = bundle.recommend(week.users, k=100, repeat=REPEAT)
    offline = week.recommend(model, cpu, prior=week.recent_popularity(), prior_weight=ALPHA, repeat=REPEAT)
    assert (served[:, :REPEAT] == offline[:, :REPEAT]).all()
    # The toy items all share one vector, so many scores tie and the order among ties is arbitrary: compare the
    # blended score at every position instead of the item rows.
    queries = augmented_queries(bundle.user_vectors(week.users))
    score = lambda rows: np.take_along_axis(queries @ bundle.item_vectors.T, rows, axis=1)  # noqa: E731
    np.testing.assert_allclose(score(served[:, REPEAT:]), score(offline[:, REPEAT:]), atol=1e-5)


def test_repeat_lists_match_offline_candidates(bundle_dir):
    path, _, week = bundle_dir
    bundle = ServingBundle(path)
    offline = week.repeat_candidates()
    served = bundle.repeat_lists(week.users, 12)
    assert [s.tolist() for s in served] == [o.tolist() for o in offline]


def test_customer_lookup(bundle_dir):
    bundle = ServingBundle(bundle_dir[0])
    assert bundle.lookup("c2") == 2
    assert bundle.lookup("nope") is None


def test_api_recommend_and_errors(bundle_dir, monkeypatch):
    from src.serve import app

    monkeypatch.setenv("BUNDLE_DIR", str(bundle_dir[0]))
    with TestClient(app) as client:
        r = client.post("/recommend", json={"customer_id": "c0", "k": 12}).json()
        assert len(r["items"]) == 12 and len({i["article_id"] for i in r["items"]}) == 12
        assert r["items"][0]["source"] == "repeat" and r["items"][1]["source"] == "model"
        assert r["as_of"] == "2020-09-09"
        assert client.get("/customers/c0").json()["purchases"] == 2
        assert client.post("/recommend", json={"customer_id": "nope"}).status_code == 404
        assert client.post("/recommend", json={"customer_id": "c0", "k": 0}).status_code == 422
        assert client.get("/health").json()["status"] == "ok"
