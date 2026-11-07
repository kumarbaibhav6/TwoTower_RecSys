"""Serving bundle: everything needed to recommend for any customer at a fixed point in time, without the feature files.

Written by src/export_embeddings.py, loaded by src/serve.py (the API) and src/build_index.py (the ANN benchmark).
The recommendation is the repeat + blend system from src/repeat_eval.py:

    each customer's `repeat` most recent past purchases first, then items ranked by
    cosine(user, item) + alpha * log(1 + item sales in the 7 days before the cutoff)

The popularity term is folded into the index: items are stored as [item_vector, alpha * prior] and queries as
[user_vector, 1], so one inner-product search returns the blended ranking.

Bundle files (one row per customer_idx / item row, same order as the feature files):
    config.json        run name, cutoff date, repeat, alpha, index settings, offline metrics
    model.pt           {"model": state_dict, "model_config": ...}
    item_emb.npy       [n_items, D] float32 item vectors (unit length)
    prior.npy          [n_items] float32 log1p(7-day sales)
    items.npz          item vocab ids and categorical codes (copied from the feature files)
    users.npz          user vocab ids and categorical codes (copied from the feature files)
    user_hist_rows.npy [n_customers, HIST_LEN] int32 last purchases before the cutoff, oldest first, -1 = empty
    user_n_prior.npy   [n_customers] int32 purchases before the cutoff
    user_last_day.npy  [n_customers] int16 day number of the last purchase (0 when none)
    user_repeat.npy    [n_customers, MAX_REPEAT] int32 distinct past purchases, newest first, -1 = empty
    customer_ids.npy   [n_customers] S64 the original customer_id strings
    articles.parquet   display fields per item row
    index.faiss        optional prebuilt index (src/build_index.py); otherwise exact search is built at load
"""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from src.dataset import USER_CATEGORICAL_KEYS
from src.evaluate import merge_repeat
from src.features import PAD_ID
from src.model import ModelConfig, TwoTowerModel, build_model

MAX_REPEAT = 12
ARTICLE_FIELDS = ["article_id", "prod_name", "product_type_name", "product_group_name", "colour_group_name",
                  "index_group_name", "detail_desc"]


def augmented_items(item_emb: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    return np.ascontiguousarray(np.hstack([item_emb, alpha * prior[:, None]]), dtype=np.float32)


def augmented_queries(user_vecs: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.hstack([user_vecs, np.ones((len(user_vecs), 1), np.float32)]), dtype=np.float32)


def make_index(vectors: np.ndarray, kind: str = "flat", **params) -> faiss.Index:
    """Inner-product index over `vectors`: "flat" (exact), "ivf" (nlist, nprobe) or "hnsw" (m, ef_search)."""
    dim = vectors.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, params.get("nlist", 1024), faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.nprobe = params.get("nprobe", 16)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, params.get("m", 32), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = params.get("ef_construction", 200)
        index.hnsw.efSearch = params.get("ef_search", 128)
    else:
        raise ValueError(f"unknown index type {kind!r}")
    index.add(vectors)
    return index


class ServingBundle:
    def __init__(self, bundle_dir: Path | str):
        self.dir = Path(bundle_dir)
        self.config = json.loads((self.dir / "config.json").read_text())
        self.cutoff_day = int(self.config["cutoff_day"])
        self.repeat = int(self.config["repeat"])
        self.alpha = float(self.config["alpha"])

        ck = torch.load(self.dir / "model.pt", map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ck["model_config"])
        if cfg.use_history and cfg.history_mode == "tower":
            raise NotImplementedError("serving supports history_mode='id' models (no item feature table at serve time)")
        self.model: TwoTowerModel = build_model(cfg, None, torch.device("cpu"), ck["model"]).eval()

        self.items = dict(np.load(self.dir / "items.npz"))
        self.users = dict(np.load(self.dir / "users.npz"))
        self.item_emb = np.load(self.dir / "item_emb.npy")
        self.prior = np.load(self.dir / "prior.npy")
        self.hist_rows = np.load(self.dir / "user_hist_rows.npy")
        self.n_prior = np.load(self.dir / "user_n_prior.npy")
        self.last_day = np.load(self.dir / "user_last_day.npy")
        self.repeat_items = np.load(self.dir / "user_repeat.npy")
        self.articles = pd.read_parquet(self.dir / "articles.parquet")

        self.customer_ids = np.load(self.dir / "customer_ids.npy")
        self._id_order = np.argsort(self.customer_ids)
        self._sorted_ids = self.customer_ids[self._id_order]

        self.item_vectors = augmented_items(self.item_emb, self.prior, self.alpha)
        self.index_config = self.config.get("index", {"type": "flat"})
        kind = self.index_config["type"]
        if kind == "flat":  # exact search builds instantly, so it is not stored
            self.index = make_index(self.item_vectors, "flat")
        else:
            self.index = faiss.read_index(str(self.dir / "index.faiss"))
            if kind == "hnsw":
                self.index.hnsw.efSearch = self.index_config["ef_search"]
            elif kind == "ivf":
                faiss.extract_index_ivf(self.index).nprobe = self.index_config["nprobe"]

    @property
    def n_customers(self) -> int:
        return len(self.customer_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_emb)

    def lookup(self, customer_id: str) -> int | None:
        key = customer_id.encode()
        pos = int(np.searchsorted(self._sorted_ids, key))
        if pos < len(self._sorted_ids) and self._sorted_ids[pos] == key:
            return int(self._id_order[pos])
        return None

    def user_features(self, customers: np.ndarray) -> dict[str, torch.Tensor]:
        """Same tensors as FeatureStore.user_features with today = the cutoff day."""
        rows = self.hist_rows[customers].astype(np.int64)
        valid = rows >= 0
        rows = np.where(valid, rows, 0)
        ids = np.where(valid, self.items["item_vocab"][rows], PAD_ID).astype(np.int64)
        n_prior = self.n_prior[customers].astype(np.int64)
        has_prior = n_prior > 0
        days_since = np.where(has_prior, self.cutoff_day - self.last_day[customers].astype(np.int64), 0)
        feats = {key: torch.from_numpy(self.users[key][customers].astype(np.int64)) for key in USER_CATEGORICAL_KEYS}
        feats["hist_ids"] = torch.from_numpy(ids)
        feats["hist_rows"] = torch.from_numpy(rows)
        feats["hist_mask"] = torch.from_numpy(valid)
        feats["numeric"] = torch.from_numpy(
            np.stack([np.log1p(n_prior), np.log1p(days_since), has_prior], axis=1).astype(np.float32)
        )
        return feats

    @torch.no_grad()
    def user_vectors(self, customers: np.ndarray) -> np.ndarray:
        return self.model.encode_users(self.user_features(customers)).numpy()

    def search(self, user_vecs: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(blended scores, item rows), [B, n] each, best first."""
        return self.index.search(augmented_queries(user_vecs), n)

    def repeat_lists(self, customers: np.ndarray, repeat: int) -> list[np.ndarray]:
        return [row[row >= 0].astype(np.int64) for row in self.repeat_items[customers, :repeat]]

    def recommend(self, customers: np.ndarray, k: int = 12, repeat: int | None = None) -> np.ndarray:
        """[B, k] item rows: up to `repeat` past purchases first, then the blended ranking."""
        repeat = min(self.repeat if repeat is None else repeat, MAX_REPEAT, k)
        _, ranked = self.search(self.user_vectors(customers), k + repeat)
        if repeat == 0:
            return ranked[:, :k]
        return merge_repeat(ranked, self.repeat_lists(customers, repeat), repeat, k)
