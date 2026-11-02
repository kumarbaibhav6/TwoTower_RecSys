"""Retrieval evaluation: every target-week buyer against every article, top-k by cosine similarity.

Uses the same MAP@12 / Recall@k as the baselines (src/metrics.py), so the numbers are directly comparable.
Ground truth comes from the feature sequence: unique articles each customer bought in the week.
"""

import numpy as np
import torch

from src.dataset import FeatureStore
from src.features import day_number
from src.metrics import evaluate
from src.model import TwoTowerModel
from src.split import Week

TOP_K = 100


def merge_repeat(ranked: np.ndarray, repeat_lists: list[np.ndarray], repeat: int, k: int = TOP_K) -> np.ndarray:
    """Per user: up to `repeat` past purchases first, then `ranked` (needs >= k + repeat columns) minus those."""
    merged = np.empty((len(ranked), k), dtype=ranked.dtype)
    for u, (row, own) in enumerate(zip(ranked, repeat_lists)):
        own = own[:repeat]
        merged[u] = np.concatenate([own, row[~np.isin(row, own)]])[:k]
    return merged


class WeekEvaluator:
    """Precomputes users, their history pointers and ground truth for one target week."""

    def __init__(self, store: FeatureStore, week: Week, max_users: int | None = None, seed: int = 0):
        self.store, self.week = store, week
        s = store.seq
        start_day, end_day = int(day_number([week.start])[0]), int(day_number([week.end])[0])
        self.start_day = start_day
        day = np.asarray(s["seq_day"])
        in_week = (day >= start_day) & (day <= end_day)
        cust, item = np.asarray(s["seq_customer"])[in_week], np.asarray(s["seq_item"])[in_week]

        users = np.unique(cust)
        if max_users is not None and len(users) > max_users:
            users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
        self.users = users
        keep = np.isin(cust, users)
        cust, item = cust[keep], item[keep]
        bounds = np.flatnonzero(np.diff(cust)) + 1  # rows are sorted by customer
        firsts = cust[np.r_[0, bounds]] if len(cust) else np.empty(0, dtype=cust.dtype)
        # Ground truth in item rows; items are unique per (customer, day) and deduplicated across days here.
        self.actual = {u: set(rows.tolist()) for u, rows in zip(firsts, np.split(item, bounds))}

        # History = the user's rows strictly before the week. Rows are sorted by (customer, day), so a
        # combined (customer, day) key turns both pointers into binary searches.
        keys = np.asarray(s["seq_customer"]).astype(np.int64) << 16 | day.astype(np.int64)
        u64 = users.astype(np.int64) << 16
        self.user_start = np.searchsorted(keys, u64, side="left")
        self.hist_end = np.searchsorted(keys, u64 | start_day, side="left")
        self.today = np.full(len(users), start_day, dtype=np.int64)

        train_counts = store.items["train_count"]
        self.new_items = train_counts == 0  # never bought in the training window
        self.no_history = self.hist_end == self.user_start

    @torch.no_grad()
    def item_vectors(self, model: TwoTowerModel, device: torch.device, chunk: int = 16384) -> torch.Tensor:
        n_items = len(self.store.items["article_id"])
        out = []
        for start in range(0, n_items, chunk):
            rows = np.arange(start, min(start + chunk, n_items))
            feats = {k: v.to(device) for k, v in self.store.item_features(rows).items()}
            out.append(model.encode_items(feats))
        return torch.cat(out)

    @torch.no_grad()
    def user_vectors(self, model: TwoTowerModel, device: torch.device, chunk: int = 8192) -> torch.Tensor:
        out = []
        for start in range(0, len(self.users), chunk):
            sl = slice(start, start + chunk)
            feats = self.store.user_features(self.users[sl], self.hist_end[sl], self.user_start[sl], self.today[sl])
            out.append(model.encode_users({k: v.to(device) for k, v in feats.items()}))
        return torch.cat(out)

    def recent_popularity(self, days: int = 7) -> np.ndarray:
        """log(1 + sales) per item over the `days` before the week: the 'what is selling now' prior."""
        day = np.asarray(self.store.seq["seq_day"])
        recent = (day < self.start_day) & (day >= self.start_day - days)
        counts = np.bincount(np.asarray(self.store.seq["seq_item"])[recent], minlength=len(self.new_items))
        return np.log1p(counts).astype(np.float32)

    @torch.no_grad()
    def scores_fn(self, model: TwoTowerModel, device: torch.device):
        """Item and user vectors once; returns f(user_slice) -> cosine scores, for reuse across sweeps."""
        was_training = model.training
        model.eval()
        items = self.item_vectors(model, device)
        users = self.user_vectors(model, device)
        model.train(was_training)
        return lambda sl: users[sl] @ items.T

    def repeat_candidates(self, max_items: int = TOP_K) -> list[np.ndarray]:
        """Each user's previously bought item rows, most recent first, deduplicated, at most max_items."""
        items = np.asarray(self.store.seq["seq_item"])
        out = []
        for start, end in zip(self.user_start, self.hist_end):
            past = items[start:end][::-1]  # rows are in time order; newest first
            _, first = np.unique(past, return_index=True)
            out.append(past[np.sort(first)][:max_items])
        return out

    @torch.no_grad()
    def recommend(self, model: TwoTowerModel, device: torch.device, k: int = TOP_K, chunk: int = 2048,
                  prior: np.ndarray | None = None, prior_weight: float = 0.0, scores=None,
                  repeat: int = 0, repeat_lists: list[np.ndarray] | None = None) -> np.ndarray:
        """[n_users, k] item rows, best first. Optionally adds prior_weight * prior[item] to every score.

        With repeat > 0, each user's `repeat` most recent past purchases go first and the ranked list fills the rest.
        """
        scores = scores or self.scores_fn(model, device)
        bias = None if prior is None or prior_weight == 0 else torch.from_numpy(prior_weight * prior).to(device)
        repeat = min(repeat, k)
        top = []
        for i in range(0, len(self.users), chunk):
            s = scores(slice(i, i + chunk))
            top.append(torch.topk(s if bias is None else s + bias, k + repeat, dim=1).indices.cpu())
        top = torch.cat(top).numpy()
        if repeat == 0:
            return top[:, :k]
        return merge_repeat(top, repeat_lists if repeat_lists is not None else self.repeat_candidates(repeat), repeat, k)

    def evaluate(self, model: TwoTowerModel | None, device: torch.device, prior: np.ndarray | None = None,
                 prior_weight: float = 0.0, scores=None, repeat: int = 0,
                 repeat_lists: list[np.ndarray] | None = None, top: np.ndarray | None = None) -> dict:
        """Metrics for the model's top-k, or for a precomputed `top` [n_users, k] of item rows."""
        if top is None:
            top = self.recommend(model, device, prior=prior, prior_weight=prior_weight, scores=scores,
                                 repeat=repeat, repeat_lists=repeat_lists)
        predicted = {u: row.tolist() for u, row in zip(self.users, top)}
        result = evaluate(self.actual, predicted)

        cold_users = self.users[self.no_history]
        cold = evaluate({u: self.actual[u] for u in cold_users}, predicted) if len(cold_users) else {}
        result["no_history_users"] = len(cold_users)
        result["no_history_map@12"] = cold.get("map@12", 0.0)
        result["no_history_recall@100"] = cold.get("recall@100", 0.0)

        # Micro-averaged recall over (user, new article) purchases: did we retrieve articles nobody bought in training?
        hits = total = 0
        for u, row in zip(self.users, top):
            new = {i for i in self.actual[u] if self.new_items[i]}
            if new:
                total += len(new)
                hits += len(new & set(row.tolist()))
        result["new_item_purchases"] = total
        result["new_item_recall@100"] = hits / total if total else 0.0
        return result
