"""PyTorch Dataset over the feature files written by src/features.py.

One example = one training-window purchase (deduplicated per customer/day/article). Batches are built with
vectorised NumPy indexing in `__getitems__`, which DataLoader calls with a whole list of indices.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import HIST_LEN, PAD_ID, vocab_ids
from src.split import MIN_ITEM_PURCHASES, MIN_USER_PURCHASES

USER_CATEGORICAL_KEYS = ["user_vocab", "age_bucket", "club_status", "news_frequency"]
ITEM_CATEGORICAL_KEYS = ["item_vocab", "product_group_name", "colour_group_name", "product_type_name", "index_group_name"]


class FeatureStore:
    """Memory-mapped feature arrays plus lookups shared by training, evaluation and serving."""

    def __init__(self, features_dir: Path, use_text: bool = True):
        self.dir = Path(features_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.users = dict(np.load(self.dir / "users.npz"))
        self.items = dict(np.load(self.dir / "items.npz"))
        # mmap_mode="r": pages are read from disk on demand and shared by DataLoader workers.
        self.seq = {
            name: np.load(self.dir / f"{name}.npy", mmap_mode="r")
            for name in ["seq_customer", "seq_item", "seq_day", "seq_user_start", "seq_hist_end"]
        }
        text_path = self.dir / "item_text_emb.npy"
        self.text_emb = np.load(text_path, mmap_mode="r") if use_text and text_path.exists() else None

        self._set_log_q()

    def _set_log_q(self) -> None:
        counts = self.items["train_count"].astype(np.float64)
        # logQ: log of each item's share of training examples; unseen items get a pseudo-count of 1.
        self.item_log_q = np.log(np.maximum(counts, 1.0) / counts.sum()).astype(np.float32)

    def refit_to(self, last_day: int) -> None:
        """Move the end of the training window to `last_day` (e.g. to retrain on train + val).

        Recomputes what was fitted on the training window in src/features.py: item/user purchase counts,
        the ID vocabularies (same thresholds) and logQ. Sequence and history pointers are unaffected.
        """
        in_window = np.asarray(self.seq["seq_day"]) <= last_day
        item_counts = np.bincount(np.asarray(self.seq["seq_item"])[in_window], minlength=len(self.items["article_id"]))
        user_counts = np.bincount(np.asarray(self.seq["seq_customer"])[in_window], minlength=len(self.users["user_vocab"]))
        self.items["train_count"] = item_counts.astype(np.int32)
        self.items["item_vocab"] = vocab_ids(item_counts, MIN_ITEM_PURCHASES)
        self.users["user_vocab"] = vocab_ids(user_counts, MIN_USER_PURCHASES)
        self.meta = {**self.meta, "train_end_day": int(last_day), "vocab_sizes": {
            **self.meta["vocab_sizes"],
            "item_vocab": int(self.items["item_vocab"].max()) + 1,
            "user_vocab": int(self.users["user_vocab"].max()) + 1,
        }}
        self._set_log_q()

    def history(self, hist_end: np.ndarray, user_start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Last HIST_LEN purchases in [user_start, hist_end), oldest first.

        Returns (item vocab ids with PAD_ID where empty, item rows with 0 where empty, validity mask).
        """
        positions = hist_end[:, None] - HIST_LEN + np.arange(HIST_LEN)
        valid = positions >= user_start[:, None]
        rows = np.where(valid, self.seq["seq_item"][np.where(valid, positions, 0)], 0).astype(np.int64)
        ids = np.where(valid, self.items["item_vocab"][rows], PAD_ID).astype(np.int64)
        return ids, rows, valid

    def user_features(self, customers: np.ndarray, hist_end: np.ndarray, user_start: np.ndarray,
                      today: np.ndarray) -> dict[str, torch.Tensor]:
        hist_ids, hist_rows, valid = self.history(hist_end, user_start)
        n_prior = hist_end - user_start
        has_prior = n_prior > 0
        last_day = self.seq["seq_day"][np.maximum(hist_end - 1, 0)].astype(np.int64)
        days_since = np.where(has_prior, today - last_day, 0)
        feats = {key: torch.from_numpy(self.users[key][customers].astype(np.int64)) for key in USER_CATEGORICAL_KEYS}
        feats["hist_ids"] = torch.from_numpy(hist_ids)
        feats["hist_rows"] = torch.from_numpy(hist_rows)  # used when history is pooled over item-tower outputs
        feats["hist_mask"] = torch.from_numpy(valid)
        feats["numeric"] = torch.from_numpy(
            np.stack([np.log1p(n_prior), np.log1p(days_since), has_prior], axis=1).astype(np.float32)
        )
        return feats

    def item_table(self) -> dict[str, torch.Tensor]:
        """Features of every item, for models that look up history items on the GPU (text kept in float16)."""
        table = {key: torch.from_numpy(self.items[key].astype(np.int64)) for key in ITEM_CATEGORICAL_KEYS}
        if self.text_emb is not None:
            table["text_emb"] = torch.from_numpy(np.asarray(self.text_emb))
        return table

    def item_features(self, item_rows: np.ndarray) -> dict[str, torch.Tensor]:
        feats = {key: torch.from_numpy(self.items[key][item_rows].astype(np.int64)) for key in ITEM_CATEGORICAL_KEYS}
        if self.text_emb is not None:
            feats["text_emb"] = torch.from_numpy(self.text_emb[item_rows].astype(np.float32))
        return feats


class TwoTowerDataset(Dataset):
    """(user features, item features, item row, log q) for every training-window purchase."""

    def __init__(self, store: FeatureStore, max_day: int | None = None, min_day: int | None = None):
        """Examples are purchases with min_day <= day <= max_day (default: the whole training window).

        Histories always reach back to the user's first purchase; min_day only limits which purchases are targets.
        """
        self.store = store
        max_day = store.meta["train_end_day"] if max_day is None else max_day
        day = np.asarray(store.seq["seq_day"])
        keep = day <= max_day if min_day is None else (day <= max_day) & (day >= min_day)
        self.rows = np.flatnonzero(keep).astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitems__(self, indices: list[int]) -> dict:
        rows = self.rows[np.asarray(indices)]
        s = self.store.seq
        item_rows = s["seq_item"][rows].astype(np.int64)
        return {
            "user": self.store.user_features(
                s["seq_customer"][rows], s["seq_hist_end"][rows].astype(np.int64),
                s["seq_user_start"][rows].astype(np.int64), s["seq_day"][rows].astype(np.int64),
            ),
            "item": self.store.item_features(item_rows),
            # Raw item rows (not vocab ids) identify duplicates for accidental-hit masking.
            "item_row": torch.from_numpy(item_rows),
            "item_log_q": torch.from_numpy(self.store.item_log_q[item_rows]),
        }

    def __getitem__(self, index: int) -> dict:
        return self.__getitems__([index])


def _identity(batch):
    return batch


def make_loader(dataset: TwoTowerDataset, batch_size: int, shuffle: bool = True, num_workers: int = 2,
                drop_last: bool = True, seed: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_identity,  # the dataset already returns a collated batch
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,  # in-batch negatives want full batches
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
