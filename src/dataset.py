"""PyTorch Dataset over the feature files written by src/features.py.

One example = one training-window purchase (deduplicated per customer/day/article). Batches are built with
vectorised NumPy indexing in `__getitems__`, which DataLoader calls with a whole list of indices.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.features import HIST_LEN, PAD_ID

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

        counts = self.items["train_count"].astype(np.float64)
        # logQ: log of each item's share of training examples; unseen items get a pseudo-count of 1.
        self.item_log_q = np.log(np.maximum(counts, 1.0) / counts.sum()).astype(np.float32)

    def history(self, hist_end: np.ndarray, user_start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Last HIST_LEN item vocab ids in [user_start, hist_end), oldest first; PAD_ID where empty."""
        positions = hist_end[:, None] - HIST_LEN + np.arange(HIST_LEN)
        valid = positions >= user_start[:, None]
        rows = self.seq["seq_item"][np.where(valid, positions, 0)]
        ids = np.where(valid, self.items["item_vocab"][rows], PAD_ID).astype(np.int64)
        return ids, valid

    def user_features(self, customers: np.ndarray, hist_end: np.ndarray, user_start: np.ndarray,
                      today: np.ndarray) -> dict[str, torch.Tensor]:
        hist_ids, valid = self.history(hist_end, user_start)
        n_prior = hist_end - user_start
        has_prior = n_prior > 0
        last_day = self.seq["seq_day"][np.maximum(hist_end - 1, 0)].astype(np.int64)
        days_since = np.where(has_prior, today - last_day, 0)
        feats = {key: torch.from_numpy(self.users[key][customers].astype(np.int64)) for key in USER_CATEGORICAL_KEYS}
        feats["hist_ids"] = torch.from_numpy(hist_ids)
        feats["hist_mask"] = torch.from_numpy(valid)
        feats["numeric"] = torch.from_numpy(
            np.stack([np.log1p(n_prior), np.log1p(days_since), has_prior], axis=1).astype(np.float32)
        )
        return feats

    def item_features(self, item_rows: np.ndarray) -> dict[str, torch.Tensor]:
        feats = {key: torch.from_numpy(self.items[key][item_rows].astype(np.int64)) for key in ITEM_CATEGORICAL_KEYS}
        if self.text_emb is not None:
            feats["text_emb"] = torch.from_numpy(self.text_emb[item_rows].astype(np.float32))
        return feats


class TwoTowerDataset(Dataset):
    """(user features, item features, item row, log q) for every training-window purchase."""

    def __init__(self, store: FeatureStore, max_day: int | None = None):
        self.store = store
        max_day = store.meta["train_end_day"] if max_day is None else max_day
        self.rows = np.flatnonzero(np.asarray(store.seq["seq_day"]) <= max_day).astype(np.int64)

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
