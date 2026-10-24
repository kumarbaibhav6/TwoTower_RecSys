"""Time-based train / validation / test split.

Each target week is predicted from everything strictly before it:
    val  = 2020-09-09 .. 2020-09-15, history = all purchases <= 2020-09-08
    test = 2020-09-16 .. 2020-09-22, history = all purchases <= 2020-09-15

Model weights are trained on the train window (<= 2020-09-08) only; the test history also includes
the val week because, at prediction time, that week has already happened.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data_loader import load

TRAIN_END = pd.Timestamp("2020-09-08")

# Sparsity thresholds from the plan, applied to the training window only.
MIN_USER_PURCHASES = 5  # keep users with > 5 purchases
MIN_ITEM_PURCHASES = 20  # keep items with > 20 purchases


@dataclass(frozen=True)
class Week:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive

    @property
    def history_end(self) -> pd.Timestamp:
        return self.start - pd.Timedelta(days=1)


VAL = Week("val", pd.Timestamp("2020-09-09"), pd.Timestamp("2020-09-15"))
TEST = Week("test", pd.Timestamp("2020-09-16"), pd.Timestamp("2020-09-22"))
WEEKS = {w.name: w for w in (VAL, TEST)}


def load_transactions(
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    columns: list[str] | None = None,
    sample_frac: float | None = None,
    customers: np.ndarray | None = None,
) -> pd.DataFrame:
    """Transactions with start <= t_dat <= end (both inclusive, either optional).

    Restricting to `customers` happens while reading, one row group at a time, so the full table is
    never in memory.
    """
    filters = []
    if start is not None:
        filters.append(("t_dat", ">=", start))
    if end is not None:
        filters.append(("t_dat", "<=", end))
    if customers is not None:
        filters.append(("customer_idx", "in", customers.tolist()))
    return load("transactions", sample_frac, columns=columns, filters=filters or None)


def ground_truth(week: Week, sample_frac: float | None = None) -> pd.Series:
    """customer_idx -> sorted array of unique article_ids bought in the week (buyers only).

    Repeat purchases of the same article count once, matching the Kaggle evaluation.
    """
    tx = load_transactions(week.start, week.end, ["customer_idx", "article_id"], sample_frac)
    return tx.groupby("customer_idx")["article_id"].unique().map(np.sort)


def sparsity_report(train: pd.DataFrame, eval_users: dict[str, np.ndarray]) -> dict:
    """What the user/item sparsity filter would remove from the training window.

    Also reports how many evaluation users the filter would drop, i.e. users the model must still
    serve even though they are excluded from training (cold or near-cold start).
    """
    user_counts = train["customer_idx"].value_counts()
    item_counts = train["article_id"].value_counts()
    keep_users = user_counts.index[user_counts > MIN_USER_PURCHASES]
    keep_items = item_counts.index[item_counts > MIN_ITEM_PURCHASES]
    kept = train["customer_idx"].isin(keep_users).to_numpy() & train["article_id"].isin(keep_items).to_numpy()

    report = {
        "thresholds": {"user_purchases_gt": MIN_USER_PURCHASES, "item_purchases_gt": MIN_ITEM_PURCHASES},
        "train_transactions": len(train),
        "train_users": len(user_counts),
        "train_items": len(item_counts),
        "kept_transactions": int(kept.sum()),
        "kept_users": len(keep_users),
        "kept_items": len(keep_items),
    }
    report["removed_transactions_pct"] = round(100 * (1 - report["kept_transactions"] / len(train)), 2)
    report["removed_users_pct"] = round(100 * (1 - len(keep_users) / len(user_counts)), 2)
    report["removed_items_pct"] = round(100 * (1 - len(keep_items) / len(item_counts)), 2)

    for name, users in eval_users.items():
        in_train = np.isin(users, user_counts.index)
        report[f"{name}_users"] = len(users)
        report[f"{name}_users_no_train_history_pct"] = round(100 * (1 - in_train.mean()), 2)
        report[f"{name}_users_dropped_by_filter_pct"] = round(100 * (1 - np.isin(users, keep_users).mean()), 2)
    return report
