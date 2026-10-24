"""Ranking metrics. `actual` is the set of items a user bought; `predicted` is our ranked list."""

from collections.abc import Iterable, Sequence

import numpy as np


def average_precision_at_k(actual: Iterable, predicted: Sequence, k: int = 12) -> float:
    """AP@k as defined by the H&M Kaggle competition.

    Precision is counted at each position holding a not-yet-seen relevant item, and the sum is
    normalised by min(|actual|, k), so a user with 3 purchases can score a perfect 1.0.
    """
    actual = set(actual)
    if not actual:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, item in enumerate(predicted[:k]):
        if item in actual and item not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(item)
    return score / min(len(actual), k)


def recall_at_k(actual: Iterable, predicted: Sequence, k: int) -> float:
    actual = set(actual)
    if not actual:
        return 0.0
    return len(actual & set(predicted[:k])) / len(actual)


def evaluate(actual_by_user: dict, predicted_by_user: dict, recall_ks: Sequence[int] = (12, 100)) -> dict:
    """Mean MAP@12 and Recall@k over every user in `actual_by_user`; missing predictions score 0."""
    ap, recalls = [], {k: [] for k in recall_ks}
    for user, actual in actual_by_user.items():
        predicted = predicted_by_user.get(user, [])
        ap.append(average_precision_at_k(actual, predicted, 12))
        for k in recall_ks:
            recalls[k].append(recall_at_k(actual, predicted, k))
    result = {"map@12": float(np.mean(ap))}
    result.update({f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()})
    result["users"] = len(ap)
    return result
