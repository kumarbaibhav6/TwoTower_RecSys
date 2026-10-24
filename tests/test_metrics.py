import pytest

from src.metrics import average_precision_at_k, evaluate, recall_at_k


def test_ap_perfect_ranking_is_one():
    assert average_precision_at_k({1, 2, 3}, [1, 2, 3, 9, 9]) == 1.0


def test_ap_worked_example():
    # Hits at positions 1 and 3: precisions 1/1 and 2/3, normalised by min(|actual|=3, 12).
    assert average_precision_at_k({1, 2, 3}, [1, 7, 2, 8]) == pytest.approx((1 + 2 / 3) / 3)


def test_ap_rewards_early_hits():
    assert average_precision_at_k({5}, [5, 1, 2]) > average_precision_at_k({5}, [1, 2, 5])
    assert average_precision_at_k({5}, [1, 2, 5]) == pytest.approx(1 / 3)


def test_ap_ignores_hits_beyond_k_and_duplicates():
    assert average_precision_at_k({1}, [0] * 12 + [1], k=12) == 0.0
    # The repeated 1 must not count twice: only position 1 is a hit, 1/min(2, 12) = 0.5.
    assert average_precision_at_k({1, 2}, [1, 1, 3]) == pytest.approx(0.5)


def test_ap_normalises_by_k_when_user_bought_more_than_k():
    actual = set(range(20))
    assert average_precision_at_k(actual, list(range(12))) == 1.0


def test_ap_and_recall_empty_cases():
    assert average_precision_at_k(set(), [1, 2]) == 0.0
    assert average_precision_at_k({1}, []) == 0.0
    assert recall_at_k(set(), [1], 12) == 0.0


def test_recall_at_k():
    assert recall_at_k({1, 2, 3, 4}, [1, 9, 3], k=12) == 0.5
    assert recall_at_k({1, 2}, [9, 1, 2], k=2) == 0.5


def test_evaluate_averages_over_all_actual_users_including_missing():
    result = evaluate({"a": {1}, "b": {2}}, {"a": [1]}, recall_ks=(12,))
    assert result == {"map@12": 0.5, "recall@12": 0.5, "users": 2}
