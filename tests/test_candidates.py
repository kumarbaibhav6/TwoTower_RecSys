import numpy as np
import pandas as pd

from src.candidates import bought_together, coverage, merge, same_product, seeds


def test_seeds_take_newest_first_and_skip_empty_histories():
    s = seeds([np.array([5, 3, 9]), np.array([], dtype=np.int64)], np.array([10, 11]), n_seeds=2)
    assert s.to_dict("list") == {"user": [10, 10], "seed": [5, 3], "recency": [0, 1]}


def test_bought_together_sums_partner_counts_weighted_by_recency():
    user_seeds = pd.DataFrame({"user": [1, 1], "seed": [5, 3], "recency": [0, 1]})
    pairs = pd.DataFrame({"seed": [5, 5, 3], "partner": [7, 8, 8], "count": [4, 3, 4]})
    # 8: 3/1 + 4/2 = 5 beats 7: 4/1 = 4
    assert bought_together(user_seeds, pairs, np.array([1, 2]), n=5)[0].tolist() == [8, 7]


def test_same_product_returns_other_variants_by_sales():
    codes = np.array([100, 100, 100, 200])  # items 0-2 share a product code
    sales = np.array([1.0, 5.0, 3.0, 9.0])
    user_seeds = pd.DataFrame({"user": [1], "seed": [0], "recency": [0]})
    assert same_product(user_seeds, codes, sales, np.array([1]), n=5)[0].tolist() == [1, 2]


def test_merge_respects_quotas_dedups_and_fills_with_two_tower():
    lists = {
        "repeat": [np.array([1, 2, 3])], "bought_together": [np.array([2, 4])], "same_product": [np.array([5])],
        "popularity": [np.array([6, 7])], "two_tower": [np.arange(100, 120)],
    }
    pool = merge(lists, budget=10)[0]
    # quotas at budget 10: repeat 1, bought_together 2, same_product 1, popularity 1; two_tower fills to 10
    assert pool.tolist() == [1, 2, 4, 5, 6, 100, 101, 102, 103, 104]


def test_coverage_counts_share_of_purchases_in_pool():
    r = coverage([{1, 2}, {3}], [np.array([1, 9]), np.array([4])])
    assert r["recall"] == 0.25 and r["micro_recall"] == round(1 / 3, 4)
