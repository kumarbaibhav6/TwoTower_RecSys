import numpy as np
import torch

from src.evaluate import TOP_K, WeekEvaluator, merge_repeat
from src.matched_eval import by_history, paired_bootstrap, ranked_lists, run_week
from src.model import ModelConfig, TwoTowerModel
from test_evaluate import WEEK, store  # noqa: F401  (shared fixture: 4 customers, 120 items)


def test_paired_bootstrap_identical_and_shifted():
    rng = np.random.default_rng(1)
    b = rng.random(500)
    same = paired_bootstrap(b, b.copy(), n_boot=200)
    assert same["diff"] == 0 and same["diff_ci95"] == [0, 0] and same["share_of_resamples_a_not_better"] == 1
    shifted = paired_bootstrap(b + 0.1, b, n_boot=200)  # paired: every resample differs by exactly 0.1
    assert shifted["diff_ci95"] == [0.1, 0.1] and shifted["share_of_resamples_a_not_better"] == 0


def test_rankings_share_the_repeat_slots(store):  # noqa: F811
    ev = WeekEvaluator(store, WEEK)
    model = TwoTowerModel(ModelConfig.from_meta(store.meta, use_text=False)).eval()
    prior = ev.recent_popularity()
    ranked = ranked_lists(ev, model, torch.device("cpu"), prior, alpha=0.5, depth=TOP_K + 2)
    assert (ranked["popularity"][:, 0] == np.argmax(prior)).all()
    past = ev.repeat_candidates()
    tops = {r: merge_repeat(lists, past, 2) for r, lists in ranked.items()}
    n_own = len(past[0])  # customer 0 has 2 past purchases, customer 1 none
    assert n_own == 2
    assert (tops["popularity"][0, :2] == tops["blend"][0, :2]).all()
    assert (tops["two_tower"][0, :2] == past[0]).all()


def test_run_week_reports_every_ranking_and_history_bucket(store):  # noqa: F811
    ev = WeekEvaluator(store, WEEK)
    model = TwoTowerModel(ModelConfig.from_meta(store.meta, use_text=False)).eval()
    metrics, ap, ap_seen, n_past = run_week(ev, model, torch.device("cpu"), alpha=0.5, days=7)
    assert n_past.tolist() == [2, 0]
    assert len(metrics) == len(ap) == len(ap_seen) == 3 * 4
    # customer 0 bought only new items (1002, 1119), so only customer 2 is scored on seen items
    assert all(len(a) == 1 for a in ap_seen.values())
    assert all(0 <= m["new_item_recall@12"] <= m["new_item_recall@100"] <= 1 for m in metrics.values())
    buckets = by_history(ap, n_past, repeat=6)
    assert buckets["0"]["users"] == 1 and buckets["1-5"]["users"] == 1 and buckets["21+"]["users"] == 0
