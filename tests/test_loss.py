import math

import pytest
import torch

from src.loss import in_batch_softmax_loss


def test_matches_manual_cross_entropy_without_corrections():
    u = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    i = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    loss, _ = in_batch_softmax_loss(u, i, torch.tensor([0, 1]), torch.zeros(2), 1.0,
                                    use_logq=False, mask_accidental_hits=False)
    # Row 0 logits [1.0, 0.6] label 0; row 1 logits [0.0, 0.8] label 1.
    expected = (-math.log(math.e / (math.e + math.e**0.6)) - math.log(math.e**0.8 / (1 + math.e**0.8))) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_logq_subtracts_log_frequency_per_column():
    u = torch.eye(2)
    i = torch.eye(2)
    log_q = torch.log(torch.tensor([0.9, 0.1]))
    loss, _ = in_batch_softmax_loss(u, i, torch.tensor([0, 1]), log_q, 1.0, mask_accidental_hits=False)
    logits = torch.eye(2) - log_q[None, :]
    expected = torch.nn.functional.cross_entropy(logits, torch.arange(2))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)


def test_logq_reduces_penalty_on_popular_negatives():
    # Popular item 0 is a negative for user 1; with logQ its logit is pushed down, so the loss is lower.
    u = torch.tensor([[1.0, 0.0], [0.7, 0.7]])
    i = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    log_q = torch.log(torch.tensor([0.99, 0.01]))
    rows = torch.tensor([0, 1])
    without, _ = in_batch_softmax_loss(u, i, rows, log_q, 0.1, use_logq=False, mask_accidental_hits=False)
    with_logq, _ = in_batch_softmax_loss(u, i, rows, log_q, 0.1, use_logq=True, mask_accidental_hits=False)
    assert with_logq < without


def test_accidental_hits_are_masked_but_positive_kept():
    u = torch.randn(3, 4)
    i = torch.randn(3, 4)
    rows = torch.tensor([7, 7, 9])  # rows 0 and 1 hold the same article
    loss, stats = in_batch_softmax_loss(u, i, rows, torch.zeros(3), 1.0, use_logq=False)
    logits = u @ i.T
    logits[0, 1] = logits[1, 0] = float("-inf")
    expected = torch.nn.functional.cross_entropy(logits, torch.arange(3))
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert stats["accidental_hits_per_row"] == pytest.approx(2 / 3)


def test_perfect_separation_gives_near_zero_loss():
    u = torch.eye(4)
    loss, stats = in_batch_softmax_loss(u, u.clone(), torch.arange(4), torch.zeros(4), 0.01, use_logq=False)
    assert loss.item() < 1e-6
    assert stats["in_batch_accuracy"] == 1.0
