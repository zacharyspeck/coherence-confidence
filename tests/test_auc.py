"""AUC is implemented as an explicit pairwise win rate so it can be audited by
hand. sklearn is the CHECK, not the implementation - that direction matters."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.analyze import _auc_fast, pairwise_auc, pairwise_auc_detail

TOL = 1e-9


def sklearn_auc(pos, neg) -> float:
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)]
    return float(roc_auc_score(y, s))


# ---- hand-computable cases -------------------------------------------------


def test_perfectly_separable_is_one():
    assert pairwise_auc([0.9, 0.8], [0.2, 0.1]) == 1.0


def test_perfectly_inverted_is_zero():
    assert pairwise_auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_all_ties_is_exactly_one_half():
    """The tie rule, isolated: every pair is a tie, every tie counts 0.5."""
    d = pairwise_auc_detail([0.5] * 4, [0.5] * 5)
    assert d.n_pairs == 20
    assert d.n_ties == 20
    assert d.n_wins == 0
    assert d.auc == 0.5


def test_hand_computed_mixed_case():
    """pos = [3, 1], neg = [2, 2].
    pairs: 3>2 win, 3>2 win, 1<2 loss, 1<2 loss -> 2 wins / 4 = 0.5"""
    d = pairwise_auc_detail([3.0, 1.0], [2.0, 2.0])
    assert (d.n_wins, d.n_ties, d.n_losses) == (2, 0, 2)
    assert d.auc == 0.5


def test_hand_computed_with_one_tie():
    """pos = [2, 4], neg = [2, 1, 5].
    2 vs 2 tie, 2 vs 1 win, 2 vs 5 loss, 4 vs 2 win, 4 vs 1 win, 4 vs 5 loss
    -> (3 + 0.5*1) / 6 = 0.5833..."""
    d = pairwise_auc_detail([2.0, 4.0], [2.0, 1.0, 5.0])
    assert (d.n_wins, d.n_ties, d.n_losses) == (3, 1, 2)
    assert d.auc == pytest.approx(3.5 / 6)


def test_twenty_by_twenty_is_four_hundred_pairs():
    """The brief's stated pair count for one condition."""
    d = pairwise_auc_detail(list(range(20)), list(range(100, 120)))
    assert d.n_pos == 20 and d.n_neg == 20
    assert d.n_pairs == 400


def test_five_false_above_all_true_gives_exactly_point_seven_five():
    """The mock's `known` scenario, checked on paper.
    15 of 20 true items beat all 20 false items -> 300/400 = 0.75."""
    pos = [0.05 + 0.001 * i for i in range(5)] + [
        0.80 + 0.001 * i for i in range(5, 20)
    ]
    neg = [0.30 + 0.001 * j for j in range(20)]
    d = pairwise_auc_detail(pos, neg)
    assert d.n_wins == 300
    assert d.n_ties == 0
    assert d.auc == 0.75


# ---- agreement with sklearn -----------------------------------------------


def test_matches_sklearn_on_hand_cases():
    for pos, neg in (
        ([0.9, 0.8], [0.2, 0.1]),
        ([3.0, 1.0], [2.0, 2.0]),
        ([2.0, 4.0], [2.0, 1.0, 5.0]),
        ([0.5] * 4, [0.5] * 5),
    ):
        assert abs(pairwise_auc(pos, neg) - sklearn_auc(pos, neg)) < TOL


@pytest.mark.parametrize("trial", range(30))
def test_matches_sklearn_on_random_continuous_scores(trial):
    rng = np.random.default_rng(1000 + trial)
    pos = rng.normal(0.5, 1.0, size=int(rng.integers(2, 40)))
    neg = rng.normal(0.0, 1.0, size=int(rng.integers(2, 40)))
    assert abs(pairwise_auc(pos, neg) - sklearn_auc(pos, neg)) < TOL


@pytest.mark.parametrize("trial", range(30))
def test_matches_sklearn_with_heavy_ties(trial):
    """Coarse grids force ties, which is where naive implementations diverge."""
    rng = np.random.default_rng(2000 + trial)
    pos = rng.integers(0, 4, size=20).astype(float)
    neg = rng.integers(0, 4, size=20).astype(float)
    assert abs(pairwise_auc(pos, neg) - sklearn_auc(pos, neg)) < TOL


def test_matches_sklearn_on_the_realistic_shape():
    """20 vs 20 on [0,1], the actual shape of one condition."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        pos = rng.random(20)
        neg = rng.random(20)
        assert abs(pairwise_auc(pos, neg) - sklearn_auc(pos, neg)) < TOL


# ---- the vectorized twin used inside the bootstrap ------------------------


@pytest.mark.parametrize("trial", range(30))
def test_fast_auc_matches_the_explicit_loop(trial):
    rng = np.random.default_rng(3000 + trial)
    pos = rng.integers(0, 5, size=int(rng.integers(2, 30))).astype(float)
    neg = rng.integers(0, 5, size=int(rng.integers(2, 30))).astype(float)
    assert abs(_auc_fast(pos, neg) - pairwise_auc(pos, neg)) < 1e-12


# ---- guardrails ------------------------------------------------------------


def test_raises_when_a_class_is_empty():
    with pytest.raises(ValueError, match="at least one item on each side"):
        pairwise_auc([1.0], [])
    with pytest.raises(ValueError, match="at least one item on each side"):
        pairwise_auc([], [1.0])


def test_detail_is_internally_consistent():
    rng = np.random.default_rng(11)
    d = pairwise_auc_detail(rng.random(13), rng.random(17))
    d.check()  # raises if wins+ties+losses != pairs, or auc disagrees
