"""ScoreResult is where the three-way / two-way distinction lives. If this is
wrong, every number in the repo is wrong and nothing else would notice."""

from __future__ import annotations

import math

import pytest

from src.render import ROLE_NO, ROLE_UNSURE, ROLE_YES
from src.scoring_types import ScoreResult


def make(y: float, n: float, u: float) -> ScoreResult:
    return ScoreResult(
        p_yes_raw=y, p_no_raw=n, p_unsure_raw=u, top_token=" x", top_token_prob=y
    )


def test_three_way_normalizes_over_all_three():
    r = make(0.2, 0.1, 0.1)
    assert r.mass_covered == pytest.approx(0.4)
    assert r.p_yes_3way == pytest.approx(0.5)
    assert r.p_no_3way == pytest.approx(0.25)
    assert r.p_unsure_3way == pytest.approx(0.25)
    assert r.p_yes_3way + r.p_no_3way + r.p_unsure_3way == pytest.approx(1.0)


def test_two_way_ignores_unsure_and_differs_from_three_way():
    r = make(0.2, 0.1, 0.1)
    assert r.p_yes_2way == pytest.approx(0.2 / 0.3)
    assert r.p_yes_2way != pytest.approx(r.p_yes_3way)


def test_two_way_is_nan_when_yes_and_no_are_both_zero():
    r = make(0.0, 0.0, 0.3)
    assert math.isnan(r.p_yes_2way)


def test_raw_masses_need_not_sum_to_one():
    """mass_covered is the whole point: the options are a slice of the vocab."""
    r = make(0.01, 0.02, 0.005)
    assert r.mass_covered == pytest.approx(0.035)
    assert r.p_yes_3way == pytest.approx(0.01 / 0.035)


def test_abstention_is_argmax_of_the_three():
    assert make(0.1, 0.2, 0.7).abstained is True
    assert make(0.1, 0.2, 0.7).argmax_role == ROLE_UNSURE
    assert make(0.7, 0.2, 0.1).abstained is False
    assert make(0.7, 0.2, 0.1).argmax_role == ROLE_YES
    assert make(0.2, 0.7, 0.1).argmax_role == ROLE_NO


def test_abstention_uses_only_the_three_options_not_the_whole_vocab():
    """A model whose overall argmax is some other token can still 'abstain'."""
    r = ScoreResult(
        p_yes_raw=0.001,
        p_no_raw=0.002,
        p_unsure_raw=0.004,
        top_token=" The",
        top_token_prob=0.9,
    )
    assert r.abstained is True
    assert r.mass_covered < 0.01


def test_rejects_non_probabilities():
    with pytest.raises(ValueError, match="not a probability"):
        make(1.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="not a probability"):
        make(-0.1, 0.5, 0.5)


def test_rejects_all_zero_mass():
    """All-zero means the option token ids are wrong, not that the model is shy."""
    with pytest.raises(ValueError, match="option token ids are almost certainly wrong"):
        make(0.0, 0.0, 0.0)


def test_to_dict_carries_every_derived_field():
    d = make(0.2, 0.1, 0.1).to_dict()
    for key in (
        "p_yes_raw",
        "p_no_raw",
        "p_unsure_raw",
        "mass_covered",
        "p_yes_3way",
        "p_no_3way",
        "p_unsure_3way",
        "p_yes_2way",
        "argmax_role",
        "abstained",
        "top_token",
        "top_token_prob",
    ):
        assert key in d, key


def test_ties_resolve_deterministically():
    """Exact ties must not flip between runs; max() takes the first."""
    r = make(0.3, 0.3, 0.3)
    assert r.argmax_role == ROLE_YES
    assert r.abstained is False
