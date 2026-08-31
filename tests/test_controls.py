"""The two surface-feature controls (fix-pass step 6).

Case order and option position carry no evidence. If a result moves under either,
the measurement is picking up presentation. These tests check the controls do
what they say and, just as importantly, that they change nothing they shouldn't.
"""

from __future__ import annotations

import pytest
from conftest import make_item, make_item_set

from src.mock_scorer import MockScorer
from src.render import (
    DEFAULT_OPTIONS,
    OPTION_ROTATIONS,
    case_lines,
    case_permutation,
    passage_with_case_order,
    render_prompt,
    rotate,
)
from src.score import score_items


# ---- case order ------------------------------------------------------------


def test_case_lines_are_found_by_content_not_position(item):
    """Located by matching the stored case text, so the helper survives a change
    to the passage layout."""
    idx = case_lines(item)
    assert len(idx) == 4
    lines = item.passage.split("\n")
    for slot, case in zip(idx, item.cases):
        assert lines[slot].strip() == case.text.strip()


def test_permuting_cases_keeps_every_word(item):
    order = [2, 0, 3, 1]
    permuted = passage_with_case_order(item, order)
    assert permuted != item.passage
    assert sorted(permuted.split()) == sorted(item.passage.split())


def test_permuting_cases_leaves_the_non_case_lines_alone(item):
    slots = set(case_lines(item))
    before = item.passage.split("\n")
    after = passage_with_case_order(item, [3, 2, 1, 0]).split("\n")
    for i, (a, b) in enumerate(zip(before, after)):
        if i not in slots:
            assert a == b, f"line {i} changed but is not a case line"


def test_identity_permutation_is_a_no_op(item):
    assert passage_with_case_order(item, [0, 1, 2, 3]) == item.passage


def test_permutation_actually_reorders_the_cases(item):
    slots = case_lines(item)
    lines = item.passage.split("\n")
    out = passage_with_case_order(item, [1, 0, 3, 2]).split("\n")
    assert out[slots[0]] == lines[slots[1]]
    assert out[slots[1]] == lines[slots[0]]


def test_rejects_a_non_permutation(item):
    for bad in ([0, 1, 2], [0, 0, 1, 2], [0, 1, 2, 4]):
        with pytest.raises(ValueError, match="permutation of 0..3"):
            passage_with_case_order(item, bad)


def test_case_permutation_is_deterministic_per_item_and_seed():
    a = make_item("fam_p", "coherent_true")
    assert case_permutation(a, 0) == case_permutation(a, 0)
    assert sorted(case_permutation(a, 0)) == [0, 1, 2, 3]


def test_different_seeds_and_items_give_different_permutations():
    a = make_item("fam_p", "coherent_true")
    b = make_item("fam_q", "coherent_true")
    perms = {
        tuple(case_permutation(a, 0)),
        tuple(case_permutation(a, 1)),
        tuple(case_permutation(b, 0)),
    }
    assert len(perms) > 1


def test_shuffle_cases_records_the_permutation():
    items = make_item_set(4)
    scorer = MockScorer().prepare(items)
    recs = score_items(scorer, items, progress=False, shuffle_cases=True, case_seed=7)
    assert all(sorted(r["case_order"]) == [0, 1, 2, 3] for r in recs)
    assert any(r["case_order"] != [0, 1, 2, 3] for r in recs)


def test_case_order_is_none_when_the_control_is_off():
    items = make_item_set(2)
    recs = score_items(MockScorer().prepare(items), items, progress=False)
    assert all(r["case_order"] is None for r in recs)


# ---- option position -------------------------------------------------------


def test_there_are_exactly_three_rotations():
    assert len(OPTION_ROTATIONS) == 3
    assert OPTION_ROTATIONS[0] == (0, 1, 2)
    for r in OPTION_ROTATIONS:
        assert sorted(r) == [0, 1, 2]


def test_rotation_changes_display_order_only(item):
    lines = {
        rot: [
            ln
            for ln in render_prompt(item, option_rotation=rot).split("\n")
            if ln.startswith("Options:")
        ][0]
        for rot in OPTION_ROTATIONS
    }
    assert len(set(lines.values())) == 3
    for rot, ln in lines.items():
        assert ln == "Options: " + " / ".join(rotate(DEFAULT_OPTIONS, rot))


def test_rotation_does_not_touch_the_role_mapping(item):
    """The instruction line always says Yes-means-establishes; only the list
    order moves. A model answering the question rather than picking a position
    should be unaffected."""
    for rot in OPTION_ROTATIONS:
        p = render_prompt(item, option_rotation=rot)
        assert "Answer Yes if the evidence establishes the claim" in p


def test_rotation_leaves_the_passage_alone(item):
    for rot in OPTION_ROTATIONS:
        assert item.passage in render_prompt(item, option_rotation=rot)


def test_option_rotations_report_mean_and_spread():
    items = make_item_set(4)
    scorer = MockScorer().prepare(items)
    recs = score_items(scorer, items, progress=False, option_rotations=True)
    for r in recs:
        assert len(r["rotation_p_yes_3way"]) == 3
        assert r["p_yes_3way_rotation_spread"] == pytest.approx(
            max(r["rotation_p_yes_3way"]) - min(r["rotation_p_yes_3way"])
        )
        assert r["p_yes_3way_rotation_mean"] == pytest.approx(
            sum(r["rotation_p_yes_3way"]) / 3
        )


def test_the_mock_has_zero_position_bias_by_construction():
    """The mock answers per item, so its spread must be exactly zero. If this
    ever moves, the rotation plumbing is leaking into the lookup."""
    items = make_item_set(4)
    recs = score_items(
        MockScorer().prepare(items), items, progress=False, option_rotations=True
    )
    assert all(r["p_yes_3way_rotation_spread"] == 0.0 for r in recs)
    assert not any(r["rotation_spread_flagged"] for r in recs)


def test_items_over_the_spread_limit_are_flagged():
    items = make_item_set(2)
    recs = score_items(
        MockScorer().prepare(items),
        items,
        progress=False,
        option_rotations=True,
        rotation_spread_limit=-1.0,  # force every item over the limit
    )
    assert all(r["rotation_spread_flagged"] for r in recs)


def test_headline_record_is_the_first_rotation():
    """So a run with rotations on and one with them off agree on the primary
    number, and the average is an addition rather than a silent replacement."""
    items = make_item_set(4)
    plain = score_items(MockScorer().prepare(items), items, progress=False)
    rotated = score_items(
        MockScorer().prepare(items), items, progress=False, option_rotations=True
    )
    for a, b in zip(plain, rotated):
        assert a["p_yes_3way"] == b["p_yes_3way"]


def test_both_controls_together_still_produce_one_record_per_item():
    items = make_item_set(5)
    recs = score_items(
        MockScorer().prepare(items),
        items,
        progress=False,
        shuffle_cases=True,
        option_rotations=True,
    )
    assert len(recs) == len(items)
    assert [r["item_id"] for r in recs] == [i.id for i in items]
