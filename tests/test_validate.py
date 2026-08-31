"""Every validation gate is tested twice: once on items that should pass it, and
once on items deliberately broken in exactly the way the gate exists to catch.

A gate that has only ever been run on good data is not a gate.
"""

from __future__ import annotations

import pytest
from conftest import DIMS, make_item, make_item_set

from src.models import CELLS, Case, Item, compute_word_count
from src.validate import (
    ALL_CHECKS,
    check_cell_balance,
    check_coherent_structure,
    check_diverse_structure,
    check_duplicate_passages,
    check_lexical_giveaway,
    check_no_answer_leakage,
    check_review_status,
    check_word_counts,
    run_checks,
)

# Four closing sentences with no relationship to truth. Rotated across families
# so each one lands in every cell equally often - that is what makes the set
# lexically clean rather than merely varied.
NEUTRAL_CLOSERS = (
    "Readings were entered into the shared log the same afternoon.",
    "Two staff members independently transcribed each recorded figure.",
    "The measurement window opened at the start of the reporting period.",
    "Each site kept its own copy of the raw observation sheet.",
)


def build_clean_item(family_id: str, cell: str, k: int) -> Item:
    """An item whose only cell-dependent text is a closer drawn from a shared,
    truth-independent pool."""
    coherent = cell.startswith("coherent")
    n_distinct = 1 if coherent else 4
    cases = []
    for c in range(4):
        conditions = {d: f"{d}_v{(c % n_distinct) + 1}" for d in DIMS}
        joined = ", ".join(conditions[d] for d in sorted(DIMS))
        text = (
            f"Unit {k}{c} at {joined} was observed and the recorded outcome "
            f"moved by {11 + c} points."
        )
        cases.append(Case(case_id=f"c{c + 1}", text=text, conditions=conditions))

    # A token unique to this item and unrelated to its label, standing in for the
    # incidental wording every real passage carries. Without it the fixture has no
    # per-item vocabulary at all and even an in-sample fit lands at chance, which
    # would make the grouped-vs-in-sample comparison vacuous.
    ref = f"ref{k:02d}{CELLS.index(cell)}"
    closer = NEUTRAL_CLOSERS[(k + CELLS.index(cell)) % 4]
    lead = (
        f"Report {family_id} filed as {ref} covers four observed units under "
        "one protocol."
    )
    passage = "\n".join([lead] + [c.text for c in cases] + [closer])

    truth = cell.endswith("_true")
    return Item(
        id=f"{family_id}__{cell}",
        family_id=family_id,
        cell=cell,
        claim="the intervention increases the measured outcome",
        cases=cases,
        passage=passage,
        ground_truth=truth,
        confound_note=None if truth else "Held out from the passage on purpose.",
        flaw_type=(
            None
            if truth
            else ("shared_confound" if cell == "coherent_false" else "temporal")
        ),
        word_count=compute_word_count(passage),
        domain="synthetic",
    )


@pytest.fixture(scope="module")
def clean_items():
    return [
        build_clean_item(f"fam_c{k:02d}", cell, k)
        for k in range(20)
        for cell in CELLS
    ]


# ---- gate 1: word-count parity --------------------------------------------


def test_word_counts_pass_on_a_balanced_set(clean_items):
    r = check_word_counts(clean_items)
    assert r.passed, r.failures
    assert abs(r.data["by_cell"]["coherent_true"]["deviation_from_grand"]) < 0.10


def test_word_counts_fail_when_one_cell_runs_long():
    """FALSE cells naturally want an extra clause; that is exactly the drift
    this gate exists to stop (D-005)."""
    items = []
    for k in range(20):
        for cell in CELLS:
            pad = 40 if cell == "coherent_false" else 0
            items.append(make_item(f"fam_w{k:02d}", cell, passage_extra_words=pad))
    r = check_word_counts(items)
    assert not r.passed
    assert any("coherent_false" in f for f in r.failures)
    assert r.data["by_cell"]["coherent_false"]["deviation_from_grand"] > 0.10


def test_word_count_tolerance_is_configurable():
    items = []
    for k in range(20):
        for cell in CELLS:
            pad = 4 if cell == "diverse_true" else 0
            items.append(make_item(f"fam_t{k:02d}", cell, passage_extra_words=pad))
    assert check_word_counts(items, tolerance=0.001).passed is False
    assert check_word_counts(items, tolerance=0.99).passed is True


# ---- gate 2: coherent items have exactly one value per dimension ----------


def test_coherent_structure_passes_on_coherent_items(clean_items):
    r = check_coherent_structure(clean_items)
    assert r.passed, r.failures
    assert r.data["n_checked"] == 40


def test_coherent_structure_fails_when_a_dimension_varies():
    bad = make_item("fam_bad", "coherent_true", coherent=False, n_distinct=4)
    r = check_coherent_structure([bad])
    assert not r.passed
    assert len(r.failures) == 4  # one per dimension
    assert "must be exactly 1" in r.failures[0]


def test_coherent_structure_fails_on_even_one_varying_dimension():
    bad = make_item("fam_bad", "coherent_false", coherent=False, n_distinct=2)
    r = check_coherent_structure([bad])
    assert not r.passed
    assert "2 distinct values" in r.failures[0]


# ---- gate 3: diverse items have four values per dimension -----------------


def test_diverse_structure_passes_on_diverse_items(clean_items):
    r = check_diverse_structure(clean_items)
    assert r.passed, r.failures
    assert r.data["n_checked"] == 40


def test_diverse_structure_fails_when_values_repeat():
    bad = make_item("fam_bad", "diverse_true", coherent=True, n_distinct=1)
    r = check_diverse_structure([bad])
    assert not r.passed
    assert len(r.failures) == 4
    assert "must be 4" in r.failures[0]


def test_diverse_structure_fails_at_three_distinct_values():
    """Three is not four. The boundary is the whole point of the manipulation."""
    bad = make_item("fam_bad", "diverse_false", coherent=False, n_distinct=3)
    r = check_diverse_structure([bad])
    assert not r.passed
    assert "only 3 distinct values" in r.failures[0]


# ---- gate 4: lexical giveaway ---------------------------------------------


def test_lexical_check_passes_on_a_clean_set(clean_items):
    r = check_lexical_giveaway(clean_items)
    assert r.passed, (r.summary, r.failures)
    assert r.data["accuracy"] <= 0.60


def test_lexical_check_catches_a_planted_giveaway():
    """conftest's fixture uses one fixed closing sentence per cell, so 'supplier'
    appears in all 20 coherent_false items and nowhere else. That is a giveaway,
    and the gate must catch it even under family-grouped CV."""
    r = check_lexical_giveaway(make_item_set(20))
    assert not r.passed
    assert r.data["accuracy"] > 0.60
    assert "lexical giveaway" in r.failures[0]


def test_lexical_check_names_the_offending_tokens():
    r = check_lexical_giveaway(make_item_set(20))
    tokens = {t["token"] for t in r.data["top_tokens_predicting_FALSE"]}
    tokens |= {t["token"] for t in r.data["top_tokens_predicting_TRUE"]}
    # The planted words are 'supplier'/'switched' (false) and 'calibrated' (true).
    assert any("supplier" in t or "switched" in t or "calibrated" in t for t in tokens)


def test_lexical_check_reports_per_fold_accuracies(clean_items):
    r = check_lexical_giveaway(clean_items)
    assert len(r.data["fold_accuracies"]) == 5
    assert r.data["n_features"] > 100


def test_grouped_cv_is_stricter_than_in_sample(clean_items):
    """In-sample accuracy on 80 short passages with thousands of bigram features
    is ~100% by construction, which is why D-015 uses grouped CV instead."""
    insample = check_lexical_giveaway(clean_items, cv="insample")
    grouped = check_lexical_giveaway(clean_items, cv="grouped")
    assert insample.data["accuracy"] > grouped.data["accuracy"]
    assert insample.data["accuracy"] > 0.95


def test_lexical_threshold_is_configurable(clean_items):
    assert check_lexical_giveaway(clean_items, threshold=0.0).passed is False
    assert check_lexical_giveaway(clean_items, threshold=1.0).passed is True


def test_permutation_null_is_available(clean_items):
    r = check_lexical_giveaway(clean_items, n_permutations=25)
    perm = r.data["permutation_test"]
    assert perm["n_permutations"] == 25
    assert 0.0 <= perm["p_value"] <= 1.0
    assert 0.2 <= perm["null_mean"] <= 0.8


# ---- added gates (D-020) ---------------------------------------------------


def test_cell_balance_passes_at_twenty_per_cell(clean_items):
    r = check_cell_balance(clean_items)
    assert r.passed, r.failures
    assert r.data["n_families"] == 20


def test_cell_balance_fails_on_an_unbalanced_set(clean_items):
    r = check_cell_balance(clean_items[:-1])
    assert not r.passed
    assert any("diverse_true" in f or "does not have all 4 cells" in f
               for f in r.failures)


def test_duplicate_passages_pass_when_all_distinct(clean_items):
    r = check_duplicate_passages(clean_items)
    assert r.passed
    assert r.data["n_distinct"] == 80


def test_duplicate_passages_are_caught():
    """Two items with the same passage render to the same prompt and cannot be
    told apart by any measurement."""
    a = make_item("fam_x", "coherent_true")
    b = make_item("fam_y", "coherent_false")
    b = Item.model_validate(
        {
            **b.model_dump(),
            "cases": [c.model_dump() for c in a.cases],
            "passage": a.passage,
            "word_count": a.word_count,
        }
    )
    r = check_duplicate_passages([a, b])
    assert not r.passed
    assert "fam_x__coherent_true" in r.failures[0]


def test_answer_key_leakage_passes_when_the_note_is_held_out(clean_items):
    assert check_no_answer_leakage(clean_items).passed


def test_answer_key_leakage_is_caught():
    it = make_item("fam_leak", "coherent_false")
    leaked = Item.model_validate(
        {
            **it.model_dump(),
            "confound_note": "All four sites also switched supplier in the same week.",
        }
    )
    r = check_no_answer_leakage([leaked])
    assert not r.passed
    assert "confound_note appears verbatim" in r.failures[0]


def test_cell_name_in_passage_is_caught():
    it = make_item("fam_meta", "diverse_true")
    d = it.model_dump()
    d["passage"] = it.passage + "\nThis item is diverse_true."
    d["word_count"] = compute_word_count(d["passage"])
    r = check_no_answer_leakage([Item.model_validate(d)])
    assert not r.passed
    assert "metadata token" in r.failures[0]


def test_review_status_gate(clean_items):
    assert check_review_status(clean_items).passed
    d = clean_items[0].model_dump()
    d["review_status"] = "reviewed"
    r = check_review_status([Item.model_validate(d)])
    assert not r.passed
    assert "only a human may set that" in r.failures[0]


# ---- runner ----------------------------------------------------------------


def test_run_checks_covers_every_declared_check(clean_items):
    results = run_checks(clean_items)
    assert [r.name for r in results] == list(ALL_CHECKS)


def test_clean_set_passes_every_gate(clean_items):
    results = run_checks(clean_items)
    failed = [r.name for r in results if not r.passed]
    assert failed == [], failed


def test_skip_removes_a_check(clean_items):
    results = run_checks(clean_items, skip=["no_lexical_giveaway"])
    assert "no_lexical_giveaway" not in [r.name for r in results]


def test_brief_required_flag_is_set_correctly(clean_items):
    by_name = {r.name: r for r in run_checks(clean_items)}
    for name in (
        "word_count_parity",
        "coherent_one_value_per_dimension",
        "diverse_four_values_per_dimension",
        "no_lexical_giveaway",
    ):
        assert by_name[name].required_by_brief is True
    for name in ("cell_balance", "no_duplicate_passages", "no_answer_key_leakage"):
        assert by_name[name].required_by_brief is False


def test_check_results_serialize(clean_items):
    for r in run_checks(clean_items):
        d = r.to_dict()
        assert set(d) >= {"name", "passed", "summary", "failures", "data"}
