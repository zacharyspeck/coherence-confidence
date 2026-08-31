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
    check_flaw_declarations,
    check_matched_mechanism_subset,
    check_passage_word_balance,
    closer_of,
    flaw_line_of,
    check_coherent_structure,
    check_diverse_structure,
    check_duplicate_passages,
    check_lexical_giveaway,
    check_no_answer_leakage,
    check_review_status,
    check_word_counts,
    run_checks,
)

# The D-022 balanced closer scheme: FACT x SCOPE, plus a shared tail. Each clause
# variant lands in exactly one TRUE and one FALSE item, so no word in the closers
# correlates with the label.
FACT = {
    "changed": "Ambient load rose sharply against the earlier period",
    "same": "Ambient load matched the earlier period closely",
}
SCOPE = {
    "reach": "and every unit stood on the open floor",
    "block": "and every unit stood inside a sealed cabinet",
}
TAIL = "no other adjustment was made to any unit."

# Per-family vocabulary, because the real item set has one domain per family.
# A single generic wording repeated across all 20 families is learnable across a
# grouped-CV fold boundary in a way the real set is not, and a fixture that is
# easier to crack than the thing it stands in for tests nothing.
POPULATION_SETS = [
    ("Counts cover the units that reported", "Counts cover the units on the roll"),
    ("Totals span the sites that filed", "Totals span the sites on the register"),
    ("Figures include the lines that answered", "Figures include the lines in the index"),
    ("Readings take in the cells that responded", "Readings take in the cells on record"),
    ("Tallies run over the plots that returned", "Tallies run over the plots in the plan"),
]
COMPLETENESS_SETS = [
    ("and each listed unit reported", "and a quarter of listed units stayed silent"),
    ("and each named site filed", "and a quarter of named sites stayed silent"),
    ("and each indexed line answered", "and a quarter of indexed lines stayed silent"),
    ("and each recorded cell responded", "and a quarter of recorded cells stayed silent"),
    ("and each planned plot returned", "and a quarter of planned plots stayed silent"),
]

# D-024: half the families are scope families, where BOTH false items are
# falsified by scope_mismatch. That is the matched-mechanism subset.
ASSIGN = {
    False: {
        "coherent_false": (("changed", "reach"), ("whole", "complete"), "stated_confound"),
        "coherent_true": (("changed", "block"), ("subset", "complete"), None),
        "diverse_true": (("same", "reach"), ("whole", "incomplete"), None),
        "diverse_false": (("same", "block"), ("whole", "complete"), "broken_chronology"),
    },
    True: {
        "coherent_false": (("changed", "block"), ("subset", "incomplete"), "scope_mismatch"),
        "diverse_false": (("same", "reach"), ("subset", "incomplete"), "scope_mismatch"),
        "coherent_true": (("same", "reach"), ("subset", "complete"), None),
        "diverse_true": (("changed", "block"), ("whole", "incomplete"), None),
    },
}


def build_clean_item(
    family_id: str, cell: str, k: int, scope_family: bool = False
) -> Item:
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
    assign = dict(ASSIGN[scope_family])
    if k % 4 >= 2:  # rotate the TRUE items' scope clauses (D-026)
        ct, dt = assign['coherent_true'], assign['diverse_true']
        assign['coherent_true'] = (ct[0], dt[1], ct[2])
        assign['diverse_true'] = (dt[0], ct[1], dt[2])
    (fact_key, scope_key), (pop_key, comp_key), _mech = assign[cell]
    pop = POPULATION_SETS[k % len(POPULATION_SETS)][0 if pop_key == "subset" else 1]
    comp = COMPLETENESS_SETS[k % len(COMPLETENESS_SETS)][
        0 if comp_key == "complete" else 1
    ]
    midline = f"{FACT[fact_key]}, {SCOPE[scope_key]}. {pop}, {comp}."
    lead = (
        f"Report {family_id} filed as {ref} covers four observed units under "
        "one protocol."
    )
    # 7-line layout: the flaw sits mid-passage with two cases after it.
    passage = "\n".join(
        [lead, cases[0].text, cases[1].text, midline,
         cases[2].text, cases[3].text, TAIL]
    )

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
        flaw_mechanism=ASSIGN[scope_family][cell][2],
        confound_variant="_".join(ASSIGN[scope_family][cell][0]),
        scope_variant="_".join(ASSIGN[scope_family][cell][1]),
        word_count=compute_word_count(passage),
        domain="synthetic",
    )


@pytest.fixture(scope="module")
def clean_items():
    return [
        build_clean_item(f"fam_c{k:02d}", cell, k, scope_family=(k % 2 == 1))
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
    assert len(r.data["fold_accuracies_last_seed"]) == 5
    assert r.data["n_features"] > 100


def test_lexical_check_averages_over_several_cv_shufflings(clean_items):
    """One shuffle swings the accuracy by several points on 80 items, so a
    single seed can pass or fail the same item set by luck. This actually
    happened: the first full draft passed at 58.8% on seed 0 and failed on 7 of
    10 seeds, mean 62.5%."""
    r = check_lexical_giveaway(clean_items, n_seeds=5)
    assert r.data["n_seeds"] == 5
    assert len(r.data["accuracy_per_seed"]) == 5
    assert r.data["accuracy"] == pytest.approx(
        sum(r.data["accuracy_per_seed"]) / 5, abs=1e-6
    )
    assert r.data["accuracy_max"] == max(r.data["accuracy_per_seed"])


def test_lexical_check_reports_how_many_seeds_breached_the_limit(clean_items):
    """A marginal pass has to be visible rather than silent."""
    r = check_lexical_giveaway(clean_items, threshold=0.0, n_seeds=4)
    assert r.data["n_seeds_over_threshold"] == 4
    assert "4/4 over limit" in r.summary
    r = check_lexical_giveaway(clean_items, threshold=1.0, n_seeds=4)
    assert r.data["n_seeds_over_threshold"] == 0


def test_lexical_check_fails_on_the_mean_not_a_lucky_seed():
    """Constructed so seed-to-seed variation cannot rescue a real giveaway."""
    r = check_lexical_giveaway(make_item_set(20), n_seeds=5)
    assert not r.passed
    assert r.data["accuracy"] > 0.60
    assert r.data["n_seeds_over_threshold"] >= 3


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


# ---- the passage-balance gate (D-022, D-026) -------------------------------


def test_closer_is_the_last_line_and_flaw_line_is_the_middle(clean_items):
    it = clean_items[0]
    assert closer_of(it) == it.passage.splitlines()[-1]
    assert closer_of(it) == TAIL
    assert flaw_line_of(it) == it.passage.splitlines()[3]
    # The flaw is buried: two cases and a closer come after it.
    assert len(it.passage.splitlines()) == 7


def test_balanced_passages_pass(clean_items):
    """Every word appears about as often on the TRUE side as the FALSE side."""
    r = check_passage_word_balance(clean_items)
    assert r.passed, r.failures


def test_neither_family_type_can_reach_perfect_balance(clean_items):
    """D-026, measured rather than assumed.

    Both types bottom out at 1 and neither reaches 0. For a scope family the
    reason is structural: its two false items share a mechanism, so the
    falsifying clause is 2F against at most 1T. For a confound family the
    residual here is only the per-item unique reference token, which is
    label-independent by construction. The gate's tolerance is 2 because the real
    items, whose clause variants carry more incidental word overlap than this
    fixture's, measure 2."""
    for label, sel in (("confound", 0), ("scope", 1)):
        grp = [i for i in clean_items if int(i.family_id[-2:]) % 2 == sel]
        assert check_passage_word_balance(grp, tolerance=1).passed, label
        assert not check_passage_word_balance(grp, tolerance=0).passed, label


def test_gate_catches_a_word_planted_on_one_side():
    """The failure mode this exists for: a word that only ever appears in FALSE
    items. That is what a lexical giveaway looks like at the family level."""
    items = []
    for cell in CELLS:
        it = build_clean_item("fam_q0", cell, 0)
        if not it.ground_truth:
            d = it.model_dump()
            d["passage"] = it.passage + " Contaminant contaminant contaminant."
            d["word_count"] = compute_word_count(d["passage"])
            it = Item.model_validate(d)
        items.append(it)
    r = check_passage_word_balance(items)
    assert not r.passed
    assert "fam_q0" in r.failures[0]
    assert "contaminant" in r.failures[0]
    assert "T/" in r.failures[0] and "F" in r.failures[0]


def test_balance_is_checked_per_family_not_across_the_set():
    """Two families whose imbalances cancel globally must still both fail."""
    a = [build_clean_item("fam_p0", c, 0) for c in CELLS]
    b = [build_clean_item("fam_p1", c, 1) for c in CELLS]

    def bend(items, cell, extra):
        out = []
        for it in items:
            if it.cell == cell:
                d = it.model_dump()
                d["passage"] = it.passage + " " + extra
                d["word_count"] = compute_word_count(d["passage"])
                out.append(Item.model_validate(d))
            else:
                out.append(it)
        return out

    a = bend(a, "coherent_true", "Marker marker marker marker.")
    b = bend(b, "coherent_false", "Marker marker marker marker.")
    r = check_passage_word_balance(a + b)
    assert not r.passed
    assert len(r.failures) == 2


# ---- flaw declarations -----------------------------------------------------


def test_flaw_declarations_pass_on_a_complete_set(clean_items):
    assert check_flaw_declarations(clean_items).passed


def test_missing_variant_is_caught():
    it = build_clean_item("fam_z", "coherent_false", 0)
    d = it.model_dump()
    d["scope_variant"] = None
    r = check_flaw_declarations([Item.model_validate(d)])
    assert not r.passed
    assert "must declare both" in r.failures[0]


def test_an_item_false_twice_over_is_caught():
    """Both live combinations at once means the mechanism label is not what a
    reader would actually find."""
    it = build_clean_item("fam_z", "coherent_false", 0, scope_family=True)
    d = it.model_dump()
    d["confound_variant"] = "changed_reach"
    d["flaw_mechanism"] = "scope_mismatch"
    # bypass the model's own cross-check to exercise the gate
    broken = Item.model_construct(**d)
    r = check_flaw_declarations([broken])
    assert not r.passed
    assert "false twice over" in r.failures[0]


# ---- the matched-mechanism subset ------------------------------------------


def test_matched_subset_gate_passes_at_ten_per_cell(clean_items):
    r = check_matched_mechanism_subset(clean_items)
    assert r.passed, r.failures
    assert r.data["counts_by_cell"]["coherent_false"]["scope_mismatch"] == 10
    assert r.data["counts_by_cell"]["diverse_false"]["scope_mismatch"] == 10


def test_matched_subset_gate_fails_below_the_minimum(clean_items):
    assert not check_matched_mechanism_subset(clean_items, minimum=11).passed


def test_matched_subset_gate_fails_when_families_diverge(clean_items):
    """Different families per condition would contrast scenarios, not coherence."""
    out = []
    for it in clean_items:
        if it.id == "fam_c01__diverse_false":
            d = it.model_dump()
            d["flaw_mechanism"] = "broken_chronology"
            d["scope_variant"] = "whole_incomplete"
            out.append(Item.model_validate(d))
        else:
            out.append(it)
    r = check_matched_mechanism_subset(out)
    assert not r.passed
    assert "same families" in r.failures[-1]
