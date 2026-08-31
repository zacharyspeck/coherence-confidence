"""Analysis correctness, checked against numbers derived on paper.

Every expected value in this file comes from the `known` mock construction
documented at the top of src/mock_scorer.py, not from running the code and
copying what it printed.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_records

from src.analyze import (
    Dataset,
    analyze,
    bootstrap,
    pairwise_auc_detail,
    stat_abstention_rate,
    stat_auc,
    stat_cell_mean,
    stat_main_effect_coherence,
    stat_main_effect_truth,
    stat_interaction,
    two_way_anova,
)
from src.mock_scorer import EXPECTED_AUC
from src.models import CELLS

# Derived by hand from the `known` construction. See src/mock_scorer.py.
#   coherent_true = (5*0.050 + 0.001*(0+1+2+3+4) + 15*0.800 + 0.001*(5+..+19)) / 20
#                 = (0.25 + 0.010 + 12.00 + 0.180) / 20 = 0.6220
#   diverse_true  = (9*0.050 + 0.001*36        + 11*0.800 + 0.001*154)        / 20
#                 = (0.45 + 0.036 + 8.80 + 0.154) / 20   = 0.4720
#   *_false       = (20*0.300 + 0.001*(0+..+19)) / 20 = (6.00 + 0.19) / 20 = 0.3095
EXPECTED_CELL_MEANS = {
    "coherent_true": 0.6220,
    "coherent_false": 0.3095,
    "diverse_true": 0.4720,
    "diverse_false": 0.3095,
}
EXPECTED_ABSTENTION = {
    "coherent_true": 5 / 20,
    "coherent_false": 0.0,
    "diverse_true": 9 / 20,
    "diverse_false": 0.0,
}

FAST = 300  # resamples for tests that only need the machinery to run


@pytest.fixture(scope="module")
def known():
    return make_records("known")


@pytest.fixture(scope="module")
def ds(known):
    return Dataset(known)


# ---- shape -----------------------------------------------------------------


def test_eighty_items_twenty_per_cell(ds):
    assert len(ds) == 80
    assert len(ds.families) == 20
    for c in CELLS:
        assert ds.idx_by_cell[c].size == 20


def test_each_condition_gives_four_hundred_pairs(ds):
    for coh in ("coherent", "diverse"):
        sel = np.flatnonzero(ds.coherence == coh)
        pos = ds.score[sel[ds.truth[sel]]]
        neg = ds.score[sel[~ds.truth[sel]]]
        d = pairwise_auc_detail(pos, neg)
        assert (d.n_pos, d.n_neg, d.n_pairs) == (20, 20, 400)


# ---- the hand-computed values ---------------------------------------------


def test_cell_means_match_the_hand_computed_values(ds):
    idx = np.arange(len(ds))
    for cell, expected in EXPECTED_CELL_MEANS.items():
        got = stat_cell_mean(ds, cell)(idx)
        assert got == pytest.approx(expected, abs=1e-12), cell


def test_within_condition_auc_matches_the_hand_computed_values(ds):
    idx = np.arange(len(ds))
    assert stat_auc(ds, "coherent")(idx) == pytest.approx(0.75, abs=1e-12)
    assert stat_auc(ds, "diverse")(idx) == pytest.approx(0.55, abs=1e-12)
    assert stat_auc(ds, "coherent")(idx) == EXPECTED_AUC["coherent"]
    assert stat_auc(ds, "diverse")(idx) == EXPECTED_AUC["diverse"]


def test_abstention_rates_match_the_hand_computed_values(ds):
    idx = np.arange(len(ds))
    for cell, expected in EXPECTED_ABSTENTION.items():
        assert stat_abstention_rate(ds, cell)(idx) == pytest.approx(expected)


def test_main_effects_match_the_cell_means(ds):
    idx = np.arange(len(ds))
    e = EXPECTED_CELL_MEANS
    expected_coh = (e["coherent_true"] + e["coherent_false"]) / 2 - (
        e["diverse_true"] + e["diverse_false"]
    ) / 2
    expected_truth = (e["coherent_true"] + e["diverse_true"]) / 2 - (
        e["coherent_false"] + e["diverse_false"]
    ) / 2
    assert stat_main_effect_coherence(ds)(idx) == pytest.approx(expected_coh, abs=1e-12)
    assert stat_main_effect_truth(ds)(idx) == pytest.approx(expected_truth, abs=1e-12)


# ---- the abstention policy (D-003) ----------------------------------------


def test_abstained_items_are_included_in_auc_by_default(ds):
    """This is the policy the brief made explicit. It has to be tested, not
    just documented: the whole point is that the default cannot drift."""
    idx = np.arange(len(ds))
    included = stat_auc(ds, "coherent")(idx)
    assert included == pytest.approx(0.75)
    assert np.count_nonzero(ds.abstained[ds.coherence == "coherent"]) == 5


def test_dropping_abstentions_would_inflate_auc_to_one(ds):
    """The mock abstains on exactly the 5 TRUE items the model scored lowest -
    the ones it got wrong. Dropping them removes every losing pair."""
    idx = np.arange(len(ds))
    dropped = stat_auc(ds, "coherent", drop_abstained=True)(idx)
    assert dropped == pytest.approx(1.0)
    assert dropped > stat_auc(ds, "coherent")(idx)


def test_analysis_reports_the_drop_counterfactual_as_a_diagnostic(known):
    a = analyze(known, n_resamples=FAST)
    coh = a["auc_within_condition"]["coherent"]
    assert coh["estimate"]["value"] == pytest.approx(0.75)
    assert coh["n_abstained"] == 5
    assert coh["auc_if_abstained_dropped_DIAGNOSTIC"] == pytest.approx(1.0)
    assert "D-003" in a["policy"]["abstained_items_in_auc"]
    assert "INCLUDED" in a["policy"]["abstained_items_in_auc"]


# ---- AUC is never pooled ---------------------------------------------------


def test_auc_is_reported_per_condition_not_pooled(known):
    a = analyze(known, n_resamples=FAST)
    assert set(a["auc_within_condition"]) == {"coherent", "diverse"}
    assert "pooled" not in a["auc_within_condition"]
    assert a["auc_within_condition"]["coherent"]["estimate"]["value"] != pytest.approx(
        a["auc_within_condition"]["diverse"]["estimate"]["value"]
    )


def test_pooling_would_have_given_a_different_answer(ds):
    """Shows why the split matters: pooled AUC is not either within-AUC."""
    pooled = pairwise_auc_detail(ds.score[ds.truth], ds.score[~ds.truth]).auc
    assert pooled != pytest.approx(0.75)
    assert pooled != pytest.approx(0.55)


# ---- bootstrap -------------------------------------------------------------


def test_bootstrap_returns_both_resampling_units(ds):
    est = bootstrap(ds, stat_cell_mean(ds, "coherent_true"), n_resamples=FAST)
    assert est.ci_item is not None and est.ci_family is not None
    lo, hi = est.ci_item
    assert lo <= est.value <= hi
    flo, fhi = est.ci_family
    assert flo <= est.value <= fhi


def test_bootstrap_is_deterministic_given_the_seed(ds):
    a = bootstrap(ds, stat_cell_mean(ds, "coherent_true"), n_resamples=FAST, seed=7)
    b = bootstrap(ds, stat_cell_mean(ds, "coherent_true"), n_resamples=FAST, seed=7)
    assert a.ci_item == b.ci_item
    assert a.ci_family == b.ci_family


def test_different_seeds_give_different_intervals(ds):
    a = bootstrap(ds, stat_cell_mean(ds, "coherent_true"), n_resamples=FAST, seed=1)
    b = bootstrap(ds, stat_cell_mean(ds, "coherent_true"), n_resamples=FAST, seed=2)
    assert a.ci_item != b.ci_item


def test_per_cell_statistics_give_identical_item_and_family_intervals(ds):
    """A structural identity of a fully crossed design, not a coincidence.

    Each family contributes exactly one item per cell, so resampling 20 families
    and taking their coherent_true items IS resampling 20 coherent_true items.
    If this ever stops holding, the design has stopped being fully crossed.
    """
    for stat in (stat_cell_mean(ds, "coherent_true"), stat_auc(ds, "coherent")):
        est = bootstrap(ds, stat, n_resamples=4000)
        item_w = est.ci_item[1] - est.ci_item[0]
        fam_w = est.ci_family[1] - est.ci_family[0]
        # Identical resampling DISTRIBUTIONS; the two draw sequences differ, so
        # the realized percentiles agree only to Monte Carlo error.
        assert fam_w == pytest.approx(item_w, rel=0.02)


def test_cross_cell_contrasts_gain_from_the_family_clustering(ds):
    """The 2x2 is within-family, so a clustered resample keeps the pairing and
    the contrast CI gets NARROWER - the paired-vs-unpaired variance reduction.

    This is the opposite of the usual clustering intuition and it is why D-010
    says to read ci_family for the coherence effect and the interaction.
    """
    for stat in (stat_main_effect_coherence(ds), stat_interaction(ds)):
        est = bootstrap(ds, stat, n_resamples=2000)
        item_w = est.ci_item[1] - est.ci_item[0]
        fam_w = est.ci_family[1] - est.ci_family[0]
        assert fam_w < item_w


def test_default_is_ten_thousand_resamples():
    from src.analyze import N_RESAMPLES

    assert N_RESAMPLES == 10_000


def test_resample_items_preserves_cell_sizes(ds):
    rng = np.random.default_rng(0)
    idx = ds.resample_items(rng)
    assert idx.size == len(ds)
    for c in CELLS:
        assert np.count_nonzero(ds.cells[idx] == c) == 20


def test_resample_families_keeps_families_intact(ds):
    rng = np.random.default_rng(0)
    idx = ds.resample_families(rng)
    assert idx.size == len(ds)
    # every drawn family contributes all 4 of its cells
    fams, counts = np.unique(ds.family_ids[idx], return_counts=True)
    assert set((counts % 4).tolist()) == {0}


# ---- every reported number carries a CI ------------------------------------


def test_every_reported_number_has_a_confidence_interval(known):
    a = analyze(known, n_resamples=FAST)
    blocks = [
        a["cell_means_p_yes_3way"],
        a["cell_means_p_yes_2way"],
        a["abstention_rate"],
        a["effects"],
    ]
    for block in blocks:
        for name, d in block.items():
            assert d["ci_item"] is not None, name
            assert d["ci_family"] is not None, name
            assert len(d["ci_item"]) == 2, name
    for coh, e in a["auc_within_condition"].items():
        assert e["estimate"]["ci_item"] is not None, coh
        assert e["estimate"]["ci_family"] is not None, coh


# ---- 2x2 breakdown ---------------------------------------------------------


def test_anova_has_all_four_sources(ds):
    t = two_way_anova(ds)
    sources = [r["source"] for r in t["table"]]
    assert sources == ["coherence", "truth", "coherence x truth", "residual"]
    assert t["balanced"] is True
    assert t["table"][-1]["df"] == 76  # 80 - 4


def test_anova_sums_of_squares_decompose_the_total(ds):
    t = two_way_anova(ds)
    total = float(np.sum((ds.score - np.mean(ds.score)) ** 2))
    parts = sum(r["ss"] for r in t["table"])
    assert parts == pytest.approx(total, rel=1e-9)


def test_anova_marginal_means_agree_with_cell_means(ds):
    t = two_way_anova(ds)
    cm = t["cell_means"]
    assert t["marginal_means"]["coherent"] == pytest.approx(
        (cm["coherent_true"] + cm["coherent_false"]) / 2
    )
    assert t["marginal_means"]["true"] == pytest.approx(
        (cm["coherent_true"] + cm["diverse_true"]) / 2
    )


# ---- the scenarios that simulate each possible world -----------------------


def test_coherence_driven_world_shows_a_coherence_effect_and_chance_auc():
    """The hypothesis, simulated: confidence tracks coherence, ignores truth.
    Every within-condition pair is an exact tie, so AUC is exactly 0.5."""
    a = analyze(make_records("coherence_driven"), n_resamples=FAST)
    assert a["effects"]["main_effect_coherence"]["value"] == pytest.approx(0.40)
    assert a["effects"]["main_effect_truth"]["value"] == pytest.approx(0.0, abs=1e-12)
    for coh in ("coherent", "diverse"):
        e = a["auc_within_condition"][coh]
        assert e["estimate"]["value"] == pytest.approx(0.5)
        assert e["detail"]["n_ties"] == 400


def test_truth_driven_world_shows_perfect_auc_and_no_coherence_effect():
    """The well-calibrated null."""
    a = analyze(make_records("truth_driven"), n_resamples=FAST)
    assert a["effects"]["main_effect_truth"]["value"] == pytest.approx(0.70)
    assert a["effects"]["main_effect_coherence"]["value"] == pytest.approx(0.0, abs=1e-12)
    for coh in ("coherent", "diverse"):
        assert a["auc_within_condition"][coh]["estimate"]["value"] == pytest.approx(1.0)


def test_ties_scenario_gives_exactly_one_half_everywhere():
    a = analyze(make_records("ties"), n_resamples=FAST)
    for coh in ("coherent", "diverse"):
        assert a["auc_within_condition"][coh]["estimate"]["value"] == 0.5
    assert a["effects"]["interaction"]["value"] == pytest.approx(0.0, abs=1e-12)


# ---- the clean vs confounded coherence contrast (D-004) --------------------


def test_reports_the_clean_within_true_coherence_contrast(known):
    a = analyze(known, n_resamples=FAST)
    e = a["effects"]["coherence_effect_within_true"]
    assert e["value"] == pytest.approx(
        EXPECTED_CELL_MEANS["coherent_true"] - EXPECTED_CELL_MEANS["diverse_true"],
        abs=1e-12,
    )
    assert "D-004" in e["note"]
    assert "CONFOUNDED" in a["effects"]["coherence_effect_within_false"]["note"]


# ---- two-way vs three-way --------------------------------------------------


def test_two_way_and_three_way_are_reported_separately(known):
    a = analyze(known, n_resamples=FAST)
    for c in CELLS:
        three = a["cell_means_p_yes_3way"][c]["value"]
        two = a["cell_means_p_yes_2way"][c]["value"]
        assert three != pytest.approx(two), c
    assert a["policy"]["primary_measure"].startswith("p_yes_3way")
