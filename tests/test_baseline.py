"""The baseline is what separates 'the model was moved by the evidence' from
'the model already believed the claim'. If it silently mis-joins, every delta is
wrong in a way nothing downstream would notice."""

from __future__ import annotations

import json

import pytest
from conftest import make_family, make_item_set, make_records

from src.analyze import Dataset, analyze, attach_baseline
from src.baseline import score_baselines, unique_claims
from src.mock_scorer import MASS_COVERED, MockScorer
from src.models import Item
from src.render import render_baseline_prompt, render_prompt


# ---- one prompt per family, not per item ----------------------------------


def test_one_claim_per_family():
    items = make_item_set(20)
    rows = unique_claims(items)
    assert len(rows) == 20
    assert len(items) == 80
    assert [r[0] for r in rows] == sorted({i.family_id for i in items})


def test_rejects_a_family_whose_items_disagree_about_the_claim():
    fam = make_family("fam_split")
    d = fam.items[0].model_dump()
    d["claim"] = "a completely different claim about something else entirely"
    bad = Item.model_construct(**d)  # bypass Family-level validation on purpose
    with pytest.raises(ValueError, match="two different claims"):
        unique_claims([bad, *fam.items[1:]])


# ---- the prompt genuinely has no evidence in it ---------------------------


def test_baseline_prompt_contains_no_case_text():
    item = make_item_set(1)[0]
    p = render_baseline_prompt(item.claim)
    for c in item.cases:
        assert c.text not in p
    assert item.passage not in p
    assert item.claim in p


def test_baseline_prompt_differs_from_the_evidence_prompt():
    item = make_item_set(1)[0]
    assert render_baseline_prompt(item.claim) != render_prompt(item)


def test_baseline_prompt_still_ends_in_answer_colon():
    assert render_baseline_prompt("some claim about something").endswith("Answer:")


# ---- scoring ---------------------------------------------------------------


def test_score_baselines_returns_one_record_per_family():
    items = make_item_set(20)
    scorer = MockScorer().prepare(items)
    recs = score_baselines(scorer, items, progress=False)
    assert len(recs) == 20
    for r in recs:
        assert {"family_id", "claim", "domain", "p_yes_3way", "mass_covered"} <= set(r)


def test_mock_baseline_is_the_documented_constant():
    items = make_item_set(4)
    recs = score_baselines(MockScorer().prepare(items), items, progress=False)
    for r in recs:
        assert r["p_yes_3way"] == pytest.approx(0.5)
        assert r["mass_covered"] == pytest.approx(MASS_COVERED)


# ---- joining back onto the evidence run -----------------------------------


def test_attach_baseline_joins_on_family_id(tmp_path):
    items = make_item_set(20)
    scorer = MockScorer().prepare(items)
    records = make_records("known")
    baseline_recs = score_baselines(scorer, items, progress=False)

    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"kind": "baseline", "records": baseline_recs}), encoding="utf-8"
    )

    joined = attach_baseline(records, path)
    assert len(joined) == 80
    for r in joined:
        assert r["baseline_p_yes_3way"] == pytest.approx(0.5)


def test_attach_baseline_fails_loudly_on_a_missing_family(tmp_path):
    """A partial baseline must not silently produce deltas for some items."""
    items = make_item_set(20)
    recs = score_baselines(MockScorer().prepare(items), items, progress=False)
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"kind": "baseline", "records": recs[:5]}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="missing 15 families"):
        attach_baseline(make_records("known"), path)


def test_delta_scores_are_evidence_minus_baseline(tmp_path):
    items = make_item_set(20)
    recs = score_baselines(MockScorer().prepare(items), items, progress=False)
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"kind": "baseline", "records": recs}), encoding="utf-8"
    )
    joined = attach_baseline(make_records("known"), path)

    ds = Dataset(joined)
    assert ds.scores("delta") == pytest.approx(ds.scores("3way") - 0.5)

    a = analyze(joined, n_resamples=200, which="delta")
    # Subtracting a constant shifts every cell mean by that constant...
    assert a["cell_means_p_yes_3way"]["coherent_true"]["value"] == pytest.approx(
        0.6220 - 0.5
    )
    # ...and leaves contrasts and AUC untouched.
    assert a["auc_primary_full_set"]["conditions"]["coherent"]["estimate"]["value"] == pytest.approx(
        0.75
    )
    assert a["effects"]["main_effect_coherence"]["value"] == pytest.approx(0.075)


def test_analyze_refuses_delta_without_a_baseline():
    from src.analyze import main as analyze_main

    with pytest.raises(SystemExit, match="requires --baseline"):
        analyze_main(["--run", "nonexistent.json", "--out", "x.json", "--score", "delta"])
