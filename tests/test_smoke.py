"""The step-6 smoke test, as a pytest case.

`src/smoke.py` already asserts every headline number against arithmetic done on
paper, so the job here is to make sure it runs in CI and that its expectations
are themselves derived rather than pasted.
"""

from __future__ import annotations

import json

import pytest

from src.models import CELLS, load_items
from src.smoke import EXPECTED, expected_auc, expected_false_mean, expected_true_mean
from src.smoke import run as run_smoke
from src.synth import DIMENSIONS, make_synthetic_families, make_synthetic_items
from src.validate import run_checks


# ---- the generator ---------------------------------------------------------


def test_generates_eighty_items_twenty_per_cell():
    items = make_synthetic_items(20)
    assert len(items) == 80
    for c in CELLS:
        assert sum(1 for i in items if i.cell == c) == 20


def test_generator_is_deterministic():
    a = make_synthetic_items(20, seed=0)
    b = make_synthetic_items(20, seed=0)
    assert [i.passage for i in a] == [i.passage for i in b]


def test_different_seeds_give_different_text():
    a = make_synthetic_items(5, seed=0)
    b = make_synthetic_items(5, seed=1)
    assert [i.passage for i in a] != [i.passage for i in b]


def test_synthetic_items_satisfy_the_structural_manipulation():
    for it in make_synthetic_items(20):
        for d in DIMENSIONS:
            n = len(it.distinct_values(d))
            assert n == (1 if it.coherence == "coherent" else 4), (it.id, d, n)


def test_synthetic_items_pass_every_validation_gate():
    """If the fake items could not pass the gates, the gates could not be
    exercised end to end before the real items exist."""
    results = run_checks(make_synthetic_items(20))
    assert [r.name for r in results if not r.passed] == []


def test_synthetic_items_are_labelled_as_synthetic():
    for it in make_synthetic_items(4):
        assert it.domain == "synthetic"
        assert "SYNTHETIC" in (it.notes or "")
        assert it.review_status == "unreviewed"
        if not it.ground_truth:
            assert "SYNTHETIC" in it.confound_note


def test_synthetic_families_round_trip_through_the_loader(tmp_path):
    for fam in make_synthetic_families(3):
        (tmp_path / f"{fam.family_id}.json").write_text(
            json.dumps(fam.model_dump(), indent=2), encoding="utf-8"
        )
    assert len(load_items([tmp_path])) == 12


# ---- the expectations are derived, not pasted ------------------------------


def test_expected_auc_follows_from_the_construction():
    assert expected_auc(5) == (20 - 5) * 20 / 400 == 0.75
    assert expected_auc(9) == (20 - 9) * 20 / 400 == 0.55
    assert expected_auc(0) == 1.0
    assert expected_auc(20) == 0.0


def test_expected_cell_means_follow_from_the_construction():
    assert expected_false_mean() == pytest.approx(0.3095)
    assert expected_true_mean(5) == pytest.approx(0.6220)
    assert expected_true_mean(9) == pytest.approx(0.4720)
    assert EXPECTED["cell_means"]["coherent_true"] == expected_true_mean(5)


def test_dropping_abstentions_expectation_is_one():
    """The 5 abstaining TRUE items are exactly the ones that lost every pair."""
    assert EXPECTED["auc_if_abstained_dropped"] == {"coherent": 1.0, "diverse": 1.0}


# ---- the whole pipeline ----------------------------------------------------


@pytest.mark.slow
def test_full_smoke_pipeline_runs_and_every_check_passes(tmp_path):
    rc = run_smoke(tmp_path / "smoke", n_resamples=200, keep=True, quiet=True)
    assert rc == 0
    out = tmp_path / "smoke"
    assert (out / "run_mock.json").exists()
    assert (out / "analysis_mock.json").exists()
    assert (out / "analysis_mock.md").exists()
    assert len(list((out / "items").glob("fam_*.json"))) == 20

    analysis = json.loads((out / "analysis_mock.json").read_text(encoding="utf-8"))[
        "analysis"
    ]
    assert analysis["auc_primary_full_set"]["conditions"]["coherent"]["estimate"]["value"] == 0.75
    assert analysis["auc_primary_full_set"]["conditions"]["diverse"]["estimate"]["value"] == 0.55


@pytest.mark.slow
def test_smoke_markdown_labels_the_run_as_synthetic(tmp_path):
    run_smoke(tmp_path / "smoke", n_resamples=100, keep=True, quiet=True)
    md = (tmp_path / "smoke" / "analysis_mock.md").read_text(encoding="utf-8")
    assert "SYNTHETIC RUN" in md
    assert "fixtures, not measurements" in md
    assert "400" in md  # the pair count


@pytest.mark.slow
def test_smoke_fails_loudly_if_an_expectation_is_wrong(tmp_path, monkeypatch):
    """A smoke test that cannot fail is decoration."""
    import src.smoke as smoke

    monkeypatch.setitem(smoke.EXPECTED["auc"], "coherent", 0.99)
    with pytest.raises(smoke.SmokeFailure, match="AUC within coherent"):
        run_smoke(tmp_path / "smoke2", n_resamples=100, keep=False, quiet=True)
