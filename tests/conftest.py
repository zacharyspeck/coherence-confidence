"""Shared fixtures. Builds structurally valid synthetic items so tests never
depend on the real item set existing yet."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import Case, Family, Item, compute_word_count  # noqa: E402

DIMS = ("region", "period", "device", "unit_type")

_FILLER = (
    "The team recorded the outcome for each case using the same instrument and "
    "logged every reading in a shared sheet at the end of the observation window."
)


def make_item(
    family_id: str = "fam_test",
    cell: str = "coherent_true",
    claim: str = "the treatment increases the measured outcome",
    dims: tuple[str, ...] = DIMS,
    *,
    coherent: bool | None = None,
    n_distinct: int | None = None,
    passage_extra_words: int = 0,
) -> Item:
    """Build a structurally valid item.

    coherent/n_distinct override the structure implied by `cell`, so tests can
    construct deliberately broken items for the validator.
    """
    if coherent is None:
        coherent = cell.startswith("coherent")
    if n_distinct is None:
        n_distinct = 1 if coherent else 4

    cases = []
    for k in range(4):
        conditions = {d: f"{d}_v{(k % n_distinct) + 1}" for d in dims}
        text = (
            f"Case {k + 1} was observed and the recorded outcome rose by "
            f"{10 + k} percent."
        )
        cases.append(Case(case_id=f"c{k + 1}", text=text, conditions=conditions))

    lead = "Four cases were observed under a fixed protocol."
    extra = (" pad" * passage_extra_words).strip()
    passage = "\n".join([lead] + [c.text for c in cases] + [_FILLER, extra]).strip()

    truth = cell.endswith("_true")
    return Item(
        id=f"{family_id}__{cell}",
        family_id=family_id,
        cell=cell,
        claim=claim,
        cases=cases,
        passage=passage,
        ground_truth=truth,
        confound_note=None if truth else "Synthetic flaw for testing.",
        flaw_type=(
            None
            if truth
            else ("shared_confound" if cell == "coherent_false" else "temporal")
        ),
        word_count=compute_word_count(passage),
        domain="synthetic",
        source="generated",
    )


def make_family(family_id: str = "fam_test", dims: tuple[str, ...] = DIMS) -> Family:
    cells = ("coherent_true", "coherent_false", "diverse_true", "diverse_false")
    return Family(
        family_id=family_id,
        domain="synthetic",
        claim="the treatment increases the measured outcome",
        dimensions=list(dims),
        items=[make_item(family_id=family_id, cell=c, dims=dims) for c in cells],
    )


@pytest.fixture
def item():
    return make_item()


@pytest.fixture
def family():
    return make_family()


@pytest.fixture
def item_set():
    """20 families x 4 cells = 80 structurally valid synthetic items."""
    return [i for k in range(20) for i in make_family(f"fam_s{k:02d}").items]
