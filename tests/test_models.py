"""The item model's job is to fail loudly. These tests check that it does."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DIMS, make_family, make_item

from src.models import (
    CELLS,
    Case,
    Family,
    Item,
    coherence_of,
    compute_word_count,
    items_by_cell,
    load_families,
    load_items,
    normalize_ws,
    truth_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- happy path ------------------------------------------------------------


def test_valid_item_builds(item):
    assert item.id == "fam_test__coherent_true"
    assert len(item.cases) == 4
    assert item.ground_truth is True
    assert item.confound_note is None
    assert item.review_status == "unreviewed"


def test_valid_family_builds(family):
    assert sorted(i.cell for i in family.items) == sorted(CELLS)


def test_derived_properties(item):
    assert item.coherence == "coherent"
    assert item.dimensions == sorted(DIMS)
    assert item.distinct_values("region") == {"region_v1"}


def test_diverse_item_has_four_distinct_values():
    it = make_item(cell="diverse_true")
    assert it.coherence == "diverse"
    for d in DIMS:
        assert len(it.distinct_values(d)) == 4


def test_cell_helpers():
    assert coherence_of("coherent_false") == "coherent"
    assert coherence_of("diverse_true") == "diverse"
    assert truth_of("coherent_true") is True
    assert truth_of("diverse_false") is False


def test_word_count_matches_passage(item):
    assert item.word_count == compute_word_count(item.passage)
    assert item.word_count == len(item.passage.split())


def test_normalize_ws():
    assert normalize_ws("  a\n\n b \t c  ") == "a b c"


def test_build_derives_word_count_and_id():
    ref = make_item()
    it = Item.build(
        family_id="fam_b",
        cell="coherent_true",
        claim=ref.claim,
        cases=[c.model_dump() for c in ref.cases],
        passage=ref.passage,
        ground_truth=True,
        confound_note=None,
    )
    assert it.id == "fam_b__coherent_true"
    assert it.word_count == compute_word_count(ref.passage)


# ---- the model must reject malformed items ---------------------------------


def _mutate(item: Item, **changes) -> dict:
    d = item.model_dump()
    d.update(changes)
    return d


def test_rejects_wrong_case_count(item):
    d = _mutate(item)
    d["cases"] = d["cases"][:3]
    with pytest.raises(ValueError, match="exactly 4 cases|at least 4 items"):
        Item.model_validate(d)


def test_rejects_id_family_cell_mismatch(item):
    with pytest.raises(ValueError, match="id must be"):
        Item.model_validate(_mutate(item, id="wrong__coherent_true"))


def test_rejects_ground_truth_cell_mismatch(item):
    with pytest.raises(ValueError, match="disagrees with cell"):
        Item.model_validate(_mutate(item, ground_truth=False))


def test_rejects_confound_note_on_true_item(item):
    with pytest.raises(ValueError, match="TRUE items must have confound_note=null"):
        Item.model_validate(_mutate(item, confound_note="oops"))


def test_rejects_missing_confound_note_on_false_item():
    it = make_item(cell="coherent_false")
    with pytest.raises(ValueError, match="non-empty confound_note"):
        Item.model_validate(_mutate(it, confound_note="   "))


def test_rejects_missing_flaw_type_on_false_item():
    it = make_item(cell="coherent_false")
    with pytest.raises(ValueError, match="must declare a flaw_type"):
        Item.model_validate(_mutate(it, flaw_type=None))


def test_rejects_wrong_flaw_type_for_coherent_false():
    it = make_item(cell="coherent_false")
    with pytest.raises(ValueError, match="coherent_false must use"):
        Item.model_validate(_mutate(it, flaw_type="temporal"))


def test_rejects_shared_confound_for_diverse_false():
    """A single confound cannot cover 4 cases that differ on every dimension."""
    it = make_item(cell="diverse_false")
    with pytest.raises(ValueError, match="diverse_false must use"):
        Item.model_validate(_mutate(it, flaw_type="shared_confound"))


def test_rejects_case_text_not_in_passage(item):
    d = _mutate(item)
    d["cases"][0]["text"] = "This sentence is nowhere in the passage at all."
    with pytest.raises(ValueError, match="does not appear verbatim in passage"):
        Item.model_validate(d)


def test_rejects_stale_word_count(item):
    with pytest.raises(ValueError, match="word_count="):
        Item.model_validate(_mutate(item, word_count=item.word_count + 1))


def test_rejects_wrong_number_of_dimensions():
    with pytest.raises(ValueError, match="exactly 4 condition"):
        Case(case_id="c1", text="a case that is long enough", conditions={"a": "1"})


def test_rejects_empty_condition_value():
    with pytest.raises(ValueError, match="empty value"):
        Case(
            case_id="c1",
            text="a case that is long enough",
            conditions={"a": "1", "b": "2", "c": "3", "d": "  "},
        )


def test_rejects_mismatched_dimension_keys_across_cases(item):
    d = _mutate(item)
    conds = dict(d["cases"][1]["conditions"])
    conds["extra_dim"] = conds.pop("region")
    d["cases"][1]["conditions"] = conds
    with pytest.raises(ValueError, match="identical condition dimension keys"):
        Item.model_validate(d)


def test_rejects_out_of_order_case_ids(item):
    d = _mutate(item)
    d["cases"][0]["case_id"] = "c2"
    d["cases"][1]["case_id"] = "c1"
    with pytest.raises(ValueError, match="case_ids must be c1..c4 in order"):
        Item.model_validate(d)


def test_rejects_unknown_field(item):
    with pytest.raises(ValueError):
        Item.model_validate(_mutate(item, surprise="field"))


# ---- family-level invariants ----------------------------------------------


def test_family_rejects_missing_cell(family):
    d = family.model_dump()
    d["items"] = d["items"][:3]
    with pytest.raises(ValueError, match="exactly the 4 cells"):
        Family.model_validate(d)


def test_family_rejects_divergent_claim(family):
    d = family.model_dump()
    d["items"][2]["claim"] = "a completely different claim about something else"
    with pytest.raises(ValueError, match="claim differs from the family claim"):
        Family.model_validate(d)


def test_family_rejects_divergent_dimensions(family):
    d = family.model_dump()
    d["dimensions"] = ["region", "period", "device", "other_dim"]
    with pytest.raises(ValueError, match="dimensions"):
        Family.model_validate(d)


# ---- loading ---------------------------------------------------------------


def test_load_roundtrip(tmp_path):
    fam = make_family("fam_rt")
    (tmp_path / "fam_rt.json").write_text(
        json.dumps(fam.model_dump(), indent=2), encoding="utf-8"
    )
    loaded = load_families([tmp_path])
    assert len(loaded) == 1
    assert loaded[0].family_id == "fam_rt"
    items = load_items([tmp_path])
    assert len(items) == 4
    assert sorted(items_by_cell(items)) == sorted(CELLS)


def test_load_rejects_duplicate_family_ids(tmp_path):
    fam = make_family("fam_dup")
    for name in ("fam_dup_a.json", "fam_dup_b.json"):
        (tmp_path / name).write_text(
            json.dumps(fam.model_dump(), indent=2), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="duplicate family_id"):
        load_families([tmp_path])


def test_schema_json_is_valid_json_and_agrees_with_model():
    schema = json.loads(
        (REPO_ROOT / "items" / "schema.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["cell"]["enum"] == list(CELLS)
    model_fields = set(Item.model_fields)
    schema_fields = set(schema["properties"])
    assert model_fields == schema_fields, (
        f"schema.json and models.Item disagree: "
        f"only in model={sorted(model_fields - schema_fields)}, "
        f"only in schema={sorted(schema_fields - model_fields)}"
    )
    required = set(schema["required"])
    assert required <= model_fields
