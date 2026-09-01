"""The item model's job is to fail loudly. These tests check that it does."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import DIMS, make_family, make_item

from src.models import (
    CELLS,
    CORE_CELLS,
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
    assert sorted(i.cell for i in family.items) == sorted(CORE_CELLS)


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


def test_rejects_missing_flaw_mechanism_on_false_item():
    it = make_item(cell="coherent_false")
    with pytest.raises(ValueError, match="must declare a flaw_mechanism"):
        Item.model_validate(_mutate(it, flaw_mechanism=None))


def test_stated_confound_is_confined_to_coherent_false():
    """A single stated fact cannot cover four cases that differ on every
    dimension, so this mechanism is only available under coherence (D-004)."""
    for cell in ("diverse_false",):
        it = make_item(cell=cell)
        with pytest.raises(ValueError, match="only coherent_false may use"):
            Item.model_validate(_mutate(it, flaw_mechanism="stated_confound"))


def test_scope_mismatch_is_allowed_in_both_false_cells():
    """This is the point of D-024: the matched-mechanism comparison needs the
    same mechanism present under both coherence conditions."""
    for cell in ("coherent_false", "diverse_false"):
        it = make_item(cell=cell)
        ok = Item.model_validate(
            _mutate(it, flaw_mechanism="scope_mismatch",
                    scope_variant="subset_incomplete", confound_variant="changed_block")
        )
        assert ok.flaw_mechanism == "scope_mismatch"


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
    with pytest.raises(ValueError, match="all 4 cells of the 2x2"):
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


# ---- confound_variant (D-022) ------------------------------------------------


def test_variants_are_optional_on_the_model():
    """The model allows them to be absent; src/validate.py is what requires every
    real FALSE item to declare the variant matching its mechanism."""
    ref = make_item()
    d = ref.model_dump()
    d.pop("confound_variant")
    d.pop("scope_variant")
    it = Item.model_validate(d)
    assert it.confound_variant is None and it.scope_variant is None


def test_a_live_confound_requires_the_matching_mechanism():
    """confound_variant='changed_reach' IS the falsification. An item carrying it
    must say so, or its metadata is describing prose it does not have."""
    it = make_item(cell="coherent_false")
    ok = Item.model_validate(_mutate(it, confound_variant="changed_reach"))
    assert ok.confound_variant == "changed_reach"

    it = make_item(cell="diverse_false")  # declared broken_chronology
    with pytest.raises(ValueError, match="must be 'stated_confound'"):
        Item.model_validate(_mutate(it, confound_variant="changed_reach"))


def test_a_live_scope_requires_the_matching_mechanism():
    it = make_item(cell="coherent_false")  # declared stated_confound
    with pytest.raises(ValueError, match="must be 'scope_mismatch'"):
        Item.model_validate(_mutate(it, scope_variant="subset_incomplete"))


def test_declared_mechanism_requires_its_live_variant():
    it = make_item(cell="coherent_false")
    with pytest.raises(ValueError, match="requires confound_variant='changed_reach'"):
        Item.model_validate(_mutate(it, confound_variant="same_block"))


@pytest.mark.parametrize("variant", ["changed_reach"])
def test_true_items_may_never_carry_a_live_confound(variant, item):
    with pytest.raises(ValueError, match="TRUE item cannot carry a live"):
        Item.model_validate(_mutate(item, confound_variant=variant))


@pytest.mark.parametrize("variant", ["subset_incomplete"])
def test_true_items_may_never_carry_a_live_scope(variant, item):
    with pytest.raises(ValueError, match="TRUE item cannot carry a live"):
        Item.model_validate(_mutate(item, scope_variant=variant))


@pytest.mark.parametrize(
    "confound,scope",
    [
        ("changed_block", "subset_complete"),
        ("same_reach", "whole_incomplete"),
        ("same_block", "whole_complete"),
    ],
)
def test_true_items_accept_every_non_live_combination(confound, scope, item):
    ok = Item.model_validate(
        _mutate(item, confound_variant=confound, scope_variant=scope)
    )
    assert ok.confound_variant == confound
    assert ok.scope_variant == scope
