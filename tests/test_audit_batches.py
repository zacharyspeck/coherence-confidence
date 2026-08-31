"""The blind-audit batches must actually be blind.

If the answer key leaks into a batch file, step 9 measures nothing - an auditor
would be reading the answer rather than finding it, and the audit would report
100% flaw detection on a broken item set.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from conftest import make_family

from src.models import CELLS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_audit_batches", REPO_ROOT / "scripts" / "make_audit_batches.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mab = _load_script()


@pytest.fixture
def item_dir(tmp_path):
    d = tmp_path / "items"
    d.mkdir()
    for k in range(20):
        fam = make_family(f"fam_a{k:02d}")
        (d / f"fam_a{k:02d}.json").write_text(
            json.dumps(fam.model_dump(), indent=2), encoding="utf-8"
        )
    return d


def build(item_dir, out, rnd=0):
    out = Path(out)
    rc = mab.main([
        "--items", str(item_dir),
        "--out", str(out / "audit"),
        "--key-out", str(out / "key"),
        "--round", str(rnd),
    ])
    assert rc == 0
    blind = out / "audit" / f"round{rnd}" / "blind"
    keymap = json.loads((out / "key" / f"round{rnd}.json").read_text(encoding="utf-8"))
    return blind, keymap


# ---- coverage --------------------------------------------------------------


def test_every_item_appears_exactly_once(item_dir, tmp_path):
    blind, keymap = build(item_dir, tmp_path / "audit")
    assert len(keymap) == 80
    assert len({v["item_id"] for v in keymap.values()}) == 80
    codes_in_files = []
    for p in sorted(blind.glob("batch_*.md")):
        text = p.read_text(encoding="utf-8")
        codes_in_files += [c for c in keymap if f"## ITEM {c}" in text]
    assert sorted(codes_in_files) == sorted(keymap)


def test_ten_batches_of_eight(item_dir, tmp_path):
    blind, _ = build(item_dir, tmp_path / "audit")
    files = sorted(blind.glob("batch_*.md"))
    assert len(files) == 10
    for p in files:
        assert p.read_text(encoding="utf-8").count("## ITEM ") == 8


def test_true_decoys_are_included(item_dir, tmp_path):
    """Without TRUE items in the mix, an auditor knows everything is broken and
    will manufacture a flaw for every item."""
    _, keymap = build(item_dir, tmp_path / "audit")
    n_true = sum(1 for v in keymap.values() if v["ground_truth"])
    assert n_true == 40
    assert n_true == sum(1 for v in keymap.values() if not v["ground_truth"])


# ---- blindness -------------------------------------------------------------


def test_no_batch_contains_two_items_from_one_family(item_dir, tmp_path):
    """Seeing a family's coherent_false beside its coherent_true gives the answer
    away by diff."""
    _, keymap = build(item_dir, tmp_path / "audit")
    by_batch: dict[int, list[str]] = {}
    for v in keymap.values():
        by_batch.setdefault(v["batch"], []).append(v["family_id"])
    for b, fams in by_batch.items():
        assert len(fams) == len(set(fams)), f"batch {b} repeats a family"


def test_blind_files_contain_no_cell_labels(item_dir, tmp_path):
    blind, _ = build(item_dir, tmp_path / "audit")
    for p in blind.glob("batch_*.md"):
        text = p.read_text(encoding="utf-8")
        for cell in CELLS:
            assert cell not in text


def test_blind_files_contain_no_confound_notes(item_dir, tmp_path):
    blind, keymap = build(item_dir, tmp_path / "audit")
    notes = [v["confound_note"] for v in keymap.values() if v["confound_note"]]
    assert notes
    for p in blind.glob("batch_*.md"):
        text = p.read_text(encoding="utf-8")
        for n in notes:
            assert n not in text


def test_blind_files_contain_no_ids_or_ground_truth(item_dir, tmp_path):
    blind, keymap = build(item_dir, tmp_path / "audit")
    for p in blind.glob("batch_*.md"):
        text = p.read_text(encoding="utf-8")
        assert "ground_truth" not in text
        assert "flaw_type" not in text
        for v in keymap.values():
            assert v["item_id"] not in text
    # family_id is deliberately not asserted here: the test fixture embeds it in
    # the passage text to keep the synthetic passages distinct, so an assertion
    # would be testing the fixture rather than the script. Real passages do not
    # contain it, and knowing a family id would not reveal a label in any case
    # since an auditor only ever sees one item per family.


def test_keymap_lives_outside_the_audit_tree_entirely(item_dir, tmp_path):
    """Blindness must not depend on an auditor obeying an instruction. The key
    goes in a separate tree, not merely a sibling directory."""
    blind, _ = build(item_dir, tmp_path / "audit")
    audit_root = blind.parent.parent
    assert list(audit_root.rglob("keymap.json")) == []
    assert list(audit_root.rglob("*key*.json")) == []
    assert (tmp_path / "audit" / "key" / "round0.json").exists()


def test_a_verdicts_directory_is_prepared(item_dir, tmp_path):
    blind, _ = build(item_dir, tmp_path / "audit")
    assert (blind.parent / "verdicts").is_dir()


def test_batch_files_state_that_some_items_are_fine(item_dir, tmp_path):
    blind, _ = build(item_dir, tmp_path / "audit")
    text = (blind / "batch_00.md").read_text(encoding="utf-8")
    assert "Some of these items have nothing wrong with them." in text


# ---- rounds ----------------------------------------------------------------


def test_rounds_regroup_the_items(item_dir, tmp_path):
    """A second round must be a genuinely different grouping, not a rotation of
    the same eight-item bundles past a different agent."""
    _, k0 = build(item_dir, tmp_path / "a0", rnd=0)
    _, k1 = build(item_dir, tmp_path / "a1", rnd=1)

    def bundles(k):
        out: dict[int, frozenset] = {}
        for v in k.values():
            out.setdefault(v["batch"], set()).add(v["item_id"])
        return {frozenset(s) for s in out.values()}

    assert bundles(k0) != bundles(k1)
    assert not (bundles(k0) & bundles(k1))


def test_codes_are_stable_across_rounds(item_dir, tmp_path):
    """Same item, same code - so verdicts from two rounds can be joined."""
    _, k0 = build(item_dir, tmp_path / "a0", rnd=0)
    _, k1 = build(item_dir, tmp_path / "a1", rnd=1)
    assert {c: v["item_id"] for c, v in k0.items()} == {
        c: v["item_id"] for c, v in k1.items()
    }


def test_codes_do_not_encode_the_label():
    """A code derived from the cell would leak the answer through the filename."""
    a = mab.code_for("fam_x__coherent_true")
    b = mab.code_for("fam_x__coherent_false")
    c = mab.code_for("fam_y__coherent_true")
    assert len({a, b, c}) == 3
    assert all(len(x) == 6 for x in (a, b, c))
