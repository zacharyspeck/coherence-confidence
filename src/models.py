"""Pydantic models for coherence-confidence items.

The invariants enforced here are the ones that make the 2x2 mean anything. They
are deliberately strict and fail loudly: a silently malformed item is worse than
a missing one, because it still produces a number.

See items/schema.json for the JSON Schema mirror of this model, and DECISIONS.md
D-007 (per-family dimensions), D-013 (one file per family), D-014 (word count).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Cell = Literal["coherent_true", "coherent_false", "diverse_true", "diverse_false"]
FlawType = Literal["shared_confound", "temporal", "claim_mismatch"]
ReviewStatus = Literal["unreviewed", "reviewed", "rejected"]

CELLS: tuple[Cell, ...] = (
    "coherent_true",
    "coherent_false",
    "diverse_true",
    "diverse_false",
)

N_CASES = 4
N_DIMENSIONS = 4

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip."""
    return _WS.sub(" ", text).strip()


def compute_word_count(passage: str) -> int:
    """Whitespace-token count of the passage. Single source of truth (D-014)."""
    return len(normalize_ws(passage).split())


def coherence_of(cell: str) -> Literal["coherent", "diverse"]:
    return "coherent" if cell.startswith("coherent") else "diverse"


def truth_of(cell: str) -> bool:
    return cell.endswith("_true")


class Case(BaseModel):
    """One observed case inside a passage."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^c[1-4]$")
    text: str = Field(min_length=10)
    conditions: dict[str, str]

    @model_validator(mode="after")
    def _check_conditions(self) -> "Case":
        if len(self.conditions) != N_DIMENSIONS:
            raise ValueError(
                f"case {self.case_id}: expected exactly {N_DIMENSIONS} condition "
                f"dimensions, got {len(self.conditions)}: {sorted(self.conditions)}"
            )
        for k, v in self.conditions.items():
            if not k or not k.strip():
                raise ValueError(f"case {self.case_id}: empty condition dimension name")
            if not v or not v.strip():
                raise ValueError(
                    f"case {self.case_id}: condition '{k}' has an empty value"
                )
        return self


class Item(BaseModel):
    """One item = one cell of one family."""

    model_config = ConfigDict(extra="forbid")

    id: str
    family_id: str = Field(pattern=r"^[a-z0-9_]+$")
    cell: Cell
    claim: str = Field(min_length=10)
    cases: list[Case]
    passage: str = Field(min_length=50)
    ground_truth: bool
    confound_note: str | None
    word_count: int = Field(ge=1)

    flaw_type: FlawType | None = None
    domain: str = "unspecified"
    review_status: ReviewStatus = "unreviewed"
    source: Literal["hand", "generated"] = "generated"
    notes: str | None = None

    # ---- derived -----------------------------------------------------------

    @property
    def coherence(self) -> Literal["coherent", "diverse"]:
        return coherence_of(self.cell)

    @property
    def dimensions(self) -> list[str]:
        return sorted(self.cases[0].conditions)

    def distinct_values(self, dimension: str) -> set[str]:
        return {c.conditions[dimension] for c in self.cases}

    # ---- validation --------------------------------------------------------

    @model_validator(mode="after")
    def _check_item(self) -> "Item":
        errs: list[str] = []

        if len(self.cases) != N_CASES:
            errs.append(f"expected exactly {N_CASES} cases, got {len(self.cases)}")

        expected_id = f"{self.family_id}__{self.cell}"
        if self.id != expected_id:
            errs.append(f"id must be '{expected_id}', got '{self.id}'")

        if self.ground_truth != truth_of(self.cell):
            errs.append(
                f"ground_truth={self.ground_truth} disagrees with cell='{self.cell}'"
            )

        if self.cases:
            case_ids = [c.case_id for c in self.cases]
            if case_ids != ["c1", "c2", "c3", "c4"]:
                errs.append(f"case_ids must be c1..c4 in order, got {case_ids}")

            dim_sets = [frozenset(c.conditions) for c in self.cases]
            if len(set(dim_sets)) != 1:
                errs.append(
                    "all 4 cases must carry identical condition dimension keys; got "
                    + " | ".join(sorted(",".join(sorted(d)) for d in set(dim_sets)))
                )

        # confound_note / flaw_type must agree with ground_truth.
        if self.ground_truth:
            if self.confound_note is not None:
                errs.append("TRUE items must have confound_note=null")
            if self.flaw_type is not None:
                errs.append("TRUE items must have flaw_type=null")
        else:
            if not (self.confound_note or "").strip():
                errs.append("FALSE items must have a non-empty confound_note")
            if self.flaw_type is None:
                errs.append("FALSE items must declare a flaw_type")
            elif self.cell == "coherent_false" and self.flaw_type != "shared_confound":
                errs.append(
                    f"coherent_false must use flaw_type='shared_confound', "
                    f"got '{self.flaw_type}'"
                )
            elif self.cell == "diverse_false" and self.flaw_type not in (
                "temporal",
                "claim_mismatch",
            ):
                errs.append(
                    "diverse_false must use flaw_type='temporal' or 'claim_mismatch' "
                    f"(a single confound cannot cover 4 diverse cases), got "
                    f"'{self.flaw_type}'"
                )

        # Every case sentence must actually appear in the prose the model reads.
        # Without this, the condition metadata can drift away from the passage and
        # the coherent/diverse manipulation becomes fiction.
        norm_passage = normalize_ws(self.passage)
        for c in self.cases:
            if normalize_ws(c.text) not in norm_passage:
                errs.append(
                    f"case {c.case_id} text does not appear verbatim in passage: "
                    f"{c.text[:60]!r}"
                )

        actual_wc = compute_word_count(self.passage)
        if self.word_count != actual_wc:
            errs.append(
                f"word_count={self.word_count} but passage has {actual_wc} words "
                f"(recompute with scripts/recount_words.py)"
            )

        if errs:
            raise ValueError(f"item '{self.id}': " + "; ".join(errs))
        return self

    # ---- helpers -----------------------------------------------------------

    @classmethod
    def build(cls, **kwargs) -> "Item":
        """Construct with word_count derived from the passage."""
        kwargs.setdefault("word_count", compute_word_count(kwargs["passage"]))
        kwargs.setdefault("id", f"{kwargs['family_id']}__{kwargs['cell']}")
        return cls(**kwargs)


class Family(BaseModel):
    """The 4 items of one family, as stored in a single JSON file (D-013)."""

    model_config = ConfigDict(extra="forbid")

    family_id: str
    domain: str
    claim: str
    dimensions: list[str]
    items: list[Item]

    @model_validator(mode="after")
    def _check_family(self) -> "Family":
        errs: list[str] = []

        cells = [i.cell for i in self.items]
        if sorted(cells) != sorted(CELLS):
            errs.append(f"family must contain exactly the 4 cells, got {cells}")

        for i in self.items:
            if i.family_id != self.family_id:
                errs.append(f"item {i.id} has family_id '{i.family_id}'")
            if normalize_ws(i.claim) != normalize_ws(self.claim):
                errs.append(
                    f"item {i.id} claim differs from the family claim; the claim must "
                    "be identical across all 4 cells or the 2x2 is not within-family"
                )
            if i.dimensions != sorted(self.dimensions):
                errs.append(
                    f"item {i.id} dimensions {i.dimensions} != family dimensions "
                    f"{sorted(self.dimensions)}"
                )

        if len(set(self.dimensions)) != N_DIMENSIONS:
            errs.append(
                f"family must declare exactly {N_DIMENSIONS} distinct dimensions, "
                f"got {self.dimensions}"
            )

        if errs:
            raise ValueError(f"family '{self.family_id}': " + "; ".join(errs))
        return self


# ---- loading ---------------------------------------------------------------


def load_family(path: Path) -> Family:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return Family.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - we re-raise with the file path attached
        raise ValueError(f"{path}: {exc}") from exc


def load_families(dirs: Iterable[str | Path]) -> list[Family]:
    """Load every fam_*.json under the given directories, sorted by family_id."""
    fams: list[Family] = []
    seen: dict[str, Path] = {}
    for d in dirs:
        for p in sorted(Path(d).glob("fam_*.json")):
            fam = load_family(p)
            if fam.family_id in seen:
                raise ValueError(
                    f"duplicate family_id '{fam.family_id}' in {p} and "
                    f"{seen[fam.family_id]}"
                )
            seen[fam.family_id] = p
            fams.append(fam)
    return sorted(fams, key=lambda f: f.family_id)


def load_items(dirs: Iterable[str | Path]) -> list[Item]:
    """Flatten all families in the given directories into a list of items."""
    items = [i for fam in load_families(dirs) for i in fam.items]
    ids = [i.id for i in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate item ids: {sorted(dupes)}")
    return sorted(items, key=lambda i: (i.family_id, i.cell))


def items_by_cell(items: Iterable[Item]) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {c: [] for c in CELLS}
    for i in items:
        out[i.cell].append(i)
    return out
