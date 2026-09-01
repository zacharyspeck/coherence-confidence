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

#: The 2x2 plus the surface-complexity control (D-030).
#:
#: `decorative_*` cells carry as many distinct entities as a diverse item - four
#: operators, four serial numbers, four clock times, four ticket ids - while
#: holding every dimension that bears on the claim IDENTICAL across the four
#: cases, exactly as a coherent item does. Surface busyness matches diverse;
#: evidential structure matches coherent. Whichever of those two the model's
#: confidence follows is the answer to "is the effect about evidential
#: independence, or about how much there is to parse?"
Cell = Literal[
    "coherent_true",
    "coherent_false",
    "diverse_true",
    "diverse_false",
    "decorative_true",
    "decorative_false",
]

#: HOW a false item is false. This is orthogonal to coherence on purpose (D-024):
#: `scope_mismatch` appears in BOTH false cells, which is what makes a
#: mechanism-matched AUC comparison possible.
#:
#:   stated_confound    a fact in the passage is an alternative cause that covers
#:                      every case at once. Only possible when the cases share
#:                      conditions, so only ever in coherent_false (D-004).
#:   broken_chronology  in >=2 cases the outcome is dated before the treatment.
#:   scope_mismatch     the cases establish something narrower than the claim
#:                      asserts - the outcome is tallied over a subset of the
#:                      population the claim is about.
FlawMechanism = Literal["stated_confound", "broken_chronology", "scope_mismatch"]

#: The confound clause pair (D-022). "changed"/"same" = did the potential
#: alternative cause move over the comparison period; "reach"/"block" = could it
#: reach the observed units. ONLY `changed_reach` makes an item false.
ConfoundVariant = Literal["changed_reach", "changed_block", "same_reach", "same_block"]

#: The scope clause pair (D-024). "subset"/"whole" = what population the outcome
#: was tallied over; "incomplete"/"complete" = whether that population is in fact
#: everyone the claim is about. ONLY `subset_incomplete` makes an item false.
ScopeVariant = Literal[
    "subset_incomplete", "subset_complete", "whole_incomplete", "whole_complete"
]

#: Retained for reading pre-D-024 files. Never written by anything current.
LegacyFlawType = Literal["shared_confound", "temporal", "claim_mismatch"]

ReviewStatus = Literal["unreviewed", "reviewed", "rejected"]

#: The single clause combination in each pair that falsifies an item.
LIVE_CONFOUND: str = "changed_reach"
LIVE_SCOPE: str = "subset_incomplete"

#: The four cells of the 2x2. The primary endpoint and the ANOVA use only these.
CORE_CELLS: tuple[Cell, ...] = (
    "coherent_true",
    "coherent_false",
    "diverse_true",
    "diverse_false",
)

#: The control arm. Present in the 10 scope_mismatch families only, so it is
#: directly comparable to the matched subset.
CONTROL_CELLS: tuple[Cell, ...] = ("decorative_true", "decorative_false")

#: Every cell that can exist. Per-cell reporting iterates this; the 2x2 does not.
CELLS: tuple[Cell, ...] = CORE_CELLS + CONTROL_CELLS

COHERENCE_LEVELS: tuple[str, ...] = ("coherent", "diverse", "decorative")

N_CASES = 4
N_DIMENSIONS = 4
#: Decoration dimensions per case, for items that carry them.
N_DECORATIONS = 4

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip."""
    return _WS.sub(" ", text).strip()


def compute_word_count(passage: str) -> int:
    """Whitespace-token count of the passage. Single source of truth (D-014)."""
    return len(normalize_ws(passage).split())


def coherence_of(cell: str) -> Literal["coherent", "diverse", "decorative"]:
    return cell.rsplit("_", 1)[0]  # type: ignore[return-value]


def is_core(cell: str) -> bool:
    """True for the four cells of the 2x2, false for the control arm."""
    return cell in CORE_CELLS


def truth_of(cell: str) -> bool:
    return cell.endswith("_true")


class Case(BaseModel):
    """One observed case inside a passage."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^c[1-4]$")
    text: str = Field(min_length=10)

    #: Dimensions whose VARIATION makes the evidence more independent - region,
    #: season, device, population. Varying these is the diverse manipulation.
    conditions: dict[str, str]

    #: Dimensions whose variation is pure surface: who logged it, which serial
    #: number, what time of day, which ticket. Varying these adds entities to
    #: track and adds nothing evidential. Empty except in decorative items (D-030).
    decorations: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_conditions(self) -> "Case":
        for name, d, n in (
            ("condition", self.conditions, N_DIMENSIONS),
            ("decoration", self.decorations, N_DECORATIONS),
        ):
            if name == "decoration" and not d:
                continue  # decorations are optional; conditions are not
            if len(d) != n:
                raise ValueError(
                    f"case {self.case_id}: expected exactly {n} {name} dimensions, "
                    f"got {len(d)}: {sorted(d)}"
                )
            for k, v in d.items():
                if not k or not k.strip():
                    raise ValueError(
                        f"case {self.case_id}: empty {name} dimension name"
                    )
                if not v or not v.strip():
                    raise ValueError(
                        f"case {self.case_id}: {name} '{k}' has an empty value"
                    )
        overlap = set(self.conditions) & set(self.decorations)
        if overlap:
            raise ValueError(
                f"case {self.case_id}: {sorted(overlap)} is declared as both a "
                "condition and a decoration. The whole point of the control is "
                "that those two are different kinds of thing."
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

    flaw_mechanism: FlawMechanism | None = None
    confound_variant: ConfoundVariant | None = None
    scope_variant: ScopeVariant | None = None
    salience: float | None = None
    #: Recomputed and checked at load, exactly like word_count - it is a pure
    #: function of the passage, so a stale value means a stale edit (D-030).
    surface_complexity: dict[str, float] | None = None
    domain: str = "unspecified"
    review_status: ReviewStatus = "unreviewed"
    source: Literal["hand", "generated"] = "generated"
    notes: str | None = None

    # ---- derived -----------------------------------------------------------

    @property
    def coherence(self) -> Literal["coherent", "diverse", "decorative"]:
        return coherence_of(self.cell)

    @property
    def is_core(self) -> bool:
        """In the 2x2, as opposed to the surface-complexity control arm."""
        return is_core(self.cell)

    @property
    def dimensions(self) -> list[str]:
        return sorted(self.cases[0].conditions)

    @property
    def decoration_dimensions(self) -> list[str]:
        return sorted(self.cases[0].decorations)

    def distinct_values(self, dimension: str) -> set[str]:
        return {c.conditions[dimension] for c in self.cases}

    def distinct_decoration_values(self, dimension: str) -> set[str]:
        return {c.decorations[dimension] for c in self.cases}

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

            dec_sets = [frozenset(c.decorations) for c in self.cases]
            if len(set(dec_sets)) != 1:
                errs.append(
                    "all 4 cases must carry identical decoration dimension keys; got "
                    + " | ".join(sorted(",".join(sorted(d)) for d in set(dec_sets)))
                )

            # Decorations exist to make ONE cell surface-busy without making it
            # evidentially diverse. If any other cell carried them, the control
            # would not isolate anything (D-030).
            has_dec = any(c.decorations for c in self.cases)
            if self.coherence == "decorative":
                if not has_dec:
                    errs.append(
                        "a decorative item must carry decoration dimensions; that "
                        "is the entire manipulation"
                    )
                else:
                    for dim in self.decoration_dimensions:
                        n = len(self.distinct_decoration_values(dim))
                        if n != N_CASES:
                            errs.append(
                                f"decorative item has {n} distinct values on "
                                f"decoration '{dim}'; must be {N_CASES}, or the "
                                "surface variety does not match a diverse item"
                            )
                    for dim in self.dimensions:
                        n = len(self.distinct_values(dim))
                        if n != 1:
                            errs.append(
                                f"decorative item has {n} distinct values on "
                                f"CONDITION '{dim}'; must be exactly 1. Conditions "
                                "bear on the claim, so varying them would make this "
                                "a diverse item wearing a decorative label"
                            )
            elif has_dec:
                errs.append(
                    f"{self.cell} carries decorations; only decorative_* cells may. "
                    "Decorating another cell destroys the contrast the control "
                    "exists to draw"
                )

        # confound_note / flaw_mechanism must agree with ground_truth.
        if self.ground_truth:
            if self.confound_note is not None:
                errs.append("TRUE items must have confound_note=null")
            if self.flaw_mechanism is not None:
                errs.append("TRUE items must have flaw_mechanism=null")
        else:
            if not (self.confound_note or "").strip():
                errs.append("FALSE items must have a non-empty confound_note")
            if self.flaw_mechanism is None:
                errs.append("FALSE items must declare a flaw_mechanism")
            elif (
                self.flaw_mechanism == "stated_confound"
                and self.cell != "coherent_false"
            ):
                errs.append(
                    "only coherent_false may use flaw_mechanism='stated_confound': "
                    "a single stated fact cannot cover four cases that differ on "
                    f"every dimension (D-004). Got cell='{self.cell}'"
                )

        # The clause variants ARE the mechanism, so they must agree with the
        # declared one. This is what stops an item's metadata from drifting away
        # from the prose while still looking well-formed (D-022, D-024).
        live_confound = self.confound_variant == LIVE_CONFOUND
        live_scope = self.scope_variant == LIVE_SCOPE

        if self.confound_variant is not None:
            if live_confound and self.flaw_mechanism != "stated_confound":
                errs.append(
                    f"confound_variant='{LIVE_CONFOUND}' means the alternative "
                    "cause both moved and reached the units, which falsifies the "
                    f"item; flaw_mechanism must be 'stated_confound', got "
                    f"'{self.flaw_mechanism}'"
                )
            if self.flaw_mechanism == "stated_confound" and not live_confound:
                errs.append(
                    "flaw_mechanism='stated_confound' requires "
                    f"confound_variant='{LIVE_CONFOUND}', got "
                    f"'{self.confound_variant}'"
                )

        if self.scope_variant is not None:
            if live_scope and self.flaw_mechanism != "scope_mismatch":
                errs.append(
                    f"scope_variant='{LIVE_SCOPE}' means the outcome was tallied "
                    "over less than the claimed population, which falsifies the "
                    f"item; flaw_mechanism must be 'scope_mismatch', got "
                    f"'{self.flaw_mechanism}'"
                )
            if self.flaw_mechanism == "scope_mismatch" and not live_scope:
                errs.append(
                    "flaw_mechanism='scope_mismatch' requires "
                    f"scope_variant='{LIVE_SCOPE}', got '{self.scope_variant}'"
                )

        if self.ground_truth and (live_confound or live_scope):
            errs.append(
                "a TRUE item cannot carry a live clause combination: "
                f"confound_variant='{self.confound_variant}', "
                f"scope_variant='{self.scope_variant}'"
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

        if self.surface_complexity is not None:
            from .complexity import for_item

            want = for_item(self)
            drift = {
                k: (self.surface_complexity.get(k), v)
                for k, v in want.items()
                if abs(float(self.surface_complexity.get(k, -1)) - float(v)) > 1e-6
            }
            if drift:
                errs.append(
                    "surface_complexity is stale: "
                    + ", ".join(f"{k} stored {a} but passage gives {b}"
                                for k, (a, b) in sorted(drift.items()))
                    + " (fix with scripts/apply_complexity.py)"
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
    """One family per JSON file (D-013).

    Always the 4 cells of the 2x2. The 10 scope_mismatch families additionally
    carry the 2 decorative cells, so the surface-complexity control is drawn from
    the same scenarios and claims as the matched-mechanism subset (D-030).
    """

    model_config = ConfigDict(extra="forbid")

    family_id: str
    domain: str
    claim: str
    dimensions: list[str]
    decoration_dimensions: list[str] = Field(default_factory=list)
    items: list[Item]

    @property
    def has_control_arm(self) -> bool:
        return any(i.cell in CONTROL_CELLS for i in self.items)

    @model_validator(mode="after")
    def _check_family(self) -> "Family":
        errs: list[str] = []

        cells = sorted(i.cell for i in self.items)
        core = sorted(c for c in cells if c in CORE_CELLS)
        control = sorted(c for c in cells if c in CONTROL_CELLS)
        if core != sorted(CORE_CELLS):
            errs.append(f"family must contain all 4 cells of the 2x2, got {cells}")
        if control and control != sorted(CONTROL_CELLS):
            errs.append(
                f"the control arm must be both decorative cells or neither, got "
                f"{control}. One alone gives nothing to compare"
            )
        if len(cells) != len(set(cells)):
            errs.append(f"duplicate cells in family: {cells}")

        if control:
            if len(set(self.decoration_dimensions)) != N_DECORATIONS:
                errs.append(
                    f"a family with a control arm must declare exactly "
                    f"{N_DECORATIONS} distinct decoration dimensions, got "
                    f"{self.decoration_dimensions}"
                )
            for i in self.items:
                if i.cell in CONTROL_CELLS and i.decoration_dimensions != sorted(
                    self.decoration_dimensions
                ):
                    errs.append(
                        f"item {i.id} decoration dimensions "
                        f"{i.decoration_dimensions} != family "
                        f"{sorted(self.decoration_dimensions)}"
                    )
        elif self.decoration_dimensions:
            errs.append(
                "family declares decoration dimensions but has no decorative cells"
            )

        for i in self.items:
            if i.family_id != self.family_id:
                errs.append(f"item {i.id} has family_id '{i.family_id}'")
            if normalize_ws(i.claim) != normalize_ws(self.claim):
                errs.append(
                    f"item {i.id} claim differs from the family claim; the claim must "
                    "be identical across every cell or neither the 2x2 nor the control is "
                    "within-family"
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
