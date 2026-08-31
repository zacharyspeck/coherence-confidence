"""Synthetic item generator for the plumbing smoke test (step 6).

These are FAKE items with randomly assembled text. They are structurally valid -
they satisfy the schema, the coherent/diverse manipulation, word-count parity,
and the lexical-giveaway gate - but they say nothing. Their only job is to let
the entire pipeline run end to end before any real content exists.

Nothing here ever writes to `items/draft/` or `items/seed/`.

Lexical cleanliness is built in rather than hoped for. The only cell-dependent
text is the closing sentence, and it follows the same balanced FACT x SCOPE
scheme the real items use (D-022): every clause variant lands in exactly one TRUE
and one FALSE item per family, so no word in the closers correlates with the
label. That matters here specifically - if the synthetic set could not pass the
lexical and closer-balance gates, those gates would never be exercised end to end
before the real content existed, which is the entire point of step 6.
"""

from __future__ import annotations

import random
from typing import Sequence

from .models import CELLS, Case, Family, Item, compute_word_count

REGIONS = ["ashfield", "brentmoor", "calderon", "dunwich", "eastvale", "fenwick"]
PERIODS = ["january", "february", "march", "april", "august", "november"]
DEVICES = ["handset", "terminal", "kiosk", "laptop", "tablet", "console"]
UNITS = ["batch", "cohort", "parcel", "shipment", "cluster", "panel"]
METRICS = ["units", "points", "counts", "marks", "ticks", "steps"]
VERBS = ["rise", "shift", "change", "move", "gain", "swing"]
SUBJECTS = ["throughput", "yield", "uptake", "turnout", "output", "recovery"]

DIMENSIONS = ("region", "period", "device", "unit_kind")

#: The balanced closer scheme the real items use (D-022): FACT x SCOPE plus a
#: shared tail, assigned so every clause lands in exactly one TRUE and one FALSE
#: item per family. That gives the closer zero bag-of-words signal about the
#: label, which is the property the synthetic set has to reproduce if it is going
#: to exercise the lexical and closer-balance gates honestly.
FACT = {
    "changed": "Ambient load across the district ran at double the earlier period",
    "same": "Ambient load across the district matched the earlier period closely",
}
SCOPE = {
    "reach": "and every unit logged here stood on the open floor",
    "block": "and every unit logged here stood inside a sealed cabinet",
}
TAIL = "no other change was made to any unit."

#: coherent_false is the only cell where the confound both moved AND reached the
#: units. The variant given to coherent_true vs diverse_true is rotated by family
#: index so it is orthogonal to coherence rather than confounded with it.
CLOSER_ASSIGN = {
    "coherent_false": ("changed", "reach"),
    "diverse_false": ("same", "block"),
}
CLOSER_ASSIGN_TRUE = (
    {"coherent_true": ("changed", "block"), "diverse_true": ("same", "reach")},
    {"coherent_true": ("same", "reach"), "diverse_true": ("changed", "block")},
)

CLAIM_TEMPLATE = "the {change} raises {subject}"
CHANGES = [
    "revised procedure",
    "updated setting",
    "new schedule",
    "adjusted threshold",
    "replacement part",
    "modified routine",
]


def _values(rng: random.Random, pool: Sequence[str], n_distinct: int) -> list[str]:
    """4 condition values with exactly `n_distinct` distinct entries."""
    picked = rng.sample(list(pool), n_distinct)
    return [picked[i % n_distinct] for i in range(4)]


def make_synthetic_family(index: int, seed: int = 0) -> Family:
    rng = random.Random(seed * 1000 + index)
    family_id = f"fam_syn{index:02d}"
    change = CHANGES[index % len(CHANGES)]
    subject = SUBJECTS[index % len(SUBJECTS)]
    claim = CLAIM_TEMPLATE.format(change=change, subject=subject)

    items: list[Item] = []
    for cell in CELLS:
        n_distinct = 1 if cell.startswith("coherent") else 4
        cell_rng = random.Random(rng.randrange(10**9))

        vals = {
            "region": _values(cell_rng, REGIONS, n_distinct),
            "period": _values(cell_rng, PERIODS, n_distinct),
            "device": _values(cell_rng, DEVICES, n_distinct),
            "unit_kind": _values(cell_rng, UNITS, n_distinct),
        }

        cases: list[Case] = []
        for c in range(4):
            conditions = {d: vals[d][c] for d in DIMENSIONS}
            verb = VERBS[(index + c) % len(VERBS)]
            metric = METRICS[(index + c) % len(METRICS)]
            text = (
                f"Record {family_id[4:]}{c + 1} from {conditions['region']} in "
                f"{conditions['period']} on the {conditions['device']} for one "
                f"{conditions['unit_kind']} showed a {verb} of {12 + c * 3} "
                f"{metric} against the prior window."
            )
            cases.append(Case(case_id=f"c{c + 1}", text=text, conditions=conditions))

        assign = {**CLOSER_ASSIGN, **CLOSER_ASSIGN_TRUE[index % 2]}
        fact_key, scope_key = assign[cell]
        closer = f"{FACT[fact_key]}, {SCOPE[scope_key]}; {TAIL}"
        lead = (
            f"Log {family_id[4:]}{CELLS.index(cell)} records four observations "
            f"of {subject} taken after the {change} was introduced."
        )
        passage = "\n".join([lead] + [c.text for c in cases] + [closer])

        truth = cell.endswith("_true")
        items.append(
            Item(
                id=f"{family_id}__{cell}",
                family_id=family_id,
                cell=cell,
                claim=claim,
                cases=cases,
                passage=passage,
                ground_truth=truth,
                confound_note=(
                    None
                    if truth
                    else "SYNTHETIC placeholder. This item has no real flaw; it "
                    "exists only to exercise the pipeline."
                ),
                flaw_type=(
                    None
                    if truth
                    else (
                        "shared_confound"
                        if cell == "coherent_false"
                        else ("temporal" if index % 2 == 0 else "claim_mismatch")
                    )
                ),
                closer_variant=f"{fact_key}_{scope_key}",
                word_count=compute_word_count(passage),
                domain="synthetic",
                source="generated",
                notes="SYNTHETIC smoke-test item. Never for analysis.",
            )
        )

    return Family(
        family_id=family_id,
        domain="synthetic",
        claim=claim,
        dimensions=list(DIMENSIONS),
        items=items,
    )


def make_synthetic_families(n_families: int = 20, seed: int = 0) -> list[Family]:
    return [make_synthetic_family(i, seed) for i in range(n_families)]


def make_synthetic_items(n_families: int = 20, seed: int = 0) -> list[Item]:
    return [i for f in make_synthetic_families(n_families, seed) for i in f.items]
