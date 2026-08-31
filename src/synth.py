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
#: Half the synthetic families are "scope families" where BOTH false cells are
#: falsified by scope_mismatch (D-024). Without them the matched-mechanism
#: subset - the confound-controlled primary endpoint - would never be exercised
#: end to end before the real items exist, which is the point of step 6.
CONFOUND_FAMILY_ASSIGN = {
    "coherent_false": (("changed", "reach"), ("whole", "complete"), "stated_confound"),
    "coherent_true": (("changed", "block"), ("subset", "complete"), None),
    "diverse_true": (("same", "reach"), ("whole", "incomplete"), None),
    "diverse_false": (("same", "block"), ("whole", "complete"), "broken_chronology"),
}
SCOPE_FAMILY_ASSIGN = {
    "coherent_false": (("changed", "block"), ("subset", "incomplete"), "scope_mismatch"),
    "diverse_false": (("same", "reach"), ("subset", "incomplete"), "scope_mismatch"),
    "coherent_true": (("same", "reach"), ("subset", "complete"), None),
    "diverse_true": (("changed", "block"), ("whole", "incomplete"), None),
}
POPULATION = {
    "subset": "counts cover the units that reported",
    "whole": "counts cover every unit on the list",
}
COMPLETENESS = {
    "complete": "and every unit on the list reported",
    "incomplete": "and three of the twelve listed units did not report",
}

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

    # One draw per COHERENCE LEVEL, shared by that level's true and false item -
    # exactly as in the real items, where the two differ only in the mid-passage
    # line. Drawing per cell would give each family four different vocabularies
    # and a large spurious TRUE/FALSE word imbalance.
    vals_by_coherence = {}
    for coherence, n_distinct in (("coherent", 1), ("diverse", 4)):
        r = random.Random(rng.randrange(10**9))
        vals_by_coherence[coherence] = {
            "region": _values(r, REGIONS, n_distinct),
            "period": _values(r, PERIODS, n_distinct),
            "device": _values(r, DEVICES, n_distinct),
            "unit_kind": _values(r, UNITS, n_distinct),
        }

    items: list[Item] = []
    for cell in CELLS:
        vals = vals_by_coherence["coherent" if cell.startswith("coherent") else "diverse"]

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

        scope_family = index % 2 == 1
        assign = SCOPE_FAMILY_ASSIGN if scope_family else CONFOUND_FAMILY_ASSIGN
        if index % 4 >= 2:  # rotate the TRUE items' scope clauses (D-026)
            assign = dict(assign)
            ct, dt = assign["coherent_true"], assign["diverse_true"]
            assign["coherent_true"] = (ct[0], dt[1], ct[2])
            assign["diverse_true"] = (dt[0], ct[1], dt[2])
        (fact_key, scope_key), (pop_key, comp_key), mech = assign[cell]
        midline = (
            f"{FACT[fact_key]}, {SCOPE[scope_key]}. "
            f"{POPULATION[pop_key]}, {COMPLETENESS[comp_key]}."
        )
        closer = TAIL
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
                flaw_mechanism=mech,
                confound_variant=f"{fact_key}_{scope_key}",
                scope_variant=f"{pop_key}_{comp_key}",
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
