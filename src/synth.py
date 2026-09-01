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

from .models import CORE_CELLS, Case, Family, Item, compute_word_count

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
#: Clause parts are built PER FAMILY. A single shared set gave all twenty
#: families the same falsifying sentence, which is exactly what
#: `no_duplicate_flaw_phrasing` exists to forbid - so the synthetic set could
#: not pass the gate it is supposed to exercise. Two small pools combined
#: multiplicatively give 24 distinct clause sets, enough for 20 families.
#: Pool sizes 6, 4 and 5 are pairwise co-prime enough that (i%6, i%4, i%5) is a
#: distinct triple for every i below 60 - so all 20 families differ in THREE
#: slots at once, not one. Varying a single noun left ~90% of the sentence
#: identical and the pairs still scored 0.92 similarity.
NOISE = ("ambient load", "line pressure", "supply lead time",
         "room humidity", "staff turnover", "batch age")
#: Composed from two pools so every family gets a DISTINCT phrase rather than
#: one of four. Modular indexing into short pools kept giving two families the
#: same clause with a single noun swapped, which scores 0.85+ however the
#: indices are permuted - the fix is more phrases, not better arithmetic.
CMP_MOVE = ("ran at double", "climbed by half against", "sat well above",
            "rose sharply on", "pushed clear of")
CMP_REF = ("the earlier period", "the prior window", "the previous quarter",
           "the earlier month")


def comparator(i: int) -> str:
    return f"{CMP_MOVE[i % 5]} {CMP_REF[(i // 5) % 4]}"
EXPOSURE = ("every unit logged here stood on the open floor",
            "nothing screened the units on this log",
            "all four units sat out in the open bay",
            "no unit here was shielded from it",
            "each logged unit ran in an open rack",
            "the units on this log were left uncovered throughout",
            "not one of these units had a barrier around it",
            "these four units worked in the open yard",
            "every unit in the log sat exposed to it",
            "the logged units were kept in an unenclosed space")
SHELTER = ("every unit logged here stood inside a sealed cabinet",
           "a closed enclosure screened the units on this log",
           "all four units sat behind fixed shielding",
           "every unit here was sealed away from it",
           "each logged unit ran in a shuttered rack",
           "the units on this log were kept covered throughout",
           "each of these units had a solid barrier around it",
           "these four units worked in an enclosed yard",
           "every unit in the log sat isolated from it",
           "the logged units were kept in a fully enclosed space")
POP_NOUN = ("units", "records", "entries", "cases", "runs",
            "reads", "returns", "slips", "tickets", "forms")
VERB_ACT = ("reported", "returned a figure", "were logged", "came back")
VERB_QUAL = ("on time", "to the register", "in full", "before the cutoff",
             "without a query")


def report_verb(i: int) -> str:
    return f"{VERB_ACT[i % 4]} {VERB_QUAL[(i // 4) % 5]}"
SHORT_N = ("three", "four", "five", "two", "six")
SHORT_D = ("twelve", "fifteen", "twenty", "nine")


def shortfall(i: int) -> str:
    return f"{SHORT_N[i % 5]} of the {SHORT_D[(i // 5) % 4]} listed"
FRAME = (
    "{noise} across the district {rest}",
    "Across every site in the group, {noise} {rest}",
    "The logs note that {noise} {rest}",
    "Through the whole window {noise} {rest}",
    "Site returns put {noise} at a level that {rest}",
    "Readings give {noise} as something that {rest}",
    "Over this window, {noise} {rest}",
    "The site file shows {noise} {rest}",
    "Monitoring had {noise} {rest}",
    "For the whole period {noise} {rest}",
)
POP_FRAME = (
    "Counts cover {scope}",
    "The totals here take in {scope}",
    "What is tallied below is {scope}",
    "These figures are drawn from {scope}",
    "The numbers reported are for {scope}",
    "Everything counted above comes from {scope}",
    "The tally is built on {scope}",
    "Figures below summarise {scope}",
    "What has been added up is {scope}",
    "The totals rest on {scope}",
)
#: Ten too, and for the same reason: this clause is the second half of every
#: scope_mismatch flaw sentence, so a pool smaller than the number of families
#: sharing that mechanism guarantees a shared n-gram.
COMP_FRAME = (
    "and {short} {pop} never did",
    "and {short} {pop} did not",
    "and {short} {pop} failed to",
    "and {short} {pop} are missing from that",
    "and {short} {pop} were left out",
    "and {short} {pop} fall outside it",
    "and {short} {pop} never made it in",
    "and {short} {pop} are absent",
    "and {short} {pop} were never captured",
    "and {short} {pop} did not make the set",
)
#: One unique proper noun per family. With pools of size 4-6 and twenty
#: families, collisions modulo the pool sizes are unavoidable - families 0 and
#: 10 landed on the same frame AND the same exposure. A per-family site name is
#: the only thing that is guaranteed distinct, and it also breaks any shared
#: 8-gram window that crosses it.
SITES = ("Ashgrove", "Brackhill", "Corbin", "Dunmore", "Eastcote", "Fenwick",
         "Garrow", "Hollin", "Ilford", "Jarnley", "Kestrel", "Larkhill",
         "Marbeck", "Netley", "Ockham", "Penhale", "Quarles", "Ravenhill",
         "Stanmer", "Thurlow")
#: Varied per family for the same reason: the shared case tail was producing
#: 8-gram collisions between every pair of broken_chronology items.
CASE_TAIL = ("against the prior window", "compared with the earlier run",
             "next to the previous window", "set against the last cycle",
             "relative to the earlier period", "measured on the earlier month",
             "beside the preceding window", "over the previous stretch",
             "as against the earlier pass", "on top of the last window")
#: The case sentence frame varies per family too. One shared frame left every
#: pair of broken_chronology items sharing 8-grams like "for one shipment
#: showed a" - and for that mechanism the case lines ARE the falsifying text.
CASE_FRAME = (
    "Record {tag} at {site} from {region} in {period} on the {device}, one "
    "{unit}, showed a {verb} of {n} {metric} {tail}.",
    "At {site} in {period}, {region} logged {tag} on the {device}: one {unit}, "
    "a {verb} of {n} {metric} {tail}.",
    "{tag} at {site}: a {unit} from {region}, {period}, running the {device}, "
    "moved by a {verb} of {n} {metric} {tail}.",
    "{site}: at {region} during {period} the {device} handled one {unit}; {tag} "
    "records a {verb} of {n} {metric} {tail}.",
    "One {unit} at {site} on the {device}, {region}, {period} - {tag} gives a "
    "{verb} of {n} {metric} {tail}.",
    "{site} logged {tag}: a {unit} out of {region}, {period}, on the {device}, "
    "a {verb} of {n} {metric} {tail}.",
    "From {region} at {site}, {period}: {tag} on the {device} for one {unit}, "
    "a {verb} of {n} {metric} {tail}.",
    "{tag} covers one {unit} at {site} in {region} during {period}; the "
    "{device} gave a {verb} of {n} {metric} {tail}.",
    "Entry {tag}, {site}: the {device} at {region} in {period} ran one {unit} "
    "to a {verb} of {n} {metric} {tail}.",
    "One {unit} filed as {tag} at {site}, {region}, {period}, on the {device}: "
    "a {verb} of {n} {metric} {tail}.",
)
TAIL = "no other change was made to any unit."


def family_clauses(index: int) -> dict[str, dict[str, str]]:
    """The four clause pairs for one synthetic family, varied by index."""
    # Every slot gets its OWN map of the family index. Sharing one index across
    # slots meant two families that collided on it collided on several clauses
    # at once - syn01 and syn05 differed by a single noun and scored 0.92.
    # A family's MECHANISM is decided by index % 2, so the ten families that
    # share a mechanism are index 0,2,4,... - and only those ten ever need
    # distinct flaw wording. Indexing the clause slots on k = index // 2 gives
    # those ten the pairs (0,0)...(4,1), so no two share BOTH the frame and the
    # exposure. Indexing on `index` itself made frame and exposure collide for
    # families five apart, which is where every remaining 0.7+ pair came from.
    k = index // 2
    a = index % len(NOISE)
    # EXPOSURE takes the fast index: it is the longest shared span in the
    # confound sentence, so holding it constant across five families produced a
    # shared 8-gram every time regardless of what else varied.
    # EXPOSURE, POP_FRAME and COMP_FRAME all have one entry per family that
    # shares a mechanism, so k indexes them directly and no two of those ten
    # ever repeat a clause.
    e = k % len(EXPOSURE)
    c = k % len(FRAME)
    f = k % len(POP_FRAME)
    m = k % len(COMP_FRAME)
    site = SITES[index % len(SITES)]
    noise = NOISE[a].capitalize()
    # All three keyed on k, not on index: families 1 and 13 share index % 6 and
    # so shared the population noun, which held them at 0.72 on its own.
    pop, verb, short = POP_NOUN[k % len(POP_NOUN)], report_verb(k), shortfall(k)
    single = pop[:-1]
    # The FRAME varies too, not just the words dropped into it. With a single
    # shared skeleton the pairs still scored 0.71 on nothing but a swapped noun.
    frame = FRAME[c]
    return {
        "FACT": {
            "changed": frame.format(
                noise=f"{noise} at {site}", rest=comparator(index)
            ),
            "same": frame.format(
                noise=f"{noise} at {site}",
                rest="held level with the earlier period",
            ),
        },
        "SCOPE": {
            "reach": f"and {EXPOSURE[e]}",
            "block": f"and {SHELTER[e]}",
        },
        "POPULATION": {
            "subset": POP_FRAME[f].format(
                scope=f"the {site} {pop} that {verb}"
            ),
            "whole": POP_FRAME[f].format(
                scope=f"every {site} {single} on the list"
            ),
        },
        # The verb phrase is NOT repeated here. Spelling it out twice in one
        # sentence doubled the text two families had in common and held pair
        # similarity at 0.71 however the slots were varied.
        "COMPLETENESS": {
            "complete": f"and every listed {single} did",
            "incomplete": COMP_FRAME[m].format(short=short, pop=pop),
        },
    }


#: coherent_false is the only cell where the confound both moved AND reached the
#: units. The scope families give BOTH false cells the same mechanism, so the
#: matched-mechanism subset - the confound-controlled primary endpoint - is
#: exercised end to end before the real items exist (D-024).
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
    k = index // 2  # see family_clauses: ten families share each mechanism
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
    for cell in CORE_CELLS:
        vals = vals_by_coherence["coherent" if cell.startswith("coherent") else "diverse"]

        cases: list[Case] = []
        for c in range(4):
            conditions = {d: vals[d][c] for d in DIMENSIONS}
            verb = VERBS[(index + c) % len(VERBS)]
            metric = METRICS[(index + c) % len(METRICS)]
            text = CASE_FRAME[k % len(CASE_FRAME)].format(
                tag=f"{family_id[4:]}{c + 1}",
                region=conditions["region"],
                period=conditions["period"],
                device=conditions["device"],
                unit=conditions["unit_kind"],
                verb=verb,
                n=12 + c * 3,
                metric=metric,
                site=SITES[index % len(SITES)],
                tail=CASE_TAIL[k % len(CASE_TAIL)],
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
        cl = family_clauses(index)
        midline = (
            f"{cl['FACT'][fact_key]}, {cl['SCOPE'][scope_key]}. "
            f"{cl['POPULATION'][pop_key]}, {cl['COMPLETENESS'][comp_key]}."
        )
        closer = TAIL
        lead = (
            f"Log {family_id[4:]}{CORE_CELLS.index(cell)} records four observations "
            f"of {subject} taken after the {change} was introduced."
        )
        # The 7-line layout of the real items: lead, two cases, the truth-bearing
        # mid-passage line, two more cases, closer. This line used to be built
        # and then dropped on the floor, so every synthetic passage went out with
        # no falsifying text in it at all, and every gate that reads the
        # mid-passage line was never exercised end to end before the real items
        # existed - which is the whole point of the synthetic set.
        texts = [c.text for c in cases]
        passage = "\n".join(
            [lead, texts[0], texts[1], midline, texts[2], texts[3], closer]
        )

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
