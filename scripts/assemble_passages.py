"""Rebuild every passage from its clause parts. The single source of truth for
passage LAYOUT (fix-pass steps 1 and 2).

Layout, for all 80 items:

    line 1  lead sentence            identical across the family
    line 2  case 1
    line 3  case 2
    line 4  MID-PASSAGE FACTS        everything truth-bearing lives here
    line 5  case 3
    line 6  case 4
    line 7  neutral closer           identical across the family

Two things about that layout are deliberate.

**The flaw is buried.** It sits mid-passage with two cases and a closer after it,
never beside the conclusion and never last. The previous build put the confound
in the final sentence and the blind audit rated it 3.90/5 for obviousness against
2.70 for the diverse cells; position is the main lever on that.

**The mid-passage line carries TWO independent clause pairs**, and exactly one
combination of each pair falsifies an item:

    CONFOUND   changed/same  x  reach/block     -> `changed_reach` is live
    SCOPE      subset/whole  x  incomplete/complete -> `subset_incomplete` is live

A family chooses which pair is live. That is what lets `scope_mismatch` appear in
BOTH false cells (D-024), which is what the matched-mechanism endpoint needs, and
it keeps every clause used on both sides of the TRUE/FALSE split within a family
except where that is mathematically impossible (see D-026).

    python scripts/assemble_passages.py            # rebuild
    python scripts/assemble_passages.py --check    # report, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.complexity import surface_complexity
from src.midline import build_midline
from src.models import CORE_CELLS, compute_word_count  # noqa: E402

SPEC = Path("results/family_spec.json")
CLAUSES = Path("results/scope_clauses.json")

#: (confound fact, confound scope), (population, completeness), flaw_mechanism
CONFOUND_FAMILY = {
    "coherent_false": (("changed", "reach"), ("whole", "complete"), "stated_confound"),
    "coherent_true": (("changed", "block"), ("subset", "complete"), None),
    "diverse_true": (("same", "reach"), ("whole", "incomplete"), None),
    "diverse_false": (("same", "block"), ("whole", "complete"), "broken_chronology"),
}
SCOPE_FAMILY = {
    "coherent_false": (("changed", "block"), ("subset", "incomplete"), "scope_mismatch"),
    "diverse_false": (("same", "reach"), ("subset", "incomplete"), "scope_mismatch"),
    "coherent_true": (("same", "reach"), ("subset", "complete"), None),
    "diverse_true": (("changed", "block"), ("whole", "incomplete"), None),
}

#: Swapping the two TRUE items' SCOPE clauses keeps both of them true and keeps
#: every count identical, so it is free - and it stops scope_variant from being
#: perfectly correlated with coherence. Applied to every other family.
def rotate_true_scope(assign: dict) -> dict:
    out = dict(assign)
    ct, dt = out["coherent_true"], out["diverse_true"]
    out["coherent_true"] = (ct[0], dt[1], ct[2])
    out["diverse_true"] = (dt[0], ct[1], dt[2])
    return out


CAUSAL_WORDS = (
    "because", "due to", "which explains", "therefore", "as a result",
    "consequently", "hence", "thus ",
)
HEDGE_WORDS = ("however", "although", "notably", "it should be noted", "surprisingly")

#: Imputation language in the WHOLE population clause is what made four TRUE
#: items read as false in the round-1 audit - an auditor reasonably objects that
#: a quarter of the numbers were invented. A TRUE item that reads as false
#: depresses confidence on the very cell the hypothesis says is inflated, which
#: biases the headline result TOWARDS the hypothesis. Banned outright.
#:
#: Deliberately blunt. It will occasionally flag a harmless phrase in a neutral
#: sentence and force a reword; that costs a minute. A missed imputation costs a
#: result that points the wrong way and looks clean.
IMPUTATION_WORDS = (
    "counted as", "entered as", "recorded as", "treated as", "scored as",
    "imputed", "substituted", "counted at", "entered at", "as nil",
    "as zero", "as fails", "as failures",
)


def item_path(family_id: str) -> Path:
    for d in ("items/draft", "items/seed"):
        p = Path(d) / f"{family_id}.json"
        if p.exists():
            return p
    raise SystemExit(f"no file for {family_id}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    clauses = json.loads(CLAUSES.read_text(encoding="utf-8"))

    missing = sorted(set(spec) - set(clauses))
    if missing:
        raise SystemExit(f"no authored clauses for: {missing}")

    problems: list[str] = []
    rebuilt: list[str] = []
    changed = 0
    for family_id, sp in sorted(spec.items()):
        cl = clauses[family_id]
        assign = SCOPE_FAMILY if sp["family_type"] == "scope" else CONFOUND_FAMILY
        if sorted(spec).index(family_id) % 2 == 1:
            assign = rotate_true_scope(assign)
        path = item_path(family_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        by_cell = {i["cell"]: i for i in data["items"]}

        # In a scope family the falsification moved OUT of the cases and into the
        # mid-passage line, so diverse_false's cases go back to matching
        # diverse_true's. Otherwise the old wrong-quantity wording would sit there
        # as a second, undeclared flaw.
        if sp["family_type"] == "scope":
            by_cell["diverse_false"]["cases"] = json.loads(
                json.dumps(by_cell["diverse_true"]["cases"])
            )

        # Only the 2x2. The control arm is built by scripts/build_decorative.py,
        # which reads coherent_true AFTER this has run, so run them in that order.
        for cell in CORE_CELLS:
            it = by_cell[cell]
            confound, scope, mech = assign[cell]
            midline = build_midline(sp, cl, confound, scope)
            cases = [c["text"] for c in it["cases"]]
            passage = "\n".join(
                [sp["lead"], cases[0], cases[1], midline, cases[2], cases[3],
                 cl["neutral_closer"]]
            )

            low = passage.lower()
            for w in CAUSAL_WORDS + HEDGE_WORDS:
                if w in low:
                    problems.append(f"{it['id']}: passage contains '{w.strip()}'")
            for w in IMPUTATION_WORDS:
                if w in low:
                    problems.append(
                        f"{it['id']}: imputation language '{w}' - a TRUE item that "
                        "invents values reads as false, which biases the result "
                        "toward the hypothesis"
                    )

            passage_changed = it.get("passage") != passage
            it["passage"] = passage
            it["word_count"] = compute_word_count(passage)
            # Refreshed here rather than in a later step: the Item model checks
            # it against the passage, so leaving it stale would stop the next
            # script in the chain from loading the items at all.
            it["surface_complexity"] = surface_complexity(
                passage,
                [c["conditions"] for c in it["cases"]],
                [c.get("decorations", {}) for c in it["cases"]],
            )
            it["flaw_mechanism"] = mech
            it["confound_variant"] = f"{confound[0]}_{confound[1]}"
            it["scope_variant"] = f"{scope[0]}_{scope[1]}"
            # Salience and reader_catch_rate describe how hard THIS passage's
            # flaw was to spot, so they survive exactly as long as the passage
            # does. Clearing them unconditionally meant that rebuilding one
            # family threw away the audit for the other nineteen; clearing them
            # on change keeps the invalidation honest and no wider than it has
            # to be.
            if passage_changed:
                it["salience"] = None
                it["reader_catch_rate"] = None
                it["reader_false_positive_rate"] = None
                rebuilt.append(it["id"])

            if mech == "scope_mismatch":
                key = "note_scope_coherent" if cell == "coherent_false" else "note_scope_diverse"
                if cl.get(key):
                    it["confound_note"] = cl[key]
            elif mech == "stated_confound" and cl.get("note_stated_confound"):
                it["confound_note"] = cl["note_stated_confound"]
            if mech is None:
                it["confound_note"] = None
            changed += 1

        if not args.check:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    print(f"{'checked' if args.check else 'rebuilt'} {changed} passages "
          f"across {len(spec)} families")
    print(f"{len(rebuilt)} passages actually CHANGED; their salience and reader "
          f"rates were cleared" + (f": {rebuilt[:6]}" if rebuilt else ""))
    if problems:
        print(f"\n{len(problems)} wording problems:")
        for p in problems[:30]:
            print("  ", p)
        return 1
    print("no causal, hedging or imputation language found in any passage")
    if rebuilt:
        print(
            chr(10) + f'salience and reader rates cleared on {len(rebuilt)} changed '
            'items. Re-run the hunter audit and the reader audit for those, '
            'then apply_salience.py / apply_reader_rates.py, before analyzing.'
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
