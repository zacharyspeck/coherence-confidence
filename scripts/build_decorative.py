"""Build the decorative control arm (D-030).

The objection this answers: a diverse passage names four countries, four devices
and four months where a coherent one names one of each, so a confidence
difference might come from parse load rather than from the evidence being
evidentially independent.

A decorative item splits those two apart. Its four cases carry the SAME condition
values as a coherent item - one region, one season, one population, so the
evidence is exactly as dependent - while each case carries four DISTINCT
decorations: a log serial, a clock time within the same day, a terminal, a desk.
Entity count and surface busyness land on the diverse cells; evidential structure
lands on the coherent cells.

Whichever the model's confidence follows is the answer. Nothing else in the item
changes: same family, same claim, same lead, same mid-passage line, same closer.

    python scripts/build_decorative.py
    python scripts/build_decorative.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.complexity import surface_complexity  # noqa: E402
from src.midline import build_midline
from src.models import compute_word_count, load_families  # noqa: E402

SPEC = Path("results/family_spec.json")
CLAUSES = Path("results/scope_clauses.json")

#: Deliberately clerical, and deliberately NOT people. Many families already
#: carry a person as a CONDITION - observer, auditor, technician, tutor - held
#: constant across the four cases. A varying person decoration would contradict
#: that outright. Serial, time, terminal and desk cannot be confused for
#: anything the claim generalises over.
DECORATION_DIMENSIONS = ("log_serial", "clock_time", "terminal", "desk")

#: One letter per family, so serials do not repeat across families.
SERIAL_PREFIX = "KRTMPDGLNW"

#: Four clock times inside a SINGLE day. Same day is the point - a different time
#: of day rules nothing out, so it buys no evidential independence at all.
CLOCK_BASE = (915, 1040, 1325, 1550)


#: Three of the four decorations are entity-like (they carry digits or a code),
#: which is what lifts the entity count onto the diverse cells. The fourth is a
#: plain lowercase word, which lifts distinct-token count without over-shooting
#: entity count. The exact split was chosen by measuring, not by taste.
TERMINALS = ("T7", "T8", "T9", "T10")
DESKS = ("north", "south", "east", "west")

#: decorative_false is falsified the same way as the matched subset, so the
#: control is directly comparable to it. decorative_true alternates between the
#: two TRUE-compatible scope variants so the variant is not confounded with the
#: cell (the same reasoning as D-026's rotation).
FALSE_ASSIGN = (("same", "block"), ("subset", "incomplete"), "scope_mismatch")
TRUE_ASSIGN = (
    (("same", "block"), ("subset", "complete"), None),
    (("same", "block"), ("whole", "incomplete"), None),
)


def item_path(family_id: str) -> Path:
    for d in ("items/draft", "items/seed"):
        p = Path(d) / f"{family_id}.json"
        if p.exists():
            return p
    raise SystemExit(f"no file for {family_id}")


def decorations_for(index: int, k: int) -> dict[str, str]:
    """The four decorations for case k of family `index`."""
    return {
        "log_serial": f"{SERIAL_PREFIX[index % len(SERIAL_PREFIX)]}-{3310 + index * 10 + k}",
        "clock_time": f"{CLOCK_BASE[k] + index:04d}",
        "terminal": TERMINALS[k],
        "desk": DESKS[k],
    }


def decorate(text: str, dec: dict[str, str]) -> str:
    """Append the decoration tag to a case sentence.

    Deliberately terse - four words. These passages are logs, so a trailing tag
    of serial, time, operator and desk reads naturally, and every extra word here
    pushes the cell's mean word count away from the grand mean (the gate is 10%).
    """
    body = text.rstrip()
    if body.endswith("."):
        body = body[:-1]
    return (
        f"{body}; {dec['log_serial']}, {dec['clock_time']}, "
        f"{dec['terminal']}, {dec['desk']}."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    clauses = json.loads(CLAUSES.read_text(encoding="utf-8"))

    # Purge any existing control arm from the raw JSON BEFORE loading through the
    # model. A previous run that wrote something the validator now rejects would
    # otherwise leave this script unable to fix its own output.
    purged = 0
    for d in ("items/draft", "items/seed"):
        for path in sorted(Path(d).glob("fam_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            keep = [i for i in data["items"] if not i["cell"].startswith("decorative")]
            if len(keep) != len(data["items"]):
                purged += len(data["items"]) - len(keep)
                data["items"] = keep
                data.pop("decoration_dimensions", None)
                if not args.check:
                    path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False) + chr(10),
                        encoding="utf-8",
                    )
    if purged:
        print(f"purged {purged} existing decorative item(s) before rebuilding")

    families = load_families(["items/draft", "items/seed"])

    targets = [
        f
        for f in families
        if any(
            i.cell == "diverse_false" and i.flaw_mechanism == "scope_mismatch"
            for i in f.items
        )
    ]
    print(f"{len(targets)} scope_mismatch families -> {len(targets) * 2} control items")

    built = 0
    for index, fam in enumerate(sorted(targets, key=lambda f: f.family_id)):
        sp, cl = spec[fam.family_id], clauses[fam.family_id]
        base = next(i for i in fam.items if i.cell == "coherent_true")

        cases = []
        for k, c in enumerate(base.cases):
            dec = decorations_for(index, k)
            cases.append(
                {
                    "case_id": c.case_id,
                    "text": decorate(c.text, dec),
                    "conditions": dict(c.conditions),
                    "decorations": dec,
                }
            )

        path = item_path(fam.family_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["decoration_dimensions"] = list(DECORATION_DIMENSIONS)
        data["items"] = [i for i in data["items"] if not i["cell"].startswith("decorative")]

        for cell, (confound, scope, mech) in (
            ("decorative_true", TRUE_ASSIGN[index % 2]),
            ("decorative_false", FALSE_ASSIGN),
        ):
            midline = build_midline(sp, cl, confound, scope)
            texts = [c["text"] for c in cases]
            passage = "\n".join(
                [sp["lead"], texts[0], texts[1], midline, texts[2], texts[3],
                 cl["neutral_closer"]]
            )
            note = None
            if mech:
                note = (
                    cl.get("note_scope_coherent", "")
                    + " The four cases differ only in log serial, clock time, "
                    "terminal and desk - surface detail that rules nothing out, so "
                    "the evidence is exactly as dependent as the coherent version."
                ).strip()
            data["items"].append(
                {
                    "id": f"{fam.family_id}__{cell}",
                    "family_id": fam.family_id,
                    "cell": cell,
                    "claim": fam.claim,
                    "cases": json.loads(json.dumps(cases)),
                    "passage": passage,
                    "ground_truth": cell.endswith("_true"),
                    "confound_note": note,
                    "word_count": compute_word_count(passage),
                    "flaw_mechanism": mech,
                    "confound_variant": f"{confound[0]}_{confound[1]}",
                    "scope_variant": f"{scope[0]}_{scope[1]}",
                    "salience": None,
                    "surface_complexity": surface_complexity(
                        passage,
                        [c["conditions"] for c in cases],
                        [c["decorations"] for c in cases],
                    ),
                    "domain": fam.domain,
                    "review_status": "unreviewed",
                    "source": "generated",
                    "notes": (
                        "CONTROL ARM (D-030). Condition values are identical across "
                        "all four cases, exactly as in the coherent cells. The "
                        "surface variety is decorations only - serial, time, "
                        "terminal, desk - which confer no evidential independence."
                    ),
                }
            )
            built += 1

        order = {c: n for n, c in enumerate(
            ["coherent_true", "coherent_false", "diverse_true", "diverse_false",
             "decorative_true", "decorative_false"])}
        data["items"].sort(key=lambda i: order[i["cell"]])

        if not args.check:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    print(f"{'would build' if args.check else 'built'} {built} decorative items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
