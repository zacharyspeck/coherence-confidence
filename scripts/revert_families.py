"""Revert named families to an earlier round's scope clauses.

The rule, from the fix-pass brief: *every FALSE item must still be 100% findable
in the blind audit; if quieting an item makes it unfindable, revert it and flag
it.* Findable-but-quiet is the target; unfindable is a broken item.

This applies that rule selectively, per family, so one round's regression does
not cost the whole round's gains. Reverted families are listed on stdout and
recorded in `results/reverted_families.json` so `FIX_REPORT.md` can name them.

    python scripts/revert_families.py fam_a fam_b --from results/scope_clauses_round1.json
    python scripts/revert_families.py --auto   # revert whatever the audit says is unfindable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLAUSE_KEYS = (
    "scope_population_subset",
    "scope_population_whole",
    "scope_completeness_complete",
    "scope_completeness_incomplete",
    "note_scope_coherent",
    "note_scope_diverse",
)


def unfindable_families(audit_json: Path) -> dict[str, list[str]]:
    """Families with at least one FALSE item no auditor could identify."""
    rows = json.loads(audit_json.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for r in rows.values():
        if r["ground_truth"]:
            continue
        if r["bucket"] in ("missed", "missed_vague"):
            out.setdefault(r["family_id"], []).append(r["item_id"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("families", nargs="*")
    ap.add_argument("--from", dest="src", default="results/scope_clauses_round1.json")
    ap.add_argument("--into", default="results/scope_clauses.json")
    ap.add_argument("--audit", default="results/item_audit.json")
    ap.add_argument(
        "--auto",
        action="store_true",
        help="revert every family with an unfindable FALSE item, per the audit",
    )
    ap.add_argument("--out", default="results/reverted_families.json")
    args = ap.parse_args(argv)

    src = json.loads(Path(args.src).read_text(encoding="utf-8"))
    cur = json.loads(Path(args.into).read_text(encoding="utf-8"))

    reasons: dict[str, list[str]] = {}
    names = list(args.families)
    if args.auto:
        found = unfindable_families(Path(args.audit))
        reasons.update(found)
        names += [f for f in found if f not in names]

    missing = [f for f in names if f not in src]
    if missing:
        raise SystemExit(f"no earlier clauses for: {missing}")

    for fam in names:
        for k in CLAUSE_KEYS:
            if k in src[fam]:
                cur[fam][k] = src[fam][k]
        cur[fam]["round"] = src[fam].get("round", 1)
        cur[fam]["reverted"] = True
        cur[fam]["revert_reason"] = (
            "quieting made "
            + ", ".join(reasons.get(fam, ["a FALSE item"]))
            + " unfindable in the blind audit"
        )

    Path(args.into).write_text(
        json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(args.out).write_text(
        json.dumps(
            {
                "reverted": sorted(names),
                "source": args.src,
                "unfindable_items": reasons,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if not names:
        print("nothing to revert - every FALSE item is still findable")
        return 0
    print(f"reverted {len(names)} families to {args.src}:")
    for f in sorted(names):
        why = reasons.get(f)
        print(f"  {f}" + (f"  ({', '.join(why)} unfindable)" if why else ""))
    print(f"\nwrote {args.out}")
    print("now re-run: python scripts/assemble_passages.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
