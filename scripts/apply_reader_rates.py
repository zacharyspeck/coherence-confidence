"""Write measured reader rates from the reader audit back onto the items.

`reader_catch_rate` is the fraction of plain readers - shown only the scored
prompt, never told a flaw might exist - who answered No to a FALSE item.
`reader_false_positive_rate` is the same fraction on a TRUE item, where No is
wrong. Exactly one of the two is set per item; the other is null, because the
question each answers only exists on one side of the split.

These are carried on the item so `src/analyze.py` can condition on them and so a
later reader can see which measurement the number came from. They replace
`salience` as the covariate: salience is rated by a hunter that already found
the flaw, and over the whole 100-item set it spans 2.0 to 3.0, so there is
almost nothing there to condition on (D-032).

    python scripts/apply_reader_rates.py            # write
    python scripts/apply_reader_rates.py --check    # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", default="results/reader_audit.json")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    rows = doc["items"]

    written = 0
    missing: list[str] = []
    stale: list[str] = []
    for d in ("items/draft", "items/seed"):
        for path in sorted(Path(d).glob("fam_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            dirty = False
            for it in data["items"]:
                row = rows.get(it["id"])
                if row is None or not row.get("n_reads"):
                    missing.append(it["id"])
                    continue
                if it["ground_truth"]:
                    want = (None, row["reader_false_positive_rate"])
                else:
                    want = (row["reader_catch_rate"], None)
                got = (it.get("reader_catch_rate"), it.get("reader_false_positive_rate"))
                if got != want:
                    it["reader_catch_rate"], it["reader_false_positive_rate"] = want
                    dirty = True
                    written += 1
            if dirty and not args.check:
                path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    print(f"{'would set' if args.check else 'set'} reader rates on {written} items")
    if missing:
        print(f"WARNING: {len(missing)} items have no reader reads: {missing[:8]}")
        return 1
    if stale:
        print(f"WARNING: {len(stale)} stale: {stale[:8]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
