"""Write measured salience from the blind audit back onto the items.

`salience` is the mean blind-audit explicitness (1-5) for a FALSE item: how hard
the flaw was to spot for auditors who saw only the passage. It is carried on the
item so `src/analyze.py` can report it beside every AUC and condition on it, and
so a later reader can see which items the number came from.

TRUE items get null - there is no flaw to rate.

    python scripts/apply_salience.py                    # write
    python scripts/apply_salience.py --check            # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", default="results/item_audit.json")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    rows = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    by_item = {
        r["item_id"]: r["mean_explicitness"]
        for r in rows.values()
        if r.get("mean_explicitness") is not None
    }

    written = 0
    missing: list[str] = []
    for d in ("items/draft", "items/seed"):
        for path in sorted(Path(d).glob("fam_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            dirty = False
            for it in data["items"]:
                want = None if it["ground_truth"] else by_item.get(it["id"])
                if not it["ground_truth"] and want is None:
                    missing.append(it["id"])
                if it.get("salience") != want:
                    it["salience"] = want
                    dirty = True
                    written += 1
            if dirty and not args.check:
                path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    print(f"{'would set' if args.check else 'set'} salience on {written} items")
    if missing:
        print(
            f"WARNING: {len(missing)} FALSE items have no audited salience "
            f"(their flaw was never matched to the key): {missing[:8]}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
