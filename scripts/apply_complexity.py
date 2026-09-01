"""Write the surface_complexity covariate onto every item.

It is a pure function of the passage, so this is a refresh rather than an
annotation: run it after any passage edit and the Item model will stop
complaining. The model checks the stored value against a recomputation at load,
so a stale number is an error rather than a quiet inconsistency.

    python scripts/apply_complexity.py
    python scripts/apply_complexity.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.complexity import surface_complexity  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    changed = 0
    for d in args.items:
        for path in sorted(Path(d).glob("fam_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            dirty = False
            for it in data["items"]:
                want = surface_complexity(
                    it["passage"],
                    [c["conditions"] for c in it["cases"]],
                    [c.get("decorations", {}) for c in it["cases"]],
                )
                if it.get("surface_complexity") != want:
                    it["surface_complexity"] = want
                    dirty = True
                    changed += 1
            if dirty and not args.check:
                path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    if args.check:
        print(f"{changed} item(s) have a stale surface_complexity")
        return 1 if changed else 0
    print(f"wrote surface_complexity on {changed} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
