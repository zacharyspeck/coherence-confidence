"""Recompute `word_count` for every item in a family file, in place.

The Item model refuses to load an item whose stored word_count disagrees with
its passage. That is deliberate: it catches a passage edit that forgot the
count. This script is the fix-up for when the edit was intentional.

    python scripts/recount_words.py items/draft/*.json
    python scripts/recount_words.py --check items/draft items/seed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_WS = re.compile(r"\s+")


def word_count(passage: str) -> int:
    return len(_WS.sub(" ", passage).strip().split())


def expand(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob("fam_*.json")))
        else:
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="family JSON files or directories")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report mismatches and exit 1; do not write",
    )
    args = ap.parse_args(argv)

    changed = 0
    for path in expand(args.paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        dirty = False
        for item in data.get("items", []):
            actual = word_count(item["passage"])
            if item.get("word_count") != actual:
                print(f"{path.name}:{item['id']}  {item.get('word_count')} -> {actual}")
                item["word_count"] = actual
                dirty = True
        if dirty:
            changed += 1
            if not args.check:
                path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    if args.check:
        if changed:
            print(f"FAIL: {changed} file(s) have stale word_count", file=sys.stderr)
            return 1
        print("OK: all word_count values current")
        return 0

    print(f"rewrote {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
