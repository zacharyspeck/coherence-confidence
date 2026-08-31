"""Is a run still valid against the items currently on disk?

A run records `prompts_hash`, a digest of the exact prompt text the model saw.
Recomputing it from the working tree answers the one question that decides
whether a stored result can still be quoted: were the items edited after the run?

    python scripts/verify_run.py results/run_base135m_realitems.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import provenance  # noqa: E402
from src.models import load_items  # noqa: E402
from src.render import DEFAULT_OPTIONS, render_baseline_prompt, render_prompt  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    meta = payload["meta"]
    stored = meta.get("prompts_hash")
    if not stored:
        print("run predates prompts_hash; cannot verify. Re-run it.", file=sys.stderr)
        return 2

    options = tuple(meta.get("option_words") or DEFAULT_OPTIONS)
    items = load_items(args.items)
    if payload.get("kind") == "baseline":
        seen: dict[str, str] = {}
        for i in items:
            seen.setdefault(i.family_id, i.claim)
        prompts = [render_baseline_prompt(c, options) for c in seen.values()]
    else:
        prompts = [render_prompt(i, options) for i in items]
    current = provenance.hash_prompts(prompts)

    print(f"run:     {args.run}")
    print(f"model:   {meta.get('model')}  options {meta.get('option_words')}")
    print(f"stored:  {stored}")
    print(f"current: {current}")
    if stored == current:
        print("\nVALID: the items on disk are byte-identical to what this run scored.")
        return 0
    print(
        "\nSTALE: the items have changed since this run. Its numbers describe text "
        "that no longer exists in the repo. Re-run before quoting them.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
