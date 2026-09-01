"""Write a fresh blinded review set for a human reviewer.

One item per family, from families the reviewer has not already seen, shown
exactly as a reader sees it: claim, passage, question, options. Nothing else -
no cell, no ground truth, no mechanism, no salience, no id, and the order is
shuffled so position carries no information.

The answer key goes to a SEPARATE tree, for the same reason the audit key does:
blindness should not depend on the reviewer choosing not to scroll.

    python scripts/make_blind_review.py --exclude fam_fertilizer fam_checkout
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import load_items  # noqa: E402
from src.render import DEFAULT_OPTIONS, render_prompt  # noqa: E402

BANNED = (
    "coherent", "diverse", "decorative", "ground_truth", "salience",
    "flaw_mechanism", "review_status", "confound", "scope_variant",
    "reader_catch_rate", "invisible",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--exclude", nargs="*", default=["fam_fertilizer", "fam_checkout"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="review/blind_items_2.md")
    ap.add_argument("--key-out", default="results/review_key/blind_items_2.json")
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args(argv)

    items = [i for i in load_items(args.items) if i.family_id not in args.exclude]
    by_family: dict[str, list] = {}
    for i in items:
        by_family.setdefault(i.family_id, []).append(i)

    rng = random.Random(args.seed)
    fams = sorted(by_family)
    rng.shuffle(fams)
    fams = sorted(fams[: args.n])

    # Balance TRUE against FALSE across the chosen families rather than letting
    # the draw decide. A reviewer who meets eight FALSE items in ten learns the
    # base rate and the false-positive rate stops meaning anything.
    picked = []
    want_true = args.n // 2
    for n, fam in enumerate(fams):
        pool = by_family[fam]
        want = (n < want_true)
        cands = [i for i in pool if i.ground_truth is want] or pool
        picked.append(rng.choice(sorted(cands, key=lambda i: i.id)))
    rng.shuffle(picked)

    lines = [
        "# Blind review set 2",
        "",
        f"{len(picked)} items, one per family, from families not in the first "
        "set. Claim, passage and question only - every other field is withheld, "
        "and the order is shuffled so position gives nothing away.",
        "",
        "For each, answer **Yes**, **No** or **Unsure**, and rate how hard it "
        "was to decide from 1 (obvious) to 5 (very hard).",
        "",
        "The scored prompt wraps each item exactly as shown.",
        "",
        "---",
        "",
    ]
    key = {}
    for n, it in enumerate(picked, 1):
        key[str(n)] = {
            "item_id": it.id,
            "family_id": it.family_id,
            "cell": it.cell,
            "ground_truth": it.ground_truth,
            "flaw_mechanism": it.flaw_mechanism,
            "reader_catch_rate": it.reader_catch_rate,
            "reader_false_positive_rate": it.reader_false_positive_rate,
            "salience": it.salience,
            "confound_note": it.confound_note,
        }
        lines += ["## " + str(n), "", "```", render_prompt(it), "```", "",
                  "Your answer: ______   Difficulty (1-5): ______", "", "---", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    kp = Path(args.key_out)
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")

    text = out.read_text(encoding="utf-8").lower()
    leaked = [b for b in BANNED if b in text]
    if leaked:
        raise SystemExit(f"BLIND FILE LEAKED: {leaked}")

    n_true = sum(1 for v in key.values() if v["ground_truth"])
    print(f"wrote {out}: {len(picked)} items, {n_true} TRUE / {len(picked)-n_true} FALSE")
    print(f"  families: {sorted({v['family_id'] for v in key.values()})}")
    print(f"  cells:    {sorted({v['cell'] for v in key.values()})}")
    print(f"  key (do not read before reviewing): {kp}")
    print("  verified: no cell labels, no answer key, no ids in the blind file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
