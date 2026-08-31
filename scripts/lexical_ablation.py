"""Where does the lexical giveaway live?

`src/validate.py` tells you an n-gram classifier can separate TRUE from FALSE.
It does not tell you WHICH part of the passage leaks. This splits each passage
into its three structural parts and re-runs the same grouped-CV classifier on
each in isolation:

  full            the whole passage (what validate.py scores)
  lead            first line only
  cases           the four case sentences only
  closer          last line only
  no_closer       lead + cases, closer removed
  no_dates        full passage with every date token masked

Read it as an attribution. If `closer` alone is near the full accuracy, the fix
is to rewrite closers. If `no_dates` drops a long way below `full`, the temporal
flaw is leaking through date tokens and the fix is to reuse the SAME date tokens
in the flawed item, merely exchanged between positions.

    python scripts/lexical_ablation.py --items items/draft items/seed
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import Item, load_items  # noqa: E402

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)
DATE_RE = re.compile(rf"\b\d{{1,2}}\s+({MONTHS})\b|\b({MONTHS})\b", re.I)
NUM_RE = re.compile(r"\b\d+\b")


def parts(item: Item) -> dict[str, str]:
    lines = [ln for ln in item.passage.split("\n") if ln.strip()]
    lead = lines[0] if lines else ""
    closer = lines[-1] if len(lines) > 1 else ""
    cases = "\n".join(lines[1:-1]) if len(lines) > 2 else ""
    return {
        "full": item.passage,
        "lead": lead,
        "cases": cases,
        "closer": closer,
        "no_closer": "\n".join(lines[:-1]),
        "no_dates": DATE_RE.sub(" DATE ", item.passage),
        "no_dates_no_numbers": NUM_RE.sub(" NUM ", DATE_RE.sub(" DATE ", item.passage)),
    }


def accuracy(texts: list[str], y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    if not any(t.strip() for t in texts):
        return float("nan")
    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)),
            ("clf", LogisticRegression(max_iter=5000, random_state=seed)),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(pipe, texts, y, cv=cv, groups=groups)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    items = load_items(args.items)
    y = np.array([int(i.ground_truth) for i in items])
    groups = np.array([i.family_id for i in items])
    split = [parts(i) for i in items]

    print(f"{len(items)} items, {len(set(groups))} families, chance = 50.0%")
    print(f"grouped 5-fold CV accuracy at telling TRUE from FALSE "
          f"(limit {args.threshold:.0%})\n")
    print(f"  {'part':<22}{'accuracy':>10}")
    print(f"  {'-' * 32}")
    for key in split[0]:
        acc = accuracy([s[key] for s in split], y, groups, args.seed)
        flag = "  <-- OVER LIMIT" if acc > args.threshold else ""
        print(f"  {key:<22}{acc:>9.1%}{flag}")

    print(
        "\nreading it:\n"
        "  closer ~= full        -> rewrite the closing sentences; the TRUE and\n"
        "                           coherent_false closers are too easy to tell apart\n"
        "  full >> no_dates      -> the temporal flaw leaks through date tokens; reuse\n"
        "                           the SAME tokens in the flawed item, exchanged\n"
        "                           between positions rather than replaced\n"
        "  cases ~= full         -> the claim_mismatch substitute wording repeats\n"
        "                           across families; make each family's substitute\n"
        "                           quantity idiosyncratic"
    )

    # Cell-level detail: which FALSE cell is actually detectable?
    print("\nper-cell detectability (one-vs-rest on the full passage):")
    texts = [s["full"] for s in split]
    for cell in ("coherent_false", "diverse_false", "coherent_true", "diverse_true"):
        yc = np.array([int(i.cell == cell) for i in items])
        if yc.sum() in (0, len(yc)):
            continue
        print(f"  {cell:<18}{accuracy(texts, yc, groups, args.seed):>9.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
