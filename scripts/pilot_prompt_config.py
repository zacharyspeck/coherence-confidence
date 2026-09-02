"""Pick the prompt configuration on COVERAGE, before anyone looks at an AUC.

The pilot showed mean `mass_covered` of 0.29 with a minimum of 0.037 under
`--chat-template`, and an argmax of `'Un'` on some items: the model wants to
write "Unsure" with no leading space, which is two tokens on this vocabulary
and therefore contributes nothing to the abstention mass. Under-counting
abstention inflates `p_yes_3way`, and a three-way renormalisation over 3% of
the distribution is the "ratio of rounding errors" failure the harness exists
to prevent (README failure mode 3).

So: score the same items under chat-template ON and OFF, in ONE process with
ONE model load, and choose on `mass_covered` alone.

**The decision rule is fixed here, in writing, before the numbers exist:**
take the configuration with the higher mean `mass_covered`; break a tie under
0.02 by the higher minimum. AUC is not computed here and must not enter the
choice - picking a prompt because it produced a friendlier endpoint would be
the one move that invalidates the result.

    python scripts/pilot_prompt_config.py --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import load_items  # noqa: E402
from src.render import DEFAULT_OPTIONS, render_prompt  # noqa: E402
from src.score import HFScorer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/prompt_config_pilot.json")
    args = ap.parse_args(argv)

    items = load_items(args.items)
    print(f"{len(items)} items", file=sys.stderr)

    t0 = time.time()
    scorer = HFScorer(
        args.model, device="cpu", dtype=args.dtype, chat_template=False
    )
    print(f"model loaded in {time.time() - t0:.0f}s", file=sys.stderr)

    out: dict[str, dict] = {}
    for label, use_template in (("plain", False), ("chat_template", True)):
        scorer.chat_template = use_template
        rows = []
        t1 = time.time()
        for n, it in enumerate(items, 1):
            res = scorer.score_prompt(render_prompt(it, DEFAULT_OPTIONS))
            rows.append(
                {
                    "item_id": it.id,
                    "cell": it.cell,
                    "mass_covered": res.mass_covered,
                    "top_token": res.top_token,
                    "top_token_prob": res.top_token_prob,
                    "abstained": res.abstained,
                }
            )
            if n % 4 == 0:
                print(f"  {label}: {n}/{len(items)}", file=sys.stderr)
        mass = [r["mass_covered"] for r in rows]
        out[label] = {
            "mean_mass_covered": st.mean(mass),
            "min_mass_covered": min(mass),
            "median_mass_covered": st.median(mass),
            "n_abstained": sum(1 for r in rows if r["abstained"]),
            "distinct_top_tokens": sorted({r["top_token"] for r in rows}),
            "seconds": round(time.time() - t1, 1),
            "seconds_per_item": round((time.time() - t1) / len(items), 1),
            "rows": rows,
        }
        print(
            f"{label}: mean mass {out[label]['mean_mass_covered']:.4f} "
            f"min {out[label]['min_mass_covered']:.4f} "
            f"({out[label]['seconds_per_item']}s/item)",
            file=sys.stderr,
        )

    a, b = out["plain"], out["chat_template"]
    if abs(a["mean_mass_covered"] - b["mean_mass_covered"]) < 0.02:
        winner = "plain" if a["min_mass_covered"] >= b["min_mass_covered"] else "chat_template"
        basis = "tie on mean within 0.02, decided on minimum"
    else:
        winner = "plain" if a["mean_mass_covered"] > b["mean_mass_covered"] else "chat_template"
        basis = "higher mean mass_covered"

    doc = {
        "kind": "prompt_config_pilot",
        "model": args.model,
        "decision_rule": (
            "higher mean mass_covered; tie under 0.02 broken on the minimum. "
            "Fixed in the script docstring before the numbers existed. AUC was "
            "not computed and did not enter the choice."
        ),
        "winner": winner,
        "basis": basis,
        "configs": out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"\nWINNER: {winner} ({basis})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
