"""Baseline: every claim scored with NO cases attached.

The point is subtraction. If a model answers Yes to "this fertilizer makes plants
grow taller" at 0.71 before seeing any evidence at all, then a 0.78 on the
coherent_true item is a 0.07 move, not a 0.78 endorsement. Without this, a
coherence effect and a plausibility-of-the-claim effect are indistinguishable -
and claims vary in prior plausibility across families whether or not that was
intended.

One prompt per FAMILY, not per item: all four cells of a family share a
byte-identical claim by construction, so scoring each item's claim separately
would be the same forward pass repeated four times.

    python -m src.baseline --model HuggingFaceTB/SmolLM2-135M \
        --third-option Unknown --items items/draft items/seed \
        --out results/baseline.json

Then feed it to the analysis:

    python -m src.analyze --run results/run.json --baseline results/baseline.json \
        --score delta --out results/analysis_delta.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from . import provenance
from .models import Item, load_items
from .render import DEFAULT_OPTIONS, render_baseline_prompt
from .score import Scorer, TokenizationError, add_common_args, make_scorer


def unique_claims(items: Sequence[Item]) -> list[tuple[str, str, str]]:
    """(family_id, claim, domain), one row per family, in family order.

    Raises if a family's items disagree about the claim - that would mean the
    2x2 is not within-family and the baseline could not be joined back on.
    """
    by_family: dict[str, tuple[str, str]] = {}
    for it in items:
        prev = by_family.get(it.family_id)
        if prev is None:
            by_family[it.family_id] = (it.claim, it.domain)
        elif prev[0] != it.claim:
            raise ValueError(
                f"family '{it.family_id}' has two different claims:\n"
                f"  {prev[0]!r}\n  {it.claim!r}\n"
                "The baseline is joined on family_id, so this must not happen."
            )
    return [(f, c, d) for f, (c, d) in sorted(by_family.items())]


def score_baselines(
    scorer: Scorer,
    items: Sequence[Item],
    *,
    progress: bool = True,
    checkpoint: str | None = None,
    gc_every: int = 20,
) -> list[dict[str, Any]]:
    """One forward pass per family claim, checkpointed per claim.

    Keyed on family_id rather than item_id, because that is the unit here; the
    resume logic is otherwise identical to src.score's and exists for the same
    reason (D-041).
    """
    import gc
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    options = getattr(scorer, "options", DEFAULT_OPTIONS)
    rows = unique_claims(items)

    done: dict[str, dict[str, Any]] = {}
    if checkpoint and _Path(checkpoint).exists():
        for line in _Path(checkpoint).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("family_id"):
                done[rec["family_id"]] = rec
        if progress:
            print(f"  checkpoint {checkpoint}: {len(done)} claims done",
                  file=sys.stderr)

    todo = [r for r in rows if r[0] not in done]
    fh = None
    if checkpoint:
        _Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        fh = open(checkpoint, "a", encoding="utf-8")
    try:
        for n, (family_id, claim, domain) in enumerate(todo, 1):
            res = scorer.score_prompt(render_baseline_prompt(claim, options))
            rec: dict[str, Any] = {
                "family_id": family_id,
                "claim": claim,
                "domain": domain,
            }
            rec.update(res.to_dict())
            done[family_id] = rec
            if fh is not None:
                fh.write(_json.dumps(rec) + "\n")
                fh.flush()
                _os.fsync(fh.fileno())
            if n % gc_every == 0:
                gc.collect()
            if progress and (n % 5 == 0 or n == len(todo)):
                print(f"  scored {n}/{len(todo)} claims (this process)",
                      file=sys.stderr)
    finally:
        if fh is not None:
            fh.close()

    return [done[f] for f, _, _ in rows if f in done]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--min-mass-covered",
        type=float,
        default=0.01,
        help="same gate as src.score; a baseline built on 1% of the distribution "
        "would corrupt every delta computed from it",
    )
    args = ap.parse_args(argv)

    items = load_items(args.items)
    if not items:
        raise SystemExit(f"no items found under {args.items}")
    rows = unique_claims(items)
    print(
        f"loaded {len(items)} items -> {len(rows)} distinct claims (one per family)",
        file=sys.stderr,
    )

    try:
        scorer = make_scorer(args, items)
    except TokenizationError as exc:
        print(f"\nTOKENIZATION CHECK FAILED\n\n  {exc}\n", file=sys.stderr)
        return 2

    records = score_baselines(
        scorer, items, checkpoint=args.checkpoint, gc_every=args.gc_every
    )

    meta = provenance.run_meta(**scorer.meta(), item_dirs=list(args.items))
    meta["n_claims"] = len(records)
    meta["n_items_covered"] = len(items)
    meta["items_hash"] = provenance.hash_items([i.id for i in items])
    options = getattr(scorer, "options", DEFAULT_OPTIONS)
    meta["prompts_hash"] = provenance.hash_prompts(
        render_baseline_prompt(c, options) for _, c, _ in rows
    )
    mass = [r["mass_covered"] for r in records]
    meta["mass_covered_mean"] = sum(mass) / len(mass)
    meta["mass_covered_min"] = min(mass)
    model_slug = str(meta.get("model", "mock")).replace("/", "_").replace(":", "-")

    payload = {
        "run_id": f"baseline__{model_slug}__{meta['items_hash']}"
        f"__{meta['prompt_template_hash']}",
        "kind": "baseline",
        "note": (
            "Each claim scored with NO cases attached. Subtract from the evidence "
            "run to separate what the model already believed from what the "
            "evidence moved."
        ),
        "meta": meta,
        "records": records,
    }
    out = provenance.write_json(args.out, payload)

    mean_p = sum(r["p_yes_3way"] for r in records) / len(records)
    print(f"wrote {out}  ({len(records)} claims)")
    print(f"mean baseline P(yes) with no evidence = {mean_p:.4f}")
    print(
        f"mean mass_covered = {meta['mass_covered_mean']:.4f}  "
        f"min = {meta['mass_covered_min']:.4f}"
    )
    hi = sorted(records, key=lambda r: -r["p_yes_3way"])[:3]
    lo = sorted(records, key=lambda r: r["p_yes_3way"])[:3]
    print("\nmost believed without evidence:")
    for r in hi:
        print(f"  {r['p_yes_3way']:.4f}  {r['family_id']:<22} {r['claim']}")
    print("least believed without evidence:")
    for r in lo:
        print(f"  {r['p_yes_3way']:.4f}  {r['family_id']:<22} {r['claim']}")

    if meta["mass_covered_mean"] < args.min_mass_covered:
        print(
            f"\nFAIL: mean mass_covered {meta['mass_covered_mean']:.5f} < "
            f"{args.min_mass_covered}; this baseline is not interpretable.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
