"""Build BLIND audit batches for step 9.

Each batch file contains only what the model under test would see - the claim,
the passage, and the forced question - under an opaque code. No cell label, no
ground_truth, no confound_note, no family id. The auditor therefore cannot know
whether an item is TRUE or FALSE, and cannot see the other three cells of the
same family.

Two properties make the audit mean something:

1. **TRUE items are included as decoys.** If an auditor knew every item it saw
   was FALSE it would hunt until it found something and report a flaw regardless.
   Mixing in the TRUE items turns "can the flaw be found?" into a question with a
   measurable false-positive rate.
2. **No batch contains two items from the same family.** Seeing a family's
   coherent_false next to its coherent_true would give the answer away by diff.
   The assignment `batch = (family_index + k*cell_index + offset) mod n_batches`
   with k coprime to n_batches guarantees each family lands in a different batch
   for each of its four cells.

    python scripts/make_audit_batches.py --round 0 --out results/audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import CELLS, load_items  # noqa: E402
from src.render import DEFAULT_OPTIONS, render_prompt  # noqa: E402

N_BATCHES = 10
#: One multiplier per round, each coprime to N_BATCHES so every family is spread
#: across as many different batches as it has cells - four, or six where the
#: control arm is present - and different between rounds so the groupings
#: genuinely differ rather than merely rotating.
MULTIPLIERS = (3, 7, 9)
OFFSETS = (0, 7, 4)


def code_for(item_id: str, salt: str = "cc-audit") -> str:
    h = hashlib.sha256(f"{salt}:{item_id}".encode("utf-8")).hexdigest()
    return h[:6].upper()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--out", default="results/audit")
    ap.add_argument(
        "--key-out",
        default="results/audit_key",
        help="where the answer key goes. Deliberately OUTSIDE the audit tree so "
        "an auditor pointed at its batch directory cannot wander into it.",
    )
    ap.add_argument("--round", type=int, default=0, choices=range(len(MULTIPLIERS)))
    ap.add_argument("--n-batches", type=int, default=N_BATCHES)
    args = ap.parse_args(argv)

    items = load_items(args.items)
    families = sorted({i.family_id for i in items})
    fam_index = {f: k for k, f in enumerate(families)}

    mult = MULTIPLIERS[args.round]
    offset = OFFSETS[args.round]
    batches: dict[int, list] = {b: [] for b in range(args.n_batches)}
    for it in items:
        b = (
            fam_index[it.family_id] + mult * CELLS.index(it.cell) + offset
        ) % args.n_batches
        batches[b].append(it)

    # Invariant check: no family twice in a batch.
    for b, group in batches.items():
        fams = [i.family_id for i in group]
        if len(fams) != len(set(fams)):
            raise SystemExit(
                f"batch {b} contains two items from the same family: {sorted(fams)}. "
                f"Multiplier {mult} is not coprime to n_batches={args.n_batches}."
            )

    out = Path(args.out) / f"round{args.round}"
    blind = out / "blind"
    blind.mkdir(parents=True, exist_ok=True)

    keymap: dict[str, dict] = {}
    for b, group in sorted(batches.items()):
        lines = [
            f"# Blind audit batch {b:02d} (round {args.round})",
            "",
            f"{len(group)} items. For each one, answer the question and say whether "
            "you can identify a specific reason the evidence fails to establish the "
            "claim. Some of these items have nothing wrong with them.",
            "",
        ]
        for it in group:
            c = code_for(it.id)
            keymap[c] = {
                "item_id": it.id,
                "family_id": it.family_id,
                "cell": it.cell,
                "ground_truth": it.ground_truth,
                "flaw_mechanism": it.flaw_mechanism,
                "confound_note": it.confound_note,
                "batch": b,
                "round": args.round,
            }
            lines += [
                "---",
                "",
                f"## ITEM {c}",
                "",
                f"**Claim:** {it.claim}",
                "",
                "**Report:**",
                "",
                "```",
                it.passage,
                "```",
                "",
                "**Question:** Does this evidence establish this claim? "
                f"{' / '.join(DEFAULT_OPTIONS)}",
                "",
            ]
        (blind / f"batch_{b:02d}.md").write_text("\n".join(lines), encoding="utf-8")

    # The key lives in a separate tree entirely, not merely outside blind/. An
    # auditor is told to read only its batch file, but blindness should not
    # depend on that instruction being followed.
    key_dir = Path(args.key_out)
    key_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / f"round{args.round}.json"
    key_path.write_text(
        json.dumps(keymap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "verdicts").mkdir(parents=True, exist_ok=True)

    n_false = sum(1 for v in keymap.values() if not v["ground_truth"])
    print(f"round {args.round}: {len(items)} items -> {args.n_batches} batches")
    print(f"  {n_false} FALSE items, {len(items) - n_false} TRUE decoys")
    print(f"  blind batches: {blind}")
    print(f"  key (do not show an auditor): {key_path}")

    # Belt and braces: the blind files must not contain any answer key.
    leaked = []
    for p in sorted(blind.glob("batch_*.md")):
        text = p.read_text(encoding="utf-8")
        for c, v in keymap.items():
            if v["confound_note"] and v["confound_note"][:40] in text:
                leaked.append(f"{p.name}: confound_note for {c}")
            if v["cell"] in text:
                leaked.append(f"{p.name}: cell label {v['cell']}")
    if leaked:
        raise SystemExit("ANSWER KEY LEAKED INTO BLIND FILES:\n  " + "\n  ".join(leaked))
    print("  verified: no answer key in the blind files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
