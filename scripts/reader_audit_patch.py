"""Targeted reader-audit patch: re-measure ONLY items whose passages changed.

After an item edit, `assemble_passages.py` / `build_decorative.py` clear the
edited items' reader rates and nothing else. Re-running the full 18-batch
audit to refresh a handful of items would overwrite 96 stable measurements
with fresh sampling noise; measuring the edited items in a batch of one would
change the reading context they are measured in. This does neither.

Each patch batch holds ONE target item plus 16 filler items drawn one-per-
family from the other families, shuffled, under fresh opaque codes - the same
batch size, the same blindness rules, and the exact scored prompt of the main
audit (src/audit_reader.py). Three runs give each target three independent
readers. Filler answers are collected and then DISCARDED: their items keep the
rates they already have.

    python scripts/reader_audit_patch.py make
    python scripts/reader_audit_patch.py score      # merges into reader_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit_reader import _by_cell, _clusters, code_for  # noqa: E402
from src.models import CELLS, coherence_of, load_items  # noqa: E402
from src.render import render_prompt  # noqa: E402

N_RUNS = 3
FILLERS_PER_BATCH = 16
SALT = "cc-reader-patch1"  # fresh codes: patch verdicts can never be cross-read
OUT = Path("results/reader_audit_patch")
KEY = Path("results/reader_audit_patch_key")
MAIN_AUDIT = Path("results/reader_audit.json")


def targets_of(items, ids: list[str] | None) -> list:
    """The items needing a measurement: neither reader rate is set - or an
    explicit id list, for adding reads to an item whose 3-read draw was
    dominated by one reader phenotype (see D-037 on why that happens)."""
    if ids:
        by_id = {i.id: i for i in items}
        missing = [x for x in ids if x not in by_id]
        if missing:
            raise SystemExit(f"unknown ids: {missing}")
        return [by_id[x] for x in sorted(ids)]
    return sorted(
        (i for i in items
         if i.reader_catch_rate is None and i.reader_false_positive_rate is None),
        key=lambda i: i.id,
    )


def make(items_dirs: list[str], ids: list[str] | None = None,
         runs: list[int] | None = None) -> int:
    items = load_items(items_dirs)
    targets = targets_of(items, ids)
    if not targets:
        raise SystemExit("no items are missing reader rates; nothing to measure")
    by_family: dict[str, list] = {}
    for i in items:
        by_family.setdefault(i.family_id, []).append(i)

    for run in runs if runs is not None else range(N_RUNS):
        out = OUT / f"run{run}"
        blind = out / "blind"
        blind.mkdir(parents=True, exist_ok=True)
        (out / "verdicts").mkdir(parents=True, exist_ok=True)

        keymap: dict[str, dict] = {}
        for b, target in enumerate(targets):
            rng = random.Random(f"{SALT}:{run}:{b}")
            other = sorted(f for f in by_family if f != target.family_id)
            fams = rng.sample(other, min(FILLERS_PER_BATCH, len(other)))
            group = [target] + [
                rng.choice(sorted(by_family[f], key=lambda i: i.id)) for f in fams
            ]
            rng.shuffle(group)

            lines = [
                f"# Reader batch {b:02d} (run {run})",
                "",
                f"{len(group)} independent questions. Answer each one on its own "
                "terms, in order. Nothing links them and the answers need not agree.",
                "",
                "For each, reply with exactly one of: Yes, No, Unsure.",
                "",
            ]
            for it in group:
                c = code_for(it.id, SALT)
                keymap[c] = {
                    "item_id": it.id,
                    "family_id": it.family_id,
                    "cell": it.cell,
                    "coherence": coherence_of(it.cell),
                    "ground_truth": it.ground_truth,
                    "flaw_mechanism": it.flaw_mechanism,
                    "role": "target" if it.id == target.id else "filler",
                    "batch": b,
                }
                lines += ["---", "", f"## ITEM {c}", "", "```", render_prompt(it),
                          "```", ""]
            (blind / f"batch_{b:02d}.md").write_text(
                "\n".join(lines), encoding="utf-8"
            )

        KEY.mkdir(parents=True, exist_ok=True)
        (KEY / f"run{run}.json").write_text(
            json.dumps(keymap, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        banned = list(CELLS) + ["flaw", "confound", "ground_truth",
                                "identify a specific reason", "something wrong"]
        leaked = []
        for p in sorted(blind.glob("batch_*.md")):
            text = p.read_text(encoding="utf-8").lower()
            leaked += [f"{p.name}: {w!r}" for w in banned if w.lower() in text]
        if leaked:
            raise SystemExit("PATCH BATCHES ARE CONTAMINATED:\n  " + "\n  ".join(leaked))

    run_list = list(runs) if runs is not None else list(range(N_RUNS))
    print(f"{len(targets)} target item(s): {[t.id for t in targets]}")
    print(f"runs {run_list} x {len(targets)} batches of {1 + FILLERS_PER_BATCH}, "
          f"one target per batch")
    print("verified: no cell labels, no hunting cues")
    return 0


def score(items_dirs: list[str]) -> int:
    items = {i.id: i for i in load_items(items_dirs)}
    per_target: dict[str, dict] = {}
    discarded = 0

    runs_found = sorted(
        int(p.stem.replace("run", "")) for p in KEY.glob("run*.json")
    )
    for run in runs_found:
        keymap = json.loads((KEY / f"run{run}.json").read_text(encoding="utf-8"))
        for p in sorted((OUT / f"run{run}" / "verdicts").glob("batch_*.json")):
            doc = json.loads(p.read_text(encoding="utf-8"))
            for v in doc["verdicts"]:
                meta = keymap.get(v["code"])
                if meta is None:
                    raise SystemExit(f"unknown code {v['code']} in {p}")
                if meta["role"] != "target":
                    discarded += 1
                    continue
                iid = meta["item_id"]
                it = items[iid]
                row = per_target.setdefault(iid, {
                    "item_id": iid,
                    "family_id": it.family_id,
                    "cell": it.cell,
                    "coherence": coherence_of(it.cell),
                    "ground_truth": it.ground_truth,
                    "flaw_mechanism": it.flaw_mechanism,
                    "answers": [],
                    "clusters": [],
                })
                row["answers"].append(str(v["answer"]).strip().lower())
                # p-clusters: patch readers, disjoint from the main audit's
                # r{run}b{batch} clusters by construction.
                row["clusters"].append(f"p{run}b{meta['batch']}")

    for row in per_target.values():
        a, n = row["answers"], len(row["answers"])
        row["n_reads"] = n
        row["n_no"] = a.count("no")
        row["n_yes"] = a.count("yes")
        row["n_unsure"] = a.count("unsure")
        row["abstention_rate"] = round(row["n_unsure"] / n, 4) if n else None
        rate = round(row["n_no"] / n, 4) if n else None
        if row["ground_truth"]:
            row["reader_false_positive_rate"] = rate
            row["reader_catch_rate"] = None
            row["reads_as_false"] = bool(n and 3 * row["n_no"] > n)
            row["one_dissenter"] = bool(n and row["n_no"] * 3 == n)
            row["invisible"] = False
            row["below_target"] = False
        else:
            row["reader_catch_rate"] = rate
            row["reader_false_positive_rate"] = None
            row["invisible"] = bool(n and 3 * row["n_no"] < n)
            row["one_dissenter"] = False
            row["below_target"] = bool(n and 2 * row["n_no"] < n)
            row["reads_as_false"] = False

    short = [k for k, v in per_target.items() if v["n_reads"] < N_RUNS]
    if short:
        raise SystemExit(f"targets short of {N_RUNS} reads: {short}")

    main = json.loads(MAIN_AUDIT.read_text(encoding="utf-8"))
    for iid, row in per_target.items():
        main["items"][iid] = row
    main["by_cell"] = _by_cell(main["items"])
    main["cluster_report"] = _clusters(main["items"])
    main.setdefault("patches", []).append({
        "salt": SALT,
        "targets": sorted(per_target),
        "runs": runs_found,
        "filler_reads_discarded": discarded,
    })
    MAIN_AUDIT.write_text(json.dumps(main, indent=1), encoding="utf-8")

    print(f"merged {len(per_target)} re-measured item(s) into {MAIN_AUDIT}")
    print(f"discarded {discarded} filler reads (their items keep existing rates)")
    for iid in sorted(per_target):
        r = per_target[iid]
        metric = ("fp", r["reader_false_positive_rate"]) if r["ground_truth"] \
            else ("catch", r["reader_catch_rate"])
        print(f"  {iid}: {r['n_yes']}Y/{r['n_no']}N/{r['n_unsure']}U "
              f"-> {metric[0]}={metric[1]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "score"):
        s = sub.add_parser(name)
        s.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
        if name == "make":
            s.add_argument("--ids", nargs="+", default=None,
                           help="explicit target ids (default: items missing rates)")
            s.add_argument("--runs", nargs="+", type=int, default=None,
                           help="run indices to build (default: 0..N_RUNS-1)")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        return make(args.items, args.ids, args.runs)
    return score(args.items)


if __name__ == "__main__":
    raise SystemExit(main())
