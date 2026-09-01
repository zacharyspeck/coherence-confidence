"""The READER audit: what a plain reader answers, not what a hunter finds.

The existing audit (scripts/make_audit_batches.py) tells the auditor that some
items are flawed and asks it to identify the flaw. That is a *hunter*. It
produces a near-ceiling catch rate - 94% per auditor over the 50 FALSE items -
and a `salience` rating that is only meaningful conditional on having already
found the thing. As a covariate for "how loud is this flaw" it is close to
useless, because it is measured on a population that almost never misses.

A 10-item human spot-check caught 1 of 5 flaws. That cannot coexist with 94%.
The two numbers are measuring different things, so this module measures the
other one.

A reader here sees **exactly the scored prompt** - the same instruction, the same
claim, the same passage, the same three options - and nothing else. No mention
that a flaw might exist. No instruction to look for one. No flaw description
field to fill in, because the scored model has nowhere to put one either. The
only thing recorded is which of Yes / No / Unsure it picks.

Three independent runs per item, with different groupings each run, give a
per-item rate rather than a single bit:

    reader_catch_rate           fraction answering No   on a FALSE item
    reader_false_positive_rate  fraction answering No   on a TRUE  item
    abstention_rate             fraction answering Unsure

Blindness works exactly as in the hunter audit and for the same reasons: opaque
per-item codes, TRUE items mixed in throughout (a reader who knew every item was
FALSE would answer No to everything and score 100%), the answer key written to a
separate tree, and **no two items from the same family in one batch** - seeing a
family's coherent_false beside its coherent_true would give the answer away by
diff.

    python -m src.audit_reader make --run 0
    python -m src.audit_reader score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import CELLS, coherence_of, load_items  # noqa: E402
from src.render import DEFAULT_OPTIONS, render_prompt  # noqa: E402

N_BATCHES = 6  # a family carries at most 6 items, so 6 batches is the minimum
N_RUNS = 3
SALT = "cc-reader"  # distinct from the hunter audit's codes
OUT = Path("results/reader_audit")
KEY = Path("results/reader_audit_key")

#: Below this, a FALSE item is effectively invisible to a plain reader.
INVISIBLE_AT = 0.33
#: Above this, a TRUE item reads as false.
READS_FALSE_AT = 0.33


def code_for(item_id: str, salt: str = SALT) -> str:
    h = hashlib.sha256(f"{salt}:{item_id}".encode("utf-8")).hexdigest()
    return h[:6].upper()


def batch_assignment(items, run: int, n_batches: int = N_BATCHES) -> dict[int, list]:
    """Assign items to batches so that no batch holds two items of one family.

    Each family's items are shuffled with a per-run seed and the k-th goes to
    batch k. Because a family never has more items than there are batches, the
    no-collision property is structural rather than something to check for. A
    different permutation per run means the runs genuinely regroup rather than
    relabelling the same groupings.
    """
    by_family: dict[str, list] = {}
    for it in items:
        by_family.setdefault(it.family_id, []).append(it)

    batches: dict[int, list] = {b: [] for b in range(n_batches)}
    for fam in sorted(by_family):
        group = sorted(by_family[fam], key=lambda i: i.id)
        if len(group) > n_batches:
            raise SystemExit(
                f"family {fam} has {len(group)} items but there are only "
                f"{n_batches} batches; a batch would have to hold two of them."
            )
        slots = list(range(n_batches))
        random.Random(f"{run}:{fam}").shuffle(slots)
        for it, b in zip(group, slots):
            batches[b].append(it)

    for b, group in batches.items():
        fams = [i.family_id for i in group]
        if len(fams) != len(set(fams)):
            raise SystemExit(f"batch {b} holds two items from one family")
    return batches


def make(argv_items: list[str], run: int, n_batches: int = N_BATCHES) -> int:
    items = load_items(argv_items)
    batches = batch_assignment(items, run, n_batches)

    out = OUT / f"run{run}"
    blind = out / "blind"
    blind.mkdir(parents=True, exist_ok=True)
    (out / "verdicts").mkdir(parents=True, exist_ok=True)

    keymap: dict[str, dict] = {}
    for b, group in sorted(batches.items()):
        # Order within a batch is shuffled too, so position carries nothing.
        group = list(group)
        random.Random(f"order:{run}:{b}").shuffle(group)
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
            c = code_for(it.id)
            keymap[c] = {
                "item_id": it.id,
                "family_id": it.family_id,
                "cell": it.cell,
                "coherence": coherence_of(it.cell),
                "ground_truth": it.ground_truth,
                "flaw_mechanism": it.flaw_mechanism,
            }
            lines += [
                "---",
                "",
                f"## ITEM {c}",
                "",
                "```",
                render_prompt(it),
                "```",
                "",
            ]
        (blind / f"batch_{b:02d}.md").write_text("\n".join(lines), encoding="utf-8")

    KEY.mkdir(parents=True, exist_ok=True)
    (KEY / f"run{run}.json").write_text(
        json.dumps(keymap, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # The blind files must carry nothing but the scored prompt. In particular
    # they must not name a cell, and must not contain the hunter audit's cue
    # words - the whole point is that this reader is not told to look.
    banned = list(CELLS) + [
        "flaw",
        "confound",
        "ground_truth",
        "identify a specific reason",
        "something wrong",
    ]
    leaked = []
    for p in sorted(blind.glob("batch_*.md")):
        text = p.read_text(encoding="utf-8").lower()
        for w in banned:
            if w.lower() in text:
                leaked.append(f"{p.name}: {w!r}")
    if leaked:
        raise SystemExit("READER BATCHES ARE CONTAMINATED:\n  " + "\n  ".join(leaked))

    n_false = sum(1 for v in keymap.values() if not v["ground_truth"])
    print(f"run {run}: {len(items)} items -> {n_batches} batches")
    print(f"  {n_false} FALSE, {len(items) - n_false} TRUE")
    print(f"  sizes: {[len(g) for _, g in sorted(batches.items())]}")
    print(f"  blind: {blind}")
    print("  verified: no cell labels, no hunting cues")
    return 0


def _load_verdicts(run: int) -> dict[str, list[tuple[str, str]]]:
    """code -> [(answer, cluster_id)]. The cluster is (run, batch): one reader
    answered every item in that batch, so verdicts inside it are NOT
    independent and anything that treats them as 300 separate observations
    will report an interval several times too narrow."""
    got: dict[str, list[tuple[str, str]]] = {}
    vdir = OUT / f"run{run}" / "verdicts"
    for p in sorted(vdir.glob("batch_*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        cluster = f"r{run}b{doc['batch']}"
        for v in doc["verdicts"]:
            got.setdefault(v["code"], []).append(
                (str(v["answer"]).strip().lower(), cluster)
            )
    return got


def score(items_dirs: list[str], runs: int = N_RUNS) -> dict[str, Any]:
    items = {i.id: i for i in load_items(items_dirs)}
    per_item: dict[str, dict] = {}

    for run in range(runs):
        keymap = json.loads((KEY / f"run{run}.json").read_text(encoding="utf-8"))
        answers = _load_verdicts(run)
        for code, meta in keymap.items():
            iid = meta["item_id"]
            row = per_item.setdefault(
                iid,
                {
                    "item_id": iid,
                    "family_id": meta["family_id"],
                    "cell": meta["cell"],
                    "coherence": meta["coherence"],
                    "ground_truth": meta["ground_truth"],
                    "flaw_mechanism": meta["flaw_mechanism"],
                    "answers": [],
                    "clusters": [],
                },
            )
            for ans, cluster in answers.get(code, []):
                row["answers"].append(ans)
                row["clusters"].append(cluster)

    missing = [k for k, v in per_item.items() if len(v["answers"]) < runs]
    for iid, row in per_item.items():
        a = row["answers"]
        n = len(a)
        row["n_reads"] = n
        row["n_no"] = a.count("no")
        row["n_yes"] = a.count("yes")
        row["n_unsure"] = a.count("unsure")
        row["abstention_rate"] = round(row["n_unsure"] / n, 4) if n else None
        rate = round(row["n_no"] / n, 4) if n else None
        # Compare COUNTS, not rounded rates. With three reads the quantisation is
        # 0, 1/3, 2/3, 1, and 1/3 rounds to 0.3333 - which is "above 0.33" as a
        # float and lands exactly ON a one-third threshold as a fraction. Testing
        # 3*n_no > n keeps the boundary case out, so a single dissenting reader
        # does not condemn an item. The 1-of-3 group is reported separately
        # because it is informative rather than disqualifying.
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
            # The actionable line for step 5 is 0.5, not the triage line.
            row["below_target"] = bool(n and 2 * row["n_no"] < n)
            row["reads_as_false"] = False

    doc = {
        "kind": "reader_audit",
        "runs": runs,
        "n_items": len(per_item),
        "incomplete_items": missing,
        "items": per_item,
        "by_cell": _by_cell(per_item),
        "cluster_report": _clusters(per_item),
    }
    return doc


def _clusters(per_item: dict[str, dict], n_boot: int = 4000) -> dict[str, Any]:
    """How much of the variation is the READER rather than the item.

    One agent answers every item in a batch, so a batch is a cluster. If a few
    agents adopt a strict disposition and answer No to everything, a cell mean
    swings hard while nothing about the items has changed - which is exactly
    what happened between two rounds here. Resampling CLUSTERS instead of reads
    gives an interval that admits that.
    """
    out: dict[str, Any] = {"by_cell": {}, "per_cluster": {}}
    for cell in CELLS:
        rows = [r for r in per_item.values() if r["cell"] == cell and r["n_reads"]]
        if not rows:
            continue
        # cluster -> the No-rate that cluster gave this cell
        by_cluster: dict[str, list[int]] = {}
        for r in rows:
            for ans, cl in zip(r["answers"], r["clusters"]):
                by_cluster.setdefault(cl, []).append(1 if ans == "no" else 0)
        rates = {c: sum(v) / len(v) for c, v in by_cluster.items()}
        keys = sorted(rates)
        vals = [rates[k] for k in keys]

        rng = random.Random(f"boot:{cell}")
        boots = []
        for _ in range(n_boot):
            pick = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
            boots.append(sum(pick) / len(pick))
        boots.sort()
        lo = boots[int(0.025 * len(boots))]
        hi = boots[int(0.975 * len(boots)) - 1]

        naive = st.mean(1 if a == "no" else 0 for r in rows for a in r["answers"])
        out["by_cell"][cell] = {
            "n_clusters": len(vals),
            "mean_over_clusters": round(st.mean(vals), 4),
            "naive_mean_over_reads": round(naive, 4),
            "ci95_cluster_bootstrap": [round(lo, 4), round(hi, 4)],
            "n_clusters_at_zero": sum(1 for v in vals if v == 0.0),
            "n_clusters_above_half": sum(1 for v in vals if v > 0.5),
        }
        out["per_cluster"][cell] = {k: round(rates[k], 3) for k in keys}
    return out


def _by_cell(per_item: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cell in CELLS:
        rows = [r for r in per_item.values() if r["cell"] == cell and r["n_reads"]]
        if not rows:
            continue
        truth = rows[0]["ground_truth"]
        key = "reader_false_positive_rate" if truth else "reader_catch_rate"
        vals = [r[key] for r in rows if r[key] is not None]
        out[cell] = {
            "n": len(rows),
            "ground_truth": truth,
            "metric": key,
            "mean": round(st.mean(vals), 4) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "mean_abstention": round(
                st.mean([r["abstention_rate"] for r in rows]), 4
            ),
            "n_invisible": sum(1 for r in rows if r["invisible"]),
            "n_below_target": sum(1 for r in rows if r["below_target"]),
            "n_reads_as_false": sum(1 for r in rows if r["reads_as_false"]),
            "n_one_dissenter": sum(1 for r in rows if r["one_dissenter"]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="write blind reader batches for one run")
    m.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    m.add_argument("--run", type=int, required=True)
    m.add_argument("--n-batches", type=int, default=N_BATCHES)

    s = sub.add_parser("score", help="aggregate verdicts across runs")
    s.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    s.add_argument("--runs", type=int, default=N_RUNS)
    s.add_argument("--out", default="results/reader_audit.json")

    args = ap.parse_args(argv)
    if args.cmd == "make":
        return make(args.items, args.run, args.n_batches)

    doc = score(args.items, args.runs)
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"reader audit: {doc['n_items']} items, {doc['runs']} runs")
    if doc["incomplete_items"]:
        print(f"  INCOMPLETE: {len(doc['incomplete_items'])} items short of {doc['runs']} reads")
    print()
    print(f"  {'cell':<18} {'mean':>6} {'abst':>6} {'broken':>7} {'watch':>6}")
    for cell, c in doc["by_cell"].items():
        broken = c["n_invisible"] + c["n_reads_as_false"]
        watch = c["n_below_target"] + c["n_one_dissenter"]
        print(
            f"  {cell:<18} {c['mean']:>6.2f} {c['mean_abstention']:>6.2f} "
            f"{broken:>7} {watch:>6}"
        )
    print()
    print("  broken = invisible (catch < 1/3) or reads-as-false (FP > 1/3)")
    print("  watch  = FALSE below the 0.5 target, or TRUE with one dissenter of three")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
