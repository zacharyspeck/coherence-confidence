"""Compare a control run against the primary run on the items BOTH completed.

The order and rotation controls could not be finished on this hardware. A
partial control is still informative if it is read as what it is: a paired,
within-item comparison on the overlap, not a re-estimate of the endpoint.

The paired comparison is also the sharper test. Case order carries no
evidence, so if presentation is not driving the measurement, an item's
`p_yes_3way` should barely move when its four cases are permuted. Mean |delta|
and the correlation over matched items answer that directly, and they do not
need all 100 items to be meaningful - whereas a subset AUC on 7 items a side
mostly measures its own noise.

    python scripts/compare_control.py --control results/partial_shuffled.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path


def load_jsonl(p: Path) -> dict[str, dict]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("item_id"):
            out[r["item_id"]] = r
    return out


def load_run(p: Path) -> dict[str, dict]:
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {r["item_id"]: r for r in doc["records"]}


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Explicit pairwise win rate, ties 0.5 - the same definition as analyze.py."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--primary", default="results/run_qwen3b.json")
    ap.add_argument("--control", required=True)
    ap.add_argument("--label", default="control")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    base = load_run(Path(args.primary))
    cpath = Path(args.control)
    ctrl = load_run(cpath) if cpath.suffix == ".json" else load_jsonl(cpath)

    shared = sorted(set(base) & set(ctrl))
    if not shared:
        raise SystemExit("no overlapping items")

    deltas = [ctrl[i]["p_yes_3way"] - base[i]["p_yes_3way"] for i in shared]
    a = [base[i]["p_yes_3way"] for i in shared]
    b = [ctrl[i]["p_yes_3way"] for i in shared]
    r = st.correlation(a, b) if len(shared) > 2 else float("nan")

    # Would the ranking flip? Same-cell AUCs on the overlap, both runs.
    def cell_auc(src, cond):
        pos = [src[i]["p_yes_3way"] for i in shared
               if src[i]["cell"] == f"{cond}_true"]
        neg = [src[i]["p_yes_3way"] for i in shared
               if src[i]["cell"] == f"{cond}_false"]
        return auc(pos, neg), len(pos), len(neg)

    out = {
        "kind": "partial_control_comparison",
        "label": args.label,
        "primary": args.primary,
        "control": args.control,
        "n_overlap": len(shared),
        "n_control_total": len(ctrl),
        "paired": {
            "mean_abs_delta_p_yes": round(st.mean(abs(d) for d in deltas), 4),
            "max_abs_delta_p_yes": round(max(abs(d) for d in deltas), 4),
            "mean_signed_delta": round(st.mean(deltas), 4),
            "pearson_r": round(r, 4) if r == r else None,
            "n_moved_over_0.10": sum(1 for d in deltas if abs(d) > 0.10),
            "n_crossed_0.5": sum(
                1 for i in shared
                if (base[i]["p_yes_3way"] >= 0.5) != (ctrl[i]["p_yes_3way"] >= 0.5)
            ),
        },
        "subset_auc": {},
    }
    for cond in ("coherent", "diverse"):
        ba, bp, bn = cell_auc(base, cond)
        ca, cp, cn = cell_auc(ctrl, cond)
        out["subset_auc"][cond] = {
            "primary": round(ba, 4) if ba is not None else None,
            "control": round(ca, 4) if ca is not None else None,
            "n_true": bp,
            "n_false": bn,
        }
    if all(out["subset_auc"][c]["primary"] is not None for c in ("coherent", "diverse")):
        out["subset_auc"]["gap_primary"] = round(
            out["subset_auc"]["coherent"]["primary"]
            - out["subset_auc"]["diverse"]["primary"], 4)
        out["subset_auc"]["gap_control"] = round(
            out["subset_auc"]["coherent"]["control"]
            - out["subset_auc"]["diverse"]["control"], 4)

    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
