"""Fill the data-derived sections of READER_REPORT.md.

The prose is hand-written; every number comes from the artifacts, so the report
cannot drift from what was measured. Writes BETWEEN paired markers and leaves
both in place, so re-running it regenerates rather than consuming.

    python scripts/build_reader_report.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import CELLS, load_items  # noqa: E402
from src.validate import run_checks  # noqa: E402

REPORT = Path("READER_REPORT.md")
BASELINE = Path("results/reader_audit_baseline.json")
CURRENT = Path("results/reader_audit.json")
HUNTER = Path("results/item_audit.json")
#: end of the control pass: before the reader audit existed
BEFORE_REV = "8315d9d"
FALSE_CELLS = ("coherent_false", "diverse_false", "decorative_false")
TRUE_CELLS = ("coherent_true", "diverse_true", "decorative_true")


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))["items"] if p.exists() else {}


def _hunter_rates() -> dict[str, dict]:
    """Per-auditor hunter performance, recomputed from the raw verdicts.

    A report counts as a find only if the auditor QUOTED the falsifying span -
    a shared 5-gram with `flaws.flaw_sentence` - rather than merely asserting
    that something was wrong. That is a deterministic stand-in for the separate
    judge pass, and on this data it reproduces the judged figure exactly: 94 of
    100 reads, with reported and matched identical, so every flaw the hunter
    reported was the right one.
    """
    from src.flaws import flaw_sentence, ngrams

    items = {i.id: i for i in load_items(["items/draft", "items/seed"])}
    out: dict[str, dict] = {}
    rounds = sorted(Path("results/audit").glob("round*"))
    for rd in rounds:
        n = int(rd.name.replace("round", ""))
        kp = Path("results/audit_key") / f"round{n}.json"
        if not kp.exists():
            continue
        key = json.loads(kp.read_text(encoding="utf-8"))
        for p in sorted((rd / "verdicts").glob("batch_*.json")):
            for v in json.loads(p.read_text(encoding="utf-8"))["verdicts"]:
                m = key.get(v["code"])
                if not m or m["ground_truth"]:
                    continue
                it = items.get(m["item_id"])
                if it is None:
                    continue
                r = out.setdefault(it.cell, {"reads": 0, "matched": 0})
                r["reads"] += 1
                if v.get("flaw_found") and ngrams(
                    flaw_sentence(it.passage, it.flaw_mechanism), 5
                ) & ngrams(v.get("flaw_description") or "", 5):
                    r["matched"] += 1
    return out


def hunter_vs_reader() -> str:
    cur, base = _load(CURRENT), _load(BASELINE)
    hunter = _hunter_rates()
    clusters = (json.loads(CURRENT.read_text(encoding="utf-8")).get("cluster_report")
                or {}).get("by_cell", {})

    rows = [
        "| cell | n | hunter per-auditor | reader catch (baseline) | "
        "reader catch (now) | 95% CI, clustered by reader |",
        "|---|---|---|---|---|---|",
    ]
    for cell in FALSE_CELLS:
        g = [r for r in cur.values() if r["cell"] == cell]
        if not g:
            continue
        h = hunter.get(cell)
        hs = f"{h['matched']}/{h['reads']} = {h['matched']/h['reads']:.0%}" if h else "-"
        bg = [r for r in base.values() if r["cell"] == cell]
        b = f"{st.mean(r['reader_catch_rate'] for r in bg):.2f}" if bg else "-"
        c = clusters.get(cell, {})
        ci = c.get("ci95_cluster_bootstrap")
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "-"
        rows.append(
            f"| `{cell}` | {len(g)} | {hs} | {b} | "
            f"**{st.mean(r['reader_catch_rate'] for r in g):.2f}** | {cis} |"
        )
    rows += ["", "| cell | n | reader false-positive (baseline) | now | "
             "95% CI, clustered by reader | readers giving 0 |", "|---|---|---|---|---|---|"]
    for cell in TRUE_CELLS:
        g = [r for r in cur.values() if r["cell"] == cell]
        if not g:
            continue
        bg = [r for r in base.values() if r["cell"] == cell]
        b = f"{st.mean(r['reader_false_positive_rate'] for r in bg):.2f}" if bg else "-"
        c = clusters.get(cell, {})
        ci = c.get("ci95_cluster_bootstrap")
        cis = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "-"
        z = f"{c.get('n_clusters_at_zero')}/{c.get('n_clusters')}" if c else "-"
        rows.append(
            f"| `{cell}` | {len(g)} | {b} | "
            f"**{st.mean(r['reader_false_positive_rate'] for r in g):.2f}** | {cis} | {z} |"
        )
    return "\n".join(rows)


def triage_table() -> str:
    base, cur = _load(BASELINE), _load(CURRENT)
    rows = [
        "| cell | metric | before | after | invisible / reads-as-false |",
        "|---|---|---|---|---|",
    ]
    for cell in CELLS:
        b = [r for r in base.values() if r["cell"] == cell]
        c = [r for r in cur.values() if r["cell"] == cell]
        if not c:
            continue
        false_cell = cell.endswith("_false")
        k = "reader_catch_rate" if false_cell else "reader_false_positive_rate"
        label = "catch rate" if false_cell else "false-positive rate"
        bm = f"{st.mean(r[k] for r in b):.2f}" if b else "-"
        cm = st.mean(r[k] for r in c)
        bad_b = sum(1 for r in b if r["invisible"] or r["reads_as_false"]) if b else 0
        bad_c = sum(1 for r in c if r["invisible"] or r["reads_as_false"])
        rows.append(
            f"| `{cell}` | {label} | {bm} | **{cm:.2f}** | {bad_b} -> {bad_c} |"
        )

    def spread(d, cells, key):
        v = [st.mean(r[key] for r in d.values() if r["cell"] == c) for c in cells
             if any(r["cell"] == c for r in d.values())]
        return max(v) - min(v) if v else float("nan")

    rows += ["", "| target | before | after | met |", "|---|---|---|---|"]
    for label, cells, key, limit, kind in [
        # cell MEANS, not item maxima: the target is stated per cell.
        ("every TRUE cell false-positive <= 0.15", TRUE_CELLS,
         "reader_false_positive_rate", 0.15, "max_cell_mean"),
        ("coherent_true - diverse_true gap <= 0.10",
         ("coherent_true", "diverse_true"), "reader_false_positive_rate", 0.10, "spread"),
        ("every FALSE item catch >= 0.5", FALSE_CELLS, "reader_catch_rate", 0.5, "min_item"),
        ("FALSE cell means within 0.10", FALSE_CELLS, "reader_catch_rate", 0.10, "spread"),
    ]:
        def val(d):
            if not d:
                return float("nan")
            if kind == "spread":
                return spread(d, cells, key)
            if kind == "max_cell_mean":
                return max(
                    st.mean(r[key] for r in d.values() if r["cell"] == c)
                    for c in cells
                    if any(r["cell"] == c for r in d.values())
                )
            vals = [r[key] for r in d.values() if r["cell"] in cells]
            return max(vals) if kind == "max" else min(vals)
        b, c = val(base), val(cur)
        ok = (c <= limit) if kind in ("max", "max_cell_mean", "spread") else (c >= limit)
        rows.append(
            f"| {label} | {b:.2f} | **{c:.2f}** | {'YES' if ok else '**NO**'} |"
        )
    return "\n".join(rows)


def _gates_before() -> dict[str, str]:
    out = subprocess.run(["git", "show", f"{BEFORE_REV}:results/validation.json"],
                         capture_output=True, text=True, check=False)
    if out.returncode != 0:
        return {}
    doc = json.loads(out.stdout)
    checks = doc["checks"] if isinstance(doc, dict) else doc
    return {c["name"]: ("PASS" if c["passed"] else "**FAIL**") for c in checks}


def gate_table() -> str:
    items = load_items(["items/draft", "items/seed"])
    before = _gates_before()
    rows = [f"| gate | before (`{BEFORE_REV}`) | after | measured now |",
            "|---|---|---|---|"]
    for r in run_checks(items):
        was = before.get(r.name, "— *(new)*")
        rows.append(
            f"| `{r.name}` | {was} | {'PASS' if r.passed else '**FAIL**'} | "
            f"{r.summary.replace('|', chr(92) + '|')} |"
        )
    abl = subprocess.run([sys.executable, "scripts/lexical_ablation.py"],
                         capture_output=True, text=True, check=False)
    head = abl.stdout.split("reading it:")[0]
    parts = [f"| `{b[0]}` | {b[1]} |" for b in
             (ln.strip().split() for ln in head.split("\n"))
             if len(b) == 2 and b[1].endswith("%")]
    if parts:
        rows += ["", "**Component ablation** — grouped 5-fold CV accuracy at "
                 "telling TRUE from FALSE on each part of the passage. Chance "
                 "is 50%.\n", "| passage part | accuracy |", "|---|---|"] + parts
    return "\n".join(rows)


def mechanism_table() -> str:
    cur = _load(CURRENT)
    base = _load(BASELINE)
    rows = ["| mechanism | n | catch before | catch after |", "|---|---|---|---|"]
    mechs = sorted({r["flaw_mechanism"] for r in cur.values() if r["flaw_mechanism"]})
    for m in mechs:
        c = [r for r in cur.values() if r["flaw_mechanism"] == m]
        b = [r for r in base.values() if r["flaw_mechanism"] == m]
        bm = f"{st.mean(r['reader_catch_rate'] for r in b):.2f}" if b else "-"
        rows.append(f"| `{m}` | {len(c)} | {bm} | "
                    f"**{st.mean(r['reader_catch_rate'] for r in c):.2f}** |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args(argv)
    path = Path(args.report)
    text = path.read_text(encoding="utf-8")
    blocks = {
        "<!--HUNTER_VS_READER-->": hunter_vs_reader(),
        "<!--TRIAGE-->": triage_table(),
        "<!--GATES-->": gate_table(),
        "<!--MECHANISM-->": mechanism_table(),
    }
    filled = 0
    for marker, body in blocks.items():
        close = marker.replace("<!--", "<!--/")
        if marker not in text:
            continue
        start = text.index(marker) + len(marker)
        end = text.index(close, start) if close in text[start:] else start
        text = text[:start] + "\n" + body + "\n" + text[end:]
        if close not in text:
            text = text[: start + len(body) + 2] + close + text[start + len(body) + 2:]
        filled += 1
    path.write_text(text, encoding="utf-8")
    print(f"filled {filled}/{len(blocks)} data sections in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
