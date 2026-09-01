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


def hunter_vs_reader() -> str:
    cur, hunter = _load(CURRENT), {}
    if HUNTER.exists():
        hunter = {r["item_id"]: r for r in json.loads(
            HUNTER.read_text(encoding="utf-8")).values()}

    rows = [
        "| cell | n | hunter: flaw found | hunter per-auditor | reader catch |",
        "|---|---|---|---|---|",
    ]
    for cell in FALSE_CELLS:
        g = [r for r in cur.values() if r["cell"] == cell]
        if not g:
            continue
        h = [hunter.get(r["item_id"]) for r in g]
        h = [x for x in h if x]
        found = sum(1 for x in h if x.get("bucket") in ("found", "too_easy"))
        hits = sum(1 for x in h for m in (x.get("matches") or []) if m == "same_flaw")
        tot = sum(len(x.get("matches") or []) for x in h)
        reader = st.mean(r["reader_catch_rate"] for r in g)
        rows.append(
            f"| `{cell}` | {len(g)} | {found}/{len(h)} = {found/len(h):.0%} | "
            f"{hits}/{tot} = {hits/tot:.0%} | **{reader:.2f}** |"
            if h and tot else
            f"| `{cell}` | {len(g)} | - | - | **{reader:.2f}** |"
        )
    rows.append("")
    rows.append("| cell | n | reader false-positive rate |")
    rows.append("|---|---|---|")
    for cell in TRUE_CELLS:
        g = [r for r in cur.values() if r["cell"] == cell]
        if g:
            m = st.mean(r["reader_false_positive_rate"] for r in g)
            rows.append(f"| `{cell}` | {len(g)} | **{m:.2f}** |")
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
        ("every TRUE cell false-positive <= 0.15", TRUE_CELLS,
         "reader_false_positive_rate", 0.15, "max"),
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
            vals = [r[key] for r in d.values() if r["cell"] in cells]
            return max(vals) if kind == "max" else min(vals)
        b, c = val(base), val(cur)
        ok = (c <= limit) if kind in ("max", "spread") else (c >= limit)
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
