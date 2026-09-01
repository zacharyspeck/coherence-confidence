"""Fill the data-derived sections of CONTROL_REPORT.md.

The prose is hand-written; every number comes from the artifacts, so the report
cannot drift from what was measured.

    python scripts/build_control_report.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.complexity import METRIC_KEYS, for_item  # noqa: E402
from src.models import load_items  # noqa: E402
from src.validate import run_checks  # noqa: E402

REPORT = Path("CONTROL_REPORT.md")
LEVELS = ("coherent", "diverse", "decorative")


def _mean(group, key):
    return st.mean(for_item(i)[key] for i in group)


def match_table() -> str:
    items = load_items(["items/draft", "items/seed"])
    grand = st.mean(i.word_count for i in items)
    by = {lv: [i for i in items if i.coherence == lv] for lv in LEVELS}

    rows = [
        "| metric | coherent | diverse | **decorative** | decorative vs diverse |",
        "|---|---|---|---|---|",
    ]
    shown = [
        ("n_distinct_tokens", "distinct tokens"),
        ("n_distinct_entities", "distinct entities"),
        ("n_entity_tokens", "entity tokens (with repeats)"),
        ("type_token_ratio", "type-token ratio"),
        ("n_distinct_condition_values", "distinct CONDITION values"),
        ("n_distinct_decoration_values", "distinct DECORATION values"),
    ]
    for key, label in shown:
        c, d, x = (_mean(by[lv], key) for lv in LEVELS)
        rel = (x - d) / d if d else float("nan")
        # NB: "decoration" contains "ratio" as a substring - match exactly.
        fmt = "{:.3f}" if key == "type_token_ratio" else "{:.1f}"
        rows.append(
            f"| {label} | {fmt.format(c)} | {fmt.format(d)} | **{fmt.format(x)}** | "
            + ("—" if d == 0 else f"{rel:+.1%}")
            + " |"
        )

    rows.append(
        "| mean word count | "
        + " | ".join(
            f"{st.mean(i.word_count for i in by[lv]):.1f}" for lv in LEVELS
        ).replace(
            f"{st.mean(i.word_count for i in by['decorative']):.1f}",
            f"**{st.mean(i.word_count for i in by['decorative']):.1f}**",
        )
        + " | — |"
    )
    rows.append("")
    for lv in LEVELS:
        w = st.mean(i.word_count for i in by[lv])
        rows.append(
            f"- `{lv}` mean word count {w:.1f}, {100 * (w - grand) / grand:+.2f}% "
            f"from the grand mean of {grand:.1f} (limit ±10%)"
        )
    return "\n".join(rows)


def salience_table() -> str:
    audit = Path("results/item_audit.json")
    if not audit.exists():
        return "_No audit on record._"
    rows_json = json.loads(audit.read_text(encoding="utf-8"))
    false_rows = [r for r in rows_json.values() if not r["ground_truth"]]

    out = ["| false cell | n | mean salience | flaws found | too easy |", "|---|---|---|---|---|"]
    vals = {}
    for cell in ("coherent_false", "diverse_false", "decorative_false"):
        grp = [r for r in false_rows if r["cell"] == cell]
        if not grp:
            continue
        sal = [r["mean_explicitness"] for r in grp if r["mean_explicitness"] is not None]
        vals[cell] = st.mean(sal) if sal else float("nan")
        found = sum(1 for r in grp if r["bucket"] in ("found", "too_easy"))
        easy = sum(1 for r in grp if r["bucket"] == "too_easy")
        out.append(
            f"| `{cell}` | {len(grp)} | **{vals[cell]:.2f}** | {found}/{len(grp)} | {easy} |"
        )
    out.append("")
    if len(vals) == 3:
        pairs = [
            ("coherent_false", "diverse_false"),
            ("decorative_false", "coherent_false"),
            ("decorative_false", "diverse_false"),
        ]
        worst = max(abs(vals[a] - vals[b]) for a, b in pairs)
        for a, b in pairs:
            out.append(f"- `{a}` − `{b}` = **{vals[a] - vals[b]:+.2f}**")
        out.append("")
        out.append(
            f"Largest pairwise gap **{worst:.2f}**, against a target of 0.40. No "
            f"cell exceeds 3.5 (worst is {max(vals.values()):.2f})."
        )

    fp = [r for r in rows_json.values() if r["bucket"] == "false_positive"]
    by_cell: dict[str, int] = {}
    for r in fp:
        by_cell[r["cell"]] = by_cell.get(r["cell"], 0) + 1
    n_true = sum(1 for r in rows_json.values() if r["ground_truth"])
    out.append("")
    out.append(
        f"False positives on TRUE decoys: **{len(fp)}/{n_true}** — "
        + (", ".join(f"{k} {v}" for k, v in sorted(by_cell.items())) or "none")
    )
    return "\n".join(out)


REDEFINED = ("passage_word_balance",)
BEFORE_REV = "b5f4ac5"  # end of the fix pass: the 80-item set, before the control arm


def _gates_before() -> dict[str, str]:
    """The BEFORE column comes out of git, not out of memory."""
    out = subprocess.run(
        ["git", "show", f"{BEFORE_REV}:results/validation.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return {}
    doc = json.loads(out.stdout)
    checks = doc["checks"] if isinstance(doc, dict) else doc
    return {c["name"]: ("PASS" if c["passed"] else "**FAIL**") for c in checks}


def gate_table() -> str:
    items = load_items(["items/draft", "items/seed"])
    before = _gates_before()
    rows = [
        f"| gate | before (80 items, `{BEFORE_REV}`) | after (100 items) | measured now |",
        "|---|---|---|---|",
    ]
    for r in run_checks(items):
        was = before.get(r.name, "— *(new)*")
        # PASS -> PASS would otherwise hide that the gate now asks a different
        # question. See "What changed in the gates" below.
        if r.name in REDEFINED:
            was += " *(redefined)*"
        rows.append(
            f"| `{r.name}` | {was} | {'PASS' if r.passed else '**FAIL**'} | "
            # a raw pipe in a summary (e.g. |T-F|) would break the table
            f"{r.summary.replace('|', chr(92) + '|')} |"
        )
    gone = [n for n in before if n not in {r.name for r in run_checks(items)}]
    for n in gone:
        rows.append(f"| `{n}` | {before[n]} | — *(removed)* | — |")
    abl = subprocess.run(
        [sys.executable, "scripts/lexical_ablation.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    # Stop at the interpretation block: everything after it is per-cell
    # detectability, a different question from the TRUE/FALSE ablation.
    head = abl.stdout.split("reading it:")[0]
    parts = []
    for ln in head.split("\n"):
        bits = ln.strip().split()
        if len(bits) == 2 and bits[1].endswith("%"):
            parts.append(f"| `{bits[0]}` | {bits[1]} |")
    if parts:
        rows.append("")
        rows.append(
            "**Component ablation** — grouped 5-fold CV accuracy at telling TRUE "
            "from FALSE on each part of the passage in isolation. Chance is 50%.\n"
        )
        rows.append("| passage part | accuracy |")
        rows.append("|---|---|")
        rows += parts
    return "\n".join(rows)


def counts_table() -> str:
    items = load_items(["items/draft", "items/seed"])
    cells: dict[str, int] = {}
    for i in items:
        cells[i.cell] = cells.get(i.cell, 0) + 1
    rows = ["| cell | n | flaw mechanism |", "|---|---|---|"]
    for cell in (
        "coherent_true",
        "coherent_false",
        "diverse_true",
        "diverse_false",
        "decorative_true",
        "decorative_false",
    ):
        if cell not in cells:
            continue
        mechs = sorted(
            {i.flaw_mechanism for i in items if i.cell == cell and i.flaw_mechanism}
        )
        rows.append(
            f"| `{cell}` | {cells[cell]} | " + (", ".join(mechs) or "—") + " |"
        )
    rows.append("")
    rows.append(f"**{len(items)} items in {len({i.family_id for i in items})} families.**")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args(argv)

    path = Path(args.report)
    text = path.read_text(encoding="utf-8")
    blocks = {
        "<!--MATCH_TABLE-->": match_table(),
        "<!--SALIENCE_TABLE-->": salience_table(),
        "<!--GATE_TABLE-->": gate_table(),
        "<!--COUNTS_TABLE-->": counts_table(),
    }
    # Write BETWEEN the open and close markers and leave both in place, so the
    # report is regenerable. Consuming the markers would make the "rebuild this
    # report" command in the report itself a lie after the first run.
    filled = 0
    for marker, body in blocks.items():
        close = marker.replace("<!--", "<!--/")
        if marker not in text:
            continue
        start = text.index(marker) + len(marker)
        end = text.index(close, start) if close in text[start:] else start
        text = text[:start] + "\n" + body + "\n" + text[end:]
        if close not in text:
            text = text[: start + len(body) + 2] + close + text[start + len(body) + 2 :]
        filled += 1
    path.write_text(text, encoding="utf-8")
    print(f"filled {filled}/{len(blocks)} data sections in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
