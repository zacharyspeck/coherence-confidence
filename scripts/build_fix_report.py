"""Fill the data-derived sections of FIX_REPORT.md from the artifacts.

The prose is hand-written; the tables are not. Anything that is a number comes
from `results/` so the report cannot drift from what was actually measured.

    python scripts/build_fix_report.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import load_items  # noqa: E402
from src.validate import run_checks  # noqa: E402

REPORT = Path("FIX_REPORT.md")


def salience_table() -> str:
    log = Path("results/salience_log.md")
    if not log.exists():
        return "_No salience log yet._"
    lines = log.read_text(encoding="utf-8").split("\n")
    keep = [ln for ln in lines if ln.startswith("|") or ln.startswith("<sub>")]
    return "\n".join(keep)


def gate_table() -> str:
    items = load_items(["items/draft", "items/seed"])
    results = run_checks(items)
    rows = ["| gate | result | measured |", "|---|---|---|"]
    for r in results:
        mark = "PASS" if r.passed else "**FAIL**"
        rows.append(f"| `{r.name}` | {mark} | {r.summary} |")

    abl = subprocess.run(
        [sys.executable, "scripts/lexical_ablation.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    parts = []
    for ln in abl.stdout.split("\n"):
        s = ln.strip()
        if s and "%" in s and not s.startswith("|") and "accuracy" not in s:
            bits = s.split()
            if len(bits) >= 2 and bits[1].endswith("%"):
                parts.append(f"| `{bits[0]}` | {bits[1]} |")
    ablation = ""
    if parts:
        ablation = (
            "\n\n**Component ablation** — grouped 5-fold CV accuracy at telling "
            "TRUE from FALSE, on each part of the passage in isolation. Chance is "
            "50%.\n\n| passage part | accuracy |\n|---|---|\n" + "\n".join(parts)
        )
    return "\n".join(rows) + ablation


def change_table() -> str:
    items = load_items(["items/draft", "items/seed"])
    by_cell_mech: dict[tuple[str, str], int] = {}
    for i in items:
        if i.flaw_mechanism:
            k = (i.cell, i.flaw_mechanism)
            by_cell_mech[k] = by_cell_mech.get(k, 0) + 1

    mechs = ["stated_confound", "broken_chronology", "scope_mismatch"]
    rows = ["**Counts per flaw mechanism, per cell**", ""]
    rows.append("| cell | " + " | ".join(f"`{m}`" for m in mechs) + " |")
    rows.append("|---" * (len(mechs) + 1) + "|")
    for cell in ("coherent_false", "diverse_false"):
        rows.append(
            f"| `{cell}` | "
            + " | ".join(str(by_cell_mech.get((cell, m), 0)) for m in mechs)
            + " |"
        )
    rows.append("")

    clauses = Path("results/scope_clauses.json")
    if clauses.exists():
        d = json.loads(clauses.read_text(encoding="utf-8"))
        rounds: dict[int, list[str]] = {}
        for fam, v in d.items():
            rounds.setdefault(int(v.get("round", 1)), []).append(fam)
        rows.append("**Clause round in force, per family**")
        rows.append("")
        for r in sorted(rounds):
            rows.append(f"- round {r}: {len(rounds[r])} families — "
                        + ", ".join(f"`{f}`" for f in sorted(rounds[r])))
        rows.append("")

    rev = Path("results/reverted_families.json")
    if rev.exists():
        info = json.loads(rev.read_text(encoding="utf-8"))
        if info.get("reverted"):
            rows.append("**Reverted** (quieting made a FALSE item unfindable):")
            rows.append("")
            for f in info["reverted"]:
                why = info.get("unfindable_items", {}).get(f, [])
                rows.append(f"- `{f}` — " + (", ".join(why) if why else "manual"))
        else:
            rows.append("**Reverted:** none — no FALSE item became unfindable.")
    else:
        rows.append("**Reverted:** none — no FALSE item became unfindable.")
    rows.append("")

    audit = Path("results/item_audit.json")
    if audit.exists():
        a = json.loads(audit.read_text(encoding="utf-8"))
        false_rows = [r for r in a.values() if not r["ground_truth"]]
        true_rows = [r for r in a.values() if r["ground_truth"]]
        missed = [r["item_id"] for r in false_rows if r["bucket"].startswith("missed")]
        easy = [r["item_id"] for r in false_rows if r["bucket"] == "too_easy"]
        fp = [r["item_id"] for r in true_rows if r["bucket"] == "false_positive"]
        rows += [
            "**Latest blind audit**",
            "",
            f"- FALSE items whose intended flaw was found: "
            f"**{len(false_rows) - len(missed)}/{len(false_rows)}**",
            f"- unfindable (broken): **{len(missed)}**"
            + (f" — {', '.join(missed)}" if missed else ""),
            f"- flagged too easy: **{len(easy)}**"
            + (f" — {', '.join(easy)}" if easy else ""),
            f"- TRUE decoys drawing a false positive: **{len(fp)}/{len(true_rows)}**"
            + (f" — {', '.join(fp)}" if fp else ""),
        ]
    return "\n".join(rows)


def gate_output() -> str:
    p = Path("results/model_gate_smollm135m.json")
    if not p.exists():
        return "_No model-gate run recorded._"
    d = json.loads(p.read_text(encoding="utf-8"))
    L = [
        f"Run against `{d['model']}` as a worked example "
        "(the same command works on any HF causal LM):",
        "",
        "```",
        "  third option",
    ]
    for t in d["third_option_tried"]:
        mark = "  <-- chosen" if t["word"] == d["third_option_chosen"] else ""
        L.append(f"    {t['canonical']!r:<12} {t['n_tokens']} token(s){mark}")
    L.append("")
    L.append("  option token ids")
    for role, v in d["option_token_ids"].items():
        L.append(
            f"    {role:<7} {v['word']:<8} {v['n_token_ids']} ids, "
            f"{v['n_with_leading_space']} with leading space"
        )
    L.append("")
    L.append(
        f"  coverage  mean {d['mass_covered_mean']:.4f} over "
        f"{d['n_probe_items']} probe items (need > {d['min_mass_required']})"
    )
    L.append(f"  RESULT: {'PASS' if d['passed'] else 'FAIL'}")
    L.append("```")
    L.append("")
    if d.get("notes"):
        for n in d["notes"]:
            L.append(f"- {n}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args(argv)

    path = Path(args.report)
    text = path.read_text(encoding="utf-8")
    blocks = {
        "<!--SALIENCE_TABLE-->": salience_table(),
        "<!--GATE_TABLE-->": gate_table(),
        "<!--CHANGE_TABLE-->": change_table(),
        "<!--GATE_OUTPUT-->": gate_output(),
    }
    filled = 0
    for marker, body in blocks.items():
        if marker in text:
            text = text.replace(marker, body)
            filled += 1
    path.write_text(text, encoding="utf-8")
    print(f"filled {filled}/{len(blocks)} data sections in {path}")
    if filled < len(blocks):
        print("  (missing markers are already filled - re-run from a clean template)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
