"""Aggregate the blind audit into results/item_audit.md (step 9).

Inputs
  results/audit/round<N>/keymap.json    code -> the answer key (never shown to an auditor)
  results/audit/round<N>/verdicts.json  code -> what the blind auditor said
  results/audit/matches.json            (code, round) -> did the auditor's flaw
                                        match the intended one?

Every FALSE item lands in one of four buckets:

  found          at least one blind auditor named the intended flaw
  missed         no auditor found it  -> THE ITEM IS BROKEN. If the flaw cannot
                 be found by a careful reader who is looking for one, the item is
                 not measuring reasoning, it is measuring nothing.
  too_easy       found, and auditors rated it as good as stated outright
                 (mean explicitness >= 4 of 5) -> the item gives itself away
  wrong_flaw     an auditor confidently named a DIFFERENT flaw. Worth a look:
                 either the item has a second unintended defect, or the auditor
                 was pattern-matching.

TRUE items are audited too, as decoys. Their false-positive rate is the audit's
own specificity: if auditors "find" flaws in TRUE items at a high rate, then
finding a flaw in a FALSE item is weak evidence that the intended flaw is there.

    python scripts/score_audit.py --audit results/audit --out results/item_audit.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

TOO_EASY_EXPLICITNESS = 4.0


def load_rounds(audit_dir: Path) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Returns (keymap, verdicts_by_code). Verdicts carry their round number."""
    keymap: dict[str, dict] = {}
    verdicts: dict[str, list[dict]] = defaultdict(list)

    rounds = sorted(audit_dir.glob("round*"))
    if not rounds:
        raise SystemExit(f"no round* directories under {audit_dir}")

    for rd in rounds:
        n = int(rd.name.replace("round", ""))
        km = json.loads((rd / "keymap.json").read_text(encoding="utf-8"))
        keymap.update(km)
        vpath = rd / "verdicts.json"
        if not vpath.exists():
            print(f"  (no verdicts for {rd.name}, skipping)")
            continue
        for v in json.loads(vpath.read_text(encoding="utf-8")):
            v["round"] = n
            verdicts[v["code"]].append(v)
    return keymap, verdicts


def classify(key: dict, vs: list[dict], matches: dict[str, str]) -> dict[str, Any]:
    """One row of the audit table."""
    if not vs:
        return {"bucket": "unaudited", "n_verdicts": 0}

    verdict_matches = [matches.get(f"{v['code']}:{v['round']}", "unknown") for v in vs]
    found = any(m == "same_flaw" for m in verdict_matches)
    wrong = any(m == "different_flaw" for m in verdict_matches)

    expl = [
        float(v.get("explicitness", 0))
        for v, m in zip(vs, verdict_matches)
        if m == "same_flaw" and v.get("explicitness")
    ]
    mean_expl = mean(expl) if expl else None

    if not key["ground_truth"]:
        if not found:
            bucket = "missed"
        elif mean_expl is not None and mean_expl >= TOO_EASY_EXPLICITNESS:
            bucket = "too_easy"
        else:
            bucket = "found"
    else:
        # A TRUE item "fails" the audit when an auditor invents a flaw for it.
        bucket = "false_positive" if any(v.get("flaw_found") for v in vs) else "clean"

    return {
        "bucket": bucket,
        "n_verdicts": len(vs),
        "matches": verdict_matches,
        "wrong_flaw_reported": wrong,
        "mean_explicitness": mean_expl,
        "answers": [v.get("answer") for v in vs],
        "descriptions": [v.get("flaw_description", "") for v in vs],
        "confidences": [v.get("confidence") for v in vs],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--audit", default="results/audit")
    ap.add_argument("--out", default="results/item_audit.md")
    args = ap.parse_args(argv)

    audit_dir = Path(args.audit)
    keymap, verdicts = load_rounds(audit_dir)
    mpath = audit_dir / "matches.json"
    matches = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}

    rows: dict[str, dict] = {}
    for code, key in keymap.items():
        rows[code] = {**key, **classify(key, verdicts.get(code, []), matches)}

    false_rows = {c: r for c, r in rows.items() if not r["ground_truth"]}
    true_rows = {c: r for c, r in rows.items() if r["ground_truth"]}

    def count(rs, bucket):
        return sum(1 for r in rs.values() if r["bucket"] == bucket)

    n_false = len(false_rows)
    n_missed = count(false_rows, "missed")
    n_easy = count(false_rows, "too_easy")
    n_found = count(false_rows, "found")
    n_fp = count(true_rows, "false_positive")
    n_true = len(true_rows)

    L: list[str] = []
    L.append("# Item audit — blind flaw identification\n")
    L.append(
        "Step 9. Every item was shown to independent auditors carrying **only** the "
        "claim, the passage and the forced question — no cell label, no "
        "`ground_truth`, no `confound_note`, and never two items from the same "
        "family in one batch. The TRUE items were mixed in as decoys, so an "
        "auditor could not know that everything it saw was broken.\n"
    )
    L.append("## Headline\n")
    L.append("| | count | of | rate |")
    L.append("|---|---|---|---|")
    L.append(f"| FALSE items whose intended flaw was found | {n_found + n_easy} | {n_false} | {(n_found + n_easy) / n_false:.0%} |")
    L.append(f"| **FALSE items where NO auditor found the flaw — BROKEN** | **{n_missed}** | {n_false} | {n_missed / n_false:.0%} |")
    L.append(f"| **FALSE items rated as good as stated outright — TOO EASY** | **{n_easy}** | {n_false} | {n_easy / n_false:.0%} |")
    L.append(f"| TRUE items an auditor invented a flaw for (false positives) | {n_fp} | {n_true} | {n_fp / n_true:.0%} |")
    L.append("")
    L.append(
        f"**How to read the false-positive rate.** Auditors reported a flaw in "
        f"{n_fp} of {n_true} items that have nothing wrong with them. Finding a "
        "flaw in a FALSE item is only as informative as that rate is low; treat "
        "it as the audit's own noise floor, and discount the detection rate above "
        "accordingly.\n"
    )

    # -- the two lists that need a human ------------------------------------
    for bucket, title, why in (
        (
            "missed",
            "FLAGGED — broken items (no auditor found the intended flaw)",
            "If a careful reader who is actively looking for a flaw cannot find it, "
            "the item is not testing reasoning. Either the flaw is too subtle to "
            "be a flaw, or the passage does not actually contain it. Rewrite or drop.",
        ),
        (
            "too_easy",
            "FLAGGED — too easy (flaw rated as good as stated outright)",
            "Explicitness >= 4 of 5 means auditors felt the passage announced the "
            "problem. These items measure reading rather than reasoning and will "
            "compress the truth effect. Soften the wording.",
        ),
    ):
        picked = {c: r for c, r in false_rows.items() if r["bucket"] == bucket}
        L.append(f"## {title}\n")
        if not picked:
            L.append("_None._\n")
            continue
        L.append(why + "\n")
        for c, r in sorted(picked.items(), key=lambda kv: kv[1]["item_id"]):
            L.append(f"### `{r['item_id']}`  ({r['flaw_type']})\n")
            L.append(f"- code `{c}`, {r['n_verdicts']} verdicts, "
                     f"answers {r['answers']}, matches {r['matches']}")
            if r["mean_explicitness"] is not None:
                L.append(f"- mean explicitness {r['mean_explicitness']:.1f} / 5")
            L.append(f"- **intended flaw:** {r['confound_note']}")
            for d in r["descriptions"]:
                L.append(f"- auditor said: {d}")
            L.append("")

    # -- items where an auditor named a different flaw ----------------------
    wrong = {c: r for c, r in false_rows.items() if r.get("wrong_flaw_reported")}
    L.append("## Worth a look — an auditor named a DIFFERENT flaw\n")
    if not wrong:
        L.append("_None._\n")
    else:
        L.append(
            "Either the item has a second, unintended defect, or the auditor was "
            "pattern-matching. Both are worth knowing.\n"
        )
        for c, r in sorted(wrong.items(), key=lambda kv: kv[1]["item_id"]):
            L.append(f"- `{r['item_id']}` — intended: {(r['confound_note'] or '')[:120]}...")
            for d, m in zip(r["descriptions"], r["matches"]):
                if m == "different_flaw":
                    L.append(f"  - auditor said: {d}")
        L.append("")

    # -- TRUE items that drew a false positive -------------------------------
    fps = {c: r for c, r in true_rows.items() if r["bucket"] == "false_positive"}
    L.append("## TRUE items an auditor thought were broken\n")
    if not fps:
        L.append("_None._\n")
    else:
        L.append(
            "These are decoys with no intended flaw. An auditor claiming one means "
            "either the item is not as sound as intended — worth checking — or the "
            "auditor over-reads. Both matter for interpreting the table above.\n"
        )
        for c, r in sorted(fps.items(), key=lambda kv: kv[1]["item_id"]):
            L.append(f"- `{r['item_id']}` — answers {r['answers']}")
            for d in r["descriptions"]:
                if d:
                    L.append(f"  - auditor said: {d}")
        L.append("")

    # -- per flaw type -------------------------------------------------------
    L.append("## Detection by flaw type\n")
    L.append("| flaw_type | n | found | missed | too easy |")
    L.append("|---|---|---|---|---|")
    by_flaw: dict[str, list[dict]] = defaultdict(list)
    for r in false_rows.values():
        by_flaw[r["flaw_type"] or "?"].append(r)
    for flaw, rs in sorted(by_flaw.items()):
        L.append(
            f"| {flaw} | {len(rs)} "
            f"| {sum(1 for r in rs if r['bucket'] in ('found', 'too_easy'))} "
            f"| {sum(1 for r in rs if r['bucket'] == 'missed')} "
            f"| {sum(1 for r in rs if r['bucket'] == 'too_easy')} |"
        )
    L.append("")

    # -- full table ----------------------------------------------------------
    L.append("## Every item\n")
    L.append("| item | cell | flaw | bucket | answers | explicitness |")
    L.append("|---|---|---|---|---|---|")
    for c, r in sorted(rows.items(), key=lambda kv: kv[1]["item_id"]):
        e = f"{r['mean_explicitness']:.1f}" if r.get("mean_explicitness") else ""
        L.append(
            f"| `{r['item_id']}` | {r['cell']} | {r['flaw_type'] or ''} "
            f"| {r['bucket']} | {','.join(str(a) for a in r.get('answers', []))} | {e} |"
        )
    L.append("")

    L.append("## Limitation, stated plainly\n")
    L.append(
        "The auditors and the item authors are the same model family (DECISIONS.md "
        "D-017). This measures whether the flaws are **findable**, not how hard "
        "they are for the model under test — an author and an auditor that share "
        "priors will agree more than two independent readers would. Re-running the "
        "audit with a different model, or handing the drafts to a person, is the "
        "way to get a number that means more than this one."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")

    (out.parent / "item_audit.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {out}")
    print(f"  FALSE items: {n_found} found, {n_easy} too easy, {n_missed} MISSED")
    print(f"  TRUE decoys: {n_fp}/{n_true} drew a false positive")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
