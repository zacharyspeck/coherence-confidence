"""END-TO-END SMOKE TEST ON SYNTHETIC DATA. Step 6 of the brief, run first.

Generates 80 fake items, scores them with the mock, validates, analyzes, and
checks every headline number against a value derived on paper. The point is to
prove the plumbing works *before* the content exists, so that when a real number
looks strange later, the pipeline is not on the list of suspects.

    python -m src.smoke                 # run it
    python -m src.smoke --keep          # leave the artifacts in results/smoke/

Nothing here touches items/draft/ or items/seed/, and nothing here downloads a
model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from . import provenance
from .analyze import analyze, attach_baseline, to_markdown
from .mock_scorer import EXPECTED_AUC, MockScorer
from .models import CORE_CELLS, load_items
from .score import build_payload, score_items
from .synth import make_synthetic_families
from .validate import format_report, run_checks

# ---------------------------------------------------------------------------
# The expected values, written out as arithmetic rather than as constants.
# See the docstring of src/mock_scorer.py for the construction.
# ---------------------------------------------------------------------------

M = {"coherent": 5, "diverse": 9}  # TRUE items placed below every FALSE item


def expected_true_mean(m: int) -> float:
    """(m low TRUE items at 0.050+0.001i) + (20-m high ones at 0.800+0.001i), /20."""
    low = sum(0.050 + 0.001 * i for i in range(m))
    high = sum(0.800 + 0.001 * i for i in range(m, 20))
    return (low + high) / 20


def expected_false_mean() -> float:
    """20 FALSE items at 0.300 + 0.001*j, /20."""
    return sum(0.300 + 0.001 * j for j in range(20)) / 20


def expected_auc(m: int) -> float:
    """(20-m) TRUE items beat all 20 FALSE items; the other m lose all 20."""
    return (20 - m) * 20 / 400


EXPECTED = {
    "cell_means": {
        "coherent_true": expected_true_mean(M["coherent"]),
        "coherent_false": expected_false_mean(),
        "diverse_true": expected_true_mean(M["diverse"]),
        "diverse_false": expected_false_mean(),
    },
    "auc": {c: expected_auc(m) for c, m in M.items()},
    "abstention": {
        "coherent_true": M["coherent"] / 20,
        "coherent_false": 0.0,
        "diverse_true": M["diverse"] / 20,
        "diverse_false": 0.0,
    },
    # Dropping abstentions removes exactly the TRUE items that lost every pair.
    "auc_if_abstained_dropped": {"coherent": 1.0, "diverse": 1.0},
}

TOL = 1e-9


class SmokeFailure(AssertionError):
    pass


def check(label: str, got: Any, want: Any, tol: float = TOL) -> str:
    if isinstance(want, float):
        ok = abs(float(got) - want) <= tol
        line = f"  {'OK ' if ok else 'FAIL'}  {label:<52} got {got:.6f}  want {want:.6f}"
    else:
        ok = got == want
        line = f"  {'OK ' if ok else 'FAIL'}  {label:<52} got {got!r}  want {want!r}"
    if not ok:
        raise SmokeFailure(line.strip())
    return line


# ---------------------------------------------------------------------------


def run(outdir: Path, n_resamples: int, keep: bool, quiet: bool = False) -> int:
    say = (lambda *a: None) if quiet else print

    items_dir = outdir / "items"
    if outdir.exists():
        shutil.rmtree(outdir)
    items_dir.mkdir(parents=True)

    say("=" * 74)
    say("SMOKE TEST - synthetic items, mock scorer, no model, no real content")
    say("=" * 74)

    # -- 1. generate ---------------------------------------------------------
    say("\n[1/7] generating 80 synthetic items (20 families x 4 cells)")
    families = make_synthetic_families(20)
    for fam in families:
        (items_dir / f"{fam.family_id}.json").write_text(
            json.dumps(fam.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    say(f"      wrote {len(families)} family files to {items_dir}")

    # -- 2. reload through the real loader -----------------------------------
    say("\n[2/7] reloading through src.models.load_items (full schema validation)")
    items = load_items([items_dir])
    check("item count", len(items), 80)
    for c in CORE_CELLS:
        check(f"items in {c}", sum(1 for i in items if i.cell == c), 20)
    say(f"      {len(items)} items loaded and validated")

    # -- 3. validate ---------------------------------------------------------
    say("\n[3/7] running src.validate gates")
    results = run_checks(items)
    say(format_report(results))
    failed = [r.name for r in results if not r.passed]
    if failed:
        raise SmokeFailure(f"synthetic items failed validation: {failed}")

    # -- 4. score with the mock ---------------------------------------------
    say("\n[4/7] scoring with MockScorer(scenario='known')")
    scorer = MockScorer(scenario="known").prepare(items)
    records = score_items(scorer, items, progress=False)
    payload = build_payload(scorer, items, records, item_dirs=[str(items_dir)])
    run_path = provenance.write_json(outdir / "run_mock.json", payload)
    check("records written", len(records), 80)
    say(f"      wrote {run_path}")

    # -- 5. analyze ----------------------------------------------------------
    say(f"\n[5/7] analyzing ({n_resamples} bootstrap resamples)")
    analysis = analyze(records, n_resamples=n_resamples)
    meta = dict(payload["meta"], run_id=payload["run_id"])
    provenance.write_json(
        outdir / "analysis_mock.json",
        {"kind": "analysis", "source_meta": meta, "analysis": analysis},
    )
    (outdir / "analysis_mock.md").write_text(
        to_markdown(analysis, meta), encoding="utf-8"
    )
    say(f"      wrote {outdir / 'analysis_mock.json'} and .md")

    # -- 6. check every headline number against the arithmetic ---------------
    say("\n[6/7] checking headline numbers against values computed on paper")
    lines: list[str] = []

    for cell, want in EXPECTED["cell_means"].items():
        lines.append(
            check(
                f"mean P(yes) 3-way, {cell}",
                analysis["cell_means_p_yes_3way"][cell]["value"],
                want,
            )
        )
    for coh, want in EXPECTED["auc"].items():
        e = analysis["auc_primary_full_set"]["conditions"][coh]
        lines.append(check(f"AUC within {coh}", e["estimate"]["value"], want))
        lines.append(check(f"pairs in {coh}", e["detail"]["n_pairs"], 400))
        lines.append(
            check(
                f"wins in {coh}",
                e["detail"]["n_wins"],
                int(want * 400),
            )
        )
        lines.append(check(f"AUC[{coh}] matches mock constant", want, EXPECTED_AUC[coh]))
    for cell, want in EXPECTED["abstention"].items():
        lines.append(
            check(f"abstention rate, {cell}", analysis["abstention_rate"][cell]["value"], want)
        )
    for coh, want in EXPECTED["auc_if_abstained_dropped"].items():
        lines.append(
            check(
                f"AUC[{coh}] IF abstentions were dropped",
                analysis["auc_primary_full_set"]["conditions"][coh][
                    "auc_if_abstained_dropped_DIAGNOSTIC"
                ],
                want,
            )
        )

    lines.append(
        check(
            "PRIMARY ENDPOINT: AUC(coherent) - AUC(diverse)",
            analysis["auc_primary_full_set"]["gap_coherent_minus_diverse"]["value"],
            EXPECTED["auc"]["coherent"] - EXPECTED["auc"]["diverse"],
        )
    )
    lines.append(
        check(
            "matched-mechanism subset uses the same families in both conditions",
            analysis["auc_matched_mechanism"]["families_match_across_conditions"],
            True,
        )
    )
    for coh in ("coherent", "diverse"):
        lines.append(
            check(
                f"matched-subset pairs, {coh}",
                analysis["auc_matched_mechanism"]["conditions"][coh]["detail"]["n_pairs"],
                100,
            )
        )

    e = EXPECTED["cell_means"]
    lines.append(
        check(
            "main effect of coherence",
            analysis["effects"]["main_effect_coherence"]["value"],
            (e["coherent_true"] + e["coherent_false"]) / 2
            - (e["diverse_true"] + e["diverse_false"]) / 2,
        )
    )
    lines.append(
        check(
            "main effect of truth",
            analysis["effects"]["main_effect_truth"]["value"],
            (e["coherent_true"] + e["diverse_true"]) / 2
            - (e["coherent_false"] + e["diverse_false"]) / 2,
        )
    )
    lines.append(
        check(
            "interaction",
            analysis["effects"]["interaction"]["value"],
            (e["coherent_true"] - e["coherent_false"])
            - (e["diverse_true"] - e["diverse_false"]),
        )
    )

    # CIs must exist on everything, and must bracket the point estimate.
    n_ci = 0
    for block in (
        analysis["cell_means_p_yes_3way"],
        analysis["cell_means_p_yes_2way"],
        analysis["abstention_rate"],
        analysis["effects"],
    ):
        for name, d in block.items():
            if d["ci_item"] is None or d["ci_family"] is None:
                raise SmokeFailure(f"{name} has no confidence interval")
            lo, hi = d["ci_item"]
            if not (lo - 1e-12 <= d["value"] <= hi + 1e-12):
                raise SmokeFailure(f"{name}: point estimate {d['value']} outside CI")
            n_ci += 1
    lines.append(f"  OK    {'every reported number carries a 95% CI':<52} {n_ci} of them")

    say("\n".join(lines))

    # -- the hand check, spelled out ----------------------------------------
    say("\n" + "-" * 74)
    say("AUC BY HAND (coherent condition)")
    say("-" * 74)
    say(
        f"""  20 TRUE items vs 20 FALSE items                    = {20 * 20} pairs
  {M['coherent']} TRUE items were scored below every FALSE item  -> they win 0 pairs
  {20 - M['coherent']} TRUE items were scored above every FALSE item  -> they win 20 pairs each
  wins  = {20 - M['coherent']} x 20                                       = {(20 - M['coherent']) * 20}
  ties  =                                                  = 0
  AUC   = (wins + 0.5*ties) / pairs = {(20 - M['coherent']) * 20} / 400          = {expected_auc(M['coherent'])}

  {M['coherent']} of the TRUE items abstain (Unsure is their argmax).
  They are INCLUDED above, per DECISIONS.md D-003.
  Had they been dropped, every losing pair would vanish and the AUC would
  read 1.00 instead of {expected_auc(M['coherent'])} - which is the whole reason the policy exists."""
    )

    # -- 7. the baseline leg -------------------------------------------------
    say("\n[7/7] baseline: every claim with NO cases attached")
    from .baseline import score_baselines, unique_claims

    base_records = score_baselines(scorer, items, progress=False)
    provenance.write_json(
        outdir / "baseline_mock.json",
        {"kind": "baseline", "meta": scorer.meta(), "records": base_records},
    )
    check("one baseline row per family", len(base_records), 20)
    check("one claim per family", len(unique_claims(items)), 20)
    check(
        "mock baseline P(yes) is the documented constant",
        base_records[0]["p_yes_3way"],
        0.5,
    )

    delta_records = attach_baseline(
        [dict(r) for r in records], outdir / "baseline_mock.json"
    )
    delta = analyze(delta_records, n_resamples=n_resamples, which="delta")
    lines = [
        check(
            "delta cell mean, coherent_true (evidence - baseline)",
            delta["cell_means_p_yes_3way"]["coherent_true"]["value"],
            EXPECTED["cell_means"]["coherent_true"] - 0.5,
        ),
        # Subtracting a per-family constant shifts means but must not touch
        # within-condition AUC or any contrast.
        check(
            "delta leaves AUC[coherent] unchanged",
            delta["auc_primary_full_set"]["conditions"]["coherent"]["estimate"]["value"],
            EXPECTED["auc"]["coherent"],
        ),
        check(
            "delta leaves the coherence main effect unchanged",
            delta["effects"]["main_effect_coherence"]["value"],
            analysis["effects"]["main_effect_coherence"]["value"],
        ),
    ]
    say("\n".join(lines))
    say(f"      wrote {outdir / 'baseline_mock.json'}")

    say("\n" + "=" * 74)
    say("SMOKE TEST PASSED - the pipeline is wired correctly.")
    say("These numbers are FIXTURES, not measurements. No model was involved.")
    say("=" * 74)

    if not keep:
        shutil.rmtree(outdir)
        say(f"\nremoved {outdir} (pass --keep to inspect the artifacts)")
    else:
        say(f"\nartifacts kept in {outdir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default="results/smoke")
    ap.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="bootstrap resamples; the smoke test does not need the full 10,000",
    )
    ap.add_argument("--keep", action="store_true", help="keep the generated artifacts")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        return run(Path(args.outdir), args.n_resamples, args.keep, args.quiet)
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED\n  {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
