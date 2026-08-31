"""Item-set validation. Fails the build when the items are broken.

The four gates the brief specifies:

1. **Word-count parity** — no cell's mean word count may sit more than 10% away
   from the grand mean. Length is a cue the lexical check cannot see, because
   that one is unigram/bigram based.
2. **Coherent structure** — a coherent item must have exactly 1 distinct value on
   every condition dimension.
3. **Diverse structure** — a diverse item must have exactly 4 distinct values on
   every condition dimension.
4. **No lexical giveaway** — a logistic regression on unigrams+bigrams must not
   predict true vs false above 60%. Above that, the model under test could win by
   reading surface wording instead of reasoning, and the item set is broken. The
   top predictive tokens are printed so they can be fixed.

Gate 4 is averaged over 5 CV shufflings and fails on the mean (D-023). One
shuffle swings the accuracy by several points on 80 items; the first full draft
of the item set ranged 58.8%-68.8% across ten seeds and passed or failed
depending on which one was used.

Four more gates were added because they catch failures that would otherwise
produce a confident, meaningless number: cell balance, duplicate passages, and
answer-key leakage (D-020), plus closer word balance (D-022), which enforces the
per-family invariant that makes gate 4 pass in the first place — every word in a
family's four closing sentences must appear equally often on the TRUE and FALSE
sides. Gate 4 measures the symptom across the whole set; that one measures the
cause, per family, so a single drifting family gets named rather than averaged
away.

    python -m src.validate --items items/draft items/seed
    python -m src.validate --items items/draft items/seed --out results/validation.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from . import provenance
from .models import CELLS, Item, load_items, normalize_ws

WORD_COUNT_TOLERANCE = 0.10
LEXICAL_THRESHOLD = 0.60
N_LEXICAL_SEEDS = 5
N_PER_CELL = 20
CLOSER_IMBALANCE_TOLERANCE = 0
TOP_TOKENS = 25


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    required_by_brief: bool = True
    failures: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "summary": self.summary,
            "required_by_brief": self.required_by_brief,
            "n_failures": len(self.failures),
            "failures": self.failures,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# 1. Word-count parity
# ---------------------------------------------------------------------------


def check_word_counts(
    items: Sequence[Item], tolerance: float = WORD_COUNT_TOLERANCE
) -> CheckResult:
    by_cell: dict[str, list[int]] = defaultdict(list)
    for i in items:
        by_cell[i.cell].append(i.word_count)

    grand = float(np.mean([i.word_count for i in items]))
    rows: dict[str, Any] = {}
    failures: list[str] = []
    for cell in CELLS:
        counts = by_cell.get(cell, [])
        if not counts:
            continue
        mean = float(np.mean(counts))
        dev = (mean - grand) / grand if grand else float("nan")
        rows[cell] = {
            "n": len(counts),
            "mean": round(mean, 3),
            "min": int(min(counts)),
            "max": int(max(counts)),
            "deviation_from_grand": round(dev, 5),
        }
        if abs(dev) > tolerance:
            failures.append(
                f"cell '{cell}' mean word count {mean:.1f} is {dev:+.1%} from the "
                f"grand mean {grand:.1f} (limit +/-{tolerance:.0%})"
            )

    worst = max((abs(r["deviation_from_grand"]) for r in rows.values()), default=0.0)
    return CheckResult(
        name="word_count_parity",
        passed=not failures,
        summary=(
            f"grand mean {grand:.1f} words; largest cell deviation {worst:+.2%} "
            f"(limit +/-{tolerance:.0%})"
        ),
        failures=failures,
        data={"grand_mean": round(grand, 3), "tolerance": tolerance, "by_cell": rows},
    )


# ---------------------------------------------------------------------------
# 2 & 3. Structural manipulation checks
# ---------------------------------------------------------------------------


def check_coherent_structure(items: Sequence[Item]) -> CheckResult:
    """Coherent = all 4 cases share EVERY irrelevant condition."""
    failures: list[str] = []
    checked = 0
    for it in items:
        if it.coherence != "coherent":
            continue
        checked += 1
        for dim in it.dimensions:
            vals = it.distinct_values(dim)
            if len(vals) > 1:
                failures.append(
                    f"{it.id}: coherent item has {len(vals)} distinct values on "
                    f"'{dim}' ({sorted(vals)}); must be exactly 1"
                )
    return CheckResult(
        name="coherent_one_value_per_dimension",
        passed=not failures,
        summary=f"{checked} coherent items checked across their dimensions",
        failures=failures,
        data={"n_checked": checked},
    )


def check_diverse_structure(items: Sequence[Item]) -> CheckResult:
    """Diverse = all 4 cases differ on EVERY irrelevant condition."""
    failures: list[str] = []
    checked = 0
    for it in items:
        if it.coherence != "diverse":
            continue
        checked += 1
        for dim in it.dimensions:
            vals = it.distinct_values(dim)
            if len(vals) < 4:
                failures.append(
                    f"{it.id}: diverse item has only {len(vals)} distinct values on "
                    f"'{dim}' ({sorted(vals)}); must be 4"
                )
    return CheckResult(
        name="diverse_four_values_per_dimension",
        passed=not failures,
        summary=f"{checked} diverse items checked across their dimensions",
        failures=failures,
        data={"n_checked": checked},
    )


# ---------------------------------------------------------------------------
# 4. Lexical giveaway
# ---------------------------------------------------------------------------


def check_lexical_giveaway(
    items: Sequence[Item],
    threshold: float = LEXICAL_THRESHOLD,
    *,
    cv: str = "grouped",
    n_splits: int = 5,
    seed: int = 0,
    n_seeds: int = N_LEXICAL_SEEDS,
    top_k: int = TOP_TOKENS,
    n_permutations: int = 0,
) -> CheckResult:
    """Can a bag-of-ngrams classifier tell TRUE from FALSE?

    Cross-validated and **grouped by family** (D-015): all 4 cells of a family
    land in the same fold, so the classifier cannot memorize a family's
    vocabulary and read the answer off it from the other side of the split.
    In-sample accuracy on 80 short passages with thousands of bigram features is
    ~100% by construction and would fail the build unconditionally.

    Only the passage text is used - the claim is identical across all 4 cells of
    a family, so it carries no true/false signal and would only add noise.

    Averaged over `n_seeds` CV shufflings. With 80 items in 20 family groups, a
    single shuffle moves the accuracy by several points: on the first full draft
    of the item set the accuracy ranged from 58.8% to 68.8% across ten seeds and
    the build passed or failed depending on which one was used. Reporting the
    mean is what stops the gate from being decided by luck; `accuracy_max` and
    `n_seeds_over_threshold` are reported alongside so a marginal pass is visible
    rather than silent.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        GroupKFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.pipeline import Pipeline

    texts = [i.passage for i in items]
    y = np.array([int(i.ground_truth) for i in items])
    groups = np.array([i.family_id for i in items])

    pipe = Pipeline(
        [
            ("vec", CountVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=5000, random_state=seed),
            ),
        ]
    )

    def make_splitter(s: int):
        if cv == "grouped":
            return (
                StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s),
                {"groups": groups},
            )
        if cv == "stratified":
            return (
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s),
                {},
            )
        if cv == "insample":
            return None, {}
        raise ValueError(f"unknown cv mode {cv!r}")

    seeds = [seed] if cv == "insample" else [seed + k for k in range(max(1, n_seeds))]
    per_seed: list[float] = []
    scores = np.array([])
    for s in seeds:
        splitter, split_args = make_splitter(s)
        if splitter is None:
            pipe.fit(texts, y)
            scores = np.array([float(pipe.score(texts, y))])
        else:
            scores = cross_val_score(pipe, texts, y, cv=splitter, **split_args)
        per_seed.append(float(np.mean(scores)))

    acc = float(np.mean(per_seed))
    acc_max = float(np.max(per_seed))
    n_over = int(sum(1 for a in per_seed if a > threshold))

    # Refit on everything, only to name the offending tokens.
    pipe.fit(texts, y)
    vec: CountVectorizer = pipe.named_steps["vec"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    top_true = [
        {"token": str(names[i]), "coef": round(float(coefs[i]), 4)}
        for i in order[::-1][:top_k]
    ]
    top_false = [
        {"token": str(names[i]), "coef": round(float(coefs[i]), 4)}
        for i in order[:top_k]
    ]

    perm: dict[str, Any] | None = None
    if n_permutations and splitter is not None:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n_permutations):
            # Shuffle labels WITHIN family, so the null keeps the design's
            # 2-true/2-false-per-family structure.
            y_perm = y.copy()
            for fam in np.unique(groups):
                m = groups == fam
                y_perm[m] = rng.permutation(y[m])
            null.append(
                float(
                    np.mean(
                        cross_val_score(pipe, texts, y_perm, cv=splitter, **split_args)
                    )
                )
            )
        null_arr = np.array(null)
        perm = {
            "n_permutations": n_permutations,
            "null_mean": round(float(null_arr.mean()), 4),
            "null_p95": round(float(np.percentile(null_arr, 95)), 4),
            "p_value": round(float((null_arr >= acc).mean()), 4),
        }

    failures: list[str] = []
    if acc > threshold:
        failures.append(
            f"a unigram+bigram logistic regression reaches {acc:.1%} "
            f"({cv} CV, mean of {len(seeds)} shufflings) at telling TRUE from "
            f"FALSE, above the {threshold:.0%} limit. There is a lexical "
            "giveaway; see top_tokens_predicting_* for what to rewrite, and "
            "scripts/lexical_ablation.py for which part of the passage leaks."
        )

    return CheckResult(
        name="no_lexical_giveaway",
        passed=not failures,
        summary=(
            f"{cv} {n_splits}-fold CV accuracy {acc:.1%} "
            f"(mean of {len(seeds)} shufflings; max {acc_max:.1%}; "
            f"{n_over}/{len(seeds)} over limit) "
            f"(limit {threshold:.0%}, chance 50%)"
        ),
        failures=failures,
        data={
            "cv": cv,
            "accuracy": round(acc, 4),
            "accuracy_max": round(acc_max, 4),
            "accuracy_per_seed": [round(a, 4) for a in per_seed],
            "n_seeds": len(seeds),
            "n_seeds_over_threshold": n_over,
            "fold_accuracies_last_seed": [round(float(s), 4) for s in scores],
            "threshold": threshold,
            "n_features": int(len(names)),
            "n_items": len(items),
            "top_tokens_predicting_TRUE": top_true,
            "top_tokens_predicting_FALSE": top_false,
            "permutation_test": perm,
            "note": (
                "Coefficients come from a refit on ALL items and are for "
                "diagnosis only; the accuracy above is the cross-validated one."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Additional gates (D-020)
# ---------------------------------------------------------------------------


def check_cell_balance(
    items: Sequence[Item], n_per_cell: int = N_PER_CELL
) -> CheckResult:
    counts = Counter(i.cell for i in items)
    failures = [
        f"cell '{c}' has {counts.get(c, 0)} items, expected {n_per_cell}"
        for c in CELLS
        if counts.get(c, 0) != n_per_cell
    ]
    fams = defaultdict(set)
    for i in items:
        fams[i.family_id].add(i.cell)
    incomplete = sorted(f for f, cs in fams.items() if len(cs) != 4)
    failures += [f"family '{f}' does not have all 4 cells" for f in incomplete]
    return CheckResult(
        name="cell_balance",
        passed=not failures,
        summary=f"{len(items)} items, {len(fams)} families, per-cell {dict(counts)}",
        required_by_brief=False,
        failures=failures,
        data={"counts": dict(counts), "n_families": len(fams)},
    )


def check_duplicate_passages(items: Sequence[Item]) -> CheckResult:
    """Two items with the same passage render to the same prompt and cannot be
    told apart by any measurement. This bit the test fixtures once already."""
    seen: dict[str, list[str]] = defaultdict(list)
    for i in items:
        seen[normalize_ws(i.passage).lower()].append(i.id)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    failures = [
        f"identical passage shared by {ids}: {k[:80]!r}..." for k, ids in dupes.items()
    ]
    return CheckResult(
        name="no_duplicate_passages",
        passed=not failures,
        summary=f"{len(seen)} distinct passages across {len(items)} items",
        required_by_brief=False,
        failures=failures,
        data={"n_distinct": len(seen), "n_duplicate_groups": len(dupes)},
    )


def check_no_answer_leakage(items: Sequence[Item]) -> CheckResult:
    """The confound_note is the answer key. It must never appear in the passage,
    and the passage must not name the cell or the flaw type."""
    failures: list[str] = []
    banned_words = ("ground_truth", "coherent_true", "coherent_false",
                    "diverse_true", "diverse_false", "confound_note", "flaw_mechanism")
    for it in items:
        p = normalize_ws(it.passage).lower()
        if it.confound_note:
            note = normalize_ws(it.confound_note).lower()
            if note and note in p:
                failures.append(f"{it.id}: confound_note appears verbatim in passage")
        for w in banned_words:
            if w in p:
                failures.append(f"{it.id}: passage contains metadata token '{w}'")
    return CheckResult(
        name="no_answer_key_leakage",
        passed=not failures,
        summary=f"{len(items)} passages scanned for answer-key leakage",
        required_by_brief=False,
        failures=failures,
    )


def closer_of(item: Item) -> str:
    """The final line of the passage - where truth lives (D-022)."""
    lines = [ln for ln in item.passage.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def check_closer_balance(
    items: Sequence[Item], tolerance: int = 0
) -> CheckResult:
    """Enforce the D-022 invariant mechanically.

    Within a family, every word used in the four closing sentences must appear
    the same number of times on the TRUE side as on the FALSE side. That is what
    makes the closer carry zero bag-of-words signal about the label, and it is
    the property that took the item set from a 75% lexical ceiling down to chance.

    The lexical gate measures the symptom across the whole set; this measures the
    cause, per family, so a single family drifting out of balance is named
    instead of being averaged away.
    """
    word_re = __import__("re").compile(r"[a-z0-9']+")
    by_family: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_family[i.family_id].append(i)

    failures: list[str] = []
    worst = 0
    per_family: dict[str, int] = {}
    for fam, group in sorted(by_family.items()):
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for it in group:
            side = 0 if it.ground_truth else 1
            for w in word_re.findall(closer_of(it).lower()):
                counts[w][side] += 1
        bad = {w: (t, f) for w, (t, f) in counts.items() if abs(t - f) > tolerance}
        imbalance = max((abs(t - f) for t, f in counts.values()), default=0)
        per_family[fam] = imbalance
        worst = max(worst, imbalance)
        if bad:
            shown = sorted(bad.items(), key=lambda kv: -abs(kv[1][0] - kv[1][1]))[:8]
            failures.append(
                f"{fam}: closer words unbalanced across TRUE/FALSE "
                + ", ".join(f"'{w}' {t}T/{f}F" for w, (t, f) in shown)
                + (f" (+{len(bad) - len(shown)} more)" if len(bad) > len(shown) else "")
            )

    return CheckResult(
        name="closer_word_balance",
        passed=not failures,
        summary=(
            f"{len(by_family)} families checked; worst per-word TRUE/FALSE "
            f"imbalance in a family's closers = {worst} (limit {tolerance})"
        ),
        required_by_brief=False,
        failures=failures,
        data={"tolerance": tolerance, "imbalance_by_family": per_family},
    )


def check_review_status(items: Sequence[Item]) -> CheckResult:
    """Informational: nothing may be pre-marked reviewed by the build (D-008)."""
    bad = [i.id for i in items if i.review_status != "unreviewed"]
    return CheckResult(
        name="all_items_unreviewed",
        passed=not bad,
        summary=f"{len(items) - len(bad)}/{len(items)} items are 'unreviewed'",
        required_by_brief=False,
        failures=[f"{i} is not marked unreviewed; only a human may set that" for i in bad],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = (
    "word_count_parity",
    "coherent_one_value_per_dimension",
    "diverse_four_values_per_dimension",
    "no_lexical_giveaway",
    "cell_balance",
    "no_duplicate_passages",
    "no_answer_key_leakage",
    "closer_word_balance",
    "all_items_unreviewed",
)


def run_checks(
    items: Sequence[Item],
    *,
    tolerance: float = WORD_COUNT_TOLERANCE,
    lexical_threshold: float = LEXICAL_THRESHOLD,
    cv: str = "grouped",
    n_permutations: int = 0,
    n_per_cell: int = N_PER_CELL,
    n_lexical_seeds: int = N_LEXICAL_SEEDS,
    closer_imbalance_tolerance: int = CLOSER_IMBALANCE_TOLERANCE,
    skip: Sequence[str] = (),
    seed: int = 0,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    plan = [
        ("word_count_parity", lambda: check_word_counts(items, tolerance)),
        ("coherent_one_value_per_dimension", lambda: check_coherent_structure(items)),
        ("diverse_four_values_per_dimension", lambda: check_diverse_structure(items)),
        (
            "no_lexical_giveaway",
            lambda: check_lexical_giveaway(
                items,
                lexical_threshold,
                cv=cv,
                seed=seed,
                n_seeds=n_lexical_seeds,
                n_permutations=n_permutations,
            ),
        ),
        ("cell_balance", lambda: check_cell_balance(items, n_per_cell)),
        ("no_duplicate_passages", lambda: check_duplicate_passages(items)),
        ("no_answer_key_leakage", lambda: check_no_answer_leakage(items)),
        (
            "closer_word_balance",
            lambda: check_closer_balance(items, closer_imbalance_tolerance),
        ),
        ("all_items_unreviewed", lambda: check_review_status(items)),
    ]
    for name, fn in plan:
        if name in skip:
            continue
        results.append(fn())
    return results


def format_report(results: Sequence[CheckResult]) -> str:
    lines: list[str] = []
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        tag = "" if r.required_by_brief else "  (added, D-020)"
        lines.append(f"[{mark}] {r.name}{tag}")
        lines.append(f"       {r.summary}")
        for f in r.failures[:20]:
            lines.append(f"       - {f}")
        if len(r.failures) > 20:
            lines.append(f"       ... and {len(r.failures) - 20} more")
    return "\n".join(lines)


def format_top_tokens(res: CheckResult) -> str:
    d = res.data
    if "top_tokens_predicting_TRUE" not in d:
        return ""
    lines = [
        "",
        "Most predictive n-grams (refit on all items; diagnosis only).",
        "If the CV accuracy is near or above the limit, these are what to rewrite.",
        "",
        f"{'-> TRUE':<34}{'-> FALSE'}",
    ]
    a = d["top_tokens_predicting_TRUE"]
    b = d["top_tokens_predicting_FALSE"]
    for i in range(max(len(a), len(b))):
        left = f"{a[i]['token']:<22}{a[i]['coef']:+.3f}" if i < len(a) else ""
        right = f"{b[i]['token']:<22}{b[i]['coef']:+.3f}" if i < len(b) else ""
        lines.append(f"{left:<34}{right}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--out", default=None, help="write a JSON report here")
    ap.add_argument("--tolerance", type=float, default=WORD_COUNT_TOLERANCE)
    ap.add_argument("--lexical-threshold", type=float, default=LEXICAL_THRESHOLD)
    ap.add_argument(
        "--lexical-cv",
        default="grouped",
        choices=["grouped", "stratified", "insample"],
        help="grouped = 5-fold CV with all 4 cells of a family in one fold (D-015)",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="label-permutation null for the lexical check (within-family shuffle)",
    )
    ap.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    ap.add_argument(
        "--lexical-seeds",
        type=int,
        default=N_LEXICAL_SEEDS,
        help="CV shufflings to average the lexical accuracy over. A single "
        "shuffle swings it by several points on 80 items, so one seed can pass "
        "or fail the same item set by luck.",
    )
    ap.add_argument("--skip", nargs="*", default=[], choices=list(ALL_CHECKS))
    ap.add_argument(
        "--closer-imbalance-tolerance",
        type=int,
        default=CLOSER_IMBALANCE_TOLERANCE,
        help="max allowed TRUE/FALSE count difference for any single word across "
        "a family's four closing sentences (D-022)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        items = load_items(args.items)
    except ValueError as exc:
        print(f"[FAIL] items failed to load:\n  {exc}", file=sys.stderr)
        return 1
    if not items:
        print(f"[FAIL] no items found under {args.items}", file=sys.stderr)
        return 1

    results = run_checks(
        items,
        tolerance=args.tolerance,
        lexical_threshold=args.lexical_threshold,
        cv=args.lexical_cv,
        n_permutations=args.permutations,
        n_per_cell=args.n_per_cell,
        n_lexical_seeds=args.lexical_seeds,
        closer_imbalance_tolerance=args.closer_imbalance_tolerance,
        skip=args.skip,
        seed=args.seed,
    )

    print(f"validating {len(items)} items from {args.items}\n")
    print(format_report(results))
    lex = next((r for r in results if r.name == "no_lexical_giveaway"), None)
    if lex is not None:
        print(format_top_tokens(lex))

    failed = [r for r in results if not r.passed]
    print("")
    if failed:
        print(f"VALIDATION FAILED: {len(failed)}/{len(results)} checks failed "
              f"({', '.join(r.name for r in failed)})")
    else:
        print(f"VALIDATION PASSED: {len(results)}/{len(results)} checks")

    if args.out:
        provenance.write_json(
            args.out,
            {
                "kind": "validation",
                "meta": provenance.run_meta(
                    item_dirs=list(args.items),
                    n_items=len(items),
                    tolerance=args.tolerance,
                    lexical_threshold=args.lexical_threshold,
                    lexical_cv=args.lexical_cv,
                ),
                "passed": not failed,
                "checks": [r.to_dict() for r in results],
            },
        )
        print(f"wrote {args.out}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
