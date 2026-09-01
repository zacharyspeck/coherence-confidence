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
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from . import provenance
from .complexity import SURFACE_MATCH_KEYS, for_item
from .models import (
    CELLS,
    CONTROL_CELLS,
    CORE_CELLS,
    Item,
    load_items,
    normalize_ws,
)

WORD_COUNT_TOLERANCE = 0.10
LEXICAL_THRESHOLD = 0.60
N_LEXICAL_SEEDS = 5
N_PER_CELL = 20
#: Global TRUE/FALSE word balance. Thresholds are set from the failure this
#: gate actually caught: 'any' at 14T/4F across 10 families and 'no' at
#: 17T/7F across 13, both introduced when the control arm made the COMPLETE
#: clause TRUE-leaning. Anything at that level must fail.
PASSAGE_IMBALANCE_TOLERANCE = 4
#: A word is only flagged if it is BOTH absolutely and relatively lopsided.
#: Without the relative test, high-frequency function words trip the gate on
#: pure length drift - 'the' at 29T/33F is a 4-count difference and no signal
#: whatsoever.
PASSAGE_IMBALANCE_RATIO = 0.40
#: A word confined to one family cannot be learned under family-grouped CV -
#: it is either in the held-out fold or in training, never usefully in both.
#: So a one-sided word only counts against us if it spans families.
PASSAGE_MIN_FAMILIES = 3
MIN_MATCHED_PER_CELL = 10
#: The control arm must match the diverse cells this closely on surface
#: complexity, or it is not controlling for the thing it claims to.
SURFACE_MATCH_TOLERANCE = 0.10
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
    """Coherent = all 4 cases share EVERY condition dimension.

    Decorative items are held to the same rule, and that is the point: their
    surface variety is decorations, never conditions. If a condition ever varied
    in a decorative item it would be a diverse item wearing a control label, and
    the control would silently stop controlling.
    """
    failures: list[str] = []
    checked = 0
    for it in items:
        if it.coherence not in ("coherent", "decorative"):
            continue
        checked += 1
        for dim in it.dimensions:
            vals = it.distinct_values(dim)
            if len(vals) > 1:
                failures.append(
                    f"{it.id}: {it.coherence} item has {len(vals)} distinct values "
                    f"on condition '{dim}' ({sorted(vals)}); must be exactly 1"
                )
    return CheckResult(
        name="coherent_one_value_per_dimension",
        passed=not failures,
        summary=(
            f"{checked} coherent + decorative items checked across their "
            "condition dimensions"
        ),
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
    """The 2x2 must be balanced; the control arm must be balanced with itself.

    The control arm is deliberately half the size - it exists in the 10
    scope_mismatch families only, so it is drawn from the same scenarios as the
    matched-mechanism subset. What matters is that its two cells are equal, so
    its AUC has as many true as false items.
    """
    counts = Counter(i.cell for i in items)
    failures = [
        f"core cell '{c}' has {counts.get(c, 0)} items, expected {n_per_cell}"
        for c in CORE_CELLS
        if counts.get(c, 0) != n_per_cell
    ]

    control = [counts.get(c, 0) for c in CONTROL_CELLS]
    if any(control) and len(set(control)) != 1:
        failures.append(
            f"control arm is unbalanced: {dict(zip(CONTROL_CELLS, control))}. "
            "Its AUC needs as many true items as false ones"
        )

    fams = defaultdict(set)
    for i in items:
        fams[i.family_id].add(i.cell)
    incomplete = sorted(
        f for f, cs in fams.items() if not set(CORE_CELLS).issubset(cs)
    )
    failures += [f"family '{f}' does not have all 4 cells of the 2x2" for f in incomplete]
    half = sorted(
        f
        for f, cs in fams.items()
        if len(cs & set(CONTROL_CELLS)) == 1
    )
    failures += [
        f"family '{f}' has half a control arm; it must have both cells or neither"
        for f in half
    ]
    n_control_fams = sum(1 for cs in fams.values() if set(CONTROL_CELLS) <= cs)
    return CheckResult(
        name="cell_balance",
        passed=not failures,
        summary=(
            f"{len(items)} items, {len(fams)} families "
            f"({n_control_fams} with a control arm), per-cell {dict(counts)}"
        ),
        required_by_brief=False,
        failures=failures,
        data={
            "counts": dict(counts),
            "n_families": len(fams),
            "n_control_families": n_control_fams,
        },
    )


def check_control_surface_match(
    items: Sequence[Item], tolerance: float = SURFACE_MATCH_TOLERANCE
) -> CheckResult:
    """The control arm must match the DIVERSE cells on surface complexity.

    This is the gate that makes the control mean anything. A decorative item is
    supposed to be as busy to read as a diverse one - as many distinct tokens, as
    many named entities - while carrying a coherent item's evidential structure.
    If the surface match drifts, a decorative-vs-diverse difference stops being
    attributable to evidential independence and the control answers nothing.

    It also asserts the other half: condition variety must match COHERENT. Both
    halves have to hold at once or the item is not a control, it is just a third
    condition.
    """
    by_level: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_level[i.coherence].append(i)

    dec, div, coh = (
        by_level.get("decorative", []),
        by_level.get("diverse", []),
        by_level.get("coherent", []),
    )
    if not dec:
        return CheckResult(
            name="control_surface_match",
            passed=True,
            summary="no control arm present; nothing to match",
            required_by_brief=False,
        )
    if not div or not coh:
        return CheckResult(
            name="control_surface_match",
            passed=False,
            summary="cannot check the control arm without both other levels",
            required_by_brief=False,
            failures=["diverse or coherent cells missing"],
        )

    def mean(group: Sequence[Item], key: str) -> float:
        return sum(for_item(i)[key] for i in group) / len(group)

    failures: list[str] = []
    rows: dict[str, Any] = {}
    for key in SURFACE_MATCH_KEYS:
        d, v, c = mean(dec, key), mean(div, key), mean(coh, key)
        rel = (d - v) / v if v else float("inf")
        rows[key] = {
            "decorative": round(d, 3),
            "diverse": round(v, 3),
            "coherent": round(c, 3),
            "decorative_vs_diverse": round(rel, 5),
        }
        if abs(rel) > tolerance:
            failures.append(
                f"{key}: decorative {d:.1f} vs diverse {v:.1f} is {rel:+.1%}, "
                f"outside +/-{tolerance:.0%}. The control no longer matches the "
                "surface complexity it exists to hold fixed"
            )

    key = "n_distinct_condition_values"
    d, v, c = mean(dec, key), mean(div, key), mean(coh, key)
    rows[key] = {
        "decorative": round(d, 3),
        "diverse": round(v, 3),
        "coherent": round(c, 3),
    }
    if abs(d - c) > 1e-6:
        failures.append(
            f"{key}: decorative {d:.1f} must equal coherent {c:.1f}. The control's "
            "evidential structure has to match the coherent cells exactly"
        )
    if d >= v:
        failures.append(
            f"{key}: decorative {d:.1f} is not below diverse {v:.1f}, so the "
            "control is not separating surface variety from evidential variety"
        )

    worst = max(abs(rows[k]["decorative_vs_diverse"]) for k in SURFACE_MATCH_KEYS)
    return CheckResult(
        name="control_surface_match",
        passed=not failures,
        summary=(
            f"{len(dec)} decorative items; largest surface gap vs diverse "
            f"{worst:+.1%} (limit +/-{tolerance:.0%}); condition values "
            f"{rows[key]['decorative']:.1f} vs coherent "
            f"{rows[key]['coherent']:.1f}, diverse {rows[key]['diverse']:.1f}"
        ),
        required_by_brief=False,
        failures=failures,
        data=rows,
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
    """The final line of the passage. Since the fix pass this is a neutral
    procedural sentence carrying no truth signal - the flaw lives mid-passage."""
    lines = [ln for ln in item.passage.split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def flaw_line_of(item: Item) -> str:
    """The mid-passage line, where every truth-bearing clause now lives."""
    lines = [ln for ln in item.passage.split("\n") if ln.strip()]
    return lines[3] if len(lines) >= 7 else ""


_WORD_RE = re.compile(r"[a-z0-9']+")


def check_passage_word_balance(
    items: Sequence[Item],
    tolerance: int = PASSAGE_IMBALANCE_TOLERANCE,
    ratio: float = PASSAGE_IMBALANCE_RATIO,
    min_families: int = PASSAGE_MIN_FAMILIES,
) -> CheckResult:
    """No word may lean TRUE or FALSE across the whole set.

    A word is flagged only when all three hold:

      absolute    |T - F| > tolerance
      relative    |T - F| / (T + F) > ratio
      learnable   it appears in at least `min_families` families

    **Why global and not per-family.** This gate is a proxy for
    `no_lexical_giveaway`, which trains on 16 families and tests on 4. What such
    a classifier can exploit is a GLOBAL association between a word and the
    label. A word that leans TRUE inside one family and FALSE inside another
    cancels and is invisible to it. Checking per family flagged 'in' at 3T/0F in
    one family while it sat at 100T/97F globally - a 1.5% split and no signal at
    all. Per-family lopsidedness is also arithmetically forced here: when both
    FALSE items of a family share a mechanism the falsifying clause is 2F against
    at most 1T, and 3F against at most 2T in the families carrying the control
    arm (D-026, D-030). Demanding per-family balance meant demanding the
    impossible and then reporting noise.

    The three conditions each rule out a different false alarm: the absolute test
    stops high-frequency function words tripping on length drift, the relative
    test stops words appearing once or twice, and the family test stops words
    confined to a single family, which cannot cross a grouped fold boundary in
    either direction.

    `no_lexical_giveaway` remains the authority. This names the specific words
    behind a failure there, which that gate cannot.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    families_with: dict[str, set[str]] = defaultdict(set)
    for it in items:
        side = 0 if it.ground_truth else 1
        words = _WORD_RE.findall(it.passage.lower())
        for w in words:
            counts[w][side] += 1
        for w in set(words):
            families_with[w].add(it.family_id)

    flagged: list[tuple[float, str, int, int, int]] = []
    worst = 0.0
    for w, (t, f) in counts.items():
        diff, total = abs(t - f), t + f
        rel = diff / total if total else 0.0
        n_fam = len(families_with[w])
        if diff > tolerance and n_fam >= min_families:
            worst = max(worst, rel)
        if diff > tolerance and rel > ratio and n_fam >= min_families:
            flagged.append((rel, w, t, f, n_fam))

    flagged.sort(reverse=True)
    failures = [
        f"'{w}' is {t}T/{f}F across {n} families - a {r:.0%} split. A classifier "
        "trained on other families can carry that straight across a fold boundary"
        for r, w, t, f, n in flagged[:12]
    ]
    if len(flagged) > 12:
        failures.append(f"... and {len(flagged) - 12} more")

    # Per-family worst is kept as data: it is not a pass/fail criterion, but a
    # family drifting far from the rest is still worth a look by eye.
    per_family: dict[str, float] = {}
    by_family: dict[str, list[Item]] = defaultdict(list)
    for i in items:
        by_family[i.family_id].append(i)
    for fam, group in by_family.items():
        c: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for it in group:
            side = 0 if it.ground_truth else 1
            for w in _WORD_RE.findall(it.passage.lower()):
                c[w][side] += 1
        per_family[fam] = max(
            (abs(t - f) / (t + f) for t, f in c.values() if abs(t - f) > tolerance),
            default=0.0,
        )

    return CheckResult(
        name="passage_word_balance",
        passed=not flagged,
        summary=(
            f"{len(counts)} distinct words; flagged when |T-F| > {tolerance}, the "
            f"split exceeds {ratio:.0%}, and the word spans >= {min_families} "
            f"families. {len(flagged)} flagged; worst qualifying split {worst:.0%}"
        ),
        required_by_brief=False,
        failures=failures,
        data={
            "abs_tolerance": tolerance,
            "ratio_tolerance": ratio,
            "min_families": min_families,
            "n_flagged": len(flagged),
            "flagged": [
                {"word": w, "true": t, "false": f, "families": n, "split": round(r, 4)}
                for r, w, t, f, n in flagged[:20]
            ],
            "worst_ratio_by_family": {k: round(v, 4) for k, v in per_family.items()},
        },
    )


def check_flaw_declarations(items: Sequence[Item]) -> CheckResult:
    """Every FALSE item must declare BOTH clause variants, and they must agree
    with its mechanism. The Item model checks consistency when a variant is
    present; this checks that it is present at all."""
    failures: list[str] = []
    for it in items:
        if it.ground_truth:
            continue
        if it.confound_variant is None or it.scope_variant is None:
            failures.append(
                f"{it.id}: FALSE item must declare both confound_variant and "
                f"scope_variant (got {it.confound_variant!r}, {it.scope_variant!r})"
            )
            continue
        live_c = it.confound_variant == "changed_reach"
        live_s = it.scope_variant == "subset_incomplete"
        if not (live_c or live_s) and it.flaw_mechanism != "broken_chronology":
            failures.append(
                f"{it.id}: declares '{it.flaw_mechanism}' but carries no live clause "
                "combination, so nothing in the passage falsifies it"
            )
        if live_c and live_s:
            failures.append(
                f"{it.id}: carries BOTH live combinations, so it is false twice over "
                "and the mechanism label is not what a reader would find"
            )
    return CheckResult(
        name="flaw_declarations_complete",
        passed=not failures,
        summary=f"{sum(1 for i in items if not i.ground_truth)} FALSE items checked",
        required_by_brief=False,
        failures=failures,
    )


def check_matched_mechanism_subset(
    items: Sequence[Item], minimum: int = MIN_MATCHED_PER_CELL
) -> CheckResult:
    """The confound-controlled endpoint needs the SAME mechanism present in both
    false cells, in the same families. Without it there is no way to answer
    'aren't the coherent-false items just easier to catch'."""
    by_cell: dict[str, Counter] = defaultdict(Counter)
    fams: dict[str, set[str]] = defaultdict(set)
    for i in items:
        if i.flaw_mechanism:
            by_cell[i.cell][i.flaw_mechanism] += 1
            if i.flaw_mechanism == "scope_mismatch":
                fams[i.cell].add(i.family_id)

    failures: list[str] = []
    for cell in ("coherent_false", "diverse_false"):
        n = by_cell[cell].get("scope_mismatch", 0)
        if n < minimum:
            failures.append(
                f"{cell} has only {n} scope_mismatch items, need at least {minimum}"
            )
    if fams["coherent_false"] != fams["diverse_false"]:
        only_c = sorted(fams["coherent_false"] - fams["diverse_false"])
        only_d = sorted(fams["diverse_false"] - fams["coherent_false"])
        failures.append(
            "the matched subset does not draw on the same families in both "
            f"conditions; only coherent: {only_c}, only diverse: {only_d}"
        )
    return CheckResult(
        name="matched_mechanism_subset",
        passed=not failures,
        summary=(
            f"scope_mismatch: {by_cell['coherent_false'].get('scope_mismatch', 0)} "
            f"coherent_false, {by_cell['diverse_false'].get('scope_mismatch', 0)} "
            f"diverse_false, over {len(fams['coherent_false'])} shared families"
        ),
        required_by_brief=False,
        failures=failures,
        data={"counts_by_cell": {k: dict(v) for k, v in by_cell.items()}},
    )


#: Two FALSE items may not be falsified by the same words. Above this
#: similarity (1.0 - normalized edit distance) the phrasing is a recurrence.
FLAW_SIMILARITY_LIMIT = 0.70
FLAW_NGRAM = 8


def check_duplicate_flaw_phrasing(
    items: Sequence[Item],
    limit: float = FLAW_SIMILARITY_LIMIT,
    n: int = FLAW_NGRAM,
) -> CheckResult:
    """No two FALSE items in DIFFERENT families may share flaw phrasing.

    A recurring falsifying sentence is a template. It gives the lexical
    classifier something to learn, and it lets a reader who has met one item
    recognise the next by shape rather than by reading it.

    **Within a family the sharing is deliberate and is exempted here.** The
    matched-mechanism endpoint (D-024) compares coherent_false against
    diverse_false inside the ten scope families, and it holds by making the
    falsifying sentence *identical* across those cells so that only coherence
    differs. Breaking that identity would put phrasing back into the one
    endpoint built to have nothing in it but coherence. Cross-family recurrence
    has no such justification, so that is what this fails on. See D-032.
    """
    from src.flaws import flaw_sentence, shared_ngrams, similarity

    false_items = [i for i in items if not i.ground_truth]
    spans = {i.id: flaw_sentence(i.passage, i.flaw_mechanism) for i in false_items}
    fam = {i.id: i.family_id for i in false_items}
    mech = {i.id: i.flaw_mechanism for i in false_items}

    failures: list[str] = []
    worst = 0.0
    worst_pair = ("", "")
    n_within = 0
    for a, b in combinations(sorted(spans), 2):
        if fam[a] == fam[b]:
            n_within += 1
            continue
        r = similarity(spans[a], spans[b])
        if r > worst:
            worst, worst_pair = r, (a, b)
        shared = shared_ngrams(spans[a], spans[b], n)
        if r > limit:
            failures.append(
                f"{a} and {b} are falsified by near-identical wording "
                f"(similarity {r:.2f} > {limit}); they are in different families, "
                "so nothing requires them to match"
            )
        elif shared:
            failures.append(
                f"{a} and {b} share a {n}-gram in their falsifying text: "
                f"{sorted(shared)[0]!r}"
            )

    by_mech = Counter(mech.values())
    return CheckResult(
        name="no_duplicate_flaw_phrasing",
        passed=not failures,
        summary=(
            f"{len(false_items)} FALSE items, "
            f"{len(spans) * (len(spans) - 1) // 2 - n_within} cross-family pairs; "
            f"worst similarity {worst:.2f} (limit {limit}), no shared {n}-grams; "
            f"{n_within} within-family pairs exempt (matched mechanism, D-024)"
        ),
        required_by_brief=False,
        failures=failures,
        data={
            "limit": limit,
            "ngram": n,
            "worst_similarity": round(worst, 4),
            "worst_pair": list(worst_pair),
            "n_within_family_exempt": n_within,
            "by_mechanism": dict(by_mech),
        },
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
    "passage_word_balance",
    "flaw_declarations_complete",
    "matched_mechanism_subset",
    "no_duplicate_flaw_phrasing",
    "control_surface_match",
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
    closer_imbalance_tolerance: int = PASSAGE_IMBALANCE_TOLERANCE,
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
            "passage_word_balance",
            lambda: check_passage_word_balance(items, closer_imbalance_tolerance),
        ),
        ("flaw_declarations_complete", lambda: check_flaw_declarations(items)),
        ("matched_mechanism_subset", lambda: check_matched_mechanism_subset(items)),
        ("no_duplicate_flaw_phrasing", lambda: check_duplicate_flaw_phrasing(items)),
        ("control_surface_match", lambda: check_control_surface_match(items)),
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
        default=PASSAGE_IMBALANCE_TOLERANCE,
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
