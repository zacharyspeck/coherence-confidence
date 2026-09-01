"""Analysis: cell means, within-condition AUC, abstention, bootstrap CIs, 2x2.

Three things here are deliberate and load-bearing.

**AUC is computed WITHIN each coherence condition, never pooled.** 20 true vs 20
false = 400 pairs in the coherent condition, and separately 400 in the diverse
condition. Pooling all 80 items would let a coherence main effect - the very
thing under test - masquerade as discrimination, because coherent items would
sit above diverse items regardless of truth.

**AUC is an explicit pairwise win rate**, ties counting 0.5, written out as a
loop over pairs rather than delegated to sklearn, so the number can be audited
by hand. `tests/test_auc.py` asserts it matches `sklearn.roc_auc_score` to 1e-9;
the point of the assertion is that the readable implementation is the one that
ships and the library is the check, not the other way round.

**Abstained items are INCLUDED in the AUC, using their p_yes_3way** (D-003).
They are never dropped. Dropping them would let a model inflate its AUC by
abstaining on exactly the items it would have gotten wrong. The counterfactual
`auc_if_abstained_dropped` is computed and reported *as a diagnostic* so the
size of that effect is visible rather than hypothetical - it is not the headline
number and is labelled as such.

Every reported number carries a bootstrap 95% CI from 10,000 resamples, under
two resampling units (D-010): stratified-by-item, and clustered by family. Items
come in families of 4 and are not independent within a family, so the item-level
CI is anti-conservative; both are printed so the gap is visible.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import provenance
from .models import CELLS, CORE_CELLS

N_RESAMPLES = 10_000
ALPHA = 0.05
BOOTSTRAP_SEED = 12345

#: The two levels of the 2x2. The primary endpoint contrasts exactly these.
COHERENCE_LEVELS = ("coherent", "diverse")
#: Plus the surface-complexity control arm (D-030).
ALL_LEVELS = ("coherent", "diverse", "decorative")


# ---------------------------------------------------------------------------
# AUC: explicit pairwise win rate
# ---------------------------------------------------------------------------


@dataclass
class AUCDetail:
    """Every quantity needed to check the AUC by hand."""

    auc: float
    n_pos: int
    n_neg: int
    n_pairs: int
    n_wins: int
    n_ties: int
    n_losses: int

    def check(self) -> None:
        assert self.n_pairs == self.n_pos * self.n_neg
        assert self.n_wins + self.n_ties + self.n_losses == self.n_pairs
        expected = (self.n_wins + 0.5 * self.n_ties) / self.n_pairs
        assert abs(self.auc - expected) < 1e-12


def pairwise_auc_detail(pos: Sequence[float], neg: Sequence[float]) -> AUCDetail:
    """AUC as the literal probability that a random positive outranks a random
    negative, ties counting one half. Written as the double loop on purpose.

    `pos` = scores of items whose ground_truth is True.
    `neg` = scores of items whose ground_truth is False.
    """
    if not len(pos) or not len(neg):
        raise ValueError(
            f"AUC needs at least one item on each side; got {len(pos)} true and "
            f"{len(neg)} false"
        )

    wins = ties = losses = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
            else:
                losses += 1

    n_pairs = len(pos) * len(neg)
    detail = AUCDetail(
        auc=(wins + 0.5 * ties) / n_pairs,
        n_pos=len(pos),
        n_neg=len(neg),
        n_pairs=n_pairs,
        n_wins=wins,
        n_ties=ties,
        n_losses=losses,
    )
    detail.check()
    return detail


def pairwise_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    return pairwise_auc_detail(pos, neg).auc


def _auc_fast(pos: np.ndarray, neg: np.ndarray) -> float:
    """Vectorized twin of `pairwise_auc`, for the 10,000 bootstrap resamples.

    `tests/test_auc.py` asserts these two agree exactly on random inputs; the
    readable one is the definition, this one is an optimization of it.
    """
    diff = pos[:, None] - neg[None, :]
    wins = float(np.count_nonzero(diff > 0))
    ties = float(np.count_nonzero(diff == 0))
    return (wins + 0.5 * ties) / (pos.size * neg.size)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class Dataset:
    """Scored records in array form, plus the index bookkeeping the bootstrap
    needs. Nothing here interprets; it only slices."""

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("no records to analyze")
        self.records = list(records)
        self.item_ids = np.array([r["item_id"] for r in records])
        self.family_ids = np.array([r["family_id"] for r in records])
        self.cells = np.array([r["cell"] for r in records])
        self.coherence = np.array([r["coherence"] for r in records])
        self.truth = np.array([bool(r["ground_truth"]) for r in records])
        self.score = np.array([float(r["p_yes_3way"]) for r in records])
        self.score_2way = np.array([float(r["p_yes_2way"]) for r in records])
        self.abstained = np.array([bool(r["abstained"]) for r in records])
        self.mass = np.array([float(r["mass_covered"]) for r in records])
        self.baseline = np.array(
            [r.get("baseline_p_yes_3way", np.nan) for r in records], dtype=float
        )
        self.mechanism = np.array(
            [r.get("flaw_mechanism") or "" for r in records], dtype=object
        )
        self.surface = np.array(
            [
                float((r.get("surface_complexity") or {}).get("n_distinct_entities", np.nan))
                for r in records
            ],
            dtype=float,
        )
        self.salience = np.array(
            [
                float(r["salience"]) if r.get("salience") is not None else np.nan
                for r in records
            ],
            dtype=float,
        )

        self.families = sorted(set(self.family_ids.tolist()))
        self.idx_by_cell = {
            c: np.flatnonzero(self.cells == c) for c in CELLS
        }
        self.idx_by_family = {
            f: np.flatnonzero(self.family_ids == f) for f in self.families
        }

    def __len__(self) -> int:
        return len(self.records)

    def scores(self, which: str = "3way") -> np.ndarray:
        return {
            "3way": self.score,
            "2way": self.score_2way,
            "delta": self.score - self.baseline,
        }[which]

    # -- resampling units (D-010) -------------------------------------------

    def resample_items(self, rng: np.random.Generator) -> np.ndarray:
        """Stratified by cell: each cell is resampled to its own size, so cell
        n never drifts and the 2x2 stays balanced across resamples."""
        parts = [
            rng.choice(idx, size=idx.size, replace=True)
            for idx in self.idx_by_cell.values()
            if idx.size
        ]
        return np.concatenate(parts)

    def resample_families(self, rng: np.random.Generator) -> np.ndarray:
        """Cluster bootstrap: whole families are drawn with replacement, which
        respects the fact that the 4 cells of a family share a scenario."""
        chosen = rng.choice(len(self.families), size=len(self.families), replace=True)
        return np.concatenate([self.idx_by_family[self.families[i]] for i in chosen])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


@dataclass
class Estimate:
    """A number and what we know about how much to trust it."""

    value: float
    n: int
    ci_item: tuple[float, float] | None = None
    ci_family: tuple[float, float] | None = None
    n_resamples: int = N_RESAMPLES
    n_failed_resamples: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ci_item"] = list(self.ci_item) if self.ci_item else None
        d["ci_family"] = list(self.ci_family) if self.ci_family else None
        return d

    def fmt(self) -> str:
        if self.ci_item is None:
            return f"{self.value:.4f}"
        lo, hi = self.ci_item
        return f"{self.value:.4f} [{lo:.4f}, {hi:.4f}]"


def _percentile_ci(values: list[float], alpha: float = ALPHA) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def bootstrap(
    ds: Dataset,
    stat: Callable[[np.ndarray], float],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    note: str = "",
) -> Estimate:
    """Both resampling units, 10,000 resamples each.

    A resample that makes the statistic undefined (e.g. an AUC resample that
    happens to contain no false items) is counted and skipped rather than
    silently imputed - the count is reported.
    """
    point = stat(np.arange(len(ds)))

    out: dict[str, list[float]] = {"item": [], "family": []}
    failed = 0
    for unit, resampler in (
        ("item", ds.resample_items),
        ("family", ds.resample_families),
    ):
        rng = np.random.default_rng(seed)
        for _ in range(n_resamples):
            try:
                v = stat(resampler(rng))
            except (ValueError, ZeroDivisionError, IndexError):
                failed += 1
                continue
            if math.isfinite(v):
                out[unit].append(v)
            else:
                failed += 1

    return Estimate(
        value=float(point),
        n=len(ds),
        ci_item=_percentile_ci(out["item"]),
        ci_family=_percentile_ci(out["family"]),
        n_resamples=n_resamples,
        n_failed_resamples=failed,
        note=note,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def stat_cell_mean(ds: Dataset, cell: str, which: str = "3way") -> Callable:
    scores = ds.scores(which)

    def f(idx: np.ndarray) -> float:
        sel = idx[ds.cells[idx] == cell]
        if sel.size == 0:
            raise ValueError(f"resample contained no {cell} items")
        return float(np.mean(scores[sel]))

    return f


def stat_abstention_rate(ds: Dataset, cell: str) -> Callable:
    def f(idx: np.ndarray) -> float:
        sel = idx[ds.cells[idx] == cell]
        if sel.size == 0:
            raise ValueError(f"resample contained no {cell} items")
        return float(np.mean(ds.abstained[sel]))

    return f


def stat_auc(
    ds: Dataset, coherence: str, which: str = "3way", drop_abstained: bool = False
) -> Callable:
    """AUC within one coherence condition.

    `drop_abstained=True` exists only to quantify what the D-003 policy is
    protecting against. It is never the headline number.
    """
    scores = ds.scores(which)

    def f(idx: np.ndarray) -> float:
        sel = idx[ds.coherence[idx] == coherence]
        if drop_abstained:
            sel = sel[~ds.abstained[sel]]
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            raise ValueError(
                f"AUC within {coherence} needs both classes; got {pos.size} true, "
                f"{neg.size} false"
            )
        return _auc_fast(pos, neg)

    return f


def _mean_where(ds: Dataset, idx: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> float:
    sel = idx[mask[idx]]
    if sel.size == 0:
        raise ValueError("resample contained no items in one of the 2x2 margins")
    return float(np.mean(scores[sel]))


def stat_main_effect_coherence(ds: Dataset, which: str = "3way") -> Callable:
    """mean(coherent) - mean(diverse), collapsing over truth."""
    scores = ds.scores(which)
    coh = ds.coherence == "coherent"

    def f(idx: np.ndarray) -> float:
        return _mean_where(ds, idx, coh, scores) - _mean_where(ds, idx, ~coh, scores)

    return f


def stat_main_effect_truth(ds: Dataset, which: str = "3way") -> Callable:
    """mean(true) - mean(false), collapsing over coherence."""
    scores = ds.scores(which)

    def f(idx: np.ndarray) -> float:
        return _mean_where(ds, idx, ds.truth, scores) - _mean_where(
            ds, idx, ~ds.truth, scores
        )

    return f


def stat_interaction(ds: Dataset, which: str = "3way") -> Callable:
    """(coherent_true - coherent_false) - (diverse_true - diverse_false)."""
    scores = ds.scores(which)

    def cellmean(idx: np.ndarray, cell: str) -> float:
        sel = idx[ds.cells[idx] == cell]
        if sel.size == 0:
            raise ValueError(f"resample contained no {cell} items")
        return float(np.mean(scores[sel]))

    def f(idx: np.ndarray) -> float:
        return (
            cellmean(idx, "coherent_true") - cellmean(idx, "coherent_false")
        ) - (cellmean(idx, "diverse_true") - cellmean(idx, "diverse_false"))

    return f


def stat_cell_contrast(ds: Dataset, a: str, b: str, which: str = "3way") -> Callable:
    """mean(cell a) - mean(cell b)."""
    scores = ds.scores(which)

    def f(idx: np.ndarray) -> float:
        sa = idx[ds.cells[idx] == a]
        sb = idx[ds.cells[idx] == b]
        if sa.size == 0 or sb.size == 0:
            raise ValueError(f"resample missing {a} or {b}")
        return float(np.mean(scores[sa]) - np.mean(scores[sb]))

    return f


# ---------------------------------------------------------------------------
# Two-way ANOVA (2x2, balanced)
# ---------------------------------------------------------------------------


@dataclass
class AnovaRow:
    source: str
    ss: float
    df: int
    ms: float
    f: float | None
    p: float | None
    partial_eta_sq: float | None


def two_way_anova(ds: Dataset, which: str = "3way") -> dict[str, Any]:
    """Coherence x Truth on the item scores, over the FOUR CORE CELLS only.

    The decorative control is a third level of coherence, so including it would
    make this a 3x2 and change what the "main effect of coherence" means. The
    control is reported as its own comparison instead (D-030).

    Balanced-design formulas, written out rather than pulled from a library, for
    the same auditability reason as the AUC. Reported as a *breakdown*, not as a
    significance test to lean on: n is 20 per cell and items within a family are
    correlated, so the F-test's independence assumption is violated. The
    bootstrap CIs on the effect estimates are the number to trust.
    """
    scores = ds.scores(which)
    core = np.array([c in CORE_CELLS for c in ds.cells])
    coh = (ds.coherence == "coherent") & core
    div = (ds.coherence == "diverse") & core

    cell_ns = {c: ds.idx_by_cell[c].size for c in CORE_CELLS}
    balanced = len(set(cell_ns.values())) == 1 and all(cell_ns.values())
    n = min(cell_ns.values())

    grand = float(np.mean(scores[core]))
    means = {c: float(np.mean(scores[ds.idx_by_cell[c]])) for c in CORE_CELLS}
    m_coh = {
        "coherent": float(np.mean(scores[coh])),
        "diverse": float(np.mean(scores[div])),
    }
    m_truth = {
        True: float(np.mean(scores[ds.truth & core])),
        False: float(np.mean(scores[~ds.truth & core])),
    }

    ss_coh = n * 2 * sum((m - grand) ** 2 for m in m_coh.values())
    ss_truth = n * 2 * sum((m - grand) ** 2 for m in m_truth.values())
    ss_inter = 0.0
    for c in CORE_CELLS:
        ci = "coherent" if c.startswith("coherent") else "diverse"
        ti = c.endswith("_true")
        ss_inter += n * (means[c] - m_coh[ci] - m_truth[ti] + grand) ** 2
    ss_within = float(
        sum(
            float(np.sum((scores[ds.idx_by_cell[c]] - means[c]) ** 2))
            for c in CORE_CELLS
        )
    )

    df_within = sum(cell_ns.values()) - 4
    ms_within = ss_within / df_within if df_within > 0 else float("nan")

    try:
        from scipy import stats as _stats

        def pval(f: float) -> float:
            return float(_stats.f.sf(f, 1, df_within))

    except ImportError:  # pragma: no cover - scipy is a hard dep

        def pval(f: float) -> float:
            return float("nan")

    rows: list[AnovaRow] = []
    for source, ss in (
        ("coherence", ss_coh),
        ("truth", ss_truth),
        ("coherence x truth", ss_inter),
    ):
        f = ss / ms_within if ms_within and math.isfinite(ms_within) else None
        rows.append(
            AnovaRow(
                source=source,
                ss=ss,
                df=1,
                ms=ss,
                f=f,
                p=pval(f) if f is not None else None,
                partial_eta_sq=(ss / (ss + ss_within)) if (ss + ss_within) else None,
            )
        )
    rows.append(
        AnovaRow("residual", ss_within, df_within, ms_within, None, None, None)
    )

    return {
        "balanced": balanced,
        "cell_n": cell_ns,
        "grand_mean": grand,
        "cell_means": means,
        "marginal_means": {
            "coherent": m_coh["coherent"],
            "diverse": m_coh["diverse"],
            "true": m_truth[True],
            "false": m_truth[False],
        },
        "table": [asdict(r) for r in rows],
        "scope": "the four core cells only; the decorative control is not in this table",
        "caveat": (
            "F-tests assume independent observations. Items come in families of "
            "4 sharing a scenario, so they are not independent; treat these as a "
            "variance breakdown and read the bootstrap CIs on the effect "
            "estimates for inference."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


MECHANISMS = ("stated_confound", "broken_chronology", "scope_mismatch")


def _subset_families(ds: "Dataset", coherence: str, mechanism: str | None) -> set[str]:
    """Families whose FALSE item in this condition uses `mechanism`.

    `None` means no restriction. Returned as a family set rather than an item
    mask so the TRUE items come from exactly the same families - otherwise the
    matched comparison would contrast different scenarios, not different
    coherence.
    """
    if mechanism is None:
        return set(ds.families)
    cell = f"{coherence}_false"
    return {
        ds.family_ids[i]
        for i in range(len(ds))
        if ds.cells[i] == cell and ds.mechanism[i] == mechanism
    }


def stat_auc_subset(
    ds: "Dataset", coherence: str, mechanism: str | None, which: str = "3way"
) -> Callable:
    """AUC within one coherence condition, restricted to the families whose
    FALSE item in that condition uses `mechanism`."""
    scores = ds.scores(which)
    fams = _subset_families(ds, coherence, mechanism)
    keep = np.array([f in fams for f in ds.family_ids])

    def f(idx: np.ndarray) -> float:
        sel = idx[(ds.coherence[idx] == coherence) & keep[idx]]
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            raise ValueError(
                f"AUC within {coherence}/{mechanism} needs both classes; got "
                f"{pos.size} true, {neg.size} false"
            )
        return _auc_fast(pos, neg)

    return f


def stat_auc_gap(ds: "Dataset", mechanism: str | None, which: str = "3way") -> Callable:
    """THE PRIMARY ENDPOINT: AUC(coherent) - AUC(diverse).

    Koriat's consensuality principle predicts this is NEGATIVE - agreement among
    the retrieved considerations raises confidence without raising accuracy, so
    the confidence-accuracy relationship degrades exactly where the evidence
    agrees with itself. AUC(coherent) below 0.5 is the crossover: confidence
    running backwards against truth.
    """
    coh = stat_auc_subset(ds, "coherent", mechanism, which)
    div = stat_auc_subset(ds, "diverse", mechanism, which)

    def f(idx: np.ndarray) -> float:
        return coh(idx) - div(idx)

    return f


def _auc_block(
    ds: "Dataset",
    mechanism: str | None,
    which: str,
    boot: Callable,
    label: str,
) -> dict[str, Any]:
    """One AUC comparison: both conditions, the gap, and the salience covariate."""
    scores = ds.scores(which)
    out: dict[str, Any] = {"label": label, "mechanism": mechanism, "conditions": {}}
    fam_sets = {}

    for coh in COHERENCE_LEVELS:
        fams = _subset_families(ds, coh, mechanism)
        fam_sets[coh] = fams
        sel = np.array(
            [
                i
                for i in range(len(ds))
                if ds.coherence[i] == coh and ds.family_ids[i] in fams
            ],
            dtype=int,
        )
        if sel.size == 0:
            continue
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            continue
        detail = pairwise_auc_detail(pos.tolist(), neg.tolist())
        est = boot(stat_auc_subset(ds, coh, mechanism, which))
        false_idx = sel[~ds.truth[sel]]
        sal = ds.salience[false_idx]
        sal = sal[np.isfinite(sal)]
        out["conditions"][coh] = {
            "estimate": est.to_dict(),
            "detail": asdict(detail),
            "n_families": len(fams),
            "n_abstained": int(np.count_nonzero(ds.abstained[sel])),
            "mean_salience_of_false_items": float(np.mean(sal)) if sal.size else None,
            "n_with_salience": int(sal.size),
        }
        # Diagnostic only: what the D-003 policy is protecting against.
        n_abst = int(np.count_nonzero(ds.abstained[sel]))
        if n_abst:
            kept = sel[~ds.abstained[sel]]
            p2 = scores[kept[ds.truth[kept]]]
            n2 = scores[kept[~ds.truth[kept]]]
            out["conditions"][coh]["auc_if_abstained_dropped_DIAGNOSTIC"] = (
                _auc_fast(p2, n2) if p2.size and n2.size else None
            )
            out["conditions"][coh]["diagnostic_note"] = (
                "NOT a reported result. It shows how far the AUC would move if "
                "abstentions were discarded, which is why D-003 forbids it."
            )

    if len(out["conditions"]) == 2:
        gap = boot(
            stat_auc_gap(ds, mechanism, which),
            note="PRIMARY ENDPOINT. Negative = the consensuality prediction.",
        )
        out["gap_coherent_minus_diverse"] = gap.to_dict()
        s_coh = out["conditions"]["coherent"]["mean_salience_of_false_items"]
        s_div = out["conditions"]["diverse"]["mean_salience_of_false_items"]
        out["salience_gap_coherent_minus_diverse"] = (
            None if s_coh is None or s_div is None else s_coh - s_div
        )
        out["families_match_across_conditions"] = (
            fam_sets["coherent"] == fam_sets["diverse"]
        )
        out["families"] = {k: sorted(v) for k, v in fam_sets.items()}
    return out



# ---------------------------------------------------------------------------
# The surface-complexity control (D-030)
# ---------------------------------------------------------------------------


def control_families(ds: "Dataset") -> set[str]:
    """Families carrying the decorative arm."""
    return {
        ds.family_ids[i]
        for i in range(len(ds))
        if ds.coherence[i] == "decorative"
    }


def stat_auc_level(ds: "Dataset", level: str, fams: set[str], which: str) -> Callable:
    """AUC within one coherence LEVEL, restricted to a family set."""
    scores = ds.scores(which)
    keep = np.array([f in fams for f in ds.family_ids])

    def f(idx: np.ndarray) -> float:
        sel = idx[(ds.coherence[idx] == level) & keep[idx]]
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            raise ValueError(f"AUC for {level} needs both classes")
        return _auc_fast(pos, neg)

    return f


def control_comparison(
    ds: "Dataset", which: str, boot: Callable
) -> dict[str, Any]:
    """AUC for all three coherence levels, inside the control families only.

    Restricting to those families is what makes the three numbers comparable:
    same scenarios, same claims, same falsification mechanism. The only thing
    that differs is whether the four cases vary on dimensions that bear on the
    claim (diverse), on nothing at all (coherent), or only on surface detail
    (decorative).

    A decorative AUC sitting with coherent says the effect is about evidential
    independence. Sitting with diverse says it is about how much there is to
    parse, and the story is wrong.
    """
    fams = control_families(ds)
    if not fams:
        return {"present": False, "reason": "no decorative items in this run"}

    scores = ds.scores(which)
    keep = np.array([f in fams for f in ds.family_ids])
    out: dict[str, Any] = {"present": True, "n_families": len(fams), "levels": {}}

    for level in ALL_LEVELS:
        sel = np.array(
            [
                i
                for i in range(len(ds))
                if ds.coherence[i] == level and keep[i]
            ],
            dtype=int,
        )
        if sel.size == 0:
            continue
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            continue
        detail = pairwise_auc_detail(pos.tolist(), neg.tolist())
        est = boot(stat_auc_level(ds, level, fams, which))
        ent = ds.surface[sel]
        ent = ent[np.isfinite(ent)]
        out["levels"][level] = {
            "estimate": est.to_dict(),
            "detail": asdict(detail),
            "mean_distinct_entities": float(np.mean(ent)) if ent.size else None,
            "mean_distinct_condition_values": float(
                np.mean(
                    [
                        (ds.records[i].get("surface_complexity") or {}).get(
                            "n_distinct_condition_values", np.nan
                        )
                        for i in sel
                    ]
                )
            ),
        }

    lv = out["levels"]
    if {"coherent", "diverse", "decorative"} <= set(lv):
        a_dec = lv["decorative"]["estimate"]["value"]
        a_coh = lv["coherent"]["estimate"]["value"]
        a_div = lv["diverse"]["estimate"]["value"]
        d_coh, d_div = abs(a_dec - a_coh), abs(a_dec - a_div)

        out["gap_decorative_minus_coherent"] = boot(
            lambda idx: stat_auc_level(ds, "decorative", fams, which)(idx)
            - stat_auc_level(ds, "coherent", fams, which)(idx),
            note="near zero = the control behaves like a coherent item",
        ).to_dict()
        out["gap_decorative_minus_diverse"] = boot(
            lambda idx: stat_auc_level(ds, "decorative", fams, which)(idx)
            - stat_auc_level(ds, "diverse", fams, which)(idx),
            note="near zero = the control behaves like a diverse item",
        ).to_dict()

        if abs(d_coh - d_div) < 0.02:
            tracks, verdict = "neither", (
                "The decorative control sits between the coherent and diverse "
                f"cells (AUC {a_dec:.3f} against {a_coh:.3f} coherent and "
                f"{a_div:.3f} diverse), so this run does not separate evidential "
                "independence from surface complexity."
            )
        elif d_coh < d_div:
            tracks, verdict = "coherent", (
                f"The decorative control tracks the COHERENT cells (AUC {a_dec:.3f} "
                f"against {a_coh:.3f} coherent and {a_div:.3f} diverse), so the "
                "effect is about evidential independence and not about how much "
                "there is to parse."
            )
        else:
            tracks, verdict = "diverse", (
                f"The decorative control tracks the DIVERSE cells (AUC {a_dec:.3f} "
                f"against {a_div:.3f} diverse and {a_coh:.3f} coherent), so the "
                "effect is about surface complexity and the evidential-independence "
                "story is wrong."
            )
        out["tracks"] = tracks
        out["verdict"] = verdict
        out["distance_to_coherent"] = round(d_coh, 5)
        out["distance_to_diverse"] = round(d_div, 5)
    return out



# ---------------------------------------------------------------------------
# Covariates: salience and surface complexity
# ---------------------------------------------------------------------------


def covariate_models(ds: "Dataset", which: str = "3way") -> dict[str, Any]:
    """Does the coherence effect survive conditioning on the covariates?

    "Caught" means the model did not endorse a false claim: p_yes below 0.5 on a
    FALSE item. Three nested logistic models are fitted over the FALSE items:

        caught ~ coherent
        caught ~ coherent + salience
        caught ~ coherent + salience + surface complexity

    The coherence coefficient across those three is the answer to the two
    standing objections at once - "the coherent flaws are just louder" and "the
    diverse items are just harder to read". If it holds up as each covariate is
    added, neither explains it.

    The decorative arm answers the surface question more directly than any
    regression can, because it holds surface complexity fixed by construction
    rather than adjusting for it after the fact. This is the supporting check.
    """
    idx = np.array(
        [
            i
            for i in range(len(ds))
            if not ds.truth[i]
            and ds.coherence[i] in ("coherent", "diverse")
        ],
        dtype=int,
    )
    have_sal = idx[np.isfinite(ds.salience[idx])]
    have_all = have_sal[np.isfinite(ds.surface[have_sal])]

    if have_all.size < 8:
        return {
            "fitted": False,
            "reason": (
                f"only {have_all.size} FALSE items carry both salience and "
                "surface_complexity; run the blind audit, then "
                "scripts/apply_salience.py and scripts/apply_complexity.py"
            ),
        }

    scores = ds.scores(which)
    y = (scores[have_all] < 0.5).astype(int)
    if len(set(y.tolist())) < 2:
        return {
            "fitted": False,
            "reason": (
                f"catch outcome is constant ({int(y.sum())}/{y.size} caught); a "
                "logistic fit is undefined"
            ),
            "catch_rate": float(y.mean()),
        }

    coh = (ds.coherence[have_all] == "coherent").astype(float)
    sal = ds.salience[have_all]
    srf = ds.surface[have_all]
    # Standardised so the coefficients are comparable in size.
    def z(v: np.ndarray) -> np.ndarray:
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else v * 0.0

    from sklearn.linear_model import LogisticRegression

    def fit(X: np.ndarray) -> list[float]:
        m = LogisticRegression(max_iter=5000, C=np.inf)
        m.fit(X, y)
        return [float(m.intercept_[0]), *[float(c) for c in m.coef_[0]]]

    specs = {
        "coherence_only": np.column_stack([coh]),
        "plus_salience": np.column_stack([coh, z(sal)]),
        "plus_salience_and_surface": np.column_stack([coh, z(sal), z(srf)]),
    }
    out_models = {}
    for name, X in specs.items():
        c = fit(X)
        names = ["intercept", "coherent", "salience_z", "surface_z"][: X.shape[1] + 1]
        out_models[name] = dict(zip(names, c))

    coh_path = [out_models[k]["coherent"] for k in specs]
    survives = all(abs(v) > 0.1 and np.sign(v) == np.sign(coh_path[0]) for v in coh_path)

    return {
        "fitted": True,
        "n_false_items": int(have_all.size),
        "catch_rate": float(y.mean()),
        "outcome": "caught = p_yes_3way < 0.5 on a FALSE item",
        "note": "core cells only; the decorative arm is the direct control instead",
        "models": out_models,
        "coherence_coefficient_path": [round(v, 4) for v in coh_path],
        "coherence_survives_conditioning": bool(survives),
        "reading": (
            "Read the coherence coefficient down the three models. If it keeps "
            "its sign and size as salience and then surface complexity are added, "
            "neither covariate explains the coherence effect. If it collapses "
            "toward zero when a covariate enters, that covariate was doing the "
            "work."
        ),
    }


def salience_model(ds: "Dataset", which: str = "3way") -> dict[str, Any]:
    """Back-compat alias; the covariate models subsume it."""
    return covariate_models(ds, which)


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


MECHANISMS = ("stated_confound", "broken_chronology", "scope_mismatch")


def _subset_families(ds: "Dataset", coherence: str, mechanism: str | None) -> set[str]:
    """Families whose FALSE item in this condition uses `mechanism`.

    `None` means no restriction. Returned as a family set rather than an item
    mask so the TRUE items come from exactly the same families - otherwise the
    matched comparison would contrast different scenarios, not different
    coherence.
    """
    if mechanism is None:
        return set(ds.families)
    cell = f"{coherence}_false"
    return {
        ds.family_ids[i]
        for i in range(len(ds))
        if ds.cells[i] == cell and ds.mechanism[i] == mechanism
    }


def stat_auc_subset(
    ds: "Dataset", coherence: str, mechanism: str | None, which: str = "3way"
) -> Callable:
    """AUC within one coherence condition, restricted to the families whose
    FALSE item in that condition uses `mechanism`."""
    scores = ds.scores(which)
    fams = _subset_families(ds, coherence, mechanism)
    keep = np.array([f in fams for f in ds.family_ids])

    def f(idx: np.ndarray) -> float:
        sel = idx[(ds.coherence[idx] == coherence) & keep[idx]]
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            raise ValueError(
                f"AUC within {coherence}/{mechanism} needs both classes; got "
                f"{pos.size} true, {neg.size} false"
            )
        return _auc_fast(pos, neg)

    return f


def stat_auc_gap(ds: "Dataset", mechanism: str | None, which: str = "3way") -> Callable:
    """THE PRIMARY ENDPOINT: AUC(coherent) - AUC(diverse).

    Koriat's consensuality principle predicts this is NEGATIVE - agreement among
    the retrieved considerations raises confidence without raising accuracy, so
    the confidence-accuracy relationship degrades exactly where the evidence
    agrees with itself. AUC(coherent) below 0.5 is the crossover: confidence
    running backwards against truth.
    """
    coh = stat_auc_subset(ds, "coherent", mechanism, which)
    div = stat_auc_subset(ds, "diverse", mechanism, which)

    def f(idx: np.ndarray) -> float:
        return coh(idx) - div(idx)

    return f


def _auc_block(
    ds: "Dataset",
    mechanism: str | None,
    which: str,
    boot: Callable,
    label: str,
) -> dict[str, Any]:
    """One AUC comparison: both conditions, the gap, and the salience covariate."""
    scores = ds.scores(which)
    out: dict[str, Any] = {"label": label, "mechanism": mechanism, "conditions": {}}
    fam_sets = {}

    for coh in COHERENCE_LEVELS:
        fams = _subset_families(ds, coh, mechanism)
        fam_sets[coh] = fams
        sel = np.array(
            [
                i
                for i in range(len(ds))
                if ds.coherence[i] == coh and ds.family_ids[i] in fams
            ],
            dtype=int,
        )
        if sel.size == 0:
            continue
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            continue
        detail = pairwise_auc_detail(pos.tolist(), neg.tolist())
        est = boot(stat_auc_subset(ds, coh, mechanism, which))
        false_idx = sel[~ds.truth[sel]]
        sal = ds.salience[false_idx]
        sal = sal[np.isfinite(sal)]
        out["conditions"][coh] = {
            "estimate": est.to_dict(),
            "detail": asdict(detail),
            "n_families": len(fams),
            "n_abstained": int(np.count_nonzero(ds.abstained[sel])),
            "mean_salience_of_false_items": float(np.mean(sal)) if sal.size else None,
            "n_with_salience": int(sal.size),
        }
        # Diagnostic only: what the D-003 policy is protecting against.
        n_abst = int(np.count_nonzero(ds.abstained[sel]))
        if n_abst:
            kept = sel[~ds.abstained[sel]]
            p2 = scores[kept[ds.truth[kept]]]
            n2 = scores[kept[~ds.truth[kept]]]
            out["conditions"][coh]["auc_if_abstained_dropped_DIAGNOSTIC"] = (
                _auc_fast(p2, n2) if p2.size and n2.size else None
            )
            out["conditions"][coh]["diagnostic_note"] = (
                "NOT a reported result. It shows how far the AUC would move if "
                "abstentions were discarded, which is why D-003 forbids it."
            )

    if len(out["conditions"]) == 2:
        gap = boot(
            stat_auc_gap(ds, mechanism, which),
            note="PRIMARY ENDPOINT. Negative = the consensuality prediction.",
        )
        out["gap_coherent_minus_diverse"] = gap.to_dict()
        s_coh = out["conditions"]["coherent"]["mean_salience_of_false_items"]
        s_div = out["conditions"]["diverse"]["mean_salience_of_false_items"]
        out["salience_gap_coherent_minus_diverse"] = (
            None if s_coh is None or s_div is None else s_coh - s_div
        )
        out["families_match_across_conditions"] = (
            fam_sets["coherent"] == fam_sets["diverse"]
        )
        out["families"] = {k: sorted(v) for k, v in fam_sets.items()}
    return out



# ---------------------------------------------------------------------------
# The surface-complexity control (D-030)
# ---------------------------------------------------------------------------


def control_families(ds: "Dataset") -> set[str]:
    """Families carrying the decorative arm."""
    return {
        ds.family_ids[i]
        for i in range(len(ds))
        if ds.coherence[i] == "decorative"
    }


def stat_auc_level(ds: "Dataset", level: str, fams: set[str], which: str) -> Callable:
    """AUC within one coherence LEVEL, restricted to a family set."""
    scores = ds.scores(which)
    keep = np.array([f in fams for f in ds.family_ids])

    def f(idx: np.ndarray) -> float:
        sel = idx[(ds.coherence[idx] == level) & keep[idx]]
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            raise ValueError(f"AUC for {level} needs both classes")
        return _auc_fast(pos, neg)

    return f


def control_comparison(
    ds: "Dataset", which: str, boot: Callable
) -> dict[str, Any]:
    """AUC for all three coherence levels, inside the control families only.

    Restricting to those families is what makes the three numbers comparable:
    same scenarios, same claims, same falsification mechanism. The only thing
    that differs is whether the four cases vary on dimensions that bear on the
    claim (diverse), on nothing at all (coherent), or only on surface detail
    (decorative).

    A decorative AUC sitting with coherent says the effect is about evidential
    independence. Sitting with diverse says it is about how much there is to
    parse, and the story is wrong.
    """
    fams = control_families(ds)
    if not fams:
        return {"present": False, "reason": "no decorative items in this run"}

    scores = ds.scores(which)
    keep = np.array([f in fams for f in ds.family_ids])
    out: dict[str, Any] = {"present": True, "n_families": len(fams), "levels": {}}

    for level in ALL_LEVELS:
        sel = np.array(
            [
                i
                for i in range(len(ds))
                if ds.coherence[i] == level and keep[i]
            ],
            dtype=int,
        )
        if sel.size == 0:
            continue
        pos = scores[sel[ds.truth[sel]]]
        neg = scores[sel[~ds.truth[sel]]]
        if pos.size == 0 or neg.size == 0:
            continue
        detail = pairwise_auc_detail(pos.tolist(), neg.tolist())
        est = boot(stat_auc_level(ds, level, fams, which))
        ent = ds.surface[sel]
        ent = ent[np.isfinite(ent)]
        out["levels"][level] = {
            "estimate": est.to_dict(),
            "detail": asdict(detail),
            "mean_distinct_entities": float(np.mean(ent)) if ent.size else None,
            "mean_distinct_condition_values": float(
                np.mean(
                    [
                        (ds.records[i].get("surface_complexity") or {}).get(
                            "n_distinct_condition_values", np.nan
                        )
                        for i in sel
                    ]
                )
            ),
        }

    lv = out["levels"]
    if {"coherent", "diverse", "decorative"} <= set(lv):
        a_dec = lv["decorative"]["estimate"]["value"]
        a_coh = lv["coherent"]["estimate"]["value"]
        a_div = lv["diverse"]["estimate"]["value"]
        d_coh, d_div = abs(a_dec - a_coh), abs(a_dec - a_div)

        out["gap_decorative_minus_coherent"] = boot(
            lambda idx: stat_auc_level(ds, "decorative", fams, which)(idx)
            - stat_auc_level(ds, "coherent", fams, which)(idx),
            note="near zero = the control behaves like a coherent item",
        ).to_dict()
        out["gap_decorative_minus_diverse"] = boot(
            lambda idx: stat_auc_level(ds, "decorative", fams, which)(idx)
            - stat_auc_level(ds, "diverse", fams, which)(idx),
            note="near zero = the control behaves like a diverse item",
        ).to_dict()

        if abs(d_coh - d_div) < 0.02:
            tracks, verdict = "neither", (
                "The decorative control sits between the coherent and diverse "
                f"cells (AUC {a_dec:.3f} against {a_coh:.3f} coherent and "
                f"{a_div:.3f} diverse), so this run does not separate evidential "
                "independence from surface complexity."
            )
        elif d_coh < d_div:
            tracks, verdict = "coherent", (
                f"The decorative control tracks the COHERENT cells (AUC {a_dec:.3f} "
                f"against {a_coh:.3f} coherent and {a_div:.3f} diverse), so the "
                "effect is about evidential independence and not about how much "
                "there is to parse."
            )
        else:
            tracks, verdict = "diverse", (
                f"The decorative control tracks the DIVERSE cells (AUC {a_dec:.3f} "
                f"against {a_div:.3f} diverse and {a_coh:.3f} coherent), so the "
                "effect is about surface complexity and the evidential-independence "
                "story is wrong."
            )
        out["tracks"] = tracks
        out["verdict"] = verdict
        out["distance_to_coherent"] = round(d_coh, 5)
        out["distance_to_diverse"] = round(d_div, 5)
    return out



# ---------------------------------------------------------------------------
# Covariates: salience and surface complexity
# ---------------------------------------------------------------------------


def salience_model(ds: "Dataset", which: str = "3way") -> dict[str, Any]:
    """Logistic regression of catch-rate on salience and coherence.

    "Caught" means the model did not endorse a false claim: p_yes below 0.5 on a
    FALSE item. The question this answers is the one a sceptical reader asks
    first - how much of any AUC gap could be explained by the coherent flaws
    simply being louder? If the coherence coefficient survives conditioning on
    salience, the gap is not a salience artifact.
    """
    false_idx = np.array(
        [i for i in range(len(ds)) if not ds.truth[i] and np.isfinite(ds.salience[i])],
        dtype=int,
    )
    if false_idx.size < 8:
        return {
            "fitted": False,
            "reason": (
                f"only {false_idx.size} FALSE items carry a salience value; run the "
                "blind audit and scripts/apply_salience.py first"
            ),
        }

    scores = ds.scores(which)
    y = (scores[false_idx] < 0.5).astype(int)  # caught
    sal = ds.salience[false_idx]
    coh = (ds.coherence[false_idx] == "coherent").astype(float)

    if len(set(y.tolist())) < 2:
        return {
            "fitted": False,
            "reason": (
                f"catch outcome is constant ({int(y.sum())}/{y.size} caught); a "
                "logistic fit is undefined"
            ),
            "catch_rate": float(y.mean()),
        }

    from sklearn.linear_model import LogisticRegression

    def fit(X: np.ndarray) -> list[float]:
        m = LogisticRegression(max_iter=5000, C=np.inf)
        m.fit(X, y)
        return [float(m.intercept_[0]), *[float(c) for c in m.coef_[0]]]

    X_both = np.column_stack([sal, coh])
    coefs = fit(X_both)
    coefs_sal_only = fit(sal.reshape(-1, 1))
    coefs_coh_only = fit(coh.reshape(-1, 1))

    # Bootstrap the coefficients by resampling families, matching the rest of the
    # repo's inference (D-010).
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[list[float]] = []
    fam_of = {f: k for k, f in enumerate(ds.families)}
    by_fam: dict[int, list[int]] = {}
    for j, i in enumerate(false_idx):
        by_fam.setdefault(fam_of[ds.family_ids[i]], []).append(j)
    fam_keys = sorted(by_fam)
    for _ in range(2000):
        pick = rng.choice(len(fam_keys), size=len(fam_keys), replace=True)
        rows = [j for k in pick for j in by_fam[fam_keys[k]]]
        yb = y[rows]
        if len(set(yb.tolist())) < 2:
            continue
        try:
            m = LogisticRegression(max_iter=5000, C=np.inf)
            m.fit(X_both[rows], yb)
            draws.append([float(m.intercept_[0]), *[float(c) for c in m.coef_[0]]])
        except Exception:  # noqa: BLE001 - a degenerate resample is not an error
            continue

    arr = np.array(draws) if draws else np.zeros((0, 3))
    ci = (
        [list(map(float, np.percentile(arr[:, k], [2.5, 97.5]))) for k in range(3)]
        if arr.shape[0] > 50
        else [None, None, None]
    )

    return {
        "fitted": True,
        "n_false_items": int(false_idx.size),
        "catch_rate": float(y.mean()),
        "outcome": "caught = p_yes_3way < 0.5 on a FALSE item",
        "salience_mean": float(sal.mean()),
        "salience_range": [float(sal.min()), float(sal.max())],
        "model_caught_on_salience_and_coherence": {
            "intercept": coefs[0],
            "salience": coefs[1],
            "coherent": coefs[2],
            "ci_family": {"intercept": ci[0], "salience": ci[1], "coherent": ci[2]},
            "n_bootstrap_fits": int(arr.shape[0]),
        },
        "model_caught_on_salience_only": {
            "intercept": coefs_sal_only[0],
            "salience": coefs_sal_only[1],
        },
        "model_caught_on_coherence_only": {
            "intercept": coefs_coh_only[0],
            "coherent": coefs_coh_only[1],
        },
        "reading": (
            "A positive salience coefficient means louder flaws get caught more "
            "often, which is the artifact we are controlling for. The coherence "
            "coefficient in the two-predictor model is the coherence effect AFTER "
            "conditioning on salience: if it stays away from zero, the AUC gap is "
            "not simply the coherent flaws being easier to see."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------


def analyze(
    records: Sequence[dict[str, Any]],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    which: str = "3way",
) -> dict[str, Any]:
    ds = Dataset(records)

    def boot(st, note=""):
        return bootstrap(ds, st, n_resamples=n_resamples, seed=seed, note=note)

    # ---- PRIMARY ENDPOINT ---------------------------------------------------
    primary = _auc_block(
        ds, None, which, boot, "FULL SET - all 20 true vs 20 false per condition"
    )
    matched = _auc_block(
        ds,
        "scope_mismatch",
        which,
        boot,
        "MATCHED MECHANISM - scope_mismatch in both conditions",
    )
    control = control_comparison(ds, which, boot)

    by_mechanism = {}
    for m in MECHANISMS:
        block = _auc_block(ds, m, which, boot, f"mechanism = {m}")
        if block["conditions"]:
            by_mechanism[m] = block

    # ---- diagnostics --------------------------------------------------------
    cell_means = {
        c: boot(stat_cell_mean(ds, c, which)).to_dict()
        for c in CELLS
        if ds.idx_by_cell[c].size
    }
    cell_means_2way = {
        c: boot(stat_cell_mean(ds, c, "2way")).to_dict()
        for c in CELLS
        if ds.idx_by_cell[c].size
    }
    abstention = {
        c: boot(stat_abstention_rate(ds, c)).to_dict()
        for c in CELLS
        if ds.idx_by_cell[c].size
    }

    effects = {
        "main_effect_coherence": boot(
            stat_main_effect_coherence(ds, which),
            note="mean(coherent) - mean(diverse), collapsing over truth",
        ).to_dict(),
        "main_effect_truth": boot(
            stat_main_effect_truth(ds, which),
            note="mean(true) - mean(false), collapsing over coherence",
        ).to_dict(),
        "interaction": boot(
            stat_interaction(ds, which),
            note="(coh_true - coh_false) - (div_true - div_false)",
        ).to_dict(),
        "coherence_effect_within_true": boot(
            stat_cell_contrast(ds, "coherent_true", "diverse_true", which),
            note=(
                "D-004: the CLEAN coherence contrast. Both cells are flawless, so "
                "neither the flaw-type confound nor flaw salience can touch it."
            ),
        ).to_dict(),
        "coherence_effect_within_false": boot(
            stat_cell_contrast(ds, "coherent_false", "diverse_false", which),
            note="D-004: mixes coherence with whatever differs between the mechanisms.",
        ).to_dict(),
    }

    mech_counts: dict[str, dict[str, int]] = {}
    for i in range(len(ds)):
        if ds.mechanism[i]:
            mech_counts.setdefault(ds.cells[i], {}).setdefault(ds.mechanism[i], 0)
            mech_counts[ds.cells[i]][ds.mechanism[i]] += 1

    sal_by_cell: dict[str, Any] = {}
    for c in ("coherent_false", "diverse_false"):
        idx = ds.idx_by_cell[c]
        v = ds.salience[idx]
        v = v[np.isfinite(v)]
        sal_by_cell[c] = {
            "n": int(v.size),
            "mean": float(np.mean(v)) if v.size else None,
            "max": float(np.max(v)) if v.size else None,
        }

    return {
        "endpoints": {
            "primary": (
                "AUC(coherent) - AUC(diverse) on the FULL set. Koriat's "
                "consensuality principle predicts NEGATIVE: coherence raises "
                "confidence without raising accuracy, so the "
                "confidence-accuracy relationship degrades where the evidence "
                "agrees with itself. AUC(coherent) below 0.5 is the crossover."
            ),
            "confound_controlled": (
                "The same gap on the MATCHED-MECHANISM subset, where both "
                "conditions are falsified by scope_mismatch, so flaw mechanism "
                "cannot differ between them and only coherence does (D-024)."
            ),
            "surface_complexity_control": (
                "AUC for all three coherence levels inside the 10 families that "
                "carry the control arm. A decorative item is as busy to read as a "
                "diverse one and as evidentially dependent as a coherent one, so "
                "which of the two its AUC sits with says whether the effect is "
                "about evidential independence or about parse load (D-030)."
            ),
            "diagnostics_not_endpoints": (
                "Cell means, the 2x2 effects and the ANOVA are diagnostics. They "
                "describe confidence level; the endpoint is the "
                "confidence-accuracy relationship, which is the AUC."
            ),
        },
        "policy": {
            "abstained_items_in_auc": "INCLUDED (D-003) using p_yes_3way",
            "auc_scope": "WITHIN each coherence condition, never pooled",
            "auc_implementation": "explicit pairwise win rate, ties = 0.5",
            "primary_measure": "p_yes_3way = p_yes / (p_yes + p_no + p_unsure)",
            "bootstrap": (
                f"{n_resamples} resamples, percentile 95% CI, two units: "
                "stratified-by-item and clustered-by-family (D-010)"
            ),
            "which_ci_to_read": (
                "Per-cell numbers (cell means, abstention rates, within-condition "
                "AUC): ci_item and ci_family are IDENTICAL by construction, since "
                "each family contributes exactly one item per cell. For the AUC "
                "GAP and the cross-cell contrasts: read ci_family. The 2x2 is "
                "within-family, so the clustered resample keeps the pairing and "
                "ci_item throws that pairing away."
            ),
        },
        "n_items": len(ds),
        "n_families": len(ds.families),
        "cell_n": {c: int(ds.idx_by_cell[c].size) for c in CELLS},
        "score_used": which,
        "auc_primary_full_set": primary,
        "auc_matched_mechanism": matched,
        "auc_by_mechanism": by_mechanism,
        "surface_complexity_control": control,
        "mechanism_counts_by_cell": mech_counts,
        "salience_by_cell": sal_by_cell,
        "covariate_models": covariate_models(ds, which),
        "cell_means_p_yes_3way": cell_means,
        "cell_means_p_yes_2way": cell_means_2way,
        "abstention_rate": abstention,
        "effects": effects,
        "anova": two_way_anova(ds, which),
        "mass_covered": {
            "mean": float(np.mean(ds.mass)),
            "min": float(np.min(ds.mass)),
            "max": float(np.max(ds.mass)),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_ci(ci) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _auc_table(block: dict[str, Any]) -> list[str]:
    L: list[str] = []
    L.append("| condition | AUC | 95% CI (family) | pairs | wins | ties | mean salience |")
    L.append("|---|---|---|---|---|---|---|")
    for coh in COHERENCE_LEVELS:
        e = block["conditions"].get(coh)
        if not e:
            continue
        est, det = e["estimate"], e["detail"]
        sal = e["mean_salience_of_false_items"]
        L.append(
            f"| {coh} | **{est['value']:.4f}** | "
            f"{_fmt_ci(est['ci_family'])} | {det['n_pairs']} | {det['n_wins']} | "
            f"{det['n_ties']} | {'—' if sal is None else f'{sal:.2f}'} |"
        )
    gap = block.get("gap_coherent_minus_diverse")
    if gap:
        sg = block.get("salience_gap_coherent_minus_diverse")
        L.append("")
        L.append(
            f"**AUC(coherent) − AUC(diverse) = {gap['value']:+.4f}**  "
            f"95% CI (family) {_fmt_ci(gap['ci_family'])}"
        )
        lo, hi = gap["ci_family"] or (float("nan"), float("nan"))
        if gap["value"] < 0:
            direction = "consistent with the consensuality prediction"
        elif gap["value"] > 0:
            direction = "OPPOSITE to the consensuality prediction"
        else:
            direction = "exactly zero — neither direction"
        crosses = lo <= 0 <= hi
        L.append(
            f"— sign is {direction}"
            + (
                "; the interval spans zero, so the direction is not resolved."
                if crosses
                else "; the interval excludes zero."
            )
        )
        for coh, e in block["conditions"].items():
            v = e["estimate"]["value"]
            if v < 0.5:
                L.append(
                    f"— **AUC({coh}) = {v:.4f} is below 0.5**: confidence runs "
                    "backwards against truth in that condition. That is the "
                    "crossover, not merely a smaller effect."
                )
        if sg is not None:
            if abs(sg) < 0.10:
                reading = (
                    "effectively equalized, so this AUC gap is not a salience "
                    "artifact in either direction"
                )
            elif sg > 0:
                reading = (
                    "the coherent flaws were louder, which pushes this AUC gap "
                    "upward and therefore *against* the hypothesis - the test is "
                    "conservative here"
                )
            else:
                reading = (
                    "the DIVERSE flaws were louder, which pushes this AUC gap "
                    "downward and therefore *towards* the hypothesis - discount "
                    "accordingly"
                )
            L.append(f"— salience gap over the same items: {sg:+.2f} of 5 — {reading}.")
        if not block.get("families_match_across_conditions", True):
            L.append(
                "— **WARNING:** the two conditions do not draw on the same "
                "families here, so scenario content differs between them as well "
                "as coherence."
            )
    return L


def to_markdown(analysis: dict[str, Any], meta: dict[str, Any]) -> str:
    a = analysis
    L: list[str] = []
    L.append("# coherence-confidence — analysis\n")
    L.append(f"- run: `{meta.get('run_id', '?')}`")
    L.append(f"- model: `{meta.get('model', '?')}` (revision `{meta.get('revision')}`)")
    L.append(f"- template hash: `{meta.get('prompt_template_hash')}`")
    L.append(f"- options: `{meta.get('option_words')}`")
    L.append(f"- scored: {a['n_items']} items in {a['n_families']} families")
    L.append(f"- timestamp: {meta.get('timestamp_utc')}")
    if str(meta.get("model", "")).startswith("mock"):
        L.append("\n> **SYNTHETIC RUN.** These are fixtures, not measurements.\n")
    L.append("")

    # ---- 1. PRIMARY ---------------------------------------------------------
    L.append("## 1. PRIMARY ENDPOINT — AUC(coherent) vs AUC(diverse)\n")
    L.append(f"> {a['endpoints']['primary']}\n")
    L += _auc_table(a["auc_primary_full_set"])
    L.append("")

    ctrl = a.get("surface_complexity_control") or {}
    if ctrl.get("verdict"):
        L.append(f"**Surface-complexity control: {ctrl['verdict']}**")
        L.append(
            f"(decorative AUC is {ctrl['distance_to_coherent']:.3f} from coherent "
            f"and {ctrl['distance_to_diverse']:.3f} from diverse; section 3 has "
            "the table.)"
        )
    elif ctrl.get("present") is False:
        L.append(
            "**Surface-complexity control: not run** — "
            f"{ctrl.get('reason', 'no decorative items')}."
        )
    L.append("")

    # ---- 2. CONFOUND-CONTROLLED --------------------------------------------
    L.append("## 2. The same endpoint, confound-controlled\n")
    L.append(f"> {a['endpoints']['confound_controlled']}\n")
    m = a["auc_matched_mechanism"]
    if m["conditions"]:
        L += _auc_table(m)
        L.append("")
        L.append(
            "Both conditions here are falsified by the *same* mechanism, so a gap "
            "cannot be attributed to one cell's flaws being a different kind of "
            "thing from the other's. This is the number to quote when asked "
            "whether the coherent-false items are simply easier to catch."
        )
    else:
        L.append(
            "_No matched-mechanism subset available — no `scope_mismatch` items "
            "in one or both conditions._"
        )
    L.append("")

    # ---- 3. THE CONTROL -----------------------------------------------------
    L.append("## 3. Surface-complexity control\n")
    L.append(f"> {a['endpoints']['surface_complexity_control']}\n")
    if not ctrl.get("present"):
        L.append(f"_Not run: {ctrl.get('reason', 'no decorative items')}._\n")
    else:
        L.append(
            "| level | AUC | 95% CI (family) | pairs | distinct entities | condition values |"
        )
        L.append("|---|---|---|---|---|---|")
        for level in ALL_LEVELS:
            e = ctrl["levels"].get(level)
            if not e:
                continue
            ent = e["mean_distinct_entities"]
            L.append(
                f"| {level} | **{e['estimate']['value']:.4f}** | "
                f"{_fmt_ci(e['estimate']['ci_family'])} | {e['detail']['n_pairs']} | "
                f"{'—' if ent is None else f'{ent:.1f}'} | "
                f"{e['mean_distinct_condition_values']:.1f} |"
            )
        L.append("")
        L.append(
            "Read the last two columns together. The decorative row matches "
            "**diverse** on distinct entities and **coherent** on condition "
            "values - that is the control working. Which AUC it lands nearer is "
            "the result."
        )
        L.append("")
        for key, label in (
            ("gap_decorative_minus_coherent", "decorative − coherent"),
            ("gap_decorative_minus_diverse", "decorative − diverse"),
        ):
            g = ctrl.get(key)
            if g:
                L.append(
                    f"- **{label} = {g['value']:+.4f}**  95% CI (family) "
                    f"{_fmt_ci(g['ci_family'])}"
                )
        L.append("")
        L.append(f"**{ctrl['verdict']}**")
    L.append("")

    # ---- 4. PER MECHANISM ---------------------------------------------------
    L.append("## 4. Per-mechanism breakdown\n")
    counts = a.get("mechanism_counts_by_cell", {})
    if counts:
        L.append("| cell | " + " | ".join(MECHANISMS) + " |")
        L.append("|---" * (len(MECHANISMS) + 1) + "|")
        for cell in ("coherent_false", "diverse_false"):
            row = counts.get(cell, {})
            L.append(
                f"| `{cell}` | " + " | ".join(str(row.get(x, 0)) for x in MECHANISMS) + " |"
            )
        L.append("")
    for mech, block in a["auc_by_mechanism"].items():
        L.append(f"### `{mech}`\n")
        L += _auc_table(block)
        L.append("")

    # ---- 4. SALIENCE --------------------------------------------------------
    L.append("## 5. Salience, reported as a covariate\n")
    sal = a["salience_by_cell"]
    L.append("| cell | n with salience | mean | max |")
    L.append("|---|---|---|---|")
    for c, v in sal.items():
        mv = "—" if v["mean"] is None else f"{v['mean']:.2f}"
        xv = "—" if v["max"] is None else f"{v['max']:.2f}"
        L.append(f"| `{c}` | {v['n']} | {mv} | {xv} |")
    L.append("")
    sm = a["covariate_models"]
    if not sm.get("fitted"):
        L.append(f"_Conditioning models not fitted: {sm.get('reason')}_\n")
    else:
        L.append(
            f"Logistic models of **catch-rate** ({sm['outcome']}) over "
            f"{sm['n_false_items']} FALSE items in the core cells; overall catch "
            f"rate {sm['catch_rate']:.1%}.\n"
        )
        L.append("| model | coherent | salience (z) | surface (z) |")
        L.append("|---|---|---|---|")
        for name, m in sm["models"].items():
            L.append(
                f"| `{name}` | **{m['coherent']:+.3f}** | "
                + (f"{m['salience_z']:+.3f}" if "salience_z" in m else "—")
                + " | "
                + (f"{m['surface_z']:+.3f}" if "surface_z" in m else "—")
                + " |"
            )
        L.append("")
        L.append(
            f"Coherence coefficient across the three: "
            f"{sm['coherence_coefficient_path']}. "
            + (
                "**It holds its sign and size, so neither salience nor surface "
                "complexity explains it.**"
                if sm["coherence_survives_conditioning"]
                else "**It does not hold up under conditioning - read the path "
                "above to see which covariate absorbs it.**"
            )
        )
        L.append("")
        L.append(f"> {sm['reading']}")
    L.append("")

    # ---- 6. DIAGNOSTICS -----------------------------------------------------
    L.append("## 6. Diagnostics (not endpoints)\n")
    L.append(f"> {a['endpoints']['diagnostics_not_endpoints']}\n")

    L.append("### Mean confidence per cell — P(yes) three-way\n")
    L.append("| cell | n | mean | 95% CI (item) |")
    L.append("|---|---|---|---|")
    for c in CELLS:
        d = a["cell_means_p_yes_3way"].get(c)
        if not d:
            continue
        lo, hi = d["ci_item"]
        L.append(
            f"| `{c}` | {a['cell_n'][c]} | {d['value']:.4f} | [{lo:.4f}, {hi:.4f}] |"
        )
    L.append("")

    L.append("### Mean confidence per cell — P(yes) two-way (yes vs no only)\n")
    L.append("| cell | mean | 95% CI (item) |")
    L.append("|---|---|---|")
    for c in CELLS:
        d = a["cell_means_p_yes_2way"].get(c)
        if not d:
            continue
        lo, hi = d["ci_item"]
        L.append(f"| `{c}` | {d['value']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    L.append("")

    L.append("### Abstention rate per cell\n")
    L.append("Fraction of items where the third option has the highest probability.\n")
    L.append("| cell | rate | 95% CI (item) |")
    L.append("|---|---|---|")
    for c in CELLS:
        d = a["abstention_rate"].get(c)
        if not d:
            continue
        lo, hi = d["ci_item"]
        L.append(f"| `{c}` | {d['value']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    L.append("")
    for name, block in (
        ("full set", a["auc_primary_full_set"]),
        ("matched subset", a["auc_matched_mechanism"]),
    ):
        for coh, e in block.get("conditions", {}).items():
            if e.get("n_abstained"):
                L.append(
                    f"- {name}, `{coh}`: {e['n_abstained']} abstained, all INCLUDED "
                    "in the AUC above (D-003)."
                )
    L.append("")

    L.append("### 2x2 effects on mean confidence\n")
    L.append(
        "> Read the family-clustered CI for these. The 2x2 is within-family, so "
        "clustering keeps the pairing; the item-level interval discards it. "
        "See DECISIONS.md D-010.\n"
    )
    L.append("| effect | estimate | 95% CI (item) | **95% CI (family)** |")
    L.append("|---|---|---|---|")
    for k, d in a["effects"].items():
        L.append(
            f"| {k} | {d['value']:+.4f} | {_fmt_ci(d['ci_item'])} | "
            f"{_fmt_ci(d['ci_family'])} |"
        )
    L.append("")
    for k, d in a["effects"].items():
        if d.get("note"):
            L.append(f"- **{k}** — {d['note']}")
    L.append("")

    L.append("### Two-way ANOVA-style breakdown\n")
    L.append("| source | SS | df | MS | F | p | partial eta^2 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in a["anova"]["table"]:
        f = f"{r['f']:.3f}" if r["f"] is not None else ""
        p = f"{r['p']:.5f}" if r["p"] is not None else ""
        pe = f"{r['partial_eta_sq']:.4f}" if r["partial_eta_sq"] is not None else ""
        L.append(
            f"| {r['source']} | {r['ss']:.5f} | {r['df']} | {r['ms']:.5f} | "
            f"{f} | {p} | {pe} |"
        )
    L.append(f"\n> {a['anova']['caveat']}\n")

    mm = a["mass_covered"]
    L.append("### Coverage\n")
    L.append(
        f"- option mass covered: mean {mm['mean']:.4f}, min {mm['min']:.4f}, "
        f"max {mm['max']:.4f}"
    )
    L.append(
        "- low coverage means the three options hold little of the next-token "
        "distribution and the renormalized numbers are ratios of small numbers."
    )
    return "\n".join(L) + "\n"


def attach_baseline(
    records: list[dict[str, Any]], baseline_path: str | Path
) -> list[dict[str, Any]]:
    """Join a baseline run (claims with no cases) onto the evidence records."""
    payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    by_family = {r["family_id"]: r for r in payload["records"]}
    missing = sorted({r["family_id"] for r in records} - set(by_family))
    if missing:
        raise SystemExit(
            f"baseline run is missing {len(missing)} families, e.g. {missing[:3]}"
        )
    for r in records:
        b = by_family[r["family_id"]]
        r["baseline_p_yes_3way"] = b["p_yes_3way"]
        r["baseline_p_yes_2way"] = b["p_yes_2way"]
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="results/*.json from src.score")
    ap.add_argument("--out", required=True, help="analysis JSON path")
    ap.add_argument("--markdown", default=None, help="defaults to --out with .md")
    ap.add_argument("--baseline", default=None, help="results/*.json from src.baseline")
    ap.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument(
        "--score",
        default="3way",
        choices=["3way", "2way", "delta"],
        help="'delta' subtracts the baseline and requires --baseline",
    )
    args = ap.parse_args(argv)

    # Argument errors before file I/O, so a bad invocation fails on the argument
    # rather than on a missing file three lines later.
    if args.score == "delta" and not args.baseline:
        raise SystemExit("--score delta requires --baseline")

    payload = json.loads(Path(args.run).read_text(encoding="utf-8"))
    records = payload["records"]
    if args.baseline:
        records = attach_baseline(records, args.baseline)

    analysis = analyze(
        records, n_resamples=args.n_resamples, seed=args.seed, which=args.score
    )
    meta = dict(payload.get("meta", {}))
    meta["run_id"] = payload.get("run_id")

    out_payload = {
        "kind": "analysis",
        "analysis_meta": provenance.run_meta(
            source_run=str(args.run),
            source_run_id=payload.get("run_id"),
            baseline_run=args.baseline,
            n_resamples=args.n_resamples,
            bootstrap_seed=args.seed,
            score_used=args.score,
        ),
        "source_meta": meta,
        "analysis": analysis,
    }
    out = provenance.write_json(args.out, out_payload)
    md_path = Path(args.markdown or str(Path(args.out).with_suffix(".md")))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(to_markdown(analysis, meta), encoding="utf-8")

    print(f"wrote {out}")
    print(f"wrote {md_path}")
    for label, block in (
        ("PRIMARY", analysis["auc_primary_full_set"]),
        ("MATCHED", analysis["auc_matched_mechanism"]),
    ):
        conds = block.get("conditions") or {}
        if not conds:
            print(f"  {label}: no items in this subset")
            continue
        parts = " ".join(
            f"{c}={e['estimate']['value']:.4f}({e['detail']['n_pairs']}p)"
            for c, e in conds.items()
        )
        gap = block.get("gap_coherent_minus_diverse")
        gtxt = ""
        if gap:
            lo, hi = gap["ci_family"] or (float("nan"), float("nan"))
            gtxt = f"  gap={gap['value']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"  {label}: {parts}{gtxt}")
    print("  (diagnostics) " + ", ".join(
        f"{k}={analysis['effects'][k]['value']:+.4f}"
        for k in ("main_effect_coherence", "main_effect_truth", "interaction")
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
