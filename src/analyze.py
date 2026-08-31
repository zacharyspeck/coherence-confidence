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
from .models import CELLS

N_RESAMPLES = 10_000
ALPHA = 0.05
BOOTSTRAP_SEED = 12345

COHERENCE_LEVELS = ("coherent", "diverse")


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
        self.baseline = np.array([r.get("baseline_p_yes_3way", np.nan) for r in records], dtype=float)

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
    """Coherence x Truth on the item scores.

    Balanced-design formulas, written out rather than pulled from a library, for
    the same auditability reason as the AUC. Reported as a *breakdown*, not as a
    significance test to lean on: n is 20 per cell and items within a family are
    correlated, so the F-test's independence assumption is violated. The
    bootstrap CIs on the effect estimates are the number to trust.
    """
    scores = ds.scores(which)
    coh = ds.coherence == "coherent"

    cell_ns = {c: ds.idx_by_cell[c].size for c in CELLS}
    balanced = len(set(cell_ns.values())) == 1 and all(cell_ns.values())
    n = min(cell_ns.values())

    grand = float(np.mean(scores))
    means = {c: float(np.mean(scores[ds.idx_by_cell[c]])) for c in CELLS}
    m_coh = {
        "coherent": float(np.mean(scores[coh])),
        "diverse": float(np.mean(scores[~coh])),
    }
    m_truth = {
        True: float(np.mean(scores[ds.truth])),
        False: float(np.mean(scores[~ds.truth])),
    }

    ss_coh = n * 2 * sum((m - grand) ** 2 for m in m_coh.values())
    ss_truth = n * 2 * sum((m - grand) ** 2 for m in m_truth.values())
    ss_inter = 0.0
    for c in CELLS:
        ci = "coherent" if c.startswith("coherent") else "diverse"
        ti = c.endswith("_true")
        ss_inter += n * (means[c] - m_coh[ci] - m_truth[ti] + grand) ** 2
    ss_within = float(
        sum(
            float(np.sum((scores[ds.idx_by_cell[c]] - means[c]) ** 2))
            for c in CELLS
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


def analyze(
    records: Sequence[dict[str, Any]],
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    which: str = "3way",
) -> dict[str, Any]:
    ds = Dataset(records)
    boot = lambda st, note="": bootstrap(  # noqa: E731 - local alias for density
        ds, st, n_resamples=n_resamples, seed=seed, note=note
    )

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

    auc: dict[str, Any] = {}
    for coh in COHERENCE_LEVELS:
        sel = np.flatnonzero(ds.coherence == coh)
        if sel.size == 0:
            continue
        pos = ds.scores(which)[sel[ds.truth[sel]]]
        neg = ds.scores(which)[sel[~ds.truth[sel]]]
        detail = pairwise_auc_detail(pos.tolist(), neg.tolist())
        est = boot(
            stat_auc(ds, coh, which),
            note="abstained items INCLUDED, per DECISIONS.md D-003",
        )
        entry: dict[str, Any] = {
            "estimate": est.to_dict(),
            "detail": asdict(detail),
        }
        # Diagnostic only: what the abstention policy is protecting against.
        n_abst = int(np.count_nonzero(ds.abstained[sel]))
        if n_abst:
            try:
                dropped = stat_auc(ds, coh, which, drop_abstained=True)(
                    np.arange(len(ds))
                )
            except ValueError as exc:
                dropped = None
                entry["auc_if_abstained_dropped_error"] = str(exc)
            entry["auc_if_abstained_dropped_DIAGNOSTIC"] = dropped
            entry["n_abstained"] = n_abst
            entry["diagnostic_note"] = (
                "auc_if_abstained_dropped is NOT a reported result. It shows how "
                "much AUC would move if abstentions were discarded, which is why "
                "D-003 forbids discarding them."
            )
        else:
            entry["n_abstained"] = 0
        auc[coh] = entry

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
                "the flaw-type confound between coherent_false and diverse_false "
                "cannot touch this one."
            ),
        ).to_dict(),
        "coherence_effect_within_false": boot(
            stat_cell_contrast(ds, "coherent_false", "diverse_false", which),
            note=(
                "D-004: CONFOUNDED. coherent_false is broken by a shared confound "
                "and diverse_false by bad dates or claim mismatch, so this mixes "
                "coherence with flaw type."
            ),
        ).to_dict(),
    }

    flaw_breakdown: dict[str, Any] = {}
    for flaw in ("temporal", "claim_mismatch"):
        sel = [i for i, r in enumerate(records) if r.get("flaw_type") == flaw]
        if sel:
            arr = ds.scores(which)[np.array(sel)]
            flaw_breakdown[flaw] = {
                "n": len(sel),
                "mean": float(np.mean(arr)),
                "note": "D-004 split of diverse_false by how the item is broken",
            }

    return {
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
                "each family contributes exactly one item per cell. For the "
                "cross-cell contrasts (main_effect_coherence, interaction, "
                "coherence_effect_within_*): read ci_family. The 2x2 is "
                "within-family, so the clustered resample keeps the pairing and "
                "ci_item throws that pairing away."
            ),
        },
        "n_items": len(ds),
        "n_families": len(ds.families),
        "cell_n": {c: int(ds.idx_by_cell[c].size) for c in CELLS},
        "score_used": which,
        "cell_means_p_yes_3way": cell_means,
        "cell_means_p_yes_2way": cell_means_2way,
        "abstention_rate": abstention,
        "auc_within_condition": auc,
        "effects": effects,
        "anova": two_way_anova(ds, which),
        "diverse_false_flaw_breakdown": flaw_breakdown,
        "mass_covered": {
            "mean": float(np.mean(ds.mass)),
            "min": float(np.min(ds.mass)),
            "max": float(np.max(ds.mass)),
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _est_line(d: dict[str, Any]) -> str:
    lo, hi = d["ci_item"] or (float("nan"), float("nan"))
    flo, fhi = d["ci_family"] or (float("nan"), float("nan"))
    return f"{d['value']:.4f} | [{lo:.4f}, {hi:.4f}] | [{flo:.4f}, {fhi:.4f}]"


def to_markdown(analysis: dict[str, Any], meta: dict[str, Any]) -> str:
    L: list[str] = []
    a = analysis
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

    L.append("## Policy\n")
    for k, v in a["policy"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## Cell means — P(yes) three-way\n")
    L.append("| cell | n | mean | 95% CI (item) | 95% CI (family) |")
    L.append("|---|---|---|---|---|")
    for c in CELLS:
        d = a["cell_means_p_yes_3way"].get(c)
        if not d:
            continue
        lo, hi = d["ci_item"]
        flo, fhi = d["ci_family"]
        L.append(
            f"| `{c}` | {a['cell_n'][c]} | {d['value']:.4f} | "
            f"[{lo:.4f}, {hi:.4f}] | [{flo:.4f}, {fhi:.4f}] |"
        )
    L.append("")

    L.append("## Cell means — P(yes) two-way (yes vs no only)\n")
    L.append("| cell | mean | 95% CI (item) |")
    L.append("|---|---|---|")
    for c in CELLS:
        d = a["cell_means_p_yes_2way"].get(c)
        if not d:
            continue
        lo, hi = d["ci_item"]
        L.append(f"| `{c}` | {d['value']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    L.append("")

    L.append("## AUC within condition\n")
    L.append("20 true vs 20 false per condition = 400 pairs. Never pooled.\n")
    L.append("| condition | AUC | 95% CI (item) | 95% CI (family) | pairs | wins | ties |")
    L.append("|---|---|---|---|---|---|---|")
    for coh, e in a["auc_within_condition"].items():
        est, det = e["estimate"], e["detail"]
        lo, hi = est["ci_item"]
        flo, fhi = est["ci_family"]
        L.append(
            f"| {coh} | {est['value']:.4f} | [{lo:.4f}, {hi:.4f}] | "
            f"[{flo:.4f}, {fhi:.4f}] | {det['n_pairs']} | {det['n_wins']} | "
            f"{det['n_ties']} |"
        )
    L.append("")
    for coh, e in a["auc_within_condition"].items():
        if e.get("n_abstained"):
            drop = e.get("auc_if_abstained_dropped_DIAGNOSTIC")
            drop_s = f"{drop:.4f}" if isinstance(drop, float) else str(drop)
            L.append(
                f"- `{coh}`: {e['n_abstained']} abstained, all INCLUDED above. "
                f"If they were dropped the AUC would read **{drop_s}** — "
                "which is exactly why D-003 forbids dropping them."
            )
    L.append("")

    L.append("## Abstention rate per cell\n")
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

    L.append("## 2x2 effects\n")
    L.append(
        "> **Read the family-clustered CI for these.** The 2x2 is within-family, "
        "so clustering keeps the pairing; the item-level interval discards it "
        "and comes out too wide. (For the per-cell tables above, the two CIs are "
        "identical by construction.) See DECISIONS.md D-010.\n"
    )
    L.append("| effect | estimate | 95% CI (item) | **95% CI (family)** |")
    L.append("|---|---|---|---|")
    for k, d in a["effects"].items():
        lo, hi = d["ci_item"]
        flo, fhi = d["ci_family"]
        L.append(
            f"| {k} | {d['value']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
            f"[{flo:+.4f}, {fhi:+.4f}] |"
        )
    L.append("")
    for k, d in a["effects"].items():
        if d.get("note"):
            L.append(f"- **{k}** — {d['note']}")
    L.append("")

    L.append("## Two-way ANOVA-style breakdown\n")
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

    if a["diverse_false_flaw_breakdown"]:
        L.append("## diverse_false, split by flaw type (D-004)\n")
        L.append("| flaw_type | n | mean P(yes) |")
        L.append("|---|---|---|")
        for k, v in a["diverse_false_flaw_breakdown"].items():
            L.append(f"| {k} | {v['n']} | {v['mean']:.4f} |")
        L.append("")

    m = a["mass_covered"]
    L.append("## Coverage\n")
    L.append(
        f"- option mass covered: mean {m['mean']:.4f}, min {m['min']:.4f}, "
        f"max {m['max']:.4f}"
    )
    L.append(
        "- low coverage means the three options hold little of the next-token "
        "distribution and the renormalized numbers are ratios of small numbers."
    )
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    for coh, e in analysis["auc_within_condition"].items():
        print(f"  AUC[{coh}] = {e['estimate']['value']:.4f}  ({e['detail']['n_pairs']} pairs)")
    for k in ("main_effect_coherence", "main_effect_truth", "interaction"):
        print(f"  {k} = {analysis['effects'][k]['value']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
