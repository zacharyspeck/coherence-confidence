"""Deterministic mock scorer with hand-computable answers.

Step 6 of the brief: prove the plumbing works before the content exists. The
mock returns *known* probabilities, chosen so that every number the analysis
reports can be derived on paper and asserted exactly.

The `known` scenario, which is the default
------------------------------------------
Within each coherence condition the 20 TRUE and 20 FALSE items are ordered by
family_id and assigned `p_yes_3way` as follows (m = 5 for coherent, 9 for
diverse):

    TRUE,  index i < m   ->  0.050 + 0.001*i     (below every FALSE item)
    TRUE,  index i >= m  ->  0.800 + 0.001*i     (above every FALSE item)
    FALSE, index j       ->  0.300 + 0.001*j     (strictly between the two)

So exactly (20 - m) of the TRUE items beat all 20 FALSE items and the other m
lose to all 20:

    AUC = (20 - m) * 20 / 400 = (20 - m) / 20
    coherent: (20-5)/20 = 0.75      diverse: (20-9)/20 = 0.55

Cell means fall out the same way, e.g.

    coherent_true = (5*0.050 + 0.001*10 + 15*0.800 + 0.001*180) / 20 = 0.6220
    *_false       = (20*0.300 + 0.001*190) / 20                     = 0.3095

Abstention is defined as `p_yes_3way < 0.2`, which picks out exactly the m
low-scoring TRUE items - the ones the model got *wrong*. That makes the
abstention policy (D-003) testable: including them gives AUC 0.75 in the
coherent condition, dropping them gives 1.00. Dropping abstentions inflates.

Raw masses are the three-way probabilities scaled by `MASS_COVERED = 0.6`, so
the renormalization path in `ScoreResult` is genuinely exercised rather than
being an identity.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .models import Item
from .render import (
    DEFAULT_OPTIONS,
    render_baseline_prompt,
    render_prompt,
    template_hash,
)
from .scoring_types import ScoreResult

#: Fraction of next-token mass the three options hold, in the mock.
MASS_COVERED = 0.6

#: p_yes_3way below this counts as an abstention in the `known` scenario.
ABSTAIN_THRESHOLD = 0.2

#: Number of TRUE items placed below every FALSE item, per condition.
M_BY_COHERENCE = {"coherent": 5, "diverse": 9}

EXPECTED_AUC = {
    "coherent": (20 - M_BY_COHERENCE["coherent"]) / 20,  # 0.75
    "diverse": (20 - M_BY_COHERENCE["diverse"]) / 20,  # 0.55
}

SCENARIOS = ("known", "separable", "ties", "coherence_driven", "truth_driven")


class MockScorerError(RuntimeError):
    pass


def _split_probs(s: float, abstain: bool) -> tuple[float, float, float]:
    """Turn a target p_yes_3way into three raw masses.

    Non-abstaining items put most of the remainder on No; abstaining items put
    most of it on Unsure, which makes Unsure the argmax whenever s < 0.41.
    """
    rest = 1.0 - s
    if abstain:
        p_no, p_unsure = rest * 0.3, rest * 0.7
        if p_unsure <= s:
            raise MockScorerError(
                f"cannot construct an abstention at p_yes_3way={s:.3f}: Unsure "
                f"would be {p_unsure:.3f}, not the argmax. Abstention requires "
                "s < 0.4117."
            )
    else:
        p_no, p_unsure = rest * 0.7, rest * 0.3
    return s * MASS_COVERED, p_no * MASS_COVERED, p_unsure * MASS_COVERED


def _target_p_yes(item: Item, rank: int, scenario: str) -> tuple[float, bool]:
    """(target p_yes_3way, abstain) for one item. Pure function of the design."""
    if scenario == "known":
        m = M_BY_COHERENCE[item.coherence]
        if item.ground_truth:
            s = (0.050 + 0.001 * rank) if rank < m else (0.800 + 0.001 * rank)
        else:
            s = 0.300 + 0.001 * rank
        return s, s < ABSTAIN_THRESHOLD

    if scenario == "separable":
        s = 0.900 if item.ground_truth else 0.100
        return s, False

    if scenario == "ties":
        return 0.500, False

    if scenario == "coherence_driven":
        # Confidence tracks coherence and ignores truth: the hypothesis, simulated.
        s = 0.850 if item.coherence == "coherent" else 0.450
        return s, False

    if scenario == "truth_driven":
        # The well-calibrated null: confidence tracks truth and ignores coherence.
        s = 0.850 if item.ground_truth else 0.150
        return s, False

    raise MockScorerError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")


def _signature(prompt: str) -> str:
    """A prompt key that is invariant to the two surface controls, and to
    nothing else.

    The mock answers per ITEM, so it has to recognise the same item under a case
    permutation (--shuffle-cases) or an option rotation (--option-rotations).
    Sorting the lines absorbs the first; sorting the option list absorbs the
    second. Any other difference still misses, which is what keeps the mock's
    strictness useful - a genuine render.py change must still fail loudly.
    """
    lines = []
    for ln in prompt.split(chr(10)):
        if ln.startswith("Options:"):
            parts = sorted(x.strip() for x in ln[len("Options:"):].split("/"))
            ln = "Options: " + " / ".join(parts)
        lines.append(ln)
    return chr(10).join(sorted(lines))


class MockScorer:
    """Satisfies the same Protocol as HFScorer. Downstream cannot tell them apart.

    `prepare(items)` must be called before scoring: the mock's answer for an item
    depends on that item's rank within its (coherence, truth) group, which is a
    property of the whole set rather than of one prompt.
    """

    def __init__(
        self,
        scenario: str = "known",
        seed: int = 0,
        options: Sequence[str] = DEFAULT_OPTIONS,
    ) -> None:
        if scenario not in SCENARIOS:
            raise MockScorerError(
                f"unknown scenario {scenario!r}; expected one of {SCENARIOS}"
            )
        self.scenario = scenario
        self.seed = seed
        self.options = tuple(options)
        self._by_prompt: dict[str, ScoreResult] = {}
        self._by_item: dict[str, ScoreResult] = {}
        self._prepared = False

    # -- setup ---------------------------------------------------------------

    def prepare(self, items: Sequence[Item]) -> "MockScorer":
        groups: dict[tuple[str, bool], list[Item]] = {}
        for it in items:
            groups.setdefault((it.coherence, it.ground_truth), []).append(it)

        for group in groups.values():
            group.sort(key=lambda i: i.family_id)
            for rank, it in enumerate(group):
                s, abstain = _target_p_yes(it, rank, self.scenario)
                p_yes, p_no, p_unsure = _split_probs(s, abstain)
                top = max(
                    zip((p_yes, p_no, p_unsure), self.options), key=lambda t: t[0]
                )
                res = ScoreResult(
                    p_yes_raw=p_yes,
                    p_no_raw=p_no,
                    p_unsure_raw=p_unsure,
                    top_token=" " + top[1],
                    top_token_prob=top[0],
                )
                prompt = _signature(render_prompt(it, self.options))
                if prompt in self._by_prompt:
                    # Two items rendering to the same prompt means the 2x2 has
                    # collapsed: the model would be answering the same question
                    # and the cells would differ only in metadata. Never silent.
                    raise MockScorerError(
                        f"item {it.id} renders to a prompt already claimed by "
                        "another item. Two items with identical prompts cannot "
                        "be distinguished by any measurement; check that the "
                        "passages actually differ between cells."
                    )
                self._by_prompt[prompt] = res
                self._by_item[it.id] = res

        # Baselines: one flat value per claim, so subtracting the baseline in the
        # mock world is a clean constant shift and is easy to verify by hand.
        for it in items:
            self._by_prompt.setdefault(
                _signature(render_baseline_prompt(it.claim, self.options)),
                ScoreResult(
                    p_yes_raw=0.5 * MASS_COVERED,
                    p_no_raw=0.35 * MASS_COVERED,
                    p_unsure_raw=0.15 * MASS_COVERED,
                    top_token=" " + self.options[0],
                    top_token_prob=0.5 * MASS_COVERED,
                ),
            )

        self._prepared = True
        return self

    # -- Scorer protocol -----------------------------------------------------

    def score_prompt(self, prompt: str) -> ScoreResult:
        if not self._prepared:
            raise MockScorerError("MockScorer.prepare(items) was never called")
        try:
            return self._by_prompt[_signature(prompt)]
        except KeyError:
            raise MockScorerError(
                "prompt not in the prepared set - the mock is keyed on the exact "
                "rendered prompt, so this means render.py changed between "
                "prepare() and scoring, or an item is missing from prepare()."
            ) from None

    def score_prompts(self, prompts: Iterable[str]) -> list[ScoreResult]:
        return [self.score_prompt(p) for p in prompts]

    def for_item(self, item_id: str) -> ScoreResult:
        return self._by_item[item_id]

    def meta(self) -> dict[str, Any]:
        return {
            "scorer": "MockScorer",
            "model": f"mock:{self.scenario}",
            "revision": None,
            "device": "none",
            "dtype": "float64",
            "scenario": self.scenario,
            "mass_covered_by_construction": MASS_COVERED,
            "abstain_threshold": ABSTAIN_THRESHOLD,
            "expected_auc": EXPECTED_AUC if self.scenario == "known" else None,
            "temperature": 1.0,
            "determinism": {"seed": self.seed, "note": "mock is a pure function"},
            "prompt_template_hash": template_hash(self.options),
            "option_words": list(self.options),
            "options": None,
            "WARNING": "SYNTHETIC. These numbers are fixtures, not measurements.",
        }
