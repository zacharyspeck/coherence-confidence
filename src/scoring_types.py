"""The shape of a single measurement. Shared by the real scorer and the mock so
the pipeline downstream cannot tell them apart (that is the point of step 6).

Options are addressed by ROLE (`yes`/`no`/`unsure`), never by surface string, so
that swapping the third option word (D-019) changes nothing downstream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .render import ROLE_NO, ROLE_UNSURE, ROLE_YES


@dataclass(frozen=True)
class ScoreResult:
    """Probability mass at the answer position, for one prompt.

    `*_raw` are absolute probabilities out of the full vocabulary softmax, so
    `mass_covered = p_yes_raw + p_no_raw + p_unsure_raw` says how much of the
    model's next-token distribution the three options actually account for. A low
    value means the model wanted to say something else entirely and the
    renormalized numbers are built on sand - which is why it is always reported.
    """

    p_yes_raw: float
    p_no_raw: float
    p_unsure_raw: float
    top_token: str
    top_token_prob: float

    def __post_init__(self) -> None:
        for name in ("p_yes_raw", "p_no_raw", "p_unsure_raw"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} is not a probability")
        if self.mass_covered <= 0.0:
            raise ValueError(
                "all three option probabilities are zero - the option token ids "
                "are almost certainly wrong for this tokenizer"
            )

    # ---- derived -----------------------------------------------------------

    @property
    def mass_covered(self) -> float:
        return self.p_yes_raw + self.p_no_raw + self.p_unsure_raw

    @property
    def p_yes_3way(self) -> float:
        """THE primary measure."""
        return self.p_yes_raw / self.mass_covered

    @property
    def p_no_3way(self) -> float:
        return self.p_no_raw / self.mass_covered

    @property
    def p_unsure_3way(self) -> float:
        return self.p_unsure_raw / self.mass_covered

    @property
    def p_yes_2way(self) -> float:
        """Yes vs No only, ignoring Unsure. Reported separately, never primary."""
        denom = self.p_yes_raw + self.p_no_raw
        if denom <= 0.0:
            return float("nan")
        return self.p_yes_raw / denom

    @property
    def argmax_role(self) -> str:
        pairs = (
            (self.p_yes_raw, ROLE_YES),
            (self.p_no_raw, ROLE_NO),
            (self.p_unsure_raw, ROLE_UNSURE),
        )
        return max(pairs, key=lambda t: t[0])[1]

    @property
    def abstained(self) -> bool:
        """The third option has the highest probability of the three."""
        return self.argmax_role == ROLE_UNSURE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            mass_covered=self.mass_covered,
            p_yes_3way=self.p_yes_3way,
            p_no_3way=self.p_no_3way,
            p_unsure_3way=self.p_unsure_3way,
            p_yes_2way=self.p_yes_2way,
            argmax_role=self.argmax_role,
            abstained=self.abstained,
        )
        return d
