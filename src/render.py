"""Prompt rendering. Single source of truth.

Nothing else in this repo is allowed to build a prompt string. If the template or
the option words change, `template_hash()` changes, and every run JSON records
that hash - so two runs made under different templates can never be silently
pooled (D-009).

The prompt is completion-style and ends in `Answer:` with no trailing space, so
the natural next token is ` Yes` / ` No` / ` Unsure`. Chat templating is opt-in
(`--chat-template`) precisely because applying one changes the token immediately
before the answer position, which changes the option logits model-specifically.

The three options are addressed internally by ROLE (`yes`/`no`/`unsure`), not by
their surface strings, because the surface string of the third option is
configurable: "Unsure" is the default per the brief (D-002) but it is not a
single token on every vocabulary, and when it is not, the option word has to
change or the readout is invalid (D-019).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from .models import Item, normalize_ws

#: Canonical role names. Fixed forever; only the surface words vary.
ROLES: tuple[str, str, str] = ("yes", "no", "unsure")
ROLE_YES, ROLE_NO, ROLE_UNSURE = ROLES

#: Default surface words. The third is a single-token stand-in for "Not enough
#: evidence" (DECISIONS.md D-002); the instruction line spells out the mapping.
DEFAULT_OPTIONS: tuple[str, str, str] = ("Yes", "No", "Unsure")

#: Back-compat alias for the default option words.
OPTION_WORDS: tuple[str, str, str] = DEFAULT_OPTIONS

_INSTRUCTION_TEMPLATE = (
    "Read the report, then answer the question.\n"
    "Answer {yes} if the evidence establishes the claim, {no} if the evidence "
    "shows the claim is not established, or {unsure} if the evidence is not "
    "enough to decide.\n"
    # Chat-tuned models decorate their answers ("**Yes**"); at the answer
    # position that puts most of the next-token mass on formatting tokens
    # instead of the options (D-049). The sentence lowers that; the prefill
    # auto-discovery in score.py absorbs whatever habit remains.
    "Reply with one word only. No formatting, no markdown, no punctuation."
)

_EVIDENCE_TEMPLATE = """{instruction}

Claim: {claim}

Report:
{passage}

Question: Does this evidence establish this claim?
Options: {option_line}
Answer:"""

_BASELINE_TEMPLATE = """{instruction}

Claim: {claim}

Report:
No supporting cases were provided.

Question: Does this evidence establish this claim?
Options: {option_line}
Answer:"""


#: The three rotations of the option list. Position 0 is the first option shown.
#: Roles are unchanged - only the ORDER the reader sees changes - so a model that
#: is answering the question rather than picking a position should be unaffected.
OPTION_ROTATIONS: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def rotate(seq: Sequence[str], rotation: Sequence[int]) -> list[str]:
    return [seq[i] for i in rotation]


def _check(options: Sequence[str]) -> tuple[str, str, str]:
    if len(options) != 3:
        raise ValueError(f"expected exactly 3 option words, got {list(options)}")
    if len(set(options)) != 3:
        raise ValueError(f"option words must be distinct, got {list(options)}")
    return tuple(options)  # type: ignore[return-value]


def instruction(options: Sequence[str] = DEFAULT_OPTIONS) -> str:
    y, n, u = _check(options)
    return _INSTRUCTION_TEMPLATE.format(yes=y, no=n, unsure=u)


def case_lines(item: Item) -> list[int]:
    """Indices of the passage lines that are case sentences.

    Found by matching the stored case text rather than by position, so this
    survives a change to the passage layout. The Item model guarantees every case
    appears verbatim, so a miss here is a real inconsistency, not a parse failure.
    """
    lines = item.passage.split("\n")
    norm = [normalize_ws(ln) for ln in lines]
    out = []
    for c in item.cases:
        target = normalize_ws(c.text)
        try:
            out.append(norm.index(target))
        except ValueError:
            raise ValueError(
                f"item {item.id}: case {c.case_id} is not a whole passage line, so "
                "case order cannot be permuted safely"
            ) from None
    return out


def passage_with_case_order(item: Item, order: Sequence[int]) -> str:
    """The passage with its four case sentences permuted, everything else fixed.

    Case order is an irrelevant surface feature. If a model's answer moves when
    only the order changes, the measurement is picking up presentation rather
    than evidence, and `--shuffle-cases` is how that gets checked.
    """
    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError(f"case order must be a permutation of 0..3, got {list(order)}")
    lines = item.passage.split("\n")
    slots = case_lines(item)
    texts = [lines[i] for i in slots]
    for slot, src in zip(slots, order):
        lines[slot] = texts[src]
    return "\n".join(lines)


def case_permutation(item: Item, seed: int) -> list[int]:
    """A fixed, per-item permutation. Deterministic given (item id, seed), so a
    run can be reproduced and the permutation is recorded either way."""
    import hashlib
    import random

    h = hashlib.sha256(f"{seed}:{item.id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    order = [0, 1, 2, 3]
    rng.shuffle(order)
    return order


#: The instruction line under the default options, for tests and docs.
INSTRUCTION: str = instruction()


def option_line(options: Sequence[str] = DEFAULT_OPTIONS) -> str:
    return " / ".join(_check(options))


def render_prompt(
    item: Item,
    options: Sequence[str] = DEFAULT_OPTIONS,
    *,
    case_order: Sequence[int] | None = None,
    option_rotation: Sequence[int] | None = None,
) -> str:
    """The full evidence prompt for an item.

    `case_order` permutes the four case sentences; `option_rotation` permutes the
    ORDER the three options are displayed in, leaving their roles alone. Both are
    controls for surface features that should not matter (fix-pass step 6).
    """
    _check(options)
    shown = rotate(options, option_rotation) if option_rotation else list(options)
    passage = (
        passage_with_case_order(item, case_order) if case_order else item.passage
    )
    return _EVIDENCE_TEMPLATE.format(
        instruction=instruction(options),
        claim=item.claim.strip(),
        passage=passage.strip(),
        option_line=" / ".join(shown),
    )


def render_baseline_prompt(
    claim: str, options: Sequence[str] = DEFAULT_OPTIONS
) -> str:
    """The same prompt with NO cases attached (src/baseline.py).

    Used to subtract out whatever the model already believed about the claim
    before any evidence was shown.
    """
    return _BASELINE_TEMPLATE.format(
        instruction=instruction(options),
        claim=claim.strip(),
        option_line=option_line(options),
    )


def canonical_forms(options: Sequence[str] = DEFAULT_OPTIONS) -> dict[str, str]:
    """The exact string the model must emit next for each role, given that the
    prompt ends in `Answer:` with no trailing space.

    This is what has to be a single token for the three-way readout to be valid.
    """
    y, n, u = _check(options)
    return {ROLE_YES: " " + y, ROLE_NO: " " + n, ROLE_UNSURE: " " + u}


def template_hash(options: Sequence[str] = DEFAULT_OPTIONS) -> str:
    blob = "\x00".join(
        [
            _INSTRUCTION_TEMPLATE,
            _EVIDENCE_TEMPLATE,
            _BASELINE_TEMPLATE,
            *_check(options),
        ]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


#: Hash under the default options.
TEMPLATE_HASH: str = template_hash()
