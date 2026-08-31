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

from .models import Item

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
    "enough to decide."
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


def _check(options: Sequence[str]) -> tuple[str, str, str]:
    if len(options) != 3:
        raise ValueError(f"expected exactly 3 option words, got {list(options)}")
    if len(set(options)) != 3:
        raise ValueError(f"option words must be distinct, got {list(options)}")
    return tuple(options)  # type: ignore[return-value]


def instruction(options: Sequence[str] = DEFAULT_OPTIONS) -> str:
    y, n, u = _check(options)
    return _INSTRUCTION_TEMPLATE.format(yes=y, no=n, unsure=u)


#: The instruction line under the default options, for tests and docs.
INSTRUCTION: str = instruction()


def option_line(options: Sequence[str] = DEFAULT_OPTIONS) -> str:
    return " / ".join(_check(options))


def render_prompt(item: Item, options: Sequence[str] = DEFAULT_OPTIONS) -> str:
    """The full evidence prompt for an item."""
    return _EVIDENCE_TEMPLATE.format(
        instruction=instruction(options),
        claim=item.claim.strip(),
        passage=item.passage.strip(),
        option_line=option_line(options),
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
