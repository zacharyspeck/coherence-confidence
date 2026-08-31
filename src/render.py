"""Prompt rendering. Single source of truth.

Nothing else in this repo is allowed to build a prompt string. If the template
changes, `TEMPLATE_HASH` changes, and every run JSON records that hash - so two
runs made under different templates can never be silently pooled (D-009).

The prompt is completion-style and ends in `Answer:` with no trailing space, so
the natural next token is ` Yes` / ` No` / ` Unsure`. Chat templating is opt-in
(`--chat-template`) precisely because applying one changes the token immediately
before the answer position, which changes the option logits in a model-specific
way.
"""

from __future__ import annotations

import hashlib

from .models import Item

# The three options as presented to the model. "Unsure" is a single-token
# stand-in for "Not enough evidence" (DECISIONS.md D-002). The instruction line
# below spells out the intended mapping.
OPTION_WORDS: tuple[str, str, str] = ("Yes", "No", "Unsure")

INSTRUCTION = (
    "Read the report, then answer the question.\n"
    "Answer Yes if the evidence establishes the claim, No if the evidence shows "
    "the claim is not established, or Unsure if the evidence is not enough to "
    "decide."
)

_EVIDENCE_TEMPLATE = """{instruction}

Claim: {claim}

Report:
{passage}

Question: Does this evidence establish this claim?
Options: Yes / No / Unsure
Answer:"""

_BASELINE_TEMPLATE = """{instruction}

Claim: {claim}

Report:
No supporting cases were provided.

Question: Does this evidence establish this claim?
Options: Yes / No / Unsure
Answer:"""


def render_prompt(item: Item) -> str:
    """The full evidence prompt for an item."""
    return _EVIDENCE_TEMPLATE.format(
        instruction=INSTRUCTION,
        claim=item.claim.strip(),
        passage=item.passage.strip(),
    )


def render_baseline_prompt(claim: str) -> str:
    """The same prompt with NO cases attached (src/baseline.py).

    Used to subtract out whatever the model already believed about the claim
    before any evidence was shown.
    """
    return _BASELINE_TEMPLATE.format(instruction=INSTRUCTION, claim=claim.strip())


def _template_hash() -> str:
    blob = "\x00".join(
        [INSTRUCTION, _EVIDENCE_TEMPLATE, _BASELINE_TEMPLATE, *OPTION_WORDS]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


TEMPLATE_HASH: str = _template_hash()
