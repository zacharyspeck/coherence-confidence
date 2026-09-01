"""Where the falsifying text of an item actually sits, and how to compare it.

Every FALSE item is falsified by a specific span of the passage. Which span
depends on the mechanism, because the assembler puts them in different places:

    stated_confound     first sentence of the mid-passage line (the CONFOUND pair)
    scope_mismatch      second sentence of the mid-passage line (the SCOPE pair)
    broken_chronology   the case lines - the flaw is the dates, not a clause

Isolating that span is what lets the build check that two items are not falsified
by the same words, and lets a report quote the flaw without quoting the passage.
"""

from __future__ import annotations

import difflib
import re
from typing import Iterable

#: The 7-line layout from scripts/assemble_passages.py.
MID_LINE = 3
CASE_LINES = (1, 2, 4, 5)


def _sentences(line: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", line) if s.strip()]


def flaw_sentence(passage: str, flaw_mechanism: str | None) -> str:
    """The span of `passage` that carries the falsification."""
    lines = passage.split("\n")
    if flaw_mechanism == "broken_chronology":
        return " ".join(lines[i] for i in CASE_LINES if i < len(lines))
    if len(lines) <= MID_LINE:
        return passage
    parts = _sentences(lines[MID_LINE])
    # Key off the END of the line, not a fixed index. The mid-passage line has
    # two shapes - the confound is one sentence when it is live (`reach`) and
    # two when it is blocked, because the protection is stated first (see
    # src/midline.py). The SCOPE pair is always the final sentence either way,
    # so counting from the back is the only stable anchor. Indexing from the
    # front silently returned the confound sentence for every blocked item once
    # the second shape existed.
    if flaw_mechanism == "scope_mismatch":
        return parts[-1]
    if flaw_mechanism == "stated_confound":
        return " ".join(parts[:-1]) if len(parts) > 1 else lines[MID_LINE]
    return lines[MID_LINE]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()


def similarity(a: str, b: str) -> float:
    """1.0 is identical. The complement of normalized edit distance."""
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def ngrams(text: str, n: int = 8) -> set[str]:
    w = normalize(text).split()
    return {" ".join(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


def shared_ngrams(a: str, b: str, n: int = 8) -> set[str]:
    return ngrams(a, n) & ngrams(b, n)


def flaw_sentences(items: Iterable) -> dict[str, str]:
    """item id -> its falsifying span, for every FALSE item."""
    return {
        i.id: flaw_sentence(i.passage, i.flaw_mechanism)
        for i in items
        if not i.ground_truth
    }
