"""Surface complexity: how much there is to parse, independent of what it means.

The co-founder's objection, in its last remaining form: a diverse passage names
four countries, four devices and four months where a coherent one names one of
each. More entities to track. A confidence difference could come from parse load
rather than from the evidence being evidentially independent.

Two things address that. The `decorative_*` cells (D-030) match diverse surface
busyness while holding the evidential structure coherent. And these numbers,
which are carried on every item and reported as a covariate, so the question can
be answered by conditioning rather than by argument.

Every metric here is a pure function of the passage text, so it can be recomputed
and checked - nothing can drift the way a hand-maintained annotation would.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

#: A word: letters, digits, apostrophes, internal hyphens.
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

#: Tokens that look like a named entity or an identifier rather than prose.
#: The heuristic is deliberately simple so a reader can check it by eye:
#:   - anything containing a digit (serial numbers, times, dates, ticket ids)
#:   - anything capitalised that is NOT starting a line or following a full stop
#: It will miss lowercase proper nouns and over-count the odd title-cased term.
#: Both failure modes hit every cell equally, which is all the covariate needs.
_SENTENCE_END = (".", "!", "?", ":", ";")


def _line_tokens(line: str) -> list[tuple[str, bool]]:
    """(token, is_sentence_initial) for one line."""
    out: list[tuple[str, bool]] = []
    initial = True
    for m in _TOKEN.finditer(line):
        out.append((m.group(0), initial))
        tail = line[m.end() : m.end() + 1]
        initial = tail in _SENTENCE_END
    return out


def entity_tokens(passage: str) -> list[str]:
    """Entity-like tokens, in order, duplicates kept."""
    found: list[str] = []
    for line in passage.split("\n"):
        for tok, initial in _line_tokens(line):
            if any(ch.isdigit() for ch in tok):
                found.append(tok)
            elif tok[0].isupper() and not initial:
                found.append(tok)
    return found


def word_tokens(passage: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(passage)]


def surface_complexity(
    passage: str,
    conditions: Sequence[dict[str, str]] = (),
    decorations: Sequence[dict[str, str]] = (),
) -> dict[str, Any]:
    """The covariate, as a plain dict.

    `n_distinct_entities` and `type_token_ratio` are what the objection is about -
    they measure how busy the surface is. `n_distinct_condition_values` is the
    evidential variety, which is what the coherent/diverse manipulation actually
    changes. A decorative item is built to be high on the first two and low on
    the third; that separation is the whole control.
    """
    words = word_tokens(passage)
    ents = entity_tokens(passage)
    distinct_words = set(words)

    cond_values = {v for d in conditions for v in d.values()}
    dec_values = {v for d in decorations for v in d.values()}

    return {
        "n_tokens": len(words),
        "n_distinct_tokens": len(distinct_words),
        "type_token_ratio": round(len(distinct_words) / len(words), 5) if words else 0.0,
        "n_entity_tokens": len(ents),
        "n_distinct_entities": len({e.lower() for e in ents}),
        "n_distinct_condition_values": len(cond_values),
        "n_distinct_decoration_values": len(dec_values),
    }


def for_item(item) -> dict[str, Any]:
    """Compute the covariate for an Item (avoids a circular import)."""
    return surface_complexity(
        item.passage,
        [c.conditions for c in item.cases],
        [c.decorations for c in item.cases],
    )


METRIC_KEYS: tuple[str, ...] = (
    "n_tokens",
    "n_distinct_tokens",
    "type_token_ratio",
    "n_entity_tokens",
    "n_distinct_entities",
    "n_distinct_condition_values",
    "n_distinct_decoration_values",
)

#: The metrics the decorative cells must match the diverse cells on. These are
#: the ones the objection is about; condition values are deliberately NOT here,
#: because decorative items are supposed to be low on those.
SURFACE_MATCH_KEYS: tuple[str, ...] = ("n_distinct_tokens", "n_distinct_entities")


def mean_of(items: Iterable, key: str) -> float:
    vals = [for_item(i)[key] for i in items]
    return sum(vals) / len(vals) if vals else 0.0
