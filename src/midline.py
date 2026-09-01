"""The mid-passage line - the only truth-bearing line in a passage.

One function, imported by both `scripts/assemble_passages.py` and
`scripts/build_decorative.py`, which each used to carry their own copy. A layout
change that reached the 2x2 but not the control arm is exactly the kind of drift
that produces a clean-looking wrong number.

The line carries two independent clause pairs. Exactly one combination of each
falsifies:

    CONFOUND   changed/same  x  reach/block          -> changed_reach is live
    SCOPE      subset/whole  x  incomplete/complete  -> subset_incomplete is live

**Why `block` gets its own sentence, first.**

The old construction hung the blocking fact off the confound as a subordinate
clause:

    Rainfall in each plot's season ran a fifth higher than the year before, and
    every plot stood under cover, on a fixed watering schedule.

A reader meets the confound, forms the objection, and never re-reads far enough
to credit the clause that disarms it. In a 10-item human check that item was
called FALSE at difficulty 2.0 - confidently wrong - and the hunter audit has
four more like it among the 20 coherent_true items. A TRUE item that reads as
false depresses the confidence-accuracy relationship on the very cell the
hypothesis says is inflated, so it biases the headline *towards* the hypothesis.
That is the most dangerous direction an error can point (D-027, D-034).

So when the confound is blocked, the protection is stated first, positively, as
its own sentence, and the confound follows:

    Every plot stood under cover all season, watered only on a fixed schedule.
    Rainfall in each plot's season ran a fifth higher than the year before.

The reader is told the rain could not reach the plants before being told it
rained. `reach` keeps the single-sentence form, because there the confound is
live and nothing is being credited.

**This applies to every item carrying `block`, not only to coherent_true.** The
block clause appears twice on each side of the TRUE/FALSE split within a family
(D-034); rewriting it in one cell only would make the construction itself a
perfect predictor of TRUE for those items, which is precisely the lexical
giveaway the build gates against.
"""

from __future__ import annotations


def build_midline(spec: dict, clauses: dict, confound: tuple, scope: tuple) -> str:
    """Assemble the mid-passage line from a family's authored clause parts."""
    fact = spec[f"confound_fact_{confound[0]}"]
    pop = clauses[f"scope_population_{scope[0]}"]
    comp = clauses[f"scope_completeness_{scope[1]}"]

    if confound[1] == "block":
        protection = spec["confound_scope_block_sentence"]
        return f"{protection} {fact}. {pop}, {comp}."

    return f"{fact}, {spec[f'confound_scope_{confound[1]}']}. {pop}, {comp}."
