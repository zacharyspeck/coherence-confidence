# PREDICTIONS

**Provenance, stated plainly.** No `PREDICTIONS.md` existed in this repository —
not in the working tree and not in any commit (`git log --all --diff-filter=A --
PREDICTIONS.md` returns nothing). This file was created on 2026-09-01 **before
any scoring run produced an AUC**, at the point where exactly one pilot item had
been scored and no analysis had been run. It does not invent a prediction: it
transcribes the one already committed to `DECISIONS.md` **D-025**, which is in
git history predating every result below, plus the hypothesis as restated in the
run instruction of the same evening. Nothing here is edited after results land.

## The prediction

From **D-025** (committed, verbatim):

> the analysis leads with **AUC(coherent) − AUC(diverse)** and prints the
> matched-mechanism version of the same quantity directly beneath it. […] a
> negative gap is the prediction, and AUC(coherent) below 0.5 is the crossover.

Stated as the endpoint:

1. **Primary.** AUC(coherent) < AUC(diverse). The gap
   `AUC(coherent) − AUC(diverse)` is predicted **negative**.
2. **Crossover.** AUC(coherent) < 0.5 would mean confidence is *inversely*
   related to truth when the evidence agrees with itself — the strong form.
3. **Matched subset.** The same sign is predicted on the `scope_mismatch`
   subset, where both false cells are falsified by the identical mechanism.
   This is the number to quote when challenged, because it removes the
   mechanism confound of D-024.
4. **Decorative control (D-030).** If the effect is about *evidential
   independence*, the decorative arm — same condition values as coherent, but
   surface variety like diverse — tracks **coherent**. If it is about how busy
   the passage looks to read, it tracks **diverse**.

## What would falsify it

- A **positive** gap: AUC(coherent) > AUC(diverse). A reversal.
- A gap whose family-clustered bootstrap CI **spans zero**: a null.
- The decorative arm tracking **diverse**, which would mean any coherence
  effect is confounded with surface complexity.

Any of these is a real result and gets reported as it came out.

## Diagnostics, explicitly NOT the endpoint

Mean confidence per cell, abstention rate, and the 2x2 breakdown are
diagnostics (D-025). A model can show a large coherence effect on mean
confidence while still ranking true above false perfectly well — that would
**not** be the effect this experiment is about. The confidence-accuracy
relationship is AUC.
