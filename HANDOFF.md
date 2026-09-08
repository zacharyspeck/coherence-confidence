# HANDOFF

Orientation for a new maintainer. The README reflects the current state of
the experiment (100 items, the control arm, the audits, the Kaggle results);
if anything here and there ever disagrees, `DECISIONS.md` is the record to
trust.

## What the experiment tests

Whether a language model's confidence tracks how much the evidence in its
context **agrees with itself** (coherence) rather than whether that evidence
**establishes the claim** (truth) — Koriat's consensuality principle, applied
to an LM. Every item is a short evidence report: four observed cases, a claim,
and the forced question *"Does this evidence establish this claim? Yes / No /
Unsure"*. The model's probability mass on Yes is read from the logits at the
final position — no generation, no sampling.

If the model is a good reasoner, confidence follows the truth dimension and is
roughly invariant to coherence. If it is running a consistency heuristic,
confidence rises when the four cases agree on every irrelevant surface
condition — even when a stated confound or a scope restriction in the passage
means they establish nothing.

## The four cells (and the control)

A 2x2 within-family design. 20 families; each holds the claim and scenario
fixed and varies only these two axes:

| | TRUE (claim established) | FALSE (claim not established) |
|---|---|---|
| **COHERENT** — all 4 cases share every irrelevant condition (one region, one month, one operator...) | `coherent_true` | `coherent_false` |
| **DIVERSE** — all 4 cases differ on every irrelevant condition | `diverse_true` | `diverse_false` |

Everything truth-bearing lives in one mid-passage line carrying two clause
pairs — CONFOUND (changed/same x reach/block) and SCOPE (subset/whole x
incomplete/complete). Exactly one combination of each falsifies
(`changed_reach`, `subset_incomplete`); every other combination is a defused
near-miss, so FALSE items differ from TRUE items by clause *combination*, not
by vocabulary. `src/validate.py` gates that a cross-validated lexical
classifier stays near chance.

There is also a **decorative control** (D-030): `decorative_true` /
`decorative_false`, 20 items. Same condition values as the coherent cells
(evidence exactly as dependent), but each case carries four varying
decorations — serial, clock time, terminal, desk. If confidence follows
surface busyness, decorative behaves like diverse; if it follows evidential
independence, it behaves like coherent. That is what separates "the model
reads agreement as evidence" from "the model is confused by clutter".

100 items total, all `review_status: "unreviewed"`; `items/final/` is empty by
design — promotion is a human act (D-008).

## Exact command sequence

Built on Windows / Git Bash, Python 3.13:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
PY=.venv/Scripts/python.exe            # on mac/linux: .venv/bin/python

# 0. plumbing check — no model, no real items
$PY -m src.smoke

# 1. validation gates (13; the build is red if any fail)
$PY -m src.validate --items items/draft items/seed

# 2. tokenization check FIRST for any new model (see README for why)
$PY -m src.score --model <HF_MODEL> --check-tokenization-only

# 3. score
$PY -m src.score --model <HF_MODEL> --items items/draft items/seed \
    --out results/run_<name>.json

# 4. baseline: every claim with NO evidence attached (prior subtraction)
$PY -m src.baseline --model <HF_MODEL> --items items/draft items/seed \
    --out results/baseline_<name>.json

# 5. analyze
$PY -m src.analyze --run results/run_<name>.json --out results/analysis_<name>.json

# tests
$PY -m pytest tests/ -q
```

If you edit any passage: `scripts/assemble_passages.py` then
`scripts/build_decorative.py` then `scripts/apply_complexity.py` rebuild the
items and clear the measured rates on exactly the changed ones; re-run the
reader audit for those (`scripts/reader_audit_patch.py` + the workflow in
`scripts/wf_reader_patch.js`), then `scripts/apply_reader_rates.py`, then the
gates. Edits without that chain leave stale measurements attached to new text.

## What to look at in the output

In `results/analysis_<name>.json` (every number carries a bootstrap 95% CI):

1. **The primary endpoint: AUC within coherent vs AUC within diverse** —
   computed separately, never pooled (pooling would let a coherence main
   effect masquerade as discrimination). Equal AUCs → confidence
   discriminates truth equally well in both regimes. AUC(coherent) below
   AUC(diverse) → agreement is degrading discrimination: the consensuality
   result.
2. **Mean `p_yes_3way` per cell**, especially `coherent_false` vs
   `diverse_false`. Confidence rising on false-but-coherent evidence is the
   effect itself.
3. **The decorative control's position.** Between coherent and diverse means
   the surface-complexity objection is live; sitting on the coherent cells
   kills it.
4. **Abstention per cell** — abstained items are INCLUDED in AUC at their
   `p_yes_3way`, never dropped (D-003), so check abstention isn't
   concentrating in one cell.
5. **The covariate models** — coherence effect before/after conditioning on
   `reader_catch_rate` (a measured per-item findability rate; see below) and
   surface stats. An effect that survives conditioning is not "the flaws were
   just harder to see".
6. `mass_covered` per item in the run file. Low coverage means the prompt
   template is wrong for that model and the renormalized numbers are garbage
   that looks publishable (D-009).

The only run so far is SmolLM2-135M as plumbing proof — it answers yes to
everything (all six cells within 0.03). Treat it as pipe-cleaning, not
evidence. The real experiment is steps 3–5 on models with actual
discrimination.

## How the items were audited

Two instruments, deliberately different (D-032):

- **Hunter audit** (`results/audit/`): auditors told a flaw may exist, asked
  to name it. Per-auditor catch 94/100 on FALSE items, finds matched
  deterministically to the flaw span. Ceiling-ish by construction.
- **Reader audit** (`results/reader_audit.json`): readers see the exact
  scored prompt, no hint a flaw might exist, three independent reads per
  item. Yields `reader_catch_rate` (FALSE) / `reader_false_positive_rate`
  (TRUE) carried on each item — the analysis covariate. Cluster caveat: one
  agent answers a whole batch, so 300 reads ≈ 18 readers; every cell figure
  in `READER_REPORT.md` carries a cluster-bootstrap CI (D-037).

Current state: every FALSE item at reader catch ≥ 0.5, the three FALSE cell
means within 0.10 of each other. `READER_REPORT.md` and `CONTROL_REPORT.md`
regenerate from artifacts via `scripts/build_reader_report.py` /
`scripts/build_control_report.py`.

## Known open issues

1. **`coherent_true` reader false-positive rate ~0.25** (diverse_true ~0.03).
   Not mislabeled items: a careful human blind review accepted all of them,
   and hunter objections all name the same conservative clause
   (`whole_incomplete` — figures cover everyone including partial compliers,
   which *dilutes toward the null*). The rate comes from a minority of
   reader-instances who reject the coherent-TRUE gestalt wholesale, and it
   cannot be worded away — the softening attempt failed the gates as a
   TRUE-side giveaway (D-036). Since the scored model is drawn from the same
   population as the audit readers, treat the coherent/diverse TRUE asymmetry
   as a measured property to report alongside the endpoint, not as noise.
   Disposition options are in D-036; D-038 has the evidence.
2. **No naive human reader has seen these items.** Both human reviews were by
   the same person; before the second they had read the reports, so they knew
   the design's shape (where the load-bearing line sits, how the clause
   grammar works). Human catch rates from review #2 (10/10) are from a
   briefed expert, not a cold reader.
3. **Item review is one person, twice.** All 100 items are `unreviewed`;
   nothing is promoted to `items/final/`. The blind sets
   (`review/blind_items.md`, `review/blind_items_2.md`, keys under
   `results/review_key/` — answer before opening a key) cover 20 of 100
   items, so 80 have never been read by any human.

Every design call, bug, and deviation is logged in `DECISIONS.md`
(D-001…D-050), in order, with the reasoning. When something in the numbers
looks odd, check there first — the odds are good it is already written down.
