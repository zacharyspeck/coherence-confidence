# coherence-confidence

**Research question:** Does a language model's confidence track how much the
evidence in its context *agrees with itself* (coherence) rather than whether
that evidence actually *establishes the claim* (truth)? Koriat's consensuality
principle, applied to an LM: if confidence is a consistency heuristic, the
confidence-accuracy relationship should *degrade* when the evidence is
coherent.

**Result, in one line:** the opposite. At 32B the model tells sound evidence
from unsound clearly better when the evidence all agrees with itself than when
it is diverse — the gap AUC(coherent) − AUC(diverse) is **+0.225
[+0.025, +0.435]**, the opposite sign to the pre-registered prediction, and
the interval excludes zero. In the diverse condition the AUC point estimate
sits below 0.5 at all three sizes, but the 32B interval ([0.27, 0.52]) spans
0.5, so the licensed claim there is *no separation* between true and false,
not a reversal.

## The headline numbers

| model | AUC(coherent) | AUC(diverse) | gap (coh − div) | 95% CI (family) |
|---|---|---|---|---|
| Qwen2.5-3B (CPU, old instrument) | 0.5550 | 0.4475 | +0.1075 | [−0.0375, +0.2475] |
| Qwen3-8B (fp16) | 0.5275 | 0.4550 | +0.0725 | [−0.0375, +0.2075] |
| **Qwen3-32B (4-bit)** | **0.6175** | **0.3925** | **+0.2250** | **[+0.0250, +0.4350]** |
| Qwen3-32B, matched `scope_mismatch` subset | 0.7600 | 0.4100 | +0.3500 | [+0.0988, +0.6735] |

The matched-subset row is the number to quote when challenged: both conditions
are falsified by the identical mechanism, so only coherence differs (D-024).
The 3B row was measured under the old instrument (`Unsure` option, user-turn
cue, coverage 0.49) and is directional context, not a same-ruler comparison.
Full tables, controls, and caveats: `RESULTS.md`.

## Pre-registration

The predicted direction (coherence should degrade the confidence-accuracy
relationship) is in this repo's first commit (6ba9c9b, 2026-08-30). It was
locked as the endpoint AUC(coherent) − AUC(diverse) < 0 in `DECISIONS.md`
D-025 (commit 83a9c47, 2026-08-31 09:49) before any run on a Qwen model; the
only earlier scoring was a SmolLM2-135M plumbing check that morning
(`results/analysis_base135m_realitems.md`). `PREDICTIONS.md` transcribes D-025
and was committed together with the first Qwen2.5-3B results (7d182c0,
2026-09-02); it has not been edited since.

## Design

A 2x2 within-family design plus a control arm: **100 items = 20 families x 4
cells + 20 decorative-control items**. Each item is a short passage containing
4 observed cases, a claim, and the forced question:

> Does this evidence establish this claim? Yes / No / Unknown

(The third option is `Unknown` rather than `Unsure` — see D-019 for why: the
option word must be a single token in its presented form on the scored model's
vocabulary, and bare `Unsure` is not on Qwen's.)

| | TRUE (claim established) | FALSE (claim not established) |
|---|---|---|
| **COHERENT** — all 4 cases share every irrelevant condition (one region, one month, one operator…) | `coherent_true` | `coherent_false` |
| **DIVERSE** — all 4 cases differ on every irrelevant condition | `diverse_true` | `diverse_false` |

Every family holds the claim and scenario fixed and varies only coherence and
truth, so the 2x2 contrast is within-family. Everything truth-bearing lives in
one mid-passage line; FALSE items differ from TRUE items by clause
*combination*, not vocabulary, and a validation gate keeps a cross-validated
lexical classifier at chance.

The **decorative control** (D-030): `decorative_true` / `decorative_false`,
20 items with the same condition values as the coherent cells but four varying
decorations per case. If confidence follows surface busyness, decorative
behaves like diverse; if it follows evidential independence, it behaves like
coherent. On the fixed instrument at both 8B and 32B it tracks coherent.

`src/validate.py` enforces **13 gates** (word-count balance, condition-value
counts, the lexical classifier, cell balance, duplicate passages, answer-key
leakage, closer word balance, and more — D-015, D-020, D-022, D-023). Items
carry `review_status: "unreviewed"` as a build-time field; the validation
story is the gates plus `READER_REPORT.md` and `FIX_REPORT.md` (human review),
and `items/final/` is unused.

## Measurement

`src/score.py` renders the prompt ending in `Answer:` and reads the **logits
at the final position** — no generation, no sampling. It extracts probability
mass for the three options and reports:

- `p_yes_3way = p_yes / (p_yes + p_no + p_unknown)` — the primary measure
- `p_yes_2way = p_yes / (p_yes + p_no)` — reported separately
- `abstained = argmax(p_yes, p_no, p_unknown) == unknown`

Things that are easy to get wrong, all handled by gates rather than by care:

1. **Tokenization.** `"Yes"`, `" Yes"`, `"yes"`, `" YES"` are all different
   token ids; mass is summed over the full casing x leading-space variant set
   per option. `tests/test_tokenization.py` asserts the variant sets are
   non-empty for the tokenizer in use.
2. **Multi-token options.** The third option must be a single token in its
   presented form; `--require-canonical-single-token` is ON by default and
   fails the run rather than silently reading abstention as near-zero;
   `--third-option <Word>` swaps the word (D-019). Check first for any new
   model: `python -m src.score --model <NAME> --check-tokenization-only`.
3. **Silent low coverage.** `mass_covered = p_yes + p_no + p_unknown` is
   reported per item and the run fails below `--min-mass-covered`. This
   catches an instruct model given an untemplated prompt (D-009) and a
   hybrid-thinking model spending its next token on `<think>` (D-046).
4. **The chat-template answer cue is assistant prefill** (D-048), and it
   self-adapts on first use when the model's house style (e.g. markdown bold)
   would otherwise absorb the answer mass (D-049). The discovered prefill is
   recorded in run meta.

## Analysis

`src/analyze.py` reports, with bootstrap 95% CIs (10,000 resamples, both
item-stratified and family-clustered) on every number:

- **AUC within coherent and AUC within diverse, separately — never pooled**
  (pooling would let a coherence main effect masquerade as discrimination).
  The gap is a first-class statistic with its own family-clustered CI.
- the matched-mechanism subset, the decorative-control verdict, a
  per-mechanism breakdown, and covariate models (does the coherence effect
  survive conditioning on `reader_catch_rate` and surface complexity?)
- mean `p_yes_3way` / `p_yes_2way` and abstention rate per cell, and a 2x2
  ANOVA-style breakdown — diagnostics, explicitly not endpoints (D-025)

AUC is an **explicit pairwise win rate** (ties = 0.5), asserted against
`sklearn.metrics.roc_auc_score` to 1e-9 in `tests/test_auc.py`.

**Abstention policy (explicit, and load-bearing):** abstained items are
**INCLUDED** in the AUC using their `p_yes_3way`, never dropped. Dropping
abstentions would let a model inflate its AUC by abstaining on exactly the
items it finds hard (D-003).

## Reader audit provenance

The three blind reader passes in `results/reader_audit/` were answered by
language-model readers — subagent instances of the coding agent that built
this repo; the audit artifacts do not record a specific model name (D-032,
D-037) — shown only the scored prompt (third option `Unsure` at audit time;
renamed `Unknown` on Kaggle for tokenization, D-019). The 1-5 salience
ratings were made by the item author.

## What this does NOT establish

- **One model family.** Qwen only — 3B (previous generation, old instrument),
  8B, 32B. No cross-family replication; "LLMs do X" is not a claim these data
  can carry.
- **The 32B is 4-bit quantized** (nf4, fp16 compute) while the 8B is fp16;
  the size comparison is not precision-matched, and the "grows with scale"
  reading rests on exactly two points on the fixed instrument.
- **`reader_catch_rate` comes from model readers, not humans** (n=18–27
  reader-instances, cluster-bootstrapped); the two human blind reviews are by
  the same person.
- **Absolute discrimination is weak everywhere.** The best cell anywhere is
  0.76; most are within noise of chance. The result is about the *asymmetry*
  between conditions, not about competence.

The full list, including the `coherent_true` reader false-positive rate and
option-position bias, is `RESULTS.md` section 9.

## Reading order

`README.md` → `RESULTS.md` → `PREDICTIONS.md` → `DECISIONS.md` →
`READER_REPORT.md` / `FIX_REPORT.md` / `CONTROL_REPORT.md` → `HANDOFF.md`

## Reproduce

```bash
# Windows (this repo was built on Windows / Git Bash); mac/linux: .venv/bin/python
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# tests
.venv/Scripts/python.exe -m pytest tests/ -q

# are the stored Kaggle runs still valid against the items on disk?
.venv/Scripts/python.exe scripts/verify_run.py results/kaggle/run_qwen3_32b.json
.venv/Scripts/python.exe scripts/verify_run.py results/kaggle/run_qwen3_8b.json

# regenerate the headline analysis from the stored run
.venv/Scripts/python.exe -m src.analyze --run results/kaggle/run_qwen3_32b.json \
    --baseline results/kaggle/baseline_qwen3_32b.json --out results/kaggle/analysis_qwen3_32b.json
```

To re-run the measurement itself: `kaggle_run.ipynb`, one **Run all** on a
Kaggle 2x T4 session, ~2 h — `KAGGLE.md` has the full recipe.

Numbers in the write-up are from commit 1370a81; the linked snapshot is tag
v1.0-blog.
