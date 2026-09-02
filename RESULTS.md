# RESULTS — Qwen2.5-3B-Instruct, 100 items, CPU

Run 2026-09-02. Model `Qwen/Qwen2.5-3B-Instruct`, revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, CPU, bfloat16, chat template on,
options `Yes / No / Unsure` read from the final-position logits. All 100 items,
20 families, six cells. Artifacts: `results/run_qwen3b.json`,
`results/baseline_qwen3b.json`, `results/analysis_qwen3b.{json,md}`.

## The headline

**The model's ability to tell sound evidence from unsound did not collapse when
the evidence agreed with itself — if anything it was better there, and the
difference is not statistically distinguishable from zero.**

    AUC(coherent) = 0.5550        AUC(diverse) = 0.4475
    gap = +0.1075                 95% CI (family-clustered) [-0.0375, +0.2475]

The prediction was a **negative** gap. The measured gap is **positive** and its
interval **spans zero**.

## Does it match PREDICTIONS.md?

**No.** Flatly: the primary endpoint came out with the sign opposite to the
pre-registered prediction, and the confidence interval includes zero. This is a
null result on the primary endpoint, with a point estimate pointing the wrong
way for the consensuality hypothesis. It is not weak support and it is not a
trend in the predicted direction.

One prediction *was* met, in the mirror image of how it was meant. The
pre-registration named `AUC(coherent) < 0.5` as "the crossover" — confidence
running backwards against truth. A crossover did occur, but in the other
condition: **AUC(diverse) = 0.4475**, below 0.5. Where the four cases differ on
every surface condition, this model's confidence is *inversely* related to
whether the evidence establishes the claim.

## Every number, with its interval

### Primary endpoint (full set, 400 pairs per condition)

| condition | AUC | 95% CI (family-clustered) |
|---|---|---|
| coherent | 0.5550 | [0.4475, 0.6675] |
| diverse | 0.4475 | [0.3525, 0.5400] |
| **gap (coherent − diverse)** | **+0.1075** | **[−0.0375, +0.2475]** |

### Matched-mechanism subset — the number to quote when challenged

Both conditions falsified by `scope_mismatch`, so flaw mechanism cannot differ
between them and only coherence does (D-024). 100 pairs per condition.

| condition | AUC | 95% CI (family-clustered) |
|---|---|---|
| coherent | 0.6900 | [0.5000, 0.8889] |
| diverse | 0.5200 | [0.3438, 0.7200] |
| **gap** | **+0.1700** | **[−0.1562, +0.4531]** |

Same story, larger and less certain: opposite sign to the prediction, interval
spans zero.

### Per-mechanism

| mechanism | condition | AUC | 95% CI (family) |
|---|---|---|---|
| `stated_confound` | coherent | 0.4800 | [0.2900, 0.6800] |
| `broken_chronology` | diverse | 0.4300 | [0.2400, 0.5918] |
| `scope_mismatch` | coherent | 0.6900 | [0.5000, 0.8889] |
| `scope_mismatch` | diverse | 0.5200 | [0.3438, 0.7200] |

The model is at or below chance on two of the three mechanisms. Only
`scope_mismatch` in the coherent condition is meaningfully above 0.5, and its
interval touches it.

### Covariate: does the coherence effect survive conditioning?

Logistic models of catch-rate over the 40 core-cell FALSE items (overall catch
rate 62.5%):

| model | coherence coefficient |
|---|---|
| `coherence_only` | +0.647 |
| `plus_reader_catch_rate` | +0.690 |
| `plus_reader_catch_rate_and_surface` | **−0.266** |

**It does not survive.** Adding the measured per-item findability rate leaves it
alone; adding surface complexity **inverts** it. Surface complexity was doing
the work.

### Diagnostics (not endpoints)

| cell | n | mean P(yes) | 95% CI (item) | abstention |
|---|---|---|---|---|
| `coherent_true` | 20 | 0.4977 | [0.3513, 0.6392] | 0.000 |
| `coherent_false` | 20 | 0.4392 | [0.3160, 0.5659] | 0.000 |
| `diverse_true` | 20 | 0.3461 | [0.2219, 0.4820] | 0.000 |
| `diverse_false` | 20 | 0.4169 | [0.2683, 0.5758] | 0.000 |
| `decorative_true` | 10 | 0.7459 | [0.5670, 0.8847] | 0.000 |
| `decorative_false` | 10 | 0.6350 | [0.4083, 0.8425] | 0.000 |

2x2 effects on mean confidence, family-clustered CIs: coherence
−0.0161 [−0.1475, +0.1223]; truth +0.0172 [−0.0588, +0.0867]; interaction
+0.1293 [−0.0149, +0.2631]. The clean within-TRUE coherence contrast (D-004) is
+0.1516 [−0.0482, +0.3426]. Every one of them spans zero.

**Abstention was exactly zero in all six cells.** The model never picked Unsure
as the argmax. See the caveat on abstention measurement below — this number is
partly an artifact.

**Baseline (claims with no evidence attached):** mean P(yes) = **0.0015**. With
no cases in front of it the model essentially never says yes, so the evidence
runs are movement from a floor of ~0, not endorsement of already-plausible
claims. Baseline coverage was low (0.06) and that number is correspondingly
soft.

## What the decorative arm says

This is the control built to separate "the evidence agrees with itself" from
"the passage is busy to read" (D-030). A decorative item has the **same
condition values** as a coherent one — so the evidence is exactly as dependent —
but carries the **entity count of a diverse one** (25.2 distinct entities against
diverse's 24.7 and coherent's 13.2).

| level | AUC | 95% CI (family) | distinct entities | condition values |
|---|---|---|---|---|
| coherent | 0.6900 | [0.5000, 0.8889] | 13.2 | 4.0 |
| diverse | 0.5200 | [0.3438, 0.7200] | 24.7 | 16.0 |
| **decorative** | **0.5800** | [0.4688, 0.7408] | 25.2 | 4.0 |

The pre-committed sentence, generated by `analyze.py` from the numbers rather
than written afterwards:

> **The decorative control tracks the DIVERSE cells (AUC 0.580 against 0.520
> diverse and 0.690 coherent), so the effect is about surface complexity and the
> evidential-independence story is wrong.**

decorative − coherent = −0.1100 [−0.3200, +0.1251]; decorative − diverse =
+0.0600 [−0.1406, +0.3600]. Both intervals span zero, so the control is
suggestive rather than decisive — but it points the same way as the covariate
model, which independently inverted the coherence coefficient when surface
complexity entered. Two different instruments, same conclusion: whatever
separates the conditions here travels with parse load, not with evidential
independence.

## What this run does NOT establish

1. **One model, one size.** Qwen2.5-3B-Instruct only. No size ladder. A 3B may
   simply lack the capability the hypothesis is about; the honest read is
   "this model does not show the effect", not "the effect does not exist".
2. **The model is barely above chance at the task at all.** AUC 0.5550 and
   0.4475 are close to coin-flipping. A confidence-accuracy *relationship* is
   hard to degrade when it is nearly absent to begin with. This is the single
   biggest reason to run a larger model before concluding anything.
3. **CPU, memory-starved, non-ideal conditions.** Run at ~5 GB available RAM
   against a model that does not fit resident (D-042). This affects wall-clock
   only, not the arithmetic — logits are deterministic — but it is why the run
   took all night in ~15-item fragments.
4. **Abstention is undercounted, so the three-way split is distorted.** Mean
   `mass_covered` was 0.486: about half the next-token mass sits outside
   {Yes, No, Unsure}. On this tokenizer bare `Unsure` is two tokens while
   ` Unsure` is one, and the model's preferred continuation after the chat
   template is the unspaced form, whose mass is not counted (D-043). The
   reported zero abstention across all six cells is partly this artifact.
   `p_yes_3way` is a renormalisation over captured mass; the AUC ranking is
   valid only insofar as that capture is unbiased across cells — plausible,
   unverified.
5. **Readers are models, not people.** `reader_catch_rate`, used as the
   covariate, comes from model readers. Two human blind reviews exist, both by
   the same person, the second after reading the reports (D-038).
6. **`coherent_true` carries a known 0.25 reader false-positive rate** (D-036).
   A quarter of model readers call those items false. They are not mislabeled —
   a careful human accepted all of them — but items that read as false in the
   TRUE cell depress AUC(coherent) specifically, which biases *toward* the
   consensuality prediction. The result came out against the prediction anyway,
   with that handicap in place.
7. **Controls incomplete at time of writing.** See the section below.

## Order and rotation controls

`--shuffle-cases` (case order permuted per item; case order carries no evidence,
so a result that moves under it is measuring presentation) and
`--option-rotations` (all three orderings of Yes/No/Unsure, measuring option
position bias) were launched after the primary analysis was secured. Their
status is recorded in `DECISIONS.md`; where a control did not finish, that is
stated rather than papered over. **The headline above does not depend on them** —
they test whether it moves, and an unfinished control is an open question, not a
silent pass.

## Reproducing this on a bigger model

The whole point of the caveats above is that this wants a larger model. On a
GPU box every one of these is a single forward pass per item and the full set
takes minutes.

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
PY=.venv/Scripts/python.exe          # mac/linux: .venv/bin/python

# 0. ALWAYS FIRST on a new model: can its tokenizer represent the three options?
$PY -m src.score --model <MODEL> --check-tokenization-only
#    If ' Unsure' is not a single token, pick one that is:
#    --third-option Unknown   (Unknown / Maybe / Neither all work on Qwen)

# 1. gates on the item set (13 checks, red blocks the run)
$PY -m src.validate --items items/draft items/seed

# 2. the scored run  (drop --device/--dtype to use CUDA defaults)
$PY -m src.score --model <MODEL> --chat-template \
    --items items/draft items/seed \
    --checkpoint results/partial_<name>.jsonl \
    --out results/run_<name>.json

# 3. baseline: every claim with no evidence attached
$PY -m src.baseline --model <MODEL> --chat-template \
    --items items/draft items/seed \
    --checkpoint results/partial_baseline_<name>.jsonl \
    --out results/baseline_<name>.json

# 4. controls
$PY -m src.score --model <MODEL> --chat-template --shuffle-cases \
    --items items/draft items/seed --out results/run_<name>_shuffled.json
$PY -m src.score --model <MODEL> --chat-template --option-rotations \
    --items items/draft items/seed --out results/run_<name>_rotations.json

# 5. analysis
$PY -m src.analyze --run results/run_<name>.json \
    --baseline results/baseline_<name>.json \
    --out results/analysis_<name>.json
```

Read `results/analysis_<name>.md` top to bottom: section 1 is the primary
endpoint, section 2 the matched subset, section 3 the decorative control with
its pre-committed sentence, section 5 the covariate path, section 6 the
diagnostics. `--checkpoint` is optional on fast hardware; it exists because this
run had to survive being killed roughly every ten minutes.

**What would change the conclusion:** a model with an AUC meaningfully above
chance on this task. If a larger model reaches, say, 0.75 in the diverse
condition and drops to 0.6 in the coherent one, the effect is real and this run
was simply below the capability floor. If it stays flat and symmetric at high
AUC, the hypothesis is in trouble for real. Either is worth knowing; neither can
be decided from a model that is near chance in both conditions.
