# MORNING REPORT — coherence-confidence

Built overnight, autonomously. 11 commits, one per numbered step of the brief.
Everything below is reproducible from the repo; every design call made without
asking is in `DECISIONS.md` (D-001 … D-023).

**Your job this morning is reviewing the 80 items.** Nothing was finalized —
`items/final/` is empty by design and all 80 items are `review_status:
"unreviewed"`, including the two hand-written seed families (D-008).

---

## 1. TL;DR

| | |
|---|---|
| Built | all 11 steps, all committed |
| Tests | **284 passing**, 0 failing (282 offline + 2 that need the network) |
| Item set | **80 items, 20 families × 4 cells, 20 per cell** |
| Validation | **9/9 gates pass** |
| Blind audit | 40/40 flaws found, **0 broken items**, 0 false positives on 40 decoys |
| Real model | full path verified end to end on SmolLM2-135M |
| Big model | **not downloaded** — that is your call, and §7 tells you what to check first |

**Three things I would read before anything else:**

1. **D-019** — the brief's third option `"Unsure"` is **3 tokens** on SmolLM2, so
   its probability mass is unreadable. There is now a hard gate for this and you
   must run it against whatever model you pick. §7.
2. **D-009 (revised)** — the default plain prompt is right for base models and
   catastrophically wrong for instruct models: option coverage 0.008 vs 0.847, a
   factor of ~100. §7.
3. **`results/item_audit.md` → "The asymmetry that matters"** — the flaw in
   `coherent_false` is **1.20 points more salient** (of 5) than in
   `diverse_false`. This is a real confound in the design. It happens to be
   conservative, but it must be reported. §5.

---

## 2. What got built

```
src/models.py        Pydantic model + loaders. 15 enforced invariants.
src/render.py        THE only place a prompt string is built.
src/score.py         THE measurement. Final-position logits, option mass, 3 gates.
src/scoring_types.py ScoreResult: 3-way vs 2-way, mass_covered, abstention.
src/mock_scorer.py   Deterministic mock with hand-computable outputs.
src/synth.py         80 fake items for the plumbing test.
src/smoke.py         End-to-end smoke test, 22 numbers checked against arithmetic.
src/analyze.py       Cells, within-condition AUC, bootstrap CIs, 2x2, ANOVA.
src/validate.py      9 item-set gates.
src/baseline.py      Every claim with NO cases attached.
src/provenance.py    Determinism + run provenance incl. prompts_hash.

scripts/recount_words.py     fix word_count after a passage edit
scripts/lexical_ablation.py  WHICH part of the passage leaks?  (§4)
scripts/make_audit_batches.py  build blind audit batches
scripts/score_audit.py         aggregate the audit
scripts/verify_run.py          is a stored run still valid against the items on disk?
```

~6,900 lines of Python across `src/`, `scripts/` and `tests/`.

## 3. What passed

```
$ .venv/Scripts/python.exe -m pytest tests/ -q
284 passed

$ .venv/Scripts/python.exe -m src.smoke
SMOKE TEST PASSED — 22 headline numbers match values computed on paper

$ .venv/Scripts/python.exe -m src.validate --items items/draft items/seed
[PASS] word_count_parity          grand mean 151.2 words; largest cell deviation +0.61%
[PASS] coherent_one_value_per_dimension    40 items
[PASS] diverse_four_values_per_dimension   40 items
[PASS] no_lexical_giveaway        53.0% (mean of 5 shufflings; max 56.2%; 0/5 over limit)
[PASS] cell_balance               80 items, 20 families, 20 per cell
[PASS] no_duplicate_passages      80 distinct passages
[PASS] no_answer_key_leakage      80 passages scanned
[PASS] closer_word_balance        worst per-family TRUE/FALSE word imbalance = 0
[PASS] all_items_unreviewed       80/80
```

A real model was also run over the **real** item set, end to end, as a final
wiring check (`results/run_base135m_realitems.json`,
`results/analysis_base135m_realitems.md`):

```
SmolLM2-135M (base), --third-option Unknown, plain prompt, 80 real items
mean mass_covered = 0.7326   min = 0.6399          <- healthy coverage
AUC[coherent]     = 0.5125   (400 pairs)
AUC[diverse]      = 0.4750   (400 pairs)
main_effect_coherence = -0.0039
main_effect_truth     = -0.0007
interaction           = +0.0012
```

**Do not read that as a result.** A 135M base model cannot do this task, and
every number sitting on chance is the expected — and reassuring — outcome: it
says the pipeline does not manufacture an effect out of items that a model cannot
actually discriminate. `scripts/verify_run.py` confirms the stored run is
byte-identical to the items now on disk (`prompts_hash 0cb3e4b8afcd`).

The smoke test (step 6, run before any content existed) checks the mock's
hand-computed values exactly: cell means 0.6220 / 0.3095 / 0.4720 / 0.3095,
AUC 0.75 coherent (400 pairs, 300 wins, 0 ties) and 0.55 diverse, abstention
0.25 / 0.00 / 0.45 / 0.00, and that dropping abstentions would move the coherent
AUC from **0.75 to 1.00** — the D-003 policy made concrete rather than asserted.

## 4. What failed, and what I did about it

Four things broke during the build. All four are fixed; three of them changed the
design, so they matter to you.

### 4.1 `"Unsure"` is not a single token (D-019) — **needs your decision**

The brief specifies a single-token stand-in for "not enough evidence". Measured on
the first real tokenizer:

```
role     canonical    n_tok  ids  variants
yes      ' Yes'           1   4   [' Yes', ' yes', 'Yes', 'yes']
no       ' No'            1   6   [' NO', ' No', ' no', 'NO', 'No', 'no']
unsure   ' Unsure'        3   1   [' unsure']          <-- unreadable
unsure   ' Unknown'       1   4   [' Unknown', ' unknown', 'Unknown', 'unknown']
```

The prompt ends in `Answer:`, so the model's next token *is* that string. A
3-token option reads near-zero for structural reasons, which looks exactly like a
model that never abstains. `--require-canonical-single-token` is **on by default**
and turns that into a hard failure naming the working alternatives. I used
`--third-option Unknown` for the verification runs.

Note the other half of this: `No` resolves to **six** token ids and `Yes` to
four. Reading a single id — the obvious implementation — would have thrown away
most of the No mass and biased P(yes) upward by a model-specific amount.

### 4.2 The plain prompt is wrong for instruct models (D-009, revised)

Mean `mass_covered` (how much of the next-token distribution the three options
hold):

| model | prompt | mass_covered |
|---|---|---|
| SmolLM2-135M-**Instruct** | plain (the default) | **0.0082** |
| SmolLM2-135M-**Instruct** | `--chat-template` | **0.8475** |
| SmolLM2-135M (**base**) | plain (the default) | **0.7104** |

On the instruct model with a plain prompt the argmax token was `<|im_end|>` on
**all 80 prompts at mean probability 0.886** — it is chat-tuned, so a completion
prompt is off-distribution and it just wants to end the turn. The renormalized
numbers were still computable and still looked publishable (`p_yes_3way` averaged
0.53, argmax split 56 No / 24 Yes) while being ratios of masses under 1% of the
distribution. The `--min-mass-covered` gate caught it. Full write-up:
`results/step7_real_model_path.md`.

### 4.3 The item set had a lexical giveaway, and it was structural (D-022)

The first complete 80-item draft failed the lexical gate at 62.8%.
`scripts/lexical_ablation.py` (written for this) located it exactly:

| passage part | first draft | now |
|---|---|---|
| full passage | 58.8% | 56.2% |
| lead only | 50.0% | 50.0% |
| cases only | 48.8% | 50.0% |
| **closer only** | **65.0%** | **50.0%** |
| no closer | 48.8% | 50.0% |
| no dates | 61.3% | 48.8% |

The cause was not sloppy wording. Under the original construction three of the
four items in a family shared a "TRUE closer" and `coherent_false` had a distinct
"confound closer", so a classifier that learns nothing except *"does this passage
carry the TRUE closer"* scores **75%** — a ceiling no amount of careful phrasing
could get under.

**The fix.** Every closer is now `{FACT}, {SCOPE}; {TAIL}` where FACT is
CHANGED/SAME on the same quantity, SCOPE is REACH/BLOCK, and TAIL is identical
across the family:

| cell | closer | reading |
|---|---|---|
| `coherent_false` | CHANGED + REACH | confound moved *and* reached the units — false |
| `coherent_true` | CHANGED + BLOCK | same confound named, cannot reach — sound |
| `diverse_true` | SAME + REACH | nothing moved — sound |
| `diverse_false` | SAME + BLOCK | nothing moved; false for its own reason |

Each clause variant appears in **exactly one TRUE and one FALSE item per
family**, so the closer carries zero bag-of-words signal. Only the *conjunction*
CHANGED-and-REACH is diagnostic, and a linear model over unigrams and bigrams
cannot represent a conjunction of two non-adjacent clauses. The giveaway is now
logical rather than lexical, which is what the experiment needs.

Enforced mechanically by a new gate, `closer_word_balance`: within each family
every word in the four closers must appear equally often on the TRUE and FALSE
sides. Worst imbalance across 20 families is now **0**.

I also **rotated** which TRUE cell gets which variant across 10 of the 20
families, so the closer variant is orthogonal to coherence rather than perfectly
correlated with it. Every item records its `closer_variant`, so you can check
directly whether the closing sentence moved confidence.

### 4.4 The lexical gate could be passed by luck (D-023)

While checking the gate's own reliability: on the first draft, single-seed
accuracy ranged **58.8%–68.8%** across ten seeds and breached the 60% limit on
**7 of 10**. With `seed=0` the build passed. It should not have.

The gate now averages 5 CV shufflings, fails on the **mean**, and prints max and
`n/N over limit` so a marginal pass cannot read as a comfortable one.

## 5. The items, and what the audit found

80 items, 20 families × 4 cells, balanced 16 per domain across agriculture,
business_metrics, clinical, education and manufacturing. Flaw types:
20 `shared_confound` (all `coherent_false`), 10 `temporal`, 10 `claim_mismatch`.
Passages 142–154 words, cell means within 0.61% of the grand mean.

**Blind audit (step 9), full results in `results/item_audit.md`.** Auditors saw
only the claim, passage and question under an opaque code — no cell label, no
ground truth, no answer key, and never two items from the same family in one
batch. All 40 TRUE items were mixed in as decoys. Two independent rounds with
different groupings; 160 verdicts; reported flaws then matched against the answer
key by separate strict judges.

| | |
|---|---|
| FALSE items whose intended flaw was found | **40 / 40** |
| **Items flagged BROKEN (flaw not findable)** | **0** |
| TRUE decoys that drew a false positive | **0 / 40** |
| Forced answers matching ground truth | 160 / 160 |
| Auditor named a *different* specific flaw | 0 |

**The finding that matters is not that table.** Mean explicitness (1–5, how
obvious the flaw was):

| | n | mean explicitness |
|---|---|---|
| `temporal` | 10 | 2.55 |
| `claim_mismatch` | 10 | 2.85 |
| `shared_confound` | 20 | 3.90 |
| **`coherent_false`** | 20 | **3.90** |
| **`diverse_false`** | 20 | **2.70** |

**The flaw in `coherent_false` is 1.20 points more salient than the flaw in
`diverse_false`.** This is D-004 arriving in the data rather than a wording slip:
a shared confound has to be *stated* in the passage to be a shared confound at
all, whereas reversed dates and a substituted quantity are things a reader has to
notice. Because AUC is computed within each coherence condition, this predicts
AUC(coherent) > AUC(diverse) from salience alone.

**The direction is lucky.** The hypothesis predicts coherence *inflates*
confidence and therefore *depresses* AUC(coherent). This artifact pushes the
other way, so it makes the test conservative — a coherence effect found in spite
of it is stronger evidence, not weaker. It still has to be reported.

Two things already in place to separate them: `coherence_effect_within_true`
(both cells flawless, so salience cannot touch it) is reported separately, and
`diverse_false` is tagged `temporal` vs `claim_mismatch` so the two can be
compared against `coherent_false` without pooling.

16 items are flagged "too easy" under my pre-registered threshold (mean
explicitness ≥ 4). **All 16 are `coherent_false`.** I would not act on that count
directly: the brief's own construction *requires* the confound to be in the
passage, so a 4 is close to inherent, and nothing scored 5 ("the passage states
the problem outright"). The asymmetry above is the real signal.

## 6. Every assumption I made

Full reasoning in `DECISIONS.md`. The ones that could change your results:

| | assumption | why it could matter |
|---|---|---|
| D-001 | The brief says "78 drafts" but also "20 per cell, 80 total". I built **80** (20 families × 4). | 78 would give 86 items and unbalanced cells, breaking the stated 400-pair AUC. |
| D-002 / D-019 | Third option presented as a single word; `"Unsure"` kept as the default but **gated**. | The word you end up using changes `template_hash`; runs across different words can never be pooled. |
| D-003 | Abstentions **included** in AUC. No CLI flag exists to drop them. | Dropping would have moved the mock's coherent AUC 0.75 → 1.00. |
| D-004 | `coherent_false` and `diverse_false` are false for structurally different reasons. Forced by the brief. | See §5 — this is the main threat to the interpretation. |
| D-009 | Plain completion prompt by default, chat template opt-in. | Wrong for instruct models by a factor of 100 in coverage. Check `mass_covered`. |
| D-010 | Both item-level and family-clustered bootstrap CIs, 10,000 resamples. | For the coherence effect and the interaction, **read `ci_family`** — the 2x2 is within-family and the item-level interval throws the pairing away. Measured width ratio 0.577. |
| D-015 | Lexical check is 5-fold CV **grouped by family**. | In-sample would be ~100% by construction and the gate would be useless. |
| D-017 | Auditors and item authors are the same model family. | The audit measures *findability*, not difficulty for the model under test. |
| D-022 | Closers rebuilt on a balanced 2-clause scheme; TRUE-cell variants rotated. | `coherent_true` and `diverse_true` no longer share a closer — arithmetically unavoidable given balance. Rotation makes it orthogonal to coherence. |

## 7. What to do next, in order

**Step 1 — check your model's tokenizer before anything else.** Costs seconds,
downloads no weights, and answers the one question that invalidates everything
downstream:

```bash
cd coherence-confidence
.venv/Scripts/python.exe -m src.score --model <YOUR_MODEL> --check-tokenization-only
```

Read the `n_tok` column. Every option must be **1**. If `unsure` is not, the
command prints which candidate words are single tokens on that vocabulary; pick
one and pass `--third-option <Word>` to every command below.

**Step 2 — score, with the right prompt style for your model class.**

```bash
# BASE model (no chat tuning): plain prompt, the default
.venv/Scripts/python.exe -m src.score --model <YOUR_MODEL> --third-option <WORD> \
    --items items/draft items/seed --out results/run_<name>.json

# INSTRUCT / chat-tuned model: you almost certainly need --chat-template
.venv/Scripts/python.exe -m src.score --model <YOUR_MODEL> --third-option <WORD> \
    --chat-template --items items/draft items/seed --out results/run_<name>.json
```

**Check the `mean mass_covered` line it prints.** Below ~0.3 the numbers are not
interpretable and the run exits non-zero. If that happens, look at `top_token` in
the records — that is what diagnosed the instruct-model problem in minutes.

**Step 3 — baseline, so you can subtract what the model already believed.**

```bash
.venv/Scripts/python.exe -m src.baseline --model <YOUR_MODEL> --third-option <WORD> \
    [--chat-template] --items items/draft items/seed --out results/baseline_<name>.json
```

On SmolLM2-135M the priors ranged 0.515 ("this fertilizer makes plants grow
taller") to 0.614 ("this routing change cuts delivery time") — a 10-point spread
across families before any evidence. Do not skip this.

**Step 4 — analyze, both raw and baseline-subtracted.**

```bash
.venv/Scripts/python.exe -m src.analyze --run results/run_<name>.json \
    --baseline results/baseline_<name>.json --out results/analysis_<name>.json

.venv/Scripts/python.exe -m src.analyze --run results/run_<name>.json \
    --baseline results/baseline_<name>.json --score delta \
    --out results/analysis_<name>_delta.json
```

Both write a markdown twin next to the JSON. Read `analysis_<name>.md` top to
bottom; it states the abstention policy, which CI to read for which quantity, and
the ANOVA's independence caveat inline.

**Step 5 — reviewing the items (your actual task).** One file per family, four
cells adjacent, so the review is reading one file top to bottom and checking that
only the intended things changed:

```bash
ls items/draft/          # 18 generated families
ls items/seed/           # 2 hand-written families - read these first, they are the template
```

Then re-validate and re-audit after any edit:

```bash
.venv/Scripts/python.exe scripts/recount_words.py items/draft items/seed
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed
.venv/Scripts/python.exe scripts/lexical_ablation.py
```

The word-count check will refuse to load an item whose stored count no longer
matches its passage — that is deliberate, and `recount_words.py` is the fix.

**Step 6 — before quoting any stored run**, confirm the items have not moved
under it:

```bash
.venv/Scripts/python.exe scripts/verify_run.py results/run_<name>.json
```

## 8. Open items I did not do

1. **The `coherent_false` salience gap (§5).** Softening the 16 flagged items
   would narrow the 1.20-point gap. I did not touch them: it is a substantive
   content change, the gap is conservative rather than flattering, and you asked
   to review the items yourself. If you do soften them, re-run the audit —
   `scripts/make_audit_batches.py` and `scripts/score_audit.py` make that ~10
   minutes.
2. **A second auditing model.** D-017: the auditors share an author with the item
   writer, so 40/40 is evidence about findability, not difficulty. Re-running the
   audit under a different model would be the single cheapest way to strengthen
   the item-quality claim.
3. **No large model downloaded.** As instructed. Everything in §7 works unchanged
   on any HF causal LM.
4. **`items/final/` is empty.** Promotion is deliberately not scriptable (D-008).
5. **Batched scoring is implemented but defaults to `--batch-size 1`.** Left
   padding plus attention-mask-derived `position_ids` is correct as written, but
   I did not have a large enough model to verify batched output matches unbatched
   to floating-point tolerance. If you turn it up, check a handful of items
   against `--batch-size 1` first.

## 9. Reproduce the whole thing

```bash
git clone <this repo> && cd coherence-confidence
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

.venv/Scripts/python.exe -m pytest tests/ -q          # 284 pass
.venv/Scripts/python.exe -m src.smoke                 # no model needed
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed
```
