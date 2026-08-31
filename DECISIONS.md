# DECISIONS

Every ambiguity resolved without asking. Bias throughout: pick the option that is
**easier to reverse**.

Format: `D-NNN` | what was ambiguous | what I chose | why | how to reverse.

---

## D-001 — "78 drafts" vs "20 items per cell"

**Ambiguous.** The brief says *"write the 2 seed families by hand, then generate
78 drafts"* but also specifies *"20 items per cell, 80 total"* and that each seed
family gets *"all 4 versions"*. 2 hand-written families x 4 cells = 8 items, so
80 - 8 = **72** generated items, not 78.

**Chose:** the 20-per-cell constraint wins. **20 families x 4 cells = 80 items**:
2 seed families hand-written (8 items) + 18 generated families (72 items). Each
cell ends up with exactly 20 items, which is what the AUC math in the brief
("20 true vs 20 false = 400 pairs") requires.

**Why:** 400 pairs per within-condition AUC is stated as a hard number and only
holds at 20/cell. Generating 78 would give 86 items and unbalanced cells.

**Reverse:** generate 6 more items (1.5 families) if the literal 78 was intended;
items are independent files, nothing else depends on the count except the
`400 pairs` assertion in `tests/test_analyze.py`.

---

## D-002 — Third option is "Unsure", not "Not enough evidence"

**Ambiguous only in wording** — the brief already directs this, but the exact
stand-in token was mine to pick.

**Chose:** the prompt literally offers `Yes / No / Unsure`, and the answer is read
from the single next-token distribution.

**Why:** "Not enough evidence" is 3-5 tokens depending on tokenizer. Reading it
from a single next-token distribution is impossible; scoring it as a sequence
would put it on a different footing than the 1-token Yes/No (length bias,
different normalization), which silently corrupts the three-way comparison. A
single-token stand-in keeps all three options on identical footing.

**Cost of this choice, stated plainly:** "Unsure" is a slightly different speech
act than "not enough evidence" — it can read as epistemic hedging about the
model's own state rather than a claim about the evidence. The prompt therefore
spells out the mapping in the instruction line: *"Unsure = the evidence is not
enough to decide."* This is a wording mitigation, not a fix.

**Reverse:** `src/render.py` holds `DEFAULT_OPTIONS` and the instruction text in
one place, and options are addressed by role (`yes`/`no`/`unsure`) everywhere
else, so the surface word can be swapped with `--third-option` without touching
anything downstream. Swapping in a sequence-scored multi-token option would mean
changing `HFScorer.score_prompts` only.

**See also D-019** — measuring the tokenizer showed "Unsure" does not satisfy
this decision's own single-token requirement on every model, which is now a hard
gate rather than an assumption.

---

## D-003 — Abstentions are INCLUDED in AUC

**Chose:** items where `Unsure` is the argmax are included in both AUCs using
their `p_yes_3way` value. They are never dropped, never imputed to 0.5.

**Why:** dropping them makes abstention a free way to inflate AUC — a model that
abstains on exactly the items it would have gotten wrong looks like a better
discriminator than one that guesses. The abstention rate is reported *separately
per cell* so it can be read alongside AUC rather than laundered into it.

**Made visible rather than hypothetical:** `analyze.py` also computes
`auc_if_abstained_dropped_DIAGNOSTIC` and prints it next to the real AUC, so the
size of the effect this policy prevents is on the page. On the mock's `known`
scenario the coherent AUC is 0.75 including abstentions and **1.00** if they are
dropped — the mock abstains on exactly the 5 TRUE items it scored lowest, i.e.
the ones it got wrong.

**Reverse:** `analyze.py::stat_auc` takes `drop_abstained`, but no CLI flag
exposes it. Using it means editing the call site, so it can never happen by
accident or by habit.

---

## D-004 — coherent_false and diverse_false are false for structurally different reasons

**Not reversible — this is forced by the design, and it is the main threat to the
interpretation.**

`coherent_false` is broken by a **single shared confound** that explains all four
cases. `diverse_false` cannot be broken that way (a single confound cannot cover
four cases that differ on every dimension), so per the brief it is broken by
either (a) **temporal impossibility** — the effect is dated before the cause in
>=2 of the 4 cases, or (b) **claim mismatch** — the cases don't measure what the
claim asserts.

**Consequence:** the coherence main effect is not perfectly clean, because the
*kind* of flaw covaries with coherence in the FALSE rows. A coherence effect
could in principle be "shared confounds are harder to spot than bad dates"
rather than "coherence inflates confidence".

**Mitigations applied:**
1. The TRUE rows are structurally identical across coherence (no flaw at all in
   either), so the coherence main effect **within TRUE items** is clean and is
   reported separately in the analysis output as `coherence_effect_within_true`.
2. `diverse_false` flaw type is recorded per item in `confound_note` and in a
   machine-readable `flaw_type` field, split roughly evenly between `temporal`
   and `claim_mismatch`, so the two can be analyzed separately.

**Recorded here rather than fixed** because fixing it would require breaking the
brief's stated construction of `diverse_false`.

---

## D-005 — Word-count parity strategy

**Ambiguous.** `validate.py` must fail if any cell's mean word count deviates >10%
from the grand mean, but the FALSE cells naturally need an extra clause (the
confound sentence / the dates).

**Chose:** every item is written to a **target of 95-125 words**, and the TRUE
items carry a matched-length neutral detail sentence in the same slot where FALSE
items carry the confound. The neutral sentence is *contentful but
non-diagnostic* (e.g. describing measurement procedure), never filler.

**Why:** the alternative — letting FALSE items run long — makes length itself a
cue, which the lexical-giveaway check would not catch (it is unigram/bigram
based, not length based).

**Reverse:** the neutral sentences are the last sentence of each TRUE passage and
are individually editable.

---

## D-006 — Model for the end-to-end scorer smoke test

**Chose:** `HuggingFaceTB/SmolLM2-135M-Instruct` (~270 MB fp32).

**Why:** the brief says "smallest model that will download quickly (e.g. a 1B)"
and explicitly says not to download a large model. 135M downloads in seconds and
exercises exactly the same code path as any causal LM. The point of step 7 is to
prove the *path* runs, not to get a meaningful number — and its numbers are
explicitly NOT interpretable as a result.

**Reverse:** `--model` is a plain CLI arg. Any HF causal LM name works.

---

## D-007 — Condition dimensions are per-family, not global

**Ambiguous.** The brief names four dimensions (region, time, device, unit type)
from Family A/B, but "device" is meaningless for an agriculture item.

**Chose:** every family declares its own **4 named dimensions** in the item's
`cases[].conditions` dict. The validator enforces the *structure* (coherent = 1
distinct value per dimension; diverse = 4 distinct) and that every case in an item
carries the same dimension keys — it does not enforce a global dimension
vocabulary.

**Why:** forcing "device" onto a soil-trial item would produce nonsense passages,
and nonsense is a bigger threat to the experiment than schema tidiness.

**Reverse:** add a global enum to `schema.json` if a fixed vocabulary is wanted.

---

## D-008 — `items/final/` stays empty

**Chose:** the build writes **nothing** to `items/final/`. All 80 items live in
`items/draft/` (72) and `items/seed/` (8) with `review_status: "unreviewed"`.

**Why:** the brief says "Do not finalize any item. My job in the morning is
reviewing them." The seed families are hand-written but they are still unreviewed
*by the human*, so they are marked unreviewed too — being hand-written is not the
same as being approved.

**Reverse:** `scripts/promote.py` is intentionally not written. Promotion should
be a deliberate human act.

---

## D-009 — Prompt format is base-LM style, not chat-template style

**Chose:** `src/render.py` emits a plain completion-style prompt ending in
`Answer:` with no chat template applied, and this is the default for all models.
`--chat-template` is available as an opt-in flag.

**Why:** applying a chat template changes the token immediately preceding the
answer position, which changes the Yes/No/Unsure logits in a model-specific way.
Keeping the default template-free makes runs across different models comparable,
which is the whole point of a 2x2 whose effect sizes get compared. Mixing
templated and untemplated runs would be the silent kind of error this repo is
built to avoid.

**MEASURED, and the default turns out to be model-class-dependent.** On the
synthetic smoke items, mean `mass_covered` (how much of the next-token
distribution the three options hold):

| model | prompt | mean mass_covered |
|---|---|---|
| SmolLM2-135M-**Instruct** | plain (default) | **0.0082** |
| SmolLM2-135M-**Instruct** | `--chat-template` | **0.8475** |
| SmolLM2-135M (**base**) | plain (default) | **0.7104** |

A factor of ~100. On the instruct model with a plain prompt the argmax token was
`<|im_end|>` on all 80 prompts at mean probability 0.886 — it is chat-tuned, so a
completion-style prompt ending in `Answer:` is off-distribution and it just wants
to end the turn. The renormalized numbers were still computable and still looked
publishable (`p_yes_3way` averaged 0.53, argmax split 56 No / 24 Yes); they were
ratios of masses under 1% of the distribution. The `--min-mass-covered` gate is
what caught it.

**So:** the plain-prompt default is correct for **base** models and wrong for
**instruct** models. Use `--chat-template` for anything instruction-tuned, and
check `mass_covered` in the run output before believing any number. Details and
the reproduction commands are in `results/step7_real_model_path.md`.

**Reverse:** `--chat-template` flag, already implemented; the run JSON records
which was used, and `template_hash` differs between the two, so runs can never be
silently pooled.

---

## D-010 — Bootstrap resampling unit

**Ambiguous.** Resample items, or resample families?

**Chose:** resample **items within cell** (stratified: each cell resampled to its
own n) for cell means and abstention rates, and resample **pairs' underlying
items** for AUC — i.e. resample the 20 true and 20 false scores independently and
recompute the pairwise win rate.

**Why:** it is the standard, and it matches the unit the brief names ("20 true vs
20 false = 400 pairs").

**Corrected after measuring.** My first draft of this decision said the
item-level CIs would be anti-conservative (too narrow) because items within a
family are correlated. That is the standard argument and it is **wrong for this
design**. Measured on the mock (3,000 resamples), CI width ratio family/item:

| statistic | item width | family width | ratio |
|---|---|---|---|
| cell mean, `coherent_true` | 0.2676 | 0.2676 | **1.000** |
| AUC within coherent | 0.3500 | 0.3500 | **1.000** |
| main effect of truth | 0.2256 | 0.2812 | 1.247 |
| main effect of coherence | 0.2276 | 0.1312 | **0.577** |
| interaction | 0.4551 | 0.2625 | **0.577** |
| coherence effect within TRUE | 0.4554 | 0.2625 | 0.576 |

Two things fall out of the design, and both are structural rather than artifacts
of the mock:

1. **For any per-cell statistic the two units coincide.** The design is fully
   crossed with exactly one item per cell per family, so resampling 20 families
   and taking their `coherent_true` items *is* resampling 20 `coherent_true`
   items. The two resampling **distributions** are identical; the realized
   percentiles differ only by Monte Carlo noise (~2e-4 at 2,000 resamples),
   because the two resamplers consume the RNG stream differently.
2. **For contrasts that span cells, clustering makes the CI NARROWER, not
   wider.** The 2x2 is within-family by construction, so a family-clustered
   resample keeps each family's four cells together and the contrast is a
   *paired* comparison. That is a variance reduction, the same one a paired
   t-test buys over an unpaired one. The item-level CI on the coherence effect
   is the anti-conservative one only in the sense of being wrong — it is too
   *wide*, and it throws away the pairing the design was built to exploit.

**Consequence for reading the output:** for cell means and AUCs, either CI will
do (they are identical). **For the coherence effect, the interaction, and the
within-TRUE contrast, read `ci_family`** — the item-level interval ignores the
pairing. `analyze.py` prints this guidance in the markdown report so it does not
have to be remembered.

**Reverse:** both are computed and reported on every number; nothing has to be
re-run to switch which one is read.

---

## D-011 — Determinism

**Chose:** `seed=0` set on `random`, `numpy`, and `torch`; `torch.use_deterministic_algorithms(True)`
where available; `model.eval()` + `torch.no_grad()`; dtype forced to `float32` on
CPU; `temperature` recorded in the run metadata as `1.0` even though it does not
affect raw logits (the brief asked for it to be set anyway, and recording it
prevents a future reader from assuming sampling happened).

Each run JSON records: model name, resolved HF **revision commit hash**, dtype,
device, prompt template hash, code git SHA, and UTC timestamp.

**Reverse:** `--seed`.

---

## D-012 — No network at item-generation time

**Chose:** items are authored as data files in-repo. No LLM API call is made by
any script in `src/` to generate items.

**Why:** items must be diffable, reviewable, and stable across runs. A generation
script that hits an API would make the item set non-reproducible.

**Reverse:** n/a — the item JSONs are the artifact.

---

## D-013 — Storage format is one JSON file per family

**Chose:** `items/draft/fam_<id>.json` holds all 4 cells of one family as a list.

**Why:** the 2x2 is only interpretable family-by-family; keeping the four
versions adjacent in one file makes human review (the stated morning task) a
matter of reading one file top to bottom and checking that only the intended
things changed.

**Reverse:** trivial reshape; loaders glob and flatten regardless.

---

## D-014 — `word_count` counts the rendered passage only

**Chose:** `word_count` = whitespace-token count of `passage`, excluding the
claim, the question, and the option list.

**Why:** the claim and question are identical across all 4 cells of a family, so
including them would dilute exactly the between-cell length differences the
validator exists to catch.

**Reverse:** one function, `models.py::compute_word_count`.

---

## D-015 — Lexical-giveaway check uses cross-validated accuracy

**Ambiguous.** The brief says "a logistic regression on unigrams+bigrams can
predict true vs false above 60%" — in-sample or held-out?

**Chose:** **5-fold stratified cross-validated** accuracy, grouped by family so
that all 4 cells of a family are in the same fold.

**Why:** in-sample accuracy on 80 short passages with thousands of bigram features
is ~100% by construction and would fail the build unconditionally, making the gate
useless. Grouping by family prevents the classifier from learning "this family's
vocabulary" and cheating across the fold boundary.

**Reverse:** `--lexical-cv {grouped,stratified,insample}` flag on `validate.py`.

---

## D-016 — Seed families' extra dimension

**Chose:** Family A (fertilizer) uses dimensions `species / location / season /
observer`; Family B (checkout) uses `device / country / month / traffic_source`.
The brief names three varying things for each; a fourth was added so every family
has exactly 4 dimensions.

**Why:** uniform dimensionality makes the coherent/diverse validator checks
symmetric across families and keeps the "shares *every* irrelevant condition"
manipulation strong.

**Reverse:** drop the 4th dimension from both files; validator reads dimension
keys from the item.

---

## D-017 — Self-check audit is blind-ish, not truly blind

**Chose:** step 9's per-item flaw identification was performed by independent
agents given **only the rendered passage, the claim, and the question** — not the
cell label, not `ground_truth`, and not `confound_note`. Each FALSE item was
audited by multiple independent agents and their verdicts pooled.

**Why:** it is the closest achievable approximation to a blind check within one
build. Stated limitation: the auditing model shares an author with the item
writer, so "the flaw is findable" is evidence about *findability*, not about
*difficulty for the model under test*. Recorded in `results/item_audit.md`.

**Reverse:** re-run with a different model, or hand the drafts to a human.

---

## D-019 — The third option must be a single token in its *presented* form, and this is a hard gate

**Found while building, not anticipated by the brief. Read this one.**

D-002 chose a single-token stand-in for "Not enough evidence". Measuring the
actual tokenizer showed the stand-in the brief names does not satisfy its own
requirement on the first model I tried:

```
HuggingFaceTB/SmolLM2-135M-Instruct
  " Yes"     -> 1 token       variant set: {' Yes', ' yes', 'Yes', 'yes'}          (4 ids)
  " No"      -> 1 token       variant set: {' NO',' No',' no','NO','No','no'}      (6 ids)
  " Unsure"  -> 3 tokens      variant set: {' unsure'}                             (1 id)
```

The prompt ends in `Answer:` with no trailing space, so the model's very next
token *is* the canonical form. If a model wants to answer Unsure it puts its mass
on `" Un"`, which is not in the option's id set — and `" Un"` is far too
promiscuous a prefix to count (Under, Unlike, Until...), so a first-token
fallback would be worse than the disease. The measured result is that the third
option reads artificially near zero, abstention rate reads near zero, and
`p_yes_3way` collapses toward the two-way number — **silently**. That is exactly
the class of bug the brief asked to avoid, one level down.

**Chose:**
1. `"Unsure"` stays the default option word, as the brief specifies.
2. `src/score.py` computes, per option, whether the **canonical presented form**
   (`" " + word`) is a single token, and records `canonical_n_tokens` in every
   run file.
3. `--require-canonical-single-token` is **ON by default** and makes a
   multi-token option a hard `TokenizationError` rather than a silent bias. The
   error names the offending option, its token count, and the candidate words
   that *are* single tokens on that tokenizer.
4. `--third-option <Word>` swaps the surface word. Options are addressed
   internally by role (`yes`/`no`/`unsure`), never by surface string, so nothing
   downstream changes when the word does.
5. `template_hash` incorporates the option words, so a run with `Unknown` can
   never be pooled with a run using `Unsure`.

**Measured single-token candidates on SmolLM2-135M:** `Unknown`, `Maybe`,
`Neither` (each 1 token in the `" Word"` form, with the same full casing/space
variant coverage as Yes and No). `Unclear`, `Uncertain`, `Insufficient` are 2
tokens; `Undecided` is 3.

**What I used and why:** the end-to-end real-model run in step 7 uses
`--third-option Unknown`. It is the closest in meaning to "not enough evidence"
among the words that are actually readable, and it matches Yes/No's variant
profile exactly, so no option is structurally advantaged.

**This needs a human decision in the morning**, because the answer depends on the
real model: run `python -m src.score --model <yours> --check-tokenization-only`
to see the table for the chosen model before running anything else.

**Reverse:** `--no-require-canonical-single-token` downgrades the gate to a
warning and lets the run proceed with `Unsure`. Deliberately verbose to type.

---

## D-018 — Torch install is CPU-only

**Chose:** `torch` from the PyTorch CPU wheel index.

**Why:** no GPU assumption should be baked into an item-authoring repo, the smoke
model is 135M, and CPU wheels are ~200 MB vs multi-GB for CUDA.

**Reverse:** reinstall torch from the CUDA index; `score.py` already does
`--device auto` and will pick up `cuda` if present.

---

## D-020 — Three validation gates beyond the four the brief lists

**Chose:** `src/validate.py` also fails on **cell balance** (20 per cell, all
families complete), **duplicate passages**, and **answer-key leakage**
(`confound_note` appearing in the passage, or a metadata token like
`coherent_false` appearing in the prose). They are tagged
`required_by_brief: false` in the report so the four gates the brief asked for
are still distinguishable at a glance.

**Why:** each catches a failure that otherwise produces a confident, meaningless
number rather than an error.

- Duplicate passages are the sharpest case. Two items with the same passage
  render to the same prompt and cannot be told apart by *any* measurement — the
  2x2 silently becomes one measurement repeated. This actually happened while
  building: the test fixtures rendered all 80 synthetic passages identically and
  every cell mean came back equal. Nothing failed; the numbers were just wrong.
  `MockScorer.prepare` now raises on it too.
- Answer-key leakage would let the model read the confound off the page, which
  turns a reasoning item into a reading-comprehension item.
- Cell imbalance breaks the "400 pairs" arithmetic and the balanced-ANOVA
  formulas without any of them noticing.

**Reverse:** `--skip <check_name>` disables any individual gate.

---

## D-021 — Lexical check uses the passage only, not the claim

**Chose:** the unigram+bigram classifier sees `passage` and nothing else.

**Why:** the claim is identical across all 4 cells of a family by construction,
so it carries exactly zero true/false signal. Including it would add ~20 constant
features per family that dilute the ones that matter, making the gate *less*
sensitive to a real giveaway.

**Reverse:** one line in `check_lexical_giveaway`.
