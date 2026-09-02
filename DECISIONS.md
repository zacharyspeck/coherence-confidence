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

---

## D-022 — Balanced two-clause closers, because the first draft leaked

**Found by the gate, not by inspection. This is the largest content change in the
build.**

The first complete 80-item draft failed the lexical gate. `scripts/lexical_ablation.py`
(written for this) attributes it exactly:

| passage part | grouped-CV accuracy at TRUE vs FALSE |
|---|---|
| full passage | 58.8% |
| lead only | 50.0% |
| **cases only** | **48.8%** — the case sentences leak nothing |
| **closer only** | **65.0%** — the entire leak is here |
| no closer | 48.8% |
| no dates | 61.3% — so the temporal flaw's dates are not the problem |

The cause was structural, not sloppy wording. Under the original design, three of
the four items in a family shared one "TRUE closer" and the fourth
(`coherent_false`) had a distinct "confound closer". So every word of the
confound closer was a 1-in-4 marker for FALSE. A classifier that learns nothing
except *"does this passage carry the TRUE closer"* scores **75%** — the design had
a lexical ceiling far above the 60% limit no matter how carefully the sentences
were phrased.

**Chose:** every closer is now `"{FACT}, {SCOPE}; {TAIL}"` with

- `FACT` — what the most obvious alternative explanation *did*: `CHANGED` (it
  moved sharply) or `SAME` (it did not). Same quantity in both variants.
- `SCOPE` — whether that explanation could *reach* the observed units: `REACH` or
  `BLOCK`. Both scopes must read naturally after both facts.
- `TAIL` — one clause, byte-identical across all four items of the family.

assigned as

| cell | closer | why |
|---|---|---|
| `coherent_false` | CHANGED + REACH | confound live and reaching; covers all four cases at once *only* because they share every condition |
| `coherent_true` | CHANGED + BLOCK | same confound named, cannot reach the units |
| `diverse_true` | SAME + REACH | nothing to worry about |
| `diverse_false` | SAME + BLOCK | nothing to worry about; false for its own reason |

**Why this works:** `CHANGED` appears in exactly one TRUE and one FALSE item. So
does `SAME`, so does `REACH`, so does `BLOCK`. Every clause is perfectly balanced
against the label, so a bag-of-words model gets **zero** signal from the closer.
Only the *conjunction* CHANGED-and-REACH is diagnostic, and a linear model over
unigrams and bigrams cannot represent a conjunction of two non-adjacent clauses.
The giveaway is now logical rather than lexical, which is exactly the property
the experiment needs.

**Cost, stated plainly:** `coherent_true` and `diverse_true` no longer share a
closer (CHANGED+BLOCK vs SAME+REACH). Perfect within-family balance is arithmetically
incompatible with the two TRUE items being identical — with 2 TRUE and 2 FALSE
per family, if both TRUE items carry the same clauses those clauses appear twice
on the TRUE side and the FALSE side cannot balance them. So D-004's mitigation #1
("the TRUE rows are structurally identical across coherence") is now weaker: the
`coherence_effect_within_true` contrast carries a closer difference as well as a
condition difference.

**And how it was fixed — applied, not deferred.** Leaving CHANGED+BLOCK on
`coherent_true` in every family would have *confounded* the closer variant with
coherence. It is now **rotated**: in 10 of the 20 families `coherent_true` carries
CHANGED+BLOCK and `diverse_true` carries SAME+REACH; in the other 10 they are
swapped. Balance is untouched by the swap (CHANGED still appears in one TRUE and
one FALSE item, and so on), and the closer variant is now orthogonal to coherence
rather than perfectly correlated with it.

Rotated families: `fam_checkout`, `fam_driptape`, `fam_filter`, `fam_inhaler`,
`fam_nestbox`, `fam_physio`, `fam_reading`, `fam_seedcoat`, `fam_solder`,
`fam_tutoring`.

Every item now records which closing sentence it carries in a new
`closer_variant` field (`changed_reach` | `changed_block` | `same_reach` |
`same_block`), so the analysis can check directly whether the variant moved
confidence. The model enforces the one rule that matters: **only `coherent_false`
may be `changed_reach`**, because that combination — the confound moved *and*
reached the units — is what makes an item false, and no other cell is allowed to
contain it.

**Result.** Every part of the passage is now at chance in isolation:

| passage part | before | after |
|---|---|---|
| full passage | 58.8% | 56.2% |
| **closer only** | **65.0%** | **50.0%** |
| cases only | 48.8% | 50.0% |
| no dates | 61.3% | 48.8% |

and the gated mean over 5 shufflings went from **62.8% (3/5 over limit)** to
**53.0% (0/5 over limit)**.

---

## D-023 — The lexical gate averages over 5 CV shufflings

**Found the same way.** On the first full draft, single-seed grouped-CV accuracy
ranged from **58.8% to 68.8%** across ten seeds and breached the 60% limit on
**7 of 10**. With `seed=0` the build passed. It should not have.

**Chose:** `check_lexical_giveaway` averages over `--lexical-seeds` (default 5)
CV shufflings and fails on the **mean**. `accuracy_max`, `accuracy_per_seed` and
`n_seeds_over_threshold` are reported alongside, and the summary line reads e.g.
`accuracy 62.8% (mean of 5 shufflings; max 68.8%; 3/5 over limit)` so a marginal
pass can never look like a comfortable one.

**Why:** 80 items in 20 groups is a small sample; one shuffle is not an estimate.
A gate that a rerun can flip is not a gate.

**Reverse:** `--lexical-seeds 1` restores the old single-shuffle behaviour.

---

# Fix pass (D-024 onward)

## D-024 — `flaw_mechanism`, and a mechanism that appears in BOTH false cells

**The problem.** The blind audit measured flaw salience at **3.90** in
`coherent_false` and **2.70** in `diverse_false`. Louder flaws get caught more
often, which inflates AUC(coherent). Since the hypothesis predicts AUC(coherent)
is *lower*, that handicap is survivable if the effect appears anyway and fatal if
it does not — a null could not be distinguished from an effect cancelled by
louder flaws.

**Chose:** every FALSE item now declares a `flaw_mechanism`, and the taxonomy is
deliberately **orthogonal to coherence**:

| mechanism | how the item is false | where it can appear |
|---|---|---|
| `stated_confound` | a fact in the passage is an alternative cause covering every case at once | `coherent_false` only — a single stated fact cannot cover four cases that differ on every dimension (D-004) |
| `broken_chronology` | in >=2 cases the outcome is dated before the treatment | either |
| `scope_mismatch` | the outcome is tallied over a subset of the population the claim is about | **both** |

`scope_mismatch` is the matched mechanism. It is falsifiable from the text alone,
uses no external knowledge and hides nothing: the passage states what population
the outcome covers and, separately, whether that population is everyone. The
reader has to put the two together.

**Assignment.** 10 "scope families" have `scope_mismatch` in *both* false cells;
the other 10 keep `stated_confound` + `broken_chronology`. Counts:
`coherent_false` = 10 stated_confound + 10 scope_mismatch; `diverse_false` = 10
broken_chronology + 10 scope_mismatch.

**Why the same 10 families for both cells:** the matched subset then draws on
identical scenarios in both conditions, so the only thing differing is coherence.
`analyze.py` checks this and prints a warning if it ever stops holding.

**Reverse:** `flaw_mechanism` is a per-item field; the subset is selected at
analysis time, not baked in.

---

## D-025 — AUC is the primary endpoint; mean confidence is a diagnostic

**Chose:** the analysis leads with **AUC(coherent) − AUC(diverse)** and prints
the matched-mechanism version of the same quantity directly beneath it. Cell
means, the 2x2 effects and the ANOVA moved to a section explicitly headed
"Diagnostics (not endpoints)".

**Why:** the consensuality principle is a claim about the *confidence-accuracy
relationship*, not about confidence level. A model could show a large coherence
effect on mean confidence while ranking true above false perfectly well — that
would not be the effect. AUC is the confidence-accuracy relationship; a negative
gap is the prediction, and AUC(coherent) below 0.5 is the crossover.

The gap is now a first-class statistic with its own family-clustered bootstrap
CI, rather than something the reader computes by eye from two numbers.

**Reverse:** ordering lives in `to_markdown`; every quantity is still in the JSON
regardless of where it prints.

---

## D-026 — The clause assignment is forced, and my first one leaked 62% by construction

The mid-passage line carries two independent clause pairs; exactly one
combination of each falsifies an item:

    CONFOUND  changed/same x reach/block          -> `changed_reach` is live
    SCOPE     subset/whole x incomplete/complete  -> `subset_incomplete` is live

**The mistake.** My first assignment gave scope families `subset_incomplete` on
both false items and left confound families self-balancing. That looks fine per
family, but globally it put `complete` at 20T/10F and `incomplete` at 20T/30F. A
classifier that learns nothing except "complete -> TRUE, incomplete -> FALSE"
scores **62.5%** on 80 items. Measured on the fixture: **75%**. The design leaked
before a single sentence was written.

**The fix is not a matter of taste — it is the unique solution.** Writing a for
TRUE items at (subset,complete), b at (whole,incomplete), c at (whole,complete),
and d/e/f for the non-scope FALSE items across the same three combinations, with
20 scope-FALSE items pinned at (subset,incomplete), balance requires

    a = d + 20      (subset)        b + c = e + f   (whole)
    b = e + 20      (incomplete)    a + c = d + f   (complete)

Substituting into a+b+c = 40 gives c = -d-e, and since all three are
non-negative, **c = d = e = 0**. So a = 20, b = 20, f = 20:

| items | clause pair |
|---|---|
| 20 TRUE | `subset_complete` |
| 20 TRUE | `whole_incomplete` |
| 20 FALSE (non-scope) | `whole_complete` |
| 20 FALSE (scope) | `subset_incomplete` |

There is no other balanced assignment. Every scope clause now sits at exactly
20T/20F, and the confound clauses at 20T/20F each as well. Verified in the item
set, not assumed.

**Consequence.** Only the *conjunction* of two non-adjacent clauses is
diagnostic, and a linear model over unigrams and bigrams cannot represent that.
Measured on the rebuilt items: lexical gate **49.3%** (0/5 shufflings over the
limit), and every passage component at or below chance in isolation — cases
53.8%, closers 50.0%, lead 50.0%.

**Residual, and why the tolerance is 2 rather than 0.** Per family, perfect
balance is impossible: when both false items share a mechanism — exactly what the
matched-mechanism subset requires — the falsifying clause is 2F against at most
1T, since a TRUE item carrying it would not be true. That is a hard +1. Real
clause wordings add about one more through incidental function-word overlap
between variants of unequal length. Measured: fixture 1, real items 2. The
per-family gate exists to name a family that drifts past that; the lexical gate
is the real test.

**Also fixed here:** the TRUE items' scope clauses are rotated between
`coherent_true` and `diverse_true` in half the families, so `scope_variant` is
orthogonal to coherence rather than perfectly correlated with it. The swap keeps
both items true and changes no count, so it is free.

---

## D-027 — No imputation in any passage, because it made TRUE items read as false

**Found by the round-1 blind audit, and it is a worse bug than the one the fix
pass was chartered to solve.**

Round 1 hit every salience target — gap 0.32 against a 0.4 limit, no cell above
3.5, all 40 FALSE items still findable. But false positives on the TRUE decoys
went from **0/40 to 4/40**, and all four objected to the same thing:

> "Pass rates cover every enrolled student, non-entrants counted as fails, and
> about a quarter of each cohort's roll was not entered"
> — *auditor: "25 percent scored as automatic fails, so the pass rate is partly
> fabricated"*

The auditor is right. And **all four were `coherent_true`**. A TRUE item that
reads as false depresses confidence on exactly the cell the consensuality
prediction says should be *inflated*, so it pushes AUC(coherent) down — **towards
the hypothesis**. The salience gap it replaced pushed the other way and was
therefore conservative. This one flatters. That makes it the more dangerous of
the two, and it is why round 2 happened even though round 1 had passed.

**Root cause.** The COMPLETENESS clause said units were missing *outcome data*,
which forced the WHOLE population clause to invent values for them.

**Chose:** the completeness clause now describes incomplete **participation in
the intervention** — missed sessions, partial uptake, intermittent use — never
missing measurements. Then nothing has to be imputed and all four combinations
stand on their own:

| combination | reading |
|---|---|
| SUBSET + INCOMPLETE | tally covers full participants only, and a quarter are not — **FALSE** |
| SUBSET + COMPLETE | the qualification excludes nobody — TRUE |
| WHOLE + INCOMPLETE | some took part patchily, outcome measured for all of them — TRUE |
| WHOLE + COMPLETE | TRUE |

**Enforced, not requested:** `scripts/assemble_passages.py` fails the build on
`counted as`, `entered as`, `recorded as`, `treated as`, `scored as`, `imputed`,
`substituted`, `as nil`, `as zero`, `as fails` anywhere in a passage.

**General lesson worth keeping:** the decoys earn their place. Without 40 TRUE
items in the audit there would have been no false-positive rate, and a bias
pointing *at* the hypothesis would have shipped looking like a clean result.

---

## D-028 — The model gate refuses; it does not warn

**Chose:** `src/model_gate.py` runs three pre-flight checks and raises
`ModelGateError` on any miss, before any item is scored.

1. The abstention option must be a single token **in the form the model would
   actually emit** (`" Unsure"`, not `"Unsure"`). Candidates are tried in the
   order `Unsure, Maybe, Unclear, Unknown` and every attempt is recorded with its
   token count, so the fallback is auditable rather than silent.
2. Yes and No must resolve to non-empty variant sets summed over casing and
   leading space.
3. `mass_covered` must exceed **0.5** on a probe sample. If a plain prompt fails,
   the gate retries with the chat template and, if that fixes it, tells you the
   model is instruction-tuned and to use `--chat-template`.

Measured on SmolLM2-135M: `' Unsure'` 3 tokens, `' Maybe'` 1, `' Unclear'` 2,
`' Unknown'` 1 → chose `Maybe`; Yes 4 ids, No 6 ids, Maybe 4 ids; coverage 0.7069.

**Why refuse rather than warn:** every one of these has already produced a
number in this project that looked publishable and was not. A warning is
something a tired person scrolls past at 2am.

**Reverse:** `--min-mass`, `--third-option` and `--n-probe` are all flags; the
gate is a module, so a caller can catch `ModelGateError` if it genuinely wants to
proceed.

---

## D-029 — Two surface-feature controls, off by default

**Chose:** `--shuffle-cases` and `--option-rotations` on `src.score`, both off by
default so the primary number is not silently a different quantity.

- `--shuffle-cases` permutes each item's four case sentences with a fixed
  per-item permutation (deterministic in `(item_id, seed)`) and records it on
  every record. Case order carries no evidence; a result that moves under it is
  measuring presentation.
- `--option-rotations` scores every item under all three rotations of the option
  list, reports the mean and the spread in P(yes), and flags spread > 0.05. The
  **headline record stays the first rotation**, so a run with the control on and
  one with it off agree on the primary number and the average is an addition
  rather than a silent replacement.

`prompts_hash` deliberately digests the **unpermuted** prompts, so it still
identifies the item text regardless of which controls were on.

---

## D-030 — The decorative control: surface complexity held apart from evidential independence

**The objection.** A diverse passage names four countries, four devices and four
months where a coherent one names one of each. More entities to track. A
confidence difference could come from parse load rather than from the evidence
being evidentially independent. This is the last form of the co-founder's
objection and nothing in the design closed it — the matched-mechanism subset
holds the flaw fixed, not the reading difficulty.

**Chose:** a third arm. 20 items — 10 true, 10 false — in the same 10
`scope_mismatch` families as the matched subset, so claim, scenario and
falsification mechanism are all held fixed. In them:

- every dimension that **bears on the claim** (region, season, population,
  treatment context) is **identical across all four cases**, exactly as in a
  coherent item;
- each case carries four **distinct decorations** — a log serial, a clock time
  inside the same day, a terminal, a desk — which rule nothing out and so confer
  no evidential independence whatsoever.

Surface busyness lands on the diverse cells; evidential structure lands on the
coherent cells. Measured:

| level | n | distinct tokens | distinct entities | condition values | words |
|---|---|---|---|---|---|
| coherent | 40 | 77.5 | 13.7 | 4.0 | 177.8 |
| diverse | 40 | 95.5 | 26.1 | 16.0 | 177.9 |
| **decorative** | 20 | **95.8** | **25.2** | **4.0** | 194.1 |

Decorative vs diverse: distinct tokens **+0.3%**, distinct entities **−3.3%**,
both inside the 10% target. Word count sits +7.2% from the grand mean, inside
10%. Condition variety equals coherent exactly and is a quarter of diverse.

**What it decides, and `analyze.py` says which in one sentence in section 1:**

| result | reading |
|---|---|
| decorative AUC tracks **coherent** | the effect is about evidential independence |
| decorative AUC tracks **diverse** | the effect is surface complexity and the story is wrong |
| decorative sits between | this run does not separate the two |

**`decorations` is a separate field from `conditions`, and that is the point.**
Conditions are dimensions whose variation confers evidential independence;
decorations are dimensions whose variation confers none. The model forbids a key
being both, and forbids any non-decorative cell carrying decorations — decorating
another cell would destroy the contrast. That check earned its place on the first
run: `fam_solder` has `operator` as a *condition*, held constant across its
coherent cases, so a varying `operator` decoration would have contradicted the
passage outright. Decorations are now clerical and never people.

**Gated** by `control_surface_match`, which fails if either half drifts: the
surface must stay within 10% of diverse **and** the condition variety must equal
coherent exactly and stay below diverse. Half a control is not a control.

**Also added:** `surface_complexity` on every item — token counts, distinct
tokens, type-token ratio, entity counts, condition and decoration variety. A pure
function of the passage, recomputed and checked at load exactly like
`word_count`, so it cannot drift. It is reported beside every AUC and enters the
nested conditioning models in `covariate_models`.

**Honest limit.** The decorative arm exists in 10 families, not 20, so its AUC
rests on 100 pairs against the primary endpoint's 400. It is a control, not a
second primary endpoint, and the report presents it that way.

**Reverse:** the arm is 20 items in the existing family files and is selected at
analysis time. Deleting the `decorative_*` items restores the previous set
exactly; every gate and statistic already handles their absence.

---

## D-031 — The word-balance gate is global, not per-family

**Why it changed.** Adding a 6-item family broke `passage_word_balance`, and the
failure was instructive: it flagged `'in'` at 3T/0F inside one family while that
word sat at **100T/97F globally** — a 1.5% split and no signal at all.

The gate is a proxy for `no_lexical_giveaway`, which trains on 16 families and
tests on 4. What such a classifier can exploit is a **global** association
between a word and the label. A word leaning TRUE in one family and FALSE in
another cancels and is invisible to it. Per-family balance is also arithmetically
unreachable here: when both FALSE items of a family share a mechanism the
falsifying clause is 2F against at most 1T (D-026), and 3F against at most 2T in
the ten families carrying the control arm. The gate was demanding the impossible
and then reporting noise.

**Chose:** flag a word only when all three hold —

| test | rule | rules out |
|---|---|---|
| absolute | \|T − F\| > 4 | high-frequency function words drifting on passage length |
| relative | \|T − F\| / (T + F) > 40% | words appearing once or twice |
| learnable | present in ≥ 3 families | words that cannot cross a grouped fold boundary in either direction |

Per-family worst is still computed and reported as data, because a family far
from the rest is worth a look by eye. It is no longer a pass/fail criterion.

**Reframed, it immediately found something real.** `'any'` at 14T/4F across 10
families and `'no'` at 17T/7F across 13 — because all ten scope families used the
same `"and no X in any Y"` frame for the COMPLETE clause, and the control arm
made COMPLETE lean TRUE. Fixed by giving each of the ten a **different**
construction; dispersing beats mirroring, which would have concentrated the lean
into a single word at 15T/0F. The pronoun `'it'` (7T/2F) was removed from the
four clauses carrying it.

Verified both ways: the gate fails on the pre-fix wording and passes on the
current items with **0 words flagged**, worst qualifying split 17%. The lexical
gate — still the authority — reads **48.4%**.

---

## D-032 - The reader audit: measuring what a reader answers, not what a hunter finds

**The number that forced this.** A 10-item human spot-check caught **1 of 5**
flaws and read one TRUE item as FALSE. The blind audit reports **94%** per
auditor over the 50 FALSE items. Those two cannot both describe the same
quantity.

They do not. The existing audit hands the auditor this instruction:

> For each one, answer the question and say whether you can identify a specific
> reason the evidence fails to establish the claim. Some of these items have
> nothing wrong with them.

That is a **hunter**. It announces that flaws exist, asks for one, and gives a
field to write it in. Near-ceiling performance is what it is built to produce.
Two consequences:

1. `salience` is rated *conditional on having found the flaw*. It measures how
   explicit a flaw looks to someone already looking at it, which is not how
   loud it is.
2. As a covariate the range is nearly empty - the whole 100-item set spans 2.0
   to 3.0 - so conditioning on it can barely move a coefficient.

**Chose:** a second audit, `src/audit_reader.py`, alongside the first rather
than replacing it. A reader there sees **exactly `render_prompt(item)`** - the
same instruction, claim, passage and three options the scored model gets - and
nothing else. No mention that a flaw might exist, no request for a description,
no explicitness field, because the scored model has nowhere to put one either.
The only thing recorded is which of Yes / No / Unsure it picks.

Three independent runs per item, regrouped each run, give a rate rather than a
bit:

| measure | definition |
|---|---|
| `reader_catch_rate` | fraction answering **No** on a FALSE item |
| `reader_false_positive_rate` | fraction answering **No** on a TRUE item |
| `abstention_rate` | fraction answering **Unsure** |

Blindness works as in the hunter audit and for the same reasons: opaque codes,
TRUE items mixed throughout, the key in a separate tree, and **no two items from
one family in a batch**. The batch builder also refuses to emit a file
containing a cell name or any of the hunter's cue words, so the reader cannot be
accidentally told to look.

**`reader_catch_rate` replaces `salience` as the covariate** in `analyze.py`.
Hunter salience stays in the output, labelled as what it is.

**Reverse:** additive. The hunter audit, its files and its salience column are
untouched; deleting `results/reader_audit*` restores the previous behaviour.

**Known staleness at the time of writing:** the four `fam_driptape` items were
reworded (below) after the baseline reader batches were generated, so their
baseline rates describe the previous wording. They are re-measured in the final
round; the baseline is used only for triage.

---

## D-033 - Duplicate flaw phrasing: the recurrence found was the control, not a defect

**The instruction.** Items 2, 6 and 10 of the blind review share a 21-word
identical falsifying sentence; scan all 50 FALSE items and rewrite so that no
two share flaw phrasing.

**What the scan found.** `src/flaws.py` isolates the falsifying span of each
FALSE item - first sentence of the mid-passage line for `stated_confound`,
second for `scope_mismatch`, the case lines for `broken_chronology` - and
compares all 1225 pairs on normalized edit distance and shared 8-grams:

| | pairs above 0.70 similarity | shared 8-grams |
|---|---|---|
| within a family | 30 | many |
| across families | **1** (0.705) | **0** |

Every one of the 30 within-family pairs is a scope family's
`coherent_false` / `diverse_false` / `decorative_false` trio. Items 2, 6 and 10
are exactly that trio for `fam_checkout`.

**That identity is the matched-mechanism endpoint (D-024).** Section 2 of the
analysis - the number quoted when someone asks whether the coherent-false items
are simply easier to catch - compares those cells inside the ten scope families
and holds by making the falsifying sentence *identical* so that nothing but
coherence differs. Varying the phrasing would put wording back into the one
endpoint constructed to have nothing in it but coherence.

**Chose:** enforce the rule **across** families, exempt it **within** a family,
and say so in the gate's own docstring. The single cross-family pair - two
agricultural families that had both landed on rainfall as their stated confound
- was real and is fixed: `fam_driptape` now uses a seed-variety confound, which
drops the worst cross-family similarity to well under the limit.

`no_duplicate_flaw_phrasing` fails the build on either a cross-family pair above
0.70 similarity or a single shared 8-gram.

**This is a deliberate deviation from the instruction as written** and the
report says so plainly, because the alternative reading - literally no two items
share phrasing - would silently cost the confound-controlled endpoint. If that
trade is wanted anyway, the exemption is one condition in
`check_duplicate_flaw_phrasing` and the gate is already written to fail without
it.

**Also changed:** `scripts/assemble_passages.py` cleared `salience` on all 80
items every run, so rebuilding one family threw away the audit for the other
nineteen. It now clears salience and the reader rates only on items whose
passage actually changed. Rewording `fam_driptape` invalidated 4 items instead
of 80.

---

## D-034 - The blocking clause becomes a leading sentence, in every cell that carries it

**The failure.** A TRUE item read as FALSE in a human spot-check, rated
difficulty 2.0 - confidently wrong, not hesitantly wrong. The reader audit then
put a number on it: `coherent_true` false-positive rate **0.22**, against 0.07
for `diverse_true`. Not a few bad items either - 13 of 20 had at least one
reader of three calling them false. A distributed pull, not an outlier.

**The construction that caused it.** The confound and the fact that disarms it
shared one sentence, with the disarming half subordinate:

> Rainfall in each plot's season ran a fifth higher than the year before, and
> every plot stood under cover, on a fixed watering schedule.

A reader meets the confound, forms the objection, and answers. The clause that
makes it harmless arrives after the objection has already formed.

**Chose:** state the protection first, positively, as its own sentence.

> Every plot stood under cover all season, watered only on a fixed schedule.
> Rainfall in each plot's season ran a fifth higher than the year before.

Twenty such sentences authored, one per family, checked for causal, hedging and
imputation language.

**Applied to every item carrying `block`, not only to `coherent_true`.** This is
the part worth arguing with. The brief says "rewrite every coherent_true item",
but the block clause appears in exactly two cells per family - one TRUE and one
FALSE (`coherent_true` and `diverse_false` in a confound family;
`coherent_false` and `diverse_true` in a scope family). Rewriting one cell would
have made the construction itself a perfect predictor of TRUE for those items -
precisely the lexical giveaway the build gates against. Rewriting all four keeps
the clause balanced across the split, and the word-count shift lands equally on
every cell for the same reason. 40 items changed, 2 per family.

**Result:**

| cell | before | after | target |
|---|---|---|---|
| `coherent_true` FP | 0.22 | **0.13** | <= 0.15 |
| `diverse_true` FP | 0.07 | **0.05** | |
| `decorative_true` FP | 0.10 | **0.00** | |
| gap coherent - diverse | 0.15 | **0.08** | <= 0.10 |

**Reverse:** one branch in `src/midline.py`. Deleting it and re-running the
assembler restores the previous wording exactly.

---

## D-035 - broken_chronology was the invisible mechanism

**Found by the reader audit, not by inspection.** All five FALSE items a plain
reader missed at baseline were `broken_chronology`, and the hunter audit had
rated every one of them "found". A date inversion inside a case line is exactly
what a reader skims past and a hunter, told to look, does not.

**Two fixes, neither adding vocabulary.**

*Round 1 - adjacency.* The treatment date and the measurement date sat ten words
apart, with the outcome between them:

> fitted 4 May, ran 340 hours longer before failure by 27 March than the same
> machine last cycle

They are now adjacent, so the comparison is one glance rather than a scan:

> fitted 4 May, measured by 27 March, ran 340 hours longer before failure than
> the same machine last cycle

The same tokens, repositioned. Applied to all four cells of the ten confound
families so the family stays internally consistent and the TRUE/FALSE balance
holds. Catch rate 0.78 -> 0.82, items below 0.5 went 5 -> 2.

*Round 2 - a second cue.* Still eight of ten chronology items sat below
ceiling, so a THIRD case is now inverted as well: three of the four cases carry
a measurement dated before the treatment instead of two. This is the "second cue
elsewhere in the passage" the brief allows, and it costs nothing lexically
because an inversion swaps two date tokens between positions rather than
introducing any. The answer key was updated to match - "three quarters of the
evidence", "that leaves case two alone".

**Not done:** no causal language was restored, and nothing moved to the final
sentence. Both were explicitly off the table and both would have worked.

---

## D-036 - whole_incomplete reads as a defect, and it cannot be softened

**What the measurement found.** After the D-034 rewrite the FALSE cells met every
target, and `coherent_true` did not: 0.25 against a limit of 0.15, with a
reader-clustered 95% interval of [0.06, 0.41].

**The cause, named exactly.** The hunter's own descriptions of its false
positives on TRUE items say the same thing every time, across all three TRUE
cells:

> "The scored population did not all receive the program: *Scores cover every
> pupil on each class roll, and a quarter of each roll missed at least one
> program session.*"

That is the `whole_incomplete` clause pair. The design labels it TRUE and is
right to: including partial compliers in the denominator dilutes toward the
null, so a positive result measured that way is conservative. Readers do not
read it that way - a quarter of the denominator untreated reads as a defect.

**Not coherence.** The strict readers are specific rather than severe. In the
four batches that rejected `coherent_true` at 1.00, 1.00, 0.60 and 1.00 they
rejected `diverse_true` at 0.12, 0.00, 0.00 and 0.00.

**The obvious fix was tried and the build rejected it.** Softening the shortfall
- one missed session instead of a quarter of them - was applied to all ten
confound families, where `whole_incomplete` is carried only by TRUE items and so
looked safe:

    'single' is 8T/0F across 8 families - a 100% split. A classifier trained on
    other families can carry that straight across a fold boundary.

`passage_word_balance` was correct to fail it. The information *the shortfall
here is minor* exists only in TRUE items, so any wording carrying it is a
giveaway whatever words it uses. In the ten scope families it is worse: that
same clause string is the `subset_incomplete` flaw in 30 FALSE items, so
softening would blunt what the FALSE cells rest on.

Reverted. `results/scope_clauses.json` is back to its round-3 state and every
passage is byte-identical to the commit before the experiment.

**So this is a tension in the balanced-clause design (D-026), not a wording
miss:** one clause must read as innocuous in a TRUE item and as falsifying in a
FALSE one, and readers do not split it where the design does. Three ways out,
none of them a wording change, all of them the reviewer's call:

1. **Accept and report.** Carry the `coherent_true` false-positive rate as a
   known property. It is already the covariate the analysis conditions on.
2. **Rebalance.** Drop `whole_incomplete` from the TRUE cells and reassign from
   scratch - D-026 showed the current assignment is the unique balanced one, so
   this means changing what the four cells are.
3. **Re-label.** If a careful reader reliably holds that a quarter
   non-compliance means the evidence does not establish the claim, then the
   ground-truth label is the thing that is wrong.

**Stopped here** rather than editing further. Three of the five permitted rounds
were spent on items; the fourth was spent on the experiment above, which argued
for stopping. Tuning wording against a measurement whose own interval is
[0.06, 0.41] is chasing a number, which the brief explicitly forbids.

---

## D-037 - The reader audit is clustered, and the interval has to admit it

**One agent answers every item in its batch**, so a batch is a cluster and the
verdicts inside it are not independent. Treating 300 reads as 300 observations
gives an interval several times too narrow.

**How this surfaced.** Between two rounds, with exactly one `coherent_true`
passage changed, that cell moved from **0.00 to 0.25**. The 15 dissenting reads
were perfectly concentrated in 4 of 18 batches - 4/4, 4/4, 3/5, 4/4 - and zero
in the other fourteen. Nothing about the items explains that. A few readers
reject the cell wholesale and the rest reject none of it.

**Chose:** `_clusters()` in `src/audit_reader.py` reports, per cell, the mean
over clusters, the naive mean over reads, a 95% interval from bootstrapping
CLUSTERS, and how many clusters sat at zero. Every cell figure in
`READER_REPORT.md` carries that interval.

**Consequence for reading any single round:** a cell mean is a draw from a
bimodal process with an effective n of 18, not 300. The round that reported
`coherent_true` at 0.00 was a lucky draw, and reporting it as the result would
have been cherry-picking. The report leads with the final measurement and shows
the interval.

**What would actually narrow it:** more clusters, not more reads - smaller
batches, or one item per reader. Twelve batches per run instead of six would
double the cluster count for the same token cost. Not done here; recorded as the
next thing to do if the TRUE-cell number needs to be resolved rather than
reported.

## D-038 - Human blind review #2: 10/10, all five flaws named, zero TRUE read as FALSE

**The measurement.** The reviewer scored `review/blind_items_2.md` blind: 10/10
correct against `results/review_key/blind_items_2.json`, all five FALSE flaws
identified with the mechanism the key names (population restriction, direction
of selection, or the class-size confound - not generic suspicion), and 0/5
false positives on TRUE items, including 0/3 on `coherent_true`. Review #1 on
the pre-fix items was 5/10 with 1 of 5 flaws caught. The always-yes baseline on
this balanced set is 5/10.

**Caveat that limits what #2 proves.** Between the reviews the reviewer read
`READER_REPORT.md` and D-036, so they knew the mid-passage line carries the
load and knew the intent-to-treat reading of `whole_incomplete`. Their item-9
reasoning ("dilutes rather than fakes") is that briefing applied. The
before/after on a CONSTANT reader is the model reader audit (D-032); the human
number corroborates it but cannot independently establish it.

**What it settles about the coherent_true 0.25 (D-036).** The items are not
mislabeled: a careful reader accepts all three contested items and can
articulate why. The model readers' dissent also fails a tracking test: the one
TRUE item with a genuinely weak defuse (below) was passed 3/3 by readers, while
the three airtight `coherent_true` items each drew exactly one No. The 0.25 is
a property of a minority reader phenotype triggered by the coherent-TRUE
gestalt, not of the items - but the scored model is drawn from the same
population as the audit readers, so the coherent/diverse TRUE asymmetry stays
in the report as a measured property (D-036 option 1), not noise.

**New defect found, not fixed (no item edits authorized).** The
`fam_pricing` block clause - "Every line was stocked from a fixed year-round
range" - fixes the assortment but not the sales/shelf mix within it, so it does
not deductively block the confound ("premium stock's share of each store's
shelf hit a record, double the earlier period") the way the other block
sentences do (paid plan carries no trial period; hired machines carry their own
tooling; independent roll the scheme never covered). Two independent careful
readers converged on this: the human (rated it difficulty 4, the only 4 in the
set) and one of two hunter reads on `fam_pricing__diverse_true` (round 0:
"a fixed range of SKUs does not fix the sales/shelf mix within it"). The fix,
when authorized, is a block that closes the causal channel categorically, e.g.
lines priced and picked from a fixed planogram slot - it must appear in both
cells that carry the block clause for this family, per D-034.

## D-039 - The fam_pricing block made categorical, and the decorative builder made honest

**Authorized fix for the D-038 defect.** The family's block clause becomes
"Every line was priced and picked from a fixed planogram slot of its own,
which no shelf reallocation touched" (`results/family_spec.json`). Like the
other block sentences - a paid plan carries no trial period, a hired machine
carries its own tooling - it closes the channel categorically: a slot the
reallocation never touched cannot transmit a shelf-share change to the line's
orders, whereas the old "fixed year-round range" fixed the assortment but not
the sales mix within it.

**Four cells, not two.** The instruction said both cells carrying the block;
in this family that is four - `coherent_false`, `diverse_true`, and both
decorative items, because the control arm builds every item on `same_block`.
The wording changed in all four (2 TRUE / 2 FALSE, so the global word balance
and the lexical gate are untouched by construction), for D-034's reason:
rewriting a subset would make the sentence itself a truth predictor.

**A builder bug found on the way.** `scripts/build_decorative.py` purged and
rebuilt all 20 control items and dropped `salience` and the reader rates on
every one - including the 18 whose passages came back byte-identical. Same
defect class assemble_passages.py already fixed: invalidation wider than the
change. It now keeps measured fields aside and carries them over exactly when
the rebuilt passage is unchanged.

**Re-measurement was targeted, not global.** A full 18-batch reader audit
re-run would have overwritten 96 stable measurements with fresh sampling noise
to refresh 4 items. `scripts/reader_audit_patch.py` instead builds patch
batches of the same size and blindness as the main audit - ONE target plus 16
fillers drawn one-per-family, fresh opaque codes - runs three independent
readers per target, discards the filler reads, and merges only the targets
into `results/reader_audit.json` (patch clusters `p{run}b{batch}`, disjoint
from the main audit's). The hunter numbers for these four items predate the
edit and are labeled by their round as always; the flaw spans (the scope
pair) are byte-identical, so the hunter catch evidence is about text that did
not change.

**The reads that exist are the reads that count.** Two protocol notes, both
in the direction of using more data rather than choosing among draws. (1) The
first 3-read draw on `fam_pricing__coherent_false` came back 1 No / 2 Unsure -
but both Unsure verdicts were from readers who abstained on 5 of their 17
batch items, a phenotype the main audit's 18 readers never showed. Three reads
cannot separate that reader draw from an item property, so the item got three
MORE readers (runs 3-5), not a re-roll. (2) A workflow-arguments bug re-ran
the original 12 batches instead of the 3 new ones, overwriting the first
draw's verdict files with fresh reads. Every read was kept: the redraw was
relabeled runs 6-8, the first draw's target verdicts were reconstructed into
runs 0-2 from the already-merged audit (marked `reconstructed` in the files),
and nothing was selected on its outcome.

**Measured result, 9 reads on the contested item and 6 on the rest:**

    fam_pricing__coherent_false   0 Yes / 6 No / 3 Unsure  -> catch 0.67
    fam_pricing__decorative_false 0 Yes / 6 No             -> catch 1.00
    fam_pricing__diverse_true     6 Yes                    -> fp 0.00
    fam_pricing__decorative_true  6 Yes                    -> fp 0.00

The two TRUE cells read clean under the categorical block. `coherent_false`
dropped from its pre-fix 1.00 to 0.67 - equal to the set-wide worst-item
floor, still over the 0.5 target, with not one reader fooled (0 Yes). The old
1.00 was partly the LEAKY block doing illegitimate work: a reader who
(correctly) distrusted "fixed range" answered No for the wrong reason, and
that No counted as a catch of a flaw it never saw. The fix traded a
false-positive catch channel for honest abstention; the scope flaw alone now
carries the item.
