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

**Reverse:** `src/render.py` has `OPTION_WORDS` and the instruction text in one
place. Swapping in a sequence-scored multi-token option means changing
`score.py::score_item` only.

---

## D-003 — Abstentions are INCLUDED in AUC

**Chose:** items where `Unsure` is the argmax are included in both AUCs using
their `p_yes_3way` value. They are never dropped, never imputed to 0.5.

**Why:** dropping them makes abstention a free way to inflate AUC — a model that
abstains on exactly the items it would have gotten wrong looks like a better
discriminator than one that guesses. The abstention rate is reported *separately
per cell* so it can be read alongside AUC rather than laundered into it.

**Reverse:** `analyze.py::compute_auc` takes the score vector; an
`--auc-drop-abstained` variant would be a filter at the call site. Deliberately
not implemented, so that using it is a visible edit rather than a flag.

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

**Reverse:** `--chat-template` flag, already implemented; the run JSON records
which was used so runs can never be silently pooled.

---

## D-010 — Bootstrap resampling unit

**Ambiguous.** Resample items, or resample families?

**Chose:** resample **items within cell** (stratified: each cell resampled to its
own n) for cell means and abstention rates, and resample **pairs' underlying
items** for AUC — i.e. resample the 20 true and 20 false scores independently and
recompute the pairwise win rate.

**Why:** it is the standard, and it matches the unit the brief names ("20 true vs
20 false = 400 pairs"). Noted honestly: because items come in families of 4, items
within a family are not independent, so these CIs are **anti-conservative**
(too narrow). A family-level cluster bootstrap is also implemented and reported
alongside as `ci_clustered`, so both are visible.

**Reverse:** `--bootstrap-unit {item,family}` flag; both are computed and
reported by default.

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

## D-018 — Torch install is CPU-only

**Chose:** `torch` from the PyTorch CPU wheel index.

**Why:** no GPU assumption should be baked into an item-authoring repo, the smoke
model is 135M, and CPU wheels are ~200 MB vs multi-GB for CUDA.

**Reverse:** reinstall torch from the CUDA index; `score.py` already does
`--device auto` and will pick up `cuda` if present.
