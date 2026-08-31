# coherence-confidence

**Research question:** Does a language model's confidence track how much the
evidence in its context *agrees with itself* (coherence) rather than whether
that evidence actually *establishes the claim* (truth)?

If a model is a good reasoner, its confidence should be driven by the truth
dimension and be roughly invariant to the coherence dimension. If instead it is
running something closer to a fluency/consistency heuristic, confidence will
rise when the four observed cases share every irrelevant surface condition
(same region, same month, same device, same unit type) even when a confound in
the passage fully explains those cases.

## Design

A 2x2 between-items design. Each item is a short passage containing **4 observed
cases**, a **claim**, and the forced question:

> Does this evidence establish this claim? Yes / No / Unsure

| | TRUE (claim holds, no alternative explanation) | FALSE (claim does not hold) |
|---|---|---|
| **COHERENT** (all 4 cases share every irrelevant condition — 1 distinct value per dimension) | `coherent_true` | `coherent_false` — a single confound in the passage explains all 4 cases at once |
| **DIVERSE** (all 4 cases differ on every irrelevant condition — 4 distinct values per dimension) | `diverse_true` | `diverse_false` — a single confound cannot cover diverse cases, so it breaks differently: either the dates don't work (effect precedes cause in >=2 cases) or the cases don't show what the claim says |

20 items per cell, **80 items total**, organized as **20 families x 4 cells**.
Every family holds the claim and the underlying scenario fixed and varies only
coherence and truth, so the 2x2 contrast is within-family.

The key asymmetry to keep in mind while reading items: `coherent_false` and
`diverse_false` are both false, but they are false *for different structural
reasons*. That is forced by the design — a single shared confound is only
available when the cases share conditions. This is documented as a known
confound-of-the-confound in `DECISIONS.md` (D-004).

## Measurement

`src/score.py` renders the prompt ending in `Answer:` and reads the **logits at
the final position** — no generation, no sampling. It extracts probability mass
for the three options and reports:

- `p_yes_3way = p_yes / (p_yes + p_no + p_unsure)` — the primary measure
- `p_yes_2way = p_yes / (p_yes + p_no)` — reported separately
- `abstained = argmax(p_yes, p_no, p_unsure) == unsure`

Two things that are easy to get wrong and are handled explicitly:

1. **Tokenization.** `"Yes"`, `" Yes"`, `"yes"`, `" YES"` are all different token
   IDs. Probability mass is summed over the full casing x leading-space variant
   set for each option. `tests/test_tokenization.py` asserts the variant sets are
   non-empty for the tokenizer in use and fails loudly otherwise.
2. **Multi-token options.** "Not enough evidence" is multi-token, so it cannot be
   read from a single next-token distribution. The prompt presents the third
   option as the single-token stand-in **"Unsure"**. See `DECISIONS.md` D-002.

## Analysis

`src/analyze.py` reports, with bootstrap 95% CIs (10,000 resamples) on every
number:

- mean `p_yes_3way` and `p_yes_2way` per cell
- **AUC within coherent** (20 true vs 20 false = 400 pairs) and **AUC within
  diverse**, separately — never pooled, because pooling would let a coherence
  main effect masquerade as discrimination
- abstention rate per cell
- a two-way ANOVA-style breakdown: main effect of coherence, main effect of
  truth, and the interaction

AUC is implemented as an **explicit pairwise win rate** (ties = 0.5) so the
number is auditable by hand, and `tests/test_auc.py` asserts it matches
`sklearn.metrics.roc_auc_score` to 1e-9.

**Abstention policy (explicit, and load-bearing):** abstained items are
**INCLUDED** in the AUC using their `p_yes_3way`. They are never dropped.
Dropping abstentions would let a model inflate its AUC by abstaining on exactly
the items it finds hard. See `DECISIONS.md` D-003.

## Item validation

`src/validate.py` fails the build if any of these hold:

- any cell's mean word count is more than 10% from the grand mean
- any coherent item has >1 distinct value on any condition dimension
- any diverse item has <4 distinct values on any dimension
- a logistic regression on unigrams+bigrams predicts true vs false above 60%
  (cross-validated) — that means a lexical giveaway exists and the items are
  broken. The top predictive tokens are printed so they can be fixed.

## Layout

```
items/
  schema.json         JSON Schema for a single item
  seed/               the 2 hand-written seed families (8 items) - the template
  draft/              generated drafts, review_status="unreviewed"
  final/              EMPTY. Nothing is finalized by the build; that is a human step.
src/
  models.py           Pydantic models + loaders
  render.py           passage/prompt rendering (single source of truth)
  score.py            the measurement (HF causal LM, final-position logits)
  mock_scorer.py      deterministic mock scorer for plumbing tests
  analyze.py          cells, AUC, abstention, bootstrap CIs, 2x2 breakdown
  validate.py         item-set validation gates
  baseline.py         every claim with NO cases attached (prior subtraction)
results/              run outputs (JSON + markdown)
tests/                pytest
```

## Quickstart

```bash
# Windows (this repo was built on Windows / Git Bash)
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 1. plumbing check, no model, no real items
.venv/Scripts/python.exe -m src.smoke

# 2. validate the item set
.venv/Scripts/python.exe -m src.validate --items items/draft items/seed

# 3. score with a real model
.venv/Scripts/python.exe -m src.score --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --items items/draft items/seed --out results/run_smollm135m.json

# 4. analyze
.venv/Scripts/python.exe -m src.analyze --run results/run_smollm135m.json \
    --out results/analysis_smollm135m.json

# 5. baseline (claims with no evidence)
.venv/Scripts/python.exe -m src.baseline --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --items items/draft items/seed --out results/baseline_smollm135m.json

# tests
.venv/Scripts/python.exe -m pytest tests/ -q
```

Every design call made without asking is logged in `DECISIONS.md`.
Status at handoff is in `MORNING_REPORT.md`.
