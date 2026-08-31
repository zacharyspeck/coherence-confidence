# Step 7 — real scorer path, verified end to end

**These numbers are not a result.** They were produced on the *synthetic* items
from `src/synth.py`, whose passages are randomly assembled and carry no real
truth or coherence content. The only claim made here is that the real scorer path
runs, and that its two safety gates fire when they should.

Model: `HuggingFaceTB/SmolLM2-135M-Instruct` and `HuggingFaceTB/SmolLM2-135M`
(base). 135M parameters, ~270 MB, seconds to download — per D-006, deliberately
the smallest thing that exercises the same code path as any causal LM. No large
model was downloaded.

## What ran

| # | config | items | mean `mass_covered` | outcome |
|---|---|---|---|---|
| 1 | Instruct, default third option `Unsure` | — | — | **exit 2**, tokenization gate |
| 2 | Instruct, `--third-option Unknown`, plain prompt | 80 | **0.0082** | **exit 1**, coverage gate |
| 3 | Instruct, `--third-option Unknown`, `--chat-template` | 24 | **0.8475** | ok |
| 4 | Base, `--third-option Unknown`, plain prompt | 24 | **0.7104** | ok |
| 5 | Base, `--third-option Unknown`, plain prompt | 80 | **0.7123** | ok, analyzed |

Both gates fired on real weights before any number was reported. That is the
whole point of them.

## Gate 1 — the tokenization gate (D-019)

```
TOKENIZATION CHECK FAILED

  canonical option form(s) are not single tokens on this tokenizer:
  ' Unsure' is 3 tokens. ... Single-token alternatives on this
  tokenizer: ['Unknown', 'Maybe', 'Neither'].
```

Full option table for this tokenizer:

```
role     canonical    n_tok   ids  variants
yes      ' Yes'           1 OK     4  [' Yes', ' yes', 'Yes', 'yes']
no       ' No'            1 OK     6  [' NO', ' No', ' no', 'NO', 'No', 'no']
unsure   ' Unsure'        3 BAD    1  [' unsure']
unsure   ' Unknown'       1 OK     4  [' Unknown', ' unknown', 'Unknown', 'unknown']
```

Two things worth carrying into the morning:

- The **variant sum is not cosmetic.** `No` resolves to **six** token ids on this
  vocabulary and `Yes` to four. Reading a single id would have discarded most of
  the No mass and biased P(yes) upward by a model-specific amount.
- `' Unsure'` being 3 tokens is not a rounding issue. Its mass would have read
  near zero, the abstention rate would have read near zero, and `p_yes_3way`
  would have quietly collapsed onto the two-way number.

## Gate 2 — the coverage gate, and what it caught

Run 2 completed all 80 forward passes and then refused to report:

```
FAIL: mean mass_covered 0.00824 < 0.01. The three options hold almost none of
the next-token distribution; p_yes_3way is not interpretable.
```

The diagnosis is in the records: **the argmax token was `<|im_end|>` on all 80
prompts, at mean probability 0.886.** SmolLM2-*Instruct* is chat-tuned, so a
plain completion-style prompt ending in `Answer:` is off-distribution and the
model's overwhelming preference is to end the turn.

The renormalized numbers were still *computable* — `p_yes_3way` averaged 0.53 and
the argmax role split 56 No / 24 Yes — and they would have looked perfectly
publishable. They were ratios of masses under 1% of the distribution.

**Both fixes work:**
- applying the chat template raises coverage from 0.008 to **0.847**
- using the **base** model with the plain prompt gives **0.710**

## Consequence for D-009

The repo's default is a plain completion-style prompt with no chat template. That
default is right for **base** models and wrong for **instruct** models, and the
gap is a factor of ~100 in option coverage. D-009 has been updated with these
numbers. `--chat-template` exists and is recorded per run; the template hash
differs between the two, so templated and untemplated runs can never be pooled.

## Run 5, for completeness

Full 80 synthetic items, base model, plain prompt:

```
mean mass_covered = 0.7123   min = 0.6609
AUC[coherent] = 0.6150  (400 pairs)
AUC[diverse]  = 0.4675  (400 pairs)
main_effect_coherence = -0.0281
main_effect_truth     = +0.0039
interaction           = +0.0085
```

`main_effect_truth = +0.004` is the sanity check that matters: the synthetic
items have **no real truth content**, their TRUE/FALSE labels are arbitrary with
respect to the text, and the pipeline correctly reports approximately nothing.
A large truth effect here would have meant a leak.

## Reproduce

```bash
cd coherence-confidence
.venv/Scripts/python.exe -m src.smoke --keep --quiet

# see the option table without downloading weights
.venv/Scripts/python.exe -m src.score --model <MODEL> --check-tokenization-only

# base model
.venv/Scripts/python.exe -m src.score --model HuggingFaceTB/SmolLM2-135M \
    --third-option Unknown --items results/smoke/items \
    --out results/smoke/run_base135m_full.json

# instruct model
.venv/Scripts/python.exe -m src.score --model HuggingFaceTB/SmolLM2-135M-Instruct \
    --third-option Unknown --chat-template --items results/smoke/items \
    --out results/smoke/run_instruct135m.json
```
