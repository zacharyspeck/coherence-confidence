# Running the experiment on Kaggle (2x T4)

`kaggle_run.ipynb` runs the full ladder — two models, five scored stages each,
analysis, zip — from a single **Run all**.

## Notebook settings

| setting | value |
|---|---|
| Settings → Accelerator | **GPU T4 x2** (both GPUs are required) |
| Settings → Internet | **ON** |
| Add-ons → Secrets | `GITHUB_TOKEN` (below), attached to this notebook |
| Persistence | not needed; everything relevant lands in `/kaggle/working` |

## The `GITHUB_TOKEN` secret

The repo is private, so the clone in cell 1 reads a GitHub token from Kaggle
Secrets.

1. GitHub → **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token.**
2. Resource owner: your account. Expiration: your call (30 days is plenty).
3. Repository access: **Only select repositories** → `coherence-confidence`.
4. Permissions → Repository permissions → **Contents: Read-only**. Grant
   nothing else.
5. Generate, copy the token.
6. In the Kaggle notebook editor: **Add-ons → Secrets → Add secret.** Label
   exactly `GITHUB_TOKEN`, value = the token. Make sure the checkbox attaching
   it to this notebook is on.

The notebook never prints the token and scrubs it from the cloned repo's git
config immediately after cloning, so it cannot leak through the output zip.

## The models — and why they are not the ones first asked for

The brief asked for a Qwen3.x 8B / 27B instruct ladder. Checked against
Hugging Face (2026-09): the 27B models in that line — `Qwen/Qwen3.6-27B` and
`Qwen/Qwen3.8-27B` — are **multimodal** (`AutoModelForMultimodalLM`, ~54 GB in
bf16) and have **no 8B text sibling**. They fit neither `src/score.py`'s
causal-LM measurement path nor this machine's disk. The nearest same-family
text-only ladder, which is what the size comparison needs, is:

| tag | HF repo | precision | why |
|---|---|---|---|
| `qwen3_8b` | `Qwen/Qwen3-8B` | fp16, sharded across both T4s | ~16 GB of weights; does not fit one 15 GB T4 |
| `qwen3_32b` | `unsloth/Qwen3-32B-bnb-4bit` | pre-quantized nf4 (~19.5 GB) | the official bf16 checkpoint is ~65 GB, which exceeds Kaggle's disk — and bnb quantize-at-load would still have to download all of it. The unsloth repo is the standard bitsandbytes 4-bit export of the same weights |

Two hardware notes baked into the notebook flags:

- **fp16, not bf16.** T4s are sm_75 — they have no bf16 units. `--dtype
  float16` everywhere. For the 32B this takes an extra step the flags alone
  cannot do: the unsloth repo bakes `bnb_4bit_compute_dtype: bfloat16` into
  its config, and on the pinned transformers a user-passed
  `BitsAndBytesConfig` is ignored for pre-quantized checkpoints — so
  `score.py` rewrites the checkpoint's own config at load to compute in
  fp16, prints a NOTE when it does, and records it in run meta as
  `bnb_compute_dtype_overridden`. Without that, every 4-bit matmul would run
  in emulated bf16: slower, and a numeric confound against the 8B's fp16
  path in the exact size comparison this notebook exists to make.
- **`--no-thinking` is load-bearing.** Qwen3 models are hybrid-thinking: with
  the default template the next token after the prompt is `<think>`, so every
  answer option reads near zero and the run looks like a model with no
  opinion. `--no-thinking` passes `enable_thinking=False` to the chat
  template. If it were ever dropped, the coverage gate (below) fails the run
  loudly rather than letting the numbers through.

## What runs, per model

1. **Tokenization gate** with `--third-option Unknown` — verifies ` Yes` /
   ` No` / ` Unknown` are each one token with disjoint id sets. `Unknown`
   replaces `Unsure` because bare `Unsure` is two tokens on the Qwen
   vocabulary, which under-counted abstention in the CPU pilot (D-043).
2. **Score** — all 100 items, final-position logits, checkpointed per item.
   Then a hard **coverage gate: mean `mass_covered` ≥ 0.5 or the notebook
   raises.** Below that, the renormalized probabilities are ratios of
   rounding errors and nothing downstream is interpretable.
3. **Baseline** — each family's claim with no evidence attached (prior
   subtraction).
4. **Controls** — `--shuffle-cases` (does presentation order move answers?)
   and `--option-rotations` (does Yes/No/Unknown ordering?). Each is a full
   separate run comparable to stage 2. These were the two controls the CPU
   run could not finish.
5. **Analysis** — AUC(coherent) vs AUC(diverse) with family-clustered
   bootstrap CIs, the matched-mechanism subset, the decorative-arm verdict
   sentence, per-mechanism breakdown, covariates, diagnostics. The headline
   section of each `analysis_<tag>.md` is printed at the end of cell 4.

Every scored stage writes a `results/partial_*.jsonl` checkpoint after each
item, so if the kernel dies mid-run, **Run all again**: completed items are
skipped and the run continues from where it stopped. Two honest caveats on
resume. Each finished stage still reloads its model once (~1–5 min) before
confirming it has nothing left to score, so resuming after a death deep in
the 32B block costs tens of minutes of reloads, not seconds. And every
checkpoint carries a stamp of the flags it was written under — resuming
under *different* measurement flags is refused loudly rather than allowed to
produce a mixed-provenance run file (delete the checkpoint to rescore).

The zip also stays honest: cell 1 moves the repo's committed `results/`
history to `results_from_repo/`, so `results/` — and the download — contain
only what this session produced.

## Expected runtime

| phase | estimate |
|---|---|
| installs + clone | ~3 min |
| model downloads (~16 GB + ~19.5 GB) | 10–25 min |
| qwen3_8b: gate + 100 + 20 + 100 + 300 + analysis | ~25–40 min |
| qwen3_32b (4-bit): same stages | ~45–90 min |
| **total** | **~1.5–2.5 h** — well inside a Kaggle session |

The 32B is slower per token than its size alone suggests: bitsandbytes
dequantizes per matmul and the layers are split across two GPUs, so activations
hop the PCIe bus every pass.

## Output

`/kaggle/working/coherence_confidence_results.zip` — the entire `results/`
tree (runs, baselines, controls, checkpoints, analyses). Download from the
notebook viewer's **Output** tab. Compare `results/analysis_qwen3_8b.md`
against `results/analysis_qwen3_32b.md`, section 1 first; the CPU-run numbers
they extend are in `RESULTS.md` (Qwen2.5-3B: gap +0.1075 [−0.0375, +0.2475],
opposite sign to the D-025 prediction, decorative arm tracking diverse).

## Troubleshooting

- **`clone failed`** — the secret is missing, unattached, expired, or lacks
  Contents: Read access to the repo.
- **`COVERAGE ... BELOW 0.5`** — the model is putting its next-token mass
  somewhere other than the three options. First suspect: thinking mode
  (is `--no-thinking` still in `COMMON`?). Second: the option table printed
  by the gate (are there unexpected multi-token forms?).
- **CUDA OOM on the 8B** — another process is holding GPU memory; Restart &
  Run all. The 8B needs both T4s nearly empty.
- **`expected the 8B to shard across both T4s`** — accelerator is set to a
  single-GPU option; switch to GPU T4 x2.
