"""Prove the chat-template prefill fix on a real tokenizer before any GPU run.

The D-048 bug: with --chat-template, the prompt's trailing "Answer:" sat
inside the USER turn, so the model's first assistant token was it starting to
write "Answer" itself - top token 'Answer', mass on the options ~0. The fix
prefills "Answer:" after the assistant tag. This script loads a small model of
the SAME family as the target run, scores real items, and refuses to pass
unless every item has mass_covered above the floor AND an argmax inside the
option variant set.

On failure it prints the rendered prompt tail and the top-10 next tokens with
probabilities, so the next failure is self-explanatory instead of a mystery.

    python scripts/verify_prefill.py --model Qwen/Qwen3-0.6B --n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import load_items  # noqa: E402
from src.render import render_prompt  # noqa: E402
from src.score import HFScorer  # noqa: E402

OPTS = ("Yes", "No", "Unknown")


def diagnose(scorer: HFScorer, prompt: str) -> None:
    import torch

    text = scorer._prepare(prompt)
    print(f"  PROMPT TAIL (last 200 chars): {text[-200:]!r}")
    enc = scorer.tokenizer(text, return_tensors="pt").to(scorer.model.device)
    with torch.no_grad():
        logits = scorer.model(**enc).logits[0, -1, :].float()
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, 10)
    print("  TOP 10 NEXT TOKENS:")
    for p, i in zip(top.values, top.indices):
        print(f"    {p.item():.4f}  {scorer.tokenizer.decode([int(i)])!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--mass-floor", type=float, default=0.5)
    ap.add_argument("--no-chat-template", dest="chat_template",
                    action="store_false")
    ap.set_defaults(chat_template=True)
    args = ap.parse_args(argv)

    items = load_items(args.items)[: args.n]
    scorer = HFScorer(
        args.model,
        dtype=args.dtype,
        device_map=args.device_map,
        chat_template=args.chat_template,
        no_thinking=True,
        options=OPTS,
    )
    allowed = {
        v for opt in scorer.option_tokens.values()
        for v in opt.id_to_variant.values()
    }
    print(f"prefill={scorer.ANSWER_PREFILL!r} chat_template={args.chat_template}")

    failures = 0
    for it in items:
        prompt = render_prompt(it, OPTS)
        res = scorer.score_prompt(prompt)
        ok = res.mass_covered > args.mass_floor and res.top_token in allowed
        mark = "ok " if ok else "FAIL"
        print(f"{mark} {it.id}: p_yes={res.p_yes_3way:.3f} "
              f"mass={res.mass_covered:.3f} top={res.top_token!r}")
        if not ok:
            failures += 1
            diagnose(scorer, prompt)

    if failures:
        print(f"\n{failures}/{len(items)} items FAILED the "
              f"mass>{args.mass_floor} / argmax-in-options check")
        return 1
    print(f"\nPASS: all {len(items)} items covered above {args.mass_floor} "
          "with an option-token argmax")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
