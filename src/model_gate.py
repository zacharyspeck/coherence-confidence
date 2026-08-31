"""Pre-flight gate. Refuses to run on any miss (fix-pass step 7).

Every check here has already cost this project a wrong number once, so none of
them is precautionary:

1. **The abstention option must be a single token in the form the model would
   emit.** The prompt ends in `Answer:`, so the next token *is* `" Unsure"`. On
   SmolLM2 that string is three tokens, which makes the third option read near
   zero and abstention look impossible. The gate tries `Unsure`, then `Maybe`,
   `Unclear`, `Unknown`, and logs which one it settled on.
2. **Yes and No must resolve to non-empty variant sets**, summed over casing and
   leading space. On SmolLM2 `No` resolves to six token ids and `Yes` to four;
   reading a single id would discard most of the No mass.
3. **`mass_covered` must exceed 0.5 on a probe sample.** A chat-tuned model given
   a plain completion prompt put 88.6% of its next-token mass on `<|im_end|>` and
   0.8% on the three options - and still produced renormalized numbers that
   looked publishable. If the plain prompt fails, the gate retries with the chat
   template and tells you to use it.

    python -m src.model_gate --model <NAME>
    python -m src.model_gate --model <NAME> --chat-template
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from . import provenance
from .models import Item, load_items
from .render import DEFAULT_OPTIONS, render_prompt

#: Tried in order. The first whose " <Word>" form is a single token wins.
THIRD_OPTION_ORDER: tuple[str, ...] = ("Unsure", "Maybe", "Unclear", "Unknown")

MIN_MASS_COVERED = 0.5
PROBE_ITEMS = 6


class ModelGateError(RuntimeError):
    """Raised when a model cannot be measured honestly. Never downgraded."""


@dataclass
class GateReport:
    model: str
    passed: bool = False
    options: list[str] = field(default_factory=list)
    third_option_tried: list[dict[str, Any]] = field(default_factory=list)
    third_option_chosen: str | None = None
    third_option_was_default: bool = True
    option_token_ids: dict[str, Any] = field(default_factory=dict)
    chat_template: bool = False
    chat_template_required: bool = False
    mass_covered_mean: float | None = None
    mass_covered_min: float | None = None
    min_mass_required: float = MIN_MASS_COVERED
    n_probe_items: int = 0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_third_option(
    tokenizer, preferred: str | None = None, order: Sequence[str] = THIRD_OPTION_ORDER
) -> tuple[str | None, list[dict[str, Any]]]:
    """First candidate whose canonical `" <Word>"` form is one token.

    Returns (chosen, tried) where `tried` records every candidate and its token
    count, so the choice is auditable rather than a silent fallback.
    """
    from .score import n_tokens

    candidates = list(order)
    if preferred and preferred in candidates:
        candidates.remove(preferred)
        candidates.insert(0, preferred)
    elif preferred:
        candidates.insert(0, preferred)

    tried: list[dict[str, Any]] = []
    chosen: str | None = None
    for word in candidates:
        n = n_tokens(tokenizer, " " + word)
        tried.append({"word": word, "canonical": " " + word, "n_tokens": n})
        if n == 1 and chosen is None:
            chosen = word
    return chosen, tried


def run_gate(
    model_name: str,
    *,
    third_option: str | None = None,
    chat_template: bool = False,
    device: str = "auto",
    dtype: str = "float32",
    variant_strategy: str = "explicit",
    seed: int = 0,
    probe_items: Sequence[Item] | None = None,
    n_probe: int = PROBE_ITEMS,
    min_mass: float = MIN_MASS_COVERED,
    auto_chat_template: bool = True,
) -> tuple[GateReport, Any]:
    """Run every pre-flight check. Returns (report, scorer) or raises.

    The scorer is returned so the caller does not load the weights twice.
    """
    from transformers import AutoTokenizer

    from .score import HFScorer, build_option_table

    rep = GateReport(model=model_name, chat_template=chat_template)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -- 1. the abstention option -------------------------------------------
    chosen, tried = choose_third_option(tokenizer, third_option)
    rep.third_option_tried = tried
    rep.third_option_chosen = chosen
    rep.third_option_was_default = chosen == DEFAULT_OPTIONS[2]
    if chosen is None:
        rep.failures.append(
            "no abstention option is a single token on this tokenizer. Tried "
            + ", ".join(f"{t['word']} ({t['n_tokens']} tokens)" for t in tried)
            + ". The three-way readout cannot be made valid by swapping the word; "
            "the option would have to be scored as a sequence, which puts it on a "
            "different footing from the one-token Yes/No."
        )
        raise ModelGateError(_format(rep))
    if not rep.third_option_was_default:
        rep.notes.append(
            f"third option is {chosen!r}, not the default 'Unsure', because "
            f"' Unsure' is {tried[0]['n_tokens']} tokens on this tokenizer"
        )

    options = (DEFAULT_OPTIONS[0], DEFAULT_OPTIONS[1], chosen)
    rep.options = list(options)

    # -- 2. Yes / No variant sets -------------------------------------------
    table = build_option_table(tokenizer, options, strategy=variant_strategy)
    rep.option_token_ids = {
        role: {
            "word": o.word,
            "n_token_ids": len(o.token_ids),
            "n_with_leading_space": o.n_with_leading_space,
            "canonical_n_tokens": o.canonical_n_tokens,
            "variants": sorted(o.id_to_variant.values()),
        }
        for role, o in table.items()
    }
    for role, o in table.items():
        if not o.token_ids:
            rep.failures.append(f"option {role!r} resolved to no token ids at all")
        if o.n_with_leading_space == 0:
            rep.failures.append(
                f"option {role!r} has no leading-space token id; the prompt ends in "
                "'Answer:' so its mass would read artificially low"
            )
    if rep.failures:
        raise ModelGateError(_format(rep))

    # -- 3. coverage on a probe sample --------------------------------------
    if probe_items is None:
        probe_items = load_items(["items/draft", "items/seed"])
    probe = list(probe_items)[:n_probe]
    rep.n_probe_items = len(probe)

    def probe_mass(use_chat: bool):
        sc = HFScorer(
            model_name,
            device=device,
            dtype=dtype,
            variant_strategy=variant_strategy,
            chat_template=use_chat,
            seed=seed,
            options=options,
            require_canonical_single_token=True,
        )
        res = [sc.score_prompt(render_prompt(i, options)) for i in probe]
        masses = [r.mass_covered for r in res]
        return sc, sum(masses) / len(masses), min(masses), res

    scorer, mean_mass, min_mass_seen, res = probe_mass(chat_template)

    if mean_mass < min_mass and not chat_template and auto_chat_template:
        rep.notes.append(
            f"plain prompt covered only {mean_mass:.4f} of the next-token mass; "
            f"top token was {res[0].top_token!r}. Retrying with the chat template."
        )
        scorer2, mean2, min2, _ = probe_mass(True)
        if mean2 >= min_mass:
            rep.chat_template = True
            rep.chat_template_required = True
            scorer, mean_mass, min_mass_seen = scorer2, mean2, min2
            rep.notes.append(
                f"chat template lifts coverage to {mean2:.4f}. This model is "
                "instruction-tuned; use --chat-template for the real run."
            )

    rep.mass_covered_mean = mean_mass
    rep.mass_covered_min = min_mass_seen
    rep.min_mass_required = min_mass
    if mean_mass < min_mass:
        rep.failures.append(
            f"mean mass_covered {mean_mass:.4f} is below {min_mass}. The three "
            "options hold too little of the next-token distribution for "
            "p_yes_3way to mean anything. Most common cause: an "
            "instruction-tuned model given a plain completion prompt - the top "
            f"token here was {res[0].top_token!r}."
        )

    rep.passed = not rep.failures
    if not rep.passed:
        raise ModelGateError(_format(rep))
    return rep, scorer


def _format(rep: GateReport) -> str:
    lines = [f"MODEL GATE FAILED for {rep.model}", ""]
    for f in rep.failures:
        lines.append(f"  FAIL  {f}")
    for n in rep.notes:
        lines.append(f"  note  {n}")
    lines.append("")
    lines.append("  Nothing was scored. Fix the above or pick another model.")
    return "\n".join(lines)


def format_report(rep: GateReport) -> str:
    L = [f"MODEL GATE — {rep.model}", ""]
    L.append("  third option")
    for t in rep.third_option_tried:
        mark = "<-- chosen" if t["word"] == rep.third_option_chosen else ""
        L.append(f"    {t['canonical']!r:<14} {t['n_tokens']} token(s)  {mark}")
    L.append("")
    L.append("  option token ids")
    for role, d in rep.option_token_ids.items():
        L.append(
            f"    {role:<7} {d['word']:<10} {d['n_token_ids']} ids, "
            f"{d['n_with_leading_space']} with leading space, canonical "
            f"{d['canonical_n_tokens']} token(s)"
        )
    L.append("")
    L.append(
        f"  coverage  mean {rep.mass_covered_mean:.4f}, min {rep.mass_covered_min:.4f} "
        f"over {rep.n_probe_items} probe items (need > {rep.min_mass_required})"
    )
    L.append(f"  chat template: {'ON' if rep.chat_template else 'off'}"
             + ("  (REQUIRED for this model)" if rep.chat_template_required else ""))
    for n in rep.notes:
        L.append(f"  note  {n}")
    L.append("")
    L.append("  RESULT: PASS" if rep.passed else "  RESULT: FAIL")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--third-option", default=None)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--items", nargs="+", default=["items/draft", "items/seed"])
    ap.add_argument("--n-probe", type=int, default=PROBE_ITEMS)
    ap.add_argument("--min-mass", type=float, default=MIN_MASS_COVERED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    items = load_items(args.items)
    try:
        rep, _scorer = run_gate(
            args.model,
            third_option=args.third_option,
            chat_template=args.chat_template,
            device=args.device,
            dtype=args.dtype,
            probe_items=items,
            n_probe=args.n_probe,
            min_mass=args.min_mass,
        )
    except ModelGateError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(format_report(rep))
    if args.out:
        provenance.write_json(
            args.out, {"kind": "model_gate", "meta": provenance.run_meta(), **rep.to_dict()}
        )
        print(f"\nwrote {args.out}")
    print("\nRun with:")
    print(
        f"  python -m src.score --model {rep.model} "
        f"--third-option {rep.third_option_chosen}"
        + (" --chat-template" if rep.chat_template else "")
        + " --items items/draft items/seed --out results/run_x.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
