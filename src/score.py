"""THE MEASUREMENT.

Loads any HuggingFace causal LM, renders each item's prompt ending in `Answer:`,
reads the logits at the **final position**, softmaxes over the full vocabulary,
and extracts probability mass for the three options.

Three failure modes this file exists to prevent
-----------------------------------------------

1. **The tokenization bug.** `"Yes"`, `" Yes"`, `"yes"`, `" YES"` are *different
   token ids*. Reading only `tokenizer.encode("Yes")[0]` silently throws away
   most of the probability mass, and the amount thrown away varies by model, so
   it corrupts exactly the cross-model comparison the experiment is for. Here,
   mass is summed over the full casing x leading-space variant set, the sets are
   asserted non-empty and mutually disjoint, and the resolved token ids are
   written into every result file so the choice is auditable after the fact.

2. **The multi-token option.** The *canonical* continuation - the exact string
   the model must emit next, i.e. `" Unsure"` given a prompt ending in `Answer:`
   - has to be a single token or its mass cannot be read at all. This is not
   hypothetical: on SmolLM2's vocabulary `" Unsure"` is **three** tokens while
   `" Yes"` and `" No"` are one each, so the third option would read
   artificially near-zero and abstention would look impossible. `--require-
   canonical-single-token` (ON by default) turns that from a silent bias into a
   loud failure, and suggests replacements. See DECISIONS.md D-019.

3. **Silent low coverage.** If the three options together hold almost none of the
   next-token mass, the renormalized `p_yes_3way` is a ratio of two rounding
   errors. `mass_covered` is computed and reported for every single item, and the
   run fails if the mean falls below `--min-mass-covered`.

Usage
-----
    python -m src.score --model HuggingFaceTB/SmolLM2-135M-Instruct \
        --third-option Unknown --items items/draft items/seed \
        --out results/run.json

    python -m src.score --mock --mock-scenario known \
        --items items/draft items/seed --out results/run_mock.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from . import provenance
from .models import Item, load_items
from .render import (
    DEFAULT_OPTIONS,
    OPTION_ROTATIONS,
    ROLES,
    canonical_forms,
    case_permutation,
    render_prompt,
    template_hash,
)
from .scoring_types import ScoreResult

# ---------------------------------------------------------------------------
# Tokenization: the part that is easy to get wrong
# ---------------------------------------------------------------------------

#: Leading-context variants tried for every option word. The empty prefix covers
#: tokenizers that fold the space into the preceding token; the space prefix
#: covers the (much more common) case where ` Yes` is its own token.
PREFIXES: tuple[str, ...] = ("", " ")

VARIANT_STRATEGIES: tuple[str, ...] = ("explicit", "vocab_scan", "union")

#: Offered in the error message when the third option is not a single token.
THIRD_OPTION_CANDIDATES: tuple[str, ...] = (
    "Unsure",
    "Unknown",
    "Maybe",
    "Unclear",
    "Uncertain",
    "Insufficient",
    "Neither",
    "Undecided",
)


def casing_variants(word: str) -> list[str]:
    """All casing forms of a word: as-written, lower, UPPER, Capitalized, Title."""
    return sorted({word, word.lower(), word.upper(), word.capitalize(), word.title()})


def string_variants(word: str, prefixes: Sequence[str] = PREFIXES) -> list[str]:
    """The full casing x leading-space cross product, as strings."""
    return sorted({p + c for c in casing_variants(word) for p in prefixes})


@dataclass
class OptionTokens:
    """The resolved single-token ids that count as one answer option."""

    role: str
    word: str
    canonical: str
    variants_tried: list[str]
    token_ids: list[int] = field(default_factory=list)
    #: token id -> the variant string that produced it (the audit trail)
    id_to_variant: dict[int, str] = field(default_factory=dict)
    n_with_leading_space: int = 0
    canonical_is_single_token: bool = False
    canonical_n_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "word": self.word,
            "canonical": self.canonical,
            "canonical_is_single_token": self.canonical_is_single_token,
            "canonical_n_tokens": self.canonical_n_tokens,
            "variants_tried": self.variants_tried,
            "token_ids": sorted(self.token_ids),
            "id_to_variant": {str(k): v for k, v in sorted(self.id_to_variant.items())},
            "n_token_ids": len(self.token_ids),
            "n_with_leading_space": self.n_with_leading_space,
        }


class TokenizationError(RuntimeError):
    """Raised loudly when option tokens cannot be resolved. Never swallowed."""


def _encode(tokenizer, text: str) -> list[int]:
    for encode in (
        lambda t: tokenizer.encode(t, add_special_tokens=False),
        lambda t: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t)),
    ):
        try:
            ids = encode(text)
        except Exception:  # noqa: BLE001 - tokenizer APIs vary; try the next route
            continue
        if isinstance(ids, int):
            ids = [ids]
        ids = [int(i) for i in (ids or []) if i is not None]
        if ids:
            return ids
    return []


def _encode_single(tokenizer, text: str) -> int | None:
    """Return the token id if `text` encodes to exactly one non-UNK token."""
    ids = _encode(tokenizer, text)
    unk = getattr(tokenizer, "unk_token_id", None)
    if len(ids) == 1 and ids[0] != unk:
        return ids[0]
    return None


def n_tokens(tokenizer, text: str) -> int:
    return len(_encode(tokenizer, text))


def _vocab_scan(tokenizer, word: str) -> dict[int, str]:
    """Every vocabulary entry that decodes to exactly this word, modulo case and
    a single leading space. Catches `Ġyes` / `▁Yes` style pieces that the
    explicit round-trip can miss on unusual tokenizers."""
    found: dict[int, str] = {}
    needle = word.lower()
    try:
        vocab = tokenizer.get_vocab()
    except Exception as exc:  # noqa: BLE001
        raise TokenizationError(f"tokenizer has no usable get_vocab(): {exc}") from exc

    for piece, tid in vocab.items():
        # Cheap prefilter first; exact decode only on the handful that survive.
        if needle not in piece.lower():
            continue
        try:
            decoded = tokenizer.convert_tokens_to_string([piece])
        except Exception:  # noqa: BLE001
            continue
        if decoded.strip().lower() != needle:
            continue
        if decoded != decoded.strip() and decoded != " " + decoded.strip():
            continue  # e.g. "\nYes" - not a leading-space variant
        found[int(tid)] = decoded
    return found


def build_option_tokens(
    tokenizer,
    word: str,
    strategy: str = "explicit",
    role: str = "",
) -> OptionTokens:
    """Resolve the token id set for one answer option.

    Raises if it comes back empty - an empty set means every probability read for
    this option is zero, which would look like a confident model rather than a
    broken harness.
    """
    if strategy not in VARIANT_STRATEGIES:
        raise ValueError(f"strategy must be one of {VARIANT_STRATEGIES}, got {strategy}")

    canonical = " " + word
    opt = OptionTokens(
        role=role or word.lower(),
        word=word,
        canonical=canonical,
        variants_tried=string_variants(word),
        canonical_n_tokens=n_tokens(tokenizer, canonical),
    )
    opt.canonical_is_single_token = opt.canonical_n_tokens == 1

    if strategy in ("explicit", "union"):
        for v in opt.variants_tried:
            tid = _encode_single(tokenizer, v)
            if tid is not None and tid not in opt.id_to_variant:
                opt.id_to_variant[tid] = v

    if strategy in ("vocab_scan", "union"):
        for tid, decoded in _vocab_scan(tokenizer, word).items():
            opt.id_to_variant.setdefault(tid, decoded)

    opt.token_ids = sorted(opt.id_to_variant)
    opt.n_with_leading_space = sum(
        1 for v in opt.id_to_variant.values() if v.startswith(" ")
    )

    if not opt.token_ids:
        raise TokenizationError(
            f"no single-token id found for option {word!r} with strategy "
            f"{strategy!r}. Tried {opt.variants_tried!r}. This tokenizer cannot "
            "represent the option in one token, so every probability read for it "
            "would be exactly zero - which is indistinguishable from a model that "
            "never wants to answer it. Try --variant-strategy union, or pick a "
            "different option word."
        )
    return opt


def build_option_table(
    tokenizer,
    options: Sequence[str] = DEFAULT_OPTIONS,
    strategy: str = "explicit",
) -> dict[str, OptionTokens]:
    """Resolve all three options by ROLE and assert the id sets are disjoint."""
    table = {
        role: build_option_tokens(tokenizer, word, strategy, role=role)
        for role, word in zip(ROLES, options)
    }

    seen: dict[int, str] = {}
    collisions: list[str] = []
    for role, opt in table.items():
        for tid in opt.token_ids:
            if tid in seen:
                collisions.append(
                    f"token {tid} claimed by both {seen[tid]!r} and {role!r}"
                )
            seen[tid] = role
    if collisions:
        raise TokenizationError(
            "option token id sets overlap, so probability mass would be "
            "double-counted: " + "; ".join(collisions)
        )
    return table


def suggest_single_token_options(
    tokenizer, candidates: Sequence[str] = THIRD_OPTION_CANDIDATES
) -> list[str]:
    """Which candidate third-option words ARE a single token on this tokenizer."""
    return [c for c in candidates if n_tokens(tokenizer, " " + c) == 1]


def check_canonical_single_token(
    tokenizer, table: dict[str, OptionTokens], *, strict: bool = True
) -> list[str]:
    """The gate for failure mode 2.

    The prompt ends in `Answer:` with no trailing space, so the model's next
    token IS the canonical form (` Yes` / ` No` / ` Unsure`). If a canonical form
    spans several tokens, that option's mass is unreadable from a single
    next-token distribution and it will score artificially near zero.
    """
    bad = [r for r, o in table.items() if not o.canonical_is_single_token]
    if not bad:
        return []

    detail = "; ".join(
        f"{table[r].canonical!r} is {table[r].canonical_n_tokens} tokens" for r in bad
    )
    alts = suggest_single_token_options(tokenizer)
    msg = (
        f"canonical option form(s) are not single tokens on this tokenizer: {detail}. "
        "The prompt ends in 'Answer:' so the model's very next token IS that "
        "string; a multi-token option cannot be read from one next-token "
        "distribution and will score near zero for structural reasons, which "
        "looks exactly like a model that never abstains. "
        f"Single-token alternatives on this tokenizer: {alts or 'NONE FOUND'}. "
        "Fix with --third-option <word>, or accept the bias explicitly with "
        "--no-require-canonical-single-token (and say so in the writeup)."
    )
    if strict:
        raise TokenizationError(msg)
    print("WARNING: " + msg, file=sys.stderr)
    return bad


def warn_if_no_space_variants(table: dict[str, OptionTokens]) -> list[str]:
    """A weaker version of the same concern, for the variant sets."""
    bad = [r for r, o in table.items() if o.n_with_leading_space == 0]
    if bad:
        print(
            f"WARNING: no leading-space token id found for option role(s) {bad}. "
            "The prompt ends in 'Answer:' so the natural continuation has a "
            "leading space; these options will read artificially low. Consider "
            "--variant-strategy union.",
            file=sys.stderr,
        )
    return bad


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class Scorer(Protocol):
    """Everything downstream depends only on this. The mock satisfies it too."""

    options: tuple[str, str, str]

    def score_prompt(self, prompt: str) -> ScoreResult: ...

    def meta(self) -> dict[str, Any]: ...


class HFScorer:
    """Real scorer: final-position logits from a HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        variant_strategy: str = "explicit",
        chat_template: bool = False,
        seed: int = provenance.SEED_DEFAULT,
        temperature: float = 1.0,
        add_special_tokens: bool = True,
        options: Sequence[str] = DEFAULT_OPTIONS,
        require_canonical_single_token: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.temperature = temperature  # irrelevant to raw logits; recorded anyway
        self.chat_template = chat_template
        self.add_special_tokens = add_special_tokens
        self.options = tuple(options)  # type: ignore[assignment]
        self.determinism = provenance.set_determinism(seed)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype_name = dtype
        torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, dtype=torch_dtype
            )
        except TypeError:  # transformers < 5 spelling
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, revision=revision, torch_dtype=torch_dtype
            )
        self.model.to(self.device)
        self.model.eval()

        self.option_tokens = build_option_table(
            self.tokenizer, self.options, strategy=variant_strategy
        )
        self.variant_strategy = variant_strategy
        self.canonical_violations = check_canonical_single_token(
            self.tokenizer, self.option_tokens, strict=require_canonical_single_token
        )
        self.missing_space_variants = warn_if_no_space_variants(self.option_tokens)
        self.revision = provenance.resolve_revision(model_name, self.model)
        self.single_token_alternatives = suggest_single_token_options(self.tokenizer)

    # -- prompt -> probabilities --------------------------------------------

    def _prepare(self, prompt: str) -> str:
        if not self.chat_template:
            return prompt
        # Opt-in only (D-009): this changes the token immediately before the
        # answer position, which changes the option logits model-specifically.
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def score_prompt(self, prompt: str) -> ScoreResult:
        return self.score_prompts([prompt])[0]

    def score_prompts(self, prompts: Sequence[str]) -> list[ScoreResult]:
        torch = self.torch
        texts = [self._prepare(p) for p in prompts]

        if len(texts) == 1:
            enc = self.tokenizer(
                texts[0],
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]
        else:
            # Left padding, so the final position is the real final token for
            # every sequence. position_ids must be derived from the attention
            # mask or absolute-position models silently read the wrong offsets.
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            enc = self.tokenizer(
                list(texts),
                return_tensors="pt",
                padding=True,
                add_special_tokens=self.add_special_tokens,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            mask = enc["attention_mask"]
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            with torch.no_grad():
                try:
                    logits = self.model(
                        **enc, position_ids=position_ids
                    ).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**enc).logits[:, -1, :]

        probs = torch.softmax(logits.to(torch.float64), dim=-1)

        results: list[ScoreResult] = []
        for row in probs:
            mass = {
                role: float(row[opt.token_ids].sum().item())
                for role, opt in self.option_tokens.items()
            }
            top_id = int(torch.argmax(row).item())
            results.append(
                ScoreResult(
                    p_yes_raw=mass["yes"],
                    p_no_raw=mass["no"],
                    p_unsure_raw=mass["unsure"],
                    top_token=self.tokenizer.decode([top_id]),
                    top_token_prob=float(row[top_id].item()),
                )
            )
        return results

    def meta(self) -> dict[str, Any]:
        return {
            "scorer": "HFScorer",
            "model": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype_name,
            "chat_template": self.chat_template,
            "add_special_tokens": self.add_special_tokens,
            "variant_strategy": self.variant_strategy,
            "temperature": self.temperature,
            "determinism": self.determinism,
            "option_words": list(self.options),
            "options": {r: o.to_dict() for r, o in self.option_tokens.items()},
            "canonical_multi_token_roles": self.canonical_violations,
            "options_missing_leading_space_variants": self.missing_space_variants,
            "single_token_third_option_candidates": self.single_token_alternatives,
            "prompt_template_hash": template_hash(self.options),
            "vocab_size": int(getattr(self.model.config, "vocab_size", 0)),
        }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def record_for(item: Item, res: ScoreResult) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "item_id": item.id,
        "family_id": item.family_id,
        "cell": item.cell,
        "coherence": item.coherence,
        "ground_truth": item.ground_truth,
        "domain": item.domain,
        "flaw_mechanism": item.flaw_mechanism,
        "confound_variant": item.confound_variant,
        "scope_variant": item.scope_variant,
        "salience": item.salience,
        "surface_complexity": item.surface_complexity,
        "word_count": item.word_count,
        "review_status": item.review_status,
    }
    rec.update(res.to_dict())
    return rec


def score_items(
    scorer: Scorer,
    items: Sequence[Item],
    *,
    batch_size: int = 1,
    progress: bool = True,
    shuffle_cases: bool = False,
    case_seed: int = 0,
    option_rotations: bool = False,
    rotation_spread_limit: float = 0.05,
) -> list[dict[str, Any]]:
    """Score every item, optionally under the two surface-feature controls.

    `shuffle_cases` permutes the four case sentences with a fixed per-item
    permutation, which is recorded on the record. Case order carries no evidence,
    so a result that moves under it is measuring presentation.

    `option_rotations` scores every item under all three rotations of the option
    list and averages P(yes). The spread across rotations is the size of the
    position bias, reported per item and flagged above
    `rotation_spread_limit`.
    """
    options = getattr(scorer, "options", DEFAULT_OPTIONS)
    rotations = list(OPTION_ROTATIONS) if option_rotations else [None]

    plan: list[tuple[int, tuple | None, list[int] | None]] = []
    orders: list[list[int] | None] = []
    for n, item in enumerate(items):
        order = case_permutation(item, case_seed) if shuffle_cases else None
        orders.append(order)
        for rot in rotations:
            plan.append((n, rot, order))

    prompts = [
        render_prompt(items[n], options, case_order=order, option_rotation=rot)
        for n, rot, order in plan
    ]

    results: list[ScoreResult] = []
    if batch_size > 1 and hasattr(scorer, "score_prompts"):
        for start in range(0, len(prompts), batch_size):
            results.extend(scorer.score_prompts(prompts[start : start + batch_size]))
            if progress:
                print(f"  scored {len(results)}/{len(prompts)}", file=sys.stderr)
    else:
        for n, prompt in enumerate(prompts, 1):
            results.append(scorer.score_prompt(prompt))
            if progress and (n % 20 == 0 or n == len(prompts)):
                print(f"  scored {n}/{len(prompts)}", file=sys.stderr)

    by_item: dict[int, list[ScoreResult]] = {}
    for (n, _rot, _order), res in zip(plan, results):
        by_item.setdefault(n, []).append(res)

    records: list[dict[str, Any]] = []
    for n, item in enumerate(items):
        got = by_item[n]
        # The headline record is the FIRST rotation, so a run with rotations on
        # and one with them off agree on the primary number; the average and the
        # spread are reported alongside rather than silently replacing it.
        rec = record_for(item, got[0])
        rec["case_order"] = orders[n]
        if option_rotations:
            vals = [r.p_yes_3way for r in got]
            spread = max(vals) - min(vals)
            rec["rotation_p_yes_3way"] = vals
            rec["rotation_labels"] = [list(r) for r in OPTION_ROTATIONS]
            rec["p_yes_3way_rotation_mean"] = sum(vals) / len(vals)
            rec["p_yes_3way_rotation_spread"] = spread
            rec["rotation_spread_flagged"] = spread > rotation_spread_limit
        records.append(rec)
    return records


def build_payload(
    scorer: Scorer,
    items: Sequence[Item],
    records: list[dict[str, Any]],
    *,
    kind: str = "evidence",
    **extra: Any,
) -> dict[str, Any]:
    meta = provenance.run_meta(**scorer.meta(), **extra)
    meta["n_items"] = len(items)
    meta["items_hash"] = provenance.hash_items([i.id for i in items])
    options = getattr(scorer, "options", DEFAULT_OPTIONS)
    meta["prompts_hash"] = provenance.hash_prompts(
        render_prompt(i, options) for i in items
    )
    meta["prompts_hash_note"] = (
        "digest of the UNPERMUTED prompts, so it identifies the item text "
        "regardless of which surface controls were on"
    )
    mass = [r["mass_covered"] for r in records]
    meta["mass_covered_mean"] = sum(mass) / len(mass) if mass else 0.0
    meta["mass_covered_min"] = min(mass) if mass else 0.0
    model_slug = str(meta.get("model", "mock")).replace("/", "_").replace(":", "-")
    return {
        "run_id": f"{model_slug}__{meta['items_hash']}__{meta['prompt_template_hash']}",
        "kind": kind,
        "meta": meta,
        "records": records,
    }


def make_scorer(
    args: argparse.Namespace, items: Sequence[Item] | None = None
) -> Scorer:
    options = (DEFAULT_OPTIONS[0], DEFAULT_OPTIONS[1], args.third_option)
    if args.mock:
        from .mock_scorer import MockScorer

        scorer = MockScorer(
            scenario=args.mock_scenario, seed=args.seed, options=options
        )
        if items is not None:
            # The mock's answer for an item depends on that item's rank within
            # its (coherence, truth) group, so it needs the whole set up front.
            scorer.prepare(items)
        return scorer
    if not args.model:
        raise SystemExit("--model is required unless --mock is passed")
    return HFScorer(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        variant_strategy=args.variant_strategy,
        chat_template=args.chat_template,
        seed=args.seed,
        temperature=args.temperature,
        options=options,
        require_canonical_single_token=args.require_canonical_single_token,
    )


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", help="HuggingFace causal LM name")
    ap.add_argument("--revision", default=None, help="pin a specific weights revision")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument(
        "--variant-strategy", default="explicit", choices=list(VARIANT_STRATEGIES)
    )
    ap.add_argument("--chat-template", action="store_true", help="opt-in; see D-009")
    ap.add_argument(
        "--third-option",
        default=DEFAULT_OPTIONS[2],
        help="surface word for the abstain option. Default 'Unsure' (D-002). Must "
        "be a single token in its ' <Word>' form on the tokenizer in use (D-019).",
    )
    ap.add_argument(
        "--no-require-canonical-single-token",
        dest="require_canonical_single_token",
        action="store_false",
        help="downgrade the multi-token-option failure to a warning. Only do this "
        "if you intend to report the resulting bias.",
    )
    ap.set_defaults(require_canonical_single_token=True)
    ap.add_argument("--seed", type=int, default=provenance.SEED_DEFAULT)
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="does not affect raw logits; recorded so nobody assumes sampling",
    )
    ap.add_argument("--mock", action="store_true", help="use the deterministic mock")
    ap.add_argument("--mock-scenario", default="known")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--shuffle-cases",
        action="store_true",
        help="permute each item's four case sentences with a fixed per-item "
        "permutation, recorded on every record. Case order carries no "
        "evidence, so a result that moves under it is measuring presentation.",
    )
    ap.add_argument("--case-seed", type=int, default=0)
    ap.add_argument(
        "--option-rotations",
        action="store_true",
        help="score every item under all three rotations of the option list "
        "and report the spread in P(yes). Measures option position bias.",
    )
    ap.add_argument("--rotation-spread-limit", type=float, default=0.05)
    ap.add_argument(
        "--items", nargs="+", default=["items/draft", "items/seed"], help="directories"
    )


def check_tokenization_only(args: argparse.Namespace) -> int:
    """Load ONLY the tokenizer and print the option table. Run this before
    committing to a model - it is much cheaper than downloading weights and it
    answers the one question that invalidates everything downstream (D-019)."""
    from transformers import AutoTokenizer

    options = (DEFAULT_OPTIONS[0], DEFAULT_OPTIONS[1], args.third_option)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    table = build_option_table(tok, options, strategy=args.variant_strategy)

    print(f"tokenizer: {args.model}")
    print(f"options:   {list(options)}   strategy: {args.variant_strategy}")
    print(f"{'role':<8} {'canonical':<12} {'n_tok':>5}  {'ids':>4}  variants")
    for role, opt in table.items():
        flag = "OK " if opt.canonical_is_single_token else "BAD"
        print(
            f"{role:<8} {opt.canonical!r:<12} {opt.canonical_n_tokens:>5} {flag} "
            f"{len(opt.token_ids):>4}  {sorted(opt.id_to_variant.values())}"
        )

    bad = check_canonical_single_token(tok, table, strict=False)
    warn_if_no_space_variants(table)
    alts = suggest_single_token_options(tok)
    print(f"\nsingle-token third-option candidates: {alts or 'NONE'}")
    if bad:
        print(f"RESULT: NOT USABLE with third option {args.third_option!r} (roles {bad})")
        return 1
    print("RESULT: usable")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score items with a causal LM.")
    add_common_args(ap)
    ap.add_argument(
        "--check-tokenization-only",
        action="store_true",
        help="load only the tokenizer, print the option table, and exit",
    )
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--min-mass-covered",
        type=float,
        default=0.01,
        help="fail if the mean of (p_yes+p_no+p_unsure) falls below this; a low "
        "value means the renormalized numbers are ratios of rounding errors",
    )
    args = ap.parse_args(argv)

    if args.check_tokenization_only:
        if not args.model:
            raise SystemExit("--check-tokenization-only needs --model")
        return check_tokenization_only(args)
    if not args.out:
        raise SystemExit("--out is required")

    items = load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no items found under {args.items}")
    print(f"loaded {len(items)} items from {args.items}", file=sys.stderr)

    try:
        scorer = make_scorer(args, items)
    except TokenizationError as exc:
        # A clean, actionable message beats a traceback for the one failure mode
        # a user is most likely to hit on a new model.
        print(f"\nTOKENIZATION CHECK FAILED\n\n  {exc}\n", file=sys.stderr)
        print(
            "  Run `python -m src.score --model <name> --check-tokenization-only` "
            "to see the full option table.",
            file=sys.stderr,
        )
        return 2

    records = score_items(
        scorer,
        items,
        batch_size=args.batch_size,
        shuffle_cases=args.shuffle_cases,
        case_seed=args.case_seed,
        option_rotations=args.option_rotations,
        rotation_spread_limit=args.rotation_spread_limit,
    )
    payload = build_payload(
        scorer,
        items,
        records,
        item_dirs=list(args.items),
        shuffle_cases=args.shuffle_cases,
        case_seed=args.case_seed if args.shuffle_cases else None,
        option_rotations=args.option_rotations,
    )

    out = provenance.write_json(args.out, payload)
    m = payload["meta"]
    print(f"wrote {out}  ({len(records)} records)")
    if args.option_rotations:
        spreads = [r["p_yes_3way_rotation_spread"] for r in records]
        flagged = [r["item_id"] for r in records if r["rotation_spread_flagged"]]
        print(
            f"option position bias: mean spread {sum(spreads) / len(spreads):.4f}, "
            f"max {max(spreads):.4f}; {len(flagged)} item(s) over "
            f"{args.rotation_spread_limit}"
        )
        for i in flagged[:10]:
            print(f"  FLAGGED {i}")
    print(
        f"mean mass_covered = {m['mass_covered_mean']:.4f}  "
        f"min = {m['mass_covered_min']:.4f}"
    )
    if m["mass_covered_mean"] < args.min_mass_covered:
        print(
            f"FAIL: mean mass_covered {m['mass_covered_mean']:.5f} < "
            f"{args.min_mass_covered}. The three options hold almost none of the "
            "next-token distribution; p_yes_3way is not interpretable. Inspect "
            "'top_token' in the records.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
