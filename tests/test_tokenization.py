"""The tokenization bug the whole measurement hinges on.

'Yes', ' Yes', 'yes', ' YES' are different token ids. If only one is read, most
of the probability mass silently disappears - and the amount that disappears is
model-specific, so it corrupts exactly the cross-model comparison this repo
exists to make.

The brief requires: assert the variant sets are non-empty for the tokenizer in
use, and fail loudly if not. Both halves are tested here - offline against a
fake tokenizer, and (marked `model`) against the real one.
"""

from __future__ import annotations

import os

import pytest
from fake_tokenizer import CollidingTokenizer, FakeTokenizer, NoYesTokenizer

from src.render import DEFAULT_OPTIONS, ROLES
from src.score import (
    TokenizationError,
    build_option_table,
    build_option_tokens,
    casing_variants,
    check_canonical_single_token,
    n_tokens,
    string_variants,
    suggest_single_token_options,
    warn_if_no_space_variants,
)

SMOKE_MODEL = os.environ.get("CC_SMOKE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")


# ---- variant construction --------------------------------------------------


def test_casing_variants_cover_the_four_forms():
    v = casing_variants("Yes")
    assert set(v) == {"Yes", "yes", "YES"}
    v = casing_variants("unsure")
    assert {"unsure", "UNSURE", "Unsure"} <= set(v)


def test_string_variants_include_both_prefixes():
    v = string_variants("Yes")
    assert " Yes" in v and "Yes" in v
    assert " yes" in v and "yes" in v
    assert " YES" in v and "YES" in v
    assert len(v) == 2 * len(casing_variants("Yes"))


# ---- resolution against a tokenizer ---------------------------------------


@pytest.fixture
def tok():
    return FakeTokenizer()


def test_variant_sets_are_non_empty(tok):
    """The core assertion the brief asks for."""
    for word in DEFAULT_OPTIONS:
        opt = build_option_tokens(tok, word)
        assert opt.token_ids, f"empty token id set for option {word!r}"


def test_all_six_variants_resolve(tok):
    opt = build_option_tokens(tok, "Yes")
    assert len(opt.token_ids) == 6, opt.id_to_variant
    assert set(opt.id_to_variant.values()) == set(string_variants("Yes"))


def test_leading_space_variants_are_counted(tok):
    opt = build_option_tokens(tok, "Yes")
    assert opt.n_with_leading_space == 3


def test_sum_over_variants_is_not_just_the_first_id(tok):
    """Regression guard for the actual bug: one id is not enough."""
    opt = build_option_tokens(tok, "Yes")
    assert len(opt.token_ids) > 1


def test_fails_loudly_when_an_option_is_unrepresentable():
    with pytest.raises(TokenizationError, match="no single-token id found"):
        build_option_tokens(NoYesTokenizer(), "Yes")


def test_fails_loudly_on_overlapping_option_ids():
    with pytest.raises(TokenizationError, match="overlap"):
        build_option_table(CollidingTokenizer())


def test_option_table_ids_are_disjoint(tok):
    table = build_option_table(tok)
    ids = [t for o in table.values() for t in o.token_ids]
    assert len(ids) == len(set(ids))
    assert set(table) == set(ROLES)


def test_rejects_unknown_strategy(tok):
    with pytest.raises(ValueError, match="strategy must be one of"):
        build_option_tokens(tok, "Yes", strategy="nonsense")


# ---- vocab_scan strategy ---------------------------------------------------


def test_vocab_scan_finds_the_same_ids(tok):
    explicit = build_option_tokens(tok, "Yes", strategy="explicit")
    scanned = build_option_tokens(tok, "Yes", strategy="vocab_scan")
    assert set(scanned.token_ids) == set(explicit.token_ids)


def test_vocab_scan_rejects_substring_matches(tok):
    """'yesterday' contains 'yes' and must not be counted."""
    scanned = build_option_tokens(tok, "Yes", strategy="vocab_scan")
    decoded = {tok.decode([i]).strip().lower() for i in scanned.token_ids}
    assert decoded == {"yes"}


def test_vocab_scan_rejects_newline_prefixed_forms(tok):
    """'\\nYes' is in the fake vocab; it is not a leading-space variant."""
    scanned = build_option_tokens(tok, "Yes", strategy="vocab_scan")
    assert all(not tok.decode([i]).startswith("\n") for i in scanned.token_ids)


def test_vocab_scan_rejects_nobody_for_no(tok):
    scanned = build_option_tokens(tok, "No", strategy="vocab_scan")
    decoded = {tok.decode([i]).strip().lower() for i in scanned.token_ids}
    assert decoded == {"no"}


def test_union_is_a_superset_of_both(tok):
    a = set(build_option_tokens(tok, "No", strategy="explicit").token_ids)
    b = set(build_option_tokens(tok, "No", strategy="vocab_scan").token_ids)
    u = set(build_option_tokens(tok, "No", strategy="union").token_ids)
    assert a <= u and b <= u


# ---- the leading-space warning --------------------------------------------


def test_warns_when_no_leading_space_variant_exists(tok, capsys):
    table = build_option_table(tok)
    for opt in table.values():
        opt.n_with_leading_space = 0
    bad = warn_if_no_space_variants(table)
    assert set(bad) == set(ROLES)
    assert "leading-space" in capsys.readouterr().err


def test_no_warning_for_a_healthy_tokenizer(tok, capsys):
    assert warn_if_no_space_variants(build_option_table(tok)) == []
    assert capsys.readouterr().err == ""


# ---- the canonical-form gate (D-019) ---------------------------------------
#
# The prompt ends in 'Answer:' with no trailing space, so the model's very next
# token IS ' Yes' / ' No' / ' Unsure'. If one of those is multi-token, its mass
# cannot be read from a single next-token distribution and it scores near zero
# for structural reasons - which looks identical to a model that never abstains.


def test_canonical_forms_are_single_tokens_on_a_healthy_tokenizer(tok):
    table = build_option_table(tok)
    assert check_canonical_single_token(tok, table, strict=True) == []
    for opt in table.values():
        assert opt.canonical_is_single_token
        assert opt.canonical_n_tokens == 1


def test_canonical_gate_fails_loudly_on_a_multi_token_option(tok):
    """This is the real SmolLM2 situation: ' Unsure' is 3 tokens there."""
    broken = FakeTokenizer()
    for piece in ("ĠUnsure", "Unsure", "Ġunsure", "unsure", "ĠUNSURE", "UNSURE"):
        broken._vocab.pop(piece, None)
    broken._vocab["ĠUn"] = 900
    broken._vocab["sure"] = 901
    broken._vocab["Ġunsure"] = 902  # only the lowercase spaced form survives
    broken._inv = {v: k for k, v in broken._vocab.items()}

    table = build_option_table(broken)
    assert table["unsure"].token_ids, "the variant set should still be non-empty"
    assert not table["unsure"].canonical_is_single_token

    with pytest.raises(TokenizationError, match="not single tokens"):
        check_canonical_single_token(broken, table, strict=True)

    # Non-strict downgrades it to a warning that names the offending role.
    bad = check_canonical_single_token(broken, table, strict=False)
    assert bad == ["unsure"]


def test_canonical_gate_suggests_working_alternatives(tok):
    alts = suggest_single_token_options(tok, candidates=("Unsure", "Nonexistentword"))
    assert "Unsure" in alts
    assert "Nonexistentword" not in alts


def test_n_tokens_helper(tok):
    assert n_tokens(tok, " Yes") == 1
    assert n_tokens(tok, " definitely not in this vocabulary") > 1


def test_custom_third_option_flows_through(tok):
    table = build_option_table(tok, options=("Yes", "No", "Unsure"))
    assert table["unsure"].word == "Unsure"
    assert set(table) == set(ROLES)


# ---- the real tokenizer ----------------------------------------------------


@pytest.mark.model
def test_real_tokenizer_variant_sets_are_non_empty():
    """Runs against the tokenizer actually in use. Fails loudly if any option
    cannot be represented as a single token."""
    transformers = pytest.importorskip("transformers")
    try:
        real = transformers.AutoTokenizer.from_pretrained(SMOKE_MODEL)
    except Exception as exc:  # noqa: BLE001 - offline is a skip, not a failure
        pytest.skip(f"cannot load {SMOKE_MODEL}: {exc}")

    table = build_option_table(real, strategy="explicit")
    for word, opt in table.items():
        assert opt.token_ids, (
            f"tokenizer {SMOKE_MODEL} has no single-token id for option {word!r}; "
            f"tried {opt.variants_tried}"
        )
        assert opt.n_with_leading_space > 0, (
            f"option {word!r} has no leading-space token id on {SMOKE_MODEL}; the "
            "prompt ends in 'Answer:' so its mass would read artificially low"
        )
    ids = [t for o in table.values() for t in o.token_ids]
    assert len(ids) == len(set(ids)), "option token ids overlap on the real tokenizer"


@pytest.mark.model
def test_real_tokenizer_has_a_workable_third_option():
    """The gate must either pass, or name a replacement that does pass.

    On SmolLM2 the default ' Unsure' is 3 tokens, so this test is what turns
    'the abstain option silently reads near zero' into a fact on the record.
    """
    transformers = pytest.importorskip("transformers")
    try:
        real = transformers.AutoTokenizer.from_pretrained(SMOKE_MODEL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot load {SMOKE_MODEL}: {exc}")

    alts = suggest_single_token_options(real)
    assert alts, (
        f"no candidate third-option word is a single token on {SMOKE_MODEL}; the "
        "three-way readout cannot be made valid by swapping the word alone"
    )

    table = build_option_table(real, options=("Yes", "No", alts[0]))
    assert check_canonical_single_token(real, table, strict=True) == []
