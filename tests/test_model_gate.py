"""The pre-flight gate must refuse, not warn (fix-pass step 7).

Each check here corresponds to a way this project has already produced a wrong
number, so each is tested against a tokenizer that fails it.
"""

from __future__ import annotations

import os

import pytest
from fake_tokenizer import FakeTokenizer, NoYesTokenizer

from src.model_gate import (
    THIRD_OPTION_ORDER,
    GateReport,
    ModelGateError,
    choose_third_option,
    format_report,
)
from src.score import TokenizationError, build_option_table

SMOKE_MODEL = os.environ.get("CC_SMOKE_MODEL", "HuggingFaceTB/SmolLM2-135M")


# ---- the abstention option -------------------------------------------------


def test_candidate_order_is_the_one_specified():
    assert THIRD_OPTION_ORDER == ("Unsure", "Maybe", "Unclear", "Unknown")


def test_prefers_unsure_when_it_is_a_single_token():
    tok = FakeTokenizer()
    chosen, tried = choose_third_option(tok)
    assert chosen == "Unsure"
    assert tried[0]["word"] == "Unsure"
    assert tried[0]["n_tokens"] == 1


def test_falls_through_the_candidate_order_and_records_every_attempt():
    """The real SmolLM2 situation: ' Unsure' is 3 tokens, so a later candidate
    has to be used - and which one has to be on the record."""
    tok = FakeTokenizer()
    for piece in ("ĠUnsure", "Unsure", "Ġunsure", "unsure", "ĠUNSURE", "UNSURE"):
        tok._vocab.pop(piece, None)
    for word in ("Maybe", "Unclear"):
        for piece in (word, "Ġ" + word, word.lower(), "Ġ" + word.lower()):
            tok._vocab.pop(piece, None)
    for piece in ("Unknown", "ĠUnknown", "unknown", "Ġunknown"):
        tok._vocab[piece] = 900 + len(tok._vocab)
    tok._inv = {v: k for k, v in tok._vocab.items()}

    chosen, tried = choose_third_option(tok)
    assert chosen == "Unknown"
    assert [t["word"] for t in tried] == list(THIRD_OPTION_ORDER)
    assert tried[0]["n_tokens"] > 1  # Unsure was tried and rejected
    assert tried[-1]["n_tokens"] == 1


def test_returns_none_when_no_candidate_works():
    tok = FakeTokenizer()
    for word in THIRD_OPTION_ORDER:
        for piece in (word, "Ġ" + word, word.lower(), "Ġ" + word.lower(),
                      word.upper(), "Ġ" + word.upper()):
            tok._vocab.pop(piece, None)
    tok._inv = {v: k for k, v in tok._vocab.items()}
    chosen, tried = choose_third_option(tok)
    assert chosen is None
    assert all(t["n_tokens"] > 1 for t in tried)


def test_an_explicit_preference_is_tried_first():
    tok = FakeTokenizer()
    for piece in ("Unknown", "ĠUnknown", "unknown", "Ġunknown"):
        tok._vocab[piece] = 900 + len(tok._vocab)
    tok._inv = {v: k for k, v in tok._vocab.items()}
    chosen, tried = choose_third_option(tok, preferred="Unknown")
    assert tried[0]["word"] == "Unknown"
    assert chosen == "Unknown"


def test_a_preference_that_does_not_tokenize_falls_through():
    """Asking for a word the vocabulary cannot represent must not override the
    check - it falls through to the next candidate that can."""
    chosen, tried = choose_third_option(FakeTokenizer(), preferred="Unknown")
    assert tried[0]["word"] == "Unknown" and tried[0]["n_tokens"] > 1
    assert chosen == "Unsure"


# ---- Yes / No variant sets -------------------------------------------------


def test_yes_no_variant_sets_must_be_non_empty():
    with pytest.raises(TokenizationError, match="no single-token id found"):
        build_option_table(NoYesTokenizer(), ("Yes", "No", "Unsure"))


def test_variant_sets_sum_over_casing_and_leading_space():
    table = build_option_table(FakeTokenizer(), ("Yes", "No", "Unsure"))
    for role, opt in table.items():
        assert len(opt.token_ids) > 1, role
        assert opt.n_with_leading_space > 0, role


# ---- report shape ----------------------------------------------------------


def test_report_renders_a_failure_without_coverage_numbers():
    """A gate that fails before the probe must still print something useful."""
    rep = GateReport(model="x", failures=["nope"], passed=False)
    assert rep.passed is False
    assert rep.to_dict()["failures"] == ["nope"]


def test_report_formats_when_complete():
    rep = GateReport(
        model="m",
        passed=True,
        options=["Yes", "No", "Unknown"],
        third_option_tried=[{"word": "Unsure", "canonical": " Unsure", "n_tokens": 3},
                            {"word": "Maybe", "canonical": " Maybe", "n_tokens": 1}],
        third_option_chosen="Maybe",
        option_token_ids={
            "yes": {"word": "Yes", "n_token_ids": 4, "n_with_leading_space": 2,
                    "canonical_n_tokens": 1, "variants": []},
        },
        mass_covered_mean=0.71,
        mass_covered_min=0.66,
        n_probe_items=6,
    )
    text = format_report(rep)
    assert "RESULT: PASS" in text
    assert "Maybe" in text and "<-- chosen" in text
    assert "0.7100" in text


def test_model_gate_error_is_not_a_warning():
    assert issubclass(ModelGateError, RuntimeError)


# ---- against a real model --------------------------------------------------


@pytest.mark.model
def test_gate_passes_on_the_smoke_model_and_names_its_third_option():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(SMOKE_MODEL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot load {SMOKE_MODEL}: {exc}")

    chosen, tried = choose_third_option(tok)
    assert chosen is not None, f"no usable abstention option on {SMOKE_MODEL}"
    by_word = {t["word"]: t["n_tokens"] for t in tried}
    # The finding that motivated the whole gate.
    assert by_word["Unsure"] > 1
    assert by_word[chosen] == 1

    table = build_option_table(tok, ("Yes", "No", chosen))
    for role, opt in table.items():
        assert opt.token_ids, role
        assert opt.canonical_is_single_token, role
