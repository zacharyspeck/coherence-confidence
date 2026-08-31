from __future__ import annotations

from conftest import make_item

from src.render import (
    INSTRUCTION,
    OPTION_WORDS,
    TEMPLATE_HASH,
    render_baseline_prompt,
    render_prompt,
)


def test_prompt_ends_in_answer_colon(item):
    p = render_prompt(item)
    assert p.endswith("Answer:"), repr(p[-40:])
    # no trailing space: the leading-space variant of the option is the natural
    # next token, and a trailing space would double it.
    assert not p.endswith("Answer: ")


def test_prompt_contains_claim_passage_and_options(item):
    p = render_prompt(item)
    assert item.claim in p
    assert item.passage in p
    for w in OPTION_WORDS:
        assert w in p


def test_prompt_never_leaks_the_answer_key():
    it = make_item(cell="coherent_false")
    p = render_prompt(it)
    assert it.confound_note not in p
    assert it.cell not in p
    assert "ground_truth" not in p
    assert (it.flaw_mechanism or "") not in p


def test_baseline_prompt_has_no_cases(item):
    p = render_baseline_prompt(item.claim)
    assert item.claim in p
    for c in item.cases:
        assert c.text not in p
    assert p.endswith("Answer:")


def test_baseline_and_evidence_prompts_share_instruction(item):
    assert INSTRUCTION in render_prompt(item)
    assert INSTRUCTION in render_baseline_prompt(item.claim)


def test_template_hash_is_stable_and_short():
    assert len(TEMPLATE_HASH) == 16
    assert TEMPLATE_HASH == TEMPLATE_HASH


def test_all_four_cells_render_identically_except_the_passage():
    """The only thing that may differ between cells is the passage."""
    prompts = {}
    for cell in ("coherent_true", "coherent_false", "diverse_true", "diverse_false"):
        it = make_item(cell=cell)
        prompts[cell] = render_prompt(it).replace(it.passage, "<PASSAGE>")
    assert len(set(prompts.values())) == 1, prompts
