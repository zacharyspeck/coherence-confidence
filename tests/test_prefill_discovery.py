"""discover_answer_prefill (D-049): the walk past a model's formatting habit.

The real case this encodes: Qwen3-8B answers "**Yes**", so after "Answer:" the
top token is ' **' holding 0.84 and the options hold 0.16. The discovery must
append the formatting token and find the options one level down - and must
refuse to append anything that is NOT pure formatting, because appending a
content token would bias the readout.
"""

import pytest

from src.score import discover_answer_prefill, is_formatting_token


def make_probe(script):
    """probe returning script[text_suffix_after_base]; asserts expected calls."""
    calls = []

    def probe(text):
        calls.append(text)
        suffix = text[len("BASE"):]
        return script[suffix]

    probe.calls = calls
    return probe


def test_markdown_bold_habit_descends_one_level():
    # Qwen3-8B, as observed on Kaggle: 0.84 on ' **', options at 0.16; after
    # absorbing ' **' the model answers 'Yes' (no leading space) at 0.92.
    top10_0 = [(0.8397, " **"), (0.1603, " Yes")]
    top10_1 = [(0.92, "Yes"), (0.05, "No")]
    probe = make_probe({
        "Answer:": (" **", False, 0.1603, top10_0),
        "Answer: **": ("Yes", True, 0.92, top10_1),
    })
    got = discover_answer_prefill(probe, "BASE")
    assert got == "Answer: **"
    assert probe.calls == ["BASEAnswer:", "BASEAnswer: **"]


def test_clean_model_keeps_seed():
    probe = make_probe({"Answer:": (" Yes", True, 0.9999, [(0.99, " Yes")])})
    assert discover_answer_prefill(probe, "BASE") == "Answer:"
    assert probe.calls == ["BASEAnswer:"]


def test_content_token_refused_with_distribution():
    # A model that wants to write prose ('The evidence...') must NOT have
    # 'The' appended - that would bias the readout. Raise, and the message
    # must carry the top-10 so the failure is self-explanatory.
    top10 = [(0.7, "The"), (0.1, " Yes")]
    probe = make_probe({"Answer:": ("The", False, 0.1, top10)})
    with pytest.raises(RuntimeError) as exc:
        discover_answer_prefill(probe, "BASE")
    msg = str(exc.value)
    assert "not a formatting token" in msg
    assert "'The'" in msg and "0.7000" in msg  # the distribution is printed


def test_depth_exhaustion_raises():
    # Formatting all the way down, mass never recovering: stop at depth 3.
    row = ("\n", False, 0.01, [(0.9, "\n")])
    probe = make_probe({
        "Answer:": row,
        "Answer:\n": row,
        "Answer:\n\n": row,
        "Answer:\n\n\n": row,
    })
    with pytest.raises(RuntimeError) as exc:
        discover_answer_prefill(probe, "BASE")
    assert "depth 3" in str(exc.value)
    assert len(probe.calls) == 4  # seed probe + one per appended level


def test_option_argmax_below_floor_raises():
    # Top token IS an option but the three options still hold under half the
    # mass: extending cannot help, and appending an option would be absurd.
    probe = make_probe({"Answer:": (" Yes", True, 0.3, [(0.3, " Yes")])})
    with pytest.raises(RuntimeError) as exc:
        discover_answer_prefill(probe, "BASE")
    assert "an option" in str(exc.value)


@pytest.mark.parametrize(
    "text,expect",
    [
        (" **", True), ("*", True), ("_", True), ("`", True), (":", True),
        ("\n", True), ("  ", True), ("~", True),
        (" Yes", False), ("The", False), ("", False), (" *Y", False),
    ],
)
def test_is_formatting_token(text, expect):
    assert is_formatting_token(text) is expect
