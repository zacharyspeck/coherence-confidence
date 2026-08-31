"""A minimal GPT-2-shaped tokenizer so the tokenization logic can be tested
offline, with no model download. Implements only the surface `src.score` uses.

`Ġ` marks a leading space, exactly as in a byte-level BPE vocabulary.
"""

from __future__ import annotations


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        if vocab is None:
            vocab = {"<unk>": 0}
            nxt = 1
            for word in ("Yes", "No", "Unsure"):
                for form in (word, word.lower(), word.upper()):
                    for piece in (form, "Ġ" + form):
                        if piece not in vocab:
                            vocab[piece] = nxt
                            nxt += 1
            # Distractors: substring matches that must NOT be swept in by the
            # vocab scan, and a newline-prefixed form that is not a leading-space
            # variant.
            for piece in (
                "the",
                "Ġthe",
                "yesterday",
                "Ġyesterday",
                "Nobody",
                "ĠNobody",
                "ĊYes",
            ):
                vocab[piece] = nxt
                nxt += 1
        self._vocab = dict(vocab)
        self._inv = {v: k for k, v in self._vocab.items()}

    # -- surface used by src.score ------------------------------------------

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def tokenize(self, text: str) -> list[str]:
        piece = ("Ġ" + text[1:]) if text.startswith(" ") else text
        if piece in self._vocab:
            return [piece]
        # Unknown: return more than one piece so `_encode_single` rejects it.
        return ["<unk>", "<unk>"]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._vocab.get(tokens, self.unk_token_id)
        return [self._vocab.get(t, self.unk_token_id) for t in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self.convert_tokens_to_ids(self.tokenize(text))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).replace("Ġ", " ").replace("Ċ", "\n")

    def decode(self, ids: list[int]) -> str:
        return self.convert_tokens_to_string([self._inv[i] for i in ids])


class NoYesTokenizer(FakeTokenizer):
    """A tokenizer that cannot represent 'Yes' in a single token."""

    def __init__(self) -> None:
        super().__init__()
        for piece in list(self._vocab):
            if piece.lstrip("ĠĊ").lower().startswith("yes"):
                del self._vocab[piece]
        self._inv = {v: k for k, v in self._vocab.items()}


class CollidingTokenizer(FakeTokenizer):
    """A tokenizer where 'No' and 'Unsure' resolve to a shared id."""

    def __init__(self) -> None:
        super().__init__()
        shared = self._vocab["No"]
        self._vocab["Unsure"] = shared
        self._inv = {v: k for k, v in self._vocab.items()}
