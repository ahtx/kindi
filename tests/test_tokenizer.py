"""Tokenizer failure behavior on a synthetic Qwen3.5 vocabulary."""

import pytest

from kindi.tokenizer import Tok, _B2U

ADDED = 248044  # the first added-token slot Tok reads


def make_tok(tmp_path):
    """A minimal valid vocabulary and an empty merges table."""
    vocab = [_B2U[b] for b in range(256)] + [""] * (ADDED - 256) + ["<|im_start|>"]
    merges = tmp_path / "merges.txt"
    merges.write_text("", encoding="utf-8")
    return Tok(vocab, str(merges))


def test_a_missing_piece_is_not_silently_encoded_as_token_zero(tmp_path):
    tok = make_tok(tmp_path)
    del tok.sid["c"]  # a corrupt pack: the byte token is absent

    with pytest.raises(ValueError, match="c"):
        tok.encode("c")
