# SPDX-License-Identifier: AGPL-3.0-only
"""Bonsai adapter tests: registry identity, manifest resolution and generation."""

import pytest

from kindi.backends import registry


def test_bonsai_resolves_by_model_id_unloaded():
    from kindi.backends.bonsai import BonsaiBackend

    for model_id in ("bonsai2-2b", "bonsai2-4b", "bonsai2-9b", "bonsai2-27b"):
        backend = registry.resolve(model_id)
        assert isinstance(backend, BonsaiBackend)
        assert backend.loaded is False


def test_bonsai_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("bonsai2-2b", config={"pack": "no-such-pack-dir"})


def test_bonsai_model_id_for_manifest():
    from kindi.backends.bonsai import model_id_for_manifest

    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-2B"}) == "bonsai2-2b"
    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-27B"}) == "bonsai2-27b"
    # The 27B GGUF names itself only "Hf"; the size must come from source_file,
    # the same fallback config.py uses to name the served model.
    assert model_id_for_manifest({
        "model": "Hf",
        "source_file": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    }) == "bonsai2-27b"
    assert model_id_for_manifest({"model": "Ternary-Bonsai-2-XL"}) is None
    assert model_id_for_manifest({"model": "Other-Model"}) is None
    assert model_id_for_manifest({}) is None


class _FakeBonsaiTok(object):
    """The tokenizer surface BonsaiBackend.generate uses: per-token decode and stop ids."""

    def __init__(self, pieces):
        self.pieces = dict(pieces)
        self.sid = {"<|endoftext|>": 1, "<|im_end|>": 2}

    def decode(self, ids):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class _FakeBonsaiEngine(object):
    """feed/step in the adapter's order: prefill the prompt, then one prediction per step."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def feed(self, tok, pos):
        self.calls.append(("feed", int(tok), int(pos)))

    def step(self, tok, pos):
        self.calls.append(("step", int(tok), int(pos)))
        return self.script.pop(0) if self.script else 2      # 2 is <|im_end|>: a short script ends


def _bonsai(script):
    from kindi.backends.bonsai import BonsaiBackend

    backend = BonsaiBackend()
    backend.loaded = True
    backend.ctx = 4096
    # "END" is split as "EN" + "D" by the tokenizer, which is what the scanner must survive
    backend.tok = _FakeBonsaiTok({10: "The sky is ", 11: "EN", 12: "D", 13: "blue."})
    backend.engine = _FakeBonsaiEngine(script)
    return backend


def test_bonsai_generate_stops_on_a_stop_string_split_across_two_pieces():
    """The StopScanner swap has to hold on the adapter's own decoded pieces.

    "END" arrives as "EN" then "D": neither piece contains the stop, only the
    accumulated text does. A scanner that looked at the new piece alone, or that
    dropped its overlap between hits, would miss it and finish the turn on 13.
    """
    from kindi.backends.base import SamplingParams

    backend = _bonsai([10, 11, 12, 13])
    events = list(backend.generate([5, 6], SamplingParams(max_tokens=4, stop=("END",))))
    assert [e.token_id for e in events if e.kind == "token"] == [10, 11, 12]
    assert events[-1].kind == "done"
    assert events[-1].finish_reason == "stop"
    assert backend.engine.calls == [
        ("reset",),
        ("feed", 5, 0),
        ("step", 6, 1),
        ("step", 10, 2),
        ("step", 11, 3),
    ]

    early = list(_bonsai([10, 11, 12, 13]).generate(
        [5, 6], SamplingParams(max_tokens=4, stop=("EN",))))
    assert [e.token_id for e in early if e.kind == "token"] == [10, 11]
    assert early[-1].finish_reason == "stop"
