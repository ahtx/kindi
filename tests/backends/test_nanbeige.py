# SPDX-License-Identifier: AGPL-3.0-only
"""Nanbeige adapter tests: registry identity, generation positions and stop handling."""

import pytest

from kindi.backends import registry


def test_nanbeige_resolves_by_model_id_unloaded():
    from kindi.backends.nanbeige import NanbeigeBackend

    backend = registry.resolve("nanbeige-33b")
    assert isinstance(backend, NanbeigeBackend)
    assert backend.loaded is False


def test_nanbeige_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("nanbeige-33b", config={"pack": "no-such-pack-dir"})


def test_nanbeige_model_id_for_manifest():
    from kindi.backends.nanbeige import model_id_for_manifest

    assert model_id_for_manifest({"backend": "kindi_nanbeige"}) == "nanbeige-33b"
    assert model_id_for_manifest({"backend": "kindi_gemma4"}) is None
    assert model_id_for_manifest({}) is None
    assert model_id_for_manifest(None) is None


class _FakeNanbeigeEngine(object):
    """Records the (tok, pos) that generate() feeds, like Engine.generate does."""

    def __init__(self):
        self.calls = []

    def prefill_batch(self, ids, pos0=0):
        self.calls.append(("prefill", list(ids), pos0))
        return 100                     # prediction for position len(ids)

    def step(self, tok, pos):
        self.calls.append(("step", int(tok), int(pos)))
        return 101


def test_nanbeige_generate_feeds_each_token_at_its_own_position():
    """Regression: generate() must step the first generated token at len(prompt),
    not len(prompt) + 1."""
    from kindi.backends.base import SamplingParams
    from kindi.backends.nanbeige import NanbeigeBackend

    class _FakeTok(object):
        bos = 1
        tokens = ["", "a"]
        types = [0, 0]
        eos = None

        def decode(self, ids):
            return "?"

    backend = NanbeigeBackend()
    backend.loaded = True
    backend.engine = _FakeNanbeigeEngine()
    backend.tok = _FakeTok()
    backend.stop_ids = set()

    params = SamplingParams(max_tokens=3)
    events = list(backend.generate([10, 11, 12], params))
    tokens = [e.token_id for e in events if e.kind == "token"]
    assert tokens == [100, 101, 101]
    assert events[-1].kind == "done"
    assert NanbeigeBackend.capabilities.supports_speculation is False
    assert events[-1].drafted == 0 and events[-1].accepted == 0
    assert backend.engine.calls == [
        ("prefill", [10, 11, 12], 0),
        ("step", 100, 3),
        ("step", 101, 4),
    ]


class _FakeNanbeigeTok(object):
    bos = 1
    eos = None
    tokens = [""] * 32
    types = [0] * 32

    def __init__(self, pieces):
        self.pieces = dict(pieces)

    def decode(self, ids):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class _FakeNanbeigeScriptEngine(object):
    """prefill_batch/step plus the logits window _sample_next reads."""

    def __init__(self, script, vocab=32):
        import numpy as np
        from types import SimpleNamespace

        self.script, self.vocab, self.n, self.calls = list(script), vocab, 0, []
        self.card = SimpleNamespace(dev=0, logits=np.full(vocab, -10.0))
        self.cards = [self.card]

        class _Device(object):
            def __init__(self, dev):
                self.dev = dev

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class _Cuda(object):
            Device = _Device

        self.cp = SimpleNamespace(cuda=_Cuda(), asnumpy=np.asarray,
                                  flatnonzero=np.flatnonzero)

    def _logits(self):
        import numpy as np

        z = np.full(self.vocab, -10.0)
        if self.n < len(self.script):
            z[self.script[self.n]] = 10.0
        self.n += 1
        self.card.logits = z
        return z

    def prefill_batch(self, ids, pos0=0):
        self.calls.append(("prefill", list(ids), pos0))
        return int(self._logits().argmax())

    def step(self, tok, pos):
        self.calls.append(("step", int(tok), int(pos)))
        return int(self._logits().argmax())


def _nanbeige(script, pieces=None):
    from kindi.backends.nanbeige import NanbeigeBackend

    backend = NanbeigeBackend()
    backend.loaded = True
    backend.ctx = 4096
    backend.tok = _FakeNanbeigeTok(pieces or {10: "The sky is ", 11: "EN", 12: "D", 13: "blue."})
    backend.engine = _FakeNanbeigeScriptEngine(script)
    backend.stop_ids = set()
    return backend


def test_nanbeige_generate_stops_on_a_stop_string_split_across_two_pieces():
    from kindi.backends.base import SamplingParams

    events = list(_nanbeige([10, 11, 12, 13]).generate(
        [5, 6], SamplingParams(max_tokens=4, stop=("END",))))
    assert [e.token_id for e in events if e.kind == "token"] == [10, 11, 12]
    assert events[-1].finish_reason == "stop"


def test_nanbeige_token_filter_receives_the_prompt():
    from kindi.backends.base import SamplingParams

    seen = []

    def filt(state):
        seen.append(list(state["prompt_ids"]))
        return {10}

    events = list(_nanbeige([10]).generate([5, 6, 7], SamplingParams(max_tokens=1,
                                                                    token_filter=filt)))
    assert seen == [[5, 6, 7]]
    assert [e.token_id for e in events if e.kind == "token"] == [10]
