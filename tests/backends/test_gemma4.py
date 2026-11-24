# SPDX-License-Identifier: AGPL-3.0-only
"""Gemma 4 adapter tests: registry identity, load wiring and generation."""

import pytest

from kindi.backends import registry


def test_gemma4_resolves_by_model_id_unloaded():
    from kindi.backends.gemma4 import Gemma4Backend

    backend = registry.resolve("gemma4-e4b")
    assert isinstance(backend, Gemma4Backend)
    assert backend.loaded is False


def test_gemma4_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("gemma4-e4b", config={"pack": "no-such-pack-dir"})


def test_gemma4_model_id_for_manifest():
    from kindi.backends.gemma4 import model_id_for_manifest

    assert model_id_for_manifest({"backend": "kindi_gemma4"}) == "gemma4-e4b"
    assert model_id_for_manifest({"backend": "kindi_dsv41"}) is None
    assert model_id_for_manifest({}) is None
    assert model_id_for_manifest(None) is None


def test_gemma4_load_needs_the_model_path(tmp_path):
    (tmp_path / "kindi_pack.json").write_text('{"backend": "kindi_gemma4"}', encoding="utf-8")
    with pytest.raises(registry.BackendUnavailable, match="model_path"):
        registry.resolve("gemma4-e4b", config={"pack": str(tmp_path)})


def test_gemma4_load_wires_pack_ctx_two_card_choice_tokenizer_and_stop_ids(tmp_path, monkeypatch):
    """`load()` is executable CPU-side even though no Gemma 4 pack exists here.

    The engine/tokenizer modules are recording fakes, so this proves the wiring
    a first GPU run would otherwise be the first to discover: the manifest gains
    the pack path, the requested ctx reaches the chosen engine class, a two-card
    request picks `TPEngine`, the tokenizer is built from the manifest's
    `model_path`, and the adapter's stop ids come from that tokenizer.
    """
    import json
    import sys
    import types

    from kindi.backends.gemma4 import Gemma4Backend

    calls = {}

    class _Tok(object):
        bos, eos = 2, 1
        tokens = ["<eos>", "a", "b", "c"]
        stop_ids = {1, 9}

        def __init__(self, model_path):
            calls["tok"] = model_path

    class _FakeEngine(object):
        def __init__(self, man, ctx=None, log=None):
            calls["engine"] = (man, ctx, log)
            self.ctx = 4096                  # the loader clamps 16384 down to what fits
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeTPEngine(_FakeEngine):
        pass

    gemma4_mod = types.ModuleType("kindi.gemma4")
    gemma4_mod.Engine = _FakeEngine
    tp_mod = types.ModuleType("kindi.gemma4_tp")
    tp_mod.TPEngine = _FakeTPEngine
    tok_mod = types.ModuleType("kindi.gemma4_tokenizer")
    tok_mod.Tok = _Tok
    monkeypatch.setitem(sys.modules, "kindi.gemma4", gemma4_mod)
    monkeypatch.setitem(sys.modules, "kindi.gemma4_tp", tp_mod)
    monkeypatch.setitem(sys.modules, "kindi.gemma4_tokenizer", tok_mod)
    (tmp_path / "kindi_pack.json").write_text(json.dumps({
        "backend": "kindi_gemma4", "model": "gemma4-e4b", "model_path": "models/gemma4-e4b.gguf",
    }), encoding="utf-8")

    for tp_mode, engine_class in (("0", _FakeEngine), ("1", _FakeTPEngine)):
        monkeypatch.setenv("KINDI_G4_TP", tp_mode)
        calls.clear()
        logged = []
        log_sink = logged.append
        backend = Gemma4Backend()
        loaded = backend.load({"pack": str(tmp_path), "ctx": 16384, "log": log_sink})
        assert loaded is backend
        assert type(backend.engine) is engine_class
        man, ctx, log = calls["engine"]
        assert ctx == 16384
        assert log == log_sink
        assert man["pack_path"] == str(tmp_path)
        assert man["model_path"] == "models/gemma4-e4b.gguf"
        assert calls["tok"] == "models/gemma4-e4b.gguf"
        assert backend.ctx == 4096
        assert backend.stop_ids == {1, 9}
        assert backend.loaded is True
        engine = backend.engine
        backend.close()
        assert engine.closed is True
        assert backend.loaded is False


class _FakeG4Cp(object):
    """The two cupy calls the adapter makes (argmax/asnumpy), on numpy."""

    @staticmethod
    def argmax(z):
        import numpy as np

        class _Item(object):
            def __init__(self, v):
                self.v = v

            def item(self):
                return self.v

        return _Item(int(np.argmax(z)))

    @staticmethod
    def asnumpy(z):
        import numpy as np

        return np.asarray(z, dtype=np.float32)


class _FakeG4Engine(object):
    """Logits: token `peak` after the prompt, `peak + 1` on every later step."""

    def __init__(self, peak=5):
        self.cp = _FakeG4Cp()
        self.calls = []
        self.peak = peak
        self.steps = 0

    def reset(self):
        self.calls.append(("reset",))
        self.steps = 0

    def forward_batch(self, tokens, pos):
        self.calls.append(("forward_batch", list(tokens), pos))

    def forward(self, token, pos, logits=True):
        self.calls.append(("forward", int(token), pos, logits))
        self.steps += 1
        z = [0.0] * 8
        z[self.peak if self.steps == 1 else self.peak + 1] = 9.0
        return z


def _gemma4(engine):
    from kindi.backends.gemma4 import Gemma4Backend

    class _Tok(object):
        bos, eos = 2, 1
        tokens = ["", "a", "b", "c"]
        stop_ids = {1}

        def decode_bytes(self, i):
            return ("b" if i == 5 else "x" if i == 6 else "").encode()

        def decode(self, ids):
            return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8")

    backend = Gemma4Backend()
    backend.loaded = True
    backend.ctx = 64
    backend.tok = _Tok()
    backend.engine = engine
    backend.stop_ids = set(_Tok.stop_ids)
    return backend


def test_gemma4_generate_prefills_in_one_batch_and_steps_its_own_position():
    from kindi.backends.base import SamplingParams

    eng = _FakeG4Engine(peak=5)
    events = list(_gemma4(eng).generate([10, 11, 12], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6, 6]
    assert events[-1].finish_reason == "length"
    assert eng.calls == [
        ("reset",),
        ("forward_batch", [10, 11], 0),
        ("forward", 12, 2, True),        # the last prompt token returns the first logits
        ("forward", 5, 3, True),         # generated token n sits at len(prompt) + n
        ("forward", 6, 4, True),
    ]


def test_gemma4_generate_stops_on_a_stop_id_and_on_stop_text():
    from kindi.backends.base import SamplingParams

    class _Breaks(_FakeG4Engine):
        def __init__(self):
            _FakeG4Engine.__init__(self, peak=1)      # 1 is eos

    events = list(_gemma4(_Breaks()).generate([10], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == []
    assert events[-1].finish_reason == "stop"

    class _Text(_FakeG4Engine):
        def __init__(self):
            _FakeG4Engine.__init__(self, peak=5)      # decodes as "b", then "x" -> "bx"

    backend = _gemma4(_Text())
    events = list(backend.generate([10], SamplingParams(max_tokens=4, stop=("bx",))))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6]
    assert events[-1].finish_reason == "stop"


def test_gemma4_token_filter_argmaxes_over_the_allowed_ids_at_temperature_0():
    from kindi.backends.base import SamplingParams

    eng = _FakeG4Engine(peak=5)                       # logits favour 5
    params = SamplingParams(max_tokens=1, token_filter=lambda st: {6})
    events = list(_gemma4(eng).generate([10], params))
    assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_gemma4_token_filter_receives_the_prompt():
    from kindi.backends.base import SamplingParams

    backend = _gemma4(_FakeG4Engine(peak=5))
    seen = []

    def filt(state):
        seen.append((list(state["prompt_ids"]), list(state["generated"])))
        return {6}

    events = list(backend.generate([10, 11, 12], SamplingParams(max_tokens=1, token_filter=filt)))
    assert seen == [([10, 11, 12], [])]
    assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_gemma4_apply_chat_template_needs_a_loaded_pack():
    from kindi.backends.gemma4 import Gemma4Backend

    with pytest.raises(registry.BackendUnavailable, match="no model loaded"):
        Gemma4Backend().apply_chat_template([{"role": "user", "content": "hi"}])
