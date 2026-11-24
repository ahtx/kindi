# SPDX-License-Identifier: AGPL-3.0-only
"""DeepSeek V4.1 adapter tests: registry identity, load wiring and generation."""

import pytest

from kindi.backends import registry


def test_dsv41_resolves_by_model_id_unloaded():
    from kindi.backends.dsv41 import DeepseekV41Backend

    backend = registry.resolve("deepseek-v41-flash")
    assert isinstance(backend, DeepseekV41Backend)
    assert backend.loaded is False


def test_dsv41_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("deepseek-v41-flash", config={"pack": "no-such-pack-dir"})


def test_dsv41_model_id_for_manifest():
    from kindi.backends.dsv41 import model_id_for_manifest

    assert model_id_for_manifest({"backend": "kindi_dsv41"}) == "deepseek-v41-flash"
    assert model_id_for_manifest({"backend": "kindi_gemma4"}) is None
    assert model_id_for_manifest({}) is None
    assert model_id_for_manifest(None) is None


def test_dsv41_load_needs_the_model_path_and_says_what_to_add(tmp_path):
    (tmp_path / "kindi_pack.json").write_text('{"backend": "kindi_dsv41"}', encoding="utf-8")
    with pytest.raises(registry.BackendUnavailable, match="model_path"):
        registry.resolve("deepseek-v41-flash", config={"pack": str(tmp_path)})


def test_dsv41_load_wires_model_path_ctx_tmax_tokenizer_and_stop_id(tmp_path, monkeypatch):
    """`load()` is executable CPU-side even though no DeepSeek V4.1 weights exist here."""
    import json
    import sys
    import types

    from kindi.backends.dsv41 import DeepseekV41Backend

    calls = {}

    class _Tok(object):
        vocab = ["<eos>", "a", "b"]

    class _FakeModel(object):
        def __init__(self, path, ctx=None, tmax=None):
            calls["model"] = (path, ctx, tmax)
            self.ctx = 2048                  # the engine clamps 4096 down to what fits
            self.tok = _Tok()
            self.eos = 760

    dsv41_mod = types.ModuleType("kindi.dsv41")
    dsv41_mod.Model = _FakeModel
    monkeypatch.setitem(sys.modules, "kindi.dsv41", dsv41_mod)
    (tmp_path / "kindi_pack.json").write_text(json.dumps({
        "backend": "kindi_dsv41", "model": "DeepSeek-V4.1-Flash",
        "model_path": "shards/deepseek-00001.gguf",
    }), encoding="utf-8")

    backend = DeepseekV41Backend()
    loaded = backend.load({"pack": str(tmp_path), "ctx": 4096, "tmax": 16,
                           "log": lambda line: None})
    assert loaded is backend
    assert calls["model"] == ("shards/deepseek-00001.gguf", 4096, 16)
    assert isinstance(backend.engine, _FakeModel)
    assert backend.ctx == 2048
    assert backend.tok is backend.engine.tok
    assert backend.stop_ids == {760}
    assert backend.loaded is True
    assert backend.info()["ctx"] == 2048


class _FakeDsTok(object):
    vocab = ["", "a", "b", "c"]

    def encode(self, text, special=True):
        return [9]

    def decode_bytes(self, i):
        return ("b" if i == 5 else "x" if i == 6 else "").encode()

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8")


class _FakeDsEngine(object):
    """Model.forward: the whole prompt once, then one token per step (numpy logits)."""

    def __init__(self, peak=5):
        import numpy as np

        self._np = np
        self.peak = peak
        self.calls = []
        self.steps = 0
        self.tok = _FakeDsTok()
        self.eos = 1
        self.ctx = 64

    def reset(self):
        self.calls.append(("reset",))
        self.steps = 0

    def forward(self, ids):
        import numpy as np

        self.calls.append(("forward", list(ids)))
        self.steps += 1
        z = np.zeros(8, dtype=np.float32)
        z[self.peak if self.steps == 1 else self.peak + 1] = 9.0
        return z


def _dsv41(engine):
    from kindi.backends.dsv41 import DeepseekV41Backend

    backend = DeepseekV41Backend()
    backend.loaded = True
    backend.ctx = 64
    backend.engine = engine
    backend.tok = engine.tok
    backend.stop_ids = {engine.eos}
    return backend


def test_dsv41_generate_feeds_the_prompt_at_once_then_one_token_per_step():
    from kindi.backends.base import SamplingParams

    eng = _FakeDsEngine(peak=5)
    events = list(_dsv41(eng).generate([10, 11, 12], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6, 6]
    assert events[-1].finish_reason == "length"
    assert eng.calls == [("reset",), ("forward", [10, 11, 12]), ("forward", [5]), ("forward", [6])]


def test_dsv41_generate_stops_on_eos_and_token_filter():
    from kindi.backends.base import SamplingParams

    eng = _FakeDsEngine(peak=1)                       # 1 is eos
    events = list(_dsv41(eng).generate([10], SamplingParams(max_tokens=3)))
    assert [e.token_id for e in events if e.kind == "token"] == []
    assert events[-1].finish_reason == "stop"

    eng = _FakeDsEngine(peak=5)
    params = SamplingParams(max_tokens=1, token_filter=lambda st: {6})
    events = list(_dsv41(eng).generate([10], params))
    assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_dsv41_generate_stops_on_a_stop_string_split_across_two_pieces():
    """dsv41's own pieces are "b" then "x" through _FakeDsTok, so "bx" only exists in the
    accumulated text - the case a per-piece check would miss."""
    from kindi.backends.base import SamplingParams

    events = list(_dsv41(_FakeDsEngine(peak=5)).generate(
        [10], SamplingParams(max_tokens=4, stop=("bx",))))
    assert [e.token_id for e in events if e.kind == "token"] == [5, 6]
    assert events[-1].finish_reason == "stop"


def test_dsv41_token_filter_receives_the_prompt():
    from kindi.backends.base import SamplingParams

    backend = _dsv41(_FakeDsEngine(peak=5))
    seen = []

    def filt(state):
        seen.append((list(state["prompt_ids"]), list(state["generated"])))
        return {6}

    events = list(backend.generate([10, 11, 12], SamplingParams(max_tokens=1, token_filter=filt)))
    assert seen == [([10, 11, 12], [])]
    assert [e.token_id for e in events if e.kind == "token"] == [6]


def test_dsv41_apply_chat_template_needs_a_loaded_pack():
    from kindi.backends.dsv41 import DeepseekV41Backend

    with pytest.raises(registry.BackendUnavailable, match="no model loaded"):
        DeepseekV41Backend().apply_chat_template([{"role": "user", "content": "hi"}])
