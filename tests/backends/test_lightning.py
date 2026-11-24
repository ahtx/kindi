# SPDX-License-Identifier: AGPL-3.0-only
"""Lightning adapter tests: registry identity, generation and stop handling."""

import pytest

from kindi.backends import registry


def test_lightning_resolves_by_model_id_unloaded():
    from kindi.backends.lightning import LightningBackend

    backend = registry.resolve("nemotron-lightning-30b")
    assert isinstance(backend, LightningBackend)
    assert backend.loaded is False


def test_lightning_load_rejects_a_missing_pack_directory():
    with pytest.raises(registry.BackendUnavailable, match="pack directory"):
        registry.resolve("nemotron-lightning-30b", config={"pack": "no-such-pack-dir"})


def test_lightning_model_id_for_manifest():
    from kindi.backends.lightning import model_id_for_manifest

    assert model_id_for_manifest({"backend": "kindi_lightning"}) == "nemotron-lightning-30b"
    assert model_id_for_manifest({"backend": "kindi_nanbeige"}) is None
    assert model_id_for_manifest({}) is None
    assert model_id_for_manifest(None) is None


class _FakeLightningTok(object):
    bos, eos, vocab, types = 1, 2, ["", "a", "b", "c"], [0, 0, 0, 0]

    def decode(self, ids):
        return "" if not ids else "b" if ids[0] == 101 else "?"


class _FakeLightningEngine(object):
    """Prefill/step plus the candidate window the sampler reads."""

    def __init__(self, cand=(), sc=()):
        import numpy as np
        from types import SimpleNamespace

        self.calls = []
        self.closed = False
        self.cand, self.sc = list(cand), list(sc)
        logits = np.zeros(128, dtype=np.float64)
        for token_id, score in zip(self.cand, self.sc):
            if 0 <= token_id < logits.size:
                logits[token_id] = score
        self.logits = logits
        self.cp = SimpleNamespace(asnumpy=np.asarray, flatnonzero=np.flatnonzero)

    def close(self):
        self.closed = True

    def reset(self):
        self.calls.append(("reset",))

    def prefill(self, ids, pos=None):
        self.calls.append(("prefill", list(ids), pos))
        return 100

    def step(self, tok):
        self.calls.append(("step", int(tok)))
        return 101

    def candidates(self, w):
        import numpy as np

        return np.array(self.cand, dtype=np.int64), np.array(self.sc, dtype=np.float64)


def _lightning(greedy_engine=None):
    from kindi.backends.lightning import LightningBackend

    backend = LightningBackend()
    backend.loaded = True
    backend.ctx = 4096
    backend.tok = _FakeLightningTok()
    backend.engine = greedy_engine or _FakeLightningEngine()
    backend.stop_ids = {backend.tok.eos}
    return backend


def test_lightning_close_releases_its_engine():
    backend = _lightning()
    engine = backend.engine
    backend.close()
    assert engine.closed is True
    assert backend.engine is None
    assert backend.loaded is False


def test_lightning_generate_prefills_resets_and_stops_on_a_control_token():
    from kindi.backends.base import SamplingParams

    backend = _lightning()
    backend.stop_ids = {101}              # the second generated token is a control token
    events = list(backend.generate([10, 11], SamplingParams(max_tokens=4)))
    assert [e.token_id for e in events if e.kind == "token"] == [100, 101]
    assert events[-1].finish_reason == "stop"
    assert backend.engine.calls == [("reset",), ("prefill", [10, 11], 0), ("step", 100)]


def test_lightning_generate_reports_length_and_honours_stop_text():
    from kindi.backends.base import SamplingParams

    plain = list(_lightning().generate([10], SamplingParams(max_tokens=2)))
    assert plain[-1].finish_reason == "length"
    assert [e.token_id for e in plain if e.kind == "token"] == [100, 101]

    by_text = list(_lightning().generate([10], SamplingParams(max_tokens=3, stop=("?",))))
    assert by_text[-1].finish_reason == "stop"


def test_lightning_token_filter_argmaxes_over_the_allowed_candidates():
    from kindi.backends.base import SamplingParams

    engine = _FakeLightningEngine(cand=[7, 8, 9], sc=[3.0, 9.0, 1.0])
    backend = _lightning(engine)
    params = SamplingParams(max_tokens=1, token_filter=lambda state: {7, 9})
    events = list(backend.generate([10], params))
    assert [e.token_id for e in events if e.kind == "token"] == [7]


def test_lightning_token_filter_receives_the_prompt():
    from kindi.backends.base import SamplingParams

    engine = _FakeLightningEngine(cand=[7, 8, 9], sc=[3.0, 9.0, 1.0])
    seen = []

    def filt(state):
        seen.append((list(state["prompt_ids"]), list(state["generated"])))
        return {7, 9}

    list(_lightning(engine).generate([10, 11], SamplingParams(max_tokens=1, token_filter=filt)))
    assert seen[0][0] == [10, 11]


def test_lightning_refuses_a_prompt_longer_than_the_context():
    from kindi.backends.base import SamplingParams

    backend = _lightning()
    backend.ctx = 3
    with pytest.raises(registry.BackendUnavailable, match="context is 3"):
        list(backend.generate([1, 2, 3, 4], SamplingParams(max_tokens=1)))


def test_lightning_apply_chat_template_needs_a_loaded_pack():
    from kindi.backends.lightning import LightningBackend

    with pytest.raises(registry.BackendUnavailable, match="no model loaded"):
        LightningBackend().apply_chat_template([{"role": "user", "content": "hi"}])


class _FakePieceTok(object):
    eos = None

    def __init__(self, pieces):
        self.pieces = dict(pieces)

    def decode(self, ids):
        return "".join(self.pieces.get(int(i), "?") for i in ids)


class _FakeLightningScriptEngine(object):
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def reset(self):
        self.calls.append(("reset",))

    def prefill(self, ids, pos=None):
        self.calls.append(("prefill", list(ids), pos))
        return self.script.pop(0) if self.script else 2

    def step(self, tok):
        self.calls.append(("step", int(tok)))
        return self.script.pop(0) if self.script else 2


def test_lightning_generate_stops_on_a_stop_string_split_across_two_pieces():
    from kindi.backends.base import SamplingParams
    from kindi.backends.lightning import LightningBackend

    backend = LightningBackend()
    backend.loaded = True
    backend.ctx = 4096
    backend.tok = _FakePieceTok({10: "The sky is ", 11: "EN", 12: "D", 13: "blue."})
    backend.engine = _FakeLightningScriptEngine([10, 11, 12, 13])
    backend.stop_ids = set()
    events = list(backend.generate([5, 6], SamplingParams(max_tokens=4, stop=("END",))))
    assert [e.token_id for e in events if e.kind == "token"] == [10, 11, 12]
    assert events[-1].finish_reason == "stop"
