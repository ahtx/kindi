"""Request-level tests for the response_format/GBNF server path.

These tests use a stub engine and a scripted sampler, so they need no GPU or model
pack. They pin the boundary the follow-up PR adds: validation/compilation before the
engine, the backend-capability gate, and constrained generation through the real
`Kindi.generate` loop and HTTP route.
"""
import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gbnf_fixtures import CANONICAL_JSON
from kindi import server
from kindi.sampling import Sampler


class _FakeEngine(object):
    ctx = 4096

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _FakeRunner(object):
    def __init__(self):
        self.eng = _FakeEngine()
        self.tokens = [1, 2, 3]
        self.ckpt = {0: object()}


class _FakeKindi(object):
    cp = None
    supports_logprobs = False
    supports_json_grammar = True
    native = False
    nanbeige = None
    variant = "bonsai27"

    def __init__(self):
        self.cfg = {"model_name": "bonsai2-2b", "api_key": ""}
        self.pack_id = "fake"
        self.last_req = None
        self.eng = _FakeEngine()
        self.runner = _FakeRunner()
        self.lock = threading.Lock()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
                     "read": 0, "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.calls = 0
        self.t0 = 0.0

    def head_active(self):
        return "exact"

    def encode_text(self, text, special=True):
        return [ord(c) % 256 for c in str(text)] or [1]

    def decode_ids(self, ids):
        return "x" * len(ids)

    def after_turn(self, locked=False):
        return None

    def acquire(self, req):
        return self

    def release(self, view):
        return None

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def generate(self, req, cancel=None):
        if not isinstance(req.get("messages"), list):
            raise TypeError("messages must be a list")
        for m in req["messages"]:
            if not isinstance(m, dict) or "role" not in m:
                raise TypeError("missing role")
        cached = len(self.runner.tokens)
        self.calls += 1
        self.last_req = req
        yield "text", "ok"
        yield "done", {
            "content": "ok", "tool_calls": [], "reasoning": "", "finish": "stop", "logprobs": None,
            "usage": {"prompt_tokens": cached, "completion_tokens": 1, "total_tokens": cached + 1,
                      "prompt_tokens_details": {"cached_tokens": cached}},
            "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 0.0},
        }


@pytest.fixture
def app_client(monkeypatch, tmp_path):
    pack = tmp_path / "text-pack"
    pack.mkdir()
    (pack / "kindi_pack.json").write_text('{"model": "bonsai2-2b"}', encoding="utf-8")
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "pack": str(pack)},
                            log=lambda *a: None)
    with TestClient(app) as client:
        yield client, K


def _post(client, response_format=None, **extra):
    body = {"messages": [{"role": "user", "content": "hello"}]}
    if response_format is not None:
        body["response_format"] = response_format
    body.update(extra)
    return client.post("/v1/chat/completions", json=body)


def test_response_format_types_are_accepted_at_the_request_boundary(app_client):
    client, K = app_client
    formats = (
        {"type": "text"},
        {"type": "json_object"},
        {"type": "json_schema", "json_schema": {"name": "x", "schema": {
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}},
        {"type": "grammar", "grammar": 'root ::= "ok" | "no"'},
    )
    for i, rf in enumerate(formats):
        r = _post(client, rf)
        assert r.status_code == 200, r.text
        assert K.calls == i + 1


def test_invalid_response_formats_are_400_and_never_reach_the_engine(app_client):
    client, K = app_client
    cases = (
        "json_object",
        {"type": "yaml"},
        {"type": "grammar"},
        {"type": "grammar", "grammar": "root ::= ["},
        {"type": "grammar", "grammar": 'root ::= "[" root "]" | "x"'},
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {"schema": "x"}},
        {"type": "json_schema", "json_schema": {"schema": {"type": "frob"}}},
    )
    for rf in cases:
        r = _post(client, rf)
        assert r.status_code == 400, (rf, r.text)
        assert r.json()["error"]["param"] == "response_format", (rf, r.text)
    assert K.calls == 0


def test_tools_and_structured_output_are_mutually_exclusive(app_client):
    client, K = app_client
    r = _post(client, {"type": "json_object"},
              tools=[{"type": "function", "function": {"name": "f"}}])
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "response_format"
    assert K.calls == 0


def test_recursive_grammar_opt_in_reaches_the_compiler(app_client, monkeypatch):
    client, K = app_client
    recursive = 'root ::= "[" root "]" | "x"'
    monkeypatch.delenv("KINDI_GBNF_MAX_DEPTH", raising=False)
    r = _post(client, {"type": "grammar", "grammar": recursive})
    assert r.status_code == 400, r.text
    assert "recursive" in r.json()["error"]["message"]
    assert K.calls == 0

    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", "3")
    r = _post(client, {"type": "grammar", "grammar": recursive})
    assert r.status_code == 200, r.text
    assert K.calls == 1


def test_structured_output_is_rejected_on_a_backend_without_the_hook(app_client):
    client, K = app_client
    K.supports_json_grammar = False
    K.variant = "lightning"
    r = _post(client, {"type": "json_object"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "response_format"
    assert K.calls == 0


def test_a_streamed_structured_request_is_validated_before_sse(app_client):
    client, K = app_client
    r = _post(client, {"type": "json_object"}, stream=True)
    assert r.status_code == 200, r.text
    assert "[DONE]" in r.text
    assert K.calls == 1

    r = _post(client, {"type": "grammar"}, stream=True)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "response_format"
    assert K.calls == 1


class _StopTok(object):
    """A character codec for driving the real generate() loop without a model."""

    vocab = ["v%d" % i for i in range(256)]

    def encode(self, s):
        return [ord(c) % 256 for c in str(s)] or [1]

    def decode_bytes(self, t):
        return bytes([t % 256])


class _StopEngine(object):
    ctx = 4096
    hsl = None
    shortlist = 0


class _StopRunner(object):
    def prepare(self, *a, **k):
        return None, 0

    def step(self, tok):
        return None

    def idle(self):
        return None


def _constrained_kindi(script):
    """A real `Kindi` whose tokens come from `script`: every pick must be inside the
    sampler's allowed set, so a token the constraint forbids is never emitted."""
    K = object.__new__(server.Kindi)
    K.cfg = {"model_name": "fake", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
             "max_tokens": 512, "repeat_stop": False, "fast_clusters": 0}
    K.native = False
    K.nanbeige = None
    K.variant = None
    K.supports_json_grammar = True
    K.tok = _StopTok()
    K.eng = _StopEngine()
    K.runner = _StopRunner()
    K.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
              "read": 0, "generated": 0, "started": 0.0}
    K.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
    K.log = lambda *a: None
    K.head_mode = lambda: "exact"

    def pick(*a, **k):
        allowed = set(k.get("allowed").tolist()) if k.get("allowed") is not None else None
        for i, t in enumerate(script):
            if allowed is None or t in allowed:
                script.pop(i)
                return t
        raise AssertionError("no scripted token fits the grammar")

    K.sample = pick
    K._pick = pick
    return K


def test_http_constrained_reply_never_emits_a_forbidden_byte(app_client, monkeypatch):
    """The constraint is decode-side: a scripted byte the grammar forbids is skipped
    through the real `Kindi.generate` loop, while the same script without a grammar
    emits it."""
    client, K = app_client
    doc = '{"a": 1}'
    # \x01 is in no part of the canonical JSON grammar, so it cannot become legal later.
    script = [ord(c) for c in doc[:6]] + [1] + [ord(c) for c in doc[6:]] + [248046]

    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", "3")
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "60000")
    K.generate = _constrained_kindi(list(script)).generate
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        "response_format": {"type": "grammar", "grammar": CANONICAL_JSON},
    })
    assert r.status_code == 200, r.text
    content = r.json()["choices"][0]["message"]["content"]
    assert content == doc, content
    assert "\x01" not in content

    # the control must actually contain the byte, or the equality above proves nothing
    K.generate = _constrained_kindi(list(script)).generate
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
    })
    assert r.status_code == 200, r.text
    assert "\x01" in r.json()["choices"][0]["message"]["content"]


def test_http_json_schema_is_constrained_and_compiled_once(app_client, monkeypatch):
    client, K = app_client
    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"]}
    doc = '{"name": "Ada"}'
    script = [ord(c) for c in doc] + [248046]
    K.generate = _constrained_kindi(list(script)).generate

    calls = []
    from kindi import jsongrammar

    real = jsongrammar.grammar_from_schema

    def counted(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(jsongrammar, "grammar_from_schema", counted)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
    })
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"name": "Ada"}
    assert len(calls) == 1


class _ConstrainedHostEngine(object):
    def __init__(self):
        self.ylog = np.array([2.0, 1.0, 0.0])
        self.ylog_n = 3
        self.ylog_map = None


class _RecordingRng(object):
    def __init__(self):
        self.idx = None

    def choice(self, idx, p=None):
        self.idx = [int(i) for i in np.asarray(idx)]
        return self.idx[0]


def test_sampler_allowed_set_overrides_greedy():
    # token 1 is allowed, token 0 is the argmax: the grammar wins
    assert Sampler(_ConstrainedHostEngine())(0, 0.0, 1.0, 0, None, allowed=[1]) == 1


def test_sampler_allowed_set_with_temperature_stays_inside():
    rng = _RecordingRng()
    out = Sampler(_ConstrainedHostEngine())(0, 1.0, 1.0, 0, rng, allowed=[1, 2])
    assert set(rng.idx) == {1, 2}
    assert out in (1, 2)


def test_sampler_empty_allowed_intersection_is_a_loud_error():
    with pytest.raises(ValueError, match="no allowed token"):
        Sampler(_ConstrainedHostEngine())(0, 0.0, 1.0, 0, None, allowed=[])


def test_structured_output_does_not_enter_the_thinking_channel(app_client):
    client, K = app_client
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"]}
    doc = '{"a": "ok"}'
    K.generate = _constrained_kindi([ord(c) for c in doc] + [248046]).generate
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        "enable_thinking": True,
        "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
    })
    assert r.status_code == 200, r.text
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"a": "ok"}


def test_gemma_json_schema_is_constrained(monkeypatch):
    # The top logit is always on '<' (never grammar-legal); the pick must skip it.
    from kindi import gemma4_chat

    texts = ["{", '"ok"', ":", "true", "}", "<", "!", "<eos>"]
    eos = 7

    def logits(peak, second=None):
        z = np.full(8, -10.0)
        z[peak] = 100.0
        if second is not None:
            z[second] = 50.0
        return z

    script = [logits(5, 0), logits(5, 1), logits(5, 2),
              logits(5, 3), logits(5, 4), logits(7)]

    class Tok(object):
        tokens = list(texts)
        stop_ids = {7}
        ids = {}
        eos = 7

        def encode(self, s):
            return [1]

        def decode_bytes(self, i):
            return self.tokens[int(i)].encode("utf-8")

    class Engine(object):
        ctx = 4096
        shortlist = 0
        fast_centroids = None

        def __init__(self):
            self.script = list(script)

        def reset(self):
            pass

        def forward_batch(self, ids, pos):
            pass

        def forward(self, tok, pos, logits=False):
            return self.script.pop(0)

    class Cache(object):
        def prepare(self, ids):
            return 0

        def advanced(self, ids, checkpoint=False):
            pass

        def append(self, tok):
            pass

    class Np(object):
        asnumpy = staticmethod(np.asarray)
        float32 = np.float32

        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr(gemma4_chat, "render", lambda *a, **k: "prompt")
    K = object.__new__(server.Kindi)
    K.cfg = {"model_name": "fake-gemma", "temperature": 0.0, "top_p": 1.0, "top_k": 0,
             "max_tokens": 64, "fast_clusters": 0, "head": "exact"}
    K.native = True
    K.variant = "gemma4"
    K.tok = Tok()
    K.eng = Engine()
    K.g4cache = Cache()
    K.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
              "read": 0, "generated": 0, "started": 0.0}
    K.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
    K.log = lambda *a: None
    K.cp = Np()
    K.head_mode = lambda: "exact"
    K.model_path = ""

    items = list(K._generate_gemma({
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64,
        "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        "response_format": {"type": "json_schema", "json_schema": {"schema": {
            "type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}},
    }))
    done = next(v for kind, v in items if kind == "done")
    assert done["finish"] == "stop"
    assert json.loads(done["content"]) == {"ok": True}
