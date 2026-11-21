# SPDX-License-Identifier: AGPL-3.0-only
"""Admission-control tests (no GPU or model pack)."""
import threading

import pytest
from fastapi.testclient import TestClient

from kindi import server
from kindi.admission import Gate, QueueFull, RateLimiter


def test_gate_admits_one_running_plus_the_configured_waiters():
    gate = Gate(2)
    for _ in range(3):
        gate.enter()
    assert gate.waiting == 2
    with pytest.raises(QueueFull, match="KINDI_QUEUE_MAX=2"):
        gate.enter()
    assert gate.rejected == 1
    gate.release()
    gate.enter()
    assert gate.waiting == 2


def test_gate_release_is_safe_to_repeat():
    gate = Gate(0)
    gate.enter()
    gate.release()
    gate.release()
    assert gate.waiting == 0


def test_rate_limiter_refuses_with_a_retry_after():
    limiter = RateLimiter(1, window_s=60)
    assert limiter.check("a") == 0
    retry = limiter.check("a")
    assert retry >= 1
    assert limiter.rejected == 1
    assert limiter.check("b") == 0       # another client has its own budget


def test_rate_limiter_is_disabled_at_zero():
    limiter = RateLimiter(0)
    for _ in range(10):
        assert limiter.check("a") == 0


class _FakeEngine(object):
    ctx = 4096


class _FakeKindi(object):
    cp = None

    def __init__(self):
        self.eng = _FakeEngine()
        self.lock = threading.RLock()
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0,
                     "read": 0, "generated": 0, "started": 0.0}
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0}
        self.t0 = 0.0

    def head_active(self):
        return "exact"

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def check_prompt(self, req):
        return [1]

    def generate(self, req, cancel=None):
        yield "text", "ok"
        yield "done", {"content": "ok", "tool_calls": [], "reasoning": "", "finish": "stop",
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                       "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 1.0}}

    def after_turn(self, locked=False):
        return None

    def acquire(self, req):
        return self

    def release(self, view):
        return None


def _client(monkeypatch, tmp_path, **cfg):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda config, log: K)
    base = {"model_name": "bonsai2-2b", "api_key": "", "pack": str(tmp_path)}
    base.update(cfg)
    app = server.create_app(base, log=lambda *a: None)
    return TestClient(app)


def test_oversized_body_returns_413_before_the_engine(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, max_body_mb=1)
    body = {"messages": [{"role": "user", "content": "x" * (1024 * 1024 + 100)}]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 413, r.text
    assert r.json()["error"]["type"] == "request_too_large"


def test_per_client_rate_limit_returns_429_with_retry_after(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path, rate_limit_per_min=1)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    first = c.post("/v1/chat/completions", json=body)
    assert first.status_code == 200, first.text
    second = c.post("/v1/chat/completions", json=body)
    assert second.status_code == 429, second.text
    assert int(second.headers["Retry-After"]) >= 1
    assert second.json()["error"]["type"] == "rate_limit_exceeded"


class _BlockingKindi(_FakeKindi):
    def __init__(self):
        super(_BlockingKindi, self).__init__()
        self.held = threading.Event()
        self.go = threading.Event()

    def check_prompt(self, req):
        self.held.set()
        assert self.go.wait(10), "first request was not released"
        return [1]


def test_queue_full_returns_429_with_retry_after(monkeypatch, tmp_path):
    K = _BlockingKindi()
    monkeypatch.setattr(server, "_make_engine", lambda config, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "pack": str(tmp_path),
                             "queue_max": 0}, log=lambda *a: None)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    first = {}

    with TestClient(app) as c1, TestClient(app) as c2:
        def run_first():
            try:
                first["response"] = c1.post("/v1/chat/completions", json=body)
            except BaseException as e:  # surfaced after the thread joins
                first["error"] = e

        thread = threading.Thread(target=run_first)
        thread.start()
        try:
            assert K.held.wait(5), "first request never reached the prompt check"
            r = c2.post("/v1/chat/completions", json=body)
            assert r.status_code == 429, r.text
            assert r.headers["Retry-After"] == "1"
            assert r.json()["error"]["type"] == "server_busy"
        finally:
            K.go.set()
            thread.join(10)

    assert not thread.is_alive()
    assert "error" not in first, first
    assert first["response"].status_code == 200, first["response"].text


def test_queue_limit_is_refused_before_scheduling(monkeypatch, tmp_path):
    gate = Gate(0)
    gate.enter()                       # hold the only slot, as a running request would
    assert gate.waiting == 0
    with pytest.raises(QueueFull):
        gate.enter()
    gate.release()
