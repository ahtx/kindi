# SPDX-License-Identifier: AGPL-3.0-only
"""Anthropic Messages translation and HTTP tests (no GPU or model pack)."""
import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from kindi import config, server
from kindi.api import messages
from kindi.api.body import RequestTooLarge, read_json_body


def test_system_messages_tools_and_tool_results_translate():
    body = {
        "model": "claude-test",
        "max_tokens": 64,
        "system": [{"type": "text", "text": "be brief"}],
        "stop_sequences": ["END"],
        "tools": [{"name": "lookup", "description": "find", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "tool", "name": "lookup"},
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"x": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "42"}]},
            ]},
        ],
    }
    req = messages.to_chat(body)
    assert req["messages"][0] == {"role": "system", "content": "be brief"}
    assert req["messages"][1] == {"role": "user", "content": "hi"}
    assert req["messages"][2]["tool_calls"][0]["function"]["name"] == "lookup"
    assert req["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "42"}
    assert req["max_tokens"] == 64
    assert req["stop"] == ["END"]
    assert req["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert req["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_image_and_extended_thinking_are_refused():
    with pytest.raises(ValueError, match="image"):
        messages.to_chat({"max_tokens": 8, "messages": [
            {"role": "user", "content": [{"type": "image", "source": {}}]}]})
    with pytest.raises(ValueError, match="extended thinking"):
        messages.to_chat({"max_tokens": 8, "thinking": {"type": "enabled"},
                          "messages": [{"role": "user", "content": "hi"}]})


def test_message_object_maps_tool_use_and_usage():
    info = {"content": "ok", "tool_calls": [
        {"id": "toolu_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"x\": 1}"}}],
        "finish": "tool_calls",
        "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                  "prompt_tokens_details": {"cached_tokens": 2}}}
    obj = messages.message_object(info, "claude-test", "msg_1")
    assert obj["id"] == "msg_1"
    assert obj["stop_reason"] == "tool_use"
    assert obj["content"][0] == {"type": "text", "text": "ok"}
    assert obj["content"][1]["input"] == {"x": 1}
    assert obj["usage"] == {"input_tokens": 3, "output_tokens": 2,
                            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 2}


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
        self.calls = 0

    def head_active(self):
        return "exact"

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def check_prompt(self, req):
        self.calls += 1
        return [1]

    def generate(self, req, cancel=None):
        yield "text", "hello"
        yield "done", {"content": "hello", "tool_calls": [], "reasoning": "", "finish": "stop",
                       "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                       "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 1.0}}

    def after_turn(self, locked=False):
        return None


def _client(monkeypatch, **cfg):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda config, log: K)
    base = {"model_name": "bonsai2-2b", "api_key": ""}
    base.update(cfg)
    return TestClient(server.create_app(base, log=lambda *a: None)), K


def test_http_messages_non_streaming(monkeypatch):
    c, _ = _client(monkeypatch)
    r = c.post("/v1/messages", json={"max_tokens": 8,
                                    "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["stop_reason"] == "end_turn"
    assert body["content"] == [{"type": "text", "text": "hello"}]


def test_http_messages_stream_has_anthropic_event_order(monkeypatch):
    c, _ = _client(monkeypatch)
    with c.stream("POST", "/v1/messages", json={
            "max_tokens": 8, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}) as r:
        assert r.status_code == 200, r.text
        text = "".join(r.iter_text())
    order = ["event: message_start", "event: content_block_start", "event: content_block_delta",
             "event: content_block_stop", "event: message_delta", "event: message_stop"]
    pos = [text.index(name) for name in order]
    assert pos == sorted(pos)
    assert '"type": "message_stop"' in text


def test_http_messages_accepts_x_api_key(monkeypatch):
    c, _ = _client(monkeypatch, api_key="test-key")
    body = {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/messages", json=body).status_code == 401
    assert c.post("/v1/messages", json=body, headers={"x-api-key": "wrong"}).status_code == 401
    r = c.post("/v1/messages", json=body, headers={"x-api-key": "test-key"})
    assert r.status_code == 200, r.text


class _ChunkedStream(object):
    def __init__(self):
        self._chunks = iter([b'{"max_tokens": 8, "messages": [', b"x" * 2048])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration


class _ChunkedRequest(object):
    headers = {}

    def stream(self):
        return _ChunkedStream()


def test_config_accepts_max_body_mb(tmp_path):
    path = tmp_path / "kindi.conf"
    path.write_text("[kindi]\nmax_body_mb = 2\n", encoding="utf-8")
    assert config.load(str(path))["max_body_mb"] == 2


def test_body_reader_caps_chunked_uploads_too():
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(RequestTooLarge):
            loop.run_until_complete(read_json_body(_ChunkedRequest(), 1024))
    finally:
        loop.close()


def test_http_messages_rejects_oversized_body_before_the_engine(monkeypatch):
    c, K = _client(monkeypatch, max_body_mb=1)
    r = c.post("/v1/messages", json={
        "max_tokens": 8, "messages": [{"role": "user", "content": "x" * (1024 * 1024 + 100)}]})
    assert r.status_code == 413, r.text
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "request_too_large"
    assert "exceeds" in r.json()["error"]["message"]
    assert K.calls == 0


def test_http_messages_refuses_thinking_with_anthropic_error(monkeypatch):
    c, _ = _client(monkeypatch)
    r = c.post("/v1/messages", json={"max_tokens": 8, "thinking": {"type": "enabled"},
                                    "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400, r.text
    assert r.json()["type"] == "error"
    assert "extended thinking" in r.json()["error"]["message"]
