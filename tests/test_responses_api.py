# SPDX-License-Identifier: AGPL-3.0-only
"""Responses API translation and HTTP tests (no GPU or model pack)."""
import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from kindi import config, server
from kindi.api import responses
from kindi.api.body import RequestTooLarge, read_json_body


def test_input_string_instructions_and_tools_translate_to_chat():
    req = responses.to_chat({
        "input": "hello",
        "instructions": "be brief",
        "max_output_tokens": 32,
        "tools": [{"type": "function", "name": "lookup",
                   "description": "find a value", "parameters": {"type": "object"}}],
        "tool_choice": {"type": "function", "name": "lookup"},
    })
    assert req["messages"][0] == {"role": "system", "content": "be brief"}
    assert req["messages"][1] == {"role": "user", "content": "hello"}
    assert req["max_tokens"] == 32
    assert req["tools"][0]["function"]["name"] == "lookup"
    assert req["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_message_items_and_function_outputs_round_trip():
    req = responses.to_chat({"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "go"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": ""}],
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}]},
        {"type": "function_call_output", "call_id": "call_1", "output": "42"},
    ]})
    assert req["messages"][0]["content"] == "go"
    assert req["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup"
    assert req["messages"][2] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}


@pytest.mark.parametrize("field,value", [
    ("previous_response_id", "resp_x"),
    ("store", True),
    ("background", True),
    ("text", {"format": {"type": "json_object"}}),
    ("input", [{"type": "input_image", "image_url": "x"}]),
])
def test_unsupported_responses_fields_are_refused(field, value):
    body = {"input": "hi"}
    body[field] = value
    with pytest.raises(ValueError):
        responses.to_chat(body)


def test_response_object_maps_text_and_tool_calls():
    info = {"content": "ok", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"x\": 1}"}}],
        "finish": "tool_calls", "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    obj = responses.response_object(info, "bonsai2-2b", "resp_1", 1, {"parallel_tool_calls": True})
    assert obj["id"] == "resp_1"
    assert obj["output"][0]["content"][0]["text"] == "ok"
    assert obj["output"][1]["type"] == "function_call"
    assert obj["output"][1]["call_id"] == "call_1"
    assert obj["usage"]["input_tokens"] == 3


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
        self.fail_after_text = False

    def head_active(self):
        return "exact"

    def drafting(self):
        return {"enabled": False, "method": None, "drafter": None, "last": None}

    def progress(self):
        return None

    def after_turn(self, locked=False):
        return None

    def check_prompt(self, req):
        self.calls += 1
        return [1]

    def generate(self, req, cancel=None):
        info = {"content": "hello", "tool_calls": [], "reasoning": "", "finish": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "timing": {"prompt_s": 0.0, "gen_s": 0.0, "gen_tok_s": 1.0}}
        yield "text", "hello"
        if self.fail_after_text:
            raise RuntimeError("engine exploded")
        yield "done", info


@pytest.fixture
def client(monkeypatch, tmp_path):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "", "pack": str(tmp_path)},
                            log=lambda *a: None)
    with TestClient(app) as c:
        yield c, K


def test_http_responses_non_streaming(client):
    c, K = client
    r = c.post("/v1/responses", json={"input": "hello"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "hello"
    assert K.calls == 1


def test_http_responses_stream_has_created_delta_and_completed(client):
    c, _ = client
    with c.stream("POST", "/v1/responses", json={"input": "hello", "stream": True}) as r:
        assert r.status_code == 200, r.text
        text = "".join(r.iter_text())
    assert "event: response.created" in text
    assert "event: response.output_text.delta" in text
    assert "event: response.completed" in text
    assert '"hello"' in text


def test_http_responses_stream_emits_failed_when_the_engine_raises(client):
    c, K = client
    K.fail_after_text = True
    with c.stream("POST", "/v1/responses", json={"input": "hello", "stream": True}) as r:
        assert r.status_code == 200, r.text
        text = "".join(r.iter_text())
    assert "event: response.created" in text
    assert "event: response.output_text.delta" in text
    assert "event: response.failed" in text
    assert '"code": "server_error"' in text
    assert "kindi: engine exploded" in text
    assert "event: response.completed" not in text
    assert K.calls == 1


def test_http_responses_rejects_stateless_previous_response(client):
    c, K = client
    r = c.post("/v1/responses", json={"input": "hello", "previous_response_id": "resp_1"})
    assert r.status_code == 400, r.text
    assert "previous_response_id" in r.json()["error"]["message"]
    assert K.calls == 0


class _ChunkedStream(object):
    def __init__(self):
        self._chunks = iter([b'{"input": "', b"x" * 2048])

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


def test_http_responses_rejects_oversized_body_before_the_engine(monkeypatch, tmp_path):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "",
                             "pack": str(tmp_path), "max_body_mb": 1},
                            log=lambda *a: None)
    with TestClient(app) as c:
        r = c.post("/v1/responses", json={"input": "x" * (1024 * 1024 + 100)})
    assert r.status_code == 413, r.text
    assert r.json()["error"]["code"] == "request_too_large"
    assert K.calls == 0


def test_http_responses_requires_auth_when_configured(monkeypatch, tmp_path):
    K = _FakeKindi()
    monkeypatch.setattr(server, "_make_engine", lambda cfg, log: K)
    app = server.create_app({"model_name": "bonsai2-2b", "api_key": "test-key",
                             "pack": str(tmp_path)}, log=lambda *a: None)
    with TestClient(app) as c:
        assert c.post("/v1/responses", json={"input": "hello"}).status_code == 401
        r = c.post("/v1/responses", json={"input": "hello"},
                   headers={"Authorization": "Bearer test-key"})
        assert r.status_code == 200, r.text
