# SPDX-License-Identifier: AGPL-3.0-only
"""Error-envelope unit and HTTP tests (T5.1)."""

import pytest

from kindi.api import errors


def test_envelope_has_the_openai_shape():
    body = errors.error_body("bad thing", error_type="invalid_request_error", param="messages")
    error = body["error"]
    assert set(error) == {"message", "type", "param", "code"}
    assert error["message"] == "bad thing"
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "messages"
    assert error["code"] == "invalid_request_error"


def test_context_length_exceeded_keeps_its_code():
    status, body, _headers = errors.response(
        errors.OpenAIError("too long", status=400, code="context_length_exceeded",
                           param="messages"))
    assert status == 400
    assert body["error"]["code"] == "context_length_exceeded"
    assert body["error"]["param"] == "messages"


def test_plain_message_becomes_a_400_envelope():
    status, body, _headers = errors.response("nope")
    assert status == 400
    assert body["error"]["message"] == "nope"
    assert body["error"]["param"] is None


def test_authentication_error_keeps_its_type():
    status, body, _headers = errors.response(
        errors.OpenAIError("invalid API key", status=401,
                           error_type="authentication_error"))
    assert status == 401
    assert body["error"]["type"] == "authentication_error"
