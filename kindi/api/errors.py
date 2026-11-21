# SPDX-License-Identifier: AGPL-3.0-only
"""OpenAI-style error envelope for the Kindi API (T5.1).

Every error the server returns has the shape agent frameworks already parse::

    {"error": {"message": str, "type": str, "param": str | None, "code": str | None}}

``type`` keeps the values existing clients see (``invalid_request_error``,
``authentication_error``, ``request_too_large``, ``rate_limit_exceeded``,
``server_busy``, ``server_error``, ``cancelled``). ``code`` defaults to the type
for backwards compatibility; the one exception the API promises is
``context_length_exceeded``, which stays a 400 with that code so client
libraries recognise a context-window error. ``param`` names the request field at
fault when the server knows it, else None.

This module has no FastAPI dependency so it can be imported and unit-tested on a
CPU-only host and on Python 3.6.
"""


class OpenAIError(Exception):
    """An error that maps directly onto the OpenAI error envelope."""

    def __init__(self, message, status=400, error_type="invalid_request_error",
                 code=None, param=None):
        Exception.__init__(self, message)
        self.message = message
        self.status = int(status)
        self.error_type = error_type
        self.code = code if code is not None else error_type
        self.param = param

    def to_dict(self):
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def error_body(message, error_type="invalid_request_error", code=None, param=None):
    """Build the envelope dict for an error."""
    return OpenAIError(message, error_type=error_type, code=code, param=param).to_dict()


def response(error, status=None, headers=None):
    """Convert an OpenAIError (or a plain message) into a framework-neutral
    (status, body, headers) triple. The FastAPI adapter in server.py turns the
    triple into a JSONResponse; keeping the mapping here makes it testable
    without FastAPI."""
    if isinstance(error, OpenAIError):
        return error.status if status is None else status, error.to_dict(), dict(headers or {})
    body = error_body(str(error))
    return (400 if status is None else status), body, dict(headers or {})
