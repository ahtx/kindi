"""The OpenAI error body every Kindi endpoint returns (PR #58, atimelessrage):

    {"error": {"message": str, "type": str, "param": str or None, "code": str or None}}

OpenAI's SDKs and litellm read type and code; code is the type unless a more exact one is known (the one that client
libraries act on is context_length_exceeded, a 400 with param "messages"). param names the request field at fault,
or is None. No FastAPI import, so the CPU tests and kindi route use it as well."""


def body(message, typ="invalid_request_error", code=None, param=None):
    return {"error": {"message": str(message), "type": typ, "param": param, "code": typ if code is None else code}}


def overflow(message):
    """the body of a prompt that does not fit the context"""
    return body(message, "invalid_request_error", "context_length_exceeded", "messages")
