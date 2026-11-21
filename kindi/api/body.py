# SPDX-License-Identifier: AGPL-3.0-only
"""Read a JSON request body without buffering more than a configured limit.

The admission PR adds the same cap to the chat and tokenizer routes. This
module keeps an adapter route honest on its own: merging it without that PR
still refuses an oversized upload instead of letting the server read until the
client stops.
"""
import json


class RequestTooLarge(ValueError):
    """The JSON body exceeds the configured byte limit."""


async def read_json_body(request, limit):
    """Read and parse JSON, rejecting a body larger than ``limit`` bytes.

    The declared Content-Length is checked first; a chunked or lying request is
    still bounded by counting the bytes as they arrive.
    """
    declared = request.headers.get("content-length")
    if declared:
        try:
            declared = int(declared)
        except ValueError:
            raise ValueError("the request has an invalid Content-Length")
        if declared > limit:
            raise RequestTooLarge("the request body exceeds %d bytes" % limit)
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise RequestTooLarge("the request body exceeds %d bytes" % limit)
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ValueError("the body is not JSON")
