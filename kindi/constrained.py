# SPDX-License-Identifier: AGPL-3.0-only
"""Which `response_format` types constrain decoding (REQ-0018).

One place for the list, so validation, generation and the backend-capability gate
cannot disagree about what structured output means.

No imports, so this is safe from every module (including the server's lazy imports).
"""

STRUCTURED_TYPES = ("json_object", "json_schema", "grammar")


def is_structured(response_format):
    """True when a response_format asks for decode-side constraint."""
    return isinstance(response_format, dict) and response_format.get("type") in STRUCTURED_TYPES


__all__ = ["STRUCTURED_TYPES", "is_structured"]
