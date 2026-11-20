"""JSON readers that never leave a pack file open.

``json.load(open(path))`` hands the handle to the garbage collector. That is usually
harmless, but on Windows a reader that still holds a pack file can block a concurrent
``kindi convert`` or a cache rewrite. Read pack and cache JSON through ``load_json``
instead; it opens UTF-8 text and closes the handle before returning.
"""
import json


def load_json(path, encoding="utf-8", **kwargs):
    """Load one JSON document from ``path`` and close the file before returning."""
    with open(path, "r", encoding=encoding) as handle:
        return json.load(handle, **kwargs)
