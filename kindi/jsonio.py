"""JSON reads that close their file at once (PR #50, atimelessrage).

json.load(open(path)) leaves the handle to the garbage collector. On Windows an open handle on a pack or cache file
can block a `kindi convert` or a cache rewrite that runs at the same time. The engine load path reads its pack and
cache JSON through load_json, which opens the file as UTF-8 and closes it before it returns."""
import json


def load_json(path, encoding="utf-8"):
    """the JSON document in `path`; the file is closed when this returns"""
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)
