"""Chat template renderer for Inkling-2-Mini GGUF files (compatible with Python 3.6 + Jinja2 3.0)."""
import json
import re

from jinja2 import ChainableUndefined, Environment

from .gguf import GGUF

_compiled = {}
_GEN_TAG_RE = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}")


def _raise(message):
    raise ValueError(str(message))


def _tojson(value, indent=None, sort_keys=False, separators=None):
    kw = {"ensure_ascii": False, "sort_keys": bool(sort_keys)}
    if indent is not None:
        kw["indent"] = indent
    if separators is not None:
        kw["separators"] = tuple(separators)
    return json.dumps(value, **kw)


def environment():
    env = Environment(undefined=ChainableUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    return env


def _template(model_path):
    if model_path not in _compiled:
        kv = GGUF(model_path).kv
        src = kv.get("tokenizer.chat_template")
        if not src:
            raise ValueError("kindi: the GGUF has no chat template")
        # Inkling's template wraps assistant turns in {% generation %}...{% endgeneration %}, a HuggingFace/minja
        # training tag not present in Jinja2 3.0.
        src = _GEN_TAG_RE.sub("", src)
        _compiled[model_path] = environment().from_string(src)
    return _compiled[model_path]


def _normalize_messages(messages):
    """Canonicalize tool_calls[*].function.arguments from JSON strings to dicts (required by Inkling's template)."""
    out = []
    for m in messages:
        calls = m.get("tool_calls")
        if calls:
            m = dict(m)
            fixed = []
            for c in calls:
                c = dict(c)
                f = dict(c.get("function") or {})
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        f["arguments"] = json.loads(args or "{}")
                    except ValueError:
                        f["arguments"] = {}
                elif args is None:
                    f["arguments"] = {}
                c["function"] = f
                fixed.append(c)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def render(model_path, messages, tools=None, enable_thinking=True, reasoning_effort=None, add_generation_prompt=True):
    values = {"messages": _normalize_messages(messages), "add_generation_prompt": bool(add_generation_prompt)}
    if reasoning_effort is not None:
        values["reasoning_effort"] = reasoning_effort
    elif not enable_thinking:
        values["reasoning_effort"] = 0
    if tools:
        values["tools"] = tools
    try:
        return _template(model_path).render(**values)
    except Exception as e:  # noqa: BLE001
        raise ValueError("kindi: the Inkling-2 chat template failed: %s" % e)
