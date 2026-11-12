"""The chat template of a Nemotron-H MoE GGUF (from the file), rendered with jinja2 as llama.cpp and Hugging Face
render it: system / user / assistant / tool messages, the thinking block (<think>), and tool calls in the XML form
(<tool_call><function=name><parameter=key>value</parameter></function></tool_call>)."""
import json

from jinja2 import ChainableUndefined, Environment

from .gguf import GGUF

_compiled = {}


def _raise(message):
    raise ValueError(str(message))


def _items(value):
    """the |items filter of Jinja2 3.1 (the template iterates tool parameters with it); the image has Jinja2 3.0"""
    return list(value.items()) if hasattr(value, "items") else []


def environment():
    env = Environment(undefined=ChainableUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    env.filters["items"] = _items
    env.globals["raise_exception"] = _raise
    return env


def _template(model_path):
    if model_path not in _compiled:
        kv = GGUF(model_path).kv
        src = kv.get("tokenizer.chat_template")
        if not src:
            raise ValueError("kindi: the GGUF has no chat template")
        _compiled[model_path] = environment().from_string(src)
    return _compiled[model_path]


def _args(messages):
    """tool-call arguments as a mapping (the template iterates them); clients send a JSON string"""
    out = []
    for m in messages:
        calls = m.get("tool_calls")
        if calls:
            m = dict(m)
            fixed = []
            for c in calls:
                c = dict(c)
                f = dict(c.get("function") or {})
                if isinstance(f.get("arguments"), str):
                    try:
                        f["arguments"] = json.loads(f["arguments"] or "{}")
                    except ValueError:
                        f["arguments"] = {}
                c["function"] = f
                fixed.append(c)
            m["tool_calls"] = fixed
        out.append(m)
    return out


def render(model_path, messages, tools=None, enable_thinking=True, add_generation_prompt=True):
    values = {"messages": _args(messages), "add_generation_prompt": bool(add_generation_prompt),
              "enable_thinking": bool(enable_thinking)}
    if tools:
        values["tools"] = tools
    try:
        return _template(model_path).render(**values)
    except Exception as e:  # noqa: BLE001
        raise ValueError("kindi: the Lightning chat template failed: %s" % e)
