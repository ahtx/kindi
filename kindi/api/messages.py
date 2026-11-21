# SPDX-License-Identifier: AGPL-3.0-only
"""Translate the Anthropic Messages API onto Kindi's chat path.

The translation is stateless and intentionally narrow: text, tool_use,
tool_result and function declarations are supported; image/audio/document
blocks and extended thinking are refused by name instead of being silently
ignored. Everything else is a re-shaping of the same chat request that
``Kindi.generate`` already serves.
"""
import json
import uuid

_REJECTED = {
    "thinking": "extended thinking is not implemented",
    "mcp_servers": "MCP servers are not implemented",
    "container": "containers are not implemented",
    "context_management": "context management is not implemented",
}


def _text(value, where):
    """Anthropic text from a string or a list of text blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for part in value:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                kind = part.get("type")
                if kind == "text":
                    if not isinstance(part.get("text"), str):
                        raise ValueError("%s has a text block without a string text" % where)
                    out.append(part["text"])
                elif kind in ("image", "document", "audio", "video"):
                    raise ValueError("%s: %s input is not supported by this server" % (where, kind))
                else:
                    raise ValueError("%s: unsupported content block %r" % (where, kind))
            else:
                raise ValueError("%s has a content block that is not an object" % where)
        return "".join(out)
    raise ValueError("%s must be a string or a list of text blocks" % where)


def messages(body):
    out = []
    system = body.get("system")
    if system is not None and system != "":
        out.append({"role": "system", "content": _text(system, "system")})
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty list")
    for i, item in enumerate(raw):
        where = "messages[%d]" % i
        if not isinstance(item, dict):
            raise ValueError("%s must be an object" % where)
        role = item.get("role")
        if role not in ("user", "assistant"):
            raise ValueError("%s has unsupported role %r" % (where, role))
        content = item.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise ValueError("%s.content must be a string or a list of blocks" % where)
        text = []
        tool_calls = []
        for j, block in enumerate(content):
            bwhere = "%s.content[%d]" % (where, j)
            if isinstance(block, str):
                text.append(block)
                continue
            if not isinstance(block, dict):
                raise ValueError("%s must be an object" % bwhere)
            kind = block.get("type")
            if kind == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError("%s has no string text" % bwhere)
                text.append(block["text"])
            elif kind == "tool_use":
                if role != "assistant":
                    raise ValueError("%s: tool_use is only valid in an assistant message" % bwhere)
                if not isinstance(block.get("id"), str) or not block["id"]:
                    raise ValueError("%s needs a tool_use id" % bwhere)
                if not isinstance(block.get("name"), str) or not block["name"]:
                    raise ValueError("%s needs a tool_use name" % bwhere)
                args = block.get("input", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                tool_calls.append({"id": block["id"], "type": "function",
                                   "function": {"name": block["name"], "arguments": args}})
            elif kind == "tool_result":
                if role != "user":
                    raise ValueError("%s: tool_result is only valid in a user message" % bwhere)
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("%s needs a tool_use_id" % bwhere)
                out.append({"role": "tool", "tool_call_id": call_id,
                            "content": _text(block.get("content"), bwhere + ".content")})
            elif kind in ("image", "document", "audio", "video"):
                raise ValueError("%s: %s input is not supported by this server" % (bwhere, kind))
            elif kind == "thinking":
                raise ValueError("%s: extended thinking is not implemented" % bwhere)
            else:
                raise ValueError("%s: unsupported content block %r" % (bwhere, kind))
        if text or tool_calls:
            msg = {"role": role, "content": "".join(text)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
    if not out or all(m["role"] == "system" for m in out):
        raise ValueError("messages must contain a user or assistant turn")
    return out


def tools(value):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("tools must be a list")
    out = []
    for i, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise ValueError("tools[%d] must be an object" % i)
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tools[%d] needs a name" % i)
        fn = {"name": name}
        if tool.get("description") is not None:
            fn["description"] = tool["description"]
        if tool.get("input_schema") is not None:
            fn["parameters"] = tool["input_schema"]
        out.append({"type": "function", "function": fn})
    return out


def tool_choice(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("tool_choice must be an object")
    kind = value.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool":
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice needs a name for type tool")
        return {"type": "function", "function": {"name": name}}
    raise ValueError("tool_choice.type must be auto, any or tool")


def to_chat(body):
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    for key, why in _REJECTED.items():
        if body.get(key) not in (None, False, [], {}):
            raise ValueError("%s is not supported: %s" % (key, why))
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    req = {"messages": messages(body), "max_tokens": max_tokens,
           "stream": bool(body.get("stream"))}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if body.get(src) is not None:
            req[dst] = body[src]
    if body.get("stop_sequences") is not None:
        if not isinstance(body["stop_sequences"], list):
            raise ValueError("stop_sequences must be a list")
        req["stop"] = body["stop_sequences"]
    chat_tools = tools(body.get("tools"))
    if chat_tools:
        req["tools"] = chat_tools
        choice = tool_choice(body.get("tool_choice"))
        if choice is not None:
            req["tool_choice"] = choice
    return req


def _message_id():
    return "msg_" + uuid.uuid4().hex[:24]


def _tool_input(arguments):
    try:
        value = json.loads(arguments or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def content_blocks(info):
    blocks = []
    text = info.get("content") or ""
    if text or not info.get("tool_calls"):
        blocks.append({"type": "text", "text": text})
    for tc in info.get("tool_calls") or []:
        fn = tc.get("function") or {}
        blocks.append({"type": "tool_use", "id": tc.get("id") or ("toolu_%d" % len(blocks)),
                       "name": fn.get("name") or "", "input": _tool_input(fn.get("arguments"))})
    return blocks


def stop_reason(info):
    if info.get("tool_calls"):
        return "tool_use"
    if info.get("finish") == "length":
        return "max_tokens"
    return "end_turn"


def usage(info):
    u = info.get("usage") or {}
    prompt = int(u.get("prompt_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    return {"input_tokens": max(0, prompt - cached), "output_tokens": int(u.get("completion_tokens") or 0),
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached}


def message_object(info, model_id, message_id=None):
    return {"id": message_id or _message_id(), "type": "message", "role": "assistant",
            "model": model_id, "content": content_blocks(info), "stop_reason": stop_reason(info),
            "stop_sequence": None, "usage": usage(info)}


def event(name, data):
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(data, ensure_ascii=False))
