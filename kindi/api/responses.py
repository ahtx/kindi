# SPDX-License-Identifier: AGPL-3.0-only
"""Translate the OpenAI Responses API onto Kindi's chat path.

This is deliberately a stateless adapter. It maps ``POST /v1/responses`` input,
tools and generation fields to the request shape ``Kindi.generate`` already
understands, then maps the finished chat result back to a Responses object. It
does not store responses, so ``previous_response_id`` and ``store`` are refused
instead of being silently ignored.

The module has no FastAPI or engine import, so the translation is testable on a
CPU-only host and on Python 3.6.
"""
import json
import time
import uuid

_REJECTED = {
    "previous_response_id": "stateless: send the previous turns in input",
    "store": "stateless: Kindi does not persist responses",
    "background": "background responses are not implemented",
    "truncation": "truncation is not implemented; keep input within the context",
    "include": "include is not implemented",
    "reasoning": "reasoning configuration is not implemented",
    "max_tool_calls": "max_tool_calls is not implemented",
    "conversation": "conversations are not implemented; send input",
    "prompt": "prompt templates are not implemented; send input",
}


def _text(value, where):
    """Extract text from a Responses content string or input_text block list."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for part in value:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                kind = part.get("type")
                if kind in ("input_text", "output_text", "text"):
                    if not isinstance(part.get("text"), str):
                        raise ValueError("%s has a text block without a string text" % where)
                    out.append(part["text"])
                elif kind in ("input_image", "image_url"):
                    raise ValueError("%s: image input is not supported by this server" % where)
                else:
                    raise ValueError("%s: unsupported content block %r" % (where, kind))
            else:
                raise ValueError("%s has a content block that is not an object" % where)
        return "".join(out)
    raise ValueError("%s must be a string or a list of text blocks" % where)


def messages(body):
    """Responses ``instructions`` + ``input`` as OpenAI chat messages."""
    raw = body.get("input")
    out = []
    if isinstance(raw, str):
        out.append({"role": "user", "content": raw})
    elif isinstance(raw, list):
        for i, item in enumerate(raw):
            where = "input[%d]" % i
            if isinstance(item, str):
                out.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                raise ValueError("%s must be an object or a string" % where)
            kind = item.get("type")
            if kind == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("%s needs a function_call_output call_id" % where)
                out.append({"role": "tool", "tool_call_id": call_id,
                            "content": _text(item.get("output"), where + ".output")})
                continue
            if kind == "function_call":
                call_id = item.get("call_id") or item.get("id")
                name = item.get("name")
                arguments = item.get("arguments", "{}")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("%s needs a function_call call_id" % where)
                if not isinstance(name, str) or not name:
                    raise ValueError("%s needs a function_call name" % where)
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                out.append({"role": "assistant", "content": "",
                            "tool_calls": [{"id": call_id, "type": "function",
                                            "function": {"name": name, "arguments": arguments}}]})
                continue
            if kind not in (None, "message"):
                raise ValueError("%s: unsupported input item %r" % (where, kind))
            role = item.get("role") or "user"
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError("%s has unsupported role %r" % (where, role))
            if role == "tool":
                call_id = item.get("call_id") or item.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("%s: a tool message needs call_id" % where)
                out.append({"role": "tool", "tool_call_id": call_id,
                            "content": _text(item.get("content"), where + ".content")})
            else:
                msg = {"role": role, "content": _text(item.get("content"), where + ".content")}
                if role == "assistant" and item.get("tool_calls"):
                    calls = []
                    for j, tc in enumerate(item["tool_calls"]):
                        fn = tc.get("function") if isinstance(tc, dict) else None
                        if not isinstance(fn, dict) or not fn.get("name"):
                            raise ValueError("%s.tool_calls[%d] needs a function name" % (where, j))
                        calls.append({"id": tc.get("id") or "call_%d" % j, "type": "function",
                                      "function": {"name": fn["name"],
                                                   "arguments": fn.get("arguments") or "{}"}})
                    msg["tool_calls"] = calls
                out.append(msg)
    else:
        raise ValueError("input must be a string or a list of message items")

    instructions = body.get("instructions")
    if instructions is not None and instructions != "":
        text = instructions if isinstance(instructions, str) else _text(instructions, "instructions")
        out.insert(0, {"role": "system", "content": text})
    if not out:
        raise ValueError("input must contain at least one message")
    return out


def tools(value):
    """Responses function tools are flatter than chat-completions tools."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("tools must be a list")
    out = []
    for i, tool in enumerate(value):
        if not isinstance(tool, dict):
            raise ValueError("tools[%d] must be an object" % i)
        if tool.get("type") != "function":
            raise ValueError("tools[%d]: only function tools are supported" % i)
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tools[%d] needs a name" % i)
        row = {"name": name}
        if fn.get("description") is not None:
            row["description"] = fn["description"]
        if fn.get("parameters") is not None:
            row["parameters"] = fn["parameters"]
        out.append({"type": "function", "function": row})
    return out


def tool_choice(value):
    if value is None or value in ("auto", "none", "required"):
        return value
    if isinstance(value, dict):
        if value.get("type") == "function" and isinstance(value.get("name"), str):
            return {"type": "function", "function": {"name": value["name"]}}
        if isinstance(value.get("function"), dict) and value["function"].get("name"):
            return value
    raise ValueError("tool_choice must be auto, none, required or a named function")


def to_chat(body):
    """Translate a Responses request body into a chat-completions body.

    Unknown generation controls are refused by their Responses name, so a client
    cannot believe it changed behaviour that is silently ignored.
    """
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    for key, why in _REJECTED.items():
        if body.get(key) not in (None, False, [], {}):
            raise ValueError("%s is not supported: %s" % (key, why))

    req = {"messages": messages(body), "stream": bool(body.get("stream"))}
    max_out = body.get("max_output_tokens")
    if max_out is not None:
        if not isinstance(max_out, int) or isinstance(max_out, bool) or max_out < 1:
            raise ValueError("max_output_tokens must be a positive integer")
        req["max_tokens"] = max_out
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"),
                     ("top_k", "top_k"), ("seed", "seed")):
        if body.get(src) is not None:
            req[dst] = body[src]
    if body.get("parallel_tool_calls") is not None:
        req["parallel_tool_calls"] = bool(body["parallel_tool_calls"])
    chat_tools = tools(body.get("tools"))
    if chat_tools:
        req["tools"] = chat_tools
        choice = tool_choice(body.get("tool_choice"))
        if choice is not None:
            req["tool_choice"] = choice
    text = body.get("text")
    if text is not None:
        if not isinstance(text, dict):
            raise ValueError("text must be an object")
        fmt = text.get("format")
        if fmt is not None and not (isinstance(fmt, dict) and fmt.get("type") == "text"):
            raise ValueError("text.format is not supported by this PR; use plain text")
    return req


def _new(prefix):
    return prefix + uuid.uuid4().hex[:24]


def _tool_items(info, created):
    out = []
    for i, tc in enumerate(info.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args = fn.get("arguments", "{}")
        if not isinstance(args, str):
            args = json.dumps(args)
        call_id = tc.get("id") or "call_%d" % i
        out.append({"id": _new("fc_"), "type": "function_call", "status": "completed",
                    "call_id": call_id, "name": fn.get("name") or "", "arguments": args})
    return out


def output_items(info):
    items = []
    text = info.get("content") or ""
    if text or not info.get("tool_calls"):
        items.append({"id": _new("msg_"), "type": "message", "role": "assistant",
                      "status": "completed",
                      "content": [{"type": "output_text", "text": text, "annotations": []}]})
    items.extend(_tool_items(info, None))
    return items


def new_id(prefix):
    """A fresh response/item id (the server uses this so streamed and final ids agree)."""
    return _new(prefix)


def response_object(info, model_id, response_id=None, created=None, request=None):
    """Build the non-streaming Responses object from a chat ``done`` event."""
    response_id = response_id or _new("resp_")
    created = int(created if created is not None else time.time())
    usage = info.get("usage") or {}
    status = "incomplete" if info.get("finish") == "length" else "completed"
    obj = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "model": model_id,
        "output": output_items(info),
        "parallel_tool_calls": bool((request or {}).get("parallel_tool_calls", True)),
        "tool_choice": (request or {}).get("tool_choice", "auto"),
        "tools": (request or {}).get("tools") or [],
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "input_tokens_details": {"cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "metadata": (request or {}).get("metadata") or {},
    }
    return obj


def in_progress(response_id, model_id, created, request=None):
    """The response object carried by ``response.created`` / ``response.in_progress``."""
    obj = response_object({}, model_id, response_id, created, request)
    obj.update(status="in_progress", incomplete_details=None, output=[],
               usage=None, error=None)
    return obj


def event(name, data):
    """One Responses SSE event."""
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(data, ensure_ascii=False))
