"""The Anthropic Messages API (POST /v1/messages, /v1/messages/count_tokens) over Kindi's chat path (PR #64,
atimelessrage).

A Messages request becomes a chat completions request; the server runs it through the same endpoint as an OpenAI
client (the same slots, queue, priority, prompt cache, tool-call parser and thinking), always streamed inside, and
this module turns the OpenAI chunks into Anthropic's answer or its events.

- In: system (text or text blocks), user/assistant turns with text, tool_use and tool_result blocks, tools
  (name, description, input_schema), tool_choice auto/any/tool/none, max_tokens (required), temperature, top_p,
  top_k, stop_sequences, thinking {"type": "enabled", "budget_tokens": N} (the model's thinking switch and budget).
  Thinking blocks sent back in assistant turns are dropped (the chat template does not take old thinking).
  cache_control and metadata are accepted and have no effect. Image and document blocks get HTTP 400: the models
  read text only.
- Out: content blocks thinking, text, tool_use in that order; stop_reason end_turn, max_tokens or tool_use (a stop
  sequence shows as end_turn); usage input_tokens, output_tokens and cache_read_input_tokens (prompt-cache hits).
- Stream: message_start, content_block_start/delta/stop (thinking_delta, text_delta, input_json_delta),
  message_delta, message_stop; a ping event while a long prompt is read; an error event if the engine fails.
- Errors: {"type": "error", "error": {"type", "message"}} with Anthropic's types and status codes.
- Auth: x-api-key or Authorization: Bearer, when the server has an api_key.

No FastAPI import; standard library only."""
import json
import uuid

ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
               404: "not_found_error", 413: "request_too_large", 429: "rate_limit_error", 500: "api_error",
               529: "overloaded_error"}


class BadRequest(ValueError):
    pass


def error(status, message):
    """(status, body) of an Anthropic error"""
    return status, {"type": "error", "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message}}


def from_openai_error(status, body):
    """an OpenAI error body from the chat path, as an Anthropic error; server_busy is Anthropic's overloaded 529"""
    e = (body or {}).get("error") or {}
    if e.get("code") == "server_busy":
        status = 529
    return error(status, str(e.get("message") or body))


def _text(v, where):
    if isinstance(v, str):
        return v
    if not isinstance(v, list):
        raise BadRequest("%s: a string or a list of text blocks" % where)
    out = []
    for i, b in enumerate(v):
        if isinstance(b, str):
            out.append(b)
            continue
        t = b.get("type") if isinstance(b, dict) else None
        if t == "text" and isinstance(b.get("text"), str):
            out.append(b["text"])
        elif t in ("image", "document", "search_result"):
            raise BadRequest("%s[%d]: %s blocks are not supported; the model reads text only" % (where, i, t))
        else:
            raise BadRequest("%s[%d]: a %r block is not supported here" % (where, i, t))
    return "".join(out)


def to_chat(body):
    """the chat completions request for a Messages request (raises BadRequest)"""
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    mt = body.get("max_tokens")
    if not isinstance(mt, int) or isinstance(mt, bool) or mt < 1:
        raise BadRequest("max_tokens: a positive integer is required")
    out = []
    if body.get("system") not in (None, "", []):
        out.append({"role": "system", "content": _text(body["system"], "system")})
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("messages: a non-empty list is required")
    for i, m in enumerate(msgs):
        w = "messages[%d]" % i
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ("user", "assistant"):
            raise BadRequest("%s.role must be user or assistant" % w)
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": role, "content": c})
            continue
        if not isinstance(c, list):
            raise BadRequest("%s.content: a string or a list of blocks" % w)
        text, calls = [], []
        for j, b in enumerate(c):
            bw = "%s.content[%d]" % (w, j)
            t = b.get("type") if isinstance(b, dict) else None
            if t == "text" and isinstance(b.get("text"), str):
                text.append(b["text"])
            elif t in ("thinking", "redacted_thinking"):
                continue
            elif t == "tool_use" and role == "assistant":
                if not b.get("id") or not b.get("name"):
                    raise BadRequest("%s: tool_use needs id and name" % bw)
                calls.append({"id": b["id"], "type": "function", "function": {
                    "name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
            elif t == "tool_result" and role == "user":
                if not b.get("tool_use_id"):
                    raise BadRequest("%s: tool_result needs tool_use_id" % bw)
                r = _text(b.get("content") if b.get("content") is not None else "", bw + ".content")
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                            "content": ("Error: " + r) if b.get("is_error") else r})
            elif t in ("image", "document", "search_result"):
                raise BadRequest("%s: %s blocks are not supported; the model reads text only" % (bw, t))
            else:
                raise BadRequest("%s: a %r block is not supported in a %s turn" % (bw, t, role))
        if text or calls:
            mm = {"role": role, "content": "".join(text)}
            if calls:
                mm["tool_calls"] = calls
            out.append(mm)
    if not any(m["role"] != "system" for m in out):
        raise BadRequest("messages: at least one user or assistant turn with content is required")
    req = {"messages": out, "max_tokens": mt, "stream": True, "stream_options": {"include_usage": True}}
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            req[k] = body[k]
    if body.get("stop_sequences"):
        if not isinstance(body["stop_sequences"], list):
            raise BadRequest("stop_sequences: a list of strings")
        req["stop"] = body["stop_sequences"]
    th = body.get("thinking")
    if isinstance(th, dict) and th.get("type") == "enabled":
        req["chat_template_kwargs"] = {"enable_thinking": True}
        if isinstance(th.get("budget_tokens"), int):
            req["thinking_budget_tokens"] = th["budget_tokens"]
    tools = body.get("tools")
    if tools:
        if not isinstance(tools, list):
            raise BadRequest("tools: a list")
        ft = []
        for i, t in enumerate(tools):
            if not isinstance(t, dict) or not t.get("name"):
                raise BadRequest("tools[%d]: a name is required" % i)
            f = {"name": t["name"], "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}
            if t.get("description"):
                f["description"] = t["description"]
            ft.append({"type": "function", "function": f})
        req["tools"] = ft
        tc = body.get("tool_choice")
        if isinstance(tc, dict):
            kind = tc.get("type")
            if kind == "any":
                req["tool_choice"] = "required"
            elif kind == "tool":
                if not tc.get("name"):
                    raise BadRequest("tool_choice: type tool needs a name")
                req["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
            elif kind == "none":
                req["tool_choice"] = "none"
            elif kind not in (None, "auto"):
                raise BadRequest("tool_choice.type: auto, any, tool or none")
    return req


def message_id():
    return "msg_" + uuid.uuid4().hex[:24]


class Translator(object):
    """OpenAI stream lines in, Anthropic events out; also keeps the whole message for a non-streamed answer."""

    def __init__(self, model, mid=None):
        self.model, self.id = model, mid or message_id()
        self.blocks, self.kind = [], None          # the blocks so far; the open block's kind
        self.calls = {}                            # tool index -> block number
        self.stop = "end_turn"
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0}
        self.failed = None

    @staticmethod
    def ev(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(dict(data, type=name), ensure_ascii=False))

    def start(self):
        return [self.ev("message_start", {"message": {
            "id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": dict(self.usage)}})]

    def _close(self):
        if self.kind is None:
            return []
        out = []
        if self.kind == "thinking":
            out.append(self.ev("content_block_delta", {"index": len(self.blocks) - 1,
                                                        "delta": {"type": "signature_delta", "signature": ""}}))
        self.kind = None
        return out + [self.ev("content_block_stop", {"index": len(self.blocks) - 1})]

    def _open(self, block):
        out = self._close()
        self.blocks.append(block)
        self.kind = block["type"]
        first = {k: v for k, v in block.items() if not k.startswith("_")}
        return out + [self.ev("content_block_start", {"index": len(self.blocks) - 1, "content_block": first})]

    def _add(self, kind, text):
        out = [] if self.kind == kind else self._open({"type": kind, kind: ""})
        self.blocks[-1][kind] += text
        dk = "thinking_delta" if kind == "thinking" else "text_delta"
        return out + [self.ev("content_block_delta", {"index": len(self.blocks) - 1,
                                                       "delta": {"type": dk, kind: text}})]

    def line(self, ln):
        """the events for one line of the chat stream"""
        ln = ln.strip()
        if ln.startswith(":"):
            return [self.ev("ping", {})]           # the server's progress comment while a long prompt is read
        if not ln.startswith("data:"):
            return []
        d = ln[5:].strip()
        if d == "[DONE]":
            return []
        try:
            o = json.loads(d)
        except ValueError:
            return []
        if "error" in o:
            st, body = from_openai_error(400 if o["error"].get("type") == "invalid_request_error" else 500, o)
            self.failed = (st, body)
            return self._close() + [self.ev("error", {"error": body["error"]})]
        out = []
        u = o.get("usage")
        if u:
            cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
            self.usage.update(input_tokens=max(0, int(u.get("prompt_tokens") or 0) - cached),
                              output_tokens=int(u.get("completion_tokens") or 0), cache_read_input_tokens=cached)
        for ch in o.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("reasoning_content"):
                out += self._add("thinking", dl["reasoning_content"])
            if dl.get("content"):
                out += self._add("text", dl["content"])
            for tc in dl.get("tool_calls") or []:
                i = tc.get("index", 0)
                f = tc.get("function") or {}
                if i not in self.calls:
                    out += self._open({"type": "tool_use", "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:20],
                                       "name": f.get("name") or "", "input": {}, "_args": ""})
                    self.calls[i] = len(self.blocks) - 1
                if f.get("arguments"):
                    self.blocks[self.calls[i]]["_args"] += f["arguments"]
                    out.append(self.ev("content_block_delta", {"index": self.calls[i], "delta": {
                        "type": "input_json_delta", "partial_json": f["arguments"]}}))
            fr = ch.get("finish_reason")
            if fr:
                self.stop = {"tool_calls": "tool_use", "length": "max_tokens"}.get(fr, "end_turn")
        return out

    def end(self):
        out = self._close()
        if self.failed:
            return out
        return out + [self.ev("message_delta", {"delta": {"stop_reason": self.stop, "stop_sequence": None},
                                                "usage": dict(self.usage)}),
                      self.ev("message_stop", {})]

    def message(self):
        content = []
        for b in self.blocks:
            b = dict(b)
            if b["type"] == "tool_use":
                try:
                    b["input"] = json.loads(b.pop("_args") or "{}")
                except ValueError:
                    b["input"] = {}
            elif b["type"] == "thinking":
                b["signature"] = ""
            content.append(b)
        if not content:
            content = [{"type": "text", "text": ""}]
        return {"id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": content,
                "stop_reason": self.stop, "stop_sequence": None, "usage": dict(self.usage)}
