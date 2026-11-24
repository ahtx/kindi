"""OpenAI request fields that change what the model must write: tool_choice and response_format (issue #38).

Before 0.7, every engine read only tool_choice "none" and dropped the rest, and response_format was ignored, so a
client that asked for a forced call or a JSON schema got free text with HTTP 200. Now a request gets what it asks
for, or a clear HTTP 400 that names the field:

- tool_choice "auto" (or none given): the model decides. "none": the tools are not shown to the model.
- tool_choice "required", or {"type": "function", "function": {"name": N}}: the reply is a tool call (of N). The
  engines that use the <tool_call><function=...> format (the Bonsai models) start the reply with that opening, so
  the model writes the call; the other engines answer 400 until they can force a call.
- response_format {"type": "text"}: the default. "json_object" and "json_schema": Kindi does not constrain the
  output to a schema yet, so the request gets a 400 instead of unchecked text. KINDI_RESPONSE_FORMAT=ignore restores
  the old behaviour (the field is ignored) for clients that always send it.

Standard library only. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only."""
import os


class RequestError(ValueError):
    """a request field Kindi cannot honour: the server answers HTTP 400 with this message and param"""

    def __init__(self, message, param):
        ValueError.__init__(self, message)
        self.param = param


def _tool_names(tools):
    return [((t.get("function") or t).get("name")) for t in tools or [] if isinstance(t, dict)]


def check_response_format(req):
    rf = req.get("response_format")
    if rf is None or os.environ.get("KINDI_RESPONSE_FORMAT", "") == "ignore":
        return
    kind = rf.get("type") if isinstance(rf, dict) else rf
    if kind in (None, "text"):
        return
    if kind in ("json_object", "json_schema"):
        raise RequestError(
            "response_format type '%s' is not supported: Kindi does not constrain the output to JSON or a schema yet, "
            "and it does not ignore the request silently. Remove response_format and describe the format in the "
            "prompt (or start the server with KINDI_RESPONSE_FORMAT=ignore to accept and ignore it)." % kind,
            "response_format")
    raise RequestError("response_format type '%s' is not known (text, json_object or json_schema)" % kind,
                       "response_format")


def tool_choice(req, can_force):
    """-> (tools, force): the tools to show the model, and None or the name to call ("" = any tool).
    can_force: this engine can make the reply start with a call. Raises RequestError for a 400."""
    tools = req.get("tools") or None
    tc = req.get("tool_choice")
    if tc in (None, "auto"):
        return tools, None
    if tc == "none":
        return None, None
    if tc == "required":
        name = ""
    elif isinstance(tc, dict) and tc.get("type") == "function" and isinstance(tc.get("function"), dict):
        name = tc["function"].get("name") or ""
        if not name:
            raise RequestError("tool_choice names no function", "tool_choice")
    else:
        raise RequestError("tool_choice must be \"auto\", \"none\", \"required\" or "
                           "{\"type\": \"function\", \"function\": {\"name\": ...}}", "tool_choice")
    if not tools:
        raise RequestError("tool_choice asks for a tool call, but the request has no tools", "tool_choice")
    if name and name not in _tool_names(tools):
        raise RequestError("tool_choice names the function '%s', which is not in tools (%s)"
                           % (name, ", ".join(n for n in _tool_names(tools) if n)), "tool_choice")
    if not can_force:
        raise RequestError("tool_choice %s is not supported by this model's engine yet: it cannot make the reply a "
                           "tool call. Use \"auto\" (the model decides) or \"none\"."
                           % ("\"required\"" if not name else "for the function '%s'" % name), "tool_choice")
    if name:   # only the named tool is shown, so the call (and its arguments) follow that tool's schema
        tools = [t for t in tools if (t.get("function") or t).get("name") == name]
    return tools, name


def forced_call_prefix(name):
    """the reply's opening that makes it a tool call, in the <tool_call><function=...> format (chat.TC_RE)"""
    return "<tool_call>\n<function=" + (name + ">\n" if name else "")


def check_messages(req):
    """messages is a list of objects, each with a string role (PR #58: a string here was an HTTP 500)"""
    if not isinstance(req, dict):
        raise RequestError("the request body must be a JSON object", None)
    m = req.get("messages")
    if not isinstance(m, list):
        raise RequestError("messages must be a list of {\"role\": ..., \"content\": ...} objects", "messages")
    for i, x in enumerate(m):
        if not isinstance(x, dict) or not isinstance(x.get("role"), str):
            raise RequestError("messages[%d] must be an object with a string \"role\"" % i, "messages")


def check(req, can_force):
    """all checks; -> (tools, force)"""
    check_messages(req)
    check_response_format(req)
    return tool_choice(req, can_force)
