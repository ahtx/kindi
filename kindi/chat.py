"""Chat template and tool calls for Ternary-Bonsai-2 Qwen3.5 models (template from the model file, rendered directly and
checked byte for byte against jinja2 with order-preserving tojson, as llama.cpp and Hugging Face render it).

render(messages, tools, enable_thinking, variant=..., reasoning_effort=..., preserve_thinking=...) -> (prompt
body, generation prompt). Two template variants: "qwen35" (Ternary-Bonsai-2 2B/4B/9B) and "bonsai27"
(Ternary-Bonsai-2-27B: reasoning-effort instructions in the system prompt, past reasoning kept, no </think>
splitting of content, scalar tool arguments as JSON); template_variant() picks one from the pack. Tools go in a <tools> block of JSON
schemas; the model calls them as <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>, which
parse_tool_calls() turns into OpenAI tool_calls."""
import json
import os
import re
import uuid


TOOL_INSTR = ('\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n'
              '<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
              '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines'
              '\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the '
              'specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> '
              'XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function '
              'call in natural language BEFORE the function call, but NOT after\n- If there is no function call '
              'available, answer the question like normal with your current knowledge and do not tell the user about '
              'function calls\n</IMPORTANT>')
GEN_PROMPT_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
GEN_PROMPT_ON = "<|im_start|>assistant\n<think>\n"


def content_of(m):
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and "text" in p)
    return str(c)


def _argval(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


_XHIGH_INSTR = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                 "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity "
                 "in the final answer.")
REASONING_EFFORTS = {
    "xhigh": _XHIGH_INSTR,
    "medium": "",
    # Bonsai 2.5 27B's card does not support `low`: selecting it makes the model behave close to `xhigh`, and the
    # template's own `low` text is not what the model was trained on, so `low` renders the `xhigh` instruction.
    "low": _XHIGH_INSTR,
}


def template_variant(src):
    """"bonsai27" or "qwen35" from a chat template text, a pack manifest dict or a pack directory. A pack made
    before the manifest stored the template falls back to the model and source file names."""
    if isinstance(src, str) and os.path.isdir(src):
        try:
            src = json.load(open(os.path.join(src, "kindi_pack.json")))
        except (OSError, ValueError):
            return "qwen35"
    if isinstance(src, dict):
        t = src.get("chat_template")
        if not t:
            name = (str(src.get("model", "")) + " " + str(src.get("source_file", ""))).lower()
            return "bonsai27" if re.search(r"27b", name) else "qwen35"
        src = t
    return "bonsai27" if "reasoning_effort" in (src or "") else "qwen35"


def render(messages, tools=None, enable_thinking=False, variant="qwen35", reasoning_effort=None,
           preserve_thinking=None):
    """The GGUF chat template (text only), add_generation_prompt=True."""
    if variant == "bonsai27":
        return _render27(messages, tools, enable_thinking, reasoning_effort, preserve_thinking)
    if not messages:
        raise ValueError("No messages provided.")
    out = ""
    if tools:
        out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for t in tools:
            out += "\n" + json.dumps(t, ensure_ascii=False)
        out += "\n</tools>" + TOOL_INSTR
        if messages[0]["role"] == "system":
            c = content_of(messages[0]).strip()
            if c:
                out += "\n\n" + c
        out += "<|im_end|>\n"
    elif messages[0]["role"] == "system":
        out += "<|im_start|>system\n" + content_of(messages[0]).strip() + "<|im_end|>\n"
    last_q, multi = len(messages) - 1, True
    for idx in range(len(messages) - 1, -1, -1):
        m = messages[idx]
        if multi and m["role"] == "user":
            c = content_of(m).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                multi, last_q = False, idx
    if multi:
        raise ValueError("No user query found in messages.")
    for i, m in enumerate(messages):
        c, r = content_of(m).strip(), m["role"]
        if r == "system":
            if i != 0:
                raise ValueError("System message must be at the beginning.")
        elif r == "user":
            out += "<|im_start|>user\n" + c + "<|im_end|>\n"
        elif r == "assistant":
            rc = m.get("reasoning_content")
            if not isinstance(rc, str):
                rc = ""
                if "</think>" in c:
                    rc = c.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    c = c.split("</think>")[-1].lstrip("\n")
            rc = rc.strip()
            if i > last_q:
                out += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + c
            else:
                out += "<|im_start|>assistant\n" + c
            for k, tc in enumerate(m.get("tool_calls") or []):
                f = tc.get("function", tc)
                if k == 0:
                    out += ("\n\n<tool_call>\n<function=" if c.strip() else "<tool_call>\n<function=") + f["name"] + ">\n"
                else:
                    out += "\n<tool_call>\n<function=" + f["name"] + ">\n"
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                for an, av in (args or {}).items():
                    out += "<parameter=" + an + ">\n" + _argval(av) + "\n</parameter>\n"
                out += "</function>\n</tool_call>"
            out += "<|im_end|>\n"
        elif r == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += "\n<tool_response>\n" + c + "\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
        else:
            raise ValueError("Unexpected message role.")
    gp = GEN_PROMPT_ON if enable_thinking else GEN_PROMPT_OFF
    return out, gp


def _argval27(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)


def _render27(messages, tools, enable_thinking, reasoning_effort, preserve_thinking):
    """The Ternary-Bonsai-2-27B template. enable_thinking is a bool here (the template's "undefined" case is the
    same as True); reasoning_effort None means the template default, xhigh; preserve_thinking None means True."""
    if not messages:
        raise ValueError("No messages provided.")
    instr = ""
    if enable_thinking:
        eff = "xhigh" if reasoning_effort is None else reasoning_effort
        if eff not in REASONING_EFFORTS:
            raise ValueError("Unexpected reasoning effort %s. Supported types are xhigh (default), medium, and low." % eff)
        instr = REASONING_EFFORTS[eff]
    keep = preserve_thinking is None or preserve_thinking is True
    out = ""
    if tools:
        out += "<|im_start|>system\n"
        if instr:
            out += instr + "\n\n"
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for t in tools:
            out += "\n" + json.dumps(t, ensure_ascii=False)
        out += "\n</tools>" + TOOL_INSTR
        if messages[0]["role"] == "system":
            c = content_of(messages[0]).strip()
            if c:
                out += "\n\n" + c
        out += "<|im_end|>\n"
    elif messages[0]["role"] == "system":
        c = content_of(messages[0]).strip()
        if c:
            out += "<|im_start|>system\n" + (instr + "\n\n" if instr else "") + c + "<|im_end|>\n"
        elif instr:
            out += "<|im_start|>system\n" + instr + "<|im_end|>\n"
    elif instr:
        out += "<|im_start|>system\n" + instr + "<|im_end|>\n"
    last_q, multi = len(messages) - 1, True
    for idx in range(len(messages) - 1, -1, -1):
        m = messages[idx]
        if multi and m["role"] == "user":
            c = content_of(m).strip()
            if not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
                multi, last_q = False, idx
    if multi:
        raise ValueError("No user query found in messages.")
    for i, m in enumerate(messages):
        c, r = content_of(m).strip(), m["role"]
        if r == "system":
            if i != 0:
                raise ValueError("System message must be at the beginning.")
        elif r == "user":
            out += "<|im_start|>user\n" + c + "<|im_end|>\n"
        elif r == "assistant":
            rc = m.get("reasoning_content")
            rc = rc.strip() if isinstance(rc, str) else ""
            if keep or i > last_q:
                out += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + c
            else:
                out += "<|im_start|>assistant\n" + c
            for k, tc in enumerate(m.get("tool_calls") or []):
                f = tc.get("function", tc)
                if k == 0:
                    out += ("\n\n<tool_call>\n<function=" if c.strip() else "<tool_call>\n<function=") + f["name"] + ">\n"
                else:
                    out += "\n<tool_call>\n<function=" + f["name"] + ">\n"
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {}
                for an, av in (args or {}).items():
                    out += "<parameter=" + an + ">\n" + _argval27(av) + "\n</parameter>\n"
                out += "</function>\n</tool_call>"
            out += "<|im_end|>\n"
        elif r == "tool":
            if i > 0 and messages[i - 1]["role"] != "tool":
                out += "<|im_start|>user"
            out += "\n<tool_response>\n" + c + "\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                out += "<|im_end|>\n"
        else:
            raise ValueError("Unexpected message role.")
    gp = GEN_PROMPT_ON if enable_thinking else GEN_PROMPT_OFF
    return out, gp


TC_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
PAR_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_tool_calls(text, tools):
    """-> (content before the first call, [OpenAI tool_calls])."""
    schema = {}
    for t in tools or []:
        f = t.get("function", t)
        schema[f.get("name")] = ((f.get("parameters") or {}).get("properties") or {})
    calls = []
    for m in TC_RE.finditer(text):
        name = m.group(1).strip()
        props = schema.get(name, {})
        args = {}
        for pm in PAR_RE.finditer(m.group(2)):
            k, v = pm.group(1).strip(), pm.group(2)
            typ = (props.get(k) or {}).get("type")
            if typ in ("string", None) and not (typ is None and v.strip()[:1] in "[{"):
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except ValueError:
                    args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    i = text.find("<tool_call>")
    # a call cut off by max_tokens (or malformed) parses to nothing: return its raw text rather than dropping it
    content = (text[:i] if i >= 0 and calls else text).strip()
    return content, calls
