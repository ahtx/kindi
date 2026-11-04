"""The Nanbeige 4.x chat template (from the GGUF), rendered directly: system / user / assistant / tool messages,
the thinking block, and tool calls in the XML or the JSON form. render() gives the same text as llama.cpp's template
engine for these cases; prompt_ids() tokenizes it as llama.cpp's server does (the leading <|im_start|> is the BOS)."""
import json

DEFAULT_SYSTEM = "你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。"
DEFAULT_TOOL_SYSTEM = ("你是一位工具函数调用专家，你会得到一个问题和一组可能的工具函数。根据问题，你需要进行一个或多个函数/工具调用以实现目的，"
                       "请尽量尝试探索通过工具解决问题。\n如果没有一个函数可以使用，请直接使用自然语言回复用户。\n"
                       "如果给定的问题缺少函数所需的参数，请使用自然语言进行提问，向用户询问必要信息。\n"
                       "如果调用结果已经足够回答用户问题，请对历史结果进行总结，使用自然语言回复用户。")
TOOLS_HEAD = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
TOOLS_JSON = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
              "<tool_call></tool_call> XML tags:\n"
              "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
TOOLS_XML = ("\n</tools>\n\nFor each function call, output the function name and arguments within the following XML "
             "format:\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
             "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\n"
             "multiple lines\n</parameter>\n</function>\n</tool_call><|im_end|>\n")
MEDIA = ("image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio")


def _tojson(v):
    return json.dumps(v, ensure_ascii=False)


def visible_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out += item.get("text", "")
            elif isinstance(item, str):
                out += item
            elif isinstance(item, dict) and item.get("type") in MEDIA:
                media = item["type"].replace("_url", "").replace("input_", "")
                out += ("<reminder>You are unable to process this " + media + " because you don't have multi-modal "
                        "input ability. Try different methods.</reminder>")
        return out
    return str(content)


def render(messages, tools=None, enable_thinking=True, tool_call_format="xml", preserve_thinking=None,
           add_generation_prompt=True):
    out = ""
    first_sys = bool(messages) and messages[0].get("role", "") == "system"
    if tools:
        out += "<|im_start|>system\n"
        out += visible_text(messages[0].get("content")) + "\n\n" if first_sys else DEFAULT_TOOL_SYSTEM
        out += TOOLS_HEAD
        for t in tools:
            out += "\n" + _tojson(t)
        out += TOOLS_JSON if tool_call_format == "json" else TOOLS_XML
    elif first_sys:
        out += "<|im_start|>system\n" + visible_text(messages[0].get("content")) + "<|im_end|>\n"
    else:
        out += "<|im_start|>system\n" + DEFAULT_SYSTEM + "<|im_end|>\n"
    last_query = len(messages) - 1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        c = visible_text(m.get("content"))
        if m.get("role", "") == "user" and not (c.startswith("<tool_response>") and c.endswith("</tool_response>")):
            last_query = i
            break
    for i, m in enumerate(messages):
        role = m.get("role", "")
        content = visible_text(m.get("content"))
        if role == "system":
            if i != 0:
                raise ValueError("System message must be at the beginning.")
        elif role == "assistant":
            rc = m.get("reasoning_content")
            if not isinstance(rc, str):
                rc = ""
                if "</think>" in content:
                    rc = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    content = content.split("</think>")[-1].lstrip("\n").rstrip("\n")
            content, rc = content.strip(), rc.strip()
            if preserve_thinking is False and i < last_query:
                out += "<|im_start|>assistant\n<think>\n\n</think>\n\n" + content
            else:
                out += "<|im_start|>assistant\n<think>\n" + rc + "\n</think>\n\n" + content
            calls = m.get("tool_calls") or []
            for j, tc in enumerate(calls):
                tc = tc.get("function", tc)
                args = tc.get("arguments", {})
                if tool_call_format == "json":
                    if (j == 0 and content) or j > 0:
                        out += "\n"
                    out += '<tool_call>\n{"name": "' + tc["name"] + '", "arguments": '
                    out += (args if isinstance(args, str) else _tojson(args)) + "}\n</tool_call>"
                else:
                    if isinstance(args, str):
                        args = json.loads(args) if args.strip() else {}
                    if j == 0:
                        out += ("\n\n" if content.strip() else "") + "<tool_call>\n<function=" + tc["name"] + ">\n"
                    else:
                        out += "\n<tool_call>\n<function=" + tc["name"] + ">\n"
                    for k, v in args.items():
                        out += "<parameter=" + k + ">\n" + (v if isinstance(v, str) else _tojson(v)) + "\n</parameter>\n"
                    out += "</function>\n</tool_call>"
            out += "<|im_end|>\n"
        elif role == "tool":
            if i > 0 and messages[i - 1].get("role", "") != "tool":
                out += "<|im_start|>user"
            out += "\n<tool_response>\n" + content + "\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1].get("role", "") != "tool":
                out += "<|im_end|>\n"
        elif role:
            out += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
    return out


def prompt_ids(tok, text):
    """token ids as llama.cpp's server builds them: a leading BOS text in the template output is the BOS token"""
    bos = tok.tokens[tok.bos] if tok.bos is not None else None
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return tok.encode(text)
