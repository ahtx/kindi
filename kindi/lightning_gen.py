"""Generation for the Lightning backend in Kindi's server (server.Kindi with backend "kindi_lightning"): the chat
template of the GGUF, the thinking block as reasoning, tool calls in the XML form, and prompt reuse.

Prompt reuse: the Mamba states cannot go back to an earlier position, so the backend keeps one copy of them at the
start of the last assistant header of each prompt (lightning.Engine.snapshot). A request whose prompt continues the
engine's sequence reads only the new tokens; a request that shares the prompt up to that copy (a new turn of the
same chat) restores it and reads from there; any other request reads its whole prompt."""
import codecs
import os
import re
import time

import numpy as np

from .nanbeige_gen import CALL_OPEN, _lcp, parse_calls

IM_START = "<|im_start|>"
THINK_OPEN, THINK_CLOSES = "<think>", ("</think>", "</thought>", "</thinking>")
# the model sometimes opens with these, and sometimes leaves out the ">" (an opener at the end of a line)
THINK_OPENS = (THINK_OPEN, "<thought>", "<thinking>", "<think\n", "<thought\n", "<thinking\n")
FUNC_OPEN = "<function="  # a call whose <tool_call> opener the model left out
CALL_CLOSE = "</tool_call>"
CALL_ALT = "⟨tool_call>"   # the look-alike opener the model sometimes writes (U+27E8; parse_calls takes it)
MISCALL = CALL_CLOSE + "\n" + FUNC_OPEN   # a call that the model opened with </tool_call> (also inside thinking)
# a role marker in the text ("<|user|>", "<|channel|>", ...: not tokens of this model) starts a turn that the model
# must not write: the reply ends before it
ROLE_MARK = re.compile(r"<\|[a-z_]{2,24}\|>")
ROLE_PART = re.compile(r"<(\|[a-z_]{0,24})?$")
# NVIDIA's recommended sampling for Nemotron 3.5 Lightning (model card and generation_config.json), used when neither
# the request nor the configuration sets a value; repeat_stop is on unless the configuration turns it off
MODEL_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0}


def json_key(call):
    """a parsed call as (name, arguments): two calls with the same key are the same call"""
    f = call.get("function") or {}
    return f.get("name"), f.get("arguments")


def cut_calls(tail, n):
    """the call tail up to the end of its n-th </tool_call> (all of it when it has fewer)"""
    end = 0
    for _ in range(n):
        i = tail.find(CALL_CLOSE, end)
        if i < 0:
            return tail
        end = i + len(CALL_CLOSE)
    return tail[:end]


_PARAM_EQ = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_\-]*)=")


def normalize_calls(tail):
    """the call-syntax slips that the model makes: <parameter=name=value> (an "=" where the ">" belongs) and
    </function_call> for </function>"""
    return _PARAM_EQ.sub(r"<parameter=\1>", tail).replace("</function_call>", "</function>")


def calls_of(tail, tools):
    """the calls of a call tail, after normalize_calls()"""
    return parse_calls(normalize_calls(tail), tools)


def close_call(tail):
    """close the tags that a cut call left open: the last <parameter=...>, then <function=...>, then <tool_call>"""
    t = tail.rstrip()
    if t.rfind("<parameter=") > t.rfind("</parameter>"):
        t += "\n</parameter>"
    if t.rfind(FUNC_OPEN) > t.rfind("</function>"):
        t += "\n</function>"
    if t.rfind(CALL_OPEN) > t.rfind(CALL_CLOSE) or (FUNC_OPEN in t and CALL_CLOSE not in t):
        t += "\n" + CALL_CLOSE
    return t


class ThinkSorter(object):
    """splits streamed text into ("reasoning", ...) and ("text", ...) pieces. Thinking ends at </think> (the model
    sometimes writes </thought>); a <think> in the answer opens thinking again, and a closing tag outside thinking is
    dropped, so no tag reaches the visible text. From <tool_call> on, the text is kept as the call tail. A partial tag
    at the end is held back, and dropped when the reply ends there. A role marker (ROLE_MARK) ends the reply:
    stopped is then True."""

    def __init__(self, thinking):
        self.mode = "think" if thinking else "answer"
        self.buf = ""
        self.tail = ""
        self.stopped = False
        self.prev = ""          # the last character sent out (for a tag at the start of the buffer)

    def _first(self, tags):
        hits = [(self.buf.find(t), -len(t), t) for t in tags if self.buf.find(t) >= 0]
        return min(hits)[::2] if hits else (-1, None)      # (the longest tag at the first position)

    def feed(self, s, final=False):
        out = []
        if self.stopped:
            return out
        self.buf += s
        m = ROLE_MARK.search(self.buf)
        if m:
            self.buf, self.stopped, final = self.buf[:m.start()], True, True
        while True:
            if self.mode == "calls":
                self.tail += self.buf
                self.buf = ""
                return out
            if self.mode == "think":
                i, tag = self._first(THINK_CLOSES + THINK_OPENS + (CALL_OPEN, CALL_ALT, MISCALL, CALL_CLOSE))
                kind = "reasoning"
            else:
                i, tag = self._first(THINK_OPENS + (CALL_OPEN, CALL_ALT, MISCALL, FUNC_OPEN, CALL_CLOSE) + THINK_CLOSES)
                kind = "text"
            if i >= 0 and tag == CALL_CLOSE and not final and MISCALL.startswith(self.buf[i:]):
                if self.buf[:i]:                               # (it may still become a call: wait)
                    out.append((kind, self.buf[:i]))
                self.buf = self.buf[i:]
                return out
            if i >= 0 and tag in (CALL_OPEN, CALL_ALT, FUNC_OPEN) and \
                    (self.buf[i - 1] if i > 0 else self.prev) == "`":
                # `<tool_call>` in backticks: the model writes about the call format (in its thinking or its
                # answer); it is text, not a call
                out.append((kind, self.buf[:i + len(tag)]))
                self.prev, self.buf = tag[-1], self.buf[i + len(tag):]
                continue
            if i >= 0:
                if self.buf[:i]:
                    out.append((kind, self.buf[:i]))
                rest = self.buf[i + len(tag):]
                if tag in (CALL_OPEN, CALL_ALT, FUNC_OPEN, MISCALL):
                    if tag == MISCALL:
                        self.tail = CALL_OPEN + self.buf[i + len(CALL_CLOSE):]
                    else:
                        self.tail = (CALL_OPEN if tag == FUNC_OPEN else "") + self.buf[i:]
                    self.buf, self.mode = "", "calls"
                elif tag in THINK_OPENS:
                    self.buf, self.mode = rest, "think"        # (a second <think> inside thinking: dropped)
                elif tag == CALL_CLOSE:
                    self.buf = rest                            # a stray </tool_call>: dropped
                elif self.mode == "think":
                    self.buf, self.mode = rest.lstrip("\n"), "answer"
                else:
                    self.buf = rest                            # a closing tag in the answer: dropped
                continue
            tags = THINK_OPENS + THINK_CLOSES + (CALL_OPEN, CALL_ALT, CALL_CLOSE, MISCALL, FUNC_OPEN)
            j = max(self.buf.rfind("<"), self.buf.rfind(CALL_ALT[0]))
            part = self.buf[j:] if j >= 0 else ""
            partial = bool(part) and (ROLE_PART.match(part) is not None or any(t.startswith(part) for t in tags))
            if final:                  # the reply ends: a partial tag or marker at the end is dropped
                if partial:
                    self.buf = self.buf[:j]
                keep = 0
            else:
                keep = max(len(t) for t in tags) - 1
                if partial:            # (a role marker can be longer than the tags)
                    keep = max(keep, len(part))
            cut = max(0, len(self.buf) - keep)
            if cut:
                out.append((kind, self.buf[:cut]))
                self.prev, self.buf = self.buf[cut - 1], self.buf[cut:]
            return out


class LightningBackend(object):
    def __init__(self, man, cfg, log=print):
        from .gguf import GGUF
        from .lightning import Engine
        from .lightning_tokenizer import LightningTokenizer
        self.log = log
        self.path = man["model_path"]
        self.tok = LightningTokenizer(GGUF(self.path).kv)
        self.tok.tokens = self.tok.vocab
        # the tokens that end a turn: every control token of the GGUF (type 3: <|im_end|> = EOS, </s>, <|im_start|>,
        # [INST], ...). A control token never belongs in a reply, and <|im_start|> would start a new turn.
        self.stop_ids = {self.tok.eos} | {i for i, t in enumerate(self.tok.types) if t == 3}
        self.im_start = self.tok.sid[IM_START]
        self.eng = Engine(self.path, ctx=int(cfg["ctx"]), log=log)
        self.cached = []            # the tokens in the engine's state, in order
        self.snap = []              # the tokens before the Mamba-state copy
        if os.environ.get("KINDI_LT_SELFCHECK", "1") != "0":
            self.selfcheck()

    def selfcheck(self):
        """before the first request: the same greedy tokens twice, from a prompt read in one pass and token by token,
        and not a degenerate run. A failure raises, so that the service restarts."""
        from .lightning_chat import render
        eng = self.eng
        ids = self.tok.encode(render(self.path, [{"role": "user", "content": "Count from one to ten in words."}],
                                     enable_thinking=False))
        a = eng.generate(ids, 10)
        b = eng.generate(ids, 10)
        eng.reset()
        eng.prefill(ids[:1], 0)
        for t in ids[1:-1]:
            eng.step(t)
        c = [eng.step(ids[-1])]
        while len(c) < 10:
            c.append(eng.step())
        self.cached = []
        ok = a == b and len(set(a)) > 3
        if not ok:
            raise RuntimeError("Lightning self-check failed: %s / %s" % (a, b))
        self.log("Lightning self-check: %s (token by token: %s)" % (
            repr(self.tok.decode(a))[:60], "the same" if c == a else "differs after a near tie"))

    def _pick(self, greedy, temperature, top_p, top_k, rng, min_p=0.0, penalty=None):
        """sample the next token from the card's logits: the candidates within 30 temperatures of the best (at most
        256), llama.cpp's repeat penalty (penalty: (repeat_penalty, recent tokens)), then top-k, top-p, min-p and the
        temperature (sampling.Sampler._pick). Greedy without a penalty on a recent token: the engine's token."""
        if temperature <= 0 and (penalty is None or penalty[0] <= 1.0 or greedy not in penalty[1]):
            return greedy
        from .sampling import Sampler, penalize
        # the candidates within 16 temperatures of the best (relative probability under 1e-7 is never drawn); when
        # more than 4096 qualify, a narrower window (the sampler keeps the 256 best in any case)
        w = 16.0 * max(temperature, 0.1)
        got = self.eng.candidates(w)
        while got is None and w > 0.5:
            w /= 2
            got = self.eng.candidates(w)
        if got is not None:
            cand, sc = got
        else:                                              # (more than 4096 candidates: the whole vocabulary)
            cp = self.eng.cp
            z = self.eng.logits
            idx = cp.flatnonzero(z >= z.max() - 30.0 * max(temperature, 0.1))
            cand = cp.asnumpy(idx).astype(np.int64)
            sc = cp.asnumpy(z[idx]).astype(np.float64)
        if cand.size > 256:
            keep = np.argpartition(sc, -256)[-256:]
            cand, sc = cand[keep], sc[keep]
        if penalty is not None and penalty[0] > 1.0:
            sc = penalize(cand, sc, penalty[0], penalty[1])
        if temperature <= 0:
            return int(cand[int(np.argmax(sc))])
        return Sampler._pick(cand, sc, temperature, top_p, top_k, rng, min_p)

    def _read(self, ids):
        """bring the engine to the end of ids, reusing what it holds; returns (greedy next token, reused tokens)"""
        eng = self.eng
        reuse = _lcp(self.cached, ids)
        if reuse == len(self.cached) and reuse < len(ids):
            start = reuse                                  # the prompt continues the engine's sequence
        elif self.snap and _lcp(self.snap, ids) == len(self.snap) and len(self.snap) < len(ids):
            eng.restore()
            start = len(self.snap)
        else:
            eng.reset()
            start = 0
        reused = start
        # the copy of the Mamba states: at the last assistant header of this prompt
        mark = max((i for i, t in enumerate(ids) if t == self.im_start), default=-1)
        if start < mark:
            eng.prefill(ids[start:mark], start)
            eng.snapshot()
            self.snap = list(ids[:mark])
            start = mark
        greedy = eng.prefill(ids[start:], start)
        self.cached = list(ids)
        return greedy, reused

    def generate(self, K, req, cancel=None):
        from .lightning_chat import render
        from .server import THINK_CLOSE, _looping
        cfg = K.cfg
        tools = req.get("tools") or None
        if req.get("tool_choice") == "none":
            tools = None
        kw = req.get("chat_template_kwargs") or {}
        thinking = kw.get("enable_thinking", req.get("enable_thinking", True)) is not False
        ids = self.tok.encode(render(self.path, req.get("messages") or [], tools=tools, enable_thinking=thinking))
        if len(ids) >= self.eng.ctx:
            raise OverflowError("the prompt has %d tokens; context is %d" % (len(ids), self.eng.ctx))
        if req.get("_kindi_check"):
            return
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731
        from .config import DEFAULTS
        dflt = lambda k: cfg[k] if str(cfg.get(k)) != DEFAULTS.get(k) else MODEL_SAMPLING[k]  # noqa: E731
        temperature = float(g("temperature", dflt("temperature")))
        top_p = float(g("top_p", dflt("top_p")))
        top_k = int(g("top_k", dflt("top_k")))
        min_p = float(g("min_p", dflt("min_p")))
        rp = float(g("repeat_penalty", cfg.get("repeat_penalty", 1.0)))
        rlast = int(g("repeat_last_n", cfg.get("repeat_last_n", 64)))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        max_new = max(1, min(max_new, self.eng.ctx - len(ids)))
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        # on unless the configuration sets it (the Kindi default 0 is for the Bonsai models)
        repeat_stop = True if str(cfg.get("repeat_stop")) == DEFAULTS.get("repeat_stop")             else bool(int(cfg["repeat_stop"]))
        budget = req.get("thinking_budget_tokens") or req.get("thinking_budget") or cfg.get("thinking_budget") or 0
        budget = int(budget) if thinking else 0
        forced = False
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        started = time.time()
        K.busy.update(active=True, phase="prompt", prompt_tokens=len(ids), cached=0, read=0, generated=0,
                      started=started)
        K.stats["requests"] += 1
        sorter = ThinkSorter(thinking)
        generated, text, finish = [], "", "length"
        got = {"text": [], "reasoning": []}
        reuse = 0
        # one call per reply unless the client asks for parallel calls (as llama.cpp); with parallel calls, a call
        # that repeats an earlier one (same name and arguments) ends the reply. Without this the model can write
        # the same call 20 times (3,000+ tokens) before it stops.
        parallel = bool(req.get("parallel_tool_calls", False))
        closes = 0
        try:
            greedy, reuse = self._read(ids)
            K.busy.update(read=len(ids), cached=reuse, phase="generating")
            prompt_done = time.time()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            queue = []
            for n in range(max_new):
                if cancel is not None and cancel.is_set():
                    finish = "stop"
                    break
                pen = (rp, set((ids + generated)[-rlast:] if rlast > 0 else ids + generated)) if rp > 1.0 else None
                token = queue.pop(0) if queue else self._pick(greedy, temperature, top_p, top_k, rng, min_p, pen)
                if token in self.stop_ids:
                    if os.environ.get("KINDI_LOG_RAW"):
                        self.log("Lightning: stop token %d %r" % (token, self.tok.decode_bytes(token)))
                    finish = "stop"
                    break
                generated.append(token)
                K.busy["generated"] = len(generated)
                loop = repeat_stop and len(generated) % 16 == 0 and _looping(generated)
                if loop and sorter.mode != "think":
                    self.log("Lightning: the reply repeats a block of tokens; stopped (repeat_stop)")
                    finish = "stop"
                    break
                if sorter.mode == "think" and not forced and (loop or (budget and len(generated) >= budget)):
                    forced = True             # thinking budget spent: close the thinking, then let the model answer
                    self.log("Lightning: thinking budget of %d tokens reached; closing the thinking"
                             % (budget or len(generated)))
                    queue = self.tok.encode(THINK_CLOSE)
                full = self.eng.pos >= self.eng.ctx        # the context is full: this token is the last
                if n + 1 < max_new and not full:
                    greedy = self.eng.step(token)
                    self.cached.append(token)
                piece = decoder.decode(self.tok.decode_bytes(token))
                text += piece
                if next((x for x in stop if x and x in text), None):
                    finish = "stop"
                    break
                for kind, part in sorter.feed(piece):
                    got[kind].append(part)
                    yield kind, part
                if sorter.stopped:
                    self.log("Lightning: the reply wrote a role marker; stopped there")
                    finish = "stop"
                    break
                if tools and sorter.tail and sorter.tail.count(CALL_CLOSE) > closes:
                    closes = sorter.tail.count(CALL_CLOSE)
                    # only calls of the offered tools count: the model often copies the chat template's example
                    # call (example_function_name) before the real one
                    names = {(t.get("function") or {}).get("name") for t in tools}
                    done_calls = [c for c in calls_of(sorter.tail, tools) if json_key(c)[0] in names]
                    keys = [json_key(c) for c in done_calls]
                    if done_calls and (not parallel or len(set(keys)) < len(keys)):
                        # the </tool_call> that just arrived completed this call (a real one, or the repeat)
                        last = sorter.tail.rfind(CALL_CLOSE)
                        if parallel:          # drop the repeat: cut at the </tool_call> before it
                            last = sorter.tail.rfind(CALL_CLOSE, 0, last)
                            sorter.tail = sorter.tail[:last + len(CALL_CLOSE)] if last >= 0 else sorter.tail
                        else:
                            sorter.tail = sorter.tail[:last + len(CALL_CLOSE)]
                        self.log("Lightning: %s; stopped after the call%s" % (
                            "a call repeated an earlier one" if parallel else "one tool call per reply",
                            "s" if parallel else ""))
                        finish = "stop"
                        break
                if full:
                    break
            for kind, part in sorter.feed("", final=True):
                got[kind].append(part)
                yield kind, part
        finally:
            K.busy["active"] = False
        elapsed = max(0.001, time.time() - prompt_done)
        calls = calls_of(sorter.tail, tools) if (tools and sorter.tail) else []
        if tools and not calls and sorter.tail and finish == "stop":
            # the reply ended inside a call (a stop token before </tool_call>): close the open tags and parse again.
            # parse_calls keeps a call only when every required parameter is there, so a cut call is not invented.
            calls = calls_of(close_call(sorter.tail), tools)
        if calls:             # a call of a tool that was not offered (the template's example call) is not returned
            names = {(t.get("function") or {}).get("name") for t in tools}
            calls = [c for c in calls if json_key(c)[0] in names]
        if os.environ.get("KINDI_LOG_RAW"):   # debug (off by default): the raw reply and what the parser made of it
            import json
            with open(os.environ["KINDI_LOG_RAW"], "a") as f:
                f.write(json.dumps({"text": text, "tail": sorter.tail, "calls": calls}) + "\n")
        if tools and not calls and "<function=" in "".join(got["text"]):
            body = "".join(got["text"])
            j = body.index("<function=")
            calls = calls_of(CALL_OPEN + body[j:], tools)
            if calls:                         # the call text is not part of the visible answer
                got["text"] = [body[:j].replace(CALL_OPEN, "").replace(CALL_CLOSE, "").rstrip()]
        if sorter.tail and not calls:
            got["text"].append(sorter.tail)
            yield "text", sorter.tail
        if finish == "stop" and sorter.mode == "think" and not got["text"]:
            ans = "".join(got["reasoning"]).strip()          # an answer inside an open thinking block
            if tools and CALL_OPEN in ans:
                calls = calls_of(ans[ans.index(CALL_OPEN):], tools)
                if calls:
                    ans = ans[:ans.index(CALL_OPEN)].strip()
                    got["reasoning"] = [ans] if ans else []
                    ans = ""
            if ans:
                got["text"], got["reasoning"] = [ans], []
                yield "text", ans
        if calls:
            finish = "tool_calls"
        K.stats["prompt_tokens"] += len(ids)
        K.stats["cached_tokens"] += reuse
        K.stats["completion_tokens"] += len(generated)
        self.log("Lightning: prompt %d (%d reused, %.2f s), %d tokens at %.1f tok/s, %s" % (
            len(ids), reuse, prompt_done - started, len(generated), len(generated) / elapsed, finish))
        yield "done", {"content": "".join(got["text"]).strip(), "tool_calls": calls,
                       "reasoning": "".join(got["reasoning"]).strip() or None, "finish": finish, "logprobs": None,
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                                 "total_tokens": len(ids) + len(generated),
                                 "prompt_tokens_details": {"cached_tokens": reuse}},
                       "timing": {"prompt_s": round(prompt_done - started, 2), "prompt_new_tokens": len(ids) - reuse,
                                  "gen_s": round(elapsed, 2), "gen_tok_s": round(len(generated) / elapsed, 2)}}
