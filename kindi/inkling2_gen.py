"""Generation for the Inkling 2 Mini backend in Kindi's server (server.Kindi with backend "kindi_inkling2"): the chat
template of the GGUF, the thinking message as reasoning, tool calls in the model's JSON form as OpenAI tool_calls,
the sampling values of the GGUF, and prompt reuse.

The model writes its reply as messages, each a special token sequence:
  <|message_model|><|content_thinking|>...<|end_message|>        thinking  -> "reasoning"
  <|message_model|><|content_text|>...<|end_message|>            the answer -> "text"
  <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>   a tool call
and ends the reply with <|content_model_end_sampling|> (the GGUF's EOS).

Prompt reuse: the engine keeps the tokens it holds (the last prompt and the reply it wrote). A prompt that continues
them reads only its new tokens. The state at the end of the last prompt is also copied (inkling2.Engine.snapshot:
the short-conv history and the sliding-window KV rings), so a prompt that continues the last prompt but not the
reply (the client re-renders the assistant turn differently) goes back there and reads from that point."""
import codecs
import json
import os
import time
import uuid

import numpy as np

MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING = "<|message_model|>", "<|content_text|>", "<|content_thinking|>"
INVOKE_JSON, INVOKE_TEXT, END_MESSAGE = "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>", "<|end_message|>"
# a token that starts another speaker's turn ends the reply
TURN_ENDS = ("<|content_model_end_sampling|>", "<|endoftext|>", "<|message_user|>", "<|message_system|>",
             "<|message_tool|>")


def model_sampling(kv):
    """the sampling values that the GGUF recommends (general.sampling.*: Thinking Machines Lab's model card values
    are temperature 1.0, top-p 1.0, top-k off, min-p 0)"""
    out = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}
    for k, key in (("temperature", "general.sampling.temp"), ("top_p", "general.sampling.top_p"),
                   ("top_k", "general.sampling.top_k"), ("min_p", "general.sampling.min_p")):
        if kv.get(key) is not None:
            out[k] = type(out[k])(kv[key])
    return out


def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def parse_call(header, body, names):
    """one tool call: the message header (the tool name) and its JSON body {"name": ..., "args": {...}} -> an OpenAI
    tool call, or None when it is not a call of an offered tool"""
    try:
        obj, _ = json.JSONDecoder().raw_decode(body.strip())
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or header.strip()
    args = obj.get("args", obj.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return None
    if not isinstance(args, dict) or (names and name not in names):
        return None
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class Reply(object):
    """sorts the reply's tokens into reasoning, text and tool calls, one token at a time. feed() returns the pieces to
    stream as (kind, text); ended is True after a token that ends the reply."""

    def __init__(self, tok, names):
        self.tok, self.names = tok, names
        self.sid = {s: tok.sid[s] for s in (MESSAGE_MODEL, CONTENT_TEXT, CONTENT_THINKING, INVOKE_JSON, END_MESSAGE)
                    + TURN_ENDS + (INVOKE_TEXT,) if s in tok.sid}
        self.ends = {self.sid[s] for s in TURN_ENDS if s in self.sid}
        self.kind = "text"            # (text before any message header is answer text)
        self.header, self.body = "", ""
        self.calls, self.bad = [], []
        self.ended = False
        self.dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def feed(self, t):
        s = self.sid
        if t in self.ends:
            self.ended = True
            return []
        if t == s.get(MESSAGE_MODEL):
            self.kind, self.header = "header", ""
            return []
        if t == s.get(CONTENT_THINKING):
            self.kind = "reasoning"
            return []
        if t == s.get(CONTENT_TEXT):
            self.kind = "text"
            return []
        if t in (s.get(INVOKE_JSON), s.get(INVOKE_TEXT)):
            self.kind, self.body = "call", ""
            return []
        if t == s.get(END_MESSAGE):
            if self.kind == "call":
                c = parse_call(self.header, self.body, self.names)
                if c is not None:
                    self.calls.append(c)
                else:
                    self.bad.append(self.header + self.body)
            self.kind = "between"
            return []
        if t in self.tok.special:
            return []                 # another control token: not text
        piece = self.dec.decode(self.tok.decode_bytes(t))
        if self.kind == "header":
            self.header += piece
        elif self.kind == "call":
            self.body += piece
        elif self.kind in ("text", "reasoning") and piece:
            return [(self.kind, piece)]
        return []


class InklingBackend(object):
    name = "Inkling"

    def __init__(self, man, cfg, log=print):
        from .inkling2 import Engine
        from .inkling2_tokenizer import Inkling2Tokenizer
        self.log = log
        self.path = man["model_path"]
        self.eng = Engine(self.path, ctx=int(cfg["ctx"]), log=log)
        self.tok = Inkling2Tokenizer(self.eng.gguf.kv)
        self.tok.tokens = self.tok.vocab
        self.tok.stop_ids = {self.tok.sid[s] for s in TURN_ENDS if s in self.tok.sid}
        self.sampling = model_sampling(self.eng.gguf.kv)
        self.cached = []              # the tokens in the engine, in order
        self.snap = []                # the prompt whose end state is copied (engine.snapshot)
        self.snap_state = None
        if os.environ.get("KINDI_INKLING_SELFCHECK", "1") != "0":
            self.selfcheck()

    # ---- prompt reading
    def _render(self, req, tools):
        from .inkling2_chat import render
        kw = req.get("chat_template_kwargs") or {}
        effort = kw.get("reasoning_effort", req.get("reasoning_effort"))
        if effort is None:
            effort = os.environ.get("KINDI_INKLING_EFFORT") or None
        if isinstance(effort, str):
            try:
                effort = float(effort)
            except ValueError:
                effort = effort.strip().lower()
        thinking = kw.get("enable_thinking", req.get("enable_thinking", True)) is not False
        return render(self.path, req.get("messages") or [], tools=tools, enable_thinking=thinking,
                      reasoning_effort=effort)

    def _read(self, ids, want_logits):
        """bring the engine to the end of ids, reusing what it holds; returns (greedy next token, logits of the last
        position or None, reused tokens)"""
        eng = self.eng
        if self.cached and _lcp(self.cached, ids) == len(self.cached) and len(self.cached) < len(ids):
            start = len(self.cached)                       # the prompt continues the engine's tokens
        elif self.snap and _lcp(self.snap, ids) == len(self.snap) and len(self.snap) < len(ids):
            eng.restore(self.snap_state)                   # the prompt continues the last prompt
            start = len(self.snap)
        else:
            eng.reset()
            start = 0
        self.cached = []                                   # (until the read is done)
        body, last = ids[start:-1], ids[-1]
        if body:
            eng.prefill(body, start)
        else:
            eng.pos = start
        if want_logits:
            row = eng.forward_logits([last])[-1]
            greedy = int(np.argmax(row))
        else:
            row = None
            greedy = eng.step(last)
        self.cached = list(ids)
        self.snap_state, self.snap = eng.snapshot(), list(ids)
        return greedy, row, start

    def _next(self, token, want_logits):
        if want_logits:
            row = self.eng.forward_logits([token])[-1]
            return int(np.argmax(row)), row
        return self.eng.step(token), None

    @staticmethod
    def _pick(greedy, row, temperature, top_p, top_k, rng, min_p, penalty):
        if row is None:
            return greedy
        from .sampling import Sampler, penalize
        w = 30.0 * max(temperature, 0.1)
        cand = np.flatnonzero(row >= row.max() - w).astype(np.int64)
        sc = row[cand].astype(np.float64)
        if cand.size > 256:
            keep = np.argpartition(sc, -256)[-256:]
            cand, sc = cand[keep], sc[keep]
        if penalty is not None:
            sc = penalize(cand, sc, penalty[0], penalty[1])
        if temperature <= 0:
            return int(cand[int(np.argmax(sc))])
        return Sampler._pick(cand, sc, temperature, top_p, top_k, rng, min_p)

    def selfcheck(self):
        """before the first request: a greedy reply read twice from the start gives the same tokens, and going back
        to the end of the prompt (engine.snapshot / restore) after a reply longer than the sliding-window slack
        gives them again. A failure raises, so that the service restarts."""
        from .inkling2_chat import render
        ids = self.tok.encode(render(self.path, [{"role": "user", "content": "Count from one to ten in words."}],
                                     enable_thinking=False))
        n, eng = 24, self.eng

        def reply(first):
            out = [first]
            while len(out) < n:
                out.append(eng.step(out[-1]))
            return out
        eng.reset()
        eng.prefill(ids[:-1], 0)
        snap = eng.snapshot()
        a = reply(eng.step(ids[-1]))
        for _ in range(200):                               # more than the 128 spare slots of a window ring
            eng.step(a[-1])
        eng.restore(snap)
        b = reply(eng.step(ids[-1]))
        eng.reset()
        eng.prefill(ids[:-1], 0)
        row = eng.forward_logits([ids[-1]])[-1]           # the sampling path: the same first token
        eng.reset()
        self.cached, self.snap = [], []
        ok = a == b and len(set(a)) > 3 and int(np.argmax(row)) == a[0]
        if not ok:
            raise RuntimeError("Inkling self-check failed: %s / %s (logits argmax %d)" % (a, b, int(np.argmax(row))))
        self.log("Inkling self-check: %r, the same after going back to the prompt's end" % self.tok.decode(a)[:60])

    # ---- one request
    def generate(self, K, req, cancel=None):
        from .config import DEFAULTS
        from .oai import check
        from .server import ContextOverflow, _looping
        cfg = K.cfg
        tools, _ = check(req, can_force=False)            # a forced call or a JSON schema gets a 400, not free text
        ids = self.tok.encode(self._render(req, tools))
        if not ids:
            raise ValueError("the Inkling-2 chat template produced an empty prompt")
        if len(ids) >= self.eng.ctx:
            raise ContextOverflow(len(ids), self.eng.ctx)
        if req.get("_kindi_check"):
            return
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731
        dflt = lambda k: cfg[k] if str(cfg.get(k)) != DEFAULTS.get(k) else self.sampling[k]  # noqa: E731
        temperature = float(g("temperature", dflt("temperature")))
        top_p = float(g("top_p", dflt("top_p")))
        top_k = int(g("top_k", dflt("top_k")))
        min_p = float(g("min_p", dflt("min_p")))
        rp = float(g("repeat_penalty", cfg.get("repeat_penalty", 1.0)))
        rlast = int(g("repeat_last_n", cfg.get("repeat_last_n", 64)))
        want_logits = temperature > 0 or rp > 1.0
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        max_new = max(1, min(max_new, self.eng.ctx - len(ids)))
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        repeat_stop = True if str(cfg.get("repeat_stop")) == DEFAULTS.get("repeat_stop") else bool(int(cfg["repeat_stop"]))
        parallel = bool(req.get("parallel_tool_calls", False))
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        names = {(t.get("function") or t).get("name") for t in tools} if tools else set()
        reply = Reply(self.tok, names)
        started = time.time()
        K.busy.update(active=True, phase="prompt", prompt_tokens=len(ids), cached=0, read=0, generated=0,
                      started=started)
        K.stats["requests"] += 1
        generated, text, finish = [], "", "length"
        got = {"text": [], "reasoning": []}
        reuse = 0
        try:
            greedy, row, reuse = self._read(ids, want_logits)
            K.busy.update(read=len(ids), cached=reuse, phase="generating")
            prompt_done = time.time()
            for n in range(max_new):
                if cancel is not None and cancel.is_set():
                    finish = "stop"
                    break
                pen = (rp, set((ids + generated)[-rlast:] if rlast > 0 else ids + generated)) if rp > 1.0 else None
                token = self._pick(greedy, row, temperature, top_p, top_k, rng, min_p, pen)
                pieces = reply.feed(token)
                if reply.ended:
                    finish = "stop"
                    break
                generated.append(token)
                K.busy["generated"] = len(generated)
                if repeat_stop and len(generated) % 16 == 0 and _looping(generated):
                    self.log(self.name + ": the reply repeats a block of tokens; stopped (repeat_stop)")
                    finish = "stop"
                    break
                full = self.eng.pos >= self.eng.ctx
                if n + 1 < max_new and not full:
                    greedy, row = self._next(token, want_logits)
                    self.cached.append(token)
                hit = False
                for kind, part in pieces:
                    text += part
                    got[kind].append(part)
                    yield kind, part
                    hit = hit or bool(next((x for x in stop if x and x in text), None))
                if hit:
                    finish = "stop"
                    break
                if reply.calls and reply.kind == "between" and not parallel:
                    finish = "stop"           # one call per reply unless the client asks for parallel calls
                    break
                if full:
                    break
        finally:
            K.busy["active"] = False
        elapsed = max(0.001, time.time() - prompt_done)
        calls = reply.calls if tools else []
        if reply.bad:
            self.log(self.name + ": a tool call did not parse: %r" % reply.bad[0][:200])
        if calls:
            finish = "tool_calls"
        K.stats["prompt_tokens"] += len(ids)
        K.stats["cached_tokens"] += reuse
        K.stats["completion_tokens"] += len(generated)
        self.log(self.name + ": prompt %d (%d reused, %.2f s), %d tokens at %.1f tok/s, %s" % (
            len(ids), reuse, prompt_done - started, len(generated), len(generated) / elapsed, finish))
        yield "done", {"content": "".join(got["text"]).strip(), "tool_calls": calls,
                       "reasoning": "".join(got["reasoning"]).strip() or None, "finish": finish, "logprobs": None,
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                                 "total_tokens": len(ids) + len(generated),
                                 "prompt_tokens_details": {"cached_tokens": reuse}},
                       "timing": {"prompt_s": round(prompt_done - started, 2), "prompt_new_tokens": len(ids) - reuse,
                                  "gen_s": round(elapsed, 2), "gen_tok_s": round(len(generated) / elapsed, 2),
                                  "engine": "inkling"}}

    # ---- /health
    def drafting(self):
        """DSpark drafting is not used by the server: on the GTX 1080 it slowed decoding (docs/dev/INKLING2.md)"""
        return {"enabled": False, "method": None}
