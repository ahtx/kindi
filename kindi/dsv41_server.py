"""Kindi OpenAI-compatible server for DeepSeek-V4.1-Flash (kindi.dsv41).

  GET  /health                 status, the request being served, the counters of the last request
  GET  /v1/models              the model and its real context (context_length, max_model_len, meta.n_ctx)
  POST /v1/chat/completions    messages (+ tools) -> answer; stream=true for SSE; tool calls as OpenAI tool_calls;
                               dry_run=true -> the exact prompt_tokens, no generation
  POST /v1/tokenize            {"content": text} or {"messages": [...], "tools": [...]} -> count, tokens, max_model_len
  POST /v1/detokenize          {"tokens": [...]} -> text

A prompt that does not fit is an HTTP 400 (code context_length_exceeded, OpenAI's wording) before it waits for the
engine. A stream gets an SSE comment ": kindi progress {...}" every KINDI_KEEPALIVE_S (10) seconds while nothing else
goes out (queued, reading the prompt); kindi_keepalive=true gives a non-streaming request spaces before its JSON.
After each answer the state is saved to disk in the background (--turn-min, with --cache-dir); in thinking mode the
state just before "<think>" is saved while the prompt is read. --slots N keeps the states of other conversations in
RAM (--slot-dir, a tmpfs) instead of the disk. kindi_priority (or X-Kindi-Priority): chat (the default) goes before
background; a background request yields to a chat request between prompt chunks. A request's temperature, top_p,
top_k and min_p are used (--sampling auto): with DSpark when the engine has speculative sampling, else plain sampled
decoding. --repeat-stop stops a reply caught in a loop. /health has drafting, queue and slots.

One request at a time (one engine); concurrent requests queue. The prompt is DeepSeek's own V4.1 encoding
(encoding.py from the model release, --encoding), tool calls are its DSML blocks, parsed back with the same file.
A reply that max_tokens (or the context) cuts inside its tool calls has finish_reason "length", the text before the
block as content and the cut call with the unclosed start of its arguments, like OpenAI; never the DSML markup.
A request with no max_tokens may use the remaining context (--max-tokens 0).
Decoding is greedy with DSpark drafts (the output equals plain greedy decoding); temperature and top_p are ignored.
Thinking is off unless the request sets chat_template_kwargs.enable_thinking (or thinking), or a reasoning_effort;
the thinking then comes back as reasoning_content.

Prompt cache: the tokens of the previous request stay in the caches and only the new part of a prompt is computed
(Model.prefill_cached); with --cache-dir, each prompt's state is also saved to disk (once it took at least
--ckpt-min new tokens), so a later prompt that extends an older one starts from that state. --cache-gb caps the
directory (oldest files go first). The shared prefix of agent requests (the system prompt and the tool schemas) gets
its own saved state: the tokens the prompt has in common with its system-only rendering, cut at a multiple of
--tmax (so the prompt chunks after it start where a full prefill's chunks start). A new conversation with the same
system prompt and tools then computes only its own messages. When the system prompt changes from one session to the
next only near its end (an agent's session id or date before the tool schemas), the part it shares with the last
sessions' prefixes (cache-dir/prefixes.json) gets a saved state too, so a new session starts from there.
--no-prompt-cache computes every prompt from the start (the reference for the cache check)."""
import argparse
import codecs
import importlib.util
import json
import os
import re
import select
import signal
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import dsv41

MODEL_ID = "deepseek-v4.1-flash"
# the per-session part an agent puts at the end of its system prompt (hermes: "Conversation started: <date>",
# Session ID, Model, Platform and its runtime-environment block). DeepSeek's encoding renders the tool schemas
# after the system text, so this part would sit in front of ~13K tokens of schemas and every new session (and
# every new day) would compute the schemas again. It is moved after the schemas (Engine._encode).
SYSTEM_TAIL = "\n\nConversation started: "
SYSTEM_TAIL_MAX = 4000                                  # only a short tail at the very end is moved

EFFORT = {"minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}
RESERVE = 8                                             # positions kept free after the prompt
# while nothing else is sent (waiting for the engine, reading a long prompt), a stream gets an SSE comment this often
KEEPALIVE_S = float(os.environ.get("KINDI_KEEPALIVE_S", "10"))


PRIO_CHAT, PRIO_BACKGROUND = 0, 1
_PRIO_NAMES = {"chat": PRIO_CHAT, "interactive": PRIO_CHAT, "high": PRIO_CHAT,
               "background": PRIO_BACKGROUND, "cron": PRIO_BACKGROUND, "batch": PRIO_BACKGROUND, "low": PRIO_BACKGROUND}


def priority_of(req, headers=None):
    """chat (the default) or background, from kindi_priority or the header X-Kindi-Priority"""
    v = req.get("kindi_priority") or (headers.get("X-Kindi-Priority") if headers is not None else None) or "chat"
    return _PRIO_NAMES.get(str(v).strip().lower(), PRIO_CHAT)


class Gate(object):
    """The engine: one request at a time, chat requests before background ones (0.6.0 priority classes). A
    background request that is reading its prompt stops between two chunks while a chat request waits (run's
    preempt), and goes on after it from the saved state."""

    def __init__(self):
        self.cv = threading.Condition()
        self.busy = False
        self.wait = {PRIO_CHAT: 0, PRIO_BACKGROUND: 0}

    def acquire(self, blocking=True, prio=PRIO_CHAT):
        with self.cv:
            if not blocking:
                if self.busy or self.wait[PRIO_CHAT] or self.wait[PRIO_BACKGROUND]:
                    return False
                self.busy = True
                return True
            self.wait[prio] += 1
            try:
                while self.busy or (prio > PRIO_CHAT and self.wait[PRIO_CHAT] > 0):
                    self.cv.wait()
            finally:
                self.wait[prio] -= 1
            self.busy = True
            return True

    def release(self):
        with self.cv:
            self.busy = False
            self.cv.notify_all()

    def locked(self):
        return self.busy

    def chat_waiting(self):
        return self.wait[PRIO_CHAT] > 0

    def waiting(self):
        return self.wait[PRIO_CHAT] + self.wait[PRIO_BACKGROUND]

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def _looping(gen, min_span=240):
    """True when the last tokens are one block of 4..256 tokens repeated (at least 4 times and min_span tokens):
    a reply stuck in a loop"""
    n = len(gen)
    for P in range(4, 257):
        reps = max(4, -(-min_span // P))
        if P * reps > n:
            break
        tail = gen[n - P:]
        if all(gen[n - (k + 1) * P:n - k * P] == tail for k in range(1, reps)):
            return True
    return False


def _accepts(fn, name):
    """the engine function takes this keyword (an engine hook that may not exist yet)"""
    import inspect
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _distribution(logits, temperature, top_p=1.0, top_k=0, min_p=0.0):
    """(candidate ids, probabilities): temperature first, then top-k, then top-p, then min-p (the order of the
    engine's kindi.dsv41.distribution, used when it exists, so plain and speculative sampling agree)"""
    import numpy as np
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    k = int(top_k or 0)
    idx = np.argpartition(-z, k - 1)[:k] if 0 < k < z.size else np.arange(z.size)
    idx = idx[np.argsort(-z[idx])]
    p = np.exp(z[idx] - z[idx[0]])
    p /= p.sum()
    if top_p and top_p < 1.0:
        keep = (np.cumsum(p) - p) < top_p
        idx, p = idx[keep], p[keep]
        p = p / p.sum()
    if min_p and min_p > 0.0:
        keep = p >= min_p * p[0]
        idx, p = idx[keep], p[keep]
        p = p / p.sum()
    return idx, p


class ContextOverflow(ValueError):
    """The prompt does not fit the context. OpenAI's wording, so that client libraries (litellm in the terminus-2
    and hermes harnesses, the OpenAI SDKs) see a context-window error and shorten the conversation instead of
    sending the same request again."""

    def __init__(self, n, ctx):
        self.n, self.ctx = int(n), int(ctx)
        ValueError.__init__(self, "This model's maximum context length is %d tokens. However, your messages resulted "
                            "in %d tokens (%d are needed for the reply to start). Please reduce the length of the "
                            "messages." % (self.ctx, self.n, self.n + RESERVE))


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def find_encoding(a):
    """the encoding.py to use: --encoding, $KINDI_DSV_ENCODING, then encoding.py in the --dspark directory or next to
    the model (tools/install_dsv41.sh puts it in the dspark directory). Stops with a clear message when none exists."""
    c = [a.encoding, os.environ.get("KINDI_DSV_ENCODING")]
    if a.dspark:
        c.append(os.path.join(a.dspark, "encoding.py"))
    c.append(os.path.join(os.path.dirname(os.path.abspath(a.model)), "encoding.py"))
    for p in c:
        if p and os.path.isfile(os.path.expanduser(p)):
            return os.path.expanduser(p)
    raise SystemExit("DeepSeek-V4.1 encoding.py not found (tried: %s). Download encoding/encoding.py of "
                     "deepseek-ai/DeepSeek-V4.1-Flash-1017 into the --dspark directory, or give --encoding FILE."
                     % ", ".join(p for p in c if p))


def load_encoding(path):
    spec = importlib.util.spec_from_file_location("dsv41_encoding", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _text(content):
    """OpenAI content (a string or a list of parts) as text; images are not supported by this engine"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if isinstance(p, str):
            out.append(p)
        elif p.get("type") == "text":
            out.append(p.get("text", ""))
        else:
            out.append("[%s omitted]" % p.get("type", "part"))
    return "\n\n".join(out)


def to_encoding(messages, tools):
    """OpenAI messages + tools -> the message list of DeepSeek's encode_messages (tools on the first system message)"""
    out = []
    for m in messages:
        m = dict(m)
        role = m.get("role")
        if role == "developer":
            role = m["role"] = "system"
        if role in ("system", "assistant", "tool", "user"):
            m["content"] = _text(m.get("content"))
        if role == "assistant":
            if m.get("reasoning") and not m.get("reasoning_content"):
                m["reasoning_content"] = m["reasoning"]
            if not m.get("tool_calls"):
                m.pop("tool_calls", None)
        out.append(m)
    if tools:
        if not out or out[0].get("role") != "system":
            out.insert(0, {"role": "system", "content": ""})
        out[0]["tools"] = tools
    return out


class Stream(object):
    """Splits the completion text into reasoning, content and the DSML tool-call block as it grows; text that could
    be the start of a marker (</think>, the tool-call block, a stop string) is held back until it is decided."""

    def __init__(self, enc, thinking, stops):
        self.mark = "\n\n<" + enc.dsml_token + enc.tool_calls_block_name
        self.mode = "think" if thinking else "content"
        self.stops = [s for s in stops if s]
        self.text = ""
        self.i = 0                                      # emitted up to here
        self.stopped = False                            # a stop string was found

    @staticmethod
    def _hold(s, keys):
        """length of the longest tail of s that is a proper prefix of one of keys"""
        h = 0
        for k in keys:
            for n in range(min(len(k) - 1, len(s)), h, -1):
                if s.endswith(k[:n]):
                    h = n
                    break
        return h

    def feed(self, piece, final=False, cut=False):
        """add text; returns [(kind, text)] to emit, kind "reasoning" or "content". final: the reply is over, held
        text goes out; with cut (the output limit ended it) the start of the tool-call mark is still withheld."""
        self.text += piece
        out = []
        while True:
            rest = self.text[self.i:]
            if self.mode == "think":
                j = rest.find("</think>")
                if j >= 0:
                    if j:
                        out.append(("reasoning", rest[:j]))
                    self.i += j + len("</think>")
                    self.mode = "content"
                    continue
                n = len(rest) if final else len(rest) - self._hold(rest, ["</think>"])
                if n > 0:
                    out.append(("reasoning", rest[:n]))
                    self.i += n
                return out
            if self.mode == "content":
                hits = [(rest.find(k), k) for k in [self.mark] + self.stops]
                hits = [(j, k) for j, k in hits if j >= 0]
                if hits:
                    j, k = min(hits)
                    if j:
                        out.append(("content", rest[:j]))
                    self.i += j
                    if k == self.mark:
                        self.mode = "tools"
                    else:
                        self.stopped = True
                        self.mode = "done"
                    return out
                keys = ([self.mark] if cut else []) if final else [self.mark] + self.stops
                n = len(rest) - self._hold(rest, keys)
                if n > 0:
                    out.append(("content", rest[:n]))
                    self.i += n
            return out


_PARAM_HEAD = re.compile(r' name="(.*?)" string="(true|false)">', re.S)
_CALL_HEAD = re.compile(r'\s*name="(.*?)">\n', re.S)


def truncated_tool_calls(enc, block):
    """The tool calls of a DSML block that the output limit cut off (block: the text from the tool-call mark on).
    Like OpenAI's answer to a call cut by max_tokens: complete calls as usual, and the call being written with its
    name and the start of its arguments JSON (never closed, so it does not parse; a client must not run it). A call
    whose name is not complete yet is left out. No DSML markup ever reaches the result."""
    d = enc.dsml_token
    ctag = getattr(enc, "tool_call_tag_name", " invoke")
    ptag = getattr(enc, "tool_parameter_tag_name", " parameter")
    c_open, c_close = "<%s%s" % (d, ctag), "</%s%s>" % (d, ctag)
    p_open, p_close = "<%s%s" % (d, ptag), "</%s%s>" % (d, ptag)
    js = lambda v: json.dumps(v, ensure_ascii=False)  # noqa: E731  (encoding.py's to_json)
    calls, pos = [], 0
    while True:
        i = block.find(c_open, pos)
        if i < 0:
            break
        m = _CALL_HEAD.match(block, i + len(c_open))
        if not m:
            break                                       # the name is not complete: no call
        parts, tail, closed, pos = [], None, False, m.end()
        while True:
            j, k = block.find(p_open, pos), block.find(c_close, pos)
            if k >= 0 and (j < 0 or k < j):
                closed, pos = True, k + len(c_close)
                break
            h = _PARAM_HEAD.match(block, j + len(p_open)) if j >= 0 else None
            if not h:
                break                                   # cut between parameters or inside a parameter's header
            e = block.find(p_close, h.end())
            v = block[h.end():] if e < 0 else block[h.end():e]
            if e < 0:                                   # cut inside the value
                v = v[:len(v) - Stream._hold(v, [p_close])]
                tail = js(h.group(1)) + ": " + (js(v)[:-1] if h.group(2) == "true" else v)
                break
            parts.append(js(h.group(1)) + ": " + (js(v) if h.group(2) == "true" else v))
            pos = e + len(p_close)
        args = "{" + ", ".join(parts + ([tail] if tail is not None else [])) + ("}" if closed else "")
        name = m.group(1).partition("::")[2] or m.group(1)
        calls.append({"type": "function", "function": {"name": name, "arguments": args}})
        if not closed:
            break
    return calls


class Engine(object):
    def __init__(self, a):
        self.a = a
        self.enc = load_encoding(a.encoding)
        self.model = dsv41.Model(a.model, ctx=a.ctx, tmax=a.tmax)
        if a.dspark:
            self.model.dspark_load(a.dspark)
        self.lock = Gate()                              # one request at a time, chat before background
        self.yields = 0                                 # background requests that yielded to a chat request
        self.slot_keys = []                             # RAM slots (--slots), the oldest first
        self.state = {"busy": False}
        self.last = {}
        self.cal = tuple(float(x) for x in a.cal.split(","))
        self.reqdir = os.path.join(a.cache_dir, "last-requests") if a.cache_dir else None
        self._turn_file = None                          # (tokens, path, key) of the last turn saved (after_turn)

    @property
    def waiting(self):
        """requests waiting for the engine (a warm-up and a turn save yield to them)"""
        return self.lock.waiting()

    def tokens(self, req):
        """the prompt ids of a request; also sets the shared-prefix marks that prefill() uses (engine lock held)"""
        ids, think, self.pre, self.stable, head = self._render(req, meta=True)
        if head is not None:
            self.head = head
        return ids, think

    def count(self, req):
        """the prompt ids of a request, with no side effects (no lock needed: the token count, the dry run and the
        context check before a request waits for the engine)"""
        return self._render(req, meta=False)[0]

    def _render(self, req, meta=True):
        tools = req.get("tools") or None
        ctk = req.get("chat_template_kwargs") or {}
        effort = req.get("reasoning_effort")
        think = ctk.get("enable_thinking", ctk.get("thinking", req.get("enable_thinking")))
        if isinstance(req.get("thinking"), dict):
            think = req["thinking"].get("type") == "enabled"
        if think is None:
            think = effort not in (None, "none")
        if isinstance(effort, str):
            effort = EFFORT.get(effort, "high")
        msgs = to_encoding(req.get("messages") or [], tools)
        mode = "thinking" if think else "chat"
        kw = dict(thinking_mode=mode, reasoning_effort=effort if think else None)
        kw["keep"] = bool(req.get("kindi_keep_system_tail"))
        prompt, cut = self._encode(msgs, **kw)
        ids = self.model.tok.encode(prompt)
        pre = 0                                         # the shared system + tools prefix (see prefill)
        stable = 0                                      # the same without the per-session tail
        head = None
        if meta and self.a.cache_dir and msgs and msgs[0].get("role") == "system" and len(msgs) > 1:
            head = self._common(self.model.tok.encode(self._encode(msgs[:1], **kw)[0]), ids, cut=False)
            pre = head - head % self.a.tmax
            if cut:
                stable = min(pre, self._common(self.model.tok.encode(prompt[:cut]), ids))
        return ids, bool(think), pre, stable, head

    def _common(self, head, ids, cut=True):
        c, n = 0, min(len(head), len(ids) - 1)
        while c < n and head[c] == ids[c]:
            c += 1
        return c - c % self.a.tmax if cut else c

    def _encode(self, msgs, keep=False, **kw):
        """encode_messages, with the per-session tail of the system prompt (SYSTEM_TAIL) moved after the tool
        schemas, so the schemas stay in the prefix that sessions share. Returns (prompt, where the tail starts or 0)."""
        s0 = msgs[0] if msgs else {}
        c = s0.get("content") or ""
        k = c.rfind(SYSTEM_TAIL) if s0.get("role") == "system" and s0.get("tools") else -1
        if keep or self.a.keep_system_tail or k < 0 or len(c) - k > SYSTEM_TAIL_MAX:
            return self.enc.encode_messages(msgs, **kw), 0
        tail = c[k:]
        moved = [dict(s0, content=c[:k])] + list(msgs[1:])
        prompt = self.enc.encode_messages(moved, **kw)
        schemas = self.enc.render_tools(self.enc.tools_from_openai_format(s0["tools"]))
        j = prompt.find(schemas)
        if j < 0:
            return self.enc.encode_messages(msgs, **kw), 0
        j += len(schemas)
        return prompt[:j] + tail + prompt[j:], j

    def _think_cut(self, ids):
        """in thinking mode: the position just before the generation prompt's "<think>" (even). The next prompt
        renders this turn as "</think>answer", so the state there is the one that a later prompt starts with."""
        try:
            th = list(self.model.tok.encode(getattr(self.enc, "thinking_start_token", "<think>")))
        except Exception:  # noqa: BLE001
            return None
        if not th or list(ids[-len(th):]) != th:
            return None
        cut = len(ids) - len(th)
        return cut - cut % 2

    def _slot_save(self, cd):
        """save on evict into a RAM slot (--slots N in --slot-dir, a tmpfs): the state is written there, the disk
        cache gets a link to it (so the normal lookup finds it), and the oldest slot goes when there are more
        than N. A conversation that comes back loads from RAM, with no disk write for the switch."""
        M, sd = self.model, self.a.slot_dir
        n = M.checkpoint_save(sd)
        key = M._ckpt_key(M.hist[:n])
        link = os.path.join(cd, "%s.kds" % key)
        if not os.path.exists(link):
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.join(sd, "%s.kds" % key), link)
            with open(os.path.join(cd, "index.jsonl"), "a") as f:
                f.write(json.dumps({"n": n, "key": key}) + chr(10))
        if key in self.slot_keys:
            self.slot_keys.remove(key)
        self.slot_keys.append(key)
        while len(self.slot_keys) > self.a.slots:
            old = self.slot_keys.pop(0)
            for p in (os.path.join(sd, "%s.kds" % old), os.path.join(cd, "%s.kds" % old)):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return n

    def prefill(self, ids, think=False):
        """prompt cache + disk checkpoint; returns (logits of the last token, tokens computed)"""
        M, cd = self.model, self.a.cache_dir
        if self.a.no_prompt_cache:
            M.reset()
            return M.forward(ids), len(ids)
        if cd and M.pos >= self.a.ckpt_min:
            # save on evict: a request that cannot reuse the state in memory (another client, a branch, a
            # rewritten history) would overwrite it, so the state goes to disk (or a RAM slot) first and the
            # previous conversation resumes from there instead of computing its prompt again
            c, n = 0, min(len(ids), M.pos)
            while c < n and ids[c] == M.hist[c]:
                c += 1
            if c < M.pos - self.a.tmax:
                try:
                    if self.a.slots > 0:
                        self._slot_save(cd)
                    else:
                        M.checkpoint_save(cd)
                        self._trim(cd)
                except (OSError, RuntimeError) as e:
                    _log("save on evict failed: %s" % e)
        extra = 0
        pre = getattr(self, "pre", 0)
        if cd and pre >= self.a.ckpt_min:
            # saved states at the end of the part shared with earlier sessions' prefixes and at the end of this
            # prefix, computed in that order (each a multiple of tmax, so the chunks match a full prefill's)
            pts = sorted(set(p for p in (self._shared(ids, pre, cd), getattr(self, "stable", 0), pre)
                             if p >= self.a.ckpt_min))
            for p in pts:
                if M.checkpoint_lookup(ids[:p + 1], cd)[0] < p:
                    _, k = M.prefill_cached(ids[:p], cd)
                    extra += k
                    M.checkpoint_save(cd)
                    self._trim(cd)
            self._remember(ids[:pre], cd)
        # thinking mode (0.6.0): the state just before "<think>" is saved after any prompt of --turn-min new tokens
        # (the end of a thinking reply is never a start of the next prompt, so the save after the turn skips it)
        cut = self._think_cut(ids) if think and self.a.turn_min > 0 else None
        least = self.a.turn_min if cut else self.a.ckpt_min
        if not cd or len(ids) < least + 2:
            lg, n = M.prefill_cached(ids)
            return lg, n + extra
        m = cut if cut else len(ids) - 1
        m -= m % 2
        lg, n = M.prefill_cached(ids[:m], cd)
        if n >= least:
            M.checkpoint_save(cd)
            self._trim(cd)
        return M.forward(ids[m:]), extra + n + len(ids) - m

    def _shared(self, ids, pre, cd):
        """the longest common prefix of ids[:pre] with an earlier session's prefix, cut at a multiple of tmax"""
        best = 0
        for q in self._prefixes(cd):
            c, n = 0, min(pre, len(q))
            while c < n and q[c] == ids[c]:
                c += 1
            best = max(best, c - c % self.a.tmax)
        return best

    def _prefixes(self, cd):
        try:
            return json.load(open(os.path.join(cd, "prefixes.json")))
        except (OSError, ValueError):
            return []

    def _remember(self, toks, cd, keep=8):
        ps = [q for q in self._prefixes(cd) if q != list(toks)][-(keep - 1):] + [list(toks)]
        with open(os.path.join(cd, "prefixes.json.tmp"), "w") as f:
            json.dump(ps, f)
        os.replace(os.path.join(cd, "prefixes.json.tmp"), os.path.join(cd, "prefixes.json"))

    def _trim(self, cd):
        # regular files only: a RAM slot is a link into --slot-dir (its own count limit applies, see _slot_save)
        fs = sorted((os.path.getmtime(os.path.join(cd, f)), os.path.join(cd, f)) for f in os.listdir(cd)
                    if f.endswith(".kds") and not os.path.islink(os.path.join(cd, f)))
        total = sum(os.path.getsize(f) for _, f in fs)
        cap = self.a.cache_gb * 2**30
        for _, f in fs[:-1]:
            if total <= cap:
                break
            total -= os.path.getsize(f)
            os.remove(f)

    def _store(self, req):
        """keep the last request body of each agent platform (the "Platform: x" line of the system prompt), so
        warm() can put that conversation back in memory after a restart"""
        ms = req.get("messages") or []
        if not self.reqdir or not ms or ms[0].get("role") not in ("system", "developer"):
            return
        m = re.search(r"\nPlatform: ([A-Za-z0-9_-]+)", _text(ms[0].get("content")))
        if not m:
            return
        try:
            os.makedirs(self.reqdir, exist_ok=True)
            f = os.path.join(self.reqdir, m.group(1) + ".json")
            with open(f + ".tmp", "w") as fh:
                json.dump(req, fh)
            os.replace(f + ".tmp", f)
        except OSError as e:
            _log("request store failed: %s" % e)

    def warm_fresh(self, platform, system=None, tools=None):
        """the prefix a NEW session of this platform starts with (system prompt + tool schemas + the per-session
        tail), computed and saved to disk, so a new session computes only its first message. system: the prompt
        as the agent would build it now (the agent side builds it with its own prompt builder); the session-context block that the
        agent's gateway appends after it, and the tools, come from the last stored request of the platform."""
        f = os.path.join(self.reqdir or "", "%s.json" % platform)
        s0, stools = None, None
        if self.reqdir and os.path.isfile(f):
            b = json.load(open(f))
            s0, stools = _text((b.get("messages") or [{}])[0].get("content")), b.get("tools")
        tools = stools or tools
        if system and s0:
            k = s0.find("\n\n## Current Session Context")
            if k >= 0 and "## Current Session Context" not in system:
                system += s0[k:]
        system = system or s0
        if not system or not tools:
            return {"fresh": platform, "status": "nothing to build from"}
        t0 = time.time()
        ids, _ = self.tokens({"messages": [{"role": "system", "content": system}, {"role": "user", "content": "."}],
                              "tools": tools})
        M, cd = self.model, self.a.cache_dir
        k = self.head - self.head % 2
        computed = 0
        status = "warmed"
        M.cancel = self._someone_waiting
        try:
            for p in sorted(set(x for x in (self.stable, k) if x >= self.a.ckpt_min)):
                if M.checkpoint_lookup(ids[:p + 1], cd)[0] < p:
                    _, n = M.prefill_cached(ids[:p], cd)
                    computed += n
                    M.checkpoint_save(cd)
        except dsv41.Cancelled:
            status = "yielded to a request"
            self._save_quiet()
        finally:
            M.cancel = None
        if status == "warmed" and not computed:
            status = "already saved"
        r = {"fresh": platform, "status": status, "prefix": k,
             "computed": computed, "seconds": round(time.time() - t0, 2), "built": bool(system is not s0)}
        _log(json.dumps(r))
        return r

    def _someone_waiting(self):
        return self.waiting > 0

    def _save_quiet(self):
        try:
            if self.a.cache_dir and self.model.pos >= self.a.ckpt_min:
                self.model.checkpoint_save(self.a.cache_dir)
        except (OSError, RuntimeError):
            pass

    def warm(self, platform):
        """compute the last stored request of a platform into the caches (no answer); a no-op when the state in
        memory already holds it. Returns the counters."""
        f = os.path.join(self.reqdir or "", "%s.json" % platform)
        if not self.reqdir or not os.path.isfile(f):
            return {"warm": platform, "status": "no stored request"}
        t0 = time.time()
        ids, _ = self.tokens(json.load(open(f)))
        M = self.model
        n = len(ids) - 1
        if M.pos >= n and list(M.hist[:n]) == list(ids[:n]):
            computed, status = 0, "already in memory"
        else:
            M.cancel = self._someone_waiting
            try:
                _, computed = self.prefill(ids)
                status = "warmed"
            except dsv41.Cancelled:
                computed, status = M.pos, "yielded to a request"
                self._save_quiet()
            finally:
                M.cancel = None
        r = {"warm": platform, "status": status, "prompt": len(ids), "computed": computed,
             "seconds": round(time.time() - t0, 2), "stored": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(f)))}
        _log(json.dumps(r))
        return r

    def run(self, req, emit, gone=None, preempt=None):
        """one completion; emit(kind, data) gets ("reasoning"|"content", text) while it runs and returns True when
        the client is gone; gone() says whether the client disconnected (checked between prompt chunks and decode
        steps: the work stops, the state is saved and the slot is free). preempt(): a background request stops
        between prompt chunks when it is true (a chat request waits); dsv41.Cancelled is raised, the state is
        saved. Returns the result dict."""
        if gone is not None and gone():
            raise dsv41.Cancelled("client gone before the start")
        if self.a.dump:                                 # the request bodies, for replay tests
            os.makedirs(self.a.dump, exist_ok=True)
            with open(os.path.join(self.a.dump, "%s-%s.json" % (time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:6])), "w") as f:
                json.dump(req, f)
        self._store(req)
        ids, think = self.tokens(req)
        self._last_think = think
        M, enc = self.model, self.enc
        room = M.ctx - len(ids) - RESERVE
        if room < 1:
            raise ContextOverflow(len(ids), M.ctx)
        maxn = int(req.get("max_completion_tokens") or req.get("max_tokens") or self.a.max_tokens or room)
        maxn = max(1, min(maxn, room))
        self.state.update(prompt=len(ids), computed=0, generated=0, phase="reading prompt", start_pos=None)
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        st = Stream(enc, think, stops)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        gen = []
        t0 = time.time()
        if preempt is not None:
            M.cancel = lambda: bool((gone is not None and gone()) or preempt())
        else:
            M.cancel = gone
        try:
            lg, computed = self.prefill(ids, think)
        except dsv41.Cancelled:
            self._save_after_cancel(len(ids), t0)
            raise
        finally:
            M.cancel = None
        t1 = time.time()
        self.state.update(prompt=len(ids), computed=computed, prefill_s=round(t1 - t0, 2), generated=0,
                          phase="generating")
        client_gone = gone
        gone = [False]
        got = {"reasoning": [], "content": []}
        emit0 = emit

        def emit(kind, s):
            got[kind].append(s)
            return emit0(kind, s)

        looped = [False]

        def on_tokens(toks):
            n0 = len(gen)
            gen.extend(toks)
            self.state["generated"] = len(gen)
            piece = "".join(dec.decode(M.tok.decode_bytes(t)) for t in toks if t != M.eos)
            for kind, s in st.feed(piece):
                if emit(kind, s):
                    gone[0] = True
            if client_gone is not None and client_gone():
                gone[0] = True
            # the repeat guard: a reply caught in a loop stops (checked every 16 tokens)
            if self.a.repeat_stop and len(gen) // 16 > n0 // 16 and _looping(gen):
                looped[0] = True
                _log("stopped a repeating loop after %d tokens" % len(gen))
            return gone[0] or st.stopped or looped[0]
        stats = {}
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731
        temp = float(g("temperature", self.a.temperature)) if self.a.sampling == "auto" else 0.0
        samp = dict(temperature=temp, top_p=float(g("top_p", 1.0)), top_k=int(g("top_k", 0)),
                    min_p=float(g("min_p", 0.0)))
        import numpy as np
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        spark = getattr(M, "dspark", False)
        if spark and temp > 0 and not _accepts(M.generate_dspark, "temperature"):
            spark = False                               # no DSpark rejection sampling in this engine: plain sampling
        if spark:
            kw = dict(samp, rng=rng) if temp > 0 else {}
            M.generate_dspark(None, maxn, lg=lg, on_tokens=on_tokens, auto=True, cal=self.cal, stats=stats, **kw)
        elif temp > 0:
            self._sampled(lg, maxn, on_tokens, rng, **samp)
        else:
            self._greedy(lg, maxn, on_tokens)
        stats["sampled"] = temp > 0
        stats["dspark"] = bool(spark)
        t2 = time.time()
        eos = bool(gen) and gen[-1] == M.eos
        cut = not eos and not st.stopped
        end = "" if cut else dec.decode(b"", final=True)    # cut inside a character: no U+FFFD
        for kind, s in st.feed(end, final=st.mode != "tools", cut=cut):
            emit(kind, s)
        msg = {"role": "assistant", "content": "".join(got["content"]), "reasoning_content": "".join(got["reasoning"]),
               "tool_calls": []}
        finish = "stop" if ((eos or st.stopped) and not looped[0]) else "length"
        if st.mode == "tools" and cut:                  # the output limit (or the repeat guard) cut the tool calls
            msg["tool_calls"] = truncated_tool_calls(enc, st.text[st.i:])
            for tc in msg["tool_calls"]:
                tc["id"] = "call_" + uuid.uuid4().hex[:24]
            _log("tool calls cut off after %d tokens: %d returned, finish length" % (len(gen), len(msg["tool_calls"])))
        elif st.mode == "tools":
            try:
                full = st.text + (enc.eos_token if eos else "")
                msg = enc.parse_message_from_completion_text(full, thinking_mode="thinking" if think else "chat")
                for tc in msg["tool_calls"]:
                    tc["id"] = "call_" + uuid.uuid4().hex[:24]
                finish = "tool_calls" if msg["tool_calls"] else finish
            except Exception as e:                       # malformed DSML: give the text back as content
                _log("tool-call parse failed:", repr(e)[:200])
                tail = st.text[st.i:]
                emit("content", tail)
                msg = {"role": "assistant", "content": "".join(got["content"]),
                       "reasoning_content": "".join(got["reasoning"]), "tool_calls": []}
        res = {"ids": ids, "gen": gen, "computed": computed, "finish": finish, "msg": msg, "think": think,
               "text": st.text, "prefill_s": t1 - t0, "decode_s": t2 - t1, "stats": stats}
        self.last = {"prompt": len(ids), "computed": computed, "prefill_s": round(t1 - t0, 2),
                     "prefill_tok_s": round(computed / max(t1 - t0, 1e-9), 1), "generated": len(gen),
                     "decode_tok_s": round(len(gen) / max(t2 - t1, 1e-9), 2), "finish": finish,
                     "accepted": stats.get("accepted"), "drafted": stats.get("drafted"), "steps": stats.get("steps"),
                     "dspark": stats["dspark"], "sampled": stats["sampled"], "loop_stopped": looped[0]}
        _log(json.dumps(self.last))
        return res

    def _sampled(self, lg, n, on_tokens, rng, temperature, top_p=1.0, top_k=0, min_p=0.0):
        """plain sampled decoding (no drafts): the same distribution as the engine's DSpark sampling"""
        dist = getattr(dsv41, "distribution", None) or _distribution
        for _ in range(n):
            d = dist(lg, temperature, top_p, top_k, min_p)
            idx, p = d if isinstance(d, tuple) else (np.arange(len(d)), d)   # engine: p over the vocabulary
            y = int(idx[rng.choice(len(idx), p=p)])
            if on_tokens([y]) or y == self.model.eos:
                break
            lg = self.model.forward([y])

    def drafting(self):
        """speculative drafting, for /health (Kindi Bridge shows it)"""
        M, last = self.model, self.last or {}
        on = bool(getattr(M, "dspark", False))
        out = {"enabled": on, "method": "dspark" if on else None, "drafter": "DSpark" if on else None,
               "sampling": _accepts(M.generate_dspark, "temperature") if on else None, "last": None}
        if last.get("drafted"):
            out["last"] = {"steps": last.get("steps"), "drafted": last["drafted"], "accepted": last.get("accepted", 0),
                           "acceptance": round((last.get("accepted") or 0) / float(last["drafted"]), 3)}
        return out

    def progress(self):
        """the request being served: phase, the prompt read so far (the model position), generated"""
        s = self.state
        if not s.get("busy"):
            return None
        out = {"phase": s.get("phase", "starting"), "prompt_tokens": s.get("prompt", 0),
               "generated": s.get("generated", 0)}
        if out["phase"] == "reading prompt":
            out["read"] = min(self.model.pos, s.get("prompt", 0))
        return out

    def after_turn(self):
        """after an answer has been sent: save the state at the end of the turn to disk, in the background. Earlier
        only a prompt that computed --ckpt-min or more new tokens was saved, so after a restart a short turn
        started again from an older state. The save yields: it does not start while a request waits for the
        engine (the state stays in memory for that request), and a request that comes during the save waits only
        for this one save."""
        if not self.a.cache_dir or self.a.no_prompt_cache or self.a.turn_min <= 0:
            return
        if getattr(self, "_last_think", False):
            # thinking mode: the next prompt renders this turn as "</think>answer" without the reasoning, so the
            # state at the end of the reply is never a start of it (the useful state, just before "<think>", is
            # gone once the reply is longer than the --tmax window). Chat mode renders the reply as generated.
            return
        threading.Thread(target=self._turn_save, daemon=True, name="dsv-turn-save").start()

    def _turn_save(self):
        time.sleep(0.05)                                # the next request of a fast client gets the engine first
        if self.waiting or not self.lock.acquire(blocking=False):
            return
        try:
            M, cd = self.model, self.a.cache_dir
            n = M.pos - M.pos % 2
            if n < self.a.turn_min or os.path.exists(os.path.join(cd, "%s.kds" % M._ckpt_key(M.hist[:n]))):
                return
            t0 = time.time()
            self.state.update(saving=True)
            if self.a.slots > 0:
                # RAM slots: the turn's state goes to RAM, not to the disk (a state is about 400 MB); the newest
                # --slots conversations stay there, and the state in memory goes to the disk on stop (SIGTERM)
                self._slot_save(cd)
                _log(json.dumps({"turn_saved": n, "slot": True, "seconds": round(time.time() - t0, 2)}))
                return
            M.checkpoint_save(cd)
            path = os.path.join(cd, "%s.kds" % M._ckpt_key(M.hist[:n]))
            # the previous turn of this conversation is a prefix of this one: its file is not needed any more
            prev = self._turn_file
            if prev and prev[0] < n and prev[2] == M._ckpt_key(M.hist[:prev[0]]) and prev[1] != path:
                try:
                    os.remove(prev[1])
                except OSError:
                    pass
            self._turn_file = (n, path, M._ckpt_key(M.hist[:n]))
            self._trim(cd)
            _log(json.dumps({"turn_saved": n, "seconds": round(time.time() - t0, 2)}))
        except Exception as e:  # noqa: BLE001 - saving is a convenience; serving goes on
            _log("turn save failed: %r" % e)
        finally:
            self.state.update(saving=False)
            self.lock.release()

    def _save_after_cancel(self, prompt, t0):
        M, cd = self.model, self.a.cache_dir
        saved = 0
        try:
            if cd and M.pos >= self.a.ckpt_min:
                saved = M.checkpoint_save(cd)
        except (OSError, RuntimeError) as e:
            _log("save after cancel failed: %s" % e)
        _log(json.dumps({"cancelled": "client gone", "prompt": prompt, "at": M.pos, "saved": saved,
                         "seconds": round(time.time() - t0, 2)}))

    def _greedy(self, lg, n, on_tokens):
        import numpy as np
        for _ in range(n):
            y = int(np.argmax(lg))
            if on_tokens([y]) or y == self.model.eos:
                break
            lg = self.model.forward([y])


class _Held(object):
    """releases an already acquired lock on exit"""
    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.lock.release()


def make_handler(E):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _err(self, code, msg, typ="invalid_request_error"):
            self._json(code, {"error": {"message": msg, "type": typ, "code": typ}})

        def _overflow(self, e):
            self._json(400, {"error": {"message": str(e), "type": "invalid_request_error", "param": "messages",
                                       "code": "context_length_exceeded"}})

        def _body(self):
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")

        def _tokenize(self):
            try:
                req = self._body()
            except Exception:
                return self._err(400, "invalid JSON")
            ctx = E.model.ctx
            if req.get("messages") is not None:   # a chat request: the whole prompt, as the chat endpoint builds it
                ids = E.count(req)
            else:
                text = req.get("content", req.get("prompt"))
                if not isinstance(text, str):
                    return self._err(400, "send content (text) or messages")
                ids = list(E.model.tok.encode(text))
            out = {"count": len(ids), "max_model_len": ctx, "fits": len(ids) + RESERVE <= ctx}
            if req.get("return_tokens", True) is not False:
                out["tokens"] = [int(i) for i in ids]
            return self._json(200, out)

        def _detokenize(self):
            try:
                ids = [int(i) for i in self._body().get("tokens") or []]
            except Exception:
                return self._err(400, "send tokens: a list of token ids")
            text = b"".join(E.model.tok.decode_bytes(i) for i in ids).decode("utf-8", "replace")
            return self._json(200, {"prompt": text, "content": text})

        def _auth(self):
            if not E.a.api_key:
                return True
            if self.headers.get("Authorization", "") == "Bearer " + E.a.api_key:
                return True
            self._err(401, "invalid API key", "authentication_error")
            return False

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self._json(200, {"status": "ok", "model": MODEL_ID, "state": E.state, "last": E.last,
                                        "ctx": E.model.ctx, "dspark": bool(getattr(E.model, "dspark", False)),
                                        "drafting": E.drafting(),
                                        "queue": {"chat": E.lock.wait[PRIO_CHAT],
                                                  "background": E.lock.wait[PRIO_BACKGROUND], "yields": E.yields},
                                        "slots": len(E.slot_keys)})
            if not self._auth():
                return
            if self.path.rstrip("/") == "/v1/models":
                ctx = E.model.ctx
                return self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                        "created": 0, "owned_by": "kindi", "context_length": ctx,
                                        "max_model_len": ctx, "max_context_length": ctx, "meta": {"n_ctx": ctx}}]})
            self._err(404, "not found")

        def do_POST(self):
            if not self._auth():
                return
            if self.path.rstrip("/") == "/v1/kindi/warm":
                try:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    with E.lock:
                        E.state.update(busy=True, since=time.strftime("%H:%M:%S"))
                        try:
                            pf = str(body.get("platform") or "telegram")
                            r = {}
                            if body.get("fresh"):
                                r["fresh"] = E.warm_fresh(pf, body.get("system"), body.get("tools"))
                            if body.get("session", True) and not E.waiting:
                                r.update(E.warm(pf))
                            elif body.get("session", True):
                                r.update({"warm": pf, "status": "skipped: a request is waiting"})
                        finally:
                            E.state.update(busy=False)
                    return self._json(200, r)
                except Exception as e:
                    _log("warm failed:", repr(e))
                    return self._err(500, repr(e), "server_error")
            if self.path.rstrip("/") in ("/v1/tokenize", "/tokenize"):
                return self._tokenize()
            if self.path.rstrip("/") in ("/v1/detokenize", "/detokenize"):
                return self._detokenize()
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._err(404, "not found")
            try:
                req = self._body()
            except Exception:
                return self._err(400, "invalid JSON")
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(req.get("stream"))
            usage_in_stream = bool((req.get("stream_options") or {}).get("include_usage"))
            # every request is counted before it waits for the engine: a prompt that does not fit gets its HTTP 400
            # at once, and a dry run (dry_run: true) ends here with the exact prompt_tokens
            try:
                n = len(E.count(req))
            except Exception as e:
                return self._err(400, "the request cannot be rendered: %s" % e)
            fits = n + RESERVE <= E.model.ctx
            if req.get("dry_run") or req.get("kindi_dry_run"):
                return self._json(200, {"object": "kindi.prompt_count", "model": MODEL_ID, "fits": fits,
                                        "max_model_len": E.model.ctx,
                                        "usage": {"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n}})
            if not fits:
                return self._overflow(ContextOverflow(n, E.model.ctx))
            keepalive = stream or bool(req.get("kindi_keepalive")) or \
                self.headers.get("X-Kindi-Keepalive", "") in ("1", "true")
            wlock = threading.Lock()                    # the ticker and the request thread both write
            mine = {"started": False, "done": False}

            def write(b):
                with wlock:
                    self.wfile.write(b)
                    self.wfile.flush()

            def chunk(delta, finish=None, usage=None):
                c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                     "choices": [] if usage is not None and delta is None else
                     [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if usage is not None:
                    c["usage"] = usage
                write(b"data: " + json.dumps(c).encode() + b"\n\n")

            def ticker():
                """an SSE comment with the progress (a stream), or a space before the JSON (kindi_keepalive), every
                KEEPALIVE_S while nothing else goes out"""
                while not mine["done"]:
                    time.sleep(KEEPALIVE_S)
                    if mine["done"]:
                        return
                    note = (E.progress() or {"phase": "finishing"}) if mine["started"] else {"phase": "queued"}
                    try:
                        write((": kindi progress " + json.dumps(note) + "\n\n").encode() if stream else b" ")
                    except OSError:
                        return
            if keepalive:
                # the status is sent now (the prompt is known to fit); later errors go in the body
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                if stream:
                    chunk({"role": "assistant", "content": ""})
                threading.Thread(target=ticker, daemon=True, name="dsv-keepalive").start()

            def client_gone():
                """the client closed the connection (hermes /stop, a new session, a timeout)"""
                try:
                    r, _, _ = select.select([self.connection], [], [], 0)
                    return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
                except (OSError, ValueError):
                    return True

            def emit(kind, s):
                if not stream or not s:
                    return False
                try:
                    chunk({"reasoning_content" if kind == "reasoning" else "content": s})
                    return False
                except OSError:
                    return True
            def reply(code, obj):
                """a JSON answer: after an early 200 (kindi_keepalive) the body goes on the open response"""
                if keepalive and not stream:
                    write(json.dumps(obj).encode())
                else:
                    self._json(code, obj)

            answered = False
            prio = priority_of(req, self.headers)
            yielded = [False]
            while True:
                yielded[0] = False
                E.lock.acquire(prio=prio)
                mine["started"] = True
                with _Held(E.lock):
                    E.state.update(busy=True, since=time.strftime("%H:%M:%S"), phase="starting")
                    try:
                        try:
                            r = E.run(req, emit, gone=client_gone,
                                      preempt=E.lock.chat_waiting if prio > PRIO_CHAT else None)
                        except dsv41.Cancelled:
                            if prio > PRIO_CHAT and not client_gone():
                                # a background request yields to a chat request while it reads its prompt: its state is
                                # saved, and it goes on after the chat request
                                E.yields += 1
                                yielded[0] = True
                                _log("a background request yields to a chat request")
                                continue
                            self.close_connection = True
                            return
                        except ValueError as e:
                            over = isinstance(e, ContextOverflow)
                            err = {"message": str(e), "type": "invalid_request_error",
                                   "code": "context_length_exceeded" if over else "invalid_request_error"}
                            if stream:
                                write(b"data: " + json.dumps({"error": err}).encode() + b"\n\n")
                                write(b"data: [DONE]\n\n")
                                return
                            return reply(400, {"error": err})
                        usage = {"prompt_tokens": len(r["ids"]), "completion_tokens": len(r["gen"]),
                                 "total_tokens": len(r["ids"]) + len(r["gen"]),
                                 "prompt_tokens_details": {"cached_tokens": len(r["ids"]) - r["computed"]}}
                        m = r["msg"]
                        tcs = [{"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}
                               for i, tc in enumerate(m.get("tool_calls") or [])]
                        if stream:
                            try:
                                if tcs:
                                    chunk({"tool_calls": tcs})
                                chunk({}, r["finish"])
                                if usage_in_stream:
                                    chunk(None, usage=usage)
                                write(b"data: [DONE]\n\n")
                                answered = True
                            except OSError:
                                pass
                            self.close_connection = True
                            return
                        out = {"role": "assistant", "content": m.get("content") or (None if tcs else "")}
                        if r["think"]:
                            out["reasoning_content"] = m.get("reasoning_content", "")
                        if tcs:
                            out["tool_calls"] = [{k: v for k, v in tc.items() if k != "index"} for tc in tcs]
                        reply(200, {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                                    "choices": [{"index": 0, "message": out, "finish_reason": r["finish"]}],
                                    "usage": usage})
                        answered = True
                    except Exception as e:
                        _log("request failed:", repr(e))
                        if not stream:
                            try:
                                reply(500, {"error": {"message": repr(e), "type": "server_error", "code": "server_error"}})
                            except OSError:
                                pass
                    finally:
                        if not yielded[0]:
                            mine["done"] = True
                        E.state.update(busy=False, phase="")
                        if answered:
                            E.after_turn()                  # the client has its answer: save the turn in the background
                break
    return H


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kindi.dsv41_server")
    ap.add_argument("model")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--tmax", type=int, default=512)
    ap.add_argument("--dspark", default=None, help="DSpark drafter directory (converted by kindi.dsv41_dspark)")
    ap.add_argument("--cal", default="1,0", help="DSpark draft length calibration a,b")
    ap.add_argument("--encoding", default=None,
                    help="DeepSeek-V4.1 encoding.py (encoding/encoding.py of the model release; default: "
                         "$KINDI_DSV_ENCODING, else encoding.py in the --dspark directory or next to the model)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--ckpt-min", type=int, default=512, help="save a prompt state once it took this many new tokens")
    ap.add_argument("--slots", type=int, default=0,
                    help="RAM slots for other conversations: save on evict writes to --slot-dir (a tmpfs) and keeps the "
                         "newest N (0: save on evict to --cache-dir)")
    ap.add_argument("--slot-dir", default="/dev/shm/kindi-dsv41-slots")
    ap.add_argument("--sampling", choices=("auto", "off"), default="auto",
                    help="auto: a request's temperature, top_p, top_k and min_p are used; off: always greedy")
    ap.add_argument("--temperature", type=float, default=0.0, help="the temperature of a request that sets none")
    ap.add_argument("--repeat-stop", type=int, default=1, help="1: stop a reply caught in a repeating loop")
    ap.add_argument("--turn-min", type=int, default=64,
                    help="after each answer, save the state of a conversation this long or longer, in the background "
                         "(0: off; needs --cache-dir)")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="max_tokens of a request that sets none (0: up to the remaining context)")
    ap.add_argument("--no-prompt-cache", action="store_true", help="compute every prompt from the start")
    ap.add_argument("--keep-system-tail", action="store_true",
                    help="keep an agent's per-session system-prompt tail in front of the tool schemas (see SYSTEM_TAIL)")
    ap.add_argument("--dump", default=None, help="write every request body to this directory (replay tests)")
    ap.add_argument("--api-key", default=os.environ.get("KINDI_API_KEY"))
    a = ap.parse_args(argv)
    a.encoding = find_encoding(a)
    E = Engine(a)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(E))

    def stop(sig, frame):
        # on stop, wait for the request being served, then save the state in memory so the conversation resumes
        # from disk after the restart (warm turns compute too little to be saved as they go)
        def save():
            with E.lock:
                try:
                    if a.cache_dir and not a.no_prompt_cache and E.model.pos >= a.ckpt_min:
                        t0 = time.time()
                        M = E.model
                        link = os.path.join(a.cache_dir, "%s.kds" % M._ckpt_key(M.hist[:M.pos - M.pos % 2]))
                        if os.path.islink(link):            # a RAM slot (lost on reboot): write the real file
                            os.remove(link)
                        E.model.checkpoint_save(a.cache_dir)
                        _log("state saved on stop: %d tokens in %.1f s" % (E.model.pos, time.time() - t0))
                except Exception as e:
                    _log("save on stop failed: %r" % e)
            os._exit(0)
        threading.Thread(target=save, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    _log("serving %s on %s:%d (ctx %d, DSpark %s, cache %s)" % (MODEL_ID, a.host, a.port, a.ctx, bool(a.dspark),
                                                                   a.cache_dir))
    srv.serve_forever()


if __name__ == "__main__":
    main()
