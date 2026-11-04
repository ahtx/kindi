"""Generation for the Nanbeige backend in Kindi's server (server.Kindi with backend "kindi_nanbeige"): the Nanbeige
chat template, the thinking block as reasoning, tool calls in the XML or the JSON form, and prompt-prefix reuse (the
engine keeps the KV of the last request by position, so a new request reads only the tokens after the common prefix).
Kinds and the "done" info are the same as for the other backends."""
import codecs
import json
import os
import re
import time
import uuid

import numpy as np

from .lookup import Lookup

THINK_END = "</think>"
CALL_OPEN = "<tool_call>"


def _lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _value(v, typ):
    v = v.strip("\n")
    if typ and typ != "string":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def parse_calls(text, tools):
    """tool calls after the answer: <function=name><parameter=k>v</parameter></function>, or the JSON form"""
    types = {}
    for t in tools or []:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties") or {}
        types[f.get("name")] = {k: (p or {}).get("type") for k, p in props.items()}
    calls = []
    for body in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S):
        body = body.strip()
        m = re.match(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", body, re.S)
        if m:
            name, inner = m.group(1).strip(), m.group(2)
            args = {}
            for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", inner, re.S):
                args[k.strip()] = _value(v, types.get(name, {}).get(k.strip()))
        else:
            try:
                j = json.loads(body)
                name, args = j.get("name"), j.get("arguments", {})
            except ValueError:
                continue
        if name:
            calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


class Sorter(object):
    """splits streamed text into ("reasoning", ...) and ("text", ...) pieces; from <tool_call> on, the text is kept as
    the call tail. A partial tag at the end is held back."""

    def __init__(self, thinking):
        self.mode = "think" if thinking else "answer"
        self.buf = ""
        self.tail = ""

    def feed(self, s, final=False):
        out = []
        self.buf += s
        while True:
            if self.mode == "calls":
                self.tail += self.buf
                self.buf = ""
                return out
            tag = THINK_END if self.mode == "think" else CALL_OPEN
            kind = "reasoning" if self.mode == "think" else "text"
            i = self.buf.find(tag)
            if i >= 0:
                if self.buf[:i]:
                    out.append((kind, self.buf[:i]))
                if self.mode == "think":
                    self.buf = self.buf[i + len(tag):].lstrip("\n")
                    self.mode = "answer"
                else:
                    self.tail = self.buf[i:]
                    self.buf = ""
                    self.mode = "calls"
                continue
            keep = 0 if final else len(tag) - 1
            cut = max(0, len(self.buf) - keep)
            if cut:
                out.append((kind, self.buf[:cut]))
                self.buf = self.buf[cut:]
            return out


class NanbeigeBackend(object):
    def __init__(self, man, cfg, log=print):
        from .gguf import GGUF
        from .spm import SPM
        from .nanbeige import Engine
        self.log = log
        self.tok = SPM(GGUF(man["model_path"]).kv)
        # every control token ends the turn (</s>, <|im_end|>, <|endoftext|>, a new <|im_start|>), as llama.cpp's
        # end-of-generation set does; the model sometimes ends an answer with </s>
        self.stop_ids = {i for i, ty in enumerate(self.tok.types) if ty == 3} | {self.tok.eos}
        self.eng = Engine(man["model_path"], ctx=cfg["ctx"], log=log)
        self.cached = []            # the token at each position whose K/V is on the cards
        # prompt-lookup drafting (config key lookup): up to this many drafted tokens per check pass, 0: off
        self.lookup = max(0, min(7, int(cfg.get("lookup") or 0)))
        self.lk_stats = [0] * 6     # this request: check passes, drafted tokens, kept tokens (lookup; MTP head)
        self.mtp = False            # config key mtp_head: an MTP head file, a draft source
        hp = os.path.expanduser(cfg.get("mtp_head") or "")
        if hp and os.path.exists(hp):
            self.eng.load_mtp(hp)
            self.mtp = True
        elif hp:
            log("Nanbeige: no MTP head file at %s; drafting without it" % hp)
        if os.environ.get("KINDI_NB_SELFCHECK", "1") != "0":
            self.selfcheck()

    def selfcheck(self):
        """before the first request: the same greedy tokens from single steps (twice) and from one check pass, and
        not a degenerate run. The first use of the prompt and check buffers happens here, not in a client's
        request. A failure raises, so that the service restarts."""
        from .nanbeige_chat import render, prompt_ids
        eng = self.eng
        ids = prompt_ids(self.tok, render([{"role": "user", "content": "Count from one to ten in words."}],
                                          enable_thinking=False))
        for attempt in range(2):
            a = eng.generate(ids, 10)
            b = eng.generate(ids, 10)
            t = eng.prefill_batch(ids)
            c = [t] + eng.check(a[:8], len(ids))
            ok = a == b and c[:9] == a[:9] and len(set(a)) > 3
            if ok:
                break
            self.log("Nanbeige self-check %d failed: steps %s, again %s, check pass %s" % (attempt + 1, a, b, c[:9]))
            for cd in eng.cards:
                with eng.cp.cuda.Device(cd.dev):
                    eng.cp.cuda.Device().synchronize()
        self.cached = []                        # the K/V on the cards is the check's, not a client's
        if not ok:
            raise RuntimeError("Nanbeige self-check failed: the engine gives different tokens for the same prompt")
        self.log("Nanbeige self-check: %s" % repr(self.tok.decode(a))[:60])

    def _pick(self, greedy, temperature, top_p, top_k, rng, row=None):
        """sample the next token. The candidates are found on the card: every token whose logit is within 30
        temperatures of the best (each other token has relative probability under e^-30), at most 256 of the best.
        Only those few scores cross PCIe, not the whole vocabulary. With row, the scores are row `row` of the last
        check pass, not the last step's."""
        if temperature <= 0:
            return greedy
        from .sampling import Sampler
        cp = self.eng.cp
        c = self.eng.cards[-1]
        with cp.cuda.Device(c.dev):
            z = (c.logits.ravel() if row is None else c.LOG[row])[:len(self.tok.tokens)]
            idx = cp.flatnonzero(z >= z.max() - 30.0 * temperature)
            cand = cp.asnumpy(idx).astype(np.int64)
            sc = cp.asnumpy(z[idx]).astype(np.float64)
        if cand.size > 256:
            keep = np.argpartition(sc, -256)[-256:]
            cand, sc = cand[keep], sc[keep]
        return Sampler._pick(cand, sc, temperature, top_p, top_k, rng)

    def generate(self, K, req, cancel=None):
        from .nanbeige_chat import render, prompt_ids
        cfg = K.cfg
        tools = req.get("tools") or None
        if req.get("tool_choice") == "none":
            tools = None
        kw = req.get("chat_template_kwargs") or {}
        thinking = kw.get("enable_thinking", req.get("enable_thinking", True)) is not False
        ids = prompt_ids(self.tok, render(req.get("messages") or [], tools=tools, enable_thinking=thinking))
        if len(ids) >= self.eng.ctx:
            raise OverflowError("the prompt has %d tokens; context is %d" % (len(ids), self.eng.ctx))
        if req.get("_kindi_check"):                       # Kindi.check_prompt(): the prompt is valid, nothing to run
            return
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731
        temperature = float(g("temperature", cfg["temperature"]))
        top_p = float(g("top_p", cfg["top_p"]))
        top_k = int(g("top_k", cfg["top_k"]))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        max_new = max(1, min(max_new, self.eng.ctx - len(ids)))
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        repeat_stop = bool(cfg.get("repeat_stop"))
        from .server import _looping, THINK_CLOSE
        budget = req.get("thinking_budget_tokens") or req.get("thinking_budget") or cfg.get("thinking_budget") or 0
        budget = int(budget) if thinking else 0
        forced = False
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        started = time.time()
        reuse = min(_lcp(self.cached, ids), len(ids) - 1)
        K.busy.update(active=True, phase="prompt", prompt_tokens=len(ids), cached=reuse, read=reuse, generated=0,
                      started=started)
        K.stats["requests"] += 1
        if os.environ.get("KINDI_NB_LOGREQ"):
            self.log("Nanbeige request: %s" % json.dumps({k: v for k, v in req.items() if k not in ("messages", "tools")})
                     + " roles=%s tools=%d" % ([m.get("role") for m in req.get("messages") or []], len(tools or [])))
            with open(os.environ["KINDI_NB_LOGREQ"], "a") as f:
                f.write(json.dumps(req)[:200000] + "\n")
        sorter = Sorter(thinking)
        self.lk_stats = [0] * 6
        generated, text, finish = [], "", "length"
        got = {"text": [], "reasoning": []}
        try:
            self.cached = self.cached[:reuse]
            if len(ids) - reuse > 1:
                greedy = self.eng.prefill_batch(ids[reuse:], reuse)
            else:
                greedy = self.eng.step(ids[-1], len(ids) - 1)
            self.cached = list(ids)
            K.busy.update(read=len(ids), phase="generating")
            prompt_done = time.time()
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            pos = len(ids)
            pending = False                   # a queued step whose result (the greedy token, the logits) is not read yet
            look = Lookup(ids) if self.lookup else None
            queue = []                        # tokens a check pass chose: (token, fed, row); fed: its K/V is on the
            hrow = None                       # cards; row: the pass's row that gave it (the MTP head reads its state)
            mt = [1.0, 2.0]                   # MTP head: kept / checked drafts, decayed (prior 50%)
            for n in range(max_new):
                if pending:
                    greedy = self.eng.step_end()
                    pending = False
                if cancel is not None and cancel.is_set():
                    finish = "stop"
                    break
                if queue:
                    token, fed, hrow = queue.pop(0)
                else:
                    token, fed, hrow = self._pick(greedy, temperature, top_p, top_k, rng), False, None
                if token in self.stop_ids:
                    finish = "stop"
                    break
                generated.append(token)
                if look is not None:
                    look.add(token)
                K.busy["generated"] = len(generated)
                loop = repeat_stop and len(generated) % 16 == 0 and _looping(generated)
                if loop and sorter.mode != "think":
                    self.log("Nanbeige: the reply repeats a block of tokens; stopped (repeat_stop)")
                    finish = "stop"
                    break
                if sorter.mode == "think" and not forced and (loop or (budget and len(generated) >= budget)):
                    # thinking budget spent (or a loop in the thinking): close the thinking for the model, then let
                    # it answer, as the other backends do
                    forced = True
                    self.log("Nanbeige: thinking budget of %d tokens reached; closing the thinking"
                             % (budget or len(generated)))
                    close = self.tok.encode(THINK_CLOSE, add_bos=False)
                    # the encoder puts a lone "▁" (the SPM space prefix) before text that follows a control token;
                    # the model writes the newlines without it
                    close = [t for i, t in enumerate(close) if not (self.tok.tokens[t] == "▁" and i + 1 < len(close)
                                                                   and self.tok.tokens[close[i + 1]] == "<0x0A>")]
                    queue, hrow = [], None    # drafted tokens after this one are void (a fed token is fed again)
                    for i, t in enumerate([token] + close):
                        greedy = self.eng.step(t, pos)
                        self.cached.append(t)
                        pos += 1
                        if i:
                            generated.append(t)
                            if look is not None:
                                look.add(t)
                        piece = decoder.decode(self.tok.piece(t))
                        text += piece
                        for kind, part in sorter.feed(piece):
                            got[kind].append(part)
                            yield kind, part
                    continue
                more = n + 1 < max_new
                d = look.draft(min(self.lookup, max_new - n - 2, max(0, self.eng.ctx - pos - 2))) \
                    if (more and not fed and look is not None) else []
                head = False
                if not d and self.mtp and more and not fed and n + 2 < max_new and pos + 2 < self.eng.ctx \
                        and (mt[0] / mt[1] > 0.4 or n % 16 == 0):
                    # the MTP head drafts one token (while its acceptance pays for the check; every 16th token it
                    # drafts anyway, to measure it again)
                    d, head = [self.eng.mtp_draft(token, hrow)], True
                if fed:                   # a check pass fed it already
                    self.cached.append(token)
                    pos += 1
                elif d:                   # check the drafted tokens in one pass; keep each one the model itself picks
                    tops = self.eng.check([token] + d, pos)
                    self.cached.append(token)
                    pos += 1
                    for j, top in enumerate(tops):
                        # a draft token is kept when it is the token the model picks there (greedy: its top token;
                        # sampled: a draw from its distribution), so the output distribution is unchanged
                        x = self._pick(top, temperature, top_p, top_k, rng, row=j)
                        ok = j < len(d) and x == d[j]
                        queue.append((x, ok, j))
                        if not ok:
                            break
                    j = len(queue) - 1
                    if head:
                        mt[0], mt[1] = 0.9 * mt[0] + j, 0.9 * mt[1] + 1
                    else:
                        look.kept(j, len(d))
                    o = 3 if head else 0
                    self.lk_stats[o] += 1
                    self.lk_stats[o + 1] += len(d)
                    self.lk_stats[o + 2] += j
                elif more:                # queue the next step first: the text work and the client write below run
                    self.eng.step_begin(token, pos)          # while the cards compute it
                    self.cached.append(token)
                    pos += 1
                    pending = True
                piece = decoder.decode(self.tok.piece(token))
                text += piece
                if next((x for x in stop if x and x in text), None):
                    finish = "stop"
                    break
                for kind, part in sorter.feed(piece):
                    got[kind].append(part)
                    yield kind, part
            if pending:
                self.eng.step_end()
            for kind, part in sorter.feed("", final=True):
                got[kind].append(part)
                yield kind, part
        finally:
            K.busy["active"] = False
        elapsed = max(0.001, time.time() - prompt_done)
        if finish == "length" and generated:
            self.log("Nanbeige: hit max_tokens; last tokens %s" % [(t, self.tok.tokens[t], self.tok.types[t])
                                                                     for t in generated[-6:]])
        calls = parse_calls(sorter.tail, tools) if (tools and sorter.tail) else []
        if sorter.tail and not calls:
            got["text"].append(sorter.tail)
            yield "text", sorter.tail
        if finish == "stop" and sorter.mode == "think" and not got["text"]:
            # the model answered inside the open thinking block and ended the turn (it skips </think> on short
            # answers): that text is the answer, so the client gets a reply
            ans = "".join(got["reasoning"]).strip()
            if tools and CALL_OPEN in ans:               # a tool call inside the open thinking block
                calls = parse_calls(ans[ans.index(CALL_OPEN):], tools)
                if calls:
                    finish = "tool_calls"
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
        lk = self.lk_stats
        self.log("Nanbeige: prompt %d (%d reused, %.2f s), %d tokens at %.1f tok/s, %s%s" % (
            len(ids), reuse, prompt_done - started, len(generated), len(generated) / elapsed, finish,
            "".join(", %s %d passes, %d of %d drafted tokens kept" % (nm, lk[i], lk[i + 2], lk[i + 1])
                    for nm, i in (("lookup", 0), ("MTP head", 3)) if lk[i])))
        yield "done", {"content": "".join(got["text"]).strip(), "tool_calls": calls,
                       "reasoning": "".join(got["reasoning"]).strip() or None, "finish": finish, "logprobs": None,
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                                 "total_tokens": len(ids) + len(generated),
                                 "prompt_tokens_details": {"cached_tokens": reuse}},
                       "timing": {"prompt_s": round(prompt_done - started, 2), "prompt_new_tokens": len(ids) - reuse,
                                  "gen_s": round(elapsed, 2), "gen_tok_s": round(len(generated) / elapsed, 2)}}
