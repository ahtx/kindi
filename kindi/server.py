"""Kindi OpenAI-compatible server.\n\n  GET  /health                 status, live progress of the request being served, GPU memory\n  GET  /v1/models              the model and its real context (context_length, max_model_len, meta.n_ctx)\n  POST /v1/chat/completions    messages (+ tools) -> answer; stream=true for SSE; tool calls as OpenAI tool_calls;\n                               dry_run=true -> the exact prompt_tokens, no generation\n  POST /v1/tokenize            {"content": text} or {"messages": [...], "tools": [...]} -> count, tokens\n  POST /v1/detokenize          {"tokens": [...]} -> text\n\nA prompt that does not fit is an HTTP 400 (code context_length_exceeded, OpenAI's wording) before the request waits\nfor the engine. A stream gets an SSE comment ": kindi progress {...}" every KINDI_KEEPALIVE_S (10) seconds while\nnothing else goes out; kindi_keepalive=true gives a non-streaming request spaces before its JSON. After each answer\nthe end-of-turn state is saved to disk in the background (turn_save, see cache.Turns).\n\nOne request at a time (one engine on one GPU); concurrent requests queue. Thinking is off unless the request sets\nchat_template_kwargs.enable_thinking (or top-level enable_thinking); a stream then carries the thinking as\ndelta.reasoning_content. thinking_budget_tokens (or thinking_budget) caps it: at the budget the server closes the\nthinking itself and the model goes on to answer. temperature 0 = greedy (token-exact with llama.cpp in the exact head\nmode)."""
import asyncio
import codecs
import concurrent.futures
import json
import os
import threading
import time
import uuid

import numpy as np

from . import COMMIT, __version__, chat
from .cache import Cancelled, Runner
from .engine import EOS
from .sampling import Sampler


THINK_CLOSE = "\n\nI have thought enough; time to act.\n</think>\n\n"

# while a long prompt is read nothing else is sent: an SSE comment this often keeps the stream alive and tells the
# client how far the read is (Node's fetch, which pi uses, drops a body that is silent for 300 s)
KEEPALIVE_S = float(os.environ.get("KINDI_KEEPALIVE_S", "10"))


class ContextOverflow(OverflowError):
    """The prompt does not fit the context. The message is OpenAI's wording: client libraries (litellm, which the
    terminus-2 agent harness uses, and the OpenAI SDKs) recognise it as a context-window error and shorten the
    conversation in place of sending the same request again."""

    def __init__(self, n, ctx, reserve=1):
        self.n, self.ctx = int(n), int(ctx)
        OverflowError.__init__(self, "This model's maximum context length is %d tokens. However, your messages "
                               "resulted in %d tokens (%d are needed for the reply to start). Please reduce the length "
                               "of the messages." % (self.ctx, self.n, self.n + reserve))


def _looping(gen, min_span=240):
    """True when the last tokens are one block of 4..256 tokens repeated (at least 4 times and min_span tokens):\n    a model stuck in a loop. Short legitimate repeats (table rules, a few identical lines) stay below it."""
    n = len(gen)
    for P in range(4, 257):
        reps = max(4, -(-min_span // P))
        if P * reps > n:
            break
        tail = gen[n - P:]
        if all(gen[n - (k + 1) * P:n - k * P] == tail for k in range(1, reps)):
            return True
    return False



def _two_cards():
    """Gemma 4 in tensor parallel: KINDI_G4_TP=1 always, 0 never; auto (the default) when exactly two GPUs of the
    same kind are visible"""
    mode = os.environ.get("KINDI_G4_TP", "auto")
    if mode in ("0", "1"):
        return mode == "1"
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n != 2:
            return False
        names = [cp.cuda.runtime.getDeviceProperties(i)["name"] for i in range(2)]
        return names[0] == names[1]
    except Exception:
        return False

class Kindi(object):
    """Engine + tokenizer + cache + sampler, shared by the HTTP layer and `kindi chat`."""

    def __init__(self, cfg, log=print):
        try:
            import cupy as cp
        except ImportError:              # a CPU-only host (KINDI_DEVICE=cpu)
            cp = None
        from .engine import make_engine
        from .tokenizer import Tok
        self.cp, self.cfg, self.log = cp, cfg, log
        pack = cfg["pack"]
        if not os.path.exists(os.path.join(pack, "kindi_pack.json")):
            raise SystemExit("kindi: no model pack at %s - run: kindi convert <model.gguf>" % pack)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        self.pack_id = (man.get("source_sha256") or "pack")[:12]
        self.native = man.get("backend") == "kindi_gemma4"
        self.busy = {"active": False, "phase": "", "prompt_tokens": 0, "cached": 0, "read": 0, "generated": 0,
                     "started": 0.0}
        self.nanbeige = None
        if man.get("backend") == "kindi_nanbeige":        # llama-arch K/I-quant GGUF (Nanbeige 4.3 33B)
            from .nanbeige_gen import NanbeigeBackend
            self.variant = "nanbeige"
            self.nanbeige = NanbeigeBackend(man, cfg, log=log)
            self.eng, self.tok, self.runner, self.sample = self.nanbeige.eng, self.nanbeige.tok, None, None
        elif man.get("backend") == "kindi_lightning":     # Nemotron-H MoE GGUF (Nemotron 3.5 Lightning 30B-A3B)
            from .lightning_gen import LightningBackend
            self.variant = "lightning"
            self.nanbeige = LightningBackend(man, cfg, log=log)   # (the same backend interface as Nanbeige's)
            self.eng, self.tok, self.runner, self.sample = self.nanbeige.eng, self.nanbeige.tok, None, None
        elif self.native:
            from .gemma4 import Engine as NativeEngine
            from .gemma4_tokenizer import Tok as GemmaTok
            self.variant = "gemma4"
            self.model_path = man["model_path"]
            man["pack_path"] = pack
            self.tok = GemmaTok(self.model_path)
            if _two_cards():
                from .gemma4_tp import TPEngine as NativeEngine
            self.eng = NativeEngine(man, ctx=cfg["ctx"], log=log)
            from .gemma4_cache import G4Cache
            self.g4cache = G4Cache(self.eng, log=log)
            self.runner = None
            self.sample = None
        else:
            self.variant = chat.template_variant(pack)      # qwen35 (2B/4B/9B) or bonsai27 (the 27B template)
            clusters = None
            cl_path = os.path.join(pack, "head_clusters.npz")
            if os.path.exists(cl_path):
                clusters = dict(np.load(cl_path))
            self.eng = make_engine(pack, ctx=cfg["ctx"], clusters=clusters, log=log)
            self.tok = Tok(self.eng.vocab, os.path.join(pack, "merges.txt"))
            self.runner = Runner(self.eng, cfg, self.pack_id, log, progress=self.busy)
            self.sample = Sampler(self.eng)
        self.lock = threading.RLock()        # reentrant: a background request runs waiting chat requests inside its own turn
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0}
        self.t0 = time.time()

    def head_mode(self):
        """exact | fast: the state file (written by `kindi head`) wins over the config."""
        try:
            v = open(os.path.join(self.cfg["state_dir"], "head_mode")).read().strip().lower()
            if v in ("exact", "fast"):
                return v
        except OSError:
            pass
        return self.cfg["head"]

    def head_active(self):
        """the head that runs: "fast" only when the engine has a fast head (the 27B, Nanbeige and the CPU engine have none)"""
        m = self.head_mode()
        if m == "fast" and getattr(self.eng, "hsl", None) is None and getattr(self.eng, "fast_centroids", None) is None:
            return "exact"
        return m

    def _pick(self, nxt, temperature, top_p, top_k, rng, min_p=0.0, penalty=None):
        """the next token: already drawn by speculative sampling, or sampled now"""
        ps = getattr(self.eng, "spec_presampled", None)
        if ps is not None:
            self.eng.spec_presampled = None
            return ps
        return self.sample(nxt, temperature, top_p, top_k, rng, min_p, penalty)

    def _g4_text(self,token):
        return self.tok.decode_bytes(token).decode("utf-8","replace")

    def _g4_call_token(self,logits,grammar,gst,temperature,top_p,top_k,rng):
        """the next token inside a tool call: the best candidates that keep the call syntax valid. The search widens
        (64, 1,024, 8,192 candidates) until one fits; if none fits, the free choice stands and the parser decides."""
        for width in (64,1024,8192):
            width=min(width,logits.size)
            idx=self.cp.argpartition(logits,-width)[-width:]
            cand=self.cp.asnumpy(idx).astype(np.int64,copy=False); sc=self.cp.asnumpy(logits[idx]).astype(np.float64)
            order=np.argsort(-sc); cand,sc=cand[order],sc[order]
            keep=[i for i,c in enumerate(cand) if int(c) not in self.tok.stop_ids
                  and grammar.advance(gst,self._g4_text(int(c))) is not None]
            if keep:
                cand,sc=cand[keep],sc[keep]
                if temperature<=0: return int(cand[0])
                return Sampler._pick(cand,sc,temperature,top_p,top_k,rng)
        return int(self.cp.argmax(logits).item())

    def _generate_gemma(self,req,cancel=None):
        from .gemma4_chat import render as render_gemma
        from . import gemma4_tools
        cfg=self.cfg; messages=gemma4_tools.prepare_messages(req.get("messages") or [])
        tools=req.get("tools") or None
        if req.get("tool_choice")=="none": tools=None
        kw=dict(req.get("chat_template_kwargs") or {})
        kw["enable_thinking"]=bool(kw.get("enable_thinking",req.get("enable_thinking",False)))
        kw["conversational_mode"]=bool(kw.get("conversational_mode",False))
        prompt=render_gemma(self.model_path,messages,tools,kw,True)
        thinking_open=prompt.endswith(gemma4_tools.THINK_OPEN+"thought\n")   # after a tool response, thinking on
        sorter=gemma4_tools.Stream(thinking_open)
        ids=self.tok.encode(prompt)
        if not ids: raise ValueError("Gemma 4 chat template produced an empty prompt")
        if len(ids)>=self.eng.ctx:
            raise ContextOverflow(len(ids),self.eng.ctx)
        if req.get("_kindi_check"): return ids            # check_prompt(): the prompt is valid, nothing to run
        self.eng.shortlist=cfg["fast_clusters"] if self.head_mode()=="fast" and self.eng.fast_centroids is not None else 0
        max_new=int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        max_new=max(1,min(max_new,self.eng.ctx-len(ids)))
        temperature=float(req.get("temperature",cfg["temperature"]))
        top_p=float(req.get("top_p",cfg["top_p"])); top_k=int(req.get("top_k",cfg["top_k"]))
        # OpenAI logprobs: the model's own distribution (the logits at temperature 1), as llama.cpp reports it
        want_lp=bool(req.get("logprobs")); n_lp=max(0,min(20,int(req.get("top_logprobs") or 0)))
        lp_out=[] if want_lp else None
        seed=req.get("seed")
        rng=np.random.RandomState(seed if seed is not None else int(time.time()*1000)%2**31)
        stop=req.get("stop") or []
        if isinstance(stop,str): stop=[stop]
        self.eng.reset()
        started=time.time()
        self.busy.update(active=True,phase="prompt",prompt_tokens=len(ids),cached=0,read=0,generated=0,started=started)
        self.stats["requests"]+=1
        self.log("Gemma 4 request: %d prompt tokens (max %d new)"%(len(ids),max_new))
        try:
            batch_size=max(1,min(512,int(os.environ.get("KINDI_G4_PREFILL_BATCH","512"))))   # the KV ring takes 512 at once
            end_prompt=len(ids)-1
            pos=self.g4cache.prepare(ids)                  # positions already on the card (the prompt cache)
            self.busy["cached"]=pos
            while pos<end_prompt:
                if cancel is not None and cancel.is_set():
                    self.busy["active"]=False; return
                end=min(end_prompt,pos+batch_size)
                if end-pos>1:
                    self.eng.forward_batch(ids[pos:end],pos)
                else:
                    self.eng.forward(ids[pos],pos,logits=False)
                self.g4cache.advanced(ids[:end],checkpoint=True)
                pos=end;self.busy["read"]=pos
            if cancel is not None and cancel.is_set():
                self.busy["active"]=False;return
            logits=self.eng.forward(ids[-1],len(ids)-1,logits=True)
            self.g4cache.advanced(ids,checkpoint=True)
            self.busy["read"]=len(ids)
            prompt_done=time.time()
            self.busy["phase"]="generating"
            decoder=codecs.getincrementaldecoder("utf-8")(errors="replace")
            content=[]; generated=[]; finish="length"; text=""
            # inside a tool call, only tokens that keep the call syntax valid are sampled (as llama.cpp's grammar does)
            grammar=gemma4_tools.CallGrammar([(t.get("function") or t).get("name") for t in tools]) if tools else None
            call_open=self.tok.ids.get(gemma4_tools.CALL_OPEN)
            parallel=bool(req.get("parallel_tool_calls",False))
            gst=None
            for n in range(max_new):
                if cancel is not None and cancel.is_set():
                    finish="stop"; break
                if gst is not None:
                    token=self._g4_call_token(logits,grammar,gst,temperature,top_p,top_k,rng)
                elif temperature<=0:
                    token=int(self.cp.argmax(logits).item())
                else:
                    if top_k>0 and top_k<logits.size:
                        shortlist=self.cp.argpartition(logits,-top_k)[-top_k:]
                        scores=self.cp.asnumpy(logits[shortlist])
                        candidates=self.cp.asnumpy(shortlist).astype(np.int64,copy=False)
                    else:
                        scores=self.cp.asnumpy(logits)
                        candidates=np.arange(scores.size,dtype=np.int64)
                    token=Sampler._pick(candidates,scores,temperature,top_p,top_k,rng)
                if token in self.tok.stop_ids:
                    finish="stop"; break
                if want_lp:
                    lz=logits.astype(self.cp.float32); m=lz.max()
                    lse=float((m+self.cp.log(self.cp.exp(lz-m).sum())).item())
                    ent={"token":self._g4_text(token),"id":int(token),"logprob":float(lz[token].item())-lse}
                    if n_lp:
                        ti=self.cp.argpartition(-lz,n_lp)[:n_lp]; tv=self.cp.asnumpy(lz[ti]); ti=self.cp.asnumpy(ti)
                        ent["top_logprobs"]=[{"token":self._g4_text(int(ti[j])),"id":int(ti[j]),"logprob":float(tv[j])-lse}
                                             for j in np.argsort(-tv)]
                    lp_out.append(ent)
                generated.append(token); self.busy["generated"]=len(generated)
                if n+1<max_new:      # queue the next step first: the text work below runs while the GPU computes it
                    logits=self.eng.forward(token,len(ids)+n); self.g4cache.append(token)
                piece=decoder.decode(self.tok.decode_bytes(token))
                text+=piece
                hit=next((x for x in stop if x and x in text),None)
                if hit:
                    text=text[:text.index(hit)]; finish="stop"; break
                if piece:
                    content.append(piece)
                    for kind,part in sorter.feed(piece):
                        # with tools, the text goes out once, parsed, at the end: a client never sees call markup
                        if kind=="text" and tools: continue
                        yield kind,part
                if grammar is not None:
                    if gst is None:
                        if token==call_open: gst=grammar.start()
                    else:
                        gst=grammar.advance(gst,self._g4_text(token))
                        if grammar.done(gst):
                            gst=None
                            if not parallel:               # one call per reply, as llama.cpp without parallel_tool_calls
                                finish="stop"; break
            self.busy["active"]=False
        except BaseException:
            self.busy["active"]=False
            raise
        elapsed=max(0.001,time.time()-prompt_done)
        self.stats["prompt_tokens"]+=len(ids)
        self.stats["cached_tokens"]+=self.busy.get("cached",0)
        self.stats["completion_tokens"]+=len(generated)
        self.log("Gemma 4 done: %d tokens in %.1fs (%.1f tok/s, %s)"%
                 (len(generated),elapsed,len(generated)/elapsed,finish))
        raw=(gemma4_tools.THINK_OPEN+"thought\n" if thinking_open else "")+"".join(content)
        reply,reasoning,calls=gemma4_tools.split_reply(raw)
        if not tools and calls: reply,calls=raw.strip(),[]
        if tools and not calls and gemma4_tools.CALL_CLOSE in raw:
            self.log("Gemma 4: call markup but no call parsed; the reply starts %r"%raw[:160])
        if calls: finish="tool_calls"
        if tools and reply: yield "text",reply
        yield "done",{"content":reply,"tool_calls":calls,"reasoning":reasoning,"finish":finish,"logprobs":lp_out,
          "usage":{"prompt_tokens":len(ids),"completion_tokens":len(generated),"total_tokens":len(ids)+len(generated),
                   "prompt_tokens_details":{"cached_tokens":self.busy.get("cached",0)}},
          "timing":{"prompt_s":round(prompt_done-started,2),"prompt_new_tokens":len(ids),
                    "gen_s":round(elapsed,2),"gen_tok_s":round(len(generated)/elapsed,2)}}

    def check_prompt(self, req):
        """Renders and counts the prompt of a request without the GPU or the lock, and returns its token ids. Raises
        ContextOverflow (the prompt does not fit the context) or ValueError (the prompt is not valid), so that a
        request gets an HTTP 400 at once: before the first byte of a stream, and before it waits for the engine."""
        g = self.generate(dict(req, _kindi_check=True))
        try:
            while True:
                next(g)
        except StopIteration as e:
            return list(e.value or [])

    def encode_text(self, text, special=True):
        """the token ids of plain text (no chat template)"""
        try:
            return list(self.tok.encode(text, add_bos=special))     # SentencePiece tokenizers (Nanbeige)
        except TypeError:
            return list(self.tok.encode(text))

    def decode_ids(self, ids):
        dec = getattr(self.tok, "decode", None)
        if dec is not None:
            return dec([int(i) for i in ids])
        return b"".join(self.tok.decode_bytes(int(i)) for i in ids).decode("utf-8", "replace")

    def after_turn(self, locked=False):
        """after a request has been answered, on the engine thread: queue the end-of-turn state for disk.
        locked: the caller holds self.lock already (the stream generator)."""
        r = getattr(self, "runner", None)
        if r is None or not hasattr(r, "turn_end"):
            return
        if locked:
            r.turn_end()
        else:
            with self.lock:
                r.turn_end()

    def drafting(self):
        """speculative drafting, for /health (Kindi Bridge shows it): on or off, the method, and the last request's
        drafts and how many were accepted"""
        last = getattr(self, "last_spec", None)
        out = {"enabled": False, "method": None, "drafter": None, "last": None}
        if self.nanbeige is not None:
            m = [n for n, on in (("lookup", int(self.cfg.get("lookup") or 0) > 0),
                                 ("mtp", bool(self.cfg.get("mtp_head")))) if on]
            out.update(enabled=bool(m), method="+".join(m) or None)
        elif not self.native and hasattr(self.eng, "spec_begin"):
            mode = os.environ.get("KINDI_SPEC", "0")
            if mode not in ("0", "", "off"):
                dr = getattr(self.eng, "drafter", None)
                out.update(enabled=True, method=getattr(self.eng, "spec_mode", None) or mode,
                           drafter=getattr(getattr(dr, "eng", None), "pack", None) or (type(dr).__name__ if dr else None))
        if last and last.get("drafted"):
            out["last"] = {"passes": last.get("passes", 0), "drafted": last["drafted"], "accepted": last.get("accepted", 0),
                           "acceptance": round(last.get("accepted", 0) / float(last["drafted"]), 3)}
        return out

    def progress(self):
        """the request being served: phase, tokens reused, read and to read, generated"""
        b = self.busy
        if not b["active"]:
            return None
        return {"phase": b["phase"], "elapsed_s": round(time.time() - b["started"]),
                "prompt_tokens": b["prompt_tokens"], "reused": b["cached"], "read": b["read"],
                "to_read": max(0, b["prompt_tokens"] - b["cached"]), "generated": b["generated"]}

    def generate(self, req, cancel=None):
        """Yields ("text", piece) while generating, then ("done", info). cancel: a threading.Event set when the
        client disconnects (prompt reading and generation stop at the next token)."""
        if self.nanbeige is not None:
            return (yield from self.nanbeige.generate(self, req, cancel))
        if self.native:
            return (yield from self._generate_gemma(req, cancel))
        cfg = self.cfg
        messages = req.get("messages") or []
        tools = req.get("tools") or None
        if req.get("tool_choice") == "none":
            tools = None
        think = bool((req.get("chat_template_kwargs") or {}).get("enable_thinking", req.get("enable_thinking", False)))
        budget = req.get("thinking_budget_tokens") or req.get("thinking_budget") or cfg.get("thinking_budget") or 0
        budget = int(budget) if think else 0
        g = lambda k, d: d if req.get(k) is None else req.get(k)  # noqa: E731
        temperature = float(g("temperature", cfg["temperature"]))
        top_p = float(g("top_p", cfg["top_p"]))
        top_k = int(g("top_k", cfg["top_k"]))
        min_p = float(g("min_p", cfg.get("min_p", 0)))
        rp = float(g("repeat_penalty", cfg.get("repeat_penalty", 1.0)))
        rlast = int(g("repeat_last_n", cfg.get("repeat_last_n", 64)))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or cfg["max_tokens"])
        repeat_stop = bool(cfg.get("repeat_stop"))
        stops = req.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        seed = req.get("seed")
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 2 ** 31)
        kw = req.get("chat_template_kwargs") or {}
        effort = req.get("reasoning_effort") or kw.get("reasoning_effort") or cfg.get("reasoning_effort") or None
        body, gp = chat.render(messages, tools, think, variant=self.variant, reasoning_effort=effort,
                               preserve_thinking=kw.get("preserve_thinking"))
        Tb = self.tok.encode(body)
        T = Tb + self.tok.encode(gp)
        ctx = self.eng.ctx
        if len(T) + 1 >= ctx:
            raise ContextOverflow(len(T), ctx)
        if req.get("_kindi_check"):                       # check_prompt(): the prompt is valid, nothing to run
            return T
        fast = self.head_mode() == "fast"
        if fast and self.eng.hsl is None:
            fast = False
        self.eng.shortlist = cfg["fast_clusters"] if fast else 0
        max_new = max(1, min(max_new, ctx - len(T) - 1))
        t0 = time.time()
        self.busy.update(active=True, phase="matching cache", prompt_tokens=len(T), cached=0, read=0, generated=0,
                         started=t0)
        self.log("request: %d prompt tokens (max %d new)" % (len(T), max_new))
        try:
            im_end = 248046                                   # <|im_end|> closes the first message
            opening_end = T.index(im_end) + 1 if im_end in T else len(T)
            session = req.get("_kindi_session") or req.get("kindi_session") or req.get("user")
            nxt, cached = self.runner.prepare(T, len(Tb), opening_end, cancel, session=session)
        except Cancelled as e:
            self.busy["active"] = False
            self.log("stopped: %s" % e)
            return
        except BaseException:
            self.busy["active"] = False
            raise
        t1 = time.time()
        self.log("prompt: %d reused, %d read in %.1fs" % (cached, len(T) - cached, t1 - t0))
        self.busy["phase"] = "generating"
        pen = rp != 1.0 and rlast != 0
        if hasattr(self.eng, "spec_begin"):
            if pen:                                            # the speculative paths do not apply it: no drafts
                self.eng.spec_begin(False)
            else:                                              # speculative decoding (KINDI_SPEC), min-p included
                self.eng.spec_begin(temperature <= 0, temperature, top_p, top_k, rng, min_p)
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        text, gen, finish = "", [], "length"
        # while thinking, pieces go out as ("reasoning", ...) - the last 8 characters are held back in case they are
        # the start of "</think>"; after it, as ("text", ...)
        st = {"think": think, "sent": 0, "forced": False, "lead": think}

        def route(piece):
            if not st["think"]:
                if st["lead"]:                                # the newlines that follow </think>
                    piece = piece.lstrip("\n")
                    st["lead"] = not piece
                return [("text", piece)] if piece else []
            out = []
            if "</think>" in text:
                end = text.index("</think>")
                if end > st["sent"]:
                    out.append(("reasoning", text[st["sent"]:end]))
                after = text[end + len("</think>"):].lstrip("\n")
                st["think"] = False
                if after:
                    st["lead"] = False
                    out.append(("text", after))
            elif len(text) - 8 > st["sent"]:
                out.append(("reasoning", text[st["sent"]:len(text) - 8]))
                st["sent"] = len(text) - 8
            return out

        # repeat_penalty: over the last repeat_last_n tokens of the prompt and the reply (-1: all of them)
        recent = lambda: (rp, set((T + gen)[-rlast:] if rlast > 0 else T + gen)) if pen else None  # noqa: E731
        tok = self.sample(nxt, temperature, top_p, top_k, rng, min_p, recent())
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self.log("stopped: the client went away after %d tokens" % len(gen))
                    return
                if tok in EOS:
                    finish = "stop"
                    break
                gen.append(tok)
                self.busy["generated"] = len(gen)
                piece = dec.decode(self.tok.decode_bytes(tok))
                text += piece
                hit = None if st["think"] else next((s for s in stops if s and s in text), None)
                if hit:
                    text = text[:text.index(hit)]
                    finish = "stop"
                    break
                for item in route(piece):
                    yield item
                if len(gen) >= max_new:
                    break
                loop = repeat_stop and len(gen) % 16 == 0 and _looping(gen)
                if loop and not st["think"]:
                    self.log("stopped a repeating loop after %d tokens" % len(gen))
                    finish = "length"
                    break
                nxt = self.runner.step(tok)
                if loop and st["think"] and not st["forced"]:
                    budget = len(gen)                         # a loop in the thinking: close it now, then answer
                if st["think"] and budget and len(gen) >= budget and not st["forced"]:
                    # thinking budget spent: close the thinking for the model, then let it answer
                    st["forced"] = True
                    self.log("thinking budget of %d tokens reached; closing the thinking" % budget)
                    for ft in self.tok.encode(THINK_CLOSE):
                        gen.append(ft)
                        piece = dec.decode(self.tok.decode_bytes(ft))
                        text += piece
                        for item in route(piece):
                            yield item
                        nxt = self.runner.step(ft)
                tok = self._pick(nxt, temperature, top_p, top_k, rng, min_p, recent())
        finally:
            sp = self.eng.spec_end() if hasattr(self.eng, "spec_end") else None
            self.busy["active"] = False
            if hasattr(self.runner, "idle"):
                self.runner.idle()                            # (KINDI_CKPT_VRAM: checkpoint host copies now)
        if sp:
            self.last_spec = dict(sp)
        if sp and sp["passes"]:
            self.log("speculative: %d tokens in %d passes, %d of %d drafted tokens accepted" % (
                len(gen), sp["passes"], sp["accepted"], sp["drafted"]))
        if st["think"] and len(text) > st["sent"]:
            yield "reasoning", text[st["sent"]:]
        t2 = time.time()
        self.log("done: %d tokens in %.1fs (%s)" % (len(gen), t2 - t1, finish))
        for k, v in (("requests", 1), ("prompt_tokens", len(T)), ("cached_tokens", cached),
                     ("completion_tokens", len(gen))):
            self.stats[k] += v
        reasoning = ""
        if "</think>" in text:
            reasoning, text = text.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "").strip()
            text = text.lstrip("\n")
        content, calls = chat.parse_tool_calls(text, tools) if tools else (text.strip(), [])
        if calls:
            finish = "tool_calls"
        yield "done", {"content": content, "tool_calls": calls, "reasoning": reasoning, "finish": finish,
                       "usage": {"prompt_tokens": len(T), "completion_tokens": len(gen),
                                 "total_tokens": len(T) + len(gen), "prompt_tokens_details": {"cached_tokens": cached}},
                       "timing": {"prompt_s": round(t1 - t0, 2), "prompt_new_tokens": len(T) - cached,
                                  "gen_s": round(t2 - t1, 2),
                                  "gen_tok_s": round(len(gen) / (t2 - t1), 2) if t2 > t1 else 0.0}}


# Every engine call runs on this one thread. CuPy keeps a cuBLAS/hipBLAS handle per thread: on the default executor,
# each new worker thread made a new handle, and on a full card that fails (CUBLAS_STATUS_ALLOC_FAILED) once the K/V and
# the parked conversations hold the memory. One thread makes one handle, at its first use, as one request did before.
# (A handle made before the engine would load the cuBLAS code ahead of the context fit and cost small cards context.)
GPU_THREAD = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kindi-gpu")

# priority classes (0.6.0): every request names a class with "kindi_priority" (or the header X-Kindi-Priority).
# chat (the default) goes before background (cron jobs, batch work, agents' side work). A background request that
# is reading its prompt stops between two chunks when a chat request arrives; the chat request runs, and the
# background request then goes on from the prompt cache (its tokens read so far stay in the cache, or parked).
PRIO_TURN, PRIO_CHAT, PRIO_BACKGROUND = -1, 0, 1
_PRIO_NAMES = {"chat": PRIO_CHAT, "interactive": PRIO_CHAT, "high": PRIO_CHAT,
               "background": PRIO_BACKGROUND, "cron": PRIO_BACKGROUND, "batch": PRIO_BACKGROUND, "low": PRIO_BACKGROUND}


def priority_of(req, headers=None):
    v = req.get("kindi_priority") or (headers.get("x-kindi-priority") if headers is not None else None) or "chat"
    return _PRIO_NAMES.get(str(v).strip().lower(), PRIO_CHAT)


class Scheduler(object):
    """The engine jobs in priority order on the one engine thread: each task on GPU_THREAD runs the best job waiting
    at that moment (the lowest priority number, then the oldest)."""

    def __init__(self):
        import heapq
        self.heapq, self.heap, self.lk, self.seq = heapq, [], threading.Lock(), 0

    def submit(self, fn, prio=PRIO_CHAT):
        f = concurrent.futures.Future()
        with self.lk:
            self.heapq.heappush(self.heap, (prio, self.seq, fn, f))
            self.seq += 1
        GPU_THREAD.submit(self._run_one)
        return f

    def _pop(self, max_prio=None):
        with self.lk:
            if not self.heap or (max_prio is not None and self.heap[0][0] > max_prio):
                return None
            return self.heapq.heappop(self.heap)

    def _run(self, job):
        _, _, fn, f = job
        if not f.set_running_or_notify_cancel():
            return
        try:
            f.set_result(fn())
        except BaseException as e:  # noqa: BLE001 - the waiting request reports it
            f.set_exception(e)

    def _run_one(self):
        job = self._pop()
        if job is not None:
            self._run(job)

    def waiting(self, max_prio=PRIO_CHAT):
        """jobs waiting with this priority or a better one"""
        with self.lk:
            return sum(1 for j in self.heap if j[0] <= max_prio)

    def counts(self):
        with self.lk:
            return {"chat": sum(1 for j in self.heap if j[0] == PRIO_CHAT),
                    "background": sum(1 for j in self.heap if j[0] == PRIO_BACKGROUND)}

    def run_waiting(self, max_prio=PRIO_CHAT):
        """run the waiting jobs of this priority or better now, on this thread (a background request yields)"""
        n = 0
        while True:
            job = self._pop(max_prio)
            if job is None:
                return n
            self._run(job)
            n += 1


SCHED = Scheduler()


class _Yield(object):
    """the cancel event of a background request: also set while it reads its prompt and a chat request waits"""

    def __init__(self, cancel, K):
        self.cancel, self.K, self.fired = cancel, K, False

    def is_set(self):
        if self.cancel.is_set():
            return True
        if self.K.busy.get("phase") in ("matching cache", "reading prompt", "prompt") and SCHED.waiting(PRIO_CHAT):
            self.fired = True
            return True
        return False

    def set(self):
        self.cancel.set()


def run_generate(K, req, cancel, prio):
    """K.generate with the priority classes: a background request that yields to chat requests runs them here, then
    starts again (from the prompt cache). Yields what K.generate yields."""
    while True:
        c = _Yield(cancel, K) if prio >= PRIO_BACKGROUND else cancel
        done = False
        for item in K.generate(req, c):
            if item[0] == "done":
                done = True
            yield item
        if done or cancel.is_set() or not getattr(c, "fired", False):
            return
        K.yields = getattr(K, "yields", 0) + 1
        K.log("a background request yields to %d chat request(s) while it reads its prompt" % SCHED.waiting(PRIO_CHAT))
        SCHED.run_waiting(PRIO_CHAT)


def _after_engine_error(K, e):
    """An engine error other than a bad request. Drop the cache state (it may be half updated), and if the GPU
    context is broken (an illegal memory access is sticky in CUDA: every later call fails too), exit so that the
    container's restart policy brings the server back, in place of answering every request with an error."""
    r = getattr(K, "runner", None)
    if r is not None:
        try:
            r.eng.reset(); r.tokens = []; r.ckpt.clear()
        except Exception:
            pass
    if getattr(K, "cp", None) is None or os.environ.get("KINDI_DEVICE", "gpu") == "cpu":
        return                                   # the CPU engine has no GPU context to lose
    broken = "IllegalAddress" in str(e) or "illegal memory access" in str(e)
    try:
        K.cp.cuda.runtime.deviceSynchronize()
        K.cp.cuda.runtime.memGetInfo()
    except Exception:
        broken = True
    if broken:
        print("[kindi] the GPU context is broken (%s): exiting so that the server restarts" % e, flush=True)
        threading.Timer(1.0, os._exit, (70,)).start()


def _make_engine(cfg, log):
    return Kindi(cfg, log)


def create_app(cfg, log=print):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    K = GPU_THREAD.submit(_make_engine, cfg, log).result()
    model_id = cfg["model_name"]
    app = FastAPI(title="Kindi", version=__version__)

    def _err(code, msg, typ="invalid_request_error"):
        return JSONResponse({"error": {"message": msg, "type": typ, "code": typ}}, status_code=code)

    def _auth(request):
        key = cfg["api_key"]
        return not key or request.headers.get("authorization", "") == "Bearer " + key

    @app.get("/health")
    def health():
        free = total = 0
        if K.cp is not None and os.environ.get("KINDI_DEVICE", "gpu") != "cpu":
            free, total = K.cp.cuda.runtime.memGetInfo()
        b = K.busy
        busy = None
        if b["active"]:
            busy = {"phase": b["phase"], "elapsed_s": round(time.time() - b["started"]),
                    "prompt_tokens": b["prompt_tokens"], "reused": b["cached"],
                    "prompt_read": "%d/%d" % (b["read"], b["prompt_tokens"] - b["cached"]), "generated": b["generated"]}
        return {"ok": True, "kindi": __version__, "commit": COMMIT, "model": model_id, "ctx": K.eng.ctx, "head": K.head_active(),
                "uptime_s": round(time.time() - K.t0), "busy": busy, "stats": K.stats, "drafting": K.drafting(),
                "queue": dict(SCHED.counts(), yields=getattr(K, "yields", 0)),
                "gpu_free_mib": round(free / 1048576.0), "gpu_total_mib": round(total / 1048576.0)}

    def _overflow(e):
        return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error", "param": "messages",
                                       "code": "context_length_exceeded"}}, status_code=400)

    @app.get("/v1/models")
    def models(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        ctx = K.eng.ctx
        # the real window under every name that agent harnesses read (vLLM, llama.cpp, LM Studio, OpenRouter style)
        return {"object": "list", "data": [{"id": model_id, "object": "model", "created": int(K.t0),
                                            "owned_by": "kindi", "context_length": ctx, "max_model_len": ctx,
                                            "max_context_length": ctx, "meta": {"n_ctx": ctx}}]}

    async def _tokenize(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        try:
            req = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "the body is not JSON")
        ctx = K.eng.ctx
        try:
            if req.get("messages") is not None:   # a chat request: the whole prompt, as the chat endpoint builds it
                ids = await asyncio.get_event_loop().run_in_executor(None, K.check_prompt, dict(req, stream=False))
            else:
                text = req.get("content", req.get("prompt"))
                if not isinstance(text, str):
                    return _err(400, "send content (text) or messages")
                ids = K.encode_text(text, req.get("add_special_tokens", True) is not False)
        except ContextOverflow as e:              # counting still works: report the count and that it does not fit
            return {"count": e.n, "max_model_len": ctx, "fits": False, "tokens": []}
        except ValueError as e:
            return _err(400, str(e))
        out = {"count": len(ids), "max_model_len": ctx, "fits": len(ids) + 1 < ctx}
        if req.get("return_tokens", True) is not False:
            out["tokens"] = ids
        return out

    async def _detokenize(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        try:
            req = await request.json()
            ids = [int(i) for i in req.get("tokens") or []]
        except Exception:  # noqa: BLE001
            return _err(400, "send tokens: a list of token ids")
        text = K.decode_ids(ids)
        return {"prompt": text, "content": text}

    app.add_api_route("/tokenize", _tokenize, methods=["POST"])
    app.add_api_route("/v1/tokenize", _tokenize, methods=["POST"])
    app.add_api_route("/detokenize", _detokenize, methods=["POST"])
    app.add_api_route("/v1/detokenize", _detokenize, methods=["POST"])

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if not _auth(request):
            return _err(401, "invalid API key", "authentication_error")
        try:
            req = await request.json()
        except Exception:  # noqa: BLE001
            return _err(400, "the body is not JSON")
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        # every request is counted before it waits for the engine: a prompt that does not fit gets its HTTP 400 at
        # once (a queued request used to wait for the request ahead of it, then fail), and a dry run ends here
        dry = bool(req.get("dry_run") or req.get("kindi_dry_run"))
        try:
            ids = await asyncio.get_event_loop().run_in_executor(None, K.check_prompt, req)
        except ContextOverflow as e:
            if dry:
                return {"object": "kindi.prompt_count", "model": model_id, "fits": False, "max_model_len": K.eng.ctx,
                        "usage": {"prompt_tokens": e.n, "completion_tokens": 0, "total_tokens": e.n}}
            return _overflow(e)
        except ValueError as e:
            return _err(400, str(e))
        except Exception:  # noqa: BLE001 - anything else: the request itself reports it
            ids = None
        if dry:
            n = len(ids or [])
            return {"object": "kindi.prompt_count", "model": model_id, "fits": n + 1 < K.eng.ctx,
                    "max_model_len": K.eng.ctx, "usage": {"prompt_tokens": n, "completion_tokens": 0,
                                                          "total_tokens": n}}

        def chunk(delta, finish=None, usage=None):
            d = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage is not None:
                d["usage"] = usage
            return "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"

        cancel = threading.Event()
        mine = {"started": False}             # this request holds the engine (else it waits behind another one)
        prio = priority_of(req, request.headers)
        if request.headers.get("x-kindi-session"):
            req["_kindi_session"] = request.headers["x-kindi-session"]      # the conversation this request belongs to

        def progress_note():
            if not mine["started"]:
                return {"phase": "queued"}
            return K.progress() or {"phase": "finishing"}

        if req.get("stream"):
            def sse():
                with K.lock:
                    mine["started"] = True
                    yield chunk({"role": "assistant", "content": ""})
                    held, in_call = "", False
                    try:
                        for kind, val in run_generate(K, req, cancel, prio):
                            if kind == "reasoning":
                                yield chunk({"reasoning_content": val})
                            elif kind == "text":
                                held += val
                                if in_call:
                                    continue                  # keep the call text until it parses (or not)
                                if req.get("tools") and "<tool_call>" in held:
                                    pre = held[:held.index("<tool_call>")]
                                    if pre:
                                        yield chunk({"content": pre})
                                    held, in_call = held[held.index("<tool_call>"):], True
                                    continue
                                # hold back a possible start of '<tool_call>' / '<think>' / '</think>'
                                cut = max([held.rfind("<")] + [0]) if "<" in held[-12:] else len(held)
                                out, held = held[:cut], held[cut:]
                                if out:
                                    yield chunk({"content": out})
                            else:
                                info = val
                                if info["tool_calls"] and held and not in_call:
                                    yield chunk({"content": held})        # text held back before the calls
                                if info["tool_calls"]:
                                    yield chunk({"tool_calls": [dict(tc, index=i)
                                                                for i, tc in enumerate(info["tool_calls"])]})
                                elif held:
                                    # also an unfinished call (cut off by max_tokens): send its raw text, do not drop it
                                    yield chunk({"content": held})
                                inc = (req.get("stream_options") or {}).get("include_usage")
                                yield chunk({}, info["finish"], info["usage"] if inc else None)
                    except OverflowError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error",
                                                               "code": "context_length_exceeded"}}) + "\n\n"
                    except ValueError as e:
                        yield "data: " + json.dumps({"error": {"message": str(e), "type": "invalid_request_error"}}) + "\n\n"
                    except Exception as e:
                        yield "data: " + json.dumps({"error": {"message": "kindi: %s" % e, "type": "server_error"}}) + "\n\n"
                        _after_engine_error(K, e)
                    yield "data: [DONE]\n\n"
                    if not cancel.is_set():
                        K.after_turn(locked=True)     # the client has its answer; the next request waits for this
            async def relay():
                # the generator runs in a worker thread; if the client disconnects, Starlette cancels this
                # coroutine, the finally sets `cancel`, and the engine stops at its next token
                loop = asyncio.get_event_loop()
                q = asyncio.Queue()
                END = object()

                def pump():
                    try:
                        for item in sse():
                            loop.call_soon_threadsafe(q.put_nowait, item)
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, END)
                fut = asyncio.wrap_future(SCHED.submit(pump, prio))
                try:
                    while True:
                        # an SSE comment every KEEPALIVE_S while nothing else is sent (waiting in the queue, reading
                        # a long prompt): it keeps the stream alive and says how far the read is. SSE clients
                        # ignore comment lines.
                        try:
                            item = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_S)
                        except asyncio.TimeoutError:
                            yield ": kindi progress " + json.dumps(progress_note()) + "\n\n"
                            continue
                        if item is END:
                            break
                        yield item
                finally:
                    cancel.set()
                    await asyncio.wait([fut], timeout=30)
            return StreamingResponse(relay(), media_type="text/event-stream")

        def run():
            with K.lock:
                mine["started"] = True
                info = None
                for kind, val in run_generate(K, req, cancel, prio):
                    if kind == "done":
                        info = val
                return info

        def answer(info):
            msg = {"role": "assistant",
                   "content": info["content"] if (info["content"] or not info["tool_calls"]) else None}
            if info["tool_calls"]:
                msg["tool_calls"] = info["tool_calls"]
            if info["reasoning"]:
                msg["reasoning_content"] = info["reasoning"]
            choice = {"index": 0, "message": msg, "finish_reason": info["finish"]}
            if info.get("logprobs") is not None:
                choice["logprobs"] = {"content": info["logprobs"]}
            return {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                    "choices": [choice], "usage": info["usage"], "timing": info["timing"]}

        def failed(e):
            """(status, body) of a request that raised"""
            if isinstance(e, OverflowError):
                return 400, {"error": {"message": str(e), "type": "invalid_request_error", "param": "messages",
                                       "code": "context_length_exceeded"}}
            if isinstance(e, ValueError):
                return 400, {"error": {"message": str(e), "type": "invalid_request_error",
                                       "code": "invalid_request_error"}}
            _after_engine_error(K, e)
            return 500, {"error": {"message": "kindi: %s" % e, "type": "server_error", "code": "server_error"}}

        loop = asyncio.get_event_loop()
        # kindi_keepalive (or the header X-Kindi-Keepalive: 1): a non-streaming client behind a proxy with an idle
        # timeout gets a space every KEEPALIVE_S until the JSON answer. JSON allows whitespace before the value, but
        # the status is then 200 before the answer is known: an error comes as an {"error": ...} body with status
        # 200. So it is off unless the client asks for it.
        if req.get("kindi_keepalive") or request.headers.get("x-kindi-keepalive", "") in ("1", "true"):
            async def body():
                fut = asyncio.wrap_future(SCHED.submit(run, prio))
                try:
                    while True:
                        done, _ = await asyncio.wait([fut], timeout=KEEPALIVE_S)
                        if done:
                            break
                        yield " "
                    try:
                        info = fut.result()
                        out = answer(info) if info is not None else {"error": {"message": "request cancelled",
                                                                               "type": "cancelled"}}
                    except Exception as e:  # noqa: BLE001
                        out = failed(e)[1]
                    yield json.dumps(out, ensure_ascii=False)
                    if not cancel.is_set():
                        SCHED.submit(K.after_turn, PRIO_TURN)
                finally:
                    cancel.set()
            return StreamingResponse(body(), media_type="application/json")
        try:
            fut = asyncio.wrap_future(SCHED.submit(run, prio))
            while not fut.done():
                done, _ = await asyncio.wait([fut], timeout=1.0)
                if not done and await request.is_disconnected():
                    cancel.set()
            info = fut.result()
            if info is None:
                return _err(499, "request cancelled", "cancelled")
        except Exception as e:  # noqa: BLE001
            code, out = failed(e)
            return JSONResponse(out, status_code=code)
        # the save of the turn's end state runs on the engine thread after this answer is sent, before the next request
        SCHED.submit(K.after_turn, PRIO_TURN)
        return answer(info)

    return app


def serve(cfg, log=print):
    import uvicorn
    app = create_app(cfg, log)
    log("serving on http://%s:%d/v1 (model %s)" % (cfg["host"], cfg["port"], cfg["model_name"]))
    uvicorn.run(app, host=cfg["host"], port=int(cfg["port"]), log_level="warning")
