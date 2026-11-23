"""Qwen3.8-Flash-Next (the qwen4exp card engine, kindi.qwen4exp_gpu) in Kindi's server: server.Kindi with backend
"kindi_qwen4exp".

The chat side is the Lightning backend's (kindi.lightning_gen): the model's own chat template from the GGUF (Qwen
style: <|im_start|> turns, the thinking block, tool calls in the XML form <tool_call><function=...>), the thinking
block as reasoning, the tool-call parser, the repeat guard and sampling. Only the engine differs, through Engine
below, and the defaults of sampling (Qwen's published values).

Prompt reuse: the engine's state after a prompt (GDN states, attention caches, the n-gram context) is kept; a request
whose prompt continues the last prompt reads only the new tokens. The decode runs on a copy of that state (the CUDA
graph decoder), so the reply's tokens are not in the kept state: the next turn renders the reply again in its prompt
anyway. A second copy is kept at the last assistant header of each prompt (a new turn of the same chat starts
there), as the Lightning backend does."""
import os
import time

import numpy as np

from .lightning_gen import LightningBackend, ThinkSorter  # noqa: F401  (ThinkSorter: the tests import it from here)
from . import lightning_gen as _lg

# Qwen's published sampling for the Qwen3.8 models (thinking / no thinking), used when neither the request nor the
# configuration sets a value
SAMPLING_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
SAMPLING_PLAIN = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def _clone(st):
    """a deep copy of a qwen4exp state dict (tensors cloned)"""
    out = {}
    for k, v in st.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.clone() if hasattr(vv, "clone") else vv) for kk, vv in v.items()}
        elif hasattr(v, "clone"):
            out[k] = v.clone()
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def band(z, temperature):
    """(token ids, logits as float64) of the tokens of logit row z (a card tensor) within 16 temperatures of the best
    (each other token has relative probability under e^-16 against the best), at most 256 of the best; None when
    more than 4096 qualify even at a narrow band (the caller then takes the best token). The candidates of plain
    sampling and of drafted sampling, so both sample the same distribution."""
    z = z.float().reshape(-1)
    w = 16.0 * max(temperature, 0.1)
    got = None
    while True:
        idx = (z >= z.max() - w).nonzero().reshape(-1)
        if idx.numel() <= 4096:
            got = idx.cpu().numpy().astype(np.int64), z[idx].double().cpu().numpy()
            break
        if w <= 0.5:
            break
        w /= 2
    if got is None:
        return None
    cand, sc = got
    if cand.size > 256:
        keep = np.argpartition(sc, -256)[-256:]
        cand, sc = cand[keep], sc[keep]
    return cand, sc


def row_dist(z, temperature, top_p, top_k, min_p=0.0, penalty=None, got=None):
    """the sampler's distribution for logit row z: (token ids, probabilities) after the repeat penalty, top-k, top-p,
    min-p and the temperature (kindi.sampling.distribution); one-hot on the best token at temperature 0 or when
    the band is too wide (as FlashNextBackend._pick). got: band(z, temperature) when the caller has it."""
    from .sampling import distribution, penalize
    got = band(z, temperature) if got is None else got
    if got is None:
        return np.array([int(z.reshape(-1).argmax())], dtype=np.int64), np.ones(1)
    cand, sc = got
    if penalty is not None and penalty[0] > 1.0:
        sc = penalize(cand, sc, penalty[0], penalty[1])
    if temperature <= 0:
        return np.array([int(cand[int(np.argmax(sc))])], dtype=np.int64), np.ones(1)
    return distribution(cand, sc, temperature, top_p, top_k, min_p)


class DraftSampler(object):
    """speculative sampling for the drafted stream (kindi.qwen4exp_mtp.GraphDrafter, sampler=): each row of a
    verify window gets the distribution plain sampling would use there (row_dist, with the repeat penalty over the
    history up to that row), and kindi.sampling.spec_accept keeps each draft (the head's argmax, so q is one-hot)
    with probability p(draft), else draws from p without it. The output is distributed as plain sampling; the
    request's random generator (its seed) draws every random number."""

    def __init__(self, temperature, top_p, top_k, min_p, rp, rlast, rng):
        self.t, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k or 0)
        self.min_p, self.rp, self.rlast, self.rng = float(min_p or 0.0), float(rp or 1.0), int(rlast), rng

    def _pen(self, hist):
        if self.rp <= 1.0:
            return None
        return self.rp, set(hist[-self.rlast:] if self.rlast > 0 else hist)

    def dist(self, z, hist):
        return row_dist(z, self.t, self.top_p, self.top_k, self.min_p, self._pen(hist))

    def first(self, z, ids):
        idx, p = self.dist(z, ids)
        return int(self.rng.choice(idx, p=p))

    def accept(self, lgT, drafts, ids, out):
        from .sampling import spec_accept
        drafts = list(drafts)
        if self.rp > 1.0:
            hist = list(ids) + list(out)

            def hd(i):
                return hist + drafts[:i]
        else:
            def hd(i):
                return ()
        return spec_accept(lambda i: self.dist(lgT[i], hd(i)), drafts, self.rng)


# the card memory of the parts beside the hot experts (GB; FLASHNEXT-STATUS: dense 3.64, head 0.53; the compact MTP head
# 0.71 + its draft vocabulary 0.11; the CUDA context, graphs and buffers 1.2) for KINDI_FN_DRAFT=auto's check
DENSE_GB, HEAD_GB, RESERVE_GB = 4.17, 0.82, 1.2


class Engine(object):
    """The interface the Lightning backend drives (reset, prefill, step, snapshot, restore, candidates, generate,
    pos, ctx), over kindi.qwen4exp_gpu.Model. The prompt goes through Model.logits on a kept state; the decode
    steps through the CUDA-graph decoder loaded from a copy of it."""

    def __init__(self, path, ctx, log=print):
        import torch
        from .qwen4exp_gpu import Model
        self.torch = torch
        devs = [int(x) for x in os.environ.get("KINDI_FN_CARDS", "0").split(",") if x.strip() != ""]
        from .cores import default_cpu_list
        # the CPU expert pool: one thread per physical core of every socket (KINDI_FN_CPUS overrides; 0.7 had 0-15)
        cpus = os.environ.get("KINDI_FN_CPUS", os.environ.get("KINDI_MOE_CPUS0")) or default_cpu_list()
        t0 = time.time()
        # KINDI_FN_EXPERT_GB: the VRAM for hot experts on each card ("6" or "6,3"); default: what is free
        eg = os.environ.get("KINDI_FN_EXPERT_GB")
        eg = [float(x) for x in eg.split(",")] if eg else None
        if eg is not None and len(eg) == 1:
            eg = eg * len(devs)
        # the context: the decoder's caches hold KINDI_QXG_LMAX positions and the expert plan keeps room for them,
        # so it is set before the model loads (the configured ctx, else the engine's default)
        # the MTP head when the server does not draft (KINDI_FN_DRAFT unset or 0): left out (KINDI_FN_HEAD_IDLE=0, the
        # default), since a loaded head costs the plain decode: every decode product goes to the tensor cores (slower
        # than the one-token dense kernels on the 3060) and its experts and draft vocabulary take 0.8-1.5 GB from the
        # hot experts (card hit 0.45-0.48 against 0.62 without it, B2). KINDI_FN_HEAD_IDLE=1 loads it anyway.
        # KINDI_FN_DRAFT=auto: draft only where it pays (the engine's rule: at least KINDI_MTP_MIN_HOT_GB, now 0, of hot
        # experts beside the head on the card); the head is loaded only when the card's free memory can hold the dense
        # layers, the head and that much of experts, so a card that would not draft does not pay for an idle head
        # default auto (0.7): B32-B38 on the 3060 (12 GB) showed drafting k = 2 paying with the head's small cache,
        # so auto needs only the dense layers, the head and the reserve (KINDI_MTP_MIN_HOT_GB, now 0, adds a floor)
        mode = os.environ.get("KINDI_FN_DRAFT", "auto")
        if mode == "auto":
            need = DENSE_GB + HEAD_GB + RESERVE_GB + float(os.environ.get("KINDI_MTP_MIN_HOT_GB", "0"))
            free = torch.cuda.mem_get_info(devs[0] if devs else 0)[0] / 1e9
            if len(devs) != 1 or free < need:
                mode = "0"
                log("Flash-Next: KINDI_FN_DRAFT=auto: %s; plain decode, no head" % (
                    "drafting runs on one card" if len(devs) != 1 else
                    "%.1f GB free on the card, drafting pays from %.1f GB" % (free, need)))
        self.draft_mode = mode
        if (mode not in ("1", "auto") and os.environ.get("KINDI_FN_HEAD_IDLE", "0") == "0"
                and "KINDI_MTP" not in os.environ):
            os.environ["KINDI_MTP"] = "0"
            log("Flash-Next: drafting is off and KINDI_FN_HEAD_IDLE=0: the MTP head is not loaded")
        if "KINDI_QXG_LMAX" not in os.environ and int(ctx) > 0:
            os.environ["KINDI_QXG_LMAX"] = str(int(ctx))
        self.gm = Model(path, devices=devs, cpu_experts=cpus, expert_gb=eg, log=log)
        lmax = int(os.environ.get("KINDI_QXG_LMAX", "8192"))
        self.ctx = lmax
        # drafting with the MTP head (KINDI_FN_DRAFT=1): the engine's drafted stream (greedy, the same tokens as
        # plain decoding on sm_80+; on sm_50-75 the tokens can differ at near-ties, KINDI_FN_DRAFT=0 for exact output),
        # on one card of sm_50 or later with the mailbox pass and in-process CPU experts (Amir 11-22: on by default there too)
        sm = torch.cuda.get_device_capability(devs[0] if devs else 0)
        self.can_draft = (bool(self.gm.mtp) and len(devs) == 1 and sm[0] >= 5
                          and os.environ.get("KINDI_QXG_PASS", "0") == "1" and not cpus.startswith("tcp:"))
        # auto (default): drafting whenever it can run (the head loaded, one card, the mailbox pass). B32-B38 (3060,
        # unseen chats): drafting k = 2 with the CUDA-core window kernels beats plain on chat (23.9 against 16-19) and
        # after a 4K prompt (19.0), with the head's 12 GB-card plan (gm.draft_ok, the 6 GB rule, no longer decides)
        self.draft_on = self.can_draft and mode in ("1", "auto")
        if mode == "auto":
            log("Flash-Next: KINDI_FN_DRAFT=auto: drafting %s" % ("on" if self.draft_on else "off (the engine's rule)"))
        self._gen, self.dstats = None, {}
        self.log = log
        self.st, self.pos, self.dec, self.live = None, 0, None, False
        self.row = None                 # the logits of the last position (prompt or step), on the card
        self._snap = None
        self.devices = devs
        log("Flash-Next engine ready on cards %s (%.0f s; context %d, MTP head %s)" % (
            devs, time.time() - t0, self.ctx, "loaded" if self.gm.mtp else "off"))

    # ---- the state
    def reset(self):
        self.st, self.pos, self.live = self.gm.new_state(), 0, False

    def snapshot(self):
        self._snap = _clone(self.st)

    def restore(self):
        self.st, self.live = _clone(self._snap), False
        self.pos = self.st["pos"]

    # ---- reading and writing
    def prefill(self, ids, start):
        """read ids at positions start ..; returns the greedy next token"""
        if self.st is None or start == 0:
            self.reset()
        if start != self.st["pos"]:
            raise RuntimeError("Flash-Next: prefill at %d, the kept state is at %d" % (start, self.st["pos"]))
        if self.st["pos"] + len(ids) >= self.ctx:
            raise OverflowError("the prompt has %d tokens; context is %d" % (self.st["pos"] + len(ids), self.ctx))
        lg = self.gm.logits(list(ids), last_n=1, state=self.st)
        self.row = lg[-1].float()
        self.pos, self.live = self.st["pos"], False
        return int(self.row.argmax())

    def start_drafted(self, ids, n, cancel=None, stop=(), sampler=None):
        """reply of ids with MTP drafts (kindi.qwen4exp_mtp.GraphDrafter.stream; greedy, or sampled with a
        DraftSampler): returns the first token; each step() then gives the next accepted token. The drafter reads the
        prompt itself (its head needs the prompt's cache), so the kept state is not this prompt's afterwards."""
        from .qwen4exp_mtp import GraphDrafter
        self.dstats = {}
        self._gen = GraphDrafter(self.gm, k=int(os.environ.get("KINDI_FN_DRAFT_K", "2"))).stream(
            list(ids), n, cancel=cancel, stop=tuple(stop), stats=self.dstats, sampler=sampler)
        self.st, self.live, self.pos = None, False, len(ids)
        first = int(next(self._gen))
        self._dhist = list(ids) + [first]             # the prompt and the tokens handed out
        return first

    def end_drafted(self):
        if self._gen is not None:
            self._gen.close()
        self._gen = None

    def step(self, tok=None):
        """write tok at the next position (the decoder, loaded from the kept state at the first step); returns the
        greedy next token"""
        if self._gen is not None and tok is not None and int(tok) != self._dhist[-1]:
            # the caller writes another token than the drafter's (the thinking budget closes the thinking): the
            # drafted stream ends; the plain path reads the reply so far with tok, and goes on from there
            h = self._dhist[:-1] + [int(tok)]
            self.end_drafted()
            self.reset()
            return self.prefill(h, 0)
        if self._gen is not None:                     # a drafted reply: the drafter's next accepted token
            self.pos += 1
            try:
                t = int(next(self._gen))
            except StopIteration:
                return int(tok)
            self._dhist.append(t)
            return t
        if not self.live:
            if self.dec is None:
                self.dec = self.gm.decoder(self.ctx)
            self.dec.load(self.st)
            self.live = True
        self.row = self.dec.step(int(tok)).float().reshape(-1)
        self.pos += 1
        return int(self.row.argmax())

    def generate(self, ids, n):
        self.reset()
        t = self.prefill(ids, 0)
        out = [t]
        while len(out) < n:
            out.append(self.step(out[-1]))
        return out

    def drafting_now(self):
        """a drafted reply is running: step() hands out the drafter's tokens (sampled already when sampling)"""
        return self._gen is not None

    def candidates(self, w):
        """(token ids, logits) of the tokens within w of the best, or None when more than 4096 qualify"""
        z = self.row
        idx = (z >= z.max() - w).nonzero().reshape(-1)
        if idx.numel() > 4096:
            return None
        return idx.cpu().numpy().astype(np.int64), z[idx].double().cpu().numpy()

    # ---- telemetry
    def stats(self):
        """the expert and drafting numbers for /health (what the engine reports; empty when it reports nothing)"""
        gm, out = self.gm, {}
        if hasattr(gm, "stats"):                      # the engine's cumulative numbers since load
            try:
                out = dict(gm.stats())
            except Exception:  # noqa: BLE001 - telemetry only
                out = {}
            out.setdefault("mtp_head", bool(gm.mtp))
            hx = out.get("hot_experts")
            if isinstance(hx, dict):                  # (an engine that reports it per card: the total, and per card)
                out["hot_experts_by_card"], out["hot_experts"] = hx, int(sum(hx.values()))
            out["cards"] = len(self.devices)
            if self.dstats:
                out["drafting_last"] = dict(self.dstats)
            return out
        out["hot_experts"] = int(sum(getattr(gm, "hot_count", {}).get(l, 0) for l in range(gm.L)))
        out["mtp_head"] = bool(gm.mtp)
        out["cards"] = len(self.devices)
        log = getattr(self.dec, "step_log", None) if self.dec is not None else None
        if log is not None:
            try:
                rows = log()
            except Exception:  # noqa: BLE001 - telemetry only
                rows = []
            tok = sum(r[1] for r in rows) if rows else 0
            picks = sum(r[4] for r in rows) if rows else 0
            on = sum(r[5] for r in rows) if rows else 0
            if tok:
                out["last"] = {"tokens": tok, "cpu_expert_ms_per_token": round(1000.0 * sum(r[3] for r in rows) / tok, 1),
                               "card_hit_rate": round(on / float(picks), 3) if picks else None}
                self._last_stats = out["last"]
        if "last" not in out and getattr(self, "_last_stats", None):
            out["last"] = self._last_stats
        return out


class FlashNextBackend(LightningBackend):
    """The Lightning backend's chat handling with the Flash-Next engine, tokenizer and sampling defaults"""

    name = "Flash-Next"

    def __init__(self, man, cfg, log=print):
        from .gguf import GGUF
        from .lightning_tokenizer import LightningTokenizer
        self.log = log
        self.path = man["model_path"]
        # <think>, </think> and <tool_call> are user-defined tokens (type 4) in this GGUF: whole tokens, as in the
        # model's training (as text pieces, a thinking-off reply starts with <|im_end|>: an empty answer)
        self.tok = LightningTokenizer(GGUF(self.path).kv, user_defined=True)
        self.tok.tokens = self.tok.vocab
        self.stop_ids = {self.tok.eos} | {i for i, t in enumerate(self.tok.types) if t == 3}
        self.im_start = self.tok.sid[_lg.IM_START]
        self.eng = Engine(self.path, ctx=int(cfg["ctx"]), log=log)
        self.cached, self.snap = [], []
        if os.environ.get("KINDI_FN_SELFCHECK", "1") != "0":
            self.selfcheck()

    def selfcheck(self):
        """before the first request: the same greedy tokens twice from the same prompt, not a degenerate run"""
        from .lightning_chat import render
        ids = self.tok.encode(render(self.path, [{"role": "user", "content": "Count from one to ten in words."}],
                                     enable_thinking=False))
        a = self.eng.generate(ids, 10)
        b = self.eng.generate(ids, 10)
        self.cached, self.snap = [], []
        ok = lambda t: len(set(t)) > 3 and t[0] not in self.stop_ids
        # fast mode (the default) drops CPU expert picks at a high router rank, and which experts are on the CPU
        # changes as the card cache fills after the first run: two runs can then differ at a near-tie. The exact
        # path (KINDI_FAST=0) must repeat token for token; fast mode must give two sound runs with the same start.
        exact = os.environ.get("KINDI_FAST", "1") != "1"
        if not (ok(a) and ok(b) and (a == b if exact else a[0] == b[0])):
            raise RuntimeError("Flash-Next self-check failed: %s / %s" % (a, b))
        if a != b:
            self.log("Flash-Next self-check: fast mode, the second run differs after token %d (KINDI_FAST=0 is exact)"
                     % next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
        self.log("Flash-Next self-check: %s" % repr(self.tok.decode(a))[:60])

    def _pick(self, greedy, temperature, top_p, top_k, rng, min_p=0.0, penalty=None):
        if temperature <= 0 and (penalty is None or penalty[0] <= 1.0 or greedy not in penalty[1]):
            return greedy
        if self.eng.drafting_now():                   # drafted sampling: the drafter drew this token (DraftSampler)
            return greedy
        got = band(self.eng.row, temperature)
        if got is None:
            return greedy
        idx, p = row_dist(self.eng.row, temperature, top_p, top_k, min_p, penalty, got=got)
        if temperature <= 0:
            return int(idx[0])
        return int(rng.choice(idx, p=p))

    def _read(self, ids):
        """the kept state holds the last prompt (not its reply): reuse it when ids continues that prompt, or the copy
        at its last assistant header; else read from the start"""
        eng = self.eng
        prev = getattr(self, "_prompt", [])
        if prev and _lg._lcp(prev, ids) == len(prev) and len(prev) < len(ids) and eng.st is not None \
                and eng.st["pos"] == len(prev):
            start = len(prev)
        elif self.snap and _lg._lcp(self.snap, ids) == len(self.snap) and len(self.snap) < len(ids):
            eng.restore()
            start = len(self.snap)
        else:
            eng.reset()
            start = 0
        reused = start
        mark = max((i for i, t in enumerate(ids) if t == self.im_start), default=-1)
        if start < mark:
            eng.prefill(ids[start:mark], start)
            eng.snapshot()
            self.snap = list(ids[:mark])
            start = mark
        greedy = eng.prefill(ids[start:], start)
        self._prompt = list(ids)
        self.cached = list(ids)
        return greedy, reused

    def _read_drafted(self, ids):
        """a request with drafting on: the drafter reads the whole prompt (no reuse) and gives the reply (greedy; or,
        for a sampled request, by speculative sampling with the request's sampling and random generator)"""
        d = self._draft
        sampler = None
        if d["sampled"]:
            s = self._samp
            sampler = DraftSampler(s["temperature"], s["top_p"], s["top_k"], s["min_p"], s["rp"], s["rlast"], self._rng)
        greedy = self.eng.start_drafted(ids, d["n"], cancel=d["cancel"], stop=sorted(self.stop_ids), sampler=sampler)
        self._prompt, self.cached, self.snap = [], [], []
        return greedy, 0

    def generate(self, K, req, cancel=None):
        # Qwen's sampling defaults for the request's thinking mode (MODEL_SAMPLING of the Lightning backend)
        kw = req.get("chat_template_kwargs") or {}
        thinking = kw.get("enable_thinking", req.get("enable_thinking", True)) is not False
        saved = dict(_lg.MODEL_SAMPLING)
        _lg.MODEL_SAMPLING.update(SAMPLING_THINK if thinking else SAMPLING_PLAIN)
        # drafting, when the engine can draft: a greedy request (temperature 0 given, no repeat penalty) takes the
        # drafter's greedy stream, as in 0.7.0; any other request (0.7.1) drafts with speculative sampling
        # (DraftSampler: the output is distributed as plain sampling). KINDI_FN_DRAFT_SAMPLED=0: sampled requests
        # decode plainly
        greedy_t = req.get("temperature") is not None and float(req.get("temperature")) <= 0
        greedy_req = greedy_t and float(req.get("repeat_penalty") or 1.0) <= 1.0
        sampled_ok = not greedy_t and os.environ.get("KINDI_FN_DRAFT_SAMPLED", "1") != "0"
        self._draft = None
        if self.eng.draft_on and (greedy_req or sampled_ok) and not req.get("_kindi_check"):
            n = int(req.get("max_tokens") or req.get("max_completion_tokens") or K.cfg["max_tokens"])
            self._draft = {"n": n, "cancel": (lambda: cancel.is_set()) if cancel is not None else None,
                           "sampled": not greedy_req}
        read = self._read
        if self._draft is not None:
            self._read = self._read_drafted
        try:
            for item in LightningBackend.generate(self, K, req, cancel):
                if item[0] == "done":
                    item[1]["timing"]["engine"] = "flash-next"
                yield item
        finally:
            self._read = read
            self.eng.end_drafted()
            _lg.MODEL_SAMPLING.clear()
            _lg.MODEL_SAMPLING.update(saved)
            self.cached = list(getattr(self, "_prompt", []))
            if self._draft is not None and self.eng.dstats:
                K.last_spec = {"drafted": self.eng.dstats.get("drafted", 0), "accepted": self.eng.dstats.get("accepted", 0),
                               "passes": self.eng.dstats.get("passes", 0), "sampled": bool(self._draft["sampled"])}

    # ---- /health
    def drafting(self):
        """the MTP head is loaded by default; greedy requests draft with it when KINDI_FN_DRAFT=1 and the card can
        (sm_50+, one card, KINDI_QXG_PASS=1)"""
        on = bool(self.eng.draft_on)
        return {"enabled": on, "method": "mtp" if self.eng.gm.mtp else None,
                "drafter": "MTP head" if self.eng.gm.mtp else None}

    def health_extra(self):
        """the card fields (the engine is PyTorch, not CuPy) and the expert numbers"""
        torch = self.eng.torch
        out = {"experts": self.eng.stats(), "device": "gpu"}
        try:
            dv = self.eng.devices[0] if self.eng.devices else 0
            free, total = torch.cuda.mem_get_info(dv)
            out["gpu_free_mib"], out["gpu_total_mib"] = round(free / 1048576.0), round(total / 1048576.0)
            p = torch.cuda.get_device_properties(dv)
            u = getattr(p, "uuid", None)
            if u is not None:
                out["gpu_uuid"] = "GPU-" + str(u)
            bus = getattr(p, "pci_bus_id", None)
            if bus is not None:
                out["gpu_pci"] = "%08x:%02x:%02x.0" % (getattr(p, "pci_domain_id", 0), bus, getattr(p, "pci_device_id", 0))
        except Exception:  # noqa: BLE001 - telemetry only
            pass
        return out
