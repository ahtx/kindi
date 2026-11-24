# SPDX-License-Identifier: AGPL-3.0-only
"""Static CPU+GPU layer tandem for the dense Bonsai packs.

Two placements are supported:

* ``cpu-first``: the CPU owns layers ``0..n-1``, the GPU owns ``n..NL-1`` and the output head;
* ``gpu-first``: the GPU owns ``0..NL-n-1`` and the embedding, the CPU owns the last ``n`` layers
  and the output head.

The hidden state is handed over once per token, between two complete residual streams.  Each
backend keeps its own GDN state and K/V cache, so no per-layer synchronisation is needed inside a
token.  The CPU engine is the existing AVX2/FMA/F16C backend; the GPU engine loads only the
layers in its range (``Engine(layers=...)``), so a split really reduces VRAM rather than loading
the whole model and ignoring part of it.

This is deliberately not a MoE engine.  A dense model needs every weight on every token, so an
expert-cache policy has nothing to cache.  The MoE approach is a separate design in
``docs/TANDEM-RUNTIME.md`` and is not implemented here.
"""
import json
import os

import numpy as np

from .cpu import Engine as CPUEngine
from .engine import Engine as GPUEngine

MODES = ("cpu-first", "gpu-first")


def _normalise_mode(mode):
    return str(mode or "off").strip().lower().replace("_", "-")


def _parse_layers(text):
    """-> an int count, a tuple of layer numbers, or a ``gdn:N`` spec (returned as a string)."""
    text = str(text).strip()
    try:
        return int(text)
    except ValueError:
        pass
    low = text.lower()
    if low.startswith("gdn:") or low == "auto" or low.startswith("auto:"):
        return low
    parts = [p.strip() for p in text.replace(";", ",").split(",") for p in p.split() if p.strip()]
    flat = []
    for part in parts:
        # a-b or a:b both mean the inclusive contiguous block
        if "-" in part[1:] or ":" in part:
            sep = "-" if "-" in part[1:] else ":"
            lo, hi = part.split(sep, 1)
            flat.extend(range(int(lo), int(hi) + 1))
        else:
            flat.append(int(part))
    return tuple(flat)


SCRATCH_MIB = 192          # measured plain-engine scratch at 24K (113 MiB on the 9B, 157 on the 27B) + slack


def pack_bytes(pack, man=None):
    """-> {'layers': {il: bytes}, 'embed': bytes, 'head': bytes, 'f32': bytes} for the device footprint.

    A ternary tensor costs ``rows * ne0 / 4`` bytes of bcodes plus one fp16 scale per 128-column group;
    an f32 tensor costs 4 bytes per value. The plain engine uploads exactly the selected tensors and the
    whole f32 file, so these sizes are the engine's device bytes, not the pack's.
    """
    man = man if man is not None else json.load(open(os.path.join(pack, "kindi_pack.json")))
    bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
    idx = json.load(open(os.path.join(pack, "index.json")))["tensors"]

    def nbytes(name):
        rec = bi.get(name)
        if rec is not None:
            return rec["rows"] * rec["ne0"] // 4 + rec["rows"] * (rec["ne0"] // 128) * 2
        return int(idx.get(name, {}).get("nfloats", 0)) * 4

    layers = {}
    for name in bi:
        if name.startswith("blk."):
            il = int(name.split(".")[1])
            layers[il] = layers.get(il, 0) + nbytes(name)
    for name in idx:
        if name.startswith("blk.") and name not in bi:
            il = int(name.split(".")[1])
            layers[il] = layers.get(il, 0) + nbytes(name)
    head_name = man.get("head", "token_embd.weight")
    return {"layers": layers, "embed": nbytes("token_embd.weight"), "head": nbytes(head_name),
            "f32": int(man.get("f32_bytes", 0))}


def kv_bytes(dims, ctx):
    """fp16 K and V bytes for one attention layer at this context."""
    return int(ctx) * int(dims["kv_heads"]) * int(dims["head_dim"]) * 2 * 2


def gpu_estimate(dims, sizes, ctx, cpu_layers=()):
    """Device bytes the plain engine needs when the GPU owns every layer except ``cpu_layers``.

    This is the engine's real allocation: the whole f32 file (it is uploaded as one file), the output
    head, the embedding unless the CPU owns layer 0, each remaining layer's tensors and the fp16 K/V of
    each remaining attention layer, plus the measured scratch allowance.
    """
    NL = int(dims["layers"])
    interval = max(1, int(dims["full_attention_interval"]))
    cpu = frozenset(int(x) for x in cpu_layers)
    gpu = [il for il in range(NL) if il not in cpu]
    total = sizes["f32"] + SCRATCH_MIB * 2 ** 20
    if gpu:
        total += sizes["head"]
    if 0 not in cpu:                                     # the CPU embeds when it owns layer 0
        total += sizes["embed"]
    for il in gpu:
        total += sizes["layers"].get(il, 0)
        if (il + 1) % interval == 0:
            total += kv_bytes(dims, ctx)
    return total


def plan_cpu_layers(dims, sizes, ctx, free_bytes, reserve_mib=512, target_mib=None, log=None):
    """The smallest CPU layer set that makes the GPU side fit, or ``()`` when no split is needed.

    Gated-delta layers go first, ascending: they cost no device K/V and little per-token CPU time, so
    they free VRAM at the lowest rate price (see ``gdn_first``). Attention layers are the last resort,
    because a CPU attention layer's cost grows with the context. The estimate is the plain engine's
    actual allocation (selected weights + the f32 file + the head + fp16 K/V of the attention layers the
    GPU keeps) plus a measured scratch allowance; on the 27B at 24K it predicts 7,778 MiB against a
    measured 7,783 MiB plan-engine peak.
    """
    log = log or (lambda *a: None)
    NL = int(dims["layers"])
    interval = max(1, int(dims["full_attention_interval"]))
    budget = int(free_bytes) - int(reserve_mib) * 2 ** 20
    free_target = None if target_mib is None else int(target_mib) * 2 ** 20
    full = gpu_estimate(dims, sizes, ctx, ())

    order = [il for il in range(NL) if (il + 1) % interval] + [il for il in range(NL)
                                                              if (il + 1) % interval == 0]
    cpu = []
    while True:
        est = gpu_estimate(dims, sizes, ctx, cpu)
        over = est > budget
        short = free_target is not None and (full - est) < free_target
        if not over and not short:
            break
        if len(cpu) + 1 >= NL:
            raise SystemExit("kindi: tandem auto cannot fit this model at %d tokens on the free GPU memory "
                             "(needs about %d MiB more); reduce --ctx"
                             % (int(ctx), max(est - budget, 0) // 2 ** 20))
        cpu.append(order[len(cpu)])
    if not cpu:
        return ()
    cpu = tuple(sorted(cpu))
    est = gpu_estimate(dims, sizes, ctx, cpu)
    log("tandem auto: %d CPU layer%s (%s) takes the GPU side to ~%d MiB (%d MiB freed; %d MiB budget)"
        % (len(cpu), "" if len(cpu) == 1 else "s", ",".join(str(x) for x in cpu),
           est // 2 ** 20, (full - est) // 2 ** 20, budget // 2 ** 20))
    return cpu


def gdn_first(dims, count):
    """The first ``count`` layers that are not full-attention, ascending.

    A full-attention layer on the CPU is the worst thing a split can do to decode rate: its
    cost grows with the context and its K/V would otherwise have lived in VRAM, which is the
    memory a split is trying to save. A gated-delta (GDN) layer is the opposite: its per-token
    CPU cost is context-independent and its device state is tiny, so it is the layer to move.
    Attention layers stay on the GPU; that keeps the split's rate flat with position.
    """
    layers = int(dims.get("layers", 24))
    interval = max(1, int(dims.get("full_attention_interval", 4)))
    gdn = [il for il in range(layers) if (il + 1) % interval]
    count = int(count)
    if not 1 <= count < len(gdn):
        raise SystemExit("kindi: gdn:%d needs between 1 and %d CPU layers (the model has %d GDN layers)"
                         % (count, len(gdn) - 1, len(gdn)))
    return tuple(gdn[:count])


class _TandemCacheView(object):
    """The two owners' state in the shape ``cache.Runner`` expects.

    A checkpoint must contain the CPU and GPU recurrent state as well as the K/V cache of every
    attention layer, whichever owner runs it.  The lists below are the backends' own arrays (NumPy
    on the CPU, CuPy on the GPU), so a restore writes them in place; the cache never needs to know
    which side owns a layer.
    """

    def __init__(self, eng):
        self.tandem = eng
        self.ctx = int(eng.ctx)
        interval = int(eng.cpu.dims["full_attention_interval"])
        cpu_layers = frozenset(eng.cpu_range)
        gpu_layers = frozenset(eng.gpu_range)
        self.M, self.conv, self.Kc, self.Vc = [], [], [], []
        cpu_m = cpu_att = 0
        for il in range(eng.NL):
            if (il + 1) % interval == 0:
                if il in cpu_layers:
                    k, v = eng.cpu.Kc[cpu_att], eng.cpu.Vc[cpu_att]
                elif il in gpu_layers:
                    ki = eng.gpu._kvi[il]
                    k, v = eng.gpu.Kc[ki], eng.gpu.Vc[ki]
                else:
                    raise RuntimeError("kindi: layer %d has no tandem K/V owner" % il)
                if k is None or v is None:
                    raise RuntimeError("kindi: layer %d has no tandem K/V state" % il)
                self.Kc.append(k)
                self.Vc.append(v)
                if il in cpu_layers:
                    cpu_att += 1
            else:
                if il in cpu_layers:
                    m, c = eng.cpu.M[cpu_m], eng.cpu.conv[cpu_m]
                elif il in gpu_layers:
                    m, c = eng.gpu.M[il], eng.gpu.conv[il]
                else:
                    raise RuntimeError("kindi: layer %d has no tandem recurrent owner" % il)
                if m is None or c is None:
                    raise RuntimeError("kindi: layer %d has no tandem recurrent state" % il)
                self.M.append(m)
                self.conv.append(c)
                if il in cpu_layers:
                    cpu_m += 1
        # cache.engine_fingerprint() reads these; a tandem split has no chunk path or VRAM slot.
        self.g5 = bool(getattr(eng.gpu, "g5", False))
        self.g5l = bool(getattr(eng.gpu, "g5l", False))
        self.kv = getattr(eng.gpu, "kv", None) or "f16"
        self.chunk = {}
        self.chunk_ok = False
        self.cards = None
        self.ckpt_slot = None
        self.batch_max = 1

    def cache_fingerprint(self):
        return self.tandem.cache_fingerprint()

    def reset(self):
        self.tandem.reset()

    def feed(self, tok, pos):
        self.tandem.feed(tok, pos)

    def step(self, tok, pos):
        return self.tandem.step(tok, pos)


class TandemEngine(object):
    """A dense layer split over the CPU engine and one GPU engine."""

    def __init__(self, pack, ctx=4096, cpu_layers=None, mode="cpu-first", log=print):
        self.log = log or (lambda *a: None)
        manifest = os.path.join(pack, "kindi_pack.json")
        man = json.load(open(manifest))
        if man.get("backend") in ("kindi_gemma4", "kindi_nanbeige"):
            raise SystemExit("kindi: tandem currently supports the dense Bonsai packs only")
        dims = dict(man.get("dimensions") or {})
        self.NL = int(dims.get("layers", 24))
        if self.NL < 2:
            raise SystemExit("kindi: tandem needs at least two layers")
        self.mode = _normalise_mode(mode)
        if self.mode not in MODES:
            raise SystemExit("kindi: tandem mode must be cpu-first or gpu-first, not %r" % mode)
        if cpu_layers is None:
            cpu_layers = max(1, self.NL // 4)
        auto_target = None
        if isinstance(cpu_layers, str):
            cpu_layers = _parse_layers(cpu_layers)
            if isinstance(cpu_layers, str):
                kind, _, arg = cpu_layers.partition(":")
                if kind == "gdn":
                    cpu_layers = gdn_first(dims, arg)
                elif kind == "auto":
                    cpu_layers = "auto"
                    auto_target = int(arg) if arg.strip() else None
                else:
                    raise SystemExit("kindi: --tandem-cpu-layers %r is not a count, a layer set or gdn:N/auto"
                                     % cpu_layers)
        if cpu_layers == "auto":
            from .engine import cp as _cp
            if _cp is None:
                raise SystemExit("kindi: tandem auto needs a CUDA/ROCm GPU to read the free memory")
            free = int(_cp.cuda.runtime.memGetInfo()[0])
            sizes = pack_bytes(pack, man)
            need = gpu_estimate(dims, sizes, ctx) // 2 ** 20
            if auto_target is None and need <= (free // 2 ** 20) - 512:
                raise SystemExit("kindi: this model already fits at %d tokens (~%d MiB needed, %d MiB free); "
                                 "a split would only make it slower - run without --tandem, or ask for a "
                                 "target with --tandem-cpu-layers auto:<MiB>"
                                 % (int(ctx), need, free // 2 ** 20))
            cpu_layers = plan_cpu_layers(dims, sizes, ctx, free,
                                         reserve_mib=int(os.environ.get("KINDI_TANDEM_RESERVE_MB", "512")),
                                         target_mib=auto_target, log=self.log)
        cpu_first = self.mode == "cpu-first"
        if isinstance(cpu_layers, (list, tuple)):
            explicit = tuple(int(x) for x in cpu_layers)
            if (not explicit or len(set(explicit)) != len(explicit)
                    or explicit != tuple(sorted(explicit))
                    or explicit[0] < 0 or explicit[-1] >= self.NL
                    or len(explicit) >= self.NL):
                raise SystemExit("kindi: tandem cpu layers %r must be a non-empty sorted subset of "
                                 "0..%d with at least one GPU layer" % (cpu_layers, self.NL - 1))
            self.cpu_layers = list(explicit)
            self.cpu_range = explicit
            gpu = tuple(il for il in range(self.NL) if il not in frozenset(explicit))
            self.gpu_range = gpu
        else:
            cpu_layers = int(cpu_layers)
            if not 1 <= cpu_layers < self.NL:
                raise SystemExit("kindi: tandem cpu layers must be between 1 and %d (got %d)"
                                 % (self.NL - 1, cpu_layers))
            self.cpu_layers = cpu_layers
            if cpu_first:
                self.cpu_range = tuple(range(0, cpu_layers))
                self.gpu_range = tuple(range(cpu_layers, self.NL))
            else:
                self.gpu_range = tuple(range(0, self.NL - cpu_layers))
                self.cpu_range = tuple(range(self.NL - cpu_layers, self.NL))
        self.cpu_first = cpu_first
        self._gpu_head = self.NL - 1 in self.gpu_range
        self.ctx = int(ctx)

        # Load only the CPU-owned slice: converting a whole 27B pack on the CPU
        # cost ~365 s even when the CPU owned 16 of 64 layers.
        self.cpu = CPUEngine(pack, ctx=ctx, log=log, layers=self.cpu_range,
                             embed=0 in self.cpu_range, head=self.NL - 1 in self.cpu_range)
        self.gpu = GPUEngine(pack, ctx=ctx, log=log, layers=self.gpu_range,
                             embed=0 in self.gpu_range, head=self._gpu_head)
        self.ctx = min(int(self.cpu.ctx), int(self.gpu.ctx))
        self.D = int(self.cpu.D)
        if int(self.gpu.D) != self.D:
            raise SystemExit("kindi: the CPU and GPU engines disagree on the hidden width")
        self.vocab = self.cpu.vocab
        self.hsl = None                    # the fast head is not used by a tandem split
        self.shortlist = 0
        if self._gpu_head:
            # Sampler uses the same GPU-kernel path as the normal engine: the logits and aout live on the card.
            self.k = self.gpu.k
            self.aout = self.gpu.aout
        self._sync_logits()
        def _rng(rng):
            full = rng == tuple(range(rng[0], rng[-1] + 1))
            return "%d..%d" % (rng[0], rng[-1]) if full else ",".join(str(x) for x in rng)
        self.log("tandem %s: CPU layers %s, GPU layers %s, %d-token context, %s head"
                 % (self.mode, _rng(self.cpu_range), _rng(self.gpu_range),
                    self.ctx, "GPU" if self._gpu_head else "CPU"))

    def _sync_logits(self):
        owner = self.gpu if self._gpu_head else self.cpu
        self.ylog, self.ylog_n, self.ylog_map = owner.ylog, owner.ylog_n, owner.ylog_map

    def _check_pos(self, pos):
        pos = int(pos)
        if not 0 <= pos < self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        return pos

    def _run(self, tok, pos, head):
        """Run one token through both backends; return the head token when head is true."""
        pos = self._check_pos(pos)
        cpu_set = frozenset(self.cpu_range)
        cur = self.cpu if 0 in cpu_set else self.gpu
        cur.begin(tok)
        for il in range(self.NL):
            want = self.cpu if il in cpu_set else self.gpu
            if want is not cur:
                want.set_hidden(cur.get_hidden())
                cur = want
            cur.layer(il, pos)
        if not head:
            if cur is self.gpu:
                self.gpu.flush()
            return None
        out = self.gpu.finish() if self._gpu_head else self.cpu.head()
        self._sync_logits()
        return int(out)

    def reset(self):
        self.cpu.reset()
        self.gpu.reset()

    def feed(self, tok, pos):
        """Process one prompt token without evaluating the output head."""
        self._run(tok, pos, False)

    def step(self, tok, pos):
        """Process one token and return the greedy next token; the logits stay on the head's backend."""
        return self._run(tok, pos, True)

    def generate(self, ids, n):
        self.reset()
        ids = list(ids)
        for pos, tok in enumerate(ids[:-1]):
            self.feed(tok, pos)
        pos = len(ids) - 1
        out = [self.step(ids[-1], pos)]
        pos += 1
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out


class TandemRunner(object):
    """The small request runner the server needs when prompt-cache snapshots are unavailable.

    A tandem state is split over two ownership domains; the normal cache checkpoints assume one
    engine's M/conv/Kc/Vc arrays.  Until that state is checkpointed jointly, the tandem path reads
    the prompt without prefix reuse.  It keeps the server's prepare/step contract unchanged.
    """

    def __init__(self, eng, log=print, progress=None):
        self.eng, self.log, self.progress = eng, log, progress or {}
        self.tokens = []
        self.ckpt = {}
        self.cache_errors = 0
        self.last_cache_error = ""

    def prepare(self, T, boundary, opening_end=None, cancel=None):
        from .cache import Cancelled
        self.eng.reset()
        self.tokens = []
        self.progress.update(phase="reading prompt", cached=0, read=0)
        if not T:
            raise ValueError("the prompt is empty")
        for pos, tok in enumerate(T[:-1]):
            if cancel is not None and cancel.is_set():
                raise Cancelled("the client went away after %d of %d prompt tokens" % (pos, len(T)))
            self.eng.feed(tok, pos)
            self.progress["read"] = pos + 1
        nxt = self.eng.step(T[-1], len(T) - 1)
        self.tokens = list(T)
        self.progress["read"] = len(T)
        return nxt, 0

    def step(self, tok):
        nxt = self.eng.step(tok, len(self.tokens))
        self.tokens.append(tok)
        return nxt

    def idle(self):
        pass
