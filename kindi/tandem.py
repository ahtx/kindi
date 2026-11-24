# SPDX-License-Identifier: AGPL-3.0-only
"""Static CPU+GPU layer tandem for the dense Bonsai packs.

Two placements are supported:

* ``cpu-first``: the CPU owns layers ``0..n-1``, the GPU owns ``n..NL-1`` and the output head;
* ``gpu-first``: the GPU owns ``0..NL-n-1`` and the embedding, the CPU owns the last ``n`` layers
  and the output head.

The hidden state is handed over once per token, between two complete residual streams.  Each
backend keeps its own GDN state and K/V cache, so no per-layer synchronisation is needed inside a
token.  The CPU engine is the existing AVX2/FMA/F16C backend (with AVX2/F16C no-FMA and AVX-only
fallbacks).  The GPU side is either the plain engine (``Engine(layers=...)``, fp16 K/V) or - the
default - the plan engine (``TPEngine(layers=...)``, q8_0 K/V): both load only the layers in their
range and allocate K/V only for the attention layers they own, so an offloaded layer really frees
VRAM instead of being loaded and ignored.  ``--tandem-gpu auto|plan|plain`` picks the owner
(``KINDI_TANDEM_GPU`` sets a default); a plan owner is one card, which is the only case it cannot
serve.

This is deliberately not a MoE engine.  A dense model needs every weight on every token, so an
expert-cache policy has nothing to cache.  The MoE approach is a separate design in
``docs/TANDEM-RUNTIME.md`` and is not implemented here.
"""
import hashlib
import json
import os

import numpy as np

from .cpu import Engine as CPUEngine
from .engine import Engine as GPUEngine

MODES = ("cpu-first", "gpu-first")
PlanEngine = None          # kindi.tp.TPEngine, imported on first use (it needs CuPy at import time)


def plan_engine_class():
    """The plan engine class, imported lazily so the CPU-only tests and the plain path never need it."""
    global PlanEngine
    if PlanEngine is None:
        from .tp import TPEngine as _TP
        PlanEngine = _TP
    return PlanEngine


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
PLAN_SCRATCH_MIB = 160     # the plan engine's own fixed buffers (prompt chunk buffers are not reserved here)
Q8_0 = 1.0 + 2.0 / 32      # q8_0: one int8 per value plus one fp16 scale per 32 values, two arrays (K and V)
GPU_OWNERS = ("auto", "plan", "plain")
MIB = 2 ** 20


def vram_budget_bytes(free_bytes, reserve_mib=512, budget_mib=None):
    """The GPU owner's planning ceiling: free memory less a reserve, capped by an explicit budget.

    ``reserve_mib`` is the headroom left for the display, the driver and allocation noise;
    ``budget_mib`` is an optional hard ceiling for the card (a P106-100 limited to 6 GiB, or an
    8 GiB display card limited to 7 GiB). A live free-memory figure still wins when it is lower:
    the planner never promises memory that is not currently available. ``budget_mib`` of 0 or
    None means no explicit ceiling.
    """
    try:
        reserve = int(reserve_mib or 0)
        ceiling = int(budget_mib or 0)
    except (TypeError, ValueError):
        raise SystemExit("kindi: tandem VRAM reserve and budget must be integers in MiB") from None
    if reserve < 0 or ceiling < 0:
        raise SystemExit("kindi: tandem VRAM reserve and budget cannot be negative")
    available = max(0, int(free_bytes) - reserve * MIB)
    if ceiling:
        available = min(available, ceiling * MIB)
    return available


def _pick_owner(kind, log=None):
    """-> 'plan' or 'plain' for the GPU side of a split.

    ``auto`` takes the plan engine whenever it can run: it is the default engine for every model it
    supports (the 2B/4B/9B are 1.5-2.5x faster there, and the 27B's kernels only exist there), its q8_0
    K/V is half the size of the plain engine's fp16 K/V, and it loads only the tensors of the layers it
    owns - which is what makes an offloaded layer free real VRAM. ``plain`` stays available for an A/B
    comparison and for the one case the plan engine cannot serve: a split across two GPU cards (a tandem
    GPU owner is one card).
    """
    kind = str(kind or "auto").strip().lower()
    if kind not in GPU_OWNERS:
        raise SystemExit("kindi: the tandem GPU owner must be auto, plan or plain (got %r)" % kind)
    log = log or (lambda *a: None)
    if kind != "auto":
        return kind
    from .engine import cp
    if cp is None or os.environ.get("KINDI_TP", "") == "0":
        return "plain"
    try:
        cards = int(os.environ.get("KINDI_CARDS") or cp.cuda.runtime.getDeviceCount())
    except Exception:                                              # noqa: BLE001 - no usable card: plain owner
        return "plain"
    if cards > 1:
        log("tandem: %d cards visible, the GPU owner stays the plain engine (a plan owner is one card)" % cards)
        return "plain"
    return "plan"


def pack_bytes(pack, man=None):
    """-> {'layers': {il: bytes}, 'f32_layers': {il: bytes}, 'f32': bytes, 'f32_global': bytes,
    'embed', 'head'} for the device footprint.

    A ternary tensor costs ``rows * ne0 / 4`` bytes of bcodes plus one fp16 scale per 128-column group;
    an f32 tensor costs 4 bytes per value. The plain engine uploads exactly the selected tensors and the
    whole f32 file; the plan engine reads per tensor, so only the owned layers' norms reach the card. These
    are the engine's device bytes, not the pack's.
    """
    man = man if man is not None else json.load(open(os.path.join(pack, "kindi_pack.json")))
    bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
    idx = json.load(open(os.path.join(pack, "index.json")))["tensors"]

    def nbytes(name):
        rec = bi.get(name)
        if rec is not None:
            return rec["rows"] * rec["ne0"] // 4 + rec["rows"] * (rec["ne0"] // 128) * 2
        return int(idx.get(name, {}).get("nfloats", 0)) * 4

    layers, f32_layers = {}, {}
    for name in bi:
        if name.startswith("blk."):
            il = int(name.split(".")[1])
            layers[il] = layers.get(il, 0) + nbytes(name)
    for name in idx:
        if name.startswith("blk."):
            il = int(name.split(".")[1])
            if name in bi:
                continue
            layers[il] = layers.get(il, 0) + nbytes(name)
            if "f32_off" in idx[name]:
                f32_layers[il] = f32_layers.get(il, 0) + nbytes(name)
    head_name = man.get("head", "token_embd.weight")
    f32 = int(man.get("f32_bytes", 0))
    return {"layers": layers, "f32_layers": f32_layers, "embed": nbytes("token_embd.weight"),
            "head": nbytes(head_name), "f32": f32,
            "f32_global": max(0, f32 - sum(f32_layers.values()))}


def kv_bytes(dims, ctx, mode="f16"):
    """K and V bytes for one attention layer at this context: fp16 (plain) or q8_0 (plan)."""
    values = int(ctx) * int(dims["kv_heads"]) * int(dims["head_dim"]) * 2      # K and V
    if mode == "q8_0":
        return int(values * Q8_0)
    return values * 2                                                          # fp16


def gpu_estimate(dims, sizes, ctx, cpu_layers=(), owner="plain"):
    """Device bytes the GPU owner needs when it owns every layer except ``cpu_layers``.

    ``plain``: the whole f32 file (it is uploaded as one file), the output head, the embedding unless the
    CPU owns layer 0, each remaining layer's tensors, the fp16 K/V of each remaining attention layer and
    the measured plain scratch.

    ``plan``: the same weights, but the f32 file only for the owned layers (the plan engine reads tensor by
    tensor), q8_0 K/V (half of fp16), no device copy of the embedding (it lives in pinned host memory) and
    the plan engine's own scratch. This is the reason to prefer the plan owner: an offloaded layer costs
    its weights, and the layers that stay cost half the K/V for the same context.
    """
    NL = int(dims["layers"])
    interval = max(1, int(dims["full_attention_interval"]))
    cpu = frozenset(int(x) for x in cpu_layers)
    gpu = [il for il in range(NL) if il not in cpu]
    f32l = sizes.get("f32_layers") or {}
    if owner == "plan":
        total = int(sizes.get("f32_global", 0)) + PLAN_SCRATCH_MIB * 2 ** 20
        if gpu:
            total += sizes["head"]
        for il in gpu:
            total += sizes["layers"].get(il, 0) + int(f32l.get(il, 0))
            if (il + 1) % interval == 0:
                total += kv_bytes(dims, ctx, "q8_0")
        return total
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


def plan_cpu_layers(dims, sizes, ctx, free_bytes, reserve_mib=512, target_mib=None, log=None,
                    owner="plain", vram_budget_mib=None):
    """The smallest CPU layer set that makes the GPU side fit, or ``()`` when no split is needed.

    Gated-delta layers go first, ascending: they cost no device K/V and little per-token CPU time, so
    they free VRAM at the lowest rate price (see ``gdn_first``). Attention layers are the last resort,
    because a CPU attention layer's cost grows with the context. The estimate is the GPU owner's actual
    allocation (see ``gpu_estimate``) plus a measured scratch allowance; with the plain owner on the 27B
    at 24K it predicts 7,778 MiB against a measured 7,783 MiB peak.

    The fit budget is ``vram_budget_bytes(free_bytes, reserve_mib, vram_budget_mib)``. The explicit
    budget is what makes a split happen on a card with plenty of free memory but a lower usable
    ceiling, such as a display card that must leave room for the desktop.
    """
    log = log or (lambda *a: None)
    NL = int(dims["layers"])
    interval = max(1, int(dims["full_attention_interval"]))
    budget = vram_budget_bytes(free_bytes, reserve_mib, vram_budget_mib)
    free_target = None if target_mib is None else int(target_mib) * 2 ** 20
    full = gpu_estimate(dims, sizes, ctx, (), owner=owner)

    order = [il for il in range(NL) if (il + 1) % interval] + [il for il in range(NL)
                                                              if (il + 1) % interval == 0]
    cpu = []
    while True:
        est = gpu_estimate(dims, sizes, ctx, cpu, owner=owner)
        over = est > budget
        short = free_target is not None and (full - est) < free_target
        if not over and not short:
            break
        if len(cpu) + 1 >= NL:
            raise SystemExit("kindi: tandem auto cannot fit this model at %d tokens on the usable GPU "
                             "budget of %d MiB (needs about %d MiB more); reduce --ctx, the reserve "
                             "or the ceiling"
                             % (int(ctx), budget // 2 ** 20, max(est - budget, 0) // 2 ** 20))
        cpu.append(order[len(cpu)])
    if not cpu:
        return ()
    cpu = tuple(sorted(cpu))
    est = gpu_estimate(dims, sizes, ctx, cpu, owner=owner)
    ceiling = (" (budget %d MiB)" % int(vram_budget_mib) if vram_budget_mib else "")
    log("tandem auto: %d CPU layer%s (%s) takes the %s GPU side to ~%d MiB "
        "(%d MiB freed; %d MiB budget%s)"
        % (len(cpu), "" if len(cpu) == 1 else "s", ",".join(str(x) for x in cpu), owner,
           est // 2 ** 20, (full - est) // 2 ** 20, budget // 2 ** 20, ceiling))
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

    def __init__(self, pack, ctx=4096, cpu_layers=None, mode="cpu-first", log=print, gpu=None,
                 vram_budget_mib=None, reserve_mib=None):
        self.log = log or (lambda *a: None)
        manifest = os.path.join(pack, "kindi_pack.json")
        man = json.load(open(manifest))
        self.gpu_kind = _pick_owner(gpu if gpu is not None else os.environ.get("KINDI_TANDEM_GPU", "auto"),
                                    log=self.log)
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
            need = gpu_estimate(dims, sizes, ctx, owner=self.gpu_kind) // 2 ** 20
            try:
                if reserve_mib is None:
                    reserve_mib = os.environ.get("KINDI_TANDEM_RESERVE_MB", "512")
                if vram_budget_mib is None:
                    vram_budget_mib = os.environ.get("KINDI_TANDEM_VRAM_MB", "0") or 0
                reserve_mib = int(reserve_mib or 0)
                vram_budget_mib = int(vram_budget_mib or 0) or None
            except (TypeError, ValueError):
                raise SystemExit("kindi: tandem VRAM reserve and budget must be integers in MiB") from None
            budget = vram_budget_bytes(free, reserve_mib, vram_budget_mib)
            if auto_target is None and need <= budget // 2 ** 20:
                raise SystemExit("kindi: this model already fits at %d tokens (~%d MiB needed, %d MiB usable%s); "
                                 "a split would only make it slower - run without --tandem, ask for a "
                                 "target with --tandem-cpu-layers auto:<MiB>, or lower --vram-budget-mib"
                                 % (int(ctx), need, budget // 2 ** 20,
                                    " (%d MiB reserve)" % reserve_mib if reserve_mib else ""))
            if cpu_layers == "auto":
                cpu_layers = plan_cpu_layers(dims, sizes, ctx, free,
                                             reserve_mib=reserve_mib, target_mib=auto_target,
                                             log=self.log, owner=self.gpu_kind,
                                             vram_budget_mib=vram_budget_mib)
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
        if self.gpu_kind == "plan":
            # the plan engine loads only the GPU-owned tensors and allocates q8_0 K/V for the owned
            # attention layers only: an offloaded layer frees its weights and its K/V, not just its time
            Plan = plan_engine_class()
            self.gpu = Plan(pack, ctx=ctx, log=log, layers=self.gpu_range,
                            embed=0 in self.gpu_range, head=self._gpu_head)
        else:
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
        self.log("tandem %s: CPU layers %s, %s engine on layers %s, %d-token context, %s head"
                 % (self.mode, _rng(self.cpu_range), self.gpu_kind, _rng(self.gpu_range),
                    self.ctx, "GPU" if self._gpu_head else "CPU"))

    def _sync_logits(self):
        owner = self.gpu if self._gpu_head else self.cpu
        self.ylog, self.ylog_n, self.ylog_map = owner.ylog, owner.ylog_n, owner.ylog_map

    def cache_fingerprint(self):
        """The split and the CPU arithmetic environment, for a saved opening's fingerprint."""
        return {"mode": self.mode, "cpu_layers": self.cpu_layers,
                "cpu_range": list(self.cpu_range), "gpu_range": list(self.gpu_range),
                "gpu_owner": self.gpu_kind, "gpu_kv": getattr(self.gpu, "kv", None) or "f16",
                "cpu_isa": getattr(self.cpu, "isa", None),
                "cpu_threads": getattr(self.cpu, "threads", None),
                "cpu_ctx": int(self.cpu.ctx), "gpu_ctx": int(self.gpu.ctx),
                "code": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]}

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
