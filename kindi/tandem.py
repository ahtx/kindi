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
        cpu_layers = int(cpu_layers)
        if not 1 <= cpu_layers < self.NL:
            raise SystemExit("kindi: tandem cpu layers must be between 1 and %d (got %d)"
                             % (self.NL - 1, cpu_layers))
        self.cpu_layers = cpu_layers
        self.cpu_first = self.mode == "cpu-first"
        if self.cpu_first:
            self.cpu_range = tuple(range(0, cpu_layers))
            self.gpu_range = tuple(range(cpu_layers, self.NL))
        else:
            self.gpu_range = tuple(range(0, self.NL - cpu_layers))
            self.cpu_range = tuple(range(self.NL - cpu_layers, self.NL))
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
        self.log("tandem %s: CPU layers %d..%d, GPU layers %d..%d, %d-token context, %s head"
                 % (self.mode, self.cpu_range[0], self.cpu_range[-1], self.gpu_range[0], self.gpu_range[-1],
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
        if self.cpu_first:
            self.cpu.begin(tok)
            for il in self.cpu_range:
                self.cpu.layer(il, pos)
            self.gpu.set_hidden(self.cpu.get_hidden())
            for il in self.gpu_range:
                self.gpu.layer(il, pos)
            if not head:
                self.gpu.flush()
                return None
            out = self.gpu.finish()
        else:
            self.gpu.begin(tok)
            for il in self.gpu_range:
                self.gpu.layer(il, pos)
            self.cpu.set_hidden(self.gpu.get_hidden())
            for il in self.cpu_range:
                self.cpu.layer(il, pos)
            if not head:
                return None
            out = self.cpu.head()
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
