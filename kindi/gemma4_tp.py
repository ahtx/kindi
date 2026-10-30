"""Kindi Gemma 4 on two cards in tensor parallel (Nemotron 3.5 Bolt E4B on two Quadro K1200s).

Each card holds one half of every layer:
  attention  query heads [4c, 4c+4) and KV head c (rows of attn_q, attn_k, attn_v); attn_output takes the same
             columns and gives a partial sum of all D outputs;
  FFN        ffn_gate / ffn_up rows [c F/2, (c+1) F/2), ffn_down the same columns (a partial sum again);
  head       token_embd rows [c V/2, (c+1) V/2).
Everything else (norms, the per-layer input path, per_layer_model_proj) is on both cards. The input embedding
row and the per-layer embedding row are read from host memory. The two partial sums of each layer meet through
mapped host memory (k_g4_tp_xsum) and are added in the same order on both cards, so the residual stream is the
same on the two cards.

Decode: each card replays its own CUDA graph; the exchanges wait on the GPU, so the host only launches both
graphs and joins the two halves of the logits. Prompt: one thread per card; the partials meet at a barrier.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import concurrent.futures
import copy
import ctypes
import os
import threading

import numpy as np

from .gemma4 import Engine, TensorFile, inspect_model, _QUANTS


class SliceFile(TensorFile):
    """the GGUF tensors as one card sees them: split tensors are cut to this card's rows or columns"""

    def __init__(self, path, card, cards, plan):
        TensorFile.__init__(self, path)
        self.card, self.cards, self.plan, self._cut = card, cards, plan, {}
        for name, how in plan.items():
            t = self.tensors[name]; rows, cols = t["shape"]
            t["shape"] = (rows // cards, cols) if how == "rows" else (rows, cols // cards)

    def raw(self, name):
        how = self.plan.get(name)
        if how is None:
            return TensorFile.raw(self, name)
        if name not in self._cut:
            t = self.tensors[name]; block, size = _QUANTS[t["type"]]
            rows, cols = t["shape"]
            full_rows, full_cols = (rows * self.cards, cols) if how == "rows" else (rows, cols * self.cards)
            nb = full_cols // block
            a = np.frombuffer(memoryview(self.map)[t["offset"]:t["offset"] + full_rows * nb * size], dtype=np.uint8)
            a = a.reshape(full_rows, nb, size)
            c = self.card
            if how == "rows":
                part = a[c * rows:(c + 1) * rows]
            else:
                if (cols % block) != 0:
                    raise ValueError("%s: a card's columns are not whole blocks" % name)
                part = a[:, c * (cols // block):(c + 1) * (cols // block)]
            self._cut[name] = np.ascontiguousarray(part).reshape(-1)
        return memoryview(self._cut[name])


def _plan(info, file):
    """which tensors are split, and how"""
    d = info["dimensions"]; plan = {"token_embd.weight": "rows"}
    for i in range(d["layers"]):
        p = "blk.%d." % i
        for n in ("attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"):
            plan[p + n + ".weight"] = "rows"
        for n in ("attn_output", "ffn_down"):
            plan[p + n + ".weight"] = "cols"
    return {n: h for n, h in plan.items() if n in file.tensors}


class Shard(Engine):
    """one card's engine: half the heads and half the FFN, with the exchange hooks filled in"""

    def __init__(self, tp, card, manifest, ctx, info, file, log):
        self.tp, self.card = tp, card
        Engine.__init__(self, manifest, ctx=ctx, log=log, file=file, info=info)
        cp = self.cp
        self.emb_host = True                          # the input row: any token, from the full table on the host
        self._emb_table = tp.emb_full
        self._emb_row = cp.empty(tp.emb_full.shape[1], dtype=cp.uint8)
        self.ctr = cp.zeros(1, dtype=cp.int32); self.fail = cp.zeros(1, dtype=cp.int32)

    def _xch(self, partial):
        cp = self.cp; out = cp.empty_like(partial)
        self.ops.k.k_g4_tp_xsum((1,), (1024,), (partial, np.uint64(self.tp.hb_ptr), np.uint64(self.tp.flag_ptr),
                                np.int32(self.card), self.ctr, self.fail, out, np.int32(partial.size)))
        return out

    def _xch_b(self, partial):
        return self.tp.barrier_sum(self.card, partial)


class TPEngine(object):
    """Nemotron / Gemma 4 on two cards; forward() and forward_batch() as Engine's"""

    def __init__(self, manifest, ctx=8192, log=print):
        import cupy as cp
        self.cp, self.log = cp, log
        n = cp.cuda.runtime.getDeviceCount()
        if n < 2:
            raise RuntimeError("kindi: tensor parallel needs two visible GPUs, %d found" % n)
        path = manifest["model_path"]
        info = inspect_model(path)
        d = info["dimensions"]
        if d["heads"] % 2 or d["kv_heads"] % 2 or d["ffn_length"] % 512:
            raise ValueError("kindi: this model does not split in two (heads %d, KV heads %d, FFN %d)"
                             % (d["heads"], d["kv_heads"], d["ffn_length"]))
        half = copy.deepcopy(info); hd = half["dimensions"]
        hd["heads"] //= 2; hd["kv_heads"] //= 2; hd["ffn_length"] //= 2
        probe = TensorFile(path)
        plan = _plan(info, probe)
        V, D = d["vocab"], d["width"]
        self.emb_full = np.frombuffer(probe.raw("token_embd.weight"), dtype=np.uint8).reshape(V, (D // 256) * 110)
        self._probe = probe                                    # keeps the mapping behind emb_full open
        flags = cp.cuda.runtime.hostAllocMapped | cp.cuda.runtime.hostAllocPortable
        self.hmem = cp.cuda.PinnedMemory(2 * 2 * D * 4, flags=flags)
        self.fmem = cp.cuda.PinnedMemory(64, flags=flags)
        np.ctypeslib.as_array((ctypes.c_int32 * 16).from_address(self.fmem.ptr))[:] = 0
        self.hb_ptr, self.flag_ptr = self.hmem.ptr, self.fmem.ptr   # UVA: the host pointers work on both cards
        self._bar = threading.Barrier(2); self._slot = [None, None]
        # one long-lived thread per card: CuPy keeps a BLAS handle per thread and device, and a new thread per prompt
        # batch would make a new handle each time (that fails on a full card). Each thread makes its handle once, at its
        # first prompt batch: made before the context fit, the cuBLAS code would cost a 4 GB card most of its context.
        self._work = [concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="kindi-tp%d" % c)
                      for c in range(2)]
        self.cards = []
        for c in range(2):
            with cp.cuda.Device(c):
                self.cards.append(Shard(self, c, manifest, ctx, half, SliceFile(path, c, 2, plan), log))
        self.ctx = min(s.ctx for s in self.cards)
        self.V, self.D = V, D
        self.info, self.shortlist, self.fast_centroids = info, 0, None
        self._graphs = None
        log("Kindi Gemma 4 tensor parallel ready on %s + %s, ctx %d" %
            (self.cards[0].residency["device"], self.cards[1].residency["device"], self.ctx))

    # ---------------------------------------------------------------- prompt
    def barrier_sum(self, card, partial):
        """prompt path: card 0's partial + card 1's, in that order, on both cards"""
        cp = self.cp
        cp.cuda.Stream.null.synchronize()
        self._slot[card] = partial
        self._bar.wait()
        other = self._slot[1 - card]
        with cp.cuda.Device(card):
            o = cp.empty_like(partial); o.data.copy_from_device(other.data, other.nbytes)
            out = partial + o if card == 0 else o + partial
        cp.cuda.Stream.null.synchronize()
        self._bar.wait()                                       # both have read before the slots are reused
        return out

    def forward_batch(self, tokens, pos):
        err = [None, None]

        def run(c):
            try:
                with self.cp.cuda.Device(c):
                    self.cards[c].forward_batch(tokens, pos)
            except BaseException as e:                         # pragma: no cover - reported below
                err[c] = e; self._bar.abort()
        for f in [self._work[c].submit(run, c) for c in range(2)]:
            f.result()
        for e in err:
            if e is not None:
                raise e

    # ---------------------------------------------------------------- decode
    def _build(self, token, pos):
        """capture one decode graph per card. The warm-up runs both cards at once (their exchanges wait for each
        other on the GPU), then each card's step is captured."""
        cp = self.cp
        from . import gpudrv
        P = ctypes.c_void_p; gs = []
        for c, s in enumerate(self.cards):
            with cp.cuda.Device(c):
                g = dict(st=cp.zeros(2, dtype=cp.int32), stream=cp.cuda.Stream(), pool=cp.cuda.MemoryPool(), drv=gpudrv.driver())
                g["st"].set(np.array([token, pos], dtype=np.int32))
                s._ple_row.set(s._ple_table[int(token)]) if s.ple_host else None
                s._emb_row.set(self.emb_full[int(token)])
                gs.append(g)
        for c, s in enumerate(self.cards):                      # warm-up: compiles every kernel; no host wait between
            with cp.cuda.Device(c), cp.cuda.using_allocator(gs[c]["pool"].malloc), gs[c]["stream"]:
                s._forward_graph(gs[c]["st"])
        for c in range(2):
            with cp.cuda.Device(c):
                gs[c]["stream"].synchronize()
        for c, s in enumerate(self.cards):
            g = gs[c]
            with cp.cuda.Device(c), cp.cuda.using_allocator(g["pool"].malloc), g["stream"]:
                assert g["drv"].cuStreamBeginCapture_v2(P(g["stream"].ptr), 2) == 0
                g["out"] = s._forward_graph(g["st"])
                graph = P(); assert g["drv"].cuStreamEndCapture(P(g["stream"].ptr), ctypes.byref(graph)) == 0
                ex = P(); assert g["drv"].cuGraphInstantiateWithFlags(ctypes.byref(ex), graph, ctypes.c_ulonglong(0)) == 0
                g["graph"], g["exec"] = graph, ex
        self._graphs = gs

    def forward(self, token, pos, logits=True):
        cp = self.cp; token, pos = int(token), int(pos)
        if pos < 0 or pos >= self.ctx:
            raise OverflowError("Gemma 4 context limit reached")
        if self._graphs is None:
            self._build(token, pos)
        gs = self._graphs
        for c, s in enumerate(self.cards):                      # both launches go out before either is waited on
            with cp.cuda.Device(c):
                if s.ple_host: s._ple_row.set(s._ple_table[token])
                s._emb_row.set(self.emb_full[token])
                gs[c]["st"].set(np.array([token, pos], dtype=np.int32), stream=gs[c]["stream"])
                assert gs[c]["drv"].cuGraphLaunch(gs[c]["exec"], ctypes.c_void_p(gs[c]["stream"].ptr)) == 0
        with cp.cuda.Device(1):
            gs[1]["stream"].synchronize()
        with cp.cuda.Device(0):
            gs[0]["stream"].synchronize()
            lo = gs[0]["out"]; hi = cp.empty_like(gs[1]["out"])
            hi.data.copy_from_device(gs[1]["out"].data, gs[1]["out"].nbytes)
            for s in self.cards:
                with cp.cuda.Device(s.card):
                    if int(s.fail.get()[0]):
                        raise RuntimeError("kindi: tensor parallel exchange timed out on card %d" % s.card)
            return cp.concatenate((lo, hi))

    ring_slots = 1024

    def step(self, token, pos):
        return int(self.cp.argmax(self.forward(token, pos)).item())

    def reset(self):
        return None

    def close(self):
        for s in self.cards:
            s.close()
