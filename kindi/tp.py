"""Kindi two-card tensor parallel engine (the 27B on two Quadro K1200s).

Each card holds one half of every layer and computes it:
  GDN      q/k groups [8c, 8c+8) with their v heads (tiled order: head rep*NK + nk), conv channels, gates, state;
  attention K/V heads [2c, 2c+2) with their G q heads each (q rows are [q | gate] per head);
  FFN      gate/up rows of whole Hadamard blocks - with an odd block count the middle block is computed on both
           cards - and ffn_down columns split exactly in half, so both cards do the same work;
  head     output rows [c * V/2, (c + 1) * V/2).
The three row-parallel products (ssm_out, attn_output, ffn_down) take the card's half of the input columns and
give a partial sum of all D outputs. k_xrms swaps the two partials through mapped pinned host memory, adds
[card 0 + card 1] in the same order on both cards, and does the next RMS norm and Hadamard fold, so the residual
stream is bit-identical on the two cards. The embedding row is read from host memory by both cards.
K/V cache: q8_0 (llama.cpp's -ctk q8_0 -ctv q8_0: blocks of 32 int8 with an fp16 scale); attention scores use
llama.cpp's flash-attention arithmetic for that cache (the query in q8_0, integer block dot products).

A launch plan records one token's kernels once (function, grid, block, ctypes arguments) and replays them through
cuLaunchKernel; only the token id and the position change between tokens (Dyn arguments). The exchange epochs are
counted on the GPU."""
import ctypes
import json
import os
import time

import numpy as np
import cupy as cp

from . import hostram
from . import layout
from .engine import Kernels, DEFAULT_DIMS, VOCAB

rt = cp.cuda.runtime
i32, f32t = np.int32, np.float32


POW5 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def g5_blob(B, S):
    """B: (C/4, R) uint8, byte s = trits of columns 4s .. 4s+3 (t0 + 3 t1 + 9 t2 + 27 t3); S: (C/128, R) uint16 fp16
    scales -> uint32 words [g][r][7]: per 128-column group and row, 24 bytes of 5 trits (columns 5j .. 5j+4), 2 bytes
    of 4 trits (columns 120 + 4m ..), the fp16 scale. Rows in chunks to bound the host memory."""
    ns, R = B.shape
    C = ns * 4
    NG = C // 128
    out = np.empty((NG, R, 28), dtype=np.uint8)
    for r0 in range(0, R, 16384):
        r1 = min(R, r0 + 16384)
        b = B[:, r0:r1].astype(np.uint16)
        tr = np.stack([b % 3, (b // 3) % 3, (b // 9) % 3, b // 27], axis=1).reshape(NG, 128, r1 - r0)
        b5 = (tr[:, :120].reshape(NG, 24, 5, r1 - r0) * POW5[None, None, :, None]).sum(axis=2, dtype=np.uint16)
        b4 = (tr[:, 120:].reshape(NG, 2, 4, r1 - r0) * POW5[None, None, :4, None]).sum(axis=2, dtype=np.uint16)
        out[:, r0:r1, :24] = b5.transpose(0, 2, 1)
        out[:, r0:r1, 24:26] = b4.transpose(0, 2, 1)
        out[:, r0:r1, 26:28] = np.ascontiguousarray(S[:, r0:r1]).view(np.uint8).reshape(NG, r1 - r0, 2)
    return out.reshape(-1).view(np.uint32)


def g5_shape(R, NG):
    """threads, rows per thread, column splits for the 5-trit GEMV (measured best on the GTX 1080)."""
    rpt = 8 if R >= 8192 else 4
    sp = 8 if R >= 16384 else 16
    return 512, rpt, max(1, min(sp, NG))


def g5_gx(R, nt, rpt):
    """blocks along the rows: rpt 0 = the lane kernel (k_g5n4: 4 tiles of 16 rows per warp)"""
    return (R // 16 + nt // 8 - 1) // (nt // 8) if rpt == 0 else (R + nt * rpt - 1) // (nt * rpt)


# (rows, 128-column groups, row-parallel) -> (threads, rows per thread, splits): measured bests per GPU (the best
# split count follows the number of compute units; 16 row-parallel splits fail the 27B verify on the 840M).
# Other AMD GPUs are timed at start.
_G5_780M = {(1024, 40, False): (512, 8, 32), (5120, 48, True): (512, 4, 4), (5120, 136, True): (512, 4, 8),
            (6144, 40, False): (512, 4, 8), (10240, 40, False): (512, 8, 4), (12288, 40, False): (512, 4, 4),
            (17408, 40, False): (512, 8, 2), (248320, 40, False): (512, 8, 1)}
_G5_840M = {(1024, 40, False): (512, 8, 8), (5120, 48, True): (512, 4, 4), (5120, 136, True): (512, 4, 8),
            (6144, 40, False): (512, 8, 16), (10240, 40, False): (512, 8, 8), (12288, 40, False): (512, 8, 2),
            (17408, 40, False): (512, 8, 4), (248320, 40, False): (512, 8, 2)}
# Strix Halo: (threads, rows per thread, splits, kernel), timed over the layers' tensors (DRAM, not the 32 MB cache);
# "a" = k_g5a* (no 5832-entry tables), "as" = k_g5a*s (and the parts added in the same launch)
_G5_8060S = {
    # 27B
    (10240, 40, False): (512, 1, 4, "as"), (6144, 40, False): (512, 1, 4, "as"), (5120, 48, True): (512, 1, 8, "a"),
    (17408, 40, False): (512, 2, 2, "as"), (5120, 136, True): (512, 1, 4, "a"), (12288, 40, False): (256, 2, 4, "a"),
    (1024, 40, False): (512, 1, 16, "as"), (248320, 40, False): (512, 1, 2, "as"),
    # 9B
    (8192, 32, False): (512, 1, 4, "as"), (4096, 32, False): (512, 1, 4, "as"), (4096, 32, True): (512, 1, 8, "a"),
    (12288, 32, False): (512, 2, 4, "as"), (4096, 96, True): (512, 1, 8, "a"), (1024, 32, False): (512, 1, 16, "as"),
    (248320, 32, False): (512, 1, 1, "a"),
    # 4B
    (8192, 20, False): (512, 1, 4, "as"), (4096, 20, False): (512, 1, 4, "as"), (2560, 32, True): (512, 1, 8, "a"),
    (9216, 20, False): (512, 1, 2, "as"), (2560, 72, True): (512, 1, 8, "a"), (1024, 20, False): (512, 1, 16, "as"),
    (248320, 20, False): (512, 1, 1, "a"),
    # 2B
    (6144, 16, False): (512, 1, 4, "as"), (2048, 16, False): (512, 1, 8, "as"), (2048, 16, True): (512, 1, 8, "a"),
    (2048, 48, True): (512, 1, 8, "a"), (4096, 16, False): (512, 1, 8, "as"), (512, 16, False): (512, 1, 16, "as"),
    (248320, 16, False): (512, 1, 1, "a")}
G5_PINNED_HIP = {"gfx1103": _G5_780M, "gfx1102": _G5_780M,     # Radeon 780M, 12 CUs (run as gfx1102)
                 "gfx1152": _G5_840M,                            # Radeon 840M (Ryzen AI 5 340), 4 CUs
                 "gfx1151": _G5_8060S}                           # Radeon 8060S (Ryzen AI MAX+ 395, Strix Halo), 40 CUs


def _g5_pins(s):
    if "g5pins" not in s.__dict__:
        arch = rt.getDeviceProperties(cp.cuda.Device().id).get("gcnArchName", b"")
        arch = (arch.decode() if isinstance(arch, bytes) else str(arch)).split(":")[0]
        s.g5pins = dict(G5_PINNED_HIP.get(arch, {}))
        # KINDI_G5_PINS="R,NG,rowpar,threads,rpt,splits;...": replace single shapes (a test of other configs)
        for e in filter(None, os.environ.get("KINDI_G5_PINS", "").split(";")):
            v = [int(x) for x in e.split(",")]
            s.g5pins[(v[0], v[1], bool(v[2]))] = tuple(v[3:6])
    return s.g5pins


def g5_tuned(s, name, R, NG, rowpar):
    """(threads, rows per thread, splits) for a 5-trit product of shape (R, 128 NG) on card s: timed once per shape."""
    pins = _g5_pins(s) if rt.is_hip and not s.g5l and os.environ.get("KINDI_G5_PIN", "1") == "1" else {}
    if (R, NG, rowpar) in pins and pins[(R, NG, rowpar)][2] * R <= s.gpart.size:
        c = tuple(pins[(R, NG, rowpar)])
        s.__dict__.setdefault("g5cfg", {})[(R, NG, rowpar)] = c[:3]
        s.__dict__.setdefault("g5var", {})[(R, NG, rowpar)] = c[3] if len(c) > 3 else "r"   # (threads, rpt, splits, kernel)
        return c[:3]
    if os.environ.get("KINDI_G5_TUNE", "1") != "1":
        nt, rpt, sp = g5_shape(R, NG)
        if s.g5l:
            nt, rpt = 256, 0
        return nt, rpt, (min(sp, 16) if rowpar else sp)
    key = (R, NG, rowpar)
    cache = s.__dict__.setdefault("g5cfg", {})
    if key in cache:
        return cache[key]
    K = s.k
    W5 = s.w[name][1]
    Ws = [W5]
    if rt.is_hip:
        # time the configs over the layers' tensors of this shape: one tensor run again and again stays in the
        # 32 MB cache of Strix Halo (400-480 "GB/s" there) and picks configs that lose when the weights come from DRAM
        Ws = [w[1] for w in s.w.values() if w[0] == "g5" and int(w[2]) == R and int(w[3]) == 128 * NG][:24] or [W5]
    Wt = (Ws * 10)[:max(10, len(Ws))]
    x = cp.zeros(128 * NG, dtype=cp.float32)
    fns = {0: K.k_g5n4, 1: K.k_g5r1, 2: K.k_g5r2, 4: K.k_g5r4, 8: K.k_g5r8}
    best = None
    ev0, ev1 = cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1"), cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1")
    for rpt in ((0,) if s.g5l else (2, 4, 8)):
        for nt in ((256,) if s.g5l else (256, 512)):
            gx = g5_gx(R, nt, rpt)
            for sp in (1, 2, 4, 8, 16, 32):
                # HIP: row-parallel products with 16 splits fail the 27B verify (840M), so 8 at most there
                if sp > NG or (rowpar and sp > (8 if rt.is_hip else 16)) or sp * R > s.gpart.size or gx * sp < 4:
                    continue
                f = fns[rpt]

                def run():
                    f((gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x))
                    if sp > 1 and not rowpar:
                        K.k_sumparts(((R + 255) // 256,), (256,), (s.gpart, s.y if R <= s.y.size else s.gpart, i32(R), i32(sp)))
                for _ in range(3):
                    run()
                ev0.record()
                for W5 in Wt:
                    run()
                ev1.record()
                ev1.synchronize()
                ms = cp.cuda.get_elapsed_time(ev0, ev1) / len(Wt) + (0.00002 * sp if rowpar else 0.0)   # k_sumparts reads the parts
                if best is None or ms < best[0]:
                    best = (ms, nt, rpt, sp)
    fz = None
    if rt.is_hip and not rowpar and not s.g5l and os.environ.get("KINDI_G5_FUSED", "1") == "1":
        # HIP: the column-split products that add their parts in the same launch (k_g5r*s), smaller blocks too
        ffn = {1: K.k_g5r1s, 2: K.k_g5r2s, 4: K.k_g5r4s, 8: K.k_g5r8s}
        for rpt in (1, 2, 4, 8):
            for nt in (128, 256, 512):
                gx = g5_gx(R, nt, rpt)
                for sp in (2, 4, 8, 16, 32):
                    if sp > NG or sp * R > s.gpart.size or gx > s.g5cnt.size or gx * sp < 4:
                        continue
                    f = ffn[rpt]
                    out = s.y if R <= s.y.size else s.qkv
                    for _ in range(3):
                        f((gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x, out, s.g5cnt))
                    ev0.record()
                    for W5 in Wt:
                        f((gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x, out, s.g5cnt))
                    ev1.record()
                    ev1.synchronize()
                    ms = cp.cuda.get_elapsed_time(ev0, ev1) / len(Wt)
                    if ms < best[0]:
                        best, fz = (ms, nt, rpt, sp), True
    var = "s" if fz else "r"
    if rt.is_hip and not s.g5l and os.environ.get("KINDI_G5_AB", "1") == "1":
        # HIP: k_g5a* (no 5832-entry tables, bit-identical): small blocks become cheap, so more blocks, fewer splits
        fab = {1: K.k_g5a1, 2: K.k_g5a2, 4: K.k_g5a4, 8: K.k_g5a8}
        for rpt in (1, 2, 4, 8):
            for nt in (64, 128, 256, 512):
                gx = g5_gx(R, nt, rpt)
                for sp in (1, 2, 4, 8, 16, 32):
                    if sp > NG or (rowpar and sp > 8) or sp * R > s.gpart.size or gx * sp < 4:
                        continue
                    f = fab[rpt]

                    def run():
                        f((gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x))
                        if sp > 1 and not rowpar:
                            K.k_sumparts(((R + 255) // 256,), (256,), (s.gpart, s.y if R <= s.y.size else s.gpart, i32(R), i32(sp)))
                    for _ in range(3):
                        run()
                    ev0.record()
                    for W5 in Wt:
                        run()
                    ev1.record()
                    ev1.synchronize()
                    ms = cp.cuda.get_elapsed_time(ev0, ev1) / len(Wt) + (0.00002 * sp if rowpar else 0.0)
                    if ms < best[0]:
                        best, var = (ms, nt, rpt, sp), "a"
                    if sp > 1 and not rowpar and gx <= s.g5cnt.size:
                        # the same with the parts added by the last block (k_g5a*s: no k_sumparts launch)
                        fs = {1: K.k_g5a1s, 2: K.k_g5a2s, 4: K.k_g5a4s, 8: K.k_g5a8s}[rpt]
                        out = s.y if R <= s.y.size else s.qkv
                        for _ in range(3):
                            fs((gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x, out, s.g5cnt))
                        ev0.record()
                        for W5 in Wt:
                            fs((gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x, out, s.g5cnt))
                        ev1.record()
                        ev1.synchronize()
                        ms = cp.cuda.get_elapsed_time(ev0, ev1) / len(Wt)
                        if ms < best[0]:
                            best, var = (ms, nt, rpt, sp), "as"
    s.__dict__.setdefault("g5var", {})[key] = var
    if os.environ.get("KINDI_G5_TUNELOG"):
        print("g5 tune %s: %s %s %.1f us, %.0f GB/s" % (key, best[1:], var, best[0] * 1e3, R * NG * 28 / best[0] / 1e6), flush=True)
    cache[key] = best[1:]
    return best[1:]


def _stream_to_card(path, chunk=1 << 26):
    """a uint32 file -> device array, read in chunks (256 MB); each chunk's page cache is dropped after the copy, so
    the host never holds more than one chunk (several servers can load at once)"""
    n = os.path.getsize(path) // 4
    out = cp.empty(n, dtype=cp.uint32)
    fd = os.open(path, os.O_RDONLY)
    try:
        with os.fdopen(os.dup(fd), "rb") as f:
            o = 0
            while o < n:
                a = np.fromfile(f, dtype=np.uint32, count=min(chunk, n - o))
                out[o:o + a.size].set(a)
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(fd, 4 * o, 4 * a.size, os.POSIX_FADV_DONTNEED)
                o += a.size
                del a
    finally:
        os.close(fd)
    return out


def _ranges(*rs):
    return np.concatenate([np.arange(a, b) for a, b in rs]).astype(np.int64)


# ====================================================================================== launch plans

class Dyn(object):
    """An argument a + b * var (var = 'pos' or 'tok'); ptr=True for a device address."""

    def __init__(self, var, a, b=1, ptr=False):
        self.var, self.a, self.b, self.ptr = var, int(a), int(b), ptr

    def holder(self):
        return ctypes.c_uint64(self.a) if self.ptr else ctypes.c_int32(self.a)


class Plan(object):
    def __init__(self):
        self.segs = [[]]                 # a new segment starts at every exchange
        self.dyn = []                    # (holder, Dyn)

    def cut(self):
        if self.segs[-1]:
            self.segs.append([])

    def add(self, kern, grid, block, args):
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = tuple(block) + (1,) * (3 - len(block))
        hs = []
        for a in args:
            if isinstance(a, cp.ndarray):
                h = ctypes.c_uint64(a.data.ptr)
            elif isinstance(a, Dyn):
                h = a.holder()
                self.dyn.append((h, a))
            elif isinstance(a, np.int32):
                h = ctypes.c_int32(int(a))
            elif isinstance(a, np.float32):
                h = ctypes.c_float(float(a))
            elif isinstance(a, np.uint64):
                h = ctypes.c_uint64(int(a))
            else:
                raise TypeError("plan argument %r" % type(a))
            hs.append(h)
        arr = (ctypes.c_void_p * len(hs))(*[ctypes.addressof(h) for h in hs])
        self.segs[-1].append((ctypes.c_void_p(kern.kernel.ptr), grid, block, arr, hs))

    def call(self, fn):
        """a host step in the plan: fn(n) runs at its place, with the number of tokens of the pass (chunk.feed)"""
        self.segs[-1].append(("call", fn))

    def set(self, tok, pos):
        for h, d in self.dyn:
            h.value = d.a + d.b * (pos if d.var == "pos" else tok)


_cuda = None


def _bsync_ev():
    return 1 if os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1" else 0


def _drv():
    global _cuda
    if _cuda is None:
        from . import gpudrv
        _cuda = gpudrv.driver()
        if gpudrv.HIP:                       # ROCm: the HIP stand-in has its argument types set
            return _cuda
        _cuda.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        _cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        _cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _cuda.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        _cuda.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _cuda.cuEventSynchronize.argtypes = [ctypes.c_void_p]
    return _cuda


# ====================================================================================== one card


class Shard(object):
    def __init__(self, tp, card, log):
        self.tp, self.card, self.log = tp, card, log
        self.D, self.NL, self.HD, self.HK, self.HB = tp.D, tp.NL, tp.HD, tp.HK, tp.HB
        self.dims = tp.dims
        self.NK, self.NV = tp.NKg // tp.W, tp.NVg // tp.W
        self.NH, self.NHKV = tp.NHg // tp.W, tp.NHKVg // tp.W
        self.G = self.NH // self.NHKV
        self.g0, self.g1 = tp.ffn_rows[card]           # gate/up rows (whole Hadamard blocks)
        self.d0, self.d1 = tp.ffn_cols[card]           # ffn_down columns
        self.NFF = self.g1 - self.g0
        self.CONV_CH = 2 * self.NK * self.HK + self.NV * self.HK
        self.NATT = self.NL // self.dims["full_attention_interval"]
        self.R_head = tp.head_split[card + 1] - tp.head_split[card]
        with cp.cuda.Device(card):
            self.k = Kernels()
            prop = rt.getDeviceProperties(card)
            self.nsm = prop["multiProcessorCount"]
            g5 = os.environ.get("KINDI_G5", "")
            self.g5 = g5 == "1" or (g5 != "0" and prop["major"] >= 6)
            # lane-owned tables (k_g5n4); on HIP the row kernels (k_g5r*) decode ~20% faster (780M, 27B)
            self.g5l = self.g5 and os.environ.get("KINDI_G5L", "0" if rt.is_hip and os.environ.get("KINDI_TC", "0") != "1" else "1") == "1"
            self.dp4a = prop["major"] * 10 + prop["minor"] >= 61 and os.environ.get("KINDI_DP4A", "1" if rt.is_hip else "0") == "1"   # dp4a scores: slower than the float path at long context on the 1080 (measured), off by default
            self._load()
            self._alloc_fixed()

    # ------------------------------------------------------------------ weights: this card's slices
    def _slices(self):
        tp, c, HK, HD = self.tp, self.card, self.HK, self.HD
        NK, NV, NKg = self.NK, self.NV, tp.NKg
        rep = NV // NK
        k0 = c * NK
        qk_rows = [(k0 * HK, (k0 + NK) * HK), (NKg * HK + k0 * HK, NKg * HK + (k0 + NK) * HK)]
        v_rows = [(2 * NKg * HK + (r * NKg + k0) * HK, 2 * NKg * HK + (r * NKg + k0 + NK) * HK) for r in range(rep)]
        gate_rows = [(a - 2 * NKg * HK, b - 2 * NKg * HK) for a, b in v_rows]
        vh = _ranges(*[(r * NKg + k0, r * NKg + k0 + NK) for r in range(rep)])          # this card's v heads
        gv0 = c * NV * HK                                                              # grouped GDN output columns
        h0, kvh0, a0 = c * self.NH, c * self.NHKV, c * self.NH * HD
        tern, flt = {}, {}
        for il in range(self.NL):
            p = "blk.%d." % il
            if (il + 1) % self.dims["full_attention_interval"]:
                tern[p + "attn_qkv.weight"] = (_ranges(*(qk_rows + v_rows)), None)
                tern[p + "attn_gate.weight"] = (_ranges(*gate_rows), None)
                tern[p + "ssm_out.weight"] = (None, (gv0, gv0 + NV * HK))
                flt[p + "ssm_alpha.weight"] = ("rows16", vh)
                flt[p + "ssm_beta.weight"] = ("rows16", vh)
                flt[p + "ssm_dt.bias"] = ("elems", vh)
                flt[p + "ssm_a"] = ("elems", vh)
                flt[p + "ssm_conv1d.weight"] = ("rows", _ranges(*(qk_rows + v_rows)))
            else:
                tern[p + "attn_q.weight"] = (_ranges((h0 * 2 * HD, (h0 + self.NH) * 2 * HD)), None)
                tern[p + "attn_k.weight"] = (_ranges((kvh0 * HD, (kvh0 + self.NHKV) * HD)), None)
                tern[p + "attn_v.weight"] = (_ranges((kvh0 * HD, (kvh0 + self.NHKV) * HD)), None)
                tern[p + "attn_output.weight"] = (None, (a0, a0 + self.NH * HD))
            tern[p + "ffn_gate.weight"] = (_ranges((self.g0, self.g1)), None)
            tern[p + "ffn_up.weight"] = (_ranges((self.g0, self.g1)), None)
            tern[p + "ffn_down.weight"] = (None, (self.d0, self.d1))
        tern["output.weight"] = (_ranges((tp.head_split[c], tp.head_split[c + 1])), None)
        self.cols = {"sV": (gv0, gv0 + NV * HK), "sA": (a0, a0 + self.NH * HD), "sF": (self.g0, self.g1)}
        return tern, flt

    def _load(self):
        tp = self.tp
        t0 = time.time()
        tern, flt = self._slices()
        tern_load = {} if self.g5 else tern       # 5-trit weights: the cache files, made from the slices below if missing
        bparts, dparts, bi = [], [], {}
        bo = do = 0
        if not self.g5:
            # the 2-bit weights of this card, written straight into one array each: one host copy (the parts and their
            # concatenation held two - ~7 GB per card for the 27B - and stalled a 15 GB host without swap)
            nb = nd = 0
            for name, (rows, cols) in tern_load.items():
                rec = tp.bi[name]
                r = len(rows) if rows is not None else rec["rows"]
                c = (cols[1] - cols[0]) if cols is not None else rec["ne0"]
                nb, nd = nb + c // 4 * r, nd + c // 128 * r
            hostram.check(nb + 2 * nd, "card %d's weights" % self.card, self.log)
            bc_all, dp_all = np.empty(nb, dtype=np.uint8), np.empty(nd, dtype=np.uint16)
        for name, (rows, cols) in tern_load.items():
            rec = tp.bi[name]
            R, C = rec["rows"], rec["ne0"]
            ns, P = C // 4, C // 128
            B = tp.bcm[rec["bc_off"]:rec["bc_off"] + ns * R].reshape(ns, R)
            S = tp.dpm[rec["dpm_off"] // 2:rec["dpm_off"] // 2 + P * R].reshape(P, R)
            if rows is not None:
                B, S = B[:, rows], S[:, rows]
            if cols is not None:
                B, S = B[cols[0] // 4:cols[1] // 4], S[cols[0] // 128:cols[1] // 128]
            bi[name] = {"bc_off": bo, "dpm_off": do * 2, "rows": B.shape[1], "ne0": B.shape[0] * 4}
            if self.g5:
                pass                              # 5-trit blobs are made tensor by tensor below, straight to the cache file
            else:
                np.copyto(bc_all[bo:bo + B.size].reshape(B.shape), B)
                np.copyto(dp_all[do:do + S.size].reshape(S.shape), S)
            bo += B.size
            do += S.size
        if self.g5:
            # g5cache: the repacked weights, kept next to the pack (keyed by the pack's source hash and the slice)
            key = "%s-w%d%s-c%d" % ((tp.man.get("source_sha256") or "pack")[:12], tp.W, getattr(tp, "split_tag", ""), self.card)
            cpath = os.path.join(tp.pack, "g5_%s.bin" % key)
            ipath = os.path.join(tp.pack, "g5_%s.json" % key)
            lpath = os.path.join(tp.pack, "g5n_%s.bin" % key)
            lane = False
            dual = not self.g5l and os.environ.get("KINDI_G5_DUAL", "1" if rt.is_hip else "0") == "1"
            # (re)built when a file this card uses is missing: a row cache made without the lane copy (by an older
            # build, or a run with dual off) left HIP prompts token by token (780M 27B read 10.8 tok/s)
            # a cache without one of this card's tensors (made before the tied-head alias: the 2B / 4B's output.weight)
            # is made again
            stale = os.path.exists(ipath) and not set(tern) <= set(json.load(open(ipath)))
            if stale or not (os.path.exists(cpath) and os.path.exists(ipath)) or \
                    ((self.g5l or dual) and not os.path.exists(lpath)):
                # one tensor at a time, from the pack's memory maps to the cache files (the row layout, and the 16-row
                # lane layout when it is used: k_g5n4 decode, or the chunked prompt GEMM next to row decode): the host
                # holds one tensor's blob, not the card's weights (a 2 GiB container loads the 27B)
                offs, o = {}, 0
                fr = open(cpath + ".tmp", "wb")
                fl = open(lpath + ".tmp", "wb") if (self.g5l or dual) else None
                for name, (rows, cols) in tern.items():
                    rec = tp.bi[name]
                    R, C = rec["rows"], rec["ne0"]
                    ns, P = C // 4, C // 128
                    B = tp.bcm[rec["bc_off"]:rec["bc_off"] + ns * R].reshape(ns, R)
                    S = tp.dpm[rec["dpm_off"] // 2:rec["dpm_off"] // 2 + P * R].reshape(P, R)
                    if rows is not None:
                        B, S = B[:, rows], S[:, rows]
                    if cols is not None:
                        B, S = B[cols[0] // 4:cols[1] // 4], S[cols[0] // 128:cols[1] // 128]
                    R, C = B.shape[1], B.shape[0] * 4
                    blob = g5_blob(B, S)
                    del B, S
                    offs[name] = (o, R, C)
                    o += blob.size
                    blob.tofile(fr)
                    if fl is not None:
                        # 16-row tiles, byte position major: tile m of group g = 28 x 16 bytes, byte p of rows 16m .. 16m+15
                        NG = C // 128
                        v = blob.view(np.uint8).reshape(NG, R // 16, 16, 28)
                        np.ascontiguousarray(v.transpose(0, 1, 3, 2)).tofile(fl)
                    del blob
                fr.close()
                os.replace(cpath + ".tmp", cpath)
                if fl is not None:
                    fl.close()
                    os.replace(lpath + ".tmp", lpath)
                json.dump(offs, open(ipath, "w"))
            offs = {n: tuple(v) for n, v in json.load(open(ipath)).items()}
            lane = self.g5l and os.path.exists(lpath)
            if self.g5l and not lane:
                raise SystemExit("kindi: the lane-layout cache %s is missing; delete %s to rebuild both" % (lpath, cpath))
            del bparts, dparts
            self.bc = _stream_to_card(lpath if lane else cpath)       # 256 MB at a time
            self.dp = cp.zeros(1, dtype=cp.uint16)
            self.w = {n: ("g5", self.bc[o:], i32(R), i32(C)) for n, (o, R, C) in offs.items()}
            # row-layout decode weights plus the lane-layout copy for the chunked prompt GEMM (k_mm_g5n needs it):
            # KINDI_G5_DUAL=1, the default on HIP, where the card's memory is the host's (unified memory)
            self.wl = None
            if dual and os.path.exists(lpath):
                self.bcl = _stream_to_card(lpath)
                self.wl = {n: ("g5", self.bcl[o:], i32(R), i32(C)) for n, (o, R, C) in offs.items()}
        else:
            bc, dp = bc_all, dp_all
            del bc_all, dp_all
            dims = layout.repack(bi, bc, dp)
            # straight from this host array (cp.asarray stages a numpy array through a pinned copy of the same size:
            # a second host copy of the card's weights during the upload)
            self.bc, self.dp = cp.empty(bc.shape, dtype=bc.dtype), cp.empty(dp.shape, dtype=dp.dtype)
            self.bc.set(bc)
            self.dp.set(dp)
            del bc, dp
            self.w = {n: (self.bc[bi[n]["bc_off"]:], self.dp[bi[n]["dpm_off"] // 2:], i32(R), i32(C))
                      for n, (R, C) in dims.items()}
        # float tensors: the replicated norms, and this card's slices of the GDN gate weights (kept in their source
        # bf16: the pack widened them exactly, the top 16 bits are the bf16 value) and the conv
        self._f = {}
        for name, rec in tp.idx["tensors"].items():
            if "f32_off" not in rec:
                continue
            off = rec["f32_off"] // 4
            a = np.asarray(tp.flm[off:off + rec["rows"] * rec["ne0"]])
            how = flt.get(name)
            if how is not None:
                kind, sel = how
                if kind in ("rows", "rows16"):
                    a = a.reshape(rec["rows"], rec["ne0"])[sel]
                else:
                    a = a[sel]
                if kind == "rows16":
                    a = (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
            if a.dtype != np.uint16:
                a = np.ascontiguousarray(a, dtype=np.float32)
            self._f[name] = cp.asarray(a.reshape(-1))
        s = tp.signs
        self.sD = cp.asarray(s[self.D])
        self.sV = cp.asarray(s[tp.NVg * self.HK][self.cols["sV"][0]:self.cols["sV"][1]])
        self.sA = cp.asarray(s[tp.NHg * self.HD][self.cols["sA"][0]:self.cols["sA"][1]])
        self.sF = cp.asarray(s[tp.NFFg][self.cols["sF"][0]:self.cols["sF"][1]])
        cp.get_default_pinned_memory_pool().free_all_blocks()
        self.log("card %d: %.2f GB of weights (%.0fs)" % (self.card, (self.bc.nbytes + self.dp.nbytes) / 1e9,
                                                          time.time() - t0))

    def f(self, name):
        return self._f[name]

    def _alloc_fixed(self):
        z = lambda n, t=cp.float32: cp.zeros(n, dtype=t)  # noqa: E731
        D, HK, HD = self.D, self.HK, self.HD
        work = max(D, self.NFF, self.NV * HK)
        self.y, self.xp, self.xn, self.zv = z(max(self.R_head, D)), z(work), z(D), z(self.NV * HK)
        self.hA, self.hB = z(D), z(D)
        self.part = z(self.nsm * work)
        if self.g5:
            self.gpart = z(16 * max(self.R_head, self.NFF, 2 * self.NH * HD, self.CONV_CH))
            self.g5cnt = cp.zeros(4096, dtype=cp.uint32)     # k_g5r*s: blocks done for each row range (back to 0 after)
        self.fail = cp.zeros(1, dtype=cp.int32)
        self.xctr = cp.zeros(8, dtype=cp.int32)
        self.qkv, self.cs = z(self.CONV_CH), z(self.CONV_CH)
        self.qkbuf, self.qkvbuf = z(2 * self.NK * HK), z(self.NHKV * HD)
        self.kn, self.qn, self.qr, self.kro = z(self.NHKV * HD), z(self.NH * 2 * HD), z(self.NH * HD), z(self.NHKV * HD)
        self.fo, self.fo6 = z(self.NH * HD), z(self.NFF)
        self.outgdn, self.att, self.ffg, self.ffu = z(self.NV * HK), z(self.NH * HD), z(self.NFF), z(self.NFF)
        self.gb, self.gg = z(self.NV), z(self.NV)
        self.qq, self.qd = cp.zeros(self.NH * HD, dtype=cp.int8), z(self.NH * HD // 32)
        self.aout = cp.zeros(1, dtype=cp.int32)
        interval = self.dims["full_attention_interval"]
        self.M = [cp.zeros((self.NV, HK, HK) if (il + 1) % interval else (1,), dtype=cp.float32)
                  for il in range(self.NL)]
        self.conv = [z(3 * self.CONV_CH) for _ in range(self.NL)]

    def per_pos_bytes(self):
        kv = self.NATT * 2 * self.NHKV * (self.HD + self.HD // 32 * 2)      # q8_0 K and V
        return kv + self.NH * 4 + (self.NH * self.HD * 4) // 256 + self.dims["rope_dim"] * 4 + 64

    def free_bytes(self):
        with cp.cuda.Device(self.card):
            return rt.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()

    def alloc_ctx(self, ctx):
        self.ctx = ctx
        z = lambda n, t=cp.float32: cp.zeros(n, dtype=t)  # noqa: E731
        with cp.cuda.Device(self.card):
            self.sbuf = z(self.NH * ctx)
            self.apart = z(((ctx + 255) // 256) * self.NH * self.HD)
            th = np.arange(ctx, dtype=np.float64)[:, None] * self.tp.inv_freq[None, :]
            self.cos_t = cp.asarray(np.cos(th).astype(np.float32))
            self.sin_t = cp.asarray(np.sin(th).astype(np.float32))
            # one allocation per array kind, a view per layer: 64 separate arrays each lose up to a 2 MiB page to
            # rounding, which the context fit does not see
            def kv(dt, last):
                a = cp.zeros((self.NATT, ctx, self.NHKV, last), dtype=dt)
                return [a[i] for i in range(self.NATT)]
            self.Kq, self.Vq = kv(cp.int8, self.HD), kv(cp.int8, self.HD)
            self.Kd, self.Vd = kv(cp.float16, self.HD // 32), kv(cp.float16, self.HD // 32)
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def reset(self):
        with cp.cuda.Device(self.card):
            for a in self.M + self.conv + self.Kq + self.Kd + self.Vq + self.Vd:
                a.fill(0)

    # ------------------------------------------------------------------ the plan
    def build_plans(self):
        s, tp, K = self, self.tp, self.k
        D, HB, HK, HD = s.D, s.HB, s.HK, s.HD
        NK, NV, NH, NHKV, NFF = s.NK, s.NV, s.NH, s.NHKV, s.NFF
        if HB != 1024 and not (HB in (512, 4096) and tp.W == 1):   # one card: the 2B / 4B / 9B folds work as well
            raise SystemExit("kindi: the tensor parallel engine needs Hadamard blocks of 1024")
        st = {"h": s.hA, "h2": s.hB, "pending": 0}

        def fold(P, kern, width, args, out):
            """a fold kernel over `width` values: Hadamard blocks of 512 / 1024 in one launch; larger blocks (the 9B's
            4096) as 512-point chunks on many blocks (HB = -512, as the single-card engine) and k_fwht_hi for the
            stages above 512 (one block for each 4096 values was slow on the 40-CU Strix Halo)"""
            if HB > 1024:
                P.add(kern, (width // 512,), (512,), tuple(args) + (i32(-512),))
                P.add(K.k_fwht_hi, ((width // 512 // (HB // 512) * 512 + 255) // 256,), (256,), (out, i32(HB), i32(width)))
            else:
                P.add(kern, (width // HB,), (512,), tuple(args) + (i32(HB),))

        def mixin(P, wname):
            """RMS norm + fold of the residual stream; with a pending row-parallel partial, the exchange and the
            residual add first (k_xrms)."""
            if st["pending"] and tp.W == 1:
                # one card: the pending row-parallel output (gy column parts, or one vector) is the whole sum; its
                # parts are added once (k_sumparts), then k_rmsfold adds it to the residual and folds
                # (HIP: k_rmsfold adds the parts itself, in k_sumparts' order: one launch less per block; each launch
                # costs ~2 us on Strix Halo)
                if rt.is_hip:
                    fold(P, K.k_rmsfold, D, (st["h"], s.part if st["pending"] > 1 else s.y, st["h2"], s.f(wname), s.sD,
                                             s.xn, s.xp, i32(D), i32(st["pending"])), s.xp)
                else:
                    if st["pending"] > 1:
                        P.add(K.k_sumparts, ((D + 255) // 256,), (256,), (s.part, s.y, i32(D), i32(st["pending"])))
                    fold(P, K.k_rmsfold, D, (st["h"], s.y, st["h2"], s.f(wname), s.sD, s.xn, s.xp, i32(D), i32(1)), s.xp)
                st["h"], st["h2"] = st["h2"], st["h"]
            elif st["pending"]:
                P.cut()
                P.add(K.k_xrms, (1,), (1024,), (s.part if st["pending"] > 1 else s.y, i32(st["pending"]),
                                               np.uint64(tp.xbuf_ptr), np.uint64(tp.flag_ptr), i32(s.card), s.fail,
                                               s.xctr, st["h"], st["h2"], s.f(wname), s.sD, s.xn, s.xp, i32(D)))
                st["h"], st["h2"] = st["h2"], st["h"]
            else:
                fold(P, K.k_rmsfold, D, (st["h"], s.y, st["h2"], s.f(wname), s.sD, s.xn, s.xp, i32(D), i32(0)), s.xp)
            st["pending"] = 0

        def gemv(P, names, x, outs):
            if s.g5:
                return gemv5(P, names, x, outs, s.part if outs[0] is s.y else None)
            a = [s.w[n] for n in names]
            args = []
            for i in range(3):
                if i < len(a):
                    args += [a[i][0], a[i][1], outs[i], a[i][2]]
                else:
                    args += [a[0][0], a[0][1], outs[0], i32(0)]
            R0, C0 = int(a[0][2]), int(a[0][3])
            if len(a) == 1 and s.nsm > 1 and (R0 >> 4) < 128 * s.nsm and (C0 >> 9) >= 2:
                # few row groups: split the rows over gx blocks and the 512-column phases over gy blocks
                # (gx * gy = nsm), the shape with the least idle work (k_g4 strides rows by gridDim.x and splits
                # phases by gridDim.y; the gy parts are added in the exchange)
                ng, nph = R0 >> 4, C0 >> 9
                best = None
                for gx in range(1, s.nsm + 1):
                    if s.nsm % gx:
                        continue
                    gy = s.nsm // gx
                    if gy > nph:
                        continue
                    cost = -(-(-(-ng // gx)) // 32) * -(-nph // gy)
                    if best is None or cost < best[0]:
                        best = (cost, gx, gy)
                _, gx, gy = best
                if gy > 1:
                    args[2] = s.part
                P.add(K.k_g4, (gx, gy), (1024,), tuple(args) + (x, a[0][3]))
                return gy
            P.add(K.k_g4, (s.nsm,), (1024,), tuple(args) + (x, a[0][3]))
            return 1

        def gemv5(P, names, x, outs, part):
            """5-trit products, one launch per tensor. A row-parallel output (part given, D rows) leaves its column
            parts in `part` for the next norm to add (returns their count); any other output adds its parts here."""
            ret = 1
            for n, out in zip(names, outs):
                _, W5, R, C = s.w[n]
                R, NG = int(R), int(C) // 128
                nt, rpt, sp = g5_tuned(s, n, R, NG, part is not None and R == D)
                var = s.__dict__.get("g5var", {}).get((R, NG, part is not None and R == D), "r")
                f = ({0: K.k_g5n4, 1: K.k_g5r1, 2: K.k_g5r2, 4: K.k_g5r4, 8: K.k_g5r8} if var != "a" else
                     {1: K.k_g5a1, 2: K.k_g5a2, 4: K.k_g5a4, 8: K.k_g5a8})[rpt]
                gx = g5_gx(R, nt, rpt)
                if part is not None and R == D:
                    P.add(f, (gx, sp), (nt,), (W5, part, i32(R), i32(NG), x))
                    ret = sp
                elif sp == 1:
                    P.add(f, (gx, 1), (nt,), (W5, out, i32(R), i32(NG), x))
                elif var in ("s", "as"):
                    fs = ({1: K.k_g5r1s, 2: K.k_g5r2s, 4: K.k_g5r4s, 8: K.k_g5r8s} if var == "s" else
                          {1: K.k_g5a1s, 2: K.k_g5a2s, 4: K.k_g5a4s, 8: K.k_g5a8s})[rpt]
                    P.add(fs, (gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x, out, s.g5cnt))
                else:
                    P.add(f, (gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x))
                    P.add(K.k_sumparts, ((R + 255) // 256,), (256,), (s.gpart, out, i32(R), i32(sp)))
            return ret

        def gdn(P, il, state_only=False):
            p = "blk.%d." % il
            mixin(P, p + "attn_norm.weight")
            gemv(P, [p + "attn_qkv.weight", p + "attn_gate.weight"], s.xp, [s.qkv, s.zv])
            nblk = NV + 2 * NK + (NV * HK + 255) // 256
            P.add(K.k_gdn_pre, (nblk,), (256,), (s.xn, s.f(p + "ssm_beta.weight"), s.f(p + "ssm_alpha.weight"),
                                                 s.f(p + "ssm_dt.bias"), s.f(p + "ssm_a"), s.gb, s.gg, i32(D),
                                                 s.qkv, s.conv[il], s.f(p + "ssm_conv1d.weight"), s.cs, s.qkbuf,
                                                 i32(s.CONV_CH), i32(NV), i32(NK), i32(HK)))
            P.add(K.k_gdn_deltan, (NV,), (HK,), (s.M[il], s.qkbuf[:NK * HK], s.qkbuf[NK * HK:], s.cs[2 * NK * HK:],
                                                 s.gg, s.gb, s.f(p + "ssm_norm.weight"), s.outgdn,
                                                 f32t(1.0 / np.sqrt(HK)), i32(NK)))
            if state_only:                         # a prompt token's last layer: only the recurrent state is used later
                st["pending"] = 0
                return
            fold(P, K.k_smfold, NV * HK, (s.zv, s.outgdn, s.sV, s.xp, i32(HK), i32(NK), i32(NV // NK)), s.xp)
            st["pending"] = gemv(P, [p + "ssm_out.weight"], s.xp, [s.y])

        def attn(P, il, kvi, kv_only=False):
            p = "blk.%d." % il
            mixin(P, p + "attn_norm.weight")
            if kv_only:                            # a prompt token's last layer: only its K/V are used later
                gemv(P, [p + "attn_k.weight", p + "attn_v.weight"], s.xp, [s.kro, s.qkvbuf])
                P.add(K.k_rmsw, ((NHKV + 7) // 8,), (256,), (s.kro, s.f(p + "attn_k_norm.weight"), s.kn, i32(HD), i32(NHKV)))
                rd0 = s.cos_t.shape[1] * 4
                P.add(K.k_rope2, (NHKV,), (256,), (s.kn, s.kn, Dyn("pos", s.cos_t.data.ptr, rd0, ptr=True),
                                                   Dyn("pos", s.sin_t.data.ptr, rd0, ptr=True), s.kro, s.kro, i32(0), i32(HD)))
                nb0 = NHKV * HD // 32
                P.add(K.k_store_q8, ((64 * nb0 + 1023) // 1024,), (min(1024, 64 * nb0),), (s.kro, s.qkvbuf,
                                                        Dyn("pos", s.Kq[kvi].data.ptr, NHKV * HD, ptr=True),
                                                        Dyn("pos", s.Kd[kvi].data.ptr, nb0 * 2, ptr=True),
                                                        Dyn("pos", s.Vq[kvi].data.ptr, NHKV * HD, ptr=True),
                                                        Dyn("pos", s.Vd[kvi].data.ptr, nb0 * 2, ptr=True), i32(nb0)))
                st["pending"] = 0
                return
            gemv(P, [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], s.xp, [s.qn, s.kro, s.qkvbuf])
            P.add(K.k_rmsw, ((NHKV + 7) // 8,), (256,), (s.kro, s.f(p + "attn_k_norm.weight"), s.kn, i32(HD), i32(NHKV)))
            P.add(K.k_rmsw_s, ((NH + 7) // 8,), (256,), (s.qn, i32(2 * HD), s.f(p + "attn_q_norm.weight"), s.fo, i32(HD), i32(NH)))
            rd = s.cos_t.shape[1] * 4
            P.add(K.k_rope2, (NH + NHKV,), (256,), (s.fo, s.kn, Dyn("pos", s.cos_t.data.ptr, rd, ptr=True),
                                                    Dyn("pos", s.sin_t.data.ptr, rd, ptr=True), s.qr, s.kro, i32(NH), i32(HD)))
            nblk = NHKV * HD // 32
            P.add(K.k_store_q8, ((64 * nblk + 1023) // 1024,), (min(1024, 64 * nblk),), (s.kro, s.qkvbuf,
                                                     Dyn("pos", s.Kq[kvi].data.ptr, NHKV * HD, ptr=True),
                                                     Dyn("pos", s.Kd[kvi].data.ptr, nblk * 2, ptr=True),
                                                     Dyn("pos", s.Vq[kvi].data.ptr, NHKV * HD, ptr=True),
                                                     Dyn("pos", s.Vd[kvi].data.ptr, nblk * 2, ptr=True), i32(nblk)))
            P.add(K.k_q8row, ((NH * HD + 1023) // 1024,), (1024,), (s.qr, s.qq, s.qd, i32(NH * HD // 32)))
            n = lambda: Dyn("pos", 1, 1)  # noqa: E731
            P.add(K.k_attn_sq8d if s.dp4a else K.k_attn_sq8t, (NHKV, (s.ctx + 255) // 256), (256,), (s.Kq[kvi], s.Kd[kvi], s.qq, s.qd, s.sbuf, n(), i32(NHKV), i32(s.G)))
            P.add(K.k_softmax_rows, (NH,), (256,), (s.sbuf, n()))
            P.add(K.k_attn_vq8w, (NHKV, 1, (s.ctx + 255) // 256), (128,), (s.sbuf, s.Vq[kvi], s.Vd[kvi], s.apart, n(),
                                                                         i32(NH), i32(NHKV), i32(s.G)))
            P.add(K.k_attn_combn, (NH,), (256,), (s.apart, s.qn, s.att, n(), i32(NH)))
            fold(P, K.k_fold_fwht, NH * HD, (s.att, s.sA, s.xp), s.xp)
            st["pending"] = gemv(P, [p + "attn_output.weight"], s.xp, [s.y])

        def ffn(P, il):
            p = "blk.%d." % il
            mixin(P, p + "post_attention_norm.weight")
            gemv(P, [p + "ffn_gate.weight", p + "ffn_up.weight"], s.xp, [s.ffg, s.ffu])
            fold(P, K.k_smfold, NFF, (s.ffg, s.ffu, s.sF, s.fo6, i32(0), i32(0), i32(0)), s.fo6)
            st["pending"] = gemv(P, [p + "ffn_down.weight"], s.fo6[s.d0 - s.g0:], [s.y])

        def body_plan(feed):
            st.update(h=s.hA, h2=s.hB, pending=0)
            P = Plan()
            emb = tp.emb
            P.add(K.k_deq_emb4, (D // HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, st["h"],
                                                  Dyn("tok", 0, 1), i32(D), i32(HB)))
            kvi = 0
            for il in range(s.NL):
                last = feed and il == s.NL - 1
                if (il + 1) % s.dims["full_attention_interval"] == 0:
                    attn(P, il, kvi, kv_only=last)
                    kvi += 1
                else:
                    gdn(P, il, state_only=last)
                if not last:
                    ffn(P, il)
            return P

        s.plan_feed = body_plan(True)
        body = body_plan(False)
        head = Plan()
        mixin(head, "output_norm.weight")
        gemv(head, ["output.weight"], s.xp, [s.y])
        head.add(K.k_argmax, (1,), (1024,), (s.y, i32(s.R_head), s.aout))
        s.plan_body, s.plan_head = body, head
        drv = _drv()
        c = ctypes.c_void_p()
        cp.cuda.Device(s.card).synchronize()
        drv.cuCtxGetCurrent(ctypes.byref(c))
        s.ctxh = c
        s.events = []
        for _ in range(8):
            e = ctypes.c_void_p()
            drv.cuEventCreate(ctypes.byref(e), 2 | _bsync_ev())   # CU_EVENT_DISABLE_TIMING (+ BLOCKING_SYNC)
            s.events.append(e)


# ====================================================================================== the engine

class TPEngine(object):
    """Same interface as Engine (reset, feed, step, generate) over two cards."""

    def __init__(self, pack, ctx=24576, clusters=None, log=print):
        from .sched import blocking_sync
        blocking_sync(log)
        t0 = time.time()
        self.log = log
        self.W = int(os.environ.get("KINDI_CARDS") or min(2, rt.getDeviceCount()))
        if self.W not in (1, 2) or rt.getDeviceCount() < self.W:
            raise SystemExit("kindi: the plan engine runs on 1 or 2 cards (%d found)" % rt.getDeviceCount())
        man = self.man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        self.pack = pack
        self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))
        self.D, self.NL = self.dims["width"], self.dims["layers"]
        self.NHg, self.NHKVg, self.HD = self.dims["heads"], self.dims["kv_heads"], self.dims["head_dim"]
        self.NKg, self.HK = self.dims["groups"], self.dims["state_size"]
        self.NVg = self.dims["gdn_inner"] // self.HK
        self.NFFg = self.dims["ffn"]
        self.inv_freq = np.float64(self.dims["rope_base"]) ** (-2.0 * np.arange(self.dims["rope_dim"] // 2) / self.dims["rope_dim"])
        if self.W == 1:
            self.ffn_rows, self.ffn_cols, self.head_split = [(0, self.NFFg)], [(0, self.NFFg)], [0, VOCAB]
        else:
            for x in (self.NKg, self.NVg, self.NHKVg):
                if x % 2:
                    raise SystemExit("kindi: dimensions do not split over two cards")
            # FFN: gate/up in whole Hadamard blocks (an odd middle block on both cards), down columns split in half
            nb = self.NFFg // self.HB
            half = self.NFFg // 2
            if half % 512:
                raise SystemExit("kindi: the FFN width does not split over two cards")
            self.ffn_rows = [(0, (nb + 1) // 2 * self.HB), (nb // 2 * self.HB, self.NFFg)]
            self.ffn_cols = [(0, half), (half, self.NFFg)]
            if os.environ.get("KINDI_TC", "0") == "1":
                # the tc plans (spec.py) take whole 1024-column units: the odd middle block goes to card 0
                m = (nb + 1) // 2 * self.HB
                self.ffn_rows = self.ffn_cols = [(0, m), (m, self.NFFg)]
                self.split_tag = "-b"
            self.head_split = [0, VOCAB // 2, VOCAB]
        self.idx = json.load(open(os.path.join(pack, "index.json")))
        self.bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        self.tied_head = "output.weight" not in self.bi   # tied embeddings (2B, 4B): the head is token_embd
        if self.tied_head:
            self.bi["output.weight"] = self.bi["token_embd.weight"]
        self.bcm = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
        self.dpm = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.uint16, mode="r")
        self.flm = np.memmap(os.path.join(pack, "f32.bin"), dtype=np.float32, mode="r")
        self.signs = {w: np.fromfile(os.path.join(pack, "signs_%d.f32" % w), dtype=np.float32)
                      for w in (self.D, self.NVg * self.HK, self.NHg * self.HD, self.NFFg)}
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        self._host_buffers()
        self._embedding()
        self.cards = [Shard(self, c, log) for c in range(self.W)]
        # the context: what the card with less free memory can hold
        from . import chunk as _chunk
        _chunk.reserve(self)                       # the chunked prompt path's buffers, before the context fit
        reserve = int(os.environ.get("KINDI_MEM_RESERVE_MB", "32")) * 2 ** 20
        fit = min((s.free_bytes() - reserve) // s.per_pos_bytes() for s in self.cards) // 256 * 256
        self.ctx = int(min(int(ctx), fit))
        if self.ctx < 1024:
            raise SystemExit("kindi: not enough GPU memory for a 1024-token context")
        if self.ctx < int(ctx):
            log("context reduced from %d to %d tokens to fit the GPU memory" % (int(ctx), self.ctx))
        for s in self.cards:
            s.alloc_ctx(self.ctx)
        for s in self.cards:
            with cp.cuda.Device(s.card):
                s.build_plans()
        self.NB = int(os.environ.get("KINDI_BATCH", "4"))          # prompt tokens per batch (1: off; at most 6)
        if 1 < self.NB <= 6:
            for s in self.cards:
                with cp.cuda.Device(s.card):        # kernel handles belong to the device that is current
                    s.plan_batch = build_batch_plan(s, self.NB)
            self.batch_ok = True
        _chunk.build(self)
        self.ylog_n, self.ylog_map = VOCAB, None
        self._state_views()
        log("plan engine ready: %d card%s, ctx %d, K/V q8_0, weights %s (%.1fs)" % (self.W, "s" if self.W > 1 else "", self.ctx, ("5-trit lane" if self.cards[0].g5l else "5-trit") if self.cards[0].g5 else "2-bit", time.time() - t0))

    def _host_buffers(self):
        flags = rt.hostAllocMapped | rt.hostAllocPortable
        # [1 + 8 slots][2 parities][2 cards][D] floats: slot 0 for one token, slots 1.. for the tokens of a batch
        self.xmem = cp.cuda.PinnedMemory(9 * 4 * 4 * self.D, flags=flags)
        self.fmem = cp.cuda.PinnedMemory(64, flags=flags)
        np.ctypeslib.as_array((ctypes.c_int32 * 16).from_address(self.fmem.ptr))[:] = 0
        self.xbuf_ptr, self.flag_ptr = self.xmem.ptr, self.fmem.ptr        # UVA: the host pointer is valid on both cards

    def _embedding(self):
        """token_embd in the GEMV layout, in mapped pinned host memory (k_deq_emb3 reads one row per token)."""
        rec = self.bi["token_embd.weight"]
        R, C = rec["rows"], rec["ne0"]
        ns, P = C // 4, C // 128
        bc = np.array(self.bcm[rec["bc_off"]:rec["bc_off"] + ns * R])
        dp = np.array(self.dpm[rec["dpm_off"] // 2:rec["dpm_off"] // 2 + P * R])
        layout.repack({"e": {"bc_off": 0, "dpm_off": 0, "rows": R, "ne0": C}}, bc, dp)
        flags = rt.hostAllocMapped | rt.hostAllocPortable
        self.emem_b = cp.cuda.PinnedMemory(bc.nbytes, flags=flags)
        self.emem_d = cp.cuda.PinnedMemory(dp.nbytes, flags=flags)
        np.ctypeslib.as_array((ctypes.c_uint8 * bc.nbytes).from_address(self.emem_b.ptr))[:] = bc
        np.ctypeslib.as_array((ctypes.c_uint16 * dp.size).from_address(self.emem_d.ptr))[:] = dp
        self.emb = {"bc": self.emem_b.ptr, "dp": self.emem_d.ptr}

    # ------------------------------------------------------------------ driver
    def reset(self):
        for s in self.cards:
            s.reset()

    def _run(self, tok, pos, head):
        """Replay one token on both cards, segment by segment, alternating cards (both halves of an exchange are
        launched before either card's next segment). Every 8 segments a card records an event, and the host waits
        for the event of 16 segments back: the host never fills a card's launch queue while that card waits in an
        exchange whose other half is not launched yet."""
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        drv = _drv()
        L = drv.cuLaunchKernel
        plans = []
        for s in self.cards:
            if head:
                s.plan_body.set(tok, pos)
                s.plan_head.set(tok, pos)
                segs = s.plan_body.segs + s.plan_head.segs
            else:
                s.plan_feed.set(tok, pos)                  # a prompt token: the last layer only stores its K/V
                segs = s.plan_feed.segs
            plans.append(segs)
        ring = ([], [])
        prev = ctypes.c_void_p()
        drv.cuCtxGetCurrent(ctypes.byref(prev))          # restored at the end: CuPy follows the current context
        for i in range(len(plans[0])):
            for c, s in enumerate(self.cards):
                drv.cuCtxSetCurrent(s.ctxh)
                if i % 8 == 0 and i >= 16:
                    drv.cuEventSynchronize(ring[c].pop(0))
                for fn, g, b, arr, _ in plans[c][i]:
                    if L(fn, g[0], g[1], g[2], b[0], b[1], b[2], 0, None, arr, None):
                        raise RuntimeError("kindi: a kernel launch failed on card %d" % s.card)
                if i % 8 == 7:
                    e = s.events[(i // 8) % 8]
                    drv.cuEventRecord(e, None)
                    ring[c].append(e)
        drv.cuCtxSetCurrent(prev)

    def _check(self):
        for s in self.cards:
            with cp.cuda.Device(s.card):
                if int(s.fail[0]):
                    raise RuntimeError("kindi: card %d timed out in the exchange (epoch %d)" % (s.card, int(s.fail[0])))

    def feed(self, tok, pos):
        self._run(tok, pos, head=False)

    def step(self, tok, pos):
        self._run(tok, pos, head=True)
        best = None
        for s in self.cards:
            with cp.cuda.Device(s.card):
                i = int(s.aout[0])
                v = float(s.y[i])
            gi = self.head_split[s.card] + i
            if best is None or v > best[0]:
                best = (v, gi)
        self._check()
        self._best, self._ylog_ok = best[1], False
        return best[1]

    def generate(self, ids, n):
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        pos = len(ids)
        out = [self.step(ids[-1], pos - 1)]
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out


# ====================================================================================== the cache's view of the state
# kindi.cache saves and restores eng.M, eng.conv, eng.Kc and eng.Vc (checkpoints, parked conversations, saved
# openings) with .nbytes, .set(host), cp.asnumpy(x), x[:p] and x[:p] = values. Dist gives the same operations over the
# two cards' halves: the host copy is the parts' raw bytes side by side (per position for the K/V cache).

class Dist(object):
    def __init__(self, parts, rows=False, p=None):
        self.parts, self.rows, self.p = parts, rows, p       # parts: [(card, cupy array)]

    @property
    def nbytes(self):
        return sum(a.nbytes for _, a in self.parts)

    def __getitem__(self, sl):
        if not self.rows or not isinstance(sl, slice) or sl.start not in (None, 0) or sl.step not in (None, 1):
            raise IndexError("Dist supports [:p] on the K/V cache only")
        return Dist(self.parts, True, sl.stop)

    def __array__(self, dtype=None):
        out = []
        for card, a in self.parts:
            with cp.cuda.Device(card):
                if self.rows:
                    p = a.shape[0] if self.p is None else self.p
                    out.append(cp.asnumpy(a[:p]).reshape(p, -1).view(np.uint8))
                else:
                    out.append(cp.asnumpy(a).reshape(-1).view(np.uint8))
        return np.concatenate(out, axis=1 if self.rows else 0)

    def set(self, h):
        h = np.ascontiguousarray(cp.asnumpy(h) if isinstance(h, cp.ndarray) else h).reshape(-1).view(np.uint8)
        o = 0
        for card, a in self.parts:
            with cp.cuda.Device(card):
                a.set(h[o:o + a.nbytes].view(a.dtype).reshape(a.shape))
            o += a.nbytes

    def __setitem__(self, sl, val):
        if not self.rows or not isinstance(sl, slice) or sl.start not in (None, 0):
            raise IndexError("Dist supports [:p] = values on the K/V cache only")
        v = cp.asnumpy(val) if isinstance(val, cp.ndarray) else np.asarray(val)
        p = sl.stop
        v = np.ascontiguousarray(v).reshape(p, -1).view(np.uint8)
        o = 0
        for card, a in self.parts:
            w = a[0].nbytes
            with cp.cuda.Device(card):
                # in place: a full card has no room for the temporary device copy that cp.asarray makes
                a[:p].set(np.ascontiguousarray(v[:, o:o + w]).view(a.dtype).reshape((p,) + a.shape[1:]))
            o += w


def _state_views(self):
    cs = self.cards
    c0 = cs[0]
    self.M = [Dist([(s.card, s.M[i]) for s in cs]) for i in range(self.NL)]
    self.conv = [Dist([(s.card, s.conv[i]) for s in cs]) for i in range(self.NL)]
    self.Kc = [Dist([x for s in cs for x in ((s.card, s.Kq[i]), (s.card, s.Kd[i]))], rows=True) for i in range(c0.NATT)]
    self.Vc = [Dist([x for s in cs for x in ((s.card, s.Vq[i]), (s.card, s.Vd[i]))], rows=True) for i in range(c0.NATT)]
    self.hsl, self.shortlist = None, 0
    self.k = c0.k                                 # the sampler's kernels run on card 0
    with cp.cuda.Device(0):
        self._ylog = cp.zeros(VOCAB, dtype=cp.float32)
        self.aout = cp.zeros(1, dtype=cp.int32)
    self._ylog_ok = True


def _ylog_get(self):
    """The full logits on card 0 for the sampler (gathered only when a request samples)."""
    if not self._ylog_ok:
        c0 = self.cards[0]
        h = self.head_split[1]
        with cp.cuda.Device(0):
            self._ylog[:h] = c0.y[:c0.R_head]
        if len(self.cards) > 1:
            c1 = self.cards[1]
            with cp.cuda.Device(1):
                y1 = cp.asnumpy(c1.y[:c1.R_head])
            with cp.cuda.Device(0):
                self._ylog[h:] = cp.asarray(y1)
        with cp.cuda.Device(0):
            self.aout[0] = self._best
        self._ylog_ok = True
    return self._ylog


TPEngine._state_views = _state_views
TPEngine.ylog = property(_ylog_get, lambda self, v: None)


# ====================================================================================== prompt tokens in batches
# Prompt reading runs at the decode rate: each token pays its own 128 card-to-card exchanges (~12 ms). A batch of NB
# prompt tokens shares them: one k_xrmsN launch per exchange point swaps all NB tokens' partials (a block per token,
# in parallel). The matrix products still run once per token (two tokens' lookup tables do not fit in 48 KB of
# shared memory), and everything that carries state from token to token - the GDN conv and delta steps, and each
# token's attention after the K/V stores - runs in token order, so the results are exactly those of one-token feeds.

class DynT(Dyn):
    """The k-th token of a batch."""

    def __init__(self, k):
        Dyn.__init__(self, "tok%d" % k, 0, 1)


def _plan_set(self, tok, pos, toks=None):
    for h, d in self.dyn:
        if d.var == "pos":
            h.value = d.a + d.b * pos
        elif d.var == "tok":
            h.value = d.a + d.b * tok
        else:
            h.value = d.a + d.b * toks[int(d.var[3:])]


Plan.set = _plan_set


def build_batch_plan(s, NB):
    """A plan that feeds NB consecutive prompt tokens (positions pos .. pos+NB-1) on card s."""
    tp, K = s.tp, s.k
    D, HB, HK, HD = s.D, s.HB, s.HK, s.HD
    NK, NV, NH, NHKV, NFF = s.NK, s.NV, s.NH, s.NHKV, s.NFF
    z = lambda n, t=cp.float32: cp.zeros(n, dtype=t)  # noqa: E731
    work = max(D, NFF, NV * HK)
    with cp.cuda.Device(s.card):
        HA, HBUF, XN = z(NB * D), z(NB * D), z(NB * D)
        XP, PART, Y = z(NB * work), z(NB * s.nsm * work), z(NB * D)
        QKV, ZV = z(NB * s.CONV_CH), z(NB * NV * HK)
        QN, KRO, VB = z(NB * NH * 2 * HD), z(NB * NHKV * HD), z(NB * NHKV * HD)
        s.bctr = s.xctr                     # the batch blocks use counters 1.. of the same array
    v = lambda a, k, n: a[k * n:(k + 1) * n]  # noqa: E731
    st = {"h": HA, "h2": HBUF, "pending": 0}

    def gemv(P, names, x, outs, part):
        if s.g5:
            ret = 1
            for n, out in zip(names, outs):
                _, W5, R, C = s.w[n]
                R, NG = int(R), int(C) // 128
                nt, rpt, sp = g5_tuned(s, n, R, NG, part is not None and R == D)
                var = s.__dict__.get("g5var", {}).get((R, NG, part is not None and R == D), "r")
                f = ({0: K.k_g5n4, 1: K.k_g5r1, 2: K.k_g5r2, 4: K.k_g5r4, 8: K.k_g5r8} if var != "a" else
                     {1: K.k_g5a1, 2: K.k_g5a2, 4: K.k_g5a4, 8: K.k_g5a8})[rpt]
                gx = g5_gx(R, nt, rpt)
                if part is not None and R == D:
                    P.add(f, (gx, sp), (nt,), (W5, part, i32(R), i32(NG), x))
                    ret = sp
                elif sp == 1:
                    P.add(f, (gx, 1), (nt,), (W5, out, i32(R), i32(NG), x))
                elif var in ("s", "as"):
                    fs = ({1: K.k_g5r1s, 2: K.k_g5r2s, 4: K.k_g5r4s, 8: K.k_g5r8s} if var == "s" else
                          {1: K.k_g5a1s, 2: K.k_g5a2s, 4: K.k_g5a4s, 8: K.k_g5a8s})[rpt]
                    P.add(fs, (gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x, out, s.g5cnt))
                else:
                    P.add(f, (gx, sp), (nt,), (W5, s.gpart, i32(R), i32(NG), x))
                    P.add(K.k_sumparts, ((R + 255) // 256,), (256,), (s.gpart, out, i32(R), i32(sp)))
            return ret
        a = [s.w[n] for n in names]
        args = []
        for i in range(3):
            if i < len(a):
                args += [a[i][0], a[i][1], outs[i], a[i][2]]
            else:
                args += [a[0][0], a[0][1], outs[0], i32(0)]
        R0, C0 = int(a[0][2]), int(a[0][3])
        if len(a) == 1 and s.nsm > 1 and (R0 >> 4) < 128 * s.nsm and (C0 >> 9) >= 2:
            ng, nph = R0 >> 4, C0 >> 9
            best = None
            for gx in range(1, s.nsm + 1):
                if s.nsm % gx:
                    continue
                gy = s.nsm // gx
                if gy > nph:
                    continue
                cost = -(-(-(-ng // gx)) // 32) * -(-nph // gy)
                if best is None or cost < best[0]:
                    best = (cost, gx, gy)
            _, gx, gy = best
            if gy > 1:
                args[2] = part
            P.add(K.k_g4, (gx, gy), (1024,), tuple(args) + (x, a[0][3]))
            return gy
        P.add(K.k_g4, (s.nsm,), (1024,), tuple(args) + (x, a[0][3]))
        return 1

    def mixin(P, wname):
        if st["pending"] and tp.W == 1:
            for k in range(NB):
                if st["pending"] > 1:
                    P.add(K.k_sumparts, ((D + 255) // 256,), (256,), (v(PART, k, s.nsm * work), v(Y, k, D), i32(D),
                                                                      i32(st["pending"])))
                P.add(K.k_rmsfold, (D // HB,), (512,), (v(st["h"], k, D), v(Y, k, D), v(st["h2"], k, D), s.f(wname), s.sD,
                                                         v(XN, k, D), v(XP, k, work), i32(D), i32(1), i32(HB)))
            st["h"], st["h2"] = st["h2"], st["h"]
        elif st["pending"]:
            P.cut()
            src = PART if st["pending"] > 1 else Y
            P.add(K.k_xrmsN, (NB,), (1024,), (src, i32(s.nsm * work if st["pending"] > 1 else D), i32(st["pending"]),
                                              np.uint64(tp.xbuf_ptr), np.uint64(tp.flag_ptr), i32(s.card), s.fail,
                                              s.bctr, st["h"], st["h2"], s.f(wname), s.sD, XN, XP, i32(work), i32(D)))
            st["h"], st["h2"] = st["h2"], st["h"]
        else:
            for k in range(NB):
                P.add(K.k_rmsfold, (D // HB,), (512,), (v(st["h"], k, D), s.y, v(st["h2"], k, D), s.f(wname), s.sD,
                                                         v(XN, k, D), v(XP, k, work), i32(D), i32(0), i32(HB)))
        st["pending"] = 0

    def out(k):                             # where token k's row-parallel output goes: its part or its y
        return v(PART, k, s.nsm * work), v(Y, k, D)

    def gdn(P, il, last):
        p = "blk.%d." % il
        mixin(P, p + "attn_norm.weight")
        for k in range(NB):
            gemv(P, [p + "attn_qkv.weight", p + "attn_gate.weight"], v(XP, k, work),
                 [v(QKV, k, s.CONV_CH), v(ZV, k, NV * HK)], None)
        nblk = NV + 2 * NK + (NV * HK + 255) // 256
        pend = 0
        for k in range(NB):                 # in token order: the conv shift register and the delta state
            P.add(K.k_gdn_pre, (nblk,), (256,), (v(XN, k, D), s.f(p + "ssm_beta.weight"), s.f(p + "ssm_alpha.weight"),
                                                 s.f(p + "ssm_dt.bias"), s.f(p + "ssm_a"), s.gb, s.gg, i32(D),
                                                 v(QKV, k, s.CONV_CH), s.conv[il], s.f(p + "ssm_conv1d.weight"), s.cs,
                                                 s.qkbuf, i32(s.CONV_CH), i32(NV), i32(NK), i32(HK)))
            P.add(K.k_gdn_deltan, (NV,), (HK,), (s.M[il], s.qkbuf[:NK * HK], s.qkbuf[NK * HK:], s.cs[2 * NK * HK:],
                                                 s.gg, s.gb, s.f(p + "ssm_norm.weight"), s.outgdn,
                                                 f32t(1.0 / np.sqrt(HK)), i32(NK)))
            if last:
                continue
            P.add(K.k_smfold, (NV * HK // HB,), (512,), (v(ZV, k, NV * HK), s.outgdn, s.sV, v(XP, k, work), i32(HK),
                                                          i32(NK), i32(NV // NK), i32(HB)))
            pt, yt = out(k)
            pend = gemv(P, [p + "ssm_out.weight"], v(XP, k, work), [yt], pt)
        st["pending"] = pend

    def attn(P, il, kvi, last):
        p = "blk.%d." % il
        mixin(P, p + "attn_norm.weight")
        for k in range(NB):
            if last:
                gemv(P, [p + "attn_k.weight", p + "attn_v.weight"], v(XP, k, work),
                     [v(KRO, k, NHKV * HD), v(VB, k, NHKV * HD)], None)
            else:
                gemv(P, [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], v(XP, k, work),
                     [v(QN, k, NH * 2 * HD), v(KRO, k, NHKV * HD), v(VB, k, NHKV * HD)], None)
        rd = s.cos_t.shape[1] * 4
        nb0 = NHKV * HD // 32
        pend = 0
        for k in range(NB):                 # in token order: token k sees the K/V of positions <= pos + k
            kr = v(KRO, k, NHKV * HD)
            P.add(K.k_rmsw, ((NHKV + 7) // 8,), (256,), (kr, s.f(p + "attn_k_norm.weight"), s.kn, i32(HD), i32(NHKV)))
            if not last:
                P.add(K.k_rmsw_s, ((NH + 7) // 8,), (256,), (v(QN, k, NH * 2 * HD), i32(2 * HD), s.f(p + "attn_q_norm.weight"),
                                                             s.fo, i32(HD), i32(NH)))
                P.add(K.k_rope2, (NH + NHKV,), (256,), (s.fo, s.kn, Dyn("pos", s.cos_t.data.ptr + k * rd, rd, ptr=True),
                                                        Dyn("pos", s.sin_t.data.ptr + k * rd, rd, ptr=True), s.qr, s.kro,
                                                        i32(NH), i32(HD)))
            else:
                P.add(K.k_rope2, (NHKV,), (256,), (s.kn, s.kn, Dyn("pos", s.cos_t.data.ptr + k * rd, rd, ptr=True),
                                                   Dyn("pos", s.sin_t.data.ptr + k * rd, rd, ptr=True), s.kro, s.kro,
                                                   i32(0), i32(HD)))
            P.add(K.k_store_q8, ((64 * nb0 + 1023) // 1024,), (min(1024, 64 * nb0),), (s.kro, v(VB, k, NHKV * HD),
                                                    Dyn("pos", s.Kq[kvi].data.ptr + k * NHKV * HD, NHKV * HD, ptr=True),
                                                    Dyn("pos", s.Kd[kvi].data.ptr + k * nb0 * 2, nb0 * 2, ptr=True),
                                                    Dyn("pos", s.Vq[kvi].data.ptr + k * NHKV * HD, NHKV * HD, ptr=True),
                                                    Dyn("pos", s.Vd[kvi].data.ptr + k * nb0 * 2, nb0 * 2, ptr=True), i32(nb0)))
            if last:
                continue
            n = lambda: Dyn("pos", 1 + k, 1)  # noqa: E731
            P.add(K.k_q8row, ((NH * HD + 1023) // 1024,), (1024,), (s.qr, s.qq, s.qd, i32(NH * HD // 32)))
            P.add(K.k_attn_sq8d if s.dp4a else K.k_attn_sq8t, (NHKV, (s.ctx + 255) // 256), (256,), (s.Kq[kvi], s.Kd[kvi], s.qq, s.qd, s.sbuf, n(),
                                                                       i32(NHKV), i32(s.G)))
            P.add(K.k_softmax_rows, (NH,), (256,), (s.sbuf, n()))
            P.add(K.k_attn_vq8w, (NHKV, 1, (s.ctx + 255) // 256), (128,), (s.sbuf, s.Vq[kvi], s.Vd[kvi], s.apart, n(),
                                                                          i32(NH), i32(NHKV), i32(s.G)))
            P.add(K.k_attn_combn, (NH,), (256,), (s.apart, v(QN, k, NH * 2 * HD), s.att, n(), i32(NH)))
            P.add(K.k_fold_fwht, (NH * HD // HB,), (512,), (s.att, s.sA, v(XP, k, work), i32(HB)))
            pt, yt = out(k)
            pend = gemv(P, [p + "attn_output.weight"], v(XP, k, work), [yt], pt)
        st["pending"] = pend

    def ffn(P, il):
        p = "blk.%d." % il
        mixin(P, p + "post_attention_norm.weight")
        pend = 0
        for k in range(NB):
            gemv(P, [p + "ffn_gate.weight", p + "ffn_up.weight"], v(XP, k, work), [s.ffg, s.ffu], None)
            P.add(K.k_smfold, (NFF // HB,), (512,), (s.ffg, s.ffu, s.sF, s.fo6, i32(0), i32(0), i32(0), i32(HB)))
            pt, yt = out(k)
            pend = gemv(P, [p + "ffn_down.weight"], s.fo6[s.d0 - s.g0:], [yt], pt)
        st["pending"] = pend

    P = Plan()
    emb = tp.emb
    for k in range(NB):
        P.add(K.k_deq_emb4, (D // HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, v(HA, k, D),
                                                 DynT(k), i32(D), i32(HB)))
    kvi = 0
    for il in range(s.NL):
        last = il == s.NL - 1
        if (il + 1) % s.dims["full_attention_interval"] == 0:
            attn(P, il, kvi, last)
            kvi += 1
        else:
            gdn(P, il, last)
        if not last:
            ffn(P, il)
    return P


def _feed_batch(self, toks, pos):
    """Feed len(toks) consecutive prompt tokens (positions pos ..): batches of self.NB, the rest one by one."""
    from . import chunk as _chunk
    i = _chunk.feed(self, toks, pos)               # whole chunks first (KINDI_CHUNK=0: off)
    while i < len(toks):
        if len(toks) - i >= self.NB and getattr(self, "batch_ok", False):
            if pos + i + self.NB > self.ctx:
                raise OverflowError("position %d is past the context (%d)" % (pos + i + self.NB - 1, self.ctx))
            drv = _drv()
            L = drv.cuLaunchKernel
            plans = []
            for s in self.cards:
                s.plan_batch.set(toks[i], pos + i, toks[i:i + self.NB])
                plans.append(s.plan_batch.segs)
            prev = ctypes.c_void_p()
            drv.cuCtxGetCurrent(ctypes.byref(prev))
            ring = ([], [])
            for j in range(len(plans[0])):
                for c, s in enumerate(self.cards):
                    drv.cuCtxSetCurrent(s.ctxh)
                    if j % 8 == 0 and j >= 16:
                        drv.cuEventSynchronize(ring[c].pop(0))
                    for fn, g, b, arr, _ in plans[c][j]:
                        if L(fn, g[0], g[1], g[2], b[0], b[1], b[2], 0, None, arr, None):
                            raise RuntimeError("kindi: a kernel launch failed on card %d" % s.card)
                    if j % 8 == 7:
                        e = s.events[(j // 8) % 8]
                        drv.cuEventRecord(e, None)
                        ring[c].append(e)
            drv.cuCtxSetCurrent(prev)
            i += self.NB
        else:
            self.feed(toks[i], pos + i)
            i += 1


TPEngine.feed_batch = _feed_batch
