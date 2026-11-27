"""Qwen3.8-Flash-Next (qwen4exp) on cards: the layer split over one or more CUDA cards, the hot routed experts on the
cards and the other experts on the CPU (kindi/qx_cpu.c), with the card kernels of kindi/qx_gpu.cu.

This is the step-3 engine of docs/dev/FLASHNEXT-GPU.md: the kernels are Kindi's own (qx_gpu.cu); PyTorch (CUDA) is
the glue for memory, the small element-wise steps, attention and the causal convolutions. It follows
kindi/qwen4exp_ref.py step by step and keeps the same cached state, so the two can be compared layer by layer.

Routed experts: a card computes the (token, slot) pairs of the experts it holds, the CPU the others, and the value
rows combine in slot order. Both use the same integer-dot arithmetic, so a value row has the same bits on either
side, and the result does not depend on how the experts are placed.

Placement: layers are split over the cards by bytes; each card then takes the most used experts of its own layers
(kindi/moe.py plan() with the routing counts of kindi/qwen4exp-experts.json) into the VRAM left after its dense
weights and a reserve."""
import ctypes
import json
import math
import time
import os

# the caching allocator in expandable segments (KINDI_QXG_EXPSEG, 1; before torch makes its first card allocation):
# the layer-major prompt's large, short-lived buffers otherwise leave reserved but unusable blocks behind, and the
# drafted long read peaked above the plain one (B62, 3060: 11.39 against 11.07 GB). Rented RTX 3080 test machine: the drafted read
# peak 9.53 -> 9.31 GiB, reads and bits unchanged. A PYTORCH_CUDA_ALLOC_CONF set by the caller is kept.
if os.environ.get("KINDI_QXG_EXPSEG", "1") == "1":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

from kindi import moe
from kindi import qwen4exp_quant as Q
from kindi import qwen4exp_ref as R

HERE = os.path.dirname(os.path.abspath(__file__))
# the prompt's dense GEMMs: tf32 (default; B4 gate 174/174), fp32,
PGEMM = os.environ.get("KINDI_QXG_PGEMM", "tf32")
# or fp16 (weights and inputs rounded to fp16, fp32 accumulation and output, cuBLAS GemmEx: about 1.5-2x TF32)
# prompt chunks of at least this many tokens use dequantize + GEMM (KINDI_QXG_GEMM_T; 0 = never); matrices of up to
# this many values (the output head is bigger and runs on the last row only)
GEMM_MIN_T = int(os.environ.get("KINDI_QXG_GEMM_T", "32")) or 1 << 30
GEMM_MAX_VALUES = 64 << 20
# KINDI_QXG_PFUSE=1: the prompt's gated-residual mixers and residual writes on the fused T-row kernels (k_rms_gT,
# k_silu_s, k_mix_outT, k_writeT: the window's kernels) instead of PyTorch's element-wise ops; changes the prompt's
# bits (as TF32 does), so the gate decides
PFUSE = os.environ.get("KINDI_QXG_PFUSE", "0") == "1"
QCHUNK = int(os.environ.get("KINDI_QXG_QCHUNK", "64"))    # queries per attention step of a prompt chunk (B24: 64
# gives 256's bits and speed with a quarter of the scores' memory)


# Limit each prompt attention GEMM to keys visible to its last query. Different GEMM shapes
# may change rounding. Keep this opt-in until the matched prompt and token checks pass.
CAUSAL_PREFILL = os.environ.get("KINDI_QXG_CAUSAL_PREFILL", "0") == "1"

# The 0.8 prompt-read profile of the streamed prefill (one sm_80+ card with the experts page-locked: the RTX 3060
# replays in docs/dev/bench-0.8.0/MATRIX.md). Model._g_defaults turns it on for those cards (KINDI_QXG_PROFILE=0: off);
# every variable set in the environment still wins. Scheduling and upload items (the same bits):
PROFILE_BASE = {"KINDI_QXG_COMPACT_STREAM": "1", "KINDI_QXG_STREAM_UPLOAD": "packed-pipeline", "KINDI_QXG_RING": "64",
                "KINDI_QXG_PREFETCH_PHASES": "1", "KINDI_MTP_STAGE_PROMPT": "1"}
# by prompt length and read (plain / drafted with the MTP head); the expert batch is then limited by the free memory
PROFILE_PLAIN = ({"KINDI_QXG_LM_TOK": "4096", "KINDI_QXG_PREFETCH_PREFIX": "400", "KINDI_QXG_PREFETCH_ORDER": "usage",
                  "KINDI_QXG_PREFETCH_SLICE": "0"},
                 {"KINDI_QXG_LM_TOK": "16384", "KINDI_QXG_PREFETCH_PREFIX": "512", "KINDI_QXG_PREFETCH_ORDER": "index",
                  "KINDI_QXG_PREFETCH_SLICE": "8192", "KINDI_QXG_PREFETCH_COMPUTE_GROUP": "128",
                  "KINDI_QXG_TENSOR_QSA": "fp16", "KINDI_QXG_DEQ_REUSE": "1"})
PROFILE_DRAFT = ({"KINDI_QXG_LM_TOK_DRAFT": "4096", "KINDI_QXG_PREFETCH_PREFIX": "320",
                  "KINDI_QXG_PREFETCH_ORDER": "usage", "KINDI_QXG_PREFETCH_SLICE": "0",
                  "KINDI_QXG_PREFETCH_COMPUTE_GROUP": "128", "KINDI_QXG_TENSOR_QSA": "fp16"},
                 {"KINDI_QXG_LM_TOK_DRAFT": "8192", "KINDI_QXG_PREFETCH_PREFIX": "400",
                  "KINDI_QXG_PREFETCH_ORDER": "index", "KINDI_QXG_PREFETCH_SLICE": "4096",
                  "KINDI_QXG_PREFETCH_COMPUTE_GROUP": "128", "KINDI_QXG_TENSOR_QSA": "fp16"})
PROFILE_LONG = 8192          # prompts above this many tokens use the long-prompt profile
_PROFILE = {}                # the items in force now (empty: the profile is off)


def env(key, default=None):
    """a setting: the environment's value, else the prompt-read profile's, else default"""
    v = os.environ.get(key)
    if v is not None:
        return v
    return _PROFILE.get(key, default)


class _CopyAttributes(ctypes.Structure):
    # CUDA CUmemcpyAttributes_v1 with two CUmemLocation fields.
    _fields_ = [("source_order", ctypes.c_int), ("source_location", ctypes.c_int * 2),
                ("destination_location", ctypes.c_int * 2), ("flags", ctypes.c_uint)]


def _submit_weight_copies(cu, stream, origins, destinations, sizes):
    batch = getattr(cu, "cuMemcpyBatchAsync_v2", None)
    if batch is not None:
        p = ctypes.c_void_p
        batch.argtypes = [p, p, p, ctypes.c_size_t, ctypes.POINTER(_CopyAttributes), p, ctypes.c_size_t, p]
        batch.restype = ctypes.c_int
        # CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        attr = _CopyAttributes(source_order=1)
        attr_offset = ctypes.c_size_t(0)
        rc = batch(destinations.ctypes.data, origins.ctypes.data, sizes.ctypes.data, origins.size,
                   ctypes.byref(attr), ctypes.byref(attr_offset), 1, stream)
        if rc != 0:
            raise RuntimeError(f"cuMemcpyBatchAsync_v2 failed ({rc})")
        return
    for d, s, size in zip(destinations.flat, origins.flat, sizes.flat):
        rc = cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(int(d)), ctypes.c_void_p(int(s)), ctypes.c_size_t(int(size)), stream)
        if rc != 0:
            raise RuntimeError(f"cuMemcpyHtoDAsync_v2 failed ({rc})")


def _fast_config():
    """the fast mode (lossy, ON by default since 11-22, Amir: defaults give speed unless a high quality risk; kbench
    B27 62/65 fast = 62/65 exact): KINDI_FAST=0 is the exact path. KINDI_FAST=1 (default) turns on the items marked (on); each item's own variable
    wins (0 leaves it out; set alone it works without KINDI_FAST). The exact path is untouched when off.
    - KINDI_FAST_TAIL=R (on, 7): decode drops a CPU expert pick at router rank >= R (0-based; not on the card, not
      fetched): not computed, its value row zero, the router weights not renormalised. B21-B22 on the 3060: decode
      +7%; teacher-forced top-1 agreement 0.961 on 48 chats, but only 12 of 48 greedy answers the same over 64 tokens
    - KINDI_FAST_ATTN=tf32 (on) | fp16 (opt-in only): the prompt's attention products in TF32 (16K read +3.6%), or
      fp16 inputs with an fp32 softmax (+2.9%)
    Measured and left out (B22): fp16 dense prompt GEMMs (slower on the 3060), attention over the union of a query
    block's QSA selections (+1.5%: the union is nearly every key; a per-query sparse kernel is a 0.8 item)."""
    on = os.environ.get("KINDI_FAST", "1") == "1"
    item = lambda k, d: os.environ.get(k) or (d if on else "0")  # noqa: E731
    f = {"tail": int(item("KINDI_FAST_TAIL", "7")), "attn": item("KINDI_FAST_ATTN", "tf32")}
    f["attn"] = f["attn"] if f["attn"] in ("tf32", "fp16") else ""
    return f


FAST = _fast_config()


def fast_summary():
    """one line for the start-up log: which fast items are on"""
    it = (["tail skip at rank >= %d" % FAST["tail"]] if FAST["tail"] > 0 else []) + \
        (["%s prompt attention" % FAST["attn"]] if FAST["attn"] else [])
    return "fast mode ON (lossy): " + ", ".join(it) if it else "fast mode off (exact)"


H = R.H
_lib = None


def hotset_path(model_path):
    """where the adaptive cache keeps its decode set between sessions: KINDI_QXG_HOTSET (a path, or 0: off), else
    beside the CPU kernels' cache, one file per model file"""
    v = os.environ.get("KINDI_QXG_HOTSET", "")
    if v == "0":
        return None
    if v:
        return v
    d = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    try:
        size = os.path.getsize(model_path)
    except OSError:
        return None
    return os.path.join(d, "qwen4exp-hot-%s-%d.json" % (os.path.basename(model_path), size))


def load_hotset(path, n_exp):
    """{layer: [experts]} from a saved decode set, or None"""
    if not path or not os.path.exists(path):
        return None
    try:
        import json
        d = json.load(open(path))
        return {int(l): [int(e) for e in es if 0 <= int(e) < n_exp] for l, es in d["hot"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def lib():
    """the card kernels (kindi/qx_gpu.cu through kindi/qx_gpu.py: NVRTC + the driver API)"""
    global _lib
    if _lib is None:
        from kindi.qx_gpu import Kernels
        _lib = Kernels(os.environ.get("KINDI_QXG_ARCH") or None)
    return _lib


def xtiles(pstart, bn=64):
    """the tile list of the tiled expert GEMM (k_xtile_*): int32 [n, 4] rows (expert, first pair, end pair, 0) of at
    most bn pairs, experts in order, and tstart (expert u's tiles are tstart[u] .. tstart[u + 1] - 1)"""
    pstart = np.asarray(pstart, np.int64)
    cnt = np.diff(pstart)
    nt = (cnt + bn - 1) // bn
    tstart = np.concatenate([[0], np.cumsum(nt)]).astype(np.int64)
    u = np.repeat(np.arange(len(cnt)), nt)
    p0 = pstart[u] + (np.arange(int(tstart[-1])) - tstart[u]) * bn
    p1 = np.minimum(p0 + bn, pstart[u + 1])
    return np.stack([u, p0, p1, np.zeros_like(u)], 1).astype(np.int32), tstart


def _xmm(L, d_, st, ty, cols, rows, ab, part, mats_d, pstart_d, ptok_d, pslot_d, n, in_pair, xq, xs, out, ldo, xt,
         u0, u1):
    """the prompt's experts u0 .. u1 - 1 on int8 tensor cores (mats_d, pstart_d: of expert 0): the tiled GEMM when
    xt = (tiles on the card, tstart) is given and it takes the matrix, else k_xmm_tc; the same bits either way"""
    if u1 <= u0:
        return
    if xt is not None:
        a, b = int(xt[1][u0]), int(xt[1][u1])
        if L.xtile_ok(ty, cols, rows, b - a):
            L.xmm_tile(d_, st, ty, cols, rows, ab, _p(mats_d), part, 0, _p(xt[0]) + 16 * a, b - a, _p(ptok_d),
                       _p(pslot_d), n, in_pair, xq, xs, out, ldo)
            return
    L.xmm_tc(d_, st, ty, cols, rows, ab, u1 - u0, _p(mats_d) + 24 * u0, part, _p(pstart_d) + 4 * u0, _p(ptok_d),
             _p(pslot_d), n, in_pair, xq, xs, out, ldo)


def _upload(dv, *arrays):
    """numpy arrays to card dv in one copy: packed (16-byte aligned) into one page-locked staging tensor from
    PyTorch's caching host allocator (kept until the copy is done), one asynchronous copy on the current stream;
    returns card views with the arrays' dtypes and shapes"""
    offs, n = [], 0
    for a in arrays:
        offs.append(n)
        n += (a.nbytes + 15) // 16 * 16
    host = torch.empty(max(n, 16), dtype=torch.uint8, pin_memory=True)
    hb = host.numpy()
    for a, o in zip(arrays, offs):
        hb[o:o + a.nbytes] = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
    card = host.to(dv, non_blocking=True)
    tdt = {np.dtype(np.int64): torch.int64, np.dtype(np.int32): torch.int32}
    return [card[o:o + a.nbytes].view(tdt[a.dtype]).view(*a.shape) for a, o in zip(arrays, offs)]


def _st(t):
    """(card, PyTorch stream) of a tensor"""
    return t.device.index, torch.cuda.current_stream(t.device).cuda_stream


def _p(t):
    return t.data_ptr()


class RemoteExperts(object):
    """the CPU experts of another process (tools/qx_cpu_serve.py), e.g. in WSL when the cards run on Windows:
    the same forward_parts() as kindi.qx_cpu.Experts, over a TCP socket"""

    def __init__(self, addr):
        import socket
        host, port = addr.rsplit(":", 1)
        self.s = socket.create_connection((host, int(port)))
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.K = H
        self.sec = 0.0

    def _recv(self, n):
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            k = self.s.recv_into(mv[got:], n - got)
            if not k:
                raise ConnectionError("qx_cpu_serve closed the connection")
            got += k
        return buf

    def forward_parts(self, l, x, ids, skip, parts):
        import struct
        import time
        t0 = time.time()
        T, n = ids.shape
        x = np.ascontiguousarray(x, np.float32)
        hdr = struct.pack("<4i", l, T, n, x.shape[1])
        self.s.sendall(hdr + x.tobytes() + np.ascontiguousarray(ids, np.int32).tobytes() +
                       np.ascontiguousarray(skip, np.uint8).tobytes())
        r = struct.unpack("<i", self._recv(4))[0]
        if r < 0:
            raise ValueError("qx_cpu_serve: layer %d failed (%d)" % (l, r))
        rows = np.frombuffer(self._recv(4 * T * n * x.shape[1]), np.float32).reshape(T, n, -1)
        cold = ~skip.astype(bool)
        parts[cold] = rows[cold]
        self.sec += time.time() - t0
        return r

    def forward_sum(self, l, x, ids, skip, w):
        """the CPU's slots of layer l weighted and summed per token -> [T, K] (a tenth of forward_parts' bytes; the
        order of the sum differs from the card's combine: for routing traces, not for exact decoding)"""
        import struct
        import time
        t0 = time.time()
        T, n = ids.shape
        x = np.ascontiguousarray(x, np.float32)
        hdr = struct.pack("<4i", -l - 1, T, n, x.shape[1])
        self.s.sendall(hdr + x.tobytes() + np.ascontiguousarray(ids, np.int32).tobytes() +
                       np.ascontiguousarray(skip, np.uint8).tobytes() + np.ascontiguousarray(w, np.float32).tobytes())
        r = struct.unpack("<i", self._recv(4))[0]
        if r < 0:
            raise ValueError("qx_cpu_serve: layer %d failed (%d)" % (l, r))
        out = np.frombuffer(self._recv(4 * T * x.shape[1]), np.float32).reshape(T, -1)
        self.sec += time.time() - t0
        return out

    def stats(self):
        return {"seconds": self.sec, "bytes": 0}


def rms(x, w, eps, group=None):
    if group is not None:
        s = x.shape
        x = x.reshape(*s[:-1], -1, group)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return x.reshape(s) * w
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class Model(object):
    """devices: CUDA device numbers; the layers are split over them in this order (the output head on the last).
    expert_gb: VRAM for hot experts on each card (None: what is free after the dense weights and reserve_gb).
    cpu_experts: the cores of the CPU expert pool ("" with every expert on the cards).
    mtp: the MTP head file (kindi/qwen4exp_mtp_import.py) or None; its layer and all its experts go on the last card."""

    def __init__(self, path, devices=(0,), expert_gb=None, reserve_gb=1.2, usage=None, cpu_experts=None,
                 split=None, log=print, mtp=None):
        # the MTP head: the given file, else KINDI_MTP_FILE, else next to the model the compact head
        # (qwen4exp-mtp-q2.gguf, Q2_0 experts, tools/qwen4exp_mtp_compact.py; KINDI_MTP_COMPACT=0 skips it) or
        # qwen4exp-mtp.gguf; KINDI_MTP=0 leaves it out
        if os.environ.get("KINDI_MTP", "1") == "0":
            mtp = None
        elif mtp is None:
            here = os.path.dirname(os.path.abspath(path))
            names = ["qwen4exp-mtp.gguf"]
            if os.environ.get("KINDI_MTP_COMPACT", "1") != "0":
                names.insert(0, "qwen4exp-mtp-q2.gguf")
            near = os.environ.get("KINDI_MTP_FILE") or next(
                (os.path.join(here, n) for n in names if os.path.exists(os.path.join(here, n))), "")
            mtp = near if near and os.path.exists(near) else None
        self.ref = R.Model(path, mtp=mtp)        # the config, the file's tensors, and the host-side helpers
        self.ref.gdn_heads = "tile"
        m = self.ref
        W = m.w
        self.devices = [torch.device("cuda", d) for d in devices]
        self.L = m.L
        self.log = log
        self._willneed()
        if cpu_experts is None:                  # one thread per physical core (kindi.cores)
            from .cores import default_cpu_list
            cpu_experts = default_cpu_list()
        self._g_defaults(cpu_experts)
        lib()
        # layer split by dense bytes (the experts are placed after it)
        dense = {l: sum(self._nbytes(n) for n in W.t if n.startswith("blk.%d." % l) and "_exps." not in n)
                 for l in range(self.L)}
        if split is None:
            n = len(self.devices)
            tot, acc, split, cur = sum(dense.values()), 0, [], 0
            for l in range(self.L):
                acc += dense[l]
                split.append(cur)
                if cur < n - 1 and acc >= tot * (cur + 1) / n:
                    cur += 1
        self.dev_of = list(split)
        # dense weights to the cards: raw blocks for the matrices, fp32 for the vectors
        self.mat, self.vec = {}, {}
        for name in W.t:
            if "_exps." in name or name in ("token_embd.weight", "per_layer_token_embd.weight"):
                continue
            d = self._dev_of_name(name)
            mm, base, ne, ty = W.t[name]
            if len(ne) == 2 and ne[1] > 1 and not name.endswith("conv1d.weight"):
                raw = np.frombuffer(mm, np.uint8, self._nbytes(name), base)
                self.mat[name] = (torch.from_numpy(raw.copy()).to(d), ty, ne[1], ne[0])
            else:
                self.vec[name] = W.get(name).float().to(d)
        torch.cuda.synchronize()
        self.mtp = bool(mtp)
        lib().tc_drafting = self.mtp            # KINDI_QXG_TC=auto: with the head, every decode product on tensor cores
        self.cache_of = {}                     # layer -> its card's CardCache
        self.slot = {}                         # layer -> int32 array [n_exp]: slot on the card or -1
        self.hot = {}                          # layer -> uint8 tensor [n_hot][gate|up|down]
        self.hot_count = {}
        if self.mtp:                           # the head's 512 experts (IQ4_NL 1.41 GB, Q2_0 0.71 GB), on the last card
            # KINDI_MTP_HEAD_N (default: all): only the head's N most-used experts on the card, by the counts in
            # KINDI_MTP_USAGE (a JSON list of n_exp counts, written by a run with KINDI_MTP_USAGE_OUT; without it the
            # first N); a pick of an expert not on the card adds nothing to the draft (drafts only: the verifying
            # pass is the main model's, so the output stays exact; acceptance may drop)
            hn = int(os.environ.get("KINDI_MTP_HEAD_N", "0") or 0)
            hexp = list(range(m.n_exp))
            if 0 < hn < m.n_exp:
                uf = os.environ.get("KINDI_MTP_USAGE")
                cnt = np.asarray(json.load(open(uf)), np.int64) if uf else np.zeros(m.n_exp, np.int64)
                hexp = sorted(int(e) for e in np.argsort(-cnt, kind="stable")[:hn])
            self._load_hot(self.L, hexp, self.devices[-1])
            self.head_counts = None
            if os.environ.get("KINDI_MTP_USAGE_OUT"):
                # every pick of every head step (drafts and cache rows alike), written to the file at exit (made
                # here, before any graph capture, so the graphs keep a fixed buffer)
                import atexit
                self.head_counts = torch.zeros(m.n_exp, dtype=torch.int32, device=self.devices[-1])
                out_path = os.environ["KINDI_MTP_USAGE_OUT"]
                atexit.register(lambda: json.dump(self.head_counts.cpu().tolist(), open(out_path, "w")))
            self.log("MTP head: layer %d on %s, %d experts %s (%.2f GB)" % (
                self.L, self.devices[-1], m.n_exp, Q.NAMES.get(W.t["blk.%d.ffn_gate_exps.weight" % self.L][3]),
                self.hot[self.L].numel() / 1e9))
            self._load_draft_vocab(W)
        # hot experts
        eb = {}
        for l in range(self.L):
            eb[l] = sum(self._nbytes("blk.%d.ffn_%s_exps.weight" % (l, k)) for k in ("gate", "up", "down")) // m.n_exp
        self.eb = eb
        spec = moe.MoESpec(range(self.L), m.n_exp, m.n_used, eb)
        # the seed profile: the given one, else KINDI_QXG_USAGE (a routing trace's OUT.json), else the shipped one
        use = moe.Usage.load(spec, usage or os.environ.get("KINDI_QXG_USAGE")
                             or os.path.join(HERE, "qwen4exp-experts.json"))
        self._prefetch_priority = {
            l: np.argsort(-np.asarray(counts, dtype=np.float64), kind="stable") for l, counts in use.counts.items()
        }
        # the last session's decode set (written by the adaptive cache, kindi/qwen4exp_cache.py save_hot): those
        # experts first, so a fresh start decodes like a warmed-up one (B23: card hits 0.10 -> 0.38 over the first
        # ~100 tokens of a cold start; the first run's decode 14.6-18.3 against 19.9-20.1). Placement only: no bits.
        self.hotset_path = hotset_path(path)
        saved = load_hotset(self.hotset_path, m.n_exp)
        if saved and use.has_counts():
            top = max(max(c) for c in use.counts.values()) + 1.0
            for l, es in saved.items():
                if l in use.counts:
                    for e in es:
                        use.counts[l][e] += top
            self.log("card cache seeded from the last session's decode set (%d experts, %s)" % (
                sum(len(v) for v in saved.values()), self.hotset_path))
        for di, dv in enumerate(self.devices):
            layers = [l for l in range(self.L) if self.dev_of[l] == di]
            if expert_gb is not None:
                room = expert_gb[di] * 1e9 if isinstance(expert_gb, (list, tuple)) else expert_gb * 1e9
            else:
                # what is free after the dense weights, less what grows with the context on this card: the
                # decoder's caches, the prompt state and one prompt chunk's attention scores (at the decode cache
                # length KINDI_QXG_LMAX), and the fixed reserve (CUDA context, graphs, work buffers)
                free, _ = torch.cuda.mem_get_info(dv)
                # B12 (11-16, 3060): decode is not limited by the card's expert cache when the CPU experts are fast
                # (no cache at all decodes at the same speed), and a cache 2.1 GB smaller than the free memory decodes
                # 9% faster (the card's own expert work is on the token's critical path). That holds where the CPU
                # pool's measured rate is a fair share of the card's bandwidth: _cache_cal decides, the old reserve
                # otherwise; KINDI_QXG_RESERVE_GB sets the whole reserve
                cal_gb = 0.0 if os.environ.get("KINDI_QXG_RESERVE_GB") else self._cache_cal(dv, path, cpu_experts)
                reserve_gb = float(os.environ.get("KINDI_QXG_RESERVE_GB") or (reserve_gb + cal_gb))
                self._draft_bytes = 0.0
                room = free - reserve_gb * 1e9 - self._context_bytes(layers)
                # the drafter's decode-time buffers (window states, value rows, margin) come out of the calibration's
                # reserve first, only the rest out of the expert cache; they and the layer-major prefill's buffers
                # are never held at once, so the cache gives up the larger of the two rests, not their sum
                draft_take = max(0.0, self._draft_bytes - cal_gb * 1e9)
                if self._draft_bytes:
                    self.log("drafting buffers: %.2f GB, %.2f GB of it from the calibration's reserve" % (
                        self._draft_bytes / 1e9, min(self._draft_bytes, cal_gb * 1e9) / 1e9))
                lm_take = 0.0
                lmb = 0.0
                if self.layer_major in ("auto", True):
                    Fd = m.w.t["blk.0.ffn_gate_exps.weight"][2][1]
                    lmb = self._lm_bytes(getattr(self, "prompt_chunk", 1024), Fd,
                                         int(os.environ.get("KINDI_QXG_LMAX", "8192")))
                    # the calibration's reserve (memory the decode does not want for experts, B12) holds the
                    # layer-major prefill's memory first: only the rest comes out of the expert cache (B23, 3060:
                    # layer-major forced took it all from the cache, card hits 0.38 -> 0.01, decode 20 -> 17.5)
                    take = max(0.0, lmb - cal_gb * 1e9)
                    if self.layer_major is True:         # forced: _context_bytes counted all of it already
                        room += lmb
                    if self.layer_major == "auto":
                        # on where its memory leaves the experts a real share of the card (KINDI_QXG_LM_MIN_HOT_GB,
                        # 3), or comes out of the calibration's reserve alone
                        self.layer_major = take == 0 or room - take >= float(
                            os.environ.get("KINDI_QXG_LM_MIN_HOT_GB", "3")) * 1e9
                    if self.layer_major:
                        lm_take = take
                    self.log("layer-major prompt %s: %.2f GB it needs, %.2f GB of it from the calibration's reserve; "
                             "%.2f GB for the experts" % ("on" if self.layer_major else "off (not enough card memory)",
                                                          lmb / 1e9, min(lmb, cal_gb * 1e9) / 1e9,
                                                          (room - max(lm_take, draft_take)) / 1e9))
                room -= max(lm_take, draft_take)
            room = max(0.0, room)
            sp = moe.MoESpec(layers, m.n_exp, m.n_used, {l: eb[l] for l in layers})
            u = moe.Usage(sp, {l: use.counts[l] for l in layers}) if use.has_counts() else moe.Usage(sp)
            pl = moe.plan(sp, u, room, min_card=0)
            place = {l: pl.on(l, "card") for l in layers}
            # KINDI_QXG_EXT (auto: on with the MTP head, on one card with a measured plan, Linux; 1 also without the
            # head, 0 off): the elastic tail. Without drafting a bigger cache decodes slower on the 3060 (B28 R3: 17.4
            # against 19.3; B55: plain decode after a long read 14 tok/s with the tails). What the prompt's
            # buffers take (the stream's chunk buffers, the scores, layer-major's) sits idle while the decode runs
            # (B26: peak 8.1 of 12 GB); that much (less the drafter's window buffers and a margin,
            # KINDI_QXG_EXT_MARGIN_GB 0.25; KINDI_QXG_EXT_GB sets it) of the plan's next experts gets slots in tails of
            # the class buffers that have card memory only between prompt reads (CardCache.ext_on / ext_off)
            ext = None
            ext_mode = os.environ.get("KINDI_QXG_EXT", "auto")
            if (expert_gb is None and len(self.devices) == 1 and (ext_mode == "1" or ext_mode == "auto" and self.mtp)
                    and os.name != "nt"):
                xb = float(os.environ.get("KINDI_QXG_EXT_GB", "0") or 0) * 1e9
                self._ext_keep = None
                if not xb:
                    # rc7: sized from the free card memory when the decode starts (CardCache.ext_on): the plan's next
                    # experts up to KINDI_QXG_EXT_MAX_GB (4) get tail slots (address space only), and each decode start
                    # backs as many as leave KINDI_QXG_EXT_KEEP_GB (1.5) of the card free (B64-B65, 3060: a 2.5-3 GB
                    # tail, 34.2-34.5 tok/s against 31.6 with the 1.35 GB of the reserve rule)
                    xb = float(os.environ.get("KINDI_QXG_EXT_MAX_GB", "4")) * 1e9
                    self._ext_keep = float(os.environ.get("KINDI_QXG_EXT_KEEP_GB", "1.5")) * 1e9
                self.log("elastic cache: up to %.2f GB of tail slots, %s" % (
                    xb / 1e9, "backed while %.2f GB of the card stays free" % (self._ext_keep / 1e9)
                    if self._ext_keep is not None else "all backed (KINDI_QXG_EXT_GB)"))
                if xb > 0:
                    pl2 = moe.plan(sp, u, room + xb, min_card=0)
                    ext = {l: [e for e in pl2.on(l, "card") if e not in set(place[l])] for l in layers}
                    self.log("elastic cache: %.2f GB of tail slots (card memory only during the decode)" % (xb / 1e9))
            from kindi.qwen4exp_cache import CardCache
            cc = CardCache(self, dv, layers, place, ext=ext)     # slots by expert size; the slot tables and buffers
            cc.ext_keep = getattr(self, "_ext_keep", None)
            if os.environ.get("KINDI_QXG_ADAPT", "0") == "1" and not cc.start_adaptive():
                self.log("KINDI_QXG_ADAPT=1: the adaptive cache needs the qx_cpu library (Linux); static cache")
            for l in layers:
                self.cache_of[l] = cc
            n_hot = sum(len(place[l]) for l in layers)
            self.log("card %s: layers %d-%d, dense %.2f GB, %d hot experts (%.2f GB)" % (
                dv, layers[0], layers[-1], sum(dense[l] for l in layers) / 1e9, n_hot,
                sum(len(place[l]) * eb[l] for l in layers) / 1e9))
        if self.layer_major == "auto":             # a given expert_gb: no room was measured, so chunked
            self.layer_major = False
        self.all_on_cards = all(self.hot_count[l] == m.n_exp for l in range(self.L))
        # drafting pays only while the head leaves enough of the card to the main model's experts (B4, 3060: the
        # head and its reserves halved the card's share of the picks and drafting lost): draft_ok when the hot set
        # is at least KINDI_MTP_MIN_HOT_GB (0 since B38: drafting pays with the head's small cache); logged
        hot_gb = sum(self.hot_count[l] * eb[l] for l in range(self.L)) / 1e9
        self.draft_ok = bool(self.mtp) and hot_gb >= float(os.environ.get("KINDI_MTP_MIN_HOT_GB", "0"))
        if self.mtp:
            self.log("drafting %s: %.2f GB of experts on the cards beside the head (KINDI_MTP_MIN_HOT_GB %s)" % (
                "pays (draft_ok)" if self.draft_ok else "would not pay (draft_ok false)", hot_gb,
                os.environ.get("KINDI_MTP_MIN_HOT_GB", "0")))
        self.cpu = getattr(self, "cpu", None)              # made by the cache calibration, or below
        if not self.all_on_cards and self.cpu is None:
            if cpu_experts.startswith("tcp:"):
                self.cpu = RemoteExperts(cpu_experts[4:])
            else:
                from kindi.qx_cpu import Experts
                self.cpu = Experts(path, cpu_experts)
        self.times = {"cpu_experts": 0.0}

    # ---------------------------------------------------------------- setup helpers
    def _cache_cal(self, dv, path, cpu_experts):
        """the extra reserve (GB) that shrinks the card's expert cache: 2.1 when the CPU experts' rate (one token, ten
        picks of real experts already in memory, the pool as configured) is at least KINDI_QXG_CAL_RATIO (0.05) of the
        card's memory bandwidth (B12: the 3060 host, about 30 of 334 GB/s), else 0 (the cache as large as it fits).
        The decision is kept per host, card, CPU set and model in ~/.cache/kindi/qwen4exp-cal.json and reused at
        every start (11-17: measuring at every start flipped it: a cold page cache made the CPU look disk-bound);
        a new measurement only replaces it when it lands clearly on the other side (below half or above twice the
        threshold). Logged. KINDI_QXG_CAL=0: no measurement, 0; KINDI_QXG_CAL=new: measure and store again."""
        mode = os.environ.get("KINDI_QXG_CAL", "1")
        if mode == "0" or cpu_experts.startswith("tcp:"):
            return 0.0
        if "extra_gb" in getattr(self, "cache_cal", {}):    # measured once (several cards)
            return self.cache_cal["extra_gb"]
        import json
        import socket
        import time
        ratio = float(os.environ.get("KINDI_QXG_CAL_RATIO", "0.05"))
        store = os.path.join(os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache",
                                                                                      "kindi"), "qwen4exp-cal.json")
        key = "%s|%s|%s|%s|%d" % (socket.gethostname(), torch.cuda.get_device_name(dv), cpu_experts,
                                  os.path.basename(path), os.path.getsize(path))
        try:
            known = json.load(open(store))
        except (OSError, ValueError):
            known = {}
        prev = known.get(key) if mode != "new" else None
        m = self.ref
        try:
            if getattr(self, "cpu", None) is None:          # the CPU experts, made now (the plan needs their rate)
                from kindi.qx_cpu import Experts
                self.cpu = Experts(path, cpu_experts)
            rng = np.random.default_rng(0)
            K = H
            ts, nb = [], 0
            for l in (1, m.L // 2, m.L - 2):
                eb = sum(self._nbytes("blk.%d.ffn_%s_exps.weight" % (l, k)) for k in ("gate", "up", "down")) // m.n_exp
                x = rng.standard_normal((1, K)).astype(np.float32)
                ids = rng.choice(m.n_exp, m.n_used, replace=False).astype(np.int32)[None]
                for rep in range(5):                         # the same experts: the first call reads their pages
                    parts = np.zeros((1, m.n_used, K), np.float32)    # in, the rest time the CPU's own rate
                    t = time.perf_counter()
                    self.cpu.forward_parts(l, x, ids, np.zeros((1, m.n_used), np.uint8), parts)
                    dt = time.perf_counter() - t
                    if rep:
                        ts.append(dt)
                        nb += m.n_used * eb
            cpu_gbs = nb / sorted(ts)[len(ts) // 2] / len(ts) / 1e9      # the median call
            a = torch.empty(64 << 20, dtype=torch.float32, device=dv)
            b = torch.empty_like(a)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            b.copy_(a)
            e0.record()
            for _ in range(5):
                b.copy_(a)
            e1.record()
            e1.synchronize()
            card_gbs = 5 * 2 * a.numel() * 4 / (e0.elapsed_time(e1) / 1e3) / 1e9
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:                               # the old reserve
            self.log("cache calibration failed (%r): the cache as large as it fits" % (e,))
            return float(prev["extra_gb"]) if prev else 0.0
        r = cpu_gbs / card_gbs
        if prev is not None and (prev["extra_gb"] > 0) != (r >= ratio) and ratio / 2 <= r <= ratio * 2:
            extra, how = float(prev["extra_gb"]), "kept from %s (this run is near the threshold)" % prev["when"]
        else:
            extra, how = (2.1 if r >= ratio else 0.0), "measured"
            known[key] = {"extra_gb": extra, "ratio": round(r, 4), "when": time.strftime("%Y-%m-%d %H:%M")}
            try:
                os.makedirs(os.path.dirname(store), exist_ok=True)
                with open(store + ".tmp", "w") as f:
                    json.dump(known, f, indent=1)
                os.replace(store + ".tmp", store)
            except OSError:
                pass
        self.log("cache calibration: CPU experts %.1f GB/s, card %.0f GB/s (ratio %.3f, threshold %.3f), %s: %s" % (
            cpu_gbs, card_gbs, r, ratio, how,
            "the cache 2.1 GB smaller (decode is not cache-bound here)" if extra else "the cache as large as it fits"))
        self.cache_cal = {"cpu_gbs": round(cpu_gbs, 2), "card_gbs": round(card_gbs, 1), "extra_gb": extra}
        return extra

    def _g_defaults(self, cpu_experts):
        """phase G's defaults (B3: the DN 64 gate 174/174, prompt 63 -> 169 tok/s), where the cards support them:
        - KINDI_QX_DN (the down input's blocks): 64 when every card is sm_80+ and the CPU experts are in this process
          (a remote CPU server reads its own environment, so it stays 32 there unless set on both sides); set into
          the environment so the CPU library uses the same blocks (card rows == CPU rows needs both)
        - KINDI_QXG_TCX (the prompt's experts on int8 tensor cores): auto = sm_80+
        - KINDI_QXG_STREAM (the prompt's experts all on the card): auto = sm_80+, one card, in-process CPU experts and
          the experts page-locked (KINDI_QXG_PIN=1: it takes about 50 GB of locked RAM, so the host decides)
        - prompt_chunk: the chunk the callers should feed (4096 with the stream, else 256)
        Any of them set explicitly wins."""
        sm80 = all(torch.cuda.get_device_capability(d)[0] >= 8 for d in self.devices)
        local = not cpu_experts.startswith("tcp:")
        dn = int(os.environ.get("KINDI_QX_DN") or (64 if sm80 and local else 32))
        os.environ["KINDI_QX_DN"] = str(dn)
        lib().DN = dn
        try:
            from kindi import qx_cpu
            if qx_cpu._lib is not None:                # already loaded: set its copy of the switch too
                ctypes.c_int.in_dll(qx_cpu._lib, "qx_dn").value = dn
        except Exception:
            pass
        tcx = os.environ.get("KINDI_QXG_TCX", "auto")
        self.tcx = sm80 if tcx == "auto" else tcx == "1"
        stream = os.environ.get("KINDI_QXG_STREAM", "auto")
        self.stream = (sm80 and local and len(self.devices) == 1 and os.environ.get("KINDI_QXG_PIN", "0") == "1")             if stream == "auto" else stream == "1"
        # 1024 with the stream: as fast as 4096 on the 3060 (B3: 131 against 126 tok/s at 4K), and its chunk buffers
        # (reserved in the expert plan) are a quarter, so about 0.8 GB more of the card holds experts while decoding
        self.prompt_chunk = int(os.environ.get("KINDI_QXG_CHUNK") or (1024 if self.stream else 256))
        # the prompt layer by layer (each expert streamed once per KINDI_QXG_LM_TOK tokens, not once per chunk):
        # KINDI_QXG_LAYER_MAJOR: auto (default with the stream: on when the card memory allows, decided in the
        # expert plan), 1, 0
        lm = os.environ.get("KINDI_QXG_LAYER_MAJOR", "auto")
        self.layer_major = (False if not self.stream else "auto") if lm == "auto" else lm == "1"
        # the 0.8 prompt-read profile (PROFILE_*): on with the stream (KINDI_QXG_PROFILE=0: off). With the fast mode
        # on (the default) it also takes the fp16 prompt GEMMs, the fused prompt kernels and the causal key prefix
        # (KINDI_QXG_PGEMM, KINDI_QXG_PFUSE, KINDI_QXG_CAUSAL_PREFILL; they change the prompt's rounding, as TF32 does)
        global PGEMM, PFUSE, CAUSAL_PREFILL
        self.profile = bool(self.stream) and os.environ.get("KINDI_QXG_PROFILE", "1") != "0"
        if self.profile:
            _PROFILE.update(PROFILE_BASE)
            if os.environ.get("KINDI_FAST", "1") == "1":
                PGEMM = os.environ.get("KINDI_QXG_PGEMM", "fp16")
                PFUSE = os.environ.get("KINDI_QXG_PFUSE", "1") == "1"
                CAUSAL_PREFILL = os.environ.get("KINDI_QXG_CAUSAL_PREFILL", "1") == "1"
        self.log("phase G: down blocks %d, prompt experts on tensor cores %s, streamed %s, prompt chunk %d, "
                 "prompt-read profile %s" % (dn, "on" if self.tcx else "off", "on" if self.stream else "off",
                                             self.prompt_chunk, "on (%s prompt GEMMs)" % PGEMM if self.profile else "off"))
        self.log(fast_summary())

    def _willneed(self):
        """the token and per-layer token embedding tables are read row by row on the host for every token (about 17
        random rows a token from a 29 GB table): on a spinning disk each row not in the page cache is a seek (B3:
        about 110 ms of every draft window). Ask the kernel to read them in the background now: KINDI_QXG_WILLNEED
        auto (default: only when the available RAM covers them with a margin), 1 (always), 0 (never)."""
        mode = os.environ.get("KINDI_QXG_WILLNEED", "auto")
        if mode == "0":
            return
        import mmap as _mm
        if not hasattr(_mm, "MADV_WILLNEED"):
            return
        tabs = []
        for name in ("per_layer_token_embd.weight", "token_embd.weight"):
            if name in self.ref.w.t:
                mm, base, ne, ty = self.ref.w.t[name]
                tabs.append((mm, base, Q.nbytes(ty, int(np.prod(ne)))))
        need = sum(n for _, _, n in tabs)
        avail = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
        except OSError:
            pass
        # auto: only when the RAM that is available covers the tables with room to spare (16 GB + a quarter of the
        # tables); otherwise the read-ahead would push out pages worth more than these rows
        if mode == "auto" and avail < need * 1.25 + 16e9:
            self.log("embedding tables: %.1f GB, %.1f GB of RAM available: no read-ahead (KINDI_QXG_WILLNEED=1 forces it)"
                     % (need / 1e9, avail / 1e9))
            return
        tot = 0
        for mm, base, n in tabs:
            lo = base // _mm.PAGESIZE * _mm.PAGESIZE
            try:
                mm.madvise(_mm.MADV_WILLNEED, lo, min(len(mm) - lo, n + base - lo))
                tot += n
            except (OSError, ValueError):
                pass
        if tot:
            self.log("embedding tables: %.1f GB asked into the page cache (read ahead in the background; %.0f GB of RAM "
                     "available)" % (tot / 1e9, avail / 1e9))
        # KINDI_QXG_PLE_LOCK (auto: when the RAM covers them after the experts' page lock): the tables' pages locked (read
        # in, then never evicted), on a background thread. B30 (3060 host, spinning disk): with DeepSeek's page
        # cache beside it, 7 of the 29 GB per-layer table had been evicted, and decode on new text ran at 14.3 tok/s
        # against 19.3 on text whose rows were cached. Same bytes: no bits change.
        lock = os.environ.get("KINDI_QXG_PLE_LOCK", "auto")
        if lock == "0":
            return
        if lock == "auto":
            # what stays available once the experts are page-locked too (KINDI_QXG_PIN=1: all of them), with the same
            # headroom as the read-ahead: smaller hosts keep the read-ahead only
            pin = 0
            if os.environ.get("KINDI_QXG_PIN", "0") == "1":
                pin = sum(Q.nbytes(ty, int(np.prod(ne))) for name, (mm_, b_, ne, ty) in self.ref.w.t.items()
                          if "_exps." in name)
            if avail - pin < need * 1.25 + 16e9:
                self.log("embedding tables: not locked in RAM (%.0f GB available, %.0f GB of it for the experts' "
                         "page lock; KINDI_QXG_PLE_LOCK=1 forces it)" % (avail / 1e9, pin / 1e9))
                return
        import threading

        dv0 = self.devices[0]

        def run():
            torch.cuda.set_device(dv0)                   # the primary context current on this thread
            torch.cuda.synchronize(dv0)
            cu, done, t0 = lib().cu, 0, time.time()
            for mm, base, n in tabs:
                a0 = np.frombuffer(mm, np.uint8).ctypes.data
                lo = base // _mm.PAGESIZE * _mm.PAGESIZE
                hi = min(len(mm), (base + n + _mm.PAGESIZE - 1) // _mm.PAGESIZE * _mm.PAGESIZE)
                r = cu.cuMemHostRegister_v2(ctypes.c_void_p(a0 + lo), ctypes.c_size_t(hi - lo), 0x08 | 0x02)
                if r == 0:
                    done += hi - lo
                else:
                    self.log("embedding tables: page-locking failed (CUDA error %d); read ahead only" % r)
            if done:
                self.log("embedding tables: %.1f GB locked in RAM in %.0f s (KINDI_QXG_PLE_LOCK)" % (
                    done / 1e9, time.time() - t0))
        self._ple_lock = threading.Thread(target=run, daemon=True)
        self._ple_lock.start()

    def _lm_bytes(self, T, Fd, lmax):
        """the layer-major prefill's extra card memory: the experts of KINDI_QXG_LM_TOK tokens at once (beyond one
        chunk's), and every prompt row's streams (4 x H fp32) and its shared-expert rows kept through the layers"""
        m = self.ref
        Tl = max(T, int(os.environ.get("KINDI_QXG_LM_TOK", "4096")))
        return (Tl - T) * m.n_used * (Fd * 13 + H * 4) + lmax * (m.hc * H * 4 + 2 * H * 4)

    def _lm_fit(self, lm_tok, pc, floor):
        """the profile's expert batch, halved until its extra buffers fit the card's free memory now (the elastic
        tails are released for a prompt read) with 1 GB to spare; never below floor (the batch the plan reserves)"""
        m, dv = self.ref, self.devices[0]
        Fd = m.w.t["blk.0.ffn_gate_exps.weight"][2][1]
        free = torch.cuda.mem_get_info(dv)[0] + torch.cuda.memory_reserved(dv) - torch.cuda.memory_allocated(dv)
        row = m.n_used * (Fd * 13 + H * 4)
        while lm_tok > floor and (lm_tok - floor) * row > free - (1 << 30):
            lm_tok = max(floor, lm_tok // 2 // pc * pc)
        return lm_tok

    def _context_bytes(self, layers, chunk=256):
        """VRAM that grows with the context for these layers: per attention layer the decoder's K/V and indexer
        caches and the prompt's K/V state (both lmax positions), the scores of one prompt chunk; per Gated
        DeltaNet layer its state"""
        m = self.ref
        lmax = int(os.environ.get("KINDI_QXG_LMAX", "8192"))
        att = [l for l in layers if (l + 1) % m.interval == 0]
        per_att = 2 * (2 * lmax * m.nkv * m.hd * 4 + lmax * 128 * 4 * 1.25)
        # one query sub-chunk's scores over the whole context, three times: the scores, the masked copy and the softmax
        # (B5: the 16K prompt ran out of card memory in attn once the stream's reserve shrank)
        scores = 3 * m.nh * min(chunk, QCHUNK) * lmax * 4 if att else 0
        gdn = (len(layers) - len(att)) * m.ssm_heads * m.ssm_state * m.ssm_state * 4
        extra = 0
        if getattr(self, "mtp", None) and self.L in getattr(self, "hot", {}):
            # drafting: the head's own attention caches (layer L, not in `layers`), and the window buffers made at
            # the first window (k + 1 <= 5 tokens: the delta-rule state after each token, the value rows of every
            # pick, the conv states): these come after the expert plan, so they are reserved here (B2: the 16K
            # drafted row ran out of card memory)
            # the window buffers: for windows of the drafter's k + 1 tokens (KINDI_FN_DRAFT_K, 2), at most 5
            tw, nsel = min(5, int(os.environ.get("KINDI_FN_DRAFT_K", "2")) + 1), m.n_used
            extra = per_att
            # B3: the 16K drafted row was still 312 MB short with the above; cc2 (memprobe.py) puts the drafter's
            # growth over the plain path at about 0.27 GB a doubling of the prompt: a measured margin.
            # B25 (3060): the window buffers and this margin, counted on top of the calibration's reserve, left the
            # main model no experts (card hits 0.37 -> 0.01). They are decode-time memory, made after the prompt's
            # (layer-major) buffers are released, so they live in the calibration's reserve like those: kept apart
            # here (self._draft_bytes) and placed by the plan
            self._draft_bytes = (tw * gdn + 2 * tw * nsel * H * 4 * len(layers) + 64e6
                                 + float(os.environ.get("KINDI_MTP_RESERVE_GB", "0.6")) * 1e9)
        if getattr(self, "stream", False):
            # the prompt stream (phase G) works on whole chunks of prompt_chunk tokens: per pick the gate, up and h rows
            # (fp32), h quantized, and the value rows, plus the chunk's QSA mask over the context and a margin (B4:
            # the 16K rows ran out of card memory in the stream's buffers at chunk 4096)
            T = getattr(self, "prompt_chunk", 4096)
            Fd = m.w.t["blk.0.ffn_gate_exps.weight"][2][1]
            extra += T * m.n_used * (Fd * 13 + H * 4) + T * lmax + 3e8
            self._prompt_bytes = T * m.n_used * (Fd * 13 + H * 4) + T * lmax + scores
            if getattr(self, "layer_major", False) is True:
                extra += self._lm_bytes(T, Fd, lmax)
        return len(att) * per_att + scores + gdn + extra


    def _nbytes(self, name):
        mm, base, ne, ty = self.ref.w.t[name]
        return Q.nbytes(ty, int(np.prod(ne)))

    def _dev_of_name(self, name):
        if name.startswith("blk.") and int(name.split(".")[1]) < self.L:
            return self.devices[self.dev_of[int(name.split(".")[1])]]
        return self.devices[-1]                # output head and output mixer, the MTP head

    def _load_draft_vocab(self, W):
        """the drafter's reduced output head: the first KINDI_MTP_VOCAB_N (32768; 0: none) tokens of the draft
        vocabulary (kindi/qwen4exp-draft-vocab.json, by count in our own corpus, tools/qwen4exp_draft_vocab.py), as
        the matching rows of output.weight ("mtp.draft_out") and of token_embd (draft_emb, raw), plus the map from a
        row to its token. The draft chain then runs on the card without the host (GraphDecoder.draft_chain)."""
        self.draft_map = None
        n = int(os.environ.get("KINDI_MTP_VOCAB_N", "32768"))
        path = os.environ.get("KINDI_MTP_VOCAB") or os.path.join(HERE, "qwen4exp-draft-vocab.json")
        if n <= 0 or not os.path.exists(path):
            return
        if path.endswith(".npy"):
            ids = np.load(path)[:n].astype(np.int64)
        else:
            with open(path, encoding="utf-8") as f:
                ids = np.asarray(json.load(f)["ids"][:n], np.int64)
        dv = self.devices[-1]
        w, ty, rows, cols = self.mat["output.weight"]
        rb = Q.nbytes(ty, cols)
        sel = torch.from_numpy(ids).to(w.device)
        self.mat["mtp.draft_out"] = (w.view(rows, rb)[sel].contiguous().view(-1).to(dv), ty, len(ids), cols)
        mm, base, ne, ety = W.t["token_embd.weight"]
        erb = Q.nbytes(ety, ne[0])
        allr = np.frombuffer(mm, np.uint8, ne[1] * erb, base).reshape(ne[1], erb)
        self.draft_emb = (torch.from_numpy(allr[ids]).to(dv).view(-1), ety, ne[0], erb)
        self.draft_map = torch.from_numpy(ids.astype(np.int32)).to(dv)
        self.log("MTP draft vocabulary: %d tokens (%.0f MB on %s)" % (
            len(ids), (self.mat["mtp.draft_out"][0].numel() + self.draft_emb[0].numel()) / 1e6, dv))

    def _load_hot(self, l, experts, dv):
        W = self.ref.w
        slot = np.full(self.ref.n_exp, -1, np.int32)
        if experts:
            parts = []
            for k in ("gate", "up", "down"):
                mm, base, ne, ty = W.t["blk.%d.ffn_%s_exps.weight" % (l, k)]
                parts.append((mm, base, self._nbytes("blk.%d.ffn_%s_exps.weight" % (l, k)) // ne[2]))
            buf = torch.empty((len(experts), sum(p[2] for p in parts)), dtype=torch.uint8, device=dv)
            for s, e in enumerate(experts):
                row = np.concatenate([np.frombuffer(mm, np.uint8, n, base + e * n) for mm, base, n in parts])
                buf[s].copy_(torch.from_numpy(row))
                slot[e] = s
            self.hot[l] = buf
        self.slot[l] = slot
        self.hot_count[l] = len(experts)

    # ---------------------------------------------------------------- pieces
    def mv(self, name, x):
        """x [T, cols] fp32 on the tensor's card -> [T, rows]"""
        w, ty, rows, cols = self.mat[name]
        x = x.contiguous()
        dv, st = _st(w)
        if x.shape[0] >= GEMM_MIN_T and rows * cols <= GEMM_MAX_VALUES:
            # a prompt chunk: the matrix once as fp32 (fp16) into a scratch buffer, then a GEMM. In the layer-major
            # prompt (_logits_lm) the layer's matrices may stay dequantized for all its chunks (self._dq: the same
            # values, so the same bits; only the dequantization is not repeated)
            half = PGEMM == "fp16"
            dq = getattr(self, "_dq", None)
            wd = dq.get(name) if dq is not None else None
            if wd is None:
                if dq is not None:
                    wd = torch.empty((rows, cols), dtype=torch.float16 if half else torch.float32, device=w.device)
                    dq[name] = wd
                else:
                    scr = self.__dict__.setdefault("_scratch16" if half else "_scratch", {})
                    buf = scr.get(w.device)
                    if buf is None or buf.numel() < rows * cols:
                        buf = scr[w.device] = torch.empty(max(rows * cols, 1 << 24), device=w.device,
                                                          dtype=torch.float16 if half else torch.float32)
                    wd = buf[:rows * cols].view(rows, cols)
                (lib().deq16 if half else lib().deq)(dv, st, ty, _p(w), rows, cols, Q.nbytes(ty, cols), _p(wd))
            if half:
                # the same input often goes to several matrices in a row (q/k/v, the mixer's down and inject, the
                # router and the shared expert): its fp16 copy is made once
                key = (x.data_ptr(), tuple(x.shape), x._version)
                memo = getattr(self, "_xh", None)
                if memo is not None and memo[0] == key:
                    xh = memo[1]
                else:
                    xh = x.half()
                    self._xh = (key, xh, x)                  # x kept alive: its address cannot be reused meanwhile
                y = torch.empty((x.shape[0], rows), dtype=torch.float32, device=w.device)
                lib().gemm16(st, _p(xh), _p(wd), _p(y), x.shape[0], rows, cols)
                return y
            if PGEMM == "tf32":
                # KINDI_QXG_PGEMM=tf32 (phase G): the prompt's GEMMs on tensor cores with TF32 inputs (fp32 in and
                # out, about 2x on Ampere); the prompt's arithmetic changes (decode does not), so the gate decides
                prev = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    return x @ wd.T
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = prev
            return x @ wd.T
        y = torch.empty((x.shape[0], rows), dtype=torch.float32, device=w.device)
        lib().dense(dv, st, ty, _p(w), rows, cols, Q.nbytes(ty, cols), _p(x), x.shape[0], _p(y))
        return y

    def mv_multi(self, names, x):
        """one token through several matrices of one card in one launch (k_dense8m); returns their outputs. The
        outputs and the descriptor are made once per name list and reused (static: the decode graphs capture them)"""
        cache = self.__dict__.setdefault("_mvm", {})
        tc = lib().tc_on(x.device.index)
        t1 = not tc and lib().t1_on(x.device.index)     # below sm_70: k_dense8mT2_1 (k_dense8m's bits, faster there)
        key = (tuple(names), x.device, tc, t1)
        if key not in cache:
            from kindi.qx_gpu import Kernels
            outs, rows8, first = [], [], 0
            for n in names:
                w, ty, rows, cols = self.mat[n]
                split, rpb = (0, 16) if tc else Kernels.planT(rows, cols, 1) if t1 else Kernels.dense8_plan(rows, cols)
                y = torch.empty((1, rows), dtype=torch.float32, device=x.device)
                outs.append(y)
                rows8 += [w.data_ptr(), y.data_ptr(), Q.nbytes(ty, cols), ty, rows, cols, split, first]
                first += (rows + rpb - 1) // rpb
            desc = torch.tensor(rows8, dtype=torch.int64).to(x.device)
            cache[key] = (outs, desc, first)
        outs, desc, nblocks = cache[key]
        if tc:
            lib().dense_tcm(*_st(x), _p(desc), len(names), nblocks, _p(x.contiguous()), 1)
        elif t1:
            lib().dense8mT(*_st(x), _p(desc), len(names), nblocks, _p(x.contiguous()), 1, x.shape[-1])
        else:
            lib().dense8m(*_st(x), _p(desc), len(names), nblocks, _p(x.contiguous()), x.shape[-1])
        return outs

    def mixer(self, h, pre, inject=True):
        m = self.ref
        T = h.shape[0]
        if PFUSE:
            return self._mixer_fused(h, pre, inject)
        hn = rms(h, self.vec[pre + "_norm.weight"], m.eps, group=H)
        x = F.silu(self.mv(pre + "_down.weight", hn) / m.hc)
        x = torch.sigmoid(self.mv(pre + "_up.weight", x)).view(T, m.hc, H)
        x = (x * hn.view(T, m.hc, H)).mean(1)
        if not inject:
            return x
        wr = 2 * torch.sigmoid(self.mv(pre + "_inject.weight", hn) / m.hc)
        return x, h, wr

    def _mixer_fused(self, h, pre, inject=True):
        """mixer() on the fused T-row kernels (KINDI_QXG_PFUSE=1)"""
        m, L = self.ref, lib()
        T, hc = h.shape[0], m.hc
        h = h.contiguous()
        d_, st = _st(h)
        hn = torch.empty_like(h)
        L.rms_gT(d_, st, _p(h), _p(self.vec[pre + "_norm.weight"]), _p(hn), hc, H, m.eps, T)
        dd = self.mv(pre + "_down.weight", hn).contiguous()
        L.silu_s(d_, st, _p(dd), dd.numel(), 1.0 / hc)
        u = self.mv(pre + "_up.weight", dd).contiguous()
        x = torch.empty((T, H), dtype=torch.float32, device=h.device)
        if not inject:
            L.mix_outT(d_, st, _p(u), _p(hn), 0, _p(x), 0, hc, H, hc, T, hc * H, hc * H, 0, H, 0)
            return x
        j = self.mv(pre + "_inject.weight", hn).contiguous()
        wr = torch.empty((T, hc), dtype=torch.float32, device=h.device)
        L.mix_outT(d_, st, _p(u), _p(hn), _p(j), _p(x), _p(wr), hc, H, hc, T, hc * H, hc * H, hc, H, hc)
        return x, h, wr

    def write(self, hin, y, wr):
        """the residual write (R.Model.write); on the fused T-row kernel with KINDI_QXG_PFUSE=1"""
        if not PFUSE:
            return R.Model.write(hin, y, wr)
        T, hc = hin.shape[0], self.ref.hc
        hin, y, wr = hin.contiguous(), y.contiguous(), wr.contiguous()
        out = torch.empty_like(hin)
        lib().writeT(*_st(hin), _p(hin), _p(y), 0, 0, _p(wr), _p(out), hc, H, T, hc * H, H, hc)
        return out

    def ple(self, l, h, ids, st):
        m = self.ref
        T = h.shape[0]
        rows = m.ple_rows(ids, st["ids"] if st else None)
        emb = self.emb_rows(np.asarray(rows).reshape(-1), h.device, "per_layer_token_embd.weight").view(T, -1)
        p = "blk.%d.ple_" % l
        key = rms(self.mv(p + "key.weight", emb), self.vec[p + "norm_key.weight"], m.eps, group=H).view(T, m.hc, H)
        val = self.mv(p + "value.weight", emb)
        qry = rms(h, self.vec[p + "norm_query.weight"], m.eps, group=H).view(T, m.hc, H)
        gate = (key * qry).sum(-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = (torch.sigmoid(gate) * val[:, None, :]).reshape(T, -1)
        gvn = rms(gv, self.vec[p + "norm_conv.weight"], m.eps, group=H)
        k, dil = m.ple_kernel, m.ple_n
        cw = self.vec[p + "conv1d.weight"]
        pad = (k - 1) * dil
        prev = st["ple"].get(l) if st else None
        x = torch.cat([torch.zeros(pad, gvn.shape[1], device=h.device) if prev is None else prev, gvn])
        if st is not None:
            st["ple"][l] = x[-pad:].clone()
        y = F.silu(F.conv1d(x.T[None], cw[:, None, :], groups=cw.shape[0], dilation=dil))[0].T
        return gv + y

    def gdn(self, l, x, st):
        m = self.ref
        p = "blk.%d." % l
        T = x.shape[0]
        nk, nv, dk = m.ssm_groups, m.ssm_heads, m.ssm_state
        dv = self.vec[p + "ssm_norm.weight"].numel()
        qkv = self.mv(p + "attn_qkv.weight", x)
        z = self.mv(p + "attn_gate.weight", x)
        b = self.mv(p + "ssm_beta.weight", x)
        a = self.mv(p + "ssm_alpha.weight", x)
        cw = self.vec[p + "ssm_conv1d.weight"]
        k = cw.shape[1]
        prev = st["conv"].get(l) if st else None
        if PFUSE and dk == 128 and dv == 128:
            return self._gdn_fused(l, x, st, qkv, z, b, a, cw, prev)
        qkv = torch.cat([torch.zeros(k - 1, qkv.shape[1], device=x.device) if prev is None else prev, qkv])
        if st is not None:
            st["conv"][l] = qkv[-(k - 1):].clone()
        qkv = F.silu(F.conv1d(qkv.T[None], cw[:, None, :], groups=cw.shape[0]))[0].T
        q, kk, v = torch.split(qkv, [nk * dk, nk * dk, nv * dv], dim=-1)
        q = q.reshape(T, nk, dk)
        kk = kk.reshape(T, nk, dk)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(b).contiguous()
        g = (self.vec[p + "ssm_a"] * F.softplus(a + self.vec[p + "ssm_dt.bias"])).contiguous()
        S = st["S"].get(l) if st else None
        if S is None:
            S = torch.zeros(nv, dk, dv, device=x.device)
        out = torch.empty(T, nv, dv, device=x.device)
        q, kk, v = q.contiguous(), kk.contiguous(), v.contiguous()
        lib().gdn(*_st(out), T, nk, nv, _p(q), _p(kk), _p(v), _p(g), _p(beta), _p(S), _p(out))
        if st is not None:
            st["S"][l] = S
        o = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + m.eps) * self.vec[p + "ssm_norm.weight"]
        o = o * torch.sigmoid(z.view(T, nv, dv))
        return self.mv(p + "ssm_out.weight", o.reshape(T, -1))

    def _gdn_fused(self, l, x, st, qkv, z, b, a, cw, prev):
        """gdn() after its projections on fused kernels (KINDI_QXG_PFUSE=1): conv + silu + q/k norms + beta/g in
        three launches, the recurrence, then the gated norm (k_gnormT)"""
        m, L = self.ref, lib()
        p = "blk.%d." % l
        T = x.shape[0]
        nk, nv = m.ssm_groups, m.ssm_heads
        C, kw = cw.shape
        cqk = 2 * nk * 128
        dvc = x.device
        d_, s_ = _st(qkv)
        qkv, b, a = qkv.contiguous(), b.contiguous(), a.contiguous()
        qk = torch.empty((T, cqk), dtype=torch.float32, device=dvc)
        v = torch.empty((T, C - cqk), dtype=torch.float32, device=dvc)
        state = torch.empty((kw - 1, C), dtype=torch.float32, device=dvc) if st is not None else None
        beta = torch.empty((T, nv), dtype=torch.float32, device=dvc)
        g = torch.empty((T, nv), dtype=torch.float32, device=dvc)
        L.gdn_pre(d_, s_, _p(qkv), _p(prev) if prev is not None else 0, _p(cw), _p(qk), _p(v),
                  _p(state) if state is not None else 0, T, C, kw, cqk, nk, _p(b), _p(a), _p(self.vec[p + "ssm_a"]),
                  _p(self.vec[p + "ssm_dt.bias"]), _p(beta), _p(g), nv)
        if st is not None:
            st["conv"][l] = state
        S = st["S"].get(l) if st else None
        if S is None:
            S = torch.zeros(nv, 128, 128, device=dvc)
        out = torch.empty(T, nv, 128, device=dvc)
        L.gdn(d_, s_, T, nk, nv, _p(qk), _p(qk) + T * nk * 128 * 4, _p(v), _p(g), _p(beta), _p(S), _p(out))
        if st is not None:
            st["S"][l] = S
        o = torch.empty((T, nv * 128), dtype=torch.float32, device=dvc)
        z = z.contiguous()
        L.gnormT(d_, s_, _p(out), _p(z), _p(self.vec[p + "ssm_norm.weight"]), _p(o), nv, m.eps, T, nv * 128, nv * 128)
        return self.mv(p + "ssm_out.weight", o)

    def attn(self, l, x, st):
        m = self.ref
        p = "blk.%d." % l
        T = x.shape[0]
        p0 = st["pos"] if st else 0
        nh, nkv, hd = m.nh, m.nkv, m.hd
        qg = self.mv(p + "attn_q.weight", x).view(T, nh, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
        q = rms(q, self.vec[p + "attn_q_norm.weight"], m.eps)
        k = rms(self.mv(p + "attn_k.weight", x).view(T, nkv, hd), self.vec[p + "attn_k_norm.weight"], m.eps)
        v = self.mv(p + "attn_v.weight", x).view(T, nkv, hd)
        pos = torch.arange(p0, p0 + T, device=x.device)
        q, k = self._rope(q, pos), self._rope(k, pos)
        if st is not None:
            if l in st["k"]:
                k, v = torch.cat([st["k"][l], k]), torch.cat([st["v"][l], v])
            st["k"][l], st["v"][l] = k, v
        kr = self.mv(p + "indexer.k_proj.weight", x)
        if st is not None:
            if l in st["ik"]:
                kr = torch.cat([st["ik"][l], kr])
            st["ik"][l] = kr
        if (env("KINDI_QXG_TENSOR_QSA") == "fp16" and kr.shape[0] >= 2 * R.QSA_TOKENS
                and hd in (128, 256) and nh // nkv <= 16 and torch.cuda.get_device_capability(x.device)[0] >= 8):
            try:
                from kindi.qx_prefill_qsa import selected_attention
            except ModuleNotFoundError as error:
                if error.name != "triton":
                    raise
            else:
                selected, valid = R.qsa_positions(self._index_q(l, x, pos), kr, pos,
                                                   self.vec[p + "indexer.k_norm.weight"], m.eps, self._rope)
                attended = selected_attention(q, k, v, selected, valid)
                return self.mv(p + "attn_output.weight", attended.reshape(T, -1) * torch.sigmoid(gate))
        if CAUSAL_PREFILL and kr.shape[0] <= R.QSA_TOKENS:
            # Every visible key fits. Keep the raw index keys above for future chunks.
            mask = torch.arange(kr.shape[0], device=x.device)[None, :] <= pos[:, None]
        else:
            mask = R.qsa_mask(self._index_q(l, x, pos), kr, pos, self.vec[p + "indexer.k_norm.weight"], m.eps, self._rope)
        # grouped: Q head h reads KV head h // rep, without copying K and V out to every Q head (at 16K positions
        # those copies took about 800 MB)
        rep = nh // nkv
        qg = q.reshape(T, nkv, rep, hd)
        if FAST["attn"]:
            o = self._attn_fast(qg, k, v, mask, hd).reshape(T, -1)
            return self.mv(p + "attn_output.weight", o * torch.sigmoid(gate))
        outs = []
        for a in range(0, T, QCHUNK):                 # queries in sub-chunks: the scores stay QCHUNK x keys
            end = min(k.shape[0], p0 + a + QCHUNK) if CAUSAL_PREFILL else k.shape[0]
            s = torch.einsum("tgrd,sgd->grts", qg[a:a + QCHUNK], k[:end]) / math.sqrt(hd)
            s = s.masked_fill(~mask[None, None, a:a + QCHUNK, :end], float("-inf"))
            outs.append(torch.einsum("grts,sgd->tgrd", torch.softmax(s, -1), v[:end]))
        o = torch.cat(outs).reshape(T, -1)
        return self.mv(p + "attn_output.weight", o * torch.sigmoid(gate))

    def _attn_fast(self, qg, k, v, mask, hd):
        """the prompt's attention in the fast mode (lossy, see _fast_config): KINDI_FAST_ATTN=tf32, the products in
        TF32; fp16, fp16 inputs with an fp32 softmax"""
        fa = FAST["attn"]
        prev = torch.backends.cuda.matmul.allow_tf32
        prevh = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_tf32 = fa == "tf32" or prev
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            outs = []
            p0 = k.shape[0] - qg.shape[0]
            for a in range(0, qg.shape[0], QCHUNK):
                end = min(k.shape[0], p0 + a + QCHUNK) if CAUSAL_PREFILL else k.shape[0]
                mk = mask[a:a + QCHUNK, :end]
                qq = qg[a:a + QCHUNK] / math.sqrt(hd)
                if fa == "fp16":
                    s = torch.einsum("tgrd,sgd->grts", qq.half(), k[:end].half()).float()
                else:
                    s = torch.einsum("tgrd,sgd->grts", qq, k[:end])
                pr = torch.softmax(s.masked_fill(~mk[None, None], float("-inf")), -1)
                if fa == "fp16":
                    outs.append(torch.einsum("grts,sgd->tgrd", pr.half(), v[:end].half()).float())
                else:
                    outs.append(torch.einsum("grts,sgd->tgrd", pr, v[:end]))
            return torch.cat(outs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = prevh

    def _index_q(self, l, x, pos):
        """the QSA indexer queries [T, heads, d] (normed, RoPE at pos)"""
        m = self.ref
        p = "blk.%d." % l
        di = self.mat[p + "indexer.k_proj.weight"][2]
        qi = self.mv(p + "indexer.q_proj.weight", x).view(x.shape[0], -1, di)
        return self._rope(rms(qi, self.vec[p + "indexer.q_norm.weight"], m.eps), pos)

    def _rope(self, x, pos):
        m = self.ref
        half = m.rope_dim // 2
        inv = getattr(self, "_inv", None)
        if inv is None:
            inv = self._inv = {}
        if x.device not in inv:                     # cached per card (no host copy inside a graph capture)
            inv[x.device] = m.inv_freq.to(x.device)
        ang = (pos[:, None].double() * inv[x.device][None, :]).float()
        c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x1, x2, rest = x[..., :half], x[..., half:m.rope_dim], x[..., m.rope_dim:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, rest], dim=-1)

    def _stream_ok(self, l, T):
        """KINDI_QXG_STREAM=1 (phase G): a prompt chunk of at least KINDI_QXG_STREAM_MIN tokens (64) gets all its
        experts on the card: the cached ones in place, the others copied from the page-locked file pages into a ring
        of card rows and computed there (int8 tensor cores, the CPU's bits), so no value rows come back from the CPU.
        Needs one card, the expert pages page-locked (as KINDI_QXG_PIN) and sm_80+."""
        if not getattr(self, "stream", False) or T < int(os.environ.get("KINDI_QXG_STREAM_MIN", "64")):
            return None
        cc = self.cache_of.get(l)
        if cc is None or len(self.devices) != 1 or torch.cuda.get_device_capability(self.devices[0])[0] < 8:
            return None
        if not getattr(cc, "pinned", False):
            cc.pinned = cc._pin()
            if not cc.pinned:
                return None
        return cc

    def _prefetch_expert_prefix(self, l, tokens, limit):
        self._prefetched = None
        cc = self._stream_ok(l, tokens)
        if cc is None:
            return
        missing = self.slot[l] < 0
        priority = self._prefetch_priority.get(l)
        if env("KINDI_QXG_PREFETCH_ORDER") == "usage" and priority is not None:
            cold = np.sort(priority[missing[priority]][:limit])
        else:
            cold = np.flatnonzero(missing)[:limit]
        if not cold.size:
            return
        source = cc.src[l]
        widths = np.array([width for _, _, width in source], np.uintp)
        total = int(widths.sum()) * cold.size
        dv = self.devices[0]
        bank = getattr(self, "_prefetch_bank", None)
        new_bank = bank is None or bank.numel() < total
        if new_bank:
            available = torch.cuda.mem_get_info(dv)[0] + torch.cuda.memory_reserved(dv) - torch.cuda.memory_allocated(dv)
            if available < total + (384 << 20):
                return
            try:
                bank = torch.empty(total, dtype=torch.uint8, device=dv)
            except torch.OutOfMemoryError:
                return
            self._prefetch_bank = bank
        side = getattr(self, "_ring_stream", None)
        if side is None:
            side = self._ring_stream = torch.cuda.Stream(dv)
        if new_bank:
            # The main stream may still use the caching allocator's previous occupant.
            side.wait_stream(torch.cuda.current_stream(dv))
        reusable = getattr(self, "_prefetch_reusable", None)
        if reusable is not None:
            side.wait_event(reusable)
        hosts = np.array([np.frombuffer(mm, np.uint8).ctypes.data + base for mm, base, _ in source], np.uintp)
        planes = cold.size * np.array([0, widths[0], widths[0] + widths[1]], np.uintp)
        entries = np.uintp(bank.data_ptr()) + np.arange(cold.size, dtype=np.uintp)[:, None] * widths + planes
        starts = np.concatenate(([0], np.flatnonzero(np.diff(cold) != 1) + 1))
        lengths = np.diff(np.append(starts, cold.size)).astype(np.uintp)
        origins = hosts + cold[starts, None].astype(np.uintp) * widths
        sizes = lengths[:, None] * widths
        destinations = entries[starts]
        cu, copy_stream = lib().cu, ctypes.c_void_p(side.cuda_stream)
        early = None
        if env("KINDI_QXG_PREFETCH_PHASES") == "1":
            _submit_weight_copies(cu, copy_stream, np.ascontiguousarray(origins[:, :2]),
                                  np.ascontiguousarray(destinations[:, :2]), np.ascontiguousarray(sizes[:, :2]))
            early = torch.cuda.Event()
            early.record(side)
            origins, destinations, sizes = (np.ascontiguousarray(a[:, 2:3]) for a in (origins, destinations, sizes))
        _submit_weight_copies(cu, copy_stream, origins, destinations, sizes)
        bank.record_stream(side)
        ready = torch.cuda.Event()
        ready.record(side)
        slots = np.full(self.ref.n_exp, -1, np.int32)
        slots[cold] = np.arange(cold.size)
        self._prefetched = l, slots, entries, early if early is not None else ready, ready if early is not None else None
        self._prefetch_reusable = ready      # no consumer is also a valid reuse case
        self.times["stream_bytes"] = self.times.get("stream_bytes", 0) + total
        self.times["stream_copies"] = self.times.get("stream_copies", 0) + len(starts) * len(widths)
        self.times["prefetch_experts"] = self.times.get("prefetch_experts", 0) + cold.size

    def _experts_stream(self, l, cc, x, idx_h, parts):
        """every pick of the chunk on the card (see _stream_ok): experts grouped as in experts(); the cached ones in
        one pass; the others in batches of KINDI_QXG_RING / 2 experts, each batch's copies (a side stream, direct
        from the page-locked pages) landing in one half of the ring while the other half computes"""
        m, L = self.ref, lib()
        T, n = idx_h.shape
        K, Fd = H, m.w.t["blk.%d.ffn_gate_exps.weight" % l][2][1]
        gty = m.w.t["blk.%d.ffn_gate_exps.weight" % l][3]
        dty = m.w.t["blk.%d.ffn_down_exps.weight" % l][3]
        dv = x.device
        flat = idx_h.reshape(-1)
        coalesce = env("KINDI_QXG_STREAM_UPLOAD") == "packed-pipeline"
        uq, fi = np.unique(flat, return_index=True)
        cached = self.slot[l][uq] >= 0
        prefetched = getattr(self, "_prefetched", None)
        ahead = prefetched[1][uq] >= 0 if prefetched is not None and prefetched[0] == l else np.zeros_like(cached)
        ahead &= ~cached
        # Keep cached experts in first-use order. Sort cold experts by index for contiguous copies.
        absent = ~(cached | ahead)
        misses = uq[absent] if coalesce else uq[absent][np.argsort(fi[absent])]
        order = np.concatenate([uq[cached][np.argsort(fi[cached])], uq[ahead], misses])
        n_hot = int(cached.sum())
        n_ready = n_hot + int(ahead.sum())
        ready_group = max(1, int(env("KINDI_QXG_PREFETCH_COMPUTE_GROUP", str(n_ready - n_hot))))
        ready_ranges = [(u, min(u + ready_group, n_ready)) for u in range(n_hot, n_ready, ready_group)]
        umap = np.full(m.n_exp, -1, np.int64)
        umap[order] = np.arange(len(order))
        u_of = umap[flat]
        perm = np.argsort(u_of, kind="stable")
        ptok = (np.arange(T * n) // n)[perm].astype(np.int32)
        pslot = (np.arange(T * n) % n)[perm].astype(np.int32)
        pstart = np.concatenate([[0], np.cumsum(np.bincount(u_of, minlength=len(order)))]).astype(np.int32)
        src = cc.src[l]                                      # (mm, base, bytes per expert) of gate, up, down
        nbs = [nb for _, _, nb in src]
        rowb = sum(nbs)
        R = max(2, int(env("KINDI_QXG_RING", "32")) // 2 * 2)
        ring = getattr(self, "_ring", None)
        if ring is None or ring.shape[0] < R or ring.device != dv:   # rows of the largest expert of any layer
            wmax = max(sum(nb for _, _, nb in parts_) for parts_ in cc.src.values())
            self._ring = ring = torch.empty((R, wmax), dtype=torch.uint8, device=dv)
        buf = self.hot[l]
        mats = np.empty(3 * len(order), np.int64)
        for u, e in enumerate(order[:n_hot]):
            a = buf.data_ptr() + int(self.slot[l][e]) * buf.shape[1]
            mats[3 * u], mats[3 * u + 1], mats[3 * u + 2] = a, a + nbs[0], a + nbs[0] + nbs[1]
        if n_ready > n_hot:
            mats.reshape(-1, 3)[n_hot:n_ready] = prefetched[2][prefetched[1][uq[ahead]]]
        half = R // 2
        compact = env("KINDI_QXG_COMPACT_STREAM") == "1"
        if coalesce:
            widths = np.array(nbs, dtype=np.uintp)
            planes = R * np.array([0, nbs[0], nbs[0] + nbs[1]], dtype=np.uintp)
            rows = np.arange(len(misses), dtype=np.uintp) % R
            destinations = np.uintp(ring.data_ptr()) + rows[:, None] * widths + planes
            mats.reshape(-1, 3)[n_ready:] = destinations
        else:
            for i in range(len(misses)):
                a = ring.data_ptr() + (i % R) * ring.shape[1]
                u = n_ready + i
                mats[3 * u], mats[3 * u + 1], mats[3 * u + 2] = a, a + nbs[0], a + nbs[0] + nbs[1]
        xt = None
        group_start = {}
        npair = int(pstart[-1])
        if compact:
            # One group consumes its scratch before the next group reuses it on the compute stream.
            groups = [(0, n_hot)] if n_hot else []
            groups += ready_ranges
            groups += [(n_ready + b, n_ready + min(b + half, len(misses))) for b in range(0, len(misses), half)]
            boundaries = []
            offset = 0
            for u0, u1 in groups:
                p0, p1 = int(pstart[u0]), int(pstart[u1])
                boundaries.append(pstart[u0:u1 + 1] - p0)
                group_start[u0] = offset
                offset += u1 - u0 + 1
            npair = max(int(pstart[u1] - pstart[u0]) for u0, u1 in groups)
            packed_start = np.concatenate(boundaries)
        else:
            packed_start = pstart
        if L.XTILE:                                          # the tiled expert GEMM's tile list
            tl, tstart = xtiles(pstart, L.XT_BN)
            if compact:
                for u0, u1 in groups:
                    tiles = tl[tstart[u0]:tstart[u1]]
                    tiles[:, 0] -= u0
                    tiles[:, 1:3] -= pstart[u0]
            mats_d, pstart_d, ptok_d, pslot_d, tl_d = _upload(dv, mats, packed_start, ptok, pslot, tl)
            xt = (tl_d, tstart)
        else:
            mats_d, pstart_d, ptok_d, pslot_d = _upload(dv, mats, packed_start, ptok, pslot)
        xc = x.contiguous()
        xq = torch.empty((T, K), dtype=torch.int8, device=dv)
        xd = torch.empty((T, K // 256), dtype=torch.float32, device=dv)
        gg = torch.empty((npair, Fd), dtype=torch.float32, device=dv)
        vv = torch.empty((npair, Fd), dtype=torch.float32, device=dv)
        h = gg if compact else torch.empty((npair, Fd), dtype=torch.float32, device=dv)
        hq = torch.empty((npair, Fd), dtype=torch.int8, device=dv)
        hd = torch.empty((npair, Fd // L.DN), dtype=torch.float32, device=dv)
        main = torch.cuda.current_stream(dv)
        d_, st = _st(xc)
        L.quant256(d_, st, _p(xc), T * K, _p(xq), _p(xd))

        def run(u0, u1, down_ready=None):                    # experts u0 .. u1 - 1 (their pairs are contiguous)
            if u1 <= u0:
                return
            p0, p1 = int(pstart[u0]), int(pstart[u1])
            count = (p1 - p0) * Fd
            md, pd = mats_d, pstart_d
            tk, sk, tiles = ptok_d, pslot_d, xt
            lo, hi = u0, u1
            gp, vp, hp, hqp, hdp = gg[p0:], vv[p0:], h[p0:], hq[p0:], hd[p0:]
            if compact:
                md, pd = mats_d[3 * u0:], pstart_d[group_start[u0]:]
                tk, sk = ptok_d[p0:], pslot_d[p0:]
                if xt is not None:
                    first_tile = int(xt[1][u0])
                    tiles = (xt[0][first_tile:], xt[1][u0:u1 + 1] - first_tile)
                lo, hi = 0, u1 - u0
                gp, vp, hp, hqp, hdp = gg, vv, h, hq, hd
            for part, o in ((0, gg), (1, vv)):
                _xmm(L, d_, st, gty, K, Fd, 256, part, md, pd, tk, sk, n, 0, _p(xq), _p(xd), _p(o),
                     Fd, tiles, lo, hi)
            L.silu_mul(d_, st, _p(gp), _p(vp), _p(hp), count)
            L.quant_dn(d_, st, _p(hp), count, _p(hqp), _p(hdp))
            if down_ready is not None:
                main.wait_event(down_ready)
            if L.DN == 64:
                _xmm(L, d_, st, dty, Fd, K, 64, 2, md, pd, tk, sk, n, 1, _p(hq), _p(hd), _p(parts),
                     K, tiles, lo, hi)
            else:
                L.down(d_, st, dty, K, Fd, n, hi - lo, _p(md[3 * lo:]), _p(pd[lo:]), _p(tk), _p(sk), _p(hq), _p(hd),
                       _p(parts))

        side = getattr(self, "_ring_stream", None)
        if side is None:
            side = self._ring_stream = torch.cuda.Stream(dv)
        side.wait_stream(main)                               # the ring's last users (the previous layer) come first
        run(0, n_hot)
        if n_ready > n_hot:
            main.wait_event(prefetched[3])
            for start, end in ready_ranges:
                run(start, end, prefetched[4])
            self._prefetch_reusable = torch.cuda.Event()
            self._prefetch_reusable.record(main)
            self.times["prefetch_hits"] = self.times.get("prefetch_hits", 0) + n_ready - n_hot
        cu, ss = L.cu, ctypes.c_void_p(side.cuda_stream)
        hosts = [(np.frombuffer(mm, np.uint8).ctypes.data + base, nb) for mm, base, nb in src]
        if coalesce:
            sources = np.array([a for a, _ in hosts], dtype=np.uintp) + misses.astype(np.uintp)[:, None] * widths

            def upload(starts, lengths, columns):
                origin = np.ascontiguousarray(sources[starts, columns])
                dest = np.ascontiguousarray(destinations[starts, columns])
                sizes = np.ascontiguousarray(lengths[:, None] * widths[columns])
                _submit_weight_copies(cu, ss, origin, dest, sizes)

        done = [None, None]                                  # per ring half: the event after its last compute
        copy_ops = 0
        for b0 in range(0, len(misses), half):
            hf = (b0 // half) % 2
            if done[hf] is not None:
                side.wait_event(done[hf])                    # that half's previous batch has been computed
            end = min(b0 + half, len(misses))
            if coalesce:
                # Both index order and ring layout are contiguous within each run.
                starts = np.concatenate(([b0], np.flatnonzero(np.diff(misses[b0:end]) != 1) + b0 + 1))
                lengths = np.diff(np.append(starts, end)).astype(np.uintp)
                upload(starts, lengths, slice(0, 2))
                copy_ops += len(starts) * len(hosts)
            else:
                for i in range(b0, end):
                    e = int(misses[i])
                    dst = ring.data_ptr() + (i % R) * ring.shape[1]
                    for a, nb in hosts:
                        cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(dst), ctypes.c_void_p(a + e * nb), ctypes.c_size_t(nb), ss)
                        dst += nb
                copy_ops += (end - b0) * len(hosts)
            landed = torch.cuda.Event()
            landed.record(side)
            down_ready = None
            if coalesce:
                upload(starts, lengths, slice(2, 3))
                down_ready = torch.cuda.Event()
                down_ready.record(side)
            main.wait_event(landed)
            run(n_ready + b0, n_ready + end, down_ready)
            done[hf] = torch.cuda.Event()
            done[hf].record(main)
        self.times["stream_bytes"] = self.times.get("stream_bytes", 0) + len(misses) * rowb
        self.times["stream_experts"] = self.times.get("stream_experts", 0) + len(misses)
        self.times["stream_copies"] = self.times.get("stream_copies", 0) + copy_ops

    def experts(self, l, x, idx, top, idx_h=None):
        """the routed experts: the card's hot experts here, the others on the CPU, combined in slot order"""
        m = self.ref
        T, n = idx.shape
        cc = self._stream_ok(l, T)
        if cc is not None:                                   # phase G: every pick on the card
            K = H
            parts = torch.empty((T, n, K), dtype=torch.float32, device=x.device)
            self._experts_stream(l, cc, x, idx.int().cpu().numpy() if idx_h is None else idx_h, parts)
            out = torch.empty((T, K), dtype=torch.float32, device=x.device)
            w = top.float().contiguous()
            lib().combine(*_st(out), T, n, K, _p(w), _p(parts), _p(out))
            return out, parts
        K, Fd = H, m.w.t["blk.%d.ffn_gate_exps.weight" % l][2][1]
        dv = x.device
        idx_h = idx.int().cpu().numpy()
        hot = self.slot[l][idx_h] >= 0                         # [T, n]
        parts = torch.empty((T, n, K), dtype=torch.float32, device=dv)
        L = lib()
        if hot.any():
            gty = m.w.t["blk.%d.ffn_gate_exps.weight" % l][3]
            dty = m.w.t["blk.%d.ffn_down_exps.weight" % l][3]
            hp = np.nonzero(hot.reshape(-1))[0]                 # the card's (token, slot) pairs, in order
            e_hot = idx_h.reshape(-1)[hp]
            uq, fi = np.unique(e_hot, return_index=True)
            uexp = uq[np.argsort(fi)]                           # experts in first-use order
            umap = np.full(m.n_exp, -1, np.int64)
            umap[uexp] = np.arange(len(uexp))
            u_of = umap[e_hot]
            perm = np.argsort(u_of, kind="stable")             # grouped by expert, (token, slot) order kept
            ptok = (hp // n)[perm].astype(np.int32)
            pslot = (hp % n)[perm].astype(np.int32)
            pstart = np.concatenate([[0], np.cumsum(np.bincount(u_of, minlength=len(uexp)))]).astype(np.int32)
            buf = self.hot[l]
            gb = self._nbytes("blk.%d.ffn_gate_exps.weight" % l) // m.n_exp
            base = buf.data_ptr()
            rowb = buf.shape[1]
            mats = np.empty(3 * len(uexp), np.int64)
            for u, e in enumerate(uexp):
                a = base + int(self.slot[l][e]) * rowb
                mats[3 * u], mats[3 * u + 1], mats[3 * u + 2] = a, a + gb, a + 2 * gb
            tcx = getattr(self, "tcx", False) and torch.cuda.get_device_capability(dv)[0] >= 8
            xt = None
            if tcx and L.XTILE:                              # the tiled expert GEMM's tile list
                tl, tstart = xtiles(pstart, L.XT_BN)
                mats_d, pstart_d, ptok_d, pslot_d, tl_d = _upload(dv, mats, pstart, ptok, pslot, tl)
                xt = (tl_d, tstart)
            else:
                mats_d, pstart_d, ptok_d, pslot_d = _upload(dv, mats, pstart, ptok, pslot)
            xc = x.contiguous()
            xq = torch.empty((T, K), dtype=torch.int8, device=dv)
            xd = torch.empty((T, K // 256), dtype=torch.float32, device=dv)
            npair = int(pstart[-1])
            h = torch.empty((npair, Fd), dtype=torch.float32, device=dv)
            hq = torch.empty((npair, Fd), dtype=torch.int8, device=dv)
            hd = torch.empty((npair, Fd // L.DN), dtype=torch.float32, device=dv)
            d_, st = _st(xc)
            L.quant256(d_, st, _p(xc), T * K, _p(xq), _p(xd))
            # KINDI_QXG_TCX=1 (item G, sm_80+): gate/up, and the down on KINDI_QX_DN=64, on int8 tensor cores (the
            # CPU's bits); otherwise the CUDA-core rows
            if tcx:
                gg = torch.empty((npair, Fd), dtype=torch.float32, device=dv)
                vv = torch.empty((npair, Fd), dtype=torch.float32, device=dv)
                for part, o in ((0, gg), (1, vv)):
                    _xmm(L, d_, st, gty, K, Fd, 256, part, mats_d, pstart_d, ptok_d, pslot_d, n, 0, _p(xq), _p(xd),
                         _p(o), Fd, xt, 0, len(uexp))
                L.silu_mul(d_, st, _p(gg), _p(vv), _p(h), npair * Fd)
                del gg, vv
            else:
                L.gate_up(d_, st, gty, K, Fd, len(uexp), _p(mats_d), _p(pstart_d), _p(ptok_d), _p(xq), _p(xd), _p(h))
            L.quant_dn(d_, st, _p(h), npair * Fd, _p(hq), _p(hd))
            if tcx and L.DN == 64:
                _xmm(L, d_, st, dty, Fd, K, 64, 2, mats_d, pstart_d, ptok_d, pslot_d, n, 1, _p(hq), _p(hd), _p(parts),
                     K, xt, 0, len(uexp))
            else:
                L.down(d_, st, dty, K, Fd, n, len(uexp), _p(mats_d), _p(pstart_d), _p(ptok_d), _p(pslot_d), _p(hq),
                       _p(hd), _p(parts))
        if not hot.all() and os.environ.get("KINDI_QXG_PROMPT_SUM") == "1" and hasattr(self.cpu, "forward_sum"):
            # tracing: the CPU's slots come back summed per token (not the exact slot-order combine)
            out = torch.empty((T, K), dtype=torch.float32, device=dv)
            w = top.float().contiguous()
            parts[torch.from_numpy(~hot).to(dv)] = 0.0
            L.combine(*_st(out), T, n, K, _p(w), _p(parts), _p(out))
            cs = self.cpu.forward_sum(l, x.float().cpu().numpy(), idx_h, hot.astype(np.uint8),
                                      top.float().cpu().numpy())
            return out + torch.from_numpy(cs).to(dv), parts
        if not hot.all():
            import time
            t0 = time.time()
            ph = np.zeros((T, n, K), np.float32)
            self.cpu.forward_parts(l, x.float().cpu().numpy(), idx_h, hot.astype(np.uint8), ph)
            cold = torch.from_numpy(~hot).to(dv)
            parts[cold] = torch.from_numpy(ph[~hot]).to(dv)   # only the CPU's rows cross the link
            self.times["cpu_experts"] += time.time() - t0
        out = torch.empty((T, K), dtype=torch.float32, device=dv)
        w = top.float().contiguous()
        L.combine(*_st(out), T, n, K, _p(w), _p(parts), _p(out))
        return out, parts

    def moe(self, l, x):
        m = self.ref
        p = "blk.%d." % l
        probs = torch.softmax(self.mv(p + "ffn_gate_inp.weight", x), dim=-1)
        top, idx = probs.topk(m.n_used, dim=-1)
        top = top / top.sum(-1, keepdim=True)
        out, _ = self.experts(l, x, idx, top)
        sh = self.mv(p + "ffn_down_shexp.weight",
                     F.silu(self.mv(p + "ffn_gate_shexp.weight", x)) * self.mv(p + "ffn_up_shexp.weight", x))
        sg = torch.sigmoid(x @ self.vec[p + "ffn_gate_inp_shexp.weight"])
        return out + sg[:, None] * sh

    # ---------------------------------------------------------------- forward
    def layer(self, l, h, ids, st=None):
        """one layer on its card: h [T, 4 x 2560] (moved to the card) -> h after the layer"""
        m = self.ref
        dv = self.devices[self.dev_of[l]]
        h = h.to(dv)
        with torch.cuda.device(dv):
            if l in m.ple_layers:
                h = h + self.ple(l, h, ids, st)
            x, hin, wr = self.mixer(h, "blk.%d.hc_attn" % l)
            y = self.attn(l, x, st) if (l + 1) % m.interval == 0 else self.gdn(l, x, st)
            h = self.write(hin, y, wr)
            x, hin, wr = self.mixer(h, "blk.%d.hc_ffn" % l)
            return self.write(hin, self.moe(l, x), wr)

    def head(self, h, last_n=None):
        dv = self.devices[-1]
        h = h.to(dv)
        with torch.cuda.device(dv):
            x = self.mixer(h, "output_hc", inject=False)
            if last_n:
                x = x[-last_n:]
            return self.mv("output.weight", x)

    def emb_rows(self, ids, dv, name="token_embd.weight", min_rows=32):
        """rows ids [n] of an embedding table as fp32 on card dv. A prompt's rows (n >= 32) are gathered raw on the
        host and decoded on the card (KINDI_QXG_EMB_GPU, 1): the host decode took 1.0 s per 4,096 tokens of the
        token table on the 3060 host, paid by the prompt and again by the drafter's head rows (B53-B54), and the
        per-layer table's rows (about 17 a token) the same way. Fewer rows: the host decode."""
        m = self.ref
        if len(ids) < min_rows or os.environ.get("KINDI_QXG_EMB_GPU", "1") != "1":
            return m.w.rows(name, list(ids)).to(dv)
        mm, base, ne, ty = m.w.t[name]
        rb = Q.nbytes(ty, ne[0])
        tab = np.frombuffer(mm, np.uint8, rb * ne[1], base).reshape(ne[1], rb)
        raw = torch.from_numpy(tab[np.asarray(ids, np.int64)]).to(dv)
        out = torch.empty((len(ids), ne[0]), dtype=torch.float32, device=dv)
        d_, st = _st(out)
        lib().deq(d_, st, ty, _p(raw), len(ids), ne[0], rb, _p(out))
        return out

    def ext_off(self, keep_windows=None):
        """the elastic tails' memory back to the driver (a long prompt read is next), and the drafter's window
        buffers and graphs with it (KINDI_QXG_WIN_FREE, 1): the plan gives the layer-major prompt and the drafter's
        windows the same reserve, so they must not be held at once (B51, 3060: the second drafted long prompt read
        with the windows still allocated, 4K 360 / 16K 319 against 400 / 351 plain, peak 11.54 GB). Staged short
        reads can keep warm graphs when there is enough room. Freed windows are captured before the next decode."""
        dec = getattr(self, "_dec", None)
        for cc in set(self.cache_of.values()):
            cc.ext_off(dec)
        if keep_windows is None:
            keep_windows = os.environ.get("KINDI_QXG_WIN_FREE", "1") != "1"
        if dec is not None and not keep_windows:
            dec.free_windows()

    def logits(self, ids, last_n=None, state=None, tap=None, draft_prefill_retry=False):
        m = self.ref
        ids = list(ids)
        if len(ids) > int(os.environ.get("KINDI_QXG_EXT_PROMPT", "256")):
            keep_windows = False if draft_prefill_retry else None
            if (not draft_prefill_retry and tap is not None and state is not None and state["pos"] + len(ids) <= 4096
                    and env("KINDI_MTP_STAGE_PROMPT") == "1" and "KINDI_QXG_WIN_FREE" not in os.environ
                    and getattr(getattr(self, "_dec", None), "_wins", None)):
                dv = self.devices[0]
                free, _ = torch.cuda.mem_get_info(dv)
                keep_windows = free + torch.cuda.memory_reserved(dv) - torch.cuda.memory_allocated(dv) >= (2 << 30)
            self.ext_off(keep_windows)
        st = state
        pc = getattr(self, "prompt_chunk", 0)
        if st is not None and pc and len(ids) > pc and last_n and last_n <= pc and getattr(self, "layer_major", False) \
                and len(self.devices) == 1:
            if getattr(self, "profile", False):
                _PROFILE.update((PROFILE_DRAFT if tap is not None else PROFILE_PLAIN)[
                    int(st["pos"] + len(ids) > PROFILE_LONG)])
            try:
                return self._logits_lm(ids, last_n, st, pc, tap, draft_prefill_retry)
            finally:
                if getattr(self, "profile", False):
                    _PROFILE.clear()
                    _PROFILE.update(PROFILE_BASE)
                self._dq = None
                self._prefetched = None
                self._prefetch_bank = None
                self._prefetch_reusable = None
        if st is not None and tap is None and pc and len(ids) > pc and last_n and last_n <= pc:
            # a long prompt in the engine's chunks (the memory plan reserves one chunk's buffers), whatever the caller
            # passes; the state carries over, the last chunk's rows are returned
            for i in range(0, len(ids), pc):
                out = self.logits(ids[i:i + pc], last_n=min(last_n, len(ids[i:i + pc])), state=st)
            return out
        with torch.no_grad():
            # the hc copies of the row are made on the card (1 / hc of the bytes cross the link)
            h = self.emb_rows(ids, self.devices[0]).repeat(1, m.hc)
            for l in range(self.L):
                if tap is not None:
                    tap(l, h)
                h = self.layer(l, h, ids, st)
            if tap is not None:
                tap(self.L, h)
            out = self.head(h, last_n)
        if st is not None:
            st["pos"] += len(ids)
            ctx = m.ple_n - 1
            prev = list(st["ids"]) if st["ids"] is not None else [m.ple_eos] * ctx
            st["ids"] = np.asarray((prev + ids)[-ctx:], np.int64)
        return out

    def _logits_lm(self, ids, last_n, st, pc, tap=None, draft_prefill_retry=False):
        """the prompt layer by layer (KINDI_QXG_LAYER_MAJOR, phase G): for each layer every chunk of pc tokens runs its
        mixers, attention or Gated DeltaNet, router and shared expert exactly as logits() would chunk by chunk (the
        same shapes, the state as it would be at that chunk), then the routed experts of up to KINDI_QXG_LM_TOK
        tokens go in one call (each expert streamed and decoded once for them, not once per chunk; a pick's row
        does not depend on its batch), then each chunk's combine and write. So the logits have the chunked path's
        bits. tap(l, h) sees each chunk's rows (as the chunked path's calls would)."""
        m = self.ref
        dv = self.devices[0]
        ctx = m.ple_n - 1
        chunks, pos, prev = [], st["pos"], list(st["ids"]) if st["ids"] is not None else None
        for i in range(0, len(ids), pc):
            c = ids[i:i + pc]
            chunks.append((c, pos, None if prev is None else np.asarray(prev, np.int64)))
            pos += len(c)
            prev = ((prev if prev is not None else [m.ple_eos] * ctx) + c)[-ctx:]
        # a drafted read (tap: the drafter's head rows) holds the drafter's head and its buffers beside layer-major's,
        # so its expert batch is smaller (KINDI_QXG_LM_TOK_DRAFT, 3072): B66 (3060) drafted 16K read peaked at 11.20
        # GB with 4096 against the 11.07 cap (B67: 2048 gave 10.60 GB, reads 433 / 389). A pick's row does not depend
        # on its batch: the same bits. Plain reads keep KINDI_QXG_LM_TOK (4096).
        key = "KINDI_QXG_LM_TOK_DRAFT" if tap is not None else "KINDI_QXG_LM_TOK"
        lm_tok = int(env(key, "3072" if tap is not None else "4096"))
        if draft_prefill_retry:
            lm_tok = min(lm_tok, 3072)
        lm_tok = max(pc, lm_tok // pc * pc)
        if key not in os.environ and key in _PROFILE and not draft_prefill_retry:
            lm_tok = self._lm_fit(lm_tok, pc, 3072 if tap is not None else 4096)
        per = max(1, lm_tok // pc)                       # chunks per experts call
        with torch.no_grad(), torch.cuda.device(dv):
            hs = [self.emb_rows(c, dv).repeat(1, m.hc) for c, _, _ in chunks]
            reuse = env("KINDI_QXG_DEQ_REUSE", "auto")
            prefetch = 0 if draft_prefill_retry else max(0, int(env("KINDI_QXG_PREFETCH_PREFIX", "0")))
            min_stage = 4096
            if tap is not None and env("KINDI_MTP_STAGE_PROMPT") == "1":
                min_stage = max(1024, min(4096, lm_tok))
            route_slice = int(env("KINDI_QXG_PREFETCH_SLICE", "0"))
            slice_chunks = min(per, max(1, route_slice // pc)) if prefetch and route_slice > 0 else per
            for l in range(self.L):
                p = "blk.%d." % l
                # KINDI_QXG_DEQ_REUSE (auto: when the card has the room): the layer's dense matrices dequantized once
                # for all its chunks, not once per chunk
                self._dq = None
                if len(chunks) > 1 and reuse != "0":
                    need = sum(r * c for n, (_, _, r, c) in self.mat.items() if n.startswith(p)
                               and r * c <= GEMM_MAX_VALUES) * (2 if PGEMM == "fp16" else 4)
                    if reuse == "1" or torch.cuda.mem_get_info(dv)[0] >= need + (1 << 30):
                        self._dq = {}
                for b0 in range(0, len(chunks), per):
                    if prefetch:
                        tokens = sum(len(c) for c, _, _ in chunks[b0:b0 + per])
                        if tokens >= min_stage:
                            self._prefetch_expert_prefix(l, tokens, prefetch)
                    part = []
                    for ci in range(b0, min(b0 + per, len(chunks))):
                        c, p0, ctx_ids = chunks[ci]
                        st["pos"], st["ids"] = p0, ctx_ids
                        h = hs[ci]
                        if tap is not None:
                            tap(l, h)
                        if l in m.ple_layers:
                            h = h + self.ple(l, h, c, st)
                        x, hin, wr = self.mixer(h, "blk.%d.hc_attn" % l)
                        y = self.attn(l, x, st) if (l + 1) % m.interval == 0 else self.gdn(l, x, st)
                        h = self.write(hin, y, wr)
                        x, hin, wr = self.mixer(h, "blk.%d.hc_ffn" % l)
                        probs = torch.softmax(self.mv(p + "ffn_gate_inp.weight", x), dim=-1)
                        top, idx = probs.topk(m.n_used, dim=-1)
                        top = top / top.sum(-1, keepdim=True)
                        sh = self.mv(p + "ffn_down_shexp.weight", F.silu(self.mv(p + "ffn_gate_shexp.weight", x)) *
                                     self.mv(p + "ffn_up_shexp.weight", x))
                        sg = torch.sigmoid(x @ self.vec[p + "ffn_gate_inp_shexp.weight"])
                        part.append((ci, x, idx, top, sh, sg, hin, wr))
                    if b0 + per >= len(chunks):
                        self._dq = None
                    step = slice_chunks if getattr(self, "_prefetched", None) is not None else per
                    routing = torch.cat([q[2] for q in part]).int().cpu().numpy() if step < len(part) else None
                    for group in range(0, len(part), step):
                        work = part[group:group + step]
                        idx = torch.cat([q[2] for q in work])
                        picks = None if routing is None else routing[group * pc:group * pc + idx.shape[0]]
                        route_args = {"idx_h": picks} if picks is not None else {}
                        out, _ = self.experts(l, torch.cat([q[1] for q in work]), idx,
                                              torch.cat([q[3] for q in work]), **route_args)
                        a = 0
                        for ci, x, idx, top, sh, sg, hin, wr in work:
                            T = x.shape[0]
                            hs[ci] = self.write(hin, out[a:a + T] + sg[:, None] * sh, wr)
                            a += T
                        del out, _, work
                    del part
                    self._prefetched = None
            self._dq = None
            if tap is not None:
                for h in hs:
                    tap(self.L, h)
            out = self.head(hs[-1], last_n)
        st["pos"] = pos
        st["ids"] = np.asarray(prev[-ctx:], np.int64)
        return out

    new_state = staticmethod(R.Model.new_state)

    def mtp_logits(self, hc, ids, state=None):
        """the MTP head on the last card (as kindi.qwen4exp_ref.Model.mtp_logits): hc [T, 4 x 2560] the streams after
        the last layer at positions p (or the head's own streams), ids [T] the tokens at p + 1 -> (draft logits
        [T, vocab], the head's streams). state: the head's own new_state() (its attention cache; "pos" is the
        position of the first row)."""
        m = self.ref
        L, dv = self.L, self.devices[-1]
        T = hc.shape[0]
        with torch.no_grad(), torch.cuda.device(dv):
            hc = hc.to(dv)
            e = rms(m.w.rows("token_embd.weight", list(ids)).to(dv), self.vec["mtp.norm_emb.weight"], m.eps)
            e = self.mv("mtp.fc_emb.weight", e)
            hn = rms(hc, self.vec["mtp.norm_hid.weight"], m.eps).view(T * m.hc, H)
            h = (self.mv("mtp.fc_hid.weight", hn).view(T, m.hc, H) + e[:, None, :]).reshape(T, m.hc * H)
            x, hin, wr = self.mixer(h, "blk.%d.hc_attn" % L)
            h = R.Model.write(hin, self.attn(L, x, state), wr)
            x, hin, wr = self.mixer(h, "blk.%d.hc_ffn" % L)
            h = R.Model.write(hin, self.moe(L, x), wr)
            out = self.mv("output.weight", self.mixer(h, "mtp.output_hc", inject=False))
        if state is not None:
            state["pos"] += T
        return out, h

    def mtp_kv(self, hc, ids, state, chunk=512):
        """the head's attention cache rows for prompt positions (streams hc [T, 4H] at p, the tokens at p + 1): only
        the fusion, the attention mixer and the K / V / indexer-key projections run, not the head's experts or its
        output head (a prompt row is never drafted from). Chunked, so long prompts fit."""
        m = self.ref
        L, dv = self.L, self.devices[-1]
        p = "blk.%d." % L
        nkv, hd = m.nkv, m.hd
        with torch.no_grad(), torch.cuda.device(dv):
            for a in range(0, hc.shape[0], chunk):
                h_, ids_ = hc[a:a + chunk].to(dv), list(ids[a:a + chunk])
                T = h_.shape[0]
                e = rms(self.emb_rows(ids_, dv), self.vec["mtp.norm_emb.weight"], m.eps)
                e = self.mv("mtp.fc_emb.weight", e)
                hn = rms(h_, self.vec["mtp.norm_hid.weight"], m.eps).view(T * m.hc, H)
                h = (self.mv("mtp.fc_hid.weight", hn).view(T, m.hc, H) + e[:, None, :]).reshape(T, m.hc * H)
                x, _, _ = self.mixer(h, "blk.%d.hc_attn" % L)
                p0 = state["pos"]
                k = rms(self.mv(p + "attn_k.weight", x).view(T, nkv, hd), self.vec[p + "attn_k_norm.weight"], m.eps)
                v = self.mv(p + "attn_v.weight", x).view(T, nkv, hd)
                k = self._rope(k, torch.arange(p0, p0 + T, device=dv))
                kr = self.mv(p + "indexer.k_proj.weight", x)
                for d, t_ in (("k", k), ("v", v), ("ik", kr)):
                    state[d][L] = torch.cat([state[d][L], t_]) if L in state[d] else t_
                state["pos"] += T

    def generate_drafted(self, ids, n, k=3):
        """greedy with MTP drafts checked in multi-token windows (kindi.qwen4exp_mtp.GraphDrafter); the same tokens
        as generate(). Needs the head and the mailbox pass (KINDI_QXG_PASS=1)."""
        from kindi.qwen4exp_mtp import GraphDrafter
        return GraphDrafter(self, k=k).generate(ids, n)

    def generate(self, ids, n, on_token=None, graphs=None):
        """greedy: the prompt (prefill path), then n - 1 one-token steps (the CUDA-graph decode path unless
        graphs=False or KINDI_QXG_GRAPH=0)"""
        import time
        if graphs is None:
            graphs = os.environ.get("KINDI_QXG_GRAPH", "1") != "0"
        st = self.new_state()
        t0 = time.time()
        nxt = int(self.logits(ids, last_n=1, state=st)[-1].argmax())
        torch.cuda.synchronize()
        t_prompt = time.time() - t0
        if graphs:
            dec = self.decoder()
            dec.load(st)
        out, times = [nxt], []
        for _ in range(n - 1):
            t1 = time.time()
            if graphs:
                nxt = int(dec.step(nxt).argmax())
            else:
                nxt = int(self.logits([nxt], last_n=1, state=st)[-1].argmax())
            times.append(time.time() - t1)
            out.append(nxt)
            if on_token:
                on_token(nxt, times[-1])
        return out, t_prompt, times

    def stats(self):
        """a stable summary for a server's /health (cumulative since load; nothing is reset): hot experts per card,
        the MTP head, the decode steps' tokens, ms a token, CPU-expert ms a token and the card's share of the
        picks, the PCIe share's counters when on"""
        by = {str(dv): int(sum(self.hot_count[l] for l in range(self.L) if self.devices[self.dev_of[l]] == dv))
              for dv in self.devices}
        out = {"hot_experts": sum(by.values()), "hot_experts_by_card": by,
               "mtp": bool(self.mtp), "lmax": int(os.environ.get("KINDI_QXG_LMAX", "8192"))}
        dec = getattr(self, "_dec", None)
        t = getattr(dec, "totals", None) if dec is not None else None
        if t and t["tokens"]:
            out.update(decode_steps=t["steps"], decode_tokens=t["tokens"],
                       ms_per_token=round(1e3 * t["wall_s"] / t["tokens"], 2),
                       cpu_ms_per_token=round(1e3 * t["cpu_s"] / t["tokens"], 2),
                       card_hit=round(t["card_picks"] / max(1, t["picks"]), 4))
        if dec is not None and getattr(dec, "fetch", None) is not None:
            out["fetch_on"] = True
        return out

    def decoder(self, lmax=None):
        if getattr(self, "_dec", None) is None:
            self._dec = GraphDecoder(self, lmax or int(os.environ.get("KINDI_QXG_LMAX", "8192")))
        return self._dec


class GraphDecoder(object):
    """One-token decode with CUDA graphs. Each layer is two graphs over static buffers: "pre" (mixers, attention or
    Gated DeltaNet, the router, the card's hot experts and the shared expert; the picks and the expert input are
    copied to pinned host memory) and "post" (the CPU's value rows in, the combine, the residual write). Between
    them the host waits for the picks only if some expert of this token is not on the card; with every expert on
    the cards a layer needs no host step at all. The head is a third graph.

    The state of the prefill (kindi.qwen4exp_ref new_state() dict) is copied into the static buffers by load():
    the attention K/V go into caches of lmax positions (scored with a mask up to the position), the conv and
    delta-rule states into fixed buffers that the graphs update in place."""

    def __init__(self, gm, lmax):
        self.g, self.m, self.lmax = gm, gm.ref, lmax
        m = self.m
        self.K, self.nsel = H, m.n_used
        self.pos = {dv: torch.zeros((), dtype=torch.long, device=dv) for dv in gm.devices}
        self.hin, self.hout = {}, {}
        self.S, self.conv, self.kc, self.vc, self.plec, self.rk, self.bk = {}, {}, {}, {}, {}, {}, {}
        self.idx, self.top, self.x2, self.hin2, self.wr2, self.sh, self.hotp, self.coldp = ({} for _ in range(8))
        self.idx_h, self.x_h, self.cold_h, self.slotmap = {}, {}, {}, {}
        self._xb = {}                          # layer -> the expert pass's buffers (the PCIe share's own pass)
        self.ev = {}
        self.sdot = {}
        self._gsus = {}
        self.ple_emb = None
        self.graphs = None
        # the fused one-token kernels of qx_gpu.cu for the steps between the matrix products (KINDI_QXG_FUSED=0:
        # the PyTorch steps, as before; both are checked against the plain decode)
        self.fused = os.environ.get("KINDI_QXG_FUSED", "1") != "0"
        # KINDI_QXG_COLDSUM=1: the CPU sends the sum of its slots (one row) instead of one row per slot; for cards
        # behind narrow links. The decode bits then depend on the placement (off: they do not).
        self.coldsum = os.environ.get("KINDI_QXG_COLDSUM", "0") == "1"
        # KINDI_QXG_PASS=1: the whole pass (48 layers and the head) is one CUDA graph; each layer's CPU experts go
        # through a mailbox in mapped host memory (kindi/moeaccel.cuh, moeaccel.c): no Python and no driver call
        # per layer. One card and the in-process CPU experts (kindi.qx_cpu.Experts) only.
        self.passmode = (os.environ.get("KINDI_QXG_PASS", "0") == "1" and len(gm.devices) == 1
                         and gm.cpu is not None and hasattr(gm.cpu, "layer") and not self.coldsum)
        self.top_h, self.csum, self.csum_h = {}, {}, {}
        nv, dk = m.ssm_heads, m.ssm_state
        for l in range(m.L):
            dv = gm.devices[gm.dev_of[l]]
            z = lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt, device=dv)   # noqa: E731
            self.hin[l] = z(1, m.hc * H) if l == 0 or gm.dev_of[l] != gm.dev_of[l - 1] else None
            if (l + 1) % m.interval == 0:
                self.kc[l], self.vc[l] = z(lmax, m.nkv, m.hd), z(lmax, m.nkv, m.hd)
                di = gm.mat["blk.%d.indexer.k_proj.weight" % l][2]
                self.rk[l], self.bk[l] = z(lmax, di), z(lmax // R.QSA_RATIO, di)
            else:
                dvv = gm.vec["blk.%d.ssm_norm.weight" % l].numel()
                self.S[l] = z(nv, dk, dvv)
                C = gm.vec["blk.%d.ssm_conv1d.weight" % l].shape[0]
                self.conv[l] = z(gm.vec["blk.%d.ssm_conv1d.weight" % l].shape[1] - 1, C)
            if l in m.ple_layers:
                self.plec[l] = z((m.ple_kernel - 1) * m.ple_n, m.hc * H)
                self.ple_emb = z(1, gm.mat["blk.%d.ple_key.weight" % l][3])
            self.idx[l] = z(self.nsel, dt=torch.int32)
            self.top[l] = z(self.nsel)
            self.hotp[l] = z(self.nsel, H)
            self.coldp[l] = z(self.nsel, H)
            self.slotmap[l] = torch.from_numpy(gm.slot[l]).to(dv)
            self.idx_h[l] = torch.zeros(self.nsel, dtype=torch.int32).pin_memory()
            self.x_h[l] = torch.zeros(1, H).pin_memory()
            self.cold_h[l] = torch.zeros(1, self.nsel, H).pin_memory()
            self.top_h[l] = torch.zeros(self.nsel).pin_memory()
            self.csum[l] = z(1, H)
            self.csum_h[l] = torch.zeros(1, H).pin_memory()
        if gm.mtp:                                   # the MTP head (layer L): its own attention caches and slots
            l = m.L
            dv = gm.devices[-1]
            z = lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt, device=dv)   # noqa: E731
            self.kc[l], self.vc[l] = z(lmax, m.nkv, m.hd), z(lmax, m.nkv, m.hd)
            di = gm.mat["blk.%d.indexer.k_proj.weight" % l][2]
            self.rk[l], self.bk[l] = z(lmax, di), z(lmax // R.QSA_RATIO, di)
            self.idx[l], self.top[l] = z(self.nsel, dt=torch.int32), z(self.nsel)
            self.hotp[l], self.coldp[l] = z(self.nsel, H), z(self.nsel, H)
            self.slotmap[l] = torch.from_numpy(gm.slot[l]).to(dv)
            self.hpos = {dv: torch.zeros((), dtype=torch.long, device=dv)}
            self.mtp_hc, self.mtp_e = z(1, m.hc * H), z(1, H)
            self.ghead_draft = None

    # ------------------------------------------------------------ the MTP head as one-token graphs
    def _head_body(self, with_logits, out="output.weight"):
        g, m = self.g, self.m
        l, dv = m.L, g.devices[-1]
        e = g.mv("mtp.fc_emb.weight", rms(self.mtp_e, g.vec["mtp.norm_emb.weight"], m.eps))
        hn = rms(self.mtp_hc, g.vec["mtp.norm_hid.weight"], m.eps).view(m.hc, H)
        h = (g.mv("mtp.fc_hid.weight", hn) + e).reshape(1, m.hc * H)
        keep = self.pos[dv]
        self.pos[dv] = self.hpos[dv]                 # the head's attention runs at its own position
        try:
            self._pre_a(l, h)
            if getattr(g, "head_counts", None) is not None:   # KINDI_MTP_USAGE_OUT: the head's routing counts
                lib().count(*_st(self.idx[l]), _p(self.idx[l]), self.nsel, _p(g.head_counts))
            self._pre_b(l)
            h2 = self._post(l)
        finally:
            self.pos[dv] = keep
        self.mtp_hout = h2
        if with_logits:
            self.mtp_lg = g.mv(out, self._mixer(h2, "mtp.output_hc", inject=False))

    # ------------------------------------------------------------ the draft chain as one graph per accepted count
    def _capture_chain(self, k):
        """graphs ch[j], j = 0..k: the head's cache rows of j accepted places (inputs upd_hc / upd_e, the head's
        streams and token embeddings), then k draft steps from (ch_hc0, ch_e0): each step's logits over the draft
        vocabulary, its pick on the card (k_pick: token, index, probability into ch_tok / ch_sel / ch_pr), the next
        step's token embedding on the card (k_deq_row) and its streams. The head's position runs on the card (hpos,
        set once before the replay). One replay and one small read per pass, instead of 2k + j launches and syncs."""
        g, m = self.g, self.m
        dv = g.devices[-1]
        z = lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt, device=dv)   # noqa: E731
        self.upd_hc, self.upd_e = z(k, m.hc * H), z(k, H)
        self.ch_hc0, self.ch_e0 = z(1, m.hc * H), z(1, H)
        self.ch_tok, self.ch_sel, self.ch_pr = z(k, dt=torch.int32), z(k, dt=torch.int32), z(k)
        if self.ghead_draft is None:
            self._capture_head()                      # the multi-matrix descriptors
        g.mv("mtp.draft_out", torch.zeros(1, H, device=dv))
        ew, ety, ecols, erb = g.draft_emb
        nv = g.mat["mtp.draft_out"][2]
        torch.cuda.synchronize(dv)
        cap = torch.cuda.Stream(dv)
        pool = torch.cuda.graph_pool_handle()
        self.gchain = []
        keep = self.mtp_lg, self.mtp_hout             # the one-step head graphs' outputs (head_step returns them)
        self._in_pass = True
        try:
            with torch.cuda.device(dv):
                for j in range(k + 1):
                    gc = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(gc, pool=pool, stream=cap):
                        L = lib()
                        hp = self.hpos[dv]
                        for r in range(j):
                            self.mtp_hc.copy_(self.upd_hc[r:r + 1])
                            self.mtp_e.copy_(self.upd_e[r:r + 1])
                            self._head_body(False)
                            hp.add_(1)
                        self.mtp_hc.copy_(self.ch_hc0)
                        self.mtp_e.copy_(self.ch_e0)
                        for i in range(k):
                            self._head_body(True, "mtp.draft_out")
                            d_, st = _st(self.mtp_lg)
                            L.pick(d_, st, _p(self.mtp_lg), nv, _p(g.draft_map), _p(self.ch_tok[i:]),
                                   _p(self.ch_sel[i:]), _p(self.ch_pr[i:]))
                            hp.add_(1)
                            if i < k - 1:
                                self.mtp_hc.copy_(self.mtp_hout)
                                L.deq_row(d_, st, ety, _p(ew), ecols, erb, _p(self.ch_sel[i:]), _p(self.mtp_e))
                    self.gchain.append(gc)
            self.chain_k = k
        finally:
            self._in_pass = False
            self.mtp_lg, self.mtp_hout = keep

    def draft_chain(self, start, upd_hc, upd_toks, hc0, tok0, k):
        """the head's rows for the accepted places (upd_hc [j, 4H] on the card, upd_toks their tokens), from
        position start, then k drafts from streams hc0 and token tok0 -> (tokens, probabilities) as lists. The
        head's position ends at start + j + k (the drafting rows after start + j are written again later)."""
        m, dv = self.m, self.g.devices[-1]
        if getattr(self, "chain_k", None) != k:
            self._capture_chain(k)
        j = len(upd_toks)
        self.hpos[dv].fill_(start)
        if j:
            self.upd_hc[:j].copy_(upd_hc)
            self.upd_e[:j].copy_(self._erows("token_embd.weight", list(upd_toks)))
        self.ch_hc0.copy_(hc0)
        self.ch_e0.copy_(self._erows("token_embd.weight", [tok0]))
        self.gchain[j].replay()
        r = torch.cat([self.ch_tok.float(), self.ch_pr]).cpu()   # token ids < 2^24: exact as fp32
        self.hp = start + j + k
        return [int(t) for t in r[:k]], r[k:].tolist()

    def _capture_head(self):
        g, m = self.g, self.m
        dv = g.devices[-1]
        p = "blk.%d." % m.L
        with torch.cuda.device(dv):                  # the multi-matrix descriptors, made outside the capture
            x4, x1 = torch.zeros(1, m.hc * H, device=dv), torch.zeros(1, H, device=dv)
            for pre in ("hc_attn", "hc_ffn"):
                g.mv_multi([p + pre + "_down.weight", p + pre + "_inject.weight"], x4)
            g.mv_multi([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "indexer.k_proj.weight",
                        p + "indexer.q_proj.weight"], x1)
            g.mv_multi([p + "ffn_gate_inp.weight", p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight"], x1)
            torch.cuda.synchronize(dv)
        cap = torch.cuda.Stream(dv)
        self._in_pass = True
        try:
            with torch.cuda.device(dv):
                gd, gu = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(gd, pool=torch.cuda.graph_pool_handle(), stream=cap):
                    self._head_body(True)
                hout_d = self.mtp_hout
                with torch.cuda.graph(gu, pool=torch.cuda.graph_pool_handle(), stream=cap):
                    self._head_body(False)
            # each graph writes its own output streams: head_step returns the replayed graph's (the draft graph's
            # streams feed the next draft step)
            self._hout = {True: hout_d, False: self.mtp_hout}
            self.ghead_draft, self.ghead_upd = gd, gu
        finally:
            self._in_pass = False

    def load_head(self, hst):
        """the head's cache from its plain-path state (Model.mtp_logits over the prompt)"""
        m, l = self.m, self.m.L
        T = hst["pos"]
        self.kc[l].zero_(); self.vc[l].zero_(); self.rk[l].zero_(); self.bk[l].zero_()
        if l in hst["k"]:
            n = hst["k"][l].shape[0]
            self.kc[l][:n].copy_(hst["k"][l]); self.vc[l][:n].copy_(hst["v"][l]); self.rk[l][:n].copy_(hst["ik"][l])
            nb, ra = T // R.QSA_RATIO, R.QSA_RATIO
            if nb:
                p = "blk.%d." % l
                pooled = hst["ik"][l][:nb * ra].view(nb, ra, -1).float().mean(1)
                bk = self.g._rope(rms(pooled, self.g.vec[p + "indexer.k_norm.weight"], m.eps)[:, None, :],
                                  torch.arange(nb, device=pooled.device) * ra)[:, 0, :]
                self.bk[l][:nb].copy_(bk)
        self.hp = T
        if self.ghead_draft is None:
            self._capture_head()

    def head_step(self, hc, tok, logits=True):
        """one head step at the head's position hp: streams hc [1, 4H] (on the card), token tok -> (draft logits or
        None, the head's streams); the head's position moves one"""
        m, dv = self.m, self.g.devices[-1]
        self.hpos[dv].fill_(self.hp)
        if hc is not self.mtp_hc:
            self.mtp_hc.copy_(hc)
        self.mtp_e.copy_(self._erows("token_embd.weight", [tok]))
        (self.ghead_draft if logits else self.ghead_upd).replay()
        self.hp += 1
        return (self.mtp_lg if logits else None), self._hout[bool(logits)]

    # ------------------------------------------------------------ the pieces with static state
    def _gdn(self, l, x):
        g, m = self.g, self.m
        p = "blk.%d." % l
        nk, nv, dk = m.ssm_groups, m.ssm_heads, m.ssm_state
        dv = g.vec[p + "ssm_norm.weight"].numel()
        if self.fused:
            qkv, z, b, a = g.mv_multi([p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_beta.weight",
                                       p + "ssm_alpha.weight"], x)
        else:
            qkv = g.mv(p + "attn_qkv.weight", x)
            z = g.mv(p + "attn_gate.weight", x)
            b = g.mv(p + "ssm_beta.weight", x)
            a = g.mv(p + "ssm_alpha.weight", x)
        cw = g.vec[p + "ssm_conv1d.weight"]
        if self.fused and dk == 128 and dv == 128:
            return g.mv(p + "ssm_out.weight", self._gdn_rest(l, qkv, z, b, a))
        full = torch.cat([self.conv[l], qkv])
        self.conv[l].copy_(full[1:])
        qkv = F.silu((full * cw.T).sum(0, keepdim=True))           # the causal conv for the one new position
        q, kk, v = torch.split(qkv, [nk * dk, nk * dk, nv * dv], dim=-1)
        q = q.reshape(1, nk, dk)
        kk = kk.reshape(1, nk, dk)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(b).contiguous()
        gg = (g.vec[p + "ssm_a"] * F.softplus(a + g.vec[p + "ssm_dt.bias"])).contiguous()
        out = torch.empty(1, nv, dv, device=x.device)
        q, kk, v = q.contiguous(), kk.contiguous(), v.contiguous()
        S = self.S[l]
        lib().gdn(*_st(out), 1, nk, nv, _p(q), _p(kk), _p(v), _p(gg), _p(beta), _p(S), _p(out))
        o = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + m.eps) * g.vec[p + "ssm_norm.weight"]
        o = o * torch.sigmoid(z.view(1, nv, dv))
        return g.mv(p + "ssm_out.weight", o.reshape(1, -1))

    def _gdn_rest(self, l, qkv, z, b, a):
        """one token's Gated DeltaNet step after its projections (fused kernels): the conv and delta-rule states
        move one position; returns the gated, normed output [1, nv * dv] (before ssm_out)"""
        g, m = self.g, self.m
        p = "blk.%d." % l
        nk, nv = m.ssm_groups, m.ssm_heads
        dv = g.vec[p + "ssm_norm.weight"].numel()
        cw = g.vec[p + "ssm_conv1d.weight"]
        L = lib()
        d_, st = _st(qkv)
        C, kw = cw.shape
        y = torch.empty(C, device=qkv.device)
        L.conv1(d_, st, _p(qkv), _p(self.conv[l]), _p(cw), _p(y), C, kw)
        raw = torch.empty(nv * dv, device=qkv.device)
        L.gdn1(d_, st, nk, nv, _p(y), _p(b), _p(a), _p(g.vec[p + "ssm_a"]), _p(g.vec[p + "ssm_dt.bias"]),
               _p(self.S[l]), _p(raw))
        o = torch.empty(1, nv * dv, device=qkv.device)
        L.gnorm(d_, st, _p(raw), _p(z), _p(g.vec[p + "ssm_norm.weight"]), _p(o), nv, m.eps)
        return o

    def _attn(self, l, x):
        g, m = self.g, self.m
        p = "blk.%d." % l
        nh, nkv, hd = m.nh, m.nkv, m.hd
        pos = self.pos[x.device]
        if self.fused:
            qg, kx, vx, self._ik, self._iq = g.mv_multi(
                [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "indexer.k_proj.weight",
                 p + "indexer.q_proj.weight"], x)
        else:
            qg, kx, vx = g.mv(p + "attn_q.weight", x), g.mv(p + "attn_k.weight", x), g.mv(p + "attn_v.weight", x)
            self._ik = self._iq = None
        return g.mv(p + "attn_output.weight", self._attn_rest(l, x, qg, kx, vx, pos))

    def _qsa(self, l, qg, kx, vx, ik, iq, pos, T):
        """KINDI_QXG_QSAK=1: _attn_rest for T tokens at positions pos [T] on the k_qsa_* kernels (the same for one
        token and a window); returns o * sigmoid(gate) [T, nh * hd]"""
        g, m = self.g, self.m
        p = "blk.%d." % l
        nh, nkv, hd = m.nh, m.nkv, m.hd
        dv = qg.device
        di = self.rk[l].shape[1]
        nih = iq.shape[1] // di
        ra = R.QSA_RATIO
        nb = self.bk[l].shape[0]
        nk = R.QSA_TOKENS // ra
        S = min(nk, nb) * ra + ra - 1
        if not hasattr(g, "_inv") or dv not in g._inv:
            g._rope(torch.zeros(1, 1, m.rope_dim, device=dv), torch.zeros(1, dtype=torch.long, device=dv))
        inv = g._inv[dv]
        f32 = torch.float32
        qo = torch.empty((T, nh, hd), dtype=f32, device=dv)
        qi = torch.empty((T, nih, di), dtype=f32, device=dv)
        sc = torch.empty((T, nb), dtype=f32, device=dv)
        sel = torch.empty((T, S), dtype=torch.int32, device=dv)
        ok = torch.empty((T, S), dtype=torch.uint8, device=dv)
        part = torch.empty((T, nh, 16, hd + 2), dtype=f32, device=dv)
        out = torch.empty((T, nh * hd), dtype=f32, device=dv)
        qg, kx, vx, ik, iq = (t.contiguous() for t in (qg, kx, vx, ik, iq))
        lib().qsa(*_st(qg), T, _p(pos), _p(qg), _p(kx), _p(vx), _p(ik), _p(iq), _p(g.vec[p + "attn_q_norm.weight"]),
                  _p(g.vec[p + "attn_k_norm.weight"]), _p(g.vec[p + "indexer.q_norm.weight"]),
                  _p(g.vec[p + "indexer.k_norm.weight"]), _p(inv), m.rope_dim // 2, m.eps, nh, nkv, nih, hd, di, ra,
                  nk, nb, self.lmax, _p(self.kc[l]), _p(self.vc[l]), _p(self.rk[l]), _p(self.bk[l]), _p(qo), _p(qi),
                  _p(sc), _p(sel), _p(ok), _p(part), _p(out))
        return out

    def _attn_rest(self, l, x, qg, kx, vx, pos):
        """one token's attention after its projections: norms, RoPE, the caches at pos, the QSA selection, the
        attention; returns o * sigmoid(gate) [1, nh * hd] (before attn_output)"""
        g, m = self.g, self.m
        if lib().QSAK and self._ik is not None:
            return self._qsa(l, qg, kx, vx, self._ik, self._iq, pos.reshape(1), 1)
        p = "blk.%d." % l
        nh, nkv, hd = m.nh, m.nkv, m.hd
        qg = qg.view(1, nh, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:].reshape(1, nh * hd)
        q = rms(q, g.vec[p + "attn_q_norm.weight"], m.eps)
        k = rms(kx.view(1, nkv, hd), g.vec[p + "attn_k_norm.weight"], m.eps)
        v = vx.view(1, nkv, hd)
        pv = pos.view(1)
        q, k = g._rope(q, pv), g._rope(k, pv)
        self.kc[l].index_copy_(0, pv, k)
        self.vc[l].index_copy_(0, pv, v)
        sel, ok = self._select(l, x, pos, pv)                  # at most budget + ratio - 1 positions
        kk, vv = self.kc[l].index_select(0, sel), self.vc[l].index_select(0, sel)
        rep = nh // nkv
        qr = q.view(nkv, rep, hd)
        s = torch.einsum("grd,sgd->grs", qr, kk) / math.sqrt(hd)
        s = s.masked_fill(~ok[None, None, :], float("-inf"))
        o = torch.einsum("grs,sgd->grd", torch.softmax(s, -1), vv).reshape(1, -1)
        return o * torch.sigmoid(gate)

    def _select(self, l, x, pos, pv):
        """the QSA selection for the one new token, on the card (R.qsa_mask with the block keys kept up to date):
        the raw key goes into the cache, the block holding the position gets its key rewritten (it is used only
        once it is complete), the best budget / ratio visible blocks and the tail are kept. Returns the positions to
        attend (a fixed count: budget + ratio - 1, so attention costs the same at any context length) and which of
        them are real (a block not visible yet, or a tail slot past the position, is not)"""
        g, m = self.g, self.m
        p = "blk.%d." % l
        ra = R.QSA_RATIO
        dev = x.device
        if self._ik is not None:
            self.rk[l].index_copy_(0, pv, self._ik)
            di = self._ik.shape[1]
            qi = g._rope(rms(self._iq.view(1, -1, di), g.vec[p + "indexer.q_norm.weight"], m.eps), pv)[0]
        else:
            self.rk[l].index_copy_(0, pv, g.mv(p + "indexer.k_proj.weight", x))
            qi = g._index_q(l, x, pv)[0]                               # [heads, d]
        b = torch.div(pos, ra, rounding_mode="floor")
        pooled = self.rk[l].index_select(0, b * ra + torch.arange(ra, device=dev)).mean(0, keepdim=True)
        bkey = g._rope(rms(pooled, g.vec[p + "indexer.k_norm.weight"], m.eps)[:, None, :], (b * ra).view(1))[:, 0, :]
        self.bk[l].index_copy_(0, b.view(1), bkey)
        nb = self.bk[l].shape[0]
        nvis = torch.div(pos + 1, ra, rounding_mode="floor")
        sc = torch.relu(qi @ self.bk[l].T).sum(0) / math.sqrt(qi.shape[-1])
        sc = sc.masked_fill(torch.arange(nb, device=dev) >= nvis, float("-inf"))
        top = sc.topk(min(R.QSA_TOKENS // ra, nb))
        ar = torch.arange(ra, device=dev)
        toks = (top.indices[:, None] * ra + ar[None, :]).reshape(-1)
        tok_ok = (top.values > float("-inf")).repeat_interleave(ra)
        tail = nvis * ra + ar[:ra - 1]
        sel = torch.cat([toks, tail]).clamp(max=self.lmax - 1)
        ok = torch.cat([tok_ok, tail <= pos])
        return sel, ok

    def _ple(self, l, h):
        g, m = self.g, self.m
        p = "blk.%d.ple_" % l
        emb = self.ple_emb
        return self._ple_rest(l, h, g.mv(p + "key.weight", emb), g.mv(p + "value.weight", emb))

    def _ple_rest(self, l, h, kraw, val):
        """one token's n-gram embedding step after its two projections (the conv state moves one position)"""
        g, m = self.g, self.m
        p = "blk.%d.ple_" % l
        key = rms(kraw, g.vec[p + "norm_key.weight"], m.eps, group=H).view(1, m.hc, H)
        qry = rms(h, g.vec[p + "norm_query.weight"], m.eps, group=H).view(1, m.hc, H)
        gate = (key * qry).sum(-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = (torch.sigmoid(gate) * val[:, None, :]).reshape(1, -1)
        gvn = rms(gv, g.vec[p + "norm_conv.weight"], m.eps, group=H)
        cw = g.vec[p + "conv1d.weight"]                                # (channels, k), taps every ple_n rows
        full = torch.cat([self.plec[l], gvn])
        self.plec[l].copy_(full[1:])
        taps = full[torch.arange(0, full.shape[0], m.ple_n, device=h.device)]
        return gv + F.silu((taps * cw.T).sum(0, keepdim=True))

    def _mixer(self, h, pre, inject=True):
        """the gated-residual mixer (Model.mixer) with the fused kernels"""
        g, m = self.g, self.m
        if not self.fused:
            return g.mixer(h, pre, inject)
        L = lib()
        d_, st = _st(h)
        hn = torch.empty_like(h)
        L.rms_g(d_, st, _p(h), _p(g.vec[pre + "_norm.weight"]), _p(hn), m.hc, H, m.eps)
        if inject:
            dd, j = g.mv_multi([pre + "_down.weight", pre + "_inject.weight"], hn)
        else:
            dd = g.mv(pre + "_down.weight", hn)
        L.silu_s(d_, st, _p(dd), dd.numel(), 1.0 / m.hc)
        u = g.mv(pre + "_up.weight", dd)
        x = torch.empty(1, H, device=h.device)
        if not inject:
            L.mix_out(d_, st, _p(u), _p(hn), 0, _p(x), 0, m.hc, H, m.hc)
            return x
        wr = torch.empty(1, m.hc, device=h.device)
        L.mix_out(d_, st, _p(u), _p(hn), _p(j), _p(x), _p(wr), m.hc, H, m.hc)
        return x, h, wr

    def _write(self, hin, y, wr, y2=None, sdot=None):
        """the residual write (Model.write), with an optional second term sigmoid(sdot) * y2 (the shared expert)"""
        if not self.fused:
            if y2 is not None:
                y = y + torch.sigmoid(sdot)[:, None] * y2
            return R.Model.write(hin, y, wr)
        out = torch.empty_like(hin)
        lib().write(*_st(hin), _p(hin), _p(y), _p(y2) if y2 is not None else 0, _p(sdot) if sdot is not None else 0,
                    _p(wr), _p(out), self.m.hc, H)
        return out

    def _pre(self, l, h):
        """layer l up to its routed experts: the mixers, attention or Gated DeltaNet, the router (_pre_a);
        then _pre_b: the card's hot experts and the shared expert"""
        self._pre_a(l, h)
        self._pre_b(l)

    def _pre_a(self, l, h):
        g, m = self.g, self.m
        L = lib()
        if l in m.ple_layers:
            h = h + self._ple(l, h)
        x, hin, wr = self._mixer(h, "blk.%d.hc_attn" % l)
        y = self._attn(l, x) if (l + 1) % m.interval == 0 or l == m.L else self._gdn(l, x)
        h = self._write(hin, y, wr)
        x, hin, wr = self._mixer(h, "blk.%d.hc_ffn" % l)
        p = "blk.%d." % l
        if self.fused:
            logit, gs, us = g.mv_multi([p + "ffn_gate_inp.weight", p + "ffn_gate_shexp.weight",
                                        p + "ffn_up_shexp.weight"], x)
            L.router(*_st(x), _p(logit), logit.shape[1], m.n_used, _p(self.idx[l]), _p(self.top[l]))
        else:
            probs = torch.softmax(g.mv(p + "ffn_gate_inp.weight", x), dim=-1)
            top, idx = probs.topk(m.n_used, dim=-1)
            top = top / top.sum(-1, keepdim=True)
            self.idx[l].copy_(idx[0].int())
            self.top[l].copy_(top[0])
        self.hin2[l], self.wr2[l] = hin, wr
        self.x2[l] = x
        if self.fused:
            self._gsus[l] = (gs, us)
        if getattr(self, "_in_pass", False):
            return                            # the pass graph posts them to the mailbox instead
        # the picks and the expert input to the host (for the CPU experts): first, so the CPU can start while the
        # card does the hot experts and the shared expert (_pre_b)
        self.idx_h[l].copy_(self.idx[l], non_blocking=True)
        self.x_h[l].copy_(x, non_blocking=True)
        if self.coldsum:
            self.top_h[l].copy_(self.top[l], non_blocking=True)

    def _pre_b(self, l):
        """the card's part of layer l's routed experts (its hot experts) and the shared expert"""
        g, m = self.g, self.m
        L = lib()
        p = "blk.%d." % l
        x = self.x2[l]
        if self.fused:
            gs, us = self._gsus[l]
        # the card's hot experts, picked on the card
        if l in g.hot:
            K, Fd = H, m.w.t[p + "ffn_gate_exps.weight"][2][1]
            gty, dty = m.w.t[p + "ffn_gate_exps.weight"][3], m.w.t[p + "ffn_down_exps.weight"][3]
            xc = x.contiguous()
            xq = torch.empty((1, K), dtype=torch.int8, device=x.device)
            xd = torch.empty((1, K // 256), dtype=torch.float32, device=x.device)
            hh = torch.empty((self.nsel, Fd), dtype=torch.float32, device=x.device)
            hq = torch.empty((self.nsel, Fd), dtype=torch.int8, device=x.device)
            hd = torch.empty((self.nsel, Fd // L.DN), dtype=torch.float32, device=x.device)
            d_, st = _st(xc)
            buf = g.hot[l]
            L.quant256(d_, st, _p(xc), K, _p(xq), _p(xd))
            L.gate_up_slots(d_, st, gty, K, Fd, self.nsel, _p(self.idx[l]), _p(self.slotmap[l]), _p(buf),
                            buf.shape[1], _p(xq), _p(xd), _p(hh))
            L.quant_dn(d_, st, _p(hh), self.nsel * Fd, _p(hq), _p(hd))
            self._xb[l] = (xq, xd, hh, hq, hd, self.idx[l], self.hotp[l], self.nsel, self.nsel)
            L.down_slots(d_, st, gty, dty, K, Fd, self.nsel, _p(self.idx[l]), _p(self.slotmap[l]), _p(buf),
                         buf.shape[1], _p(hq), _p(hd), _p(self.hotp[l]))
        if not self.fused:
            gs, us = g.mv(p + "ffn_gate_shexp.weight", x), g.mv(p + "ffn_up_shexp.weight", x)
        if self.fused:
            hs = torch.empty_like(gs)
            L.swiglu(*_st(gs), _p(gs), _p(us), _p(hs), gs.numel())
            self.sh[l] = g.mv(p + "ffn_down_shexp.weight", hs)
            self.sdot[l] = torch.empty(1, device=x.device)
            L.dot(*_st(x), _p(x), _p(g.vec[p + "ffn_gate_inp_shexp.weight"]), H, _p(self.sdot[l]))
        else:
            self.sh[l] = g.mv(p + "ffn_down_shexp.weight", F.silu(gs) * us)
            self.sdot[l] = (x @ g.vec[p + "ffn_gate_inp_shexp.weight"]).view(1)

    def _post(self, l):
        dv = self.idx[l].device
        out = torch.empty((1, H), dtype=torch.float32, device=dv)
        if self.coldsum:
            lib().combine_hot(*_st(out), self.nsel, H, _p(self.top[l]), _p(self.idx[l]), _p(self.slotmap[l]),
                              _p(self.hotp[l]), _p(self.csum[l]), _p(out))
        else:
            lib().combine_sel(*_st(out), self.nsel, H, _p(self.top[l]), _p(self.idx[l]), _p(self.slotmap[l]),
                              _p(self.hotp[l]), _p(self.coldp[l]), _p(out))
        return self._write(self.hin2[l], out, self.wr2[l], y2=self.sh[l], sdot=self.sdot[l])

    # ------------------------------------------------------------ capture and run
    def _capture(self):
        g, m = self.g, self.m
        self.graphs, self.hout = [], {}
        # one memory pool per card for all the graphs: they replay in the order they were captured, so the
        # temporaries of one graph can reuse those of the previous ones (separate pools took gigabytes)
        pools, cap = {}, {}
        h = None
        for l in range(m.L):
            dv = g.devices[g.dev_of[l]]
            with torch.cuda.device(dv):
                if self.hin[l] is not None:
                    h = self.hin[l]
                s = torch.cuda.Stream(dv)
                s.wait_stream(torch.cuda.current_stream(dv))
                with torch.cuda.stream(s):                  # warm-up outside the graph (allocations, autotuning)
                    self._pre(l, h)
                    self._post(l)
                torch.cuda.current_stream(dv).wait_stream(s)
                gp, gb, gq = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                if dv not in pools:
                    # a pool and a capture stream per card (torch's default capture stream lives on one card)
                    pools[dv] = torch.cuda.graph_pool_handle()
                    cap[dv] = torch.cuda.Stream(dv)
                with torch.cuda.graph(gp, pool=pools[dv], stream=cap[dv]):
                    self._pre_a(l, h)
                with torch.cuda.graph(gb, pool=pools[dv], stream=cap[dv]):
                    self._pre_b(l)
                with torch.cuda.graph(gq, pool=pools[dv], stream=cap[dv]):
                    hout = self._post(l)
                self.hout[l] = hout
                self.graphs.append((dv, gp, gb, gq))
                self.ev[l] = torch.cuda.Event()
                h = hout
        dv = g.devices[-1]
        with torch.cuda.device(dv):
            hl = self.hout[m.L - 1]
            self.hhead = hl if g.dev_of[m.L - 1] == len(g.devices) - 1 else torch.zeros_like(hl, device=dv)
            gh = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gh, pool=pools[dv], stream=cap[dv]):
                x = self._mixer(self.hhead, "output_hc", inject=False)
                self.logits = g.mv("output.weight", x)
            self.ghead = gh
        if self.passmode:
            self._capture_pass(pools, cap)

    def _capture_pass(self, pools, cap):
        """the pass graph: the pass number, then per layer the mixers and attention or Gated DeltaNet and the
        router (_pre_a), the card's experts and the shared expert (_pre_b), the mailbox post and wait, the combine and
        write (_post), then the head"""
        from kindi import qx_cpu
        g, m = self.g, self.m
        dv = g.devices[0]
        C = qx_cpu.lib()
        off = (ctypes.c_long * 4)()
        C.ma_box_layout(1, self.nsel, H * 4, H * 4, 1, off)
        self.box_off = list(off)
        self.box_h, self.box_d = lib().host_mapped(dv.index, self.box_off[3] * m.L)
        self.seq_d = torch.zeros(1, dtype=torch.int32, device=dv)
        self.seq_h = 0
        self.ma_handles = np.array([g.cpu.layer[l] for l in range(m.L)], np.int32)
        self.ma_oncard = np.ascontiguousarray(np.stack([g.slot[l] >= 0 for l in range(m.L)]).astype(np.uint8))
        self.ma_ctx = C.qx_ma_ctx_new(m.L, m.n_exp, self.nsel, self.ma_handles.ctypes.data, self.ma_oncard.ctypes.data)
        self.ma_fn = C.qx_ma_fn_ptr()
        if FAST["tail"] > 0:
            C.qx_ma_ctx_set_tail(self.ma_ctx, FAST["tail"], H)
        caches = [c for c in g.cache_of.values() if c.adapt is not None]
        if caches:                                        # the mailbox step notes the picks itself (no Python)
            C.qx_ma_ctx_set_note(self.ma_ctx, caches[0].note_table(m.L))
            self.ma_noted = True
        self._fetch_setup(dv)
        self._in_pass = True
        try:
            with torch.cuda.device(dv):
                gpass = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gpass, pool=torch.cuda.graph_pool_handle(), stream=cap[dv]):
                    L = lib()
                    d_, st = _st(self.seq_d)
                    L.ma_seq(d_, st, _p(self.seq_d))
                    h = self.hin[0]
                    for l in range(m.L):
                        self._pre_a(l, h)
                        box = self.box_d + l * self.box_off[3]
                        # posted right after the router: the CPU starts while the card does its own experts
                        L.ma_post(d_, st, box, _p(self.seq_d), _p(self.idx[l]), _p(self.x2[l]), 1, self.nsel, H,
                                  self.box_off[1])
                        self._fetch(l, self.idx[l], self.nsel, d_, st)
                        self._pre_b(l)
                        self._fetch_join(l, d_, st)
                        L.ma_wait(d_, st, box, _p(self.seq_d), _p(self.coldp[l]), _p(self.idx[l]),
                                  _p(self.slotmap[l]), 1, self.nsel, H, self.box_off[2])
                        h = self._post(l)
                        self._fetch_reset(l, d_, st)
                    x = self._mixer(h, "output_hc", inject=False)
                    self.logits_pass = g.mv("output.weight", x)
                self.gpass = gpass
        finally:
            self._in_pass = False

    # ------------------------------------------------------------ the PCIe share of the misses (item 7)
    # KINDI_QXG_FETCH = S > 0: in the mailbox pass and the windows, right after a layer's picks are posted, the card
    # copies up to S of the layer's missed experts from the page-locked expert pages (KINDI_QXG_PIN's mapping) into
    # the spare rows of the layer's class buffer and computes them with its cached ones; the CPU skips them (the
    # same rule in k_fetch_plan and qx_ma_fn). The share per layer, nf[l] <= S, lives in mapped host memory and
    # changes between passes only: set_fetch_share() (KINDI_QXG_FETCH_N, default S). Exact: card rows == CPU rows.
    def set_tail(self, r):
        """the fast mode's tail skip for this decoder's mailbox pass (rank >= r dropped; 0 off). The per-layer path
        reads FAST["tail"] at every step."""
        if getattr(self, "ma_ctx", None) is not None:
            from kindi import qx_cpu
            qx_cpu.lib().qx_ma_ctx_set_tail(self.ma_ctx, int(r), H)

    def _fetch_setup(self, dv):
        from kindi import qx_cpu
        g, m = self.g, self.m
        self.fetch = None
        S = int(os.environ.get("KINDI_QXG_FETCH", "0") or 0)
        if S <= 0:
            return
        cc = [c for c in {id(c): c for c in g.cache_of.values()}.values()]
        if len(cc) != 1 or len(g.devices) != 1 or cc[0].nspare <= 0:
            g.log("KINDI_QXG_FETCH: needs one card and its cache made with the spare rows; off")
            return
        cc = cc[0]
        if not getattr(cc, "pinned", False):
            cc.pinned = cc._pin()
        src = {l: cc.device_src(l) for l in range(m.L)}
        if not cc.pinned or any(v is None for v in src.values()):
            g.log("KINDI_QXG_FETCH: the expert pages are not page-locked and mapped; off")
            return
        S = cc.nspare
        nf_h, nf_d = lib().host_mapped(dv.index, 4 * m.L)
        nf = np.ctypeslib.as_array(ctypes.cast(nf_h, ctypes.POINTER(ctypes.c_int32)), (m.L,))
        lst = torch.zeros(S + 1, dtype=torch.int32, device=dv)
        self.fetch = {"S": S, "cc": cc, "src": src, "nf": nf, "nf_d": nf_d, "lst": lst}
        qx_cpu.lib().qx_ma_ctx_set_fetch(self.ma_ctx, nf_h, S)
        # the share: a fixed count a layer (default 1: B2 showed the CPU's time per layer is mostly fixed, so each
        # fetched expert saves less than its copy costs past the first one or two), or auto (the rate model)
        mode = os.environ.get("KINDI_QXG_FETCH_N", "1")
        if mode == "auto":
            # the share per layer from the two rates: the link's (timed here: one copy from the page-locked pages)
            # and the CPU's per missed expert (qx_stats, step by step); a layer's misses split so both finish together
            f = self.fetch
            f["auto"], f["miss"], f["cpu_ms"], f["qx"] = True, np.zeros(m.L), None, None
            lo, (sz, dptr) = next(iter(cc._addr.items()))
            nb = min(sz, 64 << 20)
            tmp = torch.empty(nb, dtype=torch.uint8, device=dv)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ms = []
            for _ in range(3):
                e0.record()
                lib().cu.cuMemcpyHtoDAsync_v2(ctypes.c_uint64(tmp.data_ptr()), ctypes.c_void_p(lo), ctypes.c_size_t(nb),
                                              ctypes.c_void_p(torch.cuda.current_stream(dv).cuda_stream))
                e1.record()
                e1.synchronize()
                ms.append(e0.elapsed_time(e1))
            del tmp
            f["gbs"] = nb / (min(ms) / 1e3) / 1e9
            f["link_ms"] = np.array([sum(f["src"][l][1]) / (f["gbs"] * 1e6) for l in range(m.L)])
            self.set_fetch_share(0)
            g.log("KINDI_QXG_FETCH: %d spare slots a class, share auto (link %.1f GB/s, %.2f ms an expert)" % (
                S, f["gbs"], float(f["link_ms"].mean())))
        else:
            self.set_fetch_share(int(mode))
            g.log("KINDI_QXG_FETCH: %d spare slots a class, share %s experts a layer" % (S, int(nf[0])))

    def _fetch_tune(self, picks_of, T):
        """auto share, after a pass or window: the missed experts per layer (moving average), the CPU's ms per
        expert from this step, then nf[l] = misses * cpu / (cpu + link)"""
        f = getattr(self, "fetch", None)
        if f is None or not f.get("auto"):
            return
        from kindi import qx_cpu
        m = self.m
        P = np.stack([np.asarray(picks_of(l)) for l in range(m.L)])          # [L, T * nsel]
        miss = self.ma_oncard[np.arange(m.L)[:, None], P] == 0
        Ps = np.where(miss, P, -1)
        Ps.sort(axis=1)
        distinct = ((Ps[:, 1:] != Ps[:, :-1]) & (Ps[:, 1:] >= 0)).sum(1) + (Ps[:, 0] >= 0)
        f["miss"] = 0.8 * f["miss"] + 0.2 * distinct
        a, sec = (ctypes.c_longlong * 4)(), ctypes.c_double()
        qx_cpu.lib().qx_stats(a, ctypes.byref(sec))
        now = (a[1], sec.value)
        if f["qx"] is not None and now[0] > f["qx"][0]:
            c = 1e3 * (now[1] - f["qx"][1]) / (now[0] - f["qx"][0])
            f["cpu_ms"] = c if f["cpu_ms"] is None else 0.7 * f["cpu_ms"] + 0.3 * c
        f["qx"] = now
        if f["cpu_ms"] is not None:
            c = f["cpu_ms"]
            self.set_fetch_share(np.rint(f["miss"] * c / (c + f["link_ms"])).astype(np.int32))

    def set_fetch_share(self, n):
        """missed experts a layer the card fetches (an int, or one per layer); only between passes"""
        f = getattr(self, "fetch", None)
        if f is None:
            return
        f["nf"][:] = np.clip(np.asarray(n, np.int32), 0, f["S"])

    def fetch_stats(self):
        """picks, picks on the card's cache, picks of fetched experts, fetched experts since the last call"""
        from kindi import qx_cpu
        if getattr(self, "fetch", None) is None or getattr(self, "ma_ctx", None) is None:
            return None
        a = (ctypes.c_longlong * 4)()
        qx_cpu.lib().qx_ma_ctx_stats(self.ma_ctx, a)
        return {"picks": a[0], "cache": a[1], "fetched_picks": a[2], "fetched": a[3]}

    def _fetch(self, l, idx, n, d_, st):
        """the plan and the copies on the side stream, beside the card's own experts and the shared expert (the
        slot table is not touched until _fetch_join)"""
        f = getattr(self, "fetch", None)
        if f is None:
            return
        cc, L = f["cc"], lib()
        ci = cc.cls_of[l]
        s3, nb = f["src"][l]
        dv = self.g.devices[0]
        main = torch.cuda.current_stream(dv)
        side = f.setdefault("side", torch.cuda.Stream(dv))
        ea = torch.cuda.Event()
        ea.record(main)
        side.wait_event(ea)
        ss = side.cuda_stream
        L.fetch_plan(d_, ss, _p(idx), n, _p(self.slotmap[l]), f["nf_d"], l, f["S"], cc.spare0[ci], _p(f["lst"]))
        L.fetch_copy(d_, ss, _p(f["lst"]), f["S"], s3, nb, _p(cc.buf[ci]), cc.buf[ci].shape[1], cc.spare0[ci])
        f["landed"] = torch.cuda.Event()
        f["landed"].record(side)

    def _fetch_join(self, l, d_, st):
        """after the card's own experts: wait for the copies, mark the fetched experts in the slot table, and run
        gate/up and down for their picks only (their spare rows); the picks' quantized input is the expert pass's"""
        f = getattr(self, "fetch", None)
        if f is None or l not in self.g.hot:
            return
        cc, L, m = f["cc"], lib(), self.m
        ci = cc.cls_of[l]
        sp0 = cc.spare0[ci]
        torch.cuda.current_stream(self.g.devices[0]).wait_event(f["landed"])
        L.fetch_apply(d_, st, _p(f["lst"]), f["S"], _p(self.slotmap[l]), sp0)
        xq, xd, hh, hq, hd, idx, hotp, n, tok_nsel = self._xb[l]
        p = "blk.%d." % l
        K, Fd = H, m.w.t[p + "ffn_gate_exps.weight"][2][1]
        gty, dty = m.w.t[p + "ffn_gate_exps.weight"][3], m.w.t[p + "ffn_down_exps.weight"][3]
        buf = self.g.hot[l]
        L.gate_up_slots(d_, st, gty, K, Fd, n, _p(idx), _p(self.slotmap[l]), _p(buf), buf.shape[1], _p(xq), _p(xd),
                        _p(hh), tok_nsel=tok_nsel, min_slot=sp0)
        L.quant_dn(d_, st, _p(hh), n * Fd, _p(hq), _p(hd))          # the same rows again, and the fetched ones
        L.down_slots(d_, st, gty, dty, K, Fd, n, _p(idx), _p(self.slotmap[l]), _p(buf), buf.shape[1], _p(hq), _p(hd),
                     _p(hotp), min_slot=sp0)

    def _fetch_reset(self, l, d_, st):
        f = getattr(self, "fetch", None)
        if f is not None:
            lib().fetch_reset(d_, st, _p(f["lst"]), f["S"], _p(self.slotmap[l]))

    # ------------------------------------------------------------ the multi-token window (the drafter's check)
    # T tokens through the 48 layers in one graph (one card, the mailbox pass): every matrix is read once for the
    # T tokens (k_dense8T / k_dense8mT: each token's row with the bits of a one-token launch), and every step that
    # carries state (the n-gram conv, the attention caches, the QSA blocks, the Gated DeltaNet conv and delta rule)
    # runs token after token with the one-token code. So token t of a window has the bits of the t-th of T one-token
    # steps. The conv and delta-rule states after each token are kept, so a window can be cut back to its first
    # n tokens (refused drafts).
    def mvT(self, name, X):
        g = self.g
        w, ty, rows, cols = g.mat[name]
        T = X.shape[0]
        Y = torch.empty((T, rows), dtype=torch.float32, device=X.device)
        if lib().tc_on(X.device.index, X.shape[0]):
            lib().dense_tc(*_st(X), ty, _p(w), rows, cols, Q.nbytes(ty, cols), _p(X.contiguous()), T, _p(Y))
        else:
            lib().dense8T(*_st(X), ty, _p(w), rows, cols, Q.nbytes(ty, cols), _p(X.contiguous()), T, _p(Y))
        return Y

    def mv_multiT(self, names, X):
        self._prep_multiT(names, X.shape[0], X.device)
        tc = lib().tc_on(X.device.index, X.shape[0])
        outs, desc, nblocks = self._mvmT[(tuple(names), X.shape[0], X.device, tc)][:3]
        if tc:
            lib().dense_tcm(*_st(X), _p(desc), len(names), nblocks, _p(X.contiguous()), X.shape[0])
        else:
            lib().dense8mT(*_st(X), _p(desc), len(names), nblocks, _p(X.contiguous()), X.shape[0], X.shape[1])
        return outs

    def _prep_multiT(self, names, T, device):
        """the outputs and the descriptor of a multi-matrix launch for T tokens (made before a graph capture)"""
        from kindi.qx_gpu import Kernels
        g = self.g
        cache = self.__dict__.setdefault("_mvmT", {})
        tc = lib().tc_on(device.index, T)
        key = (tuple(names), T, device, tc)
        X = torch.empty(1, device=device)
        if key not in cache:
            outs, rows8, first = [], [], 0
            for n in names:
                w, ty, rows, cols = g.mat[n]
                split, rpb = (0, 16) if tc else Kernels.planT(rows, cols, T)
                y = torch.empty((T, rows), dtype=torch.float32, device=X.device)
                outs.append(y)
                rows8 += [w.data_ptr(), y.data_ptr(), Q.nbytes(ty, cols), ty, rows, cols, split, first]
                first += (rows + rpb - 1) // rpb
            cache[key] = (outs, torch.tensor(rows8, dtype=torch.int64).to(X.device), first)

    def _mixerT(self, h, pre, inject=True):
        g, m = self.g, self.m
        L = lib()
        d_, st = _st(h)
        T = h.shape[0]
        hn = torch.empty_like(h)
        L.rms_gT(d_, st, _p(h), _p(g.vec[pre + "_norm.weight"]), _p(hn), m.hc, H, m.eps, T)
        if inject:
            dd, j = self.mv_multiT([pre + "_down.weight", pre + "_inject.weight"], hn)
        else:
            dd = self.mvT(pre + "_down.weight", hn)
        L.silu_s(d_, st, _p(dd), dd.numel(), 1.0 / m.hc)
        u = self.mvT(pre + "_up.weight", dd)
        x = torch.empty(T, H, device=h.device)
        if not inject:
            L.mix_outT(d_, st, _p(u), _p(hn), 0, _p(x), 0, m.hc, H, m.hc, T, u.shape[1], hn.shape[1], 0, H, 0)
            return x
        wr = torch.empty(T, m.hc, device=h.device)
        L.mix_outT(d_, st, _p(u), _p(hn), _p(j), _p(x), _p(wr), m.hc, H, m.hc, T, u.shape[1], hn.shape[1], j.shape[1],
                   H, m.hc)
        return x, h, wr

    def _writeT(self, hin, y, wr, y2=None, sdot=None):
        out = torch.empty_like(hin)
        d_, st = _st(hin)
        lib().writeT(d_, st, _p(hin), _p(y), _p(y2) if y2 is not None else 0, _p(sdot) if sdot is not None else 0,
                     _p(wr), _p(out), self.m.hc, H, hin.shape[0], hin.shape[1], H, self.m.hc)
        return out

    def _pre_aT(self, l, h):
        g, m = self.g, self.m
        L = lib()
        p = "blk.%d." % l
        T = h.shape[0]
        if l in m.ple_layers:
            # one launch for both (the same input; each row with the bits of its own launch)
            kraw, val = self.mv_multiT(["blk.%d.ple_key.weight" % l, "blk.%d.ple_value.weight" % l], self.pleT)
            rows = []
            for t in range(T):
                rows.append(self._ple_rest(l, h[t:t + 1], kraw[t:t + 1], val[t:t + 1]))
                self.plec_hist[l][t].copy_(self.plec[l])
            h = h + torch.cat(rows)
        x, hin, wr = self._mixerT(h, "blk.%d.hc_attn" % l)
        outs = []
        dv = x.device
        if (l + 1) % m.interval == 0:
            qg, kx, vx, ik, iq = self.mv_multiT(
                [p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight", p + "indexer.k_proj.weight",
                 p + "indexer.q_proj.weight"], x)
            if lib().QSAK:
                y = self.mvT(p + "attn_output.weight", self._qsa(l, qg, kx, vx, ik, iq, self.posw[dv], T))
            else:
                for t in range(T):
                    self._ik, self._iq = ik[t:t + 1], iq[t:t + 1]
                    outs.append(self._attn_rest(l, x[t:t + 1], qg[t:t + 1], kx[t:t + 1], vx[t:t + 1],
                                                self.posw[dv][t]))
                y = self.mvT(p + "attn_output.weight", torch.cat(outs))
        else:
            qkv, z, b, a = self.mv_multiT([p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_beta.weight",
                                           p + "ssm_alpha.weight"], x)
            # the T tokens' conv and delta rule in one launch each, in token order, keeping the states after each
            nk, nv = m.ssm_groups, m.ssm_heads
            dvv = g.vec[p + "ssm_norm.weight"].numel()
            cw = g.vec[p + "ssm_conv1d.weight"]
            C, kw = cw.shape
            d_, st = _st(x)
            yc = torch.empty(T, C, device=dv)
            L.conv1T(d_, st, _p(qkv), _p(self.conv[l]), _p(cw), _p(yc), C, kw, T, qkv.shape[1], _p(self.conv_hist[l]))
            raw = torch.empty(T, nv * dvv, device=dv)
            L.gdn1T(d_, st, nk, nv, _p(yc), _p(b), _p(a), _p(g.vec[p + "ssm_a"]), _p(g.vec[p + "ssm_dt.bias"]),
                    _p(self.S[l]), _p(raw), T, C, b.shape[1], nv * dvv, _p(self.S_hist[l]))
            o = torch.empty(T, nv * dvv, device=dv)
            L.gnormT(d_, st, _p(raw), _p(z), _p(g.vec[p + "ssm_norm.weight"]), _p(o), nv, m.eps, T, nv * dvv,
                     z.shape[1])
            y = self.mvT(p + "ssm_out.weight", o)
        h = self._writeT(hin, y, wr)
        x, hin, wr = self._mixerT(h, "blk.%d.hc_ffn" % l)
        logit, gs, us = self.mv_multiT([p + "ffn_gate_inp.weight", p + "ffn_gate_shexp.weight",
                                        p + "ffn_up_shexp.weight"], x)
        d_, st = _st(x)
        L.routerT(d_, st, _p(logit), logit.shape[1], m.n_used, _p(self.idxT[l]), _p(self.topT[l]), T)
        self.hin2T[l], self.wr2T[l], self.x2T[l], self.gsusT[l] = hin, wr, x, (gs, us)

    def _pre_bT(self, l):
        g, m = self.g, self.m
        L = lib()
        p = "blk.%d." % l
        x = self.x2T[l]
        T = x.shape[0]
        gs, us = self.gsusT[l]
        n = T * self.nsel
        if l in g.hot:
            K, Fd = H, m.w.t[p + "ffn_gate_exps.weight"][2][1]
            gty, dty = m.w.t[p + "ffn_gate_exps.weight"][3], m.w.t[p + "ffn_down_exps.weight"][3]
            xc = x.contiguous()
            xq = torch.empty((T, K), dtype=torch.int8, device=x.device)
            xd = torch.empty((T, K // 256), dtype=torch.float32, device=x.device)
            hh = torch.empty((n, Fd), dtype=torch.float32, device=x.device)
            hq = torch.empty((n, Fd), dtype=torch.int8, device=x.device)
            hd = torch.empty((n, Fd // L.DN), dtype=torch.float32, device=x.device)
            d_, st = _st(xc)
            buf = g.hot[l]
            L.quant256(d_, st, _p(xc), T * K, _p(xq), _p(xd))
            L.gate_up_slots(d_, st, gty, K, Fd, n, _p(self.idxT[l]), _p(self.slotmap[l]), _p(buf),
                            buf.shape[1], _p(xq), _p(xd), _p(hh), tok_nsel=self.nsel)
            L.quant_dn(d_, st, _p(hh), n * Fd, _p(hq), _p(hd))
            self._xb[l] = (xq, xd, hh, hq, hd, self.idxT[l], self.hotpT[l], n, self.nsel)
            L.down_slots(d_, st, gty, dty, K, Fd, n, _p(self.idxT[l]), _p(self.slotmap[l]), _p(buf),
                         buf.shape[1], _p(hq), _p(hd), _p(self.hotpT[l]))
        hs = torch.empty_like(gs)
        L.swiglu(*_st(gs), _p(gs), _p(us), _p(hs), gs.numel())
        self.shT[l] = self.mvT(p + "ffn_down_shexp.weight", hs)
        self.sdotT[l] = torch.empty(T, device=x.device)
        L.dotT(*_st(x), _p(x), _p(g.vec[p + "ffn_gate_inp_shexp.weight"]), H, _p(self.sdotT[l]), T, H)

    def _postT(self, l):
        T = self.x2T[l].shape[0]
        dv = self.x2T[l].device
        out = torch.empty((T, H), dtype=torch.float32, device=dv)
        d_, st = _st(out)
        lib().combine_selT(d_, st, self.nsel, H, _p(self.topT[l]), _p(self.idxT[l]), _p(self.slotmap[l]),
                           _p(self.hotpT[l]), _p(self.coldpT[l]), _p(out), T)
        return self._writeT(self.hin2T[l], out, self.wr2T[l], y2=self.shT[l], sdot=self.sdotT[l])

    def _erows(self, name, ids):
        """a decode step's embedding rows on the card (KINDI_QXG_EMB_STEP, 1: the raw rows gathered on the host and
        decoded by k_deq, bit-equal to the host decode, which took about 0.25 ms a token row on the 3060 host and
        sat on every window's host prep; 0: the host decode)"""
        g = self.g
        if os.environ.get("KINDI_QXG_EMB_STEP", "1") == "1":
            return g.emb_rows(ids, g.devices[-1], name, min_rows=1)
        return g.ref.w.rows(name, list(ids)).to(g.devices[-1])

    def free_windows(self):
        """the window buffers and graphs back to the driver (captured again on the next window)"""
        if getattr(self, "_wbuf", None) is None and not getattr(self, "_wins", None):
            return
        self._wbuf = self._wins = self.gwin = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _capture_window(self, T):
        """the window graph for T tokens (needs the mailbox pass: KINDI_QXG_PASS=1)"""
        from kindi import qx_cpu
        g, m = self.g, self.m
        if not self.passmode:
            raise RuntimeError("the multi-token window needs KINDI_QXG_PASS=1 (one card, in-process CPU experts)")
        dv = g.devices[0]
        if getattr(self, "_wbuf", None) is not None and self.boxT_max < T:   # a longer window: new buffers, new graphs
            self._wbuf = self._wins = self.gwin = None
            torch.cuda.empty_cache()
        TW = max(T, getattr(self, "win_max", 4))       # the most tokens a window holds (drafts + 1; 5 at most)
        if getattr(self, "_wbuf", None) is None:       # the buffers, made once for TW tokens and shared by all sizes
            z = lambda *s, dt=torch.float32: torch.zeros(s, dtype=dt, device=dv)   # noqa: E731
            wb = self._wbuf = {}
            wb["idx"] = {l: z(TW, self.nsel, dt=torch.int32) for l in range(m.L)}
            wb["top"] = {l: z(TW, self.nsel) for l in range(m.L)}
            wb["hotp"] = {l: z(TW * self.nsel, H) for l in range(m.L)}
            wb["coldp"] = {l: z(TW * self.nsel, H) for l in range(m.L)}
            wb["S"] = {l: z(TW, *self.S[l].shape) for l in self.S}
            wb["conv"] = {l: z(TW, *self.conv[l].shape) for l in self.conv}
            wb["plec"] = {l: z(TW, *self.plec[l].shape) for l in self.plec}
            wb["pos"] = torch.zeros(TW, dtype=torch.long, device=dv)
            wb["hin"] = z(TW, m.hc * H)
            wb["ple"] = z(TW, self.ple_emb.shape[1]) if self.ple_emb is not None else None
            C = qx_cpu.lib()
            off = (ctypes.c_long * 4)()
            C.ma_box_layout(TW, self.nsel, H * 4, H * 4, 1, off)
            wb["boff"] = list(off)
            wb["bh"], wb["bd"] = lib().host_mapped(dv.index, wb["boff"][3] * m.L)
            self._wins = {}
        wb = self._wbuf
        self.idxT = {l: t[:T] for l, t in wb["idx"].items()}
        self.topT = {l: t[:T] for l, t in wb["top"].items()}
        self.hotpT = {l: t[:T * self.nsel] for l, t in wb["hotp"].items()}
        self.coldpT = {l: t[:T * self.nsel] for l, t in wb["coldp"].items()}
        self.hin2T, self.wr2T, self.x2T, self.gsusT, self.shT, self.sdotT = {}, {}, {}, {}, {}, {}
        self.S_hist, self.conv_hist, self.plec_hist = wb["S"], wb["conv"], wb["plec"]
        self.posw = {dv: wb["pos"][:T]}
        self.hinT = wb["hin"][:T]
        self.pleT = wb["ple"][:T] if wb["ple"] is not None else None
        self.boxT_off, self.boxT_h, self.boxT_d, self.boxT_max = wb["boff"], wb["bh"], wb["bd"], TW
        for l in range(m.L):                          # the multi-matrix launches' descriptors, before the capture
            p = "blk.%d." % l
            for pre in ("hc_attn", "hc_ffn"):
                self._prep_multiT([p + pre + "_down.weight", p + pre + "_inject.weight"], T, dv)
            if (l + 1) % m.interval == 0:
                self._prep_multiT([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
                                   p + "indexer.k_proj.weight", p + "indexer.q_proj.weight"], T, dv)
            else:
                self._prep_multiT([p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_beta.weight",
                                   p + "ssm_alpha.weight"], T, dv)
            self._prep_multiT([p + "ffn_gate_inp.weight", p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight"],
                              T, dv)
            if l in m.ple_layers:
                self._prep_multiT([p + "ple_key.weight", p + "ple_value.weight"], T, dv)
        torch.cuda.synchronize(dv)
        cap = torch.cuda.Stream(dv)
        self._in_pass = True
        try:
            with torch.cuda.device(dv):
                gw = torch.cuda.CUDAGraph()
                if not self._wins:                     # one memory pool for the window graphs (replayed one at a time)
                    self._wpool = torch.cuda.graph_pool_handle()
                # KINDI_QXG_TIMELINE=1: the card's global timer per layer at 5 points (start, after the post, after
                # the card's experts and the shared expert, after the CPU's rows, end) into self.tl[T] (ns)
                tl = None
                if os.environ.get("KINDI_QXG_TIMELINE") == "1":
                    tl = self.__dict__.setdefault("tl", {}).setdefault(
                        T, torch.zeros(5 * m.L + 1, dtype=torch.int64, device=g.devices[0]))
                with torch.cuda.graph(gw, pool=self._wpool, stream=cap):
                    L = lib()
                    d_, st = _st(self.seq_d)
                    L.ma_seq(d_, st, _p(self.seq_d))
                    h = self.hinT
                    for l in range(m.L):
                        if tl is not None:
                            L.ts(d_, st, _p(tl), 5 * l)
                        self._pre_aT(l, h)
                        box = self.boxT_d + l * self.boxT_off[3]
                        L.ma_post(d_, st, box, _p(self.seq_d), _p(self.idxT[l]), _p(self.x2T[l]), T, self.nsel, H,
                                  self.boxT_off[1])
                        if tl is not None:
                            L.ts(d_, st, _p(tl), 5 * l + 1)
                        self._fetch(l, self.idxT[l], T * self.nsel, d_, st)
                        self._pre_bT(l)
                        self._fetch_join(l, d_, st)
                        if tl is not None:
                            L.ts(d_, st, _p(tl), 5 * l + 2)
                        L.ma_wait(d_, st, box, _p(self.seq_d), _p(self.coldpT[l]), _p(self.idxT[l]),
                                  _p(self.slotmap[l]), T, self.nsel, H, self.boxT_off[2])
                        if tl is not None:
                            L.ts(d_, st, _p(tl), 5 * l + 3)
                        h = self._postT(l)
                        self._fetch_reset(l, d_, st)
                        if tl is not None:
                            L.ts(d_, st, _p(tl), 5 * l + 4)
                    self.hlastT = h
                    x = self._mixerT(h, "output_hc", inject=False)
                    self.logitsT = self.mvT("output.weight", x)
                    if tl is not None:
                        L.ts(d_, st, _p(tl), 5 * m.L)
                self.gwin, self.winT = gw, T
                self._wins[T] = (gw, self.hlastT, self.logitsT)
        finally:
            self._in_pass = False

    def window(self, toks):
        """len(toks) tokens at once -> (logits [T, vocab], the streams after the last layer [T, 4 x 2560]); the
        state moves len(toks) positions (cut it back with keep())"""
        from kindi import qx_cpu
        g, m = self.g, self.m
        T = len(toks)
        t_in = time.time()
        if not 1 <= T <= 5:
            raise ValueError("a window holds 1 to 5 tokens (k_dense8T)")
        if T not in (getattr(self, "_wins", None) or {}):
            self._capture_window(T)
        else:                                         # the graph of this size, and the views of the shared buffers
            self.gwin, self.hlastT, self.logitsT = self._wins[T]
            wb = self._wbuf
            self.posw = {g.devices[0]: wb["pos"][:T]}
            self.hinT = wb["hin"][:T]
            self.pleT = wb["ple"][:T] if wb["ple"] is not None else None
            self.winT = T
        if self.T + T > self.lmax:
            raise ValueError("the decode caches hold %d positions (KINDI_QXG_LMAX)" % self.lmax)
        self._ext_on()
        dv = g.devices[0]
        self.posw[dv].copy_(torch.arange(self.T, self.T + T))
        self.hinT.copy_(self._erows("token_embd.weight", list(toks)).repeat(1, m.hc))
        if self.pleT is not None:
            ctx = list(self.ctx)
            rows = []
            for t in toks:
                rows.append(m.ple_rows([t], np.asarray(ctx, np.int64)))
                ctx = (ctx + [t])[-(m.ple_n - 1):]
            r = np.concatenate(rows)
            self.pleT.copy_(self._erows("per_layer_token_embd.weight", r.reshape(-1)).view(T, -1))
        self.seq_h += 1
        t_win = time.time()
        self.t_prep = getattr(self, "t_prep", 0.0) + t_win - t_in        # the window's host split (drafter log)
        self.gwin.replay()
        t_srv = time.time()
        self.t_launch = t_srv - t_win                 # the replay call's host time (timeline)
        rr = qx_cpu.lib().ma_box_serve_pass(self.boxT_h, m.L, self.boxT_max, self.nsel, H * 4, H * 4, 1, self.seq_h,
                                            self.ma_fn, self.ma_ctx, 60.0)
        t_srv = time.time() - t_srv
        if rr != 0:
            raise RuntimeError("window pass: layer %d failed or timed out" % (-rr - 1))
        picks_of = lambda l: np.ctypeslib.as_array(       # noqa: E731
            ctypes.cast(self.boxT_h + l * self.boxT_off[3] + self.boxT_off[0], ctypes.POINTER(ctypes.c_int32)),
            (T * self.nsel,))
        self._note_step("window", T, t_win, t_srv, picks_of)
        t_c = time.time()
        self._fetch_tune(picks_of, T)
        self._cache_step(picks_of)
        self.t_cache = getattr(self, "t_cache", 0.0) + time.time() - t_c   # host work after the window (drafter log)
        self.t_run = getattr(self, "t_run", 0.0) + t_c - t_win
        self._win_toks, self._win_T0, self._ctx_at_window = list(toks), self.T, list(self.ctx)
        self.T += T
        self.ctx = (self.ctx + list(toks))[-(m.ple_n - 1):]
        return self.logitsT, self.hlastT

    def keep(self, n):
        """keep the first n tokens of the last window (1 <= n <= T): the positions and the conv, delta-rule and
        n-gram states go back to those after token n - 1 (the attention caches need nothing: later positions are
        written again before they are read)"""
        T = len(self._win_toks)
        if n == T:
            return
        for l in self.S:
            self.S[l].copy_(self.S_hist[l][n - 1])
            self.conv[l].copy_(self.conv_hist[l][n - 1])
        for l in self.plec:
            self.plec[l].copy_(self.plec_hist[l][n - 1])
        self.T = self._win_T0 + n
        self.ctx = (self._ctx_at_window + self._win_toks[:n])[-(self.m.ple_n - 1):]

    def _cache_step(self, picks_of):
        """the adaptive card cache: this step's picks, then the swaps (between steps, on the card's stream)"""
        caches = {id(c): c for c in self.g.cache_of.values() if c.adapt is not None}
        if not caches:
            return
        if not getattr(self, "ma_noted", False):
            for l in range(self.m.L):
                c = self.g.cache_of.get(l)
                if c is not None and c.adapt is not None:
                    c.note(l, picks_of(l))
        for c in caches.values():
            with torch.cuda.device(c.dv):
                c.after_step(self)

    def load(self, st):
        """copy a prefill state (new_state() dict, filled by Model.logits) into the static buffers"""
        m = self.m
        if getattr(self, "fetch", None) is not None:
            self.fetch["qx"] = None                   # the prompt's CPU work is not a decode rate
        T = st["pos"]
        if T + 1 > self.lmax:
            raise ValueError("the decode caches hold %d positions (KINDI_QXG_LMAX)" % self.lmax)
        for l in range(m.L):
            if l in self.kc:
                self.kc[l].zero_()
                self.vc[l].zero_()
                self.kc[l][:T].copy_(st["k"][l])
                self.vc[l][:T].copy_(st["v"][l])
                self.rk[l].zero_()
                self.rk[l][:T].copy_(st["ik"][l])
                nb, ra = T // R.QSA_RATIO, R.QSA_RATIO
                self.bk[l].zero_()
                if nb:
                    p = "blk.%d." % l
                    pooled = st["ik"][l][:nb * ra].view(nb, ra, -1).float().mean(1)
                    bk = self.g._rope(rms(pooled, self.g.vec[p + "indexer.k_norm.weight"], m.eps)[:, None, :],
                                      torch.arange(nb, device=pooled.device) * ra)[:, 0, :]
                    self.bk[l][:nb].copy_(bk)
            else:
                self.S[l].copy_(st["S"][l])
                self.conv[l].copy_(st["conv"][l])
            if l in self.plec:
                self.plec[l].copy_(st["ple"][l])
        self.T, self.ctx = T, list(st["ids"])
        for dv in self.g.devices:                    # give the prefill's cached blocks back before the graphs
            with torch.cuda.device(dv):              # (on a shared card, VRAM over-commit makes the driver page)
                torch.cuda.empty_cache()
        if self.graphs is None:
            saved = [(l, self.S[l].clone()) for l in self.S] + [(l, self.conv[l].clone()) for l in self.conv]
            # the warm-ups write the attention caches only at the current position and at position 0: keep those
            # rows (a copy of the whole caches is 2x their memory at a long context)
            ra = R.QSA_RATIO
            lo, hi = T - T % ra, min(self.lmax, T + 64)
            spans = [(0, ra), (lo, hi)] if lo >= ra else [(0, hi)]
            savedk = {l: [(a, b, self.kc[l][a:b].clone(), self.vc[l][a:b].clone(), self.rk[l][a:b].clone(),
                           self.bk[l][a // ra:b // ra + 1].clone()) for a, b in spans] for l in self.kc}
            savedp = {l: self.plec[l].clone() for l in self.plec}
            self._capture()                                  # the warm-up runs moved the states: put them back
            for l, t in saved[:len(self.S)]:
                self.S[l].copy_(t)
            for l, t in saved[len(self.S):]:
                self.conv[l].copy_(t)
            for l, spans_l in savedk.items():
                for a, b, k, v, rk, bk in spans_l:
                    self.kc[l][a:b].copy_(k)
                    self.vc[l][a:b].copy_(v)
                    self.rk[l][a:b].copy_(rk)
                    self.bk[l][a // ra:a // ra + bk.shape[0]].copy_(bk)
            for l, t in savedp.items():
                self.plec[l].copy_(t)
            for dv in self.g.devices:
                with torch.cuda.device(dv):
                    torch.cuda.empty_cache()

    def step_log(self):
        """the steps since the last call: [(kind, tokens, wall s, CPU-expert s, picks, picks on a card)] (each step
        and window logs itself: the timings and hit rates of every run); the pool's timers restart too"""
        out, self._log = getattr(self, "_log", []), []
        if hasattr(getattr(self.g, "cpu", None), "pool_profile") and not out:
            self.g.cpu.pool_profile()
        return out

    def step_summary(self):
        """a summary of step_log(): ms a token (wall, CPU experts), the card's share of the picks"""
        rows = self.step_log()
        if not rows:
            return {}
        tok = sum(r[1] for r in rows)
        picks = sum(r[4] for r in rows)
        pool = self.g.cpu.pool_profile() if hasattr(getattr(self.g, "cpu", None), "pool_profile") else {}
        out = {"steps": len(rows), "tokens": tok, "ms_per_token": round(1000 * sum(r[2] for r in rows) / tok, 2),
               "cpu_ms_per_token": round(1000 * sum(r[3] for r in rows) / tok, 2),
               "card_hit": round(sum(r[5] for r in rows) / picks, 4) if picks else None, "pool": pool}
        fs = self.fetch_stats()
        if fs is not None:                             # the PCIe share: the picks it took from the CPU
            out["fetch"] = dict(fs, share_of_misses=round(fs["fetched_picks"] / max(1, fs["picks"] - fs["cache"]), 4))
            f = self.fetch
            if f.get("auto"):
                out["fetch"].update(link_gbs=round(f["gbs"], 2), nf_mean=round(float(f["nf"].mean()), 2),
                                    cpu_ms_per_expert=None if f["cpu_ms"] is None else round(f["cpu_ms"], 3))
        return out

    def _note_step(self, kind, ntok, t0, cpu_s, picks_of):
        g, m = self.g, self.m
        picks = hits = 0
        for l in range(m.L):
            ids = np.asarray(picks_of(l)).reshape(-1)
            picks += ids.size
            hits += int((g.slot[l][ids] >= 0).sum())
        wall = time.time() - t0
        self.__dict__.setdefault("_log", []).append((kind, ntok, wall, cpu_s, picks, hits))
        tot = self.__dict__.setdefault("totals", {"steps": 0, "tokens": 0, "wall_s": 0.0, "cpu_s": 0.0, "picks": 0,
                                                  "card_picks": 0})
        tot["steps"] += 1; tot["tokens"] += ntok; tot["wall_s"] += wall; tot["cpu_s"] += cpu_s
        tot["picks"] += picks; tot["card_picks"] += hits

    def _ext_on(self):
        """KINDI_QXG_EXT: the elastic tails get their card memory back before the decode's next replay"""
        g = self.g
        ccs = [cc for cc in set(g.cache_of.values()) if cc.vbuf and not cc.ext_live and not cc.ext_dead]
        if not ccs:
            return
        torch.cuda.synchronize(g.devices[0])
        torch.cuda.empty_cache()
        n = sum(cc.ext_on(self) for cc in ccs)
        if not getattr(self, "_ext_logged", False):
            self._ext_logged = True
            g.log("elastic cache: %d experts in the decode-time tails (%.2f GB free after)" % (
                n, torch.cuda.mem_get_info(g.devices[0])[0] / 1e9))

    def step(self, tok):
        """one token -> its logits [vocab] (on the last card); the state moves one position"""
        t_step = time.time()
        g, m = self.g, self.m
        if self.T + 1 > self.lmax:
            raise ValueError("the decode caches hold %d positions (KINDI_QXG_LMAX)" % self.lmax)
        self._ext_on()
        for dv in g.devices:
            self.pos[dv].fill_(self.T)
        emb = self._erows("token_embd.weight", [tok]).repeat(1, m.hc)
        self.hin[0].copy_(emb)
        if self.ple_emb is not None:
            rows = m.ple_rows([tok], np.asarray(self.ctx, np.int64))
            self.ple_emb.copy_(self._erows("per_layer_token_embd.weight", rows.reshape(-1)).view(1, -1))
        cpu = g.cpu
        cpu_s = 0.0
        if self.passmode:
            from kindi import qx_cpu
            self.seq_h += 1
            self.gpass.replay()
            t_srv = time.time()
            r = qx_cpu.lib().ma_box_serve_pass(self.box_h, m.L, 1, self.nsel, H * 4, H * 4, 1, self.seq_h,
                                               self.ma_fn, self.ma_ctx, 60.0)
            t_srv = time.time() - t_srv
            if r != 0:
                raise RuntimeError("mailbox pass: layer %d failed or timed out" % (-r - 1))
            picks_of = lambda l: np.ctypeslib.as_array(   # noqa: E731
                ctypes.cast(self.box_h + l * self.box_off[3] + self.box_off[0], ctypes.POINTER(ctypes.c_int32)),
                (self.nsel,))
            self._note_step("pass", 1, t_step, t_srv, picks_of)
            self._fetch_tune(picks_of, 1)
            self._cache_step(picks_of)
            self.T += 1
            self.ctx = (self.ctx + [tok])[-(m.ple_n - 1):]
            return self.logits_pass[0]
        for l, (dv, gp, gb, gq) in enumerate(self.graphs):
            with torch.cuda.device(dv):
                if self.hin[l] is not None and l > 0:
                    self.hin[l].copy_(self.hout[l - 1])
                gp.replay()
                need_cpu = cpu is not None and g.hot_count[l] < m.n_exp
                if need_cpu:
                    self.ev[l].record(torch.cuda.current_stream(dv))
                gb.replay()                          # the card's experts run while the CPU does its own
                if need_cpu:
                    self.ev[l].synchronize()
                    ids = self.idx_h[l].numpy()[None, :]
                    hot = (g.slot[l][ids] >= 0)
                    if not hot.all():
                        t0 = time.time()
                        skip = hot.astype(np.uint8)
                        if FAST["tail"] > 0:     # fast mode: missed picks at rank >= tail dropped, rows zero
                            drop = ~hot[0] & (np.arange(hot.shape[1]) >= FAST["tail"])
                            skip[0, drop] = 1
                            self.cold_h[l].numpy()[0, drop] = 0.0
                        cpu.forward_parts(l, self.x_h[l].numpy(), ids, skip, self.cold_h[l].numpy())
                        g.times["cpu_experts"] += time.time() - t0
                        cpu_s += time.time() - t0
                    if self.coldsum:
                        # the CPU's slots summed in slot order (acc = acc + w * value, fp32), one row to the card
                        acc = np.zeros(H, np.float32)
                        w, rows = self.top_h[l].numpy(), self.cold_h[l].numpy()[0]
                        for s_ in np.nonzero(~hot[0])[0]:
                            acc = acc + np.float32(w[s_]) * rows[s_]
                        self.csum_h[l].numpy()[0] = acc
                        self.csum[l].copy_(self.csum_h[l], non_blocking=True)
                    else:
                        # the CPU's rows go to the card in one copy: from its first to its last slot (the card's
                        # own rows in between are not read)
                        cs = np.nonzero(~hot[0])[0]
                        if len(cs):
                            a_, b_ = int(cs[0]), int(cs[-1]) + 1
                            self.coldp[l][a_:b_].copy_(self.cold_h[l][0][a_:b_], non_blocking=True)
                gq.replay()
        with torch.cuda.device(g.devices[-1]):
            if self.hhead is not self.hout[m.L - 1]:
                self.hhead.copy_(self.hout[m.L - 1])
            self.ghead.replay()
        self._note_step("step", 1, t_step, cpu_s, lambda l: self.idx_h[l].numpy())
        self._cache_step(lambda l: self.idx_h[l].numpy())
        self.T += 1
        self.ctx = (self.ctx + [tok])[-(m.ple_n - 1):]
        return self.logits[0]
