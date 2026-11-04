"""Tensor-core decode and speculative decoding for the one-card plan engine (sm_80+, KINDI_TC=1; on Pascal sm_61
the same plans run the dp4a kernels of tcp.cu, bit-identical products).

One plan per token count T = 1 .. 8: T tokens at positions pos .. pos + T - 1 in one pass, every matrix product an
int8 tensor-core product of all T tokens (tc.cu: each token's result is bit-identical to a pass of that token
alone), every per-token kernel a token-batched wrapper of the decode kernel (same code), each token's attention the
decode kernels with that token's position count, the GDN steps in token order inside one launch. T = 1 is the
decode; T > 1 with the head is the speculative verify. So speculative greedy output is the non-speculative greedy
output token for token, by construction.

A verify pass leaves the GDN state (M, conv) untouched (wb = 0) and keeps each layer's GDN inputs; once the number
n of accepted tokens is known, the commit plan advances the state by n tokens with the same code (spec.cu).

Each plan replays as a CUDA graph (one kernel node per launch, in plan order): the plan engine was launch-bound on
the cluster host (1301 launches per token, ~16 us each through ctypes). Only the nodes with position- or
token-dependent arguments are updated per pass (cuGraphExecKernelNodeSetParams).

The weights are repacked on the card into the tc layout at load (same size as the 5-trit lane layout), so the
lane-table kernels (k_g5n4, the chunk GEMM) cannot run on them: with KINDI_TC=1 prompts go through T = 8 passes,
or through the tensor-core chunk GEMM once it exists. KINDI_TC=0 (the default) leaves everything as it was.

Speculative decoding (greedy requests; the server calls spec_begin / spec_end around a reply):
  KINDI_SPEC = 0 (default) | ngram (prompt lookup) | draft2b (a Ternary-Bonsai-2 2B pack, KINDI_SPEC_PACK, on the
  same card, built first with context KINDI_SPEC_CTX, default 73728) | both (n-gram when it proposes 2+ tokens,
  else the 2B); KINDI_SPEC_K = draft length (default 4, at most 7; 3 is best with the 2B). Sampled requests use
  speculative sampling (KINDI_SPEC_SAMPLE=0: greedy requests only)."""
import ctypes
import json
import os
import re
import time

import numpy as np
import cupy as cp

from . import tp as _tp
from . import chunk as _chunk
from . import tc as _tc
from . import sampling
from . import mtp as _mtp

HERE = os.path.dirname(os.path.abspath(__file__))
i32, f32t = np.int32, np.float32
TMAX = max(2, min(8, int(os.environ.get("KINDI_TC_TMAX", "8"))))   # the largest T-token pass (verify needs k + 1)
WRAP = _chunk.WRAP + ("k_argmax", "k_softmax_rows", "k_attn_vq8w", "k_attn_combn", "k_deq_emb4")
FUSED = ("k_rmsfold_qT", "k_smfold_qT", "k_fold_fwht_qT", "k_softmax_m", "k_attn_vq8w_m", "k_attn_combn_m", "k_attn_sq8m",
         "k_deq_emb4_p", "k_copyf", "k_fd_q8_4", "k_fd_q8_6", "k_fd_q8_8", "k_fd_comb",
         "k_rmsfold2_1024", "k_rmsfold2_512", "k_att_tc_T")
FD_CH = 512                              # positions per block of the split flash-decode attention (tcf.cu)
QB = 4                                   # queries per verify attention launch (k_attn_sq8m)
# KINDI_SPEC=auto thresholds (positions), measured on an RTX 3080 at 190 W and an RTX 3060 (docs/MTP.md,
# docs/dev/bench-0.4.0/mtp.md): the 2B's own head k=2 up to 8K, off above; the 27B with the 2B drafter k=2 below
# 1K and k=3 from 1K. B2 (the 2B's head extends the 2B draft) was slower than B0 on both cards: off by default
AUTO_MTP_K2_MAX = int(os.environ.get("KINDI_SPEC_MTP_K2_MAX", "8192"))
AUTO_B0_K2_MAX = int(os.environ.get("KINDI_SPEC_B0_K2_MAX", "1024"))
AUTO_B2_MAX = int(os.environ.get("KINDI_SPEC_B2_MAX", "0"))
# the speculation guard works from this position on: below it, speculation was faster on every measured card, and
# the guard's pauses cost 9-12% (RTX 3060, 512 and 24K)
GUARD_FROM = int(os.environ.get("KINDI_SPEC_GUARD_FROM", "32768"))


def wanted():
    return os.environ.get("KINDI_TC", "0") == "1"


# ====================================================================================== kernels
class _Fn(object):
    """a driver Function with the .kernel.ptr the plans expect"""

    def __init__(self, f):
        self.kernel = f
        self.f = f

    def __call__(self, *a, **k):
        return self.f(*a, **k)


_TCF_HIP = (
    ("""    unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3));
    word |= __shfl_xor_sync(0xffffffffu, word, 1);
    word |= __shfl_xor_sync(0xffffffffu, word, 2);
    if(on && (lane & 3) == 0){
      const int k4 = col >> 2, s = k4 >> 3, h = (k4 >> 2) & 1, t = k4 & 3;
      XQ[(((long long)n * NG + g) * 4 + t) * 8 + 2 * s + h] = word;
    }
    __syncthreads();""",
     """    __shared__ unsigned char tcq_qb[8][128];
    tcq_qb[w >> 2][col] = (unsigned char)(qi & 255);
    __syncthreads();
    if(on && (w & 3) == 0) XQ[((long long)n * NG + g) * 32 + lane] = tcq_word5(tcq_qb[w >> 2], lane);
    __syncthreads();"""),
    ("""  unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3));
  word |= __shfl_xor_sync(0xffffffffu, word, 1);
  word |= __shfl_xor_sync(0xffffffffu, word, 2);
  const int g = (col0 >> 7) + (w >> 2), col = (w & 3) * 32 + lane;
  if((lane & 3) == 0){
    const int k4 = col >> 2, s = k4 >> 3, hh = (k4 >> 2) & 1, t = k4 & 3;
    XQ[(((long long)n * NG + g) * 4 + t) * 8 + 2 * s + hh] = word;
  }
  __syncthreads();""",
     """  __shared__ unsigned char tcq_qb[8][128];
  const int g = (col0 >> 7) + (w >> 2), col = (w & 3) * 32 + lane;
  tcq_qb[w >> 2][col] = (unsigned char)(qi & 255);
  __syncthreads();
  if((w & 3) == 0) XQ[((long long)n * NG + g) * 32 + lane] = tcq_word5(tcq_qb[w >> 2], lane);
  __syncthreads();"""),
)


class SpecKernels(object):
    def __init__(self):
        base = open(os.path.join(HERE, "kindi.cu"), encoding="utf-8").read()
        # (the device copies are made from the kernel text; _wrappers reads the parameters from the first "(", so the
        # copies are taken from a text without the launch bounds)
        wr, _ = _chunk._wrappers(re.sub(r"__launch_bounds__\(\d+\) (k_\w+)\(", r"\1(", base), WRAP)
        own = open(os.path.join(HERE, "spec.cu"), encoding="utf-8").read()
        hip = cp.cuda.runtime.is_hip
        tcs = open(os.path.join(HERE, "tc_hip.cu" if hip else "tc.cu"), encoding="utf-8").read()
        tcf = open(os.path.join(HERE, "tcf.cu"), encoding="utf-8").read()
        pas = not hip and _tc.pascal()
        names = _tc.NAMES_PASCAL if pas else _tc.NAMES
        if pas:
            # Pascal (sm_61): tcp.cu, the same products with dp4a on the pl layout; tca.cu (k_att_tc_T) is mma.sync
            # only. Compiled for the card itself (no PTX patch).
            tcs = open(os.path.join(HERE, "tcp.cu"), encoding="utf-8").read()
            if _tc.lane():
                tcs += "\n" + open(os.path.join(HERE, "tcl.cu"), encoding="utf-8").read()
            mod = cp.RawModule(code=base + "\n" + wr + "\n" + own + "\n" + tcs + "\n" + tcf)
        elif hip:
            # RDNA3: dp4a is v_dot4_i32_iu8; tca.cu (k_att_tc_T) is mma.sync only
            tcf = tcf.replace('asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));',
                              "d = __builtin_amdgcn_sudot4(true, a, true, b, c, false);")
            # the fused quantisers store the int8 rows in tc_hip.cu's 5-column order (tcq_word5)
            for old, new in _TCF_HIP:
                assert tcf.count(old) == 1, old
                tcf = tcf.replace(old, new)
            mod = cp.RawModule(code=base + "\n" + tcs + "\n" + wr + "\n" + own + "\n" + tcf)
        else:
            tcf += chr(10) + open(os.path.join(HERE, "tca.cu"), encoding="utf-8").read()
            p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            mod = _tc._patched_module(base + "\n" + wr + "\n" + own + "\n" + tcs + "\n" + tcf,
                                      "sm_%d%d" % (p["major"], p["minor"]))
        self.mod = mod
        for k in ([w + "_T" for w in WRAP] + ["k_sgdn_pre_T", "k_sgdn_deltan_T", "k_sgdn_deltan_C", "k_sconv_C", "k_xsum_T"] +
                  list(names) + (["k_g5t4"] + ["k_g5t2_%d" % n for n in (2, 3, 4)] + ["k_g5t1_%d" % n for n in (5, 6, 7, 8)]
                   if pas and _tc.lane() else []) +
                  [f for f in FUSED if not ((hip or pas) and f == "k_att_tc_T")]):
            f = mod.get_function(k)
            w = _Fn(f)
            if hip or pas:                                # a RawKernel: the plans want its driver function (.ptr)
                w.kernel = f.kernel
            setattr(self, k, w)
        self.mv = {2: self.k_tc_mv2, 4: self.k_tc_mv4, 8: self.k_tc_mv8}
        self.mv3 = {2: self.k_tc_mv3_2, 4: self.k_tc_mv3_4, 8: self.k_tc_mv3_8}


# ====================================================================================== CUDA graphs
class _KP(ctypes.Structure):
    _fields_ = [("func", ctypes.c_void_p), ("gx", ctypes.c_uint), ("gy", ctypes.c_uint), ("gz", ctypes.c_uint),
                ("bx", ctypes.c_uint), ("by", ctypes.c_uint), ("bz", ctypes.c_uint), ("smem", ctypes.c_uint),
                ("params", ctypes.c_void_p), ("extra", ctypes.c_void_p)]


_g = None


def _gdrv():
    global _g
    if _g is None:
        from . import gpudrv
        d = gpudrv.driver()
        if gpudrv.HIP:                       # ROCm: the HIP stand-in has its argument types set
            _g = d
            return _g
        P = ctypes.c_void_p
        d.cuGraphCreate.argtypes = [ctypes.POINTER(P), ctypes.c_uint]
        d.cuGraphAddKernelNode.argtypes = [ctypes.POINTER(P), P, ctypes.POINTER(P), ctypes.c_size_t, ctypes.POINTER(_KP)]
        d.cuGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(P), P, ctypes.c_ulonglong]
        d.cuGraphExecKernelNodeSetParams.argtypes = [P, P, ctypes.POINTER(_KP)]
        d.cuGraphLaunch.argtypes = [P, P]
        d.cuGraphDestroy.argtypes = [P]
        d.cuGraphExecDestroy.argtypes = [P]
        d.cuCtxGetCurrent.argtypes = [ctypes.POINTER(P)]
        d.cuCtxSetCurrent.argtypes = [P]
        _g = d
    return _g


class Graph(object):
    """A plan as a CUDA graph: one kernel node per launch, each depending on the one before."""

    def __init__(self, plan):
        d = _gdrv()
        self.plan = plan
        dynids = set(id(h) for h, _ in plan.dyn)
        g = ctypes.c_void_p()
        if d.cuGraphCreate(ctypes.byref(g), 0):
            raise RuntimeError("kindi: cuGraphCreate failed")
        self.g = g
        self.nodes, self.dyn = [], []
        prev = None
        for seg in plan.segs:
            for fn, gr, bl, arr, hs in seg:
                kp = _KP(fn.value, gr[0], gr[1], gr[2], bl[0], bl[1], bl[2], 0, ctypes.cast(arr, ctypes.c_void_p), None)
                node = ctypes.c_void_p()
                if prev is None:
                    r = d.cuGraphAddKernelNode(ctypes.byref(node), g, None, 0, ctypes.byref(kp))
                else:
                    dep = (ctypes.c_void_p * 1)(prev)
                    r = d.cuGraphAddKernelNode(ctypes.byref(node), g, dep, 1, ctypes.byref(kp))
                if r:
                    raise RuntimeError("kindi: cuGraphAddKernelNode failed (%d)" % r)
                self.nodes.append((node, kp, arr, hs))
                if any(id(h) in dynids for h in hs):
                    self.dyn.append((node, kp))
                prev = node.value
        ex = ctypes.c_void_p()
        r = d.cuGraphInstantiateWithFlags(ctypes.byref(ex), g, 0)
        if r:
            raise RuntimeError("kindi: cuGraphInstantiate failed (%d)" % r)
        self.ex = ex
        # the context the graph belongs to: the server launches from its request threads, which may have no
        # current context yet (a launch there dereferences a null context inside libcuda)
        self.ctx = ctypes.c_void_p()
        d.cuCtxGetCurrent(ctypes.byref(self.ctx))

    def launch(self, toks=None, pos=None):
        """toks/pos (optional): set the plan's Dyn holders first (the same call as Replay.launch on HIP)"""
        if toks is not None:
            self.plan.set(toks[0], pos, toks)
        d = _gdrv()
        if d.cuCtxSetCurrent(self.ctx):
            raise RuntimeError("kindi: cuCtxSetCurrent failed")
        for node, kp in self.dyn:
            if d.cuGraphExecKernelNodeSetParams(self.ex, node, ctypes.byref(kp)):
                raise RuntimeError("kindi: cuGraphExecKernelNodeSetParams failed")
        if d.cuGraphLaunch(self.ex, None):
            raise RuntimeError("kindi: cuGraphLaunch failed")


class Replay(object):
    """HIP: a plan replayed launch by launch (hipModuleLaunchKernel reads the Dyn holders at every launch); the
    Graph interface. A captured HIP graph keeps the capture's kernel arguments, so it is not used by default."""

    def __init__(self, plan):
        self.plan = plan
        self.nodes = [n for seg in plan.segs for n in seg]
        self.L = _gdrv().hipModuleLaunchKernel

    def launch(self, toks=None, pos=None):
        if toks is None:
            toks, pos = [0] * TMAX, 0
        self.plan.set(toks[0], pos, toks)
        L = self.L
        for fn, g, b, arr, _ in self.nodes:
            if L(fn, g[0], g[1], g[2], b[0], b[1], b[2], 0, None, arr, None):
                raise RuntimeError("kindi: a plan launch failed")


if cp.cuda.runtime.is_hip and os.environ.get("KINDI_TC_GRAPHS", "0") != "1":
    Graph = Replay


# ====================================================================================== weights: tc layout
def repack(s, log):
    """the card's 5-trit lane weights -> the tc layout, tensor by tensor through a scratch buffer"""
    t0 = time.time()
    if _tc.lane(s.card):                     # sm_60: the lane-table products read the lane layout as it is
        s.tc_layout = s.tc_lane = True
        log("card %d: lane-table products (sm_60), weights in the 5-trit lane layout" % s.card)
        return
    SK = s.sk
    big = max(int(R) * (int(C) // 128) * 7 for _, (_, _, R, C) in s.w.items())
    tmp = cp.empty(big, dtype=cp.uint32)
    dual = cp.cuda.runtime.is_hip and os.environ.get("KINDI_TC_DUAL", "1") == "1"
    if dual:
        # a second copy for the tc layout; the lane layout stays as s.wl for the chunk path
        s.bct = cp.empty_like(s.bc)
        s.wl, wt = s.w, {}
    for name, (tag, W5, R, C) in s.w.items():
        R, C = int(R), int(C)
        if R % 16 or C % 1024:
            raise SystemExit("kindi tc: %s (%d x %d) does not fit the tc layout" % (name, R, C))
        n = R * (C // 128) * 7
        if dual:
            o = (W5.data.ptr - s.bc.data.ptr) // 4
            Wt = s.bct[o:]
            SK.k_tc_pack(((n * 4 + 255) // 256,), (256,), (W5, Wt, i32(R), i32(C // 128)))
            wt[name] = (tag, Wt, i32(R), i32(C))
            continue
        SK.k_tc_pack(((n * 4 + 255) // 256,), (256,), (W5, tmp, i32(R), i32(C // 128)))
        W5[:n] = tmp[:n]
    if dual:
        s.w = wt
    cp.cuda.Device(s.card).synchronize()
    del tmp
    cp.get_default_memory_pool().free_all_blocks()
    s.tc_layout = True
    log("card %d: weights repacked for the %s (%.1fs)" % (s.card, "dp4a kernels" if _tc.pascal(s.card) else "tensor cores",
                                                         time.time() - t0))


def lane_sp(s, R, C):
    """the lane path's column splits of an R x C product: enough blocks for two per SM (one config for every token
    count, so a token's result never depends on how many share the pass)"""
    bx = -(-(R // 16) // 32)                 # the decode kernel's blocks (k_g5n4: 8 warps of 4 tiles)
    return max(1, min(8, C // 128, -(-2 * s.nsm // bx)))


def tune(s, R, C, rowpar=None):
    """(warps per block, column splits) for an R x C product: the least graph time for 1 + 8 tokens (one config for
    every token count, so a token's result never depends on how many share the pass). Timed as CUDA graphs of 10
    products (plus the parts sum unless the residual's k_rmsfold adds the parts), as the plans run."""
    if rowpar is None:
        rowpar = R == s.D
    cache = s.__dict__.setdefault("tccfg", {})
    if (R, C) in cache:
        return cache[(R, C)]
    if os.environ.get("KINDI_TC_TUNE", "0") != "1":
        # a fixed rule (the timed bests on the 3080), so a restart never changes a split and its rounding
        NU = C // 1024
        cfg = (2, 1) if not rowpar else (8 if NU > 8 else 4, min(NU, 6))
        cache[(R, C)] = cfg
        return cfg
    SK = s.sk
    name = next(n for n, (_, _, r, c) in s.w.items() if int(r) == R and int(c) == C)
    Wt = s.w[name][1]
    NG, NU = C // 128, C // 1024
    XQ = cp.zeros(TMAX * NG * 32, dtype=cp.uint32)
    XS = cp.zeros(TMAX * NG, dtype=cp.float32)
    XA = cp.zeros(TMAX * NG, dtype=cp.int32)
    Y = cp.zeros(TMAX * 17 * R, dtype=cp.float32) if R <= 65536 else cp.zeros(TMAX * R, dtype=cp.float32)
    O = cp.zeros(TMAX * R, dtype=cp.float32)
    ev0, ev1 = cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1"), cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1")
    best = None
    for nw in (2, 4, 8):
        gx = (R // 16 + nw - 1) // nw
        for sp in (1, 2, 3, 4, 6, 8, 12, 17):
            if sp > NU or (sp > 1 and Y.size < sp * TMAX * R):
                continue
            tot = 0.0
            for T in (1, TMAX):
                P = _tp.Plan()
                for _ in range(10):
                    P.add(SK.mv[nw], (gx, sp), (32 * nw,), (Wt, Y, i32(R), i32(NU), XQ, XS, XA, i32(T)))
                    if sp > 1 and not rowpar:
                        P.add(SK.k_tc_sum, ((R * T + 255) // 256,), (256,), (Y, O, i32(R), i32(T), i32(sp)))
                g = Graph(P)
                g.launch()
                ev0.record()
                for _ in range(3):
                    g.launch()
                ev1.record()
                ev1.synchronize()
                tot += cp.cuda.get_elapsed_time(ev0, ev1) / 30
                del g
            if best is None or tot < best[0]:
                best = (tot, nw, sp)
    cache[(R, C)] = best[1:]
    s.log("tc config %6d x %5d: %d warps, %2d splits (%.1f us for 1 + 8 tokens)" % (R, C, best[1], best[2], best[0] * 1e3))
    return best[1:]


def tune3(s, Rs, C):
    """warps per block for the merged product of the tensors with rows Rs (same input, no splits): graph-timed"""
    cache = s.__dict__.setdefault("tccfg3", {})
    key = (tuple(Rs), C)
    if key in cache:
        return cache[key]
    if os.environ.get("KINDI_TC_TUNE", "0") != "1":
        cache[key] = 8 if sum(Rs) >= 8192 else 2
        return cache[key]
    SK = s.sk
    ws = [s.w[next(n for n, (_, _, r, c) in s.w.items() if int(r) == R and int(c) == C)][1] for R in Rs]
    NG, NU = C // 128, C // 1024
    XQ = cp.zeros(TMAX * NG * 32, dtype=cp.uint32)
    XS = cp.zeros(TMAX * NG, dtype=cp.float32)
    XA = cp.zeros(TMAX * NG, dtype=cp.int32)
    ys = [cp.zeros(TMAX * R, dtype=cp.float32) for R in Rs]
    ev0, ev1 = cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1"), cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1")
    best = None
    for nw in (2, 4, 8):
        tot = 0.0
        for T in (1, TMAX):
            P = _tp.Plan()
            for _ in range(10):
                P.add(SK.mv3[nw], (sum(R // 16 for R in Rs) // nw + 1,), (32 * nw,), mv3_args(ws, ys, Rs, NU, XQ, XS, XA, T))
            g = Graph(P)
            g.launch()
            ev0.record()
            for _ in range(3):
                g.launch()
            ev1.record()
            ev1.synchronize()
            tot += cp.cuda.get_elapsed_time(ev0, ev1) / 30
            del g
        if best is None or tot < best[0]:
            best = (tot, nw)
    cache[key] = best[1]
    s.log("tc merged %s x %5d: %d warps (%.1f us for 1 + 8 tokens)" % ("+".join(str(R) for R in Rs), C, best[1], best[0] * 1e3))
    return best[1]


def mv3_args(ws, ys, Rs, NU, XQ, XS, XA, T):
    a = []
    for i in range(3):
        if i < len(ws):
            a += [ws[i], ys[i], i32(Rs[i])]
        else:
            a += [ws[0], ys[0], i32(0)]
    return tuple(a) + (i32(NU), XQ, XS, XA, i32(T))


# ====================================================================================== buffers and plans
def reserve(eng):
    """buffers for T <= 8 token passes, allocated before the context fit (on each card)"""
    if eng.W == 2:
        # the exchange of a T pass (k_xsum_T): per token slot [2 parities][2 cards][D] floats and a flag pair, in mapped
        # host memory (its own slots: the decode plans' k_xrms keep theirs)
        flags = cp.cuda.runtime.hostAllocMapped | cp.cuda.runtime.hostAllocPortable
        D = eng.cards[0].D
        eng.tcx_mem = cp.cuda.PinnedMemory(TMAX * 4 * D * 4, flags=flags)
        eng.tcx_flags = cp.cuda.PinnedMemory(TMAX * 2 * 4, flags=flags)
        np.ctypeslib.as_array((ctypes.c_int32 * (TMAX * 2)).from_address(eng.tcx_flags.ptr))[:] = 0
    for s in eng.cards:
        _reserve_card(eng, s)


def _reserve_card(eng, s):
    with cp.cuda.Device(s.card):
        z = lambda n, t=cp.float32: cp.zeros(int(n), dtype=t)  # noqa: E731
        D, NFF, HK, HD = s.D, s.NFF, s.HK, s.HD
        NK, NV, NH, NHKV, CC = s.NK, s.NV, s.NH, s.NHKV, s.CONV_CH
        work = max(D, NFF, NV * HK)
        T = TMAX
        b = s.tcb = {"work": work}
        b["HA"], b["HB"], b["XN"], b["Y"] = z(T * D), z(T * D), z(T * D), z(T * D)
        b["XP"] = z(T * work)
        b["U1"], b["U2"], b["ZV"] = z(T * NFF), z(T * NFF), z(T * NV * HK)
        b["OUT"] = z(T * max(NV * HK, NH * HD))
        b["QN"], b["KRO"], b["VB"], b["KN"] = z(T * 2 * NH * HD), z(T * NHKV * HD), z(T * NHKV * HD), z(T * NHKV * HD)
        b["FO"], b["QR"] = z(T * NH * HD), z(T * NH * HD)
        b["QQ"], b["QD"] = cp.zeros(T * NH * HD, dtype=cp.int8), z(T * NH * HD // 32)
        NGm = work // 128
        b["XQ"], b["XS"], b["XA"] = cp.zeros(T * NGm * 32, dtype=cp.uint32), z(T * NGm), cp.zeros(T * NGm, dtype=cp.int32)
        b["YH"], b["AOUT"] = z(T * s.R_head), cp.zeros(T, dtype=cp.int32)
        b["ctl"] = cp.zeros(16, dtype=cp.int32)
        if eng.W == 2:
            b["YX"] = z(T * D)                    # the row-parallel sums after the exchange
            s.tcctr = cp.zeros(TMAX, dtype=cp.int32)
        if os.environ.get("KINDI_CKPT_VRAM", "0") == "1" and not _building_drafter[0] and eng.W == 1:   # the cache's VRAM slot
            from .cache import _Slot
            eng.ckpt_slot = _Slot(list(s.M) + list(s.conv), len(s.M))
        # verify attention: scores and value parts of QB queries (k_attn_sq8m and the _m kernels)
        cx = int(getattr(eng, "_req_ctx", 0) or 131072)
        nchx = (cx + 255) // 256
        b["SB"], b["AP"] = z(QB * NH * cx), z(QB * nchx * NH * 256)
        b["ss"], b["ps"] = NH * cx, nchx * NH * 256
        # KINDI_FD=1: the split flash-decode attention's per-chunk sums and (max, exp-sum), QB queries
        if os.environ.get("KINDI_FD", "0") == "1":
            nfd = (cx + FD_CH - 1) // FD_CH
            b["FDP"], b["FDM"] = z(QB * nfd * NH * 256), z(QB * nfd * NH * 2)
            b["fps"], b["fms"] = nfd * NH * 256, nfd * NH * 2
        interval = s.dims["full_attention_interval"]
        gl = [il for il in range(s.NL) if (il + 1) % interval]
        b["L"] = {il: {"QKV": z(T * CC), "QK": z(T * 2 * NK * HK), "CS": z(T * CC), "GG": z(T * NV), "GB": z(T * NV)}
                  for il in gl}
        # split products: the parts buffer for the largest sp x T x R of the tuned configs
        mx = 0
        for name, (_, _, R, C) in s.w.items():
            nw, sp = (8, lane_sp(s, int(R), int(C))) if getattr(s, "tc_lane", False) else tune(s, int(R), int(C))
            if sp > 1:
                mx = max(mx, sp * T * int(R))
        b["PART"] = z(max(mx, 1))
        p0 = "blk.0."
        ia = s.dims["full_attention_interval"] - 1
        for names in () if getattr(s, "tc_lane", False) else (
                [p0 + "attn_qkv.weight", p0 + "attn_gate.weight"], [p0 + "ffn_gate.weight", p0 + "ffn_up.weight"],
                ["blk.%d.attn_q.weight" % ia, "blk.%d.attn_k.weight" % ia, "blk.%d.attn_v.weight" % ia]):
            tune3(s, [int(s.w[n][2]) for n in names], int(s.w[names[0]][3]))


def build_plan(s, T, head, wb, aout=None, tokp=None):
    """aout: where the argmaxes go (default the AOUT buffer); tokp: T = 1 only, the token id is read from this device
    int (a draft step chained to the step before) instead of a plan argument"""
    tp, K, SK, b = s.tp, s.k, s.sk, s.tcb
    D, HB, HK, HD = s.D, s.HB, s.HK, s.HD
    NK, NV, NH, NHKV, NFF, CC = s.NK, s.NV, s.NH, s.NHKV, s.NFF, s.CONV_CH
    work = b["work"]
    XP, XN, Y, OUT = b["XP"], b["XN"], b["Y"], b["OUT"]
    XQ, XS, XA, PART = b["XQ"], b["XS"], b["XA"], b["PART"]
    P = _tp.Plan()
    st = {"h": b["HA"], "h2": b["HB"], "pending": 0}
    TT = i32(T)

    def tadd(kern, gx, block, args, strides):
        P.add(kern, (gx, T), block, tuple(args) + (TT,) + tuple(i32(x) for x in strides))

    def gemm(name, X, ldx, Yo, quant, resid=False):
        """resid: the product goes to the residual stream; its column parts stay in PART for k_rmsfold to add
        (returns the part count)"""
        _, Wt, R, C = s.w[name]
        R, C = int(R), int(C)
        NG, NU = C // 128, C // 1024
        if getattr(s, "tc_lane", False):
            # sm_60: the lane-table product of the fp32 rows X (k_g5t4 for one token: four tiles per warp)
            sp = lane_sp(s, R, C)
            # (one token: kindi.cu's k_g5n4 itself, the faster of the two on the P100; tcl.cu's kernels give the same bits)
            out = Yo if sp == 1 else PART
            if T == 1:
                P.add(K.k_g5n4, (-(-(R // 16) // 32), sp), (256,), (Wt, out, i32(R), i32(NG), X))
            else:
                U = 2 if T <= 4 else 1
                P.add(getattr(SK, "k_g5t%d_%d" % (U, T)), (-(-(R // 16) // (9 * U)), sp), (288,),
                      (Wt, out, i32(R), i32(NG), X, i32(ldx), TT))
            if sp == 1:
                return 1
            if resid:
                return sp
            P.add(SK.k_tc_sum, ((R * T + 255) // 256,), (256,), (PART, Yo, i32(R), TT, i32(sp)))
            return 1
        nw, sp = tune(s, R, C)
        if quant:
            P.add(SK.k_tc_q8, (NG, T), (128,), (X, i32(ldx), XQ, XS, XA, i32(NG), TT))
        gx = (R // 16 + nw - 1) // nw
        if sp == 1:
            P.add(SK.mv[nw], (gx, 1), (32 * nw,), (Wt, Yo, i32(R), i32(NU), XQ, XS, XA, TT))
            return 1
        P.add(SK.mv[nw], (gx, sp), (32 * nw,), (Wt, PART, i32(R), i32(NU), XQ, XS, XA, TT))
        if resid:
            return sp
        P.add(SK.k_tc_sum, ((R * T + 255) // 256,), (256,), (PART, Yo, i32(R), TT, i32(sp)))
        return 1

    def gemm3(names, outs):
        """products of the same input in one launch (no splits)"""
        if getattr(s, "tc_lane", False):
            for n, o in zip(names, outs):
                gemm(n, XP, work, o, False)
            return
        ws = [s.w[n][1] for n in names]
        Rs = [int(s.w[n][2]) for n in names]
        C = int(s.w[names[0]][3])
        nw = tune3(s, Rs, C)
        P.add(SK.mv3[nw], (sum(R // 16 for R in Rs) // nw + 1,), (32 * nw,), mv3_args(ws, outs, Rs, C // 1024, XQ, XS, XA, T))

    def mixin(wname):
        """the residual add, RMS norm and Hadamard fold of every token, and their int8 rows for the next products"""
        add = st["pending"]
        r = PART if add > 1 else Y
        if add and tp.W == 2:
            # two cards: this card's row-parallel partial (its add column parts) + the other card's, for every token
            P.add(SK.k_xsum_T, (T,), (1024,), (r, i32(add), i32(D), np.uint64(tp.tcx_mem.ptr), np.uint64(tp.tcx_flags.ptr),
                                              i32(s.card), s.fail, s.tcctr, b["YX"]))
            add, r = 1, b["YX"]
        if _tc.pascal(s.card):
            # the one-pass fold (tcf.cu k_rmsfold2) is the default on Pascal (P40, 27B: 2.2 vs 3.7 ms per token); set in
            # the environment so that the prompt cache's engine fingerprint sees it
            os.environ.setdefault("KINDI_D1", "1")
        rf2 = os.environ.get("KINDI_D1", "0") == "1" and HB in (512, 1024)
        P.add({512: SK.k_rmsfold2_512, 1024: SK.k_rmsfold2_1024}[HB] if rf2 else SK.k_rmsfold_qT, (D // HB, T),
              (HB if rf2 else 512,), (st["h"], r, st["h2"], s.f(wname), s.sD, XN, XP, i32(D), i32(add),
                                                       i32(HB), TT, i32(D), i32(max(add, 1) * D), i32(D), i32(D), i32(work),
                                                       XQ, XS, XA))
        if st["pending"]:
            st["h"], st["h2"] = st["h2"], st["h"]
        st["pending"] = 0

    def gdn(il):
        p = "blk.%d." % il
        L = b["L"][il]
        mixin(p + "attn_norm.weight")
        gemm3([p + "attn_qkv.weight", p + "attn_gate.weight"], [L["QKV"], b["ZV"]])
        nblk = NV + 2 * NK + (NV * HK + 255) // 256
        P.add(SK.k_sgdn_pre_T, (nblk, T), (256,), (XN, i32(D), s.f(p + "ssm_beta.weight"), s.f(p + "ssm_alpha.weight"),
                                                   s.f(p + "ssm_dt.bias"), s.f(p + "ssm_a"), L["GB"], L["GG"], i32(NV), i32(D),
                                                   L["QKV"], i32(CC), s.conv[il], s.f(p + "ssm_conv1d.weight"), L["CS"], i32(CC),
                                                   L["QK"], i32(2 * NK * HK), i32(CC), i32(NV), i32(NK), i32(HK), TT, i32(wb)))
        P.add(SK.k_sgdn_deltan_T, (NV,), (HK,), (s.M[il], L["QK"], L["QK"][NK * HK:], i32(2 * NK * HK), L["CS"][2 * NK * HK:],
                                                 i32(CC), L["GG"], L["GB"], i32(NV), s.f(p + "ssm_norm.weight"), OUT,
                                                 i32(NV * HK), f32t(1.0 / np.sqrt(HK)), i32(NK), TT, i32(wb)))
        P.add(SK.k_smfold_qT, (NV * HK // HB, T), (512,), (b["ZV"], OUT, s.sV, XP, i32(HK), i32(NK), i32(NV // NK), i32(HB),
                                                           TT, i32(NV * HK), i32(NV * HK), i32(work), i32(NV * HK), XQ, XS, XA))
        st["pending"] = gemm(p + "ssm_out.weight", XP, work, Y, False, resid=True)

    def attn(il, kvi):
        p = "blk.%d." % il
        mixin(p + "attn_norm.weight")
        L, QL = NHKV * HD, NH * HD
        QN, KRO, VB, KN, FO, QR, QQ, QD = b["QN"], b["KRO"], b["VB"], b["KN"], b["FO"], b["QR"], b["QQ"], b["QD"]
        gemm3([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], [QN, KRO, VB])
        tadd(SK.k_rmsw_T, (NHKV + 7) // 8, (256,), (KRO, s.f(p + "attn_k_norm.weight"), KN, i32(HD), i32(NHKV)), (L, 0, L))
        tadd(SK.k_rmsw_s_T, (NH + 7) // 8, (256,), (QN, i32(2 * HD), s.f(p + "attn_q_norm.weight"), FO, i32(HD), i32(NH)),
             (2 * QL, 0, QL))
        rd = s.cos_t.shape[1]
        cs_ = _tp.Dyn("pos", s.cos_t.data.ptr, rd * 4, ptr=True)
        sn_ = _tp.Dyn("pos", s.sin_t.data.ptr, rd * 4, ptr=True)
        tadd(SK.k_rope2_T, NH + NHKV, (256,), (FO, KN, cs_, sn_, QR, KRO, i32(NH), i32(HD)), (QL, L, rd, rd, QL, L))
        nb0 = L // 32
        tadd(SK.k_store_q8_T, (64 * nb0 + 1023) // 1024, (min(1024, 64 * nb0),),
             (KRO, VB, _tp.Dyn("pos", s.Kq[kvi].data.ptr, L, ptr=True), _tp.Dyn("pos", s.Kd[kvi].data.ptr, nb0 * 2, ptr=True),
              _tp.Dyn("pos", s.Vq[kvi].data.ptr, L, ptr=True), _tp.Dyn("pos", s.Vd[kvi].data.ptr, nb0 * 2, ptr=True), i32(nb0)),
             (L, L, L, nb0, L, nb0))
        tadd(SK.k_q8row_T, (QL + 1023) // 1024, (1024,), (QR, QQ, QD, i32(QL // 32)), (QL, QL, QL // 32))
        nch = (s.ctx + 255) // 256
        if "FDP" in b:
            # split flash-decode: K and V read once per query, chunks of FD_CH positions merged by k_fd_comb (the
            # same kernels for decode and verify, so a token's result does not depend on the pass it is in)
            nfd = (s.ctx + FD_CH - 1) // FD_CH
            fps, fms = np.uint64(b["fps"]), np.uint64(b["fms"])
            for k0 in range(0, T, QB):
                qn = min(QB, T - k0)
                fdk = {4: SK.k_fd_q8_4, 6: SK.k_fd_q8_6, 8: SK.k_fd_q8_8}[s.G]
                P.add(fdk, (NHKV, nfd, qn), (256,), (s.Kq[kvi], s.Kd[kvi], s.Vq[kvi], s.Vd[kvi], QQ[k0 * QL:],
                                                     QD[k0 * QL // 32:], i32(QL), b["FDP"], fps, b["FDM"], fms,
                                                     _tp.Dyn("pos", 1 + k0, 1), i32(NHKV)))
                P.add(SK.k_fd_comb, (NH, qn), (256,), (b["FDP"], fps, b["FDM"], fms, QN[k0 * 2 * QL:], i32(2 * QL),
                                                       OUT[k0 * QL:], i32(QL), _tp.Dyn("pos", 1 + k0, 1), i32(NH)))
        elif T == 1:                            # decode: the decode kernels
            n = lambda: _tp.Dyn("pos", 1, 1)  # noqa: E731
            P.add(K.k_attn_sq8t, (NHKV, nch), (256,), (s.Kq[kvi], s.Kd[kvi], QQ, QD, s.sbuf, n(), i32(NHKV), i32(s.G)))
            P.add(K.k_softmax_rows, (NH,), (256,), (s.sbuf, n()))
            P.add(K.k_attn_vq8w, (NHKV, 1, nch), (128,), (s.sbuf, s.Vq[kvi], s.Vd[kvi], s.apart, n(), i32(NH), i32(NHKV),
                                                          i32(s.G)))
            P.add(K.k_attn_combn, (NH,), (256,), (s.apart, QN, OUT, n(), i32(NH)))
        for k0 in (range(0, T, QB) if T > 1 and "FDP" not in b else ()):
            # verify: up to QB queries per launch, query k sees the positions up to its own (pos + k + 1), each with
            # the decode kernels' arithmetic (k_attn_sq8m reads each K row once for all of them)
            qn = min(QB, T - k0)
            ss, ps = np.uint64(b["ss"]), np.uint64(b["ps"])
            n0 = lambda: _tp.Dyn("pos", 1 + k0, 1)  # noqa: E731
            P.add(SK.k_attn_sq8m, (NHKV, nch), (256,), (s.Kq[kvi], s.Kd[kvi], QQ[k0 * QL:], QD[k0 * QL // 32:], i32(QL),
                                                        b["SB"], ss, n0(), i32(qn), i32(NHKV), i32(s.G)))
            P.add(SK.k_softmax_m, (NH, qn), (256,), (b["SB"], ss, n0(), i32(qn)))
            P.add(SK.k_attn_vq8w_m, (NHKV, qn, nch), (128,), (b["SB"], ss, s.Vq[kvi], s.Vd[kvi], b["AP"], ps, n0(), i32(qn),
                                                              i32(NH), i32(NHKV), i32(s.G)))
            P.add(SK.k_attn_combn_m, (NH, qn), (256,), (b["AP"], ps, QN[k0 * 2 * QL:], i32(2 * QL), OUT[k0 * QL:], i32(QL),
                                                        n0(), i32(qn), i32(NH)))
        P.add(SK.k_fold_fwht_qT, (QL // HB, T), (512,), (OUT, s.sA, XP, i32(HB), TT, i32(QL), i32(work), i32(QL), XQ, XS, XA))
        st["pending"] = gemm(p + "attn_output.weight", XP, work, Y, False, resid=True)

    def ffn(il):
        p = "blk.%d." % il
        mixin(p + "post_attention_norm.weight")
        gemm3([p + "ffn_gate.weight", p + "ffn_up.weight"], [b["U1"], b["U2"]])
        P.add(SK.k_smfold_qT, (NFF // HB, T), (512,), (b["U1"], b["U2"], s.sF, XP, i32(0), i32(0), i32(0), i32(HB), TT,
                                                       i32(NFF), i32(NFF), i32(work), i32(NFF), XQ, XS, XA))
        st["pending"] = gemm(p + "ffn_down.weight", XP, work, Y, False, resid=True)

    emb = tp.emb
    for k in range(T):
        if tokp is not None:
            P.add(SK.k_deq_emb4_p, (D // HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, b["HA"][k * D:],
                                                         tokp, i32(D), i32(HB)))
        else:
            P.add(K.k_deq_emb4, (D // HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, b["HA"][k * D:],
                                                     _tp.DynT(k), i32(D), i32(HB)))
    kvi = 0
    for il in range(s.NL):
        if (il + 1) % s.dims["full_attention_interval"] == 0:
            attn(il, kvi)
            kvi += 1
        else:
            gdn(il)
        ffn(il)
    if head:
        mixin("output_norm.weight")
        gemm(s.head_name, XP, work, b["YH"], False)
        tadd(SK.k_argmax_T, 1, (1024,), (b["YH"], i32(s.R_head), b["AOUT"] if aout is None else aout), (s.R_head, 1))
    return P


def _copies(P, SK, pairs):
    for src, dst in pairs:
        n = int(src.size)
        P.add(SK.k_copyf, (min(1024, (n + 255) // 256),), (256,), (src, dst, np.uint64(n)))


def build_draft(s, k, M0, C0):
    """k chained greedy steps of a draft model in one graph: the GDN state is saved first (M0, C0), step i reads
    position pos + i and the token step i - 1 chose (AOUT[i - 1], on the card); the k choices land in AOUT[0 .. k)"""
    SK, b = s.sk, s.tcb
    P = _tp.Plan()
    _copies(P, SK, list(zip(s.M, M0)) + list(zip(s.conv, C0)))
    for i in range(k):
        Pi = build_plan(s, 1, True, 1, aout=b["AOUT"][i:], tokp=None if i == 0 else b["AOUT"][i - 1:])
        for h, d in Pi.dyn:
            if d.var == "pos":
                d.a += d.b * i
        P.segs[-1].extend(e for seg in Pi.segs for e in seg)
        P.dyn.extend(Pi.dyn)
    return P


def build_restore(s, M0, C0):
    P = _tp.Plan()
    _copies(P, s.sk, list(zip(M0, s.M)) + list(zip(C0, s.conv)))
    return P


def build_commit(s):
    """advance the GDN state by ctl[0] tokens from the inputs a verify pass kept"""
    SK, b = s.sk, s.tcb
    NK, NV, HK, CC = s.NK, s.NV, s.HK, s.CONV_CH
    P = _tp.Plan()
    for il, L in sorted(b["L"].items()):
        p = "blk.%d." % il
        P.add(SK.k_sconv_C, ((CC + 255) // 256,), (256,), (s.conv[il], L["QKV"], i32(CC), i32(CC), b["ctl"]))
        P.add(SK.k_sgdn_deltan_C, (NV,), (HK,), (s.M[il], L["QK"], L["QK"][NK * HK:], i32(2 * NK * HK), L["CS"][2 * NK * HK:],
                                                 i32(CC), L["GG"], L["GB"], i32(NV), s.f(p + "ssm_norm.weight"), b["OUT"],
                                                 i32(NV * HK), f32t(1.0 / np.sqrt(HK)), i32(NK), b["ctl"]))
    return P


def build(eng, log):
    t0 = time.time()
    for s in eng.cards:
        with cp.cuda.Device(s.card):
            s.tc_plans = {}
            for T in range(1, TMAX + 1):
                for head in (True, False):
                    wb = 0 if (head and T > 1) else 1
                    s.tc_plans[(T, head)] = Graph(build_plan(s, T, head, wb))
            s.tc_commit = Graph(build_commit(s))
    s = eng.cards[0]
    eng.tc = True
    with cp.cuda.Device(s.card):
        free = rt_free()
    log("tensor-core plans ready: T = 1 .. %d, %d nodes per decode token (%.1fs; %d MiB free on the card)" % (
        TMAX, len(s.tc_plans[(1, True)].nodes), time.time() - t0, free // 2 ** 20))


def rt_free():
    return cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()


# ====================================================================================== the engine's side
def run(eng, toks, pos, head):
    T = len(toks)
    if pos + T > eng.ctx:
        raise OverflowError("position %d is past the context (%d)" % (pos + T - 1, eng.ctx))
    tk = list(toks) + [0] * (TMAX - T)
    for s in eng.cards:
        s.tc_plans[(T, head)].launch(tk, pos)
    if eng.W == 2:
        _gdrv().cuCtxSetCurrent(eng.cards[0].tc_plans[(T, head)].ctx)   # CuPy work after the pass stays on card 0
    if head and _mtp.DUMP is not None:
        _mtp.DUMP.rows(eng, toks, pos)


def _hist(self):
    h = getattr(self, "_spdr", None)
    if h is None:
        h = self._spdr = NgramDraft()
    return h


def _hist_set(self, pos, toks):
    """the drafter's token history: positions < pos stay, toks go at pos .. (unknown positions are -1)"""
    h = _hist(self)
    if pos < len(h.h):
        h.truncate(pos)
    while len(h.h) < pos:
        h.push(-1)
    for t in toks:
        h.push(int(t))


def _known(self, toks):
    """the engine state now holds toks (restored, not read token by token): the drafters' history gets them, from
    the first position where it differs (a saved opening loaded from disk left -1 there)"""
    h = _hist(self).h
    n = min(len(h), len(toks))
    i = 0
    while i < n and h[i] == toks[i]:
        i += 1
    if i < len(toks) or len(h) > len(toks):
        _hist_set(self, i, list(toks[i:]))


def _spec_flush(self):
    """commit the tokens of the pending verify pass that were used (the GDN state then matches the positions fed)"""
    if getattr(self, "_sppend", False):
        commit(self, 1 + self._spn)
        self._sppend, self._spq = False, []


class _Guard(object):
    """Speculation only while it pays: tokens per second of speculative passes vs plain steps, each measured as
    wall time per call (every pass ends in a readback, so the wall time is the pass), totalled over windows of
    WIN tokens. The rates are kept across requests and reset when the depth moves by more than 8K tokens (the cost
    of a pass depends on it). At a new depth: one spec window, one PLAIN-token window of plain steps, then spec
    windows; two spec windows in a row below the plain rate x MARGIN pause speculation for PAUSE tokens, after
    which it is probed again (the first pass after a switch is not timed: it also catches the drafter up). A probe
    after a pause needs one window: a slow one pauses again for twice as long (up to MAXPAUSE), a fast one resumes."""
    WIN, PLAIN, PAUSE, MAXPAUSE, MARGIN = 48, 16, 256, 4096, 0.97
    BOUND = float(os.environ.get("KINDI_SPEC_GUARD_BOUND", "2.5"))   # verify pass <= BOUND plain steps (T <= 8: 2.28x measured)

    BOUND_DEPTH = int(os.environ.get("KINDI_SPEC_GUARD_BOUND_DEPTH", "8192"))   # deepest measured position

    def __init__(self):
        self.spec_r = self.plain_r = None
        self.depth = -10 ** 9
        self.pauses = self.probes = self.bad = 0
        self.pause_len, self.probing = self.PAUSE, False
        self._start("spec")

    def _start(self, mode):
        self.tv_n, self.tv_t = 0, 0.0                    # verify passes and their time, this window
        self.mode, self.n, self.t, self.skip = mode, 0, 0.0, mode == "spec"
        self.left = {"spec": self.WIN, "plain": self.PLAIN, "pause": self.pause_len}[mode]

    def at(self, pos):
        if abs(pos - self.depth) > 8192:
            self.spec_r = self.plain_r = None
            self.bad, self.depth = 0, pos
            self.pause_len, self.probing = self.PAUSE, False
            self._start("spec")

    def want_spec(self):
        return self.mode == "spec"

    def span(self, t0):
        """wall time since the last note (it includes the early-return steps that hand out accepted tokens and
        the drafter's catch-up), or since t0 when there is no recent note (a new request, or after a long gap)"""
        now = time.perf_counter()
        last, self.t_last = getattr(self, "t_last", None), now
        own = now - t0
        if last is None or last > t0 or now - last > 20 * own + 1.0:
            return own
        return now - last

    def note(self, spec, ntok, dt, tv=0.0):
        lp = os.environ.get("KINDI_SPEC_GUARD_LOG")
        if lp:
            with open(lp, "a") as f:
                f.write("%.4f %s %s n=%d dt=%.4f tv=%.4f left=%d\n" % (time.perf_counter(), self.mode,
                        "S" if spec else "P", ntok, dt, tv, self.left))
        if spec and tv > 0:
            self.tv_n, self.tv_t = self.tv_n + 1, self.tv_t + tv
        self.left -= ntok
        if spec and self.skip:
            self.skip = False
        elif dt > 0:
            self.n, self.t = self.n + ntok, self.t + dt
        if self.left > 0:
            return
        r = self.n / self.t if self.t > 0 else None
        if self.mode == "spec":
            if r is not None:
                self.spec_r = r
            if self.plain_r is None:                       # measure plain steps once at this depth
                # the bound (verify pass <= BOUND plain steps) is measured only up to BOUND_DEPTH; deeper, the
                # drafter and verify costs grow, so measure the plain rate instead of assuming it
                ub = (self.BOUND * self.tv_n / self.tv_t
                      if self.BOUND > 0 and self.tv_t > 0 and self.depth <= self.BOUND_DEPTH else None)
                if ub is not None and self.spec_r is not None and self.spec_r > ub:
                    self.plain_r = ub                      # an upper bound of the plain rate: spec wins anyway
                    self._start("spec")
                    return
                self._start("plain")
                return
            slow = self.spec_r is not None and self.spec_r < self.plain_r * self.MARGIN
            self.bad = self.bad + 1 if slow else 0
            if self.probing:                               # the one window after a pause decides
                self.probing = False
                if slow:
                    self.pause_len = min(self.pause_len * 2, self.MAXPAUSE)
                    self.bad = 0
                    self.pauses += 1
                    self._start("pause")
                    return
                self.pause_len = self.PAUSE
            if self.bad >= 2:
                self.bad = 0
                self.pauses += 1
                self._start("pause")
            else:
                self._start("spec")
        elif self.mode == "plain":
            if r is not None:
                self.plain_r = r
            self._start("spec")
        else:                                              # paused: plain steps keep the plain rate fresh
            if r is not None:
                self.plain_r = r
            self.probes += 1
            self.probing = True
            self._start("spec")


def _guard(self):
    g = getattr(self, "_sguard", None)
    if g is None:
        g = self._sguard = _Guard()
    return g


def tc_step(self, tok, pos):
    self.spec_presampled = None
    q = getattr(self, "_spq", None)
    if q and q[0][0] == pos and q[0][1] == tok:      # a verified token: its successor is known already
        _, _, out, row = q.pop(0)
        self._spn += 1
        _hist(self).push(int(tok))
        self._best, self._ylog_ok, self._ylog_row = out, True, row
        if getattr(self, "spec_sp", None) is not None:
            self.spec_presampled = out               # (sampling: the server takes it instead of sampling again)
        if not q:
            _spec_flush(self)
        return out
    _spec_flush(self)
    _hist_set(self, pos, [tok])
    k = getattr(self, "spec_k", 0) if getattr(self, "spec_ok", False) else 0
    if k and getattr(self, "spec_auto", False) and self.spec_mode == "mtp":
        k = 2 if pos <= AUTO_MTP_K2_MAX else 0      # measured: k=2 best up to 8K; above, no gain (off)
    elif k and getattr(self, "spec_auto", False) and self.spec_mode == "draft2b":
        k = 2 if pos < AUTO_B0_K2_MAX else 3        # measured: k=2 best at 512, k=3 from 1K
    k = min(k, self.ctx - pos - 1)
    G = _guard(self) if k > 0 and os.environ.get("KINDI_SPEC_GUARD", "1") == "1" and pos >= GUARD_FROM else None
    if G is not None:
        G.at(pos)
        if not G.want_spec():
            k = 0
    t0 = time.perf_counter()
    dr2 = getattr(self, "drafter", None) if k > 0 and self.spec_mode in ("draft2b", "both") else None
    mt = getattr(self, "mtp", None)
    d = []
    if k > 0 and self.spec_mode == "mtp":
        d = mt.draft(tok, pos, k)
    elif k > 0:
        d = _hist(self).propose(k) if self.spec_mode in ("ngram", "both") else []
        if dr2 is not None and len(d) < (2 if self.spec_mode == "both" else 1):   # "both": n-gram if it has 2+
            dr2.b2_on = getattr(self, "spec_b2", True) and (not getattr(self, "spec_auto", False) or pos < AUTO_B2_MAX)
            d = dr2.draft(_hist(self).h, pos, k)
        else:
            dr2 = None
    if d:
        tv0 = time.perf_counter()
        A = verify(self, [tok] + d, pos)
        tv = time.perf_counter() - tv0
        if getattr(self, "spec_sp", None) is not None:
            # sampling: rejection sampling against the server's own distribution of each row
            E = spec_accept(lambda i: _dist(self, i, A[i]), d, self.spec_sp[3])
            j = len(E) - 1
            A = E + A[len(E):]
            self._spq_sampled = True
        else:
            j = 0
            while j < len(d) and d[j] == A[j]:
                j += 1
        if dr2 is not None:
            dr2.accept(j)
        if mt is not None:
            mt.seen(pos, j + 1, A[:j + 1])
        self._spq = [(pos + i, d[i - 1], A[i], i) for i in range(1, j + 1)]
        self._sppend, self._spn = True, 0
        st = self.spec_stats
        st["passes"] += 1
        st["drafted"] += len(d)
        st["accepted"] += j
        if not self._spq:
            _spec_flush(self)
        if G is not None:
            G.note(True, j + 1, G.span(t0), tv)
        self._best, self._ylog_ok, self._ylog_row = A[0], True, 0
        if getattr(self, "spec_sp", None) is not None:
            self.spec_presampled = A[0]
        return A[0]
    run(self, [tok], pos, True)
    best = argmaxes(self, 1)[0]
    if mt is not None:
        mt.seen(pos, 1, [best])
    if G is not None:
        G.note(False, 1, G.span(t0))
    if k:
        self.spec_stats["passes"] += 1
    self._best, self._ylog_ok, self._ylog_row = best, True, 0
    return best


def tc_feed(self, tok, pos):
    _spec_flush(self)
    _hist_set(self, pos, [tok])
    if getattr(self, "mtp", None) is not None:
        self.mtp.gap()
    run(self, [tok], pos, _mtp.DUMP is not None)


def tc_feed_batch(self, toks, pos):
    _spec_flush(self)
    _hist_set(self, pos, toks)
    if getattr(self, "mtp", None) is not None:
        self.mtp.gap()
    if _mtp.DUMP is not None:                    # the recorder hook (tools): every row with the head
        i = 0
        while i < len(toks):
            n = min(TMAX, len(toks) - i)
            run(self, toks[i:i + n], pos + i, True)
            if n > 1:
                commit(self, n)
            i += n
        return
    i = _chunk.feed(self, toks, pos)             # whole chunks first (tensor-core GEMMs), the rest T <= 8 at a time
    while i < len(toks):
        n = min(TMAX, len(toks) - i)
        run(self, toks[i:i + n], pos + i, False)
        i += n


def spec_begin(self, greedy, temperature=0.0, top_p=1.0, top_k=0, rng=None):
    """speculative decoding for the steps that follow: greedy requests, or sampled ones when the server passes its
    sampling settings and generator (speculative sampling)"""
    mode = os.environ.get("KINDI_SPEC", "0")
    self.spec_auto = mode == "auto"
    if mode == "auto":
        # KINDI_SPEC=auto (the measured choice on a 3080): a model with its own MTP head drafts with it (greedy);
        # a model with the 2B drafter uses it, plus the 2B's head (B2) below KINDI_SPEC_B2_MAX tokens of context
        if getattr(self, "drafter", None) is not None:
            mode = "draft2b"
        elif getattr(self, "mtp", None) is not None and greedy:
            mode = "mtp"
        else:
            mode = "0"
    if mode in ("draft2b", "both") and getattr(self, "drafter", None) is None:
        mode = "ngram"
    if mode == "mtp" and (getattr(self, "mtp", None) is None or not greedy):
        mode = "0"                               # the MTP head drafts for greedy requests only (for now)
    if _mtp.DUMP is not None:
        mode = "0"
    self.spec_mode = mode
    self.spec_k = int(os.environ.get("KINDI_SPEC_K", "4")) if mode in ("ngram", "draft2b", "both", "mtp") else 0
    if self.spec_auto and mode == "draft2b":
        self.spec_k = 3
    self.spec_b2 = greedy                        # B2 (the 2B's head extends the draft): greedy requests only
    self.spec_k = max(0, min(self.spec_k, TMAX - 1))
    self.spec_sp = None
    if not greedy and rng is not None and os.environ.get("KINDI_SPEC_SAMPLE", "1") == "1" and self.W == 1:
        self.spec_sp = (float(temperature), float(top_p), int(top_k), rng)
        if getattr(self, "_spsampler", None) is None:
            self._spsampler = sampling.Sampler(self)
    self.spec_presampled = None
    self.spec_ok = (bool(greedy) or self.spec_sp is not None) and self.spec_k > 0
    if self.spec_ok and getattr(self, "drafter", None) is not None:
        self.drafter.begin()
    self.spec_stats = {"passes": 0, "drafted": 0, "accepted": 0}


def spec_end(self):
    _spec_flush(self)
    self.spec_ok, self.spec_sp, self.spec_presampled = False, None, None
    st = getattr(self, "spec_stats", None)
    G = getattr(self, "_sguard", None)
    if st is not None and G is not None:
        st["guard"] = {"mode": G.mode, "spec_tok_s": G.spec_r, "plain_tok_s": G.plain_r, "pauses": G.pauses}
    return st


def tc_ylog(self):
    s = self.cards[0]
    r = getattr(self, "_ylog_row", 0)
    if getattr(self, "_aout_for", None) != (r, self._best):      # the sampler's band is relative to this row's best
        with cp.cuda.Device(s.card):
            self.aout[0] = self._best
        self._aout_for = (r, self._best)
    if self.W == 1:
        return s.tcb["YH"][r * s.R_head:(r + 1) * s.R_head]
    if getattr(self, "_ylog_for", None) != (r, self._best):      # two cards: both halves of the row on card 0
        c1 = self.cards[1]
        with cp.cuda.Device(c1.card):
            y1 = cp.asnumpy(c1.tcb["YH"][r * c1.R_head:(r + 1) * c1.R_head])
        with cp.cuda.Device(s.card):
            h = self.head_split[1]
            self._ylog[:h] = s.tcb["YH"][r * s.R_head:(r + 1) * s.R_head]
            self._ylog[h:] = cp.asarray(y1)
        self._ylog_for = (r, self._best)
    return self._ylog


def _dist(self, row, best):
    """the server sampler's distribution (temperature, top-k, top-p; kindi.sampling.Sampler's own code) for row
    `row` of the last pass: (token ids, probabilities)"""
    temperature, top_p, top_k, _ = self.spec_sp
    self._ylog_row, self._best, self._ylog_ok = row, best, True
    smp = self._spsampler
    yl, nl, ymap = self.ylog, int(self.ylog_n), self.ylog_map
    smp.cn.fill(0)
    kk = int(top_k or 0)
    band = np.float32(sampling.BAND_T * temperature)
    if 0 < kk <= 1024:
        self.k.k_topk((1,), (1024,), (yl, self.aout, np.int32(nl), np.int32(kk), band, smp.ci, smp.cv, smp.cn,
                                      np.int32(sampling.CAP)))
    else:
        self.k.k_band((32,), (256,), (yl, self.aout, np.int32(nl), band, smp.ci, smp.cv, smp.cn, np.int32(sampling.CAP)))
    nc = int(smp.cn.get()[0])
    if 0 < nc <= sampling.CAP:
        ci = smp.ci[:nc]
        cand = cp.asnumpy(ymap[ci] if ymap is not None else ci).astype(np.int64)
        z = cp.asnumpy(smp.cv[:nc]).astype(np.float64)
        o = np.argsort(cand)
        cand, z = cand[o], z[o]
    else:
        cand = cp.asnumpy(ymap[:nl]).astype(np.int64) if ymap is not None else np.arange(nl)
        z = cp.asnumpy(yl[:nl]).astype(np.float64)
        ok = cand >= 0
        cand, z = cand[ok], z[ok]
    k = max(1, min(kk or len(cand), len(cand)))
    sel = np.argpartition(-z, k - 1)[:k] if k < len(cand) else np.arange(len(cand))
    idx = cand[sel]
    zz = z[sel] / temperature
    zz -= zz.max()
    p = np.exp(zz)
    p /= p.sum()
    order = np.argsort(-p)
    idx, p = idx[order], p[order]
    if top_p and top_p < 1.0:
        keep = (np.cumsum(p) - p) < top_p
        idx, p = idx[keep], p[keep]
        p /= p.sum()
    return idx, p


def spec_accept(dists, d, rng):
    """speculative sampling with a deterministic draft (q = one-hot on d[i]): draft i is kept with probability
    p_i(d[i]); at the first rejection the token is drawn from p_i without d[i] (renormalised); if all are kept, the
    next token is drawn from p_k. dists(i) -> (ids, probs) of the target distribution after token i of the pass.
    The emitted tokens are distributed exactly as tokens sampled one by one from the target."""
    out = []
    for i, di in enumerate(d):
        idx, p = dists(i)
        pd = float(p[idx == di].sum())
        if rng.random_sample() < pd:
            out.append(di)
            continue
        r = np.where(idx == di, 0.0, p)
        rs = r.sum()
        out.append(int(rng.choice(idx, p=r / rs)) if rs > 0 else int(rng.choice(idx, p=p)))
        return out
    idx, p = dists(len(d))
    out.append(int(rng.choice(idx, p=p)))
    return out


def verify(eng, toks, pos):
    """the T = len(toks) argmaxes after each token, the GDN state left as it was"""
    run(eng, toks, pos, True)
    return argmaxes(eng, len(toks))


def argmaxes(eng, T):
    """the argmaxes of the last pass's T rows; two cards: each card's best of its half of the vocabulary, the larger
    logit wins (a tie goes to card 0, as in TPEngine.step)"""
    if eng.W == 1:
        return [int(x) for x in cp.asnumpy(eng.cards[0].tcb["AOUT"][:T])]
    best = None
    for s in eng.cards:
        with cp.cuda.Device(s.card):
            b = s.tcb
            i = b["AOUT"][:T]
            v = cp.asnumpy(b["YH"][cp.arange(T, dtype=cp.int64) * s.R_head + i])
            g = cp.asnumpy(i).astype(np.int64) + eng.head_split[s.card]
        if best is None:
            best = [v, g]
        else:
            m = v > best[0]
            best = [np.where(m, v, best[0]), np.where(m, g, best[1])]
    return [int(x) for x in best[1]]


def commit(eng, n):
    for s in eng.cards:
        with cp.cuda.Device(s.card):
            s.tcb["ctl"][0] = n
        s.tc_commit.launch()
    if eng.W == 2:
        _gdrv().cuCtxSetCurrent(eng.cards[0].tc_commit.ctx)


# ====================================================================================== drafters
class NgramDraft(object):
    """prompt lookup: the tokens that followed the latest earlier occurrence of the longest matching suffix
    (n = nmax .. nmin)"""

    def __init__(self, nmax=4, nmin=1):
        self.nmax, self.nmin = nmax, nmin
        self.idx = {}
        self.h = []

    def reset(self, hist):
        self.idx, self.h = {}, []
        for t in hist:
            self.push(t)

    def truncate(self, L):
        """drop positions >= L (index entries past L or over changed tokens are caught when looked up)"""
        del self.h[L:]

    def push(self, t):
        h = self.h
        h.append(t)
        L = len(h)
        # index the n-grams that END one before the new token: their continuation (the new token) now exists
        for n in range(self.nmin, self.nmax + 1):
            if L - 1 - n >= 0:
                self.idx[tuple(h[L - 1 - n:L - 1])] = L - 1

    def propose(self, k):
        h = self.h
        for n in range(self.nmax, self.nmin - 1, -1):
            if len(h) < n:
                continue
            pat = tuple(h[-n:])
            if -1 in pat:
                continue
            j = self.idx.get(pat)
            if j is not None and j < len(h) and j >= n and tuple(h[j - n:j]) == pat:
                return [t for t in h[j:j + k] if t >= 0][:k] if -1 not in h[j:j + k] else []
        return []


def generate_spec(eng, ids, n, k=None, drafter=None, stats=None):
    """greedy generation with speculative verify: the same tokens as eng.generate(ids, n)"""
    k = int(os.environ.get("KINDI_SPEC_K", "4")) if k is None else k
    k = max(0, min(k, TMAX - 1))
    dr = drafter or NgramDraft()
    eng.reset()
    eng.feed_batch(ids[:-1], 0)
    dr.reset(list(ids))
    pos, last, out = len(ids) - 1, ids[-1], []
    st = stats if stats is not None else {}
    st.setdefault("passes", 0)
    st.setdefault("drafted", 0)
    st.setdefault("accepted", 0)
    while len(out) < n:
        d = dr.propose(min(k, n - len(out) - 1)) if k else []
        if not d:
            a = [eng.step(last, pos)]
        else:
            A = verify(eng, [last] + d, pos)
            j = 0
            while j < len(d) and d[j] == A[j]:
                j += 1
            a = A[:j + 1]
            commit(eng, j + 1)
            st["drafted"] += len(d)
            st["accepted"] += j
        st["passes"] += 1
        for t in a:
            dr.push(t)
        out += a
        pos += len(a)
        last = a[-1]
    return out[:n]


class _Copy(object):
    """a graph of device copies"""

    def __init__(self, s, pairs):
        P = _tp.Plan()
        _copies(P, s.sk, pairs)
        with cp.cuda.Device(s.card):
            self.g = Graph(P)

    def launch(self):
        self.g.launch()


class Draft2B(object):
    """A small model of the same family on the same card as the drafter (greedy). It follows the main model's
    token history lazily: before drafting at pos it reads the tokens it has not seen (or, if the history changed
    under it, everything again). A draft is one graph: the GDN state is saved, then k chained steps run on the card
    (each step's token is the step before's argmax). After the verify the state is put back and the accepted tokens
    are read again (or, when all were accepted, only the last), so it always holds exactly the accepted history."""

    def __init__(self, eng2):
        self.e = eng2
        self.h = []                              # the tokens the draft model has read, positions 0 ..
        s = eng2.cards[0]
        with cp.cuda.Device(s.card):
            self.M0 = [m.copy() for m in s.M]
            self.C0 = [c.copy() for c in s.conv]
            self.g = {}
            self.restore = Graph(build_restore(s, self.M0, self.C0))
            # the state where the last request's prompt ended: a later request that shares the history up to there
            # (the next agent turn re-renders the reply, so it often differs right after it) resumes from it
            self.M1 = [m.copy() for m in s.M]
            self.C1 = [c.copy() for c in s.conv]
        self.save1 = _Copy(s, list(zip(s.M, self.M1)) + list(zip(s.conv, self.C1)))
        self.load1 = _Copy(s, list(zip(self.M1, s.M)) + list(zip(self.C1, s.conv)))
        self.snap = None                         # (base, position, tokens) of the saved state
        self.base = 0                            # the main model's position of the draft model's position 0
        self.rebases = 0
        self.want_snap = False
        self.pend = None
        self.refeeds = 0
        self.resumes = 0
        self.mtp = None                          # KINDI_SPEC_DRAFT_HEAD: the 2B's own MTP head extends its draft
        self.b2 = False
        self.b2_on = True

    def begin(self):
        self.want_snap = True

    def _sync(self, hist, pos, k):
        """the draft model reads hist[base:pos] (its positions 0 ..). A window: when the next draft would not fit
        its context, it starts again from the last half of its context (positions rebased; RoPE attention only
        sees relative positions, and the main model verifies every token, so this only costs acceptance)."""
        W = self.e.ctx
        if pos < self.base or pos - self.base + k + 16 > W:   # outside the window (a later, longer or shorter history)
            self.base = max(0, pos - W // 2)
            self.e.reset()
            self.h, self.snap = [], None
            self.rebases += 1
        b, h = self.base, self.h
        p = 0
        n = min(len(h), pos - b)
        while p < n and h[p] == hist[b + p]:
            p += 1
        if p < len(h):                           # the history changed under the draft model
            sp = self.snap
            if sp is not None and sp[0] == b and sp[1] <= p and sp[2] == list(hist[b:b + sp[1]]):
                self.load1.launch()              # back to where the last prompt ended
                del h[sp[1]:]
                p = sp[1]
                self.resumes += 1
            else:                                # read it again (the last half window at most)
                self.e.reset()
                self.h = h = []
                self.base = b = max(0, pos - W // 2) if pos - b > W // 2 else b
                p = 0
                self.refeeds += 1
        if p < pos - b:
            new = list(hist[b + p:pos])
            if min(new) < 0:                     # an unknown token (-1): never fed to the draft model
                self.e.reset()
                self.h, self.snap = [], None
                return
            self.e.feed_batch(new, p)
            h.extend(new)

    def draft(self, hist, pos, k):
        """k greedy draft tokens after hist[:pos + 1] (hist[pos] = the token the main model reads now)"""
        if any(t < 0 for t in hist[max(0, pos - 8):pos + 1]) or k + 17 > self.e.ctx:
            return []
        self._sync(hist, pos, k)
        p2 = pos - self.base
        if p2 < 0 or p2 + k + 1 > self.e.ctx or len(self.h) != p2:
            return []
        if self.want_snap:
            self.save1.launch()
            self.snap, self.want_snap = (self.base, p2, list(self.h)), False
        s = self.e.cards[0]
        g = self.g.get(k)
        if g is None:
            with cp.cuda.Device(s.card):
                g = self.g[k] = Graph(build_draft(s, k, self.M0, self.C0))
        x = hist[pos]
        if self.mtp is not None and self.b2_on:
            # mode B2: one 2B step, then the 2B's head guesses the rest of the draft from that step's hidden state
            # (unchecked by the 2B; the main model verifies every token)
            g = self.g.get(1)
            if g is None:
                with cp.cuda.Device(s.card):
                    g = self.g[1] = Graph(build_draft(s, 1, self.M0, self.C0))
            g.launch([x] + [0] * TMAX, p2)
            d = [int(cp.asnumpy(s.tcb["AOUT"][:1])[0])]
            if k > 1:
                self.mtp.gap()
                self.mtp.seen(p2, 1, d)
                d += self.mtp.draft(d[0], p2 + 1, k - 1)
            self.pend, self.b2 = (p2, x, d), True
            return d
        g.launch([x] + [0] * TMAX, p2)
        d = [int(t) for t in cp.asnumpy(s.tcb["AOUT"][:k])]
        self.pend, self.b2 = (p2, x, d), False
        return d

    def accept(self, j):
        """the verify accepted d[:j]: the draft model's history becomes hist[base:pos] + [x] + d[:j]"""
        p2, x, d = self.pend
        self.pend = None
        if self.b2:                              # B2: the 2B read only x; back to before it, read x + the accepted
            self.restore.launch()
            self.e.feed_batch([x] + d[:j], p2)
            self.h[p2:] = [x] + d[:j]
            return
        if j == len(d):                          # all accepted: it read x, d[:-1]; d[-1] is still to read
            self.e.feed(d[-1], p2 + len(d))
            self.h[p2:] = [x] + d
        else:
            self.restore.launch()
            self.e.feed_batch([x] + d[:j], p2)
            self.h[p2:] = [x] + d[:j]


# ====================================================================================== streaming load
def load_stream(self):
    """Shard._load for one card from the cached lane-layout weights, read in 256 MB pieces straight into card
    memory (the full load holds the 5.6 GB file plus a pinned staging copy in host memory). False: not cached."""
    tp = self.tp
    key = "%s-w%d%s-c%d" % ((tp.man.get("source_sha256") or "pack")[:12], tp.W, getattr(tp, "split_tag", ""), self.card)
    ipath = os.path.join(tp.pack, "g5_%s.json" % key)
    lpath = os.path.join(tp.pack, "g5n_%s.bin" % key)
    if not (os.path.exists(ipath) and os.path.exists(lpath)):
        return False
    t0 = time.time()
    _, flt = self._slices()
    offs = {n: tuple(v) for n, v in json.load(open(ipath)).items()}
    nw = os.path.getsize(lpath) // 4
    self.bc = cp.empty(nw, dtype=cp.uint32)
    step = 64 << 20
    with open(lpath, "rb") as f:
        for o in range(0, nw, step):
            n = min(step, nw - o)
            self.bc[o:o + n].set(np.fromfile(f, dtype=np.uint32, count=n))
    self.dp = cp.zeros(1, dtype=cp.uint16)
    self.w = {n: ("g5", self.bc[o:], i32(R), i32(C)) for n, (o, R, C) in offs.items()}
    # float tensors: as Shard._load
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
    self.log("card %d: %.2f GB of weights, streamed (%.0fs)" % (self.card, self.bc.nbytes / 1e9, time.time() - t0))
    return True


# ====================================================================================== install
_installed = [False]
_drafter = [None]
_building_drafter = [False]


def install(log=print):
    """hook the tc path into the plan engine (call before TPEngine is built); no-op unless KINDI_TC=1"""
    if _installed[0] or not wanted():
        return
    _installed[0] = True
    o_load, o_reserve, o_build = _tp.Shard._load, _chunk.reserve, _chunk.build

    o_slices, o_bplans = _tp.Shard._slices, _tp.Shard.build_plans

    def _slices(self):
        tern, flt = o_slices(self)
        if "output.weight" not in self.tp.bi:          # tied embedding (the 2B): the head is token_embd
            tern["token_embd.weight"] = tern.pop("output.weight")
        self.head_name = "output.weight" if "output.weight" in self.tp.bi else "token_embd.weight"
        return tern, flt

    def _bplans(self):
        if getattr(self, "tc_layout", False):          # the tensor-core plans replace them (any Hadamard block)
            drv = _tp._drv()
            cp.cuda.Device(self.card).synchronize()
            if cp.cuda.runtime.is_hip:             # HIP: the device index (tp.py hipSetDevice(s.ctxh))
                c = ctypes.c_int()
                drv.hipGetDevice(ctypes.byref(c))
                self.ctxh = c.value
            else:                                  # CUDA: the current context
                c = ctypes.c_void_p()
                drv.cuCtxGetCurrent(ctypes.byref(c))
                self.ctxh = c
            return None
        return o_bplans(self)

    _tp.Shard._slices, _tp.Shard.build_plans = _slices, _bplans

    def _load(self):
        if not (self.tp.W in (1, 2) and self.g5l and _tc.supported(self.card) and load_stream(self)):
            o_load(self)
        if self.tp.W in (1, 2) and self.g5l and _tc.supported(self.card):
            self.sk = SpecKernels()
            repack(self, self.log)

    def _reserve(eng):
        o_reserve(eng)                          # the chunk buffers (chunk.py runs its GEMMs on the tc kernels)
        if getattr(eng.cards[0], "tc_layout", False):
            reserve(eng)

    def _build(eng):
        o_build(eng)
        if getattr(eng.cards[0], "tc_layout", False):
            build(eng, eng.log)
            if _drafter[0] is not None and eng is not _drafter[0]:
                eng.drafter = Draft2B(_drafter[0])
                eng.log("speculative draft model attached: %s, ctx %d" % (_drafter[0].pack, _drafter[0].ctx))
                if os.environ.get("KINDI_SPEC_DRAFT_HEAD"):
                    eng.drafter.mtp = _mtp.MTPDraft(_drafter[0], os.environ["KINDI_SPEC_DRAFT_HEAD"], log=eng.log)
            if os.environ.get("KINDI_SPEC") in ("mtp", "auto") and os.environ.get("KINDI_MTP_HEAD") and not _building_drafter[0]:
                eng.mtp = _mtp.MTPDraft(eng, os.environ["KINDI_MTP_HEAD"], log=eng.log)

    _tp.Shard._load = _load
    _chunk.reserve = _reserve
    _chunk.build = _build
    o_step, o_feed, o_fb, o_ylog = _tp.TPEngine.step, _tp.TPEngine.feed, _tp.TPEngine.feed_batch, _tp.TPEngine.ylog

    def step(self, tok, pos):
        return tc_step(self, tok, pos) if getattr(self, "tc", False) else o_step(self, tok, pos)

    def feed(self, tok, pos):
        return tc_feed(self, tok, pos) if getattr(self, "tc", False) else o_feed(self, tok, pos)

    def feed_batch(self, toks, pos):
        return tc_feed_batch(self, toks, pos) if getattr(self, "tc", False) else o_fb(self, toks, pos)

    def ylog(self):
        return tc_ylog(self) if getattr(self, "tc", False) else o_ylog.fget(self)

    o_init = _tp.TPEngine.__init__

    def init_(self, pack, ctx=24576, *a, **k):
        self._req_ctx = int(ctx)                  # the verify attention buffers are sized before the context fit
        os.environ["KINDI_BATCH"] = "1"           # (the tensor-core plans replace the batch plan)
        os.environ.setdefault("KINDI_MEM_RESERVE_MB", "256")   # the graphs are instantiated after the context fit
        sm = os.environ.get("KINDI_SPEC")
        hps = [os.environ.get("KINDI_MTP_HEAD") if sm in ("mtp", "auto") else None,
               os.environ.get("KINDI_SPEC_DRAFT_HEAD") if sm in ("draft2b", "both", "auto") else None]
        hps = [h for h in hps if h]
        if hps and not _building_drafter[0] and not getattr(self, "_mtp_rsv", 0):
            # the MTP heads' weights and buffers: kept out of the context fit (loaded after the build)
            self._mtp_rsv = sum(os.path.getsize(h) // 2 ** 20 + 160 for h in hps)
            os.environ["KINDI_MEM_RESERVE_MB"] = str(int(os.environ["KINDI_MEM_RESERVE_MB"]) + self._mtp_rsv)
        o_init(self, pack, ctx, *a, **k)

    _tp.TPEngine.__init__ = init_
    # KINDI_SPEC=draft2b|both: the draft model (KINDI_SPEC_PACK, a Ternary-Bonsai-2 2B pack) on the same card, built
    # first so the main model's context fit sees the memory it takes
    if os.environ.get("KINDI_SPEC", "0") in ("draft2b", "both", "auto") and os.environ.get("KINDI_SPEC_PACK"):
        _building_drafter[0] = True
        cards = os.environ.get("KINDI_CARDS")
        os.environ["KINDI_CARDS"] = "1"
        try:
            _drafter[0] = _tp.TPEngine(os.environ["KINDI_SPEC_PACK"], ctx=int(os.environ.get("KINDI_SPEC_CTX") or os.environ.get("KINDI_SPEC_WINDOW", "8192")),
                                       log=lambda *a: log("draft: " + " ".join(str(x) for x in a)))
        finally:
            if cards is None:
                os.environ.pop("KINDI_CARDS", None)
            else:
                os.environ["KINDI_CARDS"] = cards
        _building_drafter[0] = False
    o_reset = _tp.TPEngine.reset

    def reset_(self):
        self._sppend, self._spq = False, []
        _hist(self).truncate(0)
        if getattr(self, "mtp", None) is not None:
            self.mtp.reset()
        return o_reset(self)

    _tp.TPEngine.step, _tp.TPEngine.feed, _tp.TPEngine.feed_batch = step, feed, feed_batch
    _tp.TPEngine.reset = reset_
    _tp.TPEngine.spec_begin = lambda self, *a, **k: spec_begin(self, *a, **k) if getattr(self, "tc", False) else None
    _tp.TPEngine.spec_end = lambda self: spec_end(self) if getattr(self, "tc", False) else None
    _tp.TPEngine.spec_known = lambda self, toks: _known(self, toks) if getattr(self, "tc", False) else None
    _tp.TPEngine.ylog = property(ylog, lambda self, v: None)
