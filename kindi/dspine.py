"""DSpine block drafter for the Ternary-Bonsai-2 27B (Qwen3.8-27B family), on the same card as the target.

The drafter (Tencent's DSpine, Apache-2.0 weights; this file and dspine.cu are Kindi's own implementation) is a
5-layer transformer that drafts a whole block of tokens in one pass. Its context is the target's own hidden states:
the outputs of target layers 1, 16, 31, 46 and 61 of every token the target has read, concatenated and projected (fc)
to one vector per position, from which each draft layer keeps a K/V ring of the last 8192 positions. A draft block is
the current token followed by mask tokens; each layer attends to the ring and causally within the block, and a code
bridge between layers passes a low-rank message from each block row to the next. The block's rows go through the
target's own output norm and head (the tensor-core products of the ternary head), and a top-16 refinement chain picks
each draft token given the one before.

Exactness: the target is unchanged. The plans only copy the five layers' outputs into a feature buffer (k_feat_T), and
every draft token is verified by the target's T-token pass, so greedy output is the non-speculative greedy output token
for token, and sampled requests use spec.spec_accept (speculative sampling with a deterministic draft).

KINDI_SPEC=dspine (or auto, when KINDI_DSPINE names the weights directory): tc_step drafts with it. Block 8: the
current token and 7 drafts (the verify plans stop at T = 8). Weights: linears int4 (group 64, symmetric) on the card,
the bridge in fp16 on the card, the bridge codebook and the precomputed head x write table in mapped host memory."""
import ctypes
import json
import math
import os
import struct
import time

import numpy as np
import cupy as cp

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = (1, 16, 31, 46, 61)
NSLOT = len(LAYERS)
i32, f32 = np.int32, np.float32
NULL = np.uint64(0)                             # a null device pointer argument
BLOCK = 8                                       # the current token + 7 drafts (verify plans: T <= 8)
WIN = 8192                                      # the draft's sliding window (and its K/V ring)
TOPK = 16


def wanted(eng=None):
    """KINDI_SPEC=dspine|auto with KINDI_DSPINE set (the weights directory)"""
    return os.environ.get("KINDI_SPEC", "0") in ("dspine", "auto") and bool(os.environ.get("KINDI_DSPINE"))


def fits(s):
    """the drafter's target: the 27B shape on one NVIDIA card with the tensor-core plans (no lane layout, no HP)"""
    from . import spec as _spec
    return (s.D == 5120 and s.NL == 64 and s.tp.W == 1 and not cp.cuda.runtime.is_hip and not getattr(s, "tc_lane", False)
            and not _spec.HP and getattr(s, "tc_layout", False))


def reserve_mb():
    return int(os.environ.get("KINDI_DSPINE_MB", "1400"))


_mod = {}


def kernels():
    dev = cp.cuda.Device().id
    m = _mod.get(dev)
    if m is None:
        from . import tc as _tc
        src = open(os.path.join(HERE, "dspine.cu"), encoding="utf-8").read()
        inc = "-I" + os.path.join(cp.cuda.get_cuda_path() or "/usr/local/cuda", "include")      # cuda_fp16.h
        if _tc.pascal():
            m = cp.RawModule(code=src, options=("--std=c++14", inc))
        else:                                    # sm_80+: compute_75 PTX retargeted (the image's NVRTC predates sm_86)
            p = cp.cuda.runtime.getDeviceProperties(dev)
            m = _tc._patched_module(src, "sm_%d%d" % (p["major"], p["minor"]), ("--std=c++14", inc))
        _mod[dev] = m
    return m


def feat_kernel():
    return kernels().get_function("k_feat_T")


# ====================================================================================== weights
def _st_index(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


class _ST(object):
    """the tensors of the safetensors files in a directory (read on demand, BF16 / F32)"""

    def __init__(self, d):
        self.t = {}
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".safetensors"):
                p = os.path.join(d, fn)
                hdr, base = _st_index(p)
                for k, v in hdr.items():
                    if k != "__metadata__":
                        self.t[k[6:] if k.startswith("model.") else k] = (p, base, v)

    def shape(self, k):
        return tuple(self.t[k][2]["shape"])

    def host(self, k, rows=None):
        """float32 numpy array (rows: a (start, stop) slice of the first dimension)"""
        p, base, v = self.t[k]
        s0, s1 = v["data_offsets"]
        shp = v["shape"]
        es = {"BF16": 2, "F16": 2, "F32": 4}[v["dtype"]]
        rl = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        r0, r1 = rows if rows is not None else (0, shp[0])
        with open(p, "rb") as f:
            f.seek(base + s0 + r0 * rl * es)
            raw = f.read((r1 - r0) * rl * es)
        if v["dtype"] == "BF16":
            a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F16":
            a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            a = np.frombuffer(raw, dtype=np.float32).copy()
        return a.reshape((r1 - r0,) + tuple(shp[1:]))


class QAct(object):
    """int8 activations of up to 16 tokens (k_q8act): per group of 64, the values, the scale and the sum"""

    def __init__(self, M, Kmax):
        self.q = cp.zeros(16 * Kmax // 4, dtype=cp.uint32)
        self.s = cp.zeros(16 * Kmax // 64, dtype=cp.float32)
        self.sum = cp.zeros(16 * Kmax // 64, dtype=cp.int32)
        self.k = M.get_function("k_q8act")
        self.K = 0

    def emit(self, E, X, ldx, n, K):
        assert n <= 16 and K % 64 == 0
        E(self.k, (K // 64, n), (64,), (X, i32(ldx), i32(K), self.q, self.s, self.sum))
        self.K = K


class Lin4(object):
    """an int4 linear (N outputs x K inputs, group 64, symmetric) on the card: k_q4mm8 / k_q4mm16 (dp4a with int8
    activations from QAct)"""

    def __init__(self, st, name, M):
        N, K = st.shape(name)
        assert K % 64 == 0, (name, K)
        self.N, self.K = N, K
        self.w = cp.empty(N * (K // 8), dtype=cp.uint32)
        self.s = cp.empty(N * (K // 64), dtype=cp.float16)
        step = max(1, (64 << 20) // (K * 4))
        for r0 in range(0, N, step):
            r1 = min(N, r0 + step)
            a = cp.asarray(st.host(name, (r0, r1))).reshape(r1 - r0, K // 64, 64)
            sc = cp.maximum(cp.abs(a).max(axis=2, keepdims=True), 1e-12) / 7.0
            q = (cp.clip(cp.rint(a / sc), -7, 7) + 8).astype(cp.uint8).reshape(r1 - r0, K)
            b = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(cp.uint8)
            self.w[r0 * (K // 8):r1 * (K // 8)] = cp.ascontiguousarray(b).view(cp.uint32).reshape(-1)
            self.s[r0 * (K // 64):r1 * (K // 64)] = sc.reshape(-1).astype(cp.float16)
            del a, sc, q, b
        # 4 rows per warp (32 per block) for every shape: one row per warp (k_q4mm*n) was slower on the 3080 for the
        # narrow products too (q 0.075 vs 0.053 ms, K/V 0.026 vs 0.023 ms; KINDI_DSPINE_NARROW=1 tries it)
        self.rows = 8 if (N <= 4096 and os.environ.get("KINDI_DSPINE_NARROW") == "1") else 32
        sfx = "n" if self.rows == 8 else ""
        self.k8, self.k16 = M.get_function("k_q4mm8" + sfx), M.get_function("k_q4mm16" + sfx)
        self._qa = None
        self.M = M

    def mm(self, E, qa, Y, ldy, n, beta=0.0):
        """Y[:n] (+)= x W^T for the n (<= 16) tokens qa holds"""
        assert qa.K == self.K and n <= 16
        E(self.k8 if n <= 8 else self.k16, ((self.N + self.rows - 1) // self.rows,), (256,),
          (self.w, self.s, qa.q, qa.s, qa.sum, Y, i32(ldy), i32(self.N), i32(self.K), i32(n), f32(beta)))

    def __call__(self, X, ldx, Y, ldy, n, beta=0.0):
        """Y[:n] (+)= X[:n] W^T now (any n: in pieces of 16; its own activation buffer)"""
        if self._qa is None:
            self._qa = QAct(self.M, self.K)
        for m0 in range(0, n, 16):
            m = min(16, n - m0)
            self._qa.emit(_direct, X[m0 * ldx:], ldx, m, self.K)
            self.mm(_direct, self._qa, Y[m0 * ldy:], ldy, m, beta)


def _direct(fn, grid, block, args):
    """a launch now (the plan form: _Planner)"""
    fn(grid, block, args)


class _Fk(object):
    """a cupy Function with the .kernel the plans read (kern.kernel.ptr)"""

    def __init__(self, f):
        self.kernel = f


class _Planner(object):
    """E(fn, grid, block, args) -> a node of a Kindi plan (replayed as one CUDA graph)"""

    def __init__(self):
        from . import tp as _tp
        self.P = _tp.Plan()

    def __call__(self, fn, grid, block, args):
        self.P.add(fn if hasattr(fn, "kernel") else _Fk(fn), grid, block, args)


def _mapped(a):
    """a numpy array copied into mapped pinned host memory; returns (memory, device address)"""
    flags = cp.cuda.runtime.hostAllocMapped | cp.cuda.runtime.hostAllocPortable
    mem = cp.cuda.PinnedMemory(a.nbytes, flags=flags)
    np.ctypeslib.as_array((ctypes.c_uint8 * a.nbytes).from_address(mem.ptr))[:] = a.reshape(-1).view(np.uint8)
    return mem, mem.ptr


# ====================================================================================== the drafter
class DSpine(object):
    def __init__(self, eng, d, log=print):
        t0 = time.time()
        self.e = eng
        s = self.s = eng.cards[0]
        self.log = log
        cfg = json.load(open(os.path.join(d, "config.json")))
        dc = cfg["dflash_config"]
        if tuple(dc["target_layer_ids"]) != LAYERS:
            raise SystemExit("kindi dspine: target layers %r (Kindi exports %r)" % (dc["target_layer_ids"], LAYERS))
        self.H, self.NH, self.NKV, self.HD = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.NL, self.I = cfg["num_hidden_layers"], cfg["intermediate_size"]
        self.eps = float(cfg["rms_norm_eps"])
        self.theta = float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"])
        self.mask = int(dc["mask_token_id"])
        br = dc["selected_prefix_code_bridge"]
        self.C, self.R, self.beps = int(br["code_size"]), int(br["rank"]), float(br["eps"])
        self.sqc = math.sqrt(self.C)
        assert (self.H, self.NH, self.NKV, self.HD) == (s.D, 32, 8, 128), "kindi dspine: unexpected shapes"
        H, C, R, T = self.H, self.C, self.R, BLOCK
        with cp.cuda.Device(s.card):
            M = self.M = kernels()
            self.kf = {n: M.get_function(n) for n in (
                "k_rms_rows", "k_headnorm_rope", "k_ring_store", "k_ring_mod", "k_rpos_set", "k_rpos_cut", "k_att_part",
                "k_att_comb", "k_silu_mul", "k_gather_h", "k_gather_tok", "k_gather_sid", "k_h16mm8", "k_h16mm16",
                "k_bridge_a", "k_bridge_codes", "k_bridge_msg", "k_bridge_out", "k_top16a", "k_top16b", "k_tail_base",
                "k_tail_sc", "k_chain")}
            st = _ST(d)
            f32a = lambda k: cp.asarray(st.host(k).reshape(-1))  # noqa: E731
            f16a = lambda k: cp.asarray(st.host(k)).astype(cp.float16)  # noqa: E731
            self.fc = Lin4(st, "fc.weight", M)
            self.qa = QAct(M, NSLOT * H)                         # (the largest input: fc's)
            self.hidden_norm = f32a("hidden_norm.weight")
            self.L = []
            for l in range(self.NL):
                p = "layers.%d." % l
                a = p + "self_attn."
                self.L.append({
                    "q": Lin4(st, a + "q_proj.weight", M), "k": Lin4(st, a + "k_proj.weight", M),
                    "v": Lin4(st, a + "v_proj.weight", M), "o": Lin4(st, a + "o_proj.weight", M),
                    "qn": f32a(a + "q_norm.weight"), "kn": f32a(a + "k_norm.weight"),
                    "ln1": f32a(p + "input_layernorm.weight"), "ln2": f32a(p + "post_attention_layernorm.weight"),
                    "g": Lin4(st, p + "mlp.gate_proj.weight", M), "u": Lin4(st, p + "mlp.up_proj.weight", M),
                    "d": Lin4(st, p + "mlp.down_proj.weight", M)})
            self.norm = f32a("norm.weight")
            b = "selected_prefix_bridge."
            self.packed = cp.concatenate([f16a(b + "gate.weight"), f16a(b + "predecessor_gate.weight"),
                                          f16a(b + "prev.weight")], 0)                 # (3R, C)
            self.pv = self.packed[R:]                                                    # (2R, C): the tail's rows
            self.gate_b = f32a(b + "gate.bias")
            early = st.host(b + "early_read_weights")                                    # (NL-1, C, H)
            self.read = [cp.asarray(early[l]).astype(cp.float16) for l in range(self.NL - 1)] + [f16a(b + "read_weight")]
            del early
            fw = st.host(b + "feature_write_weights")                                    # (NL, H, R)
            self.write = [cp.asarray(fw[l]).astype(cp.float16) for l in range(self.NL)]
            wlast = fw[-1]
            del fw
            # the codebook (V x C) in mapped host memory, fp16 (rows gathered per draft: the anchor, the tail's ids)
            cbook = st.host(b + "codebook").astype(np.float16)
            self.V = cbook.shape[0]
            self._cb_mem, self.cb = _mapped(cbook)
            del cbook
            # buffers (block rows: T = BLOCK; tail rows: n = T - 1, 16 candidates each)
            n, D, I, nq = T - 1, H, self.I, self.NH * self.HD
            z = lambda *k: cp.zeros(int(np.prod(k)), dtype=cp.float32)  # noqa: E731
            zi = lambda *k: cp.zeros(int(np.prod(k)), dtype=cp.int32)  # noqa: E731
            self.x, self.res, self.xin, self.hh = z(T, D), z(T, D), z(T, D), z(T, D)
            self.q, self.qr = z(T, nq), z(T, nq)
            self.kb, self.vb, self.kr = z(T, 1024), z(T, 1024), z(T, 1024)
            self.att = z(T, nq)
            self.u1, self.u2 = z(T, I), z(T, I)
            self.nrm, self.delta = z(T, D), z(T, D)
            self.proj, self.codes = z(T, C), z(T, C)
            self.pk, self.msg, self.rr = z(T, 3 * R), z(T, R), z(T)
            self.ctx, self.ctxn = z(16, D), z(16, D)
            self.ck, self.cv = z(16, 1024), z(16, 1024)
            self.Kr = [cp.zeros((WIN + 16) * 1024, dtype=cp.float16) for _ in range(self.NL)]
            self.Vr = [cp.zeros((WIN + 16) * 1024, dtype=cp.float16) for _ in range(self.NL)]
            self.rpos = cp.full(WIN, -1, dtype=cp.int32)
            self.nch = (WIN + T + 255) // 256
            self.Op = z(8 * self.nch * 32 * 128)
            self.Mp, self.Lp = z(8 * self.nch * 32), z(8 * self.nch * 32)
            self.cb0 = z(C)
            self.NB = 64                                               # top-16 stage 1 blocks per row
            self.bv, self.bi, self.cand = z(n, self.NB, 16), zi(n, self.NB, 16), zi(n, 16)
            self.base, self.idx, self.dout = z(n, 16), zi(n, 16), zi(T)
            self.cbr, self.tsv, self.tg = z(n * 16, C), z(n * 16, 2 * R), z(n * 16, R)
            # the head on the card's own products
            self._head_init()
            self.hY2 = cp.zeros(T * self.hR, dtype=cp.float32)
            # the mask token's embedding: block rows 1 .. T-1 of x, fixed
            self.emask = z(D)
            self._embed(self.mask, self.emask)
            for t in range(1, T):
                self.x[t * D:(t + 1) * D] = self.emask
            # the head x write table G = head . (norm (.) write_last), (V, R), for the tail
            self._G_init(cp.asarray(wlast))
            cp.get_default_memory_pool().free_all_blocks()
            # the draft and the small ingests as graphs (one launch each; tok and pos are plan arguments)
            from . import tp as _tp
            from .spec import Graph
            Pd = _Planner()
            self._draft_body(Pd, _tp.Dyn("tok", 0, 1), _tp.Dyn("pos", 0, 1))
            self.gd = Graph(Pd.P)
            self.gi = {}
            F = s.tcb["FEAT"]
            for m in range(1, T + 1):
                Pi = _Planner()
                self._ingest_body(Pi, F, m, _tp.Dyn("pos", 0, 1))
                self.gi[m] = Graph(Pi.P)
            self.nodes = len(self.gd.nodes)
        self.reset()
        self.stats = {"drafts": 0, "ingested": 0, "draft_s": 0.0}
        log("dspine drafter ready: %d layers, int4 linears, ring %d, block %d, %d graph nodes per draft (%.1fs)" % (
            self.NL, WIN, BLOCK, self.nodes, time.time() - t0))

    # ------------------------------------------------------------------ the target's pieces
    def _embed(self, tok, out):
        s = self.s
        emb = s.tp.emb
        s.k.k_deq_emb4((s.D // s.HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, out, i32(tok),
                                                i32(s.D), i32(s.HB)))

    def _head_init(self):
        from . import spec as _spec
        s, D = self.s, self.s.D
        _, Wt, R, C = s.w[s.head_name]
        self.hR, self.hC = int(R), int(C)
        nw, sp = _spec.tune(s, self.hR, self.hC)
        self.hnw, self.hsp = nw, sp
        self.hmv = s.sk.mv[nw]
        self.hW = Wt
        if getattr(s, "w2", None):
            self.hW, self.hmv = s.w2[s.head_name], s.sk.mvb[nw]
        T = BLOCK
        z = lambda n, t=cp.float32: cp.zeros(int(n), dtype=t)  # noqa: E731
        NG = D // 128
        self.hXN, self.hXP, self.hH2 = z(T * D), z(T * D), z(T * D)
        self.hXQ, self.hXS, self.hXA = z(T * NG * 32, cp.uint32), z(T * NG), z(T * NG, cp.int32)
        self.hY = z(T * self.hR)
        self.hP = z(sp * T * self.hR) if sp > 1 else self.hY
        self.ones = cp.ones(D, dtype=cp.float32)
        self.nug = _spec.nug(self.hC)

    def head(self, X, n, w, Y, E=_direct):
        """Y[:n] (vocabulary rows) = the target head of rms(X[:n]) * w (n <= 8): the target's own fold, int8 rows and
        tensor-core products"""
        s, SK, D = self.s, self.s.sk, self.s.D
        E(SK.k_rmsfold_qT, (D // s.HB, n), (512,), (X, X, self.hH2, w, s.sD, self.hXN, self.hXP, i32(D), i32(0), i32(s.HB),
                                                    i32(n), i32(D), i32(D), i32(D), i32(D), i32(D), self.hXQ, self.hXS,
                                                    self.hXA, i32(0)))
        gx = (self.hR // 16 + self.hnw - 1) // self.hnw
        out = Y if self.hsp == 1 else self.hP
        E(self.hmv, (gx, self.hsp), (32 * self.hnw,), (self.hW, out, i32(self.hR), i32(self.nug), self.hXQ, self.hXS,
                                                       self.hXA, i32(n)))
        if self.hsp > 1:
            E(SK.k_tc_sum, ((self.hR * n + 255) // 256,), (256,), (self.hP, Y, i32(self.hR), i32(n), i32(self.hsp)))

    def _G_init(self, wl):
        """G[v] = head[v] . (norm * write_last[:, j]) for j < R: the tail's rank-R term (fp16, mapped host memory)"""
        D, R = self.H, self.R
        U = (wl * self.norm[:, None]).T.copy()                   # (R, H): the R vectors
        G = np.empty((self.V, R), dtype=np.float16)
        X = cp.zeros(BLOCK * D, dtype=cp.float32)
        for j0 in range(0, R, BLOCK):
            n = min(BLOCK, R - j0)
            u = U[j0:j0 + n]
            r = cp.sqrt((u * u).mean(axis=1))
            X[:n * D] = (u / r[:, None]).reshape(-1)
            self.head(X, n, self.ones, self.hY)
            y = self.hY[:n * self.hR].reshape(n, self.hR)[:, :self.V] * r[:, None]
            G[:, j0:j0 + n] = cp.asnumpy(y.T).astype(np.float16)
        self._g_mem, self.Gp = _mapped(G)
        del G

    # ------------------------------------------------------------------ context
    def reset(self):
        with cp.cuda.Device(self.s.card):
            self.rpos.fill(-1)
        self.top = 0                                             # the next position to read (the ring holds < top)

    def truncate(self, pos):
        """drop positions >= pos (a rewind)"""
        with cp.cuda.Device(self.s.card):
            self.kf["k_rpos_cut"](((WIN + 255) // 256,), (256,), (self.rpos, i32(pos), i32(WIN)))
        self.top = min(self.top, pos)

    def _ingest_body(self, E, F, n, pos):
        """rows F[:n] (NSLOT x D floats each) = positions pos .. into every layer's ring (n <= 16)"""
        D, kf = self.H, self.kf
        self.qa.emit(E, F, NSLOT * D, n, NSLOT * D)
        self.fc.mm(E, self.qa, self.ctx, D, n)
        E(kf["k_rms_rows"], (n,), (256,), (self.ctx, i32(D), self.hidden_norm, i32(D), self.ctxn, i32(D), i32(D), f32(self.eps)))
        self.qa.emit(E, self.ctxn, D, n, D)
        for l, Ly in enumerate(self.L):
            Ly["k"].mm(E, self.qa, self.ck, 1024, n)
            Ly["v"].mm(E, self.qa, self.cv, 1024, n)
            E(kf["k_headnorm_rope"], (n, 8), (128,), (self.ck, i32(1024), Ly["kn"], self.ck, i32(1024), NULL, pos,
                                                       f32(self.theta), f32(self.eps)))
            E(kf["k_ring_mod"], (n,), (256,), (self.ck, i32(1024), self.Kr[l], pos, i32(WIN)))
            E(kf["k_ring_mod"], (n,), (256,), (self.cv, i32(1024), self.Vr[l], pos, i32(WIN)))
        E(kf["k_rpos_set"], (1,), (256,), (self.rpos, pos, i32(n), i32(WIN)))

    def ingest(self, F, pos, n):
        """the features of positions pos .. pos + n - 1 (F: n rows of NSLOT x D floats) into every layer's ring"""
        if n <= 0:
            return
        if pos < self.top:
            self.truncate(pos)
        s = self.s
        with cp.cuda.Device(s.card):
            if n <= BLOCK and F.data.ptr == s.tcb["FEAT"].data.ptr:
                self.gi[n].launch([0] * 8, pos)
            else:
                for m0 in range(0, n, 16):
                    self._ingest_body(_direct, F[m0 * NSLOT * self.H:], min(16, n - m0), i32(pos + m0))
        self.top = max(self.top, pos + n)
        self.stats["ingested"] += n

    # ------------------------------------------------------------------ the draft
    def _draft_body(self, E, tok, pos):
        """the block (tok at pos, then mask tokens) through the 5 layers and the bridge, the head and the tail: the
        BLOCK - 1 draft tokens in dout"""
        s, SK, kf = self.s, self.s.sk, self.kf
        T, H, C, R, I, nq = BLOCK, self.H, self.C, self.R, self.I, self.NH * self.HD
        n = T - 1
        emb = s.tp.emb
        cpy = lambda a, b, m: E(SK.k_copyf, (min(1024, (m + 255) // 256),), (256,), (a, b, np.uint64(m)))  # noqa: E731
        E(s.k.k_deq_emb4, (H // s.HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, self.x, tok,
                                                 i32(H), i32(s.HB)))
        cpy(self.x, self.res, T * H)
        E(kf["k_rms_rows"], (T,), (256,), (self.x, i32(H), self.L[0]["ln1"], i32(H), self.xin, i32(H), i32(H), f32(self.eps)))
        E(kf["k_gather_tok"], (1,), (256,), (np.uint64(self.cb), tok, i32(C), f32(1.0), self.cb0))
        for l, Ly in enumerate(self.L):
            self.qa.emit(E, self.xin, H, T, H)
            Ly["q"].mm(E, self.qa, self.q, nq, T)
            Ly["k"].mm(E, self.qa, self.kb, 1024, T)
            Ly["v"].mm(E, self.qa, self.vb, 1024, T)
            E(kf["k_headnorm_rope"], (T, self.NH), (128,), (self.q, i32(nq), Ly["qn"], self.qr, i32(nq), NULL, pos,
                                                             f32(self.theta), f32(self.eps)))
            E(kf["k_headnorm_rope"], (T, 8), (128,), (self.kb, i32(1024), Ly["kn"], self.kr, i32(1024), NULL, pos,
                                                       f32(self.theta), f32(self.eps)))
            E(kf["k_ring_store"], (T,), (256,), (self.kr, i32(1024), self.Kr[l], NULL, i32(WIN)))
            E(kf["k_ring_store"], (T,), (256,), (self.vb, i32(1024), self.Vr[l], NULL, i32(WIN)))
            E(kf["k_att_part"], (8, self.nch), (256,), (self.qr, i32(nq), self.Kr[l], self.Vr[l], self.rpos, i32(WIN), i32(T),
                                                         pos, i32(WIN), f32(1.0 / math.sqrt(self.HD)), self.Op, self.Mp,
                                                         self.Lp, i32(self.nch)))
            E(kf["k_att_comb"], (8, 4 * T), (128,), (self.Op, self.Mp, self.Lp, i32(self.nch), i32(T), self.att, i32(nq)))
            cpy(self.res, self.hh, T * H)                                  # h = res + o(att)
            self.qa.emit(E, self.att, nq, T, nq)
            Ly["o"].mm(E, self.qa, self.hh, H, T, beta=1.0)
            E(kf["k_rms_rows"], (T,), (256,), (self.hh, i32(H), Ly["ln2"], i32(H), self.xin, i32(H), i32(H), f32(self.eps)))
            self.qa.emit(E, self.xin, H, T, H)
            Ly["g"].mm(E, self.qa, self.u1, I, T)
            Ly["u"].mm(E, self.qa, self.u2, I, T)
            E(kf["k_silu_mul"], ((T * I + 255) // 256,), (256,), (self.u1, self.u2, self.u1, i32(T * I)))
            self.qa.emit(E, self.u1, I, T, I)
            Ly["d"].mm(E, self.qa, self.hh, H, T, beta=1.0)
            # the code bridge: res = hh + r (msg . write), xin = rms(res) * the next norm
            E(kf["k_bridge_a"], (T,), (256,), (self.hh, i32(H), f32(self.beps), self.nrm, self.rr))
            E(kf["k_h16mm8"], ((C + 31) // 32,), (256,), (self.read[l], self.nrm, i32(H), self.proj, i32(C), i32(C), i32(H),
                                                          i32(T), f32(0.0)))
            E(kf["k_bridge_codes"], (T,), (256,), (self.proj, i32(C), f32(self.sqc), f32(self.beps), self.cb0, self.codes))
            E(kf["k_h16mm8"], ((3 * R + 31) // 32,), (256,), (self.packed, self.codes, i32(C), self.pk, i32(3 * R), i32(3 * R),
                                                              i32(C), i32(T), f32(0.0)))
            E(kf["k_bridge_msg"], (T,), (256,), (self.pk, self.gate_b, i32(R), self.msg))
            E(kf["k_h16mm8"], ((H + 31) // 32,), (256,), (self.write[l], self.msg, i32(R), self.delta, i32(H), i32(H), i32(R),
                                                          i32(T), f32(0.0)))
            E(kf["k_bridge_out"], (T,), (256,), (self.hh, self.delta, self.rr, i32(H), self.res))
            nxt = self.norm if l == self.NL - 1 else self.L[l + 1]["ln1"]
            E(kf["k_rms_rows"], (T,), (256,), (self.res, i32(H), nxt, i32(H), self.xin, i32(H), i32(H), f32(self.eps)))
        # rows 1 .. T-1: 16 candidates each from the target head of the block rows, base scores from the raw rows
        self.head(self.res[H:], n, self.norm, self.hY, E)
        E(kf["k_top16a"], (n, self.NB), (256,), (self.hY, i32(self.hR), i32(self.V), self.bv, self.bi))
        E(kf["k_top16b"], (n,), (1024,), (self.bv, self.bi, i32(self.NB * 16), self.cand))
        self.head(self.hh[H:], n, self.norm, self.hY2, E)
        E(kf["k_tail_base"], (n,), (32,), (self.hY2, i32(self.hR), self.cand, self.rr[1:], self.base))
        E(kf["k_gather_sid"], (n * 16,), (256,), (np.uint64(self.cb), self.cand, tok, i32(C), f32(self.sqc), self.cbr))
        for m0 in range(0, n * 16, 16):
            E(kf["k_h16mm16"], ((2 * R + 31) // 32,), (256,), (self.pv, self.cbr[m0 * C:], i32(C), self.tsv[m0 * 2 * R:],
                                                               i32(2 * R), i32(2 * R), i32(C), i32(16), f32(0.0)))
        E(kf["k_gather_h"], (n * 16,), (256,), (np.uint64(self.Gp), self.cand, i32(R), f32(1.0), self.tg))
        E(kf["k_tail_sc"], (n, 16), (256,), (self.pk, self.tsv, self.gate_b, self.tg, self.base, self.rr[1:], i32(R),
                                             self.idx))
        E(kf["k_chain"], (1,), (32,), (self.idx, self.cand, i32(n), self.dout))

    def draft(self, tok, pos, k):
        """k <= BLOCK - 1 draft tokens after token tok at position pos (the ring holds the positions before pos)"""
        if k <= 0:
            return []
        t0 = time.perf_counter()
        with cp.cuda.Device(self.s.card):
            self.gd.launch([int(tok)] + [0] * 7, pos)
            d = cp.asnumpy(self.dout[:BLOCK - 1])
        self.stats["drafts"] += 1
        self.stats["draft_s"] += time.perf_counter() - t0
        return [int(x) for x in d[:k]]
