"""Kindi CPU engine (KINDI_DEVICE=cpu): the 2B / 4B / 9B models on the CPU, token-exact with llama.cpp.

The whole forward pass of a token runs in C (kcpu.c, one OpenMP parallel region per token). Same interface as
kindi.engine.Engine: reset(), feed(tok, pos), step(tok, pos) -> greedy token, generate(ids, n), logits in
self.ylog[:self.ylog_n]. Loads a Kindi pack (kindi convert) and repacks the ternary weights into pair codes
("perm8": per 8-row block and 128-column group, 64 words of 32 bits; one vpermps gives 8 rows x 2 columns).
kcpu.c is built on first use with gcc (-O3 -march=native -fopenmp) into KINDI_CPU_CACHE (default ~/.cache/kindi).
Threads: OMP_NUM_THREADS (with OMP_PROC_BIND=close; on a two-socket host, the cores of one socket)."""
import ctypes
import json
import os
import time

import numpy as np

VOCAB = 248320
DEFAULT_DIMS = {"layers": 24, "width": 2048, "ffn": 6144, "heads": 8, "kv_heads": 2,
                "groups": 16, "head_dim": 256, "rope_dim": 64, "state_size": 128, "gdn_inner": 2048,
                "rope_base": 1e7, "full_attention_interval": 4}
HERE = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32
PF = ctypes.POINTER(ctypes.c_float)
PU8 = ctypes.POINTER(ctypes.c_uint8)
PU16 = ctypes.POINTER(ctypes.c_uint16)


class Mat(ctypes.Structure):
    _fields_ = [("W", PU8), ("S", PF), ("R", ctypes.c_int), ("C", ctypes.c_int)]


class Layer(ctypes.Structure):
    _fields_ = [("is_attn", ctypes.c_int), ("attn_norm", PF), ("post_norm", PF),
                ("qkv", Mat), ("gate", Mat), ("ssm_out", Mat),
                ("beta_w", PF), ("alpha_w", PF), ("dt_b", PF), ("ssm_a", PF), ("conv_w", PF), ("ssm_norm", PF),
                ("M", PF), ("conv", PF),
                ("q", Mat), ("k", Mat), ("v", Mat), ("o", Mat), ("q_norm", PF), ("k_norm", PF),
                ("Kc", PU16), ("Vc", PU16),
                ("ffg", Mat), ("ffu", Mat), ("ffd", Mat)]


_SCR = ["h", "xn", "xp", "y", "qkv", "zv", "cs", "qk", "gout", "outn", "m", "qn", "kr", "vv", "qr", "kro", "sc",
        "att", "ffg", "ffu", "T", "logits", "gb", "gg"]


class Model(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_int) for n in ("D", "NL", "NFF", "NH", "NHKV", "HD", "NK", "HK", "NV", "CONV_CH", "HB",
                                             "rope_half", "ctx", "vocab")]
                + [("L", ctypes.POINTER(Layer)), ("emb", Mat), ("head", Mat)]
                + [(n, PF) for n in ("out_norm", "sD", "sV", "sA", "sF", "cos_t", "sin_t")]
                + [(n, PF) for n in _SCR] + [("red", ctypes.POINTER(ctypes.c_double))])


def _lib():
    src = os.path.join(HERE, "kcpu.c")
    cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, "libkcpu.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        if os.system("gcc -O3 -march=native -fopenmp -fno-math-errno -shared -fPIC -o %s %s -lm" % (so, src)):
            raise SystemExit("kindi: could not build the CPU engine (%s); it needs gcc with OpenMP" % so)
    lib = ctypes.CDLL(so)
    lib.kc_init()
    lib.kc_step.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kc_step.restype = ctypes.c_int
    assert lib.kc_sizeof_model() == ctypes.sizeof(Model) and lib.kc_sizeof_layer() == ctypes.sizeof(Layer)
    return lib


_CODE = np.array([0, 1, 2, 3, 8, 4, 5, 6, 7], dtype=np.uint32)     # t0 + 3 t1 -> pair code (4 = both zero -> 8)


def paircodes(codes, R, C, chunk=4096):
    """Pack bytes [C/4][R] (base-3, 4 trits each) -> uint32 [R/8][C/128][64]: word p = the 8 rows' 4-bit codes of
    columns 2p, 2p+1 (row-block major: one thread's rows are one contiguous stream)."""
    G = C // 128
    out = np.empty((R // 8, G, 64), dtype=np.uint32)
    codes = codes.reshape(C // 4, R)
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        b = np.asarray(codes[:, r0:r1]).T.astype(np.uint8)                  # [rows][C/4]
        t = np.stack([b % 3, b // 3 % 3, b // 9 % 3, b // 27 % 3], axis=2).reshape(r1 - r0, G, 64, 2)
        pc = _CODE[t[..., 0].astype(np.uint32) + 3 * t[..., 1].astype(np.uint32)]      # [rows][G][64]
        pc = pc.reshape((r1 - r0) // 8, 8, G, 64)
        word = np.zeros(((r1 - r0) // 8, G, 64), dtype=np.uint32)
        for l in range(8):
            word |= pc[:, l] << np.uint32(4 * l)
        out[r0 // 8:r1 // 8] = word
    return out


def _p(a, t=PF):
    return a.ctypes.data_as(t)


class Engine(object):
    def __init__(self, pack, ctx=4096, clusters=None, log=print, threads=None):
        t0 = time.time()
        if threads:
            os.environ["OMP_NUM_THREADS"] = str(threads)
        self.lib = _lib()
        self.ctx = int(ctx)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        d = self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))
        self.head_name = man.get("head", "token_embd.weight")
        D, NL, NFF = d["width"], d["layers"], d["ffn"]
        NH, NHKV, HD = d["heads"], d["kv_heads"], d["head_dim"]
        NK, HK = d["groups"], d["state_size"]
        NV = d["gdn_inner"] // HK
        CC = 2 * NK * HK + NV * HK
        interval = d["full_attention_interval"]
        idx = json.load(open(os.path.join(pack, "index.json")))["tensors"]
        bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        bc = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
        dp = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.float16, mode="r")
        self.keep = []                    # every array the C side points at

        def keep(a):
            self.keep.append(a)
            return a
        mats = {}
        for name, rec in bi.items():
            R, C = rec["rows"], rec["ne0"]
            ns, G = C // 4, C // 128
            o, q = rec["bc_off"], rec["dpm_off"] // 2
            W = keep(paircodes(bc[o:o + ns * R], R, C))
            S = keep(np.ascontiguousarray(np.asarray(dp[q:q + G * R]).reshape(G, R // 8, 8).transpose(1, 0, 2).astype(F32)))
            mats[name] = Mat(_p(W, PU8), _p(S), R, C)
        fl = keep(np.fromfile(os.path.join(pack, "f32.bin"), dtype=F32))

        def f(name):
            rec = idx[name]
            off = rec["f32_off"] // 4
            return _p(fl[off:off + rec["rows"] * rec["ne0"]])
        sg = lambda w: keep(np.fromfile(os.path.join(pack, "signs_%d.f32" % w), dtype=F32))  # noqa: E731
        z = lambda n, t=F32: keep(np.zeros(n, dtype=t))  # noqa: E731
        Ls = (Layer * NL)()
        self.M, self.conv, self.Kc, self.Vc = [], [], [], []
        for il in range(NL):
            p = "blk.%d." % il
            L = Ls[il]
            L.is_attn = int((il + 1) % interval == 0)
            L.attn_norm, L.post_norm = f(p + "attn_norm.weight"), f(p + "post_attention_norm.weight")
            L.ffg, L.ffu, L.ffd = mats[p + "ffn_gate.weight"], mats[p + "ffn_up.weight"], mats[p + "ffn_down.weight"]
            if L.is_attn:
                L.q, L.k, L.v, L.o = (mats[p + "attn_q.weight"], mats[p + "attn_k.weight"], mats[p + "attn_v.weight"],
                                      mats[p + "attn_output.weight"])
                L.q_norm, L.k_norm = f(p + "attn_q_norm.weight"), f(p + "attn_k_norm.weight")
                Kc, Vc = z(self.ctx * NHKV * HD, np.uint16), z(self.ctx * NHKV * HD, np.uint16)
                # token-major views ([ctx][NHKV][HD], as on the GPU engines) for the prompt cache; same memory
                self.Kc.append(Kc.reshape(self.ctx, NHKV, HD))
                self.Vc.append(Vc.reshape(self.ctx, NHKV, HD))
                L.Kc, L.Vc = _p(Kc, PU16), _p(Vc, PU16)
            else:
                L.qkv, L.gate, L.ssm_out = mats[p + "attn_qkv.weight"], mats[p + "attn_gate.weight"], mats[p + "ssm_out.weight"]
                L.beta_w, L.alpha_w = f(p + "ssm_beta.weight"), f(p + "ssm_alpha.weight")
                L.dt_b, L.ssm_a, L.conv_w, L.ssm_norm = (f(p + "ssm_dt.bias"), f(p + "ssm_a"), f(p + "ssm_conv1d.weight"),
                                                         f(p + "ssm_norm.weight"))
                Mst, cv = z(NV * HK * HK), z(3 * CC)
                self.M.append(Mst)
                self.conv.append(cv)
                L.M, L.conv = _p(Mst), _p(cv)
        self.Ls = Ls
        m = self.m = Model()
        for k, v in (("D", D), ("NL", NL), ("NFF", NFF), ("NH", NH), ("NHKV", NHKV), ("HD", HD), ("NK", NK),
                     ("HK", HK), ("NV", NV), ("CONV_CH", CC), ("HB", self.HB), ("rope_half", d["rope_dim"] // 2),
                     ("ctx", self.ctx), ("vocab", VOCAB)):
            setattr(m, k, v)
        m.L = ctypes.cast(Ls, ctypes.POINTER(Layer))
        m.emb, m.head = mats["token_embd.weight"], mats[self.head_name]
        m.out_norm = f("output_norm.weight")
        m.sD, m.sV, m.sA, m.sF = (_p(sg(D)), _p(sg(NV * HK)), _p(sg(NH * HD)), _p(sg(NFF)))
        rh = d["rope_dim"] // 2
        inv = np.float64(d["rope_base"]) ** (-2.0 * np.arange(rh) / d["rope_dim"])
        th = np.arange(self.ctx, dtype=np.float64)[:, None] * inv[None, :]
        m.cos_t, m.sin_t = _p(keep(np.cos(th).astype(F32))), _p(keep(np.sin(th).astype(F32)))
        W = max(D, NFF, NV * HK, NH * HD)
        sizes = {"h": D, "xn": D, "xp": W, "y": D, "qkv": CC, "zv": NV * HK, "cs": CC, "qk": 2 * NK * HK,
                 "gout": NV * HK, "outn": NV * HK, "m": W, "qn": NH * 2 * HD, "kr": NHKV * HD, "vv": NHKV * HD,
                 "qr": NH * HD, "kro": NHKV * HD, "sc": NH * self.ctx, "att": NH * HD, "ffg": NFF, "ffu": NFF,
                 "T": W // 2 * 8, "logits": max(VOCAB, mats[self.head_name].R), "gb": NV, "gg": NV}
        for k in _SCR:
            a = z(sizes[k])
            setattr(m, k, _p(a))
            if k == "logits":
                self.logits = a
        m.red = _p(z(256, np.float64), ctypes.POINTER(ctypes.c_double))
        self.ylog, self.ylog_n, self.ylog_map = self.logits, VOCAB, None
        self.hsl, self.shortlist = None, 0
        self.shortlist = 0
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        (log or (lambda *a: None))("kindi-cpu ready: %s, %s threads, ctx %d (%.1fs)" % (
            man.get("model"), os.environ.get("OMP_NUM_THREADS", os.cpu_count()), self.ctx, time.time() - t0))

    def reset(self):
        for a in self.M + self.conv + self.Kc + self.Vc:
            a.fill(0)

    def feed(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 0)

    def step(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        return self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 1)

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
