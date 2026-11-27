"""The Flash-Next CPU path (0.8 item C1): kindi/fncpu.c built on first use with the thread pool of moe_cpu.c.

Weights are repacked at load into Kindi's CPU forms (fncpu.c: K4 4-bit table codes, K8 int8 codes), each pool thread
writing and later computing its own slice of every matrix (NUMA-local by first touch)."""
import ctypes
import hashlib
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("moe_cpu.c", "fnmodel.c")          # fnmodel.c includes fncpu.c
_lib = None


def lib():
    global _lib
    if _lib is None:
        srcs = [open(os.path.join(HERE, f), encoding="utf-8").read() for f in SOURCES]
        if PROMPT_CHUNK > 256:
            srcs[1] = f"#define FM_TMAX {PROMPT_CHUNK}\n" + srcs[1]
        grids = open(os.path.join(HERE, "nbq_grids.h"), encoding="utf-8").read()
        cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
        os.makedirs(cache, exist_ok=True)
        inc = open(os.path.join(HERE, "fncpu.c"), encoding="utf-8").read()
        tag = hashlib.sha1(("fc1" + grids + inc + "".join(srcs)).encode()).hexdigest()[:12]
        so = os.path.join(cache, "fncpu_%s.so" % tag)
        if not os.path.exists(so):
            objs = []
            for name, src in zip(SOURCES, srcs):
                c = os.path.join(cache, "%s_%s.c" % (name[:-2], tag))
                with open(c, "w", encoding="utf-8", newline="") as f:
                    f.write(src)
                subprocess.check_call(["gcc", "-O3", "-march=native", "-pthread", "-fPIC", "-I", HERE, "-c",
                                       "-o", c + ".o", c])
                objs.append(c + ".o")
            subprocess.check_call(["gcc", "-shared", "-pthread", "-o", so + ".tmp"] + objs + ["-lm", "-ldl"])
            os.replace(so + ".tmp", so)
            for o in objs:
                os.remove(o)
                os.remove(o[:-2])
        L = ctypes.CDLL(so)
        V, I = ctypes.c_void_p, ctypes.c_int
        L.moe_pool_init.argtypes = [V, I, I, ctypes.c_float]
        L.moe_pool_init.restype = I
        L.moe_pool_set_spin.argtypes = [ctypes.c_long]
        L.fc_repack.argtypes = [I, V, I, I, I]
        L.fc_repack.restype = V
        L.fc_bytes.argtypes = [V]
        L.fc_bytes.restype = ctypes.c_size_t
        L.fc_kind.argtypes = [V]
        L.fc_exact.argtypes = [I, V, I, I]
        L.fc_exact.restype = V
        L.fc_matvec.argtypes = [V, I, I, V, V]
        L.fc_matvec.restype = I
        L.fc_matvec_st.argtypes = [V, I, I, V, V]
        L.fc_matvec_st.restype = I
        L.fc_bench.argtypes = [I, V, V, I]
        L.fc_bench.restype = ctypes.c_double
        L.fc_gemm_st.argtypes = [V, I, I, V, V]
        L.fc_gemm_st.restype = I
        L.fc_prefill_bench.argtypes = [V, I, I, I, V, V]
        L.fc_prefill_bench.restype = ctypes.c_double
        L.fc_expert_bench.argtypes = [V, V, V, I, I, V, I, I, V, V]
        L.fc_expert_bench.restype = ctypes.c_double
        _lib = L
    return _lib


_pool = None


def pool(cpus=None):
    """start the pool once: one thread per physical core; on two NUMA nodes, node 0's cores first"""
    global _pool
    if _pool is None:
        from .cores import physical_cpus, default_cpu_list, parse_list
        if cpus is None:
            c0, c1 = physical_cpus(0), physical_cpus(1)
            if not c0:
                c0, c1 = parse_list(default_cpu_list()), []
        else:
            c0, c1 = cpus, []
        order = list(c0) + list(c1)
        L = lib()
        if L.moe_pool_init((ctypes.c_int * len(order))(*order), len(c0), len(c1),
                           len(c0) / float(len(order))) != 0:
            raise RuntimeError("fncpu: pool init failed")
        _pool = (len(c0), len(c1))
    return _pool


class Tensor(object):
    """a repacked matrix (or the experts of a layer): ty, R rows, C columns, E experts, from the file's bytes"""

    def __init__(self, ty, raw, R, C, E=1):
        L = lib()
        self.ty, self.R, self.C, self.E = ty, R, C, E
        if isinstance(raw, np.ndarray):
            self._raw = raw                        # (kept while the repack reads it)
            ptr = raw.ctypes.data
        else:
            self._raw = np.frombuffer(bytes(raw), np.uint8)
            ptr = self._raw.ctypes.data
        self.h = L.fc_repack(ty, ptr, R, C, E)
        self._raw = None
        if not self.h:
            raise ValueError("fncpu: cannot repack type %d (%d x %d)" % (ty, R, C))
        self.bytes = L.fc_bytes(self.h)

    @classmethod
    def exact(cls, ty, raw, R, C):
        """a float matrix kept in fp32 (fncpu.c fc_exact)"""
        self = cls.__new__(cls)
        self.ty, self.R, self.C, self.E = ty, R, C, 1
        self._raw = np.ascontiguousarray(raw)
        self.h = lib().fc_exact(ty, self._raw.ctypes.data, R, C)
        self._raw = None
        if not self.h:
            raise ValueError("fncpu: no exact form for type %d" % ty)
        self.bytes = lib().fc_bytes(self.h)
        return self

    def gemm(self, x, e=0):
        """y = W_e x through the fp32 prompt (GEMM) path, any number of rows of x, on the calling thread"""
        x = np.ascontiguousarray(np.atleast_2d(x), np.float32)
        y = np.empty((x.shape[0], self.R), np.float32)
        if lib().fc_gemm_st(self.h, e, x.shape[0], x.ctypes.data, y.ctypes.data) != 0:
            raise ValueError("fncpu: GEMM product failed")
        return y

    def matvec(self, x, e=0, single=False):
        x = np.ascontiguousarray(np.atleast_2d(x), np.float32)
        y = np.empty((x.shape[0], self.R), np.float32)
        f = lib().fc_matvec_st if single else lib().fc_matvec
        if f(self.h, e, x.shape[0], x.ctypes.data, y.ctypes.data) != 0:
            raise ValueError("fncpu: product failed")
        return y


# ---------------------------------------------------------------------------------------------- the whole model
S_SLOTS = ["ha_down", "ha_up", "ha_inj", "hf_down", "hf_up", "hf_inj", "qkv", "z", "beta", "alpha", "sout", "q", "k",
           "v", "iq", "ik", "o", "router", "sg", "su", "sd", "eg", "eu", "ed", "pk", "pv", "out_down", "out_up", "head"]
V_SLOTS = ["ha_norm", "hf_norm", "conv", "ssm_a", "dt", "ssm_norm", "qn", "kn", "iqn", "ikn", "shg", "pnk", "pnq",
           "pnc", "pconv", "out_norm"]
T_NAMES = {"hc_attn_down": "ha_down", "hc_attn_up": "ha_up", "hc_attn_inject": "ha_inj", "hc_ffn_down": "hf_down",
           "hc_ffn_up": "hf_up", "hc_ffn_inject": "hf_inj", "attn_qkv": "qkv", "attn_gate": "z", "ssm_beta": "beta",
           "ssm_alpha": "alpha", "ssm_out": "sout", "attn_q": "q", "attn_k": "k", "attn_v": "v",
           "indexer.q_proj": "iq", "indexer.k_proj": "ik", "attn_output": "o", "ffn_gate_inp": "router",
           "ffn_gate_shexp": "sg", "ffn_up_shexp": "su", "ffn_down_shexp": "sd", "ffn_gate_exps": "eg",
           "ffn_up_exps": "eu", "ffn_down_exps": "ed", "ple_key": "pk", "ple_value": "pv"}
V_NAMES = {"hc_attn_norm": "ha_norm", "hc_ffn_norm": "hf_norm", "ssm_conv1d": "conv", "ssm_a": "ssm_a",
           "ssm_dt.bias": "dt", "ssm_norm": "ssm_norm", "attn_q_norm": "qn", "attn_k_norm": "kn",
           "indexer.q_norm": "iqn", "indexer.k_norm": "ikn", "ffn_gate_inp_shexp": "shg", "ple_norm_key": "pnk",
           "ple_norm_query": "pnq", "ple_norm_conv": "pnc", "ple_conv1d": "pconv"}
TMAX = 8                       # tokens a call when every row's logits are wanted
# KINDI_FNCPU_EXACT_ROUTER=1: the router with its bf16 weights and fp32 inputs (default off)
EXACT_ROUTER = os.environ.get("KINDI_FNCPU_EXACT_ROUTER", "0") == "1"      # (1: on; measured no gate gain)
PROMPT_CHUNK = int(os.environ.get("KINDI_FNCPU_CHUNK", "1024"))   # tokens a call otherwise (fnmodel.c FM_TMAX)
if not 1 <= PROMPT_CHUNK <= 4096:
    raise ValueError("KINDI_FNCPU_CHUNK must be between 1 and 4096")


def _setup_model_api(L):
    V, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    L.fm_new.argtypes = [I, I, I, F, F]
    L.fm_new.restype = V
    L.fm_set_t.argtypes = [V, I, I, V]
    L.fm_set_v.argtypes = [V, I, I, V]
    L.fm_set_embd.argtypes = [V, V, I, ctypes.c_long, V, I, ctypes.c_long, V, V, V, I]
    L.fm_reset.argtypes = [V]
    L.fm_forward.argtypes = [V, I, V, I, V]
    L.fm_forward.restype = I
    L.fm_pos.argtypes = [V]
    L.fm_pos.restype = I
    L.fm_prof.argtypes = [V, V]
    L.fm_set_gemm.argtypes = [I, I, I, I, I]
    L.fm_set_tb.argtypes = [I]
    L.fm_set_mtp.argtypes = [V, V, V, V, V, V, V, V, V, V, I]
    L.fm_streams.argtypes = [V, I, I, V]
    L.fm_streams.restype = I
    L.fm_mtp.argtypes = [V, I, V, V, I, V]
    L.fm_mtp.restype = I
    L.fm_ckpt.argtypes = [V, I]
    L.fm_ckpt.restype = I
    L.fm_rollback.argtypes = [V, I, I]
    L.fm_rollback.restype = I
    L.fm_barrier_bench.argtypes = [I, I]
    L.fm_barrier_bench.restype = ctypes.c_double
    L.fm_barrier_stats.argtypes = [V]
    L.fm_barrier_sites.argtypes = [V]


def _mtp_path(path):
    """the MTP head file beside the model (qwen4exp-mtp-q2.gguf, else qwen4exp-mtp.gguf), or KINDI_FNCPU_MTP (a
    path, or 0: none)"""
    env = os.environ.get("KINDI_FNCPU_MTP", "")
    if env == "0":
        return None
    if env and os.path.exists(env):
        return env
    d = os.path.dirname(os.path.abspath(path))
    for name in ("qwen4exp-mtp-q2.gguf", "qwen4exp-mtp.gguf"):
        if os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def _warm_ple(path, offset, size, log):
    """Read the mapped table sequentially into the filesystem cache using a bounded buffer."""
    from . import hostram
    available = hostram.available()
    if available is not None and available < size + hostram.MARGIN:
        log("Flash-Next CPU: %.1f GB of RAM available for %.1f GB of PLE; using on-demand reads" %
            (available / 1e9, size / 1e9))
        return
    log("Flash-Next CPU: warming %.1f GB of PLE into the filesystem cache" % (size / 1e9))
    with open(path, "rb", buffering=0) as f:
        f.seek(offset)
        buf = memoryview(bytearray(min(size, 8 << 20)))
        while size:
            got = f.readinto(buf[:size])
            if not got:
                raise RuntimeError("fncpu: truncated PLE table in %s" % path)
            size -= got
    log("Flash-Next CPU: PLE table warm")


class Model(object):
    """Flash-Next on the CPU alone: every weight repacked (fncpu.c), the forward pass in C (fnmodel.c).
    logits(ids, last_n, state) has the interface of kindi.qwen4exp_gpu.Model.logits for the gate; a state is the
    engine's own (one sequence at a time: state=None starts a new one)."""

    def __init__(self, path, ctx=8192, cpus=None, log=print):
        import time
        from .gguf import GGUF
        from . import qwen4exp_quant as Q
        from .qwen4exp_ref import shards
        t0 = time.time()
        pool(cpus)
        L = lib()
        _setup_model_api(L)
        self.L = L
        t, kv, self._keep = {}, {}, []
        for p in shards(path):
            g = GGUF(p)
            mm = np.memmap(p, np.uint8, mode="r")
            self._keep.append(mm)
            kv.update(g.kv)
            for name, ne, ty, off in g.tensors:
                t[name] = (ty, list(ne), mm, g.data_start + off)
        a = "qwen4exp"
        self.nl = kv[a + ".block_count"]
        self.V = t["output.weight"][1][1]
        self.m = L.fm_new(self.nl, int(ctx), self.V, float(kv[a + ".attention.layer_norm_rms_epsilon"]),
                          float(kv[a + ".rope.freq_base"]))
        self.ctx = int(ctx)
        nbytes = 0

        def raw(name):
            ty, ne, mm, off = t[name]
            n = Q.nbytes(ty, int(np.prod(ne)))
            return mm[off:off + n]

        def vec(name):
            ty, ne, mm, off = t[name]
            v = np.ascontiguousarray(Q.decode(ty, raw(name).tobytes(), int(np.prod(ne))).astype(np.float32))
            self._keep.append(v)
            return v.ctypes.data

        for name in sorted(t):
            ty, ne, mm, off = t[name]
            mt = None
            if name.startswith("blk."):
                parts = name.split(".")
                l = int(parts[1])
                key = ".".join(parts[2:]).replace(".weight", "")
            else:
                l, key = -1, name.replace(".weight", "")
            if key in T_NAMES and len(ne) >= 2:
                C, R, E = ne[0], ne[1], (ne[2] if len(ne) > 2 else 1)
                if key == "ffn_gate_inp" and EXACT_ROUTER:
                    T = Tensor.exact(ty, raw(name), R, C)
                else:
                    T = Tensor(ty, raw(name), R, C, E)
                nbytes += T.bytes
                self._keep.append(T)
                L.fm_set_t(self.m, l, S_SLOTS.index(T_NAMES[key]), T.h)
            elif key in V_NAMES or (key == "ffn_gate_inp_shexp"):
                L.fm_set_v(self.m, l, V_SLOTS.index(V_NAMES[key]), vec(name))
        for key, slot in (("output_hc_down", "out_down"), ("output_hc_up", "out_up"), ("output", "head")):
            ty, ne, mm, off = t[key + ".weight"]
            T = Tensor(ty, raw(key + ".weight"), ne[1], ne[0], 1)
            nbytes += T.bytes
            self._keep.append(T)
            L.fm_set_t(self.m, -1, S_SLOTS.index(slot), T.h)
        L.fm_set_v(self.m, -1, V_SLOTS.index("out_norm"), vec("output_hc_norm.weight"))
        ety, ene, emm, eoff = t["token_embd.weight"]
        pty, pne, pmm, poff = t["per_layer_token_embd.weight"]
        # the per-layer embedding table (28.8 GB) is read at random by the first prompts; warm it into the page cache
        # (when RAM allows): "bg" (default) in a background thread while the engine serves, "1" before serving, "0" off
        warm = os.environ.get("KINDI_FNCPU_WARM_PLE", "bg")
        if warm == "1":
            _warm_ple(pmm.filename, poff, Q.nbytes(pty, int(np.prod(pne))), log)
        elif warm == "bg":
            import threading
            threading.Thread(target=_warm_ple, args=(pmm.filename, poff, Q.nbytes(pty, int(np.prod(pne))), log),
                             name="fncpu-warm-ple", daemon=True).start()
        i64 = lambda k: np.ascontiguousarray(np.array(kv[a + ".ple." + k], dtype=np.int64))   # noqa: E731
        self._ple = [i64("layer_multipliers"), i64("head_offsets"), i64("head_vocab_sizes")]
        L.fm_set_embd(self.m, emm.ctypes.data + eoff, ety, Q.nbytes(ety, ene[0]), pmm.ctypes.data + poff, pty,
                      Q.nbytes(pty, pne[0]), self._ple[0].ctypes.data, self._ple[1].ctypes.data,
                      self._ple[2].ctypes.data, int(kv[a + ".ple.eos_token_id"]))
        self.has_mtp = False
        mtp = _mtp_path(path)
        if mtp:
            nbytes += self._load_mtp(mtp, t, Q, raw, log)
        L.fm_reset(self.m)
        self.bytes = nbytes
        log("Flash-Next CPU: %d layers repacked, %.1f GB, %d threads (%.0f s)" % (
            self.nl, nbytes / 1e9, sum(_pool), time.time() - t0))

    def _load_mtp(self, mtp, t, Q, raw_main, log):
        """the MTP head (its decoder layer as layer L, its own projections) and the draft head: the rows of
        output.weight for the draft vocabulary (kindi/qwen4exp-draft-vocab.json, as the card engine)"""
        import json
        from .gguf import GGUF
        L, nl = self.L, self.nl
        g = GGUF(mtp)
        mm = np.memmap(mtp, np.uint8, mode="r")
        self._keep.append(mm)
        ht = {name: (ty, list(ne), mm, g.data_start + off) for name, ne, ty, off in g.tensors}

        def raw(name):
            ty, ne, mm_, off = ht[name]
            return mm_[off:off + Q.nbytes(ty, int(np.prod(ne)))]

        def vec(name):
            ty, ne, mm_, off = ht[name]
            v = np.ascontiguousarray(Q.decode(ty, raw(name).tobytes(), int(np.prod(ne))).astype(np.float32))
            self._keep.append(v)
            return v.ctypes.data

        def tens(name):
            ty, ne, mm_, off = ht[name]
            T = Tensor(ty, raw(name), ne[1], ne[0], ne[2] if len(ne) > 2 else 1)
            self._keep.append(T)
            return T

        nbytes = 0
        pre = "blk.%d." % nl
        for name in sorted(ht):
            if not name.startswith(pre):
                continue
            key = name[len(pre):].replace(".weight", "")
            ty, ne, mm_, off = ht[name]
            if key in T_NAMES and len(ne) >= 2:
                T = tens(name)
                nbytes += T.bytes
                L.fm_set_t(self.m, nl, S_SLOTS.index(T_NAMES[key]), T.h)
            elif key in V_NAMES:
                L.fm_set_v(self.m, nl, V_SLOTS.index(V_NAMES[key]), vec(name))
        parts = [tens("mtp.fc_emb.weight"), tens("mtp.fc_hid.weight"), tens("mtp.output_hc_down.weight"),
                 tens("mtp.output_hc_up.weight")]
        vpath = os.environ.get("KINDI_MTP_VOCAB") or os.path.join(HERE, "qwen4exp-draft-vocab.json")
        ids = np.ascontiguousarray(np.array(json.load(open(vpath))["ids"], dtype=np.int32))
        ty, ne, mm_, off = t["output.weight"]
        rb = Q.nbytes(ty, ne[0])
        rows = np.asarray(mm_[off:off + rb * ne[1]]).reshape(ne[1], rb)[ids]
        draft = Tensor(ty, np.ascontiguousarray(rows).reshape(-1), len(ids), ne[0], 1)
        self._keep.extend([draft, ids])
        nbytes += sum(T.bytes for T in parts) + draft.bytes
        L.fm_set_mtp(self.m, parts[0].h, parts[1].h, parts[2].h, parts[3].h, draft.h, vec("mtp.norm_emb.weight"),
                     vec("mtp.norm_hid.weight"), vec("mtp.output_hc_norm.weight"), ids.ctypes.data, len(ids))
        self.has_mtp = True
        self.H = int(4 * 2560)
        log("Flash-Next CPU: MTP head %s, draft vocabulary %d tokens" % (os.path.basename(mtp), len(ids)))
        return nbytes

    # ---- drafting with the MTP head
    def streams(self, r0, nr):
        out = np.empty((nr, self.H), np.float32)
        if self.L.fm_streams(self.m, r0, nr, out.ctypes.data) != 0:
            raise RuntimeError("fncpu: no such stream rows")
        return out

    def mtp(self, hc, ids, pos):
        """the head on the rows hc (main streams at positions pos ..; None: its own last streams, one row) with the
        tokens ids at p + 1; returns the draft for the last row"""
        ids = np.ascontiguousarray(np.array(ids, dtype=np.int32))
        ptr = None
        if hc is not None:
            hc = np.ascontiguousarray(hc, np.float32)
            ptr = hc.ctypes.data
        d = self.L.fm_mtp(self.m, len(ids), ptr, ids.ctypes.data, int(pos), None)
        if d < 0:
            raise RuntimeError("fncpu: MTP head call failed")
        return d

    def prefill_mtp(self, ids):
        """read ids from a reset state, writing the head's rows for them as it goes; returns the last logits and
        the pending head input (the last position's streams)"""
        self.reset()
        ids = list(ids)
        pending, lg = None, None
        for i in range(0, len(ids), PROMPT_CHUNK):
            c = ids[i:i + PROMPT_CHUNK]
            if pending is not None:
                self.mtp(pending, [c[0]], i - 1)
            last = i + len(c) == len(ids)
            arr = np.ascontiguousarray(np.array(c, dtype=np.int32))
            lg = np.empty((1, self.V), np.float32) if last else None
            if self.L.fm_forward(self.m, len(c), arr.ctypes.data, 0 if last else -1,
                                 lg.ctypes.data if last else None) != 0:
                raise RuntimeError("fncpu: forward failed (context %d full?)" % self.ctx)
            H = self.streams(0, len(c))
            if len(c) > 1:
                self.mtp(H[:-1], c[1:], i)
            pending = H[-1:].copy()
        return lg[-1], pending

    def drafted(self, ids, n, k=2, sampler=None, stop=(), cancel=None, stats=None):
        """the reply of ids with MTP drafts, as a generator of accepted tokens: greedy (each token the main model's
        argmax) or sampled by speculative sampling (sampler: kindi.qwen4exp_gen.DraftSampler)"""
        import torch
        t0 = time.time()
        lg, Hrows = self.prefill_mtp(ids)
        hist = list(ids)
        first = int(lg.argmax()) if sampler is None else int(sampler.first(torch.from_numpy(lg), hist))
        if stats is not None:
            stats.update({"passes": 0, "drafted": 0, "accepted": 0, "prompt_s": time.time() - t0})
        yield first
        out = [first]
        if first in stop or n <= 1:
            return
        self.L.fm_ckpt(self.m, 1)
        try:
            q0, Hids, t, P = len(ids) - 1, [first], first, len(ids)
            while len(out) < n:
                if cancel is not None and cancel():
                    return
                drafts = [self.mtp(Hrows, Hids, q0)]
                for j in range(1, k):
                    drafts.append(self.mtp(None, [drafts[-1]], P + j - 1))
                win = np.ascontiguousarray(np.array([t] + drafts, dtype=np.int32))
                lgw = np.empty((len(win), self.V), np.float32)
                if self.L.fm_forward(self.m, len(win), win.ctypes.data, 1, lgw.ctypes.data) != 0:
                    raise RuntimeError("fncpu: verify failed (context %d full?)" % self.ctx)
                H = self.streams(0, len(win))
                if sampler is None:
                    a = 0
                    while a < k and int(lgw[a].argmax()) == drafts[a]:
                        a += 1
                    new = drafts[:a] + [int(lgw[a].argmax())]
                else:
                    new = sampler.accept(torch.from_numpy(lgw), drafts, hist, out)
                    a = len(new) - 1
                if self.L.fm_rollback(self.m, len(win), 1 + a) != 0:
                    raise RuntimeError("fncpu: rollback failed")
                if stats is not None:
                    stats["passes"] += 1
                    stats["drafted"] += k
                    stats["accepted"] += a
                for tok in new:
                    out.append(tok)
                    yield tok
                    if tok in stop or len(out) >= n:
                        return
                Hrows, Hids, q0, t, P = H[:a + 1].copy(), list(new), P, new[-1], P + a + 1
        finally:
            self.L.fm_ckpt(self.m, 0)

    def reset(self):
        self.L.fm_reset(self.m)

    PHASES = ("mixers", "gdn", "attention", "router+shared", "experts", "embed+head", "ple", "gdn_recur")

    def prof(self):
        """seconds per phase since the last call"""
        a = np.zeros(8, np.float64)
        self.L.fm_prof(self.m, a.ctypes.data)
        return dict(zip(self.PHASES, a.tolist()))

    def barrier_stats(self):
        """since the last call: mean and max over threads of the TSC cycles waiting in barriers, barriers a thread"""
        a = np.zeros(3, np.float64)
        self.L.fm_barrier_stats(a.ctypes.data)
        return a

    @property
    def pos(self):
        return self.L.fm_pos(self.m)

    def forward(self, ids, all_rows=False):
        """Read ids and return the logits of the last token, or of every token."""
        return self._forward(ids, all_rows=all_rows, state_only=False)

    def _forward(self, ids, *, all_rows, state_only):
        ids = list(ids)
        outs = []
        step = TMAX if all_rows else PROMPT_CHUNK
        for i in range(0, len(ids), step):
            c = np.ascontiguousarray(np.array(ids[i:i + step], dtype=np.int32))
            n = len(c)
            emit = not state_only and (all_rows or i + n == len(ids))
            lg = np.empty((n if all_rows else 1, self.V), np.float32) if emit else None
            output = int(all_rows) if emit else -1
            if self.L.fm_forward(self.m, n, c.ctypes.data, output, lg.ctypes.data if emit else None) != 0:
                raise RuntimeError("fncpu: forward failed (context %d full?)" % self.ctx)
            if emit:
                outs.append(lg)
        if state_only:
            return None
        return np.concatenate(outs) if all_rows else outs[-1]

    def logits(self, ids, last_n=None, state=None):
        """the gate's interface: a new sequence each call unless state is given (a dict this class fills)"""
        import torch
        if state is None or not state.get("live"):
            self.reset()
            if state is not None:
                state["live"] = True
        ids = list(ids)
        if last_n:
            head = ids[:-last_n] if last_n < len(ids) else []
            if head:
                self._forward(head, all_rows=False, state_only=True)
            out = self.forward(ids[len(head):], all_rows=True)
        else:
            out = self.forward(ids, all_rows=True)
        return torch.from_numpy(out)
