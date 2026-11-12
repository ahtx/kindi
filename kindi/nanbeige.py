"""Kindi engine for llama-architecture GGUF files with K-quant and I-quant weights (Nanbeige 4.3 33B IQ3_XXS).

The layers are split in order over one or more CUDA cards. Each card replays one CUDA graph per token; between the
cards only the hidden state (fp32, 28 KB) moves. The kernels are in nbq.cu."""
import mmap
import os
import time
import numpy as np
from .gguf import GGUF

BLOCK = {12: 144, 13: 176, 18: 98, 21: 110, 22: 82}       # bytes per block of 256 weights in the file
BLOCK_BODY = {12: 144, 13: 176, 18: 96, 21: 108, 22: 80}  # the same on the card (I-quants: without the fp16 scale)
IQ = (18, 21, 22)                                           # repacked: fp16 scale array + aligned body
MV = {12: "nbq_mv_q4k", 13: "nbq_mv_q5k", 18: "nbq_mv_iq3xxs", 21: "nbq_mv_iq3s", 22: "nbq_mv_iq2s"}
NULL = np.uint64(0)
FUSED_QKV = os.environ.get("KINDI_NB_FUSED_QKV", "1") == "1"
NSPLIT = 16                                                 # decode attention: position splits per K/V head
KERNELS = ("nbq_mv_q4k", "nbq_mv_q5k", "nbq_mv_iq3xxs", "nbq_mv_iq3s", "nbq_mv_iq2s", "nbq_mv2_iq3xxs", "nbq_repack",
           "nbq_add_rms_q8", "nbq_add_rms", "nbq_q8", "nbq_swiglu_q8", "nbq_embed_t21", "nbq_embed_t12", "nbq_embed_t13", "nbq_rope_store",
           "nbq_attn_part", "nbq_attn_merge", "nbq_add", "nbq_argmax_part", "nbq_argmax_final",
           "nbq_mm_q4k", "nbq_mm_q5k", "nbq_mm_iq3xxs", "nbq_mm_iq3s", "nbq_mm_iq2s", "nbq_mm_iq2s_q4k", "nbq_attn_part_g5", "nbq_fa_prefill_g5",
           "nbq_add_rms_q8_rows", "nbq_swiglu_q8_rows", "nbq_q8_rows", "nbq_rope_store_rows", "nbq_fa_prefill", "nbq_qkv_rope", "nbq_attn_merge_q8", "nbq_attn_merge_q8_rows", "nbq_argmax_rows",
           "nbq_rope_store_q8", "nbq_attn_q8", "nbq_attn_q8_g5", "nbq_fa_prefill_q8", "nbq_fa_prefill_q8_g5") + tuple(
    "nbq_mvn_%s_%d" % (t, n) for t in ("q4k", "q5k", "iq3xxs", "iq3s", "iq2s") for n in range(1, 9))
TCV = {12: "nbq_tcv_q4k", 13: "nbq_tcv_q5k", 18: "nbq_tcv_iq3xxs", 21: "nbq_tcv_iq3s", 22: "nbq_tcv_iq2s"}
KERNELS = KERNELS + tuple(TCV.values()) + ("nbq_tcv_sum",)
KERNELS = KERNELS + tuple("nbq_%s_%d" % (k, n) for k in ("mv2n_iq3xxs", "qkvn_rope") for n in (2, 3, 4))
MVC = {12: "nbq_mvc_q4k", 13: "nbq_mvc_q5k", 18: "nbq_mvc_iq3xxs", 21: "nbq_mvc_iq3s", 22: "nbq_mvc_iq2s"}   # Kepler
KDEQ = {12: "nbq_kdeq_q4k", 13: "nbq_kdeq_q5k", 18: "nbq_kdeq_iq3xxs", 21: "nbq_kdeq_iq3s", 22: "nbq_kdeq_iq2s"}
KBLAS = os.environ.get("KINDI_NB_KBLAS", "1") == "1"
KDQ_BYTES = int(os.environ.get("KINDI_NB_KDQ_MB", "128")) << 20                                        # Kepler prompt: fp32 weight rows per cublasSgemm
KMM = {12: "nbq_kmm_q4k", 13: "nbq_kmm_q5k", 18: "nbq_kmm_iq3xxs", 21: "nbq_kmm_iq3s", 22: "nbq_kmm_iq2s"}
K35_KERNELS = tuple(MVC.values()) + tuple(KMM.values()) + tuple(KDEQ.values()) + ("nbq_mvc_sum", "nbq_kreorder", "nbq_kx2f", "nbq_kcopy")
CPU_F = float(os.environ.get("KINDI_NB_CPUF", "0.55"))        # Kepler host layers: FFN rows on the CPU in a step
HOST_SLOTS = int(os.environ.get("KINDI_NB_SLOTS", "2"))     # device buffers for the host layers (whole layers)
KRB = 8                                                      # nbq_k35.cu NBQ_KRB: rows per batch of nbq_mvc
MVN = {12: "nbq_mvn_q4k", 13: "nbq_mvn_q5k", 18: "nbq_mvn_iq3xxs", 21: "nbq_mvn_iq3s", 22: "nbq_mvn_iq2s"}
FA_SMEM = 2 * 64 * 136 * 2                                   # nbq_fa.cu: K and V tiles
MM = {12: "nbq_mm_q4k", 13: "nbq_mm_q5k", 18: "nbq_mm_iq3xxs", 21: "nbq_mm_iq3s", 22: "nbq_mm_iq2s", (22, 12): "nbq_mm_iq2s_q4k"}
GEMM_SMEM = 2 * 128 * 144 + 128 * 8 * 4 + 3 * 128 * 4 * 4    # NBQ_GEMM_SMEM in nbq_mm.cu
FA_KERNELS = ("nbq_fa_prefill", "nbq_fa_prefill_g5", "nbq_fa_prefill_q8", "nbq_fa_prefill_q8_g5")
TC_KERNELS = set(MM.values()) | set(TCV.values()) | set(FA_KERNELS)
KV_TYPES = ("f16", "q8_0")                                  # the K/V cache (config kv_cache)
KV_DEFAULT = "auto"                                         # f16 when it fits at ctx, else q8_0
CHUNK = 512                                                 # prompt tokens per pass
ATT_SUB = 32                                                # prompt tokens per attention launch


def source():
    here = os.path.dirname(os.path.abspath(__file__))
    return "".join(open(os.path.join(here, f), encoding="utf-8").read() for f in ("nbq_grids.h", "nbq.cu", "nbq_mm.cu", "nbq_fa.cu", "nbq_q8kv.cu", "nbq_mvn.cu", "nbq_tcv.cu", "nbq_k35.cu"))


def module(cp, src, tc=True, extra=()):
    """compile the kernels for the current card. NVRTC 10.2 (the release image) does not know sm_80 and later, so
    there the kernels go through compute_75 PTX that is marked as PTX 7.1 for this card (as kindi/tc.py does): the
    tensor-core instructions in nbq_mm/nbq_fa/nbq_tcv then load on sm_80+"""
    fast = () if os.environ.get("KINDI_NB_FASTMATH", "1") == "0" else ("--use_fast_math",)
    opts = fast + ("-std=c++14",) + (() if tc else ("-DNBQ_TC=0",)) + tuple(extra)
    if cp.cuda.nvrtc.getVersion()[0] < 11 and tc:            # (cards before sm_80 compile for themselves)
        from .tc import _patched_module
        p = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        inc = ("-I" + os.path.join(cp.cuda.get_cuda_path() or "/usr/local/cuda", "include"),)   # cuda_fp16.h
        return _patched_module(src, "sm_%d%d" % (p["major"], p["minor"]), opts + inc)
    return cp.RawModule(code=src, options=opts)


class _RtGraph(object):
    """a CUDA graph made through the CUDA runtime library directly, for CuPy before 10 (the release image has CuPy
    9.6, which has no Stream.begin_capture)"""
    rt = None

    @classmethod
    def lib(cls):
        if cls.rt is None:
            import ctypes
            import ctypes.util
            for name in ("libcudart.so", "libcudart.so.10.2", ctypes.util.find_library("cudart")):
                try:
                    cls.rt = ctypes.CDLL(name)
                    break
                except (OSError, TypeError):
                    continue
        return cls.rt

    def __init__(self, s):
        import ctypes
        g, e = ctypes.c_void_p(), ctypes.c_void_p()
        rt = self.lib()
        assert rt.cudaStreamEndCapture(ctypes.c_void_p(s.ptr), ctypes.byref(g)) == 0, "cudaStreamEndCapture failed"
        assert rt.cudaGraphInstantiate(ctypes.byref(e), g, None, None, ctypes.c_size_t(0)) == 0,             "cudaGraphInstantiate failed"
        self.g, self.e = g, e

    def launch(self, s):
        import ctypes
        assert self.lib().cudaGraphLaunch(self.e, ctypes.c_void_p(s.ptr)) == 0, "cudaGraphLaunch failed"


def _begin_capture(s):
    if hasattr(s, "begin_capture"):
        return s.begin_capture()
    import ctypes
    assert _RtGraph.lib().cudaStreamBeginCapture(ctypes.c_void_p(s.ptr), 2) == 0, "cudaStreamBeginCapture failed"  # relaxed


def _end_capture(s):
    return s.end_capture() if hasattr(s, "end_capture") else _RtGraph(s)


def _smem(cp, f, n):
    """allow n bytes of dynamic shared memory for kernel f (a RawModule function or a Module function)"""
    try:
        f.max_dynamic_shared_size_bytes = n
    except AttributeError:
        cp.cuda.driver.funcSetAttribute(f.ptr, 8, n)   # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES


def _bus(s):
    """a PCI bus address in one form: nvidia-smi gives 00000000:01:00.0, CUDA gives 0000:01:00.0"""
    s = str(s).strip().lower()
    dom, rest = s.split(":", 1) if s.count(":") == 2 else ("0", s)
    return "%x:%s" % (int(dom, 16), rest)


def _free_bytes(cp, devices):
    """free memory per card, from nvidia-smi when it runs (under WSL, CUDA's numbers do not follow what Windows holds
    on a card), else from CUDA"""
    out = []
    used = {}
    try:
        import subprocess
        q = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.used,memory.total,display_active",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for ln in q.strip().splitlines():
            f = [x.strip() for x in ln.split(",")]
            # a card that drives a desktop (WDDM, more than 1 GB used) pages well before it is full: keep 500 MB more
            extra = 500 if f[3].lower() in ("enabled", "yes") and int(f[1]) > 1024 else 0
            used[_bus(f[0])] = ((int(f[1]) + extra) * 2 ** 20, int(f[2]) * 2 ** 20)
    except Exception:
        pass
    for d in devices:
        with cp.cuda.Device(d):
            f, t = cp.cuda.runtime.memGetInfo()
        # match the card by its PCI bus address: nvidia-smi's index order is not CUDA's device order, and
        # CUDA_VISIBLE_DEVICES renumbers the cards
        try:
            key = _bus(cp.cuda.runtime.deviceGetPCIBusId(d))
        except Exception:
            key = None
        if key in used:                    # WSL's CUDA reports about 1.1 GB less than is free; nvidia-smi is right
            f = used[key][1] - used[key][0]
        out.append(f)
    return out


def kepler(cp, dev):
    """sm_3x, sm_5x and sm_60 (P100): no __dp4a, no tensor cores; the fp32-exact kernels of nbq_k35.cu"""
    return int(cp.cuda.Device(dev).compute_capability) < 61


class _Replay(object):
    """stands in for a CUDA graph where CuPy has no stream capture (CuPy 9 in the CUDA 10.2 image that Kepler cards
    use): launch() makes the launches again"""
    def __init__(self, fn):
        self.fn = fn

    def launch(self, stream):
        with stream:
            self.fn()


def _capture(stream, fn, replay=False):
    """a CUDA graph of the launches that fn() makes on stream (CuPy 9: through the runtime library, _RtGraph);
    replay: fn() again at each launch (cards with host layers: their copies and CPU parts are queued from Python)"""
    if replay:
        return _Replay(fn)
    with stream:
        _begin_capture(stream)
        fn()
        return _end_capture(stream)


class Split(object):
    """a host-layer matrix in two parts by rows: parts[0] = rows [0, rg) and parts[1] = rows [rg, rows) (Weight
    objects with an r0). The card computes both in prompt and check passes; in a step the CPU computes parts[1] from
    its index form (wi, wb: nbq_cpu_expand_idx at load) while the card computes parts[0]."""
    def __init__(self, parts, raw, ty, rows, cols):
        self.parts, self.raw, self.ty, self.rows, self.cols = parts, raw, ty, rows, cols
        self.rg = parts[1].r0


class Weight(object):
    """one matrix on one card: body (uint8), d (fp16 block scales, I-quants only), type, rows, cols"""
    def __init__(self, body, d, ty, rows, cols):
        self.body, self.d, self.ty, self.rows, self.cols = body, d, ty, rows, cols


class Card(object):
    """the layers [lo, hi) on one device, with their buffers, caches and kernels"""
    def __init__(self, cp, dev, lo, hi, cfg):
        self.cp, self.dev, self.lo, self.hi = cp, dev, lo, hi
        self.k35 = kepler(cp, dev)
        # KINDI_NB_KFAST=0 (tests): Kepler with the row-major weights and the kernels of the other cards (dp4a
        # emulated with KINDI_NB_KINT=1); the prompt then goes token by token
        kfast = os.environ.get("KINDI_NB_KFAST", "1") == "1"
        opts = () if kfast else ("-DNBQ_KROWMAJOR",) + (("-DNBQ_K35_INT",) if os.environ.get("KINDI_NB_KINT") == "1" else ())
        self.k35 = self.k35 and kfast
        self.kslow = kepler(cp, dev) and not kfast
        self.host = set()                  # layers whose weights stay in pinned host memory (see Engine._plan)
        tc = (cfg is None or cfg.get("tc", True)) and not kepler(cp, dev)
        # the bank-free IQ3_XXS codebook of the decode products (nbq.cu NBQ_REP): KINDI_NB_REP=0 / 1, default sm_80+
        rv = os.environ.get("KINDI_NB_REP", "")
        from .tc import ga100           # (GA100 only: on an RTX 3090 it cost 14%, 54.3 -> 46.6 tok/s)
        if (rv == "1") if rv in ("0", "1") else ga100(dev):
            opts = opts + ("-DNBQ_REP=1",)
        with cp.cuda.Device(dev):
            mod = module(cp, source(), tc, opts)
            self.k = {n: mod.get_function(n) for n in KERNELS + (K35_KERNELS if self.k35 else ())
                      if tc or n not in TC_KERNELS}
            if self.k35:                       # 8-byte banks (Kepler): nbq_kmm reads float4 rows of its tiles
                import ctypes
                cu = ctypes.CDLL("libcuda.so.1")
                for n in tuple(KMM.values()) + tuple(MVC.values()):
                    cu.cuFuncSetSharedMemConfig(ctypes.c_void_p(getattr(self.k[n], "kernel", self.k[n]).ptr), 2)
            for n in MM.values() if tc else ():
                _smem(cp, self.k[n], GEMM_SMEM)
            for n in KERNELS:
                if n.startswith("nbq_mvn_") and tc:          # (the launches use none now: stage_x is off; sm_61
                    _smem(cp, self.k[n], 65536)               # cannot give a block more than 48 KB)
            for n in FA_KERNELS if tc else ():
                _smem(cp, self.k[n], FA_SMEM)
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.w = {}
        self.kc, self.vc, self.kd, self.vd = {}, {}, {}, {}    # kd, vd: the q8_0 scales
        self.graph = {}


class Engine(object):
    def __init__(self, path, ctx=4096, devices=None, split=None, keep=None, reserve_mb=None, kv=None, log=print):
        import cupy as cp
        self.kv = kv or os.environ.get("KINDI_KV_CACHE") or KV_DEFAULT
        if self.kv not in KV_TYPES + ("auto",):
            raise SystemExit("kindi: kv_cache must be auto or one of %s, not %r" % (", ".join(KV_TYPES), self.kv))
        self.q8 = self.kv == "q8_0"
        ctx = (ctx + 255) // 256 * 256 if ctx > 0 else 0  # the attention kernels read whole tiles of 256 positions
        self.cp, self.log, self.ctx = cp, log, ctx
        g = GGUF(path)
        self.g, kv = g, g.kv
        a = kv["general.architecture"]
        self.n_layer = kv[a + ".block_count"]
        self.n_embd = kv[a + ".embedding_length"]
        self.n_ff = kv[a + ".feed_forward_length"]
        self.n_head = kv[a + ".attention.head_count"]
        self.n_kv = kv[a + ".attention.head_count_kv"]
        self.eps = float(kv[a + ".attention.layer_norm_rms_epsilon"])
        self.base = float(kv.get(a + ".rope.freq_base", 10000.0))
        self.hd = self.n_embd // self.n_head
        self.vocab = len(kv["tokenizer.ggml.tokens"])
        self.G = self.n_head // self.n_kv
        assert self.hd == 128 and self.G in (5, 7), "nbq.cu is written for 128-wide heads, 5 or 7 per K/V head"
        self.keep = keep or {}                             # bytes to leave free per device (for a draft model)
        self.reserve = int(os.environ.get("KINDI_NB_RESERVE_MB", "300") if reserve_mb is None else reserve_mb) * 2 ** 20
        self.tinfo = {n: (ne, ty, off) for n, ne, ty, off in g.tensors}
        f = open(path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        ndev = cp.cuda.runtime.getDeviceCount()
        devices = list(range(ndev)) if devices is None else list(devices)
        # the embedding in pinned host memory: a step reads one row of it (5 KB over PCIe). Default on Kepler,
        # where the cards are small.
        self.embd_host = os.environ.get("KINDI_NB_EMBD_HOST", "1" if kepler(cp, devices[0]) else "0") == "1"
        if self.kv == "auto":
            # auto: the f16 cache when the whole model and its cache fit on the cards at ctx, else q8_0 (half the
            # memory) on cards with dp4a, so a long context stays on the cards instead of using host layers
            self.kv = "f16"
            dp4a = min(int(cp.cuda.Device(d).compute_capability) for d in devices) >= 61 and not kepler(cp, devices[0])
            if self.ctx > 0 and dp4a and split is None:
                ok, _ = self._fit(devices, _free_bytes(cp, devices), self.ctx, quiet=True)
                if not ok or self.n_host:
                    self.kv, self.q8 = "q8_0", True
                    log("kv_cache auto: f16 does not fit ctx %d on the cards, using q8_0" % self.ctx)
        g5 = "" if self.G == 7 else "_g%d" % self.G
        self.k_attn = ("nbq_attn_q8" if self.q8 else "nbq_attn_part") + g5
        self.k_fa = ("nbq_fa_prefill_q8" if self.q8 else "nbq_fa_prefill") + g5
        if split is None:
            split = self._plan(devices)
        bounds = [0] + list(split) + [self.n_layer]
        self.nsm = cp.cuda.Device(devices[0]).attributes["MultiProcessorCount"]
        # tensor cores (sm_80+) for the prompt; Kepler has its own prompt path (nbq_k35.cu), the other cards before
        # sm_80 read the prompt through check passes of 8 tokens
        cc = min(int(cp.cuda.Device(d).compute_capability) for d in devices)
        self.tc = cc >= 80 and os.environ.get("KINDI_NB_TC", "1") != "0"
        self.cards = [Card(cp, d, bounds[i], bounds[i + 1], {"tc": self.tc}) for i, d in enumerate(devices)]
        self.k35 = any(c.k35 for c in self.cards)
        self.kslow = any(c.kslow for c in self.cards)
        if self.q8 and (self.k35 or self.kslow or cc < 61):
            raise SystemExit("kindi: the q8_0 K/V cache needs dp4a (compute capability 6.1 or newer); use kv_cache f16")
        if not self.tc and not self.k35:
            log("compute capability %d: no tensor-core prompt kernels; the prompt is read 8 tokens per pass" % cc)
        last = self.cards[-1]
        last.host = set(range(last.hi - getattr(self, "n_host", 0), last.hi))
        t0 = time.time()
        for c in self.cards:
            self._load_card(c, first=c is self.cards[0], last=c is self.cards[-1])
        self.log("loaded %d layers on %s in %.1f s (layers per card: %s)" % (
            self.n_layer, devices, time.time() - t0, [c.hi - c.lo for c in self.cards]))
        self.pinned_tok = cp.cuda.alloc_pinned_memory(16)
        self.h_tok = np.frombuffer(self.pinned_tok, np.int32, 4)

    # ------------------------------------------------------------------ loading
    def _nbytes(self, name):
        ne, ty, _ = self.tinfo[name]
        n = int(np.prod(ne))
        return n * 4 if ty == 0 else n // 256 * BLOCK[ty]

    def _plan(self, devices):
        """split the layers by the free memory of the cards (the first card also holds the embedding, the last the
        head). Cards of the same speed get the same number of layers; with cards of different memory bandwidth, the
        faster cards are filled first (a step reads every layer once, one card after the other). With ctx 0 the
        context is the largest that fits (in steps of 256 positions, up to the model's training context)."""
        cp = self.cp
        free0 = _free_bytes(cp, devices)
        bw = []
        for d in devices:
            a = cp.cuda.Device(d).attributes
            bw.append(a["MemoryClockRate"] * a["GlobalMemoryBusWidth"])
        if self.ctx <= 0:
            top = min(int(self.g.kv.get("llama.context_length") or 32768), 262144)
            lo, hi = 0, top // 256
            while lo < hi:                                 # the largest ctx (x 256) whose plan fits
                mid = (lo + hi + 1) // 2
                if self._fit(devices, free0, mid * 256, quiet=True)[0]:
                    lo = mid
                else:
                    hi = mid - 1
            if lo == 0:
                raise MemoryError("the model does not fit on %s at any context" % devices)
            self.ctx = lo * 256
            self.log("ctx: %d positions fit on %s (the largest in steps of 256)" % (self.ctx, devices))
        ok, cap = self._fit(devices, free0, self.ctx)
        if not ok:
            raise MemoryError("the model needs %.1f GB for its layers at ctx %d, the cards have room for %d of %d layers"
                              % (self._per(self.ctx) * self.n_layer / 1e9, self.ctx, sum(cap), self.n_layer))
        if len(set(bw)) == 1:
            n = [min(c, -(-self.n_layer // len(cap))) for c in cap]      # balanced, capped by the room on each card
        else:
            n = [0] * len(cap)
            for i in sorted(range(len(cap)), key=lambda j: -bw[j]):     # the fastest cards first
                n[i] = min(cap[i], self.n_layer - sum(n))
        while sum(n) < self.n_layer:
            i = max(range(len(n)), key=lambda j: cap[j] - n[j])
            n[i] += 1
        while sum(n) > self.n_layer:
            n[max(range(len(n)), key=lambda j: n[j])] -= 1
        self.log("plan: ctx %d, layers per card %s (room for %s), memory bandwidth %s GB/s" % (
            self.ctx, n, cap, [round(b * 2 / 8e6) for b in bw]))
        return [sum(n[:i + 1]) for i in range(len(n) - 1)]

    def _per(self, ctx):
        """bytes of the largest layer with its K/V cache at ctx"""
        w = max(sum(self._nbytes("blk.%d.%s.weight" % (il, t)) for t in ("attn_q", "attn_k", "attn_v", "attn_output",
                                                                          "ffn_gate", "ffn_up", "ffn_down"))
                for il in range(self.n_layer))
        return w + self._kvl(ctx)

    def _kvl(self, ctx):
        """bytes of one layer's K/V cache at ctx (q8_0: 32 int8 and an fp16 scale per 32 values)"""
        return 2 * self.n_kv * ctx * (self.hd + self.hd // 16 if self.q8 else self.hd * 2)

    def _fit(self, devices, free0, ctx, quiet=False):
        """(fits, layers per card) at ctx; may set self.n_host (KINDI_NB_HOST=1, the last card)"""
        per = self._per(ctx)
        E, F, C = self.n_embd, self.n_ff, CHUNK
        pbuf = C * (3 * E * 4 + F + F // 8 + (E + 2 * self.n_kv * self.hd) * 4 + 2 * F * 4)   # _pbufs
        free = [f - self.reserve - pbuf - self.keep.get(d, 0) for f, d in zip(free0, devices)]
        if KBLAS:                                          # Kepler prompts: the fp32 weight block and inputs
            free = [f - (KDQ_BYTES + C * max(E, F) * 4 if kepler(self.cp, d) else 0) for f, d in zip(free, devices)]
        if not self.embd_host:
            free[0] -= self._nbytes("token_embd.weight")
        free[-1] -= self._nbytes("output.weight") + self.vocab * 4 * 2
        cap = [max(0, int(f // per)) for f in free]
        self.n_host = 0
        if sum(cap) < self.n_layer and os.environ.get("KINDI_NB_HOST", "1") == "1":
            # host layers: the layers that do not fit keep their weights in pinned host memory (the last card).
            # At each pass the card copies them, one layer after the other, into HOST_SLOTS layer buffers, while
            # it computes the layers before them (_hcopy). Their K/V stay on the card.
            kvl = self._kvl(ctx)
            slots = HOST_SLOTS * (per - kvl)
            dev = lambda h: max(0, int((free[-1] - h * kvl - (slots if h else 0)) // per))
            while sum(cap[:-1]) + self.n_host + dev(self.n_host) < self.n_layer and (self.n_host + 1) * kvl <= free[-1]:
                self.n_host += 1
            cap[-1] = self.n_host + dev(self.n_host)
        if not quiet:
            self.log("fit: ctx %d, %.0f MB per layer, free MB %s, room for %s layers%s" % (
                ctx, per / 2 ** 20, [int(f / 2 ** 20) for f in free], cap,
                ", %d layers in host memory (%d slots)" % (self.n_host, HOST_SLOTS) if self.n_host else ""))
        return sum(cap) >= self.n_layer, cap

    def _raw(self, name):
        ne, ty, off = self.tinfo[name]
        start = self.g.data_start + off
        return np.frombuffer(self.mm, np.uint8, self._nbytes(name), start), ne, ty

    def _host(self, a):
        """a copy of device array a in pinned host memory, as a device array (UVA: the card reads it over PCIe)"""
        cp = self.cp
        mem = cp.cuda.alloc_pinned_memory(a.nbytes)
        h = cp.ndarray(a.shape, a.dtype, cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(mem.ptr, a.nbytes, mem), 0))
        cp.cuda.runtime.memcpy(h.data.ptr, a.data.ptr, a.nbytes, 3)
        return h

    def _upload(self, c, name, host=False, rr=None):
        if host and c.k35 and CPU_F > 0 and rr is None and self.tinfo[name][1] == 18 \
                and name.split(".")[-2] in ("ffn_gate", "ffn_up", "ffn_down"):
            # a host layer FFN matrix: rows [0, rg) for the card, [rg, rows) for the CPU in a step
            ne = self.tinfo[name][0]
            rows, cols = int(ne[1]), int(ne[0])
            rg = int(rows * (1 - CPU_F)) // 128 * 128
            parts = []
            for a, b in ((0, rg), (rg, rows)):
                p = self._upload(c, name, host=True, rr=(a, b))
                p.r0 = a
                parts.append(p)
            return Split(parts, self._raw(name)[0], 18, rows, cols)
        w = self._upload_dev(c, name, rr)
        if c.k35 and isinstance(w, Weight) and name != "token_embd.weight" and not hasattr(w, "nc"):
            w = self._kreorder(c, w)
        if host and isinstance(w, Weight):
            self.cp.cuda.Device().synchronize()
            if name == "token_embd.weight":               # read in place over PCIe (one row per token)
                body = self._host(w.body)
                nc = getattr(w, "nc", None)
                w = Weight(body, body if w.d is w.body else self._host(w.d), w.ty, w.rows, w.cols)
                if nc:
                    w.nc = nc
            else:                                         # a host layer: copied into a slot at each pass (_hcopy)
                w.hb = self._host(w.body)
                w.hd = None if w.d is w.body else self._host(w.d)
                w.body = w.d = None
        return w

    # ------------------------------------------------------------------ host layers
    def _hslots(self, c):
        """the device buffers of the host layers: HOST_SLOTS slots, each with room for every matrix of a layer.
        Host layer i (in order) uses slot i mod HOST_SLOTS; its Weight objects point into that slot."""
        cp = self.cp
        c.hl = sorted(c.host)
        S = min(HOST_SLOTS, len(c.hl))
        c.hslot = []
        for s in range(S):
            bufs = {}
            for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
                for j in range(len(self._pieces(c.w[(c.hl[s], t)]))):
                    ws = [self._pieces(c.w[(il, t)])[j] for il in c.hl[s::S]]
                    nb = max(w.hb.nbytes for w in ws)
                    nd = max(w.hd.nbytes if w.hd is not None else 0 for w in ws)
                    bufs[(t, j)] = (cp.empty(nb, cp.uint8), cp.empty(max(1, nd // 2), cp.float16))
            c.hslot.append(bufs)
        for i, il in enumerate(c.hl):
            for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
                for j, w in enumerate(self._pieces(c.w[(il, t)])):
                    b, d = c.hslot[i % S][(t, j)]
                    w.body = b[:w.hb.nbytes]
                    w.d = d[:w.hd.size] if w.hd is not None else w.body
        c.hstream = cp.cuda.Stream(non_blocking=True)
        c.hready = {il: cp.cuda.Event(disable_timing=True) for il in c.hl}
        c.hfree = [cp.cuda.Event(disable_timing=True) for _ in range(S)]
        self.log("host layers %s: %d slots, %.0f MB on the card, %.0f MB pinned" % (
            c.hl, S, sum(a.nbytes + b.nbytes for bf in c.hslot for a, b in bf.values()) / 2 ** 20,
            sum(w.hb.nbytes for il in c.hl for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down") for w in self._pieces(c.w[(il, t)])) / 2 ** 20))
        if any(isinstance(c.w[(il, "ffn_gate")], Split) for il in c.hl):
            from . import nanbeige_cpu
            self.cpu = nanbeige_cpu.lib()
            c.hxq = np.frombuffer(cp.cuda.alloc_pinned_memory(self.n_ff), np.int8, self.n_ff)
            c.hxs = np.frombuffer(cp.cuda.alloc_pinned_memory(self.n_ff // 32 * 4), np.float32, self.n_ff // 32)
            mem = cp.cuda.alloc_pinned_memory(self.n_ff * 4)
            c.hy = np.frombuffer(mem, np.float32, self.n_ff)
            c.hyd = cp.ndarray((self.n_ff,), cp.float32, cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(mem.ptr, self.n_ff * 4, mem), 0))
            c.hev = cp.cuda.Event(disable_timing=True)
            for il in c.hl:                      # the CPU rows in the index form of nbq_cpu.c
                for t in ("ffn_gate", "ffn_up", "ffn_down"):
                    w = c.w[(il, t)]
                    if isinstance(w, Split):
                        n = w.rows - w.rg
                        w.wi = np.empty(n * w.cols // 4, np.uint16)
                        w.wb = np.empty(n * w.cols // 256 * 6, np.uint8)
                        self.cpu.nbq_cpu_expand_idx(w.raw.ctypes.data, w.rg, w.rows, w.cols, w.wi.ctypes.data,
                                                    w.wb.ctypes.data)
            self.log("host layer FFN: %.0f %% of the rows on the CPU in a step (KINDI_NB_CPUF)" % (CPU_F * 100))

    @staticmethod
    def _pieces(w):
        return w.parts if isinstance(w, Split) else [w]

    def _hcopy(self, c, i):
        """queue the copy of host layer i into its slot, after the layer before it in that slot is done (in a step,
        only the card parts of the Split matrices)"""
        S = len(c.hslot)
        il = c.hl[i]
        c.hstream.wait_event(c.hfree[i % S])
        for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
            ps = self._pieces(c.w[(il, t)])
            for w in (ps[:1] if c.hstep and len(ps) > 1 else ps):
                self.cp.cuda.runtime.memcpyAsync(w.body.data.ptr, w.hb.data.ptr, w.hb.nbytes, 1, c.hstream.ptr)
                if w.hd is not None:
                    self.cp.cuda.runtime.memcpyAsync(w.d.data.ptr, w.hd.data.ptr, w.hd.nbytes, 1, c.hstream.ptr)
        c.hready[il].record(c.hstream)

    def _hstart(self, c, step=False):
        """at the start of a pass over the layers of card c: the first slots fill while the card layers run"""
        c.hstep = step
        if c.host:
            for i in range(len(c.hslot)):
                self._hcopy(c, i)

    def _hpre(self, c, il):
        if il in c.host:
            c.stream.wait_event(c.hready[il])

    def _hpost(self, c, il):
        if il in c.host:
            i = c.hl.index(il)
            c.hfree[i % len(c.hslot)].record(c.stream)
            if i + len(c.hslot) < len(c.hl):
                self._hcopy(c, i + len(c.hslot))

    def _kreorder(self, c, w):
        """Kepler: the chunk-major layout of nbq_mvc (chunks of 4 blocks, row after row)"""
        cp = self.cp
        nb, nc = w.cols // 256, (w.cols // 256 + 3) // 4
        body = cp.empty(nc * w.rows * 4 * BLOCK_BODY[w.ty], cp.uint8)
        iq = w.d is not w.body
        d = cp.empty(nc * w.rows * 4, cp.float16) if iq else body
        c.k["nbq_kreorder"]((4 * self.nsm,), (256,), (w.body, w.d if iq else NULL, body, d if iq else NULL,
                                                       np.int32(w.rows), np.int32(nb), np.int32(BLOCK_BODY[w.ty])))
        cp.cuda.Device().synchronize()
        k = Weight(body, d, w.ty, w.rows, w.cols)
        k.nc = nc
        c.kpart = max(getattr(c, "kpart", 0), nc * w.rows)
        return k

    def _upload_dev(self, c, name, rr=None):
        cp = self.cp
        raw, ne, ty = self._raw(name)
        if ty == 0:
            return cp.asarray(raw.view(np.float32).copy())
        rows, cols = int(ne[1]), int(ne[0])
        if rr is not None:                                 # rows [rr[0], rr[1]) only
            rb = cols // 256 * BLOCK[ty]
            raw = raw[rr[0] * rb:rr[1] * rb]
            rows = rr[1] - rr[0]
        if ty not in IQ and c.k35 and name != "token_embd.weight":
            # Kepler: the chunk-major layout of nbq_mvc, made here on the host (the head is 0.6 GB)
            nb, nc = cols // 256, (cols // 256 + 3) // 4
            a = np.zeros((rows, nc * 4, BLOCK[ty]), np.uint8)
            a[:, :nb] = raw.reshape(rows, nb, BLOCK[ty])
            src = cp.asarray(np.ascontiguousarray(a.reshape(rows, nc, 4 * BLOCK[ty]).transpose(1, 0, 2)).reshape(-1))
            del a
            w = Weight(src, src, ty, rows, cols)
            w.nc = nc
            c.kpart = max(getattr(c, "kpart", 0), nc * rows)
            return w
        src = cp.asarray(raw)
        if ty not in IQ:
            return Weight(src, src, ty, rows, cols)
        nblk = rows * cols // 256
        body = cp.empty(nblk * (BLOCK[ty] - 2), cp.uint8)
        d = cp.empty(nblk, cp.float16)
        c.k["nbq_repack"](((nblk + 7) // 8,), (32, 8), (src, body, d, np.int64(nblk), np.int32(BLOCK[ty])))
        del src
        return Weight(body, d, ty, rows, cols)

    def _load_card(self, c, first, last):
        cp = self.cp
        with cp.cuda.Device(c.dev):
            if first:
                c.w["embd"] = self._upload(c, "token_embd.weight", host=self.embd_host)
            for il in range(c.lo, c.hi):
                for t in ("attn_norm", "ffn_norm"):
                    c.w[(il, t)] = self._upload(c, "blk.%d.%s.weight" % (il, t))
                for t in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
                    c.w[(il, t)] = self._upload(c, "blk.%d.%s.weight" % (il, t), host=il in c.host)
                    if il in c.host:
                        cp.get_default_memory_pool().free_all_blocks()
                if c.k35 or c.host:                  # the repack / reorder copies: back to the driver
                    cp.get_default_memory_pool().free_all_blocks()
                if self.q8:
                    for a in (c.kc, c.vc):
                        a[il] = cp.zeros((self.n_kv, self.ctx, self.hd), cp.int8)
                    for a in (c.kd, c.vd):
                        a[il] = cp.zeros((self.n_kv, self.ctx, self.hd // 32), cp.float16)
                else:
                    c.kc[il] = cp.zeros((self.n_kv, self.ctx, self.hd), cp.float16)
                    c.vc[il] = cp.zeros((self.n_kv, self.ctx, self.hd), cp.float16)
            if c.host:
                self._hslots(c)
            if last:
                c.w["out_norm"] = self._upload(c, "output_norm.weight")
                c.w["out"] = self._upload(c, "output.weight")
                c.logits = cp.zeros(self.vocab, cp.float32)
                c.am_v = cp.zeros(256, cp.float32)
                c.am_i = cp.zeros(256, cp.int32)
                c.tok_out = cp.zeros(4, cp.int32)
            E, F = self.n_embd, self.n_ff
            c.x = cp.zeros(E, cp.float32)
            c.br = cp.zeros(E, cp.float32)
            c.xq = cp.zeros(max(E, F), cp.int8)
            c.xs = cp.zeros(max(E, F) // 32, cp.float32)
            c.q = cp.zeros(E, cp.float32)
            c.kk = cp.zeros(self.n_kv * self.hd, cp.float32)
            c.vv = cp.zeros(self.n_kv * self.hd, cp.float32)
            c.att = cp.zeros(E, cp.float32)
            c.gu = cp.zeros(2 * F, cp.float32)
            c.part_o = cp.zeros(self.n_head * NSPLIT * self.hd, cp.float32)
            c.part_ms = cp.zeros(self.n_head * NSPLIT * 2, cp.float32)
            c.pos = cp.zeros(4, cp.int32)
            c.tok = cp.zeros(4, cp.int32)
            if c.k35 and getattr(c, "kpart", 0):    # the nbq_mvc chunk sums (made here, not inside a graph capture)
                c.KPART = cp.zeros(c.kpart, cp.float32)
            cp.cuda.Device().synchronize()

    # ------------------------------------------------------------------ one token
    def _mv(self, c, w, y, add=None):
        if isinstance(w, Split):
            return self._mv_split(c, w, y, add)
        if c.k35:
            return self._mvc(c, w, y, add)
        c.k[MV[w.ty]]((self._grid(w.rows),), (256,), (w.body, w.d, c.xq, c.xs, y, add if add is not None else NULL,
                                                    np.int32(w.rows), np.int32(w.cols)))

    def _mv_split(self, c, w, y, add=None):
        """a step: the card computes rows [0, rg) and the CPU rows [rg, rows) at the same time; the CPU waits only
        for the input (copied to the host before the card part starts)"""
        cp = self.cp
        assert add is None
        a = w.parts[0]
        cp.cuda.runtime.memcpyAsync(c.hxq.ctypes.data, c.xq.data.ptr, w.cols, 2, c.stream.ptr)
        cp.cuda.runtime.memcpyAsync(c.hxs.ctypes.data, c.xs.data.ptr, w.cols // 32 * 4, 2, c.stream.ptr)
        c.hev.record(c.stream)
        self._mvc(c, a, y)
        c.hev.synchronize()
        self.cpu.nbq_cpu_dot_idx(w.wi.ctypes.data, w.wb.ctypes.data, w.rows - w.rg, w.cols, c.hxq.ctypes.data,
                                 c.hxs.ctypes.data, c.hy.ctypes.data)
        # the CPU rows go in by a kernel that reads the pinned memory: a memcpy would wait in the copy engine
        # behind the host-layer copies of the next layers
        n = w.rows - w.rg
        c.k["nbq_kcopy"]((min(64, -(-n // 256)),), (256,), (c.hyd, y[w.rg:], np.int32(n)))

    def _mvc(self, c, w, y, add=None):
        """Kepler: y = W x by column chunks (nbq_mvc), then the sum of the chunks"""
        if getattr(c, "KPART", None) is None:
            with self.cp.cuda.Device(c.dev):
                c.KPART = self.cp.zeros(c.kpart, self.cp.float32)
                self.cp.cuda.Device().synchronize()
        wpc = max(1, min(w.rows // KRB, int(os.environ.get("KINDI_NB_KWARPS", "2560")) // w.nc))
        rpw = -(-w.rows // (wpc * KRB)) * KRB
        wpc = -(-w.rows // rpw)
        c.k[MVC[w.ty]]((-(-wpc // 4), w.nc), (128,), (w.body, w.d, c.xq, c.xs, c.KPART, np.int32(w.rows),
                                                     np.int32(w.cols), np.int32(rpw)))
        c.k["nbq_mvc_sum"]((-(-w.rows // 256),), (256,), (c.KPART, np.int32(w.nc), np.int32(w.rows), y,
                                                        add if add is not None else NULL))

    def _grid(self, rows):
        return min((rows + 7) // 8, self.nsm * int(os.environ.get("KINDI_NB_BPS", "4")))

    def _kv(self, c, il):
        """the cache arguments of the attention kernels for layer il"""
        return (c.kc[il], c.kd[il], c.vc[il], c.vd[il]) if self.q8 else (c.kc[il], c.vc[il])

    def _store_q8(self, c, il, q, k, v, ld, n, pos, theta_scale):
        """RoPE and the q8_0 K/V store of n rows (q, k, v: row 0; row t at + t * ld floats)"""
        c.k["nbq_rope_store_q8"]((self.n_head + self.n_kv, n), (128,), (q, k, v, np.int32(ld), pos, np.int32(self.n_head),
                                 np.int32(self.n_kv), theta_scale, c.kc[il], c.kd[il], c.vc[il], c.vd[il],
                                 np.int32(self.ctx)))

    def _layers(self, c, first, head):
        """the kernel launches of one token on card c (recorded into its graph)"""
        k, E, F = c.k, self.n_embd, self.n_ff
        eps = np.float32(self.eps)
        if first:
            k["nbq_embed_t%d" % c.w["embd"].ty](((E // 32 + 7) // 8, 1), (256,), (c.w["embd"].body, c.w["embd"].d, c.tok,
                                                                          np.int32(E), c.x))
        theta_scale = np.float32(self.base ** (-2.0 / self.hd))
        scale = np.float32(1.0 / np.sqrt(self.hd))
        self._hstart(c, step=True)
        for il in range(c.lo, c.hi):
            self._hpre(c, il)
            add = c.br if il > c.lo else NULL
            k["nbq_add_rms_q8"]((1,), (1024,), (c.x, add, c.w[(il, "attn_norm")], np.int32(E), eps, c.xq, c.xs))
            wq, wk, wv = c.w[(il, "attn_q")], c.w[(il, "attn_k")], c.w[(il, "attn_v")]
            if FUSED_QKV and (wq.ty, wk.ty, wv.ty) == (22, 22, 12) and not c.k35 and not self.q8:
                k["nbq_qkv_rope"]((self._grid((wq.rows + wk.rows + wv.rows) // 2),), (256,),
                                  (wq.body, wq.d, wk.body, wk.d, wv.body, c.xq, c.xs, np.int32(self.n_head),
                                   np.int32(self.n_kv), np.int32(E), c.pos, theta_scale, c.q, c.kc[il], c.vc[il],
                                   np.int32(self.ctx)))
            else:
                self._mv(c, wq, c.q)
                self._mv(c, wk, c.kk)
                self._mv(c, wv, c.vv)
                if self.q8:
                    self._store_q8(c, il, c.q, c.kk, c.vv, 0, 1, c.pos, theta_scale)
                else:
                    k["nbq_rope_store"]((self.n_head + self.n_kv,), (64,), (c.q, c.kk, c.vv, c.pos, np.int32(self.n_head),
                                        np.int32(self.n_kv), theta_scale, c.kc[il], c.vc[il], np.int32(self.ctx)))
            k[self.k_attn]((self.n_kv, NSPLIT), (256,), (c.q,) + self._kv(c, il) + (c.pos, np.int32(self.ctx), scale,
                               c.part_o, c.part_ms, np.int32(0)))
            k["nbq_attn_merge_q8"]((self.n_head,), (128,), (c.part_o, c.part_ms, np.int32(NSPLIT), c.xq, c.xs))
            self._mv(c, c.w[(il, "attn_output")], c.br)
            k["nbq_add_rms_q8"]((1,), (1024,), (c.x, c.br, c.w[(il, "ffn_norm")], np.int32(E), eps, c.xq, c.xs))
            wg, wu = c.w[(il, "ffn_gate")], c.w[(il, "ffn_up")]
            if wg.ty == wu.ty == 18 and not c.k35:
                k["nbq_mv2_iq3xxs"]((self._grid(2 * F),), (256,), (wg.body, wg.d, wu.body, wu.d, c.xq, c.xs, c.gu,
                                    np.int32(F), np.int32(E)))
            else:
                self._mv(c, wg, c.gu)
                self._mv(c, wu, c.gu[F:])
            k["nbq_swiglu_q8"](((F // 32 + 7) // 8,), (256,), (c.gu, np.int32(F), c.xq, c.xs))
            self._mv(c, c.w[(il, "ffn_down")], c.br)
            self._hpost(c, il)
        if head:
            k["nbq_add_rms"]((1,), (1024,), (c.x, c.br, c.w["out_norm"], np.int32(E), eps, NULL, c.xq, c.xs))
            self._mv(c, c.w["out"], c.logits)
            k["nbq_argmax_part"]((256,), (256,), (c.logits, np.int32(self.vocab), c.am_v, c.am_i))
            k["nbq_argmax_final"]((1,), (32,), (c.am_v, c.am_i, np.int32(256), c.tok_out, NULL))
        else:
            k["nbq_add"](((E + 255) // 256,), (256,), (c.x, c.br, np.int32(E)))

    def _graph(self, c, head):
        cp = self.cp
        key = bool(head)
        if key not in c.graph:
            with cp.cuda.Device(c.dev):
                c.graph[key] = _capture(c.stream, lambda: self._layers(c, c is self.cards[0], head), replay=bool(c.host))
        return c.graph[key]

    def step(self, tok, pos, head=True):
        """feed one token at position pos; with head, return the greedy next token"""
        self.step_begin(tok, pos, head)
        return self.step_end(head)

    def step_begin(self, tok, pos, head=True):
        """queue one step on the cards and return at once; step_end() waits for it (the host can do other work
        while the cards compute)"""
        cp = self.cp
        self.h_tok[0], self.h_tok[1] = tok, pos
        if not hasattr(self, "h_sx"):
            self.h_sx = [np.frombuffer(cp.cuda.alloc_pinned_memory(self.n_embd * 4), np.float32, self.n_embd)
                         for _ in self.cards]
            for c in self.cards:
                with cp.cuda.Device(c.dev):
                    c.ev_step = cp.cuda.Event(disable_timing=True)
        nb = self.n_embd * 4
        for i, c in enumerate(self.cards):
            last = i == len(self.cards) - 1
            g = self._graph(c, head and last)
            with cp.cuda.Device(c.dev):
                if i == 0:
                    cp.cuda.runtime.memcpyAsync(c.tok.data.ptr, self.h_tok.ctypes.data, 4, 1, c.stream.ptr)
                else:                         # the hidden state comes through pinned host memory, ordered by an event
                    c.stream.wait_event(self.cards[i - 1].ev_step)
                    cp.cuda.runtime.memcpyAsync(c.x.data.ptr, self.h_sx[i].ctypes.data, nb, 1, c.stream.ptr)
                cp.cuda.runtime.memcpyAsync(c.pos.data.ptr, self.h_tok.ctypes.data + 4, 4, 1, c.stream.ptr)
                g.launch(c.stream)
                if not last:
                    cp.cuda.runtime.memcpyAsync(self.h_sx[i + 1].ctypes.data, c.x.data.ptr, nb, 2, c.stream.ptr)
                    c.ev_step.record(c.stream)
        c = self.cards[-1]
        with cp.cuda.Device(c.dev):
            if head:
                cp.cuda.runtime.memcpyAsync(self.h_tok.ctypes.data + 8, c.tok_out.data.ptr, 4, 2, c.stream.ptr)

    def step_end(self, head=True):
        c = self.cards[-1]
        with self.cp.cuda.Device(c.dev):
            c.stream.synchronize()
        return int(self.h_tok[2]) if head else None

    def step_async(self, tok, pos, chain=False):
        """one-card engines: queue one step (tok at pos; with chain, the token is the previous step's output, on the
        card) and return a slot; result(slot) waits and returns the greedy token"""
        cp = self.cp
        c = self.cards[0]
        assert len(self.cards) == 1
        if not hasattr(self, "h_as"):
            self.h_as = np.frombuffer(cp.cuda.alloc_pinned_memory(8 * 16), np.int32, 32)
            self.n_as = 0
        k = self.n_as % 8
        self.n_as += 1
        h = self.h_as[4 * k:4 * k + 4]
        h[0], h[1] = tok, pos
        g = self._graph(c, True)
        with cp.cuda.Device(c.dev):
            if chain:
                cp.cuda.runtime.memcpyAsync(c.tok.data.ptr, c.tok_out.data.ptr, 4, 3, c.stream.ptr)
            else:
                cp.cuda.runtime.memcpyAsync(c.tok.data.ptr, h.ctypes.data, 4, 1, c.stream.ptr)
            cp.cuda.runtime.memcpyAsync(c.pos.data.ptr, h.ctypes.data + 4, 4, 1, c.stream.ptr)
            g.launch(c.stream)
            cp.cuda.runtime.memcpyAsync(h.ctypes.data + 8, c.tok_out.data.ptr, 4, 2, c.stream.ptr)
        return k

    def result(self, k):
        c = self.cards[0]
        with self.cp.cuda.Device(c.dev):
            c.stream.synchronize()
        return int(self.h_as[4 * k + 2])

    def generate_spec1(self, ids, n, draft, stats=None):
        """greedy output of this model with one draft token per pass. While the first card of this model checks
        [t, d], the draft (on another card or stream) goes on as if d is right: it feeds d and its own guess e of
        the next token. When d and e are both right, the next proposal is ready when the check ends."""
        t = self.prefill_batch(ids)
        draft.prefill_batch(ids)
        t0 = time.time()
        out, pos = [t], len(ids)
        passes = free = 0
        d = draft.step(t, pos)                         # the proposal for pos + 1
        while len(out) < n:
            # the draft goes first: work queued on a card behind the check's event wait would wait for it too (WSL)
            ke = draft.step_async(d, pos + 1)          # e: the draft's guess of the token after d
            kf = draft.step_async(0, pos + 2, chain=True)   # f: the draft's guess after e
            done = self.check([t, d], pos, wait=False)
            got = done()
            e, f = draft.result(ke), draft.result(kf)
            passes += 1
            if got[0] == d:                            # d taken; got[1] follows it
                out += [d, got[1]]
                t, pos = got[1], pos + 2
                if e == got[1]:
                    d = f                              # the draft already fed d and e: f is the next proposal
                    free += 1
                else:
                    d = draft.step(t, pos)
            else:
                out.append(got[0])
                t, pos = got[0], pos + 1
                d = draft.step(t, pos)
        if stats is not None:
            stats["passes"], stats["free"], stats["t_gen"] = passes, free, time.time() - t0
        return out[:n]

    # ------------------------------------------------------------------ prompt
    def _pbufs(self, c):
        cp, E, F, C = self.cp, self.n_embd, self.n_ff, CHUNK
        if not hasattr(c, "X"):
            with cp.cuda.Device(c.dev):
                c.X = cp.zeros((C, E), cp.float32)
                c.BR = cp.zeros((C, E), cp.float32)
                c.XQ = cp.zeros((C, F), cp.int8)
                c.XS = cp.zeros((C, F // 32), cp.float32)
                c.QKV = cp.zeros((C, E + 2 * self.n_kv * self.hd), cp.float32)
                c.ATT = cp.zeros((C, E), cp.float32)
                c.GU = cp.zeros((C, 2 * F), cp.float32)
                c.P = cp.zeros(C, cp.int32)
                c.TOK = cp.zeros(C, cp.int32)
                c.PO = cp.zeros(ATT_SUB * self.n_head * NSPLIT * self.hd, cp.float32)
                c.PMS = cp.zeros(ATT_SUB * self.n_head * NSPLIT * 2, cp.float32)
                cp.cuda.Device().synchronize()   # the zeros run on the null stream; c.stream does not wait for it

    def _kmm(self, c, w, nt, y, ldy, yo, fresh=True):
        """Kepler: Y[t][yo + r] = W x_t for nt prompt tokens. Default: the weights as fp32 in blocks of rows
        (nbq_kdeq) times the inputs as fp32 (nbq_kx2f) with cublasSgemm; KINDI_NB_KBLAS=0: nbq_kmm (fp32 tiles of
        64 rows x 64 tokens, about half the speed on the K6000)"""
        if KBLAS:
            return self._kblas(c, w, nt, y, ldy, yo, fresh)
        assert w.rows % 64 == 0
        c.k[KMM[w.ty]]((w.rows // 64, -(-nt // 64)), (128,), (w.body, w.d, c.XQ, c.XS, y, np.int32(w.rows),
                                                              np.int32(w.cols), np.int32(nt), np.int32(ldy), np.int32(yo)))

    def _kblas(self, c, w, nt, y, ldy, yo, fresh=True):
        cp = self.cp
        try:
            from cupy_backends.cuda.libs import cublas
        except ImportError:
            from cupy.cuda import cublas
        if getattr(c, "KDQ", None) is None:
            c.KDQ = cp.empty(KDQ_BYTES // 4, cp.float32)
            c.KXF = cp.empty(CHUNK * max(self.n_embd, self.n_ff), cp.float32)
            c.kab = np.array([1.0, 0.0], np.float32)
            cp.cuda.Device().synchronize()
        if fresh:                                      # the inputs (XQ, XS) as fp32; segments of _mm3 share them
            c.k["nbq_kx2f"]((min(4096, -(-nt * w.cols // 256)),), (256,), (c.XQ, c.XS, np.int32(nt), np.int32(w.cols), c.KXF))
        h = cp.cuda.device.get_cublas_handle()
        cublas.setStream(h, c.stream.ptr)
        rb = max(64, KDQ_BYTES // (4 * w.cols) // 64 * 64)
        for r0 in range(0, w.rows, rb):
            nr = min(rb, w.rows - r0)
            c.k[KDEQ[w.ty]]((nr, -(-(w.cols // 32) // 128)), (128,), (w.body, w.d, np.int32(w.rows), np.int32(w.cols),
                                                                     np.int32(r0), np.int32(nr), c.KDQ))
            cublas.sgemm(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, nr, nt, w.cols, c.kab.ctypes.data, c.KDQ.data.ptr,
                         w.cols, c.KXF.data.ptr, w.cols, c.kab.ctypes.data + 4, y.data.ptr + 4 * (yo + r0), ldy)

    def _mm(self, c, w, nt, y, ldy, yo):
        if isinstance(w, Split):
            return self._mm3(c, [(w, yo)], nt, y, ldy)
        if c.k35:
            return self._kmm(c, w, nt, y, ldy, yo)
        self._mm3(c, [(w, yo)], nt, y, ldy)

    def _mm3(self, c, segs, nt, y, ldy):
        if any(isinstance(w, Split) for w, _ in segs):
            segs = [(p, yo + p.r0) for w, yo in segs for p in self._pieces(w)]
        if c.k35:
            for i, (w, yo) in enumerate(segs):
                self._kmm(c, w, nt, y, ldy, yo, fresh=i == 0)
            return
        """one launch for up to three matrices with the same columns: segs = [(weight, y column offset)]"""
        tys = [w.ty for w, _ in segs]
        name = MM[tys[0]] if len(set(tys)) == 1 else MM[(tys[0], tys[-1])]
        segs = segs + [(None, 0)] * (3 - len(segs))
        args, nblk = [], 0
        for w, yo in segs:
            if w is None:
                args += [NULL, NULL, np.int32(0), np.int32(0)]
            else:
                args += [w.body, w.d, np.int32(w.rows), np.int32(yo)]
                nblk += (w.rows + 127) // 128
        cols = segs[0][0].cols
        c.k[name]((nblk, (nt + 127) // 128), (256,), tuple(args) + (c.XQ, c.XS, y, np.int32(cols), np.int32(nt),
                                                                     np.int32(ldy)), shared_mem=GEMM_SMEM)

    def _prompt_card(self, c, nt, pos0, first):
        k, E, F = c.k, self.n_embd, self.n_ff
        eps = np.float32(self.eps)
        ld = E + 2 * self.n_kv * self.hd
        theta_scale = np.float32(self.base ** (-2.0 / self.hd))
        scale = np.float32(1.0 / np.sqrt(self.hd))
        if first:
            k["nbq_embed_t%d" % c.w["embd"].ty](((E // 32 + 7) // 8, nt), (256,), (c.w["embd"].body, c.w["embd"].d, c.TOK,
                                                                           np.int32(E), c.X))
        self._hstart(c)
        for il in range(c.lo, c.hi):
            self._hpre(c, il)
            add = c.BR if il > c.lo else NULL
            k["nbq_add_rms_q8_rows"]((nt,), (1024,), (c.X, add, c.w[(il, "attn_norm")], np.int32(E), eps, c.XQ, c.XS))
            self._mm3(c, [(c.w[(il, "attn_q")], 0), (c.w[(il, "attn_k")], E),
                          (c.w[(il, "attn_v")], E + self.n_kv * self.hd)], nt, c.QKV, ld)
            if self.q8:
                self._store_q8(c, il, c.QKV, c.QKV[0, E:], c.QKV[0, E + self.n_kv * self.hd:], ld, nt, c.P, theta_scale)
            else:
                k["nbq_rope_store_rows"]((self.n_head + self.n_kv, nt), (64,), (c.QKV, c.P, np.int32(self.n_head),
                                         np.int32(self.n_kv), theta_scale, c.kc[il], c.vc[il], np.int32(self.ctx)))
            if os.environ.get("KINDI_NB_FA", "1") == "1" and not c.k35:     # Kepler: no tensor cores
                k[self.k_fa](((self.G * nt + 63) // 64, self.n_kv), (128,),
                                    (c.QKV, np.int32(ld), np.int32(nt), np.int32(pos0)) + self._kv(c, il) +
                                    (np.int32(self.ctx), scale, c.ATT), shared_mem=FA_SMEM)
            else:
                for t0 in range(0, nt, ATT_SUB):
                    n = min(ATT_SUB, nt - t0)
                    k[self.k_attn]((self.n_kv, NSPLIT, n), (256,), (c.QKV[t0],) + self._kv(c, il) + (c.P[t0:],
                                       np.int32(self.ctx), scale, c.PO, c.PMS, np.int32(ld)))
                    k["nbq_attn_merge"]((self.n_head, n), (128,), (c.PO, c.PMS, np.int32(NSPLIT), c.ATT[t0]))
            k["nbq_q8_rows"](((E // 32 + 7) // 8, nt), (256,), (c.ATT, np.int32(E), c.XQ, c.XS))
            self._mm(c, c.w[(il, "attn_output")], nt, c.BR, E, 0)
            k["nbq_add_rms_q8_rows"]((nt,), (1024,), (c.X, c.BR, c.w[(il, "ffn_norm")], np.int32(E), eps, c.XQ, c.XS))
            self._mm3(c, [(c.w[(il, "ffn_gate")], 0), (c.w[(il, "ffn_up")], F)], nt, c.GU, 2 * F)
            k["nbq_swiglu_q8_rows"](((F // 32 + 7) // 8, nt), (256,), (c.GU, np.int32(F), c.XQ, c.XS))
            self._mm(c, c.w[(il, "ffn_down")], nt, c.BR, E, 0)
            self._hpost(c, il)
        k["nbq_add"](((nt * E + 255) // 256,), (256,), (c.X, c.BR, np.int32(nt * E)))

    def prefill_batch(self, ids, pos0=0, chunk=None, hidden=None):
        """feed ids from position pos0; return the greedy token after the last. The prompt goes in chunks; card i
        works on chunk j while card i + 1 works on chunk j - 1 (events order the hidden-state copies). hidden: an
        array of len(ids) x n_embd that gets the last layer's output at each position (before the output norm)."""
        if self.kslow:
            return self.prefill(ids, pos0)
        if self.k35 and os.environ.get("KINDI_NB_KROWS", "0") == "1":
            return self._prefill_rows(ids, pos0)
        cp, E = self.cp, self.n_embd
        if not self.tc and not self.k35:
            return self._prefill_check(ids, pos0, hidden)
        # 256-token chunks fill the pipeline of two cards sooner; past 4K tokens 512 use the tensor cores better
        chunk = min(CHUNK, chunk or int(os.environ.get("KINDI_NB_PCHUNK", "256" if len(ids) <= 4096 and not self.k35 else "512")))
        if not hasattr(self, "h_px"):
            self.h_ptok = np.frombuffer(cp.cuda.alloc_pinned_memory(CHUNK * 4), np.int32, CHUNK)
            self.h_px = [None] + [np.frombuffer(cp.cuda.alloc_pinned_memory(CHUNK * E * 4), np.float32, CHUNK * E)
                                  for _ in self.cards[1:]]
        for c in self.cards:
            self._pbufs(c)
            if not hasattr(c, "ev_done"):
                with cp.cuda.Device(c.dev):
                    c.ev_done = [cp.cuda.Event(disable_timing=True) for _ in range(2)]   # card done with chunk j (j % 2)
                    c.ev_copied = cp.cuda.Event(disable_timing=True)   # this card copied the previous card's chunk
                    c.ev_copied.record(c.stream)
        if len(ids) > self.h_ptok.size:
            self.h_ptok = np.frombuffer(cp.cuda.alloc_pinned_memory(len(ids) * 4), np.int32, len(ids))
        self.h_ptok[:len(ids)] = ids
        for j, s in enumerate(range(0, len(ids), chunk)):
            nt = min(chunk, len(ids) - s)
            for i, c in enumerate(self.cards):
                if j >= 2:
                    c.ev_done[j % 2].synchronize()     # keep at most one chunk queued per card, so that the
                                                       # launches of the next card are not held up behind it
                with cp.cuda.Device(c.dev), c.stream:
                    if i + 1 < len(self.cards):
                        c.stream.wait_event(self.cards[i + 1].ev_copied)   # X may be written again
                    if i == 0:
                        cp.cuda.runtime.memcpyAsync(c.TOK.data.ptr, self.h_ptok.ctypes.data + 4 * s, nt * 4, 1, c.stream.ptr)
                    else:                                              # card to card through pinned host memory
                        c.stream.wait_event(self.cards[i - 1].ev_done[j % 2])
                        cp.cuda.runtime.memcpyAsync(c.X.data.ptr, self.h_px[i].ctypes.data, nt * E * 4, 1, c.stream.ptr)
                        c.ev_copied.record(c.stream)
                    c.P[:nt] = cp.arange(pos0 + s, pos0 + s + nt, dtype=cp.int32)
                    self._prompt_card(c, nt, pos0 + s, i == 0)
                    if i + 1 < len(self.cards):
                        cp.cuda.runtime.memcpyAsync(self.h_px[i + 1].ctypes.data, c.X.data.ptr, nt * E * 4, 2, c.stream.ptr)
                    elif hidden is not None:
                        hidden[s:s + nt] = cp.asnumpy(c.X[:nt]).astype(hidden.dtype)
                    c.ev_done[j % 2].record(c.stream)
        c = self.cards[-1]
        with cp.cuda.Device(c.dev), c.stream:
            c.x[:] = c.X[nt - 1]
            c.br[:] = 0
            k = c.k
            k["nbq_add_rms"]((1,), (1024,), (c.x, c.br, c.w["out_norm"], np.int32(E), np.float32(self.eps), NULL, c.xq, c.xs))
            self._mv(c, c.w["out"], c.logits)
            k["nbq_argmax_part"]((256,), (256,), (c.logits, np.int32(self.vocab), c.am_v, c.am_i))
            k["nbq_argmax_final"]((1,), (32,), (c.am_v, c.am_i, np.int32(256), c.tok_out, NULL))
            t = int(c.tok_out[0].get())
        return t

    # ------------------------------------------------------------------ N tokens in one pass (speculative check)
    def _prefill_check(self, ids, pos0, hidden):
        """the prompt without tensor cores: check passes of up to 8 tokens (the decode kernels); afterwards the last
        card holds the last token's state as after prefill_batch (x, br, logits)"""
        cp = self.cp
        c = self.cards[-1]
        t = None
        for s in range(0, len(ids), 8):
            part = ids[s:s + 8]
            if len(part) == 1:
                t = self.step(part[0], pos0 + s)
                if hidden is not None:
                    with cp.cuda.Device(c.dev):
                        hidden[s] = cp.asnumpy(c.x + c.br).astype(hidden.dtype)
                continue
            t = self.check(part, pos0 + s)[-1]
            with cp.cuda.Device(c.dev), c.stream:
                if hidden is not None:
                    hidden[s:s + len(part)] = cp.asnumpy(c.X[:len(part)]).astype(hidden.dtype)
                if s + 8 >= len(ids):
                    c.x[:] = c.X[len(part) - 1]
                    c.br[:] = 0
                    c.logits[:] = c.LOG[len(part) - 1]
        with cp.cuda.Device(c.dev):
            c.stream.synchronize()
        return t

    def _mvn(self, c, w, n, y, ldy, yo, add=None, xq=None, xs=None):
        if isinstance(w, Split):
            for p in w.parts:
                self._mvn(c, p, n, y, ldy, yo + p.r0, add, xq, xs)
            return
        sm = 0                                             # stage_x in nbq_mvn.cu is off
        name = "%s_%d" % (MVN[w.ty], n)
        c.k[name]((self._grid(w.rows),), (256,), (w.body, w.d, c.XQ if xq is None else xq, c.XS if xs is None else xs, y,
                                                 add if add is not None else NULL, np.int32(w.rows), np.int32(w.cols),
                                                 np.int32(ldy), np.int32(yo)), shared_mem=sm)

    def _tcv(self, c, w, n, y, ldy, yo, xq=None, xs=None):
        """y[t][yo + r] = W x_t for n <= 8 inputs on tensor cores; long rows are split into ranges of runs, so that
        about KINDI_NB_TWARPS warps per SM have work"""
        tiles = (w.rows + 15) // 16
        want = self.nsm * int(os.environ.get("KINDI_NB_TWARPS", "24"))
        ns = max(1, min(int(os.environ.get("KINDI_NB_TSPLIT", "0")) or (want + tiles - 1) // tiles, w.cols // 32 // 16))
        if ns > 1 and (getattr(c, "TPART", None) is None or c.TPART.size < ns * 8 * w.rows):
            with self.cp.cuda.Device(c.dev):
                c.TPART = self.cp.zeros(ns * 8 * w.rows, self.cp.float32)
        part = c.TPART if ns > 1 else NULL
        c.k[TCV[w.ty]](((tiles + 3) // 4, ns), (128,),
                       (w.body, w.d, c.XQ if xq is None else xq, c.XS if xs is None else xs, y, np.int32(w.rows),
                        np.int32(w.cols), np.int32(n), np.int32(ldy), np.int32(yo), part))
        if ns > 1:
            c.k["nbq_tcv_sum"](((w.rows + 255) // 256, n), (256,), (part, np.int32(ns), np.int32(w.rows), np.int32(n), y,
                                                                    np.int32(ldy), np.int32(yo)))

    def _check_card(self, c, n, first, head):
        """the launches of an n-token pass on card c (recorded into its graph): the same per-token arithmetic as
        step(), so the n results are the ones n single steps give"""
        k, E, F = c.k, self.n_embd, self.n_ff
        eps = np.float32(self.eps)
        ld = E + 2 * self.n_kv * self.hd
        theta_scale = np.float32(self.base ** (-2.0 / self.hd))
        scale = np.float32(1.0 / np.sqrt(self.hd))
        if first:
            k["nbq_embed_t%d" % c.w["embd"].ty](((E // 32 + 7) // 8, n), (256,), (c.w["embd"].body, c.w["embd"].d, c.TOK,
                                                                          np.int32(E), c.X))
        self._hstart(c)
        for il in range(c.lo, c.hi):
            self._hpre(c, il)
            add = c.BR if il > c.lo else NULL
            k["nbq_add_rms_q8_rows"]((n,), (1024,), (c.X, add, c.w[(il, "attn_norm")], np.int32(E), eps, c.XQ, c.XS))
            wq, wk, wv = c.w[(il, "attn_q")], c.w[(il, "attn_k")], c.w[(il, "attn_v")]
            if FUSED_QKV and (wq.ty, wk.ty, wv.ty) == (22, 22, 12) and n <= 4 and not c.k35 and not self.q8:
                k["nbq_qkvn_rope_%d" % n]((self._grid((wq.rows + wk.rows + wv.rows) // 2),), (256,),
                                          (wq.body, wq.d, wk.body, wk.d, wv.body, c.XQ, c.XS, np.int32(self.n_head),
                                           np.int32(self.n_kv), np.int32(E), c.P, theta_scale, c.QKV, np.int32(ld),
                                           c.kc[il], c.vc[il], np.int32(self.ctx)))
            else:
                self._mvn(c, wq, n, c.QKV, ld, 0)
                self._mvn(c, wk, n, c.QKV, ld, E)
                self._mvn(c, wv, n, c.QKV, ld, E + self.n_kv * self.hd)
                if self.q8:
                    self._store_q8(c, il, c.QKV, c.QKV[0, E:], c.QKV[0, E + self.n_kv * self.hd:], ld, n, c.P,
                                   theta_scale)
                else:
                    k["nbq_rope_store_rows"]((self.n_head + self.n_kv, n), (64,), (c.QKV, c.P, np.int32(self.n_head),
                                             np.int32(self.n_kv), theta_scale, c.kc[il], c.vc[il], np.int32(self.ctx)))
            k[self.k_attn]((self.n_kv, NSPLIT, n), (256,), (c.QKV,) + self._kv(c, il) + (c.P, np.int32(self.ctx), scale,
                           c.PO, c.PMS, np.int32(ld)))
            k["nbq_attn_merge_q8_rows"]((self.n_head, n), (128,), (c.PO, c.PMS, np.int32(NSPLIT), c.XQ, c.XS))
            self._mvn(c, c.w[(il, "attn_output")], n, c.BR, E, 0)
            k["nbq_add_rms_q8_rows"]((n,), (1024,), (c.X, c.BR, c.w[(il, "ffn_norm")], np.int32(E), eps, c.XQ, c.XS))
            wg, wu = c.w[(il, "ffn_gate")], c.w[(il, "ffn_up")]
            if wg.ty == wu.ty == 18 and n <= 4 and not c.k35:
                k["nbq_mv2n_iq3xxs_%d" % n]((self._grid(2 * F),), (256,), (wg.body, wg.d, wu.body, wu.d, c.XQ, c.XS, c.GU,
                                                                         np.int32(F), np.int32(E)))
            else:
                self._mvn(c, wg, n, c.GU, 2 * F, 0)
                self._mvn(c, wu, n, c.GU, 2 * F, F)
            k["nbq_swiglu_q8_rows"](((F // 32 + 7) // 8, n), (256,), (c.GU, np.int32(F), c.XQ, c.XS))
            self._mvn(c, c.w[(il, "ffn_down")], n, c.BR, E, 0)
            self._hpost(c, il)
        k["nbq_add"](((n * E + 255) // 256,), (256,), (c.X, c.BR, np.int32(n * E)))
        if head:
            k["nbq_add_rms_q8_rows"]((n,), (1024,), (c.X, NULL, c.w["out_norm"], np.int32(E), eps, c.XQ, c.XS))
            self._mvn(c, c.w["out"], n, c.LOG, self.vocab, 0)
            k["nbq_argmax_rows"]((n,), (1024,), (c.LOG, np.int32(self.vocab), np.int32(self.vocab), c.TOKS))

    def _prefill_rows(self, ids, pos0=0):
        """Kepler, KINDI_NB_KROWS=1: the prompt in check() passes of 8 tokens (slow; for tests of nbq_kmm)"""
        t, i = None, 0
        while i < len(ids):
            nt = min(8, len(ids) - i)
            last = i + nt == len(ids)
            if nt == 1:
                t = self.step(ids[i], pos0 + i, head=last)
            else:
                t = self.check(ids[i:i + nt], pos0 + i, head=last)[-1]
            i += nt
        return t

    def check(self, toks, pos0, wait=True, head=True):
        """feed toks (2 .. 8 of them) at positions pos0, pos0 + 1, ...; return the greedy token after each (with
        wait=False, return a function that waits and returns them)"""
        cp, E = self.cp, self.n_embd
        n = len(toks)
        assert 2 <= n <= 8
        if not hasattr(self, "h_ck"):
            self.h_ck = np.frombuffer(cp.cuda.alloc_pinned_memory(64), np.int32, 16)
            self.h_cx = [np.frombuffer(cp.cuda.alloc_pinned_memory(8 * E * 4), np.float32, 8 * E) for _ in self.cards]
            for c in self.cards:
                self._pbufs(c)
                with cp.cuda.Device(c.dev):
                    c.ev_step = getattr(c, "ev_step", None) or cp.cuda.Event(disable_timing=True)
            c = self.cards[-1]
            with cp.cuda.Device(c.dev):
                c.LOG = cp.zeros((8, self.vocab), cp.float32)
                c.TOKS = cp.zeros(8, cp.int32)
            for c in self.cards:                 # the zeros above run on the null stream, which the cards' streams
                with cp.cuda.Device(c.dev):      # (non-blocking) do not wait for
                    cp.cuda.Device().synchronize()
        self.h_ck[:n] = toks
        self.h_ck[8:8 + n] = np.arange(pos0, pos0 + n, dtype=np.int32)
        for i, c in enumerate(self.cards):
            last = i == len(self.cards) - 1
            key = ("check", n, head)
            with cp.cuda.Device(c.dev):
                if key not in c.graph:
                    c.graph[key] = _capture(c.stream, lambda c=c, f=i == 0, h=last and head: self._check_card(c, n, f, h),
                                            replay=bool(c.host))
                if i == 0:
                    cp.cuda.runtime.memcpyAsync(c.TOK.data.ptr, self.h_ck.ctypes.data, n * 4, 1, c.stream.ptr)
                else:
                    c.stream.wait_event(self.cards[i - 1].ev_step)
                    cp.cuda.runtime.memcpyAsync(c.X.data.ptr, self.h_cx[i].ctypes.data, n * E * 4, 1, c.stream.ptr)
                cp.cuda.runtime.memcpyAsync(c.P.data.ptr, self.h_ck.ctypes.data + 32, n * 4, 1, c.stream.ptr)
                c.graph[key].launch(c.stream)
                if not last:
                    cp.cuda.runtime.memcpyAsync(self.h_cx[i + 1].ctypes.data, c.X.data.ptr, n * E * 4, 2, c.stream.ptr)
                    c.ev_step.record(c.stream)
        c = self.cards[-1]
        with cp.cuda.Device(c.dev):
            if head:
                cp.cuda.runtime.memcpyAsync(self.h_ck.ctypes.data, c.TOKS.data.ptr, n * 4, 2, c.stream.ptr)

        def done():
            with cp.cuda.Device(c.dev):
                c.stream.synchronize()
            return [int(x) for x in self.h_ck[:n]]
        return done() if wait else done

    def generate_spec(self, ids, n, draft, k=4, stats=None):
        """greedy output of this model, n tokens, with draft proposing k tokens per pass. Every token is this
        model's own greedy choice: a proposal is kept only where it equals the check pass's result."""
        t = self.prefill_batch(ids)
        draft.prefill_batch(ids)
        out, pos = [t], len(ids)
        passes, behind = 0, None
        while len(out) < n:
            kk = min(k, n - len(out))
            if kk < 1:
                break
            prop, dt = [], t
            for i in range(kk):
                if i == 0 and behind is not None:          # the draft has not seen the last proposal: feed both
                    dt = draft.check([behind, dt], pos - 1)[1]
                else:
                    dt = draft.step(dt, pos + i)
                prop.append(dt)
            behind = None
            got = self.check([t] + prop, pos) if kk >= 1 else []
            passes += 1
            j = 0
            while j < kk and prop[j] == got[j]:
                j += 1
            new = prop[:j] + [got[j]]
            if j == kk:                                  # all taken: the draft has not seen its last proposal yet
                behind = prop[-1]
            out += new
            pos += j + 1
            t = got[j]
        if stats is not None:
            stats["passes"] = passes
        return out[:n]

    # ------------------------------------------------------------------ MTP head (a draft source)
    def load_mtp(self, path):
        """load an MTP head file onto the last card"""
        cp = self.cp
        z = np.load(path)
        c, c0 = self.cards[-1], self.cards[0]
        E = self.n_embd
        assert int(z["E"]) == E, "the head is for another model"
        here = os.path.dirname(os.path.abspath(__file__))
        with cp.cuda.Device(c.dev):
            mod = module(cp, open(os.path.join(here, "nbq_mtp.cu"), encoding="utf-8").read())
            c.km = {n: mod.get_function(n) for n in ("mtp_in", "mtp_rms", "mtp_gemv", "mtp_swiglu")}
            c.mw = {n: cp.asarray(z[n]) for n in z.files if n != "E"}
            self.mtp_ffn = c.mw["gate"].shape[0]
            Fm = self.mtp_ffn
            c.m_e = cp.zeros(E, cp.float32)
            c.m_in = cp.zeros(2 * E, cp.float32)
            c.m_u = cp.zeros(E, cp.float32)
            c.m_v = cp.zeros(E, cp.float32)
            c.m_gu = cp.zeros(2 * Fm, cp.float32)
            c.m_f = cp.zeros(E, cp.float32)
            c.m_log = cp.zeros(self.vocab, cp.float32)
            c.m_tok = cp.zeros(4, cp.int32)
            c.ev_m = cp.cuda.Event(disable_timing=True)
        with cp.cuda.Device(c0.dev):
            c0.m_tok = cp.zeros(4, cp.int32)
            c0.m_e = cp.zeros(E, cp.float32)
            c0.ev_m = cp.cuda.Event(disable_timing=True)
        self.h_m = np.frombuffer(cp.cuda.alloc_pinned_memory(16 + E * 4), np.uint8, 16 + E * 4)
        for cc in self.cards:                 # the zeros above run on the null stream; the card streams do not wait
            with cp.cuda.Device(cc.dev):
                cp.cuda.Device().synchronize()
        self.mtp = True
        self.log("MTP head %s: ffn %d, %.0f MB on device %d" % (os.path.basename(path), Fm,
                 sum(a.nbytes for a in c.mw.values()) / 2 ** 20, c.dev))

    def mtp_draft(self, tok, row=None):
        """the head's guess of the token after tok, from the hidden state that gave tok: the last step's (row None),
        row `row` of the last check pass, or (row "chain") the head's own guess of it from the last mtp_draft"""
        cp, E, Fm = self.cp, self.n_embd, self.mtp_ffn
        c, c0 = self.cards[-1], self.cards[0]
        ht = self.h_m[:16].view(np.int32)
        he = self.h_m[16:].view(np.float32)
        ht[0] = tok
        with cp.cuda.Device(c0.dev), c0.stream:       # the embedding row is on the first card
            cp.cuda.runtime.memcpyAsync(c0.m_tok.data.ptr, ht.ctypes.data, 4, 1, c0.stream.ptr)
            c0.k["nbq_embed_t%d" % c0.w["embd"].ty](((E // 32 + 7) // 8, 1), (256,), (c0.w["embd"].body,
                                                    c0.w["embd"].d, c0.m_tok, np.int32(E), c0.m_e))
            if c is not c0:
                cp.cuda.runtime.memcpyAsync(he.ctypes.data, c0.m_e.data.ptr, E * 4, 2, c0.stream.ptr)
                c0.ev_m.record(c0.stream)
        with cp.cuda.Device(c.dev), c.stream:
            s, k, w = c.stream, c.km, c.mw
            if c is not c0:
                s.wait_event(c0.ev_m)
                cp.cuda.runtime.memcpyAsync(c.m_e.data.ptr, he.ctypes.data, E * 4, 1, s.ptr)
            h, hadd = (c.x, c.br) if row is None else (c.m_f, None) if row == "chain" else (c.X[row], None)
            eps = np.float32(self.eps)
            k["mtp_in"]((1,), (1024,), (c.m_e, h, hadd if hadd is not None else NULL, w["ne"], w["nh"],
                                        np.int32(E), eps, c.m_in))
            k["mtp_gemv"](((E + 7) // 8,), (256,), (w["w_in"], w["s_in"], c.m_in, np.int32(E), np.int32(2 * E), NULL,
                                                     c.m_u))
            k["mtp_rms"]((1,), (1024,), (c.m_u, w["nf"], np.int32(E), eps, c.m_v))
            k["mtp_gemv"](((Fm + 7) // 8,), (256,), (w["gate"], w["s_gate"], c.m_v, np.int32(Fm), np.int32(E), NULL,
                                                      c.m_gu))
            k["mtp_gemv"](((Fm + 7) // 8,), (256,), (w["up"], w["s_up"], c.m_v, np.int32(Fm), np.int32(E), NULL,
                                                      c.m_gu[Fm:]))
            k["mtp_swiglu"](((Fm + 255) // 256,), (256,), (c.m_gu, c.m_gu[Fm:], np.int32(Fm), c.m_gu))
            k["mtp_gemv"](((E + 7) // 8,), (256,), (w["down"], w["s_down"], c.m_gu, np.int32(E), np.int32(Fm), c.m_u,
                                                     c.m_f))
            c.k["nbq_add_rms"]((1,), (1024,), (c.m_f, NULL, c.w["out_norm"], np.int32(E), eps, NULL, c.xq, c.xs))
            self._mv(c, c.w["out"], c.m_log)
            c.k["nbq_argmax_part"]((256,), (256,), (c.m_log, np.int32(self.vocab), c.am_v, c.am_i))
            c.k["nbq_argmax_final"]((1,), (32,), (c.am_v, c.am_i, np.int32(256), c.m_tok, NULL))
            cp.cuda.runtime.memcpyAsync(ht.ctypes.data + 4, c.m_tok.data.ptr, 4, 2, s.ptr)
            s.synchronize()
        return int(ht[1])

    def logits(self):
        c = self.cards[-1]
        with self.cp.cuda.Device(c.dev):
            return self.cp.asnumpy(c.logits)

    def prefill(self, ids, pos=0):
        """feed ids from position pos; return the greedy token after the last"""
        for i, t in enumerate(ids[:-1]):
            self.step(t, pos + i, head=False)
        return self.step(ids[-1], pos + len(ids) - 1)

    def generate(self, ids, n, batch=True):
        t = self.prefill_batch(ids) if batch else self.prefill(ids)
        out = [t]
        pos = len(ids)
        while len(out) < n:
            t = self.step(t, pos)
            out.append(t)
            pos += 1
        return out
