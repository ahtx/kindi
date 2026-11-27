"""Kindi engine for Thinking Machines Inkling-2-Mini (GGUF architecture inkling2_loop).

Provides unified CPU and GPU/heterogeneous execution across:
  - cpu          : AVX2/F16C (or portable SSE) host CPU engine with persistent thread pool
  - hetero_k420  : 2 GB Kepler/small GPU + CPU concurrent split (Loop-Core experts + dense + partial lm_head)
  - hybrid_multi : Multi-GPU (e.g. 2x Quadro K1200) + CPU cold experts with zero host duplicate
  - gpu_full_multi: Multi-GPU full-VRAM split holding 100% of weights across GPUs
  - gpu_pinned   : Single 6-12 GB card + zero-copy pinned host cold experts + dynamic LRU DCACHE
  - gpu_full     : Single card holding the full model in VRAM

Uses zero top-level CuPy dependency so CPU-only and prefill/host-init run without initializing CUDA until needed.
"""
import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import time

import numpy as np

from .cores import physical_cores
from .gguf import GGUF
from .hostram import check as ram_check
from .moe import MoESpec, Usage, plan

IK_NL_PHYS = 24
IK_CORE_START = 6
IK_CORE_END = 18
IK_N_PASS = 3
IK_NL_VIRT = 48
IK_D = 2304
IK_QDIM = 3072
IK_GQADIM = 1024
IK_FF = 1152
IK_FF2 = 2304
IK_NE = 128
IK_NU = 8
IK_V_UNPAD = 200058
IK_MAXB = 128

TY_F32, TY_Q4_0, TY_Q8, TY_Q5K, TY_BF16, TY_NVFP4, TY_IQ2XXS_128, TY_IQ2XS_128 = 0, 2, 8, 13, 30, 40, 116, 117
_BLK_PARAMS = {
    TY_F32: (4, 1),
    TY_Q4_0: (18, 32),
    TY_Q8: (34, 32),
    TY_Q5K: (176, 256),
    TY_BF16: (2, 1),
    TY_NVFP4: (36, 64),
    TY_IQ2XXS_128: (34, 128),
    TY_IQ2XS_128: (38, 128),
}
GT_COUNT = 5
LT_COUNT = 42

HERE = os.path.dirname(os.path.abspath(__file__))

_GLOBAL_NAMES = (
    "token_embd.weight",
    "token_embd_norm.weight",
    "output.weight",
    "output_norm.weight",
    "output_res_query.weight",
)

_LAYER_SUFFIXES = (
    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
    "attn_gate.weight", "attn_rel.weight", "attn_sink_proj.weight", "attn_q_clip.weight",
    "attn_rel_bias.weight", "attn_rel_clip.weight", "attn_k_conv.weight", "attn_v_conv.weight",
    "attn_post_conv.weight", "ffn_post_conv.weight", "attn_norm.weight",
    "attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight",
    "attn_k_conv_proj.weight", "attn_v_conv_proj.weight", "attn_res_query.weight",
    "ffn_res_query.weight", "ffn_norm.weight", "ffn_gate_inp.weight", "exp_probs_b.bias",
    "ffn_gate_up_exps.weight", "ffn_gate_up_exps.scale",
    "ffn_down_exps.weight", "ffn_down_exps.scale",
    "loop_msg_query.weight",
    "loop_q_ctrl_down.weight", "loop_q_ctrl_gain.weight", "loop_q_ctrl_forget.weight",
    "loop_k_ctrl_down.weight", "loop_k_ctrl_gain.weight", "loop_k_ctrl_forget.weight",
    "loop_v_ctrl_down.weight", "loop_v_ctrl_gain.weight", "loop_v_ctrl_forget.weight",
    "loop_ffn_ctrl_down.weight", "loop_ffn_ctrl_gain.weight", "loop_ffn_ctrl_forget.weight",
)


class _IkGpuHooks(ctypes.Structure):
    _fields_ = [
        ("ctx", ctypes.c_void_p),
        ("forward_step", ctypes.c_void_p),
        ("qkvg_start", ctypes.c_void_p),
        ("qkvg_wait", ctypes.c_void_p),
        ("o_start", ctypes.c_void_p),
        ("o_wait", ctypes.c_void_p),
        ("moe_start", ctypes.c_void_p),
        ("moe_wait", ctypes.c_void_p),
        ("head_start", ctypes.c_void_p),
        ("head_wait", ctypes.c_void_p),
        ("reset", ctypes.c_void_p),
        ("on_roll", ctypes.c_void_p),
        ("split_q", ctypes.c_int),
        ("split_k", ctypes.c_int),
        ("split_v", ctypes.c_int),
        ("split_g", ctypes.c_int),
        ("split_o", ctypes.c_int),
        ("split_head", ctypes.c_int),
    ]


def _cc():
    """the C compiler: $CC when set, clang on macOS (Xcode command line tools), else gcc"""
    return os.environ.get("CC") or ("clang" if sys.platform == "darwin" else "gcc")


def _arch_flag():
    """-march=native on x86; -mcpu=native on arm64 (Apple silicon, AArch64 Linux), which turns on NEON dotprod/fp16"""
    m = platform.machine().lower()
    return "-mcpu=native" if m in ("arm64", "aarch64") else "-march=native"


def _cpu_has_i8mm():
    """the CPU has the AArch64 int8 matrix multiply (SMMLA): macOS sysctl, else /proc/cpuinfo"""
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_I8MM"], capture_output=True, text=True,
                                  timeout=5).stdout.strip() == "1"
        except Exception:  # noqa: BLE001
            return False
    try:
        with open("/proc/cpuinfo") as f:
            return any(line.startswith("Features") and " i8mm" in line for line in f)
    except OSError:
        return False


def _extra_flags(cc, arch):
    """arm64 with i8mm whose compiler does not turn it on for -mcpu=native (Apple clang 16 on M2-M4 does not):
    add the target feature, so the SMMLA prompt kernels are built. [] elsewhere."""
    if arch != "-mcpu=native" or os.environ.get("KINDI_NO_I8MM") == "1" or not _cpu_has_i8mm():
        return []
    try:
        macros = subprocess.run([cc, arch, "-dM", "-E", "-x", "c", os.devnull], capture_output=True, text=True,
                                timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    if "__ARM_FEATURE_MATMUL_INT8" in macros:
        return []
    if "clang" in cc or "__clang__" in macros:
        return ["-Xclang", "-target-feature", "-Xclang", "+i8mm"]
    return []


def _compile_cpu():
    src_path = os.path.join(HERE, "inkling2_cpu.c")
    tbl_path = os.path.join(HERE, "inkling2_iq2_tables.h")
    src = open(src_path, "rb").read()
    tbl = open(tbl_path, "rb").read() if os.path.exists(tbl_path) else b""
    cc, arch = _cc(), _arch_flag()
    extra = _extra_flags(cc, arch)
    h = hashlib.sha256(src + tbl + " ".join(extra).encode()).hexdigest()[:16]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache_dir, exist_ok=True)
    so_path = os.path.join(cache_dir, "inkling2_cpu_%s.so" % h)
    if not os.path.exists(so_path):
        tmp = so_path + ".tmp.%d" % os.getpid()
        cmd = [cc, "-O3", "-std=gnu11", arch] + extra + ["-Wall", "-Wextra",
               "-I" + HERE, "-shared", "-fPIC", src_path, "-o", tmp, "-pthread", "-lm", "-ldl"]
        subprocess.check_call(cmd)
        os.replace(tmp, so_path)
    lib = ctypes.CDLL(so_path)
    lib.ik_init.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
    ]
    lib.ik_init.restype = ctypes.c_int
    lib.ik_set_gpu_hooks.argtypes = [ctypes.POINTER(_IkGpuHooks)]
    lib.ik_set_gpu_hooks.restype = None
    lib.ik_reset.argtypes = []
    lib.ik_reset.restype = None
    lib.ik_state_bytes.argtypes = []
    lib.ik_state_bytes.restype = ctypes.c_size_t
    lib.ik_snapshot.argtypes = [ctypes.c_void_p]
    lib.ik_snapshot.restype = None
    lib.ik_restore.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ik_restore.restype = None
    lib.ik_get_profile.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double)]
    lib.ik_get_profile.restype = None
    lib.ik_reset_timers.argtypes = []
    lib.ik_reset_timers.restype = None
    lib.ik_max_batch.argtypes = []
    lib.ik_max_batch.restype = ctypes.c_int
    lib.ik_n_threads.argtypes = []
    lib.ik_n_threads.restype = ctypes.c_int
    lib.ik_conv_hist_ptr.argtypes = []
    lib.ik_conv_hist_ptr.restype = ctypes.c_void_p
    lib.ik_forward.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ik_forward.restype = ctypes.c_int
    lib.ik_forward_greedy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
    lib.ik_forward_greedy.restype = ctypes.c_int
    lib.ik_forward_greedy_batch.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
                                            ctypes.POINTER(ctypes.c_int)]
    lib.ik_forward_greedy_batch.restype = ctypes.c_int
    lib.ik_enable_conv_trace.argtypes = [ctypes.c_int]
    lib.ik_enable_conv_trace.restype = None
    lib.ik_rollback_conv.argtypes = [ctypes.c_int]
    lib.ik_rollback_conv.restype = None
    lib.ik_get_tok_embd_normed.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.ik_get_tok_embd_normed.restype = None
    lib.ik_head_argmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
    lib.ik_head_argmax.restype = ctypes.c_int
    lib.ik_head_argmax_val.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.ik_head_argmax_val.restype = ctypes.c_int
    lib.ik_set_dspark_targets.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.ik_set_dspark_targets.restype = None
    lib.ik_get_dspark_features.argtypes = []
    lib.ik_get_dspark_features.restype = ctypes.c_void_p
    lib.ik_get_tok_embd_ptr.argtypes = []
    lib.ik_get_tok_embd_ptr.restype = ctypes.c_void_p
    lib.ik_get_tok_norm_ptr.argtypes = []
    lib.ik_get_tok_norm_ptr.restype = ctypes.c_void_p
    lib.ik_get_output_head_ptr.argtypes = []
    lib.ik_get_output_head_ptr.restype = ctypes.c_void_p
    lib.ik_imatrix_start.argtypes = []
    lib.ik_imatrix_start.restype = None
    lib.ik_imatrix_stop_and_save.argtypes = [ctypes.c_char_p]
    lib.ik_imatrix_stop_and_save.restype = ctypes.c_int
    lib.ik_free.argtypes = []
    lib.ik_free.restype = None
    return lib


class Spec(object):
    """Inkling-2-Mini dimensions and tensor offset/type tables parsed from GGUF."""

    def __init__(self, g):
        kv = g.kv
        a = kv["general.architecture"]
        if a != "inkling2_loop":
            raise ValueError("expected inkling2_loop architecture, got %r" % a)
        self.n_layer = int(kv[a + ".block_count"])
        self.n_embd = int(kv[a + ".embedding_length"])
        self.ff = int(kv[a + ".expert_feed_forward_length"])
        self.n_exp = int(kv[a + ".expert_count"])
        self.n_used = int(kv[a + ".expert_used_count"])
        self.vocab_padded = int(kv[a + ".vocab_size"])
        self.vocab = int(kv[a + ".vocab_size_unpadded"])
        self.core_start = int(kv[a + ".loop.start"])
        self.core_end = int(kv[a + ".loop.end"])
        self.n_passes = int(kv[a + ".loop.count"])
        self.swa = int(kv[a + ".attention.sliding_window"])
        self.swa_pattern = [bool(x) for x in kv[a + ".attention.sliding_window_pattern"]]
        shape = (self.n_layer, self.n_embd, self.ff, self.n_exp, self.n_used, self.vocab,
                 self.core_start, self.core_end, self.n_passes, self.swa)
        expected = (IK_NL_PHYS, IK_D, IK_FF, IK_NE, IK_NU, IK_V_UNPAD,
                    IK_CORE_START, IK_CORE_END, IK_N_PASS, 1024)
        if shape != expected:
            raise ValueError("Inkling 2 Mini native kernels require dimensions %r, got %r" % (expected, shape))
        self.data_start = int(g.data_start)

        tmap = {nm: (ne, int(ty), int(off)) for nm, ne, ty, off in g.tensors}
        self.tmap = tmap
        self.g_offs = (ctypes.c_uint64 * GT_COUNT)()
        self.g_tys = (ctypes.c_int * GT_COUNT)()
        for i, nm in enumerate(_GLOBAL_NAMES):
            if nm in tmap:
                self.g_offs[i] = tmap[nm][2]
                self.g_tys[i] = tmap[nm][1]
            else:
                self.g_offs[i] = 0
                self.g_tys[i] = -1

        self.l_offs = (ctypes.c_uint64 * (self.n_layer * LT_COUNT))()
        self.l_tys = (ctypes.c_int * (self.n_layer * LT_COUNT))()
        for ip in range(self.n_layer):
            for j, suf in enumerate(_LAYER_SUFFIXES):
                nm = "blk.%d.%s" % (ip, suf)
                idx = ip * LT_COUNT + j
                if nm in tmap:
                    self.l_offs[idx] = tmap[nm][2]
                    self.l_tys[idx] = tmap[nm][1]
                else:
                    self.l_offs[idx] = 0
                    self.l_tys[idx] = -1

        self.expert_bytes = {}
        for ip in range(self.n_layer):
            gu_ty = int(self.l_tys[ip * LT_COUNT + 25])
            dn_ty = int(self.l_tys[ip * LT_COUNT + 27])
            g_bpe, g_blk = _BLK_PARAMS[gu_ty]
            d_bpe, d_blk = _BLK_PARAMS[dn_ty]
            gu_b = self.ff * 2 * (self.n_embd // g_blk) * g_bpe
            dn_b = self.n_embd * (self.ff // d_blk) * d_bpe
            self.expert_bytes[ip] = gu_b + dn_b

    def vram_weight_bytes(self):
        """Return exact GPU-resident model weight bytes (all tensors except CPU token_embd)."""
        tot = 0
        for nm, (ne, ty, _) in self.tmap.items():
            if nm in ("token_embd.weight", "token_embd_norm.weight"):
                continue
            elems = 1
            for d in ne:
                elems *= int(d)
            bpe, blk = _BLK_PARAMS[int(ty)]
            tot += (elems // blk) * bpe
        return tot


def _probe_gpus():
    """Return list of (index, name, cc_int, free_mb, total_mb) without top-level cupy import."""
    if os.environ.get("KINDI_DEVICE", "gpu").lower() == "cpu":
        return []
    if os.environ.get("KINDI_INKLING2_MODE", "").strip().lower() == "cpu":
        return []
    try:
        import cupy as cp
        out = []
        cur = cp.cuda.runtime.getDevice()
        try:
            n = cp.cuda.runtime.getDeviceCount()
            for i in range(n):
                cp.cuda.Device(i).use()
                p = cp.cuda.runtime.getDeviceProperties(i)
                name = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
                cc = int(p["major"]) * 10 + int(p["minor"])
                fr, tot = cp.cuda.runtime.memGetInfo()
                out.append((i, name, cc, fr >> 20, tot >> 20))
        finally:
            cp.cuda.Device(cur).use()
        return out
    except Exception:
        return []


def _choose_mode(gpus, vram_mb=None):
    forced = os.environ.get("KINDI_INKLING2_MODE", "").strip().lower()
    if forced:
        return forced
    if not gpus:
        return "cpu"
    frees = sorted((g[3] for g in gpus), reverse=True)
    tots = sorted((g[4] for g in gpus), reverse=True)
    if vram_mb is not None:
        if frees[0] >= vram_mb + 450:
            return "gpu_full"
        if len(gpus) >= 2 and frees[0] >= vram_mb / 2.0 + 260 and frees[1] >= vram_mb / 2.0 + 260:
            return "gpu_full_multi"
    if tots[0] >= 18000:
        return "gpu_full"
    if len(gpus) >= 2 and tots[0] >= 3500 and tots[1] >= 3500 and (tots[0] + tots[1]) >= 6500:
        return "hybrid_multi"
    if tots[0] >= 6000:
        return "gpu_pinned"
    if tots[0] >= 1400:
        return "hetero_k420"
    return "cpu"


def _load_usage(spec):
    """Load inkling2-experts.json if present, else return loop-weighted synthetic usage prioritizing core [6..17]."""
    upath = os.path.join(HERE, "inkling2-experts.json")
    if os.path.exists(upath):
        return Usage.load(MoESpec(range(spec.n_layer), spec.n_exp, spec.n_used, spec.expert_bytes), upath)
    ms = MoESpec(range(spec.n_layer), spec.n_exp, spec.n_used, spec.expert_bytes)
    # Without profiled counts, prioritize Loop Core [6..17] (3x picks/tok vs 1x outside core)
    counts = {}
    for ip in range(spec.n_layer):
        w = 300 if IK_CORE_START <= ip < IK_CORE_END else 100
        counts[ip] = [w] * spec.n_exp
    return Usage(ms, counts=counts)


class Engine(object):
    def __init__(self, model_path, ctx=4096, device=None, log=print):
        self.model_path = os.path.abspath(model_path)
        self.ctx = max(512, int(ctx))
        self.log = log
        t0 = time.time()
        self.gguf = GGUF(self.model_path)
        self.spec = Spec(self.gguf)
        self.gpus = _probe_gpus()
        vram_mb = self.spec.vram_weight_bytes() / float(1 << 20)
        self.mode = _choose_mode(self.gpus, vram_mb)
        self.device = 0 if device is None else int(device)

        env_th = os.environ.get("KINDI_THREADS") or os.environ.get("OMP_NUM_THREADS")
        n_threads = int(env_th) if env_th else physical_cores()   # one thread per physical core (kindi.cores)
        self.n_threads = n_threads

        self.lib = _compile_cpu()
        self.on_gpu = np.zeros((IK_NL_PHYS, IK_NE), dtype=np.uint8)
        self.gpu_ctx = None
        zero_copy_gpu_mem = 0

        # Placement policy when GPUs are active (GPU hooks attached below once inkling2_gpu is ready)
        if self.mode in ("gpu_full", "gpu_full_multi"):
            self.on_gpu.fill(2)
            zero_copy_gpu_mem = 1
            host_need = 2 << 30
        elif self.mode == "gpu_pinned":
            # GPU drives full forward with zero-copy pinned cold experts
            zero_copy_gpu_mem = 1
            host_need = 10 << 30
        elif self.mode == "hybrid_multi":
            zero_copy_gpu_mem = 1
            host_need = 2 << 30
        elif self.mode == "hetero_k420":
            host_need = 4 << 30
        else:
            host_need = 4 << 30

        # If GPU runtime module is unavailable in current env (e.g. non-GPU test), fall back cleanly to CPU
        gpu_mod = None
        if self.mode != "cpu":
            try:
                from . import inkling2_gpu as _ik_gpu
                gpu_mod = _ik_gpu
            except ImportError:
                self.mode = "cpu"
                self.on_gpu.fill(0)
                zero_copy_gpu_mem = 0
                host_need = 4 << 30

        ram_check(host_need, "Inkling-2-Mini (%s)" % self.mode, log=log)

        rc = self.lib.ik_init(
            self.model_path.encode("utf-8"),
            ctypes.c_size_t(self.spec.data_start),
            self.spec.g_offs, self.spec.g_tys,
            self.spec.l_offs, self.spec.l_tys,
            ctypes.c_int(self.ctx),
            self.on_gpu.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(zero_copy_gpu_mem),
            ctypes.c_int(self.n_threads),
        )
        if rc != 0:
            raise RuntimeError("ik_init failed (%d)" % rc)

        if gpu_mod is not None:
            usage = _load_usage(self.spec)
            self.gpu_ctx = gpu_mod.attach(self, usage, log=log)

        self.pos = 0
        # tokens per forward call: the C build's IK_MAXB (512 on AArch64) on the CPU path; 128, what the GPU paths
        # are sized for, whenever a GPU takes part
        self.maxb = int(self.lib.ik_max_batch()) if self.gpu_ctx is None else IK_MAXB
        self._state_sz = int(self.lib.ik_state_bytes())
        self.dspark = None
        self.dspark_path = None
        self._dspark_feat_buf = None
        self._embd_buf = (ctypes.c_float * IK_D)()
        self._is_full_gpu = (self.gpu_ctx is not None and self.mode not in ("cpu", "hetero_k420"))
        self.log("loaded Inkling-2-Mini (%s, %d CPU threads, ctx %d) in %.1f s" % (
            self.mode, self.lib.ik_n_threads(), self.ctx, time.time() - t0))

    def attach_dspark(self, gguf_path, threads=None):
        """Attach a CPU DSpark drafter (Q8_0, zero VRAM) to accelerate greedy decoding."""
        from .inkling2_dspark import DSparkDrafter
        if self.dspark is not None:
            try:
                self.dspark.close()
            except Exception:
                pass
            self.dspark = None
        thr = int(threads or self.n_threads)
        self.dspark = DSparkDrafter(gguf_path, max_ctx=max(self.ctx + 16, 4096), threads=thr)
        self.dspark_path = os.path.abspath(gguf_path)
        self._dspark_feat_buf = np.empty((IK_MAXB, 4, IK_D), dtype=np.float32)
        self.log("attached DSpark drafter %s (%d CPU threads, layers=%s, block=%d)" % (
            os.path.basename(self.dspark_path), thr, self.dspark.target_layers, self.dspark.block_size))
        return self.dspark

    def reset(self):
        self.lib.ik_reset()
        self.pos = 0

    def snapshot(self):
        """the state at the current position that the positions alone do not give back (the short-conv history and
        the sliding-window KV rings); restore() goes back to it after later tokens. On a card that runs the whole
        forward pass the copy is kept by the GPU library (one copy: a new snapshot replaces it)."""
        if self._is_full_gpu:
            if self.gpu_ctx.lib.ik_gpu_save_state(self.gpu_ctx.ctx_ptr) != 0:
                raise RuntimeError("Inkling-2: the state copy failed")
            return None, self.pos
        buf = (ctypes.c_uint8 * self._state_sz)()
        self.lib.ik_snapshot(buf)
        return bytes(buf), self.pos

    def restore(self, snap):
        state, pos = snap
        if self._is_full_gpu:
            if self.gpu_ctx.lib.ik_gpu_load_state(self.gpu_ctx.ctx_ptr) != 0:
                raise RuntimeError("Inkling-2: no state copy to restore")
        else:
            buf = ctypes.create_string_buffer(state, self._state_sz)
            self.lib.ik_restore(buf, ctypes.c_int(pos))
        self.pos = int(pos)

    def prefill(self, ids, pos=None, return_gap=False):
        if pos is not None and pos != self.pos:
            self.pos = int(pos)
        n = len(ids)
        assert self.pos + n <= self.ctx, "prompt exceeds ctx %d" % self.ctx
        out_tok = ctypes.c_int()
        out_gap = ctypes.c_float()
        c0 = 0
        while c0 < n:
            part = ids[c0:c0 + self.maxb]
            T = len(part)
            arr = (ctypes.c_int * T)(*[int(x) for x in part])
            last = (c0 + T == n)
            if last:
                rc = self.lib.ik_forward_greedy(arr, T, self.pos, ctypes.byref(out_tok), ctypes.byref(out_gap))
            else:
                rc = self.lib.ik_forward(arr, T, self.pos, None, 0)
            if rc != 0:
                raise RuntimeError("ik_forward failed (%d)" % rc)
            self.pos += T
            c0 += T
        if return_gap:
            return int(out_tok.value), float(out_gap.value)
        return int(out_tok.value)

    def step(self, tok, return_gap=False):
        assert self.pos < self.ctx, "context full (%d)" % self.ctx
        arr = (ctypes.c_int * 1)(int(tok))
        out_tok = ctypes.c_int()
        out_gap = ctypes.c_float()
        rc = self.lib.ik_forward_greedy(arr, 1, self.pos, ctypes.byref(out_tok), ctypes.byref(out_gap))
        if rc != 0:
            raise RuntimeError("ik_forward_greedy failed (%d)" % rc)
        self.pos += 1
        if return_gap:
            return int(out_tok.value), float(out_gap.value)
        return int(out_tok.value)

    def forward_logits(self, ids, pos=None):
        """Run forward on ids[0..T-1] returning float32 numpy array [T, vocab]."""
        if pos is not None:
            self.pos = int(pos)
        T = len(ids)
        out = np.empty((T, IK_V_UNPAD), dtype=np.float32)
        c0 = 0
        while c0 < T:
            part = ids[c0:c0 + self.maxb]
            tlen = len(part)
            arr = (ctypes.c_int * tlen)(*[int(x) for x in part])
            lptr = out[c0:c0 + tlen].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rc = self.lib.ik_forward(arr, tlen, self.pos, lptr, 1)
            if rc != 0:
                raise RuntimeError("ik_forward logits failed (%d)" % rc)
            self.pos += tlen
            c0 += tlen
        return out

    def generate(self, ids, max_new, stop=(), return_gaps=False):
        self.reset()
        nxt, gap = self.prefill(ids, pos=0, return_gap=True)
        out, gaps = [], []
        stop_set = set(stop)
        for _ in range(max_new):
            out.append(nxt)
            gaps.append(gap)
            if nxt in stop_set or len(out) >= max_new or self.pos >= self.ctx:
                break
            nxt, gap = self.step(nxt, return_gap=True)
        return (out, gaps) if return_gaps else out

    def generate_dspark(self, ids, max_new, stop=(), return_stats=False):
        """Experimental greedy speculative generation using the DSpark drafter."""
        empty_stats = {"passes": 0, "drafted": 0, "accepted": 0, "mean_step": 0.0, "accept_rate": 0.0}
        if self.dspark is None:
            out = self.generate(ids, max_new, stop=stop)
            return (out, empty_stats) if return_stats else out
        if max_new <= 0:
            return ([], empty_stats) if return_stats else []

        from .inkling2_dspark import DS_HEAD_ARGMAX_CB
        self.reset()
        self.dspark.reset()
        stop_set = set(stop)
        n = len(ids)
        assert n <= self.ctx, "prompt exceeds ctx %d" % self.ctx

        tids = (ctypes.c_int * 4)(*[int(x) for x in self.dspark.target_layers])
        if self._is_full_gpu:
            g_lib = self.gpu_ctx.lib
            g_ctx = self.gpu_ctx.ctx_ptr
            if self._dspark_feat_buf is None:
                self._dspark_feat_buf = np.empty((IK_MAXB, 4, IK_D), dtype=np.float32)
            feat_ptr = int(self._dspark_feat_buf.ctypes.data)
            g_lib.ik_gpu_set_dspark_targets(g_ctx, 4, tids, ctypes.c_void_p(feat_ptr))
            head_cb = ctypes.cast(g_lib.ik_gpu_head_argmax_val, DS_HEAD_ARGMAX_CB)
            head_user = g_ctx
        else:
            self.lib.ik_set_dspark_targets(4, tids)
            feat_ptr = int(self.lib.ik_get_dspark_features())
            head_cb = ctypes.cast(self.lib.ik_head_argmax_val, DS_HEAD_ARGMAX_CB)
            head_user = 0

        out = []
        passes = 0
        drafted_tot = 0
        accepted_tot = 0
        emitted_tot = 0

        try:
            out_tok = ctypes.c_int()
            out_gap = ctypes.c_float()
            c0 = 0
            while c0 < n:
                part = ids[c0:c0 + IK_MAXB]
                T = len(part)
                p0 = self.pos
                arr = (ctypes.c_int * T)(*[int(x) for x in part])
                if c0 + T == n:
                    rc = self.lib.ik_forward_greedy(arr, T, p0, ctypes.byref(out_tok), ctypes.byref(out_gap))
                else:
                    rc = self.lib.ik_forward(arr, T, p0, None, 0)
                if rc != 0:
                    raise RuntimeError("dspark prefill failed (%d)" % rc)
                self.dspark.inject(p0, T, feat_ptr)
                self.pos += T
                c0 += T

            anchor = int(out_tok.value)
            out.append(anchor)
            if anchor not in stop_set and len(out) < max_new and self.pos < self.ctx:
                if self._is_full_gpu:
                    g_lib.ik_gpu_enable_conv_trace(g_ctx, 1)
                else:
                    self.lib.ik_enable_conv_trace(1)

                bsz = int(self.dspark.block_size)
                in_arr = (ctypes.c_int * bsz)()
                out_arr = (ctypes.c_int * bsz)()

                while len(out) < max_new and self.pos < self.ctx:
                    remain = max_new - len(out)
                    ctx_room = self.ctx - self.pos
                    if remain == 1 or ctx_room < 2:
                        nxt = self.step(anchor)
                        out.append(nxt)
                        break

                    n_draft = min(bsz - 1, ctx_room - 1, remain - 1)
                    self.lib.ik_get_tok_embd_normed(int(anchor), self._embd_buf)
                    drafts = self.dspark.draft(
                        anchor, self.pos, n_draft, ctypes.addressof(self._embd_buf), head_cb, head_user,
                    )

                    T = 1 + n_draft
                    in_arr[0] = int(anchor)
                    for i in range(n_draft):
                        in_arr[i + 1] = int(drafts[i])

                    pos0 = self.pos
                    if self._is_full_gpu:
                        rc = g_lib.ik_gpu_forward_greedy_batch(g_ctx, T, pos0, in_arr, out_arr)
                    else:
                        rc = self.lib.ik_forward_greedy_batch(in_arr, T, pos0, out_arr)
                    if rc != 0:
                        raise RuntimeError("dspark verify forward failed (%d)" % rc)

                    m = n_draft
                    for i in range(n_draft):
                        if int(out_arr[i]) != int(drafts[i]):
                            m = i
                            break
                    k_acc = m + 1

                    if self._is_full_gpu:
                        g_lib.ik_gpu_rollback_conv(g_ctx, k_acc)
                    else:
                        self.lib.ik_rollback_conv(k_acc)

                    self.dspark.inject(pos0, k_acc, feat_ptr)
                    self.pos = pos0 + k_acc
                    anchor = int(out_arr[m])

                    passes += 1
                    drafted_tot += n_draft
                    accepted_tot += m
                    emitted_tot += k_acc

                    hit_stop = False
                    for i in range(k_acc):
                        tk = int(out_arr[i])
                        out.append(tk)
                        if tk in stop_set or len(out) >= max_new:
                            hit_stop = True
                            break
                    if hit_stop:
                        break
        finally:
            if self._is_full_gpu:
                self.gpu_ctx.lib.ik_gpu_enable_conv_trace(self.gpu_ctx.ctx_ptr, 0)
                self.gpu_ctx.lib.ik_gpu_set_dspark_targets(self.gpu_ctx.ctx_ptr, 0, None, None)
            else:
                self.lib.ik_enable_conv_trace(0)
                self.lib.ik_set_dspark_targets(0, None)

        stats = {
            "passes": passes,
            "drafted": drafted_tot,
            "accepted": accepted_tot,
            "mean_step": (emitted_tot / float(passes)) if passes else 0.0,
            "accept_rate": (accepted_tot / float(drafted_tot)) if drafted_tot else 0.0,
        }
        return (out, stats) if return_stats else out

    def profile(self, reset=False):
        cnt = np.zeros((IK_NL_PHYS, IK_NE), dtype=np.uint64)
        tm = (ctypes.c_double * 6)()
        self.lib.ik_get_profile(cnt.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), tm)
        d = {"emb": tm[0], "attn": tm[1], "dense": tm[2], "moe": tm[3], "head": tm[4], "total": tm[5]}
        if reset:
            self.lib.ik_reset_timers()
        return cnt, d

    def save_usage(self, out_path):
        cnt, _ = self.profile()
        ms = MoESpec(range(self.spec.n_layer), self.spec.n_exp, self.spec.n_used, self.spec.expert_bytes)
        counts = {ip: [int(x) for x in cnt[ip]] for ip in range(self.spec.n_layer)}
        u = Usage(ms, counts=counts)
        u.save(out_path)
        return u

    def close(self):
        if getattr(self, "dspark", None) is not None:
            try:
                self.dspark.close()
            except Exception:
                pass
            self.dspark = None
        if self.gpu_ctx is not None:
            try:
                self.gpu_ctx.close()
            except Exception:
                pass
            self.gpu_ctx = None
        if self.lib is not None:
            self.lib.ik_free()
            self.lib = None
