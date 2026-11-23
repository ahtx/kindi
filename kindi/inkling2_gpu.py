"""GPU / Hetero planner & launcher for Kindi Inkling-2-Mini (inkling2_loop).

Compiles inkling2_gpu.cu cleanly inside containers without g++ (cc1plus):
  1. Device section (#ifdef __CUDACC__) -> PTX via libnvrtc (cupy.cuda.nvrtc), cached in ~/.cache/kindi/
  2. Host section (#else) -> shared library via gcc -x c -O3 -std=gnu11 -shared -fPIC -lcudart -ldl
Plans expert & layer placement across hetero_k420 / hybrid_multi / gpu_pinned / gpu_full,
and binds C hooks directly into libinkling2_cpu so inference has zero Python overhead per token or layer.
"""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

from .inkling2 import (
    IK_CORE_END,
    IK_CORE_START,
    IK_D,
    IK_GQADIM,
    IK_NE,
    IK_NL_PHYS,
    IK_QDIM,
    IK_V_UNPAD,
    TY_NVFP4,
    TY_Q4_0,
    _IkGpuHooks,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_libnvrtc():
    for p in (
        "/usr/local/cuda/lib64/libnvrtc.so",
        "/usr/local/cuda-10.2/lib64/libnvrtc.so",
        "/usr/local/cuda/lib64/libnvrtc.so.10.2",
        "libnvrtc.so",
    ):
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    raise RuntimeError("libnvrtc.so not found for Inkling2 GPU compilation")


def _compile_ptx_for_cc(src_bytes, src_hash, tbl_bytes, cc, cache_dir):
    eff = max(30, min(75, int(cc)))
    ptx_path = os.path.join(cache_dir, "inkling2_gpu_compute%d_%s.ptx" % (eff, src_hash))
    if os.path.exists(ptx_path):
        return open(ptx_path, "rb").read()

    nvrtc = _find_libnvrtc()
    prog = ctypes.c_void_p()
    c_hdrs = (ctypes.c_char_p * 1)(tbl_bytes)
    c_names = (ctypes.c_char_p * 1)(b"inkling2_iq2_tables.h")
    rc = nvrtc.nvrtcCreateProgram(
        ctypes.byref(prog),
        ctypes.c_char_p(src_bytes),
        ctypes.c_char_p(b"inkling2_gpu.cu"),
        ctypes.c_int(1), c_hdrs, c_names,
    )
    if rc != 0:
        raise RuntimeError("nvrtcCreateProgram failed (%d)" % rc)
    try:
        opts = [
            ("--gpu-architecture=compute_%d" % eff).encode("utf-8"),
            b"-D__CUDACC__=1",
        ]
        c_opts = (ctypes.c_char_p * len(opts))(*opts)
        rc = nvrtc.nvrtcCompileProgram(prog, ctypes.c_int(len(opts)), c_opts)
        if rc != 0:
            log_sz = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            nvrtc.nvrtcGetProgramLog(prog, log_buf)
            raise RuntimeError("NVRTC compile failed for compute_%d (%d):\n%s" % (
                eff, rc, log_buf.value.decode("utf-8", "replace")))
        ptx_sz = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_sz))
        ptx_buf = ctypes.create_string_buffer(ptx_sz.value)
        nvrtc.nvrtcGetPTX(prog, ptx_buf)
        ptx_bytes = ptx_buf.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    tmp = ptx_path + ".tmp.%d" % os.getpid()
    with open(tmp, "wb") as f:
        f.write(ptx_bytes)
    os.replace(tmp, ptx_path)
    return ptx_bytes


def _compile_gpu(ccs):
    src_path = os.path.join(HERE, "inkling2_gpu.cu")
    tbl_path = os.path.join(HERE, "inkling2_iq2_tables.h")
    src = open(src_path, "rb").read()
    tbl = open(tbl_path, "rb").read() if os.path.exists(tbl_path) else b""
    h = hashlib.sha256(src + tbl).hexdigest()[:16]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache_dir, exist_ok=True)

    so_path = os.path.join(cache_dir, "inkling2_gpu_host_%s.so" % h)
    if not os.path.exists(so_path):
        tmp = so_path + ".tmp.%d.so" % os.getpid()
        cuda_inc = "/usr/local/cuda/include"
        cuda_lib = "/usr/local/cuda/lib64"
        if not os.path.exists(cuda_inc) and os.path.exists("/usr/local/cuda-10.2/include"):
            cuda_inc = "/usr/local/cuda-10.2/include"
            cuda_lib = "/usr/local/cuda-10.2/lib64"
        cmd = [
            "gcc", "-x", "c", "-O3", "-std=gnu11",
            "-Wall", "-Wextra",
            "-shared", "-fPIC",
            "-I" + HERE,
            "-I" + cuda_inc,
            src_path,
            "-o", tmp,
            "-L" + cuda_lib,
            "-Wl,-rpath," + cuda_lib,
            "-lcudart", "-ldl", "-lm",
        ]
        subprocess.check_call(cmd)
        os.replace(tmp, so_path)

    ptx_map = {}
    for c in ccs:
        ptx_map[int(c)] = _compile_ptx_for_cc(src, h, tbl, int(c), cache_dir)

    lib = ctypes.CDLL(so_path)
    lib.ik_gpu_init.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(_IkGpuHooks),
    ]
    lib.ik_gpu_init.restype = ctypes.c_void_p
    lib.ik_gpu_free.argtypes = [ctypes.c_void_p]
    lib.ik_gpu_free.restype = None
    lib.ik_gpu_set_dspark_targets.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
    ]
    lib.ik_gpu_set_dspark_targets.restype = None
    lib.ik_gpu_head_argmax.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ]
    lib.ik_gpu_head_argmax.restype = ctypes.c_int
    lib.ik_gpu_head_argmax_val.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.ik_gpu_head_argmax_val.restype = ctypes.c_int
    lib.ik_gpu_enable_conv_trace.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ik_gpu_enable_conv_trace.restype = None
    lib.ik_gpu_rollback_conv.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ik_gpu_rollback_conv.restype = None
    lib.ik_gpu_save_state.argtypes = [ctypes.c_void_p]
    lib.ik_gpu_save_state.restype = ctypes.c_int
    lib.ik_gpu_load_state.argtypes = [ctypes.c_void_p]
    lib.ik_gpu_load_state.restype = ctypes.c_int
    lib.ik_gpu_forward_greedy_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.ik_gpu_forward_greedy_batch.restype = ctypes.c_int
    return lib, ptx_map


class _GpuHandle(object):
    def __init__(self, lib, ctx_ptr, engine):
        self.lib = lib
        self.ctx_ptr = ctx_ptr
        self.engine = engine

    def close(self):
        if self.engine is not None and self.engine.lib is not None:
            empty = _IkGpuHooks()
            self.engine.lib.ik_set_gpu_hooks(ctypes.byref(empty))
        if self.ctx_ptr is not None and self.lib is not None:
            self.lib.ik_gpu_free(self.ctx_ptr)
            self.ctx_ptr = None


def _rank_experts(spec, usage, layers):
    """Return list of (density, ip, e, eb) sorted highest value/byte first."""
    items = []
    eb_map = getattr(spec, "expert_bytes", None)
    cnt_map = getattr(usage, "counts", None) if usage is not None else None
    for ip in layers:
        eb = int(eb_map[ip]) if isinstance(eb_map, dict) else int(eb_map or 4478976)
        cnts = cnt_map[ip] if cnt_map is not None else None
        for e in range(IK_NE):
            # Profiled counts already accumulate 3 passes for Loop Core [6..17]; fallback usage weights 300 vs 100
            score = float(cnts[e]) + 0.1 if cnts is not None else (300.0 if IK_CORE_START <= ip < IK_CORE_END else 100.0)
            items.append((score / float(eb), ip, e, eb))
    items.sort(reverse=True)
    return items


def _align16_split(full_rows, gpu_frac):
    gpu_rows = (int(full_rows * gpu_frac) // 16) * 16
    if gpu_rows < 0:
        gpu_rows = 0
    if gpu_rows > full_rows:
        gpu_rows = full_rows
    return full_rows - gpu_rows


def attach(engine, usage, log=print):
    gpus = engine.gpus
    if not gpus:
        return None
    mode_name = engine.mode
    spec = engine.spec

    kv_total_mb = (40 * (1024 + 128) * 4096 + 8 * engine.ctx * 4096) >> 20
    exp_plan = np.zeros((IK_NL_PHYS, IK_NE), dtype=np.uint8)
    n_dcache = 0
    split_q = IK_QDIM
    split_k = IK_GQADIM
    split_v = IK_GQADIM
    split_g = IK_QDIM
    split_o = IK_D
    split_head = IK_V_UNPAD
    is_q4_head = (int(spec.g_tys[2]) == TY_Q4_0)
    is_q4_dense = (int(spec.l_tys[0]) == TY_Q4_0)
    ctrl_save = 80 if (not is_q4_dense and int(spec.l_tys[6 * 42 + 30]) == 8) else 0
    dense_total_mb = 410 if is_q4_dense else (835 - ctrl_save)
    head_total_mb = 250 if is_q4_head else 495

    if mode_name == "hetero_k420":
        dev_list = [engine.device]
        ccs = [g[2] for g in gpus if g[0] == engine.device] or [gpus[0][2]]
        free_mb = next((g[3] for g in gpus if g[0] == engine.device), gpus[0][3])
        c_mode = 0
        layer_dev = [0] * IK_NL_PHYS
        split_frac = float(os.environ.get("KINDI_K420_SPLIT", "0.10"))
        split_q = _align16_split(IK_QDIM, split_frac)
        split_k = _align16_split(IK_GQADIM, split_frac)
        split_v = _align16_split(IK_GQADIM, split_frac)
        split_g = _align16_split(IK_QDIM, split_frac)
        split_o = _align16_split(IK_D, split_frac)
        head_gpu_rows = (int(IK_V_UNPAD * split_frac) // 16) * 16
        split_head = IK_V_UNPAD - head_gpu_rows

        dense_core_mb = int(round(12 * (14.5 if is_q4_dense else 27.6) * split_frac))
        head_mb = int(round(float(head_total_mb) * (head_gpu_rows / float(IK_V_UNPAD))))
        reserve_mb = 180
        exp_budget_bytes = max(0, free_mb - dense_core_mb - head_mb - reserve_mb) * (1 << 20)
        max_exps = int(os.environ.get("KINDI_K420_MAX_EXPS", "64"))
        n_placed = 0
        for _, ip, e, eb in _rank_experts(spec, usage, range(IK_CORE_START, IK_CORE_END)):
            if max_exps > 0 and n_placed >= max_exps:
                break
            if exp_budget_bytes >= eb:
                exp_plan[ip, e] = 2
                exp_budget_bytes -= eb
                n_placed += 1
        engine.on_gpu.fill(0)
        n_stat = int((exp_plan == 2).sum())
        log("inkling2 hetero_k420: GPU%d (cc%d, %d MiB free), split=%.2f, head_gpu=%d, %d Loop-Core experts in VRAM" % (
            dev_list[0], ccs[0], free_mb, split_frac, head_gpu_rows, n_stat))

    elif mode_name == "gpu_full_multi":
        sel_gpus = sorted(gpus, key=lambda x: x[3], reverse=True)[:2]
        dev_list = [g[0] for g in sel_gpus]
        ccs = [g[2] for g in sel_gpus]
        c_mode = 1
        layer_dev = [0] * 12 + [1] * 12
        exp_plan.fill(2)
        engine.on_gpu.fill(2)
        log("inkling2 gpu_full_multi: GPUs %s (cc%s, free %s MiB), all %d experts in VRAM (0 swap)" % (
            dev_list, ccs, [g[3] for g in sel_gpus], IK_NL_PHYS * IK_NE))

    elif mode_name == "hybrid_multi":
        sel_gpus = sorted(gpus, key=lambda x: x[3], reverse=True)[:2]
        dev_list = [g[0] for g in sel_gpus]
        ccs = [g[2] for g in sel_gpus]
        c_mode = 1
        layer_dev = [0] * 12 + [1] * 12
        fixed_mb = (dense_total_mb // 2) + (head_total_mb // 2) + (kv_total_mb // 2) + 150
        reserve_mb = 260
        for d_idx, ginfo in enumerate(sel_gpus):
            free_mb = ginfo[3]
            budget = max(0, free_mb - fixed_mb - reserve_mb) * (1 << 20)
            lays = range(0, 12) if d_idx == 0 else range(12, 24)
            for _, ip, e, eb in _rank_experts(spec, usage, lays):
                if budget >= eb:
                    exp_plan[ip, e] = 2
                    engine.on_gpu[ip, e] = 2
                    budget -= eb
        log("inkling2 hybrid_multi: GPUs %s (cc%s), %d/%d experts in VRAM (0 swap target)" % (
            dev_list, ccs, int((exp_plan == 2).sum()), IK_NL_PHYS * IK_NE))

    elif mode_name == "gpu_pinned":
        ginfo = max(gpus, key=lambda x: x[3])
        dev_list = [ginfo[0]]
        ccs = [ginfo[2]]
        c_mode = 1
        layer_dev = [0] * IK_NL_PHYS
        supports_dc = int(spec.l_tys[25]) in (TY_NVFP4, 13)
        default_dc = ("128" if ginfo[3] >= 7000 else "105") if supports_dc else "0"
        n_dcache = int(os.environ.get("KINDI_INKLING2_DCACHE", default_dc)) if supports_dc else 0
        dcache_mb = int(round(n_dcache * 4.35))
        fixed_mb = dense_total_mb + head_total_mb + kv_total_mb + 180 + dcache_mb
        reserve_mb = 800
        budget = max(0, ginfo[3] - fixed_mb - reserve_mb) * (1 << 20)
        for _, ip, e, eb in _rank_experts(spec, usage, range(IK_NL_PHYS)):
            if budget >= eb:
                exp_plan[ip, e] = 2
                budget -= eb
        for ip in range(IK_NL_PHYS):
            for e in range(IK_NE):
                if exp_plan[ip, e] == 0:
                    exp_plan[ip, e] = 1
        engine.on_gpu.fill(2)
        log("inkling2 gpu_pinned: GPU%d (%s, cc%d, %d MiB free), %d static + %d DCACHE (%d MiB) + pinned cold" % (
            dev_list[0], ginfo[1], ccs[0], ginfo[3], int((exp_plan == 2).sum()), n_dcache, dcache_mb))

    elif mode_name == "gpu_full":
        ginfo = max(gpus, key=lambda x: x[4])
        dev_list = [ginfo[0]]
        ccs = [ginfo[2]]
        c_mode = 1
        layer_dev = [0] * IK_NL_PHYS
        exp_plan.fill(2)
        engine.on_gpu.fill(2)
        log("inkling2 gpu_full: GPU%d (%s, cc%d, %d MiB), all %d experts in VRAM" % (
            dev_list[0], ginfo[1], ccs[0], ginfo[3], IK_NL_PHYS * IK_NE))

    else:
        return None

    lib, ptx_map = _compile_gpu(ccs)
    n_dev = len(dev_list)
    c_devs = (ctypes.c_int * n_dev)(*dev_list)
    c_ldev = (ctypes.c_int * IK_NL_PHYS)(*layer_dev)
    c_exp = exp_plan.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    c_ptx = (ctypes.c_char_p * n_dev)(*(ptx_map[int(c)] for c in ccs))
    conv_ptr = engine.lib.ik_conv_hist_ptr() if hasattr(engine.lib, "ik_conv_hist_ptr") else None
    moe_fn = ctypes.cast(engine.lib.ik_cpu_moe_slots, ctypes.c_void_p)
    hooks = _IkGpuHooks()

    ctx_ptr = lib.ik_gpu_init(
        engine.model_path.encode("utf-8"),
        ctypes.c_size_t(spec.data_start),
        spec.g_offs, spec.g_tys,
        spec.l_offs, spec.l_tys,
        ctypes.c_int(engine.ctx), ctypes.c_int(c_mode),
        ctypes.c_int(n_dev), c_devs, c_ldev,
        c_exp, ctypes.c_int(n_dcache),
        ctypes.c_int(split_q), ctypes.c_int(split_k), ctypes.c_int(split_v),
        ctypes.c_int(split_g), ctypes.c_int(split_o), ctypes.c_int(split_head),
        ctypes.c_void_p(conv_ptr), moe_fn,
        c_ptx, ctypes.byref(hooks),
    )
    if not ctx_ptr:
        raise RuntimeError("ik_gpu_init failed for mode %s" % mode_name)
    engine.lib.ik_set_gpu_hooks(ctypes.byref(hooks))
    return _GpuHandle(lib, ctx_ptr, engine)
