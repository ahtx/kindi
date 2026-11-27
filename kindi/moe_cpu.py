"""The routed experts of the DeepSeek-V4.1-Flash engine on the host CPUs: moe_cpu.c (AVX2, own thread pool), built on
first use with gcc into KINDI_CPU_CACHE (default ~/.cache/kindi).

Placement: KINDI_MOE_CPUS0 / KINDI_MOE_CPUS1 are the cores of NUMA node 0 / 1 (one thread per physical core, lists
like "0-19" and "20-39"); unset, one logical CPU per physical core of that node (kindi.cores; "" for CPUS1: node 0
only). KINDI_MOE_SHARE0 is node 0's share of the rows (its share of the memory bandwidth; the default fits the
2-socket development host, where node 1 has half the memory channels)."""
import ctypes
import hashlib
import os
import subprocess

import numpy as np

from .cores import default_cpu_list

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None
_pool = False


def lib():
    global _lib
    if _lib is None:
        src = open(os.path.join(HERE, "moe_cpu.c"), encoding="utf-8").read()
        cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
        os.makedirs(cache, exist_ok=True)
        tag = hashlib.sha1(src.encode()).hexdigest()[:12]
        so = os.path.join(cache, "moe_cpu_%s.so" % tag)
        if not os.path.exists(so):
            c = os.path.join(cache, "moe_cpu_%s.c" % tag)
            open(c, "w").write(src)
            subprocess.check_call(["gcc", "-O3", "-march=native", "-pthread", "-shared", "-fPIC", "-o", so + ".tmp", c,
                                   "-lm"])
            os.replace(so + ".tmp", so)
        _lib = ctypes.CDLL(so)
        V, I, Fl = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        _lib.moe_pool_init.argtypes = [V, I, I, Fl]
        _lib.moe_pool_init.restype = I
        _lib.moe_layer_new.argtypes = [I, I, I, Fl]
        _lib.moe_layer_new.restype = V
        _lib.moe_layer_direct.argtypes = [I, I, I, Fl, V, V, V]
        _lib.moe_layer_direct.restype = V
        _lib.moe_layer_set_expert.argtypes = [V, I, V, V, V]
        _lib.moe_layer_set_expert.restype = None
        _lib.moe_forward.argtypes = [V, I, V, I, V, V, V, I]
        _lib.moe_forward.restype = I
    return _lib


def _cpus(text):
    out = []
    if not text:
        return out
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def pool():
    """start the thread pool once (the calling thread becomes thread 0, pinned to the first core of node 0)"""
    global _pool
    if not _pool:
        c0 = _cpus(os.environ.get("KINDI_MOE_CPUS0") or default_cpu_list(0))
        c1 = _cpus(os.environ.get("KINDI_MOE_CPUS1") or default_cpu_list(1)) if os.environ.get("KINDI_MOE_CPUS1") != "" else []
        share0 = float(os.environ.get("KINDI_MOE_SHARE0", "0.667"))
        arr = (ctypes.c_int * len(c0 + c1))(*(c0 + c1))
        if lib().moe_pool_init(arr, len(c0), len(c1), share0) != 0:
            raise RuntimeError("moe_cpu: pool init failed")
        _pool = True
    return lib()


class Layer:
    """the routed experts of one layer. gate, up: uint8 arrays of GGUF MXFP4 blocks [E][F][K/32 * 17], down
    [E][K][F/32 * 17] (for example views into a memory map of the GGUF file). copy=True places them on the NUMA
    nodes (the fast path); copy=False reads them in place."""

    def __init__(self, K, F, E, gate, up, down, clamp=10.0, copy=True):
        L = pool()
        self.K, self.F, self.E = K, F, E
        self._keep = (gate, up, down)
        if copy:
            self.h = L.moe_layer_new(K, F, E, clamp)
            if not self.h:
                raise MemoryError("could not allocate packed MoE weights")
            gs, ds = F * (K // 32) * 17, K * (F // 32) * 17
            for e in range(E):
                L.moe_layer_set_expert(self.h, e, gate.ctypes.data + e * gs, up.ctypes.data + e * gs,
                                       down.ctypes.data + e * ds)
            self._keep = None
        else:
            self.h = L.moe_layer_direct(K, F, E, clamp, gate.ctypes.data, up.ctypes.data, down.ctypes.data)

    def forward(self, x, ids, weights, weight_in_h=True):
        """x [T, K] float32, ids [T, n] int32, weights [T, n] float32 -> [T, K] float32 (routed experts only)"""
        x = np.ascontiguousarray(x, np.float32)
        ids = np.ascontiguousarray(ids, np.int32)
        weights = np.ascontiguousarray(weights, np.float32)
        T, n = ids.shape
        out = np.empty((T, self.K), np.float32)
        r = lib().moe_forward(self.h, T, x.ctypes.data, n, ids.ctypes.data, weights.ctypes.data, out.ctypes.data,
                              1 if weight_in_h else 0)
        if r < 0:
            raise ValueError("moe_cpu: bad expert id or more than 16 experts per token")
        return out
