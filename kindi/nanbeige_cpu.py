"""The CPU part of the host layers of the Nanbeige engine: nbq_cpu.c (AVX2, OpenMP), built on first use with gcc
into KINDI_CPU_CACHE (default ~/.cache/kindi). See Engine._hcpu in nanbeige.py."""
import ctypes
import hashlib
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
_lib = None


def _grid_source():
    """the IQ3_XXS codebook of nbq_grids.h as a C array"""
    text = open(os.path.join(HERE, "nbq_grids.h"), encoding="utf-8").read()
    m = re.search(r"nbq_iq3xxs_grid\[256\]\s*=\s*\{([^}]*)\}", text)
    return "#include <stdint.h>\nconst uint32_t nbq_iq3xxs_grid_c[256] = {%s};\n" % m.group(1)


def lib():
    global _lib
    if _lib is None:
        src = _grid_source() + open(os.path.join(HERE, "nbq_cpu.c"), encoding="utf-8").read()
        cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
        os.makedirs(cache, exist_ok=True)
        tag = hashlib.sha1(src.encode()).hexdigest()[:12]
        so = os.path.join(cache, "nbq_cpu_%s.so" % tag)
        if not os.path.exists(so):
            c = os.path.join(cache, "nbq_cpu_%s.c" % tag)
            open(c, "w").write(src)
            subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-ffp-contract=off", "-shared", "-fPIC",
                                   "-o", so + ".tmp", c])
            os.replace(so + ".tmp", so)
        _lib = ctypes.CDLL(so)
        V, I = ctypes.c_void_p, ctypes.c_int
        _lib.nbq_cpu_expand_idx.argtypes = [V, I, I, I, V, V]
        _lib.nbq_cpu_expand_idx.restype = None
        _lib.nbq_cpu_dot_idx.argtypes = [V, V, I, I, V, V, V]
        _lib.nbq_cpu_dot_idx.restype = None
    return _lib
