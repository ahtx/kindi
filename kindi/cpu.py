"""Kindi CPU engine (KINDI_DEVICE=cpu): the 2B / 4B / 9B models on the CPU, token-exact with llama.cpp.

The whole forward pass of a token runs in C (kcpu.c, one OpenMP parallel region per token). Same interface as
kindi.engine.Engine: reset(), feed(tok, pos), step(tok, pos) -> greedy token, generate(ids, n), logits in
self.ylog[:self.ylog_n]. Loads a Kindi pack (kindi convert) and repacks the ternary weights into pair codes
("perm8": per 8-row block and 128-column group, 64 words of 32 bits; one vpermps gives 8 rows x 2 columns).
kcpu.c is built on first use into KINDI_CPU_CACHE (default ~/.cache/kindi): gcc on Linux, and on Windows
clang with the MSVC headers and libomp that its driver finds (KINDI_CC, KINDI_LLVM override). Five ISA tiers
are defined (AVX2/FMA/F16C, AVX2/F16C without FMA, AVX2/FMA without F16C, AVX2, AVX-only); the host's flags
select one, it is built with explicit -m flags, and the loaded library reports its tier in kc_isa_name().
KINDI_CPU_ISA=auto|avx2-fma-f16c|avx2-f16c-no-fma|avx2-fma|avx2|avx-only overrides the selection.
Threads: OMP_NUM_THREADS, or one per physical core of this process's CPUs by default (OMP_PROC_BIND=close,
OMP_PLACES=cores). The physical-core default is the portable choice; 8 threads (two SMT pairs) can be
faster under WSL2/libgomp on the 2B/9B, but 10/12 and native Windows 8 are slower. See
docs/CPU-SUPPORT.md for the sweep and _configure_threads for the default.

Tandem entry point: begin(tok) embeds a token, layer(il, pos) runs exactly one block, and head()
returns the greedy token from the residual left by the last layer. The whole-model step() remains the
single-region path; the per-layer calls are for handing one block to or from another backend."""
import ctypes
import hashlib
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np

import json


def load_json(path):
    """Read one UTF-8 JSON document and close the handle before returning."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

VOCAB = 248320
DEFAULT_DIMS = {"layers": 24, "width": 2048, "ffn": 6144, "heads": 8, "kv_heads": 2,
                "groups": 16, "head_dim": 256, "rope_dim": 64, "state_size": 128, "gdn_inner": 2048,
                "rope_base": 1e7, "full_attention_interval": 4}
HERE = os.path.dirname(os.path.abspath(__file__))
F32 = np.float32
PF = ctypes.POINTER(ctypes.c_float)
PU8 = ctypes.POINTER(ctypes.c_uint8)
PU16 = ctypes.POINTER(ctypes.c_uint16)

# One library per ISA tier, built on demand with explicit flags so a cache can serve different CPUs.
# The labels are also the cache tag and what kcpu.c:kc_isa_name() reports back.
ISA_BUILDS = {
    "avx2-fma-f16c": ("-mavx2", "-mfma", "-mf16c"),
    "avx2-f16c-no-fma": ("-mavx2", "-mno-fma", "-mf16c"),
    "avx2-fma": ("-mavx2", "-mfma", "-mno-f16c"),
    "avx2": ("-mavx2", "-mno-fma", "-mno-f16c"),
    "avx-only": ("-mavx", "-mno-avx2", "-mno-fma", "-mno-f16c"),
}
ISA_ORDER = ("avx2-fma-f16c", "avx2-f16c-no-fma", "avx2-fma", "avx2", "avx-only")
ISA_FEATURES = {
    "avx2-fma-f16c": ("avx", "avx2", "fma", "f16c"),
    "avx2-f16c-no-fma": ("avx", "avx2", "f16c"),
    "avx2-fma": ("avx", "avx2", "fma"),
    "avx2": ("avx", "avx2"),
    "avx-only": ("avx",),
}
FMA_ISAS = ("avx2-fma-f16c", "avx2-fma")


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


def _llvm_root(cc):
    """The clang install whose lib/ holds libomp.lib and bin/ libomp.dll, or None (KINDI_LLVM wins)."""
    root = os.environ.get("KINDI_LLVM")
    if root:
        return root
    exe = shutil.which(cc)
    if not exe:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(exe)))
    return root if os.path.isdir(os.path.join(root, "lib")) else None


def _compiler_flags(cc=None):
    """The ISA flags the compiler enables for this host, or None when it cannot be probed.

    Windows has no /proc/cpuinfo, so the compiler that will build the library is asked what
    -march=native defines. The same probe answers the CRT question: the static libcmt expf
    contains FMA instructions even with -mno-fma.
    """
    cc = cc or os.environ.get("KINDI_CC", "clang" if os.name == "nt" else "gcc")
    try:
        result = subprocess.run([cc, "-march=native", "-dM", "-E", "-x", "c", "-"],
                                input=b"", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if result.returncode:
        return None
    macros = result.stdout.decode("utf-8", "replace")
    return {name for name, macro in (("avx", "__AVX__"), ("avx2", "__AVX2__"),
                                     ("fma", "__FMA__"), ("f16c", "__F16C__"))
            if macro in macros}


def _native_has_fma(cc=None):
    """Whether the compiler's -march=native enables FMA on this host (False if unknown)."""
    flags = _compiler_flags(cc)
    return bool(flags and "fma" in flags)


def _build_cmd(src, so, win=None, native_fma=None, isa=None):
    """The compiler command line for one ISA tier: gcc on Linux; clang and libomp on Windows.

    `isa` names a tier in ISA_BUILDS and replaces `-march=native` with explicit flags, so the
    cached library does not depend on the build host. clang's driver finds the MSVC headers and
    Windows SDK through vswhere, so no vcvars shell is needed; the build stays an argument vector
    (paths from KINDI_CPU_CACHE are literal). -ffp-contract=off keeps Windows output on the same
    side of the FMA rounding as the gcc build that the references were made with.
    """
    if win is None:
        win = os.name == "nt"
    if isa is not None and isa not in ISA_BUILDS:
        raise ValueError("unknown CPU ISA tier %r" % isa)
    march = list(ISA_BUILDS[isa]) if isa is not None else ["-march=native"]
    if not win:
        return ["gcc", "-O3"] + march + ["-fopenmp", "-fno-math-errno", "-shared", "-fPIC",
                                          "-o", so, src, "-lm"]
    cc = os.environ.get("KINDI_CC", "clang")
    cmd = [cc, "-O3"] + march + ["-fopenmp", "-fno-math-errno", "-ffp-contract=off",
                                  "-shared", "-o", so, src]
    root = _llvm_root(cc)
    if root:
        cmd += ["-L", os.path.join(root, "lib"), "-llibomp"]
    # The static libcmt expf contains FMA instructions even with -mno-fma; use the dynamic
    # UCRT when the selected tier does not use FMA, so a no-FMA AVX2 build and an AVX-only
    # (Sandy Bridge) build cannot fault in the math runtime while FMA builds keep the
    # existing static CRT.
    uses_fma = (isa in FMA_ISAS) if isa is not None else (
        _native_has_fma() if native_fma is None else native_fma)
    if not uses_fma:
        cmd += ["-Xlinker", "/nodefaultlib:libcmt", "-Xlinker", "/defaultlib:msvcrt"]
    return cmd


def _build(src, so, isa=None):
    """Compile kcpu.c for one ISA tier without a shell, so paths from KINDI_CPU_CACHE stay literal."""
    if isa is not None:
        native_fma = isa in FMA_ISAS
    elif os.name == "nt":
        native_fma = _native_has_fma()
    else:
        native_fma = True
    cmd = _build_cmd(src, so, native_fma=native_fma, isa=isa)
    print("kindi: building CPU library with %s" % " ".join(cmd), file=sys.stderr, flush=True)
    try:
        result = subprocess.run(cmd)
    except OSError as e:
        raise SystemExit("kindi: could not run %s (%s); it needs gcc with OpenMP, or clang and libomp on Windows"
                         % (cmd[0], e))
    if result.returncode:
        raise SystemExit("kindi: could not build the CPU engine for %s (%s); it needs gcc with OpenMP, or "
                         "clang and libomp on Windows, and a CPU with AVX (docs/CPU-SUPPORT.md)"
                         % (isa or "native", so))


def _cache_key(isa=None):
    """Identify the build that a cached CPU library came from.

    An explicit-tier key covers the OS, architecture, compiler and flags, so it can be shared by
    hosts with the same tier. The native fallback (no flags exposed) also covers the host CPU
    identity: a `-march=native` binary must never be handed to a different machine.
    """
    parts = [os.name, platform.machine(),
             os.environ.get("KINDI_CC", "gcc" if os.name != "nt" else "clang"),
             "isa=" + (isa or "native")]
    if isa is None:
        native_fma = True
        if os.name == "nt":
            native_fma = _native_has_fma()
            parts.append(os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or ""))
            parts.append("fma" if native_fma else "no-fma")
        else:
            info = _read("/proc/cpuinfo") or ""
            parts.extend(line.strip() for line in info.splitlines()
                         if line.startswith(("model name", "flags", "Features")))
    else:
        native_fma = isa in FMA_ISAS
    parts.append(" ".join(_build_cmd("kcpu.c", "libkcpu", native_fma=native_fma, isa=isa)))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _so_path(cache=None, isa=None):
    """The cache path for one ISA tier; the key covers the tier, compiler and OS."""
    cache = cache or os.environ.get("KINDI_CPU_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "kindi")
    ext = "dll" if os.name == "nt" else "so"
    return os.path.join(cache, "libkcpu-%s.%s" % (_cache_key(isa), ext))


def _lib():
    src = os.path.join(HERE, "kcpu.c")
    cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache, exist_ok=True)
    isa = select_isa()
    so = _so_path(cache, isa)
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        check_features()
        _build(src, so, isa)
    if os.name == "nt":
        root = _llvm_root(os.environ.get("KINDI_CC", "clang"))
        if root and os.path.isdir(os.path.join(root, "bin")):
            os.add_dll_directory(os.path.join(root, "bin"))      # libomp.dll the built DLL links against
    lib = ctypes.CDLL(so)
    lib.kc_init()
    lib.kc_isa.restype = ctypes.c_int
    lib.kc_isa_name.restype = ctypes.c_char_p
    built = lib.kc_isa_name().decode("ascii")
    if isa is not None and built != isa:
        raise SystemExit("kindi: the CPU library at %s reports %s, expected %s; delete it and retry"
                         % (so, built, isa))
    lib.kindi_isa = built
    lib.kc_step.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.kc_step.restype = ctypes.c_int
    lib.kc_embed.argtypes = [ctypes.POINTER(Model), ctypes.c_int]
    lib.kc_layer.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_int]
    lib.kc_layer.restype = ctypes.c_int
    lib.kc_layer_ffn_input.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_int]
    lib.kc_layer_ffn_input.restype = ctypes.c_int
    lib.kc_gemv.argtypes = [ctypes.POINTER(Model), ctypes.POINTER(Mat), PF, PF]
    lib.kc_gemv.restype = ctypes.c_int
    lib.kc_head.argtypes = [ctypes.POINTER(Model)]
    lib.kc_head.restype = ctypes.c_int
    assert lib.kc_sizeof_model() == ctypes.sizeof(Model) and lib.kc_sizeof_layer() == ctypes.sizeof(Layer)
    return lib


def _cpu_list(s):
    """Parse a Linux CPU list ("0-3,8,10-11") into [0, 1, 2, 3, 8, 10, 11]."""
    out = []
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1 if hi else int(lo) + 1))
    return out


def _affinity():
    """The CPUs this process may run on, or None where the OS does not say (Linux sched_getaffinity)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def _read(path):
    """The text of a system file, or None when it is not there."""
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


CPU_FEATURES = (("avx", "AVX"), ("avx2", "AVX2"), ("fma", "FMA3"), ("f16c", "F16C"))
CPU_REQUIRED = (("avx", "AVX"),)

CPU_TIER_LABELS = {
    "avx2-fma-f16c": "AVX2/FMA/F16C fast path",
    "avx2-f16c-no-fma": "AVX2/F16C without FMA (separate multiply/add)",
    "avx2-fma": "AVX2/FMA with software F16C",
    "avx2": "AVX2 with software F16C and no FMA",
    "avx-only": "AVX-only fallback",
    "unknown": "unknown until the compiler selects the path",
    "unsupported": "unsupported (AVX missing)",
}


def _cpu_model(text):
    """The model name from /proc/cpuinfo, or the platform's best fallback."""
    if text:
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("model name", "Model", "Hardware"):
                value = value.strip()
                if value:
                    return value
    return platform.processor() or platform.machine() or "unknown CPU"


def _cpu_flags(text):
    """The /proc/cpuinfo feature set, or None when this OS does not expose it."""
    if not text:
        return None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() in ("flags", "features"):
            return set(value.split())
    return None


def feature_report():
    """CPU model, core counts and AVX/AVX2/FMA3/F16C support.

    A feature value is None when the OS does not expose CPU flags; the compile-time guard in
    kcpu.c remains the authority in that case.  This report is for `kindi info` and the preflight
    before the CPU library is built.
    """
    text = _read("/proc/cpuinfo")
    flags = _cpu_flags(text)
    if flags is None and os.name == "nt":
        flags = _compiler_flags()          # Windows has no /proc/cpuinfo; ask the build compiler
    return {"model": _cpu_model(text), "physical": _physical_cores(), "logical": os.cpu_count(),
            "features": {name: None if flags is None else name in flags for name, _ in CPU_FEATURES}}


def cpu_tier(report=None):
    """The `kcpu.c` path this host should use: AVX2 (with optional FMA/F16C) or AVX-only.

    AVX remains the baseline.  AVX2 is used whenever the compiler enables it; FMA is optional,
    and an AVX2/F16C host without it takes the separate multiply/add branch in `rows8`.  An
    AVX-only host uses the scalar fallback.  Unknown flags make no claim: the compile-time checks
    in `kcpu.c` remain the authority.
    """
    report = feature_report() if report is None else report
    f = report["features"]
    if f.get("avx") is False:
        return "unsupported"
    if f.get("avx") is None:
        return "unknown"
    if f.get("avx2"):
        if f.get("f16c") and f.get("fma"):
            return "avx2-fma-f16c"
        if f.get("f16c"):
            return "avx2-f16c-no-fma"
        if f.get("fma"):
            return "avx2-fma"
        return "avx2"
    return "avx-only"


def select_isa(report=None, forced=None):
    """The ISA tier to load: the best one this CPU supports, or KINDI_CPU_ISA / `forced`.

    None means the OS exposes no flags; the caller then keeps the `-march=native` build and the
    compile-time gates in kcpu.c are the authority. A forced tier that this CPU explicitly does
    not support is an error: loading it would execute an illegal instruction.
    """
    report = feature_report() if report is None else report
    flags = report["features"]
    choice = os.environ.get("KINDI_CPU_ISA", "auto") if forced is None else forced
    choice = (choice or "auto").strip().lower().replace("_", "-")
    choice = {"native": "auto", "avxonly": "avx-only",
              "avx2-f16c": "avx2-f16c-no-fma"}.get(choice, choice)
    if choice != "auto" and choice not in ISA_BUILDS:
        raise SystemExit("kindi: KINDI_CPU_ISA must be auto or one of %s (got %r)"
                         % (", ".join(ISA_ORDER), choice))
    if choice != "auto":
        missing = [name.upper() for name in ISA_FEATURES[choice] if flags.get(name) is False]
        if missing:
            raise SystemExit("kindi: KINDI_CPU_ISA=%s needs %s; this CPU lacks %s"
                             % (choice, ", ".join(n.upper() for n in ISA_FEATURES[choice]),
                                ", ".join(missing)))
        return choice
    tier = cpu_tier(report)
    if tier == "unsupported":
        raise SystemExit("kindi: the CPU engine needs AVX; this CPU lacks AVX (%s)" % report["model"])
    return None if tier == "unknown" else tier


def missing_features(report=None):
    """Labels of required ISA features that are explicitly absent, or ()."""
    report = feature_report() if report is None else report
    return tuple(label for name, label in CPU_REQUIRED if report["features"].get(name) is False)


def check_features():
    """Stop before compiling when /proc/cpuinfo says this CPU cannot run kcpu.c."""
    report = feature_report()
    missing = missing_features(report)
    if missing:
        raise SystemExit("kindi: the CPU engine needs AVX; this CPU lacks %s (%s)"
                         % (", ".join(missing), report["model"]))
    return report


def _siblings(cpu):
    """The hyperthread siblings of a CPU from sysfs, or None when the topology is not exposed."""
    sib = _read("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu)
    return None if sib is None else _cpu_list(sib)


def _cpuinfo_cores(cpus):
    """Fallback: count the (physical id, core id) pairs of `cpus` in /proc/cpuinfo."""
    text = _read("/proc/cpuinfo")
    if text is None:
        return None
    seen, cur = set(), {}

    def flush():
        if cur.get("processor") in cpus:
            seen.add((cur.get("physical id", 0), cur.get("core id", cur["processor"])))
        cur.clear()
    for line in text.splitlines():
        if not line.strip():
            flush()
            continue
        key, _, val = line.partition(":")
        if key.strip() in ("processor", "physical id", "core id"):
            try:
                cur[key.strip()] = int(val)
            except ValueError:
                pass
    flush()
    return len(seen) or None


def _count_core_records(buf):
    """Count RelationProcessorCore (0) records in GetLogicalProcessorInformationEx output."""
    n, off = 0, 0
    while off + 8 <= len(buf):
        rel, size = struct.unpack_from("<II", buf, off)
        if size < 8:
            break
        if rel == 0:
            n += 1
        off += size
    return n or None


def _windows_cores():
    """Physical cores of the machine from the Windows topology API, or None."""
    try:
        k32 = ctypes.windll.kernel32
        need = ctypes.c_ulong(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(need))
        buf = ctypes.create_string_buffer(need.value)
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(need)):
            return None
        return _count_core_records(buf.raw[:need.value])
    except (AttributeError, OSError):
        return None


def _physical_cores():
    """How many physical cores this process may run on; the logical CPU count when the topology is unknown."""
    if os.name == "nt":
        n = _windows_cores()
        if n:
            return n
    cpus = _affinity()
    if not cpus:
        cpus = list(range(os.cpu_count() or 1))
    cores = set()
    for c in cpus:
        sib = _siblings(c)
        if sib is None:
            break
        cores.add(min([s for s in sib if s in cpus] or sib))
    else:
        if cores:
            return len(cores)
    return _cpuinfo_cores(set(cpus)) or len(cpus)


def _configure_threads(threads=None):
    """Resolve the OpenMP thread count and binding before the C library loads (OpenMP reads them once).

    An explicit `threads` wins, then OMP_NUM_THREADS already in the environment, then one thread per
    physical core of this process's CPUs, bound to those cores.  The physical-core default is the
    portable choice.  Eight threads (two SMT pairs) can be faster in WSL2/libgomp on the 2B/9B, but
    4B read regresses, 10/12 are slower, and native Windows/clang is slower at 8; explicit values
    remain available for a measured workload."""
    if threads:
        n = int(threads)
        os.environ["OMP_NUM_THREADS"] = str(n)
        return n
    env = os.environ.get("OMP_NUM_THREADS")
    if env:
        try:
            return int(env)
        except ValueError:
            return env
    n = _physical_cores()
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    return n


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


def _p(a, t: Any = PF) -> Any:
    return a.ctypes.data_as(t)


class Engine(object):
    def __init__(self, pack, ctx=4096, clusters=None, log=print, threads=None, layers=None, embed=True, head=True,
                 head_rows=None):
        t0 = time.time()
        self.threads = _configure_threads(threads)
        self.lib = _lib()
        self.isa = getattr(self.lib, "kindi_isa", "native")
        self.ctx = int(ctx)
        man = load_json(os.path.join(pack, "kindi_pack.json"))
        d = self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))
        self.head_name = man.get("head", "token_embd.weight")
        D, NL, NFF = d["width"], d["layers"], d["ffn"]
        self.D, self.NL = D, NL
        # Tandem splits load only the slice this backend owns (see TandemEngine):
        # converting every layer of a 27B pack on the CPU took ~365 s even when
        # the CPU owned 16 of 64 layers.  layers=None keeps the full-model load.
        self.layers = None if layers is None else tuple(int(x) for x in layers)
        self.layers_set = None if self.layers is None else set(self.layers)
        if self.layers is not None:
            if not self.layers:
                raise SystemExit("kindi: CPU engine needs at least one layer")
            bad = [il for il in self.layers if not 0 <= il < NL]
            if bad:
                raise SystemExit("kindi: CPU layer %d is outside 0..%d" % (bad[0], NL - 1))
        self.need_embed, self.need_head = bool(embed), bool(head)
        self.head_rows = None if head_rows is None else (int(head_rows[0]), int(head_rows[1]))
        self.head_off, self.head_count = 0, VOCAB
        if self.head_rows is not None:
            r0, r1 = self.head_rows
            if not 0 <= r0 < r1 <= VOCAB:
                raise SystemExit("kindi: head rows must be a non-empty range inside 0..%d" % (VOCAB - 1))
            if r0 % 16 or r1 % 16:
                raise SystemExit("kindi: head row bounds must be multiples of 16")
            if self.head_name == "token_embd.weight" and self.need_embed:
                raise SystemExit("kindi: a partial tied head cannot also be the embedding")
            self.head_off, self.head_count = r0, r1 - r0
        NH, NHKV, HD = d["heads"], d["kv_heads"], d["head_dim"]
        NK, HK = d["groups"], d["state_size"]
        NV = d["gdn_inner"] // HK
        CC = 2 * NK * HK + NV * HK
        interval = d["full_attention_interval"]
        idx = load_json(os.path.join(pack, "index.json"))["tensors"]
        bi = load_json(os.path.join(pack, "bcodes_index.json"))["tensors"]
        bc = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
        dp = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.float16, mode="r")
        self.keep = []                    # every array the C side points at

        def keep(a):
            self.keep.append(a)
            return a
        def _wanted(name):
            if self.layers is None:
                return True
            if name == "token_embd.weight":
                return self.need_embed or (self.need_head and name == self.head_name)
            if name == self.head_name:
                return self.need_head
            if name.startswith("blk."):
                try:
                    return int(name.split(".")[1]) in self.layers_set
                except (IndexError, ValueError):
                    return False
            return False

        mats = {}
        self._mat_arrays = {}
        for name, rec in bi.items():
            if not _wanted(name):
                continue
            R, C = rec["rows"], rec["ne0"]
            ns, G = C // 4, C // 128
            o, q = rec["bc_off"], rec["dpm_off"] // 2
            if name == self.head_name and self.head_rows is not None:
                r0, r1 = self.head_rows
                b = np.ascontiguousarray(np.asarray(bc[o:o + ns * R]).reshape(ns, R)[:, r0:r1])
                s = np.asarray(dp[q:q + G * R]).reshape(G, R // 8, 8)[:, r0 // 8:r1 // 8, :].transpose(1, 0, 2)
                W = keep(paircodes(b, r1 - r0, C))
                S = keep(np.ascontiguousarray(s.astype(F32)))
                mats[name] = Mat(_p(W, PU8), _p(S), r1 - r0, C)
                self._mat_arrays[name] = (W, S)
                continue
            W = keep(paircodes(bc[o:o + ns * R], R, C))
            S = keep(np.ascontiguousarray(np.asarray(dp[q:q + G * R]).reshape(G, R // 8, 8).transpose(1, 0, 2).astype(F32)))
            mats[name] = Mat(_p(W, PU8), _p(S), R, C)
            self._mat_arrays[name] = (W, S)
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
            if self.layers is not None and il not in self.layers_set:
                continue                    # unowned layer: zero Layer, never called on this backend
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
        _zero = Mat()
        m.emb = mats["token_embd.weight"] if self.need_embed and "token_embd.weight" in mats else _zero
        m.head = mats[self.head_name] if self.need_head and self.head_name in mats else _zero
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
                 "T": W // 2 * 8, "logits": max(VOCAB, mats[self.head_name].R if self.need_head and self.head_name in mats else VOCAB),
                 "gb": NV, "gg": NV}
        for k in _SCR:
            a = z(sizes[k])
            setattr(m, k, _p(a))
            if k == "logits":
                self.logits = a
        m.red = _p(z(256, np.float64), ctypes.POINTER(ctypes.c_double))
        if self.head_rows is None:
            self.ylog, self.ylog_n, self.ylog_map = self.logits, VOCAB, None
        else:
            self.ylog, self.ylog_n = self.logits, self.head_count
            self.ylog_map = np.arange(self.head_off, self.head_off + self.head_count, dtype=np.int64)
        self.hidden = np.ctypeslib.as_array(m.h, shape=(D,))   # in-place residual for a tandem handoff
        self.hsl, self.shortlist = None, 0
        self.shortlist = 0
        self.vocab = load_json(os.path.join(pack, "vocab.json"))
        (log or (lambda *a: None))("kindi-cpu ready: %s, %s, %s threads, ctx %d (%.1fs)" % (
            man.get("model"), self.isa, self.threads, self.ctx, time.time() - t0))

    def reset(self):
        for a in self.M + self.conv + self.Kc + self.Vc:
            a.fill(0)

    def begin(self, tok):
        """Embed one token into self.hidden; the next layer() call starts layer 0."""
        self.lib.kc_embed(ctypes.byref(self.m), int(tok))

    def layer(self, il, pos):
        """Run exactly one block on self.hidden (the complete residual entering the layer)."""
        il, pos = int(il), int(pos)
        if not 0 <= il < self.NL:
            raise IndexError("layer %d is outside 0..%d" % (il, self.NL - 1))
        if not 0 <= pos < self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        if self.lib.kc_layer(ctypes.byref(self.m), il, pos):
            raise ValueError("kindi: the CPU layer entry point rejected layer %d" % il)

    def layer_pre_ffn(self, il, pos):
        """Run a layer up to the folded FFN input, leaving the down GEMV to the caller."""
        if self.lib.kc_layer_ffn_input(ctypes.byref(self.m), int(il), int(pos)):
            raise ValueError("kindi: the CPU FFN-input entry point rejected layer %d" % il)

    def layer_finish(self):
        """Add the caller-filled FFN-down output to the CPU residual stream."""
        self.hidden += np.ctypeslib.as_array(self.m.y, shape=(self.D,))

    def ffn_input_host(self):
        """A copy of the folded FFN input left by layer_pre_ffn."""
        return np.ctypeslib.as_array(self.m.xp, shape=(int(self.m.NFF),)).copy()

    def output_y(self):
        """The mutable per-layer FFN-down output buffer (D floats)."""
        return np.ctypeslib.as_array(self.m.y, shape=(self.D,))

    def feed_layers(self, tok, pos):
        """The per-layer equivalent of feed(): embed then run every block."""
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        self.begin(tok)
        for il in range(self.NL):
            self.layer(il, pos)

    def head(self):
        """Greedy token from the residual left by the last layer (or by begin() plus layers)."""
        tok = int(self.lib.kc_head(ctypes.byref(self.m)))
        return tok + self.head_off if self.head_rows is not None else tok

    def gemv_rows(self, name, rows, x):
        """One loaded ternary matrix over a row range, with the complete input vector."""
        r0, r1 = (int(rows[0]), int(rows[1]))
        W, S = self._mat_arrays[name]
        if r0 < 0 or r1 > (W.shape[0] * 8) or r0 >= r1 or r0 % 8 or r1 % 8:
            raise ValueError("kindi: row range %r is not an 8-row slice of %s" % (rows, name))
        C = int(S.shape[1] * 128)
        x = np.ascontiguousarray(x, dtype=F32)
        if x.shape != (C,):
            raise ValueError("kindi: input has shape %r, expected (%d,)" % (x.shape, C))
        w = np.ascontiguousarray(W[r0 // 8:r1 // 8])
        s = np.ascontiguousarray(S[r0 // 8:r1 // 8])
        A = Mat(_p(w, PU8), _p(s), r1 - r0, C)
        y = np.empty(r1 - r0, dtype=F32)
        if self.lib.kc_gemv(ctypes.byref(self.m), ctypes.byref(A), _p(x), _p(y)):
            raise ValueError("kindi: the CPU row GEMV rejected %s" % name)
        return y

    def feed(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 0)

    def step(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        best = int(self.lib.kc_step(ctypes.byref(self.m), int(tok), int(pos), 1))
        return best + self.head_off if self.head_rows is not None else best

    def step_layers(self, tok, pos):
        """The per-layer equivalent of step(): embed, every block, then the exact head."""
        self.feed_layers(tok, pos)
        return self.head()

    def set_hidden(self, values):
        """Copy a residual from another backend into the CPU layer input."""
        x = np.asarray(values, dtype=F32)
        if x.shape != (self.D,):
            raise ValueError("hidden state has shape %r, expected (%d,)" % (x.shape, self.D))
        np.copyto(self.hidden, x)

    def get_hidden(self):
        """A copy of the residual left by the last layer/head."""
        return self.hidden.copy()

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
