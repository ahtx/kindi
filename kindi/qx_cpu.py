"""The routed experts of Qwen3.8-Flash-Next (qwen4exp) on the host CPUs: qx_cpu.c on the thread pool of moe_cpu.c,
optionally with the expert reads of xstore.c. Built on first use with gcc into KINDI_CPU_CACHE (default
~/.cache/kindi).

Settings:
  KINDI_QX_CPUS      the pool's cores (default "0-15"); one thread per core, the calling thread becomes thread 0
  KINDI_QX_XS_GB     > 0: stream the experts through xstore.c with a RAM cache of that many GB (direct I/O from the
                     model file); default 0: read them in place from the mapped file (the page cache)
  KINDI_XS_THREADS   xstore I/O threads (default 8 here), KINDI_XS_CHUNK bytes per read (default 1 MB)

The kernels give the same bits for a token whatever the batch, the threads and the expert source; see qx_cpu.c."""
import ctypes
import hashlib
import os
import sys
import subprocess

import numpy as np

from .gguf import GGUF

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("moe_cpu.c", "xstore.c", "qx_cpu.c", "moeaccel.c")
_lib = None
_pool = None


def lib():
    global _lib
    if _lib is None:
        srcs = [open(os.path.join(HERE, f), encoding="utf-8").read() for f in SOURCES]
        grids = open(os.path.join(HERE, "nbq_grids.h"), encoding="utf-8").read()
        cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
        os.makedirs(cache, exist_ok=True)
        hdr = open(os.path.join(HERE, "moeaccel.h"), encoding="utf-8").read()
        tag = hashlib.sha1(("qx1" + grids + hdr + "".join(srcs)).encode()).hexdigest()[:12]
        so = os.path.join(cache, "qx_cpu_%s.so" % tag)
        if not os.path.exists(so):
            objs = []
            for name, src in zip(SOURCES, srcs):
                c = os.path.join(cache, "%s_%s.c" % (name[:-2], tag))
                with open(c, "w", encoding="utf-8", newline="") as f:
                    f.write(src)
                subprocess.check_call(["gcc", "-O3", "-march=native", "-pthread", "-fPIC", "-ffp-contract=off",
                                       "-I", HERE, "-c", "-o", c + ".o", c])
                objs.append(c + ".o")
            subprocess.check_call(["gcc", "-shared", "-pthread", "-o", so + ".tmp"] + objs + ["-lm", "-ldl"])
            os.replace(so + ".tmp", so)
            for o in objs:
                os.remove(o)
                os.remove(o[:-2])
        L = ctypes.CDLL(so)
        V, I, Fl, D, Lo = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_double, ctypes.c_long
        L.moe_pool_init.argtypes = [V, I, I, Fl]
        L.moe_pool_init.restype = I
        L.moe_pool_stop.restype = None
        L.qx_map.argtypes = [ctypes.c_char_p, ctypes.POINTER(Lo)]
        L.qx_map.restype = V
        L.qx_add_layer.argtypes = [I, I, I, I, I, V, V, V]
        L.qx_add_layer.restype = I
        L.qx_layer_set_xs.argtypes = [I, I]
        L.qx_forward.argtypes = [I, I, V, I, V, V, V]
        L.qx_forward.restype = I
        L.qx_forward_parts.argtypes = [I, I, V, I, V, V, V]
        L.qx_forward_parts.restype = I
        L.qx_expf.argtypes = [Fl]
        L.qx_expf.restype = Fl
        L.qx_dequant.argtypes = [I, V, Lo, V]
        L.qx_dequant.restype = I
        L.qx_quant.argtypes = [V, I, V, V]
        L.qx_row_dot.argtypes = [I, V, I, I, V, V, V]
        L.qx_row_dot_ref.argtypes = [I, V, I, V, V]
        L.qx_row_dot_ref.restype = Fl
        L.qx_quant256.argtypes = [V, I, V, V]
        L.qx_row_dot256.argtypes = [I, V, I, I, V, V, V]
        L.qx_row_dot256_ref.argtypes = [I, V, I, V, V]
        L.qx_row_dot256_ref.restype = Fl
        L.qx_transcode.argtypes = [I, V, Lo, V]
        L.qx_transcode.restype = I
        L.moe_pool_set_spin.argtypes = [ctypes.c_long]
        L.moe_pool_prof.argtypes = [V, V]
        L.moe_pool_prof.restype = I
        L.qx_ma_ctx_new.argtypes = [I, I, I, V, V]
        L.qx_ma_ctx_new.restype = V
        L.qx_ma_fn_ptr.restype = V
        L.qx_ma_ctx_set_note.argtypes = [V, V]
        # KINDI_QX_VDEC: the code decodes (qx_cpu.c qx_vdec; bit 0 vector Q2_0, bit 1 gathered i-quant grids)
        # KINDI_QX_DN: the down input's activation block, 32 or 64 (item G; the card reads the same variable)
        ctypes.c_int.in_dll(L, "qx_dn").value = int(os.environ.get("KINDI_QX_DN", "32"))
        if os.environ.get("KINDI_QX_ONE"):            # 0: no one-vector kernels (qx_one, A/B)
            ctypes.c_int.in_dll(L, "qx_one").value = int(os.environ["KINDI_QX_ONE"])
        if os.environ.get("KINDI_QX_VDEC"):
            ctypes.c_int.in_dll(L, "qx_vdec").value = int(os.environ["KINDI_QX_VDEC"])
        L.qx_ma_ctx_set_fetch.argtypes = [V, V, I]
        L.qx_ma_ctx_set_tail.argtypes = [V, I, I]
        L.qx_ma_ctx_dropped.argtypes = [V]
        L.qx_ma_ctx_dropped.restype = ctypes.c_longlong
        L.qx_ma_ctx_stats.argtypes = [V, V]
        L.ma_box_layout.argtypes = [I, I, Lo, Lo, I, V]
        L.ma_box_serve_pass.argtypes = [V, I, I, I, Lo, Lo, I, ctypes.c_uint, V, V, D]
        L.ma_box_serve_pass.restype = I
        L.ma_timeline.argtypes = [V, I]
        L.ma_timeline.restype = None
        L.qx_phases.argtypes = [V, I]
        L.qx_phases.restype = None
        L.qx_read_bw.argtypes = [I, I]
        L.qx_read_bw.restype = ctypes.c_double
        L.qx_row_bytes.argtypes = [I, I]
        L.qx_row_bytes.restype = Lo
        L.qx_stats.argtypes = [V, ctypes.POINTER(D)]
        L.xs_add_file.argtypes = [V, Lo, ctypes.c_char_p]
        L.xs_add_file.restype = I
        L.xs_add_layer.argtypes = [I, V, V, V, Lo, Lo]
        L.xs_add_layer.restype = I
        L.xs_start.argtypes = [D, I, Lo]
        L.xs_start.restype = I
        L.xs_print_stats.argtypes = [ctypes.c_char_p]
        _lib = L
    return _lib


def _nodes_of(cl):
    """the NUMA nodes that hold the cores of cl: [set of cores of node 0, of node 1, ...] (nodes without one of
    them left out)"""
    fake = os.environ.get("KINDI_QX_NUMA_FAKE")         # tests on one node: the first n cores play node 0
    if fake:
        return [set(cl[:int(fake)]), set(cl[int(fake):])]
    out = []
    base = "/sys/devices/system/node"
    try:
        names = sorted((n for n in os.listdir(base) if n.startswith("node") and n[4:].isdigit()), key=lambda n: int(n[4:]))
    except OSError:
        return [set(cl)]
    for n in names:
        try:
            with open(os.path.join(base, n, "cpulist")) as f:
                cs = set(cpus(f.read().strip()))
        except OSError:
            continue
        if cs & set(cl):
            out.append(cs)
    return out or [set(cl)]


def Q_nbytes(ty, ne):
    from .qwen4exp_quant import nbytes
    n = 1
    for v in ne:
        n *= int(v)
    return nbytes(ty, n)


def _mem_available():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def cpus(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class Experts(object):
    """the routed experts of every MoE layer of a qwen4exp GGUF (all shards), on the CPU"""

    def __init__(self, path, cpu_list=None, xs_gb=None):
        from .qwen4exp_ref import shards
        L = lib()
        cl = cpus(cpu_list or os.environ.get("KINDI_QX_CPUS", "0-15"))
        self.files, t = [], {}
        for p in shards(path):
            g = GGUF(p)
            size = ctypes.c_long()
            base = L.qx_map(p.encode(), ctypes.byref(size))
            if not base:
                raise OSError("qx_cpu: cannot map %s" % p)
            self.files.append((p, base, size.value))
            for name, ne, ty, off in g.tensors:
                t[name] = (base + g.data_start + off, ne, ty)
        xs_gb = float(os.environ.get("KINDI_QX_XS_GB", "0")) if xs_gb is None else xs_gb
        if xs_gb > 0:
            for p, base, size in self.files:
                L.xs_add_file(base, size, p.encode())
        # the i-quant gate/up in nibble form (qx_transcode: same values, a faster decode) when the RAM allows it
        nib = os.environ.get("KINDI_QX_NIB", "auto")
        if xs_gb > 0 or nib == "0":
            on = False
        else:
            need = 0
            for name, (ptr_, ne, ty) in t.items():
                if name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight")) and ty in (18, 21, 22):
                    need += ne[0] // 256 * ne[1] * ne[2] * 146
            on = nib == "1" or _mem_available() > 1.5 * need + 8e9
        ctypes.c_int.in_dll(L, "qx_nib").value = 1 if on else 0
        ctypes.c_int.in_dll(L, "qx_collapse").value = int(os.environ.get("KINDI_QX_COLLAPSE", "1"))
        # the down matrices in anonymous memory (KINDI_QX_DNCOPY, auto: when the RAM holds them beside the nibble
        # copies; xstore keeps the file mapping)
        dnc = os.environ.get("KINDI_QX_DNCOPY", "auto")
        dneed = sum(Q_nbytes(ty, ne) for name, (ptr_, ne, ty) in t.items() if name.endswith("ffn_down_exps.weight"))
        ctypes.c_int.in_dll(L, "qx_dncopy").value = 1 if xs_gb <= 0 and (
            dnc == "1" or dnc == "auto" and _mem_available() > 1.5 * dneed + 8e9 + (need if on else 0)) else 0
        # the pool: per NUMA node when the cores span two nodes and the RAM holds a node copy of every expert (each
        # node's threads then read only their node's memory); it stays awake between calls (KINDI_MOE_SPIN pauses,
        # default 4,000,000: the card's part of a layer is shorter than a sleep and a wake-up); the calling thread
        # (the host side of every call) stays on the pool's first core (KINDI_QX_HOSTPIN=0: back to all cores)
        global _pool
        self.numa = None
        if _pool is None:
            nodes = _nodes_of(cl)
            numa = os.environ.get("KINDI_QX_NUMA", "auto")
            copy = 0
            for name, (ptr_, ne, ty) in t.items():
                if "_exps.weight" in name:
                    if on and ty in (18, 21, 22) and not name.endswith("ffn_down_exps.weight"):
                        copy += ne[0] // 256 * ne[1] * ne[2] * 146
                    else:
                        copy += Q_nbytes(ty, ne)
            split = (len(nodes) == 2 and numa != "0" and xs_gb <= 0
                     and (numa == "1" or _mem_available() > 1.3 * copy + 16e9))
            if split:
                c0 = [c for c in cl if c in nodes[0]]
                c1 = [c for c in cl if c in nodes[1]]
                order, n0, n1 = c0 + c1, len(c0), len(c1)
            else:
                order, n0, n1 = cl, len(cl), 0
            keep = os.sched_getaffinity(0)
            arr = (ctypes.c_int * len(order))(*order)
            if L.moe_pool_init(arr, n0, n1, (n0 / float(n0 + n1)) if n1 else 1.0) != 0:
                raise RuntimeError("qx_cpu: pool init failed")
            L.moe_pool_set_spin(int(os.environ.get("KINDI_MOE_SPIN", "4000000")))
            if os.environ.get("KINDI_QX_HOSTPIN", "1") == "0":
                os.sched_setaffinity(0, keep)
            ctypes.c_int.in_dll(L, "qx_numa").value = 1 if split else 0
            _pool = len(order)
            self.numa = (n0, n1) if split else None
        self.threads = _pool
        self.layer, self.types = {}, {}
        l = 0
        while "blk.%d.ffn_gate_exps.weight" % l in t:
            pre = "blk.%d.ffn_" % l
            g, u, d = t[pre + "gate_exps.weight"], t[pre + "up_exps.weight"], t[pre + "down_exps.weight"]
            K, F, E = g[1][0], g[1][1], g[1][2]
            if u[2] != g[2]:
                raise ValueError("layer %d: gate and up types differ" % l)
            h = L.qx_add_layer(K, F, E, g[2], d[2], g[0], u[0], d[0])
            if h < 0:
                raise ValueError("layer %d: expert types %d/%d are not supported" % (l, g[2], d[2]))
            if xs_gb > 0:
                xl = L.xs_add_layer(E, g[0], u[0], d[0], L.qx_row_bytes(g[2], K) * F, L.qx_row_bytes(d[2], F) * K)
                if xl < 0:
                    raise RuntimeError("xstore: layer %d not added (%d)" % (l, xl))
                L.qx_layer_set_xs(h, xl)
            self.layer[l] = h
            self.types[l] = (g[2], d[2])
            self.K = K
            l += 1
        if xs_gb > 0 and L.xs_start(xs_gb * 1e9, int(os.environ.get("KINDI_XS_THREADS", "8")),
                                    int(os.environ.get("KINDI_XS_CHUNK", str(1 << 20)))) < 1:
            raise RuntimeError("xstore: start failed")
        self.xs = xs_gb > 0
        gb = lambda n: ctypes.c_longlong.in_dll(L, n).value / 1e9          # noqa: E731
        ahp = 0
        try:
            with open("/proc/self/smaps_rollup") as f:
                for line in f:
                    if line.startswith("AnonHugePages:"):
                        ahp = int(line.split()[1]) * 1024
        except OSError:
            pass
        print("qx_cpu: %d layers, nibble gate/up %.2f GB, down copies %.2f GB (KINDI_QX_DNCOPY), node copies %.2f GB; "
              "process AnonHugePages %.2f GB" % (l, gb("qx_nib_bytes"), gb("qx_dncopy_bytes"), gb("qx_numa_bytes"),
                                                 ahp / 1e9), file=sys.stderr, flush=True)

    def forward(self, l, x, ids, weights):
        """x [T, K] float32, ids [T, n] int32, weights [T, n] float32 -> [T, K] float32 (routed experts only)"""
        x = np.ascontiguousarray(x, np.float32)
        ids = np.ascontiguousarray(ids, np.int32)
        weights = np.ascontiguousarray(weights, np.float32)
        T, n = ids.shape
        out = np.empty((T, self.K), np.float32)
        r = lib().qx_forward(self.layer[l], T, x.ctypes.data, n, ids.ctypes.data, weights.ctypes.data,
                             out.ctypes.data)
        if r < 0:
            raise ValueError("qx_cpu: layer %d call failed (%d)" % (l, r))
        return out

    def forward_parts(self, l, x, ids, skip, parts):
        """the value rows of the (token, slot) pairs not marked in skip [T, n] (uint8) into parts [T, n, K] float32
        (a writable C-contiguous array; the marked rows are left as they are)"""
        x = np.ascontiguousarray(x, np.float32)
        ids = np.ascontiguousarray(ids, np.int32)
        skip = np.ascontiguousarray(skip, np.uint8)
        T, n = ids.shape
        assert parts.flags.c_contiguous and parts.dtype == np.float32 and parts.shape == (T, n, self.K)
        r = lib().qx_forward_parts(self.layer[l], T, x.ctypes.data, n, ids.ctypes.data, skip.ctypes.data,
                                   parts.ctypes.data)
        if r < 0:
            raise ValueError("qx_cpu: layer %d call failed (%d)" % (l, r))
        return r

    def pool_profile(self, reset=True):
        """the pool's phase timers since the last call: per call (ms) the host's wall time and its wait for the last
        thread, and over the threads the mean / max of wake latency, work and barrier time"""
        L = lib()
        out = (ctypes.c_double * (4 * 128))()
        host = (ctypes.c_double * 3)()
        n = L.moe_pool_prof(out, host)
        if reset:
            L.moe_pool_prof_reset()
        calls = host[2]
        if not calls:
            return {}
        a = np.array(out[:4 * n]).reshape(n, 4)
        per = a[:, :3] / np.maximum(a[:, 3:4], 1)
        r = lambda v: round(float(v), 3)                 # noqa: E731
        return {"calls": int(calls), "wall_ms": r(host[0] / calls), "tail_ms": r(host[1] / calls),
                "wake_ms": r(per[1:, 0].mean()) if n > 1 else 0.0, "wake_max_ms": r(per[1:, 0].max()) if n > 1 else 0.0,
                "work_ms": r(per[:, 1].mean()), "work_max_ms": r(per[:, 1].max()),
                "barrier_ms": r(per[:, 2].mean()), "barrier_max_ms": r(per[:, 2].max()), "threads": n}

    def stats(self):
        a, s = (ctypes.c_longlong * 4)(), ctypes.c_double()
        lib().qx_stats(a, ctypes.byref(s))
        return {"calls": a[0], "experts": a[1], "pairs": a[2], "bytes": a[3], "seconds": s.value}
