"""DeepSeek-V4.1-Flash on the host CPUs: dsv41_cpu.c (the forward pass) + moe_cpu.c (the routed experts), built on
first use with gcc into KINDI_CPU_CACHE (default ~/.cache/kindi).

Placement (core lists like "0-19" and "20-23,32-39"):
  KINDI_MOE_CPUS0 / KINDI_MOE_CPUS1   expert pool, one thread per physical core of NUMA node 0 / 1
  KINDI_DSV_CPUS0 / KINDI_DSV_CPUS1   dense pool workers (default: the expert cores themselves, with the expert
                                      pool's idle workers at SCHED_IDLE; KINDI_DSV_PLACE=siblings: their
                                      hyper-thread siblings)
  KINDI_MOE_SHARE0 / KINDI_DSV_SHARE0 node 0's share of the rows (its share of the memory bandwidth)
The defaults fit the 2-socket development host: node 1 has half the memory channels, and its cores 24-31 belong to
a VM.

CLI:
  python -m kindi.dsv41 generate MODEL-00001-of-0000N.gguf --prompt TEXT [-n 64] [--chat] [--system TEXT]
  python -m kindi.dsv41 bench MODEL... [--ctx 2048,8192] [--gen 32]
  python -m kindi.dsv41 tokens MODEL... --prompts FILE.jsonl
"""
import argparse
import ctypes
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from .gguf import GGUF
from .dsv41_tokenizer import Tok

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = 129280
_lib = None

BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"


def gpu_lib(cache):
    """dsv41_gpu.cu built with nvcc (KINDI_NVCC, default /usr/local/cuda/bin/nvcc) for the card's architecture
    (KINDI_CUDA_ARCH, default sm_86); dsv41_cpu.c loads it with dlopen"""
    src = open(os.path.join(HERE, "dsv41_gpu.cu"), encoding="utf-8").read()
    hdr = open(os.path.join(HERE, "dsv41_gpu_fwd.cuh"), encoding="utf-8").read()
    arch = os.environ.get("KINDI_CUDA_ARCH", "sm_86")
    tag = hashlib.sha1((arch + src + hdr).encode()).hexdigest()[:12]
    so = os.path.join(cache, "dsv41_gpu_%s.so" % tag)
    if not os.path.exists(so):
        cu = os.path.join(cache, "dsv41_gpu_%s.cu" % tag)
        open(cu, "w", encoding="utf-8").write(src)
        nvcc = os.environ.get("KINDI_NVCC", "/usr/local/cuda/bin/nvcc")
        subprocess.check_call([nvcc, "-O3", "-arch=" + arch, "-shared", "-Xcompiler", "-fPIC", "-fmad=false",
                               "-I", HERE, "-o", so + ".tmp", cu])
        os.replace(so + ".tmp", so)
    return so


def cuda_devices():
    """[(device, name, compute capability major * 10 + minor, total bytes)] in CUDA's default order, or []"""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return []
    if cu.cuInit(0) != 0:
        return []
    n = ctypes.c_int()
    if cu.cuDeviceGetCount(ctypes.byref(n)) != 0:
        return []
    out = []
    for i in range(n.value):
        d, ma, mi, mem = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_size_t()
        name = ctypes.create_string_buffer(256)
        cu.cuDeviceGet(ctypes.byref(d), i)
        cu.cuDeviceGetName(name, 256, d)
        cu.cuDeviceGetAttribute(ctypes.byref(ma), 75, d)      # COMPUTE_CAPABILITY_MAJOR
        cu.cuDeviceGetAttribute(ctypes.byref(mi), 76, d)      # COMPUTE_CAPABILITY_MINOR
        cu.cuDeviceTotalMem_v2(ctypes.byref(mem), d)
        out.append((i, name.value.decode(), ma.value * 10 + mi.value, mem.value))
    return out


def gpu2_libs(cache):
    """expert cards: every CUDA device after device 0 (the main card) with sm_50+ and >= 4 GB gets its own copy of
    dsv41_gpu2.cu built for its architecture (KINDI_NVCC2, default /usr/local/cuda-12.4/bin/nvcc: CUDA 13 has no
    sm_5x). KINDI_DSV_GPU2=0 turns this off; KINDI_DSV_GPU2_DEVS=1,2 picks the devices."""
    want = os.environ.get("KINDI_DSV_GPU2_DEVS")
    devs = [d for d in cuda_devices()[1:] if d[2] >= 50 and d[3] >= 4 << 30]
    if want:
        devs = [d for d in cuda_devices() if str(d[0]) in want.split(",")]
    src = open(os.path.join(HERE, "dsv41_gpu2.cu"), encoding="utf-8").read()
    nvcc = os.environ.get("KINDI_NVCC2", "/usr/local/cuda-12.4/bin/nvcc")
    libs = []
    for dev, name, cc, mem in devs:
        arch = "sm_%d" % cc
        tag = hashlib.sha1((arch + src).encode()).hexdigest()[:12]
        so = os.path.join(cache, "dsv41_gpu2_%s.so" % tag)
        if not os.path.exists(so):
            cu = os.path.join(cache, "dsv41_gpu2_%s.cu" % tag)
            open(cu, "w", encoding="utf-8").write(src)
            subprocess.check_call([nvcc, "-O3", "-arch=" + arch, "-shared", "-Xcompiler", "-fPIC", "-fmad=false",
                                   "-Wno-deprecated-gpu-targets", "-o", so + ".tmp", cu, "-lcuda"])
            os.replace(so + ".tmp", so)
        own = os.path.join(cache, "dsv41_gpu2_%s_dev%d.so" % (tag, dev))   # one file per card: its own runtime state
        if not os.path.exists(own):
            shutil.copyfile(so, own + ".tmp")
            os.replace(own + ".tmp", own)
        libs.append((dev, own))
        print("dsv41: expert card %d: %s (sm_%d, %.1f GB)" % (dev, name, cc, mem / 2 ** 30), file=sys.stderr)
    return libs


def moe_stats():
    """the expert pool's cumulative counters: calls, u12 calls, CPU experts, pairs, expert bytes, cycles, TSC Hz"""
    a = (ctypes.c_longlong * 7)()
    lib().moe_stats(ctypes.addressof(a))
    return list(a)


def moe_phases():
    """the expert pool's per-thread phase cycles (moe_cpu.c moe_ph [128][6]), summed over the pool's threads:
    gate/up work, barrier 1, quantize, barrier 2, down work, gate/up kernels"""
    arr = (ctypes.c_ulonglong * (128 * 6)).in_dll(lib(), "moe_ph")
    n = len(_cpus(os.environ.get("KINDI_MOE_CPUS0", "0-19"))) + len(_cpus(os.environ.get("KINDI_MOE_CPUS1", "20-23,32-39")))
    return [sum(arr[t * 6 + k] for t in range(n)) for k in range(6)], n


def stats_line(kind, s0, s1, n_tok, decode_s, draft_s=0.0, steps=0, p0=None, p1=None):
    """KINDI_DS_STATS: one stderr line splitting a decode into CPU experts / draft / the rest (GPU dense + host work)"""
    d = [b - a for a, b in zip(s0, s1)]
    moe_s = d[5] / float(s1[6] or 1)
    if p0 is not None and p1 is not None:
        (a, n), (b, _) = p0, p1
        per = [(y - x) / float(s1[6] or 1) / max(n, 1) / max(d[0], 1) * 1e6 for x, y in zip(a, b)]
        print("dsv41 phases %s (us per call, mean over %d threads): gate/up %.0f, barrier1 %.0f, quantize %.1f, "
              "barrier2 %.0f, down %.0f; whole call %.0f" % (kind, n, per[0], per[1], per[2], per[3], per[4],
                                                            moe_s / max(d[0], 1) * 1e6), file=sys.stderr, flush=True)
    print("dsv41 stats %s: %d tokens, %d steps, decode %.2f s = CPU experts %.2f s (%.0f%%) + draft %.2f s + "
          "GPU dense and other %.2f s; expert calls %d (u12 %d), CPU experts per call %.1f, expert bytes per token %.2f GB"
          % (kind, n_tok, steps, decode_s, moe_s, 100.0 * moe_s / max(decode_s, 1e-9), draft_s,
             decode_s - moe_s - draft_s, d[0], d[1], d[2] / max(d[0], 1), d[4] / max(n_tok, 1) / 1e9),
          file=sys.stderr, flush=True)


def lib():
    global _lib
    if _lib is None:
        srcs = [open(os.path.join(HERE, f), encoding="utf-8").read() for f in ("dsv41_cpu.c", "moe_cpu.c")]
        cache = os.environ.get("KINDI_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "kindi")
        os.makedirs(cache, exist_ok=True)
        tag = hashlib.sha1(("fpc1" + "".join(srcs)).encode()).hexdigest()[:12]
        so = os.path.join(cache, "dsv41_cpu_%s.so" % tag)
        if not os.path.exists(so):
            objs = []
            for name, src in zip(("dsv41_cpu", "moe_cpu"), srcs):
                c = os.path.join(cache, "%s_%s.c" % (name, tag))
                open(c, "w", encoding="utf-8").write(src)
                # no implicit FMA contraction, so a code change does not move results by an ulp
                subprocess.check_call(["gcc", "-O3", "-march=native", "-pthread", "-fPIC", "-ffp-contract=off", "-c",
                                       "-o", c + ".o", c])
                objs.append(c + ".o")
            subprocess.check_call(["gcc", "-shared", "-pthread", "-o", so + ".tmp"] + objs + ["-lm", "-ldl"])
            os.replace(so + ".tmp", so)
        if os.environ.get("KINDI_DSV_GPU", "0") not in ("", "0") and not os.environ.get("KINDI_DSV_GPU_LIB"):
            os.environ["KINDI_DSV_GPU_LIB"] = gpu_lib(cache)
        if (os.environ.get("KINDI_DSV_GPU", "0") not in ("", "0", "1") and os.environ.get("KINDI_DSV_GPU2", "1") != "0"
                and not os.environ.get("KINDI_DSV_GPU2_LIBS")):
            cards = gpu2_libs(cache)
            if cards:
                os.environ["KINDI_DSV_GPU2_LIBS"] = ",".join(c[1] for c in cards)
                os.environ["KINDI_DSV_GPU2_DEVS"] = ",".join(str(c[0]) for c in cards)
        L = ctypes.CDLL(so)
        V, I, Fl, D = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_double
        Lo = ctypes.c_long
        L.dsv_add_tensor.argtypes = [ctypes.c_char_p, V, I, Lo, Lo, Lo]
        L.dsv_init.argtypes = [I, I, V, I, I, Fl, V, I, I, Fl, I]
        L.dsv_init.restype = I
        L.dsv_set_engram.argtypes = [V, V, V, V, I]
        L.dsv_forward.argtypes = [V, I, V, I]
        L.dsv_forward.restype = I
        L.dsv_forward2.argtypes = [V, I, V]
        L.dsv_forward2.restype = I
        L.dsv_set_flags.argtypes = [I]
        L.dsv_get_flags.restype = I
        L.dsv_pos.restype = I
        L.dsv_set_pos.argtypes = [I]
        L.dsv_dump_open.argtypes = [ctypes.c_char_p]
        L.dsv_times.argtypes = [V]
        L.dsv_bench_dense.argtypes = [I, I]
        L.dsv_map_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(Lo)]
        L.dsv_map_file.restype = V
        L.dsv_drop_file.argtypes = [V, Lo, ctypes.c_char_p]
        L.dsv_drop_file.restype = I
        L.dsv_move_file_pages.argtypes = [ctypes.c_char_p, I, I, Lo]
        L.dsv_move_file_pages.restype = Lo
        L.dsv_place_ranges.argtypes = [ctypes.c_char_p, I, V, V, V]
        L.dsv_place_ranges.restype = Lo
        L.dsv_interleave_file_pages.argtypes = [ctypes.c_char_p, I, I]
        L.dsv_interleave_file_pages.restype = Lo
        L.dsv_union_count.argtypes = []
        L.dsv_union_count.restype = Lo
        L.moe_stats.argtypes = [V]
        L.moe_stats.restype = None
        L.dsv_bench_dense.restype = D
        L.dsv_bench_attn.argtypes = [I, I]
        L.dsv_bench_attn.restype = D
        L.dsv_compare.argtypes = [V, I]
        L.dsv_compare.restype = I
        L.dsv_state_save.argtypes = [ctypes.c_char_p]
        L.dsv_state_save.restype = I
        L.dsv_state_load.argtypes = [ctypes.c_char_p]
        L.dsv_state_load.restype = I
        L.dsv_dspark_init.argtypes = [I]
        L.dsv_dspark_init.restype = I
        L.dsv_dspark_push.argtypes = [I, I, I]
        L.dsv_dspark_push.restype = I
        L.dsv_dspark_draft.argtypes = [I, I, V, V]
        L.dsv_dspark_draft.restype = I
        _lib = L
    return _lib


def _cpus(text):
    out = []
    for part in text.split(","):
        if part:
            a, _, b = part.partition("-")
            out += list(range(int(a), int(b or a) + 1))
    return out


def _siblings(cpus):
    """the hyper-thread sibling of each core (from sysfs; the core itself when there is none)"""
    out = []
    for c in cpus:
        try:
            s = open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c).read().strip()
            sib = [x for x in _cpus(s) if x != c]
            out.append(sib[0] if sib else c)
        except OSError:
            out.append(c)
    return out


def shards(path):
    """all shards of a split GGUF, given any one of them"""
    if "-of-" in path:
        base = path[:path.rindex("-", 0, path.rindex("-of-"))]
        return sorted(glob.glob(base + "-*-of-*.gguf"))
    return [path]


class Cancelled(Exception):
    """raised by Model.forward between prompt chunks when Model.cancel() returns True"""


def distribution(logits, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0):
    """the sampling distribution of the DeepSeek server (plain sampled decode and speculative sampling use this one
    function, so both give the same distribution): temperature, then top-k, then top-p (nucleus), then min-p
    (relative to the most likely token). temperature <= 0 gives the greedy one-hot distribution."""
    lg = np.asarray(logits, np.float64)
    p = np.zeros(lg.shape[0], np.float64)
    if temperature <= 0:
        p[int(np.argmax(lg))] = 1.0
        return p
    z = lg / float(temperature)
    keep = np.ones(lg.shape[0], bool)
    if top_k and 0 < top_k < lg.shape[0]:
        kth = np.partition(z, -top_k)[-top_k]
        keep &= z >= kth
    z = np.where(keep, z, -np.inf)
    e = np.exp(z - np.max(z))
    q = e / e.sum()
    if top_p < 1.0:
        order = np.argsort(-q, kind="stable")
        c = np.cumsum(q[order])
        n = int(np.searchsorted(c, top_p, side="left")) + 1       # the smallest prefix with mass >= top_p
        cut = np.zeros_like(keep)
        cut[order[:n]] = True
        q = np.where(cut, q, 0.0)
        q /= q.sum()
    if min_p > 0.0:
        q = np.where(q >= min_p * q.max(), q, 0.0)
        q /= q.sum()
    return q


def spec_accept(dists, draft, rng):
    """exact speculative sampling with deterministic drafts (the draft distribution is a point mass): draft token i is
    kept with probability p_i(draft_i); at the first rejection the replacement is drawn from p_i with draft_i removed
    (the residual max(p - q, 0), renormalised). dists: the target distributions after each prefix (len(draft) + 1 of
    them). Returns (number of kept drafts a, the next token: the replacement at a rejection, else a sample of
    dists[a])."""
    for i, d in enumerate(draft):
        pi = dists[i]
        if rng.random() < pi[d]:
            continue
        r = pi.copy()
        r[d] = 0.0
        s = r.sum()
        if s <= 0:                                   # p_i was a point mass on d: cannot be rejected
            continue
        return i, int(rng.choice(len(r), p=r / s))
    a = len(draft)
    return a, int(rng.choice(len(dists[a]), p=dists[a]))


class Snapshot(object):
    """a full engine state in a RAM file (/dev/shm), with the tokens it holds; drop() frees it"""
    def __init__(self, path, n, hist):
        self.path, self.n, self.hist = path, n, hist
        self.nbytes = os.path.getsize(path) if os.path.exists(path) else 0

    def drop(self):
        try:
            os.remove(self.path)
        except OSError:
            pass
        self.nbytes = 0


class Model(object):
    def __init__(self, path, ctx=8192, tmax=64, expert_copy=None, verbose=True):
        L = lib()
        files = shards(path)
        self.meta = GGUF(files[0]).kv
        expert_only = []
        t0 = time.time()
        for f in files:
            g = GGUF(f)
            size = ctypes.c_long(0)
            p = L.dsv_map_file(f.encode(), ctypes.byref(size))
            if not p:
                raise RuntimeError("dsv41: cannot map %s" % f)
            base = p + g.data_start
            for name, ne, ty, off in g.tensors:
                ne = list(ne) + [1] * (3 - len(ne))
                L.dsv_add_tensor(name.encode(), base + off, ty, ne[0], ne[1], ne[2])
            if g.tensors and all(n.endswith("_exps.weight") for n, _, _, _ in g.tensors):
                expert_only.append((f, p, size.value))
        m0 = _cpus(os.environ.get("KINDI_MOE_CPUS0", "0-19"))
        m1 = _cpus(os.environ.get("KINDI_MOE_CPUS1", "20-23,32-39"))
        # dense workers: on the expert pool's own cores ("same"; the calling thread is thread 0 of both pools) or on
        # their hyper-thread siblings ("siblings")
        same = os.environ.get("KINDI_DSV_PLACE", "same") == "same"
        if same:
            os.environ.setdefault("KINDI_MOE_IDLE", "1")
        d0 = _cpus(os.environ["KINDI_DSV_CPUS0"]) if "KINDI_DSV_CPUS0" in os.environ else (m0[1:] if same else _siblings(m0))
        d1 = _cpus(os.environ["KINDI_DSV_CPUS1"]) if "KINDI_DSV_CPUS1" in os.environ else (m1 if same else _siblings(m1))
        ms = float(os.environ.get("KINDI_MOE_SHARE0", "0.667"))
        ds = float(os.environ.get("KINDI_DSV_SHARE0", "0.667"))
        if expert_copy is None:
            # default: experts read in place from the page cache (fast start, the page cache is shared with other
            # processes); KINDI_MOE_COPY=1 copies them into node-split memory (faster, but the whole model no
            # longer fits the page cache next to the copy, so the expert-only shards are dropped from it)
            expert_copy = os.environ.get("KINDI_MOE_COPY", "0") != "0"
        mc = (ctypes.c_int * len(m0 + m1))(*(m0 + m1))
        dc = (ctypes.c_int * max(1, len(d0 + d1)))(*(d0 + d1))
        r = L.dsv_init(ctx, tmax, dc, len(d0), len(d1), ds, mc, len(m0), len(m1), ms, 1 if expert_copy else 0)
        if r:
            raise RuntimeError("dsv41: init failed (%d)" % r)
        if expert_copy and os.environ.get("KINDI_MOE_DROP_CACHE", "1") != "0":
            # the copied experts no longer need the page cache of the shards that hold only experts; dropping it
            # keeps the engram table (read at random, one row per n-gram) resident without memory pressure
            for f, p, size in expert_only:
                L.dsv_drop_file(p, size, f.encode())
        kv = self.meta
        tokmap = np.array(kv["deepseek41.engram.token_map"], np.int32)
        mult = np.array(kv["deepseek41.ple.layer_multipliers"], np.int64)
        primes = np.array(kv["deepseek41.ple.head_vocab_sizes"], np.int64)
        offs = np.array(kv["deepseek41.ple.head_offsets"], np.int64)
        self._eng = (tokmap, mult, primes, offs)
        L.dsv_set_engram(tokmap.ctypes.data, mult.ctypes.data, primes.ctypes.data, offs.ctypes.data,
                         int(kv["deepseek41.engram.pad_token_id"]))
        self.tok = Tok(kv["tokenizer.ggml.tokens"], kv["tokenizer.ggml.token_type"], kv["tokenizer.ggml.merges"])
        self.eos = self.tok.sid[EOS]
        self.ctx, self.tmax = ctx, tmax
        self.hist = []                                   # the tokens at positions 0 .. pos - 1 (prompt cache)
        self.logits = np.zeros(VOCAB, np.float32)
        if verbose:
            print("dsv41: loaded in %.1f s (dense cores %s / %s, expert cores %s / %s, experts %s)"
                  % (time.time() - t0, d0, d1, m0, m1, "copied" if expert_copy else "in place"), file=sys.stderr)

    @property
    def pos(self):
        return lib().dsv_pos()

    def reset(self):
        lib().dsv_reset()
        self.hist = []

    def rewind(self, pos):
        """continue from position pos (for speculation: drop rejected draft tokens). The caches past pos are
        rewritten by the next forward; a batch of tokens gives the same bits as the tokens one at a time, so a
        verification pass over k drafted tokens equals k single steps."""
        if not 0 <= pos <= self.pos:
            raise ValueError("dsv41: rewind to %d, position is %d" % (pos, self.pos))
        lib().dsv_set_pos(pos)
        del self.hist[pos:]

    def can_rewind(self, pos):
        """True when rewind(pos) is valid: an even position inside the reach of the window ring (tmax back from the
        current position). A state further back needs a saved state (checkpoint_save or state_snapshot)."""
        return 0 <= pos <= self.pos and pos % 2 == 0 and pos >= self.pos - self.tmax

    def state_snapshot(self, ram_dir=None):
        """the current state (positions 0 .. pos - 1, rounded down to an even position) as a Snapshot in host RAM"""
        ram_dir = ram_dir or os.environ.get("KINDI_DSV_SNAP_DIR", "/dev/shm/kindi-dsv41")
        os.makedirs(ram_dir, exist_ok=True)
        n = self.pos - self.pos % 2
        if n < self.pos:
            self.rewind(n)
        path = os.path.join(ram_dir, "%s-%d.kds" % (self._ckpt_key(self.hist[:n]), n))
        if not os.path.exists(path):
            if lib().dsv_state_save((path + ".tmp").encode()):
                raise RuntimeError("dsv41: snapshot failed")
            os.replace(path + ".tmp", path)
        return Snapshot(path, n, list(self.hist[:n]))

    def state_restore(self, snap):
        """go back to a Snapshot; returns its position"""
        p = lib().dsv_state_load(snap.path.encode())
        if p != snap.n:
            raise RuntimeError("dsv41: snapshot restore failed (%d)" % p)
        self.hist = list(snap.hist)
        return p

    def _check_cancel(self):
        """between prompt chunks: stop when the caller's cancel() says so (the client went away); the state stays
        consistent (positions 0 .. pos - 1 are complete)"""
        c = getattr(self, "cancel", None)
        if c is not None and c():
            raise Cancelled("cancelled at position %d" % self.pos)

    def forward(self, ids, all_logits=False, push=True):
        """feed ids at the next positions; returns the logits of the last token ([len(ids), VOCAB] if all_logits).
        With the DSpark drafter loaded, every chunk's positions go into its rings (push=False: the caller pushes the
        accepted ones, as speculation does)"""
        ids = np.ascontiguousarray(ids, np.int32)
        out = np.zeros((len(ids), VOCAB) if all_logits else VOCAB, np.float32)
        a0 = 0
        # prompt chunks: two full chunks at a time through the pipelined path (KINDI_PIPE=0: off)
        while (not all_logits and os.environ.get("KINDI_PIPE", "1") != "0" and len(ids) - a0 >= 2 * self.tmax
               and self.tmax > 8):
            last = len(ids) - a0 == 2 * self.tmax
            self._check_cancel()
            p0 = self.pos
            r = lib().dsv_forward2(ids[a0:].ctypes.data, self.tmax, out.ctypes.data if last else None)
            if r == 1:
                break
            if r:
                raise RuntimeError("dsv41: forward2 failed")
            self.hist.extend(int(x) for x in ids[a0:a0 + 2 * self.tmax])
            if push and getattr(self, "dspark", False):
                n = min(self.tmax, 128)
                lib().dsv_dspark_push(self.tmax - n, n, p0 + 2 * self.tmax - n)
            a0 += 2 * self.tmax
        for a in range(a0, len(ids), self.tmax):
            b = min(len(ids), a + self.tmax)
            dst = out[a:b] if all_logits else out
            if b - a > 16:
                self._check_cancel()
            p0 = self.pos
            r = lib().dsv_forward(ids[a:].ctypes.data, b - a, dst.ctypes.data if (all_logits or b == len(ids)) else None,
                                  1 if all_logits else 0)
            if r:
                raise RuntimeError("dsv41: forward failed (context full?)")
            self.hist.extend(int(x) for x in ids[a:b])
            if push and getattr(self, "dspark", False):
                n = min(b - a, 128)
                lib().dsv_dspark_push(b - a - n, n, p0 + b - a - n)
        return out

    def _ckpt_key(self, toks):
        return hashlib.sha1(np.asarray(toks, np.int32).tobytes()).hexdigest()

    def checkpoint_save(self, cache_dir):
        """save the current state (positions 0 .. pos - 1) to cache_dir, keyed by the hash of those tokens"""
        os.makedirs(cache_dir, exist_ok=True)
        n = self.pos - self.pos % 2                     # an even position (no pending 2:1 pooling partner)
        if n < self.pos:
            self.rewind(n)
        key = self._ckpt_key(self.hist[:n])
        path = os.path.join(cache_dir, "%s.kds" % key)
        if not os.path.exists(path):
            if lib().dsv_state_save((path + ".tmp").encode()):
                raise RuntimeError("dsv41: checkpoint save failed")
            os.replace(path + ".tmp", path)
            with open(os.path.join(cache_dir, "index.jsonl"), "a") as f:
                f.write(json.dumps({"n": n, "key": key}) + chr(10))
        return n

    def checkpoint_lookup(self, ids, cache_dir):
        """the longest saved prefix of ids: (length, path) or (0, None)"""
        idx = os.path.join(cache_dir, "index.jsonl")
        if not os.path.exists(idx):
            return 0, None
        best = (0, None)
        for line in open(idx):
            e = json.loads(line)
            if best[0] < e["n"] < len(ids) and self._ckpt_key(ids[:e["n"]]) == e["key"]:
                p = os.path.join(cache_dir, "%s.kds" % e["key"])
                if os.path.exists(p):
                    best = (e["n"], p)
        return best

    def prefill_cached(self, ids, cache_dir=None):
        """Prompt cache for agent turns: the tokens already in the caches are kept. The common prefix of ids and the
        current context is not computed again; the position is rewound to the end of that prefix (rounded down to an
        even position, so the 2:1 compressor needs no pooling partner from before; at most tmax back, the reach of the
        window ring) and only the rest is fed. Returns (logits of the last token, number of tokens computed). The
        caches are indexed by position, so the result is the one of a full prefill."""
        ids = list(ids)
        c = 0
        n = min(len(ids) - 1, len(self.hist))           # the last token is always fed (its logits are needed)
        while c < n and ids[c] == self.hist[c]:
            c += 1
        c -= c % 2
        if c < self.pos - self.tmax:
            c = 0                                       # beyond the window ring: compute from the start
        if cache_dir:                                   # a saved state with a longer common prefix
            n, path = self.checkpoint_lookup(ids, cache_dir)
            if n > c:
                p = lib().dsv_state_load(path.encode())
                if p == n:
                    self.hist = list(ids[:n])
                    c = n
        if c == 0:
            self.reset()
        elif c < self.pos:
            self.rewind(c)
        return self.forward(ids[c:]), len(ids) - c

    def dspark_load(self, d):
        """the DSpark drafter converted by kindi.dsv41_dspark (d/kindi-dspark.bin + .json); needs KINDI_DSV_GPU=2"""
        L = lib()
        idx = json.load(open(os.path.join(d, "kindi-dspark.bin.json")))
        size = ctypes.c_long(0)
        p = L.dsv_map_file(os.path.join(d, "kindi-dspark.bin").encode(), ctypes.byref(size))
        if not p:
            raise RuntimeError("dsv41: cannot map the DSpark file")
        for name, e in idx.items():
            shp = list(e["shape"]) + [1, 1, 1]
            L.dsv_add_tensor(("dspark." + name).encode(), p + e["off"], {"q8": 8, "f32": 0, "mxfp4": 39}[e["kind"]],
                             shp[0], shp[1], shp[2])
        cfg = json.load(open(os.path.join(d, "config.json")))
        if L.dsv_dspark_init(int(cfg.get("dspark_noise_token_id", 128799))):
            raise RuntimeError("dsv41: DSpark init failed")
        self.dspark = True

    def generate_dspark(self, ids, n, conf_min=0.0, kmax=5, stop_eos=True, stats=None, auto=False, cal=(1.0, 0.0),
                        ms=(45.0, 2.0, 50.0), lg=None, on_tokens=None, temperature=0.0, top_p=1.0, top_k=0,
                        min_p=0.0, rng=None):
        """greedy with DSpark drafts: after each step the drafter proposes 5 tokens; the drafts whose confidence logit
        is >= conf_min (a prefix, at most kmax) go through one forward with the next token; drafts are kept up to the
        first one that differs from the greedy choice, the position is rewound past the rest, and only the kept
        positions go into the drafter's rings. The output equals generate(). Returns (new ids, prefill s, decode s).
        lg: the logits of the prompt's last token when the prompt is already fed (ids is then ignored, e.g. after
        prefill_cached); on_tokens(list): called with the tokens of each step, a true result stops the generation.
        temperature > 0: exact speculative sampling (spec_accept) from distribution(temperature, top_p, top_k, min_p);
        the output has the distribution of plain sampled decoding. rng: a numpy Generator (default: a fresh one)."""
        L = lib()
        sample = temperature > 0
        if sample and rng is None:
            rng = np.random.default_rng()

        def dist(row):
            return distribution(row, temperature, top_p, top_k, min_p)
        forced = None                                    # the replacement token after a rejected draft
        t0 = time.time()
        if lg is None:
            lg = self.forward(ids)
        t1 = time.time()
        out = []
        st = stats if stats is not None else {}
        for key in ("steps", "drafted", "accepted", "draft_s"):
            st.setdefault(key, 0)
        tok = np.zeros(8, np.int32)
        conf = np.zeros(8, np.float32)
        union = (1.0, 1.69, 2.28, 2.80, 3.28, 3.73, 4.15, 4.54)   # expert bytes of m consecutive tokens (route_spec.py)
        # KINDI_DS_UNION (default 1; 0 = the fixed table): the expert term uses the measured CPU expert union of recent verify steps instead of the
        # fixed average table. ub[m] = running mean of the distinct experts read on the CPU (main layers, card-resident
        # experts excluded) by a forward of m positions; the factor for m positions is ub[m] / (single-position union,
        # estimated from every observed m through the table). Only the choice of k changes; every draft is still
        # verified, so the output is unchanged (greedy) and the sampled distribution is unchanged.
        uni = os.environ.get("KINDI_DS_UNION", "1") != "0" and hasattr(L, "dsv_union_count")   # default on (11-11: +1.9% DSpark, exact)
        ub = {}

        def ufac(m):
            if not uni or m not in ub:
                return union[m - 1]
            est1 = [ub[j] / union[j - 1] for j in ub]
            u1 = ub[1] if 1 in ub else sum(est1) / len(est1)
            return ub[m] / u1 if u1 > 0 else union[m - 1]

        def cost(k):
            return ms[0] + ms[1] * k + ms[2] * ufac(k + 1)
        ng = tuple(float(x) for x in os.environ["KINDI_DS_NGRAM"].split(",")) if os.environ.get("KINDI_DS_NGRAM") else None
        want_stats = os.environ.get("KINDI_DS_STATS", "0") != "0"
        s_start = moe_stats() if want_stats else None
        ph_start = moe_phases() if want_stats else None
        draft0, steps0 = st["draft_s"], st["steps"]
        while len(out) < n:
            if forced is not None:
                y, forced = forced, None
            elif sample:
                d0 = dist(lg)
                y = int(rng.choice(len(d0), p=d0))
            else:
                y = int(np.argmax(lg))
            out.append(y)
            if (stop_eos and y == self.eos) or len(out) >= n:
                if on_tokens:
                    on_tokens([y])
                break
            pos_last = self.pos - 1
            td = time.time()
            if L.dsv_dspark_draft(pos_last, y, tok.ctypes.data, conf.ctypes.data):
                raise RuntimeError("dsv41: draft failed")
            st["draft_s"] += time.time() - td
            kcap = min(kmax, n - len(out), self.tmax - 1)
            # prompt lookup next to the drafter (KINDI_DS_NGRAM="bonus,strong,c0"): where the continuation of the
            # latest earlier occurrence of the context's longest suffix agrees with the drafts, their confidence rises
            # by bonus; when it disagrees from the first draft on, a match of >= strong tokens replaces the drafts if
            # the first draft's confidence is below c0. Exact either way (every draft is verified).
            ngc, use_ng = None, False
            if ng:
                cand, mlen = self._ngram(y, kcap)
                agree = 0
                while agree < len(cand) and agree < kcap and cand[agree] == int(tok[1 + agree]):
                    agree += 1
                conf_used = conf.copy()
                conf_used[:agree] += ng[0]
                use_ng = agree == 0 and len(cand) > 0 and mlen >= ng[1] and conf[0] < ng[2]
                st["ng_found"] = st.get("ng_found", 0) + (len(cand) > 0)
                st["ng_agree"] = st.get("ng_agree", 0) + agree
                ngc = cand
            else:
                conf_used = conf
            if auto:
                # expected tokens per ms: E(k) = 1 + sum_j prod_{i<=j} p_i with p_i = sigmoid(a * conf_i + b);
                # cost(k) = gpu + extra per position + experts x union factor of k + 1 positions (+ the draft, paid anyway)
                best, k, e, pr = 1.0 / cost(0), 0, 1.0, 1.0
                for j in range(kcap):
                    pr *= 1.0 / (1.0 + np.exp(-(cal[0] * conf_used[j] + cal[1])))
                    e += pr
                    if e / cost(j + 1) > best:
                        best, k = e / cost(j + 1), j + 1
            else:
                k = 0
                while k < kcap and conf[k] >= conf_min:
                    k += 1
            draft = [int(x) for x in tok[1:1 + k]]
            if use_ng:
                draft = list(ngc[:kcap])
                st["ng_steps"] = st.get("ng_steps", 0) + 1
            p0 = self.pos
            u_before = L.dsv_union_count() if uni else 0
            rows = self.forward([y] + draft, all_logits=True, push=False)
            if uni:
                m_ = len(draft) + 1
                u_ = float(L.dsv_union_count() - u_before)
                ub[m_] = u_ if m_ not in ub else 0.85 * ub[m_] + 0.15 * u_
                st["union_experts"] = st.get("union_experts", 0) + u_
            a = 0
            if sample:
                a, nxt = spec_accept([dist(rows[i]) for i in range(len(draft) + 1)], draft, rng)
                if a < len(draft):
                    forced = nxt                         # the residual sample replaces the rejected draft
            else:
                while a < len(draft) and int(np.argmax(rows[a])) == draft[a]:
                    a += 1
            st["steps"] += 1; st["drafted"] += len(draft); st["accepted"] += a
            if use_ng:
                st["ng_accepted"] = st.get("ng_accepted", 0) + a
            st.setdefault("conf_log", []).extend((float(conf[i]), int(i < a)) for i in range(len(draft)))
            L.dsv_dspark_push(0, 1 + a, p0)
            out.extend(draft[:a])
            if a < len(draft):
                self.rewind(p0 + 1 + a)
            lg = rows[a]
            if stop_eos and self.eos in draft[:a]:
                out = out[:out.index(self.eos) + 1]
                if on_tokens:
                    on_tokens([y] + draft[:draft.index(self.eos) + 1])
                break
            if on_tokens and on_tokens([y] + draft[:a]):
                break
        if want_stats:
            stats_line("dspark", s_start, moe_stats(), len(out[:n]), time.time() - t1, st["draft_s"] - draft0,
                       st["steps"] - steps0, ph_start, moe_phases())
        return out[:n], t1 - t0, time.time() - t1

    def _ngram(self, y, kmax, lmax=8, lmin=3):
        """prompt lookup: the tokens after the latest earlier occurrence of the longest suffix (lmin..lmax tokens) of
        the context + y; returns (continuation of up to kmax tokens, suffix length) or ([], 0)"""
        h = np.asarray(self.hist[-4096:] + [y], np.int32)
        for L_ in range(min(lmax, len(h) - 1), lmin - 1, -1):
            pat = h[-L_:]
            win = np.lib.stride_tricks.sliding_window_view(h[:-1], L_)
            hit = np.nonzero((win == pat).all(axis=1))[0]
            if len(hit):
                i = int(hit[-1]) + L_
                cont = [int(x) for x in h[i:i + kmax]]
                if cont:
                    return cont, L_
        return [], 0

    def generate(self, ids, n, stop_eos=True, on_token=None, gaps=None):
        """greedy; returns (new ids, prefill seconds, decode seconds); gaps (a list) receives top1 - top2 logit"""
        t0 = time.time()
        lg = self.forward(ids)
        t1 = time.time()
        out = []
        want_stats = os.environ.get("KINDI_DS_STATS", "0") != "0"
        s_start = moe_stats() if want_stats else None
        ph_start = moe_phases() if want_stats else None
        for _ in range(n):
            nxt = int(np.argmax(lg))
            out.append(nxt)
            if gaps is not None:
                gaps.append(float(lg[nxt] - np.max(np.delete(lg, nxt))))
            if on_token:
                on_token(nxt)
            if stop_eos and nxt == self.eos:
                break
            if len(out) == n:
                break
            lg = self.forward([nxt])
        if want_stats:
            stats_line("plain", s_start, moe_stats(), len(out), time.time() - t1, 0.0, len(out), ph_start, moe_phases())
        return out, t1 - t0, time.time() - t1

    def generate_spec(self, ids, n, k=4, ngram=8, min_match=3, stop_eos=True, stats=None):
        """greedy with prompt-lookup drafts, gated: a draft is made only when the context's last min_match..ngram tokens
        occurred before (longest match first); the drafted tokens are the ones that followed that occurrence. The
        next token plus the drafts go through one forward; drafts are kept up to the first one that differs from the
        greedy choice and the position is rewound past the rest. A rejected draft costs a whole verify step (the
        union of the drafts' experts), so the draft length adapts: +1 after a fully accepted draft, halved after a
        miss, and no draft on the step after a miss at the first drafted token. A batch gives the same bits as single
        steps, so the output equals generate(). Returns (new ids, prefill s, decode s); stats gets steps / drafted /
        accepted."""
        t0 = time.time()
        lg = self.forward(ids)
        t1 = time.time()
        ctx = list(ids)
        out = []
        st = stats if stats is not None else {}
        for key in ("steps", "drafted", "accepted"):
            st.setdefault(key, 0)
        dl, cool = k - 1, 0
        while len(out) < n:
            nxt = int(np.argmax(lg))
            out.append(nxt)
            ctx.append(nxt)
            if (stop_eos and nxt == self.eos) or len(out) >= n:
                break
            draft = []
            room = min(dl, n - len(out), self.tmax - 1)
            if cool:
                cool -= 1
            elif room > 0:
                for m in range(min(ngram, len(ctx) - 1), min_match - 1, -1):
                    key = ctx[-m:]
                    for j in range(len(ctx) - m - 1, -1, -1):        # latest earlier occurrence
                        if ctx[j:j + m] == key:
                            draft = ctx[j + m:j + m + room]
                            break
                    if draft:
                        break
            if not draft:
                lg = self.forward([nxt])
                st["steps"] += 1
                continue
            p0 = self.pos
            rows = self.forward([nxt] + draft, all_logits=True)
            a = 0
            while a < len(draft) and int(np.argmax(rows[a])) == draft[a]:
                a += 1
            st["steps"] += 1; st["drafted"] += len(draft); st["accepted"] += a
            out.extend(draft[:a])
            ctx.extend(draft[:a])
            if a < len(draft):
                self.rewind(p0 + 1 + a)
                dl = max(1, dl // 2)
                if a == 0:
                    cool = 1
            else:
                dl = min(k - 1, dl + 1)
            lg = rows[a]
            if stop_eos and self.eos in draft[:a]:
                out = out[:out.index(self.eos) + 1]
                break
        return out[:n], t1 - t0, time.time() - t1

    def times(self):
        t = np.zeros(13, np.float64)
        lib().dsv_times(t.ctypes.data)
        d = dict(zip(("dense", "moe", "attn", "idx", "total", "mix", "post", "engram", "compress", "qkv", "oquant",
                      "route", "x7"), t.tolist()))
        d["other"] = d["total"] - d["dense"] - d["moe"] - d["attn"] - d["idx"]
        return d


def chat_prompt(user, system=None, thinking=False):
    """DeepSeek-V4.1 chat format (one user turn, no tools)"""
    s = BOS
    if system:
        s += "<｜System｜>" + system
    return s + "<｜User｜>" + user + "<｜Assistant｜>" + ("<think>" if thinking else "</think>")


def _load_avg():
    try:
        return open("/proc/loadavg").read().split()[0]
    except OSError:
        return "?"


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data", "dsv41")
# the gate prompts and the reference continuations (gate_prompts.jsonl, gate_ref.txt) are not in the repository:
# they come from real agent sessions. KINDI_DSV_GATE names their directory.
GATE_DIR = os.environ.get("KINDI_DSV_GATE", os.path.expanduser("~/.cache/kindi/dsv41-gate"))


def check_sha(model_path, sums=None):
    """sha256 of every shard against the published list (tests/data/dsv41/SHA256SUMS-*.txt); returns the bad names"""
    sums = sums or os.path.join(DATA_DIR, "SHA256SUMS-UD-Q4_K_XL.txt")
    want = {}
    for line in open(sums):
        f = line.split()
        if len(f) >= 2:
            want[f[1]] = f[0]
    bad = []
    for path in shards(model_path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(64 << 20), b""):
                h.update(blk)
        name = os.path.basename(path)
        ok = want.get(name) == h.hexdigest()
        print("%s %s" % ("OK " if ok else "BAD", name), file=sys.stderr, flush=True)
        if not ok:
            bad.append(name)
    return bad


VERIFY_FLAGS = 160          # F_LLAMA_FA | F_F16_CACHE: the numerics of the reference build the gate data comes from


def verify(a, model, max_tie=0.75, min_agree=0.985):
    """the release gate: teacher-forced on 20 prompts x 64 tokens against the reference implementation's greedy
    continuations (GATE_DIR: gate_prompts.jsonl, gate_ref.txt "GEN tok:margin ..."). It runs with the reference's
    numerics (VERIFY_FLAGS, unless --flags is given): with Kindi's default numerics (DeepSeek's own FP8 / FP4
    roundings) the comparison mixes two rounding schemes. PASS needs >= 98.5% top-1 agreement and every disagreement a
    near-tie (the reference's own top-1 margin <= max_tie). The bar is calibrated on the accepted 11-03 gates
    (1265 to 1273 of 1280, near-ties only); 0.6.0 measured 1266/1280, largest margin 0.623 (11-10).
    Prints one line per prompt and the verdict; returns the exit code (0 = PASS)."""
    prompts = a.prompts or os.path.join(GATE_DIR, "gate_prompts.jsonl")
    ref = a.ref or os.path.join(GATE_DIR, "gate_ref.txt")
    refs = [[(int(x.split(":")[0]), float(x.split(":")[1])) for x in line.split()[1:]]
            for line in open(ref) if line.startswith("GEN")]
    tot = agree = 0
    worst = 0.0
    t0 = time.time()
    for j, r in zip(map(json.loads, open(prompts, encoding="utf-8")), refs):
        model.reset()
        toks = [t for t, _ in r]
        lg = model.forward(model.tok.encode(j["text"]))[None]
        if len(toks) > 1:
            lg = np.concatenate([lg, model.forward(toks[:-1], all_logits=True)])
        ok = 0
        for p, (t, rm) in enumerate(r):
            if int(np.argmax(lg[p])) == t:
                ok += 1
            else:
                worst = max(worst, rm)
        tot += len(r)
        agree += ok
        print("prompt %d: %d/%d" % (j["id"], ok, len(r)), file=sys.stderr, flush=True)
    passed = tot > 0 and agree >= min_agree * tot and worst <= max_tie
    print("dsv41 verify: %s - %d/%d top-1 agree (%.2f%%), largest reference margin at a disagreement %.3f "
          "(near-tie limit %.2f), %.0f s" % ("PASS" if passed else "FAIL", agree, tot, 100.0 * agree / max(tot, 1),
                                             worst, max_tie, time.time() - t0))
    return 0 if passed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kindi.dsv41")
    ap.add_argument("cmd", choices=["generate", "bench", "tokens", "bench-dense", "teacher", "verify", "numa", "gpucmp",
                                    "specbench", "cachetest"])
    ap.add_argument("--spec", type=int, default=4, help="specbench: positions per verify step (1 + drafts)")
    ap.add_argument("--dspark", default=None, help="specbench: DSpark drafter directory (converted)")
    ap.add_argument("--conf", type=float, default=0.0, help="specbench --dspark: draft while confidence logit >= this")
    ap.add_argument("--auto", default=None, help="specbench --dspark: choose k by expected tokens per ms; "
                    "calibration a,b of p = sigmoid(a * conf + b)")
    ap.add_argument("model")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--system", default=None)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the chat format")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--ctx", default="8192")
    ap.add_argument("--tmax", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--ids", default=None, help="file of token ids (instead of --prompt)")
    ap.add_argument("--range", default=None, help="prompt ids a-b (with --prompts)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None, help="teacher: reference continuations, one line per prompt: "
                    "GEN tok:margin ...")
    ap.add_argument("--dump", default=None, help="write per-layer tensors to this file")
    ap.add_argument("--flags", type=int, default=None)
    ap.add_argument("--sha", action="store_true", help="verify: also check every shard's sha256 first (slow)")
    ap.add_argument("--free1", default="12", help="numa: GB to keep free on node 1")
    ap.add_argument("--interleave", action="store_true", help="numa: spread the expert shards 2:1 over the nodes")
    ap.add_argument("--rowsplit", action="store_true", help="numa: each expert's first rows on node 0, the rest on node 1 "
                    "(the rows each node's threads compute; share KINDI_MOE_SHARE0)")
    a = ap.parse_args(argv)
    ctxs = [int(x) for x in a.ctx.split(",")]
    if a.cmd == "numa":
        # make room on NUMA node 1 for the node-bound dense weights: move page-cache pages of the model (largest
        # shard first) to node 0, until node 1 has --free1 GB free
        def free1():
            for line in open("/sys/devices/system/node/node1/meminfo"):
                if "MemFree" in line:
                    return int(line.split()[-2]) * 1024
        need = int(float(a.free1) * 2**30)
        if a.rowsplit:
            share0 = float(os.environ.get("KINDI_MOE_SHARE0", "0.667"))
            gu_rows = int(round(2304 // 4 * share0)) * 4      # the split of moe_layer_direct (lroundf of the groups)
            dn_rows = int(round(5120 // 4 * share0)) * 4
            for f in shards(a.model):
                g = GGUF(f)
                offs, lens, nodes = [], [], []
                for name, ne, ty, off in g.tensors:
                    if not name.endswith("_exps.weight"):
                        continue
                    cols, rows, ne_ = ne[0], ne[1], ne[2]
                    rb = cols // 32 * 17
                    split = dn_rows if "down" in name else gu_rows
                    base = g.data_start + off
                    for e in range(ne_):
                        s0 = base + e * rows * rb
                        offs += [s0, s0 + split * rb]
                        lens += [split * rb, (rows - split) * rb]
                        nodes += [0, 1]
                if offs:
                    t0 = time.time()
                    o = np.array(offs, np.int64); ln = np.array(lens, np.int64); nd = np.array(nodes, np.int32)
                    moved = lib().dsv_place_ranges(f.encode(), len(offs), o.ctypes.data, ln.ctypes.data, nd.ctypes.data)
                    print("%s: %.1f GB placed by rows in %.0f s" % (os.path.basename(f), moved / 2**30, time.time() - t0),
                          file=sys.stderr, flush=True)
        if a.interleave:
            # expert-only shards: 2 of 3 pages on node 0, 1 of 3 on node 1 (the bandwidth ratio of the two nodes)
            for f in shards(a.model):
                g = GGUF(f)
                if g.tensors and all(n.endswith("_exps.weight") for n, _, _, _ in g.tensors):
                    t0 = time.time()
                    lib().dsv_interleave_file_pages(f.encode(), 3, 1)
                    print("%s: interleaved in %.0f s" % (os.path.basename(f), time.time() - t0), file=sys.stderr)
        for f in sorted(shards(a.model), key=os.path.getsize, reverse=True):
            if free1() >= need:
                break
            moved = lib().dsv_move_file_pages(f.encode(), 1, 0, need - free1() + (1 << 28))
            print("%s: moved %.1f GB to node 0" % (os.path.basename(f), moved / 2**30), file=sys.stderr)
        print("node 1 free: %.1f GB" % (free1() / 2**30), file=sys.stderr)
        return
    if a.cmd == "tokens":
        m = GGUF(shards(a.model)[0]).kv
        tok = Tok(m["tokenizer.ggml.tokens"], m["tokenizer.ggml.token_type"], m["tokenizer.ggml.merges"])
        for line in open(a.prompts, encoding="utf-8"):
            j = json.loads(line)
            print(json.dumps({"id": j["id"], "tokens": tok.encode(j["text"])}))
        return
    if a.cmd == "verify" and a.sha:
        bad = check_sha(a.model)
        if bad:
            print("dsv41 verify: FAIL - %d shard(s) do not match the published sha256" % len(bad))
            return 1
    model = Model(a.model, ctx=max(ctxs) + a.n + 64, tmax=a.tmax)
    if a.flags is not None:
        lib().dsv_set_flags(a.flags)
    elif a.cmd == "verify":
        lib().dsv_set_flags(VERIFY_FLAGS)
    if a.dump:
        lib().dsv_dump_open(a.dump.encode())
    if a.cmd == "verify":
        return verify(a, model)
    if a.cmd == "teacher":
        # feed the reference's greedy continuation; at every position compare Kindi's top-1 with the reference token
        refs = [[(int(x.split(":")[0]), float(x.split(":")[1])) for x in line.split()[1:]]
                for line in open(a.ref) if line.startswith("GEN")]
        fo = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
        for j, ref in zip(map(json.loads, open(a.prompts, encoding="utf-8")), refs):
            model.reset()
            ids = model.tok.encode(j["text"])
            toks = [t for t, _ in ref]
            first = model.forward(ids)
            lg = first[None]
            if len(toks) > 1:
                lg = np.concatenate([lg, model.forward(toks[:-1], all_logits=True)])
            rows = []
            for p, (t, rmargin) in enumerate(ref):
                k = int(np.argmax(lg[p]))
                srt = np.sort(lg[p])
                rows.append({"pos": p, "ref": t, "kindi": k, "kindi_margin": round(float(srt[-1] - srt[-2]), 4),
                             "ref_margin": rmargin, "ref_tok_deficit": round(float(lg[p][k] - lg[p][t]), 4)})
            bad = [r for r in rows if r["ref"] != r["kindi"]]
            fo.write(json.dumps({"id": j["id"], "n": len(rows), "agree": len(rows) - len(bad), "mismatch": bad,
                                 "min_kindi_margin": min(r["kindi_margin"] for r in rows)}) + "\n")
            fo.flush()
            print("prompt %d: %d/%d positions agree" % (j["id"], len(rows) - len(bad), len(rows)), file=sys.stderr)
        return
    if a.cmd == "specbench" and a.dspark:
        model.dspark_load(a.dspark)
    if a.cmd == "specbench":
        # each prompt: plain greedy, then prompt-lookup speculation from the same start; the tokens must be equal
        p0, p1 = (int(x) for x in a.range.split("-")) if a.range else (0, 1 << 30)
        tot = {"plain_s": 0.0, "spec_s": 0.0, "tokens": 0}
        for j in map(json.loads, open(a.prompts, encoding="utf-8")):
            if not p0 <= j["id"] <= p1:
                continue
            ids = model.tok.encode(j["text"])
            model.reset()
            g1, _, d1 = model.generate(ids, a.n, stop_eos=False)
            model.reset()
            st = {}
            if a.dspark:
                g2, _, d2 = model.generate_dspark(ids, a.n, conf_min=a.conf, kmax=a.spec, stop_eos=False, stats=st,
                                                  auto=a.auto is not None,
                                                  cal=tuple(float(x) for x in (a.auto or "1,0").split(",")))
                if os.environ.get("KINDI_DS_LOG"):
                    with open(os.environ["KINDI_DS_LOG"], "a") as fl:
                        fl.write(json.dumps({"id": j["id"], "conf_acc": st.get("conf_log", [])}) + chr(10))
            else:
                g2, _, d2 = model.generate_spec(ids, a.n, k=a.spec, stop_eos=False, stats=st)
            same = g1 == g2
            tot["plain_s"] += d1; tot["spec_s"] += d2; tot["tokens"] += len(g1) - 1
            print("prompt %d: %s, plain %.2f tok/s, spec %.2f tok/s, %d steps, accepted %d of %d drafted, draft %.1f ms/step%s"
                  % (j["id"], "same tokens" if same else "TOKENS DIFFER", (len(g1) - 1) / d1, (len(g2) - 1) / d2,
                     st["steps"], st["accepted"], st["drafted"], 1e3 * st.get("draft_s", 0) / max(st["steps"], 1),
                     (", conf/accept " + " ".join("%.1f:%d" % c for c in st.get("conf_log", [])[:12])) if st.get("conf_log") else ""),
                  file=sys.stderr, flush=True)
        print("total: plain %.2f tok/s, spec %.2f tok/s" % (tot["tokens"] / tot["plain_s"], tot["tokens"] / tot["spec_s"]),
              file=sys.stderr)
        return
    if a.cmd == "cachetest":
        # an agent-like sequence of turns: each prompt extends the previous one (+ assistant reply + tool output), one
        # diverges near the end; the cached path must give the same logits as a cold prefill, and computes fewer tokens
        js = [json.loads(l) for l in open(a.prompts, encoding="utf-8")][:2]
        base = model.tok.encode(js[0]["text"])
        more = model.tok.encode(js[1]["text"])[:300]
        turns = [base, base + more[:150], base + more[:300], base + more[:250] + more[:40]]
        for i, ids in enumerate(turns):
            t0 = time.time()
            lg, n = model.prefill_cached(ids)
            dt = time.time() - t0
            hist = list(model.hist)
            model.reset()
            t1 = time.time()
            ref = model.forward(ids)
            dt_cold = time.time() - t1
            same = np.array_equal(lg, ref)
            model.reset(); model.forward(hist)            # back to the cached state for the next turn
            print("turn %d: %d tokens, computed %d (%.2f s, %.0f tok/s effective) vs cold %.2f s; logits %s"
                  % (i, len(ids), n, dt, len(ids) / dt, dt_cold, "IDENTICAL" if same else "DIFFER"), file=sys.stderr, flush=True)
        # checkpoint: the same continuation from the in-memory state and from a state restored from disk (the same
        # chunking, so the logits must be identical), and the argmax against a cold prefill
        import shutil
        cdir = os.path.join(os.path.expanduser("~"), ".cache", "kindi", "dsv41-ckpt-test")
        shutil.rmtree(cdir, ignore_errors=True)
        ids = turns[2] + more[:20]
        model.reset(); model.forward(turns[2])
        t0 = time.time(); n = model.checkpoint_save(cdir); ts = time.time() - t0
        mem = model.forward(ids[n:])                       # in memory, from position n
        model.reset()
        t0 = time.time(); lg, nc = model.prefill_cached(ids, cache_dir=cdir); dt = time.time() - t0
        model.reset(); ref = model.forward(ids)
        sz = sum(os.path.getsize(os.path.join(cdir, f)) for f in os.listdir(cdir))
        print("checkpoint: saved %d positions (%.0f MB) in %.2f s; restored + %d tokens in %.2f s; restored vs in-memory logits %s;"
              " argmax restored %d cold %d (rel diff %.1e)" % (n, sz / 1e6, ts, nc, dt, "IDENTICAL" if np.array_equal(lg, mem) else "DIFFER",
              int(np.argmax(lg)), int(np.argmax(ref)), float(np.linalg.norm(lg - ref) / np.linalg.norm(ref))), file=sys.stderr, flush=True)
        return
    if a.cmd == "gpucmp":
        # the CPU path and the GPU path (KINDI_DSV_GPU=2) on the same tokens: a prompt chunk, then -n decode steps
        ids = model.tok.encode(chat_prompt(a.prompt or "What is the capital of France? Answer in one short sentence."))
        arr = np.ascontiguousarray(ids, np.int32)
        nxt = lib().dsv_compare(arr.ctypes.data, len(ids))
        for _ in range(a.n):
            one = np.array([nxt], np.int32)
            nxt = lib().dsv_compare(one.ctypes.data, 1)
        return
    if a.cmd == "bench-dense":
        for T in (1, 4, 16, 64):
            if T <= a.tmax:
                lib().dsv_bench_dense(5, T)
        return
    if a.cmd == "generate":
        if a.prompts:
            p0, p1 = (int(x) for x in a.range.split("-")) if a.range else (0, 1 << 30)
            fo = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
            for line in open(a.prompts, encoding="utf-8"):
                j = json.loads(line)
                if not p0 <= j["id"] <= p1:
                    continue
                model.reset()
                ids = model.tok.encode(j["text"])
                gaps = []
                out, tp, tg = model.generate(ids, a.n, gaps=gaps)
                fo.write(json.dumps({"id": j["id"], "prompt_tokens": len(ids), "prefill_s": round(tp, 2),
                                     "gen_s": round(tg, 2), "tokens": out, "gaps": [round(g, 4) for g in gaps],
                                     "text": model.tok.decode(out),
                                     "load": _load_avg()}, ensure_ascii=False) + "\n")
                fo.flush()
                print("prompt %d: %d tokens, prefill %.1f s (%.1f tok/s), %d generated in %.1f s (%.2f tok/s)"
                      % (j["id"], len(ids), tp, len(ids) / tp, len(out), tg, max(len(out) - 1, 1) / tg),
                      file=sys.stderr)
            return
        if a.ids:
            ids = [int(x) for x in open(a.ids).read().split()]
        else:
            ids = model.tok.encode(chat_prompt(a.prompt, a.system, a.think) if a.chat else a.prompt)

        def show(t):
            sys.stdout.write(model.tok.decode_bytes(t).decode("utf-8", errors="replace"))
            sys.stdout.flush()
        out, tp, tg = model.generate(ids, a.n, on_token=show)
        print("\nTOKENS", " ".join(map(str, out)), file=sys.stderr)
        print("\n[%d prompt tokens: prefill %.2f s (%.1f tok/s); %d tokens in %.2f s (%.2f tok/s); load %s]"
              % (len(ids), tp, len(ids) / tp, len(out), tg, max(len(out) - 1, 1) / max(tg, 1e-9), _load_avg()),
              file=sys.stderr)
        return
    # bench: prefill of a synthetic prompt at each context (or of the first n tokens of --prompts' first text), then
    # decode
    rng = np.random.default_rng(0)
    text_ids = model.tok.encode(json.loads(open(a.prompts, encoding="utf-8").readline())["text"]) if a.prompts else None
    for n in ctxs:
        model.reset()
        if text_ids is not None and len(text_ids) >= n:
            ids = text_ids[:n]
        else:
            ids = [model.tok.sid[BOS]] + rng.integers(1000, 100000, n - 1).tolist()
        kp = model.times()
        la = _load_avg()
        t0 = time.time()
        lg = model.forward(ids)
        tp = time.time() - t0
        t1 = time.time()
        k0 = model.times()
        nxt = int(np.argmax(lg))
        for _ in range(a.gen):
            lg = model.forward([nxt])
            nxt = int(np.argmax(lg))
        tg = time.time() - t1
        k1 = model.times()
        if os.environ.get("KINDI_BENCH_ATTN"):
            print("attention us per layer call:", [round(lib().dsv_bench_attn(l, 20), 1) for l in (0, 2, 3, 20, 21, 39)],
                  file=sys.stderr)
        print(json.dumps({"ctx": n, "prefill_tok_s": round(n / tp, 2), "prefill_s": round(tp, 1),
                          "prefill_s_parts": {k: round(k0[k] - kp[k], 1) for k in k1},
                          "decode_tok_s": round(a.gen / tg, 3),
                          "decode_ms": {k: round((k1[k] - k0[k]) / a.gen * 1e3, 1) for k in k1},
                          "load_before": la, "load_after": _load_avg()}), flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
