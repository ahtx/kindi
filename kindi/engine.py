"""Kindi engine for Ternary-Bonsai-2 Qwen3.5 hybrid models with ternary weights.

Per token: embedding row -> hybrid GDN/attention blocks with SwiGLU FFNs -> output norm -> tied head.
Every matrix product is a ternary GEMV (k_g4, lookup tables, see docs/HOW-IT-WORKS.md); the activations entering it
are Hadamard-folded (the transform Prism ML folded into the weights), fused into the preceding norm.
Arithmetic follows llama.cpp's reference for this model (fp64 norm sums, fp16 K/V cache), so greedy output matches
it token for token (kindi verify)."""
import json
import os
import time

import numpy as np
try:
    import cupy as cp
except ImportError:                   # a CPU-only host (KINDI_DEVICE=cpu)
    cp = None


def _cupy_arch_fix():
    """CuPy 9.6 picks the NVRTC target with min() on strings: min("120", "75") is "120", which NVRTC 10.2 rejects
    (NVRTC_ERROR_INVALID_OPTION for every kernel on Blackwell, sm_120). The same choice, compared as numbers."""
    try:
        from cupy.cuda import compiler, device
        if cp.cuda.runtime.is_hip or getattr(compiler, "_kindi_arch_fix", False):
            return
        mx = compiler._get_max_compute_capability()
    except Exception:
        return

    def _get_arch():
        a = device.Device().compute_capability
        if a in compiler._tegra_archs:
            return a
        return a if int(a) <= int(mx) else mx

    compiler._get_arch = _get_arch
    compiler._kindi_arch_fix = True


if cp is not None:
    _cupy_arch_fix()

from . import layout

VOCAB = 248320
DEFAULT_DIMS = {"layers": 24, "width": 2048, "ffn": 6144, "heads": 8, "kv_heads": 2,
                "groups": 16, "head_dim": 256, "rope_dim": 64, "state_size": 128, "gdn_inner": 2048,
                "rope_base": 1e7, "full_attention_interval": 4}
EOS = (248044, 248046)
FIRST_SPECIAL = 248044                 # Qwen3.5 vocabulary: ids 248044.. are the special / control tokens
CU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kindi.cu")
KERNELS = ("k_g4", "k_g4l", "k_fold_fwht", "k_rmsfold", "k_smfold", "k_l2w", "k_rmsw", "k_gdn_gates",
           "k_gdn_conv", "k_gdn_delta", "k_rope", "k_deq_emb3", "k_attn_s4w", "k_softmax_rows", "k_attn_v4p",
           "k_attn_comb", "k_argmax", "k_sumparts", "k_fwht_hi", "k_expand", "k_mask_neg", "k_band", "k_topk",
           "k_attn_sg", "k_attn_vg", "k_xchg",
           "k_xchg2", "k_attn_vgn", "k_attn_combn", "k_store_h", "k_cpq", "k_xchg3",
           "k_xrms", "k_gdn_pre", "k_gdn_deltan", "k_rmsw_s", "k_rope2", "k_store_q8", "k_q8row",
           "k_attn_sq8", "k_attn_vq8",
           "k_attn_sq8f", "k_attn_vq8f", "k_deq_emb4",
           "k_attn_sq8t", "k_attn_vq8u", "k_attn_vq8g", "k_xrmsN", "k_attn_vq8w", "k_g5r4", "k_g5r8", "k_attn_sq8d", "k_g5r1", "k_g5r2", "k_g5n4",
           "k_g5r1s", "k_g5r2s", "k_g5r4s", "k_g5r8s",
           "k_g5a1", "k_g5a2", "k_g5a4", "k_g5a8",
           "k_g5a1s", "k_g5a2s", "k_g5a4s", "k_g5a8s")



def _no_thp():
    """ROCm: no khugepaged collapses in this process (they evict the GPU queues mid-kernel; see AMD-NOTES.md)"""
    if cp is None or not getattr(cp.cuda.runtime, "is_hip", False) or os.environ.get("KINDI_THP", "0") == "1":
        return
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)       # PR_SET_THP_DISABLE, inherited by children
    except (OSError, AttributeError):
        pass


_no_thp()

def _upload(a):
    """host array -> card with a plain synchronous copy: cp.asarray stages it in a pinned host buffer of the same size,
    and a GB-sized cudaHostAlloc fails (cudaErrorInvalidValue) on a host that is short of free RAM"""
    g = cp.empty(a.shape, dtype=a.dtype)
    g.set(np.ascontiguousarray(a), stream=cp.cuda.Stream.null)
    return g


class Kernels(object):
    def __init__(self):
        mod = cp.RawModule(code=open(CU, encoding="utf-8").read())
        for k in KERNELS:
            setattr(self, k, mod.get_function(k))



def file_to_card(path, dtype, chunk=1 << 28):
    """a binary file -> device array, read 256 MB at a time; each chunk's page cache is dropped after the copy"""
    dt = np.dtype(dtype)
    n = os.path.getsize(path) // dt.itemsize
    out = cp.empty(n, dtype=dt)
    step = max(1, chunk // dt.itemsize)
    with open(path, "rb") as f:
        o = 0
        while o < n:
            a = np.fromfile(f, dtype=dt, count=min(step, n - o))
            out[o:o + a.size].set(a)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), o * dt.itemsize, a.size * dt.itemsize, os.POSIX_FADV_DONTNEED)
            o += a.size
            del a
    return out

class Engine(object):
    """One sequence at a time. step(tok, pos) -> next token (greedy) with the logits left in self.ylog;
    feed(tok, pos) processes a prompt token without the output head."""

    def __init__(self, pack, ctx=4096, clusters=None, log=print):
        t0 = time.time()
        self.log = log
        self.ctx = int(ctx)
        man = json.load(open(os.path.join(pack, "kindi_pack.json")))
        if man.get("pack_format") != 1:
            raise SystemExit("kindi: pack format %r is not supported - re-run kindi convert" % man.get("pack_format"))
        self.dims = dict(DEFAULT_DIMS, **man.get("dimensions", {}))
        self.HB = int(man.get("hadamard_block", 512))          # Hadamard block of the folded inputs (2B/4B 512, 9B 4096)
        self.head_name = man.get("head", "token_embd.weight")   # the tied embedding, or a separate output.weight
        self.D, self.NL, self.NFF = self.dims["width"], self.dims["layers"], self.dims["ffn"]
        self.NH, self.NHKV, self.HD = self.dims["heads"], self.dims["kv_heads"], self.dims["head_dim"]
        self.NK, self.HK = self.dims["groups"], self.dims["state_size"]
        self.NV = self.dims["gdn_inner"] // self.HK
        self.CONV_CH = 2 * self.NK * self.HK + self.NV * self.HK
        self.NATT = self.NL // self.dims["full_attention_interval"]
        if self.NV % self.NK or self.NH != self.NHKV * 4:
            raise SystemExit("kindi: unsupported grouped head dimensions")
        self.inv_freq = np.float64(self.dims["rope_base"]) ** (-2.0 * np.arange(self.dims["rope_dim"] // 2) / self.dims["rope_dim"])
        self.k = Kernels()
        self._f32 = {}
        self.split_groups = int(os.environ.get("KINDI_SPLIT_GROUPS", "128"))   # column-split a single GEMV below this many 16-row groups per SM
        # one GEMV block per multiprocessor (K420: 1 SMX, K1200: 4 SMMs); a 16-row group stays in one warp, so the
        # sums and the output are the same for any count
        prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.nsm = int(os.environ.get("KINDI_GEMV_BLOCKS") or prop["multiProcessorCount"])
        unit = {3: "SMX", 5: "SMM"}.get(prop["major"], "SM")       # Kepler SMX, Maxwell SMM, later generations SM
        log("GPU: %s, sm_%d%d, %d %s%s, %d MiB" % (prop["name"].decode(), prop["major"], prop["minor"],
                                                   prop["multiProcessorCount"], unit,
                                                   "" if prop["multiProcessorCount"] == 1 else "s",
                                                   prop["totalGlobalMem"] // 2 ** 20))
        self.idx = json.load(open(os.path.join(pack, "index.json")))
        self.bi = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
        self.hsl = None
        self.hs_force = None
        if clusters is not None:
            # the special tokens (end of text, chat markers, <tool_call>, </think>, ...) go in one extra cluster that the
            # fast head always scores: in k-means clusters they can stay out of the shortlist for good (the 9B never
            # closed its thinking with the fast head)
            assign = np.array(clusters["assign"], dtype=np.int64)
            cent = np.asarray(clusters["centroids"], dtype=np.float32)
            self.hs_force = cent.shape[0]
            assign[FIRST_SPECIAL:] = self.hs_force
            clusters = {"assign": assign, "centroids": np.vstack([cent, np.zeros((1, cent.shape[1]), np.float32)])}
            bc = np.fromfile(os.path.join(pack, "bcodes.bin"), dtype=np.uint8)
            dp = np.fromfile(os.path.join(pack, "dpm.bin"), dtype=np.uint16)
            bc, dp, self.hsl = layout.reorder_head(self.bi, bc, dp, clusters, self.head_name)
            dims = layout.repack(self.bi, bc, dp)
            self.bc, self.dp = _upload(bc), _upload(dp)
            del bc, dp
        else:
            # tensor by tensor from memory maps of the pack, repacked on the host and copied to the card: the host
            # holds one tensor, not the weights (a 2 GiB container runs the 9B)
            bcm = np.memmap(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, mode="r")
            dpm = np.memmap(os.path.join(pack, "dpm.bin"), dtype=np.uint16, mode="r")
            self.bc, self.dp = cp.zeros(bcm.size, dtype=cp.uint8), cp.zeros(dpm.size, dtype=cp.uint16)
            dims = {}
            for name, rec in self.bi.items():
                ns, P = rec["ne0"] // 4, rec["ne0"] // 128
                o, d = rec["bc_off"], rec["dpm_off"] // 2
                b, q = np.array(bcm[o:o + ns * rec["rows"]]), np.array(dpm[d:d + P * rec["rows"]])
                dims.update(layout.repack({name: dict(rec, bc_off=0, dpm_off=0)}, b, q))
                self.bc[o:o + b.size].set(b)
                self.dp[d:d + q.size].set(q)
            del bcm, dpm
        self.w = {n: (self.bc[self.bi[n]["bc_off"]:], self.dp[self.bi[n]["dpm_off"] // 2:], np.int32(R), np.int32(C))
                  for n, (R, C) in dims.items()}
        self.fl = file_to_card(os.path.join(pack, "f32.bin"), cp.float32)
        # Hadamard sign vectors, one per folded input width: D (norm outputs), NV*HK (GDN output), NH*HD (attention
        # output; 2048 = D on the 2B, 4096 != D on the 4B) and NFF (FFN down)
        sg = lambda w: cp.asarray(np.fromfile(os.path.join(pack, "signs_%d.f32" % w), dtype=np.float32))  # noqa: E731
        self.sD, self.sV, self.sA, self.sF = sg(self.D), sg(self.NV * self.HK), sg(self.NH * self.HD), sg(self.NFF)
        # cp.asarray stages each upload through pinned host memory and CuPy keeps those blocks: give them back (the
        # 9B weights left ~4 GB of host RAM pinned)
        cp.get_default_pinned_memory_pool().free_all_blocks()
        self.vocab = json.load(open(os.path.join(pack, "vocab.json")))
        z = lambda n, t=cp.float32: cp.zeros(n, dtype=t)  # noqa: E731
        Rh = VOCAB if self.hsl is None else self.hsl["Rp"]
        work = max(self.D, self.NFF, self.NV * self.HK)
        self.y, self.xp, self.h, self.xn, self.zv = z(max(Rh, self.D)), z(work), z(self.D), z(self.D), z(self.NV * self.HK)
        self.h2 = z(self.D)
        self.part = z(self.nsm * max(self.D, self.NFF, self.NV * self.HK))   # column-split GEMV partial sums
        self.qkv, self.cs = z(self.CONV_CH), z(self.CONV_CH)
        self.qkbuf, self.qkvbuf = z(2 * self.NK * self.HK), z(self.NHKV * self.HD)
        self.kn, self.qn, self.qr, self.kro = z(self.NHKV * self.HD), z(self.NH * 2 * self.HD), z(self.NH * self.HD), z(self.NHKV * self.HD)
        self.fo, self.fo6 = z(self.NH * self.HD), z(self.NFF)
        self.outgdn, self.att, self.ffg, self.ffu = z(self.NV * self.HK), z(self.NH * self.HD), z(self.NFF), z(self.NFF)
        self.gb, self.gg, self.gout = z(self.NV), z(self.NV), z(self.NV * self.HK)
        # fit the context into the memory that is left (a 9B on a 4 GB card, a 2B on a 1 GB card): bytes per position
        # = K/V cache (fp16) + attention scores + attention partials + RoPE tables
        from . import chunk1 as _chunk1
        _chunk1.reserve(self)                      # the chunked prompt path's buffers, before the context fit
        per_pos = (self.NATT * 2 * self.NHKV * self.HD * 2 + self.NH * 4 + (self.NH * self.HD * 4) // 256
                   + self.dims["rope_dim"] * 4)
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        budget = free - int(os.environ.get("KINDI_MEM_RESERVE_MB", "96")) * 2 ** 20
        fit = max(0, budget) // per_pos // 256 * 256
        if fit < self.ctx:
            if fit < 1024:
                raise SystemExit("kindi: not enough GPU memory for this model: %d MiB free after the weights"
                                 % (free // 2 ** 20))
            log("context reduced from %d to %d tokens to fit %d MiB of free GPU memory" % (self.ctx, fit, free // 2 ** 20))
            self.ctx = fit
        self.sbuf = z(self.NH * self.ctx)
        self.apart = z(((self.ctx + 255) // 256) * self.NH * self.HD)   # attention partials: NH x HD per 256-position chunk
        self.aout = cp.zeros(1, dtype=cp.int32)
        th = np.arange(self.ctx, dtype=np.float64)[:, None] * self.inv_freq[None, :]
        self.cos_t = cp.asarray(np.cos(th).astype(np.float32))
        self.sin_t = cp.asarray(np.sin(th).astype(np.float32))
        # recurrent (GDN) state for the 18 GDN layers; fp16 K/V cache for the 6 attention layers
        interval = self.dims["full_attention_interval"]
        self.M = [cp.zeros((self.NV, self.HK, self.HK) if (il + 1) % interval else (1,), dtype=cp.float32)
                  for il in range(self.NL)]
        self.conv = [z(3 * self.CONV_CH) for _ in range(self.NL)]
        self.Kc = [cp.zeros((self.ctx, self.NHKV, self.HD), dtype=cp.float16) for _ in range(self.NATT)]
        self.Vc = [cp.zeros((self.ctx, self.NHKV, self.HD), dtype=cp.float16) for _ in range(self.NATT)]
        self.shortlist = 0                 # > 0: fast head over this many clusters
        if self.hsl is not None:
            m = self.hsl
            self.hs_cent = cp.asarray(m["cent"])
            self.hs_permtok, self.hs_permtok_h, self.hs_permpos = cp.asarray(m["permtok"]), m["permtok"], m["permpos"]
            self.hs_gstart, self.hs_gcount = cp.asarray(m["gstart"]), cp.asarray(m["gcount"])
            self.hs_glist = cp.zeros(m["Rp"] // 16, dtype=cp.int32)
            self.hs_yc = z(m["Rp"])
            self.hs_tokc = cp.zeros(m["Rp"], dtype=cp.int32)
        self.ylog, self.ylog_n, self.ylog_map = self.y, VOCAB, None
        log("engine ready: ctx %d, head %s (%.1fs)" % (self.ctx, "reordered" if self.hsl else "plain", time.time() - t0))

    # ------------------------------------------------------------------------------------------- helpers
    def f32(self, name):
        v = self._f32.get(name)              # cached: a fresh view per call costs Python time on every launch
        if v is None:
            rec = self.idx["tensors"][name]
            off = rec["f32_off"] // 4
            v = self._f32[name] = self.fl[off:off + rec["rows"] * rec["ne0"]]
        return v

    def gemv(self, names, x, outs, parts_ok=False):
        """Up to three ternary tensors with the same input in one launch."""
        a = [self.w[n] for n in names]
        args = []
        for i in range(3):
            if i < len(a):
                args += [a[i][0], a[i][1], outs[i], a[i][2]]
            else:
                args += [a[0][0], a[0][1], outs[0], np.int32(0)]
        R0, C = int(a[0][2]), int(a[0][3])
        if len(a) == 1 and self.nsm > 1 and (R0 >> 4) < self.split_groups * self.nsm and (C >> 9) >= self.nsm:
            # few row groups for many columns (the 4B's 2560-row outputs): each block takes 1/nsm of the columns of
            # every row, so a phase has 4x the work for the same table build and barriers; then add the parts
            args[2] = self.part
            self.k.k_g4((1, self.nsm), (1024,), tuple(args) + (x, a[0][3]))
            if parts_ok:                          # the next rmsfold adds the parts (a block output in self.part)
                return self.nsm
            self.k.k_sumparts(((R0 + 255) // 256,), (256,), (self.part, outs[0], np.int32(R0), np.int32(self.nsm)))
            return 1
        self.k.k_g4((self.nsm,), (1024,), tuple(args) + (x, a[0][3]))
        return 1

    def fold_args(self, width):
        """Hadamard blocks above 512: the fold kernels do 512-point chunks on all SMs (HB = -512, unscaled) and
        k_fwht_hi finishes the block. Returns (grid, HB argument)."""
        if self.HB > 512:
            return (width // 512,), np.int32(-512)
        return (width // self.HB,), np.int32(self.HB)

    def fold_hi(self, buf, width):
        if self.HB > 512:
            self.k.k_fwht_hi(((width // 512 // (self.HB // 512) * 512 + 255) // 256,), (256,),
                             (buf, np.int32(self.HB), np.int32(width)))

    def rmsfold(self, x, wname):
        """RMS norm + Hadamard fold of the residual stream (self.h; x is ignored). If a block left its output in
        self.y, the kernel adds it first (that block's residual add, fused here) and writes the sum to the second
        buffer, which then becomes self.h."""
        grid, hb = self.fold_args(self.D)
        self.k.k_rmsfold(grid, (512,), (self.h, self.part if self.pending > 1 else self.y, self.h2,
                                        self.f32(wname), self.sD, self.xn,
                                        self.xp, np.int32(self.D), np.int32(self.pending), hb))
        self.fold_hi(self.xp, self.D)
        if self.pending:
            self.h, self.h2 = self.h2, self.h
        self.pending = 0
        return self.xn

    def rms_rows(self, x, wname, rows, n, out):
        self.k.k_rmsw(((rows + 7) // 8,), (256,), (x, self.f32(wname), out, np.int32(n), np.int32(rows)))
        return out

    def reset(self):
        for a in self.M + self.conv + self.Kc + self.Vc:
            a.fill(0)

    def embed(self, tok):
        a = self.w["token_embd.weight"]
        # the embedding rows are in cluster order only when the head is the tied embedding
        row = int(self.hs_permpos[tok]) if self.hsl is not None and self.head_name == "token_embd.weight" else int(tok)
        self.k.k_deq_emb3((1,), (1024,), (a[0], a[1], self.sD, self.h, np.int32(row), np.int32(self.D), np.int32(self.HB)))
        return self.h

    # ------------------------------------------------------------------------------------------- blocks
    def gdn(self, il, h):
        p = "blk.%d." % il
        xn = self.rmsfold(h, p + "attn_norm.weight")
        self.gemv([p + "attn_qkv.weight", p + "attn_gate.weight"], self.xp, [self.qkv, self.zv])
        self.k.k_gdn_gates((self.NV,), (256,), (xn, self.f32(p + "ssm_beta.weight"), self.f32(p + "ssm_alpha.weight"),
                                           self.f32(p + "ssm_dt.bias"), self.f32(p + "ssm_a"), self.gb, self.gg,
                                           np.int32(self.D)))
        self.k.k_gdn_conv(((self.CONV_CH + 255) // 256,), (256,),
                          (self.qkv, self.conv[il], self.f32(p + "ssm_conv1d.weight"), self.cs, np.int32(self.CONV_CH)))
        self.k.k_l2w((4,), (256,), (self.cs, self.qkbuf, np.int32(self.HK), np.int32(2 * self.NK)))
        self.k.k_gdn_delta((self.NV,), (self.HK,), (self.M[il], self.qkbuf[:self.NK * self.HK], self.qkbuf[self.NK * self.HK:], self.cs[2 * self.NK * self.HK:],
                                          self.gg, self.gb, self.gout, np.float32(1.0 / np.sqrt(self.HK)), np.int32(self.NK)))
        outn = self.rms_rows(self.gout, p + "ssm_norm.weight", self.NV, self.HK, self.outgdn)
        grid, hb = self.fold_args(self.NV * self.HK)
        self.k.k_smfold(grid, (512,), (self.zv, outn, self.sV, self.xp, np.int32(self.HK), np.int32(self.NK), np.int32(self.NV // self.NK), hb))
        self.fold_hi(self.xp, self.NV * self.HK)
        npart = self.gemv([p + "ssm_out.weight"], self.xp, [self.y], parts_ok=True)
        self.pending = npart
        return h

    def attn(self, il, h, pos, kvi):
        p = "blk.%d." % il
        self.rmsfold(h, p + "attn_norm.weight")
        self.gemv([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"], self.xp,
                  [self.qn, self.kro, self.qkvbuf])
        Kn = self.rms_rows(self.kro, p + "attn_k_norm.weight", self.NHKV, self.HD, self.kn)
        qc = self.qn.reshape(self.NH, 2 * self.HD)[:, :self.HD].copy()
        q = self.rms_rows(qc, p + "attn_q_norm.weight", self.NH, self.HD, self.fo)
        cs, sn = self.cos_t[pos], self.sin_t[pos]
        self.k.k_rope((self.NH,), (256,), (q, cs, sn, self.qr, np.int32(self.HD)))
        self.k.k_rope((self.NHKV,), (256,), (Kn, cs, sn, self.kro, np.int32(self.HD)))
        self.Kc[kvi][pos] = self.kro.reshape(self.NHKV, self.HD)
        self.Vc[kvi][pos] = self.qkvbuf.reshape(self.NHKV, self.HD)
        n = pos + 1
        nch = (n + 255) // 256
        Kall, Vall = self.Kc[kvi][:n], self.Vc[kvi][:n]
        self.k.k_attn_s4w((self.NHKV, min(32, (n + 31) // 32)), (256,), (Kall, self.qr, self.sbuf, np.int32(n), np.int32(self.NHKV)))
        self.k.k_softmax_rows((self.NH,), (256,), (self.sbuf, np.int32(n)))
        self.k.k_attn_v4p((self.NHKV, 1, nch), (128,), (self.sbuf, Vall, self.apart, np.int32(n), np.int32(self.NH), np.int32(self.NHKV)))
        self.k.k_attn_comb((self.NH,), (256,), (self.apart, self.qn, self.att, np.int32(nch), np.int32(self.NH)))
        grid, hb = self.fold_args(self.NH * self.HD)
        self.k.k_fold_fwht(grid, (512,), (self.att, self.sA, self.xp, hb))
        self.fold_hi(self.xp, self.NH * self.HD)
        npart = self.gemv([p + "attn_output.weight"], self.xp, [self.y], parts_ok=True)
        self.pending = npart
        return h

    def ffn(self, il, h):
        p = "blk.%d." % il
        self.rmsfold(h, p + "post_attention_norm.weight")
        self.gemv([p + "ffn_gate.weight", p + "ffn_up.weight"], self.xp, [self.ffg, self.ffu])
        grid, hb = self.fold_args(self.NFF)
        self.k.k_smfold(grid, (512,), (self.ffg, self.ffu, self.sF, self.fo6, np.int32(0), np.int32(0), np.int32(0), hb))
        self.fold_hi(self.fo6, self.NFF)
        npart = self.gemv([p + "ffn_down.weight"], self.fo6, [self.y], parts_ok=True)
        self.pending = npart
        return h

    def body(self, tok, pos):
        if pos >= self.ctx:
            raise OverflowError("position %d is past the context (%d)" % (pos, self.ctx))
        h = self.embed(tok)
        self.pending = 0
        kvi = 0
        for il in range(self.NL):
            if (il + 1) % self.dims["full_attention_interval"] == 0:
                h = self.attn(il, h, pos, kvi)
                kvi += 1
            else:
                h = self.gdn(il, h)
            h = self.ffn(il, h)
        return h

    def feed(self, tok, pos):
        """A prompt token whose logits are not needed."""
        self.body(tok, pos)

    def feed_batch(self, toks, pos):
        """Prompt tokens pos ..: whole chunks through kindi.chunk1 (KINDI_CHUNK=0: off), the rest one by one."""
        from . import chunk1 as _chunk1
        _chunk1.feed_batch(self, toks, pos)

    def step(self, tok, pos):
        h = self.body(tok, pos)
        self.rmsfold(h, "output_norm.weight")
        return self.head()

    # ------------------------------------------------------------------------------------------- output head
    def head(self):
        """-> greedy next token id. Leaves the logits in self.ylog[:self.ylog_n], their argmax slot in self.aout and
        (reordered head) each slot's token id in self.ylog_map."""
        a = self.w[self.head_name]
        if self.hsl is None:
            self.gemv([self.head_name], self.xp, [self.y])
            self.k.k_argmax((1,), (1024,), (self.y, np.int32(VOCAB), self.aout))
            self.ylog, self.ylog_n, self.ylog_map = self.y, VOCAB, None
            return int(self.aout[0])
        Rp = self.hsl["Rp"]
        M = self.shortlist
        if M <= 0 or M >= self.hsl["K"]:
            self.gemv([self.head_name], self.xp, [self.y])
            self.k.k_mask_neg((64,), (256,), (self.y, self.hs_permtok, np.int32(Rp)))
            self.k.k_argmax((1,), (1024,), (self.y, np.int32(Rp), self.aout))
            self.ylog, self.ylog_n, self.ylog_map = self.y, Rp, self.hs_permtok
            return int(self.hs_permtok_h[int(self.aout[0])])
        cs = self.hs_cent @ self.xp[:self.D]
        cs[self.hs_force] = 3.0e38                      # the special-token cluster is always in the shortlist
        sel = cp.argsort(-cs)[:M].astype(cp.int32)
        cnt = self.hs_gcount[sel]
        off = (cp.cumsum(cnt) - cnt).astype(cp.int32)
        ng = int(cnt.sum())
        self.k.k_expand((M,), (128,), (sel, self.hs_gstart, self.hs_gcount, off, self.hs_glist))
        zr = np.int32(0)
        self.k.k_g4l((self.nsm,), (1024,), (a[0], a[1], self.hs_yc, a[2], a[0], a[1], self.hs_yc, zr, a[0], a[1], self.hs_yc,
                                     zr, self.xp, a[3], self.hs_glist, np.int32(ng), self.hs_permtok, self.hs_tokc))
        n = ng * 16
        self.k.k_mask_neg((16,), (256,), (self.hs_yc, self.hs_tokc, np.int32(n)))
        self.k.k_argmax((1,), (1024,), (self.hs_yc, np.int32(n), self.aout))
        self.ylog, self.ylog_n, self.ylog_map = self.hs_yc, n, self.hs_tokc
        return int(self.hs_tokc[int(self.aout[0])])

    def generate(self, ids, n):
        """Greedy: the n tokens after the prompt ids (fresh state)."""
        self.reset()
        for i, t in enumerate(ids[:-1]):
            self.feed(t, i)
        pos = len(ids)
        out = [self.step(ids[-1], pos - 1)]
        while len(out) < n:
            out.append(self.step(out[-1], pos))
            pos += 1
        return out


PLAN_SMALL_HIP = ("gfx1151",)      # the AMD GPUs where the 2B / 4B / 9B take the plan engine (measured)


def make_engine(pack, ctx=4096, clusters=None, log=print):
    """Engine on one GPU, or the two-card tensor parallel engine (kindi.tp) when the weights do not fit one card and
    two GPUs are present. KINDI_TP=1 / 0 forces the choice."""
    if os.environ.get("KINDI_DEVICE", "gpu") == "cpu" or cp is None:
        from .cpu import Engine as CPUEngine
        if clusters is not None:
            log("the fast head is not available on the CPU: exact head")
        return CPUEngine(pack, ctx=ctx, log=log)
    from .sched import blocking_sync
    blocking_sync(log)                      # before the first CUDA call (KINDI_BLOCKING_SYNC=0: off)
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    force = os.environ.get("KINDI_TP", "")
    weights = int(man.get("bcodes_bytes", 0)) + int(man.get("dpm_bytes", 0))
    two = cp.cuda.runtime.getDeviceCount() >= 2
    dims = man.get("dimensions", {})
    # the plan engine (kindi.tp) also runs on one card: it is the one that has the 27B's kernels (6 q heads per K/V
    # head, Hadamard blocks of 1024, q8_0 K/V)
    plan_model = dims.get("heads", 8) != 4 * dims.get("kv_heads", 2) or int(man.get("hadamard_block", 512)) == 1024
    # AMD GPUs with many CUs (Strix Halo, gfx1151): the 2B, 4B and 9B (512- and 4096-point folds) also run faster on
    # the plan engine's 5-trit products (2B write 173 -> 259, 9B 52 -> 89 tokens/s); measured there only
    if cp.cuda.runtime.is_hip and int(man.get("hadamard_block", 512)) in (512, 4096) and not two:
        arch = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id).get("gcnArchName", b"")
        arch = (arch.decode() if isinstance(arch, bytes) else str(arch)).split(":")[0]
        plan_model = plan_model or arch in PLAN_SMALL_HIP
    if force == "1" or (force != "0" and (plan_model or (two and weights > 0.9 * cp.cuda.runtime.memGetInfo()[1]))):
        from .tp import TPEngine
        from . import spec
        spec.install(log)                       # KINDI_TC=1: tensor-core plans (one sm_80+ card); no-op otherwise
        if clusters is not None:
            log("the fast head is not available with two cards: exact head")
        return TPEngine(pack, ctx=ctx, log=log)
    return Engine(pack, ctx=ctx, clusters=clusters, log=log)
