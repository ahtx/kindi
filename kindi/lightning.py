"""Kindi engine for Nemotron-H MoE GGUF files (architecture nemotron_h_moe: Nemotron 3.5 Lightning 30B-A3B, ternary
Q2_0 weights with a Q8_0 embedding and head).

The 52 layers are of three kinds: Mamba-2 mixers, MoE feed-forward layers (128 relu^2 experts, 6 per token, and one
shared expert) and attention layers (GQA, no positions). Each layer is x += mixer(rms_norm(x)). The weights are read
from the GGUF file itself and repacked at load time (lightning.cu has the layouts). One card holds the model; when the
experts do not fit, the least used experts of each layer stay in pinned host memory and the card reads them over PCIe
(the kernels take a pointer per expert, on the card or on the host).

A step replays one CUDA graph: the token, the position and the routing stay on the card. A prompt goes through in
chunks: the MoE layers group the chunk's tokens by expert."""
import json
import mmap
import os
import time

import numpy as np

from .gguf import GGUF
from .moe import MoESpec, Usage, plan
from .nanbeige import _capture, _free_bytes, module

Q2_0, Q8_0, F32 = 42, 8, 0
NULL = np.uint64(0)
CHUNK = 512                                                 # prompt tokens per pass (Engine.chunk)
NSPLIT = int(os.environ.get("KINDI_LT_NSPLIT", "32"))           # decode attention: position ranges per K/V head
NSPLIT_TC = int(os.environ.get("KINDI_LT_NSPLIT_TC", "64"))     # the same for lt_fa_dec (tensor cores)
KERNELS = ("lt_embed", "lt_rms_quant", "lt_quant", "lt_mv_q2", "lt_mv_q2_down", "lt_mv_q8", "lt_argmax_part",
           "lt_argmax_final", "lt_router", "lt_topk", "lt_conv", "lt_ssm", "lt_gnorm_quant", "lt_kv_store",
           "lt_attn_dec", "lt_attn_merge", "lt_attn_pre", "lt_mm_q2", "lt_moe_lists", "lt_moe_combine", "lt_advance",
           "lt_set", "lt_mmd_q2", "lt_mmd32_q2", "lt_attn_pre8", "lt_attn_pre6", "lt_mvdg_q2", "lt_stage_dec", "lt_cand",
           "lt_router_topk", "lt_quant2", "lt_cache_dec")
K35_KERNELS = ("lt_mvk_q2", "lt_mvk_q8", "lt_mvk_sum", "lt_mvk_down_sum", "lt_mmk_q2", "lt_mmk32_q2", "lt_mmf_q2",
               "lt_mmf32_q2", "lt_kdeq_q2", "lt_kx2f", "lt_mvkg_q2")
KBLAS = os.environ.get("KINDI_LT_KBLAS", "1") == "1"         # Kepler / P100 dense prompts: cublasSgemm (_kblas)
KDQ_BYTES = 128 << 20                                       # _kblas: fp32 weight rows per cublasSgemm
K35_EXACT = os.environ.get("KINDI_LT_K35_EXACT") == "1"    # prompts: the exact run sums (lt_mmk) instead of lt_mmf
# decode with host experts: device slots per MoE layer that keep the last used host experts (least recently used out).
# GTX 1080, a text: 6 slots per layer +6% (12: -2%, 24: -8%: the slots take the place of experts on the card).
# decode slots per MoE layer for host experts: 10 (0.4.5: 6). Text write test, 11-10: an RTX 2070 SUPER 8 GB on
# PCIe 3.0 x16, 106.4 -> 112.4 tokens/s at 512 and 110.3 -> 115.1 at 8K; an RTX 3080 on PCIe x1 with 2.6 GB held
# back, 7.18 -> 7.88 and 8.36 -> 9.05. (More slots leave fewer experts on the card: 85 -> 81 on the 2070 SUPER.)
DCACHE = int(os.environ.get("KINDI_LT_DCACHE", "10"))
KRPB = 64                                                   # lt_mvk_*: rows per block
HERE = os.path.dirname(os.path.abspath(__file__))


class Q2(object):
    """a Q2_0 matrix (or a stack of experts) on the card: u32 code words and fp16 scales; tc / ts = the pointer
    tables (one entry per expert) that the kernels take"""
    def __init__(self, rows, cols, n=1):
        self.rows, self.cols, self.n = rows, cols, n


class _Tables(object):
    """a matrix with other pointer tables (the staged copies of its host experts)"""
    def __init__(self, w, tc, ts):
        self.rows, self.cols, self.n, self.tc, self.ts = w.rows, w.cols, w.n, tc, ts


class Spec(object):
    """the model dimensions from the GGUF metadata"""
    def __init__(self, g):
        kv = g.kv
        a = kv["general.architecture"]
        self.n_layer = int(kv[a + ".block_count"])
        self.n_embd = int(kv[a + ".embedding_length"])
        ff = kv[a + ".feed_forward_length"]
        kvh = kv[a + ".attention.head_count_kv"]
        ff = ff if isinstance(ff, list) else [ff] * self.n_layer
        kvh = kvh if isinstance(kvh, list) else [kvh] * self.n_layer
        self.kind = ["attn" if kvh[i] else "moe" if ff[i] else "ssm" for i in range(self.n_layer)]
        self.n_head = int(kv[a + ".attention.head_count"])
        self.n_kv = max(kvh)
        self.hd = int(kv[a + ".attention.key_length"])
        self.eps = float(kv[a + ".attention.layer_norm_rms_epsilon"])
        self.d_conv = int(kv[a + ".ssm.conv_kernel"])
        self.d_state = int(kv[a + ".ssm.state_size"])
        self.n_group = int(kv[a + ".ssm.group_count"])
        self.d_inner = int(kv[a + ".ssm.inner_size"])
        self.n_sh = int(kv[a + ".ssm.time_step_rank"])          # Mamba heads
        self.n_exp = int(kv[a + ".expert_count"])
        self.n_used = int(kv[a + ".expert_used_count"])
        self.n_ff = int(kv[a + ".expert_feed_forward_length"])
        self.n_ffsh = int(kv[a + ".expert_shared_feed_forward_length"])
        self.w_norm = bool(kv.get(a + ".expert_weights_norm", True))
        self.w_scale = float(kv.get(a + ".expert_weights_scale", 1.0))
        self.ctx_train = int(kv.get(a + ".context_length", 262144))
        self.vocab = len(kv["tokenizer.ggml.tokens"])
        self.d_xbc = self.d_inner + 2 * self.n_group * self.d_state
        self.d_in = self.d_inner + self.d_xbc + self.n_sh       # in_proj rows: z, xBC, dt
        assert self.hd == 128 and self.n_head // self.n_kv <= 16, "lightning.cu: 128-wide heads, <= 16 per K/V head"
        assert self.d_inner // self.n_sh == 64 and self.d_state == 128 and self.d_conv == 4, \
            "lightning.cu: Mamba-2 head dim 64, state 128, conv 4"
        assert self.n_exp <= 1024 and self.n_exp % 32 == 0 and self.n_used <= 16 and self.w_norm
        assert self.n_embd % 64 == 0 and self.n_ff % 64 == 0 and self.n_ffsh % 64 == 0


class Engine(object):
    def __init__(self, path, ctx=32768, device=None, log=print, experts_on_card=None):
        import cupy as cp
        self.cp, self.log = cp, log
        g = GGUF(path)
        self.g, self.s = g, Spec(g)
        s = self.s
        self.tinfo = {n: (ne, ty, off) for n, ne, ty, off in g.tensors}
        f = open(path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        self.dev = cp.cuda.Device(0 if device is None else device)
        self.dev.use()
        self.cc = int(self.dev.compute_capability)
        self.ctx = (max(256, int(ctx)) + 255) // 256 * 256 if int(ctx) > 0 else 0     # 0: the largest that fits
        # cards without dp4a (sm_35, sm_37, sm_60): the fp32-exact kernels of lightning_k35.cu
        self.k35 = self.cc < 61 or os.environ.get("KINDI_LT_K35") == "1"
        src = open(os.path.join(HERE, "lightning.cu"), encoding="utf-8").read()
        if self.k35:
            src += open(os.path.join(HERE, "lightning_k35.cu"), encoding="utf-8").read()
        mod = module(cp, src, tc=False)
        self.k = {n: mod.get_function(n) for n in KERNELS + (K35_KERNELS if self.k35 else ())}
        if self.k35 and self.cc < 50:                      # Kepler: 8-byte shared memory banks for the tile loads
            import ctypes
            cu = ctypes.CDLL("libcuda.so.1")
            for n in ("lt_mmk_q2", "lt_mmk32_q2", "lt_mmf_q2", "lt_mmf32_q2"):
                cu.cuFuncSetSharedMemConfig(ctypes.c_void_p(getattr(self.k[n], "kernel", self.k[n]).ptr), 2)
        # tensor cores for the prompt on sm_75+ (lightning_tc.cu; Turing: LT_SM75); the other cards use lt_mm_q2
        self.tc = self.cc >= 75 and os.environ.get("KINDI_LT_TC", "1") != "0"
        # prompt attention without tensor cores: lt_attn_pre6 on sm_60+ (2.2x on a GTX 1080), lt_attn_pre8 on Kepler
        self.pre6 = os.environ.get("KINDI_LT_PRE6", "1" if self.cc >= 60 else "0") == "1"
        if self.tc:
            tmod = module(cp, open(os.path.join(HERE, "lightning_tc.cu"), encoding="utf-8").read(), tc=True,
                          extra=("-DLT_SM75",) if self.cc < 80 or os.environ.get("KINDI_LT_SM75") == "1" else ())
            self.k["lt_mm_tc"] = tmod.get_function("lt_mm_tc")
            self.k["lt_mm_tc32"] = tmod.get_function("lt_mm_tc32")
            self.k["lt_fa"] = tmod.get_function("lt_fa")
            self.k["lt_fa_dec"] = tmod.get_function("lt_fa_dec")
        self.stream = cp.cuda.Stream(non_blocking=True)
        # prompt tokens per pass: more on big cards (bigger MoE groups fill the tiles better): 2048 without dp4a and
        # 11 GB+, 1024 with tensor cores and 10 GB+
        total = cp.cuda.runtime.memGetInfo()[1]
        big = 2048 if self.k35 and total >= 11 * 2 ** 30 else 1024 if self.cc >= 75 and total >= 9.5 * 2 ** 30 else CHUNK
        if self.ctx == 0 or self.ctx > 32768 and not self.k35:   # long contexts: the room goes to the K/V cache
            big = CHUNK
        self.chunk = int(os.environ.get("KINDI_LT_CHUNK") or big)
        t0 = time.time()
        # the plan leaves KINDI_LT_RESERVE_MB (350 MB) free for what it does not count (the CUDA context, compiled
        # kernels, the memory pool's rounding). On 8 GB cards with host experts that was not always enough (an RTX 2070
        # SUPER: out of memory at the decode slots). On an out-of-memory error, free everything, keep 512 MB more
        # free, and plan again (up to 3 times).
        self.extra_mb = 0
        before = set(self.__dict__)
        for attempt in range(4):
            self.layers = [dict() for _ in range(s.n_layer)]
            try:
                self._plan(experts_on_card)
                self._load()
                self._buffers()
                break
            except cp.cuda.memory.OutOfMemoryError as e:
                if attempt == 3:
                    raise
                # how much was missing: the failed allocation minus what was still free (card + the pool's free
                # blocks); keep that much more free, plus 96 MB, so the plan gives up as few experts as it can
                import re as _re
                m = _re.search(r"allocating ([\d,]+) bytes", str(e))
                want = int(m.group(1).replace(",", "")) if m else 512 * 2 ** 20
                pool = cp.get_default_memory_pool()
                short = max(0, want - cp.cuda.runtime.memGetInfo()[0] - (pool.total_bytes() - pool.used_bytes()))
                for k in set(self.__dict__) - before:
                    if k not in ("extra_mb", "layers"):
                        delattr(self, k)
                self.layers = None
                import gc
                gc.collect()
                pool.free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                self.extra_mb += max(96, -(-short // 2 ** 20) + 96) if attempt == 0 else 512
                log("plan: out of card memory (%s); planning again with %d MB more kept free"
                    % (str(e).split("(")[0].strip(), self.extra_mb))
        self.graph = None
        self.pos = 0
        log("loaded %s on device %d (sm_%d) in %.1f s: ctx %d, experts on the card %d of %d per layer" % (
            os.path.basename(path), self.dev.id, self.cc, time.time() - t0, self.ctx, self.n_card, s.n_exp))

    # ------------------------------------------------------------------ loading
    def _raw(self, name):
        ne, ty, off = self.tinfo[name]
        n = int(np.prod(ne))
        nb = {Q2_0: n // 64 * 18, Q8_0: n // 32 * 34, F32: n * 4}[ty]
        return np.frombuffer(self.mm, np.uint8, nb, self.g.data_start + off), ne, ty

    def _f32(self, name):
        raw, ne, ty = self._raw(name)
        assert ty == F32, name
        a = self.cp.empty(int(np.prod(ne)), self.cp.float32)
        a.set(raw.view(np.float32))
        return a

    def _expert_bytes(self):
        s = self.s
        return 2 * s.n_ff * s.n_embd // 64 * 18                 # one expert: up + down

    def _plan(self, experts_on_card):
        """how many experts of each MoE layer stay on the card: all when they fit with the context and the buffers"""
        s = self.s
        free = _free_bytes(self.cp, [self.dev.id])[0]
        dense = 0
        for n, (ne, ty, off) in self.tinfo.items():
            if "_exps" in n or n == "token_embd.weight":
                continue
            dense += int(np.prod(ne)) // 64 * 18 if ty == Q2_0 else int(np.prod(ne)) // 32 * 34 if ty == Q8_0 \
                else int(np.prod(ne)) * 4
        n_attn = s.kind.count("attn")
        n_moe = s.kind.count("moe")
        per_pos = n_attn * 2 * s.n_kv * s.hd * 2                # K/V bytes per position
        if self.ctx == 0:      # ctx 0: all experts on the card (if they fit at 4K) and the rest of the card for K/V
            rest = free - dense - s.kind.count("ssm") * (s.n_sh * 64 * s.d_state + s.d_xbc * 3) * 4 \
                - self._buf_bytes() - self._reserve()
            rest -= n_moe * self._expert_bytes() * (s.n_exp if experts_on_card is None else int(experts_on_card))
            # (the GGUF says 1M; the model's position limit is 262,144)
            self.ctx = int(min(s.ctx_train, 262144, max(4096, rest // per_pos)) // 256 * 256)
        kv = per_pos * self.ctx
        state = s.kind.count("ssm") * (s.n_sh * 64 * s.d_state + s.d_xbc * 3) * 4
        reserve = self._reserve()
        self.dcache = max(s.n_used, DCACHE)
        room = free - dense - kv - state - self._buf_bytes() - reserve
        eb = self._expert_bytes()
        fit = max(0, int(room // (n_moe * eb)))
        if fit < s.n_exp:          # host experts: two staging buffers (prompts) of (n_exp - n) experts each,
            # and the decode slots (self.dcache per layer)
            fit = max(0, int((room - 2 * s.n_exp * eb - n_moe * self.dcache * eb) // ((n_moe - 2) * eb)))
        self.n_card = min(s.n_exp, fit if experts_on_card is None else int(experts_on_card))
        if self.n_card < s.n_used:
            need = dense + kv + state + self._buf_bytes() + reserve + n_moe * s.n_used * eb
            raise MemoryError("Lightning needs %.1f GB on the card at ctx %d; %.1f GB are free" % (
                need / 1e9, self.ctx, free / 1e9))
        self.placement, self.max_host = None, s.n_exp - self.n_card
        if experts_on_card is None and self.n_card < s.n_exp:
            self._plan_weighted(room, eb, n_moe)
        where = self.placement.describe() if self.placement else "%d of %d experts per layer on the card" % (
            self.n_card, s.n_exp)
        self.log("plan: %.2f GB dense weights, %.2f GB K/V at ctx %d, %s%s" % (
            dense / 1e9, kv / 1e9, self.ctx, where, "" if self.n_card == s.n_exp else ", the rest in host memory"))

    def _reserve(self):
        """the bytes the plan keeps free: KINDI_LT_RESERVE_MB, plus what a failed first plan added"""
        return (int(os.environ.get("KINDI_LT_RESERVE_MB", "350")) + getattr(self, "extra_mb", 0)) * 2 ** 20

    def _plan_weighted(self, room, eb, n_moe):
        """host experts and measured use counts: give the card slots by use over all MoE layers (kindi/moe.py).
        Each layer keeps at least m experts on the card; the two prompt staging buffers hold n_exp - m experts.
        Try a few m and keep the plan that serves the most measured picks from the card."""
        s = self.s
        spec = MoESpec([il for il, k in enumerate(s.kind) if k == "moe"], s.n_exp, s.n_used, eb, shared=True)
        usage = Usage.load(spec, os.path.join(HERE, "lightning-experts.json"))
        if not usage.has_counts() or os.environ.get("KINDI_MOE_WEIGHTED", "1") == "0":
            return
        best = None
        for m in sorted(set(list(range(max(s.n_used, self.n_card - 64), self.n_card + 1, 4)) + [self.n_card])):
            budget = room - 2 * (s.n_exp - m) * eb - n_moe * self.dcache * eb
            if budget < m * n_moe * eb:
                continue
            p = plan(spec, usage, budget, min_card=m, weighted=True)
            f = p.served("card")
            if best is None or f > best[0] + 1e-9:
                best = (f, p, m)
        if best is not None:
            self.placement, self.max_host = best[1], s.n_exp - best[2]
            self.n_card = min(best[1].counts("card").values())

    def _usage(self):
        """expert order per layer, most used first (lightning-experts.json, measured on a mixed text; else index)"""
        p = os.path.join(HERE, "lightning-experts.json")
        if os.path.exists(p):
            return {int(k): v for k, v in json.load(open(p))["order"].items()}
        return {}

    def _q2(self, name, host_rows=None):
        """upload a Q2_0 tensor: codes and scales; for an expert stack, the experts listed in host_rows go to pinned
        host memory"""
        cp = self.cp
        raw, ne, ty = self._raw(name)
        assert ty == Q2_0, name
        cols = int(ne[0])
        rows = int(np.prod(ne[1:]))
        n = int(ne[2]) if len(ne) > 2 else 1
        per = rows // n
        nb = cols // 64
        w = Q2(per, cols, n)
        blk = raw.reshape(rows * nb, 18)
        cw, sw = per * nb * 16, per * nb * 2                   # bytes per expert
        on = [e for e in range(n) if not host_rows or e not in host_rows]
        off = [e for e in range(n) if host_rows and e in host_rows]
        w.codes = cp.empty(len(on) * cw, cp.uint8)
        w.sc = cp.empty(len(on) * sw // 2, cp.float16)
        pc, ps = [0] * n, [0] * n
        for i, e in enumerate(on):
            b = blk[e * per * nb:(e + 1) * per * nb]
            w.codes[i * cw:(i + 1) * cw].set(np.ascontiguousarray(b[:, 2:]).reshape(-1))
            w.sc[i * sw // 2:(i + 1) * sw // 2].set(np.ascontiguousarray(b[:, :2]).view(np.float16).reshape(-1))
            pc[e] = w.codes.data.ptr + i * cw
            ps[e] = w.sc.data.ptr + i * sw
        w.hoff, w.cw, w.sw = off, cw, sw
        w.hbytes = len(off) * (cw + sw)
        if off:
            w.hmem = cp.cuda.alloc_pinned_memory(len(off) * (cw + sw))
            h = np.frombuffer(w.hmem, np.uint8, len(off) * (cw + sw))
            for i, e in enumerate(off):
                b = blk[e * per * nb:(e + 1) * per * nb]
                h[i * cw:(i + 1) * cw] = b[:, 2:].reshape(-1)
                h[len(off) * cw + i * sw:len(off) * cw + (i + 1) * sw] = b[:, :2].reshape(-1)
                pc[e] = w.hmem.ptr + i * cw
                ps[e] = w.hmem.ptr + len(off) * cw + i * sw
        w.tc = cp.asarray(np.array(pc, np.uint64))
        w.ts = cp.asarray(np.array(ps, np.uint64))
        return w

    def _q2cat(self, names):
        """several Q2_0 matrices with the same columns as one (rows stacked): q, k, v"""
        cp = self.cp
        parts = [self._raw(n) for n in names]
        cols = int(parts[0][1][0])
        blk = np.concatenate([p[0].reshape(-1, 18) for p in parts])
        rows = blk.shape[0] // (cols // 64)
        w = Q2(rows, cols)
        w.codes = cp.empty(blk.shape[0] * 16, cp.uint8)
        w.codes.set(np.ascontiguousarray(blk[:, 2:]).reshape(-1))
        w.sc = cp.empty(blk.shape[0], cp.float16)
        w.sc.set(np.ascontiguousarray(blk[:, :2]).view(np.float16).reshape(-1))
        w.tc = cp.asarray(np.array([w.codes.data.ptr], np.uint64))
        w.ts = cp.asarray(np.array([w.sc.data.ptr], np.uint64))
        return w

    def _load(self):
        cp, s = self.cp, self.s
        # the embedding stays in pinned host memory: a token reads one row (2.9 KB)
        raw, ne, ty = self._raw("token_embd.weight")
        assert ty == Q8_0
        self.emb_mem = cp.cuda.alloc_pinned_memory(raw.nbytes)
        np.frombuffer(self.emb_mem, np.uint8, raw.nbytes)[:] = raw
        # the head: lane-order int8 weights and fp16 scales
        raw, ne, ty = self._raw("output.weight")
        assert ty == Q8_0 and int(ne[1]) == s.vocab
        blk = raw.reshape(-1, 34)
        self.head_sc = cp.empty(blk.shape[0], cp.float16)
        self.head_sc.set(np.ascontiguousarray(blk[:, :2]).view(np.float16).reshape(-1))
        self.head_q = cp.empty(blk.shape[0] * 8, cp.int32)
        R = 65536
        for r0 in range(0, blk.shape[0], R):
            q = blk[r0:r0 + R, 2:].reshape(-1, 2, 4, 4)          # [block][group][j][k] -> word k, byte j
            self.head_q[r0 * 8:(r0 + q.shape[0]) * 8].set(
                np.ascontiguousarray(q.transpose(0, 1, 3, 2)).reshape(-1).view(np.int32))
        self.out_norm = self._f32("output_norm.weight")
        order = self._usage()
        for il, kind in enumerate(s.kind):
            L, p = self.layers[il], "blk.%d." % il
            L["norm"] = self._f32(p + "attn_norm.weight")
            if kind == "ssm":
                L["in"] = self._q2(p + "ssm_in.weight")
                L["out"] = self._q2(p + "ssm_out.weight")
                L["conv_w"] = self._f32(p + "ssm_conv1d.weight")
                L["conv_b"] = self._f32(p + "ssm_conv1d.bias")
                L["A"] = self._f32(p + "ssm_a")
                L["D"] = self._f32(p + "ssm_d")
                L["dtb"] = self._f32(p + "ssm_dt.bias")
                L["gnorm"] = self._f32(p + "ssm_norm.weight")
            elif kind == "attn":
                L["qkv"] = self._q2cat([p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"])
                L["o"] = self._q2(p + "attn_output.weight")
            else:
                L["router"] = self._f32(p + "ffn_gate_inp.weight")
                L["rbias"] = self._f32(p + "exp_probs_b.bias")
                if self.placement is not None:
                    host = set(self.placement.on(il, "host")) or None
                else:
                    ordr = order.get(il, list(range(s.n_exp)))
                    host = set(ordr[self.n_card:]) if self.n_card < s.n_exp else None
                L["up"] = self._q2(p + "ffn_up_exps.weight", host)
                L["down"] = self._q2(p + "ffn_down_exps.weight", host)
                if host:
                    self._stage_tables(L, s.kind[:il].count("moe") % 2)
                    hidx = np.full(s.n_exp, -1, np.int32)
                    for i, e in enumerate(L["up"].hoff):
                        hidx[e] = i
                    L["hidx"] = cp.asarray(hidx)
                    L["dci"] = sum("hidx" in M for M in self.layers) - 1  # (its row in the decode slot tables)
                L["shup"] = self._q2(p + "ffn_up_shexp.weight")
                L["shdown"] = self._q2(p + "ffn_down_shexp.weight")
                if not host:                    # decode: the shared expert as entry n_exp of the up table
                    u, sh = L["up"], L["shup"]
                    L["upx"] = _Tables(u, cp.concatenate([u.tc, sh.tc]), cp.concatenate([u.ts, sh.ts]))
        cp.cuda.Device().synchronize()

    def _stage_tables(self, L, b):
        """prompt pointer tables of a layer with host experts: those experts are read from staging buffer b, where
        _stage() copies the layer's host blocks (up, then down, each laid out as in its pinned block)"""
        cp = self.cp
        if not hasattr(self, "stage"):
            n = max(L["up"].hbytes + L["down"].hbytes, self.max_host * self._expert_bytes())   # the largest layer
            self.stage = [cp.empty(n, cp.uint8) for _ in range(2)]
            self.copy_stream = cp.cuda.Stream(non_blocking=True)
            self.ev_staged = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
            self.ev_used = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
        base = self.stage[b].data.ptr
        for key, b0 in (("up", base), ("down", base + L["up"].hbytes)):
            w = L[key]
            pc, ps = [int(v) for v in w.tc.get()], [int(v) for v in w.ts.get()]
            nh = len(w.hoff)
            for i, e in enumerate(w.hoff):
                pc[e] = b0 + i * w.cw
                ps[e] = b0 + nh * w.cw + i * w.sw
            w.tcp = cp.asarray(np.array(pc, np.uint64))
            w.tsp = cp.asarray(np.array(ps, np.uint64))
        L["stage"] = b

    def _dec_slots(self):
        """decode with host experts: the device slots of each MoE layer (self.dcache experts: lt_cache_dec keeps the
        last used ones there), the slot owners and ages, and the per-choice pointer tables that lt_stage_dec fills"""
        if getattr(self, "dscr", None) is None:
            cp, s = self.cp, self.s
            L = next(L for L in self.layers if "hidx" in L)
            nl = sum("hidx" in M for M in self.layers)
            self.dsz = (L["up"].cw, L["up"].sw, L["down"].cw, L["down"].sw)
            self.dscr = cp.empty(nl * self.dcache * sum(self.dsz), cp.uint8)
            self.downer = cp.full((nl, self.dcache), -1, cp.int32)
            self.dage = cp.zeros((nl, self.dcache), cp.int32)
            self.dtick = cp.zeros(nl, cp.int32)
            self.dslot, self.dmiss = cp.zeros(s.n_used, cp.int32), cp.zeros(s.n_used, cp.int32)
            self.dsl = [cp.zeros(s.n_used, cp.uint64) for _ in range(4)]
            self.ident = cp.arange(s.n_used, dtype=cp.int32)
            cp.cuda.Device().synchronize()

    def _stage(self, il):
        """queue the copy of layer il's host experts into its staging buffer (copy stream), after the layer that
        used the buffer before it; the compute stream waits for the copy before the layer's grouped products"""
        L = self.layers[il]
        b = L["stage"]
        cs = self.copy_stream
        cs.wait_event(self.ev_used[b])
        dst = self.stage[b].data.ptr
        for key in ("up", "down"):
            w = L[key]
            self.cp.cuda.runtime.memcpyAsync(dst, w.hmem.ptr, w.hbytes, 1, cs.ptr)
            dst += w.hbytes
        self.ev_staged[b].record(cs)

    def _buf_bytes(self):
        s, C = self.s, self.chunk
        f = C * (2 * s.n_embd + s.d_in + s.d_xbc + 2 * s.d_inner + s.n_exp + s.n_ffsh + s.n_embd) * 4
        f += C * s.n_used * (s.n_ff + s.n_embd) * 4 + C * s.n_used * s.n_ff * 1.5
        if self.k35 and KBLAS:                             # _kblas: the fp32 weight block and inputs
            f += KDQ_BYTES + C * max(s.n_embd, s.d_inner, s.n_ffsh) * 4
        return int(f + C * max(s.d_in, s.n_ffsh) * 1.5 + s.vocab * 4 + (64 << 20))

    def _buffers(self):
        cp, s, C = self.cp, self.s, self.chunk
        E, U = s.n_embd, s.n_used
        z = lambda *shape, dt=cp.float32: cp.zeros(shape, dt)
        self.x = z(C, E)
        self.xn = z(C, E)
        W = max(E, s.d_inner, s.n_ffsh, s.n_head * s.hd)
        self.xq, self.xd, self.xs = z(C * W // 4, dt=cp.int32), z(C * W // 32), z(C * W // 32, dt=cp.int32)
        self.zx = z(C, s.d_in)
        self.xbc = z(C, s.d_xbc)
        self.y = z(C, max(s.d_inner, s.n_head * s.hd + 2 * s.n_kv * s.hd))
        self.att = z(C, s.n_head * s.hd)
        self.lg = z(C, s.n_exp)
        self.ids = z(C, U, dt=cp.int32)
        self.gw = z(C, U)
        self.cnt, self.off = z(s.n_exp, dt=cp.int32), z(s.n_exp, dt=cp.int32)
        self.xrow, self.where = z(C * U, dt=cp.int32), z(C * U, dt=cp.int32)
        self.u = z(C * U, s.n_ff)
        self.uq, self.ud, self.us = z(C * U * s.n_ff // 4, dt=cp.int32), z(C * U * s.n_ff // 32), \
            z(C * U * s.n_ff // 32, dt=cp.int32)
        self.yd = z(C * U, E)
        self.ush = z(C, s.n_ffsh)
        self.shq, self.shd, self.shs = z(C * s.n_ffsh // 4, dt=cp.int32), z(C * s.n_ffsh // 32), \
            z(C * s.n_ffsh // 32, dt=cp.int32)
        self.ysh = z(C, E)
        ns = max(NSPLIT, NSPLIT_TC)
        self.pm, self.pl = z(s.n_head * ns), z(s.n_head * ns)
        self.pacc = z(s.n_head * ns * s.hd)
        self.logits = z(s.vocab)
        self.am_v, self.am_i = z(64), z(64, dt=cp.int32)
        if self.k35:                                        # lt_mvk_* partial sums (the head: 3 chunks of its rows)
            self.kpart = z(max(3 * s.vocab, 4 * U * max(s.d_in, s.n_ffsh, E)))
            self.kpart2 = z(8 * E)
        self.tok = z(C, dt=cp.int32)
        self.dids = z(16, dt=cp.int32)                  # decode: the step's experts (and the shared expert)
        self.rcnt = z(4, dt=cp.int32)                   # lt_router_topk's block counter
        self.tok_out = z(4, dt=cp.int32)
        self.dpos = z(4, dt=cp.int32)
        self.kc, self.vc, self.st, self.cs = {}, {}, {}, {}
        for il, kind in enumerate(s.kind):
            if kind == "attn":
                self.kc[il] = z(self.ctx, s.n_kv * s.hd, dt=cp.float16)
                self.vc[il] = z(self.ctx, s.n_kv * s.hd, dt=cp.float16)
            elif kind == "ssm":
                self.st[il] = z(s.n_sh, 64, s.d_state)
                self.cs[il] = z(s.d_xbc, 3)
        self.h = np.frombuffer(cp.cuda.alloc_pinned_memory(64), np.int32, 16)
        if any("hidx" in L for L in self.layers):        # (not inside the decode graph's capture)
            self._dec_slots()
        cp.cuda.Device().synchronize()

    # ------------------------------------------------------------------ kernels
    def _mv(self, w, y, acc=0, ids=NULL, slots=1, xs=0, ys=0, xq=None, xd=None, xsum=None, rlast=0):
        """decode: y = W x (x = the quantized input: self.xq unless given)"""
        runs = w.cols // 32
        if self.k35:
            nch = (runs + 31) // 32
            self.k["lt_mvk_q2"]((nch, (w.rows + KRPB - 1) // KRPB, slots), (128,),
                                (w.tc, w.ts, ids, np.int32(w.rows), np.int32(w.cols), self.xq if xq is None else xq,
                                 self.xd if xd is None else xd, self.xs if xsum is None else xsum, np.int32(xs),
                                 self.kpart, np.int32(KRPB)))
            self.k["lt_mvk_sum"](((w.rows + 255) // 256, slots), (256,), (self.kpart, np.int32(nch),
                                                                         np.int32(w.rows), y, np.int32(ys),
                                                                         np.int32(acc)))
            return
        self.k["lt_mv_q2"](((max(w.rows, rlast) + 7) // 8, slots), (128,),
                           (w.tc, w.ts, ids, np.int32(w.rows), np.int32(w.cols),
                            self.xq if xq is None else xq, self.xd if xd is None else xd,
                            self.xs if xsum is None else xsum, np.int32(xs), y, np.int32(ys), np.int32(acc),
                            np.int32(rlast)))

    def _mm(self, w, T, y, ldy, acc=0, xq=None, xd=None, xsum=None, groups=None):
        """prompt: Y = W X for T tokens (groups: the MoE lists, then T = the largest group)"""
        cnt, off, xrow = groups if groups else (NULL, NULL, NULL)
        if getattr(w, "tcp", None) is not None:         # host experts: read from the staging buffer
            w = _Tables(w, w.tcp, w.tsp)
        if self.k35 and KBLAS and not groups and not K35_EXACT:
            return self._kblas(w, T, y, ldy, acc, xq, xd)
        kg = os.environ.get("KINDI_LT_KGRP", "mmk32")      # MoE groups: lt_mmk32 (exact sums) is the fastest here
        if self.k35 and not K35_EXACT and not (groups and kg == "mmk32"):    # scales folded into the fp32 tiles
            br, tn = (256, 32) if groups and kg == "mmf32" else (128, 128)
            self.k["lt_mmf%s_q2" % ("32" if br == 256 else "")](((w.rows + br - 1) // br, (T + tn - 1) // tn,
                                                             w.n if groups else 1), (256,),
                                                            (w.tc, w.ts, np.int32(w.rows), np.int32(w.cols), cnt, off,
                                                             xrow, np.int32(T), self.xq if xq is None else xq,
                                                             self.xd if xd is None else xd,
                                                             self.xs if xsum is None else xsum, y, np.int32(ldy),
                                                             np.int32(acc)))
            return
        if self.k35:
            tn = 32 if groups else 64
            self.k["lt_mmk%s_q2" % ("32" if tn == 32 else "")](((w.rows + 127) // 128, (T + tn - 1) // tn,
                                                             w.n if groups else 1), (256,),
                                (w.tc, w.ts, np.int32(w.rows), np.int32(w.cols), cnt, off, xrow, np.int32(T),
                                 self.xq if xq is None else xq, self.xd if xd is None else xd,
                                 self.xs if xsum is None else xsum, y, np.int32(ldy), np.int32(acc)))
            return
        if self.tc:
            tn = 32 if groups else 64
            self.k["lt_mm_tc%s" % ("32" if groups else "")](
                ((w.rows + 127) // 128, (T + tn - 1) // tn, w.n if groups else 1), (256,),
                (w.tc, w.ts, np.int32(w.rows), np.int32(w.cols), cnt, off, xrow, np.int32(T),
                 self.xq if xq is None else xq, self.xd if xd is None else xd,
                 self.xs if xsum is None else xsum, y, np.int32(ldy), np.int32(acc)))
            return
        tn = 32 if groups else 64                         # dp4a cards without tensor cores (sm_61, sm_62)
        self.k["lt_mmd%s_q2" % ("32" if groups else "")](((w.rows + 127) // 128, (T + tn - 1) // tn,
                                                         w.n if groups else 1), (256,),
                                                        (w.tc, w.ts, np.int32(w.rows), np.int32(w.cols), cnt, off,
                                                         xrow, np.int32(T), self.xq if xq is None else xq,
                                                         self.xd if xd is None else xd,
                                                         self.xs if xsum is None else xsum, y, np.int32(ldy),
                                                         np.int32(acc)))

    def _kblas(self, w, T, y, ldy, acc, xq=None, xd=None):
        """Kepler / P100 dense prompt product: blocks of weight rows as fp32 (lt_kdeq_q2) times the inputs as fp32
        (lt_kx2f) with cublasSgemm, as nanbeige.py's Kepler path does (cuBLAS reaches far more of these cards' fp32
        rate than a CUDA C tile). Numbers: a dequantized fp32 product."""
        cp = self.cp
        try:
            from cupy_backends.cuda.libs import cublas
        except ImportError:
            from cupy.cuda import cublas
        if getattr(self, "kdq", None) is None:
            self.kdq = cp.empty(KDQ_BYTES // 4, cp.float32)
            s = self.s
            self.kxf = cp.empty(self.chunk * max(s.n_embd, s.d_inner, s.n_ffsh), cp.float32)
            self.kab = np.array([1.0, 0.0, 1.0], np.float32)
            cp.cuda.Device().synchronize()
        k = self.k
        n = T * w.cols
        k["lt_kx2f"](((n + 255) // 256,), (256,), (self.xq if xq is None else xq, self.xd if xd is None else xd,
                                                   np.int32(T), np.int32(w.cols), self.kxf))
        h = cp.cuda.device.get_cublas_handle()
        cublas.setStream(h, self.stream.ptr)
        rb = max(64, KDQ_BYTES // (4 * w.cols) // 64 * 64)
        beta = self.kab.ctypes.data + (8 if acc else 4)
        for r0 in range(0, w.rows, rb):
            nr = min(rb, w.rows - r0)
            nw = nr * (w.cols // 16)
            k["lt_kdeq_q2"](((nw + 255) // 256,), (256,), (w.codes, w.sc, np.int32(w.cols), np.int32(r0), np.int32(nr),
                                                           self.kdq))
            cublas.sgemm(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, nr, T, w.cols, self.kab.ctypes.data,
                         self.kdq.data.ptr, w.cols, self.kxf.data.ptr, w.cols, beta, y.data.ptr + 4 * r0, ldy)

    def _groups4(self, T):
        """the MoE groups of a prompt chunk in pieces of up to 4 tokens, for lt_mvkg_q2: (device list, count). One
        wait for the group sizes."""
        cp = self.cp
        self.stream.synchronize()
        cnt = self.cnt.get()
        g = np.array([(e, j) for e in range(self.s.n_exp) for j in range(0, int(cnt[e]), 4)], np.int32).reshape(-1)
        if getattr(self, "glist", None) is None or self.glist.size < g.size:
            self.glist = cp.empty(max(g.size, 2 * (self.chunk * self.s.n_used // 4 + self.s.n_exp)), cp.int32)
        self.glist[:g.size].set(g)
        return self.glist, g.size // 2

    def _mvkg(self, w, ng, y, ldy, xrow, xq, xd, xs):
        """MoE groups 4 tokens at a time: lt_mvkg_q2 (fp32-exact, cards without dp4a) or lt_mvdg_q2 (dp4a)"""
        self.k["lt_mvkg_q2" if self.k35 else "lt_mvdg_q2"](((w.rows + 15) // 16, ng), (128,),
                             (w.tc, w.ts, np.int32(w.rows), np.int32(w.cols), self.glist, self.cnt, self.off,
                              xrow, xq, xd, xs, y, np.int32(ldy)))

    def _quant(self, x, rows, cols, ld, relu2, xq, xd, xs):
        runs = rows * (cols // 32)
        self.k["lt_quant"](((runs + 7) // 8,), (256,), (x, np.int32(rows), np.int32(cols), np.int32(ld),
                                                         np.int32(relu2), xq, xd, xs))

    def _layers(self, T):
        """the launches of one pass over T tokens (T = 1: a decode step) at positions *dpos .. *dpos + T - 1"""
        k, s = self.k, self.s
        E = s.n_embd
        eps = np.float32(s.eps)
        scale = np.float32(1.0 / np.sqrt(s.hd))
        gq = s.n_head // s.n_kv
        if T > 1 and hasattr(self, "stage"):            # a prompt chunk: stage the first two MoE layers
            for j in [j for j in range(s.n_layer) if s.kind[j] == "moe"][:2]:
                self._stage(j)
        for il, kind in enumerate(s.kind):
            L = self.layers[il]
            k["lt_rms_quant"]((T, 4 if T == 1 else 1), (256,), (self.x, L["norm"], np.int32(E), eps,
                                                                 self.xn if kind == "moe" else NULL,
                                            self.xq, self.xd, self.xs))
            if kind == "ssm":
                if T == 1:
                    self._mv(L["in"], self.zx)
                else:
                    self._mm(L["in"], T, self.zx, s.d_in)
                k["lt_conv"](((s.d_xbc + 255) // 256,), (256,), (self.zx, np.int32(s.d_in), np.int32(s.d_inner),
                                                                 np.int32(s.d_xbc), np.int32(T), L["conv_w"],
                                                                 L["conv_b"], self.cs[il], self.xbc))
                k["lt_ssm"]((s.n_sh, 1), (256,), (self.xbc, self.zx, np.int32(s.d_in), np.int32(s.d_inner + s.d_xbc),
                                                np.int32(T), np.int32(s.n_sh), np.int32(s.n_group), L["A"], L["D"],
                                                L["dtb"], self.st[il], self.y))
                k["lt_gnorm_quant"]((T, s.n_group), (128,), (self.y, L["gnorm"], np.int32(s.d_inner),
                                                   np.int32(s.d_inner // s.n_group), eps, self.xq, self.xd, self.xs))
                if T == 1:
                    self._mv(L["out"], self.x, acc=1)
                else:
                    self._mm(L["out"], T, self.x, E, acc=1)
            elif kind == "attn":
                w = L["qkv"]
                ld = w.rows
                if T == 1:
                    self._mv(w, self.y)
                else:
                    self._mm(w, T, self.y, ld)
                k["lt_kv_store"]((T,), (256,), (self.y, np.int32(ld), np.int32(s.n_head), np.int32(s.n_kv), self.dpos,
                                               np.int32(0), self.kc[il], self.vc[il]))
                if T == 1 and self.tc and gq == 16:
                    k["lt_fa_dec"]((NSPLIT_TC, s.n_kv), (32,), (self.y, self.kc[il], self.vc[il], self.dpos, scale,
                                                               self.pm, self.pl, self.pacc))
                    k["lt_attn_merge"](((s.n_head + 7) // 8,), (256,), (self.pm, self.pl, self.pacc,
                                                                        np.int32(s.n_head), np.int32(NSPLIT_TC),
                                                                        self.att))
                elif T == 1:
                    k["lt_attn_dec"]((s.n_kv, NSPLIT), (512,), (self.y, self.kc[il], self.vc[il], self.dpos,
                                                               np.int32(s.n_kv), np.int32(gq), scale, self.pm, self.pl,
                                                               self.pacc))
                    k["lt_attn_merge"](((s.n_head + 7) // 8,), (256,), (self.pm, self.pl, self.pacc,
                                                                        np.int32(s.n_head), np.int32(NSPLIT), self.att))
                elif self.tc and gq == 16:
                    k["lt_fa"](((16 * T + 63) // 64, s.n_kv), (128,), (self.y, np.int32(ld), np.int32(T), self.dpos,
                                                                       np.int32(0), self.kc[il], self.vc[il], scale,
                                                                       self.att))
                else:                       # fp32 (cards without tensor cores): 6 or 8 tokens per block
                    tb = 6 if self.pre6 and gq == 16 else 8
                    k["lt_attn_pre%d" % tb]((s.n_kv, (T + tb - 1) // tb), (256,), (self.y, np.int32(ld), self.kc[il],
                                                                               self.vc[il], self.dpos, np.int32(0),
                                                                               np.int32(T), np.int32(s.n_kv),
                                                                               np.int32(gq), scale, self.att))
                nhd = s.n_head * s.hd
                self._quant(self.att, T, nhd, nhd, 0, self.xq, self.xd, self.xs)
                if T == 1:
                    self._mv(L["o"], self.x, acc=1)
                else:
                    self._mm(L["o"], T, self.x, E, acc=1)
            else:
                U, F, FS = s.n_used, s.n_ff, s.n_ffsh
                if T == 1 and "upx" in L and self.cc >= 60:     # a step, all experts on the card: 4 launches
                    k["lt_router_topk"](((s.n_exp + 1) // 2,), (64,), (L["router"], self.xn, np.int32(s.n_exp),
                                                                      np.int32(E), self.lg, L["rbias"], np.int32(U),
                                                                      np.float32(s.w_scale), self.dids, self.gw,
                                                                      self.rcnt))
                    if self.k35:                # (lt_mvk_q2 has one row count: the shared expert separately)
                        self._mv(L["up"], self.u, ids=self.dids, slots=U, ys=F)
                        self._mv(L["shup"], self.u.ravel()[U * F:])
                    else:
                        self._mv(L["upx"], self.u, ids=self.dids, slots=U + 1, ys=F, rlast=FS)
                    k["lt_quant2"](((U * F + FS) // 32 // 8 + 1,), (256,),
                                   (self.u, np.int32(U), np.int32(F), self.uq, self.ud, self.us,
                                    self.u.ravel()[U * F:], np.int32(1), np.int32(FS), self.shq, self.shd, self.shs))
                    down, shdown = L["down"], L["shdown"]
                    k["lt_mv_q2_down"](((E + 3) // 4,), (256 // 2,),
                                       (down.tc, down.ts, self.dids, self.gw, np.int32(U), np.int32(E), np.int32(F),
                                        self.uq, self.ud, self.us, shdown.codes, shdown.sc, np.int32(FS),
                                        self.shq, self.shd, self.shs, self.x))
                    if getattr(self, "hook", None):
                        self.hook(il, T)
                    continue
                k["lt_router"](((s.n_exp * ((T + 7) // 8) + 1) // 2,), (64,), (L["router"], self.xn, np.int32(s.n_exp),
                                                                                np.int32(E), np.int32(T), self.lg))
                k["lt_topk"]((T,), (32,), (self.lg, L["rbias"], np.int32(s.n_exp), np.int32(U),
                                                np.float32(s.w_scale), self.ids, self.gw))
                up, down, shup, shdown = L["up"], L["down"], L["shup"], L["shdown"]
                ids, slotted = self.ids, False
                if T == 1 and "hidx" in L:      # host experts: copy the chosen ones to device slots first
                    self._dec_slots()
                    a, b, c, d = self.dsz
                    li, C = L["dci"], self.dcache
                    k["lt_cache_dec"]((1,), (32,), (self.ids, L["hidx"], self.downer[li], self.dage[li],
                                                    self.dtick[li:], np.int32(C), np.int32(U), self.dslot,
                                                    self.dmiss))
                    k["lt_stage_dec"]((U, 32), (256,), (self.ids, L["hidx"], up.tc, up.ts, down.tc, down.ts,
                                                         np.int64(a), np.int64(b), np.int64(c), np.int64(d),
                                                         np.uint64(self.dscr.data.ptr + li * C * (a + b + c + d)),
                                                         self.dslot, self.dmiss, *self.dsl))
                    up, down = _Tables(up, self.dsl[0], self.dsl[1]), _Tables(down, self.dsl[2], self.dsl[3])
                    ids, slotted = self.ident, True
                if T == 1:
                    self._mv(up, self.u, ids=NULL if slotted else ids, slots=U, ys=F)
                    self._mv(shup, self.ush)
                    self._quant(self.u, U, F, F, 1, self.uq, self.ud, self.us)
                    self._quant(self.ush, 1, FS, FS, 1, self.shq, self.shd, self.shs)
                    n = U * (F // 32)
                    rsh = FS // 32
                    if self.k35:
                        c1, c2 = (F // 32 + 31) // 32, (FS // 32 + 31) // 32
                        k["lt_mvk_q2"]((c1, (E + KRPB - 1) // KRPB, U), (128,),
                                       (down.tc, down.ts, ids, np.int32(E), np.int32(F), self.uq, self.ud, self.us,
                                        np.int32(F // 32), self.kpart, np.int32(KRPB)))
                        k["lt_mvk_q2"]((c2, (E + KRPB - 1) // KRPB, 1), (128,),
                                       (shdown.tc, shdown.ts, NULL, np.int32(E), np.int32(FS), self.shq, self.shd,
                                        self.shs, np.int32(0), self.kpart2, np.int32(KRPB)))
                        k["lt_mvk_down_sum"](((E + 255) // 256,), (256,), (self.kpart, np.int32(U), np.int32(c1),
                                                                           self.gw, self.kpart2, np.int32(c2),
                                                                           np.int32(E), self.x))
                    else:
                        k["lt_mv_q2_down"](((E + 3) // 4,), (128,),
                                           (down.tc, down.ts, ids, self.gw, np.int32(U), np.int32(E), np.int32(F),
                                            self.uq, self.ud, self.us, shdown.codes, shdown.sc, np.int32(FS),
                                            self.shq, self.shd, self.shs, self.x), shared_mem=(n + rsh) * 40)
                else:
                    k["lt_moe_lists"]((1,), (1024,), (self.ids, np.int32(T), np.int32(U), np.int32(s.n_exp),
                                                      self.cnt, self.off, self.xrow, self.where))
                    staged = "stage" in L
                    if staged:                  # this layer's host experts: staged by the copy stream
                        self.stream.wait_event(self.ev_staged[L["stage"]])
                    four = self.k35 and os.environ.get("KINDI_LT_KGRP", "mvkg") == "mvkg" or \
                        not self.k35 and not self.tc and os.environ.get("KINDI_LT_DGRP", "mvdg") == "mvdg"
                    if four and not staged:          # cards without tensor cores: 4 tokens at a time, decode-style
                        _, ng = self._groups4(T)
                        self._mvkg(up, ng, self.u, F, self.xrow, self.xq, self.xd, self.xs)
                        self._quant(self.u, T * U, F, F, 1, self.uq, self.ud, self.us)
                        self._mvkg(down, ng, self.yd, E, NULL, self.uq, self.ud, self.us)
                    else:
                        self._mm(up, T, self.u, F, groups=(self.cnt, self.off, self.xrow))
                        self._quant(self.u, T * U, F, F, 1, self.uq, self.ud, self.us)
                        self._mm(down, T, self.yd, E, xq=self.uq, xd=self.ud, xsum=self.us,
                                 groups=(self.cnt, self.off, NULL))
                    if staged:                  # the buffer is free: stage the MoE layer after the next one
                        self.ev_used[L["stage"]].record(self.stream)
                        nxt = [j for j in range(il + 1, s.n_layer) if s.kind[j] == "moe"]
                        if len(nxt) > 1:
                            self._stage(nxt[1])
                    self._mm(shup, T, self.ush, FS)
                    self._quant(self.ush, T, FS, FS, 1, self.shq, self.shd, self.shs)
                    self._mm(shdown, T, self.ysh, E, xq=self.shq, xd=self.shd, xsum=self.shs)
                    k["lt_moe_combine"]((T,), (256,), (self.yd, self.ysh, self.where, self.gw, np.int32(U),
                                                       np.int32(E), self.x))
            if getattr(self, "hook", None):
                self.hook(il, T)

    def _head(self, row):
        """greedy token from row `row` of x into tok_out[0] (and tok[0], the next step's input)"""
        k, s = self.k, self.s
        E = s.n_embd
        xr = self.x[row:row + 1]
        k["lt_rms_quant"]((1, 4), (256,), (xr, self.out_norm, np.int32(E), np.float32(s.eps), NULL, self.xq, self.xd,
                                         self.xs))
        if self.k35:
            nch = (E // 32 + 31) // 32
            k["lt_mvk_q8"]((nch, (s.vocab + KRPB - 1) // KRPB), (128,), (self.head_q, self.head_sc, np.int32(s.vocab),
                                                                       np.int32(E), self.xq, self.xd, self.xs,
                                                                       self.kpart, np.int32(KRPB)))
            k["lt_mvk_sum"](((s.vocab + 255) // 256, 1), (256,), (self.kpart, np.int32(nch), np.int32(s.vocab),
                                                                 self.logits, np.int32(0), np.int32(0)))
        else:
            k["lt_mv_q8"](((s.vocab + 7) // 8,), (256,), (self.head_q, self.head_sc, np.int32(s.vocab), np.int32(E),
                                                         self.xq, self.xd, self.logits), shared_mem=E // 32 * 36)
        nb = (s.vocab + 4095) // 4096
        k["lt_argmax_part"]((nb,), (256,), (self.logits, np.int32(s.vocab), self.am_v, self.am_i))
        k["lt_argmax_final"]((1,), (32,), (self.am_v, self.am_i, np.int32(nb), self.tok_out, self.tok))

    def _step_launches(self):
        self.k["lt_embed"]((1,), (256,), (np.uint64(self.emb_mem.ptr), self.tok, np.int32(self.s.n_embd), self.x))
        self._layers(1)
        self._head(0)
        self.k["lt_advance"]((1,), (32,), (self.dpos, np.int32(1)))

    # ------------------------------------------------------------------ public
    def reset(self):
        """forget the sequence: zero the Mamba states (the K/V cache is overwritten by position)"""
        self.stream.synchronize()
        for a in list(self.st.values()) + list(self.cs.values()):
            a.fill(0)
        self.cp.cuda.Device().synchronize()             # (the fills run on the default stream)
        self.pos = 0

    def snapshot(self):
        """keep a copy of the Mamba states at the current position (restore() goes back to it: the K/V before that
        position stays valid while the same tokens stay there)"""
        if not hasattr(self, "snap"):
            self.snap = {il: (a.copy(), self.cs[il].copy()) for il, a in self.st.items()}
        self.stream.synchronize()
        for il, a in self.st.items():
            self.snap[il][0][...] = a
            self.snap[il][1][...] = self.cs[il]
        self.cp.cuda.Device().synchronize()
        self.snap_pos = self.pos

    def restore(self):
        for il, (a, c) in self.snap.items():
            self.st[il][...] = a
            self.cs[il][...] = c
        self.cp.cuda.Device().synchronize()
        self.pos = self.snap_pos

    def prefill(self, ids, pos=None):
        """read ids at positions pos .. (default: after the current sequence); return the greedy next token. The
        Mamba states must hold the sequence up to pos (reset() for pos 0)."""
        cp = self.cp
        pos = self.pos if pos is None else pos
        assert pos + len(ids) <= self.ctx, "the prompt does not fit the context (%d)" % self.ctx
        with self.stream:
            for c0 in range(0, len(ids), self.chunk):
                part = ids[c0:c0 + self.chunk]
                T = len(part)
                self.k["lt_set"]((1,), (32,), (self.dpos, np.int32(pos + c0)))
                self.tok[:T].set(np.asarray(part, np.int32), stream=self.stream)
                self.k["lt_embed"]((T,), (256,), (np.uint64(self.emb_mem.ptr), self.tok, np.int32(self.s.n_embd),
                                                  self.x))
                self._layers(T)
            self._head(T - 1)
            self.k["lt_set"]((1,), (32,), (self.dpos, np.int32(pos + len(ids))))
            cp.cuda.runtime.memcpyAsync(self.h.ctypes.data + 8, self.tok_out.data.ptr, 4, 2, self.stream.ptr)
        self.stream.synchronize()
        self.pos = pos + len(ids)
        return int(self.h[2])

    def step(self, tok=None):
        """read one token (default: the last greedy token, already on the card) at the current position; return the
        greedy next token"""
        cp = self.cp
        if self.graph is None:
            if os.environ.get("KINDI_LT_GRAPH", "1") == "1":
                self.graph = _capture(self.stream, self._step_launches)
            else:
                self.graph = _capture(self.stream, self._step_launches, replay=True)
        assert self.pos < self.ctx, "the context is full (%d)" % self.ctx
        if tok is not None:
            with self.stream:
                self.k["lt_set"]((1,), (32,), (self.tok, np.int32(tok)))
        self.graph.launch(self.stream)
        cp.cuda.runtime.memcpyAsync(self.h.ctypes.data + 8, self.tok_out.data.ptr, 4, 2, self.stream.ptr)
        self.stream.synchronize()
        self.pos += 1
        return int(self.h[2])

    def candidates(self, w, cap=4096):
        """the tokens whose logit is within w of the best, after the last step or prompt: (ids, logits), or None
        when more than cap qualify. One kernel, one copy, one wait (the server's sampling path)."""
        cp = self.cp
        if getattr(self, "c_cnt", None) is None:
            self.c_cnt, self.c_i, self.c_v = cp.zeros(1, cp.int32), cp.zeros(cap, cp.int32), cp.zeros(cap, cp.float32)
            self.h_c = np.frombuffer(cp.cuda.alloc_pinned_memory(8 * cap + 64), np.uint8, 8 * cap + 64)
            cp.cuda.Device().synchronize()
        nb = (self.s.vocab + 4095) // 4096
        with self.stream:
            self.k["lt_set"]((1,), (32,), (self.c_cnt, np.int32(0)))
            self.k["lt_cand"]((64,), (256,), (self.logits, np.int32(self.s.vocab), self.am_v, np.int32(nb),
                                              np.float32(w), np.int32(cap), self.c_cnt, self.c_i, self.c_v))
            hp = self.h_c.ctypes.data
            cp.cuda.runtime.memcpyAsync(hp, self.c_cnt.data.ptr, 4, 2, self.stream.ptr)
            cp.cuda.runtime.memcpyAsync(hp + 64, self.c_i.data.ptr, 4 * cap, 2, self.stream.ptr)
            cp.cuda.runtime.memcpyAsync(hp + 64 + 4 * cap, self.c_v.data.ptr, 4 * cap, 2, self.stream.ptr)
        self.stream.synchronize()
        n = int(self.h_c[:4].view(np.int32)[0])
        if n > cap:
            return None
        idx = self.h_c[64:64 + 4 * cap].view(np.int32)[:n].astype(np.int64)
        val = self.h_c[64 + 4 * cap:64 + 8 * cap].view(np.float32)[:n].astype(np.float64)
        o = np.lexsort((idx, -val))                      # a fixed order: by logit, then by id
        return idx[o], val[o]

    def generate(self, ids, n, stop=()):
        """greedy: n tokens after ids (from position 0); stops after a token in stop"""
        self.reset()
        out = [self.prefill(ids, 0)]
        while len(out) < n and out[-1] not in stop:
            out.append(self.step())
        return out
