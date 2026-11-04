"""Prompt reading in chunks: the matrix products of up to KINDI_CHUNK prompt tokens (default 256) run as one GEMM
per weight (each weight tile is unpacked once and used by 128 tokens), the GDN steps run T tokens per launch, and
the per-token work around them runs T tokens per launch (blockIdx.y = token). The decode path is unchanged.

Kernels: the GEMM (k_mm_g5n) is the layout-independent mm_core with the 5-trit lane-layout unpack stage; the
multi-token GDN kernels (k_gdn_pre_T, k_gdn_deltan_T) keep the per-token arithmetic of k_gdn_pre and k_gdn_deltan;
the _T kernels are generated here from the decode kernels (a device copy plus a wrapper that offsets every pointer
by token * stride), so their arithmetic is the decode kernels' own. Attention (chunk_attn: the chunk's K/V are
stored first, then each token sees the positions <= its own) runs TS tokens per launch as tiled products with the
decode kernels' arithmetic (bit-identical); TS is chosen per chunk from the free scratch (U2 for the scores, XP
for the chunk parts - both dead during attention), so it is planned at feed time, not in the plan.

One or two cards (W = 2: the row-parallel outputs are exchanged between plan segments, card 0's part added first on
both cards), 5-trit lane weights (1080) or 2-bit k_g4 weights (K1200, K420: mm_core with the 2-bit unpack stage of
chunk1.cu, fp32). KINDI_CHUNK=0: prompts go through the batch plan as before."""
import ctypes
import os
import re

import numpy as np
import cupy as cp

from . import tp as _tp

HERE = os.path.dirname(os.path.abspath(__file__))
i32 = np.int32
WRAP = ("k_rmsfold", "k_smfold", "k_rmsw", "k_rmsw_s", "k_rope2", "k_store_q8", "k_q8row", "k_fold_fwht")
OWN = ("k_gdn_pre_T", "k_gdn_deltan_T", "k_sumsplits", "k_mm_g5n", "k_att_qk_T", "k_att_softmax_T", "k_att_pv_T",
       "k_att_comb_T", "k_att_flash_T", "k_att_merge_T", "k_q8x_T", "k_q8x_T2", "k_mm_g5q", "k_mm_g4", "k_mm_g4s", "k_mm_g4k", "k_add2")


def _kernel_text(src, name):
    m = re.search(r'extern "C" __global__ void (?:__launch_bounds__\(\d+\) )?%s\(' % name, src)
    if not m:
        raise SystemExit("kindi chunk: kernel %s not found" % name)
    i = src.index("{", m.end())
    depth, j = 0, i
    while True:
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    head = src[m.start():i]
    params = head[head.index("(") + 1:head.rindex(")")]
    return params, src[i:j + 1]


def _port(src):
    """the HIP/CUDA portability preamble of kindi.cu (KHIP, KSHFLX, KSHFL): copies of its kernels need it"""
    a, b = src.find("#ifndef KHIP   /* KINDI-PORT-BEGIN */"), src.find("/* KINDI-PORT-END */")
    return src[a:b + len("/* KINDI-PORT-END */")] + chr(10) if a >= 0 and b > a else ""


def _wrappers(src, names=WRAP):
    """device copies of the decode kernels in names (default WRAP), and token-batched wrappers k_*_T(..., int T_, strides...)"""
    out, nptr = [_port(src)], {}
    for name in names:
        params, body = _kernel_text(src, name)
        ps = [p.strip() for p in params.split(",")]
        ids = [re.findall(r"[A-Za-z_]\w*", p)[-1] for p in ps]
        ptr = [("*" in p) for p in ps]
        out.append("__device__ __forceinline__ void d_%s(%s)%s\n" % (name, params, body))
        strides = ["int s_%s" % i for i, q in zip(ids, ptr) if q]
        call = ", ".join("%s + t_ * s_%s" % (i, i) if q else i for i, q in zip(ids, ptr))
        out.append('extern "C" __global__ void %s_T(%s, int T_, %s){\n  if((int)blockIdx.y >= T_) return;\n'
                   '  const long long t_ = blockIdx.y;\n  d_%s(%s);\n}\n' % (name, params, ", ".join(strides), name, call))
        nptr[name] = sum(ptr)
    return "\n".join(out), nptr


class F16Kernels(object):
    """f16p.cu (the fp16 prompt path)"""
    def __init__(self):
        m = cp.RawModule(code=open(os.path.join(HERE, "f16p.cu"), encoding="utf-8").read(), options=("-std=c++11",))
        for n in ("k_f16_dq", "k_f16_dq5", "k_f16_x", "k_f16_acc"):
            setattr(self, n, m.get_function(n))


def f16_prompt(card):
    """the fp16 prompt path (f16p.cu): on by default for sm_60 (Tesla P100); KINDI_P100_FP16_PROMPT=1 turns it on for any
    card of sm_53 or later, =0 turns it off"""
    if cp.cuda.runtime.is_hip:
        return False
    p = cp.cuda.runtime.getDeviceProperties(card)
    cc = p["major"] * 10 + p["minor"]
    v = os.environ.get("KINDI_P100_FP16_PROMPT", "")
    return cc >= 53 and (v == "1" or (v != "0" and cc == 60))


def _gemm_f16(s, Wt, R, C, X, ldx, Yb, ldy, quant, lane=False):
    """the plan step of one product on the fp16 path: Y[t][r] = sum over pieces of 512 columns of the fp16 GEMM of
    the piece, added in fp32 (f16p.cu). lane: Wt in the 5-trit lane layout (else the pl layout of tcp.cu)"""
    try:
        from cupy_backends.cuda.libs import cublas
    except ImportError:
        from cupy.cuda import cublas
    c = s.chunk
    F, X16, W16, C16 = c["F16"], c["X16"], c["W16"], c["C16"]
    NG = C // 128
    one, zero = np.array([1.0], np.float16), np.array([0.0], np.float16)
    # cublasGemmEx's compute type: a cudaDataType before CUDA 11, a cublasComputeType_t since
    ct = 64 if cp.cuda.runtime.runtimeGetVersion() >= 11000 else 2     # CUBLAS_COMPUTE_16F / CUDA_R_16F
    R16 = 2                                                            # CUDA_R_16F

    def step(n):
        with cp.cuda.Device(s.card):
            if quant:
                F.k_f16_x((min(4096, (n * C + 255) // 256),), (256,), (X, i32(ldx), i32(C), i32(n), X16))
            h = cp.cuda.device.get_cublas_handle()
            cublas.setStream(h, 0)
            for g0 in range(0, NG, 4):
                ng = min(4, NG - g0)
                if lane:
                    F.k_f16_dq5(((R * ng * 26 + 255) // 256,), (256,), (Wt, i32(R), i32(NG), i32(g0), i32(ng), W16))
                else:
                    F.k_f16_dq(((R * ng * 7 + 255) // 256,), (256,), (Wt, i32(R), i32(NG), i32(g0), i32(ng), W16))
                cublas.gemmEx(h, cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N, R, n, 128 * ng,
                              one.ctypes.data, W16.data.ptr, R16, 128 * ng, X16.data.ptr + 2 * 128 * g0, R16, C,
                              zero.ctypes.data, C16.data.ptr, R16, R, ct, -1)
                F.k_f16_acc((min(4096, (n * R + 255) // 256),), (256,), (C16, i32(R), i32(n), Yb, i32(ldy), i32(g0 == 0)))
    return step


class ChunkKernels(object):
    def __init__(self):
        base = open(os.path.join(HERE, "kindi.cu"), encoding="utf-8").read()
        wr, self.nptr = _wrappers(base)
        own = open(os.path.join(HERE, "chunk.cu"), encoding="utf-8").read()
        own1 = open(os.path.join(HERE, "chunk1.cu"), encoding="utf-8").read()
        mod = cp.RawModule(code=base + "\n" + wr + "\n" + own + "\n" + own1)
        for k in WRAP:
            setattr(self, k + "_T", mod.get_function(k + "_T"))
        for k in OWN:
            setattr(self, k, mod.get_function(k))


def enabled(eng):
    v = os.environ.get("KINDI_CHUNK", "256")
    return v not in ("0", "", "1") and eng.W in (1, 2) and all(s.g5l or not s.g5 or getattr(s, "wl", None) for s in eng.cards)


def reserve(eng):
    """the chunk buffers, allocated before the context is fitted to the free memory"""
    eng.chunk_ok = False
    if not enabled(eng):
        return
    T = int(os.environ.get("KINDI_CHUNK", "512" if all(f16_prompt(s.card) for s in eng.cards) else "256"))
    for s in eng.cards:
        with cp.cuda.Device(s.card):
            z = lambda n, t=cp.float32: cp.zeros(int(n), dtype=t)  # noqa: E731
            D, NFF, HK, HD = s.D, s.NFF, s.HK, s.HD
            work = max(D, NFF, s.NV * HK)
            c = s.chunk = {"T": T, "work": work}
            c["HA"], c["HB"], c["XN"], c["Y"] = z(T * D), z(T * D), z(T * D), z(T * D)
            c["XP"] = z(T * work)
            uw = max(NFF, s.CONV_CH + 2 * s.NK * HK, 2 * s.NH * HD, 3 * s.NHKV * HD + 2 * s.NH * HD)
            c["U1"], c["U2"] = z(T * uw), z(T * uw)
            c["OUT"] = z(T * max(s.NV * HK, s.NH * HD))
            # split parts: the 5-trit path splits only the small K/V products; the 2-bit path may split D-row ones
            ctc = getattr(s, "tc_layout", False) and getattr(s, "wl", None) is None and not getattr(s, "tc_lane", False)
            # (HIP keeps the lane copy, sm_60 the lane layout itself: no tc GEMMs)
            c["SPL"] = z(1 if ctc else 8 * T * max(s.NHKV * HD, D))   # (small models split D-row products too)   # (tc: not used)
            c["YP"] = z(T * D) if eng.W == 2 else None          # the other card's row-parallel part
            tv = os.environ.get("KINDI_CHUNK_TILE", "")
            c["small"] = tv == "64" or (tv != "128" and cp.cuda.runtime.getDeviceProperties(s.card)["major"] < 5)
            c["GB"], c["GG"] = z(T * s.NV), z(T * s.NV)
            c["QQ"], c["QD"] = cp.zeros(T * s.NH * HD, dtype=cp.int8), z(T * s.NH * HD // 32)
            prop = cp.cuda.runtime.getDeviceProperties(s.card)
            c["q8"] = os.environ.get("KINDI_CHUNK_Q8", "1") == "1" and prop["major"] * 10 + prop["minor"] >= 61
            if c["q8"] and not ctc:   # int8 activations (dp4a): XQ[k4][t], XS[group][t]
                c["XQ"], c["XS"] = cp.zeros(T * work // 4, dtype=cp.uint32), z(T * work // 128)
            c["tc"] = ctc                               # tc layout (spec.py): tensor-core GEMMs, XQ in the tc order
            c["f16"] = f16_prompt(s.card) and (ctc or s.g5)
            if c["f16"]:                                # the fp16 prompt path (f16p.cu): inputs, a weight piece, a result
                rw = max(NFF, D, uw, s.NV * HK, s.NH * HD)
                c["X16"] = cp.zeros(T * work, dtype=cp.float16)
                c["W16"] = cp.zeros(rw * 512, dtype=cp.float16)
                c["C16"] = cp.zeros(T * rw, dtype=cp.float16)
                c["F16"] = F16Kernels()
            if c["tc"]:
                c["XQ"], c["XS"] = cp.zeros(T * work // 4, dtype=cp.uint32), z(T * work // 128)
                c["XA"] = cp.zeros(T * work // 128, dtype=cp.int32)
            s.ck = ChunkKernels()               # kernel handles belong to the current device
            if not cp.cuda.runtime.is_hip:                        # CUDA only (fixed bank layout on ROCm)
                fp = ctypes.c_void_p(s.ck.k_mm_g4k.kernel.ptr)
                _tp._drv().cuFuncSetSharedMemConfig(fp, 2)        # 8-byte banks (Kepler)
                _tp._drv().cuFuncSetCacheConfig(fp, 1)            # prefer shared memory


def _set(P, pos, T, toks):
    for h, d in P.dyn:
        if d.var == "pos":
            h.value = d.a + d.b * pos
        elif d.var == "T":
            h.value = d.a + d.b * T
        else:
            h.value = d.a + d.b * toks[int(d.var[3:])]


def build(eng):
    if not enabled(eng):
        return
    for s in eng.cards:
        with cp.cuda.Device(s.card):
            s.plan_chunk = _build(s)
    eng.chunk_ok = True
    eng.batch_max = eng.cards[0].chunk["T"]     # the server may hand over whole chunks


def _build(s):
    tp, K, CK, c = s.tp, s.k, s.ck, s.chunk
    T, work = c["T"], c["work"]
    D, HB, HK, HD = s.D, s.HB, s.HK, s.HD
    NK, NV, NH, NHKV, NFF, CC = s.NK, s.NV, s.NH, s.NHKV, s.NFF, s.CONV_CH
    W2 = tp.W == 2
    nT = lambda: _tp.Dyn("T", 0, 0)  # noqa: E731
    TT = _tp.Dyn("T", 0, 1)
    U1, U2, OUT, XP, Y, XN = c["U1"], c["U2"], c["OUT"], c["XP"], c["Y"], c["XN"]
    v = lambda a, k, n: a[k * n:(k + 1) * n]  # noqa: E731
    st = {"h": c["HA"], "h2": c["HB"], "pending": 0}
    P = _tp.Plan()
    P.cores = {}
    P.xch = set()                           # segments after which the row-parallel parts are exchanged (W = 2)
    if HD != 256:
        raise SystemExit("kindi chunk: the chunk attention kernels expect a head size of 256")

    def tadd(kern, gx, block, args, strides):
        name = kern.__name__ if hasattr(kern, "__name__") else None
        P.add(kern, (gx, T), block, tuple(args) + (_tp.Dyn("T", 0, 1),) + tuple(i32(x) for x in strides))

    def gemm(name, X, ldx, Yb, ldy, quant=False):
        if c["tc"]:
            return gemm_tc(name, X, ldx, Yb, ldy, quant)
        if not s.g5:
            return gemm4(name, X, ldx, Yb, ldy)
        _, W5, R, C = (getattr(s, "wl", None) or s.w)[name]       # the lane layout (a second copy when decode is row)
        R, C = int(R), int(C)
        if c["f16"]:                                     # KINDI_P100_FP16_PROMPT=1: cuBLAS fp16 (f16p.cu)
            P.call(_gemm_f16(s, W5, R, C, X, ldx, Yb, ldy, quant, lane=True))
            return
        NG = C // 128
        gx, gy = R // 128, (T + 127) // 128
        if c["q8"]:
            if quant:                       # the first product of an input: its int8 rows
                if os.environ.get("KINDI_Q8X2", "1") == "1":      # coalesced stores (same values)
                    P.add(CK.k_q8x_T2, (NG, (T + 63) // 64), (256,), (X, i32(ldx), c["XQ"], c["XS"], i32(T), TT))
                else:
                    P.add(CK.k_q8x_T, (NG, T), (128,), (X, i32(ldx), c["XQ"], c["XS"], i32(T), TT))
            P.add(CK.k_mm_g5q, (gx, gy), (256,), (W5, i32(R), i32(NG), c["XQ"], c["XS"], i32(T), Yb, i32(ldy), TT))
            return
        sp = 1 if gx * gy >= 40 else max(1, min(8, NG, -(-80 // (gx * gy))))
        if sp == 1:
            P.add(CK.k_mm_g5n, (gx, gy, 1), (256,), (W5, i32(R), i32(NG), X, i32(ldx), Yb, i32(ldy), _tp.Dyn("T", 0, 1), i32(0)))
        else:
            assert sp * T * R <= c["SPL"].size
            P.add(CK.k_mm_g5n, (gx, gy, sp), (256,), (W5, i32(R), i32(NG), X, i32(ldx), c["SPL"], i32(R), _tp.Dyn("T", 0, 1), i32(0)))
            P.add(CK.k_sumsplits, ((T * R + 255) // 256,), (256,), (c["SPL"], Yb, i32(ldy), i32(R), _tp.Dyn("T", 0, 1), i32(sp)))

    def gemm_tc(name, X, ldx, Yb, ldy, quant):
        # int8 tensor-core GEMM on the tc layout (tc.cu k_tc_mm: 64 tokens per block row, groups in order)
        SK = s.sk
        _, Wt, R, C = s.w[name]
        R, C = int(R), int(C)
        NG = C // 128
        if c["f16"]:                                     # KINDI_P100_FP16_PROMPT=1: cuBLAS fp16 (f16p.cu)
            P.call(_gemm_f16(s, Wt, R, C, X, ldx, Yb, ldy, quant))
            return
        if quant:
            P.add(SK.k_tc_q8, (NG, T), (128,), (X, i32(ldx), c["XQ"], c["XS"], c["XA"], i32(NG), TT))
        if hasattr(SK, "k_tp_mm"):                       # Pascal (tcp.cu): dp4a, 128 rows x 128 tokens per block
            P.add(SK.k_tp_mm, (R // 128, (T + 127) // 128), (256,),
                  (Wt, i32(R), i32(C // 1024), c["XQ"], c["XS"], c["XA"], Yb, i32(ldy), TT))
            return
        if os.environ.get("KINDI_TC_MM", "2") == "2":     # activations staged in shared memory (k_tc_mm2)
            P.add(SK.k_tc_mm2, ((R // 16 + 7) // 8, (T + 63) // 64), (256,),
                  (Wt, i32(R), i32(C // 1024), c["XQ"], c["XS"], c["XA"], Yb, i32(ldy), TT))
            return
        nb = int(os.environ.get("KINDI_TC_MM_NB", "8"))
        f = {4: SK.k_tc_mm4, 8: SK.k_tc_mm8, 16: SK.k_tc_mm16}[nb]
        P.add(f, ((R // 16 + 3) // 4, (T + 8 * nb - 1) // (8 * nb)), (128,),
              (Wt, i32(R), i32(C // 1024), c["XQ"], c["XS"], c["XA"], Yb, i32(ldy), TT))

    def gemm4(name, X, ldx, Yb, ldy):
        Wb, Sd, R, C = s.w[name]
        R, C = int(R), int(C)
        kt = os.environ.get("KINDI_CHUNK_TILE", "") != "128"      # the 128 x 64, 16-column-step tile (see chunk1.py)
        f = CK.k_mm_g4k if kt else CK.k_mm_g4
        gx, gy = (R // 128, (T + 63) // 64) if kt else (R // 128, (T + 127) // 128)
        slots = 2 * s.nsm
        sp = 1 if gx * gy >= slots else max(1, min(8, C // 128, -(-2 * slots // (gx * gy)), c["SPL"].size // (T * R)))
        if sp == 1:
            P.add(f, (gx, gy, 1), (256,), (Wb, Sd, i32(R), i32(C), X, i32(ldx), Yb, i32(ldy), TT))
        else:
            P.add(f, (gx, gy, sp), (256,), (Wb, Sd, i32(R), i32(C), X, i32(ldx), c["SPL"], i32(R), TT))
            P.add(CK.k_sumsplits, ((T * R + 255) // 256,), (256,), (c["SPL"], Yb, i32(ldy), i32(R), TT, i32(sp)))

    def mixin(wname):
        if st["pending"] and W2:
            # the parts are exchanged after this segment; then Y = card 0's part + card 1's part on both cards
            P.xch.add(len(P.segs) - 1)
            P.cut()
            a, b = (Y, c["YP"]) if s.card == 0 else (c["YP"], Y)
            P.add(CK.k_add2, ((T * D + 255) // 256,), (256,), (a, b, Y, _tp.Dyn("T", 0, D)))
        if st["pending"]:
            tadd(CK.k_rmsfold_T, D // HB, (512,), (st["h"], Y, st["h2"], s.f(wname), s.sD, XN, XP, i32(D), i32(1), i32(HB)),
                 (D, D, D, 0, 0, D, work))
            st["h"], st["h2"] = st["h2"], st["h"]
        else:
            tadd(CK.k_rmsfold_T, D // HB, (512,), (st["h"], Y, st["h2"], s.f(wname), s.sD, XN, XP, i32(D), i32(0), i32(HB)),
                 (D, D, D, 0, 0, D, work))
        st["pending"] = 0

    def gdn(il, last):
        p = "blk.%d." % il
        mixin(p + "attn_norm.weight")
        QKV, QK, ZV, CS = U1, U1[T * CC:], U2, U2[T * NV * HK:]
        gemm(p + "attn_qkv.weight", XP, work, QKV, CC, True)
        gemm(p + "attn_gate.weight", XP, work, ZV, NV * HK)
        nblk = NV + 2 * NK + (NV * HK + 255) // 256
        P.add(CK.k_gdn_pre_T, (nblk, T), (256,), (XN, i32(D), s.f(p + "ssm_beta.weight"), s.f(p + "ssm_alpha.weight"),
                                                s.f(p + "ssm_dt.bias"), s.f(p + "ssm_a"), c["GB"], c["GG"], i32(NV), i32(D),
                                                QKV, i32(CC), s.conv[il], s.f(p + "ssm_conv1d.weight"), CS, i32(CC),
                                                QK, i32(2 * NK * HK), i32(CC), i32(NV), i32(NK), i32(HK), TT))
        P.add(CK.k_gdn_deltan_T, (NV,), (HK,), (s.M[il], QK, QK[NK * HK:], i32(2 * NK * HK), CS[2 * NK * HK:], i32(CC),
                                                 c["GG"], c["GB"], i32(NV), s.f(p + "ssm_norm.weight"), OUT, i32(NV * HK),
                                                 np.float32(1.0 / np.sqrt(HK)), i32(NK), TT))
        if last:
            st["pending"] = 0
            return
        tadd(CK.k_smfold_T, NV * HK // HB, (512,), (ZV, OUT, s.sV, XP, i32(HK), i32(NK), i32(NV // NK), i32(HB)),
             (NV * HK, NV * HK, 0, work))
        gemm(p + "ssm_out.weight", XP, work, Y, D, True)
        st["pending"] = 1

    def attn(il, kvi, last):
        p = "blk.%d." % il
        mixin(p + "attn_norm.weight")
        L = NHKV * HD
        QN = U1
        KRO, VB, KN, FO, QR = U2, U2[T * L:], U2[2 * T * L:], U2[3 * T * L:], U2[3 * T * L + T * NH * HD:]
        if not last:
            gemm(p + "attn_q.weight", XP, work, QN, 2 * NH * HD, True)
        gemm(p + "attn_k.weight", XP, work, KRO, L, last)
        gemm(p + "attn_v.weight", XP, work, VB, L)
        tadd(CK.k_rmsw_T, (NHKV + 7) // 8, (256,), (KRO, s.f(p + "attn_k_norm.weight"), KN, i32(HD), i32(NHKV)), (L, 0, L))
        rd = s.cos_t.shape[1]
        cs_ = _tp.Dyn("pos", s.cos_t.data.ptr, rd * 4, ptr=True)
        sn_ = _tp.Dyn("pos", s.sin_t.data.ptr, rd * 4, ptr=True)
        if not last:
            tadd(CK.k_rmsw_s_T, (NH + 7) // 8, (256,), (QN, i32(2 * HD), s.f(p + "attn_q_norm.weight"), FO, i32(HD), i32(NH)),
                 (2 * NH * HD, 0, NH * HD))
            tadd(CK.k_rope2_T, NH + NHKV, (256,), (FO, KN, cs_, sn_, QR, KRO, i32(NH), i32(HD)),
                 (NH * HD, L, rd, rd, NH * HD, L))
        else:
            tadd(CK.k_rope2_T, NHKV, (256,), (KN, KN, cs_, sn_, KRO, KRO, i32(0), i32(HD)), (L, L, rd, rd, L, L))
        nb0 = L // 32
        tadd(CK.k_store_q8_T, (64 * nb0 + 1023) // 1024, (min(1024, 64 * nb0),),
             (KRO, VB, _tp.Dyn("pos", s.Kq[kvi].data.ptr, L, ptr=True), _tp.Dyn("pos", s.Kd[kvi].data.ptr, nb0 * 2, ptr=True),
              _tp.Dyn("pos", s.Vq[kvi].data.ptr, L, ptr=True), _tp.Dyn("pos", s.Vd[kvi].data.ptr, nb0 * 2, ptr=True), i32(nb0)),
             (L, L, L, nb0, L, nb0))
        if last:
            st["pending"] = 0
            return
        QL = NH * HD
        tadd(CK.k_q8row_T, (QL + 1023) // 1024, (1024,), (QR, c["QQ"], c["QD"], i32(QL // 32)), (QL, QL, QL // 32))
        P.cores[len(P.segs) - 1] = kvi       # the attention itself runs here, planned per chunk (run_core)
        P.cut()
        tadd(CK.k_fold_fwht_T, QL // HB, (512,), (OUT, s.sA, XP, i32(HB)), (QL, 0, work))
        gemm(p + "attn_output.weight", XP, work, Y, D, True)
        st["pending"] = 1

    def ffn(il):
        p = "blk.%d." % il
        mixin(p + "post_attention_norm.weight")
        gemm(p + "ffn_gate.weight", XP, work, U1, NFF, True)
        gemm(p + "ffn_up.weight", XP, work, U2, NFF)
        tadd(CK.k_smfold_T, NFF // HB, (512,), (U1, U2, s.sF, XP, i32(0), i32(0), i32(0), i32(HB)), (NFF, NFF, 0, work))
        gemm(p + "ffn_down.weight", XP[s.d0 - s.g0:], work, Y, D, True)
        st["pending"] = 1

    emb = tp.emb
    for k in range(T):
        P.add(K.k_deq_emb4, (D // HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, v(c["HA"], k, D),
                                                 _tp.DynT(k), i32(D), i32(HB)))
    kvi = 0
    for il in range(s.NL):
        last = il == s.NL - 1
        if (il + 1) % s.dims["full_attention_interval"] == 0:
            attn(il, kvi, last)
            kvi += 1
        else:
            gdn(il, last)
        if not last:
            ffn(il)
    return P


def run_core(s, kvi, pos, T):
    """attention of the chunk's T tokens (positions pos ..) in groups of TS tokens: scores in U2, chunk parts in XP"""
    c, CK = s.chunk, s.ck
    NH, NHKV, G, QL = s.NH, s.NHKV, s.G, s.NH * s.HD
    if os.environ.get("KINDI_CHUNK_ATT", "flash") != "exact":
        # positions split over Z blocks when there are enough of them (each block then has >= 16 tiles of 64)
        Z = max(1, min(int(os.environ.get("KINDI_CHUNK_SPLIT", "4")), (pos + T) // 1024, c["XP"].size // (NHKV * T * G * 260)))
        fl = CK.k_att_flash_T
        if getattr(s, "tc_layout", False) and os.environ.get("KINDI_ATT_TC", "0") == "1":
            fl = s.sk.k_att_tc_T                 # tensor cores (tca.cu): int8 scores, fp16 P V; same interface
        fl(((T * G + 63) // 64, NHKV, Z), (256,), (s.Kq[kvi], s.Kd[kvi], s.Vq[kvi], s.Vd[kvi], c["QQ"], c["QD"],
                                                                  c["U1"], i32(2 * QL), c["OUT"], i32(QL), i32(pos), i32(T),
                                                                  i32(NHKV), i32(G), i32(NH), c["XP"]))
        if Z > 1:
            CK.k_att_merge_T((T * G, NHKV), (256,), (c["XP"], i32(Z), c["U1"], i32(2 * QL), c["OUT"], i32(QL), i32(T),
                                                      i32(NHKV), i32(G)))
        return
    ld = (pos + T + 127) // 128 * 128
    nch = (pos + T + 255) // 256
    Sb, AP = c["U2"], c["XP"]
    TS = max(1, min(T, int(os.environ.get("KINDI_CHUNK_TS", "64")), Sb.size // (NH * ld), AP.size // (nch * NH * 256)))
    for t0 in range(0, T, TS):
        ts = min(TS, T - t0)
        nmax = pos + t0 + ts
        a = (i32(pos), i32(t0))
        CK.k_att_qk_T(((nmax + 127) // 128, (ts * G + 63) // 64, NHKV), (256,),
                      (s.Kq[kvi], s.Kd[kvi], c["QQ"], c["QD"], Sb, i32(ld)) + a + (i32(ts), i32(NHKV), i32(G), i32(NH)))
        CK.k_att_softmax_T((NH, ts), (256,), (Sb, i32(ld)) + a + (i32(NH),))
        CK.k_att_pv_T(((nmax + 255) // 256, (ts * G + 63) // 64, NHKV * 4), (256,),
                      (Sb, i32(ld), s.Vq[kvi], s.Vd[kvi], AP) + a + (i32(ts), i32(NHKV), i32(G), i32(NH)))
        CK.k_att_comb_T((NH, ts), (256,), (AP, c["U1"], i32(2 * QL), c["OUT"], i32(QL)) + a + (i32(ts), i32(NH)))


def _exchange(eng, n):
    """after a segment on both cards: each card's row-parallel part (Y, n tokens) to the other card's YP. A card's
    copy waits for the other card's products; the other card waits for that copy before it may overwrite its Y."""
    cs = eng.cards
    ready, done = [], []
    for s in cs:
        with cp.cuda.Device(s.card):
            e = cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1", disable_timing=True)
            e.record()
            ready.append(e)
    for s in cs:
        o = cs[1 - s.card]
        with cp.cuda.Device(s.card):
            st = cp.cuda.get_current_stream()
            st.wait_event(ready[o.card])
            cp.cuda.runtime.memcpyPeerAsync(s.chunk["YP"].data.ptr, s.card, o.chunk["Y"].data.ptr, o.card,
                                            n * s.D * 4, st.ptr)
            e = cp.cuda.Event(block=os.environ.get("KINDI_BLOCKING_SYNC", "1") == "1", disable_timing=True)
            e.record()
            done.append(e)
    for s in cs:
        o = cs[1 - s.card]
        with cp.cuda.Device(o.card):
            cp.cuda.get_current_stream().wait_event(done[s.card])


def feed(eng, toks, pos):
    """Feed prompt tokens in chunks; returns how many were fed (the rest, fewer than KINDI_CHUNK_MIN or past the
    context, go through the batch plan)."""
    if not getattr(eng, "chunk_ok", False):
        return 0
    cs = eng.cards
    T = cs[0].chunk["T"]
    mn = int(os.environ.get("KINDI_CHUNK_MIN", "32"))
    drv = _tp._drv()
    # HIP (the tp driver is libamdhip64, see tp._drv): module launches take cuLaunchKernel's arguments, and a
    # card's ctxh is its device index (tp.py sets it with hipSetDevice)
    hip = not hasattr(drv, "cuLaunchKernel")
    L = drv.hipModuleLaunchKernel if hip else drv.cuLaunchKernel
    ctx_set = drv.hipSetDevice if hip else drv.cuCtxSetCurrent
    i = 0
    if hip:
        prev = cp.cuda.runtime.getDevice()
    else:
        prev = ctypes.c_void_p()
        drv.cuCtxGetCurrent(ctypes.byref(prev))
    # the runtime's current device too: a `with cp.cuda.Device(n)` below (_exchange, run_core) entered while the driver
    # context is the other card's restores THAT card on exit, and the caller's CuPy work (the sampler's buffers on
    # card 0) then ran on card 1: cudaErrorInvalidValue after a chunked read on two cards
    dev0 = cp.cuda.runtime.getDevice()
    try:
        while len(toks) - i >= mn and pos + i + T <= eng.ctx:
            n = min(T, len(toks) - i)
            for s in cs:
                _set(s.plan_chunk, pos + i, n, list(toks[i:i + n]) + [0] * (T - n))
            P0 = cs[0].plan_chunk
            for j in range(len(P0.segs)):
                for s in cs:
                    ctx_set(s.ctxh)
                    for e in s.plan_chunk.segs[j]:
                        if e[0] == "call":                     # a host step (the fp16 path: cuBLAS)
                            e[1](n)
                            ctx_set(s.ctxh)
                            continue
                        fn, g, b, arr, _ = e
                        if L(fn, g[0], g[1], g[2], b[0], b[1], b[2], 0, None, arr, None):
                            raise RuntimeError("kindi: a chunk kernel launch failed on card %d" % s.card)
                if j in P0.xch:
                    _exchange(eng, n)
                if j in P0.cores:
                    for s in cs:
                        with cp.cuda.Device(s.card):
                            run_core(s, s.plan_chunk.cores[j], pos + i, n)
            i += n
    finally:
        ctx_set(prev)
        cp.cuda.runtime.setDevice(dev0)
    return i
