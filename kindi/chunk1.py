"""Prompt reading in chunks on the single-card engine (engine.py: the 2B, 4B and 9B; 2-bit weights, fp16 K/V).

The same pieces as kindi/chunk.py for the plan engine: one GEMM per weight over up to KINDI_CHUNK tokens (mm_core with
the 2-bit k_g4-layout unpack stage, fp32 - no dp4a, so it runs on sm_30 and sm_50), the decode kernels wrapped for
T tokens per launch (same arithmetic), the GDN conv and delta steps in token order inside one launch each, and a
one-pass attention over the fp16 cache. Engine.feed_batch(toks, pos) feeds whole chunks and the rest one by one.
KINDI_CHUNK=0: off (the engine as before)."""
import os

import numpy as np
import cupy as cp

from . import chunk as _c

i32 = np.int32
WRAP1 = ("k_rmsfold", "k_smfold", "k_fwht_hi", "k_rmsw", "k_rmsw_s", "k_rope", "k_l2w", "k_gdn_gates", "k_fold_fwht")
OWN1 = ("k_mm_g4", "k_mm_g4s", "k_mm_g4k", "k_gdn_conv_T", "k_gdn_delta_T", "k_att_flash16_T", "k_att_flash16k_T", "k_att_merge_T", "k_sumsplits")


class Chunk1Kernels(object):
    def __init__(self):
        base = open(os.path.join(_c.HERE, "kindi.cu"), encoding="utf-8").read()
        wr, self.nptr = _c._wrappers(base, WRAP1)
        own = open(os.path.join(_c.HERE, "chunk.cu"), encoding="utf-8").read()
        own1 = open(os.path.join(_c.HERE, "chunk1.cu"), encoding="utf-8").read()
        mod = cp.RawModule(code=base + "\n" + wr + "\n" + own + "\n" + own1)
        for k in WRAP1:
            setattr(self, k + "_T", mod.get_function(k + "_T"))
        for k in OWN1:
            setattr(self, k, mod.get_function(k))


def reserve(eng):
    """the chunk buffers, before the engine fits its context into the free memory"""
    eng.chunk_ok = False
    major = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["major"]
    # 256-token chunks from sm_50 (K1200: 1.2-1.7x); 128 on sm_30 (K420 1 GB: 1.2-1.25x with the Kepler kernels, and
    # the smaller buffers keep the 24,576-token context)
    v = os.environ.get("KINDI_CHUNK", "256" if major >= 5 else "128")
    if v in ("0", "", "1") or eng.HD != 256:
        return
    T = int(v)
    z = lambda n, t=cp.float32: cp.zeros(int(n), dtype=t)  # noqa: E731
    D, NFF, HK, HD, NH, NHKV, NV, NK, CC = eng.D, eng.NFF, eng.HK, eng.HD, eng.NH, eng.NHKV, eng.NV, eng.NK, eng.CONV_CH
    work = max(D, NFF, NV * HK, NH * HD)
    c = eng.chunk = {"T": T, "work": work}
    c["HA"], c["HB"], c["XN"], c["Y"] = z(T * D), z(T * D), z(T * D), z(T * D)
    c["XP"] = z(T * work)
    big = max(NFF, CC + 2 * NK * HK, 2 * NH * HD, NV * HK + CC, 3 * NHKV * HD + 2 * NH * HD)
    c["U1"], c["U2"] = z(T * big), z(T * big)
    c["OUT"], c["GOUT"] = z(T * max(NV * HK, NH * HD)), z(T * NV * HK)
    c["SPL"] = z(8 * T * max(NHKV * HD, D))
    c["GB"], c["GG"] = z(T * NV), z(T * NV)
    c["big"] = big
    prop = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    tv = os.environ.get("KINDI_CHUNK_TILE", "")
    c["small"] = tv == "64" or (tv != "128" and prop["major"] < 5)
    eng.ck = Chunk1Kernels()
    if not cp.cuda.runtime.is_hip:
        # 8-byte shared memory banks (Kepler; the float4 tile loads) and the larger shared memory split (CUDA only;
        # on ROCm the bank layout is fixed)
        import ctypes
        from . import gpudrv
        drv = gpudrv.driver()
        fp = ctypes.c_void_p(eng.ck.k_mm_g4k.kernel.ptr)
        drv.cuFuncSetSharedMemConfig(fp, 2)
        drv.cuFuncSetCacheConfig(fp, 1)
    eng.chunk_ok = True
    eng.batch_max = T


def _gemm(eng, name, X, ldx, Y, ldy, T):
    c, CK = eng.chunk, eng.ck
    W, S, R, C = eng.w[name]
    R, C = int(R), int(C)
    # 128 rows x 64 tokens, 16-column steps (13 KB: several blocks per SM): K420 106 vs 57 GFLOPS, K1200 620 vs 465;
    # KINDI_CHUNK_TILE=128: the 128 x 128 tile
    kt = os.environ.get("KINDI_CHUNK_TILE", "") != "128"
    f = CK.k_mm_g4k if kt else CK.k_mm_g4
    gx, gy = (R // 128, (T + 63) // 64) if kt else (R // 128, (T + 127) // 128)
    nb = C // 128
    slots = 2 * eng.nsm
    sp = 1 if gx * gy >= slots else max(1, min(8, nb, -(-2 * slots // (gx * gy)), c["SPL"].size // (T * R)))
    if sp == 1:
        f((gx, gy, 1), (256,), (W, S, i32(R), i32(C), X, i32(ldx), Y, i32(ldy), i32(T)))
    else:
        f((gx, gy, sp), (256,), (W, S, i32(R), i32(C), X, i32(ldx), c["SPL"], i32(R), i32(T)))
        CK.k_sumsplits(((T * R + 255) // 256,), (256,), (c["SPL"], Y, i32(ldy), i32(R), i32(T), i32(sp)))


def _T(kern, grid, block, args, strides, T):
    g = tuple(grid) if isinstance(grid, tuple) else (grid,)
    kern((g[0], T), block, tuple(args) + (i32(T),) + tuple(i32(x) for x in strides))


def _fold(eng, kern, width, args, strides, xo, T):
    """a fold kernel (Hadamard blocks of HB; above 512: 512-point chunks, then k_fwht_hi) for T rows"""
    (gx,), hb = eng.fold_args(width)
    _T(kern, gx, (512,), tuple(args) + (hb,), strides, T)
    if eng.HB > 512:
        w = eng.chunk["work"]
        _T(eng.ck.k_fwht_hi_T, (width // 512 // (eng.HB // 512) * 512 + 255) // 256, (256,),
           (xo, i32(eng.HB), i32(width)), (w,), T)


def feed_chunk(eng, toks, pos):
    c, CK = eng.chunk, eng.ck
    T = len(toks)
    D, HB, HK, HD, NH, NHKV, NV, NK, CC, NFF = (eng.D, eng.HB, eng.HK, eng.HD, eng.NH, eng.NHKV, eng.NV, eng.NK,
                                                 eng.CONV_CH, eng.NFF)
    G, L, QL, work = NH // NHKV, NHKV * HD, NH * HD, c["work"]
    XP, XN, Y, U1, U2, OUT = c["XP"], c["XN"], c["Y"], c["U1"], c["U2"], c["OUT"]
    st = {"h": c["HA"], "h2": c["HB"], "pending": 0}
    a = eng.w["token_embd.weight"]
    for k, tok in enumerate(toks):
        row = int(eng.hs_permpos[tok]) if eng.hsl is not None and eng.head_name == "token_embd.weight" else int(tok)
        eng.k.k_deq_emb3((1,), (1024,), (a[0], a[1], eng.sD, st["h"][k * D:(k + 1) * D], i32(row), i32(D), i32(HB)))

    def mixin(wname):
        (gx,), hb = eng.fold_args(D)
        _T(CK.k_rmsfold_T, gx, (512,), (st["h"], Y, st["h2"], eng.f32(wname), eng.sD, XN, XP, i32(D),
                                        i32(1 if st["pending"] else 0), hb), (D, D, D, 0, 0, D, work), T)
        if HB > 512:
            _T(CK.k_fwht_hi_T, (D // 512 // (HB // 512) * 512 + 255) // 256, (256,), (XP, i32(HB), i32(D)), (work,), T)
        if st["pending"]:
            st["h"], st["h2"] = st["h2"], st["h"]
        st["pending"] = 0

    kvi = 0
    for il in range(eng.NL):
        p = "blk.%d." % il
        last = il == eng.NL - 1
        if (il + 1) % eng.dims["full_attention_interval"] == 0:
            mixin(p + "attn_norm.weight")
            QN, KRO, VB, KN, FO, QR = U1, U2, U2[T * L:], U2[2 * T * L:], U2[3 * T * L:], U2[3 * T * L + T * QL:]
            if not last:
                _gemm(eng, p + "attn_q.weight", XP, work, QN, 2 * QL, T)
            _gemm(eng, p + "attn_k.weight", XP, work, KRO, L, T)
            _gemm(eng, p + "attn_v.weight", XP, work, VB, L, T)
            _T(CK.k_rmsw_T, (NHKV + 7) // 8, (256,), (KRO, eng.f32(p + "attn_k_norm.weight"), KN, i32(HD), i32(NHKV)),
               (L, 0, L), T)
            rd = eng.cos_t.shape[1]
            cs, sn = eng.cos_t[pos:], eng.sin_t[pos:]
            _T(CK.k_rope_T, NHKV, (256,), (KN, cs, sn, KRO, i32(HD)), (L, rd, rd, L), T)
            eng.Kc[kvi][pos:pos + T] = KRO[:T * L].reshape(T, NHKV, HD)
            eng.Vc[kvi][pos:pos + T] = VB[:T * L].reshape(T, NHKV, HD)
            if not last:
                _T(CK.k_rmsw_s_T, (NH + 7) // 8, (256,), (QN, i32(2 * HD), eng.f32(p + "attn_q_norm.weight"), FO, i32(HD),
                                                         i32(NH)), (2 * QL, 0, QL), T)
                _T(CK.k_rope_T, NH, (256,), (FO, cs, sn, QR, i32(HD)), (QL, rd, rd, QL), T)
                Z = max(1, min(int(os.environ.get("KINDI_CHUNK_SPLIT", "4")), (pos + T) // 1024,
                               XP.size // (NHKV * T * G * 260)))
                fl, rows = (CK.k_att_flash16k_T, 32) if c["small"] else (CK.k_att_flash16_T, 64)
                fl(((T * G + rows - 1) // rows, NHKV, Z), (256,), (eng.Kc[kvi], eng.Vc[kvi], QR, i32(QL), QN,
                                                                            i32(2 * QL), OUT, i32(QL), i32(pos), i32(T),
                                                                            i32(NHKV), i32(G), i32(NH), XP))
                if Z > 1:
                    CK.k_att_merge_T((T * G, NHKV), (256,), (XP, i32(Z), QN, i32(2 * QL), OUT, i32(QL), i32(T), i32(NHKV),
                                                             i32(G)))
                _fold(eng, CK.k_fold_fwht_T, QL, (OUT, eng.sA, XP), (QL, 0, work), XP, T)
                _gemm(eng, p + "attn_output.weight", XP, work, Y, D, T)
                st["pending"] = 1
            kvi += 1
        else:
            mixin(p + "attn_norm.weight")
            QKV, QK, ZV, CS = U1, U1[T * CC:], U2, U2[T * NV * HK:]
            _gemm(eng, p + "attn_qkv.weight", XP, work, QKV, CC, T)
            _gemm(eng, p + "attn_gate.weight", XP, work, ZV, NV * HK, T)
            _T(CK.k_gdn_gates_T, NV, (256,), (XN, eng.f32(p + "ssm_beta.weight"), eng.f32(p + "ssm_alpha.weight"),
                                              eng.f32(p + "ssm_dt.bias"), eng.f32(p + "ssm_a"), c["GB"], c["GG"], i32(D)),
               (D, 0, 0, 0, 0, NV, NV), T)
            CK.k_gdn_conv_T(((CC + 255) // 256,), (256,), (QKV, i32(CC), eng.conv[il], eng.f32(p + "ssm_conv1d.weight"),
                                                           CS, i32(CC), i32(CC), i32(T)))
            _T(CK.k_l2w_T, (2 * NK + 7) // 8, (256,), (CS, QK, i32(HK), i32(2 * NK)), (CC, 2 * NK * HK), T)
            CK.k_gdn_delta_T((NV,), (HK,), (eng.M[il], QK, QK[NK * HK:], i32(2 * NK * HK), CS[2 * NK * HK:], i32(CC),
                                            c["GG"], c["GB"], i32(NV), c["GOUT"], i32(NV * HK),
                                            np.float32(1.0 / np.sqrt(HK)), i32(NK), i32(T)))
            if last:
                continue
            _T(CK.k_rmsw_T, (NV + 7) // 8, (256,), (c["GOUT"], eng.f32(p + "ssm_norm.weight"), OUT, i32(HK), i32(NV)),
               (NV * HK, 0, NV * HK), T)
            _fold(eng, CK.k_smfold_T, NV * HK, (ZV, OUT, eng.sV, XP, i32(HK), i32(NK), i32(NV // NK)),
                  (NV * HK, NV * HK, 0, work), XP, T)
            _gemm(eng, p + "ssm_out.weight", XP, work, Y, D, T)
            st["pending"] = 1
        if last:
            continue
        mixin(p + "post_attention_norm.weight")
        _gemm(eng, p + "ffn_gate.weight", XP, work, U1, NFF, T)
        _gemm(eng, p + "ffn_up.weight", XP, work, U2, NFF, T)
        _fold(eng, CK.k_smfold_T, NFF, (U1, U2, eng.sF, XP, i32(0), i32(0), i32(0)), (NFF, NFF, 0, work), XP, T)
        _gemm(eng, p + "ffn_down.weight", XP, work, Y, D, T)
        st["pending"] = 1


def feed_batch(eng, toks, pos):
    """Feed prompt tokens: whole chunks (at least KINDI_CHUNK_MIN tokens) through feed_chunk, the rest one by one."""
    i = 0
    if getattr(eng, "chunk_ok", False):
        T = eng.chunk["T"]
        mn = int(os.environ.get("KINDI_CHUNK_MIN", "32"))
        while len(toks) - i >= mn:
            n = min(T, len(toks) - i)
            if pos + i + n > eng.ctx:
                raise OverflowError("position %d is past the context (%d)" % (pos + i + n - 1, eng.ctx))
            feed_chunk(eng, list(toks[i:i + n]), pos + i)
            i += n
    while i < len(toks):
        eng.feed(toks[i], pos + i)
        i += 1
