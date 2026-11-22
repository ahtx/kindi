"""Weight decoders for the qwen4exp (Qwen3.8-Flash-Next) GGUF files: every type those files use, block bytes -> fp32.

These are the CPU reference decoders (NumPy, vectorized over blocks). The GPU and fast CPU kernels must give the
same values. The I-quant layouts and codebooks are the ones of the Kindi Nanbeige engine (nbq.cu, nbq_grids.h); here
the fp16 block scale stays at the front of each block, as in the file.

decode(ty, raw, n) -> float32 array of n values (n a multiple of the block size)."""
import os
import re

import numpy as np

F32, F16, Q8_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, BF16, Q2_0 = (
    0, 1, 8, 12, 13, 14, 18, 20, 21, 22, 23, 30, 42)

# type -> (values per block, bytes per block)
BLOCK = {
    F32: (1, 4), F16: (1, 2), BF16: (1, 2),
    Q8_0: (32, 34), IQ4_NL: (32, 18), Q2_0: (64, 18),
    Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210),
    IQ3_XXS: (256, 98), IQ3_S: (256, 110), IQ2_S: (256, 82), IQ4_XS: (256, 136),
}
NAMES = {F32: "F32", F16: "F16", Q8_0: "Q8_0", Q4_K: "Q4_K", Q5_K: "Q5_K", Q6_K: "Q6_K", IQ3_XXS: "IQ3_XXS",
         IQ4_NL: "IQ4_NL", IQ3_S: "IQ3_S", IQ2_S: "IQ2_S", IQ4_XS: "IQ4_XS", BF16: "BF16", Q2_0: "Q2_0"}

# the 16 levels of IQ4_NL / IQ4_XS (a non-uniform 4-bit code)
IQ4_LEVELS = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def nbytes(ty, n):
    per, size = BLOCK[ty]
    if n % per:
        raise ValueError("%d values is not a whole number of %s blocks" % (n, NAMES.get(ty, ty)))
    return n // per * size


_GRIDS = None


def _grids():
    """the IQ3_XXS (256 x 4), IQ3_S (512 x 4) and IQ2_S (1024 x 8) codebooks of nbq_grids.h as uint8 arrays"""
    global _GRIDS
    if _GRIDS is None:
        text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbq_grids.h"), encoding="utf-8").read()

        def table(name, n, width):
            body = re.search(r"%s\[%d\]\s*=\s*\{([^}]*)\}" % (name, n), text).group(1)
            vals = [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\d+", body)]
            if len(vals) != n:
                raise ValueError("%s: %d entries, expected %d" % (name, len(vals), n))
            dt = np.uint32 if width == 4 else np.uint64
            return np.array(vals, dtype=dt).view(np.uint8).reshape(n, width)
        _GRIDS = (table("nbq_iq3xxs_grid", 256, 4), table("nbq_iq3s_grid", 512, 4), table("nbq_iq2s_grid", 1024, 8))
    return _GRIDS


def _f16(b, off):
    return b[:, off:off + 2].copy().view(np.float16).astype(np.float32)[:, 0]


def _signs(bits8):
    """(..., k) sign bytes -> (..., k, 8) of +1 / -1 (bit i set = value i negated)"""
    s = (bits8[..., None] >> np.arange(8, dtype=np.uint8)) & 1
    return 1.0 - 2.0 * s.astype(np.float32)


def _scale_min_k(sc):
    """the 8 six-bit scales and mins of a Q4_K / Q5_K block from its 12 scale bytes: (nb, 12) -> 2 x (nb, 8)"""
    sc = sc.astype(np.int32)
    d = np.empty((sc.shape[0], 8), np.int32)
    m = np.empty_like(d)
    for j in range(8):
        k, u = j & 3, j >> 2
        x, y, z = sc[:, k], sc[:, k + 4], sc[:, k + 8]
        sh = 2 * u
        d[:, j] = ((z if u else x) & 15) | ((x >> sh) & 48)
        m[:, j] = (((z >> 4) if u else y) & 15) | ((y >> sh) & 48)
    return d, m


def _q4k(b):
    d, dm = _f16(b, 0), _f16(b, 2)
    sc, mn = _scale_min_k(b[:, 4:16])
    qs = b[:, 16:144].reshape(-1, 4, 32)                       # 4 chunks of 32 bytes, each 2 runs (low, high nibble)
    q = np.stack([qs & 15, qs >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
    return (d[:, None, None] * sc[:, :, None] * q - dm[:, None, None] * mn[:, :, None]).reshape(-1, 256)


def _q5k(b):
    d, dm = _f16(b, 0), _f16(b, 2)
    sc, mn = _scale_min_k(b[:, 4:16])
    qh = b[:, 16:48]                                           # bit j of byte k = high bit of value k of run j
    qs = b[:, 48:176].reshape(-1, 4, 32)
    lo = np.stack([qs & 15, qs >> 4], axis=2).reshape(-1, 8, 32).astype(np.int32)
    hi = ((qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1).astype(np.int32)
    q = (lo | (hi << 4)).astype(np.float32)
    return (d[:, None, None] * sc[:, :, None] * q - dm[:, None, None] * mn[:, :, None]).reshape(-1, 256)


def _q6k(b):
    ql, qh = b[:, 0:128].astype(np.int32), b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(b, 208)
    out = np.empty((b.shape[0], 256), np.float32)
    for n in range(2):                                         # two halves of 128 values
        L, H, S = ql[:, 64 * n:64 * n + 64], qh[:, 32 * n:32 * n + 32], sc[:, 8 * n:8 * n + 8]
        parts = [(L[:, 0:32] & 15) | (((H >> 0) & 3) << 4), (L[:, 32:64] & 15) | (((H >> 2) & 3) << 4),
                 (L[:, 0:32] >> 4) | (((H >> 4) & 3) << 4), (L[:, 32:64] >> 4) | (((H >> 6) & 3) << 4)]
        for p, q in enumerate(parts):
            s = np.repeat(S[:, 2 * p:2 * p + 2], 16, axis=1)        # scale per 16 values
            out[:, 128 * n + 32 * p:128 * n + 32 * p + 32] = d[:, None] * s * (q - 32).astype(np.float32)
    return out


def _q8_0(b):
    return _f16(b, 0)[:, None] * b[:, 2:34].view(np.int8).astype(np.float32)


def _iq4nl(b):
    qs = b[:, 2:18]
    q = np.concatenate([qs & 15, qs >> 4], axis=1)
    return _f16(b, 0)[:, None] * IQ4_LEVELS[q]


def _iq4xs(b):
    d = _f16(b, 0)
    sh = b[:, 2:4].copy().view(np.uint16)[:, 0].astype(np.int32)
    sl = b[:, 4:8].astype(np.int32)
    qs = b[:, 8:136].reshape(-1, 8, 16)
    out = np.empty((b.shape[0], 8, 32), np.float32)
    for ib in range(8):
        ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32).astype(np.float32)
        q = np.concatenate([qs[:, ib] & 15, qs[:, ib] >> 4], axis=1)
        out[:, ib] = dl[:, None] * IQ4_LEVELS[q]
    return out.reshape(-1, 256)


def _q2_0(b):
    # 64 two-bit codes in 16 bytes: code i is bits 2 (i % 4) of byte i // 4; value = d * (code - 1)
    qs = b[:, 2:18]
    q = np.stack([(qs >> (2 * k)) & 3 for k in range(4)], axis=2).reshape(-1, 64).astype(np.float32)
    return _f16(b, 0)[:, None] * (q - 1.0)


def _iq3xxs(b):
    g3, _, _ = _grids()
    d = _f16(b, 0)
    qs = b[:, 2:66].reshape(-1, 8, 8)                          # per run: 8 grid indices of 4 values
    aux = b[:, 66:98].copy().view(np.uint32).reshape(-1, 8).astype(np.uint64)
    s7 = np.stack([(aux >> (7 * l)) & 127 for l in range(4)], axis=2).astype(np.uint8)     # (nb, 8, 4)
    par = np.unpackbits(s7[..., None], axis=-1).sum(-1) & 1
    s8 = s7 | (par.astype(np.uint8) << 7)
    sign = _signs(s8).reshape(-1, 8, 32)
    vals = g3[qs].reshape(-1, 8, 32).astype(np.float32)
    db = d[:, None] * (0.5 + (aux >> 28).astype(np.float32)) * 0.5
    return (db[:, :, None] * vals * sign).reshape(-1, 256)


def _iq3s(b):
    _, g3s, _ = _grids()
    d = _f16(b, 0)
    qs = b[:, 2:66].reshape(-1, 8, 8).astype(np.int32)
    qh = b[:, 66:74].astype(np.int32)                          # bit j of qh[run] = bit 8 of index j of the run
    hi = (qh[:, :, None] >> np.arange(8)[None, None, :]) & 1
    idx = qs | (hi << 8)
    signs = b[:, 74:106].reshape(-1, 8, 4)
    sign = _signs(signs).reshape(-1, 8, 32)
    vals = g3s[idx].reshape(-1, 8, 32).astype(np.float32)
    sc = b[:, 106:110].astype(np.int32)
    s = np.stack([(sc[:, r >> 1] >> (4 * (r & 1))) & 15 for r in range(8)], axis=1)
    db = d[:, None] * (1 + 2 * s).astype(np.float32)
    return (db[:, :, None] * vals * sign).reshape(-1, 256)


def _iq2s(b):
    _, _, g2 = _grids()
    d = _f16(b, 0)
    qs = b[:, 2:34].reshape(-1, 8, 4).astype(np.int32)        # per run: 4 grid indices of 8 values
    signs = b[:, 34:66].reshape(-1, 8, 4)
    qh = b[:, 66:74].astype(np.int32)
    hi = np.stack([(qh >> (2 * l)) & 3 for l in range(4)], axis=2)
    idx = qs | (hi << 8)
    vals = g2[idx].reshape(-1, 8, 32).astype(np.float32)
    sign = _signs(signs).reshape(-1, 8, 32)
    sc = b[:, 74:82].astype(np.int32)
    s = np.repeat(np.stack([sc & 15, sc >> 4], axis=2), 16, axis=2).astype(np.float32)   # (nb, 8, 32)
    return (d[:, None, None] * (0.5 + s) * 0.25 * vals * sign).reshape(-1, 256)


_DEC = {Q4_K: _q4k, Q5_K: _q5k, Q6_K: _q6k, Q8_0: _q8_0, IQ4_NL: _iq4nl, IQ4_XS: _iq4xs, Q2_0: _q2_0,
        IQ3_XXS: _iq3xxs, IQ3_S: _iq3s, IQ2_S: _iq2s}


def decode(ty, raw, n):
    """n values of type ty from the bytes raw (a buffer or uint8 array) -> float32 array of n"""
    a = np.frombuffer(raw, dtype=np.uint8, count=nbytes(ty, n))
    if ty == F32:
        return a.view(np.float32).copy()
    if ty == F16:
        return a.view(np.float16).astype(np.float32)
    if ty == BF16:
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    per, size = BLOCK[ty]
    return _DEC[ty](a.reshape(-1, size)).reshape(-1).astype(np.float32, copy=False)
