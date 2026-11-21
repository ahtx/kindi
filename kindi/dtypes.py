# SPDX-License-Identifier: AGPL-3.0-only
"""Host-side codecs for the unquantized GGUF float formats Kindi understands.

BF16 is the top 16 bits of an IEEE float32.  The FP8 names follow the OCP
formats used by GGML's ``GGML_TYPE_F8_E4M3`` and ``GGML_TYPE_F8_E5M2``:

* E4M3FN: one sign bit, four exponent bits (bias 7), three mantissa bits,
  finite values only.  ``0x7f`` and ``0xff`` are NaN; the largest finite value
  is 448.
* E5M2: one sign bit, five exponent bits (bias 15), two mantissa bits, IEEE
  infinities and NaNs.

Decoding is exact and vectorised.  Encoding (`f32_to_fp8`) rounds to nearest,
ties to even, and saturates: E4M3FN has no infinity, so an overflow saturates to
448 instead of becoming NaN the way a decode of ``0x7f`` is.  E5M2 keeps IEEE
infinities.  The pair is the host reference for the CUDA Gemma 4 codec and for
the FP8 K/V cache conversion (roadmap #8): it is used when a Bonsai pack carries
an FP8 auxiliary tensor (the ternary language weights are a separate format).
"""
import numpy as np

BF16 = 30
FP8_E4M3 = 43
FP8_E5M2 = 44
FP8_TYPES = (FP8_E4M3, FP8_E5M2)


def bf16_to_f32(bits):
    """uint16 BF16 bit patterns -> float32 (the low 16 mantissa bits are zero)."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def fp8_to_f32(raw, kind):
    """uint8 FP8 bit patterns -> float32 for ``FP8_E4M3`` or ``FP8_E5M2``."""
    if kind not in FP8_TYPES:
        raise ValueError("unsupported FP8 type %r" % (kind,))
    b = np.asarray(raw, dtype=np.uint8).astype(np.uint32)
    sign = np.where(b & 0x80, -1.0, 1.0).astype(np.float32)
    if kind == FP8_E4M3:
        exp = (b >> 3) & 0x0f
        mant = b & 0x07
        finite = ~((exp == 15) & (mant == 7))
        value = np.where(
            exp == 0,
            mant.astype(np.float32) * np.float32(2.0 ** -9),
            (np.float32(1.0) + mant.astype(np.float32) * np.float32(0.125))
            * np.exp2(exp.astype(np.int32) - 7).astype(np.float32),
        )
        out = sign * value
        out[~finite] = np.nan
        return out
    exp = (b >> 2) & 0x1f
    mant = b & 0x03
    value = np.where(
        exp == 0,
        mant.astype(np.float32) * np.float32(2.0 ** -16),
        (np.float32(1.0) + mant.astype(np.float32) * np.float32(0.25))
        * np.exp2(exp.astype(np.int32) - 15).astype(np.float32),
    )
    inf = exp == 31
    nan = inf & (mant != 0)
    out = sign * value
    out[inf & ~nan] = np.where(sign[inf & ~nan] < 0, -np.inf, np.inf)
    out[nan] = np.nan
    return out


def f32_to_fp8(values, kind):
    """float32 values -> uint8 FP8 bit patterns for ``FP8_E4M3`` or ``FP8_E5M2``.

    Round to nearest, ties to even (an exact halfway value rounds to the even
    mantissa, matching `np.rint`).  NaN keeps its sign in the canonical NaN
    pattern (E4M3 ``0x7f``, E5M2 ``0x7d``).  E4M3FN has no infinity, so any
    finite value above 448 saturates to 448 (``0x7e``) and an infinite input
    becomes NaN; E5M2 encodes infinities as ``0x7c``.  Subnormals use one
    uniform step below the smallest normal, exactly as in `fp8_to_f32`.
    """
    if kind not in FP8_TYPES:
        raise ValueError("unsupported FP8 type %r" % (kind,))
    x = np.asarray(values, dtype=np.float32)
    sign = np.where(np.signbit(x), 0x80, 0x00).astype(np.uint8)
    a = np.abs(x.astype(np.float64))
    nan = np.isnan(a)
    inf = np.isinf(a)
    if kind == FP8_E4M3:
        bias, mant_bits, step_exp, max_finite = 7, 3, -9, 448.0
        nan_byte, max_byte, inf_byte = 0x7F, 0x7E, None
    else:
        bias, mant_bits, step_exp, max_finite = 15, 2, -16, 57344.0
        nan_byte, max_byte, inf_byte = 0x7D, 0x7B, 0x7C
    a = np.where(np.isfinite(a), a, 0.0)  # non-finite handled by masks below; avoids cast warnings
    step = float(2.0 ** step_exp)
    frac, expo = np.frexp(a)          # a = frac * 2**expo, frac in [0.5, 1)
    exp = (expo - 1).astype(np.int64)  # a in [2**exp, 2**(exp+1))
    mant_frac = frac * 2.0 - 1.0        # in [0, 1)
    normal = exp + bias >= 1
    q = np.rint(mant_frac * float(1 << mant_bits))
    carry = q == (1 << mant_bits)
    e = np.where(carry, exp + 1, exp)
    m = np.where(carry, 0.0, q)
    out = np.zeros(a.shape, dtype=np.int64)
    efield = (e + bias).astype(np.int64)
    out[normal] = (efield[normal] << mant_bits) | m[normal].astype(np.int64)
    sq = np.rint(a / step).astype(np.int64)
    sub = sq[(~normal) & (sq < (1 << mant_bits))]
    out[(~normal) & (sq < (1 << mant_bits))] = sub
    out[(~normal) & (sq >= (1 << mant_bits))] = 1 << mant_bits  # rounds up to smallest normal
    body = np.where(a >= max_finite, max_byte, out)
    body = np.where(a == 0.0, 0, body)
    body = np.where(inf, inf_byte if inf_byte is not None else nan_byte, body)
    body = np.where(nan, nan_byte, body)
    return ((body.astype(np.int64) | sign.astype(np.int64))).astype(np.uint8)
