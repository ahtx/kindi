# SPDX-License-Identifier: AGPL-3.0-only
"""Reference tests for the host BF16 / FP8 codecs."""
import numpy as np

from kindi.dtypes import BF16, FP8_E4M3, FP8_E5M2, bf16_to_f32, f32_to_fp8, fp8_to_f32


def _scalar_fp8(byte, kind):
    """Independent scalar reference for every possible 8-bit pattern."""
    sign = -1.0 if byte & 0x80 else 1.0
    if kind == FP8_E4M3:
        exp, mant = (byte >> 3) & 0x0f, byte & 0x07
        if exp == 15 and mant == 7:
            return float("nan")
        value = mant * 2.0 ** -9 if exp == 0 else (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
    else:
        exp, mant = (byte >> 2) & 0x1f, byte & 0x03
        if exp == 31:
            return float("nan") if mant else sign * float("inf")
        value = mant * 2.0 ** -16 if exp == 0 else (1.0 + mant / 4.0) * 2.0 ** (exp - 15)
    return sign * value


def test_every_e4m3_and_e5m2_pattern_decodes_like_the_scalar_reference():
    patterns = np.arange(256, dtype=np.uint8)
    for kind in (FP8_E4M3, FP8_E5M2):
        got = fp8_to_f32(patterns, kind)
        want = np.array([_scalar_fp8(int(b), kind) for b in patterns], dtype=np.float32)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.array_equal(np.isinf(got), np.isinf(want))
        assert np.array_equal(np.signbit(got[np.isfinite(got)]), np.signbit(want[np.isfinite(want)]))
        assert np.allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=0, atol=0)


def test_named_fp8_values():
    e4 = fp8_to_f32(np.array([0x00, 0x01, 0x38, 0x3C, 0x7E, 0x7F, 0xB8], dtype=np.uint8), FP8_E4M3)
    assert e4[:5].tolist() == [0.0, 2.0 ** -9, 1.0, 1.5, 448.0]
    assert np.isnan(e4[5]) and e4[6] == -1.0
    e5 = fp8_to_f32(np.array([0x00, 0x01, 0x3C, 0x40, 0x7C, 0x7D], dtype=np.uint8), FP8_E5M2)
    assert e5[:4].tolist() == [0.0, 2.0 ** -16, 1.0, 2.0]
    assert np.isinf(e5[4]) and np.isnan(e5[5])


def test_fp8_encode_of_every_finite_pattern_round_trips_exactly():
    for kind in (FP8_E4M3, FP8_E5M2):
        patterns = np.arange(256, dtype=np.uint8)
        values = fp8_to_f32(patterns, kind)
        finite = np.isfinite(values)
        assert np.array_equal(f32_to_fp8(values[finite], kind), patterns[finite])


def test_fp8_encode_named_values_and_saturation():
    e4 = f32_to_fp8(np.array([0.0, 1.0, 1.5, 448.0, 1000.0, -1.0, np.nan, np.inf], dtype=np.float32), FP8_E4M3)
    assert e4.tolist() == [0x00, 0x38, 0x3C, 0x7E, 0x7E, 0xB8, 0x7F, 0x7F]
    e5 = f32_to_fp8(np.array([0.0, 1.0, 2.0, 57344.0, np.inf, -np.inf, np.nan], dtype=np.float32), FP8_E5M2)
    assert e5.tolist() == [0x00, 0x3C, 0x40, 0x7B, 0x7C, 0xFC, 0x7D]


def test_fp8_encode_ties_round_to_even_and_subnormals_round_up_to_normal():
    # 1.0625 is halfway between 1.0 (mant 0) and 1.125 (mant 1): ties to even -> 1.0
    assert f32_to_fp8(np.array([1.0625], dtype=np.float32), FP8_E4M3).tolist() == [0x38]
    # halfway between subnormal steps 1x and 2x 2**-9 rounds to the even step 2x
    assert f32_to_fp8(np.array([1.5 * 2.0 ** -9], dtype=np.float32), FP8_E4M3).tolist() == [0x02]
    # halfway between 7 top subnormals and the smallest normal (8x) rounds to the normal
    assert f32_to_fp8(np.array([7.5 * 2.0 ** -9], dtype=np.float32), FP8_E4M3).tolist() == [0x08]


def test_fp8_encode_refuses_unknown_kinds():
    import pytest
    with pytest.raises(ValueError, match="unsupported FP8 type 99"):
        f32_to_fp8(np.array([1.0], dtype=np.float32), 99)


def test_bf16_is_the_top_half_of_float32():
    bits = np.array([0x0000, 0x3F80, 0xC000, 0x7F80], dtype=np.uint16)
    got = bf16_to_f32(bits)
    assert got.tolist()[:3] == [0.0, 1.0, -2.0]
    assert np.isinf(got[3])
    assert BF16 == 30
