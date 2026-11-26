# SPDX-License-Identifier: AGPL-3.0-only
"""The CPU-side half of `kindi intel-norm`: the reference and the row decoder, with no device.

The probe's OpenCL kernels need an Arc; these checks pin the two numpy pieces the kernels are judged
against, so a wrong reference cannot silently pass a device run:

* `ref_fold` is an orthonormal Hadamard transform per block - energy is preserved, and with a unit
  sign table it is its own inverse (the 1/sqrt(hb) scaling twice is the identity);
* `ref_rmsfold` normalises the vector to unit mean square before the weight and the fold;
* `_row` decodes the pack's own byte layout: four balanced trits per byte in phases of 128 columns,
  one fp16 scale per (phase, row).
"""
import numpy as np

from kindi import intel_layer


def test_ref_fold_is_orthonormal_and_self_inverse_with_unit_signs():
    hb, blocks = 16, 3
    rng = np.random.default_rng(11)
    x = rng.standard_normal(hb * blocks).astype(np.float32)
    ones = np.ones_like(x)
    folded = intel_layer.ref_fold(x, ones, hb)
    assert np.isclose(np.sum(folded.astype(np.float64) ** 2),
                      np.sum(x.astype(np.float64) ** 2), rtol=1e-5)
    assert np.allclose(intel_layer.ref_fold(folded, ones, hb), x, atol=1e-5)


def test_ref_fold_applies_the_sign_before_the_blocks():
    hb = 8
    rng = np.random.default_rng(12)
    x = rng.standard_normal(hb).astype(np.float32)
    sign = np.where(np.arange(hb) % 2, -1.0, 1.0).astype(np.float32)
    # fold(v, sg) is fold(sg*v, 1): the sign is applied before the transform, not to its output.
    assert np.array_equal(intel_layer.ref_fold(x, sign, hb),
                          intel_layer.ref_fold(x * sign, np.ones(hb, np.float32), hb))
    delta = np.zeros(hb, dtype=np.float32)
    delta[0] = 1.0
    # FWHT's first row is all +1, so a delta at index 0 folds to a constant.
    assert np.allclose(intel_layer.ref_fold(delta, np.ones(hb, np.float32), hb),
                       1.0 / np.sqrt(hb))


def test_ref_rmsfold_normalises_before_the_weight():
    n, hb = 32, 8
    x = np.full(n, 2.0, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    ones = np.ones(n, dtype=np.float32)
    sigma, xn, _ = intel_layer.ref_rmsfold(x, zeros, ones, ones, hb, 1)
    assert np.isclose(float(sigma), 0.5, atol=1e-6)      # mean square 4 -> 1/sqrt(4)
    assert np.allclose(xn, 1.0, atol=1e-6)


def test_ref_rmsfold_folds_a_block_of_zeros_to_zero():
    n, hb = 32, 8
    x = np.full(n, 2.0, dtype=np.float32)
    x[hb:] = 0.0
    _, xn, xp = intel_layer.ref_rmsfold(x, np.zeros(n, np.float32), np.ones(n, np.float32),
                                        np.ones(n, np.float32), hb, 1)
    assert np.allclose(xn[hb:], 0.0)
    assert np.allclose(xp[hb:], 0.0)                     # a zero block stays zero through FWHT


def test_ref_rmsfold_residual_add_happens_before_the_sum():
    n, hb = 8, 8
    h = np.ones(n, dtype=np.float32)
    res = np.full(n, 3.0, dtype=np.float32)
    ones = np.ones(n, np.float32)
    _, added, _ = intel_layer.ref_rmsfold(h, res, ones, ones, hb, 1)
    _, plain, _ = intel_layer.ref_rmsfold(h, res, ones, ones, hb, 0)
    assert np.allclose(added, 1.0, atol=1e-6)            # (1 + 3) is already unit RMS
    assert np.allclose(plain, 1.0, atol=1e-6)            # add=0 ignores the residual
    # ...but only because both vectors are constant; a non-constant residual must move the scale.
    res2 = np.arange(n, dtype=np.float32)
    s_add, _, _ = intel_layer.ref_rmsfold(h, res2, ones, ones, hb, 1)
    s_plain, _, _ = intel_layer.ref_rmsfold(h, res2, ones, ones, hb, 0)
    assert not np.isclose(float(s_add), float(s_plain))


def test_row_decode_reads_the_pack_layout():
    cols, rows, row = 128, 4, 2
    scales = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float16)   # row 2's scale is 2.0
    codes = np.zeros((32, rows), dtype=np.uint8)
    codes[0, row] = 0        # base-3 digits 0,0,0,0 -> trits (-1,-1,-1,-1)
    codes[1, row] = 80       # 2 + 2*3 + 2*9 + 2*27 -> digits 2 -> trits (+1,+1,+1,+1)
    entry = {"ne0": cols, "rows": rows, "bc_off": 0, "dpm_off": 0}
    out = intel_layer._row(codes.reshape(-1), scales, entry, row)
    assert out.shape == (cols,)
    assert np.array_equal(out[:4], np.array([-2.0, -2.0, -2.0, -2.0], dtype=np.float32))
    assert np.array_equal(out[4:8], np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
    assert np.array_equal(out[8:], np.full(cols - 8, -2.0, dtype=np.float32))  # byte 0 -> trit -1
