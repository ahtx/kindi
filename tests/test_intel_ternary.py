# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only tests for the packed ternary GEMV probe (REQ-0020 M2 prototype)."""
import numpy as np

from kindi import intel_ternary
from kindi.convert import phase_bytes


def _digits(bcodes, rows, cols):
    codes = np.asarray(bcodes, dtype=np.int32).reshape(cols // 128, 32, rows)
    return np.stack([codes % 3, (codes // 3) % 3, (codes // 9) % 3, (codes // 27) % 3],
                    axis=2).transpose(3, 0, 1, 2).reshape(rows, cols)


def test_the_pack_is_the_converter_layout():
    trits, bcodes, dpm, x = intel_ternary.matrix(16, 512, seed=7)
    assert np.array_equal(bcodes, np.ascontiguousarray(phase_bytes(trits).T))
    assert bcodes.shape == (512 // 4, 16)          # [slot][row], 4 trits per byte
    assert dpm.shape == (512 // 128, 16)           # [phase][row], one scale per 128 weights
    assert dpm.dtype == np.float16
    assert x.shape == (512,)
    assert x.shape == (512,)
    # non-integer on purpose: an integer test input cannot tell a float GEMV from an int one
    assert np.any(x != np.round(x)) and np.abs(x).max() <= 1.9


def test_the_packed_bytes_decode_back_to_the_trits():
    trits, bcodes, dpm, x = intel_ternary.matrix(16, 512, seed=7)
    assert np.array_equal(_digits(bcodes, 16, 512), trits)


def test_the_reference_matches_an_independent_float64_dot():
    trits, bcodes, dpm, x = intel_ternary.matrix(16, 512, seed=7)
    got = intel_ternary.reference(bcodes, dpm, x, 16, 512)
    scales = dpm.astype(np.float64).T.repeat(128, axis=1)
    want = ((trits.astype(np.float64) - 1.0) * x[None, :] * scales).sum(axis=1)
    assert np.array_equal(got.astype(np.float64), want)


def test_the_parts_cover_every_phase_exactly_once():
    """Phase ph belongs to part ph % parts: the sum of the partials is the whole GEMV."""
    for rows, cols in ((16, 512), (5, 1024), (3, 4096)):
        _trits, bcodes, dpm, x = intel_ternary.matrix(rows, cols, seed=7)
        want = intel_ternary.reference(bcodes, dpm, x, rows, cols)
        for parts in (1, 2, 8, 32, 64):
            got = intel_ternary.partials(bcodes, dpm, x, rows, cols, parts)
            assert got.shape == (parts, rows)
            assert np.array_equal(got.sum(axis=0, dtype=np.float32), want), (rows, cols, parts)


def test_a_part_only_touches_its_own_phases():
    """A phase the part does not own contributes nothing to it."""
    _trits, bcodes, dpm, x = intel_ternary.matrix(8, 512, seed=7)
    full = intel_ternary.partials(bcodes, dpm, x, 8, 512, 2)
    zeroed = np.array(x, dtype=np.float32)
    for phase in range(1, 512 // 128, 2):      # blank every phase part 1 owns
        zeroed[phase * 128:(phase + 1) * 128] = 0.0
    part0 = intel_ternary.partials(bcodes, dpm, zeroed, 8, 512, 2)
    assert np.array_equal(part0[0], full[0])
    assert np.array_equal(part0[1], np.zeros(8, dtype=np.float32))


def test_the_default_split_is_one_part_per_phase_capped():
    assert intel_ternary.default_parts(128) == 1
    assert intel_ternary.default_parts(1024) == 8
    assert intel_ternary.default_parts(4096) == 32
    assert intel_ternary.default_parts(8192) == 64
    assert intel_ternary.default_parts(65536) == intel_ternary.MAX_PARTS


def test_the_probe_reports_the_split_it_used():
    report = intel_ternary.run_probe(rows=8, cols=1024, loader="missing-opencl-loader-probe.dll")
    assert report["shape"]["parts"] == 8
    assert "phases" in report["shape"]["parts_source"]
    explicit = intel_ternary.run_probe(rows=8, cols=1024, parts=3,
                                       loader="missing-opencl-loader-probe.dll")
    assert explicit["shape"]["parts"] == 3
    assert explicit["shape"]["parts_source"] == "--parts"
    assert report["kernel"]["names"] == ["ternary_gemv_part4", "ternary_gemv_reduce"]
    scalar = intel_ternary.run_probe(rows=8, cols=1024, kernel="scalar",
                                     loader="missing-opencl-loader-probe.dll")
    assert scalar["kernel"]["names"] == ["ternary_gemv_part", "ternary_gemv_reduce"]
    assert scalar["kernel"]["variant"] == "scalar"


def test_the_variant_defaults_to_the_four_row_kernel_when_it_fits():
    """Both variants are bit-exact; `auto` only picks the faster layout."""
    assert intel_ternary.resolve_variant("auto", 8) == "rows4"
    assert intel_ternary.resolve_variant("auto", 4) == "rows4"
    assert intel_ternary.resolve_variant("auto", 3) == "scalar"
    assert intel_ternary.resolve_variant("rows4", 2) == "rows4"
    assert intel_ternary.resolve_variant("scalar", 4096) == "scalar"
    assert intel_ternary.resolve_variant("nonsense", 8) is None
    assert set(intel_ternary.VARIANTS) == {"auto", "scalar", "rows4"}


def test_an_unknown_kernel_is_refused_before_any_device_work():
    report = intel_ternary.run_probe(rows=8, cols=1024, kernel="nonsense",
                                     loader="missing-opencl-loader-probe.dll")
    assert "--kernel" in report["error"]
    assert report["result"] == "fail"


def test_the_probe_is_flagged_as_a_probe():
    report = intel_ternary.run_probe(loader="missing-opencl-loader-probe.dll")
    assert report["result"] == "fail"
    assert report["loader"]["available"] is False
    assert report["support_claim"] is False
    assert report["probe"] == "intel-ternary-gemv"
    assert "no sycl" in report["backend"]
    assert "OpenCL loader unavailable" in report["error"] or "could not load" in report["error"]


def test_the_shape_is_validated_before_any_device_work():
    assert "multiple of 128" in intel_ternary.run_probe(cols=100)["error"]
    assert "--rows" in intel_ternary.run_probe(rows=0)["error"]
    assert "--repeats" in intel_ternary.run_probe(repeats=0)["error"]
    assert "--device" in intel_ternary.run_probe(device=-1)["error"]
    assert "--parts" in intel_ternary.run_probe(parts=0)["error"]
    assert "--parts" in intel_ternary.run_probe(parts=intel_ternary.MAX_PARTS + 1)["error"]
