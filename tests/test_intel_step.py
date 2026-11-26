# SPDX-License-Identifier: AGPL-3.0-only
"""The Intel step wiring: the float host model of the ternary GEMV, without a GPU.

The pack kernels accumulate the folded input in float32 (a real GEMV), and the host model in
`intel_ternary` is the bit-exact reference for them. These tests pin the two halves of that: the
model is float (with non-integer inputs it must NOT agree with the old int-truncated model, which
is the bug that made `pack_gemv4` return garbage on real activations), the split model is
consistent with the whole, and `_rms_rows` mirrors kcpu.c's rms_rows (mean square, not sum).
"""
import numpy as np

from kindi import intel_step
from kindi import intel_ternary


def _int_model(bcodes, dpm, x, rows, cols):
    """The pre-fix host model: x truncated to int32 before the dot."""
    phases = cols // intel_ternary.PHASE
    codes = np.asarray(bcodes, dtype=np.uint8).reshape(phases, 32, rows)
    scales = np.asarray(dpm, dtype=np.float16).astype(np.float32)
    out = np.zeros(rows, dtype=np.float32)
    for ph in range(phases):
        b = codes[ph].astype(np.int32)
        xv = np.asarray(x, dtype=np.int32)[ph * intel_ternary.PHASE:(ph + 1) * intel_ternary.PHASE].reshape(32, 4)
        dot = ((b % 3 - 1) * xv[:, 0][:, None]
               + (b // 3 % 3 - 1) * xv[:, 1][:, None]
               + (b // 9 % 3 - 1) * xv[:, 2][:, None]
               + (b // 27 % 3 - 1) * xv[:, 3][:, None]).sum(axis=0)
        out = (out + dot.astype(np.float32) * scales[ph]).astype(np.float32)
    return out


def test_float_model_differs_from_int_model_on_real_activations():
    _trits, bcodes, dpm, x = intel_ternary.matrix(64, 512)
    f = intel_ternary.reference(bcodes, dpm, x, 64, 512)
    i = _int_model(bcodes, dpm, x, 64, 512)
    assert not np.allclose(f, i)
    # and the integer model is wrong by a large factor, not a rounding error
    assert np.abs(f - i).max() > 1.0


def test_float_model_still_matches_int_model_on_integer_activations():
    _trits, bcodes, dpm, _x = intel_ternary.matrix(64, 512)
    x = ((np.arange(512, dtype=np.int64) * 5 + 3) % 7 - 3).astype(np.float32)
    f = intel_ternary.reference(bcodes, dpm, x, 64, 512)
    i = _int_model(bcodes, dpm, x, 64, 512)
    assert np.array_equal(f, i)


def test_partials_sum_to_reference():
    for parts in (1, 2, 4):
        _trits, bcodes, dpm, x = intel_ternary.matrix(64, 512)
        split = intel_ternary.partials(bcodes, dpm, x, 64, 512, parts).sum(axis=0, dtype=np.float32)
        assert np.array_equal(split, intel_ternary.reference(bcodes, dpm, x, 64, 512))


def test_rms_rows_uses_mean_square():
    rng = np.random.RandomState(7)
    x = rng.randn(37).astype(np.float32)
    w = (1.0 + 0.1 * rng.randn(37)).astype(np.float32)
    got = intel_step._rms_rows(x, w, 37)
    acc = np.sum(x.astype(np.float64) * x.astype(np.float64))
    sc = np.float32(1.0) / np.float32(np.sqrt(np.float32(acc / np.float64(37)) + np.float32(1e-6)))
    want = ((x * sc) * w).astype(np.float32)
    assert np.array_equal(got, want)


def _tiny_pack(tmp_path):
    """A pack directory with just what a tokenizer needs: vocab.json and merges.txt."""
    import json
    vocab = ["!"] + ["a%d" % i for i in range(250)] + ["<|im_end|>", "<|endoftext|>"]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tmp_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    return str(tmp_path)


def test_load_tokenizer_and_stop_ids_read_the_pack(tmp_path):
    """`intel-run` decodes text with the pack's own vocab; the stop set is the two end tokens."""
    tok = intel_step.load_tokenizer(_tiny_pack(tmp_path))
    assert len(tok.vocab) == 253
    assert intel_step._stop_ids(tok) == {251, 252}


def test_run_generate_reports_the_platform_error_instead_of_raising(tmp_path, monkeypatch):
    """A run with no usable GPU must come back as a report with `ok: False`, like run_step does."""
    tok = intel_step.load_tokenizer(_tiny_pack(tmp_path))
    monkeypatch.setattr(intel_step, "ArcStep", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no device")))
    report = intel_step.run_generate(str(tmp_path), [1, 2], 4, tok=tok)
    assert report["ok"] is False
    assert "no device" in report["error"]
    assert "error" in intel_step.render_generate(report)
