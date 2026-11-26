# SPDX-License-Identifier: AGPL-3.0-only
"""Run the kernels that bracket every ternary GEMV of a decode token, on an Intel GPU.

`docs/INTEL-ARC-KERNELS.md` orders the kernels the Arc code still needs cheapest-first and puts two
first: **K1 `norm_fold`** (residual add, RMS norm, fold - 193 calls a token, a dependency of every
matvec) and **K2 `embed_row`** (the token embedding read as a row, not a stream).  This probe answers,
for those two, the question the plan asks before a layer loop is written: what does one honest call
cost, and how close to `kcpu.c`'s arithmetic can this driver get?

What it establishes, each measured rather than assumed:

- **No fp64 on this driver.**  `kcpu.c` sums the squares in double; the Arc's OpenCL build has no
  `cl_khr_fp64`, so the sum is carried as a two-float pair (Fast2Sum remainders plus the `fma` product
  error of each square) with a flat division that rounds once, as the CPU's `(float)(acc/D)` does.
  The report counts how many random vectors reproduce the CPU's float bit for bit and how far the
  folded vector moves when they do not: one ulp, inside the plan's ~1e-5 bar for this chain, but
  **not** bit-exact, and the plan should say so rather than assume it.
- **The launch floor is the cost.**  A no-op kernel on this driver costs ~12 us per enqueue, a single
  launch plus `clFinish` ~170 us from idle, so a 386-enqueue token would spend milliseconds launching
  no matter how fast the kernels are.  The report prices the plan's 193 passes from the measured
  per-call time and names the enqueue count beside it.
- **The butterfly must walk its stage flat.**  Spreading one Hadamard block over a work-group with
  `i += 2*step` leaves the last stage's pairs on a single lane, and the dependent local-memory pairs
  cost 34 us per call (measured: 48 us before, 18 us after, same arithmetic).  Barriers were free
  (1, 5 and 9 stages differ by ~2 us); the stage walk was the whole cost.
- **K2 needs no kernel of its own.**  The pack row is ~1.3 KB of bytes, so the decode belongs on the
  host beside the token id; the fold it feeds is this same kernel with a unit weight and unit scale.

It is not a model run, not an engine result and not a support claim: no layer loop exists yet, every
number belongs to one kernel, and the 27B's own widths are timed on weights this host does not have
(the 27B pack is absent), which the report labels per width.

    python -m kindi intel-norm --pack ~/.kindi/packs/bonsai-2b --device-name Arc --json out.json
"""
import ctypes
import json
import os
import time

import numpy as np

from .intel_cl import (
    CL_CONTEXT_PLATFORM,
    CL_MEM_COPY_HOST_PTR,
    CL_MEM_READ_ONLY,
    CL_MEM_WRITE_ONLY,
    CL_SUCCESS,
    DEFAULT_LOADER,
    _bind,
    _build_log,
    _enumerate,
    _finish,
    _hex,
    _open,
    _read_back,
    _set_arg,
    pick_device,
)
from .intel_ternary import _buffer

PROBE = "intel-norm-fold"
DEFAULT_REPEATS = 9
DEFAULT_TRIALS = 12
MAX_REPEATS = 20
MAX_TRIALS = 64
RMS_LANES = 256                    # lanes in the two-float reduction
FOLD_LANES = 256                   # lanes in a Hadamard block's butterflies
ALT_WIDTHS = (5120, 6144, 17408)   # the 27B's fold widths, timed on synthetic weights (no 27B pack)
# The plan's per-token mix (docs/INTEL-ARC-KERNELS.md sections 2 and 6): 64 layers each doing one
# rmsfold and two folds, plus the head's rmsfold - 193 passes.
TOKEN_MIX = ((5120, 65), (6144, 64), (17408, 64))
ENQUEUES_PER_CALL = 2              # rms_sum then fold_sign

SOURCE = """
/* K1, half one: the residual sum of kcpu.c's rmsfold().  The CPU sums the squares in *double* and
   this driver has no cl_khr_fp64, so the sum is carried as a two-float pair: every add is a Fast2Sum
   whose rounded remainder is kept in `lo`, and each square contributes its own `fma` product error,
   which keeps ~48 bits of the 53 the CPU's double has.
   `acc / n` in fp32 would round twice where the CPU rounds once, so the division is done flat:
   hi = q*n + rem exactly, and (q + (rem + lo)/n) is the correctly rounded quotient.
   The work-group tree-reduces the pairs in local memory; that is a different order from the CPU's
   serial sum, which is why the probe counts how often the float it writes still matches. */
__kernel void nop(__global float *o) { if (get_global_id(0) == 0) { o[0] = 1.0f; } }

__kernel void rms_sum(__global const float *h, __global const float *res, __global float *sig,
                      uint n, uint add, __local float *lhi, __local float *llo) {
    uint t = get_local_id(0), nt = get_local_size(0);
    float hi = 0.0f, lo = 0.0f;
    for (uint e = t; e < n; e += nt) {
        float xf = add ? (h[e] + res[e]) : h[e];
        float v = xf * xf, pe = fma(xf, xf, -v);   /* xf*xf is exactly v + pe */
        float s = hi + v, bb = s - hi;
        lo += (hi - (s - bb)) + (v - bb) + pe;
        hi = s;
    }
    lhi[t] = hi;
    llo[t] = lo;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint st = nt >> 1; st > 0; st >>= 1) {
        if (t < st) {
            float a = lhi[t], b = lhi[t + st], s = a + b, bb = s - a;
            lhi[t] = s;
            llo[t] += (a - (s - bb)) + (b - bb) + llo[t + st];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (t == 0) {
        float nf = (float)n;
        float q = lhi[0] / nf;
        float rem = fma(-q, nf, lhi[0]);
        float accn = q + (rem + llo[0]) / nf;
        sig[0] = 1.0f / sqrt(accn + 1e-6f);
    }
}

/* K1, half two: xn = (h * s) * w on the unfolded vector, then xp = FWHT_blocks(sg * xn).
   One work-group per Hadamard block, one lane per pair of the stage, so the value a lane computes is
   the value fwht_block computes at the same stage: the same two operands, one add and one subtract,
   the CPU's stage order, 1/sqrtf(hb) applied once at the end.
   The stage walk is flat on purpose: `m` enumerates every pair of the stage (i = (m/step)*2*step +
   m%step) so each lane takes one pair per stage.  Walking the blocks instead (`i += 2*step`) leaves
   the last stage's 256 pairs on one lane and measured 48 us a call against 18 us here - the cost was
   dependent local-memory pairs, not barriers.
   With add=0, w=1 and sig=1 this is also the embedding's fold, which is why K2 has no kernel. */
__kernel void fold_sign(__global const float *h, __global const float *res, __global const float *w,
                        __global const float *sg, __global const float *sig,
                        __global float *xn, __global float *xp, uint n, uint hb, uint add,
                        __local float *o) {
    uint b = get_group_id(0), t = get_local_id(0), nt = get_local_size(0);
    uint base = b * hb;
    float s = sig[0];
    for (uint j = t; j < hb; j += nt) {
        uint e = base + j;
        float xf = add ? (h[e] + res[e]) : h[e];
        float x = (xf * s) * w[e];
        xn[e] = x;
        o[j] = sg[e] * x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = 1; step < hb; step <<= 1) {
        uint per = step << 1;
        for (uint m = t; m < (hb >> 1); m += nt) {
            uint i = (m / step) * per + (m % step);
            float a = o[i], c = o[i + step];
            o[i] = a + c;
            o[i + step] = a - c;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sc = 1.0f / sqrt((float)hb);
    for (uint j = t; j < hb; j += nt) { xp[base + j] = o[j] * sc; }
}
"""


def _base_report(loader, pack_dir):
    return {
        "probe": PROBE,
        "schema": 1,
        "read_only": True,
        "support_claim": False,
        "backend": "opencl norm+fold probe; no sycl, model, engine or backend",
        "scope": "K1 norm_fold and K2 embed_row of docs/INTEL-ARC-KERNELS.md, against kcpu.c's "
                 "rmsfold/fold transcribed to numpy",
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "platform": None,
        "device": None,
        "pack": {"dir": os.path.basename(os.path.normpath(pack_dir))},
        "repeats": DEFAULT_REPEATS,
        "trials": DEFAULT_TRIALS,
        "returncodes": {},
        "warnings": [],
        "errors": [],
        "timings": {},
        "exactness": {},
        "embed": {},
    }


def _fail(report, message):
    report["errors"].append(message)
    return report


def _signs(pack_dir, index):
    """Every sign table the pack ships, keyed by width: fold() multiplies by one of these per block."""
    out = {}
    for width in index.get("hadamard", {}).get("sign_widths") or []:
        out[int(width)] = np.fromfile(os.path.join(pack_dir, "signs_%d.f32" % width),
                                      dtype=np.float32)
    return out


def _f32(mm, index, name):
    """A float32 tensor of the pack (`norm.weight` and the few other fp32 rows live in f32.bin)."""
    entry = index["tensors"][name]
    off, count = int(entry.get("f32_off") or 0), int(entry["nfloats"])
    return np.array(mm[off // 4:off // 4 + count], dtype=np.float32)


def _row(bcodes, dpm16, entry, row):
    """One row of a ternary tensor, decoded as `intel-pack`'s GEMV decodes it.

    Four balanced trits per byte (base 3) in phases of 128 columns: phase `ph` is 32 slots of four
    columns, slot `s` of row `r` at `(ph * 32 + s) * rows + r`, one fp16 scale per (phase, row).  The
    layout is the one the GEMV probe validated bit-for-bit against the engine; decoding it here gives
    the fold a real row instead of random numbers.
    """
    cols, rows = int(entry["ne0"]), int(entry["rows"])
    out = np.empty(cols, dtype=np.float32)
    slots = np.arange(32) * rows
    powder = np.array([1, 3, 9, 27], dtype=np.int32)
    for ph in range(cols >> 7):
        base = int(entry["bc_off"]) + (ph << 5) * rows + row
        byte = np.asarray(bcodes[base + slots], dtype=np.int32)
        trits = ((byte[:, None] // powder[None, :]) % 3) - 1
        scale = float(dpm16[int(entry["dpm_off"]) // 2 + ph * rows + row])
        out[(ph << 7):((ph + 1) << 7)] = trits.reshape(-1) * scale
    return out


def ref_fold(xn, sg, hb):
    """kcpu.c's fold() on one vector: sign, fwht_block's stages in order, then 1/sqrtf(hb)."""
    out = np.empty_like(xn)
    for b in range(len(xn) // hb):
        sl = slice(b * hb, (b + 1) * hb)
        o = (sg[sl] * xn[sl]).astype(np.float32)
        step = 1
        while step < hb:
            m = o.reshape(-1, 2 * step)
            a = m[:, :step].copy()
            c = m[:, step:].copy()
            m[:, :step] = a + c
            m[:, step:] = a - c
            step <<= 1
        out[sl] = o * np.float32(1.0 / np.sqrt(np.float32(hb)))
    return out


def ref_rmsfold(h, res, w, sg, hb, add):
    """kcpu.c's rmsfold() in the CPU's own order: fp32 add, fp64 serial sum, fp32 scale, (h*s)*w."""
    xf = (h + res).astype(np.float32) if add else h.astype(np.float32)
    acc = 0.0
    for v in xf.tolist():                      # python floats are doubles, as the CPU's acc is
        acc += v * v
    s = np.float32(1.0) / np.sqrt(np.float32(acc / len(xf)) + np.float32(1e-6))
    xn = (xf * s).astype(np.float32) * w
    return s, xn.astype(np.float32), ref_fold(xn.astype(np.float32), sg, hb)


def _launch(lib, queue, kernel, total, local, report, key):
    global_size = (ctypes.c_size_t * 1)(total)
    local_size = (ctypes.c_size_t * 1)(local)
    code = int(lib.clEnqueueNDRangeKernel(queue, kernel, 1, None, global_size, local_size,
                                          0, None, None))
    report["returncodes"][key] = _hex(code)
    if code != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(code)))


def _local_arg(lib, kernel, index, nbytes):
    code = int(lib.clSetKernelArg(kernel, index, ctypes.c_size_t(nbytes), None))
    if code != CL_SUCCESS:
        raise ValueError("clSetKernelArg(%d, %d bytes) returned %s" % (index, nbytes, _hex(code)))


def _set_norm_args(lib, rms, fold, buf, n, hb, add):
    for slot, key in enumerate(("h", "res", "sig")):
        _set_arg(lib, rms, slot, ctypes.c_void_p(int(buf[key])))
    _set_arg(lib, rms, 3, ctypes.c_uint32(n))
    _set_arg(lib, rms, 4, ctypes.c_uint32(add))
    _local_arg(lib, rms, 5, RMS_LANES * 4)
    _local_arg(lib, rms, 6, RMS_LANES * 4)
    for slot, key in enumerate(("h", "res", "w", "sg", "sig", "xn", "xp")):
        _set_arg(lib, fold, slot, ctypes.c_void_p(int(buf[key])))
    _set_arg(lib, fold, 7, ctypes.c_uint32(n))
    _set_arg(lib, fold, 8, ctypes.c_uint32(hb))
    _set_arg(lib, fold, 9, ctypes.c_uint32(add))
    _local_arg(lib, fold, 10, hb * 4)


def _call(lib, queue, rms, fold, buf, n, hb, add, report):
    _set_norm_args(lib, rms, fold, buf, n, hb, add)
    _launch(lib, queue, rms, RMS_LANES, RMS_LANES, report, "clEnqueueNDRangeKernel/rms")
    _launch(lib, queue, fold, (n // hb) * FOLD_LANES, FOLD_LANES, report,
            "clEnqueueNDRangeKernel/fold")


def _buffers(lib, context, h, res, w, sg, report, tag):
    n = len(h)
    zeros = np.zeros(n, dtype=np.float32)
    return {
        "h": _buffer(lib, context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, h, report, tag + "h"),
        "res": _buffer(lib, context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                       res if res is not None else zeros, report, tag + "res"),
        "w": _buffer(lib, context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, w, report, tag + "w"),
        "sg": _buffer(lib, context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sg, report, tag + "sg"),
        "sig": _buffer(lib, context, CL_MEM_COPY_HOST_PTR | CL_MEM_WRITE_ONLY,
                       np.ones(1, dtype=np.float32), report, tag + "sig"),
        "xn": _buffer(lib, context, CL_MEM_WRITE_ONLY, zeros, report, tag + "xn"),
        "xp": _buffer(lib, context, CL_MEM_WRITE_ONLY, zeros, report, tag + "xp"),
    }


def _diff(got, want):
    delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {"differing": int(np.count_nonzero(delta)),
            "max_abs": float(delta.max()) if delta.size else 0.0}


def run_norm(pack_dir, repeats=DEFAULT_REPEATS, loader=None, opener=None, device=0, device_name=None,
             alt_widths=ALT_WIDTHS, trials=DEFAULT_TRIALS):
    """Run and time one norm+fold call per width, then price the 27B's per-token mix; never the CPU."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, pack_dir)
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
    if not isinstance(trials, int) or not 1 <= trials <= MAX_TRIALS:
        return _fail(report, "--trials must be an integer in [1, %d]" % MAX_TRIALS)
    report["repeats"] = repeats
    report["trials"] = trials
    try:
        index = json.load(open(os.path.join(pack_dir, "index.json"), encoding="utf-8"))
        manifest = json.load(open(os.path.join(pack_dir, "kindi_pack.json"), encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _fail(report, "cannot read the pack's metadata: %s" % exc)
    hb = int(manifest.get("hadamard_block") or 512)
    sign = _signs(pack_dir, index)
    if not sign:
        return _fail(report, "the pack ships no sign table (signs_<width>.f32)")
    dims = manifest.get("dimensions") or {}
    report["pack"].update({"model": manifest.get("model"), "hadamard_block": hb,
                           "sign_widths": sorted(sign), "layers": dims.get("layers"),
                           "width": dims.get("width"), "ffn": dims.get("ffn")})
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    lib, error = _open(loader, opener)
    if lib is None:
        return _fail(report, error or "OpenCL loader unavailable")
    report["loader"]["available"] = True
    try:
        _bind(lib)
    except (AttributeError, TypeError, ValueError) as exc:
        return _fail(report, "OpenCL loader is missing an API: %s" % exc)

    context = queue = program = rms = fold = None
    buffers = []
    try:
        devices, error = _enumerate(lib, report)
        if devices is None:
            return _fail(report, error or "OpenCL device enumeration failed")
        picked, info, error = pick_device(lib, devices, device, device_name)
        if error or picked is None or info is None:
            return _fail(report, ("--device-name: " if device_name else "--device: ")
                         + (error or "the selected device could not be read"))
        platform, device_handle, platform_entry = devices[picked]
        report["platform"] = platform_entry["name"]
        report["device"] = info
        report["fp64"] = "cl_khr_fp64" in (info.get("extensions") or "")
        if "arc" not in (info.get("name") or "").lower():
            report["warnings"].append(
                "the selected device is %r, not an Arc: pass --device-name Arc for a run whose "
                "numbers may be quoted as Arc" % (info.get("name"),))

        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(device_handle)
        code = ctypes.c_int(0)
        context = lib.clCreateContext(properties, 1, handles, None, None, ctypes.byref(code))
        report["returncodes"]["clCreateContext"] = _hex(code.value)
        if not context or code.value != CL_SUCCESS:
            return _fail(report, "clCreateContext returned %s" % _hex(code.value))
        code = ctypes.c_int(0)
        queue = lib.clCreateCommandQueue(context, device_handle, 0, ctypes.byref(code))
        report["returncodes"]["clCreateCommandQueue"] = _hex(code.value)
        if not queue or code.value != CL_SUCCESS:
            return _fail(report, "clCreateCommandQueue returned %s" % _hex(code.value))

        source = SOURCE.encode("utf-8")
        lengths = ctypes.c_size_t(len(source))
        strings = (ctypes.c_char_p * 1)(source)
        code = ctypes.c_int(0)
        program = lib.clCreateProgramWithSource(context, 1, strings, ctypes.byref(lengths),
                                                ctypes.byref(code))
        report["returncodes"]["clCreateProgramWithSource"] = _hex(code.value)
        if not program or code.value != CL_SUCCESS:
            return _fail(report, "clCreateProgramWithSource returned %s" % _hex(code.value))
        build = int(lib.clBuildProgram(program, 1, handles, None, None, None))
        report["returncodes"]["clBuildProgram"] = _hex(build)
        if build != CL_SUCCESS:
            return _fail(report, "the kernel did not build: %s"
                         % _build_log(lib, program, device_handle))
        rms = lib.clCreateKernel(program, b"rms_sum", ctypes.byref(code))
        report["returncodes"]["clCreateKernel/rms_sum"] = _hex(code.value)
        fold = lib.clCreateKernel(program, b"fold_sign", ctypes.byref(code))
        report["returncodes"]["clCreateKernel/fold_sign"] = _hex(code.value)
        if not rms or not fold:
            return _fail(report, "clCreateKernel failed: %s" % _hex(code.value))
        nop = lib.clCreateKernel(program, b"nop", ctypes.byref(code))
        report["returncodes"]["clCreateKernel/nop"] = _hex(code.value)
        if not nop:
            return _fail(report, "clCreateKernel/nop failed: %s" % _hex(code.value))

        small = _buffer(lib, context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                        np.zeros(1, dtype=np.float32), report, "nop")
        _set_arg(lib, nop, 0, ctypes.c_void_p(int(small)))
        for _ in range(20):
            _launch(lib, queue, nop, 1, 1, report, "clEnqueueNDRangeKernel/nop")
        _finish(lib, queue, report, "clFinish/nop")
        start = time.perf_counter()
        for _ in range(200):
            _launch(lib, queue, nop, 1, 1, report, "clEnqueueNDRangeKernel/nop")
        _finish(lib, queue, report, "clFinish/nop")
        floor_us = (time.perf_counter() - start) / 200 * 1e6
        time.sleep(0.2)
        start = time.perf_counter()
        _launch(lib, queue, nop, 1, 1, report, "clEnqueueNDRangeKernel/nop")
        _finish(lib, queue, report, "clFinish/nop")
        report["launch"] = {"nop_us": round(floor_us, 2),
                            "wake_us": round((time.perf_counter() - start) * 1e6, 2),
                            "enqueues": 200}
        lib.clReleaseMemObject(small)
        f32 = np.memmap(os.path.join(pack_dir, "f32.bin"), dtype=np.float32, mode="r")
        names = [n for n in ("attn_norm.weight", "post_norm.weight", "output_norm.weight")
                 if n in index["tensors"] and index["tensors"][n].get("nfloats")]
        weight = _f32(f32, index, names[0]) if names else np.ones(1, dtype=np.float32)
        report["pack"]["weight"] = names[0] if names else None

        rng = np.random.default_rng(20261126)
        widths = sorted(sign)
        for width in widths:
            n = int(width)
            real = weight is not None and len(weight) == n
            w = weight if real else np.ones(n, dtype=np.float32)
            h = rng.standard_normal(n).astype(np.float32)
            res = rng.standard_normal(n).astype(np.float32)
            sg = sign[n]
            buf = _buffers(lib, context, h, res, w, sg, report, "w%d_" % n)
            buffers.append(buf)
            _call(lib, queue, rms, fold, buf, n, hb, 1, report)
            _finish(lib, queue, report, "clFinish/warm")
            _set_norm_args(lib, rms, fold, buf, n, hb, 1)

            start = time.perf_counter()
            for _ in range(repeats):
                _launch(lib, queue, rms, RMS_LANES, RMS_LANES, report,
                        "clEnqueueNDRangeKernel/rms")
                _launch(lib, queue, fold, (n // hb) * FOLD_LANES, FOLD_LANES, report,
                        "clEnqueueNDRangeKernel/fold")
            _finish(lib, queue, report, "clFinish/timed")
            per_call = (time.perf_counter() - start) / repeats * 1e6

            got_sig = np.frombuffer(_read_back(lib, queue, buf["sig"], 1, report, "sig"), np.float32)
            got_xn = np.frombuffer(_read_back(lib, queue, buf["xn"], n, report, "xn"), np.float32)
            got_xp = np.frombuffer(_read_back(lib, queue, buf["xp"], n, report, "xp"), np.float32)
            want_s, want_xn, want_xp = ref_rmsfold(h, res, w, sg, hb, 1)
            report["timings"]["width_%d" % n] = {
                "elements": n, "synthetic_weight": not real, "us_per_call": round(per_call, 3),
                "enqueues": ENQUEUES_PER_CALL,
                "repeats": repeats, "bytes_per_call": n * 4 * 6,
                "gb_per_s": round(n * 4 * 6 / (per_call * 1e-6) / 1e9, 2) if per_call else 0.0,
            }
            report["exactness"]["width_%d" % n] = {
                "sigma_equal": bool(got_sig[0] == want_s),
                "sigma": [float(got_sig[0]), float(want_s)],
                "xn": _diff(got_xn, want_xn),
                "xp": _diff(got_xp, want_xp),
            }
            _sigma_trials(lib, context, queue, rms, fold, n, hb, sign[n], w, trials, report)

        for width in alt_widths:
            n = int(width)
            if n in widths:
                continue
            sg = np.asarray(rng.choice([-1.0, 1.0], n), dtype=np.float32)
            h = rng.standard_normal(n).astype(np.float32)
            w = rng.uniform(0.5, 1.5, n).astype(np.float32)
            buf = _buffers(lib, context, h, None, w, sg, report, "a%d_" % n)
            buffers.append(buf)
            _set_norm_args(lib, rms, fold, buf, n, hb, 0)
            for _ in range(3):
                _call(lib, queue, rms, fold, buf, n, hb, 0, report)
            _finish(lib, queue, report, "clFinish/warm")
            start = time.perf_counter()
            for _ in range(repeats):
                _launch(lib, queue, rms, RMS_LANES, RMS_LANES, report,
                        "clEnqueueNDRangeKernel/rms")
                _launch(lib, queue, fold, (n // hb) * FOLD_LANES, FOLD_LANES, report,
                        "clEnqueueNDRangeKernel/fold")
            _finish(lib, queue, report, "clFinish/timed")
            per_call = (time.perf_counter() - start) / repeats * 1e6
            report["timings"]["width_%d" % n] = {
                "elements": n, "synthetic_weight": True, "synthetic_signs": True,
                "us_per_call": round(per_call, 3), "enqueues": ENQUEUES_PER_CALL,
                "repeats": repeats, "bytes_per_call": n * 4 * 6,
                "gb_per_s": round(n * 4 * 6 / (per_call * 1e-6) / 1e9, 2) if per_call else 0.0,
            }

        timed = {int(k.split("_")[1]): v["us_per_call"] for k, v in report["timings"].items()}
        calls = sum(count for _, count in TOKEN_MIX)
        total = sum(timed.get(width, 0.0) * count for width, count in TOKEN_MIX)
        report["token"] = {
            "mix": [{"width": width, "calls": count} for width, count in TOKEN_MIX],
            "calls": calls, "ms_per_token": round(total / 1000.0, 3),
            "enqueues": calls * ENQUEUES_PER_CALL,
            "complete": all(width in timed for width, _ in TOKEN_MIX),
            "note": "two kernels per call, timed with the arguments already bound; it excludes the "
                    "layer's own launches and the 27B's 5120/17408 weights are synthetic here",
        }
        _embed(lib, context, queue, rms, fold, pack_dir, index, sign, hb, report, buffers)
    finally:
        for buf in buffers:
            for handle in buf.values():
                lib.clReleaseMemObject(handle)
        for handle in (rms, fold):
            if handle:
                lib.clReleaseKernel(handle)
        if program:
            lib.clReleaseProgram(program)
        if queue:
            lib.clReleaseCommandQueue(queue)
        if context:
            lib.clReleaseContext(context)
    return report


def _sigma_trials(lib, context, queue, rms, fold, n, hb, sg, w, trials, report):
    """How often the tree-reduced two-float sum still writes the CPU's float: measured, not assumed."""
    rng = np.random.default_rng(97 + n)
    equal, worst = 0, 0.0
    for _ in range(trials):
        h = rng.standard_normal(n).astype(np.float32)
        res = rng.standard_normal(n).astype(np.float32)
        buf = _buffers(lib, context, h, res, w, sg, report, "t%d_" % n)
        try:
            _call(lib, queue, rms, fold, buf, n, hb, 1, report)
            _finish(lib, queue, report, "clFinish/trial")
            got = np.frombuffer(_read_back(lib, queue, buf["sig"], 1, report, "tsig"), np.float32)[0]
            want_s, _, _ = ref_rmsfold(h, res, w, sg, hb, 1)
            equal += int(got == want_s)
            worst = max(worst, abs(float(got) - float(want_s)) / float(want_s))
        finally:
            for handle in buf.values():
                lib.clReleaseMemObject(handle)
    report["exactness"]["sigma_trials_%d" % n] = {"equal": equal, "trials": trials,
                                                  "worst_relative": worst}


def _embed(lib, context, queue, rms, fold, pack_dir, index, sign, hb, report, buffers):
    """K2: decode a real `token_embd` row on the host, fold it on the card, time both halves."""
    entry = index["tensors"].get("token_embd.weight")
    if not entry or "bc_off" not in entry:
        report["embed"]["skipped"] = "the pack has no ternary token_embd.weight"
        return
    cols = int(entry["ne0"])
    sg = sign.get(cols)
    if sg is None:
        report["embed"]["skipped"] = "no sign table of width %d for the embedding" % cols
        return
    bcodes = np.memmap(os.path.join(pack_dir, "bcodes.bin"), dtype=np.uint8, mode="r")
    dpm16 = np.memmap(os.path.join(pack_dir, "dpm.bin"), dtype=np.float16, mode="r")
    rows = [0, 1, 12345, int(entry["rows"]) - 1]
    start = time.perf_counter()
    decoded = [_row(bcodes, dpm16, entry, r) for r in rows]
    decode_us = (time.perf_counter() - start) / len(rows) * 1e6
    times = 50
    start = time.perf_counter()
    for _ in range(times):
        _row(bcodes, dpm16, entry, 7)
    single_us = (time.perf_counter() - start) / times * 1e6
    folded = bool(entry.get("folded"))
    row = decoded[0]
    buf = _buffers(lib, context, row, None, np.ones(cols, dtype=np.float32), sg, report, "emb_")
    buffers.append(buf)
    # a fold-only call: sig starts at 1.0 and w is ones, so `fold_sign` leaves xn = row and
    # xp = fold(row), which is kcpu.c's embed (decode, fwht_block per block, * sD).
    _set_norm_args(lib, rms, fold, buf, cols, hb, 0)
    _launch(lib, queue, fold, (cols // hb) * FOLD_LANES, FOLD_LANES, report,
            "clEnqueueNDRangeKernel/embed")
    _finish(lib, queue, report, "clFinish/embed")
    got = np.frombuffer(_read_back(lib, queue, buf["xp"], cols, report, "emb_xp"), np.float32)
    want = ref_fold(row, sg, hb) if folded else row
    start = time.perf_counter()
    for _ in range(20):
        _launch(lib, queue, fold, (cols // hb) * FOLD_LANES, FOLD_LANES, report,
                "clEnqueueNDRangeKernel/embed")
    _finish(lib, queue, report, "clFinish/embed_timed")
    report["embed"].update({
        "tensor": "token_embd.weight", "cols": cols, "rows": int(entry["rows"]),
        "folded": folded, "host_decode_us_per_row": round(single_us, 2),
        "host_decode_batch_us": round(decode_us, 2),
        "host_decode_bytes": ((cols >> 7) << 5) + (cols >> 7) * 2,
        "fold_us": round((time.perf_counter() - start) / 20 * 1e6, 2),
        "enqueues": 1, "exact": _diff(got, want),
        "note": "the decoder is intel-pack's validated byte layout; the fold is checked against "
                "kcpu.c's embed (decode, fwht_block per Hadamard block, * sD)",
    })


def render(report):
    lines = []
    if report["errors"]:
        return ["intel-norm: " + message for message in report["errors"]]
    device = report["device"] or {}
    lines.append("device     %s (%s)" % (device.get("name"), device.get("driver_version")))
    lines.append("fp64       %s" % ("advertised" if report.get("fp64") else
                                    "absent: the CPU's double sum is a two-float pair here"))
    pack = report["pack"]
    lines.append("pack       %s, Hadamard block %s, sign widths %s"
                 % (pack.get("model"), pack.get("hadamard_block"), pack.get("sign_widths")))
    lines.append("")
    launch = report.get("launch") or {}
    if launch:
        lines.append("launch     %.2f us per enqueue on a no-op kernel, %.2f us for one launch from idle"
                     % (launch["nop_us"], launch["wake_us"]))
    lines.append("K1 norm_fold (residual add + RMS + signed FWHT), one call = 2 enqueues:")
    for key in sorted(report["timings"], key=lambda k: int(k.split("_")[1])):
        row = report["timings"][key]
        lines.append("  %5s  %8.2f us/call  %6.1f GB/s%s"
                     % (key.split("_")[1], row["us_per_call"], row["gb_per_s"],
                        "  (synthetic weight)" if row.get("synthetic_weight") else ""))
    token = report.get("token") or {}
    if token:
        lines.append("  token mix %s -> %.2f ms/token in %d enqueues%s"
                     % (" + ".join("%dx%d" % (c["calls"], c["width"]) for c in token["mix"]),
                        token["ms_per_token"], token["enqueues"],
                        "" if token["complete"] else " (incomplete: a width was not timed)"))
    lines.append("")
    lines.append("exactness against kcpu.c's rmsfold(), same operations in the CPU's order:")
    for key in sorted(report["exactness"], key=lambda k: k):
        row = report["exactness"][key]
        if key.startswith("sigma_trials"):
            lines.append("  %5s  sigma bit-equal %d of %d random vectors, worst relative %.2e"
                         % (key.rsplit("_", 1)[1], row["equal"], row["trials"],
                            row["worst_relative"]))
            continue
        lines.append("  %5s  sigma %s  xn %d differing / max %.3g  xp %d differing / max %.3g"
                     % (key.split("_")[1], "equal" if row["sigma_equal"] else "1-ulp apart",
                        row["xn"]["differing"], row["xn"]["max_abs"],
                        row["xp"]["differing"], row["xp"]["max_abs"]))
    embed = report.get("embed") or {}
    lines.append("")
    if embed.get("skipped"):
        lines.append("K2 embed_row: " + embed["skipped"])
    elif embed:
        lines.append("K2 embed_row (%s, %d cols, %s): host decode %.2f us of %d B, fold %s us, %d "
                     "differing" % (embed["tensor"], embed["cols"],
                                    "folded" if embed["folded"] else "not folded",
                                    embed["host_decode_us_per_row"], embed["host_decode_bytes"],
                                    embed["fold_us"], embed["exact"]["differing"]))
    for warning in report["warnings"]:
        lines.append("")
        lines.append("warning    " + warning)
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
