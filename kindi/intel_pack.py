# SPDX-License-Identifier: AGPL-3.0-only
"""Run every ternary matmul of a real pack on an Intel GPU through the pack's own bytes.

The OpenCL probes (`kindi intel-smoke`, `intel-cl`, `intel-ternary`) prove the device, a dense
GEMM and the 4-trit/byte layout on synthetic matrices. This probe answers the next question a
backend has to answer first: how fast can the card stream a **converted model's** ternary weights,
at the shapes that model actually has?

It reads `bcodes.bin`, `dpm.bin` and `index.json` exactly as `kindi convert` wrote them - there is no
Intel-specific pack and no conversion step - uploads the two buffers unchanged, and runs the
phase-parallel four-row GEMV at every distinct `(rows, cols)` among the pack's ternary tensors, then
the persistent eight-row kernel over a work list of *all* of them - one launch per upload chunk
instead of one launch pair per matrix - so the same run reports both paths on the same bytes. The
pack's scales are fp16; they are widened to float32 on the host (exact) because OpenCL C 1.2 has no
portable half arithmetic, as in `intel-ternary`.

What it reports is a **projection**: the sum of one pass over every ternary tensor, which is the
memory floor of a decode token (attention, norms, the GDN state update and the output head's
non-ternary work are excluded). It is not a model run, not an engine result and not a support
claim. It also validates one real tensor bit for bit against the host model, so "the bytes are read
correctly" is checked and not assumed.

    python -m kindi intel-pack --pack ~/.kindi/packs/bonsai2.5-27b --device-name Arc --json out.json
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
from .intel_ternary import MAX_PARTS, PHASE, _buffer, _launch_1d, _launch_2d, default_parts, partials

PROBE = "intel-pack-gemv"
TERNARY_TYPES = {143, 142, 42}          # PTQ1_0, PQ2_0, Q2_0: every converter output is phase-grouped
MIN_PHASES = 1
MAX_REPEATS = 20
DEFAULT_REPEATS = 5
VALIDATE_MAX_BYTES = 32 << 20           # validate the smallest tensor below this many packed bytes

SOURCE = """
/* Four rows per work-item, at a byte offset into the pack's single bcodes buffer.  The layout is
   [slot][row] inside a tensor, so four consecutive bytes of a slot are four rows of the same four
   columns: one vload4 per slot, and the same per-phase int32 accumulation as the synthetic probe,
   which keeps every partial bit-identical to the scalar kernel. */
#define TRI_DOT(dst, byte, c) do {                                          \\
    int bb = (int)(byte);                                                   \\
    int g1 = (bb * 171) >> 9;                                               \\
    int g2 = (g1 * 171) >> 9;                                               \\
    int g3 = (g2 * 171) >> 9;                                               \\
    (dst) += ((bb - 3 * g1) - 1) * x[(c)]                              \\
           + ((g1 - 3 * g2) - 1) * x[(c) + 1]                          \\
           + ((g2 - 3 * g3) - 1) * x[(c) + 2]                          \\
           + ((g3 - 3 * ((g3 * 171) >> 9)) - 1) * x[(c) + 3];           \\
} while (0)

__kernel void pack_gemv4(__global const uchar *codes,
                         __global const float *scales,
                         __global const float *x,
                         __global float *partial,
                         uint rows, uint cols, uint parts, ulong coff, ulong soff) {
    uint r0 = get_global_id(0) << 2;
    uint part = get_global_id(1);
    uint phases = cols >> 7;
    if (r0 >= rows || part >= parts) {
        return;
    }
    int full = (r0 + 3u) < rows;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint ph = part; ph < phases; ph += parts) {
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        ulong base = coff + ((ulong)(ph << 5) * rows) + r0;
        uint col = ph << 7;
        for (uint s = 0; s < 32; ++s) {
            ulong off = base + (ulong)s * rows;
            uint c = col + (s << 2);
            if (full) {
                uchar4 q = vload4(0, codes + off);
                TRI_DOT(d0, q.s0, c);
                TRI_DOT(d1, q.s1, c);
                TRI_DOT(d2, q.s2, c);
                TRI_DOT(d3, q.s3, c);
            } else {
                TRI_DOT(d0, codes[off], c);
                if (r0 + 1u < rows) { TRI_DOT(d1, codes[off + 1u], c); }
                if (r0 + 2u < rows) { TRI_DOT(d2, codes[off + 2u], c); }
                if (r0 + 3u < rows) { TRI_DOT(d3, codes[off + 3u], c); }
            }
        }
        ulong sb = soff + (ulong)ph * rows + r0;
        t0 += d0 * scales[sb];
        if (r0 + 1u < rows) { t1 += d1 * scales[sb + 1u]; }
        if (r0 + 2u < rows) { t2 += d2 * scales[sb + 2u]; }
        if (r0 + 3u < rows) { t3 += d3 * scales[sb + 3u]; }
    }
    ulong pb = (ulong)part * rows + r0;
    partial[pb] = t0;
    if (r0 + 1u < rows) { partial[pb + 1u] = t1; }
    if (r0 + 2u < rows) { partial[pb + 2u] = t2; }
    if (r0 + 3u < rows) { partial[pb + 3u] = t3; }
}

/* Eight rows per work-item over a *work list* of matrices: one launch for a whole tensor list.

   Why: one work-item per row leaves the card idle, and the phase split pays a second reduce
   launch per matrix.  With eight rows per work-item one vload8 covers eight consecutive rows of
   a slot, and the work list lets one launch cover every matrix, so a decode pass costs one
   enqueue instead of 400-800 (measured on the 27B shapes: 253 GB/s against 143 GB/s for the
   per-shape path, docs/benchmarks/2026-11-25-i7-8086k-arc-a770-pack/).

   parts=1 only, so each row has exactly one partial and no reduce pass is needed: phase ph is
   simply summed into the row's accumulator in order, which is the same order `pack_gemv4` uses
   at parts=1, so the values are bit-identical to the per-shape path.

   `pref[mi]` is where matrix mi's work items start, `ce = ceil(rows/8)` items each; `coffs` and
   `soffs` are the matrix's byte offsets into the two shared pack buffers; `ppref[mi]` is where its
   rows start in `partial`.  Both pack buffers are 8-byte aligned in a Kindi pack, so the vload8
   on the row axis is safe. */
__kernel void pack_gemv_persist(__global const uchar *codes,
                                __global const float *scales,
                                __global const float *x,
                                __global float *partial,
                                __global const uint *mrows,
                                __global const uint *mcols,
                                __global const uint *pref,
                                __global const ulong *coffs,
                                __global const ulong *soffs,
                                __global const uint *ppref,
                                uint nmats, uint total) {
    uint stride = get_global_size(0);
    for (uint w = get_global_id(0); w < total; w += stride) {
        uint lo = 0, hi = nmats;
        while (lo + 1 < hi) { uint mid = (lo + hi) >> 1; if (pref[mid] <= w) { lo = mid; } else { hi = mid; } }
        uint mi = lo;
        uint rows = mrows[mi], cols = mcols[mi];
        uint r0 = (w - pref[mi]) << 3;
        if (r0 >= rows) {
            continue;
        }
        uint phases = cols >> 7;
        ulong coff = coffs[mi], soff = soffs[mi];
        uint nrows = rows - r0;                       /* 8, or the tail block */
        int full = nrows >= 8u;
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float t4 = 0.0f, t5 = 0.0f, t6 = 0.0f, t7 = 0.0f;
        for (uint ph = 0; ph < phases; ++ph) {
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f, d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
            ulong base = coff + ((ulong)(ph << 5) * rows) + r0;
            uint col = ph << 7;
            for (uint s = 0; s < 32; ++s) {
                ulong off = base + (ulong)s * rows;
                uint c = col + (s << 2);
                if (full) {
                    uchar8 q = vload8(0, codes + off);
                    TRI_DOT(d0, q.s0, c);
                    TRI_DOT(d1, q.s1, c);
                    TRI_DOT(d2, q.s2, c);
                    TRI_DOT(d3, q.s3, c);
                    TRI_DOT(d4, q.s4, c);
                    TRI_DOT(d5, q.s5, c);
                    TRI_DOT(d6, q.s6, c);
                    TRI_DOT(d7, q.s7, c);
                } else {
                    TRI_DOT(d0, codes[off], c);
                    if (1u < nrows) { TRI_DOT(d1, codes[off + 1u], c); }
                    if (2u < nrows) { TRI_DOT(d2, codes[off + 2u], c); }
                    if (3u < nrows) { TRI_DOT(d3, codes[off + 3u], c); }
                    if (4u < nrows) { TRI_DOT(d4, codes[off + 4u], c); }
                    if (5u < nrows) { TRI_DOT(d5, codes[off + 5u], c); }
                    if (6u < nrows) { TRI_DOT(d6, codes[off + 6u], c); }
                    if (7u < nrows) { TRI_DOT(d7, codes[off + 7u], c); }
                }
            }
            ulong sb = soff + (ulong)ph * rows + r0;
            t0 += d0 * scales[sb];
            if (1u < nrows) { t1 += d1 * scales[sb + 1u]; }
            if (2u < nrows) { t2 += d2 * scales[sb + 2u]; }
            if (3u < nrows) { t3 += d3 * scales[sb + 3u]; }
            if (4u < nrows) { t4 += d4 * scales[sb + 4u]; }
            if (5u < nrows) { t5 += d5 * scales[sb + 5u]; }
            if (6u < nrows) { t6 += d6 * scales[sb + 6u]; }
            if (7u < nrows) { t7 += d7 * scales[sb + 7u]; }
        }
        ulong pb = (ulong)ppref[mi] + r0;
        partial[pb] = t0;
        if (1u < nrows) { partial[pb + 1u] = t1; }
        if (2u < nrows) { partial[pb + 2u] = t2; }
        if (3u < nrows) { partial[pb + 3u] = t3; }
        if (4u < nrows) { partial[pb + 4u] = t4; }
        if (5u < nrows) { partial[pb + 5u] = t5; }
        if (6u < nrows) { partial[pb + 6u] = t6; }
        if (7u < nrows) { partial[pb + 7u] = t7; }
    }
}

__kernel void pack_gemv_reduce(__global const float *partial,
                               __global float *y,
                               uint rows, uint parts) {
    uint row = get_global_id(0);
    if (row >= rows) {
        return;
    }
    float total = 0.0f;
    for (uint part = 0; part < parts; ++part) {
        total += partial[(ulong)part * rows + row];
    }
    y[row] = total;
}
"""
KERNEL = "pack_gemv4"
REDUCE = "pack_gemv_reduce"
PERSIST = "pack_gemv_persist"
ROWS_PER_ITEM = 8                       # rows one work-item of the persistent kernel covers
PERSIST_MAX_GRID = 1 << 20
PERSIST_CHUNK_FALLBACK = 1 << 30        # when the device does not report its allocation limit
PERSIST_CHUNK_SHARE = 0.94              # of CL_DEVICE_MAX_MEM_ALLOC_SIZE, leaving the driver margin


def _fail(report, message):
    report["error"] = message
    report["result"] = "fail"
    return report


def pack_tensors(pack_dir):
    """(tensors, manifest): every ternary tensor in the pack, in file order, with its byte offsets."""
    with open(os.path.join(pack_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    with open(os.path.join(pack_dir, "kindi_pack.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tensors = []
    for name, rec in index["tensors"].items():
        if int(rec.get("type", -1)) not in TERNARY_TYPES:
            continue
        rows, cols = int(rec["rows"]), int(rec["ne0"])
        if cols % PHASE or "bc_off" not in rec or "dpm_off" not in rec:
            continue
        tensors.append({
            "name": name, "rows": rows, "cols": cols, "phases": cols // PHASE,
            "bc_off": int(rec["bc_off"]), "dpm_off": int(rec["dpm_off"]),
            "codes_bytes": rows * cols // 4, "scale_bytes": rows * (cols // PHASE) * 2,
        })
    return tensors, manifest, index


def shapes_of(tensors):
    """The distinct (rows, cols) a run has to time, largest first, with how many tensors use each."""
    seen = {}
    for t in tensors:
        key = (t["rows"], t["cols"])
        entry = seen.setdefault(key, {"rows": t["rows"], "cols": t["cols"], "tensors": 0,
                                      "bytes": t["codes_bytes"] + t["scale_bytes"],
                                      "first": t["name"]})
        entry["tensors"] += 1
    return sorted(seen.values(), key=lambda s: s["bytes"], reverse=True)


def project(tensors, seconds_by_shape):
    """The decode floor from one pass over every ternary tensor: bytes, seconds and tokens/s."""
    total_bytes = 0
    total_seconds = 0.0
    for t in tensors:
        total_bytes += t["codes_bytes"] + t["scale_bytes"]
        total_seconds += seconds_by_shape[(t["rows"], t["cols"])]
    return {
        "weight_bytes": int(total_bytes),
        "seconds": round(total_seconds, 6),
        "tokens_per_second": round(1.0 / total_seconds, 3) if total_seconds > 0 else None,
        "effective_gbps": round(total_bytes / total_seconds / 1e9, 3) if total_seconds > 0 else None,
    }


def _base_report(loader, pack_dir):
    return {
        "probe": PROBE,
        "schema": 1,
        "read_only": True,
        "support_claim": False,
        "backend": "opencl pack-reading gemv probe; no sycl, model run, engine or backend",
        "scope": "every ternary matmul of a converted pack at its own shape, from the pack's "
                 "on-disk 4-trit/byte bytes; a projection of the decode memory floor, not a model run",
        "pack": {"dir": os.path.basename(os.path.abspath(pack_dir)), "model": None, "source_file": None,
                 "bcodes_bytes": None, "dpm_bytes": None, "tensors": None, "ternary": None,
                 "shapes": None},
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "platform": None,
        "device_count": None,
        "device_index": None,
        "device": None,
        "kernel": {"name": KERNEL, "reduce": REDUCE, "split": "phase ph -> part ph % parts",
                   "exactness": "one real tensor is compared bit for bit at parts=1; the timed runs "
                                "use the default parts"},
        "shapes": [],
        "persistent": {"kernel": PERSIST, "parts": 1, "rows_per_item": ROWS_PER_ITEM,
                       "seconds": None, "mean_ms": None, "weight_bytes": None,
                       "tokens_per_second": None,
                       "effective_gbps": None, "speedup": None, "chunks": [], "launches": 0,
                       "validation": {"tensor": None, "rows": None, "cols": None, "mismatches": None},
                       "excluded": "attention, norms, GDN state update, output-head f32 work, "
                                   "tokenizer and sampling"},
        "validation": {"tensor": None, "rows": None, "cols": None, "mismatches": None},
        "projection": {"weight_bytes": None, "seconds": None, "tokens_per_second": None,
                       "effective_gbps": None, "excluded": "attention, norms, GDN state update, "
                       "output-head f32 work, tokenizer and sampling"},
        "repeats": None,
        "warnings": [],
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def work_lists(chunk):
    """The persistent kernel's argument arrays for a chunk of tensors, offsets rebased to the chunk.

    `pref[mi]` is where matrix mi's work items start (ceil(rows/8) per matrix), `ppref[mi]` where its
    rows start in the partial buffer, and coffs/soffs are its byte offsets in the two chunk buffers.
    """
    rows = np.array([t["rows"] for t in chunk], dtype=np.uint32)
    cols = np.array([t["cols"] for t in chunk], dtype=np.uint32)
    each = (rows.astype(np.uint64) + (ROWS_PER_ITEM - 1)) // ROWS_PER_ITEM
    codes = np.array([t["codes_bytes"] for t in chunk], dtype=np.uint64)
    scales = np.array([t["rows"] * t["phases"] * 4 for t in chunk], dtype=np.uint64)
    return {
        "mrows": rows, "mcols": cols,
        "pref": _prefix(each, np.uint32),
        "ppref": _prefix(rows.astype(np.uint64), np.uint32),
        "coffs": _prefix(codes, np.uint64),
        "soffs": _prefix(scales, np.uint64),
        "work": int(each.sum()),
        "rows": int(rows.sum()),
    }


def _prefix(sizes, dtype):
    """Cumulative starts of `sizes`: [0, s0, s0+s1, ...] without the trailing total."""
    out = np.zeros(len(sizes), dtype=dtype)
    if len(sizes) > 1:
        out[1:] = np.cumsum(sizes, dtype=dtype)[:-1]
    return out


def chunk_bytes(device):
    """The largest upload to attempt: the card's per-allocation limit, less a margin.

    A 27B pack's 6.7 GB of codes is over the Arc A770's 4 GiB limit, so the persistent pass needs
    chunks; one chunk per allocation means every extra chunk is a launch and a smaller grid, which
    costs bandwidth (7 chunks measured 154 GB/s, 2 chunks 235 GB/s on the 2.5 27B), so the chunk is
    as large as the driver will take.
    """
    limit = (device or {}).get("max_mem_alloc_mib")
    if not limit:
        return PERSIST_CHUNK_FALLBACK
    return int(limit * (1 << 20) * PERSIST_CHUNK_SHARE)


def chunks_of(tensors, cap=None):
    """Split the tensor list into uploads whose codes and scales each fit in `cap` bytes."""
    cap = cap or PERSIST_CHUNK_FALLBACK
    chunk, codes, scales = [], 0, 0
    for tensor in tensors:
        code_bytes = tensor["codes_bytes"]
        scale_bytes = tensor["rows"] * tensor["phases"] * 4
        if chunk and (codes + code_bytes > cap or scales + scale_bytes > cap):
            yield chunk
            chunk, codes, scales = [], 0, 0
        chunk.append(tensor)
        codes += code_bytes
        scales += scale_bytes
    if chunk:
        yield chunk


def _persist_validate(lib, context, queue, kernel, x_buf, bcodes, dpm16, small, expected, report):
    """One persistent launch over a single real tensor, compared bit for bit with the host model."""
    rows = small["rows"]
    lists = work_lists([small])
    holds = [lists["mrows"], lists["mcols"], lists["pref"], lists["coffs"], lists["soffs"],
             lists["ppref"]]
    codes_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        bcodes[small["bc_off"]:small["bc_off"] + small["codes_bytes"]],
                        report, "clCreateBufferPersistValidationCodes")
    scales_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         widened(dpm16, small), report, "clCreateBufferPersistValidationScales")
    part_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                       report, "clCreateBufferPersistValidation")
    arrays = [_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, report,
                      "clCreateBufferPersistValidationList") for a in holds]
    _set_persist_args(lib, kernel, codes_buf, scales_buf, x_buf, part_buf, arrays, 1, lists["work"])
    _launch_1d(lib, queue, kernel, min(lists["work"], PERSIST_MAX_GRID), report,
               "clEnqueuePersistValidation")
    _finish(lib, queue, report, "clFinishPersistValidation")
    raw = _read_back(lib, queue, part_buf, rows, report, "clReadPersistValidation")
    actual = np.frombuffer(raw, dtype=np.float32)
    report["persistent"]["validation"].update({
        "tensor": small["name"], "rows": rows, "cols": small["cols"],
        "mismatches": int(np.count_nonzero(actual != expected)),
    })
    for handle in [codes_buf, scales_buf, part_buf] + arrays:
        lib.clReleaseMemObject(handle)


def _persist_pass(lib, context, queue, kernel, x_buf, bcodes, dpm16, tensors, repeats, report,
                  cap=None):
    """Time one persistent launch per upload chunk: a work list over every ternary tensor, parts=1.

    The pack's codes are ~5.8 GB on the 27B, over the driver's per-allocation limit, so the tensors
    go up in chunks. Each chunk is one launch of its whole work list, and the chunk times are
    summed - what a decode pass pays once the weights are resident.
    """
    section = report["persistent"]
    seconds = 0.0
    weight_bytes = 0
    section["chunk_bytes"] = cap
    for chunk in chunks_of(tensors, cap):
        lists = work_lists(chunk)
        codes = np.concatenate([bcodes[t["bc_off"]:t["bc_off"] + t["codes_bytes"]] for t in chunk])
        scales = np.concatenate([widened(dpm16, t) for t in chunk])
        holds = [lists["mrows"], lists["mcols"], lists["pref"], lists["coffs"], lists["soffs"],
                 lists["ppref"]]
        codes_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes,
                            report, "clCreateBufferPersistCodes")
        scales_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales,
                             report, "clCreateBufferPersistScales")
        part_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY,
                           np.zeros(lists["rows"], dtype=np.float32),
                           report, "clCreateBufferPersistPartial")
        arrays = [_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, report,
                          "clCreateBufferPersistList") for a in holds]
        _set_persist_args(lib, kernel, codes_buf, scales_buf, x_buf, part_buf, arrays,
                          len(chunk), lists["work"])
        grid = min(lists["work"], PERSIST_MAX_GRID)
        _launch_1d(lib, queue, kernel, grid, report, "clEnqueuePersistWarmup")
        _finish(lib, queue, report, "clFinishPersistWarmup")
        started = time.perf_counter()
        for _ in range(repeats):
            _launch_1d(lib, queue, kernel, grid, report, "clEnqueuePersist")
        _finish(lib, queue, report, "clFinishPersist")
        elapsed = (time.perf_counter() - started) / repeats
        moved = sum(t["codes_bytes"] + t["rows"] * t["phases"] * 2 for t in chunk)
        seconds += elapsed
        weight_bytes += moved
        section["launches"] += 1
        section["chunks"].append({"tensors": len(chunk), "rows": lists["rows"],
                                  "work": lists["work"], "grid": grid,
                                  "mean_ms": round(elapsed * 1000.0, 4),
                                  "gbps": round(moved / elapsed / 1e9, 3) if elapsed > 0 else None})
        for handle in [codes_buf, scales_buf, part_buf] + arrays:
            lib.clReleaseMemObject(handle)
    section.update({
        "weight_bytes": int(weight_bytes),
        "seconds": round(seconds, 6),
        "mean_ms": round(seconds * 1000.0, 4),
        "tokens_per_second": round(1.0 / seconds, 3) if seconds > 0 else None,
        "effective_gbps": round(weight_bytes / seconds / 1e9, 3) if seconds > 0 else None,
    })


def run_pack(pack_dir, repeats=DEFAULT_REPEATS, loader=None, opener=None, device=0,
             device_name=None, parts=None, persist=True):
    """Time every distinct ternary shape of a converted pack on the Intel GPU; never the CPU."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, pack_dir)
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
    report["repeats"] = repeats
    try:
        tensors, manifest, index = pack_tensors(pack_dir)
    except (OSError, ValueError, KeyError) as exc:
        return _fail(report, "cannot read the pack: %s" % exc)
    if not tensors:
        return _fail(report, "the pack has no ternary tensors (is it a Kindi pack?)")
    shapes = shapes_of(tensors)
    report["persistent"]["parts"] = 1
    report["pack"].update({
        "model": manifest.get("model"), "source_file": manifest.get("source_file"),
        "tensors": len(index["tensors"]), "ternary": len(tensors), "shapes": len(shapes),
    })
    if parts is None:
        parts = default_parts(max(s["cols"] for s in shapes))
    elif not isinstance(parts, int) or not 1 <= parts <= MAX_PARTS:
        return _fail(report, "--parts must be an integer in [1, %d]" % MAX_PARTS)
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    try:
        bcodes = np.fromfile(os.path.join(pack_dir, "bcodes.bin"), dtype=np.uint8)
        dpm16 = np.fromfile(os.path.join(pack_dir, "dpm.bin"), dtype=np.float16)
    except (OSError, ValueError) as exc:   # metadata without the bytes: a fail report, not a traceback
        return _fail(report, "cannot read the pack's byte files: %s" % exc)
    report["pack"]["bcodes_bytes"] = int(bcodes.nbytes)
    report["pack"]["dpm_bytes"] = int(dpm16.nbytes * 2)   # widened to float32 for the device

    lib, error = _open(loader, opener)
    if lib is None:
        return _fail(report, error or "OpenCL loader unavailable")
    report["loader"]["available"] = True
    try:
        _bind(lib)
    except (AttributeError, TypeError, ValueError) as exc:
        return _fail(report, "OpenCL loader is missing an API: %s" % exc)

    context = queue = program = kernel = reduce_kernel = None
    buffers = []
    try:
        devices, error = _enumerate(lib, report)
        if devices is None:
            return _fail(report, error or "OpenCL device enumeration failed")
        index_, info, error = pick_device(lib, devices, device, device_name)
        if error or index_ is None or info is None:
            return _fail(report, ("--device-name: " if device_name else "--device: ")
                         + (error or "the selected device could not be read"))
        platform, device_handle, platform_entry = devices[index_]
        report["platform"] = platform_entry["name"]
        report["device"] = info
        if "arc" not in (info.get("name") or "").lower():
            report["warnings"].append(
                "the selected device is %r, not an Arc: pass --device-name Arc for a run whose "
                "numbers may be quoted as Arc" % (info.get("name"),))

        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(device_handle)
        error_code = ctypes.c_int(0)
        context = lib.clCreateContext(properties, 1, handles, None, None, ctypes.byref(error_code))
        report["returncodes"]["clCreateContext"] = _hex(error_code.value)
        if not context or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateContext returned %s" % _hex(error_code.value))
        error_code = ctypes.c_int(0)
        queue = lib.clCreateCommandQueue(context, device_handle, 0, ctypes.byref(error_code))
        report["returncodes"]["clCreateCommandQueue"] = _hex(error_code.value)
        if not queue or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateCommandQueue returned %s" % _hex(error_code.value))

        source = SOURCE.encode("utf-8")
        lengths = ctypes.c_size_t(len(source))
        strings = (ctypes.c_char_p * 1)(source)
        error_code = ctypes.c_int(0)
        program = lib.clCreateProgramWithSource(context, 1, strings, ctypes.byref(lengths),
                                                ctypes.byref(error_code))
        report["returncodes"]["clCreateProgramWithSource"] = _hex(error_code.value)
        if not program or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateProgramWithSource returned %s" % _hex(error_code.value))
        devices_array = (ctypes.c_void_p * 1)(device_handle)
        code = int(lib.clBuildProgram(program, 1, devices_array, None, None, None))
        report["returncodes"]["clBuildProgram"] = _hex(code)
        log = _build_log(lib, program, device_handle)
        if code != CL_SUCCESS:
            first = (log or "").strip().splitlines()
            return _fail(report, "clBuildProgram returned %s%s" % (
                _hex(code), "" if not first else ": " + first[0][:200]))
        error_code = ctypes.c_int(0)
        kernel = lib.clCreateKernel(program, KERNEL.encode("ascii"), ctypes.byref(error_code))
        report["returncodes"]["clCreateKernel"] = _hex(error_code.value)
        if not kernel or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateKernel(%s) returned %s" % (KERNEL, _hex(error_code.value)))
        error_code = ctypes.c_int(0)
        reduce_kernel = lib.clCreateKernel(program, REDUCE.encode("ascii"), ctypes.byref(error_code))
        report["returncodes"]["clCreateKernelReduce"] = _hex(error_code.value)
        if not reduce_kernel or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateKernel(%s) returned %s" % (REDUCE, _hex(error_code.value)))

        x_max = max(s["cols"] for s in shapes)
        x = (((np.arange(x_max, dtype=np.int64) * 5 + 3) % 7 - 3) * 0.375
             + 0.125 * ((np.arange(x_max, dtype=np.int64) * 3) % 5)).astype(np.float32)
        x_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x,
                        report, "clCreateBufferX")
        buffers.append(x_buf)

        # one small tensor, parts=1: the real bytes against the host model, bit for bit
        small = min(tensors, key=lambda t: t["codes_bytes"])
        validation = report["validation"]
        if small["codes_bytes"] <= VALIDATE_MAX_BYTES:
            rows, cols = small["rows"], small["cols"]
            codes = bcodes[small["bc_off"]:small["bc_off"] + small["codes_bytes"]]
            expected = partials(codes, dpm_view(dpm16, small), x[:cols], rows, cols, 1)[0]
            codes_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes,
                                report, "clCreateBufferValidationCodes")
            scales_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 widened(dpm16, small), report, "clCreateBufferValidationScales")
            y_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                            report, "clCreateBufferValidation")
            part_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                               report, "clCreateBufferValidationPartial")
            _set_args(lib, kernel, codes_buf, scales_buf, x_buf, part_buf, rows, cols, 1, 0, 0)
            _set_reduce_args(lib, reduce_kernel, part_buf, y_buf, rows, 1)
            _launch_2d(lib, queue, kernel, (rows + 3) // 4, 1, report, "clEnqueueValidation")
            _launch_1d(lib, queue, reduce_kernel, rows, report, "clEnqueueValidationReduce")
            _finish(lib, queue, report, "clFinishValidation")
            raw = _read_back(lib, queue, y_buf, rows, report, "clReadValidation")
            actual = np.frombuffer(raw, dtype=np.float32)
            differs = actual != expected
            validation.update({"tensor": small["name"], "rows": rows, "cols": cols,
                               "mismatches": int(np.count_nonzero(differs))})
            for handle in (codes_buf, scales_buf, y_buf, part_buf):
                lib.clReleaseMemObject(handle)
            bad = int(validation["mismatches"] or 0)
            if bad:
                return _fail(report, "the pack's bytes do not decode exactly: %d of %d rows of %r "
                                     "differ from the host model" % (bad, rows, small["name"]))
        else:
            report["warnings"].append("the smallest ternary tensor is larger than %d MiB; the "
                                      "device result was not checked against the host"
                                      % (VALIDATE_MAX_BYTES >> 20))

        # A single bcodes buffer is over the driver's per-allocation limit on this card (the 27B's
        # tensor bytes are ~5.8 GB), so each shape uploads its own tensor's slice instead.
        seconds_by_shape = {}
        for shape in shapes:
            rows, cols = shape["rows"], shape["cols"]
            tensor = next(t for t in tensors if t["name"] == shape["first"])
            codes_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                bcodes[tensor["bc_off"]:tensor["bc_off"] + tensor["codes_bytes"]],
                                report, "clCreateBufferCodes_%dx%d" % (rows, cols))
            scales_buf = _buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 widened(dpm16, tensor), report,
                                 "clCreateBufferScales_%dx%d" % (rows, cols))
            y_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY, np.zeros(rows, dtype=np.float32),
                            report, "clCreateBufferY_%dx%d" % (rows, cols))
            part_buf = _buffer(lib, context, CL_MEM_WRITE_ONLY,
                               np.zeros(rows * parts, dtype=np.float32), report,
                               "clCreateBufferPartial_%dx%d" % (rows, cols))
            _set_args(lib, kernel, codes_buf, scales_buf, x_buf, part_buf, rows, cols, parts, 0, 0)
            _set_reduce_args(lib, reduce_kernel, part_buf, y_buf, rows, parts)
            _launch_2d(lib, queue, kernel, (rows + 3) // 4, parts, report, "clEnqueueWarmup")
            _launch_1d(lib, queue, reduce_kernel, rows, report, "clEnqueueWarmupReduce")
            _finish(lib, queue, report, "clFinishWarmup")
            started = time.perf_counter()
            for _ in range(repeats):
                _launch_2d(lib, queue, kernel, (rows + 3) // 4, parts, report, "clEnqueueKernel")
                _launch_1d(lib, queue, reduce_kernel, rows, report, "clEnqueueReduce")
            _finish(lib, queue, report, "clFinish")
            seconds = (time.perf_counter() - started) / repeats
            seconds_by_shape[(rows, cols)] = seconds
            report["shapes"].append({
                "rows": rows, "cols": cols, "tensors": shape["tensors"],
                "mean_ms": round(seconds * 1000.0, 4),
                "gbps": round(shape["bytes"] / seconds / 1e9, 3) if seconds > 0 else None,
            })
            for handle in (codes_buf, scales_buf, y_buf, part_buf):
                lib.clReleaseMemObject(handle)

        if persist:
            error_code = ctypes.c_int(0)
            persist_kernel = lib.clCreateKernel(program, PERSIST.encode("ascii"),
                                                ctypes.byref(error_code))
            report["returncodes"]["clCreateKernelPersist"] = _hex(error_code.value)
            if not persist_kernel or error_code.value != CL_SUCCESS:
                return _fail(report, "clCreateKernel(%s) returned %s"
                             % (PERSIST, _hex(error_code.value)))
            try:
                if validation["mismatches"] is not None:
                    _persist_validate(lib, context, queue, persist_kernel, x_buf, bcodes, dpm16,
                                      small, expected, report)
                    bad = report["persistent"]["validation"]["mismatches"]
                    if bad:
                        return _fail(report, "the persistent kernel does not reproduce the "
                                             "per-shape rows: %d of %d rows of %r differ"
                                     % (bad, small["rows"], small["name"]))
                _persist_pass(lib, context, queue, persist_kernel, x_buf, bcodes, dpm16, tensors,
                              repeats, report, chunk_bytes(report["device"]))
            finally:
                report["returncodes"]["clReleaseKernelPersist"] = _hex(
                    lib.clReleaseKernel(persist_kernel))

        report["projection"] = project(tensors, seconds_by_shape)
        report["projection"]["excluded"] = ("attention, norms, GDN state update, output-head f32 "
                                            "work, tokenizer and sampling")
        persistent = report["persistent"]
        if persistent["seconds"]:
            persistent["speedup"] = round(
                report["projection"]["seconds"] / persistent["seconds"], 3)
        report["result"] = "pass"
        return report
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return _fail(report, "OpenCL pack path failed: %s" % exc)
    finally:
        if reduce_kernel:
            report["returncodes"]["clReleaseKernelReduce"] = _hex(
                lib.clReleaseKernel(reduce_kernel))
        if kernel:
            report["returncodes"]["clReleaseKernel"] = _hex(lib.clReleaseKernel(kernel))
        if program:
            report["returncodes"]["clReleaseProgram"] = _hex(lib.clReleaseProgram(program))
        for index_, handle in enumerate(buffers):
            report["returncodes"]["clReleaseMemObject%d" % index_] = _hex(
                lib.clReleaseMemObject(handle))
        if queue:
            report["returncodes"]["clReleaseCommandQueue"] = _hex(lib.clReleaseCommandQueue(queue))
        if context:
            report["returncodes"]["clReleaseContext"] = _hex(lib.clReleaseContext(context))


def widened(dpm16, tensor):
    """The tensor's scales as float32 for the device: OpenCL C 1.2 has no portable half arithmetic."""
    return np.ascontiguousarray(dpm_view(dpm16, tensor).astype(np.float32).reshape(-1))


def dpm_view(dpm16, tensor):
    """The tensor's scales as the (phases, rows) fp16 array `partials` expects."""
    count = tensor["rows"] * tensor["phases"]
    begin = tensor["dpm_off"] // 2
    return np.asarray(dpm16[begin:begin + count], dtype=np.float16).reshape(
        tensor["phases"], tensor["rows"])


def _set_reduce_args(lib, kernel, partial, y, rows, parts):
    _set_arg(lib, kernel, 0, ctypes.c_void_p(int(partial)))
    _set_arg(lib, kernel, 1, ctypes.c_void_p(int(y)))
    _set_arg(lib, kernel, 2, ctypes.c_uint32(rows))
    _set_arg(lib, kernel, 3, ctypes.c_uint32(parts))


def _set_persist_args(lib, kernel, codes, scales, x, partial, arrays, nmats, total):
    for slot, handle in enumerate((codes, scales, x, partial)):
        _set_arg(lib, kernel, slot, ctypes.c_void_p(int(handle)))
    for slot, handle in enumerate(arrays):
        _set_arg(lib, kernel, 4 + slot, ctypes.c_void_p(int(handle)))
    _set_arg(lib, kernel, 4 + len(arrays), ctypes.c_uint32(nmats))
    _set_arg(lib, kernel, 5 + len(arrays), ctypes.c_uint32(total))


def _set_args(lib, kernel, codes, scales, x, partial, rows, cols, parts, coff, soff):
    _set_arg(lib, kernel, 0, ctypes.c_void_p(int(codes)))
    _set_arg(lib, kernel, 1, ctypes.c_void_p(int(scales)))
    _set_arg(lib, kernel, 2, ctypes.c_void_p(int(x)))
    _set_arg(lib, kernel, 3, ctypes.c_void_p(int(partial)))
    _set_arg(lib, kernel, 4, ctypes.c_uint32(rows))
    _set_arg(lib, kernel, 5, ctypes.c_uint32(cols))
    _set_arg(lib, kernel, 6, ctypes.c_uint32(parts))
    _set_arg(lib, kernel, 7, ctypes.c_uint64(coff))
    _set_arg(lib, kernel, 8, ctypes.c_uint64(soff))


def render(report):
    lines = ["OpenCL pack-reading GEMV probe (a converted pack's ternary shapes; no model run, "
             "engine or support claim)"]
    pack = report["pack"]
    lines.append("  pack: %s (%s); %s ternary of %s tensors, %s shapes, %s bcodes bytes" % (
        pack["model"], pack["source_file"], pack["ternary"], pack["tensors"], pack["shapes"],
        pack["bcodes_bytes"] or 0))
    if report["device"]:
        device = report["device"]
        lines.append("  device: %s; %s CU(s), %s MiB" % (
            device["name"], device["compute_units"], device["global_mem_mib"]))
    for shape in report["shapes"]:
        lines.append("  %5d x %5d x %2d tensor(s): %.4f ms each, %s GB/s" % (
            shape["rows"], shape["cols"], shape["tensors"], shape["mean_ms"], shape["gbps"]))
    projection = report["projection"]
    if projection["tokens_per_second"] is not None:
        lines.append("  projection: %s GB moved -> %.3f ms per pass, %s GB/s effective, "
                     "%s tokens/s ceiling" % (
                         projection["weight_bytes"] / 1e9, projection["seconds"] * 1000.0,
                         projection["effective_gbps"], projection["tokens_per_second"]))
    persistent = report["persistent"]
    if persistent["tokens_per_second"] is not None:
        lines.append("  persistent: %s GB moved -> %.3f ms per pass, %s GB/s effective, "
                     "%s tokens/s ceiling (%d launch(es) over %d row(s)/item, %sx the per-shape "
                     "path)" % (
                         persistent["weight_bytes"] / 1e9, persistent["seconds"] * 1000.0,
                         persistent["effective_gbps"], persistent["tokens_per_second"],
                         persistent["launches"], persistent["rows_per_item"],
                         persistent["speedup"]))
    if persistent["validation"]["mismatches"] is not None:
        lines.append("  persistent validation: %s (%d x %d) over one work list, mismatches %s" % (
            persistent["validation"]["tensor"], persistent["validation"]["rows"],
            persistent["validation"]["cols"], persistent["validation"]["mismatches"]))
    validation = report["validation"]
    if validation["mismatches"] is not None:
        lines.append("  validation: %s (%d x %d), mismatches %s" % (
            validation["tensor"], validation["rows"], validation["cols"],
            validation["mismatches"]))
    for warning in report["warnings"]:
        lines.append("  warning: %s" % warning)
    lines.append("  result: %s%s" % (
        report["result"], "" if not report["error"] else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
