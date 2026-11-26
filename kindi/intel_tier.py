"""A770-as-cold-tier gate probe for MoE experts (ranked follow-up A3).

The NVMe file-backed cold-expert tier (`kindi/cold_experts.py`) pages overflow
MoE experts from disk; this host also has an Arc A770 with 16 GiB that could
stage them at PCIe rates instead. This probe measures the premise before any
tier is wired: sustained device-to-host copy bandwidth for expert-sized chunks
against a rough in-process disk baseline on the same host, with byte-exact
verification that the copies moved real bytes.

It allocates one device buffer, writes a pattern once, and times blocking
reads; no kernel, model, engine or support claim. The gate (device at least
2x the disk baseline with the chunk fitting the per-allocation cap) decides
whether wiring an A770-resident cold tier is worth building, not that it wins.
"""

import ctypes
import json
import os
import statistics
import tempfile
import time

from .intel_cl import (
    CL_CONTEXT_PLATFORM,
    CL_SUCCESS,
    DEFAULT_LOADER,
    _enumerate,
    _fail,
    _hex,
    _open,
    _query,
    _scalar,
    pick_device,
)

PROBE = "intel-tier"
SCHEMA = 1

CL_MEM_READ_WRITE = 0x1  # (1 << 0); WO is (1 << 1), RO is (1 << 2) per intel_cl
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_TRUE = 1

DEFAULT_MIB = 64
MIN_MIB = 1
MAX_MIB = 1024                # a chunk probe, not a VRAM fill; the cap is queried, not tested
DEFAULT_REPEATS = 5
MAX_REPEATS = 20

# The gate: wiring an A770-resident cold tier pays only if host-visible VRAM
# reads beat paging the same bytes from disk by a clear margin.
GATE_RATIO = 2.0


def _bind_tier(lib):
    """Bind only the query + context/queue/buffer/copy entry points this probe uses."""
    lib.clGetPlatformIDs.restype = ctypes.c_int
    lib.clGetPlatformIDs.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_uint)]
    lib.clGetPlatformInfo.restype = ctypes.c_int
    lib.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.clGetDeviceIDs.restype = ctypes.c_int
    lib.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint,
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.POINTER(ctypes.c_uint)]
    lib.clGetDeviceInfo.restype = ctypes.c_int
    lib.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.clCreateContext.restype = ctypes.c_void_p
    lib.clCreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
                                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.clReleaseContext.restype = ctypes.c_int
    lib.clReleaseContext.argtypes = [ctypes.c_void_p]
    lib.clCreateCommandQueue.restype = ctypes.c_void_p
    lib.clCreateCommandQueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_int)]
    lib.clReleaseCommandQueue.restype = ctypes.c_int
    lib.clReleaseCommandQueue.argtypes = [ctypes.c_void_p]
    lib.clCreateBuffer.restype = ctypes.c_void_p
    lib.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t,
                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.clReleaseMemObject.restype = ctypes.c_int
    lib.clReleaseMemObject.argtypes = [ctypes.c_void_p]
    lib.clEnqueueWriteBuffer.restype = ctypes.c_int
    lib.clEnqueueWriteBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                         ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                         ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                         ctypes.POINTER(ctypes.c_void_p)]
    lib.clEnqueueReadBuffer.restype = ctypes.c_int
    lib.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_void_p)]
    lib.clFinish.restype = ctypes.c_int
    lib.clFinish.argtypes = [ctypes.c_void_p]


def _pattern(total):
    """A deterministic non-zero pattern with C-speed construction."""
    unit = bytes(range(256))
    return (unit * (total // 256 + 1))[:total]


def _disk_baseline(total, repeats, disk_dir):
    """Rough in-process baseline: write once, time whole-file reads.

    This is not `kindi io`: same pattern, same chunk, buffered by the OS,
    reported only so the gate has a same-host denominator. No path is
    recorded in the report.
    """
    fd, path = tempfile.mkstemp(prefix="kindi-tier-", dir=disk_dir or None)
    try:
        payload = _pattern(total)
        offset = 0
        while offset < total:
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write on the disk baseline file")
            offset += written
        try:
            os.fsync(fd)
        except OSError:
            pass
        elapsed = []
        for _ in range(repeats):
            os.lseek(fd, 0, os.SEEK_SET)
            started = time.perf_counter()
            got = 0
            while got < total:
                chunk = os.read(fd, min(total - got, 1 << 20))
                if not chunk:
                    raise OSError("short read on the disk baseline file")
                got += len(chunk)
            elapsed.append(time.perf_counter() - started)
        return statistics.median(elapsed), payload
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(path)
        except OSError:
            pass


def _gate(device_mibs, disk_mibs, chunk_fits):
    ratio = None
    if disk_mibs and disk_mibs > 0 and device_mibs is not None:
        ratio = device_mibs / disk_mibs
    go = bool(chunk_fits) and ratio is not None and ratio >= GATE_RATIO
    return {
        "threshold": GATE_RATIO,
        "device_mibs": device_mibs,
        "disk_mibs": disk_mibs,
        "ratio": None if ratio is None else round(ratio, 3),
        "chunk_fits_alloc_cap": chunk_fits,
        "go": go,
        "note": ("GO wires an A770-resident cold-expert tier; NO-GO keeps the "
                 "NVMe file-backed tier and closes A3 as measured."),
    }


def _base_report(mib, repeats):
    return {
        "probe": PROBE,
        "schema": SCHEMA,
        "read_only": False,
        "support_claim": False,
        "backend": "opencl copy path only; no kernel, model, engine or tier",
        "scope": ("sustained device-to-host copy rate for expert-sized chunks "
                  "vs a same-host disk baseline (MoE cold-tier gate A3)"),
        "loader": {"name": "", "available": False, "error": None},
        "platforms": [],
        "device_count": None,
        "device_index": None,
        "device": None,
        "copy": {"mib": mib, "repeats": repeats, "median_ms": None,
                 "mibs": None, "mismatches": None},
        "disk": {"mib": mib, "repeats": repeats, "median_ms": None, "mibs": None},
        "gate": _gate(None, None, None),
        "warnings": [],
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def run_probe(mib=DEFAULT_MIB, repeats=DEFAULT_REPEATS, loader=None, opener=None,
              device=0, device_name=None, disk_dir=None):
    """Time device-to-host copies of one `mib` buffer; verify every byte."""
    report = _base_report(mib, repeats)
    if not isinstance(mib, int) or not MIN_MIB <= mib <= MAX_MIB:
        return _fail(report, "--mib must be an integer in [%d, %d]" % (MIN_MIB, MAX_MIB))
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
    if device_name is not None and not str(device_name).strip():
        return _fail(report, "--device-name must be a non-empty substring")
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    loader = loader or DEFAULT_LOADER
    report["loader"] = {"name": loader, "available": False, "error": None}
    lib, error = _open(loader, opener)
    if lib is None:
        report["loader"]["error"] = error
        return _fail(report, error)
    report["loader"]["available"] = True
    try:
        _bind_tier(lib)
    except (AttributeError, TypeError, ValueError) as exc:
        return _fail(report, "OpenCL loader is missing an API: %s" % exc)

    total = mib * 1024 * 1024
    context = None
    queue = None
    buffer = None
    try:
        devices, error = _enumerate(lib, report)
        if devices is None:
            return _fail(report, error or "OpenCL device enumeration failed")
        report["device_count"] = len(devices)
        index, _info, error = pick_device(lib, devices, index=device, name=device_name)
        if error is not None or index is None:
            return _fail(report, error or "no Intel GPU selected")
        platform, handle, _entry = devices[index]
        report["device_index"] = index
        name, _ = _query(lib.clGetDeviceInfo, handle, 0x102B)
        driver, _ = _query(lib.clGetDeviceInfo, handle, 0x102D)
        max_alloc = _scalar(lib, handle, CL_DEVICE_MAX_MEM_ALLOC_SIZE, ctypes.c_uint64)
        max_alloc_mib = None if max_alloc is None else max_alloc // (1024 * 1024)
        report["device"] = {"name": name, "driver_version": driver,
                            "max_alloc_mib": max_alloc_mib}
        if name and "arc" not in name.lower():
            report["warnings"].append(
                "selected device %r is not an Arc; numbers from it are not Arc numbers" % name)
        chunk_fits = None if max_alloc is None else total <= max_alloc
        if chunk_fits is False:
            report["gate"] = _gate(None, None, False)
            return _fail(report, "chunk %d MiB exceeds the per-allocation cap (%s MiB)" % (
                mib, max_alloc_mib))

        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(handle)
        error_code = ctypes.c_int(0)
        context = lib.clCreateContext(properties, 1, handles, None, None,
                                      ctypes.byref(error_code))
        report["returncodes"]["clCreateContext"] = _hex(error_code.value)
        if not context or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateContext returned %s" % _hex(error_code.value))

        error_code = ctypes.c_int(0)
        queue = lib.clCreateCommandQueue(context, handle, 0, ctypes.byref(error_code))
        report["returncodes"]["clCreateCommandQueue"] = _hex(error_code.value)
        if not queue or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateCommandQueue returned %s" % _hex(error_code.value))

        payload = _pattern(total)
        source = ctypes.create_string_buffer(payload)
        error_code = ctypes.c_int(0)
        buffer = lib.clCreateBuffer(context, CL_MEM_READ_WRITE, ctypes.c_size_t(total),
                                    None, ctypes.byref(error_code))
        report["returncodes"]["clCreateBuffer"] = _hex(error_code.value)
        if not buffer or error_code.value != CL_SUCCESS:
            return _fail(report, "clCreateBuffer returned %s" % _hex(error_code.value))

        code = int(lib.clEnqueueWriteBuffer(
            queue, buffer, CL_TRUE, 0, total, ctypes.cast(source, ctypes.c_void_p),
            0, None, None))
        report["returncodes"]["clEnqueueWriteBuffer"] = _hex(code)
        if code != CL_SUCCESS:
            return _fail(report, "clEnqueueWriteBuffer returned %s" % _hex(code))

        elapsed = []
        mismatches = 0
        for _ in range(repeats):
            out = ctypes.create_string_buffer(total)
            started = time.perf_counter()
            code = int(lib.clEnqueueReadBuffer(
                queue, buffer, CL_TRUE, 0, total, ctypes.cast(out, ctypes.c_void_p),
                0, None, None))
            elapsed.append(time.perf_counter() - started)
            if code != CL_SUCCESS:
                return _fail(report, "clEnqueueReadBuffer returned %s" % _hex(code))
            if out.raw != payload:
                mismatches += 1
        report["returncodes"]["clEnqueueReadBuffer"] = _hex(CL_SUCCESS)
        median = statistics.median(elapsed)
        report["copy"]["median_ms"] = round(median * 1000.0, 3)
        report["copy"]["mibs"] = round(mib / median, 2) if median > 0 else None
        report["copy"]["mismatches"] = mismatches
        if mismatches:
            return _fail(report, "copy mismatch: %d of %d reads differ" % (mismatches, repeats))

        try:
            disk_median, _ = _disk_baseline(total, repeats, disk_dir)
        except OSError as exc:
            return _fail(report, "disk baseline failed: %s" % exc)
        report["disk"]["median_ms"] = round(disk_median * 1000.0, 3)
        report["disk"]["mibs"] = round(mib / disk_median, 2) if disk_median > 0 else None
        report["gate"] = _gate(report["copy"]["mibs"], report["disk"]["mibs"], chunk_fits)
        report["result"] = "pass"
        return report
    finally:
        for release, handle_value, key in (
                (getattr(lib, "clReleaseMemObject", None), buffer, "clReleaseMemObject"),
                (getattr(lib, "clReleaseCommandQueue", None), queue, "clReleaseCommandQueue"),
                (getattr(lib, "clReleaseContext", None), context, "clReleaseContext")):
            if release is not None and handle_value:
                try:
                    report["returncodes"][key] = _hex(int(release(handle_value)))
                except (OSError, ValueError, TypeError, AttributeError):
                    pass


def dumps(report):
    """The machine interface: the whole report as sorted JSON."""
    return json.dumps(report, indent=2, sort_keys=True)


def render(report):
    """Human-readable form of `run_probe()`; the JSON form is the interface."""
    lines = ["Intel cold-tier gate probe (copies only; no kernel, model or support claim)"]
    device = report.get("device") or {}
    lines.append("device [%s]: %s (alloc cap %s MiB)" % (
        report.get("device_index"), device.get("name") or "unknown",
        device.get("max_alloc_mib")))
    copy = report.get("copy") or {}
    lines.append("device->host: %s MiB in %s ms (%s MiB/s, mismatches %s)" % (
        copy.get("mib"), copy.get("median_ms"), copy.get("mibs"), copy.get("mismatches")))
    disk = report.get("disk") or {}
    lines.append("disk baseline: %s MiB in %s ms (%s MiB/s)" % (
        disk.get("mib"), disk.get("median_ms"), disk.get("mibs")))
    gate = report.get("gate") or {}
    lines.append("gate: ratio %s vs >= %s -> %s" % (
        gate.get("ratio"), gate.get("threshold"), "GO" if gate.get("go") else "NO-GO"))
    for warning in report.get("warnings") or []:
        lines.append("warning: %s" % warning)
    if report.get("error"):
        lines.append("error: %s" % report["error"])
    lines.append("result: %s" % report.get("result"))
    return "\n".join(lines)
