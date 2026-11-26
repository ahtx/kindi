"""OpenCL kernel smoke test for Intel Arc.

This is the kernel half of M2 in REQ-0020 / ``docs/INTEL-ARC.md``.  It compiles
two OpenCL C kernels with the driver's own runtime compiler (no oneAPI, icpx or
``libksycl.so`` is needed), launches them on the selected Intel GPU and checks
the results against a host reference:

* ``vecadd`` over a known integer-valued float32 pattern, requiring a bit-exact
  result;
* a square ``gemm`` whose inputs are small integers, so every partial sum is
  exactly representable in float32 and the device result must match bit for bit.

Both kernels use only global memory and no host computation is substituted for
a failed launch: a failed build or launch is reported as ``fail``.  This proves
device compute and exactness on the card; it does **not** provide a SYCL backend,
load a model, run the Kindi engine or justify an Intel support claim.
"""
import ctypes
import json
import os
import time
from array import array
from operator import mul

CL_SUCCESS = 0
CL_DEVICE_NOT_FOUND = -1
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_COPY_HOST_PTR = 1 << 5
CL_CONTEXT_PLATFORM = 0x1084
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_PROGRAM_BUILD_LOG = 0x1183
DEFAULT_LOADER = "OpenCL.dll" if os.name == "nt" else "libOpenCL.so.1"
DEFAULT_N = 1 << 20
MAX_N = 1 << 24
DEFAULT_DIM = 128
MIN_DIM = 8
MAX_DIM = 512
DEFAULT_REPEATS = 5
MAX_REPEATS = 50
LOCAL_SIZE = 64
SOURCE = """
__kernel void vecadd(__global const float *a, __global const float *b,
                     __global float *out, uint n) {
    uint i = (uint)get_global_id(0);
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__kernel void gemm(__global const float *a, __global const float *b,
                   __global float *c, uint dim) {
    uint gid = (uint)get_global_id(0);
    uint row = gid / dim;
    uint col = gid - row * dim;
    if (row >= dim) {
        return;
    }
    float acc = 0.0f;
    for (uint k = 0; k < dim; ++k) {
        acc += a[row * dim + k] * b[k * dim + col];
    }
    c[row * dim + col] = acc;
}
"""


def _hex(code):
    return "0x%08x" % (int(code) & 0xffffffff)


def _base_report(loader, n, dim, repeats):
    return {
        "probe": "intel-cl-smoke",
        "schema": 1,
        "read_only": False,
        "support_claim": False,
        "backend": "opencl runtime-compiled kernels only; no sycl, model or engine",
        "scope": "vector-add and square GEMM on an Intel GPU, checked against a host reference",
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "platform": None,
        "device_count": None,
        "device_index": None,
        "device": None,
        "build": {"ok": False, "source_bytes": len(SOURCE.encode("utf-8")), "log": None},
        "vector_add": {"n": n, "mismatches": None, "max_abs_error": None, "elapsed_ms": None},
        "gemm": {"dim": dim, "repeats": repeats, "mismatches": None, "max_abs_error": None,
                 "elapsed_ms": None, "gflops": None},
        "warnings": [],
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def _fail(report, message):
    report["error"] = message
    report["result"] = "fail"
    return report


def _open(loader, opener):
    open_library = opener or (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)
    try:
        return open_library(loader), None
    except (OSError, TypeError, AttributeError) as exc:
        return None, "could not load %s: %s" % (loader, exc)


def _bind(lib):
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
    lib.clCreateProgramWithSource.restype = ctypes.c_void_p
    lib.clCreateProgramWithSource.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                              ctypes.POINTER(ctypes.c_char_p),
                                              ctypes.POINTER(ctypes.c_size_t),
                                              ctypes.POINTER(ctypes.c_int)]
    lib.clBuildProgram.restype = ctypes.c_int
    lib.clBuildProgram.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                   ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                   ctypes.c_void_p, ctypes.c_void_p]
    lib.clGetProgramBuildInfo.restype = ctypes.c_int
    lib.clGetProgramBuildInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                          ctypes.c_size_t, ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_size_t)]
    lib.clReleaseProgram.restype = ctypes.c_int
    lib.clReleaseProgram.argtypes = [ctypes.c_void_p]
    lib.clCreateKernel.restype = ctypes.c_void_p
    lib.clCreateKernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                   ctypes.POINTER(ctypes.c_int)]
    lib.clReleaseKernel.restype = ctypes.c_int
    lib.clReleaseKernel.argtypes = [ctypes.c_void_p]
    lib.clCreateBuffer.restype = ctypes.c_void_p
    lib.clCreateBuffer.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t,
                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.clReleaseMemObject.restype = ctypes.c_int
    lib.clReleaseMemObject.argtypes = [ctypes.c_void_p]
    lib.clSetKernelArg.restype = ctypes.c_int
    lib.clSetKernelArg.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t,
                                   ctypes.c_void_p]
    lib.clEnqueueNDRangeKernel.restype = ctypes.c_int
    lib.clEnqueueNDRangeKernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                           ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t), ctypes.c_uint,
                                           ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.POINTER(ctypes.c_void_p)]
    lib.clEnqueueReadBuffer.restype = ctypes.c_int
    lib.clEnqueueReadBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                                        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_void_p)]
    lib.clFinish.restype = ctypes.c_int
    lib.clFinish.argtypes = [ctypes.c_void_p]


def _query(getter, handle, param):
    size = ctypes.c_size_t(0)
    code = int(getter(handle, param, 0, None, ctypes.byref(size)))
    if code != CL_SUCCESS:
        return None, code
    buffer = ctypes.create_string_buffer(max(int(size.value), 1))
    code = int(getter(handle, param, size.value, buffer, ctypes.byref(size)))
    if code != CL_SUCCESS:
        return None, code
    return buffer.raw[:size.value].split(b"\0", 1)[0].decode("utf-8", "replace"), None


def _scalar(lib, device, param, ctype):
    value = ctype(0)
    written = ctypes.c_size_t(0)
    code = int(lib.clGetDeviceInfo(device, param, ctypes.sizeof(value), ctypes.byref(value),
                                   ctypes.byref(written)))
    if code != CL_SUCCESS:
        return None
    return int(value.value)


def _enumerate(lib, report):
    """Every Intel GPU across every OpenCL platform, as (platform, device) pairs."""
    count = ctypes.c_uint32(0)
    code = int(lib.clGetPlatformIDs(0, None, ctypes.byref(count)))
    report["returncodes"]["clGetPlatformIDs"] = _hex(code)
    if code != CL_SUCCESS:
        return None, "clGetPlatformIDs returned %s" % _hex(code)
    if not count.value:
        return [], None
    platforms = (ctypes.c_void_p * count.value)()
    code = int(lib.clGetPlatformIDs(count.value, platforms, ctypes.byref(count)))
    report["returncodes"]["clGetPlatformIDsHandles"] = _hex(code)
    if code != CL_SUCCESS:
        return None, "clGetPlatformIDs(handles) returned %s" % _hex(code)

    found = []
    for index, platform in enumerate(platforms):
        name, _ = _query(lib.clGetPlatformInfo, platform, CL_PLATFORM_NAME)
        vendor, _ = _query(lib.clGetPlatformInfo, platform, CL_PLATFORM_VENDOR)
        version, _ = _query(lib.clGetPlatformInfo, platform, CL_PLATFORM_VERSION)
        entry = {"index": index, "name": name, "vendor": vendor, "version": version,
                 "gpu_devices": 0}
        n = ctypes.c_uint32(0)
        code = int(lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(n)))
        handles = []
        if code == CL_SUCCESS and n.value:
            handles = (ctypes.c_void_p * n.value)()
            code = int(lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n.value, handles,
                                          ctypes.byref(n)))
            if code != CL_SUCCESS:
                return None, "clGetDeviceIDs(handles) returned %s" % _hex(code)
        elif code not in (CL_SUCCESS, CL_DEVICE_NOT_FOUND):
            return None, "clGetDeviceIDs returned %s" % _hex(code)
        entry["gpu_devices"] = len(handles)
        report["platforms"].append(entry)
        for handle in handles:
            vendor_name, _ = _query(lib.clGetDeviceInfo, handle, CL_DEVICE_VENDOR)
            if "intel" in (vendor_name or "").lower():
                found.append((platform, handle, entry))
    return found, None


def _device_report(lib, device):
    name, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_NAME)
    if not name:
        return None
    vendor, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_VENDOR)
    driver, _ = _query(lib.clGetDeviceInfo, device, CL_DRIVER_VERSION)
    version, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_VERSION)
    opencl_c, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_OPENCL_C_VERSION)
    memory = _scalar(lib, device, CL_DEVICE_GLOBAL_MEM_SIZE, ctypes.c_uint64)
    max_alloc = _scalar(lib, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, ctypes.c_uint64)
    return {
        "name": name,
        "vendor": vendor,
        "driver_version": driver,
        "device_version": version,
        "opencl_c_version": opencl_c,
        "compute_units": _scalar(lib, device, CL_DEVICE_MAX_COMPUTE_UNITS, ctypes.c_uint32),
        "max_clock_mhz": _scalar(lib, device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 ctypes.c_uint32),
        "global_mem_mib": None if memory is None else memory // (1024 * 1024),
        "max_mem_alloc_mib": None if max_alloc is None else max_alloc // (1024 * 1024),
        "max_work_group_size": _scalar(lib, device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                       ctypes.c_size_t),
    }


def device_names(lib, devices):
    """[(index, CL_DEVICE_NAME)] for every enumerated Intel GPU, in the driver's order.

    The order is not stable: on the measured Windows host the integrated UHD 630 and the discrete
    Arc A770 have come back in both orders across runs, so a benchmark that selects by index alone
    can silently measure the wrong card. Probes that publish numbers should select by name.
    """
    out = []
    for index, (_platform, handle, _entry) in enumerate(devices):
        info = _device_report(lib, handle) or {}
        out.append((index, info.get("name")))
    return out


def pick_device(lib, devices, index=0, name=None):
    """(index, info, error): a device chosen by name (case-insensitive substring) or by index."""
    if not devices:
        return None, None, "no Intel OpenCL GPU device is visible"
    named = [(i, _device_report(lib, handle) or {})
             for i, (_platform, handle, _entry) in enumerate(devices)]
    if name:
        want = name.strip().lower()
        hits = [item for item in named if want in (item[1].get("name") or "").lower()]
        if not hits:
            return None, None, "no Intel GPU matches %r (visible: %s)" % (
                name, ", ".join(info.get("name") or "?" for _i, info in named))
        return hits[0][0], hits[0][1], None
    if not isinstance(index, int) or index < 0 or index >= len(named):
        return None, None, "device %s is outside the %d visible Intel GPU(s)" % (index, len(named))
    return index, named[index][1], None


def _build_log(lib, program, device):
    raw, code = _query_program(lib, program, device, CL_PROGRAM_BUILD_LOG)
    if code != CL_SUCCESS or not raw:
        return None
    return raw[:4000]


def _query_program(lib, program, device, param):
    size = ctypes.c_size_t(0)
    code = int(lib.clGetProgramBuildInfo(program, device, param, 0, None, ctypes.byref(size)))
    if code != CL_SUCCESS:
        return None, code
    buffer = ctypes.create_string_buffer(max(int(size.value), 1))
    code = int(lib.clGetProgramBuildInfo(program, device, param, size.value, buffer,
                                         ctypes.byref(size)))
    if code != CL_SUCCESS:
        return None, code
    return buffer.raw[:size.value].split(b"\0", 1)[0].decode("utf-8", "replace"), CL_SUCCESS


def _buffer(lib, context, flags, data, report, key):
    total = len(data) * 4
    # A host pointer is only legal together with COPY_HOST_PTR; otherwise the
    # driver rejects the buffer with CL_INVALID_HOST_PTR (-37).
    pointer = (ctypes.c_void_p(data.buffer_info()[0])
               if total and flags & CL_MEM_COPY_HOST_PTR else None)
    error = ctypes.c_int(0)
    handle = lib.clCreateBuffer(context, flags, ctypes.c_size_t(total), pointer,
                                ctypes.byref(error))
    report["returncodes"][key] = _hex(error.value)
    if not handle or error.value != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(error.value)))
    return handle


def _set_arg(lib, kernel, index, value):
    pointer = ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
    code = int(lib.clSetKernelArg(kernel, index, ctypes.sizeof(value), pointer))
    if code != CL_SUCCESS:
        raise ValueError("clSetKernelArg(%d) returned %s" % (index, _hex(code)))


def _round_up(total, local):
    return ((total + local - 1) // local) * local


def _launch(lib, queue, kernel, total, report, key):
    global_size = (ctypes.c_size_t * 1)(_round_up(total, LOCAL_SIZE))
    local_size = (ctypes.c_size_t * 1)(LOCAL_SIZE)
    code = int(lib.clEnqueueNDRangeKernel(queue, kernel, 1, None, global_size, local_size,
                                          0, None, None))
    report["returncodes"][key] = _hex(code)
    if code != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(code)))


def _finish(lib, queue, report, key):
    code = int(lib.clFinish(queue))
    report["returncodes"][key] = _hex(code)
    if code != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(code)))


def _read_back(lib, queue, handle, elements, report, key):
    total = elements * 4
    out = ctypes.create_string_buffer(total)
    code = int(lib.clEnqueueReadBuffer(queue, handle, 1, 0, total,
                                       ctypes.cast(out, ctypes.c_void_p), 0, None, None))
    report["returncodes"][key] = _hex(code)
    if code != CL_SUCCESS:
        raise ValueError("%s returned %s" % (key, _hex(code)))
    return out.raw


def _vecadd_inputs(n):
    a = array("f", ((i % 17) - 8 for i in range(n)))
    b = array("f", ((i % 13) - 6 for i in range(n)))
    reference = array("f", (a[i] + b[i] for i in range(n)))
    return a, b, reference


def _gemm_inputs(dim):
    a = array("f", (((r * 7 + k * 3) % 5) - 2 for r in range(dim) for k in range(dim)))
    b = array("f", (((k * 11 + c * 13) % 7) - 3 for k in range(dim) for c in range(dim)))
    return a, b, _reference_gemm(a, b, dim)


def _reference_gemm(a, b, dim):
    """Host reference; every input is a small integer, so the sum is exact."""
    columns = [b[c::dim] for c in range(dim)]
    out = array("f", [0.0]) * (dim * dim)
    for row in range(dim):
        values = a[row * dim:(row + 1) * dim]
        base = row * dim
        for column in range(dim):
            out[base + column] = sum(map(mul, values, columns[column]))
    return out


def _compare(packed, reference):
    actual = array("f")
    actual.frombytes(packed)
    if len(actual) != len(reference):
        return len(reference), None
    mismatches = 0
    worst = 0.0
    for got, want in zip(actual, reference, strict=True):
        if got != want:
            mismatches += 1
            worst = max(worst, abs(got - want))
    return mismatches, (worst if mismatches else 0.0)


def run_smoke(n=DEFAULT_N, dim=DEFAULT_DIM, repeats=DEFAULT_REPEATS, loader=None,
              opener=None, device=0, device_name=None):
    """Compile and run the two kernels on the selected Intel GPU; never on the CPU."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader, n, dim, repeats)
    if not isinstance(n, int) or not 1 <= n <= MAX_N:
        return _fail(report, "--n must be an integer in [1, %d]" % MAX_N)
    if not isinstance(dim, int) or not MIN_DIM <= dim <= MAX_DIM:
        return _fail(report, "--dim must be an integer in [%d, %d]" % (MIN_DIM, MAX_DIM))
    if not isinstance(repeats, int) or not 1 <= repeats <= MAX_REPEATS:
        return _fail(report, "--repeats must be an integer in [1, %d]" % MAX_REPEATS)
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

    context = None
    queue = None
    program = None
    kernels = []
    buffers = []
    try:
        devices, error = _enumerate(lib, report)
        if devices is None:
            return _fail(report, error or "OpenCL device enumeration failed")
        report["device_count"] = len(devices)
        index, info, error = pick_device(lib, devices, device, device_name)
        if error or index is None or info is None:
            return _fail(report, ("--device-name: " if device_name else "--device: ")
                         + (error or "the selected device could not be read"))
        platform, device_handle, platform_entry = devices[index]
        report["device_index"] = index
        report["platform"] = platform_entry["name"]
        report["device"] = info
        if "arc" not in (info.get("name") or "").lower():
            report["warnings"].append(
                "the selected device is %r, not an Arc: the driver's device order is not stable, "
                "so pass --device-name Arc for a run whose numbers may be reported as Arc"
                % (info.get("name"),))

        properties = (ctypes.c_void_p * 3)(CL_CONTEXT_PLATFORM, platform, None)
        handles = (ctypes.c_void_p * 1)(device_handle)
        error_code = ctypes.c_int(0)
        context = lib.clCreateContext(properties, 1, handles, None, None,
                                      ctypes.byref(error_code))
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
            return _fail(report, "clCreateProgramWithSource returned %s"
                         % _hex(error_code.value))

        devices_array = (ctypes.c_void_p * 1)(device_handle)
        code = int(lib.clBuildProgram(program, 1, devices_array, None, None, None))
        report["returncodes"]["clBuildProgram"] = _hex(code)
        report["build"]["log"] = _build_log(lib, program, device_handle)
        if code != CL_SUCCESS:
            first = (report["build"]["log"] or "").strip().splitlines()
            return _fail(report, "clBuildProgram returned %s%s" % (
                _hex(code), "" if not first else ": " + first[0][:200]))
        report["build"]["ok"] = True

        for name in (b"vecadd", b"gemm"):
            error_code = ctypes.c_int(0)
            kernel = lib.clCreateKernel(program, name, ctypes.byref(error_code))
            report["returncodes"]["clCreateKernel%s" % name.decode().title()] = _hex(
                error_code.value)
            if not kernel or error_code.value != CL_SUCCESS:
                return _fail(report, "clCreateKernel(%s) returned %s"
                             % (name.decode(), _hex(error_code.value)))
            kernels.append(kernel)

        a_data, b_data, vecadd_reference = _vecadd_inputs(n)
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               a_data, report, "clCreateBufferVecAddA"))
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               b_data, report, "clCreateBufferVecAddB"))
        buffers.append(_buffer(lib, context, CL_MEM_WRITE_ONLY, array("f", [0.0]) * n,
                               report, "clCreateBufferVecAddOut"))
        _set_arg(lib, kernels[0], 0, ctypes.c_void_p(int(buffers[0])))
        _set_arg(lib, kernels[0], 1, ctypes.c_void_p(int(buffers[1])))
        _set_arg(lib, kernels[0], 2, ctypes.c_void_p(int(buffers[2])))
        _set_arg(lib, kernels[0], 3, ctypes.c_uint32(n))
        started = time.perf_counter()
        _launch(lib, queue, kernels[0], n, report, "clEnqueueNDRangeKernelVecAdd")
        _finish(lib, queue, report, "clFinishVecAdd")
        report["vector_add"]["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        packed = _read_back(lib, queue, buffers[2], n, report, "clEnqueueReadBufferVecAdd")
        mismatches, worst = _compare(packed, vecadd_reference)
        report["vector_add"]["mismatches"] = mismatches
        report["vector_add"]["max_abs_error"] = worst
        if mismatches:
            return _fail(report, "vector-add mismatch: %d of %d elements differ"
                         % (mismatches, n))

        a_data, b_data, gemm_reference = _gemm_inputs(dim)
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               a_data, report, "clCreateBufferGemmA"))
        buffers.append(_buffer(lib, context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               b_data, report, "clCreateBufferGemmB"))
        buffers.append(_buffer(lib, context, CL_MEM_WRITE_ONLY, array("f", [0.0]) * (dim * dim),
                               report, "clCreateBufferGemmOut"))
        _set_arg(lib, kernels[1], 0, ctypes.c_void_p(int(buffers[3])))
        _set_arg(lib, kernels[1], 1, ctypes.c_void_p(int(buffers[4])))
        _set_arg(lib, kernels[1], 2, ctypes.c_void_p(int(buffers[5])))
        _set_arg(lib, kernels[1], 3, ctypes.c_uint32(dim))
        started = time.perf_counter()
        for _ in range(repeats):
            _launch(lib, queue, kernels[1], dim * dim, report, "clEnqueueNDRangeKernelGemm")
        _finish(lib, queue, report, "clFinishGemm")
        seconds = time.perf_counter() - started
        report["gemm"]["elapsed_ms"] = round(seconds * 1000.0, 3)
        if seconds > 0:
            report["gemm"]["gflops"] = round(2.0 * dim ** 3 * repeats / (seconds * 1e9), 2)
        packed = _read_back(lib, queue, buffers[5], dim * dim, report,
                            "clEnqueueReadBufferGemm")
        mismatches, worst = _compare(packed, gemm_reference)
        report["gemm"]["mismatches"] = mismatches
        report["gemm"]["max_abs_error"] = worst
        if mismatches:
            return _fail(report, "GEMM mismatch: %d of %d elements differ"
                         % (mismatches, dim * dim))

        report["result"] = "pass"
        return report
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        return _fail(report, "OpenCL kernel path failed: %s" % exc)
    finally:
        for index, kernel in enumerate(kernels):
            report["returncodes"]["clReleaseKernel%d" % index] = _hex(
                lib.clReleaseKernel(kernel))
        if program:
            report["returncodes"]["clReleaseProgram"] = _hex(lib.clReleaseProgram(program))
        for index, handle in enumerate(buffers):
            report["returncodes"]["clReleaseMemObject%d" % index] = _hex(
                lib.clReleaseMemObject(handle))
        if queue:
            report["returncodes"]["clReleaseCommandQueue"] = _hex(
                lib.clReleaseCommandQueue(queue))
        if context:
            report["returncodes"]["clReleaseContext"] = _hex(lib.clReleaseContext(context))


def render(report):
    """Human-readable form of the kernel smoke report."""
    loader = report["loader"]
    lines = ["OpenCL kernel smoke (vector-add + GEMM; no SYCL, model or support claim)",
             "  loader: %s%s" % (loader["name"], "" if loader["available"] else " (unavailable)")]
    if report["device_count"] is not None:
        lines.append("  Intel GPUs: %s; selected: %s" % (report["device_count"],
                                                        report["device_index"]))
    if report["device"]:
        device = report["device"]
        lines.append("  device: %s; %s CU(s), %s MiB, %s MHz, OpenCL C %s" % (
            device["name"], device["compute_units"], device["global_mem_mib"],
            device["max_clock_mhz"], device["opencl_c_version"]))
    if report["build"]["ok"]:
        lines.append("  build: ok (%d source bytes)" % report["build"]["source_bytes"])
    vector_add = report["vector_add"]
    if vector_add["mismatches"] is not None:
        lines.append("  vector-add: n=%d; mismatches: %s; %.3f ms" % (
            vector_add["n"], vector_add["mismatches"], vector_add["elapsed_ms"]))
    gemm = report["gemm"]
    if gemm["mismatches"] is not None:
        lines.append("  gemm: dim=%d x %d; mismatches: %s; %.3f ms; %s GFLOP/s" % (
            gemm["dim"], gemm["repeats"], gemm["mismatches"], gemm["elapsed_ms"],
            gemm["gflops"]))
    lines.append("  result: %s%s" % (
        report["result"], "" if not report["error"] else " (%s)" % report["error"]))
    return lines


def dumps(report):
    return json.dumps(report, indent=2, sort_keys=True)
