"""Device-capability probe for the Intel Arc path (REQ-0020 M2 precursor).

This is a probe, not a backend: it enumerates the OpenCL platforms, picks one
Intel GPU (by index or by name substring, because the driver's device order is
not stable), and records the capability fields the Arc decode hypotheses in
``docs/ROADMAP.md`` rest on -- OpenCL C version, extension list, compute units,
clocks, global memory, per-allocation cap and local memory. It launches no
kernel, allocates no device buffer and loads no model, so it runs anywhere the
loader loads.

The verdicts are premises for the roadmap rows, not measurements of them: the
OpenCL C 1.2 profile exposes no DPAS/XMX subgroup path (that needs SYCL/ESIMD),
and a single 27B code buffer does not fit when the per-allocation cap is below
the pack's code bytes, so the step chunks codes on tensor boundaries.
"""

import ctypes
import json

from .intel_cl import (
    CL_DEVICE_GLOBAL_MEM_SIZE,
    CL_DEVICE_MAX_CLOCK_FREQUENCY,
    CL_DEVICE_MAX_COMPUTE_UNITS,
    CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_NAME,
    CL_DEVICE_OPENCL_C_VERSION,
    CL_DEVICE_VENDOR,
    CL_DEVICE_VERSION,
    CL_DRIVER_VERSION,
    DEFAULT_LOADER,
    _enumerate,
    _fail,
    _open,
    _query,
    _scalar,
    pick_device,
)

CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
CL_DEVICE_UUID_KHR = 0x105A
CL_DEVICE_UUID_SIZE = 16
CL_DEVICE_PCI_BUS_KHR = 0x410F

PROBE = "intel-caps"
SCHEMA = 1

# 27B pack code bytes (6.72 GiB of bcodes), from the 2026-11-26 27B intake:
# a single code buffer fits only when the per-allocation cap clears it.
CODES_27B_MIB = 6881


def _bind_info(lib):
    """Bind only the platform/device queries; no context, queue or kernel entry points."""
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


def _extensions(lib, device):
    text, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_EXTENSIONS)
    if not text:
        return []
    return [part for part in text.split() if part]


def _mib(value):
    return None if value is None else value // (1024 * 1024)


def _query_bytes(getter, handle, param, size):
    """Raw bytes, or None: a short write is a failed read, not a prefix."""
    buffer = ctypes.create_string_buffer(size)
    written = ctypes.c_size_t(0)
    code = int(getter(handle, param, size, buffer, ctypes.byref(written)))
    if code != 0 or int(written.value) != size:
        return None
    return buffer.raw[:size]


def _device_identity(lib, device, extensions):
    """Which hardware identity this device offers, and the raw values.

    The driver's device order is not stable (the iGPU and the Arc swap indices
    between runs), so a run must prove which card its numbers came from. The
    preference is the UUID, then the PCI bus id, then the (name, memory,
    driver) triple the report already carries. A missing extension, a short
    UUID write or an empty bus id is recorded as None, never as a guess: on
    this host's driver the UUID query succeeds yet writes only a stub, and the
    bus id comes back empty, so neither KHR extension identifies the card.
    """
    lowered = [ext.lower() for ext in extensions]
    uuid = pci_bus = None
    uuid_stub = pci_empty = False
    if "cl_khr_device_uuid" in lowered:
        raw = _query_bytes(lib.clGetDeviceInfo, device, CL_DEVICE_UUID_KHR,
                           CL_DEVICE_UUID_SIZE)
        if raw is None:
            uuid_stub = True
        else:
            uuid = raw.hex()
    if "cl_khr_pci_bus_info" in lowered:
        bus, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_PCI_BUS_KHR)
        if bus:
            pci_bus = bus
        else:
            pci_empty = True
    if uuid is not None:
        source = "uuid"
    elif pci_bus is not None:
        source = "pci"
    else:
        source = "name-memory"
    notes = []
    if uuid_stub:
        notes.append("cl_khr_device_uuid is advertised but the query does not "
                     "return 16 bytes on this device")
    if pci_empty:
        notes.append("cl_khr_pci_bus_info is advertised but the bus id is empty "
                     "on this device")
    if source == "name-memory":
        notes.append("identity is the (name, memory, driver) triple; neither KHR "
                     "identity extension is usable here")
    return uuid, pci_bus, source, notes


def _caps_report(lib, device):
    name, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_NAME)
    vendor, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_VENDOR)
    driver, _ = _query(lib.clGetDeviceInfo, device, CL_DRIVER_VERSION)
    version, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_VERSION)
    opencl_c, _ = _query(lib.clGetDeviceInfo, device, CL_DEVICE_OPENCL_C_VERSION)
    extensions = _extensions(lib, device)
    lowered = [ext.lower() for ext in extensions]
    uuid, pci_bus, identity, identity_notes = _device_identity(lib, device, extensions)
    global_mem = _scalar(lib, device, CL_DEVICE_GLOBAL_MEM_SIZE, ctypes.c_uint64)
    max_alloc = _scalar(lib, device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, ctypes.c_uint64)
    local_mem = _scalar(lib, device, CL_DEVICE_LOCAL_MEM_SIZE, ctypes.c_uint64)
    opencl_c_12 = opencl_c is not None and "1.2" in opencl_c
    has_fp16 = "cl_khr_fp16" in lowered
    has_subgroups = "cl_khr_subgroups" in lowered or "cl_intel_subgroups" in lowered
    max_alloc_mib = _mib(max_alloc)
    return {
        "name": name,
        "vendor": vendor,
        "uuid": uuid,
        "pci_bus": pci_bus,
        "driver_version": driver,
        "device_version": version,
        "opencl_c_version": opencl_c,
        "extensions": extensions,
        "compute_units": _scalar(lib, device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                 ctypes.c_uint32),
        "max_clock_mhz": _scalar(lib, device, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 ctypes.c_uint32),
        "global_mem_mib": _mib(global_mem),
        "max_alloc_mib": max_alloc_mib,
        "local_mem_kib": None if local_mem is None else local_mem // 1024,
        "max_work_group_size": _scalar(lib, device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                       ctypes.c_size_t),
        "premises": {
            # The roadmap I1/I7 rows ask about DPAS/popcount through this
            # profile: OpenCL C 1.2 exposes neither, so that work needs
            # SYCL/ESIMD, not a new OpenCL kernel.
            "opencl_c_1_2": opencl_c_12,
            "fp16_arithmetic": has_fp16,
            "subgroups": has_subgroups,
            "dpas_via_opencl": False,
            "dpas_note": ("OpenCL C 1.2 exposes no DPAS/XMX path; "
                          "mixed s8 x s2 work needs SYCL/ESIMD (roadmap I1)."),
            "single_27b_codes_fit": (None if max_alloc_mib is None
                                     else max_alloc_mib >= CODES_27B_MIB),
            "chunking_required": (None if max_alloc_mib is None
                                  else max_alloc_mib < CODES_27B_MIB),
            "device_identity": identity,
            "identity_notes": identity_notes,
        },
    }


def _base_report(loader):
    return {
        "probe": PROBE,
        "schema": SCHEMA,
        "read_only": True,
        "support_claim": False,
        "backend": "opencl capability query only; no kernel, model or engine",
        "scope": ("device caps behind the Arc decode hypotheses "
                  "(ROADMAP Intel A770 rows I1-I7)"),
        "loader": {"name": loader, "available": False, "error": None},
        "platforms": [],
        "device_count": None,
        "device_index": None,
        "device": None,
        "rebar": {"state": None,
                  "note": ("not exposed by sysfs or the Windows display stack; "
                           "capture it with `lspci -vv` (Linux) or Intel Arc "
                           "Control/GPU-Z (Windows)")},
        "warnings": [],
        "returncodes": {},
        "error": None,
        "result": "fail",
    }


def run_probe(loader=None, opener=None, device=0, device_name=None):
    """Query one Intel GPU's capability fields; never launches a kernel."""
    loader = loader or DEFAULT_LOADER
    report = _base_report(loader)
    if device_name is not None and not str(device_name).strip():
        return _fail(report, "--device-name must be a non-empty substring")
    if not isinstance(device, int) or device < 0:
        return _fail(report, "--device must be a non-negative integer")

    lib, error = _open(loader, opener)
    report["loader"] = {"name": loader, "available": lib is not None,
                        "error": error}
    if lib is None:
        return _fail(report, error)
    _bind_info(lib)

    devices, error = _enumerate(lib, report)
    if devices is None:
        return _fail(report, error)
    report["device_count"] = len(devices)
    index, _info, error = pick_device(lib, devices, index=device, name=device_name)
    if error is not None or index is None:
        return _fail(report, error or "no Intel GPU selected")
    _platform, handle, _entry = devices[index]
    report["device_index"] = index
    caps = _caps_report(lib, handle)
    report["device"] = caps
    selected = caps.get("name") or ""
    if "arc" not in selected.lower():
        report["warnings"].append(
            "selected device %r is not an Arc; numbers from it are not Arc numbers" % selected)
    profile = caps.get("opencl_c_version")
    if profile and "1.2" not in profile:
        report["warnings"].append(
            "OpenCL C profile is %r, not the 1.2 the roadmap premises assume" % profile)
    report["result"] = "pass"
    return report


def dumps(report):
    """The machine interface: the whole report as sorted JSON."""
    return json.dumps(report, indent=2, sort_keys=True)


def render(report):
    """Human-readable form of `run_probe()`; the JSON form is the interface."""
    lines = ["Intel capability probe (read-only; no kernel, model or support claim)"]
    device = report.get("device") or {}
    lines.append("device [%s]: %s" % (report.get("device_index"),
                                      device.get("name") or "unknown"))
    lines.append("  uuid=%s pci_bus=%s identity=%s" % (
        device.get("uuid") or "unknown", device.get("pci_bus") or "unknown",
        (device.get("premises") or {}).get("device_identity")))
    lines.append("  driver=%s opencl_c=%s" % (device.get("driver_version") or "unknown",
                                              device.get("opencl_c_version") or "unknown"))
    lines.append("  compute_units=%s clock_mhz=%s global_mem_mib=%s max_alloc_mib=%s" % (
        device.get("compute_units"), device.get("max_clock_mhz"),
        device.get("global_mem_mib"), device.get("max_alloc_mib")))
    lines.append("  local_mem_kib=%s max_work_group=%s extensions=%d" % (
        device.get("local_mem_kib"), device.get("max_work_group_size"),
        len(device.get("extensions") or [])))
    premises = device.get("premises") or {}
    lines.append("  premises: opencl_c_1_2=%s fp16=%s subgroups=%s dpas_via_opencl=%s" % (
        premises.get("opencl_c_1_2"), premises.get("fp16_arithmetic"),
        premises.get("subgroups"), premises.get("dpas_via_opencl")))
    lines.append("  27B single-codes fit=%s chunking=%s" % (
        premises.get("single_27b_codes_fit"), premises.get("chunking_required")))
    for warning in report.get("warnings") or []:
        lines.append("warning: %s" % warning)
    if report.get("error"):
        lines.append("error: %s" % report["error"])
    lines.append("result: %s" % report.get("result"))
    return "\n".join(lines)
