"""Intel integrated graphics: discovery, eligibility and the shared-memory budget.

This is the G0 (hardware-qualification) slice of REQ-0021 / docs/INTEL-IGPU-TANDEM.md:
pure Python, no oneAPI dependency, so it runs and is testable on a CPU-only host.
It answers three questions without a SYCL kernel in sight:

- ``discover()``: which Intel GPUs are visible (Linux ``/dev/dri`` + sysfs)?
- ``eligible(dev)``: is the device at or above Intel's oneAPI floor
  (UHD for 11th generation or newer / Iris Xe or newer / built-in Arc)?
- ``shared_budget_mb()``: how much host RAM may the iGPU use (the fit input
  that replaces VRAM in the NVIDIA/AMD path)?

There is deliberately no engine here: the SYCL backend (REQ-0020) is the kernel
half, and it needs hardware to verify. ``KINDI_DEVICE=intel`` refuses clearly
from ``kindi/engine.py`` until that backend exists, instead of silently
falling back to the CPU engine.

``discover()`` reads Linux DRM sysfs.  On Windows the display adapters come
from CIM plus the display class registry (the only place a true 16 GiB VRAM
shows up), and ``probe()`` additionally records the Level Zero loader, the
driver the display class registers, and the driver/device counts from a
read-only ``zeInit``/``zeDriverGet``/``zeDeviceGet`` enumeration.  That
enumeration fills no device structure: a guessed struct offset is not
evidence.
"""
import ctypes
import glob
import json
import os
import re
import subprocess
import sys

# PCI device-id -> (display name, architecture). Conservative by design: only
# ids whose architecture is known to sit at or above the oneAPI floor are
# listed; anything else is "unknown" and refused until measured (the same
# rule as docs/INTEL-IGPU-TANDEM.md: pre-11th-gen, e.g. UHD 630 / Gen9, is out).
# Extend this table only with a measured identity: sysfs/lspci on Linux, CIM and
# the display class registry on Windows.
DEVICE_ARCH = {
    # DG2 / Alchemist. Measured on the A770 host (2026-11-22): display driver
    # 32.0.101.9175 reports "Intel(R) Arc(TM) A770 Graphics" with 16 GiB, and the
    # Level Zero driver registered by that display class enumerates one device.
    0x56A0: ("Intel Arc A770", "Xe-HPG"),
    # Xe-LP (Gen12): 11th-13th gen Core UHD / Iris Xe
    0x9A49: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9A51: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9A71: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9AF0: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x9BCA: ("Intel UHD Graphics 770", "Xe-LP"),
    0x9BC8: ("Intel UHD Graphics 730", "Xe-LP"),
    0x46A8: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x46A6: ("Intel Iris Xe Graphics", "Xe-LP"),
    0x46B0: ("Intel UHD Graphics 750", "Xe-LP"),
    0x46A2: ("Intel UHD Graphics 710", "Xe-LP"),
    # Xe2-LPG: built-in Arc in Core Ultra (Meteor Lake / Lunar Lake / Panther Lake)
    0x7D55: ("Intel Arc 140V (built-in)", "Xe2-LPG"),
    0x7D67: ("Intel Arc 130V (built-in)", "Xe2-LPG"),
    0x7D51: ("Intel Arc 140T (built-in)", "Xe2-LPG"),
}

# Identification only: these names do not enter `arch_of()`/`eligible()`, so a
# read-only probe can name an unmeasured card without turning that into a
# support claim. Extend this table when lspci/sysfs identity is known, and the
# measured architecture table above only after a sycl-ls/Level Zero identity.
DEVICE_NAMES = {
    # DG2 / Arc A770 family; the PCI id does not expose the card's memory size.
    0x56A0: "Intel Arc A770",
}

SUPPORTED_ARCH = frozenset(("Xe-LP", "Xe2-LPG", "Xe-HPG"))

# Hybrid kernel drivers that can drive a supported iGPU on Linux.
GPU_DRIVERS = frozenset(("i915", "xe"))

_SYS_DRM = "/sys/class/drm"
_DEV_DRI = "/dev/dri"
_SYS_ACCEL = "/sys/class/accel"
_CPUINFO = "/proc/cpuinfo"
_ONEAPI_ROOT = "/opt/intel/oneapi"

CPU_FEATURES = (
    ("avx2", ("avx2",)),
    ("fma", ("fma", "fma3")),
    ("avx_vnni", ("avx_vnni", "avx_vnni_int8", "avx_vnni_int16")),
    ("amx", ("amx", "amx_tile", "amx_bf16", "amx_int8", "amx_fp16")),
)

_BDF = re.compile(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
_SYCL_DEVICE = re.compile(
    r"^\s*\[(?P<backend>[^:\]]+):(?P<kind>[^:\]]+):(?P<index>\d+)\]\s*(?P<desc>.+?)\s*$")
_OPENVINO_CODE = ("import json, openvino as ov; print(json.dumps({"
                  "'version': getattr(ov, '__version__', None), "
                  "'devices': list(ov.Core().available_devices)}))")

# Windows discovery.  These are read-only CIM/registry queries, and the Level
# Zero scripts only read the display class registration; none of them starts a
# model, allocates device memory or launches a kernel.
_PS = ("powershell", "-NoProfile", "-NonInteractive", "-Command")
_WIN_CLASS_GUID = "{4d36e968-e325-11ce-bfc1-08002be10318}"
_WIN_CLASS = "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\" + _WIN_CLASS_GUID
_WIN_L0_LOADER = "ze_loader.dll"
_WIN_VIDEO = ("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + "
              "$_.DriverVersion + '|' + $_.PNPDeviceID + '|' + $_.Status }")
_WIN_VRAM = ("0..15 | ForEach-Object { $p = '" + _WIN_CLASS + "\\' + ('{0:d4}' -f $_); "
             "$v = Get-ItemProperty $p -ErrorAction SilentlyContinue; "
             "if ($v.'HardwareInformation.qwMemorySize') { $v.DriverDesc + '|' + "
             "$v.'HardwareInformation.qwMemorySize' } }")
# No `Family` here: Win32_Processor.Family is a Win32 processor-family enum, not
# the CPUID family number /proc/cpuinfo reports, so it would mislabel the CPU.
_WIN_CPU = ("Get-CimInstance Win32_Processor | ForEach-Object { $_.Manufacturer + '|' + "
            "$_.Name + '|' + $_.NumberOfCores + '|' + $_.NumberOfLogicalProcessors }")
_WIN_L0_REGISTER = ("0..15 | ForEach-Object { $p = '" + _WIN_CLASS + "\\' + ('{0:d4}' -f $_); "
                    "$v = Get-ItemProperty $p -ErrorAction SilentlyContinue; "
                    "if ($v.LevelZeroDriverPath) { $v.DriverDesc + '|' + $v.LevelZeroDriverPath + '|' + "
                    "(Get-Item $v.LevelZeroDriverPath).VersionInfo.FileVersion + '|' + "
                    "$v.OpenCLDriverName } }")
_PNP_IDS = re.compile(r"VEN_([0-9A-Fa-f]{4})&DEV_([0-9A-Fa-f]{4})")


def _int(value, base=10):
    try:
        return int(str(value).strip(), base)
    except (TypeError, ValueError):
        return None


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (IOError, OSError):
        return ""


def arch_of(device_id):
    """(display name, architecture) for a PCI device id, or (None, None)."""
    return DEVICE_ARCH.get(int(device_id), (None, None))


def _bdf_from_card(card):
    """The PCI BDF behind a DRM card, or "" when sysfs does not expose one."""
    real = os.path.realpath(os.path.join(card, "device"))
    name = os.path.basename(real)
    return name if _BDF.fullmatch(name) else ""


def _integrated(arch, name):
    """Whether the known device is integrated; None means it is not established."""
    lowered = (name or "").lower()
    if "built-in" in lowered or "iris" in lowered or "uhd" in lowered:
        return True
    if arch == "Xe2-LPG" or arch == "Xe-LP":
        return True
    if arch == "Xe-HPG" or "arc a" in lowered or "dg2" in lowered:
        return False
    return None


def _render_for_bdf(dri_root, bdf):
    """The render node named by /dev/dri/by-path, or "" when the mapping is absent."""
    if not bdf:
        return ""
    entry = os.path.join(dri_root, "by-path", "pci-%s-render" % bdf)
    if not (os.path.islink(entry) or os.path.exists(entry)):
        return ""
    target = os.path.realpath(entry)
    return target if os.path.basename(target).startswith("renderD") else ""


def discover(drm_root=_SYS_DRM, dri_root=_DEV_DRI, glob_fn=glob.glob):
    """List visible Intel GPUs from DRM sysfs. Returns [] where unsupported.

    Each entry is a dict: render_node, pci_id, bdf, vendor_id, device_id, name,
    arch, driver, integrated and source.  On hosts without /dev/dri or sysfs
    (e.g. Windows), this honestly returns [].
    """
    found = []
    if not os.path.isdir(drm_root):
        return found
    for card in sorted(glob_fn(os.path.join(drm_root, "card[0-9]*"))):
        vendor = _read(os.path.join(card, "device", "vendor"))
        if vendor.lower() != "0x8086":
            continue
        device = _read(os.path.join(card, "device", "device"))
        try:
            device_id = int(device, 16)
        except ValueError:
            continue
        uevent = _read(os.path.join(card, "device", "uevent"))
        driver = ""
        for line in uevent.splitlines():
            if line.startswith("DRIVER="):
                driver = line.split("=", 1)[1].strip()
        name, arch = arch_of(device_id)
        name = name or DEVICE_NAMES.get(device_id)
        bdf = _bdf_from_card(card)
        render = _render_for_bdf(dri_root, bdf)
        if not render:
            nodes = sorted(glob_fn(os.path.join(dri_root, "renderD*")))
            # renderD128+ are compute-capable; match by card index order.
            card_index = os.path.basename(card)[4:]
            try:
                idx = int(card_index)
                if 0 <= idx < len(nodes):
                    render = nodes[idx]
            except ValueError:
                pass
        found.append({
            "render_node": render,
            "pci_id": os.path.basename(card),
            "bdf": bdf,
            "vendor_id": 0x8086,
            "device_id": device_id,
            "name": name,
            "arch": arch,
            "driver": driver,
            "integrated": _integrated(arch, name),
            "source": "sysfs",
        })
    return found


def _feature_map(flags, seen):
    """The four M1 CPU capability flags; None when the OS exposes no flags."""
    lowered = {flag.lower() for flag in flags}
    return {name: (None if not seen else any(alias in lowered for alias in aliases))
            for name, aliases in CPU_FEATURES}


def parse_cpuinfo(text):
    """CPU identity and M1 features from /proc/cpuinfo (or an empty report)."""
    blocks, current = [], {}
    for line in (text or "").splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        key, sep, value = line.partition(":")
        if sep:
            current[key.strip().lower()] = value.strip()
    if current:
        blocks.append(current)

    empty = {"vendor": None, "model": None, "family": None, "model_id": None,
             "stepping": None, "physical": None, "logical": None,
             "features": _feature_map((), False)}
    if not blocks:
        return empty

    first = blocks[0]
    flags, flags_seen = set(), False
    for block in blocks:
        for key in ("flags", "features"):
            if key in block:
                flags_seen = True
                flags.update(block[key].split())

    logical = len(blocks) or None
    per_socket = _int(first.get("cpu cores"))
    sockets = len({block.get("physical id", "0") for block in blocks})
    if per_socket:
        physical = per_socket * max(sockets, 1)
    else:
        physical = len({(block.get("physical id", "0"),
                         block.get("core id", block.get("processor", str(i))))
                        for i, block in enumerate(blocks)}) or None

    return {
        "vendor": first.get("vendor_id") or first.get("vendor"),
        "model": first.get("model name") or first.get("hardware"),
        "family": _int(first.get("cpu family")),
        "model_id": _int(first.get("model")),
        "stepping": _int(first.get("stepping")),
        "physical": physical,
        "logical": logical,
        "features": _feature_map(flags, flags_seen),
    }


def parse_lscpu(text):
    """The same CPU fields from lscpu, used only when cpuinfo is unavailable."""
    fields, flags = {}, set()
    for line in (text or "").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        if key in ("flags", "features"):
            flags.update(value.split())
        else:
            fields[key] = value
    per_socket = _int(fields.get("core(s) per socket"))
    sockets = _int(fields.get("socket(s)")) or 1
    return {
        "vendor": fields.get("vendor id"),
        "model": fields.get("model name"),
        "family": _int(fields.get("cpu family")),
        "model_id": _int(fields.get("model")),
        "stepping": _int(fields.get("stepping")),
        "physical": per_socket * sockets if per_socket else None,
        "logical": _int(fields.get("cpu(s)")),
        "features": _feature_map(flags, bool(flags)),
    }


def _merge_cpu(cpu, fallback):
    """Fill fields that the primary source did not expose, without guessing."""
    merged = dict(cpu)
    for key in ("vendor", "model", "family", "model_id", "stepping", "physical", "logical"):
        if merged.get(key) is None and fallback.get(key) is not None:
            merged[key] = fallback[key]
    if all(value is None for value in merged["features"].values()):
        merged["features"] = dict(fallback["features"])
    return merged


def _run(command, runner=None):
    """Run a read-only probe command and keep its output/error for the report."""
    try:
        if runner is None:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=20)
        else:
            proc = runner(command)
    except FileNotFoundError as exc:
        return {"command": command, "available": False, "returncode": None,
                "stdout": "", "stderr": "", "error": "command not found: %s" % exc}
    except subprocess.TimeoutExpired as exc:
        return {"command": command, "available": False, "returncode": None,
                "stdout": "", "stderr": "", "error": "timed out: %s" % exc}
    except OSError as exc:
        return {"command": command, "available": False, "returncode": None,
                "stdout": "", "stderr": "", "error": "could not run: %s" % exc}

    def text(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return value or ""

    returncode = getattr(proc, "returncode", None)
    stdout, stderr = text(getattr(proc, "stdout", "")), text(getattr(proc, "stderr", ""))
    error = None
    if returncode != 0:
        error = stderr.strip() or "command exited %s" % returncode
    return {"command": command, "available": returncode == 0, "returncode": returncode,
            "stdout": stdout, "stderr": stderr, "error": error}


def parse_lspci(text):
    """Intel display PCI functions from `lspci -nnk`, including their driver."""
    header = re.compile(
        r"^(?P<bdf>(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])\s+"
        r"(?P<cls>[^:]+):\s*(?P<desc>.*)$")
    entries, current = [], {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        match = header.match(line)
        if match:
            bdf = match.group("bdf")
            if bdf.count(":") == 1:
                bdf = "0000:" + bdf
            desc = match.group("desc")
            ids = re.findall(r"\[([0-9a-fA-F]{4}):([0-9a-fA-F]{4})\]", desc)
            if not ids:
                current = {}
                continue
            vendor, device = next(((v, d) for v, d in ids if v.lower() == "8086"), ids[-1])
            name = re.sub(r"\s*\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\].*$", "", desc).strip()
            cls = re.sub(r"\s*\[[0-9a-fA-F]{4}\]$", "", match.group("cls").strip())
            current = {"bdf": bdf, "vendor_id": int(vendor, 16),
                       "device_id": int(device, 16), "class": cls,
                       "name": name, "driver": ""}
            entries.append(current)
        elif current and line.startswith("Kernel driver in use:"):
            current["driver"] = line.split(":", 1)[1].strip()
    return entries


def _is_display(entry):
    cls = (entry.get("class") or "").lower()
    return any(word in cls for word in ("vga", "3d", "display"))


def _merge_lspci(devices, pci):
    """Corroborate sysfs devices with lspci and append lspci-only displays."""
    by_bdf = {entry["bdf"]: entry for entry in pci}
    used = set()
    for dev in devices:
        match = by_bdf.get(dev.get("bdf") or "")
        if match is None and not dev.get("bdf"):
            candidates = [entry for entry in pci
                          if entry["vendor_id"] == dev.get("vendor_id")
                          and entry["device_id"] == dev.get("device_id")]
            if len(candidates) == 1:
                match = candidates[0]
        if match is None:
            continue
        dev["bdf"] = match["bdf"]
        if not dev.get("driver"):
            dev["driver"] = match.get("driver") or ""
        dev["lspci_name"] = match.get("name") or ""
        used.add(match["bdf"])

    for entry in pci:
        if entry["bdf"] in used or entry.get("vendor_id") != 0x8086 or not _is_display(entry):
            continue
        name, arch = arch_of(entry["device_id"])
        name = name or DEVICE_NAMES.get(entry["device_id"])
        devices.append({
            "render_node": "", "pci_id": "", "bdf": entry["bdf"],
            "vendor_id": entry["vendor_id"], "device_id": entry["device_id"],
            "name": name or entry.get("name"), "arch": arch,
            "driver": entry.get("driver") or "", "integrated": _integrated(arch, name),
            "source": "lspci", "lspci_name": entry.get("name") or "",
        })
    return devices


def parse_sycl_ls(text):
    """SYCL devices from `sycl-ls --verbose`, preserving backend and device index."""
    devices = []
    for line in (text or "").splitlines():
        match = _SYCL_DEVICE.match(line)
        if not match:
            continue
        devices.append({
            "backend": match.group("backend").strip().lower(),
            "kind": match.group("kind").strip().lower(),
            "index": int(match.group("index")),
            "description": match.group("desc").strip(),
        })
    return devices


def _norm_name(name):
    """Whitespace/case-insensitive adapter name, for the CIM <-> registry join."""
    return " ".join((name or "").split()).lower()


def parse_windows_video(text):
    """Intel display adapters from `Win32_VideoController` (name|driver|pnp|status).

    The command line reports the adapter's own PNP instance, so the vendor and
    device ids are the hardware's, not a table lookup.  Non-Intel adapters are
    skipped, as on the sysfs path.
    """
    devices = []
    for raw in (text or "").replace("\x00", "").splitlines():
        parts = raw.strip().split("|")
        if len(parts) < 3:
            continue
        name, version, pnp = parts[0].strip(), parts[1].strip(), parts[2].strip()
        match = _PNP_IDS.search(pnp)
        if match is None or int(match.group(1), 16) != 0x8086:
            continue
        device_id = int(match.group(2), 16)
        known, arch = arch_of(device_id)
        name = name or known or DEVICE_NAMES.get(device_id)
        devices.append({
            "render_node": "", "pci_id": "", "bdf": "",
            "vendor_id": 0x8086, "device_id": device_id,
            "name": name, "arch": arch, "driver": "",
            "integrated": _integrated(arch, name),
            "source": "win32",
            "driver_version": version or None,
            "status": parts[3].strip() if len(parts) > 3 else "",
            "pnp": pnp,
            "memory_mb": None,
        })
    return devices


def parse_windows_vram(text):
    """Adapter name -> dedicated memory in bytes, from the display class keys.

    `Win32_VideoController.AdapterRAM` is a 32-bit field and reports 4 GiB-ish
    for larger cards; `HardwareInformation.qwMemorySize` is the 64-bit figure
    the driver itself stores, which is why it is the source used here.
    """
    sizes = {}
    for raw in (text or "").replace("\x00", "").splitlines():
        name, sep, size = raw.strip().rpartition("|")
        if not sep:
            continue
        try:
            value = int(size.strip())
        except ValueError:
            continue
        if value > 0:
            sizes[_norm_name(name)] = value
    return sizes


def parse_windows_cpu(text):
    """CPU identity from `Win32_Processor` (manufacturer|name|cores|logical).

    Windows has no /proc/cpuinfo, and this path does not ask a compiler for CPU
    flags: the feature map and the CPUID family/model/stepping stay unknown
    here, and `kindi info` remains the place that reports the compiler's flags.
    """
    for raw in (text or "").replace("\x00", "").splitlines():
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) < 4 or not parts[1]:
            continue
        return {
            "vendor": parts[0] or None,
            "model": parts[1],
            "family": None,
            "model_id": None,
            "stepping": None,
            "physical": _int(parts[2]),
            "logical": _int(parts[3]),
            "features": _feature_map((), False),
            "source": "win32",
        }
    return {}


def parse_level_zero_registration(text):
    """Level Zero/OpenCL drivers the display class registers (device|L0|ver|CL)."""
    entries = []
    for raw in (text or "").replace("\x00", "").splitlines():
        parts = [part.strip() for part in raw.split("|")]
        if len(parts) < 2 or not parts[1]:
            continue
        entries.append({
            "device": parts[0],
            "level_zero_driver": parts[1],
            "version": parts[2] if len(parts) > 2 and parts[2] else None,
            "opencl_driver": parts[3] if len(parts) > 3 and parts[3] else None,
        })
    return entries


def level_zero_counts(loader=None, opener=None):
    """Read-only Level Zero enumeration: driver and device counts only.

    Handles are opaque here.  `ze_device_properties_t` is a versioned struct and
    reading a name at a guessed offset is how a probe starts inventing devices,
    so this asks the runtime how many drivers and devices exist and stops there.
    ``opener`` is injectable so the enumeration is testable off Windows.
    """
    loader = loader or (_WIN_L0_LOADER if os.name == "nt" else "libze_loader.so.1")
    open_library = opener or (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)
    result = {"loader": loader, "available": False, "drivers": None,
              "devices": None, "driver_devices": None, "error": None,
              "returncodes": {}}
    try:
        lib = open_library(loader)
    except (OSError, TypeError, AttributeError) as exc:
        result["error"] = "could not load %s: %s" % (loader, exc)
        return result
    try:
        lib.zeInit.restype = ctypes.c_int
        lib.zeInit.argtypes = [ctypes.c_uint32]
        code = int(lib.zeInit(0))
        result["returncodes"]["zeInit"] = code
        if code != 0:
            result["error"] = "zeInit returned 0x%08x" % (code & 0xffffffff)
            return result

        lib.zeDriverGet.restype = ctypes.c_int
        lib.zeDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32),
                                    ctypes.POINTER(ctypes.c_void_p)]
        count = ctypes.c_uint32(0)
        code = int(lib.zeDriverGet(count, None))
        result["returncodes"]["zeDriverGet"] = code
        if code != 0:
            result["error"] = "zeDriverGet returned 0x%08x" % (code & 0xffffffff)
            return result
        result["drivers"] = count.value

        per_driver, total = [], 0
        if count.value:
            handles = (ctypes.c_void_p * count.value)()
            code = int(lib.zeDriverGet(count, handles))
            result["returncodes"]["zeDriverGetHandles"] = code
            if code != 0:
                result["error"] = "zeDriverGet(handles) returned 0x%08x" % (code & 0xffffffff)
                return result
            lib.zeDeviceGet.restype = ctypes.c_int
            lib.zeDeviceGet.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_uint32),
                                        ctypes.POINTER(ctypes.c_void_p)]
            for handle in handles:
                devices = ctypes.c_uint32(0)
                code = int(lib.zeDeviceGet(handle, devices, None))
                if code != 0:
                    per_driver.append(None)
                    continue
                per_driver.append(devices.value)
                total += devices.value
        result["driver_devices"] = per_driver
        result["devices"] = total
        result["available"] = True
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        result["error"] = "Level Zero enumeration failed: %s" % exc
    return result


def windows_level_zero(runner=None, windir=None, counts=None):
    """Level Zero state on Windows: loader, registered driver and enumeration.

    The loader DLL ships with the driver or oneAPI, but devices enumerate only
    when the display class registers a Level Zero driver, so both are recorded.
    ``counts`` injects a recorded enumeration; otherwise it runs read-only.
    """
    root = os.path.join(windir or os.environ.get("WINDIR") or r"C:\Windows", "System32")
    loader_path = os.path.join(root, _WIN_L0_LOADER)
    loader = {"path": loader_path, "present": os.path.exists(loader_path),
              "version": None, "error": None}
    if loader["present"]:
        quoted = loader_path.replace("'", "''")
        version = _run(list(_PS) + ["(Get-Item '%s').VersionInfo.FileVersion" % quoted], runner)
        lines = [line.strip() for line in version["stdout"].replace("\x00", "").splitlines()
                 if line.strip()]
        loader["version"] = lines[0] if lines else None
        loader["error"] = version["error"]
    else:
        loader["error"] = "not installed"
    registration = _run(list(_PS) + [_WIN_L0_REGISTER], runner)
    return {
        "source": "windows display class",
        "loader": loader,
        "registered": parse_level_zero_registration(registration["stdout"]),
        "enumeration": level_zero_counts() if counts is None else counts,
        "command": registration,
    }


def _rebar_field():
    """Resizable BAR state: not read here, captured with the host's own tool."""
    return {"state": None, "source": None,
            "note": "not exposed by sysfs or the Windows display stack; capture it with "
                    "`lspci -vv` (Linux) or Intel Arc Control/GPU-Z (Windows)"}


def _npu_probe(accel_root=_SYS_ACCEL):
    """Intel NPU presence from /sys/class/accel, with explicit absence."""
    if not os.path.isdir(accel_root):
        return {"source": accel_root, "available": False, "present": False,
                "devices": [], "error": "accel sysfs not available"}
    devices = []
    for accel in sorted(glob.glob(os.path.join(accel_root, "accel*"))):
        vendor = _read(os.path.join(accel, "device", "vendor"))
        if vendor.lower() != "0x8086":
            continue
        device = _read(os.path.join(accel, "device", "device"))
        uevent = _read(os.path.join(accel, "device", "uevent"))
        driver = ""
        for line in uevent.splitlines():
            if line.startswith("DRIVER="):
                driver = line.split("=", 1)[1].strip()
        devices.append({"accel": os.path.basename(accel), "vendor_id": 0x8086,
                        "device_id": _int(device, 16), "bdf": _bdf_from_card(accel),
                        "driver": driver})
    return {"source": accel_root, "available": True, "present": bool(devices),
            "devices": devices, "error": None}


def _oneapi_default_root():
    """The oneAPI install root this OS uses."""
    if os.name == "nt":
        base = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
        return os.path.join(base, "Intel", "oneAPI")
    return _ONEAPI_ROOT


def _oneapi_probe(oneapi_root=None):
    """oneAPI installation presence; never imports a oneAPI package."""
    root = oneapi_root or os.environ.get("ONEAPI_ROOT") or _oneapi_default_root()
    return {"root": root, "present": os.path.isdir(root),
            "source": "ONEAPI_ROOT" if os.environ.get("ONEAPI_ROOT") and not oneapi_root else "default"}


def _openvino_probe(runner=None):
    """Optional OpenVINO version/devices from a read-only Python subprocess."""
    result = _run([sys.executable, "-c", _OPENVINO_CODE], runner)
    # Record the interpreter by name, never by its local path: a saved report is
    # evidence, and evidence must not carry a machine root.
    result["command"] = ["python"] + result["command"][1:]
    if not result["available"]:
        return {"available": False, "version": None, "devices": [],
                "error": result["error"] or "OpenVINO probe failed"}
    payload = None
    for line in reversed(result["stdout"].splitlines()):
        try:
            payload = json.loads(line)
            break
        except ValueError:
            continue
    if not isinstance(payload, dict):
        return {"available": False, "version": None, "devices": [],
                "error": "OpenVINO probe returned no JSON"}
    devices = payload.get("devices")
    return {"available": True, "version": payload.get("version"),
            "devices": list(devices) if isinstance(devices, list) else [], "error": None}


def probe(drm_root=_SYS_DRM, dri_root=_DEV_DRI, accel_root=_SYS_ACCEL,
          cpuinfo_path=_CPUINFO, oneapi_root=None, runner=None,
          windows=None, windir=None, ze_counts=None):
    """Collect the M1 Intel checklist as JSON-safe data, with no backend claim.

    ``runner`` accepts a command list and returns a CompletedProcess-like object;
    tests use it with synthetic lscpu/lspci/sycl-ls/clinfo/OpenVINO output.  Every
    missing source remains a field: unavailable commands carry an ``error`` and
    absent hardware is represented by an empty list with ``present: false``.

    ``windows`` defaults to the host OS.  On Windows the CPU and display adapters
    come from CIM instead of /proc and sysfs, and the Level Zero loader, its
    registered driver and the driver/device counts are recorded.  ``ze_counts``
    injects a recorded enumeration so the report is testable off Windows.
    """
    windows = (os.name == "nt") if windows is None else bool(windows)
    cpuinfo_text = _read(cpuinfo_path)
    lscpu_result = _run(["lscpu"], runner)
    cpu = _merge_cpu(parse_cpuinfo(cpuinfo_text), parse_lscpu(lscpu_result["stdout"]))
    cpu["source"] = "proc_cpuinfo" if cpuinfo_text else ("lscpu" if lscpu_result["available"] else "none")

    devices = discover(drm_root=drm_root, dri_root=dri_root)
    lspci_result = _run(["lspci", "-nnk"], runner)
    pci_devices = parse_lspci(lspci_result["stdout"])
    devices = _merge_lspci(devices, pci_devices)

    sycl_result = _run(["sycl-ls", "--verbose"], runner)
    sycl_devices = parse_sycl_ls(sycl_result["stdout"])
    level_zero = [device for device in sycl_devices
                  if "level_zero" in device["backend"]]
    clinfo_result = _run(["clinfo"], runner)
    commands = {"lscpu": lscpu_result, "lspci": lspci_result,
                "sycl-ls": sycl_result, "clinfo": clinfo_result}

    if windows:
        video_result = _run(list(_PS) + [_WIN_VIDEO], runner)
        vram_result = _run(list(_PS) + [_WIN_VRAM], runner)
        cpu_result = _run(list(_PS) + [_WIN_CPU], runner)
        windows_devices = parse_windows_video(video_result["stdout"])
        sizes = parse_windows_vram(vram_result["stdout"])
        for dev in windows_devices:
            size = sizes.get(_norm_name(dev.get("name")))
            if size:
                dev["memory_mb"] = size // (1024 * 1024)
        devices.extend(windows_devices)
        windows_cpu = parse_windows_cpu(cpu_result["stdout"])
        if windows_cpu.get("model"):
            cpu = windows_cpu
        commands.update({"win-video": video_result, "win-vram": vram_result,
                         "win-cpu": cpu_result})

    report = {
        "probe": "intel-info",
        "schema": 1,
        "read_only": True,
        "support_claim": False,
        "backend": "not implemented",
        "system": "windows" if windows else "posix",
        "paths": {"drm": drm_root, "dri": dri_root, "accel": accel_root,
                  "cpuinfo": cpuinfo_path},
        "cpu": cpu,
        "devices": devices,
        "npu": _npu_probe(accel_root),
        "oneapi": _oneapi_probe(oneapi_root),
        "level_zero": {"source": "sycl-ls", "available": sycl_result["available"],
                       "error": sycl_result["error"], "devices": level_zero},
        "openvino": _openvino_probe(runner),
        "rebar": _rebar_field(),
        "commands": commands,
    }
    if windows:
        report["level_zero"]["runtime"] = windows_level_zero(
            runner=runner, windir=windir, counts=ze_counts)
    return report


def render(report):
    """Human-readable form of `probe()`; the JSON form is the machine interface."""
    def value(item):
        return "unknown" if item is None else str(item)

    def switch(item):
        return "unknown" if item is None else ("yes" if item else "no")

    cpu = report["cpu"]
    lines = ["Intel probe (read-only; no Intel backend or support claim)",
             "CPU: %s %s (family %s, model %s, stepping %s)" % (
                 value(cpu.get("vendor")), value(cpu.get("model")), value(cpu.get("family")),
                 value(cpu.get("model_id")), value(cpu.get("stepping"))),
             "  cores: %s physical / %s logical" % (value(cpu.get("physical")), value(cpu.get("logical"))),
             "  features: " + ", ".join("%s=%s" % (name, switch(flag))
                                       for name, flag in cpu["features"].items())]
    if report["devices"]:
        lines.append("Intel display devices:")
        for dev in report["devices"]:
            driver = dev.get("driver") or ("n/a" if dev.get("source") == "win32" else "unknown")
            lines.append("  %s [%s:%04x] %s arch=%s driver=%s render=%s integrated=%s source=%s" % (
                dev.get("bdf") or "?", dev.get("vendor_id", 0), dev.get("device_id") or 0,
                dev.get("name") or "unknown", dev.get("arch") or "unknown",
                driver, dev.get("render_node") or "none",
                switch(dev.get("integrated")), dev.get("source")))
            extra = []
            if dev.get("memory_mb"):
                extra.append("vram=%d MiB" % dev["memory_mb"])
            if dev.get("driver_version"):
                extra.append("driver_version=%s" % dev["driver_version"])
            if dev.get("status"):
                extra.append("status=%s" % dev["status"])
            if extra:
                lines.append("    " + ", ".join(extra))
    else:
        lines.append("Intel display devices: none")

    npu = report["npu"]
    lines.append("Intel NPU: %s (%s)" % (switch(npu.get("present")), npu.get("error") or npu.get("source")))
    for device in npu.get("devices", []):
        lines.append("  %s device=0x%04x bdf=%s driver=%s" % (
            device["accel"], device.get("device_id") or 0, device.get("bdf") or "?",
            device.get("driver") or "unknown"))

    oneapi = report["oneapi"]
    lines.append("oneAPI: %s (%s)" % ("present" if oneapi["present"] else "not installed", oneapi["root"]))
    level_zero = report["level_zero"]
    if level_zero["devices"]:
        lines.append("Level Zero devices (sycl-ls --verbose):")
        for device in level_zero["devices"]:
            lines.append("  [%s:%s:%s] %s" % (device["backend"], device["kind"],
                                               device["index"], device["description"]))
    else:
        lines.append("Level Zero devices (sycl-ls --verbose): none%s" % (
            " (%s)" % level_zero["error"] if level_zero.get("error") else ""))
    runtime = level_zero.get("runtime")
    if runtime:
        loader = runtime["loader"]
        lines.append("Level Zero runtime: loader %s%s" % (
            loader.get("version") or "unknown", "" if loader.get("present") else " (not installed)"))
        for entry in runtime.get("registered", []):
            lines.append("  %s: driver=%s version=%s" % (
                entry.get("device") or "device", entry.get("level_zero_driver") or "none",
                entry.get("version") or "unknown"))
        enumeration = runtime.get("enumeration") or {}
        if enumeration.get("available"):
            lines.append("  enumeration: %s driver(s), %s device(s)" % (
                value(enumeration.get("drivers")), value(enumeration.get("devices"))))
        else:
            lines.append("  enumeration: unavailable (%s)" % (enumeration.get("error") or "not run"))

    openvino = report["openvino"]
    if openvino["available"]:
        lines.append("OpenVINO: %s; devices: %s" % (
            value(openvino.get("version")), ", ".join(openvino["devices"]) or "none"))
    else:
        lines.append("OpenVINO: unavailable (%s)" % (openvino.get("error") or "not installed"))

    rebar = report.get("rebar") or {}
    if rebar.get("state") is None:
        lines.append("ReBAR: not read (%s)" % (rebar.get("note") or "unavailable"))
    else:
        lines.append("ReBAR: %s" % rebar["state"])

    for name, result in report["commands"].items():
        lines.append("command %s: %s" % (name, "ok" if result["available"] else result.get("error") or "unavailable"))
    return "\n".join(lines)


def eligible(dev):
    """True when the device clears the oneAPI floor (known arch + GPU driver).

    Unknown architectures are refused, not assumed: the same conservatism as
    the support matrix, which renders unmeasured cards as `not measured`.
    """
    if dev.get("arch") not in SUPPORTED_ARCH:
        return False
    if dev.get("driver") and dev["driver"] not in GPU_DRIVERS:
        return False
    return True


def eligibility_reason(dev):
    """One human-readable line explaining the eligible() verdict."""
    if dev.get("arch") is None:
        return "unknown PCI device id 0x%04x: not in the measured table" % dev.get("device_id", 0)
    if dev.get("arch") not in SUPPORTED_ARCH:
        return "%s (%s) is below the oneAPI floor (11th gen / Iris Xe or newer)" % (
            dev.get("name") or "device", dev["arch"])
    if dev.get("driver") and dev["driver"] not in GPU_DRIVERS:
        return "%s is bound to %s, not a GPU driver (i915/xe)" % (dev.get("name"), dev["driver"])
    return "%s (%s, %s) is eligible" % (dev.get("name"), dev["arch"], dev.get("driver") or "driver unknown")


def shared_budget_mb(total_ram_mb, override_pct=None):
    """Usable iGPU memory in MiB.

    Windows caps the graphics share at one-half of system RAM (Intel graphics
    memory FAQ). Core Ultra Series 2+ adds a Shared GPU Memory Override
    (default 57%, up to 87-93% with enough RAM); pass ``override_pct`` to use
    it, e.g. ``shared_budget_mb(65536, 87)``. Linux i915/xe has no such fixed
    cap: pass ``override_pct=100`` minus what the OS keeps, or measure the
    device's reported limit instead of guessing.
    """
    if override_pct is None:
        return int(total_ram_mb) // 2
    pct = max(0, min(100, int(override_pct)))
    return int(total_ram_mb) * pct // 100


def fits(budget_mb, weights_mb, kv_mb, reserve_mb=512):
    """True when a ternary pack fits the shared budget with headroom.

    ``weights_mb``/``kv_mb`` come from the pack manifest (`catalog.py`
    figures); the iGPU shares the bus with the CPU, so a reserve for the OS
    and the CPU-side layers is part of the fit, not an afterthought.
    """
    return int(weights_mb) + int(kv_mb) + int(reserve_mb) <= int(budget_mb)


def tandem_gpu_contract(engine):
    """Check an engine exposes the TandemEngine GPU-side surface.

    The tandem boundary needs ``get_hidden()`` / ``set_hidden()`` plus the
    layer-range constructor surface. A future SYCL iGPU engine must pass
    this before ``TandemEngine`` may wrap it; returns a list of missing
    attributes (empty = contract satisfied).
    """
    missing = []
    for attr in ("get_hidden", "set_hidden", "D", "vocab"):
        if not hasattr(engine, attr):
            missing.append(attr)
    return missing
