"""What each card and this CPU can really do (`kindi caps`, and the candidate list `kindi autotune` starts from).

Identification is layered so it still answers when the heavy pieces are missing: nvidia-smi first
(the driver is enough; no CUDA runtime needed), then pynvml, then CuPy. The CPU half calls the same
probe the engine uses to choose its kernel build (`kindi/cpu.py`), so the report and the engine
cannot disagree about avx2/fma/f16c.

`card_features()` is a hardware fact table: compute capability -> what the silicon has. What Kindi
can *use* from that is a separate, narrower list, because a card having FP8 tensor cores says
nothing about Kindi having an FP8 kernel for this model. The report keeps both, separately, so a
reader cannot mistake one for the other.
"""
import json
import os
import shutil
import subprocess

SMI_CMD = ["nvidia-smi",
           "--query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,memory.free,driver_version",
           "--format=csv,noheader,nounits"]

# Index of each column in SMI_CMD's output; older drivers have no compute_cap column, and the
# shorter layout is accepted by name rather than by position.
SMI_FIELDS = ("index", "name", "uuid", "pci_bus_id", "compute_cap", "memory_total_mib",
              "memory_free_mib", "driver")


def _int(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _cap(value):
    """A compute capability as "major.minor", or None when the driver does not report one."""
    text = str(value or "").strip()
    if not text or text.upper() in ("N/A", "[N/A]", "NOT SUPPORTED"):
        return None
    try:
        major, minor = text.split(".")[:2]
        return "%d.%d" % (int(major), int(minor))
    except (TypeError, ValueError):
        return None


def parse_smi(text):
    """Rows from `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`.

    The query fixes the column order, so a row is read by position; a row with one column fewer is
    the pre-`compute_cap` layout (index,name,uuid,pci,total,free,driver) and is read that way.
    """
    rows = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(SMI_FIELDS):
            values = parts[:len(SMI_FIELDS)]
        elif len(parts) == len(SMI_FIELDS) - 1:
            values = parts[:4] + [""] + parts[4:]
        else:
            continue
        row = dict(zip(SMI_FIELDS, values))
        row["index"] = _int(row["index"])
        row["memory_total_mib"] = _int(row["memory_total_mib"])
        row["memory_free_mib"] = _int(row["memory_free_mib"])
        row["compute_cap"] = _cap(row["compute_cap"])
        rows.append(row)
    return rows


def card_features(compute_cap):
    """Silicon capabilities implied by a compute capability; None when it is unknown.

    Kept to facts the architecture itself guarantees (see NVIDIA's compute-capability tables), not
    to what a driver or a Kindi build enables.
    """
    if not compute_cap:
        return None
    try:
        major, minor = [int(x) for x in str(compute_cap).split(".")[:2]]
    except (TypeError, ValueError):
        return None
    cc = major * 10 + minor
    return {
        "tensor_cores": cc >= 70,          # Volta's first-generation tensor cores
        "bf16": cc >= 80,                  # Ampere
        "tf32": cc >= 80,
        "fp8": cc >= 89,                   # Ada (8.9), Hopper (9.0), Blackwell (10.0/12.0)
        "fp4": cc >= 100,                  # Blackwell microscaling formats
        "async_copy": cc >= 80,
    }


def kindi_paths(compute_cap):
    """The Kindi features this card can actually run, from gates that exist in the tree."""
    paths = ["5-trit kernels", "dense CPU+GPU tandem split"]
    if not compute_cap:
        return paths
    try:
        major, minor = [int(x) for x in str(compute_cap).split(".")[:2]]
    except (TypeError, ValueError):
        return paths
    cc = major * 10 + minor
    if cc >= 70:
        paths.append("plan engine (tensor-core MMA)")
    if cc >= 89:
        paths.append("FP8 Gemma 4 path (on by default, KINDI_G4_FP8=0 disables)")
    return paths


def _from_smi():
    try:
        out = subprocess.check_output(SMI_CMD, stderr=subprocess.DEVNULL, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_smi(out.decode("utf-8", "replace"))


def _from_nvml():
    try:
        import pynvml
    except ImportError:
        return []
    rows = []
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            rows.append({"index": i, "name": name, "uuid": "", "pci_bus_id": "",
                         "compute_cap": "%d.%d" % (major, minor),
                         "memory_total_mib": int(mem.total) // 2 ** 20,
                         "memory_free_mib": int(mem.free) // 2 ** 20, "driver": ""})
    except Exception:  # noqa: BLE001 - NVML is a fallback; the CPU and no-card paths still work
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return rows


def _from_cupy():
    try:
        import cupy as cp
    except ImportError:
        return []
    rows = []
    try:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            p = cp.cuda.runtime.getDeviceProperties(i)
            name = p["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            rows.append({"index": i, "name": name, "uuid": "", "pci_bus_id": "",
                         "compute_cap": "%d.%d" % (p["major"], p["minor"]),
                         "memory_total_mib": int(p["totalGlobalMem"]) // 2 ** 20,
                         "memory_free_mib": None, "driver": ""})
    except Exception:  # noqa: BLE001 - no driver/no device is a normal answer here
        return []
    return rows


def parse_lscpu(text):
    """Cache sizes and the extended ISA flags that /proc/cpuinfo and lscpu expose on Linux."""
    out = {"l2_kb": None, "l3_kb": None, "avx512": None, "amx": None}
    for line in str(text or "").splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        k, v = key.strip().lower(), value.strip()
        if k in ("l2 cache", "l3 cache"):
            try:
                number = "".join(ch for ch in v if ch.isdigit())
                out["l2_kb" if k.startswith("l2") else "l3_kb"] = int(number) if number else None
            except (TypeError, ValueError):
                pass
        elif k == "flags":
            flags = set(v.lower().replace(",", " ").split())
            out["avx512"] = bool(flags & set(("avx512f", "avx512_f"))) if flags else None
            out["amx"] = bool(flags & set(("amx_tile", "amx_bf16", "amx_int8"))) if flags else None
    return out


def parse_wmi(text):
    """Processor name, cores and cache sizes from `Get-CimInstance Win32_Processor` JSON (Windows)."""
    out = {"l2_kb": None, "l3_kb": None, "avx512": None, "amx": None,
           "model": None, "physical": None, "logical": None}
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return out
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return out
    for src, dst in (("L2CacheSize", "l2_kb"), ("L3CacheSize", "l3_kb")):
        value = _int(data.get(src))
        if value:
            out[dst] = value
    name = data.get("Name")
    if isinstance(name, str) and name.strip():
        out["model"] = " ".join(name.split())
    for src, dst in (("NumberOfCores", "physical"), ("NumberOfLogicalProcessors", "logical")):
        value = _int(data.get(src))
        if value:
            out[dst] = value
    return out


def cpu_extras():
    """Cache sizes and AVX-512/AMX presence, from the OS where it exposes them.

    Windows has no /proc/cpuinfo; WMI reports the caches but not the flags, and the compiler probe
    in `kindi/cpu.py` only answers for avx/avx2/fma/f16c, so the extended flags stay None there
    rather than guessed. None always means "this OS did not tell us", never "unsupported".
    """
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
                  "NumberOfLogicalProcessors,L2CacheSize,L3CacheSize | ConvertTo-Json -Compress")
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps],
                                          stderr=subprocess.DEVNULL, timeout=30)
            return parse_wmi(out.decode("utf-8", "replace"))
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            proc = f.read()
        try:
            lscpu = subprocess.check_output(["lscpu"], stderr=subprocess.DEVNULL, timeout=20)
            lscpu = lscpu.decode("utf-8", "replace")
        except (OSError, subprocess.SubprocessError):
            lscpu = ""
        parsed = parse_lscpu(lscpu)
        flags = parse_lscpu(proc)
        for key in ("avx512", "amx"):
            if parsed[key] is None:
                parsed[key] = flags[key]
        return parsed
    except Exception:  # noqa: BLE001 - the extras are best-effort by design
        return {"l2_kb": None, "l3_kb": None, "avx512": None, "amx": None}


# kernel32's PF_*_INSTRUCTIONS_AVAILABLE values (Windows 10 SDK winnt.h). There is no constant
# for FMA3 or F16C, so those two stay unknown on Windows unless a compiler probe answers them.
WINDOWS_PF = (("avx", 39), ("avx2", 40), ("avx512", 41))


def windows_features():
    """AVX / AVX2 / AVX-512F from `IsProcessorFeaturePresent` — no compiler, no /proc needed.

    This is the part of the interrupted CPUID draft that can be settled without compiling anything;
    its AVX-512 kernels are still archived because no AVX-512 machine is available to verify them.
    """
    out = {"avx": None, "avx2": None, "avx512": None}
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        for key, number in WINDOWS_PF:
            out[key] = bool(kernel32.IsProcessorFeaturePresent(number))
    except Exception:  # noqa: BLE001 - not Windows, or no kernel32: unknown, not false
        return {"avx": None, "avx2": None, "avx512": None}
    return out


def toolchain():
    """The compiler the CPU engine would be built with, resolved the same way `kindi/cpu.py` does.

    A CPU feature nobody can compile is not an available feature: the tandem split and the CPU
    engine both need this build, so the report says whether it can happen here.
    """
    name = os.environ.get("KINDI_CC") or ("clang" if os.name == "nt" else "gcc")
    path = shutil.which(name)
    return {"name": name, "path": path, "ok": bool(path)}


def cpu_report():
    """The CPU half: the engine's own probe plus what it does not cover (caches, extended flags)."""
    from . import cpu as cpu_backend

    report = cpu_backend.feature_report()
    tier = cpu_backend.cpu_tier(report)
    extras = cpu_extras()
    # Fill what the OS would not tell the engine probe with what it will tell this command; the
    # engine's own answer always wins where it has one, so the report cannot contradict it.
    features = dict(report["features"] or {})
    if os.name == "nt":
        detected = windows_features()
        for key in ("avx", "avx2"):
            if features.get(key) is None and detected.get(key) is not None:
                features[key] = detected[key]
        if extras.get("avx512") is None and detected.get("avx512") is not None:
            extras["avx512"] = detected["avx512"]
    return {
        # WMI has the marketing name and core counts on Windows, where platform.processor()
        # answers "Intel64 Family 6 Model 158 Stepping 10"; Linux answers from /proc/cpuinfo.
        "model": extras.get("model") or report["model"],
        "physical": report["physical"] or extras.get("physical"),
        "logical": report["logical"] or extras.get("logical"),
        "features_known": any(v is not None for v in features.values()),
        "features": features,
        "tier": tier,
        "tier_label": cpu_backend.CPU_TIER_LABELS.get(tier, tier),
        "build": cpu_backend.select_isa(report) or "native (-march=native)",
        "threads_recommended": report["physical"] or report["logical"] or extras.get("physical"),
        "toolchain": toolchain(),
        "l2_kb": extras.get("l2_kb"),
        "l3_kb": extras.get("l3_kb"),
        "avx512": extras.get("avx512"),
        "amx": extras.get("amx"),
    }


def probe():
    """The whole report: cards (with their fact table) and the CPU."""
    for finder, source in ((_from_smi, "nvidia-smi"), (_from_nvml, "pynvml"), (_from_cupy, "cupy")):
        rows = finder()
        if rows:
            break
    else:
        rows, source = [], "none"
    for row in rows:
        row["features"] = card_features(row.get("compute_cap"))
        row["kindi"] = kindi_paths(row.get("compute_cap"))
    return {"gpus": rows, "gpu_source": source, "cpu": cpu_report()}


def _tri(value):
    return "unknown" if value is None else ("yes" if value else "no")


def render(report):
    """Human-readable lines; the JSON form is `probe()` itself."""
    lines = []
    rows = report.get("gpus") or []
    if not rows:
        lines.append("GPU: none detected (nvidia-smi, pynvml and CuPy all found no card)")
    else:
        lines.append("GPU: %d card%s (identified with %s)" %
                     (len(rows), "" if len(rows) == 1 else "s", report.get("gpu_source")))
    for row in rows:
        cap = row.get("compute_cap")
        lines.append("  card %s: %s%s" % (row.get("index"), row.get("name"),
                                          "" if cap is None else "  [sm_%s]" % cap.replace(".", "")))
        mem = "%s MiB total" % (row.get("memory_total_mib") if row.get("memory_total_mib") is not None else "?")
        if row.get("memory_free_mib") is not None:
            mem += ", %d MiB free" % row["memory_free_mib"]
        detail = [mem]
        if row.get("driver"):
            detail.append("driver %s" % row["driver"])
        if row.get("pci_bus_id"):
            detail.append("PCI %s" % row["pci_bus_id"])
        if row.get("uuid"):
            detail.append("UUID %s" % row["uuid"])
        lines.append("           " + ", ".join(detail))
        feats = row.get("features")
        if feats is None:
            lines.append("           hardware features: unknown (the driver reports no compute capability)")
        else:
            names = [name for name in ("tensor_cores", "bf16", "tf32", "fp8", "fp4",
                                       "async_copy") if feats.get(name)]
            missing = [name for name in ("tensor_cores", "bf16", "tf32", "fp8", "fp4")
                       if not feats.get(name)]
            lines.append("           hardware: %s%s" %
                         (", ".join(names) or "none of the gated features",
                          "" if not missing else " (no %s)" % ", ".join(missing)))
        lines.append("           Kindi can use: %s" % ", ".join(row.get("kindi") or []))
    cpu = report.get("cpu") or {}
    lines.append("CPU: %s, %s physical / %s logical threads" %
                 (cpu.get("model"), cpu.get("physical"), cpu.get("logical")))
    feats = cpu.get("features") or {}
    lines.append("     " + ", ".join("%s=%s" % (label, _tri(feats.get(name)))
                                     for name, label in
                                     (("avx", "AVX"), ("avx2", "AVX2"), ("fma", "FMA3"),
                                      ("f16c", "F16C"))))
    extras = []
    if cpu.get("l3_kb"):
        extras.append("L3 %s MiB" % (cpu["l3_kb"] // 1024))
    if cpu.get("l2_kb"):
        extras.append("L2 %s MiB" % (cpu["l2_kb"] // 1024))
    extras.append("AVX-512=%s" % _tri(cpu.get("avx512")))
    extras.append("AMX=%s" % _tri(cpu.get("amx")))
    lines.append("     " + ", ".join(extras))
    lines.append("     Kindi CPU path: %s (build %s), %s threads recommended" %
                 (cpu.get("tier_label"), cpu.get("build"), cpu.get("threads_recommended")))
    tc = cpu.get("toolchain") or {}
    if tc.get("path"):
        lines.append("     CPU engine toolchain: %s" % tc["path"])
    elif tc:
        lines.append("     CPU engine toolchain: %s not found on PATH (set KINDI_CC, and KINDI_LLVM on "
                     "Windows) - the CPU engine and the tandem split cannot be built here" % tc.get("name"))
    if cpu.get("features_known") is False:
        lines.append("     note: this OS exposes no CPU flags to Python; the build compiler probe "
                     "(kindi/cpu.py, KINDI_CC) is the authority, and KINDI_CPU_ISA can name the tier")
    else:
        missing = [label for name, label in (("fma", "FMA3"), ("f16c", "F16C"))
                   if (cpu.get("features") or {}).get(name) is None]
        if missing:
            lines.append("     note: %s not exposed by this OS; the build compiler probe "
                         "(kindi/cpu.py, KINDI_CC) answers them and picks the tier" % "/".join(missing))
    if cpu.get("avx512"):
        lines.append("     note: AVX-512 is present but no Kindi kernel uses it yet (a named roadmap gate)")
    return lines


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(prog="kindi caps", description="what each card and this CPU can do")
    p.add_argument("--json", action="store_true", help="print the whole report as JSON")
    a = p.parse_args(argv)
    report = probe()
    if a.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for line in render(report):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
