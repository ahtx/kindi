# SPDX-License-Identifier: AGPL-3.0-only
"""Measured memory-tier rates behind `kindi io`: RAM copy, disk read, PCIe copies.

`kindi plan` and `kindi autotune` currently assume how fast this host copies in RAM, reads a cold
expert from disk, and moves bytes to the card. BigMoeOnEdge's `--auto` plans a MoE cache from probes
of exactly this class (run in a process of its own, because a driver allocation must not bias the
host numbers), and its benchmark reports flash I/O apart from compute. This module is the same idea
for Kindi: measure this machine, print the method next to the number, and never turn another
engine's number into ours.

The disk probe writes a file and reads it back unbuffered where the OS allows it (O_DIRECT on Linux,
FILE_FLAG_NO_BUFFERING on Windows), because a buffered read of a file this process just wrote
measures RAM, not the disk. Where neither is available the report says `buffered` instead of
pretending.
"""
import argparse
import json
import mmap
import os
import sys
import tempfile
import time

DEFAULT_MIB = 128
DISK_CHUNK = 4 * 2 ** 20          # O_DIRECT/NO_BUFFERING reads need a block-aligned chunk
VALID_PROBES = ("ram", "disk", "pcie")


def _bytes(mib):
    return max(1, int(mib)) * 2 ** 20


def _skip_names(skip):
    """Validate `--skip` names before any probe starts."""
    if isinstance(skip, str):
        skip = skip.split(",")
    names = {str(name).strip() for name in (skip or ())}
    names.discard("")
    unknown = sorted(names - set(VALID_PROBES))
    if unknown:
        raise ValueError("unknown probe%s: %s (valid probes: %s)" % (
            "" if len(unknown) == 1 else "s",
            ", ".join(unknown),
            ", ".join(VALID_PROBES),
        ))
    return names


def _median(values):
    values = sorted(values)
    return values[len(values) // 2] if len(values) % 2 else \
        0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2])


def ram_copy(mib=DEFAULT_MIB, repeats=5):
    """A memcpy rate: two buffers, the same copy a miss copies into a staging buffer."""
    n = _bytes(mib)
    src, dst = bytearray(n), bytearray(n)
    dst[:] = src                      # warm both buffers before timing
    times = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        dst[:] = src
        times.append(time.perf_counter() - t0)
    seconds = _median(times)
    return {"available": True, "mib": n / 2 ** 20, "seconds": round(seconds, 6),
            "mib_s": round((n / 2 ** 20) / seconds, 1) if seconds else None,
            "method": "memcpy into a second buffer, 1 thread"}


def _posix_direct_read(path, size):
    if not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"):
        raise OSError("O_DIRECT/preadv unavailable")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, min(DISK_CHUNK, size))
    view = memoryview(buf)
    off = 0
    try:
        t0 = time.perf_counter()
        while off < size:
            want = min(len(view), size - off)
            got = os.preadv(fd, [view[:want]], off)
            if got <= 0:
                break
            off += got
        seconds = time.perf_counter() - t0
    finally:
        view.release()
        buf.close()
        os.close(fd)
    return seconds, off


def _windows_nobuffer_read(path, size):
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                             ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    generic_read, share_read, open_existing = 0x80000000, 1, 3
    no_buffering, sequential = 0x20000000, 0x08000000
    handle = k32.CreateFileW(str(path), generic_read, share_read, None, open_existing,
                             no_buffering | sequential, None)
    if not handle or handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    chunk = min(DISK_CHUNK, size)
    buf = mmap.mmap(-1, chunk)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    read = wintypes.DWORD(0)
    off = 0
    try:
        t0 = time.perf_counter()
        while off < size:
            if not k32.ReadFile(handle, ctypes.c_void_p(addr), chunk, ctypes.byref(read), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if read.value == 0:
                break
            off += read.value
        seconds = time.perf_counter() - t0
    finally:
        k32.CloseHandle(handle)
        buf.close()
    return seconds, off


def _buffered_read(path, size):
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    off = 0
    with open(path, "rb", buffering=0) as f:
        t0 = time.perf_counter()
        while off < size:
            block = f.read(min(DISK_CHUNK, size - off))
            if not block:
                break
            off += len(block)
        seconds = time.perf_counter() - t0
    return seconds, off


def disk_read(mib=DEFAULT_MIB, directory=None):
    """Sequential disk read of a file this process just wrote, unbuffered when possible."""
    size = _bytes(mib)
    size = ((size + DISK_CHUNK - 1) // DISK_CHUNK) * DISK_CHUNK
    fd, path = tempfile.mkstemp(prefix="kindi-io-", suffix=".bin", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            block = b"\0" * 2 ** 20
            for _ in range(size // len(block)):
                f.write(block)
            f.flush()
            os.fsync(f.fileno())
        unbuffered = True
        try:
            if os.name == "nt":
                seconds, got = _windows_nobuffer_read(path, size)
                method = "FILE_FLAG_NO_BUFFERING"
            else:
                seconds, got = _posix_direct_read(path, size)
                method = "O_DIRECT"
        except OSError:
            unbuffered = False
            seconds, got = _buffered_read(path, size)
            method = "buffered (the page cache may inflate this)"
        mib_done = got / 2 ** 20
        return {"available": got > 0, "mib": mib_done, "seconds": round(seconds, 6),
                "mib_s": round(mib_done / seconds, 1) if seconds and got else None,
                "unbuffered": unbuffered, "method": method}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _cuda():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return None


def pcie_copy(mib=64, repeats=5):
    """Pinned-host <-> device copy rates through the active CUDA device."""
    cp = _cuda()
    if cp is None:
        return {"available": False, "reason": "CuPy/CUDA is not available"}
    n = _bytes(min(int(mib), 512))
    rt = cp.cuda.runtime
    host = dev = None
    try:
        host = cp.cuda.alloc_pinned_memory(n)      # engine.py allocates its staging the same way
        dev = rt.malloc(n)
        h2d, d2h = [], []
        for i in range(max(1, int(repeats)) + 1):
            t0 = time.perf_counter()
            rt.memcpy(dev, int(host.ptr), n, rt.memcpyHostToDevice)
            if i:
                h2d.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            rt.memcpy(int(host.ptr), dev, n, rt.memcpyDeviceToHost)
            if i:
                d2h.append(time.perf_counter() - t0)
        name = rt.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        up, down = _median(h2d), _median(d2h)
        gib = n / 2 ** 30
        return {"available": True, "mib": n / 2 ** 20, "device": name,
                "h2d_gib_s": round(gib / up, 2), "d2h_gib_s": round(gib / down, 2),
                "method": "cudaMemcpy, pinned host, %d timed runs" % max(1, int(repeats))}
    except Exception as exc:                       # a driver/toolkit mismatch must not abort the report
        return {"available": False, "reason": str(exc)}
    finally:
        if dev is not None:
            rt.free(dev)
        # the pinned pool owns host, as in the engine


def probe(mib=DEFAULT_MIB, disk_dir=None, skip=()):
    skip = _skip_names(skip)
    out = {}
    out["ram"] = {"available": False, "reason": "skipped"} if "ram" in skip else ram_copy(mib)
    out["disk"] = ({"available": False, "reason": "skipped"} if "disk" in skip
                   else disk_read(mib, directory=disk_dir))
    out["pcie"] = ({"available": False, "reason": "skipped"} if "pcie" in skip
                   else pcie_copy(min(int(mib), 64)))
    return out


def save_report(report, path):
    """Write a probe report that `load_or_probe()` and `kindi plan` can reuse."""
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return path


def read_report(path):
    """Read and shape-check a saved `kindi io` report."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise ValueError("the report is not a JSON object")
    if not any(name in report for name in VALID_PROBES):
        raise ValueError("the report has no ram, disk or pcie section")
    for name in VALID_PROBES:
        if name in report and not isinstance(report[name], dict):
            raise ValueError("the report's %s section is not an object" % name)
    return report


def rates(report):
    """Numeric rates from a `probe()` report, in bytes/s where they were measured.

    A buffered disk read is reported but not exposed as `disk_bytes_s`: it is a page-cache rate,
    not the device's, so a sizing model must not use it as a disk assumption.
    """
    report = report or {}
    out = {"source": "kindi io"}
    ram = report.get("ram") or {}
    if ram.get("available") and ram.get("mib_s"):
        out["ram_mib_s"] = float(ram["mib_s"])
        out["ram_bytes_s"] = out["ram_mib_s"] * 2 ** 20
    disk = report.get("disk") or {}
    if disk.get("available") and disk.get("mib_s"):
        out["disk_mib_s"] = float(disk["mib_s"])
        if disk.get("unbuffered", True):
            out["disk_bytes_s"] = out["disk_mib_s"] * 2 ** 20
        else:
            out["disk_buffered"] = True
    pcie = report.get("pcie") or {}
    if pcie.get("available"):
        for key, out_key in (("h2d_gib_s", "pcie_h2d_gib_s"), ("d2h_gib_s", "pcie_d2h_gib_s")):
            if pcie.get(key) is not None:
                out[out_key] = float(pcie[key])
                if key == "h2d_gib_s":
                    out["pcie_bytes_s"] = out[out_key] * 2 ** 30
        if pcie.get("device"):
            out["pcie_device"] = pcie["device"]
    return out if len(out) > 1 else {}


def load_rates(path):
    """Normalize a saved `kindi io --json FILE` report into the rates model."""
    out = rates(read_report(path))
    if not out:
        raise ValueError("the report has no usable numeric rate")
    return out


def load_or_probe(path=None, mib=DEFAULT_MIB, log=None, label="kindi"):
    """Use a saved report when it has rates; otherwise measure, with a visible reason.

    `path` is the explicit `--io-report`/`KINDI_IO_REPORT` value. With no path, the
    old behavior remains: measure once. The log line names the fallback and the
    command that saves a report for the next run.
    """
    if path:
        try:
            out = load_rates(path)
        except (OSError, ValueError, TypeError) as exc:
            if log:
                why = "no saved I/O report" if not os.path.exists(path) else "unusable saved I/O report"
                log("%s: %s at %s (%s); measuring RAM/disk/PCIe now" %
                    (label, why, path, exc))
        else:
            if log:
                log("%s: using saved I/O report at %s (%s)" %
                    (label, path, rate_line(out)))
            return out
    elif log:
        log("%s: no saved I/O report; measuring RAM/disk/PCIe now (save one with `kindi io --json FILE`)"
            % label)
    try:
        return rates(probe(mib=mib))
    except Exception as exc:                       # a probe failure must not abort a plan
        if log:
            log("%s: I/O probe failed (%s); slot sizing uses its default" % (label, exc))
        return None


def rate_line(rates):
    """A compact human-readable list of the measured assumptions."""
    if not rates:
        return "not measured"
    parts = []
    if rates.get("ram_mib_s") is not None:
        parts.append("RAM %.1f MiB/s" % rates["ram_mib_s"])
    if rates.get("pcie_h2d_gib_s") is not None:
        parts.append("PCIe H2D %.1f GiB/s" % rates["pcie_h2d_gib_s"])
    if rates.get("disk_mib_s") is not None:
        suffix = " (buffered, not used)" if rates.get("disk_buffered") else ""
        parts.append("disk %.1f MiB/s%s" % (rates["disk_mib_s"], suffix))
    return ", ".join(parts) if parts else "not measured"


def _rate(value, unit):
    return "%.1f %s" % (value, unit) if value is not None else "-"


def render(report):
    lines = []
    ram = report.get("ram") or {}
    if ram.get("available"):
        lines.append("RAM copy:  %-10s (%d MiB, %s)" %
                     (_rate(ram["mib_s"], "MiB/s"), ram["mib"], ram["method"]))
    else:
        lines.append("RAM copy:  unavailable - %s" % ram.get("reason", "no result"))
    disk = report.get("disk") or {}
    if disk.get("available"):
        lines.append("disk read: %-10s (%d MiB, %s)" %
                     (_rate(disk["mib_s"], "MiB/s"), disk["mib"], disk["method"]))
        if not disk.get("unbuffered"):
            lines.append("           the read was buffered: treat this as a cache/RAM rate, not the device's")
    else:
        lines.append("disk read: unavailable - %s" % disk.get("reason", "no result"))
    pcie = report.get("pcie") or {}
    if pcie.get("available"):
        lines.append("PCIe:      H2D %-8s D2H %-8s (%d MiB, pinned, %s)" %
                     (_rate(pcie["h2d_gib_s"], "GiB/s"), _rate(pcie["d2h_gib_s"], "GiB/s"),
                      pcie["mib"], pcie["device"]))
    else:
        lines.append("PCIe:      unavailable - %s" % pcie.get("reason", "no result"))
    lines.append("")
    lines.append("These are this host's numbers. Slot sizing and any cold-tier (disk) streaming plan "
                 "read them; no other engine's rate is ours.")
    return lines


def run(mib=DEFAULT_MIB, disk_dir=None, skip="", as_json=False, save_path=None):
    skip = [s.strip() for s in str(skip or "").split(",") if s.strip()]
    try:
        report = probe(mib=mib, disk_dir=disk_dir, skip=skip)
    except ValueError as exc:
        print("kindi io: %s" % exc, file=sys.stderr)
        return 2
    if isinstance(as_json, str) and not save_path:
        save_path = as_json
        as_json = True
    if save_path:
        try:
            save_report(report, save_path)
        except OSError as exc:
            print("kindi io: could not write %s (%s)" % (save_path, exc), file=sys.stderr)
            return 2
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for line in render(report):
            print(line)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kindi io",
                                 description="measure this host's memory-tier rates (RAM copy, disk read, PCIe copies)")
    ap.add_argument("--mib", type=int, default=DEFAULT_MIB,
                    help="buffer/file size per probe in MiB (default %d)" % DEFAULT_MIB)
    ap.add_argument("--disk-dir", default=None,
                    help="where to create the temporary disk-probe file (default: the system temp dir)")
    ap.add_argument("--skip", default="", help="comma-separated probes to skip: ram,disk,pcie")
    ap.add_argument("--json", nargs="?", const=True, default=False, metavar="FILE",
                    help="print the report as JSON, or write it to FILE for plan/autotune reuse")
    a = ap.parse_args(argv)
    return run(a.mib, a.disk_dir, a.skip, a.json)


if __name__ == "__main__":
    sys.exit(main())
