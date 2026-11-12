"""Host-RAM preflight for an engine load. Without swap, a host that runs short of RAM while a model loads can stall
for minutes (no ssh, no ping) instead of failing. A load that cannot fit stops at once with a message that says what
to do."""
import os

MARGIN = 1 << 30                       # the rest of the load (tokenizer, plans, Python) besides the weights


def _meminfo(key):
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _windows():
    """(total, available) bytes from GlobalMemoryStatusEx on native Windows, else None"""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return int(m.ullTotalPhys), int(m.ullAvailPhys)
    except (OSError, AttributeError, ValueError):
        return None


def total():
    """physical RAM in bytes (MemTotal, or GlobalMemoryStatusEx on native Windows), or None when unknown"""
    w = _windows()
    return w[0] if w else _meminfo("MemTotal")


def available():
    """free RAM in bytes (MemAvailable, or GlobalMemoryStatusEx on native Windows), or None when unknown"""
    w = _windows()
    return w[1] if w else _meminfo("MemAvailable")


def check(need, what, log=print, avail=None):
    """stop with a clear message when less than need + MARGIN bytes are free. KINDI_HOST_RAM_CHECK=0 skips it."""
    if os.environ.get("KINDI_HOST_RAM_CHECK", "1") != "1":
        return
    if avail is None:
        avail = available()
    if avail is None:
        return
    want = need + MARGIN
    log("host RAM for %s: %.1f GB needed, %.1f GB free" % (what, want / 2 ** 30, avail / 2 ** 30))
    if avail < want:
        raise SystemExit(
            "kindi: %s needs about %.1f GB of free host RAM to load, and %.1f GB is free.\n"
            "  To free RAM: stop the other model servers and large programs on this computer, then start Kindi again.\n"
            "  To load anyway (the computer can become slow or stop answering until the load ends): KINDI_HOST_RAM_CHECK=0"
            % (what, want / 2 ** 30, avail / 2 ** 30))
