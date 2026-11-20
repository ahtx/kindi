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


def _cgroup_limit(root=None):
    """The effective RAM limit of the cgroup the process runs in, or None when unknown/unlimited.

    A container's physical RAM is not the budget Kindi should plan against: the process can be
    killed when it crosses the cgroup limit even though the host has more free memory. Read both
    the cgroup v2 (memory.max) and v1 (memory/memory.limit_in_bytes) files and return the most
    restrictive positive value. `max`/huge sentinels mean unlimited.
    """
    root = root or "/sys/fs/cgroup"
    limits = []
    for relpath in ("memory.max", "memory/memory.limit_in_bytes"):
        try:
            with open(os.path.join(root, relpath)) as f:
                text = f.read().strip()
        except (OSError, ValueError):
            continue
        if text == "max":
            continue
        try:
            value = int(text)
        except (TypeError, ValueError):
            continue
        if value > 0 and value != 9223372036854771712:
            limits.append(value)
    return min(limits) if limits else None


def _cgroup_usage(root=None):
    """RAM the process's cgroup already charges in bytes, or None when unknown.

    A container's limit is only half the story: once the cgroup has filled most of it, the room
    left for a load is limit minus current usage, whatever the host's free memory says. Read both
    the cgroup v2 (memory.current) and v1 (memory/memory.usage_in_bytes) counters and return the
    largest value seen, the pessimistic reading when both files are present.
    """
    root = root or "/sys/fs/cgroup"
    usages = []
    for relpath in ("memory.current", "memory/memory.usage_in_bytes"):
        try:
            with open(os.path.join(root, relpath)) as f:
                text = f.read().strip()
        except (OSError, ValueError):
            continue
        try:
            value = int(text)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            usages.append(value)
    return max(usages) if usages else None


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
    """physical RAM in bytes (MemTotal, or GlobalMemoryStatusEx on native Windows), capped at the
    cgroup limit when the process runs in a memory-limited container; None when unknown"""
    w = _windows()
    base = w[0] if w else _meminfo("MemTotal")
    limit = _cgroup_limit()
    if base and limit:
        return min(base, limit)
    return base or limit


def available():
    """free RAM in bytes (MemAvailable, or GlobalMemoryStatusEx on native Windows), or None when
    unknown. A cgroup limit caps the value and the RAM the cgroup already uses is subtracted:
    MemAvailable is the host's figure and says nothing about the room left in the container's own
    budget. Returns 0 when the cgroup has no room left."""
    w = _windows()
    base = w[1] if w else _meminfo("MemAvailable")
    limit = _cgroup_limit()
    if base and limit:
        usage = _cgroup_usage()
        room = limit - usage if usage is not None else limit
        return min(base, max(0, room))
    return base


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
