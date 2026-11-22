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


# The memory cgroup (#40). /proc/meminfo shows the host's RAM inside a container, so a server in a container with
# --memory 14g sized its caches for the host's RAM and the kernel ended it at 14 GB. A parent cgroup's limit also
# applies (systemd slices, Kubernetes pods), so every cgroup from the process's own up to the mount root counts.
_CG_ROOT = "/sys/fs/cgroup"
_NO_LIMIT = 1 << 60                   # cgroup v1 reports "no limit" as a number near 2**63 (9223372036854771712)


def _cgroup_dirs(proc="/proc/self/cgroup", root=_CG_ROOT):
    """[(directory, v2)] of this process's memory cgroup, its own first, then each parent up to the mount root"""
    try:
        with open(proc) as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        v2 = parts[0] == "0" and parts[1] == ""
        if not v2 and "memory" not in parts[1].split(","):
            continue
        base = root if v2 else os.path.join(root, "memory")
        d = os.path.normpath(base + "/" + parts[2].lstrip("/"))
        while True:
            if os.path.isdir(d):
                out.append((d, v2))
            if len(d) <= len(base):
                break
            d = os.path.dirname(d)
        if not any(x[0] == base for x in out) and os.path.isdir(base):
            out.append((base, v2))    # in a container's cgroup namespace, the process's cgroup is the mount root
    return out


def _root_dirs(root):
    """the v2 and v1 file places under one cgroup directory (a test's root, or the mount root)"""
    return [(root, True), (os.path.join(root, "memory"), False)]


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cgroup_limit(root=None):
    """The effective RAM limit of the cgroup the process runs in, or None when unknown/unlimited.

    A container's physical RAM is not the budget Kindi should plan against: the process can be
    killed when it crosses the cgroup limit even though the host has more free memory. Read the
    cgroup v2 (memory.max) and v1 (memory.limit_in_bytes) files of the process's cgroup and of each
    parent, and return the most restrictive positive value. `max`/huge sentinels mean unlimited.
    With `root`, only that directory is read (memory.max and memory/memory.limit_in_bytes).
    """
    limits = []
    for d, v2 in (_root_dirs(root) if root else _cgroup_dirs()):
        text = _read(os.path.join(d, "memory.max" if v2 else "memory.limit_in_bytes"))
        if not text or text == "max":
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if 0 < value < _NO_LIMIT:
            limits.append(value)
    return min(limits) if limits else None


def _cgroup_usage(root=None):
    """RAM the process's cgroup already charges and cannot give back, in bytes, or None when unknown.

    A container's limit is only half the story: once the cgroup has filled most of it, the room
    left for a load is limit minus current usage, whatever the host's free memory says. Read the
    cgroup v2 (memory.current) and v1 (memory.usage_in_bytes) counters of the process's own cgroup
    and return the largest value seen, the pessimistic reading when both files are present. The
    inactive file pages in memory.stat (clean page cache, such as the mapped model file) are not
    counted: the kernel reclaims them before it ends the process, and `docker stats` leaves them out too.
    """
    if root:
        dirs = _root_dirs(root)
    else:                             # the process's own cgroup in each hierarchy (v2, and v1's memory controller)
        dirs, seen = [], set()
        for d, v2 in _cgroup_dirs():
            if v2 not in seen:
                seen.add(v2)
                dirs.append((d, v2))
    usages = []
    for d, v2 in dirs:
        text = _read(os.path.join(d, "memory.current" if v2 else "memory.usage_in_bytes"))
        try:
            value = int(text)
        except (TypeError, ValueError):
            continue
        key = "inactive_file" if v2 else "total_inactive_file"
        for line in (_read(os.path.join(d, "memory.stat")) or "").splitlines():
            k, _, v = line.partition(" ")
            if k == key:
                try:
                    value -= int(v)
                except ValueError:
                    pass
                break
        usages.append(max(0, value))
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
