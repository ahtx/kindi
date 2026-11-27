"""The CPU's physical cores, for Kindi's default thread counts.

Decode on the CPU is bound by memory bandwidth and by the vector units, which two SMT threads of one core share: one
thread per physical core is faster than one per logical CPU (Ryzen 9 7940HS, 8 cores / 16 threads: 11.5
tok/s with 8 threads, 10.0-10.8 with 16). So wherever Kindi picks a thread count or a CPU list by itself, it takes one
logical CPU per physical core (the lowest-numbered sibling), within the process's CPU affinity. The env settings
(OMP_NUM_THREADS, KINDI_FN_CPUS, KINDI_QX_CPUS, KINDI_MOE_CPUS0/1, ...) still override it.

Linux: /sys/devices/system/cpu/cpuN/topology (package and core id), else /proc/cpuinfo ("physical id", "core id").
Windows: psutil.cpu_count(logical=False) when psutil is there, else WMI (Win32_Processor.NumberOfCores).
macOS: the performance cores (sysctl hw.perflevel0.physicalcpu).
Otherwise: half the logical CPUs when SMT is known to be on, else all of them."""
import os
import subprocess
import sys

_SYS = "/sys/devices/system/cpu"


def parse_list(text):
    """"0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11]"""
    out = []
    for part in str(text).strip().split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def format_list(cpus):
    """[0, 1, 2, 3, 8, 10, 11] -> "0-3,8,10-11" """
    cpus = sorted(set(int(c) for c in cpus))
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else "%d-%d" % (cpus[i], cpus[j]))
        i = j + 1
    return ",".join(out)


def _allowed():
    """the logical CPUs this process may run on"""
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return list(range(os.cpu_count() or 1))


def _read(path):
    with open(path) as f:
        return f.read().strip()


def _topology(root=_SYS, cpuinfo="/proc/cpuinfo"):
    """{logical cpu: (package, core)} from sysfs, else from /proc/cpuinfo; {} when neither says"""
    topo = {}
    try:
        for name in os.listdir(root):
            if name.startswith("cpu") and name[3:].isdigit():
                t = os.path.join(root, name, "topology")
                try:
                    topo[int(name[3:])] = (int(_read(os.path.join(t, "physical_package_id"))),
                                           int(_read(os.path.join(t, "core_id"))))
                except (OSError, ValueError):
                    pass
    except OSError:
        pass
    if topo:
        return topo
    try:
        with open(cpuinfo) as f:
            text = f.read()
    except OSError:
        return {}
    for block in text.split("\n\n"):
        kv = {}
        for line in block.splitlines():
            k, _, v = line.partition(":")
            kv[k.strip()] = v.strip()
        if "processor" in kv and "core id" in kv:
            try:
                topo[int(kv["processor"])] = (int(kv.get("physical id", "0") or 0), int(kv["core id"]))
            except ValueError:
                pass
    return topo


def _node_cpus(root="/sys/devices/system/node"):
    """[[cpus of node 0], [cpus of node 1], ...] (nodes with no CPU left out); [] when unknown"""
    try:
        names = sorted((n for n in os.listdir(root) if n.startswith("node") and n[4:].isdigit()), key=lambda n: int(n[4:]))
    except OSError:
        return []
    out = []
    for n in names:
        try:
            cs = parse_list(_read(os.path.join(root, n, "cpulist")))
        except (OSError, ValueError):
            continue
        if cs:
            out.append(cs)
    return out


def _smt_on():
    """True / False when the host says whether SMT is on, None when it does not"""
    try:
        return _read(os.path.join(_SYS, "smt", "active")) == "1"
    except OSError:
        return None


def _windows_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:  # noqa: BLE001 - optional module
        pass
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out) if out.isdigit() and int(out) > 0 else None
    except Exception:  # noqa: BLE001 - WMI is optional
        return None


def physical_cpus(node=None, topo=None, allowed=None, nodes=None):
    """one logical CPU per physical core (the lowest sibling), within the affinity, sorted; node: only the cores of
    that NUMA node (its index among the nodes that have CPUs). [] when the topology is unknown."""
    topo = _topology() if topo is None else topo
    allowed = _allowed() if allowed is None else allowed
    if not topo:
        return []
    ok = set(allowed)
    if node is not None:
        nodes = _node_cpus() if nodes is None else nodes
        if node >= len(nodes):
            return []
        ok &= set(nodes[node])
    first = {}
    for c in sorted(ok):
        if c in topo:
            first.setdefault(topo[c], c)
    return sorted(first.values())


def _darwin_perf_cores():
    """macOS on Apple silicon: the performance cores (hw.perflevel0.physicalcpu); None elsewhere. The thread pool
    waits for its slowest thread on every op, so efficiency cores slow decode down instead of adding to it."""
    if sys.platform != "darwin":
        return None
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
            if out.isdigit() and int(out) > 0:
                return int(out)
        except Exception:  # noqa: BLE001 - sysctl is optional
            pass
    return None


def physical_cores():
    """the number of physical cores this process can use (at least 1)"""
    cs = physical_cpus()
    if cs:
        return len(cs)
    n = _darwin_perf_cores()
    if n:
        return n
    logical = len(_allowed())
    if os.name == "nt":
        n = _windows_cores()
        if n:
            return max(1, min(n, logical))
    smt = _smt_on()
    return max(1, logical // 2 if smt else logical)


def default_cpu_list(node=None):
    """the default CPU list ("0-19,..."): one logical CPU per physical core, of one NUMA node or of all"""
    cs = physical_cpus(node)
    if cs:
        return format_list(cs)
    allowed = _allowed()
    if node is not None:
        nodes = _node_cpus()
        if node < len(nodes):
            allowed = [c for c in allowed if c in set(nodes[node])]
        elif node > 0:
            return ""
    n = max(1, len(allowed) // 2) if _smt_on() else len(allowed)
    return format_list(allowed[:n])


def default_omp_threads():
    """OMP_NUM_THREADS for the dense CPU engines: the physical cores of the first NUMA node (on one socket: every
    physical core). The OpenMP pool runs one parallel region per token; across sockets its barriers cost more than
    the second socket's cores give (install.sh's CPU profile takes the cores of socket 0 too)."""
    cs = physical_cpus(0) or physical_cpus()
    return len(cs) if cs else physical_cores()


def set_omp_default(threads=None):
    """before the OpenMP library loads: OMP_NUM_THREADS = threads, or when unset, the physical cores
    (default_omp_threads) with OMP_PLACES=cores and OMP_PROC_BIND=close unless set. Returns the thread count."""
    if threads:
        os.environ["OMP_NUM_THREADS"] = str(int(threads))
    elif not os.environ.get("OMP_NUM_THREADS"):
        os.environ["OMP_NUM_THREADS"] = str(default_omp_threads())
        os.environ.setdefault("OMP_PLACES", "cores")
        os.environ.setdefault("OMP_PROC_BIND", "close")
    return int(os.environ["OMP_NUM_THREADS"])
