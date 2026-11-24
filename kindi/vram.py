"""The VRAM peak during `kindi bench --sample-vram` (PR #51, atimelessrage).

Free memory after the load does not show the peak of the run: the prompt pass and the plans take scratch memory and
give it back. A thread reads each card's device-wide memory.used from nvidia-smi (all allocations, not only CuPy's
pool) every interval seconds and keeps the peak. The cards are the ones this process sees, matched by PCI bus id:
CUDA_VISIBLE_DEVICES and CUDA's own device order both renumber cards, nvidia-smi's index does not."""
import subprocess
import threading
import time

QUERY = "--query-gpu=pci.bus_id,memory.used,memory.total,memory.free"


def _bus(b):
    """a PCI bus id in one form: nvidia-smi prints 00000000:01:00.0, CuPy 0000:01:00.0"""
    b = str(b).strip().lower()
    dom, _, rest = b.partition(":")
    return "%04x:%s" % (int(dom, 16), rest) if rest else b


def parse(text):
    """{bus id: (used, total, free) MiB} from nvidia-smi's csv rows (noheader, nounits)"""
    out = {}
    for ln in str(text or "").splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) != 4:
            continue
        try:
            out[_bus(f[0])] = tuple(int(float(x)) for x in f[1:])
        except ValueError:
            continue
    return out


def smi(runner=None):
    """nvidia-smi's rows, or {} when it is missing or fails"""
    try:
        if runner is not None:
            return parse(runner(["nvidia-smi", QUERY, "--format=csv,noheader,nounits"]))
        r = subprocess.run(["nvidia-smi", QUERY, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=5)
        return parse(r.stdout) if r.returncode == 0 else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def visible_buses(cp):
    """the PCI bus ids of the CUDA devices this process sees, in CUDA order"""
    out = []
    for i in range(cp.cuda.runtime.getDeviceCount()):
        try:
            out.append(_bus(cp.cuda.Device(i).pci_bus_id))
        except Exception:                        # noqa: BLE001 - a card CuPy cannot name is left out
            pass
    return out


class Monitor(object):
    """Samples the cards in `buses` on a thread. sample() -> {bus: (used, total, free)} can be given for tests."""

    def __init__(self, buses, interval=0.5, sample=None):
        self.buses = list(buses)
        self.interval = max(0.05, float(interval))
        self.sample = sample or smi
        self.base, self.peak = {}, {}             # bus -> (used, total, free); peak -> (used, total, free, phase)
        self.phase = "load"
        self.n = 0
        self._stop = threading.Event()
        self._t = None

    def _take(self):
        rows = self.sample()
        for b in self.buses:
            if b not in rows:
                continue
            self.n += 1
            self.base.setdefault(b, rows[b])
            if b not in self.peak or rows[b][0] > self.peak[b][0]:
                self.peak[b] = rows[b] + (self.phase,)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._take()

    def start(self):
        self._take()
        self._t = threading.Thread(target=self._loop, name="kindi-vram", daemon=True)
        self._t.start()
        return self

    def mark(self, phase):
        self._take()
        self.phase = phase

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5)
        self._take()

    def lines(self):
        """one log line per card"""
        if not self.peak:
            return ["VRAM peak: not available (nvidia-smi gave no row for this process's cards)"]
        return ["VRAM peak on %s: %d MiB used of %d (%+d MiB over the start), %d MiB free at the peak, during %s; "
                "%d samples every %.2f s" % (b, u, t, u - self.base[b][0], f, ph, self.n, self.interval)
                for b, (u, t, f, ph) in sorted(self.peak.items())]
