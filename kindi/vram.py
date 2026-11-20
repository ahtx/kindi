"""Sampled-peak VRAM monitoring for `kindi bench --sample-vram`.

Device-wide `memory.used` from nvidia-smi (so the peak includes every
allocation, not just the CuPy pool), sampled on a background thread while
the benchmark runs. Free-after-load numbers alone cannot prove a
larger-than-VRAM split fits; a sampled peak can. The row read is the one
for the device index the engine runs on: nvidia-smi lists every GPU on the
host, so the bare first row can be the wrong card on a multi-GPU machine
(the installer gives each server its own CUDA_VISIBLE_DEVICES).
"""
import os
import subprocess
import threading
import time

SMI_CMD = ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free",
           "--format=csv,noheader,nounits"]


def parse_smi(text, device=0):
    """Parse the `index, used, total, free` row for `device` from nvidia-smi csv output.

    Legacy rows without an index column (one GPU) are accepted as device 0."""
    rows = []
    for ln in str(text).splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) < 3:
            continue
        try:
            vals = [int(x) for x in parts]
        except ValueError:
            continue
        rows.append(vals)
    for vals in rows:
        if len(vals) == 4:
            idx, used, total, free = vals
            if idx == device:
                return {"used_mib": used, "total_mib": total, "free_mib": free}
        elif len(vals) >= 3 and device == 0:
            used, total, free = vals[:3]
            return {"used_mib": used, "total_mib": total, "free_mib": free}
    return None


def visible_device(device):
    """Map a CuPy device ordinal to the physical nvidia-smi index.

    CUDA_VISIBLE_DEVICES lists the physical cards the process sees, in order;
    without it the ordinal is the physical index."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return device
    cards = [x.strip() for x in vis.split(",") if x.strip() not in ("", "Void")]
    if 0 <= device < len(cards):
        try:
            return int(cards[device])
        except ValueError:
            return device
    return device


def gpu_memory(runner=None, device=0):
    """Device-wide VRAM of one GPU from nvidia-smi, or None when it cannot be read."""
    try:
        if runner is not None:
            out = runner(SMI_CMD)
        else:
            out = subprocess.run(SMI_CMD, capture_output=True, text=True, timeout=5)
            out = out.stdout if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return parse_smi(out, device=visible_device(device)) if out is not None else None


class VramMonitor(object):
    """Sample device-wide VRAM while work runs; report the peak and its delta.

    `sampler` is injectable so tests can drive the monitor without a GPU.
    """

    def __init__(self, interval=0.5, sampler=None, device=0):
        self.interval = max(0.05, float(interval))
        self.device = int(device)
        self.sampler = sampler or (lambda: gpu_memory(device=self.device))
        self.samples = []
        self.phase = "startup"
        self.baseline = None
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.perf_counter()

    def _sample(self, phase=None):
        mem = self.sampler()
        if mem is None:
            return None
        rec = dict(mem)
        rec["at_s"] = round(time.perf_counter() - self._t0, 3)
        rec["phase"] = phase or self.phase
        self.samples.append(rec)
        return rec

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._t0 = time.perf_counter()
        self.baseline = self._sample("startup")
        self._thread = threading.Thread(target=self._run, name="vram-monitor", daemon=True)
        self._thread.start()

    def mark(self, phase):
        self.phase = phase
        self._sample(phase)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._sample("stop")

    def summary(self):
        if not self.samples:
            return {"available": False, "interval_s": self.interval}
        peak = max(self.samples, key=lambda r: r["used_mib"])
        base = self.baseline or self.samples[0]
        return {
            "available": True,
            "source": "nvidia-smi memory.used (device-wide, GPU %d)" % self.device,
            "interval_s": self.interval,
            "baseline_used_mib": base["used_mib"],
            "peak_used_mib": peak["used_mib"],
            "peak_delta_mib": peak["used_mib"] - base["used_mib"],
            "peak_free_mib": peak["free_mib"],
            "peak_phase": peak["phase"],
            "sample_count": len(self.samples),
        }
