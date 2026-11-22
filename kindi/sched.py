"""Blocking sync: while the host waits for the card (event, stream or context synchronize, a blocking copy), the
thread sleeps instead of spinning a CPU core. Without it every generating server keeps one core at 100% - with seven
servers on a 4-core host, the waits stretch. On by default (the output is the same); KINDI_BLOCKING_SYNC=0 keeps the
driver default (spin).

The NVIDIA 580 driver (the first with CUDA 13.0) wakes the sleeping thread late: blocking sync cost 5 to 15% of the
write speed there (issue #30), against 1 to 3% on the 470 driver. Without KINDI_BLOCKING_SYNC set, it is off on 580+
drivers. An explicit KINDI_BLOCKING_SYNC=1 or =0 always wins."""
import ctypes
import os

CU_CTX_SCHED_BLOCKING_SYNC = 4
CU_EVENT_BLOCKING_SYNC = 1
SPIN_FROM_CUDA = 13000      # cuDriverGetVersion of the 580 driver (CUDA 13.0) and later
_done = [False]


def default_on(cuda_version):
    """the default without KINDI_BLOCKING_SYNC: on before the 580 driver (cuDriverGetVersion < 13000), off from it"""
    return not cuda_version or cuda_version < SPIN_FROM_CUDA


def _kfd_gfx():
    """the gfx target versions of the AMD GPUs in the KFD topology (110501 = gfx1151)"""
    out = set()
    try:
        root = "/sys/class/kfd/kfd/topology/nodes"
        for n in os.listdir(root):
            for line in open(os.path.join(root, n, "properties")):
                if line.startswith("gfx_target_version"):
                    out.add(int(line.split()[1]))
    except OSError:
        pass
    return out


def blocking_sync(log=print):
    """set CU_CTX_SCHED_BLOCKING_SYNC on the primary context of every visible card (once per process)"""
    if _done[0]:
        return
    _done[0] = True
    if os.environ.get("KINDI_BLOCKING_SYNC") not in (None, "1"):
        log("blocking sync: off (KINDI_BLOCKING_SYNC=%s)" % os.environ["KINDI_BLOCKING_SYNC"])
        return
    from . import gpudrv
    if gpudrv.HIP:
        # Strix Halo (gfx1151): the blocking waits woke late and at random (9B write 21-41 tokens/s against 51.6
        # spinning); spin there, unless KINDI_BLOCKING_SYNC=1 is set. The Radeon PRO W6800 (gfx1030): the 2B writes
        # 240 -> 253 tokens/s spinning, steadier runs, the 27B 41.6 -> 42.2
        spin = _kfd_gfx() & {110501, 100300}
        if "KINDI_BLOCKING_SYNC" not in os.environ and spin:
            os.environ["KINDI_BLOCKING_SYNC"] = "0"
            log("blocking sync: off on %s (spin wait)" % ", ".join("gfx%d%d%x" % (g // 10000, g // 100 % 100, g % 100)
                                                                 for g in sorted(spin)))
            return
        # ROCm: no libcuda; hipSetDeviceFlags(hipDeviceScheduleBlockingSync) on every device
        try:
            hip = gpudrv.hip_lib()
            n = ctypes.c_int()
            hip.hipGetDeviceCount(ctypes.byref(n))
            for i in range(n.value):
                hip.hipSetDevice(i)
                hip.hipSetDeviceFlags(4)
            log("blocking sync: HIP blocking sync on %d device(s)" % n.value)
        except OSError as e:
            log("blocking sync: %s - driver default kept" % e)
        return
    try:
        cu = gpudrv.libcuda()
        if cu.cuInit(0) != 0:
            log("blocking sync: cuInit failed - driver default kept")
            return
        if "KINDI_BLOCKING_SYNC" not in os.environ:
            v = ctypes.c_int()
            cv = v.value if cu.cuDriverGetVersion(ctypes.byref(v)) == 0 else 0
            on = default_on(cv)
            os.environ["KINDI_BLOCKING_SYNC"] = "1" if on else "0"     # the events (tp, spec, chunk) follow it
            if not on:
                log("blocking sync: off by default on this driver (CUDA %d.%d, the 580 driver or later: the sleeping "
                    "thread wakes late); KINDI_BLOCKING_SYNC=1 turns it on" % (cv // 1000, cv % 1000 // 10))
                return
            log("blocking sync: on by default (driver CUDA %d.%d, before the 580 driver)" % (cv // 1000, cv % 1000 // 10))
        else:
            log("blocking sync: KINDI_BLOCKING_SYNC=1 (set)")
        n = ctypes.c_int()
        cu.cuDeviceGetCount(ctypes.byref(n))
        for i in range(n.value):
            d = ctypes.c_int()
            cu.cuDeviceGet(ctypes.byref(d), i)
            r = cu.cuDevicePrimaryCtxSetFlags(d, CU_CTX_SCHED_BLOCKING_SYNC)
            f, a = ctypes.c_uint(), ctypes.c_int()
            cu.cuDevicePrimaryCtxGetState(d, ctypes.byref(f), ctypes.byref(a))
            log("blocking sync: card %d primary context flags 0x%x (set: %d)" % (i, f.value, r))
    except OSError as e:
        log("blocking sync: %s - driver default kept" % e)
