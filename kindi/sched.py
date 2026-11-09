"""Blocking sync: while the host waits for the card (event, stream or context synchronize, a blocking copy), the
thread sleeps instead of spinning a CPU core. Without it every generating server keeps one core at 100% - with seven
servers on a 4-core host, the waits stretch. On by default (the output is the same); KINDI_BLOCKING_SYNC=0 keeps the
driver default (spin)."""
import ctypes
import os

CU_CTX_SCHED_BLOCKING_SYNC = 4
CU_EVENT_BLOCKING_SYNC = 1
_done = [False]


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
    if _done[0] or os.environ.get("KINDI_BLOCKING_SYNC", "1") != "1":
        return
    _done[0] = True
    from . import gpudrv
    if gpudrv.HIP:
        # Strix Halo (gfx1151): the blocking waits woke late and at random (9B write 21-41 tokens/s against 51.6
        # spinning); spin there, unless KINDI_BLOCKING_SYNC=1 is set
        if "KINDI_BLOCKING_SYNC" not in os.environ and _kfd_gfx() & {110501}:
            os.environ["KINDI_BLOCKING_SYNC"] = "0"
            log("blocking sync: off on gfx1151 (spin wait)")
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
        cu = ctypes.CDLL("libcuda.so.1")
        if cu.cuInit(0) != 0:
            log("blocking sync: cuInit failed - driver default kept")
            return
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
