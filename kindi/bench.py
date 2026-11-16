"""kindi bench: generation speed at several context positions, prompt-reading speed, and GPU memory.
With KINDI_DEVICE=cpu (or without CuPy) it times the CPU engine with the host clock."""
import os
import time


def _cupy():
    if os.environ.get("KINDI_DEVICE", "gpu") == "cpu":
        return None
    try:
        import cupy as cp
        return cp
    except ImportError:
        return None


def run(cfg, log=print):
    import numpy as np
    cp = _cupy()
    from .engine import make_engine
    pack = cfg["pack"]
    mp = os.path.join(pack, "kindi_pack.json")
    man = __import__("json").load(open(mp)) if os.path.exists(mp) else {}
    if man.get("backend") == "kindi_nanbeige":
        if cp is None:
            log("kindi bench: the Nanbeige backend runs only on an NVIDIA GPU (KINDI_DEVICE=cpu is set or CuPy is missing)")
            return 2
        from .nanbeige_cli import bench
        return bench(cfg, man, log=log)
    if man.get("backend") == "kindi_lightning":
        if cp is None:
            log("kindi bench: the Lightning backend runs only on an NVIDIA GPU (KINDI_DEVICE=cpu is set or CuPy is missing)")
            return 2
        from .lightning_cli import bench
        return bench(cfg, man, log=log)
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=cfg["ctx"], clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log)
    if cp is not None:
        free, total = cp.cuda.runtime.memGetInfo()
        log("GPU memory: %.0f MiB free of %.0f with a %d-token context" % (free / 2 ** 20, total / 2 ** 20, eng.ctx))
        sync = cp.cuda.Device().synchronize
    else:
        log("CPU engine: %s threads, %d-token context" % (os.environ.get("OMP_NUM_THREADS", os.cpu_count()), eng.ctx))
        sync = lambda: None                    # noqa: E731 - the CPU engine returns when the work is done
    eng.reset()

    def timed(fn, n):
        fn(0)                                  # warm-up (first launches compile/load kernels)
        sync()
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        sync()
        return (time.perf_counter() - t0) / n * 1000

    ms = timed(lambda i: eng.feed(1000 + i, 10 + i), 20)
    log("prompt reading:       %5.1f tokens/s" % (1000 / ms))
    modes = [("exact", 0)] + ([("fast", cfg["fast_clusters"])] if eng.hsl is not None else [])
    for name, M in modes:
        eng.shortlist = M
        for p in [x for x in (100, 2000, 8000, 16000, 24000) if x + 20 < eng.ctx]:
            ms = timed(lambda i: eng.step(2000 + i, p + i), 10)
            log("generation (%s head) at position %5d: %5.1f tokens/s" % (name, p, 1000 / ms))
    eng.shortlist = 0
    return 0
