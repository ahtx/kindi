"""kindi - command line.

  kindi convert MODEL.gguf      build the weight pack from the PTQ1_0 GGUF (once; ~1 minute)
  kindi fast-head               build the cluster file for the fast head (optional; ~2 minutes)
  kindi verify                  check greedy output against llama.cpp's reference (token for token)
  kindi bench                   measure speed and GPU memory
  kindi serve                   run the OpenAI-compatible server
  kindi chat                    chat in the terminal (no server needed)
  kindi head exact|fast         choose the output head of a running server (next request, no restart)
  kindi info                    show the configuration, the pack and the GPU
Global options: --config FILE, --pack DIR, --state-dir DIR, --ctx N."""
import argparse
import json
import os
import sys
import time

from . import __version__
from . import config as C


def log(msg):
    sys.stdout.write("[kindi %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def _setup_gpu_env(cfg):
    """Kernel caches under state_dir, set before CUDA starts: CuPy's (CUDA C -> PTX) and the driver's JIT cache
    (PTX -> sm_30 code, ~1 minute on a K420 - without a writable cache the driver redoes it on every start)."""
    kdir = os.path.join(cfg["state_dir"], "kernels")
    jdir = os.path.join(cfg["state_dir"], "kernels", "driver")
    os.makedirs(jdir, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", kdir)
    os.environ.setdefault("CUDA_CACHE_PATH", jdir)
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(512 * 1024 * 1024))


def cmd_convert(cfg, a):
    if not a.gguf:
        raise SystemExit("kindi convert: give the model GGUF")
    from .convert import convert
    out = a.out or cfg["pack"]
    if os.path.exists(os.path.join(out, "kindi_pack.json")) and not a.force:
        raise SystemExit("kindi: %s already holds a pack (use --force to rebuild)" % out)
    manifest = convert(a.gguf, out, log=log)
    if manifest.get("backend") == "kindi_gemma4":
        log("Gemma 4 model ready for Kindi native CUDA inference")
    elif manifest.get("backend") == "kindi_lightning":
        log("Lightning model ready; validate with: kindi verify")
    else:
        log("next: kindi verify")
    return 0


def cmd_fast_head(cfg, a):
    _setup_gpu_env(cfg)
    from .lightning_cli import selected
    if selected(cfg):raise SystemExit("kindi: Lightning supports the exact full-vocabulary head")
    manifest_path=os.path.join(cfg["pack"],"kindi_pack.json")
    manifest=json.load(open(manifest_path)) if os.path.isfile(manifest_path) else {}
    if manifest.get("backend")=="kindi_gemma4":
        from .gemma4 import build_fast_head
        build_fast_head(manifest["model_path"],cfg["pack"],K=a.clusters,iters=a.iters,log=log)
    else:
        from .cluster import build
        build(cfg["pack"], K=a.clusters, iters=a.iters, log=log)
    log("restart `kindi serve`, then `kindi head fast` to use it")
    return 0


def cmd_verify(cfg, a):
    _setup_gpu_env(cfg)
    from .lightning_cli import selected, verify
    if selected(cfg):return verify(cfg,log=log)
    from .verify import run
    return run(cfg, log=log)


def cmd_bench(cfg, a):
    _setup_gpu_env(cfg)
    from .lightning_cli import selected, bench
    if selected(cfg):return bench(cfg,log=log)
    from .bench import run
    return run(cfg, log=log)


def cmd_serve(cfg, a):
    _setup_gpu_env(cfg)
    from .server import serve
    serve(cfg, log=log)
    return 0


def cmd_chat(cfg, a):
    _setup_gpu_env(cfg)
    from .lightning_cli import selected, chat
    if selected(cfg):return chat(cfg,a,log=log)
    from .server import Kindi
    K = Kindi(cfg, log=log)
    msgs = []
    if a.system:
        msgs.append({"role": "system", "content": a.system})
    print("Kindi %s - type a message, /reset to start over, /exit to quit." % __version__)
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip() in ("/exit", "/quit"):
            return 0
        if line.strip() == "/reset":
            msgs = msgs[:1] if a.system else []
            continue
        if not line.strip():
            continue
        msgs.append({"role": "user", "content": line})
        sys.stdout.write("kindi> ")
        reply = ""
        for kind, val in K.generate({"messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens}):
            if kind == "text":
                reply += val
                sys.stdout.write(val)
                sys.stdout.flush()
            else:
                sys.stdout.write("\n  (%.1f tokens/s)" % val["timing"]["gen_tok_s"])
        msgs.append({"role": "assistant", "content": reply})


def cmd_head(cfg, a):
    from .lightning_cli import selected
    if selected(cfg):
        if a.mode!='exact':raise SystemExit("kindi: Lightning supports the exact full-vocabulary head")
        log("Lightning head: exact");return 0
    os.makedirs(cfg["state_dir"], exist_ok=True)
    if a.mode == "fast" and not os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz")):
        raise SystemExit("kindi: the fast head needs the cluster file - run: kindi fast-head")
    with open(os.path.join(cfg["state_dir"], "head_mode"), "w") as f:
        f.write(a.mode + "\n")
    log("head mode: %s (the next request uses it)" % a.mode)
    return 0


def cmd_models(cfg, a):
    from .catalog import list_models
    list_models(cfg, log=log)
    return 0


def cmd_use(cfg, a):
    from .catalog import use
    return use(cfg, a.model, gguf=a.gguf, log=log)


def cmd_info(cfg, a):
    print("Kindi %s" % __version__)
    print("config: %s" % cfg["source"])
    for k in sorted(C.DEFAULTS):
        v = cfg[k]
        print("  %-17s %s" % (k, "(set)" if k == "api_key" and v else v))
    man = os.path.join(cfg["pack"], "kindi_pack.json")
    if os.path.exists(man):
        m = json.load(open(man))
        print("pack: %s - %s, from %s" % (cfg["pack"], m.get("model"), m.get("source_file")))
        if m.get("backend") == "kindi_gemma4":
            print("backend: Kindi Gemma 4 CUDA")
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"],"head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
        elif m.get("backend") == "kindi_lightning":
            print("backend: native Kindi Lightning CUDA; exact head; FP32 state/KV")
        else:
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
    else:
        print("pack: none at %s (kindi convert MODEL.gguf)" % cfg["pack"])
    try:
        import cupy as cp
        d = cp.cuda.Device()
        free, total = cp.cuda.runtime.memGetInfo()
        name = cp.cuda.runtime.getDeviceProperties(d.id)["name"]
        name = name.decode() if isinstance(name, bytes) else name
        print("GPU: %s, compute %s, %.0f of %.0f MiB free" % (name, d.compute_capability, free / 2 ** 20, total / 2 ** 20))
    except Exception as e:  # noqa: BLE001
        print("GPU: not available (%s)" % e)
    return 0


def _malloc_tuning():
    """Checkpoints are dozens of 2 MB host arrays per state copy. glibc raises its mmap threshold as large blocks are
    freed, after which such arrays come from the heap of each thread and are never given back (a 9B server grew by
    ~60 MB per request). A fixed 1 MB threshold keeps every such array in its own mapping, returned when freed."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").mallopt(-3, 1 << 20)       # M_MMAP_THRESHOLD = -3
    except (OSError, AttributeError):
        pass                                               # not glibc: nothing to tune


def main(argv=None):
    _malloc_tuning()
    p = argparse.ArgumentParser(prog="kindi", description="Kindi %s - native CUDA inference for ternary models and Gemma 4" % __version__)
    p.add_argument("--config")
    p.add_argument("--pack")
    p.add_argument("--state-dir", dest="state_dir")
    p.add_argument("--ctx", type=int)
    p.add_argument('--lightning-kv',choices=('fp32','fp16','q8_0'))
    p.add_argument("--version", action="version", version="kindi " + __version__)
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("convert", help="build the weight pack from the GGUF")
    s.add_argument("gguf", nargs="?")
    s.add_argument("--out", help="pack directory (default: the configured pack)")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("fast-head", help="build the fast-head cluster file")
    s.add_argument("--clusters", type=int, default=1024)
    s.add_argument("--iters", type=int, default=12)
    sub.add_parser("verify", help="check against llama.cpp's reference output")
    sub.add_parser("bench", help="measure speed")
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--api-key", dest="api_key")
    s.add_argument("--head", choices=("exact", "fast"))
    s = sub.add_parser("chat", help="chat in the terminal")
    s.add_argument("--system", default="")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    s = sub.add_parser("head", help="exact or fast output head for a running server")
    s.add_argument("mode", choices=("exact", "fast"))
    sub.add_parser("info", help="show configuration, pack and GPU")
    sub.add_parser("models", help="list the models and what fits this machine")
    s = sub.add_parser("use", help="download a model, build its pack and select it")
    s.add_argument("model")
    s.add_argument("--gguf", help="use this GGUF file instead of downloading it")
    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    ov = {k: getattr(a, k, None) for k in ("pack", "state_dir", "ctx", "host", "port", "api_key", "head", "lightning_kv")}
    cfg = C.load(a.config, ov)
    return {"convert": cmd_convert, "fast-head": cmd_fast_head, "verify": cmd_verify, "bench": cmd_bench,
            "serve": cmd_serve, "chat": cmd_chat, "head": cmd_head, "info": cmd_info,
            "models": cmd_models, "use": cmd_use}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
