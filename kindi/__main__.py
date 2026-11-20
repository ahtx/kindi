"""kindi - command line.

  kindi convert MODEL.gguf      build the weight pack from the PTQ1_0 GGUF (once; ~1 minute)
  kindi fast-head               build the cluster file for the fast head (optional; ~2 minutes)
  kindi verify                  check greedy output against llama.cpp's reference (token for token)
  kindi bench                   measure speed and GPU memory
  kindi io                      measure this host's memory-tier rates (RAM copy, disk read, PCIe)
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
    else:
        log("next: kindi verify")
    return 0


def cmd_fast_head(cfg, a):
    _setup_gpu_env(cfg)
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
    from .verify import run
    return run(cfg, log=log)


def cmd_bench(cfg, a):
    _setup_gpu_env(cfg)
    from .bench import run
    return run(cfg, log=log)


def cmd_io(cfg, a):
    from . import io_probe
    mib = a.mib if a.mib is not None else io_probe.DEFAULT_MIB
    return io_probe.run(mib, a.disk_dir, a.skip, a.json)


def cmd_serve(cfg, a):
    _setup_gpu_env(cfg)
    if getattr(a, "slots", None):
        cfg["slots"] = a.slots
    if int(cfg.get("slots") or 1) > 1:                  # decode slots: the engine reads KINDI_SLOTS (kindi/slots.py)
        os.environ["KINDI_SLOTS"] = str(int(cfg["slots"]))
    from .server import serve
    serve(cfg, log=log)
    return 0


def cmd_chat(cfg, a):
    _setup_gpu_env(cfg)
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
    os.makedirs(cfg["state_dir"], exist_ok=True)
    if a.mode == "fast" and not os.path.exists(os.path.join(cfg["pack"], "head_clusters.npz")):
        raise SystemExit("kindi: the fast head needs the cluster file - run: kindi fast-head")
    with open(os.path.join(cfg["state_dir"], "head_mode"), "w") as f:
        f.write(a.mode + "\n")
    log("head mode: %s (the next request uses it)" % a.mode)
    return 0


def cmd_models(cfg, a):
    from .catalog import list_models
    list_models(cfg, log=log, as_json=getattr(a, "json", False))
    return 0


def cmd_use(cfg, a):
    from .catalog import use
    return use(cfg, a.model, gguf=a.gguf, log=log)


def cmd_cards(cfg, a):
    from . import cards
    return cards.main(["--json"] if a.json else [])


def cmd_plan(cfg, a):
    from . import place
    return place.main(a.rest)


def cmd_route(cfg, a):
    from . import route
    return route.main(a.rest)


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
        if m.get("backend") == "kindi_qwen4exp":
            print("backend: Kindi Flash-Next (qwen4exp: card engine, CPU experts, MTP head)")
        elif m.get("backend") == "kindi_lightning":
            print("backend: Kindi Lightning CUDA (Nemotron-H MoE), exact head")
        elif m.get("backend") == "kindi_inkling2":
            print("backend: Kindi Inkling 2 (looped MoE: CPU, hybrid or full-VRAM CUDA)")
        elif m.get("backend") == "kindi_gemma4":
            print("backend: Kindi Gemma 4 CUDA")
            print("fast head: %s" % ("available" if os.path.exists(os.path.join(cfg["pack"],"head_clusters.npz"))
                                     else "not built (kindi fast-head)"))
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
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] in ("plan", "route"):       # their own options (kindi plan --help), no config needed
        if args[0] == "plan":
            from . import place
            return place.main(args[1:])
        from . import route
        return route.main(args[1:])
    _malloc_tuning()
    p = argparse.ArgumentParser(prog="kindi", description="Kindi %s - native CUDA inference for ternary models and Gemma 4" % __version__)
    p.add_argument("--config")
    p.add_argument("--pack")
    p.add_argument("--state-dir", dest="state_dir")
    p.add_argument("--ctx", type=int)
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
    s = sub.add_parser("io", help="measure this host's memory-tier rates: RAM copy, disk read, PCIe copies")
    s.add_argument("--mib", type=int, default=None, help="buffer/file size per probe in MiB (default 128)")
    s.add_argument("--disk-dir", default=None,
                   help="where to create the temporary disk-probe file (default: the system temp dir)")
    s.add_argument("--skip", default="", help="comma-separated probes to skip: ram,disk,pcie")
    s.add_argument("--json", nargs="?", const=True, default=False, metavar="FILE",
                   help="print the report as JSON, or write it to FILE for plan/autotune reuse")
    s = sub.add_parser("inkling", help="run experimental Inkling 2 Mini text inference")
    s.add_argument("gguf", help="local Inkling 2 Mini GGUF")
    s.add_argument("prompt")
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--raw", action="store_true", help="use a raw completion prompt")
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--api-key", dest="api_key")
    s.add_argument("--head", choices=("exact", "fast"))
    s.add_argument("--slots", type=int, help="requests written at the same time (decode slots; one card, KINDI_TC=1)")
    s = sub.add_parser("chat", help="chat in the terminal")
    s.add_argument("--system", default="")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    s = sub.add_parser("head", help="exact or fast output head for a running server")
    s.add_argument("mode", choices=("exact", "fast"))
    sub.add_parser("info", help="show configuration, pack and GPU")
    s = sub.add_parser("models", help="list the models and what fits this machine")
    s.add_argument("--json", action="store_true", help="one JSON list (id, what, size_gb, fits, downloaded)")
    s = sub.add_parser("cards", help="list every GPU: VRAM, free memory, PCIe link, bandwidth, supported or not")
    s.add_argument("--json", action="store_true")
    s = sub.add_parser("plan", help="place models on the cards and write the plan file (kindi plan --help)", add_help=False)
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("route", help="one OpenAI endpoint in front of the plan's servers (kindi route --help)", add_help=False)
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s = sub.add_parser("use", help="download a model, build its pack and select it")
    s.add_argument("model")
    s.add_argument("--gguf", help="use this GGUF file instead of downloading it")
    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return 1
    if a.cmd == "inkling":
        from .inkling2_cli import run
        if a.max_tokens < 1 or (a.ctx is not None and a.ctx < 512):
            p.error("inkling requires --max-tokens >= 1 and --ctx >= 512")
        return run(a)
    ov = {k: getattr(a, k, None) for k in ("pack", "state_dir", "ctx", "host", "port", "api_key", "head")}
    cfg = C.load(a.config, ov)
    return {"convert": cmd_convert, "fast-head": cmd_fast_head, "verify": cmd_verify, "bench": cmd_bench,
            "io": cmd_io, "serve": cmd_serve, "chat": cmd_chat, "head": cmd_head, "info": cmd_info,
            "models": cmd_models, "use": cmd_use, "cards": cmd_cards, "plan": cmd_plan, "route": cmd_route}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
