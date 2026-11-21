"""kindi - command line.

  kindi convert MODEL.gguf      build the weight pack from the PTQ1_0 GGUF (once; ~1 minute)
  kindi fast-head               build the cluster file for the fast head (optional; ~2 minutes)
  kindi verify                  check greedy output against llama.cpp's reference (token for token)
  kindi bench                   measure speed and GPU memory
  kindi serve                   run the OpenAI-compatible server
  kindi chat                    chat in the terminal (no server needed; saves the session unless --no-record;
                                --resume SESSION_ID continues a saved session in place)
  kindi sessions                list, triage, tag and export chat sessions as raw markdown
  kindi head exact|fast         choose the output head of a running server (next request, no restart)
  kindi info                    show the configuration, the pack and the GPU
Global options: --config FILE, --pack DIR, --state-dir DIR, --ctx N."""
import argparse
import json
import os
import re
import sys
import time

from . import __version__
from . import config as C


def log(msg):
    sys.stdout.write("[kindi %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


def cmd_sessions(cfg, a):
    """kindi sessions list|show|export|triage|tag|status|import (host-only; reads state_dir/sessions)."""
    from .sessions import SessionStore, STATUSES
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))

    def _flags(r):
        out = ""
        if r.get("unread"):
            out += "  [unread]"
        if r.get("protected"):
            out += "  [protected]"
        return out
    if a.action == "list":
        rows = st.list(status=a.status, tag=a.tag)
        if not rows:
            log("no sessions")
            return 0
        for r in rows:
            log("%s  %-24s %-13s %d msgs  %s%s" % (r["id"], r["title"][:24], r["status"],
                                               len(r.get("messages") or []),
                                               ",".join(r.get("tags") or []), _flags(r)))
        return 0
    if a.action == "show":
        if not a.id:
            raise SystemExit("kindi sessions show: give a session id")
        rec = st.get(a.id[0])
        st.mark_read(a.id[0])
        sys.stdout.write(st.export_markdown(rec))
        return 0
    if a.action == "triage":
        rows = st.triage()
        if not rows:
            log("triage queue is empty")
            return 0
        for r in rows:
            log("%s  %-24s %-13s %d msgs  %s%s" % (r["id"], r["title"][:24], r["status"],
                                               len(r.get("messages") or []),
                                               ",".join(r.get("tags") or []), _flags(r)))
        return 0
    if a.action == "protect":
        if not a.id:
            raise SystemExit("kindi sessions protect: give a session id")
        rec = st.set_protected(a.id[0], on=not a.off)
        log("session %s %sprotected" % (a.id[0], "un" if a.off else ""))
        return 0
    if a.action == "export":
        if not a.id:
            raise SystemExit("kindi sessions export: give at least one session id")
        for sid in a.id:
            path = st.export_file(st.get(sid), a.out or os.getcwd())
            log("exported %s -> %s" % (sid, path))
        return 0
    if a.action == "tag":
        if not a.id or not a.tag:
            raise SystemExit("kindi sessions tag: give a session id and --tag")
        st.add_tag(a.id[0], a.tag)
        log("tagged %s with %s" % (a.id[0], a.tag))
        return 0
    if a.action == "status":
        if not a.id or not a.status:
            raise SystemExit("kindi sessions status: give a session id and --status")
        if a.status not in STATUSES:
            raise SystemExit("kindi sessions: status must be one of %s (got %r)" % (STATUSES, a.status))
        st.set_status(a.id[0], a.status)
        log("session %s -> %s" % (a.id[0], a.status))
        return 0
    if a.action == "import":
        if not a.id:
            raise SystemExit("kindi sessions import: give a markdown file (or - for stdin)")
        text = sys.stdin.read() if a.id[0] == "-" else open(a.id[0], encoding="utf-8").read()
        rec = st.import_markdown(text)
        log("imported %s (%d msgs)" % (rec["id"], len(rec["messages"])))
        return 0
    if a.action == "prune":
        rows = st.prune(older_than_days=a.days, statuses=None if a.all else ("resolved",),
                        dry_run=bool(a.dry_run))
        if not rows:
            log("no sessions older than %d days" % a.days)
            return 0
        kept = sum(1 for r in rows if r.get("protected"))
        for r in rows:
            log("%s  %-24s %-13s updated %s%s" % (r["id"], r["title"][:24], r["status"], r["updated"],
                                                  "  (protected: kept)" if r.get("protected") else ""))
        log("%s %d session(s)" % ("would prune" if a.dry_run else "pruned", len(rows) - kept))
        if kept:
            log("%d protected session(s) kept" % kept)
        return 0
    if a.action == "stats":
        stats = st.stats()
        if a.json:
            import json as _json
            sys.stdout.write(_json.dumps(stats, indent=2) + "\n")
            return 0
        log("sessions: %d (open %d, needs-review %d, resolved %d, unread %d, protected %d)" % (
            stats["sessions"], stats["by_status"]["open"],
            stats["by_status"]["needs-review"], stats["by_status"]["resolved"],
            stats["unread"], stats["protected"]))
        log("messages: %d, stored bytes: %d" % (stats["messages"], stats["bytes"]))
        if stats["sessions"]:
            log("oldest updated %s, newest updated %s" % (stats["oldest_updated"], stats["newest_updated"]))
        return 0
    raise SystemExit("kindi sessions: unknown action")


def _record_chat(cfg, msgs):
    """Save a chat transcript as a session (host-only; no GPU touched)."""
    from .sessions import SessionStore
    convo = [m for m in msgs if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    if not convo:
        return None
    title = next((str(m["content"]).strip().splitlines()[0][:60] for m in convo if m["role"] == "user"), "chat")
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))
    rec = st.create(title=title, messages=[(m["role"], m["content"]) for m in convo])
    return rec["id"]


def _resume_chat(cfg, sid):
    """Load a stored session's turns as the chat seed; unknown or malformed ids refuse loudly."""
    from .sessions import SessionStore
    if not sid or not re.fullmatch(r"[0-9a-f]{12}", str(sid)):
        raise SystemExit("kindi chat --resume: the session id must be 12 hex characters (got %r)" % sid)
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))
    try:
        rec = st.get(sid)
    except KeyError:
        raise SystemExit("kindi chat --resume: no session %s (kindi sessions list)" % sid)
    return rec, [{"role": m["role"], "content": m["content"]}
                  for m in rec.get("messages", [])
                  if m.get("role") in ("system", "user", "assistant")]


def _append_chat(cfg, sid, msgs):
    """Append the new turns of a resumed chat back to its stored session."""
    from .sessions import SessionStore
    convo = [m for m in msgs if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()]
    if not convo:
        return None
    st = SessionStore(os.path.join(cfg["state_dir"], "sessions"))
    rec = st.append_messages(sid, [(m["role"], m["content"]) for m in convo])
    return rec["id"]


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


def cmd_serve(cfg, a):
    _setup_gpu_env(cfg)
    from .server import serve
    serve(cfg, log=log)
    return 0


def cmd_chat(cfg, a):
    _setup_gpu_env(cfg)
    from .server import Kindi
    K = Kindi(cfg, log=log)
    msgs, seed_len = [], 0
    if a.resume:
        rec, seeded = _resume_chat(cfg, a.resume)
        msgs = [{"role": "system", "content": a.system}] if a.system else []
        msgs += seeded
        seed_len = len(msgs)
        print("resumed session %s (%d prior turns)" % (rec["id"], len(seeded)))
    elif a.system:
        msgs.append({"role": "system", "content": a.system})
    print("Kindi %s - type a message, /reset to start over, /exit to quit." % __version__)
    while True:
        try:
            line = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip() in ("/exit", "/quit"):
            break
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
    if not a.no_record:
        if a.resume:
            sid = _append_chat(cfg, a.resume, msgs[seed_len:])
            if sid:
                print("\nappended to session %s (kindi sessions list)" % sid)
        else:
            sid = _record_chat(cfg, msgs)
            if sid:
                print("\nsaved session %s (kindi sessions list)" % sid)


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
        if m.get("backend") == "kindi_lightning":
            print("backend: Kindi Lightning CUDA (Nemotron-H MoE), exact head")
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
    s = sub.add_parser("serve", help="run the OpenAI-compatible server")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--api-key", dest="api_key")
    s.add_argument("--head", choices=("exact", "fast"))
    s = sub.add_parser("chat", help="chat in the terminal")
    s.add_argument("--system", default="")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    s.add_argument("--no-record", action="store_true", help="do not save this chat as a session (state_dir/sessions)")
    s.add_argument("--resume", metavar="SESSION_ID", help="continue a saved session (kindi sessions list) instead of starting a new one; new turns append back to it")
    s = sub.add_parser("sessions", help="list, triage, tag and export chat sessions (state_dir/sessions)")
    s.add_argument("action", choices=("list", "show", "export", "triage", "tag", "status", "import", "prune", "stats", "protect"))
    s.add_argument("--off", action="store_true", help="protect: clear the protection flag instead of setting it")
    s.add_argument("id", nargs="*", help="session id (show/export/tag/status) or file (import, - for stdin)")
    s.add_argument("--out", help="export directory (default: current directory)")
    s.add_argument("--tag", help="tag filter (list) or tag to add (tag)")
    s.add_argument("--status", help="status filter (list) or new status (status)")
    s.add_argument("--days", type=int, default=30, help="prune: delete sessions not updated for this many days")
    s.add_argument("--all", action="store_true", help="prune: delete sessions of any status (default: resolved only)")
    s.add_argument("--dry-run", action="store_true", help="prune: list what would be deleted without deleting")
    s.add_argument("--json", action="store_true", help="stats: print the summary as JSON")
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
    ov = {k: getattr(a, k, None) for k in ("pack", "state_dir", "ctx", "host", "port", "api_key", "head")}
    cfg = C.load(a.config, ov)
    return {"convert": cmd_convert, "fast-head": cmd_fast_head, "verify": cmd_verify, "bench": cmd_bench,
            "serve": cmd_serve, "chat": cmd_chat, "head": cmd_head, "info": cmd_info,
            "models": cmd_models, "use": cmd_use, "sessions": cmd_sessions}[a.cmd](cfg, a)


if __name__ == "__main__":
    sys.exit(main())
