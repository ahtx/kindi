"""Entry point of the published Kindi images (python3 -m kindi.container [COMMAND ...]).

Without KINDI_MODEL it is the same as `python3 -m kindi`: the image serves the pack at KINDI_PACK.

With KINDI_MODEL, the first start prepares the model and later starts reuse it:

  KINDI_MODEL=bonsai-9b                 a model id from `kindi models`: the GGUF is downloaded into /data/models
                                        (with resume and a sha256 check), unless it is already there
  KINDI_MODEL=/data/models/X.gguf       a GGUF file in the mounted volume (a model id's file is sha256-checked once)

The pack is built once into /data/packs/<name> and selected for the command. A pack that is complete and matches the
GGUF (its sha256, or the size and time of an unknown file) is reused. Two containers that start at the same time on
the same volume build it once: the second waits for the first. An interrupted build leaves no half pack behind.

KINDI_VERIFY_ON_START=1 runs `kindi verify` once for each pack and kind of GPU before the server starts, and stops
the container if it does not pass. KINDI_DOWNLOAD=0 refuses downloads (air-gapped hosts). KINDI_DATA (default
/data) is the mounted volume.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import fcntl
import json
import os
import shutil
import sys
import time

PREPARE_FOR = ("serve", "verify", "bench", "chat", "info", "fast-head")
SIDE = ".kindi-container.json"


def log(msg):
    sys.stdout.write("[kindi container %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
    sys.stdout.flush()


class Stop(Exception):
    """a clear error for the person who started the container (no traceback)"""


def _catalog():
    from . import catalog
    return catalog


def _free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _manifest(pack):
    try:
        with open(os.path.join(pack, "kindi_pack.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _side(pack):
    try:
        with open(os.path.join(pack, SIDE)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def resolve(spec, data):
    """KINDI_MODEL -> (name, gguf path, expected sha256 or None, catalog entry or None)"""
    cat = _catalog()
    if spec in cat.MODELS:
        m = cat.MODELS[spec]
        return spec, os.path.join(data, "models", m["file"]), m["sha"], m
    if spec.endswith(".gguf") or os.sep in spec:
        path = spec if os.path.isabs(spec) else os.path.join(data, "models", spec)
        if not os.path.isfile(path):
            have = sorted(f for f in os.listdir(os.path.join(data, "models")) if f.endswith(".gguf")) \
                if os.path.isdir(os.path.join(data, "models")) else []
            raise Stop("KINDI_MODEL=%s: no such file in the container. Mount the folder that holds it (for example "
                       "-v /path/to/models:%s/models) and give its path in the container. GGUF files in %s/models: %s"
                       % (spec, data, data, ", ".join(have) or "none"))
        base = os.path.basename(path)
        for mid, m in cat.MODELS.items():         # a known file: check its sha256 once, and keep the model id
            if m["file"] == base:
                return mid, path, m["sha"], m
        return os.path.splitext(base)[0], path, None, None
    raise Stop("KINDI_MODEL=%r is neither a model id nor a .gguf path. Model ids: %s"
               % (spec, ", ".join(sorted(cat.MODELS))))


def _checked(gguf, sha):
    """sha256-check a known GGUF once (a marker beside it records the pass, as the installer does)"""
    mark = gguf + ".sha256-ok"
    if os.path.isfile(mark):
        return
    log("checking the sha256 of %s (once)" % os.path.basename(gguf))
    from .convert import sha256
    got = sha256(gguf)
    if got != sha:
        raise Stop("%s has sha256 %s, not the expected %s. The file is damaged or a different model: delete it "
                   "(it is downloaded again) or give its path with a different name" % (gguf, got, sha))
    try:
        open(mark, "w").close()
    except OSError:
        pass                                      # a read-only models folder: the check runs on each start


def _download(mid, m, gguf, log):
    if os.environ.get("KINDI_DOWNLOAD", "1") == "0":
        raise Stop("%s is not in %s and KINDI_DOWNLOAD=0. Put %s there, or allow the download"
                   % (mid, os.path.dirname(gguf), m["file"]))
    if not m.get("repo"):
        raise Stop("%s has no download address: put %s into %s" % (mid, m["file"], os.path.dirname(gguf)))
    d = os.path.dirname(gguf)
    os.makedirs(d, exist_ok=True)
    have = os.path.getsize(gguf + ".part") if os.path.exists(gguf + ".part") else 0
    if _free_bytes(d) < m["size"] - have + (1 << 30):
        raise Stop("%s needs %.1f GB of free disk in %s for the download (free: %.1f GB)"
                   % (m["file"], m["size"] / 1e9, d, _free_bytes(d) / 1e9))
    cat = _catalog()
    log("downloading %s (%.1f GB) from Hugging Face; a stopped download resumes on the next start"
        % (m["file"], m["size"] / 1e9))
    try:
        cat._download("%s/%s/resolve/main/%s" % (cat.HF, m["repo"], m["file"]), gguf, m["size"], m["sha"], log)
    except SystemExit as e:
        raise Stop(str(e))
    except OSError as e:
        raise Stop("the download of %s failed: %s (it resumes on the next start)" % (m["file"], e))
    try:
        open(gguf + ".sha256-ok", "w").close()
    except OSError:
        pass


def _current(pack, gguf, sha):
    """the pack is complete and was built from this GGUF"""
    man = _manifest(pack)
    if not man:
        return False
    from .convert import PACK_FORMAT
    if man.get("pack_format", PACK_FORMAT) != PACK_FORMAT:
        log("the pack in %s has format %s; this release uses %s: building it again" %
            (pack, man.get("pack_format"), PACK_FORMAT))
        return False
    if sha:
        return man.get("source_sha256") == sha
    st = os.stat(gguf)
    side = _side(pack) or {}
    return side.get("size") == st.st_size and side.get("mtime_ns") == st.st_mtime_ns and \
        man.get("source_file") == os.path.basename(gguf)


def prepare(spec, data="/data"):
    """make sure the pack for KINDI_MODEL exists and matches; return its path"""
    if not os.path.isdir(data):
        raise Stop("%s is not mounted. Mount a folder or a named volume there (-v kindi-data:%s)" % (data, data))
    name, gguf, sha, m = resolve(spec, data)
    packs = os.path.join(data, "packs")
    pack = os.path.join(packs, name)
    try:
        os.makedirs(packs, exist_ok=True)
    except OSError as e:
        raise Stop("cannot write %s (%s). The container user (%d:%d) must own the volume: use a named volume, "
                   "or set user: in compose to the owner of the folder" % (packs, e, os.getuid(), os.getgid()))
    if os.path.isfile(gguf) and _current(pack, gguf, sha):
        log("model %s: pack %s is ready" % (name, pack))
        return pack
    lock = open(os.path.join(packs, ".%s.lock" % name), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("model %s: another container is preparing it; waiting" % name)
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if m and not os.path.isfile(gguf):
            _download(name, m, gguf, log)
        elif sha:
            _checked(gguf, sha)
        if _current(pack, gguf, sha):           # the other container built it while this one waited
            log("model %s: pack %s is ready" % (name, pack))
            return pack
        size = os.path.getsize(gguf)
        if _free_bytes(packs) < size * 1.3 + (1 << 30):
            raise Stop("building the pack of %s needs about %.1f GB of free disk in %s (free: %.1f GB)"
                       % (name, size * 1.3 / 1e9, packs, _free_bytes(packs) / 1e9))
        tmp = "%s.building-%d" % (pack, os.getpid())
        shutil.rmtree(tmp, ignore_errors=True)
        log("model %s: building the pack from %s (once; about a minute per GB)" % (name, os.path.basename(gguf)))
        from .convert import convert
        t0 = time.time()
        try:
            convert(gguf, tmp, log=log)
        except Exception as e:                    # noqa: BLE001 - a clear message, and no half pack
            shutil.rmtree(tmp, ignore_errors=True)
            raise Stop("kindi convert failed for %s: %s: %s" % (gguf, type(e).__name__, e))
        st = os.stat(gguf)
        with open(os.path.join(tmp, SIDE), "w") as f:
            json.dump({"model": spec, "gguf": os.path.basename(gguf), "size": st.st_size,
                       "mtime_ns": st.st_mtime_ns, "sha256": sha, "built": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        if os.path.lexists(pack):
            old = "%s.old-%d" % (pack, os.getpid())
            os.rename(pack, old)
            os.rename(tmp, pack)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(tmp, pack)
        log("model %s: pack ready in %.0f s (%s)" % (name, time.time() - t0, pack))
        return pack
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _gpu_kind():
    try:
        import cupy as cp
        p = cp.cuda.runtime.getDeviceProperties(0)
        n = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
        return "".join(c if c.isalnum() else "-" for c in n).strip("-") or "gpu"
    except Exception:                             # noqa: BLE001
        return "cpu"


def verify_once(pack, kmain):
    mark = os.path.join(pack, ".verified-%s" % _gpu_kind())
    if os.path.isfile(mark):
        return
    log("KINDI_VERIFY_ON_START=1: kindi verify for this pack on this kind of GPU (once)")
    if kmain(["verify"]) != 0:
        raise Stop("kindi verify did not pass for %s; the server does not start (see the log above)" % pack)
    open(mark, "w").close()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv) or ["serve"]
    # compose files pass optional settings as empty strings; ROCm and the config must not see an empty value
    for k in ("HSA_OVERRIDE_GFX_VERSION", "KINDI_API_KEY", "KINDI_MODEL_NAME"):
        if os.environ.get(k, None) == "":
            del os.environ[k]
    from .__main__ import main as kmain
    spec = os.environ.get("KINDI_MODEL", "").strip()
    try:
        if spec and args[0] in PREPARE_FOR and "--pack" not in args:
            pack = prepare(spec, os.environ.get("KINDI_DATA", "/data"))
            os.environ["KINDI_PACK"] = pack
            if args[0] == "serve" and os.environ.get("KINDI_VERIFY_ON_START") == "1":
                verify_once(pack, kmain)
    except Stop as e:
        log("ERROR: %s" % e)
        return 2
    return kmain(args)


if __name__ == "__main__":
    sys.exit(main())
