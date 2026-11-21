# SPDX-License-Identifier: AGPL-3.0-only
"""Expose Ollama's content-addressed GGUF blobs under readable .gguf names.

Ollama keeps the weights as `blobs/sha256-<hash>` and describes each model in
`manifests/<registry>/<namespace>/<model>/<tag>`. Kindi (and llama-server) only
need the GGUF path, so this script reads those manifests and makes one link per
model, named after the manifest:

    python tools/ollama_links.py                     # $OLLAMA_MODELS, else ~/.ollama/models
    python tools/ollama_links.py -m <ollama-root>/models -o ~/.kindi/models/ollamaLinks
    python tools/ollama_links.py --dry-run

Nothing is copied and nothing is downloaded. A symbolic link is used when the
host allows it; otherwise a hard link (same volume) is, because Windows needs
Developer Mode or an elevated prompt to create symlinks. If the manifest tree is
missing, the blobs themselves are scanned for the `GGUF` magic, so a bare
`blobs` folder or a folder of `.gguf` files works too.
"""
import argparse
import errno
import json
import os
import re
import sys
from pathlib import Path

MODEL_MEDIA = "application/vnd.ollama.image.model"


def default_models_dir():
    """Ollama's own environment variable first, then its per-user default."""
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env).expanduser() if env else Path.home() / ".ollama" / "models"


def default_output_dir():
    return Path.home() / ".kindi" / "models" / "ollamaLinks"


def _clean_name(rel_parts):
    """`registry.ollama.ai/library/gemma3/4b` -> `gemma3-4b`; other registries keep their host."""
    parts = list(rel_parts)
    if parts and parts[0] == "registry.ollama.ai":
        parts = parts[1:]
    if parts and parts[0] == "library":
        parts = parts[1:]
    name = "-".join(parts)
    return re.sub(r'[\\/:*?"<>|]', "-", name).strip("-") or "model"


def _from_manifests(manifests, blobs):
    found = []
    for path in sorted(manifests.rglob("*")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        layers = [l for l in doc.get("layers") or [] if l.get("mediaType") == MODEL_MEDIA]
        if not layers:
            continue
        # a multimodal model can name more than one model layer; the weights are the largest
        layer = max(layers, key=lambda l: int(l.get("size") or 0))
        digest = str(layer.get("digest") or "")
        if not digest.startswith("sha256:"):
            continue
        found.append({"name": _clean_name(path.relative_to(manifests).parts),
                      "digest": digest, "blob": blobs / digest.replace(":", "-"),
                      "size": int(layer.get("size") or 0), "manifest": path})
    return found


def _from_blobs(blobs):
    found = []
    if not blobs.is_dir():
        return found
    for path in sorted(blobs.iterdir()):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as f:
                if f.read(4) != b"GGUF":
                    continue
        except OSError:
            continue
        name = path.name if path.name.endswith(".gguf") else path.name + ".gguf"
        found.append({"name": name[:-5], "digest": "", "blob": path,
                      "size": path.stat().st_size, "manifest": None})
    return found


def _unique(entries):
    """Two registries can sanitize to the same file name; keep both, numbered."""
    seen = {}
    for e in entries:
        base = e["name"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        if n:
            e["name"] = "%s-%d" % (base, n + 1)
    return entries


def discover(models_dir):
    """One entry per stored model: dict of name, digest, blob, size, manifest.

    Manifests are the source of readable names. If the manifest tree is absent,
    or names every blob that is not on disk, the blobs are scanned for the GGUF
    magic instead (a bare `blobs` directory or a directory of `.gguf` files).
    """
    root = Path(models_dir)
    manifests, blobs = root / "manifests", root / "blobs"
    found = _from_manifests(manifests, blobs if blobs.is_dir() else root) if manifests.is_dir() else []
    if not any(e["blob"].exists() for e in found):
        found = _from_blobs(blobs if blobs.is_dir() else root)
    return _unique(found)


def points_to(link, blob):
    """True when `link` already resolves to `blob` (a symlink or a hard link)."""
    try:
        return os.path.samefile(str(link), str(blob))
    except OSError:
        return False


def make_link(blob, target, kind="auto"):
    """Create the link and return the kind that was made.

    `auto` tries a symbolic link first and falls back to a hard link when the
    host refuses (Windows without Developer Mode: WinError 1314).
    """
    blob, target = Path(blob), Path(target)
    if kind in ("auto", "symlink"):
        try:
            os.symlink(str(blob.resolve()), str(target))
            return "symlink"
        except OSError:
            if kind == "symlink":
                raise
    os.link(str(blob), str(target))
    return "hardlink"


def _link_hint(exc):
    """Why a link can fail on Windows: hard links stay on one volume, symlinks need a privilege."""
    if getattr(exc, "winerror", None) == 1314 or exc.errno == errno.EXDEV:
        return ("; a hard link cannot cross volumes and a symlink needs Developer Mode, so put -o "
                "on the models' drive or enable Developer Mode")
    return ""

def _gib(n):
    return "%.2f GiB" % (n / 2 ** 30) if n >= 2 ** 30 else "%.1f MiB" % (n / 2 ** 20)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ollama_links.py",
        description="Link the GGUF models Ollama already downloaded under readable .gguf names.")
    ap.add_argument("-m", "--models-dir", default=None,
                    help="Ollama's models directory (default: $OLLAMA_MODELS, else ~/.ollama/models)")
    ap.add_argument("-o", "--out", default=None,
                    help="where the links go (default: ~/.kindi/models/ollamaLinks)")
    ap.add_argument("--link", choices=("auto", "symlink", "hardlink"), default="auto",
                    help="link kind; auto uses a symlink when the host allows it, else a hard link")
    ap.add_argument("--dry-run", action="store_true", help="print what would be linked, change nothing")
    ap.add_argument("--force", action="store_true", help="replace a file that is not already the same blob")
    args = ap.parse_args(argv)

    models = Path(args.models_dir).expanduser() if args.models_dir else default_models_dir()
    out = Path(args.out).expanduser() if args.out else default_output_dir()
    if not models.is_dir():
        print("ollama_links: no such directory: %s (pass -m, or set OLLAMA_MODELS)" % models,
              file=sys.stderr)
        return 2
    entries = discover(models)
    if not entries:
        print("ollama_links: no GGUF model found under %s" % models, file=sys.stderr)
        return 1

    print("models: %s\nlinks:  %s" % (models, out))
    made = skipped = failed = 0
    for e in entries:
        target = out / (e["name"] + ".gguf")
        if not e["blob"].exists():
            print("  [--] %-40s blob missing: %s" % (target.name, e["blob"]), file=sys.stderr)
            failed += 1
            continue
        if target.exists() or target.is_symlink():
            if points_to(target, e["blob"]):
                print("  [ok] %-40s already linked" % target.name)
                skipped += 1
                continue
            if not args.force:
                print("  [!!] %-40s exists and is not this blob; pass --force" % target.name,
                      file=sys.stderr)
                failed += 1
                continue
            if not args.dry_run:
                target.unlink()
        if args.dry_run:
            print("  [->] %-40s %s -> %s" % (target.name, _gib(e["size"]), e["blob"].name))
            made += 1
            continue
        out.mkdir(parents=True, exist_ok=True)
        try:
            kind = make_link(e["blob"], target, args.link)
        except OSError as exc:
            print("  [!!] %-40s %s%s" % (target.name, exc, _link_hint(exc)), file=sys.stderr)
            failed += 1
            continue
        print("  [ok] %-40s %s -> %s (%s)" % (target.name, _gib(e["size"]), e["blob"].name, kind))
        made += 1
    verb = "would link" if args.dry_run else "linked"
    print("%s %d, %d already linked, %d failed" % (verb, made, skipped, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
