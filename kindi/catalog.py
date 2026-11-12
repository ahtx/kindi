"""The models Kindi runs: `kindi models` lists them and what fits this machine, `kindi use ID` downloads one (with
resume and a sha256 check), builds its pack and selects it.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import configparser
import hashlib
import os
import sys
import urllib.request

HF = "https://huggingface.co"

# id: file, download URL (None: give the file with --gguf), sha256, size in bytes, card memory in MiB for one card
# (and for each of two cards in tensor parallel, 0 when the model does not split), and a short description
MODELS = {
    "bonsai-2b": dict(file="Ternary-Bonsai-2-2B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-2B-gguf",
                      sha="36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa", size=429183456,
                      one=900, two=0, what="Ternary-Bonsai-2 2B (ternary weights)"),
    "bonsai-4b": dict(file="Ternary-Bonsai-2-4B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-4B-gguf",
                      sha="56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60", size=949549920,
                      one=1900, two=0, what="Ternary-Bonsai-2 4B (ternary weights)"),
    "bonsai-9b": dict(file="Ternary-Bonsai-2-9B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-9B-gguf",
                      sha="3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b", size=1997486976,
                      one=3800, two=0, what="Ternary-Bonsai-2 9B (ternary weights)"),
    "bonsai-27b": dict(file="Ternary-Bonsai-2-27B-PTQ1_0.gguf", repo="prism-ml/Ternary-Bonsai-2-27B-gguf",
                       sha="53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3", size=5946648928,
                       one=7600, two=3800, what="Ternary-Bonsai-2 27B (ternary weights)"),
    "nemotron-e4b": dict(file="NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q3_K_XL.gguf", repo="unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF",
                         sha="3f8032b9d2c18536c4ac51b97959a563cc27434de99823640c98147edc62ce9b", size=4219133120,
                         one=3300, two=2400, what="NVIDIA Nemotron 3.5 Bolt E4B (Gemma 4), UD-Q3_K_XL"),
    "nanbeige-33b": dict(file="AH-Nanbeige4.3-33B-IQ3_XXS.gguf", repo="ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF",
                         sha="9d4f18e8a22573c94b18a455147e807a8b9d2cd4b30302f2a4639c2db0a6c239", size=13108685952,
                         one=16200, two=0, split=(15500, 700), what="AH-Nanbeige4.3-33B, IQ3_XXS",
                         card="https://huggingface.co/ahtx/AH-Nanbeige4.3-33B-IQ3_XXS-GGUF"),
    # Nota AI's ternary Nemotron 3.5 Lightning 30B-A3B, GGUF by boro64; licence: OpenMDW 1.1 (openmdw.ai/license/1-1)
    # One card of 10 GB holds all of it; on 6 to 8 GB cards the least used experts stay in host memory.
    "lightning-30b": dict(file="Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf",
                          repo="boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF",
                          sha="38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3", size=9472656320,
                          one=5800, two=0, what="Nemotron 3.5 Lightning 30B-A3B, ternary (Nota AI), Q2_0",
                          card="https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF"),
}


def _cards():
    """(name, MiB) of each visible GPU; [] without a GPU"""
    try:
        import cupy as cp
        out = []
        for i in range(cp.cuda.runtime.getDeviceCount()):
            p = cp.cuda.runtime.getDeviceProperties(i)
            n = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
            out.append((n, int(p["totalGlobalMem"]) >> 20))
        return out
    except Exception:
        return []


def fits(m, cards):
    """'one card', 'two cards' (or 'N cards' for a model that splits its layers over any number) or '' for this
    machine"""
    if any(mb >= m["one"] for _, mb in cards):
        return "one card"
    if m.get("split"):                    # (total MiB, MiB on each card): the largest cards first
        need, each, have = m["split"][0], m["split"][1], 0
        for n, mb in enumerate(sorted((mb for _, mb in cards), reverse=True), 1):
            have += mb
            if have >= need + each * n:
                return "%d cards" % n
        return ""
    if m["two"] and len(cards) >= 2 and sorted(mb for _, mb in cards)[-2] >= m["two"]:
        return "two cards"
    return ""


def model_dir(cfg):
    return os.path.join(os.path.dirname(os.path.expanduser(cfg["pack"]).rstrip("/")) or ".", "models")


def list_models(cfg, log=print):
    cards = _cards()
    log("GPUs: %s" % (", ".join("%s (%d MiB)" % c for c in cards) or "none (the CPU engine runs the Bonsai models)"))
    d = model_dir(cfg)
    for mid, m in MODELS.items():
        have = os.path.isfile(os.path.join(d, m["file"]))
        log("  %-13s %-52s %5.1f GB  %-10s %s" % (mid, m["what"], m["size"] / 1e9, fits(m, cards) or "too large",
                                                  "downloaded" if have else ""))
    log("kindi use ID  downloads a model, builds its pack and selects it")


def _download(url, path, size, sha, log):
    """resume a partial download, then check the sha256"""
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have < size:
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-" % have} if have else {})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
            if have and r.status != 206:
                raise SystemExit("kindi: the server does not resume downloads; delete %s and retry" % part)
            got, step = have, 1 << 30
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b); got += len(b)
                if got // step != (got - len(b)) // step:
                    log("  %.1f of %.1f GB" % (got / 1e9, size / 1e9))
    log("checking sha256")
    h = hashlib.sha256()
    with open(part, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != sha:
        raise SystemExit("kindi: %s has the wrong sha256 (delete it and retry)" % part)
    os.replace(part, path)


def _select(cfg, pack, log):
    """write pack = PACK into the config file in use (or ~/.kindi/kindi.conf)"""
    src = cfg.get("source", "")
    path = src if src and os.path.isfile(src) else os.path.expanduser("~/.kindi/kindi.conf")
    c = configparser.ConfigParser()
    if os.path.isfile(path):
        c.read(path)
    if not c.has_section("kindi"):
        c.add_section("kindi")
    c.set("kindi", "pack", pack)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        c.write(f)
    log("selected: pack = %s in %s" % (pack, path))
    env = os.environ.get("KINDI_PACK")
    if env and os.path.abspath(os.path.expanduser(env)) != os.path.abspath(pack):
        link = os.path.expanduser(env)
        if os.path.islink(link) or not os.path.exists(link):    # the installer's layout: KINDI_PACK is a link
            tmp = link + ".new"
            if os.path.lexists(tmp):
                os.remove(tmp)
            os.symlink(pack, tmp); os.replace(tmp, link)
            log("KINDI_PACK (%s) now points to %s" % (link, pack))
        else:
            log("note: KINDI_PACK=%s is set in the environment and wins over the config file" % env)


def use(cfg, mid, gguf=None, log=print):
    if mid not in MODELS:
        raise SystemExit("kindi: unknown model %r (kindi models lists them)" % mid)
    m = MODELS[mid]
    if m.get("card"):
        log("model card: %s" % m["card"])
    cards = _cards()
    if cards and not fits(m, cards) and m.get("split"):
        log("note: %s needs %d MiB on one card, or %d MiB in total on several (NVIDIA sm_80 and later); this machine "
            "may not hold it" % (mid, m["one"], m["split"][0] + 2 * m["split"][1]))
    elif cards and not fits(m, cards):
        log("note: %s needs a card with %d MiB%s; this machine may not hold it" %
            (mid, m["one"], " (or two with %d MiB each)" % m["two"] if m["two"] else ""))
    if not gguf:
        d = model_dir(cfg); os.makedirs(d, exist_ok=True)
        gguf = os.path.join(d, m["file"])
        if not os.path.isfile(gguf):
            if not m["repo"]:
                raise SystemExit("kindi: %s has no download address in this version; download %s and run "
                                 "kindi use %s --gguf FILE" % (mid, m["file"], mid))
            log("downloading %s (%.1f GB)" % (m["file"], m["size"] / 1e9))
            _download("%s/%s/resolve/main/%s" % (HF, m["repo"], m["file"]), gguf, m["size"], m["sha"], log)
    pack = os.path.join(os.path.dirname(model_dir(cfg)), "packs", mid)
    from .convert import convert
    if not os.path.isfile(os.path.join(pack, "kindi_pack.json")):
        convert(gguf, pack, log=log)
    _select(cfg, pack, log)
    log("restart kindi serve to load %s" % mid)
    return 0
