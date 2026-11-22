"""Pack integrity (B16): the size, a fast hash and the sha256 of every pack file, recorded in kindi_pack.json by
kindi convert ("files"), and checked before the pack is used.

  - size: every time an engine opens the pack (cheap);
  - the full sha256: when a 5-trit cache (g5_*.bin) is built from the pack, and in kindi verify;
  - each cache records the sha256 of the bcodes.bin it was made from; a cache of other bcodes is made again.

Why: on 11-15 a pack's bcodes.bin on a test host turned out to be changed in place (B15): every cache built from it
gave garbage, while an older cache still worked, and nothing said which file was wrong. A mismatch now stops with
the file's name. A pack made before 0.7.0 has no "files" record: it is used as before (`kindi convert --force` adds
it).

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""
import glob
import hashlib
import json
import os

FAST_BLOCK = 1 << 16          # the fast hash reads 64 KiB ...
FAST_STRIDE = 1 << 26         # ... at every 64 MiB (and the last 64 KiB): about 110 reads for a 7 GB bcodes.bin


class PackError(RuntimeError):
    pass


def pack_files(d):
    """the files of a weight pack that the engines read"""
    names = ["bcodes.bin", "dpm.bin", "f32.bin", "index.json", "bcodes_index.json", "vocab.json", "merges.txt"]
    names += sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "signs_*.f32")))
    return [n for n in names if os.path.exists(os.path.join(d, n))]


def fast_hash(path):
    """blake2b of the size and of 64 KiB at every 64 MiB and at the end: catches a changed region quickly, not every
    changed byte (the sha256 does that)"""
    n = os.path.getsize(path)
    h = hashlib.blake2b(str(n).encode(), digest_size=16)
    with open(path, "rb") as f:
        offs = list(range(0, n, FAST_STRIDE)) + [max(0, n - FAST_BLOCK)]
        for o in offs:
            f.seek(o)
            h.update(f.read(FAST_BLOCK))
    return h.hexdigest()


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def record(d, log=None):
    """{file: {size, fast, sha256}} of the pack in d (kindi convert writes it into kindi_pack.json as "files")"""
    out = {}
    for n in pack_files(d):
        p = os.path.join(d, n)
        if log and os.path.getsize(p) > (1 << 30):
            log("hashing %s ..." % n)
        out[n] = {"size": os.path.getsize(p), "fast": fast_hash(p), "sha256": sha256(p)}
    return out


def check(d, level="size", man=None):
    """check the pack in d against its record; level: size | fast | full. Returns the number of files checked
    (0: the pack has no record, made before 0.7.0). Raises PackError naming the first file that differs."""
    if man is None:
        man = json.load(open(os.path.join(d, "kindi_pack.json")))
    files = man.get("files")
    if not files:
        return 0
    for n, rec in sorted(files.items()):
        p = os.path.join(d, n)
        if not os.path.exists(p):
            raise PackError("pack %s: %s is missing (convert the model again: kindi convert --force)" % (d, n))
        size = os.path.getsize(p)
        if size != rec["size"]:
            raise PackError("pack %s: %s has %d bytes, the pack was made with %d: the file changed after "
                            "kindi convert (convert the model again: kindi convert --force)" % (d, n, size, rec["size"]))
        if level in ("fast", "full") and fast_hash(p) != rec["fast"]:
            raise PackError("pack %s: %s is not the file kindi convert wrote (fast hash): it changed after the "
                            "conversion (convert the model again: kindi convert --force)" % (d, n))
        if level == "full" and sha256(p) != rec["sha256"]:
            raise PackError("pack %s: %s is not the file kindi convert wrote (sha256 %s..., recorded %s...): it "
                            "changed after the conversion (convert the model again: kindi convert --force)"
                            % (d, n, sha256(p)[:16], rec["sha256"][:16]))
    return len(files)


def source_of(man):
    """the recorded sha256 of bcodes.bin (what a cache built from this pack records), or None"""
    return (man.get("files") or {}).get("bcodes.bin", {}).get("sha256")
