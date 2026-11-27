"""Fetch tensors of the official Qwen3.8-Flash-Next release (BF16 safetensors) by HTTP byte range, without downloading
the shards: the MTP block (31 tensors, 5.21 GB) for kindi/qwen4exp_mtp.py, or a few main-model tensors for checks.

usage: python tools/qwen4exp_mtp_fetch.py OUTDIR [--match '^mtp\\.'] [--skip-experts] [--repo URL]

Writes OUTDIR/<name>.bin (the raw little-endian data) and OUTDIR/tensors.json (name -> dtype, shape, sha of nothing:
the sizes are checked against the shard headers). Resumable: a tensor whose .bin has the right size is kept.
The release is under the Qwen Community License 1.0 (LICENSE in the release; see docs/LICENSES or the README)."""
import argparse
import json
import os
import re
import struct
import sys
import time
import urllib.request

REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main"
CHUNK = 64 << 20
_final = {}


def _get(url, a, b, tries=6):
    """bytes [a, b] of url (inclusive), retried"""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            if len(data) != b - a + 1:
                raise IOError("short read %d of %d" % (len(data), b - a + 1))
            return data
        except Exception as e:  # noqa: BLE001 (network errors of every kind are retried)
            if i == tries - 1:
                raise
            time.sleep(2 + 4 * i)
            sys.stderr.write("retry %s: %s\n" % (url.rsplit("/", 1)[-1][:60], e))


def resolve(repo, name):
    """the shard's URL after redirects (the file host's CDN), resolved once"""
    if name not in _final:
        req = urllib.request.Request(repo.rstrip("/") + "/" + name, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            _final[name] = r.geturl()
    return _final[name]


def index(repo):
    with urllib.request.urlopen(repo.rstrip("/") + "/model.safetensors.index.json", timeout=120) as r:
        return json.loads(r.read())["weight_map"]


def header(repo, shard):
    """the safetensors header of a shard: name -> (dtype, shape, absolute begin, end)"""
    url = resolve(repo, shard)
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    h = json.loads(_get(url, 8, 7 + n))
    out = {}
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        out[k] = (v["dtype"], v["shape"], 8 + n + a, 8 + n + b)
    return out


def fetch(repo, shard, begin, end, path):
    """bytes [begin, end) of a shard into path, in chunks; kept if already complete"""
    if os.path.exists(path) and os.path.getsize(path) == end - begin:
        return False
    url = resolve(repo, shard)
    with open(path + ".part", "wb") as f:
        for a in range(begin, end, CHUNK):
            f.write(_get(url, a, min(a + CHUNK, end) - 1))
    os.replace(path + ".part", path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--match", default=r"^mtp\.")
    ap.add_argument("--skip-experts", action="store_true", help="leave out the routed experts (the dense parts only)")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    wm = index(a.repo)
    names = sorted(k for k in wm if re.search(a.match, k) and not (a.skip_experts and ".mlp.experts." in k))
    if not names:
        raise SystemExit("no tensor matches %r" % a.match)
    info, total, got = {}, 0, 0
    heads = {}
    for k in names:
        sh = wm[k]
        if sh not in heads:
            heads[sh] = header(a.repo, sh)
        dt, shape, b0, b1 = heads[sh][k]
        info[k] = {"dtype": dt, "shape": shape, "shard": sh, "bytes": b1 - b0}
        total += b1 - b0
    print("%d tensors, %.2f GB, from %d shards" % (len(names), total / 1e9, len(heads)), flush=True)
    t0 = time.time()
    for k in names:
        sh = info[k]["shard"]
        _, _, b0, b1 = heads[sh][k]
        new = fetch(a.repo, sh, b0, b1, os.path.join(a.outdir, k + ".bin"))
        got += b1 - b0
        if new:
            print("  %-70s %9.1f MB  (%.0f%%)" % (k, (b1 - b0) / 1e6, 100.0 * got / total), flush=True)
    with open(os.path.join(a.outdir, "tensors.json"), "w", encoding="utf-8", newline="") as f:
        json.dump(info, f, indent=1)
    print("done in %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
