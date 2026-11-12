"""DeepSeek-V4.1-Flash hot experts with the shared MoE placement policy (kindi/moe.py): measured routing counts ->
moe.plan() -> the engine's KINDI_HOT_LIST ("layer expert count" lines; the engine fills the main card, then each
expert card, in list order, as far as VRAM allows).
usage: python tools/dsv41_hotplan.py OUT.txt --card-gb G [--usage-out U.json] (--trace T.trace ... | --list OLD.txt)
  --card-gb: the VRAM the hot experts may use on all cards together (e.g. 0 on a full 3060 + 7 on an M4000)
  --trace: ref_trace files (as tools/hot_list.py reads them); --list: an older hot list with counts (3rd column)
  --prompt: count prompt tokens of the traces too (decode routing follows the same experts)
  --usage-out: also save the counts as a moe.Usage JSON (the installer and Kindi's other MoE engines read this format)"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kindi.moe import MoESpec, Usage, plan  # noqa: E402

NL, NS, NE = 40, 6, 384
D, FF = 5120, 2304                 # as dsv41_cpu.c
# one expert in the GGUF's MXFP4 (17 bytes per 32 weights): gate + up [FF][D] and down [D][FF]
EXPERT_BYTES = 2 * FF * (D // 32) * 17 + D * (FF // 32) * 17


def counts_from_traces(paths, prompt):
    rec = np.dtype([("pid", "<i4"), ("pos", "<i4"), ("gen", "<i4"), ("ids", "<i2", (NL, NS))])
    c = np.zeros((NL, NE), np.int64)
    for p in paths:
        d = np.fromfile(p, dtype=rec)
        keep = np.ones(len(d), bool) if prompt else d["gen"] == 1
        g = d["ids"][keep].astype(np.int64)
        for l in range(NL):
            c[l] += np.bincount(g[:, l].ravel(), minlength=NE)
    return c


def counts_from_list(path):
    c = np.zeros((NL, NE), np.int64)
    for line in open(path):
        f = line.split()
        if len(f) >= 3:
            c[int(f[0]), int(f[1])] = int(f[2])
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--card-gb", type=float, required=True)
    ap.add_argument("--trace", nargs="*", default=[])
    ap.add_argument("--list", default=None)
    ap.add_argument("--prompt", action="store_true")
    ap.add_argument("--usage-out", default=None)
    a = ap.parse_args(argv)
    if not a.trace and not a.list:
        ap.error("give --trace files or --list")
    c = counts_from_traces(a.trace, a.prompt) if a.trace else counts_from_list(a.list)
    spec = MoESpec(range(NL), NE, NS, EXPERT_BYTES)
    usage = Usage(spec, {l: c[l].tolist() for l in range(NL)}, meta={"model": "deepseek-v4.1-flash"})
    # min_card=0: an expert that is not on a card runs on the CPU from host memory, so no layer needs a minimum
    pl = plan(spec, usage, int(a.card_gb * 2**30), min_card=0)
    tot = {l: float(c[l].sum()) or 1.0 for l in range(NL)}
    card = sorted(((c[l, e] / tot[l], l, e) for l in range(NL) for e in pl.on(l, "card")), reverse=True)
    rest = sorted(((c[l, e] / tot[l], l, e) for l in range(NL) for e in pl.on(l, "host")), reverse=True)
    with open(a.out, "w") as f:
        for _, l, e in card + rest:
            f.write("%d %d %d\n" % (l, e, c[l, e]))
    if a.usage_out:
        usage.save(a.usage_out)
    print("%d experts fit %.1f GB (%.1f MB each): %s" % (len(card), a.card_gb, EXPERT_BYTES / 2**20, pl.describe()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
