"""DeepSeek-V4.1-Flash: the expert-card settings calibrated at start (F9, Amir 11-15: "yes auto calibrate").

At load the engine times its expert card (ms for a call of one pair, ms for each more pair) and the CPU's experts (ms
per distinct expert of a verify-sized call at the configured threads); dsv_xc_measure in dsv41_cpu.c. The result is
cached per card UUIDs + model + CPU lists in the state dir, so a later start skips the timing unless the cards change.
decide() turns it into two settings, with the reasons printed:

- KINDI_XC_MAXP, the expert card's pairs per token: as many as the card computes within 80% of the CPU's time for one
  layer of a 6-token verify forward. A wide band keeps today's 2 (1 to 5 pairs a token give 2, so a timing off by
  30% does not move it); a much faster card gets more (at 6, no cap), a much slower one 1.
- KINDI_XC_ADAPT, the adaptive expert cache with staging copies: on only if (1) at the adaptive hit rate the card
  still finishes its pairs within 80% of the CPU's time for the rest, and (2) a swap gains more than it costs: the
  picks it moves to the card (11.9 per swap, 11-15 window 2) times what a pick saves on the CPU less what it costs on
  the card, against the host memory it takes (an expert read and written once: 2 x 18.8 MB at the 87 GB/s that the
  CPU experts use).

The hot-expert lists are not decided here: the engine loads as many experts of the list as each card's free VRAM
holds at start (cudaMemGetInfo), so the card's actual memory sets their number already.

An explicit KINDI_XC_MAXP or KINDI_XC_ADAPT always wins. With the calibration constants of window 2 the decision on
the DeepSeek host (RTX 3060 + Quadro M4000, 2 x E5-2673 v4) is MAXP 2, ADAPT off: today's settings
(tools/test_dsv41_xcal.py)."""
import ctypes
import json
import os
import sys
import time

NS, T_VERIFY, U_VERIFY = 6, 6, 20      # picks per token; a typical verify forward; its distinct CPU experts per layer
#                                        (20: the live service, 11-15: 177 ms of host experts for 40 layers of a 6-token forward)
CPU_SHARE_STATIC = 0.88                 # the picks the CPU computes with the static card list (12% on the card)
CPU_SHARE_ADAPT = 0.67                  # with the adaptive cache (33% on the card, window 2)
PICKS_PER_SWAP = 11.9                   # card picks gained per swapped expert (window 2: 25,240 picks / 2,112 swaps)
EXPERT_BYTES = 3 * 2304 * (5120 // 32) * 17
HOST_GBS = 87.0                         # the CPU experts' memory rate (window 3), which a staging copy competes with
MARGIN = 0.8                            # the card must be done within this share of the CPU's time


def decide(m, env=None):
    """m: dict(card_ms1, card_ms_pair, cpu_ms_expert) (card_* None when there is no expert card).
    Returns dict(maxp, adapt, reasons): maxp an int (0 = no cap) or None (leave the default), adapt 0/1."""
    env = os.environ if env is None else env
    reasons = []
    cpu_layer = m["cpu_ms_expert"] * U_VERIFY                      # one layer's CPU experts, static list
    maxp, adapt = None, 0
    if m.get("card_ms1") is None:
        reasons.append("no expert card: nothing to set")
    else:
        c1, slope = m["card_ms1"], max(m["card_ms_pair"], 1e-3)
        pairs = (MARGIN * cpu_layer - c1) / slope + 1                 # pairs the card does in its share of the time
        cap = pairs / T_VERIFY
        maxp = 1 if cap < 1.0 else 2 if cap < 5 else (0 if cap >= NS else int(cap))
        reasons.append("pair cap: the card does %.1f pairs (%.2f ms + %.3f ms each) in %.0f%% of the CPU's %.2f ms for "
                       "a layer, %.2f per token of a %d-token forward -> %s" % (
                           pairs, c1, slope, 100 * MARGIN, cpu_layer, cap, T_VERIFY,
                           "no cap" if maxp == 0 else maxp))
        # the adaptive cache
        picks_cpu = NS * T_VERIFY * CPU_SHARE_STATIC
        cpu_ms_pick = m["cpu_ms_expert"] * U_VERIFY / picks_cpu       # what one pick costs the CPU, on average
        gain = PICKS_PER_SWAP * (cpu_ms_pick - slope)
        copy = 2 * EXPERT_BYTES / (HOST_GBS * 1e6)                   # ms of the CPU experts' memory time per swap
        card_pairs = NS * T_VERIFY * (1 - CPU_SHARE_ADAPT)
        card_ms = c1 + slope * (card_pairs - 1)
        cpu_rest = cpu_layer * CPU_SHARE_ADAPT / CPU_SHARE_STATIC
        fits = card_ms <= MARGIN * cpu_rest
        adapt = int(gain > copy and fits)
        reasons.append("adaptive card: a swap gains %.2f ms (%.1f picks x (%.3f CPU - %.3f card ms a pick)) against "
                       "%.2f ms of copy; the card's %.1f pairs a layer take %.2f ms against %.0f%% of the CPU's %.2f ms "
                       "-> %s" % (gain, PICKS_PER_SWAP, cpu_ms_pick, slope, copy, card_pairs, card_ms, 100 * MARGIN,
                                   cpu_rest, "on" if adapt else "off"))
    if env.get("KINDI_XC_MAXP"):
        reasons.append("KINDI_XC_MAXP=%s is set: it wins" % env["KINDI_XC_MAXP"])
        maxp = None
    if env.get("KINDI_XC_ADAPT"):
        reasons.append("KINDI_XC_ADAPT=%s is set: it wins" % env["KINDI_XC_ADAPT"])
        adapt = 0
    return dict(maxp=maxp, adapt=adapt, reasons=reasons)


def card_uuids():
    """the UUIDs of the visible CUDA devices, in CUDA order (device 0 = the main card)"""
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        if cu.cuInit(0) != 0:
            return []
        n = ctypes.c_int()
        cu.cuDeviceGetCount(ctypes.byref(n))
        out = []
        for i in range(n.value):
            d = ctypes.c_int()
            cu.cuDeviceGet(ctypes.byref(d), i)
            u = (ctypes.c_ubyte * 16)()
            if cu.cuDeviceGetUuid(u, d) != 0:
                return []
            b = bytes(u).hex()
            out.append("GPU-%s-%s-%s-%s-%s" % (b[:8], b[8:12], b[12:16], b[16:20], b[20:]))
        return out
    except OSError:
        return []


def calibrate(L, model_path, cpus, verbose=True):
    """measure (or read the cache), decide, apply; returns the decision"""
    state = os.path.expanduser(os.environ.get("KINDI_STATE_DIR", "~/.kindi/state"))
    key = "|".join([os.path.basename(model_path), ",".join(card_uuids()), cpus])
    path = os.path.join(state, "dsv41-xcal.json")
    cache = {}
    try:
        cache = json.load(open(path))
    except (OSError, ValueError):
        pass
    m = cache.get(key)
    src = "cached"
    if not m or os.environ.get("KINDI_XC_RECAL"):
        out = (ctypes.c_double * 5)()
        t0 = time.time()
        L.dsv_xc_measure(out)
        m = dict(card_ms1=out[0] if out[4] else None, card_ms_pair=out[1] if out[4] else None, cpu_ms_expert=out[2],
                 layer=int(out[3]), when=time.strftime("%Y-%m-%d %H:%M"))
        src = "measured in %.1f s" % (time.time() - t0)
        cache[key] = m
        try:
            os.makedirs(state, exist_ok=True)
            json.dump(cache, open(path + ".tmp", "w"), indent=1)
            os.replace(path + ".tmp", path)
        except OSError:
            pass
    d = decide(m)
    d["measured"], d["source"] = m, src   # for /health (Kindi Monitor shows the decision and what it came from)
    L.dsv_xc_set(-9 if d["maxp"] is None else d["maxp"], d["adapt"])
    if verbose:
        print("dsv41: expert-card calibration (%s): card %s ms + %s ms a pair, CPU %.3f ms an expert" % (
            src, "%.2f" % m["card_ms1"] if m["card_ms1"] is not None else "-",
            "%.3f" % m["card_ms_pair"] if m["card_ms_pair"] is not None else "-", m["cpu_ms_expert"]), file=sys.stderr)
        for r in d["reasons"]:
            print("dsv41:   " + r, file=sys.stderr)
    return d
