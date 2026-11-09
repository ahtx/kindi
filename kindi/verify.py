"""kindi verify: greedy output vs llama.cpp's reference for the same model file (4 prompts, the first 40 tokens
llama.cpp generated from them at temperature 0). Every token must match. reference.json is the 2B; reference-4b.json,
-9b.json and -27b.json the others (the 27B with llama.cpp's q8_0 K/V cache, as Kindi's 27B engine)."""
import json
import os
import re
import time


def reference_for(pack):
    """the reference file of the pack's model (by the size in its source file name), or None"""
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    m = re.search(r"-(\d+)B[-_.]", man.get("source_file", "") + ".") or re.search(r"(\d+)B", man.get("model", ""))
    size = (m.group(1) + "b") if m else "2b"
    p = os.path.join(here, "reference.json" if size == "2b" else "reference-%s.json" % size)
    return p if os.path.exists(p) else None


GEMMA4_MIN_PREFIX = 12      # Gemma 4: every prompt must agree with llama.cpp on at least this many first tokens


def run_gemma4(cfg, man, log=print):
    """Nemotron / Gemma 4: Kindi quantizes activations to int8 and sums in its own order, so it is not token-exact
    with llama.cpp. Each prompt must agree with llama.cpp's greedy output on its first GEMMA4_MIN_PREFIX tokens."""
    import cupy as cp
    here = os.path.dirname(os.path.abspath(__file__))
    rp = os.path.join(here, "reference-%s.json" % man.get("model_id", ""))
    if not os.path.exists(rp):
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    man = dict(man, pack_path=cfg["pack"])
    from .server import _two_cards
    if _two_cards():
        from .gemma4_tp import TPEngine as E
    else:
        from .gemma4 import Engine as E
    eng = E(man, ctx=2048, log=log)
    ok_all, tok, dt = True, 0, 0.0
    for i, r in enumerate(ref["prompts"]):
        ids, want = r["ids"], r["expected"]
        cp.cuda.Device().synchronize(); t0 = time.time()
        eng.forward_batch(ids[:-1], 0); t = int(cp.argmax(eng.forward(ids[-1], len(ids) - 1)).item()); got = [t]
        for pos in range(len(ids), len(ids) + len(want) - 1):
            t = int(cp.argmax(eng.forward(t, pos)).item()); got.append(t)
        cp.cuda.Device().synchronize(); dt += time.time() - t0; tok += len(ids) + len(got)
        pre = next((k for k in range(len(want)) if got[k] != want[k]), len(want))
        ok = pre >= GEMMA4_MIN_PREFIX
        ok_all &= ok
        log("prompt %d %-60s %s  the first %d of %d tokens agree" % (i + 1, repr(r["prompt"])[:60], "PASS" if ok else "FAIL",
                                                                     pre, len(want)))
    log("%s - %.1f tokens/s over the run" % (("ALL PASS: each prompt agrees with llama.cpp on its first %d tokens or more"
                                              % GEMMA4_MIN_PREFIX) if ok_all else "FAIL", tok / dt))
    return 0 if ok_all else 1


# a first difference where llama.cpp's own top-2 log-probability gap (reference "gap", tools/bonsai_gaps.py) is under
# TIE_GAP nats, or where Kindi gives the token of another llama.cpp run ("alt"), is a tie, as in nanbeige_cli
TIE_GAP = 0.01


def run(cfg, log=print):
    try:
        import cupy as cp
    except ImportError:
        cp = None
    from .engine import make_engine
    import numpy as np
    pack = cfg["pack"]
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    if man.get("backend") == "kindi_gemma4":
        return run_gemma4(cfg, man, log=log)
    if man.get("backend") == "kindi_nanbeige":
        from .nanbeige_cli import verify
        return verify(cfg, man, log=log)
    rp = reference_for(pack)
    if rp is None:
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    cl = os.path.join(pack, "head_clusters.npz")
    eng = make_engine(pack, ctx=1024, clusters=dict(np.load(cl)) if os.path.exists(cl) else None, log=log)
    ok_all, tok, dt, ties = True, 0, 0.0, 0
    for i, r in enumerate(ref["prompts"]):
        sync = (lambda: cp.cuda.Device().synchronize()) if cp is not None and hasattr(eng, "k") else (lambda: None)
        sync()
        t0 = time.time()
        got = eng.generate(r["ids"], len(r["expected"]))
        sync()
        t = time.time() - t0
        n = sum(1 for a, b in zip(got, r["expected"]) if a == b)
        ok = got == r["expected"]
        state, j, gap = "PASS" if ok else "FAIL", None, None
        if not ok:
            j = next(k for k, (a, b) in enumerate(zip(got, r["expected"])) if a != b)
            gap = r["gap"][j] if r.get("gap") and j < len(r["gap"]) else None
            alt = r.get("alt") or []
            if (gap is not None and gap < TIE_GAP) or (j < len(alt) and alt[:j + 1] == got[:j + 1]):
                # a dead heat in llama.cpp's own top two, or the token another llama.cpp run of the same file gives
                # there (tools/bonsai_gaps.py "alt"): the Nanbeige backend's rule
                state = "TIE"
                ties += 1
        ok_all &= state != "FAIL"
        tok += len(r["ids"]) + len(got)
        dt += t
        log("prompt %d %-60s %s  %d/%d tokens match  (%.1fs)" % (i + 1, repr(r["prompt"])[:60], state, n, len(r["expected"]), t))
        if not ok:
            log("   first difference at token %d%s" % (j, " (llama.cpp top-2 gap there %.4f)" % gap if gap is not None else ""))
    if ok_all and ties:
        log("PASS: %d of %d prompts token-exact, %d end at a llama.cpp tie, 0 fail - %.1f tokens/s over the run" % (
            len(ref["prompts"]) - ties, len(ref["prompts"]), ties, tok / dt))
    else:
        log("%s - %.1f tokens/s over the run" % ("ALL PASS: token-exact with llama.cpp" if ok_all else "FAIL", tok / dt))
    return 0 if ok_all else 1
