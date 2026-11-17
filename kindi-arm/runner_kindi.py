#!/usr/bin/env python3
"""runner_kindi.py — controlled Kindi ARM benchmark (2026-11-17 protocol).
One Engine instance; token-exact verify (same metric as kindi.verify: blended),
then PURE decode/prefill speeds, kc_times stage breakdown, runtime thread/affinity proof.
Usage: python3 runner_kindi.py <pack> <variant-label> <out.json>
Env: KINDI_DEVICE=cpu OMP_NUM_THREADS=4 KINDI_CPU_CACHE=... KINDI_CPU_CFLAGS=...
"""
import ctypes, json, os, sys, threading, time

pack, label, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ.setdefault("KINDI_DEVICE", "cpu")
_here = os.path.dirname(os.path.abspath(__file__))
# Engine lives in kindi/ — next to this script (phone layout) or at the repo root (repo layout).
ROOT = _here if os.path.isdir(os.path.join(_here, "kindi")) else os.path.dirname(_here)
sys.path.insert(0, ROOT)
from kindi.cpu import Engine, _lib
from kindi.tokenizer import Tok

res = {"variant": label, "pack": pack, "env": {k: os.environ.get(k) for k in
       ("OMP_NUM_THREADS", "KINDI_CPU_CFLAGS", "KINDI_CPU_CACHE", "KINDI_DEVICE")}}

def selfstatus():
    d = {}
    for line in open("/proc/self/status"):
        if line.startswith(("Threads:", "Cpus_allowed_list:")):
            k, v = line.split(":", 1); d[k.strip()] = v.strip()
    return d

res["status_at_start"] = selfstatus()
samples = []
stop = False
def sampler():
    while not stop:
        samples.append(selfstatus().get("Threads"))
        time.sleep(0.5)
th = threading.Thread(target=sampler, daemon=True); th.start()

t0 = time.time()
eng = Engine(pack, ctx=1024, threads=int(os.environ.get("KINDI_BENCH_THREADS", "4")), log=lambda *a: res.setdefault("load_log", []).append(" ".join(map(str, a))))
res["load_s"] = round(time.time() - t0, 2)
lib = eng.lib
res["so_path"] = getattr(lib, "_name", "?")

# --- token-exact verify (metric identical to kindi.verify.run) ---
ref = json.load(open(os.path.join(ROOT, "kindi", "reference.json")))
ok_all, tok, dt, prompts = True, 0, 0.0, []
for i, r in enumerate(ref["prompts"]):
    t0 = time.time()
    got = eng.generate(r["ids"], len(r["expected"]))
    t = time.time() - t0
    ok = got == r["expected"]
    ok_all &= ok
    tok += len(r["ids"]) + len(got); dt += t
    prompts.append({"i": i, "ok": ok, "match": sum(a == b for a, b in zip(got, r["expected"])),
                    "n": len(got), "s": round(t, 2)})
res["verify"] = {"all_pass": ok_all, "prompts": prompts, "blended_tok_s": round(tok / dt, 3)}
if os.environ.get("KINDI_BENCH_VERIFY_ONLY"):
    stop = True; time.sleep(0.3)
    res["thread_samples_max"] = max((int(s) for s in samples if s), default=None)
    res["status_at_end"] = selfstatus()
    res["decode"] = []; res["prefill"] = []; res["kc_times_frac"] = {}
    json.dump(res, open(out, "w"), indent=1)
    print("VERIFY_ONLY", ok_all, [p["match"] for p in prompts]); raise SystemExit(0)

# --- pure decode speed (short prompt, timed decode only) ---
vocab = json.load(open(os.path.join(pack, "vocab.json")))
tk = Tok(vocab, os.path.join(pack, "merges.txt"))
short = tk.encode("The quick brown fox")
dec = []
for g in (64, 256):
    eng.reset()
    for i, t in enumerate(short[:-1]): eng.feed(t, i)
    first = eng.step(short[-1], len(short) - 1)
    out_ids, pos = [first], len(short)
    t0 = time.perf_counter()
    for _ in range(g - 1):
        nt = eng.step(out_ids[-1], pos); out_ids.append(nt); pos += 1
    el = time.perf_counter() - t0
    dec.append({"g": g, "s": round(el, 3), "tok_s": round(g / el, 3)})
res["decode"] = dec

# --- prefill speed ---
base = tk.encode("hello world " * 200)
pre = []
for n in (256, 1024):
    ids = (base * ((n // len(base)) + 1))[:n]
    eng.reset()
    t0 = time.perf_counter()
    for i, t in enumerate(ids[:-1]): eng.feed(t, i)
    eng.step(ids[-1], len(ids) - 1)
    el = time.perf_counter() - t0
    pre.append({"n": n, "s": round(el, 3), "tok_s": round(n / el, 3)})
res["prefill"] = pre

# --- kc_times stage breakdown (0 embed 1 norm+fold 2 gemv 3 gdn-other 4 attn-other 5 ffn-other 6 head) ---
try:
    kt = (ctypes.c_double * 8).in_dll(lib, "kc_times")
    tot = sum(kt) or 1.0
    res["kc_times_frac"] = {str(i): round(kt[i] / tot, 4) for i in range(8)}
    res["kc_times_total_s"] = round(tot, 2)
except Exception as e:
    res["kc_times_error"] = str(e)

stop = True; time.sleep(0.6)
res["thread_samples_max"] = max(int(s) for s in samples if s) if samples else None
res["status_at_end"] = selfstatus()
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("variant", "load_s", "verify", "decode", "prefill",
                                      "kc_times_frac", "thread_samples_max", "status_at_start")}, indent=1))