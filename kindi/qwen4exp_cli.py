"""kindi convert / verify for Qwen3.8-Flash-Next GGUF files (architecture qwen4exp; backend "kindi_qwen4exp"). The
pack holds only kindi_pack.json: the engine (kindi.qwen4exp_gpu) reads the weights from the GGUF shards.

kindi verify is the Flash-Next gate (tools/qwen4exp_gate.py's test, on the card engine): the reference prompts and
the black-box greedy run with its top-2 log-probabilities (kindi/reference-qwen38-flash-next.jsonl, 8 prompts, 174
generated tokens) are read teacher-forced, and Kindi must pick the same top-1 token at every generated position. A
miss counts as a near-tie when the reference's own top-2 log-probabilities are within 0.35 nats (the gate's
--tie). It passes with no miss outside a near-tie; on the release file it gives 173 of 174 with the one miss a
near-tie (gap 0.14)."""
import json
import os
import time

MODEL_ID = "qwen38-flash-next"
TIE = 0.35


def is_qwen4exp(g):
    return g.kv.get("general.architecture") == "qwen4exp"


def convert(g, gguf_path, out_dir, version, pack_format, sha256=None, log=print):
    from .packcheck import fast_hash
    from .qwen4exp_ref import shards
    kv = g.kv
    files = shards(gguf_path)
    os.makedirs(out_dir, exist_ok=True)
    man = {
        "kindi_version": version, "pack_format": pack_format, "backend": "kindi_qwen4exp",
        "model": kv.get("general.name"), "model_id": MODEL_ID,
        "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
        # the shards' sizes and fast hashes (the installer checks their sha256 at download; a full hash of 83 GB here
        # would take minutes): kindi serve and kindi verify check the sizes before they load
        "shards": [{"file": os.path.basename(f), "size": os.path.getsize(f), "fast": fast_hash(f)} for f in files],
        "source_sha256": None,
        "context_length": kv.get("qwen4exp.context_length"),
    }
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("recognized %s (qwen4exp, %d shards): the Kindi Flash-Next engine; the weights stay in the GGUF files"
        % (man["model"], len(files)))
    return man


def check_shards(man):
    """the shards are the ones the pack was made from (size; fast hash): raises SystemExit naming the file"""
    from .packcheck import fast_hash
    d = os.path.dirname(man["model_path"])
    for s in man.get("shards") or []:
        p = os.path.join(d, s["file"])
        if not os.path.exists(p):
            raise SystemExit("kindi: the Flash-Next shard %s is missing" % p)
        if os.path.getsize(p) != s["size"]:
            raise SystemExit("kindi: the Flash-Next shard %s has %d bytes, the pack was made with %d"
                             % (p, os.path.getsize(p), s["size"]))
        if fast_hash(p) != s["fast"]:
            raise SystemExit("kindi: the Flash-Next shard %s is not the file the pack was made from (fast hash)" % p)


def reference():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "reference-%s.jsonl" % MODEL_ID)


def gate(model, rows, log=print, tie=TIE):
    """teacher-forced top-1 agreement with the reference rows; returns the summary dict"""
    agree = total = ties = 0
    for r in rows:
        ids, gen = r["ids"], r["gen"]
        seq = ids + [g["id"] for g in gen]
        t1 = time.time()
        lg = model.logits(seq[:-1], last_n=len(gen))
        mine = lg.argmax(-1).cpu().tolist()
        miss = []
        for i, (g, k) in enumerate(zip(gen, mine)):
            total += 1
            if k == g["id"]:
                agree += 1
                continue
            top = g.get("top") or []
            gap = (top[0][1] - top[1][1]) if len(top) >= 2 and None not in (top[0][1], top[1][1]) else None
            t = gap is not None and abs(gap) <= tie
            ties += t
            miss.append("token %d (reference gap %s%s)" % (i, "%.3f" % gap if gap is not None else "?",
                                                           ", a near-tie" if t else ""))
        log("prompt %-40r %s  %d/%d tokens  (%.1fs)%s" % (r.get("prompt", "")[:40], "PASS" if not miss else (
            "TIE" if all("near-tie" in x for x in miss) else "FAIL"), len(gen) - len(miss), len(gen),
            time.time() - t1, ("  " + "; ".join(miss)) if miss else ""))
    return {"agree": agree, "total": total, "near_tie_misses": ties, "pass": agree + ties == total}


def verify(cfg, man, log=print):
    rp = reference()
    if not os.path.exists(rp):
        log("no reference for this model in this version of Kindi")
        return 2
    check_shards(man)
    rows = [json.loads(l) for l in open(rp) if l.strip()]
    # verify checks the exact path against the reference: fast mode (the default) is off here unless set explicitly
    os.environ.setdefault("KINDI_FAST", "0")
    from .qwen4exp_gen import Engine
    t0 = time.time()
    eng = Engine(man["model_path"], ctx=0, log=log)
    s = gate(eng.gm, rows, log=log)
    log("%s: %d of %d tokens the reference's top-1, %d misses at near-ties (%.0f s)" % (
        "PASS" if s["pass"] else "FAIL", s["agree"], s["total"], s["near_tie_misses"], time.time() - t0))
    return 0 if s["pass"] else 1
