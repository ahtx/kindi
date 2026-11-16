"""kindi convert / verify / bench for llama-architecture K-quant and I-quant GGUF files (backend "kindi_nanbeige").
The pack holds only kindi_pack.json: the engine (nanbeige.py) reads the weights from the GGUF file itself."""
import json
import os
import time

SUPPORTED_TYPES = {0, 12, 13, 18, 21, 22}         # F32, Q4_K, Q5_K, IQ3_XXS, IQ3_S, IQ2_S
TIE_GAP = 0.01                                     # nats: a dead heat between llama.cpp's top two tokens


def is_llama_iq(g):
    """a llama-architecture GGUF that the nanbeige engine runs"""
    if g.kv.get("general.architecture") != "llama":
        return False
    return {t for _, _, t, _ in g.tensors} <= SUPPORTED_TYPES


def model_id(g):
    name = (g.kv.get("general.basename") or g.kv.get("general.name") or "llama").lower().replace(" ", "")
    size = (g.kv.get("general.size_label") or "").lower()
    return "%s-%s" % (name.split(".")[0].rstrip("0123456789"), size) if size else name


def convert(g, gguf_path, out_dir, version, pack_format, sha256, log=print):
    from collections import Counter
    os.makedirs(out_dir, exist_ok=True)
    kv = g.kv
    man = {
        "kindi_version": version, "pack_format": pack_format, "backend": "kindi_nanbeige",
        "model": kv.get("general.name"), "model_id": model_id(g),
        "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
        "source_sha256": sha256(gguf_path), "context_length": kv.get("llama.context_length"),
        "tensor_types": dict(Counter(str(t) for _, _, t, _ in g.tensors)),
        "chat_template": kv.get("tokenizer.chat_template", ""),
    }
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("recognized %s (llama architecture, %d tensors): Kindi I-quant engine; the weights stay in the GGUF file"
        % (man["model"], len(g.tensors)))
    return man


def _engine(cfg, man, ctx, log):
    from .nanbeige import Engine
    return Engine(man["model_path"], ctx=ctx, kv=(cfg or {}).get("kv_cache"), log=log)


def _reference(here, man, kv):
    """the llama.cpp reference for this model file and K/V cache type, reference-<model id>-<first 8 digits of the
    file's sha256>-<kv>.json; else the model's first reference, reference-<model id>.json (its f16 K/V file)"""
    mid, sha = man.get("model_id", ""), man.get("source_sha256") or ""
    names = ["reference-%s-%s-%s.json" % (mid, sha[:8], kv)]
    if kv == "f16":                                    # only the f16 cache falls back to the model's first reference
        names.append("reference-%s.json" % mid)
    for n in names:
        if os.path.exists(os.path.join(here, n)):
            return os.path.join(here, n)
    return None


def verify(cfg, man, log=print):
    """greedy output against llama.cpp (CUDA, default batch) on the reference prompts (raw, chat, 512 / 2K / 8K).
    Every token must agree, with one exception: llama.cpp itself gives other tokens with another batch size (-ub 1,
    the "alt" run of the reference) at points where its top two choices are close. A prompt whose first difference
    is such a point (Kindi gives the token of the alt run there, or the top-2 gap is under TIE_GAP) is a tie and
    ends there. Any other difference fails. The tokenizer and the chat template are checked on the same prompts."""
    from .gguf import GGUF
    from .spm import SPM
    from .nanbeige_chat import render, prompt_ids
    here = os.path.dirname(os.path.abspath(__file__))
    want_kv = (cfg or {}).get("kv_cache") or os.environ.get("KINDI_KV_CACHE") or ""
    rp = _reference(here, man, want_kv if want_kv in ("f16", "q8_0") else "f16")
    if rp is None:
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    need = max(len(r["ids"]) + len(r["expected"]) for r in ref["prompts"])
    eng = _engine(cfg, man, need + 64, log)
    if eng.kv != (want_kv or "f16"):                   # kv_cache auto: the reference of the cache the engine chose
        rp = _reference(here, man, eng.kv)
        if eng.kv != "f16" and (rp is None or not os.path.basename(rp).endswith("-%s.json" % eng.kv)):
            log("no llama.cpp reference for kv_cache %s" % eng.kv)
            return 2
        ref = json.load(open(rp))
    log("reference %s: %s" % (os.path.basename(rp), ref["source"].split(";")[0]))
    tok = SPM(GGUF(man["model_path"]).kv)
    ok_tok = True
    for r in ref["prompts"]:
        if "messages" in r:
            ids = prompt_ids(tok, render(r["messages"]))
        elif "text" in r:
            ids = tok.encode(r["text"])
        else:
            continue
        if ids != r["ids"]:
            ok_tok = False
            log("tokenizer / template: %s differs from llama.cpp" % r["prompt"][:50])
    log("tokenizer and chat template: %s" % ("the same ids as llama.cpp" if ok_tok else "DIFFERENT ids"))
    fails = ties = 0
    t0 = time.time()
    for r in ref["prompts"]:
        want = r["expected"]
        got = eng.generate(r["ids"], len(want))
        pre = next((k for k in range(len(want)) if got[k] != want[k]), len(want))
        gap = r["gap"][pre] if pre < len(want) and r.get("gap") else None
        alt = r.get("alt") or []
        if pre == len(want):
            state = "PASS"
        elif (gap is not None and gap < TIE_GAP) or (pre < len(alt) and alt[:pre + 1] == got[:pre + 1]):
            state = "TIE"                              # a dead heat, or the token llama.cpp itself gives with -ub 1
            ties += 1
        else:
            state = "FAIL"
            fails += 1
        log("%-4s %-44s %2d of %2d tokens agree%s" % (state, repr(r["prompt"])[:44], pre, len(want),
            "" if state == "PASS" else " (llama.cpp top-2 gap there %.4f)" % gap if gap is not None else ""))
    n = len(ref["prompts"])
    drift = ref.get("f16_drift")
    if eng.kv != "f16" and drift:
        # a lossy K/V cache: two q8_0 implementations round differently, and llama.cpp's own q8_0 run differs from its
        # -ub 1 run at many points, so token-exact agreement with it is reported but is not the test. The test: the
        # tokens that agree with the f16 reference before the first difference, summed over the prompts, are at least
        # as many as llama.cpp's own q8_0 run keeps (the cache drifts from f16 no more than llama.cpp's q8_0 does)
        log("against llama.cpp q8_0: %d of %d prompts token-exact, %d at a tie, %d differ (information only)"
            % (n - ties - fails, n, ties, fails))
        base = json.load(open(_reference(here, man, "f16")))
        agree = 0
        for r in base["prompts"]:
            want = r["expected"]
            got = eng.generate(r["ids"], len(want))
            agree += next((k for k in range(len(want)) if got[k] != want[k]), len(want))
        ok = agree >= drift["tokens"] and ok_tok
        log("%s: q8_0 drift from f16: Kindi keeps %d of %d tokens, llama.cpp q8_0 keeps %d (%.0f s)" % (
            "PASS" if ok else "FAIL", agree, drift["of"], drift["tokens"], time.time() - t0))
        return 0 if ok else 1
    log("%s: %d of %d prompts token-exact, %d end at a llama.cpp tie, %d fail (%.0f s)" % (
        "PASS" if not fails and ok_tok else "FAIL", n - ties - fails, n, ties, fails, time.time() - t0))
    return 0 if not fails and ok_tok else 1


def bench(cfg, man, log=print):
    import random
    ctx = int(cfg.get("ctx") or 16640)
    eng = _engine(cfg, man, ctx, log)
    for d in [x for x in (0, 2048, 8192, 16384) if x + 140 < eng.ctx]:
        t = eng.step(1000, d)
        for i in range(4):
            t = eng.step(t, d + 1 + i)
        t0 = time.time()
        for i in range(128):
            t = eng.step(t, d + 5 + i)
        log("writing at position %5d: %6.2f tokens/s" % (d, 128 / (time.time() - t0)))
    for n in [x for x in (512, 2048, 8192, 16384) if x + 8 < eng.ctx]:
        ids = [random.randrange(1000, 100000) for _ in range(n)]
        eng.prefill_batch(ids[:64])
        t0 = time.time()
        eng.prefill_batch(ids)
        log("reading a %5d-token prompt: %7.1f tokens/s" % (n, n / (time.time() - t0)))
    return 0
