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
    return Engine(man["model_path"], ctx=ctx, log=log)


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
    rp = os.path.join(here, "reference-%s.json" % man.get("model_id", ""))
    if not os.path.exists(rp):
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
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
    need = max(len(r["ids"]) + len(r["expected"]) for r in ref["prompts"])
    eng = _engine(cfg, man, need + 64, log)
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
    log("%s: %d of %d prompts token-exact, %d end at a llama.cpp tie, %d fail (%.0f s)" % (
        "PASS" if not fails and ok_tok else "FAIL", n - ties - fails, n, ties, fails, time.time() - t0))
    return 0 if not fails and ok_tok else 1


def bench(cfg, man, log=print):
    import random
    ctx = int(cfg.get("ctx") or 16640)
    eng = _engine(cfg, man, ctx, log)
    for d in [x for x in (0, 2048, 8192, 16384) if x + 80 < eng.ctx]:
        t = eng.step(1000, d)
        for i in range(4):
            t = eng.step(t, d + 1 + i)
        t0 = time.time()
        for i in range(64):
            t = eng.step(t, d + 5 + i)
        log("writing at position %5d: %6.2f tokens/s" % (d, 64 / (time.time() - t0)))
    for n in [x for x in (512, 2048, 8192, 16384) if x + 8 < eng.ctx]:
        ids = [random.randrange(1000, 100000) for _ in range(n)]
        eng.prefill_batch(ids[:64])
        t0 = time.time()
        eng.prefill_batch(ids)
        log("reading a %5d-token prompt: %7.1f tokens/s" % (n, n / (time.time() - t0)))
    return 0
