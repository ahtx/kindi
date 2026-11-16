"""kindi convert / verify / bench for Nemotron-H MoE GGUF files (backend "kindi_lightning"). The pack holds only
kindi_pack.json: the engine (lightning.py) reads the weights from the GGUF file itself."""
import json
import os
import random
import time

SUPPORTED_TYPES = {0, 8, 42}                       # F32, Q8_0, Q2_0


def is_lightning(g):
    return g.kv.get("general.architecture") == "nemotron_h_moe"


def convert(g, gguf_path, out_dir, version, pack_format, sha256, log=print):
    from collections import Counter
    types = {t for _, _, t, _ in g.tensors}
    if not types <= SUPPORTED_TYPES:
        raise SystemExit("kindi convert: this Nemotron-H MoE file has tensor types %s; Kindi reads F32, Q8_0 and "
                         "Q2_0" % sorted(types - SUPPORTED_TYPES))
    os.makedirs(out_dir, exist_ok=True)
    kv = g.kv
    man = {
        "kindi_version": version, "pack_format": pack_format, "backend": "kindi_lightning",
        "model": kv.get("general.name"), "model_id": "nemotron-lightning-30b",
        "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
        "source_sha256": sha256(gguf_path), "context_length": kv.get("nemotron_h_moe.context_length"),
        "tensor_types": dict(Counter(str(t) for _, _, t, _ in g.tensors)),
    }
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("recognized %s (Nemotron-H MoE, %d tensors): Kindi Lightning engine; the weights stay in the GGUF file"
        % (man["model"], len(g.tensors)))
    return man


def _engine(cfg, man, ctx, log):
    from .lightning import Engine
    return Engine(man["model_path"], ctx=ctx, log=log)


def noise_floor(ref):
    """the largest llama.cpp top-2 gap at which llama.cpp's own run with one token per batch (alt) leaves its default
    run: below it, llama.cpp itself does not decide the token"""
    floor = 0.0
    for r in ref["prompts"]:
        e, a, g = r["expected"], r.get("alt") or [], r.get("gap") or []
        k = next((i for i in range(min(len(e), len(a))) if e[i] != a[i]), None)
        if k is not None and k < len(g) and g[k] is not None:
            floor = max(floor, g[k])
    return floor


def verify(cfg, man, log=print):
    """greedy output against llama.cpp (CUDA, default batch) on the reference prompts (raw, chat, 512 / 2K / 8K /
    32K). A prompt passes when every token agrees. It is a tie when its first difference is a point where
    llama.cpp's own run with one token per batch (alt) gives Kindi's token, or where llama.cpp's top-2 gap is under
    the noise floor (noise_floor(): the largest gap at which llama.cpp's two runs disagree, measured in the
    reference). Any other difference fails. The tokenizer and the chat template are checked on the same prompts."""
    from .gguf import GGUF
    from .lightning_chat import render
    from .lightning_tokenizer import LightningTokenizer
    here = os.path.dirname(os.path.abspath(__file__))
    rp = os.path.join(here, "reference-%s.json" % man.get("model_id", ""))
    if not os.path.exists(rp):
        log("no llama.cpp reference for this model in this version of Kindi")
        return 2
    ref = json.load(open(rp))
    floor = noise_floor(ref)
    tok = LightningTokenizer(GGUF(man["model_path"]).kv)
    ok_tok = True
    for r in ref["prompts"]:
        if "messages" in r:
            text = render(man["model_path"], r["messages"])
            if text != r.get("template", text):
                ok_tok = False
                log("chat template: %s differs from llama.cpp" % r["prompt"][:50])
            ids = tok.encode(text)
        elif "text" in r:
            ids = tok.encode(r["text"])
        else:
            continue
        if ids != r["ids"]:
            ok_tok = False
            log("tokenizer: %s differs from llama.cpp" % r["prompt"][:50])
    log("tokenizer and chat template: %s" % ("the same ids as llama.cpp" if ok_tok else "DIFFERENT ids"))
    need = max(len(r["ids"]) + len(r["expected"]) for r in ref["prompts"])
    eng = _engine(cfg, man, need + 64, log)
    fails = ties = 0
    t0 = time.time()
    for r in ref["prompts"]:
        want = r["expected"]
        got = eng.generate(r["ids"], len(want), stop=(tok.eos,))
        pre = next((k for k in range(len(want)) if k >= len(got) or got[k] != want[k]), len(want))
        gap = r["gap"][pre] if pre < len(want) and r.get("gap") else None
        alt = r.get("alt") or []
        if pre == len(want):
            state = "PASS"
        elif (pre < len(alt) and alt[:pre + 1] == got[:pre + 1]) or (gap is not None and gap <= floor):
            state = "TIE"
            ties += 1
        else:
            state = "FAIL"
            fails += 1
        why = ""
        if state != "PASS":
            why = " (llama.cpp top-2 gap there %.4f%s)" % (gap, "; llama.cpp -ub 1 gives Kindi's token"
                                                             if pre < len(alt) and alt[pre] == got[pre] else "")
        log("%-4s %-44s %2d of %2d tokens agree%s" % (state, repr(r["prompt"])[:44], pre, len(want), why))
    n = len(ref["prompts"])
    log("%s: %d of %d prompts token-exact, %d end at a llama.cpp tie (noise floor %.3f), %d fail (%.0f s)" % (
        "PASS" if not fails and ok_tok else "FAIL", n - ties - fails, n, ties, floor, fails, time.time() - t0))
    return 0 if not fails and ok_tok else 1


def bench(cfg, man, log=print):
    """write speed at positions 0 / 8K / 32K (64 tokens after a 16-token warm-up) and read speed of 512 / 8K / 32K
    token prompts, for the configured context (ctx 0: the largest that fits)"""
    ctx = int(cfg.get("ctx") or 0)
    from .lightning import Engine
    eng = Engine(man["model_path"], ctx=ctx or 33280, log=log)
    sync = eng.stream.synchronize
    random.seed(7)
    rows = []
    for n in [x for x in (512, 8192, 32768) if x + 96 <= eng.ctx]:
        ids = [random.randrange(1000, 100000) for _ in range(n)]
        eng.reset()
        eng.prefill(ids[:64], 0)
        eng.reset()
        sync()
        t0 = time.time()
        t = eng.prefill(ids, 0)
        tr = time.time() - t0
        for _ in range(16):
            t = eng.step(t)
        t0 = time.time()
        for _ in range(64):
            t = eng.step(t)
        tw = time.time() - t0
        rows.append((n, n / tr, 64 / tw))
        log("context %5d: read %7.1f tokens/s, write %6.2f tokens/s (at position %d)" % (n, n / tr, 64 / tw, n + 16))
    return 0
