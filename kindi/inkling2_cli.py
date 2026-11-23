"""Local text inference for an Inkling 2 Mini GGUF, and kindi convert for it (backend "kindi_inkling2": the pack
holds only kindi_pack.json; the engine reads the weights from the GGUF file itself)."""
import json
import os
import sys

from .inkling2 import Engine
from .inkling2_tokenizer import Inkling2Tokenizer


def run(args):
    def log(message):
        print(message, file=sys.stderr)

    eng = Engine(args.gguf, ctx=args.ctx or 2048, log=log)
    try:
        tok = Inkling2Tokenizer(eng.gguf.kv)
        prompt = args.prompt
        if not args.raw:
            from .inkling2_chat import render
            prompt = render(eng.model_path, [{"role": "user", "content": prompt}], enable_thinking=False)
        ids = tok.encode(prompt, special=not args.raw)
        if not ids or len(ids) > eng.ctx:
            log("inkling: prompt must encode to between 1 and %d tokens" % eng.ctx)
            return 2
        stop = {tok.eos, tok.eot}
        out = eng.generate(ids, args.max_tokens, stop=stop)
        if out and out[-1] in stop:
            out.pop()
        print(tok.decode(out))
        return 0
    finally:
        eng.close()


def is_inkling(g):
    return g.kv.get("general.architecture") == "inkling2_loop"


def convert(g, gguf_path, out_dir, version, pack_format, sha256, log=print):
    from collections import Counter
    os.makedirs(out_dir, exist_ok=True)
    kv = g.kv
    man = {
        "kindi_version": version, "pack_format": pack_format, "backend": "kindi_inkling2",
        "model": kv.get("general.name"), "model_id": "inkling-2-mini",
        "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
        "source_sha256": sha256(gguf_path), "context_length": kv.get("inkling2_loop.context_length"),
        "tensor_types": dict(Counter(str(t) for _, _, t, _ in g.tensors)),
    }
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(man, f, indent=2)
    log("recognized %s (Inkling 2, %d tensors): Kindi Inkling engine; the weights stay in the GGUF file"
        % (man["model"], len(g.tensors)))
    return man
