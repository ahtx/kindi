"""kindi convert: supported Ternary-Bonsai-2 PTQ1_0 GGUF -> Kindi weight pack. Exact (same information, new layout).

Ternary tensors (GGUF type 143 = PTQ1_0: per 128 weights, qs[24] | qh[2] | fp16 scale d; 5 trits per byte) are
re-coded 4 trits per byte (byte = t0 + 3 t1 + 9 t2 + 27 t3, t in {0,1,2}, weight = (t-1) * d) and laid out
slot-major ([slot][row]); the scales are transposed to [phase][row]. The engine repacks this at load time into its
own layout (layout.py). The Hadamard rotation Prism ML folded into the weights stays folded: the engine applies the
matching transform to the activations, using the sign vectors stored in the GGUF (prism.hadamard.*).
Float32 tensors are copied verbatim. Vocabulary and BPE merges come from the GGUF tokenizer metadata.

Pack files: bcodes.bin, dpm.bin, f32.bin, signs_2048.f32, signs_6144.f32, vocab.json, merges.txt, index.json,
bcodes_index.json, kindi_pack.json."""
import hashlib
import json
import os
import time

import numpy as np

from . import __version__
from .gguf import GGUF

PTQ1_0, F32, BF16 = 143, 0, 30
POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint32)
PROFILES = {
    (24, 2048, 6144, 8, 2, 16): "Ternary-Bonsai-2-2B",
    (32, 2560, 9216, 16, 4, 16): "Ternary-Bonsai-2-4B",
    (32, 4096, 12288, 16, 4, 16): "Ternary-Bonsai-2-9B",
    (64, 5120, 17408, 24, 4, 16): "Ternary-Bonsai-2-27B",
}
PACK_FORMAT = 1


def trits_from_raw(raw):
    """raw (nb, 28) uint8 PTQ1_0 blocks -> (nb, 128) trits in {0,1,2} (ggml's staged element order)."""
    nb = raw.shape[0]
    out = np.empty((nb, 128), dtype=np.uint8)
    for n in range(5):
        q = (raw[:, 0:16].astype(np.uint32) * POW3[n]) & 0xFF
        out[:, n * 16:(n + 1) * 16] = (q * 3) >> 8
    for n in range(5):
        q = (raw[:, 16:24].astype(np.uint32) * POW3[n]) & 0xFF
        out[:, 80 + n * 8:88 + n * 8] = (q * 3) >> 8
    for n in range(4):
        q = (raw[:, 24:26].astype(np.uint32) * POW3[n]) & 0xFF
        out[:, 120 + n * 2:122 + n * 2] = (q * 3) >> 8
    return out


def phase_bytes(tr):
    """(nb, 128) trits -> (nb, 32) bytes, byte s = t0 + 3 t1 + 9 t2 + 27 t3 of elements 4s .. 4s+3."""
    g = tr.reshape(-1, 32, 4).astype(np.int32)
    return (g[:, :, 0] + 3 * g[:, :, 1] + 9 * g[:, :, 2] + 27 * g[:, :, 3]).astype(np.uint8)


def selftest():
    rng = np.random.RandomState(7)
    tr = rng.randint(0, 3, size=(4096, 128)).astype(np.uint8)
    b = phase_bytes(tr).astype(np.int32)
    back = np.stack([b % 3, (b // 3) % 3, (b // 9) % 3, (b // 27) % 3], axis=2).reshape(-1, 128)
    assert np.array_equal(back, tr), "trit/byte round trip failed"


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            c = f.read(bufsize)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def convert(gguf_path, out_dir, log=print):
    t0 = time.time()
    selftest()
    g = GGUF(gguf_path)
    if g.kv.get("general.architecture") == "nemotron_h_moe":
        from .packed import PackedFile
        from .lightning import ModelSpec
        source = PackedFile(gguf_path)
        spec = ModelSpec(source.gguf)
        manifest = {"kindi_version": __version__, "pack_format": 1, "backend": "kindi_lightning",
            "model": g.kv.get("general.name"), "model_id": "nemotron-lightning-30b-a3b-ternary",
            "model_path": os.path.abspath(gguf_path), "source_sha256": source.source_hash(),
            "context_length": spec.max_context, "tensor_count": len(source.tensors), "source_file": os.path.basename(gguf_path),
            "adapter": "nota_nemotron_3_5_lightning", "router": "sigmoid_corrected_topk", "expert_activation": "relu2",
            "position_encoding": "none", "weight_storage": "gguf_q2_0_q8_0_f32"}
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f: json.dump(manifest, f, indent=2)
        log("recognized Lightning GGUF; native Kindi CUDA backend")
        return manifest
    if g.kv.get("general.architecture") == "gemma4":
        from .gemma4 import inspect_model
        info = inspect_model(gguf_path)
        os.makedirs(out_dir, exist_ok=True)
        manifest = {
            "kindi_version": __version__, "pack_format": PACK_FORMAT, "backend": "kindi_gemma4",
            "model": info["name"], "model_id": "nemotron-bolt-e4b" if "Nemotron" in info["name"] else "gemma4",
            "source_file": os.path.basename(gguf_path), "model_path": os.path.abspath(gguf_path),
            "source_sha256": sha256(gguf_path), "context_length": info["context_length"],
            "dimensions": info["dimensions"], "tensor_count": info["tensor_count"],
            "tensor_types": info["tensor_types"], "native_status": "Kindi Gemma 4 CUDA decoder",
        }
        with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        log("recognized Gemma 4 GGUF: %d tensors, %s quantization; Kindi native CUDA decoder" %
            (info["tensor_count"], info["tensor_types"]))
        return manifest
    if g.kv.get("general.architecture") != "qwen35":
        raise SystemExit("kindi convert: expected a qwen35 or Gemma 4 GGUF")
    profile = (g.kv.get("qwen35.block_count"), g.kv.get("qwen35.embedding_length"),
               g.kv.get("qwen35.feed_forward_length"), g.kv.get("qwen35.attention.head_count"),
               g.kv.get("qwen35.attention.head_count_kv"), g.kv.get("qwen35.ssm.group_count"))
    if profile not in PROFILES:
        raise SystemExit("kindi convert: unsupported Ternary-Bonsai-2 dimensions %r" % (profile,))
    types = {t[2] for t in g.tensors}
    if not types <= {PTQ1_0, F32, BF16}:
        raise SystemExit("kindi convert: needs the PTQ1_0 GGUF (found tensor types %s)" % sorted(types))
    hd = {k.split("prism.hadamard.")[-1]: v for k, v in g.kv.items() if k.startswith("prism.hadamard.")}
    sv = np.array(hd["sign_values"], dtype=np.float32)
    signs, off = {}, 0
    for w in hd["sign_widths"]:
        signs[int(w)] = sv[off:off + int(w)]
        off += int(w)
    folded = set(hd.get("weight_names", []))
    if hd.get("tied_output"):
        folded |= set(hd.get("inverse_weight_names", []))
    os.makedirs(out_dir, exist_ok=True)
    for width, values in signs.items():
        values.astype("<f4").tofile(os.path.join(out_dir, "signs_%d.f32" % width))
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(g.kv["tokenizer.ggml.tokens"], f)
    with open(os.path.join(out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for m in g.kv["tokenizer.ggml.merges"]:
            f.write(m + "\n")

    index = {"tensors": {}, "base": g.data_start,
             "hadamard": {"tied_output": bool(hd.get("tied_output")), "sign_widths": [int(x) for x in hd["sign_widths"]]}}
    bindex = {"tensors": {}}
    fb = open(os.path.join(out_dir, "bcodes.bin"), "wb")
    fd = open(os.path.join(out_dir, "dpm.bin"), "wb")
    ff = open(os.path.join(out_dir, "f32.bin"), "wb")
    src = open(gguf_path, "rb")
    bc_off = dpm_off = f32_off = 0
    n_t = 0
    for name, ne, ty, toff in g.tensors:
        ne0 = ne[0]
        rows = 1
        for x in ne[1:]:
            rows *= x
        rec = {"dims": list(reversed(ne)), "type": ty, "folded": name in folded, "rows": rows, "ne0": ne0}
        src.seek(g.data_start + toff)
        if ty == PTQ1_0:
            bpr = ne0 // 128
            out = np.empty((rows, bpr * 32), dtype=np.uint8)
            dm = np.empty((rows, bpr), dtype=np.uint16)
            RC = 16384
            for c0 in range(0, rows, RC):
                c1 = min(rows, c0 + RC)
                raw = np.frombuffer(src.read(28 * (c1 - c0) * bpr), dtype=np.uint8).reshape((c1 - c0) * bpr, 28)
                out[c0:c1] = phase_bytes(trits_from_raw(raw)).reshape(c1 - c0, bpr * 32)
                dm[c0:c1] = np.frombuffer(raw[:, 26:28].tobytes(), dtype=np.uint16).reshape(c1 - c0, bpr)
            fb.write(np.ascontiguousarray(out.T).tobytes())          # [slot][row]
            fd.write(np.ascontiguousarray(dm.T).tobytes())           # [phase][row]
            rec.update({"bc_off": bc_off, "dpm_off": dpm_off})
            bindex["tensors"][name] = {"bc_off": bc_off, "dpm_off": dpm_off, "rows": rows, "ne0": ne0}
            bc_off += rows * bpr * 32
            dpm_off += rows * bpr * 2
        else:
            n = rows * ne0
            if ty == BF16:                      # the 27B stores ssm_alpha / ssm_beta in bf16: widen exactly to f32
                v = np.frombuffer(src.read(2 * n), dtype="<u2").astype(np.uint32) << 16
                ff.write(v.astype("<u4").tobytes())
            else:
                ff.write(np.frombuffer(src.read(4 * n), dtype="<f4").tobytes())
            rec.update({"f32_off": f32_off, "nfloats": n})
            f32_off += 4 * n
        index["tensors"][name] = rec
        n_t += 1
        if n_t % 40 == 0:
            log("  %d / %d tensors" % (n_t, len(g.tensors)))
    for h in (fb, fd, ff, src):
        h.close()
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(out_dir, "bcodes_index.json"), "w") as f:
        json.dump(bindex, f)
    log("hashing the source file ...")
    manifest = {"kindi_version": __version__, "pack_format": PACK_FORMAT, "model": g.kv.get("general.name"),
                "source_file": os.path.basename(gguf_path), "source_sha256": sha256(gguf_path),
                "tensors": n_t, "bcodes_bytes": bc_off, "dpm_bytes": dpm_off, "f32_bytes": f32_off,
                "vocab": len(g.kv["tokenizer.ggml.tokens"]), "context_length": g.kv.get("qwen35.context_length"),
                "dimensions": {"layers": profile[0], "width": profile[1], "ffn": profile[2],
                               "heads": profile[3], "kv_heads": profile[4], "groups": profile[5],
                               "head_dim": g.kv["qwen35.attention.key_length"],
                               "rope_dim": g.kv["qwen35.rope.dimension_count"],
                               "state_size": g.kv["qwen35.ssm.state_size"],
                               "gdn_inner": g.kv["qwen35.ssm.inner_size"],
                               "rope_base": g.kv["qwen35.rope.freq_base"],
                               "full_attention_interval": g.kv["qwen35.full_attention_interval"]},
                # Hadamard block of the folded inputs (prism v2: 512, v1: e.g. 4096) and the output head tensor
                "hadamard_block": int(hd.get("block_size", 512)),
                "head": "output.weight" if any(t[0] == "output.weight" for t in g.tensors) else "token_embd.weight",
                # the chat template: the 27B's differs from the 2B/4B/9B one (chat.template_variant)
                "chat_template": g.kv.get("tokenizer.chat_template", "")}
    with open(os.path.join(out_dir, "kindi_pack.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log("pack written to %s (%.0f MB) in %.0fs" % (out_dir, (bc_off + dpm_off + f32_off) / 1e6, time.time() - t0))
    return manifest
