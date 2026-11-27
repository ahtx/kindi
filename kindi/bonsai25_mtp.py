"""Extract Prism ML's Bonsai 2.5 27B NextN block into Kindi's MTP head format.

Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .catalog import MODELS
from .gguf import GGUF


def convert(source, destination):
    g = GGUF(source)
    kv = g.kv
    if (kv.get("general.name") != "Ternary-Bonsai-2.5-27B" or kv.get("general.architecture") != "qwen35"
            or kv.get("qwen35.nextn_predict_layers") != 1):
        raise ValueError("expected a Ternary Bonsai 2.5 27B checkpoint with one NextN block")
    meta = dict(D=5120, NH=24, NHKV=4, HD=256, NFF=17408, W=64,
                rope_dim=kv["qwen35.rope.dimension_count"], rope_theta=kv["qwen35.rope.freq_base"],
                base_sha256=MODELS["bonsai25-27b"]["sha"])
    mats = {"fc": ("nextn.eh_proj", 5120, 10240), "q": ("attn_q", 12288, 5120),
            "k": ("attn_k", 1024, 5120), "v": ("attn_v", 1024, 5120),
            "o": ("attn_output", 5120, 6144), "gate": ("ffn_gate", 17408, 5120),
            "up": ("ffn_up", 17408, 5120), "down": ("ffn_down", 5120, 17408)}
    norms = {"e_norm": "nextn.enorm", "h_norm": "nextn.hnorm", "in_norm": "attn_norm",
             "post_norm": "post_attention_norm", "out_norm": "nextn.shared_head_norm",
             "q_norm": "attn_q_norm", "k_norm": "attn_k_norm"}
    tensors = {name: (shape, ty, g.data_start + off) for name, shape, ty, off in g.tensors}

    def array(name, shape, ty, dtype, disk_shape):
        actual, storage, offset = tensors["blk.64." + name + ".weight"]
        if actual != shape or storage != ty:
            raise ValueError(f"unexpected {name} shape or type: {actual}, {storage}")
        return np.memmap(source, mode="r", offset=offset, dtype=dtype, shape=disk_shape)

    out = {"meta": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
    for key, name in norms.items():
        size = 256 if key in ("q_norm", "k_norm") else 5120
        out[key] = array(name, [size], 0, "<f4", (size,))
    for key, (name, rows, cols) in mats.items():
        blocks = array(name, [cols, rows], 2, "u1", (rows, cols // 32, 18))
        codes = np.empty((rows, cols), dtype=np.int8)
        scales = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, 64):
            b = blocks[start:start + 64]
            d = b[:, :, :2].copy().view("<f2").astype(np.float32)
            nibbles = np.concatenate((b[:, :, 2:] & 15, b[:, :, 2:] >> 4), axis=2)
            w = ((nibbles.astype(np.int8) - 8) * d).reshape(-1, cols)
            if not np.isfinite(w).all():
                raise ValueError(f"nonfinite weights in {name}")
            s = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127
            scales[start:start + len(w)] = s
            codes[start:start + len(w)] = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
        out[key + ".w"], out[key + ".s"] = codes, scales
    dest = Path(destination)
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".npz", delete=False) as tmp:
        path = tmp.name
    try:
        np.savez(path, **out)
        os.replace(path, dest)
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination", nargs="?",
                        help="default: bonsai25-mtp.npz in the selected pack, where kindi uses it by default")
    args = parser.parse_args()
    if not args.destination:
        from .config import BONSAI25_HEAD, load
        args.destination = os.path.join(load()["pack"], BONSAI25_HEAD)
    convert(args.source, args.destination)
