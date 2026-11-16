"""DeepSeek-V4.1-Flash DSpark drafter (mtp.0-2 of the original checkpoint): conversion into the engine's formats.

The drafter only proposes tokens; the main model checks every one, so its weights do not need to be exact:
- FP8 E4M3 matrices with E8M0 scales per 32x32 block -> Q8_0 rows (GGUF block layout: fp16 scale + 32 int8);
- BF16 / F32 vectors and the router -> F32;
- FP4 experts (int8 [rows][cols / 2], low nibble = even element, E8M0 scale per 32) -> GGUF MXFP4 blocks (17 bytes:
  E8M0 scale, byte j = element j | element j + 16 << 4). This repacking is exact (the same codes and scales).

usage: python -m kindi.dsv41_dspark convert DSPARK_DIR      (writes DSPARK_DIR/kindi-dspark.bin + .json)
       python -m kindi.dsv41_dspark check DSPARK_DIR        (FP4 repack vs a direct dequantization)"""
import json
import os
import struct
import sys

import numpy as np

SHARDS = ["model-00044-of-00048.safetensors", "model-00045-of-00048.safetensors", "model-00046-of-00048.safetensors"]
NSTAGE, NEXP = 3, 128
FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)


def e4m3_table():
    t = np.zeros(256, np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            v = 0.0                                   # NaN code: not used by weights
        elif e == 0:
            v = m / 8.0 * 2.0 ** -6
        else:
            v = (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = e4m3_table()


class Shards(object):
    def __init__(self, d):
        self.t = {}
        for sh in SHARDS:
            path = os.path.join(d, sh)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            mm = np.memmap(path, np.uint8, "r")
            for k, v in h.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.t[k] = (v["dtype"], v["shape"], mm, 8 + n + a, 8 + n + b)

    def raw(self, k):
        dt, shape, mm, a, b = self.t[k]
        return dt, shape, mm[a:b]

    def f32(self, k):
        dt, shape, r = self.raw(k)
        if dt == "F32":
            return np.frombuffer(r, np.float32).reshape(shape).copy()
        if dt == "BF16":
            return (np.frombuffer(r, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        raise ValueError("%s: %s" % (k, dt))


def fp8_to_q8(sh, name):
    """FP8 E4M3 [rows][cols] with E8M0 [rows / 32][cols / 32] -> GGUF Q8_0 bytes [rows][cols / 32][34]"""
    _, shape, w = sh.raw(name + ".weight")
    _, sshape, s = sh.raw(name + ".scale")
    rows, cols = shape
    v = E4M3[np.frombuffer(w, np.uint8).reshape(rows, cols)]
    sc = np.ldexp(1.0, np.frombuffer(s, np.uint8).astype(np.int32) - 127).astype(np.float32).reshape(sshape)
    v = v * np.repeat(np.repeat(sc, 32, 0), 32, 1)[:rows, :cols]
    b = v.reshape(rows, cols // 32, 32)
    amax = np.abs(b).max(-1)
    d = (amax / 127).astype(np.float16)
    inv = np.where(amax > 0, 127 / np.where(amax > 0, amax, 1), 0).astype(np.float32)
    q = np.clip(np.rint(b * inv[..., None]), -127, 127).astype(np.int8)
    out = np.empty((rows, cols // 32, 34), np.uint8)
    out[:, :, :2] = d.view(np.uint8).reshape(rows, cols // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out


def fp4_to_mxfp4(sh, name):
    """FP4 int8 [rows][cols / 2] + E8M0 [rows][cols / 32] -> GGUF MXFP4 [rows][cols / 32][17]"""
    _, shape, w = sh.raw(name + ".weight")
    _, _, s = sh.raw(name + ".scale")
    rows, half = shape
    cols = half * 2
    p = np.frombuffer(w, np.uint8).reshape(rows, cols // 32, 16)
    codes = np.empty((rows, cols // 32, 32), np.uint8)
    codes[..., 0::2] = p & 15
    codes[..., 1::2] = p >> 4
    out = np.empty((rows, cols // 32, 17), np.uint8)
    out[..., 0] = np.frombuffer(s, np.uint8).reshape(rows, cols // 32)
    out[..., 1:] = codes[..., :16] | (codes[..., 16:] << 4)
    return out


def convert(d):
    sh = Shards(d)
    out = os.path.join(d, "kindi-dspark.bin")
    index = {}
    with open(out + ".tmp", "wb") as f:
        def put(name, arr, kind, shape):
            off = f.tell()
            f.write(np.ascontiguousarray(arr).tobytes())
            pad = (-f.tell()) % 64
            f.write(b"\0" * pad)
            index[name] = {"off": off, "kind": kind, "shape": list(shape)}
        put("main_proj", fp8_to_q8(sh, "mtp.0.main_proj"), "q8", (5120, 15360))
        put("main_norm", sh.f32("mtp.0.main_norm.weight"), "f32", (5120,))
        for st in range(NSTAGE):
            p = "mtp.%d." % st
            for nm, gg, shp in (("q_a", "attn.wq_a", (1280, 5120)), ("q_b", "attn.wq_b", (32768, 1280)),
                                ("kv", "attn.wkv", (512, 5120)), ("o_a", "attn.wo_a", (8192, 4096)),
                                ("o_b", "attn.wo_b", (5120, 8192)), ("sh_gate", "ffn.shared_experts.w1", (2304, 5120)),
                                ("sh_up", "ffn.shared_experts.w3", (2304, 5120)),
                                ("sh_down", "ffn.shared_experts.w2", (5120, 2304))):
                put("%d.%s" % (st, nm), fp8_to_q8(sh, p + gg), "q8", shp)
            for nm, gg in (("sinks", "attn.attn_sink"), ("kv_norm", "attn.kv_norm.weight"), ("q_norm", "attn.q_norm.weight"),
                           ("attn_norm", "attn_norm.weight"), ("ffn_norm", "ffn_norm.weight"), ("gate", "ffn.gate.weight"),
                           ("bias", "ffn.gate.bias"), ("hc_attn_fn", "hc_attn_fn"), ("hc_attn_base", "hc_attn_base"),
                           ("hc_attn_scale", "hc_attn_scale"), ("hc_ffn_fn", "hc_ffn_fn"), ("hc_ffn_base", "hc_ffn_base"),
                           ("hc_ffn_scale", "hc_ffn_scale")):
                a = sh.f32(p + gg)
                put("%d.%s" % (st, nm), a, "f32", a.shape)
            for wn, nm in (("w1", "gate_exps"), ("w3", "up_exps"), ("w2", "down_exps")):
                arrs = [fp4_to_mxfp4(sh, p + "ffn.experts.%d.%s" % (e, wn)) for e in range(NEXP)]
                put("%d.%s" % (st, nm), np.stack(arrs), "mxfp4", (NEXP,) + arrs[0].shape[:2])
                print("dspark: stage %d %s" % (st, nm), file=sys.stderr, flush=True)
        for nm, gg in (("norm", "mtp.2.norm.weight"), ("conf", "mtp.2.confidence_head.proj.weight"),
                       ("markov_embed", "mtp.2.markov_head.embed.weight"), ("markov_head", "mtp.2.markov_head.head.weight")):
            a = sh.f32(gg)
            put(nm, a, "f32", a.shape)
    os.replace(out + ".tmp", out)
    with open(out + ".json", "w") as f:
        json.dump(index, f, indent=0)
    print("dspark: wrote %s (%.2f GB)" % (out, os.path.getsize(out) / 1e9), file=sys.stderr)


def check(d):
    """the repacked MXFP4 blocks dequantize to the same values as the checkpoint's FP4 + scales"""
    sh = Shards(d)
    name = "mtp.1.ffn.experts.7.w2"
    g = fp4_to_mxfp4(sh, name)
    _, shape, w = sh.raw(name + ".weight")
    _, _, s = sh.raw(name + ".scale")
    rows, half = shape
    p = np.frombuffer(w, np.uint8).reshape(rows, half)
    ref = np.empty((rows, half * 2), np.float32)
    ref[:, 0::2] = FP4[p & 15]
    ref[:, 1::2] = FP4[p >> 4]
    ref *= np.repeat(np.ldexp(1.0, np.frombuffer(s, np.uint8).astype(np.int32) - 127).reshape(rows, -1), 32, 1)
    lo, hi = g[..., 1:] & 15, g[..., 1:] >> 4
    vals = np.concatenate([FP4[lo], FP4[hi]], -1) * np.ldexp(1.0, g[..., :1].astype(np.int32) - 127)
    same = np.array_equal(vals.reshape(rows, -1), ref)
    print("dspark check: %s repacked MXFP4 == checkpoint FP4: %s" % (name, "exact" if same else "DIFFERENT"))
    q = fp8_to_q8(sh, "mtp.0.attn.wq_a")
    _, shp, w8 = sh.raw("mtp.0.attn.wq_a.weight")
    _, ssh, s8 = sh.raw("mtp.0.attn.wq_a.scale")
    v = E4M3[np.frombuffer(w8, np.uint8).reshape(shp)] * np.repeat(np.repeat(
        np.ldexp(1.0, np.frombuffer(s8, np.uint8).astype(np.int32) - 127).reshape(ssh), 32, 0), 32, 1)
    dq = q[..., 2:].view(np.int8).astype(np.float32) * q[..., :2].copy().view(np.float16).astype(np.float32)
    err = np.sqrt(((dq.reshape(v.shape) - v) ** 2).mean() / (v ** 2).mean())
    print("dspark check: FP8 -> Q8_0 (wq_a) relative RMS error %.2e (drafter only)" % err)


if __name__ == "__main__":
    {"convert": convert, "check": check}[sys.argv[1]](sys.argv[2])
