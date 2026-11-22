"""The MTP head of Qwen3.8-Flash-Next for Kindi: the BF16 tensors fetched from the official release
(tools/qwen4exp_mtp_fetch.py) into one small GGUF file next to the model (qwen4exp-mtp.gguf).

The release is under the Qwen Community License 1.0; the file keeps its notice in general.license.

Layout of the file (read by kindi.qwen4exp_ref.Weights as an extra file):
  - the head's decoder layer under the main model's names with layer number L (48): blk.48.hc_attn_*, attn_*,
    indexer.*, hc_ffn_*, ffn_gate_inp, ffn_*_shexp, ffn_gate_inp_shexp, ffn_{gate,up,down}_exps
  - its input: mtp.norm_emb, mtp.norm_hid, mtp.fc_emb, mtp.fc_hid; its output mixer: mtp.output_hc_{norm,down,up}
  - dense matrices stay BF16 (the card's dense kernels read BF16), norm vectors F32, the routed experts IQ4_NL
    (quantized here: 32 values, an fp16 scale and the 16 fixed levels; about 1.41 GB)."""
import json
import os
import struct

import numpy as np

from . import qwen4exp_quant as Q

LICENSE_NOTICE = ("Qwen3.8-Flash-Next MTP head, converted from Qwen/Qwen3.8-Flash-Next. Copyright (c) 2026 Qwen. "
                  "Licensed under the Qwen Community License 1.0 (LICENSE in the official release).")
LAYER = 48

# HF name (under mtp.layers.0.) -> the main model's GGUF name (under blk.48.); filled in from the main-model check
# in docs/dev/FLASHNEXT-MTP.md
LAYER_MAP = {
    "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
}
TOP_MAP = {
    "pre_fc_norm_embedding.weight": "mtp.norm_emb.weight",
    "pre_fc_norm_hidden.weight": "mtp.norm_hid.weight",
    "fc_embedding.weight": "mtp.fc_emb.weight",
    "fc_hidden.weight": "mtp.fc_hid.weight",
    "hyper_connection_mixer.hc_norm.weight": "mtp.output_hc_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight": "mtp.output_hc_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight": "mtp.output_hc_up.weight",
}


# from tools/qwen4exp_mtp_check.py map (the main model's tensors in the release against the GGUF, 11-15): every RMS
# norm weight is stored as w - 1 (applied as 1 + w); the shared-expert gate as is; the fused indexer projection is
# [q; k]; gate_up holds the gate rows, then the up rows; down is [K][F] per expert as in the GGUF
NORM_PLUS_ONE = ("mtp.norm_emb.weight", "mtp.norm_hid.weight", "mtp.output_hc_norm.weight",
                 "blk.%d.hc_attn_norm.weight" % LAYER, "blk.%d.hc_ffn_norm.weight" % LAYER,
                 "blk.%d.attn_q_norm.weight" % LAYER, "blk.%d.attn_k_norm.weight" % LAYER,
                 "blk.%d.indexer.q_norm.weight" % LAYER, "blk.%d.indexer.k_norm.weight" % LAYER)
INDEX_QK_FIRST = "q"
GATE_UP_ORDER = "halves"


# ---------------------------------------------------------------- IQ4_NL
_LV = Q.IQ4_LEVELS.astype(np.float32)
_MID = (_LV[1:] + _LV[:-1]) / 2


def iq4nl(x):
    """[n] float32 (n a multiple of 32) -> IQ4_NL blocks [n / 32, 18] uint8. Per block: the scale that maps the value
    of largest magnitude onto the end level of its sign (-127 or 113), and three nearby scales; the one with the
    least squared error is kept; each value takes the nearest level."""
    b = x.reshape(-1, 32).astype(np.float32)
    i = np.abs(b).argmax(1)
    vmax = b[np.arange(len(b)), i]
    best_err = np.full(len(b), np.inf, np.float32)
    best_q = np.zeros(b.shape, np.uint8)
    best_d = np.zeros(len(b), np.float32)
    for end in (-127.0, 113.0):
        for f in (1.0, 0.95, 0.9):
            d = vmax / end * f
            d16 = d.astype(np.float16).astype(np.float32)        # the stored scale
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(d16[:, None] != 0, b / d16[:, None], 0.0)
            q = np.searchsorted(_MID, y).astype(np.uint8)
            err = ((b - d16[:, None] * _LV[q]) ** 2).sum(1)
            better = err < best_err
            best_err[better], best_d[better], best_q[better] = err[better], d16[better], q[better]
    out = np.empty((len(b), 18), np.uint8)
    out[:, 0:2] = best_d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:18] = best_q[:, :16] | (best_q[:, 16:] << 4)
    return out


# ---------------------------------------------------------------- Q2_0 (the compact draft experts)
def q2_0(x):
    """[n] float32 (n a multiple of 64) -> Q2_0 blocks [n / 64, 18] uint8: an fp16 scale d and 64 two-bit codes q,
    value d * (q - 1), so the levels are -d, 0, d, 2d (a negative d mirrors them). Per block: the value of largest
    magnitude is put on the far level of either sign (2d or -d) at six shrink factors; each candidate takes the
    nearest codes, then twice a least-squares scale for those codes and the nearest codes again; the least squared
    error wins."""
    b = x.reshape(-1, 64).astype(np.float32)
    n = len(b)
    vmax = b[np.arange(n), np.abs(b).argmax(1)]
    best_err = np.full(n, np.inf, np.float32)
    best_q = np.zeros(b.shape, np.uint8)
    best_d = np.zeros(n, np.float32)

    def codes(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(d[:, None] != 0, b / d[:, None], 0.0)
        return np.clip(np.rint(y) + 1, 0, 3).astype(np.uint8)

    def keep(d, q):
        err = ((b - d[:, None] * (q.astype(np.float32) - 1)) ** 2).sum(1)
        better = err < best_err
        best_err[better], best_d[better], best_q[better] = err[better], d[better], q[better]

    for top in (2.0, -1.0):
        for f in (1.0, 0.85, 0.7, 0.6, 0.5, 0.4):
            d = (vmax / top * f).astype(np.float16).astype(np.float32)
            q = codes(d)
            keep(d, q)
            for _ in range(2):
                lv = q.astype(np.float32) - 1
                den = (lv * lv).sum(1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    d = np.where(den > 0, (b * lv).sum(1) / den, d).astype(np.float16).astype(np.float32)
                q = codes(d)
                keep(d, q)
    out = np.empty((n, 18), np.uint8)
    out[:, 0:2] = best_d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    q = best_q.reshape(n, 16, 4)
    out[:, 2:18] = q[:, :, 0] | (q[:, :, 1] << 2) | (q[:, :, 2] << 4) | (q[:, :, 3] << 6)
    return out


def compact(src, out, log=print, chunk=16):
    """the head file src with its experts as Q2_0 (the drafter's compact experts: about half the bytes; drafts
    only, so it changes acceptance, never the output) -> out. The other tensors and the metadata are copied."""
    from kindi.gguf import GGUF
    import mmap
    g = GGUF(src)
    fh = open(src, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    ts = []
    for name, ne, ty, off in g.tensors:
        nb = Q.nbytes(ty, int(np.prod(ne)))
        raw = mm[g.data_start + off:g.data_start + off + nb]
        if "_exps." in name:
            per = ne[0] * ne[1]
            parts = []
            for e0 in range(0, ne[2], chunk):
                e1 = min(ne[2], e0 + chunk)
                bpe = Q.nbytes(ty, per)
                x = Q.decode(ty, raw[e0 * bpe:e1 * bpe], (e1 - e0) * per)
                parts.append(q2_0(x).tobytes())
            data = b"".join(parts)
            log("%s: %s -> Q2_0, %.0f -> %.0f MB" % (name, Q.NAMES.get(ty, ty), nb / 1e6, len(data) / 1e6))
            ts.append((name, ne, Q.Q2_0, data))
        else:
            ts.append((name, ne, ty, raw))
    kv = {k: v for k, v in g.kv.items() if k != "general.alignment"}
    kv["kindi.mtp_experts"] = "Q2_0 from the head file's experts (tools/qwen4exp_mtp_compact.py)"
    write_gguf(out, kv, ts)
    mm.close()
    fh.close()


# ---------------------------------------------------------------- a small GGUF writer
def _s(b):
    b = b.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key, v):
    if isinstance(v, str):
        return _s(key) + struct.pack("<I", 8) + _s(v)
    if isinstance(v, bool):
        return _s(key) + struct.pack("<I?", 7, v)
    if isinstance(v, int):
        return _s(key) + struct.pack("<Iq", 11, v)
    if isinstance(v, float):
        return _s(key) + struct.pack("<If", 6, v)
    raise TypeError(key)


def write_gguf(path, kv, tensors, align=32):
    """tensors: [(name, ne (fastest first), type, bytes)] in order"""
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv) + 1)
    items = [_kv("general.alignment", int(align))] + [_kv(k, v) for k, v in kv.items()]
    infos, off = [], 0
    for name, ne, ty, data in tensors:
        infos.append(_s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
                     + struct.pack("<IQ", ty, off))
        off += (len(data) + align - 1) // align * align
    blob = head + b"".join(items) + b"".join(infos)
    pad = (-len(blob)) % align
    with open(path + ".tmp", "wb") as f:
        f.write(blob + b"\0" * pad)
        for name, ne, ty, data in tensors:
            f.write(data)
            f.write(b"\0" * ((-len(data)) % align))
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- conversion
def _bf16(path, shape):
    a = np.fromfile(path, np.uint16)
    assert a.size == int(np.prod(shape)), path
    return a.reshape(shape)


def _f32(raw16):
    return (raw16.astype(np.uint32) << 16).view(np.float32)


def convert(src, out, norm_plus_one, index_qk_first="q", gate_up_order="halves", log=print):
    """src: the directory written by tools/qwen4exp_mtp_fetch.py; out: the GGUF path.
    norm_plus_one: names (GGUF) of the norm weights stored as (w - 1) in the release (applied as 1 + w)"""
    info = json.load(open(os.path.join(src, "tensors.json"), encoding="utf-8"))
    raw = lambda k: _bf16(os.path.join(src, k + ".bin"), info[k]["shape"])   # noqa: E731
    tensors = []

    def add(name, a16):
        """a BF16 matrix/vector: norm vectors as F32 (+1 where the release stores w - 1), the rest as BF16"""
        if a16.ndim == 1 or a16.shape[0] == 1:          # vectors (and [1, n] gates) as F32 vectors
            v = _f32(a16.reshape(-1))
            if name in norm_plus_one:
                v = v + 1.0
            tensors.append((name, [v.size], Q.F32, v.astype(np.float32).tobytes()))
        else:
            tensors.append((name, list(a16.shape[::-1]), Q.BF16, np.ascontiguousarray(a16).tobytes()))

    for k, g in TOP_MAP.items():
        add(g, raw("mtp." + k))
    p = "mtp.layers.0."
    for k, g in LAYER_MAP.items():
        add("blk.%d.%s" % (LAYER, g), raw(p + k))
    qk = raw(p + "self_attn.indexer.index_qk_proj.weight")
    nk = info[p + "self_attn.indexer.k_layernorm.weight"]["shape"][0]
    if index_qk_first == "q":
        q, k = qk[:-nk], qk[-nk:]
    else:
        k, q = qk[:nk], qk[nk:]
    add("blk.%d.indexer.q_proj.weight" % LAYER, q)
    add("blk.%d.indexer.k_proj.weight" % LAYER, k)
    # routed experts -> IQ4_NL, expert by expert
    gu_shape, dn_shape = info[p + "mlp.experts.gate_up_proj"]["shape"], info[p + "mlp.experts.down_proj"]["shape"]
    E, two_f, K = gu_shape
    Fd = two_f // 2
    gu = np.memmap(os.path.join(src, p + "mlp.experts.gate_up_proj.bin"), np.uint16, "r", shape=tuple(gu_shape))
    dn = np.memmap(os.path.join(src, p + "mlp.experts.down_proj.bin"), np.uint16, "r", shape=tuple(dn_shape))
    gate, up, down = bytearray(), bytearray(), bytearray()
    for e in range(E):
        m = _f32(np.asarray(gu[e]))
        g_, u_ = (m[:Fd], m[Fd:]) if gate_up_order == "halves" else (m[0::2], m[1::2])
        gate += iq4nl(g_.reshape(-1)).tobytes()
        up += iq4nl(u_.reshape(-1)).tobytes()
        d_ = _f32(np.asarray(dn[e]))
        if d_.shape != (K, Fd):
            d_ = d_.T
        down += iq4nl(np.ascontiguousarray(d_).reshape(-1)).tobytes()
        if log and e % 64 == 63:
            log("  experts %d/%d" % (e + 1, E))
    tensors.append(("blk.%d.ffn_gate_exps.weight" % LAYER, [K, Fd, E], Q.IQ4_NL, bytes(gate)))
    tensors.append(("blk.%d.ffn_up_exps.weight" % LAYER, [K, Fd, E], Q.IQ4_NL, bytes(up)))
    tensors.append(("blk.%d.ffn_down_exps.weight" % LAYER, [Fd, K, E], Q.IQ4_NL, bytes(down)))
    kv = {"general.architecture": "qwen4exp-mtp", "general.name": "Qwen3.8-Flash-Next MTP head",
          "general.license": LICENSE_NOTICE, "general.source.repo": "Qwen/Qwen3.8-Flash-Next",
          "qwen4exp-mtp.layer": LAYER, "qwen4exp-mtp.expert_type": "IQ4_NL"}
    write_gguf(out, kv, tensors)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="the fetched MTP tensors (tools/qwen4exp_mtp_fetch.py) -> qwen4exp-mtp.gguf")
    ap.add_argument("src")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    convert(a.src, a.out, NORM_PLUS_ONE, INDEX_QK_FIRST, GATE_UP_ORDER)
    print("wrote %s (%.2f GB)" % (a.out, os.path.getsize(a.out) / 1e9))


if __name__ == "__main__":
    main()
