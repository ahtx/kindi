"""LoRA adapters: y = W x + (alpha / r) B (A x) on the ternary projections.

The engine never sees the unrotated activation u of a folded projection: it computes xp = R u (the Hadamard fold,
R = blockdiag(H_HB / sqrt(HB)) diag(s)) and y = W' xp with the stored ternary W' (so W_eff = W' R). An adapter
trained on W_eff reads u: B A u = B (A R^T) xp. So the A rows are folded once, here (A' = A R^T: each row of A goes
through the same transform as the activations), and the adapter then reads the rotated activation the engine already
has. No extra Hadamard per token.

Kindi stores a converted adapter as FP16 in the pack: PACK/lora/<hash>/lora.npz (per adapted tensor the folded A'
[r][in] and B' = scale * B [out][r]) + meta.json. The hash covers the adapter files and the pack's source model; the
prompt cache keys its saved openings by it.

Accepted input: a PEFT folder (adapter_config.json + adapter_model.safetensors) with HF Qwen3.5 module names, or
GGUF tensor names (blk.N.attn_q ...); a single .safetensors / .npz with GGUF names (lora_alpha / basis from a
neighbouring adapter_config.json or the npz keys). basis "unrotated" (default: A reads u, W_eff convention) or
"rotated" (A already reads xp: no fold). Not adapted: the embeddings and the output head (the fast head stays valid),
ssm_alpha / ssm_beta (their GEMV is fused into the gate kernel)."""
import hashlib
import json
import os
import re
import struct
import time

import numpy as np

# HF Qwen3.5 module (after "layers.N.") -> GGUF tensor (after "blk.N.")
HF_TO_GGUF = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
    "linear_attn.in_proj_b": "ssm_beta", "linear_attn.in_proj_a": "ssm_alpha",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}
# GGUF tensors whose input is folded (norm outputs, the GDN output, the attention output, the FFN product)
FOLDED = {"attn_qkv", "attn_gate", "attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up", "ssm_out", "attn_output",
          "ffn_down"}
UNSUPPORTED = {"token_embd", "output"}
# fp32 GEMVs fused into the GDN gate kernel: their adapter is merged into the fp32 weights at load (W += s B A, exact,
# no cost per token); they read the unrotated norm output, so no fold
MERGED = {"ssm_alpha", "ssm_beta"}
LORA_FORMAT = 1


# ------------------------------------------------------------------------------------------------ readers
def read_safetensors(path):
    """-> {name: float32 array}. F32 / F16 / BF16 only."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        for name, rec in head.items():
            if name == "__metadata__":
                continue
            a, b = rec["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            dt = rec["dtype"]
            if dt == "F32":
                v = np.frombuffer(raw, dtype="<f4")
            elif dt == "F16":
                v = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif dt == "BF16":
                v = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                raise SystemExit("kindi lora: tensor %s has dtype %s (F32/F16/BF16 only)" % (name, dt))
            out[name] = v.reshape(rec["shape"]).astype(np.float32)
    return out


def _gguf_name(key):
    """An adapter key -> (layer, gguf tensor, 'A'|'B') or None."""
    m = re.search(r"blk\.(\d+)\.([a-z_]+?)(?:\.weight)?\.lora_([ab])\b", key, re.I)
    if m:
        return int(m.group(1)), m.group(2), m.group(3).upper()
    m = re.search(r"layers\.(\d+)\.(.+?)\.lora_([AB])(?:\.[a-z_]+)?\.weight$", key)
    if m:
        mod = m.group(2)
        if mod not in HF_TO_GGUF:
            raise SystemExit("kindi lora: no GGUF tensor for module %r (%s)" % (mod, key))
        return int(m.group(1)), HF_TO_GGUF[mod], m.group(3)
    return None


def read_adapter(path):
    """-> (pairs {(layer, tensor): [A, B]}, config dict, list of source files)."""
    cfg, files = {}, []
    if os.path.isdir(path):
        cj = os.path.join(path, "adapter_config.json")
        if os.path.exists(cj):
            cfg = json.load(open(cj))
            files.append(cj)
        st = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
        npz = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".npz")]
        src = st or npz
        if len(src) != 1:
            raise SystemExit("kindi lora: %s needs exactly one .safetensors (or .npz) file" % path)
        path = src[0]
    else:
        cj = os.path.join(os.path.dirname(os.path.abspath(path)), "adapter_config.json")
        if os.path.exists(cj):
            cfg = json.load(open(cj))
            files.append(cj)
    files.append(path)
    if path.endswith(".npz"):
        z = np.load(path)
        tens = {k: z[k].astype(np.float32) for k in z.files if "lora_" in k}
        for k in ("lora_alpha", "r", "basis"):
            if k in z.files:
                v = z[k]
                cfg[k] = str(v) if v.dtype.kind in "US" else v.item()
    else:
        tens = read_safetensors(path)
    pairs = {}
    cfg["_hf_names"] = any(_gguf_name(k) is not None and "blk." not in k for k in tens)
    for key, v in tens.items():
        g = _gguf_name(key)
        if g is None:
            raise SystemExit("kindi lora: unrecognised adapter tensor %r" % key)
        il, t, ab = g
        pairs.setdefault((il, t), [None, None])[0 if ab == "A" else 1] = v
    for k, (a, b) in pairs.items():
        if a is None or b is None:
            raise SystemExit("kindi lora: blk.%d.%s has only one of lora_A / lora_B" % k)
    return pairs, cfg, files


# ------------------------------------------------------------------------------------------------ math
def fold_rows(A, signs, hb):
    """A [r][n] (float64) -> A R^T: every row through the activation fold (signs, normalised hb-block FWHT)."""
    x = (A * signs[None, :]).reshape(A.shape[0], -1, hb).copy()
    h = 1
    while h < hb:
        x = x.reshape(A.shape[0], -1, hb // (2 * h), 2, h)
        a, b = x[:, :, :, 0, :], x[:, :, :, 1, :]
        x = np.stack([a + b, a - b], axis=3)
        h *= 2
    return x.reshape(A.shape) / np.sqrt(hb)


def vhead_perm(NK, NV):
    """perm[j] = the GGUF position of HF V head j. Hugging Face groups the V heads by K head (V head j reads K head
    j // (NV / NK)); the GGUF stores them so that V head j reads K head j % NK (Kindi's k_gdn_delta and k_smfold
    use that order). So HF head j = k * G + g (k = K head, g = its copy) sits at GGUF g * NK + k."""
    G = NV // NK
    j = np.arange(NV)
    return (j % G) * NK + j // G


def vhead_rows(NK, NV, HK):
    """HF V-row index -> GGUF V-row index, for the NV * HK rows of the GDN value / gate / output width"""
    perm = vhead_perm(NK, NV)
    return (perm[:, None] * HK + np.arange(HK)[None, :]).reshape(-1)


def hf_to_gguf_order(t, A, B, rows_v, NK, HK):
    """An adapter trained in Hugging Face's V-head order -> the GGUF order of the stored weights (NV != NK: 4B, 9B).
    attn_qkv: the V rows of B (after the 2 NK HK q / k rows); attn_gate: all rows of B. ssm_out keeps its columns:
    the GGUF stores them in HF order (k_smfold writes the GDN output in that order)."""
    if t == "attn_qkv":
        qk = 2 * NK * HK
        Bg = B.copy()
        Bg[qk + rows_v] = B[qk:]
        return A, Bg
    if t == "attn_gate":
        Bg = np.empty_like(B)
        Bg[rows_v] = B
        return A, Bg
    return A, B                               # ssm_out: its input (k_smfold's output) is already in HF order


def _scale(cfg, t, r):
    alpha = (cfg.get("alpha_pattern") or {}).get(t, cfg.get("lora_alpha", r))
    rr = (cfg.get("rank_pattern") or {}).get(t, r)
    if rr != r:
        raise SystemExit("kindi lora: rank of %s is %d in the file but %d in the config" % (t, r, rr))
    return float(alpha) / (np.sqrt(r) if cfg.get("use_rslora") else r)


def convert(adapter, pack, log=print):
    """Read an adapter, fold it for the pack, write PACK/lora/<hash>/. Returns the hash."""
    t0 = time.time()
    pairs, cfg, files = read_adapter(adapter)
    man = json.load(open(os.path.join(pack, "kindi_pack.json")))
    idx = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"]
    hb = int(man.get("hadamard_block", 512))
    basis = str(cfg.get("basis", "unrotated"))
    if basis not in ("unrotated", "rotated"):
        raise SystemExit("kindi lora: basis must be unrotated or rotated, not %r" % basis)
    if cfg.get("fan_in_fan_out"):
        raise SystemExit("kindi lora: fan_in_fan_out adapters are not supported")
    signs = {}
    h = hashlib.sha256()
    for f in files:
        h.update(open(f, "rb").read())
    h.update((man.get("source_sha256") or "").encode())
    h.update(("format%d" % LORA_FORMAT).encode())
    key = h.hexdigest()[:16]
    dims = man.get("dimensions", {})
    NK, HK = int(dims.get("groups", 16)), int(dims.get("state_size", 128))
    NV = int(dims.get("gdn_inner", NK * HK)) // HK
    rows_v = None
    if cfg.get("_hf_names") and NV != NK:
        rows_v = vhead_rows(NK, NV, HK)
    out = {}
    fidx = json.load(open(os.path.join(pack, "index.json")))["tensors"]
    for (il, t), (A, B) in sorted(pairs.items()):
        if t in MERGED:
            name = "blk.%d.%s.weight" % (il, t)
            if name not in fidx:
                raise SystemExit("kindi lora: the pack has no tensor %s" % name)
            R, C = fidx[name]["rows"], fidx[name]["ne0"]
            r = A.shape[0]
            if A.shape != (r, C) or B.shape != (R, r):
                raise SystemExit("kindi lora: %s: A %s / B %s do not fit the %d x %d weight" % (name, A.shape, B.shape,
                                                                                               R, C))
            if basis != "unrotated":
                raise SystemExit("kindi lora: %s reads the unrotated activation; basis must be unrotated" % name)
            if rows_v is not None:                # HF V-head order -> GGUF: one row per V head
                Bg = np.empty_like(B)
                Bg[vhead_perm(NK, NV)] = B
                B = Bg
            out[name + ".delta"] = (B.astype(np.float64) @ A.astype(np.float64) * _scale(cfg, t, r)).astype(np.float32)
            continue
        if t in UNSUPPORTED or t not in FOLDED:
            raise SystemExit("kindi lora: blk.%d.%s cannot carry an adapter in Kindi" % (il, t))
        name = "blk.%d.%s.weight" % (il, t)
        if name not in idx:
            raise SystemExit("kindi lora: the pack has no tensor %s" % name)
        R, C = idx[name]["rows"], idx[name]["ne0"]
        r = A.shape[0]
        if A.shape != (r, C) or B.shape != (R, r):
            raise SystemExit("kindi lora: %s: A %s / B %s do not fit the %d x %d weight" % (name, A.shape, B.shape, R, C))
        if C not in signs:                    # the sign vector of the input width (signs_2048, signs_6144, ...)
            signs[C] = np.fromfile(os.path.join(pack, "signs_%d.f32" % C), dtype=np.float32).astype(np.float64)
        if rows_v is not None:
            A, B = hf_to_gguf_order(t, A, B, rows_v, NK, HK)
        Ad = A.astype(np.float64)
        if basis == "unrotated":
            Ad = fold_rows(Ad, signs[C], hb)
        out[name + ".a"] = Ad.astype(np.float16)
        out[name + ".b"] = (B.astype(np.float64) * _scale(cfg, t, r)).astype(np.float16)
    d = os.path.join(pack, "lora", key)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "lora.npz"), **out)
    meta = {"format": LORA_FORMAT, "hash": key, "source": os.path.abspath(adapter), "basis": basis,
            "tensors": len(pairs), "merged": sorted(k[:-6] for k in out if k.endswith(".delta")),
            "ranks": sorted({int(v.shape[0]) for k, v in out.items() if k.endswith(".a")}),
            "bytes": int(sum(v.nbytes for v in out.values())), "created": time.time()}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    log("adapter %s: %d tensors, rank %s, %.1f MB FP16 -> %s (%.1fs)" % (key, len(pairs), meta["ranks"],
                                                                     meta["bytes"] / 1e6, d, time.time() - t0))
    return key


def resolve(spec, pack, log=print):
    """KINDI_LORA / --lora value -> (hash, {tensor name: (A' fp16, B' fp16)}) or (None, None).
    spec: a converted hash (in PACK/lora), a converted folder, or an adapter (converted now if not yet)."""
    if not spec:
        return None, None
    d = None
    if re.match(r"^[0-9a-f]{16}$", spec) and os.path.isdir(os.path.join(pack, "lora", spec)):
        d = os.path.join(pack, "lora", spec)
    elif os.path.exists(os.path.join(spec, "lora.npz")):
        d = spec
    elif os.path.exists(spec):
        d = os.path.join(pack, "lora", convert(spec, pack, log=log))
    else:
        raise SystemExit("kindi lora: %s is neither a converted adapter nor an adapter file" % spec)
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "lora.npz"))
    ts = {}
    for k in z.files:
        if k.endswith(".a"):
            n = k[:-2]
            ts[n] = (z[k], z[n + ".b"])
        elif k.endswith(".delta"):                # merged into an fp32 weight at load (MERGED)
            ts[k[:-6]] = ("merge", z[k])
    return meta["hash"], ts
