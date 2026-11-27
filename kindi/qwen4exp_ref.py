"""CPU reference forward pass for qwen4exp GGUF files (Qwen3.8-Flash-Next: the preview of the Qwen4 architecture).

This is the correctness reference for the Kindi Flash-Next engine, not the fast path. It runs the whole model in fp32
with PyTorch on the CPU and reads the weights through kindi.qwen4exp_quant. The fast kernels (CPU expert kernels,
multi-card layer split) are checked against it, and it is checked against the black-box reference (the token gate in
tools/qwen4exp_gate.py).

Architecture (from the model card and config): 48 layers in 12 groups of (3 Gated DeltaNet + 1 Qwen Sparse Attention),
each followed by a 512-expert MoE (top 10 + 1 gated shared expert). The residual is 4 streams of 2560 joined by gated
residual (hyper-connection) mixers. Layer 1 adds hashed bigram/trigram embeddings (PLE). QSA selects 512 blocks of 4
tokens (2048 tokens) per query, scored by a small indexer (qsa_mask); with 2048 visible tokens or fewer it selects
every visible token, so attention is dense causal up to that length.

Weight conventions of these GGUF files: RMSNorm weights include the +1 of the (1 + w) form, except the gated norm of
Gated DeltaNet; ssm_a holds -exp(A_log)."""
import math
import mmap
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

from kindi import qwen4exp_quant as Q
from kindi.gguf import GGUF

H = 2560
_ROWS_PREFETCH = os.environ.get("KINDI_ROWS_PREFETCH", "1") != "0"
QSA_TOKENS = 2048          # the indexer's token budget: up to this many visible tokens, every token is selected
QSA_RATIO = 4              # tokens per key block


def qsa_mask(qi, kraw, pos, knorm, eps, rope, budget=QSA_TOKENS, ratio=QSA_RATIO):
    """Qwen Sparse Attention block selection (the transformers qwen4_exp indexer), on any torch device.
    qi [T, heads, d]: the indexer queries of the new tokens (normed, RoPE at their positions); kraw [P, d]: the raw
    indexer keys of positions 0 .. P - 1; pos [T]: the new tokens' positions (the last is P - 1). Returns the bool
    mask [T, P] of the keys each query attends: per query, the visible complete blocks of `ratio` tokens are scored by
    sum over heads of relu(q . block key) / sqrt(d), where a block key is the mean of its raw keys, normed and RoPE'd
    at the block's first position; the best budget / ratio blocks are kept, plus the tokens after the last complete
    block (the tail). With at most budget visible tokens this is the causal mask."""
    T, P = qi.shape[0], kraw.shape[0]
    dev = kraw.device
    causal = torch.arange(P, device=dev)[None, :] <= pos[:, None]
    if P <= budget:
        return causal
    nb = P // ratio
    top, nvis = _qsa_top(qi, kraw, pos, knorm, eps, rope, budget, ratio)
    sel = torch.zeros(T, nb, dtype=torch.bool, device=dev)
    sel.scatter_(1, top.indices, top.values > float("-inf"))
    tok = torch.zeros(T, P, dtype=torch.bool, device=dev)
    tok[:, :nb * ratio] = sel.repeat_interleave(ratio, dim=1)
    tail = torch.arange(P, device=dev)[None, :] >= (nvis * ratio)[:, None]
    return (tok | tail) & causal


def _qsa_top(qi, kraw, pos, knorm, eps, rope, budget, ratio):
    nb, d = kraw.shape[0] // ratio, kraw.shape[1]
    dev = kraw.device
    bk = kraw[:nb * ratio].view(nb, ratio, d).float().mean(1)
    bk = rope(rms(bk, knorm, eps)[:, None, :], torch.arange(nb, device=dev) * ratio)[:, 0, :]
    sc = torch.relu(torch.einsum("thd,bd->thb", qi.float(), bk)).sum(1) / math.sqrt(d)     # [T, nb]
    nvis = (pos + 1) // ratio                                                                  # complete blocks
    sc = sc.masked_fill(torch.arange(nb, device=dev)[None, :] >= nvis[:, None], float("-inf"))
    return sc.topk(min(budget // ratio, nb), dim=-1), nvis


def qsa_positions(qi, kraw, pos, knorm, eps, rope, budget=QSA_TOKENS, ratio=QSA_RATIO):
    """Selected key positions and validity for contexts longer than the QSA budget."""
    top, nvis = _qsa_top(qi, kraw, pos, knorm, eps, rope, budget, ratio)
    offsets = torch.arange(ratio, device=pos.device)
    positions = (top.indices[:, :, None] * ratio + offsets).flatten(1)
    valid = (top.values > float("-inf")).repeat_interleave(ratio, dim=1)
    tail = nvis[:, None] * ratio + offsets[:-1]
    valid_tail = tail <= pos[:, None]
    positions = torch.cat((positions, tail.clamp_max(kraw.shape[0] - 1)), dim=1)
    return positions.to(torch.int32), torch.cat((valid, valid_tail), dim=1)


def shards(path):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(3))
    return ["%s-%05d-of-%05d.gguf" % (m.group(1), i, n) for i in range(1, n + 1)]


class Weights(object):
    """the tensors of all shards, decoded on demand (dense tensors are kept, experts and embedding rows are not)"""

    def __init__(self, path, extra=()):
        self.files, self.t, self.kv = [], {}, {}
        main = shards(path)
        for p in main + list(extra):
            g = GGUF(p)
            fh = open(p, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.files.append((fh, mm))
            if p in main:                      # the extra files (the MTP head) keep their own metadata
                self.kv.update(g.kv)
            else:
                self.extra_kv = getattr(self, "extra_kv", {})
                self.extra_kv.update(g.kv)
            for name, ne, ty, off in g.tensors:
                self.t[name] = (mm, g.data_start + off, ne, ty)
        self.cache = {}

    def has(self, name):
        return name in self.t

    def _raw(self, name, start_val, n):
        mm, base, ne, ty = self.t[name]
        b0 = base + Q.nbytes(ty, start_val) if start_val else base
        return Q.decode(ty, mm[b0:b0 + Q.nbytes(ty, n)], n)

    def get(self, name):
        """the whole tensor as fp32: a matrix (rows = ne[1], cols = ne[0]) or a vector"""
        w = self.cache.get(name)
        if w is None:
            mm, base, ne, ty = self.t[name]
            n = int(np.prod(ne))
            a = self._raw(name, 0, n)
            w = torch.from_numpy(a.reshape(ne[1], ne[0]) if len(ne) == 2 else a)
            self.cache[name] = w
        return w

    def rows(self, name, idx):
        """rows idx of a 2-D tensor (embeddings), decoded row by row. With several rows, every row's pages are first
        asked for at once (madvise WILLNEED): on a disk the reads of the rows not in the page cache then run in
        parallel instead of one seek after another (B29, 3060 host: about 60 ms of every draft window went to the
        n-gram table's rows, read one after another; KINDI_ROWS_PREFETCH=0 turns this off)"""
        mm, base, ne, ty = self.t[name]
        if len(idx) > 1 and _ROWS_PREFETCH and hasattr(mmap, "MADV_WILLNEED"):
            nb = Q.nbytes(ty, ne[0])
            pg = mmap.PAGESIZE
            for r in idx:
                b0 = base + Q.nbytes(ty, int(r) * ne[0])
                lo = b0 // pg * pg
                try:
                    mm.madvise(mmap.MADV_WILLNEED, lo, b0 + nb - lo)
                except (OSError, ValueError):
                    break
        out = np.empty((len(idx), ne[0]), np.float32)
        for i, r in enumerate(idx):
            out[i] = self._raw(name, int(r) * ne[0], ne[0])
        return torch.from_numpy(out)

    def expert(self, name, e):
        """matrix e of a 3-D expert tensor (ne = [cols, rows, experts])"""
        mm, base, ne, ty = self.t[name]
        n = ne[0] * ne[1]
        return torch.from_numpy(self._raw(name, e * n, n).reshape(ne[1], ne[0]))


def rms(x, w, eps=1e-6, group=None):
    if group is not None:
        s = x.shape
        x = x.reshape(*s[:-1], -1, group)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x = x.reshape(s)
    else:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * w


class Model(object):
    def __init__(self, path, mtp=None):
        """mtp: the MTP head file (kindi/qwen4exp_mtp_import.py), or None"""
        self.w = W = Weights(path, [mtp] if mtp else ())
        kv = W.kv
        a = kv["general.architecture"]
        if a != "qwen4exp":
            raise ValueError("not a qwen4exp file: %s" % a)
        g = lambda k: kv["%s.%s" % (a, k)]     # noqa: E731
        self.L = g("block_count")
        self.interval = g("full_attention_interval")
        self.nh, self.nkv, self.hd = g("attention.head_count"), g("attention.head_count_kv"), g("attention.key_length")
        self.rope_dim, self.rope_base = g("rope.dimension_count"), g("rope.freq_base")
        self.eps = g("attention.layer_norm_rms_epsilon")
        self.n_exp, self.n_used = g("expert_count"), g("expert_used_count")
        self.hc = g("hyper_connection.count")
        self.ssm_groups, self.ssm_state = g("ssm.group_count"), g("ssm.state_size")
        self.ssm_heads = g("ssm.time_step_rank")
        self.ple_layers = list(g("ple.layers"))
        self.ple_n = g("ple.ngram_size")
        self.ple_heads = g("ple.heads_per_ngram")
        self.ple_eos = g("ple.eos_token_id")
        self.ple_mult = np.array(g("ple.layer_multipliers"), dtype=np.int64)
        self.ple_off = np.array(g("ple.head_offsets"), dtype=np.int64)
        self.ple_size = np.array(g("ple.head_vocab_sizes"), dtype=np.int64)
        self.ple_kernel = g("ple.conv_kernel")
        # layout switches (fixed by the token gate on 11-14): how a V head of Gated DeltaNet finds its QK head, and how
        # attn_q holds Q and its gate. The GGUF file stores the 48 V heads (and the matching rows/columns of attn_qkv,
        # ssm_conv1d, attn_gate, ssm_alpha, ssm_beta, ssm_a, ssm_dt and ssm_out) in "tile" order: V head h uses
        # QK head h % nk. transformers' order is "group" (h // (nv / nk), repeat_interleave); it fails the gate.
        # attn_q is "head": [q | gate] per head, as in transformers ("block" = all Q, then all gates; fails).
        self.gdn_heads = "tile"
        self.q_layout = "head"
        half = self.rope_dim // 2
        self.inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float64) / self.rope_dim))
        assert self.inv_freq.numel() == half

    decode_threads = 1
    _executor = None
    # the routed experts on the C kernels (kindi.qx_cpu.Experts: Q8 activations, the file's own blocks) instead of
    # fp32 from the decoded weights; and a routing trace hook trace(layer, idx [T, n_used])
    cpu_experts = None
    trace = None
    tap = None          # tap(layer, h): the residual streams before each layer (and tap(L, h) after the last)

    def _pool(self):
        """a thread pool for decoding the experts of a layer (NumPy releases the GIL in the block decoders)"""
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(self.decode_threads)
        return self._executor

    # ------------------------------------------------------------------ pieces
    def mixer(self, h, pre, inject=True):
        """gated residual: (T, hc*H) -> branch input (T, H) [, the stream input, the 4 write gates]"""
        W = self.w
        T = h.shape[0]
        hn = rms(h, W.get(pre + "_norm.weight"), self.eps, group=H)
        m = F.silu(hn @ W.get(pre + "_down.weight").T / self.hc)
        m = torch.sigmoid(m @ W.get(pre + "_up.weight").T).view(T, self.hc, H)
        x = (m * hn.view(T, self.hc, H)).mean(1)
        if not inject:
            return x
        wr = 2 * torch.sigmoid(hn @ W.get(pre + "_inject.weight").T / self.hc)
        return x, h, wr

    @staticmethod
    def write(hin, y, wr):
        return hin + (y[:, None, :] * wr[:, :, None]).reshape(hin.shape)

    def ple_rows(self, ids, prev=None):
        """rows of the n-gram (PLE) table that the token ids read: (T, heads_per_ngram x (ngram_size - 1)).
        prev: the ngram_size - 1 tokens before ids (None: the start of a sequence, filled with the eos id).

        For a token at t and an order k (2 .. ngram_size), the key is the xor of tok[t - j] * multiplier[j] over
        j < k, where a token before the start of t's segment (a segment begins after each eos token) counts as the
        eos id; head i of order k reads row offset[i] + key mod size[i] (int64 arithmetic, wrapping)."""
        e, n = self.ple_eos, self.ple_n
        lead = [e] * (n - 1) if prev is None else [int(v) for v in prev]
        toks = np.asarray(lead + [int(v) for v in ids], dtype=np.int64)
        t = np.arange(len(toks))
        # where each position's segment starts: one past the last eos strictly before it
        last_eos = np.maximum.accumulate(np.where(toks == e, t, -1))
        seg_start = np.empty_like(t)
        seg_start[0] = 0
        seg_start[1:] = last_eos[:-1] + 1
        lagged = [np.where(t - j >= seg_start, toks[np.maximum(t - j, 0)], e) for j in range(n)]
        cols = []
        with np.errstate(over="ignore"):
            key = lagged[0] * self.ple_mult[0]
            for k in range(2, n + 1):
                key = key ^ (lagged[k - 1] * self.ple_mult[k - 1])
                first = (k - 2) * self.ple_heads
                size = self.ple_size[first:first + self.ple_heads]
                base = self.ple_off[first:first + self.ple_heads]
                cols.append(base[None, :] + np.mod(key[:, None], size[None, :]))
        return np.concatenate(cols, axis=1)[n - 1:]

    def ple(self, l, h, ids, st=None):
        W = self.w
        T = h.shape[0]
        rows = self.ple_rows(ids, st["ids"] if st else None)                                            # (T, 16)
        emb = W.rows("per_layer_token_embd.weight", rows.reshape(-1)).view(T, -1)          # (T, 16 * 160)
        p = "blk.%d.ple_" % l
        key = rms(emb @ W.get(p + "key.weight").T, W.get(p + "norm_key.weight"), self.eps, group=H).view(T, self.hc, H)
        val = emb @ W.get(p + "value.weight").T                              # (T, H)
        qry = rms(h, W.get(p + "norm_query.weight"), self.eps, group=H).view(T, self.hc, H)
        gate = (key * qry).sum(-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = (torch.sigmoid(gate) * val[:, None, :]).reshape(T, -1)
        gvn = rms(gv, W.get(p + "norm_conv.weight"), self.eps, group=H)
        k, dil = self.ple_kernel, self.ple_n
        cw = W.get(p + "conv1d.weight")                                       # (channels, k)
        pad = (k - 1) * dil
        prev = st["ple"].get(l) if st else None
        x = torch.cat([torch.zeros(pad, gvn.shape[1]) if prev is None else prev, gvn])
        if st is not None:
            st["ple"][l] = x[-pad:].clone()
        y = F.silu(F.conv1d(x.T[None], cw[:, None, :], groups=cw.shape[0], dilation=dil))[0].T
        return gv + y

    def gdn(self, l, x, st=None):
        """Gated DeltaNet over the new tokens, from the state st (None: a new sequence, zero state)"""
        W = self.w
        p = "blk.%d." % l
        T = x.shape[0]
        nk, nv, dk = self.ssm_groups, self.ssm_heads, self.ssm_state
        dv = W.get(p + "ssm_norm.weight").numel()
        qkv = x @ W.get(p + "attn_qkv.weight").T                              # (T, 2 nk dk + nv dv)
        z = x @ W.get(p + "attn_gate.weight").T
        b = x @ W.get(p + "ssm_beta.weight").T
        a = x @ W.get(p + "ssm_alpha.weight").T
        cw = W.get(p + "ssm_conv1d.weight")                                   # (channels, 4)
        k = cw.shape[1]
        prev = st["conv"].get(l) if st else None
        qkv = torch.cat([torch.zeros(k - 1, qkv.shape[1]) if prev is None else prev, qkv])
        if st is not None:
            st["conv"][l] = qkv[-(k - 1):].clone()
        qkv = F.silu(F.conv1d(qkv.T[None], cw[:, None, :], groups=cw.shape[0]))[0].T
        q, kk, v = torch.split(qkv, [nk * dk, nk * dk, nv * dv], dim=-1)
        q = q.reshape(T, nk, dk)
        kk = kk.reshape(T, nk, dk)
        v = v.reshape(T, nv, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
        q = q / math.sqrt(dk)
        rep = nv // nk
        if self.gdn_heads == "tile":
            q, kk = q.repeat(1, rep, 1), kk.repeat(1, rep, 1)
        else:
            q, kk = q.repeat_interleave(rep, dim=1), kk.repeat_interleave(rep, dim=1)
        beta = torch.sigmoid(b)
        g = W.get(p + "ssm_a") * F.softplus(a + W.get(p + "ssm_dt.bias"))   # ssm_a = -exp(A_log)
        S = st["S"].get(l) if st else None
        S = torch.zeros(nv, dk, dv) if S is None else S
        out = torch.empty(T, nv, dv)
        for t in range(T):
            S = S * g[t].exp()[:, None, None]
            kt, vt = kk[t], v[t]
            mem = (S * kt[:, :, None]).sum(1)
            delta = (vt - mem) * beta[t][:, None]
            S = S + kt[:, :, None] * delta[:, None, :]
            out[t] = (S * q[t][:, :, None]).sum(1)
        if st is not None:
            st["S"][l] = S
        o = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps) * W.get(p + "ssm_norm.weight")
        o = o * torch.sigmoid(z.view(T, nv, dv))
        return o.reshape(T, -1) @ W.get(p + "ssm_out.weight").T

    def rope(self, x, pos):
        """NeoX-style rotation of the first rope_dim dims of each head; x (T, heads, d)"""
        half = self.rope_dim // 2
        ang = (pos[:, None].double() * self.inv_freq[None, :]).float()       # (T, half)
        c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x1, x2, rest = x[..., :half], x[..., half:self.rope_dim], x[..., self.rope_dim:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, rest], dim=-1)

    def attn(self, l, x, st=None):
        W = self.w
        p = "blk.%d." % l
        T = x.shape[0]
        p0 = st["pos"] if st else 0
        nh, nkv, hd = self.nh, self.nkv, self.hd
        qg = x @ W.get(p + "attn_q.weight").T
        if self.q_layout == "block":
            q, gate = qg[:, :nh * hd].reshape(T, nh, hd), qg[:, nh * hd:]
        else:
            qg = qg.view(T, nh, 2 * hd)
            q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
        q = rms(q, W.get(p + "attn_q_norm.weight"), self.eps)
        k = rms((x @ W.get(p + "attn_k.weight").T).view(T, nkv, hd), W.get(p + "attn_k_norm.weight"), self.eps)
        v = (x @ W.get(p + "attn_v.weight").T).view(T, nkv, hd)
        pos = torch.arange(p0, p0 + T)
        q, k = self.rope(q, pos), self.rope(k, pos)
        if st is not None:
            if l in st["k"]:
                k, v = torch.cat([st["k"][l], k]), torch.cat([st["v"][l], v])
            st["k"][l], st["v"][l] = k, v
        # the indexer: raw keys of every position (kept in the state), queries of the new tokens
        ni = W.t[p + "indexer.q_proj.weight"][2][1] // W.t[p + "indexer.k_proj.weight"][2][1]
        kr = x @ W.get(p + "indexer.k_proj.weight").T
        if st is not None:
            if l in st["ik"]:
                kr = torch.cat([st["ik"][l], kr])
            st["ik"][l] = kr
        qi = rms((x @ W.get(p + "indexer.q_proj.weight").T).view(T, ni, -1), W.get(p + "indexer.q_norm.weight"),
                 self.eps)
        qi = self.rope(qi, pos)
        mask = qsa_mask(qi, kr, pos, W.get(p + "indexer.k_norm.weight"), self.eps, self.rope)
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        s = torch.einsum("thd,shd->hts", q, k) / math.sqrt(hd)
        s = s.masked_fill(~mask[None], float("-inf"))
        o = torch.einsum("hts,shd->thd", torch.softmax(s, -1), v).reshape(T, -1)
        return (o * torch.sigmoid(gate)) @ W.get(p + "attn_output.weight").T

    def moe(self, l, x):
        W = self.w
        p = "blk.%d." % l
        probs = torch.softmax(x @ W.get(p + "ffn_gate_inp.weight").T, dim=-1)
        top, idx = probs.topk(self.n_used, dim=-1)
        top = top / top.sum(-1, keepdim=True)
        if self.trace is not None:
            self.trace(l, idx)
        if self.cpu_experts is not None and l < self.L:
            out = torch.from_numpy(self.cpu_experts.forward(l, x.numpy(), idx.int().numpy(), top.numpy()))
        else:
            out = self._moe_fp32(p, x, idx, top)
        sh = (F.silu(x @ W.get(p + "ffn_gate_shexp.weight").T) * (x @ W.get(p + "ffn_up_shexp.weight").T)) \
            @ W.get(p + "ffn_down_shexp.weight").T
        sg = torch.sigmoid(x @ W.get(p + "ffn_gate_inp_shexp.weight"))
        return out + sg[:, None] * sh

    def _moe_fp32(self, p, x, idx, top):
        """the routed experts in fp32 from the decoded weights"""
        W = self.w
        out = torch.zeros_like(x)
        experts = torch.unique(idx).tolist()

        def load(e):
            return [W.expert(p + n, e) for n in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")]
        mats = self._pool().map(load, experts) if self.decode_threads > 1 else map(load, experts)
        for e, (wg, wu, wd) in zip(experts, mats):
            tok, slot = torch.where(idx == e)
            xe = x[tok]
            ye = (F.silu(xe @ wg.T) * (xe @ wu.T)) @ wd.T
            out.index_add_(0, tok, ye * top[tok, slot][:, None])
        return out

    # ------------------------------------------------------------------ forward
    @staticmethod
    def new_state():
        """the cached state of a sequence (positions, n-gram context, conv and delta-rule states, attention K/V)"""
        return {"pos": 0, "ids": None, "ple": {}, "conv": {}, "S": {}, "k": {}, "v": {}, "ik": {}}

    def mtp_logits(self, hc, ids, state=None):
        """the MTP head (one decoder layer, number L): hc [T, hc*H], the main model's streams after its last layer at
        positions p (or the head's own streams from its previous step), ids [T] the tokens at p + 1 -> (draft logits for
        p + 2 [T, vocab], the head's streams [T, hc*H]). state: the head's own new_state() (its attention cache)."""
        W, L = self.w, self.L
        T = hc.shape[0]
        with torch.no_grad():
            e = rms(W.rows("token_embd.weight", list(ids)), W.get("mtp.norm_emb.weight"), self.eps)
            e = e @ W.get("mtp.fc_emb.weight").T
            hn = rms(hc, W.get("mtp.norm_hid.weight"), self.eps).view(T, self.hc, H)
            h = (hn @ W.get("mtp.fc_hid.weight").T + e[:, None, :]).reshape(T, self.hc * H)
            x, hin, wr = self.mixer(h, "blk.%d.hc_attn" % L)
            h = self.write(hin, self.attn(L, x, state), wr)
            x, hin, wr = self.mixer(h, "blk.%d.hc_ffn" % L)
            h = self.write(hin, self.moe(L, x), wr)
            out = self.mixer(h, "mtp.output_hc", inject=False) @ W.get("output.weight").T
        if state is not None:
            state["pos"] += T
        return out, h

    def logits(self, ids, last_n=None, log=None, state=None, streams=None):
        """fp32 logits for each position of the token ids; last_n keeps only the last rows. state (new_state()):
        the ids continue that sequence, and the state is updated; None: a new sequence"""
        W = self.w
        ids = list(ids)
        st = state
        with torch.no_grad():
            h = W.rows("token_embd.weight", ids).repeat(1, self.hc)
            for l in range(self.L):
                if self.tap is not None:
                    self.tap(l, h)
                if l in self.ple_layers:
                    h = h + self.ple(l, h, ids, st)
                x, hin, wr = self.mixer(h, "blk.%d.hc_attn" % l)
                y = self.attn(l, x, st) if (l + 1) % self.interval == 0 else self.gdn(l, x, st)
                h = self.write(hin, y, wr)
                x, hin, wr = self.mixer(h, "blk.%d.hc_ffn" % l)
                h = self.write(hin, self.moe(l, x), wr)
                if log:
                    log("layer %d done" % l)
            if self.tap is not None:
                self.tap(self.L, h)
            if streams is not None:
                streams.append(h)                  # the streams after the last layer (the MTP head's input)
            x = self.mixer(h, "output_hc", inject=False)
            if last_n:
                x = x[-last_n:]
            out = x @ W.get("output.weight").T
        if st is not None:
            st["pos"] += len(ids)
            ctx = self.ple_n - 1
            prev = list(st["ids"]) if st["ids"] is not None else [self.ple_eos] * ctx
            st["ids"] = np.asarray((prev + ids)[-ctx:], np.int64)
        return out

    def generate(self, ids, n, on_token=None):
        """greedy: prompt ids, then n tokens one at a time from the cached state; returns (tokens, prompt seconds,
        seconds of each generated token)"""
        import time
        st = self.new_state()
        t0 = time.time()
        nxt = int(self.logits(ids, last_n=1, state=st)[-1].argmax())
        t_prompt = time.time() - t0
        out, times = [nxt], []
        for _ in range(n - 1):
            t1 = time.time()
            nxt = int(self.logits([nxt], last_n=1, state=st)[-1].argmax())
            times.append(time.time() - t1)
            out.append(nxt)
            if on_token:
                on_token(nxt, times[-1])
        return out, t_prompt, times

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="qwen4exp CPU reference: greedy next tokens for token ids")
    ap.add_argument("model")
    ap.add_argument("ids", help="comma-separated token ids")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    m = Model(a.model)
    m.decode_threads = max(1, a.threads)
    ids = [int(v) for v in a.ids.split(",")]
    lg = m.logits(ids, last_n=1, log=lambda s: print(s, file=sys.stderr, flush=True))
    top = torch.topk(lg[-1], 5)
    print("top5", list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])))


if __name__ == "__main__":
    main()
