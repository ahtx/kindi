"""Tensor-core attention for selected keys during Flash-Next prefill."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_attention(
    QUERY, KEY, VALUE, SELECTED, VALID, PART, LSE,
    NKV: tl.constexpr, HEADS_PER_KV: tl.constexpr, WIDTH: tl.constexpr, HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr, TILE_KEYS: tl.constexpr, SCALE: tl.constexpr,
):
    token, group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = tl.arange(0, 16)
    dims = tl.arange(0, HEAD_DIM)
    keys = tl.arange(0, TILE_KEYS)
    num_heads: tl.constexpr = NKV * HEADS_PER_KV
    tiles_per_split: tl.constexpr = (WIDTH + SPLITS * TILE_KEYS - 1) // (SPLITS * TILE_KEYS)
    head_base = (token * num_heads + group * HEADS_PER_KV + heads) * HEAD_DIM
    query = tl.load(QUERY + head_base[:, None] + dims[None, :],
                    mask=heads[:, None] < HEADS_PER_KV, other=0)

    maximum = tl.full((16,), float("-inf"), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    result = tl.full((16, HEAD_DIM), 0, tl.float32)
    for tile in range(tiles_per_split):
        at = (split * tiles_per_split + tile) * TILE_KEYS + keys
        visible = tl.load(VALID + token * WIDTH + at, at < WIDTH, other=False)
        pos = tl.load(SELECTED + token * WIDTH + at, visible, other=0)
        base = (pos * NKV + group) * HEAD_DIM
        key = tl.load(KEY + base[None, :] + dims[:, None], visible[None, :], other=0)
        value = tl.load(VALUE + base[:, None] + dims[None, :], visible[:, None], other=0)
        score = tl.dot(query, key) * SCALE
        score = tl.where(visible[None, :], score, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        old_factor = tl.exp2(maximum - safe_max)
        prob = tl.exp2(score - safe_max[:, None])
        denominator = denominator * old_factor + tl.sum(prob, axis=1)
        result *= old_factor[:, None]
        result = tl.dot(prob.to(tl.float16), value, result)
        maximum = next_max

    denominator_safe = tl.where(denominator > 0, denominator, 1.0)
    partial = head_base * SPLITS + split * HEAD_DIM
    tl.store(PART + partial[:, None] + dims[None, :], result / denominator_safe[:, None],
             mask=heads[:, None] < HEADS_PER_KV)
    tl.store(LSE + partial // HEAD_DIM, maximum + tl.log2(denominator_safe), mask=heads < HEADS_PER_KV)


@triton.jit
def _merge_attention(PART, LSE, OUT, HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    split = tl.arange(0, SPLITS)
    dim = tl.arange(0, HEAD_DIM)
    lse = tl.load(LSE + row * SPLITS + split)
    score = tl.exp2(lse - tl.max(lse, axis=0))
    mass = tl.sum(score, axis=0)
    partial = tl.load(PART + ((row * SPLITS + split[:, None]) * HEAD_DIM + dim[None, :]))
    result = tl.sum(partial * score[:, None], axis=0) / mass
    tl.store(OUT + row * HEAD_DIM + dim, result)


def selected_attention(query, key, value, positions, valid):
    """Use the same QSA positions as the reference, with grouped-query tensor-core products."""
    n_tokens, n_heads, head_dim = query.shape
    n_kv = key.shape[1]
    heads_per_kv = n_heads // n_kv
    assert heads_per_kv <= 16 and head_dim in (128, 256)
    assert positions.shape == valid.shape and positions.shape[0] == n_tokens
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    query, key, value = (tensor.contiguous().half() for tensor in (query, key, value))
    positions, valid = positions.contiguous(), valid.contiguous()
    splits = 4
    tile_keys = 32
    width = positions.shape[1]
    for start in range(0, n_tokens, 64):
        count = min(64, n_tokens - start)
        parts = torch.empty((count, n_heads, splits, head_dim), device=query.device, dtype=torch.float32)
        lse = torch.empty((count, n_heads, splits), device=query.device, dtype=torch.float32)
        _selected_attention[(count, n_kv, splits)](
            query[start:], key, value, positions[start:], valid[start:], parts, lse,
            n_kv, heads_per_kv, width, head_dim, splits, tile_keys, math.log2(math.e) / math.sqrt(head_dim),
            num_warps=8, num_stages=1,
        )
        _merge_attention[(count * n_heads,)](parts, lse, output[start:], head_dim, splits, num_warps=4)
    return output
