"""CPU reference checks for the Turing tensor-core kernels (kindi/tc_turing.cu).

The Turing module is a numerical clone of the Ampere one (kindi/tc.cu): same weight layout, same int8 activations,
same scaling and the same accumulation order.  Comparing the two backends therefore proves only that they agree, not
that either is right -- a bug in the shared arithmetic would pass unnoticed.  These tests compare the Turing kernels
against an independent NumPy model written from the *documented* semantics, not from the kernels:

  * the 5-trit source layout S (kindi/tp.py g5_blob + the lane transform in tp.py:362): per 128-column group and
    16-row tile, 28 byte positions of 16 bytes; byte position p < 24 holds five trits of columns 5p..5p+4
    (weight 3**(c % 5), least significant trit first), p = 24, 25 hold four trits of columns 120..127, and
    p = 26, 27 are the low/high byte of the row's fp16 weight scale;
  * a trit is a ternary weight: the tensor-core A operand is the trit itself (0, 1, 2) and the int8 dot product is
    corrected afterwards by subtracting the sum of the activations, so A . q - sum(q) = sum((trit - 1) * q) = w . q
    with w in {-1, 0, 1} (kindi/convert.py: "weight = (t-1) * d");
  * int8 activations are per 128-column group: amax = max |x| over the group, scale = amax / 127 (fp32),
    q = round-half-to-even (x * 127 / amax) (__float2int_rn), XA = sum of the 128 int8 values;
  * a group contributes (int_sum - XA) * (scale * fp16_weight_scale) to a row of the product, accumulated in fp32
    over the groups in order (with fnms/fma contraction or with a rounded multiply -- both emissions are accepted).

Covered: k_tc_pack (through the trit decode and the fp16 scales, and end to end), k_tc_q8 (XQ/ XS/ XA decoded
independently), k_tc_mv2/4/8 with and without column splits, k_tc_mv3_2/4/8 (one, two, three tensors),
k_tc_mm2/4/8/16, k_tc_sum, odd token counts, short 16-row tiles, several units, strided output rows, all-zero and
all-two weights, zero and huge inputs, and exact round-half-even ties.

The kernels are built through the production path -- kindi.spec.SpecKernels(backend=tc_turing), i.e. compute_75 PTX
loaded by tc_turing.load_module -- so the loader is exercised too.  A CUDA device and cupy are needed; the sm_75 PTX
is JIT-compiled, so the suite also runs on newer cards (it has been run on sm_89):

    python -m pytest tests/test_tc_turing_reference.py -q
    python tests/test_tc_turing_reference.py          # same, with a summary table
"""
import sys

import numpy as np
import pytest

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()                     # a CUDA runtime, not only the Python package
except Exception:                                        # cupy missing: the tests skip, the model stays importable
    cp = None

pytestmark = pytest.mark.skipif(cp is None, reason="the reference kernels need cupy and a CUDA device")

EPS = 2.0 ** -24                                         # fp32 unit roundoff


# ==================================================================================== the CPU model (NumPy only)
def build_src(trits, dsc):
    """the 5-trit source layout S of a tensor.

    trits: (NG, R, 128) uint8 in 0..2, dsc: (NG, R) float16 (one weight scale per group and row); returns S as flat
    bytes, element (G, r, p) at ((G * NT16 + r // 16) * 28 + p) * 16 + (r % 16).
    """
    trits = np.asarray(trits, dtype=np.uint8)
    NG, R, C = trits.shape
    assert C == 128 and R % 16 == 0
    S = np.zeros((NG, R // 16, 28, 16), dtype=np.uint8)
    r = np.arange(R)
    tile, rr = r >> 4, r & 15
    for c in range(128):                                  # 5 trits per byte for columns 0..119, 4 for 120..127
        p, e = (c // 5, c % 5) if c < 120 else (24 + (c - 120) // 4, (c - 120) % 4)
        S[:, tile, p, rr] += trits[:, :, c] * np.uint8(3 ** e)
    sc = np.asarray(dsc, dtype=np.float16).view(np.uint16)
    S[:, tile, 26, rr] = (sc & 0xFF).astype(np.uint8)
    S[:, tile, 27, rr] = (sc >> 8).astype(np.uint8)
    return S.reshape(-1)


def ref_trits(S, NG, R):
    """decode S back to trits: an inverse of build_src, to prove the reference layout round-trips"""
    S = np.asarray(S, dtype=np.uint8).reshape(NG, R // 16, 28, 16)
    r = np.arange(R)
    tile, rr = r >> 4, r & 15
    out = np.zeros((NG, R, 128), dtype=np.uint8)
    for c in range(128):
        p, e = (c // 5, c % 5) if c < 120 else (24 + (c - 120) // 4, (c - 120) % 4)
        out[:, :, c] = (S[:, tile, p, rr] // np.uint8(3 ** e)) % np.uint8(3)
    return out


def ref_scales(S, NG, R):
    """the fp16 weight scales read back out of S as (NG, R) float16"""
    S = np.asarray(S, dtype=np.uint8).reshape(NG, R // 16, 28, 16)
    r = np.arange(R)
    tile, rr = r >> 4, r & 15
    lo = S[:, tile, 26, rr].astype(np.uint16)
    hi = S[:, tile, 27, rr].astype(np.uint16)
    return (lo | (hi << 8)).astype(np.uint16).view(np.float16)


def ref_q8(X):
    """X (T, NG * 128) fp32 -> (q (T, NG, 128) int32, xs (T, NG) fp32, xa (T, NG) int32), as k_tc_q8 computes them"""
    X = np.asarray(X, dtype=np.float32)
    T, ldx = X.shape
    NG = ldx // 128
    V = X.reshape(T, NG, 128)
    amax = np.abs(V).max(axis=2).astype(np.float32)                       # fmaxf tree: the exact maximum
    with np.errstate(divide="ignore"):
        idv = np.where(amax != 0, np.float32(127.0) / amax, np.float32(0.0)).astype(np.float32)
    q = np.rint(V * idv[:, :, None]).astype(np.int32)                     # fp32 product, ties to even
    xs = (amax / np.float32(127.0)).astype(np.float32)
    xa = q.sum(axis=2, dtype=np.int32)
    return q, xs, xa


def ref_decode_xq(XQ, NG, ntok):
    """XQ (ntok * NG * 32 uint32) -> q (ntok, NG, 128) int32, decoding the staging layout of k_tc_q8:
    XQ[((n NG + g) 4 + t) 8 + 2 s + h] holds 4 int8 of columns 32 s + 16 h + 4 t + byte"""
    w = np.asarray(XQ, dtype=np.uint32).reshape(ntok, NG, 4, 8)
    out = np.zeros((ntok, NG, 128), dtype=np.int32)
    for t in range(4):
        for s in range(4):
            for h in range(2):
                word = w[:, :, t, 2 * s + h]
                for b in range(4):
                    v = ((word >> np.uint32(8 * b)) & np.uint32(0xFF)).astype(np.int32)
                    out[:, :, 32 * s + 16 * h + 4 * t + b] = (v ^ 0x80) - 0x80
    return out


def ref_product(trits, dsc, X, sp=1):
    """the product of a tensor against X the way the tc kernels compute it.

    trits (NG, R, 128) 0..2, dsc (NG, R) float16, X (T, NG * 128) fp32, sp = column splits of k_tc_mv (each split
    covers a contiguous run of units, and k_tc_sum adds the parts in split order).

    returns (y64, y32, yfma, tabs, NG): the exact value, the fp32 value of a rounded multiply plus sequential fp32
    sums, the fp32 value with the product fused into the add, the per-element sum of |term| for a rounding bound, and
    the number of terms (groups).
    """
    trits = np.asarray(trits, dtype=np.int32)
    NG, R, C = trits.shape
    q, xs, xa = ref_q8(X)
    T = q.shape[0]
    isum = np.einsum("grc,tgc->tgr", trits, q)                            # exact integer dot products (A = trit)
    corr = (isum - xa[:, :, None]).astype(np.float64)                     # exact, |corr| <= 16256
    d = np.asarray(dsc, dtype=np.float16).astype(np.float32)              # (NG, R)
    xsd = (xs[:, :, None] * d[None, :, :]).astype(np.float32)             # xs0 * d0
    terms = corr * xsd.astype(np.float64)                                 # exact f32 x f32 products
    y64 = terms.sum(axis=1)
    NU = NG // 8
    sp = max(1, min(int(sp), NU))
    bounds = [(p * NU // sp, (p + 1) * NU // sp) for p in range(sp)]      # the units of each split
    parts32, partsfma = [], []
    for u0, u1 in bounds:
        a = np.zeros((T, R), dtype=np.float32)
        f = np.zeros((T, R), dtype=np.float32)
        for g in range(u0 * 8, u1 * 8):
            a = (a + terms[:, g, :].astype(np.float32)).astype(np.float32)
            f = (terms[:, g, :] + f.astype(np.float64)).astype(np.float32)
        parts32.append(a)
        partsfma.append(f)
    y32, yfma = parts32[0], partsfma[0]
    for p in range(1, sp):                                               # k_tc_sum
        y32 = (y32 + parts32[p]).astype(np.float32)
        yfma = (yfma + partsfma[p]).astype(np.float32)
    return y64, y32, yfma, np.abs(terms).sum(axis=1), NG


def max_ulp(a, b):
    """the largest distance in ulp between two fp32 arrays (entries where either side is not finite are skipped)"""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size == 0:
        return 0
    ua = a.view(np.uint32).astype(np.int64)
    ub = b.view(np.uint32).astype(np.int64)
    ka = np.where(ua >> 31, np.int64(0xFFFFFFFF) - ua, ua + np.int64(0x80000000))   # monotone over floats
    kb = np.where(ub >> 31, np.int64(0xFFFFFFFF) - ub, ub + np.int64(0x80000000))
    return int(np.abs(ka - kb).max())


# ==================================================================================== the device side
SK = None
RESULTS = []


def kernels():
    global SK
    if SK is None:
        from kindi import spec, tc_turing
        SK = spec.SpecKernels(backend=tc_turing)
    return SK


def i32(x):
    return np.int32(x)


def pack(S, R, NG):
    """k_tc_pack: the 5-trit source layout -> the tc weight layout; returns the packed bytes as a device array

    Both layouts are 28 bytes per (group, row): the kernel writes NT16 * NU * 3584 = 28 R NG bytes, one byte per
    thread (the grid is 4x that, as in spec.repack)."""
    sk = kernels()
    n = R * NG * 7                                        # 3584-byte blocks / 4 threads each
    src = cp.asarray(np.asarray(S, dtype=np.uint8))
    dst = cp.empty(R * NG * 28, dtype=np.uint8)
    sk.k_tc_pack(((n * 4 + 255) // 256,), (256,), (src, dst, i32(R), i32(NG)))
    return dst


def q8(X, NG):
    """k_tc_q8: fp32 rows -> XQ (uint32), XS (fp32), XA (int32)"""
    sk = kernels()
    T, ldx = X.shape
    Xd = cp.asarray(np.asarray(X, dtype=np.float32))
    XQ = cp.zeros(T * NG * 32, dtype=cp.uint32)
    XS = cp.zeros(T * NG, dtype=cp.float32)
    XA = cp.zeros(T * NG, dtype=cp.int32)
    sk.k_tc_q8((NG, T), (128,), (Xd, i32(ldx), XQ, XS, XA, i32(NG), i32(T)))
    return XQ, XS, XA


def mv(nw, W, R, NG, XQ, XS, XA, ntok, sp):
    """k_tc_mvNW with sp column splits -> y (ntok, R); the parts are summed with k_tc_sum when sp > 1"""
    sk = kernels()
    NU = NG // 8
    sp = max(1, min(sp, NU))
    P = cp.zeros(ntok * sp * R, dtype=cp.float32)
    sk.mv[nw](((R // 16 + nw - 1) // nw, sp), (32 * nw,), (W, P, i32(R), i32(NU), XQ, XS, XA, i32(ntok)))
    if sp == 1:
        return cp.asnumpy(P).reshape(ntok, R)
    y = cp.zeros(ntok * R, dtype=cp.float32)
    sk.k_tc_sum(((R * ntok + 255) // 256,), (256,), (P, y, i32(R), i32(ntok), i32(sp)))
    return cp.asnumpy(y).reshape(ntok, R)


def mv3(nw, Ws, Rs, NG, XQ, XS, XA, ntok):
    """k_tc_mv3_NW: one, two or three products of the same input in one launch -> [y0, y1, y2] (ntok, R)"""
    sk = kernels()
    NU = NG // 8
    n = len(Ws)
    assert n == len(Rs) and 1 <= n <= 3
    Wt = list(Ws) + [cp.zeros(1, dtype=np.uint32)] * (3 - n)              # R = 0: the slot is inactive
    Rt = list(Rs) + [0] * (3 - n)
    ys = [cp.zeros(max(1, ntok * R), dtype=cp.float32) for R in Rt]
    gx = ((sum(Rt) // 16) + nw - 1) // nw
    sk.mv3[nw]((gx,), (32 * nw,), (Wt[0], ys[0], i32(Rt[0]), Wt[1], ys[1], i32(Rt[1]),
                                   Wt[2], ys[2], i32(Rt[2]), i32(NU), XQ, XS, XA, i32(ntok)))
    return [cp.asnumpy(y).reshape(ntok, R) for y, R in zip(ys[:n], Rt[:n])]


def mm(kind, W, R, NG, XQ, XS, XA, T, ldy=None):
    """k_tc_mm2 (shared-memory staging, 64-token blocks) or k_tc_mm4/8/16 -> Y (T, ldy)"""
    sk = kernels()
    NU = NG // 8
    ldy = R if ldy is None else ldy
    Y = cp.zeros(T * ldy, dtype=cp.float32)
    if kind == 2:
        sk.k_tc_mm2(((R // 16 + 7) // 8, (T + 63) // 64), (256,),
                    (W, i32(R), i32(NU), XQ, XS, XA, Y, i32(ldy), i32(T)))
    else:
        getattr(sk, "k_tc_mm%d" % kind)(((R // 16 + 3) // 4, (T + 8 * kind - 1) // (8 * kind)), (128,),
                                        (W, i32(R), i32(NU), XQ, XS, XA, Y, i32(ldy), i32(T)))
    return cp.asnumpy(Y).reshape(T, ldy)


def check(tag, gpu, model, results=None):
    """compare one product against the model: an exact match with one of the two fp32 emissions, or a bound"""
    y64, y32, yfma, tabs, ng = model
    g = np.asarray(gpu, dtype=np.float32)
    assert g.shape == y64.shape, "%s: shape %s, model %s" % (tag, g.shape, y64.shape)
    u32, ufma = max_ulp(g, y32), max_ulp(g, yfma)
    u64 = max_ulp(g, y64.astype(np.float32))
    bound = (4.0 + 2.0 * ng) * EPS * (tabs + 1e-30)                     # sequential fp32 sum of ng terms
    err = np.abs(g.astype(np.float64) - y64)
    finite = bool(np.all(np.isfinite(g) == np.isfinite(y64)))
    ok = finite and ((u32 == 0 or ufma == 0) or bool(np.all(err <= bound)))
    (RESULTS if results is None else results).append((tag, ok, u32, ufma, u64, float(err.max())))
    return ok


# ==================================================================================== inputs
def _trits(NG, R, kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "zero":
        return np.zeros((NG, R, 128), dtype=np.uint8)
    if kind == "two":
        return np.full((NG, R, 128), 2, dtype=np.uint8)
    if kind == "tails":                                   # only columns 120..127 (the 4-trit bytes) carry twos
        t = np.zeros((NG, R, 128), dtype=np.uint8)
        t[:, :, 120:] = 2
        return t
    if kind == "binary":                                  # only -1 and +1 (no zero weights)
        return (rng.integers(0, 2, (NG, R, 128), dtype=np.uint8) * 2).astype(np.uint8)
    if kind == "random":
        return rng.integers(0, 3, (NG, R, 128), dtype=np.uint8)
    raise ValueError(kind)


def _scales(NG, R, kind, seed=0):
    rng = np.random.default_rng(seed + 1)
    if kind == "zero":
        return np.zeros((NG, R), dtype=np.float16)
    if kind == "subnormal":
        return (rng.integers(1, 64, (NG, R)) * np.float16(2.0 ** -24)).astype(np.float16)
    if kind == "large":
        return (rng.uniform(0.5, 1.0, (NG, R)) * 65504.0).astype(np.float16)
    return rng.uniform(0.002, 0.02, (NG, R)).astype(np.float16)


def _inputs(ntok, NG, kind, seed=0):
    rng = np.random.default_rng(seed + 2)
    shape = (ntok, NG * 128)
    if kind == "zero":
        return np.zeros(shape, dtype=np.float32)
    if kind == "ties":                                    # amax = 254 makes x * 127/254 = x/2 exact: ties at .5
        x = rng.integers(-254, 255, shape).astype(np.float32)
        x[:, 0] = 254.0
        return x
    if kind == "huge":
        return (rng.standard_normal(shape) * 1e18).astype(np.float32)
    if kind == "normal":
        return rng.standard_normal(shape).astype(np.float32)
    raise ValueError(kind)


def _model(trits, dsc, X, sp=1):
    return ref_product(trits, dsc, X, sp)


def _report(tag, results):
    bad = [r for r in results if not r[1]]
    for tag_, _, u32, ufma, u64, err in bad:
        print("FAIL %-30s ulp(mul)=%d ulp(fma)=%d ulp(64)=%d err=%.3g" % (tag_, u32, ufma, u64, err))
    print("%s: %d/%d cases match the model" % (tag, len(results) - len(bad), len(results)))
    assert not bad


# ==================================================================================== tests
def test_reference_layout():
    """the reference's own S builder/decoder round-trip (guards the model against typos)"""
    t = _trits(8, 32, "random")
    d = _scales(8, 32, "random")
    S = build_src(t, d)
    assert np.array_equal(ref_trits(S, 8, 32), t)
    assert np.array_equal(ref_scales(S, 8, 32), d)


def test_activations():
    """k_tc_q8: XQ, XS and XA equal the model, for round-half-even ties, a zero group and a zero input"""
    NG, ntok = 8, 7
    for kind in ("ties", "normal", "huge", "zero"):
        X = _inputs(ntok, NG, kind)
        q, xs, xa = ref_q8(X)
        XQ, XS, XA = q8(X, NG)
        assert np.array_equal(ref_decode_xq(cp.asnumpy(XQ), NG, ntok), q), "XQ (%s)" % kind
        assert np.array_equal(cp.asnumpy(XS).reshape(ntok, NG), xs), "XS (%s)" % kind
        assert np.array_equal(cp.asnumpy(XA).reshape(ntok, NG), xa), "XA (%s)" % kind
    # the ties case really contains half-way values (otherwise it tests nothing)
    X = _inputs(4, NG, "ties")
    q, xs, xa = ref_q8(X)
    frac = np.abs(X.reshape(4, NG, 128) * (np.float32(127.0) / np.abs(X.reshape(4, NG, 128)).max(axis=2,
                                                                                              keepdims=True)))
    assert np.count_nonzero(np.abs(frac - np.floor(frac) - 0.5) < 1e-6) > 100


def test_pack_layout():
    """k_tc_pack: every data byte decodes to a base-3 number below 243 and every scale byte pair is the fp16 scale

    The packed byte of five trits is v = sum(trit * 3**j) rescaled from 0..242 to 0..255 (rounding up), so it
    decodes back with v = byte * 243 // 256.  Checking that on all bytes proves the column/row order of the packing
    behind the product tests: a misplaced trit changes some byte's value."""
    NG, R = 8, 32
    for kind, sk in (("random", "random"), ("zero", "zero"), ("two", "subnormal"), ("tails", "large")):
        t = _trits(NG, R, kind, seed=3)
        d = _scales(NG, R, sk)
        S = build_src(t, d)
        W = cp.asnumpy(pack(S, R, NG))
        assert W.size == (R // 16) * (NG // 8) * 3584
        data = W.reshape(R // 16, NG // 8, 7, 32, 16)     # [tile][unit][i 0..6][lane][byte]
        for i in range(6):                                # i 0..5 = words 0..23, i 6 = words 24..27
            for b16 in range(16):
                if (i == 6 and b16 >= 8):                 # word 26/27 (the scales) start at b16 = 8
                    continue
                v = (data[:, :, i, :, b16].astype(np.uint32) * 243) // 256
                assert v.max() <= 242, "packed byte of %s does not decode to a base-3 number" % kind
        # scales: bytes (i 6, lane, b16) with b16 = 8 + 4 (wk - 26) + (bq) are the fp16 scales, G = 2 t + (wk - 26)
        got = np.zeros((NG, R), dtype=np.uint16)
        for m in range(R // 16):
            for u in range(NG // 8):
                for lane in range(32):
                    g8, t = lane >> 2, lane & 3
                    for wk in (26, 27):
                        for bq in range(4):
                            G = u * 8 + 2 * t + (wk - 26)
                            r = 16 * m + g8 + 8 * (bq >> 1)
                            byte = np.uint16(data[m, u, 6, lane, 8 + 4 * (wk - 26) + bq])
                            got[G, r] |= byte << np.uint16(8 * (bq & 1))
        assert np.array_equal(got.view(np.float16), d), "packed scales of %s" % kind
    print("k_tc_pack: every data byte decodes to a base-3 number below 243 and every scale is the fp16 scale")


def test_products():
    """k_tc_mv2/4/8 against the model: zero/two/tails/binary/random weights, zero/tie/huge/normal inputs,
    three tokens, 48 rows, three units, and one to three column splits"""
    results = []
    NG, R, ntok = 24, 48, 3
    for wk in ("zero", "two", "tails", "binary", "random"):
        for ik in ("zero", "ties", "huge", "normal"):
            t = _trits(NG, R, wk, seed=7)
            d = _scales(NG, R, "zero" if wk == "zero" else ("large" if ik == "huge" else "random"))
            X = _inputs(ntok, NG, ik, seed=11)
            W = pack(build_src(t, d), R, NG)
            XQ, XS, XA = q8(X, NG)
            for nw in (2, 4, 8):
                for sp in (1, 2, 3):
                    check("mv%d sp%d %s/%s" % (nw, sp, wk, ik),
                          mv(nw, W, R, NG, XQ, XS, XA, ntok, sp), _model(t, d, X, sp), results)
    _report("vector products", results)


def test_token_counts():
    """every token count 1..9 for mv (odd tails, the 2t/2t+1 pairing) and 1..70 for the chunked products"""
    results = []
    NG, R = 16, 48
    t = _trits(NG, R, "random", seed=5)
    d = _scales(NG, R, "random")
    W = pack(build_src(t, d), R, NG)
    for ntok in range(1, 9):                              # the vector kernels cover 1 .. 8 tokens (2 t, 2 t + 1)
        X = _inputs(ntok, NG, "normal", seed=ntok)
        XQ, XS, XA = q8(X, NG)
        for nw in (2, 8):
            for sp in (1, 2):
                check("mv%d sp%d ntok%d" % (nw, sp, ntok), mv(nw, W, R, NG, XQ, XS, XA, ntok, sp),
                      _model(t, d, X, sp), results)
        check("mv3 ntok%d" % ntok, mv3(4, [W], [R], NG, XQ, XS, XA, ntok)[0], _model(t, d, X), results)
    for T in (1, 7, 8, 9, 63, 64, 65, 70):
        X = _inputs(T, NG, "normal", seed=T)
        XQ, XS, XA = q8(X, NG)
        model = _model(t, d, X)
        for kind in (2, 4, 8, 16):
            check("mm%d T%d" % (kind, T), mm(kind, W, R, NG, XQ, XS, XA, T), model, results)
        check("mm4 T%d strided" % T, mm(4, W, R, NG, XQ, XS, XA, T, ldy=R + 5)[:, :R], model, results)
    _report("token counts", results)


def test_shapes_and_merges():
    """short tiles (16/48/96 rows here: inactive warps and blocks), four units, and k_tc_mv3 with 1, 2, 3 tensors"""
    results = []
    NG = 32
    for R in (16, 48, 96):
        t = _trits(NG, R, "random", seed=R)
        d = _scales(NG, R, "random")
        W = pack(build_src(t, d), R, NG)
        X = _inputs(5, NG, "normal", seed=R)
        XQ, XS, XA = q8(X, NG)
        for nw in (2, 8):
            for sp in (1, 4):
                check("R%d mv%d sp%d" % (R, nw, sp), mv(nw, W, R, NG, XQ, XS, XA, 5, sp),
                      _model(t, d, X, sp), results)
        check("R%d mm2" % R, mm(2, W, R, NG, XQ, XS, XA, 5), _model(t, d, X), results)
    Rs, ts, ds, Ws = (32, 16, 48), [], [], []
    for j, R in enumerate(Rs):
        t = _trits(NG, R, "random", seed=100 + j)
        d = _scales(NG, R, "random", seed=100 + j)
        ts.append(t)
        ds.append(d)
        Ws.append(pack(build_src(t, d), R, NG))
    X = _inputs(4, NG, "normal", seed=99)
    XQ, XS, XA = q8(X, NG)
    for nw in (2, 8):
        for k in (1, 2, 3):
            got = mv3(nw, Ws[:k], Rs[:k], NG, XQ, XS, XA, 4)
            for j in range(k):
                check("mv3_%d %d tensors #%d" % (nw, k, j), got[j], _model(ts[j], ds[j], X), results)
    _report("shapes and merged products", results)


def test_sum():
    """k_tc_sum: the parts of a split product add up in order, in fp32"""
    sk = kernels()
    rng = np.random.default_rng(4)
    R, ntok, sp = 48, 5, 3
    P = (rng.standard_normal(ntok * sp * R) * 1e3).astype(np.float32)
    y = cp.zeros(ntok * R, dtype=cp.float32)
    sk.k_tc_sum(((R * ntok + 255) // 256,), (256,), (cp.asarray(P), y, i32(R), i32(ntok), i32(sp)))
    ref = np.zeros((ntok, R), dtype=np.float32)
    for p in range(sp):
        ref = (ref + P.reshape(ntok, sp, R)[:, p, :]).astype(np.float32)
    assert np.array_equal(cp.asnumpy(y).reshape(ntok, R), ref)


def test_whole_chain():
    """one random tensor of 96 rows and 64 groups: pack -> q8 -> mv/mm agree with the model in every entry, and the
    entries that are bit-identical to the two fp32 emissions are counted (the bound check is the assertion)"""
    NG, R, ntok = 64, 96, 8
    t = _trits(NG, R, "random", seed=77)
    d = _scales(NG, R, "random")
    W = pack(build_src(t, d), R, NG)
    X = _inputs(ntok, NG, "normal", seed=78)
    XQ, XS, XA = q8(X, NG)
    check("chain mm2", mm(2, W, R, NG, XQ, XS, XA, ntok), _model(t, d, X))
    check("chain mm16", mm(16, W, R, NG, XQ, XS, XA, ntok), _model(t, d, X))
    for sp in (1, 4):
        y = mv(4, W, R, NG, XQ, XS, XA, ntok, sp)
        y64, y32, yfma, tabs, ng = _model(t, d, X, sp)
        e32 = int(np.count_nonzero(y.view(np.uint32) == y32.view(np.uint32)))
        efma = int(np.count_nonzero(y.view(np.uint32) == yfma.view(np.uint32)))
        check("chain mv4 sp%d" % sp, y, _model(t, d, X, sp))
        print("whole chain sp%d: bit-identical to the fused emission %d/%d, to the rounded one %d/%d, max err %.3g"
              % (sp, efma, y.size, e32, y.size, float(np.abs(y - y64).max())))
        if max(e32, efma) != y.size:
            print("note: %d entries differ from both modelled emissions but stay within the rounding bound"
                  % (y.size - max(e32, efma)))


def main():
    if cp is None:
        print("cupy is not installed; skipping")
        return 0
    p = cp.cuda.runtime.getDeviceProperties(0)
    print("device: %s (sm_%d%d; the Turing module is compute_75 PTX, JIT-compiled)"
          % (p["name"], p["major"], p["minor"]))
    fails = 0
    for fn in (test_reference_layout, test_activations, test_pack_layout, test_products, test_token_counts,
               test_shapes_and_merges, test_sum, test_whole_chain):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print("FAILED %-24s %s" % (fn.__name__, e))
        else:
            print("ok     %s" % fn.__name__)
    print("%d test(s) failed" % fails if fails else "all tests passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
