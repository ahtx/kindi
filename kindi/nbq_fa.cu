/* nbq_fa.cu - prompt attention on fp16 tensor cores (appended to nbq.cu at compile time).

   Causal attention of nt prompt tokens at positions pos0 + t over the cache [0, pos0 + t]. Query rows are
   (token, head of the K/V group): row r = 7 t + j. grid (rows / 64, nkv), 128 threads; warp w takes rows
   16 w .. +15 against tiles of 64 positions, with a running max and sum per row (FA2). Q is scaled, then rounded
   to fp16; P is rounded to fp16 for P V; the sums are fp32. */
#define NBQ_FLD 136               /* shared row stride in halves */
#define NBQ_MMAH(c, a0, a1, a2, a3, b0, b1) \
  NBQ_ASM("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ u32 nbq_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<u32*>(&h);
}

/* 8 q8_0 values (one fp16 scale) -> 8 fp16 values q * d */
__device__ __forceinline__ uint4 nbq_deq8(uint2 b, float d) {
    uint4 r;
    u32* w = reinterpret_cast<u32*>(&r);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const u32 x = i < 2 ? b.x : b.y;
        const int sh = 16 * (i & 1);
        w[i] = nbq_h2((float)(signed char)((x >> sh) & 255u) * d, (float)(signed char)((x >> (sh + 8)) & 255u) * d);
    }
    return r;
}

/* Q8: the cache is q8_0 (nbq_q8kv.cu): kc, vc int8 [nkv][ctx][128], kd, vd their fp16 scales [nkv][ctx][4] */
template <int G, bool Q8>
__device__ __forceinline__ void nbq_fa_t(const float* qkv, int ld, int nt, int pos0,
        const void* kc_, const __half* kd, const void* vc_, const __half* vd, int ctx, float scale, float* out) {
    const __half* kc = reinterpret_cast<const __half*>(kc_);
    const __half* vc = reinterpret_cast<const __half*>(vc_);
    extern __shared__ __align__(16) unsigned char fsm[];
    __half* Ks = reinterpret_cast<__half*>(fsm);            /* [64][NBQ_FLD] */
    __half* Vs = Ks + 64 * NBQ_FLD;                          /* [64][NBQ_FLD] */
    const int hk = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, t4 = lane & 3;
    const int R = G * nt, r0 = blockIdx.x * 64;
    const int ra = r0 + 16 * warp + gid, rb = ra + 8;       /* the two rows of this thread */
    const int ca = min(ra, R - 1), cb = min(rb, R - 1);
    const int ta = ca / G, tb = cb / G;
    const int la = pos0 + ta, lb = pos0 + tb;               /* the last position each row sees */
    const int lmax = pos0 + min(r0 + 63, R - 1) / G;
    u32 qa[8][4];                                            /* Q fragments: 8 steps of 16 dims */
    {
        const float* pa = qkv + (size_t)ta * ld + (hk * G + ca % G) * 128;
        const float* pb = qkv + (size_t)tb * ld + (hk * G + cb % G) * 128;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            int d = 16 * k + 2 * t4;
            qa[k][0] = nbq_h2(pa[d] * scale, pa[d + 1] * scale);
            qa[k][1] = nbq_h2(pb[d] * scale, pb[d + 1] * scale);
            qa[k][2] = nbq_h2(pa[d + 8] * scale, pa[d + 9] * scale);
            qa[k][3] = nbq_h2(pb[d + 8] * scale, pb[d + 9] * scale);
        }
    }
    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float ma = -1e30f, mb = -1e30f, sa = 0.0f, sb = 0.0f;
    const __half* vbase = vc + (size_t)hk * ctx * 128;
    for (int p0 = 0; p0 <= lmax; p0 += 64) {
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 8; h++) {                       /* K and V tiles: 64 positions x 16 pieces of 8 halves */
            int idx = tid + 128 * h, pc = idx >> 6, p = idx & 63;
            int vp = idx >> 4, vpc = idx & 15;
            uint4 kv, vv;
            if (Q8) {
                const size_t rk = (size_t)hk * ctx + p0 + p, rv = (size_t)hk * ctx + p0 + vp;
                kv = nbq_deq8(*reinterpret_cast<const uint2*>(reinterpret_cast<const signed char*>(kc_) + rk * 128 + 8 * pc),
                              __half2float(kd[rk * 4 + (pc >> 2)]));
                vv = nbq_deq8(*reinterpret_cast<const uint2*>(reinterpret_cast<const signed char*>(vc_) + rv * 128 + 8 * vpc),
                              __half2float(vd[rv * 4 + (vpc >> 2)]));
            } else {
                kv = *reinterpret_cast<const uint4*>(kc + nbq_kofs(hk, ctx, p0 + p, pc));
                vv = *reinterpret_cast<const uint4*>(vbase + (size_t)(p0 + vp) * 128 + 8 * vpc);
            }
            *reinterpret_cast<uint4*>(Ks + p * NBQ_FLD + 8 * pc) = kv;
            *reinterpret_cast<uint4*>(Vs + vp * NBQ_FLD + 8 * vpc) = vv;
        }
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            const __half* kr = Ks + (8 * n + gid) * NBQ_FLD + 2 * t4;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                u32 b0 = *reinterpret_cast<const u32*>(kr + 16 * k), b1 = *reinterpret_cast<const u32*>(kr + 16 * k + 8);
                NBQ_MMAH(s[n], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
            }
        }
        float mxa = -1e30f, mxb = -1e30f;                    /* causal mask and the running max */
#pragma unroll
        for (int n = 0; n < 8; n++) {
            int pc = p0 + 8 * n + 2 * t4;
            if (pc > la) s[n][0] = -1e30f;
            if (pc + 1 > la) s[n][1] = -1e30f;
            if (pc > lb) s[n][2] = -1e30f;
            if (pc + 1 > lb) s[n][3] = -1e30f;
            mxa = fmaxf(mxa, fmaxf(s[n][0], s[n][1]));
            mxb = fmaxf(mxb, fmaxf(s[n][2], s[n][3]));
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
        const float fa = expf(ma - na), fb = expf(mb - nb);
        ma = na; mb = nb;
        sa *= fa; sb *= fb;
#pragma unroll
        for (int n = 0; n < 16; n++) { o[n][0] *= fa; o[n][1] *= fa; o[n][2] *= fb; o[n][3] *= fb; }
        u32 pf[4][4];
#pragma unroll
        for (int n = 0; n < 8; n++) {
            float e0 = expf(s[n][0] - na), e1 = expf(s[n][1] - na), e2 = expf(s[n][2] - nb), e3 = expf(s[n][3] - nb);
            sa += e0 + e1; sb += e2 + e3;
            pf[n >> 1][(n & 1) * 2] = nbq_h2(e0, e1);
            pf[n >> 1][(n & 1) * 2 + 1] = nbq_h2(e2, e3);
        }
        /* O += P V: 16 positions per step, 8 dims per tile; the V fragments come from ldmatrix.trans */
        const int m8 = lane >> 3, r8 = lane & 7;
#pragma unroll
        for (int k = 0; k < 4; k++) {
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                const __half* addr = Vs + (16 * k + (m8 & 1) * 8 + r8) * NBQ_FLD + 8 * n + (m8 >> 1) * 8;
                u32 b0, b1, b2, b3;
                u32 sad = (u32)__cvta_generic_to_shared(addr);
                NBQ_ASM("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(sad));
                NBQ_MMAH(o[n], pf[k][0], pf[k][1], pf[k][2], pf[k][3], b0, b1);
                NBQ_MMAH(o[n + 1], pf[k][0], pf[k][1], pf[k][2], pf[k][3], b2, b3);
            }
        }
    }
    sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
    sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
    const float ia = 1.0f / sa, ib = 1.0f / sb;
    const int lo = gridDim.y * G * 128;
    float* oa = out + (size_t)ta * lo + (hk * G + ca % G) * 128;
    float* ob = out + (size_t)tb * lo + (hk * G + cb % G) * 128;
#pragma unroll
    for (int n = 0; n < 16; n++) {
        int d = 8 * n + 2 * t4;
        if (ra < R) { oa[d] = o[n][0] * ia; oa[d + 1] = o[n][1] * ia; }
        if (rb < R) { ob[d] = o[n][2] * ib; ob[d + 1] = o[n][3] * ib; }
    }
}
#define NBQ_FA(G, NAME) extern "C" __global__ void __launch_bounds__(128) NAME(const float* qkv, int ld, int nt, \
    int pos0, const __half* kc, const __half* vc, int ctx, float scale, float* out) { \
    nbq_fa_t<G, false>(qkv, ld, nt, pos0, kc, nullptr, vc, nullptr, ctx, scale, out); }
#define NBQ_FA_Q8(G, NAME) extern "C" __global__ void __launch_bounds__(128) NAME(const float* qkv, int ld, int nt,     int pos0, const signed char* kq, const __half* kd, const signed char* vq, const __half* vd, int ctx, float scale,     float* out) { nbq_fa_t<G, true>(qkv, ld, nt, pos0, kq, kd, vq, vd, ctx, scale, out); }
#if NBQ_TC
NBQ_FA(7, nbq_fa_prefill)
NBQ_FA(5, nbq_fa_prefill_g5)
NBQ_FA_Q8(7, nbq_fa_prefill_q8)
NBQ_FA_Q8(5, nbq_fa_prefill_q8_g5)
#endif

#if NBQ_HMMA
/* Volta (sm_70): the same causal prompt attention as nbq_fa_t (f16 cache), on mma.sync m8n8k4. Block: 64 query
   rows of one K/V head (row r = G t + j), 128 threads; tiles of 64 positions with a running max and sum per row
   (FA2). Warp w takes rows 16 w .. +15; its quad pair qp takes the 8 rows of group qp & 1 and, for S = Q K^T, the
   positions 32 (qp >> 1) .. +31 of the tile; for O += P V, the dims 64 (qp >> 1) .. +63. Accumulator element e of
   a lane is row (lane & 1) + (e & 2) (+ 4 in the upper half of the quad pair), column (e & 4) + (lane & 2) + (e & 1)
   of its 8 x 8 tile, so a lane's two rows are the same in S and in O. Q is scaled and rounded to fp16, P is fp16 for
   P V, the sums are fp32 (as nbq_fa_t). */
#define NBQ_VLD 72                /* the shared row stride of V transposed and of P, in halves */
#define NBQ_FAV_SMEM ((64 * NBQ_FLD + 64 * NBQ_FLD + 128 * NBQ_VLD + 4 * 16 * NBQ_VLD) * 2)
template <int G>
__device__ __forceinline__ void nbq_fa_v(const float* qkv, int ld, int nt, int pos0, const __half* kc,
                                         const __half* vc, int ctx, float scale, float* out) {
    extern __shared__ __align__(16) unsigned char fsm[];
    __half* Qs = reinterpret_cast<__half*>(fsm);            /* [64][NBQ_FLD] */
    __half* Ks = Qs + 64 * NBQ_FLD;                          /* [64][NBQ_FLD] */
    __half* Vt = Ks + 64 * NBQ_FLD;                          /* [128 dims][NBQ_VLD positions] */
    const int hk = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __half* P = Vt + 128 * NBQ_VLD + warp * 16 * NBQ_VLD;    /* this warp's [16 rows][64 positions] */
    const int R = G * nt, r0 = blockIdx.x * 64;
    const int qp = (lane >> 2) & 3, hi = lane >> 4, qr = (lane & 3) + 4 * hi, rg = qp & 1, hf = qp >> 1;
    const int lra = 8 * rg + (lane & 1) + 4 * hi, lrb = lra + 2;            /* this lane's rows in the warp's 16 */
    const int ra = r0 + 16 * warp + lra, rb = r0 + 16 * warp + lrb;
    const int ca = min(ra, R - 1), cb = min(rb, R - 1);
    const int la = pos0 + ca / G, lb = pos0 + cb / G;        /* the last position each row sees */
    const int lmax = pos0 + min(r0 + 63, R - 1) / G;
    for (int idx = tid; idx < 64 * 64; idx += 128) {        /* Q: 64 rows x 64 pairs of dims, scaled, fp16 */
        const int r = idx >> 6, d = 2 * (idx & 63), rr = min(r0 + r, R - 1);
        const float* q = qkv + (size_t)(rr / G) * ld + (hk * G + rr % G) * 128 + d;
        *reinterpret_cast<__half2*>(Qs + r * NBQ_FLD + d) = __floats2half2_rn(q[0] * scale, q[1] * scale);
    }
    float o[8][8];
#pragma unroll
    for (int m = 0; m < 8; m++)
#pragma unroll
        for (int e = 0; e < 8; e++) o[m][e] = 0.0f;
    float ma = -1e30f, mb = -1e30f, sa = 0.0f, sb = 0.0f;
    const __half* vbase = vc + (size_t)hk * ctx * 128;
    for (int p0 = 0; p0 <= lmax; p0 += 64) {
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 8; h++) {                       /* K: 64 positions x 16 pieces; V transposed */
            const int idx = tid + 128 * h, pc = idx >> 6, p = idx & 63, vp = idx >> 4, vpc = idx & 15;
            *reinterpret_cast<uint4*>(Ks + p * NBQ_FLD + 8 * pc) = *reinterpret_cast<const uint4*>(kc + nbq_kofs(hk, ctx, p0 + p, pc));
            const uint4 vv = *reinterpret_cast<const uint4*>(vbase + (size_t)(p0 + vp) * 128 + 8 * vpc);
            const __half* hv = reinterpret_cast<const __half*>(&vv);
#pragma unroll
            for (int e = 0; e < 8; e++) Vt[(8 * vpc + e) * NBQ_VLD + vp] = hv[e];
        }
        __syncthreads();
        float s[4][8];
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 8; e++) s[n][e] = 0.0f;
        const __half* qrow = Qs + (16 * warp + 8 * rg + qr) * NBQ_FLD;
#pragma unroll 4
        for (int k = 0; k < 32; k++) {
            const uint2 a = *reinterpret_cast<const uint2*>(qrow + 4 * k);
#pragma unroll
            for (int n = 0; n < 4; n++) {
                const uint2 b = *reinterpret_cast<const uint2*>(Ks + (8 * (4 * hf + n) + qr) * NBQ_FLD + 4 * k);
                NBQ_HMMA884(s[n], a, b);
            }
        }
        float mxa = -1e30f, mxb = -1e30f;
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int p = p0 + 8 * (4 * hf + n) + (e & 4) + (lane & 2) + (e & 1);
                if (e & 2) { if (p > lb) s[n][e] = -1e30f; mxb = fmaxf(mxb, s[n][e]); }
                else { if (p > la) s[n][e] = -1e30f; mxa = fmaxf(mxa, s[n][e]); }
            }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 8));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 8));
        const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
        const float fa = expf(ma - na), fb = expf(mb - nb);
        ma = na; mb = nb;
        sa *= fa; sb *= fb;
#pragma unroll
        for (int m = 0; m < 8; m++)
#pragma unroll
            for (int e = 0; e < 8; e++) o[m][e] *= (e & 2) ? fb : fa;
#pragma unroll
        for (int n = 0; n < 4; n++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const float ev = expf(s[n][e] - ((e & 2) ? nb : na));
                if (e & 2) sb += ev; else sa += ev;
                P[((e & 2) ? lrb : lra) * NBQ_VLD + 8 * (4 * hf + n) + (e & 4) + (lane & 2) + (e & 1)] = __float2half_rn(ev);
            }
        __syncwarp();
        const __half* prow = P + (8 * rg + qr) * NBQ_VLD;
#pragma unroll 4
        for (int k = 0; k < 16; k++) {
            const uint2 a = *reinterpret_cast<const uint2*>(prow + 4 * k);
#pragma unroll
            for (int m = 0; m < 8; m++) {
                const uint2 b = *reinterpret_cast<const uint2*>(Vt + (8 * (8 * hf + m) + qr) * NBQ_VLD + 4 * k);
                NBQ_HMMA884(o[m], a, b);
            }
        }
        __syncwarp();
    }
    sa += __shfl_xor_sync(0xffffffffu, sa, 2); sa += __shfl_xor_sync(0xffffffffu, sa, 8);
    sb += __shfl_xor_sync(0xffffffffu, sb, 2); sb += __shfl_xor_sync(0xffffffffu, sb, 8);
    const float ia = 1.0f / sa, ib = 1.0f / sb;
    const int lo = gridDim.y * G * 128;
    float* oa = out + (size_t)(ca / G) * lo + (hk * G + ca % G) * 128;
    float* ob = out + (size_t)(cb / G) * lo + (hk * G + cb % G) * 128;
#pragma unroll
    for (int m = 0; m < 8; m++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = 8 * (8 * hf + m) + (e & 4) + (lane & 2) + (e & 1);
            if (e & 2) { if (rb < R) ob[d] = o[m][e] * ib; }
            else if (ra < R) oa[d] = o[m][e] * ia;
        }
}
#define NBQ_FAV(G, NAME) extern "C" __global__ void __launch_bounds__(128) NAME(const float* qkv, int ld, int nt, \
    int pos0, const __half* kc, const __half* vc, int ctx, float scale, float* out) { \
    nbq_fa_v<G>(qkv, ld, nt, pos0, kc, vc, ctx, scale, out); }
NBQ_FAV(7, nbq_fav_prefill)
NBQ_FAV(5, nbq_fav_prefill_g5)
#endif
