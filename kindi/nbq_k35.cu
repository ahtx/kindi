/* nbq_k35.cu - kernels for Kepler (sm_35, sm_37: K6000, K4200, K40, K80), appended to nbq.cu.

   GK110 has no __dp4a and no tensor cores; its integer multiply and __popc run at a quarter of the fp32 rate, and
   the other integer operations at about half (measured on a K6000: about 80 per clock and SMX).
   - Decode (nbq_mvc): the int8 dot product of a run (32 weights) in fp32, exactly. Every weight value s of these
     types is in -63 .. 63. One PRMT puts u = s + 64 (1 .. 127) into bits 16 .. 22 of 2^23, which is the float
     2^16 (192 + s); one FFMA multiplies it by the input (an integer float, -127 .. 127) and adds it to the sum.
     Each product has at most 15 significant bits and a sum of 16 at most 20, so every step is exact, and
     sum * 2^-16 - 192 * (sum of the inputs) is the int32 dot product. A warp owns 32 runs of the input, so the
     input is converted once and not read again for each row. The weights are in a chunk-major layout
     (nbq_kreorder), so that a warp reads contiguous memory.
   - Prompts (nbq_kmm): an fp32 tile product of the decoded weights and the int8 inputs.
   On other cards the file is empty. */
#ifdef NBQ_K35

/* byte k of u (1 .. 127) as the float 2^16 (128 + u) */
#define NBQ_BH(u, k) __int_as_float(__byte_perm((u), 0x4b000000u, 0x7054 + ((k) << 8)))
/* byte k of the int8 word q as a float (-128 .. 127) */
#define NBQ_BF(q, k) (__int_as_float(__byte_perm((u32)(q) ^ 0x80808080u, 0x4b000000u, 0x7650 + (k))) - 8388736.0f)

__device__ __forceinline__ void nbq_kxf(int4 a, int4 c, float* x) {
    const int q[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        x[4 * i] = NBQ_BF(q[i], 0); x[4 * i + 1] = NBQ_BF(q[i], 1);
        x[4 * i + 2] = NBQ_BF(q[i], 2); x[4 * i + 3] = NBQ_BF(q[i], 3);
    }
}

/* 2^16 * sum (192 + s_i) x_i over 16 weights (4 words of u) */
__device__ __forceinline__ float nbq_kdot16(const u32* u, const float* x) {
    float s[4];                                        /* four chains (FFMA latency); the sums are exact */
#pragma unroll
    for (int i = 0; i < 4; i++) {
        s[i] = __fmul_rn(NBQ_BH(u[i], 0), x[4 * i]); s[i] = __fmaf_rn(NBQ_BH(u[i], 1), x[4 * i + 1], s[i]);
        s[i] = __fmaf_rn(NBQ_BH(u[i], 2), x[4 * i + 2], s[i]); s[i] = __fmaf_rn(NBQ_BH(u[i], 3), x[4 * i + 3], s[i]);
    }
    return (s[0] + s[1]) + (s[2] + s[3]);
}

/* run `sub` of a block (the decode of the other cards, nbq_run) as 8 words of u = s + 64 */
template <int T>
__device__ __forceinline__ void nbq_krun(const u8* b, const __half* dh, int sub, u32 u[8], float& s0, float& s1,
                                         float& mn, const typename Tab<T>::E* tab) {
    int v[8];
    nbq_run<T>(b, dh, sub, v, s0, s1, mn, tab);
#pragma unroll
    for (int i = 0; i < 8; i++) u[i] = ((u32)v[i] ^ 0x80808080u) - 0x40404040u;
}

/* the IQ3_XXS sign table: entry s7 (the 8th sign bit makes the parity even) = the two XOR words that turn 4 codebook
   bytes g (1 .. 62) into u = 64 + g, or 64 - g = (g ^ 63) + 1 where the sign is set; the ADD words follow from them
   ((X ^ 0x40404040) & 0x01010101) */
__device__ __forceinline__ const uint2* nbq_ksigns2(uint2* sh) {
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        const u32 s8 = i | ((__popc(i) & 1u) << 7);
        u32 x[2] = {0x40404040u, 0x40404040u};
        for (int b = 0; b < 8; b++)
            if (s8 >> b & 1u) x[b >> 2] ^= 0x7fu << (8 * (b & 3));
        sh[i] = make_uint2(x[0], x[1]);
    }
    __syncthreads();
    return sh;
}
__device__ __forceinline__ u32 nbq_ku(u32 g, u32 x) { return (g ^ x) + ((x ^ 0x40404040u) & 0x01010101u); }

/* ---------------------------------------------------------------- column chunks
   nbq_mvc: a warp owns 32 runs of the input (1024 columns, one run per lane), converts them to floats once, and
   then goes over a range of rows: per row, each lane decodes its run of that row (the 32 lanes read 32
   consecutive runs: coalesced) and the warp sums the 32 run products. The input is not read again per row, so
   the loop is the weight loads, the decode and the products. part[chunk][row] gets the sum of a chunk;
   nbq_mvc_sum adds the chunks in order (deterministic). Grid (row sets, chunks), 4 warps per block. */
/* the sign table of the I-quants: entry n (4 sign bits) = the XOR word and the ADD word that turn 4 codebook
   bytes g (0 .. 63) into u = 64 + g, or 64 - g = (g ^ 63) + 1 where the bit is set. 16 entries of 8 bytes fill
   16 banks once: the lookups of a warp never conflict. */
__device__ __forceinline__ const uint2* nbq_knib(uint2* sh) {
    if (threadIdx.x < 16) {
        u32 x = 0x40404040u, a = 0u;
        for (int b = 0; b < 4; b++)
            if (threadIdx.x >> b & 1u) { x ^= 0x7fu << (8 * b); a |= 1u << (8 * b); }
        sh[threadIdx.x] = make_uint2(x, a);
    }
    __syncthreads();
    return sh;
}
#define NBQ_KU(g, n) ((g) ^ (n).x) + (n).y

template <int T>
__device__ __forceinline__ void nbq_kdec(const u8* b, const __half* dh, int sub, u32 u[8], float& s0, float& s1,
                                         float& mn, const typename Tab<T>::E* tab, const uint2* nib, const uint2* s7t) {
    if (T == T_IQ3XXS) {
        const uint2 qs = __ldg(reinterpret_cast<const uint2*>(b + 8 * sub));
        const u32 aux = __ldg(reinterpret_cast<const u32*>(b + 64 + 4 * sub));
        const float d = __half2float(__ldg(dh));
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint2 m = s7t[(aux >> (7 * l)) & 127u];        /* (__popc on GK110: a quarter of the rate) */
            const u32 w2 = l < 2 ? qs.x : qs.y;
            const u32 g0 = tab[__byte_perm(w2, 0u, 0x4440 + 2 * (l & 1))];
            const u32 g1 = tab[__byte_perm(w2, 0u, 0x4441 + 2 * (l & 1))];
            u[2 * l] = nbq_ku(g0, m.x);
            u[2 * l + 1] = nbq_ku(g1, m.y);
        }
        s0 = s1 = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        mn = 0.0f;
    } else if (T == T_IQ3S) {
        const float d = __half2float(__ldg(dh));
        const u32 qa = __ldg(reinterpret_cast<const u32*>(b + 8 * sub)), qb = __ldg(reinterpret_cast<const u32*>(b + 8 * sub + 4));
        const u32 qh = (__ldg(reinterpret_cast<const u32*>(b + 64 + 4 * (sub >> 2))) >> (8 * (sub & 3))) & 255u;
        const u32 sg = __ldg(reinterpret_cast<const u32*>(b + 72 + 4 * sub));
        const u32 scw = __ldg(reinterpret_cast<const u32*>(b + 104));
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const u32 s8 = (sg >> (8 * l)) & 255u, w2 = l < 2 ? qa : qb;
            u[2 * l] = NBQ_KU(tab[(((w2 >> (16 * (l & 1))) & 255u) | ((qh << (8 - 2 * l)) & 256u))], nib[s8 & 15u]);
            u[2 * l + 1] = NBQ_KU(tab[(((w2 >> (16 * (l & 1) + 8)) & 255u) | ((qh << (7 - 2 * l)) & 256u))], nib[s8 >> 4]);
        }
        s0 = s1 = d * (float)(1 + 2 * ((scw >> (4 * sub)) & 15u));
        mn = 0.0f;
    } else if (T == T_IQ2S) {
        const float d = __half2float(__ldg(dh));
        const u32 qs = __ldg(reinterpret_cast<const u32*>(b + 4 * sub));
        const u32 sg = __ldg(reinterpret_cast<const u32*>(b + 32 + 4 * sub));
        const uint2 hw = __ldg(reinterpret_cast<const uint2*>(b + 64)), cw = __ldg(reinterpret_cast<const uint2*>(b + 72));
        const u32 qh = ((sub < 4 ? hw.x : hw.y) >> (8 * (sub & 3))) & 255u, sc = ((sub < 4 ? cw.x : cw.y) >> (8 * (sub & 3))) & 255u;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const unsigned long long g = tab[((qs >> (8 * l)) & 255u) | ((qh << (8 - 2 * l)) & 0x300u)];
            const u32 s8 = (sg >> (8 * l)) & 255u;
            u[2 * l] = NBQ_KU((u32)g, nib[s8 & 15u]);
            u[2 * l + 1] = NBQ_KU((u32)(g >> 32), nib[s8 >> 4]);
        }
        s0 = d * (0.5f + (float)(sc & 15u)) * 0.25f;
        s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        mn = 0.0f;
    } else if (T == T_Q4K || T == T_Q5K) {
        /* 16-byte loads: the header (d, dmin, the 12 scale bytes), 32 bytes of nibbles (and of high bits) */
        const uint4 h = __ldg(reinterpret_cast<const uint4*>(b));
        const float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xffffu)));
        const float dm = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
        int sc, m;                                     /* kq_scale_min on the words y, z, w (scale bytes 0-11) */
        {
            const int k = 8 * (sub & 3);
            const u32 by = (h.y >> k) & 255u, bz = (h.z >> k) & 255u;
            if (sub < 4) { sc = by & 63; m = bz & 63; }
            else { const u32 bw = (h.w >> k) & 255u; sc = (bw & 15) | ((by >> 6) << 4); m = (bw >> 4) | ((bz >> 6) << 4); }
        }
        const uint4* ql = reinterpret_cast<const uint4*>(b + (T == T_Q4K ? 16 : 48) + 32 * (sub >> 1));
        const uint4 q0 = __ldg(ql), q1 = __ldg(ql + 1);
        const u32 q[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
        const int sh = 4 * (sub & 1);
        if (T == T_Q5K) {
            const uint4 e0 = __ldg(reinterpret_cast<const uint4*>(b + 16)), e1 = __ldg(reinterpret_cast<const uint4*>(b + 32));
            const u32 e[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
#pragma unroll
            for (int k = 0; k < 8; k++)
                u[k] = (((q[k] >> sh) & 0x0f0f0f0fu) | (((e[k] >> sub) & 0x01010101u) << 4)) + 0x40404040u;
        } else {
#pragma unroll
            for (int k = 0; k < 8; k++) u[k] = ((q[k] >> sh) & 0x0f0f0f0fu) + 0x40404040u;
        }
        s0 = s1 = d * (float)sc;
        mn = dm * (float)m;
    } else {
        nbq_krun<T>(b, dh, sub, u, s0, s1, mn, tab);
    }
}

/* the sums over the 32 lanes of v[0 .. RB): a butterfly that halves the rows at each step, then plain steps; lane
   L ends with row L % RB in v[0] (no long chain of dependent shuffles per row) */
template <int RB>
__device__ __forceinline__ float nbq_krows(float* v, int lane) {
#pragma unroll
    for (int s = RB / 2; s >= 1; s >>= 1) {
        const bool up = lane & s;
#pragma unroll
        for (int j = 0; j < s; j++) {
            const float send = up ? v[j] : v[j + s], keep = up ? v[j + s] : v[j];
            v[j] = keep + __shfl_xor_sync(0xffffffffu, send, s);
        }
    }
#pragma unroll
    for (int s = RB; s < 32; s <<= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], s);
    return v[0];
}

template <int T, int RB>
__device__ __forceinline__ void nbq_mvc(const u8* w, const __half* wd, const int* xq, const float* xs, float* part,
                                        int rows, int cols, int rpw) {
    NBQ_TAB(T, tab);
    __shared__ uint2 sg_s[16], s7_s[T == T_IQ3XXS ? 128 : 1];
    const uint2* sg = nbq_knib(sg_s);
    const uint2* s7t = T == T_IQ3XXS ? nbq_ksigns2(s7_s) : s7_s;
    const int lane = threadIdx.x & 31, nb = cols >> 8, nr = nb * 8;
    const int g = blockIdx.y * 32 + lane;              /* this lane's run */
    const bool on = g < nr;
    const int sub = g & 7;
    float x[32], sl = 0.0f, sh = 0.0f, xsg = 0.0f, sx;
    if (on) {
        const int4* xp = reinterpret_cast<const int4*>(xq + 8 * g);
        nbq_kxf(xp[0], xp[1], x);
        float sq[8];
#pragma unroll
        for (int i = 0; i < 8; i++) sq[i] = (x[4 * i] + x[4 * i + 1]) + (x[4 * i + 2] + x[4 * i + 3]);
        sl = (sq[0] + sq[1]) + (sq[2] + sq[3]); sh = (sq[4] + sq[5]) + (sq[6] + sq[7]);
        xsg = xs[g];
    } else {
#pragma unroll
        for (int i = 0; i < 32; i++) x[i] = 0.0f;
    }
    sx = sl + sh; sl *= -192.0f; sh *= -192.0f;
    /* rows [r0, r1) of this warp; rpw and rows are multiples of RB */
    const int r0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * rpw, r1 = min(rows, r0 + rpw);
    /* the chunk-major layout (nbq_kreorder): chunk c holds 4 blocks of every row, row after row */
    const int rstep = 4 * BT<T>::B;
    const u8* b = w + ((size_t)blockIdx.y * rows + r0) * rstep + (lane >> 3) * BT<T>::B;
    const __half* dh = wd + ((size_t)blockIdx.y * rows + r0) * 4 + (lane >> 3);
    for (int rb = r0; rb < r1; rb += RB) {
        float v[RB];
#pragma unroll
        for (int i = 0; i < RB; i++, b += rstep, dh += 4) {
            u32 u[8]; float s0, s1, mn;
            nbq_kdec<T>(b, dh, sub, u, s0, s1, mn, tab, sg, s7t);
            const float lo = __fmaf_rn(nbq_kdot16(u, x), 1.52587890625e-05f, sl);
            const float hi = __fmaf_rn(nbq_kdot16(u + 4, x + 16), 1.52587890625e-05f, sh);
            float f = nbq_runf(s0, lo, s1, hi);
            if (T == T_Q4K || T == T_Q5K) f = __fmaf_rn(-mn, sx, f);
            v[i] = on ? __fmul_rn(xsg, f) : 0.0f;
        }
        const float t = nbq_krows<RB>(v, lane);
        if (lane < RB) part[(size_t)blockIdx.y * rows + rb + lane] = t;
    }
}
#define NBQ_MVC(T, NAME) extern "C" __global__ void __launch_bounds__(128) NAME(const u8* w, const __half* wd,     const int* xq, const float* xs, float* part, int rows, int cols, int rpw) { nbq_mvc<T, NBQ_KRB>(w, wd, xq, xs, part, rows, cols, rpw); }
#ifndef NBQ_KRB
#define NBQ_KRB 8
#endif
NBQ_MVC(T_Q4K, nbq_mvc_q4k)
NBQ_MVC(T_Q5K, nbq_mvc_q5k)
NBQ_MVC(T_IQ3XXS, nbq_mvc_iq3xxs)
NBQ_MVC(T_IQ3S, nbq_mvc_iq3s)
NBQ_MVC(T_IQ2S, nbq_mvc_iq2s)

/* the chunk-major layout of the nbq_mvc kernels: dst[c][r][4 blocks] = src[r][4 c .. 4 c + 4) (blocks past the row
   are zeros), the same for the fp16 scales when there are any (I-quants) */
extern "C" __global__ void nbq_kreorder(const u8* src, const __half* sd, u8* dst, __half* dd, int rows, int nb, int bsz) {
    const int nc = (nb + 3) / 4;
    const long long n = (long long)nc * rows * 4;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(i & 3), r = (int)((i >> 2) % rows), c = (int)((i >> 2) / rows), blk = 4 * c + k;
        u8* o = dst + i * bsz;
        if (blk < nb) {
            const u8* p = src + ((long long)r * nb + blk) * bsz;
            for (int j = 0; j < bsz; j++) o[j] = p[j];
            if (dd) dd[i] = sd[(long long)r * nb + blk];
        } else {
            for (int j = 0; j < bsz; j++) o[j] = 0;
            if (dd) dd[i] = __float2half(0.0f);
        }
    }
}

/* y[r] (+ add[r]) = the sum of part[c][r] over the chunks c, in order */
extern "C" __global__ void nbq_mvc_sum(const float* part, int nc, int rows, float* y, const float* add) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += gridDim.x * blockDim.x) {
        float s = 0.0f;
        for (int c = 0; c < nc; c++) s += part[(size_t)c * rows + r];
        y[r] = add ? add[r] + s : s;
    }
}

/* ---------------------------------------------------------------- prompt: matrix x matrix
   nbq_kmm: Y[t][yo + r] = W x_t for nt inputs, an fp32 tile product: a block takes 64 rows x 64 tokens and goes
   over the columns one run (32) at a time. Half the threads decode the 64 runs of the weights into shared memory
   as floats (scale * value - min), the other half the 64 runs of the inputs (xs * q); then each thread adds 4 rows
   x 8 tokens. (A 128 x 64 tile with 8 x 8 per thread was slower on the K6000: 138 registers, 8 warps per SMX.) Run it with
   8-byte shared memory banks. The weights are in the chunk-major layout. (Not the bits of the decode path: the products are fp32
   products of scaled values. llama.cpp also uses other arithmetic for prompts than for single tokens.) */
#define KMM_LD 68                 /* shared row stride in floats: a 64-wide tile row + 4 */
template <int T>
__device__ __forceinline__ void nbq_kmm(const u8* w, const __half* wd, const signed char* xq, const float* xs,
                                        float* y, int rows, int cols, int nt, int ldy, int yo) {
    NBQ_TAB(T, tab);
    __shared__ uint2 sg_s[16], s7_s[T == T_IQ3XXS ? 128 : 1];
    const uint2* sg = nbq_knib(sg_s);
    const uint2* s7t = T == T_IQ3XXS ? nbq_ksigns2(s7_s) : s7_s;
    __shared__ __align__(16) float Ws[32 * KMM_LD], Xs[32 * KMM_LD];
    const int tid = threadIdx.x, nb = cols >> 8, nr = nb * 8;
    const int R0 = blockIdx.x * 64, T0 = blockIdx.y * 64;
    const int ty = tid >> 3, tx = tid & 7;             /* rows 4 ty .. +4, tokens 8 tx .. +8 */
    float acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    const int lr = tid & 63;                           /* the row (tid < 64) or token (tid >= 64) this thread loads */
    for (int g = 0; g < nr; g++) {
        if (tid < 64) {
            const size_t kb = ((size_t)(g >> 5) * rows + R0 + lr) * 4 + ((g >> 3) & 3);
            u32 u[8]; float s0, s1, mn;
            nbq_kdec<T>(w + kb * BT<T>::B, wd + kb, g & 7, u, s0, s1, mn, tab, sg, s7t);
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const float v = __int_as_float(__byte_perm(u[i], 0x4b000000u, 0x7650 + k)) - 8388672.0f;
                    Ws[(4 * i + k) * KMM_LD + lr] = __fmaf_rn(i < 4 ? s0 : s1, v, -mn);
                }
            }
        } else {
            const int t = T0 + lr;
            if (t < nt) {
                const int4* xp = reinterpret_cast<const int4*>(xq + (size_t)t * cols + 32 * g);
                float x[32];
                nbq_kxf(__ldg(xp), __ldg(xp + 1), x);
                const float sc = __ldg(xs + (size_t)t * (cols >> 5) + g);
#pragma unroll
                for (int k = 0; k < 32; k++) Xs[k * KMM_LD + lr] = sc * x[k];
            } else {
#pragma unroll
                for (int k = 0; k < 32; k++) Xs[k * KMM_LD + lr] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll 8
        for (int k = 0; k < 32; k++) {
            const float4 a = *reinterpret_cast<const float4*>(Ws + k * KMM_LD + 4 * ty);
            const float4 b0 = *reinterpret_cast<const float4*>(Xs + k * KMM_LD + 8 * tx);
            const float4 b1 = *reinterpret_cast<const float4*>(Xs + k * KMM_LD + 8 * tx + 4);
            const float wa[4] = {a.x, a.y, a.z, a.w}, xb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = __fmaf_rn(wa[i], xb[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int t = T0 + 8 * tx + j;
        if (t < nt)
            *reinterpret_cast<float4*>(y + (size_t)t * ldy + yo + R0 + 4 * ty) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
    }
}
#define NBQ_KMM(T, NAME) extern "C" __global__ void __launch_bounds__(128) NAME(const u8* w, const __half* wd,     const signed char* xq, const float* xs, float* y, int rows, int cols, int nt, int ldy, int yo) {     nbq_kmm<T>(w, wd, xq, xs, y, rows, cols, nt, ldy, yo); }
NBQ_KMM(T_Q4K, nbq_kmm_q4k)
NBQ_KMM(T_Q5K, nbq_kmm_q5k)
NBQ_KMM(T_IQ3XXS, nbq_kmm_iq3xxs)
NBQ_KMM(T_IQ3S, nbq_kmm_iq3s)
NBQ_KMM(T_IQ2S, nbq_kmm_iq2s)

/* ---------------------------------------------------------------- prompt through cuBLAS
   nbq_kdeq: rows [r0, r0 + nr) of a weight matrix as fp32 (scale * value - min), row-major [nr][cols]; one thread
   per run. nbq_kx2f: the int8 inputs as fp32 (xs * q), [nt][cols]. nanbeige.py multiplies them with cublasSgemm. */
template <int T>
__device__ __forceinline__ void nbq_kdeq(const u8* w, const __half* wd, int rows, int cols, int r0, int nr, float* out) {
    NBQ_TAB(T, tab);
    __shared__ uint2 sg_s[16], s7_s[T == T_IQ3XXS ? 128 : 1];
    const uint2* sg = nbq_knib(sg_s);
    const uint2* s7t = T == T_IQ3XXS ? nbq_ksigns2(s7_s) : s7_s;
    const int nr32 = cols >> 5, g = blockIdx.y * blockDim.x + threadIdx.x, rl = blockIdx.x;
    if (g >= nr32 || rl >= nr) return;
    const size_t kb = ((size_t)(g >> 5) * rows + r0 + rl) * 4 + ((g >> 3) & 3);
    u32 u[8]; float s0, s1, mn;
    nbq_kdec<T>(w + kb * BT<T>::B, wd + kb, g & 7, u, s0, s1, mn, tab, sg, s7t);
    float4* o = reinterpret_cast<float4*>(out + (size_t)rl * cols + 32 * g);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float s = i < 4 ? s0 : s1;
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; k++)
            v[k] = __fmaf_rn(s, __int_as_float(__byte_perm(u[i], 0x4b000000u, 0x7650 + k)) - 8388672.0f, -mn);
        o[i] = make_float4(v[0], v[1], v[2], v[3]);
    }
}
#define NBQ_KDEQ(T, NAME) extern "C" __global__ void __launch_bounds__(128) NAME(const u8* w, const __half* wd,     int rows, int cols, int r0, int nr, float* out) { nbq_kdeq<T>(w, wd, rows, cols, r0, nr, out); }
NBQ_KDEQ(T_Q4K, nbq_kdeq_q4k)
NBQ_KDEQ(T_Q5K, nbq_kdeq_q5k)
NBQ_KDEQ(T_IQ3XXS, nbq_kdeq_iq3xxs)
NBQ_KDEQ(T_IQ3S, nbq_kdeq_iq3s)
NBQ_KDEQ(T_IQ2S, nbq_kdeq_iq2s)

extern "C" __global__ void nbq_kx2f(const signed char* xq, const float* xs, int nt, int cols, float* out) {
    const size_t n = (size_t)nt * cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = xs[i >> 5] * (float)xq[i];
}

/* dst[i] = src[i]; src may be pinned host memory (read over PCIe by the SMs, not by a copy engine, so that the
   copy does not wait behind the host-layer copies of nanbeige.py) */
extern "C" __global__ void nbq_kcopy(const float* src, float* dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

#endif
