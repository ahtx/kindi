/* lightning_k35.cu - kernels for cards without dp4a (Kepler sm_35 / sm_37: K6000, K80; Pascal sm_60: P100),
   appended to lightning.cu at compile time. The numbers are the same exact integer run sums as on the other cards.

   - Decode (lt_mvk_q2): the run sums in fp32 without integer multiplies, as nbq_k35.cu does: one PRMT puts a code
     byte b (< 128) into bits 16 .. 23 of 2^23, which is the float 2^16 (128 + b); one FFMA multiplies it by the input
     (an integer float, -127 .. 127) and adds it. A product has at most 15 significant bits and a chain of 16 at most
     19, so every step is exact; sum * 2^-16 - 128 * sum(q) is the integer dot product. Lane l of a warp owns run
     32 c + l of the input (chunk c), so it converts its 32 inputs once and keeps them for all its rows; each block
     writes one partial sum per (chunk, row), and lt_mvk_sum adds the chunks in a fixed order.
   - Prompts (lt_mmf_q2): an fp32 tile product of the digits c - 1 and the integer inputs; a run's 32 products are
     summed exactly, then scaled. */

#define LT_KF(m, k) __int_as_float(__byte_perm((m), 0x4b000000u, 0x7044 + ((k) << 8)))   /* 2^16 (128 + byte k) */
#define LT_QF(q, k) (__int_as_float(__byte_perm((u32)(q) ^ 0x80808080u, 0x4b000000u, 0x7650 + (k))) - 8388736.0f)

/* the 32 inputs of a run (eight lane-order words) as floats */
__device__ __forceinline__ void lt_kx(const int* q, float* x) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int w = q[i];
        x[4 * i] = LT_QF(w, 0); x[4 * i + 1] = LT_QF(w, 1); x[4 * i + 2] = LT_QF(w, 2); x[4 * i + 3] = LT_QF(w, 3);
    }
}

/* 2^16 * sum (128 + c) q over one run of ternary codes (two code words), two exact chains of 16 */
__device__ __forceinline__ float lt_kdot_q2(uint2 w, const float* x) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const u32 m = (w.x >> (2 * i)) & LT_M, n = (w.y >> (2 * i)) & LT_M;
        s0 = __fmaf_rn(LT_KF(m, 0), x[4 * i], s0); s0 = __fmaf_rn(LT_KF(m, 1), x[4 * i + 1], s0);
        s0 = __fmaf_rn(LT_KF(m, 2), x[4 * i + 2], s0); s0 = __fmaf_rn(LT_KF(m, 3), x[4 * i + 3], s0);
        s1 = __fmaf_rn(LT_KF(n, 0), x[16 + 4 * i], s1); s1 = __fmaf_rn(LT_KF(n, 1), x[16 + 4 * i + 1], s1);
        s1 = __fmaf_rn(LT_KF(n, 2), x[16 + 4 * i + 2], s1); s1 = __fmaf_rn(LT_KF(n, 3), x[16 + 4 * i + 3], s1);
    }
    return s0 + s1;
}

/* decode partials: part[slot][chunk][row] = sum over the runs of the chunk of d_w * dx * sum((c - 1) q).
   grid (chunks, ceil(rows / rpb), slots), 128 threads (4 warps; warp w takes rows r0 + w, r0 + w + 4, ...).
   Weights and inputs as lt_mv_q2 (ids NULL: slot = matrix). The loads of four rows go out together: they are
   unconditional (a lane past the last run, or a row past the end, reads a valid neighbour and is masked after), so
   the compiler does not wait for each one; the indices are 32-bit (a matrix has fewer than 2^31 words). */
extern "C" __global__ void __launch_bounds__(128) lt_mvk_q2(const u32* const* __restrict__ codes,
        const __half* const* __restrict__ sc, const int* __restrict__ ids, int rows, int cols,
        const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, int xs, float* part,
        int rpb) {
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int slot = blockIdx.z, ch = blockIdx.x, nch = gridDim.x;
    const int runs = cols >> 5, nb = cols >> 6, h = ch * 32 + lane;
    const bool on = h < runs;
    const int hc = on ? h : runs - 1;
    float x[32];
    const int xi = slot * xs + hc;
    lt_kx(xq + xi * 8, x);
    const float d = on ? dx[xi] : 0.0f;
    const float sq = 129.0f * (float)sx[xi];
    const int e = ids ? ids[slot] : slot;
    const uint2* wb = reinterpret_cast<const uint2*>(codes[e]) + hc;
    const __half* sb = sc[e] + (hc >> 1);
    const int r0 = blockIdx.y * rpb, r1 = min(rows, r0 + rpb);
    float* po = part + (slot * nch + ch) * rows;
    uint2 w[4];
    unsigned short sv[4];
#define LT_ROWS(rb) do { \
        for (int u_ = 0; u_ < 4; u_++) { \
            const unsigned r_ = (unsigned)min((rb) + 4 * u_, r1 - 1); \
            w[u_] = __ldg(wb + r_ * (unsigned)runs); \
            sv[u_] = __ldg(reinterpret_cast<const unsigned short*>(sb) + r_ * (unsigned)nb); \
        } } while (0)
    LT_ROWS(r0 + wp);
    for (int rb = r0 + wp; rb < r1; rb += 16) {
        const uint2 c[4] = {w[0], w[1], w[2], w[3]};
        const unsigned short cv[4] = {sv[0], sv[1], sv[2], sv[3]};
        if (rb + 16 < r1) LT_ROWS(rb + 16);          /* the next four rows load while these are computed */
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float v = __half2float(__ushort_as_half(cv[u])) * d * (lt_kdot_q2(c[u], x) * (1.0f / 65536.0f) - sq);
            v = warp_sum(v);
            if (lane == 0 && rb + 4 * u < r1) po[rb + 4 * u] = v;
        }
    }
#undef LT_ROWS
}

/* the head: Q8_0 rows (lane-order int8, fp16 scale per run). A weight byte can be 128 or more, which the 2^16 form
   cannot hold (bit 23 is the lowest exponent bit), so each weight is converted to its exact float (PRMT, FADD); a
   run's 32 products sum exactly (|sum| < 2^20). Partials as lt_mvk_q2 (one slot); two rows at a time. */
extern "C" __global__ void __launch_bounds__(128) lt_mvk_q8(const int* __restrict__ wq, const __half* __restrict__ sc,
        int rows, int cols, const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx,
        float* part, int rpb) {
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int ch = blockIdx.x, nch = gridDim.x;
    const int runs = cols >> 5, h = ch * 32 + lane;
    const bool on = h < runs;
    const int hc = on ? h : runs - 1;
    float x[32];
    lt_kx(xq + hc * 8, x);
    const float d = on ? dx[hc] : 0.0f;
    const int4* wb = reinterpret_cast<const int4*>(wq) + 2 * hc;
    const __half* sb = sc + hc;
    const int r0 = blockIdx.y * rpb, r1 = min(rows, r0 + rpb);
    float* po = part + (long long)ch * rows;
    for (int rb = r0 + wp; rb < r1; rb += 8) {
        int4 a[2], b[2];
        float sv[2];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const unsigned r = (unsigned)min(rb + 4 * u, r1 - 1);
            a[u] = wb[r * (unsigned)runs * 2];
            b[u] = wb[r * (unsigned)runs * 2 + 1];
            sv[u] = __half2float(sb[r * (unsigned)runs]);
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int w[8] = {a[u].x, a[u].y, a[u].z, a[u].w, b[u].x, b[u].y, b[u].z, b[u].w};
            float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s0 = __fmaf_rn(LT_QF(w[i], 0), x[4 * i], s0); s0 = __fmaf_rn(LT_QF(w[i], 1), x[4 * i + 1], s0);
                s0 = __fmaf_rn(LT_QF(w[i], 2), x[4 * i + 2], s0); s0 = __fmaf_rn(LT_QF(w[i], 3), x[4 * i + 3], s0);
                s1 = __fmaf_rn(LT_QF(w[4 + i], 0), x[16 + 4 * i], s1); s1 = __fmaf_rn(LT_QF(w[4 + i], 1), x[17 + 4 * i], s1);
                s1 = __fmaf_rn(LT_QF(w[4 + i], 2), x[18 + 4 * i], s1); s1 = __fmaf_rn(LT_QF(w[4 + i], 3), x[19 + 4 * i], s1);
            }
            float v = sv[u] * d * (s0 + s1);
            v = warp_sum(v);
            if (lane == 0 && rb + 4 * u < r1) po[rb + 4 * u] = v;
        }
    }
}

/* y[slot * ys + r] (= or +=) sum over the chunks of part[slot][chunk][r] */
extern "C" __global__ void lt_mvk_sum(const float* __restrict__ part, int nch, int rows, float* y, int ys, int acc) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x, slot = blockIdx.y;
    if (r >= rows) return;
    float s = 0.0f;
    for (int c = 0; c < nch; c++) s += part[((long long)slot * nch + c) * rows + r];
    float* o = y + (long long)slot * ys + r;
    *o = acc ? *o + s : s;
}

/* MoE down for one token: x[r] += sum_s gw[s] * (sum_c p1[s][c][r]) + sum_c p2[c][r] */
extern "C" __global__ void lt_mvk_down_sum(const float* __restrict__ p1, int nslot, int nc1,
                                           const float* __restrict__ gw, const float* __restrict__ p2, int nc2,
                                           int rows, float* x) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float tot = 0.0f;
    for (int s = 0; s < nslot; s++) {
        float a = 0.0f;
        for (int c = 0; c < nc1; c++) a += p1[((long long)s * nc1 + c) * rows + r];
        tot += gw[s] * a;
    }
    float b = 0.0f;
    for (int c = 0; c < nc2; c++) b += p2[(long long)c * rows + r];
    x[r] += tot + b;
}

/* prompt: Y = W X in fp32 for Q2_0 W (rows x cols) and T quantized inputs; the grouped (MoE) form as lt_mm_q2.
   Tile 128 rows x 16 TN tokens, 256 threads; thread (ty, tx) computes rows 8 ty .. 8 ty + 7 for tokens TN tx ..
   TN tx + TN - 1. One run (32 columns) per step: the digits c - 1 and the inputs go into shared memory as floats,
   k-major (16 KB + 8 KB at most, so that two blocks fit the 48 KB of a Kepler SMX), while the next run's words wait
   in registers. A run's products sum exactly; then they are scaled (the scales of a thread's rows and tokens come
   from global memory) and added: the same numbers as the int8 paths. TN = 4 (64 tokens) or 2 (32: MoE groups). */
template <int TN>
__device__ __forceinline__ void lt_mmk_t(const u32* const* __restrict__ codes, const __half* const* __restrict__ sc,
        int rows, int cols, const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow,
        int T, const int* __restrict__ xq, const float* __restrict__ dx, float* y, int ldy, int acc) {
    const int NT = 16 * TN;                              /* tokens per block */
    const int NB = 8 * NT / 256;                         /* input words per thread and run */
    __shared__ __align__(16) float As[32][128];
    __shared__ __align__(16) float Bs[32][NT];
    const int e = blockIdx.z;
    const int n = cnt ? cnt[e] : T;
    const int c0 = blockIdx.y * NT;
    if (c0 >= n) return;
    const int o = cnt ? off[e] : 0;
    const int r0 = blockIdx.x * 128;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int runs = cols >> 5;
    const u32* cw = codes[e];
    const __half* cs = sc[e];
    /* this thread's loads: code word aw of row ar, and input words bw .. bw + NB - 1 of token bt */
    const int ar = tid >> 1, aw = tid & 1, arow = r0 + ar;
    const int bt = tid / (256 / NT), bw = (tid % (256 / NT)) * NB;
    const int bxr = c0 + bt < n ? (xrow ? xrow[o + c0 + bt] : o + c0 + bt) : -1;
    int txr[TN];                                         /* the input rows of this thread's own tokens */
#pragma unroll
    for (int j = 0; j < TN; j++) {
        const int c = c0 + TN * tx + j;
        txr[j] = c < n ? (xrow ? xrow[o + c] : o + c) : -1;
    }
    u32 wa = arow < rows ? cw[(long long)arow * (cols >> 4) + aw] : 0x55555555u;
    int wb[NB];
#pragma unroll
    for (int i = 0; i < NB; i++) wb[i] = bxr >= 0 ? xq[(long long)bxr * runs * 8 + bw + i] : 0;
    float a[8][TN];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) a[i][j] = 0.0f;
    for (int h = 0; h < runs; h++) {
        /* the fetched run into shared memory (digit k = 16 word + 4 shift + byte: the lane order of the inputs) */
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const u32 m = (wa >> (2 * i)) & LT_M;
#pragma unroll
            for (int j = 0; j < 4; j++) As[16 * aw + 4 * i + j][ar] = (float)((m >> (8 * j)) & 255u) - 1.0f;
        }
#pragma unroll
        for (int i = 0; i < NB; i++) {
            const int w = wb[i], k = 4 * (bw + i);
            Bs[k][bt] = LT_QF(w, 0); Bs[k + 1][bt] = LT_QF(w, 1); Bs[k + 2][bt] = LT_QF(w, 2); Bs[k + 3][bt] = LT_QF(w, 3);
        }
        __syncthreads();
        if (h + 1 < runs) {                              /* the next run loads while this one is computed */
            wa = arow < rows ? cw[(long long)arow * (cols >> 4) + 2 * (h + 1) + aw] : 0x55555555u;
#pragma unroll
            for (int i = 0; i < NB; i++) wb[i] = bxr >= 0 ? xq[((long long)bxr * runs + h + 1) * 8 + bw + i] : 0;
        }
        float ad[8], bd[TN];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = r0 + 8 * ty + i;
            ad[i] = r < rows ? __half2float(cs[(long long)r * (cols >> 6) + (h >> 1)]) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < TN; j++) bd[j] = txr[j] >= 0 ? dx[(long long)txr[j] * runs + h] : 0.0f;
        float s[8][TN];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) s[i][j] = 0.0f;
#pragma unroll 4
        for (int k = 0; k < 32; k++) {
            const float4 a0 = *reinterpret_cast<const float4*>(&As[k][8 * ty]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[k][8 * ty + 4]);
            const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float bv[TN];
            if (TN == 4) {
                const float4 b = *reinterpret_cast<const float4*>(&Bs[k][4 * tx]);
                bv[0] = b.x; bv[1 % TN] = b.y; bv[2 % TN] = b.z; bv[3 % TN] = b.w;
            } else {
                const float2 b = *reinterpret_cast<const float2*>(&Bs[k][2 * tx]);
                bv[0] = b.x; bv[1 % TN] = b.y;
            }
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) s[i][j] = __fmaf_rn(av[i], bv[j], s[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) a[i][j] += ad[i] * (bd[j] * s[i][j]);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = r0 + 8 * ty + i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int c = c0 + TN * tx + j;
            if (c >= n) continue;
            float* p = y + (long long)(o + c) * ldy + r;
            *p = acc ? *p + a[i][j] : a[i][j];
        }
    }
}

#define LT_MMK_ARGS const u32* const* __restrict__ codes, const __half* const* __restrict__ sc, int rows, int cols, \
    const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow, int T, \
    const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy, int acc
extern "C" __global__ void __launch_bounds__(256) lt_mmk_q2(LT_MMK_ARGS) {
    lt_mmk_t<4>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, y, ldy, acc); }
extern "C" __global__ void __launch_bounds__(256) lt_mmk32_q2(LT_MMK_ARGS) {
    lt_mmk_t<2>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, y, ldy, acc); }

/* prompt on Kepler and the P100, scales folded: Y = W X with the weight tile holding (c - 1) * d_w (exact) and the
   input tile holding q * dx (one fp32 rounding per input value), so the inner loop is a plain fp32 product with no
   per-run step. Numbers: as a dequantized fp32 product (they differ from the exact run sums by fp32 rounding).
   Thread tile RM rows x TN tokens; block 16 RM rows x 16 TN tokens, 256 threads; one run (32 columns) per step, the
   next run's words in registers. Dense: RM = TN = 8 (128 x 128); MoE groups: RM = 16, TN = 2 (256 x 32). */
template <int RM, int TN>
__device__ __forceinline__ void lt_mmf_t(const u32* const* __restrict__ codes, const __half* const* __restrict__ sc,
        int rows, int cols, const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow,
        int T, const int* __restrict__ xq, const float* __restrict__ dx, float* y, int ldy, int acc) {
    const int BR = 16 * RM, NT = 16 * TN;
    const int NA = 2 * BR / 256;                         /* code words per thread and run */
    const int NBW = 8 * NT / 256;                        /* input words per thread and run */
    __shared__ __align__(16) float As[32][BR];
    __shared__ __align__(16) float Bs[32][NT];
    const int e = blockIdx.z;
    const int n = cnt ? cnt[e] : T;
    const int c0 = blockIdx.y * NT;
    if (c0 >= n) return;
    const int o = cnt ? off[e] : 0;
    const int r0 = blockIdx.x * BR;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int runs = cols >> 5;
    const u32* cw = codes[e];
    const __half* cs = sc[e];
    /* loads: code word aw of rows ar + 128 i; input words bw .. bw + NBW - 1 of token bt */
    const int ar = tid >> 1, aw = tid & 1;
    const int bt = tid / (256 / NT), bw = (tid % (256 / NT)) * NBW;
    const int bxr = c0 + bt < n ? (xrow ? xrow[o + c0 + bt] : o + c0 + bt) : -1;
    u32 wa[NA];
    float da[NA];
    int wb[NBW];
    float db;
#define LT_FETCHF(h) do { \
        for (int i_ = 0; i_ < NA; i_++) { \
            const int r_ = r0 + ar + 128 * i_; \
            wa[i_] = r_ < rows ? cw[(long long)r_ * (cols >> 4) + 2 * (h) + aw] : 0x55555555u; \
            da[i_] = r_ < rows ? __half2float(cs[(long long)r_ * (cols >> 6) + ((h) >> 1)]) : 0.0f; \
        } \
        for (int i_ = 0; i_ < NBW; i_++) wb[i_] = bxr >= 0 ? xq[((long long)bxr * runs + (h)) * 8 + bw + i_] : 0; \
        db = bxr >= 0 ? dx[(long long)bxr * runs + (h)] : 0.0f; \
    } while (0)
    LT_FETCHF(0);
    float a[RM][TN];
#pragma unroll
    for (int i = 0; i < RM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) a[i][j] = 0.0f;
    for (int h = 0; h < runs; h++) {
#pragma unroll
        for (int i = 0; i < NA; i++) {
#pragma unroll
            for (int s4 = 0; s4 < 4; s4++) {
                const u32 m = (wa[i] >> (2 * s4)) & LT_M;
#pragma unroll
                for (int b = 0; b < 4; b++)
                    As[16 * aw + 4 * s4 + b][ar + 128 * i] = ((float)((m >> (8 * b)) & 255u) - 1.0f) * da[i];
            }
        }
#pragma unroll
        for (int i = 0; i < NBW; i++) {
            const int w = wb[i], k = 4 * (bw + i);
            Bs[k][bt] = LT_QF(w, 0) * db; Bs[k + 1][bt] = LT_QF(w, 1) * db;
            Bs[k + 2][bt] = LT_QF(w, 2) * db; Bs[k + 3][bt] = LT_QF(w, 3) * db;
        }
        __syncthreads();
        if (h + 1 < runs) LT_FETCHF(h + 1);
#pragma unroll 4
        for (int k = 0; k < 32; k++) {
            float av[RM], bv[TN];
#pragma unroll
            for (int i = 0; i < RM; i += 4) {
                const float4 t4 = *reinterpret_cast<const float4*>(&As[k][RM * ty + i]);
                av[i] = t4.x; av[i + 1] = t4.y; av[i + 2] = t4.z; av[i + 3] = t4.w;
            }
            if (TN >= 4) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    const float4 t4 = *reinterpret_cast<const float4*>(&Bs[k][TN * tx + j]);
                    bv[j] = t4.x; bv[(j + 1) % TN] = t4.y; bv[(j + 2) % TN] = t4.z; bv[(j + 3) % TN] = t4.w;
                }
            } else {
                const float2 t2 = *reinterpret_cast<const float2*>(&Bs[k][TN * tx]);
                bv[0] = t2.x; bv[1 % TN] = t2.y;
            }
#pragma unroll
            for (int i = 0; i < RM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) a[i][j] = __fmaf_rn(av[i], bv[j], a[i][j]);
        }
        __syncthreads();
    }
#undef LT_FETCHF
#pragma unroll
    for (int i = 0; i < RM; i++) {
        const int r = r0 + RM * ty + i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int c = c0 + TN * tx + j;
            if (c >= n) continue;
            float* p = y + (long long)(o + c) * ldy + r;
            *p = acc ? *p + a[i][j] : a[i][j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) lt_mmf_q2(LT_MMK_ARGS) {
    lt_mmf_t<8, 8>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, y, ldy, acc); }
extern "C" __global__ void __launch_bounds__(256) lt_mmf32_q2(LT_MMK_ARGS) {
    lt_mmf_t<16, 2>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, y, ldy, acc); }

/* Kepler and P100 prompts, dense matrices: rows [r0, r0 + nr) of a Q2_0 matrix as fp32 (row-major, natural column
   order) for cublasSgemm, and the quantized inputs of T tokens as fp32 in the natural column order. */
extern "C" __global__ void lt_kdeq_q2(const u32* __restrict__ codes, const __half* __restrict__ sc, int cols, int r0,
                                      int nr, float* out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;      /* one code word (16 weights) */
    const int wpr = cols >> 4;
    if (i >= (long long)nr * wpr) return;
    const int r = (int)(i / wpr), w = (int)(i % wpr);
    const u32 c = codes[(long long)(r0 + r) * wpr + w];
    const float d = __half2float(sc[(long long)(r0 + r) * (cols >> 6) + (w >> 2)]);
    float* o = out + (long long)r * cols + 16 * w;
#pragma unroll
    for (int k = 0; k < 16; k++) o[k] = ((float)((c >> (2 * k)) & 3u) - 1.0f) * d;
}

extern "C" __global__ void lt_kx2f(const int* __restrict__ xq, const float* __restrict__ dx, int T, int cols, float* out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;      /* one input value */
    if (i >= (long long)T * cols) return;
    const int t = (int)(i / cols), p = (int)(i % cols);
    const int run = p >> 5, g = (p >> 4) & 1, r = p & 15;                     /* value 4 j + k of group g */
    const int w = xq[((long long)t * (cols >> 5) + run) * 8 + 4 * g + (r & 3)];
    out[i] = (float)((signed char)(w >> (8 * (r >> 2)))) * dx[(long long)t * (cols >> 5) + run];
}

/* MoE groups on Kepler and the P100 as a decode-style product over 4 tokens at a time (the groups hold few tokens):
   grid (ceil(rows / 16), ngroups), 128 threads; warp w takes rows 4 w .. 4 w + 3 of the block. Group g is
   (expert glist[2 g], first list entry glist[2 g + 1]): up to 4 entries of that expert's range [off[e], off[e] +
   cnt[e]). Lane l owns run 32 c + l of each chunk c: its 32 inputs for the 4 tokens stay in registers, and each code
   byte becomes a float once (PRMT) for 4 FFMAs. The run sums are exact (as lt_mvk_q2); the 32 lanes' sums are added
   through shared memory. Output row of entry p: p (y row stride ldy). */
extern "C" __global__ void __launch_bounds__(128) lt_mvkg_q2(const u32* const* __restrict__ codes,
        const __half* const* __restrict__ sc, int rows, int cols, const int* __restrict__ glist,
        const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow,
        const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy) {
    __shared__ float red[4][16][33];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int e = glist[2 * blockIdx.y], j0 = glist[2 * blockIdx.y + 1];
    const int o = off[e] + j0, nt = min(4, cnt[e] - j0);
    const int runs = cols >> 5, nb = cols >> 6, nch = (runs + 31) >> 5;
    int xr[4];
#pragma unroll
    for (int t = 0; t < 4; t++) {
        const int p = o + min(t, nt - 1);
        xr[t] = xrow ? xrow[p] : p;
    }
    const uint2* wb = reinterpret_cast<const uint2*>(codes[e]);
    const unsigned short* sb = reinterpret_cast<const unsigned short*>(sc[e]);
    const int r0 = blockIdx.x * 16 + wp * 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int t = 0; t < 4; t++) acc[i][t] = 0.0f;
    for (int c = 0; c < nch; c++) {
        const int h = c * 32 + lane;
        const bool on = h < runs;
        const int hc = on ? h : runs - 1;
        float x[4][32], dd[4], sq[4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            const int xi = xr[t] * runs + hc;
            lt_kx(xq + xi * 8, x[t]);
            dd[t] = on && t < nt ? dx[xi] : 0.0f;
            sq[t] = 129.0f * (float)sx[xi];
        }
        uint2 wn[4];
        unsigned short sn[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {                    /* the four rows' loads go out together */
            const unsigned r = (unsigned)min(r0 + i, rows - 1);
            wn[i] = __ldg(wb + r * (unsigned)runs + hc);
            sn[i] = __ldg(sb + r * (unsigned)nb + (hc >> 1));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint2 w = wn[i];
            const float ds = __half2float(__ushort_as_half(sn[i]));
            float s0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, s1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int q = 0; q < 4; q++) {
                const u32 m = (w.x >> (2 * q)) & LT_M, mm = (w.y >> (2 * q)) & LT_M;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const float f0 = LT_KF(m, b), f1 = LT_KF(mm, b);
#pragma unroll
                    for (int t = 0; t < 4; t++) {
                        s0[t] = __fmaf_rn(f0, x[t][4 * q + b], s0[t]);
                        s1[t] = __fmaf_rn(f1, x[t][16 + 4 * q + b], s1[t]);
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < 4; t++) acc[i][t] += ds * dd[t] * ((s0[t] + s1[t]) * (1.0f / 65536.0f) - sq[t]);
        }
    }
    /* add the lanes: lane l < 16 gives value l = (row l / 4, token l % 4) of its warp */
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int t = 0; t < 4; t++) red[wp][4 * i + t][lane] = acc[i][t];
    __syncwarp();
    if (lane < 16) {
        float v = 0.0f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) v += red[wp][lane][k];
        const int r = r0 + (lane >> 2), t = lane & 3;
        if (r < rows && t < nt) y[(long long)(o + t) * ldy + r] = v;
    }
}
