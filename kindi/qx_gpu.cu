/* qx_gpu.cu - Qwen3.8-Flash-Next (GGUF qwen4exp) on a card: the routed experts of the five expert formats, the
   dense matrix-vector products of every weight type of the file, the activation quantization and the Gated
   DeltaNet recurrence. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.

   Routed experts: the same arithmetic as kindi/qx_cpu.c, so a (token, expert) value row has the same bits on a card
   and on the CPU, and the hot experts of a card and the CPU experts combine into the same result:
   - activations: per 32 values dx = fp16(amax / 127), q = rint(x * 127 / amax) (IEEE division, no fast math);
   - per 32-value block, 8 lanes of 4 values: L = dp4a(codes, q) (exact), A_i = A_i + (float)(L * m) * (dw * dx);
   - per row: ((A0 + A4) + (A2 + A6)) + ((A1 + A5) + (A3 + A7));
   - h = silu(gate) * up with qx_expf (the same float operations as qx_cpu.c);
   - out[t] = sum over the token's slots in order of w * value.
   Every float operation is an explicit __fmul_rn / __fadd_rn / __fdiv_rn, so nvcc cannot contract it into an FMA.
   One row of one expert is computed by 8 threads (one per lane) for up to 8 (token, slot) pairs at once.

   Dense weights: one warp per row, each thread one value of a 32-value block (dequantized with the formulas of
   kindi/qwen4exp_quant.py), fp32 accumulation for up to 16 tokens, a warp sum at the end. Not bit-matched to the
   CPU reference (its PyTorch fp32 sums use another order); checked against it with a tolerance.

   Gated DeltaNet: one block of 128 threads per V head; thread j keeps column j of the 128 x 128 state in registers.
   V head h reads QK head h % 16 (the tile order of the GGUF file). Tokens run in order inside the kernel.

   Build: at run time with NVRTC (kindi/qx_gpu.py), so no CUDA toolkit is needed: the file includes no system
   header, and the kernels are extern "C" so the launcher finds them by name. dp4a on sm_61+, byte products before. */
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef decltype(sizeof(0)) size_t;      /* what the compiler uses (unsigned long on Linux NVRTC) */

/* cards before sm_61 (Maxwell sm_50/52, Pascal sm_60, Kepler sm_35/37) have no dp4a instruction: the same sum as
   four signed byte products (exact, so the output is the same bits as on the dp4a cards) */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
static __device__ __forceinline__ int xg_dp4a(int a, int b, int c) {
    return c + (a << 24 >> 24) * (b << 24 >> 24) + (a << 16 >> 24) * (b << 16 >> 24) + (a << 8 >> 24) * (b << 8 >> 24)
             + (a >> 24) * (b >> 24);
}
#define __dp4a xg_dp4a
#endif

#include "nbq_grids.h"
#include "moeaccel.cuh"

static __device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
static __device__ __forceinline__ float f2h2f(float x) {       /* round to fp16 (nearest even) and back */
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h2f(h);
}

enum { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ3_XXS = 18, T_IQ4_NL = 20,
       T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_BF16 = 30, T_Q2_0 = 42 };
#define PB 8
#define DTOK 16

__constant__ int8_t c_lv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static __device__ __forceinline__ float f16(const uint8_t* p) { return h2f((unsigned short)(p[0] | (p[1] << 8))); }
static __device__ __forceinline__ uint32_t u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static __device__ __forceinline__ int pack4(uint32_t mag, int sg) {     /* 4 magnitude bytes, 4 sign bits */
    uint32_t out = 0;
    for (int k = 0; k < 4; k++) {
        int v = (int)((mag >> (8 * k)) & 255);
        if ((sg >> k) & 1) v = -v;
        out |= (uint32_t)(uint8_t)(int8_t)v << (8 * k);
    }
    return (int)out;
}

/* bytes and values of a block of the formats */
static __device__ int blk_vals(int ty) {
    switch (ty) {
    case T_F32: case T_F16: case T_BF16: return 1;
    case T_Q8_0: case T_IQ4_NL: return 32;
    case T_Q2_0: return 64;
    default: return 256;
    }
}
static __device__ int blk_bytes(int ty) {
    switch (ty) {
    case T_F32: return 4; case T_F16: case T_BF16: return 2;
    case T_Q8_0: return 34; case T_IQ4_NL: return 18; case T_Q2_0: return 18;
    case T_Q4_K: return 144; case T_Q5_K: return 176; case T_Q6_K: return 210;
    case T_IQ3_XXS: return 98; case T_IQ3_S: return 110; case T_IQ2_S: return 82; case T_IQ4_XS: return 136;
    }
    return 0;
}

/* the codes of values 4i .. 4i + 3 of 32-value sub-block j of block b (expert formats), with m and dw (qx_cpu.c) */
static __device__ __forceinline__ int codes4(int ty, const uint8_t* b, int j, int i, int* m, float* dw) {
    switch (ty) {
    case T_IQ2_S: {
        const int l = i >> 1, qh = b[66 + j], sc = b[74 + j];
        const uint64_t g = nbq_iq2s_grid[b[2 + 4 * j + l] | (((qh >> (2 * l)) & 3) << 8)];
        *m = i < 4 ? 2 * (sc & 15) + 1 : 2 * (sc >> 4) + 1;
        *dw = __fmul_rn(f16(b), 0.125f);
        return pack4((uint32_t)(g >> (32 * (i & 1))), b[34 + 4 * j + l] >> (4 * (i & 1)));
    }
    case T_IQ3_XXS: {
        const uint32_t aux = u32(b + 66 + 4 * j);
        const int s7 = (aux >> (7 * (i >> 1))) & 127, sg = s7 | ((__popc(s7) & 1) << 7);
        *m = 2 * (int)(aux >> 28) + 1;
        *dw = __fmul_rn(f16(b), 0.25f);
        return pack4(nbq_iq3xxs_grid[b[2 + 8 * j + i]], sg >> (4 * (i & 1)));
    }
    case T_IQ3_S: {
        const int idx = b[2 + 8 * j + i] | (((b[66 + j] >> i) & 1) << 8);
        *m = 2 * ((b[106 + (j >> 1)] >> (4 * (j & 1))) & 15) + 1;
        *dw = f16(b);
        return pack4(nbq_iq3s_grid[idx], b[74 + 4 * j + (i >> 1)] >> (4 * (i & 1)));
    }
    case T_IQ4_XS: case T_IQ4_NL: {
        const uint8_t* q = ty == T_IQ4_XS ? b + 8 + 16 * j : b + 2;
        uint32_t out = 0;
        for (int k = 0; k < 4; k++) {
            const int byte = q[4 * (i & 3) + k], v = i < 4 ? byte & 15 : byte >> 4;
            out |= (uint32_t)(uint8_t)c_lv4[v] << (8 * k);
        }
        if (ty == T_IQ4_XS) {
            const int sh = b[2] | (b[3] << 8);
            *m = (((b[4 + j / 2] >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4)) - 32;
        } else
            *m = 1;
        *dw = f16(b);
        return (int)out;
    }
    default: {                                            /* Q2_0 */
        const int byte = b[2 + 8 * j + i];
        uint32_t out = 0;
        for (int k = 0; k < 4; k++) out |= (uint32_t)(uint8_t)(int8_t)(((byte >> (2 * k)) & 3) - 1) << (8 * k);
        *m = 1; *dw = f16(b);
        return (int)out;
    }
    }
}

/* qx_expf of qx_cpu.c, with the same operations */
static __device__ __forceinline__ float qx_expf(float x) {
    x = fminf(fmaxf(x, -88.f), 88.f);
    const float n = rintf(__fmul_rn(x, 1.44269504f));
    float r = __fadd_rn(x, -__fmul_rn(n, 0.693145752f));
    r = __fadd_rn(r, -__fmul_rn(n, 1.42860677e-06f));
    float p = 1.98412698e-4f;
    p = __fadd_rn(__fmul_rn(p, r), 1.38888889e-3f);
    p = __fadd_rn(__fmul_rn(p, r), 8.33333333e-3f);
    p = __fadd_rn(__fmul_rn(p, r), 4.16666667e-2f);
    p = __fadd_rn(__fmul_rn(p, r), 1.66666667e-1f);
    p = __fadd_rn(__fmul_rn(p, r), 0.5f);
    p = __fadd_rn(__fmul_rn(p, r), 1.f);
    p = __fadd_rn(__fmul_rn(p, r), 1.f);
    return ldexpf(p, (int)n);
}

/* ---------------------------------------------------------------- activations */
extern "C" __global__ void k_quant(const float* x, long long nblk, int8_t* q, float* d) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float* xb = x + 32 * b;
    float amax = 0.f;
    for (int k = 0; k < 32; k++) { const float a = fabsf(xb[k]); if (a > amax) amax = a; }
    d[b] = f2h2f(__fdiv_rn(amax, 127.f));
    const float id = amax != 0.f ? __fdiv_rn(127.f, amax) : 0.f;
    for (int k = 0; k < 32; k++) q[32 * b + k] = (int8_t)rintf(__fmul_rn(xb[k], id));
}

/* the expert input x in blocks of 256 values with one scale (qx_cpu.c qx_quant256) */
/* blocks of ab values (64: the down input on the DN64 path, item G) with k_quant256's steps */
extern "C" __global__ void k_quant_ab(const float* x, long long nblk, int8_t* q, float* d, int ab) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float* xb = x + ab * b;
    float amax = 0.f;
    for (int k = 0; k < ab; k++) { const float a = fabsf(xb[k]); if (a > amax) amax = a; }
    d[b] = f2h2f(__fdiv_rn(amax, 127.f));
    const float id = amax != 0.f ? __fdiv_rn(127.f, amax) : 0.f;
    for (int k = 0; k < ab; k++) q[ab * b + k] = (int8_t)rintf(__fmul_rn(xb[k], id));
}
/* k_quant_ab / k_quant256 with one warp per block of ab values (ab = 32 k, k <= 8): lane i takes values i, i + 32, ...
   (coalesced); the same amax (a max is order-free), the same d and the same q, so the same bits */
extern "C" __global__ void k_quant_w(const float* x, long long nblk, int8_t* q, float* d, int ab) {
    const long long b = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31, per = ab >> 5;
    if (b >= nblk) return;
    const float* xb = x + ab * b;
    float v[8];
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (k < per) { v[k] = xb[lane + 32 * k]; const float a = fabsf(v[k]); if (a > amax) amax = a; }
#pragma unroll
    for (int o = 16; o; o >>= 1) { const float a = __shfl_xor_sync(0xffffffffu, amax, o); if (a > amax) amax = a; }
    if (lane == 0) d[b] = f2h2f(__fdiv_rn(amax, 127.f));
    const float id = amax != 0.f ? __fdiv_rn(127.f, amax) : 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (k < per) q[ab * b + lane + 32 * k] = (int8_t)rintf(__fmul_rn(v[k], id));
}
extern "C" __global__ void k_quant256(const float* x, long long nblk, int8_t* q, float* d) {
    const long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    const float* xb = x + 256 * b;
    float amax = 0.f;
    for (int k = 0; k < 256; k++) { const float a = fabsf(xb[k]); if (a > amax) amax = a; }
    d[b] = f2h2f(__fdiv_rn(amax, 127.f));
    const float id = amax != 0.f ? __fdiv_rn(127.f, amax) : 0.f;
    for (int k = 0; k < 256; k++) q[256 * b + k] = (int8_t)rintf(__fmul_rn(xb[k], id));
}

/* ---------------------------------------------------------------- routed experts */
/* one row of one expert for np <= PB pairs, by the 8 threads of a group (lane = threadIdx.x & 7); lane 0 gets the
   results */
static __device__ __forceinline__ void row_dot(int ty, const uint8_t* row, int ncols, int np, const int8_t* const* xq,
                                               const float* const* xd, float* res) {
    const int lane = threadIdx.x & 7, per = blk_vals(ty), bytes = blk_bytes(ty), sub = per / 32;
    float A[PB];
    for (int p = 0; p < PB; p++) A[p] = 0.f;
    for (int blk = 0; blk < ncols / 32; blk++) {
        int m; float dw;
        const int c = codes4(ty, row + (size_t)(blk / sub) * bytes, blk % sub, lane, &m, &dw);
        for (int p = 0; p < np; p++) {
            const int L = __dp4a(c, *(const int*)(xq[p] + 32 * blk + 4 * lane), 0);
            const float s = __fmul_rn(dw, xd[p][blk]);
            A[p] = __fadd_rn(A[p], __fmul_rn((float)(L * m), s));
        }
    }
    const unsigned mask = __activemask();
    const int base = (threadIdx.x & 31) & ~7;
    for (int p = 0; p < np; p++) {
        float v[8];
        for (int i = 0; i < 8; i++) v[i] = __shfl_sync(mask, A[p], base + i);
        res[p] = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[4]), __fadd_rn(v[2], v[6])),
                           __fadd_rn(__fadd_rn(v[1], v[5]), __fadd_rn(v[3], v[7])));
    }
}

/* two rows (r0, r1: same format and width) for one activation vector, interleaved block by block: the same
   operations per row as row_dot, so the same bits, with two independent chains in flight */
static __device__ __forceinline__ void row_dot2(int ty, const uint8_t* row0, const uint8_t* row1, int ncols,
                                                const int8_t* xq, const float* xd, float* res0, float* res1) {
    const int lane = threadIdx.x & 7, per = blk_vals(ty), bytes = blk_bytes(ty), sub = per / 32;
    float A0 = 0.f, A1 = 0.f;
    for (int blk = 0; blk < ncols / 32; blk++) {
        int m0, m1; float dw0, dw1;
        const int c0 = codes4(ty, row0 + (size_t)(blk / sub) * bytes, blk % sub, lane, &m0, &dw0);
        const int c1 = codes4(ty, row1 + (size_t)(blk / sub) * bytes, blk % sub, lane, &m1, &dw1);
        const int xv = *(const int*)(xq + 32 * blk + 4 * lane);
        const float dx = xd[blk];
        const int L0 = __dp4a(c0, xv, 0), L1 = __dp4a(c1, xv, 0);
        A0 = __fadd_rn(A0, __fmul_rn((float)(L0 * m0), __fmul_rn(dw0, dx)));
        A1 = __fadd_rn(A1, __fmul_rn((float)(L1 * m1), __fmul_rn(dw1, dx)));
    }
    const unsigned mask = __activemask();
    const int base = (threadIdx.x & 31) & ~7;
    float v[8];
    for (int i = 0; i < 8; i++) v[i] = __shfl_sync(mask, A0, base + i);
    *res0 = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[4]), __fadd_rn(v[2], v[6])),
                      __fadd_rn(__fadd_rn(v[1], v[5]), __fadd_rn(v[3], v[7])));
    for (int i = 0; i < 8; i++) v[i] = __shfl_sync(mask, A1, base + i);
    *res1 = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[4]), __fadd_rn(v[2], v[6])),
                      __fadd_rn(__fadd_rn(v[1], v[5]), __fadd_rn(v[3], v[7])));
}

/* gate/up rows on 256-value activation blocks (qx_cpu.c qx_row_dot256_n): per weight block, the integer sum of
   its sub-blocks (each lane's dp4a products times m, then the 8 lanes summed: exact, any order), then
   A = A + (float)I * (dw * xs[block * per / 256]). Every lane ends with the same A. */
static __device__ __forceinline__ int lane_isum(int v) {
    v += __shfl_xor_sync(__activemask(), v, 1);
    v += __shfl_xor_sync(__activemask(), v, 2);
    v += __shfl_xor_sync(__activemask(), v, 4);
    return v;
}
static __device__ __forceinline__ void row_dot256(int ty, const uint8_t* row, int ncols, int np, const int8_t* const* xq,
                                                  const float* const* xs, float* res, int ab = 256) {
    const int lane = threadIdx.x & 7, per = blk_vals(ty), bytes = blk_bytes(ty), sub = per / 32;
    float A[PB];
    for (int p = 0; p < PB; p++) A[p] = 0.f;
    for (int bk = 0; bk < ncols / per; bk++) {
        int I[PB];
        for (int p = 0; p < PB; p++) I[p] = 0;
        float dw = 0.f;
        for (int j = 0; j < sub; j++) {
            int m;
            const int c = codes4(ty, row + (size_t)bk * bytes, j, lane, &m, &dw);
            const int off = 32 * (bk * sub + j) + 4 * lane;
            for (int p = 0; p < np; p++) I[p] += __dp4a(c, *(const int*)(xq[p] + off), 0) * m;
        }
        const int xb = bk * per / ab;
        for (int p = 0; p < np; p++)
            A[p] = __fadd_rn(A[p], __fmul_rn((float)lane_isum(I[p]), __fmul_rn(dw, xs[p][xb])));
    }
    for (int p = 0; p < np; p++) res[p] = A[p];
}
/* two rows for one activation vector (the decode slots): the same steps per row */
static __device__ __forceinline__ void row_dot2_256(int ty, const uint8_t* row0, const uint8_t* row1, int ncols,
                                                    const int8_t* xq, const float* xs, float* res0, float* res1,
                                                    int ab = 256) {
    const int lane = threadIdx.x & 7, per = blk_vals(ty), bytes = blk_bytes(ty), sub = per / 32;
    float A0 = 0.f, A1 = 0.f;
    for (int bk = 0; bk < ncols / per; bk++) {
        int I0 = 0, I1 = 0;
        float dw0 = 0.f, dw1 = 0.f;
        for (int j = 0; j < sub; j++) {
            int m0, m1;
            const int c0 = codes4(ty, row0 + (size_t)bk * bytes, j, lane, &m0, &dw0);
            const int c1 = codes4(ty, row1 + (size_t)bk * bytes, j, lane, &m1, &dw1);
            const int xv = *(const int*)(xq + 32 * (bk * sub + j) + 4 * lane);
            I0 += __dp4a(c0, xv, 0) * m0;
            I1 += __dp4a(c1, xv, 0) * m1;
        }
        const float dx = xs[bk * per / ab];
        A0 = __fadd_rn(A0, __fmul_rn((float)lane_isum(I0), __fmul_rn(dw0, dx)));
        A1 = __fadd_rn(A1, __fmul_rn((float)lane_isum(I1), __fmul_rn(dw1, dx)));
    }
    *res0 = A0;
    *res1 = A1;
}

/* the pairs of one call: union expert u has pairs [pstart[u], pstart[u + 1]); pair p is (ptok[p], pslot[p]);
   mats[3u + 0/1/2] = gate/up/down of the expert (row-major blocks as in the file) */
extern "C" __global__ void k_gate_up(int ty, int K, int F, const uint8_t* const* mats, const int* pstart, const int* ptok,
                          const int8_t* xq, const float* xd, float* h) {
    const int u = blockIdx.y, r = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (r >= F) return;
    const long long rb = (long long)(K / blk_vals(ty)) * blk_bytes(ty);
    const uint8_t* gr = mats[3 * u] + r * rb;
    const uint8_t* ur = mats[3 * u + 1] + r * rb;
    for (int p0 = pstart[u]; p0 < pstart[u + 1]; p0 += PB) {
        const int np = min(PB, pstart[u + 1] - p0);
        const int8_t* q[PB]; const float* d[PB];
        for (int i = 0; i < np; i++) { const int t = ptok[p0 + i]; q[i] = xq + (size_t)t * K; d[i] = xd + (size_t)t * (K / 256); }
        float g[PB], v[PB];
        row_dot256(ty, gr, K, np, q, d, g);
        row_dot256(ty, ur, K, np, q, d, v);
        if ((threadIdx.x & 7) == 0)
            for (int i = 0; i < np; i++) {
                const float sl = __fdiv_rn(g[i], __fadd_rn(1.f, qx_expf(-g[i])));
                h[(size_t)(p0 + i) * F + r] = __fmul_rn(sl, v[i]);
            }
    }
}

/* down rows; the value of pair p goes to parts[(ptok[p] * nsel + pslot[p]) * K + r] */
/* dab: the down input's activation block (32: per 32-value sub-block float steps; 64: the DN64 path, integer sums
   per weight block and one float step, as gate/up on 256) */
extern "C" __global__ void k_down(int ty, int K, int F, int nsel, const uint8_t* const* mats, const int* pstart, const int* ptok,
                       const int* pslot, const int8_t* hq, const float* hd, float* parts, int dab) {
    const int u = blockIdx.y, r = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    if (r >= K) return;
    const long long rb = (long long)(F / blk_vals(ty)) * blk_bytes(ty);
    const uint8_t* dr = mats[3 * u + 2] + r * rb;
    for (int p0 = pstart[u]; p0 < pstart[u + 1]; p0 += PB) {
        const int np = min(PB, pstart[u + 1] - p0);
        const int8_t* q[PB]; const float* d[PB];
        for (int i = 0; i < np; i++) { q[i] = hq + (size_t)(p0 + i) * F; d[i] = hd + (size_t)(p0 + i) * (F / dab); }
        float v[PB];
        if (dab == 64) row_dot256(ty, dr, F, np, q, d, v, 64);
        else row_dot(ty, dr, F, np, q, d, v);
        if ((threadIdx.x & 7) == 0)
            for (int i = 0; i < np; i++) parts[((size_t)ptok[p0 + i] * nsel + pslot[p0 + i]) * K + r] = v[i];
    }
}

extern "C" __global__ void k_combine(int T, int nsel, int K, const float* w, const float* parts, float* out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * K) return;
    const int t = (int)(i / K), c = (int)(i % K);
    float acc = 0.f;
    for (int s = 0; s < nsel; s++) acc = __fadd_rn(acc, __fmul_rn(w[t * nsel + s], parts[((size_t)t * nsel + s) * K + c]));
    out[i] = acc;
}

/* ---------------------------------------------------------------- one-token decode, slot by slot */
/* For a decode step (one token): slot s of the router's picks reads expert idx[s]; if slotmap[idx[s]] >= 0 the
   expert is on this card, at hot + slot * rowbytes (gate | up | down); otherwise the block exits (the CPU computes
   it). No host work: the picks stay on the card. Same arithmetic as k_gate_up / k_down. */
extern "C" __global__ void k_gate_up_slots(int ty, int K, int F, const int* idx, const int* slotmap, const uint8_t* hot,
                                           long long rowbytes, const int8_t* xq, const float* xd, float* h,
                                           int tok_nsel, int min_slot) {
    /* slot s belongs to token s / tok_nsel (several tokens: the verify window); its input row is that token's */
    const int s = blockIdx.y, r = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    xq += (size_t)(s / tok_nsel) * K;
    xd += (size_t)(s / tok_nsel) * (K / 256);
    const int sl = slotmap[idx[s]];
    if (sl < min_slot || r >= F) return;                  /* min_slot >= 0: also the not-cached picks; > 0: only the
                                                             fetched experts' spare rows (the PCIe share's own pass) */
    const long long rb = (long long)(K / blk_vals(ty)) * blk_bytes(ty);
    const uint8_t* base = hot + (size_t)sl * rowbytes;
    float g, v;
    row_dot2_256(ty, base + r * rb, base + (size_t)F * rb + r * rb, K, xq, xd, &g, &v);
    if ((threadIdx.x & 7) == 0) {
        const float sl_ = __fdiv_rn(g, __fadd_rn(1.f, qx_expf(-g)));
        h[(size_t)s * F + r] = __fmul_rn(sl_, v);
    }
}
extern "C" __global__ void k_down_slots(int tyg, int ty, int K, int F, const int* idx, const int* slotmap,
                                        const uint8_t* hot, long long rowbytes, const int8_t* hq, const float* hd,
                                        float* parts, int dab, int min_slot) {
    /* each group of 8 threads does rows r and r + K / 2 together; dab as in k_down */
    const int s = blockIdx.y, r = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8, half = K / 2;
    const int sl = slotmap[idx[s]];
    if (sl < min_slot || r >= half) return;
    const long long rbg = (long long)(K / blk_vals(tyg)) * blk_bytes(tyg);
    const long long rb = (long long)(F / blk_vals(ty)) * blk_bytes(ty);
    const uint8_t* dn = hot + (size_t)sl * rowbytes + 2 * (size_t)F * rbg;
    float v0, v1;
    if (dab == 64)
        row_dot2_256(ty, dn + r * rb, dn + (size_t)(r + half) * rb, F, hq + (size_t)s * F, hd + (size_t)s * (F / 64),
                     &v0, &v1, 64);
    else
        row_dot2(ty, dn + r * rb, dn + (size_t)(r + half) * rb, F, hq + (size_t)s * F, hd + (size_t)s * (F / 32), &v0, &v1);
    if ((threadIdx.x & 7) == 0) {
        parts[(size_t)s * K + r] = v0;
        parts[(size_t)s * K + r + half] = v1;
    }
}
/* out[c] = sum over slots in order of w[s] * (the card's row if the expert is here, else the CPU's row) */
extern "C" __global__ void k_combine_sel(int nsel, int K, const float* w, const int* idx, const int* slotmap,
                                         const float* hotp, const float* coldp, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float acc = 0.f;
    for (int s = 0; s < nsel; s++) {
        const float v = slotmap[idx[s]] >= 0 ? hotp[(size_t)s * K + c] : coldp[(size_t)s * K + c];
        acc = __fadd_rn(acc, __fmul_rn(w[s], v));
    }
    out[c] = acc;
}

/* KINDI_QXG_COLDSUM: out[c] = (sum over the card's slots in order of w * value) + coldsum[c], where coldsum is the
   CPU's sum over its slots in order (one row crosses the link instead of one per CPU slot). Deterministic, but the
   bits then depend on the placement (which slots are on the card). */
extern "C" __global__ void k_combine_hot(int nsel, int K, const float* w, const int* idx, const int* slotmap,
                                         const float* hotp, const float* coldsum, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float acc = 0.f;
    for (int s = 0; s < nsel; s++)
        if (slotmap[idx[s]] >= 0) acc = __fadd_rn(acc, __fmul_rn(w[s], hotp[(size_t)s * K + c]));
    out[c] = __fadd_rn(acc, coldsum[c]);
}

/* ---------------------------------------------------------------- dense weights */
/* value v (0 .. 31) of 32-value block blk of a row, fp32 (the formulas of qwen4exp_quant.py) */
static __device__ __forceinline__ float deq(int ty, const uint8_t* row, int blk, int v) {
    switch (ty) {
    case T_F32: return ((const float*)row)[32 * blk + v];
    case T_F16: return h2f(((const uint16_t*)row)[32 * blk + v]);
    case T_BF16: return __uint_as_float((uint32_t)((const uint16_t*)row)[32 * blk + v] << 16);
    case T_Q8_0: { const uint8_t* b = row + 34 * blk; return f16(b) * (float)(int8_t)b[2 + v]; }
    case T_Q2_0: {
        const uint8_t* b = row + 18 * (blk >> 1);
        const int i = 32 * (blk & 1) + v;
        return f16(b) * (float)(((b[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
    }
    case T_Q4_K: case T_Q5_K: {
        const uint8_t* b = row + (size_t)(blk >> 3) * (ty == T_Q4_K ? 144 : 176);
        const int j = blk & 7, k = j & 3, u = j >> 2;
        const int x = b[4 + k], y = b[4 + k + 4], z = b[4 + k + 8];
        const int sc = ((u ? z : x) & 15) | ((x >> (2 * u)) & 48);
        const int mn = ((u ? (z >> 4) : y) & 15) | ((y >> (2 * u)) & 48);
        const uint8_t* qs = b + (ty == T_Q4_K ? 16 : 48) + 32 * (j >> 1);
        int q = (j & 1) ? qs[v] >> 4 : qs[v] & 15;
        if (ty == T_Q5_K) q |= ((b[16 + v] >> j) & 1) << 4;
        return f16(b) * (float)sc * (float)q - f16(b + 2) * (float)mn;
    }
    case T_Q6_K: {
        const uint8_t* b = row + (size_t)(blk >> 3) * 210;
        const int n = (blk & 7) >> 2, p = blk & 3;
        const uint8_t* ql = b + 64 * n; const uint8_t* qh = b + 128 + 32 * n;
        const int lo = (p & 1) ? ql[32 + v] : ql[v];
        const int q = ((p & 2) ? lo >> 4 : lo & 15) | (((qh[v] >> (2 * p)) & 3) << 4);
        const float s = (float)(int8_t)b[192 + 8 * n + 2 * p + (v >= 16)];
        return f16(b + 208) * s * (float)(q - 32);
    }
    default: {                                            /* the expert formats: the code times m times dw */
        const int per = blk_vals(ty), sub = per / 32;
        int m; float dw;
        const int c = codes4(ty, row + (size_t)(blk / sub) * blk_bytes(ty), blk % sub, v >> 2, &m, &dw);
        return dw * (float)m * (float)(int8_t)(c >> (8 * (v & 3)));
    }
    }
}

/* Y[t][r] = sum_c W[r][c] X[t][c], t < T <= DTOK. A block of 8 warps holds 8 / split rows; the split warps of a row
   take its 32-value blocks in turn (split 8 for matrices with few rows and long rows), and their sums are added in
   warp order, so the result does not depend on the timing. */
extern "C" __global__ void k_dense(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X, int T,
                                   float* Y, int ldy, int split) {
    __shared__ float part[8][DTOK];
    const int warp = threadIdx.x / 32, v = threadIdx.x & 31, rpb = 8 / split;
    const int r = blockIdx.x * rpb + warp / split, ws = warp % split;
    float acc[DTOK];
    for (int t = 0; t < DTOK; t++) acc[t] = 0.f;
    if (r < rows) {
        const uint8_t* row = W + (size_t)r * rb;
        for (int blk = ws; blk < cols / 32; blk += split) {
            const float w = deq(ty, row, blk, v);
            for (int t = 0; t < T; t++) acc[t] += w * X[(size_t)t * cols + 32 * blk + v];
        }
    }
    for (int t = 0; t < T; t++) {
        float x = acc[t];
        for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
        if (v == 0) part[warp][t] = x;
    }
    __syncthreads();
    if (r < rows && ws == 0 && v < T) {
        float x = 0.f;
        for (int k = 0; k < split; k++) x += part[warp + k][v];
        Y[(size_t)v * ldy + r] = x;
    }
}

/* One token: y[r] = sum_c W[r][c] x[c]. One warp per row (split > 1: that many warps per row, summed in warp
   order); lane l takes the 32-value sub-blocks l, l + 32, ... of its part of the row, decodes each one with its scale
   read once, and adds its 32 products in order; the lanes are summed by a shuffle tree. */
static __device__ __forceinline__ float sub_dot(int ty, const uint8_t* row, int blk, const float* x) {
    float acc = 0.f;
    switch (ty) {
    case T_BF16: {
        const uint16_t* w = (const uint16_t*)row + 32 * blk;
        for (int v = 0; v < 32; v++) acc += __uint_as_float((uint32_t)w[v] << 16) * x[v];
        return acc;
    }
    case T_F16: {
        const uint16_t* w = (const uint16_t*)row + 32 * blk;
        for (int v = 0; v < 32; v++) acc += h2f(w[v]) * x[v];
        return acc;
    }
    case T_F32: {
        const float* w = (const float*)row + 32 * blk;
        for (int v = 0; v < 32; v++) acc += w[v] * x[v];
        return acc;
    }
    case T_Q8_0: {
        const uint8_t* b = row + 34 * blk;
        for (int v = 0; v < 32; v++) acc += (float)(int8_t)b[2 + v] * x[v];
        return f16(b) * acc;
    }
    case T_IQ4_NL: {
        const uint8_t* b = row + 18 * blk;
        for (int v = 0; v < 16; v++) acc += (float)c_lv4[b[2 + v] & 15] * x[v] + (float)c_lv4[b[2 + v] >> 4] * x[v + 16];
        return f16(b) * acc;
    }
    case T_IQ4_XS: {
        const uint8_t* b = row + (size_t)(blk >> 3) * 136;
        const int j = blk & 7, sh = b[2] | (b[3] << 8);
        const int ls = ((b[4 + j / 2] >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4);
        const uint8_t* q = b + 8 + 16 * j;
        for (int v = 0; v < 16; v++) acc += (float)c_lv4[q[v] & 15] * x[v] + (float)c_lv4[q[v] >> 4] * x[v + 16];
        return f16(b) * (float)(ls - 32) * acc;
    }
    case T_Q4_K: case T_Q5_K: {
        const uint8_t* b = row + (size_t)(blk >> 3) * (ty == T_Q4_K ? 144 : 176);
        const int j = blk & 7, k = j & 3, u = j >> 2;
        const int xx = b[4 + k], y = b[4 + k + 4], z = b[4 + k + 8];
        const int sc = ((u ? z : xx) & 15) | ((xx >> (2 * u)) & 48);
        const int mn = ((u ? (z >> 4) : y) & 15) | ((y >> (2 * u)) & 48);
        const uint8_t* qs = b + (ty == T_Q4_K ? 16 : 48) + 32 * (j >> 1);
        float xs = 0.f;
        for (int v = 0; v < 32; v++) {
            int q = (j & 1) ? qs[v] >> 4 : qs[v] & 15;
            if (ty == T_Q5_K) q |= ((b[16 + v] >> j) & 1) << 4;
            acc += (float)q * x[v];
            xs += x[v];
        }
        return f16(b) * (float)sc * acc - f16(b + 2) * (float)mn * xs;
    }
    case T_Q6_K: {
        const uint8_t* b = row + (size_t)(blk >> 3) * 210;
        const int n = (blk & 7) >> 2, p = blk & 3;
        const uint8_t* ql = b + 64 * n + ((p & 1) ? 32 : 0);
        const uint8_t* qh = b + 128 + 32 * n;
        float a0 = 0.f, a1 = 0.f;
        for (int v = 0; v < 32; v++) {
            const int q = ((p & 2) ? ql[v] >> 4 : ql[v] & 15) | (((qh[v] >> (2 * p)) & 3) << 4);
            if (v < 16) a0 += (float)(q - 32) * x[v]; else a1 += (float)(q - 32) * x[v];
        }
        const float d = f16(b + 208);
        return d * (float)(int8_t)b[192 + 8 * n + 2 * p] * a0 + d * (float)(int8_t)b[192 + 8 * n + 2 * p + 1] * a1;
    }
    default: {
        for (int v = 0; v < 32; v++) acc += deq(ty, row, blk, v) * x[v];
        return acc;
    }
    }
}

extern "C" __global__ void k_dense1(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                    float* Y, int split) {
    __shared__ float part[8];
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, rpb = 8 / split;
    const int r = blockIdx.x * rpb + warp / split, ws = warp % split;
    float acc = 0.f;
    if (r < rows) {
        const uint8_t* row = W + (size_t)r * rb;
        for (int blk = ws * 32 + lane; blk < cols / 32; blk += 32 * split) acc += sub_dot(ty, row, blk, X + 32 * blk);
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (r < rows && ws == 0 && lane == 0) {
        float x = 0.f;
        for (int k = 0; k < split; k++) x += part[warp + k];
        Y[r] = x;
    }
}

/* 8 bytes at p (any alignment) as two words, from aligned 32-bit loads and a funnel shift (the quant blocks of the
   file are only 2-byte aligned, so vector loads are not allowed; single-byte loads cost 8 instructions). May read up
   to 3 bytes past p + 8, inside the same allocation. */
static __device__ __forceinline__ uint2 ld8(const uint8_t* p) {
    const size_t a = (size_t)p;
    const uint32_t* w = (const uint32_t*)(a & ~(size_t)3);
    const int sh = (int)(a & 3) * 8;
    const uint32_t w0 = w[0], w1 = w[1], w2 = sh ? w[2] : 0u;
    uint2 r;
    r.x = __funnelshift_r(w0, w1, sh);
    r.y = __funnelshift_r(w1, w2, sh);
    return r;
}
static __device__ __forceinline__ int byte_of(uint2 v, int k) { return (int)(((k < 4 ? v.x : v.y) >> (8 * (k & 3))) & 255); }

/* One token, coalesced: a warp step covers 8 sub-blocks of 32 values (one 256-value superblock, or 8 blocks of the
   32-value formats); lane l decodes the 8 values 8 (l & 3) .. + 7 of sub-block (l >> 2), so neighbouring lanes read
   neighbouring bytes. Returns the sum over those 8 values of w * x. */
/* small non-negative ints to float without the I2F unit (a quarter of the FMA rate on sm_86): 2^23 + q has q in
   its mantissa, so (2^23 + q) - (2^23 + off) = q - off exactly (|q|, off < 2^22): the same floats as (float)(q - off).
   KINDI_QXG_I2F=0 (XG_I2F 0): the plain conversions (A/B). */
#ifndef XG_I2F
#define XG_I2F 1
#endif
#if XG_I2F
#define QX_UF(q, off) __fsub_rn(__int_as_float(0x4B000000 | (int)(q)), 8388608.f + (float)(off))
#define QX_SF8(b) __fsub_rn(__int_as_float(0x4B000000 | (int)((b) ^ 0x80u)), 8388736.f)
#else
#define QX_UF(q, off) ((float)((int)(q) - (off)))
#define QX_SF8(b) ((float)(int8_t)(b))
#endif
/* byte k of a packed word of small ints q as the float q - off, exactly: PRMT builds 0x4B0000qq (2^23 + q), then
   one subtraction (the same value as QX_UF(q, off) / (float)(q - off)) */
static __device__ __forceinline__ float kq_f(uint32_t w, int k, float off) {
    return __fsub_rn(__int_as_float(__byte_perm(w, 0x4B000000u, 0x7440u | k)), 8388608.f + off);
}
/* the 16 IQ4 levels as int8 bytes in 4 words (xg_lv4's table) */
static __device__ __forceinline__ uint32_t kq_lv4(uint32_t nb) {
    uint32_t t[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        t[k] = (uint32_t)(uint8_t)c_lv4[4 * k] | ((uint32_t)(uint8_t)c_lv4[4 * k + 1] << 8) |
               ((uint32_t)(uint8_t)c_lv4[4 * k + 2] << 16) | ((uint32_t)(uint8_t)c_lv4[4 * k + 3] << 24);
    uint32_t u = nb & 0x07070707u;
    u |= u >> 4;
    const uint32_t sel = (u & 0xFFu) | ((u >> 8) & 0xFF00u);
    const uint32_t lo = __byte_perm(t[0], t[1], sel), hi = __byte_perm(t[2], t[3], sel);
    const uint32_t mk = ((nb >> 3) & 0x01010101u) * 255u;
    return (lo & ~mk) | (hi & mk);
}
static __device__ __forceinline__ float dot8(int ty, const uint8_t* row, int j, int part, const float* x,
                                             const float* lv) {
    const float4 xa = *(const float4*)x, xb = *(const float4*)(x + 4);
    const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
    float acc = 0.f;
    switch (ty) {
    case T_BF16: {
        const uint4 w = *(const uint4*)((const uint16_t*)row + 32 * j + 8 * part);
        const uint32_t u[4] = {w.x, w.y, w.z, w.w};
        for (int k = 0; k < 4; k++) {
            acc += __uint_as_float(u[k] << 16) * xv[2 * k];
            acc += __uint_as_float(u[k] & 0xffff0000u) * xv[2 * k + 1];
        }
        return acc;
    }
    case T_Q8_0: {
        const uint8_t* b = row + 34 * j;
        const uint2 q = ld8(b + 2 + 8 * part);
        const uint32_t q0 = q.x ^ 0x80808080u, q1 = q.y ^ 0x80808080u;      /* int8 + 128 */
#pragma unroll
        for (int k = 0; k < 4; k++) acc += kq_f(q0, k, 128.f) * xv[k];
#pragma unroll
        for (int k = 0; k < 4; k++) acc += kq_f(q1, k, 128.f) * xv[4 + k];
        return __fmul_rn(f16(b), acc);       /* not fused with the caller's add (item 9: same bits as before) */
    }
    case T_IQ4_NL: {
        const uint2 q = ld8(row + 18 * j + 2 + 8 * (part & 1));
        const int sh4 = part < 2 ? 0 : 4;
        const uint32_t v0 = kq_lv4((q.x >> sh4) & 0x0F0F0F0Fu) ^ 0x80808080u, v1 = kq_lv4((q.y >> sh4) & 0x0F0F0F0Fu) ^ 0x80808080u;
#pragma unroll
        for (int k = 0; k < 4; k++) acc += kq_f(v0, k, 128.f) * xv[k];
#pragma unroll
        for (int k = 0; k < 4; k++) acc += kq_f(v1, k, 128.f) * xv[4 + k];
        return __fmul_rn(f16(row + 18 * j), acc);
    }
    case T_IQ4_XS: {
        const uint8_t* b = row + (size_t)(j >> 3) * 136;
        const int jj = j & 7, sh = b[2] | (b[3] << 8);
        const int ls = ((b[4 + jj / 2] >> (4 * (jj & 1))) & 15) | (((sh >> (2 * jj)) & 3) << 4);
        const uint2 q = ld8(b + 8 + 16 * jj + 8 * (part & 1));
        const int sh4 = part < 2 ? 0 : 4;
        const uint32_t v0 = kq_lv4((q.x >> sh4) & 0x0F0F0F0Fu) ^ 0x80808080u, v1 = kq_lv4((q.y >> sh4) & 0x0F0F0F0Fu) ^ 0x80808080u;
#pragma unroll
        for (int k = 0; k < 4; k++) acc += kq_f(v0, k, 128.f) * xv[k];
#pragma unroll
        for (int k = 0; k < 4; k++) acc += kq_f(v1, k, 128.f) * xv[4 + k];
        return __fmul_rn(f16(b) * (float)(ls - 32), acc);
    }
    case T_Q4_K: case T_Q5_K: {
        const uint8_t* b = row + (size_t)(j >> 3) * (ty == T_Q4_K ? 144 : 176);
        const int jj = j & 7, k4 = jj & 3, u = jj >> 2;
        const int xx = b[4 + k4], y = b[4 + k4 + 4], z = b[4 + k4 + 8];
        const int sc = ((u ? z : xx) & 15) | ((xx >> (2 * u)) & 48);
        const int mn = ((u ? (z >> 4) : y) & 15) | ((y >> (2 * u)) & 48);
        const uint2 qv = ld8(b + (ty == T_Q4_K ? 16 : 48) + 32 * (jj >> 1) + 8 * part);
        const uint2 hv = ty == T_Q5_K ? ld8(b + 16 + 8 * part) : make_uint2(0u, 0u);
        const int sh4 = (jj & 1) ? 4 : 0;
        float xs = 0.f;
        uint32_t q0 = (qv.x >> sh4) & 0x0F0F0F0Fu, q1 = (qv.y >> sh4) & 0x0F0F0F0Fu;
        if (ty == T_Q5_K) { q0 |= ((hv.x >> jj) & 0x01010101u) << 4; q1 |= ((hv.y >> jj) & 0x01010101u) << 4; }
#pragma unroll
        for (int k = 0; k < 8; k++) {
            acc += kq_f(k < 4 ? q0 : q1, k & 3, 0.f) * xv[k];
            xs += xv[k];
        }
        return f16(b) * (float)sc * acc - f16(b + 2) * (float)mn * xs;
    }
    case T_Q6_K: {
        const uint8_t* b = row + (size_t)(j >> 3) * 210;
        const int jj = j & 7, n = jj >> 2, p = jj & 3;
        const uint2 lv8 = ld8(b + 64 * n + ((p & 1) ? 32 : 0) + 8 * part);
        const uint2 hv8 = ld8(b + 128 + 32 * n + 8 * part);
        const int sh4 = (p & 2) ? 4 : 0;
        const uint32_t q0 = ((lv8.x >> sh4) & 0x0F0F0F0Fu) | (((hv8.x >> (2 * p)) & 0x03030303u) << 4);
        const uint32_t q1 = ((lv8.y >> sh4) & 0x0F0F0F0Fu) | (((hv8.y >> (2 * p)) & 0x03030303u) << 4);
#pragma unroll
        for (int k = 0; k < 8; k++) acc += kq_f(k < 4 ? q0 : q1, k & 3, 32.f) * xv[k];
        return __fmul_rn(f16(b + 208) * (float)(int8_t)b[192 + 8 * n + 2 * p + (part >> 1)], acc);
    }
    default:
        for (int k = 0; k < 8; k++) acc += deq(ty, row, j, 8 * part + k) * xv[k];
        return acc;
    }
}

/* item 9: the body above inlined once per dense weight type (the type a constant: its decode without the switch,
   unrolled); the same operations in the same order, so the same bits */
#define DENSE_TYPES(F) F(T_Q6_K) F(T_Q4_K) F(T_Q5_K) F(T_IQ4_XS) F(T_BF16) F(T_Q8_0) F(T_IQ4_NL)

__device__ int qx_q6pf = 1;
__device__ int qx_d8pf = 1;
#ifndef XG_D8MINB
#define XG_D8MINB 1
#endif
static __device__ __forceinline__ void dense8_body_any(int ty, const uint8_t* W, int rows, int cols, long long rb,
                                                   const float* X, float* Y, int split, int bid) {
    __shared__ float part_[8];
    __shared__ float lv[16];                 /* the IQ4 levels: lanes index them at random (constant memory would
                                                serialize) */
    if (threadIdx.x < 16) lv[threadIdx.x] = (float)c_lv4[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, rpb = 8 / split;
    const int r = bid * rpb + warp / split, ws = warp % split;
    const int nsub = cols / 32;
    float acc = 0.f;
    if (r < rows && ty == T_Q6_K && qx_q6pf) {
        /* Q6_K, the next step's bytes loaded before this step's arithmetic (dot8's operations, in its order) */
        const uint8_t* row = W + (size_t)r * rb;
        const int pt = lane & 3;
        int j = 8 * ws + (lane >> 2);
        uint2 la = make_uint2(0u, 0u), ha = make_uint2(0u, 0u);
        int sa = 0, da = 0;
        float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xb = xa;
#define Q6_LOAD(J, L, Hh, S, D, XA, XB) do {                                                                                   const uint8_t* b_ = row + (size_t)((J) >> 3) * 210;                                                                const int jj_ = (J) & 7, n_ = jj_ >> 2, p_ = jj_ & 3;                                                              L = ld8(b_ + 64 * n_ + ((p_ & 1) ? 32 : 0) + 8 * pt);                                                              Hh = ld8(b_ + 128 + 32 * n_ + 8 * pt);                                                                             S = (int)(int8_t)b_[192 + 8 * n_ + 2 * p_ + (pt >> 1)];                                                            D = b_[208] | (b_[209] << 8);                                                                                      XA = *(const float4*)(X + 32 * (J) + 8 * pt); XB = *(const float4*)(X + 32 * (J) + 8 * pt + 4);                } while (0)
        if (j < nsub) Q6_LOAD(j, la, ha, sa, da, xa, xb);
        for (; j < nsub; j += 8 * split) {
            const int jn = j + 8 * split;
            uint2 lb = la, hb = ha;
            int sb = sa, db = da;
            float4 ya = xa, yb = xb;
            if (jn < nsub) Q6_LOAD(jn, lb, hb, sb, db, ya, yb);
            const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            const int p = (j & 7) & 3, sh4 = (p & 2) ? 4 : 0;
            float a = 0.f;
            for (int k = 0; k < 8; k++) {
                const int q = ((byte_of(la, k) >> sh4) & 15) | (((byte_of(ha, k) >> (2 * p)) & 3) << 4);
                a += QX_UF(q, 32) * xv[k];
            }
            acc += __fmul_rn(h2f((unsigned short)da) * (float)sa, a);
            la = lb; ha = hb; sa = sb; da = db; xa = ya; xb = yb;
        }
#undef Q6_LOAD
    } else if (r < rows && qx_d8pf) {
        /* two steps at a time: both steps' loads are issued before either's arithmetic (the second step's index is
           clamped to a valid one when it is past the end and its result is then not added); the sums are added in
           step order, so the same bits */
        const uint8_t* row = W + (size_t)r * rb;
        const int pt = lane & 3;
        for (int j0 = 8 * ws; j0 < nsub; j0 += 16 * split) {
            const int ja = j0 + (lane >> 2), jb0 = ja + 8 * split;
            const int jb = jb0 < nsub ? jb0 : (ja < nsub ? ja : 0);
            const int jac = ja < nsub ? ja : 0;
            const float ra = dot8(ty, row, jac, pt, X + 32 * jac + 8 * pt, lv);
            const float rb_ = dot8(ty, row, jb, pt, X + 32 * jb + 8 * pt, lv);
            if (ja < nsub) acc += ra;
            if (jb0 < nsub) acc += rb_;
        }
    } else if (r < rows) {
        const uint8_t* row = W + (size_t)r * rb;
        for (int j0 = 8 * ws; j0 < nsub; j0 += 8 * split) {
            const int j = j0 + (lane >> 2), pt = lane & 3;
            if (j < nsub) acc += dot8(ty, row, j, pt, X + 32 * j + 8 * pt, lv);
        }
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) part_[warp] = acc;
    __syncthreads();
    if (r < rows && ws == 0 && lane == 0) {
        float x = 0.f;
        for (int k = 0; k < split; k++) x += part_[warp + k];
        Y[r] = x;
    }
}
static __device__ __forceinline__ void dense8_body(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X, float* Y, int split, int bid) {
    switch (ty) {
#define DT_C(TYC) case TYC: dense8_body_any(TYC, W, rows, cols, rb, X, Y, split, bid); return;
    DENSE_TYPES(DT_C)
#undef DT_C
    default: dense8_body_any(ty, W, rows, cols, rb, X, Y, split, bid);
    }
}

#ifndef XG_IDENSE
#define XG_IDENSE 0
#endif
/* the one-token kernels' body: k_dense8's (float), or with KINDI_QXG_IDENSE the window kernels' with one row a warp
   (integer products of the 8-bit inputs: the same bits as every window size) */
static __device__ void dense8_body1(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                    const int8_t* XQ, float* Y, int split, int bid);
extern "C" __global__ void __launch_bounds__(256, XG_D8MINB) k_dense8(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                    float* Y, int split, const int8_t* XQ) {
    dense8_body1(ty, W, rows, cols, rb, X, XQ, Y, split, blockIdx.x);
}

/* dot8 for T tokens (x of token t at x + t * ldx): the 8 weights decoded once, then for each token the products
   and the final scaling written exactly as in dot8, so each token's value has dot8's bits */
/* the 8 weights of part `part` of 32-value sub-block j, as floats, with their scaling: kind 0: w; 1: s1 * w;
   2: s1 * w - s2 (K-quants with a minimum); 3: bf16 pairs (w). Shared by dot8T and the tensor-core path. */
static __device__ __forceinline__ void decode8(int ty, const uint8_t* row, int j, int part, const float* lv, float* w,
                                               float* s1o, float* s2o, int* kindo) {
    float s1 = 1.f, s2 = 0.f;
    int kind = 0;
    switch (ty) {
    case T_BF16: {
        const uint4 q = *(const uint4*)((const uint16_t*)row + 32 * j + 8 * part);
        const uint32_t u[4] = {q.x, q.y, q.z, q.w};
        for (int k = 0; k < 4; k++) { w[2 * k] = __uint_as_float(u[k] << 16); w[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u); }
        kind = 3;
        break;
    }
    case T_Q8_0: {
        const uint8_t* b = row + 34 * j;
        const uint2 q = ld8(b + 2 + 8 * part);
        const uint32_t q0 = q.x ^ 0x80808080u, q1 = q.y ^ 0x80808080u;
#pragma unroll
        for (int k = 0; k < 4; k++) { w[k] = kq_f(q0, k, 128.f); w[4 + k] = kq_f(q1, k, 128.f); }
        s1 = f16(b); kind = 1;
        break;
    }
    case T_IQ4_NL: {
        const uint2 q = ld8(row + 18 * j + 2 + 8 * (part & 1));
        const int sh4 = part < 2 ? 0 : 4;
        for (int k = 0; k < 8; k++) w[k] = lv[(byte_of(q, k) >> sh4) & 15];
        s1 = f16(row + 18 * j); kind = 1;
        break;
    }
    case T_IQ4_XS: {
        const uint8_t* b = row + (size_t)(j >> 3) * 136;
        const int jj = j & 7, sh = b[2] | (b[3] << 8);
        const int ls = ((b[4 + jj / 2] >> (4 * (jj & 1))) & 15) | (((sh >> (2 * jj)) & 3) << 4);
        const uint2 q = ld8(b + 8 + 16 * jj + 8 * (part & 1));
        const int sh4 = part < 2 ? 0 : 4;
        for (int k = 0; k < 8; k++) w[k] = lv[(byte_of(q, k) >> sh4) & 15];
        s1 = f16(b) * (float)(ls - 32); kind = 1;
        break;
    }
    case T_Q4_K: case T_Q5_K: {
        const uint8_t* b = row + (size_t)(j >> 3) * (ty == T_Q4_K ? 144 : 176);
        const int jj = j & 7, k4 = jj & 3, u = jj >> 2;
        const int xx = b[4 + k4], y = b[4 + k4 + 4], z = b[4 + k4 + 8];
        const int sc = ((u ? z : xx) & 15) | ((xx >> (2 * u)) & 48);
        const int mn = ((u ? (z >> 4) : y) & 15) | ((y >> (2 * u)) & 48);
        const uint2 qv = ld8(b + (ty == T_Q4_K ? 16 : 48) + 32 * (jj >> 1) + 8 * part);
        const uint2 hv = ty == T_Q5_K ? ld8(b + 16 + 8 * part) : make_uint2(0u, 0u);
        const int sh4 = (jj & 1) ? 4 : 0;
        uint32_t q0 = (qv.x >> sh4) & 0x0F0F0F0Fu, q1 = (qv.y >> sh4) & 0x0F0F0F0Fu;
        if (ty == T_Q5_K) { q0 |= ((hv.x >> jj) & 0x01010101u) << 4; q1 |= ((hv.y >> jj) & 0x01010101u) << 4; }
#pragma unroll
        for (int k = 0; k < 4; k++) { w[k] = kq_f(q0, k, 0.f); w[4 + k] = kq_f(q1, k, 0.f); }
        s1 = f16(b) * (float)sc; s2 = f16(b + 2) * (float)mn; kind = 2;
        break;
    }
    case T_Q6_K: {
        const uint8_t* b = row + (size_t)(j >> 3) * 210;
        const int jj = j & 7, n = jj >> 2, p = jj & 3;
        const uint2 lv8 = ld8(b + 64 * n + ((p & 1) ? 32 : 0) + 8 * part);
        const uint2 hv8 = ld8(b + 128 + 32 * n + 8 * part);
        const int sh4 = (p & 2) ? 4 : 0;
        const uint32_t q0 = ((lv8.x >> sh4) & 0x0F0F0F0Fu) | (((hv8.x >> (2 * p)) & 0x03030303u) << 4);
        const uint32_t q1 = ((lv8.y >> sh4) & 0x0F0F0F0Fu) | (((hv8.y >> (2 * p)) & 0x03030303u) << 4);
#pragma unroll
        for (int k = 0; k < 4; k++) { w[k] = kq_f(q0, k, 32.f); w[4 + k] = kq_f(q1, k, 32.f); }
        s1 = f16(b + 208) * (float)(int8_t)b[192 + 8 * n + 2 * p + (part >> 1)]; kind = 1;
        break;
    }
    default:
        for (int k = 0; k < 8; k++) w[k] = deq(ty, row, j, 8 * part + k);
        kind = 0;
        break;
    }
    *s1o = s1; *s2o = s2; *kindo = kind;
}

static __device__ __forceinline__ void dot8T(int ty, const uint8_t* row, int j, int part, const float* x, int ldx,
                                             int T, const float* lv, float* accT) {
    float w[8];
    float s1, s2;
    int kind;
    decode8(ty, row, j, part, lv, w, &s1, &s2, &kind);
    for (int t = 0; t < T; t++) {
        const float4 xa = *(const float4*)(x + (size_t)t * ldx), xb = *(const float4*)(x + (size_t)t * ldx + 4);
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float acc = 0.f, r;
        if (kind == 3) {
            for (int k = 0; k < 4; k++) { acc += w[2 * k] * xv[2 * k]; acc += w[2 * k + 1] * xv[2 * k + 1]; }
            r = acc;
        } else if (kind == 2) {
            float xs = 0.f;
            for (int k = 0; k < 8; k++) { acc += w[k] * xv[k]; xs += xv[k]; }
            r = s1 * acc - s2 * xs;
        } else {
            for (int k = 0; k < 8; k++) acc += w[k] * xv[k];
            r = kind == 1 ? __fmul_rn(s1, acc) : acc;     /* not fused with the add below (item 9) */
        }
        accT[t] += r;
    }
}

/* ---------------------------------------------------------------- dense products on tensor cores (sm_80+)
   Y [T][ldy] = X [T][cols] W^T for T <= 8 tokens: a block of 4 warps takes 64 rows; per chunk of up to 256 columns
   each warp decodes its 16 rows (decode8) into fp16 in shared memory, the T inputs are staged in fp16 (columns past
   T are zeros), and mma.m16n8k16 (fp16 in, fp32 accumulate) runs over the chunk. A token's column never mixes with
   the others, so one token alone and the same token in a window give the same bits. */
#define TC_CH 64                               /* columns per step of a warp */
#define TC_LDW (TC_CH + 8)                      /* row stride in halves (padding: fewer bank conflicts) */
#define TC_WARPS 8                              /* warps per block: the columns are split among them */
static __device__ __forceinline__ unsigned short f2h_bits(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
/* a block takes 16 rows; its 8 warps take the column steps w, w + 8, ... (each warp's own buffers, no block sync in
   the loop); each step: the warp decodes its 16 x 64 weights (decode8) and the T x 64 inputs into fp16 in shared
   memory, then 4 mma.m16n8k16 (fp16 in, fp32 accumulate). At the end the 8 warps' partial sums are added in warp
   order (a fixed order: the result does not depend on timing or on the other tokens). */
static __device__ __forceinline__ void dense_tc_body_any(int ty, const uint8_t* W, int rows, int cols, long long rb,
                                                     const float* X, int ldx, int T, float* Y, int ldy, int bid) {
    __shared__ __align__(16) unsigned short wh[TC_WARPS][16 * TC_LDW];
    __shared__ __align__(16) unsigned short xh[TC_WARPS][8 * TC_LDW];
    __shared__ float red[TC_WARPS][32][4];
    __shared__ float lv[16];
    if (threadIdx.x < 16) lv[threadIdx.x] = (float)c_lv4[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = bid * 16;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned short* wr = wh[warp];
    unsigned short* xr = xh[warp];
    const int ar = lane >> 2, ac = (lane & 3) * 2;
    for (int i = T * TC_LDW + lane; i < 8 * TC_LDW; i += 32) xr[i] = 0;   /* the token columns past T: zeros */
    for (int c0 = warp * TC_CH; c0 < cols; c0 += TC_WARPS * TC_CH) {
        const int kc = min(TC_CH, cols - c0);
        __syncwarp();
        for (int i = lane; i < T * kc; i += 32) {
            const int t = i / kc, k = i % kc;
            xr[t * TC_LDW + k] = f2h_bits(X[(size_t)t * ldx + c0 + k]);
        }
        const int ng = kc / 8;
        for (int u = lane; u < 16 * ng; u += 32) {
            const int rr = u / ng, g = u % ng, r = r0 + rr;
            unsigned short* o = wr + rr * TC_LDW + 8 * g;
            if (r < rows) {
                const int v0 = c0 + 8 * g;
                float w[8], s1, s2;
                int kind;
                decode8(ty, W + (size_t)r * rb, v0 >> 5, (v0 >> 3) & 3, lv, w, &s1, &s2, &kind);
                unsigned pk[4];
                for (int k = 0; k < 4; k++) {
                    const float v0 = kind == 1 ? s1 * w[2 * k] : kind == 2 ? s1 * w[2 * k] - s2 : w[2 * k];
                    const float v1 = kind == 1 ? s1 * w[2 * k + 1] : kind == 2 ? s1 * w[2 * k + 1] - s2 : w[2 * k + 1];
                    pk[k] = (unsigned)f2h_bits(v0) | ((unsigned)f2h_bits(v1) << 16);
                }
                *(uint4*)o = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            } else
                *(uint4*)o = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncwarp();
        for (int kk = 0; kk < kc; kk += 16) {
            const unsigned a0 = *(const unsigned*)(wr + ar * TC_LDW + kk + ac);
            const unsigned a1 = *(const unsigned*)(wr + (ar + 8) * TC_LDW + kk + ac);
            const unsigned a2 = *(const unsigned*)(wr + ar * TC_LDW + kk + ac + 8);
            const unsigned a3 = *(const unsigned*)(wr + (ar + 8) * TC_LDW + kk + ac + 8);
            const unsigned b0 = *(const unsigned*)(xr + ar * TC_LDW + kk + ac);
            const unsigned b1 = *(const unsigned*)(xr + ar * TC_LDW + kk + ac + 8);
#if __CUDA_ARCH__ >= 800
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                         "{%0,%1,%2,%3};"
                         : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
        }
    }
    for (int k = 0; k < 4; k++) red[warp][lane][k] = c[k];
    __syncthreads();
    if (warp == 0) {
        for (int k = 0; k < 4; k++) {
            float v = 0.f;
            for (int w2 = 0; w2 < TC_WARPS; w2++) v += red[w2][lane][k];
            c[k] = v;
        }
        const int ra = r0 + ar, n0 = ac;
        if (ra < rows) {
            if (n0 < T) Y[(size_t)n0 * ldy + ra] = c[0];
            if (n0 + 1 < T) Y[(size_t)(n0 + 1) * ldy + ra] = c[1];
        }
        if (ra + 8 < rows) {
            if (n0 < T) Y[(size_t)n0 * ldy + ra + 8] = c[2];
            if (n0 + 1 < T) Y[(size_t)(n0 + 1) * ldy + ra + 8] = c[3];
        }
    }
}
static __device__ __forceinline__ void dense_tc_body(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X, int ldx, int T, float* Y, int ldy, int bid) {
    switch (ty) {
#define DT_C(TYC) case TYC: dense_tc_body_any(TYC, W, rows, cols, rb, X, ldx, T, Y, ldy, bid); return;
    DENSE_TYPES(DT_C)
#undef DT_C
    default: dense_tc_body_any(ty, W, rows, cols, rb, X, ldx, T, Y, ldy, bid);
    }
}
extern "C" __global__ void k_dense_tc(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                      float* Y, int T) {
    dense_tc_body(ty, W, rows, cols, rb, X, cols, T, Y, rows, blockIdx.x);
}
/* several matrices of the same input in one launch: desc as k_dense8m (W, Y, row bytes, type, rows, cols, -,
   first block) */
extern "C" __global__ void k_dense_tcm(const long long* desc, int nm, const float* X, int T) {
    int i = 0;
    while (i + 1 < nm && (long long)blockIdx.x >= desc[(i + 1) * 8 + 7]) i++;
    const long long* d = desc + i * 8;
    dense_tc_body((int)d[3], (const uint8_t*)d[0], (int)d[4], (int)d[5], d[2], X, (int)d[5], T, (float*)d[1],
                  (int)d[4], (int)(blockIdx.x - d[7]));
}

/* dense8_body for T <= 8 tokens at once (X [T][cols], Y [T][ldy]): each token's row result goes through exactly
   the operations of dense8_body, in the same order, so it has the same bits as a one-token launch; the weights are
   read once for the T tokens (the multi-token verify window of the drafter) */
#define DT_MAX 5
static __device__ __forceinline__ void dense8_bodyT_any(int ty, const uint8_t* W, int rows, int cols, long long rb,
                                                    const float* X, float* Y, int ldy, int T, int split, int bid) {
    __shared__ float part_[DT_MAX][8];
    __shared__ float lv[16];
    if (threadIdx.x < 16) lv[threadIdx.x] = (float)c_lv4[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, rpb = 8 / split;
    const int r = bid * rpb + warp / split, ws = warp % split;
    const int nsub = cols / 32;
    float acc[DT_MAX];
    for (int t = 0; t < DT_MAX; t++) acc[t] = 0.f;
    if (r < rows) {
        const uint8_t* row = W + (size_t)r * rb;
        for (int j0 = 8 * ws; j0 < nsub; j0 += 8 * split) {
            const int j = j0 + (lane >> 2), pt = lane & 3;
            if (j < nsub) dot8T(ty, row, j, pt, X + 32 * j + 8 * pt, cols, T, lv, acc);
        }
    }
    for (int t = 0; t < T; t++) {
        float a = acc[t];
        for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) part_[t][warp] = a;
    }
    __syncthreads();
    if (r < rows && ws == 0 && lane == 0)
        for (int t = 0; t < T; t++) {
            float x = 0.f;
            for (int k = 0; k < split; k++) x += part_[t][warp + k];
            Y[(size_t)t * ldy + r] = x;
        }
}
static __device__ __forceinline__ void dense8_bodyT(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X, float* Y, int ldy, int T, int split, int bid) {
    switch (ty) {
#define DT_C(TYC) case TYC: dense8_bodyT_any(TYC, W, rows, cols, rb, X, Y, ldy, T, split, bid); return;
    DENSE_TYPES(DT_C)
#undef DT_C
    default: dense8_bodyT_any(ty, W, rows, cols, rb, X, Y, ldy, T, split, bid);
    }
}
extern "C" __global__ void k_dense8T(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                     float* Y, int split, int T) {
    dense8_bodyT(ty, W, rows, cols, rb, X, Y, rows, T, split, blockIdx.x);
}

/* several one-token products of the same input in one launch: desc holds 8 int64 per matrix (W, Y, row bytes,
   type, rows, cols, split, first block); the blocks of all matrices form one grid, so the small matrices run beside
   the big ones instead of one launch each */
extern "C" __global__ void __launch_bounds__(256, XG_D8MINB) k_dense8m(const long long* desc, int nm, const float* X,
                                                                        const int8_t* XQ) {
    int i = 0;
    while (i + 1 < nm && (long long)blockIdx.x >= desc[(i + 1) * 8 + 7]) i++;
    const long long* d = desc + i * 8;
    dense8_body1((int)d[3], (const uint8_t*)d[0], (int)d[4], (int)d[5], d[2], X, XQ, (float*)d[1], (int)d[6],
                 (int)(blockIdx.x - d[7]));
}
/* k_dense8m for T tokens (each output [T][rows] of its matrix) */
extern "C" __global__ void k_dense8mT(const long long* desc, int nm, const float* X, int T) {
    int i = 0;
    while (i + 1 < nm && (long long)blockIdx.x >= desc[(i + 1) * 8 + 7]) i++;
    const long long* d = desc + i * 8;
    dense8_bodyT((int)d[3], (const uint8_t*)d[0], (int)d[4], (int)d[5], d[2], X, (float*)d[1], (int)d[4], T,
                 (int)d[6], (int)(blockIdx.x - d[7]));
}

/* KINDI_QXG_D8T2 (default 1): k_dense8T / k_dense8mT with two rows a warp. The T tokens' 8 inputs of a step are
   loaded once for both rows (the inputs, not the weights, were most of the loads at T >= 2); each row goes through
   dot8T's operations in the same order and the same reductions, so every row keeps the bits of a one-token k_dense8
   launch. Rows per block: twice k_dense8's. */
#ifndef XG_RW
#define XG_RW 2
#endif
/* the raw bytes of one lane's 8 values of sub-block j (part pt) for the pipelined window path: kraw_fetch loads
   them, kraw_decode makes decode8's values, scales and kind from them (the same integer and float steps) */
struct KRaw { uint2 a, b; int c0, c1, c2, c3; };
static __device__ __forceinline__ void kraw_fetch(int ty, const uint8_t* row, int jb, int jj, int pt, KRaw& r) {
    /* jb: the superblock (the same for every lane of the warp), jj: the lane's sub-block in it (fixed for the loop) */
    if (ty == T_Q6_K) {
        /* a 210-byte block starts on a word or 2 bytes past one, and the lane's quants sit at multiples of 8 from its
           start: one word-aligned base and shift for the block, then read-only word loads (ld8's words and shifts) */
        const size_t a = (size_t)(row + (size_t)jb * 210);
        const int n = jj >> 2, p = jj & 3, sh = (int)(a & 2) * 8;
        const unsigned int* w = (const unsigned int*)(a & ~(size_t)3);
        const unsigned int* wl = w + 16 * n + ((p & 1) ? 8 : 0) + 2 * pt;
        const unsigned int* wh = w + 32 + 8 * n + 2 * pt;
        const unsigned int l0 = __ldg(wl), l1 = __ldg(wl + 1), l2 = sh ? __ldg(wl + 2) : 0u;
        const unsigned int h0 = __ldg(wh), h1 = __ldg(wh + 1), h2 = sh ? __ldg(wh + 2) : 0u;
        r.a.x = __funnelshift_r(l0, l1, sh); r.a.y = __funnelshift_r(l1, l2, sh);
        r.b.x = __funnelshift_r(h0, h1, sh); r.b.y = __funnelshift_r(h1, h2, sh);
        const uint8_t* b = (const uint8_t*)a;
        r.c0 = (int)__ldg((const signed char*)(b + 192 + 8 * n + 2 * p + (pt >> 1)));
        r.c1 = __ldg((const unsigned short*)(b + 208));    /* blocks and rows 2-byte aligned */
    } else if (ty == T_IQ4_XS) {
        /* every dense matrix is its own card allocation and an IQ4_XS row is a multiple of 136 bytes, so its blocks
           are 8-byte aligned: the 8-byte header in one load, the quants in one */
        const uint8_t* b = row + (size_t)jb * 136;
        const uint2 h = __ldg((const uint2*)b);
        r.a = __ldg((const uint2*)(b + 8 + 16 * jj + 8 * (pt & 1)));
        r.c0 = (h.y >> (8 * (jj >> 1))) & 255u;
        r.c1 = h.x >> 16;
        r.c2 = h.x & 0xFFFFu;
    } else {                                               /* Q4_K, Q5_K: 16-byte aligned blocks (144, 176) */
        const uint8_t* b = row + (size_t)jb * (ty == T_Q4_K ? 144 : 176);
        const int k4 = jj & 3;
        const uint4 h = __ldg((const uint4*)b);           /* d, dmin, the 12 scale bytes */
        r.c0 = (h.y >> (8 * k4)) & 255u; r.c1 = (h.z >> (8 * k4)) & 255u; r.c2 = (h.w >> (8 * k4)) & 255u;
        r.c3 = (int)h.x;
        r.a = __ldg((const uint2*)(b + (ty == T_Q4_K ? 16 : 48) + 32 * (jj >> 1) + 8 * pt));
        r.b = ty == T_Q5_K ? __ldg((const uint2*)(b + 16 + 8 * pt)) : make_uint2(0u, 0u);
    }
}
static __device__ __forceinline__ void kraw_decode(int ty, const KRaw& r, int jj, int pt, const float* lv, float* w,
                                                   float* s1o, float* s2o, int* kindo) {
    (void)lv;
    if (ty == T_Q6_K) {
        const int p = jj & 3, sh4 = (p & 2) ? 4 : 0;
        const uint32_t q0 = ((r.a.x >> sh4) & 0x0F0F0F0Fu) | (((r.b.x >> (2 * p)) & 0x03030303u) << 4);
        const uint32_t q1 = ((r.a.y >> sh4) & 0x0F0F0F0Fu) | (((r.b.y >> (2 * p)) & 0x03030303u) << 4);
#pragma unroll
        for (int k = 0; k < 4; k++) { w[k] = kq_f(q0, k, 32.f); w[4 + k] = kq_f(q1, k, 32.f); }
        *s1o = h2f((unsigned short)r.c1) * (float)r.c0; *s2o = 0.f; *kindo = 1;
    } else if (ty == T_IQ4_XS) {
        const int ls = ((r.c0 >> (4 * (jj & 1))) & 15) | (((r.c1 >> (2 * jj)) & 3) << 4);
        const int sh4 = pt < 2 ? 0 : 4;
        const uint32_t v0 = kq_lv4((r.a.x >> sh4) & 0x0F0F0F0Fu) ^ 0x80808080u;   /* level + 128 */
        const uint32_t v1 = kq_lv4((r.a.y >> sh4) & 0x0F0F0F0Fu) ^ 0x80808080u;
#pragma unroll
        for (int k = 0; k < 4; k++) { w[k] = kq_f(v0, k, 128.f); w[4 + k] = kq_f(v1, k, 128.f); }
        *s1o = h2f((unsigned short)r.c2) * (float)(ls - 32); *s2o = 0.f; *kindo = 1;
    } else {                                               /* Q4_K, Q5_K */
        const int u = jj >> 2, xx = r.c0, y = r.c1, z = r.c2;
        const int sc = ((u ? z : xx) & 15) | ((xx >> (2 * u)) & 48);
        const int mn = ((u ? (z >> 4) : y) & 15) | ((y >> (2 * u)) & 48);
        const int sh4 = (jj & 1) ? 4 : 0;
        uint32_t q0 = (r.a.x >> sh4) & 0x0F0F0F0Fu, q1 = (r.a.y >> sh4) & 0x0F0F0F0Fu;
        if (ty == T_Q5_K) {
            q0 |= ((r.b.x >> jj) & 0x01010101u) << 4;
            q1 |= ((r.b.y >> jj) & 0x01010101u) << 4;
        }
#pragma unroll
        for (int k = 0; k < 4; k++) { w[k] = kq_f(q0, k, 0.f); w[4 + k] = kq_f(q1, k, 0.f); }
        *s1o = h2f((unsigned short)(r.c3 & 0xFFFF)) * (float)sc;
        *s2o = h2f((unsigned short)((unsigned)r.c3 >> 16)) * (float)mn; *kindo = 2;
    }
}
/* KINDI_QXG_IDENSE: the inputs as 8-bit integers per 32 values (k_qx8: q = round(x * 127 / max|x|), d = max|x| / 127,
   the same for every product of that input), the weights' packed small integers taken against them with dp4a; then
   per lane r = s1 * (d * isum) (minus s2 * (d * the inputs' integer sum) for the K-quants with minimums), in this
   order in every kernel: one token and every window size give a row the same bits */
extern "C" __global__ void k_qx8(const float* X, int n32, int8_t* Q, float* D) {
    const int b = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (b >= n32) return;
    const float x = X[(size_t)32 * b + lane];
    float m = fabsf(x);
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float inv = m > 0.f ? __fdiv_rn(127.f, m) : 0.f;
    int q = __float2int_rn(__fmul_rn(x, inv));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    Q[(size_t)32 * b + lane] = (int8_t)q;
    if (lane == 0) D[b] = __fdiv_rn(m, 127.f);
}
/* the float of a small int (|i| < 2^22), exactly, on the full-rate pipes (I2F is quarter rate) */
static __device__ __forceinline__ float i2f_small(int i) {
    return __fsub_rn(__int_as_float(0x4B400000 + i), 12582912.f);
}
/* unsigned bytes of a against signed bytes of b, plus c */
static __device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
    const int ib = (int)b;                       /* no dp4a: the four byte products (exact: the same sum) */
    return c + (int)(a & 255u) * (ib << 24 >> 24) + (int)((a >> 8) & 255u) * (ib << 16 >> 24) +
           (int)((a >> 16) & 255u) * (ib << 8 >> 24) + (int)(a >> 24) * (ib >> 24);
#else
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#endif
}
/* kraw_decode's integers, packed: the 8 weights as bytes (Q6_K: 0..63, its 32 taken off through the inputs' sum;
   Q4_K / Q5_K: unsigned; IQ4_XS: the signed levels), the scales as kraw_decode makes them; kind 1 Q6_K, 2 the
   K-quants with minimums, 3 IQ4_XS */
static __device__ __forceinline__ void kraw_ibytes(int ty, const KRaw& r, int jj, int pt, uint32_t* w0, uint32_t* w1,
                                                   float* s1o, float* s2o, int* kindo) {
    if (ty == T_Q6_K) {
        const int p = jj & 3, sh4 = (p & 2) ? 4 : 0;
        *w0 = ((r.a.x >> sh4) & 0x0F0F0F0Fu) | (((r.b.x >> (2 * p)) & 0x03030303u) << 4);
        *w1 = ((r.a.y >> sh4) & 0x0F0F0F0Fu) | (((r.b.y >> (2 * p)) & 0x03030303u) << 4);
        *s1o = h2f((unsigned short)r.c1) * (float)r.c0; *s2o = 0.f; *kindo = 1;
    } else if (ty == T_IQ4_XS) {
        const int ls = ((r.c0 >> (4 * (jj & 1))) & 15) | (((r.c1 >> (2 * jj)) & 3) << 4);
        const int sh4 = pt < 2 ? 0 : 4;
        *w0 = kq_lv4((r.a.x >> sh4) & 0x0F0F0F0Fu);
        *w1 = kq_lv4((r.a.y >> sh4) & 0x0F0F0F0Fu);
        *s1o = h2f((unsigned short)r.c2) * (float)(ls - 32); *s2o = 0.f; *kindo = 3;
    } else {
        const int u = jj >> 2, xx = r.c0, y = r.c1, z = r.c2;
        const int sc = ((u ? z : xx) & 15) | ((xx >> (2 * u)) & 48);
        const int mn = ((u ? (z >> 4) : y) & 15) | ((y >> (2 * u)) & 48);
        const int sh4 = (jj & 1) ? 4 : 0;
        uint32_t q0 = (r.a.x >> sh4) & 0x0F0F0F0Fu, q1 = (r.a.y >> sh4) & 0x0F0F0F0Fu;
        if (ty == T_Q5_K) {
            q0 |= ((r.b.x >> jj) & 0x01010101u) << 4;
            q1 |= ((r.b.y >> jj) & 0x01010101u) << 4;
        }
        *w0 = q0; *w1 = q1;
        *s1o = h2f((unsigned short)(r.c3 & 0xFFFF)) * (float)sc;
        *s2o = h2f((unsigned short)((unsigned)r.c3 >> 16)) * (float)mn; *kindo = 2;
    }
}
template <int T>
static __device__ __forceinline__ void xt_load(const float* X, int cols, int j, int pt, float (*xa)[8]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const float* x_ = X + (size_t)t * cols + 32 * j + 8 * pt;
        const float4 u_ = *(const float4*)x_, v_ = *(const float4*)(x_ + 4);
        xa[t][0] = u_.x; xa[t][1] = u_.y; xa[t][2] = u_.z; xa[t][3] = u_.w;
        xa[t][4] = v_.x; xa[t][5] = v_.y; xa[t][6] = v_.z; xa[t][7] = v_.w;
    }
}
#ifndef XG_Q6T
#define XG_Q6T 1
#endif
#ifndef XG_KQ5
#define XG_KQ5 3                                      /* Q5_K in the pipelined path for windows up to this many tokens */
#endif
template <int T, int RW>
static __device__ __forceinline__ void dense8_bodyT2_any(int ty, const uint8_t* W, int rows, int cols, long long rb,
                                                         const float* X, const int8_t* XQ, float* Y, int ldy,
                                                         int split, int bid) {
    __shared__ float part2[RW][T][8];
    __shared__ float lv[16];
    if (threadIdx.x < 16) lv[threadIdx.x] = (float)c_lv4[threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31, rpb = 8 / split;
    const int r0 = bid * RW * rpb + RW * (warp / split), ws = warp % split;
    const int nsub = cols / 32;
    float acc[RW][T];
#pragma unroll
    for (int q = 0; q < RW; q++)
#pragma unroll
        for (int t = 0; t < T; t++) acc[q][t] = 0.f;
#if XG_IDENSE
    if (r0 < rows && (ty == T_Q6_K || ty == T_Q4_K || ty == T_IQ4_XS || ty == T_Q5_K)) {
        const int pt = lane & 3, jj = lane >> 2, nsb = nsub >> 3;
        const uint8_t* rowp[RW];
#pragma unroll
        for (int q = 0; q < RW; q++) rowp[q] = W + (size_t)(r0 + q < rows ? r0 + q : r0) * rb;
        const float* DX = (const float*)(XQ + (size_t)T * cols);
        for (int jb = ws; jb < nsb; jb += split) {
            const int j = 8 * jb + jj;
            KRaw ra[RW];
#pragma unroll
            for (int q = 0; q < RW; q++) kraw_fetch(ty, rowp[q], jb, jj, pt, ra[q]);
            uint2 xq[T];
            float dx[T];
            int sx[T];
#pragma unroll
            for (int t = 0; t < T; t++) {
                xq[t] = __ldg((const uint2*)(XQ + (size_t)t * cols + 32 * j + 8 * pt));
                dx[t] = __ldg(DX + (size_t)t * nsub + j);
                sx[t] = __dp4a((int)xq[t].x, 0x01010101, __dp4a((int)xq[t].y, 0x01010101, 0));
            }
#pragma unroll
            for (int q = 0; q < RW; q++) {
                uint32_t w0, w1;
                float s1, s2;
                int kind;
                kraw_ibytes(ty, ra[q], jj, pt, &w0, &w1, &s1, &s2, &kind);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    int is;
                    if (kind == 3) is = __dp4a((int)w0, (int)xq[t].x, __dp4a((int)w1, (int)xq[t].y, 0));
                    else is = dp4a_us(w0, xq[t].x, dp4a_us(w1, xq[t].y, 0));
                    if (kind == 1) is -= 32 * sx[t];
                    float r = __fmul_rn(s1, __fmul_rn(dx[t], i2f_small(is)));
                    if (kind == 2) r = __fsub_rn(r, __fmul_rn(s2, __fmul_rn(dx[t], i2f_small(sx[t]))));
                    acc[q][t] = __fadd_rn(acc[q][t], r);
                }
            }
        }
    } else
#endif
    if (r0 < rows && XG_Q6T && (ty == T_Q6_K || ty == T_Q4_K || ty == T_IQ4_XS || (T <= XG_KQ5 && ty == T_Q5_K))) {
        /* the K-quants and IQ4_XS: each step's raw bytes and inputs loaded, then the values, scales and sums made as
           decode8 and the branches below make them, in the same order: the same bits. The loop runs over superblocks
           (warp-uniform; nsub is a multiple of 8 for these types); each lane's sub-block and part stay fixed, and
           each row's start is made once. (Loading the next step's bytes ahead was slower: registers.) */
        const int pt = lane & 3, jj = lane >> 2, nsb = nsub >> 3;
        const uint8_t* rowp[RW];
#pragma unroll
        for (int q = 0; q < RW; q++) rowp[q] = W + (size_t)(r0 + q < rows ? r0 + q : r0) * rb;
        for (int jb = ws; jb < nsb; jb += split) {
            KRaw ra[RW];
            float xa[T][8];
#pragma unroll
            for (int q = 0; q < RW; q++) kraw_fetch(ty, rowp[q], jb, jj, pt, ra[q]);
            xt_load<T>(X, cols, 8 * jb + jj, pt, xa);
            /* the K-quants' x sums: the same adds in the same order for every row, so made once a step */
            float xsum[T];
            if (ty == T_Q4_K || ty == T_Q5_K) {
#pragma unroll
                for (int t = 0; t < T; t++) {
                    float xs = 0.f;
                    for (int k = 0; k < 8; k++) xs += xa[t][k];
                    xsum[t] = xs;
                }
            }
#pragma unroll
            for (int q = 0; q < RW; q++) {
                float w[8], s1, s2;
                int kind;
                kraw_decode(ty, ra[q], jj, pt, lv, w, &s1, &s2, &kind);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    float a = 0.f, r;
                    if (kind == 2) {
                        const float xs = xsum[t];
                        for (int k = 0; k < 8; k++) a += w[k] * xa[t][k];
                        r = s1 * a - s2 * xs;
                    } else {
                        for (int k = 0; k < 8; k++) a += w[k] * xa[t][k];
                        r = __fmul_rn(s1, a);
                    }
                    acc[q][t] += r;
                }
            }
        }
    } else if (r0 < rows) {
        for (int j0 = 8 * ws; j0 < nsub; j0 += 8 * split) {
            const int j = j0 + (lane >> 2), pt = lane & 3;
            if (j >= nsub) continue;
            float xv[T][8];
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float* x = X + (size_t)t * cols + 32 * j + 8 * pt;
                const float4 xa = *(const float4*)x, xb = *(const float4*)(x + 4);
                xv[t][0] = xa.x; xv[t][1] = xa.y; xv[t][2] = xa.z; xv[t][3] = xa.w;
                xv[t][4] = xb.x; xv[t][5] = xb.y; xv[t][6] = xb.z; xv[t][7] = xb.w;
            }
#pragma unroll
            for (int q = 0; q < RW; q++) {
                float w[8], s1, s2;
                int kind;
                decode8(ty, W + (size_t)(r0 + q < rows ? r0 + q : r0) * rb, j, pt, lv, w, &s1, &s2, &kind);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    float a = 0.f, r;
                    if (kind == 3) {
                        for (int k = 0; k < 4; k++) { a += w[2 * k] * xv[t][2 * k]; a += w[2 * k + 1] * xv[t][2 * k + 1]; }
                        r = a;
                    } else if (kind == 2) {
                        float xs = 0.f;
                        for (int k = 0; k < 8; k++) { a += w[k] * xv[t][k]; xs += xv[t][k]; }
                        r = s1 * a - s2 * xs;
                    } else {
                        for (int k = 0; k < 8; k++) a += w[k] * xv[t][k];
                        r = kind == 1 ? __fmul_rn(s1, a) : a;
                    }
                    acc[q][t] += r;
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < RW; q++)
#pragma unroll
        for (int t = 0; t < T; t++) {
            float a = acc[q][t];
            for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) part2[q][t][warp] = a;
        }
    __syncthreads();
    if (ws == 0 && lane < RW * T) {
        const int q = lane / T, t = lane % T, r = r0 + q;
        if (r < rows) {
            float x = 0.f;
            for (int k = 0; k < split; k++) x += part2[q][t][warp + k];
            Y[(size_t)t * ldy + r] = x;
        }
    }
}
static __device__ __forceinline__ void dense8_bodyT2(int ty, const uint8_t* W, int rows, int cols, long long rb,
                                                     const float* X, const int8_t* XQ, float* Y, int ldy, int T,
                                                     int split, int bid) {
    switch (T) {
#define T2_C(TT) case TT: switch (ty) { \
        case T_Q6_K: dense8_bodyT2_any<TT, XG_RW>(T_Q6_K, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        case T_Q4_K: dense8_bodyT2_any<TT, XG_RW>(T_Q4_K, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        case T_Q5_K: dense8_bodyT2_any<TT, XG_RW>(T_Q5_K, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        case T_IQ4_XS: dense8_bodyT2_any<TT, XG_RW>(T_IQ4_XS, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        case T_BF16: dense8_bodyT2_any<TT, XG_RW>(T_BF16, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        case T_Q8_0: dense8_bodyT2_any<TT, XG_RW>(T_Q8_0, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        case T_IQ4_NL: dense8_bodyT2_any<TT, XG_RW>(T_IQ4_NL, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; \
        default: dense8_bodyT2_any<TT, XG_RW>(ty, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return; }
    T2_C(1) T2_C(2) T2_C(3) T2_C(4)
    default: T2_C(5)
#undef T2_C
    }
}
#ifndef XG_D8TMINB
#define XG_D8TMINB 1
#endif
extern "C" __global__ void __launch_bounds__(256, XG_D8TMINB) k_dense8T2(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                      float* Y, int split, int T, const int8_t* XQ) {
    dense8_bodyT2(ty, W, rows, cols, rb, X, XQ, Y, rows, T, split, blockIdx.x);
}
/* one kernel per window size (KINDI_QXG_T2FIX): each gets its own register budget (the all-sizes kernel holds the
   largest one's and spills under the 3-blocks launch bounds) */
template <int TT>
static __device__ __forceinline__ void dense8_bodyT2_fix(int ty, const uint8_t* W, int rows, int cols, long long rb,
                                                         const float* X, const int8_t* XQ, float* Y, int ldy,
                                                         int split, int bid) {
    switch (ty) {
    case T_Q6_K: dense8_bodyT2_any<TT, XG_RW>(T_Q6_K, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    case T_Q4_K: dense8_bodyT2_any<TT, XG_RW>(T_Q4_K, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    case T_Q5_K: dense8_bodyT2_any<TT, XG_RW>(T_Q5_K, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    case T_IQ4_XS: dense8_bodyT2_any<TT, XG_RW>(T_IQ4_XS, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    case T_BF16: dense8_bodyT2_any<TT, XG_RW>(T_BF16, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    case T_Q8_0: dense8_bodyT2_any<TT, XG_RW>(T_Q8_0, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    case T_IQ4_NL: dense8_bodyT2_any<TT, XG_RW>(T_IQ4_NL, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    default: dense8_bodyT2_any<TT, XG_RW>(ty, W, rows, cols, rb, X, XQ, Y, ldy, split, bid); return;
    }
}
static __device__ void dense8_body1(int ty, const uint8_t* W, int rows, int cols, long long rb, const float* X,
                                    const int8_t* XQ, float* Y, int split, int bid) {
#if XG_IDENSE
    switch (ty) {
#define DT1_C(TYC) case TYC: dense8_bodyT2_any<1, 1>(TYC, W, rows, cols, rb, X, XQ, Y, rows, split, bid); return;
    DENSE_TYPES(DT1_C)
#undef DT1_C
    default: dense8_bodyT2_any<1, 1>(ty, W, rows, cols, rb, X, XQ, Y, rows, split, bid);
    }
#else
    (void)XQ;
    dense8_body(ty, W, rows, cols, rb, X, Y, split, bid);
#endif
}
#define T2FIX_K(TT) extern "C" __global__ void __launch_bounds__(256, XG_D8TMINB) k_dense8T2_##TT(int ty, const uint8_t* W, int rows,                                                                              int cols, long long rb, const float* X,                                                                              float* Y, int split, int T, const int8_t* XQ) {     dense8_bodyT2_fix<TT>(ty, W, rows, cols, rb, X, XQ, Y, rows, split, blockIdx.x); } extern "C" __global__ void __launch_bounds__(256, XG_D8TMINB) k_dense8mT2_##TT(const long long* desc, int nm,                                                                               const float* X, int T, const int8_t* XQ) {     int i = 0;     while (i + 1 < nm && (long long)blockIdx.x >= desc[(i + 1) * 8 + 7]) i++;     const long long* d = desc + i * 8;     dense8_bodyT2_fix<TT>((int)d[3], (const uint8_t*)d[0], (int)d[4], (int)d[5], d[2], X, XQ, (float*)d[1], (int)d[4],                           (int)d[6], (int)(blockIdx.x - d[7])); }
T2FIX_K(1) T2FIX_K(2) T2FIX_K(3)
#undef T2FIX_K
extern "C" __global__ void __launch_bounds__(256, XG_D8TMINB) k_dense8mT2(const long long* desc, int nm, const float* X,
                                                                         int T, const int8_t* XQ) {
    int i = 0;
    while (i + 1 < nm && (long long)blockIdx.x >= desc[(i + 1) * 8 + 7]) i++;
    const long long* d = desc + i * 8;
    dense8_bodyT2((int)d[3], (const uint8_t*)d[0], (int)d[4], (int)d[5], d[2], X, XQ, (float*)d[1], (int)d[4], T,
                  (int)d[6], (int)(blockIdx.x - d[7]));
}

/* ---------------------------------------------------------------- fused one-token steps (the decode graphs) */
/* The steps between the matrix products of one decode token, each as one kernel (they were 4-30 small PyTorch
   kernels each). Same formulas as kindi/qwen4exp_ref.py; fp32. */

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
/* sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the result */
static __device__ __forceinline__ float block_sum(float v, float* sh32) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh32[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; i++) t += sh32[i];
    return t;
}
static __device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
static __device__ __forceinline__ float silu_(float x) { return x / (1.f + expf(-x)); }

/* out[g][c] = h[g][c] * rsqrt(mean_c h[g][c]^2 + eps) * w[g * n + c]; one block per group g */
extern "C" __global__ void k_rms_g(const float* h, const float* w, float* out, int n, float eps) {
    __shared__ float sh32[32];
    const float* x = h + (size_t)blockIdx.x * n;
    float v = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) v += x[c] * x[c];
    const float r = rsqrtf(block_sum(v, sh32) / n + eps);
    for (int c = threadIdx.x; c < n; c += blockDim.x)
        out[(size_t)blockIdx.x * n + c] = x[c] * r * w[(size_t)blockIdx.x * n + c];
}
/* x = silu(x * s) in place */
extern "C" __global__ void k_silu_s(float* x, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = silu_(x[i] * s);
}
/* the gated-residual mixer's output: x[c] = mean_g sigmoid(u[g][c]) * hn[g][c]; wr[g] = 2 sigmoid(inj[g] / hc)
   (inj may be null: no write gates) */
extern "C" __global__ void k_mix_out(const float* u, const float* hn, const float* inj, float* x, float* wr, int G,
                                     int n, float hc) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) {
        float acc = 0.f;
        for (int g = 0; g < G; g++) acc += sigm(u[(size_t)g * n + c]) * hn[(size_t)g * n + c];
        x[c] = acc / G;
    }
    if (inj && blockIdx.x == 0 && threadIdx.x < G) wr[threadIdx.x] = 2.f * sigm(inj[threadIdx.x] / hc);
}
/* the residual write: out[g][c] = hin[g][c] + (y[c] + sigmoid(sdot) * y2[c]) * wr[g]  (y2 null: no second term) */
extern "C" __global__ void k_write(const float* hin, const float* y, const float* y2, const float* sdot,
                                   const float* wr, float* out, int G, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= G * n) return;
    const int g = i / n, c = i % n;
    float v = y[c];
    if (y2) v += sigm(sdot[0]) * y2[c];
    out[i] = hin[i] + v * wr[g];
}
/* out[0] = sum x[c] v[c]; one block */
extern "C" __global__ void k_dot(const float* x, const float* v, int n, float* out) {
    __shared__ float sh32[32];
    float a = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) a += x[c] * v[c];
    a = block_sum(a, sh32);
    if (threadIdx.x == 0) out[0] = a;
}
/* SwiGLU of the shared expert: out = silu(g) * u */
extern "C" __global__ void k_swiglu(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = silu_(g[i]) * u[i];
}
/* the router: softmax over n logits, the k largest (ties: the lower index), their probabilities renormalized to
   sum 1; one block of n threads (n <= 1024) */
/* the router's top k of pv[0..blockDim) (probabilities; -1 = none): the same choice and order as k rounds of a block
   argmax (the largest value, ties to the lower index), in two stages without k block barriers: each warp takes its
   own top k (warp argmax rounds, no barrier; the block's top k are among them), then warp 0 takes the top k of the
   warps' candidates. idx/top as the rounds wrote them. (KINDI_QXG_ROUTER2) */
#ifndef XG_ROUTER2
#define XG_ROUTER2 1
#endif
static __device__ __forceinline__ void router_topk2(const float* pv, int k, int* idx, float* top, float* cv, int* ci) {
    const int t = threadIdx.x, lane = t & 31, w = t >> 5, nw = blockDim.x >> 5;
    float v = pv[t];
    for (int r = 0; r < k; r++) {
        float bv = v;
        int bi = t;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { cv[w * k + r] = bv; ci[w * k + r] = bi; }
        if (bi == t) v = -1.f;
    }
    __syncthreads();
    if (w == 0) {
        const int nc = nw * k;
        float mv[16];
        int mi[16];
        int cnt = 0;
        for (int j = lane; j < nc && cnt < 16; j += 32) { mv[cnt] = cv[j]; mi[cnt] = ci[j]; cnt++; }
        for (int r = 0; r < k; r++) {
            float bv = -2.f;
            int bi = 0x7fffffff;
            for (int j = 0; j < cnt; j++)
                if (mv[j] > bv || (mv[j] == bv && mi[j] < bi)) { bv = mv[j]; bi = mi[j]; }
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { idx[r] = bi; top[r] = bv; }
            for (int j = 0; j < cnt; j++)
                if (mi[j] == bi) mv[j] = -2.f;
        }
    }
    __syncthreads();
}
extern "C" __global__ void k_router(const float* logit, int n, int k, int* idx, float* top) {
    __shared__ float sh32[32], pv[1024], tv[32];
    __shared__ int ti[32];
    const int t = threadIdx.x;
    const float x = t < n ? logit[t] : __int_as_float(0xff800000);
    float m = x;                                           /* block max */
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __syncthreads();
    if ((t & 31) == 0) sh32[t >> 5] = m;
    __syncthreads();
    m = __int_as_float(0xff800000);
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, sh32[i]);
    const float e = t < n ? expf(x - m) : 0.f;
    const float sum = block_sum(e, sh32);
    pv[t] = t < n ? e / sum : -1.f;
    __syncthreads();
    if (XG_ROUTER2 && k <= 32 && (int)(blockDim.x >> 5) * k <= 512) {
        __shared__ float cv_[512];
        __shared__ int ci_[512];
        router_topk2(pv, k, idx, top, cv_, ci_);
    } else
    for (int r = 0; r < k; r++) {                          /* k rounds of a block argmax */
        float bv = pv[t];
        int bi = t;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __syncthreads();
        if ((t & 31) == 0) { tv[t >> 5] = bv; ti[t >> 5] = bi; }
        __syncthreads();
        if (t == 0) {
            float b = tv[0];
            int bi2 = ti[0];
            for (int i = 1; i < (int)(blockDim.x >> 5); i++)
                if (tv[i] > b || (tv[i] == b && ti[i] < bi2)) { b = tv[i]; bi2 = ti[i]; }
            idx[r] = bi2;
            top[r] = b;
            pv[bi2] = -1.f;
        }
        __syncthreads();
    }
    if (t == 0) {
        float s = 0.f;
        for (int r = 0; r < k; r++) s += top[r];
        for (int r = 0; r < k; r++) top[r] = top[r] / s;
    }
}
/* the causal conv of Gated DeltaNet for one new position, per channel c: y = silu(sum_j w[c][j] full[j]) with
   full = (state[0..kw-2][c], x[c]); the state shifts by one in place (each thread owns its channel) */
extern "C" __global__ void k_conv1(const float* x, float* state, const float* w, float* y, int C, int kw) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int j = 0; j < kw - 1; j++) acc += w[(size_t)c * kw + j] * state[(size_t)j * C + c];
    acc += w[(size_t)c * kw + kw - 1] * x[c];
    for (int j = 0; j < kw - 2; j++) state[(size_t)j * C + c] = state[(size_t)(j + 1) * C + c];
    state[(size_t)(kw - 2) * C + c] = x[c];
    y[c] = silu_(acc);
}
/* the delta rule for one token. conv: the conv outputs [q (nk x 128) | k (nk x 128) | v (nv x 128)]; V head h uses
   QK head h % nk (the file's tile order). Block (h, cc): columns 32 cc .. + 31 of head h's 128 x 128 state; warp r
   of the 4 holds rows 32 r .. + 31 of those columns. out: the raw head outputs [nv][128] (the gated norm is k_gnorm). */
extern "C" __global__ void k_gdn1(int nk, const float* conv, const float* b, const float* a, const float* ssm_a,
                                  const float* dtb, float* S, float* out) {
    __shared__ float sq[128], sk[128], red[4][32], nrm[2];
    const int h = blockIdx.x, cc = blockIdx.y, t = threadIdx.x, jl = t & 31, r = t >> 5;
    const int j = cc * 32 + jl, kh = h % nk;
    sq[t] = conv[kh * 128 + t];
    sk[t] = conv[nk * 128 + kh * 128 + t];
    __syncthreads();
    if (r < 2) {
        const float* src = r == 0 ? sq : sk;
        float v = 0.f;
        for (int i = jl; i < 128; i += 32) v += src[i] * src[i];
        v = warp_sum(v);
        if (jl == 0) nrm[r] = v;
    }
    __syncthreads();
    const float qs = rsqrtf(nrm[0] + 1e-6f) / sqrtf(128.f), ks = rsqrtf(nrm[1] + 1e-6f);
    const float bt = sigm(b[h]);
    const float xa = a[h] + dtb[h];
    const float dec = expf(ssm_a[h] * (xa > 20.f ? xa : log1pf(expf(xa))));
    float* Sh = S + (size_t)h * 128 * 128;
    float sv[32], pm = 0.f;
    for (int ii = 0; ii < 32; ii++) {
        const int i = r * 32 + ii;
        sv[ii] = Sh[i * 128 + j] * dec;
        pm += sv[ii] * (sk[i] * ks);
    }
    red[r][jl] = pm;
    __syncthreads();
    const float mem = red[0][jl] + red[1][jl] + red[2][jl] + red[3][jl];
    const float delta = (conv[2 * nk * 128 + h * 128 + j] - mem) * bt;
    __syncthreads();
    float po = 0.f;
    for (int ii = 0; ii < 32; ii++) {
        const int i = r * 32 + ii;
        sv[ii] += (sk[i] * ks) * delta;
        po += sv[ii] * (sq[i] * qs);
        Sh[i * 128 + j] = sv[ii];
    }
    red[r][jl] = po;
    __syncthreads();
    if (r == 0) out[h * 128 + j] = red[0][jl] + red[1][jl] + red[2][jl] + red[3][jl];
}
/* the gated norm of Gated DeltaNet: o = x * rsqrt(mean x^2 + eps) * w * sigmoid(z), per head of 128; block per head */
extern "C" __global__ void k_gnorm(const float* x, const float* z, const float* w, float* o, float eps) {
    __shared__ float sh32[32];
    const int h = blockIdx.x, j = threadIdx.x;
    const float v = x[h * 128 + j];
    const float r = rsqrtf(block_sum(v * v, sh32) / 128.f + eps);
    o[h * 128 + j] = v * r * w[j] * sigm(z[h * 128 + j]);
}

/* ---------------------------------------------------------------- the multi-token window's steps, T tokens a launch
   Token t = blockIdx.z (or a loop inside, for the steps that carry state from token to token); each token's
   operations are those of the one-token kernel above, in the same order, so a window's token t has the bits of the
   t-th one-token step. ld*: the distance between two tokens' rows (floats). */
extern "C" __global__ void k_rms_gT(const float* h, const float* w, float* out, int n, float eps, long long ld) {
    __shared__ float sh32[32];
    const long long o = (long long)blockIdx.z * ld;
    const float* x = h + o + (size_t)blockIdx.x * n;
    float v = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) v += x[c] * x[c];
    const float r = rsqrtf(block_sum(v, sh32) / n + eps);
    for (int c = threadIdx.x; c < n; c += blockDim.x)
        out[o + (size_t)blockIdx.x * n + c] = x[c] * r * w[(size_t)blockIdx.x * n + c];
}
extern "C" __global__ void k_mix_outT(const float* u, const float* hn, const float* inj, float* x, float* wr, int G,
                                      int n, float hc, long long ldu, long long ldhn, long long ldinj, long long ldx,
                                      long long ldwr) {
    const int t = blockIdx.z;
    u += t * ldu; hn += t * ldhn; x += t * ldx;
    if (inj) { inj += t * ldinj; wr += t * ldwr; }
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) {
        float acc = 0.f;
        for (int g = 0; g < G; g++) acc += sigm(u[(size_t)g * n + c]) * hn[(size_t)g * n + c];
        x[c] = acc / G;
    }
    if (inj && blockIdx.x == 0 && threadIdx.x < G) wr[threadIdx.x] = 2.f * sigm(inj[threadIdx.x] / hc);
}
extern "C" __global__ void k_writeT(const float* hin, const float* y, const float* y2, const float* sdot,
                                    const float* wr, float* out, int G, int n, long long ldh, long long ldy,
                                    long long ldwr) {
    const int t = blockIdx.z;
    hin += t * ldh; out += t * ldh; y += t * ldy; wr += t * ldwr;
    if (y2) { y2 += t * ldy; sdot += t; }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= G * n) return;
    const int g = i / n, c = i % n;
    float v = y[c];
    if (y2) v += sigm(sdot[0]) * y2[c];
    out[i] = hin[i] + v * wr[g];
}
extern "C" __global__ void k_dotT(const float* x, const float* v, int n, float* out, long long ldx) {
    __shared__ float sh32[32];
    x += blockIdx.z * ldx;
    float a = 0.f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) a += x[c] * v[c];
    a = block_sum(a, sh32);
    if (threadIdx.x == 0) out[blockIdx.z] = a;
}
extern "C" __global__ void k_routerT(const float* logit, int n, int k, int* idx, float* top) {
    __shared__ float sh32[32], pv[1024], tv[32];
    __shared__ int ti[32];
    logit += (size_t)blockIdx.z * n; idx += (size_t)blockIdx.z * k; top += (size_t)blockIdx.z * k;
    const int t = threadIdx.x;
    const float x = t < n ? logit[t] : __int_as_float(0xff800000);
    float m = x;
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    __syncthreads();
    if ((t & 31) == 0) sh32[t >> 5] = m;
    __syncthreads();
    m = __int_as_float(0xff800000);
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) m = fmaxf(m, sh32[i]);
    const float e = t < n ? expf(x - m) : 0.f;
    const float sum = block_sum(e, sh32);
    pv[t] = t < n ? e / sum : -1.f;
    __syncthreads();
    if (XG_ROUTER2 && k <= 32 && (int)(blockDim.x >> 5) * k <= 512) {
        __shared__ float cv_[512];
        __shared__ int ci_[512];
        router_topk2(pv, k, idx, top, cv_, ci_);
    } else
    for (int r = 0; r < k; r++) {
        float bv = pv[t];
        int bi = t;
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        __syncthreads();
        if ((t & 31) == 0) { tv[t >> 5] = bv; ti[t >> 5] = bi; }
        __syncthreads();
        if (t == 0) {
            float b = tv[0];
            int bi2 = ti[0];
            for (int i = 1; i < (int)(blockDim.x >> 5); i++)
                if (tv[i] > b || (tv[i] == b && ti[i] < bi2)) { b = tv[i]; bi2 = ti[i]; }
            idx[r] = bi2;
            top[r] = b;
            pv[bi2] = -1.f;
        }
        __syncthreads();
    }
    if (t == 0) {
        float s = 0.f;
        for (int r = 0; r < k; r++) s += top[r];
        for (int r = 0; r < k; r++) top[r] = top[r] / s;
    }
}
extern "C" __global__ void k_combine_selT(int nsel, int K, const float* w, const int* idx, const int* slotmap,
                                          const float* hotp, const float* coldp, float* out) {
    const int t = blockIdx.z;
    w += (size_t)t * nsel; idx += (size_t)t * nsel;
    hotp += (size_t)t * nsel * K; coldp += (size_t)t * nsel * K; out += (size_t)t * K;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float acc = 0.f;
    for (int s = 0; s < nsel; s++) {
        const float v = slotmap[idx[s]] >= 0 ? hotp[(size_t)s * K + c] : coldp[(size_t)s * K + c];
        acc = __fadd_rn(acc, __fmul_rn(w[s], v));
    }
    out[c] = acc;
}
/* the causal conv for T tokens in order (each thread owns its channel); the state after token t -> hist[t] */
extern "C" __global__ void k_conv1T(const float* x, float* state, const float* w, float* y, int C, int kw, int T,
                                    long long ldx, float* hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    for (int t = 0; t < T; t++) {
        const float* xt = x + t * ldx;
        float acc = 0.f;
        for (int j = 0; j < kw - 1; j++) acc += w[(size_t)c * kw + j] * state[(size_t)j * C + c];
        acc += w[(size_t)c * kw + kw - 1] * xt[c];
        for (int j = 0; j < kw - 2; j++) state[(size_t)j * C + c] = state[(size_t)(j + 1) * C + c];
        state[(size_t)(kw - 2) * C + c] = xt[c];
        y[(size_t)t * C + c] = silu_(acc);
        for (int j = 0; j < kw - 1; j++) hist[((size_t)t * (kw - 1) + j) * C + c] = state[(size_t)j * C + c];
    }
}
/* the delta rule for T tokens in order: k_gdn1's steps per token; the state after token t -> Shist[t] */
extern "C" __global__ void k_gdn1T(int nk, const float* conv, const float* b, const float* a, const float* ssm_a,
                                   const float* dtb, float* S, float* out, int T, long long ldconv, long long ldba,
                                   long long ldout, float* Shist) {
    __shared__ float sq[128], sk[128], red[4][32], nrm[2];
    const int h = blockIdx.x, cc = blockIdx.y, t = threadIdx.x, jl = t & 31, r = t >> 5;
    const int j = cc * 32 + jl, kh = h % nk;
    const int nv = gridDim.x;
    float* Sh = S + (size_t)h * 128 * 128;
    for (int tok = 0; tok < T; tok++) {
        const float* cv = conv + tok * ldconv;
        __syncthreads();
        sq[t] = cv[kh * 128 + t];
        sk[t] = cv[nk * 128 + kh * 128 + t];
        __syncthreads();
        if (r < 2) {
            const float* src = r == 0 ? sq : sk;
            float v = 0.f;
            for (int i = jl; i < 128; i += 32) v += src[i] * src[i];
            v = warp_sum(v);
            if (jl == 0) nrm[r] = v;
        }
        __syncthreads();
        const float qs = rsqrtf(nrm[0] + 1e-6f) / sqrtf(128.f), ks = rsqrtf(nrm[1] + 1e-6f);
        const float bt = sigm(b[tok * ldba + h]);
        const float xa = a[tok * ldba + h] + dtb[h];
        const float dec = expf(ssm_a[h] * (xa > 20.f ? xa : log1pf(expf(xa))));
        float sv[32], pm = 0.f;
        for (int ii = 0; ii < 32; ii++) {
            const int i = r * 32 + ii;
            sv[ii] = Sh[i * 128 + j] * dec;
            pm += sv[ii] * (sk[i] * ks);
        }
        red[r][jl] = pm;
        __syncthreads();
        const float mem = red[0][jl] + red[1][jl] + red[2][jl] + red[3][jl];
        const float delta = (cv[2 * nk * 128 + h * 128 + j] - mem) * bt;
        __syncthreads();
        float po = 0.f;
        float* Hh = Shist + ((size_t)tok * nv + h) * 128 * 128;
        for (int ii = 0; ii < 32; ii++) {
            const int i = r * 32 + ii;
            sv[ii] += (sk[i] * ks) * delta;
            po += sv[ii] * (sq[i] * qs);
            Sh[i * 128 + j] = sv[ii];
            Hh[i * 128 + j] = sv[ii];
        }
        red[r][jl] = po;
        __syncthreads();
        if (r == 0) out[tok * ldout + h * 128 + j] = red[0][jl] + red[1][jl] + red[2][jl] + red[3][jl];
    }
}
extern "C" __global__ void k_gnormT(const float* x, const float* z, const float* w, float* o, float eps, long long ldx,
                                    long long ldz) {
    __shared__ float sh32[32];
    x += blockIdx.z * ldx; o += blockIdx.z * ldx; z += blockIdx.z * ldz;
    const int h = blockIdx.x, j = threadIdx.x;
    const float v = x[h * 128 + j];
    const float r = rsqrtf(block_sum(v * v, sh32) / 128.f + eps);
    o[h * 128 + j] = v * r * w[j] * sigm(z[h * 128 + j]);
}

/* the whole matrix as fp32 (rows x cols, row-major), for prompt chunks: one dequantization, then a GEMM, instead of
   reading the quantized weights again for every 16 tokens */
extern "C" __global__ void k_deq(int ty, const uint8_t* W, int rows, int cols, long long rb, float* out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    out[i] = deq(ty, W + (size_t)r * rb, c >> 5, c & 31);
}

/* k_deq's values rounded to fp16 (round to nearest even): the weights of the fp16 prompt GEMMs */
extern "C" __global__ void k_deq16(int ty, const uint8_t* W, int rows, int cols, long long rb, unsigned short* out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rows * cols) return;
    const int r = (int)(i / cols), c = (int)(i % cols);
    unsigned short hv;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hv) : "f"(deq(ty, W + (size_t)r * rb, c >> 5, c & 31)));
    out[i] = hv;
}

/* ---------------------------------------------------------------- the draft chain (MTP head, on the card) */
/* the draft's pick from logits [n] of the draft vocabulary: the largest (the lowest index on a tie), its softmax
   probability over the n, its index sel[0] and the token it stands for, tok[0] = map[sel[0]]. One block of 1024. */
extern "C" __global__ void k_pick(const float* x, int n, const int* map, int* tok, int* sel, float* prob) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float sh32[32];
    float bv = -3.0e38f;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        if (v > bv) { bv = v; bi = i; }                   /* i grows: the first of equal values stays */
    }
    for (int o = 16; o; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, bv, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v2 > bv || (v2 == bv && i2 < bi)) { bv = v2; bi = i2; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    bv = sv[0]; bi = si[0];
    for (int k = 1; k < nw; k++)
        if (sv[k] > bv || (sv[k] == bv && si[k] < bi)) { bv = sv[k]; bi = si[k]; }
    float e = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) e += expf(x[i] - bv);
    e = block_sum(e, sh32);
    if (threadIdx.x == 0) {
        sel[0] = bi;
        tok[0] = map[bi];
        prob[0] = 1.f / e;
    }
}

/* row sel[0] of W (rows of rb bytes, cols values) as fp32: the next draft step's token embedding */
extern "C" __global__ void k_deq_row(int ty, const uint8_t* W, int cols, long long rb, const int* sel, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    out[c] = deq(ty, W + (size_t)sel[0] * rb, c >> 5, c & 31);
}

/* ---------------------------------------------------------------- Gated DeltaNet */
/* q, k [T][nk][128] (L2-normed, q scaled), v [T][nv][128], g, beta [T][nv]; S [nv][128][128] (in and out);
   out [T][nv][128]. Block h, thread j: column j of S of V head h. */
extern "C" __global__ void k_gdn(int T, int nk, int nv, const float* q, const float* k, const float* v, const float* g,
                      const float* beta, float* S, float* out) {
    const int h = blockIdx.x, j = threadIdx.x, kh = h % nk;
    __shared__ float sk[2][128], sq[2][128];
    float s[128];
    float* Sh = S + (size_t)h * 128 * 128;
    for (int i = 0; i < 128; i++) s[i] = Sh[i * 128 + j];
    /* the next token's inputs load during this token's work (the same operations in the same order as before) */
    float nk_ = 0.f, nq_ = 0.f, nv_ = 0.f, ng_ = 0.f, nb_ = 0.f;
    if (T > 0) { nk_ = k[(size_t)kh * 128 + j]; nq_ = q[(size_t)kh * 128 + j]; nv_ = v[(size_t)h * 128 + j]; ng_ = g[h]; nb_ = beta[h]; }
    for (int t = 0; t < T; t++) {
        const int pb = t & 1;
        sk[pb][j] = nk_;
        sq[pb][j] = nq_;
        const float vj = nv_, gg = ng_, bt = nb_;
        __syncthreads();
        if (t + 1 < T) {
            nk_ = k[((size_t)(t + 1) * nk + kh) * 128 + j];
            nq_ = q[((size_t)(t + 1) * nk + kh) * 128 + j];
            nv_ = v[((size_t)(t + 1) * nv + h) * 128 + j];
            ng_ = g[(size_t)(t + 1) * nv + h];
            nb_ = beta[(size_t)(t + 1) * nv + h];
        }
        const float dec = expf(gg);
        float mem = 0.f;
        for (int i = 0; i < 128; i++) { s[i] *= dec; mem += s[i] * sk[pb][i]; }
        const float delta = (vj - mem) * bt;
        float o = 0.f;
        for (int i = 0; i < 128; i++) { s[i] += sk[pb][i] * delta; o += s[i] * sq[pb][i]; }
        out[((size_t)t * nv + h) * 128 + j] = o;
    }
    for (int i = 0; i < 128; i++) Sh[i * 128 + j] = s[i];
}

/* k_gdn with each V head's 128 state columns split over 4 blocks (grid (nv, 4)): block (h, cc), thread t keeps
   column 32 cc + (t & 31) for rows 32 (t >> 5) .. + 31 in registers; the two row sums of a token (mem, o) are taken
   per 32 rows and then added in row-group order, as k_gdn1T does (the window's order, not k_gdn's). 4x the blocks
   and a quarter of the serial chain per token. Same inputs and outputs as k_gdn. */
extern "C" __global__ void k_gdn4(int T, int nk, int nv, const float* q, const float* k, const float* v,
                                  const float* g, const float* beta, float* S, float* out) {
    __shared__ float sq[2][128], sk[2][128], red[2][4][32];
    const int h = blockIdx.x, t = threadIdx.x, jl = t & 31, r = t >> 5, j = blockIdx.y * 32 + jl, kh = h % nk;
    float s[32];
    float* Sh = S + (size_t)h * 128 * 128;
#pragma unroll
    for (int ii = 0; ii < 32; ii++) s[ii] = Sh[(r * 32 + ii) * 128 + j];
    /* the next token's inputs are loaded during this token's work (registers), so their latency is hidden */
    float nk_ = 0.f, nq_ = 0.f, nv_ = 0.f, ng_ = 0.f, nb_ = 0.f;
    if (T > 0) {
        nk_ = k[(size_t)kh * 128 + t]; nq_ = q[(size_t)kh * 128 + t];
        nv_ = v[(size_t)h * 128 + j]; ng_ = g[h]; nb_ = beta[h];
    }
    for (int tok = 0; tok < T; tok++) {
        const int pb = tok & 1;
        sk[pb][t] = nk_;
        sq[pb][t] = nq_;
        const float vj = nv_, gg = ng_, bt = nb_;
        __syncthreads();
        if (tok + 1 < T) {
            nk_ = k[((size_t)(tok + 1) * nk + kh) * 128 + t];
            nq_ = q[((size_t)(tok + 1) * nk + kh) * 128 + t];
            nv_ = v[((size_t)(tok + 1) * nv + h) * 128 + j];
            ng_ = g[(size_t)(tok + 1) * nv + h];
            nb_ = beta[(size_t)(tok + 1) * nv + h];
        }
        const float dec = expf(gg);
        float pm = 0.f;
#pragma unroll
        for (int ii = 0; ii < 32; ii++) { s[ii] *= dec; pm += s[ii] * sk[pb][r * 32 + ii]; }
        red[0][r][jl] = pm;
        __syncthreads();
        const float mem = red[0][0][jl] + red[0][1][jl] + red[0][2][jl] + red[0][3][jl];
        const float delta = (vj - mem) * bt;
        float po = 0.f;
#pragma unroll
        for (int ii = 0; ii < 32; ii++) { s[ii] += sk[pb][r * 32 + ii] * delta; po += s[ii] * sq[pb][r * 32 + ii]; }
        red[1][r][jl] = po;
        __syncthreads();
        if (r == 0) out[((size_t)tok * nv + h) * 128 + j] = red[1][0][jl] + red[1][1][jl] + red[1][2][jl] + red[1][3][jl];
    }
#pragma unroll
    for (int ii = 0; ii < 32; ii++) Sh[(r * 32 + ii) * 128 + j] = s[ii];
}

/* the prompt's Gated DeltaNet inputs (KINDI_QXG_PFUSE=1), two launches instead of PyTorch's ~15 element-wise ops:
   k_gdn_conv: y[t][c] = silu(sum_j w[c][j] x[t - kw + 1 + j][c]) over the causal window (rows before the chunk from
   prev [kw - 1][C], zeros when prev is null); q channels to qk [T][nk 128], k channels to qk + T nk 128, v channels
   to v [T][C - 2 nk 128];
   the last kw - 1 input rows to state [kw - 1][C] (after all reads: state may not alias prev).
   k_gdn_norm: block (head, t): q heads (head < nk) L2-normed and / sqrt(128), k heads L2-normed, in place; block
   (0, t) also makes beta = sigmoid(b) and g = ssm_a * softplus(a + dt_bias) for the nv V heads. */
extern "C" __global__ void k_gdn_conv(const float* x, const float* prev, const float* w, float* qk, float* v,
                                      float* state, int T, int C, int kw, int cqk) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)T * C) return;
    const int t = (int)(i / C), c = (int)(i % C);
    float acc = 0.f;
    for (int j = 0; j < kw; j++) {
        const int tt = t - (kw - 1) + j;
        const float xv = tt >= 0 ? x[(size_t)tt * C + c] : (prev ? prev[(size_t)(kw - 1 + tt) * C + c] : 0.f);
        acc += w[(size_t)c * kw + j] * xv;
    }
    const float y = silu_(acc);
    const int hq = cqk / 2;                                  /* q channels, then k channels: separate [T][nk 128] */
    if (c < hq) qk[(size_t)t * hq + c] = y;
    else if (c < cqk) qk[(size_t)T * hq + (size_t)t * hq + c - hq] = y;
    else v[(size_t)t * (C - cqk) + c - cqk] = y;
}
extern "C" __global__ void k_gdn_state(const float* x, const float* prev, float* state, int T, int C, int kw) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)(kw - 1) * C) return;
    const int r = (int)(i / C), c = (int)(i % C), tt = T - (kw - 1) + r;
    state[i] = tt >= 0 ? x[(size_t)tt * C + c] : (prev ? prev[(size_t)(kw - 1 + tt) * C + c] : 0.f);
}
extern "C" __global__ void k_gdn_norm(float* qk, int nk, const float* b, const float* a, const float* ssm_a,
                                      const float* dtb, float* beta, float* g, int nv) {
    __shared__ float sh32[32];
    const int hd = blockIdx.x, t = blockIdx.y, j = threadIdx.x;
    float* p = hd < nk ? qk + ((size_t)t * nk + hd) * 128 : qk + (size_t)gridDim.y * nk * 128 + ((size_t)t * nk + hd - nk) * 128;
    const float v = p[j];
    const float r = rsqrtf(block_sum(v * v, sh32) + 1e-6f);
    p[j] = hd < nk ? v * r / sqrtf(128.f) : v * r;
    if (hd == 0 && j < nv) {
        beta[(size_t)t * nv + j] = sigm(b[(size_t)t * nv + j]);
        const float xa = a[(size_t)t * nv + j] + dtb[j];
        g[(size_t)t * nv + j] = ssm_a[j] * (xa > 20.f ? xa : log1pf(expf(xa)));
    }
}

/* ---------------------------------------------------------------- the PCIe share of the misses (item 7) */
/* A share of each layer's missed experts is copied from the page-locked expert pages (mapped into the card's
   address space) into spare slots of the layer's class buffer and computed on the card; the CPU computes the rest.
   The rule, the same on both sides (qx_ma_fn in qx_cpu.c): walk the picks in order (token by token, slot by slot);
   a missed expert not yet taken is taken while fewer than min(nf[l], smax) are; every pick of a taken expert is the
   card's. The card's and the CPU's expert rows have the same bits, so the share changes the speed only.
   k_fetch_plan: one thread; marks the taken experts in the slot table (spare0 + k) and lists them (count at
   list[smax]). k_fetch_copy: block (x, k, p) copies chunk x of part p (gate, up, down) of taken expert k.
   k_fetch_reset: the slot table back to -1 after the layer's combine. */
extern "C" __global__ void k_fetch_plan(const int* ids, int n, int* slotmap, const int* nf, int l, int smax, int spare0,
                                        int* list) {
    if (threadIdx.x || blockIdx.x) return;
    int want = *(volatile const int*)(nf + l);
    if (want > smax) want = smax;
    int c = 0;
    for (int i = 0; i < n && c < want; i++) {
        const int e = ids[i];
        if (slotmap[e] >= 0) continue;            /* on the card */
        int dup = 0;
        for (int q = 0; q < c && !dup; q++) dup = list[q] == e;
        if (!dup) list[c++] = e;                 /* a missed expert not taken yet */
    }
    list[smax] = c;
    (void)spare0;
    __threadfence();
}
/* the taken experts into the slot table (spare0 + k), once their copies have landed */
extern "C" __global__ void k_fetch_apply(const int* list, int smax, int* slotmap, int spare0) {
    const int k = threadIdx.x;
    if (k < list[smax]) slotmap[list[k]] = spare0 + k;
}
extern "C" __global__ void k_fetch_copy(const int* list, int smax, const uint8_t* s0, const uint8_t* s1,
                                        const uint8_t* s2, long long n0, long long n1, long long n2, uint8_t* buf,
                                        long long rowbytes, int spare0, long long chunk) {
    const int k = blockIdx.y, p = blockIdx.z;
    if (k >= list[smax]) return;
    const long long e = list[k];
    const long long n = p == 0 ? n0 : p == 1 ? n1 : n2;
    const uint8_t* src = (p == 0 ? s0 : p == 1 ? s1 : s2) + e * n;
    uint8_t* dst = buf + (long long)(spare0 + k) * rowbytes + (p == 0 ? 0 : p == 1 ? n0 : n0 + n1);
    const long long a = (long long)blockIdx.x * chunk;
    if (a >= n) return;
    const long long b = a + chunk < n ? a + chunk : n;
    if ((((size_t)src | (size_t)dst | (size_t)a | (size_t)b) & 15) == 0) {
        const uint4* s = (const uint4*)(src + a);
        uint4* d = (uint4*)(dst + a);
        for (long long i = threadIdx.x; i < (b - a) / 16; i += blockDim.x) d[i] = s[i];
    } else {
        for (long long i = a + threadIdx.x; i < b; i += blockDim.x) dst[i] = src[i];
    }
}
extern "C" __global__ void k_fetch_reset(const int* list, int smax, int* slotmap) {
    const int k = threadIdx.x;
    if (k < list[smax]) slotmap[list[k]] = -1;
}

/* h = silu(g) * v for the pairs (k_gate_up's epilogue) */
extern "C" __global__ void k_silu_mul(const float* g, const float* v, float* h, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float sl = __fdiv_rn(g[i], __fadd_rn(1.f, qx_expf(-g[i])));
    h[i] = __fmul_rn(sl, v[i]);
}
/* ---------------------------------------------------------------- experts on int8 tensor cores (item G, sm_80+)
   The prompt's expert rows with mma.m16n8k16 (s8 x s8 -> s32): a warp takes 16 rows of one expert, eight pairs at
   a time. Per 16 values the products are exact integers; each is times its row's sub-scale m (the two halves of an
   IQ2_S sub-block have their own m) and summed per weight block, then one float step per weight block,
   A = A + (float)I * (dw * xs[token][block * per / ab]), in block order: the steps of qx_cpu.c's 256-style rows
   (gate/up with ab = 256, down with ab = 64 on KINDI_QX_DN=64), so a pair's value has the CPU's bits.
   in_pair = 1 (the down): pair p's input is row p (its h), its value goes to out[(ptok[p] * nsel + pslot[p]) * ldo
   + r]; in_pair = 0 (gate/up): the input is the token's row ptok[p], the value goes to out[p * ldo + r].
   Block: 4 warps = 64 rows; grid (rows / 64, experts). */
#define XT_NG 8                                    /* groups of 8 pairs per pass over a row */
extern "C" __global__ void k_xmm_tc(int ty, int cols, int rows, int ab, const uint8_t* const* mats, int part,
                                    const int* pstart, const int* ptok, const int* pslot, int nsel, int in_pair,
                                    const int8_t* xq, const float* xs, float* out, int ldo) {
    __shared__ int tile[4][16][8];
    __shared__ int msm[4][16][2];
    __shared__ float dwsm[4][16];
    __shared__ float xsc[8 * XT_NG];                  /* their activation scales for this weight block */
    __shared__ int xt[8 * XT_NG][64 + 1];               /* the pass's inputs of this weight block (<= 256 values), as
                                                          words; loaded once for the block's 4 warps, coalesced */
    const int u = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rw = blockIdx.x * 64 + 16 * w;            /* the warp's first row */
    const int per = blk_vals(ty), bytes = blk_bytes(ty), sub = per / 32, nbk = cols / per, nxs = cols / ab;
    const long long rb = (long long)(cols / per) * bytes;
    const uint8_t* M = mats[3 * u + part];
    const int p_lo = pstart[u], p_hi = pstart[u + 1];
#if __CUDA_ARCH__ >= 800
    for (int pp = p_lo; pp < p_hi; pp += 8 * XT_NG) {
        const int ng = min(XT_NG, (p_hi - pp + 7) / 8);    /* the groups of 8 this pass has (a prompt chunk of 1024
                                                           tokens gives about 20 pairs an expert: 3 of 8 groups) */
        float A[XT_NG][4];
        for (int G = 0; G < XT_NG; G++)
            for (int c = 0; c < 4; c++) A[G][c] = 0.f;
        for (int bk = 0; bk < nbk; bk++) {
            __syncthreads();                            /* the previous block's tile is no longer read */
            const int wpb = per / 4;                    /* words of a pair in this weight block */
            for (int q = threadIdx.x; q < 8 * ng * wpb; q += blockDim.x) {
                const int pr = q / wpb, wd = q % wpb, n = pp + pr;
                xt[pr][wd] = n < p_hi ? *(const int*)(xq + (size_t)(in_pair ? n : ptok[n]) * cols + bk * per + 4 * wd) : 0;
            }
            for (int q = threadIdx.x; q < 8 * XT_NG; q += blockDim.x) {
                const int n = pp + q;
                xsc[q] = n < p_hi ? xs[(size_t)(in_pair ? n : ptok[n]) * nxs + bk * per / ab] : 0.f;
            }
            __syncthreads();
            int I[XT_NG][4];
            for (int G = 0; G < XT_NG; G++) for (int c = 0; c < 4; c++) I[G][c] = 0;
            for (int j = 0; j < sub; j++) {
                for (int k = 0; k < 4; k++) {           /* decode: 16 rows x 8 lanes of 4 codes */
                    const int idx = lane + 32 * k, row = idx >> 3, l8 = idx & 7;
                    int m = 0, c = 0; float dw = 0.f;
                    if (rw + row < rows) c = codes4(ty, M + (size_t)(rw + row) * rb + (size_t)bk * bytes, j, l8, &m, &dw);
                    tile[w][row][l8] = c;
                    if ((l8 & 3) == 0) msm[w][row][l8 >> 2] = m;
                    if (l8 == 0) dwsm[w][row] = dw;
                }
                __syncwarp();
                for (int h = 0; h < 2; h++) {
                    const int a0 = tile[w][g][4 * h + t], a1 = tile[w][g + 8][4 * h + t];
                    const int m0 = msm[w][g][h], m1 = msm[w][g + 8][h];
#pragma unroll
                    for (int G = 0; G < XT_NG; G++) {
                        if (G >= ng) break;
                        const int b0 = xt[8 * G + g][8 * j + 4 * h + t];
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
                                     "{%0,%1,%2,%3};"
                                     : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3) : "r"(a0), "r"(a1), "r"(b0));
                        I[G][0] += m0 * d0; I[G][1] += m0 * d1; I[G][2] += m1 * d2; I[G][3] += m1 * d3;
                    }
                }
                __syncwarp();
            }
            /* the float step: element c is row g (c < 2) or g + 8, pair pp + 8G + 2t + (c & 1) */
            const float dw0 = dwsm[w][g], dw1 = dwsm[w][g + 8];
#pragma unroll
            for (int G = 0; G < XT_NG; G++)
                for (int c = 0; c < 4; c++) {
                    const int n = pp + 8 * G + 2 * t + (c & 1);
                    if (G >= ng || n >= p_hi) continue;
                    const float s = __fmul_rn(c < 2 ? dw0 : dw1, xsc[8 * G + 2 * t + (c & 1)]);
                    A[G][c] = __fadd_rn(A[G][c], __fmul_rn((float)I[G][c], s));
                }
            __syncwarp();
        }
        for (int G = 0; G < XT_NG; G++)
            for (int c = 0; c < 4; c++) {
                const int n = pp + 8 * G + 2 * t + (c & 1), r = rw + g + (c < 2 ? 0 : 8);
                if (n >= p_hi || r >= rows) continue;
                if (in_pair) out[((size_t)ptok[n] * nsel + pslot[n]) * ldo + r] = A[G][c];
                else out[(size_t)n * ldo + r] = A[G][c];
            }
    }
#endif
}

/* ---------------------------------------------------------------- tiled int8 expert GEMM (item G, sm_80+)
   The same value rows as k_xmm_tc (and so as qx_cpu.c), computed as a tiled GEMM. A block takes 128 rows of one
   expert and up to 64 of its pairs (one entry of a tile list made on the host: expert, first pair, end pair), so a
   ragged set of experts is one launch of (rows / 128) x tiles blocks. The k dimension runs in slices of 64 values:
   - the weights of a slice are decoded once per block into int8 rows in shared memory (a thread decodes one row's
     32-value sub-block; its raw bytes are loaded one slice ahead into registers; the grids are in shared memory);
   - the slice's inputs (64 pairs x 64 bytes) and activation steps arrive with cp.async, three slices deep;
   - both tiles have their 16-byte chunks swizzled by row, so ldmatrix reads them without bank conflicts;
   - 8 warps, each 32 rows x 32 pairs: per 32 values 2 A and 2 B ldmatrix.x4 feed 8 mma.
   Integer sums: per 16 values (IQ2_S, two m per sub-block) or per 32 (one m) the exact s8 x s8 sum is taken times the
   row's m; the formats without m (Q2_0, IQ4_NL) sum a whole weight block on the tensor cores. Integer sums are exact
   in any order, so I equals k_xmm_tc's I; then per weight block, in block order, k_xmm_tc's float step
   A = A + (float)I * (dw * xs[pair][block * per / ab]). Bits: k_xmm_tc's, i.e. the CPU's. */
#define XG_BM 128
#define XG_BN 64
static __device__ __forceinline__ uint32_t xg_sa(const void* p) {
    uint64_t a;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p));
    return (uint32_t)a;
}
static __device__ __forceinline__ uint32_t xg_u16(const uint8_t* p) { return *(const uint16_t*)p; }
static __device__ __forceinline__ uint32_t xg_u32(const uint8_t* p) { return xg_u16(p) | (xg_u16(p + 2) << 16); }
/* 4 magnitude bytes (1 .. 127, never 0: true of all three grids) with the signs of the low 4 bits of s; the same
   bytes as pack4: a negative byte is 256 - v = (v ^ 255) + 1, and no carry leaves a byte since v >= 1 */
static __device__ __forceinline__ uint32_t xg_sign(uint32_t mag, uint32_t s) {
    const uint32_t lo = ((s & 15u) * 0x00204081u) & 0x01010101u;
    return (mag ^ (lo * 255u)) + lo;
}
/* 4 nibbles (one per byte) to the bytes of c_lv4: the low 8 entries from (t0, t1), the high 8 from (t2, t3) */
static __device__ __forceinline__ uint32_t xg_lv4(uint32_t nb, const uint32_t* t) {
    uint32_t u = nb & 0x07070707u;
    u |= u >> 4;
    const uint32_t sel = (u & 0xFFu) | ((u >> 8) & 0xFF00u);
    const uint32_t lo = __byte_perm(t[0], t[1], sel), hi = __byte_perm(t[2], t[3], sel);
    const uint32_t mk = ((nb >> 3) & 0x01010101u) * 255u;
    return (lo & ~mk) | (hi & mk);
}

/* per format: values and bytes of a weight block, how m goes (0: per 16, 1: per 32, 2: none), the grid words kept in
   shared memory; fetch() loads one 32-value sub-block's raw bytes (2-byte loads: blocks are only 2-aligned), decode()
   makes its 8 code words (codes4's for i = 0..7), the m of its two halves and dw */
template <int TY> struct XG;
template <> struct XG<T_IQ2_S> {
    enum { PER = 256, BYTES = 82, MODE = 0, GW = 2048, NR = 4 };
    static __device__ __forceinline__ const uint32_t* grid() { return (const uint32_t*)nbq_iq2s_grid; }
    static __device__ __forceinline__ void fetch(const uint8_t* b, int j, uint32_t* r) {
        r[0] = xg_u16(b); r[1] = xg_u32(b + 2 + 4 * j); r[2] = xg_u32(b + 34 + 4 * j);
        r[3] = b[66 + j] | ((uint32_t)b[74 + j] << 8);
    }
    static __device__ __forceinline__ void decode(const uint32_t* r, int j, const uint32_t* gt, const uint32_t* lv,
                                                  uint32_t* q, int* m, float* dw) {
        const uint32_t qh = r[3] & 255u, sc = (r[3] >> 8) & 255u;
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t idx = ((r[1] >> (8 * l)) & 255u) | (((qh >> (2 * l)) & 3u) << 8), s = (r[2] >> (8 * l)) & 255u;
            const uint2 gg = *(const uint2*)&gt[2 * idx];
            q[2 * l] = xg_sign(gg.x, s);
            q[2 * l + 1] = xg_sign(gg.y, s >> 4);
        }
        m[0] = 2 * (int)(sc & 15u) + 1; m[1] = 2 * (int)(sc >> 4) + 1;
        *dw = __fmul_rn(h2f((unsigned short)r[0]), 0.125f);
    }
};
template <> struct XG<T_IQ3_XXS> {
    enum { PER = 256, BYTES = 98, MODE = 1, GW = 256, NR = 4 };
    static __device__ __forceinline__ const uint32_t* grid() { return nbq_iq3xxs_grid; }
    static __device__ __forceinline__ void fetch(const uint8_t* b, int j, uint32_t* r) {
        r[0] = xg_u16(b); r[1] = xg_u32(b + 2 + 8 * j); r[2] = xg_u32(b + 6 + 8 * j); r[3] = xg_u32(b + 66 + 4 * j);
    }
    static __device__ __forceinline__ void decode(const uint32_t* r, int j, const uint32_t* gt, const uint32_t* lv,
                                                  uint32_t* q, int* m, float* dw) {
        const uint32_t aux = r[3];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t idx = (r[1 + (i >> 2)] >> (8 * (i & 3))) & 255u;
            const uint32_t s7 = (aux >> (7 * (i >> 1))) & 127u, sg = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
            q[i] = xg_sign(gt[idx], sg >> (4 * (i & 1)));
        }
        m[0] = m[1] = 2 * (int)(aux >> 28) + 1;
        *dw = __fmul_rn(h2f((unsigned short)r[0]), 0.25f);
    }
};
template <> struct XG<T_IQ3_S> {
    enum { PER = 256, BYTES = 110, MODE = 1, GW = 512, NR = 4 };
    static __device__ __forceinline__ const uint32_t* grid() { return nbq_iq3s_grid; }
    static __device__ __forceinline__ void fetch(const uint8_t* b, int j, uint32_t* r) {
        r[0] = xg_u16(b) | ((uint32_t)b[106 + (j >> 1)] << 16) | ((uint32_t)b[66 + j] << 24);
        r[1] = xg_u32(b + 2 + 8 * j); r[2] = xg_u32(b + 6 + 8 * j); r[3] = xg_u32(b + 74 + 4 * j);
    }
    static __device__ __forceinline__ void decode(const uint32_t* r, int j, const uint32_t* gt, const uint32_t* lv,
                                                  uint32_t* q, int* m, float* dw) {
        const uint32_t qh = r[0] >> 24, sc = (r[0] >> 16) & 255u;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t idx = ((r[1 + (i >> 2)] >> (8 * (i & 3))) & 255u) | (((qh >> i) & 1u) << 8);
            q[i] = xg_sign(gt[idx], (r[3] >> (8 * (i >> 1))) >> (4 * (i & 1)));
        }
        m[0] = m[1] = 2 * (int)((sc >> (4 * (j & 1))) & 15u) + 1;
        *dw = h2f((unsigned short)(r[0] & 0xFFFFu));
    }
};
template <> struct XG<T_IQ4_XS> {
    enum { PER = 256, BYTES = 136, MODE = 1, GW = 1, NR = 6 };
    static __device__ __forceinline__ const uint32_t* grid() { return 0; }
    static __device__ __forceinline__ void fetch(const uint8_t* b, int j, uint32_t* r) {
        r[0] = xg_u16(b) | ((uint32_t)b[4 + (j >> 1)] << 16); r[1] = xg_u16(b + 2);
#pragma unroll
        for (int k = 0; k < 4; k++) r[2 + k] = xg_u32(b + 8 + 16 * j + 4 * k);
    }
    static __device__ __forceinline__ void decode(const uint32_t* r, int j, const uint32_t* gt, const uint32_t* lv,
                                                  uint32_t* q, int* m, float* dw) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            q[k] = xg_lv4(r[2 + k] & 0x0F0F0F0Fu, lv);
            q[4 + k] = xg_lv4((r[2 + k] >> 4) & 0x0F0F0F0Fu, lv);
        }
        const uint32_t sl = r[0] >> 16;
        m[0] = m[1] = (int)(((sl >> (4 * (j & 1))) & 15u) | (((r[1] >> (2 * j)) & 3u) << 4)) - 32;
        *dw = h2f((unsigned short)(r[0] & 0xFFFFu));
    }
};
template <> struct XG<T_IQ4_NL> {
    enum { PER = 32, BYTES = 18, MODE = 2, GW = 1, NR = 5 };
    static __device__ __forceinline__ const uint32_t* grid() { return 0; }
    static __device__ __forceinline__ void fetch(const uint8_t* b, int j, uint32_t* r) {
        r[0] = xg_u16(b);
#pragma unroll
        for (int k = 0; k < 4; k++) r[1 + k] = xg_u32(b + 2 + 4 * k);
    }
    static __device__ __forceinline__ void decode(const uint32_t* r, int j, const uint32_t* gt, const uint32_t* lv,
                                                  uint32_t* q, int* m, float* dw) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
            q[k] = xg_lv4(r[1 + k] & 0x0F0F0F0Fu, lv);
            q[4 + k] = xg_lv4((r[1 + k] >> 4) & 0x0F0F0F0Fu, lv);
        }
        m[0] = m[1] = 1;
        *dw = h2f((unsigned short)r[0]);
    }
};
template <> struct XG<T_Q2_0> {
    enum { PER = 64, BYTES = 18, MODE = 2, GW = 1, NR = 3 };
    static __device__ __forceinline__ const uint32_t* grid() { return 0; }
    static __device__ __forceinline__ void fetch(const uint8_t* b, int j, uint32_t* r) {
        r[0] = xg_u16(b); r[1] = xg_u32(b + 2 + 8 * j); r[2] = xg_u32(b + 6 + 8 * j);
    }
    static __device__ __forceinline__ void decode(const uint32_t* r, int j, const uint32_t* gt, const uint32_t* lv,
                                                  uint32_t* q, int* m, float* dw) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint32_t x = (r[1 + (i >> 2)] >> (8 * (i & 3))) & 255u;
            const uint32_t v = (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u;   /* 2-bit field k -> byte k */
            q[i] = ((v | 0x80808080u) - 0x01010101u) ^ 0x80808080u;                 /* bytewise v - 1 */
        }
        m[0] = m[1] = 1;
        *dw = h2f((unsigned short)r[0]);
    }
};

static __device__ __forceinline__ void xg_mma16(int* d, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}
static __device__ __forceinline__ void xg_mma32(int* d, const uint32_t* a, const uint32_t* b, int c0, int c1, int c2,
                                                int c3) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%11,%12,%13};"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3));
}
static __device__ __forceinline__ void xg_ldm4(uint32_t* r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
static __device__ __forceinline__ void xg_cp16(uint32_t dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(n));
}
static __device__ __forceinline__ void xg_cp4(uint32_t dst, const void* src, int n) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(dst), "l"(src), "r"(n));
}
static __device__ __forceinline__ void xg_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
static __device__ __forceinline__ void xg_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
static __device__ __forceinline__ int xg_c4(const int4& v, int k) { return k == 0 ? v.x : k == 1 ? v.y : k == 2 ? v.z : v.w; }

template <int TY>
static __device__ __forceinline__ void xtile_body(int cols, int ab, const uint8_t* const* mats, int part, int ubase,
                                                  const int4* tiles, const int* ptok, const int* pslot, int nsel,
                                                  int in_pair, const int8_t* xq, const float* xs, float* out, int ldo) {
#if __CUDA_ARCH__ >= 800
    typedef XG<TY> D;
    enum { SUBS = D::PER / 32 };                    /* 32-value sub-blocks of a weight block */
    __shared__ __align__(128) uint8_t As[2][XG_BM * 64];
    __shared__ __align__(128) uint8_t Bs[3][XG_BN * 64];
    __shared__ __align__(16) int mts[2][XG_BM][4];   /* m per 16 values of the slice */
    __shared__ float dws[2][XG_BM][2];              /* dw of the row's two sub-blocks of the slice */
    __shared__ float xss[3][XG_BN][2];              /* the pairs' activation steps for the two sub-blocks */
    __shared__ __align__(16) uint32_t gts[D::GW];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int4 tl = tiles[blockIdx.y];
    const int u = tl.x - ubase, p0 = tl.y, np = tl.z - tl.y, rw0 = blockIdx.x * XG_BM;
    const int S = cols / 64, nxs = cols / ab;
    const long long rb = (long long)(cols / D::PER) * D::BYTES;
    if (D::GW > 1) {
        const uint32_t* gsrc = D::grid();
        for (int i = tid; i < D::GW; i += blockDim.x) gts[i] = gsrc[i];
    }
    uint32_t lv[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        lv[k] = (uint32_t)(uint8_t)c_lv4[4 * k] | ((uint32_t)(uint8_t)c_lv4[4 * k + 1] << 8) |
                ((uint32_t)(uint8_t)c_lv4[4 * k + 2] << 16) | ((uint32_t)(uint8_t)c_lv4[4 * k + 3] << 24);
    /* decode role: row drow, sub-block dh of each slice */
    const int drow = tid >> 1, dh = tid & 1;
    const uint8_t* rowp = mats[3 * u + part] + (size_t)(rw0 + drow) * rb;
    /* input role: pair bp, 16-byte chunk bc of each slice; activation steps: pair tid >> 1, sub-block tid & 1 */
    const int bp = tid >> 2, bc = tid & 3;
    const int8_t* bsrc = xq;
    int bn = 0;
    if (bp < np) {
        const int n = p0 + bp;
        bsrc = xq + (size_t)(in_pair ? n : ptok[n]) * cols + 16 * bc;
        bn = 16;
    }
    const uint32_t bdst = xg_sa(&Bs[0][0]) + bp * 64 + ((bc ^ ((bp >> 1) & 3)) << 4);
    const float* xsrc = xs;
    int xn = 0;
    if (tid < 128 && (tid >> 1) < np) {
        const int n = p0 + (tid >> 1);
        xsrc = xs + (size_t)(in_pair ? n : ptok[n]) * nxs;
        xn = 4;
    }
    const uint32_t xdst = xg_sa(&xss[0][0][0]) + 4 * tid;
    uint32_t raw[D::NR];
    /* the warp's tile: rows 32 wm .. + 32, pairs 32 wn .. + 32; nv of its 4 groups of 8 pairs have pairs */
    const int wm = warp & 3, wn = warp >> 2;
    const int nv = min(4, max(0, (np - 32 * wn + 7) / 8));
    int I[2][4][4];
    float A[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) { I[a][b][c] = 0; A[a][b][c] = 0.f; }
    const uint32_t as0 = xg_sa(&As[0][0]), bs0 = xg_sa(&Bs[0][0]);
    const int mi = lane >> 3, rr = lane & 7;
#define XG_ISSUE(s_) do {                                                                                          \
        const int st_ = (s_) % 3;                                                                                  \
        xg_cp16(bdst + st_ * (XG_BN * 64), bsrc + (bn ? 64 * (s_) : 0), bn);                                       \
        if (tid < 128)                                                                                             \
            xg_cp4(xdst + st_ * (XG_BN * 8), xsrc + (xn ? ((2 * (s_) + (tid & 1)) / SUBS) * D::PER / ab : 0), xn); \
    } while (0)
#define XG_FETCH(s_) do {                                                                                          \
        const int js_ = 2 * (s_) + dh;                                                                             \
        D::fetch(rowp + (size_t)(js_ / SUBS) * D::BYTES, js_ % SUBS, raw);                                         \
    } while (0)
#define XG_DECODE(s_) do {                                                                                         \
        const int bf_ = (s_) & 1, js_ = 2 * (s_) + dh;                                                             \
        uint32_t q_[8]; int m_[2]; float dw_;                                                                      \
        D::decode(raw, js_ % SUBS, gts, lv, q_, m_, &dw_);                                                         \
        uint8_t* row_ = &As[bf_][drow * 64];                                                                       \
        const int sw_ = (drow >> 1) & 3;                                                                           \
        *(uint4*)(row_ + (((2 * dh) ^ sw_) << 4)) = make_uint4(q_[0], q_[1], q_[2], q_[3]);                        \
        *(uint4*)(row_ + (((2 * dh + 1) ^ sw_) << 4)) = make_uint4(q_[4], q_[5], q_[6], q_[7]);                    \
        *(int2*)&mts[bf_][drow][2 * dh] = make_int2(m_[0], m_[1]);                                                 \
        dws[bf_][drow][dh] = dw_;                                                                                  \
    } while (0)
    XG_FETCH(0);
    XG_ISSUE(0);
    xg_commit();
    if (S > 1) XG_ISSUE(1);
    xg_commit();
    if (D::GW > 1) __syncthreads();                  /* the grid is in shared memory */
    XG_DECODE(0);
    if (S > 1) XG_FETCH(1);
    for (int s = 0; s < S; s++) {
        xg_wait1();
        __syncthreads();                            /* slice s decoded and landed; slice s - 1 no longer read */
        if (s + 2 < S) XG_ISSUE(s + 2);
        xg_commit();
        const int bf = s & 1, st = s % 3;
        if (nv > 0) {
            int4 mr[2][2];
#pragma unroll
            for (int a = 0; a < 2; a++) {
                mr[a][0] = *(const int4*)&mts[bf][32 * wm + 16 * a + g][0];
                mr[a][1] = *(const int4*)&mts[bf][32 * wm + 16 * a + g + 8][0];
            }
#pragma unroll
            for (int q = 0; q < 2; q++) {
                uint32_t af[2][4], bq[4][2];
#pragma unroll
                for (int a = 0; a < 2; a++) {
                    const int row = 32 * wm + 16 * a + 8 * (mi & 1) + rr, ch = 2 * q + (mi >> 1);
                    xg_ldm4(af[a], as0 + bf * (XG_BM * 64) + row * 64 + ((ch ^ ((row >> 1) & 3)) << 4));
                }
#pragma unroll
                for (int pr = 0; pr < 2; pr++) {
                    const int row = 32 * wn + 8 * (2 * pr + (mi >> 1)) + rr, ch = 2 * q + (mi & 1);
                    uint32_t r4[4];
                    xg_ldm4(r4, bs0 + st * (XG_BN * 64) + row * 64 + ((ch ^ ((row >> 1) & 3)) << 4));
                    bq[2 * pr][0] = r4[0]; bq[2 * pr][1] = r4[1]; bq[2 * pr + 1][0] = r4[2]; bq[2 * pr + 1][1] = r4[3];
                }
#pragma unroll
                for (int a = 0; a < 2; a++)
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        if (b < nv) {
                            int* Ib = I[a][b];
                            if (D::MODE == 2) {
                                xg_mma32(Ib, af[a], bq[b], Ib[0], Ib[1], Ib[2], Ib[3]);
                            } else if (D::MODE == 1) {
                                int d[4];
                                xg_mma32(d, af[a], bq[b], 0, 0, 0, 0);
                                const int m0 = xg_c4(mr[a][0], 2 * q), m1 = xg_c4(mr[a][1], 2 * q);
                                Ib[0] += m0 * d[0]; Ib[1] += m0 * d[1]; Ib[2] += m1 * d[2]; Ib[3] += m1 * d[3];
                            } else {
#pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    int d[4];
                                    xg_mma16(d, af[a][2 * h], af[a][2 * h + 1], bq[b][h]);
                                    const int m0 = xg_c4(mr[a][0], 2 * q + h), m1 = xg_c4(mr[a][1], 2 * q + h);
                                    Ib[0] += m0 * d[0]; Ib[1] += m0 * d[1]; Ib[2] += m1 * d[2]; Ib[3] += m1 * d[3];
                                }
                            }
                        }
                    }
                if ((2 * s + q + 1) % SUBS == 0) {        /* a weight block ends: the float step, in block order */
#pragma unroll
                    for (int a = 0; a < 2; a++) {
                        const float dw0 = dws[bf][32 * wm + 16 * a + g][q], dw1 = dws[bf][32 * wm + 16 * a + g + 8][q];
#pragma unroll
                        for (int b = 0; b < 4; b++) {
                            if (b < nv) {
#pragma unroll
                                for (int c = 0; c < 4; c++) {
                                    const float sc = __fmul_rn(c < 2 ? dw0 : dw1, xss[st][32 * wn + 8 * b + 2 * t + (c & 1)][q]);
                                    A[a][b][c] = __fadd_rn(A[a][b][c], __fmul_rn((float)I[a][b][c], sc));
                                    I[a][b][c] = 0;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (s + 1 < S) {
            XG_DECODE(s + 1);                       /* into the other buffer: its last reader was slice s - 1 */
            if (s + 2 < S) XG_FETCH(s + 2);
        }
    }
#undef XG_ISSUE
#undef XG_FETCH
#undef XG_DECODE
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) {
            if (b >= nv) continue;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int pr = 32 * wn + 8 * b + 2 * t + (c & 1), r = rw0 + 32 * wm + 16 * a + g + 8 * (c >> 1);
                if (pr >= np) continue;
                const int n = p0 + pr;
                if (in_pair) out[((size_t)ptok[n] * nsel + pslot[n]) * ldo + r] = A[a][b][c];
                else out[(size_t)n * ldo + r] = A[a][b][c];
            }
        }
#endif
}
#define XG_KERNEL(name, TY)                                                                                        \
    extern "C" __global__ void __launch_bounds__(256, 2)                                                           \
    name(int cols, int ab, const uint8_t* const* mats, int part, int ubase, const int4* tiles, const int* ptok,    \
         const int* pslot, int nsel, int in_pair, const int8_t* xq, const float* xs, float* out, int ldo) {        \
        xtile_body<TY>(cols, ab, mats, part, ubase, tiles, ptok, pslot, nsel, in_pair, xq, xs, out, ldo);          \
    }
XG_KERNEL(k_xtile_iq2s, T_IQ2_S)
XG_KERNEL(k_xtile_iq3xxs, T_IQ3_XXS)
XG_KERNEL(k_xtile_iq3s, T_IQ3_S)
XG_KERNEL(k_xtile_iq4xs, T_IQ4_XS)
XG_KERNEL(k_xtile_iq4nl, T_IQ4_NL)
XG_KERNEL(k_xtile_q20, T_Q2_0)
#define QSA_NINF __int_as_float(0xff800000)
/* ---------------------------------------------------------------- decode attention (QSA) for T tokens (KINDI_QXG_QSAK)
   One-token steps and verify windows run the same four kernels (T = 1 or the window), so token t of a window has the
   bits of the t-th one-token step. A token's work does not depend on the other tokens: the caches at earlier
   positions are written first (k_qsa_prep), a block key is used only once its block is complete (and then every
   token computes it from the same rows), and the selection and attention of token t read only positions <= pos[t].
   k_qsa_prep: grid (T, nh + nkv + nih + 1), hd threads: the q heads normed and RoPE'd (qo), the k heads normed and
     RoPE'd into the cache at pos (with v), the indexer's query heads normed and RoPE'd (qi), the indexer key row.
   k_qsa_bkey: grid T: the key of the block holding pos (mean of its ra rows, normed, RoPE'd at its first position).
   k_qsa_sel: grid T: the block scores sum_h relu(qi_h . bk_b) / sqrt(di) over the visible blocks, the best nk of
     them (ties: the lower block), in block order, then the tail; positions and whether each is real.
   k_qsa_attn: grid (T, nkv): the rep query heads of a KV head over the selected positions (16 at a time, softmax
     kept running), times sigmoid(gate). */
static __device__ __forceinline__ float qsa_bsum(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < nw; i++) s += sh[i];
    return s;
}
/* x[d] normed (x * rsqrt(mean x^2 + eps) * w) and RoPE'd at position p on its first 2 * half values, n = blockDim */
static __device__ __forceinline__ float qsa_norm_rope(float x, const float* w, float eps, long long p,
                                                      const double* inv, int half, float* sx, float* sh) {
    const int d = threadIdx.x, n = blockDim.x;
    const float r = rsqrtf(qsa_bsum(x * x, sh) / (float)n + eps);
    const float y = x * r * w[d];
    sx[d] = y;
    __syncthreads();
    float o = y;
    if (d < 2 * half) {
        const int i = d < half ? d : d - half;
        const float a = (float)((double)p * inv[i]);
        const float c = cosf(a), s = sinf(a);
        o = d < half ? sx[d] * c - sx[d + half] * s : sx[d] * c + sx[d - half] * s;
    }
    __syncthreads();
    return o;
}
extern "C" __global__ void k_qsa_prep(const long long* pos, const float* qg, const float* kx, const float* vx,
                                      const float* ik, const float* iq, const float* qn, const float* kn,
                                      const float* iqn, const double* inv, int half, float eps, int nh, int nkv,
                                      int nih, int di, float* qo, float* kc, float* vc, float* rk, float* qi) {
    __shared__ float sx[256], sh[32];
    const int t = blockIdx.x, y = blockIdx.y, d = threadIdx.x, hd = blockDim.x;
    const long long p = pos[t];
    if (y < nh) {
        const float v = qsa_norm_rope(qg[((size_t)t * nh + y) * 2 * hd + d], qn, eps, p, inv, half, sx, sh);
        qo[((size_t)t * nh + y) * hd + d] = v;
    } else if (y < nh + nkv) {
        const int g = y - nh;
        const float v = qsa_norm_rope(kx[((size_t)t * nkv + g) * hd + d], kn, eps, p, inv, half, sx, sh);
        kc[((size_t)p * nkv + g) * hd + d] = v;
        vc[((size_t)p * nkv + g) * hd + d] = vx[((size_t)t * nkv + g) * hd + d];
    } else if (y < nh + nkv + nih) {
        const int h = y - nh - nkv;
        float x = d < di ? iq[((size_t)t * nih + h) * di + d] : 0.f;
        /* di < blockDim: the norm over di values only (the extra threads hold zeros) */
        const float r = rsqrtf(qsa_bsum(x * x, sh) / (float)di + eps);
        const float yv = d < di ? x * r * iqn[d] : 0.f;
        sx[d] = yv;
        __syncthreads();
        float o = yv;
        if (d < 2 * half) {
            const int i = d < half ? d : d - half;
            const float a = (float)((double)p * inv[i]);
            const float c = cosf(a), s = sinf(a);
            o = d < half ? sx[d] * c - sx[d + half] * s : sx[d] * c + sx[d - half] * s;
        }
        if (d < di) qi[((size_t)t * nih + h) * di + d] = o;
    } else if (d < di) {
        rk[(size_t)p * di + d] = ik[(size_t)t * di + d];
    }
}
extern "C" __global__ void k_qsa_bkey(const long long* pos, const float* rk, const float* kn, const double* inv,
                                      int half, float eps, int ra, float* bk) {
    __shared__ float sx[256], sh[32];
    const int t = blockIdx.x, d = threadIdx.x, di = blockDim.x;
    const long long b = pos[t] / ra;
    float s = 0.f;
    for (int r = 0; r < ra; r++) s += rk[(size_t)(b * ra + r) * di + d];
    const float v = qsa_norm_rope(s / (float)ra, kn, eps, b * ra, inv, half, sx, sh);
    bk[(size_t)b * di + d] = v;
}
/* the order key of a float: larger float -> larger unsigned (-inf lowest) */
static __device__ __forceinline__ unsigned qsa_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
extern "C" __global__ void k_qsa_sel(const long long* pos, const float* qi, const float* bk, int nb, int nih, int di,
                                     int ra, int nk, long long lmax, float* sc, int* sel, unsigned char* ok) {
    __shared__ float sq[4 * 128];
    __shared__ unsigned hist[256];
    __shared__ unsigned sh_pref, sh_want, sh_cnt[2];
    __shared__ int scan[1024];
    const int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const long long p = pos[t];
    const int nvis = (int)((p + 1) / ra);
    float* s = sc + (size_t)t * nb;
    for (int i = tid; i < nih * di; i += nt) sq[i] = qi[(size_t)t * nih * di + i];
    __syncthreads();
    const float isq = 1.f / sqrtf((float)di);
    for (int b = tid; b < nb; b += nt) {
        float v = QSA_NINF;
        if (b < nvis) {
            float acc = 0.f;
            for (int h = 0; h < nih; h++) {
                float dt = 0.f;
                for (int k = 0; k < di; k++) dt += sq[h * di + k] * bk[(size_t)b * di + k];
                acc += fmaxf(dt, 0.f);
            }
            v = acc * isq;
        }
        s[b] = v;
    }
    __syncthreads();
    const int K = nk < nb ? nk : nb;
    int* so = sel + (size_t)t * (K * ra + ra - 1);
    unsigned char* oo = ok + (size_t)t * (K * ra + ra - 1);
    /* the K-th largest key among the nb scores (radix select, 8 bits a pass); then every b with a larger key, and the
       lowest-index b's with an equal key up to K, in block order */
    unsigned pref = 0, want = K;
    for (int sft = 24; sft >= 0; sft -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        const unsigned hm = sft == 24 ? 0u : (0xFFFFFFFFu << (sft + 8));
        for (int b = tid; b < nb; b += nt) {
            const unsigned k = qsa_key(s[b]);
            if ((k & hm) == (pref & hm)) atomicAdd(&hist[(k >> sft) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned c = 0;
            int dgt = 255;
            for (; dgt > 0; dgt--) { if (c + hist[dgt] >= want) break; c += hist[dgt]; }
            sh_pref = pref | ((unsigned)dgt << sft);
            sh_want = want - c;
        }
        __syncthreads();
        pref = sh_pref; want = sh_want;
        __syncthreads();
    }
    const unsigned thr = pref;            /* the K-th largest key; want: how many of its ties to take */
    /* block order: a scan over b of "taken" (key > thr, or key == thr and among the first want equals) */
    int base = 0, eqbase = 0;
    for (int b0 = 0; b0 < nb; b0 += nt) {
        const int b = b0 + tid;
        unsigned k = b < nb ? qsa_key(s[b]) : 0u;
        const int eq = b < nb && k == thr;
        scan[tid] = eq;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) {
            const int v = tid >= o ? scan[tid - o] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        const int eqrank = eqbase + scan[tid] - eq;      /* equals before this b */
        const int tot_eq = scan[nt - 1];
        __syncthreads();
        const int take = b < nb && (k > thr || (eq && eqrank < (int)want));
        scan[tid] = take;
        __syncthreads();
        for (int o = 1; o < nt; o <<= 1) {
            const int v = tid >= o ? scan[tid - o] : 0;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }
        if (take) {
            const int i = base + scan[tid] - 1;
            const int real = s[b] > QSA_NINF;
            for (int r = 0; r < ra; r++) {
                long long q = (long long)b * ra + r;
                so[i * ra + r] = (int)(q < lmax - 1 ? q : lmax - 1);
                oo[i * ra + r] = (unsigned char)real;
            }
        }
        const int tot = scan[nt - 1];
        __syncthreads();
        base += tot; eqbase += tot_eq;
    }
    if (tid < ra - 1) {
        const long long q = (long long)nvis * ra + tid;
        so[K * ra + tid] = (int)(q < lmax - 1 ? q : lmax - 1);
        oo[K * ra + tid] = (unsigned char)(q <= p);
    }
    (void)sh_cnt;
}
#define QSA_TILE 16
#define QSA_NS 16                                  /* position splits of a (token, KV head): partial softmaxes */
/* grid (T, nkv, QSA_NS): split z takes positions [z S / NS, (z + 1) S / NS); per tile of 16: the rep x 16 scores
   (a warp each), the new running max and the exponentials once (shared), then each thread (one value dimension)
   updates the rep accumulators. Writes its partial (max, sum, acc) per head to part. */
extern "C" __global__ void k_qsa_attn(const int* sel, const unsigned char* ok, int S, const float* qo,
                                      const float* kc, const float* vc, int nkv, int rep, float* part) {
    __shared__ float Ks[QSA_TILE][256], Vs[QSA_TILE][256], Qs[12][256], sc[12][QSA_TILE], mf[12][2];
    __shared__ int ps[QSA_TILE];
    __shared__ unsigned char po[QSA_TILE];
    const int t = blockIdx.x, g = blockIdx.y, z = blockIdx.z, d = threadIdx.x, hd = blockDim.x;
    const int lane = d & 31, warp = d >> 5, nw = hd >> 5, nh = nkv * rep;
    for (int r = 0; r < rep; r++) Qs[r][d] = qo[((size_t)t * nh + g * rep + r) * hd + d];
    if (d < rep) { mf[d][0] = QSA_NINF; mf[d][1] = 0.f; }      /* running max, running sum */
    float acc[12];
    for (int r = 0; r < 12; r++) acc[r] = 0.f;
    const float isq = 1.f / sqrtf((float)hd);
    const int* st = sel + (size_t)t * S;
    const unsigned char* ot = ok + (size_t)t * S;
    const int s0 = (int)((long long)z * S / QSA_NS), s1 = (int)((long long)(z + 1) * S / QSA_NS);
    for (int j0 = s0; j0 < s1; j0 += QSA_TILE) {
        const int nj = s1 - j0 < QSA_TILE ? s1 - j0 : QSA_TILE;
        __syncthreads();
        if (d < nj) { ps[d] = st[j0 + d]; po[d] = ot[j0 + d]; }
        /* a tile with no real position changes nothing (scores -inf: the max stays, the factor is exp(0) = 1, the
           exponentials 0, so sums and accumulators keep their bits): skipped. Short contexts select mostly padding
           (the blocks past the context, then the tail), so most tiles of the 16 splits are skipped. */
        if (!__syncthreads_or(d < nj && po[d])) continue;
        for (int j = 0; j < nj; j++) {
            Ks[j][d] = kc[((size_t)ps[j] * nkv + g) * hd + d];
            Vs[j][d] = vc[((size_t)ps[j] * nkv + g) * hd + d];
        }
        __syncthreads();
        for (int i = warp; i < rep * nj; i += nw) {
            const int r = i / nj, j = i % nj;
            float v = 0.f;
            for (int k = lane; k < hd; k += 32) v += Qs[r][k] * Ks[j][k];
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) sc[r][j] = po[j] ? v * isq : QSA_NINF;
        }
        __syncthreads();
        __shared__ float fac[12], ex[12][QSA_TILE];
        if (d < rep) {                                /* head d: new max, the old sum's factor, the new sum */
            float mx = mf[d][0];
            for (int j = 0; j < nj; j++) mx = fmaxf(mx, sc[d][j]);
            float f = 1.f, ls = mf[d][1];
            if (mx != QSA_NINF) {
                f = expf(mf[d][0] - mx);
                ls *= f;
                for (int j = 0; j < nj; j++) { const float e = expf(sc[d][j] - mx); ex[d][j] = e; ls += e; }
                mf[d][0] = mx;
            } else {
                for (int j = 0; j < nj; j++) ex[d][j] = 0.f;
            }
            mf[d][1] = ls; fac[d] = f;
        }
        __syncthreads();
        for (int r = 0; r < rep; r++) {
            float a = acc[r] * fac[r];
            for (int j = 0; j < nj; j++) a += ex[r][j] * Vs[j][d];
            acc[r] = a;
        }
    }
    __syncthreads();
    /* part [T][nh][NS][hd + 2] */
    for (int r = 0; r < rep; r++) {
        float* pp = part + ((((size_t)t * nh + g * rep + r) * QSA_NS) + z) * (hd + 2);
        pp[d] = acc[r];
        if (d == 0) { pp[hd] = mf[r][0]; pp[hd + 1] = mf[r][1]; }
    }
}
/* grid (T, nh): the NS partials of a head in split order, then times sigmoid(gate) */
extern "C" __global__ void k_qsa_comb(const float* part, const float* qg, int nh, float* out) {
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x, hd = blockDim.x;
    const float* pp = part + ((size_t)t * nh + h) * QSA_NS * (hd + 2);
    float M = QSA_NINF;
    for (int z = 0; z < QSA_NS; z++) M = fmaxf(M, pp[z * (hd + 2) + hd]);
    float L = 0.f, O = 0.f;
    for (int z = 0; z < QSA_NS; z++) {
        const float mz = pp[z * (hd + 2) + hd];
        if (mz == QSA_NINF) continue;
        const float f = expf(mz - M);
        L += pp[z * (hd + 2) + hd + 1] * f;
        O += pp[z * (hd + 2) + d] * f;
    }
    const float gt = qg[((size_t)t * nh + h) * 2 * hd + hd + d];
    out[((size_t)t * nh + h) * hd + d] = (O / L) * (1.f / (1.f + expf(-gt)));
}

/* counts[ids[i]] += 1 for the n picks (the MTP head's routing counts, KINDI_MTP_USAGE_OUT) */
extern "C" __global__ void k_count(const int* ids, int n, int* counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(counts + ids[i], 1);
}

/* buf[i] = the card's global timer (ns): the per-layer timeline of a window (KINDI_QXG_TIMELINE) */
extern "C" __global__ void k_ts(unsigned long long* buf, int i) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (threadIdx.x == 0) buf[i] = t;
}

/* ---------------------------------------------------------------- decode experts on the card, fast (KINDI_QXG_FSLOTS)
   k_gate_up_slots / k_down_slots with whole 32-value sub-blocks a thread: the thread loads the sub-block's raw bytes,
   decodes its 32 codes (XG<TY>, the grids in shared memory) and takes the 8 dp4a sums against the input's 32 bytes,
   times m (per 16 or per 32 values; none for Q2_0 / IQ4_NL). The 8 (or 2, 1) threads of a weight block add their
   integers (exact in any order), and one thread per row then takes the float steps A = A + (float)I * (dw * dx) in
   block order: the integers and the float steps of row_dot2_256, so the same bits (card rows == CPU rows).
   Gate/up: XG_GL lanes per row (16, a half-warp; gate and up rows together), K / 32 sub-blocks over them in rounds;
   grid (F / (256 / XG_GL), slots), 256 threads. Down (DN 64): a half-warp per row, F / 32 sub-blocks over 16 lanes;
   grid (K / 16, slots), 256 threads. */
#ifndef XG_GL
#define XG_GL 16
#endif
template <int TY>
static __device__ __forceinline__ int fs_isum(const uint32_t* q, const int* xw, const int* m) {
    typedef XG<TY> D;
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) s0 = __dp4a((int)q[k], xw[k], s0);
#pragma unroll
    for (int k = 4; k < 8; k++) s1 = __dp4a((int)q[k], xw[k], s1);
    if (D::MODE == 2) return s0 + s1;
    return s0 * m[0] + s1 * m[1];
}
template <int TY>
static __device__ __forceinline__ void fs_gate_up(int K, int F, const int* idx, const int* slotmap, const uint8_t* hot,
                                                  long long rowbytes, const int8_t* xq, const float* xd, float* h,
                                                  int tok_nsel, int min_slot, uint32_t* gts, int (*Ig)[40],
                                                  int (*Iu)[40], float (*dg)[40], float (*du)[40]) {
    typedef XG<TY> D;
    enum { SUBS = D::PER / 32 };
    enum { GL = XG_GL, NW = 256 / XG_GL };
    const int s = blockIdx.y, warp = threadIdx.x / GL, lane = threadIdx.x % GL, r = blockIdx.x * NW + warp;
    const int sl = slotmap[idx[s]];
    if (sl < min_slot) return;                             /* the whole block: one slot */
    if (D::GW > 1)
        for (int i = threadIdx.x; i < D::GW; i += blockDim.x) gts[i] = D::grid()[i];
    uint32_t lv[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        lv[k] = (uint32_t)(uint8_t)c_lv4[4 * k] | ((uint32_t)(uint8_t)c_lv4[4 * k + 1] << 8) |
                ((uint32_t)(uint8_t)c_lv4[4 * k + 2] << 16) | ((uint32_t)(uint8_t)c_lv4[4 * k + 3] << 24);
    __syncthreads();
    if (r >= F) return;
    const long long rb = (long long)(K / D::PER) * D::BYTES;
    const uint8_t* g_row = hot + (size_t)sl * rowbytes + (size_t)r * rb;
    const uint8_t* u_row = hot + (size_t)sl * rowbytes + (size_t)F * rb + (size_t)r * rb;
    const int8_t* x = xq + (size_t)(s / tok_nsel) * K;
    const float* xs = xd + (size_t)(s / tok_nsel) * (K / 256);
    const int nsb = K / 32, nblk = K / D::PER;
#pragma unroll 2
    for (int q0 = 0; q0 < nsb; q0 += GL) {
        const int qq = q0 + lane;
        int ig = 0, iu = 0;
        float dwg = 0.f, dwu = 0.f;
        if (qq < nsb) {
            const int b = qq / SUBS, j = qq % SUBS;
            uint32_t rg[D::NR], ru[D::NR], cg[8], cu[8];
            int mg[2], mu[2];
            D::fetch(g_row + (size_t)b * D::BYTES, j, rg);
            D::fetch(u_row + (size_t)b * D::BYTES, j, ru);
            const int4 xa = *(const int4*)(x + 32 * qq), xb = *(const int4*)(x + 32 * qq + 16);
            const int xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            D::decode(rg, j, gts, lv, cg, mg, &dwg);
            D::decode(ru, j, gts, lv, cu, mu, &dwu);
            ig = fs_isum<TY>(cg, xw, mg);
            iu = fs_isum<TY>(cu, xw, mu);
        }
#pragma unroll
        for (int o = 1; o < SUBS; o <<= 1) {                /* the block's sub-blocks: SUBS consecutive lanes */
            ig += __shfl_xor_sync(0xffffffffu, ig, o);
            iu += __shfl_xor_sync(0xffffffffu, iu, o);
        }
        if (qq < nsb && qq % SUBS == 0) {
            Ig[warp][qq / SUBS] = ig; Iu[warp][qq / SUBS] = iu;
            dg[warp][qq / SUBS] = dwg; du[warp][qq / SUBS] = dwu;
        }
    }
    __syncwarp();
    if (lane == 0) {
        float A0 = 0.f, A1 = 0.f;
        for (int bk = 0; bk < nblk; bk++) {
            const float dx = xs[bk * D::PER / 256];
            A0 = __fadd_rn(A0, __fmul_rn((float)Ig[warp][bk], __fmul_rn(dg[warp][bk], dx)));
            A1 = __fadd_rn(A1, __fmul_rn((float)Iu[warp][bk], __fmul_rn(du[warp][bk], dx)));
        }
        const float sl_ = __fdiv_rn(A0, __fadd_rn(1.f, qx_expf(-A0)));
        h[(size_t)s * F + r] = __fmul_rn(sl_, A1);
    }
}
template <int TY>
static __device__ __forceinline__ void fs_down(int K, int F, const int* idx, const int* slotmap, const uint8_t* hot,
                                               long long rowbytes, long long dn_off, const int8_t* hq, const float* hd,
                                               float* parts, int min_slot) {
    typedef XG<TY> D;
    enum { SUBS = D::PER / 32 };
    __shared__ int Id[16][24];
    __shared__ float dd[16][24];
    const int s = blockIdx.y, hw = threadIdx.x >> 4, ln = threadIdx.x & 15, r = blockIdx.x * 16 + hw;
    const int sl = slotmap[idx[s]];
    if (sl < min_slot) return;
    uint32_t lv[4];
#pragma unroll
    for (int k = 0; k < 4; k++)
        lv[k] = (uint32_t)(uint8_t)c_lv4[4 * k] | ((uint32_t)(uint8_t)c_lv4[4 * k + 1] << 8) |
                ((uint32_t)(uint8_t)c_lv4[4 * k + 2] << 16) | ((uint32_t)(uint8_t)c_lv4[4 * k + 3] << 24);
    if (r >= K) return;
    const long long rb = (long long)(F / D::PER) * D::BYTES;
    const uint8_t* row = hot + (size_t)sl * rowbytes + dn_off + (size_t)r * rb;
    const int8_t* x = hq + (size_t)s * F;
    const float* xs = hd + (size_t)s * (F / 64);
    const int nsb = F / 32, nblk = F / D::PER;
    for (int q0 = 0; q0 < nsb; q0 += 16) {
        const int qq = q0 + ln;
        int iv = 0;
        float dw = 0.f;
        if (qq < nsb) {
            const int b = qq / SUBS, j = qq % SUBS;
            uint32_t rr[D::NR], c[8];
            int m[2];
            D::fetch(row + (size_t)b * D::BYTES, j, rr);
            const int4 xa = *(const int4*)(x + 32 * qq), xb = *(const int4*)(x + 32 * qq + 16);
            const int xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
            D::decode(rr, j, (const uint32_t*)0, lv, c, m, &dw);
            iv = fs_isum<TY>(c, xw, m);
        }
#pragma unroll
        for (int o = 1; o < SUBS; o <<= 1) iv += __shfl_xor_sync(0xffffffffu, iv, o);
        if (qq < nsb && qq % SUBS == 0) { Id[hw][qq / SUBS] = iv; dd[hw][qq / SUBS] = dw; }
    }
    __syncwarp();
    if (ln == 0) {
        float A = 0.f;
        for (int bk = 0; bk < nblk; bk++)
            A = __fadd_rn(A, __fmul_rn((float)Id[hw][bk], __fmul_rn(dd[hw][bk], xs[bk * D::PER / 64])));
        parts[(size_t)s * K + r] = A;
    }
}
extern "C" __global__ void k_fgu(int ty, int K, int F, const int* idx, const int* slotmap, const uint8_t* hot,
                                 long long rowbytes, const int8_t* xq, const float* xd, float* h, int tok_nsel,
                                 int min_slot) {
    /* one copy of the shared arrays for all formats (each template's own would add up past 48 KB) */
    __shared__ __align__(16) uint32_t gts[2048];
    __shared__ int Ig[256 / XG_GL][40], Iu[256 / XG_GL][40];
    __shared__ float dg[256 / XG_GL][40], du[256 / XG_GL][40];
#define FS_ARGS K, F, idx, slotmap, hot, rowbytes, xq, xd, h, tok_nsel, min_slot, gts, Ig, Iu, dg, du
    switch (ty) {
    case T_IQ2_S: fs_gate_up<T_IQ2_S>(FS_ARGS); return;
    case T_IQ3_XXS: fs_gate_up<T_IQ3_XXS>(FS_ARGS); return;
    case T_IQ3_S: fs_gate_up<T_IQ3_S>(FS_ARGS); return;
    default: fs_gate_up<T_IQ4_XS>(FS_ARGS); return;
    }
#undef FS_ARGS
}
extern "C" __global__ void k_fdn(int ty, int K, int F, const int* idx, const int* slotmap, const uint8_t* hot,
                                 long long rowbytes, long long dn_off, const int8_t* hq, const float* hd, float* parts,
                                 int min_slot) {
    if (ty == T_Q2_0) fs_down<T_Q2_0>(K, F, idx, slotmap, hot, rowbytes, dn_off, hq, hd, parts, min_slot);
    else fs_down<T_IQ4_NL>(K, F, idx, slotmap, hot, rowbytes, dn_off, hq, hd, parts, min_slot);
}
