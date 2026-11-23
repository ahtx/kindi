/* Kindi GPU & Hetero CUDA C + NVRTC engine for Thinking Machines Inkling-2-Mini (inkling2_loop).
 * Single-source dual compilation so containers without g++ (cc1plus) compile cleanly:
 *   1. Device section (#ifdef __CUDACC__) compiles to PTX via libnvrtc (cupy.cuda.nvrtc) with zero headers.
 *   2. Host section (#else) compiles as pure C11 via gcc -x c -O3 -std=gnu11 -shared -fPIC -lcudart -ldl.
 * Clean across CUDA 10.2+ and sm_30 (K420) / sm_50 (K1200) / sm_61 (GTX 1080) / sm_75+ / sm_120.
 */

#define IK_NL_PHYS     24
#define IK_CORE_START  6
#define IK_CORE_END    18
#define IK_N_PASS      3
#define IK_NL_VIRT     48
#define IK_D           2304
#define IK_NH          24
#define IK_NKV         8
#define IK_HD          128
#define IK_QDIM        (IK_NH * IK_HD)
#define IK_GQADIM      (IK_NKV * IK_HD)
#define IK_NREL        16
#define IK_RDIM        (IK_NH * IK_NREL)
#define IK_RELEXT      1024
#define IK_NCONV       4
#define IK_SWA         1024
#define IK_BLOCK_SZ    6
#define IK_GRANK       64
#define IK_FF          1152
#define IK_FF2         2304
#define IK_NE          128
#define IK_NU          8
#define IK_V_UNPAD     200058
#define IK_MAXB        128
#define IK_MAX_LOGIT_T 4
#define IK_CHSCONV     (IK_GQADIM + IK_GQADIM + IK_D + IK_D)

#define IK_RMS_EPS     1e-6f
#define IK_EXP_SCALE   8.0f
#define IK_SITU_BETA   4.0f
#define IK_SITU_LBETA  25.0f
#define IK_QK_CLIP     22.627416610717773f
#define IK_REL_CLIP    8.0f
#define IK_LOG_FLOOR   8192.0f
#define IK_LOG_ALPHA   0.1f
#define IK_LOGIT_CAP   30.0f

#define QK_Q4    32
#define BQ4      18
#define QK_Q8    32
#define BQ8      34
#define QK_NV    64
#define BNV      36
#define QK_K     256
#define BQ5K     176
#define QK_IQ2   128
#define BIQ2_XXS 34
#define BIQ2_XS  38

#define TY_F32         0
#define TY_Q4_0        2
#define TY_Q8          8
#define TY_Q5K         13
#define TY_BF16        30
#define TY_NVFP4       40
#define TY_IQ2XXS_128  116
#define TY_IQ2XS_128   117

#define IK_MAX_DEV 4

typedef struct { int tok; float m1, m2; } IkTop2;

/* ================================================================
 * PART 1: NVRTC Device Kernels (#ifdef __CUDACC__) — header-free!
 * ================================================================ */
#ifdef __CUDACC__

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef long long          int64_t;
typedef unsigned long long uint64_t;
#ifndef NULL
#define NULL 0
#endif

#include "inkling2_iq2_tables.h"

static __device__ __forceinline__ int32_t q4_unpack4(uint32_t u, int h) {
    uint32_t n = h ? ((u >> 4) & 0x0F0F0F0Fu) : (u & 0x0F0F0F0Fu);
    uint32_t t = n ^ 0x08080808u;
    return (int32_t)(t | ((t & 0x08080808u) * 30u));
}

static __device__ __forceinline__ int32_t iq2_apply_signs4(uint32_t g4, uint32_t m4) {
    uint32_t b = (m4 & 1u) | ((m4 & 2u) << 7) | ((m4 & 4u) << 14) | ((m4 & 8u) << 21);
    uint32_t mask = b * 255u;
    return (int32_t)((g4 ^ mask) + b);
}

static __device__ __forceinline__ float iq2_dot4_f32(uint32_t g4, uint32_t m4, float4 xf) {
    float s0 = (m4 & 1u) ? -xf.x : xf.x;
    float s1 = (m4 & 2u) ? -xf.y : xf.y;
    float s2 = (m4 & 4u) ? -xf.z : xf.z;
    float s3 = (m4 & 8u) ? -xf.w : xf.w;
    return (float)(g4 & 255u) * s0 + (float)((g4 >> 8) & 255u) * s1
         + (float)((g4 >> 16) & 255u) * s2 + (float)(g4 >> 24) * s3;
}

__device__ __forceinline__ float d_ldg(const float* p) {
#if __CUDA_ARCH__ >= 320
    return __ldg(p);
#else
    return *p;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
#endif

__device__ __forceinline__ float d_ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7Fu) return 0.0f;
    uint32_t e = (x >> 3) & 0xFu;
    uint32_t m = x & 7u;
    if (e == 0) return (float)m * (1.0f / 1024.0f);
    return __uint_as_float(((e + 119u) << 23) | (m << 20));
}

__device__ __forceinline__ float d_bf16(uint16_t u) {
    return __uint_as_float(((uint32_t)u) << 16);
}

__device__ __forceinline__ float d_h2f(uint16_t h) {
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ uint16_t d_f2h(float x) {
    uint16_t r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    return v;
}

__device__ __forceinline__ float block_reduce_sum_256(float v) {
    __shared__ float sm[8];
    int tid = threadIdx.x;
    v = warp_sum(v);
    if ((tid & 31) == 0) sm[tid >> 5] = v;
    __syncthreads();
    if (tid < 8) {
        /* only lanes 0-7 are here: the mask names exactly them. A full mask waits for lanes that never arrive and
           hangs on sm_70+ (independent thread scheduling); sm_61 and older did not notice. */
        float x = sm[tid];
        x += __shfl_xor_sync(0x000000FFu, x, 4);
        x += __shfl_xor_sync(0x000000FFu, x, 2);
        x += __shfl_xor_sync(0x000000FFu, x, 1);
        sm[0] = x;
    }
    __syncthreads();
    return sm[0];
}

__device__ __forceinline__ float d_situ(float g, float u) {
    float sig = 1.0f / (1.0f + expf(-g));
    float act = IK_SITU_BETA * tanhf(g / IK_SITU_BETA) * sig;
    float upv = IK_SITU_LBETA * tanhf(u / IK_SITU_LBETA);
    return act * upv;
}

/* ---------------------------------------------------------------- Repack Kernels */
extern "C" __global__ void k_repack_q8_to_g16(const uint8_t* __restrict__ raw, int rows, int nb,
                                              uint16_t* __restrict__ d_sc, int8_t* __restrict__ d_qs) {
    int g = blockIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    if (b >= nb) return;
    int r = t >> 1;
    int h = t & 1;
    int rr = g * 16 + r;
    size_t goff_b = ((size_t)g * nb + b);
    if (h == 0) {
        uint16_t sc = 0;
        if (rr < rows) {
            const uint8_t* src = raw + ((size_t)rr * nb + b) * BQ8;
            sc = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
        }
        d_sc[goff_b * 16 + r] = sc;
    }
    int4 qv = make_int4(0, 0, 0, 0);
    if (rr < rows) {
        const uint8_t* src = raw + ((size_t)rr * nb + b) * BQ8 + 2 + h * 16;
        uint32_t w0 = (uint32_t)src[0]  | ((uint32_t)src[1] << 8)  | ((uint32_t)src[2] << 16)  | ((uint32_t)src[3] << 24);
        uint32_t w1 = (uint32_t)src[4]  | ((uint32_t)src[5] << 8)  | ((uint32_t)src[6] << 16)  | ((uint32_t)src[7] << 24);
        uint32_t w2 = (uint32_t)src[8]  | ((uint32_t)src[9] << 8)  | ((uint32_t)src[10] << 16) | ((uint32_t)src[11] << 24);
        uint32_t w3 = (uint32_t)src[12] | ((uint32_t)src[13] << 8) | ((uint32_t)src[14] << 16) | ((uint32_t)src[15] << 24);
        qv = make_int4((int)w0, (int)w1, (int)w2, (int)w3);
    }
    ((int4*)(d_qs + goff_b * 512))[t] = qv;
}

extern "C" __global__ void k_repack_nvfp4_to_g16(const uint8_t* __restrict__ raw, int rows, int nb64,
                                                 uint8_t* __restrict__ d_sc, uint8_t* __restrict__ d_qs) {
    int g = blockIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    if (b >= nb64) return;
    int r = t >> 1;
    int h = t & 1;
    int rr = g * 16 + r;
    size_t goff_b = ((size_t)g * nb64 + b);
    uint16_t sc2 = 0;
    uint4 qv = make_uint4(0, 0, 0, 0);
    if (rr < rows) {
        const uint8_t* src = raw + ((size_t)rr * nb64 + b) * BNV;
        sc2 = (uint16_t)src[h * 2] | ((uint16_t)src[h * 2 + 1] << 8);
        const uint8_t* p = src + 4 + h * 16;
        uint32_t w0 = (uint32_t)p[0]  | ((uint32_t)p[1] << 8)  | ((uint32_t)p[2] << 16)  | ((uint32_t)p[3] << 24);
        uint32_t w1 = (uint32_t)p[4]  | ((uint32_t)p[5] << 8)  | ((uint32_t)p[6] << 16)  | ((uint32_t)p[7] << 24);
        uint32_t w2 = (uint32_t)p[8]  | ((uint32_t)p[9] << 8)  | ((uint32_t)p[10] << 16) | ((uint32_t)p[11] << 24);
        uint32_t w3 = (uint32_t)p[12] | ((uint32_t)p[13] << 8) | ((uint32_t)p[14] << 16) | ((uint32_t)p[15] << 24);
        qv = make_uint4(w0, w1, w2, w3);
    }
    ((uint16_t*)(d_sc + goff_b * 64))[t] = sc2;
    ((uint4*)(d_qs + goff_b * 512))[t] = qv;
}

extern "C" __global__ void k_repack_q4_to_g16(const uint8_t* __restrict__ raw, int rows, int nb,
                                              uint16_t* __restrict__ d_sc, uint8_t* __restrict__ d_qs) {
    int g = blockIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    if (b >= nb || t >= 16) return;
    int r = t;
    int rr = g * 16 + r;
    size_t goff_b = (size_t)g * nb + b;
    uint16_t sc = 0;
    uint4 qv = make_uint4(0, 0, 0, 0);
    if (rr < rows) {
        const uint8_t* src = raw + ((size_t)rr * nb + b) * BQ4;
        sc = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
        const uint8_t* p = src + 2;
        uint32_t w0 = (uint32_t)p[0]  | ((uint32_t)p[1] << 8)  | ((uint32_t)p[2] << 16)  | ((uint32_t)p[3] << 24);
        uint32_t w1 = (uint32_t)p[4]  | ((uint32_t)p[5] << 8)  | ((uint32_t)p[6] << 16)  | ((uint32_t)p[7] << 24);
        uint32_t w2 = (uint32_t)p[8]  | ((uint32_t)p[9] << 8)  | ((uint32_t)p[10] << 16) | ((uint32_t)p[11] << 24);
        uint32_t w3 = (uint32_t)p[12] | ((uint32_t)p[13] << 8) | ((uint32_t)p[14] << 16) | ((uint32_t)p[15] << 24);
        qv = make_uint4(w0, w1, w2, w3);
    }
    d_sc[goff_b * 16 + r] = sc;
    ((uint4*)(d_qs + goff_b * 256))[r] = qv;
}

extern "C" __global__ void k_repack_iq2xxs128(const uint8_t* __restrict__ raw, int rows, int nb128,
                                              uint8_t* __restrict__ d_sc, uint8_t* __restrict__ d_qs) {
    int g = blockIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    if (b >= nb128 || t >= 16) return;
    int r = t;
    int rr = g * 16 + r;
    size_t goff_b = (size_t)g * nb128 + b;
    uint16_t* sc16 = (uint16_t*)(d_sc + goff_b * 32);
    uint32_t* qs32 = (uint32_t*)(d_qs + goff_b * 512);
    uint16_t hd = 0;
    uint32_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (rr < rows) {
        const uint8_t* src = raw + ((size_t)rr * nb128 + b) * BIQ2_XXS;
        hd = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
        const uint8_t* p = src + 2;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            q[w] = (uint32_t)p[4 * w + 0] | ((uint32_t)p[4 * w + 1] << 8)
                 | ((uint32_t)p[4 * w + 2] << 16) | ((uint32_t)p[4 * w + 3] << 24);
        }
    }
    sc16[r] = hd;
    #pragma unroll
    for (int w = 0; w < 8; w++) qs32[w * 16 + r] = q[w];
}

extern "C" __global__ void k_repack_iq2xs128(const uint8_t* __restrict__ raw, int rows, int nb128,
                                             uint8_t* __restrict__ d_sc, uint8_t* __restrict__ d_qs) {
    int g = blockIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    if (b >= nb128 || t >= 16) return;
    int r = t;
    int rr = g * 16 + r;
    size_t goff_b = (size_t)g * nb128 + b;
    uint16_t* sc_hd  = (uint16_t*)(d_sc + goff_b * 96);
    uint32_t* sc_sub = (uint32_t*)(d_sc + goff_b * 96 + 32);
    uint32_t* qs32   = (uint32_t*)(d_qs + goff_b * 512);
    uint16_t hd = 0;
    uint32_t ssub = 0;
    uint32_t q[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (rr < rows) {
        const uint8_t* src = raw + ((size_t)rr * nb128 + b) * BIQ2_XS;
        hd = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
        const uint8_t* p = src + 2;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            q[w] = (uint32_t)p[4 * w + 0] | ((uint32_t)p[4 * w + 1] << 8)
                 | ((uint32_t)p[4 * w + 2] << 16) | ((uint32_t)p[4 * w + 3] << 24);
        }
        const uint8_t* sp = src + 34;
        ssub = (uint32_t)sp[0] | ((uint32_t)sp[1] << 8) | ((uint32_t)sp[2] << 16) | ((uint32_t)sp[3] << 24);
    }
    sc_hd[r] = hd;
    sc_sub[r] = ssub;
    #pragma unroll
    for (int w = 0; w < 8; w++) qs32[w * 16 + r] = q[w];
}

extern "C" __global__ void k_quant_q8(const float* __restrict__ x, int K, int32_t* __restrict__ xq4, float* __restrict__ xd) {
    int t = blockIdx.y;
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int nb = K >> 5;
    if (b >= nb) return;
    float v = x[(size_t)t * K + b * 32 + lane];
    float am = warp_max(fabsf(v));
    float sc = am * (1.0f / 127.0f);
    float is = sc != 0.0f ? (127.0f / am) : 0.0f;
    if (lane == 0) xd[(size_t)t * nb + b] = d_h2f(d_f2h(sc));
    int q = __float2int_rn(v * is);
    if (q > 127) q = 127; else if (q < -127) q = -127;
    uint32_t ub = (uint32_t)(q & 0xFF);
    uint32_t b0 = __shfl_sync(0xFFFFFFFFu, ub, (lane & ~3) + 0);
    uint32_t b1 = __shfl_sync(0xFFFFFFFFu, ub, (lane & ~3) + 1);
    uint32_t b2 = __shfl_sync(0xFFFFFFFFu, ub, (lane & ~3) + 2);
    uint32_t b3 = __shfl_sync(0xFFFFFFFFu, ub, (lane & ~3) + 3);
    if ((lane & 3) == 0) {
        xq4[((size_t)t * nb + b) * 8 + (lane >> 2)] = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
    }
}

extern "C" __global__ void k_q8g16_mm(const uint16_t* __restrict__ w_sc, const int8_t* __restrict__ w_qs,
                                      int rows, int K, int T,
                                      const float* __restrict__ xf, const int32_t* __restrict__ xq4, const float* __restrict__ xd,
                                      float* __restrict__ y, int ys) {
    int g = blockIdx.x * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    int r = t >> 1;
    int h = t & 1;
    int rr = g * 16 + r;
    if (g * 16 >= rows) return;
    int nb = K >> 5;
    int t0 = blockIdx.z * 4;
    int nt = T - t0;
    if (nt > 4) nt = 4;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

#if __CUDA_ARCH__ >= 610
    const int xoff = h * 4;
    for (int b = 0; b < nb; b++) {
        size_t boff = (size_t)g * nb + b;
        float ws = d_h2f(w_sc[boff * 16 + r]);
        int4 wv = ((const int4*)(w_qs + boff * 512))[t];
        int xb = b * 8 + xoff;
        int db = b;
        {
            const int32_t* xp0 = xq4 + ((size_t)t0 * nb) * 8 + xb;
            int4 xv0 = *(const int4*)xp0;
            int isum = __dp4a(wv.x, xv0.x, 0);
            isum = __dp4a(wv.y, xv0.y, isum);
            isum = __dp4a(wv.z, xv0.z, isum);
            isum = __dp4a(wv.w, xv0.w, isum);
            acc0 = fmaf((float)isum, ws * xd[(size_t)t0 * nb + db], acc0);
        }
        if (nt > 1) {
            const int32_t* xp1 = xq4 + ((size_t)(t0 + 1) * nb) * 8 + xb;
            int4 xv1 = *(const int4*)xp1;
            int isum = __dp4a(wv.x, xv1.x, 0);
            isum = __dp4a(wv.y, xv1.y, isum);
            isum = __dp4a(wv.z, xv1.z, isum);
            isum = __dp4a(wv.w, xv1.w, isum);
            acc1 = fmaf((float)isum, ws * xd[(size_t)(t0 + 1) * nb + db], acc1);
        }
        if (nt > 2) {
            const int32_t* xp2 = xq4 + ((size_t)(t0 + 2) * nb) * 8 + xb;
            int4 xv2 = *(const int4*)xp2;
            int isum = __dp4a(wv.x, xv2.x, 0);
            isum = __dp4a(wv.y, xv2.y, isum);
            isum = __dp4a(wv.z, xv2.z, isum);
            isum = __dp4a(wv.w, xv2.w, isum);
            acc2 = fmaf((float)isum, ws * xd[(size_t)(t0 + 2) * nb + db], acc2);
        }
        if (nt > 3) {
            const int32_t* xp3 = xq4 + ((size_t)(t0 + 3) * nb) * 8 + xb;
            int4 xv3 = *(const int4*)xp3;
            int isum = __dp4a(wv.x, xv3.x, 0);
            isum = __dp4a(wv.y, xv3.y, isum);
            isum = __dp4a(wv.z, xv3.z, isum);
            isum = __dp4a(wv.w, xv3.w, isum);
            acc3 = fmaf((float)isum, ws * xd[(size_t)(t0 + 3) * nb + db], acc3);
        }
    }
#else
    (void)xq4; (void)xd;
    if (nt == 1) {
        const float* xp0 = xf + (size_t)t0 * K;
        for (int b = 0; b < nb; b++) {
            size_t boff = (size_t)g * nb + b;
            float ws = d_h2f(w_sc[boff * 16 + r]);
            int4 wv = ((const int4*)(w_qs + boff * 512))[t];
            int k0 = b * 32 + h * 16;
            uint32_t u[4] = {(uint32_t)wv.x, (uint32_t)wv.y, (uint32_t)wv.z, (uint32_t)wv.w};
            float d = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t p = u[j];
                float4 xv = *(const float4*)(xp0 + k0 + j * 4);
                d = fmaf((float)((int32_t)(p << 24) >> 24), xv.x, d);
                d = fmaf((float)((int32_t)(p << 16) >> 24), xv.y, d);
                d = fmaf((float)((int32_t)(p << 8)  >> 24), xv.z, d);
                d = fmaf((float)((int32_t)p >> 24),         xv.w, d);
            }
            acc0 = fmaf(d, ws, acc0);
        }
    } else {
        for (int b = 0; b < nb; b++) {
            size_t boff = (size_t)g * nb + b;
            float ws = d_h2f(w_sc[boff * 16 + r]);
            int4 wv = ((const int4*)(w_qs + boff * 512))[t];
            int k0 = b * 32 + h * 16;
            uint32_t u[4] = {(uint32_t)wv.x, (uint32_t)wv.y, (uint32_t)wv.z, (uint32_t)wv.w};
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t p = u[j];
                float f0 = (float)((int32_t)(p << 24) >> 24);
                float f1 = (float)((int32_t)(p << 16) >> 24);
                float f2 = (float)((int32_t)(p << 8)  >> 24);
                float f3 = (float)((int32_t)p >> 24);
                float4 xv0 = *(const float4*)(xf + (size_t)t0 * K + k0 + j * 4);
                d0 = fmaf(f0, xv0.x, fmaf(f1, xv0.y, fmaf(f2, xv0.z, fmaf(f3, xv0.w, d0))));
                float4 xv1 = *(const float4*)(xf + (size_t)(t0 + 1) * K + k0 + j * 4);
                d1 = fmaf(f0, xv1.x, fmaf(f1, xv1.y, fmaf(f2, xv1.z, fmaf(f3, xv1.w, d1))));
                if (nt > 2) {
                    float4 xv2 = *(const float4*)(xf + (size_t)(t0 + 2) * K + k0 + j * 4);
                    d2 = fmaf(f0, xv2.x, fmaf(f1, xv2.y, fmaf(f2, xv2.z, fmaf(f3, xv2.w, d2))));
                }
                if (nt > 3) {
                    float4 xv3 = *(const float4*)(xf + (size_t)(t0 + 3) * K + k0 + j * 4);
                    d3 = fmaf(f0, xv3.x, fmaf(f1, xv3.y, fmaf(f2, xv3.z, fmaf(f3, xv3.w, d3))));
                }
            }
            acc0 = fmaf(d0, ws, acc0);
            acc1 = fmaf(d1, ws, acc1);
            if (nt > 2) acc2 = fmaf(d2, ws, acc2);
            if (nt > 3) acc3 = fmaf(d3, ws, acc3);
        }
    }
#endif

    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1);
    if (nt > 1) acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1);
    if (nt > 2) acc2 += __shfl_xor_sync(0xFFFFFFFFu, acc2, 1);
    if (nt > 3) acc3 += __shfl_xor_sync(0xFFFFFFFFu, acc3, 1);
    if (h == 0 && rr < rows) {
        y[(size_t)t0 * ys + rr] = acc0;
        if (nt > 1) y[(size_t)(t0 + 1) * ys + rr] = acc1;
        if (nt > 2) y[(size_t)(t0 + 2) * ys + rr] = acc2;
        if (nt > 3) y[(size_t)(t0 + 3) * ys + rr] = acc3;
    }
}

extern "C" __global__ void k_q4g16_mm(const uint16_t* __restrict__ w_sc, const int8_t* __restrict__ w_qs,
                                      int rows, int K, int T,
                                      const float* __restrict__ xf, const int32_t* __restrict__ xq4, const float* __restrict__ xd,
                                      float* __restrict__ y, int ys) {
    int g = blockIdx.x * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    int r = t >> 1;
    int h = t & 1;
    int rr = g * 16 + r;
    if (g * 16 >= rows) return;
    int nb = K >> 5;
    int t0 = blockIdx.z * 4;
    int nt = T - t0;
    if (nt > 4) nt = 4;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

#if __CUDA_ARCH__ >= 610
    (void)xf;
    const int xoff = h * 4;
    for (int b = 0; b < nb; b++) {
        size_t boff = (size_t)g * nb + b;
        float ws = d_h2f(w_sc[boff * 16 + r]);
        uint4 wv = ((const uint4*)((const uint8_t*)w_qs + boff * 256))[r];
        int32_t p0 = q4_unpack4(wv.x, h), p1 = q4_unpack4(wv.y, h), p2 = q4_unpack4(wv.z, h), p3 = q4_unpack4(wv.w, h);
        int xb = b * 8 + xoff;
        for (int i = 0; i < nt; i++) {
            const int32_t* xp = xq4 + ((size_t)(t0 + i) * nb) * 8 + xb;
            int4 xv = *(const int4*)xp;
            int isum = __dp4a(p0, xv.x, 0);
            isum = __dp4a(p1, xv.y, isum);
            isum = __dp4a(p2, xv.z, isum);
            isum = __dp4a(p3, xv.w, isum);
            float dx = ws * xd[(size_t)(t0 + i) * nb + b];
            if (i == 0) acc0 = fmaf((float)isum, dx, acc0);
            else if (i == 1) acc1 = fmaf((float)isum, dx, acc1);
            else if (i == 2) acc2 = fmaf((float)isum, dx, acc2);
            else acc3 = fmaf((float)isum, dx, acc3);
        }
    }
#else
    (void)xq4; (void)xd;
    if (nt == 1) {
        const float* xp0 = xf + (size_t)t0 * K;
        for (int b = 0; b < nb; b++) {
            size_t boff = (size_t)g * nb + b;
            float ws = d_h2f(w_sc[boff * 16 + r]);
            uint4 wv = ((const uint4*)((const uint8_t*)w_qs + boff * 256))[r];
            uint32_t u[4] = {wv.x, wv.y, wv.z, wv.w};
            int k0 = b * 32 + h * 16;
            float d = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t n = h ? ((u[j] >> 4) & 0x0F0F0F0Fu) : (u[j] & 0x0F0F0F0Fu);
                float4 xv = *(const float4*)(xp0 + k0 + j * 4);
                d = fmaf((float)((int)(n & 15u) - 8), xv.x, d);
                d = fmaf((float)((int)((n >> 8) & 15u) - 8), xv.y, d);
                d = fmaf((float)((int)((n >> 16) & 15u) - 8), xv.z, d);
                d = fmaf((float)((int)(n >> 24) - 8), xv.w, d);
            }
            acc0 = fmaf(d, ws, acc0);
        }
    } else {
        for (int b = 0; b < nb; b++) {
            size_t boff = (size_t)g * nb + b;
            float ws = d_h2f(w_sc[boff * 16 + r]);
            uint4 wv = ((const uint4*)((const uint8_t*)w_qs + boff * 256))[r];
            uint32_t u[4] = {wv.x, wv.y, wv.z, wv.w};
            int k0 = b * 32 + h * 16;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t n = h ? ((u[j] >> 4) & 0x0F0F0F0Fu) : (u[j] & 0x0F0F0F0Fu);
                float f0 = (float)((int)(n & 15u) - 8);
                float f1 = (float)((int)((n >> 8) & 15u) - 8);
                float f2 = (float)((int)((n >> 16) & 15u) - 8);
                float f3 = (float)((int)(n >> 24) - 8);
                float4 xv0 = *(const float4*)(xf + (size_t)t0 * K + k0 + j * 4);
                d0 = fmaf(f0, xv0.x, fmaf(f1, xv0.y, fmaf(f2, xv0.z, fmaf(f3, xv0.w, d0))));
                float4 xv1 = *(const float4*)(xf + (size_t)(t0 + 1) * K + k0 + j * 4);
                d1 = fmaf(f0, xv1.x, fmaf(f1, xv1.y, fmaf(f2, xv1.z, fmaf(f3, xv1.w, d1))));
                if (nt > 2) {
                    float4 xv2 = *(const float4*)(xf + (size_t)(t0 + 2) * K + k0 + j * 4);
                    d2 = fmaf(f0, xv2.x, fmaf(f1, xv2.y, fmaf(f2, xv2.z, fmaf(f3, xv2.w, d2))));
                }
                if (nt > 3) {
                    float4 xv3 = *(const float4*)(xf + (size_t)(t0 + 3) * K + k0 + j * 4);
                    d3 = fmaf(f0, xv3.x, fmaf(f1, xv3.y, fmaf(f2, xv3.z, fmaf(f3, xv3.w, d3))));
                }
            }
            acc0 = fmaf(d0, ws, acc0);
            acc1 = fmaf(d1, ws, acc1);
            if (nt > 2) acc2 = fmaf(d2, ws, acc2);
            if (nt > 3) acc3 = fmaf(d3, ws, acc3);
        }
    }
#endif

    acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, 1);
    if (nt > 1) acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, 1);
    if (nt > 2) acc2 += __shfl_xor_sync(0xFFFFFFFFu, acc2, 1);
    if (nt > 3) acc3 += __shfl_xor_sync(0xFFFFFFFFu, acc3, 1);
    if (h == 0 && rr < rows) {
        y[(size_t)t0 * ys + rr] = acc0;
        if (nt > 1) y[(size_t)(t0 + 1) * ys + rr] = acc1;
        if (nt > 2) y[(size_t)(t0 + 2) * ys + rr] = acc2;
        if (nt > 3) y[(size_t)(t0 + 3) * ys + rr] = acc3;
    }
}

extern "C" __global__ void k_f32_mm(const float* __restrict__ w, int rows, int K, int T,
                                    const float* __restrict__ x, int xs, float* __restrict__ y, int ys) {
    int rr = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (rr >= rows) return;
    int t0 = blockIdx.z * 4;
    int nt = T - t0;
    if (nt > 4) nt = 4;
    const float* wr = w + (size_t)rr * K;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int k = lane; k < K; k += 32) {
        float wv = wr[k];
        a0 = fmaf(wv, x[(size_t)t0 * xs + k], a0);
        if (nt > 1) a1 = fmaf(wv, x[(size_t)(t0 + 1) * xs + k], a1);
        if (nt > 2) a2 = fmaf(wv, x[(size_t)(t0 + 2) * xs + k], a2);
        if (nt > 3) a3 = fmaf(wv, x[(size_t)(t0 + 3) * xs + k], a3);
    }
    a0 = warp_sum(a0);
    if (nt > 1) a1 = warp_sum(a1);
    if (nt > 2) a2 = warp_sum(a2);
    if (nt > 3) a3 = warp_sum(a3);
    if (lane == 0) {
        y[(size_t)t0 * ys + rr] = a0;
        if (nt > 1) y[(size_t)(t0 + 1) * ys + rr] = a1;
        if (nt > 2) y[(size_t)(t0 + 2) * ys + rr] = a2;
        if (nt > 3) y[(size_t)(t0 + 3) * ys + rr] = a3;
    }
}

extern "C" __global__ void k_bf16_mm(const uint16_t* __restrict__ w, int rows, int K, int T,
                                     const float* __restrict__ x, int xs, float* __restrict__ y, int ys) {
    int rr = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    if (rr >= rows) return;
    const uint16_t* wr = w + (size_t)rr * K;
    for (int t = blockIdx.z; t < T; t += gridDim.z) {
        const float* xt = x + (size_t)t * xs;
        float a = 0.f;
        for (int k = lane; k < K; k += 32) a = fmaf(d_bf16(wr[k]), xt[k], a);
        a = warp_sum(a);
        if (lane == 0) y[(size_t)t * ys + rr] = a;
    }
}

/* ---------------------------------------------------------------- NVFP4G16 & Q5_K MoE Kernels */
__device__ __forceinline__ float init_fp4_lut_f(int lane) {
    int m = (lane < 16) ? (int)((0xC8643210u >> ((lane & 7) << 2)) & 0xFu) : 0;
    return (float)((lane & 8) ? -m : m);
}

__device__ __forceinline__ uint32_t init_fp4_lut_i(int lane) {
    int m = (lane < 16) ? (int)((0xC8643210u >> ((lane & 7) << 2)) & 0xFu) : 0;
    return (uint32_t)(uint8_t)((lane & 8) ? -m : m);
}

__device__ __forceinline__ float nvfp4_word_1(
        uint32_t v, float lut_f, const float* __restrict__ xp, int k_lo, int k_hi, float p) {
    float4 xlo = *(const float4*)(xp + k_lo);
    float4 xhi = *(const float4*)(xp + k_hi);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 0)  & 15u)), xlo.x, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 8)  & 15u)), xlo.y, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 16) & 15u)), xlo.z, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 24) & 15u)), xlo.w, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 4)  & 15u)), xhi.x, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 12) & 15u)), xhi.y, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 20) & 15u)), xhi.z, p);
    p = fmaf(__shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 28) & 15u)), xhi.w, p);
    return p;
}

__device__ __forceinline__ void nvfp4_word_n(
        uint32_t v, float lut_f, const float* xp0, const float* xp1, const float* xp2, const float* xp3,
        int nt, int k_lo, int k_hi, float* p) {
    float f0 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 0)  & 15u));
    float f1 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 8)  & 15u));
    float f2 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 16) & 15u));
    float f3 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 24) & 15u));
    float f4 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 4)  & 15u));
    float f5 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 12) & 15u));
    float f6 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 20) & 15u));
    float f7 = __shfl_sync(0xFFFFFFFFu, lut_f, (int)((v >> 28) & 15u));
    const float* xps[4] = {xp0, xp1, xp2, xp3};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (i >= nt) break;
        float4 xlo = *(const float4*)(xps[i] + k_lo);
        float4 xhi = *(const float4*)(xps[i] + k_hi);
        p[i] = fmaf(f0, xlo.x, fmaf(f1, xlo.y, fmaf(f2, xlo.z, fmaf(f3, xlo.w,
               fmaf(f4, xhi.x, fmaf(f5, xhi.y, fmaf(f6, xhi.z, fmaf(f7, xhi.w, p[i]))))))));
    }
}

__device__ __forceinline__ float nvfp4_step_1_f(
        uint32_t s2, const uint4& qv, float lut_f, const float* __restrict__ xp, int k0, float acc) {
    float d0 = d_ue4m3((uint8_t)(s2 & 0xFFu));
    float d1 = d_ue4m3((uint8_t)(s2 >> 8));
    float p0 = nvfp4_word_1(qv.x, lut_f, xp, k0 + 0,  k0 + 8,  0.f);
    p0       = nvfp4_word_1(qv.y, lut_f, xp, k0 + 4,  k0 + 12, p0);
    float p1 = nvfp4_word_1(qv.z, lut_f, xp, k0 + 16, k0 + 24, 0.f);
    p1       = nvfp4_word_1(qv.w, lut_f, xp, k0 + 20, k0 + 28, p1);
    acc = fmaf(p0, d0, acc);
    return fmaf(p1, d1, acc);
}

__device__ __forceinline__ void nvfp4_step_n_f(
        uint32_t s2, const uint4& qv, float lut_f,
        const float* xp0, const float* xp1, const float* xp2, const float* xp3,
        int nt, int k0, float* acc) {
    float d0 = d_ue4m3((uint8_t)(s2 & 0xFFu));
    float d1 = d_ue4m3((uint8_t)(s2 >> 8));
    float p0[4] = {0.f, 0.f, 0.f, 0.f};
    float p1[4] = {0.f, 0.f, 0.f, 0.f};
    nvfp4_word_n(qv.x, lut_f, xp0, xp1, xp2, xp3, nt, k0 + 0,  k0 + 8,  p0);
    nvfp4_word_n(qv.y, lut_f, xp0, xp1, xp2, xp3, nt, k0 + 4,  k0 + 12, p0);
    nvfp4_word_n(qv.z, lut_f, xp0, xp1, xp2, xp3, nt, k0 + 16, k0 + 24, p1);
    nvfp4_word_n(qv.w, lut_f, xp0, xp1, xp2, xp3, nt, k0 + 20, k0 + 28, p1);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (i < nt) {
            acc[i] = fmaf(p0[i], d0, acc[i]);
            acc[i] = fmaf(p1[i], d1, acc[i]);
        }
    }
}

#if __CUDA_ARCH__ >= 610
__device__ __forceinline__ int32_t pack4_fp4_lo(uint32_t w, uint32_t lut_i) {
    uint32_t b0 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 0)  & 15u));
    uint32_t b1 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 8)  & 15u));
    uint32_t b2 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 16) & 15u));
    uint32_t b3 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 24) & 15u));
    return (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

__device__ __forceinline__ int32_t pack4_fp4_hi(uint32_t w, uint32_t lut_i) {
    uint32_t b0 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 4)  & 15u));
    uint32_t b1 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 12) & 15u));
    uint32_t b2 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 20) & 15u));
    uint32_t b3 = __shfl_sync(0xFFFFFFFFu, lut_i, (int)((w >> 28) & 15u));
    return (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
}

__device__ __forceinline__ float nvfp4_step_1_dp4a(
        uint32_t s2, const uint4& qv, uint32_t lut_i, int h, int b,
        const int32_t* __restrict__ xq4, const float* __restrict__ xd, float acc) {
    float xds = xd[b * 2 + h];
    int x0 = (b * 2 + h) * 8;
    int is0 = __dp4a(pack4_fp4_lo(qv.x, lut_i), xq4[x0 + 0], 0);
    is0 =     __dp4a(pack4_fp4_lo(qv.y, lut_i), xq4[x0 + 1], is0);
    is0 =     __dp4a(pack4_fp4_hi(qv.x, lut_i), xq4[x0 + 2], is0);
    is0 =     __dp4a(pack4_fp4_hi(qv.y, lut_i), xq4[x0 + 3], is0);
    int is1 = __dp4a(pack4_fp4_lo(qv.z, lut_i), xq4[x0 + 4], 0);
    is1 =     __dp4a(pack4_fp4_lo(qv.w, lut_i), xq4[x0 + 5], is1);
    is1 =     __dp4a(pack4_fp4_hi(qv.z, lut_i), xq4[x0 + 6], is1);
    is1 =     __dp4a(pack4_fp4_hi(qv.w, lut_i), xq4[x0 + 7], is1);
    acc = fmaf((float)is0, d_ue4m3((uint8_t)(s2 & 0xFFu)) * xds, acc);
    return fmaf((float)is1, d_ue4m3((uint8_t)(s2 >> 8)) * xds, acc);
}

__device__ __forceinline__ void nvfp4_step_n_dp4a(
        uint32_t s2, const uint4& qv, uint32_t lut_i, int h, int b, int nt,
        const int32_t* const* xq4s, const float* const* xds, float* acc) {
    float d0 = d_ue4m3((uint8_t)(s2 & 0xFFu));
    float d1 = d_ue4m3((uint8_t)(s2 >> 8));
    int x0 = (b * 2 + h) * 8;
    int32_t p0 = pack4_fp4_lo(qv.x, lut_i);
    int32_t p1 = pack4_fp4_lo(qv.y, lut_i);
    int32_t p2 = pack4_fp4_hi(qv.x, lut_i);
    int32_t p3 = pack4_fp4_hi(qv.y, lut_i);
    int32_t p4 = pack4_fp4_lo(qv.z, lut_i);
    int32_t p5 = pack4_fp4_lo(qv.w, lut_i);
    int32_t p6 = pack4_fp4_hi(qv.z, lut_i);
    int32_t p7 = pack4_fp4_hi(qv.w, lut_i);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (i >= nt) break;
        const int32_t* xp = xq4s[i] + x0;
        float xd_val = xds[i][b * 2 + h];
        int is0 = __dp4a(p0, xp[0], 0);
        is0 =     __dp4a(p1, xp[1], is0);
        is0 =     __dp4a(p2, xp[2], is0);
        is0 =     __dp4a(p3, xp[3], is0);
        int is1 = __dp4a(p4, xp[4], 0);
        is1 =     __dp4a(p5, xp[5], is1);
        is1 =     __dp4a(p6, xp[6], is1);
        is1 =     __dp4a(p7, xp[7], is1);
        acc[i] = fmaf((float)is0, d0 * xd_val, acc[i]);
        acc[i] = fmaf((float)is1, d1 * xd_val, acc[i]);
    }
}
#endif

extern "C" __global__ void k_bucket_moe(int T, const int* __restrict__ sel, const void* const* __restrict__ exp_sc_gu,
                                        const uint8_t* __restrict__ is_cached_or_gpu,
                                        int* __restrict__ d_uniq_e, int* __restrict__ d_u_off,
                                        int* __restrict__ d_tok_idx, int* __restrict__ d_tok_slot,
                                        int* __restrict__ d_uniq_n, uint8_t* __restrict__ d_has_cold) {
    __shared__ int cnt[IK_NE];
    __shared__ int off[IK_NE + 1];
    __shared__ int seen[IK_NE];
    __shared__ int nuniq;
    __shared__ int cold;
    int tid = threadIdx.x;
    if (tid < IK_NE) { cnt[tid] = 0; seen[tid] = -1; }
    if (tid == 0) { nuniq = 0; cold = 0; }
    __syncthreads();
    if (tid == 0) {
        int tot = T * IK_NU;
        for (int i = 0; i < tot; i++) {
            int e = sel[i];
            int ok = is_cached_or_gpu ? (is_cached_or_gpu[i] != 0) : (exp_sc_gu[e] != NULL);
            if (!ok) { cold = 1; continue; }
            if (seen[e] < 0) {
                int u = nuniq++;
                seen[e] = u;
                d_uniq_e[u] = e;
            }
            cnt[seen[e]]++;
        }
        off[0] = 0;
        for (int u = 0; u < nuniq; u++) { off[u + 1] = off[u] + cnt[u]; cnt[u] = 0; }
        for (int u = 0; u <= nuniq; u++) d_u_off[u] = off[u];
        for (int i = 0; i < tot; i++) {
            int e = sel[i];
            int ok = is_cached_or_gpu ? (is_cached_or_gpu[i] != 0) : (exp_sc_gu[e] != NULL);
            if (!ok) continue;
            int u = seen[e];
            int pos = off[u] + cnt[u]++;
            d_tok_idx[pos] = i / IK_NU;
            d_tok_slot[pos] = i % IK_NU;
        }
        *d_uniq_n = nuniq;
        *d_has_cold = (uint8_t)cold;
    }
}

extern "C" __global__ void k_nvfp4g16_moe_gu(
        const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off, const int* __restrict__ d_tok_idx,
        const uint8_t* const* __restrict__ exp_gu_sc, const uint8_t* const* __restrict__ exp_gu_qs,
        const float* __restrict__ exp_scales,
        const float* __restrict__ xf, const int32_t* __restrict__ xq4, const float* __restrict__ xd,
        float* __restrict__ d_moe_h, int uniq_n) {
    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    int r = t >> 1;
    int h = t & 1;
    int rr = g_row * 16 + r;
    if (g_row * 16 >= IK_FF) return;
    int e = d_uniq_e[u];
    const uint8_t* sc = exp_gu_sc[e];
    const uint8_t* qs = exp_gu_qs[e];
    if (!sc || !qs) return;

    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    float ws = exp_scales ? exp_scales[e] : 1.0f;
    const int nb64 = IK_D / QK_NV;
    const size_t up_g_off = (size_t)(IK_FF / 16) * nb64;

#if __CUDA_ARCH__ >= 610
    (void)xf;
    uint32_t lut_i = init_fp4_lut_i(t);
    int nb32 = IK_D >> 5;
    if (p1 - p0 == 1) {
        int tk0 = d_tok_idx[p0];
        const int32_t* xq0 = xq4 + (size_t)tk0 * nb32 * 8;
        const float*   xd0 = xd  + (size_t)tk0 * nb32;
        float ag = 0.f, au = 0.f;
        for (int b = 0; b < nb64; b++) {
            size_t og = (size_t)g_row * nb64 + b;
            size_t ou = up_g_off + og;
            uint32_t sg = ((const uint16_t*)(sc + og * 64))[t];
            uint4    qg = ((const uint4*)(qs + og * 512))[t];
            ag = nvfp4_step_1_dp4a(sg, qg, lut_i, h, b, xq0, xd0, ag);
            uint32_t su = ((const uint16_t*)(sc + ou * 64))[t];
            uint4    qu = ((const uint4*)(qs + ou * 512))[t];
            au = nvfp4_step_1_dp4a(su, qu, lut_i, h, b, xq0, xd0, au);
        }
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 1);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 1);
        if (h == 0 && rr < IK_FF) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(ag * ws, au * ws);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const int32_t* xq4s[4] = {NULL, NULL, NULL, NULL};
            const float*   xds[4]  = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) {
                int tk = d_tok_idx[base_p + i];
                xq4s[i] = xq4 + (size_t)tk * nb32 * 8;
                xds[i]  = xd  + (size_t)tk * nb32;
            }
            float ag[4] = {0.f, 0.f, 0.f, 0.f};
            float au[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb64; b++) {
                size_t og = (size_t)g_row * nb64 + b;
                size_t ou = up_g_off + og;
                uint32_t sg = ((const uint16_t*)(sc + og * 64))[t];
                uint4    qg = ((const uint4*)(qs + og * 512))[t];
                nvfp4_step_n_dp4a(sg, qg, lut_i, h, b, nt, xq4s, xds, ag);
                uint32_t su = ((const uint16_t*)(sc + ou * 64))[t];
                uint4    qu = ((const uint4*)(qs + ou * 512))[t];
                nvfp4_step_n_dp4a(su, qu, lut_i, h, b, nt, xq4s, xds, au);
            }
            for (int i = 0; i < nt; i++) {
                ag[i] += __shfl_xor_sync(0xFFFFFFFFu, ag[i], 1);
                au[i] += __shfl_xor_sync(0xFFFFFFFFu, au[i], 1);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + i) * IK_FF + rr] = d_situ(ag[i] * ws, au[i] * ws);
            }
        }
    }
#else
    (void)xq4; (void)xd;
    float lut_f = init_fp4_lut_f(t);
    if (p1 - p0 == 1) {
        const float* xp0 = xf + (size_t)d_tok_idx[p0] * IK_D;
        float ag = 0.f, au = 0.f;
        for (int b = 0; b < nb64; b++) {
            size_t og = (size_t)g_row * nb64 + b;
            size_t ou = up_g_off + og;
            int k0 = b * 64 + h * 32;
            uint32_t sg = ((const uint16_t*)(sc + og * 64))[t];
            uint4    qg = ((const uint4*)(qs + og * 512))[t];
            ag = nvfp4_step_1_f(sg, qg, lut_f, xp0, k0, ag);
            uint32_t su = ((const uint16_t*)(sc + ou * 64))[t];
            uint4    qu = ((const uint4*)(qs + ou * 512))[t];
            au = nvfp4_step_1_f(su, qu, lut_f, xp0, k0, au);
        }
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 1);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 1);
        if (h == 0 && rr < IK_FF) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(ag * ws, au * ws);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const float* xp[4] = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) xp[i] = xf + (size_t)d_tok_idx[base_p + i] * IK_D;
            float ag[4] = {0.f, 0.f, 0.f, 0.f};
            float au[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb64; b++) {
                size_t og = (size_t)g_row * nb64 + b;
                size_t ou = up_g_off + og;
                int k0 = b * 64 + h * 32;
                uint32_t sg = ((const uint16_t*)(sc + og * 64))[t];
                uint4    qg = ((const uint4*)(qs + og * 512))[t];
                nvfp4_step_n_f(sg, qg, lut_f, xp[0], xp[1], xp[2], xp[3], nt, k0, ag);
                uint32_t su = ((const uint16_t*)(sc + ou * 64))[t];
                uint4    qu = ((const uint4*)(qs + ou * 512))[t];
                nvfp4_step_n_f(su, qu, lut_f, xp[0], xp[1], xp[2], xp[3], nt, k0, au);
            }
            for (int i = 0; i < nt; i++) {
                ag[i] += __shfl_xor_sync(0xFFFFFFFFu, ag[i], 1);
                au[i] += __shfl_xor_sync(0xFFFFFFFFu, au[i], 1);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + i) * IK_FF + rr] = d_situ(ag[i] * ws, au[i] * ws);
            }
        }
    }
#endif
}

extern "C" __global__ void k_nvfp4g16_moe_dn(
        const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off,
        const int* __restrict__ d_tok_idx, const int* __restrict__ d_tok_slot,
        const uint8_t* const* __restrict__ exp_dn_sc, const uint8_t* const* __restrict__ exp_dn_qs,
        const float* __restrict__ exp_scales,
        const float* __restrict__ d_moe_h, const int32_t* __restrict__ hq4, const float* __restrict__ hd,
        float* __restrict__ slot_outs, int uniq_n) {
    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    int r = t >> 1;
    int h = t & 1;
    int rr = g_row * 16 + r;
    if (g_row * 16 >= IK_D) return;
    int e = d_uniq_e[u];
    const uint8_t* sc = exp_dn_sc[e];
    const uint8_t* qs = exp_dn_qs[e];
    if (!sc || !qs) return;

    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    float ws = exp_scales ? exp_scales[e] : 1.0f;
    const int nb64 = IK_FF / QK_NV;

#if __CUDA_ARCH__ >= 610
    (void)d_moe_h;
    uint32_t lut_i = init_fp4_lut_i(t);
    int nb32 = IK_FF >> 5;
    if (p1 - p0 == 1) {
        const int32_t* hq0 = hq4 + (size_t)p0 * nb32 * 8;
        const float*   hd0 = hd  + (size_t)p0 * nb32;
        float a0 = 0.f;
        for (int b = 0; b < nb64; b++) {
            size_t ob = (size_t)g_row * nb64 + b;
            uint32_t sd = ((const uint16_t*)(sc + ob * 64))[t];
            uint4    qd = ((const uint4*)(qs + ob * 512))[t];
            a0 = nvfp4_step_1_dp4a(sd, qd, lut_i, h, b, hq0, hd0, a0);
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1);
        if (h == 0 && rr < IK_D) {
            int tk = d_tok_idx[p0], sl = d_tok_slot[p0];
            slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a0 * ws;
        }
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const int32_t* hq4s[4] = {NULL, NULL, NULL, NULL};
            const float*   hds[4]  = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) {
                hq4s[i] = hq4 + (size_t)(base_p + i) * nb32 * 8;
                hds[i]  = hd  + (size_t)(base_p + i) * nb32;
            }
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb64; b++) {
                size_t ob = (size_t)g_row * nb64 + b;
                uint32_t sd = ((const uint16_t*)(sc + ob * 64))[t];
                uint4    qd = ((const uint4*)(qs + ob * 512))[t];
                nvfp4_step_n_dp4a(sd, qd, lut_i, h, b, nt, hq4s, hds, a);
            }
            for (int i = 0; i < nt; i++) {
                a[i] += __shfl_xor_sync(0xFFFFFFFFu, a[i], 1);
                if (h == 0 && rr < IK_D) {
                    int tk = d_tok_idx[base_p + i], sl = d_tok_slot[base_p + i];
                    slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a[i] * ws;
                }
            }
        }
    }
#else
    (void)hq4; (void)hd;
    float lut_f = init_fp4_lut_f(t);
    if (p1 - p0 == 1) {
        const float* hp0 = d_moe_h + (size_t)p0 * IK_FF;
        float a0 = 0.f;
        for (int b = 0; b < nb64; b++) {
            size_t ob = (size_t)g_row * nb64 + b;
            uint32_t sd = ((const uint16_t*)(sc + ob * 64))[t];
            uint4    qd = ((const uint4*)(qs + ob * 512))[t];
            a0 = nvfp4_step_1_f(sd, qd, lut_f, hp0, b * 64 + h * 32, a0);
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1);
        if (h == 0 && rr < IK_D) {
            int tk = d_tok_idx[p0], sl = d_tok_slot[p0];
            slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a0 * ws;
        }
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const float* hp[4] = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) hp[i] = d_moe_h + (size_t)(base_p + i) * IK_FF;
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb64; b++) {
                size_t ob = (size_t)g_row * nb64 + b;
                uint32_t sd = ((const uint16_t*)(sc + ob * 64))[t];
                uint4    qd = ((const uint4*)(qs + ob * 512))[t];
                nvfp4_step_n_f(sd, qd, lut_f, hp[0], hp[1], hp[2], hp[3], nt, b * 64 + h * 32, a);
            }
            for (int i = 0; i < nt; i++) {
                a[i] += __shfl_xor_sync(0xFFFFFFFFu, a[i], 1);
                if (h == 0 && rr < IK_D) {
                    int tk = d_tok_idx[base_p + i], sl = d_tok_slot[base_p + i];
                    slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a[i] * ws;
                }
            }
        }
    }
#endif
}

__device__ __forceinline__ float q5k_row_dot_warp(const uint8_t* __restrict__ wr, int nbk,
                                                  const float* __restrict__ x, int lane) {
    float sumf = 0.0f;
    for (int b = 0; b < nbk; b++) {
        const uint8_t* w = wr + (size_t)b * BQ5K;
        uint16_t hd = (uint16_t)w[0] | ((uint16_t)w[1] << 8);
        uint16_t hm = (uint16_t)w[2] | ((uint16_t)w[3] << 8);
        float d = d_h2f(hd), dmin = d_h2f(hm);
        const uint8_t* sc_ptr = w + 4;
        const uint8_t* qh_ptr = w + 16;
        const uint8_t* qs_ptr = w + 48;
        uint8_t qhb = qh_ptr[lane];
        uint8_t hbit = 1;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int j0 = 2 * n, j1 = 2 * n + 1;
            uint8_t sc0 = (j0 < 4) ? (sc_ptr[j0] & 63u) : ((sc_ptr[j0 + 4] & 0xFu) | ((sc_ptr[j0 - 4] >> 6) << 4));
            uint8_t mn0 = (j0 < 4) ? (sc_ptr[j0 + 4] & 63u) : ((sc_ptr[j0 + 4] >> 4) | ((sc_ptr[j0] >> 6) << 4));
            uint8_t sc1 = (j1 < 4) ? (sc_ptr[j1] & 63u) : ((sc_ptr[j1 + 4] & 0xFu) | ((sc_ptr[j1 - 4] >> 6) << 4));
            uint8_t mn1 = (j1 < 4) ? (sc_ptr[j1 + 4] & 63u) : ((sc_ptr[j1 + 4] >> 4) | ((sc_ptr[j1] >> 6) << 4));
            uint8_t qb = qs_ptr[n * 32 + lane];
            float q0 = (float)((qb & 15u) + ((qhb & hbit) ? 16u : 0u));
            float q1 = (float)((qb >> 4)  + ((qhb & (hbit << 1)) ? 16u : 0u));
            float x0 = x[b * 256 + n * 64 + lane];
            float x1 = x[b * 256 + n * 64 + 32 + lane];
            sumf += (d * (float)sc0 * q0 - dmin * (float)mn0) * x0;
            sumf += (d * (float)sc1 * q1 - dmin * (float)mn1) * x1;
            hbit <<= 2;
        }
    }
    return warp_sum(sumf);
}

__device__ __forceinline__ void q5k_row_dot_n_warp(
        const uint8_t* __restrict__ wr, int nbk,
        const float* const* xptrs, int nt, float* out_acc, int lane) {
    float sumf[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < nbk; b++) {
        const uint8_t* w = wr + (size_t)b * BQ5K;
        uint16_t hd = (uint16_t)w[0] | ((uint16_t)w[1] << 8);
        uint16_t hm = (uint16_t)w[2] | ((uint16_t)w[3] << 8);
        float d = d_h2f(hd), dmin = d_h2f(hm);
        const uint8_t* sc_ptr = w + 4;
        const uint8_t* qh_ptr = w + 16;
        const uint8_t* qs_ptr = w + 48;
        uint8_t qhb = qh_ptr[lane];
        uint8_t hbit = 1;
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int j0 = 2 * n, j1 = 2 * n + 1;
            uint8_t sc0 = (j0 < 4) ? (sc_ptr[j0] & 63u) : ((sc_ptr[j0 + 4] & 0xFu) | ((sc_ptr[j0 - 4] >> 6) << 4));
            uint8_t mn0 = (j0 < 4) ? (sc_ptr[j0 + 4] & 63u) : ((sc_ptr[j0 + 4] >> 4) | ((sc_ptr[j0] >> 6) << 4));
            uint8_t sc1 = (j1 < 4) ? (sc_ptr[j1] & 63u) : ((sc_ptr[j1 + 4] & 0xFu) | ((sc_ptr[j1 - 4] >> 6) << 4));
            uint8_t mn1 = (j1 < 4) ? (sc_ptr[j1 + 4] & 63u) : ((sc_ptr[j1 + 4] >> 4) | ((sc_ptr[j1] >> 6) << 4));
            uint8_t qb = qs_ptr[n * 32 + lane];
            float w0 = d * (float)sc0 * (float)((qb & 15u) + ((qhb & hbit) ? 16u : 0u)) - dmin * (float)mn0;
            float w1 = d * (float)sc1 * (float)((qb >> 4)  + ((qhb & (hbit << 1)) ? 16u : 0u)) - dmin * (float)mn1;
            int off0 = b * 256 + n * 64 + lane;
            int off1 = off0 + 32;
            for (int t = 0; t < nt; t++) {
                const float* xt = xptrs[t];
                sumf[t] = fmaf(w0, xt[off0], fmaf(w1, xt[off1], sumf[t]));
            }
            hbit <<= 2;
        }
    }
    for (int t = 0; t < nt; t++) out_acc[t] = warp_sum(sumf[t]);
}

extern "C" __global__ void k_q5k_moe_gu(const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off,
                                        const int* __restrict__ d_tok_idx, const uint8_t* const* __restrict__ exp_q5_gu,
                                        const float* __restrict__ xf, float* __restrict__ d_moe_h, int uniq_n) {
    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int e = d_uniq_e[u];
    const uint8_t* w = exp_q5_gu[e];
    if (!w) return;
    int rr = blockIdx.x * blockDim.y + threadIdx.y;
    if (rr >= IK_FF) return;
    int lane = threadIdx.x;
    const int nbk = IK_D / QK_K;
    const size_t rb = (size_t)nbk * BQ5K;
    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    const uint8_t* wg = w + (size_t)rr * rb;
    const uint8_t* wu = w + (size_t)(IK_FF + rr) * rb;
    if (p1 - p0 == 1) {
        int tk = d_tok_idx[p0];
        float g = q5k_row_dot_warp(wg, nbk, xf + (size_t)tk * IK_D, lane);
        float up = q5k_row_dot_warp(wu, nbk, xf + (size_t)tk * IK_D, lane);
        if (lane == 0) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(g, up);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 8) {
            int nt = p1 - base_p; if (nt > 8) nt = 8;
            const float* xptrs[8];
            for (int i = 0; i < nt; i++) xptrs[i] = xf + (size_t)d_tok_idx[base_p + i] * IK_D;
            float g_acc[8], u_acc[8];
            q5k_row_dot_n_warp(wg, nbk, xptrs, nt, g_acc, lane);
            q5k_row_dot_n_warp(wu, nbk, xptrs, nt, u_acc, lane);
            if (lane == 0) {
                for (int i = 0; i < nt; i++) d_moe_h[(size_t)(base_p + i) * IK_FF + rr] = d_situ(g_acc[i], u_acc[i]);
            }
        }
    }
}

extern "C" __global__ void k_q8g16_moe_dn(const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off,
                                          const int* __restrict__ d_tok_idx, const int* __restrict__ d_tok_slot,
                                          const uint16_t* const* __restrict__ exp_dn_sc, const int8_t* const* __restrict__ exp_dn_qs,
                                          const float* __restrict__ d_moe_h, const int32_t* __restrict__ hq4, const float* __restrict__ hd,
                                          float* __restrict__ slot_outs, int uniq_n) {
    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    int t = threadIdx.x;
    int r = t >> 1;
    int h = t & 1;
    int rr = g_row * 16 + r;
    if (g_row * 16 >= IK_D) return;
    int e = d_uniq_e[u];
    const uint16_t* wsc = exp_dn_sc[e];
    const int8_t*   wqs = exp_dn_qs[e];
    if (!wsc || !wqs) return;
    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    const int nb = IK_FF / QK_Q8;

    for (int base_p = p0; base_p < p1; base_p += 4) {
        int nt = p1 - base_p; if (nt > 4) nt = 4;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int b = 0; b < nb; b++) {
            size_t ob = (size_t)g_row * nb + b;
            float ws = d_h2f(wsc[ob * 16 + r]);
            int4 wv = ((const int4*)(wqs + ob * 512))[t];
#if __CUDA_ARCH__ >= 610
            int xb = b * 8 + h * 4;
            #define DO_Q8DP(PI, AC) do { \
                int4 xv = *(const int4*)(hq4 + (size_t)(PI) * nb * 8 + xb); \
                int is = __dp4a(wv.x, xv.x, 0); is = __dp4a(wv.y, xv.y, is); is = __dp4a(wv.z, xv.z, is); is = __dp4a(wv.w, xv.w, is); \
                AC = fmaf((float)is, ws * hd[(size_t)(PI) * nb + b], AC); \
            } while (0)
            DO_Q8DP(base_p + 0, a0);
            if (nt > 1) DO_Q8DP(base_p + 1, a1);
            if (nt > 2) DO_Q8DP(base_p + 2, a2);
            if (nt > 3) DO_Q8DP(base_p + 3, a3);
            #undef DO_Q8DP
#else
            (void)hq4; (void)hd;
            uint32_t w[4] = {(uint32_t)wv.x, (uint32_t)wv.y, (uint32_t)wv.z, (uint32_t)wv.w};
            int k0 = b * 32 + h * 16;
            const float* xp0 = d_moe_h + (size_t)(base_p + 0) * IK_FF + k0;
            const float* xp1 = (nt > 1) ? (d_moe_h + (size_t)(base_p + 1) * IK_FF + k0) : NULL;
            const float* xp2 = (nt > 2) ? (d_moe_h + (size_t)(base_p + 2) * IK_FF + k0) : NULL;
            const float* xp3 = (nt > 3) ? (d_moe_h + (size_t)(base_p + 3) * IK_FF + k0) : NULL;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t p = w[j];
                float f0 = (float)((int32_t)(p << 24) >> 24);
                float f1 = (float)((int32_t)(p << 16) >> 24);
                float f2 = (float)((int32_t)(p << 8)  >> 24);
                float f3 = (float)((int32_t)p >> 24);
                float4 xv0 = *(const float4*)(xp0 + j * 4);
                d0 = fmaf(f0, xv0.x, fmaf(f1, xv0.y, fmaf(f2, xv0.z, fmaf(f3, xv0.w, d0))));
                if (nt > 1) {
                    float4 xv1 = *(const float4*)(xp1 + j * 4);
                    d1 = fmaf(f0, xv1.x, fmaf(f1, xv1.y, fmaf(f2, xv1.z, fmaf(f3, xv1.w, d1))));
                }
                if (nt > 2) {
                    float4 xv2 = *(const float4*)(xp2 + j * 4);
                    d2 = fmaf(f0, xv2.x, fmaf(f1, xv2.y, fmaf(f2, xv2.z, fmaf(f3, xv2.w, d2))));
                }
                if (nt > 3) {
                    float4 xv3 = *(const float4*)(xp3 + j * 4);
                    d3 = fmaf(f0, xv3.x, fmaf(f1, xv3.y, fmaf(f2, xv3.z, fmaf(f3, xv3.w, d3))));
                }
            }
            a0 = fmaf(d0, ws, a0);
            if (nt > 1) a1 = fmaf(d1, ws, a1);
            if (nt > 2) a2 = fmaf(d2, ws, a2);
            if (nt > 3) a3 = fmaf(d3, ws, a3);
#endif
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1);
        if (nt > 1) a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 1);
        if (nt > 2) a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 1);
        if (nt > 3) a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 1);
        if (h == 0 && rr < IK_D) {
            for (int i = 0; i < nt; i++) {
                float val = (i == 0 ? a0 : i == 1 ? a1 : i == 2 ? a2 : a3);
                int tk = d_tok_idx[base_p + i], sl = d_tok_slot[base_p + i];
                slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = val;
            }
        }
    }
}

/* ---------------------------------------------------------------- Core Transformer Kernels */
extern "C" __global__ void k_rmsnorm(float* __restrict__ dst, const float* __restrict__ src,
                                     const float* __restrict__ w, int n, int T) {
    int t = blockIdx.x;
    if (t >= T) return;
    int tid = threadIdx.x;
    const float* xs = src + (size_t)t * n;
    float* yd = dst + (size_t)t * n;
    float ss = 0.0f;
    for (int i = tid; i < n; i += 256) {
        float v = xs[i];
        ss = fmaf(v, v, ss);
    }
    float inv = rsqrtf(block_reduce_sum_256(ss) / (float)n + IK_RMS_EPS);
    if (w) {
        for (int i = tid; i < n; i += 256) yd[i] = xs[i] * inv * w[i];
    } else {
        for (int i = tid; i < n; i += 256) yd[i] = xs[i] * inv;
    }
}

extern "C" __global__ void k_mix(const float* __restrict__ b0, const float* __restrict__ b1,
                                 const float* __restrict__ b2, const float* __restrict__ b3,
                                 int nblk, const float* __restrict__ cur,
                                 const float* __restrict__ query, int nq, int T, float* __restrict__ out) {
    int t = blockIdx.x;
    if (t >= T) return;
    int tid = threadIdx.x;
    const float* ptrs[5] = {NULL, NULL, NULL, NULL, NULL};
    if (nblk > 0) ptrs[0] = b0 + (size_t)t * IK_D;
    if (nblk > 1) ptrs[1] = b1 + (size_t)t * IK_D;
    if (nblk > 2) ptrs[2] = b2 + (size_t)t * IK_D;
    if (nblk > 3) ptrs[3] = b3 + (size_t)t * IK_D;
    ptrs[nblk] = cur + (size_t)t * IK_D;
    int ns = nblk + 1;

    __shared__ float s_inv[6];
    __shared__ float s_w[4][6];

    for (int s = 0; s < ns; s++) {
        float ss = 0.0f;
        const float* p = ptrs[s];
        for (int i = tid; i < IK_D; i += 256) {
            float v = p[i];
            ss = fmaf(v, v, ss);
        }
        float tot = block_reduce_sum_256(ss);
        if (tid == 0) s_inv[s] = rsqrtf(tot / (float)IK_D + IK_RMS_EPS);
        __syncthreads();
    }

    float inv_sqrt_d = 1.0f / 48.0f;
    for (int q = 0; q < nq; q++) {
        const float* wq = query + (size_t)q * IK_D;
        for (int s = 0; s < ns; s++) {
            const float* p = ptrs[s];
            float d = 0.0f;
            for (int i = tid; i < IK_D; i += 256) d = fmaf(p[i], wq[i], d);
            float tot = block_reduce_sum_256(d);
            if (tid == 0) s_w[q][s] = tot * s_inv[s] * inv_sqrt_d;
            __syncthreads();
        }
        if (tid == 0) {
            float mx = -1e30f;
            for (int s = 0; s < ns; s++) if (s_w[q][s] > mx) mx = s_w[q][s];
            float sum = 0.0f;
            for (int s = 0; s < ns; s++) { s_w[q][s] = expf(s_w[q][s] - mx); sum += s_w[q][s]; }
            float inv = 1.0f / sum;
            for (int s = 0; s < ns; s++) s_w[q][s] *= inv;
        }
        __syncthreads();
        float* dst = out + ((size_t)t * nq + q) * IK_D;
        for (int i = tid; i < IK_D; i += 256) {
            float acc = 0.0f;
            for (int s = 0; s < ns; s++) acc = fmaf(s_w[q][s] * s_inv[s], ptrs[s][i], acc);
            dst[i] = acc;
        }
    }
}

extern "C" __global__ void k_split3(const float* __restrict__ src, float* __restrict__ a,
                                    float* __restrict__ b, float* __restrict__ c, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i >= IK_D || t >= T) return;
    const float* s = src + (size_t)t * 3 * IK_D;
    a[(size_t)t * IK_D + i] = s[i];
    b[(size_t)t * IK_D + i] = s[IK_D + i];
    c[(size_t)t * IK_D + i] = s[2 * IK_D + i];
}

extern "C" __global__ void k_ctrl_concat_norm(const float* __restrict__ cur, const float* __restrict__ msg,
                                              int msg_stride, int T, float* __restrict__ ctrl_in) {
    int t = blockIdx.y;
    int which = blockIdx.x;
    int tid = threadIdx.x;
    const float* src = (which == 0) ? (cur + (size_t)t * IK_D) : (msg + (size_t)t * msg_stride);
    float* dst = ctrl_in + (size_t)t * (2 * IK_D) + which * IK_D;
    float ss = 0.0f;
    for (int i = tid; i < IK_D; i += 256) {
        float v = src[i];
        ss = fmaf(v, v, ss);
    }
    float inv = rsqrtf(block_reduce_sum_256(ss) / (float)IK_D + IK_RMS_EPS);
    for (int i = tid; i < IK_D; i += 256) dst[i] = src[i] * inv;
}

extern "C" __global__ void k_ctrl_apply(const float* __restrict__ w_gain, const float* __restrict__ w_forget,
                                        const float* __restrict__ g, float* __restrict__ cur,
                                        const float* __restrict__ msg, int msg_stride, int T) {
    int rr = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    __shared__ float sg[IK_GRANK];
    if (threadIdx.x < IK_GRANK) sg[threadIdx.x] = g[(size_t)t * IK_GRANK + threadIdx.x];
    __syncthreads();
    if (rr >= IK_D) return;
    const float* wg = w_gain + (size_t)rr * IK_GRANK;
    const float* wf = w_forget + (size_t)rr * IK_GRANK;
    float ag = 0.0f, af = 0.0f;
    #pragma unroll
    for (int k = 0; k < IK_GRANK; k++) {
        float gv = sg[k];
        ag = fmaf(wg[k], gv, ag);
        af = fmaf(wf[k], gv, af);
    }
    float c = cur[(size_t)t * IK_D + rr];
    float m = msg[(size_t)t * msg_stride + rr];
    cur[(size_t)t * IK_D + rr] = (1.0f - tanhf(af)) * c + tanhf(ag) * m;
}

extern "C" __global__ void k_ctrl_apply_pre(const float* __restrict__ gn, const float* __restrict__ fg,
                                            float* __restrict__ cur, const float* __restrict__ msg,
                                            int msg_stride, int T) {
    int rr = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (rr >= IK_D || t >= T) return;
    float c = cur[(size_t)t * IK_D + rr];
    float m = msg[(size_t)t * msg_stride + rr];
    float gnv = gn[(size_t)t * IK_D + rr];
    float fgv = fg[(size_t)t * IK_D + rr];
    cur[(size_t)t * IK_D + rr] = (1.0f - tanhf(fgv)) * c + tanhf(gnv) * m;
}

extern "C" __global__ void k_convolve(float* __restrict__ cur, int C, int T,
                                      const float* __restrict__ weight, const float* __restrict__ dyn,
                                      float* __restrict__ hist_slot, float* __restrict__ trace_slot, int ch_off) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* h0 = hist_slot + ch_off;
    float* h1 = hist_slot + IK_CHSCONV + ch_off;
    float* h2 = hist_slot + 2 * IK_CHSCONV + ch_off;
    float x0 = h0[c], x1 = h1[c], x2 = h2[c];
    if (trace_slot && T <= 8) {
        trace_slot[0 * IK_CHSCONV + ch_off + c] = x0;
        trace_slot[1 * IK_CHSCONV + ch_off + c] = x1;
        trace_slot[2 * IK_CHSCONV + ch_off + c] = x2;
    }
    const float* w = weight + c * 4;
    float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    float sc = dyn ? (1.0f / 3.0f) : rsqrtf(5.0f);
    int n_dyn_heads = dyn ? (C / IK_HD) : 0;
    int h = dyn ? (c / IK_HD) : 0;
    for (int t = 0; t < T; t++) {
        float x3 = cur[(size_t)t * C + c];
        if (trace_slot && T <= 8) {
            trace_slot[(size_t)(3 + t) * IK_CHSCONV + ch_off + c] = x3;
        }
        float o = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
        if (dyn) {
            const float* dt = dyn + ((size_t)t * n_dyn_heads + h) * 4;
            o += x3 * dt[0] + x2 * dt[1] + x1 * dt[2] + x0 * dt[3];
        }
        cur[(size_t)t * C + c] = (x3 + o) * sc;
        x0 = x1; x1 = x2; x2 = x3;
    }
    h0[c] = x0; h1[c] = x1; h2[c] = x2;
}

extern "C" __global__ void k_rollback_conv(float* __restrict__ hist_slot, const float* __restrict__ trace_slot, int k_acc) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= IK_CHSCONV) return;
    hist_slot[0 * IK_CHSCONV + c] = trace_slot[(size_t)(k_acc + 0) * IK_CHSCONV + c];
    hist_slot[1 * IK_CHSCONV + c] = trace_slot[(size_t)(k_acc + 1) * IK_CHSCONV + c];
    hist_slot[2 * IK_CHSCONV + c] = trace_slot[(size_t)(k_acc + 2) * IK_CHSCONV + c];
}

extern "C" __global__ void k_clip_vnorm_kvstore(
        float* __restrict__ Q, float* __restrict__ K, float* __restrict__ V, float* __restrict__ R,
        const float* __restrict__ q_clip, const float* __restrict__ rel_clip,
        uint16_t* __restrict__ kv_k, uint16_t* __restrict__ kv_v,
        int kv_cap, int is_swa, int pos0, int T) {
    int blk = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    float ratio = (float)(pos0 + t + 1) / IK_LOG_FLOOR;
    float tau = is_swa ? 1.0f : (1.0f + IK_LOG_ALPHA * logf(ratio > 1.0f ? ratio : 1.0f));
    int slot = is_swa ? ((pos0 + t) % kv_cap) : (pos0 + t);

    if (blk < 24) {
        int h = blk;
        float* p = Q + ((size_t)t * IK_NH + h) * IK_HD;
        float v = p[tid] * q_clip[(size_t)h * IK_HD + tid];
        float ss = v * v;
        ss = warp_sum(ss);
        __shared__ float sm[4];
        if ((tid & 31) == 0) sm[tid >> 5] = ss;
        __syncthreads();
        float tot = sm[0] + sm[1] + sm[2] + sm[3];
        float fac = IK_QK_CLIP / (sqrtf(tot) + 1e-6f);
        if (fac > 1.0f) fac = 1.0f;
        p[tid] = v * fac * tau;
    } else if (blk < 32) {
        int h = blk - 24;
        float* p = K + ((size_t)t * IK_NKV + h) * IK_HD;
        float v = p[tid];
        float ss = warp_sum(v * v);
        __shared__ float sm[4];
        if ((tid & 31) == 0) sm[tid >> 5] = ss;
        __syncthreads();
        float tot = sm[0] + sm[1] + sm[2] + sm[3];
        float fac = IK_QK_CLIP / (sqrtf(tot) + 1e-6f);
        if (fac > 1.0f) fac = 1.0f;
        float outv = v * fac;
        p[tid] = outv;
        kv_k[((size_t)slot * IK_NKV + h) * IK_HD + tid] = d_f2h(outv);
    } else if (blk < 40) {
        int h = blk - 32;
        float* p = V + ((size_t)t * IK_NKV + h) * IK_HD;
        float v = p[tid];
        float ss = warp_sum(v * v);
        __shared__ float sm[4];
        if ((tid & 31) == 0) sm[tid >> 5] = ss;
        __syncthreads();
        float tot = sm[0] + sm[1] + sm[2] + sm[3];
        float inv = rsqrtf(tot / (float)IK_HD + IK_RMS_EPS);
        float outv = v * inv;
        p[tid] = outv;
        kv_v[((size_t)slot * IK_NKV + h) * IK_HD + tid] = d_f2h(outv);
    } else {
        int h = blk - 40;
        if (tid < IK_NREL) {
            float* p = R + ((size_t)t * IK_NH + h) * IK_NREL;
            float v = p[tid] * rel_clip[(size_t)h * IK_NREL + tid];
            float ss = v * v;
            #pragma unroll
            for (int o = 8; o > 0; o >>= 1) ss += __shfl_xor_sync(0x0000FFFFu, ss, o);   /* lanes 0-15 only */
            float fac = IK_REL_CLIP / (sqrtf(ss) + 1e-6f);
            if (fac > 1.0f) fac = 1.0f;
            p[tid] = v * fac * tau;
        }
    }
}

extern "C" __global__ void k_attn_table(
        const float* __restrict__ Q, const float* __restrict__ R,
        const float* __restrict__ sinks, const float* __restrict__ rel_bias,
        const uint16_t* __restrict__ kv_k, const uint16_t* __restrict__ kv_v,
        const float* __restrict__ gate,
        int kv_cap, int is_swa, int pos0, int T, float* __restrict__ out) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int pq = pos0 + t;
    int k_start = is_swa ? (pq + 1 > IK_SWA ? pq - IK_SWA + 1 : 0) : 0;
    int max_dist = pq - k_start;
    int max_tbl = max_dist < IK_RELEXT ? max_dist : (IK_RELEXT - 1);

    __shared__ float s_table[IK_RELEXT + 1];
    __shared__ float s_R[IK_NREL];
    __shared__ float s_mw[4], s_sw[4], s_acc[4][IK_HD];

    float sink = sinks[(size_t)t * IK_NH + h];
    if (tid < IK_NREL) s_R[tid] = R[((size_t)t * IK_NH + h) * IK_NREL + tid];
    __syncthreads();

    for (int d = tid; d <= max_tbl; d += 128) {
        const float* rb = rel_bias + (size_t)d * IK_NREL;
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < IK_NREL; k++) s = fmaf(rb[k], s_R[k], s);
        s_table[d] = s - sink;
    }
    if (tid == 0) s_table[IK_RELEXT] = -sink;
    __syncthreads();

    int hkv = h / (IK_NH / IK_NKV);
    const float* Qh = Q + ((size_t)t * IK_NH + h) * IK_HD + lane * 4;
    float q0 = Qh[0], q1 = Qh[1], q2 = Qh[2], q3 = Qh[3];
    float inv_s = rsqrtf((float)IK_HD);

    float m = (warp_id == 0) ? 0.0f : -1e30f;
    float sum = (warp_id == 0) ? 1.0f : 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (int pk = k_start + warp_id; pk <= pq; pk += 4) {
        int slot = is_swa ? (pk % kv_cap) : pk;
        size_t koff = ((size_t)slot * IK_NKV + hkv) * IK_HD + lane * 4;
        uint2 kpack = *(const uint2*)(kv_k + koff);
        uint2 vpack = *(const uint2*)(kv_v + koff);
        float k0 = d_h2f((uint16_t)(kpack.x & 0xFFFFu));
        float k1 = d_h2f((uint16_t)(kpack.x >> 16));
        float k2 = d_h2f((uint16_t)(kpack.y & 0xFFFFu));
        float k3 = d_h2f((uint16_t)(kpack.y >> 16));
        float dot = q0*k0 + q1*k1 + q2*k2 + q3*k3;
        dot = warp_sum(dot);
        int dist = pq - pk;
        if (dist > IK_RELEXT) dist = IK_RELEXT;
        float sc = dot * inv_s + s_table[dist];
        float w;
        if (sc > m) {
            float f = expf(m - sc);
            sum = sum * f + 1.0f;
            m = sc;
            w = 1.0f;
            a0 *= f; a1 *= f; a2 *= f; a3 *= f;
        } else {
            w = expf(sc - m);
            sum += w;
        }
        float v0 = d_h2f((uint16_t)(vpack.x & 0xFFFFu));
        float v1 = d_h2f((uint16_t)(vpack.x >> 16));
        float v2 = d_h2f((uint16_t)(vpack.y & 0xFFFFu));
        float v3 = d_h2f((uint16_t)(vpack.y >> 16));
        a0 = fmaf(v0, w, a0);
        a1 = fmaf(v1, w, a1);
        a2 = fmaf(v2, w, a2);
        a3 = fmaf(v3, w, a3);
    }

    if (lane == 0) { s_mw[warp_id] = m; s_sw[warp_id] = sum; }
    s_acc[warp_id][lane * 4 + 0] = a0;
    s_acc[warp_id][lane * 4 + 1] = a1;
    s_acc[warp_id][lane * 4 + 2] = a2;
    s_acc[warp_id][lane * 4 + 3] = a3;
    __syncthreads();

    float M_all = s_mw[0];
    for (int w = 1; w < 4; w++) if (s_mw[w] > M_all) M_all = s_mw[w];
    float f0 = (s_sw[0] > 0.f) ? expf(s_mw[0] - M_all) : 0.f;
    float f1 = (s_sw[1] > 0.f) ? expf(s_mw[1] - M_all) : 0.f;
    float f2 = (s_sw[2] > 0.f) ? expf(s_mw[2] - M_all) : 0.f;
    float f3 = (s_sw[3] > 0.f) ? expf(s_mw[3] - M_all) : 0.f;
    float S_all = s_sw[0]*f0 + s_sw[1]*f1 + s_sw[2]*f2 + s_sw[3]*f3;
    float inv = 1.0f / S_all;
    float val = (s_acc[0][tid]*f0 + s_acc[1][tid]*f1 + s_acc[2][tid]*f2 + s_acc[3][tid]*f3) * inv;
    size_t ooff = ((size_t)t * IK_NH + h) * IK_HD + tid;
    float gval = gate[ooff];
    out[ooff] = val * (1.0f / (1.0f + expf(-gval)));
}

extern "C" __global__ void k_residual_add_or_copy(float* __restrict__ cur, const float* __restrict__ a,
                                                  const float* __restrict__ b, int n, int do_add) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n) return;
    cur[i] = do_add ? (a[i] + b[i]) : a[i];
}

extern "C" __global__ void k_route_topk(const float* __restrict__ logits, const float* __restrict__ exp_bias,
                                        int T, int* __restrict__ sel, float* __restrict__ wsel) {
    int t = blockIdx.x;
    if (t >= T) return;
    int e = threadIdx.x;
    __shared__ float sp[IK_NE], ssc[IK_NE];
    float p = 1.0f / (1.0f + expf(-logits[(size_t)t * IK_NE + e]));
    sp[e] = p;
    ssc[e] = p + exp_bias[e];
    __syncthreads();
    if (e == 0) {
        int used[IK_NE];
        for (int i = 0; i < IK_NE; i++) used[i] = 0;
        float sumw = 0.0f;
        for (int k = 0; k < IK_NU; k++) {
            int bi = -1; float bv = -1e30f;
            for (int i = 0; i < IK_NE; i++) {
                if (!used[i] && ssc[i] > bv) { bv = ssc[i]; bi = i; }
            }
            used[bi] = 1;
            sel[t * IK_NU + k] = bi;
            wsel[t * IK_NU + k] = sp[bi];
            sumw += sp[bi];
        }
        float fac = IK_EXP_SCALE / (sumw + 1e-20f);
        for (int k = 0; k < IK_NU; k++) wsel[t * IK_NU + k] *= fac;
    }
}

extern "C" __global__ void k_moe_reduce(const float* __restrict__ slot_outs, const float* __restrict__ wsel,
                                        int T, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i >= IK_D || t >= T) return;
    float sum = 0.0f;
    #pragma unroll
    for (int s = 0; s < IK_NU; s++)
        sum = fmaf(wsel[t * IK_NU + s], slot_outs[((size_t)t * IK_NU + s) * IK_D + i], sum);
    out[(size_t)t * IK_D + i] = sum;
}

extern "C" __global__ void k_softcap(float* __restrict__ logits, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    logits[i] = IK_LOGIT_CAP * tanhf(logits[i] / IK_LOGIT_CAP);
}

extern "C" __global__ void k_top2_stage1(const float* __restrict__ lg, int n, int r_offset, IkTop2* __restrict__ blk_out) {
    int tid = threadIdx.x;
    int b0 = blockIdx.x * blockDim.x + tid;
    int step = gridDim.x * blockDim.x;
    int bt = -1; float m1 = -1e30f, m2 = -1e30f;
    for (int i = b0; i < n; i += step) {
        float x = lg[i];
        int id = r_offset + i;
        if (x > m1 || (x == m1 && id < bt)) { m2 = m1; m1 = x; bt = id; }
        else if (x > m2) m2 = x;
    }
    __shared__ IkTop2 sm[256];
    sm[tid].tok = bt; sm[tid].m1 = m1; sm[tid].m2 = m2;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            IkTop2 a = sm[tid], b = sm[tid + s];
            if (b.tok >= 0 && (a.tok < 0 || b.m1 > a.m1 || (b.m1 == a.m1 && b.tok < a.tok))) {
                float nm2 = fmaxf(a.m1, b.m2);
                sm[tid].tok = b.tok; sm[tid].m1 = b.m1; sm[tid].m2 = nm2;
            } else if (b.tok >= 0) {
                if (b.m1 > a.m2) sm[tid].m2 = b.m1;
            }
        }
        __syncthreads();
    }
    if (tid == 0) blk_out[blockIdx.x] = sm[0];
}

extern "C" __global__ void k_top2_stage2(const IkTop2* __restrict__ blk_in, int nblk, IkTop2* __restrict__ out) {
    __shared__ IkTop2 sm[256];
    int tid = threadIdx.x;
    if (tid < nblk) {
        sm[tid] = blk_in[tid];
    } else {
        sm[tid].tok = -1; sm[tid].m1 = -1e30f; sm[tid].m2 = -1e30f;
    }
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            IkTop2 a = sm[tid], b = sm[tid + s];
            if (b.tok >= 0 && (a.tok < 0 || b.m1 > a.m1 || (b.m1 == a.m1 && b.tok < a.tok))) {
                float nm2 = fmaxf(a.m1, b.m2);
                sm[tid].tok = b.tok; sm[tid].m1 = b.m1; sm[tid].m2 = nm2;
            } else if (b.tok >= 0) {
                if (b.m1 > a.m2) sm[tid].m2 = b.m1;
            }
        }
        __syncthreads();
    }
    if (tid == 0) out[0] = sm[0];
}

/* ---------------------------------------------------------------- IQ2_XXS_128 & IQ2_XS_128 MoE Kernels */
#if __CUDA_ARCH__ >= 610
static __device__ __forceinline__ int iq2_dp4a_8words(
        uint32_t ax0, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        uint64_t g0, uint64_t g1, uint64_t g2, uint64_t g3,
        const int32_t* __restrict__ xp) {
    (void)ax0;
    int32_t w0 = iq2_apply_signs4((uint32_t)g0,         s0 & 15u);
    int32_t w1 = iq2_apply_signs4((uint32_t)(g0 >> 32), s0 >> 4);
    int32_t w2 = iq2_apply_signs4((uint32_t)g1,         s1 & 15u);
    int32_t w3 = iq2_apply_signs4((uint32_t)(g1 >> 32), s1 >> 4);
    int32_t w4 = iq2_apply_signs4((uint32_t)g2,         s2 & 15u);
    int32_t w5 = iq2_apply_signs4((uint32_t)(g2 >> 32), s2 >> 4);
    int32_t w6 = iq2_apply_signs4((uint32_t)g3,         s3 & 15u);
    int32_t w7 = iq2_apply_signs4((uint32_t)(g3 >> 32), s3 >> 4);
    int4 xv0 = *(const int4*)xp;
    int4 xv1 = *(const int4*)(xp + 4);
    int s = __dp4a(w0, xv0.x, 0);
    s = __dp4a(w1, xv0.y, s);
    s = __dp4a(w2, xv0.z, s);
    s = __dp4a(w3, xv0.w, s);
    s = __dp4a(w4, xv1.x, s);
    s = __dp4a(w5, xv1.y, s);
    s = __dp4a(w6, xv1.z, s);
    s = __dp4a(w7, xv1.w, s);
    return s;
}

static __device__ __forceinline__ float iq2xxs_step_1_dp4a(
        size_t boff, int r, int b, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid,
        const int32_t* __restrict__ xq0, const float* __restrict__ xd0, float acc) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 32))[r]) * 0.25f;
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    #pragma unroll
    for (int ib = 0; ib < 4; ib++) {
        uint32_t ax0 = qs32[(2 * ib + 0) * 16 + r];
        uint32_t ax1 = qs32[(2 * ib + 1) * 16 + r];
        float lsc = d * (0.5f + (float)(ax1 >> 28));
        uint32_t s0 = s_signs[ax1 & 127u],        s1 = s_signs[(ax1 >> 7)  & 127u];
        uint32_t s2 = s_signs[(ax1 >> 14) & 127u], s3 = s_signs[(ax1 >> 21) & 127u];
        uint64_t g0 = s_grid[ax0 & 255u],          g1 = s_grid[(ax0 >> 8)  & 255u];
        uint64_t g2 = s_grid[(ax0 >> 16) & 255u],  g3 = s_grid[(ax0 >> 24) & 255u];
        int isum = iq2_dp4a_8words(ax0, s0, s1, s2, s3, g0, g1, g2, g3, xq0 + (b * 4 + ib) * 8);
        acc = fmaf((float)isum, lsc * xd0[b * 4 + ib], acc);
    }
    return acc;
}

static __device__ __forceinline__ void iq2xxs_step_n_dp4a(
        size_t boff, int r, int b, int nt, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid,
        const int32_t* const xqs[4], const float* const xds[4], float acc[4]) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 32))[r]) * 0.25f;
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    #pragma unroll
    for (int ib = 0; ib < 4; ib++) {
        uint32_t ax0 = qs32[(2 * ib + 0) * 16 + r];
        uint32_t ax1 = qs32[(2 * ib + 1) * 16 + r];
        float lsc = d * (0.5f + (float)(ax1 >> 28));
        uint32_t s0 = s_signs[ax1 & 127u],        s1 = s_signs[(ax1 >> 7)  & 127u];
        uint32_t s2 = s_signs[(ax1 >> 14) & 127u], s3 = s_signs[(ax1 >> 21) & 127u];
        uint64_t g0 = s_grid[ax0 & 255u],          g1 = s_grid[(ax0 >> 8)  & 255u];
        uint64_t g2 = s_grid[(ax0 >> 16) & 255u],  g3 = s_grid[(ax0 >> 24) & 255u];
        int32_t w0 = iq2_apply_signs4((uint32_t)g0,         s0 & 15u);
        int32_t w1 = iq2_apply_signs4((uint32_t)(g0 >> 32), s0 >> 4);
        int32_t w2 = iq2_apply_signs4((uint32_t)g1,         s1 & 15u);
        int32_t w3 = iq2_apply_signs4((uint32_t)(g1 >> 32), s1 >> 4);
        int32_t w4 = iq2_apply_signs4((uint32_t)g2,         s2 & 15u);
        int32_t w5 = iq2_apply_signs4((uint32_t)(g2 >> 32), s2 >> 4);
        int32_t w6 = iq2_apply_signs4((uint32_t)g3,         s3 & 15u);
        int32_t w7 = iq2_apply_signs4((uint32_t)(g3 >> 32), s3 >> 4);
        int qoff = (b * 4 + ib) * 8, doff = b * 4 + ib;
        for (int i = 0; i < nt; i++) {
            const int32_t* xp = xqs[i] + qoff;
            int4 xv0 = *(const int4*)xp, xv1 = *(const int4*)(xp + 4);
            int isum = __dp4a(w0, xv0.x, 0);
            isum = __dp4a(w1, xv0.y, isum); isum = __dp4a(w2, xv0.z, isum); isum = __dp4a(w3, xv0.w, isum);
            isum = __dp4a(w4, xv1.x, isum); isum = __dp4a(w5, xv1.y, isum);
            isum = __dp4a(w6, xv1.z, isum); isum = __dp4a(w7, xv1.w, isum);
            acc[i] = fmaf((float)isum, lsc * xds[i][doff], acc[i]);
        }
    }
}

static __device__ __forceinline__ float iq2xs_step_1_dp4a(
        size_t boff, int r, int b, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid,
        const int32_t* __restrict__ xq0, const float* __restrict__ xd0, float acc) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 96))[r]) * 0.25f;
    uint32_t ssub = ((const uint32_t*)(sc + boff * 96 + 32))[r];
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    #pragma unroll
    for (int ib = 0; ib < 4; ib++) {
        uint8_t sc8 = (uint8_t)((ssub >> (8 * ib)) & 0xFFu);
        float lsc0 = d * (0.5f + (float)(sc8 & 15u));
        float lsc1 = d * (0.5f + (float)(sc8 >> 4));
        uint32_t pk01 = qs32[(2 * ib + 0) * 16 + r];
        uint32_t pk23 = qs32[(2 * ib + 1) * 16 + r];
        uint32_t q0 = pk01 & 0xFFFFu, q1 = pk01 >> 16, q2 = pk23 & 0xFFFFu, q3 = pk23 >> 16;
        uint32_t s0 = s_signs[q0 >> 9], s1 = s_signs[q1 >> 9], s2 = s_signs[q2 >> 9], s3 = s_signs[q3 >> 9];
        uint64_t g0 = s_grid[q0 & 511u], g1 = s_grid[q1 & 511u], g2 = s_grid[q2 & 511u], g3 = s_grid[q3 & 511u];
        const int32_t* xp = xq0 + (b * 4 + ib) * 8;
        int4 xv0 = *(const int4*)xp, xv1 = *(const int4*)(xp + 4);
        int is0 = __dp4a(iq2_apply_signs4((uint32_t)g0, s0 & 15u), xv0.x, 0);
        is0 = __dp4a(iq2_apply_signs4((uint32_t)(g0 >> 32), s0 >> 4), xv0.y, is0);
        is0 = __dp4a(iq2_apply_signs4((uint32_t)g1, s1 & 15u), xv0.z, is0);
        is0 = __dp4a(iq2_apply_signs4((uint32_t)(g1 >> 32), s1 >> 4), xv0.w, is0);
        int is1 = __dp4a(iq2_apply_signs4((uint32_t)g2, s2 & 15u), xv1.x, 0);
        is1 = __dp4a(iq2_apply_signs4((uint32_t)(g2 >> 32), s2 >> 4), xv1.y, is1);
        is1 = __dp4a(iq2_apply_signs4((uint32_t)g3, s3 & 15u), xv1.z, is1);
        is1 = __dp4a(iq2_apply_signs4((uint32_t)(g3 >> 32), s3 >> 4), xv1.w, is1);
        acc = fmaf((lsc0 * (float)is0 + lsc1 * (float)is1), xd0[b * 4 + ib], acc);
    }
    return acc;
}

static __device__ __forceinline__ void iq2xs_step_n_dp4a(
        size_t boff, int r, int b, int nt, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid,
        const int32_t* const xqs[4], const float* const xds[4], float acc[4]) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 96))[r]) * 0.25f;
    uint32_t ssub = ((const uint32_t*)(sc + boff * 96 + 32))[r];
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    #pragma unroll
    for (int ib = 0; ib < 4; ib++) {
        uint8_t sc8 = (uint8_t)((ssub >> (8 * ib)) & 0xFFu);
        float lsc0 = d * (0.5f + (float)(sc8 & 15u));
        float lsc1 = d * (0.5f + (float)(sc8 >> 4));
        uint32_t pk01 = qs32[(2 * ib + 0) * 16 + r];
        uint32_t pk23 = qs32[(2 * ib + 1) * 16 + r];
        uint32_t q0 = pk01 & 0xFFFFu, q1 = pk01 >> 16, q2 = pk23 & 0xFFFFu, q3 = pk23 >> 16;
        uint32_t s0 = s_signs[q0 >> 9], s1 = s_signs[q1 >> 9], s2 = s_signs[q2 >> 9], s3 = s_signs[q3 >> 9];
        uint64_t g0 = s_grid[q0 & 511u], g1 = s_grid[q1 & 511u], g2 = s_grid[q2 & 511u], g3 = s_grid[q3 & 511u];
        int32_t w0 = iq2_apply_signs4((uint32_t)g0,         s0 & 15u);
        int32_t w1 = iq2_apply_signs4((uint32_t)(g0 >> 32), s0 >> 4);
        int32_t w2 = iq2_apply_signs4((uint32_t)g1,         s1 & 15u);
        int32_t w3 = iq2_apply_signs4((uint32_t)(g1 >> 32), s1 >> 4);
        int32_t w4 = iq2_apply_signs4((uint32_t)g2,         s2 & 15u);
        int32_t w5 = iq2_apply_signs4((uint32_t)(g2 >> 32), s2 >> 4);
        int32_t w6 = iq2_apply_signs4((uint32_t)g3,         s3 & 15u);
        int32_t w7 = iq2_apply_signs4((uint32_t)(g3 >> 32), s3 >> 4);
        int qoff = (b * 4 + ib) * 8, doff = b * 4 + ib;
        for (int i = 0; i < nt; i++) {
            const int32_t* xp = xqs[i] + qoff;
            int4 xv0 = *(const int4*)xp, xv1 = *(const int4*)(xp + 4);
            int is0 = __dp4a(w0, xv0.x, 0);
            is0 = __dp4a(w1, xv0.y, is0); is0 = __dp4a(w2, xv0.z, is0); is0 = __dp4a(w3, xv0.w, is0);
            int is1 = __dp4a(w4, xv1.x, 0);
            is1 = __dp4a(w5, xv1.y, is1); is1 = __dp4a(w6, xv1.z, is1); is1 = __dp4a(w7, xv1.w, is1);
            acc[i] = fmaf((lsc0 * (float)is0 + lsc1 * (float)is1), xds[i][doff], acc[i]);
        }
    }
}
#else
static __device__ __forceinline__ void iq2_acc4_n_f(
        uint32_t g4, uint32_t m4, int k4, int nt,
        const float* __restrict__ xp0, const float* __restrict__ xp1,
        const float* __restrict__ xp2, const float* __restrict__ xp3,
        float& d0, float& d1, float& d2, float& d3) {
    float f0 = (float)(g4 & 255u);
    float f1 = (float)((g4 >> 8) & 255u);
    float f2 = (float)((g4 >> 16) & 255u);
    float f3 = (float)(g4 >> 24);
    float w0 = (m4 & 1u) ? -f0 : f0;
    float w1 = (m4 & 2u) ? -f1 : f1;
    float w2 = (m4 & 4u) ? -f2 : f2;
    float w3 = (m4 & 8u) ? -f3 : f3;
    float4 xv0 = *(const float4*)(xp0 + k4);
    d0 = fmaf(w0, xv0.x, fmaf(w1, xv0.y, fmaf(w2, xv0.z, fmaf(w3, xv0.w, d0))));
    if (nt > 1) {
        float4 xv1 = *(const float4*)(xp1 + k4);
        d1 = fmaf(w0, xv1.x, fmaf(w1, xv1.y, fmaf(w2, xv1.z, fmaf(w3, xv1.w, d1))));
    }
    if (nt > 2) {
        float4 xv2 = *(const float4*)(xp2 + k4);
        d2 = fmaf(w0, xv2.x, fmaf(w1, xv2.y, fmaf(w2, xv2.z, fmaf(w3, xv2.w, d2))));
    }
    if (nt > 3) {
        float4 xv3 = *(const float4*)(xp3 + k4);
        d3 = fmaf(w0, xv3.x, fmaf(w1, xv3.y, fmaf(w2, xv3.z, fmaf(w3, xv3.w, d3))));
    }
}

static __device__ __forceinline__ float iq2xxs_step_1_f(
        size_t boff, int r, int b, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid, const float* __restrict__ xf, float acc) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 32))[r]) * 0.25f;
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    for (int ib = 0; ib < 4; ib++) {
        uint32_t ax0 = qs32[(2 * ib + 0) * 16 + r];
        uint32_t ax1 = qs32[(2 * ib + 1) * 16 + r];
        float lsc = d * (0.5f + (float)(ax1 >> 28));
        float s = 0.f;
        const float* xp = xf + b * 128 + ib * 32;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint64_t gq = s_grid[(ax0 >> (8 * q)) & 255u];
            uint32_t sq = s_signs[(ax1 >> (7 * q)) & 127u];
            s += iq2_dot4_f32((uint32_t)gq,         sq & 15u, *(const float4*)(xp + q * 8 + 0))
               + iq2_dot4_f32((uint32_t)(gq >> 32), sq >> 4,  *(const float4*)(xp + q * 8 + 4));
        }
        acc = fmaf(s, lsc, acc);
    }
    return acc;
}

static __device__ __forceinline__ void iq2xxs_step_n_f(
        size_t boff, int r, int b, int nt, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid,
        const float* __restrict__ xp0, const float* __restrict__ xp1,
        const float* __restrict__ xp2, const float* __restrict__ xp3,
        float& acc0, float& acc1, float& acc2, float& acc3) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 32))[r]) * 0.25f;
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    for (int ib = 0; ib < 4; ib++) {
        uint32_t ax0 = qs32[(2 * ib + 0) * 16 + r];
        uint32_t ax1 = qs32[(2 * ib + 1) * 16 + r];
        float lsc = d * (0.5f + (float)(ax1 >> 28));
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        int kbase = b * 128 + ib * 32;
        for (int q = 0; q < 4; q++) {
            uint64_t gq = s_grid[(ax0 >> (8 * q)) & 255u];
            uint32_t sq = s_signs[(ax1 >> (7 * q)) & 127u];
            iq2_acc4_n_f((uint32_t)gq,         sq & 15u, kbase + q * 8 + 0, nt, xp0, xp1, xp2, xp3, d0, d1, d2, d3);
            iq2_acc4_n_f((uint32_t)(gq >> 32), sq >> 4,  kbase + q * 8 + 4, nt, xp0, xp1, xp2, xp3, d0, d1, d2, d3);
        }
        acc0 = fmaf(d0, lsc, acc0);
        if (nt > 1) acc1 = fmaf(d1, lsc, acc1);
        if (nt > 2) acc2 = fmaf(d2, lsc, acc2);
        if (nt > 3) acc3 = fmaf(d3, lsc, acc3);
    }
}

static __device__ __forceinline__ float iq2xs_step_1_f(
        size_t boff, int r, int b, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid, const float* __restrict__ xf, float acc) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 96))[r]) * 0.25f;
    uint32_t ssub = ((const uint32_t*)(sc + boff * 96 + 32))[r];
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    for (int ib = 0; ib < 4; ib++) {
        uint8_t sc8 = (uint8_t)((ssub >> (8 * ib)) & 0xFFu);
        for (int half = 0; half < 2; half++) {
            float lsc = d * (0.5f + (float)(half ? (sc8 >> 4) : (sc8 & 15u)));
            uint32_t pk = qs32[(2 * ib + half) * 16 + r];
            uint32_t qa = pk & 0xFFFFu, qb = pk >> 16;
            uint32_t sa = s_signs[qa >> 9], sb = s_signs[qb >> 9];
            uint64_t ga = s_grid[qa & 511u], gb = s_grid[qb & 511u];
            const float* xp = xf + b * 128 + ib * 32 + half * 16;
            float sum = iq2_dot4_f32((uint32_t)ga,         sa & 15u, *(const float4*)(xp + 0))
                      + iq2_dot4_f32((uint32_t)(ga >> 32), sa >> 4,  *(const float4*)(xp + 4))
                      + iq2_dot4_f32((uint32_t)gb,         sb & 15u, *(const float4*)(xp + 8))
                      + iq2_dot4_f32((uint32_t)(gb >> 32), sb >> 4,  *(const float4*)(xp + 12));
            acc = fmaf(sum, lsc, acc);
        }
    }
    return acc;
}

static __device__ __forceinline__ void iq2xs_step_n_f(
        size_t boff, int r, int b, int nt, const uint8_t* __restrict__ sc, const uint8_t* __restrict__ qs,
        const uint8_t* s_signs, const uint64_t* s_grid,
        const float* __restrict__ xp0, const float* __restrict__ xp1,
        const float* __restrict__ xp2, const float* __restrict__ xp3,
        float& acc0, float& acc1, float& acc2, float& acc3) {
    float d = d_h2f(((const uint16_t*)(sc + boff * 96))[r]) * 0.25f;
    uint32_t ssub = ((const uint32_t*)(sc + boff * 96 + 32))[r];
    const uint32_t* qs32 = (const uint32_t*)(qs + boff * 512);
    for (int ib = 0; ib < 4; ib++) {
        uint8_t sc8 = (uint8_t)((ssub >> (8 * ib)) & 0xFFu);
        for (int half = 0; half < 2; half++) {
            float lsc = d * (0.5f + (float)(half ? (sc8 >> 4) : (sc8 & 15u)));
            uint32_t pk = qs32[(2 * ib + half) * 16 + r];
            uint32_t q0 = pk & 0xFFFFu, q1 = pk >> 16;
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            int kbase = b * 128 + ib * 32 + half * 16;
            uint64_t g0 = s_grid[q0 & 511u];
            uint32_t s0 = s_signs[q0 >> 9];
            iq2_acc4_n_f((uint32_t)g0,         s0 & 15u, kbase + 0, nt, xp0, xp1, xp2, xp3, d0, d1, d2, d3);
            iq2_acc4_n_f((uint32_t)(g0 >> 32), s0 >> 4,  kbase + 4, nt, xp0, xp1, xp2, xp3, d0, d1, d2, d3);
            uint64_t g1 = s_grid[q1 & 511u];
            uint32_t s1 = s_signs[q1 >> 9];
            iq2_acc4_n_f((uint32_t)g1,         s1 & 15u, kbase + 8,  nt, xp0, xp1, xp2, xp3, d0, d1, d2, d3);
            iq2_acc4_n_f((uint32_t)(g1 >> 32), s1 >> 4,  kbase + 12, nt, xp0, xp1, xp2, xp3, d0, d1, d2, d3);
            acc0 = fmaf(d0, lsc, acc0);
            if (nt > 1) acc1 = fmaf(d1, lsc, acc1);
            if (nt > 2) acc2 = fmaf(d2, lsc, acc2);
            if (nt > 3) acc3 = fmaf(d3, lsc, acc3);
        }
    }
}
#endif

extern "C" __global__ void k_iq2xxs128_moe_gu(
        const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off, const int* __restrict__ d_tok_idx,
        const uint8_t* const* __restrict__ exp_gu_sc, const uint8_t* const* __restrict__ exp_gu_qs,
        const float* __restrict__ xf, const int32_t* __restrict__ xq4, const float* __restrict__ xd,
        float* __restrict__ d_moe_h, int uniq_n) {
    __shared__ uint8_t s_signs[128];
    __shared__ uint64_t s_grid[256];
    int tid = threadIdx.y * 32 + threadIdx.x;
    s_signs[tid] = ik_ksigns_iq2xs[tid];
    s_grid[tid] = ik_iq2xxs_grid[tid];
    s_grid[tid + 128] = ik_iq2xxs_grid[tid + 128];
    __syncthreads();

    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (g_row * 16 >= IK_FF) return;
    int t = threadIdx.x, r = t & 15, h = t >> 4, rr = g_row * 16 + r;
    int e = d_uniq_e[u];
    const uint8_t* sc = exp_gu_sc[e];
    const uint8_t* qs = exp_gu_qs[e];
    if (!sc || !qs) return;
    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    const int nb128 = IK_D / QK_IQ2;
    const size_t up_g_off = (size_t)(IK_FF / 16) * nb128;

#if __CUDA_ARCH__ >= 610
    (void)xf;
    int nb32 = IK_D >> 5;
    if (p1 - p0 == 1) {
        int tk0 = d_tok_idx[p0];
        const int32_t* xq0 = xq4 + (size_t)tk0 * nb32 * 8;
        const float* xd0 = xd + (size_t)tk0 * nb32;
        float ag = 0.f, au = 0.f;
        for (int b = h; b < nb128; b += 2) {
            size_t og = (size_t)g_row * nb128 + b;
            ag = iq2xxs_step_1_dp4a(og, r, b, sc, qs, s_signs, s_grid, xq0, xd0, ag);
            au = iq2xxs_step_1_dp4a(up_g_off + og, r, b, sc, qs, s_signs, s_grid, xq0, xd0, au);
        }
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 16);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 16);
        if (h == 0 && rr < IK_FF) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(ag, au);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const int32_t* xqs[4] = {NULL, NULL, NULL, NULL};
            const float* xds[4] = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) {
                int tk = d_tok_idx[base_p + i];
                xqs[i] = xq4 + (size_t)tk * nb32 * 8;
                xds[i] = xd + (size_t)tk * nb32;
            }
            float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = h; b < nb128; b += 2) {
                size_t og = (size_t)g_row * nb128 + b;
                iq2xxs_step_n_dp4a(og, r, b, nt, sc, qs, s_signs, s_grid, xqs, xds, ag);
                iq2xxs_step_n_dp4a(up_g_off + og, r, b, nt, sc, qs, s_signs, s_grid, xqs, xds, au);
            }
            for (int i = 0; i < nt; i++) {
                ag[i] += __shfl_xor_sync(0xFFFFFFFFu, ag[i], 16);
                au[i] += __shfl_xor_sync(0xFFFFFFFFu, au[i], 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + i) * IK_FF + rr] = d_situ(ag[i], au[i]);
            }
        }
    }
#else
    (void)xq4; (void)xd;
    if (p1 - p0 == 1) {
        const float* xp0 = xf + (size_t)d_tok_idx[p0] * IK_D;
        float ag = 0.f, au = 0.f;
        for (int b = h; b < nb128; b += 2) {
            size_t og = (size_t)g_row * nb128 + b;
            ag = iq2xxs_step_1_f(og, r, b, sc, qs, s_signs, s_grid, xp0, ag);
            au = iq2xxs_step_1_f(up_g_off + og, r, b, sc, qs, s_signs, s_grid, xp0, au);
        }
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 16);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 16);
        if (h == 0 && rr < IK_FF) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(ag, au);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const float* xp0 = xf + (size_t)d_tok_idx[base_p + 0] * IK_D;
            const float* xp1 = (nt > 1) ? (xf + (size_t)d_tok_idx[base_p + 1] * IK_D) : xp0;
            const float* xp2 = (nt > 2) ? (xf + (size_t)d_tok_idx[base_p + 2] * IK_D) : xp0;
            const float* xp3 = (nt > 3) ? (xf + (size_t)d_tok_idx[base_p + 3] * IK_D) : xp0;
            float ag0 = 0.f, ag1 = 0.f, ag2 = 0.f, ag3 = 0.f;
            float au0 = 0.f, au1 = 0.f, au2 = 0.f, au3 = 0.f;
            for (int b = h; b < nb128; b += 2) {
                size_t og = (size_t)g_row * nb128 + b;
                iq2xxs_step_n_f(og, r, b, nt, sc, qs, s_signs, s_grid, xp0, xp1, xp2, xp3, ag0, ag1, ag2, ag3);
                iq2xxs_step_n_f(up_g_off + og, r, b, nt, sc, qs, s_signs, s_grid, xp0, xp1, xp2, xp3, au0, au1, au2, au3);
            }
            ag0 += __shfl_xor_sync(0xFFFFFFFFu, ag0, 16); au0 += __shfl_xor_sync(0xFFFFFFFFu, au0, 16);
            if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 0) * IK_FF + rr] = d_situ(ag0, au0);
            if (nt > 1) {
                ag1 += __shfl_xor_sync(0xFFFFFFFFu, ag1, 16); au1 += __shfl_xor_sync(0xFFFFFFFFu, au1, 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 1) * IK_FF + rr] = d_situ(ag1, au1);
            }
            if (nt > 2) {
                ag2 += __shfl_xor_sync(0xFFFFFFFFu, ag2, 16); au2 += __shfl_xor_sync(0xFFFFFFFFu, au2, 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 2) * IK_FF + rr] = d_situ(ag2, au2);
            }
            if (nt > 3) {
                ag3 += __shfl_xor_sync(0xFFFFFFFFu, ag3, 16); au3 += __shfl_xor_sync(0xFFFFFFFFu, au3, 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 3) * IK_FF + rr] = d_situ(ag3, au3);
            }
        }
    }
#endif
}

extern "C" __global__ void k_iq2xxs128_moe_dn(
        const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off,
        const int* __restrict__ d_tok_idx, const int* __restrict__ d_tok_slot,
        const uint8_t* const* __restrict__ exp_dn_sc, const uint8_t* const* __restrict__ exp_dn_qs,
        const float* __restrict__ d_moe_h, const int32_t* __restrict__ hq4, const float* __restrict__ hd,
        float* __restrict__ slot_outs, int uniq_n) {
    __shared__ uint8_t s_signs[128];
    __shared__ uint64_t s_grid[256];
    int tid = threadIdx.y * 32 + threadIdx.x;
    s_signs[tid] = ik_ksigns_iq2xs[tid];
    s_grid[tid] = ik_iq2xxs_grid[tid];
    s_grid[tid + 128] = ik_iq2xxs_grid[tid + 128];
    __syncthreads();

    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (g_row * 16 >= IK_D) return;
    int t = threadIdx.x, r = t & 15, h = t >> 4, rr = g_row * 16 + r;
    int e = d_uniq_e[u];
    const uint8_t* sc = exp_dn_sc[e];
    const uint8_t* qs = exp_dn_qs[e];
    if (!sc || !qs) return;
    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    const int nb128 = IK_FF / QK_IQ2;

#if __CUDA_ARCH__ >= 610
    (void)d_moe_h;
    int nb32 = IK_FF >> 5;
    if (p1 - p0 == 1) {
        const int32_t* hq0 = hq4 + (size_t)p0 * nb32 * 8;
        const float* hd0 = hd + (size_t)p0 * nb32;
        float a0 = 0.f;
        for (int b = h; b < nb128; b += 2) {
            a0 = iq2xxs_step_1_dp4a((size_t)g_row * nb128 + b, r, b, sc, qs, s_signs, s_grid, hq0, hd0, a0);
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
        if (h == 0 && rr < IK_D) {
            int tk = d_tok_idx[p0], sl = d_tok_slot[p0];
            slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a0;
        }
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const int32_t* hqs[4] = {NULL, NULL, NULL, NULL};
            const float* hds[4] = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) {
                hqs[i] = hq4 + (size_t)(base_p + i) * nb32 * 8;
                hds[i] = hd + (size_t)(base_p + i) * nb32;
            }
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = h; b < nb128; b += 2) {
                iq2xxs_step_n_dp4a((size_t)g_row * nb128 + b, r, b, nt, sc, qs, s_signs, s_grid, hqs, hds, a);
            }
            for (int i = 0; i < nt; i++) {
                a[i] += __shfl_xor_sync(0xFFFFFFFFu, a[i], 16);
                if (h == 0 && rr < IK_D) {
                    int tk = d_tok_idx[base_p + i], sl = d_tok_slot[base_p + i];
                    slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a[i];
                }
            }
        }
    }
#else
    (void)hq4; (void)hd;
    if (p1 - p0 == 1) {
        const float* hp0 = d_moe_h + (size_t)p0 * IK_FF;
        float a0 = 0.f;
        for (int b = h; b < nb128; b += 2) {
            a0 = iq2xxs_step_1_f((size_t)g_row * nb128 + b, r, b, sc, qs, s_signs, s_grid, hp0, a0);
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
        if (h == 0 && rr < IK_D) {
            int tk = d_tok_idx[p0], sl = d_tok_slot[p0];
            slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a0;
        }
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const float* hp0 = d_moe_h + (size_t)(base_p + 0) * IK_FF;
            const float* hp1 = (nt > 1) ? (d_moe_h + (size_t)(base_p + 1) * IK_FF) : hp0;
            const float* hp2 = (nt > 2) ? (d_moe_h + (size_t)(base_p + 2) * IK_FF) : hp0;
            const float* hp3 = (nt > 3) ? (d_moe_h + (size_t)(base_p + 3) * IK_FF) : hp0;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (int b = h; b < nb128; b += 2) {
                iq2xxs_step_n_f((size_t)g_row * nb128 + b, r, b, nt, sc, qs, s_signs, s_grid, hp0, hp1, hp2, hp3, a0, a1, a2, a3);
            }
            a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
            if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 0] * IK_NU + d_tok_slot[base_p + 0]) * IK_D + rr] = a0;
            if (nt > 1) {
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 16);
                if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 1] * IK_NU + d_tok_slot[base_p + 1]) * IK_D + rr] = a1;
            }
            if (nt > 2) {
                a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 16);
                if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 2] * IK_NU + d_tok_slot[base_p + 2]) * IK_D + rr] = a2;
            }
            if (nt > 3) {
                a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 16);
                if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 3] * IK_NU + d_tok_slot[base_p + 3]) * IK_D + rr] = a3;
            }
        }
    }
#endif
}

extern "C" __global__ void k_iq2xs128_moe_gu(
        const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off, const int* __restrict__ d_tok_idx,
        const uint8_t* const* __restrict__ exp_gu_sc, const uint8_t* const* __restrict__ exp_gu_qs,
        const float* __restrict__ xf, const int32_t* __restrict__ xq4, const float* __restrict__ xd,
        float* __restrict__ d_moe_h, int uniq_n) {
    __shared__ uint8_t s_signs[128];
    __shared__ uint64_t s_grid[512];
    int tid = threadIdx.y * 32 + threadIdx.x;
    s_signs[tid] = ik_ksigns_iq2xs[tid];
    for (int i = tid; i < 512; i += 128) s_grid[i] = ik_iq2xs_grid[i];
    __syncthreads();

    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (g_row * 16 >= IK_FF) return;
    int t = threadIdx.x, r = t & 15, h = t >> 4, rr = g_row * 16 + r;
    int e = d_uniq_e[u];
    const uint8_t* sc = exp_gu_sc[e];
    const uint8_t* qs = exp_gu_qs[e];
    if (!sc || !qs) return;
    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    const int nb128 = IK_D / QK_IQ2;
    const size_t up_g_off = (size_t)(IK_FF / 16) * nb128;

#if __CUDA_ARCH__ >= 610
    (void)xf;
    int nb32 = IK_D >> 5;
    if (p1 - p0 == 1) {
        int tk0 = d_tok_idx[p0];
        const int32_t* xq0 = xq4 + (size_t)tk0 * nb32 * 8;
        const float* xd0 = xd + (size_t)tk0 * nb32;
        float ag = 0.f, au = 0.f;
        for (int b = h; b < nb128; b += 2) {
            size_t og = (size_t)g_row * nb128 + b;
            ag = iq2xs_step_1_dp4a(og, r, b, sc, qs, s_signs, s_grid, xq0, xd0, ag);
            au = iq2xs_step_1_dp4a(up_g_off + og, r, b, sc, qs, s_signs, s_grid, xq0, xd0, au);
        }
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 16);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 16);
        if (h == 0 && rr < IK_FF) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(ag, au);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const int32_t* xqs[4] = {NULL, NULL, NULL, NULL};
            const float* xds[4] = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) {
                int tk = d_tok_idx[base_p + i];
                xqs[i] = xq4 + (size_t)tk * nb32 * 8;
                xds[i] = xd + (size_t)tk * nb32;
            }
            float ag[4] = {0.f, 0.f, 0.f, 0.f}, au[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = h; b < nb128; b += 2) {
                size_t og = (size_t)g_row * nb128 + b;
                iq2xs_step_n_dp4a(og, r, b, nt, sc, qs, s_signs, s_grid, xqs, xds, ag);
                iq2xs_step_n_dp4a(up_g_off + og, r, b, nt, sc, qs, s_signs, s_grid, xqs, xds, au);
            }
            for (int i = 0; i < nt; i++) {
                ag[i] += __shfl_xor_sync(0xFFFFFFFFu, ag[i], 16);
                au[i] += __shfl_xor_sync(0xFFFFFFFFu, au[i], 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + i) * IK_FF + rr] = d_situ(ag[i], au[i]);
            }
        }
    }
#else
    (void)xq4; (void)xd;
    if (p1 - p0 == 1) {
        const float* xp0 = xf + (size_t)d_tok_idx[p0] * IK_D;
        float ag = 0.f, au = 0.f;
        for (int b = h; b < nb128; b += 2) {
            size_t og = (size_t)g_row * nb128 + b;
            ag = iq2xs_step_1_f(og, r, b, sc, qs, s_signs, s_grid, xp0, ag);
            au = iq2xs_step_1_f(up_g_off + og, r, b, sc, qs, s_signs, s_grid, xp0, au);
        }
        ag += __shfl_xor_sync(0xFFFFFFFFu, ag, 16);
        au += __shfl_xor_sync(0xFFFFFFFFu, au, 16);
        if (h == 0 && rr < IK_FF) d_moe_h[(size_t)p0 * IK_FF + rr] = d_situ(ag, au);
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const float* xp0 = xf + (size_t)d_tok_idx[base_p + 0] * IK_D;
            const float* xp1 = (nt > 1) ? (xf + (size_t)d_tok_idx[base_p + 1] * IK_D) : xp0;
            const float* xp2 = (nt > 2) ? (xf + (size_t)d_tok_idx[base_p + 2] * IK_D) : xp0;
            const float* xp3 = (nt > 3) ? (xf + (size_t)d_tok_idx[base_p + 3] * IK_D) : xp0;
            float ag0 = 0.f, ag1 = 0.f, ag2 = 0.f, ag3 = 0.f;
            float au0 = 0.f, au1 = 0.f, au2 = 0.f, au3 = 0.f;
            for (int b = h; b < nb128; b += 2) {
                size_t og = (size_t)g_row * nb128 + b;
                iq2xs_step_n_f(og, r, b, nt, sc, qs, s_signs, s_grid, xp0, xp1, xp2, xp3, ag0, ag1, ag2, ag3);
                iq2xs_step_n_f(up_g_off + og, r, b, nt, sc, qs, s_signs, s_grid, xp0, xp1, xp2, xp3, au0, au1, au2, au3);
            }
            ag0 += __shfl_xor_sync(0xFFFFFFFFu, ag0, 16); au0 += __shfl_xor_sync(0xFFFFFFFFu, au0, 16);
            if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 0) * IK_FF + rr] = d_situ(ag0, au0);
            if (nt > 1) {
                ag1 += __shfl_xor_sync(0xFFFFFFFFu, ag1, 16); au1 += __shfl_xor_sync(0xFFFFFFFFu, au1, 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 1) * IK_FF + rr] = d_situ(ag1, au1);
            }
            if (nt > 2) {
                ag2 += __shfl_xor_sync(0xFFFFFFFFu, ag2, 16); au2 += __shfl_xor_sync(0xFFFFFFFFu, au2, 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 2) * IK_FF + rr] = d_situ(ag2, au2);
            }
            if (nt > 3) {
                ag3 += __shfl_xor_sync(0xFFFFFFFFu, ag3, 16); au3 += __shfl_xor_sync(0xFFFFFFFFu, au3, 16);
                if (h == 0 && rr < IK_FF) d_moe_h[(size_t)(base_p + 3) * IK_FF + rr] = d_situ(ag3, au3);
            }
        }
    }
#endif
}

extern "C" __global__ void k_iq2xs128_moe_dn(
        const int* __restrict__ d_uniq_e, const int* __restrict__ d_u_off,
        const int* __restrict__ d_tok_idx, const int* __restrict__ d_tok_slot,
        const uint8_t* const* __restrict__ exp_dn_sc, const uint8_t* const* __restrict__ exp_dn_qs,
        const float* __restrict__ d_moe_h, const int32_t* __restrict__ hq4, const float* __restrict__ hd,
        float* __restrict__ slot_outs, int uniq_n) {
    __shared__ uint8_t s_signs[128];
    __shared__ uint64_t s_grid[512];
    int tid = threadIdx.y * 32 + threadIdx.x;
    s_signs[tid] = ik_ksigns_iq2xs[tid];
    for (int i = tid; i < 512; i += 128) s_grid[i] = ik_iq2xs_grid[i];
    __syncthreads();

    int u = blockIdx.y;
    if (u >= uniq_n) return;
    int g_row = blockIdx.x * blockDim.y + threadIdx.y;
    if (g_row * 16 >= IK_D) return;
    int t = threadIdx.x, r = t & 15, h = t >> 4, rr = g_row * 16 + r;
    int e = d_uniq_e[u];
    const uint8_t* sc = exp_dn_sc[e];
    const uint8_t* qs = exp_dn_qs[e];
    if (!sc || !qs) return;
    int p0 = d_u_off[u], p1 = d_u_off[u + 1];
    const int nb128 = IK_FF / QK_IQ2;

#if __CUDA_ARCH__ >= 610
    (void)d_moe_h;
    int nb32 = IK_FF >> 5;
    if (p1 - p0 == 1) {
        const int32_t* hq0 = hq4 + (size_t)p0 * nb32 * 8;
        const float* hd0 = hd + (size_t)p0 * nb32;
        float a0 = 0.f;
        for (int b = h; b < nb128; b += 2) {
            a0 = iq2xs_step_1_dp4a((size_t)g_row * nb128 + b, r, b, sc, qs, s_signs, s_grid, hq0, hd0, a0);
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
        if (h == 0 && rr < IK_D) {
            int tk = d_tok_idx[p0], sl = d_tok_slot[p0];
            slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a0;
        }
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const int32_t* hqs[4] = {NULL, NULL, NULL, NULL};
            const float* hds[4] = {NULL, NULL, NULL, NULL};
            for (int i = 0; i < nt; i++) {
                hqs[i] = hq4 + (size_t)(base_p + i) * nb32 * 8;
                hds[i] = hd + (size_t)(base_p + i) * nb32;
            }
            float a[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = h; b < nb128; b += 2) {
                iq2xs_step_n_dp4a((size_t)g_row * nb128 + b, r, b, nt, sc, qs, s_signs, s_grid, hqs, hds, a);
            }
            for (int i = 0; i < nt; i++) {
                a[i] += __shfl_xor_sync(0xFFFFFFFFu, a[i], 16);
                if (h == 0 && rr < IK_D) {
                    int tk = d_tok_idx[base_p + i], sl = d_tok_slot[base_p + i];
                    slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a[i];
                }
            }
        }
    }
#else
    (void)hq4; (void)hd;
    if (p1 - p0 == 1) {
        const float* hp0 = d_moe_h + (size_t)p0 * IK_FF;
        float a0 = 0.f;
        for (int b = h; b < nb128; b += 2) {
            a0 = iq2xs_step_1_f((size_t)g_row * nb128 + b, r, b, sc, qs, s_signs, s_grid, hp0, a0);
        }
        a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
        if (h == 0 && rr < IK_D) {
            int tk = d_tok_idx[p0], sl = d_tok_slot[p0];
            slot_outs[((size_t)tk * IK_NU + sl) * IK_D + rr] = a0;
        }
    } else {
        for (int base_p = p0; base_p < p1; base_p += 4) {
            int nt = p1 - base_p; if (nt > 4) nt = 4;
            const float* hp0 = d_moe_h + (size_t)(base_p + 0) * IK_FF;
            const float* hp1 = (nt > 1) ? (d_moe_h + (size_t)(base_p + 1) * IK_FF) : hp0;
            const float* hp2 = (nt > 2) ? (d_moe_h + (size_t)(base_p + 2) * IK_FF) : hp0;
            const float* hp3 = (nt > 3) ? (d_moe_h + (size_t)(base_p + 3) * IK_FF) : hp0;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            for (int b = h; b < nb128; b += 2) {
                iq2xs_step_n_f((size_t)g_row * nb128 + b, r, b, nt, sc, qs, s_signs, s_grid, hp0, hp1, hp2, hp3, a0, a1, a2, a3);
            }
            a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 16);
            if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 0] * IK_NU + d_tok_slot[base_p + 0]) * IK_D + rr] = a0;
            if (nt > 1) {
                a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 16);
                if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 1] * IK_NU + d_tok_slot[base_p + 1]) * IK_D + rr] = a1;
            }
            if (nt > 2) {
                a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 16);
                if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 2] * IK_NU + d_tok_slot[base_p + 2]) * IK_D + rr] = a2;
            }
            if (nt > 3) {
                a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 16);
                if (h == 0 && rr < IK_D) slot_outs[((size_t)d_tok_idx[base_p + 3] * IK_NU + d_tok_slot[base_p + 3]) * IK_D + rr] = a3;
            }
        }
    }
#endif
}

/* ================================================================
 * PART 2: Pure C11 Host Orchestration (#else) — compiled with gcc -x c
 * ================================================================ */
#else

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "inkling2_gpu: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return -1; } } while (0)

#define CKV(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "inkling2_gpu: CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    } } while (0)

typedef CUresult (CUDAAPI *pfn_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (CUDAAPI *pfn_cuModuleUnload)(CUmodule);
typedef CUresult (CUDAAPI *pfn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (CUDAAPI *pfn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, CUstream, void**, void**);

static void* g_libcuda = NULL;
static pfn_cuModuleLoadData    fn_cuModuleLoadData    = NULL;
static pfn_cuModuleUnload      fn_cuModuleUnload      = NULL;
static pfn_cuModuleGetFunction fn_cuModuleGetFunction = NULL;
static pfn_cuLaunchKernel      fn_cuLaunchKernel      = NULL;

static int load_cuda_driver_api(void) {
    if (fn_cuLaunchKernel) return 0;
    g_libcuda = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_libcuda) g_libcuda = dlopen("libcuda.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!g_libcuda) { fprintf(stderr, "inkling2_gpu: dlopen libcuda.so.1 failed\n"); return -1; }
    fn_cuModuleLoadData    = (pfn_cuModuleLoadData)dlsym(g_libcuda, "cuModuleLoadData");
    fn_cuModuleUnload      = (pfn_cuModuleUnload)dlsym(g_libcuda, "cuModuleUnload");
    fn_cuModuleGetFunction = (pfn_cuModuleGetFunction)dlsym(g_libcuda, "cuModuleGetFunction");
    fn_cuLaunchKernel      = (pfn_cuLaunchKernel)dlsym(g_libcuda, "cuLaunchKernel");
    if (!fn_cuModuleLoadData || !fn_cuModuleGetFunction || !fn_cuLaunchKernel) {
        fprintf(stderr, "inkling2_gpu: dlsym CUDA driver symbols failed\n");
        return -1;
    }
    return 0;
}

/* KINDI_INKLING_TRACE=1: every launch waits for its kernel and logs its name, grid, block and time (stderr), so a
   kernel that does not finish is the last one named */
static int g_trace = -1;
static CUfunction g_trace_fn[256];
static const char* g_trace_name[256];
static int g_trace_n = 0;

static void kl_trace_name(CUfunction fn, const char* name) {
    if (g_trace_n < 256) { g_trace_fn[g_trace_n] = fn; g_trace_name[g_trace_n] = name; g_trace_n++; }
}

static inline void kl_launch(CUfunction fn, unsigned int gx, unsigned int gy, unsigned int gz,
                             unsigned int bx, unsigned int by, unsigned int bz,
                             cudaStream_t st, void** args) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    if (g_trace < 0) { const char* e = getenv("KINDI_INKLING_TRACE"); g_trace = (e && *e && *e != '0'); }
    const char* name = "?";
    struct timespec t0, t1;
    if (g_trace) {
        for (int i = 0; i < g_trace_n; i++) if (g_trace_fn[i] == fn) { name = g_trace_name[i]; break; }
        fprintf(stderr, "[ik trace] %s grid %u,%u,%u block %u,%u,%u: earlier work ...", name, gx, gy, gz, bx, by, bz);
        fflush(stderr);
        cudaStreamSynchronize(st);
        fprintf(stderr, " done; kernel ...");
        fflush(stderr);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }
    CUresult r = fn_cuLaunchKernel(fn, gx, gy, gz, bx, by, bz, 0, (CUstream)st, args, NULL);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "inkling2_gpu: cuLaunchKernel failed with CUresult=%d\n", (int)r);
    }
    if (g_trace) {
        cudaError_t ce = cudaStreamSynchronize(st);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, " %.3f ms%s%s\n", (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6,
                ce == cudaSuccess ? "" : " ERROR ", ce == cudaSuccess ? "" : cudaGetErrorString(ce));
    }
}

typedef enum {
    FN_REPACK_Q8 = 0,
    FN_REPACK_NVFP4,
    FN_REPACK_Q4,
    FN_REPACK_IQ2XXS,
    FN_REPACK_IQ2XS,
    FN_QUANT_Q8,
    FN_Q8G16_MM,
    FN_Q4G16_MM,
    FN_F32_MM,
    FN_BF16_MM,
    FN_BUCKET_MOE,
    FN_NVFP4_GU,
    FN_NVFP4_DN,
    FN_Q5K_GU,
    FN_Q8_DN,
    FN_IQ2XXS_GU,
    FN_IQ2XXS_DN,
    FN_IQ2XS_GU,
    FN_IQ2XS_DN,
    FN_RMSNORM,
    FN_MIX,
    FN_SPLIT3,
    FN_CTRL_CONCAT,
    FN_CTRL_APPLY,
    FN_CTRL_APPLY_PRE,
    FN_CONVOLVE,
    FN_ROLLBACK_CONV,
    FN_CLIP_KV,
    FN_ATTN_TABLE,
    FN_RESIDUAL,
    FN_ROUTE_TOPK,
    FN_MOE_REDUCE,
    FN_SOFTCAP,
    FN_TOP2_S1,
    FN_TOP2_S2,
    FN_COUNT
} IkKernelId;

static const char* k_kernel_names[FN_COUNT] = {
    "k_repack_q8_to_g16",
    "k_repack_nvfp4_to_g16",
    "k_repack_q4_to_g16",
    "k_repack_iq2xxs128",
    "k_repack_iq2xs128",
    "k_quant_q8",
    "k_q8g16_mm",
    "k_q4g16_mm",
    "k_f32_mm",
    "k_bf16_mm",
    "k_bucket_moe",
    "k_nvfp4g16_moe_gu",
    "k_nvfp4g16_moe_dn",
    "k_q5k_moe_gu",
    "k_q8g16_moe_dn",
    "k_iq2xxs128_moe_gu",
    "k_iq2xxs128_moe_dn",
    "k_iq2xs128_moe_gu",
    "k_iq2xs128_moe_dn",
    "k_rmsnorm",
    "k_mix",
    "k_split3",
    "k_ctrl_concat_norm",
    "k_ctrl_apply",
    "k_ctrl_apply_pre",
    "k_convolve",
    "k_rollback_conv",
    "k_clip_vnorm_kvstore",
    "k_attn_table",
    "k_residual_add_or_copy",
    "k_route_topk",
    "k_moe_reduce",
    "k_softcap",
    "k_top2_stage1",
    "k_top2_stage2"
};

typedef int (*ik_cpu_moe_fn)(int il, int ip, int T, const float* x_in, const int* sel, const uint8_t* skip_mask, float* slot_outs);

typedef struct {
    void* ctx;
    int (*forward_step)(void* ctx, int T, const int* toks, const int* poss, float* logits, int need_logits,
                        int* out_tok, float* out_gap);
    int (*qkvg_start)(void* ctx, int ip, int T, const float* in0, const float* in1, const float* in2,
                      const int8_t* xq0, const float* xd0, const int8_t* xq1, const float* xd1,
                      const int8_t* xq2, const float* xd2);
    void (*qkvg_wait)(void* ctx, int ip, int T, float* q, float* k, float* v, float* g, float* r);
    int (*o_start)(void* ctx, int ip, int T, const float* x, const int8_t* xq, const float* xd);
    void (*o_wait)(void* ctx, int ip, int T, float* out);
    int (*moe_start)(void* ctx, int ip, int T, const float* x, const int8_t* xq, const float* xd,
                     const int* sel, const float* wsel, uint8_t* on_gpu_mask);
    void (*moe_wait)(void* ctx, int ip, int T, float* out);
    int (*head_start)(void* ctx, int T, const float* x, const int8_t* xq, const float* xd);
    void (*head_wait)(void* ctx, int T, float* logits);
    void (*reset)(void* ctx);
    void (*on_roll)(void* ctx, int pos);
    int split_q, split_k, split_v, split_g, split_o, split_head;
} IkGpuHooks;

/* ---------------------------------------------------------------- GPU Model State & Upload */
typedef struct {
    int dev;
    int ty;
    int split_off;
    int rows, cols, nb;
    uint16_t* sc;
    int8_t*   qs;
} GpuQ8Mat;

typedef struct {
    GpuQ8Mat wq, wk, wv, wo, wgate, wrel;
    uint16_t *sink_proj, *k_conv_proj, *v_conv_proj;
    float *q_clip, *rel_bias, *rel_clip;
    float *k_conv, *v_conv, *attn_post_conv, *ffn_post_conv;
    float *attn_norm, *q_inp_norm, *k_inp_norm, *v_inp_norm;
    float *attn_res_q, *ffn_res_q, *ffn_norm;
    float *ffn_gate_inp, *exp_bias;
    float *loop_msg_q;
    float *loop_down[4], *loop_gain[4], *loop_forget[4];
    int loop_is_q8;
    GpuQ8Mat loop_down_q8[4], loop_gain_q8[4], loop_forget_q8[4];

    int exp_gu_type, exp_dn_type, has_exp_scale;
    float *d_exp_gu_scale, *d_exp_dn_scale;
    const uint8_t* h_exp_gu_sc[IK_NE];
    const uint8_t* h_exp_gu_qs[IK_NE];
    const uint8_t* h_exp_dn_sc[IK_NE];
    const uint8_t* h_exp_dn_qs[IK_NE];
    const uint8_t* d_exp_gu_sc[IK_NE];
    const uint8_t* d_exp_gu_qs[IK_NE];
    const uint8_t* d_exp_dn_sc[IK_NE];
    const uint8_t* d_exp_dn_qs[IK_NE];
    const uint8_t** d_ptrs_gu_sc;
    const uint8_t** d_ptrs_gu_qs;
    const uint8_t** d_ptrs_dn_sc;
    const uint8_t** d_ptrs_dn_qs;
    uint8_t on_dev[IK_NE];
} GpuPhysLayer;

typedef struct {
    uint8_t *gu_sc, *gu_qs, *dn_sc, *dn_qs;
    int cur_ip, cur_e;
    uint64_t last_use;
} DcacheSlot;

typedef struct {
    int n_dev, mode;
    int devs[IK_MAX_DEV], cc[IK_MAX_DEV];
    CUmodule cu_mod[IK_MAX_DEV];
    CUfunction fn[IK_MAX_DEV][FN_COUNT];
    cudaStream_t stream[IK_MAX_DEV], xfer_stream[IK_MAX_DEV];
    cudaEvent_t ev_done[IK_MAX_DEV];
    int ctx_cap;
    int layer_dev[IK_NL_PHYS];

    uint8_t* d_stage[IK_MAX_DEV];
    size_t stage_cap;

    GpuPhysLayer L[IK_NL_PHYS];
    GpuQ8Mat head_shard[IK_MAX_DEV];
    float *d_output_norm, *d_output_res_q;
    int head_dev_pos;

    uint16_t* kv_k[IK_NL_VIRT];
    uint16_t* kv_v[IK_NL_VIRT];
    int kv_cap[IK_NL_VIRT];
    float* d_conv_hist[IK_NL_VIRT];
    float* d_conv_trace[IK_NL_VIRT];
    int trace_active;

    float *cur[IK_MAX_DEV], *pre[IK_MAX_DEV], *bank_out[IK_MAX_DEV], *msg[IK_MAX_DEV];
    float *blk[IK_MAX_DEV][4], *blk_pre[IK_MAX_DEV][4], *bank[IK_MAX_DEV][4];
    float *in0[IK_MAX_DEV], *in1[IK_MAX_DEV], *in2[IK_MAX_DEV], *tmp3[IK_MAX_DEV], *inpsa[IK_MAX_DEV], *ffninp[IK_MAX_DEV];
    float *q[IK_MAX_DEV], *k[IK_MAX_DEV], *v[IK_MAX_DEV], *g[IK_MAX_DEV], *r[IK_MAX_DEV];
    float *sinks[IK_MAX_DEV], *kcp[IK_MAX_DEV], *vcp[IK_MAX_DEV];
    float *attn[IK_MAX_DEV], *o[IK_MAX_DEV], *ffnin[IK_MAX_DEV], *probs[IK_MAX_DEV];
    float *exps_out[IK_MAX_DEV], *moe_h[IK_MAX_DEV];
    float *ctrl_in[IK_MAX_DEV], *ctrl_g[IK_MAX_DEV], *ctrl_gn[IK_MAX_DEV], *ctrl_fg[IK_MAX_DEV];
    int32_t *xq4[IK_MAX_DEV], *xq4_b[IK_MAX_DEV], *xq4_c[IK_MAX_DEV], *hq4[IK_MAX_DEV];
    float *xd[IK_MAX_DEV], *xd_b[IK_MAX_DEV], *xd_c[IK_MAX_DEV], *hd[IK_MAX_DEV];
    int *d_sel[IK_MAX_DEV], *d_uniq_e[IK_MAX_DEV], *d_u_off[IK_MAX_DEV], *d_tok_idx[IK_MAX_DEV], *d_tok_slot[IK_MAX_DEV];
    int *d_uniq_n[IK_MAX_DEV];
    float *d_wsel[IK_MAX_DEV];
    uint8_t *d_has_cold[IK_MAX_DEV], *d_gpu_mask[IK_MAX_DEV];
    const uint8_t **d_act_gu_sc[IK_MAX_DEV], **d_act_gu_qs[IK_MAX_DEV], **d_act_dn_sc[IK_MAX_DEV], **d_act_dn_qs[IK_MAX_DEV];
    float *d_logits[IK_MAX_DEV];
    IkTop2 *d_top2_blk[IK_MAX_DEV], *d_top2_res[IK_MAX_DEV];

    float *h_pin_f, *h_pin_exps, *h_pin_logits, *h_pin_wsel, *h_pin_qkv;
    int *h_pin_sel, *h_pin_uniq_n, *h_pin_moe_meta;
    uint8_t *h_pin_cold_flag, *h_pin_gpu_mask;
    IkTop2 *h_pin_top2;

    int use_dcache, n_dcache;
    uint64_t dcache_clock;
    DcacheSlot* dcache;
    uint8_t* d_dcache_slab;
    uint8_t* d_static_slab[IK_MAX_DEV];
    void* h_cold_mapped_slab;
    void* h_cold_k20_slab;
    const uint8_t* cold_h_gu_sc[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_h_gu_qs[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_h_dn_sc[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_h_dn_qs[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_d_gu_sc[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_d_gu_qs[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_d_dn_sc[IK_NL_PHYS][IK_NE];
    const uint8_t* cold_d_dn_qs[IK_NL_PHYS][IK_NE];

    int split_q, split_k, split_v, split_g, split_o, split_head;
    int mmap_fd;
    void* mmap_ptr;
    size_t mmap_sz;
    const uint8_t* cpu_tok_embd;
    const float* cpu_tok_norm;
    float* cpu_conv_hist_ptr;
    ik_cpu_moe_fn cpu_moe_fn;
    uint8_t last_moe_mask[IK_MAXB * IK_NU];
    int last_moe_any;
    int dspark_n;
    int dspark_ids[4];
    float* h_dspark_feat;
    uint8_t* h_state;       /* ik_gpu_save_state: conv history + sliding-window KV rings, on the host */
    size_t h_state_sz;
} GpuState;

static inline int lphys(int il) {
    if (il < IK_NL_PHYS) return il;
    return IK_CORE_START + ((il - IK_NL_PHYS) % (IK_CORE_END - IK_CORE_START));
}

static void h_dequant_tok(const GpuState* G, int tok, float* dst) {
    size_t rb = (size_t)(IK_D / QK_Q8) * BQ8;
    const uint8_t* er = G->cpu_tok_embd + (size_t)tok * rb;
    float ss = 0.0f;
    for (int b = 0; b < IK_D / QK_Q8; b++) {
        const uint8_t* B = er + (size_t)b * BQ8;
        uint16_t hs; memcpy(&hs, B, 2);
        uint32_t s = ((uint32_t)(hs & 0x8000u)) << 16;
        uint32_t e = (hs >> 10) & 0x1Fu, m = hs & 0x3FFu;
        uint32_t u = (e == 0) ? (m ? 0 : s) : ((e == 31) ? (s | 0x7F800000u | (m << 13)) : (s | ((e + 112u) << 23) | (m << 13)));
        float d;
        if (e == 0 && m != 0) d = ((hs & 0x8000u) ? -1.0f : 1.0f) * ldexpf((float)m, -24);
        else memcpy(&d, &u, 4);
        const int8_t* qs = (const int8_t*)(B + 2);
        for (int j = 0; j < QK_Q8; j++) {
            float v = (float)qs[j] * d;
            dst[b * QK_Q8 + j] = v;
            ss += v * v;
        }
    }
    float inv = 1.0f / sqrtf(ss / (float)IK_D + IK_RMS_EPS);
    for (int i = 0; i < IK_D; i++) dst[i] *= inv * G->cpu_tok_norm[i];
}

static int upload_dense_slice(GpuState* G, int d, const uint8_t* raw_row0, int ty, int row_start, int rows, int K, GpuQ8Mat* out) {
    CK(cudaSetDevice(G->devs[d]));
    int nb = K >> 5;
    int rows16 = (rows + 15) & ~15;
    int is_q4 = (ty == TY_Q4_0);
    size_t qs_grp_bytes = is_q4 ? 256u : 512u;
    size_t row_bytes = (size_t)nb * (is_q4 ? BQ4 : BQ8);
    int fn_id = is_q4 ? FN_REPACK_Q4 : FN_REPACK_Q8;

    out->dev = d; out->ty = ty; out->split_off = row_start; out->rows = rows; out->cols = K; out->nb = nb;
    CK(cudaMalloc((void**)&out->sc, (size_t)(rows16 / 16) * nb * 16 * sizeof(uint16_t)));
    CK(cudaMalloc((void**)&out->qs, (size_t)(rows16 / 16) * nb * qs_grp_bytes));
    CK(cudaMemsetAsync(out->sc, 0, (size_t)(rows16 / 16) * nb * 16 * sizeof(uint16_t), G->stream[d]));
    CK(cudaMemsetAsync(out->qs, 0, (size_t)(rows16 / 16) * nb * qs_grp_bytes, G->stream[d]));

    int max_chunk = (int)(G->stage_cap / row_bytes);
    max_chunk = (max_chunk / 16) * 16;
    if (max_chunk < 16) max_chunk = 16;
    for (int r = 0; r < rows; r += max_chunk) {
        int n = rows - r; if (n > max_chunk) n = max_chunk;
        size_t bytes = (size_t)n * row_bytes;
        if (bytes > G->stage_cap) {
            CK(cudaFree(G->d_stage[d]));
            CK(cudaMalloc((void**)&G->d_stage[d], bytes));
            G->stage_cap = bytes;
        }
        CK(cudaMemcpyAsync(G->d_stage[d], raw_row0 + (size_t)(row_start + r) * row_bytes, bytes, cudaMemcpyHostToDevice, G->stream[d]));
        uint16_t* p_sc = out->sc + (size_t)(r / 16) * nb * 16;
        int8_t*   p_qs = out->qs + (size_t)(r / 16) * nb * qs_grp_bytes;
        void* args[5] = { &G->d_stage[d], &n, &nb, &p_sc, &p_qs };
        kl_launch(G->fn[d][fn_id], (unsigned)((n + 15) / 16), (unsigned)((nb + 7) / 8), 1, 32, 8, 1, G->stream[d], args);
    }
    CK(cudaStreamSynchronize(G->stream[d]));
    return 0;
}

static int upload_f32_or_bf16(int dev, cudaStream_t st, const void* src, int ty, size_t n_elem, float** dst) {
    if (!src) { *dst = NULL; return 0; }
    CK(cudaSetDevice(dev));
    CK(cudaMalloc((void**)dst, n_elem * sizeof(float)));
    if (ty == TY_F32) {
        CK(cudaMemcpyAsync(*dst, src, n_elem * sizeof(float), cudaMemcpyHostToDevice, st));
    } else {
        float* tmp = (float*)malloc(n_elem * sizeof(float));
        const uint16_t* s = (const uint16_t*)src;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t u = ((uint32_t)s[i]) << 16;
            memcpy(&tmp[i], &u, 4);
        }
        CK(cudaMemcpy(*dst, tmp, n_elem * sizeof(float), cudaMemcpyHostToDevice));
        free(tmp);
    }
    return 0;
}

static int upload_raw_bf16(int dev, cudaStream_t st, const void* src, size_t n_elem, uint16_t** dst) {
    if (!src) { *dst = NULL; return 0; }
    CK(cudaSetDevice(dev));
    CK(cudaMalloc((void**)dst, n_elem * sizeof(uint16_t)));
    CK(cudaMemcpyAsync(*dst, src, n_elem * sizeof(uint16_t), cudaMemcpyHostToDevice, st));
    return 0;
}

static int upload_nvfp4_g16(GpuState* G, int d, const uint8_t* raw, int rows, int K, uint8_t** out_sc, uint8_t** out_qs) {
    CK(cudaSetDevice(G->devs[d]));
    int nb64 = K / QK_NV;
    int rows16 = (rows + 15) & ~15;
    if (*out_sc == NULL) {
        CK(cudaMalloc((void**)out_sc, (size_t)(rows16 / 16) * nb64 * 64));
    }
    if (*out_qs == NULL) {
        CK(cudaMalloc((void**)out_qs, (size_t)(rows16 / 16) * nb64 * 512));
    }
    size_t raw_bytes = (size_t)rows * nb64 * BNV;
    if (raw_bytes > G->stage_cap) {
        CK(cudaFree(G->d_stage[d]));
        CK(cudaMalloc((void**)&G->d_stage[d], raw_bytes));
        G->stage_cap = raw_bytes;
    }
    CK(cudaMemcpyAsync(G->d_stage[d], raw, raw_bytes, cudaMemcpyHostToDevice, G->stream[d]));
    uint8_t *p_sc = *out_sc, *p_qs = *out_qs;
    void* args[5] = { &G->d_stage[d], &rows, &nb64, &p_sc, &p_qs };
    kl_launch(G->fn[d][FN_REPACK_NVFP4], (unsigned)(rows16 / 16), (unsigned)((nb64 + 7) / 8), 1, 32, 8, 1, G->stream[d], args);
    CK(cudaStreamSynchronize(G->stream[d]));
    return 0;
}

static int upload_iq2_g16(GpuState* G, int d, const uint8_t* raw, int ty, int rows, int K, uint8_t** out_sc, uint8_t** out_qs) {
    CK(cudaSetDevice(G->devs[d]));
    int nb128 = K / QK_IQ2;
    int rows16 = (rows + 15) & ~15;
    int is_xs = (ty == TY_IQ2XS_128);
    size_t sc_b = is_xs ? 96u : 32u;
    size_t bblk = is_xs ? BIQ2_XS : BIQ2_XXS;
    if (*out_sc == NULL) CK(cudaMalloc((void**)out_sc, (size_t)(rows16 / 16) * nb128 * sc_b));
    if (*out_qs == NULL) CK(cudaMalloc((void**)out_qs, (size_t)(rows16 / 16) * nb128 * 512));
    size_t raw_bytes = (size_t)rows * nb128 * bblk;
    if (raw_bytes > G->stage_cap) {
        CK(cudaFree(G->d_stage[d]));
        CK(cudaMalloc((void**)&G->d_stage[d], raw_bytes));
        G->stage_cap = raw_bytes;
    }
    CK(cudaMemcpyAsync(G->d_stage[d], raw, raw_bytes, cudaMemcpyHostToDevice, G->stream[d]));
    uint8_t *p_sc = *out_sc, *p_qs = *out_qs;
    void* args[5] = { &G->d_stage[d], &rows, &nb128, &p_sc, &p_qs };
    kl_launch(G->fn[d][is_xs ? FN_REPACK_IQ2XS : FN_REPACK_IQ2XXS],
              (unsigned)(rows16 / 16), (unsigned)((nb128 + 7) / 8), 1, 32, 8, 1, G->stream[d], args);
    CK(cudaStreamSynchronize(G->stream[d]));
    return 0;
}

static void launch_q8g16(const GpuState* G, int d, cudaStream_t st, const GpuQ8Mat* m, int T,
                         const float* xf, const int32_t* xq4, const float* xd,
                         float* y, int ys) {
    if (m->rows <= 0) return;
    int g = (m->rows + 15) >> 4;
    int warps = 4;
    int rows = m->rows, cols = m->cols;
    uint16_t* sc = m->sc;
    int8_t*   qs = m->qs;
    int fn_id = (m->ty == TY_Q4_0) ? FN_Q4G16_MM : FN_Q8G16_MM;
    void* args[10] = { &sc, &qs, &rows, &cols, &T, &xf, &xq4, &xd, &y, &ys };
    kl_launch(G->fn[d][fn_id], (unsigned)((g + warps - 1) / warps), 1, (unsigned)((T + 3) >> 2), 32, (unsigned)warps, 1, st, args);
}

static void launch_quant_if_dp4a(const GpuState* G, int d, const float* xf, int K, int T,
                                 int32_t* xq4, float* xd);

static void launch_static_moe_kernels(GpuState* G, int d, cudaStream_t st,
                                      GpuPhysLayer* gl, int tot_p, int uniq_n) {
    if (uniq_n <= 0) return;
    if (gl->exp_gu_type == TY_NVFP4) {
        void* a_gu[11] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d],
                           &gl->d_ptrs_gu_sc, &gl->d_ptrs_gu_qs, &gl->d_exp_gu_scale,
                           &G->ffnin[d], &G->xq4[d], &G->xd[d], &G->moe_h[d], &uniq_n };
        kl_launch(G->fn[d][FN_NVFP4_GU], (unsigned)(((IK_FF / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, a_gu);
        launch_quant_if_dp4a(G, d, G->moe_h[d], IK_FF, tot_p, G->hq4[d], G->hd[d]);
        void* a_dn[12] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d], &G->d_tok_slot[d],
                           &gl->d_ptrs_dn_sc, &gl->d_ptrs_dn_qs, &gl->d_exp_dn_scale,
                           &G->moe_h[d], &G->hq4[d], &G->hd[d], &G->exps_out[d], &uniq_n };
        kl_launch(G->fn[d][FN_NVFP4_DN], (unsigned)(((IK_D / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, a_dn);
    } else if (gl->exp_gu_type == TY_IQ2XXS_128 || gl->exp_gu_type == TY_IQ2XS_128) {
        int fn_gu = (gl->exp_gu_type == TY_IQ2XS_128) ? FN_IQ2XS_GU : FN_IQ2XXS_GU;
        int fn_dn = (gl->exp_dn_type == TY_IQ2XS_128) ? FN_IQ2XS_DN : FN_IQ2XXS_DN;
        void* a_gu[10] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d],
                           &gl->d_ptrs_gu_sc, &gl->d_ptrs_gu_qs,
                           &G->ffnin[d], &G->xq4[d], &G->xd[d], &G->moe_h[d], &uniq_n };
        kl_launch(G->fn[d][fn_gu], (unsigned)(((IK_FF / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, a_gu);
        launch_quant_if_dp4a(G, d, G->moe_h[d], IK_FF, tot_p, G->hq4[d], G->hd[d]);
        void* a_dn[11] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d], &G->d_tok_slot[d],
                           &gl->d_ptrs_dn_sc, &gl->d_ptrs_dn_qs,
                           &G->moe_h[d], &G->hq4[d], &G->hd[d], &G->exps_out[d], &uniq_n };
        kl_launch(G->fn[d][fn_dn], (unsigned)(((IK_D / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, a_dn);
    } else {
        void* a_gu[7] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d],
                          &gl->d_ptrs_gu_sc, &G->ffnin[d], &G->moe_h[d], &uniq_n };
        kl_launch(G->fn[d][FN_Q5K_GU], (unsigned)((IK_FF + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, a_gu);
        launch_quant_if_dp4a(G, d, G->moe_h[d], IK_FF, tot_p, G->hq4[d], G->hd[d]);
        void* a_dn[11] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d], &G->d_tok_slot[d],
                           &gl->d_ptrs_dn_sc, &gl->d_ptrs_dn_qs,
                           &G->moe_h[d], &G->hq4[d], &G->hd[d], &G->exps_out[d], &uniq_n };
        kl_launch(G->fn[d][FN_Q8_DN], (unsigned)(((IK_D / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, a_dn);
    }
}

static void launch_f32(const GpuState* G, int d, cudaStream_t st, const float* w, int rows, int K, int T,
                       const float* x, int xs, float* y, int ys) {
    int warps = 4;
    void* args[8] = { &w, &rows, &K, &T, &x, &xs, &y, &ys };
    kl_launch(G->fn[d][FN_F32_MM], (unsigned)((rows + warps - 1) / warps), 1, (unsigned)((T + 3) >> 2), 32, (unsigned)warps, 1, st, args);
}

static void launch_bf16(const GpuState* G, int d, cudaStream_t st, const uint16_t* w, int rows, int K, int T,
                        const float* x, int xs, float* y, int ys) {
    int warps = 4;
    void* args[8] = { &w, &rows, &K, &T, &x, &xs, &y, &ys };
    kl_launch(G->fn[d][FN_BF16_MM], (unsigned)((rows + warps - 1) / warps), 1, (unsigned)T, 32, (unsigned)warps, 1, st, args);
}

static void launch_quant_if_dp4a(const GpuState* G, int d, const float* xf, int K, int T,
                                 int32_t* xq4, float* xd) {
    if (G->cc[d] < 61 || T <= 0) return;
    int nb = K >> 5;
    int warps = 4;
    void* args[4] = { &xf, &K, &xq4, &xd };
    kl_launch(G->fn[d][FN_QUANT_Q8], (unsigned)((nb + warps - 1) / warps), (unsigned)T, 1, 32, (unsigned)warps, 1, G->stream[d], args);
}

/* ---------------------------------------------------------------- DCACHE Manager (gpu_pinned) */
static void dcache_prepare_cold_q8g16(const uint8_t* raw, int rows, int K, uint16_t* h_sc, int8_t* h_qs) {
    int nb = K >> 5;
    int rows16 = (rows + 15) & ~15;
    size_t row_b = (size_t)nb * BQ8;
    for (int g = 0; g < rows16 / 16; g++) {
        for (int b = 0; b < nb; b++) {
            uint16_t* ds = h_sc + ((size_t)g * nb + b) * 16;
            int8_t*   dq = h_qs + ((size_t)g * nb + b) * 512;
            for (int r = 0; r < 16; r++) {
                int rr = g * 16 + r;
                if (rr < rows) {
                    const uint8_t* src = raw + (size_t)rr * row_b + (size_t)b * BQ8;
                    uint16_t sc = (uint16_t)src[0] | ((uint16_t)src[1] << 8);
                    ds[r] = sc;
                    memcpy(dq + r * 32, src + 2, 32);
                } else {
                    ds[r] = 0;
                    memset(dq + r * 32, 0, 32);
                }
            }
        }
    }
}

static void dcache_prepare_cold_nvfp4(GpuState* G, int d, const uint8_t* raw, int rows, int K,
                                      uint8_t* h_sc, uint8_t* h_qs) {
    int nb64 = K / QK_NV;
    int rows16 = (rows + 15) & ~15;
    size_t row_b = (size_t)nb64 * BNV;
    for (int g = 0; g < rows16 / 16; g++) {
        for (int b = 0; b < nb64; b++) {
            uint8_t* dst_s = h_sc + ((size_t)g * nb64 + b) * 64;
            uint8_t* dst_q = h_qs + ((size_t)g * nb64 + b) * 512;
            for (int r = 0; r < 16; r++) {
                int rr = g * 16 + r;
                if (rr < rows) {
                    const uint8_t* src = raw + (size_t)rr * row_b + (size_t)b * BNV;
                    memcpy(dst_s + r * 4, src, 4);
                    memcpy(dst_q + r * 32, src + 4, 16);
                    memcpy(dst_q + r * 32 + 16, src + 20, 16);
                } else {
                    memset(dst_s + r * 4, 0, 4);
                    memset(dst_q + r * 32, 0, 32);
                }
            }
        }
    }
    (void)G; (void)d;
}

static int dcache_resolve_experts(GpuState* G, int d, int ip, int T, const int* sel,
                                  int* uniq_e_out, int* u_off_out, int* tok_idx_out, int* tok_slot_out) {
    GpuPhysLayer* gl = &G->L[ip];
    int seen[IK_NE], cnt[IK_NE];
    for (int i = 0; i < IK_NE; i++) seen[i] = -1;
    int uniq_n = 0;
    for (int i = 0; i < T * IK_NU; i++) {
        int e = sel[i];
        if (seen[e] < 0) {
            seen[e] = uniq_n;
            uniq_e_out[uniq_n] = e;
            cnt[uniq_n++] = 0;
        }
        cnt[seen[e]]++;
    }
    u_off_out[0] = 0;
    for (int u = 0; u < uniq_n; u++) {
        u_off_out[u + 1] = u_off_out[u] + cnt[u];
        cnt[u] = 0;
    }
    for (int i = 0; i < T * IK_NU; i++) {
        int u = seen[sel[i]];
        int k = u_off_out[u] + cnt[u]++;
        tok_idx_out[k] = i / IK_NU;
        tok_slot_out[k] = i % IK_NU;
    }

    const uint8_t* h_ptrs_gu_sc[IK_NE];
    const uint8_t* h_ptrs_gu_qs[IK_NE];
    const uint8_t* h_ptrs_dn_sc[IK_NE];
    const uint8_t* h_ptrs_dn_qs[IK_NE];

    size_t gu_sc_sz = (size_t)(IK_FF2 / 16) * (IK_D / QK_NV) * 64;
    size_t gu_qs_sz = (size_t)(IK_FF2 / 16) * (IK_D / QK_NV) * 512;
    size_t dn_sc_sz = (size_t)(IK_D / 16) * (IK_FF / QK_NV) * 64;
    size_t dn_qs_sz = (size_t)(IK_D / 16) * (IK_FF / QK_NV) * 512;

    uint64_t batch_clock = G->dcache_clock;
    for (int u = 0; u < uniq_n; u++) {
        int e = uniq_e_out[u];
        if (gl->on_dev[e]) {
            h_ptrs_gu_sc[e] = gl->d_exp_gu_sc[e];
            h_ptrs_gu_qs[e] = gl->d_exp_gu_qs[e];
            h_ptrs_dn_sc[e] = gl->d_exp_dn_sc[e];
            h_ptrs_dn_qs[e] = gl->d_exp_dn_qs[e];
        } else {
            int hit = -1;
            uint64_t best_lu = UINT64_MAX;
            int victim = -1;
            for (int c = 0; c < G->n_dcache; c++) {
                if (G->dcache[c].cur_ip == ip && G->dcache[c].cur_e == e) { hit = c; break; }
                if (G->dcache[c].last_use <= batch_clock && G->dcache[c].last_use < best_lu) {
                    best_lu = G->dcache[c].last_use;
                    victim = c;
                }
            }
            if (hit < 0 && G->n_dcache > 0 && victim >= 0) {
                hit = victim;
                DcacheSlot* s = &G->dcache[hit];
                s->cur_ip = ip; s->cur_e = e;
                cudaMemcpyAsync(s->gu_sc, G->cold_h_gu_sc[ip][e], gu_sc_sz + gu_qs_sz + dn_sc_sz + dn_qs_sz, cudaMemcpyHostToDevice, G->stream[d]);
            }
            if (hit >= 0) {
                G->dcache[hit].last_use = ++G->dcache_clock;
                h_ptrs_gu_sc[e] = G->dcache[hit].gu_sc;
                h_ptrs_gu_qs[e] = G->dcache[hit].gu_qs;
                h_ptrs_dn_sc[e] = G->dcache[hit].dn_sc;
                h_ptrs_dn_qs[e] = G->dcache[hit].dn_qs;
            } else {
                h_ptrs_gu_sc[e] = G->cold_d_gu_sc[ip][e];
                h_ptrs_gu_qs[e] = G->cold_d_gu_qs[ip][e];
                h_ptrs_dn_sc[e] = G->cold_d_dn_sc[ip][e];
                h_ptrs_dn_qs[e] = G->cold_d_dn_qs[ip][e];
            }
        }
    }
    cudaMemcpyAsync(G->d_act_gu_sc[d], h_ptrs_gu_sc, sizeof(h_ptrs_gu_sc), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_act_gu_qs[d], h_ptrs_gu_qs, sizeof(h_ptrs_gu_qs), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_act_dn_sc[d], h_ptrs_dn_sc, sizeof(h_ptrs_dn_sc), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_act_dn_qs[d], h_ptrs_dn_qs, sizeof(h_ptrs_dn_qs), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_uniq_e[d], uniq_e_out, (size_t)uniq_n * sizeof(int), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_u_off[d], u_off_out, (size_t)(uniq_n + 1) * sizeof(int), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_tok_idx[d], tok_idx_out, (size_t)(T * IK_NU) * sizeof(int), cudaMemcpyHostToDevice, G->stream[d]);
    cudaMemcpyAsync(G->d_tok_slot[d], tok_slot_out, (size_t)(T * IK_NU) * sizeof(int), cudaMemcpyHostToDevice, G->stream[d]);
    return uniq_n;
}

/* ---------------------------------------------------------------- Kernel Launch Wrappers */
static void run_k_rmsnorm(const GpuState* G, int d, cudaStream_t st, float* dst, const float* src, const float* w, int n, int T) {
    void* args[5] = { &dst, &src, &w, &n, &T };
    kl_launch(G->fn[d][FN_RMSNORM], (unsigned)T, 1, 1, 256, 1, 1, st, args);
}

static void run_k_mix(const GpuState* G, int d, cudaStream_t st,
                      const float* b0, const float* b1, const float* b2, const float* b3,
                      int nblk, const float* cur, const float* query, int nq, int T, float* out) {
    void* args[10] = { &b0, &b1, &b2, &b3, &nblk, &cur, &query, &nq, &T, &out };
    kl_launch(G->fn[d][FN_MIX], (unsigned)T, 1, 1, 256, 1, 1, st, args);
}

static void run_k_split3(const GpuState* G, int d, cudaStream_t st, const float* src, float* a, float* b, float* c, int T) {
    void* args[5] = { &src, &a, &b, &c, &T };
    kl_launch(G->fn[d][FN_SPLIT3], (unsigned)((IK_D + 255) / 256), (unsigned)T, 1, 256, 1, 1, st, args);
}

static void run_k_ctrl_concat_norm(const GpuState* G, int d, cudaStream_t st,
                                   const float* cur, const float* msg, int msg_stride, int T, float* ctrl_in) {
    void* args[5] = { &cur, &msg, &msg_stride, &T, &ctrl_in };
    kl_launch(G->fn[d][FN_CTRL_CONCAT], 2, (unsigned)T, 1, 256, 1, 1, st, args);
}

static void run_k_ctrl_apply(const GpuState* G, int d, cudaStream_t st,
                             const float* w_gain, const float* w_forget, const float* g,
                             float* cur, const float* msg, int msg_stride, int T) {
    void* args[7] = { &w_gain, &w_forget, &g, &cur, &msg, &msg_stride, &T };
    kl_launch(G->fn[d][FN_CTRL_APPLY], (unsigned)((IK_D + 127) / 128), (unsigned)T, 1, 128, 1, 1, st, args);
}

static void run_gpu_ctrl(const GpuState* G, int d, cudaStream_t st, const GpuPhysLayer* gl, int c,
                         float* target_vec, const float* mp, int T) {
    run_k_ctrl_concat_norm(G, d, st, target_vec, mp, 4 * IK_D, T, G->ctrl_in[d]);
    if (gl->loop_is_q8) {
        launch_quant_if_dp4a(G, d, G->ctrl_in[d], 2 * IK_D, T, G->xq4[d], G->xd[d]);
        launch_q8g16(G, d, st, &gl->loop_down_q8[c], T, G->ctrl_in[d], G->xq4[d], G->xd[d], G->ctrl_g[d], IK_GRANK);
        launch_quant_if_dp4a(G, d, G->ctrl_g[d], IK_GRANK, T, G->xq4[d], G->xd[d]);
        launch_q8g16(G, d, st, &gl->loop_gain_q8[c],   T, G->ctrl_g[d], G->xq4[d], G->xd[d], G->ctrl_gn[d], IK_D);
        launch_q8g16(G, d, st, &gl->loop_forget_q8[c], T, G->ctrl_g[d], G->xq4[d], G->xd[d], G->ctrl_fg[d], IK_D);
        int stride = 4 * IK_D;
        float* gn = G->ctrl_gn[d];
        float* fg = G->ctrl_fg[d];
        void* args[6] = { &gn, &fg, &target_vec, &mp, &stride, &T };
        kl_launch(G->fn[d][FN_CTRL_APPLY_PRE], (unsigned)((IK_D + 127) / 128), (unsigned)T, 1, 128, 1, 1, st, args);
    } else {
        launch_f32(G, d, st, gl->loop_down[c], IK_GRANK, 2 * IK_D, T, G->ctrl_in[d], 2 * IK_D, G->ctrl_g[d], IK_GRANK);
        run_k_ctrl_apply(G, d, st, gl->loop_gain[c], gl->loop_forget[c], G->ctrl_g[d], target_vec, mp, 4 * IK_D, T);
    }
}

static void run_k_convolve(const GpuState* G, int d, cudaStream_t st,
                           float* cur, int C, int T, const float* weight, const float* dyn,
                           float* hist_slot, float* trace_slot, int ch_off) {
    void* args[8] = { &cur, &C, &T, &weight, &dyn, &hist_slot, &trace_slot, &ch_off };
    kl_launch(G->fn[d][FN_CONVOLVE], (unsigned)((C + 127) / 128), 1, 1, 128, 1, 1, st, args);
}

static void run_k_residual(const GpuState* G, int d, cudaStream_t st,
                           float* cur, const float* a, const float* b, int n, int do_add) {
    void* args[5] = { &cur, &a, &b, &n, &do_add };
    kl_launch(G->fn[d][FN_RESIDUAL], (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, st, args);
}

/* ---------------------------------------------------------------- GPU Layer Forward & Hooks */
static void gpu_sync_copy_vec(GpuState* G, int src_d, const float* d_src, int dst_d, float* d_dst, size_t n_floats) {
    if (src_d == dst_d) {
        if (d_src != d_dst) {
            cudaSetDevice(G->devs[dst_d]);
            cudaMemcpyAsync(d_dst, d_src, n_floats * sizeof(float), cudaMemcpyDeviceToDevice, G->stream[dst_d]);
        }
        return;
    }
    size_t bytes = n_floats * sizeof(float);
    cudaSetDevice(G->devs[src_d]);
    cudaMemcpyAsync(G->h_pin_f, d_src, bytes, cudaMemcpyDeviceToHost, G->stream[src_d]);
    cudaStreamSynchronize(G->stream[src_d]);
    cudaSetDevice(G->devs[dst_d]);
    cudaMemcpyAsync(d_dst, G->h_pin_f, bytes, cudaMemcpyHostToDevice, G->stream[dst_d]);
    cudaStreamSynchronize(G->stream[dst_d]);
}

static int gpu_run_layer(GpuState* G, int il, int ip, int T, int pos0,
                         int* p_nblk, int has_msg, int nbank, int use_msg_mix) {
    int nblk = *p_nblk;
    int d = G->layer_dev[ip];
    GpuPhysLayer* gl = &G->L[ip];
    cudaSetDevice(G->devs[d]);
    cudaStream_t st = G->stream[d];
    float* hist_slot = G->d_conv_hist[il];
    float* trace_slot = (G->trace_active && G->d_conv_trace[il] && T <= 8) ? G->d_conv_trace[il] : NULL;

    if (use_msg_mix && gl->loop_msg_q) {
        const float* b0 = nbank > 0 ? G->bank[d][0] : NULL;
        const float* b1 = nbank > 1 ? G->bank[d][1] : NULL;
        const float* b2 = nbank > 2 ? G->bank[d][2] : NULL;
        const float* b3 = nbank > 3 ? G->bank[d][3] : NULL;
        run_k_mix(G, d, st, b0, b1, b2, b3, nbank, G->bank_out[d], gl->loop_msg_q, 4, T, G->msg[d]);
        has_msg = 1;
    }

    cudaMemcpyAsync(G->inpsa[d], G->cur[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);

    if (gl->attn_res_q) {
        int nq = (ip % IK_BLOCK_SZ != 5) ? 1 : 3;
        const float* b0 = nblk > 0 ? G->blk[d][0] : NULL;
        const float* b1 = nblk > 1 ? G->blk[d][1] : NULL;
        const float* b2 = nblk > 2 ? G->blk[d][2] : NULL;
        const float* b3 = nblk > 3 ? G->blk[d][3] : NULL;
        if (nq == 1) {
            run_k_mix(G, d, st, b0, b1, b2, b3, nblk, G->cur[d], gl->attn_res_q, 1, T, G->in0[d]);
            cudaMemcpyAsync(G->in1[d], G->in0[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(G->in2[d], G->in0[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
        } else {
            run_k_mix(G, d, st, b0, b1, b2, b3, nblk, G->cur[d], gl->attn_res_q, 3, T, G->tmp3[d]);
            run_k_split3(G, d, st, G->tmp3[d], G->in0[d], G->in1[d], G->in2[d], T);
        }
    } else {
        cudaMemcpyAsync(G->in0[d], G->cur[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(G->in1[d], G->cur[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(G->in2[d], G->cur[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
    }

    if (has_msg) {
        float* ins[3] = {G->in0[d], G->in1[d], G->in2[d]};
        for (int c = 0; c < 3; c++) run_gpu_ctrl(G, d, st, gl, c, ins[c], G->msg[d] + c * IK_D, T);
    }

    int is_swa = ((ip % IK_BLOCK_SZ) != 5);
    int same_12 = (is_swa && !has_msg);
    if (is_swa && has_msg) {
        run_k_rmsnorm(G, d, st, G->in0[d], G->in0[d], gl->attn_norm, IK_D, T);
        run_k_rmsnorm(G, d, st, G->in1[d], G->in1[d], gl->attn_norm, IK_D, T);
        run_k_rmsnorm(G, d, st, G->in2[d], G->in2[d], gl->attn_norm, IK_D, T);
    } else if (is_swa) {
        run_k_rmsnorm(G, d, st, G->in0[d], G->in0[d], gl->attn_norm, IK_D, T);
        cudaMemcpyAsync(G->in1[d], G->in0[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(G->in2[d], G->in0[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, st);
    } else {
        run_k_rmsnorm(G, d, st, G->in0[d], G->in0[d], gl->q_inp_norm, IK_D, T);
        run_k_rmsnorm(G, d, st, G->in1[d], G->in1[d], gl->k_inp_norm, IK_D, T);
        run_k_rmsnorm(G, d, st, G->in2[d], G->in2[d], gl->v_inp_norm, IK_D, T);
    }

    launch_quant_if_dp4a(G, d, G->in0[d], IK_D, T, G->xq4[d], G->xd[d]);
    if (!same_12) {
        launch_quant_if_dp4a(G, d, G->in1[d], IK_D, T, G->xq4_b[d], G->xd_b[d]);
        launch_quant_if_dp4a(G, d, G->in2[d], IK_D, T, G->xq4_c[d], G->xd_c[d]);
    }
    const int32_t* q1 = same_12 ? G->xq4[d] : G->xq4_b[d];
    const float*   d1 = same_12 ? G->xd[d]  : G->xd_b[d];
    const int32_t* q2 = same_12 ? G->xq4[d] : G->xq4_c[d];
    const float*   d2 = same_12 ? G->xd[d]  : G->xd_c[d];

    launch_q8g16(G, d, st, &gl->wq,    T, G->in0[d], G->xq4[d], G->xd[d], G->q[d], IK_QDIM);
    launch_q8g16(G, d, st, &gl->wk,    T, G->in1[d], q1,        d1,       G->k[d], IK_GQADIM);
    launch_q8g16(G, d, st, &gl->wv,    T, G->in2[d], q2,        d2,       G->v[d], IK_GQADIM);
    launch_q8g16(G, d, st, &gl->wgate, T, G->in0[d], G->xq4[d], G->xd[d], G->g[d], IK_QDIM);
    launch_q8g16(G, d, st, &gl->wrel,  T, G->in0[d], G->xq4[d], G->xd[d], G->r[d], IK_RDIM);

    launch_bf16(G, d, st, gl->sink_proj, IK_NH, IK_D, T, G->in0[d], IK_D, G->sinks[d], IK_NH);
    if (!is_swa) {
        launch_bf16(G, d, st, gl->k_conv_proj, IK_NCONV * IK_NKV, IK_D, T, G->in1[d], IK_D, G->kcp[d], IK_NCONV * IK_NKV);
        launch_bf16(G, d, st, gl->v_conv_proj, IK_NCONV * IK_NKV, IK_D, T, G->in2[d], IK_D, G->vcp[d], IK_NCONV * IK_NKV);
    }

    run_k_convolve(G, d, st, G->k[d], IK_GQADIM, T, gl->k_conv, is_swa ? NULL : G->kcp[d], hist_slot, trace_slot, 0);
    run_k_convolve(G, d, st, G->v[d], IK_GQADIM, T, gl->v_conv, is_swa ? NULL : G->vcp[d], hist_slot, trace_slot, IK_GQADIM);

    {
        int kv_cap = G->kv_cap[il];
        void* args[12] = { &G->q[d], &G->k[d], &G->v[d], &G->r[d], &gl->q_clip, &gl->rel_clip,
                           &G->kv_k[il], &G->kv_v[il], &kv_cap, &is_swa, &pos0, &T };
        kl_launch(G->fn[d][FN_CLIP_KV], 64, (unsigned)T, 1, 128, 1, 1, st, args);
    }
    {
        int kv_cap = G->kv_cap[il];
        void* args[12] = { &G->q[d], &G->r[d], &G->sinks[d], &gl->rel_bias,
                           &G->kv_k[il], &G->kv_v[il], &G->g[d], &kv_cap, &is_swa, &pos0, &T, &G->attn[d] };
        kl_launch(G->fn[d][FN_ATTN_TABLE], (unsigned)IK_NH, (unsigned)T, 1, 128, 1, 1, st, args);
    }

    launch_quant_if_dp4a(G, d, G->attn[d], IK_QDIM, T, G->xq4[d], G->xd[d]);
    launch_q8g16(G, d, st, &gl->wo, T, G->attn[d], G->xq4[d], G->xd[d], G->o[d], IK_D);

    run_k_convolve(G, d, st, G->o[d], IK_D, T, gl->attn_post_conv, NULL, hist_slot, trace_slot, 2 * IK_GQADIM);

    int total_d = T * IK_D;
    if (ip % IK_BLOCK_SZ == 0) {
        int bi = nblk++;
        cudaMemcpyAsync(G->blk[d][bi], G->inpsa[d], (size_t)total_d * sizeof(float), cudaMemcpyDeviceToDevice, st);
        run_k_residual(G, d, st, G->cur[d], G->o[d], NULL, total_d, 0);
    } else {
        run_k_residual(G, d, st, G->cur[d], G->inpsa[d], G->o[d], total_d, 1);
    }

    cudaMemcpyAsync(G->ffninp[d], G->cur[d], (size_t)total_d * sizeof(float), cudaMemcpyDeviceToDevice, st);
    {
        const float* b0 = nblk > 0 ? G->blk[d][0] : NULL;
        const float* b1 = nblk > 1 ? G->blk[d][1] : NULL;
        const float* b2 = nblk > 2 ? G->blk[d][2] : NULL;
        const float* b3 = nblk > 3 ? G->blk[d][3] : NULL;
        run_k_mix(G, d, st, b0, b1, b2, b3, nblk, G->cur[d], gl->ffn_res_q, 1, T, G->ffnin[d]);
    }
    if (has_msg) run_gpu_ctrl(G, d, st, gl, 3, G->ffnin[d], G->msg[d] + 3 * IK_D, T);
    run_k_rmsnorm(G, d, st, G->ffnin[d], G->ffnin[d], gl->ffn_norm, IK_D, T);

    launch_f32(G, d, st, gl->ffn_gate_inp, IK_NE, IK_D, T, G->ffnin[d], IK_D, G->probs[d], IK_NE);
    {
        void* args[5] = { &G->probs[d], &gl->exp_bias, &T, &G->d_sel[d], &G->d_wsel[d] };
        kl_launch(G->fn[d][FN_ROUTE_TOPK], (unsigned)T, 1, 1, IK_NE, 1, 1, st, args);
    }
    launch_quant_if_dp4a(G, d, G->ffnin[d], IK_D, T, G->xq4[d], G->xd[d]);

    if (G->use_dcache && gl->exp_gu_type == TY_NVFP4) {
        cudaMemcpyAsync(G->h_pin_sel, G->d_sel[d], (size_t)T * IK_NU * sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        int uniq_e[IK_NE], u_off[IK_NE + 1], tok_idx[IK_MAXB * IK_NU], tok_slot[IK_MAXB * IK_NU];
        int uniq_n = dcache_resolve_experts(G, d, ip, T, G->h_pin_sel, uniq_e, u_off, tok_idx, tok_slot);
        {
            void* args[11] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d],
                               &G->d_act_gu_sc[d], &G->d_act_gu_qs[d], &gl->d_exp_gu_scale,
                               &G->ffnin[d], &G->xq4[d], &G->xd[d], &G->moe_h[d], &uniq_n };
            kl_launch(G->fn[d][FN_NVFP4_GU], (unsigned)(((IK_FF / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, args);
        }
        int tot_p = u_off[uniq_n];
        launch_quant_if_dp4a(G, d, G->moe_h[d], IK_FF, tot_p, G->hq4[d], G->hd[d]);
        {
            void* args[12] = { &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d], &G->d_tok_slot[d],
                               &G->d_act_dn_sc[d], &G->d_act_dn_qs[d], &gl->d_exp_dn_scale,
                               &G->moe_h[d], &G->hq4[d], &G->hd[d], &G->exps_out[d], &uniq_n };
            kl_launch(G->fn[d][FN_NVFP4_DN], (unsigned)(((IK_D / 16) + 3) / 4), (unsigned)uniq_n, 1, 32, 4, 1, st, args);
        }
    } else {
        {
            const uint8_t* null_mask = NULL;
            void* args[10] = { &T, &G->d_sel[d], &gl->d_ptrs_gu_sc, &null_mask,
                               &G->d_uniq_e[d], &G->d_u_off[d], &G->d_tok_idx[d], &G->d_tok_slot[d],
                               &G->d_uniq_n[d], &G->d_has_cold[d] };
            kl_launch(G->fn[d][FN_BUCKET_MOE], 1, 1, 1, IK_NE, 1, 1, st, args);
        }
        cudaMemcpyAsync(G->h_pin_uniq_n, G->d_uniq_n[d], sizeof(int), cudaMemcpyDeviceToHost, st);
        cudaMemcpyAsync(G->h_pin_cold_flag, G->d_has_cold[d], 1, cudaMemcpyDeviceToHost, st);
        cudaStreamSynchronize(st);
        int uniq_n = *G->h_pin_uniq_n;
        int has_cold = (*G->h_pin_cold_flag) != 0;

        if (has_cold) {
            cudaMemcpyAsync(G->h_pin_sel, G->d_sel[d], (size_t)T * IK_NU * sizeof(int), cudaMemcpyDeviceToHost, G->xfer_stream[d]);
            cudaMemcpyAsync(G->h_pin_f, G->ffnin[d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToHost, G->xfer_stream[d]);
        }

        if (uniq_n > 0) {
            launch_static_moe_kernels(G, d, st, gl, T * IK_NU, uniq_n);
        }

        if (has_cold && G->cpu_moe_fn) {
            cudaStreamSynchronize(G->xfer_stream[d]);
            for (int i = 0; i < T * IK_NU; i++) G->h_pin_gpu_mask[i] = gl->on_dev[G->h_pin_sel[i]];
            G->cpu_moe_fn(il, ip, T, G->h_pin_f, G->h_pin_sel, G->h_pin_gpu_mask, G->h_pin_exps);
            if (uniq_n > 0) cudaStreamSynchronize(st);
            int i = 0, tot_slots = T * IK_NU;
            while (i < tot_slots) {
                if (G->h_pin_gpu_mask[i]) { i++; continue; }
                int j = i + 1;
                while (j < tot_slots && !G->h_pin_gpu_mask[j]) j++;
                cudaMemcpyAsync(G->exps_out[d] + (size_t)i * IK_D, G->h_pin_exps + (size_t)i * IK_D,
                                (size_t)(j - i) * IK_D * sizeof(float), cudaMemcpyHostToDevice, st);
                i = j;
            }
        }
    }

    {
        void* args[4] = { &G->exps_out[d], &G->d_wsel[d], &T, &G->o[d] };
        kl_launch(G->fn[d][FN_MOE_REDUCE], (unsigned)((IK_D + 255) / 256), (unsigned)T, 1, 256, 1, 1, st, args);
    }
    run_k_convolve(G, d, st, G->o[d], IK_D, T, gl->ffn_post_conv, NULL, hist_slot, trace_slot, 2 * IK_GQADIM + IK_D);
    run_k_residual(G, d, st, G->cur[d], G->ffninp[d], G->o[d], total_d, 1);
    *p_nblk = nblk;
    return 0;
}

static void transfer_blk_state_if_needed(GpuState* G, int src_d, int dst_d, int T, int nblk, int copy_bank, int nbank) {
    if (src_d == dst_d) return;
    size_t vf = (size_t)T * IK_D;
    gpu_sync_copy_vec(G, src_d, G->cur[src_d], dst_d, G->cur[dst_d], vf);
    for (int b = 0; b < nblk; b++)
        gpu_sync_copy_vec(G, src_d, G->blk[src_d][b], dst_d, G->blk[dst_d][b], vf);
    if (copy_bank) {
        for (int b = 0; b < nbank; b++)
            gpu_sync_copy_vec(G, src_d, G->bank[src_d][b], dst_d, G->bank[dst_d][b], vf);
        gpu_sync_copy_vec(G, src_d, G->bank_out[src_d], dst_d, G->bank_out[dst_d], vf);
    }
}

void ik_gpu_set_dspark_targets(void* ctx, int n, const int* ids, float* h_feat) {
    if (!ctx) return;
    GpuState* G = (GpuState*)ctx;
    if (n < 0) n = 0;
    if (n > 4) n = 4;
    G->dspark_n = n;
    for (int i = 0; i < n; i++) G->dspark_ids[i] = ids[i];
    G->h_dspark_feat = h_feat;
}

static void gpu_maybe_tap_dspark(GpuState* G, int visit, int d, int T, const float* d_src) {
    if (G->dspark_n <= 0 || !G->h_dspark_feat) return;
    for (int i = 0; i < G->dspark_n; i++) {
        if (G->dspark_ids[i] == visit) {
            cudaSetDevice(G->devs[d]);
            cudaMemcpyAsync(G->h_pin_f, d_src, (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToHost, G->stream[d]);
            cudaStreamSynchronize(G->stream[d]);
            for (int t = 0; t < T; t++) {
                memcpy(G->h_dspark_feat + ((size_t)t * G->dspark_n + i) * IK_D,
                       G->h_pin_f + (size_t)t * IK_D, IK_D * sizeof(float));
            }
        }
    }
}

static int gpu_forward_core(void* ctx, int T, const int* toks, const int* poss,
                            float* logits_out, int need_logits, int* out_tok, float* out_gap, int* out_all_toks) {
    GpuState* G = (GpuState*)ctx;
    int pos0 = poss[0];
    if (T <= 0 || T > IK_MAXB) return -1;

    int d0 = G->layer_dev[0];
    for (int t = 0; t < T; t++) h_dequant_tok(G, toks[t], G->h_pin_f + (size_t)t * IK_D);
    cudaSetDevice(G->devs[d0]);
    cudaMemcpyAsync(G->cur[d0], G->h_pin_f, (size_t)T * IK_D * sizeof(float), cudaMemcpyHostToDevice, G->stream[d0]);

    int nblk = 0;
    int cur_d = d0;
    int visit = 0;
    for (int il = 0; il < IK_CORE_START; il++) {
        int nd = G->layer_dev[il];
        transfer_blk_state_if_needed(G, cur_d, nd, T, nblk, 0, 0);
        cur_d = nd;
        gpu_maybe_tap_dspark(G, visit++, cur_d, T, G->cur[cur_d]);
        gpu_run_layer(G, il, il, T, pos0, &nblk, 0, 0, 0);
    }

    cudaSetDevice(G->devs[cur_d]);
    cudaMemcpyAsync(G->pre[cur_d], G->cur[cur_d], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, G->stream[cur_d]);
    int nblk_pre = nblk;
    for (int b = 0; b < nblk_pre; b++)
        cudaMemcpyAsync(G->blk_pre[cur_d][b], G->blk[cur_d][b], (size_t)T * IK_D * sizeof(float), cudaMemcpyDeviceToDevice, G->stream[cur_d]);

    int d_core0 = G->layer_dev[IK_CORE_START];
    if (cur_d != d_core0) {
        size_t vf = (size_t)T * IK_D;
        gpu_sync_copy_vec(G, cur_d, G->pre[cur_d], d_core0, G->pre[d_core0], vf);
        for (int b = 0; b < nblk_pre; b++)
            gpu_sync_copy_vec(G, cur_d, G->blk_pre[cur_d][b], d_core0, G->blk_pre[d_core0][b], vf);
        transfer_blk_state_if_needed(G, cur_d, d_core0, T, nblk, 0, 0);
        cur_d = d_core0;
    }

    int nbank = 0;
    for (int pass = 0; pass < IK_N_PASS; pass++) {
        if (pass == 0) {
            for (int ip = IK_CORE_START; ip < IK_CORE_END; ip++) {
                int nd = G->layer_dev[ip];
                transfer_blk_state_if_needed(G, cur_d, nd, T, nblk, 0, 0);
                cur_d = nd;
                gpu_maybe_tap_dspark(G, visit++, cur_d, T, G->cur[cur_d]);
                gpu_run_layer(G, ip, ip, T, pos0, &nblk, 0, 0, 0);
            }
        } else {
            nbank = 0;
            size_t vf = (size_t)T * IK_D;
            for (int b = nblk_pre + 1; b < nblk; b++) {
                gpu_sync_copy_vec(G, cur_d, G->blk[cur_d][b], d_core0, G->bank[d_core0][nbank], vf);
                nbank++;
            }
            gpu_sync_copy_vec(G, cur_d, G->cur[cur_d], d_core0, G->bank_out[d_core0], vf);

            cudaSetDevice(G->devs[d_core0]);
            cudaMemcpyAsync(G->cur[d_core0], G->pre[d_core0], vf * sizeof(float), cudaMemcpyDeviceToDevice, G->stream[d_core0]);
            nblk = nblk_pre;
            for (int b = 0; b < nblk_pre; b++)
                cudaMemcpyAsync(G->blk[d_core0][b], G->blk_pre[d_core0][b], vf * sizeof(float), cudaMemcpyDeviceToDevice, G->stream[d_core0]);
            cur_d = d_core0;

            for (int ip = IK_CORE_START; ip < IK_CORE_END; ip++) {
                int il = IK_NL_PHYS + (pass - 1) * (IK_CORE_END - IK_CORE_START) + (ip - IK_CORE_START);
                int nd = G->layer_dev[ip];
                if (nd != cur_d) {
                    transfer_blk_state_if_needed(G, cur_d, nd, T, nblk, 1, nbank);
                    cur_d = nd;
                }
                gpu_maybe_tap_dspark(G, visit++, cur_d, T, G->cur[cur_d]);
                gpu_run_layer(G, il, ip, T, pos0, &nblk, 0, nbank, 1);
            }
        }
    }

    for (int il = IK_CORE_END; il < IK_NL_PHYS; il++) {
        int nd = G->layer_dev[il];
        transfer_blk_state_if_needed(G, cur_d, nd, T, nblk, 0, 0);
        cur_d = nd;
        gpu_maybe_tap_dspark(G, visit++, cur_d, T, G->cur[cur_d]);
        gpu_run_layer(G, il, il, T, pos0, &nblk, 0, 0, 0);
    }

    if (need_logits || out_tok || out_all_toks || G->dspark_n > 0) {
        int hd = G->head_dev_pos;
        transfer_blk_state_if_needed(G, cur_d, hd, T, nblk, 0, 0);
        cudaSetDevice(G->devs[hd]);
        const float* b0 = nblk > 0 ? G->blk[hd][0] : NULL;
        const float* b1 = nblk > 1 ? G->blk[hd][1] : NULL;
        const float* b2 = nblk > 2 ? G->blk[hd][2] : NULL;
        const float* b3 = nblk > 3 ? G->blk[hd][3] : NULL;
        run_k_mix(G, hd, G->stream[hd], b0, b1, b2, b3, nblk, G->cur[hd], G->d_output_res_q, 1, T, G->in0[hd]);
        run_k_rmsnorm(G, hd, G->stream[hd], G->in0[hd], G->in0[hd], G->d_output_norm, IK_D, T);
        gpu_maybe_tap_dspark(G, visit, hd, T, G->in0[hd]);

        if (need_logits || out_tok || out_all_toks) {
            for (int s = 0; s < G->n_dev; s++) {
                if (G->head_shard[s].rows <= 0) continue;
                if (s != hd) gpu_sync_copy_vec(G, hd, G->in0[hd], s, G->in0[s], (size_t)T * IK_D);
            }
        }

        if (out_all_toks && !need_logits) {
            for (int tc0 = 0; tc0 < T; tc0 += IK_MAX_LOGIT_T) {
                int nt = T - tc0;
                if (nt > IK_MAX_LOGIT_T) nt = IK_MAX_LOGIT_T;
                for (int s = 0; s < G->n_dev; s++) {
                    if (G->head_shard[s].rows <= 0) continue;
                    cudaSetDevice(G->devs[s]);
                    const float* xchunk = G->in0[s] + (size_t)tc0 * IK_D;
                    launch_quant_if_dp4a(G, s, xchunk, IK_D, nt, G->xq4[s], G->xd[s]);
                    launch_q8g16(G, s, G->stream[s], &G->head_shard[s], nt, xchunk, G->xq4[s], G->xd[s], G->d_logits[s], G->head_shard[s].rows);
                    int rows = G->head_shard[s].rows, off = G->head_shard[s].split_off, nblk_t2 = 128;
                    for (int k = 0; k < nt; k++) {
                        float* d_log_k = G->d_logits[s] + (size_t)k * rows;
                        IkTop2* d_res_k = G->d_top2_res[s] + k;
                        void* a1[4] = { &d_log_k, &rows, &off, &G->d_top2_blk[s] };
                        kl_launch(G->fn[s][FN_TOP2_S1], 128, 1, 1, 256, 1, 1, G->stream[s], a1);
                        void* a2[3] = { &G->d_top2_blk[s], &nblk_t2, &d_res_k };
                        kl_launch(G->fn[s][FN_TOP2_S2], 1, 1, 1, 256, 1, 1, G->stream[s], a2);
                    }
                    cudaMemcpyAsync(&G->h_pin_top2[s * IK_MAX_LOGIT_T], G->d_top2_res[s],
                                    (size_t)nt * sizeof(IkTop2), cudaMemcpyDeviceToHost, G->stream[s]);
                }
                for (int s = 0; s < G->n_dev; s++) {
                    if (G->head_shard[s].rows <= 0) continue;
                    cudaSetDevice(G->devs[s]);
                    cudaStreamSynchronize(G->stream[s]);
                }
                for (int k = 0; k < nt; k++) {
                    int bt = 0;
                    float m1 = -1e30f;
                    for (int s = 0; s < G->n_dev; s++) {
                        if (G->head_shard[s].rows <= 0) continue;
                        IkTop2 r = G->h_pin_top2[s * IK_MAX_LOGIT_T + k];
                        if (r.m1 > m1 || (r.m1 == m1 && r.tok < bt)) {
                            m1 = r.m1;
                            bt = r.tok;
                        }
                    }
                    out_all_toks[tc0 + k] = bt;
                }
            }
        } else if (out_tok && !need_logits) {
            for (int s = 0; s < G->n_dev; s++) {
                if (G->head_shard[s].rows <= 0) continue;
                cudaSetDevice(G->devs[s]);
                const float* xt = G->in0[s] + (size_t)(T - 1) * IK_D;
                launch_quant_if_dp4a(G, s, xt, IK_D, 1, G->xq4[s], G->xd[s]);
                launch_q8g16(G, s, G->stream[s], &G->head_shard[s], 1, xt, G->xq4[s], G->xd[s], G->d_logits[s], G->head_shard[s].rows);
                int rows = G->head_shard[s].rows, off = G->head_shard[s].split_off, nblk_t2 = 128;
                void* a1[4] = { &G->d_logits[s], &rows, &off, &G->d_top2_blk[s] };
                kl_launch(G->fn[s][FN_TOP2_S1], 128, 1, 1, 256, 1, 1, G->stream[s], a1);
                void* a2[3] = { &G->d_top2_blk[s], &nblk_t2, &G->d_top2_res[s] };
                kl_launch(G->fn[s][FN_TOP2_S2], 1, 1, 1, 256, 1, 1, G->stream[s], a2);
                cudaMemcpyAsync(&G->h_pin_top2[s * IK_MAX_LOGIT_T], G->d_top2_res[s], sizeof(IkTop2), cudaMemcpyDeviceToHost, G->stream[s]);
            }
            int bt = 0; float m1 = -1e30f, m2 = -1e30f;
            for (int s = 0; s < G->n_dev; s++) {
                if (G->head_shard[s].rows <= 0) continue;
                cudaSetDevice(G->devs[s]);
                cudaStreamSynchronize(G->stream[s]);
                IkTop2 r = G->h_pin_top2[s * IK_MAX_LOGIT_T];
                if (r.m1 > m1 || (r.m1 == m1 && r.tok < bt)) {
                    float old = m1; m1 = r.m1; bt = r.tok;
                    if (old > m2) m2 = old;
                    if (r.m2 > m2) m2 = r.m2;
                } else if (r.m1 > m2) {
                    m2 = r.m1;
                }
            }
            *out_tok = bt;
            if (out_gap) *out_gap = IK_LOGIT_CAP * (tanhf(m1 / IK_LOGIT_CAP) - tanhf(m2 / IK_LOGIT_CAP));
        } else if (need_logits) {
            for (int tc0 = 0; tc0 < T; tc0 += IK_MAX_LOGIT_T) {
                int nt = T - tc0;
                if (nt > IK_MAX_LOGIT_T) nt = IK_MAX_LOGIT_T;
                for (int s = 0; s < G->n_dev; s++) {
                    if (G->head_shard[s].rows <= 0) continue;
                    cudaSetDevice(G->devs[s]);
                    const float* xchunk = G->in0[s] + (size_t)tc0 * IK_D;
                    launch_quant_if_dp4a(G, s, xchunk, IK_D, nt, G->xq4[s], G->xd[s]);
                    launch_q8g16(G, s, G->stream[s], &G->head_shard[s], nt, xchunk, G->xq4[s], G->xd[s], G->d_logits[s], G->head_shard[s].rows);
                    size_t sz = (size_t)nt * G->head_shard[s].rows;
                    void* asc[2] = { &G->d_logits[s], &sz };
                    kl_launch(G->fn[s][FN_SOFTCAP], (unsigned)((sz + 255) / 256), 1, 1, 256, 1, 1, G->stream[s], asc);
                    cudaMemcpyAsync(G->h_pin_logits + (size_t)s * IK_MAX_LOGIT_T * IK_V_UNPAD, G->d_logits[s], sz * sizeof(float), cudaMemcpyDeviceToHost, G->stream[s]);
                }
                for (int s = 0; s < G->n_dev; s++) {
                    if (G->head_shard[s].rows <= 0) continue;
                    cudaSetDevice(G->devs[s]);
                    cudaStreamSynchronize(G->stream[s]);
                    const float* src = G->h_pin_logits + (size_t)s * IK_MAX_LOGIT_T * IK_V_UNPAD;
                    int rows = G->head_shard[s].rows, off = G->head_shard[s].split_off;
                    for (int t = 0; t < nt; t++)
                        memcpy(logits_out + (size_t)(tc0 + t) * IK_V_UNPAD + off, src + (size_t)t * rows, (size_t)rows * sizeof(float));
                }
            }
        }
    }
    return 0;
}

static int ik_gpu_forward_step(void* ctx, int T, const int* toks, const int* poss,
                               float* logits_out, int need_logits, int* out_tok, float* out_gap) {
    return gpu_forward_core(ctx, T, toks, poss, logits_out, need_logits, out_tok, out_gap, NULL);
}

int ik_gpu_forward_greedy_batch(void* ctx, int T, int pos0, const int* toks, int* out_toks) {
    if (!ctx || !toks || !out_toks || T <= 0 || T > IK_MAXB) return -1;
    GpuState* G = (GpuState*)ctx;
    if (G->mode == 0) return -1;
    int poss[IK_MAXB];
    for (int t = 0; t < T; t++) poss[t] = pos0 + t;
    return gpu_forward_core(ctx, T, toks, poss, NULL, 0, NULL, NULL, out_toks);
}

void ik_gpu_enable_conv_trace(void* ctx, int on) {
    if (!ctx) return;
    GpuState* G = (GpuState*)ctx;
    if (G->mode == 0) return;
    if (on) {
        for (int il = 0; il < IK_NL_VIRT; il++) {
            if (G->d_conv_hist[il] && !G->d_conv_trace[il]) {
                int d = G->layer_dev[lphys(il)];
                cudaSetDevice(G->devs[d]);
                cudaMalloc((void**)&G->d_conv_trace[il], (size_t)11 * IK_CHSCONV * sizeof(float));
            }
        }
    }
    G->trace_active = on ? 1 : 0;
}

void ik_gpu_rollback_conv(void* ctx, int k_acc) {
    if (!ctx || k_acc < 1 || k_acc > 8) return;
    GpuState* G = (GpuState*)ctx;
    if (G->mode == 0) return;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        if (!G->d_conv_hist[il] || !G->d_conv_trace[il]) continue;
        int d = G->layer_dev[lphys(il)];
        cudaSetDevice(G->devs[d]);
        void* a[3] = { &G->d_conv_hist[il], &G->d_conv_trace[il], &k_acc };
        kl_launch(G->fn[d][FN_ROLLBACK_CONV], (unsigned)((IK_CHSCONV + 255) / 256), 1, 1, 256, 1, 1, G->stream[d], a);
    }
}

/* kindi serve: a host copy of the state that the positions alone do not give back (the short-conv history and the
   sliding-window KV rings), taken at the end of a prompt, so that the next request of the conversation can go back
   there after the reply. save = 1 copies to the host, 0 back to the card. Returns 0, or -1 (nothing saved, no
   memory, or a CUDA error). */
static int gpu_state_copy(GpuState* G, int save) {
    size_t cb = (size_t)3 * IK_CHSCONV * sizeof(float), need = 0;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        if (G->d_conv_hist[il]) need += cb;
        if (G->kv_k[il] && (lphys(il) % IK_BLOCK_SZ) != 5) need += (size_t)2 * G->kv_cap[il] * IK_GQADIM * sizeof(uint16_t);
    }
    if (!save && G->h_state_sz != need) return -1;
    if (save && !G->h_state) {
        G->h_state = (uint8_t*)malloc(need);
        if (!G->h_state) return -1;
    }
    size_t off = 0;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        int d = G->layer_dev[lphys(il)];
        cudaSetDevice(G->devs[d]);
        cudaStream_t st = G->stream[d];
        void* dev[3] = { G->d_conv_hist[il], NULL, NULL };
        size_t sz[3] = { cb, 0, 0 };
        if (G->kv_k[il] && (lphys(il) % IK_BLOCK_SZ) != 5) {
            dev[1] = G->kv_k[il]; dev[2] = G->kv_v[il];
            sz[1] = sz[2] = (size_t)G->kv_cap[il] * IK_GQADIM * sizeof(uint16_t);
        }
        for (int j = 0; j < 3; j++) {
            if (!dev[j]) continue;
            if (save) cudaMemcpyAsync(G->h_state + off, dev[j], sz[j], cudaMemcpyDeviceToHost, st);
            else cudaMemcpyAsync(dev[j], G->h_state + off, sz[j], cudaMemcpyHostToDevice, st);
            off += sz[j];
        }
    }
    for (int d = 0; d < G->n_dev; d++) { cudaSetDevice(G->devs[d]); cudaStreamSynchronize(G->stream[d]); }
    if (cudaGetLastError() != cudaSuccess) { G->h_state_sz = 0; return -1; }
    if (save) G->h_state_sz = need;
    return 0;
}

int ik_gpu_save_state(void* ctx) {
    GpuState* G = (GpuState*)ctx;
    if (!G || G->mode == 0) return -1;
    return gpu_state_copy(G, 1);
}

int ik_gpu_load_state(void* ctx) {
    GpuState* G = (GpuState*)ctx;
    if (!G || G->mode == 0) return -1;
    return gpu_state_copy(G, 0);
}

int ik_gpu_head_argmax(void* ctx, const float* x_2304, int* out_tok) {
    if (!ctx || !x_2304 || !out_tok) return -1;
    GpuState* G = (GpuState*)ctx;
    if (G->mode == 0) return -1;
    for (int s = 0; s < G->n_dev; s++) {
        if (G->head_shard[s].rows <= 0) continue;
        cudaSetDevice(G->devs[s]);
        cudaStream_t st = G->stream[s];
        cudaMemcpyAsync(G->in0[s], x_2304, (size_t)IK_D * sizeof(float), cudaMemcpyHostToDevice, st);
        launch_quant_if_dp4a(G, s, G->in0[s], IK_D, 1, G->xq4[s], G->xd[s]);
        launch_q8g16(G, s, st, &G->head_shard[s], 1, G->in0[s], G->xq4[s], G->xd[s], G->d_logits[s], G->head_shard[s].rows);
        int rows = G->head_shard[s].rows, off = G->head_shard[s].split_off, nblk_t2 = 128;
        void* a1[4] = { &G->d_logits[s], &rows, &off, &G->d_top2_blk[s] };
        kl_launch(G->fn[s][FN_TOP2_S1], 128, 1, 1, 256, 1, 1, st, a1);
        void* a2[3] = { &G->d_top2_blk[s], &nblk_t2, &G->d_top2_res[s] };
        kl_launch(G->fn[s][FN_TOP2_S2], 1, 1, 1, 256, 1, 1, st, a2);
        cudaMemcpyAsync(&G->h_pin_top2[s * IK_MAX_LOGIT_T], G->d_top2_res[s], sizeof(IkTop2), cudaMemcpyDeviceToHost, st);
    }
    int bt = 0;
    float m1 = -1e30f;
    for (int s = 0; s < G->n_dev; s++) {
        if (G->head_shard[s].rows <= 0) continue;
        cudaSetDevice(G->devs[s]);
        cudaStreamSynchronize(G->stream[s]);
        IkTop2 r = G->h_pin_top2[s * IK_MAX_LOGIT_T];
        if (r.m1 > m1 || (r.m1 == m1 && r.tok < bt)) {
            m1 = r.m1;
            bt = r.tok;
        }
    }
    *out_tok = bt;
    return 0;
}

int ik_gpu_head_argmax_val(void* ctx, const float* x_2304) {
    int tok = -1;
    if (ik_gpu_head_argmax(ctx, x_2304, &tok) != 0) return -1;
    return tok;
}

/* ---------------------------------------------------------------- Concurrent Split Hooks (hetero_k420) */
static void scatter_rows(const float* h_pin, int T, int gpu_rows, float* dst, int dst_off, int dst_stride) {
    if (gpu_rows <= 0) return;
    for (int t = 0; t < T; t++)
        memcpy(dst + (size_t)t * dst_stride + dst_off, h_pin + (size_t)t * gpu_rows, (size_t)gpu_rows * sizeof(float));
}

static int ik_gpu_qkvg_start(void* ctx, int ip, int T, const float* in0, const float* in1, const float* in2,
                             const int8_t* xq0, const float* xd0, const int8_t* xq1, const float* xd1,
                             const int8_t* xq2, const float* xd2) {
    (void)xq0; (void)xd0; (void)xq1; (void)xd1; (void)xq2; (void)xd2;
    GpuState* G = (GpuState*)ctx;
    GpuPhysLayer* gl = &G->L[ip];
    cudaSetDevice(G->devs[0]);
    cudaStream_t st = G->stream[0];
    int same = (in0 == in1 && in0 == in2);
    size_t vf = (size_t)T * IK_D;
    size_t sz = vf * sizeof(float);
    float* hp0 = G->h_pin_f;
    memcpy(hp0, in0, sz);
    cudaMemcpyAsync(G->in0[0], hp0, sz, cudaMemcpyHostToDevice, st);
    if (!same) {
        float* hp1 = hp0 + vf;
        float* hp2 = hp1 + vf;
        memcpy(hp1, in1, sz);
        memcpy(hp2, in2, sz);
        cudaMemcpyAsync(G->in1[0], hp1, sz, cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(G->in2[0], hp2, sz, cudaMemcpyHostToDevice, st);
    }
    const float* p1 = same ? G->in0[0] : G->in1[0];
    const float* p2 = same ? G->in0[0] : G->in2[0];
    launch_quant_if_dp4a(G, 0, G->in0[0], IK_D, T, G->xq4[0], G->xd[0]);
    if (!same) {
        launch_quant_if_dp4a(G, 0, p1, IK_D, T, G->xq4_b[0], G->xd_b[0]);
        launch_quant_if_dp4a(G, 0, p2, IK_D, T, G->xq4_c[0], G->xd_c[0]);
    }
    const int32_t* q1 = same ? G->xq4[0] : G->xq4_b[0];
    const float*   d1 = same ? G->xd[0]  : G->xd_b[0];
    const int32_t* q2 = same ? G->xq4[0] : G->xq4_c[0];
    const float*   d2 = same ? G->xd[0]  : G->xd_c[0];
    if (gl->wq.rows > 0)    launch_q8g16(G, 0, st, &gl->wq,    T, G->in0[0], G->xq4[0], G->xd[0], G->q[0], gl->wq.rows);
    if (gl->wk.rows > 0)    launch_q8g16(G, 0, st, &gl->wk,    T, p1,        q1,        d1,       G->k[0], gl->wk.rows);
    if (gl->wv.rows > 0)    launch_q8g16(G, 0, st, &gl->wv,    T, p2,        q2,        d2,       G->v[0], gl->wv.rows);
    if (gl->wgate.rows > 0) launch_q8g16(G, 0, st, &gl->wgate, T, G->in0[0], G->xq4[0], G->xd[0], G->g[0], gl->wgate.rows);
    return 0;
}

static void copy_back_rows(cudaStream_t st, const float* d_src, float* h_pin, int T, int gpu_rows, float* dst, int dst_off, int dst_stride) {
    if (gpu_rows <= 0) return;
    cudaMemcpyAsync(h_pin, d_src, (size_t)T * gpu_rows * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    scatter_rows(h_pin, T, gpu_rows, dst, dst_off, dst_stride);
}

static void ik_gpu_qkvg_wait(void* ctx, int ip, int T, float* q, float* k, float* v, float* g, float* r) {
    (void)r;
    GpuState* G = (GpuState*)ctx;
    GpuPhysLayer* gl = &G->L[ip];
    if (gl->wq.rows <= 0 && gl->wk.rows <= 0 && gl->wv.rows <= 0 && gl->wgate.rows <= 0) return;
    cudaSetDevice(G->devs[0]);
    cudaStream_t st = G->stream[0];
    float* p = G->h_pin_qkv;
    float* pq = p; if (gl->wq.rows > 0) { cudaMemcpyAsync(pq, G->q[0], (size_t)T * gl->wq.rows * sizeof(float), cudaMemcpyDeviceToHost, st); p += (size_t)T * gl->wq.rows; }
    float* pk = p; if (gl->wk.rows > 0) { cudaMemcpyAsync(pk, G->k[0], (size_t)T * gl->wk.rows * sizeof(float), cudaMemcpyDeviceToHost, st); p += (size_t)T * gl->wk.rows; }
    float* pv = p; if (gl->wv.rows > 0) { cudaMemcpyAsync(pv, G->v[0], (size_t)T * gl->wv.rows * sizeof(float), cudaMemcpyDeviceToHost, st); p += (size_t)T * gl->wv.rows; }
    float* pg = p; if (gl->wgate.rows > 0) { cudaMemcpyAsync(pg, G->g[0], (size_t)T * gl->wgate.rows * sizeof(float), cudaMemcpyDeviceToHost, st); }
    cudaStreamSynchronize(st);
    scatter_rows(pq, T, gl->wq.rows,    q, gl->wq.split_off,    IK_QDIM);
    scatter_rows(pk, T, gl->wk.rows,    k, gl->wk.split_off,    IK_GQADIM);
    scatter_rows(pv, T, gl->wv.rows,    v, gl->wv.split_off,    IK_GQADIM);
    scatter_rows(pg, T, gl->wgate.rows, g, gl->wgate.split_off, IK_QDIM);
}

static int ik_gpu_o_start(void* ctx, int ip, int T, const float* x, const int8_t* xq, const float* xd) {
    (void)xq; (void)xd;
    GpuState* G = (GpuState*)ctx;
    GpuPhysLayer* gl = &G->L[ip];
    if (gl->wo.rows <= 0) return 0;
    cudaSetDevice(G->devs[0]);
    cudaStream_t st = G->stream[0];
    size_t sz = (size_t)T * IK_QDIM * sizeof(float);
    memcpy(G->h_pin_f, x, sz);
    cudaMemcpyAsync(G->attn[0], G->h_pin_f, sz, cudaMemcpyHostToDevice, st);
    launch_quant_if_dp4a(G, 0, G->attn[0], IK_QDIM, T, G->xq4[0], G->xd[0]);
    launch_q8g16(G, 0, st, &gl->wo, T, G->attn[0], G->xq4[0], G->xd[0], G->o[0], gl->wo.rows);
    return 0;
}

static void ik_gpu_o_wait(void* ctx, int ip, int T, float* out) {
    GpuState* G = (GpuState*)ctx;
    GpuPhysLayer* gl = &G->L[ip];
    cudaSetDevice(G->devs[0]);
    copy_back_rows(G->stream[0], G->o[0], G->h_pin_qkv, T, gl->wo.rows, out, gl->wo.split_off, IK_D);
}

static int ik_gpu_moe_start(void* ctx, int ip, int T, const float* x, const int8_t* xq, const float* xd,
                            const int* sel, const float* wsel, uint8_t* on_gpu_mask) {
    (void)xq; (void)xd; (void)wsel;
    GpuState* G = (GpuState*)ctx;
    GpuPhysLayer* gl = &G->L[ip];
    int seen[IK_NE], cnt[IK_NE];
    for (int i = 0; i < IK_NE; i++) seen[i] = -1;
    int* uniq_e   = G->h_pin_moe_meta;
    int* u_off    = uniq_e + IK_NE;
    int* tok_idx  = u_off + (IK_NE + 1);
    int* tok_slot = tok_idx + IK_MAXB * IK_NU;
    int uniq_n = 0;
    for (int i = 0; i < T * IK_NU; i++) {
        int e = sel[i];
        uint8_t og = gl->on_dev[e];
        on_gpu_mask[i] = og;
        G->last_moe_mask[i] = og;
        if (!og) continue;
        if (seen[e] < 0) { seen[e] = uniq_n; uniq_e[uniq_n] = e; cnt[uniq_n++] = 0; }
        cnt[seen[e]]++;
    }
    G->last_moe_any = (uniq_n > 0);
    if (uniq_n == 0) return 0;
    u_off[0] = 0;
    for (int u = 0; u < uniq_n; u++) { u_off[u + 1] = u_off[u] + cnt[u]; cnt[u] = 0; }
    for (int i = 0; i < T * IK_NU; i++) {
        if (!on_gpu_mask[i]) continue;
        int u = seen[sel[i]];
        int k = u_off[u] + cnt[u]++;
        tok_idx[k] = i / IK_NU;
        tok_slot[k] = i % IK_NU;
    }
    cudaSetDevice(G->devs[0]);
    cudaStream_t st = G->stream[0];
    memcpy(G->h_pin_f, x, (size_t)T * IK_D * sizeof(float));
    cudaMemcpyAsync(G->ffnin[0], G->h_pin_f, (size_t)T * IK_D * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(G->d_uniq_e[0], uniq_e, (size_t)uniq_n * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(G->d_u_off[0], u_off, (size_t)(uniq_n + 1) * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(G->d_tok_idx[0], tok_idx, (size_t)u_off[uniq_n] * sizeof(int), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(G->d_tok_slot[0], tok_slot, (size_t)u_off[uniq_n] * sizeof(int), cudaMemcpyHostToDevice, st);
    {
        launch_quant_if_dp4a(G, 0, G->ffnin[0], IK_D, T, G->xq4[0], G->xd[0]);
        launch_static_moe_kernels(G, 0, st, gl, u_off[uniq_n], uniq_n);
    }
    return 0;
}

static void ik_gpu_moe_wait(void* ctx, int ip, int T, float* out) {
    (void)ip;
    GpuState* G = (GpuState*)ctx;
    if (!G->last_moe_any) return;
    cudaSetDevice(G->devs[0]);
    cudaMemcpyAsync(G->h_pin_exps, G->exps_out[0], (size_t)T * IK_NU * IK_D * sizeof(float), cudaMemcpyDeviceToHost, G->stream[0]);
    cudaStreamSynchronize(G->stream[0]);
    for (int i = 0; i < T * IK_NU; i++) {
        if (G->last_moe_mask[i]) memcpy(out + (size_t)i * IK_D, G->h_pin_exps + (size_t)i * IK_D, IK_D * sizeof(float));
    }
}

static int ik_gpu_head_start(void* ctx, int T, const float* x, const int8_t* xq, const float* xd) {
    (void)xq; (void)xd;
    GpuState* G = (GpuState*)ctx;
    if (G->head_shard[0].rows <= 0) return 0;
    cudaSetDevice(G->devs[0]);
    cudaStream_t st = G->stream[0];
    cudaMemcpyAsync(G->in0[0], x, (size_t)T * IK_D * sizeof(float), cudaMemcpyHostToDevice, st);
    if (T <= IK_MAX_LOGIT_T) {
        launch_quant_if_dp4a(G, 0, G->in0[0], IK_D, T, G->xq4[0], G->xd[0]);
        launch_q8g16(G, 0, st, &G->head_shard[0], T, G->in0[0], G->xq4[0], G->xd[0], G->d_logits[0], G->head_shard[0].rows);
    }
    return 0;
}

static void ik_gpu_head_wait(void* ctx, int T, float* logits) {
    GpuState* G = (GpuState*)ctx;
    if (G->head_shard[0].rows <= 0) return;
    cudaSetDevice(G->devs[0]);
    cudaStream_t st = G->stream[0];
    if (T <= IK_MAX_LOGIT_T) {
        copy_back_rows(st, G->d_logits[0], G->h_pin_logits, T,
                       G->head_shard[0].rows, logits, G->head_shard[0].split_off, IK_V_UNPAD);
    } else {
        for (int tc0 = 0; tc0 < T; tc0 += IK_MAX_LOGIT_T) {
            int nt = T - tc0; if (nt > IK_MAX_LOGIT_T) nt = IK_MAX_LOGIT_T;
            const float* xc = G->in0[0] + (size_t)tc0 * IK_D;
            launch_quant_if_dp4a(G, 0, xc, IK_D, nt, G->xq4[0], G->xd[0]);
            launch_q8g16(G, 0, st, &G->head_shard[0], nt, xc, G->xq4[0], G->xd[0], G->d_logits[0], G->head_shard[0].rows);
            copy_back_rows(st, G->d_logits[0], G->h_pin_logits, nt,
                           G->head_shard[0].rows, logits + (size_t)tc0 * IK_V_UNPAD, G->head_shard[0].split_off, IK_V_UNPAD);
        }
    }
}

static void ik_gpu_reset(void* ctx) {
    GpuState* G = (GpuState*)ctx;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        int d = G->layer_dev[lphys(il)];
        cudaSetDevice(G->devs[d]);
        if (G->d_conv_hist[il]) CKV(cudaMemsetAsync(G->d_conv_hist[il], 0, (size_t)3 * IK_CHSCONV * sizeof(float), G->stream[d]));
        if (G->kv_k[il]) {
            size_t kvb = (size_t)G->kv_cap[il] * IK_GQADIM * sizeof(uint16_t);
            CKV(cudaMemsetAsync(G->kv_k[il], 0, kvb, G->stream[d]));
            CKV(cudaMemsetAsync(G->kv_v[il], 0, kvb, G->stream[d]));
        }
    }
    for (int d = 0; d < G->n_dev; d++) { cudaSetDevice(G->devs[d]); cudaStreamSynchronize(G->stream[d]); }
}

static void ik_gpu_on_roll(void* ctx, int pos) {
    (void)pos;
    GpuState* G = (GpuState*)ctx;
    if (!G->cpu_conv_hist_ptr) return;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        if (!G->d_conv_hist[il]) continue;
        int d = G->layer_dev[lphys(il)];
        cudaSetDevice(G->devs[d]);
        const float* src = G->cpu_conv_hist_ptr + (size_t)il * 3 * IK_CHSCONV;
        CKV(cudaMemcpyAsync(G->d_conv_hist[il], src, (size_t)3 * IK_CHSCONV * sizeof(float), cudaMemcpyHostToDevice, G->stream[d]));
    }
    for (int d = 0; d < G->n_dev; d++) { cudaSetDevice(G->devs[d]); cudaStreamSynchronize(G->stream[d]); }
}

void ik_gpu_free(void* ctx);

static void mat_slab_bytes(int rows, int cols, int ty, size_t* out_sc, size_t* out_qs) {
    if (ty == TY_NVFP4) {
        *out_sc = (size_t)(rows / 16) * (cols / QK_NV) * 64u;
        *out_qs = (size_t)(rows / 16) * (cols / QK_NV) * 512u;
    } else if (ty == TY_IQ2XXS_128) {
        *out_sc = (size_t)(rows / 16) * (cols / QK_IQ2) * 32u;
        *out_qs = (size_t)(rows / 16) * (cols / QK_IQ2) * 512u;
    } else if (ty == TY_IQ2XS_128) {
        *out_sc = (size_t)(rows / 16) * (cols / QK_IQ2) * 96u;
        *out_qs = (size_t)(rows / 16) * (cols / QK_IQ2) * 512u;
    } else {
        *out_sc = 0; *out_qs = 0;
    }
}

static size_t raw_exp_bytes(int rows, int cols, int ty) {
    if (ty == TY_NVFP4)      return (size_t)rows * (cols / QK_NV) * BNV;
    if (ty == TY_IQ2XXS_128) return (size_t)rows * (cols / QK_IQ2) * BIQ2_XXS;
    if (ty == TY_IQ2XS_128)  return (size_t)rows * (cols / QK_IQ2) * BIQ2_XS;
    if (ty == TY_Q5K)        return (size_t)rows * (cols / QK_K) * BQ5K;
    return (size_t)rows * (cols / QK_Q8) * BQ8;
}

void* ik_gpu_init(const char* gguf_path, size_t data_start,
                  const uint64_t* g_offs, const int* g_tys,
                  const uint64_t* l_offs, const int* l_tys,
                  int ctx, int mode, int n_dev, const int* devs, const int* layer_dev,
                  const uint8_t* exp_plan, int n_dcache,
                  int split_q, int split_k, int split_v, int split_g, int split_o, int split_head,
                  float* cpu_conv_hist, ik_cpu_moe_fn cpu_moe_fn,
                  const char* const* ptx_per_dev, IkGpuHooks* out_hooks) {
    if (load_cuda_driver_api() != 0) return NULL;
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { perror("ik_gpu_init open"); return NULL; }
    struct stat st_f;
    if (fstat(fd, &st_f) != 0) { close(fd); return NULL; }
    uint8_t* mmap_ptr = (uint8_t*)mmap(NULL, (size_t)st_f.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ptr == MAP_FAILED) { close(fd); return NULL; }
    const uint8_t* base = mmap_ptr + data_start;

    GpuState* G = (GpuState*)calloc(1, sizeof(GpuState));
    G->mode = mode;
    G->n_dev = n_dev;
    G->ctx_cap = ctx;
    G->cpu_conv_hist_ptr = cpu_conv_hist;
    G->cpu_moe_fn = cpu_moe_fn;
    G->split_q = split_q; G->split_k = split_k; G->split_v = split_v;
    G->split_g = split_g; G->split_o = split_o; G->split_head = split_head;
    for (int i = 0; i < IK_NL_PHYS; i++) G->layer_dev[i] = layer_dev ? layer_dev[i] : 0;

    G->stage_cap = 4u << 20;
    for (int d = 0; d < n_dev; d++) {
        G->devs[d] = devs[d];
        cudaSetDevice(G->devs[d]);
        cudaFree(0); /* Force primary context initialization on device devs[d] */
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, G->devs[d]);
        G->cc[d] = prop.major * 10 + prop.minor;
        cudaStreamCreate(&G->stream[d]);
        cudaStreamCreate(&G->xfer_stream[d]);
        cudaEventCreateWithFlags(&G->ev_done[d], cudaEventDisableTiming);
        cudaMalloc((void**)&G->d_stage[d], G->stage_cap);

        CUresult cr = fn_cuModuleLoadData(&G->cu_mod[d], ptx_per_dev[d]);
        if (cr != CUDA_SUCCESS) {
            fprintf(stderr, "inkling2_gpu: cuModuleLoadData failed on dev %d (CUresult=%d)\n", G->devs[d], (int)cr);
            return NULL;
        }
        for (int k = 0; k < FN_COUNT; k++) {
            cr = fn_cuModuleGetFunction(&G->fn[d][k], G->cu_mod[d], k_kernel_names[k]);
            if (cr == CUDA_SUCCESS) kl_trace_name(G->fn[d][k], k_kernel_names[k]);
            if (cr != CUDA_SUCCESS) {
                fprintf(stderr, "inkling2_gpu: cuModuleGetFunction(%s) failed (CUresult=%d)\n", k_kernel_names[k], (int)cr);
                return NULL;
            }
        }
    }

    G->mmap_fd = fd;
    G->mmap_ptr = mmap_ptr;
    G->mmap_sz = (size_t)st_f.st_size;
    G->cpu_tok_embd = base + g_offs[0];
    float* tnorm = (float*)malloc(IK_D * sizeof(float));
    memcpy(tnorm, base + g_offs[1], IK_D * sizeof(float));
    G->cpu_tok_norm = tnorm;

    cudaHostAlloc((void**)&G->h_pin_f, (size_t)IK_MAXB * IK_NU * IK_D * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_exps, (size_t)IK_MAXB * IK_NU * IK_D * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_logits, (size_t)n_dev * IK_MAX_LOGIT_T * IK_V_UNPAD * sizeof(float), cudaHostAllocDefault);
    if (mode == 0) {
        cudaHostAlloc((void**)&G->h_pin_qkv, (size_t)IK_MAXB * (2 * IK_QDIM + 2 * IK_GQADIM) * sizeof(float), cudaHostAllocDefault);
    }
    cudaHostAlloc((void**)&G->h_pin_sel, (size_t)IK_MAXB * IK_NU * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_wsel, (size_t)IK_MAXB * IK_NU * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_uniq_n, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_moe_meta, (size_t)(2 * IK_NE + 1 + 2 * IK_MAXB * IK_NU) * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_cold_flag, 1, cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_gpu_mask, (size_t)IK_MAXB * IK_NU, cudaHostAllocDefault);
    cudaHostAlloc((void**)&G->h_pin_top2, (size_t)n_dev * IK_MAX_LOGIT_T * sizeof(IkTop2), cudaHostAllocDefault);

    for (int d = 0; d < n_dev; d++) {
        cudaSetDevice(G->devs[d]);
        cudaMalloc((void**)&G->cur[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->pre[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->bank_out[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->msg[d], (size_t)IK_MAXB * 4 * IK_D * 4);
        for (int b = 0; b < 4; b++) {
            cudaMalloc((void**)&G->blk[d][b], (size_t)IK_MAXB * IK_D * 4);
            cudaMalloc((void**)&G->blk_pre[d][b], (size_t)IK_MAXB * IK_D * 4);
            cudaMalloc((void**)&G->bank[d][b], (size_t)IK_MAXB * IK_D * 4);
        }
        cudaMalloc((void**)&G->in0[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->in1[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->in2[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->tmp3[d], (size_t)IK_MAXB * 3 * IK_D * 4);
        cudaMalloc((void**)&G->inpsa[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->ffninp[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->q[d], (size_t)IK_MAXB * IK_QDIM * 4);
        cudaMalloc((void**)&G->k[d], (size_t)IK_MAXB * IK_GQADIM * 4);
        cudaMalloc((void**)&G->v[d], (size_t)IK_MAXB * IK_GQADIM * 4);
        cudaMalloc((void**)&G->g[d], (size_t)IK_MAXB * IK_QDIM * 4);
        cudaMalloc((void**)&G->r[d], (size_t)IK_MAXB * IK_RDIM * 4);
        cudaMalloc((void**)&G->sinks[d], (size_t)IK_MAXB * IK_NH * 4);
        cudaMalloc((void**)&G->kcp[d], (size_t)IK_MAXB * 4 * IK_NKV * 4);
        cudaMalloc((void**)&G->vcp[d], (size_t)IK_MAXB * 4 * IK_NKV * 4);
        cudaMalloc((void**)&G->attn[d], (size_t)IK_MAXB * IK_QDIM * 4);
        cudaMalloc((void**)&G->o[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->ffnin[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->probs[d], (size_t)IK_MAXB * IK_NE * 4);
        cudaMalloc((void**)&G->exps_out[d], (size_t)IK_MAXB * IK_NU * IK_D * 4);
        cudaMalloc((void**)&G->moe_h[d], (size_t)IK_MAXB * IK_NU * IK_FF * 4);
        cudaMalloc((void**)&G->ctrl_in[d], (size_t)IK_MAXB * 2 * IK_D * 4);
        cudaMalloc((void**)&G->ctrl_g[d], (size_t)IK_MAXB * IK_GRANK * 4);
        cudaMalloc((void**)&G->ctrl_gn[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->ctrl_fg[d], (size_t)IK_MAXB * IK_D * 4);
        cudaMalloc((void**)&G->xq4[d], (size_t)IK_MAXB * ((2 * IK_D) / 4) * 4);
        cudaMalloc((void**)&G->xq4_b[d], (size_t)IK_MAXB * (IK_D / 4) * 4);
        cudaMalloc((void**)&G->xq4_c[d], (size_t)IK_MAXB * (IK_D / 4) * 4);
        cudaMalloc((void**)&G->hq4[d], (size_t)IK_MAXB * IK_NU * (IK_FF / 4) * 4);
        cudaMalloc((void**)&G->xd[d], (size_t)IK_MAXB * ((2 * IK_D) / QK_Q8) * 4);
        cudaMalloc((void**)&G->xd_b[d], (size_t)IK_MAXB * (IK_D / QK_Q8) * 4);
        cudaMalloc((void**)&G->xd_c[d], (size_t)IK_MAXB * (IK_D / QK_Q8) * 4);
        cudaMalloc((void**)&G->hd[d], (size_t)IK_MAXB * IK_NU * (IK_FF / QK_Q8) * 4);
        cudaMalloc((void**)&G->d_sel[d], (size_t)IK_MAXB * IK_NU * sizeof(int));
        cudaMalloc((void**)&G->d_uniq_e[d], IK_NE * sizeof(int));
        cudaMalloc((void**)&G->d_u_off[d], (IK_NE + 1) * sizeof(int));
        cudaMalloc((void**)&G->d_tok_idx[d], (size_t)IK_MAXB * IK_NU * sizeof(int));
        cudaMalloc((void**)&G->d_tok_slot[d], (size_t)IK_MAXB * IK_NU * sizeof(int));
        cudaMalloc((void**)&G->d_uniq_n[d], sizeof(int));
        cudaMalloc((void**)&G->d_wsel[d], (size_t)IK_MAXB * IK_NU * sizeof(float));
        cudaMalloc((void**)&G->d_has_cold[d], 1);
        cudaMalloc((void**)&G->d_gpu_mask[d], (size_t)IK_MAXB * IK_NU);
        cudaMalloc((void**)&G->d_act_gu_sc[d], IK_NE * sizeof(void*));
        cudaMalloc((void**)&G->d_act_gu_qs[d], IK_NE * sizeof(void*));
        cudaMalloc((void**)&G->d_act_dn_sc[d], IK_NE * sizeof(void*));
        cudaMalloc((void**)&G->d_act_dn_qs[d], IK_NE * sizeof(void*));
        cudaMalloc((void**)&G->d_top2_blk[d], 128 * sizeof(IkTop2));
        cudaMalloc((void**)&G->d_top2_res[d], (size_t)IK_MAX_LOGIT_T * sizeof(IkTop2));
    }

    int n_cold_nv = 0, n_cold_k20 = 0;
    if (n_dcache > 0) {
        for (int ip = 0; ip < IK_NL_PHYS; ip++) {
            int is_nv = (l_tys[ip * 42 + 25] == TY_NVFP4);
            for (int e = 0; e < IK_NE; e++) {
                if (exp_plan[ip * IK_NE + e] == 1) {
                    if (is_nv) n_cold_nv++;
                    else n_cold_k20++;
                }
            }
        }
        G->use_dcache = 1;
        G->n_dcache = n_dcache;
        G->dcache = (DcacheSlot*)calloc(n_dcache, sizeof(DcacheSlot));
        size_t gu_sc_sz = (size_t)(IK_FF2 / 16) * (IK_D / QK_NV) * 64;
        size_t gu_qs_sz = (size_t)(IK_FF2 / 16) * (IK_D / QK_NV) * 512;
        size_t dn_sc_sz = (size_t)(IK_D / 16) * (IK_FF / QK_NV) * 64;
        size_t dn_qs_sz = (size_t)(IK_D / 16) * (IK_FF / QK_NV) * 512;
        size_t exp_slot_bytes = gu_sc_sz + gu_qs_sz + dn_sc_sz + dn_qs_sz;
        cudaSetDevice(G->devs[0]);
        if (cudaMalloc((void**)&G->d_dcache_slab, (size_t)n_dcache * exp_slot_bytes) != cudaSuccess) {
            fprintf(stderr, "inkling2_gpu: dcache slab OOM (%d slots = %.1f MiB)\n",
                    n_dcache, ((size_t)n_dcache * exp_slot_bytes) / 1048576.0);
            ik_gpu_free(G);
            return NULL;
        }
        for (int c = 0; c < n_dcache; c++) {
            uint8_t* sp = G->d_dcache_slab + (size_t)c * exp_slot_bytes;
            G->dcache[c].cur_ip = -1; G->dcache[c].cur_e = -1;
            G->dcache[c].gu_sc = sp;
            G->dcache[c].gu_qs = sp + gu_sc_sz;
            G->dcache[c].dn_sc = sp + gu_sc_sz + gu_qs_sz;
            G->dcache[c].dn_qs = sp + gu_sc_sz + gu_qs_sz + dn_sc_sz;
        }
        if (n_cold_nv > 0) {
            cudaHostAlloc(&G->h_cold_mapped_slab, (size_t)n_cold_nv * exp_slot_bytes, cudaHostAllocMapped);
        }
        if (n_cold_k20 > 0) {
            size_t k20_gu_b = (size_t)IK_FF2 * (IK_D / QK_K) * BQ5K;
            size_t k20_sc_b = (size_t)(IK_D / 16) * (IK_FF / QK_Q8) * 16 * 2;
            size_t k20_qs_b = (size_t)(IK_D / 16) * (IK_FF / QK_Q8) * 512;
            cudaHostAlloc(&G->h_cold_k20_slab, (size_t)n_cold_k20 * (k20_gu_b + k20_sc_b + k20_qs_b), cudaHostAllocMapped);
        }
    }

    size_t slab_cur = 0, k20_cur = 0;
    size_t k20_gu_b = (size_t)IK_FF2 * (IK_D / QK_K) * BQ5K;
    size_t k20_sc_b = (size_t)(IK_D / 16) * (IK_FF / QK_Q8) * 16 * 2;
    size_t k20_qs_b = (size_t)(IK_D / 16) * (IK_FF / QK_Q8) * 512;

    size_t stat_bytes[IK_MAX_DEV] = {0};
    for (int ip = 0; ip < IK_NL_PHYS; ip++) {
        size_t gsc, gqs, dsc, dqs;
        mat_slab_bytes(IK_FF2, IK_D, l_tys[ip * 42 + 25], &gsc, &gqs);
        mat_slab_bytes(IK_D, IK_FF, l_tys[ip * 42 + 27], &dsc, &dqs);
        size_t per_e = gsc + gqs + dsc + dqs;
        if (per_e > 0) {
            int d = G->layer_dev[ip];
            for (int e = 0; e < IK_NE; e++) {
                if (exp_plan[ip * IK_NE + e] == 2) stat_bytes[d] += per_e;
            }
        }
    }
    size_t stat_off[IK_MAX_DEV] = {0};
    for (int d = 0; d < n_dev; d++) {
        if (stat_bytes[d] > 0) {
            cudaSetDevice(G->devs[d]);
            cudaError_t se = cudaMalloc((void**)&G->d_static_slab[d], stat_bytes[d]);
            if (se != cudaSuccess) {
                size_t f_b = 0, t_b = 0;
                cudaMemGetInfo(&f_b, &t_b);
                fprintf(stderr, "inkling2_gpu: static slab OOM on dev %d (need %.1f MiB, free %.1f / %.1f MiB): %s\n",
                        G->devs[d], stat_bytes[d] / 1048576.0,
                        f_b / 1048576.0, t_b / 1048576.0, cudaGetErrorString(se));
                ik_gpu_free(G);
                return NULL;
            }
        }
    }

    for (int ip = 0; ip < IK_NL_PHYS; ip++) {
        GpuPhysLayer* gl = &G->L[ip];
        int d = G->layer_dev[ip];
        cudaSetDevice(G->devs[d]);
        cudaStream_t st = G->stream[d];
        const uint64_t* lo = l_offs + ip * 42;
        const int* lt = l_tys + ip * 42;

        if (mode == 0) {
            if (ip >= IK_CORE_START && ip < IK_CORE_END) {
                if (split_q < IK_QDIM)   upload_dense_slice(G, d, base + lo[0], lt[0], split_q, IK_QDIM - split_q, IK_D, &gl->wq);
                if (split_k < IK_GQADIM) upload_dense_slice(G, d, base + lo[1], lt[1], split_k, IK_GQADIM - split_k, IK_D, &gl->wk);
                if (split_v < IK_GQADIM) upload_dense_slice(G, d, base + lo[2], lt[2], split_v, IK_GQADIM - split_v, IK_D, &gl->wv);
                if (split_o < IK_D)      upload_dense_slice(G, d, base + lo[3], lt[3], split_o, IK_D - split_o, IK_QDIM, &gl->wo);
                if (split_g < IK_QDIM)   upload_dense_slice(G, d, base + lo[4], lt[4], split_g, IK_QDIM - split_g, IK_D, &gl->wgate);
            }
        } else {
            upload_dense_slice(G, d, base + lo[0], lt[0], 0, IK_QDIM,   IK_D,    &gl->wq);
            upload_dense_slice(G, d, base + lo[1], lt[1], 0, IK_GQADIM, IK_D,    &gl->wk);
            upload_dense_slice(G, d, base + lo[2], lt[2], 0, IK_GQADIM, IK_D,    &gl->wv);
            upload_dense_slice(G, d, base + lo[3], lt[3], 0, IK_D,      IK_QDIM, &gl->wo);
            upload_dense_slice(G, d, base + lo[4], lt[4], 0, IK_QDIM,   IK_D,    &gl->wgate);
            upload_dense_slice(G, d, base + lo[5], lt[5], 0, IK_RDIM,   IK_D,    &gl->wrel);

            upload_raw_bf16(G->devs[d], st, base + lo[6], IK_D * IK_NH, &gl->sink_proj);
            upload_f32_or_bf16(G->devs[d], st, base + lo[7], lt[7], IK_HD * IK_NH, &gl->q_clip);
            upload_f32_or_bf16(G->devs[d], st, base + lo[8], lt[8], IK_NREL * IK_RELEXT, &gl->rel_bias);
            upload_f32_or_bf16(G->devs[d], st, base + lo[9], lt[9], IK_NREL * IK_NH, &gl->rel_clip);
            upload_f32_or_bf16(G->devs[d], st, base + lo[10], lt[10], 4 * IK_GQADIM, &gl->k_conv);
            upload_f32_or_bf16(G->devs[d], st, base + lo[11], lt[11], 4 * IK_GQADIM, &gl->v_conv);
            upload_f32_or_bf16(G->devs[d], st, base + lo[12], lt[12], 4 * IK_D, &gl->attn_post_conv);
            upload_f32_or_bf16(G->devs[d], st, base + lo[13], lt[13], 4 * IK_D, &gl->ffn_post_conv);
            upload_f32_or_bf16(G->devs[d], st, lt[14] >= 0 ? base + lo[14] : NULL, lt[14], IK_D, &gl->attn_norm);
            upload_f32_or_bf16(G->devs[d], st, lt[15] >= 0 ? base + lo[15] : NULL, lt[15], IK_D, &gl->q_inp_norm);
            upload_f32_or_bf16(G->devs[d], st, lt[16] >= 0 ? base + lo[16] : NULL, lt[16], IK_D, &gl->k_inp_norm);
            upload_f32_or_bf16(G->devs[d], st, lt[17] >= 0 ? base + lo[17] : NULL, lt[17], IK_D, &gl->v_inp_norm);
            if (lt[18] >= 0) upload_raw_bf16(G->devs[d], st, base + lo[18], IK_D * 4 * IK_NKV, &gl->k_conv_proj);
            if (lt[19] >= 0) upload_raw_bf16(G->devs[d], st, base + lo[19], IK_D * 4 * IK_NKV, &gl->v_conv_proj);
            int rq_n = ((ip % IK_BLOCK_SZ) != 5) ? IK_D : 3 * IK_D;
            upload_f32_or_bf16(G->devs[d], st, lt[20] >= 0 ? base + lo[20] : NULL, lt[20], rq_n, &gl->attn_res_q);
            upload_f32_or_bf16(G->devs[d], st, lt[21] >= 0 ? base + lo[21] : NULL, lt[21], IK_D, &gl->ffn_res_q);
            upload_f32_or_bf16(G->devs[d], st, base + lo[22], lt[22], IK_D, &gl->ffn_norm);
            upload_f32_or_bf16(G->devs[d], st, base + lo[23], lt[23], IK_NE * IK_D, &gl->ffn_gate_inp);
            upload_f32_or_bf16(G->devs[d], st, base + lo[24], lt[24], IK_NE, &gl->exp_bias);
            if (ip >= IK_CORE_START && ip < IK_CORE_END) {
                upload_f32_or_bf16(G->devs[d], st, base + lo[29], lt[29], 4 * IK_D, &gl->loop_msg_q);
                int di[4] = {30, 33, 36, 39}, gi[4] = {31, 34, 37, 40}, fi[4] = {32, 35, 38, 41};
                gl->loop_is_q8 = (lt[di[0]] == TY_Q8);
                for (int c = 0; c < 4; c++) {
                    if (lt[di[c]] == TY_Q8) {
                        upload_dense_slice(G, d, base + lo[di[c]], TY_Q8, 0, IK_GRANK, 2 * IK_D, &gl->loop_down_q8[c]);
                        upload_dense_slice(G, d, base + lo[gi[c]], TY_Q8, 0, IK_D, IK_GRANK, &gl->loop_gain_q8[c]);
                        upload_dense_slice(G, d, base + lo[fi[c]], TY_Q8, 0, IK_D, IK_GRANK, &gl->loop_forget_q8[c]);
                    } else {
                        upload_f32_or_bf16(G->devs[d], st, base + lo[di[c]], lt[di[c]], (size_t)IK_GRANK * 2 * IK_D, &gl->loop_down[c]);
                        upload_f32_or_bf16(G->devs[d], st, base + lo[gi[c]], lt[gi[c]], (size_t)IK_D * IK_GRANK, &gl->loop_gain[c]);
                        upload_f32_or_bf16(G->devs[d], st, base + lo[fi[c]], lt[fi[c]], (size_t)IK_D * IK_GRANK, &gl->loop_forget[c]);
                    }
                }
            }
        }

        gl->exp_gu_type = lt[25]; gl->exp_dn_type = lt[27];
        gl->has_exp_scale = (gl->exp_gu_type == TY_NVFP4 && lt[26] >= 0);
        if (gl->has_exp_scale) {
            upload_f32_or_bf16(G->devs[d], st, base + lo[26], TY_F32, IK_NE, &gl->d_exp_gu_scale);
            upload_f32_or_bf16(G->devs[d], st, base + lo[28], TY_F32, IK_NE, &gl->d_exp_dn_scale);
        }

        size_t gu_sc_sz, gu_qs_sz, dn_sc_sz, dn_qs_sz;
        mat_slab_bytes(IK_FF2, IK_D, gl->exp_gu_type, &gu_sc_sz, &gu_qs_sz);
        mat_slab_bytes(IK_D, IK_FF, gl->exp_dn_type, &dn_sc_sz, &dn_qs_sz);
        size_t gu_eb = raw_exp_bytes(IK_FF2, IK_D, gl->exp_gu_type);
        size_t dn_eb = raw_exp_bytes(IK_D, IK_FF, gl->exp_dn_type);
        const uint8_t* gu_raw0 = base + lo[25];
        const uint8_t* dn_raw0 = base + lo[27];

        for (int e = 0; e < IK_NE; e++) {
            uint8_t plan_v = exp_plan[ip * IK_NE + e];
            if (plan_v == 2) {
                gl->on_dev[e] = 1;
                if (gu_sc_sz > 0) {
                    if (G->d_static_slab[d]) {
                        uint8_t* sp = G->d_static_slab[d] + stat_off[d];
                        stat_off[d] += gu_sc_sz + gu_qs_sz + dn_sc_sz + dn_qs_sz;
                        gl->d_exp_gu_sc[e] = sp;
                        gl->d_exp_gu_qs[e] = sp + gu_sc_sz;
                        gl->d_exp_dn_sc[e] = sp + gu_sc_sz + gu_qs_sz;
                        gl->d_exp_dn_qs[e] = sp + gu_sc_sz + gu_qs_sz + dn_sc_sz;
                    }
                    if (gl->exp_gu_type == TY_NVFP4)
                        upload_nvfp4_g16(G, d, gu_raw0 + (size_t)e * gu_eb, IK_FF2, IK_D, (uint8_t**)&gl->d_exp_gu_sc[e], (uint8_t**)&gl->d_exp_gu_qs[e]);
                    else
                        upload_iq2_g16(G, d, gu_raw0 + (size_t)e * gu_eb, gl->exp_gu_type, IK_FF2, IK_D, (uint8_t**)&gl->d_exp_gu_sc[e], (uint8_t**)&gl->d_exp_gu_qs[e]);
                    if (gl->exp_dn_type == TY_NVFP4)
                        upload_nvfp4_g16(G, d, dn_raw0 + (size_t)e * dn_eb, IK_D, IK_FF, (uint8_t**)&gl->d_exp_dn_sc[e], (uint8_t**)&gl->d_exp_dn_qs[e]);
                    else
                        upload_iq2_g16(G, d, dn_raw0 + (size_t)e * dn_eb, gl->exp_dn_type, IK_D, IK_FF, (uint8_t**)&gl->d_exp_dn_sc[e], (uint8_t**)&gl->d_exp_dn_qs[e]);
                } else {
                    cudaMalloc((void**)&gl->d_exp_gu_sc[e], gu_eb);
                    cudaMemcpy((void*)gl->d_exp_gu_sc[e], gu_raw0 + (size_t)e * gu_eb, gu_eb, cudaMemcpyHostToDevice);
                    GpuQ8Mat tmpq;
                    upload_dense_slice(G, d, dn_raw0 + (size_t)e * dn_eb, gl->exp_dn_type, 0, IK_D, IK_FF, &tmpq);
                    gl->d_exp_dn_sc[e] = (const uint8_t*)tmpq.sc;
                    gl->d_exp_dn_qs[e] = (const uint8_t*)tmpq.qs;
                }
            } else if (plan_v == 1 && G->use_dcache && gl->exp_gu_type == TY_NVFP4) {
                uint8_t* p = (uint8_t*)G->h_cold_mapped_slab + slab_cur;
                slab_cur += gu_sc_sz + gu_qs_sz + dn_sc_sz + dn_qs_sz;
                uint8_t *hs = p, *hq = p + gu_sc_sz, *ds = hq + gu_qs_sz, *dq = ds + dn_sc_sz;
                dcache_prepare_cold_nvfp4(G, d, gu_raw0 + (size_t)e * gu_eb, IK_FF2, IK_D, hs, hq);
                dcache_prepare_cold_nvfp4(G, d, dn_raw0 + (size_t)e * dn_eb, IK_D, IK_FF, ds, dq);
                G->cold_h_gu_sc[ip][e] = hs; G->cold_h_gu_qs[ip][e] = hq;
                G->cold_h_dn_sc[ip][e] = ds; G->cold_h_dn_qs[ip][e] = dq;
                cudaHostGetDevicePointer((void**)&G->cold_d_gu_sc[ip][e], hs, 0);
                cudaHostGetDevicePointer((void**)&G->cold_d_gu_qs[ip][e], hq, 0);
                cudaHostGetDevicePointer((void**)&G->cold_d_dn_sc[ip][e], ds, 0);
                cudaHostGetDevicePointer((void**)&G->cold_d_dn_qs[ip][e], dq, 0);
            } else if (plan_v == 1 && G->use_dcache && gl->exp_gu_type != TY_NVFP4) {
                uint8_t* p = (uint8_t*)G->h_cold_k20_slab + k20_cur;
                k20_cur += k20_gu_b + k20_sc_b + k20_qs_b;
                uint8_t* hg = p;
                uint16_t* hds = (uint16_t*)(p + k20_gu_b);
                int8_t* hdq = (int8_t*)(p + k20_gu_b + k20_sc_b);
                memcpy(hg, gu_raw0 + (size_t)e * gu_eb, k20_gu_b);
                dcache_prepare_cold_q8g16(dn_raw0 + (size_t)e * dn_eb, IK_D, IK_FF, hds, hdq);
                cudaHostGetDevicePointer((void**)&gl->d_exp_gu_sc[e], hg, 0);
                cudaHostGetDevicePointer((void**)&gl->d_exp_dn_sc[e], hds, 0);
                cudaHostGetDevicePointer((void**)&gl->d_exp_dn_qs[e], hdq, 0);
            }
        }
        cudaMalloc((void**)&gl->d_ptrs_gu_sc, IK_NE * sizeof(void*));
        cudaMalloc((void**)&gl->d_ptrs_gu_qs, IK_NE * sizeof(void*));
        cudaMalloc((void**)&gl->d_ptrs_dn_sc, IK_NE * sizeof(void*));
        cudaMalloc((void**)&gl->d_ptrs_dn_qs, IK_NE * sizeof(void*));
        cudaMemcpy(gl->d_ptrs_gu_sc, gl->d_exp_gu_sc, IK_NE * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(gl->d_ptrs_gu_qs, gl->d_exp_gu_qs, IK_NE * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(gl->d_ptrs_dn_sc, gl->d_exp_dn_sc, IK_NE * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(gl->d_ptrs_dn_qs, gl->d_exp_dn_qs, IK_NE * sizeof(void*), cudaMemcpyHostToDevice);
        if ((ip & 1) == 1) madvise(G->mmap_ptr, G->mmap_sz, MADV_DONTNEED);
    }

    if (mode == 0) {
        if (split_head < IK_V_UNPAD) {
            upload_dense_slice(G, 0, base + g_offs[2], g_tys[2], split_head, IK_V_UNPAD - split_head, IK_D, &G->head_shard[0]);
            cudaSetDevice(G->devs[0]);
            cudaMalloc((void**)&G->d_logits[0], (size_t)IK_MAX_LOGIT_T * G->head_shard[0].rows * sizeof(float));
        }
        if (G->d_stage[0]) { cudaSetDevice(G->devs[0]); cudaFree(G->d_stage[0]); G->d_stage[0] = NULL; }
    } else {
        G->head_dev_pos = G->layer_dev[IK_NL_PHYS - 1];
        cudaSetDevice(G->devs[G->head_dev_pos]);
        upload_f32_or_bf16(G->devs[G->head_dev_pos], G->stream[G->head_dev_pos], base + g_offs[3], TY_F32, IK_D, &G->d_output_norm);
        upload_f32_or_bf16(G->devs[G->head_dev_pos], G->stream[G->head_dev_pos], base + g_offs[4], TY_F32, IK_D, &G->d_output_res_q);

        if (n_dev == 1) {
            upload_dense_slice(G, 0, base + g_offs[2], g_tys[2], 0, IK_V_UNPAD, IK_D, &G->head_shard[0]);
            cudaSetDevice(G->devs[0]);
            cudaMalloc((void**)&G->d_logits[0], (size_t)IK_MAX_LOGIT_T * IK_V_UNPAD * sizeof(float));
        } else {
            int mid = ((IK_V_UNPAD / n_dev) + 15) & ~15;
            int off = 0;
            for (int d = 0; d < n_dev; d++) {
                int len = (d == n_dev - 1) ? (IK_V_UNPAD - off) : mid;
                upload_dense_slice(G, d, base + g_offs[2], g_tys[2], off, len, IK_D, &G->head_shard[d]);
                cudaSetDevice(G->devs[d]);
                cudaMalloc((void**)&G->d_logits[d], (size_t)IK_MAX_LOGIT_T * len * sizeof(float));
                off += len;
            }
        }

        for (int d = 0; d < n_dev; d++) {
            if (G->d_stage[d]) { cudaSetDevice(G->devs[d]); cudaFree(G->d_stage[d]); G->d_stage[d] = NULL; }
        }

        for (int il = 0; il < IK_NL_VIRT; il++) {
            int ip = lphys(il);
            int d = G->layer_dev[ip];
            int is_swa = ((ip % IK_BLOCK_SZ) != 5);
            int cap = is_swa ? (IK_SWA + IK_MAXB) : ctx;
            G->kv_cap[il] = cap;
            cudaSetDevice(G->devs[d]);
            size_t kvb = (size_t)cap * IK_GQADIM * sizeof(uint16_t);
            cudaMalloc((void**)&G->kv_k[il], kvb);
            cudaMalloc((void**)&G->kv_v[il], kvb);
            cudaMemset(G->kv_k[il], 0, kvb);
            cudaMemset(G->kv_v[il], 0, kvb);
            cudaMalloc((void**)&G->d_conv_hist[il], (size_t)3 * IK_CHSCONV * sizeof(float));
            cudaMemset(G->d_conv_hist[il], 0, (size_t)3 * IK_CHSCONV * sizeof(float));
        }
    }

    for (int d = 0; d < n_dev; d++) {
        cudaSetDevice(G->devs[d]);
        cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            size_t f_b = 0, t_b = 0;
            cudaMemGetInfo(&f_b, &t_b);
            fprintf(stderr, "inkling2_gpu: init failed on dev %d (free %.1f / %.1f MiB): %s\n",
                    G->devs[d], f_b / 1048576.0, t_b / 1048576.0, cudaGetErrorString(ce));
            ik_gpu_free(G);
            return NULL;
        }
        cudaDeviceSynchronize();
    }
    madvise(G->mmap_ptr, G->mmap_sz, MADV_DONTNEED);

    memset(out_hooks, 0, sizeof(*out_hooks));
    out_hooks->ctx = G;
    out_hooks->reset = ik_gpu_reset;
    out_hooks->on_roll = ik_gpu_on_roll;
    if (mode == 0) {
        out_hooks->qkvg_start = ik_gpu_qkvg_start;
        out_hooks->qkvg_wait  = ik_gpu_qkvg_wait;
        out_hooks->o_start    = ik_gpu_o_start;
        out_hooks->o_wait     = ik_gpu_o_wait;
        out_hooks->moe_start  = ik_gpu_moe_start;
        out_hooks->moe_wait   = ik_gpu_moe_wait;
        out_hooks->head_start = ik_gpu_head_start;
        out_hooks->head_wait  = ik_gpu_head_wait;
        out_hooks->split_q    = split_q;
        out_hooks->split_k    = split_k;
        out_hooks->split_v    = split_v;
        out_hooks->split_g    = split_g;
        out_hooks->split_o    = split_o;
        out_hooks->split_head = split_head;
    } else {
        out_hooks->forward_step = ik_gpu_forward_step;
    }
    return G;
}

void ik_gpu_free(void* ctx) {
    if (!ctx) return;
    GpuState* G = (GpuState*)ctx;
    free(G->h_state);
    G->h_state = NULL;
    if (G->d_dcache_slab) {
        cudaSetDevice(G->devs[0]);
        cudaFree(G->d_dcache_slab);
    }
    for (int il = 0; il < IK_NL_VIRT; il++) {
        if (G->d_conv_trace[il]) {
            cudaSetDevice(G->devs[G->layer_dev[lphys(il)]]);
            cudaFree(G->d_conv_trace[il]);
        }
    }
    for (int d = 0; d < G->n_dev; d++) {
        cudaSetDevice(G->devs[d]);
        if (G->d_stage[d]) cudaFree(G->d_stage[d]);
        if (G->d_static_slab[d]) cudaFree(G->d_static_slab[d]);
        if (G->cu_mod[d] && fn_cuModuleUnload) fn_cuModuleUnload(G->cu_mod[d]);
    }
    if (G->mmap_ptr && G->mmap_ptr != MAP_FAILED) munmap(G->mmap_ptr, G->mmap_sz);
    if (G->mmap_fd > 0) close(G->mmap_fd);
    free((void*)G->cpu_tok_norm);
    if (G->h_cold_mapped_slab) cudaFreeHost(G->h_cold_mapped_slab);
    if (G->h_cold_k20_slab) cudaFreeHost(G->h_cold_k20_slab);
    if (G->h_pin_f) cudaFreeHost(G->h_pin_f);
    if (G->h_pin_exps) cudaFreeHost(G->h_pin_exps);
    if (G->h_pin_logits) cudaFreeHost(G->h_pin_logits);
    if (G->h_pin_qkv) cudaFreeHost(G->h_pin_qkv);
    if (G->h_pin_sel) cudaFreeHost(G->h_pin_sel);
    if (G->h_pin_wsel) cudaFreeHost(G->h_pin_wsel);
    if (G->h_pin_uniq_n) cudaFreeHost(G->h_pin_uniq_n);
    if (G->h_pin_moe_meta) cudaFreeHost(G->h_pin_moe_meta);
    if (G->h_pin_cold_flag) cudaFreeHost(G->h_pin_cold_flag);
    if (G->h_pin_gpu_mask) cudaFreeHost(G->h_pin_gpu_mask);
    if (G->h_pin_top2) cudaFreeHost(G->h_pin_top2);
    free(G->dcache);
    free(G);
}

#endif /* __CUDACC__ */
