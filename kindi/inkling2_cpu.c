/* Kindi Inkling-2-Mini (inkling2_loop) CPU & Heterogeneous Host Engine.
 *
 * Implements the 24-physical / 48-virtual-layer looped MoE + SWA/Full relative-bias + ShortConv architecture
 * bit-accurately against llama.cpp (b052049), with AVX2+F16C SIMD kernels (and portable non-AVX2 fallbacks),
 * core-pinned persistent pthread pool, mmap zero-duplicate weight loading, KV/conv snapshots, 4-token tiled
 * prefill/DSpark batching, fused L1-resident single-barrier T=1 MoE, parallel greedy lm_head, and async GPU
 * offload hooks for hetero_k420 / hybrid_multi / gpu_pinned / gpu_full.
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#if defined(__has_include)
#if __has_include(<os/os_sync_wait_on_address.h>)
#include <os/os_sync_wait_on_address.h>
#define IK_HAVE_OS_SYNC 1
#endif
#endif
#endif

/* IK_NO_SIMD builds the plain C kernels on any CPU (the reference the SIMD paths are tested against). */
#if !defined(IK_NO_SIMD) && defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#define IK_AVX2 1
#else
#define IK_AVX2 0
#endif

#if !defined(IK_NO_SIMD) && defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define IK_NEON 1
#else
#define IK_NEON 0
#endif

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

/* Fixed architectural dimensions for Inkling-2-Mini */
#define IK_NL_PHYS    24
#define IK_CORE_START 6
#define IK_CORE_END   18
#define IK_N_PASS     3
#define IK_NL_VIRT    48
#define IK_BLOCK_SZ   6
#define IK_D          2304
#define IK_FF         1152
#define IK_FF2        2304
#define IK_NH         24
#define IK_NKV        8
#define IK_HD         128
#define IK_QDIM       3072
#define IK_GQADIM     1024
#define IK_NREL       16
#define IK_RDIM       384
#define IK_RELEXT     1024
#define IK_SWA        1024
#define IK_NCONV      4
#define IK_CHSCONV    6656   /* 2*1024 + 2*2304 */
#define IK_NE         128
#define IK_NU         8
#define IK_GRANK      64
#define IK_V_UNPAD    200058
#define IK_KMAXT      128
/* max tokens per forward call; it is also the spare room of the sliding-window KV rings (IK_SWA + IK_MAXB slots).
 * AArch64 builds (CPU prompt reading on Apple silicon) take 512, so that an expert sees about 32 tokens per batch
 * instead of 8; Python reads it with ik_max_batch() and keeps 128 whenever a GPU path is active. */
#ifndef IK_MAXB              /* -DIK_MAXB=512 -DIK_MOE_TILE_ALL lets an x86 build test the AArch64 batching */
#if IK_NEON
#define IK_MAXB       512
#else
#define IK_MAXB       128
#endif
#endif
#define IK_LOGB       128    /* rows of the greedy/batch logits buffer (ik_forward_greedy needs one) */

#define IK_RMS_EPS    1e-6f
#define IK_EXP_SCALE  8.0f
#define IK_SITU_BETA  4.0f
#define IK_SITU_LBETA 25.0f
#define IK_QK_CLIP    22.627416610717773f
#define IK_REL_CLIP   8.0f
#define IK_LOG_FLOOR  8192.0f
#define IK_LOG_ALPHA  0.1f
#define IK_LOGIT_CAP  30.0f

#define TY_F32         0
#define TY_Q4_0        2
#define TY_Q8          8
#define TY_Q5K         13
#define TY_BF16        30
#define TY_NVFP4       40
#define TY_IQ2XXS_128  116
#define TY_IQ2XS_128   117

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

#include "inkling2_iq2_tables.h"

/* Per-layer tensor IDs in flat tensor table passed by Python */
enum {
    LT_ATTN_Q = 0, LT_ATTN_K, LT_ATTN_V, LT_ATTN_O, LT_ATTN_GATE, LT_ATTN_REL,
    LT_ATTN_SINK_PROJ, LT_ATTN_Q_CLIP, LT_ATTN_REL_BIAS, LT_ATTN_REL_CLIP,
    LT_ATTN_K_CONV, LT_ATTN_V_CONV, LT_ATTN_POST_CONV, LT_FFN_POST_CONV,
    LT_ATTN_NORM, LT_ATTN_Q_INP_NORM, LT_ATTN_K_INP_NORM, LT_ATTN_V_INP_NORM,
    LT_ATTN_K_CONV_PROJ, LT_ATTN_V_CONV_PROJ, LT_ATTN_RES_Q,
    LT_FFN_RES_Q, LT_FFN_NORM, LT_FFN_GATE_INP, LT_EXP_BIAS,
    LT_GATE_UP_EXPS, LT_GATE_UP_SCALE, LT_DOWN_EXPS, LT_DOWN_SCALE,
    LT_LOOP_MSG_Q,
    LT_LOOP_Q_D, LT_LOOP_Q_G, LT_LOOP_Q_F,
    LT_LOOP_K_D, LT_LOOP_K_G, LT_LOOP_K_F,
    LT_LOOP_V_D, LT_LOOP_V_G, LT_LOOP_V_F,
    LT_LOOP_FFN_D, LT_LOOP_FFN_G, LT_LOOP_FFN_F,
    LT_COUNT = 42
};

enum {
    GT_TOK_EMBD = 0, GT_TOK_NORM, GT_OUTPUT, GT_OUTPUT_NORM, GT_OUTPUT_RES_Q, GT_COUNT
};

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

/* ---------------------------------------------------------------- helpers & LUTs */
static float ue4m3_tab[256];
#if IK_NEON
static uint64_t ik_smask[128];   /* byte j = 0xFF where bit j of ik_ksigns_iq2xs[i] is set */
#endif
static float f16_tab[65536];
static const int8_t fp4_vals[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

static void* huge_alloc(size_t bytes) {
    size_t sz = (bytes + 63u) & ~(size_t)63u;
    if (sz == 0) sz = 64;
    void* p = NULL;
    if (posix_memalign(&p, 2u << 20, sz) != 0) {
        if (posix_memalign(&p, 64, sz) != 0) return NULL;
    } else if (sz >= (2u << 20)) {
#if defined(__linux__)
        madvise(p, sz, MADV_HUGEPAGE);
#endif
    }
    return p;
}

static inline float fp16_to_f32_soft(uint16_t h) {
    uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s;
        else {
            float v = ldexpf((float)m, -24);
            return (h & 0x8000u) ? -v : v;
        }
    } else if (e == 31) {
        u = s | 0x7F800000u | (m << 13);
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float f; memcpy(&f, &u, 4); return f;
}

static inline float h2f(uint16_t h) {
#if IK_AVX2
    return _cvtsh_ss(h);
#elif IK_NEON
    __fp16 v; memcpy(&v, &h, 2); return (float)v;
#else
    return f16_tab[h];
#endif
}

static inline uint16_t f2h(float x) {
#if IK_AVX2
    return _cvtss_sh(x, 0);
#elif IK_NEON
    __fp16 v = (__fp16)x; uint16_t h; memcpy(&h, &v, 2); return h;
#else
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = (u >> 16) & 0x8000u;
    int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t m = u & 0x7FFFFFu;
    if (e < -24) return (uint16_t)s;
    if (e < -14) {
        m |= 0x800000u;
        int sh = -14 - e;
        uint32_t rb = 1u << (12 + sh);
        uint32_t mm = (m + rb) >> (13 + sh);
        return (uint16_t)(s | mm);
    }
    if (e > 15) return (uint16_t)(s | 0x7C00u);
    m += 0x1000u;
    if (m & 0x800000u) { m = 0; e++; }
    if (e > 15) return (uint16_t)(s | 0x7C00u);
    return (uint16_t)(s | ((uint32_t)(e + 15) << 10) | (m >> 13));
#endif
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

static void init_tables(void) {
    static int inited = 0;
    if (inited) return;
    inited = 1;
#if IK_NEON
    for (int i = 0; i < 128; i++) {
        uint64_t m = 0;
        for (int j = 0; j < 8; j++) if (ik_ksigns_iq2xs[i] & (1u << j)) m |= (uint64_t)0xFF << (8 * j);
        ik_smask[i] = m;
    }
#endif
    for (int i = 0; i < 65536; i++) f16_tab[i] = fp16_to_f32_soft((uint16_t)i);
    for (int x = 0; x < 256; x++) {
        if (x == 0 || x == 0x7F) ue4m3_tab[x] = 0.f;
        else {
            int e = (x >> 3) & 0xF, m = x & 7;
            float v = (e == 0) ? ldexpf((float)m, -9) : ldexpf(1.0f + m * 0.125f, e - 7);
            ue4m3_tab[x] = v * 0.5f;
        }
    }
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- vector & quant helpers */
#if IK_AVX2
static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

static inline float amax32(const float* x) {
    __m256 m = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.f);
    m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + 0)));
    m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + 8)));
    m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + 16)));
    m = _mm256_max_ps(m, _mm256_andnot_ps(sign, _mm256_loadu_ps(x + 24)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    return _mm_cvtss_f32(_mm_max_ss(m4, _mm_movehdup_ps(m4)));
}

static void q8_quantize(const float* x, int n, int8_t* q, float* d) {
    int nb = n / QK_Q8;
    for (int b = 0; b < nb; b++) {
        float am = amax32(x + b * QK_Q8);
        float sc = am / 127.0f;
        float is = sc != 0.f ? 1.f / sc : 0.f;
        d[b] = _cvtsh_ss(_cvtss_sh(sc, 0));
        __m256 vis = _mm256_set1_ps(is);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + b * QK_Q8 + 0), vis));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + b * QK_Q8 + 8), vis));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + b * QK_Q8 + 16), vis));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + b * QK_Q8 + 24), vis));
        __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(q + b * QK_Q8), p);
    }
}

static void q8_to_nv(const int8_t* xq, int n, int8_t* xnv) {
    int nb = n / QK_NV;
    for (int b = 0; b < nb; b++) {
        const int8_t* src = xq + b * QK_NV;
        int8_t* dst = xnv + b * QK_NV;
        uint64_t a0, a1, a2, a3, a4, a5, a6, a7;
        memcpy(&a0, src + 0,  8); memcpy(&a1, src + 8,  8);
        memcpy(&a2, src + 16, 8); memcpy(&a3, src + 24, 8);
        memcpy(&a4, src + 32, 8); memcpy(&a5, src + 40, 8);
        memcpy(&a6, src + 48, 8); memcpy(&a7, src + 56, 8);
        memcpy(dst + 0,  &a0, 8); memcpy(dst + 8,  &a2, 8);
        memcpy(dst + 16, &a4, 8); memcpy(dst + 24, &a6, 8);
        memcpy(dst + 32, &a1, 8); memcpy(dst + 40, &a3, 8);
        memcpy(dst + 48, &a5, 8); memcpy(dst + 56, &a7, 8);
    }
}
#elif IK_NEON
#define IK_IQ2_TT 32  /* tokens per weight decode in the NEON IQ2 kernels (= IK_MOE_NT: one decode per expert chunk) */
/* ---- AArch64 NEON helpers. Integer dot products give the same int32 sums as the plain C kernels, and the float
 * scale arithmetic is written in the same order, so the quantized kernels match the IK_NO_SIMD build exactly. */
static inline int32x4_t ik_dot16(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);                      /* lane k += sum of bytes 4k..4k+3 */
#else
    int32x4_t s0 = vpaddlq_s16(vmull_s8(vget_low_s8(a), vget_low_s8(b)));
    int32x4_t s1 = vpaddlq_s16(vmull_high_s8(a, b));
    return vaddq_s32(acc, vpaddq_s32(s0, s1));        /* same lane layout as sdot */
#endif
}
static inline int ik_dot32(const int8x16_t w0, const int8x16_t w1, const int8_t* x) {
    return vaddvq_s32(ik_dot16(ik_dot16(vdupq_n_s32(0), w0, vld1q_s8(x)), w1, vld1q_s8(x + 16)));
}
/* negate byte j of the first (second) 8 where bit j of sign pattern i0 (i1) is set: (w ^ m) - m */
static inline int8x16_t ik_signs(int8x16_t w, unsigned i0, unsigned i1) {
    int8x16_t m = vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(ik_smask[i0]), vcreate_u64(ik_smask[i1])));
    return vsubq_s8(veorq_s8(w, m), m);
}
static inline int8x16_t ik_grid2(uint64_t a, uint64_t b) {
    return vreinterpretq_s8_u64(vcombine_u64(vcreate_u64(a), vcreate_u64(b)));
}
static inline float ik_dotf(const float* a, const float* b, int n) {
    float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    float r = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
static inline void ik_axpyf(float* y, float w, const float* x, int n) {   /* y += w * x */
    float32x4_t vw = vdupq_n_f32(w);
    int i = 0;
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_f32(vld1q_f32(y + i), vld1q_f32(x + i), vw));
    for (; i < n; i++) y[i] += w * x[i];
}

static void q8_quantize(const float* x, int n, int8_t* q, float* d) {
    int nb = n / QK_Q8;
    const int32x4_t lo = vdupq_n_s32(-127), hi = vdupq_n_s32(127);
    for (int b = 0; b < nb; b++) {
        const float* xb = x + b * QK_Q8;
        float32x4_t v[8];
        float32x4_t m = vdupq_n_f32(0.f);
        for (int j = 0; j < 8; j++) { v[j] = vld1q_f32(xb + 4 * j); m = vmaxq_f32(m, vabsq_f32(v[j])); }
        float am = vmaxvq_f32(m);
        float sc = am / 127.0f;
        float is = sc != 0.f ? 1.f / sc : 0.f;
        d[b] = h2f(f2h(sc));
        int16x8_t h[4];
        for (int j = 0; j < 4; j++) {
            int32x4_t a = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[2 * j], is)), lo), hi);
            int32x4_t c = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(vmulq_n_f32(v[2 * j + 1], is)), lo), hi);
            h[j] = vcombine_s16(vmovn_s32(a), vmovn_s32(c));
        }
        vst1q_s8(q + b * QK_Q8, vcombine_s8(vmovn_s16(h[0]), vmovn_s16(h[1])));
        vst1q_s8(q + b * QK_Q8 + 16, vcombine_s8(vmovn_s16(h[2]), vmovn_s16(h[3])));
    }
}
static void q8_to_nv(const int8_t* xq, int n, int8_t* xnv) { memcpy(xnv, xq, (size_t)n); }

#if defined(__ARM_FEATURE_MATMUL_INT8)
#define IK_MMLA 1
/* ---- i8mm (SMMLA) kernels for 2+ tokens (prompt reading). vmmlaq_s32 multiplies a 2x8 int8 tile (two weight rows)
 * by a 2x8 tile (two tokens) into a 2x2 int32 tile [rA tA, rA tB, rB tA, rB tB]. The integer sums equal the
 * sdot/plain C sums, and each (token, row) lane accumulates in the plain C order with the same fused
 * multiply-add, so the results stay bit-identical to the IK_NO_SIMD build. Odd token counts repeat the last token
 * in the second slot and drop that lane. */
#define IK_MM_TT 32         /* tokens per weight decode (16 token pairs; = IK_MOE_NT: each IQ2 block is decoded once per
                               expert chunk, the accumulators live in L1) */

static inline int8x16_t ik_pair8(const int8_t* a, const int8_t* b) { return vcombine_s8(vld1_s8(a), vld1_s8(b)); }

/* store the 2x2 tiles of one token pair q (tokens tA, tB; tB valid when hasB) for row pairs (0,1) and (2,3) */
static inline void ik_mm_store(float* y, int ys, int tA, int tB, int hasB, float32x4_t a01, float32x4_t a23) {
    y[(size_t)tA * ys + 0] = vgetq_lane_f32(a01, 0);
    y[(size_t)tA * ys + 1] = vgetq_lane_f32(a01, 2);
    y[(size_t)tA * ys + 2] = vgetq_lane_f32(a23, 0);
    y[(size_t)tA * ys + 3] = vgetq_lane_f32(a23, 2);
    if (hasB) {
        y[(size_t)tB * ys + 0] = vgetq_lane_f32(a01, 1);
        y[(size_t)tB * ys + 1] = vgetq_lane_f32(a01, 3);
        y[(size_t)tB * ys + 2] = vgetq_lane_f32(a23, 1);
        y[(size_t)tB * ys + 3] = vgetq_lane_f32(a23, 3);
    }
}

/* lanes (rA, rA, rB, rB) x (tA, tB, tA, tB) */
static inline float32x4_t ik_rr_tt(float ra, float rb, float ta, float tb) {
    const float r[4] = {ra, ra, rb, rb}, t[4] = {ta, tb, ta, tb};
    return vmulq_f32(vld1q_f32(r), vld1q_f32(t));
}

static void k_iq2xxs128_x4_mm(const uint8_t* w, size_t rb, int nb128, int T, const int8_t* xq, const float* xd,
                              int xstride, float* y, int ys) {
    const int xds = xstride / QK_Q8;
    for (int t0 = 0; t0 < T; t0 += IK_MM_TT) {
        const int nt = T - t0 < IK_MM_TT ? T - t0 : IK_MM_TT;
        const int np = (nt + 1) / 2;
        float32x4_t acc[IK_MM_TT / 2][2];
        for (int q = 0; q < IK_MM_TT / 2; q++) acc[q][0] = acc[q][1] = vdupq_n_f32(0.f);
        for (int b = 0; b < nb128; b++) {
            const uint8_t* P[4];
            float d[4];
            for (int r = 0; r < 4; r++) {
                P[r] = w + (size_t)r * rb + (size_t)b * BIQ2_XXS;
                __builtin_prefetch(P[r] + 4 * BIQ2_XXS);
                uint16_t hd; memcpy(&hd, P[r], 2);
                d[r] = h2f(hd) * 0.25f;
            }
            for (int ib = 0; ib < 4; ib++) {
                int8x16_t A[2][4];
                float dl[4];
                uint32_t ax0[4], ax1[4];
                for (int r = 0; r < 4; r++) {
                    memcpy(&ax0[r], P[r] + 2 + 8 * ib, 4);
                    memcpy(&ax1[r], P[r] + 2 + 8 * ib + 4, 4);
                    dl[r] = d[r] * (0.5f + (float)(ax1[r] >> 28));
                }
                for (int p = 0; p < 2; p++) {
                    const int ra = 2 * p, rbb = 2 * p + 1;
                    for (int l = 0; l < 4; l++)
                        A[p][l] = ik_signs(ik_grid2(ik_iq2xxs_grid[(ax0[ra] >> (8 * l)) & 255u],
                                                    ik_iq2xxs_grid[(ax0[rbb] >> (8 * l)) & 255u]),
                                           (ax1[ra] >> (7 * l)) & 127u,
                                           (ax1[rbb] >> (7 * l)) & 127u);
                }
                for (int q = 0; q < np; q++) {
                    const int tA = t0 + 2 * q, tB = (tA + 1 < t0 + nt) ? tA + 1 : tA;
                    const int8_t* xa = xq + (size_t)tA * xstride + b * QK_IQ2 + ib * 32;
                    const int8_t* xb = xq + (size_t)tB * xstride + b * QK_IQ2 + ib * 32;
                    const float xA = xd[(size_t)tA * xds + b * 4 + ib], xB = xd[(size_t)tB * xds + b * 4 + ib];
                    int8x16_t B0 = ik_pair8(xa, xb), B1 = ik_pair8(xa + 8, xb + 8);
                    int8x16_t B2 = ik_pair8(xa + 16, xb + 16), B3 = ik_pair8(xa + 24, xb + 24);
                    for (int p = 0; p < 2; p++) {
                        int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), A[p][0], B0);
                        s = vmmlaq_s32(s, A[p][1], B1);
                        s = vmmlaq_s32(s, A[p][2], B2);
                        s = vmmlaq_s32(s, A[p][3], B3);
                        acc[q][p] = vfmaq_f32(acc[q][p], vcvtq_f32_s32(s), ik_rr_tt(dl[2 * p], dl[2 * p + 1], xA, xB));
                    }
                }
            }
        }
        for (int q = 0; q < np; q++) {
            const int tA = t0 + 2 * q;
            ik_mm_store(y, ys, tA, tA + 1, tA + 1 < t0 + nt, acc[q][0], acc[q][1]);
        }
    }
}

static void k_iq2xs128_x4_mm(const uint8_t* w, size_t rb, int nb128, int T, const int8_t* xq, const float* xd,
                             int xstride, float* y, int ys) {
    const int xds = xstride / QK_Q8;
    for (int t0 = 0; t0 < T; t0 += IK_MM_TT) {
        const int nt = T - t0 < IK_MM_TT ? T - t0 : IK_MM_TT;
        const int np = (nt + 1) / 2;
        float32x4_t acc[IK_MM_TT / 2][2];
        for (int q = 0; q < IK_MM_TT / 2; q++) acc[q][0] = acc[q][1] = vdupq_n_f32(0.f);
        for (int b = 0; b < nb128; b++) {
            const uint8_t* P[4];
            float d[4];
            for (int r = 0; r < 4; r++) {
                P[r] = w + (size_t)r * rb + (size_t)b * BIQ2_XS;
                __builtin_prefetch(P[r] + 4 * BIQ2_XS);
                uint16_t hd; memcpy(&hd, P[r], 2);
                d[r] = h2f(hd) * 0.25f;
            }
            for (int ib = 0; ib < 4; ib++) {
                int8x16_t A[2][4];
                float dl0[4], dl1[4];
                uint16_t qq[4][4];
                for (int r = 0; r < 4; r++) {
                    memcpy(qq[r], P[r] + 2 + 8 * ib, 8);
                    const uint8_t sc = P[r][34 + ib];
                    dl0[r] = d[r] * (0.5f + (float)(sc & 15));
                    dl1[r] = d[r] * (0.5f + (float)(sc >> 4));
                }
                for (int p = 0; p < 2; p++) {
                    const int ra = 2 * p, rbb = 2 * p + 1;
                    for (int l = 0; l < 4; l++)
                        A[p][l] = ik_signs(ik_grid2(ik_iq2xs_grid[qq[ra][l] & 511u], ik_iq2xs_grid[qq[rbb][l] & 511u]),
                                           qq[ra][l] >> 9, qq[rbb][l] >> 9);
                }
                for (int q = 0; q < np; q++) {
                    const int tA = t0 + 2 * q, tB = (tA + 1 < t0 + nt) ? tA + 1 : tA;
                    const int8_t* xa = xq + (size_t)tA * xstride + b * QK_IQ2 + ib * 32;
                    const int8_t* xb = xq + (size_t)tB * xstride + b * QK_IQ2 + ib * 32;
                    const float xA = xd[(size_t)tA * xds + b * 4 + ib], xB = xd[(size_t)tB * xds + b * 4 + ib];
                    int8x16_t Bv[4] = {ik_pair8(xa, xb), ik_pair8(xa + 8, xb + 8),
                                       ik_pair8(xa + 16, xb + 16), ik_pair8(xa + 24, xb + 24)};
                    for (int p = 0; p < 2; p++) {
                        /* per-8 sums in the plain C order: l0, l1 with the low-nibble scale, l2, l3 with the high */
                        const float32x4_t db0 = ik_rr_tt(dl0[2 * p], dl0[2 * p + 1], xA, xB);
                        const float32x4_t db1 = ik_rr_tt(dl1[2 * p], dl1[2 * p + 1], xA, xB);
                        float32x4_t a = acc[q][p];
                        a = vfmaq_f32(a, vcvtq_f32_s32(vmmlaq_s32(vdupq_n_s32(0), A[p][0], Bv[0])), db0);
                        a = vfmaq_f32(a, vcvtq_f32_s32(vmmlaq_s32(vdupq_n_s32(0), A[p][1], Bv[1])), db0);
                        a = vfmaq_f32(a, vcvtq_f32_s32(vmmlaq_s32(vdupq_n_s32(0), A[p][2], Bv[2])), db1);
                        a = vfmaq_f32(a, vcvtq_f32_s32(vmmlaq_s32(vdupq_n_s32(0), A[p][3], Bv[3])), db1);
                        acc[q][p] = a;
                    }
                }
            }
        }
        for (int q = 0; q < np; q++) {
            const int tA = t0 + 2 * q;
            ik_mm_store(y, ys, tA, tA + 1, tA + 1 < t0 + nt, acc[q][0], acc[q][1]);
        }
    }
}

/* Q8_0: weight rows (rA, rB) of one 32-value block as four 2x8 tiles */
static void k_q8_raw_x4_mm(const uint8_t* w, size_t rb, int nb, int T, const int8_t* xq, const float* xd,
                           int xstride, float* y, int ys) {
    const int xds = xstride / QK_Q8;
    for (int t0 = 0; t0 < T; t0 += IK_MM_TT) {
        const int nt = T - t0 < IK_MM_TT ? T - t0 : IK_MM_TT;
        const int np = (nt + 1) / 2;
        float32x4_t acc[IK_MM_TT / 2][2];
        for (int q = 0; q < IK_MM_TT / 2; q++) acc[q][0] = acc[q][1] = vdupq_n_f32(0.f);
        for (int b = 0; b < nb; b++) {
            int8x16_t A[2][4];
            float ws[4];
            for (int r = 0; r < 4; r++) {
                const uint8_t* P = w + (size_t)r * rb + (size_t)b * BQ8;
                __builtin_prefetch(P + 272);
                uint16_t hs; memcpy(&hs, P, 2);
                ws[r] = h2f(hs);
            }
            for (int p = 0; p < 2; p++) {
                const int8_t* pa = (const int8_t*)(w + (size_t)(2 * p) * rb + (size_t)b * BQ8 + 2);
                const int8_t* pb = (const int8_t*)(w + (size_t)(2 * p + 1) * rb + (size_t)b * BQ8 + 2);
                for (int l = 0; l < 4; l++) A[p][l] = ik_pair8(pa + 8 * l, pb + 8 * l);
            }
            for (int q = 0; q < np; q++) {
                const int tA = t0 + 2 * q, tB = (tA + 1 < t0 + nt) ? tA + 1 : tA;
                const int8_t* xa = xq + (size_t)tA * xstride + b * QK_Q8;
                const int8_t* xb = xq + (size_t)tB * xstride + b * QK_Q8;
                const float xA = xd[(size_t)tA * xds + b], xB = xd[(size_t)tB * xds + b];
                const float tt[4] = {xA, xB, xA, xB};
                const float32x4_t xdv = vld1q_f32(tt);
                int8x16_t B0 = ik_pair8(xa, xb), B1 = ik_pair8(xa + 8, xb + 8);
                int8x16_t B2 = ik_pair8(xa + 16, xb + 16), B3 = ik_pair8(xa + 24, xb + 24);
                for (int p = 0; p < 2; p++) {
                    int32x4_t s = vmmlaq_s32(vdupq_n_s32(0), A[p][0], B0);
                    s = vmmlaq_s32(s, A[p][1], B1);
                    s = vmmlaq_s32(s, A[p][2], B2);
                    s = vmmlaq_s32(s, A[p][3], B3);
                    const float rr[4] = {ws[2 * p], ws[2 * p], ws[2 * p + 1], ws[2 * p + 1]};
                    /* plain C: sum += (float)isum * ws * xd  ==  sum + ((isum * ws) * xd) */
                    acc[q][p] = vfmaq_f32(acc[q][p], vmulq_f32(vcvtq_f32_s32(s), vld1q_f32(rr)), xdv);
                }
            }
        }
        for (int q = 0; q < np; q++) {
            const int tA = t0 + 2 * q;
            ik_mm_store(y, ys, tA, tA + 1, tA + 1 < t0 + nt, acc[q][0], acc[q][1]);
        }
    }
}
#else
#define IK_MMLA 0
#endif
#else
static void q8_quantize(const float* x, int n, int8_t* q, float* d) {
    int nb = n / QK_Q8;
    for (int b = 0; b < nb; b++) {
        float am = 0.f;
        for (int j = 0; j < QK_Q8; j++) {
            float a = fabsf(x[b * QK_Q8 + j]);
            if (a > am) am = a;
        }
        float sc = am / 127.0f;
        float is = sc != 0.f ? 1.f / sc : 0.f;
        d[b] = h2f(f2h(sc));
        for (int j = 0; j < QK_Q8; j++) {
            int v = (int)nearbyintf(x[b * QK_Q8 + j] * is);
            if (v > 127) v = 127; else if (v < -127) v = -127;
            q[b * QK_Q8 + j] = (int8_t)v;
        }
    }
}
static void q8_to_nv(const int8_t* xq, int n, int8_t* xnv) { memcpy(xnv, xq, (size_t)n); }
#endif

#ifndef IK_MMLA
#define IK_MMLA 0
#endif

static void q8k_quantize(const float* x, int n, void* out) {
    typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; } blk_q8k;
    blk_q8k* dst = (blk_q8k*)out;
    int nb = n / QK_K;
    for (int b = 0; b < nb; b++) {
        const float* xb = x + b * QK_K;
        float maxv = 0.f, amax = 0.f;
        for (int j = 0; j < QK_K; j++) {
            float a = fabsf(xb[j]);
            if (a > amax) { amax = a; maxv = xb[j]; }
        }
        if (amax == 0.f) {
            dst[b].d = 0.f;
            memset(dst[b].qs, 0, QK_K);
            memset(dst[b].bsums, 0, sizeof(dst[b].bsums));
            continue;
        }
        float iscale = -127.f / maxv;
        dst[b].d = 1.f / iscale;
        for (int j = 0; j < QK_K; j++) {
            int v = (int)nearbyintf(xb[j] * iscale);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            dst[b].qs[j] = (int8_t)v;
        }
        for (int g = 0; g < QK_K / 16; g++) {
            int s = 0;
            for (int l = 0; l < 16; l++) s += dst[b].qs[g * 16 + l];
            dst[b].bsums[g] = (int16_t)s;
        }
    }
}

static void rmsnorm_f32(float* y, const float* x, const float* w, int n) {
#if IK_AVX2
    __m256 s0 = _mm256_setzero_ps(), s1 = s0;
    for (int i = 0; i < n; i += 16) {
        __m256 a = _mm256_loadu_ps(x + i), b = _mm256_loadu_ps(x + i + 8);
        s0 = _mm256_fmadd_ps(a, a, s0);
        s1 = _mm256_fmadd_ps(b, b, s1);
    }
    float ss = hsum8(_mm256_add_ps(s0, s1));
#elif IK_NEON
    float ss = ik_dotf(x, x, n);
#else
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
#endif
    float r = 1.f / sqrtf(ss / (float)n + IK_RMS_EPS);
    if (w) {
#if IK_AVX2
        __m256 vr = _mm256_set1_ps(r);
        for (int i = 0; i < n; i += 8)
            _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vr), _mm256_loadu_ps(w + i)));
#elif IK_NEON
        int i = 0;
        for (; i + 4 <= n; i += 4)
            vst1q_f32(y + i, vmulq_f32(vmulq_n_f32(vld1q_f32(x + i), r), vld1q_f32(w + i)));
        for (; i < n; i++) y[i] = x[i] * r * w[i];
#else
        for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
#endif
    } else {
        for (int i = 0; i < n; i++) y[i] = x[i] * r;
    }
}

/* ---------------------------------------------------------------- thread pool */
typedef void (*ik_fn)(int t, void* arg);
static struct {
    int n, cpu[IK_KMAXT];
    pthread_t th[IK_KMAXT];
    _Atomic unsigned gen, done;
    _Atomic int quit, sleepers;
    ik_fn fn;
    void* arg;
} KP;

/* Sleep while *a == v / wake every sleeper. Linux: futex. macOS 14.4+: os_sync_wait_on_address. Elsewhere (and
 * with IK_FORCE_CONDVAR): one mutex + condvar; the waker changes *a before it takes the lock, the sleeper re-checks
 * *a under the lock, so no wake-up is lost. */
#if defined(__linux__) && !defined(IK_FORCE_CONDVAR)
static void ik_wait(_Atomic unsigned* a, unsigned v) { syscall(SYS_futex, a, 0 /* FUTEX_WAIT */, v, NULL, NULL, 0); }
static void ik_wake_all(_Atomic unsigned* a) { syscall(SYS_futex, a, 1 /* FUTEX_WAKE */, 0x7fffffff, NULL, NULL, 0); }
#else
static pthread_mutex_t ik_wmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ik_wcv = PTHREAD_COND_INITIALIZER;
static void ik_wait(_Atomic unsigned* a, unsigned v) {
#if defined(IK_HAVE_OS_SYNC) && !defined(IK_FORCE_CONDVAR)
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wait_on_address((void*)a, (uint64_t)v, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE);
        return;
    }
#endif
    pthread_mutex_lock(&ik_wmu);
    while (atomic_load(a) == v) pthread_cond_wait(&ik_wcv, &ik_wmu);
    pthread_mutex_unlock(&ik_wmu);
}
static void ik_wake_all(_Atomic unsigned* a) {
#if defined(IK_HAVE_OS_SYNC) && !defined(IK_FORCE_CONDVAR)
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wake_by_address_all((void*)a, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
        return;
    }
#endif
    (void)a;
    pthread_mutex_lock(&ik_wmu);
    pthread_cond_broadcast(&ik_wcv);
    pthread_mutex_unlock(&ik_wmu);
}
#endif

static void* kp_worker(void* a) {
    int t = (int)(intptr_t)a;
#if defined(__linux__)
    if (KP.cpu[t] >= 0) {
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(KP.cpu[t], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
#elif defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);   /* macOS has no pinning; ask for P-cores */
#endif
    unsigned seen = 0;
    for (;;) {
        unsigned g, spins = 0;
        while ((g = atomic_load_explicit(&KP.gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&KP.quit, memory_order_relaxed)) return NULL;
            if (++spins < 16000) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                __asm__ __volatile__("yield");
#endif
                continue;
            }
            atomic_fetch_add(&KP.sleepers, 1);
            if (atomic_load_explicit(&KP.gen, memory_order_acquire) == seen)
                ik_wait(&KP.gen, seen);
            atomic_fetch_sub(&KP.sleepers, 1);
            spins = 0;
        }
        seen = g;
        if (atomic_load_explicit(&KP.quit, memory_order_relaxed)) return NULL;
        KP.fn(t, KP.arg);
        atomic_fetch_add_explicit(&KP.done, 1, memory_order_acq_rel);
    }
}

#if defined(__linux__)
static cpu_set_t g_orig_affinity;
static int g_have_orig_affinity = 0;
#endif

static void kp_init(int req_threads) {
    if (KP.n > 0) return;
    int list[IK_KMAXT], nc = 0;
#if defined(__linux__)
    if (!g_have_orig_affinity) {
        CPU_ZERO(&g_orig_affinity);
        if (sched_getaffinity(0, sizeof(g_orig_affinity), &g_orig_affinity) == 0)
            g_have_orig_affinity = 1;
    }
    cpu_set_t avail = g_orig_affinity;
    if (!g_have_orig_affinity) {
        CPU_ZERO(&avail);
        sched_getaffinity(0, sizeof(avail), &avail);
    }
    for (int c = 0; c < 256 && nc < IK_KMAXT; c++) if (CPU_ISSET(c, &avail)) list[nc++] = c;
    if (nc == 0) { list[0] = 0; nc = 1; }
#else
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
    for (nc = 0; nc < ncpu && nc < IK_KMAXT; nc++) list[nc] = -1;   /* no pinning */
#endif
    int n = req_threads > 0 ? req_threads : nc;
    if (n > nc) n = nc;
    if (n > IK_KMAXT) n = IK_KMAXT;
    if (n < 1) n = 1;
    KP.n = n;
    atomic_store(&KP.gen, 0);
    atomic_store(&KP.quit, 0);
    atomic_store(&KP.sleepers, 0);
    KP.cpu[0] = list[0];
#if defined(__linux__)
    cpu_set_t c0; CPU_ZERO(&c0); CPU_SET(list[0], &c0);
    pthread_setaffinity_np(pthread_self(), sizeof(c0), &c0);
#elif defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    for (int t = 1; t < n; t++) {
        KP.cpu[t] = list[t];
        pthread_create(&KP.th[t], NULL, kp_worker, (void*)(intptr_t)t);
    }
}

static void kp_destroy(void) {
    if (KP.n <= 0) return;
    atomic_store(&KP.quit, 1);
    atomic_fetch_add(&KP.gen, 1);
    ik_wake_all(&KP.gen);
    for (int t = 1; t < KP.n; t++) pthread_join(KP.th[t], NULL);
    KP.n = 0;
#if defined(__linux__)
    if (g_have_orig_affinity)
        pthread_setaffinity_np(pthread_self(), sizeof(g_orig_affinity), &g_orig_affinity);
#endif
}

static void kp_run(ik_fn fn, void* arg) {
    if (KP.n <= 1) { fn(0, arg); return; }
    KP.fn = fn; KP.arg = arg;
    atomic_store_explicit(&KP.done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&KP.gen, 1, memory_order_seq_cst);
    if (atomic_load(&KP.sleepers)) ik_wake_all(&KP.gen);
    fn(0, arg);
    for (int s = 0; atomic_load_explicit(&KP.done, memory_order_acquire) != (unsigned)KP.n - 1; s++) {
        if (s < 16000) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield");
#endif
        } else sched_yield();
    }
}

/* ---------------------------------------------------------------- GEMV / GEMM kernels */
typedef struct {
    int rows, cols, type;
    uint8_t* data;
    size_t bytes;
} Mat;

static Mat wrap_mat(const uint8_t* raw, int ty, int rows, int cols) {
    size_t rb = (ty == TY_Q4_0) ? (size_t)(cols / QK_Q4) * BQ4 :
                (ty == TY_Q8)   ? (size_t)(cols / QK_Q8) * BQ8 :
                                  (size_t)cols * 4;
    Mat m = {rows, cols, ty, (uint8_t*)raw, (size_t)rows * rb};
    return m;
}

static void k_f32x4(const float* w, int K, int T, const float* x, int xs, float* y, int ys) {
#if IK_AVX2
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * xs;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int k = 0; k < K; k += 8) {
            __m256 xv = _mm256_loadu_ps(xt + k);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 0 * K + k), xv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 1 * K + k), xv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 2 * K + k), xv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 3 * K + k), xv, a3);
        }
        float* yt = y + (size_t)t * ys;
        yt[0] = hsum8(a0); yt[1] = hsum8(a1); yt[2] = hsum8(a2); yt[3] = hsum8(a3);
    }
#elif IK_NEON
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * xs;
        float32x4_t a0 = vdupq_n_f32(0.f), a1 = a0, a2 = a0, a3 = a0;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            float32x4_t xv = vld1q_f32(xt + k);
            a0 = vfmaq_f32(a0, vld1q_f32(w + 0 * (size_t)K + k), xv);
            a1 = vfmaq_f32(a1, vld1q_f32(w + 1 * (size_t)K + k), xv);
            a2 = vfmaq_f32(a2, vld1q_f32(w + 2 * (size_t)K + k), xv);
            a3 = vfmaq_f32(a3, vld1q_f32(w + 3 * (size_t)K + k), xv);
        }
        float* yt = y + (size_t)t * ys;
        yt[0] = vaddvq_f32(a0); yt[1] = vaddvq_f32(a1); yt[2] = vaddvq_f32(a2); yt[3] = vaddvq_f32(a3);
        for (; k < K; k++) for (int r = 0; r < 4; r++) yt[r] += w[(size_t)r * K + k] * xt[k];
    }
#else
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * xs;
        for (int r = 0; r < 4; r++) {
            float a = 0.f; const float* wr = w + (size_t)r * K;
            for (int k = 0; k < K; k++) a += wr[k] * xt[k];
            y[(size_t)t * ys + r] = a;
        }
    }
#endif
}

#if IK_AVX2
static void k_q8_raw_x4(const uint8_t* w, size_t rb, int nb, int T, const int8_t* xq, const float* xd,
                        int xstride, float* y, int ys) {
    const __m256i ones = _mm256_set1_epi16(1);
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        const int8_t *x0 = xq + (size_t)t * xstride, *x1 = x0 + xstride, *x2 = x1 + xstride, *x3 = x2 + xstride;
        const float *d0 = xd + (size_t)t * (xstride / QK_Q8), *d1 = d0 + xstride / QK_Q8, *d2 = d1 + xstride / QK_Q8, *d3 = d2 + xstride / QK_Q8;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        __m256 c0 = a0, c1 = a0, c2 = a0, c3 = a0;
        __m256 e0 = a0, e1 = a0, e2 = a0, e3 = a0;
        for (int b = 0; b < nb; b++) {
            size_t o = (size_t)b * BQ8;
            _mm_prefetch((const char*)(w0 + o + 272), _MM_HINT_T0);
            _mm_prefetch((const char*)(w1 + o + 272), _MM_HINT_T0);
            _mm_prefetch((const char*)(w2 + o + 272), _MM_HINT_T0);
            _mm_prefetch((const char*)(w3 + o + 272), _MM_HINT_T0);
            const __m256i xa = _mm256_loadu_si256((const __m256i*)(x0 + b * QK_Q8));
            const __m256i xb = _mm256_loadu_si256((const __m256i*)(x1 + b * QK_Q8));
            const __m256i xc = _mm256_loadu_si256((const __m256i*)(x2 + b * QK_Q8));
            const __m256i xe = _mm256_loadu_si256((const __m256i*)(x3 + b * QK_Q8));
            float da = d0[b], db = d1[b], dc = d2[b], de = d3[b];
            #define Q8R4(WP, AA, AB, AC, AE) do { \
                const uint8_t* P_ = (WP) + o; uint16_t hs_; memcpy(&hs_, P_, 2); float ws_ = _cvtsh_ss(hs_); \
                __m256i wv_ = _mm256_loadu_si256((const __m256i*)(P_ + 2)); \
                __m256i uv_ = _mm256_sign_epi8(wv_, wv_); \
                AA = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(uv_, _mm256_sign_epi8(xa, wv_)), ones)), _mm256_set1_ps(ws_ * da), AA); \
                AB = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(uv_, _mm256_sign_epi8(xb, wv_)), ones)), _mm256_set1_ps(ws_ * db), AB); \
                AC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(uv_, _mm256_sign_epi8(xc, wv_)), ones)), _mm256_set1_ps(ws_ * dc), AC); \
                AE = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(uv_, _mm256_sign_epi8(xe, wv_)), ones)), _mm256_set1_ps(ws_ * de), AE); \
            } while (0)
            Q8R4(w0, a0, b0, c0, e0); Q8R4(w1, a1, b1, c1, e1);
            Q8R4(w2, a2, b2, c2, e2); Q8R4(w3, a3, b3, c3, e3);
            #undef Q8R4
        }
        float *ya = y + (size_t)t * ys, *yb = ya + ys, *yc = yb + ys, *ye = yc + ys;
        ya[0] = hsum8(a0); ya[1] = hsum8(a1); ya[2] = hsum8(a2); ya[3] = hsum8(a3);
        yb[0] = hsum8(b0); yb[1] = hsum8(b1); yb[2] = hsum8(b2); yb[3] = hsum8(b3);
        yc[0] = hsum8(c0); yc[1] = hsum8(c1); yc[2] = hsum8(c2); yc[3] = hsum8(c3);
        ye[0] = hsum8(e0); ye[1] = hsum8(e1); ye[2] = hsum8(e2); ye[3] = hsum8(e3);
    }
    for (; t < T; t++) {
        const int8_t* xa0 = xq + (size_t)t * xstride;
        const float* da0 = xd + (size_t)t * (xstride / QK_Q8);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            size_t o = (size_t)b * BQ8;
            _mm_prefetch((const char*)(w0 + o + 272), _MM_HINT_T0);
            _mm_prefetch((const char*)(w1 + o + 272), _MM_HINT_T0);
            _mm_prefetch((const char*)(w2 + o + 272), _MM_HINT_T0);
            _mm_prefetch((const char*)(w3 + o + 272), _MM_HINT_T0);
            const __m256i xa = _mm256_loadu_si256((const __m256i*)(xa0 + b * QK_Q8));
            float dxa = da0[b];
            #define Q8R1(WP, A) do { \
                const uint8_t* P_ = (WP) + o; uint16_t hs_; memcpy(&hs_, P_, 2); float ws_ = _cvtsh_ss(hs_); \
                __m256i wv_ = _mm256_loadu_si256((const __m256i*)(P_ + 2)); \
                A = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(wv_, wv_), _mm256_sign_epi8(xa, wv_)), ones)), _mm256_set1_ps(ws_ * dxa), A); \
            } while (0)
            Q8R1(w0, a0); Q8R1(w1, a1); Q8R1(w2, a2); Q8R1(w3, a3);
            #undef Q8R1
        }
        float* ya = y + (size_t)t * ys;
        ya[0] = hsum8(a0); ya[1] = hsum8(a1); ya[2] = hsum8(a2); ya[3] = hsum8(a3);
    }
}
#elif IK_NEON
/* 4 rows x up to 4 tokens: each weight block is loaded once per tile of tokens. */
static void k_q8_raw_x4(const uint8_t* w, size_t rb, int nb, int T, const int8_t* xq, const float* xd,
                        int xstride, float* y, int ys) {
#if IK_MMLA
    if (T >= 2) { k_q8_raw_x4_mm(w, rb, nb, T, xq, xd, xstride, y, ys); return; }
#endif
    const int xds = xstride / QK_Q8;
    for (int t0 = 0; t0 < T; t0 += 4) {
        const int nt = T - t0 < 4 ? T - t0 : 4;
        float s[4][4] = {{0.f}};
        for (int b = 0; b < nb; b++) {
            for (int r = 0; r < 4; r++) {
                const uint8_t* P = w + (size_t)r * rb + (size_t)b * BQ8;
                __builtin_prefetch(P + 272);
                uint16_t hs; memcpy(&hs, P, 2);
                const float ws = h2f(hs);
                const int8x16_t w0 = vld1q_s8((const int8_t*)P + 2), w1 = vld1q_s8((const int8_t*)P + 18);
                for (int i = 0; i < nt; i++) {
                    int isum = ik_dot32(w0, w1, xq + (size_t)(t0 + i) * xstride + b * QK_Q8);
                    s[i][r] += (float)isum * ws * xd[(size_t)(t0 + i) * xds + b];
                }
            }
        }
        for (int i = 0; i < nt; i++)
            for (int r = 0; r < 4; r++) y[(size_t)(t0 + i) * ys + r] = s[i][r];
    }
}
#else
static void k_q8_raw_x4(const uint8_t* w, size_t rb, int nb, int T, const int8_t* xq, const float* xd,
                        int xstride, float* y, int ys) {
    for (int t = 0; t < T; t++) {
        const int8_t* x = xq + (size_t)t * xstride;
        const float* d = xd + (size_t)t * (xstride / QK_Q8);
        for (int r = 0; r < 4; r++) {
            const uint8_t* wr = w + (size_t)r * rb;
            float sum = 0.f;
            for (int b = 0; b < nb; b++) {
                const uint8_t* P = wr + (size_t)b * BQ8;
                uint16_t hs; memcpy(&hs, P, 2);
                const int8_t* wp = (const int8_t*)(P + 2);
                int isum = 0;
                for (int j = 0; j < 32; j++) isum += wp[j] * x[b * QK_Q8 + j];
                sum += (float)isum * h2f(hs) * d[b];
            }
            y[(size_t)t * ys + r] = sum;
        }
    }
}
#endif

/* Parallel Q8G / F32 multi-op launcher */
#define IK_MAXOP 8
typedef struct {
    const Mat* m;
    int r_end;
    const float* xf;
    const int8_t* xq;
    const float* xd;
    int xs;
    float* y;
    int ys;
} IkOp;

static struct {
    IkOp op[IK_MAXOP];
    int nop, T;
    long first[IK_MAXOP + 1];
    _Atomic long next;
} MG;

static void k_q4_0_x4(const uint8_t* w, size_t rb, int nb, int T, const int8_t* xq, const float* xd,
                      int xstride, float* y, int ys) {
#if IK_AVX2
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m256i eights = _mm256_set1_epi8(8);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int t = 0; t < T; t++) {
        const int8_t* xa0 = xq + (size_t)t * xstride;
        const float* da0 = xd + (size_t)t * (xstride / QK_Q8);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            size_t o = (size_t)b * BQ4;
            const __m256i xa = _mm256_loadu_si256((const __m256i*)(xa0 + b * QK_Q4));
            float dxa = da0[b];
            #define Q4R1(WP, A) do { \
                const uint8_t* P_ = (WP) + o; uint16_t hs_; memcpy(&hs_, P_, 2); float ws_ = _cvtsh_ss(hs_); \
                __m128i q_ = _mm_loadu_si128((const __m128i*)(P_ + 2)); \
                __m128i lo_ = _mm_and_si128(q_, m0f); \
                __m128i hi_ = _mm_and_si128(_mm_srli_epi16(q_, 4), m0f); \
                __m256i wv_ = _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo_), hi_, 1), eights); \
                A = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(wv_), _mm256_sign_epi8(xa, wv_)), ones)), _mm256_set1_ps(ws_ * dxa), A); \
            } while (0)
            Q4R1(w0, a0); Q4R1(w1, a1); Q4R1(w2, a2); Q4R1(w3, a3);
            #undef Q4R1
        }
        float* ya = y + (size_t)t * ys;
        ya[0] = hsum8(a0); ya[1] = hsum8(a1); ya[2] = hsum8(a2); ya[3] = hsum8(a3);
    }
#elif IK_NEON
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    const int8x16_t eights = vdupq_n_s8(8);
    const int xds = xstride / QK_Q8;
    for (int t0 = 0; t0 < T; t0 += 4) {
        const int nt = T - t0 < 4 ? T - t0 : 4;
        float s[4][4] = {{0.f}};
        for (int b = 0; b < nb; b++) {
            for (int r = 0; r < 4; r++) {
                const uint8_t* P = w + (size_t)r * rb + (size_t)b * BQ4;
                uint16_t hs; memcpy(&hs, P, 2);
                const float ws = h2f(hs);
                const uint8x16_t q = vld1q_u8(P + 2);
                const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m0f)), eights);
                const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), eights);
                for (int i = 0; i < nt; i++) {
                    int isum = ik_dot32(lo, hi, xq + (size_t)(t0 + i) * xstride + b * QK_Q4);
                    s[i][r] += (float)isum * ws * xd[(size_t)(t0 + i) * xds + b];
                }
            }
        }
        for (int i = 0; i < nt; i++)
            for (int r = 0; r < 4; r++) y[(size_t)(t0 + i) * ys + r] = s[i][r];
    }
#else
    for (int t = 0; t < T; t++) {
        const int8_t* x = xq + (size_t)t * xstride;
        const float* d = xd + (size_t)t * (xstride / QK_Q8);
        for (int r = 0; r < 4; r++) {
            const uint8_t* wr = w + (size_t)r * rb;
            float sum = 0.f;
            for (int b = 0; b < nb; b++) {
                const uint8_t* P = wr + (size_t)b * BQ4;
                uint16_t hs; memcpy(&hs, P, 2);
                const uint8_t* qs = P + 2;
                int isum = 0;
                for (int j = 0; j < 16; j++) {
                    isum += ((int)(qs[j] & 0xF) - 8) * (int)x[b * QK_Q4 + j]
                          + ((int)(qs[j] >> 4)  - 8) * (int)x[b * QK_Q4 + j + 16];
                }
                sum += (float)isum * h2f(hs) * d[b];
            }
            y[(size_t)t * ys + r] = sum;
        }
    }
#endif
}

static void gemv_job(int t, void* a) {
    (void)t; (void)a;
    const int CHUNK = 4;
    long tot = MG.first[MG.nop];
    for (;;) {
        long c = atomic_fetch_add_explicit(&MG.next, CHUNK, memory_order_relaxed);
        if (c >= tot) break;
        long z = c + CHUNK < tot ? c + CHUNK : tot;
        for (long it = c; it < z; it++) {
            int o = 0;
            while (it >= MG.first[o + 1]) o++;
            const IkOp* op = &MG.op[o];
            const Mat* m = op->m;
            long g = it - MG.first[o];
            int r = (int)g * 4;
            int rem = op->r_end - r;
            if (m->type == TY_Q8) {
                int nb = m->cols / QK_Q8;
                size_t rb = (size_t)nb * BQ8;
                const uint8_t* w = m->data + (size_t)r * rb;
                if (rem >= 4) {
                    k_q8_raw_x4(w, rb, nb, MG.T, op->xq, op->xd, op->xs, op->y + r, op->ys);
                } else {
                    for (int tt = 0; tt < MG.T; tt++) {
                        const int8_t* xt = op->xq + (size_t)tt * op->xs;
                        const float* dt = op->xd + (size_t)tt * (op->xs / QK_Q8);
                        for (int j = 0; j < rem; j++) {
                            const uint8_t* wr = w + (size_t)j * rb;
                            float sum = 0.f;
                            for (int b = 0; b < nb; b++) {
                                const uint8_t* P = wr + (size_t)b * BQ8;
                                uint16_t hs; memcpy(&hs, P, 2);
                                const int8_t* wp = (const int8_t*)(P + 2);
                                int isum = 0;
                                for (int k = 0; k < 32; k++) isum += wp[k] * xt[b * QK_Q8 + k];
                                sum += (float)isum * h2f(hs) * dt[b];
                            }
                            op->y[(size_t)tt * op->ys + r + j] = sum;
                        }
                    }
                }
            } else if (m->type == TY_Q4_0) {
                int nb = m->cols / QK_Q4;
                size_t rb = (size_t)nb * BQ4;
                const uint8_t* w = m->data + (size_t)r * rb;
                if (rem >= 4) {
                    k_q4_0_x4(w, rb, nb, MG.T, op->xq, op->xd, op->xs, op->y + r, op->ys);
                } else {
                    for (int tt = 0; tt < MG.T; tt++) {
                        const int8_t* xt = op->xq + (size_t)tt * op->xs;
                        const float* dt = op->xd + (size_t)tt * (op->xs / QK_Q8);
                        for (int j = 0; j < rem; j++) {
                            const uint8_t* wr = w + (size_t)j * rb;
                            float sum = 0.f;
                            for (int b = 0; b < nb; b++) {
                                const uint8_t* P = wr + (size_t)b * BQ4;
                                uint16_t hs; memcpy(&hs, P, 2);
                                const uint8_t* qs = P + 2;
                                int isum = 0;
                                for (int k = 0; k < 16; k++) {
                                    isum += ((int)(qs[k] & 0xF) - 8) * (int)xt[b * QK_Q4 + k]
                                          + ((int)(qs[k] >> 4)  - 8) * (int)xt[b * QK_Q4 + k + 16];
                                }
                                sum += (float)isum * h2f(hs) * dt[b];
                            }
                            op->y[(size_t)tt * op->ys + r + j] = sum;
                        }
                    }
                }
            } else {
                const float* w = (const float*)m->data + (size_t)r * m->cols;
                if (rem >= 4) k_f32x4(w, m->cols, MG.T, op->xf, op->xs, op->y + r, op->ys);
                else {
                    for (int tt = 0; tt < MG.T; tt++) {
                        for (int j = 0; j < rem; j++) {
                            const float* wr = w + (size_t)j * m->cols;
                            const float* xt = op->xf + (size_t)tt * op->xs;
                            float s = 0.f;
                            for (int k = 0; k < m->cols; k++) s += wr[k] * xt[k];
                            op->y[(size_t)tt * op->ys + r + j] = s;
                        }
                    }
                }
            }
        }
    }
}

static void run_ops(const IkOp* ops, int nop, int T) {
    MG.nop = nop; MG.T = T;
    MG.first[0] = 0;
    for (int i = 0; i < nop; i++) {
        MG.op[i] = ops[i];
        MG.first[i + 1] = MG.first[i] + (ops[i].r_end + 3) / 4;
    }
    atomic_store_explicit(&MG.next, 0, memory_order_relaxed);
    kp_run(gemv_job, NULL);
}

/* Raw NVFP4 4-row dot product (4-token & 2-token tiled for prefill/DSpark, plus fast 1-token path) */
#if IK_AVX2
static void k_nvfp4_x4(const uint8_t* w, size_t rb, int nb64, int T, const int8_t* xnv, const float* xd,
                       int xstride, float wscale, float* y, int ys) {
    const __m128i lut128 = _mm_loadu_si128((const __m128i*)fp4_vals);
    const __m256i lut32 = _mm256_broadcastsi128_si256(lut128);
    const __m256i mask0f = _mm256_set1_epi8(15);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
    int t = 0;
    for (; t + 4 <= T; t += 4) {
        const int8_t *xa = xnv + (size_t)t * xstride, *xb = xa + xstride, *xc = xb + xstride, *xe = xc + xstride;
        const float *da = xd + (size_t)t * (xstride / QK_Q8), *db = da + xstride / QK_Q8, *dc = db + xstride / QK_Q8, *de = dc + xstride / QK_Q8;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        __m256 b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        __m256 c0 = a0, c1 = a0, c2 = a0, c3 = a0;
        __m256 e0 = a0, e1 = a0, e2 = a0, e3 = a0;
        for (int b = 0; b < nb64; b++) {
            size_t o = (size_t)b * BNV;
            _mm_prefetch((const char*)(w0 + o + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(w1 + o + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(w2 + o + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(w3 + o + 256), _MM_HINT_T0);
            __m256i xalo = _mm256_loadu_si256((const __m256i*)(xa + b * QK_NV)), xahi = _mm256_loadu_si256((const __m256i*)(xa + b * QK_NV + 32));
            __m256i xblo = _mm256_loadu_si256((const __m256i*)(xb + b * QK_NV)), xbhi = _mm256_loadu_si256((const __m256i*)(xb + b * QK_NV + 32));
            __m256i xclo = _mm256_loadu_si256((const __m256i*)(xc + b * QK_NV)), xchi = _mm256_loadu_si256((const __m256i*)(xc + b * QK_NV + 32));
            __m256i xelo = _mm256_loadu_si256((const __m256i*)(xe + b * QK_NV)), xehi = _mm256_loadu_si256((const __m256i*)(xe + b * QK_NV + 32));
            __m256 vda = _mm256_set_ps(da[2*b+1], da[2*b+1], da[2*b+1], da[2*b+1], da[2*b], da[2*b], da[2*b], da[2*b]);
            __m256 vdb = _mm256_set_ps(db[2*b+1], db[2*b+1], db[2*b+1], db[2*b+1], db[2*b], db[2*b], db[2*b], db[2*b]);
            __m256 vdc = _mm256_set_ps(dc[2*b+1], dc[2*b+1], dc[2*b+1], dc[2*b+1], dc[2*b], dc[2*b], dc[2*b], dc[2*b]);
            __m256 vde = _mm256_set_ps(de[2*b+1], de[2*b+1], de[2*b+1], de[2*b+1], de[2*b], de[2*b], de[2*b], de[2*b]);
            #define NV_T(AWLO, WLO, AWHI, WHI, SW, XLO, XHI, VDX, ACC) do { \
                __m256i p_ = _mm256_madd_epi16(_mm256_add_epi16( \
                    _mm256_maddubs_epi16((AWLO), _mm256_sign_epi8((XLO), (WLO))), \
                    _mm256_maddubs_epi16((AWHI), _mm256_sign_epi8((XHI), (WHI)))), ones16); \
                ACC = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p_), _mm256_mul_ps((SW), (VDX)), ACC); \
            } while (0)
            #define NV_ROW4(WP, AA, AB, AC, AE) do { \
                const uint8_t* P_ = (WP) + o; \
                __m256i q_ = _mm256_loadu_si256((const __m256i*)(P_ + 4)); \
                __m256i wlo_ = _mm256_shuffle_epi8(lut32, _mm256_and_si256(q_, mask0f)); \
                __m256i whi_ = _mm256_shuffle_epi8(lut32, _mm256_and_si256(_mm256_srli_epi16(q_, 4), mask0f)); \
                __m256i awlo_ = _mm256_abs_epi8(wlo_), awhi_ = _mm256_abs_epi8(whi_); \
                __m256 sw_ = _mm256_set_ps(ue4m3_tab[P_[3]], ue4m3_tab[P_[3]], ue4m3_tab[P_[2]], ue4m3_tab[P_[2]], \
                                           ue4m3_tab[P_[1]], ue4m3_tab[P_[1]], ue4m3_tab[P_[0]], ue4m3_tab[P_[0]]); \
                NV_T(awlo_, wlo_, awhi_, whi_, sw_, xalo, xahi, vda, AA); \
                NV_T(awlo_, wlo_, awhi_, whi_, sw_, xblo, xbhi, vdb, AB); \
                NV_T(awlo_, wlo_, awhi_, whi_, sw_, xclo, xchi, vdc, AC); \
                NV_T(awlo_, wlo_, awhi_, whi_, sw_, xelo, xehi, vde, AE); \
            } while (0)
            NV_ROW4(w0, a0, b0, c0, e0); NV_ROW4(w1, a1, b1, c1, e1);
            NV_ROW4(w2, a2, b2, c2, e2); NV_ROW4(w3, a3, b3, c3, e3);
            #undef NV_ROW4
            #undef NV_T
        }
        float *ya = y + (size_t)t * ys, *yb = ya + ys, *yc = yb + ys, *ye = yc + ys;
        ya[0] = hsum8(a0)*wscale; ya[1] = hsum8(a1)*wscale; ya[2] = hsum8(a2)*wscale; ya[3] = hsum8(a3)*wscale;
        yb[0] = hsum8(b0)*wscale; yb[1] = hsum8(b1)*wscale; yb[2] = hsum8(b2)*wscale; yb[3] = hsum8(b3)*wscale;
        yc[0] = hsum8(c0)*wscale; yc[1] = hsum8(c1)*wscale; yc[2] = hsum8(c2)*wscale; yc[3] = hsum8(c3)*wscale;
        ye[0] = hsum8(e0)*wscale; ye[1] = hsum8(e1)*wscale; ye[2] = hsum8(e2)*wscale; ye[3] = hsum8(e3)*wscale;
    }
    for (; t < T; t++) {
        const int8_t* xa = xnv + (size_t)t * xstride;
        const float* da = xd + (size_t)t * (xstride / QK_Q8);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb64; b++) {
            size_t o = (size_t)b * BNV;
            _mm_prefetch((const char*)(w0 + o + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(w1 + o + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(w2 + o + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(w3 + o + 256), _MM_HINT_T0);
            __m256i xalo = _mm256_loadu_si256((const __m256i*)(xa + b * QK_NV));
            __m256i xahi = _mm256_loadu_si256((const __m256i*)(xa + b * QK_NV + 32));
            __m256 vda = _mm256_set_ps(da[2*b+1], da[2*b+1], da[2*b+1], da[2*b+1], da[2*b], da[2*b], da[2*b], da[2*b]);
            #define NV_ROW1(WP, ACCA) do { \
                const uint8_t* P_ = (WP) + o; \
                __m256i q_ = _mm256_loadu_si256((const __m256i*)(P_ + 4)); \
                __m256i wlo_ = _mm256_shuffle_epi8(lut32, _mm256_and_si256(q_, mask0f)); \
                __m256i whi_ = _mm256_shuffle_epi8(lut32, _mm256_and_si256(_mm256_srli_epi16(q_, 4), mask0f)); \
                __m256 sw_ = _mm256_set_ps(ue4m3_tab[P_[3]], ue4m3_tab[P_[3]], ue4m3_tab[P_[2]], ue4m3_tab[P_[2]], \
                                           ue4m3_tab[P_[1]], ue4m3_tab[P_[1]], ue4m3_tab[P_[0]], ue4m3_tab[P_[0]]); \
                __m256i pa_ = _mm256_madd_epi16(_mm256_add_epi16( \
                    _mm256_maddubs_epi16(_mm256_abs_epi8(wlo_), _mm256_sign_epi8(xalo, wlo_)), \
                    _mm256_maddubs_epi16(_mm256_abs_epi8(whi_), _mm256_sign_epi8(xahi, whi_))), ones16); \
                ACCA = _mm256_fmadd_ps(_mm256_cvtepi32_ps(pa_), _mm256_mul_ps(sw_, vda), ACCA); \
            } while (0)
            NV_ROW1(w0, a0); NV_ROW1(w1, a1); NV_ROW1(w2, a2); NV_ROW1(w3, a3);
            #undef NV_ROW1
        }
        float* ya = y + (size_t)t * ys;
        ya[0] = hsum8(a0) * wscale; ya[1] = hsum8(a1) * wscale; ya[2] = hsum8(a2) * wscale; ya[3] = hsum8(a3) * wscale;
    }
}
#else
static void k_nvfp4_x4(const uint8_t* w, size_t rb, int nb64, int T, const int8_t* xnv, const float* xd,
                       int xstride, float wscale, float* y, int ys) {
    for (int t = 0; t < T; t++) {
        const int8_t* x = xnv + (size_t)t * xstride;
        const float* d = xd + (size_t)t * (xstride / QK_Q8);
        for (int r = 0; r < 4; r++) {
            const uint8_t* wr = w + (size_t)r * rb;
            float sum = 0.f;
            for (int b = 0; b < nb64; b++) {
                const uint8_t* P = wr + (size_t)b * BNV;
                const uint8_t* qs = P + 4;
                for (int s = 0; s < 4; s++) {
                    int isum = 0;
                    for (int j = 0; j < 8; j++) {
                        uint8_t q = qs[s * 8 + j];
                        isum += fp4_vals[q & 15] * x[b * QK_NV + s * 16 + j]
                              + fp4_vals[q >> 4] * x[b * QK_NV + s * 16 + j + 8];
                    }
                    sum += (float)isum * (ue4m3_tab[P[s]] * d[b * 2 + (s >> 1)]);
                }
            }
            y[(size_t)t * ys + r] = sum * wscale;
        }
    }
}
#endif

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static void k_q5k_row(const uint8_t* wr, int nbk, int T, const void* q8k_buf, size_t q8k_stride_bytes,
                      float* y_base, int ys, int row_idx) {
    typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[16]; } blk_q8k;
    typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } blk_q5k;
#if IK_AVX2
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
#endif
    for (int t = 0; t < T; t++) {
        const blk_q8k* x = (const blk_q8k*)((const uint8_t*)q8k_buf + (size_t)t * q8k_stride_bytes);
        float sumf = 0.f;
        for (int b = 0; b < nbk; b++) {
            const blk_q5k* w = (const blk_q5k*)(wr + (size_t)b * BQ5K);
            float d = h2f(w->d) * x[b].d;
            float dmin = h2f(w->dmin) * x[b].d;
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) get_scale_min_k4(j, w->scales, &sc[j], &mn[j]);
            int sumb = 0;
            for (int j = 0; j < 8; j++) sumb += (int)mn[j] * ((int)x[b].bsums[2 * j] + (int)x[b].bsums[2 * j + 1]);
            sumf -= dmin * (float)sumb;
            int isum_all = 0;
            uint8_t hm = 1;
            for (int n = 0; n < 4; n++) {
                const uint8_t* ql = w->qs + n * 32;
                const int8_t* xp0 = x[b].qs + n * 64;
                const int8_t* xp1 = x[b].qs + n * 64 + 32;
#if IK_AVX2
                __m256i qhv = _mm256_loadu_si256((const __m256i*)w->qh);
                __m256i qlv = _mm256_loadu_si256((const __m256i*)ql);
                __m256i lo4 = _mm256_and_si256(qlv, m4);
                __m256i hi4 = _mm256_and_si256(_mm256_srli_epi16(qlv, 4), m4);
                __m256i hmask0 = _mm256_set1_epi8((char)(hm));
                __m256i hmask1 = _mm256_set1_epi8((char)(hm << 1));
                __m256i b5_0 = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, hmask0), _mm256_setzero_si256()), _mm256_set1_epi8(16));
                __m256i b5_1 = _mm256_andnot_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qhv, hmask1), _mm256_setzero_si256()), _mm256_set1_epi8(16));
                __m256i w0 = _mm256_or_si256(lo4, b5_0);
                __m256i w1 = _mm256_or_si256(hi4, b5_1);
                __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, _mm256_loadu_si256((const __m256i*)xp0)), ones16);
                __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, _mm256_loadu_si256((const __m256i*)xp1)), ones16);
                __m128i s0_4 = _mm_add_epi32(_mm256_castsi256_si128(p0), _mm256_extracti128_si256(p0, 1));
                s0_4 = _mm_hadd_epi32(s0_4, s0_4); s0_4 = _mm_hadd_epi32(s0_4, s0_4);
                __m128i s1_4 = _mm_add_epi32(_mm256_castsi256_si128(p1), _mm256_extracti128_si256(p1, 1));
                s1_4 = _mm_hadd_epi32(s1_4, s1_4); s1_4 = _mm_hadd_epi32(s1_4, s1_4);
                isum_all += (int)sc[2 * n] * _mm_cvtsi128_si32(s0_4) + (int)sc[2 * n + 1] * _mm_cvtsi128_si32(s1_4);
#else
                int d0 = 0, d1 = 0;
                for (int l = 0; l < 32; l++) {
                    d0 += ((ql[l] & 15) + ((w->qh[l] & hm) ? 16 : 0)) * xp0[l];
                    d1 += ((ql[l] >> 4) + ((w->qh[l] & (hm << 1)) ? 16 : 0)) * xp1[l];
                }
                isum_all += (int)sc[2 * n] * d0 + (int)sc[2 * n + 1] * d1;
#endif
                hm <<= 2;
            }
            sumf += d * (float)isum_all;
        }
        y_base[(size_t)t * ys + row_idx] = sumf;
    }
}

#if IK_AVX2
static inline __m256i ik_bytes_from_mask32(uint32_t m) {
    const __m256i shuf = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL,
                                           0x0101010101010101LL, 0x0000000000000000LL);
    const __m256i bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)m), shuf);
    __m256i is_set = _mm256_cmpeq_epi8(_mm256_and_si256(v, bits), bits);
    return _mm256_or_si256(is_set, _mm256_set1_epi8(1));
}
#endif

static void k_iq2xxs128_x4(const uint8_t* w, size_t rb, int nb128, int T, const int8_t* xq, const float* xd,
                           int xstride, float* y, int ys) {
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
#if IK_AVX2
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int t = 0; t < T; t++) {
        const int8_t* xt = xq + (size_t)t * xstride;
        const float*  dt = xd + (size_t)t * (xstride / QK_Q8);
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        const uint8_t* wps[4] = {w0, w1, w2, w3};
        float accs[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < nb128; b++) {
            size_t o = (size_t)b * BIQ2_XXS;
            for (int r = 0; r < 4; r++) {
                const uint8_t* P = wps[r] + o;
                uint16_t hd; memcpy(&hd, P, 2);
                float d = _cvtsh_ss(hd) * 0.25f;
                float rsum = 0.f;
                for (int ib = 0; ib < 4; ib++) {
                    uint32_t ax0, ax1;
                    memcpy(&ax0, P + 2 + 8 * ib, 4);
                    memcpy(&ax1, P + 2 + 8 * ib + 4, 4);
                    __m256i w_u8 = _mm256_set_epi64x(
                        (long long)ik_iq2xxs_grid[(ax0 >> 24) & 255u],
                        (long long)ik_iq2xxs_grid[(ax0 >> 16) & 255u],
                        (long long)ik_iq2xxs_grid[(ax0 >> 8)  & 255u],
                        (long long)ik_iq2xxs_grid[ax0 & 255u]);
                    uint32_t sm = (uint32_t)ik_ksigns_iq2xs[ax1 & 127u]
                                | ((uint32_t)ik_ksigns_iq2xs[(ax1 >> 7)  & 127u] << 8)
                                | ((uint32_t)ik_ksigns_iq2xs[(ax1 >> 14) & 127u] << 16)
                                | ((uint32_t)ik_ksigns_iq2xs[(ax1 >> 21) & 127u] << 24);
                    __m256i xa = _mm256_loadu_si256((const __m256i*)(xt + b * QK_IQ2 + ib * 32));
                    __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w_u8, _mm256_sign_epi8(xa, ik_bytes_from_mask32(sm))), ones16);
                    __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(p32), _mm256_extracti128_si256(p32, 1));
                    s128 = _mm_hadd_epi32(s128, s128); s128 = _mm_hadd_epi32(s128, s128);
                    int isum = _mm_cvtsi128_si32(s128);
                    rsum += (float)isum * (0.5f + (float)(ax1 >> 28)) * dt[b * 4 + ib];
                }
                accs[r] += d * rsum;
            }
        }
        (void)a0; (void)a1; (void)a2; (void)a3;
        float* ya = y + (size_t)t * ys;
        ya[0] = accs[0]; ya[1] = accs[1]; ya[2] = accs[2]; ya[3] = accs[3];
    }
#elif IK_NEON
#if IK_MMLA
    if (T >= 2) { (void)w0; (void)w1; (void)w2; (void)w3; k_iq2xxs128_x4_mm(w, rb, nb128, T, xq, xd, xstride, y, ys); return; }
#endif
    /* The 4 rows sit in the lanes of one float accumulator per token: per 32 weights the 4 row sums come out of
     * two pairwise adds and take one vector FMA, instead of a horizontal add and a scalar FMA per row. Each lane
     * keeps the plain C order over (b, ib) and the same fused multiply-add, so the result is bit-identical. */
    const int xds = xstride / QK_Q8;
    const uint8_t* wps[4] = {w0, w1, w2, w3};
    for (int t0 = 0; t0 < T; t0 += IK_IQ2_TT) {
        const int nt = T - t0 < IK_IQ2_TT ? T - t0 : IK_IQ2_TT;
        float32x4_t s[IK_IQ2_TT];
        for (int i = 0; i < nt; i++) s[i] = vdupq_n_f32(0.f);
        for (int b = 0; b < nb128; b++) {
            const uint8_t* P[4];
            float d[4];
            for (int r = 0; r < 4; r++) {
                P[r] = wps[r] + (size_t)b * BIQ2_XXS;
                __builtin_prefetch(P[r] + 4 * BIQ2_XXS);
                uint16_t hd; memcpy(&hd, P[r], 2);
                d[r] = h2f(hd) * 0.25f;
            }
            for (int ib = 0; ib < 4; ib++) {
                int8x16_t g0[4], g1[4];
                float dl[4];
                for (int r = 0; r < 4; r++) {
                    uint32_t ax0, ax1;
                    memcpy(&ax0, P[r] + 2 + 8 * ib, 4);
                    memcpy(&ax1, P[r] + 2 + 8 * ib + 4, 4);
                    g0[r] = ik_signs(ik_grid2(ik_iq2xxs_grid[ax0 & 255u], ik_iq2xxs_grid[(ax0 >> 8) & 255u]),
                                     ax1 & 127u, (ax1 >> 7) & 127u);
                    g1[r] = ik_signs(ik_grid2(ik_iq2xxs_grid[(ax0 >> 16) & 255u], ik_iq2xxs_grid[ax0 >> 24]),
                                     (ax1 >> 14) & 127u, (ax1 >> 21) & 127u);
                    dl[r] = d[r] * (0.5f + (float)(ax1 >> 28));
                }
                const float32x4_t dlv = vld1q_f32(dl);
                for (int i = 0; i < nt; i++) {
                    const int8_t* xp = xq + (size_t)(t0 + i) * xstride + b * QK_IQ2 + ib * 32;
                    const int8x16_t xa = vld1q_s8(xp), xb = vld1q_s8(xp + 16);
                    const int32x4_t z = vdupq_n_s32(0);
                    int32x4_t a0 = ik_dot16(ik_dot16(z, g0[0], xa), g1[0], xb);
                    int32x4_t a1 = ik_dot16(ik_dot16(z, g0[1], xa), g1[1], xb);
                    int32x4_t a2 = ik_dot16(ik_dot16(z, g0[2], xa), g1[2], xb);
                    int32x4_t a3 = ik_dot16(ik_dot16(z, g0[3], xa), g1[3], xb);
                    int32x4_t rows = vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3));   /* [r0 r1 r2 r3] */
                    float32x4_t db = vmulq_n_f32(dlv, xd[(size_t)(t0 + i) * xds + b * 4 + ib]);
                    s[i] = vfmaq_f32(s[i], vcvtq_f32_s32(rows), db);
                }
            }
        }
        for (int i = 0; i < nt; i++) vst1q_f32(y + (size_t)(t0 + i) * ys, s[i]);
    }
#else
    for (int t = 0; t < T; t++) {
        const int8_t* xt = xq + (size_t)t * xstride;
        const float*  dt = xd + (size_t)t * (xstride / QK_Q8);
        const uint8_t* wps[4] = {w0, w1, w2, w3};
        for (int r = 0; r < 4; r++) {
            const uint8_t* wr = wps[r];
            float sum = 0.f;
            for (int b = 0; b < nb128; b++) {
                const uint8_t* P = wr + (size_t)b * BIQ2_XXS;
                uint16_t hd; memcpy(&hd, P, 2);
                float d = h2f(hd) * 0.25f;
                for (int ib = 0; ib < 4; ib++) {
                    uint32_t ax0, ax1;
                    memcpy(&ax0, P + 2 + 8 * ib, 4);
                    memcpy(&ax1, P + 2 + 8 * ib + 4, 4);
                    float db = d * (0.5f + (float)(ax1 >> 28)) * dt[b * 4 + ib];
                    int isum = 0;
                    for (int l = 0; l < 4; l++) {
                        const uint8_t* gp = (const uint8_t*)(ik_iq2xxs_grid + ((ax0 >> (8 * l)) & 255u));
                        uint8_t sg = ik_ksigns_iq2xs[(ax1 >> (7 * l)) & 127u];
                        const int8_t* xp = xt + b * QK_IQ2 + ib * 32 + l * 8;
                        for (int j = 0; j < 8; j++)
                            isum += (int)gp[j] * ((sg & (1u << j)) ? -(int)xp[j] : (int)xp[j]);
                    }
                    sum += (float)isum * db;
                }
            }
            y[(size_t)t * ys + r] = sum;
        }
    }
#endif
}

static void k_iq2xs128_x4(const uint8_t* w, size_t rb, int nb128, int T, const int8_t* xq, const float* xd,
                          int xstride, float* y, int ys) {
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
#if IK_AVX2
    const __m128i ones128 = _mm_set1_epi16(1);
    for (int t = 0; t < T; t++) {
        const int8_t* xt = xq + (size_t)t * xstride;
        const float*  dt = xd + (size_t)t * (xstride / QK_Q8);
        const uint8_t* wps[4] = {w0, w1, w2, w3};
        float accs[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < nb128; b++) {
            size_t o = (size_t)b * BIQ2_XS;
            for (int r = 0; r < 4; r++) {
                const uint8_t* P = wps[r] + o;
                uint16_t hd; memcpy(&hd, P, 2);
                float d = _cvtsh_ss(hd) * 0.25f;
                const uint16_t* qs = (const uint16_t*)(P + 2);
                const uint8_t* sc = P + 34;
                float rsum = 0.f;
                for (int ib = 0; ib < 4; ib++) {
                    uint16_t q0 = qs[4*ib+0], q1 = qs[4*ib+1], q2 = qs[4*ib+2], q3 = qs[4*ib+3];
                    __m256i w_u8 = _mm256_set_epi64x(
                        (long long)ik_iq2xs_grid[q3 & 511u],
                        (long long)ik_iq2xs_grid[q2 & 511u],
                        (long long)ik_iq2xs_grid[q1 & 511u],
                        (long long)ik_iq2xs_grid[q0 & 511u]);
                    uint32_t sm = (uint32_t)ik_ksigns_iq2xs[q0 >> 9]
                                | ((uint32_t)ik_ksigns_iq2xs[q1 >> 9] << 8)
                                | ((uint32_t)ik_ksigns_iq2xs[q2 >> 9] << 16)
                                | ((uint32_t)ik_ksigns_iq2xs[q3 >> 9] << 24);
                    __m256i xa = _mm256_loadu_si256((const __m256i*)(xt + b * QK_IQ2 + ib * 32));
                    __m256i p16 = _mm256_maddubs_epi16(w_u8, _mm256_sign_epi8(xa, ik_bytes_from_mask32(sm)));
                    __m128i lo = _mm_madd_epi16(_mm256_castsi256_si128(p16), ones128);
                    __m128i hi = _mm_madd_epi16(_mm256_extracti128_si256(p16, 1), ones128);
                    lo = _mm_hadd_epi32(lo, lo); lo = _mm_hadd_epi32(lo, lo);
                    hi = _mm_hadd_epi32(hi, hi); hi = _mm_hadd_epi32(hi, hi);
                    int sum0 = _mm_cvtsi128_si32(lo);
                    int sum1 = _mm_cvtsi128_si32(hi);
                    uint8_t s8 = sc[ib];
                    float xd_v = dt[b * 4 + ib];
                    rsum += ((float)sum0 * (0.5f + (float)(s8 & 15)) + (float)sum1 * (0.5f + (float)(s8 >> 4))) * xd_v;
                }
                accs[r] += d * rsum;
            }
        }
        float* ya = y + (size_t)t * ys;
        ya[0] = accs[0]; ya[1] = accs[1]; ya[2] = accs[2]; ya[3] = accs[3];
    }
#elif IK_NEON
#if IK_MMLA
    if (T >= 2) { (void)w0; (void)w1; (void)w2; (void)w3; k_iq2xs128_x4_mm(w, rb, nb128, T, xq, xd, xstride, y, ys); return; }
#endif
    /* rows in vector lanes (see k_iq2xxs128_x4); the per-8 sums l0..l3 of the 4 rows are regrouped with
     * pairwise adds and unzips, then added in the plain C order l0, l1 (low-nibble scale), l2, l3 (high) */
    const int xds = xstride / QK_Q8;
    const uint8_t* wps[4] = {w0, w1, w2, w3};
    for (int t0 = 0; t0 < T; t0 += IK_IQ2_TT) {
        const int nt = T - t0 < IK_IQ2_TT ? T - t0 : IK_IQ2_TT;
        float32x4_t s[IK_IQ2_TT];
        for (int i = 0; i < nt; i++) s[i] = vdupq_n_f32(0.f);
        for (int b = 0; b < nb128; b++) {
            const uint8_t* P[4];
            float d[4];
            for (int r = 0; r < 4; r++) {
                P[r] = wps[r] + (size_t)b * BIQ2_XS;
                __builtin_prefetch(P[r] + 4 * BIQ2_XS);
                uint16_t hd; memcpy(&hd, P[r], 2);
                d[r] = h2f(hd) * 0.25f;
            }
            for (int ib = 0; ib < 4; ib++) {
                int8x16_t g0[4], g1[4];
                float dl0[4], dl1[4];
                for (int r = 0; r < 4; r++) {
                    uint16_t q[4];
                    memcpy(q, P[r] + 2 + 8 * ib, 8);
                    g0[r] = ik_signs(ik_grid2(ik_iq2xs_grid[q[0] & 511u], ik_iq2xs_grid[q[1] & 511u]), q[0] >> 9, q[1] >> 9);
                    g1[r] = ik_signs(ik_grid2(ik_iq2xs_grid[q[2] & 511u], ik_iq2xs_grid[q[3] & 511u]), q[2] >> 9, q[3] >> 9);
                    const uint8_t sc = P[r][34 + ib];
                    dl0[r] = d[r] * (0.5f + (float)(sc & 15));
                    dl1[r] = d[r] * (0.5f + (float)(sc >> 4));
                }
                const float32x4_t dl0v = vld1q_f32(dl0), dl1v = vld1q_f32(dl1);
                for (int i = 0; i < nt; i++) {
                    const int8_t* xp = xq + (size_t)(t0 + i) * xstride + b * QK_IQ2 + ib * 32;
                    const int8x16_t xa = vld1q_s8(xp), xb = vld1q_s8(xp + 16);
                    const int32x4_t z = vdupq_n_s32(0);
                    /* lane layout of each: [l_a, l_a, l_b, l_b] partial sums (l0/l1 from xa, l2/l3 from xb) */
                    int32x4_t lo01 = vpaddq_s32(ik_dot16(z, g0[0], xa), ik_dot16(z, g0[1], xa));   /* r0l0 r0l1 r1l0 r1l1 */
                    int32x4_t lo23 = vpaddq_s32(ik_dot16(z, g0[2], xa), ik_dot16(z, g0[3], xa));   /* r2l0 r2l1 r3l0 r3l1 */
                    int32x4_t hi01 = vpaddq_s32(ik_dot16(z, g1[0], xb), ik_dot16(z, g1[1], xb));
                    int32x4_t hi23 = vpaddq_s32(ik_dot16(z, g1[2], xb), ik_dot16(z, g1[3], xb));
                    const float xdv = xd[(size_t)(t0 + i) * xds + b * 4 + ib];
                    const float32x4_t db0 = vmulq_n_f32(dl0v, xdv), db1 = vmulq_n_f32(dl1v, xdv);
                    float32x4_t a = s[i];
                    a = vfmaq_f32(a, vcvtq_f32_s32(vuzp1q_s32(lo01, lo23)), db0);   /* l0 of r0..r3 */
                    a = vfmaq_f32(a, vcvtq_f32_s32(vuzp2q_s32(lo01, lo23)), db0);   /* l1 */
                    a = vfmaq_f32(a, vcvtq_f32_s32(vuzp1q_s32(hi01, hi23)), db1);   /* l2 */
                    a = vfmaq_f32(a, vcvtq_f32_s32(vuzp2q_s32(hi01, hi23)), db1);   /* l3 */
                    s[i] = a;
                }
            }
        }
        for (int i = 0; i < nt; i++) vst1q_f32(y + (size_t)(t0 + i) * ys, s[i]);
    }
#else
    for (int t = 0; t < T; t++) {
        const int8_t* xt = xq + (size_t)t * xstride;
        const float*  dt = xd + (size_t)t * (xstride / QK_Q8);
        const uint8_t* wps[4] = {w0, w1, w2, w3};
        for (int r = 0; r < 4; r++) {
            const uint8_t* wr = wps[r];
            float sum = 0.f;
            for (int b = 0; b < nb128; b++) {
                const uint8_t* P = wr + (size_t)b * BIQ2_XS;
                uint16_t hd; memcpy(&hd, P, 2);
                float d = h2f(hd) * 0.25f;
                const uint16_t* qs = (const uint16_t*)(P + 2);
                const uint8_t* sc = P + 34;
                for (int ib = 0; ib < 4; ib++) {
                    float db0 = d * (0.5f + (float)(sc[ib] & 15)) * dt[b * 4 + ib];
                    float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * dt[b * 4 + ib];
                    for (int l = 0; l < 4; l++) {
                        uint16_t q = qs[4 * ib + l];
                        const uint8_t* gp = (const uint8_t*)(ik_iq2xs_grid + (q & 511u));
                        uint8_t sg = ik_ksigns_iq2xs[q >> 9];
                        const int8_t* xp = xt + b * QK_IQ2 + ib * 32 + l * 8;
                        int isum = 0;
                        for (int j = 0; j < 8; j++)
                            isum += (int)gp[j] * ((sg & (1u << j)) ? -(int)xp[j] : (int)xp[j]);
                        sum += (float)isum * ((l < 2) ? db0 : db1);
                    }
                }
            }
            y[(size_t)t * ys + r] = sum;
        }
    }
#endif
}

typedef struct {
    int enabled;
    double in0[IK_NL_PHYS][IK_D];
    double in1[IK_NL_PHYS][IK_D];
    double in2[IK_NL_PHYS][IK_D];
    double attn[IK_NL_PHYS][IK_QDIM];
    double exp_gu[IK_NL_PHYS][IK_NE][IK_D];
    double exp_dn[IK_NL_PHYS][IK_NE][IK_FF];
    double head[IK_D];
    uint64_t cnt_dense[IK_NL_PHYS];
    uint64_t cnt_exp[IK_NL_PHYS][IK_NE];
    uint64_t cnt_head;
} IkImatrix;

static IkImatrix* g_imat = NULL;

static void imat_accum(double* acc, const float* x, int n, int T) {
    for (int t = 0; t < T; t++) {
        const float* p = x + (size_t)t * n;
        for (int i = 0; i < n; i++) {
            double v = (double)p[i];
            acc[i] += v * v;
        }
    }
}

/* ---------------------------------------------------------------- Model State */
typedef struct {
    int is_swa;
    Mat wq, wk, wv, wo, wgate, wrel;
    float *sink_proj, *q_clip, *rel_bias, *rel_clip;
    float *k_conv, *v_conv, *attn_post_conv, *ffn_post_conv;
    float *attn_norm, *q_inp_norm, *k_inp_norm, *v_inp_norm;
    float *k_conv_proj, *v_conv_proj, *attn_res_q;
    float *ffn_res_q, *ffn_norm;
    Mat ffn_gate_inp;
    float *exp_bias;
    int exp_gu_type, exp_dn_type;
    const uint8_t* exp_gu[IK_NE];
    const uint8_t* exp_dn[IK_NE];
    float exp_gu_scale[IK_NE], exp_dn_scale[IK_NE];
    int has_exp_scale;
    int has_loop;
    float* loop_msg_q;
    Mat loop_down[4], loop_gain[4], loop_forget[4];
} PhysLayer;

typedef struct {
    int fd;
    uint8_t* map;
    size_t map_sz;
    size_t data_start;
    int ctx;
    const uint8_t* tok_embd;
    float* tok_norm;
    Mat output_head;
    float* output_norm;
    float* output_res_q;
    PhysLayer L[IK_NL_PHYS];

    uint16_t* kv_k[IK_NL_VIRT];
    uint16_t* kv_v[IK_NL_VIRT];
    int kv_cap[IK_NL_VIRT];
    float* conv_hist;
    float* conv_trace;
    int trace_active;

    int max_t;
    float *buf_cur, *buf_pre, *buf_blk, *buf_blk_pre, *buf_bank, *buf_msg;
    float *buf_in0, *buf_in1, *buf_in2, *buf_inpsa, *buf_ffninp;
    float *buf_q, *buf_k, *buf_v, *buf_g, *buf_r, *buf_sinks, *buf_kcp, *buf_vcp;
    float *buf_attn, *buf_o, *buf_ffnin, *buf_probs, *buf_exps_out;
    float *buf_ctrl_in, *buf_ctrl_g, *buf_ctrl_gn, *buf_ctrl_fg;
    int8_t *buf_xq, *buf_xnv;
    float *buf_xd;
    void *buf_xq8k;

    uint64_t exp_counts[IK_NL_PHYS][IK_NE];
    double timers[6]; /* 0:emb, 1:attn/conv, 2:dense, 3:moe, 4:head, 5:total */
    IkGpuHooks gpu;
    int dspark_n;
    int dspark_ids[4];
    float* dspark_feat;
} IkModel;

static IkModel M;

static int layer_phys_idx(int il) {
    if (il < IK_NL_PHYS) return il;
    int off = il - IK_NL_PHYS;
    return IK_CORE_START + (off % (IK_CORE_END - IK_CORE_START));
}

/* ---------------------------------------------------------------- Mix, Control, Clip, Convolve */
static void do_mix(const float* const* blks, int nblk, const float* cur, const float* query, int nq, int T, float* out) {
    int ns = nblk + 1;
    float inv_sqrt_d = 1.0f / sqrtf((float)IK_D);
    float normed[8][IK_D];
    float scores[8];
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < nblk; s++) rmsnorm_f32(normed[s], blks[s] + (size_t)t * IK_D, NULL, IK_D);
        rmsnorm_f32(normed[nblk], cur + (size_t)t * IK_D, NULL, IK_D);
        for (int q = 0; q < nq; q++) {
            const float* wq = query + (size_t)q * IK_D;
            float mx = -1e30f;
            for (int s = 0; s < ns; s++) {
                float d = 0.f;
#if IK_AVX2
                __m256 a0 = _mm256_setzero_ps(), a1 = a0;
                for (int i = 0; i < IK_D; i += 16) {
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(normed[s] + i), _mm256_loadu_ps(wq + i), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(normed[s] + i + 8), _mm256_loadu_ps(wq + i + 8), a1);
                }
                d = hsum8(_mm256_add_ps(a0, a1));
#elif IK_NEON
                d = ik_dotf(normed[s], wq, IK_D);
#else
                for (int i = 0; i < IK_D; i++) d += normed[s][i] * wq[i];
#endif
                scores[s] = d * inv_sqrt_d;
                if (scores[s] > mx) mx = scores[s];
            }
            float sum = 0.f;
            for (int s = 0; s < ns; s++) { scores[s] = expf(scores[s] - mx); sum += scores[s]; }
            float inv = 1.f / sum;
            for (int s = 0; s < ns; s++) scores[s] *= inv;
            float* dst = out + ((size_t)t * nq + q) * IK_D;
            memset(dst, 0, sizeof(float) * IK_D);
            for (int s = 0; s < ns; s++) {
                float w = scores[s];
                for (int i = 0; i < IK_D; i++) dst[i] += w * normed[s][i];
            }
        }
    }
}

static void do_control(float* cur, const float* msg, int stride_msg, int T, const PhysLayer* pl, int ctrl) {
    for (int t = 0; t < T; t++) {
        rmsnorm_f32(M.buf_ctrl_in + (size_t)t * (2 * IK_D), cur + (size_t)t * IK_D, NULL, IK_D);
        rmsnorm_f32(M.buf_ctrl_in + (size_t)t * (2 * IK_D) + IK_D, msg + (size_t)t * stride_msg, NULL, IK_D);
    }
    if (pl->loop_down[ctrl].type == TY_Q8) {
        for (int t = 0; t < T; t++)
            q8_quantize(M.buf_ctrl_in + (size_t)t * (2 * IK_D), 2 * IK_D,
                        M.buf_xq + (size_t)t * (2 * IK_D),
                        M.buf_xd + (size_t)t * ((2 * IK_D) / QK_Q8));
        IkOp ops[1] = {{&pl->loop_down[ctrl], IK_GRANK, NULL, M.buf_xq, M.buf_xd, 2 * IK_D, M.buf_ctrl_g, IK_GRANK}};
        run_ops(ops, 1, T);
    } else {
        IkOp ops[1] = {{&pl->loop_down[ctrl], IK_GRANK, M.buf_ctrl_in, NULL, NULL, 2 * IK_D, M.buf_ctrl_g, IK_GRANK}};
        run_ops(ops, 1, T);
    }
    if (pl->loop_gain[ctrl].type == TY_Q8) {
        for (int t = 0; t < T; t++)
            q8_quantize(M.buf_ctrl_g + (size_t)t * IK_GRANK, IK_GRANK,
                        M.buf_xq + (size_t)t * IK_GRANK,
                        M.buf_xd + (size_t)t * (IK_GRANK / QK_Q8));
        IkOp ops2[2] = {
            {&pl->loop_gain[ctrl],   IK_D, NULL, M.buf_xq, M.buf_xd, IK_GRANK, M.buf_ctrl_gn, IK_D},
            {&pl->loop_forget[ctrl], IK_D, NULL, M.buf_xq, M.buf_xd, IK_GRANK, M.buf_ctrl_fg, IK_D},
        };
        run_ops(ops2, 2, T);
    } else {
        IkOp ops2[2] = {
            {&pl->loop_gain[ctrl],   IK_D, M.buf_ctrl_g, NULL, NULL, IK_GRANK, M.buf_ctrl_gn, IK_D},
            {&pl->loop_forget[ctrl], IK_D, M.buf_ctrl_g, NULL, NULL, IK_GRANK, M.buf_ctrl_fg, IK_D},
        };
        run_ops(ops2, 2, T);
    }
    for (int t = 0; t < T; t++) {
        float* c = cur + (size_t)t * IK_D;
        const float* m = msg + (size_t)t * stride_msg;
        const float* gn = M.buf_ctrl_gn + (size_t)t * IK_D;
        const float* fg = M.buf_ctrl_fg + (size_t)t * IK_D;
        for (int i = 0; i < IK_D; i++)
            c[i] = (1.f - tanhf(fg[i])) * c[i] + tanhf(gn[i]) * m[i];
    }
}

static void do_clip(float* v, int dim, int nvec, const float* scale, float thr) {
    for (int h = 0; h < nvec; h++) {
        float* p = v + (size_t)h * dim;
        if (scale) {
            const float* sc = scale + (size_t)h * dim;
            for (int i = 0; i < dim; i++) p[i] *= sc[i];
        }
        float ss = 0.f;
        for (int i = 0; i < dim; i++) ss += p[i] * p[i];
        float norm = sqrtf(ss) + 1e-6f;
        float fac = thr / norm;
        if (fac < 1.f) {
            for (int i = 0; i < dim; i++) p[i] *= fac;
        }
    }
}

static void do_convolve(float* cur, int C, int T, const float* weight, const float* dyn,
                        float* hist_slot, float* trace_slot, int ch_off) {
    float* h0 = hist_slot + ch_off;
    float* h1 = hist_slot + IK_CHSCONV + ch_off;
    float* h2 = hist_slot + 2 * IK_CHSCONV + ch_off;
    float sc = dyn ? (1.0f / 3.0f) : (1.0f / sqrtf(5.0f));
    int n_dyn_heads = dyn ? (C / IK_HD) : 0;

    if (trace_slot && T <= 8) {
        memcpy(trace_slot + 0 * IK_CHSCONV + ch_off, h0, (size_t)C * sizeof(float));
        memcpy(trace_slot + 1 * IK_CHSCONV + ch_off, h1, (size_t)C * sizeof(float));
        memcpy(trace_slot + 2 * IK_CHSCONV + ch_off, h2, (size_t)C * sizeof(float));
        for (int t = 0; t < T; t++) {
            memcpy(trace_slot + (size_t)(3 + t) * IK_CHSCONV + ch_off, cur + (size_t)t * C, (size_t)C * sizeof(float));
        }
    }

    if (T == 1) {
        if (!dyn) {
            for (int c = 0; c < C; c++) {
                float x0 = h0[c], x1 = h1[c], x2 = h2[c], x3 = cur[c];
                const float* w = weight + c * 4;
                float o = x0 * w[0] + x1 * w[1] + x2 * w[2] + x3 * w[3];
                h0[c] = x1; h1[c] = x2; h2[c] = x3;
                cur[c] = (x3 + o) * sc;
            }
        } else {
            for (int h = 0; h < n_dyn_heads; h++) {
                const float* dt = dyn + h * 4;
                float d0 = dt[0], d1 = dt[1], d2 = dt[2], d3 = dt[3];
                for (int d = 0; d < IK_HD; d++) {
                    int c = h * IK_HD + d;
                    float x0 = h0[c], x1 = h1[c], x2 = h2[c], x3 = cur[c];
                    const float* w = weight + c * 4;
                    float o = x0 * w[0] + x1 * w[1] + x2 * w[2] + x3 * w[3]
                            + x3 * d0 + x2 * d1 + x1 * d2 + x0 * d3;
                    h0[c] = x1; h1[c] = x2; h2[c] = x3;
                    cur[c] = (x3 + o) * sc;
                }
            }
        }
        return;
    }

    for (int c = 0; c < C; c++) {
        float x0 = h0[c], x1 = h1[c], x2 = h2[c];
        const float* w = weight + c * 4;
        int h = dyn ? (c / IK_HD) : 0;
        for (int t = 0; t < T; t++) {
            float x3 = cur[(size_t)t * C + c];
            float o = x0 * w[0] + x1 * w[1] + x2 * w[2] + x3 * w[3];
            if (dyn) {
                const float* dt = dyn + ((size_t)t * n_dyn_heads + h) * 4;
                o += x3 * dt[0] + x2 * dt[1] + x1 * dt[2] + x0 * dt[3];
            }
            cur[(size_t)t * C + c] = (x3 + o) * sc;
            x0 = x1; x1 = x2; x2 = x3;
        }
        h0[c] = x0; h1[c] = x1; h2[c] = x2;
    }
}

/* ---------------------------------------------------------------- Attention */
typedef struct {
    int il, is_swa, pos0, T;
    const float *Q, *R, *sinks, *rel_bias;
    const uint16_t *k_cache, *v_cache;
    int cap;
    float *out;
} AttnJob;

static struct {
    AttnJob j;
    _Atomic int next_h;
} AJ;

static void attn_worker(int tid, void* arg) {
    (void)tid; (void)arg;
    const AttnJob* J = &AJ.j;
    const float inv_s = 1.0f / sqrtf((float)IK_HD);
    const int kpg = IK_NH / IK_NKV;
    for (;;) {
        int h = atomic_fetch_add_explicit(&AJ.next_h, 1, memory_order_relaxed);
        if (h >= IK_NH) break;
        int hkv = h / kpg;
        float table[IK_RELEXT + 1];
        for (int t = 0; t < J->T; t++) {
            int pq = J->pos0 + t;
            const float* Qh = J->Q + ((size_t)t * IK_NH + h) * IK_HD;
            const float* Rh = J->R + ((size_t)t * IK_NH + h) * IK_NREL;
            float sink = J->sinks[(size_t)t * IK_NH + h];
            int k_start = J->is_swa ? (pq + 1 > IK_SWA ? pq - IK_SWA + 1 : 0) : 0;
            /* only the distances this query reaches (0 .. pq - k_start, capped) need a table entry */
            int dmax = pq - k_start < IK_RELEXT ? pq - k_start : IK_RELEXT - 1;
            for (int d = 0; d <= dmax; d++) {
                const float* rb = J->rel_bias + (size_t)d * IK_NREL;
#if IK_NEON
                float s = vaddvq_f32(vaddq_f32(
                    vfmaq_f32(vmulq_f32(vld1q_f32(rb), vld1q_f32(Rh)), vld1q_f32(rb + 8), vld1q_f32(Rh + 8)),
                    vfmaq_f32(vmulq_f32(vld1q_f32(rb + 4), vld1q_f32(Rh + 4)), vld1q_f32(rb + 12), vld1q_f32(Rh + 12))));
#else
                float s = 0.f;
                for (int k = 0; k < IK_NREL; k++) s += rb[k] * Rh[k];
#endif
                table[d] = s - sink;
            }
            table[IK_RELEXT] = -sink;

            float acc[IK_HD]; memset(acc, 0, sizeof(acc));
            float m = 0.0f;
            float sum = 1.0f; /* implicit sink logit = 0.0f */
            for (int pk = k_start; pk <= pq; pk++) {
                int slot = J->is_swa ? (pk % J->cap) : pk;
                const uint16_t* Kp = J->k_cache + ((size_t)slot * IK_NKV + hkv) * IK_HD;
                const uint16_t* Vp = J->v_cache + ((size_t)slot * IK_NKV + hkv) * IK_HD;
                float dot = 0.f;
#if IK_AVX2
                __m256 d0 = _mm256_setzero_ps(), d1 = d0;
                for (int i = 0; i < IK_HD; i += 16) {
                    __m256 k0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(Kp + i)));
                    __m256 k1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(Kp + i + 8)));
                    d0 = _mm256_fmadd_ps(_mm256_loadu_ps(Qh + i), k0, d0);
                    d1 = _mm256_fmadd_ps(_mm256_loadu_ps(Qh + i + 8), k1, d1);
                }
                dot = hsum8(_mm256_add_ps(d0, d1));
#elif IK_NEON
                float32x4_t d0 = vdupq_n_f32(0.f), d1 = d0;
                for (int i = 0; i < IK_HD; i += 8) {
                    float16x8_t kh = vld1q_f16((const __fp16*)(Kp + i));
                    d0 = vfmaq_f32(d0, vld1q_f32(Qh + i), vcvt_f32_f16(vget_low_f16(kh)));
                    d1 = vfmaq_f32(d1, vld1q_f32(Qh + i + 4), vcvt_high_f32_f16(kh));
                }
                dot = vaddvq_f32(vaddq_f32(d0, d1));
#else
                for (int i = 0; i < IK_HD; i++) dot += Qh[i] * h2f(Kp[i]);
#endif
                int dist = pq - pk;
                if (dist > IK_RELEXT) dist = IK_RELEXT;
                float sc = dot * inv_s + table[dist];
                float w;
                if (sc > m) {
                    float f = expf(m - sc);
                    sum = sum * f + 1.0f;
                    m = sc;
                    w = 1.0f;
                    for (int i = 0; i < IK_HD; i++) acc[i] *= f;
                } else {
                    w = expf(sc - m);
                    sum += w;
                }
#if IK_AVX2
                __m256 vw = _mm256_set1_ps(w);
                for (int i = 0; i < IK_HD; i += 8) {
                    __m256 vv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(Vp + i)));
                    __m256 va = _mm256_loadu_ps(acc + i);
                    _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(vv, vw, va));
                }
#elif IK_NEON
                for (int i = 0; i < IK_HD; i += 8) {
                    float16x8_t vh = vld1q_f16((const __fp16*)(Vp + i));
                    vst1q_f32(acc + i, vfmaq_n_f32(vld1q_f32(acc + i), vcvt_f32_f16(vget_low_f16(vh)), w));
                    vst1q_f32(acc + i + 4, vfmaq_n_f32(vld1q_f32(acc + i + 4), vcvt_high_f32_f16(vh), w));
                }
#else
                for (int i = 0; i < IK_HD; i++) acc[i] += w * h2f(Vp[i]);
#endif
            }
            float inv = 1.0f / sum;
            float* dst = J->out + ((size_t)t * IK_NH + h) * IK_HD;
            for (int i = 0; i < IK_HD; i++) dst[i] = acc[i] * inv;
        }
    }
}

/* ---------------------------------------------------------------- MoE Expert Engine */
#define MOE_GU_ROWS 16
#define MOE_DN_ROWS 16
/* tokens of one expert run through the weight-shared token tiles (IQ2 gate/up and down); beyond that, one token
 * at a time. NEON kernels give the same sums either way, so AArch64 takes up to 32 (prompt reading); x86 keeps 8. */
#if IK_NEON
#define IK_MOE_NT 32
#else
#define IK_MOE_NT 8
#endif

static struct {
    const PhysLayer* pl;
    int ip, T;
    const float* x;
    const int8_t* xq;
    const float* xd;
    const int8_t* xnv;
    const void* xq8k;
    int uniq_n, total_pairs;
    int uniq_e[IK_NE];
    int n_tok[IK_NE];
    int u_off[IK_NE + 1]; /* prefix offsets into compact MO_tok_idx / dyn_h [total_pairs] */
    int *tok_idx, *tok_slot;
    float *dyn_h, *out;
    int8_t *hnv, *hq;
    float *hd;
} MO;

static struct {
    _Atomic int next;
} MW;

static inline float situ_act(float g, float u) {
    float sig = 1.0f / (1.0f + expf(-g));
    float act = IK_SITU_BETA * tanhf(g / IK_SITU_BETA) * sig;
    float upv = IK_SITU_LBETA * tanhf(u / IK_SITU_LBETA);
    return act * upv;
}

/* Fused single-barrier T=1 path: each worker owns an entire expert end-to-end in L1! */
static void moe_fused_t1_job(int tid, void* arg) {
    (void)tid; (void)arg;
    const PhysLayer* pl = MO.pl;
    size_t rb_nv_gu = (size_t)(IK_D / QK_NV) * BNV;
    size_t rb_q5_gu = (size_t)(IK_D / QK_K) * BQ5K;
    size_t rb_i2xs_gu = (size_t)(IK_D / QK_IQ2) * BIQ2_XS;
    size_t rb_i2xxs_gu = (size_t)(IK_D / QK_IQ2) * BIQ2_XXS;
    size_t rb_nv_dn = (size_t)(IK_FF / QK_NV) * BNV;
    size_t rb_q8_dn = (size_t)(IK_FF / QK_Q8) * BQ8;
    size_t rb_i2xs_dn = (size_t)(IK_FF / QK_IQ2) * BIQ2_XS;
    size_t rb_i2xxs_dn = (size_t)(IK_FF / QK_IQ2) * BIQ2_XXS;
    for (;;) {
        int u = atomic_fetch_add_explicit(&MW.next, 1, memory_order_relaxed);
        if (u >= MO.uniq_n) break;
        int e = MO.uniq_e[u];
        const uint8_t* wgu = pl->exp_gu[e];
        const uint8_t* wdn = pl->exp_dn[e];
        if (!wgu || !wdn) continue;
        float ws_gu = (pl->has_exp_scale && pl->exp_gu_type == TY_NVFP4) ? pl->exp_gu_scale[e] : 1.0f;
        float ws_dn = (pl->has_exp_scale && pl->exp_dn_type == TY_NVFP4) ? pl->exp_dn_scale[e] : 1.0f;
        float gv[IK_FF], uv[IK_FF], hv[IK_FF];
        if (pl->exp_gu_type == TY_NVFP4) {
            for (int r = 0; r < IK_FF; r += 4) {
                k_nvfp4_x4(wgu + (size_t)r * rb_nv_gu, rb_nv_gu, IK_D / QK_NV, 1, MO.xnv, MO.xd, IK_D, ws_gu, gv + r, IK_FF);
                k_nvfp4_x4(wgu + (size_t)(IK_FF + r) * rb_nv_gu, rb_nv_gu, IK_D / QK_NV, 1, MO.xnv, MO.xd, IK_D, ws_gu, uv + r, IK_FF);
            }
        } else if (pl->exp_gu_type == TY_IQ2XS_128) {
            for (int r = 0; r < IK_FF; r += 4) {
                k_iq2xs128_x4(wgu + (size_t)r * rb_i2xs_gu, rb_i2xs_gu, IK_D / QK_IQ2, 1, MO.xq, MO.xd, IK_D, gv + r, IK_FF);
                k_iq2xs128_x4(wgu + (size_t)(IK_FF + r) * rb_i2xs_gu, rb_i2xs_gu, IK_D / QK_IQ2, 1, MO.xq, MO.xd, IK_D, uv + r, IK_FF);
            }
        } else if (pl->exp_gu_type == TY_IQ2XXS_128) {
            for (int r = 0; r < IK_FF; r += 4) {
                k_iq2xxs128_x4(wgu + (size_t)r * rb_i2xxs_gu, rb_i2xxs_gu, IK_D / QK_IQ2, 1, MO.xq, MO.xd, IK_D, gv + r, IK_FF);
                k_iq2xxs128_x4(wgu + (size_t)(IK_FF + r) * rb_i2xxs_gu, rb_i2xxs_gu, IK_D / QK_IQ2, 1, MO.xq, MO.xd, IK_D, uv + r, IK_FF);
            }
        } else {
            size_t q8k_sz = (size_t)(IK_D / QK_K) * (4 + QK_K + 32);
            for (int r = 0; r < IK_FF; r++) {
                k_q5k_row(wgu + (size_t)r * rb_q5_gu, IK_D / QK_K, 1, MO.xq8k, q8k_sz, gv + r, IK_FF, 0);
                k_q5k_row(wgu + (size_t)(IK_FF + r) * rb_q5_gu, IK_D / QK_K, 1, MO.xq8k, q8k_sz, uv + r, IK_FF, 0);
            }
        }
        for (int r = 0; r < IK_FF; r++) hv[r] = situ_act(gv[r], uv[r]);
        int8_t hq[IK_FF], hnv[IK_FF];
        float hd[IK_FF / QK_Q8];
        q8_quantize(hv, IK_FF, hq, hd);
        int sl = MO.tok_slot[MO.u_off[u]];
        float* dst = MO.out + (size_t)sl * IK_D;
        if (pl->exp_dn_type == TY_NVFP4) {
            q8_to_nv(hq, IK_FF, hnv);
            for (int r = 0; r < IK_D; r += 4)
                k_nvfp4_x4(wdn + (size_t)r * rb_nv_dn, rb_nv_dn, IK_FF / QK_NV, 1, hnv, hd, IK_FF, ws_dn, dst + r, IK_D);
        } else if (pl->exp_dn_type == TY_IQ2XS_128) {
            for (int r = 0; r < IK_D; r += 4)
                k_iq2xs128_x4(wdn + (size_t)r * rb_i2xs_dn, rb_i2xs_dn, IK_FF / QK_IQ2, 1, hq, hd, IK_FF, dst + r, IK_D);
        } else if (pl->exp_dn_type == TY_IQ2XXS_128) {
            for (int r = 0; r < IK_D; r += 4)
                k_iq2xxs128_x4(wdn + (size_t)r * rb_i2xxs_dn, rb_i2xxs_dn, IK_FF / QK_IQ2, 1, hq, hd, IK_FF, dst + r, IK_D);
        } else {
            for (int r = 0; r < IK_D; r += 4)
                k_q8_raw_x4(wdn + (size_t)r * rb_q8_dn, rb_q8_dn, IK_FF / QK_Q8, 1, hq, hd, IK_FF, dst + r, IK_D);
        }
    }
}

/* Chunked 2-phase path for T > 1 (or when few CPU experts remain in hetero_k420/hybrid_multi) */
static void moe_gu_job(int tid, void* arg) {
    (void)tid; (void)arg;
    const PhysLayer* pl = MO.pl;
    size_t rb_nv = (size_t)(IK_D / QK_NV) * BNV;
    size_t rb_q5 = (size_t)(IK_D / QK_K) * BQ5K;
    size_t rb_i2xs = (size_t)(IK_D / QK_IQ2) * BIQ2_XS;
    size_t rb_i2xxs = (size_t)(IK_D / QK_IQ2) * BIQ2_XXS;
    int nch = IK_FF / MOE_GU_ROWS;
    int tot = MO.uniq_n * nch;
    for (;;) {
        int j = atomic_fetch_add_explicit(&MW.next, 2, memory_order_relaxed);
        if (j >= tot) break;
        int z = j + 2 < tot ? j + 2 : tot;
        for (int it = j; it < z; it++) {
            int u = it / nch;
            int r0 = (it % nch) * MOE_GU_ROWS;
            int e = MO.uniq_e[u];
            const uint8_t* w = pl->exp_gu[e];
            if (!w) continue;
            int nt = MO.n_tok[u];
            int base_p = MO.u_off[u];
            float ws = (pl->has_exp_scale && pl->exp_gu_type == TY_NVFP4) ? pl->exp_gu_scale[e] : 1.0f;

            if (pl->exp_gu_type == TY_NVFP4 || pl->exp_gu_type == TY_IQ2XS_128 || pl->exp_gu_type == TY_IQ2XXS_128) {
                const int8_t* src_q = (pl->exp_gu_type == TY_NVFP4) ? MO.xnv : MO.xq;
                size_t rb = (pl->exp_gu_type == TY_NVFP4) ? rb_nv :
                            (pl->exp_gu_type == TY_IQ2XS_128) ? rb_i2xs : rb_i2xxs;
                int8_t tmp_q[IK_MOE_NT * IK_D]; float tmp_d[IK_MOE_NT * (IK_D / QK_Q8)];
#if IK_NEON || defined(IK_MOE_TILE_ALL)
                const int tiled = 1;                 /* any token count, in chunks of IK_MOE_NT */
#else
                const int tiled = nt <= IK_MOE_NT;
#endif
                if (tiled) for (int c0 = 0; c0 < nt; c0 += IK_MOE_NT) {
                    const int base_c = base_p + c0;
                    const int ntc = nt - c0 < IK_MOE_NT ? nt - c0 : IK_MOE_NT;
                    for (int ti = 0; ti < ntc; ti++) {
                        int tk = MO.tok_idx[base_c + ti];
                        memcpy(tmp_q + ti * IK_D, src_q + (size_t)tk * IK_D, IK_D);
                        memcpy(tmp_d + ti * (IK_D / QK_Q8), MO.xd + (size_t)tk * (IK_D / QK_Q8), (IK_D / QK_Q8) * 4);
                    }
                    const int8_t* x_local = tmp_q; const float* xd_local = tmp_d;
                    const int nt = ntc, base_p = base_c;   /* this chunk */
                    float gbuf[IK_MOE_NT][MOE_GU_ROWS], ubuf[IK_MOE_NT][MOE_GU_ROWS];
                    for (int r = 0; r < MOE_GU_ROWS; r += 4) {
                        if (pl->exp_gu_type == TY_NVFP4) {
                            k_nvfp4_x4(w + (size_t)(r0 + r) * rb, rb, IK_D / QK_NV, nt, x_local, xd_local, IK_D, ws, &gbuf[0][r], MOE_GU_ROWS);
                            k_nvfp4_x4(w + (size_t)(IK_FF + r0 + r) * rb, rb, IK_D / QK_NV, nt, x_local, xd_local, IK_D, ws, &ubuf[0][r], MOE_GU_ROWS);
                        } else if (pl->exp_gu_type == TY_IQ2XS_128) {
                            k_iq2xs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_D / QK_IQ2, nt, x_local, xd_local, IK_D, &gbuf[0][r], MOE_GU_ROWS);
                            k_iq2xs128_x4(w + (size_t)(IK_FF + r0 + r) * rb, rb, IK_D / QK_IQ2, nt, x_local, xd_local, IK_D, &ubuf[0][r], MOE_GU_ROWS);
                        } else {
                            k_iq2xxs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_D / QK_IQ2, nt, x_local, xd_local, IK_D, &gbuf[0][r], MOE_GU_ROWS);
                            k_iq2xxs128_x4(w + (size_t)(IK_FF + r0 + r) * rb, rb, IK_D / QK_IQ2, nt, x_local, xd_local, IK_D, &ubuf[0][r], MOE_GU_ROWS);
                        }
                    }
                    for (int ti = 0; ti < nt; ti++) {
                        float* ho = MO.dyn_h + (size_t)(base_p + ti) * IK_FF + r0;
                        for (int r = 0; r < MOE_GU_ROWS; r++) ho[r] = situ_act(gbuf[ti][r], ubuf[ti][r]);
                    }
                } else {
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti];
                        float* ho = MO.dyn_h + (size_t)(base_p + ti) * IK_FF + r0;
                        float gv[MOE_GU_ROWS], uv[MOE_GU_ROWS];
                        const int8_t* xt = src_q + (size_t)tk * IK_D;
                        const float* dt = MO.xd + (size_t)tk * (IK_D / QK_Q8);
                        for (int r = 0; r < MOE_GU_ROWS; r += 4) {
                            if (pl->exp_gu_type == TY_NVFP4) {
                                k_nvfp4_x4(w + (size_t)(r0 + r) * rb, rb, IK_D / QK_NV, 1, xt, dt, IK_D, ws, &gv[r], 4);
                                k_nvfp4_x4(w + (size_t)(IK_FF + r0 + r) * rb, rb, IK_D / QK_NV, 1, xt, dt, IK_D, ws, &uv[r], 4);
                            } else if (pl->exp_gu_type == TY_IQ2XS_128) {
                                k_iq2xs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_D / QK_IQ2, 1, xt, dt, IK_D, &gv[r], 4);
                                k_iq2xs128_x4(w + (size_t)(IK_FF + r0 + r) * rb, rb, IK_D / QK_IQ2, 1, xt, dt, IK_D, &uv[r], 4);
                            } else {
                                k_iq2xxs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_D / QK_IQ2, 1, xt, dt, IK_D, &gv[r], 4);
                                k_iq2xxs128_x4(w + (size_t)(IK_FF + r0 + r) * rb, rb, IK_D / QK_IQ2, 1, xt, dt, IK_D, &uv[r], 4);
                            }
                        }
                        for (int r = 0; r < MOE_GU_ROWS; r++) ho[r] = situ_act(gv[r], uv[r]);
                    }
                }
            } else {
                size_t q8k_sz = (size_t)(IK_D / QK_K) * (4 + QK_K + 32);
                for (int ti = 0; ti < nt; ti++) {
                    int tk = MO.tok_idx[base_p + ti];
                    const void* xk = (const uint8_t*)MO.xq8k + (size_t)tk * q8k_sz;
                    float* ho = MO.dyn_h + (size_t)(base_p + ti) * IK_FF + r0;
                    for (int r = 0; r < MOE_GU_ROWS; r++) {
                        float gf = 0.f, uf = 0.f;
                        k_q5k_row(w + (size_t)(r0 + r) * rb_q5, IK_D / QK_K, 1, xk, q8k_sz, &gf, 1, 0);
                        k_q5k_row(w + (size_t)(IK_FF + r0 + r) * rb_q5, IK_D / QK_K, 1, xk, q8k_sz, &uf, 1, 0);
                        ho[r] = situ_act(gf, uf);
                    }
                }
            }
        }
    }
}

static void moe_dn_job(int tid, void* arg) {
    (void)tid; (void)arg;
    const PhysLayer* pl = MO.pl;
    size_t rb_nv = (size_t)(IK_FF / QK_NV) * BNV;
    size_t rb_q8 = (size_t)(IK_FF / QK_Q8) * BQ8;
    size_t rb_i2xs = (size_t)(IK_FF / QK_IQ2) * BIQ2_XS;
    size_t rb_i2xxs = (size_t)(IK_FF / QK_IQ2) * BIQ2_XXS;
    int nch = IK_D / MOE_DN_ROWS;
    int tot = MO.uniq_n * nch;
    for (;;) {
        int j = atomic_fetch_add_explicit(&MW.next, 2, memory_order_relaxed);
        if (j >= tot) break;
        int z = j + 2 < tot ? j + 2 : tot;
        for (int it = j; it < z; it++) {
            int u = it / nch;
            int r0 = (it % nch) * MOE_DN_ROWS;
            int e = MO.uniq_e[u];
            const uint8_t* w = pl->exp_dn[e];
            if (!w) continue;
            int nt = MO.n_tok[u];
            int base_p = MO.u_off[u];
            float ws = (pl->has_exp_scale && pl->exp_dn_type == TY_NVFP4) ? pl->exp_dn_scale[e] : 1.0f;
            if (pl->exp_dn_type == TY_IQ2XS_128 || pl->exp_dn_type == TY_IQ2XXS_128) {
                size_t rb = (pl->exp_dn_type == TY_IQ2XS_128) ? rb_i2xs : rb_i2xxs;
#if IK_NEON || defined(IK_MOE_TILE_ALL)
                const int tiled = 1;                 /* any token count, in chunks of IK_MOE_NT */
#else
                const int tiled = nt <= IK_MOE_NT;
#endif
                if (tiled) for (int c0 = 0; c0 < nt; c0 += IK_MOE_NT) {
                    const int base_c = base_p + c0;
                    const int ntc = nt - c0 < IK_MOE_NT ? nt - c0 : IK_MOE_NT;
                    const int nt = ntc, base_p = base_c;   /* this chunk */
                    float res[IK_MOE_NT][MOE_DN_ROWS];
                    const int8_t* hq = MO.hq + (size_t)base_p * IK_FF;
                    const float* hd = MO.hd + (size_t)base_p * (IK_FF / QK_Q8);
                    for (int r = 0; r < MOE_DN_ROWS; r += 4) {
                        if (pl->exp_dn_type == TY_IQ2XS_128)
                            k_iq2xs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_FF / QK_IQ2, nt, hq, hd, IK_FF, &res[0][r], MOE_DN_ROWS);
                        else
                            k_iq2xxs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_FF / QK_IQ2, nt, hq, hd, IK_FF, &res[0][r], MOE_DN_ROWS);
                    }
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti], sl = MO.tok_slot[base_p + ti];
                        memcpy(MO.out + ((size_t)tk * IK_NU + sl) * IK_D + r0, res[ti], sizeof(float) * MOE_DN_ROWS);
                    }
                } else {
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti], sl = MO.tok_slot[base_p + ti];
                        float* dst = MO.out + ((size_t)tk * IK_NU + sl) * IK_D + r0;
                        const int8_t* hq = MO.hq + (size_t)(base_p + ti) * IK_FF;
                        const float* hd = MO.hd + (size_t)(base_p + ti) * (IK_FF / QK_Q8);
                        for (int r = 0; r < MOE_DN_ROWS; r += 4) {
                            if (pl->exp_dn_type == TY_IQ2XS_128)
                                k_iq2xs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_FF / QK_IQ2, 1, hq, hd, IK_FF, dst + r, 4);
                            else
                                k_iq2xxs128_x4(w + (size_t)(r0 + r) * rb, rb, IK_FF / QK_IQ2, 1, hq, hd, IK_FF, dst + r, 4);
                        }
                    }
                }
            } else if (pl->exp_dn_type == TY_NVFP4) {
                if (nt <= 8) {
                    float res[8][MOE_DN_ROWS];
                    const int8_t* hnv = MO.hnv + (size_t)base_p * IK_FF;
                    const float* hd = MO.hd + (size_t)base_p * (IK_FF / QK_Q8);
                    for (int r = 0; r < MOE_DN_ROWS; r += 4)
                        k_nvfp4_x4(w + (size_t)(r0 + r) * rb_nv, rb_nv, IK_FF / QK_NV, nt, hnv, hd, IK_FF, ws, &res[0][r], MOE_DN_ROWS);
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti], sl = MO.tok_slot[base_p + ti];
                        memcpy(MO.out + ((size_t)tk * IK_NU + sl) * IK_D + r0, res[ti], sizeof(float) * MOE_DN_ROWS);
                    }
                } else {
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti], sl = MO.tok_slot[base_p + ti];
                        float* dst = MO.out + ((size_t)tk * IK_NU + sl) * IK_D + r0;
                        const int8_t* hnv = MO.hnv + (size_t)(base_p + ti) * IK_FF;
                        const float* hd = MO.hd + (size_t)(base_p + ti) * (IK_FF / QK_Q8);
                        for (int r = 0; r < MOE_DN_ROWS; r += 4)
                            k_nvfp4_x4(w + (size_t)(r0 + r) * rb_nv, rb_nv, IK_FF / QK_NV, 1, hnv, hd, IK_FF, ws, dst + r, 4);
                    }
                }
            } else {
                if (nt <= 8) {
                    float res[8][MOE_DN_ROWS];
                    const int8_t* hq = MO.hq + (size_t)base_p * IK_FF;
                    const float* hd = MO.hd + (size_t)base_p * (IK_FF / QK_Q8);
                    for (int r = 0; r < MOE_DN_ROWS; r += 4)
                        k_q8_raw_x4(w + (size_t)(r0 + r) * rb_q8, rb_q8, IK_FF / QK_Q8, nt, hq, hd, IK_FF, &res[0][r], MOE_DN_ROWS);
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti], sl = MO.tok_slot[base_p + ti];
                        memcpy(MO.out + ((size_t)tk * IK_NU + sl) * IK_D + r0, res[ti], sizeof(float) * MOE_DN_ROWS);
                    }
                } else {
                    for (int ti = 0; ti < nt; ti++) {
                        int tk = MO.tok_idx[base_p + ti], sl = MO.tok_slot[base_p + ti];
                        float* dst = MO.out + ((size_t)tk * IK_NU + sl) * IK_D + r0;
                        const int8_t* hq = MO.hq + (size_t)(base_p + ti) * IK_FF;
                        const float* hd = MO.hd + (size_t)(base_p + ti) * (IK_FF / QK_Q8);
                        for (int r = 0; r < MOE_DN_ROWS; r += 4)
                            k_q8_raw_x4(w + (size_t)(r0 + r) * rb_q8, rb_q8, IK_FF / QK_Q8, 1, hq, hd, IK_FF, dst + r, 4);
                    }
                }
            }
        }
    }
}

static void run_moe_cpu(int ip, int T, const float* x_in, const int8_t* xq, const float* xd,
                        const int* sel, const uint8_t* skip_mask, float* slot_outs) {
    const PhysLayer* pl = &M.L[ip];
    MO.pl = pl; MO.ip = ip; MO.T = T; MO.x = x_in; MO.xq = xq; MO.xd = xd; MO.out = slot_outs;

    for (int t = 0; t < T; t++) {
        q8_to_nv(xq + (size_t)t * IK_D, IK_D, M.buf_xnv + (size_t)t * IK_D);
        if (pl->exp_gu_type == TY_Q5K) {
            size_t q8k_sz = (size_t)(IK_D / QK_K) * (4 + QK_K + 32);
            q8k_quantize(x_in + (size_t)t * IK_D, IK_D, (uint8_t*)M.buf_xq8k + (size_t)t * q8k_sz);
        }
    }
    MO.xnv = M.buf_xnv;
    MO.xq8k = M.buf_xq8k;

    /* Count hits per unique expert, then pack compact prefix arrays */
    int seen[IK_NE], cnt[IK_NE];
    for (int e = 0; e < IK_NE; e++) seen[e] = -1;
    MO.uniq_n = 0;
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < IK_NU; s++) {
            if (skip_mask && skip_mask[t * IK_NU + s]) continue;
            int e = sel[t * IK_NU + s];
            if (!pl->exp_gu[e]) continue;
            int u = seen[e];
            if (u < 0) {
                u = MO.uniq_n++;
                seen[e] = u;
                MO.uniq_e[u] = e;
                cnt[u] = 0;
            }
            cnt[u]++;
        }
    }
    if (MO.uniq_n == 0) return;

    MO.u_off[0] = 0;
    for (int u = 0; u < MO.uniq_n; u++) {
        MO.n_tok[u] = 0;
        MO.u_off[u + 1] = MO.u_off[u] + cnt[u];
    }
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < IK_NU; s++) {
            if (skip_mask && skip_mask[t * IK_NU + s]) continue;
            int e = sel[t * IK_NU + s];
            if (!pl->exp_gu[e]) continue;
            int u = seen[e];
            int k = MO.u_off[u] + MO.n_tok[u]++;
            MO.tok_idx[k] = t;
            MO.tok_slot[k] = s;
        }
    }

    if (__builtin_expect(g_imat && g_imat->enabled, 0)) {
        for (int u = 0; u < MO.uniq_n; u++) {
            int e = MO.uniq_e[u];
            int nt = MO.n_tok[u];
            int base_p = MO.u_off[u];
            g_imat->cnt_exp[ip][e] += (uint64_t)nt;
            for (int ti = 0; ti < nt; ti++) {
                int tk = MO.tok_idx[base_p + ti];
                imat_accum(g_imat->exp_gu[ip][e], MO.x + (size_t)tk * IK_D, IK_D, 1);
            }
        }
    }

    if (T == 1 && MO.uniq_n >= KP.n && !(g_imat && g_imat->enabled)) {
        atomic_store(&MW.next, 0);
        kp_run(moe_fused_t1_job, NULL);
        return;
    }

    atomic_store(&MW.next, 0);
    kp_run(moe_gu_job, NULL);

    if (__builtin_expect(g_imat && g_imat->enabled, 0)) {
        for (int u = 0; u < MO.uniq_n; u++) {
            int e = MO.uniq_e[u];
            int nt = MO.n_tok[u];
            int base_p = MO.u_off[u];
            imat_accum(g_imat->exp_dn[ip][e], MO.dyn_h + (size_t)base_p * IK_FF, IK_FF, nt);
        }
    }

    int tot_p = MO.u_off[MO.uniq_n];
    for (int p = 0; p < tot_p; p++) {
        q8_quantize(MO.dyn_h + (size_t)p * IK_FF, IK_FF, MO.hq + (size_t)p * IK_FF, MO.hd + (size_t)p * (IK_FF / QK_Q8));
        if (pl->exp_dn_type == TY_NVFP4)
            q8_to_nv(MO.hq + (size_t)p * IK_FF, IK_FF, MO.hnv + (size_t)p * IK_FF);
    }
    atomic_store(&MW.next, 0);
    kp_run(moe_dn_job, NULL);
}

int ik_cpu_moe_slots(int il, int ip, int T, const float* x_in, const int* sel, const uint8_t* skip_mask, float* slot_outs) {
    (void)il;
    for (int t = 0; t < T; t++) q8_quantize(x_in + (size_t)t * IK_D, IK_D, M.buf_xq + (size_t)t * IK_D, M.buf_xd + (size_t)t * (IK_D / QK_Q8));
    run_moe_cpu(ip, T, x_in, M.buf_xq, M.buf_xd, sel, skip_mask, slot_outs);
    return 0;
}

/* ---------------------------------------------------------------- Physical Layer Forward */
static void run_layer(int il, int ip, int T, int pos0, float* cur, float* blks[], int* pnblk,
                      const float* msg, const float* const* bank_ptr, int nbank, const float* bank_out) {
    PhysLayer* pl = &M.L[ip];
    float* hist_slot = M.conv_hist + (size_t)il * 3 * IK_CHSCONV;
    float* trace_slot = (M.trace_active && M.conv_trace && T <= 8) ? (M.conv_trace + (size_t)il * 11 * IK_CHSCONV) : NULL;

    double t0 = now_sec();
    if (msg == NULL && bank_ptr != NULL && pl->has_loop) {
        do_mix(bank_ptr, nbank, bank_out, pl->loop_msg_q, 4, T, M.buf_msg);
        msg = M.buf_msg;
    }

    memcpy(M.buf_inpsa, cur, (size_t)T * IK_D * 4);

    if (pl->attn_res_q) {
        int nq = pl->is_swa ? 1 : 3;
        const float* bptrs[8];
        for (int b = 0; b < *pnblk; b++) bptrs[b] = blks[b];
        do_mix(bptrs, *pnblk, cur, pl->attn_res_q, nq, T, M.buf_in0);
        if (pl->is_swa) {
            for (int t = 0; t < T; t++) {
                memcpy(M.buf_in1 + (size_t)t * IK_D, M.buf_in0 + (size_t)t * IK_D, IK_D * 4);
                memcpy(M.buf_in2 + (size_t)t * IK_D, M.buf_in0 + (size_t)t * IK_D, IK_D * 4);
            }
        } else {
            for (int t = 0; t < T; t++) {
                const float* src = M.buf_in0 + (size_t)t * 3 * IK_D;
                float s0[IK_D]; memcpy(s0, src, IK_D * 4);
                memcpy(M.buf_in1 + (size_t)t * IK_D, src + IK_D, IK_D * 4);
                memcpy(M.buf_in2 + (size_t)t * IK_D, src + 2 * IK_D, IK_D * 4);
                memcpy(M.buf_in0 + (size_t)t * IK_D, s0, IK_D * 4);
            }
        }
    } else {
        memcpy(M.buf_in0, cur, (size_t)T * IK_D * 4);
        memcpy(M.buf_in1, cur, (size_t)T * IK_D * 4);
        memcpy(M.buf_in2, cur, (size_t)T * IK_D * 4);
    }

    if (msg) {
        do_control(M.buf_in0, msg + 0 * IK_D, 4 * IK_D, T, pl, 0);
        do_control(M.buf_in1, msg + 1 * IK_D, 4 * IK_D, T, pl, 1);
        do_control(M.buf_in2, msg + 2 * IK_D, 4 * IK_D, T, pl, 2);
    }

    if (pl->is_swa && msg) {
        for (int t = 0; t < T; t++) {
            rmsnorm_f32(M.buf_in0 + (size_t)t * IK_D, M.buf_in0 + (size_t)t * IK_D, pl->attn_norm, IK_D);
            rmsnorm_f32(M.buf_in1 + (size_t)t * IK_D, M.buf_in1 + (size_t)t * IK_D, pl->attn_norm, IK_D);
            rmsnorm_f32(M.buf_in2 + (size_t)t * IK_D, M.buf_in2 + (size_t)t * IK_D, pl->attn_norm, IK_D);
        }
    } else if (pl->is_swa) {
        for (int t = 0; t < T; t++)
            rmsnorm_f32(M.buf_in0 + (size_t)t * IK_D, M.buf_in0 + (size_t)t * IK_D, pl->attn_norm, IK_D);
        memcpy(M.buf_in1, M.buf_in0, (size_t)T * IK_D * 4);
        memcpy(M.buf_in2, M.buf_in0, (size_t)T * IK_D * 4);
    } else {
        for (int t = 0; t < T; t++) {
            rmsnorm_f32(M.buf_in0 + (size_t)t * IK_D, M.buf_in0 + (size_t)t * IK_D, pl->q_inp_norm, IK_D);
            rmsnorm_f32(M.buf_in1 + (size_t)t * IK_D, M.buf_in1 + (size_t)t * IK_D, pl->k_inp_norm, IK_D);
            rmsnorm_f32(M.buf_in2 + (size_t)t * IK_D, M.buf_in2 + (size_t)t * IK_D, pl->v_inp_norm, IK_D);
        }
    }

    if (__builtin_expect(g_imat && g_imat->enabled, 0)) {
        g_imat->cnt_dense[ip] += (uint64_t)T;
        imat_accum(g_imat->in0[ip], M.buf_in0, IK_D, T);
        imat_accum(g_imat->in1[ip], M.buf_in1, IK_D, T);
        imat_accum(g_imat->in2[ip], M.buf_in2, IK_D, T);
    }

    int8_t* xq0 = M.buf_xq;
    float*  xd0 = M.buf_xd;
    for (int t = 0; t < T; t++) q8_quantize(M.buf_in0 + (size_t)t * IK_D, IK_D, xq0 + (size_t)t * IK_D, xd0 + (size_t)t * (IK_D / QK_Q8));
    int same_12 = (pl->is_swa && !msg);
    int8_t* xq1 = same_12 ? xq0 : (xq0 + (size_t)T * IK_D);
    float*  xd1 = same_12 ? xd0 : (xd0 + (size_t)T * (IK_D / QK_Q8));
    int8_t* xq2 = same_12 ? xq0 : (xq1 + (size_t)T * IK_D);
    float*  xd2 = same_12 ? xd0 : (xd1 + (size_t)T * (IK_D / QK_Q8));
    if (!same_12) {
        for (int t = 0; t < T; t++) {
            q8_quantize(M.buf_in1 + (size_t)t * IK_D, IK_D, xq1 + (size_t)t * IK_D, xd1 + (size_t)t * (IK_D / QK_Q8));
            q8_quantize(M.buf_in2 + (size_t)t * IK_D, IK_D, xq2 + (size_t)t * IK_D, xd2 + (size_t)t * (IK_D / QK_Q8));
        }
    }

    double t1 = now_sec();
    M.timers[1] += t1 - t0;

    int use_h_qkv = (M.gpu.qkvg_start && ip >= IK_CORE_START && ip < IK_CORE_END);
    if (use_h_qkv)
        M.gpu.qkvg_start(M.gpu.ctx, ip, T, M.buf_in0, M.buf_in1, M.buf_in2, xq0, xd0, xq1, xd1, xq2, xd2);
    IkOp qkv_ops[5] = {
        {&pl->wq,    use_h_qkv ? M.gpu.split_q : IK_QDIM,   NULL, xq0, xd0, IK_D, M.buf_q, IK_QDIM},
        {&pl->wk,    use_h_qkv ? M.gpu.split_k : IK_GQADIM, NULL, xq1, xd1, IK_D, M.buf_k, IK_GQADIM},
        {&pl->wv,    use_h_qkv ? M.gpu.split_v : IK_GQADIM, NULL, xq2, xd2, IK_D, M.buf_v, IK_GQADIM},
        {&pl->wgate, use_h_qkv ? M.gpu.split_g : IK_QDIM,   NULL, xq0, xd0, IK_D, M.buf_g, IK_QDIM},
        {&pl->wrel,  IK_RDIM,                               NULL, xq0, xd0, IK_D, M.buf_r, IK_RDIM},
    };
    run_ops(qkv_ops, 5, T);
    if (use_h_qkv) M.gpu.qkvg_wait(M.gpu.ctx, ip, T, M.buf_q, M.buf_k, M.buf_v, M.buf_g, M.buf_r);

    double t2 = now_sec();
    M.timers[2] += t2 - t1;

    for (int t = 0; t < T; t++) {
        const float* in0 = M.buf_in0 + (size_t)t * IK_D;
        for (int h = 0; h < IK_NH; h++) {
            const float* sp = pl->sink_proj + (size_t)h * IK_D;
            float s = 0.f;
#if IK_AVX2
            __m256 a = _mm256_setzero_ps();
            for (int k = 0; k < IK_D; k += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(sp + k), _mm256_loadu_ps(in0 + k), a);
            s = hsum8(a);
#elif IK_NEON
            s = ik_dotf(sp, in0, IK_D);
#else
            for (int k = 0; k < IK_D; k++) s += sp[k] * in0[k];
#endif
            M.buf_sinks[(size_t)t * IK_NH + h] = s;
        }
        if (!pl->is_swa) {
            const float* in1 = M.buf_in1 + (size_t)t * IK_D;
            const float* in2 = M.buf_in2 + (size_t)t * IK_D;
            for (int c = 0; c < IK_NCONV * IK_NKV; c++) {
                const float* pk = pl->k_conv_proj + (size_t)c * IK_D;
                const float* pv = pl->v_conv_proj + (size_t)c * IK_D;
                float sk = 0.f, sv = 0.f;
#if IK_AVX2
                __m256 ak = _mm256_setzero_ps(), av = ak;
                for (int k = 0; k < IK_D; k += 8) {
                    ak = _mm256_fmadd_ps(_mm256_loadu_ps(pk + k), _mm256_loadu_ps(in1 + k), ak);
                    av = _mm256_fmadd_ps(_mm256_loadu_ps(pv + k), _mm256_loadu_ps(in2 + k), av);
                }
                sk = hsum8(ak); sv = hsum8(av);
#elif IK_NEON
                sk = ik_dotf(pk, in1, IK_D); sv = ik_dotf(pv, in2, IK_D);
#else
                for (int k = 0; k < IK_D; k++) { sk += pk[k] * in1[k]; sv += pv[k] * in2[k]; }
#endif
                M.buf_kcp[(size_t)t * (IK_NCONV * IK_NKV) + c] = sk;
                M.buf_vcp[(size_t)t * (IK_NCONV * IK_NKV) + c] = sv;
            }
        }
    }

    do_convolve(M.buf_k, IK_GQADIM, T, pl->k_conv, pl->is_swa ? NULL : M.buf_kcp, hist_slot, trace_slot, 0);
    do_convolve(M.buf_v, IK_GQADIM, T, pl->v_conv, pl->is_swa ? NULL : M.buf_vcp, hist_slot, trace_slot, IK_GQADIM);

    for (int t = 0; t < T; t++) {
        float* Qt = M.buf_q + (size_t)t * IK_QDIM;
        float* Kt = M.buf_k + (size_t)t * IK_GQADIM;
        float* Vt = M.buf_v + (size_t)t * IK_GQADIM;
        float* Rt = M.buf_r + (size_t)t * IK_RDIM;
        do_clip(Qt, IK_HD, IK_NH, pl->q_clip, IK_QK_CLIP);
        do_clip(Kt, IK_HD, IK_NKV, NULL, IK_QK_CLIP);
        for (int h = 0; h < IK_NKV; h++) rmsnorm_f32(Vt + h * IK_HD, Vt + h * IK_HD, NULL, IK_HD);
        do_clip(Rt, IK_NREL, IK_NH, pl->rel_clip, IK_REL_CLIP);
        if (!pl->is_swa) {
            float ratio = (float)(pos0 + t + 1) / IK_LOG_FLOOR;
            float tau = 1.0f + IK_LOG_ALPHA * logf(ratio > 1.0f ? ratio : 1.0f);
            for (int i = 0; i < IK_QDIM; i++) Qt[i] *= tau;
            for (int i = 0; i < IK_RDIM; i++) Rt[i] *= tau;
        }
        int slot = pl->is_swa ? ((pos0 + t) % M.kv_cap[il]) : (pos0 + t);
        uint16_t* Kdst = M.kv_k[il] + (size_t)slot * IK_GQADIM;
        uint16_t* Vdst = M.kv_v[il] + (size_t)slot * IK_GQADIM;
        for (int i = 0; i < IK_GQADIM; i++) { Kdst[i] = f2h(Kt[i]); Vdst[i] = f2h(Vt[i]); }
    }

    AJ.j = (AttnJob){il, pl->is_swa, pos0, T, M.buf_q, M.buf_r, M.buf_sinks, pl->rel_bias,
                     M.kv_k[il], M.kv_v[il], M.kv_cap[il], M.buf_attn};
    atomic_store(&AJ.next_h, 0);
    kp_run(attn_worker, NULL);

    for (int t = 0; t < T; t++) {
        float* at = M.buf_attn + (size_t)t * IK_QDIM;
        const float* gt = M.buf_g + (size_t)t * IK_QDIM;
        for (int i = 0; i < IK_QDIM; i++) at[i] *= 1.0f / (1.0f + expf(-gt[i]));
        q8_quantize(at, IK_QDIM, M.buf_xq + (size_t)t * IK_QDIM, M.buf_xd + (size_t)t * (IK_QDIM / QK_Q8));
    }

    if (__builtin_expect(g_imat && g_imat->enabled, 0)) {
        imat_accum(g_imat->attn[ip], M.buf_attn, IK_QDIM, T);
    }

    double t3 = now_sec();
    M.timers[1] += t3 - t2;

    int use_h_o = (M.gpu.o_start && ip >= IK_CORE_START && ip < IK_CORE_END);
    if (use_h_o) M.gpu.o_start(M.gpu.ctx, ip, T, M.buf_attn, M.buf_xq, M.buf_xd);
    IkOp o_op[1] = {{&pl->wo, use_h_o ? M.gpu.split_o : IK_D, NULL, M.buf_xq, M.buf_xd, IK_QDIM, M.buf_o, IK_D}};
    run_ops(o_op, 1, T);
    if (use_h_o) M.gpu.o_wait(M.gpu.ctx, ip, T, M.buf_o);

    double t4 = now_sec();
    M.timers[2] += t4 - t3;

    do_convolve(M.buf_o, IK_D, T, pl->attn_post_conv, NULL, hist_slot, trace_slot, 2 * IK_GQADIM);

    if (ip % IK_BLOCK_SZ == 0) {
        int bi = (*pnblk)++;
        memcpy(M.buf_blk + (size_t)bi * T * IK_D, M.buf_inpsa, (size_t)T * IK_D * 4);
        blks[bi] = M.buf_blk + (size_t)bi * T * IK_D;
        memcpy(cur, M.buf_o, (size_t)T * IK_D * 4);
    } else {
        for (size_t i = 0; i < (size_t)T * IK_D; i++) cur[i] = M.buf_inpsa[i] + M.buf_o[i];
    }

    memcpy(M.buf_ffninp, cur, (size_t)T * IK_D * 4);

    {
        const float* bptrs[8];
        for (int b = 0; b < *pnblk; b++) bptrs[b] = blks[b];
        do_mix(bptrs, *pnblk, cur, pl->ffn_res_q, 1, T, M.buf_ffnin);
    }
    if (msg) do_control(M.buf_ffnin, msg + 3 * IK_D, 4 * IK_D, T, pl, 3);
    for (int t = 0; t < T; t++)
        rmsnorm_f32(M.buf_ffnin + (size_t)t * IK_D, M.buf_ffnin + (size_t)t * IK_D, pl->ffn_norm, IK_D);

    IkOp r_op[1] = {{&pl->ffn_gate_inp, IK_NE, M.buf_ffnin, NULL, NULL, IK_D, M.buf_probs, IK_NE}};
    run_ops(r_op, 1, T);

    int sel[IK_MAXB * IK_NU];
    float wsel[IK_MAXB * IK_NU];
    uint8_t on_gpu_mask[IK_MAXB * IK_NU];
    memset(on_gpu_mask, 0, (size_t)T * IK_NU);
    for (int t = 0; t < T; t++) {
        float* pr = M.buf_probs + (size_t)t * IK_NE;
        float sc[IK_NE];
        for (int e = 0; e < IK_NE; e++) {
            pr[e] = 1.0f / (1.0f + expf(-pr[e]));
            sc[e] = pr[e] + pl->exp_bias[e];
        }
        int used[IK_NE] = {0};
        float sumw = 0.f;
        for (int k = 0; k < IK_NU; k++) {
            int bi = -1; float bv = -1e30f;
            for (int e = 0; e < IK_NE; e++) {
                if (!used[e] && sc[e] > bv) { bv = sc[e]; bi = e; }
            }
            used[bi] = 1;
            sel[t * IK_NU + k] = bi;
            wsel[t * IK_NU + k] = pr[bi];
            sumw += pr[bi];
            M.exp_counts[ip][bi]++;
        }
        float fac = IK_EXP_SCALE / (sumw + 1e-20f);
        for (int k = 0; k < IK_NU; k++) wsel[t * IK_NU + k] *= fac;
    }

    for (int t = 0; t < T; t++)
        q8_quantize(M.buf_ffnin + (size_t)t * IK_D, IK_D, M.buf_xq + (size_t)t * IK_D, M.buf_xd + (size_t)t * (IK_D / QK_Q8));

    double t5 = now_sec();
    M.timers[2] += t5 - t4;

    if (M.gpu.moe_start)
        M.gpu.moe_start(M.gpu.ctx, ip, T, M.buf_ffnin, M.buf_xq, M.buf_xd, sel, wsel, on_gpu_mask);
    run_moe_cpu(ip, T, M.buf_ffnin, M.buf_xq, M.buf_xd, sel, on_gpu_mask, M.buf_exps_out);
    if (M.gpu.moe_wait)
        M.gpu.moe_wait(M.gpu.ctx, ip, T, M.buf_exps_out);

    for (int t = 0; t < T; t++) {
        float* dst = M.buf_o + (size_t)t * IK_D;
        memset(dst, 0, sizeof(float) * IK_D);
        for (int s = 0; s < IK_NU; s++) {
            float w = wsel[t * IK_NU + s];
            const float* src = M.buf_exps_out + ((size_t)t * IK_NU + s) * IK_D;
#if IK_AVX2
            __m256 vw = _mm256_set1_ps(w);
            for (int i = 0; i < IK_D; i += 8)
                _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_mm256_loadu_ps(src + i), vw, _mm256_loadu_ps(dst + i)));
#elif IK_NEON
            ik_axpyf(dst, w, src, IK_D);
#else
            for (int i = 0; i < IK_D; i++) dst[i] += w * src[i];
#endif
        }
    }

    double t6 = now_sec();
    M.timers[3] += t6 - t5;

    do_convolve(M.buf_o, IK_D, T, pl->ffn_post_conv, NULL, hist_slot, trace_slot, 2 * IK_GQADIM + IK_D);
    for (size_t i = 0; i < (size_t)T * IK_D; i++) cur[i] = M.buf_ffninp[i] + M.buf_o[i];
    M.timers[1] += now_sec() - t6;
}

/* Parallel lm_head softcap / greedy argmax */
static struct {
    float* logits;
    int T, r_end, tk;
    int best_tok[IK_KMAXT];
    float m1[IK_KMAXT], m2[IK_KMAXT];
} HARG;

static void head_softcap_job(int tid, void* arg) {
    (void)arg;
    int t = tid, n = KP.n;
    int lo = (int)(((long)HARG.r_end * t) / n);
    int hi = (int)(((long)HARG.r_end * (t + 1)) / n);
    for (int tk = 0; tk < HARG.T; tk++) {
        float* lg = HARG.logits + (size_t)tk * IK_V_UNPAD;
        for (int v = lo; v < hi; v++) lg[v] = IK_LOGIT_CAP * tanhf(lg[v] / IK_LOGIT_CAP);
    }
}

static void head_greedy_job(int tid, void* arg) {
    (void)arg;
    int t = tid, n = KP.n;
    int lo = (int)(((long)IK_V_UNPAD * t) / n);
    int hi = (int)(((long)IK_V_UNPAD * (t + 1)) / n);
    const float* lg = HARG.logits + (size_t)HARG.tk * IK_V_UNPAD;
    int bt = lo; float m1 = -1e30f, m2 = -1e30f;
    for (int v = lo; v < hi; v++) {
        float x = lg[v];
        if (x > m1) { m2 = m1; m1 = x; bt = v; }
        else if (x > m2) { m2 = x; }
    }
    HARG.best_tok[t] = bt; HARG.m1[t] = m1; HARG.m2[t] = m2;
}

/* ---------------------------------------------------------------- Public C API */
int ik_init(const char* gguf_path, size_t data_start, const uint64_t* g_offs, const int* g_tys,
            const uint64_t* l_offs, const int* l_tys, int ctx, const uint8_t* on_gpu_exp,
            int zero_copy_gpu_mem, int n_threads) {
    memset(&M, 0, sizeof(M));
    init_tables();
    kp_init(n_threads);
    M.fd = open(gguf_path, O_RDONLY);
    if (M.fd < 0) { perror("ik_init open"); return -1; }
    struct stat st;
    if (fstat(M.fd, &st) != 0) { close(M.fd); return -1; }
    M.map_sz = (size_t)st.st_size;
    M.map = (uint8_t*)mmap(NULL, M.map_sz, PROT_READ, MAP_SHARED, M.fd, 0);
    if (M.map == MAP_FAILED) { perror("ik_init mmap"); close(M.fd); return -1; }
    M.data_start = data_start;
    M.ctx = ctx;

    const uint8_t* base = M.map + data_start;
    #define TPTR(off) ((const uint8_t*)(base + (off)))
    M.tok_embd = TPTR(g_offs[GT_TOK_EMBD]);
    M.tok_norm = (float*)huge_alloc(IK_D * 4);
    memcpy(M.tok_norm, TPTR(g_offs[GT_TOK_NORM]), IK_D * 4);
    M.output_norm = (float*)huge_alloc(IK_D * 4);
    memcpy(M.output_norm, TPTR(g_offs[GT_OUTPUT_NORM]), IK_D * 4);
    M.output_res_q = (float*)huge_alloc(IK_D * 4);
    memcpy(M.output_res_q, TPTR(g_offs[GT_OUTPUT_RES_Q]), IK_D * 4);
    if (!zero_copy_gpu_mem) {
        M.output_head = wrap_mat(TPTR(g_offs[GT_OUTPUT]), g_tys[GT_OUTPUT], IK_V_UNPAD, IK_D);
    }

    for (int ip = 0; ip < IK_NL_PHYS; ip++) {
        PhysLayer* pl = &M.L[ip];
        const uint64_t* lo = l_offs + ip * LT_COUNT;
        const int* lt = l_tys + ip * LT_COUNT;
        pl->is_swa = ((ip % IK_BLOCK_SZ) != 5);

        if (!zero_copy_gpu_mem) {
            pl->wq    = wrap_mat(TPTR(lo[LT_ATTN_Q]),    lt[LT_ATTN_Q],    IK_QDIM,   IK_D);
            pl->wk    = wrap_mat(TPTR(lo[LT_ATTN_K]),    lt[LT_ATTN_K],    IK_GQADIM, IK_D);
            pl->wv    = wrap_mat(TPTR(lo[LT_ATTN_V]),    lt[LT_ATTN_V],    IK_GQADIM, IK_D);
            pl->wo    = wrap_mat(TPTR(lo[LT_ATTN_O]),    lt[LT_ATTN_O],    IK_D,      IK_QDIM);
            pl->wgate = wrap_mat(TPTR(lo[LT_ATTN_GATE]), lt[LT_ATTN_GATE], IK_QDIM,   IK_D);
            pl->wrel  = wrap_mat(TPTR(lo[LT_ATTN_REL]),  lt[LT_ATTN_REL],  IK_RDIM,   IK_D);
        }

        #define CP_TENS(FIELD, IDX, NELEM) do { \
            if (lt[IDX] >= 0) { \
                FIELD = (float*)huge_alloc((size_t)(NELEM) * 4); \
                if (lt[IDX] == TY_F32) memcpy(FIELD, TPTR(lo[IDX]), (size_t)(NELEM) * 4); \
                else { const uint16_t* s_ = (const uint16_t*)TPTR(lo[IDX]); for (size_t i_ = 0; i_ < (size_t)(NELEM); i_++) FIELD[i_] = bf16_to_f32(s_[i_]); } \
            } \
        } while (0)

        CP_TENS(pl->sink_proj, LT_ATTN_SINK_PROJ, IK_D * IK_NH);
        CP_TENS(pl->q_clip, LT_ATTN_Q_CLIP, IK_HD * IK_NH);
        CP_TENS(pl->rel_bias, LT_ATTN_REL_BIAS, IK_NREL * IK_RELEXT);
        CP_TENS(pl->rel_clip, LT_ATTN_REL_CLIP, IK_NREL * IK_NH);
        CP_TENS(pl->k_conv, LT_ATTN_K_CONV, 4 * IK_GQADIM);
        CP_TENS(pl->v_conv, LT_ATTN_V_CONV, 4 * IK_GQADIM);
        CP_TENS(pl->attn_post_conv, LT_ATTN_POST_CONV, 4 * IK_D);
        CP_TENS(pl->ffn_post_conv, LT_FFN_POST_CONV, 4 * IK_D);
        CP_TENS(pl->attn_norm, LT_ATTN_NORM, IK_D);
        CP_TENS(pl->q_inp_norm, LT_ATTN_Q_INP_NORM, IK_D);
        CP_TENS(pl->k_inp_norm, LT_ATTN_K_INP_NORM, IK_D);
        CP_TENS(pl->v_inp_norm, LT_ATTN_V_INP_NORM, IK_D);
        CP_TENS(pl->k_conv_proj, LT_ATTN_K_CONV_PROJ, IK_D * 4 * IK_NKV);
        CP_TENS(pl->v_conv_proj, LT_ATTN_V_CONV_PROJ, IK_D * 4 * IK_NKV);
        CP_TENS(pl->attn_res_q, LT_ATTN_RES_Q, pl->is_swa ? IK_D : 3 * IK_D);
        CP_TENS(pl->ffn_res_q, LT_FFN_RES_Q, IK_D);
        CP_TENS(pl->ffn_norm, LT_FFN_NORM, IK_D);
        CP_TENS(pl->exp_bias, LT_EXP_BIAS, IK_NE);

        if (lt[LT_FFN_GATE_INP] >= 0) {
            pl->ffn_gate_inp = (Mat){IK_NE, IK_D, TY_F32, (uint8_t*)TPTR(lo[LT_FFN_GATE_INP]), (size_t)IK_NE * IK_D * 4};
        }

        pl->exp_gu_type = lt[LT_GATE_UP_EXPS];
        pl->exp_dn_type = lt[LT_DOWN_EXPS];
        size_t gu_eb = (pl->exp_gu_type == TY_NVFP4)      ? (size_t)IK_FF2 * (IK_D / QK_NV) * BNV :
                       (pl->exp_gu_type == TY_IQ2XS_128)  ? (size_t)IK_FF2 * (IK_D / QK_IQ2) * BIQ2_XS :
                       (pl->exp_gu_type == TY_IQ2XXS_128) ? (size_t)IK_FF2 * (IK_D / QK_IQ2) * BIQ2_XXS :
                                                            (size_t)IK_FF2 * (IK_D / QK_K) * BQ5K;
        size_t dn_eb = (pl->exp_dn_type == TY_NVFP4)      ? (size_t)IK_D * (IK_FF / QK_NV) * BNV :
                       (pl->exp_dn_type == TY_IQ2XS_128)  ? (size_t)IK_D * (IK_FF / QK_IQ2) * BIQ2_XS :
                       (pl->exp_dn_type == TY_IQ2XXS_128) ? (size_t)IK_D * (IK_FF / QK_IQ2) * BIQ2_XXS :
                                                            (size_t)IK_D * (IK_FF / QK_Q8) * BQ8;
        pl->has_exp_scale = (lt[LT_GATE_UP_SCALE] >= 0);
        if (pl->has_exp_scale) {
            memcpy(pl->exp_gu_scale, TPTR(lo[LT_GATE_UP_SCALE]), IK_NE * 4);
            memcpy(pl->exp_dn_scale, TPTR(lo[LT_DOWN_SCALE]), IK_NE * 4);
        }
        const uint8_t* gu_base = TPTR(lo[LT_GATE_UP_EXPS]);
        const uint8_t* dn_base = TPTR(lo[LT_DOWN_EXPS]);
        for (int e = 0; e < IK_NE; e++) {
            int og = on_gpu_exp ? on_gpu_exp[ip * IK_NE + e] : 0;
            if (og == 2) {
                pl->exp_gu[e] = NULL;
                pl->exp_dn[e] = NULL;
            } else {
                pl->exp_gu[e] = gu_base + (size_t)e * gu_eb;
                pl->exp_dn[e] = dn_base + (size_t)e * dn_eb;
            }
        }

        pl->has_loop = (ip >= IK_CORE_START && ip < IK_CORE_END);
        if (pl->has_loop) {
            CP_TENS(pl->loop_msg_q, LT_LOOP_MSG_Q, 4 * IK_D);
            int d_idx[4] = {LT_LOOP_Q_D, LT_LOOP_K_D, LT_LOOP_V_D, LT_LOOP_FFN_D};
            int g_idx[4] = {LT_LOOP_Q_G, LT_LOOP_K_G, LT_LOOP_V_G, LT_LOOP_FFN_G};
            int f_idx[4] = {LT_LOOP_Q_F, LT_LOOP_K_F, LT_LOOP_V_F, LT_LOOP_FFN_F};
            for (int c = 0; c < 4; c++) {
                if (lt[d_idx[c]] == TY_Q8)
                    pl->loop_down[c] = wrap_mat(TPTR(lo[d_idx[c]]), TY_Q8, IK_GRANK, 2 * IK_D);
                else
                    pl->loop_down[c] = (Mat){IK_GRANK, 2 * IK_D, TY_F32, (uint8_t*)TPTR(lo[d_idx[c]]), (size_t)IK_GRANK * 2 * IK_D * 4};
                if (lt[g_idx[c]] == TY_Q8)
                    pl->loop_gain[c] = wrap_mat(TPTR(lo[g_idx[c]]), TY_Q8, IK_D, IK_GRANK);
                else
                    pl->loop_gain[c] = (Mat){IK_D, IK_GRANK, TY_F32, (uint8_t*)TPTR(lo[g_idx[c]]), (size_t)IK_D * IK_GRANK * 4};
                if (lt[f_idx[c]] == TY_Q8)
                    pl->loop_forget[c] = wrap_mat(TPTR(lo[f_idx[c]]), TY_Q8, IK_D, IK_GRANK);
                else
                    pl->loop_forget[c] = (Mat){IK_D, IK_GRANK, TY_F32, (uint8_t*)TPTR(lo[f_idx[c]]), (size_t)IK_D * IK_GRANK * 4};
            }
        }
        #undef CP_TENS
    }

    if (!zero_copy_gpu_mem) {
        for (int il = 0; il < IK_NL_VIRT; il++) {
            int ip = layer_phys_idx(il);
            int cap = M.L[ip].is_swa ? (IK_SWA + IK_MAXB) : ctx;
            M.kv_cap[il] = cap;
            size_t kvb = (size_t)cap * IK_GQADIM * 2;
            M.kv_k[il] = (uint16_t*)huge_alloc(kvb);
            M.kv_v[il] = (uint16_t*)huge_alloc(kvb);
            memset(M.kv_k[il], 0, kvb);
            memset(M.kv_v[il], 0, kvb);
        }
    }

    M.conv_hist = (float*)calloc((size_t)IK_NL_VIRT * 3 * IK_CHSCONV, 4);

    int MT = IK_MAXB;
    M.max_t = MT;
    M.buf_cur     = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_pre     = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_blk     = (float*)huge_alloc((size_t)8 * MT * IK_D * 4);
    M.buf_blk_pre = (float*)huge_alloc((size_t)8 * MT * IK_D * 4);
    M.buf_bank    = (float*)huge_alloc((size_t)4 * MT * IK_D * 4);
    M.buf_msg     = (float*)huge_alloc((size_t)MT * 4 * IK_D * 4);
    M.buf_in0     = (float*)huge_alloc((size_t)MT * 3 * IK_D * 4);
    M.buf_in1     = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_in2     = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_inpsa   = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_ffninp  = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_q       = (float*)huge_alloc((size_t)MT * IK_QDIM * 4);
    M.buf_k       = (float*)huge_alloc((size_t)MT * IK_GQADIM * 4);
    M.buf_v       = (float*)huge_alloc((size_t)MT * IK_GQADIM * 4);
    M.buf_g       = (float*)huge_alloc((size_t)MT * IK_QDIM * 4);
    M.buf_r       = (float*)huge_alloc((size_t)MT * IK_RDIM * 4);
    M.buf_sinks   = (float*)huge_alloc((size_t)MT * IK_NH * 4);
    M.buf_kcp     = (float*)huge_alloc((size_t)MT * 4 * IK_NKV * 4);
    M.buf_vcp     = (float*)huge_alloc((size_t)MT * 4 * IK_NKV * 4);
    M.buf_attn    = (float*)huge_alloc((size_t)MT * IK_QDIM * 4);
    M.buf_o       = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_ffnin   = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_probs   = (float*)huge_alloc((size_t)MT * IK_NE * 4);
    M.buf_exps_out= (float*)huge_alloc((size_t)MT * IK_NU * IK_D * 4);
    M.buf_ctrl_in = (float*)huge_alloc((size_t)MT * 2 * IK_D * 4);
    M.buf_ctrl_g  = (float*)huge_alloc((size_t)MT * IK_GRANK * 4);
    M.buf_ctrl_gn = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_ctrl_fg = (float*)huge_alloc((size_t)MT * IK_D * 4);
    M.buf_xq      = (int8_t*)huge_alloc((size_t)3 * MT * IK_QDIM);
    M.buf_xnv     = (int8_t*)huge_alloc((size_t)MT * IK_D);
    M.buf_xd      = (float*)huge_alloc((size_t)3 * MT * (IK_QDIM / QK_Q8) * 4);
    M.buf_xq8k    = huge_alloc((size_t)MT * (IK_D / QK_K) * (4 + QK_K + 32));

    int MP = MT * IK_NU;
    MO.tok_idx  = (int*)huge_alloc((size_t)MP * sizeof(int));
    MO.tok_slot = (int*)huge_alloc((size_t)MP * sizeof(int));
    MO.dyn_h    = (float*)huge_alloc((size_t)MP * IK_FF * 4);
    MO.hnv      = (int8_t*)huge_alloc((size_t)MP * IK_FF);
    MO.hq       = (int8_t*)huge_alloc((size_t)MP * IK_FF);
    MO.hd       = (float*)huge_alloc((size_t)MP * (IK_FF / QK_Q8) * 4);
    M.dspark_feat = (float*)huge_alloc((size_t)MT * 4 * IK_D * 4);
    #undef TPTR
    return 0;
}

void ik_set_gpu_hooks(const IkGpuHooks* hooks) {
    if (hooks) M.gpu = *hooks;
    else memset(&M.gpu, 0, sizeof(M.gpu));
}

void ik_reset(void) {
    if (M.conv_hist) memset(M.conv_hist, 0, (size_t)IK_NL_VIRT * 3 * IK_CHSCONV * 4);
    if (M.gpu.reset) M.gpu.reset(M.gpu.ctx);
}

void ik_enable_conv_trace(int on) {
    if (on && !M.conv_trace) {
        M.conv_trace = (float*)calloc((size_t)IK_NL_VIRT * 11 * IK_CHSCONV, 4);
    }
    M.trace_active = on ? 1 : 0;
}

void ik_rollback_conv(int k_acc) {
    if (!M.conv_trace || !M.conv_hist || k_acc < 1 || k_acc > 8) return;
    size_t plane_f = (size_t)IK_CHSCONV;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        float* hd = M.conv_hist + (size_t)il * 3 * plane_f;
        const float* td = M.conv_trace + (size_t)il * 11 * plane_f;
        for (int s = 0; s < 3; s++) {
            memcpy(hd + (size_t)s * plane_f, td + (size_t)(k_acc + s) * plane_f, plane_f * 4);
        }
    }
}

void ik_get_tok_embd_normed(int tok, float* out_2304) {
    size_t emb_rb = (size_t)(IK_D / QK_Q8) * BQ8;
    const uint8_t* er = M.tok_embd + (size_t)tok * emb_rb;
    for (int b = 0; b < IK_D / QK_Q8; b++) {
        const uint8_t* B = er + (size_t)b * BQ8;
        uint16_t hs; memcpy(&hs, B, 2);
        float d = h2f(hs);
        const int8_t* qs = (const int8_t*)(B + 2);
        for (int j = 0; j < QK_Q8; j++) out_2304[b * QK_Q8 + j] = (float)qs[j] * d;
    }
    rmsnorm_f32(out_2304, out_2304, M.tok_norm, IK_D);
}

/* The state that the positions alone do not give back: the short-conv history and, when the host holds the KV, the
   sliding-window rings (a reply longer than IK_MAXB tokens overwrites ring slots that the prompt's last 1,024
   positions still need). A snapshot at the end of a prompt lets a later request continue from there. */
static size_t ring_bytes(int il) {
    return (M.kv_k[il] && M.kv_cap[il] == IK_SWA + IK_MAXB) ? (size_t)M.kv_cap[il] * IK_GQADIM * 2 : 0;
}
size_t ik_state_bytes(void) {
    size_t b = (size_t)IK_NL_VIRT * 3 * IK_CHSCONV * 4;
    for (int il = 0; il < IK_NL_VIRT; il++) b += 2 * ring_bytes(il);
    return b;
}
void ik_snapshot(void* buf) {
    uint8_t* p = (uint8_t*)buf;
    size_t cb = (size_t)IK_NL_VIRT * 3 * IK_CHSCONV * 4;
    memcpy(p, M.conv_hist, cb); p += cb;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        size_t rb = ring_bytes(il);
        if (!rb) continue;
        memcpy(p, M.kv_k[il], rb); p += rb;
        memcpy(p, M.kv_v[il], rb); p += rb;
    }
}
void ik_restore(const void* buf, int pos) {
    const uint8_t* p = (const uint8_t*)buf;
    size_t cb = (size_t)IK_NL_VIRT * 3 * IK_CHSCONV * 4;
    memcpy(M.conv_hist, p, cb); p += cb;
    for (int il = 0; il < IK_NL_VIRT; il++) {
        size_t rb = ring_bytes(il);
        if (!rb) continue;
        memcpy(M.kv_k[il], p, rb); p += rb;
        memcpy(M.kv_v[il], p, rb); p += rb;
    }
    if (M.gpu.on_roll) M.gpu.on_roll(M.gpu.ctx, pos);
}
void ik_get_profile(uint64_t* exp_counts, double* timers6) {
    if (exp_counts) memcpy(exp_counts, M.exp_counts, sizeof(M.exp_counts));
    if (timers6) memcpy(timers6, M.timers, sizeof(M.timers));
}
void ik_reset_timers(void) { memset(M.timers, 0, sizeof(M.timers)); }
int ik_n_threads(void) { return KP.n; }
int ik_max_batch(void) { return IK_MAXB; }
float* ik_conv_hist_ptr(void) { return M.conv_hist; }

void ik_set_dspark_targets(int n, const int* ids) {
    if (n < 0) n = 0;
    if (n > 4) n = 4;
    M.dspark_n = n;
    for (int i = 0; i < n; i++) M.dspark_ids[i] = ids[i];
}

const float* ik_get_dspark_features(void) { return M.dspark_feat; }
const uint8_t* ik_get_tok_embd_ptr(void) { return M.tok_embd; }
const float* ik_get_tok_norm_ptr(void) { return M.tok_norm; }
const uint8_t* ik_get_output_head_ptr(void) { return M.output_head.data; }

static inline void maybe_tap_dspark(int visit, int T, const float* src) {
    if (M.dspark_n <= 0) return;
    for (int i = 0; i < M.dspark_n; i++) {
        if (M.dspark_ids[i] == visit) {
            for (int t = 0; t < T; t++) {
                memcpy(M.dspark_feat + ((size_t)t * M.dspark_n + i) * IK_D,
                       src + (size_t)t * IK_D, IK_D * 4);
            }
        }
    }
}

/* need_logits: 1 = logits for all T rows into logits_out, 2 = only for the last row (into row 0) */
static int run_forward_internal(const int* toks, int T, int pos0, float* logits_out, int need_logits, int apply_cap) {
    if (T <= 0 || T > M.max_t) return -1;
    double t_all0 = now_sec();
    size_t emb_rb = (size_t)(IK_D / QK_Q8) * BQ8;
    for (int t = 0; t < T; t++) {
        const uint8_t* er = M.tok_embd + (size_t)toks[t] * emb_rb;
        float* dst = M.buf_cur + (size_t)t * IK_D;
        for (int b = 0; b < IK_D / QK_Q8; b++) {
            const uint8_t* B = er + (size_t)b * BQ8;
            uint16_t hs; memcpy(&hs, B, 2);
            float d = h2f(hs);
            const int8_t* qs = (const int8_t*)(B + 2);
            for (int j = 0; j < QK_Q8; j++) dst[b * QK_Q8 + j] = (float)qs[j] * d;
        }
        rmsnorm_f32(dst, dst, M.tok_norm, IK_D);
    }
    M.timers[0] += now_sec() - t_all0;

    float* blks[8];
    int nblk = 0;
    int visit = 0;

    for (int il = 0; il < IK_CORE_START; il++) {
        maybe_tap_dspark(visit++, T, M.buf_cur);
        run_layer(il, il, T, pos0, M.buf_cur, blks, &nblk, NULL, NULL, 0, NULL);
    }

    memcpy(M.buf_pre, M.buf_cur, (size_t)T * IK_D * 4);
    int nblk_pre = nblk;
    for (int b = 0; b < nblk_pre; b++)
        memcpy(M.buf_blk_pre + (size_t)b * T * IK_D, blks[b], (size_t)T * IK_D * 4);

    for (int pass = 0; pass < IK_N_PASS; pass++) {
        const float* bank_ptrs[4];
        int nbank = 0;
        if (pass > 0) {
            for (int b = nblk_pre + 1; b < nblk; b++) {
                memcpy(M.buf_bank + (size_t)nbank * T * IK_D, blks[b], (size_t)T * IK_D * 4);
                bank_ptrs[nbank] = M.buf_bank + (size_t)nbank * T * IK_D;
                nbank++;
            }
            float* bank_out_ptr = M.buf_bank + (size_t)nbank * T * IK_D;
            memcpy(bank_out_ptr, M.buf_cur, (size_t)T * IK_D * 4);
            memcpy(M.buf_cur, M.buf_pre, (size_t)T * IK_D * 4);
            nblk = nblk_pre;
            for (int b = 0; b < nblk_pre; b++) {
                memcpy(M.buf_blk + (size_t)b * T * IK_D, M.buf_blk_pre + (size_t)b * T * IK_D, (size_t)T * IK_D * 4);
                blks[b] = M.buf_blk + (size_t)b * T * IK_D;
            }
            for (int ip = IK_CORE_START; ip < IK_CORE_END; ip++) {
                int il = IK_NL_PHYS + (pass - 1) * (IK_CORE_END - IK_CORE_START) + (ip - IK_CORE_START);
                maybe_tap_dspark(visit++, T, M.buf_cur);
                run_layer(il, ip, T, pos0, M.buf_cur, blks, &nblk, NULL, bank_ptrs, nbank, bank_out_ptr);
            }
        } else {
            for (int ip = IK_CORE_START; ip < IK_CORE_END; ip++) {
                maybe_tap_dspark(visit++, T, M.buf_cur);
                run_layer(ip, ip, T, pos0, M.buf_cur, blks, &nblk, NULL, NULL, 0, NULL);
            }
        }
    }

    for (int il = IK_CORE_END; il < IK_NL_PHYS; il++) {
        maybe_tap_dspark(visit++, T, M.buf_cur);
        run_layer(il, il, T, pos0, M.buf_cur, blks, &nblk, NULL, NULL, 0, NULL);
    }

    if (need_logits || M.dspark_n > 0) {
        double th0 = now_sec();
        const float* bptrs[8];
        for (int b = 0; b < nblk; b++) bptrs[b] = blks[b];
        do_mix(bptrs, nblk, M.buf_cur, M.output_res_q, 1, T, M.buf_in0);
        const int h0 = (need_logits == 2) ? T - 1 : 0;      /* first row that gets logits */
        for (int t = 0; t < T; t++) {
            float* xt = M.buf_in0 + (size_t)t * IK_D;
            rmsnorm_f32(xt, xt, M.output_norm, IK_D);
            if (need_logits && t >= h0)
                q8_quantize(xt, IK_D, M.buf_xq + (size_t)t * IK_D, M.buf_xd + (size_t)t * (IK_D / QK_Q8));
        }
        maybe_tap_dspark(visit, T, M.buf_in0);
        if (need_logits) {
            if (__builtin_expect(g_imat && g_imat->enabled, 0)) {
                g_imat->cnt_head += (uint64_t)T;
                imat_accum(g_imat->head, M.buf_in0, IK_D, T);
            }
            const int Th = T - h0;
            const int8_t* hxq = M.buf_xq + (size_t)h0 * IK_D;
            const float* hxd = M.buf_xd + (size_t)h0 * (IK_D / QK_Q8);
            int use_h_head = (M.gpu.head_start != NULL);
            if (use_h_head) M.gpu.head_start(M.gpu.ctx, Th, M.buf_in0 + (size_t)h0 * IK_D, hxq, hxd);
            int r_head = use_h_head ? M.gpu.split_head : IK_V_UNPAD;
            IkOp h_op[1] = {{&M.output_head, r_head, NULL, hxq, hxd, IK_D, logits_out, IK_V_UNPAD}};
            run_ops(h_op, 1, Th);
            if (use_h_head) M.gpu.head_wait(M.gpu.ctx, Th, logits_out);
            if (apply_cap) {
                HARG.logits = logits_out; HARG.T = Th; HARG.r_end = IK_V_UNPAD;
                kp_run(head_softcap_job, NULL);
            }
        }
        M.timers[4] += now_sec() - th0;
    }
    M.timers[5] += now_sec() - t_all0;
    return 0;
}

int ik_forward(const int* toks, int T, int pos0, float* logits_out, int need_logits) {
    if (M.gpu.forward_step) {
        int poss[IK_MAXB];
        for (int t = 0; t < T; t++) poss[t] = pos0 + t;
        return M.gpu.forward_step(M.gpu.ctx, T, toks, poss, logits_out, need_logits, NULL, NULL);
    }
    return run_forward_internal(toks, T, pos0, logits_out, need_logits, 1);
}

static float* g_logits_buf = NULL;

static void merge_greedy_tk(int tk, int* out_tok, float* out_gap) {
    HARG.tk = tk;
    kp_run(head_greedy_job, NULL);
    int bt = 0; float m1 = -1e30f, m2 = -1e30f;
    for (int th = 0; th < KP.n; th++) {
        float a = HARG.m1[th], b = HARG.m2[th];
        if (a > m1 || (a == m1 && HARG.best_tok[th] < bt)) {
            float old = m1;
            m1 = a; bt = HARG.best_tok[th];
            if (old > m2) m2 = old;
            if (b > m2) m2 = b;
        } else if (a > m2) {
            m2 = a;
        }
    }
    *out_tok = bt;
    if (out_gap) *out_gap = IK_LOGIT_CAP * (tanhf(m1 / IK_LOGIT_CAP) - tanhf(m2 / IK_LOGIT_CAP));
}

int ik_forward_greedy(const int* toks, int T, int pos0, int* out_tok, float* out_gap) {
    if (M.gpu.forward_step) {
        int poss[IK_MAXB];
        for (int t = 0; t < T; t++) poss[t] = pos0 + t;
        return M.gpu.forward_step(M.gpu.ctx, T, toks, poss, NULL, 0, out_tok, out_gap);
    }
    if (!g_logits_buf) g_logits_buf = (float*)huge_alloc((size_t)IK_LOGB * IK_V_UNPAD * 4);
    int rc = run_forward_internal(toks, T, pos0, g_logits_buf, 2, 0);   /* the last row's logits only */
    if (rc == 0) {
        HARG.logits = g_logits_buf; HARG.T = 1;
        merge_greedy_tk(0, out_tok, out_gap);
    }
    return rc;
}

int ik_forward_greedy_batch(const int* toks, int T, int pos0, int* out_toks) {
    if (M.gpu.forward_step || T > IK_LOGB) return -1;
    if (!g_logits_buf) g_logits_buf = (float*)huge_alloc((size_t)IK_LOGB * IK_V_UNPAD * 4);
    int rc = run_forward_internal(toks, T, pos0, g_logits_buf, 1, 0);
    if (rc == 0) {
        HARG.logits = g_logits_buf; HARG.T = T;
        for (int tk = 0; tk < T; tk++) {
            merge_greedy_tk(tk, &out_toks[tk], NULL);
        }
    }
    return rc;
}

int ik_head_argmax(const float* x_2304, int* out_tok) {
    if (!x_2304 || !out_tok || !M.output_head.data) return -1;
    if (!g_logits_buf) g_logits_buf = (float*)huge_alloc((size_t)IK_LOGB * IK_V_UNPAD * 4);
    q8_quantize(x_2304, IK_D, M.buf_xq, M.buf_xd);
    int use_h_head = (M.gpu.head_start != NULL);
    if (use_h_head) M.gpu.head_start(M.gpu.ctx, 1, x_2304, M.buf_xq, M.buf_xd);
    int r_head = use_h_head ? M.gpu.split_head : IK_V_UNPAD;
    IkOp h_op[1] = {{&M.output_head, r_head, NULL, M.buf_xq, M.buf_xd, IK_D, g_logits_buf, IK_V_UNPAD}};
    run_ops(h_op, 1, 1);
    if (use_h_head) M.gpu.head_wait(M.gpu.ctx, 1, g_logits_buf);
    HARG.logits = g_logits_buf; HARG.T = 1;
    merge_greedy_tk(0, out_tok, NULL);
    return 0;
}

int ik_head_argmax_val(void* user, const float* x_2304) {
    (void)user;
    int tok = -1;
    if (ik_head_argmax(x_2304, &tok) != 0) return -1;
    return tok;
}

void ik_imatrix_start(void) {
    if (g_imat) { free(g_imat); g_imat = NULL; }
    g_imat = (IkImatrix*)calloc(1, sizeof(IkImatrix));
    if (g_imat) g_imat->enabled = 1;
}

static void imat_write_f32(FILE* f, const double* sum, uint64_t cnt, int n) {
    float buf[IK_QDIM];
    double inv = cnt > 0 ? (1.0 / (double)cnt) : 0.0;
    for (int i = 0; i < n; i++) buf[i] = (float)(sum[i] * inv);
    fwrite(buf, sizeof(float), (size_t)n, f);
}

int ik_imatrix_stop_and_save(const char* path) {
    if (!g_imat) return -1;
    g_imat->enabled = 0;
    FILE* f = fopen(path, "wb");
    if (!f) { perror("ik_imatrix fopen"); free(g_imat); g_imat = NULL; return -1; }
    fwrite("IKIMAT1\0", 1, 8, f);
    int32_t hdr[5] = {IK_NL_PHYS, IK_NE, IK_D, IK_QDIM, IK_FF};
    fwrite(hdr, sizeof(int32_t), 5, f);
    fwrite(g_imat->cnt_dense, sizeof(uint64_t), IK_NL_PHYS, f);
    fwrite(g_imat->cnt_exp, sizeof(uint64_t), IK_NL_PHYS * IK_NE, f);
    fwrite(&g_imat->cnt_head, sizeof(uint64_t), 1, f);
    for (int ip = 0; ip < IK_NL_PHYS; ip++) imat_write_f32(f, g_imat->in0[ip], g_imat->cnt_dense[ip], IK_D);
    for (int ip = 0; ip < IK_NL_PHYS; ip++) imat_write_f32(f, g_imat->in1[ip], g_imat->cnt_dense[ip], IK_D);
    for (int ip = 0; ip < IK_NL_PHYS; ip++) imat_write_f32(f, g_imat->in2[ip], g_imat->cnt_dense[ip], IK_D);
    for (int ip = 0; ip < IK_NL_PHYS; ip++) imat_write_f32(f, g_imat->attn[ip], g_imat->cnt_dense[ip], IK_QDIM);
    imat_write_f32(f, g_imat->head, g_imat->cnt_head, IK_D);

    double layer_gu_avg[IK_D], layer_dn_avg[IK_FF];
    for (int ip = 0; ip < IK_NL_PHYS; ip++) {
        memset(layer_gu_avg, 0, sizeof(layer_gu_avg));
        memset(layer_dn_avg, 0, sizeof(layer_dn_avg));
        uint64_t tot_c = 0;
        for (int e = 0; e < IK_NE; e++) {
            uint64_t c = g_imat->cnt_exp[ip][e];
            tot_c += c;
            for (int i = 0; i < IK_D; i++) layer_gu_avg[i] += g_imat->exp_gu[ip][e][i];
            for (int i = 0; i < IK_FF; i++) layer_dn_avg[i] += g_imat->exp_dn[ip][e][i];
        }
        for (int e = 0; e < IK_NE; e++) {
            if (g_imat->cnt_exp[ip][e] >= 4) imat_write_f32(f, g_imat->exp_gu[ip][e], g_imat->cnt_exp[ip][e], IK_D);
            else imat_write_f32(f, layer_gu_avg, tot_c, IK_D);
        }
        for (int e = 0; e < IK_NE; e++) {
            if (g_imat->cnt_exp[ip][e] >= 4) imat_write_f32(f, g_imat->exp_dn[ip][e], g_imat->cnt_exp[ip][e], IK_FF);
            else imat_write_f32(f, layer_dn_avg, tot_c, IK_FF);
        }
    }
    fclose(f);
    free(g_imat);
    g_imat = NULL;
    return 0;
}

void ik_free(void) {
    if (g_imat) { free(g_imat); g_imat = NULL; }
    if (M.conv_trace) { free(M.conv_trace); M.conv_trace = NULL; }
    if (M.map && M.map != MAP_FAILED) munmap(M.map, M.map_sz);
    if (M.fd > 0) close(M.fd);
    kp_destroy();
}
