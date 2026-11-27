/* qx_cpu.c - the routed experts of Qwen3.8-Flash-Next (GGUF qwen4exp) on the host CPUs (AVX2), on the thread pool
   of moe_cpu.c (moe_pool_run). Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.

   Weights: read in place, byte for byte as the file has them (the mapped file, or the RAM slots of xstore.c, which
   hold the same bytes). Five block formats occur in the experts of these files:
     IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (256 values a block: gate and up), IQ4_NL (32), Q2_0 (64; down).
   Each 32 values of a block are an int8 "code" vector c (grid magnitude with sign, IQ4 level, or Q2_0 code - 1), an
   integer multiplier m (from the block's sub-scale) and a float scale dw (from the block's fp16 scale):
     IQ2_S   w = d * (2s + 1) / 8 * c   (two 16-value halves, each its own s)     m = 2s + 1,  dw = d / 8
     IQ3_XXS w = d * (2s + 1) / 4 * c                                             m = 2s + 1,  dw = d / 4
     IQ3_S   w = d * (2s + 1) * c                                                 m = 2s + 1,  dw = d
     IQ4_XS  w = d * (ls - 32) * c                                                m = ls - 32, dw = d
     IQ4_NL  w = d * c,  Q2_0 w = d * c                                           m = 1,       dw = d
   (the same values as kindi/qwen4exp_quant.py; qx_dequant() writes them with that module's float arithmetic, and
   tools/qx_cpu_test.py checks them bit for bit).

   Activations: per 32 values one scale dx = fp16(amax / 127) and int8 values rint(x * 127 / amax) (the rounding of
   moe_cpu.c). The dot product of a row with one token keeps 8 fp32 lanes; lane i takes values 4i .. 4i + 3 of every
   32-value block, block by block in order:
     L_i = sum(c * q) over its 4 values (exact, int32);  A_i = A_i + (float)(L_i * m) * (dw * dx)
   (m of the lane's half for IQ2_S; no FMA: -ffp-contract=off), and at the end of the row
     result = ((A0 + A4) + (A2 + A6)) + ((A1 + A5) + (A3 + A7)).
   The integer parts are exact and the float operations are the same in the same order, so the AVX2 kernel and the
   scalar kernel (qx_row_dot_ref) give the same bits, and a token's values do not depend on the batch, the pairing of
   tokens or the threads.

   One call (one MoE layer, T tokens, nsel experts each): the union of the experts; per expert its (token, slot)
   pairs. Step 1: gate and up rows of each expert for all its pairs (the codes of a block are decoded once for up
   to 8 pairs), h = silu(gate) * up. Step 2: h quantized per pair. Step 3: down rows, one value per pair and row.
   Step 4: out[t] = sum over the token's slots in order of weight * value (acc = acc + w * v from 0). */
#define _GNU_SOURCE
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define __device__
#include "nbq_grids.h"

typedef void (*moe_pool_fn)(int t, int nact, void* arg);
int moe_pool_run(moe_pool_fn fn, void* arg);
void moe_pool_barrier(void);
int moe_pool_threads(void);
int moe_pool_node_of(int t);
int moe_pool_nthr_node(int nd);
float moe_pool_share0(void);
void* moe_node_alloc(size_t bytes, int node);
int xs_acquire(int l, int n, const int* ids, const uint8_t** ptr);
void xs_release(int l, int n, const int* ids);

enum { QX_IQ3_XXS = 18, QX_IQ4_NL = 20, QX_IQ3_S = 21, QX_IQ2_S = 22, QX_IQ4_XS = 23, QX_Q2_0 = 42 };
/* "nibble" forms of the i-quants for the CPU (qx_transcode): the same integer values, 4-bit codes into a 16-entry
   table per format, so a sub-block decodes with one table shuffle instead of 8 grid lookups. Type = 100 + the
   original type. Block of 256 values, 146 bytes: d (fp16, as in the file), the sub-scales m (2 bytes per 32-value
   sub-block: values 0-15, 16-31), then 16 bytes of codes per sub-block (byte k: value k low, value k + 16 high). */
enum { QX_NIB = 100, QX_N_IQ3_XXS = 118, QX_N_IQ3_S = 121, QX_N_IQ2_S = 122 };
static int8_t qx_nlut[3][16];             /* IQ3_XXS, IQ3_S, IQ2_S: code (index | sign << 3) -> value */
static inline int qx_nlut_of(int ty) { return ty == QX_N_IQ3_XXS ? 0 : ty == QX_N_IQ3_S ? 1 : 2; }
static const float qx_ndw[3] = {0.25f, 1.f, 0.125f};
static const uint8_t qx_nmag[3][8] = {{4, 12, 20, 28, 36, 44, 52, 62}, {1, 3, 5, 7, 9, 11, 13, 15}, {8, 25, 43, 0, 0, 0, 0, 0}};
#define QX_MAXL 256
#define QX_MAXSEL 16
#define QX_PB 8                   /* pairs per pass over a row's blocks */

static const int8_t qx_lv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const float qx_lv4f[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static uint32_t qx_q2lut[256];    /* a Q2_0 byte -> its 4 values (code - 1) as int8 */
static uint8_t qx_ksign[128];     /* IQ3_XXS: 7 sign bits -> 8 (bit 7 = parity) */
static int qx_ready;

static void qx_init(void) {
    if (qx_ready) return;
    for (int b = 0; b < 256; b++) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) v |= (uint32_t)(uint8_t)(int8_t)(((b >> (2 * k)) & 3) - 1) << (8 * k);
        qx_q2lut[b] = v;
    }
    for (int s = 0; s < 128; s++) qx_ksign[s] = (uint8_t)(s | ((__builtin_popcount(s) & 1) << 7));
    for (int t = 0; t < 3; t++)
        for (int c = 0; c < 16; c++) qx_nlut[t][c] = (int8_t)(c & 8 ? -(int)qx_nmag[t][c & 7] : (int)qx_nmag[t][c & 7]);
    qx_ready = 1;
}

static inline float qx_f16(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }

/* bytes and values of a block */
static inline __attribute__((always_inline)) int qx_block(int ty, int* bytes) {
    switch (ty) {
    case QX_IQ2_S: *bytes = 82; return 256;
    case QX_IQ3_XXS: *bytes = 98; return 256;
    case QX_IQ3_S: *bytes = 110; return 256;
    case QX_IQ4_XS: *bytes = 136; return 256;
    case QX_IQ4_NL: *bytes = 18; return 32;
    case QX_Q2_0: *bytes = 18; return 64;
    case QX_N_IQ3_XXS: case QX_N_IQ3_S: case QX_N_IQ2_S: *bytes = 146; return 256;
    }
    *bytes = 0; return 0;
}
long qx_row_bytes(int ty, int n) { int b, per = qx_block(ty, &b); return per && n % per == 0 ? (long)(n / per) * b : -1; }

/* ---------------------------------------------------------------- the code vectors (AVX2) */
static inline __attribute__((always_inline)) __m256i qx_signs(uint32_t bits) {           /* bit i set -> lane i all ones */
    const __m256i sh = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bm = _mm256_set1_epi64x(0x8040201008040201LL);
    __m256i s = _mm256_shuffle_epi8(_mm256_set1_epi32((int)bits), sh);
    return _mm256_cmpeq_epi8(_mm256_and_si256(s, bm), bm);
}
static inline __attribute__((always_inline)) __m256i qx_apply(__m256i mag, __m256i neg) { return _mm256_sub_epi8(_mm256_xor_si256(mag, neg), neg); }

/* the code decodes (item 8): bit 0 = Q2_0 by vector unpacking instead of a byte table, bit 1 = the i-quant grid
   rows by AVX2 gathers instead of scalar loads; the codes are the same bytes either way (qx_cpu_test.py) */
int qx_vdec = 3;

/* Q2_0, 32 values from 8 bytes: each byte goes to the 4 bytes of its dword, each of those keeps its own 2-bit field
   (two masked compares), code - 1 = -1 - lo - 2 hi with lo, hi = 0 or -1 */
static inline __attribute__((always_inline)) __m256i qx_q2_vec(const uint8_t* q) {
    const __m256i rep = _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7);
    const __m256i m1 = _mm256_set1_epi32(0x40100401), m2 = _mm256_set1_epi32((int)0x80200802u);
    long long w; memcpy(&w, q, 8);
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi64x(w), rep);
    const __m256i lo = _mm256_cmpeq_epi8(_mm256_and_si256(v, m1), m1);
    const __m256i hi = _mm256_cmpeq_epi8(_mm256_and_si256(v, m2), m2);
    return _mm256_sub_epi8(_mm256_sub_epi8(_mm256_sub_epi8(_mm256_set1_epi8(-1), lo), hi), hi);
}

/* sub-block j (32 values) of a block: the codes, and m (two for IQ2_S: m[0] values 0-15, m[1] values 16-31) and dw */
static inline __attribute__((always_inline)) __m256i qx_codes(int ty, const uint8_t* b, int j, int* m, float* dw) {
    switch (ty) {
    case QX_IQ2_S: {
        const float d = qx_f16(b);
        const uint8_t* qs = b + 2 + 4 * j;
        const int qh = b[66 + j], sc = b[74 + j];
        __m256i mag;
        if (qx_vdec & 2) {                                /* index k = qs[k] | bits 2k, 2k + 1 of qh at bits 8-9 */
            int w4; memcpy(&w4, qs, 4);
            const __m128i hb = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(qh), _mm_setr_epi32(0, 2, 4, 6)), _mm_set1_epi32(3));
            const __m128i ix = _mm_or_si128(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(w4)), _mm_slli_epi32(hb, 8));
            mag = _mm256_i32gather_epi64((const long long*)nbq_iq2s_grid, ix, 8);
        } else
            mag = _mm256_setr_epi64x((long long)nbq_iq2s_grid[qs[0] | ((qh << 8) & 0x300)],
                                     (long long)nbq_iq2s_grid[qs[1] | ((qh << 6) & 0x300)],
                                     (long long)nbq_iq2s_grid[qs[2] | ((qh << 4) & 0x300)],
                                     (long long)nbq_iq2s_grid[qs[3] | ((qh << 2) & 0x300)]);
        uint32_t sg; memcpy(&sg, b + 34 + 4 * j, 4);
        m[0] = 2 * (sc & 15) + 1; m[1] = 2 * (sc >> 4) + 1; *dw = d * 0.125f;
        return qx_apply(mag, qx_signs(sg));
    }
    case QX_IQ3_XXS: {
        const float d = qx_f16(b);
        const uint8_t* qs = b + 2 + 8 * j;
        uint32_t aux; memcpy(&aux, b + 66 + 4 * j, 4);
        __m256i mag;
        if (qx_vdec & 2)
            mag = _mm256_i32gather_epi32((const int*)nbq_iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)qs)), 4);
        else
            mag = _mm256_setr_epi32((int)nbq_iq3xxs_grid[qs[0]], (int)nbq_iq3xxs_grid[qs[1]],
                                    (int)nbq_iq3xxs_grid[qs[2]], (int)nbq_iq3xxs_grid[qs[3]],
                                    (int)nbq_iq3xxs_grid[qs[4]], (int)nbq_iq3xxs_grid[qs[5]],
                                    (int)nbq_iq3xxs_grid[qs[6]], (int)nbq_iq3xxs_grid[qs[7]]);
        const uint32_t sg = qx_ksign[aux & 127] | (uint32_t)qx_ksign[(aux >> 7) & 127] << 8 |
                            (uint32_t)qx_ksign[(aux >> 14) & 127] << 16 | (uint32_t)qx_ksign[(aux >> 21) & 127] << 24;
        m[0] = m[1] = 2 * (int)(aux >> 28) + 1; *dw = d * 0.25f;
        return qx_apply(mag, qx_signs(sg));
    }
    case QX_IQ3_S: {
        const float d = qx_f16(b);
        const uint8_t* qs = b + 2 + 8 * j;
        const int qh = b[66 + j];
        __m256i mag;
        if (qx_vdec & 2) {                                /* index k = qs[k] | bit k of qh at bit 8 */
            const __m256i hb = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(qh), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)),
                                                _mm256_set1_epi32(1));
            const __m256i ix = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)qs)), _mm256_slli_epi32(hb, 8));
            mag = _mm256_i32gather_epi32((const int*)nbq_iq3s_grid, ix, 4);
        } else
            mag = _mm256_setr_epi32((int)nbq_iq3s_grid[qs[0] | ((qh << 8) & 256)], (int)nbq_iq3s_grid[qs[1] | ((qh << 7) & 256)],
                                    (int)nbq_iq3s_grid[qs[2] | ((qh << 6) & 256)], (int)nbq_iq3s_grid[qs[3] | ((qh << 5) & 256)],
                                    (int)nbq_iq3s_grid[qs[4] | ((qh << 4) & 256)], (int)nbq_iq3s_grid[qs[5] | ((qh << 3) & 256)],
                                    (int)nbq_iq3s_grid[qs[6] | ((qh << 2) & 256)], (int)nbq_iq3s_grid[qs[7] | ((qh << 1) & 256)]);
        uint32_t sg; memcpy(&sg, b + 74 + 4 * j, 4);
        m[0] = m[1] = 2 * ((b[106 + (j >> 1)] >> (4 * (j & 1))) & 15) + 1; *dw = d;
        return qx_apply(mag, qx_signs(sg));
    }
    case QX_IQ4_XS: {
        const float d = qx_f16(b);
        uint16_t sh; memcpy(&sh, b + 2, 2);
        const int ls = ((b[4 + j / 2] >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4);
        const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)qx_lv4));
        __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(b + 8 + 16 * j)));
        v = _mm256_and_si256(_mm256_blend_epi32(v, _mm256_srli_epi16(v, 4), 0xF0), _mm256_set1_epi8(15));
        m[0] = m[1] = ls - 32; *dw = d;
        return _mm256_shuffle_epi8(tbl, v);
    }
    case QX_N_IQ3_XXS: case QX_N_IQ3_S: case QX_N_IQ2_S: {
        const int t = qx_nlut_of(ty);
        const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)qx_nlut[t]));
        __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(b + 18 + 16 * j)));
        v = _mm256_and_si256(_mm256_blend_epi32(v, _mm256_srli_epi16(v, 4), 0xF0), _mm256_set1_epi8(15));
        m[0] = b[2 + 2 * j]; m[1] = b[3 + 2 * j]; *dw = qx_f16(b) * qx_ndw[t];
        return _mm256_shuffle_epi8(tbl, v);
    }
    case QX_IQ4_NL: {
        const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)qx_lv4));
        __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(b + 2)));
        v = _mm256_and_si256(_mm256_blend_epi32(v, _mm256_srli_epi16(v, 4), 0xF0), _mm256_set1_epi8(15));
        m[0] = m[1] = 1; *dw = qx_f16(b);
        return _mm256_shuffle_epi8(tbl, v);
    }
    default: {                                            /* Q2_0: sub-block j is bytes 2 + 8j .. 2 + 8j + 7 */
        const uint8_t* q = b + 2 + 8 * j;
        m[0] = m[1] = 1; *dw = qx_f16(b);
        if (qx_vdec & 1) return qx_q2_vec(q);
        return _mm256_setr_epi32((int)qx_q2lut[q[0]], (int)qx_q2lut[q[1]], (int)qx_q2lut[q[2]], (int)qx_q2lut[q[3]],
                                 (int)qx_q2lut[q[4]], (int)qx_q2lut[q[5]], (int)qx_q2lut[q[6]], (int)qx_q2lut[q[7]]);
    }
    }
}

/* the 8 lane sums of c * q times the sub-scale (lane i: values 4i .. 4i + 3), int32: mw holds the multiplier of each
   int16 pair sum (m[0] for lanes 0-3, m[1] for lanes 4-7), so (a + b) * m comes exactly out of the madd, the same
   integers as a multiply after the sum, without the slow 32-bit multiply */
static inline __attribute__((always_inline)) __m256i qx_lanes_m(__m256i c, __m256i q, __m256i mw) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(c), _mm256_sign_epi8(q, c)), mw);
}
static inline float qx_hsum(__m256 v) {          /* ((A0 + A4) + (A2 + A6)) + ((A1 + A5) + (A3 + A7)) */
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

/* one row (ncols values) with np <= QX_PB quantized activation vectors: acc[i] */
static void qx_row_dot_n(int ty, const uint8_t* row, int ncols, int np, const int8_t* const* xq, const float* const* xd,
                         float* acc) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    __m256 A[QX_PB];
    for (int i = 0; i < np; i++) A[i] = _mm256_setzero_ps();
    for (int bk = 0; bk < ncols / per; bk++) {
        const uint8_t* b = row + (size_t)bk * bytes;
        for (int j = 0; j < sub; j++) {
            int m[2]; float dw;
            const __m256i c = qx_codes(ty, b, j, m, &dw);
            const __m256i mw = _mm256_setr_m128i(_mm_set1_epi16((short)m[0]), _mm_set1_epi16((short)m[1]));
            const int blk = bk * sub + j;
            for (int i = 0; i < np; i++) {
                const __m256i l = qx_lanes_m(c, _mm256_loadu_si256((const __m256i*)(xq[i] + 32 * blk)), mw);
                const float s = dw * xd[i][blk];
                A[i] = _mm256_add_ps(A[i], _mm256_mul_ps(_mm256_cvtepi32_ps(l), _mm256_set1_ps(s)));
            }
        }
    }
    for (int i = 0; i < np; i++) acc[i] = qx_hsum(A[i]);
}

/* a row decoded once (for an expert with more than QX_PB pairs in a call: prompt chunks): the codes, the sub-scale
   words and dw of each 32-value sub-block; at most QX_MAXSB sub-blocks (4096 values) */
#define QX_MAXSB 128
typedef struct { __m256i c[QX_MAXSB], mw[QX_MAXSB]; float dw[QX_MAXSB]; } qx_drow;
int qx_predec = QX_PB + 1;        /* pairs from which a row is decoded once per call (a large value turns it off) */

static void qx_row_decode(int ty, const uint8_t* row, int ncols, qx_drow* D) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    for (int bk = 0; bk < ncols / per; bk++)
        for (int j = 0; j < sub; j++) {
            int m[2];
            const int blk = bk * sub + j;
            D->c[blk] = qx_codes(ty, row + (size_t)bk * bytes, j, m, &D->dw[blk]);
            D->mw[blk] = _mm256_setr_m128i(_mm_set1_epi16((short)m[0]), _mm_set1_epi16((short)m[1]));
        }
}
/* qx_row_dot_n on a decoded row: the same operations in the same order, so the same bits */
static void qx_row_dot_d(const qx_drow* D, int ncols, int np, const int8_t* const* xq, const float* const* xd, float* acc) {
    __m256 A[QX_PB];
    for (int i = 0; i < np; i++) A[i] = _mm256_setzero_ps();
    for (int blk = 0; blk < ncols / 32; blk++) {
        const __m256i c = D->c[blk], mw = D->mw[blk];
        const float dw = D->dw[blk];
        for (int i = 0; i < np; i++) {
            const __m256i l = qx_lanes_m(c, _mm256_loadu_si256((const __m256i*)(xq[i] + 32 * blk)), mw);
            const float s = dw * xd[i][blk];
            A[i] = _mm256_add_ps(A[i], _mm256_mul_ps(_mm256_cvtepi32_ps(l), _mm256_set1_ps(s)));
        }
    }
    for (int i = 0; i < np; i++) acc[i] = qx_hsum(A[i]);
}

/* ---------------------------------------------------------------- gate/up rows on 256-value activation blocks
   The expert input x is quantized in blocks of 256 values with one scale each (qx_quant256). Per weight block (256
   values for the i-quants, 32 for IQ4_NL) the products of all its 32-value sub-blocks are summed as integers, each
   times its sub-scale m (exact in int32), and then one float step: A = A + (float)I * (dw * xs). The card does the
   same steps (qx_gpu.cu row_dot256), so the values are the same bits; the integer sums make the order of the
   lanes irrelevant. */
static inline int qx_hsum_epi32(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xB1));
    return _mm_cvtsi128_si32(a);
}
static void qx_row_dot256_n(int ty, const uint8_t* row, int ncols, int np, const int8_t* const* xq,
                            const float* const* xs, float* acc, int ab) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    float A[QX_PB];
    for (int i = 0; i < np; i++) A[i] = 0.f;
    for (int bk = 0; bk < ncols / per; bk++) {
        const uint8_t* b = row + (size_t)bk * bytes;
        __m256i I[QX_PB];
        for (int i = 0; i < np; i++) I[i] = _mm256_setzero_si256();
        float dw = 0.f;
        for (int j = 0; j < sub; j++) {
            int m[2];
            const __m256i c = qx_codes(ty, b, j, m, &dw);
            const __m256i mw = _mm256_setr_m128i(_mm_set1_epi16((short)m[0]), _mm_set1_epi16((short)m[1]));
            const int off = 32 * (bk * sub + j);
            for (int i = 0; i < np; i++)
                I[i] = _mm256_add_epi32(I[i], qx_lanes_m(c, _mm256_loadu_si256((const __m256i*)(xq[i] + off)), mw));
        }
        const int xb = bk * per / ab;
        for (int i = 0; i < np; i++) A[i] = A[i] + (float)qx_hsum_epi32(I[i]) * (dw * xs[i][xb]);
    }
    for (int i = 0; i < np; i++) acc[i] = A[i];
}
/* the same on a decoded row (qx_row_decode) */
static void qx_row_dot256_d(int ty, const qx_drow* D, int ncols, int np, const int8_t* const* xq,
                            const float* const* xs, float* acc, int ab) {
    int bytes; const int sub = qx_block(ty, &bytes) / 32;
    float A[QX_PB];
    for (int i = 0; i < np; i++) A[i] = 0.f;
    for (int bk = 0; bk < ncols / (32 * sub); bk++) {
        __m256i I[QX_PB];
        for (int i = 0; i < np; i++) I[i] = _mm256_setzero_si256();
        for (int j = 0; j < sub; j++) {
            const int blk = bk * sub + j;
            for (int i = 0; i < np; i++)
                I[i] = _mm256_add_epi32(I[i], qx_lanes_m(D->c[blk], _mm256_loadu_si256((const __m256i*)(xq[i] + 32 * blk)), D->mw[blk]));
        }
        const float dw = D->dw[bk * sub];
        const int xb = bk * sub * 32 / ab;
        for (int i = 0; i < np; i++) A[i] = A[i] + (float)qx_hsum_epi32(I[i]) * (dw * xs[i][xb]);
    }
    for (int i = 0; i < np; i++) acc[i] = A[i];
}

/* ---------------------------------------------------------------- few vectors (decode and windows, item 8)
   qx_row_dot_n / qx_row_dot256_n for np = 1 .. 4 with the block type and np constants (one copy per pair: the
   decode of that type inlined, the accumulators in registers, no pointer tables): the same operations in the same
   order, so the same bits. qx_one = 0 turns them off (A/B). */
int qx_one = 1;
#define QX_NPMAX 4
static inline __attribute__((always_inline)) void qx_r32_k(const int ty, const int np, const uint8_t* row, int ncols,
                                                            const int8_t* const* xq, const float* const* xd, float* out) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    __m256 A[QX_NPMAX];
    for (int i = 0; i < np; i++) A[i] = _mm256_setzero_ps();
    for (int bk = 0; bk < ncols / per; bk++) {
        const uint8_t* b = row + (size_t)bk * bytes;
        for (int j = 0; j < sub; j++) {
            int m[2]; float dw;
            const __m256i c = qx_codes(ty, b, j, m, &dw);
            const __m256i mw = _mm256_setr_m128i(_mm_set1_epi16((short)m[0]), _mm_set1_epi16((short)m[1]));
            const int blk = bk * sub + j;
            for (int i = 0; i < np; i++) {
                const __m256i l = qx_lanes_m(c, _mm256_loadu_si256((const __m256i*)(xq[i] + 32 * blk)), mw);
                const float s_ = dw * xd[i][blk];
                A[i] = _mm256_add_ps(A[i], _mm256_mul_ps(_mm256_cvtepi32_ps(l), _mm256_set1_ps(s_)));
            }
        }
    }
    for (int i = 0; i < np; i++) out[i] = qx_hsum(A[i]);
}
static inline __attribute__((always_inline)) void qx_r256_k(const int ty, const int np, const uint8_t* row, int ncols,
                                                             const int8_t* const* xq, const float* const* xs, float* out,
                                                             const int ab) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    float A[QX_NPMAX];
    for (int i = 0; i < np; i++) A[i] = 0.f;
    for (int bk = 0; bk < ncols / per; bk++) {
        const uint8_t* b = row + (size_t)bk * bytes;
        __m256i I[QX_NPMAX];
        for (int i = 0; i < np; i++) I[i] = _mm256_setzero_si256();
        float dw = 0.f;
        for (int j = 0; j < sub; j++) {
            int m[2];
            const __m256i c = qx_codes(ty, b, j, m, &dw);
            const __m256i mw = _mm256_setr_m128i(_mm_set1_epi16((short)m[0]), _mm_set1_epi16((short)m[1]));
            const int off = 32 * (bk * sub + j);
            for (int i = 0; i < np; i++)
                I[i] = _mm256_add_epi32(I[i], qx_lanes_m(c, _mm256_loadu_si256((const __m256i*)(xq[i] + off)), mw));
        }
        const int xb = bk * per / ab;
        for (int i = 0; i < np; i++) A[i] = A[i] + (float)qx_hsum_epi32(I[i]) * (dw * xs[i][xb]);
    }
    for (int i = 0; i < np; i++) out[i] = A[i];
}
#define QX_TYPES(F) F(QX_IQ2_S) F(QX_IQ3_XXS) F(QX_IQ3_S) F(QX_IQ4_XS) F(QX_IQ4_NL) F(QX_Q2_0) \
                    F(QX_N_IQ3_XXS) F(QX_N_IQ3_S) F(QX_N_IQ2_S)
/* nr rows (rb bytes apart) with np <= QX_NPMAX vectors: out[r * os + i]; returns 0 if the type or np has no copy
   here (the caller then takes the general kernels) */
static int qx_rows32_k(int ty, int np, const uint8_t* row, long rb, int nr, int ncols, const int8_t* const* xq,
                       const float* const* xd, float* out, long os) {
    switch (ty * 8 + np) {
#define QX_C1(T, N) case T * 8 + N: for (int r = 0; r < nr; r++) qx_r32_k(T, N, row + (size_t)r * rb, ncols, xq, xd, out + r * os); return 1;
#define QX_C(T) QX_C1(T, 1) QX_C1(T, 2) QX_C1(T, 3) QX_C1(T, 4)
    QX_TYPES(QX_C)
#undef QX_C
#undef QX_C1
    }
    return 0;
}
static int qx_rows256_k(int ty, int np, const uint8_t* row, long rb, int nr, int ncols, const int8_t* const* xq,
                        const float* const* xs, float* out, long os, int ab) {
    switch (ty * 8 + np) {
#define QX_C1(T, N) case T * 8 + N: for (int r = 0; r < nr; r++) qx_r256_k(T, N, row + (size_t)r * rb, ncols, xq, xs, out + r * os, ab); return 1;
#define QX_C(T) QX_C1(T, 1) QX_C1(T, 2) QX_C1(T, 3) QX_C1(T, 4)
    QX_TYPES(QX_C)
#undef QX_C
#undef QX_C1
    }
    return 0;
}

/* ---------------------------------------------------------------- scalar reference (the test's second kernel) */
static void qx_codes_ref(int ty, const uint8_t* b, int j, int8_t c[32], int* m, float* dw) {
    switch (ty) {
    case QX_IQ2_S: {
        const int qh = b[66 + j], sc = b[74 + j];
        for (int l = 0; l < 4; l++) {
            const uint64_t g = nbq_iq2s_grid[b[2 + 4 * j + l] | (((qh >> (2 * l)) & 3) << 8)];
            const int sg = b[34 + 4 * j + l];
            for (int k = 0; k < 8; k++) { int v = (int)((g >> (8 * k)) & 255); c[8 * l + k] = (int8_t)((sg >> k) & 1 ? -v : v); }
        }
        m[0] = 2 * (sc & 15) + 1; m[1] = 2 * (sc >> 4) + 1; *dw = qx_f16(b) * 0.125f;
        return;
    }
    case QX_IQ3_XXS: {
        uint32_t aux; memcpy(&aux, b + 66 + 4 * j, 4);
        for (int l = 0; l < 4; l++) {
            const int s7 = (aux >> (7 * l)) & 127, sg = s7 | ((__builtin_popcount(s7) & 1) << 7);
            for (int h = 0; h < 2; h++) {
                const uint32_t g = nbq_iq3xxs_grid[b[2 + 8 * j + 2 * l + h]];
                for (int k = 0; k < 4; k++) { int v = (int)((g >> (8 * k)) & 255), i = 4 * h + k; c[8 * l + i] = (int8_t)((sg >> i) & 1 ? -v : v); }
            }
        }
        m[0] = m[1] = 2 * (int)(aux >> 28) + 1; *dw = qx_f16(b) * 0.25f;
        return;
    }
    case QX_IQ3_S: {
        const int qh = b[66 + j];
        for (int i = 0; i < 8; i++) {
            const uint32_t g = nbq_iq3s_grid[b[2 + 8 * j + i] | (((qh >> i) & 1) << 8)];
            for (int k = 0; k < 4; k++) {
                const int n = 4 * i + k, v = (int)((g >> (8 * k)) & 255);
                c[n] = (int8_t)((b[74 + 4 * j + n / 8] >> (n % 8)) & 1 ? -v : v);
            }
        }
        m[0] = m[1] = 2 * ((b[106 + (j >> 1)] >> (4 * (j & 1))) & 15) + 1; *dw = qx_f16(b);
        return;
    }
    case QX_IQ4_XS: {
        uint16_t sh; memcpy(&sh, b + 2, 2);
        for (int k = 0; k < 16; k++) { c[k] = qx_lv4[b[8 + 16 * j + k] & 15]; c[k + 16] = qx_lv4[b[8 + 16 * j + k] >> 4]; }
        m[0] = m[1] = (((b[4 + j / 2] >> (4 * (j & 1))) & 15) | (((sh >> (2 * j)) & 3) << 4)) - 32; *dw = qx_f16(b);
        return;
    }
    case QX_IQ4_NL:
        for (int k = 0; k < 16; k++) { c[k] = qx_lv4[b[2 + k] & 15]; c[k + 16] = qx_lv4[b[2 + k] >> 4]; }
        m[0] = m[1] = 1; *dw = qx_f16(b);
        return;
    case QX_N_IQ3_XXS: case QX_N_IQ3_S: case QX_N_IQ2_S: {
        const int t = qx_nlut_of(ty);
        for (int k = 0; k < 16; k++) { c[k] = qx_nlut[t][b[18 + 16 * j + k] & 15]; c[k + 16] = qx_nlut[t][b[18 + 16 * j + k] >> 4]; }
        m[0] = b[2 + 2 * j]; m[1] = b[3 + 2 * j]; *dw = qx_f16(b) * qx_ndw[t];
        return;
    }
    default:
        for (int n = 0; n < 32; n++) c[n] = (int8_t)(((b[2 + 8 * j + n / 4] >> (2 * (n % 4))) & 3) - 1);
        m[0] = m[1] = 1; *dw = qx_f16(b);
        return;
    }
}

float qx_row_dot_ref(int ty, const uint8_t* row, int ncols, const int8_t* xq, const float* xd) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    float A[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int bk = 0; bk < ncols / per; bk++)
        for (int j = 0; j < sub; j++) {
            int8_t c[32]; int m[2]; float dw;
            qx_codes_ref(ty, row + (size_t)bk * bytes, j, c, m, &dw);
            const int8_t* q = xq + 32 * (bk * sub + j);
            const float s = dw * xd[bk * sub + j];
            for (int i = 0; i < 8; i++) {
                int li = 0;
                for (int k = 4 * i; k < 4 * i + 4; k++) li += c[k] * q[k];
                A[i] = A[i] + (float)(li * m[i >> 2]) * s;
            }
        }
    return ((A[0] + A[4]) + (A[2] + A[6])) + ((A[1] + A[5]) + (A[3] + A[7]));
}
/* the AVX2 kernel on np vectors (tests) */
void qx_row_dot(int ty, const uint8_t* row, int ncols, int np, const int8_t* xq, const float* xd, float* out) {
    qx_init();
    if (np <= QX_NPMAX && qx_one) {
        const int8_t* q[QX_NPMAX]; const float* d[QX_NPMAX];
        for (int i = 0; i < np; i++) { q[i] = xq + (size_t)i * ncols; d[i] = xd + (size_t)i * (ncols / 32); }
        if (qx_rows32_k(ty, np, row, 0, 1, ncols, q, d, out, 0)) return;
    }
    const int8_t* q[QX_PB]; const float* d[QX_PB];
    for (int i0 = 0; i0 < np; i0 += QX_PB) {
        const int n = np - i0 < QX_PB ? np - i0 : QX_PB;
        for (int i = 0; i < n; i++) { q[i] = xq + (size_t)(i0 + i) * ncols; d[i] = xd + (size_t)(i0 + i) * (ncols / 32); }
        qx_row_dot_n(ty, row, ncols, n, q, d, out + i0);
    }
}

/* scalar reference of the 256-value path (test), and the AVX2 kernel on np vectors (xs: ncols / 256 scales each) */
float qx_row_dot256_ref(int ty, const uint8_t* row, int ncols, const int8_t* xq, const float* xs) {
    int bytes; const int per = qx_block(ty, &bytes), sub = per / 32;
    float A = 0.f;
    for (int bk = 0; bk < ncols / per; bk++) {
        int I = 0; float dw = 0.f;
        for (int j = 0; j < sub; j++) {
            int8_t c[32]; int m[2];
            qx_codes_ref(ty, row + (size_t)bk * bytes, j, c, m, &dw);
            const int8_t* q = xq + 32 * (bk * sub + j);
            for (int k = 0; k < 32; k++) I += c[k] * q[k] * m[k >> 4];
        }
        A = A + (float)I * (dw * xs[bk * per / 256]);
    }
    return A;
}
/* np vectors, activation blocks of ab values (xs: ncols / ab scales each): the kernels of the down rows on DN64 */
void qx_row_dot_ab(int ty, const uint8_t* row, int ncols, int np, const int8_t* xq, const float* xs, float* out, int ab) {
    qx_init();
    const int8_t* q[QX_PB]; const float* d[QX_PB];
    for (int i0 = 0; i0 < np; i0 += QX_PB) {
        const int n = np - i0 < QX_PB ? np - i0 : QX_PB;
        for (int i = 0; i < n; i++) { q[i] = xq + (size_t)(i0 + i) * ncols; d[i] = xs + (size_t)(i0 + i) * (ncols / ab); }
        if (!(qx_one && n <= QX_NPMAX && qx_rows256_k(ty, n, row, 0, 1, ncols, q, d, out + i0, 0, ab)))
            qx_row_dot256_n(ty, row, ncols, n, q, d, out + i0, ab);
    }
}
void qx_row_dot256(int ty, const uint8_t* row, int ncols, int np, const int8_t* xq, const float* xs, float* out) {
    qx_init();
    if (np <= QX_NPMAX && qx_one) {
        const int8_t* q[QX_NPMAX]; const float* d[QX_NPMAX];
        for (int i = 0; i < np; i++) { q[i] = xq + (size_t)i * ncols; d[i] = xs + (size_t)i * (ncols / 256); }
        if (qx_rows256_k(ty, np, row, 0, 1, ncols, q, d, out, 0, 256)) return;
    }
    const int8_t* q[QX_PB]; const float* d[QX_PB];
    for (int i0 = 0; i0 < np; i0 += QX_PB) {
        const int n = np - i0 < QX_PB ? np - i0 : QX_PB;
        for (int i = 0; i < n; i++) { q[i] = xq + (size_t)(i0 + i) * ncols; d[i] = xs + (size_t)(i0 + i) * (ncols / 256); }
        qx_row_dot256_n(ty, row, ncols, n, q, d, out + i0, 256);
    }
}

/* nblk blocks of an i-quant type into its nibble form (dst: nblk x 146 bytes); returns the nibble type, or -1 if
   the type has none or a value is not in its table */
int qx_transcode(int ty, const uint8_t* src, long nblk, uint8_t* dst) {
    qx_init();
    const int nt = ty == QX_IQ3_XXS ? QX_N_IQ3_XXS : ty == QX_IQ3_S ? QX_N_IQ3_S : ty == QX_IQ2_S ? QX_N_IQ2_S : -1;
    if (nt < 0) return -1;
    const int t = qx_nlut_of(nt);
    int bytes; qx_block(ty, &bytes);
    int8_t code_of[256];                  /* value (as uint8) -> code, -1 if none */
    memset(code_of, -1, sizeof code_of);
    for (int c = 0; c < 16; c++) if (qx_nmag[t][c & 7] || !(c & 7)) code_of[(uint8_t)qx_nlut[t][c]] = (int8_t)c;
    for (long bk = 0; bk < nblk; bk++) {
        const uint8_t* b = src + bk * bytes;
        uint8_t* o = dst + bk * 146;
        memcpy(o, b, 2);
        for (int j = 0; j < 8; j++) {
            int8_t c[32]; int m[2]; float dw;
            qx_codes_ref(ty, b, j, c, m, &dw);
            if (m[0] < 0 || m[0] > 255 || m[1] < 0 || m[1] > 255) return -1;
            o[2 + 2 * j] = (uint8_t)m[0]; o[3 + 2 * j] = (uint8_t)m[1];
            for (int k = 0; k < 16; k++) {
                const int lo = code_of[(uint8_t)c[k]], hi = code_of[(uint8_t)c[k + 16]];
                if (lo < 0 || hi < 0) return -1;
                o[18 + 16 * j + k] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
    return nt;
}

/* the weights as fp32, with the float arithmetic of kindi/qwen4exp_quant.py (the decoder test) */
int qx_dequant(int ty, const uint8_t* src, long n, float* out) {
    qx_init();
    int bytes; const int per = qx_block(ty, &bytes);
    if (!per || n % per) return -1;
    for (long bk = 0; bk < n / per; bk++) {
        const uint8_t* b = src + bk * bytes;
        float* o = out + bk * per;
        const float d = qx_f16(b);
        for (int j = 0; j < per / 32; j++) {
            int8_t c[32]; int m[2]; float dw;
            qx_codes_ref(ty, b, j, c, m, &dw);
            for (int k = 0; k < 32; k++) {
                const int h = k >= 16;
                const float a = (float)abs(c[k]), sg = c[k] < 0 ? -1.f : 1.f;
                float v;
                switch (ty) {
                case QX_IQ2_S: { const float s = (float)((m[h] - 1) / 2); v = d * (0.5f + s) * 0.25f * a * sg; break; }
                case QX_IQ3_XXS: { const float s = (float)((m[0] - 1) / 2); v = d * (0.5f + s) * 0.5f * a * sg; break; }
                case QX_IQ3_S: v = d * (float)m[0] * a * sg; break;
                case QX_IQ4_XS: v = d * (float)m[0] * (float)c[k]; break;
                case QX_IQ4_NL: v = d * (float)c[k]; break;
                default: v = d * (float)c[k]; break;   /* Q2_0: d * (code - 1) */
                }
                o[32 * j + k] = v;
            }
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- activations */
static inline float qx_fp16_round(float x) { return _cvtsh_ss(_cvtss_sh(x, 0)); }
/* blocks of 256 values with one scale (the expert input x of gate/up) */
/* the down input's activation block (item G): 32, or 64 = integer sums per down weight block (Q2_0: 64 values,
   IQ4_NL: 32) and one float step per block with the 64-value activation scale, as gate/up on 256; the card's
   KINDI_QX_DN must match (qx_gpu.cu k_down / k_down_slots) */
int qx_dn = 32;
void qx_quant_ab(const float* x, int n, int8_t* q, float* d, int ab) {
    for (int b = 0; b < n / ab; b++) {
        const float* xb = x + ab * b;
        float amax = 0.f;
        for (int k = 0; k < ab; k++) { const float a = fabsf(xb[k]); if (a > amax) amax = a; }
        d[b] = qx_fp16_round(amax / 127.f);
        const float id = amax != 0.f ? 127.f / amax : 0.f;
        for (int k = 0; k < ab; k++) q[ab * b + k] = (int8_t)rintf(xb[k] * id);
    }
}
void qx_quant256(const float* x, int n, int8_t* q, float* d) {
    for (int b = 0; b < n / 256; b++) {
        const float* xb = x + 256 * b;
        float amax = 0.f;
        for (int k = 0; k < 256; k++) { const float a = fabsf(xb[k]); if (a > amax) amax = a; }
        d[b] = qx_fp16_round(amax / 127.f);
        const float id = amax != 0.f ? 127.f / amax : 0.f;
        for (int k = 0; k < 256; k++) q[256 * b + k] = (int8_t)rintf(xb[k] * id);
    }
}
void qx_quant(const float* x, int n, int8_t* q, float* d) {
    for (int b = 0; b < n / 32; b++) {
        const float* xb = x + 32 * b;
        float amax = 0.f;
        for (int k = 0; k < 32; k++) { const float a = fabsf(xb[k]); if (a > amax) amax = a; }
        d[b] = qx_fp16_round(amax / 127.f);
        const float id = amax != 0.f ? 127.f / amax : 0.f;
        for (int k = 0; k < 32; k++) q[32 * b + k] = (int8_t)rintf(xb[k] * id);
    }
}

/* ---------------------------------------------------------------- layers */
typedef struct {
    int K, F, E;                          /* hidden size, expert width, experts */
    int tgu, tdn;                         /* types of gate/up, of down */
    long rgu, rdn;                        /* row bytes */
    const uint8_t *gate, *up, *down;      /* [E][rows][row bytes] in the mapped file */
    int xs;                               /* xstore layer id, -1: read in place */
    int numa;                             /* 1: every expert's rows split over the two nodes, each half in its node's
                                             memory (gn/un/dn): rows [0, sgu) / [0, sdn) on node 0, the rest on 1 */
    int sgu, sdn;
    uint8_t *gn[2], *un[2], *dn[2];       /* [E][rows on the node][row bytes] */
} qx_layer;
static qx_layer QL[QX_MAXL];
static int nql;

const void* qx_map(const char* path, long* size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st)) { close(fd); return NULL; }
    void* p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *size = st.st_size;
    return p;
}

/* returns the layer handle; gate/up: [E][F][K], down: [E][K][F] */
int qx_nib = 0;                       /* 1: layers added from now on keep their i-quant gate/up in nibble form */
int qx_numa = 0;                      /* 1: layers added from now on get their rows split over the pool's two nodes */
long long qx_numa_bytes = 0;          /* RAM taken by the node copies */
static struct { const uint8_t* src[3]; uint8_t* dst[3][2]; long rows[3], split[3], rb[3]; int E;
                _Atomic long next; } QN;
/* the copies' pages as 2 MB pages where the kernel can: MADV_HUGEPAGE before the first touch asks for them, but a
   fragmented host (a large page cache, THP "madvise") hands out 4 KB pages anyway, so after the copy
   MADV_COLLAPSE (Linux 6.1+) merges them synchronously; best effort, the bytes do not change */
#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif
int qx_collapse = 1;
static void qx_huge(void* p, size_t n) {
    if (!qx_collapse || !p || !n) return;
    const uintptr_t a = ((uintptr_t)p + ((1u << 21) - 1)) & ~(uintptr_t)((1u << 21) - 1);
    const uintptr_t e = ((uintptr_t)p + n) & ~(uintptr_t)((1u << 21) - 1);
    if (e > a) madvise((void*)a, e - a, MADV_COLLAPSE);
}
/* each node's copy is written by that node's threads, so its pages are local by first touch too (mbind can be
   refused, e.g. in a container: "mbind: Operation not permitted", and the pages would then land on the writer's
   node) */
static _Atomic long QN_next[2];
static void qx_numa_job(int t, int nact, void* arg) {
    (void)nact; (void)arg;
    const int nd = moe_pool_node_of(t);
    long a;
    while ((a = atomic_fetch_add_explicit(&QN_next[nd], 1, memory_order_relaxed)) < 3L * QN.E) {
        const int m = (int)(a % 3); const long e = a / 3;
        const uint8_t* src = QN.src[m] + (size_t)e * QN.rows[m] * QN.rb[m];
        const size_t b0 = (size_t)QN.split[m] * QN.rb[m], b1 = (size_t)(QN.rows[m] - QN.split[m]) * QN.rb[m];
        if (nd == 0) memcpy(QN.dst[m][0] + (size_t)e * b0, src, b0);
        else memcpy(QN.dst[m][1] + (size_t)e * b1, src + b0, b1);
    }
}
static int qx_numa_split(qx_layer* L) {
    const float s0 = moe_pool_share0();
    L->sgu = (int)lroundf(L->F * s0); L->sdn = (int)lroundf(L->K * s0);
    if (L->sgu <= 0 || L->sgu >= L->F || L->sdn <= 0 || L->sdn >= L->K) return -1;
    const uint8_t* src[3] = {L->gate, L->up, L->down};
    const long rows[3] = {L->F, L->F, L->K}, split[3] = {L->sgu, L->sgu, L->sdn}, rb[3] = {L->rgu, L->rgu, L->rdn};
    uint8_t* d[3][2] = {{0}};
    for (int m = 0; m < 3; m++)
        for (int nd = 0; nd < 2; nd++) {
            const size_t n = (size_t)L->E * (nd ? rows[m] - split[m] : split[m]) * rb[m];
            d[m][nd] = moe_node_alloc(n, nd);
            if (!d[m][nd]) return -1;
            qx_numa_bytes += (long long)n;
        }
    for (int m = 0; m < 3; m++) {
        QN.src[m] = src[m]; QN.rows[m] = rows[m]; QN.split[m] = split[m]; QN.rb[m] = rb[m];
        QN.dst[m][0] = d[m][0]; QN.dst[m][1] = d[m][1];
    }
    QN.E = L->E; atomic_store(&QN.next, 0); atomic_store(&QN_next[0], 0); atomic_store(&QN_next[1], 0);
    moe_pool_run(qx_numa_job, NULL);
    for (int m = 0; m < 3; m++)
        for (int nd = 0; nd < 2; nd++) qx_huge(d[m][nd], (size_t)L->E * (nd ? rows[m] - split[m] : split[m]) * rb[m]);
    L->gn[0] = d[0][0]; L->gn[1] = d[0][1]; L->un[0] = d[1][0]; L->un[1] = d[1][1]; L->dn[0] = d[2][0]; L->dn[1] = d[2][1];
    L->numa = 1;
    return 0;
}
long long qx_nib_bytes = 0;           /* RAM taken by the nibble copies */
/* qx_dncopy = 1: layers added from now on keep their down matrices in anonymous memory (huge pages where the kernel
   allows them) instead of the file's page cache, as the nibble form does for gate/up: the same bytes, so the same
   bits; a rented EPYC test machine read the mapped down rows at a fifth of the speed of the copies */
int qx_dncopy = 0;
long long qx_dncopy_bytes = 0;
static struct { const uint8_t* src; uint8_t* dst; size_t n; _Atomic long next; } QD;
static void qx_dc_job(int t, int nact, void* arg) {
    (void)t; (void)nact; (void)arg;
    const size_t ch = (size_t)1 << 21;
    long a;
    while ((a = atomic_fetch_add_explicit(&QD.next, 1, memory_order_relaxed)) * ch < QD.n) {
        const size_t o = (size_t)a * ch, n = QD.n - o < ch ? QD.n - o : ch;
        memcpy(QD.dst + o, QD.src + o, n);
    }
}
static uint8_t* qx_anon(size_t n) {
    const size_t al = (size_t)1 << 21, m = (n + al - 1) / al * al;
    uint8_t* p = aligned_alloc(al, m);
    if (p) madvise(p, m, MADV_HUGEPAGE);
    return p;
}
static const uint8_t* qx_copy_anon(const uint8_t* src, size_t n) {
    uint8_t* d = qx_anon(n);
    if (!d) return src;
    QD.src = src; QD.dst = d; QD.n = n; atomic_store(&QD.next, 0);
    moe_pool_run(qx_dc_job, NULL);
    qx_huge(d, n);
    qx_dncopy_bytes += (long long)n;
    return d;
}
static struct { const uint8_t* src[2]; uint8_t* dst[2]; int ty; long blk_per_e; int bytes; _Atomic long next; long n;
                _Atomic int bad; } QT;
static void qx_tc_job(int t, int nact, void* arg) {
    (void)t; (void)nact; (void)arg;
    long a;
    while ((a = atomic_fetch_add_explicit(&QT.next, 1, memory_order_relaxed)) < QT.n) {
        const int w = (int)(a & 1); const long e = a >> 1;
        if (qx_transcode(QT.ty, QT.src[w] + e * QT.blk_per_e * QT.bytes, QT.blk_per_e, QT.dst[w] + e * QT.blk_per_e * 146) < 0)
            atomic_store(&QT.bad, 1);
    }
}
int qx_add_layer(int K, int F, int E, int tgu, int tdn, const void* gate, const void* up, const void* down) {
    qx_init();
    if (nql == QX_MAXL || qx_row_bytes(tgu, K) < 0 || qx_row_bytes(tdn, F) < 0) return -1;
    qx_layer* L = &QL[nql];
    L->K = K; L->F = F; L->E = E; L->tgu = tgu; L->tdn = tdn;
    L->rgu = qx_row_bytes(tgu, K); L->rdn = qx_row_bytes(tdn, F);
    L->gate = gate; L->up = up; L->down = down; L->xs = -1;
    if (qx_nib && (tgu == QX_IQ3_XXS || tgu == QX_IQ3_S || tgu == QX_IQ2_S) && K % 256 == 0 && moe_pool_threads() > 0) {
        const long blk_per_e = (long)F * (K / 256);
        const size_t n = (size_t)E * blk_per_e * 146;
        uint8_t* g = qx_anon(n), *u = qx_anon(n);
        if (g && u) {
            int bytes; qx_block(tgu, &bytes);
            QT.src[0] = gate; QT.src[1] = up; QT.dst[0] = g; QT.dst[1] = u; QT.ty = tgu; QT.blk_per_e = blk_per_e;
            QT.bytes = bytes; QT.n = 2L * E; atomic_store(&QT.next, 0); atomic_store(&QT.bad, 0);
            moe_pool_run(qx_tc_job, NULL);
            if (!atomic_load(&QT.bad)) {
                if (!(qx_numa && moe_pool_nthr_node(1) > 0)) { qx_huge(g, n); qx_huge(u, n); }
                L->tgu = tgu + QX_NIB; L->rgu = qx_row_bytes(L->tgu, K); L->gate = g; L->up = u;
                if (qx_numa && moe_pool_nthr_node(1) > 0 && qx_numa_split(L) == 0) {
                    free(g); free(u);                 /* the node copies hold the nibble forms now */
                    L->gate = L->up = NULL;
                } else {
                    qx_nib_bytes += 2 * (long long)n;
                    if (qx_dncopy) L->down = qx_copy_anon(L->down, (size_t)E * K * L->rdn);
                }
                return nql++;
            }
        }
        free(g); free(u);
    }
    if (qx_numa && moe_pool_nthr_node(1) > 0 && qx_numa_split(L) == 0) return nql++;
    if (qx_dncopy) L->down = qx_copy_anon(L->down, (size_t)E * K * L->rdn);
    return nql++;
}
void qx_layer_set_xs(int l, int xs) { if (l >= 0 && l < nql) QL[l].xs = xs; }
void qx_layer_bytes(int l, long* gu_matrix, long* dn_matrix) { *gu_matrix = QL[l].rgu * QL[l].F; *dn_matrix = QL[l].rdn * QL[l].K; }

/* ---------------------------------------------------------------- one call */
static struct {
    const qx_layer* L;
    int T, nsel, nu;
    const int8_t* xq; const float* xd;    /* [T][K] */
    const uint8_t* mg[1024]; const uint8_t* mu_[1024]; const uint8_t* md[1024];   /* per union expert */
    int* pstart; int* ptok;               /* pairs of union expert u: [pstart[u], pstart[u + 1]) */
    float* h; int8_t* hq; float* hd;      /* [pairs][F] */
    float* val;                           /* [pairs][K] down values */
    int* tpair; const float* w; float* out;
    float* parts;                         /* not NULL: [T][nsel][K] value rows of the computed slots, no combine */
    int gr, dr;                           /* rows per work item */
    _Atomic long next[4];
    long items[4];
    /* steps 1 and 3 by node: node nd's rows [lo, hi) of every union expert, at mgn/mun/mdn[nd][u] (row lo first);
       a thread takes the items of its own node first, then helps the other node */
    const uint8_t* mgn[2][1024]; const uint8_t* mun[2][1024]; const uint8_t* mdn[2][1024];
    int sgu, sdn;
    _Atomic long next1[2], next3[2];
    long items1[2], items3[2];
} C;
static struct { int8_t* xq; float* xd; int* tpair; int* pstart; int* ptok; int* cnt; int* umap; int* uexp;
                float *h, *hd, *val; int8_t* hq; size_t tcap, pcap, ecap; } B;
static struct { long long calls, experts, pairs, bytes; double sec; } QST;
/* the last call's phases per layer handle (KINDI_QXG_TIMELINE; CLOCK_MONOTONIC ns, thread 0's view): enter, pool
   start, step 1 done, step 2 done, step 3 done, step 4 done, all threads done; then union experts and pairs */
#define QX_PH_MAX 64
static long long qx_ph[QX_PH_MAX][9];
static __thread int qx_ph_l = -1;
static inline long long qx_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
#define QX_PH(i) do { if (t == 0 && qx_ph_l >= 0) qx_ph[qx_ph_l][i] = qx_ns(); } while (0)
void qx_phases(long long* out, int n) { memcpy(out, qx_ph, sizeof(long long) * 9 * (n < QX_PH_MAX ? n : QX_PH_MAX)); }

/* exp with IEEE float multiplies and adds only (no FMA, no libm): the card kernels (qx_gpu.cu) use the same
   operations in the same order, so h = silu(gate) * up has the same bits on the CPU and on a card. Relative error
   about 2e-7 against expf. */
float qx_expf(float x) {
    if (x > 88.f) x = 88.f;
    if (x < -88.f) x = -88.f;
    const float n = rintf(x * 1.44269504f);
    float r = x - n * 0.693145752f;              /* ln 2, split: hi has 8 trailing zero bits, so n * hi is exact */
    r = r - n * 1.42860677e-06f;
    float p = 1.98412698e-4f;                   /* 1/7! .. 1/2! by Horner, then 1 + r * p, then 1 + r * p */
    p = p * r + 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 0.5f;
    p = p * r + 1.f;
    p = p * r + 1.f;
    return ldexpf(p, (int)n);
}
static inline float qx_silu(float g) { return g / (1.f + qx_expf(-g)); }

static void qx_job(int t, int nact, void* arg) {
    (void)nact; (void)arg;
    const qx_layer* L = C.L;
    long a;
    const int mynd = moe_pool_node_of(t);
    /* step 1: gate/up rows, the own node's first */
    for (int pass = 0; pass < 2; pass++) {
    const int nd = pass ? 1 - mynd : mynd, lo = nd ? C.sgu : 0, hi = nd ? L->F : C.sgu;
    if (hi <= lo) continue;
    const int per_e = (hi - lo + C.gr - 1) / C.gr;
    while ((a = atomic_fetch_add_explicit(&C.next1[nd], 1, memory_order_relaxed)) < C.items1[nd]) {
        const int u = (int)(a / per_e), r0 = lo + (int)(a % per_e) * C.gr;
        const int r1 = r0 + C.gr < hi ? r0 + C.gr : hi;
        if (C.pstart[u + 1] - C.pstart[u] >= qx_predec && L->K <= 32 * QX_MAXSB) {
            static __thread qx_drow DG, DU;      /* many pairs: decode each row once, then all its pairs */
            for (int r = r0; r < r1; r++) {
                qx_row_decode(L->tgu, C.mgn[nd][u] + (size_t)(r - lo) * L->rgu, L->K, &DG);
                qx_row_decode(L->tgu, C.mun[nd][u] + (size_t)(r - lo) * L->rgu, L->K, &DU);
                for (int p0 = C.pstart[u]; p0 < C.pstart[u + 1]; p0 += QX_PB) {
                    const int np = C.pstart[u + 1] - p0 < QX_PB ? C.pstart[u + 1] - p0 : QX_PB;
                    const int8_t* q[QX_PB]; const float* d[QX_PB];
                    for (int i = 0; i < np; i++) { const int tk = C.ptok[p0 + i]; q[i] = C.xq + (size_t)tk * L->K; d[i] = C.xd + (size_t)tk * (L->K / 256); }
                    float g[QX_PB], v[QX_PB];
                    qx_row_dot256_d(L->tgu, &DG, L->K, np, q, d, g, 256);
                    qx_row_dot256_d(L->tgu, &DU, L->K, np, q, d, v, 256);
                    for (int i = 0; i < np; i++) C.h[(size_t)(p0 + i) * L->F + r] = qx_silu(g[i]) * v[i];
                }
            }
            continue;
        }
        if (qx_one && C.pstart[u + 1] - C.pstart[u] <= QX_NPMAX) {   /* few pairs: the constant-np kernels */
            const int p0 = C.pstart[u], np = C.pstart[u + 1] - p0;
            const int8_t* q[QX_NPMAX]; const float* d[QX_NPMAX];
            for (int i = 0; i < np; i++) { const int tk = C.ptok[p0 + i]; q[i] = C.xq + (size_t)tk * L->K; d[i] = C.xd + (size_t)tk * (L->K / 256); }
            int done = 1;
            for (int ra = r0; ra < r1 && done; ra += 64) {
                const int nr = r1 - ra < 64 ? r1 - ra : 64;
                float g[64 * QX_NPMAX], v[64 * QX_NPMAX];
                done = qx_rows256_k(L->tgu, np, C.mgn[nd][u] + (size_t)(ra - lo) * L->rgu, L->rgu, nr, L->K, q, d, g, np, 256) &&
                       qx_rows256_k(L->tgu, np, C.mun[nd][u] + (size_t)(ra - lo) * L->rgu, L->rgu, nr, L->K, q, d, v, np, 256);
                for (int r = 0; r < nr && done; r++)
                    for (int i = 0; i < np; i++) C.h[(size_t)(p0 + i) * L->F + ra + r] = qx_silu(g[r * np + i]) * v[r * np + i];
            }
            if (done) continue;
        }
        for (int p0 = C.pstart[u]; p0 < C.pstart[u + 1]; p0 += QX_PB) {
            const int np = C.pstart[u + 1] - p0 < QX_PB ? C.pstart[u + 1] - p0 : QX_PB;
            const int8_t* q[QX_PB]; const float* d[QX_PB];
            for (int i = 0; i < np; i++) { const int tk = C.ptok[p0 + i]; q[i] = C.xq + (size_t)tk * L->K; d[i] = C.xd + (size_t)tk * (L->K / 256); }
            for (int r = r0; r < r1; r++) {
                float g[QX_PB], v[QX_PB];
                qx_row_dot256_n(L->tgu, C.mgn[nd][u] + (size_t)(r - lo) * L->rgu, L->K, np, q, d, g, 256);
                qx_row_dot256_n(L->tgu, C.mun[nd][u] + (size_t)(r - lo) * L->rgu, L->K, np, q, d, v, 256);
                for (int i = 0; i < np; i++) C.h[(size_t)(p0 + i) * L->F + r] = qx_silu(g[i]) * v[i];
            }
        }
    }
    }
    moe_pool_barrier();
    QX_PH(2);
    /* step 2: quantize h */
    while ((a = atomic_fetch_add_explicit(&C.next[1], 1, memory_order_relaxed)) < C.items[1])
        if (qx_dn == 64) qx_quant_ab(C.h + (size_t)a * L->F, L->F, C.hq + (size_t)a * L->F, C.hd + (size_t)a * (L->F / 32), 64);
        else qx_quant(C.h + (size_t)a * L->F, L->F, C.hq + (size_t)a * L->F, C.hd + (size_t)a * (L->F / 32));
    moe_pool_barrier();
    QX_PH(3);
    /* step 3: down rows, the own node's first */
    for (int pass = 0; pass < 2; pass++) {
    const int nd = pass ? 1 - mynd : mynd, lo = nd ? C.sdn : 0, hi = nd ? L->K : C.sdn;
    if (hi <= lo) continue;
    const int per_e = (hi - lo + C.dr - 1) / C.dr;
    while ((a = atomic_fetch_add_explicit(&C.next3[nd], 1, memory_order_relaxed)) < C.items3[nd]) {
        const int u = (int)(a / per_e), r0 = lo + (int)(a % per_e) * C.dr;
        const int r1 = r0 + C.dr < hi ? r0 + C.dr : hi;
        if (C.pstart[u + 1] - C.pstart[u] >= qx_predec && L->F <= 32 * QX_MAXSB) {
            static __thread qx_drow DD;
            for (int r = r0; r < r1; r++) {
                qx_row_decode(L->tdn, C.mdn[nd][u] + (size_t)(r - lo) * L->rdn, L->F, &DD);
                if (qx_dn == 64) {
                    for (int p0 = C.pstart[u]; p0 < C.pstart[u + 1]; p0 += QX_PB) {
                        const int np = C.pstart[u + 1] - p0 < QX_PB ? C.pstart[u + 1] - p0 : QX_PB;
                        const int8_t* q[QX_PB]; const float* d[QX_PB];
                        for (int i = 0; i < np; i++) { q[i] = C.hq + (size_t)(p0 + i) * L->F; d[i] = C.hd + (size_t)(p0 + i) * (L->F / 32); }
                        float v[QX_PB];
                        qx_row_dot256_d(L->tdn, &DD, L->F, np, q, d, v, 64);
                        for (int i = 0; i < np; i++) C.val[(size_t)(p0 + i) * L->K + r] = v[i];
                    }
                    continue;
                }
                for (int p0 = C.pstart[u]; p0 < C.pstart[u + 1]; p0 += QX_PB) {
                    const int np = C.pstart[u + 1] - p0 < QX_PB ? C.pstart[u + 1] - p0 : QX_PB;
                    const int8_t* q[QX_PB]; const float* d[QX_PB];
                    for (int i = 0; i < np; i++) { q[i] = C.hq + (size_t)(p0 + i) * L->F; d[i] = C.hd + (size_t)(p0 + i) * (L->F / 32); }
                    float v[QX_PB];
                    qx_row_dot_d(&DD, L->F, np, q, d, v);
                    for (int i = 0; i < np; i++) C.val[(size_t)(p0 + i) * L->K + r] = v[i];
                }
            }
            continue;
        }
        if (qx_one && C.pstart[u + 1] - C.pstart[u] <= QX_NPMAX) {
            const int p0 = C.pstart[u], np = C.pstart[u + 1] - p0;
            const int8_t* q[QX_NPMAX]; const float* d[QX_NPMAX];
            for (int i = 0; i < np; i++) { q[i] = C.hq + (size_t)(p0 + i) * L->F; d[i] = C.hd + (size_t)(p0 + i) * (L->F / 32); }
            int done = 1;
            for (int ra = r0; ra < r1 && done; ra += 64) {
                const int nr = r1 - ra < 64 ? r1 - ra : 64;
                float v[64 * QX_NPMAX];
                done = qx_dn == 64 ? qx_rows256_k(L->tdn, np, C.mdn[nd][u] + (size_t)(ra - lo) * L->rdn, L->rdn, nr, L->F, q, d, v, np, 64)
                                   : qx_rows32_k(L->tdn, np, C.mdn[nd][u] + (size_t)(ra - lo) * L->rdn, L->rdn, nr, L->F, q, d, v, np);
                for (int r = 0; r < nr && done; r++)
                    for (int i = 0; i < np; i++) C.val[(size_t)(p0 + i) * L->K + ra + r] = v[r * np + i];
            }
            if (done) continue;
        }
        for (int p0 = C.pstart[u]; p0 < C.pstart[u + 1]; p0 += QX_PB) {
            const int np = C.pstart[u + 1] - p0 < QX_PB ? C.pstart[u + 1] - p0 : QX_PB;
            const int8_t* q[QX_PB]; const float* d[QX_PB];
            for (int i = 0; i < np; i++) { q[i] = C.hq + (size_t)(p0 + i) * L->F; d[i] = C.hd + (size_t)(p0 + i) * (L->F / 32); }
            for (int r = r0; r < r1; r++) {
                float v[QX_PB];
                if (qx_dn == 64) qx_row_dot256_n(L->tdn, C.mdn[nd][u] + (size_t)(r - lo) * L->rdn, L->F, np, q, d, v, 64);
                else qx_row_dot_n(L->tdn, C.mdn[nd][u] + (size_t)(r - lo) * L->rdn, L->F, np, q, d, v);
                for (int i = 0; i < np; i++) C.val[(size_t)(p0 + i) * L->K + r] = v[i];
            }
        }
    }
    }
    moe_pool_barrier();
    QX_PH(4);
    /* step 4: combine, by token and 256-value pieces (parts mode: the value rows of the computed slots instead) */
    while ((a = atomic_fetch_add_explicit(&C.next[3], 1, memory_order_relaxed)) < C.items[3]) {
        const int pieces = L->K / 256, tk = (int)(a / pieces), c0 = (int)(a % pieces) * 256;
        if (C.parts) {
            for (int s = 0; s < C.nsel; s++) {
                const int pr = C.tpair[tk * C.nsel + s];
                if (pr >= 0) memcpy(C.parts + ((size_t)tk * C.nsel + s) * L->K + c0, C.val + (size_t)pr * L->K + c0, 256 * 4);
            }
            continue;
        }
        float* o = C.out + (size_t)tk * L->K;
        for (int c = c0; c < c0 + 256; c++) {
            float acc = 0.f;
            for (int s = 0; s < C.nsel; s++) {
                const float v = C.w[tk * C.nsel + s] * C.val[(size_t)C.tpair[tk * C.nsel + s] * L->K + c];
                acc = acc + v;
            }
            o[c] = acc;
        }
    }
    QX_PH(5);
}

#define QX_GROW(ptr, n, type) do { free(ptr); ptr = (type*)aligned_alloc(64, ((size_t)(n) * sizeof(type) + 63) / 64 * 64); } while (0)

static int qx_run(int l, int T, const float* x, int nsel, const int* ids, const uint8_t* skip, const float* w,
                  float* out, float* parts);
/* out [T][K] = the routed experts of layer l for x [T][K]; ids, w [T][nsel] (the router's picks and weights) */
int qx_forward(int l, int T, const float* x, int nsel, const int* ids, const float* w, float* out) {
    return qx_run(l, T, x, nsel, ids, NULL, w, out, NULL);
}
/* the value rows parts [T][nsel][K] of every (token, slot) not marked in skip [T][nsel] (the others: computed
   elsewhere, a card's hot experts); the caller combines: out = sum over slots in order of w * value */
int qx_forward_parts(int l, int T, const float* x, int nsel, const int* ids, const uint8_t* skip, float* parts) {
    return qx_run(l, T, x, nsel, ids, skip, NULL, NULL, parts);
}
static int qx_run(int l, int T, const float* x, int nsel, const int* ids, const uint8_t* skip, const float* w,
                  float* out, float* parts) {
    if (l < 0 || l >= nql || T < 1 || nsel < 1 || nsel > QX_MAXSEL || moe_pool_threads() < 1) return -1;
    struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);
    qx_ph_l = l < QX_PH_MAX ? l : -1;
    if (qx_ph_l >= 0) { memset(qx_ph[l], 0, sizeof qx_ph[l]); qx_ph[l][0] = ts0.tv_sec * 1000000000LL + ts0.tv_nsec; }
    const qx_layer* L = &QL[l];
    const size_t np = (size_t)T * nsel;
    if ((size_t)T > B.tcap) { QX_GROW(B.xq, (size_t)T * L->K, int8_t); QX_GROW(B.xd, (size_t)T * L->K / 32, float); QX_GROW(B.tpair, np, int); B.tcap = T; }
    if (np > B.pcap) {
        QX_GROW(B.ptok, np, int); QX_GROW(B.h, np * 4096, float); QX_GROW(B.hq, np * 4096, int8_t);
        QX_GROW(B.hd, np * 4096 / 32, float); QX_GROW(B.val, np * L->K, float); B.pcap = np;
        if ((size_t)T > B.tcap) B.tcap = T;
        free(B.tpair); B.tpair = malloc(sizeof(int) * np);
    }
    if ((size_t)L->E > B.ecap) {
        free(B.pstart); free(B.cnt); free(B.umap); free(B.uexp);
        B.pstart = malloc(sizeof(int) * (L->E + 1)); B.cnt = malloc(sizeof(int) * L->E);
        B.umap = malloc(sizeof(int) * L->E); B.uexp = malloc(sizeof(int) * L->E); B.ecap = L->E;
    }
    if (L->F > 4096 || L->E > 1024) return -3;
    if (L->K % 256) return -4;
    for (int t = 0; t < T; t++) qx_quant256(x + (size_t)t * L->K, L->K, B.xq + (size_t)t * L->K, B.xd + (size_t)t * (L->K / 256));
    int nu = 0;
    for (size_t i = 0; i < np; i++) if (ids[i] < 0 || ids[i] >= L->E) return -2;
    for (int e = 0; e < L->E; e++) B.umap[e] = -1;
    for (size_t i = 0; i < np; i++) {
        if (skip && skip[i]) continue;
        const int e = ids[i];
        if (B.umap[e] < 0) { B.umap[e] = nu; B.uexp[nu] = e; B.cnt[nu] = 0; nu++; }
        B.cnt[B.umap[e]]++;
    }
    if (nu == 0) return 0;
    B.pstart[0] = 0;
    for (int u = 0; u < nu; u++) { B.pstart[u + 1] = B.pstart[u] + B.cnt[u]; B.cnt[u] = B.pstart[u]; }
    for (int t = 0; t < T; t++)
        for (int s = 0; s < nsel; s++) {
            if (skip && skip[t * nsel + s]) { B.tpair[t * nsel + s] = -1; continue; }
            const int u = B.umap[ids[t * nsel + s]], p = B.cnt[u]++;
            B.ptok[p] = t; B.tpair[t * nsel + s] = p;
        }
    const size_t gm = (size_t)L->F * L->rgu, dm = (size_t)L->K * L->rdn;
    if (L->xs >= 0) {
        static const uint8_t** ptr; static int pc;
        if (nu > pc) { free(ptr); ptr = malloc(sizeof(*ptr) * 3 * 1024); pc = 1024; }
        xs_acquire(L->xs, nu, B.uexp, ptr);
        for (int u = 0; u < nu; u++) { C.mg[u] = ptr[3 * u]; C.mu_[u] = ptr[3 * u + 1]; C.md[u] = ptr[3 * u + 2]; }
    } else
        for (int u = 0; u < nu; u++) {
            const size_t e = (size_t)B.uexp[u];
            C.mg[u] = L->gate + e * gm; C.mu_[u] = L->up + e * gm; C.md[u] = L->down + e * dm;
        }
    const int nthr = moe_pool_threads();
    C.L = L; C.T = T; C.nsel = nsel; C.nu = nu; C.xq = B.xq; C.xd = B.xd;
    C.pstart = B.pstart; C.ptok = B.ptok; C.h = B.h; C.hq = B.hq; C.hd = B.hd; C.val = B.val;
    C.tpair = B.tpair; C.w = w; C.out = out; C.parts = parts;
    /* work items: enough to keep every thread busy, small enough to balance (rows of 640-wide gate/up, 2560 down) */
    C.gr = 16; C.dr = 64;
    while (C.gr > 4 && (long)nu * ((L->F + C.gr - 1) / C.gr) < 8L * nthr) C.gr /= 2;
    while (C.dr > 8 && (long)nu * ((L->K + C.dr - 1) / C.dr) < 8L * nthr) C.dr /= 2;
    C.items[0] = (long)nu * ((L->F + C.gr - 1) / C.gr);
    C.items[1] = (long)B.pstart[nu];
    C.items[2] = (long)nu * ((L->K + C.dr - 1) / C.dr);
    C.items[3] = (long)T * (L->K / 256);
    for (int i = 0; i < 4; i++) atomic_store(&C.next[i], 0);
    if (L->numa && L->xs < 0) {
        C.sgu = L->sgu; C.sdn = L->sdn;
        for (int u = 0; u < nu; u++) {
            const size_t e = (size_t)B.uexp[u];
            for (int nd = 0; nd < 2; nd++) {
                const size_t rg = nd ? (size_t)(L->F - L->sgu) : (size_t)L->sgu;
                const size_t rd = nd ? (size_t)(L->K - L->sdn) : (size_t)L->sdn;
                C.mgn[nd][u] = L->gn[nd] + e * rg * L->rgu; C.mun[nd][u] = L->un[nd] + e * rg * L->rgu;
                C.mdn[nd][u] = L->dn[nd] + e * rd * L->rdn;
            }
        }
    } else {
        C.sgu = L->F; C.sdn = L->K;
        for (int u = 0; u < nu; u++) { C.mgn[0][u] = C.mg[u]; C.mun[0][u] = C.mu_[u]; C.mdn[0][u] = C.md[u]; }
    }
    for (int nd = 0; nd < 2; nd++) {
        const int rg = nd ? L->F - C.sgu : C.sgu, rd = nd ? L->K - C.sdn : C.sdn;
        C.items1[nd] = rg > 0 ? (long)nu * ((rg + C.gr - 1) / C.gr) : 0;
        C.items3[nd] = rd > 0 ? (long)nu * ((rd + C.dr - 1) / C.dr) : 0;
        atomic_store(&C.next1[nd], 0); atomic_store(&C.next3[nd], 0);
    }
    if (qx_ph_l >= 0) { qx_ph[l][1] = qx_ns(); qx_ph[l][7] = nu; qx_ph[l][8] = B.pstart[nu]; }
    moe_pool_run(qx_job, NULL);
    if (qx_ph_l >= 0) qx_ph[l][6] = qx_ns();
    if (L->xs >= 0) xs_release(L->xs, nu, B.uexp);
    struct timespec ts1; clock_gettime(CLOCK_MONOTONIC, &ts1);
    QST.calls++; QST.experts += nu; QST.pairs += (long long)B.pstart[nu]; QST.bytes += (long long)nu * (2 * gm + dm);
    QST.sec += (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    return nu;
}

/* out: calls, union experts, pairs, expert bytes read; sec: seconds inside qx_forward */
void qx_stats(long long* out, double* sec) { out[0] = QST.calls; out[1] = QST.experts; out[2] = QST.pairs; out[3] = QST.bytes; *sec = QST.sec; }
void qx_stats_reset(void) { memset(&QST, 0, sizeof QST); }

/* ---------------------------------------------------------------- the mailbox step (moeaccel.h ma_fn) */
/* ctx: the layer handles (nl) and, per layer, which experts are on a card (oncard [nl][E], 1 = the card computes
   it); the CPU computes the other picks of the pass into the mailbox's rows ([T][nsel][K]) */
typedef struct { int nl, E, nsel; const int* handle; const unsigned char* oncard; uint8_t* skip;
                 void* const* note; const int* nf; int smax; long long st[4]; int tail, K; long long dropped; } qx_ma_ctx;
void ma_cache_note(void* c, int layer, const int* ids, int n);
/* note[l]: the F9 cache (ma_cache*) that counts layer l's picks, or NULL; set once the cache runs */
void qx_ma_ctx_set_note(void* ctx, void* const* note) { ((qx_ma_ctx*)ctx)->note = note; }
/* the PCIe share (item 7): nf[l] missed experts of layer l (at most smax) go to the card; the rule is k_fetch_plan's
   in qx_gpu.cu (the picks in order; a missed expert not yet taken is taken while fewer than the share are) */
void qx_ma_ctx_set_fetch(void* ctx, const int* nf, int smax) {
    qx_ma_ctx* c = ctx;
    c->nf = nf; c->smax = smax < 64 ? smax : 64;
}
/* fast mode (KINDI_FAST_TAIL, lossy, off by default): a pick at router rank >= tail (0: off) that is neither on the
   card nor fetched is dropped: not computed, its value row zero (no renormalisation of the router weights). Rows of
   K floats. */
void qx_ma_ctx_set_tail(void* ctx, int tail, int K) {
    qx_ma_ctx* c = ctx;
    c->tail = tail; c->K = K;
}
long long qx_ma_ctx_dropped(void* ctx) {
    qx_ma_ctx* c = ctx;
    const long long d = c->dropped;
    c->dropped = 0;
    return d;
}
/* out: picks, picks on the card's cache, picks of fetched experts, fetched experts (since the last call) */
void qx_ma_ctx_stats(void* ctx, long long* out) {
    qx_ma_ctx* c = ctx;
    memcpy(out, c->st, sizeof c->st);
    memset(c->st, 0, sizeof c->st);
}
void* qx_ma_ctx_new(int nl, int E, int nsel, const int* handle, const unsigned char* oncard) {
    qx_ma_ctx* c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->nl = nl; c->E = E; c->nsel = nsel; c->handle = handle; c->oncard = oncard;
    c->skip = malloc(64 * (size_t)nsel);
    return c;
}
int fc_has_layer(int l);
int fc_fits(int T, int nsel);
int fc_forward_parts(int l, int T, const float* x, int nsel, const int* ids, const uint8_t* skip, float* rows);
int qx_ma_fn(void* ctx, int l, int T, const int* ids, const void* in, void* rows) {
    qx_ma_ctx* c = ctx;
    if (l < 0 || l >= c->nl || T > 64) return -1;
    const unsigned char* oc = c->oncard + (size_t)l * c->E;
    if (c->note && c->note[l]) ma_cache_note(c->note[l], l, ids, T * c->nsel);
    int want = c->nf ? c->nf[l] : 0;
    if (want > c->smax) want = c->smax;
    int taken[64], nt = 0, any = 0, hits = 0, fp = 0;
    for (int i = 0; i < T * c->nsel; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->E) return -1;
        int s = oc[e];
        if (s) hits++;
        else {
            for (int j = 0; j < nt && !s; j++) s = taken[j] == e;
            if (!s && nt < want) { taken[nt++] = e; s = 1; }
            fp += s;
            if (!s && c->tail > 0 && i % c->nsel >= c->tail) {         /* fast mode: dropped, row zero */
                memset((float*)rows + (size_t)i * c->K, 0, (size_t)c->K * sizeof(float));
                c->dropped++;
                s = 1;
            }
        }
        c->skip[i] = (uint8_t)s;
        any |= !s;
    }
    c->st[0] += T * c->nsel; c->st[1] += hits; c->st[2] += fp; c->st[3] += nt;
    if (!any) return 0;
    if (fc_has_layer(l) && fc_fits(T, c->nsel))   /* KINDI_FN_FCPU=1: fncpu.c's repacked experts (0.8 C1) */
        return fc_forward_parts(l, T, (const float*)in, c->nsel, ids, c->skip, (float*)rows) < 0 ? -1 : 0;
    return qx_forward_parts(c->handle[l], T, (const float*)in, c->nsel, ids, c->skip, (float*)rows) < 0 ? -1 : 0;
}
void* qx_ma_fn_ptr(void) { return (void*)qx_ma_fn; }

/* ---------------------------------------------------------------- the pool's streaming-read ceiling (a test)
   Each pool thread reads its share of a buffer on its own node (moe_node_alloc: huge pages, node-local), 32-byte
   loads summed, the way the expert rows are read (the same threads, pinning and nodes); returns GB/s over reps
   passes of mb MB per node. */
static struct { uint8_t* buf[2]; size_t n; int reps; _Atomic long sink; } QB;
static int QB_fill;
static void qx_bw_job(int t, int nact, void* arg) {
    (void)nact; (void)arg;
    const int nd = moe_pool_node_of(t), nn = moe_pool_nthr_node(nd) > 0 ? moe_pool_nthr_node(nd) : 1;
    int k = 0;                                    /* this thread's rank among its node's threads */
    for (int i = 0; i < t; i++) k += moe_pool_node_of(i) == nd;
    const size_t per = QB.n / nn / 64 * 64, lo = (size_t)k * per;
    if (QB_fill) { memset(QB.buf[nd] + lo, 1 + nd, per); return; }
    __m256i acc = _mm256_setzero_si256();
    for (int r = 0; r < QB.reps; r++) {
        const uint8_t* p = QB.buf[nd] + lo;
        for (size_t o = 0; o < per; o += 128) {
            acc = _mm256_add_epi8(acc, _mm256_load_si256((const __m256i*)(p + o)));
            acc = _mm256_add_epi8(acc, _mm256_load_si256((const __m256i*)(p + o + 32)));
            acc = _mm256_add_epi8(acc, _mm256_load_si256((const __m256i*)(p + o + 64)));
            acc = _mm256_add_epi8(acc, _mm256_load_si256((const __m256i*)(p + o + 96)));
        }
    }
    atomic_fetch_add(&QB.sink, _mm256_extract_epi64(acc, 0));
}
double qx_read_bw(int mb, int reps) {
    const int nodes = moe_pool_nthr_node(1) > 0 ? 2 : 1;
    QB.n = (size_t)mb << 20; QB.reps = reps;
    for (int nd = 0; nd < nodes; nd++) {
        QB.buf[nd] = moe_node_alloc(QB.n, nd);
        if (!QB.buf[nd]) return -1;
    }
    QB_fill = 1; moe_pool_run(qx_bw_job, NULL); QB_fill = 0;   /* first touch by the node's own threads */
    QB.reps = 1; moe_pool_run(qx_bw_job, NULL);          /* warm: faults and TLB */
    QB.reps = reps;
    const long long t0 = qx_ns();
    moe_pool_run(qx_bw_job, NULL);
    const double s = (qx_ns() - t0) * 1e-9;
    for (int nd = 0; nd < nodes; nd++) munmap(QB.buf[nd], (QB.n + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1));
    return (double)nodes * QB.n * reps / s / 1e9;
}
