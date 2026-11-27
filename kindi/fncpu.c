/* fncpu.c - the Flash-Next (GGUF qwen4exp) weights on the CPU without a card: Kindi's own repacked formats and AVX2
   kernels (0.8 item C1). Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.

   Every weight matrix (dense or expert) is rewritten once at load into one of two Kindi CPU forms:
     K4  4-bit codes into a 16-entry int8 table per tensor; rows in pairs, byte j of a 32-value block holds value j of
         the pair's first row (low nibble) and of its second row (high nibble). IQ4_NL, IQ4_XS, Q4_K, the i-quants
         (IQ3_XXS, IQ3_S, IQ2_S: their grids hold at most 16 signed values) and Q2_0 (4 of the 16 entries).
     K8  int8 codes, row by row: Q8_0, and the float matrices (BF16/F16/F32) quantized per 32 values.
     K6  6-bit codes, row by row (Q6_K, Q5_K): per 32-value block 16 bytes of low nibbles (value i low, i + 16 high)
         and 8 bytes of top bits (byte j, bits 2k and 2k + 1: value j + 8k); the code is the unsigned 6 bits minus
         off6 (32 for Q6_K). 24 bytes a block against K8's 32: the dense matrices are read every token while writing.
         Opt-in (KINDI_FNCPU_K6=1): its unpacking made the head slower while writing, no net gain measured.
   The integer code of every value is the file's own (IQ4 level, q, q - 32, grid value with its sign, code - 1), so the
   repack loses nothing for the quantized types; the float scales are kept as
     SUP  per row and 256 values a float d (and dmin), per 32 (or 16) values an int8 multiplier (and a uint8 min):
          w = d * mult * c - dmin * min       (IQ4_XS, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ3_S, IQ2_S)
     H32  per row and 32 values the file's fp16 scale: w = d * c (IQ4_NL, Q8_0, Q2_0; the floats re-quantized)
   A product row . x quantizes x per 32 values (int8, scale dx = amax / 127) and sums per 32-value block
   (float)(sum c * q) * (d * mult * dx) - dmin * min * dx * sum q.

   Threads: the pool of moe_cpu.c (moe_pool_run; one thread per physical core, split by NUMA node). Each thread owns a
   fixed slice of every matrix's rows (pairs for K4), writes that slice at load (so its pages are on the thread's
   node by first touch) and computes it in every product. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

#define __device__ static      /* (the grids local to this file: qx_cpu.c has its own copy) */
#include "nbq_grids.h"

typedef void (*moe_pool_fn)(int t, int nact, void* arg);
int moe_pool_run(moe_pool_fn fn, void* arg);
void moe_pool_barrier(void);
int moe_pool_threads(void);
int moe_pool_node_of(int t);

enum { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ3_XXS = 18, T_IQ4_NL = 20,
       T_IQ3_S = 21, T_IQ2_S = 22, T_IQ4_XS = 23, T_BF16 = 30, T_Q2_0 = 42 };
enum { META_SUP = 0, META_H32 = 1 };
#define FC_TMAX 8                       /* tokens per product call on the decode kernels */

typedef struct {
    int R, C, E;                        /* rows, columns, experts (1 for a dense matrix) */
    int kind, meta, per16, mins, src;   /* 4 / 8; META_*; multiplier per 16 values; Q4_K/Q5_K mins; file type */
    int nb, nsb, Rp;                    /* 32-value blocks a row, 256-value blocks a row, rows padded to pairs */
    int off6;                           /* K6: the offset subtracted from the unsigned codes */
    int8_t lut[32];                     /* K4: the 16 values, twice (both 128-bit lanes) */
    uint8_t* codes;                     /* K4: [E][Rp/2][nb][32]; K8: [E][Rp][nb][32] */
    float* d;                           /* SUP: [E][Rp][nsb] */
    float* dmin;                        /* mins: [E][Rp][nsb] */
    int8_t* mult;                       /* SUP: [E][Rp][nb * (per16 ? 2 : 1)] */
    uint8_t* mn;                        /* mins: [E][Rp][nb] */
    uint16_t* h;                        /* H32: [E][Rp][nb] fp16 */
    size_t bytes;                       /* bytes of the repacked form */
} fc_tensor;

static const int8_t lv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static const uint8_t mag_xxs[8] = {4, 12, 20, 28, 36, 44, 52, 62}, mag_s[8] = {1, 3, 5, 7, 9, 11, 13, 15},
                     mag_2s[3] = {8, 25, 43};

static inline float f16(const uint8_t* p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }

/* ------------------------------------------------------------------ block decode to integer codes
   per 256 (or 32 / 64) values of the file: the integer codes c, and the scale parts. */
typedef struct { int8_t c[256]; int8_t m16[16]; uint8_t mn[8]; float d, dmin; float h[8]; } blk_t;

static void scale_min_k(const uint8_t* sc, uint8_t* d, uint8_t* m) {
    for (int j = 0; j < 8; j++) {
        const int k = j & 3, u = j >> 2, sh = 2 * u;
        const int x = sc[k], y = sc[k + 4], z = sc[k + 8];
        d[j] = (uint8_t)(((u ? z : x) & 15) | ((x >> sh) & 48));
        m[j] = (uint8_t)(((u ? (z >> 4) : y) & 15) | ((y >> sh) & 48));
    }
}

/* one block of type ty at b -> B (values per block: 256, except IQ4_NL / Q8_0 32 and Q2_0 64) */
static int decode_block(int ty, const uint8_t* b, blk_t* B) {
    switch (ty) {
    case T_Q4_K: case T_Q5_K: {
        uint8_t sc[8], mn[8];
        B->d = f16(b); B->dmin = f16(b + 2);
        scale_min_k(b + 4, sc, mn);
        const uint8_t* qh = b + 16;
        const uint8_t* qs = b + (ty == T_Q4_K ? 16 : 48);
        for (int r = 0; r < 8; r++) {
            B->m16[2 * r] = B->m16[2 * r + 1] = (int8_t)sc[r]; B->mn[r] = mn[r];
            for (int j = 0; j < 32; j++) {
                int q = (qs[(r >> 1) * 32 + j] >> (4 * (r & 1))) & 15;
                if (ty == T_Q5_K) q |= ((qh[j] >> r) & 1) << 4;
                B->c[r * 32 + j] = (int8_t)q;
            }
        }
        return 256;
    }
    case T_Q6_K: {
        const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*)(b + 192);
        B->d = f16(b + 208); B->dmin = 0;
        for (int n = 0; n < 2; n++) {
            const uint8_t* L = ql + 64 * n; const uint8_t* H = qh + 32 * n; const int8_t* S = sc + 8 * n;
            for (int j = 0; j < 32; j++) {
                const int q0 = (L[j] & 15) | (((H[j] >> 0) & 3) << 4), q1 = (L[32 + j] & 15) | (((H[j] >> 2) & 3) << 4);
                const int q2 = (L[j] >> 4) | (((H[j] >> 4) & 3) << 4), q3 = (L[32 + j] >> 4) | (((H[j] >> 6) & 3) << 4);
                B->c[128 * n + j] = (int8_t)(q0 - 32); B->c[128 * n + 32 + j] = (int8_t)(q1 - 32);
                B->c[128 * n + 64 + j] = (int8_t)(q2 - 32); B->c[128 * n + 96 + j] = (int8_t)(q3 - 32);
            }
            for (int p = 0; p < 4; p++) { B->m16[8 * n + 2 * p] = S[2 * p]; B->m16[8 * n + 2 * p + 1] = S[2 * p + 1]; }
        }
        return 256;
    }
    case T_Q8_0:
        B->h[0] = f16(b);
        memcpy(B->c, b + 2, 32);
        return 32;
    case T_IQ4_NL:
        B->h[0] = f16(b);
        for (int j = 0; j < 16; j++) { B->c[j] = (int8_t)(b[2 + j] & 15); B->c[j + 16] = (int8_t)(b[2 + j] >> 4); }
        return 32;                       /* (codes are the level indices here) */
    case T_IQ4_XS: {
        B->d = f16(b); B->dmin = 0;
        uint16_t sh; memcpy(&sh, b + 2, 2);
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
            B->m16[2 * ib] = B->m16[2 * ib + 1] = (int8_t)(ls - 32);
            for (int j = 0; j < 16; j++) {
                B->c[ib * 32 + j] = (int8_t)(b[8 + 16 * ib + j] & 15);
                B->c[ib * 32 + 16 + j] = (int8_t)(b[8 + 16 * ib + j] >> 4);
            }
        }
        return 256;
    }
    case T_Q2_0:
        B->h[0] = B->h[1] = f16(b);
        for (int i = 0; i < 64; i++) B->c[i] = (int8_t)(((b[2 + i / 4] >> (2 * (i % 4))) & 3) - 1);
        return 64;
    case T_IQ3_XXS: {
        B->d = f16(b) * 0.25f; B->dmin = 0;
        for (int r = 0; r < 8; r++) {
            uint32_t aux; memcpy(&aux, b + 66 + 4 * r, 4);
            const int s = (int)(aux >> 28);
            B->m16[2 * r] = B->m16[2 * r + 1] = (int8_t)(2 * s + 1);
            for (int l = 0; l < 4; l++) {
                uint32_t s7 = (aux >> (7 * l)) & 127;
                const uint32_t s8 = s7 | ((uint32_t)(__builtin_popcount(s7) & 1) << 7);
                for (int k = 0; k < 2; k++) {
                    const uint32_t g = nbq_iq3xxs_grid[b[2 + 8 * r + 2 * l + k]];
                    for (int v = 0; v < 4; v++) {
                        const int i = 8 * l + 4 * k + v;
                        const int mag = (g >> (8 * v)) & 255;
                        B->c[r * 32 + i] = (int8_t)((s8 >> (4 * k + v)) & 1 ? -mag : mag);
                    }
                }
            }
        }
        return 256;
    }
    case T_IQ3_S: {
        B->d = f16(b); B->dmin = 0;
        for (int r = 0; r < 8; r++) {
            const int s = (b[106 + (r >> 1)] >> (4 * (r & 1))) & 15;
            B->m16[2 * r] = B->m16[2 * r + 1] = (int8_t)(1 + 2 * s);
            for (int k = 0; k < 8; k++) {
                const int idx = b[2 + 8 * r + k] | (((b[66 + r] >> k) & 1) << 8);
                const uint32_t g = nbq_iq3s_grid[idx];
                for (int v = 0; v < 4; v++) {
                    const int i = 4 * k + v, mag = (g >> (8 * v)) & 255;
                    const int neg = (b[74 + 4 * r + i / 8] >> (i % 8)) & 1;
                    B->c[r * 32 + i] = (int8_t)(neg ? -mag : mag);
                }
            }
        }
        return 256;
    }
    case T_IQ2_S: {
        B->d = f16(b) * 0.125f; B->dmin = 0;
        for (int r = 0; r < 8; r++) {
            const int sc = b[74 + r];
            B->m16[2 * r] = (int8_t)(2 * (sc & 15) + 1); B->m16[2 * r + 1] = (int8_t)(2 * (sc >> 4) + 1);
            for (int l = 0; l < 4; l++) {
                const int idx = b[2 + 4 * r + l] | (((b[66 + r] >> (2 * l)) & 3) << 8);
                const uint64_t g = nbq_iq2s_grid[idx];
                for (int v = 0; v < 8; v++) {
                    const int i = 8 * l + v, mag = (int)((g >> (8 * v)) & 255);
                    const int neg = (b[34 + 4 * r + l] >> v) & 1;
                    B->c[r * 32 + i] = (int8_t)(neg ? -mag : mag);
                }
            }
        }
        return 256;
    }
    }
    return 0;
}

static int block_bytes(int ty) {
    switch (ty) {
    case T_Q4_K: return 144; case T_Q5_K: return 176; case T_Q6_K: return 210; case T_Q8_0: return 34;
    case T_IQ4_NL: return 18; case T_IQ4_XS: return 136; case T_Q2_0: return 18; case T_IQ3_XXS: return 98;
    case T_IQ3_S: return 110; case T_IQ2_S: return 82;
    }
    return 0;
}

/* the K4 table of a type and the index of each code value (-1: not in the table) */
static int k4_table(int ty, int8_t lut[16], int idx[256]) {
    int n = 0;
    memset(lut, 0, 16);
    switch (ty) {
    case T_IQ4_NL: case T_IQ4_XS: for (int i = 0; i < 16; i++) lut[i] = lv4[i]; n = 16; break;
    case T_Q4_K: for (int i = 0; i < 16; i++) lut[i] = (int8_t)i; n = 16; break;
    case T_Q2_0: for (int i = 0; i < 4; i++) lut[i] = (int8_t)(i - 1); n = 4; break;
    case T_IQ3_XXS: for (int i = 0; i < 8; i++) { lut[i] = (int8_t)mag_xxs[i]; lut[8 + i] = (int8_t)-mag_xxs[i]; } n = 16; break;
    case T_IQ3_S: for (int i = 0; i < 8; i++) { lut[i] = (int8_t)mag_s[i]; lut[8 + i] = (int8_t)-mag_s[i]; } n = 16; break;
    case T_IQ2_S: for (int i = 0; i < 3; i++) { lut[i] = (int8_t)mag_2s[i]; lut[8 + i] = (int8_t)-mag_2s[i]; } n = 16; break;
    default: return 0;
    }
    for (int v = 0; v < 256; v++) idx[v] = -1;
    for (int i = n - 1; i >= 0; i--) idx[(uint8_t)lut[i]] = i;
    if (ty == T_IQ2_S) for (int i = 3; i < 8; i++) idx[(uint8_t)lut[i]] = -1, idx[(uint8_t)lut[8 + i]] = -1;
    return 1;
}

/* ------------------------------------------------------------------ the tensor and its repack */
/* Placement. Each thread writes its own slice at load, so its pages should be on its NUMA node. Two things broke
   that on a host whose memory is mostly page cache: with free memory short on a node, first-touch pages went to the
   other node (zone_reclaim_mode 0); and a 2 MB huge page holds the slices of several threads (an expert's slice is
   20 KB). So the repack binds each thread's allocations to its node (KINDI_FNCPU_BIND=0: off), and huge pages are
   used only for matrices whose slices are at least 2 MB (KINDI_FNCPU_HUGE: dense (default), all, 0). */
static int env_huge(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("KINDI_FNCPU_HUGE"); v = !e || !*e || strcmp(e, "dense") == 0 ? 1 : strcmp(e, "all") == 0 ? 2 : 0; }
    return v;
}
static void* big_alloc_h(size_t n, int huge) {
    if (!n) return NULL;
    void* p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    if (n >= (2u << 20)) madvise(p, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
    return p;
}
static void* big_alloc(size_t n) { return big_alloc_h(n, env_huge() != 0); }
static void bind_here(int on) {          /* the calling (pinned) thread's allocations: its node only, or default */
    static int env = -1;
    if (env < 0) { const char* e = getenv("KINDI_FNCPU_BIND"); env = !(e && strcmp(e, "0") == 0); }
    if (!env) return;
    if (on) {
        unsigned cpu = 0, node = 0;
        if (syscall(SYS_getcpu, &cpu, &node, NULL) != 0 || node >= 64) return;
        unsigned long mask = 1UL << node;
        syscall(SYS_set_mempolicy, 2 /* MPOL_BIND */, &mask, 64UL);
    } else
        syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, NULL, 0UL);
}

/* the float matrices (BF16, F16, F32) are quantized per 32 values: d = amax / 127 (fp16), c = rint(w / d) */
static inline float src_float(int ty, const uint8_t* p, long i) {
    if (ty == T_F32) { float v; memcpy(&v, p + 4 * i, 4); return v; }
    uint16_t h; memcpy(&h, p + 2 * i, 2);
    if (ty == T_F16) return _cvtsh_ss(h);
    uint32_t u = (uint32_t)h << 16; float v; memcpy(&v, &u, 4); return v;
}

fc_tensor* fc_new(int ty, int R, int C, int E) {
    if (C % 32) return NULL;
    fc_tensor* T = calloc(1, sizeof(*T));
    T->R = R; T->C = C; T->E = E; T->src = ty;
    T->nb = C / 32;
    int8_t lut[16]; int idx[256];
    const int k4 = k4_table(ty, lut, idx);
    const char* k6e = getenv("KINDI_FNCPU_K6");
    T->kind = k4 ? 4 : ((ty == T_Q6_K || ty == T_Q5_K) && k6e && strcmp(k6e, "1") == 0) ? 6 : 8;
    T->off6 = ty == T_Q6_K ? 32 : 0;
    T->meta = (ty == T_IQ4_NL || ty == T_Q8_0 || ty == T_Q2_0 || ty == T_BF16 || ty == T_F16 || ty == T_F32) ? META_H32
                                                                                                                : META_SUP;
    if (T->meta == META_SUP && C % 256) { free(T); return NULL; }
    T->nsb = T->meta == META_SUP ? C / 256 : 0;
    T->per16 = ty == T_Q6_K || ty == T_IQ2_S;
    T->mins = ty == T_Q4_K || ty == T_Q5_K;
    T->Rp = T->kind == 4 ? (R + 1) & ~1 : R;
    memcpy(T->lut, lut, 16); memcpy(T->lut + 16, lut, 16);
    const size_t rows = (size_t)E * T->Rp;
    const size_t ncode = rows * T->nb * (T->kind == 4 ? 16 : T->kind == 6 ? 24 : 32);
    const int huge = env_huge() == 2 || (env_huge() == 1 && E == 1 && ncode / 40 >= (4u << 20));
    T->codes = big_alloc_h(ncode, huge);
    T->bytes = ncode;
    if (T->meta == META_SUP) {
        T->d = big_alloc(sizeof(float) * rows * T->nsb); T->bytes += sizeof(float) * rows * T->nsb;
        T->mult = big_alloc(rows * T->nb * (T->per16 ? 2 : 1)); T->bytes += rows * T->nb * (T->per16 ? 2 : 1);
        if (T->mins) {
            T->dmin = big_alloc(sizeof(float) * rows * T->nsb); T->mn = big_alloc(rows * T->nb);
            T->bytes += sizeof(float) * rows * T->nsb + rows * T->nb;
        }
    } else {
        T->h = big_alloc(2 * rows * T->nb); T->bytes += 2 * rows * T->nb;
    }
    if (!T->codes) { free(T); return NULL; }
    return T;
}

/* rows [r0, r1) of expert e from the file bytes src (row-major, rows of C values); for K4, r0 and r1 even (or r1 = R) */
static int repack_rows(fc_tensor* T, const uint8_t* src, int e, int r0, int r1) {
    const int ty = T->src, C = T->C, nb = T->nb;
    int8_t lut[16]; int idx[256];
    k4_table(ty, lut, idx);
    const int fl = ty == T_BF16 || ty == T_F16 || ty == T_F32;
    const long rowbytes = fl ? (long)C * (ty == T_F32 ? 4 : 2) : (long)C / (ty == T_Q8_0 || ty == T_IQ4_NL ? 32 : ty == T_Q2_0 ? 64 : 256) * block_bytes(ty);
    const uint8_t* base = src + (size_t)e * T->R * rowbytes;
    blk_t B;
    int8_t* crow = aligned_alloc(64, ((size_t)C + 63) / 64 * 64);
    for (int r = r0; r < r1; r++) {
        const size_t gr = (size_t)e * T->Rp + r;
        const uint8_t* p = base + (size_t)r * rowbytes;
        if (fl) {
            for (int b = 0; b < nb; b++) {
                float am = 0;
                for (int j = 0; j < 32; j++) { const float v = fabsf(src_float(ty, p, (long)b * 32 + j)); am = v > am ? v : am; }
                const uint16_t hd = _cvtss_sh(am / 127.f, 0);
                const float dd = _cvtsh_ss(hd), inv = dd > 0 ? 1.f / dd : 0.f;
                T->h[gr * nb + b] = hd;
                for (int j = 0; j < 32; j++) {
                    float q = rintf(src_float(ty, p, (long)b * 32 + j) * inv);
                    crow[b * 32 + j] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
                }
            }
        } else {
            int v = 0, b32 = 0;
            const int bb = block_bytes(ty);
            while (v < C) {
                const int n = decode_block(ty, p, &B);
                p += bb;
                memcpy(crow + v, B.c, n);
                if (T->meta == META_H32) {
                    for (int k = 0; k < n / 32; k++) T->h[gr * nb + b32 + k] = _cvtss_sh(B.h[k], 0);
                } else {
                    const int sb = v / 256;
                    T->d[gr * T->nsb + sb] = B.d;
                    for (int k = 0; k < 8; k++) {
                        if (T->per16) { T->mult[(gr * nb + b32 + k) * 2] = B.m16[2 * k]; T->mult[(gr * nb + b32 + k) * 2 + 1] = B.m16[2 * k + 1]; }
                        else T->mult[gr * nb + b32 + k] = B.m16[2 * k];
                        if (T->mins) T->mn[gr * nb + b32 + k] = B.mn[k];
                    }
                    if (T->mins) T->dmin[gr * T->nsb + sb] = B.dmin;
                }
                v += n; b32 += n / 32;
            }
        }
        if (T->kind == 8) {
            memcpy(T->codes + gr * nb * 32, crow, C);
        } else if (T->kind == 6) {
            for (int b = 0; b < nb; b++) {
                uint8_t* dst = T->codes + (gr * nb + b) * 24;
                uint8_t u[32];
                for (int i = 0; i < 32; i++) u[i] = (uint8_t)(crow[b * 32 + i] + T->off6);
                for (int i = 0; i < 16; i++) dst[i] = (uint8_t)((u[i] & 15) | ((u[i + 16] & 15) << 4));
                for (int j = 0; j < 8; j++) {
                    uint8_t hb = 0;
                    for (int k = 0; k < 4; k++) hb |= (uint8_t)(((u[j + 8 * k] >> 4) & 3) << (2 * k));
                    dst[16 + j] = hb;
                }
            }
        } else {
            /* the pair's byte j of block b: low nibble = first row, high = second */
            uint8_t* dst = T->codes + ((size_t)e * (T->Rp / 2) + r / 2) * nb * 32;
            const int hi = r & 1;
            for (int i = 0; i < C; i++) {
                int code = crow[i];
                if (ty != T_IQ4_NL && ty != T_IQ4_XS && ty != T_Q4_K) {
                    code = idx[(uint8_t)crow[i]];
                    if (code < 0) { free(crow); return -2; }
                }
                uint8_t* q = dst + i;
                *q = hi ? (uint8_t)((*q & 15) | (code << 4)) : (uint8_t)((*q & 0xF0) | code);
            }
        }
    }
    free(crow);
    return 0;
}

/* ------------------------------------------------------------------ activations
   int8 values with one scale per 32 values: dx[b], and fx[b] = dx * sum of the block's values (for the mins). */
static void quant_act(const float* x, int C, int8_t* q, float* dx, float* fx) {
    const int G = 32;          /* one scale per 32 values (per 256 lost 4-5 of the gate's 174 tokens) */
    const __m256 sm = _mm256_set1_ps(-0.f);
    for (int g0 = 0; g0 < C; g0 += G) {
        __m256 am = _mm256_setzero_ps();
        for (int j = 0; j < G; j += 8) am = _mm256_max_ps(am, _mm256_andnot_ps(sm, _mm256_loadu_ps(x + g0 + j)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4)); m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float a = _mm_cvtss_f32(m4);
        const float d = a / 127.f;
        const __m256 id = _mm256_set1_ps(a > 0 ? 127.f / a : 0.f);
        for (int b = g0 / 32; b < (g0 + G) / 32; b++) {
            __m256i s = _mm256_setzero_si256();
            int32_t t32[32] __attribute__((aligned(32)));
            for (int j = 0; j < 32; j += 8) {
                const __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + b * 32 + j), id));
                s = _mm256_add_epi32(s, v);
                _mm256_store_si256((__m256i*)(t32 + j), v);
            }
            const __m256i p16 = _mm256_packs_epi32(_mm256_load_si256((const __m256i*)t32), _mm256_load_si256((const __m256i*)(t32 + 8)));
            const __m256i p16b = _mm256_packs_epi32(_mm256_load_si256((const __m256i*)(t32 + 16)), _mm256_load_si256((const __m256i*)(t32 + 24)));
            __m256i p8 = _mm256_packs_epi16(p16, p16b);
            p8 = _mm256_permutevar8x32_epi32(p8, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i*)(q + b * 32), p8);
            __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
            s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
            dx[b] = d; fx[b] = d * (float)_mm_cvtsi128_si32(s4);
        }
    }
}

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}
/* the int16 multiplier vector of a 32-value block: per 32 (one value) or per 16 (the low 128 bits hold the
   maddubs pairs of values 0-15, the high 128 bits those of 16-31) */
static inline __m256i mult_vec(const fc_tensor* T, size_t gr, int b) {
    if (T->per16) return _mm256_set_m128i(_mm_set1_epi16(T->mult[(gr * T->nb + b) * 2 + 1]), _mm_set1_epi16(T->mult[(gr * T->nb + b) * 2]));
    return _mm256_set1_epi16(T->mult[gr * T->nb + b]);
}

/* the 32 int8 codes of a K6 block */
static inline __m256i k6_codes(const uint8_t* p, int off) {
    const __m128i m4 = _mm_set1_epi8(15);
    const __m128i lo = _mm_loadu_si128((const __m128i*)p);
    const __m256i l = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(lo, 4), m4), _mm_and_si128(lo, m4));
    int64_t hb; memcpy(&hb, p + 16, 8);
    __m256i h = _mm256_srlv_epi64(_mm256_set1_epi64x(hb), _mm256_set_epi64x(6, 4, 2, 0));
    h = _mm256_and_si256(h, _mm256_set1_epi8(3));
    return _mm256_sub_epi8(_mm256_or_si256(l, _mm256_slli_epi16(h, 4)), _mm256_set1_epi8((char)off));
}

/* ------------------------------------------------------------------ the kernels
   y[t * ys + r] for rows [r0, r1) of expert e (K4: pairs, r0 even), nt tokens of quantized x (q: [nt][C]).
   SUP: per 256 values the products times the int multipliers are summed as integers, then one float step
   (d * dx); the mins term (Q4_K, Q5_K) is dmin * sum(min * fx). H32: one float step per 32 values. */
static void k4_rows(const fc_tensor* T, int e, int r0, int r1, int nt, const int8_t* q, const float* dx,
                    const float* fx, float* y, long ys) {
    const int nb = T->nb, C = T->C;
    const __m256i lut = _mm256_loadu_si256((const __m256i*)T->lut);
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m256i one = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r += 2) {
        const size_t gr = (size_t)e * T->Rp + r;
        const uint8_t* cp = T->codes + ((size_t)e * (T->Rp / 2) + r / 2) * nb * 32;
        __m256 acc[2][FC_TMAX];
        for (int t = 0; t < nt; t++) acc[0][t] = acc[1][t] = _mm256_setzero_ps();
        float mins[2][FC_TMAX] = {{0}};
        if (T->meta == META_SUP) {
            for (int sb = 0; sb < T->nsb; sb++) {
                const float d0 = T->d[gr * T->nsb + sb], d1 = T->d[(gr + 1) * T->nsb + sb];
                for (int b = sb * 8; b < sb * 8 + 8; b++) {
                    const __m256i v = _mm256_loadu_si256((const __m256i*)(cp + (size_t)b * 32));
                    const __m256i c0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
                    const __m256i c1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
                    const __m256i a0 = _mm256_abs_epi8(c0), a1 = _mm256_abs_epi8(c1);
                    const __m256i mv0 = mult_vec(T, gr, b), mv1 = mult_vec(T, gr + 1, b);
                    for (int t = 0; t < nt; t++) {
                        const __m256i x = _mm256_loadu_si256((const __m256i*)(q + (size_t)t * C + b * 32));
                        const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x, c0)), mv0);
                        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x, c1)), mv1);
                        const float xd = dx[t * (C / 32) + b];
                        acc[0][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(d0 * xd), acc[0][t]);
                        acc[1][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(d1 * xd), acc[1][t]);
                    }
                }
                if (T->mins) {
                    const __m256 mn0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(T->mn + gr * nb + sb * 8))));
                    const __m256 mn1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(T->mn + (gr + 1) * nb + sb * 8))));
                    for (int t = 0; t < nt; t++) {
                        const __m256 f = _mm256_loadu_ps(fx + t * (C / 32) + sb * 8);
                        mins[0][t] += T->dmin[gr * T->nsb + sb] * hsum8(_mm256_mul_ps(mn0, f));
                        mins[1][t] += T->dmin[(gr + 1) * T->nsb + sb] * hsum8(_mm256_mul_ps(mn1, f));
                    }
                }
            }
        } else {
            for (int b = 0; b < nb; b++) {
                const __m256i v = _mm256_loadu_si256((const __m256i*)(cp + (size_t)b * 32));
                const __m256i c0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
                const __m256i c1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
                const __m256i a0 = _mm256_abs_epi8(c0), a1 = _mm256_abs_epi8(c1);
                const float s0 = _cvtsh_ss(T->h[gr * nb + b]), s1 = _cvtsh_ss(T->h[(gr + 1) * nb + b]);
                for (int t = 0; t < nt; t++) {
                    const __m256i x = _mm256_loadu_si256((const __m256i*)(q + (size_t)t * C + b * 32));
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(a0, _mm256_sign_epi8(x, c0)), one);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(x, c1)), one);
                    const float xd = dx[t * (C / 32) + b];
                    acc[0][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(s0 * xd), acc[0][t]);
                    acc[1][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(s1 * xd), acc[1][t]);
                }
            }
        }
        for (int t = 0; t < nt; t++) {
            y[t * ys + r] = hsum8(acc[0][t]) - mins[0][t];
            if (r + 1 < T->R) y[t * ys + r + 1] = hsum8(acc[1][t]) - mins[1][t];
        }
    }
}

static inline __attribute__((always_inline)) void k8_rows_g(const fc_tensor* T, int e, int r0, int r1, int nt,
        const int8_t* q, const float* dx, const float* fx, float* y, long ys, const int K6) {
    const int nb = T->nb, C = T->C;
    const __m256i one = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const size_t gr = (size_t)e * T->Rp + r;
        const int8_t* cp = (const int8_t*)T->codes + gr * nb * (K6 ? 24 : 32);
#define K8C(b) (K6 ? k6_codes((const uint8_t*)cp + (size_t)(b) * 24, T->off6) : _mm256_loadu_si256((const __m256i*)(cp + (size_t)(b) * 32)))
        __m256 acc[FC_TMAX];
        float mins[FC_TMAX] = {0};
        for (int t = 0; t < nt; t++) acc[t] = _mm256_setzero_ps();
        if (T->meta == META_SUP) {
            for (int sb = 0; sb < T->nsb; sb++) {
                const float d = T->d[gr * T->nsb + sb];
                for (int b = sb * 8; b < sb * 8 + 8; b++) {
                    const __m256i c = K8C(b);
                    const __m256i a = _mm256_abs_epi8(c);
                    const __m256i mv = mult_vec(T, gr, b);
                    for (int t = 0; t < nt; t++) {
                        const __m256i x = _mm256_loadu_si256((const __m256i*)(q + (size_t)t * C + b * 32));
                        const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(a, _mm256_sign_epi8(x, c)), mv);
                        acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * dx[t * (C / 32) + b]), acc[t]);
                    }
                }
                if (T->mins) {
                    const __m256 mn = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(T->mn + gr * nb + sb * 8))));
                    for (int t = 0; t < nt; t++)
                        mins[t] += T->dmin[gr * T->nsb + sb] * hsum8(_mm256_mul_ps(mn, _mm256_loadu_ps(fx + t * (C / 32) + sb * 8)));
                }
            }
        } else {
            for (int b = 0; b < nb; b++) {
                const __m256i c = K8C(b);
                const __m256i a = _mm256_abs_epi8(c);
                const float s = _cvtsh_ss(T->h[gr * nb + b]);
                for (int t = 0; t < nt; t++) {
                    const __m256i x = _mm256_loadu_si256((const __m256i*)(q + (size_t)t * C + b * 32));
                    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(a, _mm256_sign_epi8(x, c)), one);
                    acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(s * dx[t * (C / 32) + b]), acc[t]);
                }
            }
        }
        for (int t = 0; t < nt; t++) y[t * ys + r] = hsum8(acc[t]) - mins[t];
#undef K8C
    }
}
static void k8_rows(const fc_tensor* T, int e, int r0, int r1, int nt, const int8_t* q, const float* dx,
                    const float* fx, float* y, long ys) { k8_rows_g(T, e, r0, r1, nt, q, dx, fx, y, ys, 0); }
static void k6_rows(const fc_tensor* T, int e, int r0, int r1, int nt, const int8_t* q, const float* dx,
                    const float* fx, float* y, long ys) { k8_rows_g(T, e, r0, r1, nt, q, dx, fx, y, ys, 1); }

static void rows_any(const fc_tensor* T, int e, int r0, int r1, int nt, const int8_t* q, const float* dx,
                     const float* fx, float* y, long ys) {
    if (r0 >= r1) return;
    if (T->kind == 4) k4_rows(T, e, r0, r1, nt, q, dx, fx, y, ys);
    else if (T->kind == 6) k6_rows(T, e, r0, r1, nt, q, dx, fx, y, ys);
    else k8_rows(T, e, r0, r1, nt, q, dx, fx, y, ys);
}

/* any number of tokens: rows in blocks of 16, each block's codes (in L1/L2) used for every 8-token tile */
static void rows_big(const fc_tensor* T, int e, int r0, int r1, int nt, const int8_t* q, const float* dx,
                     const float* fx, float* y, long ys) {
    if (nt <= FC_TMAX) { rows_any(T, e, r0, r1, nt, q, dx, fx, y, ys); return; }
    const int C = T->C;
    for (int rb = r0; rb < r1; rb += 16) {
        const int re = rb + 16 < r1 ? rb + 16 : r1;
        for (int t0 = 0; t0 < nt; t0 += FC_TMAX) {
            const int m = nt - t0 < FC_TMAX ? nt - t0 : FC_TMAX;
            rows_any(T, e, rb, re, m, q + (size_t)t0 * C, dx + (size_t)t0 * (C / 32), fx + (size_t)t0 * (C / 32), y + t0 * ys, ys);
        }
    }
}

/* ------------------------------------------------------------------ prompt products in fp32 (the GEMM path)
   For many tokens a call the products are bound by arithmetic, not by weight bytes. The weights of a 16-row block
   are decoded once per 256 columns into an fp32 panel [256][16] (the same values the int8 kernels use: d * mult * c -
   dmin * min, or the fp16 scale times c), and every token of the call is multiplied against it in fp32 with a 6-token
   by 16-row register tile. The activations stay fp32 (no int8 rounding of x). x row t is x + (xi ? xi[t] : t) * xs,
   y row t is y + (yi ? yi[t] : t) * ys; acc: add to y instead of storing. */
#define GM_KC 256
#define GM_MR 6

/* rows [r, r + nr) (nr <= 16), 32-value blocks [b0, b0 + nbk) of expert e -> out [16][nbk * 32] (rows >= nr zero) */
static void gm_decode(const fc_tensor* T, int e, int r, int nr, int b0, int nbk, float* out) {
    const int nb = T->nb, ld = nbk * 32;
    const __m256i lut = _mm256_loadu_si256((const __m256i*)T->lut);
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int i = 0; i < 16; i++) {
        float* o = out + (size_t)i * ld;
        if (i >= nr) { memset(o, 0, sizeof(float) * ld); continue; }
        const int row = r + i;
        const size_t gr = (size_t)e * T->Rp + row;
        for (int b = b0; b < b0 + nbk; b++) {
            __m256i c;
            if (T->kind == 4) {
                const __m256i v = _mm256_loadu_si256((const __m256i*)(T->codes + (((size_t)e * (T->Rp / 2) + row / 2) * nb + b) * 32));
                c = _mm256_shuffle_epi8(lut, _mm256_and_si256(row & 1 ? _mm256_srli_epi16(v, 4) : v, m4));
            } else if (T->kind == 6)
                c = k6_codes(T->codes + (gr * nb + b) * 24, T->off6);
            else
                c = _mm256_loadu_si256((const __m256i*)(T->codes + (gr * nb + b) * 32));
            float s0, s1, mn = 0;
            if (T->meta == META_SUP) {
                const float d = T->d[gr * T->nsb + b / 8];
                if (T->per16) { s0 = d * T->mult[(gr * nb + b) * 2]; s1 = d * T->mult[(gr * nb + b) * 2 + 1]; }
                else s0 = s1 = d * T->mult[gr * nb + b];
                if (T->mins) mn = T->dmin[gr * T->nsb + b / 8] * T->mn[gr * nb + b];
            } else
                s0 = s1 = _cvtsh_ss(T->h[gr * nb + b]);
            const __m128i lo = _mm256_castsi256_si128(c), hi = _mm256_extracti128_si256(c, 1);
            const __m256 vs0 = _mm256_set1_ps(s0), vs1 = _mm256_set1_ps(s1), vm = _mm256_set1_ps(mn);
            float* ob = o + (b - b0) * 32;
            _mm256_store_ps(ob, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), vs0, vm));
            _mm256_store_ps(ob + 8, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), vs0, vm));
            _mm256_store_ps(ob + 16, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), vs1, vm));
            _mm256_store_ps(ob + 24, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), vs1, vm));
        }
    }
}

/* K4 rows [r, r + nr) (r even, nr <= 16) straight into the panel P [nbk * 32][16]: the 16 rows' codes of a block as
   int8 (the table look-up of each pair's nibbles), a 16 x 16 byte transpose in each 128-bit lane (values 0-15 and
   16-31 of the block), then per column the rows' codes widened, scaled and offset in fp32. About a third of the
   work of decoding row by row and transposing floats; the experts' slices are decoded once per prompt chunk for
   ~20 tokens, so the decode is a large part of their cost. */
static void gm_decode_t4(const fc_tensor* T, int e, int r, int nr, int b0, int nbk, float* P) {
    const int nb = T->nb;
    const __m256i lut = _mm256_loadu_si256((const __m256i*)T->lut);
    const __m256i m4 = _mm256_set1_epi8(15);
    float s0[16] __attribute__((aligned(32))), s1[16] __attribute__((aligned(32))), mn[16] __attribute__((aligned(32)));
    for (int b = b0; b < b0 + nbk; b++) {
        __m256i c[16];
        for (int p = 0; p < 8; p++) {
            const int row = r + 2 * p;
            if (row >= r + nr) { c[2 * p] = c[2 * p + 1] = _mm256_setzero_si256(); continue; }
            const __m256i v = _mm256_loadu_si256((const __m256i*)(T->codes + (((size_t)e * (T->Rp / 2) + row / 2) * nb + b) * 32));
            c[2 * p] = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
            c[2 * p + 1] = row + 1 < r + nr ? _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4)) : _mm256_setzero_si256();
        }
        for (int i = 0; i < 16; i++) {
            if (i >= nr) { s0[i] = s1[i] = mn[i] = 0; continue; }
            const size_t gr = (size_t)e * T->Rp + r + i;
            if (T->meta == META_SUP) {
                const float d = T->d[gr * T->nsb + b / 8];
                if (T->per16) { s0[i] = d * T->mult[(gr * nb + b) * 2]; s1[i] = d * T->mult[(gr * nb + b) * 2 + 1]; }
                else s0[i] = s1[i] = d * T->mult[gr * nb + b];
                mn[i] = T->mins ? T->dmin[gr * T->nsb + b / 8] * T->mn[gr * nb + b] : 0.f;
            } else { s0[i] = s1[i] = _cvtsh_ss(T->h[gr * nb + b]); mn[i] = 0.f; }
        }
        /* transpose: col j (low lane: value j, high lane: value 16 + j) = the 16 rows' bytes */
        __m256i t[16], u[16], v[16];
        for (int i = 0; i < 8; i++) { t[2 * i] = _mm256_unpacklo_epi8(c[2 * i], c[2 * i + 1]); t[2 * i + 1] = _mm256_unpackhi_epi8(c[2 * i], c[2 * i + 1]); }
        for (int g = 0; g < 4; g++) {          /* rows 4g .. 4g + 3: t[4g], t[4g + 1] (rows 4g, 4g+1) and t[4g + 2], t[4g + 3] */
            u[4 * g + 0] = _mm256_unpacklo_epi16(t[4 * g], t[4 * g + 2]);
            u[4 * g + 1] = _mm256_unpackhi_epi16(t[4 * g], t[4 * g + 2]);
            u[4 * g + 2] = _mm256_unpacklo_epi16(t[4 * g + 1], t[4 * g + 3]);
            u[4 * g + 3] = _mm256_unpackhi_epi16(t[4 * g + 1], t[4 * g + 3]);
        }
        for (int h = 0; h < 2; h++)            /* rows 0-7 (u[0..7]) and 8-15 (u[8..15]) */
            for (int j = 0; j < 4; j++) {
                v[8 * h + 2 * j] = _mm256_unpacklo_epi32(u[8 * h + j], u[8 * h + 4 + j]);
                v[8 * h + 2 * j + 1] = _mm256_unpackhi_epi32(u[8 * h + j], u[8 * h + 4 + j]);
            }
        const __m256 S0a = _mm256_load_ps(s0), S0b = _mm256_load_ps(s0 + 8), S1a = _mm256_load_ps(s1), S1b = _mm256_load_ps(s1 + 8);
        const __m256 Ma = _mm256_load_ps(mn), Mb = _mm256_load_ps(mn + 8);
        float* pb = P + (size_t)(b - b0) * 32 * 16;
        for (int j = 0; j < 8; j++) {
            for (int hh = 0; hh < 2; hh++) {
                const __m256i col = hh ? _mm256_unpackhi_epi64(v[j], v[j + 8]) : _mm256_unpacklo_epi64(v[j], v[j + 8]);
                const int k = 2 * j + hh;                      /* value k (low lane) and 16 + k (high lane) */
                const __m128i lo = _mm256_castsi256_si128(col), hi = _mm256_extracti128_si256(col, 1);
                float* d0 = pb + (size_t)k * 16;
                float* d1 = pb + (size_t)(16 + k) * 16;
                _mm256_store_ps(d0, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), S0a, Ma));
                _mm256_store_ps(d0 + 8, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), S0b, Mb));
                _mm256_store_ps(d1, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), S1a, Ma));
                _mm256_store_ps(d1 + 8, _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), S1b, Mb));
            }
        }
    }
}

/* in [16][ld] -> P [kc][16] (8 x 8 transposes) */
static void gm_transpose(const float* in, int ld, int kc, float* P) {
    for (int k = 0; k < kc; k += 8)
        for (int h = 0; h < 2; h++) {
            const float* s = in + (size_t)h * 8 * ld + k;
            __m256 r0 = _mm256_load_ps(s), r1 = _mm256_load_ps(s + ld), r2 = _mm256_load_ps(s + 2 * ld), r3 = _mm256_load_ps(s + 3 * ld);
            __m256 r4 = _mm256_load_ps(s + 4 * ld), r5 = _mm256_load_ps(s + 5 * ld), r6 = _mm256_load_ps(s + 6 * ld), r7 = _mm256_load_ps(s + 7 * ld);
            __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1), t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5), t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
            r0 = _mm256_shuffle_ps(t0, t2, 0x44); r1 = _mm256_shuffle_ps(t0, t2, 0xEE); r2 = _mm256_shuffle_ps(t1, t3, 0x44); r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            r4 = _mm256_shuffle_ps(t4, t6, 0x44); r5 = _mm256_shuffle_ps(t4, t6, 0xEE); r6 = _mm256_shuffle_ps(t5, t7, 0x44); r7 = _mm256_shuffle_ps(t5, t7, 0xEE);
            float* d = P + (size_t)k * 16 + h * 8;
            _mm256_store_ps(d + 0 * 16, _mm256_permute2f128_ps(r0, r4, 0x20));
            _mm256_store_ps(d + 1 * 16, _mm256_permute2f128_ps(r1, r5, 0x20));
            _mm256_store_ps(d + 2 * 16, _mm256_permute2f128_ps(r2, r6, 0x20));
            _mm256_store_ps(d + 3 * 16, _mm256_permute2f128_ps(r3, r7, 0x20));
            _mm256_store_ps(d + 4 * 16, _mm256_permute2f128_ps(r0, r4, 0x31));
            _mm256_store_ps(d + 5 * 16, _mm256_permute2f128_ps(r1, r5, 0x31));
            _mm256_store_ps(d + 6 * 16, _mm256_permute2f128_ps(r2, r6, 0x31));
            _mm256_store_ps(d + 7 * 16, _mm256_permute2f128_ps(r3, r7, 0x31));
        }
}

static inline __attribute__((always_inline)) void gm_store(float* y, __m256 a0, __m256 a1, int nr, int add) {
    if (nr == 16) {
        if (add) { a0 = _mm256_add_ps(a0, _mm256_loadu_ps(y)); a1 = _mm256_add_ps(a1, _mm256_loadu_ps(y + 8)); }
        _mm256_storeu_ps(y, a0); _mm256_storeu_ps(y + 8, a1);
    } else {
        float v[16] __attribute__((aligned(32)));
        _mm256_store_ps(v, a0); _mm256_store_ps(v + 8, a1);
        for (int i = 0; i < nr; i++) y[i] = add ? y[i] + v[i] : v[i];
    }
}
/* (named accumulators, not arrays: GCC kept arrays on the stack and stored them every step) */
static inline __attribute__((always_inline)) void gm_tile(const float* P, int kc, const float* const* xp, float* const* yp,
                                                           int nr, int add, const int MT) {
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00,
           c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    const float *x0 = xp[0], *x1 = MT > 1 ? xp[1] : x0, *x2 = MT > 2 ? xp[2] : x0, *x3 = MT > 3 ? xp[3] : x0,
                *x4 = MT > 4 ? xp[4] : x0, *x5 = MT > 5 ? xp[5] : x0;
    for (int k = 0; k < kc; k++) {
        const __m256 w0 = _mm256_load_ps(P + 16 * k), w1 = _mm256_load_ps(P + 16 * k + 8);
#define GM_T(t, xr) if (MT > t) { const __m256 xb = _mm256_broadcast_ss(xr + k); c##t##0 = _mm256_fmadd_ps(w0, xb, c##t##0); c##t##1 = _mm256_fmadd_ps(w1, xb, c##t##1); }
        GM_T(0, x0) GM_T(1, x1) GM_T(2, x2) GM_T(3, x3) GM_T(4, x4) GM_T(5, x5)
#undef GM_T
    }
    gm_store(yp[0], c00, c01, nr, add);
    if (MT > 1) gm_store(yp[1], c10, c11, nr, add);
    if (MT > 2) gm_store(yp[2], c20, c21, nr, add);
    if (MT > 3) gm_store(yp[3], c30, c31, nr, add);
    if (MT > 4) gm_store(yp[4], c40, c41, nr, add);
    if (MT > 5) gm_store(yp[5], c50, c51, nr, add);
}
/* the narrow tile for blocks of at most 8 rows (a slice of a short matrix): 6 tokens by 8 rows, even and odd
   columns in separate accumulators (12 independent sums; more token streams would collide in the L1 sets when the
   rows of x are 4 KiB multiples apart) */
static inline __attribute__((always_inline)) void gm_store8(float* y, __m256 a, int nr, int add) {
    if (nr == 8) { if (add) a = _mm256_add_ps(a, _mm256_loadu_ps(y)); _mm256_storeu_ps(y, a); return; }
    float v[8] __attribute__((aligned(32)));
    _mm256_store_ps(v, a);
    for (int i = 0; i < nr; i++) y[i] = add ? y[i] + v[i] : v[i];
}
static inline __attribute__((always_inline)) void gm_tile_n(const float* P, int kc, const float* const* xp,
                                                             float* const* yp, int nr, int add, const int MT) {
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00,
           c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    const float *x0 = xp[0], *x1 = MT > 1 ? xp[1] : x0, *x2 = MT > 2 ? xp[2] : x0, *x3 = MT > 3 ? xp[3] : x0,
                *x4 = MT > 4 ? xp[4] : x0, *x5 = MT > 5 ? xp[5] : x0;
    for (int k = 0; k < kc; k += 2) {           /* (kc is a multiple of 32) */
        const __m256 w0 = _mm256_load_ps(P + 16 * k), w1 = _mm256_load_ps(P + 16 * k + 16);
#define GM_T(t, xr) if (MT > t) { c##t##0 = _mm256_fmadd_ps(w0, _mm256_broadcast_ss(xr + k), c##t##0); c##t##1 = _mm256_fmadd_ps(w1, _mm256_broadcast_ss(xr + k + 1), c##t##1); }
        GM_T(0, x0) GM_T(1, x1) GM_T(2, x2) GM_T(3, x3) GM_T(4, x4) GM_T(5, x5)
#undef GM_T
    }
    gm_store8(yp[0], _mm256_add_ps(c00, c01), nr, add);
    if (MT > 1) gm_store8(yp[1], _mm256_add_ps(c10, c11), nr, add);
    if (MT > 2) gm_store8(yp[2], _mm256_add_ps(c20, c21), nr, add);
    if (MT > 3) gm_store8(yp[3], _mm256_add_ps(c30, c31), nr, add);
    if (MT > 4) gm_store8(yp[4], _mm256_add_ps(c40, c41), nr, add);
    if (MT > 5) gm_store8(yp[5], _mm256_add_ps(c50, c51), nr, add);
}
static void gm_tile_n1(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile_n(P, kc, xp, yp, nr, add, 1); }
static void gm_tile_n2(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile_n(P, kc, xp, yp, nr, add, 2); }
static void gm_tile_n3(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile_n(P, kc, xp, yp, nr, add, 3); }
static void gm_tile_n4(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile_n(P, kc, xp, yp, nr, add, 4); }
static void gm_tile_n5(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile_n(P, kc, xp, yp, nr, add, 5); }
static void gm_tile_n6(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile_n(P, kc, xp, yp, nr, add, 6); }
static void gm_tile1(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile(P, kc, xp, yp, nr, add, 1); }
static void gm_tile2(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile(P, kc, xp, yp, nr, add, 2); }
static void gm_tile3(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile(P, kc, xp, yp, nr, add, 3); }
static void gm_tile4(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile(P, kc, xp, yp, nr, add, 4); }
static void gm_tile5(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile(P, kc, xp, yp, nr, add, 5); }
static void gm_tile6(const float* P, int kc, const float* const* xp, float* const* yp, int nr, int add) { gm_tile(P, kc, xp, yp, nr, add, 6); }
typedef void (*gm_tile_fn)(const float*, int, const float* const*, float* const*, int, int);
static const gm_tile_fn gm_tiles[GM_MR + 1] = {NULL, gm_tile1, gm_tile2, gm_tile3, gm_tile4, gm_tile5, gm_tile6};
static const gm_tile_fn gm_tiles_n[GM_MR + 1] = {NULL, gm_tile_n1, gm_tile_n2, gm_tile_n3, gm_tile_n4, gm_tile_n5, gm_tile_n6};

/* Token blocks: with every row block's panel decoded first (per 256 columns), a block of GM_TB tokens (their 256
   columns, 96 KB) stays in L2 while all of the thread's row panels pass over it, instead of every token being
   streamed again from L3 for each 16 rows (KINDI_FNCPU_GEMM_TB=96; default 0, no blocking: measured flat). */
static int gm_tb = -1;
static int gm_t4 = 1;           /* K4 panels decoded straight into the transposed layout (gm_decode_t4; 0: rows) */
static void rows_gemm(const fc_tensor* T, int e, int r0, int r1, int nt, const float* x, long xs, const int* xi,
                      float* y, long ys, const int* yi, int acc) {
    static __thread float *tmp, *Pall; static __thread size_t pcap;
    if (gm_tb < 0) { const char* v = getenv("KINDI_FNCPU_GEMM_TB"); gm_tb = v && *v ? atoi(v) : 0; }
    if (!tmp) tmp = aligned_alloc(64, sizeof(float) * 16 * GM_KC);
    const int nrb = (r1 - r0 + 15) / 16;
    if (pcap < (size_t)nrb) { free(Pall); Pall = aligned_alloc(64, sizeof(float) * 16 * GM_KC * (size_t)nrb); pcap = nrb; }
    const int C = T->C, tb = gm_tb > 0 ? gm_tb : nt;
    for (int k0 = 0; k0 < C; k0 += GM_KC) {
        const int kc = C - k0 < GM_KC ? C - k0 : GM_KC;
        for (int b = 0; b < nrb; b++) {
            const int rb = r0 + 16 * b, nr = r1 - rb < 16 ? r1 - rb : 16;
            if (T->kind == 4 && gm_t4) gm_decode_t4(T, e, rb, nr, k0 / 32, kc / 32, Pall + (size_t)b * 16 * GM_KC);
            else {
                gm_decode(T, e, rb, nr, k0 / 32, kc / 32, tmp);
                gm_transpose(tmp, kc, kc, Pall + (size_t)b * 16 * GM_KC);
            }
        }
        for (int tb0 = 0; tb0 < nt; tb0 += tb) {
            const int tn = nt - tb0 < tb ? nt - tb0 : tb;
            const int ntile = (tn + GM_MR - 1) / GM_MR;      /* (tiles of near-equal size: 20 = 5+5+5+5) */
            for (int b = 0; b < nrb; b++) {
                const int rb = r0 + 16 * b, nr = r1 - rb < 16 ? r1 - rb : 16;
                const float* P = Pall + (size_t)b * 16 * GM_KC;
                const gm_tile_fn* tiles = nr <= 8 ? gm_tiles_n : gm_tiles;
                for (int ti = 0, t0 = tb0; ti < ntile; ti++) {
                    const int mt = (tn * (ti + 1)) / ntile - (tn * ti) / ntile;
                    const float* xp[GM_MR]; float* yp[GM_MR];
                    for (int i = 0; i < mt; i++) {
                        const int tk = t0 + i;
                        xp[i] = x + (size_t)(xi ? xi[tk] : tk) * xs + k0;
                        yp[i] = y + (size_t)(yi ? yi[tk] : tk) * ys + rb;
                    }
                    tiles[mt](P, kc, xp, yp, nr, acc || k0 > 0);
                    t0 += mt;
                }
            }
        }
    }
}

/* short matrices (a few rows: the mixers' inject, the DeltaNet gates, the indexer key): each thread decodes every
   row once (fp32, thread-local) and takes a range of tokens, so no thread works on a 1-row slice */
static void rows_short(const fc_tensor* T, int e, int t0, int t1, const float* x, long xs, float* y, long ys) {
    static __thread float* W; static __thread size_t cap;
    static __thread float* tmp;
    const int R = T->R, C = T->C;
    if (t0 >= t1) return;
    if (cap < (size_t)R * C) { free(W); W = aligned_alloc(64, sizeof(float) * (size_t)((R + 15) / 16 * 16) * C); cap = (size_t)R * C; }
    if (!tmp) tmp = aligned_alloc(64, sizeof(float) * 16 * GM_KC);
    for (int r = 0; r < R; r += 16) {
        const int nr = R - r < 16 ? R - r : 16;
        for (int k0 = 0; k0 < C; k0 += GM_KC) {
            const int kc = C - k0 < GM_KC ? C - k0 : GM_KC;
            gm_decode(T, e, r, nr, k0 / 32, kc / 32, tmp);
            for (int i = 0; i < nr; i++) memcpy(W + (size_t)(r + i) * C + k0, tmp + (size_t)i * kc, sizeof(float) * kc);
        }
    }
    for (int t = t0; t < t1; t++) {
        const float* xt = x + (size_t)t * xs;
        for (int r = 0; r < R; r++) {
            const float* w = W + (size_t)r * C;
            __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
            for (int k = 0; k < C; k += 32) {
                a = _mm256_fmadd_ps(_mm256_load_ps(w + k), _mm256_loadu_ps(xt + k), a);
                b = _mm256_fmadd_ps(_mm256_load_ps(w + k + 8), _mm256_loadu_ps(xt + k + 8), b);
                c = _mm256_fmadd_ps(_mm256_load_ps(w + k + 16), _mm256_loadu_ps(xt + k + 16), c);
                d = _mm256_fmadd_ps(_mm256_load_ps(w + k + 24), _mm256_loadu_ps(xt + k + 24), d);
            }
            y[(size_t)t * ys + r] = hsum8(_mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)));
        }
    }
}

/* tests: y [nt][R] = W_e x (fp32 x [nt][C]) through the GEMM path on the calling thread */
int fc_gemm_st(fc_tensor* T, int e, int nt, const float* x, float* y) {
    if (nt < 1 || T->kind == 32) return -1;
    rows_gemm(T, e, 0, T->R, nt, x, T->C, NULL, y, T->R, NULL, 0);
    return 0;
}

/* thread t's slice of R rows among n threads (pairs for K4) */
static inline void slice(const fc_tensor* T, int t, int n, int* r0, int* r1) {
    const int g = T->kind == 4 ? 2 : 1, units = (T->R + g - 1) / g;   /* (kind 32: rows) */
    const long a = (long)units * t / n, b = (long)units * (t + 1) / n;
    *r0 = (int)(a * g); *r1 = (int)(b * g) > T->R ? T->R : (int)(b * g);
}

/* ------------------------------------------------------------------ pool jobs */
typedef struct {
    fc_tensor* T; const uint8_t* src; int rc;
    int nt; const float* x; float* y; int e;
    int8_t* q; float* dx; float* fx;
} job_t;

static void job_repack(int t, int nact, void* arg) {
    job_t* J = arg;
    bind_here(1);
    for (int e = 0; e < J->T->E; e++) {
        int r0, r1; slice(J->T, t, nact, &r0, &r1);
        if (r0 < r1 && repack_rows(J->T, J->src, e, r0, r1) < 0) J->rc = -2;
    }
    bind_here(0);
}

/* repack a tensor from its file bytes (all experts) on the pool: each thread its own slice */
fc_tensor* fc_repack(int ty, const uint8_t* src, int R, int C, int E) {
    fc_tensor* T = fc_new(ty, R, C, E);
    if (!T) return NULL;
    job_t J = {.T = T, .src = src};
    moe_pool_run(job_repack, &J);
    if (J.rc < 0) { fprintf(stderr, "fncpu: repack of type %d failed (%d)\n", ty, J.rc); return NULL; }
    return T;
}
/* a float matrix kept as fp32 (the router: its picks decide which experts run, so it gets full precision):
   kind 32, rows of C floats in codes; products take fp32 inputs (fc_rows_f32) */
fc_tensor* fc_exact(int ty, const uint8_t* src, int R, int C) {
    if (ty != T_BF16 && ty != T_F16 && ty != T_F32) return NULL;
    fc_tensor* T = calloc(1, sizeof(*T));
    T->R = R; T->C = C; T->E = 1; T->src = ty; T->kind = 32; T->Rp = R; T->nb = C / 32;
    if (ty == T_BF16) {                          /* kept as bf16 (half the bytes; widened in the kernel) */
        T->per16 = 1;
        T->codes = big_alloc(2 * (size_t)R * C);
        T->bytes = 2 * (size_t)R * C;
        memcpy(T->codes, src, 2 * (size_t)R * C);
        return T;
    }
    T->codes = big_alloc(sizeof(float) * (size_t)R * C);
    T->bytes = sizeof(float) * (size_t)R * C;
    float* w = (float*)T->codes;
    for (size_t i = 0; i < (size_t)R * C; i++) w[i] = src_float(ty, src, (long)i);
    return T;
}
static void rows_f32(const fc_tensor* T, int r0, int r1, int nt, const float* x, float* y, long ys) {
    for (int r = r0; r < r1; r++)
        for (int t = 0; t < nt; t++) {
            __m256 s = _mm256_setzero_ps();
            const float* xt = x + (size_t)t * T->C;
            if (T->per16) {                           /* bf16 rows */
                const uint16_t* wr = (const uint16_t*)T->codes + (size_t)r * T->C;
                for (int i = 0; i < T->C; i += 8) {
                    const __m256 wv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(wr + i))), 16));
                    s = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xt + i), s);
                }
            } else {
                const float* wr = (const float*)T->codes + (size_t)r * T->C;
                for (int i = 0; i < T->C; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(wr + i), _mm256_loadu_ps(xt + i), s);
            }
            y[t * ys + r] = hsum8(s);
        }
}

size_t fc_bytes(const fc_tensor* T) { return T->bytes; }
int fc_kind(const fc_tensor* T) { return T->kind; }

static int8_t* g_q; static float *g_dx, *g_fx; static size_t g_cap;
static void act_buf(size_t n) {
    if (n <= g_cap) return;
    free(g_q); free(g_dx); free(g_fx);
    g_q = aligned_alloc(64, (n + 63) / 64 * 64); g_dx = malloc(sizeof(float) * (n / 32 + 1)); g_fx = malloc(sizeof(float) * (n / 32 + 1));
    g_cap = n;
}

static void job_mv(int t, int nact, void* arg) {
    job_t* J = arg;
    int r0, r1; slice(J->T, t, nact, &r0, &r1);
    rows_any(J->T, J->e, r0, r1, J->nt, J->q, J->dx, J->fx, J->y, J->T->R);
}

/* y [nt][R] = W_e x for nt tokens x [nt][C] (nt <= FC_TMAX), on the pool */
int fc_matvec(fc_tensor* T, int e, int nt, const float* x, float* y) {
    if (nt < 1 || nt > FC_TMAX) return -1;
    act_buf((size_t)nt * T->C);
    for (int t = 0; t < nt; t++) quant_act(x + (size_t)t * T->C, T->C, g_q + (size_t)t * T->C, g_dx + t * (T->C / 32), g_fx + t * (T->C / 32));
    job_t J = {.T = T, .nt = nt, .y = y, .e = e, .q = g_q, .dx = g_dx, .fx = g_fx};
    moe_pool_run(job_mv, &J);
    return 0;
}

/* the same on the calling thread only (tests) */
int fc_matvec_st(fc_tensor* T, int e, int nt, const float* x, float* y) {
    if (nt < 1 || nt > FC_TMAX) return -1;
    act_buf((size_t)nt * T->C);
    for (int t = 0; t < nt; t++) quant_act(x + (size_t)t * T->C, T->C, g_q + (size_t)t * T->C, g_dx + t * (T->C / 32), g_fx + t * (T->C / 32));
    rows_any(T, e, 0, T->R, nt, g_q, g_dx, g_fx, y, T->R);
    return 0;
}

/* bench: one product over nt tokens on the pool, by the GEMM path (gemm 1) or the int8 kernels; seconds per pass */
typedef struct { fc_tensor* T; int nt, gemm, reps; const float* x; float* y; } pbench_t;
static void job_pbench(int t, int nact, void* arg) {
    pbench_t* B = arg;
    int r0, r1; slice(B->T, t, nact, &r0, &r1);
    for (int k = 0; k < B->reps; k++) {
        if (B->gemm) rows_gemm(B->T, 0, r0, r1, B->nt, B->x, B->T->C, NULL, B->y, B->T->R, NULL, 0);
        else rows_big(B->T, 0, r0, r1, B->nt, g_q, g_dx, g_fx, B->y, B->T->R);
        moe_pool_barrier();
    }
}
double fc_prefill_bench(fc_tensor* T, int nt, int gemm, int reps, const float* x, float* y) {
    act_buf((size_t)nt * T->C);
    for (int t = 0; t < nt; t++) quant_act(x + (size_t)t * T->C, T->C, g_q + (size_t)t * T->C, g_dx + t * (T->C / 32), g_fx + t * (T->C / 32));
    pbench_t B = {T, nt, gemm, reps, x, y};
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    moe_pool_run(job_pbench, &B);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return ((b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec)) / reps;
}

/* bench: the routed experts of one layer for nt tokens (ids [nt][nsel]), GEMM path or int8 kernels, on the pool;
   seconds per pass. Gate/up then down, as fnmodel.c runs them (h = silu(g) * u, no weights). */
typedef struct { fc_tensor *g, *u, *d; int nt, nsel, gemm, reps, nu; const float* x; float *h, *y;
                 int *ue, *un, *up0, *pl; int8_t* hq; float *hdx, *hfx; } ebench_t;
static void job_ebench(int t, int nact, void* arg) {
    ebench_t* B = arg;
    const int HDm = B->g->C, FFm = B->g->R;
    float* go = malloc(sizeof(float) * (size_t)B->nt * FFm); float* gu = malloc(sizeof(float) * (size_t)B->nt * FFm);
    int* xi = malloc(sizeof(int) * B->nt); int* yi = malloc(sizeof(int) * B->nt);
    int8_t* gq = aligned_alloc(64, (size_t)B->nt * HDm); float* gdx = malloc(sizeof(float) * B->nt * HDm / 32);
    float* gfx = malloc(sizeof(float) * B->nt * HDm / 32); float* gy = malloc(sizeof(float) * (size_t)B->nt * HDm);
    for (int rep = 0; rep < B->reps; rep++) {
        int r0, r1; slice(B->g, t, nact, &r0, &r1);
        for (int u = 0; u < B->nu; u++) {
            const int e = B->ue[u], m = B->un[u]; const int* pl = B->pl + B->up0[u];
            for (int i = 0; i < m; i++) xi[i] = pl[i] / B->nsel;
            if (B->gemm) {
                rows_gemm(B->g, e, r0, r1, m, B->x, HDm, xi, go, FFm, NULL, 0);
                rows_gemm(B->u, e, r0, r1, m, B->x, HDm, xi, gu, FFm, NULL, 0);
            } else {
                for (int i = 0; i < m; i++) {
                    memcpy(gq + (size_t)i * HDm, g_q + (size_t)xi[i] * HDm, HDm);
                    memcpy(gdx + (size_t)i * (HDm / 32), g_dx + (size_t)xi[i] * (HDm / 32), sizeof(float) * HDm / 32);
                    memcpy(gfx + (size_t)i * (HDm / 32), g_fx + (size_t)xi[i] * (HDm / 32), sizeof(float) * HDm / 32);
                }
                rows_big(B->g, e, r0, r1, m, gq, gdx, gfx, go, FFm);
                rows_big(B->u, e, r0, r1, m, gq, gdx, gfx, gu, FFm);
            }
            for (int i = 0; i < m; i++) {
                float* h = B->h + (size_t)pl[i] * FFm;
                for (int r = r0; r < r1; r++) { const float v = go[(size_t)i * FFm + r]; h[r] = v / (1.f + expf(-v)) * gu[(size_t)i * FFm + r]; }
            }
        }
        moe_pool_barrier();
        if (!B->gemm) for (int pr = t; pr < B->nt * B->nsel; pr += nact)
            quant_act(B->h + (size_t)pr * FFm, FFm, B->hq + (size_t)pr * FFm, B->hdx + (size_t)pr * (FFm / 32), B->hfx + (size_t)pr * (FFm / 32));
        moe_pool_barrier();
        slice(B->d, t, nact, &r0, &r1);
        for (int u = 0; u < B->nu; u++) {
            const int e = B->ue[u], m = B->un[u]; const int* pl = B->pl + B->up0[u];
            for (int i = 0; i < m; i++) yi[i] = pl[i] / B->nsel;
            if (B->gemm) rows_gemm(B->d, e, r0, r1, m, B->h, FFm, pl, B->y, HDm, yi, 1);
            else {
                for (int i = 0; i < m; i++) {
                    memcpy(gq + (size_t)i * FFm, B->hq + (size_t)pl[i] * FFm, FFm);
                    memcpy(gdx + (size_t)i * (FFm / 32), B->hdx + (size_t)pl[i] * (FFm / 32), sizeof(float) * FFm / 32);
                    memcpy(gfx + (size_t)i * (FFm / 32), B->hfx + (size_t)pl[i] * (FFm / 32), sizeof(float) * FFm / 32);
                }
                rows_big(B->d, e, r0, r1, m, gq, gdx, gfx, gy, HDm);
                for (int i = 0; i < m; i++) for (int r = r0; r < r1; r++) B->y[(size_t)yi[i] * HDm + r] += gy[(size_t)i * HDm + r];
            }
        }
        moe_pool_barrier();
    }
    free(go); free(gu); free(xi); free(yi); free(gq); free(gdx); free(gfx); free(gy);
}
double fc_expert_bench(fc_tensor* g, fc_tensor* u, fc_tensor* d, int nt, int nsel, const int* ids, int gemm, int reps,
                       const float* x, float* y) {
    const int HDm = g->C, FFm = g->R, NEm = g->E, P = nt * nsel;
    act_buf((size_t)nt * HDm);
    for (int t = 0; t < nt; t++) quant_act(x + (size_t)t * HDm, HDm, g_q + (size_t)t * HDm, g_dx + t * (HDm / 32), g_fx + t * (HDm / 32));
    ebench_t B = {g, u, d, nt, nsel, gemm, reps, 0, x, malloc(sizeof(float) * (size_t)P * FFm), y};
    B.ue = malloc(sizeof(int) * NEm); B.un = calloc(NEm, sizeof(int)); B.up0 = malloc(sizeof(int) * NEm); B.pl = malloc(sizeof(int) * P);
    B.hq = aligned_alloc(64, (size_t)P * FFm); B.hdx = malloc(sizeof(float) * P * FFm / 32); B.hfx = malloc(sizeof(float) * P * FFm / 32);
    int* where = malloc(sizeof(int) * NEm);
    for (int e = 0; e < NEm; e++) where[e] = -1;
    for (int i = 0; i < P; i++) { const int e = ids[i]; if (where[e] < 0) { where[e] = B.nu; B.ue[B.nu] = e; B.un[B.nu] = 0; B.nu++; } B.un[where[e]]++; }
    int o = 0; for (int k = 0; k < B.nu; k++) { B.up0[k] = o; o += B.un[k]; B.un[k] = 0; }
    for (int i = 0; i < P; i++) { const int k = where[ids[i]]; B.pl[B.up0[k] + B.un[k]++] = i; }
    memset(y, 0, sizeof(float) * (size_t)nt * HDm);
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    moe_pool_run(job_ebench, &B);
    clock_gettime(CLOCK_MONOTONIC, &b);
    free(B.h); free(B.ue); free(B.un); free(B.up0); free(B.pl); free(B.hq); free(B.hdx); free(B.hfx); free(where);
    return ((b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec)) / reps;
}

/* ------------------------------------------------------------------ bench: a list of products as one pool job
   (each product: tensor, expert; one token), reps times; returns seconds per pass */
typedef struct { int n; fc_tensor** T; const int* e; int reps; } bench_t;
static void job_bench(int t, int nact, void* arg) {
    bench_t* B = arg;
    static float ybuf[1 << 20];
    for (int k = 0; k < B->reps; k++)
        for (int i = 0; i < B->n; i++) {
            fc_tensor* T = B->T[i];
            int r0, r1; slice(T, t, nact, &r0, &r1);
            rows_any(T, B->e[i], r0, r1, 1, g_q, g_dx, g_fx, ybuf + (T->R <= (1 << 20) ? 0 : 0), 0);
            moe_pool_barrier();
        }
}
double fc_bench(int n, fc_tensor** T, const int* e, int reps) {
    int cmax = 0;
    for (int i = 0; i < n; i++) if (T[i]->C > cmax) cmax = T[i]->C;
    act_buf(cmax);
    float* x = malloc(sizeof(float) * cmax);
    for (int i = 0; i < cmax; i++) x[i] = sinf(i * 0.37f);
    quant_act(x, cmax, g_q, g_dx, g_fx);
    free(x);
    bench_t B = {n, T, e, reps};
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    moe_pool_run(job_bench, &B);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return ((b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec)) / reps;
}

/* ------------------------------------------------------------------ the routed experts of one MoE layer for the card
   engine (kindi/qwen4exp_gpu.py through qx_cpu.c's mailbox step): the value rows [T][nsel][K] of the (token, slot)
   pairs not marked in skip, unweighted (the card weights and sums them). Each expert's weights are read once for all
   its pairs (the union over the window). */
#define FC_MAXL 64
static fc_tensor* FL[FC_MAXL][3];
int fc_set_layer(int l, fc_tensor* g, fc_tensor* u, fc_tensor* d) {
    if (l < 0 || l >= FC_MAXL) return -1;
    FL[l][0] = g; FL[l][1] = u; FL[l][2] = d;
    return 0;
}
int fc_has_layer(int l) { return l >= 0 && l < FC_MAXL && FL[l][0] != NULL; }

#define FC_PMAX 640                     /* pairs per call (64 tokens x 10) */
typedef struct {
    int l, T, nsel, K, F; const float* x; const int* ids; const uint8_t* skip; float* rows;
    int nu, ue[FC_PMAX], un[FC_PMAX], up0[FC_PMAX];       /* experts, pairs per expert, first index into plist */
    int plist[FC_PMAX];
    int8_t* xq; float *xdx, *xfx;      /* [T][K] */
    float* h;                           /* [pairs][F] */
    int8_t* hq; float *hdx, *hfx;      /* [pairs][F] */
} parts_t;
static parts_t PJ;

static void job_parts(int t, int n, void* arg) {
    parts_t* P = arg;
    const fc_tensor *G = FL[P->l][0], *U = FL[P->l][1], *D = FL[P->l][2];
    const int K = P->K, F = P->F;
    /* the inputs, one token per thread in turn */
    for (int k = t; k < P->T; k += n)
        quant_act(P->x + (size_t)k * K, K, P->xq + (size_t)k * K, P->xdx + (size_t)k * (K / 32), P->xfx + (size_t)k * (K / 32));
    moe_pool_barrier();
    int8_t qg[FC_TMAX * 4096] __attribute__((aligned(32)));
    float dxg[FC_TMAX * 128], fxg[FC_TMAX * 128];
    static __thread float og[FC_TMAX * 2048], ou[FC_TMAX * 2048];
    int r0, r1; slice(G, t, n, &r0, &r1);
    for (int u = 0; u < P->nu; u++) {
        const int e = P->ue[u];
        for (int c0 = 0; c0 < P->un[u]; c0 += FC_TMAX) {
            const int m = P->un[u] - c0 < FC_TMAX ? P->un[u] - c0 : FC_TMAX;
            for (int i = 0; i < m; i++) {
                const int k = P->plist[P->up0[u] + c0 + i] / P->nsel;
                memcpy(qg + i * K, P->xq + (size_t)k * K, K);
                memcpy(dxg + i * (K / 32), P->xdx + (size_t)k * (K / 32), sizeof(float) * (K / 32));
                memcpy(fxg + i * (K / 32), P->xfx + (size_t)k * (K / 32), sizeof(float) * (K / 32));
            }
            rows_any(G, e, r0, r1, m, qg, dxg, fxg, og, F);
            rows_any(U, e, r0, r1, m, qg, dxg, fxg, ou, F);
            for (int i = 0; i < m; i++) {
                float* hh = P->h + (size_t)P->plist[P->up0[u] + c0 + i] * F;
                for (int r = r0; r < r1; r++) { const float g = og[i * F + r]; hh[r] = g / (1.f + expf(-g)) * ou[i * F + r]; }
            }
        }
    }
    moe_pool_barrier();
    for (int i = t; i < P->T * P->nsel; i += n) {
        if (P->skip[i]) continue;
        quant_act(P->h + (size_t)i * F, F, P->hq + (size_t)i * F, P->hdx + (size_t)i * (F / 32), P->hfx + (size_t)i * (F / 32));
    }
    moe_pool_barrier();
    int8_t qd[FC_TMAX * 2048] __attribute__((aligned(32)));
    float dxd[FC_TMAX * 64], fxd[FC_TMAX * 64];
    static __thread float part[FC_TMAX * 4096];
    slice(D, t, n, &r0, &r1);
    for (int u = 0; u < P->nu; u++) {
        const int e = P->ue[u];
        for (int c0 = 0; c0 < P->un[u]; c0 += FC_TMAX) {
            const int m = P->un[u] - c0 < FC_TMAX ? P->un[u] - c0 : FC_TMAX;
            for (int i = 0; i < m; i++) {
                const int pr = P->plist[P->up0[u] + c0 + i];
                memcpy(qd + i * F, P->hq + (size_t)pr * F, F);
                memcpy(dxd + i * (F / 32), P->hdx + (size_t)pr * (F / 32), sizeof(float) * (F / 32));
                memcpy(fxd + i * (F / 32), P->hfx + (size_t)pr * (F / 32), sizeof(float) * (F / 32));
            }
            rows_any(D, e, r0, r1, m, qd, dxd, fxd, part, K);
            for (int i = 0; i < m; i++) {
                float* dst = P->rows + (size_t)P->plist[P->up0[u] + c0 + i] * K;
                memcpy(dst + r0, part + (size_t)i * K + r0, sizeof(float) * (r1 - r0));
            }
        }
    }
}

/* seconds, calls, pairs and distinct experts in fc_forward_parts since the last fc_stats() */
static double FS_sec; static long long FS_calls, FS_pairs, FS_exp;
void fc_stats(double* out) {
    out[0] = FS_sec; out[1] = (double)FS_calls; out[2] = (double)FS_pairs; out[3] = (double)FS_exp;
    FS_sec = 0; FS_calls = FS_pairs = FS_exp = 0;
}
int fc_fits(int T, int nsel) { return T >= 1 && T * nsel <= FC_PMAX; }

int fc_forward_parts(int l, int T, const float* x, int nsel, const int* ids, const uint8_t* skip, float* rows) {
    if (!fc_has_layer(l) || T < 1 || T * nsel > FC_PMAX) return -1;
    struct timespec ta, tb;
    clock_gettime(CLOCK_MONOTONIC, &ta);
    parts_t* P = &PJ;
    const fc_tensor* G = FL[l][0];
    const int K = G->C, F = G->R;
    if (K > 4096 || F > 2048) return -1;
    P->l = l; P->T = T; P->nsel = nsel; P->K = K; P->F = F; P->x = x; P->ids = ids; P->skip = skip; P->rows = rows;
    static size_t cap;
    const size_t need = (size_t)T * nsel * F + (size_t)T * K;
    if (need > cap) {
        free(P->xq); free(P->xdx); free(P->xfx); free(P->h); free(P->hq); free(P->hdx); free(P->hfx);
        P->xq = aligned_alloc(64, ((size_t)64 * K + 63) / 64 * 64); P->xdx = malloc(sizeof(float) * 64 * K / 32);
        P->xfx = malloc(sizeof(float) * 64 * K / 32);
        P->h = malloc(sizeof(float) * FC_PMAX * F); P->hq = aligned_alloc(64, ((size_t)FC_PMAX * F + 63) / 64 * 64);
        P->hdx = malloc(sizeof(float) * FC_PMAX * F / 32); P->hfx = malloc(sizeof(float) * FC_PMAX * F / 32);
        cap = (size_t)FC_PMAX * F + (size_t)64 * K;
    }
    /* the union: experts in order of first use, their pairs in order */
    P->nu = 0;
    int cnt[FC_PMAX];
    for (int i = 0; i < T * nsel; i++) {
        if (skip && skip[i]) continue;
        const int e = ids[i];
        int u = 0;
        while (u < P->nu && P->ue[u] != e) u++;
        if (u == P->nu) { P->ue[u] = e; P->un[u] = 0; P->nu++; }
        P->un[u]++;
    }
    int o = 0;
    for (int u = 0; u < P->nu; u++) { P->up0[u] = o; cnt[u] = 0; o += P->un[u]; }
    for (int i = 0; i < T * nsel; i++) {
        if (skip && skip[i]) continue;
        int u = 0;
        while (P->ue[u] != ids[i]) u++;
        P->plist[P->up0[u] + cnt[u]++] = i;
    }
    if (P->nu) moe_pool_run(job_parts, P);
    clock_gettime(CLOCK_MONOTONIC, &tb);
    FS_sec += (tb.tv_sec - ta.tv_sec) + 1e-9 * (tb.tv_nsec - ta.tv_nsec); FS_calls++; FS_pairs += o; FS_exp += P->nu;
    return 0;
}
