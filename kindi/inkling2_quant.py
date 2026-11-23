"""GGUF re-quantizer for Inkling-2-Mini (NVFP4 lossless-class & sub-4-bit IQ2_M / IQ2_XXS).

Presets:
  - nvfp4  (default): blk.20..23 -> NVFP4, blk.6..17 loop_ctrl F32 -> Q8_0 (14.44 GiB).
  - iq2_m  (8 GB full-VRAM fit): blk.6..17 gate_up -> IQ2_XS_128, all other experts -> IQ2_XXS_128,
            attn_{q,k,v,o,gate,rel} & output.weight -> Q4_0, loop_ctrl -> Q8_0 (~7.55 GiB disk, 7.10 GB VRAM).
  - iq2_xxs (max compression): all 3,072 experts -> IQ2_XXS_128, attn & output -> Q4_0, loop_ctrl -> Q8_0.

Usage:
    python3 -m kindi.inkling2_quant [--preset nvfp4|iq2_m|iq2_xxs] [--imatrix imatrix.bin] <src.gguf> <dst.gguf>
"""
import argparse
import ctypes
import hashlib
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

TY_F32 = 0
TY_Q4_0 = 2
TY_Q8 = 8
TY_Q5K = 13
TY_BF16 = 30
TY_NVFP4 = 40
TY_IQ2XXS_128 = 116
TY_IQ2XS_128 = 117

BLOCK_BYTES = {
    TY_F32: 4,
    TY_Q4_0: 18,
    TY_Q8: 34,
    TY_Q5K: 176,
    TY_BF16: 2,
    TY_NVFP4: 36,
    TY_IQ2XXS_128: 34,
    TY_IQ2XS_128: 38,
}
BLOCK_ELEMS = {
    TY_F32: 1,
    TY_Q4_0: 32,
    TY_Q8: 32,
    TY_Q5K: 256,
    TY_BF16: 1,
    TY_NVFP4: 64,
    TY_IQ2XXS_128: 128,
    TY_IQ2XS_128: 128,
}

_SCALAR_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

_C_SRC = r"""
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "inkling2_iq2_tables.h"

static const int8_t FP4_VALS[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float g_f16_tab[65536];
static float g_ue4m3_tab[256];
static int g_init_done = 0;

typedef struct {
    int grid_size;
    uint64_t* grid;
    int* map;
    uint16_t* neighbours;
} IkIq2Search;

static IkIq2Search g_iq2_xxs = {0, NULL, NULL, NULL};
static IkIq2Search g_iq2_xs  = {0, NULL, NULL, NULL};

static int iq2_cmp_pair(const void* a, const void* b) {
    const int* l = (const int*)a;
    const int* r = (const int*)b;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

static void iq2_build_search(IkIq2Search* s, int grid_size, const uint16_t* kgrid) {
    if (s->grid) return;
    const int kmap_size = 43692;
    const int nwant = 2;
    uint64_t* grid = (uint64_t*)malloc((size_t)grid_size * sizeof(uint64_t));
    for (int k = 0; k < grid_size; k++) {
        int8_t* pos = (int8_t*)(grid + k);
        for (int i = 0; i < 8; i++) pos[i] = (int8_t)(2 * ((kgrid[k] >> (2 * i)) & 3) + 1);
    }
    int* kmap = (int*)malloc((size_t)kmap_size * sizeof(int));
    for (int i = 0; i < kmap_size; i++) kmap[i] = -1;
    for (int i = 0; i < grid_size; i++) {
        const uint8_t* a = (const uint8_t*)(grid + i);
        uint16_t idx = 0;
        for (int k = 0; k < 8; k++) idx |= (uint16_t)(((a[k] - 1) / 2) << (2 * k));
        kmap[idx] = i;
    }
    int* n_per_i = (int*)calloc((size_t)kmap_size, sizeof(int));
    int* dist2 = (int*)malloc((size_t)(2 * grid_size) * sizeof(int));
    int num_nbr = 0, num_miss = 0;
    for (int i = 0; i < kmap_size; i++) {
        if (kmap[i] >= 0) continue;
        num_miss++;
        int8_t pos[8];
        for (int k = 0; k < 8; k++) pos[k] = (int8_t)(2 * ((i >> (2 * k)) & 3) + 1);
        for (int j = 0; j < grid_size; j++) {
            const int8_t* pg = (const int8_t*)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) { int d = pg[k] - pos[k]; d2 += d * d; }
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, (size_t)grid_size, 2 * sizeof(int), iq2_cmp_pair);
        int d2cur = dist2[0], n = 0, nhave = 1;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2cur) {
                if (nhave == nwant) break;
                d2cur = dist2[2 * j];
                nhave++;
            }
            n++;
        }
        n_per_i[i] = n;
        num_nbr += n;
    }
    uint16_t* nbrs = (uint16_t*)malloc((size_t)(num_nbr + num_miss) * sizeof(uint16_t));
    int counter = 0;
    for (int i = 0; i < kmap_size; i++) {
        if (kmap[i] >= 0) continue;
        int8_t pos[8];
        for (int k = 0; k < 8; k++) pos[k] = (int8_t)(2 * ((i >> (2 * k)) & 3) + 1);
        for (int j = 0; j < grid_size; j++) {
            const int8_t* pg = (const int8_t*)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) { int d = pg[k] - pos[k]; d2 += d * d; }
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, (size_t)grid_size, 2 * sizeof(int), iq2_cmp_pair);
        kmap[i] = -(counter + 1);
        int start = counter++;
        int d2cur = dist2[0], n = 0, nhave = 1;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2cur) {
                if (nhave == nwant) break;
                d2cur = dist2[2 * j];
                nhave++;
            }
            nbrs[counter++] = (uint16_t)dist2[2 * j + 1];
            n++;
        }
        nbrs[start] = (uint16_t)n;
    }
    free(dist2);
    free(n_per_i);
    s->grid_size = grid_size;
    s->grid = grid;
    s->map = kmap;
    s->neighbours = nbrs;
}

static float fp16_to_f32_soft(uint16_t h) {
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

static inline uint16_t f32_to_fp16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = (u >> 16) & 0x8000u;
    int32_t e = (int32_t)((u >> 23) & 0xFFu) - 127;
    uint32_t m = u & 0x7FFFFFu;
    if (e > 15) return (uint16_t)(s | 0x7BFFu);
    if (e < -24) return (uint16_t)s;
    if (e < -14) {
        uint32_t sig = (0x800000u | m) >> (uint32_t)(-14 - e + 13);
        return (uint16_t)(s | sig);
    }
    uint32_t r = (m + 0x0FFFu + ((m >> 13) & 1u)) >> 13;
    if (r >= 0x400u) { r = 0; e++; if (e > 15) return (uint16_t)(s | 0x7BFFu); }
    return (uint16_t)(s | ((uint32_t)(e + 15) << 10) | r);
}

static inline uint8_t fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits; memcpy(&bits, &x, 4);
    int fp32_exp = ((bits >> 23) & 0xFF) - 127;
    int fp32_man = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        int man = (int)(x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        return man < 1 ? 0 : (uint8_t)man;
    }
    if (ue4m3_exp >= 15) return 0x7E;
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) return 0x7E;
    }
    return (uint8_t)((ue4m3_exp << 3) | ue4m3_man);
}

void ikq_init(void) {
    if (g_init_done) return;
    for (int i = 0; i < 65536; i++) g_f16_tab[i] = fp16_to_f32_soft((uint16_t)i);
    for (int i = 0; i < 256; i++) {
        if (i == 0 || i == 0x7F) { g_ue4m3_tab[i] = 0.0f; continue; }
        int exp = (i >> 3) & 0xF, man = i & 0x7;
        float raw = (exp == 0) ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
        g_ue4m3_tab[i] = raw * 0.5f;
    }
    iq2_build_search(&g_iq2_xxs, 256, ik_kgrid_2bit_256);
    iq2_build_search(&g_iq2_xs,  512, ik_kgrid_2bit_512);
    g_init_done = 1;
}

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

static void dequant_q5k(const uint8_t* src, float* y, int64_t n) {
    int64_t nb = n / 256;
    for (int64_t i = 0; i < nb; i++, src += 176) {
        uint16_t hd, hmin;
        memcpy(&hd, src, 2); memcpy(&hmin, src + 2, 2);
        const float d = g_f16_tab[hd], mn = g_f16_tab[hmin];
        const uint8_t* scales = src + 4;
        const uint8_t* qh = src + 16;
        const uint8_t* ql = src + 48;
        int is = 0; uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc, m1 = mn * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }
}

static void dequant_q8(const uint8_t* src, float* y, int64_t n) {
    int64_t nb = n / 32;
    for (int64_t i = 0; i < nb; i++, src += 34) {
        uint16_t hd; memcpy(&hd, src, 2);
        const float d = g_f16_tab[hd];
        const int8_t* qs = (const int8_t*)(src + 2);
        for (int j = 0; j < 32; j++) y[i * 32 + j] = qs[j] * d;
    }
}

static void dequant_nvfp4_scaled(const uint8_t* src, float* y, int64_t n, float exp_scale) {
    int64_t nb = n / 64;
    for (int64_t b = 0; b < nb; b++, src += 36) {
        for (int s = 0; s < 4; s++) {
            float d = g_ue4m3_tab[src[s]] * exp_scale;
            const uint8_t* q = src + 4 + s * 8;
            for (int k = 0; k < 8; k++) {
                uint8_t bq = q[k];
                y[b * 64 + s * 16 + k]     = FP4_VALS[bq & 15] * d;
                y[b * 64 + s * 16 + k + 8] = FP4_VALS[bq >> 4] * d;
            }
        }
    }
}

static inline int n_round(float x) { return (int)llroundf(x); }

static float make_qp_quants(int n, int nmax, const float* x, uint8_t* L, const float* qw) {
    float mx = 0.f;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    if (mx < 1e-15f) { memset(L, 0, (size_t)n); return 0.f; }
    float iscale = (float)nmax / mx;
    for (int i = 0; i < n; i++) L[i] = (uint8_t)n_round(iscale * x[i]);
    float scale = 1.f / iscale, best_mse = 0.f;
    for (int i = 0; i < n; i++) { float d = x[i] - scale * L[i]; best_mse += qw[i] * d * d; }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float is_is = (0.1f * is + (float)nmax) / mx, sc_is = 1.f / is_is, mse = 0.f;
        for (int i = 0; i < n; i++) {
            int l = n_round(is_is * x[i]);
            if (l > nmax) l = nmax;
            float d = x[i] - sc_is * l;
            mse += qw[i] * d * d;
        }
        if (mse < best_mse) { best_mse = mse; iscale = is_is; }
    }
    float sumlx = 0.f, suml2 = 0.f;
    for (int i = 0; i < n; i++) {
        int l = n_round(iscale * x[i]);
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        sumlx += qw[i] * x[i] * l;
        suml2 += qw[i] * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int ch = 0;
        for (int i = 0; i < n; i++) {
            float w = qw[i], slx = sumlx - w * x[i] * L[i], sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0.f && sl2 > 0.f) {
                int nl = n_round(x[i] * sl2 / slx);
                if (nl > nmax) nl = nmax;
                if (nl < 0) nl = 0;
                if (nl != L[i]) {
                    slx += w * x[i] * nl; sl2 += w * nl * nl;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (uint8_t)nl; sumlx = slx; suml2 = sl2; ch++;
                    }
                }
            }
        }
        if (!ch) break;
    }
    return suml2 > 0.f ? sumlx / suml2 : 0.f;
}

#define IQ2_CACHE_MAX 512
typedef struct {
    int pattern, last_best;
    double sum_w, sum_wx, sum_wxx, sum_w1, sum_w1x, sum_w1xx;
    double lin_term[8][8], quad_term[8][8];
    double lin_coef[IQ2_CACHE_MAX], quad_coef[IQ2_CACHE_MAX];
} Iq2NCache;

static void iq2_ncache_init(Iq2NCache* c, const float* xval, const float* weight) {
    c->pattern = -1;
    c->sum_w = c->sum_wx = c->sum_wxx = c->sum_w1 = c->sum_w1x = c->sum_w1xx = 0.0;
    for (int i = 0; i < 8; i++) {
        double w = weight[i], x = xval[i], aw = fabs(w), ax = fabs(x);
        c->sum_w += aw; c->sum_wx += aw * ax; c->sum_wxx += aw * ax * ax;
        c->sum_w1 += 1.0 + aw; c->sum_w1x += (1.0 + aw) * ax; c->sum_w1xx += (1.0 + aw) * ax * ax;
        for (int q = 0; q < 8; q++) { c->lin_term[i][q] = 2.0 * w * x * q; c->quad_term[i][q] = w * q * q; }
    }
}

static int iq2_find_nbr(const uint16_t* nbrs, const uint64_t* grid, const float* xval, const float* w, float scale, int8_t* L) {
    int nn = nbrs[0], best_i = -1;
    float best_d2 = FLT_MAX;
    for (int j = 1; j <= nn; j++) {
        const int8_t* pg = (const int8_t*)(grid + nbrs[j]);
        float d2 = 0.f;
        for (int i = 0; i < 8; i++) { float diff = scale * pg[i] - xval[i]; d2 += w[i] * diff * diff; }
        if (d2 < best_d2) { best_d2 = d2; best_i = nbrs[j]; }
    }
    const int8_t* pg = (const int8_t*)(grid + best_i);
    for (int i = 0; i < 8; i++) L[i] = (int8_t)((pg[i] - 1) / 2);
    return best_i;
}

static int iq2_find_nbr_cached(Iq2NCache* c, int pat, const uint16_t* nbrs, const uint64_t* grid,
                               const float* xval, const float* w, float scale, int8_t* L) {
    int nn = nbrs[0];
    double sc = scale, ml = 7.0 * sc;
    double sz = ml * ml * c->sum_w + 2.0 * ml * c->sum_wx + c->sum_wxx;
    double szn = (1.0 + ml) * (1.0 + ml) * c->sum_w1 + 2.0 * (1.0 + ml) * c->sum_w1x + c->sum_w1xx;
    double tol = (ldexp(sz, -18) + ldexp(szn, -100)) / sc;
    if (nn <= 0 || nn > IQ2_CACHE_MAX || !(sc > 0.0 && sz < ldexp(1.0, 100) && szn < ldexp(1.0, 100) && tol > 0.0))
        return iq2_find_nbr(nbrs, grid, xval, w, scale, L);
    if (c->pattern != pat) {
        for (int j = 0; j < nn; j++) {
            const int8_t* lv = (const int8_t*)(grid + nbrs[j + 1]);
            double lin = 0.0, quad = 0.0;
            for (int i = 0; i < 8; i++) { lin += c->lin_term[i][lv[i]]; quad += c->quad_term[i][lv[i]]; }
            c->lin_coef[j] = lin; c->quad_coef[j] = quad;
        }
        c->pattern = pat; c->last_best = 0;
    }
    double scores[IQ2_CACHE_MAX];
    for (int j = 0; j < nn; j++) scores[j] = sc * c->quad_coef[j] - c->lin_coef[j];
    double b0 = scores[c->last_best];
    int bi = c->last_best;
    for (int j = 0; j < nn; j++) if (scores[j] < b0) { b0 = scores[j]; bi = j; }
    int n_ties = 0;
    for (int j = 0; j < nn; j++) if (scores[j] <= b0 + tol) n_ties++;
    if (n_ties > 1) return iq2_find_nbr(nbrs, grid, xval, w, scale, L);
    c->last_best = bi;
    int gi = nbrs[bi + 1];
    const int8_t* pg = (const int8_t*)(grid + gi);
    for (int i = 0; i < 8; i++) L[i] = (int8_t)((pg[i] - 1) / 2);
    return gi;
}

static void quant_iq2xxs128_row(const float* x, uint8_t* dst, int64_t n, const float* qw_in,
                                  double* sig2, double* err2) {
    const uint64_t* kgrid = g_iq2_xxs.grid;
    const int*      kmap  = g_iq2_xxs.map;
    const uint16_t* knbr  = g_iq2_xxs.neighbours;
    const int kMaxQ = 3;
    int64_t nbl = n / 128;
    float scales[4], weight[32], xval[32], waux[32];
    int8_t L[32], Laux[32], Laux_prev[32];
    uint8_t block_signs[4];
    uint32_t q2[8];
    Iq2NCache nc[4];

    for (int64_t ibl = 0; ibl < nbl; ibl++) {
        uint8_t* out = dst + ibl * 34;
        memset(out, 0, 34);
        memset(q2, 0, sizeof(q2));
        float max_scale = 0.f;
        const float* xbl = x + 128 * ibl;
        const float* qbl = qw_in + 128 * ibl;
        float sumx2 = 0.f;
        for (int i = 0; i < 128; i++) sumx2 += xbl[i] * xbl[i];
        float sigma2 = sumx2 * (1.0f / 128.0f);

        for (int ib = 0; ib < 4; ib++) {
            const float* xb = xbl + 32 * ib;
            const float* qw = qbl + 32 * ib;
            for (int i = 0; i < 32; i++) {
                float wv = (qw[i] > 1e-12f ? qw[i] : 1e-12f) * sqrtf(sigma2 + xb[i] * xb[i]);
                weight[i] = wv; waux[i] = sqrtf(wv);
            }
            for (int k = 0; k < 4; k++) {
                int nflip = 0; uint8_t s = 0;
                for (int i = 0; i < 8; i++) {
                    if (xb[8 * k + i] >= 0.f) xval[8 * k + i] = xb[8 * k + i];
                    else { xval[8 * k + i] = -xb[8 * k + i]; nflip++; s |= (1u << i); }
                }
                if (nflip & 1) {
                    int imin = 0; float mn = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for (int i = 1; i < 8; i++) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < mn) { mn = ax; imin = i; }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1u << imin);
                }
                block_signs[k] = s & 127u;
            }
            float mx = xval[0];
            for (int i = 1; i < 32; i++) if (xval[i] > mx) mx = xval[i];
            if (mx < 1e-15f) { scales[ib] = 0.f; memset(L, 0, 32); continue; }
            float scale = make_qp_quants(32, kMaxQ + 1, xval, (uint8_t*)L, weight);
            float eff_max = scale * kMaxQ;
            if (eff_max <= 0.f) { scales[ib] = 0.f; memset(L, 0, 32); continue; }
            float best = 0.f;
            for (int k = 0; k < 4; k++) iq2_ncache_init(&nc[k], xval + 8 * k, waux + 8 * k);
            for (int is = -6; is <= 6; is++) {
                float id = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
                float this_sc = 1.0f / id;
                for (int k = 0; k < 4; k++) {
                    for (int i = 0; i < 8; i++) {
                        int l = n_round(0.5f * (id * xval[8 * k + i] - 1.0f));
                        if (l < 0) l = 0; if (l > kMaxQ - 1) l = kMaxQ - 1;
                        Laux[8 * k + i] = (int8_t)l;
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; i++) u |= (uint16_t)(Laux[8 * k + i] << (2 * i));
                    if (kmap[u] < 0) {
                        const uint16_t* nbrs = knbr - kmap[u] - 1;
                        iq2_find_nbr_cached(&nc[k], u, nbrs, kgrid, xval + 8 * k, waux + 8 * k, this_sc, Laux + 8 * k);
                    }
                }
                if (is > -6 && memcmp(Laux, Laux_prev, 32) == 0) continue;
                memcpy(Laux_prev, Laux, 32);
                float sumqx = 0.f, sumq2 = 0.f;
                for (int i = 0; i < 32; i++) { float q = (float)(2 * Laux[i] + 1); sumqx += weight[i] * xval[i] * q; sumq2 += weight[i] * q * q; }
                if (sumq2 > 0.f && sumqx * sumqx > best * sumq2) { scale = sumqx / sumq2; best = scale * sumqx; memcpy(L, Laux, 32); }
            }
            if (scale > 0.f) {
                float id = 1.0f / scale;
                for (int k = 0; k < 4; k++) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; i++) {
                        int l = n_round(0.5f * (id * xval[8 * k + i] - 1.0f));
                        if (l < 0) l = 0; if (l > kMaxQ - 1) l = kMaxQ - 1;
                        u |= (uint16_t)(l << (2 * i));
                    }
                    int gi = kmap[u];
                    if (gi < 0) gi = iq2_find_nbr_cached(&nc[k], u, knbr - kmap[u] - 1, kgrid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                    const int8_t* pg = (const int8_t*)(kgrid + gi);
                    for (int i = 0; i < 8; i++) L[8 * k + i] = (int8_t)((pg[i] - 1) / 2);
                }
                float sumqx = 0.f, sumq2 = 0.f;
                for (int i = 0; i < 32; i++) { float q = (float)(2 * L[i] + 1); sumqx += weight[i] * xval[i] * q; sumq2 += weight[i] * q * q; }
                if (sumq2 > 0.f) scale = sumqx / sumq2;
            }
            if (scale < 0.f) { scale = -scale; for (int k = 0; k < 4; k++) block_signs[k] = (~block_signs[k]) & 127u; }
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
                int gi = kmap[u];
                q2[2 * ib + 0] |= ((uint32_t)gi << (8 * k));
                q2[2 * ib + 1] |= ((uint32_t)block_signs[k] << (7 * k));
            }
            scales[ib] = scale;
            if (scale > max_scale) max_scale = scale;
        }
        if (max_scale > 0.f) {
            float d = max_scale / 31.0f;
            uint16_t hd = f32_to_fp16(d);
            memcpy(out, &hd, 2);
            float id = 1.0f / d, df = g_f16_tab[hd] * 0.25f;
            for (int ib = 0; ib < 4; ib++) {
                int l = n_round(0.5f * (id * scales[ib] - 1.0f));
                if (l < 0) l = 0; if (l > 15) l = 15;
                q2[2 * ib + 1] |= ((uint32_t)l << 28);
            }
            memcpy(out + 2, q2, 32);
            for (int ib = 0; ib < 4; ib++) {
                float db = df * (0.5f + (float)(q2[2 * ib + 1] >> 28));
                for (int l = 0; l < 4; l++) {
                    const uint8_t* gp = (const uint8_t*)(ik_iq2xxs_grid + ((q2[2 * ib] >> (8 * l)) & 255u));
                    uint8_t sg = ik_ksigns_iq2xs[(q2[2 * ib + 1] >> (7 * l)) & 127u];
                    for (int j = 0; j < 8; j++) {
                        double w0 = xbl[32 * ib + 8 * l + j];
                        double r0 = (double)db * (double)gp[j] * ((sg & (1u << j)) ? -1.0 : 1.0);
                        double dfv = w0 - r0;
                        *sig2 += w0 * w0; *err2 += dfv * dfv;
                    }
                }
            }
        } else {
            for (int i = 0; i < 128; i++) { double w0 = xbl[i]; *sig2 += w0 * w0; *err2 += w0 * w0; }
        }
    }
}

static void quant_iq2xs128_row(const float* x, uint8_t* dst, int64_t n, const float* qw_in,
                                 double* sig2, double* err2) {
    const uint64_t* kgrid = g_iq2_xs.grid;
    const int*      kmap  = g_iq2_xs.map;
    const uint16_t* knbr  = g_iq2_xs.neighbours;
    const int kMaxQ = 3;
    int64_t nbl = n / 128;
    float scales[8], weight[16], xval[16], waux[16];
    int8_t L[16], Laux[16], Laux_prev[16];
    int is_og[2], is_oga[2], is_ogp[2];
    uint8_t block_signs[2], sc_bytes[4];
    uint16_t q2[16];
    Iq2NCache nc[2];

    for (int64_t ibl = 0; ibl < nbl; ibl++) {
        uint8_t* out = dst + ibl * 38;
        memset(out, 0, 38);
        memset(q2, 0, sizeof(q2));
        memset(sc_bytes, 0, sizeof(sc_bytes));
        float max_scale = 0.f;
        const float* xbl = x + 128 * ibl;
        const float* qbl = qw_in + 128 * ibl;
        float sumx2 = 0.f;
        for (int i = 0; i < 128; i++) sumx2 += xbl[i] * xbl[i];
        float sigma2 = sumx2 * (1.0f / 128.0f);

        for (int ib = 0; ib < 8; ib++) {
            const float* xb = xbl + 16 * ib;
            const float* qw = qbl + 16 * ib;
            for (int i = 0; i < 16; i++) {
                float wv = (qw[i] > 1e-12f ? qw[i] : 1e-12f) * sqrtf(sigma2 + xb[i] * xb[i]);
                weight[i] = wv; waux[i] = sqrtf(wv);
            }
            for (int k = 0; k < 2; k++) {
                int nflip = 0; uint8_t s = 0;
                for (int i = 0; i < 8; i++) {
                    if (xb[8 * k + i] >= 0.f) xval[8 * k + i] = xb[8 * k + i];
                    else { xval[8 * k + i] = -xb[8 * k + i]; nflip++; s |= (1u << i); }
                }
                if (nflip & 1) {
                    int imin = 0; float mn = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for (int i = 1; i < 8; i++) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < mn) { mn = ax; imin = i; }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1u << imin);
                }
                block_signs[k] = s & 127u;
            }
            float mx = xval[0];
            for (int i = 1; i < 16; i++) if (xval[i] > mx) mx = xval[i];
            memset(L, 0, 16);
            if (mx < 1e-15f) { scales[ib] = 0.f; continue; }
            float best = 0.f, scale = mx / (float)(2 * kMaxQ - 1);
            is_og[0] = is_og[1] = 1;
            for (int k = 0; k < 2; k++) iq2_ncache_init(&nc[k], xval + 8 * k, waux + 8 * k);
            for (int is = -9; is <= 9; is++) {
                float id = (2 * kMaxQ - 1 + is * 0.1f) / mx;
                float this_sc = 1.0f / id;
                for (int k = 0; k < 2; k++) {
                    for (int i = 0; i < 8; i++) {
                        int l = n_round(0.5f * (id * xval[8 * k + i] - 1.0f));
                        if (l < 0) l = 0; if (l > kMaxQ - 1) l = kMaxQ - 1;
                        Laux[8 * k + i] = (int8_t)l;
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; i++) u |= (uint16_t)(Laux[8 * k + i] << (2 * i));
                    is_oga[k] = 1;
                    if (kmap[u] < 0) {
                        is_oga[k] = 0;
                        iq2_find_nbr_cached(&nc[k], u, knbr - kmap[u] - 1, kgrid, xval + 8 * k, waux + 8 * k, this_sc, Laux + 8 * k);
                    }
                }
                if (is > -9 && memcmp(Laux, Laux_prev, 16) == 0 && is_oga[0] == is_ogp[0] && is_oga[1] == is_ogp[1]) continue;
                memcpy(Laux_prev, Laux, 16); is_ogp[0] = is_oga[0]; is_ogp[1] = is_oga[1];
                float sumqx = 0.f, sumq2 = 0.f;
                for (int i = 0; i < 16; i++) { float q = (float)(2 * Laux[i] + 1); sumqx += weight[i] * xval[i] * q; sumq2 += weight[i] * q * q; }
                if (sumq2 > 0.f && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2; best = scale * sumqx;
                    memcpy(L, Laux, 16); is_og[0] = is_oga[0]; is_og[1] = is_oga[1];
                }
            }
            if ((!is_og[0] || !is_og[1]) && scale > 0.f) {
                float id = 1.0f / scale;
                for (int k = 0; k < 2; k++) {
                    if (is_og[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; i++) {
                        int l = n_round(0.5f * (id * xval[8 * k + i] - 1.0f));
                        if (l < 0) l = 0; if (l > kMaxQ - 1) l = kMaxQ - 1;
                        u |= (uint16_t)(l << (2 * i));
                        L[8 * k + i] = (int8_t)l;
                    }
                    if (kmap[u] < 0) iq2_find_nbr_cached(&nc[k], u, knbr - kmap[u] - 1, kgrid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
                float sumqx = 0.f, sumq2 = 0.f;
                for (int i = 0; i < 16; i++) { float q = (float)(2 * L[i] + 1); sumqx += weight[i] * xval[i] * q; sumq2 += weight[i] * q * q; }
                if (sumq2 > 0.f) scale = sumqx / sumq2;
            }
            if (scale < 0.f) { scale = -scale; for (int k = 0; k < 2; k++) block_signs[k] = (~block_signs[k]) & 127u; }
            for (int k = 0; k < 2; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
                int gi = kmap[u];
                q2[2 * ib + k] = (uint16_t)(gi | (block_signs[k] << 9));
            }
            scales[ib] = scale;
            if (scale > max_scale) max_scale = scale;
        }
        if (max_scale > 0.f) {
            float d = max_scale / 31.0f;
            uint16_t hd = f32_to_fp16(d);
            memcpy(out, &hd, 2);
            float id = 1.0f / d, df = g_f16_tab[hd] * 0.25f;
            for (int ib = 0; ib < 8; ib++) {
                int l = n_round(0.5f * (id * scales[ib] - 1.0f));
                if (l < 0) l = 0; if (l > 15) l = 15;
                if ((ib & 1) == 0) sc_bytes[ib >> 1] = (uint8_t)l;
                else sc_bytes[ib >> 1] |= (uint8_t)(l << 4);
            }
            memcpy(out + 2, q2, 32);
            memcpy(out + 34, sc_bytes, 4);
            for (int ib = 0; ib < 8; ib++) {
                int lsc = (ib & 1) ? (sc_bytes[ib >> 1] >> 4) : (sc_bytes[ib >> 1] & 15);
                float db = df * (0.5f + (float)lsc);
                for (int k = 0; k < 2; k++) {
                    uint16_t q = q2[2 * ib + k];
                    const uint8_t* gp = (const uint8_t*)(ik_iq2xs_grid + (q & 511u));
                    uint8_t sg = ik_ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; j++) {
                        double w0 = xbl[16 * ib + 8 * k + j];
                        double r0 = (double)db * (double)gp[j] * ((sg & (1u << j)) ? -1.0 : 1.0);
                        double dfv = w0 - r0;
                        *sig2 += w0 * w0; *err2 += dfv * dfv;
                    }
                }
            }
        } else {
            for (int i = 0; i < 128; i++) { double w0 = xbl[i]; *sig2 += w0 * w0; *err2 += w0 * w0; }
        }
    }
}

static void quant_q4_0_row(const float* x, uint8_t* dst, int64_t n, const float* qw_in,
                           double* sig2, double* err2) {
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const float* xb = x + b * 32;
        const float* qw = qw_in + b * 32;
        uint8_t* out = dst + b * 18;
        float amax = 0.f, smax = 0.f;
        for (int j = 0; j < 32; j++) {
            float av = fabsf(xb[j]);
            if (av > amax) { amax = av; smax = xb[j]; }
        }
        float d = smax / -8.0f;
        uint8_t q[32];
        for (int it = 0; it < 3; it++) {
            float id = d != 0.f ? (1.0f / d) : 0.f;
            float sumqx = 0.f, sumq2 = 0.f;
            for (int j = 0; j < 32; j++) {
                int v = n_round(xb[j] * id) + 8;
                if (v < 0) v = 0; if (v > 15) v = 15;
                q[j] = (uint8_t)v;
                int qi = v - 8;
                float w = qw[j] > 1e-12f ? qw[j] : 1e-12f;
                sumqx += w * (float)qi * xb[j];
                sumq2 += w * (float)(qi * qi);
            }
            if (it < 2 && sumq2 > 0.f) {
                float nd = sumqx / sumq2;
                if (nd != 0.f) d = nd;
            }
        }
        uint16_t hd = f32_to_fp16(d);
        float dq = g_f16_tab[hd];
        float id = dq != 0.f ? (1.0f / dq) : 0.f;
        memcpy(out, &hd, 2);
        for (int j = 0; j < 16; j++) {
            int q0 = n_round(xb[j] * id) + 8;
            int q1 = n_round(xb[j + 16] * id) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            out[2 + j] = (uint8_t)(q0 | (q1 << 4));
            double w0 = xb[j], w1 = xb[j + 16];
            double r0 = (double)(q0 - 8) * dq, r1 = (double)(q1 - 8) * dq;
            *sig2 += w0 * w0 + w1 * w1;
            double d0 = w0 - r0, d1 = w1 - r1;
            *err2 += d0 * d0 + d1 * d1;
        }
    }
}

typedef struct {
    const uint8_t* src;
    const float* exp_scales;
    const float* qw_all;
    int src_ty, dst_ty;
    int64_t rows, cols, qw_stride;
    int n_exp;
    uint8_t* dst;
    double* sig_acc;
    double* err_acc;
    int tid, nth;
} UniJob;

static void* mat_q4_worker(void* arg) {
    UniJob* j = (UniJob*)arg;
    int64_t cols = j->cols;
    size_t src_rb = (j->src_ty == 8) ? (size_t)(cols / 32) * 34u : (size_t)cols * 4u;
    size_t dst_rb = (size_t)(cols / 32) * 18u;
    float* buf = (float*)malloc((size_t)cols * sizeof(float));
    float* ones = NULL;
    const float* qw = j->qw_all;
    if (!qw) {
        ones = (float*)malloc((size_t)cols * sizeof(float));
        for (int64_t i = 0; i < cols; i++) ones[i] = 1.0f;
        qw = ones;
    }
    double ts = 0.0, te = 0.0;
    for (int64_t r = j->tid; r < j->rows; r += j->nth) {
        const uint8_t* sr = j->src + (size_t)r * src_rb;
        if (j->src_ty == 8) dequant_q8(sr, buf, cols);
        else memcpy(buf, sr, (size_t)cols * 4u);
        quant_q4_0_row(buf, j->dst + (size_t)r * dst_rb, cols, qw, &ts, &te);
    }
    free(buf); free(ones);
    j->sig_acc[j->tid] = ts; j->err_acc[j->tid] = te;
    return NULL;
}

void ikq_quantize_mat_q4_0(const uint8_t* src, int src_ty, int64_t rows, int64_t cols, const float* qw,
                           uint8_t* dst, int n_threads, double* out_sig2, double* out_err2) {
    ikq_init();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 16) n_threads = 16;
    if (n_threads > rows) n_threads = (int)rows;
    UniJob jobs[16]; pthread_t th[16]; double sa[16], ea[16];
    for (int t = 0; t < n_threads; t++) {
        jobs[t] = (UniJob){src, NULL, qw, src_ty, 2, rows, cols, cols, 0, dst, sa, ea, t, n_threads};
        pthread_create(&th[t], NULL, mat_q4_worker, &jobs[t]);
    }
    double ts = 0.0, te = 0.0;
    for (int t = 0; t < n_threads; t++) { pthread_join(th[t], NULL); ts += sa[t]; te += ea[t]; }
    *out_sig2 = ts; *out_err2 = te;
}

static void* iq2_exp_worker(void* arg) {
    UniJob* j = (UniJob*)arg;
    int64_t rows = j->rows, cols = j->cols, n = rows * cols;
    size_t src_eb = (j->src_ty == 13) ? ((size_t)(n / 256) * 176u) :
                    (j->src_ty == 40) ? ((size_t)(n / 64) * 36u) :
                                        ((size_t)(n / 32) * 34u);
    size_t dst_rb = (j->dst_ty == 117) ? ((size_t)(cols / 128) * 38u) : ((size_t)(cols / 128) * 34u);
    size_t dst_eb = (size_t)rows * dst_rb;
    float* buf = (float*)malloc((size_t)n * sizeof(float));
    float* ones = NULL;
    if (!j->qw_all) {
        ones = (float*)malloc((size_t)cols * sizeof(float));
        for (int64_t i = 0; i < cols; i++) ones[i] = 1.0f;
    }
    double ts = 0.0, te = 0.0;
    for (int e = j->tid; e < j->n_exp; e += j->nth) {
        const uint8_t* se = j->src + (size_t)e * src_eb;
        float esc = j->exp_scales ? j->exp_scales[e] : 1.0f;
        if (j->src_ty == 13) dequant_q5k(se, buf, n);
        else if (j->src_ty == 40) dequant_nvfp4_scaled(se, buf, n, esc);
        else dequant_q8(se, buf, n);
        const float* qw = j->qw_all ? (j->qw_all + (size_t)e * j->qw_stride) : ones;
        uint8_t* de = j->dst + (size_t)e * dst_eb;
        for (int64_t r = 0; r < rows; r++) {
            if (j->dst_ty == 117) quant_iq2xs128_row(buf + r * cols, de + (size_t)r * dst_rb, cols, qw, &ts, &te);
            else                  quant_iq2xxs128_row(buf + r * cols, de + (size_t)r * dst_rb, cols, qw, &ts, &te);
        }
    }
    free(buf); free(ones);
    j->sig_acc[j->tid] = ts; j->err_acc[j->tid] = te;
    return NULL;
}

void ikq_quantize_experts_iq2(const uint8_t* src, int src_ty, const float* exp_scales,
                              int64_t rows, int64_t cols, int n_exp, int dst_ty,
                              const float* qw_all, int64_t qw_stride,
                              uint8_t* dst, int n_threads, double* out_sig2, double* out_err2) {
    ikq_init();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 16) n_threads = 16;
    if (n_threads > n_exp) n_threads = n_exp;
    UniJob jobs[16]; pthread_t th[16]; double sa[16], ea[16];
    for (int t = 0; t < n_threads; t++) {
        jobs[t] = (UniJob){src, exp_scales, qw_all, src_ty, dst_ty, rows, cols, qw_stride, n_exp, dst, sa, ea, t, n_threads};
        pthread_create(&th[t], NULL, iq2_exp_worker, &jobs[t]);
    }
    double ts = 0.0, te = 0.0;
    for (int t = 0; t < n_threads; t++) { pthread_join(th[t], NULL); ts += sa[t]; te += ea[t]; }
    *out_sig2 = ts; *out_err2 = te;
}

static inline uint8_t best_fp4(float x, float d) {
    int best = 0;
    float best_err = fabsf(FP4_VALS[0] * d - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(FP4_VALS[i] * d - x);
        if (err < best_err) { best = i; best_err = err; }
    }
    return (uint8_t)best;
}

typedef struct {
    const uint8_t* src;
    int src_ty;
    int64_t ne_exp;
    int n_exp;
    uint8_t* dst;
    float* scales;
    double* sig_acc;
    double* err_acc;
    int tid;
    int nth;
} Nvfp4Job;

static void* nvfp4_worker(void* arg) {
    Nvfp4Job* j = (Nvfp4Job*)arg;
    int64_t n = j->ne_exp;
    int64_t nb = n / 64;
    size_t src_stride = (j->src_ty == 13) ? ((size_t)(n / 256) * 176u) : ((size_t)(n / 32) * 34u);
    size_t dst_stride = (size_t)nb * 36u;
    float* buf = (float*)malloc((size_t)n * sizeof(float));
    double s_sum = 0.0, e_sum = 0.0;

    for (int e = j->tid; e < j->n_exp; e += j->nth) {
        const uint8_t* se = j->src + (size_t)e * src_stride;
        if (j->src_ty == 13) dequant_q5k(se, buf, n);
        else dequant_q8(se, buf, n);

        float amax = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            float av = fabsf(buf[i]);
            if (av > amax) amax = av;
        }
        float scale = (amax > 0.0f) ? (amax / (6.0f * 448.0f)) : 1.0f;
        float inv_scale = 1.0f / scale;
        j->scales[e] = scale;

        uint8_t* de = j->dst + (size_t)e * dst_stride;
        for (int64_t b = 0; b < nb; b++) {
            uint8_t* out = de + (size_t)b * 36u;
            for (int s = 0; s < 4; s++) {
                const float* xb = buf + b * 64 + s * 16;
                float sb_amax = 0.0f;
                for (int k = 0; k < 16; k++) {
                    float av = fabsf(xb[k] * inv_scale);
                    if (av > sb_amax) sb_amax = av;
                }
                uint8_t ue = fp32_to_ue4m3(sb_amax / 6.0f);
                out[s] = ue;
                float d = g_ue4m3_tab[ue];
                for (int k = 0; k < 8; k++) {
                    float v0 = xb[k] * inv_scale;
                    float v1 = xb[k + 8] * inv_scale;
                    uint8_t q0 = best_fp4(v0, d);
                    uint8_t q1 = best_fp4(v1, d);
                    out[4 + s * 8 + k] = (uint8_t)(q0 | (q1 << 4));
                    float w0 = xb[k], r0 = FP4_VALS[q0] * d * scale;
                    float w1 = xb[k + 8], r1 = FP4_VALS[q1] * d * scale;
                    s_sum += (double)w0 * w0 + (double)w1 * w1;
                    double d0 = (double)w0 - r0, d1 = (double)w1 - r1;
                    e_sum += d0 * d0 + d1 * d1;
                }
            }
        }
    }
    free(buf);
    j->sig_acc[j->tid] = s_sum;
    j->err_acc[j->tid] = e_sum;
    return NULL;
}

void ikq_quantize_experts_nvfp4(const uint8_t* src, int src_ty, int64_t ne_exp, int n_exp,
                                uint8_t* dst, float* scales, int n_threads,
                                double* out_sig2, double* out_err2) {
    ikq_init();
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 16) n_threads = 16;
    if (n_threads > n_exp) n_threads = n_exp;
    Nvfp4Job jobs[16];
    pthread_t th[16];
    double sig_acc[16], err_acc[16];
    for (int t = 0; t < n_threads; t++) {
        jobs[t].src = src; jobs[t].src_ty = src_ty;
        jobs[t].ne_exp = ne_exp; jobs[t].n_exp = n_exp;
        jobs[t].dst = dst; jobs[t].scales = scales;
        jobs[t].sig_acc = sig_acc; jobs[t].err_acc = err_acc;
        jobs[t].tid = t; jobs[t].nth = n_threads;
        pthread_create(&th[t], NULL, nvfp4_worker, &jobs[t]);
    }
    double ts = 0.0, te = 0.0;
    for (int t = 0; t < n_threads; t++) {
        pthread_join(th[t], NULL);
        ts += sig_acc[t]; te += err_acc[t];
    }
    *out_sig2 = ts; *out_err2 = te;
}

void ikq_quantize_f32_to_q8(const float* src, int64_t n, uint8_t* dst,
                            double* out_sig2, double* out_err2) {
    ikq_init();
    int64_t nb = n / 32;
    double ts = 0.0, te = 0.0;
    for (int64_t b = 0; b < nb; b++) {
        const float* x = src + b * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float av = fabsf(x[j]);
            if (av > amax) amax = av;
        }
        float d = amax / 127.0f;
        uint16_t hd = f32_to_fp16(d);
        float dq = g_f16_tab[hd];
        float id = d > 0.0f ? (1.0f / d) : 0.0f;
        uint8_t* out = dst + b * 34;
        memcpy(out, &hd, 2);
        int8_t* qs = (int8_t*)(out + 2);
        for (int j = 0; j < 32; j++) {
            int q = (int)llroundf(x[j] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[j] = (int8_t)q;
            double r = (double)q * dq, w = x[j], diff = w - r;
            ts += w * w; te += diff * diff;
        }
    }
    *out_sig2 = ts; *out_err2 = te;
}
"""


def _compile_helper():
    here = os.path.dirname(os.path.abspath(__file__))
    tbl_path = os.path.join(here, "inkling2_iq2_tables.h")
    tbl_bytes = open(tbl_path, "rb").read()
    key = hashlib.sha256(_C_SRC.encode("utf-8") + tbl_bytes).hexdigest()[:16]
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "kindi")
    os.makedirs(cache_dir, exist_ok=True)
    so_path = os.path.join(cache_dir, "inkling2_quant_%s.so" % key)
    if not os.path.exists(so_path):
        with tempfile.TemporaryDirectory() as td:
            c_path = os.path.join(td, "q.c")
            tmp_so = so_path + ".tmp.%d" % os.getpid()
            with open(c_path, "w", encoding="utf-8") as f:
                f.write(_C_SRC)
            subprocess.check_call([
                "gcc", "-O3", "-march=native", "-I" + here,
                "-shared", "-fPIC", "-pthread", c_path, "-o", tmp_so, "-lm",
            ])
            os.replace(tmp_so, so_path)
    lib = ctypes.CDLL(so_path)
    lib.ikq_quantize_experts_nvfp4.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.ikq_quantize_f32_to_q8.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.ikq_quantize_mat_q4_0.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.ikq_quantize_experts_iq2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    return lib


def _u(f, fmt):
    sz = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, f.read(sz))[0]


def _read_str(f):
    n = _u(f, "Q")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, ty):
    if ty == 8:
        return _read_str(f)
    if ty == 9:
        et, n = _u(f, "I"), _u(f, "Q")
        if et == 8:
            return [_read_str(f) for _ in range(n)]
        fmt = _SCALAR_FMT[et]
        sz = struct.calcsize("<" + fmt)
        return list(struct.unpack("<%d%s" % (n, fmt), f.read(sz * n)))
    return _u(f, _SCALAR_FMT[ty])


def _tensor_nbytes(ne, ty):
    elems = 1
    for d in ne:
        elems *= int(d)
    be = BLOCK_ELEMS.get(ty)
    if be is not None:
        assert ne[0] % be == 0, "ne[0]=%d not divisible by %d for ty=%d" % (ne[0], be, ty)
        return (elems // be) * BLOCK_BYTES[ty]
    # Fallback for any copied untouched ggml types via gguf table
    raise ValueError("unexpected type %d for newly computed tensor" % ty)


def _pack_tensor_info(name, ne, ty, off):
    nb = name.encode("utf-8")
    parts = [struct.pack("<Q", len(nb)), nb, struct.pack("<I", len(ne))]
    for d in ne:
        parts.append(struct.pack("<Q", int(d)))
    parts.append(struct.pack("<IQ", ty, off))
    return b"".join(parts)


def _pack_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _load_imatrix(path):
    """Load IKIMAT1 binary written by ik_imatrix_stop_and_save()."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != b"IKIMAT1\x00":
            raise ValueError("invalid imatrix magic %r in %s" % (magic, path))
        nl_phys, ne, d, qdim, ff = struct.unpack("<5i", f.read(20))
        if (nl_phys, ne, d, qdim, ff) != (24, 128, 2304, 3072, 1152):
            raise ValueError("unexpected imatrix dims %r" % ((nl_phys, ne, d, qdim, ff),))
        f.seek(nl_phys * 8 + nl_phys * ne * 8 + 8, 1)  # skip uint64 hit counts
        in0 = np.frombuffer(f.read(nl_phys * d * 4), dtype=np.float32).reshape(nl_phys, d).copy()
        in1 = np.frombuffer(f.read(nl_phys * d * 4), dtype=np.float32).reshape(nl_phys, d).copy()
        in2 = np.frombuffer(f.read(nl_phys * d * 4), dtype=np.float32).reshape(nl_phys, d).copy()
        attn = np.frombuffer(f.read(nl_phys * qdim * 4), dtype=np.float32).reshape(nl_phys, qdim).copy()
        head = np.frombuffer(f.read(d * 4), dtype=np.float32).copy()
        exp_gu = np.zeros((nl_phys, ne, d), dtype=np.float32)
        exp_dn = np.zeros((nl_phys, ne, ff), dtype=np.float32)
        for ip in range(nl_phys):
            exp_gu[ip] = np.frombuffer(f.read(ne * d * 4), dtype=np.float32).reshape(ne, d)
            exp_dn[ip] = np.frombuffer(f.read(ne * ff * 4), dtype=np.float32).reshape(ne, ff)
    return {
        "in0": in0, "in1": in1, "in2": in2,
        "attn": attn, "head": head,
        "exp_gu": exp_gu, "exp_dn": exp_dn,
    }


def _dense_imat(imat, name):
    if imat is None:
        return None
    if name == "output.weight":
        return imat["head"]
    parts = name.split(".")
    if len(parts) == 4 and parts[0] == "blk" and parts[3] == "weight":
        l = int(parts[1])
        suf = parts[2]
        if suf in ("attn_q", "attn_gate", "attn_rel"):
            return imat["in0"][l]
        if suf == "attn_k":
            return imat["in1"][l]
        if suf == "attn_v":
            return imat["in2"][l]
        if suf == "attn_output":
            return imat["attn"][l]
    return None


def quantize_gguf(src_path, dst_path, preset="nvfp4", imatrix_path=None, n_threads=None, log=print):
    """Quantize Inkling-2-Mini GGUF using preset 'nvfp4', 'iq2_m', or 'iq2_xxs'."""
    import math
    if preset not in ("nvfp4", "iq2_m", "iq2_xxs"):
        raise ValueError("unknown preset %r" % preset)
    if n_threads is None:
        n_threads = min(16, max(1, os.cpu_count() or 8))
    lib = _compile_helper()
    imat = _load_imatrix(imatrix_path) if imatrix_path else None
    if imat is not None:
        log("[quant] loaded imatrix from %s" % imatrix_path)

    with open(src_path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("%s is not a GGUF file" % src_path)
        version = _u(f, "I")
        if version != 3:
            raise ValueError("expected GGUF v3, got %d" % version)
        n_tensors, n_kv = _u(f, "Q"), _u(f, "Q")
        align = 32
        kv_chunks = []
        for _ in range(n_kv):
            kv_pos = f.tell()
            key = _read_str(f)
            vty = _u(f, "I")
            val = _read_value(f, vty)
            kv_end = f.tell()
            if key == "general.alignment":
                align = int(val)
            if key != "kindi.quant_preset":
                f.seek(kv_pos)
                kv_chunks.append(f.read(kv_end - kv_pos))
        kv_chunks.append(_pack_str("kindi.quant_preset") + struct.pack("<I", 8) + _pack_str(preset))
        out_n_kv = len(kv_chunks)
        raw_kv_bytes = b"".join(kv_chunks)
        f.seek(24)
        # Re-read kv header end safely after collecting chunks
        for _ in range(n_kv):
            _read_str(f)
            _read_value(f, _u(f, "I"))

        src_tensors = []
        for _ in range(n_tensors):
            name = _read_str(f)
            nd = _u(f, "I")
            ne = [int(_u(f, "Q")) for _ in range(nd)]
            ty = _u(f, "I")
            off = _u(f, "Q")
            src_tensors.append((name, ne, ty, off))
        pos = f.tell()
        src_data_start = (pos + align - 1) // align * align

        # Load source .scale tensors if present (needed when re-quantizing NVFP4 -> IQ2)
        src_scales = {}
        if preset in ("iq2_m", "iq2_xxs"):
            for name, ne, ty, off in src_tensors:
                if name.endswith("_exps.scale") and ty == TY_F32:
                    f.seek(src_data_start + off)
                    wname = name[:-6] + ".weight"
                    src_scales[wname] = np.frombuffer(f.read(ne[0] * 4), dtype=np.float32).copy()

    src_file_size = os.path.getsize(src_path)
    by_off = sorted(range(len(src_tensors)), key=lambda idx: src_tensors[idx][3])
    src_sizes = [0] * len(src_tensors)
    for k, idx in enumerate(by_off):
        cur_off = src_tensors[idx][3]
        nxt_off = src_tensors[by_off[k + 1]][3] if k + 1 < len(by_off) else (src_file_size - src_data_start)
        src_sizes[idx] = nxt_off - cur_off

    target_ctrls = {
        "blk.%d.loop_%s_ctrl_%s.weight" % (l, blk, mat)
        for l in range(6, 18) for blk in ("q", "k", "v", "ffn") for mat in ("down", "gain", "forget")
    }
    target_q4 = {
        "blk.%d.attn_%s.weight" % (l, kind)
        for l in range(24) for kind in ("q", "k", "v", "output", "gate", "rel")
    } | {"output.weight"}

    plan = []
    out_off = 0
    if preset == "nvfp4":
        target_exps = {
            "blk.%d.ffn_%s_exps.weight" % (l, kind)
            for l in range(20, 24) for kind in ("gate_up", "down")
        }
        for idx, (name, ne, ty, off) in enumerate(src_tensors):
            if name in target_exps:
                sname = name[:-7] + ".scale"
                sne = [ne[2]]
                snbytes = _tensor_nbytes(sne, TY_F32)
                out_off = (out_off + align - 1) // align * align
                plan.append(("scale", sname, sne, TY_F32, out_off, snbytes, idx))
                out_off += snbytes

                wnbytes = _tensor_nbytes(ne, TY_NVFP4)
                out_off = (out_off + align - 1) // align * align
                plan.append(("nvfp4", name, ne, TY_NVFP4, out_off, wnbytes, idx))
                out_off += wnbytes
            elif name in target_ctrls:
                if ty != TY_F32:
                    raise ValueError("expected F32 for %s, got %d" % (name, ty))
                wnbytes = _tensor_nbytes(ne, TY_Q8)
                out_off = (out_off + align - 1) // align * align
                plan.append(("q8", name, ne, TY_Q8, out_off, wnbytes, idx))
                out_off += wnbytes
            else:
                be = BLOCK_ELEMS.get(ty)
                nbytes = _tensor_nbytes(ne, ty) if be is not None else src_sizes[idx]
                out_off = (out_off + align - 1) // align * align
                plan.append(("copy", name, ne, ty, out_off, nbytes, idx))
                out_off += nbytes
    else:
        for idx, (name, ne, ty, off) in enumerate(src_tensors):
            if name.endswith("_exps.scale"):
                continue  # IQ2_*_128 uses native per-block scales; drop NVFP4 tensor-scale helpers
            if name.endswith("ffn_gate_up_exps.weight") or name.endswith("ffn_down_exps.weight"):
                l = int(name.split(".")[1])
                is_gu = name.endswith("ffn_gate_up_exps.weight")
                dst_ty = TY_IQ2XS_128 if (preset == "iq2_m" and is_gu and 6 <= l < 18) else TY_IQ2XXS_128
                wnbytes = _tensor_nbytes(ne, dst_ty)
                out_off = (out_off + align - 1) // align * align
                plan.append(("iq2", name, ne, dst_ty, out_off, wnbytes, idx))
                out_off += wnbytes
            elif name in target_q4:
                if ty not in (TY_Q8, TY_F32):
                    raise ValueError("expected Q8_0 or F32 for %s, got %d" % (name, ty))
                wnbytes = _tensor_nbytes(ne, TY_Q4_0)
                out_off = (out_off + align - 1) // align * align
                plan.append(("q4_0", name, ne, TY_Q4_0, out_off, wnbytes, idx))
                out_off += wnbytes
            elif name in target_ctrls and ty == TY_F32:
                wnbytes = _tensor_nbytes(ne, TY_Q8)
                out_off = (out_off + align - 1) // align * align
                plan.append(("q8", name, ne, TY_Q8, out_off, wnbytes, idx))
                out_off += wnbytes
            else:
                be = BLOCK_ELEMS.get(ty)
                nbytes = _tensor_nbytes(ne, ty) if be is not None else src_sizes[idx]
                out_off = (out_off + align - 1) // align * align
                plan.append(("copy", name, ne, ty, out_off, nbytes, idx))
                out_off += nbytes

    header_parts = [
        b"GGUF",
        struct.pack("<IQQ", 3, len(plan), out_n_kv),
        raw_kv_bytes,
    ]
    for _, name, ne, ty, off, _, _ in plan:
        header_parts.append(_pack_tensor_info(name, ne, ty, off))
    header = b"".join(header_parts)
    pad_hdr = (-len(header)) % align
    header += b"\x00" * pad_hdr
    dst_data_start = len(header)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp_dst = dst_path + ".tmp.%d" % os.getpid()

    exp_sig2 = {13: 0.0, 8: 0.0}
    exp_err2 = {13: 0.0, 8: 0.0}
    iq2_stats = {"xs_gu": [0.0, 0.0], "xxs_gu": [0.0, 0.0], "xxs_dn": [0.0, 0.0]}
    q4_stats = {"attn": [0.0, 0.0], "head": [0.0, 0.0]}
    ctrl_sig2, ctrl_err2 = 0.0, 0.0
    scale_cache = {}

    log("[quant] writing %s (%s, %d tensors, threads=%d)..." % (dst_path, preset, len(plan), n_threads))
    with open(src_path, "rb") as fin, open(tmp_dst, "wb") as fout:
        fout.write(header)
        chunk_sz = 16 << 20
        for kind, name, ne, ty, off, nbytes, src_idx in plan:
            target_pos = dst_data_start + off
            cur_pos = fout.tell()
            if target_pos > cur_pos:
                fout.write(b"\x00" * (target_pos - cur_pos))
            _, src_ne, src_ty, src_off = src_tensors[src_idx]

            if kind == "scale":
                fin.seek(src_data_start + src_off)
                w_src_bytes = _tensor_nbytes(src_ne, src_ty)
                src_buf = fin.read(w_src_bytes)
                assert len(src_buf) == w_src_bytes
                n_exp = src_ne[2]
                ne_exp = src_ne[0] * src_ne[1]
                w_nbytes = _tensor_nbytes(src_ne, TY_NVFP4)
                dst_w = (ctypes.c_uint8 * w_nbytes)()
                dst_s = (ctypes.c_float * n_exp)()
                c_src = (ctypes.c_uint8 * len(src_buf)).from_buffer_copy(src_buf)
                s2, e2 = ctypes.c_double(), ctypes.c_double()
                lib.ikq_quantize_experts_nvfp4(
                    ctypes.cast(c_src, ctypes.c_void_p), src_ty, ne_exp, n_exp,
                    ctypes.cast(dst_w, ctypes.c_void_p), ctypes.cast(dst_s, ctypes.c_void_p),
                    n_threads, ctypes.byref(s2), ctypes.byref(e2),
                )
                exp_sig2[src_ty] += s2.value
                exp_err2[src_ty] += e2.value
                sqnr = 10.0 * math.log10(s2.value / max(e2.value, 1e-30))
                w_name = src_tensors[src_idx][0]
                log("  [nvfp4] %-36s (%d->40)  %.1f MiB -> %.1f MiB  SQNR=%.2f dB" % (
                    w_name, src_ty, len(src_buf) / (1 << 20), w_nbytes / (1 << 20), sqnr))
                scale_cache[w_name] = bytes(dst_w)
                fout.write(bytes(dst_s))

            elif kind == "nvfp4":
                fout.write(scale_cache.pop(name))

            elif kind == "iq2":
                fin.seek(src_data_start + src_off)
                w_src_bytes = _tensor_nbytes(src_ne, src_ty)
                src_buf = fin.read(w_src_bytes)
                assert len(src_buf) == w_src_bytes
                cols, rows, n_exp = src_ne[0], src_ne[1], src_ne[2]
                c_src = (ctypes.c_uint8 * len(src_buf)).from_buffer_copy(src_buf)
                dst_w = (ctypes.c_uint8 * nbytes)()
                esc_arr = src_scales.get(name) if src_ty == TY_NVFP4 else None
                esc_ptr = esc_arr.ctypes.data_as(ctypes.c_void_p) if esc_arr is not None else None
                qw_ptr, qw_stride = None, cols
                if imat is not None:
                    l = int(name.split(".")[1])
                    qw_arr = imat["exp_gu"][l] if name.endswith("gate_up_exps.weight") else imat["exp_dn"][l]
                    qw_ptr = qw_arr.ctypes.data_as(ctypes.c_void_p)
                s2, e2 = ctypes.c_double(), ctypes.c_double()
                lib.ikq_quantize_experts_iq2(
                    ctypes.cast(c_src, ctypes.c_void_p), src_ty, esc_ptr,
                    rows, cols, n_exp, ty, qw_ptr, qw_stride,
                    ctypes.cast(dst_w, ctypes.c_void_p), n_threads,
                    ctypes.byref(s2), ctypes.byref(e2),
                )
                key_st = "xs_gu" if ty == TY_IQ2XS_128 else ("xxs_gu" if name.endswith("gate_up_exps.weight") else "xxs_dn")
                iq2_stats[key_st][0] += s2.value
                iq2_stats[key_st][1] += e2.value
                sqnr = 10.0 * math.log10(s2.value / max(e2.value, 1e-30))
                tag = "iq2xs" if ty == TY_IQ2XS_128 else "iq2xxs"
                log("  [%-6s] %-36s (%d->%d) %.1f MiB -> %.1f MiB  SQNR=%.2f dB" % (
                    tag, name, src_ty, ty, len(src_buf) / (1 << 20), nbytes / (1 << 20), sqnr))
                fout.write(bytes(dst_w))

            elif kind == "q4_0":
                fin.seek(src_data_start + src_off)
                w_src_bytes = _tensor_nbytes(src_ne, src_ty)
                src_buf = fin.read(w_src_bytes)
                assert len(src_buf) == w_src_bytes
                cols, rows = src_ne[0], src_ne[1]
                c_src = (ctypes.c_uint8 * len(src_buf)).from_buffer_copy(src_buf)
                dst_w = (ctypes.c_uint8 * nbytes)()
                qw_arr = _dense_imat(imat, name)
                qw_ptr = qw_arr.ctypes.data_as(ctypes.c_void_p) if qw_arr is not None else None
                s2, e2 = ctypes.c_double(), ctypes.c_double()
                lib.ikq_quantize_mat_q4_0(
                    ctypes.cast(c_src, ctypes.c_void_p), src_ty, rows, cols,
                    qw_ptr, ctypes.cast(dst_w, ctypes.c_void_p), n_threads,
                    ctypes.byref(s2), ctypes.byref(e2),
                )
                key_q4 = "head" if name == "output.weight" else "attn"
                q4_stats[key_q4][0] += s2.value
                q4_stats[key_q4][1] += e2.value
                if name == "output.weight":
                    sqnr = 10.0 * math.log10(s2.value / max(e2.value, 1e-30))
                    log("  [q4_0  ] %-36s (%d->2)   %.1f MiB -> %.1f MiB  SQNR=%.2f dB" % (
                        name, src_ty, len(src_buf) / (1 << 20), nbytes / (1 << 20), sqnr))
                fout.write(bytes(dst_w))

            elif kind == "q8":
                fin.seek(src_data_start + src_off)
                raw_f32 = fin.read(src_ne[0] * src_ne[1] * 4)
                c_src = (ctypes.c_uint8 * len(raw_f32)).from_buffer_copy(raw_f32)
                dst_q = (ctypes.c_uint8 * nbytes)()
                s2, e2 = ctypes.c_double(), ctypes.c_double()
                lib.ikq_quantize_f32_to_q8(
                    ctypes.cast(c_src, ctypes.c_void_p), src_ne[0] * src_ne[1],
                    ctypes.cast(dst_q, ctypes.c_void_p), ctypes.byref(s2), ctypes.byref(e2),
                )
                ctrl_sig2 += s2.value
                ctrl_err2 += e2.value
                fout.write(bytes(dst_q))

            else:
                fin.seek(src_data_start + src_off)
                rem = nbytes
                while rem > 0:
                    blk = fin.read(min(chunk_sz, rem))
                    if not blk:
                        raise EOFError("short read copying %s" % name)
                    fout.write(blk)
                    rem -= len(blk)

        tail_pad = (-fout.tell()) % align
        if tail_pad:
            fout.write(b"\x00" * tail_pad)

    os.replace(tmp_dst, dst_path)
    dst_size = os.path.getsize(dst_path)
    for ty, label in ((13, "blk.20..23 gate_up (Q5_K -> NVFP4)"), (8, "blk.20..23 down (Q8_0 -> NVFP4)")):
        if exp_sig2[ty] > 0:
            sqnr = 10.0 * math.log10(exp_sig2[ty] / max(exp_err2[ty], 1e-30))
            log("[quant] %s aggregate SQNR = %.2f dB" % (label, sqnr))
    for k, label in (
        ("xs_gu", "Core gate_up (IQ2_XS_128)"),
        ("xxs_gu", "Non-core/all gate_up (IQ2_XXS_128)"),
        ("xxs_dn", "Experts down (IQ2_XXS_128)"),
    ):
        if iq2_stats[k][0] > 0:
            sqnr = 10.0 * math.log10(iq2_stats[k][0] / max(iq2_stats[k][1], 1e-30))
            log("[quant] %s aggregate SQNR = %.2f dB" % (label, sqnr))
    for k, label in (("attn", "Attention dense (Q4_0, 144 tensors)"), ("head", "LM head output.weight (Q4_0)")):
        if q4_stats[k][0] > 0:
            sqnr = 10.0 * math.log10(q4_stats[k][0] / max(q4_stats[k][1], 1e-30))
            log("[quant] %s aggregate SQNR = %.2f dB" % (label, sqnr))
    if ctrl_sig2 > 0:
        sqnr = 10.0 * math.log10(ctrl_sig2 / max(ctrl_err2, 1e-30))
        log("[quant] blk.6..17 loop_*_ctrl (F32 -> Q8_0, 144 tensors) aggregate SQNR = %.2f dB" % sqnr)
    log("[quant] done: %.2f GiB -> %.2f GiB (-%.1f MiB)" % (
        src_file_size / (1 << 30), dst_size / (1 << 30), (src_file_size - dst_size) / (1 << 20)))
    return dst_path


def main():
    p = argparse.ArgumentParser(description="Quantize Inkling-2-Mini GGUF (nvfp4 | iq2_m | iq2_xxs)")
    p.add_argument("src", help="Source Inkling-2-Mini GGUF")
    p.add_argument("dst", help="Destination quantized GGUF")
    p.add_argument("--preset", choices=("nvfp4", "iq2_m", "iq2_xxs"), default="nvfp4")
    p.add_argument("--imatrix", default=None, help="Optional IKIMAT1 imatrix binary path")
    p.add_argument("--threads", type=int, default=None, help="Worker thread count")
    a = p.parse_args()
    quantize_gguf(a.src, a.dst, preset=a.preset, imatrix_path=a.imatrix, n_threads=a.threads)


if __name__ == "__main__":
    main()
