/* moeaccel.c - Kindi's shared MoE acceleration policies (F9); see moeaccel.h. Copyright (c) 2026 Amir Husain.
   SPDX-License-Identifier: AGPL-3.0-only. Plain C, no CUDA: the engines do the copies and the maths. */
#include "moeaccel.h"
#include <stdlib.h>
#include <string.h>

struct ma_cache {
    int nl, ne, nslot, window, swap;
    float decay;
    float* score;                 /* [nl * ne] aged use count */
    int* slot_of;                 /* [nl * ne] admitted slot, or -1 */
    unsigned char* state;         /* [nl * ne] 0 none, 1 cached (admitted), 2 copying, 3 excluded */
    int* owner;                   /* [nslot] key, or -1; a copying slot keeps its new key */
    unsigned char* ready;         /* [nslot] admitted */
    int ticks;
    int nlive;                    /* slots [0, nlive) have memory (0: all) */
    long long picks, hits, swaps, admits, windows;
};

static int envi(const char* n, int d) { const char* s = getenv(n); return s && s[0] ? atoi(s) : d; }
static float envf(const char* n, float d) { const char* s = getenv(n); return s && s[0] ? (float)atof(s) : d; }

ma_cache* ma_cache_new(int nl, int ne, int nslot, int window, int swap, float decay) {
    ma_cache* c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->nl = nl; c->ne = ne; c->nslot = nslot;
    c->window = envi("KINDI_MA_WINDOW", window);
    c->swap = envi("KINDI_MA_SWAP", swap);
    c->decay = envf("KINDI_MA_DECAY", decay);
    if (c->window < 1) c->window = 1;
    if (c->swap < 0) c->swap = 0;
    const size_t k = (size_t)nl * ne;
    c->score = calloc(k, sizeof(float));
    c->slot_of = malloc(k * sizeof(int));
    c->state = calloc(k, 1);
    c->owner = malloc((size_t)(nslot > 0 ? nslot : 1) * sizeof(int));
    c->ready = calloc((size_t)(nslot > 0 ? nslot : 1), 1);
    if (!c->score || !c->slot_of || !c->state || !c->owner || !c->ready) { ma_cache_free(c); return NULL; }
    for (size_t i = 0; i < k; i++) c->slot_of[i] = -1;
    for (int s = 0; s < nslot; s++) c->owner[s] = -1;
    return c;
}
void ma_cache_free(ma_cache* c) {
    if (!c) return;
    free(c->score); free(c->slot_of); free(c->state); free(c->owner); free(c->ready); free(c);
}
void ma_cache_exclude(ma_cache* c, int l, int e) {
    if (l < 0 || l >= c->nl || e < 0 || e >= c->ne) return;
    const int k = l * c->ne + e;
    if (c->state[k] == 0) c->state[k] = 3;
}
void ma_cache_exclude_layer(ma_cache* c, int l) {
    for (int e = 0; e < c->ne; e++) ma_cache_exclude(c, l, e);
}
int ma_cache_seed(ma_cache* c, const int* layer, const int* expert, int n) {
    int s = 0;
    for (int i = 0; i < n && s < c->nslot; i++) {
        if (layer[i] < 0 || layer[i] >= c->nl || expert[i] < 0 || expert[i] >= c->ne) continue;
        const int k = layer[i] * c->ne + expert[i];
        if (c->state[k]) continue;
        while (s < c->nslot && c->owner[s] >= 0) s++;
        if (s == c->nslot) break;
        c->owner[s] = k; c->ready[s] = 1; c->slot_of[k] = s; c->state[k] = 1;
        s++;
    }
    int filled = 0;
    for (int i = 0; i < c->nslot; i++) filled += c->owner[i] >= 0;
    return filled;
}
int ma_cache_slot(const ma_cache* c, int l, int e) {
    if (l < 0 || l >= c->nl || e < 0 || e >= c->ne) return -1;
    return c->slot_of[l * c->ne + e];
}
int ma_cache_owner(const ma_cache* c, int s) {
    return s >= 0 && s < c->nslot && c->ready[s] ? c->owner[s] : -1;
}
void ma_cache_note(ma_cache* c, int l, const int* ids, int n) {
    if (l < 0 || l >= c->nl) return;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= c->ne) continue;
        const int k = l * c->ne + ids[i];
        c->picks++;
        c->hits += c->slot_of[k] >= 0;
        c->score[k] += 1.f;
    }
}
int ma_cache_tick(ma_cache* c, ma_swap* out, int max) {
    c->windows++;
    if (++c->ticks < c->window) return 0;
    c->ticks = 0;
    const int nk = c->nl * c->ne;
    /* aged; a score that has decayed below 1e-20 is 0 (left alone it goes subnormal, and subnormal multiplies run
       about 100x slower on x86: 7 classes x 24,576 keys took 2.9 ms a step on a rented RTX 3080 test machine) */
    for (int k = 0; k < nk; k++) {
        const float v = c->score[k] * c->decay;
        c->score[k] = v < 1e-20f ? 0.f : v;
    }
    int want = c->swap < max ? c->swap : max, n = 0;
    if (want <= 0 || c->nslot <= 0) return 0;
    /* the want most-asked keys that are not on the card (selection by repeated maximum: want is small) */
    /* the same choice as a scan for the lowest candidate at every key, with that lowest kept until it is replaced
       (keys below it are the common case once the list is full: one compare each) */
    int* cand = malloc(sizeof(int) * want);
    int nc = 0, lo = 0;
    for (int k = 0; k < nk; k++) {
        if (c->state[k] != 0 || c->score[k] <= 0.f) continue;
        if (nc < want) {
            cand[nc++] = k;
            if (nc == want) {
                lo = 0;
                for (int j = 1; j < nc; j++) if (c->score[cand[j]] < c->score[cand[lo]]) lo = j;
            }
        } else if (c->score[k] > c->score[cand[lo]]) {
            cand[lo] = k;
            lo = 0;
            for (int j = 1; j < nc; j++) if (c->score[cand[j]] < c->score[cand[lo]]) lo = j;
        }
    }
    /* most asked first */
    for (int i = 1; i < nc; i++)
        for (int j = i; j > 0 && c->score[cand[j]] > c->score[cand[j - 1]]; j--) { int t = cand[j]; cand[j] = cand[j - 1]; cand[j - 1] = t; }
    /* victims: the admitted slots with the lowest scores (copying slots are never victims) */
    unsigned char* used = calloc((size_t)c->nslot, 1);
    for (int i = 0; i < nc; i++) {
        int v = -1;
        const int live = c->nlive > 0 && c->nlive < c->nslot ? c->nlive : c->nslot;
        for (int s = 0; s < live; s++) {
            if (used[s]) continue;
            if (c->owner[s] < 0) { v = s; break; }                 /* an empty slot first */
            if (!c->ready[s]) continue;
            if (v < 0 || c->score[c->owner[s]] < c->score[c->owner[v]]) v = s;
        }
        if (v < 0) break;
        if (c->owner[v] >= 0 && !(c->score[cand[i]] > c->score[c->owner[v]])) break;
        used[v] = 1;
        if (c->owner[v] >= 0) {                                    /* evict at once */
            const int old = c->owner[v];
            c->slot_of[old] = -1;
            if (c->state[old] == 1) c->state[old] = 0;
        }
        c->owner[v] = cand[i]; c->ready[v] = 0; c->state[cand[i]] = 2;
        out[n].slot = v; out[n].layer = cand[i] / c->ne; out[n].expert = cand[i] % c->ne;
        n++;
    }
    free(used); free(cand);
    c->swaps += n;
    return n;
}
/* slots [0, n) have card memory (the elastic tail is partly or not mapped): the policy picks targets only there */
void ma_cache_set_live(ma_cache* c, int n) { c->nlive = n; }
void ma_cache_admit(ma_cache* c, int s) {
    if (s < 0 || s >= c->nslot || c->ready[s] || c->owner[s] < 0) return;
    const int k = c->owner[s];
    c->ready[s] = 1; c->slot_of[k] = s; c->state[k] = 1;
    c->admits++;
}
void ma_cache_cancel(ma_cache* c, int s) {
    if (s < 0 || s >= c->nslot || c->ready[s] || c->owner[s] < 0) return;
    const int k = c->owner[s];
    if (c->state[k] == 2) c->state[k] = 0;
    c->owner[s] = -1; c->ready[s] = 1;
    c->swaps--;
}
void ma_cache_stats(const ma_cache* c, long long* out) {
    out[0] = c->picks; out[1] = c->hits; out[2] = c->swaps; out[3] = c->admits; out[4] = c->windows;
}

struct ma_off { float share, maxshare; long long feeds; };
ma_off* ma_off_new(float share0, float maxshare) {
    ma_off* o = calloc(1, sizeof *o);
    if (!o) return NULL;
    o->share = envf("KINDI_MA_OFF_SHARE", share0);
    o->maxshare = maxshare;
    if (o->share < 0.f) o->share = 0.f;
    if (o->share > o->maxshare) o->share = o->maxshare;
    return o;
}
void ma_off_free(ma_off* o) { free(o); }
int ma_off_count(ma_off* o, int n) {
    int k = (int)(o->share * n + 0.5f);
    return k < 0 ? 0 : k > n ? n : k;
}
void ma_off_feed(ma_off* o, double cpu_ms, int n_cpu, double copy_ms, int n_copy) {
    if (n_cpu <= 0 || n_copy <= 0 || cpu_ms <= 0 || copy_ms <= 0) return;
    const double tc = cpu_ms / n_cpu, tg = copy_ms / n_copy;
    const double f = tc / (tc + tg);                /* (1 - f) n tc = f n tg: both sides finish together */
    o->share += 0.2f * ((float)f - o->share);
    if (o->share < 0.f) o->share = 0.f;
    if (o->share > o->maxshare) o->share = o->maxshare;
    o->feeds++;
}
float ma_off_share(const ma_off* o) { return o->share; }

/* ---------------------------------------------------------------- the mailbox (see moeaccel.h, moeaccel.cuh) */
#include <stdatomic.h>
#include <time.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define MA_PAUSE() _mm_pause()
#else
#define MA_PAUSE() ((void)0)
#endif

static long ma_al(long v) { return (v + 63) / 64 * 64; }
void ma_box_layout(int tmax, int nsel, long kin_bytes, long row_bytes, int per_pair, long* off) {
    off[0] = 256;
    off[1] = off[0] + ma_al((long)tmax * nsel * 4);
    off[2] = off[1] + ma_al((long)tmax * kin_bytes);
    off[3] = off[2] + ma_al((long)tmax * (per_pair ? nsel : 1) * row_bytes);
}
static double ma_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
/* the last pass's host timeline (KINDI_QXG_TIMELINE): per layer the time the post was seen and the time the CPU's
   rows were done (CLOCK_MONOTONIC seconds) */
#define MA_TL_MAX 64
static double ma_tl[MA_TL_MAX][2];
void ma_timeline(double* out, int n) {
    for (int i = 0; i < n && i < MA_TL_MAX; i++) { out[2 * i] = ma_tl[i][0]; out[2 * i + 1] = ma_tl[i][1]; }
}
int ma_box_serve_pass(void* host, int nl, int tmax, int nsel, long kin_bytes, long row_bytes, int per_pair,
                      unsigned seq, ma_fn fn, void* ctx, double timeout_s) {
    long off[4];
    ma_box_layout(tmax, nsel, kin_bytes, row_bytes, per_pair, off);
    const double t0 = ma_now();
    for (int l = 0; l < nl; l++) {
        unsigned char* b = (unsigned char*)host + (size_t)l * off[3];
        volatile unsigned* posted = (volatile unsigned*)(b + 0);
        for (long n = 0; *posted != seq; n++) {
            MA_PAUSE();
            if ((n & 4095) == 4095 && ma_now() - t0 > timeout_s) return -(1 + l);
        }
        atomic_thread_fence(memory_order_acquire);
        if (l < MA_TL_MAX) ma_tl[l][0] = ma_now();
        const int T = *(volatile int*)(b + 192);
        int r = T >= 1 && T <= tmax ? fn(ctx, l, T, (const int*)(b + off[0]), b + off[1], b + off[2]) : -1;
        atomic_thread_fence(memory_order_release);
        if (l < MA_TL_MAX) ma_tl[l][1] = ma_now();
        if (r < 0) *(volatile unsigned*)(b + 128) = 2u;
        *(volatile unsigned*)(b + 64) = seq;          /* done: the card goes on (with the err word set on an error) */
        if (r < 0) return -(1 + l);
    }
    return 0;
}
