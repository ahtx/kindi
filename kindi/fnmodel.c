/* fnmodel.c - the whole Flash-Next (qwen4exp) forward pass on the CPU (0.8 item C1): the products on fncpu.c's
   repacked weights, everything else (hyper-connection mixers, Gated DeltaNet, Qwen Sparse Attention, the router,
   the n-gram embeddings) in fp32 here. Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only.

   The math follows kindi/qwen4exp_ref.py (the fp32 correctness reference). One call (fm_forward) runs nt tokens
   (nt <= FM_TMAX) as one job on the pool: every thread walks the layers; each product is split by rows among the
   threads (fncpu.c slices) and the steps are separated by spin barriers; the small vector steps run on thread 0, the
   Gated DeltaNet heads and the attention heads are split among the threads. */
#include <sched.h>
#include "fncpu.c"

#define HD 2560                 /* stream width */
#define NHC 4                   /* streams */
#define HH (HD * NHC)
#define HNS (HH + 16)           /* the row stride of hn (rows 40 KiB apart would share L1 sets in the GEMM tiles) */
#define LR 320                  /* the mixers' low rank */
#define NE 512                  /* experts */
#define NSEL 10
#define FF 640                  /* expert width */
#define GK 16                   /* Gated DeltaNet: QK heads */
#define GV 48                   /* V heads */
#define GD 128                  /* head size */
#define QKVW (2 * GK * GD + GV * GD)
#define AH 24                   /* attention heads */
#define AKV 2
#define AD 256
#define IH 4                    /* indexer heads */
#define ID 128
#define ROPE 64
#define QSA_TOK 2048
#define QSA_R 4
#define PLE_D 160
#define PLE_ROWS 16
#define PLE_K 4
#define PLE_DIL 3
#ifndef FM_TMAX
#define FM_TMAX 256             /* tokens a call (prompt chunks); all-rows logits only up to FC_TMAX */
#endif

enum { FM_STATE_ONLY = -1, FM_LAST_LOGITS = 0, FM_ALL_LOGITS = 1 };

typedef struct { fc_tensor *down, *up, *inject; const float* norm; } fm_mix;
typedef struct {
    int attn;
    fm_mix ha, hf;
    fc_tensor *qkv, *z, *beta, *alpha, *sout; const float *conv, *ssm_a, *dt_bias, *ssm_norm;
    fc_tensor *q, *k, *v, *iq, *ik, *o; const float *qn, *kn, *iqn, *ikn;
    fc_tensor *router, *sg, *su, *sd, *eg, *eu, *ed; const float* shg;
    int ple; fc_tensor *pk, *pv; const float *pnk, *pnq, *pnc, *pconv;
    float *S, *convst, *kc, *vc, *ikc, *bk, *pst;
} fm_layer;

enum {  /* tensor slots (fm_set_t) */
    S_HA_DOWN, S_HA_UP, S_HA_INJ, S_HF_DOWN, S_HF_UP, S_HF_INJ, S_QKV, S_Z, S_BETA, S_ALPHA, S_SOUT, S_Q, S_K, S_V,
    S_IQ, S_IK, S_O, S_ROUTER, S_SG, S_SU, S_SD, S_EG, S_EU, S_ED, S_PK, S_PV, S_OUT_DOWN, S_OUT_UP, S_HEAD
};
enum {  /* vector slots (fm_set_v) */
    V_HA_NORM, V_HF_NORM, V_CONV, V_SSM_A, V_DT, V_SSM_NORM, V_QN, V_KN, V_IQN, V_IKN, V_SHG, V_PNK, V_PNQ, V_PNC,
    V_PCONV, V_OUT_NORM
};

typedef struct {
    int L, ctx, V;
    fm_layer* l;
    fm_mix out; fc_tensor* head;
    const uint8_t* embd; int embd_ty; long embd_rb;
    const uint8_t* ple_tab; int ple_ty; long ple_rb;
    int64_t ple_mult[3], ple_off[PLE_ROWS], ple_size[PLE_ROWS]; int ple_eos;
    float eps; float inv_freq[ROPE / 2];
    int pos, prev1, prev2;
    /* work buffers */
    float *h, *hn, *x, *y, *tmp, *big, *logits, *pout;
    float *probs, *topw; int* topi;
    float *qkvb, *zb, *bb, *ab, *qb, *kb, *vb, *iqb, *ikb, *ob, *eh, *sh, *shd, *lr, *wr, *att, *pemb;
    int8_t* aq; float *adx, *afx;          /* quantized activations, several inputs */
    int nt; const int* ids; int output;
    int nu, ue[FM_TMAX * NSEL], un[FM_TMAX * NSEL], up0[FM_TMAX * NSEL], plist[FM_TMAX * NSEL];
    int8_t* ehq; float *ehdx, *ehfx;       /* the experts' h, quantized: [pairs][FF] */
    double prof[8], pt;
    double* part;                          /* per thread partial sums [threads][FM_TMAX][NHC] */
    float* qsc; unsigned char* qvis;       /* QSA block scores and the visible positions of the token in work */
    float *query_att, *query_qsc;
    unsigned char* query_qvis;
    int query_heads, packed_gdn;
    /* the MTP head (fm_mtp): its decoder layer is l[L]; the draft head holds the rows of output.weight for the
       draft vocabulary (draft_ids) */
    int has_mtp, draft_n; const int* draft_ids;
    fc_tensor *mtp_emb, *mtp_hid, *draft; fm_mix mtp_out; const float *mtp_nemb, *mtp_nhid;
    float *mtp_h, *dlog; const float* mtp_hc; int mtp_pos;
    /* verify checkpoints (fm_ckpt): the Gated DeltaNet and conv states, and the n-gram conv state, after each of the
       first CK_MAX tokens of a call, so fm_rollback can keep a prefix of a checked draft window */
    int ck_on; float *ck_S, *ck_conv, *ck_pst; int ck_prev[8][2];
} fm_model;
#define CK_MAX 3

/* ------------------------------------------------------------------ small helpers (fp32) */
static inline float silu(float v) { return v / (1.f + expf(-v)); }
static inline float sigm(float v) { return 1.f / (1.f + expf(-v)); }
/* expf on 8 lanes (range reduction by ln 2 in two parts, a degree-5 polynomial; about 1 ulp from expf) */
static inline __m256 exp8(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}
static inline __m256 sigm8(__m256 v) {
    const __m256 one = _mm256_set1_ps(1.f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), v))));
}
static void rms_w(float* o, const float* x, const float* w, int n, float eps) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    const float r = 1.f / sqrtf((float)(s / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * (w ? w[i] : 1.f);
}
static void rope_angles(const fm_model* M, int pos, float* c, float* s) {
    for (int i = 0; i < ROPE / 2; i++) {
        const double a = (double)pos * M->inv_freq[i];
        c[i] = (float)cos(a);
        s[i] = (float)sin(a);
    }
}
static void rope(float* v, const float* c, const float* s) {    /* NeoX on the first ROPE dims */
    const int half = ROPE / 2;
    for (int i = 0; i < half; i++) {
        const float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * c[i] - x2 * s[i]; v[i + half] = x2 * c[i] + x1 * s[i];
    }
}

/* the phase timer (thread 0): time since the last mark goes to bucket b (0 mixers, 1 Gated DeltaNet, 2 attention,
   3 router and shared expert, 4 routed experts, 5 embedding and head, 6 n-gram embeddings) */
static inline double now_s(void) { struct timespec a; clock_gettime(CLOCK_MONOTONIC, &a); return a.tv_sec + 1e-9 * a.tv_nsec; }
static inline void prof_mark(fm_model* M, int b) { const double x = now_s(); M->prof[b] += x - M->pt; M->pt = x; }

/* ------------------------------------------------------------------ the barrier
   Two levels: the threads of a NUMA node count on their node's line, the last one of each node on a global line,
   and the last node releases everyone through the generation word. A forward pass meets here about 1,300 times a
   token while writing; one flat counter took ~40 cross-socket read-modify-writes each time.
   KINDI_FNCPU_BARRIER=flat: the pool's own barrier. */
typedef struct { _Alignas(64) atomic_long cnt; char pad[56]; } fm_line;
static fm_line fb_node[2], fb_top;
static _Alignas(64) atomic_uint fb_gen;
static int fb_on = 1, fb_nodes, fb_size[2], fb_n = -1;
static __thread int fb_node_of = -1;
static unsigned long long fb_wait[256], fb_count[256];   /* per thread: TSC cycles waiting in the barrier, barriers */
/* KINDI_FNCPU_BARPROF=1: waiting cycles summed over threads per barrier site (source line) */
#define FB_SITES 2048
static int fb_prof;
static _Atomic unsigned long long fb_site_wait[FB_SITES];
static __thread int fb_site;
static __thread int fb_t = -1;
static void fb_setup(int n) {
    fb_size[0] = fb_size[1] = 0;
    for (int t = 0; t < n; t++) fb_size[moe_pool_node_of(t) ? 1 : 0]++;
    fb_nodes = (fb_size[0] > 0) + (fb_size[1] > 0);
    fb_n = n;
}
static inline void fm_barrier(int t, int n) {
    if (!fb_on) { moe_pool_barrier(); return; }
    if (fb_node_of < 0) fb_node_of = moe_pool_node_of(t) ? 1 : 0;
    const unsigned g = atomic_load_explicit(&fb_gen, memory_order_acquire);
    const int nd = fb_node_of;
    if (atomic_fetch_add_explicit(&fb_node[nd].cnt, 1, memory_order_acq_rel) == fb_size[nd] - 1) {
        atomic_store_explicit(&fb_node[nd].cnt, 0, memory_order_relaxed);
        if (atomic_fetch_add_explicit(&fb_top.cnt, 1, memory_order_acq_rel) == fb_nodes - 1) {
            atomic_store_explicit(&fb_top.cnt, 0, memory_order_relaxed);
            atomic_store_explicit(&fb_gen, g + 1, memory_order_release);
            return;
        }
    }
    const unsigned long long c0 = __rdtsc();
    for (int i = 0; atomic_load_explicit(&fb_gen, memory_order_acquire) == g; i++)
        if (i < (1 << 20)) _mm_pause(); else sched_yield();
    const unsigned long long w = __rdtsc() - c0;
    if (t < 256) { fb_wait[t] += w; fb_count[t]++; }
    if (fb_prof) atomic_fetch_add_explicit(&fb_site_wait[fb_site % FB_SITES], w, memory_order_relaxed);
    (void)n; (void)fb_t;
}

/* ------------------------------------------------------------------ dynamic row blocks within a node (writing)
   The barrier statistics showed the threads waiting ~23% of a token: fixed slices finish at different times (other
   processes on some cores, memory channels). While writing, the threads of a node instead take blocks of their node's
   rows (the rows their node's threads repacked, so still node-local) from a shared counter. Counter slot c of node d
   only advances: every call adds its blocks plus one failed take per thread of the node, so each thread knows the
   next call's base without a reset. Slots are reused only across barriers. KINDI_FNCPU_DYN=0: fixed slices. */
#define DYN_SLOTS 8
static int dyn_on = 1;
static fm_line dyn_cnt[2][DYN_SLOTS];
static __thread long dyn_base[DYN_SLOTS];
/* node nd's share [u0, u1) of U units, as slice() splits them among the node's threads */
static inline void node_units(int U, int nd, int* u0, int* u1) {
    const int n0 = fb_size[0], n = fb_size[0] + fb_size[1];
    const int mid = (int)((long)U * n0 / n);
    *u0 = nd ? mid : 0; *u1 = nd ? U : mid;
}
/* the next block of slot c (nblk blocks this call): its index, or -1 when they are taken (the slot's base moves on) */
static inline int dyn_take(int c, int nblk) {
    const int nd = fb_node_of;
    const long g = (long)atomic_fetch_add_explicit(&dyn_cnt[nd][c].cnt, 1, memory_order_relaxed) - dyn_base[c];
    if (g < nblk) return (int)g;
    dyn_base[c] += nblk + fb_size[nd];
    return -1;
}

/* ------------------------------------------------------------------ products on the pool
   all threads call prod(); thread 0 has quantized the inputs; each thread does its slice of each listed tensor;
   a barrier follows. in: index of the quantized input (into M->aq) per tensor; y: outputs [nt][R] */
typedef struct { fc_tensor* T; int in; int e; float* y; long ys; const float* xf; long xs;
                 const int8_t* pq; const float *pdx, *pfx; } pjob;     /* pq: a thread's private quantized input */
static size_t ACAP;                                           /* values per quantized input slot */
/* the fp32 GEMM path (fncpu.c rows_gemm) for calls of at least gemm_min tokens: the inputs stay fp32 (xf, xs) and no
   step quantizes them; KINDI_FNCPU_GEMM=0 turns it off, =N sets the token threshold (default 16) */
static int gemm_min = 16;
#define GEMM(M) (gemm_min > 0 && (M)->nt >= gemm_min)
static int short_r = 128;      /* GEMM path: matrices of at most this many rows are split by tokens (rows_short; KINDI_FNCPU_SHORT) */
static int gemm_emin = 6;       /* GEMM path: experts with fewer tokens use the int8 kernels (KINDI_FNCPU_GEMM_EMIN) */
static int gemm_2d = 1;         /* GEMM path: short-but-wide matrices split by rows and tokens (KINDI_FNCPU_GEMM_2D) */
/* writing (calls of at most PRIV_T tokens): each thread quantizes a step's whole input itself into private buffers
   instead of thread 0 (or a slice per thread) followed by a barrier; the same arithmetic, about 430 fewer barriers
   a token. Measured slower than the barriers it saves (two-level barrier ~4 us): KINDI_FNCPU_PRIV=1 turns it on */
#define PRIV_T 2
static int priv_on = 0;
#define PRIV(M) (priv_on && !GEMM(M) && (M)->nt <= PRIV_T)
typedef struct { int8_t* q; float *dx, *fx; } pbuf;
static __thread int8_t* pv_q; static __thread float *pv_dx, *pv_fx;
#define PV_CAP (PRIV_T * HH + PRIV_T * LR)
/* region 0: [PRIV_T * HH] values, region 1: [PRIV_T * LR] */
static pbuf pv(int region) {
    if (!pv_q) {
        pv_q = aligned_alloc(64, PV_CAP); pv_dx = malloc(sizeof(float) * PV_CAP / 32); pv_fx = malloc(sizeof(float) * PV_CAP / 32);
    }
    const size_t o = region ? (size_t)PRIV_T * HH : 0;
    pbuf b = {pv_q + o, pv_dx + o / 32, pv_fx + o / 32};
    return b;
}
static pbuf pv_quant(const float* x, int C, int nt, long xs) {
    pbuf b = pv(0);
    for (int k = 0; k < nt; k++) quant_act(x + k * xs, C, b.q + (size_t)k * C, b.dx + (size_t)k * C / 32, b.fx + (size_t)k * C / 32);
    return b;
}
static int attn_tiled = 1;      /* GEMM path attention: 1 one token (attn_group), 2 four-token tiles (attn_multi, measured slower), 0 off
                                   (KINDI_FNCPU_ATTN_TILE) */
static void quant_in(fm_model* M, int slot, const float* x, int C, int nt, long xs) {
    for (int t = 0; t < nt; t++)
        quant_act(x + t * xs, C, M->aq + (size_t)slot * ACAP + (size_t)t * C, M->adx + ((size_t)slot * ACAP + (size_t)t * C) / 32,
                  M->afx + ((size_t)slot * ACAP + (size_t)t * C) / 32);
}
static inline void chunk(int len, int g, int t, int n, int* a, int* b);
static void prod_rows(fm_model* M, int t, int n, const pjob* J, int nj, int nt) {
    for (int i = 0; i < nj; i++) {
        if (!J[i].T) continue;
        int r0, r1; slice(J[i].T, t, n, &r0, &r1);
        if (J[i].xf && GEMM(M) && nt == M->nt) {
            const int R = J[i].T->R;
            if (R <= short_r) {
                int a, b; chunk(nt, 1, t, n, &a, &b);
                rows_short(J[i].T, J[i].e, a, b, J[i].xf, J[i].xs, J[i].y, J[i].ys);
            } else if (R < 32 * n && gemm_2d) {
                /* few rows a thread (the mixers' down: 8): groups of about 32 rows by groups of tokens, so each
                   thread runs full 16-row tiles; a row group is decoded by each of its token groups */
                const int g = J[i].T->kind == 4 ? 2 : 1, rg = R / 32 > 0 ? R / 32 : 1, tg = n / rg;
                if (t < rg * tg) {
                    const int units = (R + g - 1) / g, ri = t % rg, ti = t / rg;
                    int ra = (int)((long)units * ri / rg) * g, rb2 = (int)((long)units * (ri + 1) / rg) * g;
                    if (rb2 > R) rb2 = R;
                    int a, b; chunk(nt, 1, ti, tg, &a, &b);
                    if (a < b && ra < rb2)
                        rows_gemm(J[i].T, J[i].e, ra, rb2, b - a, J[i].xf + (size_t)a * J[i].xs, J[i].xs, NULL,
                                  J[i].y + (size_t)a * J[i].ys, J[i].ys, NULL, 0);
                }
            } else if (r0 < r1) rows_gemm(J[i].T, J[i].e, r0, r1, nt, J[i].xf, J[i].xs, NULL, J[i].y, J[i].ys, NULL, 0);
            continue;
        }
        const size_t base = (size_t)J[i].in * ACAP;
        const int8_t* q = J[i].pq ? J[i].pq : M->aq + base;
        const float* dx = J[i].pq ? J[i].pdx : M->adx + base / 32;
        const float* fx = J[i].pq ? J[i].pfx : M->afx + base / 32;
        if (dyn_on && fb_node_of >= 0 && i < DYN_SLOTS) {
            const int g = J[i].T->kind == 4 ? 2 : 1, U = (J[i].T->R + g - 1) / g, BU = 8 / g;
            int u0, u1; node_units(U, fb_node_of, &u0, &u1);
            const int nblk = (u1 - u0 + BU - 1) / BU;
            for (int b; (b = dyn_take(i, nblk)) >= 0;) {
                const int a = (u0 + b * BU) * g, e = u0 + (b + 1) * BU < u1 ? (u0 + (b + 1) * BU) * g : u1 * g;
                rows_big(J[i].T, J[i].e, a, e < J[i].T->R ? e : J[i].T->R, nt, q, dx, fx, J[i].y, J[i].ys);
            }
            continue;
        }
        rows_big(J[i].T, J[i].e, r0, r1, nt, q, dx, fx, J[i].y, J[i].ys);
    }
    fm_barrier(t, n);
}
static void prod(fm_model* M, int t, int n, const pjob* J, int nj) {
    prod_rows(M, t, n, J, nj, M->nt);
}
#define prod(M, t, n, J, nj) (fb_site = __LINE__, prod(M, t, n, J, nj))
#define prod_rows(M, t, n, J, nj, nt) (fb_site = __LINE__, prod_rows(M, t, n, J, nj, nt))
#define T0 if (t == 0)
#define BAR (fb_site = __LINE__, fm_barrier(t, n))

/* chunk c of n of [0, len) in units of g values */
static inline void chunk(int len, int g, int t, int n, int* a, int* b) {
    const int u = len / g;
    *a = (int)((long)u * t / n) * g; *b = (int)((long)u * (t + 1) / n) * g;
}
static inline float dot8(const float* a, const float* b, int len) {
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < len; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    return hsum8(s);
}

/* the gated-residual mixer: h [nt][HH] -> x [nt][HD], write gates wr [nt][4] (inject). Every step is split among
   the threads: the streams' squared sums by 256-value chunks, then each thread normalizes and quantizes its chunk */
static void mixer_priv(fm_model* M, int t, int n, const fm_mix* mx, float* x, float* wr) {
    const int nt = M->nt;
    int a, b; chunk(HH, 256, t, n, &a, &b);
    pbuf P = pv(0), Q = pv(1);
    for (int k = 0; k < nt; k++) {
        /* the streams' sums exactly as the shared path adds them: per thread slice, then over the threads in order */
        double tot[NHC] = {0};
        for (int u = 0; u < n; u++) {
            int ua, ub; chunk(HH, 256, u, n, &ua, &ub);
            double ps[NHC] = {0};
            for (int i = ua; i < ub; i++) { const float v = M->h[k * HH + i]; ps[i / HD] += (double)v * v; }
            for (int s = 0; s < NHC; s++) tot[s] += ps[s];
        }
        float r[NHC], row[HH] __attribute__((aligned(32)));
        for (int s = 0; s < NHC; s++) r[s] = 1.f / sqrtf((float)(tot[s] / HD) + M->eps);
        for (int i = 0; i < HH; i++) row[i] = M->h[k * HH + i] * r[i / HD] * mx->norm[i];
        for (int i = a; i < b; i++) M->hn[k * HNS + i] = row[i];
        quant_act(row, HH, P.q + (size_t)k * HH, P.dx + (size_t)k * HH / 32, P.fx + (size_t)k * HH / 32);
    }
    pjob J[2] = {{mx->down, 0, 0, M->lr, LR, NULL, 0, P.q, P.dx, P.fx}, {mx->inject, 0, 0, M->wr, NHC, NULL, 0, P.q, P.dx, P.fx}};
    prod(M, t, n, J, mx->inject ? 2 : 1);
    for (int k = 0; k < nt; k++) {
        float l[LR] __attribute__((aligned(32)));
        for (int i = 0; i < LR; i++) l[i] = silu(M->lr[k * LR + i] / NHC);
        quant_act(l, LR, Q.q + (size_t)k * LR, Q.dx + (size_t)k * LR / 32, Q.fx + (size_t)k * LR / 32);
    }
    T0 if (wr) for (int k = 0; k < nt; k++) for (int s = 0; s < NHC; s++) wr[k * NHC + s] = 2.f * sigm(M->wr[k * NHC + s] / NHC);
    pjob U = {mx->up, 1, 0, M->big, HH, NULL, 0, Q.q, Q.dx, Q.fx};
    prod(M, t, n, &U, 1);
    chunk(HD, 8, t, n, &a, &b);
    const __m256 inv = _mm256_set1_ps(1.f / NHC);
    for (int k = 0; k < nt; k++)
        for (int i = a; i < b; i += 8) {
            __m256 s = _mm256_setzero_ps();
            for (int c = 0; c < NHC; c++)
                s = _mm256_fmadd_ps(sigm8(_mm256_loadu_ps(M->big + k * HH + c * HD + i)), _mm256_loadu_ps(M->hn + k * HNS + c * HD + i), s);
            _mm256_storeu_ps(x + k * HD + i, _mm256_mul_ps(s, inv));
        }
    BAR;
}

static void mixer(fm_model* M, int t, int n, const fm_mix* mx, float* x, float* wr) {
    const int nt = M->nt;
    if (PRIV(M)) { mixer_priv(M, t, n, mx, x, wr); return; }
    int a, b; chunk(HH, 256, t, n, &a, &b);
    for (int k = 0; k < nt; k++) {
        double* ps = M->part + ((size_t)t * FM_TMAX + k) * NHC;
        for (int s = 0; s < NHC; s++) ps[s] = 0;
        for (int i = a; i < b; i++) { const float v = M->h[k * HH + i]; ps[i / HD] += (double)v * v; }
    }
    BAR;
    for (int k = 0; k < nt; k++) {
        float r[NHC];
        for (int s = 0; s < NHC; s++) {
            double tot = 0;
            for (int u = 0; u < n; u++) tot += M->part[((size_t)u * FM_TMAX + k) * NHC + s];
            r[s] = 1.f / sqrtf((float)(tot / HD) + M->eps);
        }
        for (int i = a; i < b; i++) M->hn[k * HNS + i] = M->h[k * HH + i] * r[i / HD] * mx->norm[i];
        if (b > a && !GEMM(M)) {
            const size_t o = (size_t)k * HH + a;
            quant_act(M->hn + (size_t)k * HNS + a, b - a, M->aq + o, M->adx + o / 32, M->afx + o / 32);
        }
    }
    BAR;
    pjob J[2] = {{mx->down, 0, 0, M->lr, LR, M->hn, HNS}, {mx->inject, 0, 0, M->wr, NHC, M->hn, HNS}};
    prod(M, t, n, J, mx->inject ? 2 : 1);
    if (GEMM(M)) {
        for (int k = t; k < nt; k += n) {
            for (int i = 0; i < LR; i++) M->lr[k * LR + i] = silu(M->lr[k * LR + i] / NHC);
            if (wr) for (int s = 0; s < NHC; s++) wr[k * NHC + s] = 2.f * sigm(M->wr[k * NHC + s] / NHC);
        }
    } else T0 {
        for (int k = 0; k < nt; k++) {
            for (int i = 0; i < LR; i++) M->lr[k * LR + i] = silu(M->lr[k * LR + i] / NHC);
            if (wr) for (int s = 0; s < NHC; s++) wr[k * NHC + s] = 2.f * sigm(M->wr[k * NHC + s] / NHC);
        }
        quant_in(M, 1, M->lr, LR, nt, LR);
    }
    BAR;
    pjob U = {mx->up, 1, 0, M->big, HH, M->lr, LR};
    prod(M, t, n, &U, 1);
    chunk(HD, 8, t, n, &a, &b);
    const __m256 inv = _mm256_set1_ps(1.f / NHC);
    for (int k = 0; k < nt; k++)
        for (int i = a; i < b; i += 8) {
            __m256 s = _mm256_setzero_ps();
            for (int c = 0; c < NHC; c++)
                s = _mm256_fmadd_ps(sigm8(_mm256_loadu_ps(M->big + k * HH + c * HD + i)), _mm256_loadu_ps(M->hn + k * HNS + c * HD + i), s);
            _mm256_storeu_ps(x + k * HD + i, _mm256_mul_ps(s, inv));
        }
    BAR;
}

/* h += y * wr, split among the threads (callers put a barrier after) */
static void write_h(fm_model* M, int t, int n, const float* y, const float* wr) {
    int a, b; chunk(HD, 8, t, n, &a, &b);
    for (int k = 0; k < M->nt; k++)
        for (int s = 0; s < NHC; s++)
            for (int i = a; i < b; i++) M->h[k * HH + s * HD + i] += y[k * HD + i] * wr[k * NHC + s];
}

/* quantize nt rows of len values (len a multiple of 256) into slot 0, split among the threads by 256-value chunks */
static void quant_rows(fm_model* M, int t, int n, const float* x, int len) {
    int a, b; chunk(len, 256, t, n, &a, &b);
    if (b <= a) return;
    for (int k = 0; k < M->nt; k++) {
        const size_t o = (size_t)k * len + a;
        quant_act(x + o, b - a, M->aq + o, M->adx + o / 32, M->afx + o / 32);
    }
}
static void quant_par(fm_model* M, int t, int n, const float* x, int len) {
    if (!GEMM(M)) quant_rows(M, t, n, x, len);
}

/* ------------------------------------------------------------------ Gated DeltaNet */
/* Value columns have independent recurrent states. Split them across the pool
   and normalize the completed output after the columns are reunited. */
static void gdn_packed_recur(fm_model* M, int t, int n, fm_layer* L) {
    enum { PANEL = 32, PANELS = GD / PANEL };
    for (int k = t; k < M->nt; k += n) {
        float* row = M->qkvb + k * QKVW;
        for (int h = 0; h < GK; h++) {
            float* q = row + h * GD;
            float* key = row + (GK + h) * GD;
            const float rq = 1.f / sqrtf(dot8(q, q, GD) + 1e-6f) / sqrtf((float)GD);
            const float rk = 1.f / sqrtf(dot8(key, key, GD) + 1e-6f);
            for (int i = 0; i < GD; i++) { q[i] *= rq; key[i] *= rk; }
        }
        for (int h = 0; h < GV; h++) {
            const int p = k * GV + h;
            const float a = M->ab[p] + L->dt_bias[h];
            M->ab[p] = expf(L->ssm_a[h] * (a > 20.f ? a : log1pf(expf(a))));
            M->bb[p] = sigm(M->bb[p]);
        }
    }
    BAR;
    for (int job = t; job < GV * PANELS; job += n) {
        const int h = job / PANELS, col = (job % PANELS) * PANEL, hk = h % GK;
        float* state = L->S + (size_t)h * GD * GD + col;
        float packed[GD * PANEL] __attribute__((aligned(64)));
        for (int i = 0; i < GD; i++) memcpy(packed + i * PANEL, state + i * GD, PANEL * sizeof(float));
        for (int k = 0; k < M->nt; k++) {
            const float* row = M->qkvb + k * QKVW;
            const float* q = row + hk * GD;
            const float* key = row + (GK + hk) * GD;
            const float* v = row + (2 * GK + h) * GD + col;
            const float dec = M->ab[k * GV + h], beta = M->bb[k * GV + h];
            float mem[PANEL] = {0}, out[PANEL] = {0}, delta[PANEL];
            for (int i = 0; i < GD; i++) {
                float* s = packed + i * PANEL;
                for (int j = 0; j < PANEL; j++) { s[j] *= dec; mem[j] += key[i] * s[j]; }
            }
            for (int j = 0; j < PANEL; j++) delta[j] = (v[j] - mem[j]) * beta;
            for (int i = 0; i < GD; i++) {
                float* s = packed + i * PANEL;
                for (int j = 0; j < PANEL; j++) { s[j] += key[i] * delta[j]; out[j] += q[i] * s[j]; }
            }
            memcpy(M->ob + (k * GV + h) * GD + col, out, sizeof(out));
        }
        for (int i = 0; i < GD; i++) memcpy(state + i * GD, packed + i * PANEL, PANEL * sizeof(float));
    }
    BAR;
    for (int i = t; i < M->nt * GV; i += n) {
        float* out = M->ob + i * GD;
        const float* z = M->zb + i * GD;
        double s2 = 0;
        for (int j = 0; j < GD; j++) s2 += (double)out[j] * out[j];
        const float r = 1.f / sqrtf((float)(s2 / GD) + M->eps);
        for (int j = 0; j < GD; j++) out[j] = out[j] * r * L->ssm_norm[j] * sigm(z[j]);
    }
}

static void gdn(fm_model* M, int t, int n, fm_layer* L) {
    const int nt = M->nt;
    const int priv = PRIV(M);
    pbuf P = {0};
    if (priv) P = pv_quant(M->x, HD, nt, HD);
    else { if (!GEMM(M)) T0 quant_in(M, 0, M->x, HD, nt, HD); BAR; }
    pjob J[4] = {{L->qkv, 0, 0, M->qkvb, QKVW, M->x, HD, P.q, P.dx, P.fx}, {L->z, 0, 0, M->zb, GV * GD, M->x, HD, P.q, P.dx, P.fx},
                 {L->beta, 0, 0, M->bb, GV, M->x, HD, P.q, P.dx, P.fx}, {L->alpha, 0, 0, M->ab, GV, M->x, HD, P.q, P.dx, P.fx}};
    prod(M, t, n, J, 4);
    T0 prof_mark(M, 1);
    /* the short conv, split by channels (each thread keeps its channels' state), then the heads */
    {
        int c0, c1; chunk(QKVW, 32, t, n, &c0, &c1);
        for (int k = 0; k < nt; k++) {
            float* row = M->qkvb + k * QKVW;
            for (int c = c0; c < c1; c++) {
                const float* w = L->conv + c * 4;
                const float cur = row[c];
                const float v = w[0] * L->convst[c] + w[1] * L->convst[QKVW + c] + w[2] * L->convst[2 * QKVW + c] + w[3] * cur;
                L->convst[c] = L->convst[QKVW + c]; L->convst[QKVW + c] = L->convst[2 * QKVW + c]; L->convst[2 * QKVW + c] = cur;
                row[c] = silu(v);
            }
            if (M->ck_on && k < nt - 1 && k < CK_MAX) {
                float* ck = M->ck_conv + ((size_t)(L - M->l) * CK_MAX + k) * 3 * QKVW;
                for (int r = 0; r < 3; r++) memcpy(ck + r * QKVW + c0, L->convst + r * QKVW + c0, sizeof(float) * (c1 - c0));
            }
        }
    }
    BAR;
    if (M->packed_gdn && nt >= n && !M->ck_on) gdn_packed_recur(M, t, n, L);
    else for (int hv = t; hv < GV; hv += n) {
        float* S = L->S + (size_t)hv * GD * GD;       /* [dk][dv] */
        const int hk = hv % GK;
        for (int k = 0; k < nt; k++) {
            const float* row = M->qkvb + k * QKVW;
            const float* v = row + 2 * GK * GD + hv * GD;
            /* this head's q and k, l2-normed (q also / sqrt(GD)), as local copies */
            float q[GD] __attribute__((aligned(32))), kk[GD] __attribute__((aligned(32)));
            {
                const float* q0 = row + hk * GD; const float* k0 = row + GK * GD + hk * GD;
                const float rq = 1.f / sqrtf(dot8(q0, q0, GD) + 1e-6f) / sqrtf((float)GD), rk = 1.f / sqrtf(dot8(k0, k0, GD) + 1e-6f);
                for (int i = 0; i < GD; i++) { q[i] = q0[i] * rq; kk[i] = k0[i] * rk; }
            }
            const float beta = sigm(M->bb[k * GV + hv]);
            const float a = M->ab[k * GV + hv] + L->dt_bias[hv];
            const float g = L->ssm_a[hv] * (a > 20.f ? a : log1pf(expf(a)));
            const float dec = expf(g);
            float mem[GD], dl[GD];
            for (int j = 0; j < GD; j++) mem[j] = 0;
            for (int i = 0; i < GD; i++) {
                float* Si = S + i * GD; const float ki = kk[i];
                for (int j = 0; j < GD; j++) { Si[j] *= dec; mem[j] += ki * Si[j]; }
            }
            for (int j = 0; j < GD; j++) dl[j] = (v[j] - mem[j]) * beta;
            float o[GD];
            for (int j = 0; j < GD; j++) o[j] = 0;
            for (int i = 0; i < GD; i++) {
                float* Si = S + i * GD; const float ki = kk[i], qi = q[i];
                for (int j = 0; j < GD; j++) { Si[j] += ki * dl[j]; o[j] += qi * Si[j]; }
            }
            /* gated norm: rms over the head (eps), times ssm_norm, times sigmoid(z) */
            double s2 = 0; for (int j = 0; j < GD; j++) s2 += (double)o[j] * o[j];
            const float r = 1.f / sqrtf((float)(s2 / GD) + M->eps);
            float* dst = M->ob + k * GV * GD + hv * GD;
            const float* z = M->zb + k * GV * GD + hv * GD;
            for (int j = 0; j < GD; j++) dst[j] = o[j] * r * L->ssm_norm[j] * sigm(z[j]);
            if (M->ck_on && k < nt - 1 && k < CK_MAX)
                memcpy(M->ck_S + (((size_t)(L - M->l) * CK_MAX + k) * GV + hv) * GD * GD, S, sizeof(float) * GD * GD);
        }
    }
    BAR;
    T0 prof_mark(M, 7);
    if (priv) P = pv_quant(M->ob, GV * GD, nt, GV * GD);
    else { quant_par(M, t, n, M->ob, GV * GD); BAR; }
    pjob O = {L->sout, 0, 0, M->y, HD, M->ob, GV * GD, P.q, P.dx, P.fx};
    prod(M, t, n, &O, 1);
}

/* ------------------------------------------------------------------ Qwen Sparse Attention */
static void score_qsa_blocks(float* scores, const float* qi, const float* keys, int b0, int b1) {
    for (int bb = b0; bb < b1; bb++) {
        const float* bk = keys + (size_t)bb * ID;
        float sc = 0;
        for (int hh = 0; hh < IH; hh++) {
            const float d = dot8(qi + hh * ID, bk, ID);
            sc += d > 0 ? d : 0;
        }
        scores[bb] = sc / sqrtf((float)ID);
    }
}

static void select_qsa_positions(const float* scores, float* copy, unsigned char* visible, int positions) {
    const int nb = positions / QSA_R;
    const int keep = QSA_TOK / QSA_R;
    memcpy(copy, scores, sizeof(float) * nb);
    int lo = 0, hi = nb - 1, kth = keep - 1;
    while (lo < hi) {
        const float pivot = copy[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (copy[i] > pivot) i++;
            while (copy[j] < pivot) j--;
            if (i <= j) {
                const float tmp = copy[i];
                copy[i++] = copy[j];
                copy[j--] = tmp;
            }
        }
        if (kth <= j) hi = j;
        else if (kth >= i) lo = i;
        else break;
    }
    const float threshold = copy[kth];
    memset(visible, 0, positions);
    int taken = 0;
    for (int bb = 0; bb < nb; bb++) if (scores[bb] > threshold) {
        memset(visible + bb * QSA_R, 1, QSA_R);
        taken++;
    }
    for (int bb = 0; bb < nb && taken < keep; bb++) if (scores[bb] == threshold) {
        memset(visible + bb * QSA_R, 1, QSA_R);
        taken++;
    }
    memset(visible + nb * QSA_R, 1, positions % QSA_R);
}

static void attn_head(const fm_layer* L, const float* q, int kv, int positions, const unsigned char* visible,
                      float* scores, float* dst) {
    float mx = -INFINITY;
    for (int p = 0; p < positions; p++) {
        if (visible && !visible[p]) {
            scores[p] = -INFINITY;
            continue;
        }
        scores[p] = dot8(q, L->kc + ((size_t)p * AKV + kv) * AD, AD) / 16.f;
        if (scores[p] > mx) mx = scores[p];
    }
    float sum = 0;
    __m256 out[AD / 8];
    for (int i = 0; i < AD / 8; i++) out[i] = _mm256_setzero_ps();
    for (int p = 0; p < positions; p++) {
        if (scores[p] == -INFINITY) continue;
        const float w = expf(scores[p] - mx);
        sum += w;
        const float* vv = L->vc + ((size_t)p * AKV + kv) * AD;
        const __m256 wv = _mm256_set1_ps(w);
        for (int i = 0; i < AD / 8; i++) out[i] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vv + 8 * i), out[i]);
    }
    float val[AD] __attribute__((aligned(32)));
    for (int i = 0; i < AD / 8; i++) _mm256_store_ps(val + 8 * i, out[i]);
    for (int i = 0; i < AD; i++) dst[i] = val[i] / sum * sigm(q[AD + i]);
}

static void attn_pair(const fm_layer* L, const float* qa, const float* qb, int kv, int positions,
                      const unsigned char* visible, float* sa, float* sb, float* da, float* db) {
    /* The heads share KV and visibility. Each has its own reduction and softmax. */
    float ma = -INFINITY, mb = -INFINITY;
    for (int p = 0; p < positions; p++) {
        if (visible && !visible[p]) {
            sa[p] = sb[p] = -INFINITY;
            continue;
        }
        const float* k = L->kc + ((size_t)p * AKV + kv) * AD;
        __m256 xa = _mm256_setzero_ps(), xb = xa;
        for (int d = 0; d < AD; d += 8) {
            const __m256 key = _mm256_loadu_ps(k + d);
            xa = _mm256_fmadd_ps(_mm256_loadu_ps(qa + d), key, xa);
            xb = _mm256_fmadd_ps(_mm256_loadu_ps(qb + d), key, xb);
        }
        sa[p] = hsum8(xa) / 16.f;
        sb[p] = hsum8(xb) / 16.f;
        if (sa[p] > ma) ma = sa[p];
        if (sb[p] > mb) mb = sb[p];
    }
    float za = 0, zb = 0;
    __m256 oa[AD / 8], ob[AD / 8];
    for (int i = 0; i < AD / 8; i++) oa[i] = ob[i] = _mm256_setzero_ps();
    for (int p = 0; p < positions; p++) {
        if (sa[p] == -INFINITY || sb[p] == -INFINITY) {
            if (sa[p] == sb[p]) continue;
            attn_head(L, qa, kv, positions, visible, sa, da);
            attn_head(L, qb, kv, positions, visible, sb, db);
            return;
        }
        const float wa = expf(sa[p] - ma), wb = expf(sb[p] - mb);
        za += wa;
        zb += wb;
        const float* v = L->vc + ((size_t)p * AKV + kv) * AD;
        const __m256 va = _mm256_set1_ps(wa), vb = _mm256_set1_ps(wb);
        for (int i = 0; i < AD / 8; i++) {
            const __m256 value = _mm256_loadu_ps(v + 8 * i);
            oa[i] = _mm256_fmadd_ps(va, value, oa[i]);
            ob[i] = _mm256_fmadd_ps(vb, value, ob[i]);
        }
    }
    float val_a[AD] __attribute__((aligned(32))), val_b[AD] __attribute__((aligned(32)));
    for (int i = 0; i < AD / 8; i++) {
        _mm256_store_ps(val_a + 8 * i, oa[i]);
        _mm256_store_ps(val_b + 8 * i, ob[i]);
    }
    for (int i = 0; i < AD; i++) {
        da[i] = val_a[i] / za * sigm(qa[AD + i]);
        db[i] = val_b[i] / zb * sigm(qb[AD + i]);
    }
}

/* one token's 12 query heads of a KV head against its visible keys, as small matrix products (the GEMM path):
   scores S [12][V] = Q K^T / 16 with 3-head by 4-key register tiles over key blocks, one softmax per head
   (vector exp), then O [12][256] = W V with 6-head by 16-value tiles over blocks of 128 keys. q: the group's first
   head (heads 2 * AD apart, the output gate after each head's AD values); dst: AD apart. */
#define AG (AH / AKV)
static inline __m128 hsum4x(__m256 a, __m256 b, __m256 c, __m256 d) {
    const __m256 t = _mm256_hadd_ps(_mm256_hadd_ps(a, b), _mm256_hadd_ps(c, d));
    return _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
}
static void attn_group(const fm_layer* L, const float* q, int kv, int positions, const unsigned char* visible,
                       float* dst) {
    static __thread float *S, *O; static __thread int* vis; static __thread int cap;
    if (cap < positions) {
        free(S); free(vis);
        cap = positions + 1024;
        S = aligned_alloc(64, sizeof(float) * AG * (size_t)cap);
        vis = malloc(sizeof(int) * (size_t)cap);
        if (!O) O = aligned_alloc(64, sizeof(float) * AG * AD);
    }
    int V = 0;
    if (visible) { for (int p = 0; p < positions; p++) if (visible[p]) vis[V++] = p; }
    else for (int p = 0; p < positions; p++) vis[V++] = p;
    const float* kc = L->kc + (size_t)kv * AD;
    const float* vc = L->vc + (size_t)kv * AD;
    const size_t rs = (size_t)AKV * AD;                        /* the caches' row stride */
    for (int kb = 0; kb < V; kb += 64) {
        const int ke = kb + 64 < V ? kb + 64 : V;
        for (int h = 0; h < AG; h += 3) {
            const float *q0 = q + (size_t)h * 2 * AD, *q1 = q0 + 2 * AD, *q2 = q1 + 2 * AD;
            for (int v = kb; v < ke; v += 4) {
                const int nk = ke - v < 4 ? ke - v : 4;
                const float* k0 = kc + vis[v] * rs;
                const float* k1 = kc + vis[v + (nk > 1)] * rs;
                const float* k2 = kc + vis[v + (nk > 2 ? 2 : 0)] * rs;
                const float* k3 = kc + vis[v + (nk > 3 ? 3 : 0)] * rs;
                __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00,
                       a13 = a00, a20 = a00, a21 = a00, a22 = a00, a23 = a00;
                for (int d = 0; d < AD; d += 8) {
                    const __m256 x0 = _mm256_loadu_ps(k0 + d), x1 = _mm256_loadu_ps(k1 + d), x2 = _mm256_loadu_ps(k2 + d),
                                 x3 = _mm256_loadu_ps(k3 + d);
                    __m256 y = _mm256_loadu_ps(q0 + d);
                    a00 = _mm256_fmadd_ps(y, x0, a00); a01 = _mm256_fmadd_ps(y, x1, a01); a02 = _mm256_fmadd_ps(y, x2, a02); a03 = _mm256_fmadd_ps(y, x3, a03);
                    y = _mm256_loadu_ps(q1 + d);
                    a10 = _mm256_fmadd_ps(y, x0, a10); a11 = _mm256_fmadd_ps(y, x1, a11); a12 = _mm256_fmadd_ps(y, x2, a12); a13 = _mm256_fmadd_ps(y, x3, a13);
                    y = _mm256_loadu_ps(q2 + d);
                    a20 = _mm256_fmadd_ps(y, x0, a20); a21 = _mm256_fmadd_ps(y, x1, a21); a22 = _mm256_fmadd_ps(y, x2, a22); a23 = _mm256_fmadd_ps(y, x3, a23);
                }
                float r[3][4] __attribute__((aligned(16)));
                const __m128 sc = _mm_set1_ps(1.f / 16.f);
                _mm_store_ps(r[0], _mm_mul_ps(hsum4x(a00, a01, a02, a03), sc));
                _mm_store_ps(r[1], _mm_mul_ps(hsum4x(a10, a11, a12, a13), sc));
                _mm_store_ps(r[2], _mm_mul_ps(hsum4x(a20, a21, a22, a23), sc));
                for (int i = 0; i < 3; i++) for (int j = 0; j < nk; j++) S[(size_t)(h + i) * cap + v + j] = r[i][j];
            }
        }
    }
    float inv[AG];
    for (int h = 0; h < AG; h++) {
        float* sh = S + (size_t)h * cap;
        float mx = -INFINITY;
        for (int v = 0; v < V; v++) mx = sh[v] > mx ? sh[v] : mx;
        const __m256 m8 = _mm256_set1_ps(mx);
        __m256 z8 = _mm256_setzero_ps();
        int v = 0;
        for (; v + 8 <= V; v += 8) {
            const __m256 w = exp8(_mm256_sub_ps(_mm256_loadu_ps(sh + v), m8));
            _mm256_storeu_ps(sh + v, w);
            z8 = _mm256_add_ps(z8, w);
        }
        float z = hsum8(z8);
        for (; v < V; v++) { sh[v] = expf(sh[v] - mx); z += sh[v]; }
        inv[h] = 1.f / z;
    }
    memset(O, 0, sizeof(float) * AG * AD);
    for (int kb = 0; kb < V; kb += 128) {
        const int ke = kb + 128 < V ? kb + 128 : V;
        for (int h = 0; h < AG; h += 6) {
            const float *w0 = S + (size_t)h * cap, *w1 = w0 + cap, *w2 = w1 + cap, *w3 = w2 + cap, *w4 = w3 + cap,
                        *w5 = w4 + cap;
            for (int d = 0; d < AD; d += 16) {
                float* o = O + (size_t)h * AD + d;
                __m256 c00 = _mm256_load_ps(o), c01 = _mm256_load_ps(o + 8), c10 = _mm256_load_ps(o + AD),
                       c11 = _mm256_load_ps(o + AD + 8), c20 = _mm256_load_ps(o + 2 * AD), c21 = _mm256_load_ps(o + 2 * AD + 8),
                       c30 = _mm256_load_ps(o + 3 * AD), c31 = _mm256_load_ps(o + 3 * AD + 8), c40 = _mm256_load_ps(o + 4 * AD),
                       c41 = _mm256_load_ps(o + 4 * AD + 8), c50 = _mm256_load_ps(o + 5 * AD), c51 = _mm256_load_ps(o + 5 * AD + 8);
                for (int v = kb; v < ke; v++) {
                    const float* vv = vc + vis[v] * rs + d;
                    const __m256 x0 = _mm256_loadu_ps(vv), x1 = _mm256_loadu_ps(vv + 8);
                    __m256 b = _mm256_broadcast_ss(w0 + v); c00 = _mm256_fmadd_ps(b, x0, c00); c01 = _mm256_fmadd_ps(b, x1, c01);
                    b = _mm256_broadcast_ss(w1 + v); c10 = _mm256_fmadd_ps(b, x0, c10); c11 = _mm256_fmadd_ps(b, x1, c11);
                    b = _mm256_broadcast_ss(w2 + v); c20 = _mm256_fmadd_ps(b, x0, c20); c21 = _mm256_fmadd_ps(b, x1, c21);
                    b = _mm256_broadcast_ss(w3 + v); c30 = _mm256_fmadd_ps(b, x0, c30); c31 = _mm256_fmadd_ps(b, x1, c31);
                    b = _mm256_broadcast_ss(w4 + v); c40 = _mm256_fmadd_ps(b, x0, c40); c41 = _mm256_fmadd_ps(b, x1, c41);
                    b = _mm256_broadcast_ss(w5 + v); c50 = _mm256_fmadd_ps(b, x0, c50); c51 = _mm256_fmadd_ps(b, x1, c51);
                }
                _mm256_store_ps(o, c00); _mm256_store_ps(o + 8, c01); _mm256_store_ps(o + AD, c10); _mm256_store_ps(o + AD + 8, c11);
                _mm256_store_ps(o + 2 * AD, c20); _mm256_store_ps(o + 2 * AD + 8, c21); _mm256_store_ps(o + 3 * AD, c30);
                _mm256_store_ps(o + 3 * AD + 8, c31); _mm256_store_ps(o + 4 * AD, c40); _mm256_store_ps(o + 4 * AD + 8, c41);
                _mm256_store_ps(o + 5 * AD, c50); _mm256_store_ps(o + 5 * AD + 8, c51);
            }
        }
    }
    for (int h = 0; h < AG; h++) {
        const float* gate = q + (size_t)h * 2 * AD + AD;
        const __m256 iz = _mm256_set1_ps(inv[h]);
        for (int i = 0; i < AD; i += 8)
            _mm256_storeu_ps(dst + (size_t)h * AD + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(O + (size_t)h * AD + i), iz),
                                                                    sigm8(_mm256_loadu_ps(gate + i))));
    }
}

/* AQ consecutive query tokens of a chunk together against one KV head: the union of their visible keys is read once
   for all AQ x 12 query rows (scores [rows][V] by 3-row by 4-key tiles; a key a token does not see scores -inf),
   then one softmax per row and O = W V by 6-row by 16-value tiles. q0: the first token's first head of the group;
   qs: the token stride of q (AH * 2 * AD); vis[i]: token i's visible flags (NULL: every key before its position);
   pos[i]: its key count. dst: the first token's output, token stride ds. */
#define AQ 4
static void attn_multi(const fm_layer* L, const float* q0, long qs, int kv, int nq, const int* pos,
                       unsigned char* const* vis, float* dst, long ds) {
    static __thread float *S, *O; static __thread int *ul; static __thread int cap;
    const int R = nq * AG, P = pos[nq - 1];
    if (cap < P) {
        free(S); free(ul); free(O);
        cap = P + 1024;
        S = aligned_alloc(64, sizeof(float) * AQ * AG * (size_t)cap);
        ul = malloc(sizeof(int) * (size_t)cap);
        O = aligned_alloc(64, sizeof(float) * AQ * AG * AD);
    }
    int V = 0;
    for (int p = 0; p < P; p++) {
        int any = 0;
        for (int i = 0; i < nq && !any; i++) any = p < pos[i] && (!vis[i] || vis[i][p]);
        if (any) ul[V++] = p;
    }
    const float* kc = L->kc + (size_t)kv * AD;
    const float* vc = L->vc + (size_t)kv * AD;
    const size_t rs = (size_t)AKV * AD;
    /* query row r: token r / AG, head r % AG */
#define QROW(r) (q0 + (size_t)((r) / AG) * qs + (size_t)((r) % AG) * 2 * AD)
    for (int kb = 0; kb < V; kb += 64) {
        const int ke = kb + 64 < V ? kb + 64 : V;
        for (int r = 0; r < R; r += 3) {
            const float *q0r = QROW(r), *q1r = QROW(r + 1 < R ? r + 1 : r), *q2r = QROW(r + 2 < R ? r + 2 : r);
            for (int v = kb; v < ke; v += 4) {
                const int nk = ke - v < 4 ? ke - v : 4;
                const float* k0 = kc + ul[v] * rs;
                const float* k1 = kc + ul[v + (nk > 1)] * rs;
                const float* k2 = kc + ul[v + (nk > 2 ? 2 : 0)] * rs;
                const float* k3 = kc + ul[v + (nk > 3 ? 3 : 0)] * rs;
                __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00, a10 = a00, a11 = a00, a12 = a00,
                       a13 = a00, a20 = a00, a21 = a00, a22 = a00, a23 = a00;
                for (int d = 0; d < AD; d += 8) {
                    const __m256 x0 = _mm256_loadu_ps(k0 + d), x1 = _mm256_loadu_ps(k1 + d), x2 = _mm256_loadu_ps(k2 + d),
                                 x3 = _mm256_loadu_ps(k3 + d);
                    __m256 y = _mm256_loadu_ps(q0r + d);
                    a00 = _mm256_fmadd_ps(y, x0, a00); a01 = _mm256_fmadd_ps(y, x1, a01); a02 = _mm256_fmadd_ps(y, x2, a02); a03 = _mm256_fmadd_ps(y, x3, a03);
                    y = _mm256_loadu_ps(q1r + d);
                    a10 = _mm256_fmadd_ps(y, x0, a10); a11 = _mm256_fmadd_ps(y, x1, a11); a12 = _mm256_fmadd_ps(y, x2, a12); a13 = _mm256_fmadd_ps(y, x3, a13);
                    y = _mm256_loadu_ps(q2r + d);
                    a20 = _mm256_fmadd_ps(y, x0, a20); a21 = _mm256_fmadd_ps(y, x1, a21); a22 = _mm256_fmadd_ps(y, x2, a22); a23 = _mm256_fmadd_ps(y, x3, a23);
                }
                float sc[3][4] __attribute__((aligned(16)));
                const __m128 s16 = _mm_set1_ps(1.f / 16.f);
                _mm_store_ps(sc[0], _mm_mul_ps(hsum4x(a00, a01, a02, a03), s16));
                _mm_store_ps(sc[1], _mm_mul_ps(hsum4x(a10, a11, a12, a13), s16));
                _mm_store_ps(sc[2], _mm_mul_ps(hsum4x(a20, a21, a22, a23), s16));
                for (int i = 0; i < 3 && r + i < R; i++) {
                    const int tq = (r + i) / AG;
                    for (int j = 0; j < nk; j++) {
                        const int p = ul[v + j];
                        const int seen = p < pos[tq] && (!vis[tq] || vis[tq][p]);
                        S[(size_t)(r + i) * cap + v + j] = seen ? sc[i][j] : -INFINITY;
                    }
                }
            }
        }
    }
    float inv[AQ * AG];
    for (int r = 0; r < R; r++) {
        float* sh = S + (size_t)r * cap;
        float mx = -INFINITY;
        for (int v = 0; v < V; v++) mx = sh[v] > mx ? sh[v] : mx;
        const __m256 m8 = _mm256_set1_ps(mx);
        __m256 z8 = _mm256_setzero_ps();
        int v = 0;
        for (; v + 8 <= V; v += 8) {
            const __m256 w = exp8(_mm256_sub_ps(_mm256_loadu_ps(sh + v), m8));   /* (exp8 of -inf - mx: 0) */
            _mm256_storeu_ps(sh + v, w);
            z8 = _mm256_add_ps(z8, w);
        }
        float z = hsum8(z8);
        for (; v < V; v++) { sh[v] = expf(sh[v] - mx); z += sh[v]; }
        inv[r] = 1.f / z;
    }
    memset(O, 0, sizeof(float) * (size_t)R * AD);
    for (int kb = 0; kb < V; kb += 128) {
        const int ke = kb + 128 < V ? kb + 128 : V;
        for (int r = 0; r < R; r += 6) {
            const float *w0 = S + (size_t)r * cap, *w1 = w0 + cap, *w2 = w1 + cap, *w3 = w2 + cap, *w4 = w3 + cap,
                        *w5 = w4 + cap;                                   /* (R is a multiple of 6: 12 heads a token) */
            for (int d = 0; d < AD; d += 16) {
                float* o = O + (size_t)r * AD + d;
                __m256 c00 = _mm256_load_ps(o), c01 = _mm256_load_ps(o + 8), c10 = _mm256_load_ps(o + AD),
                       c11 = _mm256_load_ps(o + AD + 8), c20 = _mm256_load_ps(o + 2 * AD), c21 = _mm256_load_ps(o + 2 * AD + 8),
                       c30 = _mm256_load_ps(o + 3 * AD), c31 = _mm256_load_ps(o + 3 * AD + 8), c40 = _mm256_load_ps(o + 4 * AD),
                       c41 = _mm256_load_ps(o + 4 * AD + 8), c50 = _mm256_load_ps(o + 5 * AD), c51 = _mm256_load_ps(o + 5 * AD + 8);
                for (int v = kb; v < ke; v++) {
                    const float* vv = vc + ul[v] * rs + d;
                    const __m256 x0 = _mm256_loadu_ps(vv), x1 = _mm256_loadu_ps(vv + 8);
                    __m256 b = _mm256_broadcast_ss(w0 + v); c00 = _mm256_fmadd_ps(b, x0, c00); c01 = _mm256_fmadd_ps(b, x1, c01);
                    b = _mm256_broadcast_ss(w1 + v); c10 = _mm256_fmadd_ps(b, x0, c10); c11 = _mm256_fmadd_ps(b, x1, c11);
                    b = _mm256_broadcast_ss(w2 + v); c20 = _mm256_fmadd_ps(b, x0, c20); c21 = _mm256_fmadd_ps(b, x1, c21);
                    b = _mm256_broadcast_ss(w3 + v); c30 = _mm256_fmadd_ps(b, x0, c30); c31 = _mm256_fmadd_ps(b, x1, c31);
                    b = _mm256_broadcast_ss(w4 + v); c40 = _mm256_fmadd_ps(b, x0, c40); c41 = _mm256_fmadd_ps(b, x1, c41);
                    b = _mm256_broadcast_ss(w5 + v); c50 = _mm256_fmadd_ps(b, x0, c50); c51 = _mm256_fmadd_ps(b, x1, c51);
                }
                _mm256_store_ps(o, c00); _mm256_store_ps(o + 8, c01); _mm256_store_ps(o + AD, c10); _mm256_store_ps(o + AD + 8, c11);
                _mm256_store_ps(o + 2 * AD, c20); _mm256_store_ps(o + 2 * AD + 8, c21); _mm256_store_ps(o + 3 * AD, c30);
                _mm256_store_ps(o + 3 * AD + 8, c31); _mm256_store_ps(o + 4 * AD, c40); _mm256_store_ps(o + 4 * AD + 8, c41);
                _mm256_store_ps(o + 5 * AD, c50); _mm256_store_ps(o + 5 * AD + 8, c51);
            }
        }
    }
    for (int r = 0; r < R; r++) {
        const float* gate = QROW(r) + AD;
        float* dr = dst + (size_t)(r / AG) * ds + (size_t)(r % AG) * AD;
        const __m256 iz = _mm256_set1_ps(inv[r]);
        for (int i = 0; i < AD; i += 8)
            _mm256_storeu_ps(dr + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_load_ps(O + (size_t)r * AD + i), iz),
                                                   sigm8(_mm256_loadu_ps(gate + i))));
    }
#undef QROW
}

static void attn(fm_model* M, int t, int n, fm_layer* L) {
    const int nt = M->nt, p0 = M->pos;
    const int priv = PRIV(M);
    pbuf P = {0};
    if (priv) P = pv_quant(M->x, HD, nt, HD);
    else { if (!GEMM(M)) T0 quant_in(M, 0, M->x, HD, nt, HD); BAR; }
    pjob J[5] = {{L->q, 0, 0, M->qb, AH * 2 * AD, M->x, HD, P.q, P.dx, P.fx}, {L->k, 0, 0, M->kb, AKV * AD, M->x, HD, P.q, P.dx, P.fx},
                 {L->v, 0, 0, M->vb, AKV * AD, M->x, HD, P.q, P.dx, P.fx}, {L->iq, 0, 0, M->iqb, IH * ID, M->x, HD, P.q, P.dx, P.fx},
                 {L->ik, 0, 0, M->ikb, ID, M->x, HD, P.q, P.dx, P.fx}};
    prod(M, t, n, J, 5);
    for (int k = t; k < nt; k += n) {
        const int pos = p0 + k;
        float rc[ROPE / 2], rs[ROPE / 2];
        rope_angles(M, pos, rc, rs);
        float* qg = M->qb + k * AH * 2 * AD;
        for (int hh = 0; hh < AH; hh++) {
            float* q = qg + hh * 2 * AD;
            rms_w(q, q, L->qn, AD, M->eps);
            rope(q, rc, rs);
        }
        for (int hh = 0; hh < AKV; hh++) {
            float* kk = M->kb + k * AKV * AD + hh * AD;
            rms_w(kk, kk, L->kn, AD, M->eps);
            rope(kk, rc, rs);
            memcpy(L->kc + ((size_t)pos * AKV + hh) * AD, kk, AD * sizeof(float));
            memcpy(L->vc + ((size_t)pos * AKV + hh) * AD, M->vb + k * AKV * AD + hh * AD, AD * sizeof(float));
        }
        memcpy(L->ikc + (size_t)pos * ID, M->ikb + k * ID, ID * sizeof(float));
        if (pos + 1 > QSA_TOK) for (int hh = 0; hh < IH; hh++) {
            float* qi = M->iqb + k * IH * ID + hh * ID;
            rms_w(qi, qi, L->iqn, ID, M->eps);
            rope(qi, rc, rs);
        }
    }
    /* Cache each complete block once. Read newly projected keys directly so
       another thread's cache stores need no additional barrier. */
    if (L->bk) for (int b = p0 / QSA_R + t; b < (p0 + nt) / QSA_R; b += n) {
        float* bk = L->bk + (size_t)b * ID;
        memset(bk, 0, ID * sizeof(float));
        for (int r = 0; r < QSA_R; r++) {
            const int key_pos = b * QSA_R + r;
            const float* raw = key_pos < p0 ? L->ikc + (size_t)key_pos * ID : M->ikb + (size_t)(key_pos - p0) * ID;
            for (int i = 0; i < ID; i++) bk[i] += raw[i];
        }
        for (int i = 0; i < ID; i++) bk[i] /= QSA_R;
        float rc[ROPE / 2], rs[ROPE / 2];
        rope_angles(M, b * QSA_R, rc, rs);
        rms_w(bk, bk, L->ikn, ID, M->eps);
        rope(bk, rc, rs);
    }
    BAR;
    if (GEMM(M) && attn_tiled == 2) {
        /* AQ consecutive tokens a thread (attn_multi); each token's QSA selection first */
        static __thread unsigned char* mv; static __thread float *msc, *mcp; static __thread int mcap;
        if (mcap < M->ctx) {
            free(mv); free(msc); free(mcp);
            mv = malloc((size_t)AQ * M->ctx); msc = malloc(sizeof(float) * (M->ctx / QSA_R + 1));
            mcp = malloc(sizeof(float) * (M->ctx / QSA_R + 1)); mcap = M->ctx;
        }
        for (int k0 = t * AQ; k0 < nt; k0 += n * AQ) {
            const int nq = nt - k0 < AQ ? nt - k0 : AQ;
            int pos[AQ]; unsigned char* vis[AQ];
            for (int i = 0; i < nq; i++) {
                const int k = k0 + i;
                pos[i] = p0 + k + 1;
                vis[i] = NULL;
                if (pos[i] > QSA_TOK) {
                    vis[i] = mv + (size_t)i * M->ctx;
                    score_qsa_blocks(msc, M->iqb + k * IH * ID, L->bk, 0, pos[i] / QSA_R);
                    select_qsa_positions(msc, mcp, vis[i], pos[i]);
                }
            }
            for (int g = 0; g < AKV; g++)
                attn_multi(L, M->qb + (size_t)k0 * AH * 2 * AD + (size_t)g * AG * 2 * AD, AH * 2 * AD, g, nq, pos, vis,
                           M->ob + (size_t)k0 * AH * AD + (size_t)g * AG * AD, AH * AD);
        }
    } else if (GEMM(M) && attn_tiled) {
        for (int k = t; k < nt; k += n) {
            const int positions = p0 + k + 1;
            const int select = positions > QSA_TOK;
            if (select) {
                float* qsc = M->query_qsc ? M->query_qsc + (size_t)t * (M->ctx / QSA_R + 1) : NULL;
                static __thread float* own_sc; static __thread float* own_cp; static __thread unsigned char* own_vis;
                static __thread int own_cap;
                if (!qsc || !M->query_att) {
                    if (own_cap < M->ctx) {
                        free(own_sc); free(own_cp); free(own_vis);
                        own_sc = malloc(sizeof(float) * (M->ctx / QSA_R + 1)); own_cp = malloc(sizeof(float) * (M->ctx / QSA_R + 1));
                        own_vis = malloc(M->ctx); own_cap = M->ctx;
                    }
                }
                float* sc = M->query_qsc ? qsc : own_sc;
                float* cp = M->query_att ? M->query_att + (size_t)t * M->query_heads * M->ctx : own_cp;
                unsigned char* vis = M->query_qvis ? M->query_qvis + (size_t)t * M->ctx : own_vis;
                score_qsa_blocks(sc, M->iqb + k * IH * ID, L->bk, 0, positions / QSA_R);
                select_qsa_positions(sc, cp, vis, positions);
                for (int g = 0; g < AKV; g++)
                    attn_group(L, M->qb + (size_t)k * AH * 2 * AD + (size_t)g * AG * 2 * AD, g, positions, vis,
                               M->ob + (size_t)k * AH * AD + (size_t)g * AG * AD);
            } else
                for (int g = 0; g < AKV; g++)
                    attn_group(L, M->qb + (size_t)k * AH * 2 * AD + (size_t)g * AG * 2 * AD, g, positions, NULL,
                               M->ob + (size_t)k * AH * AD + (size_t)g * AG * AD);
        }
    } else if (M->query_att && nt >= n) {
        float* scores = M->query_att + (size_t)t * M->query_heads * M->ctx;
        float* qsc = M->query_qsc + (size_t)t * (M->ctx / QSA_R + 1);
        unsigned char* visible = M->query_qvis + (size_t)t * M->ctx;
        for (int k = t; k < nt; k += n) {
            const int positions = p0 + k + 1;
            const int select = positions > QSA_TOK;
            if (select) {
                score_qsa_blocks(qsc, M->iqb + k * IH * ID, L->bk, 0, positions / QSA_R);
                select_qsa_positions(qsc, scores, visible, positions);
            }
            for (int hh = 0; hh < AH; hh += M->query_heads) {
                const float* q = M->qb + k * AH * 2 * AD + hh * 2 * AD;
                float* dst = M->ob + k * AH * AD + hh * AD;
                const int kv = hh / (AH / AKV);
                if (M->query_heads == 2)
                    attn_pair(L, q, q + 2 * AD, kv, positions, select ? visible : NULL,
                              scores, scores + M->ctx, dst, dst + AD);
                else
                    attn_head(L, q, kv, positions, select ? visible : NULL, scores, dst);
            }
        }
    } else {
        /* heads among the threads: per (token, head) the visible keys (all up to QSA_TOK, else the selected blocks) */
        for (int k = 0; k < nt; k++) {
            const int pos = p0 + k, P = pos + 1;
            const int sel = P > QSA_TOK;
            if (sel) {
                const int nb = P / QSA_R;
                int b0, b1; chunk(nb, 1, t, n, &b0, &b1);
                score_qsa_blocks(M->qsc, M->iqb + k * IH * ID, L->bk, b0, b1);
                BAR;
                T0 select_qsa_positions(M->qsc, M->att, M->qvis, P);
                BAR;
            }
            for (int hh = t; hh < AH; hh += n) {
                const float* q = M->qb + k * AH * 2 * AD + hh * 2 * AD;
                const int kv = hh / (AH / AKV);
                float* sc = M->att + (size_t)(hh + 1) * M->ctx;          /* (row 0 is the selection scratch) */
                float* dst = M->ob + k * AH * AD + hh * AD;
                attn_head(L, q, kv, P, sel ? M->qvis : NULL, sc, dst);
            }
        }
    }
    BAR;
    if (priv) P = pv_quant(M->ob, AH * AD, nt, AH * AD);
    else { quant_par(M, t, n, M->ob, AH * AD); BAR; }
    pjob O = {L->o, 0, 0, M->y, HD, M->ob, AH * AD, P.q, P.dx, P.fx};
    prod(M, t, n, &O, 1);
}

/* ------------------------------------------------------------------ the MoE block */
static void moe(fm_model* M, int t, int n, fm_layer* L) {
    const int nt = M->nt, gemm = GEMM(M);
    const int priv = PRIV(M);
    pbuf P = {M->aq, M->adx, M->afx};               /* (the quantized x the experts gather from) */
    if (priv) P = pv_quant(M->x, HD, nt, HD);
    else {
        if (!gemm) { T0 quant_in(M, 0, M->x, HD, nt, HD); }
        else if (gemm_emin > 1) quant_rows(M, t, n, M->x, HD);   /* (the int8 kernels for experts with few tokens) */
        BAR;
    }
    const int8_t* jq = priv ? P.q : NULL;
    pjob J[3] = {{L->router, 0, 0, M->probs, NE, M->x, HD, jq, P.dx, P.fx}, {L->sg, 0, 0, M->sh, 2 * FF, M->x, HD, jq, P.dx, P.fx},
                 {L->su, 0, 0, M->sh + FF, 2 * FF, M->x, HD, jq, P.dx, P.fx}};
    if (L->router->kind == 32) {                  /* the exact router: fp32 weights and inputs */
        int r0, r1; slice(L->router, t, n, &r0, &r1);
        rows_f32(L->router, r0, r1, nt, M->x, M->probs, NE);
        prod(M, t, n, J + 1, 2);
    } else
        prod(M, t, n, J, 3);
    if (gemm || t == 0) {
        for (int k = gemm ? t : 0; k < nt; k += gemm ? n : 1) {
            float* p = M->probs + k * NE; float mx = -INFINITY;
            for (int e = 0; e < NE; e++) if (p[e] > mx) mx = p[e];
            float s = 0; for (int e = 0; e < NE; e++) { p[e] = expf(p[e] - mx); s += p[e]; }
            for (int e = 0; e < NE; e++) p[e] /= s;
            /* top NSEL (ties: the lower index first, as topk on these values) */
            float tw = 0;
            uint8_t selected[NE] = {0};
            for (int j = 0; j < NSEL; j++) {
                int bi = -1; float bv = -1;
                for (int e = 0; e < NE; e++) {
                    if (!selected[e] && p[e] > bv) { bv = p[e]; bi = e; }
                }
                if (bi >= 0) selected[bi] = 1;
                M->topi[k * NSEL + j] = bi; M->topw[k * NSEL + j] = bv; tw += bv;
            }
            for (int j = 0; j < NSEL; j++) M->topw[k * NSEL + j] /= tw;
            float* sh = M->sh + k * 2 * FF;
            for (int i = 0; i < FF; i++) M->shd[k * FF + i] = silu(sh[i]) * sh[FF + i];
        }
        if (!gemm) quant_in(M, 2, M->shd, FF, nt, FF);
    }
    BAR;
    /* the union of the tokens' experts: each expert's weights are read once for all its tokens (an MTP verify
       window, a prompt chunk); per expert its pairs (token * NSEL + slot) in order */
    T0 {
        static int where[NE];
        for (int e = 0; e < NE; e++) where[e] = -1;
        M->nu = 0;
        for (int i = 0; i < nt * NSEL; i++) {
            const int e = M->topi[i];
            if (where[e] < 0) { where[e] = M->nu; M->ue[M->nu] = e; M->un[M->nu] = 0; M->nu++; }
            M->un[where[e]]++;
        }
        int o = 0;
        for (int u = 0; u < M->nu; u++) { M->up0[u] = o; o += M->un[u]; M->un[u] = 0; }
        for (int i = 0; i < nt * NSEL; i++) { const int u = where[M->topi[i]]; M->plist[M->up0[u] + M->un[u]++] = i; }
        prof_mark(M, 3);
    }
    BAR;
    /* thread-local gather buffers (tokens of one expert) */
    static __thread int8_t* gq; static __thread float *gdx, *gfx, *go, *gu; static __thread int *gxi, *gyi;
    if (!gq) {
        gxi = malloc(sizeof(int) * FM_TMAX); gyi = malloc(sizeof(int) * FM_TMAX);
        /* (an expert has at most one pair per token: at most FM_TMAX pairs) */
        gq = aligned_alloc(64, (size_t)FM_TMAX * HD); gdx = malloc(sizeof(float) * FM_TMAX * HD / 32);
        gfx = malloc(sizeof(float) * FM_TMAX * HD / 32); go = malloc(sizeof(float) * FM_TMAX * HD);
        gu = malloc(sizeof(float) * FM_TMAX * FF);
    }
    /* experts gate and up: each thread its slice of every expert's rows, all of the expert's tokens at once;
       h = silu(g) * u on its own rows */
    if (!gemm && dyn_on && fb_node_of >= 0) {
        const int U = (L->eg->R + 1) / 2, BU = 8;                       /* (K4: pairs; blocks of 16 rows) */
        int u0, u1; node_units(U, fb_node_of, &u0, &u1);
        const int bpe = (u1 - u0 + BU - 1) / BU;
        int last = -1;
        for (int b; (b = dyn_take(6, bpe * M->nu)) >= 0;) {
            const int u = b / bpe, bb = b % bpe;
            const int r0 = (u0 + bb * BU) * 2, r1 = u0 + (bb + 1) * BU < u1 ? (u0 + (bb + 1) * BU) * 2 : (u1 * 2 < L->eg->R ? u1 * 2 : L->eg->R);
            const int e = M->ue[u], m = M->un[u];
            const int* pl = M->plist + M->up0[u];
            if (u != last) {
                for (int i = 0; i < m; i++) {
                    const int k = pl[i] / NSEL;
                    memcpy(gq + (size_t)i * HD, P.q + (size_t)k * HD, HD);
                    memcpy(gdx + (size_t)i * (HD / 32), P.dx + (size_t)k * HD / 32, sizeof(float) * HD / 32);
                    memcpy(gfx + (size_t)i * (HD / 32), P.fx + (size_t)k * HD / 32, sizeof(float) * HD / 32);
                }
                last = u;
            }
            rows_big(L->eg, e, r0, r1, m, gq, gdx, gfx, go, FF);
            rows_big(L->eu, e, r0, r1, m, gq, gdx, gfx, gu, FF);
            for (int i = 0; i < m; i++) {
                float* g = M->eh + (size_t)pl[i] * 2 * FF;
                for (int r = r0; r < r1; r++) g[r] = silu(go[(size_t)i * FF + r]) * gu[(size_t)i * FF + r];
            }
        }
    } else {
        int r0, r1; slice(L->eg, t, n, &r0, &r1);
        for (int u = 0; u < M->nu; u++) {
            const int e = M->ue[u], m = M->un[u];
            const int* pl = M->plist + M->up0[u];
            if (gemm && m >= gemm_emin) {
                for (int i = 0; i < m; i++) gxi[i] = pl[i] / NSEL;
                rows_gemm(L->eg, e, r0, r1, m, M->x, HD, gxi, go, FF, NULL, 0);
                rows_gemm(L->eu, e, r0, r1, m, M->x, HD, gxi, gu, FF, NULL, 0);
                for (int i = 0; i < m; i++) {
                    float* g = M->eh + (size_t)pl[i] * 2 * FF;
                    const float w = M->topw[pl[i]];
                    for (int r = r0; r < r1; r++) g[r] = w * silu(go[(size_t)i * FF + r]) * gu[(size_t)i * FF + r];
                }
                continue;
            }
            for (int i = 0; i < m; i++) {
                const int k = pl[i] / NSEL;
                memcpy(gq + (size_t)i * HD, P.q + (size_t)k * HD, HD);
                memcpy(gdx + (size_t)i * (HD / 32), P.dx + (size_t)k * HD / 32, sizeof(float) * HD / 32);
                memcpy(gfx + (size_t)i * (HD / 32), P.fx + (size_t)k * HD / 32, sizeof(float) * HD / 32);
            }
            rows_big(L->eg, e, r0, r1, m, gq, gdx, gfx, go, FF);
            rows_big(L->eu, e, r0, r1, m, gq, gdx, gfx, gu, FF);
            for (int i = 0; i < m; i++) {
                float* g = M->eh + (size_t)pl[i] * 2 * FF;
                const float w = gemm ? M->topw[pl[i]] : 1.f;      /* (the GEMM path puts the gate weight in eh) */
                for (int r = r0; r < r1; r++) g[r] = w * silu(go[(size_t)i * FF + r]) * gu[(size_t)i * FF + r];
            }
        }
    }
    /* the shared expert's down */
    pjob D = {L->sd, 2, 0, M->tmp, HD, M->shd, FF};
    prod(M, t, n, &D, 1);
    if (!gemm || gemm_emin > 1) for (int pr = t; pr < nt * NSEL; pr += n)
        quant_act(M->eh + (size_t)pr * 2 * FF, FF, M->ehq + (size_t)pr * FF, M->ehdx + (size_t)pr * (FF / 32), M->ehfx + (size_t)pr * (FF / 32));
    BAR;
    /* experts down: each thread its rows of the output; per expert all its tokens at once, weighted into y */
    if (!gemm && dyn_on && fb_node_of >= 0) {           /* (writing: row blocks of the node, every expert in order) */
        const int U = (L->ed->R + 1) / 2, BU = 8;
        int u0, u1; node_units(U, fb_node_of, &u0, &u1);
        const int nblk = (u1 - u0 + BU - 1) / BU;
        for (int b; (b = dyn_take(7, nblk)) >= 0;) {
            const int r0 = (u0 + b * BU) * 2, r1 = u0 + (b + 1) * BU < u1 ? (u0 + (b + 1) * BU) * 2 : (u1 * 2 < L->ed->R ? u1 * 2 : L->ed->R);
            for (int k = 0; k < nt; k++) for (int r = r0; r < r1; r++) M->y[k * HD + r] = 0;
            for (int u = 0; u < M->nu; u++) {
                const int e = M->ue[u], m = M->un[u];
                const int* pl = M->plist + M->up0[u];
                for (int i = 0; i < m; i++) {
                    memcpy(gq + (size_t)i * FF, M->ehq + (size_t)pl[i] * FF, FF);
                    memcpy(gdx + (size_t)i * (FF / 32), M->ehdx + (size_t)pl[i] * (FF / 32), sizeof(float) * FF / 32);
                    memcpy(gfx + (size_t)i * (FF / 32), M->ehfx + (size_t)pl[i] * (FF / 32), sizeof(float) * FF / 32);
                }
                rows_big(L->ed, e, r0, r1, m, gq, gdx, gfx, go, HD);
                for (int i = 0; i < m; i++) {
                    if (nt <= FC_TMAX) {             /* (kept per pair: summed below in each token's slot order) */
                        memcpy(M->pout + (size_t)pl[i] * HD + r0, go + (size_t)i * HD + r0, sizeof(float) * (r1 - r0));
                        continue;
                    }
                    const int k = pl[i] / NSEL;
                    const float w = M->topw[pl[i]];
                    float* yk = M->y + k * HD;
                    for (int r = r0; r < r1; r++) yk[r] += w * go[(size_t)i * HD + r];
                }
            }
            /* a token's experts in its own slot order (the union's order mixes the tokens of a checked window; a
               window must give each token the sums one-token decoding gives it) */
            if (nt <= FC_TMAX)
                for (int k = 0; k < nt; k++) {
                    float* yk = M->y + k * HD;
                    for (int j = 0; j < NSEL; j++) {
                        const float w = M->topw[k * NSEL + j];
                        const float* o = M->pout + (size_t)(k * NSEL + j) * HD;
                        for (int r = r0; r < r1; r++) yk[r] += w * o[r];
                    }
                }
        }
    } else {
        int r0, r1; slice(L->ed, t, n, &r0, &r1);
        for (int k = 0; k < nt; k++) for (int r = r0; r < r1; r++) M->y[k * HD + r] = 0;
        for (int u = 0; u < M->nu; u++) {
            const int e = M->ue[u], m = M->un[u];
            const int* pl = M->plist + M->up0[u];
            if (gemm && m >= gemm_emin) {        /* (the gate weights are already in eh) */
                for (int i = 0; i < m; i++) gyi[i] = pl[i] / NSEL;
                rows_gemm(L->ed, e, r0, r1, m, M->eh, 2 * FF, pl, M->y, HD, gyi, 1);
                continue;
            }
            for (int i = 0; i < m; i++) {
                memcpy(gq + (size_t)i * FF, M->ehq + (size_t)pl[i] * FF, FF);
                memcpy(gdx + (size_t)i * (FF / 32), M->ehdx + (size_t)pl[i] * (FF / 32), sizeof(float) * FF / 32);
                memcpy(gfx + (size_t)i * (FF / 32), M->ehfx + (size_t)pl[i] * (FF / 32), sizeof(float) * FF / 32);
            }
            rows_big(L->ed, e, r0, r1, m, gq, gdx, gfx, go, HD);
            for (int i = 0; i < m; i++) {
                if (!gemm && nt <= FC_TMAX) {
                    memcpy(M->pout + (size_t)pl[i] * HD + r0, go + (size_t)i * HD + r0, sizeof(float) * (r1 - r0));
                    continue;
                }
                const int k = pl[i] / NSEL;
                const float w = gemm ? 1.f : M->topw[pl[i]];
                float* yk = M->y + k * HD;
                for (int r = r0; r < r1; r++) yk[r] += w * go[(size_t)i * HD + r];
            }
        }
        if (!gemm && nt <= FC_TMAX)                  /* (each token's experts in its slot order, as above) */
            for (int k = 0; k < nt; k++) {
                float* yk = M->y + k * HD;
                for (int j = 0; j < NSEL; j++) {
                    const float w = M->topw[k * NSEL + j];
                    const float* o = M->pout + (size_t)(k * NSEL + j) * HD;
                    for (int r = r0; r < r1; r++) yk[r] += w * o[r];
                }
            }
    }
    BAR;
    T0 prof_mark(M, 4);
    T0 {
        for (int k = 0; k < nt; k++) {
            double d = 0; for (int i = 0; i < HD; i++) d += (double)M->x[k * HD + i] * L->shg[i];
            const float sg = sigm((float)d);
            for (int i = 0; i < HD; i++) M->y[k * HD + i] += sg * M->tmp[k * HD + i];
        }
    }
    BAR;
}

/* ------------------------------------------------------------------ n-gram (PLE) embeddings of layer L */
static void decode_row(int ty, const uint8_t* p, int C, float* out) {
    blk_t B; int v = 0;
    while (v < C) {
        const int nb = decode_block(ty, p, &B);
        p += block_bytes(ty);
        for (int i = 0; i < nb; i++) {
            float w;
            if (ty == T_IQ4_NL) w = B.h[0] * lv4[(int)B.c[i]];
            else if (ty == T_IQ4_XS) w = B.d * B.m16[(i / 32) * 2] * lv4[(int)B.c[i]];
            else w = 0;
            out[v + i] = w;
        }
        v += nb;
    }
}

static void ple_lookup(fm_model* M, int k, int64_t l1, int64_t l2) {
    int64_t key = (int64_t)((uint64_t)M->ids[k] * (uint64_t)M->ple_mult[0]);
    for (int ord = 2; ord <= 3; ord++) {
        const int64_t lg = ord == 2 ? l1 : l2;
        key ^= (int64_t)((uint64_t)lg * (uint64_t)M->ple_mult[ord - 1]);
        for (int i = 0; i < 8; i++) {
            const int hh = (ord - 2) * 8 + i;
            int64_t r = key % M->ple_size[hh]; if (r < 0) r += M->ple_size[hh];
            const int64_t row = M->ple_off[hh] + r;
            decode_row(M->ple_ty, M->ple_tab + row * M->ple_rb, PLE_D, M->pemb + k * PLE_ROWS * PLE_D + hh * PLE_D);
        }
    }
}

/* one token's gate: key (in big) and query normed per stream, the gated value gv (into big) and its norm (into hn) */
static void ple_gate(fm_model* M, fm_layer* L, int k) {
    float* key = M->big + (size_t)k * HH;
    float* gvn = M->hn + (size_t)k * HNS;
    float qry[HD];
    for (int s = 0; s < NHC; s++) {
        rms_w(key + s * HD, key + s * HD, L->pnk + s * HD, HD, M->eps);
        rms_w(qry, M->h + (size_t)k * HH + s * HD, L->pnq + s * HD, HD, M->eps);
        double d = 0; for (int i = 0; i < HD; i++) d += (double)key[s * HD + i] * qry[i];
        float g = (float)(d / sqrt((double)HD));
        const float a = fabsf(g) < 1e-6f ? 1e-6f : fabsf(g);
        g = sqrtf(a) * (g > 0 ? 1.f : g < 0 ? -1.f : 0.f);
        const float sg = sigm(g);
        for (int i = 0; i < HD; i++) key[s * HD + i] = sg * M->tmp[(size_t)k * HD + i];
        rms_w(gvn + s * HD, key + s * HD, L->pnc + s * HD, HD, M->eps);
    }
}

static void ple_par(fm_model* M, int t, int n, fm_layer* L) {
    const int nt = M->nt;
    const int64_t eos = M->ple_eos, P1 = M->prev1, P2 = M->prev2;
    for (int k = t; k < nt; k += n) {
        const int64_t p1 = k >= 1 ? M->ids[k - 1] : P1, p2 = k >= 2 ? M->ids[k - 2] : k == 1 ? P1 : P2;
        ple_lookup(M, k, p1, p1 == eos ? eos : p2);
    }
    BAR;
    T0 { M->prev2 = nt >= 2 ? M->ids[nt - 2] : M->prev1; M->prev1 = M->ids[nt - 1]; }
    pjob J[2] = {{L->pk, 0, 0, M->big, HH, M->pemb, PLE_ROWS * PLE_D}, {L->pv, 0, 0, M->tmp, HD, M->pemb, PLE_ROWS * PLE_D}};
    prod(M, t, n, J, 2);
    for (int k = t; k < nt; k += n) ple_gate(M, L, k);
    BAR;
    /* dilated causal conv per channel: taps at t - 9, t - 6, t - 3, t over (the 9 state rows, then the chunk) */
    int a, b; chunk(HH, 8, t, n, &a, &b);
    for (int c = a; c < b; c++) {
        const float* w = L->pconv + c * PLE_K;
        float* st = L->pst + c;
#define SEQ(j) ((j) < 9 ? st[(size_t)(j) * HH] : M->hn[(size_t)((j) - 9) * HNS + c])
        for (int k = 0; k < nt; k++) {
            const float v = w[0] * SEQ(k) + w[1] * SEQ(k + 3) + w[2] * SEQ(k + 6) + w[3] * SEQ(k + 9);
            M->h[(size_t)k * HH + c] += M->big[(size_t)k * HH + c] + silu(v);
        }
        float last[9];
        for (int r = 0; r < 9; r++) last[r] = SEQ(nt + r);
        for (int r = 0; r < 9; r++) st[(size_t)r * HH] = last[r];
#undef SEQ
    }
    BAR;
}

static void ple(fm_model* M, int t, int n, fm_layer* L) {
    const int nt = M->nt;
    if (GEMM(M)) { ple_par(M, t, n, L); return; }
    T0 {
        for (int k = 0; k < nt; k++) {
            const int64_t tok = M->ids[k], eos = M->ple_eos;
            const int64_t l1 = M->prev1, l2 = M->prev1 == eos ? eos : M->prev2;
            int64_t key = (int64_t)((uint64_t)tok * (uint64_t)M->ple_mult[0]);
            for (int ord = 2; ord <= 3; ord++) {
                const int64_t lg = ord == 2 ? l1 : l2;
                key ^= (int64_t)((uint64_t)lg * (uint64_t)M->ple_mult[ord - 1]);
                for (int i = 0; i < 8; i++) {
                    const int hh = (ord - 2) * 8 + i;
                    int64_t r = key % M->ple_size[hh]; if (r < 0) r += M->ple_size[hh];
                    const int64_t row = M->ple_off[hh] + r;
                    decode_row(M->ple_ty, M->ple_tab + row * M->ple_rb, PLE_D, M->pemb + k * PLE_ROWS * PLE_D + hh * PLE_D);
                }
            }
            M->prev2 = M->prev1; M->prev1 = (int)tok;
            if (M->ck_on && k < nt - 1 && k < CK_MAX) { M->ck_prev[k][0] = M->prev1; M->ck_prev[k][1] = M->prev2; }
        }
        if (!GEMM(M)) quant_in(M, 0, M->pemb, PLE_ROWS * PLE_D, nt, PLE_ROWS * PLE_D);
    }
    BAR;
    pjob J[2] = {{L->pk, 0, 0, M->big, HH, M->pemb, PLE_ROWS * PLE_D}, {L->pv, 0, 0, M->tmp, HD, M->pemb, PLE_ROWS * PLE_D}};
    prod(M, t, n, J, 2);
    T0 {
        for (int k = 0; k < nt; k++) {
            float* key = M->big + k * HH; float qry[HH], gv[HH];
            for (int s = 0; s < NHC; s++) {
                rms_w(key + s * HD, key + s * HD, L->pnk + s * HD, HD, M->eps);
                rms_w(qry + s * HD, M->h + k * HH + s * HD, L->pnq + s * HD, HD, M->eps);
            }
            for (int s = 0; s < NHC; s++) {
                double d = 0; for (int i = 0; i < HD; i++) d += (double)key[s * HD + i] * qry[s * HD + i];
                float g = (float)(d / sqrt((double)HD));
                const float a = fabsf(g) < 1e-6f ? 1e-6f : fabsf(g);
                g = sqrtf(a) * (g > 0 ? 1.f : g < 0 ? -1.f : 0.f);
                const float sg = sigm(g);
                for (int i = 0; i < HD; i++) gv[s * HD + i] = sg * M->tmp[k * HD + i];
            }
            float gvn[HH];
            for (int s = 0; s < NHC; s++) rms_w(gvn + s * HD, gv + s * HD, L->pnc + s * HD, HD, M->eps);
            /* dilated causal conv: taps at t - 9, t - 6, t - 3, t (state: the last 9 rows of gvn) */
            for (int c = 0; c < HH; c++) {
                const float* w = L->pconv + c * PLE_K;
                float* st = L->pst + c;                     /* rows [9][HH] */
                const float v = w[0] * st[0] + w[1] * st[3 * HH] + w[2] * st[6 * HH] + w[3] * gvn[c];
                for (int r = 0; r < 8; r++) st[r * HH] = st[(r + 1) * HH];
                st[8 * HH] = gvn[c];
                M->h[k * HH + c] += gv[c] + silu(v);
            }
            if (M->ck_on && k < nt - 1 && k < CK_MAX) memcpy(M->ck_pst + (size_t)k * 9 * HH, L->pst, sizeof(float) * 9 * HH);
        }
    }
    BAR;
}

/* ------------------------------------------------------------------ the forward job */
static void layer(fm_model* M, int t, int n, fm_layer* L) {
    if (L->ple) { T0 prof_mark(M, 5); ple(M, t, n, L); T0 prof_mark(M, 6); }
    mixer(M, t, n, &L->ha, M->x, M->wr + FM_TMAX * NHC);
    T0 prof_mark(M, 0);
    if (L->attn) attn(M, t, n, L); else gdn(M, t, n, L);
    write_h(M, t, n, M->y, M->wr + FM_TMAX * NHC);
    T0 prof_mark(M, L->attn ? 2 : 1);
    BAR;
    mixer(M, t, n, &L->hf, M->x, M->wr + FM_TMAX * NHC);
    T0 prof_mark(M, 0);
    moe(M, t, n, L);
    write_h(M, t, n, M->y, M->wr + FM_TMAX * NHC);
    T0 prof_mark(M, 4);
    BAR;
}

/* the MTP head on nt rows: hc [nt][HH] the main model's streams at positions p (or NULL: the head's own streams of
   its previous call, one row), ids [nt] the tokens at p + 1; its cache rows at M->mtp_pos .. ; leaves the draft
   logits of the last row in dlog and the head's last streams in mtp_h */
static void job_mtp(int t, int n, void* arg) {
    fm_model* M = arg;
    fb_node_of = moe_pool_node_of(t) ? 1 : 0;
    const int nt = M->nt;
    T0 {
        for (int k = 0; k < nt; k++) {
            float row[HD];
            decode_row(M->embd_ty, M->embd + (size_t)M->ids[k] * M->embd_rb, HD, row);
            rms_w(M->x + k * HD, row, M->mtp_nemb, HD, M->eps);
            const float* hc = M->mtp_hc ? M->mtp_hc + (size_t)k * HH : M->mtp_h;
            for (int s = 0; s < NHC; s++) rms_w(M->big + ((size_t)k * NHC + s) * HD, hc + s * HD, M->mtp_nhid + s * HD, HD, M->eps);
        }
        quant_in(M, 0, M->x, HD, nt, HD);
        quant_in(M, 1, M->big, HD, nt * NHC, HD);
    }
    BAR;
    pjob E = {M->mtp_emb, 0, 0, M->tmp, HD};
    prod_rows(M, t, n, &E, 1, nt);
    pjob Hh = {M->mtp_hid, 1, 0, M->ob, HD};
    prod_rows(M, t, n, &Hh, 1, nt * NHC);
    {
        int a, b; chunk(HD, 8, t, n, &a, &b);
        for (int k = 0; k < nt; k++)
            for (int s = 0; s < NHC; s++)
                for (int i = a; i < b; i++) M->h[k * HH + s * HD + i] = M->ob[((size_t)k * NHC + s) * HD + i] + M->tmp[k * HD + i];
    }
    BAR;
    layer(M, t, n, M->l + M->L);
    T0 memcpy(M->mtp_h, M->h + (size_t)(nt - 1) * HH, sizeof(float) * HH);
    mixer(M, t, n, &M->mtp_out, M->x, NULL);
    T0 quant_in(M, 0, M->x + (nt - 1) * HD, HD, 1, HD);
    BAR;
    pjob D = {M->draft, 0, 0, M->dlog, M->draft_n};
    prod_rows(M, t, n, &D, 1, 1);
}

static void job_forward(int t, int n, void* arg) {
    fm_model* M = arg;
    fb_node_of = moe_pool_node_of(t) ? 1 : 0;      /* (before any dynamic block take: every thread must count) */
    const int nt = M->nt;
    T0 {
        M->pt = now_s();
        for (int k = 0; k < nt; k++) {
            float row[HD];
            decode_row(M->embd_ty, M->embd + (size_t)M->ids[k] * M->embd_rb, HD, row);
            for (int s = 0; s < NHC; s++) memcpy(M->h + k * HH + s * HD, row, sizeof(row));
        }
    }
    BAR;
    for (int l = 0; l < M->L; l++) layer(M, t, n, M->l + l);
    if (M->output == FM_STATE_ONLY) return;
    /* The output mixer and head do not update the recurrent state or caches. */
    mixer(M, t, n, &M->out, M->x, NULL);
    const int out_rows = M->output == FM_ALL_LOGITS ? nt : 1;
    T0 quant_in(M, 0, M->x + (nt - out_rows) * HD, HD, out_rows, HD);
    BAR;
    pjob Hd = {M->head, 0, 0, M->logits, M->V};
    prod_rows(M, t, n, &Hd, 1, out_rows);
    T0 prof_mark(M, 5);
    BAR;
}

/* the phase seconds since the last call (8 buckets, see prof_mark), then cleared */
void fm_prof(fm_model* M, double* out) { memcpy(out, M->prof, sizeof(M->prof)); memset(M->prof, 0, sizeof(M->prof)); }

/* ------------------------------------------------------------------ API */
fm_model* fm_new(int L, int ctx, int V, float eps, float rope_base) {
    fm_model* M = calloc(1, sizeof(*M));
    M->L = L; M->ctx = ctx; M->V = V; M->eps = eps;
    const char* packed_gdn = getenv("KINDI_FNCPU_PACKED_GDN");
    M->packed_gdn = !(packed_gdn && strcmp(packed_gdn, "0") == 0);       /* (default on since 0.8) */
    const char* gm = getenv("KINDI_FNCPU_GEMM");
    if (gm && *gm) gemm_min = atoi(gm);
    const char* ge = getenv("KINDI_FNCPU_GEMM_EMIN");
    if (ge && *ge) gemm_emin = atoi(ge);
    const char* g2 = getenv("KINDI_FNCPU_GEMM_2D");
    if (g2 && *g2) gemm_2d = atoi(g2);
    const char* dy = getenv("KINDI_FNCPU_DYN");
    if (dy && *dy) dyn_on = atoi(dy);
    const char* bp = getenv("KINDI_FNCPU_BARPROF");
    fb_prof = bp && strcmp(bp, "1") == 0;
    const char* pe = getenv("KINDI_FNCPU_PRIV");
    if (pe && *pe) priv_on = atoi(pe);
    const char* fb = getenv("KINDI_FNCPU_BARRIER");
    if (fb && strcmp(fb, "flat") == 0) fb_on = 0;
    const char* at = getenv("KINDI_FNCPU_ATTN_TILE");
    if (at && *at) attn_tiled = atoi(at);
    const char* gs = getenv("KINDI_FNCPU_SHORT");
    if (gs && *gs) short_r = atoi(gs);
    M->l = calloc(L + 1, sizeof(fm_layer));          /* (l[L]: the MTP head's layer, when loaded) */
    for (int i = 0; i < ROPE / 2; i++) M->inv_freq[i] = (float)(1.0 / pow((double)rope_base, (2.0 * i) / ROPE));
    const int TM = FM_TMAX;
#define A(p, n) (p = calloc((size_t)(n), sizeof(*(p))))
    A(M->h, TM * HH); A(M->hn, TM * HNS); A(M->x, TM * HD); A(M->y, TM * HD); A(M->tmp, TM * HD); A(M->big, TM * HH);
    A(M->logits, (size_t)FC_TMAX * V); A(M->probs, TM * NE); A(M->topw, TM * NSEL); A(M->topi, TM * NSEL);
    A(M->qkvb, TM * QKVW); A(M->zb, TM * GV * GD); A(M->bb, TM * GV); A(M->ab, TM * GV); A(M->qb, TM * AH * 2 * AD);
    A(M->kb, TM * AKV * AD); A(M->vb, TM * AKV * AD); A(M->iqb, TM * IH * ID); A(M->ikb, TM * ID); A(M->ob, TM * HH);
    A(M->eh, TM * NSEL * 2 * FF); A(M->pout, (size_t)FC_TMAX * NSEL * HD); A(M->sh, TM * 2 * FF); A(M->shd, TM * FF); A(M->lr, TM * LR); A(M->wr, 2 * TM * NHC);
    A(M->att, (size_t)(AH + 1) * ctx); A(M->pemb, TM * PLE_ROWS * PLE_D);
    A(M->part, 256 * TM * NHC); A(M->qsc, ctx / QSA_R + 1); A(M->qvis, ctx);
    const char* query_parallel = getenv("KINDI_FNCPU_QUERY_PARALLEL");
    if (query_parallel && strcmp(query_parallel, "1") == 0) {
        const size_t workers = moe_pool_threads();
        const char* query_pair = getenv("KINDI_FNCPU_QUERY_PAIR");
        M->query_heads = query_pair && strcmp(query_pair, "1") == 0 ? 2 : 1;
        A(M->query_att, workers * M->query_heads * ctx);
        A(M->query_qsc, workers * (ctx / QSA_R + 1));
        A(M->query_qvis, workers * ctx);
        if (!M->query_att || !M->query_qsc || !M->query_qvis) {
            free(M->query_att); free(M->query_qsc); free(M->query_qvis);
            M->query_att = M->query_qsc = NULL;
            M->query_qvis = NULL;
        }
    }
    ACAP = (size_t)TM * HH;
    const int slots = 3;
    M->ehq = aligned_alloc(64, (size_t)TM * NSEL * FF); M->ehdx = calloc((size_t)TM * NSEL * FF / 32, 4);
    M->ehfx = calloc((size_t)TM * NSEL * FF / 32, 4);
    M->aq = aligned_alloc(64, (size_t)slots * ACAP); M->adx = calloc(slots * ACAP / 32, 4); M->afx = calloc(slots * ACAP / 32, 4);
    M->prev1 = M->prev2 = -1;
    return M;
}

int fm_set_t(fm_model* M, int l, int slot, fc_tensor* T) {
    fm_layer* L = l >= 0 ? M->l + l : NULL;
    switch (slot) {
    case S_HA_DOWN: L->ha.down = T; break; case S_HA_UP: L->ha.up = T; break; case S_HA_INJ: L->ha.inject = T; break;
    case S_HF_DOWN: L->hf.down = T; break; case S_HF_UP: L->hf.up = T; break; case S_HF_INJ: L->hf.inject = T; break;
    case S_QKV: L->qkv = T; break; case S_Z: L->z = T; break; case S_BETA: L->beta = T; break;
    case S_ALPHA: L->alpha = T; break; case S_SOUT: L->sout = T; break;
    case S_Q: L->q = T; L->attn = 1; break; case S_K: L->k = T; break; case S_V: L->v = T; break;
    case S_IQ: L->iq = T; break; case S_IK: L->ik = T; break; case S_O: L->o = T; break;
    case S_ROUTER: L->router = T; break; case S_SG: L->sg = T; break; case S_SU: L->su = T; break; case S_SD: L->sd = T; break;
    case S_EG: L->eg = T; break; case S_EU: L->eu = T; break; case S_ED: L->ed = T; break;
    case S_PK: L->pk = T; L->ple = 1; break; case S_PV: L->pv = T; break;
    case S_OUT_DOWN: M->out.down = T; break; case S_OUT_UP: M->out.up = T; break; case S_HEAD: M->head = T; break;
    default: return -1;
    }
    return 0;
}

int fm_set_v(fm_model* M, int l, int slot, const float* v) {
    fm_layer* L = l >= 0 ? M->l + l : NULL;
    switch (slot) {
    case V_HA_NORM: L->ha.norm = v; break; case V_HF_NORM: L->hf.norm = v; break; case V_CONV: L->conv = v; break;
    case V_SSM_A: L->ssm_a = v; break; case V_DT: L->dt_bias = v; break; case V_SSM_NORM: L->ssm_norm = v; break;
    case V_QN: L->qn = v; break; case V_KN: L->kn = v; break; case V_IQN: L->iqn = v; break; case V_IKN: L->ikn = v; break;
    case V_SHG: L->shg = v; break; case V_PNK: L->pnk = v; break; case V_PNQ: L->pnq = v; break; case V_PNC: L->pnc = v; break;
    case V_PCONV: L->pconv = v; break; case V_OUT_NORM: M->out.norm = v; break;
    default: return -1;
    }
    return 0;
}

void fm_set_embd(fm_model* M, const uint8_t* embd, int ty, long rb, const uint8_t* ple, int pty, long prb,
                 const int64_t* mult, const int64_t* off, const int64_t* size, int eos) {
    M->embd = embd; M->embd_ty = ty; M->embd_rb = rb; M->ple_tab = ple; M->ple_ty = pty; M->ple_rb = prb;
    memcpy(M->ple_mult, mult, sizeof(M->ple_mult)); memcpy(M->ple_off, off, sizeof(M->ple_off));
    memcpy(M->ple_size, size, sizeof(M->ple_size)); M->ple_eos = eos;
}

/* the state: zeroed caches (allocated on first use; the GDN states are first touched by their owner threads) */
static void job_state(int t, int n, void* arg) {
    fm_model* M = arg;
    bind_here(1);
    for (int l = 0; l < M->L; l++) {
        fm_layer* L = M->l + l;
        if (!L->attn && L->S) for (int hv = t; hv < GV; hv += n) memset(L->S + (size_t)hv * GD * GD, 0, sizeof(float) * GD * GD);
    }
    bind_here(0);
}
void fm_reset(fm_model* M) {
    for (int l = 0; l < M->L + M->has_mtp; l++) {
        fm_layer* L = M->l + l;
        if (L->attn) {
            if (!L->kc) { L->kc = big_alloc(sizeof(float) * (size_t)M->ctx * AKV * AD); L->vc = big_alloc(sizeof(float) * (size_t)M->ctx * AKV * AD);
                          L->ikc = big_alloc(sizeof(float) * (size_t)M->ctx * ID);
                          if (M->ctx > QSA_TOK) L->bk = big_alloc(sizeof(float) * (size_t)(M->ctx / QSA_R) * ID); }
        } else {
            if (!L->S) { L->S = big_alloc_h(sizeof(float) * GV * GD * GD, 0); L->convst = calloc(3 * QKVW, sizeof(float)); }
            memset(L->convst, 0, sizeof(float) * 3 * QKVW);
        }
        if (L->ple) { if (!L->pst) L->pst = calloc(9 * HH, sizeof(float)); memset(L->pst, 0, sizeof(float) * 9 * HH); }
    }
    moe_pool_run(job_state, M);
    M->pos = 0; M->prev1 = M->prev2 = M->ple_eos; M->mtp_pos = 0;
}

/* Advance nt tokens at the current position. The caller can omit logits for an intermediate chunk. */
int fm_forward(fm_model* M, int nt, const int* ids, int output, float* logits) {
    if (nt < 1 || nt > FM_TMAX || M->pos + nt > M->ctx) return -1;
    if (output != FM_STATE_ONLY && output != FM_LAST_LOGITS && output != FM_ALL_LOGITS) return -1;
    if (output == FM_ALL_LOGITS && nt > FC_TMAX) return -1;
    if (output != FM_STATE_ONLY && !logits) return -1;
    M->nt = nt; M->ids = ids; M->output = output;
    if (fb_n != moe_pool_threads()) fb_setup(moe_pool_threads());
    moe_pool_run(job_forward, M);
    if (output != FM_STATE_ONLY)
        memcpy(logits, M->logits, sizeof(float) * (size_t)M->V * (output == FM_ALL_LOGITS ? nt : 1));
    M->pos += nt;
    return 0;
}
int fm_pos(const fm_model* M) { return M->pos; }

/* ------------------------------------------------------------------ the MTP head and checked windows (API) */
/* the head's own tensors (its decoder layer goes in with fm_set_t / fm_set_v at l = L); draft: the draft vocabulary's
   rows of output.weight, ids: their token ids */
int fm_set_mtp(fm_model* M, fc_tensor* emb, fc_tensor* hid, fc_tensor* odown, fc_tensor* oup, fc_tensor* draft,
               const float* nemb, const float* nhid, const float* onorm, const int* ids, int n) {
    M->mtp_emb = emb; M->mtp_hid = hid; M->mtp_out.down = odown; M->mtp_out.up = oup; M->mtp_out.norm = onorm;
    M->mtp_out.inject = NULL; M->draft = draft; M->mtp_nemb = nemb; M->mtp_nhid = nhid; M->draft_ids = ids;
    M->draft_n = n;
    M->mtp_h = calloc(HH, sizeof(float)); M->dlog = calloc(n, sizeof(float));
    M->has_mtp = 1;
    return 0;
}
/* copy rows [r0, r0 + nr) of the streams after the last layer of the last fm_forward (before the next call) */
int fm_streams(fm_model* M, int r0, int nr, float* out) {
    if (r0 < 0 || nr < 0 || r0 + nr > M->nt) return -1;
    memcpy(out, M->h + (size_t)r0 * HH, sizeof(float) * (size_t)nr * HH);
    return 0;
}
/* the head on nt rows at its positions pos ..: hc [nt][HH] (NULL: one row from its own previous streams), ids the
   tokens at p + 1. Returns the draft (argmax over the draft vocabulary of the last row); logits: optional copy of
   that row's draft-vocabulary logits [draft_n]. The main model's state is untouched (M->h is scratch). */
int fm_mtp(fm_model* M, int nt, const float* hc, const int* ids, int pos, float* logits) {
    if (!M->has_mtp || nt < 1 || nt > FM_TMAX || (!hc && nt != 1) || pos + nt > M->ctx) return -1;
    const int save_pos = M->pos, save_nt = M->nt; const int* save_ids = M->ids; const int save_out = M->output;
    M->nt = nt; M->ids = ids; M->mtp_hc = hc; M->pos = pos;
    if (fb_n != moe_pool_threads()) fb_setup(moe_pool_threads());
    moe_pool_run(job_mtp, M);
    M->pos = save_pos; M->nt = save_nt; M->ids = save_ids; M->output = save_out;
    M->mtp_pos = pos + nt;
    if (logits) memcpy(logits, M->dlog, sizeof(float) * M->draft_n);
    int best = 0;
    for (int i = 1; i < M->draft_n; i++) if (M->dlog[i] > M->dlog[best]) best = i;
    return M->draft_ids[best];
}
/* checked windows: with on, each fm_forward of 2 .. CK_MAX + 1 tokens keeps the states after its early tokens */
int fm_ckpt(fm_model* M, int on) {
    if (on && !M->ck_S) {
        M->ck_S = big_alloc_h(sizeof(float) * (size_t)M->L * CK_MAX * GV * GD * GD, 0);
        M->ck_conv = calloc((size_t)M->L * CK_MAX * 3 * QKVW, sizeof(float));
        M->ck_pst = calloc((size_t)CK_MAX * 9 * HH, sizeof(float));
        if (!M->ck_S || !M->ck_conv || !M->ck_pst) return -1;
    }
    M->ck_on = on;
    return 0;
}
static int rb_keep;
static void job_rollback(int t, int n, void* arg) {
    fm_model* M = arg;
    const int k = rb_keep - 1;
    for (int l = 0; l < M->L; l++) {
        fm_layer* L = M->l + l;
        if (L->attn || !L->S) continue;
        for (int hv = t; hv < GV; hv += n)
            memcpy(L->S + (size_t)hv * GD * GD, M->ck_S + (((size_t)l * CK_MAX + k) * GV + hv) * GD * GD, sizeof(float) * GD * GD);
        int c0, c1; chunk(QKVW, 32, t, n, &c0, &c1);
        const float* ck = M->ck_conv + ((size_t)l * CK_MAX + k) * 3 * QKVW;
        for (int r = 0; r < 3; r++) memcpy(L->convst + r * QKVW + c0, ck + r * QKVW + c0, sizeof(float) * (c1 - c0));
        if (L->ple) {}
    }
    T0 for (int l = 0; l < M->L; l++) if (M->l[l].ple) memcpy(M->l[l].pst, M->ck_pst + (size_t)k * 9 * HH, sizeof(float) * 9 * HH);
}
/* after a checked fm_forward of nt tokens: keep its first keep tokens (1 .. nt) */
int fm_rollback(fm_model* M, int nt, int keep) {
    if (keep < 1 || keep > nt || nt - 1 > CK_MAX || !M->ck_S) return -1;
    if (keep == nt) return 0;
    rb_keep = keep;
    moe_pool_run(job_rollback, M);
    M->prev1 = M->ck_prev[keep - 1][0]; M->prev2 = M->ck_prev[keep - 1][1];
    M->pos -= nt - keep;
    return 0;
}
/* the barrier statistics since the last call: out[0] mean and out[1] max over threads of the cycles spent waiting,
   out[2] barriers per thread (thread 0); then cleared */
void fm_barrier_stats(double* out) {
    const int n = moe_pool_threads();
    double sum = 0, mx = 0;
    for (int t = 0; t < n && t < 256; t++) { sum += fb_wait[t]; if (fb_wait[t] > mx) mx = fb_wait[t]; }
    out[0] = n ? sum / n : 0; out[1] = mx; out[2] = fb_count[0];
    memset(fb_wait, 0, sizeof(fb_wait)); memset(fb_count, 0, sizeof(fb_count));
}
/* the per-site waiting (KINDI_FNCPU_BARPROF=1): out[line] = cycles summed over threads; then cleared */
void fm_barrier_sites(double* out) {
    for (int i = 0; i < FB_SITES; i++) { out[i] = (double)atomic_load(&fb_site_wait[i]); atomic_store(&fb_site_wait[i], 0); }
}
/* bench: microseconds per barrier on the pool (mode 0: the pool's flat barrier, 1: the two-level one) */
static int bb_reps, bb_mode;
static void job_bb(int t, int n, void* arg) {
    (void)arg;
    for (int i = 0; i < bb_reps; i++) { if (bb_mode) fm_barrier(t, n); else moe_pool_barrier(); }
}
double fm_barrier_bench(int mode, int reps) {
    if (fb_n != moe_pool_threads()) fb_setup(moe_pool_threads());
    bb_reps = reps; bb_mode = mode;
    const double a = now_s();
    moe_pool_run(job_bb, NULL);
    return 1e6 * (now_s() - a) / reps;
}
/* the runtime options (tests and A/B runs in one process): the GEMM token threshold, the expert minimum, the short
   matrix rows; a negative value keeps the current setting */
void fm_set_tb(int tb) {
    if (tb >= 2000) { gm_t4 = tb - 2000; return; }     /* (2000 + x: the K4 transposed decode on/off) */
    if (tb >= 1000) { gemm_2d = tb - 1000; return; }
    gm_tb = tb;
}   /* (1000 + x: 2D on/off) */       /* the GEMM token block (fncpu.c rows_gemm; 0: none) */
void fm_set_gemm(int min, int emin, int shortr, int tiled, int barrier) {
    if (barrier >= 0) fb_on = barrier & 1;
    if (barrier >= 0) priv_on = barrier >> 1 & 1;          /* (bit 1: private quantization while writing) */
    if (barrier >= 0) dyn_on = barrier >> 2 & 1;           /* (bit 2: dynamic row blocks while writing) */
    if (tiled >= 0) attn_tiled = tiled;
    if (min >= 0) gemm_min = min;
    if (emin >= 0) gemm_emin = emin;
    if (shortr >= 0) short_r = shortr;
}
