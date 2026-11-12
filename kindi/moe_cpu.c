/* moe_cpu.c - the routed experts of a large MoE model on the host CPUs (AVX2), for the DeepSeek-V4.1-Flash engine.

   Weights: MXFP4 as in the GGUF file, unchanged in value. A block is 32 weights: one E8M0 scale byte (2^(e-127)) and
   16 bytes of E2M1 codes (sign bit + 3 bits: 0, 0.5, 1, 1.5, 2, 3, 4, 6). In the GGUF block, byte j holds weight j in
   the low nibble and weight j + 16 in the high nibble. We keep the codes as they are (4.25 bits per weight); only the
   order of the blocks changes.

   Activations: per 32 values one scale d = amax / 127 (rounded to fp16) and 32 int8 values
   round-to-nearest-even(x * 127 / amax), the same as llama.cpp's CPU Q8_0 path. A block dot product is exact in
   int32: a 16-entry table (pshufb) turns each code into the doubled weight (-12..12, int8), _mm256_maddubs_epi16
   multiplies |w| with the sign-adjusted activation, and _mm256_madd_epi16 sums to int32.
   acc += int32 sum * (weight scale / 2) * (activation scale), in fp32 (8 lanes), one horizontal sum per row.

   Output: every (token, expert) pair gives its own down row values; a token's output row is
   sum over its experts, in its own selection order, of weight * value (fp32, one multiply and one add per expert).
   This depends only on the token's own inputs, so any batch (a prompt chunk, a speculation batch) gives the same
   bits as the tokens one at a time.

   Placement: every expert matrix is split by rows over the NUMA nodes (share of rows = share of memory bandwidth),
   and the rows of a node are in pages bound to that node. The threads of a node read only their own node's memory.
   Gate and up are interleaved in groups of 4 + 4 rows, so one thread makes h = silu(gate) * up for its own rows.
   Down is in groups of 4 rows. (In-place layers read the GGUF rows directly; the page cache placement is then the
   caller's business.)

   Batches: the union of the experts of all tokens is formed, with one token list per expert. Each expert row group
   (21 KB gate/up, 5 KB down) is read from memory once and stays in L1/L2 while the tokens that use it are computed,
   two tokens per pass (the codes are decoded once for both). Any number of tokens per call.

   Threads: one persistent pool, each thread pinned to one core; the steps are separated by spin barriers. Between
   calls the workers spin briefly and then sleep on a futex, so they do not take cores from other thread pools.
   Built on first use with gcc -O3 -march=native (moe_cpu.py, dsv41.py) or by tools/moe_cpu_bench.c. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
#include <linux/futex.h>

#define MOE_MAXNODE 2
#define MOE_MAXTHR 128
#define MOE_MAXSEL 16
#define QK 32
#define GU_ROWS 4                 /* gate rows (and as many up rows) per group */
#define DN_ROWS 4                 /* down rows per group */
#define CHUNK 4                   /* work items (row groups) claimed at a time */
#define SPIN_MAX 20000            /* pause loops in a barrier before a waiting thread yields its core */
#define SPIN_IDLE 4000            /* default pause loops between calls before a worker sleeps on the futex */

/* ---------------------------------------------------------------- memory on one node */
static void* node_alloc(size_t bytes, int node) {
    size_t sz = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void* p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    madvise(p, sz, MADV_HUGEPAGE);
    unsigned long mask = 1ul << node;
    /* MPOL_PREFERRED = 1: the pages go to the node, and to the other node when it is full (MPOL_BIND would call the
       OOM killer on a shared box). Check the placement with numastat. */
    if (syscall(SYS_mbind, p, sz, 1, &mask, sizeof(mask) * 8, 0) != 0) perror("moe_cpu: mbind");
    return p;
}

/* ---------------------------------------------------------------- tables */
static float e8m0_half[256];     /* 2^(e-127) / 2: the codes below are doubled weights */
static int tables_ready = 0;
static const int8_t fp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static const uint8_t fp4u[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};   /* doubled weight + 12 */

static void init_tables(void) {
    for (int e = 0; e < 256; e++) e8m0_half[e] = ldexpf(1.0f, e - 128);
    tables_ready = 1;
}

static inline float fp16_round(float x) { return _cvtsh_ss(_cvtss_sh(x, 0)); }

/* Q8_0-style quantization of n values (n % 32 == 0): q int8, d fp32 holding an fp16 value. Scale d = amax / 127
   (kept as fp16), values round-to-nearest-even(x * 127 / amax): the same rounding as llama.cpp's CPU Q8_0 path and
   as dsv41_cpu.c. */
static void quant_q8(const float* x, int n, int8_t* q, float* d) {
    const __m256 sign = _mm256_set1_ps(-0.f);
    for (int b = 0; b < n / QK; b++) {
        const float* xb = x + b * QK;
        __m256 v0 = _mm256_loadu_ps(xb), v1 = _mm256_loadu_ps(xb + 8), v2 = _mm256_loadu_ps(xb + 16),
               v3 = _mm256_loadu_ps(xb + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                 _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        float amax = _mm_cvtss_f32(m4);
        d[b] = fp16_round(amax / 127.f);
        __m256 id = _mm256_set1_ps(amax != 0.f ? 127.f / amax : 0.f);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, id), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, id), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, id), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, id), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        /* int32 -> int16 -> int8 with the lane order restored */
        __m256i p01 = _mm256_packs_epi32(i0, i1), p23 = _mm256_packs_epi32(i2, i3);
        __m256i p = _mm256_packs_epi16(p01, p23);
        p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(q + b * QK), p);
    }
}

/* per block of 32 quantized values: 12 * their sum (the correction of the unsigned-weight dot products) */
static void qsum12(const int8_t* q, int nb, int32_t* out) {
    const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        __m256i v = _mm256_madd_epi16(_mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*)(q + b * QK))), one16);
        __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e));
        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1));
        out[b] = 12 * _mm_cvtsi128_si32(a);
    }
}

/* ---------------------------------------------------------------- layer layout */
/* Gate/up group g (rows 4g..4g+3 of gate and of up), per block b: 8 scale bytes (g0..g3,u0..u3), then 8 x 16
   code bytes in the same order: 136 bytes per block, nbk blocks per group. Down group (rows 4g..4g+3), per block:
   4 scale bytes + 4 x 16 code bytes = 68 bytes. Groups [0, split) are on node 0, the rest on node 1. */
typedef struct {
    int K, F, E;                 /* hidden size, expert width, number of experts */
    int nbk, nbf;                /* blocks along K (gate/up input), along F (down input) */
    int ngu, ndn;                /* groups per expert: F / 4, K / 4 */
    int gu_split, dn_split;      /* first group on node 1 */
    size_t gu_gbytes, dn_gbytes; /* bytes per group */
    uint8_t* gu[MOE_MAXNODE];    /* [E][groups on node][gu_gbytes] */
    uint8_t* dn[MOE_MAXNODE];
    float clamp;                 /* swiglu limit (0 = none) */
    int direct;                  /* 1: the experts are read in place from the GGUF tensors (no copy, no node split) */
    const uint8_t *dgate, *dup, *ddown;  /* direct: [E][rows][blocks] of 17 bytes */
} moe_layer;

static inline const uint8_t* gu_group(const moe_layer* L, int e, int g) {
    int nd = g >= L->gu_split, base = nd ? L->gu_split : 0, cnt = nd ? L->ngu - L->gu_split : L->gu_split;
    return L->gu[nd] + ((size_t)e * cnt + (g - base)) * L->gu_gbytes;
}
static inline const uint8_t* dn_group(const moe_layer* L, int e, int g) {
    int nd = g >= L->dn_split, base = nd ? L->dn_split : 0, cnt = nd ? L->ndn - L->dn_split : L->dn_split;
    return L->dn[nd] + ((size_t)e * cnt + (g - base)) * L->dn_gbytes;
}

/* ---------------------------------------------------------------- thread pool */
typedef struct {
    int nthr, node_of[MOE_MAXTHR], rank_in_node[MOE_MAXTHR], nthr_node[MOE_MAXNODE], cpu[MOE_MAXTHR];
    float share0;                /* share of rows on node 0 */
    pthread_t tid[MOE_MAXTHR];
    _Atomic unsigned job_gen;
    _Atomic unsigned bar_count, bar_gen;
    _Atomic unsigned done_count;
    _Atomic int sleepers;
    struct { _Atomic long next; long end; char pad[48]; } wq[2][MOE_MAXTHR];  /* per step and thread: own range */
    float* part[MOE_MAXTHR];     /* per thread: down values of one work item, [pairs][groups of the item][DN_ROWS] */
    size_t part_cap;             /* pairs */
    _Atomic int quit;
    /* hyper-thread siblings for prompt chunks (KINDI_MOE_XCPUS): extra threads nthr .. nthr + nx - 1 join only jobs of
       more than 8 tokens (compute-bound, so the second thread of a core fills its idle issue slots); for single
       tokens they sleep (they would only compete for the core and wait in the barriers). Extra thread j is the
       sibling of thread xsib[j]: it steals from that thread's node, starting at its range. */
    int nx, xsib[MOE_MAXTHR];
    pthread_t xtid[MOE_MAXTHR];
    _Atomic unsigned xgen;
} moe_pool;

static moe_pool P;
static long spin_idle = SPIN_IDLE;

/* pause loops an idle worker spins before it sleeps. With the dense part on a GPU the cores have nothing else to
   do between two layers (~1-2 ms); a sleeping core drops its clock and the next call starts slow, so the GPU engine
   sets this to several milliseconds of spinning. */
void moe_pool_set_spin(long pauses) { spin_idle = pauses > 0 ? pauses : SPIN_IDLE; }

/* One call. Pairs (token, expert) are grouped by union expert u: pairs [pstart[u], pstart[u + 1]), tokens
   ascending; the pair index is also the row of h. tpair[t * nsel + s] = the pair of token t's s-th expert. */
typedef struct {
    const moe_layer* L;
    int T, nsel;
    const int8_t* xq; const float* xd;       /* [T][K] quantized inputs */
    int nu; const int* uexp; const int* pstart;
    const int* ptok; const float* pw;        /* [pairs] token and routing weight */
    const int* tpair;                        /* [T][nsel] */
    float* h;                                /* [pairs][F] */
    int8_t* hq; float* hd;                   /* [pairs][F] quantized */
    float* out;                              /* [T][K] */
    int weight_in_h;                         /* 1: the routing weight scales h (DeepSeek reference) */
    int mode;                                /* 0: experts; 1: combine only (moe_combine) */
    const float *cw, *cparts;                /* combine: weights [T][nsel], values [T][nsel][K] */
    const int* crow; const float* calt;      /* combine: (token, slot) values in row crow[] of calt instead (>= 0) */
    float* parts;                            /* not NULL: per (token, slot) row values [T][nsel][K], no combining */
    const int* pslot;                        /* [pairs] slot of each pair (parts mode) */
    int gb, gbd, chunk;                      /* row groups per work item (gate/up, down), items claimed at a time */
    int nact;                                /* threads in this job (the pool, + the siblings for prompt chunks) */
    int fast;                                /* prompt chunk with the fast kernel (fast_rows8) */
    const int32_t* xs; int32_t* hs;          /* fast: 12 * block sums of xq [T][K / 32] and hq [pairs][F / 32] */
    int u12;                                 /* bit-matched kernels with unsigned weights (rows*_u) */
    const int32_t* xl; int32_t* hl;          /* u12: per block 8 lanes of 12 * (4-activation sums), [T][K/32][8] / [pairs][F/32][8] */
} moe_job;

static moe_job J;
/* per-thread phase timing (TSC cycles, profiling): [thread][0 gate/up work, 1 barrier wait, 2 quantize work,
   3 barrier wait, 4 down work, 5 kernel calls inside gate/up] */
unsigned long long moe_ph[MOE_MAXTHR][6];

static void barrier(void) {
    unsigned gen = atomic_load_explicit(&P.bar_gen, memory_order_acquire);
    if (atomic_fetch_add_explicit(&P.bar_count, 1, memory_order_acq_rel) == (unsigned)J.nact - 1) {
        atomic_store_explicit(&P.bar_count, 0, memory_order_relaxed);
        atomic_fetch_add_explicit(&P.bar_gen, 1, memory_order_release);
    } else {
        for (int n = 0; atomic_load_explicit(&P.bar_gen, memory_order_acquire) == gen; n++)
            if (n < SPIN_MAX) _mm_pause(); else sched_yield();
    }
}

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
    return _mm_cvtss_f32(a);
}

/* 32 codes (16 bytes at p) -> 32 int8 doubled weights */
static inline __m256i decode32(const uint8_t* p, __m256i tbl, __m256i m4) {
    __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)p));
    __m256i hi = _mm256_srli_epi16(v, 4);
    return _mm256_shuffle_epi8(tbl, _mm256_and_si256(_mm256_blend_epi32(v, hi, 0xF0), m4));
}

static inline float silu(float g) { return g / (1.f + expf(-g)); }

/* A row group as 8 (or 4) row pointers into blocks of `bs` bytes: the scale byte at so, the 16 code bytes at co.
   Copied layout: bs 136 (68), so = i, co = 8 + 16 i (4 + 16 i). GGUF layout: bs 17, so 0, co 1, a pointer per row. */
typedef struct { const uint8_t* r[8]; int bs; int so[8]; int co[8]; } grp;

static inline grp gu_grp(const moe_layer* L, int e, int g) {
    grp G;
    if (L->direct) {
        size_t rb = (size_t)L->nbk * 17;
        for (int i = 0; i < 8; i++) {
            G.r[i] = (i < 4 ? L->dgate : L->dup) + ((size_t)e * L->F + g * GU_ROWS + (i & 3)) * rb;
            G.so[i] = 0; G.co[i] = 1;
        }
        G.bs = 17;
    } else {
        const uint8_t* p = gu_group(L, e, g);
        for (int i = 0; i < 8; i++) { G.r[i] = p; G.so[i] = i; G.co[i] = 8 + 16 * i; }
        G.bs = 8 + 8 * 16;
    }
    return G;
}

static inline grp dn_grp(const moe_layer* L, int e, int g) {
    grp G;
    if (L->direct) {
        size_t rb = (size_t)L->nbf * 17;
        for (int i = 0; i < 4; i++) { G.r[i] = L->ddown + ((size_t)e * L->K + g * DN_ROWS + i) * rb; G.so[i] = 0; G.co[i] = 1; }
        G.bs = 17;
    } else {
        const uint8_t* p = dn_group(L, e, g);
        for (int i = 0; i < 4; i++) { G.r[i] = p; G.so[i] = i; G.co[i] = 4 + 16 * i; }
        G.bs = 4 + 4 * 16;
    }
    return G;
}

/* Four rows (r0 .. r0 + 3 of the group) for one or two tokens over nb blocks; res[tok * 4 + i]. The per-row,
   per-token operation sequence is the same for one and two tokens, so the bits do not depend on the pairing. */
/* One-group-ahead prefetch: while the first kernel call of a work item walks its row group's blocks, each block also
   prefetches the same block of the next item's group (set in tl_pf), so the next ~22 KB (gate/up) or ~5 KB (down)
   arrive during ~30K cycles of compute instead of being fetched with an 8-block lead. Results are unchanged. */
static __thread const grp* tl_pf;
static __thread int tl_pfrows;
#define PF_BLOCK(b) do { const grp* P_ = tl_pf; if (P_) {                                                       \
        if (P_->bs == 17) { for (int i_ = 0; i_ < tl_pfrows; i_++) _mm_prefetch((const char*)(P_->r[i_] + (size_t)(b) * 17), _MM_HINT_T1); } \
        else { const char* q_ = (const char*)(P_->r[0] + (size_t)(b) * P_->bs); _mm_prefetch(q_, _MM_HINT_T1);  \
               _mm_prefetch(q_ + 64, _MM_HINT_T1); if (P_->bs > 128) _mm_prefetch(q_ + 128, _MM_HINT_T1); } } } while (0)

#define W_LOAD(i)  const __m256i w##i = decode32(G->r[r0 + i] + o + G->co[r0 + i], tbl, m4); \
                   const __m256i a##i = _mm256_abs_epi8(w##i);                                   \
                   const float s##i = e8m0_half[G->r[r0 + i][o + G->so[r0 + i]]]
#define W_DOT(i, x, dx, acc) acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(                       \
                   _mm256_maddubs_epi16(a##i, _mm256_sign_epi8(x, w##i)), ones)), _mm256_set1_ps(s##i * dx), acc)

static inline void rows4_tok1(const grp* G, int r0, int nb, const int8_t* xq, const float* xd, float res[4]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4x2));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        _mm_prefetch((const char*)(G->r[r0] + o + 8 * bs), _MM_HINT_T0);
        PF_BLOCK(b);
        const __m256i x = _mm256_loadu_si256((const __m256i*)(xq + b * QK));
        const float dx = xd[b];
        W_LOAD(0); W_LOAD(1); W_LOAD(2); W_LOAD(3);
        W_DOT(0, x, dx, c0); W_DOT(1, x, dx, c1); W_DOT(2, x, dx, c2); W_DOT(3, x, dx, c3);
    }
    res[0] = hsum8(c0); res[1] = hsum8(c1); res[2] = hsum8(c2); res[3] = hsum8(c3);
}

static inline void rows4_tok2(const grp* G, int r0, int nb, const int8_t* xq, const float* xd, const int8_t* yq,
                              const float* yd, float res[8]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4x2));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0, e0 = c0, e1 = c0, e2 = c0, e3 = c0;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        _mm_prefetch((const char*)(G->r[r0] + o + 8 * bs), _MM_HINT_T0);
        PF_BLOCK(b);
        const __m256i x = _mm256_loadu_si256((const __m256i*)(xq + b * QK));
        const __m256i y = _mm256_loadu_si256((const __m256i*)(yq + b * QK));
        const float dx = xd[b], dy = yd[b];
        W_LOAD(0); W_LOAD(1); W_LOAD(2); W_LOAD(3);
        W_DOT(0, x, dx, c0); W_DOT(1, x, dx, c1); W_DOT(2, x, dx, c2); W_DOT(3, x, dx, c3);
        W_DOT(0, y, dy, e0); W_DOT(1, y, dy, e1); W_DOT(2, y, dy, e2); W_DOT(3, y, dy, e3);
    }
    res[0] = hsum8(c0); res[1] = hsum8(c1); res[2] = hsum8(c2); res[3] = hsum8(c3);
    res[4] = hsum8(e0); res[5] = hsum8(e1); res[6] = hsum8(e2); res[7] = hsum8(e3);
}

/* Two rows (r0, r0 + 1 of the group) for four tokens: the codes of a block are decoded once for the four tokens
   (prompt chunks: the kernel is compute-bound there, and decoding is half of the port-0 work). Per row and token the
   operations are those of rows4_tok1. res[tok * 2 + i]. */
static inline void rows2_tok4(const grp* G, int r0, int nb, const int8_t* const xq[4], const float* const xd[4],
                              float res[8]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4x2));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        _mm_prefetch((const char*)(G->r[r0] + o + 8 * bs), _MM_HINT_T0);
        PF_BLOCK(b);
        W_LOAD(0); W_LOAD(1);
#define T4(t, ca, cb) { const __m256i x = _mm256_loadu_si256((const __m256i*)(xq[t] + b * QK)); const float dx = xd[t][b]; \
                        W_DOT(0, x, dx, ca); W_DOT(1, x, dx, cb); }
        T4(0, c00, c01) T4(1, c10, c11) T4(2, c20, c21) T4(3, c30, c31)
#undef T4
    }
    res[0] = hsum8(c00); res[1] = hsum8(c01); res[2] = hsum8(c10); res[3] = hsum8(c11);
    res[4] = hsum8(c20); res[5] = hsum8(c21); res[6] = hsum8(c30); res[7] = hsum8(c31);
}

/* u12 variants of the three bit-matched kernels (KINDI_MOE_U12=1). The codes decode to doubled weight + 12 (table
   fp4u, 0..24, unsigned), so maddubs(w + 12, x) needs no vpabsb / vpsignb; each int32 lane then subtracts
   12 * (the sum of its 4 activations), a per-token, per-block vector computed once (lane12). Every lane holds the same
   integer as the signed kernels' lane, and the fp32 work is unchanged, so the bits are identical (tools/isa: 25,344
   comparisons on real experts, 0 differences). Freeing the abs registers also removes the accumulator spills of the
   4-row kernel. The software prefetch covers every 64-byte line of the block 8 ahead (and every row in place). */
static inline void lane12(const int8_t* q, int nb, int32_t* out) {
    const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; b++) {
        __m256i v = _mm256_madd_epi16(_mm256_maddubs_epi16(one8, _mm256_loadu_si256((const __m256i*)(q + b * QK))), one16);
        v = _mm256_add_epi32(_mm256_slli_epi32(v, 3), _mm256_slli_epi32(v, 2));            /* 12 * v */
        _mm256_storeu_si256((__m256i*)(out + b * 8), v);
    }
}
#define PF_AHEAD(G, rbase, n, o) do { if ((G)->bs == 17) {                                                           \
        for (int i_ = 0; i_ < (n); i_++) _mm_prefetch((const char*)((G)->r[(rbase) + i_] + (o) + 8 * 17), _MM_HINT_T0); \
    } else { const char* q_ = (const char*)((G)->r[0] + (o) + 8 * (G)->bs);                                              \
        for (int l_ = 0; l_ < (G)->bs; l_ += 64) _mm_prefetch(q_ + l_, _MM_HINT_T0);                                     \
        _mm_prefetch(q_ + (G)->bs - 1, _MM_HINT_T0); } } while (0)
#define U_LOAD(i)  const __m256i w##i = decode32(G->r[r0 + i] + o + G->co[r0 + i], tbl, m4);                          \
                   const float s##i = e8m0_half[G->r[r0 + i][o + G->so[r0 + i]]]
#define U_DOT(i, x, c, dx, acc) acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(            \
                   _mm256_maddubs_epi16(w##i, x), ones), c)), _mm256_set1_ps(s##i * dx), acc)
static inline void rows4_tok1u(const grp* G, int r0, int nb, const int8_t* xq, const float* xd, const int32_t* xl,
                               float res[4]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4u));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        PF_AHEAD(G, r0, 4, o);
        PF_BLOCK(b);
        const __m256i x = _mm256_loadu_si256((const __m256i*)(xq + b * QK));
        const __m256i c = _mm256_loadu_si256((const __m256i*)(xl + b * 8));
        const float dx = xd[b];
        U_LOAD(0); U_LOAD(1); U_LOAD(2); U_LOAD(3);
        U_DOT(0, x, c, dx, c0); U_DOT(1, x, c, dx, c1); U_DOT(2, x, c, dx, c2); U_DOT(3, x, c, dx, c3);
    }
    res[0] = hsum8(c0); res[1] = hsum8(c1); res[2] = hsum8(c2); res[3] = hsum8(c3);
}
static inline void rows4_tok2u(const grp* G, int r0, int nb, const int8_t* xq, const float* xd, const int32_t* xl,
                               const int8_t* yq, const float* yd, const int32_t* yl, float res[8]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4u));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0, e0 = c0, e1 = c0, e2 = c0, e3 = c0;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        PF_AHEAD(G, r0, 4, o);
        PF_BLOCK(b);
        const __m256i x = _mm256_loadu_si256((const __m256i*)(xq + b * QK)), y = _mm256_loadu_si256((const __m256i*)(yq + b * QK));
        const __m256i cx = _mm256_loadu_si256((const __m256i*)(xl + b * 8)), cy = _mm256_loadu_si256((const __m256i*)(yl + b * 8));
        const float dx = xd[b], dy = yd[b];
        U_LOAD(0); U_LOAD(1); U_LOAD(2); U_LOAD(3);
        U_DOT(0, x, cx, dx, c0); U_DOT(1, x, cx, dx, c1); U_DOT(2, x, cx, dx, c2); U_DOT(3, x, cx, dx, c3);
        U_DOT(0, y, cy, dy, e0); U_DOT(1, y, cy, dy, e1); U_DOT(2, y, cy, dy, e2); U_DOT(3, y, cy, dy, e3);
    }
    res[0] = hsum8(c0); res[1] = hsum8(c1); res[2] = hsum8(c2); res[3] = hsum8(c3);
    res[4] = hsum8(e0); res[5] = hsum8(e1); res[6] = hsum8(e2); res[7] = hsum8(e3);
}
static inline void rows2_tok4u(const grp* G, int r0, int nb, const int8_t* const xq[4], const float* const xd[4],
                               const int32_t* const xl[4], float res[8]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4u));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        PF_AHEAD(G, r0, 2, o);
        PF_BLOCK(b);
        U_LOAD(0); U_LOAD(1);
#define T4(t, ca, cb) { const __m256i x = _mm256_loadu_si256((const __m256i*)(xq[t] + b * QK));                          \
                        const __m256i c = _mm256_loadu_si256((const __m256i*)(xl[t] + b * 8)); const float dx = xd[t][b]; \
                        U_DOT(0, x, c, dx, ca); U_DOT(1, x, c, dx, cb); }
        T4(0, c00, c01) T4(1, c10, c11) T4(2, c20, c21) T4(3, c30, c31)
#undef T4
    }
    res[0] = hsum8(c00); res[1] = hsum8(c01); res[2] = hsum8(c10); res[3] = hsum8(c11);
    res[4] = hsum8(c20); res[5] = hsum8(c21); res[6] = hsum8(c30); res[7] = hsum8(c31);
}
static int u12_env = -1;
static int pfb_env = 0;                /* KINDI_MOE_PFB: down-start prefetch before barrier 1 (blocks; 0 = off) */
static int xsteal_env = 0;             /* KINDI_MOE_XSTEAL: decode calls also steal the other node's leftover items */

/* Prompt chunks, fast (default; KINDI_MOE_FASTPF=0 for the bit-matched kernels): 8 rows x 4 tokens. Per block and token, the 8 rows' integer dot products
   (unsigned weights + 12, maddubs, madd, a horizontal-add tree, minus 12 * the block's activation sum) are whole
   int32 sums in one vector, so a block costs one convert and one FMA per token for all 8 rows (the bit-matched kernels
   do 8 per row and token). acc[row] += dot * (weight scale / 2 * activation scale): fewer roundings than the
   bit-matched kernels and a different order, so the bits differ from single steps; deterministic. */
static inline __m256i tree8(__m256i p0, __m256i p1, __m256i p2, __m256i p3, __m256i p4, __m256i p5, __m256i p6,
                            __m256i p7) {
    const __m256i a = _mm256_hadd_epi32(_mm256_hadd_epi32(p0, p1), _mm256_hadd_epi32(p2, p3));   /* r0..3 | r0..3 */
    const __m256i b = _mm256_hadd_epi32(_mm256_hadd_epi32(p4, p5), _mm256_hadd_epi32(p6, p7));   /* r4..7 | r4..7 */
    return _mm256_add_epi32(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31));
}
static inline void fast_rows8(const grp* G, int nb, const int8_t* const xq[4], const float* const xd[4],
                              const int32_t* const xs[4], float res[4][8]) {
    const __m256i tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)fp4u));
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    const int bs = G->bs;
    for (int b = 0; b < nb; b++) {
        const size_t o = (size_t)b * bs;
        _mm_prefetch((const char*)(G->r[0] + o + 8 * bs), _MM_HINT_T0);
        __m256i w[8];
        float sc[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            w[i] = decode32(G->r[i] + o + G->co[i], tbl, m4);
            sc[i] = e8m0_half[G->r[i][o + G->so[i]]];
        }
        const __m256 s8 = _mm256_loadu_ps(sc);
#define FT(t, acc) {                                                                                                \
            const __m256i x = _mm256_loadu_si256((const __m256i*)(xq[t] + b * QK));                                \
            __m256i v = tree8(_mm256_madd_epi16(_mm256_maddubs_epi16(w[0], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[1], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[2], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[3], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[4], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[5], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[6], x), ones),                               \
                              _mm256_madd_epi16(_mm256_maddubs_epi16(w[7], x), ones));                              \
            v = _mm256_sub_epi32(v, _mm256_set1_epi32(xs[t][b]));                                                  \
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), _mm256_mul_ps(s8, _mm256_set1_ps(xd[t][b])), acc); }
        FT(0, acc0) FT(1, acc1) FT(2, acc2) FT(3, acc3)
#undef FT
    }
    _mm256_storeu_ps(res[0], acc0); _mm256_storeu_ps(res[1], acc1);
    _mm256_storeu_ps(res[2], acc2); _mm256_storeu_ps(res[3], acc3);
}

/* h row of one pair from the 4 gate and 4 up values */
static inline void swiglu4(const moe_layer* L, const float* gt4, const float* up4, float w, float* h) {
    for (int i = 0; i < GU_ROWS; i++) {
        float gt = gt4[i], up = up4[i];
        if (L->clamp > 0) {
            if (gt > L->clamp) gt = L->clamp;
            if (up > L->clamp) up = L->clamp;
            if (up < -L->clamp) up = -L->clamp;
        }
        float v = silu(gt) * up;
        h[i] = J.weight_in_h ? v * w : v;
    }
}

/* Work of a step: every thread owns one contiguous range of the items of its node (one stream for the hardware
   prefetcher) and claims CHUNK items at a time from it. When its range is empty it takes chunks from the other
   threads of its node. A thread that loses its core to another process (for example to the cores pinned to a VM)
   so holds up nobody: the others finish its range. */
static void work_ranges(int step, long total, int t0, int nn) {
    for (int r = 0; r < nn; r++) {
        atomic_store_explicit(&P.wq[step][t0 + r].next, total * r / nn, memory_order_relaxed);
        P.wq[step][t0 + r].end = total * (r + 1) / nn;
    }
}
static inline int work_next(int step, int t, int t0, int nn, int* victim, long* a, long* z) {
    const int ch = J.chunk;
    for (; *victim < nn; (*victim)++) {
        int v = t0 + (t - t0 + *victim) % nn;
        long c = atomic_fetch_add_explicit(&P.wq[step][v].next, ch, memory_order_relaxed), e = P.wq[step][v].end;
        if (c < e) { *a = c; *z = c + ch < e ? c + ch : e; return 1; }
    }
    return 0;
}

/* the work of thread t for the current job */
static void run_job(int t) {
    if (J.mode == 1) {                       /* combine: out[t][d] = sum over slots in order of w * value */
        const int K = J.nsel ? (int)(size_t)J.L : 0;
        for (int tk = t; tk < J.T; tk += P.nthr)
            for (int d = 0; d < K; d++) {
                float acc = 0.f;
                for (int s = 0; s < J.nsel; s++) {
                    const int i = tk * J.nsel + s;
                    const float* src = J.crow && J.crow[i] >= 0 ? J.calt + (size_t)J.crow[i] * K : J.cparts + (size_t)i * K;
                    const float v = J.cw[i] * src[d];
                    acc = acc + v;
                }
                J.out[(size_t)tk * K + d] = acc;
            }
        return;
    }
    const moe_layer* L = J.L;
    /* in place too, each node's threads take the row groups of their node (the page cache pages of every expert are
       placed to match by `python -m kindi.dsv41 numa --rowsplit`); a sibling thread acts as its core's thread */
    const int tb = t < P.nthr ? t : P.xsib[t - P.nthr];
    const int nd = P.node_of[tb], rk = P.rank_in_node[tb];
    const int nn = P.nthr_node[nd], t0 = tb - rk;
    const int F = L->F, K = L->K, nbk = L->nbk, nbf = L->nbf, nbx = K / QK, nbh = F / QK;
    unsigned long long tA = __rdtsc(), kin = 0;
    /* step 1: gate/up. Work items = (union expert, group) of this node; the pairs of the expert two at a time. */
    /* pass 0: the items of this thread's node; pass 1 (more than 8 tokens): what is left of the other node's
       (prompt chunks are compute-bound, and the 2:1 row split by bandwidth leaves node 1's threads idle otherwise;
       single tokens are bandwidth-bound and stay node-local) */
    for (int pass = 0; pass < (P.nthr_node[1] && (J.T > 8 || xsteal_env) ? 2 : 1); pass++) {
        const int nd2 = pass ? 1 - nd : nd, t02 = nd2 ? P.nthr_node[0] : 0, nn2 = P.nthr_node[nd2];
        const int tt = pass ? t02 : tb;              /* other node: start with its first thread's range */
        const int g0 = nd2 ? L->gu_split : 0, g1 = nd2 ? L->ngu : L->gu_split, gb = J.gb, nb = (g1 - g0 + gb - 1) / gb;
        int vic = 0;
        for (long a, z; work_next(0, tt, t02, nn2, &vic, &a, &z);)
        for (long it = a; it < z; it++) {
            const int u = (int)(it / nb), ga = g0 + (int)(it % nb) * gb, gz = ga + gb < g1 ? ga + gb : g1, e = J.uexp[u];
            const int ps = J.pstart[u], pe = J.pstart[u + 1];
            /* the next row group streams in during this one's first kernel call (the next item's first group after
               the last group of the block) */
            grp PG;
#define GU_PF(g) do { if ((g) + 1 < gz) { PG = gu_grp(L, e, (g) + 1); tl_pf = &PG; tl_pfrows = 8; }              \
                      else if (it + 1 < (long)J.nu * nb) { PG = gu_grp(L, J.uexp[(it + 1) / nb], g0 + (int)((it + 1) % nb) * gb); \
                                                          tl_pf = &PG; tl_pfrows = 8; } else tl_pf = NULL; } while (0)
            int p = ps;
            if (J.fast) {                    /* 4 tokens at a time (a short group repeats its last token) */
                for (; p < pe; p += 4) {
                    const int8_t* xs4[4]; const float* ds4[4]; const int32_t* ss4[4];
                    for (int i = 0; i < 4; i++) {
                        const size_t tk = J.ptok[p + i < pe ? p + i : pe - 1];
                        xs4[i] = J.xq + tk * K; ds4[i] = J.xd + tk * nbx; ss4[i] = J.xs + tk * nbx;
                    }
                    for (int g = ga; g < gz; g++) {
                        const grp G = gu_grp(L, e, g);
                        float r[4][8];
                        fast_rows8(&G, nbk, xs4, ds4, ss4, r);
                        for (int i = 0; i < 4 && p + i < pe; i++)
                            swiglu4(L, r[i], r[i] + 4, J.pw[p + i], J.h + (size_t)(p + i) * F + g * GU_ROWS);
                    }
                }
            }
            for (; p + 3 < pe; p += 4) {
                const int8_t* xs4[4]; const float* ds4[4]; const int32_t* xl4[4];
                for (int i = 0; i < 4; i++) { xs4[i] = J.xq + (size_t)J.ptok[p + i] * K; ds4[i] = J.xd + (size_t)J.ptok[p + i] * nbx;
                                              xl4[i] = J.xl + (size_t)J.ptok[p + i] * nbx * 8; }
                for (int g = ga; g < gz; g++) {
                    const grp G = gu_grp(L, e, g);
                    if (p == ps) GU_PF(g); else tl_pf = NULL;
                    float r[4][8];                           /* gate rows 0-1, 2-3, up rows 4-5, 6-7; [tok * 2 + i] */
                    const unsigned long long k0 = __rdtsc();
                    if (J.u12) {
                        rows2_tok4u(&G, 0, nbk, xs4, ds4, xl4, r[0]); tl_pf = NULL; rows2_tok4u(&G, 2, nbk, xs4, ds4, xl4, r[1]);
                        rows2_tok4u(&G, 4, nbk, xs4, ds4, xl4, r[2]); rows2_tok4u(&G, 6, nbk, xs4, ds4, xl4, r[3]);
                    } else {
                    rows2_tok4(&G, 0, nbk, xs4, ds4, r[0]); tl_pf = NULL; rows2_tok4(&G, 2, nbk, xs4, ds4, r[1]);
                    rows2_tok4(&G, 4, nbk, xs4, ds4, r[2]); rows2_tok4(&G, 6, nbk, xs4, ds4, r[3]);
                    }
                    kin += __rdtsc() - k0;
                    for (int i = 0; i < 4; i++) {
                        const float gr[4] = {r[0][i * 2], r[0][i * 2 + 1], r[1][i * 2], r[1][i * 2 + 1]};
                        const float ur[4] = {r[2][i * 2], r[2][i * 2 + 1], r[3][i * 2], r[3][i * 2 + 1]};
                        swiglu4(L, gr, ur, J.pw[p + i], J.h + (size_t)(p + i) * F + g * GU_ROWS);
                    }
                }
            }
            for (; p + 1 < pe; p += 2) {
                const int ta = J.ptok[p], tb = J.ptok[p + 1];
                const int8_t *xa = J.xq + (size_t)ta * K, *xb = J.xq + (size_t)tb * K;
                const float *da = J.xd + (size_t)ta * nbx, *db = J.xd + (size_t)tb * nbx;
                for (int g = ga; g < gz; g++) {
                    const grp G = gu_grp(L, e, g);
                    if (p == ps) GU_PF(g); else tl_pf = NULL;
                    float gr[8], ur[8];
                    if (J.u12) {
                        const int32_t *la = J.xl + (size_t)ta * nbx * 8, *lb = J.xl + (size_t)tb * nbx * 8;
                        rows4_tok2u(&G, 0, nbk, xa, da, la, xb, db, lb, gr); tl_pf = NULL;
                        rows4_tok2u(&G, 4, nbk, xa, da, la, xb, db, lb, ur);
                    } else {
                    rows4_tok2(&G, 0, nbk, xa, da, xb, db, gr); tl_pf = NULL;
                    rows4_tok2(&G, 4, nbk, xa, da, xb, db, ur);
                    }
                    swiglu4(L, gr, ur, J.pw[p], J.h + (size_t)p * F + g * GU_ROWS);
                    swiglu4(L, gr + 4, ur + 4, J.pw[p + 1], J.h + (size_t)(p + 1) * F + g * GU_ROWS);
                }
            }
            if (p < pe) {
                const int ta = J.ptok[p];
                for (int g = ga; g < gz; g++) {
                    const grp G = gu_grp(L, e, g);
                    if (p == ps) GU_PF(g); else tl_pf = NULL;
                    float gr[4], ur[4];
                    if (J.u12) {
                        const int32_t* la = J.xl + (size_t)ta * nbx * 8;
                        rows4_tok1u(&G, 0, nbk, J.xq + (size_t)ta * K, J.xd + (size_t)ta * nbx, la, gr); tl_pf = NULL;
                        rows4_tok1u(&G, 4, nbk, J.xq + (size_t)ta * K, J.xd + (size_t)ta * nbx, la, ur);
                    } else {
                    rows4_tok1(&G, 0, nbk, J.xq + (size_t)ta * K, J.xd + (size_t)ta * nbx, gr); tl_pf = NULL;
                    rows4_tok1(&G, 4, nbk, J.xq + (size_t)ta * K, J.xd + (size_t)ta * nbx, ur);
                    }
                    swiglu4(L, gr, ur, J.pw[p], J.h + (size_t)p * F + g * GU_ROWS);
                }
            }
#undef GU_PF
        }
    }
    unsigned long long tB = __rdtsc();
    /* KINDI_MOE_PFB=n: before waiting in the barrier, prefetch the first n blocks of this thread's first down item
       (every union expert's rows), so the down step starts on cached data instead of a cold miss per stream */
    if (pfb_env > 0 && J.T <= 8) {
        const long a = atomic_load_explicit(&P.wq[1][tb].next, memory_order_relaxed);
        const int g0 = nd ? L->dn_split : 0, g1 = nd ? L->ndn : L->dn_split, g = g0 + (int)a * J.gbd;
        if (a < P.wq[1][tb].end && g < g1)
            for (int u = 0; u < J.nu; u++) {
                const grp G = dn_grp(L, J.uexp[u], g);
                for (int b = 0; b < pfb_env && b < nbf; b++) {
                    if (G.bs == 17) { for (int i = 0; i < DN_ROWS; i++) _mm_prefetch((const char*)(G.r[i] + (size_t)b * 17), _MM_HINT_T0); }
                    else { const char* q = (const char*)(G.r[0] + (size_t)b * G.bs);
                           for (int l = 0; l < G.bs; l += 64) _mm_prefetch(q + l, _MM_HINT_T0);
                           _mm_prefetch(q + G.bs - 1, _MM_HINT_T0); }
                }
            }
    }
    barrier();
    unsigned long long tC = __rdtsc();
    /* step 2: quantize h, split over all threads by pair */
    {
        const int np = J.pstart[J.nu];
        for (int i = t; i < np; i += J.nact)
        {
            quant_q8(J.h + (size_t)i * F, F, J.hq + (size_t)i * F, J.hd + (size_t)i * nbh);
            if (J.fast) qsum12(J.hq + (size_t)i * F, nbh, J.hs + (size_t)i * nbh);
            if (J.u12) lane12(J.hq + (size_t)i * F, nbh, J.hl + (size_t)i * nbh * 8);
        }
    }
    unsigned long long tD = __rdtsc();
    barrier();
    unsigned long long tE = __rdtsc();
    /* step 3: down. Work items = row groups of this node. Per row group: every union expert's group is read once
       and applied to its pairs (values into this thread's part buffer), then each token adds its experts' values
       in its own selection order. */
    for (int pass = 0; pass < (P.nthr_node[1] && (J.T > 8 || xsteal_env) ? 2 : 1); pass++) {
        const int nd2 = pass ? 1 - nd : nd, t02 = nd2 ? P.nthr_node[0] : 0, nn2 = P.nthr_node[nd2];
        const int tt = pass ? t02 : tb;
        const int g0 = nd2 ? L->dn_split : 0, g1 = nd2 ? L->ndn : L->dn_split, gbd = J.gbd;
        float* part = P.part[t];
        int vic = 0;
        for (long a, z; work_next(1, tt, t02, nn2, &vic, &a, &z);)
        for (long it = a; it < z; it++) {
            const int ga = g0 + (int)it * gbd, gz = ga + gbd < g1 ? ga + gbd : g1;
            for (int u = 0; u < J.nu; u++) {
                const int ps = J.pstart[u], pe = J.pstart[u + 1];
                grp PD;
#define DN_PF(g) do { if ((g) + 1 < gz) { PD = dn_grp(L, J.uexp[u], (g) + 1); tl_pf = &PD; tl_pfrows = 4; }       \
                      else if (u + 1 < J.nu) { PD = dn_grp(L, J.uexp[u + 1], ga); tl_pf = &PD; tl_pfrows = 4; }     \
                      else tl_pf = NULL; } while (0)
#define PART(q, g) (part + ((size_t)(q) * gbd + ((g) - ga)) * DN_ROWS)
                int p = ps;
                if (J.fast) {
                    for (; p < pe; p += 4) {
                        const int8_t* hs4[4]; const float* ds4[4]; const int32_t* ss4[4];
                        for (int i = 0; i < 4; i++) {
                            const int q = p + i < pe ? p + i : pe - 1;
                            hs4[i] = J.hq + (size_t)q * F; ds4[i] = J.hd + (size_t)q * nbh; ss4[i] = J.hs + (size_t)q * nbh;
                        }
                        for (int g = ga; g < gz; g += 2) {
                            const grp A = dn_grp(L, J.uexp[u], g), B = dn_grp(L, J.uexp[u], g + 1 < gz ? g + 1 : g);
                            grp G8 = A;
                            for (int i = 0; i < 4; i++) { G8.r[4 + i] = B.r[i]; G8.so[4 + i] = B.so[i]; G8.co[4 + i] = B.co[i]; }
                            float r[4][8];
                            fast_rows8(&G8, nbf, hs4, ds4, ss4, r);
                            for (int i = 0; i < 4 && p + i < pe; i++) {
                                memcpy(PART(p + i, g), r[i], DN_ROWS * sizeof(float));
                                if (g + 1 < gz) memcpy(PART(p + i, g + 1), r[i] + 4, DN_ROWS * sizeof(float));
                            }
                        }
                    }
                }
                for (; p + 3 < pe; p += 4) {
                    const int8_t* hs4[4]; const float* ds4[4]; const int32_t* hl4[4];
                    for (int i = 0; i < 4; i++) { hs4[i] = J.hq + (size_t)(p + i) * F; ds4[i] = J.hd + (size_t)(p + i) * nbh;
                                                  hl4[i] = J.hl + (size_t)(p + i) * nbh * 8; }
                    for (int g = ga; g < gz; g++) {
                        const grp G = dn_grp(L, J.uexp[u], g);
                        if (p == ps) DN_PF(g); else tl_pf = NULL;
                        float r0[8], r1[8];
                        if (J.u12) { rows2_tok4u(&G, 0, nbf, hs4, ds4, hl4, r0); tl_pf = NULL; rows2_tok4u(&G, 2, nbf, hs4, ds4, hl4, r1); }
                        else { rows2_tok4(&G, 0, nbf, hs4, ds4, r0); tl_pf = NULL; rows2_tok4(&G, 2, nbf, hs4, ds4, r1); }
                        for (int i = 0; i < 4; i++) {
                            float* dst = PART(p + i, g);
                            dst[0] = r0[i * 2]; dst[1] = r0[i * 2 + 1]; dst[2] = r1[i * 2]; dst[3] = r1[i * 2 + 1];
                        }
                    }
                }
                for (; p + 1 < pe; p += 2)
                    for (int g = ga; g < gz; g++) {
                        const grp G = dn_grp(L, J.uexp[u], g);
                        if (p == ps) DN_PF(g); else tl_pf = NULL;
                        float r[8];
                        if (J.u12)
                            rows4_tok2u(&G, 0, nbf, J.hq + (size_t)p * F, J.hd + (size_t)p * nbh, J.hl + (size_t)p * nbh * 8,
                                        J.hq + (size_t)(p + 1) * F, J.hd + (size_t)(p + 1) * nbh, J.hl + (size_t)(p + 1) * nbh * 8, r);
                        else
                        rows4_tok2(&G, 0, nbf, J.hq + (size_t)p * F, J.hd + (size_t)p * nbh, J.hq + (size_t)(p + 1) * F,
                                   J.hd + (size_t)(p + 1) * nbh, r);
                        memcpy(PART(p, g), r, 4 * sizeof(float));
                        memcpy(PART(p + 1, g), r + 4, 4 * sizeof(float));
                    }
                if (p < pe)
                    for (int g = ga; g < gz; g++) {
                        const grp G = dn_grp(L, J.uexp[u], g);
                        if (p == ps) DN_PF(g); else tl_pf = NULL;
                        if (J.u12) rows4_tok1u(&G, 0, nbf, J.hq + (size_t)p * F, J.hd + (size_t)p * nbh, J.hl + (size_t)p * nbh * 8, PART(p, g));
                        else rows4_tok1(&G, 0, nbf, J.hq + (size_t)p * F, J.hd + (size_t)p * nbh, PART(p, g));
                    }
#undef DN_PF
            }
            for (int g = ga; g < gz; g++) {
                if (J.parts) {
                    const int np = J.pstart[J.nu];
                    for (int q = 0; q < np; q++)
                        memcpy(J.parts + ((size_t)J.ptok[q] * J.nsel + J.pslot[q]) * K + g * DN_ROWS, PART(q, g),
                               DN_ROWS * sizeof(float));
                    continue;
                }
                for (int tk = 0; tk < J.T; tk++) {
                    float acc[DN_ROWS] = {0.f, 0.f, 0.f, 0.f};
                    for (int s = 0; s < J.nsel; s++) {
                        const int q = J.tpair[tk * J.nsel + s];
                        const float w = J.weight_in_h ? 1.f : J.pw[q];
                        const float* pv = PART(q, g);
                        for (int i = 0; i < DN_ROWS; i++) { float v = w * pv[i]; acc[i] = acc[i] + v; }
                    }
                    memcpy(J.out + (size_t)tk * K + g * DN_ROWS, acc, sizeof(acc));
                }
            }
#undef PART
        }
    }
    unsigned long long tF = __rdtsc();
    moe_ph[t][0] += tB - tA; moe_ph[t][1] += tC - tB; moe_ph[t][2] += tD - tC; moe_ph[t][3] += tE - tD;
    moe_ph[t][4] += tF - tE; moe_ph[t][5] += kin;
}

static long futex(_Atomic unsigned* a, int op, unsigned v) { return syscall(SYS_futex, a, op, v, NULL, NULL, 0); }

static void* worker(void* arg) {
    int t = (int)(intptr_t)arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P.cpu[t], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    unsigned seen = 0;
    for (;;) {
        unsigned g; long spins = 0;
        while ((g = atomic_load_explicit(&P.job_gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&P.quit, memory_order_relaxed)) return NULL;
            if (++spins < spin_idle) { _mm_pause(); continue; }
            atomic_fetch_add(&P.sleepers, 1);
            if (atomic_load_explicit(&P.job_gen, memory_order_seq_cst) == seen && !atomic_load(&P.quit))
                futex(&P.job_gen, FUTEX_WAIT_PRIVATE, seen);
            atomic_fetch_sub(&P.sleepers, 1);
            spins = 0;
        }
        seen = g;
        if (atomic_load(&P.quit)) return NULL;
        run_job(t);
        atomic_fetch_add_explicit(&P.done_count, 1, memory_order_acq_rel);
    }
}

static void* xworker(void* arg) {
    int t = (int)(intptr_t)arg;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P.cpu[t], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    unsigned seen = 0;
    for (;;) {
        unsigned g; long spins = 0;
        while ((g = atomic_load_explicit(&P.xgen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&P.quit, memory_order_relaxed)) return NULL;
            if (++spins < 2000) { _mm_pause(); continue; }   /* sleep soon: decode must not share the core */
            futex(&P.xgen, FUTEX_WAIT_PRIVATE, seen);
            spins = 0;
        }
        seen = g;
        if (atomic_load(&P.quit)) return NULL;
        run_job(t);
        atomic_fetch_add_explicit(&P.done_count, 1, memory_order_acq_rel);
    }
}

/* cpus: the cores of node 0 then node 1; n0, n1 = how many. share0 = share of rows on node 0 (bandwidth share).
   Thread 0 is the calling thread (it must run on cpus[0]). */
int moe_pool_init(const int* cpus, int n0, int n1, float share0) {
    if (!tables_ready) init_tables();
    if (n0 + n1 > MOE_MAXTHR || n0 < 1) return -1;
    memset(&P, 0, sizeof(P));
    P.nthr = n0 + n1; P.nthr_node[0] = n0; P.nthr_node[1] = n1; P.share0 = n1 ? share0 : 1.f;
    for (int t = 0; t < P.nthr; t++) {
        P.cpu[t] = cpus[t]; P.node_of[t] = t >= n0; P.rank_in_node[t] = t < n0 ? t : t - n0;
    }
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(P.cpu[0], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    for (int t = 1; t < P.nthr; t++) pthread_create(&P.tid[t], NULL, worker, (void*)(intptr_t)t);
    /* KINDI_MOE_XCPUS: the siblings, one per pool thread in the pool's order (e.g. 40-79 for cpus 0-39); "x" skips
       a pool thread (its sibling stays free, e.g. for the GPU copy threads: 40-57,x,x,60-79) */
    const char* xs = getenv("KINDI_MOE_XCPUS");
    if (xs && *xs) {
        int n = 0, k = 0;                    /* pool thread (slot), extra threads made */
        while (*xs && n < P.nthr) {
            if (*xs == 'x') { n++; xs++; if (*xs == ',') xs++; continue; }
            char* e; long a = strtol(xs, &e, 10), b = a;
            if (e == xs) break;
            if (*e == '-') b = strtol(e + 1, &e, 10);
            for (long c = a; c <= b && n < P.nthr && P.nthr + k < MOE_MAXTHR; c++, n++, k++) {
                const int t = P.nthr + k;
                P.cpu[t] = (int)c; P.xsib[k] = n; P.node_of[t] = P.node_of[n]; P.rank_in_node[t] = P.rank_in_node[n];
            }
            xs = *e == ',' ? e + 1 : e;
        }
        P.nx = k;
        for (int j = 0; j < P.nx; j++) pthread_create(&P.xtid[j], NULL, xworker, (void*)(intptr_t)(P.nthr + j));
        if (P.nx) fprintf(stderr, "moe_cpu: %d sibling threads for prompt chunks\n", P.nx);
    }
    return 0;
}

void moe_pool_stop(void) {
    atomic_store(&P.quit, 1);
    atomic_fetch_add(&P.job_gen, 1);
    futex(&P.job_gen, FUTEX_WAKE_PRIVATE, 0x7fffffff);
    for (int t = 1; t < P.nthr; t++) pthread_join(P.tid[t], NULL);
    if (P.nx) {
        atomic_fetch_add(&P.xgen, 1); futex(&P.xgen, FUTEX_WAKE_PRIVATE, 0x7fffffff);
        for (int j = 0; j < P.nx; j++) pthread_join(P.xtid[j], NULL);
        P.nx = 0;
    }
}

moe_layer* moe_layer_new(int K, int F, int E, float clamp) {
    moe_layer* L = calloc(1, sizeof(*L));
    L->K = K; L->F = F; L->E = E; L->clamp = clamp;
    L->nbk = K / QK; L->nbf = F / QK; L->ngu = F / GU_ROWS; L->ndn = K / DN_ROWS;
    int n1 = P.nthr_node[1] > 0;
    L->gu_split = n1 ? (int)lroundf(L->ngu * P.share0) : L->ngu;
    L->dn_split = n1 ? (int)lroundf(L->ndn * P.share0) : L->ndn;
    L->gu_gbytes = (size_t)L->nbk * (8 + 8 * 16);
    L->dn_gbytes = (size_t)L->nbf * (4 + 4 * 16);
    L->gu[0] = node_alloc((size_t)E * L->gu_split * L->gu_gbytes, 0);
    L->dn[0] = node_alloc((size_t)E * L->dn_split * L->dn_gbytes, 0);
    if (n1) {
        L->gu[1] = node_alloc((size_t)E * (L->ngu - L->gu_split) * L->gu_gbytes, 1);
        L->dn[1] = node_alloc((size_t)E * (L->ndn - L->dn_split) * L->dn_gbytes, 1);
    }
    return L;
}

/* A layer read in place from GGUF expert tensors (for example an mmap of the model file). gate, up: [E][F][K/32]
   blocks, down: [E][K][F/32] blocks. */
moe_layer* moe_layer_direct(int K, int F, int E, float clamp, const uint8_t* gate, const uint8_t* up, const uint8_t* down) {
    moe_layer* L = calloc(1, sizeof(*L));
    L->K = K; L->F = F; L->E = E; L->clamp = clamp; L->direct = 1;
    L->nbk = K / QK; L->nbf = F / QK; L->ngu = F / GU_ROWS; L->ndn = K / DN_ROWS;
    int n1 = P.nthr_node[1] > 0;
    L->gu_split = n1 ? (int)lroundf(L->ngu * P.share0) : L->ngu;
    L->dn_split = n1 ? (int)lroundf(L->ndn * P.share0) : L->ndn;
    L->gu_gbytes = (size_t)L->nbk * 17 * 8; L->dn_gbytes = (size_t)L->nbf * 17 * 4;
    L->dgate = gate; L->dup = up; L->ddown = down;
    return L;
}

/* rows of each node in an in-place layer: gate/up rows [0, gu_rows) and down rows [0, dn_rows) on node 0 */
void moe_layer_split_rows(const moe_layer* L, int* gu_rows, int* dn_rows) {
    *gu_rows = L->gu_split * GU_ROWS; *dn_rows = L->dn_split * DN_ROWS;
}

size_t moe_layer_bytes(const moe_layer* L) {
    return (size_t)L->E * ((size_t)L->ngu * L->gu_gbytes + (size_t)L->ndn * L->dn_gbytes);
}

/* Copy expert e from GGUF MXFP4 rows (17-byte blocks, row-major [rows][cols/32]) into the layer.
   gate, up: F rows of K; down: K rows of F. The pages are touched by the caller's thread; mbind keeps them on
   their node. */
void moe_layer_set_expert(moe_layer* L, int e, const uint8_t* gate, const uint8_t* up, const uint8_t* down) {
    for (int g = 0; g < L->ngu; g++) {
        uint8_t* G = (uint8_t*)gu_group(L, e, g);
        for (int b = 0; b < L->nbk; b++) {
            uint8_t* B = G + (size_t)b * (8 + 8 * 16);
            for (int i = 0; i < 8; i++) {
                const uint8_t* src = (i < 4 ? gate : up) + ((size_t)(g * GU_ROWS + (i & 3)) * L->nbk + b) * 17;
                B[i] = src[0];
                memcpy(B + 8 + 16 * i, src + 1, 16);
            }
        }
    }
    for (int g = 0; g < L->ndn; g++) {
        uint8_t* G = (uint8_t*)dn_group(L, e, g);
        for (int b = 0; b < L->nbf; b++) {
            uint8_t* B = G + (size_t)b * (4 + 4 * 16);
            for (int i = 0; i < 4; i++) {
                const uint8_t* src = down + ((size_t)(g * DN_ROWS + i) * L->nbf + b) * 17;
                B[i] = src[0];
                memcpy(B + 4 + 16 * i, src + 1, 16);
            }
        }
    }
}

/* Routed experts for T tokens (any number): x [T][K]; ids, wts [T][nsel]; out [T][K] (overwritten, without the
   shared expert). weight_in_h = 1 scales h by the routing weight before down (as the DeepSeek reference), 0 scales
   the down values. Returns the number of distinct experts read. */
static struct {
    size_t tcap, pcap; int ecap;
    int8_t* xq; float* xd;
    int *uexp, *pstart, *ptok, *tpair, *umap, *cnt, *pslot; float* pw;
    float* h; int8_t* hq; float* hd;
    int32_t *xs, *hs;
    int32_t *xl, *hl;
    int F, K;
} S;

static int moe_run(const moe_layer* L, int T, const float* x, int nsel, const int* ids, const float* wts, float* out,
                   int weight_in_h, const uint8_t* skip, float* parts);
static struct { long long calls, u12, experts, pairs, bytes, cycles, end_ns; } ST;

int moe_forward(const moe_layer* L, int T, const float* x, int nsel, const int* ids, const float* wts, float* out,
                int weight_in_h) {
    return moe_run(L, T, x, nsel, ids, wts, out, weight_in_h, NULL, NULL);
}

/* The row values of every (token, slot) not marked in skip [T][nsel], into parts [T][nsel][K] (the caller combines
   them: out = sum over slots in order of weight * value, as moe_forward does). Experts used only by skipped slots
   are not read. Used when some experts are computed elsewhere (a GPU cache of hot experts). */
/* out [T][K] = for each token, sum over its nsel slots in order of w * parts value (the combine of moe_forward, on the
   pool); for row values computed partly elsewhere */
int moe_combine_alt(int T, int nsel, const float* w, const float* parts, const int* row, const float* alt, int K,
                    float* out) {
    J.mode = 1; J.T = T; J.nsel = nsel; J.cw = w; J.cparts = parts; J.out = out; J.L = (const moe_layer*)(size_t)K;
    J.nact = P.nthr;
    J.crow = row; J.calt = alt;
    atomic_store_explicit(&P.done_count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&P.job_gen, 1, memory_order_seq_cst);
    if (atomic_load(&P.sleepers)) futex(&P.job_gen, FUTEX_WAKE_PRIVATE, 0x7fffffff);
    run_job(0);
    for (int n = 0; atomic_load_explicit(&P.done_count, memory_order_acquire) != (unsigned)P.nthr - 1; n++)
        if (n < SPIN_MAX) _mm_pause(); else sched_yield();
    J.mode = 0; J.crow = NULL; J.calt = NULL;
    return 0;
}
int moe_combine(int T, int nsel, const float* w, const float* parts, int K, float* out) {
    return moe_combine_alt(T, nsel, w, parts, NULL, NULL, K, out);
}

int moe_forward_parts(const moe_layer* L, int T, const float* x, int nsel, const int* ids, const uint8_t* skip,
                      float* parts) {
    return moe_run(L, T, x, nsel, ids, NULL, NULL, 0, skip, parts);
}

static int moe_run(const moe_layer* L, int T, const float* x, int nsel, const int* ids, const float* wts, float* out,
                   int weight_in_h, const uint8_t* skip, float* parts) {
    if (T < 1 || nsel < 1 || nsel > MOE_MAXSEL) return -1;
    const unsigned long long t0 = __rdtsc();
    const size_t np = (size_t)T * nsel;
    if ((size_t)T > S.tcap || L->K != S.K) {
        free(S.xq); free(S.xd); free(S.tpair); free(S.xs); free(S.xl);
        S.xq = aligned_alloc(64, ((size_t)T * L->K + 63) / 64 * 64); S.xd = malloc(sizeof(float) * T * (L->K / QK));
        S.xs = malloc(sizeof(int32_t) * T * (L->K / QK));
        S.xl = aligned_alloc(64, sizeof(int32_t) * 8 * T * (L->K / QK));
        S.tpair = malloc(sizeof(int) * T * MOE_MAXSEL);
        S.tcap = T; S.K = L->K;
    }
    if (np > S.pcap || L->F != S.F) {
        free(S.ptok); free(S.pw); free(S.h); free(S.hq); free(S.hd); free(S.pslot); free(S.hs); free(S.hl);
        S.hs = malloc(np * (L->F / QK) * 4);
        S.hl = aligned_alloc(64, np * (L->F / QK) * 8 * 4);
        S.ptok = malloc(sizeof(int) * np); S.pw = malloc(sizeof(float) * np); S.pslot = malloc(sizeof(int) * np);
        S.h = aligned_alloc(64, np * L->F * 4); S.hq = aligned_alloc(64, (np * L->F + 63) / 64 * 64);
        S.hd = malloc(np * (L->F / QK) * 4);
        S.pcap = np; S.F = L->F;
    }
    /* row groups per work item for chunks of more than 8 tokens (KINDI_MOE_GB / KINDI_MOE_GBD): the activations of
       an expert's tokens are then read once per block of groups instead of once per group */
    static int gb_env = 0, gbd_env = 0, gb1_env = 1, gbd1_env = 1;
    if (!gb_env) {
        gb_env = getenv("KINDI_MOE_GB") ? atoi(getenv("KINDI_MOE_GB")) : 4;
        gbd_env = getenv("KINDI_MOE_GBD") ? atoi(getenv("KINDI_MOE_GBD")) : 8;
        /* KINDI_MOE_GB1 / GBD1: the same for decode and verify calls (8 tokens or fewer): longer contiguous streams */
        gb1_env = getenv("KINDI_MOE_GB1") ? atoi(getenv("KINDI_MOE_GB1")) : 1;
        gbd1_env = getenv("KINDI_MOE_GBD1") ? atoi(getenv("KINDI_MOE_GBD1")) : 1;
        pfb_env = getenv("KINDI_MOE_PFB") ? atoi(getenv("KINDI_MOE_PFB")) : 0;
        xsteal_env = getenv("KINDI_MOE_XSTEAL") ? atoi(getenv("KINDI_MOE_XSTEAL")) : 0;
        if (gb_env < 1) gb_env = 1;
        if (gbd_env < 1) gbd_env = 1;
        if (gb1_env < 1) gb1_env = 1;
        if (gbd1_env < 1) gbd1_env = 1;
    }
    const int gb = T > 8 ? gb_env : gb1_env, gbd = T > 8 ? gbd_env : gbd1_env;
    if (np * gbd > P.part_cap) {
        for (int t = 0; t < P.nthr + P.nx; t++) { free(P.part[t]); P.part[t] = aligned_alloc(64, (np * gbd * DN_ROWS * 4 + 63) / 64 * 64); }
        P.part_cap = np * gbd;
    }
    if (L->E > S.ecap) {
        free(S.uexp); free(S.pstart); free(S.umap); free(S.cnt);
        S.uexp = malloc(sizeof(int) * L->E); S.pstart = malloc(sizeof(int) * (L->E + 1));
        S.umap = malloc(sizeof(int) * L->E); S.cnt = malloc(sizeof(int) * L->E);
        S.ecap = L->E;
    }
    static int fast_env = -1;
    if (fast_env < 0) fast_env = getenv("KINDI_MOE_FASTPF") ? atoi(getenv("KINDI_MOE_FASTPF")) : 1;
    const int fast = fast_env && T > 8;
    if (u12_env < 0) u12_env = getenv("KINDI_MOE_U12") ? atoi(getenv("KINDI_MOE_U12")) : 0;
    const int u12 = u12_env && !fast;
    for (int t = 0; t < T; t++) {
        quant_q8(x + (size_t)t * L->K, L->K, S.xq + (size_t)t * L->K, S.xd + (size_t)t * (L->K / QK));
        if (fast) qsum12(S.xq + (size_t)t * L->K, L->K / QK, S.xs + (size_t)t * (L->K / QK));
        if (u12) lane12(S.xq + (size_t)t * L->K, L->K / QK, S.xl + (size_t)t * (L->K / QK) * 8);
    }
    /* union in order of first use, then a counting sort of the pairs by union expert (tokens stay ascending) */
    int nu = 0;
    for (size_t i = 0; i < np; i++) if (ids[i] < 0 || ids[i] >= L->E) return -2;
    for (int e = 0; e < L->E; e++) S.umap[e] = -1;
    for (size_t i = 0; i < np; i++) {
        if (skip && skip[i]) continue;
        int e = ids[i];
        if (S.umap[e] < 0) { S.umap[e] = nu; S.uexp[nu] = e; S.cnt[nu] = 0; nu++; }
        S.cnt[S.umap[e]]++;
    }
    if (nu == 0) return 0;
    S.pstart[0] = 0;
    for (int u = 0; u < nu; u++) { S.pstart[u + 1] = S.pstart[u] + S.cnt[u]; S.cnt[u] = S.pstart[u]; }
    for (int t = 0; t < T; t++)
        for (int s = 0; s < nsel; s++) {
            if (skip && skip[t * nsel + s]) { S.tpair[t * nsel + s] = -1; continue; }
            int u = S.umap[ids[t * nsel + s]], p = S.cnt[u]++;
            S.ptok[p] = t; S.pw[p] = wts ? wts[t * nsel + s] : 0.f; S.tpair[t * nsel + s] = p; S.pslot[p] = s;
        }
    J.L = L; J.T = T; J.nsel = nsel; J.xq = S.xq; J.xd = S.xd; J.fast = fast; J.xs = S.xs; J.hs = S.hs; J.nu = nu; J.uexp = S.uexp; J.pstart = S.pstart;
    J.ptok = S.ptok; J.pw = S.pw; J.tpair = S.tpair; J.h = S.h; J.hq = S.hq; J.hd = S.hd; J.out = out;
    J.weight_in_h = weight_in_h; J.parts = parts; J.pslot = S.pslot;
    J.u12 = u12; J.xl = S.xl; J.hl = S.hl;
    J.gb = gb; J.gbd = gbd; J.chunk = gb > 1 || gbd > 1 ? 1 : CHUNK;
    J.nact = T > 8 ? P.nthr + P.nx : P.nthr;
#define NBLK(n, b) (((n) + (b) - 1) / (b))
    atomic_store_explicit(&P.done_count, 0, memory_order_relaxed);
    if (P.nthr_node[1] == 0) {
        work_ranges(0, (long)nu * NBLK(L->ngu, gb), 0, P.nthr);
        work_ranges(1, NBLK(L->ndn, gbd), 0, P.nthr);
    } else {
        work_ranges(0, (long)nu * NBLK(L->gu_split, gb), 0, P.nthr_node[0]);
        work_ranges(0, (long)nu * NBLK(L->ngu - L->gu_split, gb), P.nthr_node[0], P.nthr_node[1]);
        work_ranges(1, NBLK(L->dn_split, gbd), 0, P.nthr_node[0]);
        work_ranges(1, NBLK(L->ndn - L->dn_split, gbd), P.nthr_node[0], P.nthr_node[1]);
    }
#undef NBLK
    const unsigned long long t1 = __rdtsc();
    atomic_fetch_add_explicit(&P.job_gen, 1, memory_order_seq_cst);
    if (atomic_load(&P.sleepers)) futex(&P.job_gen, FUTEX_WAKE_PRIVATE, 0x7fffffff);
    if (J.nact > P.nthr) { atomic_fetch_add_explicit(&P.xgen, 1, memory_order_seq_cst); futex(&P.xgen, FUTEX_WAKE_PRIVATE, 0x7fffffff); }
    run_job(0);
    const unsigned long long t2 = __rdtsc();
    for (int n = 0; atomic_load_explicit(&P.done_count, memory_order_acquire) != (unsigned)J.nact - 1; n++)
        if (n < SPIN_MAX) _mm_pause(); else sched_yield();
    {   /* KINDI_MOE_PROF=N: every N calls, the mean us of: before the job (quantize, union), the job on the caller,
           the wait for the other threads after it */
        static long prof = -1, n = 0; static unsigned long long a[3];
        if (prof < 0) prof = getenv("KINDI_MOE_PROF") ? atol(getenv("KINDI_MOE_PROF")) : 0;
        if (prof) {
            const unsigned long long t3 = __rdtsc();
            a[0] += t1 - t0; a[1] += t2 - t1; a[2] += t3 - t2;
            if (++n == prof) {
                fprintf(stderr, "moe_cpu prof (T=%d last): pre %.1f us, caller job %.1f us, wait %.1f us\n",
                        T, a[0] / 2.4e3 / n, a[1] / 2.4e3 / n, a[2] / 2.4e3 / n);
                n = 0; a[0] = a[1] = a[2] = 0;
            }
        }
    }
    {   /* moe_stats(): cumulative counters for the engine's step split (KINDI_DS_STATS in dsv41.py) */
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        const long long ns = ts.tv_sec * 1000000000LL + ts.tv_nsec;
        ST.calls++; ST.u12 += J.u12; ST.experts += nu; ST.pairs += (long long)np;
        ST.bytes += (long long)nu * ((long long)2 * L->F * (L->K / QK) + (long long)L->K * (L->F / QK)) * 17;
        ST.cycles += (long long)(__rdtsc() - t0);
        ST.end_ns = ns;
    }
    return nu;
}

/* out[0..5] = calls, calls with the u12 kernels, CPU experts read, (token, expert) pairs, expert bytes read, rdtsc
   cycles inside moe_run (TSC rate: out[6] = TSC ticks per second, measured once) */
void moe_stats(long long* out) {
    static long long tsc_hz = 0;
    if (!tsc_hz) {
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a); unsigned long long c0 = __rdtsc();
        do clock_gettime(CLOCK_MONOTONIC, &b); while ((b.tv_sec - a.tv_sec) * 1000000000LL + (b.tv_nsec - a.tv_nsec) < 50000000LL);
        tsc_hz = (long long)((__rdtsc() - c0) * 1e9 / ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)));
    }
    out[0] = ST.calls; out[1] = ST.u12; out[2] = ST.experts; out[3] = ST.pairs; out[4] = ST.bytes; out[5] = ST.cycles;
    out[6] = tsc_hz;
}
