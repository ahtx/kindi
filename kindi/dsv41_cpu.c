/* dsv41_cpu.c - Kindi CPU engine for DeepSeek-V4.1-Flash (Copyright (c) 2026 Amir Husain. SPDX-License-Identifier:
   AGPL-3.0-only): the whole forward pass on the host CPUs (AVX2), except the routed experts, which moe_cpu.c computes.

   Arithmetic plan (the DeepSeek reference model, with the GGUF weight types):
   - Q8_0 matrices: the input is quantized per 32 values (scale = amax / 127, kept as fp16, values round(x / scale));
     a block dot product is exact in int32 (maddubs + madd), then one fp32 FMA with the two scales.
   - F32 matrices (router, hyper-connection mixes, engram gate): fp32 FMA.
   - Activations stay fp32. Norm sums are fp64. The reference's on-the-fly roundings are kept (each can be switched
     off with dsv_set_flags): window KV and attention output to FP8 E4M3 with power-of-2 scales per 32, indexer q/k
     to FP4 E2M1 with power-of-2 scales per 32, compressed KV to FP4 with E4M3 scales per 16.

   Placement: every dense matrix is copied into memory of the two NUMA nodes, split by rows (node 0 takes share0 of
   the rows); a thread reads its own node's rows first, then helps the other node. The dense pool runs on the
   hyper-thread siblings of the expert pool's cores, so the two pools never compete for a core while one of them
   spins. The calling thread is thread 0 of both pools.

   One call (dsv_forward) takes T <= tmax tokens at the next positions (a prompt chunk or one decode token); the
   same code serves prefill and decode, and a chunk gives the same result as its tokens one at a time, up to the
   order of fp32 sums in the batched kernels (none: every kernel loops tokens outside the reduction). */
#define _GNU_SOURCE
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>

/* moe_cpu.c (separate object) */
typedef struct moe_layer_opaque moe_layer;
int moe_pool_init(const int* cpus, int n0, int n1, float share0);
moe_layer* moe_layer_new(int K, int F, int E, float clamp);
moe_layer* moe_layer_direct(int K, int F, int E, float clamp, const uint8_t* gate, const uint8_t* up, const uint8_t* down);
void moe_layer_set_expert(moe_layer* L, int e, const uint8_t* gate, const uint8_t* up, const uint8_t* down);
int moe_forward(const moe_layer* L, int T, const float* x, int nsel, const int* ids, const float* wts, float* out,
                int weight_in_h);
void moe_pool_set_spin(long pauses);
int moe_forward_parts(const moe_layer* L, int T, const float* x, int nsel, const int* ids, const uint8_t* skip,
                      float* parts);
int moe_combine(int T, int nsel, const float* w, const float* parts, int K, float* out);
int moe_combine_alt(int T, int nsel, const float* w, const float* parts, const int* row, const float* alt, int K,
                    float* out);

/* ---------------------------------------------------------------- model shape */
#define D 5120
#define NL 40
#define NH 64
#define HD 512
#define RD 64
#define QR 1280
#define OR 1024
#define OG 8
#define NE 384
#define NS 6
#define FF 2304
#define WIN 128
#define IH 32
#define IHD 128
#define ITOPK 512
#define HC 4
#define MIX 24
#define VOCAB 129280
#define CBLK 8
#define CTOPK 2048
#define CAND_SRC 20
#define ENG_COLS 24
#define ENG_HD 256
#define QK 32
#define BQ8 34                    /* Q8_0 block: fp16 scale + 32 int8 */
#define KMAXT 128
#define TIMED(k, stmt) do { double t0_ = now(); stmt; M.tx[k] += now() - t0_; } while (0)
#define TMAXC 2048                /* largest chunk (tmax); the buffers follow the tmax given at init (0.6.0: was 512) */
#define ACS 160                   /* positions per split-attention item */
#define AGH 8                     /* heads per split-attention item */
#define HKC 8                     /* input chunks of the hyper-connection mix projection */
#define ANCH ((WIN + ITOPK + ACS - 1) / ACS)
#define CRING 16                  /* compressor inputs kept per position (ring), so a rewind of up to
                                     CRING - 2 tokens (rejected drafts) cannot overwrite a pooling partner */

static const int RATIO[NL] = {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static int is_kv_src(int l) { return l == 2 || l == 8 || l == 14 || l == 20; }
static int is_idx_src(int l) { return is_kv_src(l) || l == 24 || l == 28 || l == 32 || l == 36; }

/* numerics switches: the reference's roundings (on by default), and the fp16 storage a llama.cpp-style KV cache uses
   (off by default; for comparisons with that implementation) */
enum { F_FP8_KV = 1, F_FP8_O = 2, F_FP4_IDX = 4, F_FP4_CKV = 8, F_BF16_ENG = 16, F_F16_CACHE = 32, F_F16_Q = 64,
       F_LLAMA_FA = 128 };

/* ---------------------------------------------------------------- matrices */
enum { TY_F32 = 0, TY_Q8 = 8 };
typedef struct {
    int type, rows, cols;         /* rows = outputs, cols = inputs */
    int seg_rows, seg_cols;       /* block-diagonal: row r reads input [(r / seg_rows) * seg_cols, + cols) */
    int split;                    /* rows [0, split) are on node 0 */
    size_t rb;                    /* bytes per row */
    uint8_t* p[2];
    int gid1;                     /* handle + 1 of the copy on the GPU (dsv41_gpu.cu), 0 = none */
} Mat;

typedef struct {
    int ratio;
    Mat q_a, q_b, kv, o_a, o_b, sh_gate, sh_up, sh_down, gate_inp, hc_attn_fn, hc_ffn_fn;
    Mat cmp_kv, cmp_gate, idx_k, idx_qb, idx_proj, eng_kv, eng_gate;
    const float *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias;
    const float *hc_attn_base, *hc_attn_scale, *hc_ffn_base, *hc_ffn_scale, *cmp_norm, *idx_k_norm;
    int eng;                      /* engram slot (0, 1) or -1 */
    moe_layer* moe;
    float* win;                   /* [W][HD] window ring */
    float *ckv, *ik;              /* kv sources: [cap / ratio][HD], [cap / ratio][IHD] */
    float cst_kv[CRING][HD], cst_sc[CRING][HD];
    int* topk;                    /* index sources: [tmax][ITOPK] group indices of the current chunk */
    int* ntop;                    /* [tmax] */
} Layer;

static struct {
    int cap, tmax, W, pos, flags;
    Layer L[NL];
    Mat out;
    const float* out_norm;
    const uint8_t* tok_embd;      /* Q8_0 [VOCAB][D] (mapped file) */
    const uint8_t* eng_tab;       /* MXFP8 [rows][256] (mapped file) */
    int32_t tokmap[VOCAB];
    int64_t eng_mult[2][4], eng_prime[2][ENG_COLS], eng_off[2][ENG_COLS];
    int eng_pad;
    int32_t* ctok;                /* compressed token ids by position */
    uint8_t* cand;                /* [tmax][cap / CBLK] candidate blocks of layer 20 */
    float freq_win[RD / 2], freq_cmp[RD / 2];
    /* scratch */
    float *h, *h2, *x, *mix, *pre, *post, *comb, *apre, *qa, *q, *kvw, *ck, *cg, *iq, *iw, *o, *oa, *ao, *glog, *sg, *su,
        *mo, *fo, *eemb, *ekv, *sc, *cs, *lat, *latk, *ap, *am, *as, *hcp;
    int* anch;
    int8_t *xq, *qaq, *oq, *oaq, *suq, *eq, *lq, *latq;
    float *xd, *qad, *od, *oad, *sud, *ed, *ld, *latd;
    int* latg;
    int *sel;
    float* wsel;
    FILE* dump;
    double t_dense, t_moe, t_attn, t_idx, t_total, tx[8];
    float* hdbg;                  /* when set: h after every layer, [NL][T][HC][D] (dsv_compare) */
    int cpu_only;                 /* 1: the CPU path even when the GPU is on (dsv_compare) */
} M;

/* ---------------------------------------------------------------- numerics helpers */
static float e4m3_tab[256];

static inline float fp16_round(float x) { return _cvtsh_ss(_cvtss_sh(x, 0)); }
static inline float h2f(const uint8_t* p) { uint16_t v; memcpy(&v, p, 2); return _cvtsh_ss(v); }
static void round_f16(float* x, int n) { for (int i = 0; i < n; i++) x[i] = fp16_round(x[i]); }

static float e4m3_val(int b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = ldexpf((float)m, -9);
    else v = ldexpf(1.f + m / 8.f, e - 7);
    return s ? -v : v;
}

/* round to the nearest FP8 E4M3 value (ties to even); |x| <= 448 */
static inline float rne_e4m3(float x) {
    float a = fabsf(x);
    if (a == 0.f) return x;
    float q;
    if (a < 0x1p-6f) q = 0x1p-9f;
    else { int e; frexpf(a, &e); q = ldexpf(1.f, e - 1 - 3); }
    return nearbyintf(x / q) * q;
}
/* round to the nearest FP4 E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6; ties to even); |x| <= 6 */
static inline float rne_e2m1(float x) {
    float a = fabsf(x), q = a < 2.f ? 0.5f : a < 4.f ? 1.f : 2.f;
    return nearbyintf(x / q) * q;
}
/* 2^ceil(log2(v)) as the reference computes it from the float bits */
static inline float pow2_ceil(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    int e = (int)((b >> 23) & 0xff) - 127 + ((b & 0x7fffff) != 0);
    return ldexpf(1.f, e);
}
/* AVX2 forms of the two roundings, equal to rne_e4m3 / rne_e2m1 bit for bit: E4M3 keeps 3 mantissa bits (integer
   add with the ties-to-even carry; below 2^-6 multiples of 2^-9), E2M1 scales by the inverse quantum (a power of 2) */
static inline __m256 v_rne_e4m3(__m256 x) {
    __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.f), x);
    __m256i u = _mm256_castps_si256(x);
    __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(u, 20), _mm256_set1_epi32(1));
    __m256i r = _mm256_add_epi32(u, _mm256_add_epi32(_mm256_set1_epi32(0x7ffff), lsb));
    __m256 vn = _mm256_castsi256_ps(_mm256_and_si256(r, _mm256_set1_epi32((int)0xfff00000u)));
    __m256 vs = _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(512.f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC),
                              _mm256_set1_ps(1.f / 512.f));
    return _mm256_blendv_ps(vn, vs, _mm256_cmp_ps(a, _mm256_set1_ps(0x1p-6f), _CMP_LT_OQ));
}
static inline __m256 v_rne_e2m1(__m256 x) {
    __m256 a = _mm256_andnot_ps(_mm256_set1_ps(-0.f), x);
    __m256 lt2 = _mm256_cmp_ps(a, _mm256_set1_ps(2.f), _CMP_LT_OQ), lt4 = _mm256_cmp_ps(a, _mm256_set1_ps(4.f), _CMP_LT_OQ);
    __m256 iq = _mm256_blendv_ps(_mm256_blendv_ps(_mm256_set1_ps(0.5f), _mm256_set1_ps(1.f), lt4), _mm256_set1_ps(2.f), lt2);
    __m256 q = _mm256_blendv_ps(_mm256_blendv_ps(_mm256_set1_ps(2.f), _mm256_set1_ps(1.f), lt4), _mm256_set1_ps(0.5f), lt2);
    return _mm256_mul_ps(_mm256_round_ps(_mm256_mul_ps(x, iq), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), q);
}
static inline float amax_n(const float* x, int n) {
    __m256 m = _mm256_setzero_ps();
    for (int j = 0; j < n; j += 8) m = _mm256_max_ps(m, _mm256_andnot_ps(_mm256_set1_ps(-0.f), _mm256_loadu_ps(x + j)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    return _mm_cvtss_f32(_mm_max_ss(m4, _mm_movehdup_ps(m4)));
}

/* quantize + dequantize in place, n % blk == 0 */
static void sim_fp8(float* x, int n) {
    for (int b = 0; b < n; b += 32) {
        float s = pow2_ceil(fmaxf(amax_n(x + b, 32), 1e-4f) * (1.f / 448.f));
        __m256 vs = _mm256_set1_ps(s), is = _mm256_set1_ps(1.f / s);     /* s is a power of 2: x * (1/s) == x / s */
        for (int j = 0; j < 32; j += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + b + j), is);
            v = _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(-448.f)), _mm256_set1_ps(448.f));
            _mm256_storeu_ps(x + b + j, _mm256_mul_ps(v_rne_e4m3(v), vs));
        }
    }
}
static void sim_fp4_pow2(float* x, int n) {
    for (int b = 0; b < n; b += 32) {
        float s = pow2_ceil(fmaxf(amax_n(x + b, 32), 6.f * 0x1p-126f) * (1.f / 6.f));
        __m256 vs = _mm256_set1_ps(s), is = _mm256_set1_ps(1.f / s);
        for (int j = 0; j < 32; j += 8) {
            __m256 v = _mm256_mul_ps(_mm256_loadu_ps(x + b + j), is);
            v = _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(-6.f)), _mm256_set1_ps(6.f));
            _mm256_storeu_ps(x + b + j, _mm256_mul_ps(v_rne_e2m1(v), vs));
        }
    }
}
static void sim_fp4_e4m3(float* x, int n) {
    for (int b = 0; b < n; b += 16) {
        float s = rne_e4m3(fmaxf(amax_n(x + b, 16), 6.f * 0x1p-9f) / 6.f);
        __m256 vs = _mm256_set1_ps(s);
        for (int j = 0; j < 16; j += 8) {
            __m256 v = _mm256_div_ps(_mm256_loadu_ps(x + b + j), vs);
            v = _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(-6.f)), _mm256_set1_ps(6.f));
            _mm256_storeu_ps(x + b + j, _mm256_mul_ps(v_rne_e2m1(v), vs));
        }
    }
}

/* Q8 activation blocks: scale d = amax / 127 (kept as fp16), values round-to-nearest-even(x * 127 / amax). The
   same rounding as llama.cpp's CPU Q8_0 path, which matters: 1/d instead of 127/amax moves a 1280-row projection
   by 3e-4 relative, through values that land near a rounding boundary. Vectorized with AVX2. */
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

/* sum of squares in fp64, 4 lanes (n % 8 == 0) */
static double sumsq(const float* x, int n) {
    __m256d a0 = _mm256_setzero_pd(), a1 = a0;
    for (int i = 0; i < n; i += 8) {
        __m256d u = _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4));
        a0 = _mm256_add_pd(a0, _mm256_mul_pd(u, u));
        a1 = _mm256_add_pd(a1, _mm256_mul_pd(v, v));
    }
    double t[4];
    _mm256_storeu_pd(t, _mm256_add_pd(a0, a1));
    return (t[0] + t[1]) + (t[2] + t[3]);
}

static void rmsnorm(float* y, const float* x, const float* w, int n) {
    float r = 1.f / sqrtf((float)(sumsq(x, n) / n) + 1e-20f);
    for (int i = 0; i < n; i++) y[i] = (w ? w[i] : 1.f) * (x[i] * r);
}

/* RoPE on 64 dims, adjacent pairs as complex numbers; angle = position * freq in fp32. rope_cs makes the 32
   (cos, sin) pairs of one position once; rope applies them to as many vectors as needed. */
static void rope_cs(float* cs, int pos, const float* freq) {
    for (int i = 0; i < RD / 2; i++) {
        float a = (float)pos * freq[i];
        cs[2 * i] = cosf(a);
        cs[2 * i + 1] = sinf(a);
    }
}
static void rope(float* x, const float* cs, int inverse) {
    for (int i = 0; i < RD / 2; i++) {
        float c = cs[2 * i], s = inverse ? -cs[2 * i + 1] : cs[2 * i + 1];
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        x[2 * i] = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

static void make_freqs(float* f, double base, int yarn) {
    for (int i = 0; i < RD / 2; i++) f[i] = (float)(1.0 / pow(base, (double)(2 * i) / RD));
    if (!yarn) return;
    const double orig = 65536, factor = 16, bf = 32, bs = 1;
    double lo = floor(RD * log(orig / (bf * 2 * M_PI)) / (2 * log(base)));
    double hi = ceil(RD * log(orig / (bs * 2 * M_PI)) / (2 * log(base)));
    if (lo < 0) lo = 0;
    if (hi > RD - 1) hi = RD - 1;
    for (int i = 0; i < RD / 2; i++) {
        float ramp = (float)((i - lo) / fmax(hi - lo, 1e-3));
        ramp = fminf(fmaxf(ramp, 0.f), 1.f);
        float smooth = 1.f - ramp;
        f[i] = f[i] / (float)factor * (1.f - smooth) + f[i] * smooth;
    }
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

/* ---------------------------------------------------------------- memory on one node */
static void* node_alloc(size_t bytes, int node) {
    size_t sz = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    if (sz == 0) sz = 2u << 20;
    void* p = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    madvise(p, sz, MADV_HUGEPAGE);
    unsigned long mask = 1ul << node;
    /* MPOL_BIND: when the node's free memory is taken by the page cache, the kernel reclaims cache on that node
       instead of placing the pages on the other node (MPOL_PREFERRED did that: the dense rows of node 1 ended up
       on node 0 and decode lost 40 % in the dense part). The dense weights are 8 GB, so this cannot run a node
       out of memory while the page cache holds the model. */
    if (syscall(SYS_mbind, p, sz, 2, &mask, sizeof(mask) * 8, 0) != 0)
        syscall(SYS_mbind, p, sz, 1, &mask, sizeof(mask) * 8, 0);
    return p;
}

/* ---------------------------------------------------------------- dense thread pool */
typedef void (*kfn)(int t, void* a);
static struct {
    int n, node[KMAXT], cpu[KMAXT], nn[2];
    float share0;
    pthread_t th[KMAXT];
    _Atomic unsigned gen, done;
    _Atomic int quit;
    kfn fn;
    void* arg;
} KP;

/* Idle workers spin for a short while, then sleep on a futex. kp_park() sends them to sleep at once: the
   caller does it before the expert pool runs, so the two pools never spin against each other on one core. */
static _Atomic int kp_sleepers, kp_parked;
static long futex(_Atomic unsigned* a, int op, unsigned v) { return syscall(SYS_futex, a, op, v, NULL, NULL, 0); }

static void* kp_worker(void* a) {
    int t = (int)(intptr_t)a;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(KP.cpu[t], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    unsigned seen = 0;
    for (;;) {
        unsigned g, spins = 0;
        while ((g = atomic_load_explicit(&KP.gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&KP.quit, memory_order_relaxed)) return NULL;
            if (++spins < 20000 && !atomic_load_explicit(&kp_parked, memory_order_relaxed)) { _mm_pause(); continue; }
            atomic_fetch_add(&kp_sleepers, 1);
            if (atomic_load_explicit(&KP.gen, memory_order_acquire) == seen) futex(&KP.gen, 0 /* FUTEX_WAIT */, seen);
            atomic_fetch_sub(&kp_sleepers, 1);
            spins = 0;
        }
        seen = g;
        KP.fn(t, KP.arg);
        atomic_fetch_add_explicit(&KP.done, 1, memory_order_acq_rel);
    }
}

static void kp_park(int on) { atomic_store(&kp_parked, on); }

static void kp_run(kfn fn, void* arg) {
    KP.fn = fn; KP.arg = arg;
    atomic_store(&kp_parked, 0);
    atomic_store_explicit(&KP.done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&KP.gen, 1, memory_order_seq_cst);
    if (atomic_load(&kp_sleepers)) futex(&KP.gen, 1 /* FUTEX_WAKE */, 0x7fffffff);
    fn(0, arg);
    for (int n = 0; atomic_load_explicit(&KP.done, memory_order_acquire) != (unsigned)KP.n - 1; n++)
        if (n < 20000) _mm_pause(); else sched_yield();
}

/* fn(i, a) for i in [0, n), items handed out one at a time; n == 1 runs on the caller only */
typedef void (*ifn)(int i, void* a);
static struct { ifn fn; void* a; int n; _Atomic int next; } PF;
static void pf_job(int t, void* a) {
    (void)t; (void)a;
    for (int i; (i = atomic_fetch_add_explicit(&PF.next, 1, memory_order_relaxed)) < PF.n;) PF.fn(i, PF.a);
}
static void kp_for(int n, ifn fn, void* a) {
    if (n <= 0) return;
    if (n == 1) { fn(0, a); return; }
    PF.fn = fn; PF.a = a; PF.n = n;
    atomic_store_explicit(&PF.next, 0, memory_order_relaxed);
    kp_run(pf_job, NULL);
}

/* thread 0 is the caller (node 0); cpus = the workers' cores, node 0 first */
static int kp_init(const int* cpus, int n0, int n1, float share0) {
    if (1 + n0 + n1 > KMAXT) return -1;
    KP.n = 1 + n0 + n1; KP.nn[0] = 1 + n0; KP.nn[1] = n1; KP.share0 = n1 ? share0 : 1.f;
    KP.node[0] = 0; KP.cpu[0] = -1;
    for (int i = 0; i < n0 + n1; i++) { KP.cpu[1 + i] = cpus[i]; KP.node[1 + i] = i >= n0; }
    for (int t = 1; t < KP.n; t++) pthread_create(&KP.th[t], NULL, kp_worker, (void*)(intptr_t)t);
    return 0;
}

/* ---------------------------------------------------------------- GEMV / GEMM job */
#define MAXOP 8
static int GCH = 2;            /* row groups (of 4) claimed at a time (KINDI_DSV_GCH) */
typedef struct {
    const Mat* m;
    const float* xf;               /* F32 input [T][xs] */
    const int8_t* xq; const float* xd;   /* Q8 input [T][xs], [T][xs / 32] */
    int xs;
    float* y; int ys;              /* output [T][ys] */
} Op;
static struct {
    Op op[MAXOP];
    int nop, T;
    long first[2][MAXOP + 1];      /* per node: prefix sums of row groups */
    _Atomic long next[2];
    char pad[64];
} G;

static inline float hsum8(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_movehdup_ps(a));
    return _mm_cvtss_f32(a);
}

static inline __m256i dot32(__m256i w, __m256i x, __m256i ones) {
    __m256i s = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(s, ones);
}

/* 4 Q8_0 rows (one repacked group: per block 4 fp16 scales, then 4 x 32 codes) x T tokens, two tokens per pass
   so the weights are loaded once for both. Per row and token the arithmetic is the same in any pass: per block
   acc += float(int32 dot) * (weight scale * activation scale). */
static inline __m256 blk_dot(__m256i u, __m256i w, __m256i x, __m256i ones) {
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u, _mm256_sign_epi8(x, w)), ones));
}
static void k_q8g(const uint8_t* w, int nb, int T, const int8_t* xq, const float* xd, int xs, float* y, int ys) {
    const __m256i ones = _mm256_set1_epi16(1);
    int t = 0;
    for (; t + 2 <= T; t += 2) {
        const int8_t *x0 = xq + (size_t)t * xs, *x1 = x0 + xs;
        const float *d0 = xd + (size_t)t * (xs / QK), *d1 = d0 + xs / QK;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        for (int b = 0; b < nb; b++) {
            const uint8_t* B = w + (size_t)b * 4 * BQ8;
            float s[4], sa[4], sb[4];
            _mm_storeu_ps(s, _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)B)));
            for (int i = 0; i < 4; i++) { sa[i] = s[i] * d0[b]; sb[i] = s[i] * d1[b]; }
            __m256i w0 = _mm256_loadu_si256((const __m256i*)(B + 8)), w1 = _mm256_loadu_si256((const __m256i*)(B + 40));
            __m256i w2 = _mm256_loadu_si256((const __m256i*)(B + 72)), w3 = _mm256_loadu_si256((const __m256i*)(B + 104));
            __m256i u0 = _mm256_sign_epi8(w0, w0), u1 = _mm256_sign_epi8(w1, w1), u2 = _mm256_sign_epi8(w2, w2),
                    u3 = _mm256_sign_epi8(w3, w3);
            __m256i xa = _mm256_loadu_si256((const __m256i*)(x0 + b * QK));
            __m256i xb = _mm256_loadu_si256((const __m256i*)(x1 + b * QK));
            a0 = _mm256_fmadd_ps(blk_dot(u0, w0, xa, ones), _mm256_set1_ps(sa[0]), a0);
            a1 = _mm256_fmadd_ps(blk_dot(u1, w1, xa, ones), _mm256_set1_ps(sa[1]), a1);
            a2 = _mm256_fmadd_ps(blk_dot(u2, w2, xa, ones), _mm256_set1_ps(sa[2]), a2);
            a3 = _mm256_fmadd_ps(blk_dot(u3, w3, xa, ones), _mm256_set1_ps(sa[3]), a3);
            b0 = _mm256_fmadd_ps(blk_dot(u0, w0, xb, ones), _mm256_set1_ps(sb[0]), b0);
            b1 = _mm256_fmadd_ps(blk_dot(u1, w1, xb, ones), _mm256_set1_ps(sb[1]), b1);
            b2 = _mm256_fmadd_ps(blk_dot(u2, w2, xb, ones), _mm256_set1_ps(sb[2]), b2);
            b3 = _mm256_fmadd_ps(blk_dot(u3, w3, xb, ones), _mm256_set1_ps(sb[3]), b3);
        }
        float *ya = y + (size_t)t * ys, *yb = ya + ys;
        ya[0] = hsum8(a0); ya[1] = hsum8(a1); ya[2] = hsum8(a2); ya[3] = hsum8(a3);
        yb[0] = hsum8(b0); yb[1] = hsum8(b1); yb[2] = hsum8(b2); yb[3] = hsum8(b3);
    }
    for (; t < T; t++) {
        const int8_t* x0 = xq + (size_t)t * xs;
        const float* d0 = xd + (size_t)t * (xs / QK);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const uint8_t* B = w + (size_t)b * 4 * BQ8;
            float s[4], sa[4];
            _mm_storeu_ps(s, _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)B)));
            for (int i = 0; i < 4; i++) sa[i] = s[i] * d0[b];
            __m256i w0 = _mm256_loadu_si256((const __m256i*)(B + 8)), w1 = _mm256_loadu_si256((const __m256i*)(B + 40));
            __m256i w2 = _mm256_loadu_si256((const __m256i*)(B + 72)), w3 = _mm256_loadu_si256((const __m256i*)(B + 104));
            __m256i xa = _mm256_loadu_si256((const __m256i*)(x0 + b * QK));
            a0 = _mm256_fmadd_ps(blk_dot(_mm256_sign_epi8(w0, w0), w0, xa, ones), _mm256_set1_ps(sa[0]), a0);
            a1 = _mm256_fmadd_ps(blk_dot(_mm256_sign_epi8(w1, w1), w1, xa, ones), _mm256_set1_ps(sa[1]), a1);
            a2 = _mm256_fmadd_ps(blk_dot(_mm256_sign_epi8(w2, w2), w2, xa, ones), _mm256_set1_ps(sa[2]), a2);
            a3 = _mm256_fmadd_ps(blk_dot(_mm256_sign_epi8(w3, w3), w3, xa, ones), _mm256_set1_ps(sa[3]), a3);
        }
        float* ya = y + (size_t)t * ys;
        ya[0] = hsum8(a0); ya[1] = hsum8(a1); ya[2] = hsum8(a2); ya[3] = hsum8(a3);
    }
}

/* 4 F32 rows x T tokens */
static void k_f32x4(const float* w, size_t rf, int K, int T, const float* x, int xs, float* y, int ys) {
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * xs;
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int k = 0; k < K; k += 8) {
            __m256 xv = _mm256_loadu_ps(xt + k);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), xv, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + rf + k), xv, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 2 * rf + k), xv, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + 3 * rf + k), xv, a3);
        }
        float* yt = y + (size_t)t * ys;
        yt[0] = hsum8(a0); yt[1] = hsum8(a1); yt[2] = hsum8(a2); yt[3] = hsum8(a3);
    }
}

static void run_group(int nd, long it) {
    int o = 0;
    while (it >= G.first[nd][o + 1]) o++;
    const Op* op = &G.op[o];
    const Mat* m = op->m;
    long g = it - G.first[nd][o];
    int r = (nd ? m->split : 0) + (int)g * 4;
    const uint8_t* w = m->p[nd] + (size_t)g * 4 * m->rb;
    int xoff = m->seg_rows ? (r / m->seg_rows) * m->seg_cols : 0;
    if (m->type == TY_Q8)
        k_q8g(w, m->cols / QK, G.T, op->xq + xoff, op->xd + xoff / QK, op->xs, op->y + r, op->ys);
    else
        k_f32x4((const float*)w, m->rb / 4, m->cols, G.T, op->xf + xoff, op->xs, op->y + r, op->ys);
}

/* Work split: per node, every thread owns a contiguous range of row groups (one sequential stream per thread);
   a thread that finishes takes chunks from the others of its node, then from the other node. */
static struct { _Atomic long next; long end; char pad[48]; } WQ[KMAXT];

static void gemv_job(int t, void* a) {
    (void)a;
    for (int pass = 0, nd = KP.node[t]; pass < 2; pass++, nd ^= 1) {
        int t0 = nd ? KP.nn[0] : 0, nn = KP.nn[nd];
        for (int k = 0; k < nn; k++) {
            int v = t0 + ((pass ? 0 : t - t0) + k) % nn;
            for (;;) {
                long c = atomic_fetch_add_explicit(&WQ[v].next, GCH, memory_order_relaxed), e = WQ[v].end;
                if (c >= e) break;
                long z = c + GCH < e ? c + GCH : e;
                for (long it = c; it < z; it++) run_group(nd, it);
            }
        }
    }
}

/* dsv41_gpu.cu (dlopen when KINDI_DSV_GPU=1; library path in KINDI_DSV_GPU_LIB) */
typedef struct {
    int mat; const int8_t* xq; const float* xd; const float* xf; int xs; int seg_rows, seg_cols; float* y; int ys;
} DgOp;
typedef struct {
    int ratio, kv_src, idx_src, ks, is, eng;
    int m_qa, m_qb, m_kv, m_oa, m_ob, m_shg, m_shu, m_shd, m_gate, m_ckv, m_cg, m_ik, m_iqb, m_ip, m_ekv;
    const float *hc_attn_fn, *hc_ffn_fn;
    const float *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias, *hc_attn_base, *hc_attn_scale, *hc_ffn_base,
        *hc_ffn_scale, *cmp_norm, *idx_k_norm, *eng_gate;
} DgLayerDesc;
static int kv_src_of(int l);
static int idx_src_of(int l);
static int gpu_moe_parts(int l, int T, const float* x, const int* sel, const uint8_t* skip, float* parts);
static int gpu_combine(int T, const float* wsel, const float* parts, float* out);
static int gpu_combine_alt(int T, const float* wsel, const float* parts, const int* row, const float* alt, float* out);
static struct {
    int on;
    int (*init)(int, size_t, size_t);
    int (*mat_q8)(const uint8_t*, int, int);
    int (*mat_f32)(const float*, int, int);
    int (*gemv)(const DgOp*, int, int);
    size_t (*vram_used)(void);
    int full;                     /* KINDI_DSV_GPU=2: the whole layer on the GPU (dg_forward), experts on the CPU */
    int (*model_init)(int, int, int, const float*, const void*);
    int (*forward)(int, int, int, const float*, const float*, const int8_t*, const float*, const float*, const int*,
                   const int*, int (*)(int, int, const float*, const int*, const float*, float*), float*, int, float*);
    int (*forward2)(int, int, int, const void*, const void*, int (*)(int, int, const float*, const int*, const float*, float*),
                    float*, int);
    int (*forward_split)(int, int, int, int, const void*, const void*,
                         int (*)(int, int, const float*, const int*, const float*, float*), float*);
    int (*dspark_init)(const void*, int (*)(int, int, const float*, const int*, const float*, float*));
    int (*dspark_push)(int, int, int, const float*, int);
    int (*dspark_draft)(int, int, const float*, const float*, int, int*, float*);
    void (*set_expert_src)(int, const uint8_t*, const uint8_t*, const uint8_t*);
    void (*set_combine)(int (*)(int, const float*, const float*, float*), int (*)(int, int, const float*, const int*, const uint8_t*, float*));
    void (*set_combine_alt)(int (*)(int, const float*, const float*, const int*, const float*, float*));
    size_t (*state_size)(int);
    int (*state_save)(int, void*);
    int (*state_load)(int, const void*);
    int (*hot_add)(int, int, const uint8_t*, const uint8_t*, const uint8_t*, size_t);
    int (*hot_commit)(int (*)(int, int, const float*, const int*, const uint8_t*, float*));
    size_t (*hot_bytes)(void);
} GPU;

static int gpu_open(void) {
    const char* path = getenv("KINDI_DSV_GPU_LIB");
    void* h = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : NULL;
    if (!h) { fprintf(stderr, "dsv41: cannot load the GPU library (%s)\n", path ? dlerror() : "KINDI_DSV_GPU_LIB unset"); return -1; }
    GPU.init = dlsym(h, "dg_init"); GPU.mat_q8 = dlsym(h, "dg_mat_q8"); GPU.mat_f32 = dlsym(h, "dg_mat_f32");
    GPU.gemv = dlsym(h, "dg_gemv"); GPU.vram_used = dlsym(h, "dg_vram_used");
    GPU.model_init = dlsym(h, "dg_model_init"); GPU.forward = dlsym(h, "dg_forward"); GPU.forward2 = dlsym(h, "dg_forward2");
    GPU.forward_split = dlsym(h, "dg_forward_split");
    GPU.dspark_init = dlsym(h, "dg_dspark_init"); GPU.dspark_push = dlsym(h, "dg_dspark_push");
    GPU.dspark_draft = dlsym(h, "dg_dspark_draft");
    GPU.set_expert_src = dlsym(h, "dg_set_expert_src"); GPU.set_combine = dlsym(h, "dg_set_combine");
    GPU.set_combine_alt = dlsym(h, "dg_set_combine_alt");
    GPU.state_size = dlsym(h, "dg_state_size"); GPU.state_save = dlsym(h, "dg_state_save");
    GPU.state_load = dlsym(h, "dg_state_load");
    GPU.hot_add = dlsym(h, "dg_hot_add"); GPU.hot_commit = dlsym(h, "dg_hot_commit"); GPU.hot_bytes = dlsym(h, "dg_hot_bytes");
    if (!GPU.init || !GPU.mat_q8 || !GPU.mat_f32 || !GPU.gemv || !GPU.vram_used) return -1;
    return 0;
}

/* expert cards (dsv41_gpu2.cu, KINDI_DSV_GPU2_LIBS = one library copy per card, KINDI_DSV_GPU2_DEVS = their CUDA
   device numbers): each holds a static set of routed experts and computes the pairs whose expert it holds while the
   host computes the others (decode and verify forwards of up to 8 tokens) */
#define MAXXC 4
static struct {
    int (*init)(int);
    int (*add)(int, int, const uint8_t*, const uint8_t*, const uint8_t*, size_t);
    int (*commit)(void);
    size_t (*bytes)(void);
    int (*has)(int, int);
    int (*start)(int, int, const int8_t*, const float*, const int*, const uint8_t*);
    int (*finish)(float*);
    double (*gpu_ms)(void);
    int dev, n;
} XC[MAXXC];
static int nXC, xc_parts;         /* xc_parts (KINDI_DSV_PARTS=1): the per-pair path even when no card takes a pair */
static double xc_t_start, xc_t_wait;
static long xc_pairs, xc_all;

static void xc_report(void) {
    fprintf(stderr, "dsv41: expert cards took %ld of %ld host pairs (%.1f %%); start %.1f s, wait %.1f s\n", xc_pairs, xc_all,
            xc_all ? 100.0 * xc_pairs / xc_all : 0.0, xc_t_start, xc_t_wait);
    for (int c = 0; c < nXC; c++)
        if (XC[c].gpu_ms) fprintf(stderr, "dsv41: expert card %d busy %.2f s\n", c, XC[c].gpu_ms() / 1e3);
}

static int xc_open(void) {
    const char* libs = getenv("KINDI_DSV_GPU2_LIBS");
    const char* devs = getenv("KINDI_DSV_GPU2_DEVS");
    if (!libs || !libs[0] || !devs) return 0;
    char lb[4096], db[256];
    snprintf(lb, sizeof lb, "%s", libs); snprintf(db, sizeof db, "%s", devs);
    char *sl = NULL, *sd = NULL;
    for (char *p = strtok_r(lb, ",", &sl), *q = strtok_r(db, ",", &sd); p && q && nXC < MAXXC;
         p = strtok_r(NULL, ",", &sl), q = strtok_r(NULL, ",", &sd)) {
        void* h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "dsv41: cannot load the expert-card library %s (%s)\n", p, dlerror()); return -1; }
        XC[nXC].init = dlsym(h, "dx_init"); XC[nXC].add = dlsym(h, "dx_add"); XC[nXC].commit = dlsym(h, "dx_commit");
        XC[nXC].bytes = dlsym(h, "dx_bytes"); XC[nXC].has = dlsym(h, "dx_has"); XC[nXC].start = dlsym(h, "dx_start");
        XC[nXC].finish = dlsym(h, "dx_finish"); XC[nXC].gpu_ms = dlsym(h, "dx_gpu_ms"); XC[nXC].dev = atoi(q);
        if (!XC[nXC].init || !XC[nXC].start || XC[nXC].init(XC[nXC].dev)) return -1;
        nXC++;
    }
    if (nXC) atexit(xc_report);
    return 0;
}

static void gemv_run(const Op* ops, int nop, int T) {
    double t0 = now();
    if (GPU.on && !M.cpu_only) {
        DgOp g[MAXOP];
        int all = 1;
        for (int o = 0; o < nop && all; o++) {
            const Mat* m = ops[o].m;
            all = m->gid1 > 0;
            g[o].mat = m->gid1 - 1; g[o].xq = ops[o].xq; g[o].xd = ops[o].xd; g[o].xf = ops[o].xf; g[o].xs = ops[o].xs;
            g[o].seg_rows = m->seg_rows; g[o].seg_cols = m->seg_cols; g[o].y = ops[o].y; g[o].ys = ops[o].ys;
        }
        if (all) {
            if (GPU.gemv(g, nop, T)) { fprintf(stderr, "dsv41: GPU job failed\n"); exit(1); }
            M.t_dense += now() - t0;
            return;
        }
    }
    G.nop = nop; G.T = T;
    for (int nd = 0; nd < 2; nd++) {
        G.first[nd][0] = 0;
        for (int o = 0; o < nop; o++) {
            const Mat* m = ops[o].m;
            int nr = nd ? m->rows - m->split : m->split;
            G.first[nd][o + 1] = G.first[nd][o] + nr / 4;
        }
        long total = G.first[nd][nop];
        int f = nd ? KP.nn[0] : 0, nn = KP.nn[nd];
        for (int r = 0; r < nn; r++) {
            atomic_store_explicit(&WQ[f + r].next, total * r / nn, memory_order_relaxed);
            WQ[f + r].end = total * (r + 1) / nn;
        }
    }
    memcpy(G.op, ops, sizeof(Op) * nop);
    kp_run(gemv_job, NULL);
    M.t_dense += now() - t0;
}

static Op opq(const Mat* m, const int8_t* xq, const float* xd, int xs, float* y, int ys) {
    Op o = {m, NULL, xq, xd, xs, y, ys};
    return o;
}
static Op opf(const Mat* m, const float* xf, int xs, float* y, int ys) {
    Op o = {m, xf, NULL, NULL, xs, y, ys};
    return o;
}

/* ---------------------------------------------------------------- loading */
typedef struct { char name[96]; const void* p; int type; long ne0, ne1, ne2; } Tensor;
static Tensor* TT; static int nTT, capTT;

int dsv_add_tensor(const char* name, const void* p, int type, long ne0, long ne1, long ne2) {
    if (nTT == capTT) { capTT = capTT ? 2 * capTT : 2048; TT = realloc(TT, sizeof(Tensor) * capTT); }
    Tensor* t = &TT[nTT++];
    snprintf(t->name, sizeof t->name, "%s", name);
    t->p = p; t->type = type; t->ne0 = ne0; t->ne1 = ne1; t->ne2 = ne2;
    return 0;
}
static const Tensor* find(const char* fmt, int l) {
    char nm[96]; snprintf(nm, sizeof nm, fmt, l);
    for (int i = 0; i < nTT; i++) if (!strcmp(TT[i].name, nm)) return &TT[i];
    return NULL;
}
static const float* vecf(const char* fmt, int l) {
    const Tensor* t = find(fmt, l);
    if (!t) { fprintf(stderr, "dsv41: missing "); fprintf(stderr, fmt, l); fprintf(stderr, "\n"); exit(1); }
    return (const float*)t->p;
}

/* copy a GGUF matrix into node-split memory */
typedef struct { Mat* m; const uint8_t* src; } CopyJob;
static CopyJob CJ[1024]; static int nCJ;
static int load_mat(Mat* m, const char* fmt, int l, int seg_rows, int optional) {
    memset(m, 0, sizeof *m);
    const Tensor* t = find(fmt, l);
    if (!t) {
        if (optional) return 0;
        fprintf(stderr, "dsv41: missing "); fprintf(stderr, fmt, l); fprintf(stderr, "\n"); exit(1);
    }
    if (t->type != TY_Q8 && t->type != TY_F32) { fprintf(stderr, "dsv41: %s has type %d\n", t->name, t->type); exit(1); }
    m->type = t->type; m->cols = (int)t->ne0; m->rows = (int)t->ne1;
    m->rb = t->type == TY_Q8 ? (size_t)m->cols / QK * BQ8 : (size_t)m->cols * 4;
    if (seg_rows) { m->seg_rows = seg_rows; m->seg_cols = m->cols; }
    m->split = (int)lroundf(m->rows / 4 * KP.share0) * 4;
    if (m->split > m->rows) m->split = m->rows;
    m->p[0] = node_alloc((size_t)m->split * m->rb, 0);
    m->p[1] = node_alloc((size_t)(m->rows - m->split) * m->rb, 1);
    CJ[nCJ].m = m; CJ[nCJ].src = (const uint8_t*)t->p; nCJ++;
    return 1;
}
/* every thread copies its node's rows (first touch on the right node) */
static void copy_job(int t, void* a) {
    (void)a;
    int nd = KP.node[t], rk = 0, nn = KP.nn[nd];
    for (int u = 0; u < t; u++) rk += KP.node[u] == nd;
    for (int j = 0; j < nCJ; j++) {
        Mat* m = CJ[j].m;
        int r0 = nd ? m->split : 0, nr = nd ? m->rows - m->split : m->split;
        if (m->type != TY_Q8) {
            size_t bytes = (size_t)nr * m->rb, a0 = bytes * rk / nn, a1 = bytes * (rk + 1) / nn;
            memcpy(m->p[nd] + a0, CJ[j].src + (size_t)r0 * m->rb + a0, a1 - a0);
            continue;
        }
        /* Q8_0: groups of 4 rows, per block 4 fp16 scales then 4 x 32 codes (136 bytes, the same size) */
        int ng = nr / 4, nb = m->cols / QK, g0 = (int)((long)ng * rk / nn), g1 = (int)((long)ng * (rk + 1) / nn);
        for (int g = g0; g < g1; g++)
            for (int b = 0; b < nb; b++) {
                uint8_t* dst = m->p[nd] + ((size_t)g * nb + b) * 4 * BQ8;
                for (int i = 0; i < 4; i++) {
                    const uint8_t* src = CJ[j].src + (size_t)(r0 + 4 * g + i) * m->rb + (size_t)b * BQ8;
                    memcpy(dst + 2 * i, src, 2);
                    memcpy(dst + 8 + 32 * i, src + 2, 32);
                }
            }
    }
}

/* distinct routed experts read on the CPU by the main layers, summed over layers and calls (cards and the DSpark
   drafter excluded); the drafter's cost model reads the difference around a verify forward */
static long dsv_union_total;
long dsv_union_count(void) { return dsv_union_total; }

/* routed experts into moe_cpu.c's node-split layout; each thread takes whole experts */
static const void* EX[NL][3];
static _Atomic int ex_next;
static void expert_job(int t, void* a) {
    (void)t; (void)a;
    const size_t gb = (size_t)FF * (D / 32) * 17, db = (size_t)D * (FF / 32) * 17;
    for (int i; (i = atomic_fetch_add(&ex_next, 1)) < NL * NE;) {
        int l = i / NE, e = i % NE;
        moe_layer_set_expert(M.L[l].moe, e, (const uint8_t*)EX[l][0] + e * gb, (const uint8_t*)EX[l][1] + e * gb,
                             (const uint8_t*)EX[l][2] + e * db);
    }
}

/* thread ids of this process */
static int list_tasks(int* out, int cap) {
    DIR* d = opendir("/proc/self/task");
    int n = 0;
    if (!d) return 0;
    for (struct dirent* e; (e = readdir(d)) && n < cap;)
        if (e->d_name[0] != '.') out[n++] = atoi(e->d_name);
    closedir(d);
    return n;
}

static void* zalloc(size_t n) { void* p = aligned_alloc(64, (n + 63) & ~(size_t)63); memset(p, 0, n); return p; }

/* cpus: dense workers (node 0 then node 1); mcpus: expert pool cores (node 0 then node 1); expert_copy = 1 copies the
   experts into node-split memory (fast), 0 reads them in place from the mapped file. */
int dsv_init(int cap, int tmax, const int* cpus, int n0, int n1, float share0, const int* mcpus, int m0, int m1,
             float mshare0, int expert_copy) {
    for (int b = 0; b < 256; b++) e4m3_tab[b] = e4m3_val(b);
    int before[KMAXT * 2], nbefore = list_tasks(before, KMAXT * 2);
    if (moe_pool_init(mcpus, m0, m1, mshare0)) return -1;
    if (getenv("KINDI_MOE_IDLE") && atoi(getenv("KINDI_MOE_IDLE"))) {
        /* the expert pool's idle workers spin and yield; at SCHED_IDLE they give their core to a dense worker
           at once, so both pools can share the physical cores */
        int after[KMAXT * 2], nafter = list_tasks(after, KMAXT * 2), nidle = 0;
        for (int i = 0; i < nafter; i++) {
            int old = 0;
            for (int j = 0; j < nbefore; j++) old |= after[i] == before[j];
            if (old) continue;
            struct sched_param sp = {0};
            if (sched_setscheduler(after[i], SCHED_IDLE, &sp) == 0) nidle++;
        }
        fprintf(stderr, "dsv41: %d expert workers at SCHED_IDLE\n", nidle);
    }
    if (kp_init(cpus, n0, n1, share0)) return -2;
    if (tmax < 1 || tmax > TMAXC) return -5;
    if (getenv("KINDI_DSV_GCH")) GCH = atoi(getenv("KINDI_DSV_GCH"));
    M.cap = cap; M.tmax = tmax; M.W = WIN + tmax; M.flags = F_FP8_KV | F_FP8_O | F_FP4_IDX | F_FP4_CKV;
    make_freqs(M.freq_win, 10000.0, 0);
    make_freqs(M.freq_cmp, 160000.0, 1);
    double t0 = now();
    for (int l = 0; l < NL; l++) {
        Layer* L = &M.L[l];
        L->ratio = RATIO[l];
        load_mat(&L->q_a, "blk.%d.attn_q_a.weight", l, 0, 0);
        load_mat(&L->q_b, "blk.%d.attn_q_b.weight", l, 0, 0);
        load_mat(&L->kv, "blk.%d.attn_kv.weight", l, 0, 0);
        load_mat(&L->o_a, "blk.%d.attn_output_a.weight", l, OR, 0);
        load_mat(&L->o_b, "blk.%d.attn_output_b.weight", l, 0, 0);
        load_mat(&L->sh_gate, "blk.%d.ffn_gate_shexp.weight", l, 0, 0);
        load_mat(&L->sh_up, "blk.%d.ffn_up_shexp.weight", l, 0, 0);
        load_mat(&L->sh_down, "blk.%d.ffn_down_shexp.weight", l, 0, 0);
        load_mat(&L->gate_inp, "blk.%d.ffn_gate_inp.weight", l, 0, 0);
        load_mat(&L->hc_attn_fn, "blk.%d.hc_attn_fn.weight", l, 0, 0);
        load_mat(&L->hc_ffn_fn, "blk.%d.hc_ffn_fn.weight", l, 0, 0);
        load_mat(&L->cmp_kv, "blk.%d.attn_compressor_kv.weight", l, 0, 1);
        load_mat(&L->cmp_gate, "blk.%d.attn_compressor_gate.weight", l, 0, 1);
        load_mat(&L->idx_k, "blk.%d.indexer.attn_k.weight", l, 0, 1);
        load_mat(&L->idx_qb, "blk.%d.indexer.attn_q_b.weight", l, 0, 1);
        load_mat(&L->idx_proj, "blk.%d.indexer.proj.weight", l, 0, 1);
        L->eng = -1;
        if (load_mat(&L->eng_kv, "blk.%d.engram_kv.weight", l, 0, 1)) {
            load_mat(&L->eng_gate, "blk.%d.engram_gate.weight", l, 0, 0);
            L->eng = l == 1 ? 0 : 1;
        }
        L->sinks = vecf("blk.%d.attn_sinks.weight", l);
        L->kv_norm = vecf("blk.%d.attn_kv_a_norm.weight", l);
        L->q_norm = vecf("blk.%d.attn_q_a_norm.weight", l);
        L->attn_norm = vecf("blk.%d.attn_norm.weight", l);
        L->ffn_norm = vecf("blk.%d.ffn_norm.weight", l);
        L->bias = vecf("blk.%d.exp_probs_b.bias", l);
        L->hc_attn_base = vecf("blk.%d.hc_attn_base.weight", l);
        L->hc_attn_scale = vecf("blk.%d.hc_attn_scale.weight", l);
        L->hc_ffn_base = vecf("blk.%d.hc_ffn_base.weight", l);
        L->hc_ffn_scale = vecf("blk.%d.hc_ffn_scale.weight", l);
        if (is_kv_src(l)) {
            L->cmp_norm = vecf("blk.%d.attn_compressor_norm.weight", l);
            L->idx_k_norm = vecf("blk.%d.indexer.k_norm.weight", l);
            size_t ng = (size_t)cap / L->ratio + 1;
            L->ckv = zalloc(ng * HD * 4);
            L->ik = zalloc(ng * IHD * 4);
        }
        if (is_idx_src(l)) { L->topk = zalloc((size_t)tmax * ITOPK * 4); L->ntop = zalloc((size_t)tmax * 4); }
        L->win = zalloc((size_t)M.W * HD * 4);
        const Tensor *g = find("blk.%d.ffn_gate_exps.weight", l), *u = find("blk.%d.ffn_up_exps.weight", l),
                     *d = find("blk.%d.ffn_down_exps.weight", l);
        if (!g || !u || !d) { fprintf(stderr, "dsv41: missing experts of layer %d\n", l); return -3; }
        if (expert_copy) {
            L->moe = moe_layer_new(D, FF, NE, 10.f);
            EX[l][0] = g->p; EX[l][1] = u->p; EX[l][2] = d->p;
        } else {
            L->moe = moe_layer_direct(D, FF, NE, 10.f, g->p, u->p, d->p);
        }
    }
    load_mat(&M.out, "output.weight", 0, 0, 0);
    M.out_norm = vecf("output_norm.weight", 0);
    const Tensor* te = find("token_embd.weight", 0);
    const Tensor* pe = find("per_layer_token_embd.weight", 0);
    if (!te || !pe) return -4;
    M.tok_embd = te->p; M.eng_tab = pe->p;
    kp_run(copy_job, NULL);
    double t1 = now();
    fprintf(stderr, "dsv41: dense weights copied in %.1f s\n", t1 - t0);
    if (getenv("KINDI_DSV_GPU") && atoi(getenv("KINDI_DSV_GPU"))) {
        /* every matrix that goes through gemv_run, resident on the GPU (the hyper-connection mixes and the engram
           gate have their own CPU code) */
        if (gpu_open() || GPU.init(0, (size_t)64 << 20, (size_t)160 << 20)) return -6;
        for (int j = 0; j < nCJ; j++) {
            Mat* m = CJ[j].m;
            int skip = 0;
            for (int l = 0; l < NL; l++) skip |= m == &M.L[l].hc_attn_fn || m == &M.L[l].hc_ffn_fn || m == &M.L[l].eng_gate;
            if (skip) continue;
            int h = m->type == TY_Q8 ? GPU.mat_q8(CJ[j].src, m->rows, m->cols)
                                     : GPU.mat_f32((const float*)CJ[j].src, m->rows, m->cols);
            if (h < 0) return -6;
            m->gid1 = h + 1;
        }
        GPU.on = 1;
        if (atoi(getenv("KINDI_DSV_GPU")) >= 2) {
            if (!GPU.model_init || !GPU.forward) return -6;
            DgLayerDesc desc[NL];
            memset(desc, 0, sizeof desc);
            for (int l = 0; l < NL; l++) {
                Layer* L = &M.L[l];
                DgLayerDesc* d = &desc[l];
                d->ratio = L->ratio; d->kv_src = is_kv_src(l); d->idx_src = is_idx_src(l);
                d->ks = L->ratio ? kv_src_of(l) : -1; d->is = L->ratio ? idx_src_of(l) : -1; d->eng = L->eng;
                d->m_qa = L->q_a.gid1 - 1; d->m_qb = L->q_b.gid1 - 1; d->m_kv = L->kv.gid1 - 1; d->m_oa = L->o_a.gid1 - 1;
                d->m_ob = L->o_b.gid1 - 1; d->m_shg = L->sh_gate.gid1 - 1; d->m_shu = L->sh_up.gid1 - 1;
                d->m_shd = L->sh_down.gid1 - 1; d->m_gate = L->gate_inp.gid1 - 1; d->m_ckv = L->cmp_kv.gid1 - 1;
                d->m_cg = L->cmp_gate.gid1 - 1; d->m_ik = L->idx_k.gid1 - 1; d->m_iqb = L->idx_qb.gid1 - 1;
                d->m_ip = L->idx_proj.gid1 - 1; d->m_ekv = L->eng_kv.gid1 - 1;
                d->hc_attn_fn = vecf("blk.%d.hc_attn_fn.weight", l); d->hc_ffn_fn = vecf("blk.%d.hc_ffn_fn.weight", l);
                d->sinks = L->sinks; d->kv_norm = L->kv_norm; d->q_norm = L->q_norm; d->attn_norm = L->attn_norm;
                d->ffn_norm = L->ffn_norm; d->bias = L->bias; d->hc_attn_base = L->hc_attn_base;
                d->hc_attn_scale = L->hc_attn_scale; d->hc_ffn_base = L->hc_ffn_base; d->hc_ffn_scale = L->hc_ffn_scale;
                d->cmp_norm = L->cmp_norm; d->idx_k_norm = L->idx_k_norm;
                d->eng_gate = L->eng >= 0 ? vecf("blk.%d.engram_gate.weight", l) : NULL;
            }
            if (GPU.model_init(cap, tmax, M.out.gid1 - 1, M.out_norm, desc)) return -7;
            GPU.full = 1;
            /* prompt chunks: the card can take part of the routed experts (KINDI_SPLIT=1, dsv41_gpu_fwd.cuh) */
            if (GPU.set_expert_src && GPU.set_combine) {
                for (int l = 0; l < NL; l++)
                    GPU.set_expert_src(l, find("blk.%d.ffn_gate_exps.weight", l)->p, find("blk.%d.ffn_up_exps.weight", l)->p,
                                       find("blk.%d.ffn_down_exps.weight", l)->p);
                GPU.set_combine(gpu_combine, gpu_moe_parts);
                if (GPU.set_combine_alt) GPU.set_combine_alt(gpu_combine_alt);
            }
            /* the expert cores wait ~1-2 ms per layer for the GPU: keep them spinning (clock up) between layers */
            moe_pool_set_spin(getenv("KINDI_MOE_SPIN") ? atol(getenv("KINDI_MOE_SPIN")) : 4000000);
            /* hot experts: "layer expert" lines, most used first (tools/hot_list.py); as many as fit, keeping
               KINDI_HOT_RESERVE_MB (default 400) of VRAM free */
            const char* hl = getenv("KINDI_HOT_LIST");
            FILE* hf = hl && hl[0] && GPU.hot_add ? fopen(hl, "r") : NULL;
            if (hf) {
                size_t reserve = (size_t)(getenv("KINDI_HOT_RESERVE_MB") ? atol(getenv("KINDI_HOT_RESERVE_MB")) : 400) << 20;
                int maxn = getenv("KINDI_HOT_MAX") ? atoi(getenv("KINDI_HOT_MAX")) : 1 << 30, n = 0, l, e;
                const size_t gb = (size_t)FF * (D / 32) * 17, db = (size_t)D * (FF / 32) * 17;
                while (n < maxn && fscanf(hf, "%d %d%*[^\n]", &l, &e) == 2) {
                    if (l < 0 || l >= NL || e < 0 || e >= NE) continue;
                    const Tensor *g = find("blk.%d.ffn_gate_exps.weight", l), *u = find("blk.%d.ffn_up_exps.weight", l),
                                 *dn = find("blk.%d.ffn_down_exps.weight", l);
                    if (GPU.hot_add(l, e, (const uint8_t*)g->p + e * gb, (const uint8_t*)u->p + e * gb,
                                    (const uint8_t*)dn->p + e * db, reserve)) break;
                    n++;
                }
                GPU.hot_commit(gpu_moe_parts);
                fprintf(stderr, "dsv41: %d hot experts on the GPU (%.2f GB)\n", n, GPU.hot_bytes() / 1e9);
                /* the expert cards take the next entries of the list, in card order (KINDI_HOT_RESERVE2_MB free) */
                if (xc_open()) { fclose(hf); return -6; }
                size_t reserve2 = (size_t)(getenv("KINDI_HOT_RESERVE2_MB") ? atol(getenv("KINDI_HOT_RESERVE2_MB")) : 300) << 20;
                for (int c = 0; c < nXC; c++) {
                    while (fscanf(hf, "%d %d%*[^\n]", &l, &e) == 2) {
                        if (l < 0 || l >= NL || e < 0 || e >= NE) continue;
                        const Tensor *g = find("blk.%d.ffn_gate_exps.weight", l), *u = find("blk.%d.ffn_up_exps.weight", l),
                                     *dn = find("blk.%d.ffn_down_exps.weight", l);
                        if (XC[c].add(l, e, (const uint8_t*)g->p + e * gb, (const uint8_t*)u->p + e * gb,
                                      (const uint8_t*)dn->p + e * db, reserve2)) break;
                        XC[c].n++;
                    }
                    XC[c].commit();
                    fprintf(stderr, "dsv41: expert card %d (device %d): %d experts (%.2f GB)\n", c, XC[c].dev, XC[c].n,
                            XC[c].bytes() / 1e9);
                }
                fclose(hf);
            }
            xc_parts = getenv("KINDI_DSV_PARTS") && atoi(getenv("KINDI_DSV_PARTS"));
        }
        fprintf(stderr, "dsv41: dense matrices on the GPU in %.1f s, %.2f GB of VRAM%s\n", now() - t1, GPU.vram_used() / 1e9,
                GPU.full ? " (whole layers on the GPU)" : "");
        t1 = now();
    }
    if (expert_copy) {
        kp_run(expert_job, NULL);
        fprintf(stderr, "dsv41: experts copied in %.1f s\n", now() - t1);
    }
    /* scratch */
    size_t T = tmax;
    M.ctok = zalloc((size_t)(cap + 8) * 4);
    M.cand = zalloc(T * (cap / CBLK + 1));
    M.h = zalloc(T * HC * D * 4); M.h2 = zalloc(T * HC * D * 4); M.x = zalloc(T * D * 4);
    M.mix = zalloc(T * MIX * 4); M.pre = zalloc(T * HC * 4); M.post = zalloc(T * HC * 4); M.comb = zalloc(T * 16 * 4);
    M.apre = zalloc(T * HC * 4);
    M.qa = zalloc(T * QR * 4); M.q = zalloc(T * NH * HD * 4); M.kvw = zalloc(T * HD * 4);
    M.ck = zalloc(T * HD * 4); M.cg = zalloc(T * HD * 4); M.iq = zalloc(T * IH * IHD * 4); M.iw = zalloc(T * IH * 4);
    M.o = zalloc(T * NH * HD * 4); M.oa = zalloc(T * OG * OR * 4); M.ao = zalloc(T * D * 4);
    M.glog = zalloc(T * NE * 4); M.sg = zalloc(T * FF * 4); M.su = zalloc(T * FF * 4); M.mo = zalloc(T * D * 4);
    M.fo = zalloc(T * D * 4); M.eemb = zalloc(T * ENG_COLS * ENG_HD * 4); M.ekv = zalloc(T * (HC + 1) * D * 4);
    M.sc = zalloc(T * ((size_t)cap + 8) * 4);
    M.xq = zalloc(T * D); M.xd = zalloc(T * D / QK * 4);
    M.qaq = zalloc(T * QR); M.qad = zalloc(T * QR / QK * 4);
    M.oq = zalloc(T * NH * HD); M.od = zalloc(T * NH * HD / QK * 4);
    M.oaq = zalloc(T * OG * OR); M.oad = zalloc(T * OG * OR / QK * 4);
    M.suq = zalloc(T * FF); M.sud = zalloc(T * FF / QK * 4);
    M.eq = zalloc(T * ENG_COLS * ENG_HD); M.ed = zalloc(T * ENG_COLS * ENG_HD / QK * 4);
    M.lq = zalloc(T * D); M.ld = zalloc(T * D / QK * 4);
    M.sel = zalloc(T * NS * 4); M.wsel = zalloc(T * NS * 4);
    M.cs = zalloc(T * 2 * RD * 4);
    M.ap = zalloc(T * NH * ANCH * HD * 4); M.am = zalloc(T * NH * ANCH * 4); M.as = zalloc(T * NH * ANCH * 4);
    M.anch = zalloc(T * 4);
    M.hcp = zalloc(T * HKC * MIX * 4);
    M.lat = zalloc(T * HD * 4); M.latk = zalloc(T * IHD * 4); M.latq = zalloc(T * HD); M.latd = zalloc(T * HD / QK * 4);
    M.latg = zalloc(T * 4);
    M.pos = 0;
    return 0;
}

int dsv_set_engram(const int32_t* tokmap, const int64_t* mult, const int64_t* primes, const int64_t* offs, int pad) {
    memcpy(M.tokmap, tokmap, sizeof M.tokmap);
    for (int s = 0; s < 2; s++) {
        for (int k = 0; k < 4; k++) M.eng_mult[s][k] = mult[s * 4 + k];
        for (int c = 0; c < ENG_COLS; c++) { M.eng_prime[s][c] = primes[s * ENG_COLS + c]; M.eng_off[s][c] = offs[s * ENG_COLS + c]; }
    }
    M.eng_pad = tokmap[pad];
    return 0;
}

void dsv_set_flags(int flags) { M.flags = flags; }
int dsv_get_flags(void) { return M.flags; }
void dsv_reset(void) { M.pos = 0; }
int dsv_pos(void) { return M.pos; }
void dsv_set_pos(int p) { M.pos = p; }   /* rewind (the caches past p are rewritten by the next forward) */
void dsv_times(double* t) {
    t[0] = M.t_dense; t[1] = M.t_moe; t[2] = M.t_attn; t[3] = M.t_idx; t[4] = M.t_total;
    for (int i = 0; i < 8; i++) t[5 + i] = M.tx[i];
}

int dsv_dump_open(const char* path) {
    if (M.dump) fclose(M.dump);
    M.dump = path && path[0] ? fopen(path, "wb") : NULL;
    return M.dump ? 0 : -1;
}
/* record: char name[32], int32 layer, int32 T, int32 n, int32 pos0, then T x n floats (token stride ts) */
static void dump(const char* name, int l, int T, int n, const float* x, size_t ts) {
    if (!M.dump) return;
    char nm[32] = {0}; snprintf(nm, sizeof nm, "%s", name);
    int32_t h[4] = {l, T, n, M.pos};
    fwrite(nm, 1, 32, M.dump); fwrite(h, 4, 4, M.dump);
    for (int t = 0; t < T; t++) fwrite(x + t * ts, 4, n, M.dump);
}

/* ---------------------------------------------------------------- hyper-connections */
/* Per token: the 24 mixes -> pre / post / comb (comb made doubly stochastic by Sinkhorn); then the sub-layer input
   x = sum_c pre_in[c] * h[c], RMS-normed and quantized for the next projections. */
typedef struct { const float *scale, *base, *pre_in, *norm_w; float *pre, *post, *comb; } MixArg;

static void mix_tok(int t, void* a) {
    const MixArg* A = a;
    const float eps = 1e-6f;
    const float* x = M.h + (size_t)t * HC * D;
    float r = 1.f / sqrtf((float)(sumsq(x, HC * D) / (HC * D)) + 1e-20f);
    float m[MIX];
    for (int i = 0; i < MIX; i++) m[i] = M.mix[t * MIX + i] * r;
    const float *scale = A->scale, *base = A->base;
    float *pr = A->pre + t * HC, *po = A->post + t * HC, *c = A->comb + t * 16;
    for (int j = 0; j < HC; j++) pr[j] = 1.f / (1.f + expf(-(m[j] * scale[0] + base[j]))) + eps;
    for (int j = 0; j < HC; j++) po[j] = 2.f / (1.f + expf(-(m[j + HC] * scale[1] + base[j + HC])));
    for (int j = 0; j < HC; j++)
        for (int k = 0; k < HC; k++) c[j * HC + k] = m[j * HC + k + 2 * HC] * scale[2] + base[j * HC + k + 2 * HC];
    for (int j = 0; j < HC; j++) {
        float mx = c[j * HC];
        for (int k = 1; k < HC; k++) mx = fmaxf(mx, c[j * HC + k]);
        float s2 = 0;
        for (int k = 0; k < HC; k++) { c[j * HC + k] = expf(c[j * HC + k] - mx); s2 += c[j * HC + k]; }
        for (int k = 0; k < HC; k++) c[j * HC + k] = c[j * HC + k] / s2 + eps;
    }
    for (int it = 0; it < 20; it++) {
        if (it > 0)
            for (int j = 0; j < HC; j++) {
                float s2 = 0;
                for (int k = 0; k < HC; k++) s2 += c[j * HC + k];
                for (int k = 0; k < HC; k++) c[j * HC + k] /= s2 + eps;
            }
        for (int k = 0; k < HC; k++) {
            float s2 = 0;
            for (int j = 0; j < HC; j++) s2 += c[j * HC + k];
            for (int j = 0; j < HC; j++) c[j * HC + k] /= s2 + eps;
        }
    }
    float* xo = M.x + (size_t)t * D;
    const float* pin = A->pre_in + t * HC;
    for (int d = 0; d < D; d += 8) {
        __m256 v = _mm256_mul_ps(_mm256_set1_ps(pin[0]), _mm256_loadu_ps(x + d));
        for (int k = 1; k < HC; k++) v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_set1_ps(pin[k]), _mm256_loadu_ps(x + k * D + d)));
        _mm256_storeu_ps(xo + d, v);
    }
    rmsnorm(xo, xo, A->norm_w, D);
    quant_q8(xo, D, M.xq + (size_t)t * D, M.xd + (size_t)t * D / QK);
}

/* the 24 x 20480 F32 mix projection: too few rows for one item per row group, so the items are (4 rows, 1/HKC
   of the inputs); the partial sums are added in a fixed order */

typedef struct { const Mat* m; int T; float* part; } HcArg;
static void hcmix_item(int i, void* a) {
    const HcArg* A = a;
    const Mat* m = A->m;
    int g = i / HKC, c = i % HKC, kc = m->cols / HKC;
    for (int r = 4 * g; r < 4 * g + 4; r++) {
        const float* w = (const float*)(r < m->split ? m->p[0] + (size_t)r * m->rb : m->p[1] + (size_t)(r - m->split) * m->rb);
        w += (size_t)c * kc;
        for (int t = 0; t < A->T; t++) {
            const float* x = M.h + (size_t)t * HC * D + (size_t)c * kc;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0;
            for (int k = 0; k < kc; k += 16) {
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x + k), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k + 8), _mm256_loadu_ps(x + k + 8), a1);
            }
            A->part[((size_t)c * A->T + t) * MIX + r] = hsum8(_mm256_add_ps(a0, a1));
        }
    }
}

static void hc_in(const Mat* fn, const float* scale, const float* base, const float* pre_in, const float* norm_w,
                  float* pre, float* post, float* comb, int T) {
    double t0 = now();
    HcArg H = {fn, T, M.hcp};
    kp_for(MIX / 4 * HKC, hcmix_item, &H);
    for (int t = 0; t < T; t++)
        for (int r = 0; r < MIX; r++) {
            float v = M.hcp[(size_t)t * MIX + r];
            for (int c = 1; c < HKC; c++) v += M.hcp[((size_t)c * T + t) * MIX + r];
            M.mix[t * MIX + r] = v;
        }
    M.t_dense += now() - t0;
    MixArg A = {scale, base, pre_in, norm_w, pre, post, comb};
    TIMED(0, kp_for(T, mix_tok, &A));
}

/* h[t][j] = post[j] * y + sum_i comb[i][j] * h[t][i] (AVX2; per element the same operations as scalar code) */
static void post_tok(int t, void* a) {
    const float* y = (const float*)a + (size_t)t * D;
    float* ht = M.h + (size_t)t * HC * D;
    const float *po = M.post + t * HC, *c = M.comb + t * 16;
    for (int d = 0; d < D; d += 8) {
        __m256 v = _mm256_loadu_ps(y + d), r[HC], o[HC];
        for (int i = 0; i < HC; i++) r[i] = _mm256_loadu_ps(ht + i * D + d);
        for (int j = 0; j < HC; j++) {
            __m256 s = _mm256_mul_ps(_mm256_set1_ps(c[j]), r[0]);
            for (int i = 1; i < HC; i++) s = _mm256_add_ps(s, _mm256_mul_ps(_mm256_set1_ps(c[i * HC + j]), r[i]));
            o[j] = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(po[j]), v), s);
        }
        for (int j = 0; j < HC; j++) _mm256_storeu_ps(ht + j * D + d, o[j]);
    }
}
static void hc_post(const float* y, int T) { TIMED(1, kp_for(T, post_tok, (void*)y)); }

/* ---------------------------------------------------------------- engram */
static void eng_rows_tok(int t, void* a) {
    const Layer* L = a;
    int s = L->eng, p = M.pos + t;
    int64_t tk[4];
    int blocked = 0;
    for (int k = 0; k < 4; k++) {
        blocked |= p < k;
        tk[k] = blocked ? M.eng_pad : M.ctok[p - k];
    }
    int64_t roll = tk[0] * M.eng_mult[s][0];
    float* e = M.eemb + (size_t)t * ENG_COLS * ENG_HD;
    for (int i = 1; i < 4; i++) {
        roll ^= tk[i] * M.eng_mult[s][i];
        for (int hh = 0; hh < 8; hh++) {
            int c = (i - 1) * 8 + hh;
            int64_t row = roll % M.eng_prime[s][c] + M.eng_off[s][c];
            const uint8_t* src = M.eng_tab + (size_t)row * (ENG_HD / 32) * 33;
            for (int b = 0; b < ENG_HD / 32; b++) {
                float sc = ldexpf(1.f, (int)src[b * 33] - 127);
                for (int j = 0; j < 32; j++) {
                    float v = e4m3_tab[src[b * 33 + 1 + j]] * sc;
                    if (M.flags & F_BF16_ENG) {
                        uint32_t u; memcpy(&u, &v, 4);
                        u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
                        memcpy(&v, &u, 4);
                    }
                    e[c * ENG_HD + b * 32 + j] = v;
                }
            }
        }
    }
    quant_q8(e, ENG_COLS * ENG_HD, M.eq + (size_t)t * ENG_COLS * ENG_HD, M.ed + (size_t)t * ENG_COLS * ENG_HD / QK);
}

static void eng_gate_tok(int t, void* a) {
    const Layer* L = a;
    const Mat* gm = &L->eng_gate;          /* q_weight * k_weight, stored as one [HC][D] matrix */
    float* ht = M.h + (size_t)t * HC * D;
    const float* key = M.ekv + (size_t)t * (HC + 1) * D;
    const float* val = key + HC * D;
    for (int c = 0; c < HC; c++) {
        const float* wc = (const float*)(c < gm->split ? gm->p[0] + (size_t)c * gm->rb
                                                       : gm->p[1] + (size_t)(c - gm->split) * gm->rb);
        const float *hc = ht + c * D, *kc = key + c * D;
        double sh = 0, sk = 0, dot = 0;
        for (int d = 0; d < D; d++) {
            sh += (double)hc[d] * hc[d];
            sk += (double)kc[d] * kc[d];
            dot += (double)hc[d] * wc[d] * kc[d];
        }
        float rstd = (1.f / sqrtf((float)(sh / D) + 1e-20f)) * (1.f / sqrtf((float)(sk / D) + 1e-20f));
        float dt = (float)dot * rstd * (1.f / sqrtf((float)D));
        float g = sqrtf(fmaxf(fabsf(dt), 1e-6f));
        g = 1.f / (1.f + expf(-(dt < 0 ? -g : g)));
        float* hw = ht + c * D;
        for (int d = 0; d < D; d++) hw[d] += g * val[d];
    }
}

static void engram(Layer* L, int T) {
    TIMED(2, kp_for(T, eng_rows_tok, L));
    Op o = opq(&L->eng_kv, M.eq, M.ed, ENG_COLS * ENG_HD, M.ekv, (HC + 1) * D);
    gemv_run(&o, 1, T);
    dump("engram_rows", (int)(L - M.L), T, ENG_COLS * ENG_HD, M.eemb, ENG_COLS * ENG_HD);
    dump("engram_kv", (int)(L - M.L), T, (HC + 1) * D, M.ekv, (HC + 1) * D);
    TIMED(2, kp_for(T, eng_gate_tok, L));
}

/* ---------------------------------------------------------------- attention */
/* e^x for x <= 0 (softmax), AVX2: Cody-Waite range reduction and a degree-6 polynomial (Cephes expf), ~1 ulp */
static inline __m256 exp256(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.f));
    __m256 n = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _mm256_set1_ps(0.5f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693359375f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(-2.12194440e-4f)));
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(y, x), x), x), _mm256_set1_ps(1.f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

static inline const float* cs_of(int t, int cmp) { return M.cs + (size_t)t * 2 * RD + (cmp ? RD : 0); }

typedef struct { Layer* L; const Layer* KS; const Layer* IS; int l, T; } AttnArg;

/* llama.cpp numerics (flag 128): (token, group of AHG heads), softmax over the window and the selected compressed
   positions, with each head's sink logit. All heads share one KV vector per position (K = V), so a group reads
   each position once for its heads; the per-head arithmetic is the same as one head at a time. */
#define AHG 2
static void attn_item(int it, void* a) {
    const AttnArg* A = a;
    const Layer* L = A->L;
    const float scale = 1.f / sqrtf((float)HD);
    const float* ent[WIN + ITOPK];
    int tk = it / (NH / AHG), h0 = it % (NH / AHG) * AHG, p = M.pos + tk;
    int n = 0;
    int w0 = p - WIN + 1 < 0 ? 0 : p - WIN + 1;
    for (int w = w0; w <= p; w++) ent[n++] = L->win + (size_t)(w % M.W) * HD;
    if (L->ratio) {
        int nt = A->IS->ntop[tk];
        const int* ix = A->IS->topk + (size_t)tk * ITOPK;
        for (int k = 0; k < nt; k++) ent[n++] = A->KS->ckv + (size_t)ix[k] * HD;
    }
    const float* qg = M.q + ((size_t)tk * NH + h0) * HD;
    float* og = M.o + ((size_t)tk * NH + h0) * HD;
    if (M.flags & F_LLAMA_FA) {
        /* llama.cpp's CPU flash attention, as inferred from its outputs: fp16 query, online softmax, the output
           accumulator rounded to fp16 after every rescale and every added entry */
        static __thread float qh[AHG][HD], acc[AHG][HD];
        float mx[AHG], S[AHG];
        for (int h = 0; h < AHG; h++) {
            for (int d = 0; d < HD; d += 8)
                _mm256_storeu_ps(qh[h] + d, _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_loadu_ps(qg + h * HD + d), 0)));
            memset(acc[h], 0, sizeof acc[h]);
            mx[h] = -INFINITY; S[h] = 0.f;
        }
        for (int e = 0; e < n; e++) {
            const float* kv = ent[e];
            __m256 s8[AHG];
            for (int h = 0; h < AHG; h++) s8[h] = _mm256_setzero_ps();
            for (int d = 0; d < HD; d += 8) {
                __m256 k8 = _mm256_loadu_ps(kv + d);
                for (int h = 0; h < AHG; h++) s8[h] = _mm256_fmadd_ps(_mm256_loadu_ps(qh[h] + d), k8, s8[h]);
            }
            for (int h = 0; h < AHG; h++) {
                float s = hsum8(s8[h]) * scale, ms = 1.f, vs = 1.f;
                if (s > mx[h]) {
                    ms = mx[h] == -INFINITY ? 0.f : expf(mx[h] - s);
                    mx[h] = s;
                    __m256 m8 = _mm256_set1_ps(ms);
                    for (int d = 0; d < HD; d += 8)
                        _mm256_storeu_ps(acc[h] + d,
                                         _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(_mm256_loadu_ps(acc[h] + d), m8), 0)));
                } else {
                    vs = expf(s - mx[h]);
                }
                __m256 v8 = _mm256_set1_ps(vs);
                for (int d = 0; d < HD; d += 8) {
                    __m256 r = _mm256_add_ps(_mm256_loadu_ps(acc[h] + d), _mm256_mul_ps(v8, _mm256_loadu_ps(kv + d)));
                    _mm256_storeu_ps(acc[h] + d, _mm256_cvtph_ps(_mm256_cvtps_ph(r, 0)));
                }
                S[h] = S[h] * ms + vs;
            }
        }
        for (int h = 0; h < AHG; h++) {
            float Sh = S[h] + expf(L->sinks[h0 + h] - mx[h]);
            for (int d = 0; d < HD; d++) og[h * HD + d] = acc[h][d] / Sh;
        }
        return;
    }
}

/* Split attention (reference numerics): items = (token, group of AGH heads, chunk of ACS positions). An item
   reads its positions once for 8 heads and keeps a partial result (max, sum, unnormalized output); the
   combine step merges the chunks per head. The chunking depends only on the number of positions, so a token
   gives the same bits in a prompt chunk and alone. */

static int attn_entries(const AttnArg* A, int tk, const float** ent) {
    const Layer* L = A->L;
    int p = M.pos + tk, n = 0;
    int w0 = p - WIN + 1 < 0 ? 0 : p - WIN + 1;
    for (int w = w0; w <= p; w++) ent[n++] = L->win + (size_t)(w % M.W) * HD;
    if (L->ratio) {
        int nt = A->IS->ntop[tk];
        const int* ix = A->IS->topk + (size_t)tk * ITOPK;
        for (int k = 0; k < nt; k++) ent[n++] = A->KS->ckv + (size_t)ix[k] * HD;
    }
    return n;
}

typedef struct { const AttnArg* A; int first[TMAXC + 1]; } SplitArg;

static void attn_split_item(int i, void* a) {
    const SplitArg* S = a;
    int t = 0;
    while (i >= S->first[t + 1]) t++;
    int nch = M.anch[t], k = i - S->first[t], g = k / nch, c = k % nch;
    const float* ent[WIN + ITOPK];
    int n = attn_entries(S->A, t, ent);
    int e0 = c * ACS, ne = (n - e0 < ACS ? n - e0 : ACS);
    const float** en = ent + e0;
    const float scale = 1.f / sqrtf((float)HD);
    static __thread float sc[2][ACS + 8];
    for (int hp = 0; hp < AGH; hp += 2) {
        int h = g * AGH + hp;
        const float* qg = M.q + ((size_t)t * NH + h) * HD;
        int e = 0;
        for (; e + 4 <= ne; e += 4) {
            const float *k0 = en[e], *k1 = en[e + 1], *k2 = en[e + 2], *k3 = en[e + 3];
            __m256 s8[2][4];
            for (int u = 0; u < 2; u++)
                for (int j = 0; j < 4; j++) s8[u][j] = _mm256_setzero_ps();
            for (int d = 0; d < HD; d += 8) {
                __m256 a0 = _mm256_loadu_ps(k0 + d), a1 = _mm256_loadu_ps(k1 + d), a2 = _mm256_loadu_ps(k2 + d),
                       a3 = _mm256_loadu_ps(k3 + d);
                for (int u = 0; u < 2; u++) {
                    __m256 q8 = _mm256_loadu_ps(qg + u * HD + d);
                    s8[u][0] = _mm256_fmadd_ps(q8, a0, s8[u][0]);
                    s8[u][1] = _mm256_fmadd_ps(q8, a1, s8[u][1]);
                    s8[u][2] = _mm256_fmadd_ps(q8, a2, s8[u][2]);
                    s8[u][3] = _mm256_fmadd_ps(q8, a3, s8[u][3]);
                }
            }
            for (int u = 0; u < 2; u++)
                for (int j = 0; j < 4; j++) sc[u][e + j] = hsum8(s8[u][j]) * scale;
        }
        for (; e < ne; e++) {
            __m256 s8[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
            for (int d = 0; d < HD; d += 8) {
                __m256 k8 = _mm256_loadu_ps(en[e] + d);
                for (int u = 0; u < 2; u++) s8[u] = _mm256_fmadd_ps(_mm256_loadu_ps(qg + u * HD + d), k8, s8[u]);
            }
            for (int u = 0; u < 2; u++) sc[u][e] = hsum8(s8[u]) * scale;
        }
        for (int u = 0; u < 2; u++) {
            float mx = -1e30f;
            for (int j = 0; j < ne; j++) mx = fmaxf(mx, sc[u][j]);
            int j = 0;
            for (; j + 8 <= ne; j += 8)
                _mm256_storeu_ps(sc[u] + j, exp256(_mm256_sub_ps(_mm256_loadu_ps(sc[u] + j), _mm256_set1_ps(mx))));
            for (; j < ne; j++) sc[u][j] = expf(sc[u][j] - mx);
            float sum = 0;
            for (j = 0; j < ne; j++) sum += sc[u][j];
            size_t pi = ((size_t)t * NH + h + u) * ANCH + c;
            M.am[pi] = mx;
            M.as[pi] = sum;
        }
        for (int dc = 0; dc < HD; dc += 32) {
            __m256 o[2][4];
            for (int u = 0; u < 2; u++)
                for (int j = 0; j < 4; j++) o[u][j] = _mm256_setzero_ps();
            for (int e2 = 0; e2 < ne; e2++) {
                const float* kv = en[e2] + dc;
                __m256 p0 = _mm256_set1_ps(sc[0][e2]), p1 = _mm256_set1_ps(sc[1][e2]);
                for (int j = 0; j < 4; j++) {
                    __m256 k8 = _mm256_loadu_ps(kv + 8 * j);
                    o[0][j] = _mm256_fmadd_ps(p0, k8, o[0][j]);
                    o[1][j] = _mm256_fmadd_ps(p1, k8, o[1][j]);
                }
            }
            for (int u = 0; u < 2; u++) {
                float* dst = M.ap + (((size_t)t * NH + h + u) * ANCH + c) * HD + dc;
                for (int j = 0; j < 4; j++) _mm256_storeu_ps(dst + 8 * j, o[u][j]);
            }
        }
    }
}

/* merge the chunks of the 8 heads of output group g of token t into M.o */
static void attn_combine(const AttnArg* A, int t, int g) {
    int nch = M.anch[t];
    for (int h = g * (NH / OG); h < (g + 1) * (NH / OG); h++) {
        size_t pi = ((size_t)t * NH + h) * ANCH;
        float mx = M.am[pi];
        for (int c = 1; c < nch; c++) mx = fmaxf(mx, M.am[pi + c]);
        float w[ANCH], S = 0;
        for (int c = 0; c < nch; c++) { w[c] = expf(M.am[pi + c] - mx); S += M.as[pi + c] * w[c]; }
        S += expf(A->L->sinks[h] - mx);
        float* od = M.o + ((size_t)t * NH + h) * HD;
        for (int d = 0; d < HD; d += 8) {
            __m256 v = _mm256_mul_ps(_mm256_set1_ps(w[0] / S), _mm256_loadu_ps(M.ap + pi * HD + d));
            for (int c = 1; c < nch; c++)
                v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_set1_ps(w[c] / S), _mm256_loadu_ps(M.ap + (pi + c) * HD + d)));
            _mm256_storeu_ps(od + d, v);
        }
    }
}
static void combine_item(int i, void* a) { attn_combine(a, i / OG, i % OG); }

/* the k-th largest value (k from 0) of a[0..n); a is reordered */
static float kth_largest(float* a, int n, int k) {
    int lo = 0, hi = n - 1;
    for (;;) {
        if (lo >= hi) return a[lo];
        float piv = a[lo + (hi - lo) / 2], x;
        int lt = lo, i = lo, gt = hi;
        while (i <= gt) {
            if (a[i] > piv) { x = a[lt]; a[lt++] = a[i]; a[i++] = x; }
            else if (a[i] < piv) { x = a[gt]; a[gt--] = a[i]; a[i] = x; }
            else i++;
        }
        if (k < lt) hi = lt - 1;
        else if (k > gt) lo = gt + 1;
        else return piv;
    }
}

/* the k largest of sc[0..n), indices sorted ascending */
static int topk_idx(const float* sc, int n, int k, int* out, float* tmp) {
    if (n <= k) {
        for (int i = 0; i < n; i++) out[i] = i;
        return n;
    }
    memcpy(tmp, sc, sizeof(float) * n);
    float th = kth_largest(tmp, n, k - 1);
    int m = 0, ngt = 0;
    for (int i = 0; i < n; i++) ngt += sc[i] > th;
    int neq = k - ngt;                      /* ties at the threshold: the lowest indices */
    for (int i = 0; i < n; i++) {
        if (sc[i] > th) out[m++] = i;
        else if (sc[i] == th && neq > 0) { out[m++] = i; neq--; }
    }
    return m;
}

/* indexer: scores over (token, chunk of IDX_CH groups) items, then the top-k per token */
#define IDX_CH 256
typedef struct { const Layer* KS; Layer* L; int l, r, T, nitems; int first[TMAXC + 1]; } IdxArg;

static void idx_score_item(int i, void* a) {
    const IdxArg* A = a;
    int t = 0;
    while (i >= A->first[t + 1]) t++;
    int n = (M.pos + t + 1) / A->r, g0 = (i - A->first[t]) * IDX_CH, g1 = g0 + IDX_CH < n ? g0 + IDX_CH : n;
    const float* q = M.iq + (size_t)t * IH * IHD;
    const float* w = M.iw + (size_t)t * IH;
    const uint8_t* cand = M.cand + (size_t)t * (M.cap / CBLK + 1);
    float* sc = M.sc + (size_t)t * (M.cap + 8);
    int use_cand = A->l > CAND_SRC;
    /* q transposed to [dim][head]: per position, one broadcast of k[d] feeds the 32 heads (4 vectors) */
    float qt[IHD][IH];
    for (int hh = 0; hh < IH; hh++)
        for (int d = 0; d < IHD; d++) qt[d][hh] = q[hh * IHD + d];
    __m256 w8[IH / 8];
    for (int j = 0; j < IH / 8; j++) w8[j] = _mm256_loadu_ps(w + 8 * j);
    for (int g = g0; g < g1; g++) {
        if (use_cand && !cand[g / CBLK]) { sc[g] = -INFINITY; continue; }
        const float* k = A->KS->ik + (size_t)g * IHD;
        __m256 acc[IH / 8];
        for (int j = 0; j < IH / 8; j++) acc[j] = _mm256_setzero_ps();
        for (int d = 0; d < IHD; d++) {
            __m256 kb = _mm256_set1_ps(k[d]);
            for (int j = 0; j < IH / 8; j++) acc[j] = _mm256_fmadd_ps(_mm256_loadu_ps(qt[d] + 8 * j), kb, acc[j]);
        }
        __m256 s8 = _mm256_setzero_ps();
        for (int j = 0; j < IH / 8; j++) s8 = _mm256_fmadd_ps(_mm256_max_ps(acc[j], _mm256_setzero_ps()), w8[j], s8);
        sc[g] = hsum8(s8);
    }
}

static void idx_topk_tok(int t, void* a) {
    IdxArg* A = a;
    Layer* L = A->L;
    int n = (M.pos + t + 1) / A->r;
    if (n == 0) { L->ntop[t] = 0; return; }
    float* sc = M.sc + (size_t)t * (M.cap + 8);
    float* tmp = malloc(sizeof(float) * (size_t)n);
    if (A->l == CAND_SRC) {
        /* level one: the best CTOPK blocks of CBLK positions; the newest (partial) block is always kept */
        int nb = (n + CBLK - 1) / CBLK;
        uint8_t* cand = M.cand + (size_t)t * (M.cap / CBLK + 1);
        if (nb <= CTOPK) memset(cand, 1, nb);
        else {
            float* bs = malloc(sizeof(float) * nb);
            int* bi = malloc(sizeof(int) * nb);
            for (int b = 0; b < nb; b++) {
                float m = -INFINITY;
                for (int g = b * CBLK; g < (b + 1) * CBLK && g < n; g++) m = fmaxf(m, sc[g]);
                bs[b] = m;
            }
            bs[(n - 1) / CBLK] = INFINITY;
            int m = topk_idx(bs, nb, CTOPK, bi, tmp);
            memset(cand, 0, nb);
            for (int i = 0; i < m; i++) if (bs[bi[i]] > -INFINITY) cand[bi[i]] = 1;
            free(bi); free(bs);
        }
    }
    L->ntop[t] = topk_idx(sc, n, ITOPK, L->topk + (size_t)t * ITOPK, tmp);
    free(tmp);
}

static void indexer(Layer* L, int l, const Layer* KS, int T) {
    double t0 = now();
    IdxArg A;
    A.KS = KS; A.L = L; A.l = l; A.r = L->ratio; A.T = T;
    A.first[0] = 0;
    for (int t = 0; t < T; t++) A.first[t + 1] = A.first[t] + ((M.pos + t + 1) / A.r + IDX_CH - 1) / IDX_CH;
    A.nitems = A.first[T];
    kp_for(A.nitems, idx_score_item, &A);
    kp_for(T, idx_topk_tok, &A);
    M.t_idx += now() - t0;
}

/* compressor (kv source layers): the latents of the groups this chunk completes -> ckv, ik */
typedef struct { Layer* L; int n; } LatArg;
static void lat_item(int i, void* a) {
    const LatArg* A = a;
    Layer* L = A->L;
    int g = M.latg[i], gp = g * L->ratio;
    float cs[RD], kn[IHD], lat[HD];
    rope_cs(cs, gp, M.freq_cmp);
    rmsnorm(kn, M.latk + (size_t)i * IHD, L->idx_k_norm, IHD);
    rope(kn + IHD - RD, cs, 0);
    if (M.flags & F_FP4_IDX) sim_fp4_pow2(kn, IHD);
    if (M.flags & F_F16_CACHE) round_f16(kn, IHD);
    memcpy(L->ik + (size_t)g * IHD, kn, sizeof kn);
    memcpy(lat, M.lat + (size_t)i * HD, sizeof lat);
    rope(lat + HD - RD, cs, 0);
    if (M.flags & F_FP4_CKV) sim_fp4_e4m3(lat, HD);
    if (M.flags & F_F16_CACHE) round_f16(lat, HD);
    memcpy(L->ckv + (size_t)g * HD, lat, sizeof lat);
}

static void compressor(Layer* L, int T) {
    int r = L->ratio, nl = 0;
    for (int t = 0; t < T; t++) {
        int p = M.pos + t;
        float* lat = M.lat + (size_t)nl * HD;
        if (r == 1) {
            rmsnorm(lat, M.ck + (size_t)t * HD, L->cmp_norm, HD);
        } else {
            int slot = p % CRING, s0 = (p + CRING - 1) % CRING;   /* r == 2: this position and the one before */
            memcpy(L->cst_kv[slot], M.ck + (size_t)t * HD, HD * 4);
            memcpy(L->cst_sc[slot], M.cg + (size_t)t * HD, HD * 4);
            if ((p + 1) % r) continue;
            float pooled[HD];
            for (int d = 0; d < HD; d++) {
                float a = L->cst_sc[s0][d], b = L->cst_sc[slot][d], m = fmaxf(a, b);
                float ea = expf(a - m), eb = expf(b - m), s = ea + eb;
                pooled[d] = L->cst_kv[s0][d] * (ea / s) + L->cst_kv[slot][d] * (eb / s);
            }
            rmsnorm(lat, pooled, L->cmp_norm, HD);
        }
        quant_q8(lat, HD, M.latq + (size_t)nl * HD, M.latd + (size_t)nl * HD / QK);
        M.latg[nl++] = p / r;
    }
    if (!nl) return;
    /* index keys from the RoPE-free latents */
    Op o = opq(&L->idx_k, M.latq, M.latd, HD, M.latk, IHD);
    gemv_run(&o, 1, nl);
    LatArg A = {L, nl};
    kp_for(nl, lat_item, &A);
}

static int kv_src_of(int l) { return l < 2 ? -1 : l < 8 ? 2 : l < 14 ? 8 : l < 20 ? 14 : 20; }
static int idx_src_of(int l) { int s = kv_src_of(l); if (l >= 20) s = 20 + (l - 20) / 4 * 4; return s; }

static void qa_tok(int t, void* a) {
    const Layer* L = a;
    float* qa = M.qa + (size_t)t * QR;
    rmsnorm(qa, qa, L->q_norm, QR);
    quant_q8(qa, QR, M.qaq + (size_t)t * QR, M.qad + (size_t)t * QR / QK);
}

static void qkv_tok(int t, void* a) {
    const AttnArg* A = a;
    const Layer* L = A->L;
    const float* cs = cs_of(t, L->ratio != 0);
    int p = M.pos + t;
    for (int hh = 0; hh < NH; hh++) rope(M.q + ((size_t)t * NH + hh) * HD + HD - RD, cs, 0);
    if (M.flags & F_F16_Q) round_f16(M.q + (size_t)t * NH * HD, NH * HD);
    float* kv = M.kvw + (size_t)t * HD;
    rmsnorm(kv, kv, L->kv_norm, HD);
    rope(kv + HD - RD, cs, 0);
    if (M.flags & F_FP8_KV) sim_fp8(kv, HD);
    if (M.flags & F_F16_CACHE) round_f16(kv, HD);
    memcpy(L->win + (size_t)(p % M.W) * HD, kv, HD * 4);
    if (is_idx_src(A->l)) {
        for (int hh = 0; hh < IH; hh++) {
            float* q = M.iq + ((size_t)t * IH + hh) * IHD;
            rope(q + IHD - RD, cs_of(t, 1), 0);
            if (M.flags & F_FP4_IDX) sim_fp4_pow2(q, IHD);
        }
        for (int hh = 0; hh < IH; hh++) M.iw[t * IH + hh] *= (1.f / sqrtf((float)IHD)) * (1.f / sqrtf((float)IH));
    }
}

/* (token, group of 8 heads): remove the query's rotation from the output, then round and quantize */
static void derope_item(int i, void* a) {
    const AttnArg* A = a;
    int t = i / OG, g = i % OG;
    const float* cs = cs_of(t, A->L->ratio != 0);
    float* o = M.o + (size_t)t * NH * HD + (size_t)g * (NH / OG) * HD;
    for (int hh = 0; hh < NH / OG; hh++) rope(o + hh * HD + HD - RD, cs, 1);
}
static void oquant_item(int i, void* a) {
    (void)a;
    int t = i / OG, g = i % OG;
    size_t off = (size_t)t * NH * HD + (size_t)g * (NH / OG) * HD;
    float* o = M.o + off;
    if (M.flags & F_FP8_O) sim_fp8(o, (NH / OG) * HD);
    quant_q8(o, (NH / OG) * HD, M.oq + off, M.od + off / QK);
}
static void derope_quant_item(int i, void* a) {
    if (!(M.flags & F_LLAMA_FA)) attn_combine(a, i / OG, i % OG);
    derope_item(i, a);
    oquant_item(i, NULL);
}
static void oaquant_tok(int t, void* a) {
    (void)a;
    quant_q8(M.oa + (size_t)t * OG * OR, OG * OR, M.oaq + (size_t)t * OG * OR, M.oad + (size_t)t * OG * OR / QK);
}

static void attention(int l, int T) {
    Layer* L = &M.L[l];
    /* x (normed) is in M.x, quantized in M.xq */
    Op ops[MAXOP]; int no = 0;
    ops[no++] = opq(&L->q_a, M.xq, M.xd, D, M.qa, QR);
    ops[no++] = opq(&L->kv, M.xq, M.xd, D, M.kvw, HD);
    if (is_kv_src(l)) {
        ops[no++] = opq(&L->cmp_kv, M.xq, M.xd, D, M.ck, HD);
        if (L->ratio > 1) ops[no++] = opq(&L->cmp_gate, M.xq, M.xd, D, M.cg, HD);
    }
    if (is_idx_src(l)) ops[no++] = opq(&L->idx_proj, M.xq, M.xd, D, M.iw, IH);
    gemv_run(ops, no, T);
    TIMED(4, kp_for(T, qa_tok, L));
    dump("qr", l, T, QR, M.qa, QR);
    no = 0;
    ops[no++] = opq(&L->q_b, M.qaq, M.qad, QR, M.q, NH * HD);
    if (is_idx_src(l)) ops[no++] = opq(&L->idx_qb, M.qaq, M.qad, QR, M.iq, IH * IHD);
    gemv_run(ops, no, T);
    AttnArg A = {L, NULL, NULL, l, T};
    TIMED(4, kp_for(T, qkv_tok, &A));
    dump("q", l, T, NH * HD, M.q, NH * HD);
    dump("kv", l, T, HD, M.kvw, HD);
    if (L->ratio) {
        if (is_kv_src(l)) TIMED(3, compressor(L, T));
        if (is_idx_src(l)) indexer(L, l, &M.L[kv_src_of(l)], T);
        A.KS = &M.L[kv_src_of(l)];
        A.IS = &M.L[idx_src_of(l)];
    }
    double t0 = now();
    if (M.flags & F_LLAMA_FA) {
        kp_for(T * (NH / AHG), attn_item, &A);
    } else {
        SplitArg S;
        S.A = &A; S.first[0] = 0;
        for (int t = 0; t < T; t++) {
            const float* ent[WIN + ITOPK];
            M.anch[t] = (attn_entries(&A, t, ent) + ACS - 1) / ACS;
            S.first[t + 1] = S.first[t] + M.anch[t] * (NH / AGH);
        }
        kp_for(S.first[T], attn_split_item, &S);
        if (M.dump) kp_for(T * OG, combine_item, &A);
    }
    M.t_attn += now() - t0;
    dump("attn_raw", l, T, NH * HD, M.o, NH * HD);
    if (M.dump) {
        kp_for(T * OG, derope_item, &A);
        dump("attn_derope", l, T, NH * HD, M.o, NH * HD);
        kp_for(T * OG, oquant_item, NULL);
    } else {
        TIMED(5, kp_for(T * OG, derope_quant_item, &A));
    }
    Op oa = opq(&L->o_a, M.oq, M.od, NH * HD, M.oa, OG * OR);
    gemv_run(&oa, 1, T);
    dump("attn_wo_a", l, T, OG * OR, M.oa, OG * OR);
    TIMED(5, kp_for(T, oaquant_tok, NULL));
    Op ob = opq(&L->o_b, M.oaq, M.oad, OG * OR, M.ao, D);
    gemv_run(&ob, 1, T);
}

/* ---------------------------------------------------------------- MoE */
static void route_tok(int t, void* a) {
    const Layer* L = a;
    float s[NE], b[NE];
    for (int e = 0; e < NE; e++) {
        float v = M.glog[t * NE + e];
        float sp = v > 20.f ? v : log1pf(expf(v));
        s[e] = sqrtf(sp);
        b[e] = s[e] + L->bias[e];
    }
    int* sel = M.sel + t * NS;
    float* w = M.wsel + t * NS;
    for (int k = 0; k < NS; k++) {
        int best = -1;
        for (int e = 0; e < NE; e++) {
            int used = 0;
            for (int j = 0; j < k; j++) used |= sel[j] == e;
            if (!used && (best < 0 || b[e] > b[best])) best = e;
        }
        sel[k] = best;
    }
    float sum = 0;
    for (int k = 0; k < NS; k++) { w[k] = s[sel[k]]; sum += w[k]; }
    for (int k = 0; k < NS; k++) w[k] = w[k] / (sum + 1e-20f) * 1.5f;
}

/* shared expert: SwiGLU with the clamp, then quantize for its down projection; items = (token, 1/8 of FF) */
static void swiglu_item(int i, void* a) {
    (void)a;
    int t = i / 8, c = i % 8, f0 = c * (FF / 8), f1 = f0 + FF / 8;
    float* g = M.sg + (size_t)t * FF;
    const float* u = M.su + (size_t)t * FF;
    for (int k = f0; k < f1; k++) {
        float gt = fminf(g[k], 10.f), up = fminf(fmaxf(u[k], -10.f), 10.f);
        g[k] = gt / (1.f + expf(-gt)) * up;
    }
    quant_q8(g + f0, FF / 8, M.suq + (size_t)t * FF + f0, M.sud + ((size_t)t * FF + f0) / QK);
}

static void ffn(int l, int T) {
    Layer* L = &M.L[l];
    Op ops[3];
    ops[0] = opf(&L->gate_inp, M.x, D, M.glog, NE);
    ops[1] = opq(&L->sh_gate, M.xq, M.xd, D, M.sg, FF);
    ops[2] = opq(&L->sh_up, M.xq, M.xd, D, M.su, FF);
    gemv_run(ops, 3, T);
    TIMED(6, kp_for(T, route_tok, L); kp_for(T * 8, swiglu_item, NULL));
    Op od = opq(&L->sh_down, M.suq, M.sud, FF, M.fo, D);
    gemv_run(&od, 1, T);
    double t0 = now();
    kp_park(1);
    { int nu_ = moe_forward(L->moe, T, M.x, NS, M.sel, M.wsel, M.mo, 0); if (nu_ > 0) dsv_union_total += nu_; }
    M.t_moe += now() - t0;
    dump("moe_out", l, T, D, M.mo, D);
    dump("shexp_out", l, T, D, M.fo, D);
    for (size_t i = 0; i < (size_t)T * D; i++) M.fo[i] += M.mo[i];
}

/* ---------------------------------------------------------------- forward */
/* T tokens at positions pos..pos+T-1; logits (VOCAB floats) for the last token when logits != NULL, or for every
   token when all_logits (logits = [T][VOCAB]). */
/* KINDI_XC_MAXP (default 2; 0 = no limit): the most (token, slot) pairs one card takes per forward of T tokens is
   MAXP * T, so the card's share stays shorter than the host's experts (the card computes its pairs one after another;
   which pairs a card takes does not change any value: the card's are bit-identical to the host's).
   KINDI_XC_ASYNC (default 1): a helper thread (KINDI_XC_CPU, default 58, an idle hyper-thread sibling) quantizes the
   input and issues the card's work, so the host's experts start without waiting for those API calls. */
static struct {
    atomic_int req, done;
    int l, T, c, rc;
    const float* x; const int* sel;
    uint8_t take[8 * NS];
    int8_t* xq; float* xd;
    pthread_t th; int on;
} XH;
static void xh_issue(void) {
    for (int t = 0; t < XH.T; t++) quant_q8(XH.x + (size_t)t * D, D, XH.xq + (size_t)t * D, XH.xd + (size_t)t * (D / QK));
    XH.rc = XC[XH.c].start(XH.l, XH.T, XH.xq, XH.xd, XH.sel, XH.take) < 0 ? -1 : 0;
}
static void* xh_main(void* a) {
    (void)a;
    int cpu = getenv("KINDI_XC_CPU") ? atoi(getenv("KINDI_XC_CPU")) : 58;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    int seen = 0;
    for (;;) {
        int r;
        double t_idle = now();    /* spin while requests come (decode); sleep in 100 us steps after 20 ms idle */
        long spins = 0;
        while ((r = atomic_load_explicit(&XH.req, memory_order_acquire)) == seen) {
            _mm_pause();
            if (++spins % 4096 == 0 && now() - t_idle > 0.02) { struct timespec ts = {0, 100000}; nanosleep(&ts, NULL); }
        }
        xh_issue();
        seen = r;
        atomic_store_explicit(&XH.done, r, memory_order_release);
    }
    return NULL;
}
static int xc_maxp(void) {
    static int v = -1;
    if (v < 0) v = getenv("KINDI_XC_MAXP") ? atoi(getenv("KINDI_XC_MAXP")) : 2;
    return v;
}
/* the pairs the expert cards hold (and skip does not exclude) start on the cards; skip2 = skip + those pairs */
static int xc_start(int l, int T, const float* x, const int* sel, const uint8_t* skip, uint8_t* skip2) {
    static int8_t* xq; static float* xd; static int cap;
    if (!nXC || T > 8) return 0;
    double t0 = now();
    if (T > cap) { free(xq); free(xd); xq = malloc((size_t)8 * D); xd = malloc((size_t)8 * (D / QK) * 4); cap = 8; }
    if (XH.on == 0) {
        XH.on = (!getenv("KINDI_XC_ASYNC") || atoi(getenv("KINDI_XC_ASYNC"))) && nXC == 1 ? 1 : -1;
        if (XH.on == 1) {
            XH.xq = malloc((size_t)8 * D); XH.xd = malloc((size_t)8 * (D / QK) * 4);
            if (pthread_create(&XH.th, NULL, xh_main, NULL)) XH.on = -1;
        }
    }
    uint8_t mine[8 * NS], take[8 * NS];
    for (int i = 0; i < T * NS; i++) { skip2[i] = skip ? skip[i] : 0; mine[i] = !skip2[i]; }
    const int maxp = xc_maxp() > 0 ? xc_maxp() * T : 1 << 30;
    int any = 0;
    for (int c = 0; c < nXC; c++) {
        int k = 0;
        for (int i = 0; i < T * NS; i++) { take[i] = mine[i] && k < maxp && XC[c].has(l, sel[i]); k += take[i]; }
        if (!k) continue;
        if (XH.on == 1) {           /* one card: the helper thread issues it */
            XH.l = l; XH.T = T; XH.c = c; XH.x = x; XH.sel = sel;
            memcpy(XH.take, take, (size_t)T * NS);
            atomic_fetch_add_explicit(&XH.req, 1, memory_order_release);
        } else {
            if (!any) for (int t = 0; t < T; t++) quant_q8(x + (size_t)t * D, D, xq + (size_t)t * D, xd + (size_t)t * (D / QK));
            if (XC[c].start(l, T, xq, xd, sel, take) < 0) return -1;
        }
        for (int i = 0; i < T * NS; i++) if (take[i]) { mine[i] = 0; skip2[i] = 1; }
        any |= 1 << c; xc_pairs += k;
    }
    for (int i = 0; i < T * NS; i++) xc_all += !(skip && skip[i]);
    xc_t_start += now() - t0;
    return any;
}
static void xc_finish(int any, float* parts) {
    double t0 = now();
    if (XH.on == 1) {
        const int r = atomic_load_explicit(&XH.req, memory_order_acquire);
        while (atomic_load_explicit(&XH.done, memory_order_acquire) != r) _mm_pause();
        if (XH.rc) { fprintf(stderr, "dsv41: expert card start failed\n"); exit(1); }
    }
    for (int c = 0; c < nXC; c++) if (any >> c & 1) XC[c].finish(parts);
    xc_t_wait += now() - t0;
}

static int gpu_moe_parts(int l, int T, const float* x, const int* sel, const uint8_t* skip, float* parts) {
    double t0 = now();
    kp_park(1);
    uint8_t skip2[8 * NS];
    int any = xc_start(l, T, x, sel, skip, skip2);
    if (any < 0) return -1;
    { int nu_ = moe_forward_parts(M.L[l].moe, T, x, NS, sel, any ? skip2 : skip, parts); if (nu_ > 0) dsv_union_total += nu_; }
    if (any) xc_finish(any, parts);
    M.t_moe += now() - t0;
    return 0;
}

static int gpu_combine(int T, const float* wsel, const float* parts, float* out) {
    double t0 = now();
    moe_combine(T, NS, wsel, parts, D, out);
    M.t_moe += now() - t0;
    return 0;
}
/* the same, with the card's (token, slot) values read in place from its result rows (no scatter copy) */
static int gpu_combine_alt(int T, const float* wsel, const float* parts, const int* row, const float* alt, float* out) {
    double t0 = now();
    moe_combine_alt(T, NS, wsel, parts, row, alt, D, out);
    M.t_moe += now() - t0;
    return 0;
}

static int gpu_moe(int l, int T, const float* x, const int* sel, const float* wsel, float* mo) {
    double t0 = now();
    kp_park(1);
    if ((nXC || xc_parts) && T <= 8) {     /* per-pair values (cards + host), then the weighted sum in slot order */
        static float* parts;
        if (!parts) parts = aligned_alloc(64, (size_t)8 * NS * D * 4);
        uint8_t skip2[8 * NS];
        int any = xc_start(l, T, x, sel, NULL, skip2);
        if (any < 0) return -1;
        { int nu_ = moe_forward_parts(M.L[l].moe, T, x, NS, sel, any ? skip2 : NULL, parts); if (nu_ > 0) dsv_union_total += nu_; }
        if (any) xc_finish(any, parts);
        moe_combine(T, NS, wsel, parts, D, mo);
        M.t_moe += now() - t0;
        return 0;
    }
    { int nu_ = moe_forward(M.L[l].moe, T, x, NS, sel, wsel, mo, 0); if (nu_ > 0) dsv_union_total += nu_; }
    M.t_moe += now() - t0;
    return 0;
}

/* the GPU path: the host prepares the token-dependent inputs (RoPE tables, embeddings, engram rows, the latents the
   kv source layers complete) and computes the routed experts; everything else runs on the card */
static int forward_gpu(const int32_t* tok, int T, float* logits, int all_logits, float* dbg) {
    double t_start = now();
    static float *cs, *emb, *ed2, *csl; static int8_t* eq2; static int* lg; static int cap_t;
    if (T > cap_t) {
        free(cs); free(emb); free(eq2); free(ed2); free(csl); free(lg);
        cs = zalloc((size_t)T * 2 * RD * 4); emb = zalloc((size_t)T * D * 4);
        eq2 = zalloc((size_t)2 * T * ENG_COLS * ENG_HD); ed2 = zalloc((size_t)2 * T * ENG_COLS * ENG_HD / QK * 4);
        csl = zalloc((size_t)NL * T * RD * 4); lg = zalloc((size_t)NL * T * 4);
        cap_t = T;
    }
    for (int t = 0; t < T; t++) {
        M.ctok[M.pos + t] = M.tokmap[tok[t]];
        rope_cs(cs + (size_t)t * 2 * RD, M.pos + t, M.freq_win);
        rope_cs(cs + (size_t)t * 2 * RD + RD, M.pos + t, M.freq_cmp);
        const uint8_t* r = M.tok_embd + (size_t)tok[t] * (D / QK) * BQ8;
        float* e = emb + (size_t)t * D;
        for (int b = 0; b < D / QK; b++) {
            float d = h2f(r + b * BQ8);
            for (int j = 0; j < QK; j++) e[b * QK + j] = d * (int8_t)r[b * BQ8 + 2 + j];
        }
    }
    for (int l = 0; l < NL; l++) {
        Layer* L = &M.L[l];
        if (L->eng < 0) continue;
        TIMED(2, kp_for(T, eng_rows_tok, L));
        memcpy(eq2 + (size_t)L->eng * T * ENG_COLS * ENG_HD, M.eq, (size_t)T * ENG_COLS * ENG_HD);
        memcpy(ed2 + (size_t)L->eng * T * ENG_COLS * ENG_HD / QK, M.ed, (size_t)T * ENG_COLS * ENG_HD / QK * 4);
    }
    int nlat[NL];
    for (int l = 0; l < NL; l++) {
        nlat[l] = 0;
        if (!is_kv_src(l)) continue;
        int r = M.L[l].ratio;
        for (int t = 0; t < T; t++) {
            int p = M.pos + t;
            if (r > 1 && (p + 1) % r) continue;
            int g = p / r;
            rope_cs(csl + ((size_t)l * T + nlat[l]) * RD, g * r, M.freq_cmp);
            lg[(size_t)l * T + nlat[l]] = g;
            nlat[l]++;
        }
    }
    int nlog = logits ? (all_logits ? T : 1) : 0;
    int r = GPU.forward(M.pos, T, M.flags, cs, emb, eq2, ed2, csl, lg, nlat, gpu_moe, logits, nlog, dbg);
    if (r) { fprintf(stderr, "dsv41: GPU forward failed (%d)\n", r); return -2; }
    M.pos += T;
    double dt = now() - t_start;
    M.t_total += dt;
    return 0;
}

/* the inputs of one chunk for the card (positions M.pos .. M.pos + T - 1); set k of two (pipelined chunks) */
typedef struct { const float *cs, *emb; const int8_t* eq; const float *ed, *cs_lat; const int *latg, *nlat; } DgIn;
static void gpu_inputs(const int32_t* tok, int T, int k, DgIn* in, int* nlat) {
    static float *cs[2], *emb[2], *ed2[2], *csl[2]; static int8_t* eq2[2]; static int* lg[2]; static int cap_t[2];
    if (T > cap_t[k]) {
        free(cs[k]); free(emb[k]); free(eq2[k]); free(ed2[k]); free(csl[k]); free(lg[k]);
        cs[k] = zalloc((size_t)T * 2 * RD * 4); emb[k] = zalloc((size_t)T * D * 4);
        eq2[k] = zalloc((size_t)2 * T * ENG_COLS * ENG_HD); ed2[k] = zalloc((size_t)2 * T * ENG_COLS * ENG_HD / QK * 4);
        csl[k] = zalloc((size_t)NL * T * RD * 4); lg[k] = zalloc((size_t)NL * T * 4);
        cap_t[k] = T;
    }
    for (int t = 0; t < T; t++) {
        M.ctok[M.pos + t] = M.tokmap[tok[t]];
        rope_cs(cs[k] + (size_t)t * 2 * RD, M.pos + t, M.freq_win);
        rope_cs(cs[k] + (size_t)t * 2 * RD + RD, M.pos + t, M.freq_cmp);
        const uint8_t* r = M.tok_embd + (size_t)tok[t] * (D / QK) * BQ8;
        float* e = emb[k] + (size_t)t * D;
        for (int b = 0; b < D / QK; b++) {
            float d = h2f(r + b * BQ8);
            for (int j = 0; j < QK; j++) e[b * QK + j] = d * (int8_t)r[b * BQ8 + 2 + j];
        }
    }
    for (int l = 0; l < NL; l++) {
        Layer* L = &M.L[l];
        if (L->eng < 0) continue;
        TIMED(2, kp_for(T, eng_rows_tok, L));
        memcpy(eq2[k] + (size_t)L->eng * T * ENG_COLS * ENG_HD, M.eq, (size_t)T * ENG_COLS * ENG_HD);
        memcpy(ed2[k] + (size_t)L->eng * T * ENG_COLS * ENG_HD / QK, M.ed, (size_t)T * ENG_COLS * ENG_HD / QK * 4);
    }
    for (int l = 0; l < NL; l++) {
        nlat[l] = 0;
        if (!is_kv_src(l)) continue;
        int r = M.L[l].ratio;
        for (int t = 0; t < T; t++) {
            int p = M.pos + t;
            if (r > 1 && (p + 1) % r) continue;
            int g = p / r;
            rope_cs(csl[k] + ((size_t)l * T + nlat[l]) * RD, g * r, M.freq_cmp);
            lg[k][(size_t)l * T + nlat[l]] = g;
            nlat[l]++;
        }
    }
    in->cs = cs[k]; in->emb = emb[k]; in->eq = eq2[k]; in->ed = ed2[k]; in->cs_lat = csl[k]; in->latg = lg[k]; in->nlat = nlat;
}

/* two prompt chunks of T tokens each (tok[0 .. 2T)), pipelined on the card and the host (dg_forward2): the same bits
   as two dsv_forward calls; logits of the last token. Returns 1 when the pipelined path is not available. */
int dsv_forward2(const int32_t* tok, int T, float* logits) {
    if (!(GPU.full && !M.cpu_only && GPU.forward2) || T <= 8 || T > M.tmax || M.pos + 2 * T > M.cap) return 1;
    double t_start = now();
    DgIn a, b;
    static int nla[NL], nlb[NL];
    const int p0 = M.pos;
    gpu_inputs(tok, T, 0, &a, nla);
    M.pos = p0 + T;
    gpu_inputs(tok + T, T, 1, &b, nlb);
    M.pos = p0;
    int r = GPU.forward2(p0, T, M.flags, &a, &b, gpu_moe, logits, logits ? 1 : 0);
    if (r) { fprintf(stderr, "dsv41: GPU forward2 failed (%d)\n", r); return -2; }
    M.pos = p0 + 2 * T;
    M.t_total += now() - t_start;
    return 0;
}

/* KINDI_DS_SPLIT (default 0 = off: measured no gain, see below): a verify batch of 2..8 tokens with every position's logits runs as two halves pipelined
   on the card and the host (dg_forward_split): the card computes one half's layer while the host computes the other
   half's routed experts. The same kernels and expert calls per token as one forward of T (the same bits). */
static int split_on(void) {
    static int v = -1;
    if (v < 0) v = getenv("KINDI_DS_SPLIT") && atoi(getenv("KINDI_DS_SPLIT")) != 0;
    return v;
}
static int forward_split(const int32_t* tok, int T, float* logits) {
    double t_start = now();
    DgIn a, b;
    static int nla[NL], nlb[NL];
    const int p0 = M.pos, ta = T / 2, tb = T - ta;
    gpu_inputs(tok, ta, 0, &a, nla);
    M.pos = p0 + ta;
    gpu_inputs(tok + ta, tb, 1, &b, nlb);
    M.pos = p0;
    int r = GPU.forward_split(p0, ta, tb, M.flags, &a, &b, gpu_moe, logits);
    if (r) { fprintf(stderr, "dsv41: GPU forward_split failed (%d)\n", r); return -2; }
    M.pos = p0 + T;
    M.t_total += now() - t_start;
    return 0;
}

int dsv_forward(const int32_t* tok, int T, float* logits, int all_logits) {
    if (T < 1 || T > M.tmax || M.pos + T > M.cap) return -1;
    if (GPU.full && !M.cpu_only && GPU.forward_split && logits && all_logits && T >= 2 && T <= 8 && split_on())
        return forward_split(tok, T, logits);
    if (GPU.full && !M.cpu_only) return forward_gpu(tok, T, logits, all_logits, NULL);
    double t_start = now();
    for (int t = 0; t < T; t++) {
        M.ctok[M.pos + t] = M.tokmap[tok[t]];
        rope_cs(M.cs + (size_t)t * 2 * RD, M.pos + t, M.freq_win);
        rope_cs(M.cs + (size_t)t * 2 * RD + RD, M.pos + t, M.freq_cmp);
    }
    /* embeddings, 4 copies */
    for (int t = 0; t < T; t++) {
        const uint8_t* r = M.tok_embd + (size_t)tok[t] * (D / QK) * BQ8;
        float* h = M.h + (size_t)t * HC * D;
        for (int b = 0; b < D / QK; b++) {
            float d = h2f(r + b * BQ8);
            for (int j = 0; j < QK; j++) h[b * QK + j] = d * (int8_t)r[b * BQ8 + 2 + j];
        }
        for (int c = 1; c < HC; c++) memcpy(h + c * D, h, D * 4);
        for (int c = 0; c < HC; c++) M.pre[t * HC + c] = c == 0;
    }
    dump("embd", -1, T, D, M.h, HC * D);
    for (int l = 0; l < NL; l++) {
        Layer* L = &M.L[l];
        if (L->eng >= 0) { engram(L, T); dump("engram_h", l, T, HC * D, M.h, HC * D); }
        /* attention sub-layer: its input collapses the stream with the previous FFN's pre-mix (M.pre) */
        hc_in(&L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, M.pre, L->attn_norm, M.apre, M.post, M.comb, T);
        dump("attn_norm", l, T, D, M.x, D);
        attention(l, T);
        dump("attn_out", l, T, D, M.ao, D);
        hc_post(M.ao, T);
        /* FFN sub-layer: its input uses the attention's pre-mix; its own pre-mix goes to the next attention */
        hc_in(&L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, M.apre, L->ffn_norm, M.pre, M.post, M.comb, T);
        dump("ffn_norm", l, T, D, M.x, D);
        ffn(l, T);
        dump("ffn_out", l, T, D, M.fo, D);
        hc_post(M.fo, T);
        dump("h", l, T, HC * D, M.h, HC * D);
        if (M.hdbg) memcpy(M.hdbg + (size_t)l * T * HC * D, M.h, (size_t)T * HC * D * 4);
    }
    if (logits) {
        int t0 = all_logits ? 0 : T - 1, nt = T - t0;
        for (int t = 0; t < nt; t++) {
            const float* h = M.h + (size_t)(t0 + t) * HC * D;
            const float* pr = M.pre + (t0 + t) * HC;
            float* x = M.x + (size_t)t * D;
            for (int d = 0; d < D; d++) {
                float v = 0;
                for (int c = 0; c < HC; c++) v += pr[c] * h[c * D + d];
                x[d] = v;
            }
            rmsnorm(x, x, M.out_norm, D);
            quant_q8(x, D, M.lq + (size_t)t * D, M.ld + (size_t)t * D / QK);
        }
        dump("result_norm", -1, nt, D, M.x, D);
        for (int a = 0; a < nt; a += 8) {          /* 8 tokens per pass keeps the output buffer small */
            int n = nt - a < 8 ? nt - a : 8;
            Op o = opq(&M.out, M.lq + (size_t)a * D, M.ld + (size_t)a * D / QK, D, logits + (size_t)a * VOCAB, VOCAB);
            gemv_run(&o, 1, n);
        }
    }
    M.pos += T;
    if (M.dump) fflush(M.dump);
    M.t_total += now() - t_start;
    return 0;
}

/* The same T tokens through the CPU path and the GPU path (each keeps its own caches, so call this on successive
   chunks); prints per-layer agreement of h and of the last token's logits. Returns the argmax of the GPU logits
   (or -1). */
int dsv_compare(const int32_t* tok, int T) {
    if (!GPU.full) return -1;
    int p0 = M.pos;
    size_t n = (size_t)NL * T * HC * D;
    float *hc = malloc(n * 4), *hg = malloc(n * 4), *lc = malloc((size_t)VOCAB * 4), *lgp = malloc((size_t)VOCAB * 4);
    M.cpu_only = 1; M.hdbg = hc;
    int r = dsv_forward(tok, T, lc, 0);
    M.cpu_only = 0; M.hdbg = NULL;
    M.pos = p0;
    if (!r) r = forward_gpu(tok, T, lgp, 0, hg);
    if (r) { free(hc); free(hg); free(lc); free(lgp); return -1; }
    fprintf(stderr, "compare pos %d T %d:", p0, T);
    for (int l = 0; l < NL; l++) {
        const float *a = hc + (size_t)l * T * HC * D, *b = hg + (size_t)l * T * HC * D;
        size_t m = (size_t)T * HC * D, eq = 0;
        double e = 0, nn = 0;
        for (size_t i = 0; i < m; i++) {
            eq += memcmp(a + i, b + i, 4) == 0;
            e += ((double)a[i] - b[i]) * ((double)a[i] - b[i]); nn += (double)a[i] * a[i];
        }
        if (l < 3 || l % 8 == 7 || l == NL - 1) fprintf(stderr, " L%d %.1f%%/%.1e", l, 100.0 * eq / m, sqrt(e / (nn + 1e-30)));
    }
    int ac = 0, ag = 0;
    double e = 0, nn = 0;
    for (int i = 0; i < VOCAB; i++) {
        if (lc[i] > lc[ac]) ac = i;
        if (lgp[i] > lgp[ag]) ag = i;
        e += ((double)lc[i] - lgp[i]) * ((double)lc[i] - lgp[i]); nn += (double)lc[i] * lc[i];
    }
    fprintf(stderr, " | logits rel %.1e, argmax cpu %d gpu %d%s\n", sqrt(e / (nn + 1e-30)), ac, ag, ac == ag ? "" : " DIFFER");
    free(hc); free(hg); free(lc); free(lgp);
    return ag;
}

/* ---------------------------------------------------------------- DSpark drafter */
/* The drafter's tensors are registered as "dspark.<name>" (kindi/dsv41_dspark.py converts them): dense parts and
   vectors go to the GPU (dsv41_gpu_fwd.cuh), the experts (GGUF MXFP4, 128 per stage, 3 per token) stay on the host. */
#define DS_STAGES 3
#define DS_NE 128
#define DS_NS 3
#define DS_BLOCK 5
typedef struct {
    const uint8_t *qa, *qb, *kv, *oa, *ob, *shg, *shu, *shd;
    const float *gate, *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias, *hc_attn_fn, *hc_attn_base, *hc_attn_scale,
        *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
} DSStageDescC;
typedef struct {
    const uint8_t* main_proj;
    const float *main_norm, *norm, *conf, *membed, *mhead;
    DSStageDescC s[DS_STAGES];
} DSDescC;
static moe_layer* DSL[DS_STAGES];
static int ds_noise = 128799;

static const void* dsp(const char* name, int st) {
    char nm[96];
    if (st >= 0) snprintf(nm, sizeof nm, "dspark.%d.%s", st, name); else snprintf(nm, sizeof nm, "dspark.%s", name);
    for (int i = 0; i < nTT; i++) if (!strcmp(TT[i].name, nm)) return TT[i].p;
    fprintf(stderr, "dsv41: missing %s\n", nm);
    return NULL;
}

static int ds_moe(int stage, int T, const float* x, const int* sel, const float* wsel, float* mo) {
    double t0 = now();
    kp_park(1);
    moe_forward(DSL[stage], T, x, DS_NS, sel, wsel, mo, 0);
    M.t_moe += now() - t0;
    return 0;
}

int dsv_dspark_init(int noise_id) {
    if (!GPU.full || !GPU.dspark_init) return -1;
    ds_noise = noise_id;
    DSDescC d;
    d.main_proj = dsp("main_proj", -1); d.main_norm = dsp("main_norm", -1); d.norm = dsp("norm", -1);
    d.conf = dsp("conf", -1); d.membed = dsp("markov_embed", -1); d.mhead = dsp("markov_head", -1);
    for (int st = 0; st < DS_STAGES; st++) {
        DSStageDescC* e = &d.s[st];
        e->qa = dsp("q_a", st); e->qb = dsp("q_b", st); e->kv = dsp("kv", st); e->oa = dsp("o_a", st); e->ob = dsp("o_b", st);
        e->shg = dsp("sh_gate", st); e->shu = dsp("sh_up", st); e->shd = dsp("sh_down", st);
        e->gate = dsp("gate", st); e->sinks = dsp("sinks", st); e->kv_norm = dsp("kv_norm", st); e->q_norm = dsp("q_norm", st);
        e->attn_norm = dsp("attn_norm", st); e->ffn_norm = dsp("ffn_norm", st); e->bias = dsp("bias", st);
        e->hc_attn_fn = dsp("hc_attn_fn", st); e->hc_attn_base = dsp("hc_attn_base", st); e->hc_attn_scale = dsp("hc_attn_scale", st);
        e->hc_ffn_fn = dsp("hc_ffn_fn", st); e->hc_ffn_base = dsp("hc_ffn_base", st); e->hc_ffn_scale = dsp("hc_ffn_scale", st);
        DSL[st] = moe_layer_direct(D, FF, DS_NE, 10.f, dsp("gate_exps", st), dsp("up_exps", st), dsp("down_exps", st));
    }
    int r = GPU.dspark_init(&d, ds_moe);
    fprintf(stderr, "dsv41: DSpark drafter %s (%.2f GB of VRAM in use)\n", r ? "FAILED" : "on the GPU", GPU.vram_used() / 1e9);
    return r;
}

/* rows row0 .. row0 + n - 1 of the last forward are accepted positions pos0 ..: into the drafter's rings */
int dsv_dspark_push(int row0, int n, int pos0) {
    if (n <= 0) return 0;
    float* cs = malloc(sizeof(float) * n * RD);
    for (int i = 0; i < n; i++) rope_cs(cs + (size_t)i * RD, pos0 + i, M.freq_win);
    int r = GPU.dspark_push(row0, n, pos0, cs, M.flags);
    free(cs);
    return r;
}

/* draft DS_BLOCK tokens after position pos_last given the next token y: out_tok [DS_BLOCK + 1], conf [DS_BLOCK] */
int dsv_dspark_draft(int pos_last, int y, int* out_tok, float* conf) {
    static float emb[DS_BLOCK * D], cs[DS_BLOCK * RD];
    for (int t = 0; t < DS_BLOCK; t++) {
        const int tk = t == 0 ? y : ds_noise;
        const uint8_t* r = M.tok_embd + (size_t)tk * (D / QK) * BQ8;
        for (int b = 0; b < D / QK; b++) {
            float d = h2f(r + b * BQ8);
            for (int j = 0; j < QK; j++) emb[(size_t)t * D + b * QK + j] = d * (int8_t)r[b * BQ8 + 2 + j];
        }
        rope_cs(cs + (size_t)t * RD, pos_last + 1 + t, M.freq_win);
    }
    return GPU.dspark_draft(pos_last, y, emb, cs, M.flags, out_tok, conf);
}

/* ---------------------------------------------------------------- prompt-cache checkpoints */
/* file: "KDS1", int32 pos, int32 flags, int64 state bytes, the state (dsv41_gpu_fwd.cuh), then the token map ctok[pos]
   (the engram reads it). The GPU path only. */
int dsv_state_save(const char* path) {
    if (!GPU.full || !GPU.state_size) return -1;
    const int pos = M.pos;
    size_t n = GPU.state_size(pos);
    void* buf = malloc(n);
    if (!buf || GPU.state_save(pos, buf)) { free(buf); return -2; }
    FILE* f = fopen(path, "wb");
    if (!f) { free(buf); return -3; }
    int32_t hd[2] = {pos, M.flags}; int64_t nb = (int64_t)n;
    fwrite("KDS1", 1, 4, f); fwrite(hd, 4, 2, f); fwrite(&nb, 8, 1, f); fwrite(buf, 1, n, f);
    fwrite(M.ctok, 4, (size_t)pos, f);
    int r = ferror(f) ? -4 : 0;
    fclose(f); free(buf);
    return r;
}
/* returns the restored position, or < 0 */
int dsv_state_load(const char* path) {
    if (!GPU.full || !GPU.state_load) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) return -2;
    char mg[4]; int32_t hd[2]; int64_t nb;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "KDS1", 4) || fread(hd, 4, 2, f) != 2 || fread(&nb, 8, 1, f) != 1 ||
        hd[0] < 0 || hd[0] > M.cap || hd[1] != M.flags || (size_t)nb != GPU.state_size(hd[0])) { fclose(f); return -3; }
    void* buf = malloc((size_t)nb);
    int r = buf && fread(buf, 1, (size_t)nb, f) == (size_t)nb && fread(M.ctok, 4, (size_t)hd[0], f) == (size_t)hd[0] ? 0 : -4;
    fclose(f);
    if (!r && GPU.state_load(hd[0], buf)) r = -5;
    free(buf);
    if (r) return r;
    M.pos = hd[0];
    return hd[0];
}

/* ---------------------------------------------------------------- micro-benchmark */
/* all dense matrices of all layers once per iteration, T tokens of random activations; returns GB/s */
double dsv_bench_dense(int iters, int T) {
    int8_t* xq = zalloc((size_t)T * 32768); float* xd = zalloc((size_t)T * 32768 / QK * 4);
    float* xf = zalloc((size_t)T * HC * D * 4); float* y = zalloc((size_t)T * 32768 * 4);
    for (size_t i = 0; i < (size_t)T * 32768; i++) xq[i] = (int8_t)(i * 7 % 255 - 127);
    for (size_t i = 0; i < (size_t)T * 32768 / QK; i++) xd[i] = 0.01f;
    for (size_t i = 0; i < (size_t)T * HC * D; i++) xf[i] = 0.001f * (i % 13);
    double bytes = 0, t0 = 0;
    for (int it = -1; it < iters; it++) {
        if (it == 0) t0 = now();
        for (int l = 0; l < NL; l++) {
            Layer* L = &M.L[l];
            const Mat* qm[] = {&L->q_a, &L->q_b, &L->kv, &L->o_a, &L->o_b, &L->sh_gate, &L->sh_up, &L->sh_down,
                               &L->cmp_kv, &L->cmp_gate, &L->idx_qb, &L->idx_proj, &L->eng_kv};
            const Mat* fm[] = {&L->gate_inp, &L->hc_attn_fn, &L->hc_ffn_fn};
            for (int i = 0; i < 13; i++) {
                if (!qm[i]->rows) continue;
                int xs = qm[i]->seg_rows ? 32768 : qm[i]->cols;
                Op o = opq(qm[i], xq, xd, xs, y, qm[i]->rows);
                gemv_run(&o, 1, T);
                if (it == 0) bytes += (double)qm[i]->rows * qm[i]->rb;
            }
            for (int i = 0; i < 3; i++) {
                Op o = opf(fm[i], xf, fm[i]->cols, y, fm[i]->rows);
                gemv_run(&o, 1, T);
                if (it == 0) bytes += (double)fm[i]->rows * fm[i]->rb;
            }
        }
        Op o = opq(&M.out, xq, xd, D, y, VOCAB);
        if (T <= 1) { gemv_run(&o, 1, 1); if (it == 0) bytes += (double)M.out.rows * M.out.rb; }
    }
    double dt = (now() - t0) / iters;
    fprintf(stderr, "dsv41 bench: %.2f GB per pass, %.1f ms, %.1f GB/s (T=%d)\n", bytes / 1e9, dt * 1e3, bytes / dt / 1e9, T);
    free(xq); free(xd); free(xf); free(y);
    return bytes / dt / 1e9;
}

/* ---------------------------------------------------------------- model files */
/* map a model file read-only (shared page cache); *size receives its length */
void* dsv_map_file(const char* path, long* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    void* p = mmap(NULL, (size_t)n, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *size = n;
    return p;
}
/* unmap a file and drop its page cache (for shards whose tensors were all copied) */
int dsv_drop_file(void* p, long size, const char* path) {
    munmap(p, (size_t)size);
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    int r = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    return r;
}

/* time the attention of layer l for the last token (call after a forward); returns microseconds per call */
double dsv_bench_attn(int l, int iters) {
    Layer* L = &M.L[l];
    M.pos -= 1;
    AttnArg A = {L, L->ratio ? &M.L[kv_src_of(l)] : NULL, L->ratio ? &M.L[idx_src_of(l)] : NULL, l, 1};
    SplitArg S;
    S.A = &A; S.first[0] = 0;
    {
        const float* ent[WIN + ITOPK];
        M.anch[0] = (attn_entries(&A, 0, ent) + ACS - 1) / ACS;
        S.first[1] = M.anch[0] * (NH / AGH);
    }
    double t0 = 0, t1 = 0;
    for (int i = -1; i < iters; i++) {
        double a = now();
        if (M.flags & F_LLAMA_FA) kp_for(NH / AHG, attn_item, &A);
        else kp_for(S.first[1], attn_split_item, &S);
        double b = now();
        kp_for(OG, derope_quant_item, &A);
        if (i >= 0) { t0 += b - a; t1 += now() - b; }
    }
    M.pos += 1;
    fprintf(stderr, "dsv41: layer %d attention %.1f us, merge + rotation + quantization %.1f us\n", l, t0 / iters * 1e6,
            t1 / iters * 1e6);
    return (t0 + t1) / iters * 1e6;
}

/* Move page-cache pages of a model file from one NUMA node to another (no disk reads), up to max_bytes. The
   page cache holds the whole model; when it fills a node, the node-bound dense weights would push model pages
   out of the cache, so the warm-up step can make room this way. Returns the bytes moved. */
long dsv_move_file_pages(const char* path, int from, int to, long max_bytes) {
    long size = 0;
    uint8_t* p = dsv_map_file(path, &size);
    if (!p) return -1;
    const long pg = 4096, batch = 65536;
    void** pages = malloc(sizeof(void*) * batch);
    int *nodes = malloc(sizeof(int) * batch), *status = malloc(sizeof(int) * batch);
    long moved = 0;
    for (long off = 0; off < size && moved < max_bytes; off += batch * pg) {
        long n = (size - off) / pg < batch ? (size - off + pg - 1) / pg : batch;
        for (long i = 0; i < n; i++) pages[i] = p + off + i * pg;
        /* query: the node of each page (only pages already in the page cache are touched) */
        unsigned char vec[65536];
        if (mincore(p + off, (size_t)n * pg, vec) != 0) break;
        long k = 0;
        for (long i = 0; i < n; i++) if (vec[i] & 1) { volatile uint8_t x = ((uint8_t*)pages[i])[0]; (void)x; pages[k++] = pages[i]; }
        if (!k) continue;
        if (syscall(SYS_move_pages, 0, k, pages, NULL, status, 0) != 0) continue;
        long m = 0;
        for (long i = 0; i < k; i++) if (status[i] == from) { pages[m] = pages[i]; nodes[m] = to; m++; }
        if (m > (max_bytes - moved) / pg) m = (max_bytes - moved) / pg;
        if (m && syscall(SYS_move_pages, 0, m, pages, nodes, status, 2 /* MPOL_MF_MOVE */) == 0) moved += m * pg;
    }
    free(pages); free(nodes); free(status);
    munmap(p, (size_t)size);
    return moved;
}

/* Spread the cached pages of a model file over the two nodes: of every `period` pages, the last `n1` go to node 1
   and the others to node 0 (2 of 3 to node 0 matches node 0's share of the memory bandwidth). The experts are read
   in place by threads of both nodes, so both memory controllers then serve every layer. Returns the bytes moved. */
/* move the cached pages of byte ranges of a file to the given nodes (no disk reads); returns the bytes moved */
long dsv_place_ranges(const char* path, int n, const long* off, const long* len, const int* node) {
    long size = 0;
    uint8_t* p = dsv_map_file(path, &size);
    if (!p) return -1;
    const long pg = 4096, batch = 65536;
    void** pages = malloc(sizeof(void*) * batch);
    int *nodes = malloc(sizeof(int) * batch), *status = malloc(sizeof(int) * batch);
    long moved = 0, k = 0;
    unsigned char* vec = malloc(1 << 20);
    for (int r = 0; r < n; r++) {
        long a = off[r] / pg * pg, b = (off[r] + len[r] + pg - 1) / pg * pg;
        if (b > size) b = size;
        for (long c = a; c < b; c += (1L << 20) * pg) {
            long m = (b - c) / pg < (1L << 20) ? (b - c) / pg : (1L << 20);
            if (mincore(p + c, (size_t)m * pg, vec) != 0) break;
            for (long i = 0; i < m; i++) {
                if (!(vec[i] & 1)) continue;
                volatile uint8_t x = p[c + i * pg]; (void)x;
                pages[k] = p + c + i * pg; nodes[k] = node[r]; k++;
                if (k == batch) {
                    if (syscall(SYS_move_pages, 0, k, pages, nodes, status, 2) == 0)
                        for (long j = 0; j < k; j++) moved += status[j] >= 0 ? pg : 0;
                    k = 0;
                }
            }
        }
    }
    if (k && syscall(SYS_move_pages, 0, k, pages, nodes, status, 2) == 0)
        for (long j = 0; j < k; j++) moved += status[j] >= 0 ? pg : 0;
    free(vec); free(pages); free(nodes); free(status);
    munmap(p, (size_t)size);
    return moved;
}

long dsv_interleave_file_pages(const char* path, int period, int n1) {
    long size = 0;
    uint8_t* p = dsv_map_file(path, &size);
    if (!p) return -1;
    const long pg = 4096, batch = 65536;
    void** pages = malloc(sizeof(void*) * batch);
    int *nodes = malloc(sizeof(int) * batch), *status = malloc(sizeof(int) * batch);
    long moved = 0;
    for (long off = 0; off < size; off += batch * pg) {
        long n = (size - off) / pg < batch ? (size - off + pg - 1) / pg : batch;
        unsigned char vec[65536];
        if (mincore(p + off, (size_t)n * pg, vec) != 0) break;
        long k = 0;
        for (long i = 0; i < n; i++)
            if (vec[i] & 1) {
                volatile uint8_t x = p[off + i * pg]; (void)x;
                pages[k] = p + off + i * pg;
                nodes[k] = ((off / pg + i) % period) >= period - n1;
                k++;
            }
        if (k && syscall(SYS_move_pages, 0, k, pages, nodes, status, 2 /* MPOL_MF_MOVE */) == 0)
            for (long i = 0; i < k; i++) moved += status[i] >= 0 ? pg : 0;
    }
    free(pages); free(nodes); free(status);
    munmap(p, (size_t)size);
    return moved;
}
