/* kcpu.c - Kindi CPU backend (Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only): one token of a Ternary-Bonsai-2 (Qwen3.5 hybrid GDN/attention) model in C.
   The whole forward pass runs inside one OpenMP parallel region (barriers between the stages), so there is no
   Python or thread start-up cost per operation. Arithmetic plan as kindi/engine.py: fp64 norm sums, Hadamard folds
   with the pack's sign vectors, 81-entry fp32 table GEMV for every ternary matrix, fp16 K/V cache. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* The engine requires AVX.  Five tiers are buildable: AVX2/FMA/F16C, AVX2/F16C without FMA,
   AVX2/FMA without F16C, AVX2, and AVX-only.  kindi/cpu.py builds the tier the host supports
   and loads that library; kc_isa()/kc_isa_name() below report which tier was compiled, so a
   stale cache entry cannot run instructions the CPU lacks.  FMA is optional (separate
   multiply/add without it), and a build without F16C uses the software fp16 conversion.  Fail
   at compile time with an actionable message instead of a target-mismatch error. */
#if !defined(__AVX__)
#error "kindi CPU engine requires AVX; build with -mavx (docs/CPU-SUPPORT.md)"
#endif

#ifdef _WIN32
#define KC_API __declspec(dllexport)      /* the ctypes loader needs the names in the DLL export table */
#else
#define KC_API
#endif

typedef struct { const uint8_t* W; const float* S; int R, C; } Mat;   /* W [R/8][C/128][64] u32 pair codes, S [R/8][C/128][8] */

typedef struct {
  int is_attn;
  const float *attn_norm, *post_norm;
  /* GDN */
  Mat qkv, gate, ssm_out;
  const float *beta_w, *alpha_w, *dt_b, *ssm_a, *conv_w, *ssm_norm;
  float *M, *conv;                       /* state [NV][128][128], conv [3][CONV_CH] */
  /* attention */
  Mat q, k, v, o;
  const float *q_norm, *k_norm;
  uint16_t *Kc, *Vc;                     /* [ctx][NHKV][HD] fp16 */
  /* FFN */
  Mat ffg, ffu, ffd;
} Layer;

typedef struct {
  int D, NL, NFF, NH, NHKV, HD, NK, HK, NV, CONV_CH, HB, rope_half, ctx, vocab;
  Layer* L;
  Mat emb, head;
  const float *out_norm, *sD, *sV, *sA, *sF, *cos_t, *sin_t;
  /* scratch */
  float *h, *xn, *xp, *y, *qkv, *zv, *cs, *qk, *gout, *outn, *m, *qn, *kr, *vv, *qr, *kro, *sc, *att, *ffg, *ffu,
        *T, *logits, *gb, *gg;
  double* red;                           /* per-thread partial sums */
} Model;

/* ------------------------------------------------------------------------------------------- helpers */
/* stage timers (thread 0, between synchronised points): 0 embed 1 norm+fold 2 gemv 3 gdn-other 4 attn-other 5 ffn-other 6 head */
double kc_times[8];
static double t_last; static int t_cur;
#define TS(cat) do { if(omp_get_thread_num() == 0){ const double _t = omp_get_wtime(); kc_times[t_cur] += _t - t_last; t_last = _t; t_cur = (cat); } } while(0)
static inline void span(long long n, int* a, int* b){
  const int nt = omp_get_num_threads(), id = omp_get_thread_num();
  *a = (int)(n * id / nt); *b = (int)(n * (id + 1) / nt);
}
#if defined(__F16C__)
static inline float h2f(uint16_t h){ return _cvtsh_ss(h); }
static inline uint16_t f2h(float f){ return _cvtss_sh(f, 0); }
#else
/* IEEE binary16 conversions for AVX-only hosts.  f2h is round-to-nearest-even, matching
   _cvtss_sh(f, 0); h2f is exact for every finite, infinite and NaN input. */
static inline float h2f(uint16_t h){
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
  if(exp == 0){
    if(mant == 0) bits = sign;
    else {
      int e = -1;
      do { mant <<= 1; e++; } while(!(mant & 0x400u));
      mant &= 0x3ffu;
      bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if(exp == 31) bits = sign | 0x7f800000u | (mant << 13);
  else bits = sign | ((exp + 112u) << 23) | (mant << 13);
  union { uint32_t u; float f; } v; v.u = bits; return v.f;
}
static inline uint16_t f2h(float f){
  union { float f; uint32_t u; } v; v.f = f;
  const uint32_t sign = (v.u >> 16) & 0x8000u;
  const uint32_t abs = v.u & 0x7fffffffu;
  if(abs >= 0x7f800000u){
    if(abs > 0x7f800000u) return (uint16_t)(sign | 0x7e00u | ((abs >> 13) & 0x3ffu));
    return (uint16_t)(sign | 0x7c00u);
  }
  if(abs >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);   /* 65520 rounds to infinity */
  int exp = (int)(abs >> 23) - 127;
  const uint32_t mant = abs & 0x7fffffu;
  if(exp >= -14){
    uint32_t q = (mant | 0x800000u) >> 13;
    const uint32_t rem = (mant | 0x800000u) & 0x1fffu;
    if(rem > 0x1000u || (rem == 0x1000u && (q & 1u))) q++;
    if(q == 0x800u){ q = 0x400u; exp++; }
    const uint32_t e = (uint32_t)(exp + 15);
    if(e >= 31u) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (e << 10) | (q & 0x3ffu));
  }
  if(exp >= -25){
    const uint32_t sig = mant | 0x800000u;
    const int shift = -exp - 1;
    uint32_t q = sig >> shift;
    const uint32_t rem = sig & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if(rem > half || (rem == half && (q & 1u))) q++;
    return (uint16_t)(sign | q);
  }
  return (uint16_t)sign;
}
#endif
static inline float silu(float x){ return x / (1.0f + expf(-x)); }
static inline float sigm(float x){ return 1.0f / (1.0f + expf(-x)); }

static void fwht_block(float* x, int hb){
  for(int h = 1; h < hb; h <<= 1)
    for(int i = 0; i < hb; i += 2 * h)
      for(int j = i; j < i + h; ++j){ const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
  const float s = 1.0f / sqrtf((float)hb);
  for(int j = 0; j < hb; ++j) x[j] *= s;
}

/* out = fold(v) = FWHT_blocks(sg * v); in parallel over blocks (must be called by all threads; ends in a barrier) */
static void fold(const Model* m, const float* v, const float* sg, float* out, int n){
  #pragma omp for schedule(static)
  for(int b = 0; b < n / m->HB; ++b){
    float* o = out + (size_t)b * m->HB;
    for(int j = 0; j < m->HB; ++j) o[j] = sg[b * m->HB + j] * v[b * m->HB + j];
    fwht_block(o, m->HB);
  }
}

/* ------------------------------------------------------------------------------------------- ternary GEMV */
/* Weights as pair codes, rows in SIMD lanes: per (8-row block, 128-column group) 64 uint32; word p holds the 8 rows'
   4-bit codes for columns 2p, 2p+1 (0..7 = which of the 8 nonzero (w0, w1) pairs, 8 = both zero; 2 bits per weight).
   Per call, a table of the 8 nonzero pair sums for every column pair (m->T, C/2 x 8 floats); then one vpermps per
   8 rows x 2 columns. Per-group sums are scaled (the 8 rows' scales are contiguous) into a vector accumulator, so a
   row's result needs no horizontal sum. Every pair sum is formed once in fp32; no activation quantisation. */
KC_API void kc_init(void){}
static void pair_tables(const float* x, float* T, int p0, int p1){
  for(int p = p0; p < p1; ++p){
    const float x0 = x[2 * p], x1 = x[2 * p + 1];
    float* t = T + (size_t)p * 8;
    t[0] = -x0 - x1; t[1] = -x1; t[2] = x0 - x1; t[3] = -x0;
    t[4] = x0; t[5] = -x0 + x1; t[6] = x1; t[7] = x0 + x1;
  }
}
#if defined(__AVX2__)
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const __m256i sh = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
  const __m256i f = _mm256_set1_epi32(15);
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; ++rb){
    __m256 acc = _mm256_setzero_ps();
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = P + ((size_t)rb * G + g) * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      __m256 s0 = _mm256_setzero_ps(), s1 = s0;
      for(int p = 0; p < 64; p += 2){
        const __m256i c0 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p]), sh), f);
        const __m256i c1 = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)pw[p + 1]), sh), f);
        const __m256 v0 = _mm256_permutevar8x32_ps(_mm256_load_ps(pt + 8 * p), c0);
        const __m256 v1 = _mm256_permutevar8x32_ps(_mm256_load_ps(pt + 8 * p + 8), c1);
        const __m256 z0 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c0, 28), 31));   /* code 8 */
        const __m256 z1 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_slli_epi32(c1, 28), 31));
        s0 = _mm256_add_ps(s0, _mm256_andnot_ps(z0, v0));
        s1 = _mm256_add_ps(s1, _mm256_andnot_ps(z1, v1));
      }
#if defined(__FMA__)
      acc = _mm256_fmadd_ps(_mm256_add_ps(s0, s1), _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8), acc);
#else
      /* AVX2/F16C hosts without FMA: keep the AVX2 table/permute path and round the product
         before the add.  This is not bit-identical to FMA, but it is the documented fallback. */
      acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_add_ps(s0, s1),
                                             _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8)));
#endif
    }
    _mm256_storeu_ps(y + rb * 8, acc);
  }
}
#else
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const __m256 vfour = _mm256_set1_ps(4.0f), veight = _mm256_set1_ps(8.0f);
  for(int rb = b0; rb < b1; ++rb){
    __m256 acc = _mm256_setzero_ps();
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = (const uint32_t*)A->W + ((size_t)rb * G + g) * 64;
      const float* pt = T + (size_t)g * 64 * 8;
      __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
      for(int p = 0; p < 64; p += 2){
        const uint32_t w0 = pw[p], w1 = pw[p + 1];
        float c0[8], c1[8];
        int i0[8], i1[8];
        for(int l = 0; l < 8; ++l){
          const int a = (int)((w0 >> (4 * l)) & 15u);
          const int b = (int)((w1 >> (4 * l)) & 15u);
          c0[l] = (float)a; i0[l] = a & 3;
          c1[l] = (float)b; i1[l] = b & 3;
        }
        const __m256 f0 = _mm256_loadu_ps(c0), f1 = _mm256_loadu_ps(c1);
        const __m256i x0 = _mm256_loadu_si256((const __m256i*)i0);
        const __m256i x1 = _mm256_loadu_si256((const __m256i*)i1);
        const __m256 t0l = _mm256_broadcast_ps((const __m128*)(pt + 8 * p));
        const __m256 t0h = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 4));
        const __m256 t1l = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 8));
        const __m256 t1h = _mm256_broadcast_ps((const __m128*)(pt + 8 * p + 12));
        __m256 v0 = _mm256_blendv_ps(_mm256_permutevar_ps(t0l, x0),
                                     _mm256_permutevar_ps(t0h, x0),
                                     _mm256_cmp_ps(f0, vfour, _CMP_GE_OQ));
        __m256 v1 = _mm256_blendv_ps(_mm256_permutevar_ps(t1l, x1),
                                     _mm256_permutevar_ps(t1h, x1),
                                     _mm256_cmp_ps(f1, vfour, _CMP_GE_OQ));
        v0 = _mm256_andnot_ps(_mm256_cmp_ps(f0, veight, _CMP_EQ_OQ), v0);
        v1 = _mm256_andnot_ps(_mm256_cmp_ps(f1, veight, _CMP_EQ_OQ), v1);
        s0 = _mm256_add_ps(s0, v0);
        s1 = _mm256_add_ps(s1, v1);
      }
      acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_add_ps(s0, s1),
                                             _mm256_loadu_ps(A->S + ((size_t)rb * G + g) * 8)));
    }
    _mm256_storeu_ps(y + rb * 8, acc);
  }
}
#endif

/* inside the parallel region: up to 3 matrices with the same input; the threads build the pair tables, then split
   the concatenated rows in 8-row blocks; ends in a barrier */
static void gemv(Model* m, int n, const Mat* const* A, float* const* y, const float* x){
  const int prev = t_cur; TS(2);
  int a, b;
  span(A[0]->C / 2, &a, &b);
  pair_tables(x, m->T, a, b);
  #pragma omp barrier
  long long Ut = 0;
  for(int i = 0; i < n; ++i) Ut += A[i]->R / 8;
  int a0, a1;
  span(Ut, &a0, &a1);
  int base = 0;
  for(int i = 0; i < n; ++i){
    const int U = A[i]->R / 8;
    const int u0 = a0 > base ? a0 - base : 0, u1 = (a1 - base) < U ? (a1 - base) : U;
    base += U;
    if(u1 > u0) rows8(A[i], y[i], m->T, u0, u1);
  }
  #pragma omp barrier
  TS(prev);
}
static void gemv1(Model* m, const Mat* A, float* y, const float* x){ const Mat* a[1] = {A}; float* yy[1] = {y}; gemv(m, 1, a, yy, x); }

/* ------------------------------------------------------------------------------------------- norms */
/* residual add (if add) + RMS norm + fold of m->h into xn / xp; all threads; ends in a barrier */
static void rmsfold(Model* m, const float* w, int add){
  const int D = m->D;
  const int prev = t_cur; TS(1);
  #pragma omp single
  {
    double acc = 0.0;
    for(int e = 0; e < D; ++e){
      const float xf = add ? m->h[e] + m->y[e] : m->h[e];
      m->h[e] = xf;
      acc += (double)xf * (double)xf;
    }
    const float s = 1.0f / sqrtf((float)(acc / D) + 1e-6f);
    for(int e = 0; e < D; ++e) m->xn[e] = (m->h[e] * s) * w[e];
  }
  fold(m, m->xn, m->sD, m->xp, D);
  TS(prev);
}

static void rms_rows(const float* x, const float* w, float* out, int rows, int n){
  for(int r = 0; r < rows; ++r){
    const float* xr = x + (size_t)r * n;
    double acc = 0.0;
    for(int e = 0; e < n; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
    for(int e = 0; e < n; ++e) out[(size_t)r * n + e] = (xr[e] * sc) * w[e];
  }
}

/* ------------------------------------------------------------------------------------------- blocks */
static void gdn(Model* m, Layer* L, int add){
  TS(3);
  const int D = m->D, NV = m->NV, NK = m->NK, HK = m->HK, CC = m->CONV_CH;
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[2] = {&L->qkv, &L->gate}; float* yy[2] = {m->qkv, m->zv}; gemv(m, 2, a, yy, m->xp); }
  /* gates (per v head) and the causal conv (per channel) */
  #pragma omp for schedule(static) nowait
  for(int h = 0; h < NV; ++h){
    float ab = 0.f, aa = 0.f;
    for(int i = 0; i < D; ++i){ ab += L->beta_w[(size_t)h * D + i] * m->xn[i]; aa += L->alpha_w[(size_t)h * D + i] * m->xn[i]; }
    m->gb[h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    m->gg[h] = sp * L->ssm_a[h];
  }
  #pragma omp for schedule(static)
  for(int c = 0; c < CC; ++c){
    float* cv = L->conv;
    const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = m->qkv[c];
    const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
    cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
    m->cs[c] = silu(o);
  }
  /* L2 norm of the q and k heads */
  #pragma omp for schedule(static)
  for(int r = 0; r < 2 * NK; ++r){
    const float* xr = m->cs + (size_t)r * HK;
    double acc = 0.0;
    for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    for(int e = 0; e < HK; ++e) m->qk[(size_t)r * HK + e] = xr[e] * sc;
  }
  /* delta rule per v head, then the per-head RMS norm */
  const float scale = 1.0f / sqrtf((float)HK);
  #pragma omp for schedule(static)
  for(int h = 0; h < NV; ++h){
    const int kh = h % NK;
    const float* ks = m->qk + (size_t)(NK + kh) * HK;
    const float* qs = m->qk + (size_t)kh * HK;
    const float* v = m->cs + 2 * NK * HK + (size_t)h * HK;
    const float e = expf(m->gg[h]);
    float* Mh = L->M + (size_t)h * HK * HK;
    float sk[128], o[128], dd[128];
    for(int j = 0; j < HK; ++j) sk[j] = 0.f;
    for(int i = 0; i < HK; ++i){ const float ki = ks[i]; const float* mr = Mh + i * HK; for(int j = 0; j < HK; ++j) sk[j] += (mr[j] * e) * ki; }
    for(int j = 0; j < HK; ++j){ dd[j] = (v[j] - sk[j]) * m->gb[h]; o[j] = 0.f; }
    for(int i = 0; i < HK; ++i){
      const float ki = ks[i], qi = qs[i];
      float* mr = Mh + i * HK;
      for(int j = 0; j < HK; ++j){ const float mm = mr[j] * e + ki * dd[j]; mr[j] = mm; o[j] += mm * qi; }
    }
    for(int j = 0; j < HK; ++j) m->gout[(size_t)h * HK + j] = o[j] * scale;
    rms_rows(m->gout + (size_t)h * HK, L->ssm_norm, m->outn + (size_t)h * HK, 1, HK);
  }
  /* gated output in the grouped v-head order, then the fold */
  const int rep = NV / NK;
  #pragma omp for schedule(static)
  for(int e = 0; e < NV * HK; ++e){
    int src = e;
    if(rep > 1){ const int hd = e % HK, grp = e / HK; src = hd + HK * (grp / rep + NK * (grp % rep)); }
    m->m[e] = silu(m->zv[src]) * m->outn[src];
  }
  fold(m, m->m, m->sV, m->xp, NV * HK);
  gemv1(m, &L->ssm_out, m->y, m->xp);
}

static void attn(Model* m, Layer* L, int pos, int add){
  TS(4);
  const int NH = m->NH, NKV = m->NHKV, HD = m->HD, rh = m->rope_half, G = NH / NKV, n = pos + 1;
  rmsfold(m, L->attn_norm, add);
  { const Mat* a[3] = {&L->q, &L->k, &L->v}; float* yy[3] = {m->qn, m->kr, m->vv}; gemv(m, 3, a, yy, m->xp); }
  const float* cs = m->cos_t + (size_t)pos * rh;
  const float* sn = m->sin_t + (size_t)pos * rh;
  #pragma omp for schedule(static)
  for(int r = 0; r < NH + NKV; ++r){
    float tmp[512];
    float* dst;
    if(r < NH){ rms_rows(m->qn + (size_t)r * 2 * HD, L->q_norm, tmp, 1, HD); dst = m->qr + (size_t)r * HD; }
    else { rms_rows(m->kr + (size_t)(r - NH) * HD, L->k_norm, tmp, 1, HD); dst = m->kro + (size_t)(r - NH) * HD; }
    for(int j = 0; j < rh; ++j){
      const float a = tmp[j], b = tmp[j + rh];
      dst[j] = a * cs[j] - b * sn[j];
      dst[j + rh] = a * sn[j] + b * cs[j];
    }
    for(int j = 2 * rh; j < HD; ++j) dst[j] = tmp[j];
    if(r >= NH){
      const int kv = r - NH;
      uint16_t* K = L->Kc + ((size_t)pos * NKV + kv) * HD;
      uint16_t* V = L->Vc + ((size_t)pos * NKV + kv) * HD;
      for(int j = 0; j < HD; ++j){ K[j] = f2h(dst[j]); V[j] = f2h(m->vv[(size_t)kv * HD + j]); }
    }
  }
  /* scores sc[head][t] */
  #pragma omp for schedule(static)
  for(int t = 0; t < n; ++t){
    for(int kv = 0; kv < NKV; ++kv){
      const uint16_t* K = L->Kc + ((size_t)t * NKV + kv) * HD;
      float kf[512];
      for(int j = 0; j < HD; ++j) kf[j] = h2f(K[j]);
      for(int g = 0; g < G; ++g){
        const float* q = m->qr + (size_t)(kv * G + g) * HD;
        float a = 0.f;
        for(int j = 0; j < HD; ++j) a += kf[j] * q[j];
        m->sc[(size_t)(kv * G + g) * m->ctx + t] = a * 0.0625f;
      }
    }
  }
  /* softmax per head */
  #pragma omp for schedule(static)
  for(int hh = 0; hh < NH; ++hh){
    float* s = m->sc + (size_t)hh * m->ctx;
    float mx = -1e30f;
    for(int t = 0; t < n; ++t) mx = fmaxf(mx, s[t]);
    float tot = 0.f;
    for(int t = 0; t < n; ++t) tot += expf(s[t] - mx);
    for(int t = 0; t < n; ++t) s[t] = expf(s[t] - mx) / tot;
  }
  /* weighted V, gated; parallel over (head, 32-dim slice) */
  #pragma omp for schedule(static)
  for(int u = 0; u < NH * (HD / 32); ++u){
    const int hh = u / (HD / 32), d0 = (u % (HD / 32)) * 32, kv = hh / G;
    const float* p = m->sc + (size_t)hh * m->ctx;
    float acc[32];
    for(int j = 0; j < 32; ++j) acc[j] = 0.f;
    for(int t = 0; t < n; ++t){
      const uint16_t* V = L->Vc + ((size_t)t * NKV + kv) * HD + d0;
      const float pt = p[t];
      for(int j = 0; j < 32; ++j) acc[j] += pt * h2f(V[j]);
    }
    for(int j = 0; j < 32; ++j) m->att[(size_t)hh * HD + d0 + j] = acc[j] * sigm(m->qn[(size_t)hh * 2 * HD + HD + d0 + j]);
  }
  fold(m, m->att, m->sA, m->xp, NH * HD);
  gemv1(m, &L->o, m->y, m->xp);
}

static void ffn_input(Model* m, Layer* L){
  TS(5);
  rmsfold(m, L->post_norm, 1);
  { const Mat* a[2] = {&L->ffg, &L->ffu}; float* yy[2] = {m->ffg, m->ffu}; gemv(m, 2, a, yy, m->xp); }
  #pragma omp for schedule(static)
  for(int e = 0; e < m->NFF; ++e) m->m[e] = silu(m->ffg[e]) * m->ffu[e];
  fold(m, m->m, m->sF, m->xp, m->NFF);
}

static void ffn_down(Model* m, Layer* L){
  gemv1(m, &L->ffd, m->y, m->xp);
}

static void ffn(Model* m, Layer* L){
  ffn_input(m, L);
  ffn_down(m, L);
}

static void layer_body(Model* m, int il, int pos, int add){
  Layer* L = m->L + il;
  if(L->is_attn) attn(m, L, pos, add); else gdn(m, L, add);
  ffn(m, L);
}

static void head_body(Model* m, int add, int* best){
  TS(6);
  rmsfold(m, m->out_norm, add);
  gemv1(m, &m->head, m->logits, m->xp);
  #pragma omp single
  {
    int bi = 0; float bv = m->logits[0];
    for(int i = 1; i < m->head.R; ++i) if(m->logits[i] > bv){ bv = m->logits[i]; bi = i; }
    *best = bi;
  }
}

static void embed(Model* m, int tok){
  const int C = m->emb.C, R = m->emb.R;
  #pragma omp for schedule(static)
  for(int g = 0; g < C / 128; ++g){
    static const signed char W0[9] = {-1, 0, 1, -1, 1, -1, 0, 1, 0}, W1[9] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
    const uint32_t* pw = (const uint32_t*)m->emb.W + ((size_t)(tok / 8) * (C / 128) + g) * 64;
    const float d = m->emb.S[((size_t)(tok / 8) * (C / 128) + g) * 8 + (tok & 7)];
    for(int p = 0; p < 64; ++p){
      const int code = (pw[p] >> (4 * (tok & 7))) & 15;
      m->m[g * 128 + 2 * p] = (float)W0[code] * d;
      m->m[g * 128 + 2 * p + 1] = (float)W1[code] * d;
    }
  }
  #pragma omp for schedule(static)
  for(int bl = 0; bl < m->D / m->HB; ++bl){
    float* o = m->m + (size_t)bl * m->HB;
    fwht_block(o, m->HB);
    for(int j = 0; j < m->HB; ++j) m->h[bl * m->HB + j] = m->sD[bl * m->HB + j] * o[j];
  }
}

/* one token; head != 0: also the output head (logits in m->logits); returns the greedy token or -1 */
KC_API int kc_step(Model* m, int tok, int pos, int head){
  int best = -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    embed(m, tok);
    for(int il = 0; il < m->NL; ++il) layer_body(m, il, pos, il > 0);
    if(head) head_body(m, 1, &best);
    else {
      #pragma omp single
      { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }   /* keep h = the residual after the last block */
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return best;
}

/* Per-layer entry point for a CPU/GPU layer split (tandem).  kc_embed() puts the token vector in
   m->h; each kc_layer() call treats m->h as the complete residual entering il and leaves the
   complete residual leaving il in m->h.  The first block's residual is added to h by ffn(), and
   the FFN output is added explicitly at the end, so m->y is never carried between calls.  The
   whole-model kc_step() above stays a single OpenMP region; these calls open one region per part
   and are for handing a layer to or from another backend. */
KC_API void kc_embed(Model* m, int tok){
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  { embed(m, tok); }
  kc_times[t_cur] += omp_get_wtime() - t_last;
}

KC_API int kc_layer(Model* m, int il, int pos){
  if(il < 0 || il >= m->NL || pos < 0 || pos >= m->ctx) return -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->y[e] = 0.0f; }
    #pragma omp barrier
    layer_body(m, il, pos, 0);
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

/* The same layer without its FFN-down GEMV.  m->xp is left holding the folded FFN input;
   m->y is zero so a caller can fill the down rows from one or more backends. */
KC_API int kc_layer_ffn_input(Model* m, int il, int pos){
  if(il < 0 || il >= m->NL || pos < 0 || pos >= m->ctx) return -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  {
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->y[e] = 0.0f; }
    #pragma omp barrier
    Layer* L = m->L + il;
    if(L->is_attn) attn(m, L, pos, 0); else gdn(m, L, 0);
    ffn_input(m, L);
    #pragma omp single
    { for(int e = 0; e < m->D; ++e) m->y[e] = 0.0f; }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

/* One ternary matrix over an explicit row range, for a CPU/GPU row split.  The caller supplies
   the packed rows and the complete input vector; this entry point does not touch the layer
   schedule or the residual stream. */
KC_API int kc_gemv(Model* m, const Mat* A, const float* x, float* y){
  if(!m || !A || !x || !y) return -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  { gemv1(m, A, y, x); }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return 0;
}

KC_API int kc_head(Model* m){
  int best = -1;
  t_last = omp_get_wtime(); t_cur = 0;
  #pragma omp parallel
  { head_body(m, 0, &best); }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return best;
}

/* The compiled tier. The numbers and names match kindi/cpu.py:ISA_ORDER; the loader compares
   kc_isa_name() with the tier it selected before the first model call. */
KC_API int kc_isa(void){
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  return 4;
#elif defined(__AVX2__) && defined(__F16C__)
  return 3;
#elif defined(__AVX2__) && defined(__FMA__)
  return 2;
#elif defined(__AVX2__)
  return 1;
#else
  return 0;
#endif
}

KC_API const char* kc_isa_name(void){
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
  return "avx2-fma-f16c";
#elif defined(__AVX2__) && defined(__F16C__)
  return "avx2-f16c-no-fma";
#elif defined(__AVX2__) && defined(__FMA__)
  return "avx2-fma";
#elif defined(__AVX2__)
  return "avx2";
#else
  return "avx-only";
#endif
}

KC_API int kc_sizeof_model(void){ return (int)sizeof(Model); }
KC_API int kc_sizeof_layer(void){ return (int)sizeof(Layer); }
