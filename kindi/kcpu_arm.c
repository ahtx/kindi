/* kcpu_arm.c - Kindi CPU backend for ARM64/Android (alternative to kcpu.c)
   Port of kindi/kcpu.c (v0.4.0, identical across 0.3.0/0.4.0/0.5.0) from AVX2 to portable C + ARM NEON headers.
   The scalar kernels below are bit-exact with kcpu.c; NEON SIMD acceleration of rows8() is a future step.
   SPDX-License-Identifier: AGPL-3.0-only

   Compile (Termux, native):
     gcc -O3 -march=armv8.2-a+simd -fopenmp -shared -fPIC -o libkcpu_arm.so kcpu_arm.c -lm
   Compile (NDK cross):
     aarch64-linux-android33-clang -O3 -march=armv8.2-a -fopenmp -shared -fPIC -o libkcpu_arm.so kcpu_arm.c -lm
*/

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_thread_num(void){ return 0; }
static int omp_get_num_threads(void){ return 1; }
static double omp_get_wtime(void){ return 0.0; }
#endif

/* NEON headers for ARM SIMD (included for future SIMD optimization of rows8) */
#include <arm_neon.h>

/* AArch64 FP16 <-> FP32 conversion via __fp16 type (IEEE half-precision on ARMv8.2-A) */
static inline float h2f(uint16_t h){
  __fp16 hf;
  memcpy(&hf, &h, sizeof(hf));
  return (float)hf;
}
static inline uint16_t f2h(float f){
  __fp16 hf = (__fp16)f;
  uint16_t h;
  memcpy(&h, &hf, sizeof(h));
  return h;
}

static inline float silu(float x){ return x / (1.0f + expf(-x)); }
static inline float sigm(float x){ return 1.0f / (1.0f + expf(-x)); }

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
  int16_t *s01;                          /* v3.1 streamed-lookup per-block partial sums */
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

static void fwht_block(float* x, int hb){
  for(int h = 1; h < hb; h <<= 1)
    for(int i = 0; i < hb; i += 2 * h)
      for(int j = i; j < i + h; ++j){ const float a = x[j], b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
  const float s = 1.0f / sqrtf((float)hb);
  for(int j = 0; j < hb; ++j) x[j] *= s;
}

/* out = fold(v) = FWHT_blocks(sg * v); in parallel over blocks (must be called by all threads; ends in a barrier) */
static void fold(const Model* m, const float* v, const float* sg, float* out, int n){
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int b = 0; b < n / m->HB; ++b){
    float* o = out + (size_t)b * m->HB;
    for(int j = 0; j < m->HB; ++j) o[j] = sg[b * m->HB + j] * v[b * m->HB + j];
    fwht_block(o, m->HB);
  }
}

/* ------------------------------------------------------------------------------------------- ternary GEMV */
/* Weights as pair codes, rows in SIMD lanes: per (8-row block, 128-column group) 64 uint32; word p holds the 8 rows'
   4-bit codes for columns 2p, 2p+1 (0..7 = which of the 8 nonzero (w0, w1) pairs, 8 = both zero; 2 bits per weight).
   Per call, a table of the 8 nonzero pair sums for every column pair (m->T, C/2 x 16 floats, stride 16 = 64B so a
   whole table fits one vqtbl4q / vld4q); then one gather per 8 rows x 2 columns. Per-group sums are scaled (the 8
   rows' scales are contiguous) into a vector accumulator, so a row's result needs no horizontal sum. Every pair sum
   is formed once in fp32; no activation quantisation. */
void kc_init(void){}
/* v1 (2026-11-17): 9-entry pair tables, t[8]=0 — code 8 (zero pair) adds +0.0, exactly what
   the x86 AVX2 kernel's branchless mask contributes. Deletes ~1.9G data-dependent branches
   per token. Proven bit-exact vs the original scalar kernel: 120 cases (real 2B shapes incl.
   248320x2048 head, all-zero weights, -0.0 adversarial) on x86 AND aarch64/qemu. */
static void pair_tables(const float* x, float* T, int p0, int p1){
  for(int p = p0; p < p1; ++p){
    const float x0 = x[2 * p], x1 = x[2 * p + 1];
    float* t = T + (size_t)p * 16;
    t[0] = -x0 - x1; t[1] = -x1; t[2] = x0 - x1; t[3] = -x0;
    t[4] = x0; t[5] = -x0 + x1; t[6] = x1; t[7] = x0 + x1;
    t[8] = 0.0f;
    for(int k = 9; k < 16; ++k) t[k] = 0.0f;   /* never indexed by valid codes; keeps loads defined */
  }
}

#ifdef KINDI_NEON_ROWS8
/* v2 (2026-11-17): NEON byte-plane LUT gather. Same loop structure and per-row accumulation
   order as the scalar kernel (s0/s1 chains over p, then (s0+s1)*scale per group) -> bit-exact.
   Per word: dup->nibble split->zip gives all 8 codes in 4 instrs; vld4q deinterleaves the 64B
   table into 4 byte-plane subtables; 4x vqtbl1q looks up 8 rows at once per plane; 3 zips
   recombine into two float32x4 (rows 0-3 / 4-7). No branches, no scalar random loads.
   Technique lineage: T-MAC (g=4 int8 LUT in one NEON reg; fp16 split into byte planes),
   bitnet.cpp TL1/TL2, Vec-LUT (vlut.cpp). Here kept fp32 for bit-exactness. */
static inline void lut8(uint32_t w, const uint8_t* tab, float32x4_t* lo, float32x4_t* hi){
  const uint8x16_t b = vreinterpretq_u8_u32(vdupq_n_u32(w));
  const uint8x16_t codes = vzipq_u8(vandq_u8(b, vdupq_n_u8(15)), vshrq_n_u8(b, 4)).val[0];
  const uint8x16x4_t tb = vld4q_u8(tab);
  const uint8x16_t r0 = vqtbl1q_u8(tb.val[0], codes);
  const uint8x16_t r1 = vqtbl1q_u8(tb.val[1], codes);
  const uint8x16_t r2 = vqtbl1q_u8(tb.val[2], codes);
  const uint8x16_t r3 = vqtbl1q_u8(tb.val[3], codes);
  const uint8x16_t z01 = vzipq_u8(r0, r1).val[0];
  const uint8x16_t z23 = vzipq_u8(r2, r3).val[0];
  /* recombine at 16-bit granularity: {b0,b1}+{b2,b3} -> {b0,b1,b2,b3} per row */
  const uint16x8x2_t z = vzipq_u16(vreinterpretq_u16_u8(z01), vreinterpretq_u16_u8(z23));
  *lo = vreinterpretq_f32_u8(vreinterpretq_u8_u16(z.val[0]));
  *hi = vreinterpretq_f32_u8(vreinterpretq_u8_u16(z.val[1]));
}
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; ++rb){
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = P + ((size_t)rb * G + g) * 64;
      const float* pt = T + (size_t)g * 64 * 16;
      float32x4_t s0lo = vdupq_n_f32(0.f), s0hi = vdupq_n_f32(0.f);
      float32x4_t s1lo = vdupq_n_f32(0.f), s1hi = vdupq_n_f32(0.f);
      for(int p = 0; p < 64; p += 2){
        float32x4_t lo, hi;
        lut8(pw[p], (const uint8_t*)(pt + (size_t)16 * p), &lo, &hi);
        s0lo = vaddq_f32(s0lo, lo); s0hi = vaddq_f32(s0hi, hi);
        lut8(pw[p + 1], (const uint8_t*)(pt + (size_t)16 * p + 16), &lo, &hi);
        s1lo = vaddq_f32(s1lo, lo); s1hi = vaddq_f32(s1hi, hi);
      }
      const float* sg = A->S + ((size_t)rb * G + g) * 8;
      const float32x4_t scl = vld1q_f32(sg), sch = vld1q_f32(sg + 4);
      a0 = vaddq_f32(a0, vmulq_f32(vaddq_f32(s0lo, s1lo), scl));
      a1 = vaddq_f32(a1, vmulq_f32(vaddq_f32(s0hi, s1hi), sch));
    }
    vst1q_f32(y + (size_t)rb * 8, a0);
    vst1q_f32(y + (size_t)rb * 8 + 4, a1);
  }
}
#else
/* Scalar, bit-exact port of the AVX2 rows8 kernel.
   For each 8-row block and each 128-column group: extract the 8 nibbles (codes) from each pair-code word,
   look up the corresponding pair sum from the per-pair table T, accumulate per row, then scale by the group's
   8 row-scales. Code 8 = zero pair -> t[8]=0 (branchless, matches the x86 mask semantics). */
static void rows8(const Mat* A, float* y, const float* T, int b0, int b1){
  const int G = A->C / 128;
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; ++rb){
    float acc[8];
    for(int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = P + ((size_t)rb * G + g) * 64;
      const float* pt = T + (size_t)g * 64 * 16;
      float s0[8], s1[8];
      for(int i = 0; i < 8; ++i){ s0[i] = 0.0f; s1[i] = 0.0f; }
      for(int p = 0; p < 64; p += 2){
        uint32_t w0 = pw[p];
        uint32_t w1 = (p + 1 < 64) ? pw[p + 1] : 0u;
        const float* t0 = pt + (size_t)16 * p;
        const float* t1 = pt + (size_t)16 * p + 16;
        for(int i = 0; i < 8; ++i){
          const uint32_t c0 = (w0 >> (4u * (uint32_t)i)) & 15u;
          const uint32_t c1 = (w1 >> (4u * (uint32_t)i)) & 15u;
          s0[i] += t0[c0];
          s1[i] += t1[c1];
        }
      }
      const float* sg = A->S + ((size_t)rb * G + g) * 8;
      for(int i = 0; i < 8; ++i){
        acc[i] += (s0[i] + s1[i]) * sg[i];
      }
    }
    for(int i = 0; i < 8; ++i) y[(size_t)rb * 8 + i] = acc[i];
  }
}
#endif

#if defined(KINDI_I16_ROWS8) || defined(KINDI_I16_STREAM)
/* v3 (2026-11-17): int8-quantised activations + int16 pair-sum LUT + NEON vqtbl2q gather.
   Technique lineage: T-MAC (LUT mpGEMM, byte-split wide entries), bitnet.cpp TL1/TL2
   (per_tensor_quant 127/max, int16 accumulate, widen at block end), Vec-LUT.
   NOT bit-exact with the fp32 x86 reference (activation quantisation rounds to int8);
   correctness gate = token-prefix agreement on reference.json + textjobs label agreement,
   same contract class as the gemma4 engine (GEMMA4_MIN_PREFIX). */
static void pair_tables_i16(const float* x, int16_t* T, int p0, int p1, const float* iscale){
  for(int p = p0; p < p1; ++p){
    const float s = iscale[p / 64];
    const int q0 = (int)lrintf(x[2 * p] * s), q1 = (int)lrintf(x[2 * p + 1] * s);
    int16_t* t = T + (size_t)p * 16;
    t[0] = (int16_t)(-q0 - q1); t[1] = (int16_t)(-q1); t[2] = (int16_t)(q0 - q1); t[3] = (int16_t)(-q0);
    t[4] = (int16_t)q0; t[5] = (int16_t)(-q0 + q1); t[6] = (int16_t)q1; t[7] = (int16_t)(q0 + q1);
    t[8] = 0;
    for(int k = 9; k < 16; ++k) t[k] = 0;
  }
}
static inline int16x8_t lut8_i16_tab(uint32_t w, uint8x16x2_t tb){
  const uint8x16_t b = vreinterpretq_u8_u32(vdupq_n_u32(w));
  const uint8x16_t codes = vzipq_u8(vandq_u8(b, vdupq_n_u8(15)), vshrq_n_u8(b, 4)).val[0];
  const uint8x16_t c2 = vshlq_n_u8(codes, 1);
  const uint8x16_t idx = vaddq_u8(vzipq_u8(c2, c2).val[0],
      (uint8x16_t){0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1});
  return vreinterpretq_s16_u8(vqtbl2q_u8(tb, idx));
}
static inline int16x8_t lut8_i16(uint32_t w, const int16_t* tab){
  return lut8_i16_tab(w, vld1q_u8_x2((const uint8_t*)tab));
}
#ifdef KINDI_I16_ROWS8
static void rows8_i16(const Mat* A, float* y, const int16_t* T, int b0, int b1, const float* ascale){
  const int G = A->C / 128;
  const uint32_t* P = (const uint32_t*)A->W;
  for(int rb = b0; rb < b1; ++rb){
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    for(int g = 0; g < G; ++g){
      const uint32_t* pw = P + ((size_t)rb * G + g) * 64;
      const int16_t* pt = T + (size_t)g * 64 * 16;
      int16x8_t s0 = vdupq_n_s16(0), s1 = vdupq_n_s16(0);
      for(int p = 0; p < 64; p += 2){
        s0 = vaddq_s16(s0, lut8_i16(pw[p], pt + (size_t)16 * p));
        s1 = vaddq_s16(s1, lut8_i16(pw[p + 1], pt + (size_t)16 * p + 16));
      }
      /* s0/s1 <= 32*254 = 8128 each; sum <= 16256: int16-safe. Widen exact, then scale. */
      const int16x8_t ss = vaddq_s16(s0, s1);
      const float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(ss)));
      const float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(ss)));
      const float* sg = A->S + ((size_t)rb * G + g) * 8;
      const float32x4_t k = vdupq_n_f32(ascale[g]);
      a0 = vaddq_f32(a0, vmulq_f32(vmulq_f32(f0, vld1q_f32(sg)), k));
      a1 = vaddq_f32(a1, vmulq_f32(vmulq_f32(f1, vld1q_f32(sg + 4)), k));
    }
    vst1q_f32(y + (size_t)rb * 8, a0);
    vst1q_f32(y + (size_t)rb * 8 + 4, a1);
  }
}
#endif
#ifdef KINDI_I16_STREAM
/* v3.1: words-outer streamed lookup (Vec-LUT cache insight). Each pair-table is loaded ONCE
   per group and reused across ALL row-blocks, instead of re-read per block. Per-block int16
   partial sums live in m->s01 scratch, indexed by GLOBAL block id (gbase + b0) so the matrices
    of one gemv call occupy DISJOINT s01 regions (prior bug: local b0 -> concurrent-matrix race).
   int16 adds are exact; per-row word order and per-group fp32 scale order match v3 -> the
   two variants are bit-identical to each other (proven by oracle). */
static void rows8_i16_stream(const Mat* A, float* y, const int16_t* T, int b0, int b1,
                             const float* ascale, int16_t* s01, int gbase){
  const int G = A->C / 128;
  const uint32_t* P = (const uint32_t*)A->W;
  const int nb = b1 - b0;
  int16_t* s0 = s01 + (size_t)(gbase + b0) * 16;   /* GLOBAL s01 index: disjoint across matrices */
  int16_t* s1 = s0 + (size_t)nb * 8;
  float* yb = y + (size_t)b0 * 8;
  for(int i = 0; i < nb * 8; ++i) yb[i] = 0.f;
  for(int g = 0; g < G; ++g){
    memset(s0, 0, (size_t)nb * 16 * sizeof(int16_t));
    const int16_t* pt = T + (size_t)g * 64 * 16;
    const uint32_t* pg = P + ((size_t)b0 * G + g) * 64;
    for(int p = 0; p < 64; p += 2){
      const uint8x16x2_t tb0 = vld1q_u8_x2((const uint8_t*)(pt + (size_t)16 * p));
      const uint8x16x2_t tb1 = vld1q_u8_x2((const uint8_t*)(pt + (size_t)16 * p + 16));
      for(int rb = 0; rb < nb; ++rb){
        const uint32_t* pw = pg + (size_t)rb * G * 64;
        const int16x8_t v0 = lut8_i16_tab(pw[p], tb0);
        const int16x8_t v1 = lut8_i16_tab(pw[p + 1], tb1);
        vst1q_s16(s0 + rb * 8, vaddq_s16(vld1q_s16(s0 + rb * 8), v0));
        vst1q_s16(s1 + rb * 8, vaddq_s16(vld1q_s16(s1 + rb * 8), v1));
      }
    }
    const float32x4_t k = vdupq_n_f32(ascale[g]);
    for(int rb = 0; rb < nb; ++rb){
      const int16x8_t ss = vaddq_s16(vld1q_s16(s0 + rb * 8), vld1q_s16(s1 + rb * 8));
      const float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(ss)));
      const float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(ss)));
      const float* sg = A->S + ((size_t)(b0 + rb) * G + g) * 8;
      float32x4_t a0 = vld1q_f32(yb + rb * 8), a1 = vld1q_f32(yb + rb * 8 + 4);
      a0 = vaddq_f32(a0, vmulq_f32(vmulq_f32(f0, vld1q_f32(sg)), k));
      a1 = vaddq_f32(a1, vmulq_f32(vmulq_f32(f1, vld1q_f32(sg + 4)), k));
      vst1q_f32(yb + rb * 8, a0);
      vst1q_f32(yb + rb * 8 + 4, a1);
    }
  }
}
#endif
#endif

/* inside the parallel region: up to 3 matrices with the same input; the threads build the pair tables, then split
   the concatenated rows in 8-row blocks; ends in a barrier */
static void gemv(Model* m, int n, const Mat* const* A, float* const* y, const float* x){
  const int prev = t_cur; TS(2);
  int a, b;
  span(A[0]->C / 2, &a, &b);
#if defined(KINDI_I16_ROWS8) || defined(KINDI_I16_STREAM)
  /* v3: per-128-col-group int8 activation quantisation (bitnet.cpp per_tensor_quant pattern).
     All threads compute the small scale table redundantly (C <= 6144, negligible). */
  const int G0 = A[0]->C / 128;
  float iscale[64], ascale[64];
  for(int g = 0; g < G0; ++g){
    float mx = 0.f;
    for(int e = g * 128; e < (g + 1) * 128; ++e){ const float v = fabsf(x[e]); if(v > mx) mx = v; }
    ascale[g] = mx / 127.0f;
    iscale[g] = mx > 0.f ? 127.0f / mx : 0.f;
  }
  pair_tables_i16(x, (int16_t*)m->T, a, b, iscale);
#else
  pair_tables(x, m->T, a, b);
#endif
#ifdef _OPENMP
  #pragma omp barrier
#endif
  long long Ut = 0;
  for(int i = 0; i < n; ++i) Ut += A[i]->R / 8;
  int a0, a1;
  span(Ut, &a0, &a1);
  int base = 0;
  for(int i = 0; i < n; ++i){
    const int U = A[i]->R / 8;
    const int gbase = base;   /* matrix global starting block for disjoint s01 */
    const int u0 = a0 > base ? a0 - base : 0, u1 = (a1 - base) < U ? (a1 - base) : U;
    base += U;
#ifdef KINDI_I16_STREAM
    if(u1 > u0) rows8_i16_stream(A[i], y[i], (const int16_t*)m->T, u0, u1, ascale, m->s01, gbase);
#elif defined(KINDI_I16_ROWS8)
    if(u1 > u0) rows8_i16(A[i], y[i], (const int16_t*)m->T, u0, u1, ascale);
#else
    if(u1 > u0) rows8(A[i], y[i], m->T, u0, u1);
#endif
  }
#ifdef _OPENMP
  #pragma omp barrier
#endif
  TS(prev);
}
static void gemv1(Model* m, const Mat* A, float* y, const float* x){ const Mat* a[1] = {A}; float* yy[1] = {y}; gemv(m, 1, a, yy, x); }

/* ------------------------------------------------------------------------------------------- norms */
/* residual add (if add) + RMS norm + fold of m->h into xn / xp; all threads; ends in a barrier */
static void rmsfold(Model* m, const float* w, int add){
  const int D = m->D;
  const int prev = t_cur; TS(1);
#ifdef _OPENMP
  #pragma omp single
#endif
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
#ifdef _OPENMP
  #pragma omp for schedule(static) nowait
#endif
  for(int h = 0; h < NV; ++h){
    float ab = 0.f, aa = 0.f;
    for(int i = 0; i < D; ++i){ ab += L->beta_w[(size_t)h * D + i] * m->xn[i]; aa += L->alpha_w[(size_t)h * D + i] * m->xn[i]; }
    m->gb[h] = sigm(ab);
    const float a = aa + L->dt_b[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    m->gg[h] = sp * L->ssm_a[h];
  }
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int c = 0; c < CC; ++c){
    float* cv = L->conv;
    const float c0 = cv[c], c1 = cv[CC + c], c2 = cv[2 * CC + c], nw = m->qkv[c];
    const float o = c0 * L->conv_w[4 * c] + c1 * L->conv_w[4 * c + 1] + c2 * L->conv_w[4 * c + 2] + nw * L->conv_w[4 * c + 3];
    cv[c] = c1; cv[CC + c] = c2; cv[2 * CC + c] = nw;
    m->cs[c] = silu(o);
  }
  /* L2 norm of the q and k heads */
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int r = 0; r < 2 * NK; ++r){
    const float* xr = m->cs + (size_t)r * HK;
    double acc = 0.0;
    for(int e = 0; e < HK; ++e) acc += (double)xr[e] * (double)xr[e];
    const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
    for(int e = 0; e < HK; ++e) m->qk[(size_t)r * HK + e] = xr[e] * sc;
  }
  /* delta rule per v head, then the per-head RMS norm */
  const float scale = 1.0f / sqrtf((float)HK);
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int hh = 0; hh < NH; ++hh){
    float* s = m->sc + (size_t)hh * m->ctx;
    float mx = -1e30f;
    for(int t = 0; t < n; ++t) mx = fmaxf(mx, s[t]);
    float tot = 0.f;
    for(int t = 0; t < n; ++t) tot += expf(s[t] - mx);
    for(int t = 0; t < n; ++t) s[t] = expf(s[t] - mx) / tot;
  }
  /* weighted V, gated; parallel over (head, 32-dim slice) */
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
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

static void ffn(Model* m, Layer* L){
  TS(5);
  rmsfold(m, L->post_norm, 1);
  { const Mat* a[2] = {&L->ffg, &L->ffu}; float* yy[2] = {m->ffg, m->ffu}; gemv(m, 2, a, yy, m->xp); }
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int e = 0; e < m->NFF; ++e) m->m[e] = silu(m->ffg[e]) * m->ffu[e];
  fold(m, m->m, m->sF, m->xp, m->NFF);
  gemv1(m, &L->ffd, m->y, m->xp);
}

static void embed(Model* m, int tok){
  const int C = m->emb.C, R = m->emb.R;
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int g = 0; g < C / 128; ++g){
    static const signed char W0[9] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};
    static const signed char W1[9] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
    const uint32_t* pw = (const uint32_t*)m->emb.W + ((size_t)(tok / 8) * (C / 128) + g) * 64;
    const float d = m->emb.S[((size_t)(tok / 8) * (C / 128) + g) * 8 + (tok & 7)];
    for(int p = 0; p < 64; ++p){
      const int code = (int)((pw[p] >> (4u * (uint32_t)(tok & 7))) & 15u);
      m->m[g * 128 + 2 * p] = (float)W0[code] * d;
      m->m[g * 128 + 2 * p + 1] = (float)W1[code] * d;
    }
  }
#ifdef _OPENMP
  #pragma omp for schedule(static)
#endif
  for(int bl = 0; bl < m->D / m->HB; ++bl){
    float* o = m->m + (size_t)bl * m->HB;
    fwht_block(o, m->HB);
    for(int j = 0; j < m->HB; ++j) m->h[bl * m->HB + j] = m->sD[bl * m->HB + j] * o[j];
  }
}

/* one token; head != 0: also the output head (logits in m->logits); returns the greedy token or -1 */
int kc_step(Model* m, int tok, int pos, int head){
  int best = -1;
  t_last = omp_get_wtime(); t_cur = 0;
#ifdef _OPENMP
  #pragma omp parallel
#endif
  {
    embed(m, tok);
    for(int il = 0; il < m->NL; ++il){
      Layer* L = m->L + il;
      if(L->is_attn) attn(m, L, pos, il > 0); else gdn(m, L, il > 0);
      ffn(m, L);
    }
    if(head){
      TS(6);
      rmsfold(m, m->out_norm, 1);
      gemv1(m, &m->head, m->logits, m->xp);
#ifdef _OPENMP
      #pragma omp single
#endif
      {
        int bi = 0; float bv = m->logits[0];
        for(int i = 1; i < m->head.R; ++i) if(m->logits[i] > bv){ bv = m->logits[i]; bi = i; }
        best = bi;
      }
    } else {
#ifdef _OPENMP
      #pragma omp single
#endif
      { for(int e = 0; e < m->D; ++e) m->h[e] += m->y[e]; }   /* keep h = the residual after the last block */
    }
  }
  kc_times[t_cur] += omp_get_wtime() - t_last;
  return best;
}

int kc_sizeof_model(void){ return (int)sizeof(Model); }
int kc_sizeof_layer(void){ return (int)sizeof(Layer); }
