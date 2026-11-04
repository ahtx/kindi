/* tcp.cu - the tc.cu kernels (names, arguments, launch shapes) for Pascal (sm_61): int8 activations, dp4a, no
   tensor cores. Compiled with kindi.cu (kdp4a, KLDG) in front.

   Weights: k_tc_pack turns the 5-trit lane layout into the "pl" layout, the same size. Per 16-row tile m and 128-column
   group g (tile-major, so a warp's rows stream from one place), the 7 words of each row: words 0 .. 5 hold four
   five-trit bytes each, word 6 two four-trit bytes and the fp16 scale. In a tile-group's 112 words, word k < 6 of row
   rr is at 32 (k >> 1) + 2 rr + (k & 1) and word 6 at 96 + rr, so each of a warp's loads reads 128 (or 64) contiguous
   bytes. A byte is the TQ1 code ceil(v 256 / 3^n)
   of its n digits v = sum d_q 3^(n - 1 - q), so digit q is the high byte of 3 ((b 3^q) & 255): with two bytes in the
   16-bit lanes of a word (lo = bytes 0, 2; hi = bytes 1, 3), m_q = 3 y_q and y_(q+1) = m_q & 0x00FF00FF give digit q
   of both bytes in bytes 1 and 3 of m_q, two instructions per digit pair. Register i (0 .. 4) of word k is
   __byte_perm(m_Q, m_(Q+1), 0x7531), Q = 2 i (masks 0 .. 4 = lo digits 0 .. 4, 5 .. 9 = hi digits 0 .. 4): four
   digits (0 / 1 / 2) of the columns 20 k + 4 i .. 20 k + 4 i + 3 in order, so a register meets one natural int8
   activation word in one dp4a. Word 6: bytes 0, 1 into the lanes (byte_perm 0x4140), masks 0 .. 3, registers 30 and
   31 (columns 120 .. 127).

   Activations: k_tc_q8 and the fused quantisers (tcf.cu) as on the tensor-core path (XQ in the tc order, XS, XA).

   Numerics: tc.cu's. A group's 128-term dot product of the digits with the int8 activations is an exact integer; the
   group's activation sum is subtracted (sum (d - 1) q = sum d q - sum q); it is scaled by x-scale * w-scale and added
   in fp32, group after group, in tc.cu's expression, and the column splits are tc.cu's. So a token's result does not
   depend on how many tokens share the pass (decode = verify), and it is bit-identical to the tensor-core kernels.

   Threads: two per row (h = 0: words 0, 2, 4, 6, 17 registers and the scale; h = 1: words 1, 3, 5, 15 registers);
   their integer partial sums are added with one shuffle. A block of 32 NW threads covers 16 NW rows, as
   tc.cu's NW warps of one 16-row tile each. */

__device__ __forceinline__ float tc_h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

/* ------------------------------------------------------------------ repack: 5-trit lane layout -> pl layout */
__device__ __forceinline__ unsigned int tc_src_trit(const unsigned char* __restrict__ S, int NT16, int r, int G, int c){
  int p, e;
  if(c < 120){ p = c / 5; e = c - 5 * p; } else { p = 24 + (c - 120) / 4; e = (c - 120) & 3; }
  unsigned int x = S[(((long long)G * NT16 + (r >> 4)) * 28 + p) * 16 + (r & 15)];
  for(int i = 0; i < e; ++i) x /= 3u;
  return x % 3u;
}

extern "C" __global__ void k_tc_pack(const unsigned char* __restrict__ S, unsigned char* __restrict__ O, int R, int NG){
  /* one thread per output byte; O = [m][g] blocks of 112 words: word k < 6 of row rr at 32 (k >> 1) + 2 rr + (k & 1),
     word 6 at 96 + rr */
  const int NT16 = R >> 4;
  const long long n = (long long)R * NG * 28;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;
  const int b = (int)(idx & 3);
  const long long wi = idx >> 2;
  const int o = (int)(wi % 112);
  const int rr = o < 96 ? (o & 31) >> 1 : o - 96, k = o < 96 ? 2 * (o >> 5) + (o & 1) : 6;
  const long long mg = wi / 112;
  const int g = (int)(mg % NG), m = (int)(mg / NG), r = 16 * m + rr;
  unsigned int out;
  if(k == 6 && b >= 2){
    out = S[(((long long)g * NT16 + m) * 28 + 26 + (b & 1)) * 16 + rr];
  } else if(k == 6){
    unsigned int v = 0u;                           /* lane b, digit q: column 120 + 2 q + b */
    for(int q = 0; q < 4; ++q) v = v * 3u + tc_src_trit(S, NT16, r, g, 120 + 2 * q + b);
    out = (v * 256u + 80u) / 81u;
  } else {
    unsigned int v = 0u;                           /* half b & 1 (lo / hi), lane b >> 1, mask Q = q + 5 half */
    for(int q = 0; q < 5; ++q){
      const int Q = q + 5 * (b & 1), i = Q >> 1, j = 2 * (Q & 1) + (b >> 1);
      v = v * 3u + tc_src_trit(S, NT16, r, g, 20 * k + 4 * i + j);
    }
    out = (v * 256u + 242u) / 243u;
  }
  O[idx] = (unsigned char)out;
}

/* ------------------------------------------------------------------ activations: int8 per 128-column group */
extern "C" __global__ void k_tc_q8(const float* __restrict__ X, int ldx, unsigned int* __restrict__ XQ,
                                   float* __restrict__ XS, int* __restrict__ XA, int NG, int ntok){
  /* tc.cu's k_tc_q8: XQ[((n NG + g) 4 + t) 8 + 2 s + h] = 4 int8 of columns 32 s + 16 h + 4 t .., XS = amax / 127,
     XA = the sum of the 128 int8 values */
  const int g = blockIdx.x, n = blockIdx.y, e = threadIdx.x, lane = e & 31, w = e >> 5;
  if(n >= ntok) return;
  __shared__ float wm[4];
  __shared__ int ws[4];
  const float v = X[(long long)n * ldx + g * 128 + e];
  float a = fabsf(v);
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  if(lane == 0) wm[w] = a;
  __syncthreads();
  const float amax = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
  const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
  const int qi = __float2int_rn(v * id);
  int sum = qi;
  for(int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if(lane == 0) ws[w] = sum;
  unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3));
  word |= __shfl_xor_sync(0xffffffffu, word, 1);
  word |= __shfl_xor_sync(0xffffffffu, word, 2);
  if((lane & 3) == 0){
    const int k4 = e >> 2, s = k4 >> 3, h = (k4 >> 2) & 1, t = k4 & 3;
    XQ[(((long long)n * NG + g) * 4 + t) * 8 + 2 * s + h] = word;
  }
  __syncthreads();
  if(e == 0){
    XS[(long long)n * NG + g] = amax / 127.0f;
    XA[(long long)n * NG + g] = ws[0] + ws[1] + ws[2] + ws[3];
  }
}

/* ------------------------------------------------------------------ digits */
/* 3 y as one shift-add (sm_61 would otherwise multiply with XMADs) */
__device__ __forceinline__ unsigned int tp_x3(unsigned int y){
  unsigned int m;
  asm("{ .reg .u32 t; shl.b32 t, %1, 1; add.u32 %0, t, %1; }" : "=r"(m) : "r"(y));
  return m;
}

__device__ __forceinline__ void tp_chain5(unsigned int y, unsigned int m[5]){
#pragma unroll
  for(int q = 0; q < 5; ++q){ m[q] = tp_x3(y); y = m[q] & 0x00FF00FFu; }
}

/* the five registers of a word of four five-trit bytes */
__device__ __forceinline__ void tp_word(unsigned int w, unsigned int* r){
  unsigned int a[5], b[5];
  tp_chain5(w & 0x00FF00FFu, a);
  tp_chain5(__byte_perm(w, 0u, 0x4341), b);
  r[0] = __byte_perm(a[0], a[1], 0x7531);
  r[1] = __byte_perm(a[2], a[3], 0x7531);
  r[2] = __byte_perm(a[4], b[0], 0x7531);
  r[3] = __byte_perm(b[1], b[2], 0x7531);
  r[4] = __byte_perm(b[3], b[4], 0x7531);
}

/* the two registers of word 6 (two four-trit bytes) */
__device__ __forceinline__ void tp_word6(unsigned int w, unsigned int* r){
  unsigned int y = __byte_perm(w, 0u, 0x4140), m[4];
#pragma unroll
  for(int q = 0; q < 4; ++q){ m[q] = tp_x3(y); y = m[q] & 0x00FF00FFu; }
  r[0] = __byte_perm(m[0], m[1], 0x7531);
  r[1] = __byte_perm(m[2], m[3], 0x7531);
}

/* K (0 .. 31, columns 4 K ..) of a word of XQ's tc order at offset o */
__device__ __forceinline__ int tp_k(int o){ return 8 * ((o & 7) >> 1) + 4 * (o & 1) + (o >> 3); }

/* ------------------------------------------------------------------ the product, 1 .. 8 tokens */
#define TP_UG 8                                       /* groups per staged unit */

__device__ __forceinline__ unsigned int tp_ldw(const unsigned int* p){ return KLDG(p); }

/* a token's 32 words in shared memory, each thread's in the order of its registers, each half on 16 bytes:
   h = 0 (K 0 .. 4, 10 .. 14, 20 .. 24, 30, 31) at 0 .. 16, h = 1 (K 5 .. 9, 15 .. 19, 25 .. 29) at 20 .. 34 */
#define TP_XLD 36
__device__ __forceinline__ int tp_pos(int K){
  if(K >= 30) return K - 15;
  const int q = K / 5, e = K - 5 * q;
  return (q & 1) ? 20 + 5 * (q >> 1) + e : 5 * (q >> 1) + e;
}

struct TPS {
  unsigned int x[TP_UG][8][TP_XLD];                   /* [group][token][tp_pos(K)] */
  float xs[TP_UG][8];
  int xa[TP_UG][8];
};

__device__ __forceinline__ void tp_stage(TPS& S, const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                         const int* __restrict__ XA, int NG, int g0, int ntok){
  /* thread c = tid % 64 copies 16 bytes q = c % 8 of group G = c / 8 for tokens n = tid / 64, + nt / 64, ..: words
     4 q + e of the tc order are K = 16 (q & 1) + (q >> 1) + 4 e */
  const int tid = threadIdx.x, c = tid & 63, G = c >> 3, q = c & 7, kb = 16 * (q & 1) + (q >> 1);
  const long long sn = (long long)NG * 32;
  const uint4* p = (const uint4*)(XQ + (long long)(g0 + G) * 32) + q;
  for(int n = tid >> 6; n < ntok; n += blockDim.x >> 6){
    const uint4 v = KLDG((const uint4*)((const unsigned int*)p + n * sn));
    unsigned int* x = S.x[G][n];
    x[tp_pos(kb)] = v.x; x[tp_pos(kb + 4)] = v.y; x[tp_pos(kb + 8)] = v.z; x[tp_pos(kb + 12)] = v.w;
  }
  if(tid < 64){
    const int Gs = tid >> 3, n = tid & 7;
    if(n < ntok){
      S.xs[Gs][n] = KLDG(XS + (long long)n * NG + g0 + Gs);
      S.xa[Gs][n] = KLDG(XA + (long long)n * NG + g0 + Gs);
    }
  }
}

/* this thread's half row of one group (wv: words h, h + 2, h + 4 and, h = 0, word 6): its registers (h = 0: 17,
   h = 1: 15) and the row scale (h = 0) */
__device__ __forceinline__ void tp_regs(const unsigned int wv[4], int h, unsigned int r[17], float& d){
  tp_word(wv[0], r); tp_word(wv[1], r + 5); tp_word(wv[2], r + 10);
  if(h == 0){
    tp_word6(wv[3], r + 15);
    d = tc_h2f((unsigned short)(wv[3] >> 16));
  } else {
    r[15] = r[16] = 0u;
    d = 0.f;
  }
}

/* this thread's 4 words of group g (Wr: the row's tile at group 0, + 2 rr + h; w6: word 6 from Wr, h = 0 only) */
__device__ __forceinline__ void tp_load(const unsigned int* __restrict__ Wr, int g, int h, int w6, unsigned int v[4]){
  const unsigned int* p = Wr + (long long)g * 112;
  v[0] = tp_ldw(p); v[1] = tp_ldw(p + 32); v[2] = tp_ldw(p + 64); v[3] = h ? 0u : tp_ldw(p + w6);
}

/* rows: this thread's row, units [u_lo, u_hi), NT tokens. The groups are walked one by one (a compact loop: Pascal's
   instruction cache), the weights loaded two groups ahead; the activations are staged per unit. Every thread of the
   block takes part in the staging and the barriers. */
template <int NT>
__device__ __forceinline__ void tp_rows(TPS& S, const unsigned int* __restrict__ Wr, bool active, int h, int u_lo,
                                        int u_hi, int NG, const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                        const int* __restrict__ XA, float acc[8]){
  const int w6 = 96 - (int)((threadIdx.x >> 1) & 15);
  const int g_hi = u_hi * TP_UG;
  unsigned int na[4] = {0u, 0u, 0u, 0u}, nb[4] = {0u, 0u, 0u, 0u};
  if(active && u_lo < u_hi){
    tp_load(Wr, u_lo * TP_UG, h, w6, na);
    tp_load(Wr, u_lo * TP_UG + 1, h, w6, nb);
  }
  for(int u = u_lo; u < u_hi; ++u){
    __syncthreads();
    tp_stage(S, XQ, XS, XA, NG, u * TP_UG, NT);
    __syncthreads();
    if(!active) continue;
#pragma unroll 1
    for(int G = 0; G < TP_UG; ++G){
      const int g = u * TP_UG + G;
      unsigned int cw[4] = {na[0], na[1], na[2], na[3]};
#pragma unroll
      for(int k = 0; k < 4; ++k) na[k] = nb[k];
      if(g + 2 < g_hi) tp_load(Wr, g + 2, h, w6, nb);
      unsigned int r[17];
      float d;
      tp_regs(cw, h, r, d);
      d = __shfl_xor_sync(0xffffffffu, d, 1) + d;      /* (one half holds 0) */
#pragma unroll
      for(int j = 0; j < NT; ++j){
        const unsigned int* xr = S.x[G][j] + 20 * h;
        const uint4* x4 = (const uint4*)xr;
        int c = 0;
#pragma unroll
        for(int q = 0; q < 4; ++q){
          const uint4 v = x4[q];
          c = kdp4a((int)r[4 * q], (int)v.x, c); c = kdp4a((int)r[4 * q + 1], (int)v.y, c);
          c = kdp4a((int)r[4 * q + 2], (int)v.z, c); c = kdp4a((int)r[4 * q + 3], (int)v.w, c);
        }
        if(!h) c = kdp4a((int)r[16], (int)xr[16], c);
        c += __shfl_xor_sync(0xffffffffu, c, 1);
        acc[j] += (float)(c - S.xa[G][j]) * (S.xs[G][j] * d);
      }
    }
  }
}

/* the token count as a template argument: only that count's loop runs (a compact loop) */
__device__ __forceinline__ void tp_rows_n(const unsigned int* __restrict__ Wr, bool active, int h, int u_lo, int u_hi,
                                          int NG, const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                          const int* __restrict__ XA, int ntok, float acc[8]){
  __shared__ TPS S;
  switch(ntok){
    case 1: tp_rows<1>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    case 2: tp_rows<2>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    case 3: tp_rows<3>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    case 4: tp_rows<4>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    case 5: tp_rows<5>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    case 6: tp_rows<6>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    case 7: tp_rows<7>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
    default: tp_rows<8>(S, Wr, active, h, u_lo, u_hi, NG, XQ, XS, XA, acc); break;
  }
}

template <int NW>
__device__ __forceinline__ void tc_mv(const unsigned int* __restrict__ W, float* __restrict__ y, int R, int NU,
                                      const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                      const int* __restrict__ XA, int ntok){
  /* grid (ceil(R / 16 / NW), splits): block y covers units [by NU / splits, (by + 1) NU / splits); token n's part at
     y + (n splits + by) R (as tc.cu) */
  const int tid = threadIdx.x, h = tid & 1, r = blockIdx.x * 16 * NW + (tid >> 1), NG = NU * 8;
  const bool active = r < R;
  const int u_lo = (int)((long long)blockIdx.y * NU / gridDim.y), u_hi = (int)((long long)(blockIdx.y + 1) * NU / gridDim.y);
  float acc[8];
#pragma unroll
  for(int j = 0; j < 8; ++j) acc[j] = 0.f;
  const int rv = active ? r : 0;
  tp_rows_n(W + (long long)(rv >> 4) * NG * 112 + 2 * (rv & 15) + h, active, h, u_lo, u_hi, NG, XQ, XS, XA, ntok, acc);
  if(!active || h) return;
  const int sp = gridDim.y, by = blockIdx.y;
#pragma unroll
  for(int j = 0; j < 8; ++j) if(j < ntok) y[((long long)j * sp + by) * R + r] = acc[j];
}

template <int NW>
__device__ __forceinline__ void tc_mv3(const unsigned int* __restrict__ W0, float* __restrict__ y0, int R0,
                                       const unsigned int* __restrict__ W1, float* __restrict__ y1, int R1,
                                       const unsigned int* __restrict__ W2, float* __restrict__ y2, int R2, int NU,
                                       const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                       const int* __restrict__ XA, int ntok){
  /* the blocks along x cover the rows of W0, then W1, then W2 (no splits) */
  const int tid = threadIdx.x, h = tid & 1, NG = NU * 8;
  int r = blockIdx.x * 16 * NW + (tid >> 1);
  const unsigned int* W = W0;
  float* y = y0;
  int R = R0;
  if(r >= R0){
    r -= R0; W = W1; y = y1; R = R1;
    if(r >= R1){ r -= R1; W = W2; y = y2; R = R2; }
  }
  const bool active = r < R;
  float acc[8];
#pragma unroll
  for(int j = 0; j < 8; ++j) acc[j] = 0.f;
  const int rv = active ? r : 0;
  tp_rows_n((active ? W : W0) + (long long)(rv >> 4) * NG * 112 + 2 * (rv & 15) + h, active, h, 0, NU, NG, XQ, XS, XA, ntok, acc);
  if(!active || h) return;
#pragma unroll
  for(int j = 0; j < 8; ++j) if(j < ntok) y[(long long)j * R + r] = acc[j];
}

#define TC_MV3_ARGS const unsigned int* __restrict__ W0, float* __restrict__ y0, int R0, const unsigned int* __restrict__ W1, \
  float* __restrict__ y1, int R1, const unsigned int* __restrict__ W2, float* __restrict__ y2, int R2, int NU, \
  const unsigned int* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok
extern "C" __global__ void __launch_bounds__(64) k_tc_mv3_2(TC_MV3_ARGS){ tc_mv3<2>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv3_4(TC_MV3_ARGS){ tc_mv3<4>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv3_8(TC_MV3_ARGS){ tc_mv3<8>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }

#define TC_MV_ARGS const unsigned int* __restrict__ W, float* __restrict__ y, int R, int NU, \
  const unsigned int* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok
extern "C" __global__ void __launch_bounds__(64) k_tc_mv2(TC_MV_ARGS){ tc_mv<2>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv4(TC_MV_ARGS){ tc_mv<4>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv8(TC_MV_ARGS){ tc_mv<8>(W, y, R, NU, XQ, XS, XA, ntok); }

extern "C" __global__ void k_tc_sum(const float* __restrict__ P, float* __restrict__ y, int R, int ntok, int sp){
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)R * ntok) return;
  const long long n = i / R, r = i - n * R;
  const float* q = P + n * sp * R + r;
  float s = q[0];
  for(int p = 1; p < sp; ++p) s += q[(long long)p * R];
  y[i] = s;
}

/* ------------------------------------------------------------------ prompt chunks: Y[n ldy + r] */
/* k_tp_mm: grid (R / 128, ceil(T / 128)), 256 threads; chunk.cu's k_mm_g5q on the pl layout: per group, the block's
   8 row tiles are unpacked to int8 trits (-1 / 0 / 1) in shared memory, the 128 tokens' int8 rows (XQ, tc order) are
   staged in natural column order, and each thread's 8 tokens x 8 rows integer sums are exact (dp4a), scaled by
   x-scale * w-scale and added in fp32 group after group (= tc.cu's (c - XA) expression: the same integer). */
#define TPM_LD 132

struct TPM {
  unsigned int A[32][TPM_LD];                         /* A[K][t] */
  unsigned int B[32][TPM_LD];                         /* B[K][r]: four int8 trits */
  unsigned int raw[8 * 112];                          /* the group's words of the block's 8 tiles */
  float d[128];
  float xs[128];
};

__device__ __forceinline__ unsigned int tp_trits(unsigned int r){ return (r + 0x7F7F7F7Fu) ^ 0x80808080u; }

extern "C" __global__ void __launch_bounds__(256, 1) k_tp_mm(const uint4* __restrict__ W, int R, int NU,
    const unsigned int* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA,
    float* __restrict__ Y, int ldy, int T){
  __shared__ TPM S;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4, NG = NU * 8;
  const int r0 = blockIdx.x * 128, t0 = blockIdx.y * 128, m0 = r0 >> 4;
  float acc[8][8];
#pragma unroll
  for(int i = 0; i < 8; ++i)
#pragma unroll
    for(int j = 0; j < 8; ++j) acc[i][j] = 0.f;
  for(int g = 0; g < NG; ++g){
    __syncthreads();
    if(t < 8 * 28){
      const int tile = t / 28, p = t - tile * 28;
      ((uint4*)S.raw)[t] = KLDG(W + ((long long)(m0 + tile) * NG + g) * 28 + p);
    }
#pragma unroll
    for(int j = 0; j < 16; ++j){
      const int i = t + 256 * j, n = i >> 5, o = i & 31;
      S.A[tp_k(o)][n] = t0 + n < T ? KLDG(XQ + ((long long)(t0 + n) * NG + g) * 32 + o) : 0u;
    }
    if(t < 128) S.xs[t] = t0 + t < T ? KLDG(XS + (long long)(t0 + t) * NG + g) : 0.f;
    __syncthreads();
    {
      const int h = t >> 7, r = t & 127, tile = r >> 4, rr = r & 15;
      const unsigned int* rw = S.raw + tile * 112;
      unsigned int wv[4] = {rw[2 * rr + h], rw[32 + 2 * rr + h], rw[64 + 2 * rr + h], h ? 0u : rw[96 + rr]};
      unsigned int rg[17];
      float d;
      tp_regs(wv, h, rg, d);
#pragma unroll
      for(int i = 0; i < 15; ++i) S.B[5 * (2 * (i / 5) + h) + i % 5][r] = tp_trits(rg[i]);   /* word 2 (i / 5) + h */
      if(!h){ S.B[30][r] = tp_trits(rg[15]); S.B[31][r] = tp_trits(rg[16]); S.d[r] = d; }
    }
    __syncthreads();
    int is[8][8];
#pragma unroll
    for(int i = 0; i < 8; ++i)
#pragma unroll
      for(int j = 0; j < 8; ++j) is[i][j] = 0;
#pragma unroll 4
    for(int k4 = 0; k4 < 32; ++k4){
      const uint4 a0 = *(const uint4*)&S.A[k4][tx * 4], a1 = *(const uint4*)&S.A[k4][64 + tx * 4];
      const uint4 b0 = *(const uint4*)&S.B[k4][ty * 4], b1 = *(const uint4*)&S.B[k4][64 + ty * 4];
      const int a[8] = {(int)a0.x, (int)a0.y, (int)a0.z, (int)a0.w, (int)a1.x, (int)a1.y, (int)a1.z, (int)a1.w};
      const int b[8] = {(int)b0.x, (int)b0.y, (int)b0.z, (int)b0.w, (int)b1.x, (int)b1.y, (int)b1.z, (int)b1.w};
#pragma unroll
      for(int i = 0; i < 8; ++i)
#pragma unroll
        for(int j = 0; j < 8; ++j) is[i][j] = kdp4a(a[i], b[j], is[i][j]);
    }
#pragma unroll
    for(int i = 0; i < 8; ++i){
      const float xs = S.xs[i < 4 ? tx * 4 + i : 64 + tx * 4 + i - 4];
#pragma unroll
      for(int j = 0; j < 8; ++j){
        const float dj = S.d[j < 4 ? ty * 4 + j : 64 + ty * 4 + j - 4];
        acc[i][j] += (float)is[i][j] * (xs * dj);
      }
    }
  }
#pragma unroll
  for(int i = 0; i < 8; ++i){
    const int tok = t0 + (i < 4 ? tx * 4 + i : 64 + tx * 4 + i - 4);
    if(tok >= T) continue;
#pragma unroll
    for(int hh = 0; hh < 2; ++hh)
      *(float4*)(Y + (long long)tok * ldy + r0 + hh * 64 + ty * 4) =
          make_float4(acc[i][4 * hh], acc[i][4 * hh + 1], acc[i][4 * hh + 2], acc[i][4 * hh + 3]);
  }
}
