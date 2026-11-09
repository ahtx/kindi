/* tc.cu - int8 tensor-core matrix products (mma.sync m16n8k32 .s8, sm_80+) for 1 .. 8 tokens: decode and the
   speculative verify pass use the same kernel, so a token's outputs do not depend on how many tokens share the pass.

   Numerics (llama.cpp-style q8, the same as the chunk path's k_mm_g5q): each token's activations are quantised per
   128-column group (scale amax / 127, round to nearest even); a group's 128-term dot product with the trits is an
   exact integer; it is scaled by x-scale * w-scale and added in fp32, group after group. The trits enter the mma as
   digits 0 / 1 / 2 and the group's activation sum is subtracted (sum (d - 1) q = sum d q - sum q, exact).

   Weight layout "tc" (built on the card from the 5-trit lane layout, same size): per 16-row tile m and unit u of 8
   column groups (1024 columns), 7 x 32 uint4 = 3584 bytes, [i][lane]: lane l = (g8 = l >> 2, t = l & 3) owns the
   mma A fragments of rows g8 and g8 + 8 for all 8 groups (8 x 4 k-steps x 4 registers = 128 registers of 4 int8)
   in 26 words of five-trit bytes, plus 2 words of fp16 scales (word 26: groups 2t, word 27: groups 2t + 1, each
   [row g8 | row g8 + 8]). A five-trit byte is the TQ1 code ceil(v * 256 / 243) of v = 81 d0 + 27 d1 + 9 d2 + 3 d3 + d4,
   so digit j = byte 1 of 3 ((b * 3^j) & 255); two bytes are decoded at once in 16-bit lanes. Word k gives registers
   5k .. 5k + 4: lo = bytes 0, 2; hi = bytes 1, 3; masks m(src, j) = 3 ((src * 3^j) & 0x00FF00FF); register i is
   [x.b1, x.b3, y.b1, y.b3] of the mask pair (q = 2i, 2i + 1) with q = 0 .. 4: (lo, q), q = 5 .. 9: (hi, q - 5).
   Register rho (< 128) = group rho >> 4, k-step (rho >> 2) & 3, fragment rho & 3 (row g8 + 8 (f & 1), columns
   32 s + 16 (f >> 1) + 4 t + byte). */

__device__ __forceinline__ float tc_h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

#define TC_MMA(c, a0, a1, a2, a3, b0, b1) \
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define TC_MMAU(c, a0, a1, a2, a3, b0, b1) \
  asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ unsigned int tc_mask(unsigned int src, unsigned int p){ return 3u * ((src * p) & 0x00FF00FFu); }

#ifndef TC_INC
#define TC_INC 1
#endif
/* 3 v_Q, v_0 = src, v_{q+1} = (3 v_q) & 0x00FF00FF: the same bytes as tc_mask(src, 3^Q) (both 16-bit lanes stay
   below 2^16, so (src 3^Q) & 0x00FF00FF = v_Q), but each digit is one shift-add and one mask on from the last one,
   and the five registers of a word share the chains (no 32-bit multiplies) */
template <int Q>
__device__ __forceinline__ unsigned int tc_dig(unsigned int src){
  unsigned int v = src;
#pragma unroll
  for(int q = 0; q < Q; ++q) v = (v * 3u) & 0x00FF00FFu;
  return v * 3u;
}

/* register i (0 .. 4) of word w */
template <int I>
__device__ __forceinline__ unsigned int tc_reg(unsigned int w){
#if TC_INC
  const unsigned int lo = w & 0x00FF00FFu, hi = (w >> 8) & 0x00FF00FFu;
  const int q0 = 2 * I, q1 = 2 * I + 1;
  const unsigned int x = q0 < 5 ? tc_dig<(q0 < 5 ? q0 : 0)>(lo) : tc_dig<(q0 < 5 ? 0 : q0 - 5)>(hi);
  const unsigned int y = q1 < 5 ? tc_dig<(q1 < 5 ? q1 : 0)>(lo) : tc_dig<(q1 < 5 ? 0 : q1 - 5)>(hi);
  return __byte_perm(x, y, 0x7531);
#else
  const unsigned int lo = w & 0x00FF00FFu, hi = (w >> 8) & 0x00FF00FFu;
  const unsigned int P[5] = {1u, 3u, 9u, 27u, 81u};
  const int q0 = 2 * I, q1 = 2 * I + 1;
  const unsigned int x = q0 < 5 ? (q0 == 0 ? 3u * lo : tc_mask(lo, P[q0])) : (q0 == 5 ? 3u * hi : tc_mask(hi, P[q0 - 5]));
  const unsigned int y = q1 < 5 ? tc_mask(lo, P[q1]) : (q1 == 5 ? 3u * hi : tc_mask(hi, P[q1 - 5]));
  return __byte_perm(x, y, 0x7531);
#endif
}

template <int RHO>
__device__ __forceinline__ unsigned int tc_r(const unsigned int* w){ return tc_reg<RHO % 5>(w[RHO / 5]); }

/* Widths that are not a multiple of 1024 (the 4B: 2560): the last unit of a row tile is padded to 8 groups (the
   pad groups hold zero trits and zero scales) and the kernels skip its groups at or past NG, so a product's sums
   are those of its real groups only, in the same order. The kernels then take NUG = NU | NG << 16 (units, real
   groups) where they took NU (NG = 8 NU when the high half is 0); the activation rows keep NG groups each. A half
   unit (4 groups) loads only the uint4 rows 0 .. 3 and 6 of its 7 (groups 0 .. 3 use words 0 .. 12 and the scales). */
__device__ __forceinline__ int tc_nu(int nug){ return nug & 0xFFFF; }
__device__ __forceinline__ int tc_ng(int nug){ return ((nug >> 16) & 0x3FFF) ? ((nug >> 16) & 0x3FFF) : (nug & 0xFFFF) * 8; }
/* NUG bit 30 (KINDI_TC_HP, decode and verify passes): every token's int8 row is two rows, hi = the usual one and lo
   = its residual v - q amax / 127, quantised like hi and stored as tokens 2n and 2n + 1 (ntok counts these). Each
   thread's two mma columns then hold token t's hi and lo sums, and the output is their sum (about 14 bits of
   activation precision instead of 8, for the same weight traffic and mma count). */
__device__ __forceinline__ bool tc_hp(int nug){ return (nug >> 30) & 1; }
__device__ __forceinline__ bool tc_row_used(int i, int ng_left){ return ng_left > 4 || !(i == 4 || i == 5); }

/* ------------------------------------------------------------------ repack: 5-trit lane layout -> tc layout */
__device__ __forceinline__ unsigned int tc_src_trit(const unsigned char* __restrict__ S, int NT16, int r, int G, int c){
  int p, e;
  if(c < 120){ p = c / 5; e = c - 5 * p; } else { p = 24 + (c - 120) / 4; e = (c - 120) & 3; }
  unsigned int x = S[(((long long)G * NT16 + (r >> 4)) * 28 + p) * 16 + (r & 15)];
  for(int i = 0; i < e; ++i) x /= 3u;
  return x % 3u;
}

extern "C" __global__ void k_tc_pack(const unsigned char* __restrict__ S, unsigned char* __restrict__ O, int R, int NG){
  /* one thread per output byte; O = tiles [m][u][i 0..6][lane][16 bytes] */
  const int NT16 = R >> 4, NU = (NG + 7) >> 3;                    /* NG % 8 != 0: the last unit is padded */
  const long long n = (long long)NT16 * NU * 3584;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;
  const int b16 = (int)(idx & 15), lane = (int)((idx >> 4) & 31), i = (int)((idx >> 9) % 7);
  const long long mu = idx / 3584;
  const int u = (int)(mu % NU), m = (int)(mu / NU);
  const int wk = i * 4 + (b16 >> 2), bq = b16 & 3, g8 = lane >> 2, t = lane & 3;
  unsigned int out;
  if(wk >= 26){
    const int G = u * 8 + 2 * t + (wk - 26);
    const int r = 16 * m + g8 + 8 * (bq >> 1);
    out = G < NG ? S[(((long long)G * NT16 + m) * 28 + 26 + (bq & 1)) * 16 + (r & 15)] : 0u;   /* pad group: scale 0 */
  } else {
    unsigned int v = 0u;
    const bool src_hi = (bq & 1) != 0, lane_b = (bq & 2) != 0;      /* B0: lo a, B1: hi a, B2: lo b, B3: hi b */
    for(int j = 0; j < 5; ++j){
      const int q = (src_hi ? 5 : 0) + j, ri = q >> 1, beta = (q & 1) * 2 + (lane_b ? 1 : 0);
      const int rho = 5 * wk + ri;
      unsigned int d = 1u;
      if(rho < 128){
        const int Gp = rho >> 4, s = (rho >> 2) & 3, f = rho & 3;
        const int r = 16 * m + g8 + 8 * (f & 1), c = s * 32 + 16 * (f >> 1) + 4 * t + beta;
        d = u * 8 + Gp < NG ? tc_src_trit(S, NT16, r, u * 8 + Gp, c) : 1u;      /* pad group: zero trits */
      }
      v = v * 3u + d;
    }
    out = (v * 256u + 242u) / 243u;
  }
  O[idx] = (unsigned char)out;
}

/* ------------------------------------------------------------------ the b2 layout (KINDI_TC_B2): 2 bits a digit
   The same digits and scales as the tc layout, per 16-row tile m and unit u 9 x 32 uint4 = 4608 bytes, [i][lane]:
   word 4 G + f (G < 8) holds fragment f of group G for the 4 k-steps, the digit of register (G, s, f) byte beta at bits
   8 beta + 2 s; words 32 / 33 = the scale words 26 / 27 of the tc layout; words 34, 35 = 0. A register is the word AND
   (3 << 2 s) in each byte: 4^s times the digits (<= 128, unsigned). k_tc_mv*b run a u8 mma per k-step and add the four
   k-step sums shifted back by 2 s (exact), so the products are bit-identical to the tc layout's, with one logic
   operation per register instead of the five-trit decode (the decode is the limit on cards with many SMs). */
extern "C" __global__ void k_tc_pack2(const unsigned char* __restrict__ S, unsigned int* __restrict__ O, int R, int NG){
  /* one thread per output word; O = tiles [m][u][i 0..8][lane][4 words] */
  const int NT16 = R >> 4, NU = (NG + 7) >> 3;                    /* NG % 8 != 0: the last unit is padded */
  const long long n = (long long)NT16 * NU * 1152;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;
  const int q = (int)(idx & 3), lane = (int)((idx >> 2) & 31), i = (int)((idx >> 7) % 9);
  const long long mu = idx / 1152;
  const int u = (int)(mu % NU), m = (int)(mu / NU);
  const int wk = i * 4 + q, g8 = lane >> 2, t = lane & 3;
  unsigned int out = 0u;
  if(wk < 32 && u * 8 + (wk >> 2) < NG){                          /* pad groups: zero words (the kernels skip them) */
    const int G = wk >> 2, f = wk & 3, r = 16 * m + g8 + 8 * (f & 1);
    for(int s = 0; s < 4; ++s)
      for(int beta = 0; beta < 4; ++beta){
        const int c = s * 32 + 16 * (f >> 1) + 4 * t + beta;
        out |= tc_src_trit(S, NT16, r, u * 8 + G, c) << (8 * beta + 2 * s);
      }
  } else if(wk >= 32 && wk < 34 && u * 8 + 2 * t + (wk - 32) < NG){
    const int G = u * 8 + 2 * t + (wk - 32);
    for(int bq = 0; bq < 4; ++bq){
      const int r = 16 * m + g8 + 8 * (bq >> 1);
      out |= (unsigned int)S[(((long long)G * NT16 + m) * 28 + 26 + (bq & 1)) * 16 + (r & 15)] << (8 * bq);
    }
  }
  O[idx] = out;
}

/* ------------------------------------------------------------------ activations: int8 per 128-column group */
__device__ __forceinline__ int tcq128(float v, int n, int g, unsigned int* __restrict__ XQ, float* __restrict__ XS,
                                      int* __restrict__ XA, int NG, float* amax_out, float fixed = -1.0f){
  /* the 128 threads of the block hold columns g 128 + e of one row: its int8 row n (k_tc_q8's numbers); returns q.
     fixed >= 0: that amax instead of the row's own (the HP residual row) */
  const int e = threadIdx.x, lane = e & 31, w = e >> 5;
  __shared__ float wm[4];
  __shared__ int ws[4];
  float a = fabsf(v);
  if(fixed < 0.0f){
    for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  }
  if(lane == 0) wm[w] = fixed < 0.0f ? a : fixed;
  __syncthreads();
  const float amax = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
  const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
  int qi = __float2int_rn(v * id);
  if(fixed >= 0.0f) qi = max(-127, min(127, qi));        /* (the residual can round a hair past its bound) */
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
  *amax_out = amax;
  return qi;
}

extern "C" __global__ void k_tc_q8(const float* __restrict__ X, int ldx, unsigned int* __restrict__ XQ,
                                   float* __restrict__ XS, int* __restrict__ XA, int NG, int ntok, int hp){
  /* grid (NG, >= ntok), block 128: XQ[((n NG + g) 4 + t) 8 + 2 s + h] = 4 int8 of columns 32 s + 16 h + 4 t ..,
     XS[n NG + g] = amax / 127, XA[n NG + g] = the sum of the 128 int8 values (k_q8x_T's quantisation);
     hp: rows 2 n (hi) and 2 n + 1 (lo, the residual) */
  const int g = blockIdx.x, n = blockIdx.y;
  if(n >= ntok) return;
  const float v = X[(long long)n * ldx + g * 128 + threadIdx.x];
  float amax;
  const int qi = tcq128(v, hp ? 2 * n : n, g, XQ, XS, XA, NG, &amax);
  if(hp){
    __syncthreads();
    tcq128(v - (float)qi * (amax / 127.0f), 2 * n + 1, g, XQ, XS, XA, NG, &amax, amax / 254.0f);   /* scale amax / (254 x 127) */
  }
}

/* ------------------------------------------------------------------ the product, 1 .. 8 tokens */
template <int G>
__device__ __forceinline__ void tc_group(const unsigned int* w, const uint4 bq0, const uint4 bq1, int c[4]){
  TC_MMA(c, tc_r<16 * G + 0>(w), tc_r<16 * G + 1>(w), tc_r<16 * G + 2>(w), tc_r<16 * G + 3>(w), bq0.x, bq0.y);
  TC_MMA(c, tc_r<16 * G + 4>(w), tc_r<16 * G + 5>(w), tc_r<16 * G + 6>(w), tc_r<16 * G + 7>(w), bq0.z, bq0.w);
  TC_MMA(c, tc_r<16 * G + 8>(w), tc_r<16 * G + 9>(w), tc_r<16 * G + 10>(w), tc_r<16 * G + 11>(w), bq1.x, bq1.y);
  TC_MMA(c, tc_r<16 * G + 12>(w), tc_r<16 * G + 13>(w), tc_r<16 * G + 14>(w), tc_r<16 * G + 15>(w), bq1.z, bq1.w);
}

template <int G>
__device__ __forceinline__ void tc_unit_group(const unsigned int* w, const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                              const int* __restrict__ XA, int NG, int ntok, int gbase, int lane,
                                              float acc[4]){
  const int g8 = lane >> 2, t = lane & 3, g = gbase + G;
  uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
  if(g8 < ntok){
    const uint4* p = XQ + (((long long)g8 * NG + g) * 4 + t) * 2;
    b0 = __ldg(p);
    b1 = __ldg(p + 1);
  }
  const int n0 = 2 * t, n1 = 2 * t + 1;
  const float xs0 = n0 < ntok ? __ldg(XS + (long long)n0 * NG + g) : 0.f, xs1 = n1 < ntok ? __ldg(XS + (long long)n1 * NG + g) : 0.f;
  const int xa0 = n0 < ntok ? __ldg(XA + (long long)n0 * NG + g) : 0, xa1 = n1 < ntok ? __ldg(XA + (long long)n1 * NG + g) : 0;
  int c[4] = {0, 0, 0, 0};
  tc_group<G>(w, b0, b1, c);
  const unsigned int dw = __shfl_sync(0xffffffffu, (G & 1) ? w[27] : w[26], (lane & ~3) | (G >> 1));
  const float d0 = tc_h2f((unsigned short)(dw & 0xFFFFu)), d8 = tc_h2f((unsigned short)(dw >> 16));
  acc[0] += (float)(c[0] - xa0) * (xs0 * d0);
  acc[1] += (float)(c[1] - xa1) * (xs1 * d0);
  acc[2] += (float)(c[2] - xa0) * (xs0 * d8);
  acc[3] += (float)(c[3] - xa1) * (xs1 * d8);
}

/* The int8 activations and scales of a unit (8 groups, all tokens) are the same for every warp of a block: they are
   staged in shared memory, block-wide, one unit ahead (cp.async), so the mma chain never waits on a global load.
   [buf][group][half][token][t] uint4 (lanes read consecutive 16-byte words: no bank conflicts). */
struct TCS {
  uint4 b[2][8][2][8][4];
  float xs[2][8][8];
  int xa[2][8][8];
};

__device__ __forceinline__ unsigned int tc_smem(const void* p){
  unsigned int a;
  asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
  return a;
}
__device__ __forceinline__ void tc_cp16(void* s, const void* g){
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(tc_smem(s)), "l"(g));
}
__device__ __forceinline__ void tc_cp4(void* s, const void* g){
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(tc_smem(s)), "l"(g));
}
__device__ __forceinline__ void tc_commit(){ asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void tc_wait1(){ asm volatile("cp.async.wait_group 1;"); }
__device__ __forceinline__ void tc_wait0(){ asm volatile("cp.async.wait_group 0;"); }

__device__ __forceinline__ void tcs_load(TCS& S, int buf, const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                         const int* __restrict__ XA, int NG, int gb, int ntok){
  const int tid = threadIdx.x, nt = blockDim.x;
  for(int i = tid; i < ntok * 64; i += nt){
    const int n = i >> 6, r = i & 63, G = r >> 3, t = (r >> 1) & 3, h = r & 1;
    tc_cp16(&S.b[buf][G][h][n][t], XQ + (((long long)n * NG + gb + G) * 4 + t) * 2 + h);
  }
  for(int i = tid; i < ntok * 8; i += nt){
    const int n = i >> 3, G = i & 7;
    tc_cp4(&S.xs[buf][G][n], XS + (long long)n * NG + gb + G);
    tc_cp4(&S.xa[buf][G][n], XA + (long long)n * NG + gb + G);
  }
  tc_commit();
}

template <int G>
__device__ __forceinline__ void tc_unit_group_s(const unsigned int* w, const TCS& S, int buf, int ntok, int lane, float acc[4]){
  const int g8 = lane >> 2, t = lane & 3;
  uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
  if(g8 < ntok){ b0 = S.b[buf][G][0][g8][t]; b1 = S.b[buf][G][1][g8][t]; }
  const int n0 = 2 * t, n1 = 2 * t + 1;
  const float xs0 = n0 < ntok ? S.xs[buf][G][n0] : 0.f, xs1 = n1 < ntok ? S.xs[buf][G][n1] : 0.f;
  const int xa0 = n0 < ntok ? S.xa[buf][G][n0] : 0, xa1 = n1 < ntok ? S.xa[buf][G][n1] : 0;
  int c[4] = {0, 0, 0, 0};
  tc_group<G>(w, b0, b1, c);
  const unsigned int dw = __shfl_sync(0xffffffffu, (G & 1) ? w[27] : w[26], (lane & ~3) | (G >> 1));
  const float d0 = tc_h2f((unsigned short)(dw & 0xFFFFu)), d8 = tc_h2f((unsigned short)(dw >> 16));
  acc[0] += (float)(c[0] - xa0) * (xs0 * d0);
  acc[1] += (float)(c[1] - xa1) * (xs1 * d0);
  acc[2] += (float)(c[2] - xa0) * (xs0 * d8);
  acc[3] += (float)(c[3] - xa1) * (xs1 * d8);
}

/* the b2 layout: the four k-steps as independent u8 mma chains on the masked words, added back exactly */
template <int G>
__device__ __forceinline__ void tc2_unit_group_s(const unsigned int* w, const TCS& S, int buf, int ntok, int lane, float acc[4]){
  const int g8 = lane >> 2, t = lane & 3;
  uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
  if(g8 < ntok){ b0 = S.b[buf][G][0][g8][t]; b1 = S.b[buf][G][1][g8][t]; }
  const int n0 = 2 * t, n1 = 2 * t + 1;
  const float xs0 = n0 < ntok ? S.xs[buf][G][n0] : 0.f, xs1 = n1 < ntok ? S.xs[buf][G][n1] : 0.f;
  const int xa0 = n0 < ntok ? S.xa[buf][G][n0] : 0, xa1 = n1 < ntok ? S.xa[buf][G][n1] : 0;
  const unsigned int w0 = w[4 * G], w1 = w[4 * G + 1], w2 = w[4 * G + 2], w3 = w[4 * G + 3];
  int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0}, c2[4] = {0, 0, 0, 0}, c3[4] = {0, 0, 0, 0};
  TC_MMAU(c0, w0 & 0x03030303u, w1 & 0x03030303u, w2 & 0x03030303u, w3 & 0x03030303u, b0.x, b0.y);
  TC_MMAU(c1, w0 & 0x0C0C0C0Cu, w1 & 0x0C0C0C0Cu, w2 & 0x0C0C0C0Cu, w3 & 0x0C0C0C0Cu, b0.z, b0.w);
  TC_MMAU(c2, w0 & 0x30303030u, w1 & 0x30303030u, w2 & 0x30303030u, w3 & 0x30303030u, b1.x, b1.y);
  TC_MMAU(c3, w0 & 0xC0C0C0C0u, w1 & 0xC0C0C0C0u, w2 & 0xC0C0C0C0u, w3 & 0xC0C0C0C0u, b1.z, b1.w);
  int c[4];
#pragma unroll
  for(int k = 0; k < 4; ++k) c[k] = c0[k] + (c1[k] >> 2) + (c2[k] >> 4) + (c3[k] >> 6);
  const unsigned int dw = __shfl_sync(0xffffffffu, (G & 1) ? w[33] : w[32], (lane & ~3) | (G >> 1));
  const float d0 = tc_h2f((unsigned short)(dw & 0xFFFFu)), d8 = tc_h2f((unsigned short)(dw >> 16));
  acc[0] += (float)(c[0] - xa0) * (xs0 * d0);
  acc[1] += (float)(c[1] - xa1) * (xs1 * d0);
  acc[2] += (float)(c[2] - xa0) * (xs0 * d8);
  acc[3] += (float)(c[3] - xa1) * (xs1 * d8);
}

/* one 16-row tile per warp (active: the warp has a tile), units [u_lo, u_hi); every warp of the block takes part in
   the staging and the barriers */
template <bool B2>
__device__ __forceinline__ void tc_tile(const uint4* __restrict__ Wm, bool active, int u_lo, int u_hi, int NG,
                                        const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                        const int* __restrict__ XA, int ntok, float acc[4]){
  __shared__ TCS S;
  const int lane = threadIdx.x & 31;
  const int NV = B2 ? 9 : 7, US = NV * 32;       /* uint4 per lane and unit; uint4 per tile and unit */
  uint4 nx[NV];
  if(u_lo < u_hi){
    tcs_load(S, 0, XQ, XS, XA, NG, u_lo * 8, ntok);
    if(active){
#pragma unroll
      for(int i = 0; i < NV; ++i) if(B2 || tc_row_used(i, NG - u_lo * 8)) nx[i] = __ldg(Wm + (long long)u_lo * US + i * 32);
    }
  }
  for(int u = u_lo; u < u_hi; ++u){
    const int buf = (u - u_lo) & 1;
    if(u + 1 < u_hi){
      tcs_load(S, buf ^ 1, XQ, XS, XA, NG, (u + 1) * 8, ntok);
      tc_wait1();
    } else {
      tc_wait0();
    }
    __syncthreads();
    if(active){
      unsigned int w[4 * NV];
#pragma unroll
      for(int i = 0; i < NV; ++i){ w[4 * i] = nx[i].x; w[4 * i + 1] = nx[i].y; w[4 * i + 2] = nx[i].z; w[4 * i + 3] = nx[i].w; }
      if(u + 1 < u_hi){
#pragma unroll
        for(int i = 0; i < NV; ++i) if(B2 || tc_row_used(i, NG - (u + 1) * 8)) nx[i] = __ldg(Wm + (long long)(u + 1) * US + i * 32);
      }
      if(B2){
        tc2_unit_group_s<0>(w, S, buf, ntok, lane, acc);
        tc2_unit_group_s<1>(w, S, buf, ntok, lane, acc);
        tc2_unit_group_s<2>(w, S, buf, ntok, lane, acc);
        tc2_unit_group_s<3>(w, S, buf, ntok, lane, acc);
        if(u * 8 + 4 < NG){
          tc2_unit_group_s<4>(w, S, buf, ntok, lane, acc);
          tc2_unit_group_s<5>(w, S, buf, ntok, lane, acc);
          tc2_unit_group_s<6>(w, S, buf, ntok, lane, acc);
          tc2_unit_group_s<7>(w, S, buf, ntok, lane, acc);
        }
      } else {
        tc_unit_group_s<0>(w, S, buf, ntok, lane, acc);
        tc_unit_group_s<1>(w, S, buf, ntok, lane, acc);
        tc_unit_group_s<2>(w, S, buf, ntok, lane, acc);
        tc_unit_group_s<3>(w, S, buf, ntok, lane, acc);
        if(u * 8 + 4 < NG){
          tc_unit_group_s<4>(w, S, buf, ntok, lane, acc);
          tc_unit_group_s<5>(w, S, buf, ntok, lane, acc);
          tc_unit_group_s<6>(w, S, buf, ntok, lane, acc);
          tc_unit_group_s<7>(w, S, buf, ntok, lane, acc);
        }
      }
    }
    __syncthreads();
  }
}

template <int NW, bool B2>
__device__ __forceinline__ void tc_mv(const uint4* __restrict__ W, float* __restrict__ y, int R, int NUG,
                                      const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                      const int* __restrict__ XA, int ntok){
  /* grid (ceil(R / 16 / NW), splits), NW warps of one 16-row tile each; block y covers units
     [by NU / splits, (by + 1) NU / splits) and writes y + (n splits + by) R */
  const int lane = threadIdx.x & 31, m = blockIdx.x * NW + (threadIdx.x >> 5), NT16 = R >> 4, NU = tc_nu(NUG), NG = tc_ng(NUG);
  const bool active = m < NT16;
  const int u_lo = (int)((long long)blockIdx.y * NU / gridDim.y), u_hi = (int)((long long)(blockIdx.y + 1) * NU / gridDim.y);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  tc_tile<B2>(W + (long long)(active ? m : 0) * NU * (B2 ? 288 : 224) + lane, active, u_lo, u_hi, NG, XQ, XS, XA, ntok, acc);
  if(!active) return;
  const int g8 = lane >> 2, t = lane & 3, r0 = 16 * m + g8;
  /* token n's part p at y + (n splits + p) R: the parts of one token are R apart (k_rmsfold can add them itself) */
  const int sp = gridDim.y, by = blockIdx.y;
  if(tc_hp(NUG)){                 /* columns 2t, 2t + 1 = token t's hi and lo rows */
    if(2 * t < ntok){ float* yb = y + ((long long)t * sp + by) * R; yb[r0] = acc[0] + acc[1]; yb[r0 + 8] = acc[2] + acc[3]; }
    return;
  }
  if(2 * t < ntok){ float* yb = y + ((long long)(2 * t) * sp + by) * R; yb[r0] = acc[0]; yb[r0 + 8] = acc[2]; }
  if(2 * t + 1 < ntok){ float* yb = y + ((long long)(2 * t + 1) * sp + by) * R; yb[r0] = acc[1]; yb[r0 + 8] = acc[3]; }
}

/* k_tc_mv3: up to three products of the same input in one launch (no column splits): the blocks along x cover the
   16-row tiles of W0, then W1, then W2 (R1 = 0 / R2 = 0: fewer tensors). Each tile's work is that of tc_mv. */
template <int NW, bool B2>
__device__ __forceinline__ void tc_mv3(const uint4* __restrict__ W0, float* __restrict__ y0, int R0,
                                       const uint4* __restrict__ W1, float* __restrict__ y1, int R1,
                                       const uint4* __restrict__ W2, float* __restrict__ y2, int R2, int NUG,
                                       const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                       const int* __restrict__ XA, int ntok){
  const int lane = threadIdx.x & 31, NU = tc_nu(NUG), NG = tc_ng(NUG);
  int m = blockIdx.x * NW + (threadIdx.x >> 5);
  const uint4* W = W0;
  float* y = y0;
  int R = R0;
  if(m >= (R0 >> 4)){
    m -= R0 >> 4; W = W1; y = y1; R = R1;
    if(m >= (R1 >> 4)){ m -= R1 >> 4; W = W2; y = y2; R = R2; }
  }
  const bool active = m < (R >> 4);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  tc_tile<B2>(W + (long long)(active ? m : 0) * NU * (B2 ? 288 : 224) + lane, active, 0, NU, NG, XQ, XS, XA, ntok, acc);
  if(!active) return;
  const int g8 = lane >> 2, t = lane & 3, r0 = 16 * m + g8;
  if(tc_hp(NUG)){
    if(2 * t < ntok){ float* yb = y + (long long)t * R; yb[r0] = acc[0] + acc[1]; yb[r0 + 8] = acc[2] + acc[3]; }
    return;
  }
  if(2 * t < ntok){ float* yb = y + (long long)(2 * t) * R; yb[r0] = acc[0]; yb[r0 + 8] = acc[2]; }
  if(2 * t + 1 < ntok){ float* yb = y + (long long)(2 * t + 1) * R; yb[r0] = acc[1]; yb[r0 + 8] = acc[3]; }
}

#define TC_MV3_ARGS const uint4* __restrict__ W0, float* __restrict__ y0, int R0, const uint4* __restrict__ W1, \
  float* __restrict__ y1, int R1, const uint4* __restrict__ W2, float* __restrict__ y2, int R2, int NU, \
  const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok
extern "C" __global__ void __launch_bounds__(64) k_tc_mv3_2(TC_MV3_ARGS){ tc_mv3<2, false>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv3_4(TC_MV3_ARGS){ tc_mv3<4, false>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv3_8(TC_MV3_ARGS){ tc_mv3<8, false>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }

extern "C" __global__ void __launch_bounds__(64) k_tc_mv2(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<2, false>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv4(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<4, false>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv8(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<8, false>(W, y, R, NU, XQ, XS, XA, ntok); }

/* the same products on the b2 layout */
extern "C" __global__ void __launch_bounds__(64) k_tc_mv3b_2(TC_MV3_ARGS){ tc_mv3<2, true>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv3b_4(TC_MV3_ARGS){ tc_mv3<4, true>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv3b_8(TC_MV3_ARGS){ tc_mv3<8, true>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(64) k_tc_mv2b(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<2, true>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv4b(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<4, true>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv8b(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<8, true>(W, y, R, NU, XQ, XS, XA, ntok); }

/* ------------------------------------------------------------------ prompt chunks: up to 256 tokens */
/* k_tc_mm: grid (ceil(R / 16 / NW), ceil(T / (8 NB))), NW warps of one 16-row tile each; a warp unpacks each group's
   A fragments once and runs them against NB blocks of 8 tokens (all NG groups in order, no splits). Y[n ldy + r]. */
template <int NB>
__device__ __forceinline__ void tcm_group_scale(int c[NB][4], float acc[NB][4], const float* __restrict__ XS,
                                                const int* __restrict__ XA, int NG, int g, int n0, int T, int t,
                                                float d0, float d8){
#pragma unroll
  for(int b = 0; b < NB; ++b){
    const int na = n0 + 8 * b + 2 * t, nb = na + 1;
    const float xs0 = na < T ? __ldg(XS + (long long)na * NG + g) : 0.f, xs1 = nb < T ? __ldg(XS + (long long)nb * NG + g) : 0.f;
    const int xa0 = na < T ? __ldg(XA + (long long)na * NG + g) : 0, xa1 = nb < T ? __ldg(XA + (long long)nb * NG + g) : 0;
    acc[b][0] += (float)(c[b][0] - xa0) * (xs0 * d0);
    acc[b][1] += (float)(c[b][1] - xa1) * (xs1 * d0);
    acc[b][2] += (float)(c[b][2] - xa0) * (xs0 * d8);
    acc[b][3] += (float)(c[b][3] - xa1) * (xs1 * d8);
  }
}

template <int G, int NB>
__device__ __forceinline__ void tcm_unit_group(const unsigned int* w, const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                               const int* __restrict__ XA, int NG, int T, int n0, int gbase, int lane,
                                               float acc[NB][4]){
  const int g8 = lane >> 2, t = lane & 3, g = gbase + G;
  const unsigned int a[16] = {tc_r<16 * G + 0>(w), tc_r<16 * G + 1>(w), tc_r<16 * G + 2>(w), tc_r<16 * G + 3>(w),
                              tc_r<16 * G + 4>(w), tc_r<16 * G + 5>(w), tc_r<16 * G + 6>(w), tc_r<16 * G + 7>(w),
                              tc_r<16 * G + 8>(w), tc_r<16 * G + 9>(w), tc_r<16 * G + 10>(w), tc_r<16 * G + 11>(w),
                              tc_r<16 * G + 12>(w), tc_r<16 * G + 13>(w), tc_r<16 * G + 14>(w), tc_r<16 * G + 15>(w)};
  int c[NB][4];
#pragma unroll
  for(int b = 0; b < NB; ++b){
    c[b][0] = c[b][1] = c[b][2] = c[b][3] = 0;
    const int n = n0 + 8 * b + g8;
    uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
    if(n < T){
      const uint4* p = XQ + (((long long)n * NG + g) * 4 + t) * 2;
      b0 = __ldg(p);
      b1 = __ldg(p + 1);
    }
    TC_MMA(c[b], a[0], a[1], a[2], a[3], b0.x, b0.y);
    TC_MMA(c[b], a[4], a[5], a[6], a[7], b0.z, b0.w);
    TC_MMA(c[b], a[8], a[9], a[10], a[11], b1.x, b1.y);
    TC_MMA(c[b], a[12], a[13], a[14], a[15], b1.z, b1.w);
  }
  const unsigned int dw = __shfl_sync(0xffffffffu, (G & 1) ? w[27] : w[26], (lane & ~3) | (G >> 1));
  tcm_group_scale<NB>(c, acc, XS, XA, NG, g, n0, T, t, tc_h2f((unsigned short)(dw & 0xFFFFu)), tc_h2f((unsigned short)(dw >> 16)));
}

template <int NW, int NB>
__device__ __forceinline__ void tc_mm(const uint4* __restrict__ W, int R, int NUG, const uint4* __restrict__ XQ,
                                      const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y,
                                      int ldy, int T){
  const int lane = threadIdx.x & 31, m = blockIdx.x * NW + (threadIdx.x >> 5), NT16 = R >> 4, NU = tc_nu(NUG), NG = tc_ng(NUG);
  const int n0 = blockIdx.y * 8 * NB;
  if(m >= NT16 || n0 >= T) return;
  float acc[NB][4];
#pragma unroll
  for(int b = 0; b < NB; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const uint4* Wm = W + (long long)m * NU * 224 + lane;
  uint4 nx[7];
#pragma unroll
  for(int i = 0; i < 7; ++i) if(tc_row_used(i, NG)) nx[i] = __ldg(Wm + i * 32);
  for(int u = 0; u < NU; ++u){
    unsigned int w[28];
#pragma unroll
    for(int i = 0; i < 7; ++i){ w[4 * i] = nx[i].x; w[4 * i + 1] = nx[i].y; w[4 * i + 2] = nx[i].z; w[4 * i + 3] = nx[i].w; }
    if(u + 1 < NU){
#pragma unroll
      for(int i = 0; i < 7; ++i) if(tc_row_used(i, NG - (u + 1) * 8)) nx[i] = __ldg(Wm + (long long)(u + 1) * 224 + i * 32);
    }
    const int gb = u * 8;
    tcm_unit_group<0, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<1, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<2, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<3, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    if(gb + 4 < NG){
      tcm_unit_group<4, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
      tcm_unit_group<5, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
      tcm_unit_group<6, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
      tcm_unit_group<7, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    }
  }
  const int g8 = lane >> 2, t = lane & 3, r0 = 16 * m + g8;
#pragma unroll
  for(int b = 0; b < NB; ++b){
    const int na = n0 + 8 * b + 2 * t, nb = na + 1;
    if(na < T){ Y[(long long)na * ldy + r0] = acc[b][0]; Y[(long long)na * ldy + r0 + 8] = acc[b][2]; }
    if(nb < T){ Y[(long long)nb * ldy + r0] = acc[b][1]; Y[(long long)nb * ldy + r0 + 8] = acc[b][3]; }
  }
}

extern "C" __global__ void __launch_bounds__(128) k_tc_mm4(const uint4* __restrict__ W, int R, int NU, const uint4* __restrict__ XQ,
    const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y, int ldy, int T){ tc_mm<4, 4>(W, R, NU, XQ, XS, XA, Y, ldy, T); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mm8(const uint4* __restrict__ W, int R, int NU, const uint4* __restrict__ XQ,
    const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y, int ldy, int T){ tc_mm<4, 8>(W, R, NU, XQ, XS, XA, Y, ldy, T); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mm16(const uint4* __restrict__ W, int R, int NU, const uint4* __restrict__ XQ,
    const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y, int ldy, int T){ tc_mm<4, 16>(W, R, NU, XQ, XS, XA, Y, ldy, T); }

/* parts of a split product, ntok tokens: y[n R + r] = sum over the sp parts (in order) of P[(n sp + p) R + r] */
extern "C" __global__ void k_tc_sum(const float* __restrict__ P, float* __restrict__ y, int R, int ntok, int sp){
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)R * ntok) return;
  const long long n = i / R, r = i - n * R;
  const float* q = P + n * sp * R + r;
  float s = q[0];
  for(int p = 1; p < sp; ++p) s += q[(long long)p * R];
  y[i] = s;
}

/* ------------------------------------------------------------------ prompt chunks, shared-memory version */
/* k_tc_mm2: grid (ceil(R / 128), ceil(T / 64)), 256 threads = 8 warps of one 16-row tile each; the block's 64 tokens'
   int8 activations and scales are staged per group in shared memory (cp.async, double-buffered), so each warp reads
   its B fragments with conflict-free LDS. Same arithmetic, in the same order, as k_tc_mm. */
struct TCM {
  uint4 b[2][2][64][4];
  float xs[2][64];
  int xa[2][64];
};

__device__ __forceinline__ void tcm_stage(TCM& S, int buf, const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                          const int* __restrict__ XA, int NG, int g, int n0, int T){
  const int tid = threadIdx.x;
#pragma unroll
  for(int j = 0; j < 2; ++j){
    const int i = tid + 256 * j, n = i >> 3, t = (i >> 1) & 3, h = i & 1;
    if(n0 + n < T) tc_cp16(&S.b[buf][h][n][t], XQ + (((long long)(n0 + n) * NG + g) * 4 + t) * 2 + h);
  }
  if(tid < 64){ if(n0 + tid < T) tc_cp4(&S.xs[buf][tid], XS + (long long)(n0 + tid) * NG + g); }
  else if(tid < 128){ const int n = tid - 64; if(n0 + n < T) tc_cp4(&S.xa[buf][n], XA + (long long)(n0 + n) * NG + g); }
  tc_commit();
}

template <int G>
__device__ __forceinline__ void tcm2_group(const unsigned int* w, const TCM& S, int buf, int lane, float acc[8][4]){
  const int g8 = lane >> 2, t = lane & 3;
  const unsigned int a[16] = {tc_r<16 * G + 0>(w), tc_r<16 * G + 1>(w), tc_r<16 * G + 2>(w), tc_r<16 * G + 3>(w),
                              tc_r<16 * G + 4>(w), tc_r<16 * G + 5>(w), tc_r<16 * G + 6>(w), tc_r<16 * G + 7>(w),
                              tc_r<16 * G + 8>(w), tc_r<16 * G + 9>(w), tc_r<16 * G + 10>(w), tc_r<16 * G + 11>(w),
                              tc_r<16 * G + 12>(w), tc_r<16 * G + 13>(w), tc_r<16 * G + 14>(w), tc_r<16 * G + 15>(w)};
  const unsigned int dw = __shfl_sync(0xffffffffu, (G & 1) ? w[27] : w[26], (lane & ~3) | (G >> 1));
  const float d0 = tc_h2f((unsigned short)(dw & 0xFFFFu)), d8 = tc_h2f((unsigned short)(dw >> 16));
#pragma unroll
  for(int nb = 0; nb < 8; ++nb){
    const uint4 b0 = S.b[buf][0][nb * 8 + g8][t], b1 = S.b[buf][1][nb * 8 + g8][t];
    int c[4] = {0, 0, 0, 0};
    TC_MMA(c, a[0], a[1], a[2], a[3], b0.x, b0.y);
    TC_MMA(c, a[4], a[5], a[6], a[7], b0.z, b0.w);
    TC_MMA(c, a[8], a[9], a[10], a[11], b1.x, b1.y);
    TC_MMA(c, a[12], a[13], a[14], a[15], b1.z, b1.w);
    const int na = nb * 8 + 2 * t;
    const float xs0 = S.xs[buf][na], xs1 = S.xs[buf][na + 1];
    const int xa0 = S.xa[buf][na], xa1 = S.xa[buf][na + 1];
    acc[nb][0] += (float)(c[0] - xa0) * (xs0 * d0);
    acc[nb][1] += (float)(c[1] - xa1) * (xs1 * d0);
    acc[nb][2] += (float)(c[2] - xa0) * (xs0 * d8);
    acc[nb][3] += (float)(c[3] - xa1) * (xs1 * d8);
  }
}

#define TCM2_STEP(G) { \
    const int gi = gb + G; \
    if(gi + 1 < NG){ tcm_stage(S, (gi + 1) & 1, XQ, XS, XA, NG, gi + 1, n0, T); tc_wait1(); } else tc_wait0(); \
    __syncthreads(); \
    if(active) tcm2_group<G>(w, S, gi & 1, lane, acc); \
    __syncthreads(); }

extern "C" __global__ void __launch_bounds__(256, 2) k_tc_mm2(const uint4* __restrict__ W, int R, int NUG,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y,
    int ldy, int T){
  __shared__ TCM S;
  const int lane = threadIdx.x & 31, m = blockIdx.x * 8 + (threadIdx.x >> 5), NT16 = R >> 4, NU = tc_nu(NUG), NG = tc_ng(NUG);
  const int n0 = blockIdx.y * 64;
  const bool active = m < NT16;
  float acc[8][4];
#pragma unroll
  for(int b = 0; b < 8; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const uint4* Wm = W + (long long)(active ? m : 0) * NU * 224 + lane;
  uint4 nx[7];
  if(active){
#pragma unroll
    for(int i = 0; i < 7; ++i) if(tc_row_used(i, NG)) nx[i] = __ldg(Wm + i * 32);
  }
  tcm_stage(S, 0, XQ, XS, XA, NG, 0, n0, T);
  for(int u = 0; u < NU; ++u){
    unsigned int w[28];
#pragma unroll
    for(int i = 0; i < 7; ++i){ w[4 * i] = nx[i].x; w[4 * i + 1] = nx[i].y; w[4 * i + 2] = nx[i].z; w[4 * i + 3] = nx[i].w; }
    if(active && u + 1 < NU){
#pragma unroll
      for(int i = 0; i < 7; ++i) if(tc_row_used(i, NG - (u + 1) * 8)) nx[i] = __ldg(Wm + (long long)(u + 1) * 224 + i * 32);
    }
    const int gb = u * 8;
    TCM2_STEP(0) TCM2_STEP(1) TCM2_STEP(2) TCM2_STEP(3)
    if(gb + 4 < NG){ TCM2_STEP(4) TCM2_STEP(5) TCM2_STEP(6) TCM2_STEP(7) }
  }
  if(!active) return;
  const int g8 = lane >> 2, t = lane & 3, r0 = 16 * m + g8;
#pragma unroll
  for(int b = 0; b < 8; ++b){
    const int na = n0 + 8 * b + 2 * t, nb = na + 1;
    if(na < T){ Y[(long long)na * ldy + r0] = acc[b][0]; Y[(long long)na * ldy + r0 + 8] = acc[b][2]; }
    if(nb < T){ Y[(long long)nb * ldy + r0] = acc[b][1]; Y[(long long)nb * ldy + r0 + 8] = acc[b][3]; }
  }
}
