/* tc_turing.cu - int8 tensor-core matrix products for Turing (sm_75, KINDI_TC=1) with 1 .. 8 tokens.

   Same jobs, signatures, weight layout and numerics as tc.cu, but the MMA is Turing's int8 IMMA
   mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 (A: one .b32, B: one .b32, C/D: two .s32) instead of Ampere's
   m16n8k32. A warp still owns one 16-row tile and the same unit of 8 column groups, so every grid, buffer size and
   Python plan is unchanged.

   The "tc" weight layout is also unchanged: register rho of a lane = group rho >> 4, Ampere k-step (rho >> 2) & 3,
   Ampere fragment rho & 3. A Turing A register for k-block c (16 columns) and the 8-row half rowsel is exactly
   Ampere register 16 G + 2 c + rowsel, because Ampere fragment (s, f) covers rows g8 + 8 (f & 1) and columns
   32 s + 16 (f >> 1) + 4 t + byte = 16 (2 s + (f >> 1)) + 4 t + byte: c = 2 s + (f >> 1). The Turing B word of
   k-block c is the Ampere activation word of k-step s = c >> 1, half h = c & 1 (k = 16 c + 4 t), so the staged
   uint4 pair is used in order c = 0 .. 7 = b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w. The D pair (c0, c1) is
   row g8 + 8 rowsel, tokens 2 t and 2 t + 1, so the four results land in the same acc[0 .. 3] as the Ampere path.

   Per group this is 8 x 2 MMAs instead of 4, but every MMA still produces an exact integer sum of the same 16
   (then 128) products, scaled and added in fp32 in the same order, so a token's result is bit-identical to the
   Ampere and 5-trit paths.

   Turing has no cp.async (PTX ISA 7.0, sm_80+), so the shared-memory staging below is synchronous: plain 16/4-byte
   loads and stores, ordered by the same __syncthreads(). The double-buffered structure, the shared-memory sizes and
   the arithmetic are those of tc.cu. */

__device__ __forceinline__ float tc_h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

#define TC_MMA_T(c, a, b) \
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" \
      : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b))

__device__ __forceinline__ unsigned int tc_mask(unsigned int src, unsigned int p){ return 3u * ((src * p) & 0x00FF00FFu); }

/* register i (0 .. 4) of word w */
template <int I>
__device__ __forceinline__ unsigned int tc_reg(unsigned int w){
  const unsigned int lo = w & 0x00FF00FFu, hi = (w >> 8) & 0x00FF00FFu;
  const unsigned int P[5] = {1u, 3u, 9u, 27u, 81u};
  const int q0 = 2 * I, q1 = 2 * I + 1;
  const unsigned int x = q0 < 5 ? (q0 == 0 ? 3u * lo : tc_mask(lo, P[q0])) : (q0 == 5 ? 3u * hi : tc_mask(hi, P[q0 - 5]));
  const unsigned int y = q1 < 5 ? tc_mask(lo, P[q1]) : (q1 == 5 ? 3u * hi : tc_mask(hi, P[q1 - 5]));
  return __byte_perm(x, y, 0x7531);
}

template <int RHO>
__device__ __forceinline__ unsigned int tc_r(const unsigned int* w){ return tc_reg<RHO % 5>(w[RHO / 5]); }

/* one Turing MMA: A = the 8-row half's four int8 of k-block c, B = the activation word (k = 16 c + 4 t, col g8) */
__device__ __forceinline__ void tc_mma_t(int c[2], unsigned int a, unsigned int b){ TC_MMA_T(c, a, b); }

/* k-block C of group G: the two 8-row halves accumulate into lo (rows g8) and hi (rows g8 + 8) */
template <int G, int C>
__device__ __forceinline__ void tc_kb(const unsigned int* w, unsigned int b, int lo[2], int hi[2]){
  tc_mma_t(lo, tc_r<16 * G + 2 * C + 0>(w), b);
  tc_mma_t(hi, tc_r<16 * G + 2 * C + 1>(w), b);
}

/* ------------------------------------------------------------------ repack: 5-trit lane layout -> tc layout (same bytes as Ampere) */
__device__ __forceinline__ unsigned int tc_src_trit(const unsigned char* __restrict__ S, int NT16, int r, int G, int c){
  int p, e;
  if(c < 120){ p = c / 5; e = c - 5 * p; } else { p = 24 + (c - 120) / 4; e = (c - 120) & 3; }
  unsigned int x = S[(((long long)G * NT16 + (r >> 4)) * 28 + p) * 16 + (r & 15)];
  for(int i = 0; i < e; ++i) x /= 3u;
  return x % 3u;
}

extern "C" __global__ void k_tc_pack(const unsigned char* __restrict__ S, unsigned char* __restrict__ O, int R, int NG){
  /* one thread per output byte; O = tiles [m][u][i 0..6][lane][16 bytes] */
  const int NT16 = R >> 4, NU = NG >> 3;
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
    out = S[(((long long)G * NT16 + m) * 28 + 26 + (bq & 1)) * 16 + (r & 15)];
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
        d = tc_src_trit(S, NT16, r, u * 8 + Gp, c);
      }
      v = v * 3u + d;
    }
    out = (v * 256u + 242u) / 243u;
  }
  O[idx] = (unsigned char)out;
}

/* ------------------------------------------------------------------ activations: int8 per 128-column group (same layout as Ampere) */
extern "C" __global__ void k_tc_q8(const float* __restrict__ X, int ldx, unsigned int* __restrict__ XQ,
                                   float* __restrict__ XS, int* __restrict__ XA, int NG, int ntok){
  /* grid (NG, >= ntok), block 128: XQ[((n NG + g) 4 + t) 8 + 2 s + h] = 4 int8 of columns 32 s + 16 h + 4 t ..,
     XS[n NG + g] = amax / 127, XA[n NG + g] = the sum of the 128 int8 values */
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

/* ------------------------------------------------------------------ shared-memory staging without cp.async (sm_75) */
__device__ __forceinline__ void tc_cp16(void* s, const void* g){ *(uint4*)s = __ldg((const uint4*)g); }
__device__ __forceinline__ void tc_cp4(void* s, const void* g){ *(unsigned int*)s = __ldg((const unsigned int*)g); }
__device__ __forceinline__ void tc_commit(){}
__device__ __forceinline__ void tc_wait1(){}
__device__ __forceinline__ void tc_wait0(){}

/* ------------------------------------------------------------------ the product, 1 .. 8 tokens */
template <int G>
__device__ __forceinline__ void tc_group(const unsigned int* w, const uint4 bq0, const uint4 bq1, int c[4]){
  int lo[2] = {0, 0}, hi[2] = {0, 0};
  tc_kb<G, 0>(w, bq0.x, lo, hi);
  tc_kb<G, 1>(w, bq0.y, lo, hi);
  tc_kb<G, 2>(w, bq0.z, lo, hi);
  tc_kb<G, 3>(w, bq0.w, lo, hi);
  tc_kb<G, 4>(w, bq1.x, lo, hi);
  tc_kb<G, 5>(w, bq1.y, lo, hi);
  tc_kb<G, 6>(w, bq1.z, lo, hi);
  tc_kb<G, 7>(w, bq1.w, lo, hi);
  c[0] = lo[0]; c[1] = lo[1]; c[2] = hi[0]; c[3] = hi[1];
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
   staged in shared memory, block-wide, one unit ahead, so the mma chain reads them from shared memory instead of
   global memory. [buf][group][half][token][t] uint4 (lanes read consecutive 16-byte words: no bank conflicts). */
struct TCS {
  uint4 b[2][8][2][8][4];
  float xs[2][8][8];
  int xa[2][8][8];
};

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

/* one 16-row tile per warp (active: the warp has a tile), units [u_lo, u_hi); every warp of the block takes part in
   the staging and the barriers */
__device__ __forceinline__ void tc_tile(const uint4* __restrict__ Wm, bool active, int u_lo, int u_hi, int NG,
                                        const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                        const int* __restrict__ XA, int ntok, float acc[4]){
  __shared__ TCS S;
  const int lane = threadIdx.x & 31;
  uint4 nx[7];
  if(u_lo < u_hi){
    tcs_load(S, 0, XQ, XS, XA, NG, u_lo * 8, ntok);
    if(active){
#pragma unroll
      for(int i = 0; i < 7; ++i) nx[i] = __ldg(Wm + (long long)u_lo * 224 + i * 32);
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
      unsigned int w[28];
#pragma unroll
      for(int i = 0; i < 7; ++i){ w[4 * i] = nx[i].x; w[4 * i + 1] = nx[i].y; w[4 * i + 2] = nx[i].z; w[4 * i + 3] = nx[i].w; }
      if(u + 1 < u_hi){
#pragma unroll
        for(int i = 0; i < 7; ++i) nx[i] = __ldg(Wm + (long long)(u + 1) * 224 + i * 32);
      }
      tc_unit_group_s<0>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<1>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<2>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<3>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<4>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<5>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<6>(w, S, buf, ntok, lane, acc);
      tc_unit_group_s<7>(w, S, buf, ntok, lane, acc);
    }
    __syncthreads();
  }
}

template <int NW>
__device__ __forceinline__ void tc_mv(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
                                      const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                      const int* __restrict__ XA, int ntok){
  /* grid (ceil(R / 16 / NW), splits), NW warps of one 16-row tile each; block y covers units
     [by NU / splits, (by + 1) NU / splits) and writes y + (n splits + by) R */
  const int lane = threadIdx.x & 31, m = blockIdx.x * NW + (threadIdx.x >> 5), NT16 = R >> 4, NG = NU * 8;
  const bool active = m < NT16;
  const int u_lo = (int)((long long)blockIdx.y * NU / gridDim.y), u_hi = (int)((long long)(blockIdx.y + 1) * NU / gridDim.y);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  tc_tile(W + (long long)(active ? m : 0) * NU * 224 + lane, active, u_lo, u_hi, NG, XQ, XS, XA, ntok, acc);
  if(!active) return;
  const int g8 = lane >> 2, t = lane & 3, r0 = 16 * m + g8;
  /* token n's part p at y + (n splits + p) R: the parts of one token are R apart (k_rmsfold can add them itself) */
  const int sp = gridDim.y, by = blockIdx.y;
  if(2 * t < ntok){ float* yb = y + ((long long)(2 * t) * sp + by) * R; yb[r0] = acc[0]; yb[r0 + 8] = acc[2]; }
  if(2 * t + 1 < ntok){ float* yb = y + ((long long)(2 * t + 1) * sp + by) * R; yb[r0] = acc[1]; yb[r0 + 8] = acc[3]; }
}

/* k_tc_mv3: up to three products of the same input in one launch (no column splits): the blocks along x cover the
   16-row tiles of W0, then W1, then W2 (R1 = 0 / R2 = 0: fewer tensors). Each tile's work is that of tc_mv. */
template <int NW>
__device__ __forceinline__ void tc_mv3(const uint4* __restrict__ W0, float* __restrict__ y0, int R0,
                                       const uint4* __restrict__ W1, float* __restrict__ y1, int R1,
                                       const uint4* __restrict__ W2, float* __restrict__ y2, int R2, int NU,
                                       const uint4* __restrict__ XQ, const float* __restrict__ XS,
                                       const int* __restrict__ XA, int ntok){
  const int lane = threadIdx.x & 31, NG = NU * 8;
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
  tc_tile(W + (long long)(active ? m : 0) * NU * 224 + lane, active, 0, NU, NG, XQ, XS, XA, ntok, acc);
  if(!active) return;
  const int g8 = lane >> 2, t = lane & 3, r0 = 16 * m + g8;
  if(2 * t < ntok){ float* yb = y + (long long)(2 * t) * R; yb[r0] = acc[0]; yb[r0 + 8] = acc[2]; }
  if(2 * t + 1 < ntok){ float* yb = y + (long long)(2 * t + 1) * R; yb[r0] = acc[1]; yb[r0 + 8] = acc[3]; }
}

#define TC_MV3_ARGS const uint4* __restrict__ W0, float* __restrict__ y0, int R0, const uint4* __restrict__ W1, \
  float* __restrict__ y1, int R1, const uint4* __restrict__ W2, float* __restrict__ y2, int R2, int NU, \
  const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok
extern "C" __global__ void __launch_bounds__(64) k_tc_mv3_2(TC_MV3_ARGS){ tc_mv3<2>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv3_4(TC_MV3_ARGS){ tc_mv3<4>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv3_8(TC_MV3_ARGS){ tc_mv3<8>(W0, y0, R0, W1, y1, R1, W2, y2, R2, NU, XQ, XS, XA, ntok); }

extern "C" __global__ void __launch_bounds__(64) k_tc_mv2(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<2>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mv4(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<4>(W, y, R, NU, XQ, XS, XA, ntok); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mv8(const uint4* __restrict__ W, float* __restrict__ y, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, int ntok){ tc_mv<8>(W, y, R, NU, XQ, XS, XA, ntok); }

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

/* 16 A registers (k-blocks c, halves rowsel) of group G against one block of 8 tokens */
__device__ __forceinline__ void tcm_mma(const unsigned int a[16], const uint4 b0, const uint4 b1, int c[4]){
  int lo[2] = {0, 0}, hi[2] = {0, 0};
  tc_mma_t(lo, a[0], b0.x); tc_mma_t(hi, a[1], b0.x);
  tc_mma_t(lo, a[2], b0.y); tc_mma_t(hi, a[3], b0.y);
  tc_mma_t(lo, a[4], b0.z); tc_mma_t(hi, a[5], b0.z);
  tc_mma_t(lo, a[6], b0.w); tc_mma_t(hi, a[7], b0.w);
  tc_mma_t(lo, a[8], b1.x); tc_mma_t(hi, a[9], b1.x);
  tc_mma_t(lo, a[10], b1.y); tc_mma_t(hi, a[11], b1.y);
  tc_mma_t(lo, a[12], b1.z); tc_mma_t(hi, a[13], b1.z);
  tc_mma_t(lo, a[14], b1.w); tc_mma_t(hi, a[15], b1.w);
  c[0] = lo[0]; c[1] = lo[1]; c[2] = hi[0]; c[3] = hi[1];
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
    const int n = n0 + 8 * b + g8;
    uint4 b0 = make_uint4(0u, 0u, 0u, 0u), b1 = b0;
    if(n < T){
      const uint4* p = XQ + (((long long)n * NG + g) * 4 + t) * 2;
      b0 = __ldg(p);
      b1 = __ldg(p + 1);
    }
    tcm_mma(a, b0, b1, c[b]);
  }
  const unsigned int dw = __shfl_sync(0xffffffffu, (G & 1) ? w[27] : w[26], (lane & ~3) | (G >> 1));
  tcm_group_scale<NB>(c, acc, XS, XA, NG, g, n0, T, t, tc_h2f((unsigned short)(dw & 0xFFFFu)), tc_h2f((unsigned short)(dw >> 16)));
}

template <int NW, int NB>
__device__ __forceinline__ void tc_mm(const uint4* __restrict__ W, int R, int NU, const uint4* __restrict__ XQ,
                                      const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y,
                                      int ldy, int T){
  const int lane = threadIdx.x & 31, m = blockIdx.x * NW + (threadIdx.x >> 5), NT16 = R >> 4, NG = NU * 8;
  const int n0 = blockIdx.y * 8 * NB;
  if(m >= NT16 || n0 >= T) return;
  float acc[NB][4];
#pragma unroll
  for(int b = 0; b < NB; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const uint4* Wm = W + (long long)m * NU * 224 + lane;
  uint4 nx[7];
#pragma unroll
  for(int i = 0; i < 7; ++i) nx[i] = __ldg(Wm + i * 32);
  for(int u = 0; u < NU; ++u){
    unsigned int w[28];
#pragma unroll
    for(int i = 0; i < 7; ++i){ w[4 * i] = nx[i].x; w[4 * i + 1] = nx[i].y; w[4 * i + 2] = nx[i].z; w[4 * i + 3] = nx[i].w; }
    if(u + 1 < NU){
#pragma unroll
      for(int i = 0; i < 7; ++i) nx[i] = __ldg(Wm + (long long)(u + 1) * 224 + i * 32);
    }
    const int gb = u * 8;
    tcm_unit_group<0, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<1, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<2, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<3, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<4, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<5, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<6, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
    tcm_unit_group<7, NB>(w, XQ, XS, XA, NG, T, n0, gb, lane, acc);
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
   int8 activations and scales are staged per group in shared memory (double-buffered), so each warp reads its B
   fragments with conflict-free LDS. Same arithmetic, in the same order, as k_tc_mm. */
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
    int c[4];
    tcm_mma(a, b0, b1, c);
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

extern "C" __global__ void __launch_bounds__(256, 2) k_tc_mm2(const uint4* __restrict__ W, int R, int NU,
    const uint4* __restrict__ XQ, const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y,
    int ldy, int T){
  __shared__ TCM S;
  const int lane = threadIdx.x & 31, m = blockIdx.x * 8 + (threadIdx.x >> 5), NT16 = R >> 4, NG = NU * 8;
  const int n0 = blockIdx.y * 64;
  const bool active = m < NT16;
  float acc[8][4];
#pragma unroll
  for(int b = 0; b < 8; ++b) acc[b][0] = acc[b][1] = acc[b][2] = acc[b][3] = 0.f;
  const uint4* Wm = W + (long long)(active ? m : 0) * NU * 224 + lane;
  uint4 nx[7];
  if(active){
#pragma unroll
    for(int i = 0; i < 7; ++i) nx[i] = __ldg(Wm + i * 32);
  }
  tcm_stage(S, 0, XQ, XS, XA, NG, 0, n0, T);
  for(int u = 0; u < NU; ++u){
    unsigned int w[28];
#pragma unroll
    for(int i = 0; i < 7; ++i){ w[4 * i] = nx[i].x; w[4 * i + 1] = nx[i].y; w[4 * i + 2] = nx[i].z; w[4 * i + 3] = nx[i].w; }
    if(active && u + 1 < NU){
#pragma unroll
      for(int i = 0; i < 7; ++i) nx[i] = __ldg(Wm + (long long)(u + 1) * 224 + i * 32);
    }
    const int gb = u * 8;
    TCM2_STEP(0) TCM2_STEP(1) TCM2_STEP(2) TCM2_STEP(3) TCM2_STEP(4) TCM2_STEP(5) TCM2_STEP(6) TCM2_STEP(7)
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
