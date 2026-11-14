/* tc_hip.cu - the tc.cu kernels (names, arguments, launch shapes) for HIP / RDNA2 and RDNA3, on a dp4a-friendly layout.

   Weights: on HIP, k_tc_pack turns the 5-trit lane layout into the 5-trit ROW layout (g5_core's: per 128-column
   group and row, 28 bytes: 24 five-trit bytes c = sum_i trit_i 3^i of columns 5p .. 5p+4, two four-trit bytes of
   columns 120 + 4m .., the fp16 scale), the same size. A 243-entry table in shared memory turns a byte into its
   digits (four in one dp4a word, the fifth packed with three other bytes' fifth digits).

   Activations (k_tc_q8): llama.cpp-style q8 per 128-column group, as tc.cu (scale amax / 127, round to nearest even,
   XA = the sum of the group's 128 int8 values), stored in a 5-column order: 32 words per token and group, A[p] =
   x[5p .. 5p+3] (p < 24), B[k] = x[5(4k+j)+4] (j = 0..3, k < 6), C[m] = x[120+4m .. 123+4m] (m < 2).

   Numerics: tc.cu's. A group's 128-term dot product of the digits (0 / 1 / 2) with the int8 activations is an exact
   integer, the group's activation sum is subtracted (sum (d - 1) q = sum d q - sum q), and it is scaled by
   x-scale * w-scale and added in fp32, group after group, in tc.cu's expression. A token's result does not depend
   on how many tokens share the pass (decode = verify) or on the split of a row's bytes between threads.

   Threads: two per row (h = 0: words 0 .. 2 = bytes 0 .. 11; h = 1: words 3 .. 6 = bytes 12 .. 25 and the scale);
   their integer partial sums are added with one shuffle. A block of 32 NW threads covers 16 NW rows, as tc.cu's NW
   warps of one 16-row tile each. */

#define __shfl_xor_sync(m, v, o) __shfl_xor(v, o)
#define __shfl_up_sync(m, v, o) __shfl_up(v, o)
#define __shfl_sync(m, v, l) __shfl(v, l)

__device__ __forceinline__ float tc_h2f(unsigned short h){ return __half2float(*(const __half*)&h); }
__device__ __forceinline__ int td_dot4(unsigned int a, unsigned int b, int c){
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
  /* Each digit byte is 0, 1 or 2, so signed and unsigned interpretations are identical. No clamp. */
  return __builtin_amdgcn_sdot4((int)a, (int)b, c, false);
#else
  return __builtin_amdgcn_sudot4(false, (int)a, true, (int)b, c, false);   /* digits unsigned, activations signed */
#endif
}

/* ------------------------------------------------------------------ repack: 5-trit lane layout -> row layout */
extern "C" __global__ void k_tc_pack(const unsigned char* __restrict__ S, unsigned char* __restrict__ O, int R, int NG){
  /* one thread per output byte: O[(g R + r) 28 + p] = S[((g NT16 + r / 16) 28 + p) 16 + r % 16] */
  const long long n = (long long)R * NG * 28;
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;
  const int p = (int)(idx % 28);
  const long long gr = idx / 28;
  const int r = (int)(gr % R), g = (int)(gr / R), NT16 = R >> 4;
  O[idx] = S[(((long long)g * NT16 + (r >> 4)) * 28 + p) * 16 + (r & 15)];
}

/* ------------------------------------------------------------------ activations */
extern "C" __global__ void k_tc_q8(const float* __restrict__ X, int ldx, unsigned int* __restrict__ XQ,
                                   float* __restrict__ XS, int* __restrict__ XA, int NG, int ntok){
  const int g = blockIdx.x, n = blockIdx.y, e = threadIdx.x, lane = e & 31, w = e >> 5;
  if(n >= ntok) return;
  __shared__ float wm[4];
  __shared__ int ws[4];
  __shared__ unsigned int q[128];
  const float v = X[(long long)n * ldx + g * 128 + e];
  float a = fabsf(v);
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o));
  if(lane == 0) wm[w] = a;
  __syncthreads();
  const float amax = fmaxf(fmaxf(wm[0], wm[1]), fmaxf(wm[2], wm[3]));
  const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
  const int qi = __float2int_rn(v * id);
  q[e] = (unsigned int)qi & 255u;
  int sum = qi;
  for(int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o);
  if(lane == 0) ws[w] = sum;
  __syncthreads();
  if(e < 32){
    unsigned int word;
    if(e < 24){ const int c = 5 * e; word = q[c] | (q[c + 1] << 8) | (q[c + 2] << 16) | (q[c + 3] << 24); }
    else if(e < 30){ const int k = 20 * (e - 24); word = q[k + 4] | (q[k + 9] << 8) | (q[k + 14] << 16) | (q[k + 19] << 24); }
    else { const int c = 120 + 4 * (e - 30); word = q[c] | (q[c + 1] << 8) | (q[c + 2] << 16) | (q[c + 3] << 24); }
    XQ[((long long)n * NG + g) * 32 + e] = word;
  }
  if(e == 0){
    XS[(long long)n * NG + g] = amax / 127.0f;
    XA[(long long)n * NG + g] = ws[0] + ws[1] + ws[2] + ws[3];
  }
}

/* ------------------------------------------------------------------ the product */
#define TD_NB 8                              /* tokens staged per pass over a group */

#define TD_UG 8                              /* groups staged per barrier pair (one unit) */

struct TDS {
  unsigned long long lut[243];
  uint4 x[TD_UG][TD_NB][8];
  float xs[TD_UG][TD_NB];
  int xa[TD_UG][TD_NB];
};

__device__ __forceinline__ void td_lut(TDS& S){
  for(int c = threadIdx.x; c < 243; c += blockDim.x){
    const unsigned int d0 = c % 3, d1 = (c / 3) % 3, d2 = (c / 9) % 3, d3 = (c / 27) % 3, d4 = c / 81;
    S.lut[c] = (unsigned long long)(d0 | (d1 << 8) | (d2 << 16) | (d3 << 24)) | ((unsigned long long)d4 << 32);
  }
  __syncthreads();                             /* the table is read before the first staging barrier */
}

/* this thread's half row of group g: digit words dl (h = 0: 12 + 3 packed fifth digits; h = 1: 12 + 3 + 2) */
__device__ __forceinline__ void td_digits(const TDS& S, const unsigned int* __restrict__ b, int h, unsigned int dl[14],
                                          unsigned int d5[3], float& d){
  unsigned int wv[4];
  if(h == 0){ wv[0] = b[0]; wv[1] = b[1]; wv[2] = b[2]; wv[3] = 0u; }
  else { wv[0] = b[3]; wv[1] = b[4]; wv[2] = b[5]; wv[3] = b[6]; }
#pragma unroll
  for(int p = 0; p < 12; ++p){
    const unsigned long long e = S.lut[(wv[p >> 2] >> (8 * (p & 3))) & 255u];
    dl[p] = (unsigned int)e;
    const unsigned int f = (unsigned int)(e >> 32);
    if((p & 3) == 0) d5[p >> 2] = f; else d5[p >> 2] |= f << (8 * (p & 3));
  }
  if(h == 1){
    dl[12] = (unsigned int)S.lut[wv[3] & 255u];
    dl[13] = (unsigned int)S.lut[(wv[3] >> 8) & 255u];
    const unsigned short hs = (unsigned short)(wv[3] >> 16);
    d = tc_h2f(hs);
  } else {
    dl[12] = dl[13] = 0u;
    d = 0.f;
  }
}

/* the integer sum of the thread's half row with token j's activations (staged), both halves added (exact) */
__device__ __forceinline__ int td_sum(const TDS& S, int G, int j, int h, const unsigned int dl[14], const unsigned int d5[3]){
  const unsigned int* x = (const unsigned int*)S.x[G][j];
  int s = 0;
  const int p0 = 12 * h, k0 = 3 * h;
#pragma unroll
  for(int p = 0; p < 12; ++p) s = td_dot4(dl[p], x[p0 + p], s);
#pragma unroll
  for(int k = 0; k < 3; ++k) s = td_dot4(d5[k], x[24 + k0 + k], s);
  if(h == 1){ s = td_dot4(dl[12], x[30], s); s = td_dot4(dl[13], x[31], s); }
  return s + __shfl_xor(s, 1);
}

/* stage tokens n0 .. n0 + TD_NB - 1 of groups g0 .. g0 + TD_UG - 1 */
__device__ __forceinline__ void td_stage(TDS& S, const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                         const int* __restrict__ XA, int NG, int g0, int n0, int T){
  const int tid = threadIdx.x;
  for(int i = tid; i < TD_UG * TD_NB * 8; i += blockDim.x){
    const int G = i / (TD_NB * 8), j = (i >> 3) % TD_NB, k = i & 7, n = n0 + j;
    if(n < T) S.x[G][j][k] = ((const uint4*)(XQ + ((long long)n * NG + g0 + G) * 32))[k];
  }
  for(int i = tid; i < TD_UG * TD_NB; i += blockDim.x){
    const int G = i / TD_NB, j = i % TD_NB, n = n0 + j;
    if(n < T){ S.xs[G][j] = XS[(long long)n * NG + g0 + G]; S.xa[G][j] = XA[(long long)n * NG + g0 + G]; }
  }
}

/* rows: this thread's row r of a tensor (valid: r < R), groups [g_lo, g_hi), tokens n0 .. n0 + NT - 1 (NT a multiple
   of TD_NB); acc[j] = token n0 + j. Every thread of the block runs the barriers. */
template <int NT>
__device__ __forceinline__ void td_rows(TDS& S, const unsigned int* __restrict__ W, int R, bool valid, int r, int h,
                                        int g_lo, int g_hi, const unsigned int* __restrict__ XQ,
                                        const float* __restrict__ XS, const int* __restrict__ XA, int NG, int n0,
                                        int T, float acc[NT]){
  for(int g0 = g_lo; g0 < g_hi; g0 += TD_UG){            /* g_lo, g_hi: multiples of 8 (units) */
#pragma unroll
    for(int jb = 0; jb < NT; jb += TD_NB){
      __syncthreads();
      td_stage(S, XQ, XS, XA, NG, g0, n0 + jb, T);
      __syncthreads();
#pragma unroll 1
      for(int G = 0; G < TD_UG; ++G){
        unsigned int dl[14], d5[3];
        float d = 0.f;
        if(valid) td_digits(S, W + ((long long)(g0 + G) * R + r) * 7, h, dl, d5, d);
        d = __shfl(d, (threadIdx.x & 31) | 1);   /* the scale sits in the h = 1 half */
        if(valid){
#pragma unroll
          for(int j = 0; j < TD_NB; ++j){
            if(n0 + jb + j < T){
              const int c = td_sum(S, G, j, h, dl, d5);
              acc[jb + j] += (float)(c - S.xa[G][j]) * (S.xs[G][j] * d);
            }
          }
        }
      }
    }
  }
}

template <int NW>
__device__ __forceinline__ void tc_mv(const unsigned int* __restrict__ W, float* __restrict__ y, int R, int NU,
                                      const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                      const int* __restrict__ XA, int ntok){
  /* grid (ceil(R / 16 / NW), splits): block y covers units [by NU / splits, (by + 1) NU / splits); token n's part
     at y + (n splits + by) R (as tc.cu) */
  __shared__ TDS S;
  td_lut(S);
  const int tid = threadIdx.x, h = tid & 1, r = blockIdx.x * 16 * NW + (tid >> 1), NG = NU * 8;
  const bool valid = r < R;
  const int g_lo = (int)((long long)blockIdx.y * NU / gridDim.y) * 8, g_hi = (int)((long long)(blockIdx.y + 1) * NU / gridDim.y) * 8;
  float acc[TD_NB];
#pragma unroll
  for(int j = 0; j < TD_NB; ++j) acc[j] = 0.f;
  td_rows<TD_NB>(S, W, R, valid, r, h, g_lo, g_hi, XQ, XS, XA, NG, 0, ntok, acc);
  if(!valid || h) return;
  const int sp = gridDim.y, by = blockIdx.y;
#pragma unroll
  for(int j = 0; j < TD_NB; ++j) if(j < ntok) y[((long long)j * sp + by) * R + r] = acc[j];
}

template <int NW>
__device__ __forceinline__ void tc_mv3(const unsigned int* __restrict__ W0, float* __restrict__ y0, int R0,
                                       const unsigned int* __restrict__ W1, float* __restrict__ y1, int R1,
                                       const unsigned int* __restrict__ W2, float* __restrict__ y2, int R2, int NU,
                                       const unsigned int* __restrict__ XQ, const float* __restrict__ XS,
                                       const int* __restrict__ XA, int ntok){
  /* the blocks along x cover the 16-row tiles of W0, then W1, then W2 (no splits) */
  __shared__ TDS S;
  td_lut(S);
  const int tid = threadIdx.x, h = tid & 1, NG = NU * 8;
  int r = blockIdx.x * 16 * NW + (tid >> 1);
  const unsigned int* W = W0;
  float* y = y0;
  int R = R0;
  if(r >= R0){
    r -= R0; W = W1; y = y1; R = R1;
    if(r >= R1){ r -= R1; W = W2; y = y2; R = R2; }
  }
  const bool valid = r < R;
  float acc[TD_NB];
#pragma unroll
  for(int j = 0; j < TD_NB; ++j) acc[j] = 0.f;
  td_rows<TD_NB>(S, valid ? W : W0, valid ? R : R0, valid, valid ? r : 0, h, 0, NG, XQ, XS, XA, NG, 0, ntok, acc);
  if(!valid || h) return;
#pragma unroll
  for(int j = 0; j < TD_NB; ++j) if(j < ntok) y[(long long)j * R + r] = acc[j];
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

/* ------------------------------------------------------------------ prompt chunks: Y[n ldy + r] */
template <int NW, int NT>
__device__ __forceinline__ void tc_mm(const unsigned int* __restrict__ W, int R, int NU, const unsigned int* __restrict__ XQ,
                                      const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y,
                                      int ldy, int T){
  __shared__ TDS S;
  td_lut(S);
  const int tid = threadIdx.x, h = tid & 1, r = blockIdx.x * 16 * NW + (tid >> 1), NG = NU * 8;
  const int n0 = blockIdx.y * NT;
  if(n0 >= T) return;                          /* the whole block: no barrier is left half-run */
  const bool valid = r < R;
  float acc[NT];
#pragma unroll
  for(int j = 0; j < NT; ++j) acc[j] = 0.f;
  td_rows<NT>(S, W, R, valid, valid ? r : 0, h, 0, NG, XQ, XS, XA, NG, n0, T, acc);
  if(!valid || h) return;
#pragma unroll
  for(int j = 0; j < NT; ++j) if(n0 + j < T) Y[(long long)(n0 + j) * ldy + r] = acc[j];
}

#define TC_MM_ARGS const unsigned int* __restrict__ W, int R, int NU, const unsigned int* __restrict__ XQ, \
  const float* __restrict__ XS, const int* __restrict__ XA, float* __restrict__ Y, int ldy, int T
/* tc.cu's launch shapes: k_tc_mm{4,8,16}: grid (ceil(R / 64), ceil(T / (8 NB))), 128 threads; k_tc_mm2: grid
   (ceil(R / 128), ceil(T / 64)), 256 threads */
extern "C" __global__ void __launch_bounds__(128) k_tc_mm4(TC_MM_ARGS){ tc_mm<4, 32>(W, R, NU, XQ, XS, XA, Y, ldy, T); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mm8(TC_MM_ARGS){ tc_mm<4, 64>(W, R, NU, XQ, XS, XA, Y, ldy, T); }
extern "C" __global__ void __launch_bounds__(128) k_tc_mm16(TC_MM_ARGS){ tc_mm<4, 128>(W, R, NU, XQ, XS, XA, Y, ldy, T); }
extern "C" __global__ void __launch_bounds__(256) k_tc_mm2(TC_MM_ARGS){ tc_mm<8, 64>(W, R, NU, XQ, XS, XA, Y, ldy, T); }

extern "C" __global__ void k_tc_sum(const float* __restrict__ P, float* __restrict__ y, int R, int ntok, int sp){
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= (long long)R * ntok) return;
  const long long n = i / R, r = i - n * R;
  const float* q = P + n * sp * R + r;
  float s = q[0];
  for(int p = 1; p < sp; ++p) s += q[(long long)p * R];
  y[i] = s;
}

/* word e (0 .. 31) of a group's 5-column activation order from its 128 int8 values (k_tc_q8's packing); used by
   the fused quantisers of tcf.cu on HIP */
__device__ __forceinline__ unsigned int tcq_word5(const unsigned char* q, int e){
  if(e < 24){ const int c = 5 * e; return q[c] | (q[c + 1] << 8) | (q[c + 2] << 16) | ((unsigned int)q[c + 3] << 24); }
  if(e < 30){ const int k = 20 * (e - 24); return q[k + 4] | (q[k + 9] << 8) | (q[k + 14] << 16) | ((unsigned int)q[k + 19] << 24); }
  const int c = 120 + 4 * (e - 30);
  return q[c] | (q[c + 1] << 8) | (q[c + 2] << 16) | ((unsigned int)q[c + 3] << 24);
}
