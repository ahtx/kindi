/* tcf.cu - fused kernels of the tensor-core plans (compiled after kindi.cu's device copies, spec.cu and tc.cu).

   The producers of a product's input (k_rmsfold, k_smfold, k_fold_fwht: one block per 512/1024-column chunk) also
   quantise their own chunk for the tensor cores, with k_tc_q8's arithmetic (so the int8 rows are the same), and the
   attention stages of a verify pass run all its queries in one launch (blockIdx.y = query), each query with the
   decode kernel's own code and its own position count. */

__device__ __forceinline__ void tcq_chunk(const float* __restrict__ xrow, int c0, int nc, unsigned int* __restrict__ XQ,
                                          float* __restrict__ XS, int* __restrict__ XA, int NG, int n){
  /* the block (a multiple of 128 threads) quantises columns [c0, c0 + nc) of token n's row; k_tc_q8's numbers */
  __shared__ float qm[32];
  __shared__ int qs[32];
  const int e = threadIdx.x, lane = e & 31, w = e >> 5, nw = blockDim.x >> 5, gpb = nw >> 2;
  for(int g0 = c0 >> 7; g0 < (c0 + nc) >> 7; g0 += gpb){
    const int g = g0 + (w >> 2), col = (w & 3) * 32 + lane;
    const bool on = g < (c0 + nc) >> 7;
    const float v = on ? xrow[g * 128 + col] : 0.f;
    float a = fabsf(v);
    for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if(lane == 0) qm[w] = a;
    __syncthreads();
    const int wb = w & ~3;
    const float amax = fmaxf(fmaxf(qm[wb], qm[wb + 1]), fmaxf(qm[wb + 2], qm[wb + 3]));
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const int qi = __float2int_rn(v * id);
    int sum = qi;
    for(int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if(lane == 0) qs[w] = sum;
    unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3));
    word |= __shfl_xor_sync(0xffffffffu, word, 1);
    word |= __shfl_xor_sync(0xffffffffu, word, 2);
    if(on && (lane & 3) == 0){
      const int k4 = col >> 2, s = k4 >> 3, h = (k4 >> 2) & 1, t = k4 & 3;
      XQ[(((long long)n * NG + g) * 4 + t) * 8 + 2 * s + h] = word;
    }
    __syncthreads();
    if(on && (w & 3) == 0 && lane == 0){
      XS[(long long)n * NG + g] = amax / 127.0f;
      XA[(long long)n * NG + g] = qs[wb] + qs[wb + 1] + qs[wb + 2] + qs[wb + 3];
    }
    __syncthreads();
  }
}

/* ------------------------------------------------------------------ fast folds (FF, default on) */
/* The decode folds were latency-bound: one block per Hadamard block (5 blocks for the 27B, 1 for the 9B), loops
   of dependent global loads and a __syncthreads per butterfly stage. ff_*: the same 512 threads per block and
   the same work per thread as k_rmsfold / k_smfold / k_fold_fwht (so the double sum of squares has the same terms
   in the same order: the norm is bit-identical), but all loads are issued at once into registers (element
   c0 + j + 512 k of the chunk in v[k]), the butterflies below 32 are warp shuffles, 32 .. 256 go through shared
   memory and 512 and up stay inside the thread. Every butterfly is the same a + b / a - b of the same pair in the
   same stage order as fwht_chunks / fwht_n, so the fold is bit-identical; the int8 rows (k_tc_q8's arithmetic)
   are made from the registers. E = HB / 512 (1, 2 or 8); other blocks keep the old kernels. */
#ifndef FF
#define FF 1
#endif
#define FF_PMAX 6
#define FF_SB(E) ((E) < 8 ? (E) : 8)               /* registers per shared-memory batch (sm holds FF_SB(E) x 512) */
template <int E, int HBE = E>
__device__ __forceinline__ void ff_fwht(float* v, int j, float* sm){
  /* v[k] = column c0 + j + 512 k; Hadamard blocks of 512 HBE columns (HBE consecutive registers) */
#pragma unroll
  for(int h = 1; h < 32; h <<= 1){
#pragma unroll
    for(int k = 0; k < E; ++k){
      const float o = __shfl_xor_sync(0xffffffffu, v[k], h);
      v[k] = (j & h) ? (o - v[k]) : (v[k] + o);
    }
  }
#pragma unroll
  for(int k0 = 0; k0 < E; k0 += FF_SB(E)){
#pragma unroll
    for(int h = 32; h < 512; h <<= 1){
#pragma unroll
      for(int k = k0; k < E && k < k0 + FF_SB(E); ++k) sm[(k - k0) * 512 + j] = v[k];
      __syncthreads();
#pragma unroll
      for(int k = k0; k < E && k < k0 + FF_SB(E); ++k){
        const float o = sm[(k - k0) * 512 + (j ^ h)];
        v[k] = (j & h) ? (o - v[k]) : (v[k] + o);
      }
      __syncthreads();
    }
  }
#pragma unroll
  for(int h = 1; h < HBE; h <<= 1){
#pragma unroll
    for(int k = 0; k < E; ++k){
      if(!(k & h)){ const float a = v[k], b = v[k + h]; v[k] = a + b; v[k + h] = a - b; }
    }
  }
}

template <int E, int HBE = E>
__device__ __forceinline__ void ff_scale_store(float* v, float* __restrict__ xo, int c0, int j){
  const float sc = (HBE == 1) ? 0.044194173824159220878f : rsqrtf((float)(512 * HBE));
#pragma unroll
  for(int k = 0; k < E; ++k){ v[k] = v[k] * sc; xo[c0 + j + 512 * k] = v[k]; }
}

template <int E>
__device__ __forceinline__ void ff_q8_row(const float* v, int c0, int n, unsigned int* __restrict__ XQ, float* __restrict__ XS,
                                          int* __restrict__ XA, int NG, int* qo, float* ao){
  /* thread j holds column c0 + j + 512 k in v[k]; a group of 128 columns = 4 warps; k_tc_q8's numbers (q and amax
     of each value back in qo, ao) */
  __shared__ float qm[E * 16];
  __shared__ int qs[E * 16];
  const int j = threadIdx.x, lane = j & 31, w = j >> 5, wb = w & ~3;
#pragma unroll
  for(int k = 0; k < E; ++k){
    float a = fabsf(v[k]);
    for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if(lane == 0) qm[k * 16 + w] = a;
  }
  __syncthreads();
#pragma unroll
  for(int k = 0; k < E; ++k){
    const float amax = fmaxf(fmaxf(qm[k * 16 + wb], qm[k * 16 + wb + 1]), fmaxf(qm[k * 16 + wb + 2], qm[k * 16 + wb + 3]));
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const int qi = __float2int_rn(v[k] * id);
    qo[k] = qi;
    ao[k] = amax;
    int sum = qi;
    for(int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if(lane == 0) qs[k * 16 + w] = sum;
    unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3));
    word |= __shfl_xor_sync(0xffffffffu, word, 1);
    word |= __shfl_xor_sync(0xffffffffu, word, 2);
    const int g = ((c0 + 512 * k) >> 7) + (w >> 2), col = (w & 3) * 32 + lane;
    if((lane & 3) == 0){
      const int k4 = col >> 2, s4 = k4 >> 3, hh = (k4 >> 2) & 1, t = k4 & 3;
      XQ[(((long long)n * NG + g) * 4 + t) * 8 + 2 * s4 + hh] = word;
    }
  }
  __syncthreads();
  if((w & 3) == 0 && lane == 0){
#pragma unroll
    for(int k = 0; k < E; ++k){
      const int g = ((c0 + 512 * k) >> 7) + (w >> 2);
      const float amax = fmaxf(fmaxf(qm[k * 16 + wb], qm[k * 16 + wb + 1]), fmaxf(qm[k * 16 + wb + 2], qm[k * 16 + wb + 3]));
      XS[(long long)n * NG + g] = amax / 127.0f;
      XA[(long long)n * NG + g] = qs[k * 16 + wb] + qs[k * 16 + wb + 1] + qs[k * 16 + wb + 2] + qs[k * 16 + wb + 3];
    }
  }
}

template <int E>
__device__ __forceinline__ void ff_q8(const float* v, int c0, int n, unsigned int* __restrict__ XQ, float* __restrict__ XS,
                                      int* __restrict__ XA, int NG, int hp){
  /* the token's int8 row; hp (tc.cu tc_hp): rows 2 n (hi) and 2 n + 1 (the residual), made in one pass: the residual
     r = v - q amax / 127 is at most amax / 254, so its row takes the fixed scale amax / (254 x 127) (no second max) */
  if(!hp){
    int qi[E];
    float am[E];
    ff_q8_row<E>(v, c0, n, XQ, XS, XA, NG, qi, am);
    return;
  }
  __shared__ float qm[E * 16];
  __shared__ int qs[E * 16], qs2[E * 16];
  const int j = threadIdx.x, lane = j & 31, w = j >> 5, wb = w & ~3;
#pragma unroll
  for(int k = 0; k < E; ++k){
    float a = fabsf(v[k]);
    for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    if(lane == 0) qm[k * 16 + w] = a;
  }
  __syncthreads();
#pragma unroll
  for(int k = 0; k < E; ++k){
    const float amax = fmaxf(fmaxf(qm[k * 16 + wb], qm[k * 16 + wb + 1]), fmaxf(qm[k * 16 + wb + 2], qm[k * 16 + wb + 3]));
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const int qi = __float2int_rn(v[k] * id);
    const float r = v[k] - (float)qi * (amax / 127.0f);
    const int q2 = max(-127, min(127, __float2int_rn(r * (amax != 0.0f ? 32258.0f / amax : 0.0f))));
    int sum = qi, sum2 = q2;
    for(int o = 16; o > 0; o >>= 1){ sum += __shfl_xor_sync(0xffffffffu, sum, o); sum2 += __shfl_xor_sync(0xffffffffu, sum2, o); }
    if(lane == 0){ qs[k * 16 + w] = sum; qs2[k * 16 + w] = sum2; }
    unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3)), word2 = ((unsigned int)q2 & 255u) << (8 * (lane & 3));
    word |= __shfl_xor_sync(0xffffffffu, word, 1);
    word |= __shfl_xor_sync(0xffffffffu, word, 2);
    word2 |= __shfl_xor_sync(0xffffffffu, word2, 1);
    word2 |= __shfl_xor_sync(0xffffffffu, word2, 2);
    const int g = ((c0 + 512 * k) >> 7) + (w >> 2), col = (w & 3) * 32 + lane;
    if((lane & 3) == 0){
      const int k4 = col >> 2, s4 = k4 >> 3, hh = (k4 >> 2) & 1, t = k4 & 3;
      XQ[(((long long)(2 * n) * NG + g) * 4 + t) * 8 + 2 * s4 + hh] = word;
      XQ[(((long long)(2 * n + 1) * NG + g) * 4 + t) * 8 + 2 * s4 + hh] = word2;
    }
  }
  __syncthreads();
  if((w & 3) == 0 && lane == 0){
#pragma unroll
    for(int k = 0; k < E; ++k){
      const int g = ((c0 + 512 * k) >> 7) + (w >> 2);
      const float amax = fmaxf(fmaxf(qm[k * 16 + wb], qm[k * 16 + wb + 1]), fmaxf(qm[k * 16 + wb + 2], qm[k * 16 + wb + 3]));
      XS[(long long)(2 * n) * NG + g] = amax / 127.0f;
      XA[(long long)(2 * n) * NG + g] = qs[k * 16 + wb] + qs[k * 16 + wb + 1] + qs[k * 16 + wb + 2] + qs[k * 16 + wb + 3];
      XS[(long long)(2 * n + 1) * NG + g] = amax / 32258.0f;
      XA[(long long)(2 * n + 1) * NG + g] = qs2[k * 16 + wb] + qs2[k * 16 + wb + 1] + qs2[k * 16 + wb + 2] + qs2[k * 16 + wb + 3];
    }
  }
}

template <int E, int NM, int HBE = E, bool RAW = false>
__device__ __forceinline__ void ff_rmsfold(const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ xw,
    const float* __restrict__ w, const float* __restrict__ sg, float* __restrict__ xn, float* __restrict__ xo, int C,
    int add, int n, unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA, int hp){
  /* C = 512 NM; the residual parts (add <= FF_PMAX) are summed in part order as rsum does */
  __shared__ float sm[FF_SB(E) * 512];
  __shared__ double red[16];
  __shared__ float bscale;
  const int j = threadIdx.x, lane = j & 31, wp = j >> 5, c0 = blockIdx.x * 512 * E, b0 = c0 >> 9;
  float xv[NM];
#pragma unroll
  for(int m = 0; m < NM; ++m) xv[m] = x[j + 512 * m];
  if(add){
    /* every part's loads are issued unconditionally (a part past add re-reads part 0) so they are all in flight
       at once; only the parts below add are added, in part order */
    float rv[NM];
#pragma unroll
    for(int m = 0; m < NM; ++m) rv[m] = r[j + 512 * m];
#pragma unroll
    for(int p = 1; p < FF_PMAX; ++p){
      const float* rp = r + (long long)(p < add ? p : 0) * C;
      float t[NM];
#pragma unroll
      for(int m = 0; m < NM; ++m) t[m] = rp[j + 512 * m];
      if(p < add){
#pragma unroll
        for(int m = 0; m < NM; ++m) rv[m] += t[m];
      }
    }
#pragma unroll
    for(int m = 0; m < NM; ++m) xv[m] = xv[m] + rv[m];
  }
  double acc = 0.0;
#pragma unroll
  for(int m = 0; m < NM; ++m){ const double d = (double)xv[m]; acc += d * d; }
  acc = wsum_d(acc);
  if(lane == 0) red[wp] = acc;
  __syncthreads();
  if(wp == 0){
    double t = lane < 16 ? red[lane] : 0.0;
    t = wsum_d(t);
    if(lane == 0) bscale = 1.0f / sqrtf((float)(t / C) + 1e-6f);
  }
  __syncthreads();
  const float s = bscale;
  float v[E];
#pragma unroll
  for(int k = 0; k < E; ++k){
    float xf = 0.f;
#pragma unroll
    for(int m = 0; m < NM; ++m) if(m == b0 + k) xf = xv[m];
    const int e = c0 + j + 512 * k;
    if(add) xw[e] = xf;
    const float nv = (xf * s) * w[e];
    xn[e] = nv;
    v[k] = sg[e] * nv;
  }
  ff_fwht<E, HBE>(v, j, sm);
  if(RAW){                                   /* the first stages of a split fold (k_mix8_q finishes it) */
#pragma unroll
    for(int k = 0; k < E; ++k) xo[c0 + j + 512 * k] = v[k];
    return;
  }
  ff_scale_store<E, HBE>(v, xo, c0, j);
  ff_q8<E>(v, c0, n, XQ, XS, XA, C >> 7, hp);
}

template <int E, int HBE = E, bool RAW = false>
__device__ __forceinline__ void ff_smfold(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg,
    float* __restrict__ xo, int perm_hd, int perm_nk, int perm_rep, int C, int n, unsigned int* __restrict__ XQ,
    float* __restrict__ XS, int* __restrict__ XA, int hp){
  __shared__ float sm[FF_SB(E) * 512];
  const int j = threadIdx.x, c0 = blockIdx.x * 512 * E;
  float v[E];
#pragma unroll
  for(int k = 0; k < E; ++k){
    const int e = c0 + j + 512 * k;
    int src = e;
    if(perm_rep > 1){
      const int hd = e % perm_hd, grouped = e / perm_hd;
      const int rep = grouped % perm_rep, nk = grouped / perm_rep;
      src = hd + perm_hd * (nk + perm_nk * rep);
    }
    const float xa = a[src];
    const float m = (xa / (1.0f + expf(-xa))) * b[src];
    v[k] = sg[e] * m;
  }
  ff_fwht<E, HBE>(v, j, sm);
  if(RAW){                                   /* the first stages of a split fold (k_mix8_q finishes it) */
#pragma unroll
    for(int k = 0; k < E; ++k) xo[c0 + j + 512 * k] = v[k];
    return;
  }
  ff_scale_store<E, HBE>(v, xo, c0, j);
  ff_q8<E>(v, c0, n, XQ, XS, XA, C >> 7, hp);
}

template <int E, int HBE = E, bool RAW = false>
__device__ __forceinline__ void ff_fold(const float* __restrict__ x, const float* __restrict__ sg, float* __restrict__ xo, int C,
    int n, unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA, int hp){
  __shared__ float sm[FF_SB(E) * 512];
  const int j = threadIdx.x, c0 = blockIdx.x * 512 * E;
  float v[E];
#pragma unroll
  for(int k = 0; k < E; ++k){ const int e = c0 + j + 512 * k; v[k] = sg[e] * x[e]; }
  ff_fwht<E, HBE>(v, j, sm);
  if(RAW){                                   /* the first stages of a split fold (k_mix8_q finishes it) */
#pragma unroll
    for(int k = 0; k < E; ++k) xo[c0 + j + 512 * k] = v[k];
    return;
  }
  ff_scale_store<E, HBE>(v, xo, c0, j);
  ff_q8<E>(v, c0, n, XQ, XS, XA, C >> 7, hp);
}

/* The 9B's 4096-point folds in two launches (KINDI_FOLD_SPLIT, default on): k_*_qa does the stages below 512 of each
   512-column chunk (one block per chunk; k_rmsfold_qa computes the whole norm in every block, the same double sums
   as k_rmsfold), k_mix8_q (128 threads per 128 columns of every chunk) the stages 512, 1024 and 2048 in registers,
   the scale and the int8 rows. The same butterflies in the same order: bit-identical to the one-block fold. */
extern "C" __global__ void __launch_bounds__(512) k_rmsfold_qa(const float* __restrict__ x, const float* __restrict__ r,
    float* __restrict__ xw, const float* __restrict__ w, const float* __restrict__ sg, float* __restrict__ xn,
    float* __restrict__ xo, int add, int T_, int s_x, int s_r, int s_xw, int s_xn, int s_xo){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  ff_rmsfold<1, 8, 1, true>(x + t_ * s_x, r + t_ * s_r, xw + t_ * s_xw, w, sg, xn + t_ * s_xn, xo + t_ * s_xo, 4096, add,
                            (int)t_, 0, 0, 0, 0);
}
extern "C" __global__ void __launch_bounds__(512) k_smfold_qa(const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ sg, float* __restrict__ xo, int perm_hd, int perm_nk, int perm_rep, int T_, int s_a, int s_b,
    int s_xo, int C){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  ff_smfold<1, 1, true>(a + t_ * s_a, b + t_ * s_b, sg, xo + t_ * s_xo, perm_hd, perm_nk, perm_rep, C, (int)t_, 0,
                        0, 0, 0);
}
extern "C" __global__ void __launch_bounds__(512) k_fold_qa(const float* __restrict__ x, const float* __restrict__ sg,
    float* __restrict__ xo, int T_, int s_x, int s_xo, int C){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  ff_fold<1, 1, true>(x + t_ * s_x, sg, xo + t_ * s_xo, C, (int)t_, 0, 0, 0, 0);
}
extern "C" __global__ void __launch_bounds__(128) k_mix8_q(float* __restrict__ xo, int T_, int s_xo, int C,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA, int hp){
  /* grid (C / 4096 x 4, T): Hadamard block blockIdx.x / 4, columns 128 (blockIdx.x % 4) + j of its 8 chunks */
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  xo += t_ * s_xo;
  const int j = threadIdx.x, c0 = (blockIdx.x >> 2) * 4096 + (blockIdx.x & 3) * 128;
  float v[8];
#pragma unroll
  for(int k = 0; k < 8; ++k) v[k] = xo[c0 + j + 512 * k];
#pragma unroll
  for(int h = 1; h < 8; h <<= 1){
#pragma unroll
    for(int k = 0; k < 8; ++k){
      if(!(k & h)){ const float p = v[k], q = v[k + h]; v[k] = p + q; v[k + h] = p - q; }
    }
  }
  const float sc = rsqrtf(4096.0f);
#pragma unroll
  for(int k = 0; k < 8; ++k){ v[k] = v[k] * sc; xo[c0 + j + 512 * k] = v[k]; }
  ff_q8<8>(v, c0, (int)t_, XQ, XS, XA, C >> 7, hp);
}

__device__ __forceinline__ bool ff_on(int HB){ return HB == 512 || HB == 1024 || HB == 4096; }

extern "C" __global__ void k_rmsfold_qT(const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ xw,
    const float* __restrict__ w, const float* __restrict__ sg, float* __restrict__ xn, float* __restrict__ xo, int C,
    int add, int HB, int T_, int s_x, int s_r, int s_xw, int s_xn, int s_xo,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA, int hp){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  /* the widths of the Ternary-Bonsai-2 models: 2B 2048, 4B 2560 (HB 512), 9B 4096 (HB 4096), 27B 5120 (HB 1024) */
  const int key = HB * 64 + (C >> 9);
  if(FF && add <= FF_PMAX && (key == 512 * 64 + 4 || key == 512 * 64 + 5 || key == 4096 * 64 + 8 || key == 1024 * 64 + 10)){
    x += t_ * s_x; r += t_ * s_r; xw += t_ * s_xw; xn += t_ * s_xn; xo += t_ * s_xo;
    if(key == 512 * 64 + 4) ff_rmsfold<1, 4>(x, r, xw, w, sg, xn, xo, C, add, (int)t_, XQ, XS, XA, hp);
    else if(key == 512 * 64 + 5) ff_rmsfold<1, 5>(x, r, xw, w, sg, xn, xo, C, add, (int)t_, XQ, XS, XA, hp);
    else if(key == 4096 * 64 + 8) ff_rmsfold<8, 8>(x, r, xw, w, sg, xn, xo, C, add, (int)t_, XQ, XS, XA, hp);
    else ff_rmsfold<2, 10>(x, r, xw, w, sg, xn, xo, C, add, (int)t_, XQ, XS, XA, hp);
    return;
  }
  d_k_rmsfold(x + t_ * s_x, r + t_ * s_r, xw + t_ * s_xw, w, sg, xn + t_ * s_xn, xo + t_ * s_xo, C, add, HB);
  __syncthreads();
  const int hb = HB < 0 ? 512 : HB;
  tcq_chunk(xo + t_ * s_xo, blockIdx.x * hb, hb, XQ, XS, XA, C >> 7, (int)t_);
}

extern "C" __global__ void k_smfold_qT(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg,
    float* __restrict__ xo, int perm_hd, int perm_nk, int perm_rep, int HB, int T_, int s_a, int s_b, int s_xo, int C,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA, int hp){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  if(FF && ff_on(HB)){
    a += t_ * s_a; b += t_ * s_b; xo += t_ * s_xo;
    if(HB == 512) ff_smfold<1>(a, b, sg, xo, perm_hd, perm_nk, perm_rep, C, (int)t_, XQ, XS, XA, hp);
    else if(HB == 1024) ff_smfold<2>(a, b, sg, xo, perm_hd, perm_nk, perm_rep, C, (int)t_, XQ, XS, XA, hp);
    else ff_smfold<8>(a, b, sg, xo, perm_hd, perm_nk, perm_rep, C, (int)t_, XQ, XS, XA, hp);
    return;
  }
  d_k_smfold(a + t_ * s_a, b + t_ * s_b, sg, xo + t_ * s_xo, perm_hd, perm_nk, perm_rep, HB);
  __syncthreads();
  const int hb = HB < 0 ? 512 : HB;
  tcq_chunk(xo + t_ * s_xo, blockIdx.x * hb, hb, XQ, XS, XA, C >> 7, (int)t_);
}

extern "C" __global__ void k_fold_fwht_qT(const float* __restrict__ x, const float* __restrict__ sg, float* __restrict__ xo,
    int HB, int T_, int s_x, int s_xo, int C, unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA, int hp){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  if(FF && ff_on(HB)){
    x += t_ * s_x; xo += t_ * s_xo;
    if(HB == 512) ff_fold<1>(x, sg, xo, C, (int)t_, XQ, XS, XA, hp);
    else if(HB == 1024) ff_fold<2>(x, sg, xo, C, (int)t_, XQ, XS, XA, hp);
    else ff_fold<8>(x, sg, xo, C, (int)t_, XQ, XS, XA, hp);
    return;
  }
  d_k_fold_fwht(x + t_ * s_x, sg, xo + t_ * s_xo, HB);
  __syncthreads();
  const int hb = HB < 0 ? 512 : HB;
  tcq_chunk(xo + t_ * s_xo, blockIdx.x * hb, hb, XQ, XS, XA, C >> 7, (int)t_);
}

/* ------------------------------------------------------------------ verify attention, all queries per launch */
extern "C" __global__ void k_softmax_m(float* __restrict__ s, long long ss, int n0, int QN){
  if((int)blockIdx.y >= QN) return;
  d_k_softmax_rows(s + blockIdx.y * ss, n0 + (int)blockIdx.y);
}

extern "C" __global__ void __launch_bounds__(128) k_attn_vq8w_m(const float* __restrict__ s, long long ss,
    const uint2* __restrict__ Vq8, const unsigned short* __restrict__ Vd, float* __restrict__ part, long long ps,
    int n0, int QN, int nheads, int n_kv, int G){
  if((int)blockIdx.y >= QN) return;
  d_k_attn_vq8w(s + blockIdx.y * ss, Vq8, Vd, part + blockIdx.y * ps, n0 + (int)blockIdx.y, nheads, n_kv, G);
}

extern "C" __global__ void k_attn_combn_m(const float* __restrict__ part, long long ps, const float* __restrict__ gatev,
    int gs, float* __restrict__ att, int as, int n0, int QN, int nheads){
  if((int)blockIdx.y >= QN) return;
  const long long q = blockIdx.y;
  d_k_attn_combn(part + q * ps, gatev + q * gs, att + q * as, n0 + (int)q, nheads);
}

/* ------------------------------------------------------------------ draft graphs */
extern "C" __global__ void k_deq_emb4_p(const unsigned char* __restrict__ bc, const unsigned short* __restrict__ dp,
                                        const float* __restrict__ sg, float* __restrict__ h, const int* __restrict__ tokp,
                                        int D, int HB){
  /* k_deq_emb4 with the token id read from device memory (the previous draft step's argmax) */
  d_k_deq_emb4(bc, dp, sg, h, *tokp, D, HB);
}

extern "C" __global__ void k_copyf(const float* __restrict__ src, float* __restrict__ dst, long long n){
  for(long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

/* ------------------------------------------------------------------ decode attention: split flash-decode (KINDI_FD) */
/* k_fd_q8: grid (NHKV, ceil(ctx / FD_CH), queries), block 256. Block (kv, c, q) takes positions [c FD_CH, ...) of
   query q (which sees n0 + q positions) and K/V head kv with its G (<= 8) query heads: the scores with k_attn_sq8t's
   arithmetic per position (exact integer block sums as floats, times dq dk, the 8 blocks in order, x 1/16), the
   chunk's max and exp-sum per head, and its value sums (q8_0 V: byte x fp16 scale). part[(c NH + h) 256 + f] holds
   the chunk's sums, ml[(c NH + h) 2] its (max, exp-sum). k_fd_comb merges the chunks (in chunk order) and applies the
   output gate. K and V are read once; the chunking depends on the positions only, so a query's result is the same
   whether it runs alone (decode) or with others (verify). */
#ifndef FD_CH
#define FD_CH 512                        /* spec.py sets it (KINDI_FD_CH): a multiple of 32 */
#endif
__device__ __forceinline__ int fd_dp4a(int a, int b, int c){
#if !KHIP && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
  return kdp4a(a, b, c);                 /* sm_60 (P100): kindi.cu's integer fallback */
#else
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#endif
}
template <int G>
__device__ __forceinline__ void fd_q8(const uint4* __restrict__ Kq, const unsigned short* __restrict__ Kd,
    const unsigned int* __restrict__ Vq4, const unsigned short* __restrict__ Vd, const signed char* __restrict__ qq,
    const float* __restrict__ qd, int qs, float* __restrict__ part, long long ps, float* __restrict__ ml, long long mls,
    int n0, int n_kv){
  const int kv = blockIdx.x, c = blockIdx.y, q = blockIdx.z, tid = threadIdx.x;
  const int n = n0 + q, p0 = c * FD_CH;
  if(p0 >= n) return;
  const int len = min(FD_CH, n - p0), NH = n_kv * G;
  __shared__ float S[G][FD_CH < 512 ? 512 : FD_CH];   /* (also the value-sum reduction: 2 G 256 floats) */
  qq += (long long)q * qs;
  qd += (long long)q * (qs / 32);
  part += q * ps;
  ml += q * mls;
  /* scores: a warp takes 4 positions at a time, lane = (position, 32-value block b): the block's integer dot
     products with the G heads (dp4a, exact), then the 8 blocks chained in order across the lanes,
     acc = fma(sum_b, dq_b dk_b, acc) - k_attn_sq8t's float result, bit for bit */
  const int lane = tid & 31, wp = tid >> 5, b = lane & 7, pq = lane >> 3;
  int qw[G][8];
  float qs_[G];
#pragma unroll
  for(int j = 0; j < G; ++j){
#pragma unroll
    for(int i = 0; i < 8; ++i) qw[j][i] = j < G ? ((const int*)(qq + kv * G * 256 + j * 256 + 32 * b))[i] : 0;
    qs_[j] = j < G ? qd[kv * G * 8 + j * 8 + b] : 0.f;
  }
#pragma unroll 2
  for(int base = wp * 4; base < len; base += 32){     /* (positions past len are never read: S[.][len..] stays unset) */
    const int pp = base + pq, t = p0 + pp;
    const bool on = pp < len;
    uint4 w0 = make_uint4(0u, 0u, 0u, 0u), w1 = w0;
    float dk = 0.f;
    if(on){
      const uint4* row = Kq + ((long long)t * n_kv + kv) * 16;
      w0 = KLDG(&row[b * 2]);
      w1 = KLDG(&row[b * 2 + 1]);
      dk = h2f(KLDG(&Kd[((long long)t * n_kv + kv) * 8 + b]));
    }
    const int kw[8] = {(int)w0.x, (int)w0.y, (int)w0.z, (int)w0.w, (int)w1.x, (int)w1.y, (int)w1.z, (int)w1.w};
#pragma unroll
    for(int j = 0; j < G; ++j){
      {
        int isum = 0;
#pragma unroll
        for(int i = 0; i < 8; ++i) isum = fd_dp4a(kw[i], qw[j][i], isum);
        const float v = (float)isum, sc = qs_[j] * dk;
        float acc = __fmaf_rn(v, sc, 0.0f);
#pragma unroll
        for(int k = 1; k < 8; ++k){
          const float prev = __shfl_up_sync(0xffffffffu, acc, 1);
          if(b == k) acc = __fmaf_rn(v, sc, prev);
        }
        if(b == 7) S[j][pp] = on ? acc * 0.0625f : -1e30f;
      }
    }
  }
  __syncthreads();
  /* the chunk's max and exp-sum per head: warp j for head j */
  if(wp < G){
    float m = -1e30f;
    for(int i = lane; i < len; i += 32) m = fmaxf(m, S[wp][i]);
    for(int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for(int i = lane; i < len; i += 32){
      const float e = expf(S[wp][i] - m);
      S[wp][i] = e;
      l += e;
    }
    for(int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if(lane == 0){
      ml[((long long)c * NH + kv * G + wp) * 2] = m;
      ml[((long long)c * NH + kv * G + wp) * 2 + 1] = l;
    }
  }
  __syncthreads();
  /* value sums: 4 position groups x 64 threads, a thread = 4 features (one 32-bit load), two positions in flight;
     the groups are added in a fixed order at the end (3 into 0 .. 2, then 2 and 1 into 0) */
  const int grp = tid >> 6, f4 = tid & 63;
  float a[G][4];
#pragma unroll
  for(int j = 0; j < G; ++j) a[j][0] = a[j][1] = a[j][2] = a[j][3] = 0.f;
#pragma unroll 2
  for(int i = grp; i < len; i += 4){
    const long long r = (long long)(p0 + i) * n_kv + kv;
    const unsigned int w = KLDG(Vq4 + r * 64 + f4);
    const float d = h2f(KLDG(Vd + r * 8 + (f4 >> 3)));
    float v[4];
#pragma unroll
    for(int e = 0; e < 4; ++e) v[e] = sb2f(w >> (8 * e)) * d;
#pragma unroll
    for(int j = 0; j < G; ++j){
      const float pj = S[j][i];
#pragma unroll
      for(int e = 0; e < 4; ++e) a[j][e] += pj * v[e];
    }
  }
  __syncthreads();
  float* R = &S[0][0];                         /* (the scores are no longer needed) G x FD_CH floats */
  if(grp >= 2){
#pragma unroll
    for(int j = 0; j < G; ++j)
#pragma unroll
      for(int e = 0; e < 4; ++e) R[((grp - 2) * G + j) * 256 + 4 * f4 + e] = a[j][e];
  }
  __syncthreads();
  if(grp < 2){
#pragma unroll
    for(int j = 0; j < G; ++j)
#pragma unroll
      for(int e = 0; e < 4; ++e) a[j][e] += R[(grp * G + j) * 256 + 4 * f4 + e];
  }
  __syncthreads();
  if(grp == 1){
#pragma unroll
    for(int j = 0; j < G; ++j)
#pragma unroll
      for(int e = 0; e < 4; ++e) R[j * 256 + 4 * f4 + e] = a[j][e];
  }
  __syncthreads();
  if(grp == 0){
#pragma unroll
    for(int j = 0; j < G; ++j){
      float* pp = part + ((long long)c * NH + kv * G + j) * 256 + 4 * f4;
#pragma unroll
      for(int e = 0; e < 4; ++e) pp[e] = a[j][e] + R[j * 256 + 4 * f4 + e];
    }
  }
}

#define FD_ARGS const uint4* __restrict__ Kq, const unsigned short* __restrict__ Kd, const unsigned int* __restrict__ Vq4,     const unsigned short* __restrict__ Vd, const signed char* __restrict__ qq, const float* __restrict__ qd, int qs,     float* __restrict__ part, long long ps, float* __restrict__ ml, long long mls, int n0, int n_kv
extern "C" __global__ void __launch_bounds__(256) k_fd_q8_4(FD_ARGS){ fd_q8<4>(Kq, Kd, Vq4, Vd, qq, qd, qs, part, ps, ml, mls, n0, n_kv); }
extern "C" __global__ void __launch_bounds__(256) k_fd_q8_6(FD_ARGS){ fd_q8<6>(Kq, Kd, Vq4, Vd, qq, qd, qs, part, ps, ml, mls, n0, n_kv); }
extern "C" __global__ void __launch_bounds__(256) k_fd_q8_8(FD_ARGS){ fd_q8<8>(Kq, Kd, Vq4, Vd, qq, qd, qs, part, ps, ml, mls, n0, n_kv); }

extern "C" __global__ void k_fd_comb(const float* __restrict__ part, long long ps, const float* __restrict__ ml,
    long long mls, const float* __restrict__ gatev, int gs, float* __restrict__ att, int as, int n0, int nheads){
  /* grid (NH, queries), block 256 (a thread = a feature): the chunks merged in order, then the sigmoid gate */
  const int hh = blockIdx.x, q = blockIdx.y, f = threadIdx.x, n = n0 + q, nch = (n + FD_CH - 1) / FD_CH;
  part += q * ps;
  ml += q * mls;
  float M = -1e30f;
  for(int c = 0; c < nch; ++c) M = fmaxf(M, ml[((long long)c * nheads + hh) * 2]);
  float L = 0.f, acc = 0.f;
  for(int c = 0; c < nch; ++c){
    const float w = expf(ml[((long long)c * nheads + hh) * 2] - M);
    L += ml[((long long)c * nheads + hh) * 2 + 1] * w;
    acc += part[((long long)c * nheads + hh) * 256 + f] * w;
  }
  const float g = gatev[(long long)q * gs + hh * 512 + 256 + f];
  att[(long long)q * as + hh * 256 + f] = (acc / L) * (1.0f / (1.0f + expf(-g)));
}

/* ------------------------------------------------------------------ D1: residual add + RMS norm + fold + quantise */
/* k_rmsfold2_qT<HB>: k_rmsfold_qT with a thread per element of the block's HB-column chunk (grid (C / HB, T), block
   HB): the Hadamard transform's first 5 stages are warp shuffles and the rest go through shared memory (the same
   butterflies in the same order, so the fold is bit-identical); the int8 rows are made from registers. The sum of
   squares is the same double sum in a different order (so the norm can differ in the last bit). */
__device__ __forceinline__ float fwht_reg(float v, int i, int HB, float* sm){
  for(int h = 1; h < 32 && h < HB; h <<= 1){
    const float o = __shfl_xor_sync(0xffffffffu, v, h);
    v = (i & h) ? (o - v) : (v + o);
  }
  for(int h = 32; h < HB; h <<= 1){
    sm[i] = v;
    __syncthreads();
    const float o = sm[i ^ h];
    __syncthreads();
    v = (i & h) ? (o - v) : (v + o);
  }
  return v;
}

__device__ __forceinline__ void tcq_reg(float v, int col0, int n, unsigned int* __restrict__ XQ, float* __restrict__ XS,
                                        int* __restrict__ XA, int NG, float* qm, int* qs){
  /* thread i of the block holds column col0 + i; groups of 128 = 4 warps; k_tc_q8's numbers */
  const int i = threadIdx.x, lane = i & 31, w = i >> 5, wb = w & ~3;
  float a = fabsf(v);
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  if(lane == 0) qm[w] = a;
  __syncthreads();
  const float amax = fmaxf(fmaxf(qm[wb], qm[wb + 1]), fmaxf(qm[wb + 2], qm[wb + 3]));
  const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
  const int qi = __float2int_rn(v * id);
  int sum = qi;
  for(int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if(lane == 0) qs[w] = sum;
  unsigned int word = ((unsigned int)qi & 255u) << (8 * (lane & 3));
  word |= __shfl_xor_sync(0xffffffffu, word, 1);
  word |= __shfl_xor_sync(0xffffffffu, word, 2);
  const int g = (col0 >> 7) + (w >> 2), col = (w & 3) * 32 + lane;
  if((lane & 3) == 0){
    const int k4 = col >> 2, s = k4 >> 3, hh = (k4 >> 2) & 1, t = k4 & 3;
    XQ[(((long long)n * NG + g) * 4 + t) * 8 + 2 * s + hh] = word;
  }
  __syncthreads();
  if((w & 3) == 0 && lane == 0){
    XS[(long long)n * NG + g] = amax / 127.0f;
    XA[(long long)n * NG + g] = qs[wb] + qs[wb + 1] + qs[wb + 2] + qs[wb + 3];
  }
}

template <int HB>
__device__ __forceinline__ void rmsfold2(const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ xw,
    const float* __restrict__ w, const float* __restrict__ sg, float* __restrict__ xn, float* __restrict__ xo, int C,
    int add, int T_, int s_x, int s_r, int s_xw, int s_xn, int s_xo, unsigned int* __restrict__ XQ,
    float* __restrict__ XS, int* __restrict__ XA){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  x += t_ * s_x; r += t_ * s_r; xw += t_ * s_xw; xn += t_ * s_xn; xo += t_ * s_xo;
  __shared__ float sm[HB];
  __shared__ double red[32];
  __shared__ float bscale, qm[32];
  __shared__ int qs[32];
  const int i = threadIdx.x, lane = i & 31, wp = i >> 5, c0 = blockIdx.x * HB;
  double acc = 0.0;
  float xmine = 0.f;
  if(add <= 8){
    /* the loads of 4 chunks x 8 parts at a time (in flight together), then the same sums in the same order as the
       loop below: the block waits for memory once per 4 chunks instead of once per chunk and part */
    for(int k0 = 0; k0 * HB < C; k0 += 4){
      float xv[4], rv[4][8];
#pragma unroll
      for(int k = 0; k < 4; ++k){
        const int e = i + (k0 + k) * HB;
        xv[k] = e < C ? x[e] : 0.f;
#pragma unroll
        for(int p = 0; p < 8; ++p) rv[k][p] = (e < C && p < add) ? r[(long long)p * C + e] : 0.f;
      }
#pragma unroll
      for(int k = 0; k < 4; ++k){
        if(i + (k0 + k) * HB < C){
          float xf = xv[k];
          if(add){
            float s = rv[k][0];
#pragma unroll
            for(int p = 1; p < 8; ++p) if(p < add) s += rv[k][p];
            xf += s;
          }
          if(k0 + k == (int)blockIdx.x) xmine = xf;
          const double v = (double)xf; acc += v * v;
        }
      }
    }
  } else {
    for(int e = i; e < C; e += HB){
      float xf = x[e];
      if(add){ float rv = r[e]; for(int p = 1; p < add; ++p) rv += r[(long long)p * C + e]; xf += rv; }
      if(e == c0 + i) xmine = xf;
      const double v = (double)xf; acc += v * v;
    }
  }
  acc = wsum_d(acc);
  if(lane == 0) red[wp] = acc;
  __syncthreads();
  if(wp == 0){
    double t = lane < HB / 32 ? red[lane] : 0.0;
    t = wsum_d(t);
    if(lane == 0) bscale = 1.0f / sqrtf((float)(t / C) + 1e-6f);
  }
  __syncthreads();
  const int e = c0 + i;
  const float xf = xmine;
  if(add) xw[e] = xf;
  const float v = (xf * bscale) * w[e];
  xn[e] = v;
  float y = fwht_reg(sg[e] * v, i, HB, sm);
  y = (HB == 512) ? y * 0.044194173824159220878f : y * rsqrtf((float)HB);
  xo[e] = y;
  tcq_reg(y, c0, (int)t_, XQ, XS, XA, C >> 7, qm, qs);
}

#define RF2_ARGS const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ xw, const float* __restrict__ w, \
    const float* __restrict__ sg, float* __restrict__ xn, float* __restrict__ xo, int C, int add, int HB, int T_, int s_x, \
    int s_r, int s_xw, int s_xn, int s_xo, unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA
/* k_smfold2_qT<HB>: k_smfold_qT with a thread per element (grid (C / HB, T), block HB): the same SiLU product and the
   same butterflies in the same order (fwht_reg), so the fold and its int8 rows are bit-identical */
template <int HB>
__device__ __forceinline__ void smfold2(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg,
    float* __restrict__ xo, int perm_hd, int perm_nk, int perm_rep, int T_, int s_a, int s_b, int s_xo, int C,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  a += t_ * s_a; b += t_ * s_b; xo += t_ * s_xo;
  __shared__ float sm[HB];
  __shared__ float qm[32];
  __shared__ int qs[32];
  const int i = threadIdx.x, c0 = blockIdx.x * HB, e = c0 + i;
  int src = e;
  if(perm_rep > 1){
    const int hd = e % perm_hd, grouped = e / perm_hd;
    const int rep = grouped % perm_rep, nk = grouped / perm_rep;
    src = hd + perm_hd * (nk + perm_nk * rep);
  }
  const float xa = a[src];
  const float m = (xa / (1.0f + expf(-xa))) * b[src];
  float y = fwht_reg(sg[e] * m, i, HB, sm);
  y = (HB == 512) ? y * 0.044194173824159220878f : y * rsqrtf((float)HB);
  xo[e] = y;
  tcq_reg(y, c0, (int)t_, XQ, XS, XA, C >> 7, qm, qs);
}

#define SM2_ARGS const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg, float* __restrict__ xo, \
    int perm_hd, int perm_nk, int perm_rep, int HB, int T_, int s_a, int s_b, int s_xo, int C, unsigned int* __restrict__ XQ, \
    float* __restrict__ XS, int* __restrict__ XA
extern "C" __global__ void __launch_bounds__(1024) k_smfold2_1024(SM2_ARGS){
  smfold2<1024>(a, b, sg, xo, perm_hd, perm_nk, perm_rep, T_, s_a, s_b, s_xo, C, XQ, XS, XA); }
extern "C" __global__ void __launch_bounds__(512) k_smfold2_512(SM2_ARGS){
  smfold2<512>(a, b, sg, xo, perm_hd, perm_nk, perm_rep, T_, s_a, s_b, s_xo, C, XQ, XS, XA); }

extern "C" __global__ void __launch_bounds__(1024) k_rmsfold2_1024(RF2_ARGS){
  rmsfold2<1024>(x, r, xw, w, sg, xn, xo, C, add, T_, s_x, s_r, s_xw, s_xn, s_xo, XQ, XS, XA); }
extern "C" __global__ void __launch_bounds__(512) k_rmsfold2_512(RF2_ARGS){
  rmsfold2<512>(x, r, xw, w, sg, xn, xo, C, add, T_, s_x, s_r, s_xw, s_xn, s_xo, XQ, XS, XA); }
