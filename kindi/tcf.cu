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

extern "C" __global__ void k_rmsfold_qT(const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ xw,
    const float* __restrict__ w, const float* __restrict__ sg, float* __restrict__ xn, float* __restrict__ xo, int C,
    int add, int HB, int T_, int s_x, int s_r, int s_xw, int s_xn, int s_xo,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  d_k_rmsfold(x + t_ * s_x, r + t_ * s_r, xw + t_ * s_xw, w, sg, xn + t_ * s_xn, xo + t_ * s_xo, C, add, HB);
  __syncthreads();
  const int hb = HB < 0 ? 512 : HB;
  tcq_chunk(xo + t_ * s_xo, blockIdx.x * hb, hb, XQ, XS, XA, C >> 7, (int)t_);
}

extern "C" __global__ void k_smfold_qT(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg,
    float* __restrict__ xo, int perm_hd, int perm_nk, int perm_rep, int HB, int T_, int s_a, int s_b, int s_xo, int C,
    unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
  d_k_smfold(a + t_ * s_a, b + t_ * s_b, sg, xo + t_ * s_xo, perm_hd, perm_nk, perm_rep, HB);
  __syncthreads();
  const int hb = HB < 0 ? 512 : HB;
  tcq_chunk(xo + t_ * s_xo, blockIdx.x * hb, hb, XQ, XS, XA, C >> 7, (int)t_);
}

extern "C" __global__ void k_fold_fwht_qT(const float* __restrict__ x, const float* __restrict__ sg, float* __restrict__ xo,
    int HB, int T_, int s_x, int s_xo, int C, unsigned int* __restrict__ XQ, float* __restrict__ XS, int* __restrict__ XA){
  if((int)blockIdx.y >= T_) return;
  const long long t_ = blockIdx.y;
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
#define FD_CH 512
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
  __shared__ float S[G][FD_CH];
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
  for(int base = wp * 4; base < FD_CH; base += 32){
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
    for(int i = lane; i < FD_CH; i += 32){
      const float e = i < len ? expf(S[wp][i] - m) : 0.f;
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
  for(int e = i; e < C; e += HB){
    float xf = x[e];
    if(add){ float rv = r[e]; for(int p = 1; p < add; ++p) rv += r[(long long)p * C + e]; xf += rv; }
    const double v = (double)xf; acc += v * v;
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
  float xf = x[e];
  if(add){ float rv = r[e]; for(int p = 1; p < add; ++p) rv += r[(long long)p * C + e]; xf += rv; xw[e] = xf; }
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
extern "C" __global__ void __launch_bounds__(1024) k_rmsfold2_1024(RF2_ARGS){
  rmsfold2<1024>(x, r, xw, w, sg, xn, xo, C, add, T_, s_x, s_r, s_xw, s_xn, s_xo, XQ, XS, XA); }
extern "C" __global__ void __launch_bounds__(512) k_rmsfold2_512(RF2_ARGS){
  rmsfold2<512>(x, r, xw, w, sg, xn, xo, C, add, T_, s_x, s_r, s_xw, s_xn, s_xo, XQ, XS, XA); }
