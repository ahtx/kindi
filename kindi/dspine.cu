/* DSpine block drafter (kindi/dspine.py): the kernels. Kindi's own code. */
#include <cuda_fp16.h>
#define NEG_INF __int_as_float(0xff800000)

/* ------------------------------------------------------------------ target features */
/* k_feat_T: grid ((D + 255) / 256, T), 256 threads. out[(t nslot + slot) D + i] = the layer's output of token t: the
   residual stream h plus the layer's pending product parts r (add parts, D apart; add = 0: h alone). The plans call it
   after a feature layer's FFN product; it only reads h and r, so the target's own arithmetic is unchanged. */
extern "C" __global__ void k_feat_T(const float* __restrict__ h, const float* __restrict__ r, float* __restrict__ out,
                                    int D, int add, int slot, int nslot, int T_, int s_h, int s_r){
  const int t = blockIdx.y;
  if(t >= T_) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i >= D) return;
  float v = h[(long long)t * s_h + i];
  const float* q = r + (long long)t * s_r + i;
  for(int p = 0; p < add; ++p) v += q[(long long)p * D];
  out[((long long)t * nslot + slot) * D + i] = v;
}

/* ------------------------------------------------------------------ int4 products, M <= MM tokens */
/* Weights: per output row, K / 512 tiles of 64 words; word w of a tile = lane L = w % 32, half hh = w / 32, and its
   8 nibbles j hold k = L + 32 (8 hh + j) of the tile (q + 8, q in -7 .. 7). Scales: per row, K / 64 halves (group 64
   in k order). X: M rows of K floats (ldx apart). Y[m ldy + row] (fp32), plus beta * Y (beta 0 / 1: accumulate). Block:
   256 threads = 8 warps, 4 rows per warp (32 rows per block); a tile of X (MM x 512 floats) in shared memory. */
template<int MM>
__device__ __forceinline__ void w4mm(const unsigned int* __restrict__ W, const __half* __restrict__ S,
                                     const float* __restrict__ X, int ldx, float* __restrict__ Y, int ldy, int N, int K,
                                     int M, float beta){
  __shared__ float xs[MM][512];
  const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + warp) * 4;
  const int nt = K >> 9, ng = K >> 6;
  float acc[4][MM];
#pragma unroll
  for(int r = 0; r < 4; ++r)
#pragma unroll
    for(int m = 0; m < MM; ++m) acc[r][m] = 0.f;
  for(int t = 0; t < nt; ++t){
    __syncthreads();
    for(int e = threadIdx.x; e < MM * 512; e += 256){
      const int m = e >> 9, k = e & 511;
      xs[m][k] = m < M ? X[(long long)m * ldx + t * 512 + k] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for(int hh = 0; hh < 2; ++hh){
      unsigned int wd[4];
      float sc[4][4];
#pragma unroll
      for(int r = 0; r < 4; ++r){
        const int row = row0 + r;
        if(row < N){
          wd[r] = W[((long long)row * nt + t) * 64 + hh * 32 + L];
#pragma unroll
          for(int q = 0; q < 4; ++q) sc[r][q] = __half2float(S[(long long)row * ng + t * 8 + hh * 4 + q]);
        } else {
          wd[r] = 0x88888888u;
#pragma unroll
          for(int q = 0; q < 4; ++q) sc[r][q] = 0.f;
        }
      }
#pragma unroll
      for(int j = 0; j < 8; ++j){
        const int k = L + 32 * (8 * hh + j);
        float xv[MM];
#pragma unroll
        for(int m = 0; m < MM; ++m) xv[m] = xs[m][k];
#pragma unroll
        for(int r = 0; r < 4; ++r){
          /* q + 8 -> q exactly: the nibble in the low mantissa bits of 2^23, minus 2^23 + 8 (no int-to-float) */
          const float w = (__int_as_float(0x4B000000 | ((wd[r] >> (4 * j)) & 15u)) - 8388616.f) * sc[r][j >> 1];
#pragma unroll
          for(int m = 0; m < MM; ++m) acc[r][m] = fmaf(w, xv[m], acc[r][m]);
        }
      }
    }
  }
#pragma unroll
  for(int r = 0; r < 4; ++r)
#pragma unroll
    for(int m = 0; m < MM; ++m){
      float v = acc[r][m];
#pragma unroll
      for(int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }
  if(L == 0){
#pragma unroll
    for(int r = 0; r < 4; ++r){
      const int row = row0 + r;
      if(row >= N) continue;
#pragma unroll
      for(int m = 0; m < MM; ++m)
        if(m < M){
          float* y = Y + (long long)m * ldy + row;
          *y = acc[r][m] + (beta != 0.f ? beta * *y : 0.f);
        }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) k_w4mm8(const unsigned int* W, const __half* S, const float* X, int ldx,
                                                          float* Y, int ldy, int N, int K, int M, float beta){
  w4mm<8>(W, S, X, ldx, Y, ldy, N, K, M, beta);
}
extern "C" __global__ void __launch_bounds__(256) k_w4mm16(const unsigned int* W, const __half* S, const float* X, int ldx,
                                                           float* Y, int ldy, int N, int K, int M, float beta){
  w4mm<16>(W, S, X, ldx, Y, ldy, N, K, M, beta);
}

/* k_w4pack: grid (N, K / 512), 64 threads: float weights (row-major N x K, already divided by their group scale and
   rounded: integers -7 .. 7) to the tile layout above */
extern "C" __global__ void k_w4pack(const float* __restrict__ Q, unsigned int* __restrict__ W, int N, int K){
  const int row = blockIdx.x, t = blockIdx.y, w = threadIdx.x;
  const int L = w & 31, hh = w >> 5, nt = K >> 9;
  unsigned int v = 0;
  for(int j = 0; j < 8; ++j){
    const int k = t * 512 + L + 32 * (8 * hh + j);
    int q = (int)Q[(long long)row * K + k];
    q = q < -7 ? -7 : (q > 7 ? 7 : q);
    v |= (unsigned int)(q + 8) << (4 * j);
  }
  W[((long long)row * nt + t) * 64 + w] = v;
}

/* ------------------------------------------------------------------ row kernels */
/* k_rms_rows: grid (rows), 256 threads: y = x / sqrt(mean(x^2) + eps) * w over n values per row (rows ld apart;
   w: n values, or per head when nw < n: w[i % nw]) */
extern "C" __global__ void k_rms_rows(const float* __restrict__ x, int ldx, const float* __restrict__ w, int nw,
                                      float* __restrict__ y, int ldy, int n, float eps){
  const float* xr = x + (long long)blockIdx.x * ldx;
  float* yr = y + (long long)blockIdx.x * ldy;
  __shared__ float red[8];
  float s = 0.f;
  for(int i = threadIdx.x; i < n; i += 256){ const float v = xr[i]; s += v * v; }
  for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if(threadIdx.x < 32){
    s = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    for(int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if(threadIdx.x == 0) red[0] = s;
  }
  __syncthreads();
  const float inv = rsqrtf(red[0] / n + eps);
  for(int i = threadIdx.x; i < n; i += 256) yr[i] = xr[i] * inv * w[nw < n ? i % nw : i];
}

/* k_headnorm_rope: grid (rows, heads), 128 threads (head size 128): per head, RMS norm with w (128 values) and the
   rotate-half rope at position pos[row] (pos0 + row when pos == 0); x (rows, heads * 128, ld ldx) -> y */
extern "C" __global__ void k_headnorm_rope(const float* __restrict__ x, int ldx, const float* __restrict__ w,
                                           float* __restrict__ y, int ldy, const int* __restrict__ pos, int pos0,
                                           float theta, float eps){
  const int r = blockIdx.x, hd = blockIdx.y, i = threadIdx.x;
  const float v = x[(long long)r * ldx + hd * 128 + i];
  __shared__ float red[4], nv[128];
  float s = v * v;
  for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if((i & 31) == 0) red[i >> 5] = s;
  __syncthreads();
  const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / 128.f + eps);
  nv[i] = v * inv * w[i];
  __syncthreads();
  const int p = pos ? pos[r] : pos0 + r;
  const int f = i & 63;
  /* fp32 angle, as the reference (pos.float() * inv_freq); exp2f for theta^(-2f/128) */
  const float ang = (float)p * exp2f(-(2.f * f / 128.f) * log2f(theta));
  float sn, c;
  sincosf(ang, &sn, &c);
  const float a = nv[f], b = nv[f + 64];
  y[(long long)r * ldy + hd * 128 + i] = i < 64 ? a * c - b * sn : b * c + a * sn;
}

/* k_silu_mul: y = silu(a) * b, n values */
extern "C" __global__ void k_silu_mul(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y,
                                      int n){
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n){ const float g = a[i]; y[i] = g / (1.f + __expf(-g)) * b[i]; }
}

/* k_ring_store: grid (rows), 256 threads: rows of 1024 floats (8 K/V heads x 128) -> fp16 ring slots slot[r] */
extern "C" __global__ void k_ring_store(const float* __restrict__ x, int ldx, __half* __restrict__ ring,
                                        const int* __restrict__ slot, int slot0){
  const int r = blockIdx.x;
  const long long s = slot ? slot[r] : slot0 + r;
  for(int i = threadIdx.x; i < 1024; i += 256) ring[s * 1024 + i] = __float2half(x[(long long)r * ldx + i]);
}

/* ------------------------------------------------------------------ attention of the draft block */
/* Keys: the ring's W slots (context positions rpos[s], -1 empty) then T block slots (W .. W + T - 1). Query t of the
   block (position p0 + t) sees ring slot s when p0 - win <= rpos[s] < p0, and block slot j <= t. Q: (T, 32 heads x 128),
   already normed and roped; the 4 query heads of K/V head g are 4 g .. 4 g + 3.
   k_att_part: grid (8, nch), 256 threads; chunk c = key slots c 256 .. ; per (query head, token) the chunk's max m,
   exp-sum l and sum of exp(s - m) V (fp32). */
#define ATT_Q 32        /* 4 heads x at most 8 tokens (the verify plans stop at 8) */
extern "C" __global__ void __launch_bounds__(256) k_att_part(const float* __restrict__ Q, int ldq,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, const int* __restrict__ rpos, int W, int T, int p0,
    int win, float scale, float* __restrict__ Op, float* __restrict__ Mp, float* __restrict__ Lp, int nch){
  const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x;
  const int nq = 4 * T;
  __shared__ __half2 qs[ATT_Q][64];
  __shared__ float ps[ATT_Q][256];
  __shared__ float mx[ATT_Q], sm[ATT_Q];
  {                                     /* a chunk with no visible key (an empty or stale part of the ring): skip */
    const int s0 = c * 256 + tid;
    bool any = s0 >= W && s0 < W + T;
    if(s0 < W){ const int rp = rpos[s0]; any = rp >= 0 && rp < p0 && rp >= p0 - win; }
    if(!__syncthreads_or(any)){
      if(tid < nq){ const long long o = (long long)(g * nch + c) * ATT_Q + tid; Mp[o] = NEG_INF; Lp[o] = 0.f; }
      return;
    }
  }
  for(int e = tid; e < nq * 64; e += 256){
    const int q = e >> 6, d = (e & 63) * 2, hq = q / T, t = q - hq * T;
    const float* qr = Q + (long long)t * ldq + (4 * g + hq) * 128 + d;
    qs[q][d >> 1] = __floats2half2_rn(qr[0] * scale, qr[1] * scale);
  }
  __syncthreads();
  const int s = c * 256 + tid;
  bool okk = s < W + T;
  int jb = -1;
  if(okk){
    if(s < W){ const int rp = rpos[s]; okk = rp >= 0 && rp < p0 && rp >= p0 - win; }
    else jb = s - W;
  }
  __half2 kv[64];
  if(okk){
    const __half2* kr = (const __half2*)(Kr + ((long long)s * 8 + g) * 128);
#pragma unroll
    for(int d = 0; d < 64; ++d) kv[d] = kr[d];
  }
  for(int q = 0; q < nq; ++q){
    const int t = q % T;
    float sc = NEG_INF;
    if(okk && (jb < 0 || jb <= t)){
      sc = 0.f;
#pragma unroll
      for(int d = 0; d < 64; ++d){
        const float2 a = __half22float2(qs[q][d]), b = __half22float2(kv[d]);
        sc = fmaf(a.x, b.x, fmaf(a.y, b.y, sc));
      }
    }
    ps[q][tid] = sc;
  }
  __syncthreads();
  /* per query: max and exp-sum over the chunk (one warp per query, 8 warps) */
  const int wp = tid >> 5, ln = tid & 31;
  for(int q = wp; q < nq; q += 8){
    float m = NEG_INF;
    for(int i = ln; i < 256; i += 32) m = fmaxf(m, ps[q][i]);
    for(int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float l = 0.f;
    for(int i = ln; i < 256; i += 32){
      const float e = m == NEG_INF ? 0.f : __expf(ps[q][i] - m);
      ps[q][i] = e;
      l += e;
    }
    for(int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(0xffffffffu, l, o);
    if(ln == 0){ mx[q] = m; sm[q] = l; }
  }
  __syncthreads();
  /* sum of p V: thread = (dimension d, half of the queries) */
  const int d = tid & 127, qh = tid >> 7;
  const int q0 = qh * ((nq + 1) / 2), q1 = min(nq, q0 + (nq + 1) / 2);
  float acc[16];
#pragma unroll
  for(int i = 0; i < 16; ++i) acc[i] = 0.f;
  const int smax = min(256, W + T - c * 256);
  for(int i = 0; i < smax; ++i){
    const int ss = c * 256 + i;
    const float v = __half2float(Vr[((long long)ss * 8 + g) * 128 + d]);
#pragma unroll
    for(int k = 0; k < 16; ++k)
      if(q0 + k < q1) acc[k] = fmaf(ps[q0 + k][i], v, acc[k]);
  }
  for(int k = 0; k < 16 && q0 + k < q1; ++k){
    const int q = q0 + k;
    const long long o = ((long long)(g * nch + c) * ATT_Q + q);
    Op[o * 128 + d] = acc[k];
    if(d == 0){ Mp[o] = mx[q]; Lp[o] = sm[q]; }
  }
}

/* k_att_comb: grid (8, 4 T), 128 threads: the chunks merged; out (T, 32 x 128) */
extern "C" __global__ void k_att_comb(const float* __restrict__ Op, const float* __restrict__ Mp, const float* __restrict__ Lp,
                                      int nch, int T, float* __restrict__ out, int ldo){
  const int g = blockIdx.x, q = blockIdx.y, d = threadIdx.x;
  float m = NEG_INF;
  for(int c = 0; c < nch; ++c) m = fmaxf(m, Mp[(long long)(g * nch + c) * ATT_Q + q]);
  float l = 0.f, a = 0.f;
  for(int c = 0; c < nch; ++c){
    const long long o = (long long)(g * nch + c) * ATT_Q + q;
    const float mc = Mp[o];
    if(mc == NEG_INF) continue;
    const float e = __expf(mc - m);
    l += e * Lp[o];
    a += e * Op[o * 128 + d];
  }
  const int hq = q / T, t = q - hq * T;
  out[(long long)t * ldo + (4 * g + hq) * 128 + d] = l > 0.f ? a / l : 0.f;
}

/* ------------------------------------------------------------------ host tables (mapped memory) */
/* k_gather_h: grid (n), 256 threads: rows idx[i] of a host fp16 table (UVA pointer, row length C) -> fp32 y * mul */
extern "C" __global__ void k_gather_h(unsigned long long tab, const int* __restrict__ idx, int C, float mul,
                                      float* __restrict__ y){
  const __half* t = (const __half*)tab + (long long)idx[blockIdx.x] * C;
  for(int i = threadIdx.x; i < C; i += 256) y[(long long)blockIdx.x * C + i] = __half2float(t[i]) * mul;
}

/* k_gather_tok: grid (1), 256 threads: the row tok of a host fp16 table -> fp32 y * mul (tok a plan argument) */
extern "C" __global__ void k_gather_tok(unsigned long long tab, int tok, int C, float mul, float* __restrict__ y){
  const __half* t = (const __half*)tab + (long long)tok * C;
  for(int i = threadIdx.x; i < C; i += 256) y[i] = __half2float(t[i]) * mul;
}

/* ------------------------------------------------------------------ fp16 products, M <= MM tokens */
/* Y[m ldy + n] (+)= sum_k W[n K + k] X[m ldx + k]; W fp16 row-major (N x K, K a multiple of 512). 8 warps x 4 rows. */
template<int MM>
__device__ __forceinline__ void h16mm(const __half* __restrict__ W, const float* __restrict__ X, int ldx,
                                      float* __restrict__ Y, int ldy, int N, int K, int M, float beta){
  __shared__ float2 xs[MM][256];
  const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + warp) * 4;
  float acc[4][MM];
#pragma unroll
  for(int r = 0; r < 4; ++r)
#pragma unroll
    for(int m = 0; m < MM; ++m) acc[r][m] = 0.f;
  for(int t = 0; t < (K >> 9); ++t){
    __syncthreads();
    for(int e = threadIdx.x; e < MM * 256; e += 256){
      const int m = e >> 8, k = e & 255;
      xs[m][k] = m < M ? *(const float2*)(X + (long long)m * ldx + t * 512 + 2 * k) : make_float2(0.f, 0.f);
    }
    __syncthreads();
#pragma unroll
    for(int j = 0; j < 8; ++j){
      const int kk = j * 32 + L;
      float2 xv[MM];
#pragma unroll
      for(int m = 0; m < MM; ++m) xv[m] = xs[m][kk];
#pragma unroll
      for(int r = 0; r < 4; ++r){
        const int row = row0 + r;
        const float2 w = row < N ? __half22float2(*(const __half2*)(W + (long long)row * K + t * 512 + 2 * kk))
                                 : make_float2(0.f, 0.f);
#pragma unroll
        for(int m = 0; m < MM; ++m) acc[r][m] = fmaf(w.x, xv[m].x, fmaf(w.y, xv[m].y, acc[r][m]));
      }
    }
  }
#pragma unroll
  for(int r = 0; r < 4; ++r)
#pragma unroll
    for(int m = 0; m < MM; ++m){
      float v = acc[r][m];
#pragma unroll
      for(int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }
  if(L == 0){
#pragma unroll
    for(int r = 0; r < 4; ++r){
      const int row = row0 + r;
      if(row >= N) continue;
#pragma unroll
      for(int m = 0; m < MM; ++m)
        if(m < M){
          float* y = Y + (long long)m * ldy + row;
          *y = acc[r][m] + (beta != 0.f ? beta * *y : 0.f);
        }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) k_h16mm8(const __half* W, const float* X, int ldx, float* Y, int ldy,
                                                           int N, int K, int M, float beta){
  h16mm<8>(W, X, ldx, Y, ldy, N, K, M, beta);
}
extern "C" __global__ void __launch_bounds__(256) k_h16mm16(const __half* W, const float* X, int ldx, float* Y, int ldy,
                                                            int N, int K, int M, float beta){
  h16mm<16>(W, X, ldx, Y, ldy, N, K, M, beta);
}

/* ------------------------------------------------------------------ the code bridge (block rows) */
__device__ __forceinline__ float block_sum256(float s, float* red){
  for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __syncthreads();
  if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  float t = 0.f;
  for(int i = 0; i < 8; ++i) t += red[i];
  return t;
}

/* k_bridge_a: grid (T), 256 threads: r = rms(h) (floored at FLT_MIN), nrm = h / (r + eps) */
extern "C" __global__ void k_bridge_a(const float* __restrict__ h, int H, float eps, float* __restrict__ nrm,
                                      float* __restrict__ rr){
  __shared__ float red[8];
  const float* x = h + (long long)blockIdx.x * H;
  float s = 0.f;
  for(int i = threadIdx.x; i < H; i += 256) s += x[i] * x[i];
  const float r = sqrtf(fmaxf(block_sum256(s, red) / H, 1.1754943508222875e-38f));
  if(threadIdx.x == 0) rr[blockIdx.x] = r;
  for(int i = threadIdx.x; i < H; i += 256) nrm[(long long)blockIdx.x * H + i] = x[i] / (r + eps);
}

/* k_bridge_codes: grid (T), 256 threads: codes = sqc * proj / (|proj| + eps); row 0 = sqc * cb0 (the anchor code) */
extern "C" __global__ void k_bridge_codes(const float* __restrict__ proj, int C, float sqc, float eps,
                                          const float* __restrict__ cb0, float* __restrict__ codes){
  __shared__ float red[8];
  const int t = blockIdx.x;
  if(t == 0){
    for(int i = threadIdx.x; i < C; i += 256) codes[i] = sqc * cb0[i];
    return;
  }
  const float* p = proj + (long long)t * C;
  float s = 0.f;
  for(int i = threadIdx.x; i < C; i += 256) s += p[i] * p[i];
  const float nr = sqrtf(block_sum256(s, red));
  for(int i = threadIdx.x; i < C; i += 256) codes[(long long)t * C + i] = sqc * p[i] / (nr + eps);
}

/* k_bridge_msg: grid (T), 256 threads: pk rows = (gate | predecessor gate | prev), R each; msg[0] = 0,
   msg[t] = sigmoid(gate[t] + pgate[t - 1] + b) * prev[t - 1] */
extern "C" __global__ void k_bridge_msg(const float* __restrict__ pk, const float* __restrict__ b, int R,
                                        float* __restrict__ msg){
  const int t = blockIdx.x;
  for(int j = threadIdx.x; j < R; j += 256){
    float v = 0.f;
    if(t > 0){
      const float z = pk[(long long)t * 3 * R + j] + pk[(long long)(t - 1) * 3 * R + R + j] + b[j];
      v = pk[(long long)(t - 1) * 3 * R + 2 * R + j] / (1.f + __expf(-z));
    }
    msg[(long long)t * R + j] = v;
  }
}

/* k_bridge_out: grid (T), 256 threads: out = h + r delta (row 0: h) */
extern "C" __global__ void k_bridge_out(const float* __restrict__ h, const float* __restrict__ delta,
                                        const float* __restrict__ rr, int H, float* __restrict__ out){
  const int t = blockIdx.x;
  const float r = t > 0 ? rr[t] : 0.f;
  for(int i = threadIdx.x; i < H; i += 256){
    const long long o = (long long)t * H + i;
    out[o] = h[o] + r * delta[o];
  }
}

/* ------------------------------------------------------------------ top 16 of each row */
__device__ __forceinline__ bool better(float a, int ia, float b, int ib){ return a > b || (a == b && ia < ib); }

/* k_top16a: grid (n, NB), 256 threads: each block's 16 best (value, index) of its slice of the row (<= 4096 values) */
extern "C" __global__ void k_top16a(const float* __restrict__ Y, int ldy, int V, float* __restrict__ bv,
                                    int* __restrict__ bi){
  const int r = blockIdx.x, b = blockIdx.y, nb = gridDim.y, tid = threadIdx.x;
  const int per = (V + nb - 1) / nb, lo = b * per, hi = min(V, lo + per);
  const float* y = Y + (long long)r * ldy;
  float v[16];
  int ix[16];
#pragma unroll
  for(int c = 0; c < 16; ++c){
    const int i = lo + tid + 256 * c;
    v[c] = i < hi ? y[i] : NEG_INF;
    ix[c] = i < hi ? i : 0x7fffffff;
  }
  __shared__ float sv[8];
  __shared__ int si[8];
  for(int k = 0; k < 16; ++k){
    float bvv = NEG_INF;
    int bii = 0x7fffffff;
#pragma unroll
    for(int c = 0; c < 16; ++c) if(better(v[c], ix[c], bvv, bii)){ bvv = v[c]; bii = ix[c]; }
    for(int o = 16; o > 0; o >>= 1){
      const float ov = __shfl_xor_sync(0xffffffffu, bvv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bii, o);
      if(better(ov, oi, bvv, bii)){ bvv = ov; bii = oi; }
    }
    __syncthreads();
    if((tid & 31) == 0){ sv[tid >> 5] = bvv; si[tid >> 5] = bii; }
    __syncthreads();
    bvv = sv[0]; bii = si[0];
    for(int w = 1; w < 8; ++w) if(better(sv[w], si[w], bvv, bii)){ bvv = sv[w]; bii = si[w]; }
#pragma unroll
    for(int c = 0; c < 16; ++c) if(ix[c] == bii){ v[c] = NEG_INF; ix[c] = 0x7fffffff; }
    if(tid == 0){ bv[((long long)r * nb + b) * 16 + k] = bvv; bi[((long long)r * nb + b) * 16 + k] = bii; }
  }
}

/* k_top16b: grid (n), 1024 threads (= NB x 16 candidates): the row's 16 best, as token ids in ascending order */
extern "C" __global__ void k_top16b(const float* __restrict__ bv, const int* __restrict__ bi, int ncand,
                                    int* __restrict__ cand){
  const int r = blockIdx.x, tid = threadIdx.x;
  float v = tid < ncand ? bv[(long long)r * ncand + tid] : NEG_INF;
  int ix = tid < ncand ? bi[(long long)r * ncand + tid] : 0x7fffffff;
  __shared__ float sv[32];
  __shared__ int si[32];
  __shared__ int top[16];
  for(int k = 0; k < 16; ++k){
    float bvv = v;
    int bii = ix;
    for(int o = 16; o > 0; o >>= 1){
      const float ov = __shfl_xor_sync(0xffffffffu, bvv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bii, o);
      if(better(ov, oi, bvv, bii)){ bvv = ov; bii = oi; }
    }
    __syncthreads();
    if((tid & 31) == 0){ sv[tid >> 5] = bvv; si[tid >> 5] = bii; }
    __syncthreads();
    if(tid < 32){
      bvv = sv[tid]; bii = si[tid];
      for(int o = 16; o > 0; o >>= 1){
        const float ov = __shfl_xor_sync(0xffffffffu, bvv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bii, o);
        if(better(ov, oi, bvv, bii)){ bvv = ov; bii = oi; }
      }
      if(tid == 0) top[k] = bii;
    }
    __syncthreads();
    if(ix == top[k]){ v = NEG_INF; ix = 0x7fffffff; }
  }
  if(tid == 0){
    int a[16];
    for(int k = 0; k < 16; ++k) a[k] = top[k];
    for(int i = 1; i < 16; ++i){ const int x = a[i]; int j = i - 1; while(j >= 0 && a[j] > x){ a[j + 1] = a[j]; --j; } a[j + 1] = x; }
    for(int k = 0; k < 16; ++k) cand[r * 16 + k] = a[k];
  }
}

/* ------------------------------------------------------------------ the tail (top-16 refinement chain) */
/* k_tail_base: grid (n), 32 threads: base[t][k] = r[t] * Y2[t][cand[t][k]] (r: the raw rows rms, from row 1) */
extern "C" __global__ void k_tail_base(const float* __restrict__ Y2, int ldy, const int* __restrict__ cand,
                                       const float* __restrict__ rr, float* __restrict__ base){
  const int t = blockIdx.x, k = threadIdx.x;
  if(k < 16) base[t * 16 + k] = rr[t] * Y2[(long long)t * ldy + cand[t * 16 + k]];
}

/* k_gather_sid: grid (n x 16), 256 threads: the codebook row of each candidate predecessor (row 0: the anchor tok,
   row t: candidate k of row t - 1), times mul */
extern "C" __global__ void k_gather_sid(unsigned long long tab, const int* __restrict__ cand, int tok, int C, float mul,
                                        float* __restrict__ y){
  const int q = blockIdx.x, t = q >> 4, k = q & 15;
  const int id = t == 0 ? tok : cand[(t - 1) * 16 + k];
  const __half* src = (const __half*)tab + (long long)id * C;
  for(int i = threadIdx.x; i < C; i += 256) y[(long long)q * C + i] = __half2float(src[i]) * mul;
}

/* k_tail_sc: grid (n, 16), 256 threads, R floats of dynamic shared memory: for row t and predecessor a,
   m = sigmoid(q[t] + tvg[t, a] + b) * tvv[t, a]; sc[b] = base[t][b] + r[t] (m . tg[t, b]); idx[t][a] = argmax_b sc
   (ties: the smaller b). q: the gate part of pk row t + 1 (ld 3R); tsv rows (t 16 + a): (tvg | tvv); tg rows
   (t 16 + b): R values. */
extern "C" __global__ void k_tail_sc(const float* __restrict__ pk, const float* __restrict__ tsv, const float* __restrict__ b_,
                                     const float* __restrict__ tg, const float* __restrict__ base, const float* __restrict__ rr,
                                     int R, int* __restrict__ idx){
  const int t = blockIdx.x, a = blockIdx.y, tid = threadIdx.x;
  __shared__ float m[512];
  __shared__ float sc[16];
  const float* q = pk + (long long)(t + 1) * 3 * R;
  const float* tv = tsv + (long long)(t * 16 + a) * 2 * R;
  for(int j = tid; j < R; j += 256) m[j] = tv[R + j] / (1.f + __expf(-(q[j] + tv[j] + b_[j])));
  __syncthreads();
  const int w = tid >> 5, L = tid & 31;
  for(int bb = w; bb < 16; bb += 8){
    const float* g = tg + (long long)(t * 16 + bb) * R;
    float s = 0.f;
    for(int j = L; j < R; j += 32) s += m[j] * g[j];
    for(int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if(L == 0) sc[bb] = base[t * 16 + bb] + rr[t] * s;
  }
  __syncthreads();
  if(tid == 0){
    int best = 0;
    for(int bb = 1; bb < 16; ++bb) if(sc[bb] > sc[best]) best = bb;
    idx[t * 16 + a] = best;
  }
}

/* k_chain: one thread: d[i] = cand[i][j_i], j_0 = idx[0][0], j_{i+1} = idx[i+1][j_i] */
extern "C" __global__ void k_chain(const int* __restrict__ idx, const int* __restrict__ cand, int n, int* __restrict__ d){
  if(threadIdx.x | blockIdx.x) return;
  int j = idx[0];
  for(int i = 0; i < n; ++i){
    d[i] = cand[i * 16 + j];
    if(i + 1 < n) j = idx[(i + 1) * 16 + j];
  }
}

/* ------------------------------------------------------------------ the context ring */
/* k_ring_mod: grid (rows), 256 threads: row r (1024 floats) -> fp16 ring slot (pos0 + r) mod W */
extern "C" __global__ void k_ring_mod(const float* __restrict__ x, int ldx, __half* __restrict__ ring, int pos0, int W){
  const int r = blockIdx.x;
  const long long s = (pos0 + r) % W;
  for(int i = threadIdx.x; i < 1024; i += 256) ring[s * 1024 + i] = __float2half(x[(long long)r * ldx + i]);
}

/* k_rpos_set: grid (1), 256 threads: rpos[(pos0 + i) mod W] = pos0 + i, i < n */
extern "C" __global__ void k_rpos_set(int* __restrict__ rpos, int pos0, int n, int W){
  for(int i = threadIdx.x; i < n; i += 256) rpos[(pos0 + i) % W] = pos0 + i;
}

/* k_rpos_cut: grid (W / 256), 256 threads: forget the positions >= pos (a rewind) */
extern "C" __global__ void k_rpos_cut(int* __restrict__ rpos, int pos, int W){
  const int i = blockIdx.x * 256 + threadIdx.x;
  if(i < W && rpos[i] >= pos) rpos[i] = -1;
}

/* ------------------------------------------------------------------ int4 x int8 products (dp4a), M <= MM tokens */
/* Activations, per token and group of 64: int8 values q = rint(x / s), s = max|x| / 127, packed as 16 words in the
   order the weight nibbles unpack to (word 2w: k = 8w, 8w+2, 8w+4, 8w+6; word 2w+1: k = 8w+1, 8w+3, ..), the scale
   s and the sum of the 64 values (for the weights' + 8 offset).
   k_q8act: grid (K / 64, M), 64 threads. */
extern "C" __global__ void k_q8act(const float* __restrict__ X, int ldx, int K, unsigned int* __restrict__ Xq,
                                   float* __restrict__ Xs, int* __restrict__ Xsum){
  const int g = blockIdx.x, m = blockIdx.y, k = threadIdx.x, G = K >> 6;
  const float x = X[(long long)m * ldx + g * 64 + k];
  float a = fabsf(x);
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  __shared__ float am[2];
  __shared__ int qs[64], sm[2];
  if((k & 31) == 0) am[k >> 5] = a;
  __syncthreads();
  const float mx = fmaxf(am[0], am[1]);
  const float s = mx > 0.f ? mx / 127.f : 1.f;
  int q = __float2int_rn(x / s);
  q = q > 127 ? 127 : (q < -127 ? -127 : q);
  qs[k] = q;
  int t = q;
  for(int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  if((k & 31) == 0) sm[k >> 5] = t;
  __syncthreads();
  const long long og = (long long)m * G + g;
  if(k < 16){                        /* word k: w = k / 2, odd = k & 1 -> k values 8w + odd + 2i */
    const int w = k >> 1, od = k & 1;
    unsigned int v = 0;
    for(int i = 0; i < 4; ++i) v |= ((unsigned int)(qs[8 * w + od + 2 * i] & 255)) << (8 * i);
    Xq[og * 16 + k] = v;
  }
  if(k == 0){ Xs[og] = s; Xsum[og] = sm[0] + sm[1]; }
}

/* k_q4mm: Y[m ldy + n] (+)= sum_k W[n][k] x[m][k]; W: row-major nibbles (q + 8, byte b = k 2b low, 2b+1 high), scales
   S[n][K / 64] fp16; x from k_q8act. Block: 8 warps x 4 rows; a lane takes whole groups (16 dp4a per group and token),
   tiles of 32 groups of the activations in shared memory (word-major: lane L reads group L, no bank conflicts). */
__device__ __forceinline__ int dp4a_(int a, int b, int c){
  int d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

template<int MM, int RW>
__device__ __forceinline__ void q4mm(const unsigned int* __restrict__ W, const __half* __restrict__ S,
                                     const unsigned int* __restrict__ Xq, const float* __restrict__ Xs,
                                     const int* __restrict__ Xsum, float* __restrict__ Y, int ldy, int N, int K, int M,
                                     float beta){
  __shared__ unsigned int xs[MM][16][32];
  __shared__ float xsc[MM][32];
  __shared__ int xsm[MM][32];
  const int warp = threadIdx.x >> 5, L = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + warp) * RW;
  const int G = K >> 6;
  float acc[RW][MM];
#pragma unroll
  for(int r = 0; r < RW; ++r)
#pragma unroll
    for(int m = 0; m < MM; ++m) acc[r][m] = 0.f;
  for(int g0 = 0; g0 < G; g0 += 32){
    __syncthreads();
    for(int e = threadIdx.x; e < MM * 32 * 16; e += 256){
      const int m = e >> 9, gl = (e >> 4) & 31, w = e & 15, g = g0 + gl;
      xs[m][w][gl] = (m < M && g < G) ? Xq[((long long)m * G + g) * 16 + w] : 0u;
    }
    for(int e = threadIdx.x; e < MM * 32; e += 256){
      const int m = e >> 5, gl = e & 31, g = g0 + gl;
      const bool ok = m < M && g < G;
      xsc[m][gl] = ok ? Xs[(long long)m * G + g] : 0.f;
      xsm[m][gl] = ok ? Xsum[(long long)m * G + g] : 0;
    }
    __syncthreads();
    const int g = g0 + L;
    if(g >= G) continue;
    unsigned int lo[RW][8], hi[RW][8];
    float sw[RW];
#pragma unroll
    for(int r = 0; r < RW; ++r){
      const int row = row0 + r;
      if(row < N){
        const uint4* p = (const uint4*)(W + ((long long)row * G + g) * 8);
        const uint4 a = p[0], b = p[1];
        const unsigned int wd[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for(int w = 0; w < 8; ++w){ lo[r][w] = wd[w] & 0x0F0F0F0Fu; hi[r][w] = (wd[w] >> 4) & 0x0F0F0F0Fu; }
        sw[r] = __half2float(S[(long long)row * G + g]);
      } else {
#pragma unroll
        for(int w = 0; w < 8; ++w){ lo[r][w] = 0x08080808u; hi[r][w] = 0x08080808u; }
        sw[r] = 0.f;
      }
    }
#pragma unroll
    for(int m = 0; m < MM; ++m){
      if(m >= M) break;
      unsigned int xv[16];
#pragma unroll
      for(int w = 0; w < 16; ++w) xv[w] = xs[m][w][L];
      const float xsv = xsc[m][L];
      const int off = 8 * xsm[m][L];
#pragma unroll
      for(int r = 0; r < RW; ++r){
        int s = 0;
#pragma unroll
        for(int w = 0; w < 8; ++w){
          s = dp4a_((int)lo[r][w], (int)xv[2 * w], s);
          s = dp4a_((int)hi[r][w], (int)xv[2 * w + 1], s);
        }
        acc[r][m] = fmaf((float)(s - off), sw[r] * xsv, acc[r][m]);
      }
    }
  }
#pragma unroll
  for(int r = 0; r < RW; ++r)
#pragma unroll
    for(int m = 0; m < MM; ++m){
      float v = acc[r][m];
#pragma unroll
      for(int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      acc[r][m] = v;
    }
  if(L == 0){
#pragma unroll
    for(int r = 0; r < RW; ++r){
      const int row = row0 + r;
      if(row >= N) continue;
#pragma unroll
      for(int m = 0; m < MM; ++m)
        if(m < M){
          float* y = Y + (long long)m * ldy + row;
          *y = acc[r][m] + (beta != 0.f ? beta * *y : 0.f);
        }
    }
  }
}
extern "C" __global__ void __launch_bounds__(256) k_q4mm8(const unsigned int* W, const __half* S, const unsigned int* Xq,
                                                          const float* Xs, const int* Xsum, float* Y, int ldy, int N,
                                                          int K, int M, float beta){
  q4mm<8, 4>(W, S, Xq, Xs, Xsum, Y, ldy, N, K, M, beta);
}
extern "C" __global__ void __launch_bounds__(256) k_q4mm16(const unsigned int* W, const __half* S, const unsigned int* Xq,
                                                           const float* Xs, const int* Xsum, float* Y, int ldy, int N,
                                                           int K, int M, float beta){
  q4mm<16, 4>(W, S, Xq, Xs, Xsum, Y, ldy, N, K, M, beta);
}
/* one row per warp (8 per block): the narrow products (N <= 4096) fill more of the card */
extern "C" __global__ void __launch_bounds__(256) k_q4mm8n(const unsigned int* W, const __half* S, const unsigned int* Xq,
                                                           const float* Xs, const int* Xsum, float* Y, int ldy, int N,
                                                           int K, int M, float beta){
  q4mm<8, 1>(W, S, Xq, Xs, Xsum, Y, ldy, N, K, M, beta);
}
extern "C" __global__ void __launch_bounds__(256) k_q4mm16n(const unsigned int* W, const __half* S, const unsigned int* Xq,
                                                            const float* Xs, const int* Xsum, float* Y, int ldy, int N,
                                                            int K, int M, float beta){
  q4mm<16, 1>(W, S, Xq, Xs, Xsum, Y, ldy, N, K, M, beta);
}
