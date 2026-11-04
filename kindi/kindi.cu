/* HIP (ROCm) or CUDA: KHIP is set only when hipRTC compiles this file. CUDA builds keep their exact
   instructions; the HIP alternatives sit beside them. */
#ifndef KHIP   /* KINDI-PORT-BEGIN */
#if defined(__HIPCC_RTC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIP_PLATFORM_AMD__)
#define KHIP 1
#define KSHFLX(v, o) __shfl_xor((v), (o))
#define KSHFL(v, l) __shfl((v), (l))
#else
#define KHIP 0
#define KSHFLX(v, o) __shfl_xor_sync(0xffffffffu, (v), (o))
#define KSHFL(v, l) __shfl_sync(0xffffffffu, (v), (l))
#endif
#endif         /* KINDI-PORT-END */

/* kindi.cu - Kindi 0.1.0 CUDA kernels (sm_30 and later: Quadro K420, K1200). Compiled at run time by NVRTC (CuPy).
   Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md. */

/* ---------------------------------------------------------------- helpers */

/* __ldg (read-only data cache) exists from sm_35; the Quadro K420 is sm_30 */
#if KHIP || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350)
#define KLDG(p) (*(p))
#else
#define KLDG(p) __ldg(p)
#endif

/* sm_70+ (Volta and later; the RTX 3080 is sm_86, compiled as compute_75): CUDA 10.2 drops the non-sync warp shuffles
   there. Every __shfl_xor in this file runs with all 32 lanes of the warp active (warp-strided loops, per-warp exits),
   so the full-mask _sync form is the same operation. */
#if !KHIP && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define __shfl_xor(v, o) KSHFLX((v), (o))
#endif


#if KHIP
__device__ __forceinline__ float h2f(unsigned short h){ return __half2float(*(__half*)&h); }
#else
__device__ __forceinline__ float h2f(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
#endif

/* e^x with no special-function unit: x = k ln2 + r with k the nearest integer to x / ln2 and |r| <= ln2 / 2, r found
   with ln2 split in a 16-bit-exact high part and a low part (Cody-Waite), e^r from its Taylor series to r^7 / 7!
   in Horner form, then scaled by 2^k in two halves so that results near the float limits (k = 128, subnormals) stay
   right. Only round-to-nearest fma/mul/add are used, so every compiler version gives the same bits. */
#if !KHIP
__device__ __forceinline__ float kexpf(float x){
    const float INV_LN2 = 1.44269502162933349609375f, LN2_HI = 0.693145751953125f, LN2_LO = 1.428606765330187045e-06f;
    float c = fminf(fmaxf(x, -104.0f), 89.0f);
    float k = rintf(__fmul_rn(c, INV_LN2));
    float r = __fmaf_rn(-k, LN2_HI, c);
    r = __fmaf_rn(-k, LN2_LO, r);
    float p = 1.98412698e-4f;                          // 1/7!
    p = __fmaf_rn(p, r, 1.38888889e-3f);               // 1/6!
    p = __fmaf_rn(p, r, 8.33333377e-3f);               // 1/5!
    p = __fmaf_rn(p, r, 4.16666679e-2f);               // 1/4!
    p = __fmaf_rn(p, r, 0.166666672f);                 // 1/3!
    p = __fmaf_rn(p, r, 0.5f);
    p = __fmaf_rn(p, r, 1.0f);
    p = __fmaf_rn(p, r, 1.0f);
    int ki = (int)k, a = ki >> 1, b = ki - a;
    float y = __fmul_rn(p, __int_as_float((a + 127) << 23));
    y = __fmul_rn(y, __int_as_float((b + 127) << 23));
    return x != x ? x : y;
}
#define expf(x) kexpf(x)
#endif

#if KHIP
__device__ __forceinline__ float hf(unsigned short h){ return __half2float(*(__half*)&h); }
#else
__device__ __forceinline__ float hf(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
#endif

#if KHIP
__device__ __forceinline__ float h2f3(unsigned short h){ return __half2float(*(__half*)&h); }
#else
__device__ __forceinline__ float h2f3(unsigned short h){ float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }
#endif

__device__ __forceinline__ double wsum_d(double v){
  for(int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float wsum_f(float v){
  for(int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ void fwht_n(float* xs, int n, int j){
  /* in-place unnormalized Walsh-Hadamard transform of xs[0..n), n a power of two > 512, by 512 threads */
  for(int h = 1; h < n; h <<= 1){
    for(int t = j; t < (n >> 1); t += 512){
      const int i = (t / h) * 2 * h + (t % h);
      const float a = xs[i], b = xs[i + h];
      xs[i] = a + b; xs[i + h] = a - b;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void fwht_chunks(float* xs, int NC, int j){
  for(int h = 1; h < 512; h <<= 1){
    for(int c = 0; c < NC; c += 2){
      const int cc = c + (j >> 8);
      if(cc < NC){
        const int jj = j & 255;
        const int i = (jj / h) * 2 * h + (jj % h) + (cc << 9);
        const float a = xs[i], b = xs[i + h];
        xs[i] = a + b; xs[i + h] = a - b;
      }
    }
    __syncthreads();
  }
}


/* ---------------------------------------------------------------- ternary GEMV (decode): k_g4 / k_g4l */

#define RB 16
struct Grp { const uint4* W; const unsigned int* S; float* y; int R; int g; };
__device__ __forceinline__ Grp grp(int g, int NG0, int NG1,
    const uint4* W0, const unsigned int* S0, float* y0, int R0, const uint4* W1, const unsigned int* S1, float* y1, int R1,
    const uint4* W2, const unsigned int* S2, float* y2, int R2){
  Grp q; q.W = W0; q.S = S0; q.y = y0; q.R = R0;
  if(g >= NG0){ g -= NG0; q.W = W1; q.S = S1; q.y = y1; q.R = R1;
    if(g >= NG1){ g -= NG1; q.W = W2; q.S = S2; q.y = y2; q.R = R2; } }
  q.g = g;
  return q;
}

extern "C" __global__ void __launch_bounds__(1024) k_g4(
    const uint4* __restrict__ W0, const unsigned int* __restrict__ S0, float* __restrict__ y0, int R0,
    const uint4* __restrict__ W1, const unsigned int* __restrict__ S1, float* __restrict__ y1, int R1,
    const uint4* __restrict__ W2, const unsigned int* __restrict__ S2, float* __restrict__ y2, int R2,
    const float* __restrict__ x, int C){
  __shared__ float T[2 * 81 * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  const int CW = C >> 4, NPH = C >> 9;
  /* gridDim.y > 1: split the columns - block row y covers phases [ph_lo, ph_hi) of every row group and writes its
     partial sums to y + blockIdx.y * R (one tensor only); k_sumparts adds them in a fixed order. gridDim.y == 1:
     all phases, as before. */
  const int ph_lo = (int)((long long)blockIdx.y * NPH / gridDim.y), ph_hi = (int)((long long)(blockIdx.y + 1) * NPH / gridDim.y);
  const int NG0 = R0 >> 4, NG1 = R1 >> 4, NG = NG0 + NG1 + (R2 >> 4);
  const char* Tc = (const char*)T;
  const unsigned int l4 = lane << 2;
  const bool b0 = lane & 1, b1 = lane & 2, b2 = lane & 4, b3 = lane & 8;
  const int rfin = (b0 ? 8 : 0) + (b1 ? 4 : 0) + (b2 ? 2 : 0) + (b3 ? 1 : 0);   /* row this lane ends with */
  const bool wr = (lane & 16) == 0;
  /* the table builders (tid < 384) load the next phase's x while this phase is computed (hides DRAM latency) */
  const int bl = tid & 31, bk = (tid >> 5) & 3;
  float nx0 = 0.f, nx1 = 0.f, nx2 = 0.f, nx3 = 0.f;
  if(tid < 384){
    const float* xp = x + (ph_lo << 9) + (bl << 4) + (bk << 2);
    nx0 = xp[0]; nx1 = xp[1]; nx2 = xp[2]; nx3 = xp[3];
  }
  for(int ph = ph_lo; ph < ph_hi; ++ph){
    __syncthreads();
    if(tid < 384){
      const int l = bl, k = bk, t3 = tid >> 7;
      const float x0 = nx0, x1 = nx1, x2 = nx2, x3 = nx3;
      if(ph + 1 < ph_hi){
        const float* xq = x + ((ph + 1) << 9) + (l << 4) + (k << 2);
        nx0 = xq[0]; nx1 = xq[1]; nx2 = xq[2]; nx3 = xq[3];
      }
      float* Tk = T + (k >> 1) * 5184 + (k & 1) * 32 + l + t3 * 27 * 64;
      const float bb3 = (float)(t3 - 1) * x3;
#pragma unroll
      for(int t2 = 0; t2 < 3; ++t2){
        const float bb2 = bb3 + (float)(t2 - 1) * x2;
#pragma unroll
        for(int t1 = 0; t1 < 3; ++t1){
          const float bb1 = bb2 + (float)(t1 - 1) * x1;
          float* q = Tk + (t2 * 9 + t1 * 3) * 64;
          q[0] = bb1 - x0;
          q[64] = bb1;
          q[128] = bb1 + x0;
        }
      }
    }
    __syncthreads();
    for(int g = blockIdx.x * nw + warp; g < NG; g += gridDim.x * nw){   /* one block per SM; a 16-row group stays in one warp */
      Grp q = grp(g, NG0, NG1, W0, S0, y0, R0, W1, S1, y1, R1, W2, S2, y2, R2);
      q.y += (long long)blockIdx.y * q.R;
      const int rw = (q.g << 4) + rfin;
      const float yo = (ph > ph_lo && wr) ? q.y[rw] : 0.0f;
      uint4 wv[4];
#pragma unroll
      for(int i = 0; i < 4; ++i) wv[i] = q.W[(((long long)q.g * NPH + ph) * 4 + i) * 32 + lane];
      const unsigned int sv = q.S[((long long)q.g * NPH + ph) * 32 + lane];
      float v[16];
#pragma unroll
      for(int i = 0; i < 4; ++i){
        const unsigned int ww[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
#pragma unroll
        for(int j = 0; j < 4; ++j){
          const unsigned int w = ww[j];
          v[4 * i + j] = (*(const float*)(Tc + __byte_perm(w, l4, 0x5504u))
                        + *(const float*)(Tc + 128 + __byte_perm(w, l4, 0x5514u)))
                       + (*(const float*)(Tc + 20736 + __byte_perm(w, l4, 0x5524u))
                        + *(const float*)(Tc + 20864 + __byte_perm(w, l4, 0x5534u)));
        }
      }
      /* xor 1: 16 -> 8 (rows (b0?8:0)+i) */
      float a8[8];
#pragma unroll
      for(int i = 0; i < 8; ++i) a8[i] = (b0 ? v[8 + i] : v[i]) + __shfl_xor(b0 ? v[i] : v[8 + i], 1);
      float a4[4];
#pragma unroll
      for(int i = 0; i < 4; ++i) a4[i] = (b1 ? a8[4 + i] : a8[i]) + __shfl_xor(b1 ? a8[i] : a8[4 + i], 2);
      float a2[2];
#pragma unroll
      for(int i = 0; i < 2; ++i) a2[i] = (b2 ? a4[2 + i] : a4[i]) + __shfl_xor(b2 ? a4[i] : a4[2 + i], 4);
      /* a2[j] = row base+j summed over this lane's 8-lane (one scale) group: scale it */
      a2[0] *= h2f((unsigned short)(sv & 0xFFFFu));
      a2[1] *= h2f((unsigned short)(sv >> 16));
      float d = (b3 ? a2[1] : a2[0]) + __shfl_xor(b3 ? a2[0] : a2[1], 8);
      d += __shfl_xor(d, 16);
      if(wr) q.y[rw] = yo + d;
    }
  }
}

/* shortlist head: k_g4 over a list of 16-row groups, compact output + token id per slot */

extern "C" __global__ void __launch_bounds__(1024) k_g4l(
    const uint4* __restrict__ W0, const unsigned int* __restrict__ S0, float* __restrict__ y0, int R0,
    const uint4* __restrict__ W1, const unsigned int* __restrict__ S1, float* __restrict__ y1, int R1,
    const uint4* __restrict__ W2, const unsigned int* __restrict__ S2, float* __restrict__ y2, int R2,
    const float* __restrict__ x, int C, const int* __restrict__ glist, int ngl,
    const int* __restrict__ permtok, int* __restrict__ tokc){
  __shared__ float T[2 * 81 * 64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
  const int CW = C >> 4, NPH = C >> 9;
  const int NG = ngl;
  const char* Tc = (const char*)T;
  const unsigned int l4 = lane << 2;
  const bool b0 = lane & 1, b1 = lane & 2, b2 = lane & 4, b3 = lane & 8;
  const int rfin = (b0 ? 8 : 0) + (b1 ? 4 : 0) + (b2 ? 2 : 0) + (b3 ? 1 : 0);   /* row this lane ends with */
  const bool wr = (lane & 16) == 0;
  /* the table builders (tid < 384) load the next phase's x while this phase is computed (hides DRAM latency) */
  const int bl = tid & 31, bk = (tid >> 5) & 3;
  float nx0 = 0.f, nx1 = 0.f, nx2 = 0.f, nx3 = 0.f;
  if(tid < 384){
    const float* xp = x + (bl << 4) + (bk << 2);
    nx0 = xp[0]; nx1 = xp[1]; nx2 = xp[2]; nx3 = xp[3];
  }
  for(int ph = 0; ph < NPH; ++ph){
    __syncthreads();
    if(tid < 384){
      const int l = bl, k = bk, t3 = tid >> 7;
      const float x0 = nx0, x1 = nx1, x2 = nx2, x3 = nx3;
      if(ph + 1 < NPH){
        const float* xq = x + ((ph + 1) << 9) + (l << 4) + (k << 2);
        nx0 = xq[0]; nx1 = xq[1]; nx2 = xq[2]; nx3 = xq[3];
      }
      float* Tk = T + (k >> 1) * 5184 + (k & 1) * 32 + l + t3 * 27 * 64;
      const float bb3 = (float)(t3 - 1) * x3;
#pragma unroll
      for(int t2 = 0; t2 < 3; ++t2){
        const float bb2 = bb3 + (float)(t2 - 1) * x2;
#pragma unroll
        for(int t1 = 0; t1 < 3; ++t1){
          const float bb1 = bb2 + (float)(t1 - 1) * x1;
          float* q = Tk + (t2 * 9 + t1 * 3) * 64;
          q[0] = bb1 - x0;
          q[64] = bb1;
          q[128] = bb1 + x0;
        }
      }
    }
    __syncthreads();
    for(int g = blockIdx.x * nw + warp; g < NG; g += gridDim.x * nw){
      Grp q; q.W = W0; q.S = S0; q.y = y0; q.R = R0; q.g = glist[g];
      const int rw = (g << 4) + rfin;
      if(ph == 0 && wr) tokc[rw] = permtok[(q.g << 4) + rfin];
      const float yo = (ph && wr) ? q.y[rw] : 0.0f;
      uint4 wv[4];
#pragma unroll
      for(int i = 0; i < 4; ++i) wv[i] = q.W[(((long long)q.g * NPH + ph) * 4 + i) * 32 + lane];
      const unsigned int sv = q.S[((long long)q.g * NPH + ph) * 32 + lane];
      float v[16];
#pragma unroll
      for(int i = 0; i < 4; ++i){
        const unsigned int ww[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
#pragma unroll
        for(int j = 0; j < 4; ++j){
          const unsigned int w = ww[j];
          v[4 * i + j] = (*(const float*)(Tc + __byte_perm(w, l4, 0x5504u))
                        + *(const float*)(Tc + 128 + __byte_perm(w, l4, 0x5514u)))
                       + (*(const float*)(Tc + 20736 + __byte_perm(w, l4, 0x5524u))
                        + *(const float*)(Tc + 20864 + __byte_perm(w, l4, 0x5534u)));
        }
      }
      /* xor 1: 16 -> 8 (rows (b0?8:0)+i) */
      float a8[8];
#pragma unroll
      for(int i = 0; i < 8; ++i) a8[i] = (b0 ? v[8 + i] : v[i]) + __shfl_xor(b0 ? v[i] : v[8 + i], 1);
      float a4[4];
#pragma unroll
      for(int i = 0; i < 4; ++i) a4[i] = (b1 ? a8[4 + i] : a8[i]) + __shfl_xor(b1 ? a8[i] : a8[4 + i], 2);
      float a2[2];
#pragma unroll
      for(int i = 0; i < 2; ++i) a2[i] = (b2 ? a4[2 + i] : a4[i]) + __shfl_xor(b2 ? a4[i] : a4[2 + i], 4);
      /* a2[j] = row base+j summed over this lane's 8-lane (one scale) group: scale it */
      a2[0] *= h2f((unsigned short)(sv & 0xFFFFu));
      a2[1] *= h2f((unsigned short)(sv >> 16));
      float d = (b3 ? a2[1] : a2[0]) + __shfl_xor(b3 ? a2[0] : a2[1], 8);
      d += __shfl_xor(d, 16);
      if(wr) q.y[rw] = yo + d;
    }
  }
}


/* ---------------------------------------------------------------- norms, folds, GDN, RoPE */

extern "C" __global__ void k_fold_fwht(const float* __restrict__ x, const float* __restrict__ sg,
                            float* __restrict__ xo, int HB){
  __shared__ float xs[4096];
  const int c = blockIdx.x, j = threadIdx.x;
  if(HB == 512 || HB == -512){
    xs[j] = sg[(c << 9) + j] * x[(c << 9) + j];
    __syncthreads();
    for(int h = 1; h < 512; h <<= 1){
      if(j < 256){
        const int i = (j / h) * 2 * h + (j % h);
        const float a = xs[i], b = xs[i + h];
        xs[i] = a + b; xs[i + h] = a - b;
      }
      __syncthreads();
    }
    xo[(c << 9) + j] = (HB == 512) ? xs[j] * 0.044194173824159220878f : xs[j];
    return;
  }
  for(int q = j; q < HB; q += 512) xs[q] = sg[c * HB + q] * x[c * HB + q];
  __syncthreads();
  fwht_n(xs, HB, j);
  const float sc = rsqrtf((float)HB);
  for(int q = j; q < HB; q += 512) xo[c * HB + q] = xs[q] * sc;
}

__device__ __forceinline__ float rsum(const float* __restrict__ r, int e, int C, int n){
  float v = r[e];
  for(int i = 1; i < n; ++i) v += r[(long long)i * C + e];
  return v;
}

extern "C" __global__ void k_rmsfold(const float* __restrict__ x, const float* __restrict__ r, float* __restrict__ xw,
                          const float* __restrict__ w, const float* __restrict__ sg, float* __restrict__ xn,
                          float* __restrict__ xo, int C, int add, int HB){
  /* add = number of parts of the pending residual r (0: none; >1: the parts of a column-split GEMV, C apart, summed
     in a fixed order as k_sumparts would). grid C/512, block 512: RMS norm of x (+ r if add: the residual of the previous block, fused here) and the Hadamard
     fold of chunk blockIdx.x. Every block computes the same sum of squares in the same order (bit-identical scale) and
     writes only its own chunk; with add, x + r goes to xw (a second buffer), so no block reads a value another block
     has already changed. */
  __shared__ float xs[4096];
  __shared__ double red[16];
  __shared__ float bscale;
  const int j = threadIdx.x, lane = j & 31, wp = j >> 5, c0 = blockIdx.x * (HB < 0 ? 512 : HB);
  double acc = 0.0;
  for(int e = j; e < C; e += 512){
    const float xf = add ? x[e] + rsum(r, e, C, add) : x[e];
    const double v = (double)xf; acc += v * v;
  }
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
  for(int m = j; m < (HB < 0 ? 512 : HB); m += 512){
    const int e = c0 + m;
    const float xf = add ? x[e] + rsum(r, e, C, add) : x[e];
    if(add) xw[e] = xf;
    const float v = (xf * s) * w[e];
    xn[e] = v;
    xs[m] = sg[e] * v;
  }
  __syncthreads();
  if(HB == 512 || HB == -512){
    fwht_chunks(xs, 1, j);
    xo[c0 + j] = (HB == 512) ? xs[j] * 0.044194173824159220878f : xs[j];
  } else {
    fwht_n(xs, HB, j);
    const float sc = rsqrtf((float)HB);
    for(int m = j; m < HB; m += 512) xo[c0 + m] = xs[m] * sc;
  }
}

extern "C" __global__ void k_smfold(const float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ sg,
                         float* __restrict__ xo, int perm_hd, int perm_nk, int perm_rep, int HB){
  __shared__ float xs[4096];
  const int c = blockIdx.x, j = threadIdx.x;
  const int hb = HB < 0 ? 512 : HB;
  for(int q = j; q < hb; q += 512){
    const int e = c * hb + q;
    int src = e;
    if(perm_rep > 1){
      const int hd = e % perm_hd, grouped = e / perm_hd;
      const int rep = grouped % perm_rep, nk = grouped / perm_rep;
      src = hd + perm_hd * (nk + perm_nk * rep);
    }
    const float xa = a[src];
    const float m = (xa / (1.0f + expf(-xa))) * b[src];
    xs[q] = sg[e] * m;
  }
  __syncthreads();
  if(HB == 512 || HB == -512){
    fwht_chunks(xs, 1, j);
    xo[(c << 9) + j] = (HB == 512) ? xs[j] * 0.044194173824159220878f : xs[j];
  } else {
    fwht_n(xs, HB, j);
    const float sc = rsqrtf((float)HB);
    for(int q = j; q < HB; q += 512) xo[c * HB + q] = xs[q] * sc;
  }
}

extern "C" __global__ void k_l2w(const float* __restrict__ x, float* __restrict__ out, int n, int rows){
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if(r >= rows) return;
  const float* xr = x + (long long)r * n;
  double acc = 0.0;
  for(int e = lane; e < n; e += 32){ const double v = (double)xr[e]; acc += v * v; }
  acc = wsum_d(acc);
  const float sc = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
  for(int e = lane; e < n; e += 32) out[(long long)r * n + e] = xr[e] * sc;
}

extern "C" __global__ void k_rmsw(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int n, int rows){
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if(r >= rows) return;
  const float* xr = x + (long long)r * n;
  double acc = 0.0;
  for(int e = lane; e < n; e += 32){ const double v = (double)xr[e]; acc += v * v; }
  acc = wsum_d(acc);
  const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
  for(int e = lane; e < n; e += 32) out[(long long)r * n + e] = (xr[e] * sc) * w[e];
}

extern "C" __global__ void k_gdn_gates(const float* __restrict__ xn, const float* __restrict__ bw,
                            const float* __restrict__ aw, const float* __restrict__ dtb,
                            const float* __restrict__ ssma, float* __restrict__ beta,
                            float* __restrict__ g, int n){
  __shared__ float sb[256];
  __shared__ float sa[256];
  const int h = blockIdx.x, t = threadIdx.x;
  float ab = 0.f, aa = 0.f;
  for(int i = t; i < n; i += 256){
    const float x = xn[i];
    ab += bw[(long long)h * n + i] * x;
    aa += aw[(long long)h * n + i] * x;
  }
  sb[t] = ab; sa[t] = aa;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st){ sb[t] += sb[t + st]; sa[t] += sa[t + st]; }
    __syncthreads();
  }
  if(t == 0){
    beta[h] = 1.0f / (1.0f + expf(-sb[0]));
    const float a = sa[0] + dtb[h];
    const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
    g[h] = sp * ssma[h];
  }
}

extern "C" __global__ void k_gdn_conv(const float* __restrict__ qkv, float* __restrict__ conv,
                           const float* __restrict__ cw, float* __restrict__ cs, int C){
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if(c >= C) return;
  const float c0 = conv[c], c1 = conv[C + c], c2 = conv[2 * C + c], nw = qkv[c];
  const float o = c0 * cw[4 * c] + c1 * cw[4 * c + 1] + c2 * cw[4 * c + 2] + nw * cw[4 * c + 3];
  conv[c] = c1; conv[C + c] = c2; conv[2 * C + c] = nw;
  cs[c] = o / (1.0f + expf(-o));
}

extern "C" __global__ void k_gdn_delta(float* __restrict__ M, const float* __restrict__ q,
                            const float* __restrict__ k, const float* __restrict__ v,
                            const float* __restrict__ g, const float* __restrict__ beta,
                            float* __restrict__ out, float scale, int n_k_heads){
  const int h = blockIdx.x, j = threadIdx.x;          /* head, value column (128) */
  __shared__ float ks[128];
  __shared__ float qs[128];
  const int kh = h % n_k_heads;
  ks[j] = k[kh * 128 + j];
  qs[j] = q[kh * 128 + j];
  __syncthreads();
  const float e = expf(g[h]);
  float* Mh = M + (long long)h * 128 * 128;
  float sk = 0.f;
  for(int i = 0; i < 128; ++i) sk += (Mh[i * 128 + j] * e) * ks[i];
  const float dd = (v[h * 128 + j] - sk) * beta[h];
  float o = 0.f;
  for(int i = 0; i < 128; ++i){
    const float m = Mh[i * 128 + j] * e + ks[i] * dd;
    Mh[i * 128 + j] = m;
    o += m * qs[i];
  }
  out[h * 128 + j] = o * scale;
}

extern "C" __global__ void k_rope(const float* __restrict__ x,
                       const float* __restrict__ cc,
                       const float* __restrict__ ss,
                       float* __restrict__ out, int HD){
  const int r = blockIdx.x;
  const int j = threadIdx.x;
  const float* xr = x + (long long)r * HD;
  float* orow = out + (long long)r * HD;
  if(j < 32){
    const float a = xr[j], b = xr[j + 32];
    orow[j]      = a * cc[j] - b * ss[j];
    orow[j + 32] = a * ss[j] + b * cc[j];
  } else if(j >= 64){
    orow[j] = xr[j];
  }
}

extern "C" __global__ void k_deq_emb3(const unsigned char* __restrict__ bc, const unsigned short* __restrict__ dpb,
                           const float* __restrict__ sg, float* __restrict__ h, int row, int C, int HB){
  __shared__ float w[6144];   /* D <= 6144 (27B: 5120) */
  const int t = threadIdx.x;
  const long long r4 = row >> 2, rl = row & 3, g16 = row >> 4;
  const int rr = row & 15;
  const int lane_low = ((rr >> 3) & 1) | (((rr >> 2) & 1) << 1) | (((rr >> 1) & 1) << 2), jj = rr & 1;
  for(int e = t; e < C; e += 1024){
    const int s = e >> 2, p = e >> 7, r = e & 3;
    const unsigned int b = bc[((((long long)(row >> 4) * (C >> 9) + (s >> 7)) * 4 + (r4 & 3)) * 32 + ((s >> 2) & 31)) * 16
                              + rl * 4 + (s & 3)];
    const int dig = (r == 0) ? (b % 3) : ((r == 1) ? ((b / 3) % 3) : ((r == 2) ? ((b / 9) % 3) : ((b / 27) % 3)));
    const int lane = ((p & 3) << 3) | lane_low;
    w[e] = ((float)(dig - 1)) * h2f3(dpb[((g16 * (C >> 9) + (p >> 2)) * 32 + lane) * 2 + jj]);
  }
  __syncthreads();
  if(HB == 512){
    for(int j2 = t; j2 < C; j2 += 1024){
      const int c = j2 >> 9, j = j2 & 511;
      float acc = 0.f;
#pragma unroll 8
      for(int i = 0; i < 512; ++i){
        const float ww = w[(c << 9) + i];
        acc += (((__popc(i & j) & 1) == 1) ? -ww : ww);
      }
      h[j2] = sg[j2] * (acc * 0.044194173824159220878f);
    }
    return;
  }
  /* larger blocks: an in-place FWHT of each block (the first 512 threads), then signs and scale */
  for(int c0 = 0; c0 < C; c0 += HB){
    if(t < 512){
      for(int hh = 1; hh < HB; hh <<= 1){
        for(int q = t; q < (HB >> 1); q += 512){
          const int i = c0 + (q / hh) * 2 * hh + (q % hh);
          const float a = w[i], b = w[i + hh];
          w[i] = a + b; w[i + hh] = a - b;
        }
#if KHIP
        __syncthreads();
#else
        asm volatile("bar.sync 1, 512;");
#endif
      }
    }
  }
  __syncthreads();
  const float sc = rsqrtf((float)HB);
  for(int j2 = t; j2 < C; j2 += 1024) h[j2] = sg[j2] * (w[j2] * sc);
}


/* ---------------------------------------------------------------- attention (fp16 K/V, GQA 4:1) */

#define ATV_CH 256

extern "C" __global__ void k_attn_s4w(const uint4* __restrict__ Kh, const float* __restrict__ q, float* __restrict__ s, int n, int n_kv){
  /* grid (NHKV, ceil(n/32)), block 256: warp = 4 consecutive positions; lane = dims 8l..8l+7 with the 4 heads' q
     in registers (loaded once); one coalesced 16-byte K load per lane per position; the 16 sums (4 positions x 4
     heads) reduced by a straight-line multi-value butterfly (16 shuffles per 4 positions) */
  const int kv = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.y * 8 + warp, nwarps = gridDim.y * 8;     /* persistent: q loaded once per warp */
  if(gw * 4 >= n) return;
  float qr[4][8];
#pragma unroll
  for(int j = 0; j < 4; ++j){
    const float4 a = ((const float4*)(q + kv * 1024 + j * 256 + lane * 8))[0];
    const float4 b = ((const float4*)(q + kv * 1024 + j * 256 + lane * 8))[1];
    qr[j][0] = a.x; qr[j][1] = a.y; qr[j][2] = a.z; qr[j][3] = a.w;
    qr[j][4] = b.x; qr[j][5] = b.y; qr[j][6] = b.z; qr[j][7] = b.w;
  }
  for(int t0 = gw * 4; t0 < n; t0 += nwarps * 4){
  float v[16];
#pragma unroll
  for(int p = 0; p < 4; ++p){
    const int t = min(t0 + p, n - 1);
    const uint4 w = Kh[((long long)t * n_kv + kv) * 32 + lane];
    const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
    float kk[8];
#pragma unroll
    for(int m = 0; m < 4; ++m){ kk[2 * m] = hf((unsigned short)(ww[m] & 0xFFFFu)); kk[2 * m + 1] = hf((unsigned short)(ww[m] >> 16)); }
#pragma unroll
    for(int j = 0; j < 4; ++j){
      float a = 0.f;
#pragma unroll
      for(int i = 0; i < 8; ++i) a += kk[i] * qr[j][i];
      v[p * 4 + j] = a;
    }
  }
  { const bool h0 = (lane & 16) != 0; const float b0_0 = (h0 ? v[8] : v[0]) + __shfl_xor(h0 ? v[0] : v[8], 16); const float b0_1 = (h0 ? v[9] : v[1]) + __shfl_xor(h0 ? v[1] : v[9], 16); const float b0_2 = (h0 ? v[10] : v[2]) + __shfl_xor(h0 ? v[2] : v[10], 16); const float b0_3 = (h0 ? v[11] : v[3]) + __shfl_xor(h0 ? v[3] : v[11], 16); const float b0_4 = (h0 ? v[12] : v[4]) + __shfl_xor(h0 ? v[4] : v[12], 16); const float b0_5 = (h0 ? v[13] : v[5]) + __shfl_xor(h0 ? v[5] : v[13], 16); const float b0_6 = (h0 ? v[14] : v[6]) + __shfl_xor(h0 ? v[6] : v[14], 16); const float b0_7 = (h0 ? v[15] : v[7]) + __shfl_xor(h0 ? v[7] : v[15], 16); const bool h1 = (lane & 8) != 0; const float b1_0 = (h1 ? b0_4 : b0_0) + __shfl_xor(h1 ? b0_0 : b0_4, 8); const float b1_1 = (h1 ? b0_5 : b0_1) + __shfl_xor(h1 ? b0_1 : b0_5, 8); const float b1_2 = (h1 ? b0_6 : b0_2) + __shfl_xor(h1 ? b0_2 : b0_6, 8); const float b1_3 = (h1 ? b0_7 : b0_3) + __shfl_xor(h1 ? b0_3 : b0_7, 8); const bool h2 = (lane & 4) != 0; const float b2_0 = (h2 ? b1_2 : b1_0) + __shfl_xor(h2 ? b1_0 : b1_2, 4); const float b2_1 = (h2 ? b1_3 : b1_1) + __shfl_xor(h2 ? b1_1 : b1_3, 4); const bool h3 = (lane & 2) != 0; const float b3_0 = (h3 ? b2_1 : b2_0) + __shfl_xor(h3 ? b2_0 : b2_1, 2); const float b4_0 = b3_0 + __shfl_xor(b3_0, 1); v[0] = b4_0; }
  const int idx = ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
  const int t = t0 + (idx >> 2), j = idx & 3;
  if((lane & 1) == 0 && t < n) s[(kv * 4 + j) * n + t] = v[0] * 0.0625f;
  }
}

extern "C" __global__ void k_softmax_rows(float* __restrict__ s, int n){
  /* grid NH, block 256: max-subtracted softmax over the n positions of each head, in place */
  const int hh = blockIdx.x, t = threadIdx.x;
  float* sr = s + (long long)hh * n;
  __shared__ float red[256];
  __shared__ float pmax;
  float mx = -1e30f;
  for(int i = t; i < n; i += 256) mx = fmaxf(mx, sr[i]);
  red[t] = mx;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st) red[t] = fmaxf(red[t], red[t + st]);
    __syncthreads();
  }
  if(t == 0) pmax = red[0];
  __syncthreads();
  const float m = pmax;
  float acc0 = 0.0f;
  for(int i = t; i < n; i += 256) acc0 += expf(sr[i] - m);
  __syncthreads();
  red[t] = acc0;
  __syncthreads();
  for(int st = 128; st > 0; st >>= 1){
    if(t < st) red[t] += red[t + st];
    __syncthreads();
  }
  const float tot = red[0];
  __syncthreads();
  for(int i = t; i < n; i += 256) sr[i] = expf(sr[i] - m) / tot;
}

extern "C" __global__ void k_attn_v4p(const float* __restrict__ s, const unsigned int* __restrict__ Vh2,
                           float* __restrict__ part, int n, int nheads, int n_kv){
  /* grid (NHKV, 1, nchunks), block 128: thread = features 2f, 2f+1 (one half2 load), 4 heads; partial sums.
     The 4 heads' probabilities of a position are one float4 in shared memory (one broadcast load). */
  __shared__ float4 sp4[ATV_CH];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x){
    float4 p4 = make_float4(0.f, 0.f, 0.f, 0.f);
    if(i < m){
      const float* s0 = s + (long long)(kv * 4) * n + i0 + i;
      p4 = make_float4(s0[0], s0[n], s0[2 * n], s0[3 * n]);
    }
    sp4[i] = p4;
  }
  __syncthreads();
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
  const unsigned int* vb = Vh2 + ((long long)i0 * n_kv + kv) * 128 + f2;
#pragma unroll 8
  for(int i = 0; i < m; ++i){
    const unsigned int vv = vb[(long long)i * n_kv * 128];
    const float v0 = hf((unsigned short)(vv & 0xFFFFu)), v1 = hf((unsigned short)(vv >> 16));
    const float4 p = sp4[i];
    a0 += p.x * v0; a1 += p.y * v0; a2 += p.z * v0; a3 += p.w * v0;
    b0 += p.x * v1; b1 += p.y * v1; b2 += p.z * v1; b3 += p.w * v1;
  }
  float* pp = part + ((long long)c * nheads + kv * 4) * 256 + 2 * f2;
  pp[0] = a0; pp[1] = b0; pp[256] = a1; pp[257] = b1; pp[512] = a2; pp[513] = b2; pp[768] = a3; pp[769] = b3;
}

extern "C" __global__ void k_attn_comb(const float* __restrict__ part, const float* __restrict__ gatev, float* __restrict__ att,
                            int nch, int nheads){
  /* grid NH, block 256: sum the chunk partials in order, times the sigmoid gate */
  const int hh = blockIdx.x, f = threadIdx.x;
  float acc = part[hh * 256 + f];
  for(int c = 1; c < nch; ++c) acc += part[((long long)c * nheads + hh) * 256 + f];
  const float g = gatev[hh * 512 + 256 + f];
  att[hh * 256 + f] = acc * (1.0f / (1.0f + expf(-g)));
}


/* ---------------------------------------------------------------- head, argmax, sampling */

extern "C" __global__ void k_argmax(const float* __restrict__ y,
                         int n, int* __restrict__ out){
  __shared__ int bidx[1024];
  __shared__ float bval[1024];
  const int tid = threadIdx.x;
  float bv = -1e30f;
  int bi = 0;
  for(int e = tid; e < n; e += 1024){
    const float v = y[e];
    if(v > bv){ bv = v; bi = e; }
  }
  bidx[tid] = bi; bval[tid] = bv;
  __syncthreads();
  for(int st = 512; st > 0; st >>= 1){
    if(tid < st){
      const int i2 = tid + st;
      if(bval[i2] > bval[tid] ||
         (bval[i2] == bval[tid] && bidx[i2] < bidx[tid])){
        bval[tid] = bval[i2]; bidx[tid] = bidx[i2];
      }
    }
    __syncthreads();
  }
  if(tid == 0) *out = bidx[0];
}

extern "C" __global__ void k_expand(const int* __restrict__ sel, const int* __restrict__ gstart,
                                    const int* __restrict__ gcount, const int* __restrict__ off, int* __restrict__ glist){
  const int m = blockIdx.x, k = sel[m], n = gcount[k], o = off[m], s = gstart[k];
  for(int i = threadIdx.x; i < n; i += blockDim.x) glist[o + i] = s + i;
}

extern "C" __global__ void k_mask_neg(float* __restrict__ y, const int* __restrict__ tok, int n){
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    if(tok[i] < 0) y[i] = -1e30f;
}

extern "C" __global__ void k_band(const float* __restrict__ y, const int* __restrict__ amax, int n, float band,
                                  int* __restrict__ out_i, float* __restrict__ out_v, int* __restrict__ cnt, int cap){
  const float thr = y[amax[0]] - band;
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
    const float v = y[i];
    if(v >= thr){
      const int k = atomicAdd(cnt, 1);
      if(k < cap){ out_i[k] = i; out_v[k] = v; }
    }
  }
}
/* top-k version (one block of 1024): binary-search the largest threshold t in [max - band, max] with
   count(y >= t) >= k (16 halvings), then collect y >= t - so the candidate set always holds the true top-k */
extern "C" __global__ void k_topk(const float* __restrict__ y, const int* __restrict__ amax, int n, int k, float band,
                                  int* __restrict__ out_i, float* __restrict__ out_v, int* __restrict__ cnt, int cap){
  __shared__ int red[32];
  __shared__ float lo_s, hi_s;
  const int t = threadIdx.x, lane = t & 31, w = t >> 5;
  if(t == 0){ hi_s = y[amax[0]]; lo_s = hi_s - band; }
  __syncthreads();
  for(int it = 0; it < 17; ++it){
    const float mid = (it == 0) ? lo_s : 0.5f * (lo_s + hi_s);
    int c = 0;
    for(int i = t; i < n; i += 1024) c += (y[i] >= mid);
    for(int o = 16; o > 0; o >>= 1) c += __shfl_xor(c, o);
    if(lane == 0) red[w] = c;
    __syncthreads();
    if(t == 0){
      int tot = 0;
      for(int j = 0; j < 32; ++j) tot += red[j];
      if(it == 0){ if(tot < k) hi_s = lo_s; }   /* fewer than k within the band: keep the whole band */
      else if(tot >= k) lo_s = mid; else hi_s = mid;
    }
    __syncthreads();
    if(hi_s <= lo_s) break;
  }
  const float thr = lo_s;
  for(int i = t; i < n; i += 1024){
    const float v = y[i];
    if(v >= thr){
      const int q = atomicAdd(cnt, 1);
      if(q < cap){ out_i[q] = i; out_v[q] = v; }
    }
  }
}


extern "C" __global__ void k_sumparts(const float* __restrict__ p, float* __restrict__ y, int R, int n){
  /* y[r] = sum of the n column-split partial sums of row r, in a fixed order */
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if(r >= R) return;
  float v = p[r];
  for(int i = 1; i < n; ++i) v += p[(long long)i * R + r];
  y[r] = v;
}


extern "C" __global__ void k_fwht_hi(float* __restrict__ x, int HB, int n){
  /* the stages h >= 512 of an HB-point FWHT (HB/512 <= 8 chunks) over n elements, then the 1/sqrt(HB) scale; thread =
     one position j of the 512 in one HB block, the HB/512 values in registers */
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int nc = HB >> 9;
  if(t >= (n >> 9) / nc * 512) return;
  const int b = t >> 9, j = t & 511;
  float* xb = x + (long long)b * HB + j;
  float v[8];
#pragma unroll
  for(int c = 0; c < 8; ++c) v[c] = c < nc ? xb[c << 9] : 0.f;
#pragma unroll
  for(int h = 1; h < 8; h <<= 1){
    if(h < nc){
#pragma unroll
      for(int c = 0; c < 8; ++c){
        if(c < nc && (c & h) == 0){
          const float a = v[c], d = v[c + h];
          v[c] = a + d; v[c + h] = a - d;
        }
      }
    }
  }
  const float sc = rsqrtf((float)HB);
#pragma unroll
  for(int c = 0; c < 8; ++c) if(c < nc) xb[c << 9] = v[c] * sc;
}

/* ---------------------------------------------------------------- generic GQA attention (G q heads per K/V head, G <= 8) */

extern "C" __global__ void k_attn_sg(const uint4* __restrict__ Kh, const float* __restrict__ q, float* __restrict__ s,
                                     int n, int n_kv, int G){
  /* grid (NHKV, <=32), block 256: a warp per position (persistent), lane = dims 8l..8l+7, the G heads' q in registers */
  const int kv = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  float qr[8][8];
#pragma unroll
  for(int j = 0; j < 8; ++j)
#pragma unroll
    for(int i = 0; i < 8; ++i) qr[j][i] = j < G ? q[(kv * G + j) * 256 + lane * 8 + i] : 0.f;
  for(int t = blockIdx.y * 8 + warp; t < n; t += gridDim.y * 8){
    const uint4 w = Kh[((long long)t * n_kv + kv) * 32 + lane];
    const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
    float kk[8];
#pragma unroll
    for(int m = 0; m < 4; ++m){ kk[2 * m] = hf((unsigned short)(ww[m] & 0xFFFFu)); kk[2 * m + 1] = hf((unsigned short)(ww[m] >> 16)); }
#pragma unroll
    for(int j = 0; j < 8; ++j){
      if(j < G){
        float a = 0.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) a += kk[i] * qr[j][i];
        a = wsum_f(a);
        if(lane == 0) s[(long long)(kv * G + j) * n + t] = a * 0.0625f;
      }
    }
  }
}

extern "C" __global__ void k_attn_vg(const float* __restrict__ s, const unsigned int* __restrict__ Vh2,
                                     float* __restrict__ part, int n, int nheads, int n_kv, int G){
  /* grid (NHKV, 1, nchunks), block 128: thread = features 2f, 2f+1, the G heads of one K/V head; partial sums */
  __shared__ float sp[ATV_CH * 8];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) sp[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float a[8], b[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){ a[j] = 0.f; b[j] = 0.f; }
  const unsigned int* vb = Vh2 + ((long long)i0 * n_kv + kv) * 128 + f2;
  for(int i = 0; i < m; ++i){
    const unsigned int vv = vb[(long long)i * n_kv * 128];
    const float v0 = hf((unsigned short)(vv & 0xFFFFu)), v1 = hf((unsigned short)(vv >> 16));
#pragma unroll
    for(int j = 0; j < 8; ++j){ const float p = sp[i * 8 + j]; a[j] += p * v0; b[j] += p * v1; }
  }
#pragma unroll
  for(int j = 0; j < 8; ++j){
    if(j < G){
      float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 2 * f2;
      pp[0] = a[j]; pp[1] = b[j];
    }
  }
}

/* ---------------------------------------------------------------- two-card exchange (tensor parallel) */

extern "C" __global__ void k_xchg(const float* __restrict__ mine, float* __restrict__ parts, float* hb,
                                  volatile int* flags, int me, int epoch, int C, int* fail){
  /* one block. hb: mapped pinned host memory [2 parities][2 cards][C]. Writes this card's vector to its host slot and
     to parts[me], raises its flag, waits for the other card's flag of the same epoch and copies the other card's
     vector to parts[1 - me]. parts then holds [card 0, card 1] in the same order on both cards. Double buffering by
     epoch parity: a card reuses a slot only after the other card has finished reading it. */
  __shared__ int ok;
  const int t = threadIdx.x, par = epoch & 1;
  float* hm = hb + (long long)(par * 2 + me) * C;
  const float* ho = hb + (long long)(par * 2 + (1 - me)) * C;
  for(int e = t * 4; e < C; e += blockDim.x * 4){
    const float4 v = *(const float4*)(mine + e);
    *(float4*)(hm + e) = v;
    *(float4*)(parts + (long long)me * C + e) = v;
  }
  __threadfence_system();
  __syncthreads();
  if(t == 0){
    flags[me] = epoch;
    __threadfence_system();
    long long spins = 0; ok = 1;
    while(flags[1 - me] < epoch){ if(++spins > 4000000000LL){ *fail = epoch; ok = 0; break; } }
  }
  __syncthreads();
  if(!ok) return;
  for(int e = t * 4; e < C; e += blockDim.x * 4){
    float x, y, z, w;
#if KHIP
    { float4 v4 = *(const float4*)(ho + e); x = v4.x; y = v4.y; z = v4.z; w = v4.w; }
#else
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ho + e));
#endif
    *(float4*)(parts + (long long)(1 - me) * C + e) = make_float4(x, y, z, w);
  }
}

/* ---------------------------------------------------------------- launch-plan helpers (tensor parallel) */

extern "C" __global__ void k_xchg2(const float* __restrict__ pin, int nparts, int R, float* __restrict__ parts,
                                   float* hb, volatile int* flags, int me, int* fail, int* ctr){
  /* k_xchg with the column-split partial sums added here (pin: nparts vectors of R, in a fixed order) and the
     epoch counted on the GPU (ctr), so a recorded launch needs no per-token argument */
  __shared__ int ok, ep;
  const int t = threadIdx.x;
  if(t == 0) ep = ++ctr[0];
  __syncthreads();
  const int epoch = ep, par = epoch & 1, C = R;
  float* hm = hb + (long long)(par * 2 + me) * C;
  const float* ho = hb + (long long)(par * 2 + (1 - me)) * C;
  for(int e = t; e < C; e += blockDim.x){
    float v = pin[e];
    for(int i = 1; i < nparts; ++i) v += pin[(long long)i * R + e];
    hm[e] = v;
    parts[(long long)me * C + e] = v;
  }
  __threadfence_system();
  __syncthreads();
  if(t == 0){
    flags[me] = epoch;
    __threadfence_system();
    long long spins = 0; ok = 1;
    while(flags[1 - me] < epoch){ if(++spins > 4000000000LL){ *fail = epoch; ok = 0; break; } }
  }
  __syncthreads();
  if(!ok) return;
  for(int e = t * 4; e < C; e += blockDim.x * 4){
    float x, y, z, w;
#if KHIP
    { float4 v4 = *(const float4*)(ho + e); x = v4.x; y = v4.y; z = v4.z; w = v4.w; }
#else
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ho + e));
#endif
    *(float4*)(parts + (long long)(1 - me) * C + e) = make_float4(x, y, z, w);
  }
}

extern "C" __global__ void k_attn_vgn(const float* __restrict__ s, const unsigned int* __restrict__ Vh2,
                                      float* __restrict__ part, int n, int nheads, int n_kv, int G){
  /* k_attn_vg for a grid sized for the whole context: chunks past n return at once */
  if((int)blockIdx.z * ATV_CH >= n) return;
  __shared__ float sp[ATV_CH * 8];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) sp[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float a[8], b[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){ a[j] = 0.f; b[j] = 0.f; }
  const unsigned int* vb = Vh2 + ((long long)i0 * n_kv + kv) * 128 + f2;
  for(int i = 0; i < m; ++i){
    const unsigned int vv = vb[(long long)i * n_kv * 128];
    const float v0 = hf((unsigned short)(vv & 0xFFFFu)), v1 = hf((unsigned short)(vv >> 16));
#pragma unroll
    for(int j = 0; j < 8; ++j){ const float p = sp[i * 8 + j]; a[j] += p * v0; b[j] += p * v1; }
  }
#pragma unroll
  for(int j = 0; j < 8; ++j){
    if(j < G){
      float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 2 * f2;
      pp[0] = a[j]; pp[1] = b[j];
    }
  }
}

extern "C" __global__ void k_attn_combn(const float* __restrict__ part, const float* __restrict__ gatev,
                                        float* __restrict__ att, int n, int nheads){
  /* k_attn_comb with the chunk count taken from n */
  const int hh = blockIdx.x, f = threadIdx.x, nch = (n + ATV_CH - 1) / ATV_CH;
  float acc = part[hh * 256 + f];
  for(int c = 1; c < nch; ++c) acc += part[((long long)c * nheads + hh) * 256 + f];
  const float g = gatev[hh * 512 + 256 + f];
  att[hh * 256 + f] = acc * (1.0f / (1.0f + expf(-g)));
}

extern "C" __global__ void k_store_h(const float* __restrict__ x, unsigned short* __restrict__ dst, int n){
  /* fp32 -> fp16 (round to nearest even), the K/V cache store */
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
#if KHIP
  if(i < n){ __half hh = __float2half(x[i]); dst[i] = *(unsigned short*)&hh; }
#else
  if(i < n){ unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x[i])); dst[i] = h; }
#endif
}

extern "C" __global__ void k_cpq(const float* __restrict__ qn, float* __restrict__ q, int rows, int hd){
  /* the q half of each [q | gate] row */
  const int r = blockIdx.x, j = threadIdx.x;
  for(int e = j; e < hd; e += blockDim.x) q[r * hd + e] = qn[r * 2 * hd + e];
}

extern "C" __global__ void k_xchg3(const float* __restrict__ pin, int nparts, int R, float* __restrict__ parts,
                                   float* hb, volatile int* flags, int me, int* fail, int* ctr){
  /* k_xchg2 over gridDim.x blocks: block b owns R / gridDim.x elements (a multiple of 4 * blockDim.x), its own flag
     pair flags[2b], flags[2b+1] and its own epoch counter ctr[b], so the PCIe writes and reads of the blocks overlap */
  __shared__ int ok, ep;
  const int t = threadIdx.x, b = blockIdx.x, W = R / gridDim.x, e0 = b * W;
  if(t == 0) ep = ++ctr[b];
  __syncthreads();
  const int epoch = ep, par = epoch & 1, C = R;
  float* hm = hb + (long long)(par * 2 + me) * C;
  const float* ho = hb + (long long)(par * 2 + (1 - me)) * C;
  for(int e = e0 + t * 4; e < e0 + W; e += blockDim.x * 4){
    float4 v = *(const float4*)(pin + e);
    for(int i = 1; i < nparts; ++i){
      const float4 q = *(const float4*)(pin + (long long)i * R + e);
      v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
    }
    *(float4*)(hm + e) = v;
    *(float4*)(parts + (long long)me * C + e) = v;
  }
  __threadfence_system();
  __syncthreads();
  if(t == 0){
    flags[2 * b + me] = epoch;
    __threadfence_system();
    long long spins = 0; ok = 1;
    while(flags[2 * b + 1 - me] < epoch){ if(++spins > 4000000000LL){ *fail = epoch; ok = 0; break; } }
  }
  __syncthreads();
  if(!ok) return;
  for(int e = e0 + t * 4; e < e0 + W; e += blockDim.x * 4){
    float x, y, z, w;
#if KHIP
    { float4 v4 = *(const float4*)(ho + e); x = v4.x; y = v4.y; z = v4.z; w = v4.w; }
#else
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(x), "=f"(y), "=f"(z), "=f"(w) : "l"(ho + e));
#endif
    *(float4*)(parts + (long long)(1 - me) * C + e) = make_float4(x, y, z, w);
  }
}

/* ---------------------------------------------------------------- tensor parallel 27B: merged kernels, q8_0 K/V */

extern "C" __global__ void __launch_bounds__(1024) k_xrms(
    const float* __restrict__ pin, int nparts, float* hb, volatile int* flags, int me, int* fail, int* ctr,
    const float* __restrict__ h, float* __restrict__ h2, const float* __restrict__ w, const float* __restrict__ sg,
    float* __restrict__ xn, float* __restrict__ xo, int C){
  /* one block of 1024: the exchange (this card's partial = sum of its nparts column parts, swapped through mapped
     host memory), the residual add h2 = h + (p0 + p1), the RMS norm (fp64 sum), xn = norm * w, and the
     1024-point Hadamard fold of sg * xn into xo (C a multiple of 1024, C <= 6144) */
  __shared__ float X[6144];
  __shared__ double red[32];
  __shared__ int ok, ep;
  __shared__ float bscale;
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
  if(t == 0) ep = ++ctr[0];
  __syncthreads();
  const int epoch = ep, par = epoch & 1;
  float* hm = hb + (long long)(par * 2 + me) * C;
  const float* ho = hb + (long long)(par * 2 + (1 - me)) * C;
  for(int e = t * 4; e < C; e += 4096){
    float4 v = *(const float4*)(pin + e);
    for(int i = 1; i < nparts; ++i){
      const float4 q = *(const float4*)(pin + (long long)i * C + e);
      v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
    }
    *(float4*)(hm + e) = v;
    X[e] = v.x; X[e + 1] = v.y; X[e + 2] = v.z; X[e + 3] = v.w;
  }
  __threadfence_system();
  __syncthreads();
  if(t == 0){
    flags[me] = epoch;
    __threadfence_system();
    long long spins = 0; ok = 1;
    while(flags[1 - me] < epoch){ if(++spins > 4000000000LL){ *fail = epoch; ok = 0; break; } }
  }
  __syncthreads();
  if(!ok) return;
  double acc = 0.0;
  for(int e = t * 4; e < C; e += 4096){
    float a, b, c, d;
#if KHIP
    { float4 v4 = *(const float4*)(ho + e); a = v4.x; b = v4.y; c = v4.z; d = v4.w; }
#else
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(ho + e));
#endif
    const float o[4] = {a, b, c, d};
#pragma unroll
    for(int k = 0; k < 4; ++k){
      const float p0 = me == 0 ? X[e + k] : o[k], p1 = me == 0 ? o[k] : X[e + k];
      const float xf = h[e + k] + (p0 + p1);
      X[e + k] = xf;
      h2[e + k] = xf;
      acc += (double)xf * (double)xf;
    }
  }
  acc = wsum_d(acc);
  if(lane == 0) red[wp] = acc;
  __syncthreads();
  if(wp == 0){
    double s2 = red[lane];
    s2 = wsum_d(s2);
    if(lane == 0) bscale = 1.0f / sqrtf((float)(s2 / C) + 1e-6f);
  }
  __syncthreads();
  const float s = bscale;
  for(int e = t; e < C; e += 1024){
    const float v = (X[e] * s) * w[e];
    xn[e] = v;
    X[e] = sg[e] * v;
  }
  __syncthreads();
  const int nb = C >> 10, nbf = nb * 512;
  for(int hh = 1; hh < 1024; hh <<= 1){
    for(int q = t; q < nbf; q += 1024){
      const int blk = q >> 9, qq = q & 511;
      const int i = (blk << 10) + (qq / hh) * 2 * hh + (qq % hh);
      const float a = X[i], b = X[i + hh];
      X[i] = a + b; X[i + hh] = a - b;
    }
    __syncthreads();
  }
  const float sc = rsqrtf(1024.0f);
  for(int e = t; e < C; e += 1024) xo[e] = X[e] * sc;
}

extern "C" __global__ void k_gdn_pre(const float* __restrict__ xn, const unsigned short* __restrict__ bw,
    const unsigned short* __restrict__ aw, const float* __restrict__ dtb, const float* __restrict__ ssma,
    float* __restrict__ beta, float* __restrict__ g, int n,
    const float* __restrict__ qkv, float* __restrict__ conv, const float* __restrict__ cw, float* __restrict__ cs,
    float* __restrict__ qk, int C, int NV, int NK, int HK){
  /* block 256. blocks [0, NV): the gates of v head b (k_gdn_gates, bf16 weights). blocks [NV, NV + 2 NK): conv +
     SiLU of the HK channels of q/k head b - NV, then its L2 norm into qk (k_l2w). the rest: conv + SiLU of 256 v
     channels each (k_gdn_conv) */
  const int b = blockIdx.x, t = threadIdx.x;
  if(b < NV){
    __shared__ float sb[256];
    __shared__ float sa[256];
    float ab = 0.f, aa = 0.f;
    for(int i = t; i < n; i += 256){
      const float x = xn[i];
      ab += __uint_as_float((unsigned int)bw[(long long)b * n + i] << 16) * x;
      aa += __uint_as_float((unsigned int)aw[(long long)b * n + i] << 16) * x;
    }
    sb[t] = ab; sa[t] = aa;
    __syncthreads();
    for(int st = 128; st > 0; st >>= 1){
      if(t < st){ sb[t] += sb[t + st]; sa[t] += sa[t + st]; }
      __syncthreads();
    }
    if(t == 0){
      beta[b] = 1.0f / (1.0f + expf(-sb[0]));
      const float a = sa[0] + dtb[b];
      const float sp = (a > 20.0f) ? a : log1pf(expf(fminf(a, 20.0f)));
      g[b] = sp * ssma[b];
    }
    return;
  }
  int c;
  const bool qkh = b < NV + 2 * NK;
  if(qkh){
    if(t >= HK) { __syncthreads(); __syncthreads(); return; }
    c = (b - NV) * HK + t;
  } else {
    c = 2 * NK * HK + (b - NV - 2 * NK) * 256 + t;
    if(c >= C) return;
  }
  const float c0 = conv[c], c1 = conv[C + c], c2 = conv[2 * C + c], nw = qkv[c];
  const float o = c0 * cw[4 * c] + c1 * cw[4 * c + 1] + c2 * cw[4 * c + 2] + nw * cw[4 * c + 3];
  conv[c] = c1; conv[C + c] = c2; conv[2 * C + c] = nw;
  const float y = o / (1.0f + expf(-o));
  cs[c] = y;
  if(!qkh) return;
  __shared__ float row[128];
  __shared__ float scl;
  row[t] = y;
  __syncthreads();
  if(t < 32){
    double acc = 0.0;
    for(int e = t; e < HK; e += 32){ const double v = (double)row[e]; acc += v * v; }
    acc = wsum_d(acc);
    if(t == 0) scl = 1.0f / fmaxf(sqrtf((float)acc), 1e-6f);
  }
  __syncthreads();
  qk[(b - NV) * HK + t] = row[t] * scl;
}

extern "C" __global__ void k_gdn_deltan(float* __restrict__ M, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, const float* __restrict__ nw, float* __restrict__ out, float scale, int n_k_heads){
  /* k_gdn_delta followed by k_rmsw of the head's 128 outputs (same arithmetic, one launch) */
  const int h = blockIdx.x, j = threadIdx.x;
  __shared__ float ks[128];
  __shared__ float qs[128];
  __shared__ float os[128];
  __shared__ float scl;
  const int kh = h % n_k_heads;
  ks[j] = k[kh * 128 + j];
  qs[j] = q[kh * 128 + j];
  __syncthreads();
  const float e = expf(g[h]);
  float* Mh = M + (long long)h * 128 * 128;
  float sk = 0.f;
  for(int i = 0; i < 128; ++i) sk += (Mh[i * 128 + j] * e) * ks[i];
  const float dd = (v[h * 128 + j] - sk) * beta[h];
  float o = 0.f;
  for(int i = 0; i < 128; ++i){
    const float m = Mh[i * 128 + j] * e + ks[i] * dd;
    Mh[i * 128 + j] = m;
    o += m * qs[i];
  }
  os[j] = o * scale;
  __syncthreads();
  if(j < 32){
    double acc = 0.0;
    for(int x = j; x < 128; x += 32){ const double vv = (double)os[x]; acc += vv * vv; }
    acc = wsum_d(acc);
    if(j == 0) scl = 1.0f / sqrtf((float)(acc / 128) + 1e-6f);
  }
  __syncthreads();
  out[h * 128 + j] = (os[j] * scl) * nw[j];
}

extern "C" __global__ void k_rmsw_s(const float* __restrict__ x, int xs, const float* __restrict__ w,
                                    float* __restrict__ out, int n, int rows){
  /* k_rmsw with an input row stride */
  const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if(r >= rows) return;
  const float* xr = x + (long long)r * xs;
  double acc = 0.0;
  for(int e = lane; e < n; e += 32){ const double v = (double)xr[e]; acc += v * v; }
  acc = wsum_d(acc);
  const float sc = 1.0f / sqrtf((float)(acc / n) + 1e-6f);
  for(int e = lane; e < n; e += 32) out[(long long)r * n + e] = (xr[e] * sc) * w[e];
}

extern "C" __global__ void k_rope2(const float* __restrict__ q, const float* __restrict__ k,
                                   const float* __restrict__ cc, const float* __restrict__ ss,
                                   float* __restrict__ qo, float* __restrict__ ko, int nq, int HD){
  /* k_rope for nq q rows and then the k rows, one launch */
  const int r = blockIdx.x, j = threadIdx.x;
  const float* xr = r < nq ? q + (long long)r * HD : k + (long long)(r - nq) * HD;
  float* orow = r < nq ? qo + (long long)r * HD : ko + (long long)(r - nq) * HD;
  if(j < 32){
    const float a = xr[j], b = xr[j + 32];
    orow[j]      = a * cc[j] - b * ss[j];
    orow[j + 32] = a * ss[j] + b * cc[j];
  } else if(j >= 64){
    orow[j] = xr[j];
  }
}

__device__ __forceinline__ void q8_block(const float* __restrict__ x, signed char* __restrict__ q,
                                         unsigned short* dh, float* df, int lane){
  /* llama.cpp quantize_row_q8_0 (x86): d = amax / 127, q = round-to-nearest-even(x * 127 / amax), d stored fp16 */
  const float v = x[lane];
  float a = fabsf(v);
  for(int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o));
  const float d = a / 127.0f, id = a != 0.0f ? 127.0f / a : 0.0f;
  q[lane] = (signed char)__float2int_rn(v * id);
  if(lane == 0){
#if KHIP
    unsigned short hbits; { __half hh = __float2half(d); hbits = *(unsigned short*)&hh; }
#else
    unsigned short hbits; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hbits) : "f"(d));
#endif
    if(dh) dh[0] = hbits;
    if(df) df[0] = h2f(hbits);
  }
}

extern "C" __global__ void k_store_q8(const float* __restrict__ k, const float* __restrict__ v,
                                      signed char* __restrict__ kq, unsigned short* __restrict__ kd,
                                      signed char* __restrict__ vq, unsigned short* __restrict__ vd, int nblk){
  /* one warp per 32-value block: the first nblk warps quantise k, the next nblk v (one position's K/V row) */
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if(w >= 2 * nblk) return;
  if(w < nblk) q8_block(k + w * 32, kq + w * 32, kd + w, 0, lane);
  else q8_block(v + (w - nblk) * 32, vq + (w - nblk) * 32, vd + (w - nblk), 0, lane);
}

extern "C" __global__ void k_q8row(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ d,
                                   int nblk){
  /* the attention query in q8_0 (llama.cpp's flash attention quantises Q to K's dot type) */
  const int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if(w < nblk) q8_block(x + w * 32, q + w * 32, 0, d + w, lane);
}

extern "C" __global__ void k_attn_sq8(const uint2* __restrict__ Kq, const unsigned short* __restrict__ Kd,
                                      const uint2* __restrict__ qq, const float* __restrict__ qd,
                                      float* __restrict__ s, int n, int n_kv, int G){
  /* scores from q8_0 K and q8_0 Q: per 32-block an exact int dot product times d_q * d_k, summed over the 8 blocks.
     grid (NHKV, <=32), block 256: a warp per position, lane = bytes 8l..8l+7 (block l / 4) */
  const int kv = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, blk = lane >> 2;
  int qv[8][8];
  float qs[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){
    const uint2 a = j < G ? qq[(kv * G + j) * 32 + lane] : make_uint2(0, 0);
#pragma unroll
    for(int i = 0; i < 4; ++i){
      qv[j][i] = (int)(signed char)((a.x >> (8 * i)) & 255u);
      qv[j][4 + i] = (int)(signed char)((a.y >> (8 * i)) & 255u);
    }
    qs[j] = j < G ? qd[(kv * G + j) * 8 + blk] : 0.f;
  }
  for(int t = blockIdx.y * 8 + warp; t < n; t += gridDim.y * 8){
    const uint2 w = Kq[((long long)t * n_kv + kv) * 32 + lane];
    const float dk = h2f(Kd[((long long)t * n_kv + kv) * 8 + blk]);
    int kk[8];
#pragma unroll
    for(int i = 0; i < 4; ++i){
      kk[i] = (int)(signed char)((w.x >> (8 * i)) & 255u);
      kk[4 + i] = (int)(signed char)((w.y >> (8 * i)) & 255u);
    }
#pragma unroll
    for(int j = 0; j < 8; ++j){
      if(j < G){
        int is = 0;
#pragma unroll
        for(int i = 0; i < 8; ++i) is += kk[i] * qv[j][i];
        is += __shfl_xor(is, 1);
        is += __shfl_xor(is, 2);
        float f = (float)is * (qs[j] * dk);
        f += __shfl_xor(f, 4);
        f += __shfl_xor(f, 8);
        f += __shfl_xor(f, 16);
        if(lane == 0) s[(long long)(kv * G + j) * n + t] = f * 0.0625f;
      }
    }
  }
}

extern "C" __global__ void k_attn_vq8(const float* __restrict__ s, const unsigned short* __restrict__ Vq2,
                                      const unsigned short* __restrict__ Vd, float* __restrict__ part,
                                      int n, int nheads, int n_kv, int G){
  /* k_attn_vgn over a q8_0 V cache: thread f2 = features 2f2, 2f2 + 1 (one int8 pair), their block's fp16 scale */
  if((int)blockIdx.z * ATV_CH >= n) return;
  __shared__ float sp[ATV_CH * 8];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) sp[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float a[8], b[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){ a[j] = 0.f; b[j] = 0.f; }
  const int blk = (2 * f2) >> 5;
  for(int i = 0; i < m; ++i){
    const long long r = (long long)(i0 + i) * n_kv + kv;
    const unsigned short vv = Vq2[r * 128 + f2];
    const float d = h2f(Vd[r * 8 + blk]);
    const float v0 = (float)(signed char)(vv & 255u) * d, v1 = (float)(signed char)(vv >> 8) * d;
#pragma unroll
    for(int j = 0; j < 8; ++j){ const float p = sp[i * 8 + j]; a[j] += p * v0; b[j] += p * v1; }
  }
#pragma unroll
  for(int j = 0; j < 8; ++j){
    if(j < G){
      float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 2 * f2;
      pp[0] = a[j]; pp[1] = b[j];
    }
  }
}

/* ---------------------------------------------------------------- 27B: faster attention and embedding (same results) */

__device__ __forceinline__ float sb2f(unsigned int u){
  /* the signed byte in the low 8 bits of u as a float, exactly: 2^23 + (b + 128) - (2^23 + 128); an OR and an add
     instead of an integer-to-float conversion */
  return __int_as_float(0x4B000000 | ((u & 255u) ^ 128u)) - 8388736.0f;
}

extern "C" __global__ void k_attn_sq8f(const uint2* __restrict__ Kq, const unsigned short* __restrict__ Kd,
                                       const uint2* __restrict__ qq, const float* __restrict__ qd,
                                       float* __restrict__ s, int n, int n_kv, int G){
  /* k_attn_sq8 with float multiply-adds: every product and every 32-value block sum of two int8 vectors is an
     integer below 2^24, which fp32 holds exactly, so the scores are bit-identical; Maxwell has no fast 32-bit
     integer multiply. grid (NHKV, <=32), block 256: a warp per position, lane = bytes 8l..8l+7 (block l / 4) */
  const int kv = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, blk = lane >> 2;
  float qv[8][8];
  float qs[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){
    const uint2 a = j < G ? qq[(kv * G + j) * 32 + lane] : make_uint2(0, 0);
#pragma unroll
    for(int i = 0; i < 4; ++i){
      qv[j][i] = sb2f(a.x >> (8 * i));
      qv[j][4 + i] = sb2f(a.y >> (8 * i));
    }
    qs[j] = j < G ? qd[(kv * G + j) * 8 + blk] : 0.f;
  }
  for(int t = blockIdx.y * 8 + warp; t < n; t += gridDim.y * 8){
    const uint2 w = KLDG(&Kq[((long long)t * n_kv + kv) * 32 + lane]);
    const float dk = h2f(KLDG(&Kd[((long long)t * n_kv + kv) * 8 + blk]));
    float kk[8];
#pragma unroll
    for(int i = 0; i < 4; ++i){
      kk[i] = sb2f(w.x >> (8 * i));
      kk[4 + i] = sb2f(w.y >> (8 * i));
    }
#pragma unroll
    for(int j = 0; j < 8; ++j){
      if(j < G){
        float is = 0.f;
#pragma unroll
        for(int i = 0; i < 8; ++i) is += kk[i] * qv[j][i];
        is += __shfl_xor(is, 1);
        is += __shfl_xor(is, 2);
        float f = is * (qs[j] * dk);
        f += __shfl_xor(f, 4);
        f += __shfl_xor(f, 8);
        f += __shfl_xor(f, 16);
        if(lane == 0) s[(long long)(kv * G + j) * n + t] = f * 0.0625f;
      }
    }
  }
}

extern "C" __global__ void k_attn_vq8f(const float* __restrict__ s, const unsigned short* __restrict__ Vq2,
                                       const unsigned short* __restrict__ Vd, float* __restrict__ part,
                                       int n, int nheads, int n_kv, int G){
  /* k_attn_vq8 with the bytes widened by sb2f (exact) and the 8 probabilities of a position read as two float4 */
  if((int)blockIdx.z * ATV_CH >= n) return;
  __shared__ float4 sp[ATV_CH * 2];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  float* spf = (float*)sp;
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) spf[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float a[8], b[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){ a[j] = 0.f; b[j] = 0.f; }
  const int blk = (2 * f2) >> 5;
  for(int i = 0; i < m; ++i){
    const long long r = (long long)(i0 + i) * n_kv + kv;
    const unsigned int vv = KLDG(&Vq2[r * 128 + f2]);
    const float d = h2f(KLDG(&Vd[r * 8 + blk]));
    const float v0 = sb2f(vv) * d, v1 = sb2f(vv >> 8) * d;
    const float4 p0 = sp[i * 2], p1 = sp[i * 2 + 1];
    const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
    for(int j = 0; j < 8; ++j){ a[j] += p[j] * v0; b[j] += p[j] * v1; }
  }
#pragma unroll
  for(int j = 0; j < 8; ++j){
    if(j < G){
      float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 2 * f2;
      pp[0] = a[j]; pp[1] = b[j];
    }
  }
}

extern "C" __global__ void k_deq_emb4(const unsigned char* __restrict__ bc, const unsigned short* __restrict__ dpb,
                                      const float* __restrict__ sg, float* __restrict__ h, int row, int C, int HB){
  /* k_deq_emb3 for Hadamard blocks of HB = 1024, one block of 512 threads per Hadamard block (grid C / HB): the
     same dequantisation, butterflies and scaling per element, so the result is bit-identical */
  __shared__ float w[1024];
  const int t = threadIdx.x, c0 = blockIdx.x * HB;
  const long long r4 = row >> 2, rl = row & 3, g16 = row >> 4;
  const int rr = row & 15;
  const int lane_low = ((rr >> 3) & 1) | (((rr >> 2) & 1) << 1) | (((rr >> 1) & 1) << 2), jj = rr & 1;
  for(int q = t; q < HB; q += 512){
    const int e = c0 + q;
    const int s = e >> 2, p = e >> 7, r = e & 3;
    const unsigned int b = bc[((((long long)(row >> 4) * (C >> 9) + (s >> 7)) * 4 + (r4 & 3)) * 32 + ((s >> 2) & 31)) * 16
                              + rl * 4 + (s & 3)];
    const int dig = (r == 0) ? (b % 3) : ((r == 1) ? ((b / 3) % 3) : ((r == 2) ? ((b / 9) % 3) : ((b / 27) % 3)));
    const int lane = ((p & 3) << 3) | lane_low;
    w[q] = ((float)(dig - 1)) * h2f3(dpb[((g16 * (C >> 9) + (p >> 2)) * 32 + lane) * 2 + jj]);
  }
  __syncthreads();
  for(int hh = 1; hh < HB; hh <<= 1){
    for(int q = t; q < (HB >> 1); q += 512){
      const int i = (q / hh) * 2 * hh + (q % hh);
      const float a = w[i], bb = w[i + hh];
      w[i] = a + bb; w[i + hh] = a - bb;
    }
    __syncthreads();
  }
  const float sc = rsqrtf((float)HB);
  for(int q = t; q < HB; q += 512) h[c0 + q] = sg[c0 + q] * (w[q] * sc);
}

/* ---------------------------------------------------------------- 27B: attention scores, a thread per position */

extern "C" __global__ void __launch_bounds__(256) k_attn_sq8t(const uint4* __restrict__ Kq,
    const unsigned short* __restrict__ Kd, const signed char* __restrict__ qq, const float* __restrict__ qd,
    float* __restrict__ s, int n, int n_kv, int G){
  /* q8_0 attention scores with no cross-lane reductions: grid (NHKV, ceil(ctx / 256)), block 256, thread = one
     position, all G (<= 8) heads of the K/V head. The query (int8, exact as floats) and its block scales sit in
     shared memory and are read as float4 broadcasts. Per 32-value block, the sum of the 32 products is an exact
     integer (below 2^24), as in llama.cpp's q8_0 dot product; it is scaled by d_q * d_k and the 8 blocks are added
     in order. */
  if((int)blockIdx.y * 256 >= n) return;
  __shared__ float4 q4[8 * 64];
  __shared__ float dq[8 * 8];
  const int kv = blockIdx.x, tid = threadIdx.x, t = blockIdx.y * 256 + tid;
  float* qf = (float*)q4;
  for(int i = tid; i < G * 256; i += 256) qf[i] = (float)qq[kv * G * 256 + i];
  for(int i = tid; i < G * 8; i += 256) dq[i] = qd[kv * G * 8 + i];
  __syncthreads();
  if(t >= n) return;
  const uint4* row = Kq + ((long long)t * n_kv + kv) * 16;
  const unsigned short* drow = Kd + ((long long)t * n_kv + kv) * 8;
  float acc[8];
#pragma unroll
  for(int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll 2
  for(int b = 0; b < 8; ++b){
    float part[8];
#pragma unroll
    for(int j = 0; j < 8; ++j) part[j] = 0.f;
    const float dk = h2f(KLDG(&drow[b]));
#pragma unroll
    for(int hl = 0; hl < 2; ++hl){
      const uint4 w = KLDG(&row[b * 2 + hl]);
      const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
      float kf[16];
#pragma unroll
      for(int i = 0; i < 4; ++i)
#pragma unroll
        for(int k = 0; k < 4; ++k) kf[i * 4 + k] = sb2f(ww[i] >> (8 * k));
#pragma unroll
      for(int j = 0; j < 8; ++j){
        if(j < G){
          float p = part[j];
#pragma unroll
          for(int e = 0; e < 4; ++e){
            const float4 q = q4[j * 64 + b * 8 + hl * 4 + e];
            p += kf[e * 4] * q.x; p += kf[e * 4 + 1] * q.y; p += kf[e * 4 + 2] * q.z; p += kf[e * 4 + 3] * q.w;
          }
          part[j] = p;
        }
      }
    }
#pragma unroll
    for(int j = 0; j < 8; ++j) if(j < G) acc[j] += part[j] * (dq[j * 8 + b] * dk);
  }
#pragma unroll
  for(int j = 0; j < 8; ++j) if(j < G) s[(long long)(kv * G + j) * n + t] = acc[j] * 0.0625f;
}

extern "C" __global__ void k_attn_vq8u(const float* __restrict__ s, const unsigned short* __restrict__ Vq2,
                                       const unsigned short* __restrict__ Vd, float* __restrict__ part,
                                       int n, int nheads, int n_kv, int G){
  /* k_attn_vq8f with the position loop unrolled by 4 (four V loads in flight per thread) */
  if((int)blockIdx.z * ATV_CH >= n) return;
  __shared__ float4 sp[ATV_CH * 2];
  const int kv = blockIdx.x, f2 = threadIdx.x, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  float* spf = (float*)sp;
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) spf[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float a[8], b[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){ a[j] = 0.f; b[j] = 0.f; }
  const int blk = (2 * f2) >> 5;
  const unsigned short* vb = Vq2 + ((long long)i0 * n_kv + kv) * 128 + f2;
  const unsigned short* db = Vd + ((long long)i0 * n_kv + kv) * 8 + blk;
  const long long rs = (long long)n_kv * 128, ds = (long long)n_kv * 8;
#pragma unroll 4
  for(int i = 0; i < m; ++i){
    const unsigned int vv = KLDG(vb + i * rs);
    const float d = h2f(KLDG(db + i * ds));
    const float v0 = sb2f(vv) * d, v1 = sb2f(vv >> 8) * d;
    const float4 p0 = sp[i * 2], p1 = sp[i * 2 + 1];
    const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
    for(int j = 0; j < 8; ++j){ a[j] += p[j] * v0; b[j] += p[j] * v1; }
  }
#pragma unroll
  for(int j = 0; j < 8; ++j){
    if(j < G){
      float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 2 * f2;
      pp[0] = a[j]; pp[1] = b[j];
    }
  }
}

extern "C" __global__ void __launch_bounds__(512) k_attn_vq8g(const float* __restrict__ s,
    const unsigned short* __restrict__ Vq2, const unsigned short* __restrict__ Vd, float* __restrict__ part,
    int n, int nheads, int n_kv, int G){
  /* k_attn_vq8u with 4 thread groups per chunk: group g (threads 128 g .. 128 g + 127) takes positions g, g+4, ...
     of the chunk; the groups' sums are added in group order at the end. Same output layout as k_attn_vq8. */
  if((int)blockIdx.z * ATV_CH >= n) return;
  __shared__ float4 sp[ATV_CH * 2];
  __shared__ float red[3 * 16 * 128];
  const int kv = blockIdx.x, f2 = threadIdx.x & 127, grp = threadIdx.x >> 7, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  float* spf = (float*)sp;
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) spf[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float a[8], b[8];
#pragma unroll
  for(int j = 0; j < 8; ++j){ a[j] = 0.f; b[j] = 0.f; }
  const int blk = (2 * f2) >> 5;
  const unsigned short* vb = Vq2 + ((long long)i0 * n_kv + kv) * 128 + f2;
  const unsigned short* db = Vd + ((long long)i0 * n_kv + kv) * 8 + blk;
  const long long rs = (long long)n_kv * 128, ds = (long long)n_kv * 8;
#pragma unroll 2
  for(int i = grp; i < m; i += 4){
    const unsigned int vv = KLDG(vb + i * rs);
    const float d = h2f(KLDG(db + i * ds));
    const float v0 = sb2f(vv) * d, v1 = sb2f(vv >> 8) * d;
    const float4 p0 = sp[i * 2], p1 = sp[i * 2 + 1];
    const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
    for(int j = 0; j < 8; ++j){ a[j] += p[j] * v0; b[j] += p[j] * v1; }
  }
  if(grp > 0){
    float* r = red + (grp - 1) * 16 * 128;
#pragma unroll
    for(int j = 0; j < 8; ++j){ r[j * 128 + f2] = a[j]; r[(8 + j) * 128 + f2] = b[j]; }
  }
  __syncthreads();
  if(grp == 0){
#pragma unroll
    for(int g = 0; g < 3; ++g){
      const float* r = red + g * 16 * 128;
#pragma unroll
      for(int j = 0; j < 8; ++j){ a[j] += r[j * 128 + f2]; b[j] += r[(8 + j) * 128 + f2]; }
    }
#pragma unroll
    for(int j = 0; j < 8; ++j){
      if(j < G){
        float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 2 * f2;
        pp[0] = a[j]; pp[1] = b[j];
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(1024) k_xrmsN(
    const float* __restrict__ pin0, int pstride, int nparts, float* hb0, volatile int* flags0, int me, int* fail,
    int* ctr0, const float* __restrict__ h0, float* __restrict__ h20, const float* __restrict__ w,
    const float* __restrict__ sg, float* __restrict__ xn0, float* __restrict__ xo0, int xostride, int C){
  /* k_xrms for a batch of prompt tokens: block b = token b, with its own partials (pin0 + b * pstride), its own host
     slots (hb0 + (1 + b) * 4C), flag pair (flags0[2 (1 + b)]) and epoch counter (ctr0[1 + b]); slot 0 stays with the
     one-token k_xrms. Each block does exactly what k_xrms does for one token, so the results are the same. */
  __shared__ float X[6144];
  __shared__ double red[32];
  __shared__ int ok, ep;
  __shared__ float bscale;
  const int t = threadIdx.x, lane = t & 31, wp = t >> 5, b = blockIdx.x;
  const float* pin = pin0 + (long long)b * pstride;
  float* hb = hb0 + (long long)(1 + b) * 4 * C;
  volatile int* flags = flags0 + 2 * (1 + b);
  int* ctr = ctr0 + 1 + b;
  const float* h = h0 + (long long)b * C;
  float* h2 = h20 + (long long)b * C;
  float* xn = xn0 + (long long)b * C;
  float* xo = xo0 + (long long)b * xostride;
  if(t == 0) ep = ++ctr[0];
  __syncthreads();
  const int epoch = ep, par = epoch & 1;
  float* hm = hb + (long long)(par * 2 + me) * C;
  const float* ho = hb + (long long)(par * 2 + (1 - me)) * C;
  for(int e = t * 4; e < C; e += 4096){
    float4 v = *(const float4*)(pin + e);
    for(int i = 1; i < nparts; ++i){
      const float4 q = *(const float4*)(pin + (long long)i * C + e);
      v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
    }
    *(float4*)(hm + e) = v;
    X[e] = v.x; X[e + 1] = v.y; X[e + 2] = v.z; X[e + 3] = v.w;
  }
  __threadfence_system();
  __syncthreads();
  if(t == 0){
    flags[me] = epoch;
    __threadfence_system();
    long long spins = 0; ok = 1;
    while(flags[1 - me] < epoch){ if(++spins > 4000000000LL){ *fail = epoch; ok = 0; break; } }
  }
  __syncthreads();
  if(!ok) return;
  double acc = 0.0;
  for(int e = t * 4; e < C; e += 4096){
    float a, bb, c, d;
#if KHIP
    { float4 v4 = *(const float4*)(ho + e); a = v4.x; bb = v4.y; c = v4.z; d = v4.w; }
#else
    asm volatile("ld.global.cv.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(a), "=f"(bb), "=f"(c), "=f"(d) : "l"(ho + e));
#endif
    const float o[4] = {a, bb, c, d};
#pragma unroll
    for(int k = 0; k < 4; ++k){
      const float p0 = me == 0 ? X[e + k] : o[k], p1 = me == 0 ? o[k] : X[e + k];
      const float xf = h[e + k] + (p0 + p1);
      X[e + k] = xf;
      h2[e + k] = xf;
      acc += (double)xf * (double)xf;
    }
  }
  acc = wsum_d(acc);
  if(lane == 0) red[wp] = acc;
  __syncthreads();
  if(wp == 0){
    double s2 = red[lane];
    s2 = wsum_d(s2);
    if(lane == 0) bscale = 1.0f / sqrtf((float)(s2 / C) + 1e-6f);
  }
  __syncthreads();
  const float s = bscale;
  for(int e = t; e < C; e += 1024){
    const float v = (X[e] * s) * w[e];
    xn[e] = v;
    X[e] = sg[e] * v;
  }
  __syncthreads();
  const int nb = C >> 10, nbf = nb * 512;
  for(int hh = 1; hh < 1024; hh <<= 1){
    for(int q = t; q < nbf; q += 1024){
      const int blk = q >> 9, qq = q & 511;
      const int i = (blk << 10) + (qq / hh) * 2 * hh + (qq % hh);
      const float a = X[i], bb = X[i + hh];
      X[i] = a + bb; X[i + hh] = a - bb;
    }
    __syncthreads();
  }
  const float sc = rsqrtf(1024.0f);
  for(int e = t; e < C; e += 1024) xo[e] = X[e] * sc;
}

extern "C" __global__ void __launch_bounds__(128) k_attn_vq8w(const float* __restrict__ s,
    const uint2* __restrict__ Vq8, const unsigned short* __restrict__ Vd, float* __restrict__ part,
    int n, int nheads, int n_kv, int G){
  /* q8_0 attention values, 8 features per thread: grid (NHKV, 1, nchunks), block 128 = 4 groups of 32 threads; a
     group covers one position's 256 features (thread = 8 features, one 8-byte load, inside one 32-value scale
     block) and takes positions grp, grp+4, ... of the chunk. The 8 probabilities of a position are two float4
     broadcasts, now shared by 8 features instead of 2. The groups' sums are added in group order. */
  if((int)blockIdx.z * ATV_CH >= n) return;
  __shared__ float4 sp[ATV_CH * 2];
  __shared__ float red[3 * 8 * 256];
  const int kv = blockIdx.x, f8 = threadIdx.x & 31, grp = threadIdx.x >> 5, c = blockIdx.z;
  const int i0 = c * ATV_CH, m = min(ATV_CH, n - i0);
  float* spf = (float*)sp;
  for(int i = threadIdx.x; i < ATV_CH; i += blockDim.x)
#pragma unroll
    for(int j = 0; j < 8; ++j) spf[i * 8 + j] = (i < m && j < G) ? s[(long long)(kv * G + j) * n + i0 + i] : 0.f;
  __syncthreads();
  float acc[8][8];
#pragma unroll
  for(int j = 0; j < 8; ++j)
#pragma unroll
    for(int e = 0; e < 8; ++e) acc[j][e] = 0.f;
  const int blk = (8 * f8) >> 5;
  const uint2* vb = Vq8 + ((long long)i0 * n_kv + kv) * 32 + f8;
  const unsigned short* db = Vd + ((long long)i0 * n_kv + kv) * 8 + blk;
  const long long rs = (long long)n_kv * 32, ds = (long long)n_kv * 8;
#pragma unroll 4
  for(int i = grp; i < m; i += 4){
    const uint2 w = KLDG(vb + i * rs);
    const float d = h2f(KLDG(db + i * ds));
    float v[8];
#pragma unroll
    for(int k = 0; k < 4; ++k){ v[k] = sb2f(w.x >> (8 * k)) * d; v[4 + k] = sb2f(w.y >> (8 * k)) * d; }
    const float4 p0 = sp[i * 2], p1 = sp[i * 2 + 1];
    const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
#pragma unroll
    for(int j = 0; j < 8; ++j)
#pragma unroll
      for(int e = 0; e < 8; ++e) acc[j][e] += p[j] * v[e];
  }
  if(grp > 0){
    float* r = red + (grp - 1) * 8 * 256;
#pragma unroll
    for(int j = 0; j < 8; ++j)
#pragma unroll
      for(int e = 0; e < 8; ++e) r[j * 256 + 8 * f8 + e] = acc[j][e];
  }
  __syncthreads();
  if(grp == 0){
#pragma unroll
    for(int g = 0; g < 3; ++g){
      const float* r = red + g * 8 * 256;
#pragma unroll
      for(int j = 0; j < 8; ++j)
#pragma unroll
        for(int e = 0; e < 8; ++e) acc[j][e] += r[j * 256 + 8 * f8 + e];
    }
#pragma unroll
    for(int j = 0; j < 8; ++j){
      if(j < G){
        float* pp = part + ((long long)c * nheads + kv * G + j) * 256 + 8 * f8;
#pragma unroll
        for(int e = 0; e < 8; ++e) pp[e] = acc[j][e];
      }
    }
  }
}

/* ---------------------------------------------------------------- 5 trits per byte, a row per thread (Pascal) */
/* Weights in Prism's PTQ1_0 block order, regrouped group-major: for each 128-column group g, every row's 28-byte block
   [24 bytes: 5 trits each (columns 5j .. 5j+4, j < 24)][2 bytes: 4 trits each (columns 120 + 4m .. 120 + 4m + 3)]
   [fp16 scale], rows consecutive. 1.75 bits per weight. A byte is an index into a sum table of its columns'
   activations: 24 tables of 243 entries and 2 of 81 per group, built once per group per block in shared memory and
   shared by every row of the block - no decode arithmetic. A thread holds RPT rows; it adds its rows' 26 lookups,
   scales the group sum and adds it to the row total. */

#define G5_TAB (24 * 243 + 2 * 81)

__device__ __forceinline__ void g5_tables(float* T, float* AB, const float* __restrict__ x, int g, int t, int nt){
  /* the tables of group g: T[j*243 + c] = sum_i (trit_i(c) - 1) * x[128 g + 5 j + i], c = sum_i trit_i * 3^i.
     Two stages: AB holds, per 5-column position j, the 27 sums of its first 3 columns and the 9 sums of its last 2
     (and the two 4-column tables are built directly); then every 5-column entry is one add, A[c % 27] + B[c / 27]. */
  const float* xg = x + g * 128;
  for(int e = t; e < 24 * 36 + 2 * 81; e += nt){
    if(e < 24 * 36){
      const int j = e / 36, k = e - j * 36;
      float s;
      if(k < 27){
        const int t0 = k % 3, t1 = (k / 3) % 3, t2 = k / 9;
        s = (float)(t0 - 1) * xg[5 * j] + (float)(t1 - 1) * xg[5 * j + 1] + (float)(t2 - 1) * xg[5 * j + 2];
      } else {
        const int k2 = k - 27, t3 = k2 % 3, t4 = k2 / 3;
        s = (float)(t3 - 1) * xg[5 * j + 3] + (float)(t4 - 1) * xg[5 * j + 4];
      }
      AB[e] = s;
    } else {
      const int e2 = e - 24 * 36, m = e2 / 81;
      int c = e2 - m * 81;
      float s = 0.f;
#pragma unroll
      for(int i = 0; i < 4; ++i){ const int tr = c % 3; c /= 3; s += (float)(tr - 1) * xg[120 + 4 * m + i]; }
      T[24 * 243 + e2] = s;
    }
  }
  __syncthreads();
  for(int e = t; e < 24 * 243; e += nt){
    const int j = e / 243, c = e - j * 243;
    T[e] = AB[j * 36 + c % 27] + AB[j * 36 + 27 + c / 27];
  }
}

template <int RPT>
__device__ __forceinline__ void g5_core(const unsigned int* __restrict__ W, float* __restrict__ y, int R, int NG,
                                        const float* __restrict__ x, int ylo){
  /* grid (ceil(R / (blockDim * RPT)), splits): block x covers rows [bx * nt * RPT, ...), block y the groups
     [by * NG / splits, ...); its row sums go to y + by * R (the parts, added later in a fixed order) */
  __shared__ float T[G5_TAB];
  __shared__ float AB[24 * 36];
  const int t = threadIdx.x, nt = blockDim.x;
  const int g_lo = (int)((long long)blockIdx.y * NG / gridDim.y), g_hi = (int)((long long)(blockIdx.y + 1) * NG / gridDim.y);
  const int r0 = blockIdx.x * nt * RPT + t;
  float acc[RPT];
#pragma unroll
  for(int k = 0; k < RPT; ++k) acc[k] = 0.f;
  const char* Tc = (const char*)T;
  for(int g = g_lo; g < g_hi; ++g){
    __syncthreads();
    g5_tables(T, AB, x, g, t, nt);
    __syncthreads();
#pragma unroll
    for(int k = 0; k < RPT; ++k){
      const int r = r0 + k * nt;
      if(r < R){
        const unsigned int* b = W + ((long long)g * R + r) * 7;
        unsigned int w[7];
#pragma unroll
        for(int i = 0; i < 7; ++i) w[i] = KLDG(b + i);
        float s = 0.f;
#pragma unroll
        for(int i = 0; i < 6; ++i){
#pragma unroll
          for(int q = 0; q < 4; ++q){
            const unsigned int c = (w[i] >> (8 * q)) & 255u;
            s += *(const float*)(Tc + (((i * 4 + q) * 243 + c) << 2));
          }
        }
        const unsigned int c0 = w[6] & 255u, c1 = (w[6] >> 8) & 255u;
        s += T[24 * 243 + c0];
        s += T[24 * 243 + 81 + c1];
        acc[k] += s * h2f((unsigned short)(w[6] >> 16));
      }
    }
  }
#pragma unroll
  for(int k = 0; k < RPT; ++k){
    const int r = r0 + k * nt;
    if(r < R) y[(long long)blockIdx.y * R + r] = acc[k];
  }
}

extern "C" __global__ void __launch_bounds__(512) k_g5r4(const unsigned int* __restrict__ W, float* __restrict__ y,
                                                         int R, int NG, const float* __restrict__ x){
  g5_core<4>(W, y, R, NG, x, 0);
}
extern "C" __global__ void __launch_bounds__(512) k_g5r2(const unsigned int* __restrict__ W, float* __restrict__ y,
                                                         int R, int NG, const float* __restrict__ x){
  g5_core<2>(W, y, R, NG, x, 0);
}
extern "C" __global__ void __launch_bounds__(512) k_g5r8(const unsigned int* __restrict__ W, float* __restrict__ y,
                                                         int R, int NG, const float* __restrict__ x){
  g5_core<8>(W, y, R, NG, x, 0);
}
extern "C" __global__ void __launch_bounds__(512) k_g5r1(const unsigned int* __restrict__ W, float* __restrict__ y,
                                                         int R, int NG, const float* __restrict__ x){
  g5_core<1>(W, y, R, NG, x, 0);
}

/* ---------------------------------------------------------------- q8_0 attention scores with dp4a (sm_61+) */

/* dp4a as inline PTX: NVRTC 10.2 does not define the __dp4a intrinsic (sm_61_intrinsics are not included) */
__device__ __forceinline__ int kdp4a(int a, int b, int c){
  /* dp4a.s32.s32: four SIGNED int8 x int8 products added to c */
#if KHIP
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);    /* v_dot4_i32_iu8, both operands signed, no clamp */
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  int d; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
#else
  return c + (int)(signed char)(a & 0xff) * (int)(signed char)(b & 0xff) +
      (int)(signed char)((a >> 8) & 0xff) * (int)(signed char)((b >> 8) & 0xff) +
      (int)(signed char)((a >> 16) & 0xff) * (int)(signed char)((b >> 16) & 0xff) +
      (int)(signed char)((a >> 24) & 0xff) * (int)(signed char)((b >> 24) & 0xff);
#endif
}

extern "C" __global__ void __launch_bounds__(256) k_attn_sq8d(const uint4* __restrict__ Kq,
    const unsigned short* __restrict__ Kd, const int* __restrict__ qq, const float* __restrict__ qd,
    float* __restrict__ s, int n, int n_kv, int G){
  /* k_attn_sq8t with the 32-value block dot products as dp4a (4 int8 x int8 products into an int32): the block
     sums are the same exact integers, scaled and added over the 8 blocks in the same order, so the scores are
     bit-identical to k_attn_sq8t. On cards without dp4a (below sm_61) it does the same sums with integer
     multiply-adds. grid (NHKV, ceil(ctx / 256)), block 256, thread = one position. */
  if((int)blockIdx.y * 256 >= n) return;
  __shared__ int4 q4[8 * 16];                 /* G heads x 256 int8 = 16 int4 per head */
  __shared__ float dq[8 * 8];
  const int kv = blockIdx.x, tid = threadIdx.x, t = blockIdx.y * 256 + tid;
  int* qi = (int*)q4;
  for(int i = tid; i < G * 64; i += 256) qi[i] = qq[kv * G * 64 + i];
  for(int i = tid; i < G * 8; i += 256) dq[i] = qd[kv * G * 8 + i];
  __syncthreads();
  if(t >= n) return;
  const uint4* row = Kq + ((long long)t * n_kv + kv) * 16;
  const unsigned short* drow = Kd + ((long long)t * n_kv + kv) * 8;
  float acc[8];
#pragma unroll
  for(int j = 0; j < 8; ++j) acc[j] = 0.f;
#pragma unroll 2
  for(int b = 0; b < 8; ++b){
    int part[8];
#pragma unroll
    for(int j = 0; j < 8; ++j) part[j] = 0;
    const float dk = h2f(KLDG(&drow[b]));
#pragma unroll
    for(int hl = 0; hl < 2; ++hl){
      const uint4 w = KLDG(&row[b * 2 + hl]);
#pragma unroll
      for(int j = 0; j < 8; ++j){
        if(j < G){
          const int4 q = q4[j * 16 + b * 2 + hl];
#if __CUDA_ARCH__ >= 610
          int p = part[j];
          p = kdp4a((int)w.x, q.x, p); p = kdp4a((int)w.y, q.y, p);
          p = kdp4a((int)w.z, q.z, p); p = kdp4a((int)w.w, q.w, p);
          part[j] = p;
#else
          const unsigned int ww[4] = {w.x, w.y, w.z, w.w};
          const int qv[4] = {q.x, q.y, q.z, q.w};
          int p = part[j];
#pragma unroll
          for(int u = 0; u < 4; ++u)
#pragma unroll
            for(int k = 0; k < 4; ++k)
              p += (int)(signed char)((ww[u] >> (8 * k)) & 255u) * (int)(signed char)((((unsigned int)qv[u]) >> (8 * k)) & 255u);
          part[j] = p;
#endif
        }
      }
    }
#pragma unroll
    for(int j = 0; j < 8; ++j) if(j < G) acc[j] += (float)part[j] * (dq[j * 8 + b] * dk);
  }
#pragma unroll
  for(int j = 0; j < 8; ++j) if(j < G) s[(long long)(kv * G + j) * n + t] = acc[j] * 0.0625f;
}

/* ---------------------------------------------------------------- 5 trits per byte, a byte position per lane (Pascal) */
/* The same 28-byte blocks as k_g5r, stored in 16-row tiles byte-position-major: for group g and tile m (rows 16m ..
   16m+15), 28 uint4 words, word p = byte p of the 16 rows. Lane p of a warp owns byte position p (p < 24: columns
   5p .. 5p+4; p = 24, 25: the two 4-column bytes; p = 26, 27: the fp16 scale, low and high byte), so it only ever looks
   into its own 256-entry table, stored at entry * 32 + lane: every lookup of a warp hits 32 different banks. The
   lanes multiply their lookups by the row's scale (from lanes 26, 27 by shuffle) and keep per-row sums; the 16 rows
   of a tile are summed over the lanes once at the end (16 shuffles). Lanes 26 .. 31 have all-zero tables. */

__device__ __forceinline__ void g5n_tables(float* T, const float* __restrict__ x, int g, int lane, int wp, int nw){
  /* lane p < 26 rebuilds its own table: T[(a + 27 b) * 32 + p] = A(a) + B(b), A = the 27 sums of its first 3 columns
     (in registers), B = the 9 sums of its last 2; warp w does b = w, w + nw, ... No divisions, one pass. */
  if(lane >= 26 || wp >= 9) return;
  const float* xg = x + g * 128;
  float x0, x1, x2, x3, x4;
  if(lane < 24){
    x0 = KLDG(xg + 5 * lane); x1 = KLDG(xg + 5 * lane + 1); x2 = KLDG(xg + 5 * lane + 2);
    x3 = KLDG(xg + 5 * lane + 3); x4 = KLDG(xg + 5 * lane + 4);
  } else {
    const float* x8 = xg + 120 + 4 * (lane - 24);
    x0 = KLDG(x8); x1 = KLDG(x8 + 1); x2 = KLDG(x8 + 2); x3 = KLDG(x8 + 3); x4 = 0.f;
  }
  float A[27];
#pragma unroll
  for(int k = 0; k < 27; ++k){
    const int t0 = k % 3, t1 = (k / 3) % 3, t2 = k / 9;
    A[k] = (float)(t0 - 1) * x0 + (float)(t1 - 1) * x1 + (float)(t2 - 1) * x2;
  }
  for(int b = wp; b < 9; b += nw){
    const int t3 = b % 3, t4 = b / 3;
    const float B = (float)(t3 - 1) * x3 + (float)(t4 - 1) * x4;
    float* Tb = T + 27 * b * 32 + lane;
#pragma unroll
    for(int a = 0; a < 27; ++a) Tb[a * 32] = A[a] + B;
  }
}

template <int U>
__device__ __forceinline__ void g5n_core(const uint4* __restrict__ W, float* __restrict__ y, int R, int NG,
                                         const float* __restrict__ x){
  /* grid (ceil(R / (16 U warps)), splits); block y covers groups [by NG / splits, ...) and writes y + by R */
  __shared__ float T[256 * 32];
  const int t = threadIdx.x, nt = blockDim.x, lane = t & 31, nw = nt >> 5;
  const int NT16 = R >> 4;
  const int g_lo = (int)((long long)blockIdx.y * NG / gridDim.y), g_hi = (int)((long long)(blockIdx.y + 1) * NG / gridDim.y);
  const int m0 = (blockIdx.x * nw + (t >> 5)) * U;
  float acc[U][16];
#pragma unroll
  for(int u = 0; u < U; ++u)
#pragma unroll
    for(int k = 0; k < 16; ++k) acc[u][k] = 0.f;
  const char* Tl = (const char*)(T + lane);
  for(int e = t; e < 256 * 32; e += nt) T[e] = 0.f;   /* entries 243 .. 255 and lanes 26 .. 31 stay zero */
  for(int g = g_lo; g < g_hi; ++g){
    __syncthreads();
    g5n_tables(T, x, g, lane, t >> 5, nw);
    __syncthreads();
    uint4 w[U];
#pragma unroll
    for(int u = 0; u < U; ++u){
      const int m = m0 + u;
      w[u] = make_uint4(0u, 0u, 0u, 0u);
      if(m < NT16 && lane < 28) w[u] = KLDG(W + ((long long)g * NT16 + m) * 28 + lane);
    }
#pragma unroll
    for(int u = 0; u < U; ++u){
      const unsigned int wv[4] = {w[u].x, w[u].y, w[u].z, w[u].w};
#pragma unroll
      for(int i = 0; i < 4; ++i){
        const unsigned int lo = KSHFL(wv[i], 26), hi = KSHFL(wv[i], 27);
#pragma unroll
        for(int q = 0; q < 4; ++q){
          const unsigned int c = (wv[i] >> (8 * q)) & 255u;
          const float v = *(const float*)(Tl + (c << 7));
          const float d = h2f((unsigned short)__byte_perm(lo, hi, q | ((4 + q) << 4)));
          acc[u][4 * i + q] += v * d;
        }
      }
    }
  }
#pragma unroll
  for(int u = 0; u < U; ++u){
    float v[16];
#pragma unroll
    for(int k = 0; k < 16; ++k) v[k] = acc[u][k];
#pragma unroll
    for(int lv = 0; lv < 4; ++lv){
      const int h = 8 >> lv, msk = 16 >> lv;
      const bool up = (lane & msk) != 0;
#pragma unroll
      for(int j = 0; j < 8; ++j){
        if(j < h){
          const float send = up ? v[j] : v[j + h], keep = up ? v[j + h] : v[j];
          v[j] = keep + KSHFLX(send, msk);
        }
      }
    }
    const float s = v[0] + KSHFLX(v[0], 1);
    const int m = m0 + u;
    if(m < NT16 && !(lane & 1)) y[(long long)blockIdx.y * R + m * 16 + ((lane >> 1) & 15)] = s;
  }
}

extern "C" __global__ void __launch_bounds__(256) k_g5n4(const uint4* __restrict__ W, float* __restrict__ y,
                                                         int R, int NG, const float* __restrict__ x){
  /* 256 threads, 4 tiles (64 rows) per warp: the measured best on the GTX 1080 */
  g5n_core<4>(W, y, R, NG, x);
}
