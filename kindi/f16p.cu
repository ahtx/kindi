/* f16p.cu - the fp16 prompt path of the chunk plan (chunk.py, KINDI_P100_FP16_PROMPT=1): the products of a prompt
   chunk as cuBLAS fp16 GEMMs (P100: fp16 at twice the fp32 rate).

   Per weight and chunk, for each piece of 512 columns (4 groups of 128): k_f16_dq writes the piece of the weights
   as fp16 rows, W16[r][c] = (digit - 1) * scale; cublasGemmEx multiplies it with the fp16 inputs (fp16 sums inside
   the piece); k_f16_acc adds the fp16 piece result to the fp32 output. So the sums across the pieces are fp32.
   k_f16_x converts the fp32 inputs of a chunk to fp16 once per input.

   Weights, two layouts. (1) The 5-trit lane layout of the card (sm_60 keeps it; k_f16_dq5): per 128-column group G
   and 16-row tile m, 28 bytes of each row, byte p of row rr at ((G NT16 + m) 28 + p) 16 + rr. Byte p < 24 holds the
   digits of columns 5 p .. 5 p + 4, p = 24, 25 those of columns 120 + 4 (p - 24) .. + 3, as x = sum d_e 3^e (e = the
   column in the byte); bytes 26 and 27 are the fp16 scale. (2) The "pl" layout of tcp.cu (k_tc_pack, k_f16_dq). Per 16-row tile m and 128-column group g, a block of 112 words
   (tile-major: block (m NG + g)). Word k < 6 of row rr is at 32 (k >> 1) + 2 rr + (k & 1), word 6 at 96 + rr.
   Word k < 6: byte b has the five digits q = 0 .. 4 of columns 20 k + 4 i + j, with Q = q + 5 (b & 1), i = Q >> 1,
   j = 2 (Q & 1) + (b >> 1). Word 6: byte b < 2 has the four digits of columns 120 + 2 q + b; bytes 2 and 3 are the
   fp16 scale of the row in this group. A byte c with n digits is ceil(v 256 / 3^n) for v = sum d_q 3^(n - 1 - q):
   multiplying by 3 moves the next digit into the high byte (t = 3 t; d = t >> 8; t &= 255). */
#include <cuda_fp16.h>

/* one thread per (row, group, byte p < 26), p fastest (each thread writes 5 or 4 consecutive columns); rows
   [0, R), groups [g0, g0 + ng); W16 is [R][128 ng] */
extern "C" __global__ void k_f16_dq5(const unsigned char* __restrict__ S, int R, int NG, int g0, int ng,
                                     __half* __restrict__ W16) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= (long long)R * ng * 26) return;
  const int p = (int)(idx % 26);
  const long long rg = idx / 26;
  const int gl = (int)(rg % ng), r = (int)(rg / ng), g = g0 + gl, NT16 = R >> 4, m = r >> 4, rr = r & 15;
  const unsigned char* blk = S + ((long long)g * NT16 + m) * 28 * 16 + rr;
  const float sc = __half2float(__ushort_as_half((unsigned short)(blk[26 * 16] | (blk[27 * 16] << 8))));
  unsigned int x = blk[p * 16];
  const int n = p < 24 ? 5 : 4, c0 = p < 24 ? 5 * p : 120 + 4 * (p - 24);
  __half* out = W16 + (long long)r * (128 * ng) + 128 * gl + c0;
  for(int e = 0; e < n; ++e){
    out[e] = __float2half(sc * (float)((int)(x % 3u) - 1));
    x /= 3u;
  }
}

/* one thread per (row, group, word); rows [0, R), groups [g0, g0 + ng); W16 is [R][128 ng] */
extern "C" __global__ void k_f16_dq(const unsigned int* __restrict__ Wp, int R, int NG, int g0, int ng,
                                    __half* __restrict__ W16) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= (long long)R * ng * 7) return;
  const int k = (int)(idx % 7);
  const long long rg = idx / 7;
  const int gl = (int)(rg % ng), r = (int)(rg / ng), g = g0 + gl;
  const int m = r >> 4, rr = r & 15;
  const unsigned int* blk = Wp + ((long long)m * NG + g) * 112;
  const unsigned int w6 = blk[96 + rr];
  const float sc = __half2float(__ushort_as_half((unsigned short)(w6 >> 16)));
  __half* out = W16 + (long long)r * (128 * ng) + 128 * gl;
  if(k < 6){
    const unsigned int w = blk[32 * (k >> 1) + 2 * rr + (k & 1)];
    for(int b = 0; b < 4; ++b){
      unsigned int t = (w >> (8 * b)) & 255u;
      for(int q = 0; q < 5; ++q){
        t *= 3u;
        const int d = (int)(t >> 8);
        t &= 255u;
        const int Q = q + 5 * (b & 1), i = Q >> 1, j = 2 * (Q & 1) + (b >> 1);
        out[20 * k + 4 * i + j] = __float2half(sc * (float)(d - 1));
      }
    }
  } else {
    for(int b = 0; b < 2; ++b){
      unsigned int t = (w6 >> (8 * b)) & 255u;
      for(int q = 0; q < 4; ++q){
        t *= 3u;
        const int d = (int)(t >> 8);
        t &= 255u;
        out[120 + 2 * q + b] = __float2half(sc * (float)(d - 1));
      }
    }
  }
}

/* X16[t][c] = X[t][c] (fp16) for t < n, c < C */
extern "C" __global__ void k_f16_x(const float* __restrict__ X, int ldx, int C, int n, __half* __restrict__ X16) {
  const long long N = (long long)n * C;
  for(long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += (long long)gridDim.x * blockDim.x){
    const long long t = i / C, c = i - t * C;
    X16[i] = __float2half(X[t * ldx + c]);
  }
}

/* Y[t][r] = (first ? 0 : Y[t][r]) + C16[t][r] (fp32 sum across the pieces), r < R, t < n */
extern "C" __global__ void k_f16_acc(const __half* __restrict__ C16, int R, int n, float* __restrict__ Y, int ldy,
                                     int first) {
  const long long N = (long long)n * R;
  for(long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += (long long)gridDim.x * blockDim.x){
    const long long t = i / R, r = i - t * R;
    const float v = __half2float(C16[i]);
    float* y = Y + t * ldy + r;
    *y = first ? v : *y + v;
  }
}
