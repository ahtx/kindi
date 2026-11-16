/* dsv41_gpu.cu - Kindi DeepSeek-V4.1-Flash engine: the dense (non-expert) matrices on one CUDA GPU
   (Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only).

   Loaded by dsv41_cpu.c with dlopen when KINDI_DSV_GPU is set; the CPU engine keeps the control flow and hands
   every GEMV/GEMM job (gemv_run) to dg_gemv.

   Matrices are kept as in the GGUF, only re-laid out for coalesced reads, in groups of 16 rows (a warp per group,
   two threads per row): Q8_0 codes [group][block][16 rows][32], fp16 scales [group][block][16 rows]; F32
   [group][k / 8][16 rows][8]. One step of a warp reads 512 contiguous bytes. The arithmetic is the CPU engine's, operation for operation, so the results
   are the same bits:
   - Q8_0 x Q8 activations: per block of 32 the CPU makes 8 int32 lanes of 4 products (maddubs + madd); lane j
     accumulates acc_j = fma(float(lane_j), w_scale * x_scale, acc_j) over the blocks; the row result is
     ((a0 + a2) + (a1 + a3)) with a_k = acc_k + acc_(k+4). Here two threads share a row, each holds 4 lanes
     (dp4a gives the exact lane sums), and the same reduction is done with one shuffle.
   - F32: lane j accumulates fma(w[k + j], x[k + j]) for k = 0, 8, 16, ...; the same reduction. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK 32
#define BQ8 34
#define TTN 4                     /* tokens per pass of the GEMV kernels (template TT: 1 for single tokens) */
#define WARPS 4                   /* warps (row groups of 16) per thread block */
#define WARPS8 8                  /* the same for the 8-token verify GEMV (k_q8ws<8>) */
#define MAXMAT 1024

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "dsv41_gpu: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

typedef struct { int type, rows, cols; int8_t* q; __half* s; float* f; } GMat;
static GMat GM[MAXMAT];
static int nGM;
static cudaStream_t st;
static uint8_t *h_in, *h_out;     /* pinned staging */
static uint8_t *d_in, *d_out;
static size_t in_cap, out_cap;
static size_t vram_used;

extern "C" {

typedef struct {
    int mat;                       /* handle from dg_mat_* */
    const int8_t* xq; const float* xd; const float* xf;   /* Q8 input [T][xs] + scales, or F32 input [T][xs] */
    int xs;
    int seg_rows, seg_cols;        /* block-diagonal: row r reads input [(r / seg_rows) * seg_cols, + cols) */
    float* y; int ys;              /* output [T][ys], rows [0, rows) */
} DgOp;

int dg_init(int dev, size_t in_bytes, size_t out_bytes) {
    CK(cudaSetDevice(dev));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    in_cap = in_bytes; out_cap = out_bytes;
    CK(cudaMallocHost(&h_in, in_cap)); CK(cudaMallocHost(&h_out, out_cap));
    CK(cudaMalloc(&d_in, in_cap)); CK(cudaMalloc(&d_out, out_cap));
    vram_used = in_cap + out_cap;
    return 0;
}

size_t dg_vram_used(void) { return vram_used; }

/* a Q8_0 matrix from GGUF rows (34-byte blocks); returns its handle */
int dg_mat_q8(const uint8_t* src, int rows, int cols) {
    if (nGM == MAXMAT) return -1;
    GMat* m = &GM[nGM];
    int nb = cols / QK, rp = (rows + 15) / 16 * 16;
    size_t nq = (size_t)rp * cols, ns = (size_t)rp * nb;
    int8_t* hq = (int8_t*)calloc(nq, 1);
    __half* hs = (__half*)calloc(ns, 2);
    for (size_t r = 0; r < (size_t)rows; r++)
        for (int b = 0; b < nb; b++) {
            const uint8_t* B = src + (r * nb + b) * BQ8;
            size_t gi = ((r / 16) * nb + b) * 16 + r % 16;
            memcpy(hs + gi, B, 2);
            memcpy(hq + gi * QK, B + 2, QK);
        }
    CK(cudaMalloc(&m->q, nq)); CK(cudaMalloc(&m->s, ns * 2));
    CK(cudaMemcpy(m->q, hq, nq, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(m->s, hs, ns * 2, cudaMemcpyHostToDevice));
    free(hq); free(hs);
    m->type = 8; m->rows = rows; m->cols = cols; m->f = NULL;
    vram_used += nq + ns * 2;
    return nGM++;
}

int dg_mat_f32(const float* src, int rows, int cols) {
    if (nGM == MAXMAT) return -1;
    GMat* m = &GM[nGM];
    int rp = (rows + 15) / 16 * 16;
    size_t n = (size_t)rp * cols;
    float* hf = (float*)calloc(n, 4);
    for (size_t r = 0; r < (size_t)rows; r++)
        for (int k = 0; k < cols; k += 8)
            memcpy(hf + (((r / 16) * (cols / 8) + k / 8) * 16 + r % 16) * 8, src + r * cols + k, 32);
    CK(cudaMalloc(&m->f, n * 4));
    CK(cudaMemcpy(m->f, hf, n * 4, cudaMemcpyHostToDevice));
    free(hf);
    m->type = 0; m->rows = rows; m->cols = cols; m->q = NULL; m->s = NULL;
    vram_used += n * 4;
    return nGM++;
}

}  /* extern "C" */

/* the CPU's hsum8 order from 4 lanes per thread, the other 4 lanes in the partner thread (lane ^ 1) */
__device__ __forceinline__ float red8(float c0, float c1, float c2, float c3, int half) {
    float o0 = __shfl_xor_sync(0xffffffffu, c0, 1), o1 = __shfl_xor_sync(0xffffffffu, c1, 1);
    float o2 = __shfl_xor_sync(0xffffffffu, c2, 1), o3 = __shfl_xor_sync(0xffffffffu, c3, 1);
    /* a_k = lane_k + lane_(k+4): half 0 holds lanes 0..3 */
    float a0 = half ? o0 + c0 : c0 + o0, a1 = half ? o1 + c1 : c1 + o1;
    float a2 = half ? o2 + c2 : c2 + o2, a3 = half ? o3 + c3 : c3 + o3;
    return (a0 + a2) + (a1 + a3);
}

/* the CPU's hsum8 order over 8 lanes held by 8 consecutive threads (lane l = thread & 7); lane 0 gets the result */
__device__ __forceinline__ float red8l(float v) {
    v = v + __shfl_down_sync(0xffffffffu, v, 4, 8);     /* a_k = lane_k + lane_(k+4) */
    v = v + __shfl_down_sync(0xffffffffu, v, 2, 8);     /* (a0 + a2), (a1 + a3) */
    return v + __shfl_down_sync(0xffffffffu, v, 1, 8);
}

#define UB 8                      /* blocks loaded ahead per step (memory-level parallelism) */
#define UW 4                      /* the same for the wide kernel (16-byte loads) */

/* Q8_0 rows x Q8 activations, T tokens (TT per pass). A thread block per group of 16 rows, 8 threads per row, one
   int32 lane each (4 codes of every block); a warp reads 4 rows x 32 bytes = 128 contiguous bytes per block. */
template <int TT> __global__ void k_q8(const int8_t* __restrict__ W, const __half* __restrict__ S, int rows, int cols,
                     const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols,
                     int T, float* __restrict__ Y, int ys) {
    /* the activations of the token tile in shared memory: the per-block loads are then short, and only the weight
       loads go to memory (dynamic shared memory: TT * cols bytes + TT * cols / 32 floats) */
    extern __shared__ int4 smem_x[];
    int8_t* xs_ = (int8_t*)smem_x;
    float* xds = (float*)(xs_ + TT * cols);
    const int g = blockIdx.x, rr = threadIdx.x >> 3, l = threadIdx.x & 7;
    const int row = g * 16 + rr;
    const bool live = row < rows;
    const int nb = cols / QK;
    const int xoff = seg_rows ? ((g * 16 < rows ? g * 16 : rows - 1) / seg_rows) * seg_cols : 0;
    const int8_t* w = W + ((size_t)g * nb * 16 + rr) * QK + l * 4;
    const __half* s = S + (size_t)g * nb * 16 + rr;
    for (int t0 = 0; t0 < T; t0 += TT) {
        const int nt = T - t0 < TT ? T - t0 : TT;
        __syncthreads();
        for (int i = threadIdx.x; i < nt * cols / 4; i += blockDim.x) {
            int t = i / (cols / 4), j = i % (cols / 4);
            ((int*)xs_)[t * (cols / 4) + j] = *(const int*)(X + (size_t)(t0 + t) * xs + xoff + j * 4);
        }
        for (int i = threadIdx.x; i < nt * nb; i += blockDim.x) {
            int t = i / nb, j = i % nb;
            xds[t * nb + j] = XD[(size_t)(t0 + t) * (xs / QK) + xoff / QK + j];
        }
        __syncthreads();
        float acc[TT];
#pragma unroll
        for (int t = 0; t < TT; t++) acc[t] = 0.f;
        for (int b0 = 0; b0 < nb; b0 += UB) {
            int wv[UB]; float ws[UB];
#pragma unroll
            for (int u = 0; u < UB; u++)
                if (b0 + u < nb) { wv[u] = *(const int*)(w + (size_t)(b0 + u) * 16 * QK); ws[u] = __half2float(s[(size_t)(b0 + u) * 16]); }
#pragma unroll
            for (int u = 0; u < UB; u++) {
                if (b0 + u >= nb) break;
                const int b = b0 + u;
#pragma unroll
                for (int t = 0; t < TT; t++) {
                    if (t < nt) {
                        const int xv = ((const int*)xs_)[t * (cols / 4) + b * 8 + l];
                        const float sa = ws[u] * xds[t * nb + b];
                        acc[t] = __fmaf_rn((float)__dp4a(wv[u], xv, 0), sa, acc[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; t++) {
            float v = red8l(acc[t]);
            if (t < nt && live && l == 0) Y[(size_t)(t0 + t) * ys + row] = v;
        }
    }
}

/* (large matrices) Q8_0 rows x Q8 activations, T tokens (TT per pass). A warp per group of 16 rows, a thread pair per row; the
   next block's codes are loaded while the current one is used. */
template <int TT> __global__ void k_q8w(const int8_t* __restrict__ W, const __half* __restrict__ S, int rows, int cols,
                     const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols,
                     int T, float* __restrict__ Y, int ys) {
    const int lane = threadIdx.x & 31, half = lane & 1, rr = lane >> 1;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int row = g * 16 + rr;
    if (g * 16 >= rows) return;
    const bool live = row < rows;
    const int nb = cols / QK;
    const int xoff = seg_rows ? ((live ? row : rows - 1) / seg_rows) * seg_cols : 0;
    const int8_t* w = W + ((size_t)g * nb * 16 + rr) * QK + half * 16;
    const __half* s = S + (size_t)g * nb * 16 + rr;
    for (int t0 = 0; t0 < T; t0 += TT) {
        const int nt = T - t0 < TT ? T - t0 : TT;
        float acc[TT][4];
#pragma unroll
        for (int t = 0; t < TT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        int4 wn = *(const int4*)w;
        __half sn = s[0];
        for (int b = 0; b < nb; b++) {
            const int4 wv = wn;
            const float ws = __half2float(sn);
            if (b + 1 < nb) { wn = *(const int4*)(w + (size_t)(b + 1) * 16 * QK); sn = s[(size_t)(b + 1) * 16]; }
#pragma unroll
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    const int8_t* x = X + (size_t)(t0 + t) * xs + xoff + (size_t)b * QK + half * 16;
                    const int4 xv = *(const int4*)x;
                    const float sa = ws * XD[(size_t)(t0 + t) * (xs / QK) + xoff / QK + b];
                    acc[t][0] = __fmaf_rn((float)__dp4a(wv.x, xv.x, 0), sa, acc[t][0]);
                    acc[t][1] = __fmaf_rn((float)__dp4a(wv.y, xv.y, 0), sa, acc[t][1]);
                    acc[t][2] = __fmaf_rn((float)__dp4a(wv.z, xv.z, 0), sa, acc[t][2]);
                    acc[t][3] = __fmaf_rn((float)__dp4a(wv.w, xv.w, 0), sa, acc[t][3]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; t++) {
            float v = red8(acc[t][0], acc[t][1], acc[t][2], acc[t][3], half);
            if (t < nt && live && !half) Y[(size_t)(t0 + t) * ys + row] = v;
        }
    }
}

/* k_q8w for 2-8 tokens (verify batches): the tokens' activations and scales go to shared memory first, so the loop
   waits only for the weight loads (with them in global memory, the per-token loads stalled the warps: 77 % of the
   cycles on the long scoreboard at T = 4). Same thread layout and operation sequence as k_q8w (the same bits).
   Thread block = WARPS row groups of 16 rows; seg_rows (if any) a multiple of WARPS * 16. */
template <int TT> __global__ void k_q8ws(const int8_t* __restrict__ W, const __half* __restrict__ S, int rows, int cols,
                     const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols,
                     int T, float* __restrict__ Y, int ys) {
    extern __shared__ int4 smem_w[];
    int8_t* xs_ = (int8_t*)smem_w;
    float* xds = (float*)(xs_ + TT * cols);
    const int lane = threadIdx.x & 31, half = lane & 1, rr = lane >> 1;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int row = g * 16 + rr;
    const bool live = row < rows && g * 16 < rows;
    const int nb = cols / QK;
    const int r0 = blockIdx.x * (blockDim.x >> 5) * 16;
    const int xoff = seg_rows ? ((r0 < rows ? r0 : rows - 1) / seg_rows) * seg_cols : 0;
    const int8_t* w = W + ((size_t)g * nb * 16 + rr) * QK + half * 16;
    const __half* s = S + (size_t)g * nb * 16 + rr;
    for (int t0 = 0; t0 < T; t0 += TT) {
        const int nt = T - t0 < TT ? T - t0 : TT;
        __syncthreads();
        for (int i = threadIdx.x; i < nt * cols / 16; i += blockDim.x) {
            int t = i / (cols / 16), j = i % (cols / 16);
            ((int4*)xs_)[t * (cols / 16) + j] = *(const int4*)(X + (size_t)(t0 + t) * xs + xoff + j * 16);
        }
        for (int i = threadIdx.x; i < nt * nb; i += blockDim.x) {
            int t = i / nb, j = i % nb;
            xds[t * nb + j] = XD[(size_t)(t0 + t) * (xs / QK) + xoff / QK + j];
        }
        __syncthreads();
        if (g * 16 >= rows) continue;
        float acc[TT][4];
#pragma unroll
        for (int t = 0; t < TT; t++) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
        int4 wn = *(const int4*)w;
        __half sn = s[0];
        for (int b = 0; b < nb; b++) {
            const int4 wv = wn;
            const float ws = __half2float(sn);
            if (b + 1 < nb) { wn = *(const int4*)(w + (size_t)(b + 1) * 16 * QK); sn = s[(size_t)(b + 1) * 16]; }
#pragma unroll
            for (int t = 0; t < TT; t++) {
                if (t < nt) {
                    const int4 xv = ((const int4*)xs_)[(t * cols + b * QK + half * 16) / 16];
                    const float sa = ws * xds[t * nb + b];
                    acc[t][0] = __fmaf_rn((float)__dp4a(wv.x, xv.x, 0), sa, acc[t][0]);
                    acc[t][1] = __fmaf_rn((float)__dp4a(wv.y, xv.y, 0), sa, acc[t][1]);
                    acc[t][2] = __fmaf_rn((float)__dp4a(wv.z, xv.z, 0), sa, acc[t][2]);
                    acc[t][3] = __fmaf_rn((float)__dp4a(wv.w, xv.w, 0), sa, acc[t][3]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; t++) {
            float v = red8(acc[t][0], acc[t][1], acc[t][2], acc[t][3], half);
            if (t < nt && live && !half) Y[(size_t)(t0 + t) * ys + row] = v;
        }
    }
}

/* F32 rows x F32 activations, the same thread layout (lane l sums w[k + l] * x[k + l], k = 0, 8, ...). The weights of
   the row group come through shared memory in chunks of 256 columns (16 KB, contiguous in this layout), so each
   thread has 32 loads in flight instead of one dependent chain of loads. */
#define F32C 256
template <int TT> __global__ void __launch_bounds__(128) k_f32(const float* __restrict__ W, int rows, int cols, const float* __restrict__ X,
        int xs, int T, float* __restrict__ Y, int ys) {
    __shared__ float4 ws[F32C * 16 / 4];
    __shared__ float xsh[TT][F32C];               /* the tokens' activations of the chunk (not reloaded per row) */
    const int g = blockIdx.x, rr = threadIdx.x >> 3, l = threadIdx.x & 7;
    const int row = g * 16 + rr;
    const bool live = row < rows;
    const int nk = cols / 8;
    const float* wg = W + (size_t)g * nk * 16 * 8;
    for (int t0 = 0; t0 < T; t0 += TT) {
        const int nt = T - t0 < TT ? T - t0 : TT;
        float acc[TT];
#pragma unroll
        for (int t = 0; t < TT; t++) acc[t] = 0.f;
        float4 pre[F32C * 16 / 4 / 128];          /* the next chunk, loaded into registers while this one is used */
        const int nper = F32C * 16 / 4 / 128;
        {
            const float4* src = (const float4*)wg;
            const int kc = cols < F32C ? cols : F32C;
#pragma unroll
            for (int j = 0; j < nper; j++) { int i = threadIdx.x + j * 128; if (i < kc * 16 / 4) pre[j] = src[i]; }
        }
        for (int k0 = 0; k0 < cols; k0 += F32C) {
            const int kc = cols - k0 < F32C ? cols - k0 : F32C;
#pragma unroll
            for (int j = 0; j < nper; j++) { int i = threadIdx.x + j * 128; if (i < kc * 16 / 4) ws[i] = pre[j]; }
            for (int i = threadIdx.x; i < nt * kc; i += blockDim.x) xsh[i / kc][i % kc] = X[(size_t)(t0 + i / kc) * xs + k0 + i % kc];
            __syncthreads();
            if (k0 + F32C < cols) {
                const float4* src = (const float4*)(wg + (size_t)((k0 + F32C) / 8) * 16 * 8);
                const int kn = cols - k0 - F32C < F32C ? cols - k0 - F32C : F32C;
#pragma unroll
                for (int j = 0; j < nper; j++) { int i = threadIdx.x + j * 128; if (i < kn * 16 / 4) pre[j] = src[i]; }
            }
            const float* wsf = (const float*)ws;
#pragma unroll 8
            for (int kk = 0; kk < kc / 8; kk++) {
                const float wv = wsf[(kk * 16 + rr) * 8 + l];
#pragma unroll
                for (int t = 0; t < TT; t++)
                    if (t < nt) acc[t] = __fmaf_rn(wv, xsh[t][kk * 8 + l], acc[t]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int t = 0; t < TT; t++) {
            float v = red8l(acc[t]);
            if (t < nt && live && l == 0) Y[(size_t)(t0 + t) * ys + row] = v;
        }
    }
}

/* Several small Q8_0 matrices with the same input in one launch (more thread blocks in flight than one small matrix
   gives); the per-row arithmetic is k_q8's. */
#define MAXMULTI 6
typedef struct {
    int n, first[MAXMULTI + 1];                  /* first thread block of each matrix */
    const int8_t* W[MAXMULTI]; const __half* S[MAXMULTI]; int rows[MAXMULTI]; float* Y[MAXMULTI]; int ys[MAXMULTI];
} QMulti;
template <int TT> __global__ void __launch_bounds__(128) k_q8_multi(QMulti M, int cols, const int8_t* __restrict__ X,
                                                  const float* __restrict__ XD, int xs, int T) {
    extern __shared__ int4 smem_x[];
    int8_t* xs_ = (int8_t*)smem_x;
    float* xds = (float*)(xs_ + TT * cols);
    int mi = 0;
    while ((int)blockIdx.x >= M.first[mi + 1]) mi++;
    const int g = blockIdx.x - M.first[mi], rr = threadIdx.x >> 3, l = threadIdx.x & 7;
    const int rows = M.rows[mi], row = g * 16 + rr;
    const bool live = row < rows;
    const int nb = cols / QK;
    const int8_t* w = M.W[mi] + ((size_t)g * nb * 16 + rr) * QK + l * 4;
    const __half* s = M.S[mi] + (size_t)g * nb * 16 + rr;
    for (int t0 = 0; t0 < T; t0 += TT) {
        const int nt = T - t0 < TT ? T - t0 : TT;
        __syncthreads();
        for (int i = threadIdx.x; i < nt * cols / 4; i += blockDim.x) {
            int t = i / (cols / 4), j = i % (cols / 4);
            ((int*)xs_)[t * (cols / 4) + j] = *(const int*)(X + (size_t)(t0 + t) * xs + j * 4);
        }
        for (int i = threadIdx.x; i < nt * nb; i += blockDim.x) {
            int t = i / nb, j = i % nb;
            xds[t * nb + j] = XD[(size_t)(t0 + t) * (xs / QK) + j];
        }
        __syncthreads();
        float acc[TT];
#pragma unroll
        for (int t = 0; t < TT; t++) acc[t] = 0.f;
        for (int b0 = 0; b0 < nb; b0 += (TT == 1 ? 32 : UB)) {
            constexpr int UBK = TT == 1 ? 32 : UB;
            int wv[UBK]; float ws[UBK];
#pragma unroll
            for (int u = 0; u < UBK; u++)
                if (b0 + u < nb) { wv[u] = *(const int*)(w + (size_t)(b0 + u) * 16 * QK); ws[u] = __half2float(s[(size_t)(b0 + u) * 16]); }
#pragma unroll
            for (int u = 0; u < UBK; u++) {
                if (b0 + u >= nb) break;
                const int b = b0 + u;
#pragma unroll
                for (int t = 0; t < TT; t++) {
                    if (t < nt) {
                        const int xv = ((const int*)xs_)[t * (cols / 4) + b * 8 + l];
                        const float sa = ws[u] * xds[t * nb + b];
                        acc[t] = __fmaf_rn((float)__dp4a(wv[u], xv, 0), sa, acc[t]);
                    }
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; t++) {
            float v = red8l(acc[t]);
            if (t < nt && live && l == 0) M.Y[mi][(size_t)(t0 + t) * M.ys[mi] + row] = v;
        }
    }
}

/* F32 matrix x F32 activations for prompt chunks (the router): 64 rows x 64 tokens per thread block, 32 columns per
   step through shared memory, 4 x 4 outputs per thread; sum order: k in order per output (not the lane order of
   k_f32, like the int8 GEMM for chunks). The weights are in the 16-row layout of k_f32: [group][k / 8][16][8]. */
__global__ void __launch_bounds__(256) k_f32_gemm(const float* __restrict__ W, int rows, int cols, const float* __restrict__ X,
        int xs, int T, float* __restrict__ Y, int ys) {
    __shared__ float Ws[32][65], Xs[32][65];
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15, r0 = blockIdx.x * 64, t0 = blockIdx.y * 64, nk = cols / 8;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < cols; k0 += 32) {
        for (int i = tid; i < 64 * 32; i += 256) {
            const int rr = i / 32, kk = i % 32, row = r0 + rr, k = k0 + kk;
            Ws[kk][rr] = row < rows ? W[(((size_t)(row >> 4) * nk + k / 8) * 16 + (row & 15)) * 8 + (k & 7)] : 0.f;
            const int tk = t0 + rr;
            Xs[kk][rr] = tk < T ? X[(size_t)tk * xs + k] : 0.f;
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < 32; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = Ws[kk][ty * 4 + i]; b[i] = Xs[kk][tx * 4 + i]; }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int row = r0 + ty * 4 + i, tk = t0 + tx * 4 + j;
            if (row < rows && tk < T) Y[(size_t)tk * ys + row] = acc[i][j];
        }
}

/* Prompt chunks (T > GEMM_MIN_T): a tiled GEMM, 64 rows x 64 tokens per thread block, one Q8 block (32 values) per
   step through shared memory; each thread makes 4 rows x 4 tokens. Per output and block: int32 sum of the 32
   products (dp4a, exact), then acc = fma(float(sum), w_scale * x_scale, acc). This is not the per-lane order of the
   CPU kernels, so a prompt chunk can differ from single steps in the last bits (chunks up to GEMM_MIN_T tokens,
   which include speculation's verify steps, keep the exact GEMV path). */
#define GEMM_MIN_T 8
__global__ void __launch_bounds__(256) k_q8_gemm(const int8_t* __restrict__ W, const __half* __restrict__ S, int rows,
        int cols, const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols, int T,
        float* __restrict__ Y, int ys) {
    __shared__ int Ws[64][9], Xs[64][9];
    __shared__ float wsc[64], xsc[64];
    const int tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int r0 = blockIdx.x * 64, t0 = blockIdx.y * 64, nb = cols / QK;
    const int xoff = seg_rows ? (r0 / seg_rows) * seg_cols : 0;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int b = 0; b < nb; b++) {
        /* 64 rows x 32 bytes: row group g = r0 / 16 + i, the 16 rows of a group are 512 contiguous bytes */
        for (int i = tid; i < 64 * 8; i += 256) {
            int rr = i >> 3, j = i & 7, row = r0 + rr;
            Ws[rr][j] = row < rows ? *(const int*)(W + (((size_t)(row >> 4) * nb + b) * 16 + (row & 15)) * QK + j * 4) : 0;
            int tk = t0 + rr;
            Xs[rr][j] = tk < T ? *(const int*)(X + (size_t)tk * xs + xoff + (size_t)b * QK + j * 4) : 0;
        }
        if (tid < 64) {
            int row = r0 + tid;
            wsc[tid] = row < rows ? __half2float(S[((size_t)(row >> 4) * nb + b) * 16 + (row & 15)]) : 0.f;
        } else if (tid < 128) {
            int tk = t0 + tid - 64;
            xsc[tid - 64] = tk < T ? XD[(size_t)tk * (xs / QK) + xoff / QK + b] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int rr = ty * 4 + i;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int tt = tx * 4 + j;
                int sum = 0;
#pragma unroll
                for (int q = 0; q < 8; q++) sum = __dp4a(Ws[rr][q], Xs[tt][q], sum);
                acc[i][j] = __fmaf_rn((float)sum, wsc[rr] * xsc[tt], acc[i][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = r0 + ty * 4 + i, tk = t0 + tx * 4 + j;
            if (row < rows && tk < T) Y[(size_t)tk * ys + row] = acc[i][j];
        }
}

/* Prompt chunks on the int8 tensor cores: mma.sync m16n8k32 is one Q8 block (32 values) per step, so the block's int32
   sum comes out of the MMA exactly and the arithmetic is k_q8_gemm's: acc = fma(float(sum), w_scale * x_scale, acc),
   blocks in order (the same bits). A warp makes 16 rows (one row group: the A fragment is read straight from the
   16-row layout) x 64 tokens (8 MMAs share the A fragment); a thread block of 4 warps makes 64 rows x 64 tokens. */
#define MMA_TOK 64
__global__ void __launch_bounds__(128) k_q8_mma(const int8_t* __restrict__ W, const __half* __restrict__ S, int rows,
        int cols, const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols, int T,
        float* __restrict__ Y, int ys) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int g = blockIdx.x * 4 + warp;                  /* row group */
    if (g * 16 >= rows) return;
    const int t0 = blockIdx.y * MMA_TOK, nb = cols / QK;
    const int xoff = seg_rows ? ((g * 16) / seg_rows) * seg_cols : 0;
    const int8_t* wg = W + (size_t)g * nb * 16 * QK;
    const __half* sg = S + (size_t)g * nb * 16;
    float acc[MMA_TOK / 8][4];
#pragma unroll
    for (int n = 0; n < MMA_TOK / 8; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    for (int b = 0; b < nb; b++) {
        const int8_t* wb = wg + (size_t)b * 16 * QK;
        const int a0 = *(const int*)(wb + gid * QK + tig * 4), a1 = *(const int*)(wb + (gid + 8) * QK + tig * 4);
        const int a2 = *(const int*)(wb + gid * QK + 16 + tig * 4), a3 = *(const int*)(wb + (gid + 8) * QK + 16 + tig * 4);
        const float w0 = __half2float(sg[(size_t)b * 16 + gid]), w1 = __half2float(sg[(size_t)b * 16 + gid + 8]);
#pragma unroll
        for (int n = 0; n < MMA_TOK / 8; n++) {
            const int tk = t0 + n * 8 + gid;              /* B fragment: column (token) = gid */
            int b0 = 0, b1 = 0;
            if (tk < T) {
                const int8_t* xr = X + (size_t)tk * xs + xoff + (size_t)b * QK;
                b0 = *(const int*)(xr + tig * 4); b1 = *(const int*)(xr + 16 + tig * 4);
            }
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                         "{%0,%1,%2,%3};\n"
                         : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            /* C fragment: c0, c1 at row gid, tokens tig * 2 + {0, 1}; c2, c3 at row gid + 8 */
            const int ta = t0 + n * 8 + tig * 2, tb = ta + 1;
            const float xa = ta < T ? XD[(size_t)ta * (xs / QK) + xoff / QK + b] : 0.f;
            const float xb = tb < T ? XD[(size_t)tb * (xs / QK) + xoff / QK + b] : 0.f;
            acc[n][0] = __fmaf_rn((float)c0, w0 * xa, acc[n][0]);
            acc[n][1] = __fmaf_rn((float)c1, w0 * xb, acc[n][1]);
            acc[n][2] = __fmaf_rn((float)c2, w1 * xa, acc[n][2]);
            acc[n][3] = __fmaf_rn((float)c3, w1 * xb, acc[n][3]);
        }
    }
#pragma unroll
    for (int n = 0; n < MMA_TOK / 8; n++) {
        const int ta = t0 + n * 8 + tig * 2, r0 = g * 16 + gid, r1 = r0 + 8;
        if (ta < T) { if (r0 < rows) Y[(size_t)ta * ys + r0] = acc[n][0]; if (r1 < rows) Y[(size_t)ta * ys + r1] = acc[n][2]; }
        if (ta + 1 < T) { if (r0 < rows) Y[(size_t)(ta + 1) * ys + r0] = acc[n][1]; if (r1 < rows) Y[(size_t)(ta + 1) * ys + r1] = acc[n][3]; }
    }
}

/* ---- GEMV kernels with more loads / threads in flight (bit-identical to k_q8 / k_q8w / k_q8ws / k_f32) ----
   k_q8p (single tokens): every weight byte of a thread block requested at once (cp.async into a ring of shared-memory
   stages), RPC rows of one 16-row group per thread block, thread (rr, l) = lane l of row rr (k_q8's mapping).
   k_q8t (2..8 tokens, short matrices) and k_f32t (2..8 tokens, the router): a thread per (row, lane, token), so each
   fma chain runs on its own thread. Every lane chain sums its blocks in order with the same fmaf sequence and the
   same 8-lane reduction as the kernels they replace. KINDI_DSV_Q8P / KINDI_DSV_Q8T / KINDI_DSV_F32T = 0: off. */
#define Q8P_KC 16                 /* blocks (of 32 columns) per stage */

__device__ __forceinline__ void cp16(void* s, const void* g) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp8(void* s, const void* g) {
    unsigned a = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(a), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int RPC, int S>
__global__ void __launch_bounds__(RPC * 8) k_q8p(const int8_t* __restrict__ W, const __half* __restrict__ Sc, int rows,
        int cols, const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols,
        float* __restrict__ Y) {
    extern __shared__ int4 smem_p[];
    const int nb = cols / QK;
    const int nthr = RPC * 8;
    int8_t* xs_ = (int8_t*)smem_p;                         /* [cols] */
    float* xds = (float*)(xs_ + cols);                     /* [nb] */
    int8_t* wq = (int8_t*)(xds + nb);                      /* [S][KC][RPC][32] */
    __half* wsc = (__half*)(wq + S * Q8P_KC * RPC * QK);   /* [S][KC][RPC] */
    const int row0 = blockIdx.x * RPC, g = row0 >> 4, sub = row0 & 15;
    const int tid = threadIdx.x, rr = tid >> 3, l = tid & 7;
    const int row = row0 + rr;
    const int xoff = seg_rows ? ((row0 < rows ? row0 : rows - 1) / seg_rows) * seg_cols : 0;
    const int8_t* wg = W + ((size_t)g * nb * 16 + sub) * QK;
    const __half* sg = Sc + (size_t)g * nb * 16 + sub;
    const int nch = (nb + Q8P_KC - 1) / Q8P_KC;
    auto issue = [&](int c) {
        if (c < nch) {
            const int b0 = c * Q8P_KC, nbk = nb - b0 < Q8P_KC ? nb - b0 : Q8P_KC;
            int8_t* dq = wq + (c % S) * Q8P_KC * RPC * QK;
            __half* ds = wsc + (c % S) * Q8P_KC * RPC;
            for (int i = tid; i < nbk * RPC * 2; i += nthr) {
                const int b = i / (RPC * 2), k = i % (RPC * 2);
                cp16(dq + b * RPC * QK + k * 16, wg + (size_t)(b0 + b) * 16 * QK + k * 16);
            }
            if (RPC == 4) {
                for (int b = tid; b < nbk; b += nthr) cp8(ds + b * 4, sg + (size_t)(b0 + b) * 16);
            } else {
                for (int i = tid; i < nbk * (RPC / 8); i += nthr) {
                    const int b = i / (RPC / 8), k = i % (RPC / 8);
                    cp16(ds + b * RPC + k * 8, sg + (size_t)(b0 + b) * 16 + k * 8);
                }
            }
        }
        cp_commit();
    };
#pragma unroll
    for (int c = 0; c < S - 1; c++) issue(c);
    for (int i = tid; i < cols / 16; i += nthr) ((int4*)xs_)[i] = *(const int4*)(X + xoff + i * 16);
    for (int i = tid; i < nb; i += nthr) xds[i] = XD[xoff / QK + i];
    float acc = 0.f;
    for (int c = 0; c < nch; c++) {
        issue(c + S - 1);
        cp_wait<S - 1>();
        __syncthreads();
        const int b0 = c * Q8P_KC, nbk = nb - b0 < Q8P_KC ? nb - b0 : Q8P_KC;
        const int8_t* dq = wq + (c % S) * Q8P_KC * RPC * QK + rr * QK + l * 4;
        const __half* ds = wsc + (c % S) * Q8P_KC * RPC + rr;
        for (int b = 0; b < nbk; b++) {
            const int wv = *(const int*)(dq + b * RPC * QK);
            const float ws = __half2float(ds[b * RPC]);
            const int xv = ((const int*)xs_)[(b0 + b) * 8 + l];
            const float sa = ws * xds[b0 + b];
            acc = __fmaf_rn((float)__dp4a(wv, xv, 0), sa, acc);
        }
        __syncthreads();                               /* the stage is free for chunk c + S */
    }
    cp_wait<0>();
    float v = red8l(acc);
    if (row < rows && l == 0) Y[row] = v;
}

/* k_q8t (T = 2..8, short matrices): a thread per (row, lane, token), so each fma chain runs on its own thread
   (k_q8's lane chains: block order, the same fmaf sequence, red8l): T times the threads of k_q8. Thread block =
   RPC rows x 8 lanes (x) by T tokens (y); the weights come through L1 (the T warps of a row group read the same
   bytes), the activations straight from global memory. */
template <int RPC>
__global__ void k_q8t(const int8_t* __restrict__ W, const __half* __restrict__ Sc, int rows, int cols,
                      const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, int seg_rows, int seg_cols,
                      float* __restrict__ Y, int ys) {
    const int nb = cols / QK;
    const int row0 = blockIdx.x * RPC, g = row0 >> 4, sub = row0 & 15;
    const int rr = threadIdx.x >> 3, l = threadIdx.x & 7, t = threadIdx.y;
    const int row = row0 + rr;
    const int xoff = seg_rows ? ((row0 < rows ? row0 : rows - 1) / seg_rows) * seg_cols : 0;
    const int8_t* w = W + ((size_t)g * nb * 16 + sub + rr) * QK + l * 4;
    const __half* s = Sc + (size_t)g * nb * 16 + sub + rr;
    const int8_t* x = X + (size_t)t * xs + xoff + l * 4;
    const float* xd = XD + (size_t)t * (xs / QK) + xoff / QK;
    float acc = 0.f;
#pragma unroll 8
    for (int b = 0; b < nb; b++) {
        const int wv = __ldg((const int*)(w + (size_t)b * 16 * QK));
        const float ws = __half2float(s[(size_t)b * 16]);
        const int xv = __ldg((const int*)(x + (size_t)b * QK));
        const float sa = ws * __ldg(xd + b);
        acc = __fmaf_rn((float)__dp4a(wv, xv, 0), sa, acc);
    }
    float v = red8l(acc);
    if (row < rows && l == 0) Y[(size_t)t * ys + row] = v;
}

/* k_f32's lane chains (lane l sums w[k + l] * x[k + l], k = 0, 8, ... in order, red8l) with a thread per (row, lane,
   token): thread block = 4 rows x 8 lanes (x) by T tokens (y) */
__global__ void k_f32t(const float* __restrict__ W, int rows, int cols, const float* __restrict__ X, int xs,
                       float* __restrict__ Y, int ys) {
    const int row0 = blockIdx.x * 4, g = row0 >> 4, sub = row0 & 15;
    const int rr = threadIdx.x >> 3, l = threadIdx.x & 7, t = threadIdx.y;
    const int row = row0 + rr;
    const int nk = cols / 8;
    const float* w = W + (size_t)g * nk * 16 * 8 + (sub + rr) * 8 + l;
    const float* x = X + (size_t)t * xs + l;
    float acc = 0.f;
#pragma unroll 8
    for (int kk = 0; kk < nk; kk++) acc = __fmaf_rn(__ldg(w + (size_t)kk * 16 * 8), __ldg(x + kk * 8), acc);
    float v = red8l(acc);
    if (row < rows && l == 0) Y[(size_t)t * ys + row] = v;
}
static int q8_old;                /* tests: the kernels before k_q8p / k_q8t (gemv_bench) */
static int env_on(const char* name) { const char* v = getenv(name); return !v || atoi(v) != 0; }
template <int RPC>
static void q8p_go(const GMat* m, const int8_t* xq, const float* xd, int xs, int seg_rows, int seg_cols, float* y) {
    const int nb = m->cols / QK;
    const size_t sm = (size_t)m->cols + (size_t)nb * 4 + (size_t)4 * Q8P_KC * RPC * (QK + 2);   /* < 48 KB here */
    k_q8p<RPC, 4><<<(m->rows + RPC - 1) / RPC, RPC * 8, sm, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs, seg_rows,
                                                                   seg_cols, y);
}

/* matrices of 4096 rows and more: two threads per row (16-byte loads); smaller: eight (more warps in flight);
   prompt chunks: the tensor-core GEMM (KINDI_DSV_GEMM=2: the dp4a tiled GEMM) */
static void q8_launch(const GMat* m, const int8_t* xq, const float* xd, int xs, int seg_rows, int seg_cols, int T,
                      float* y, int ys) {
    static int q8p = -1, q8t = -1;
    if (q8p < 0) { q8p = env_on("KINDI_DSV_Q8P"); q8t = env_on("KINDI_DSV_Q8T"); }
    const int use_p = q8p && !q8_old, use_t = q8t && !q8_old;
    /* single tokens: k_q8p where it is faster (short matrices, long rows, the output head) */
    if (T == 1 && use_p && (m->rows < 1024 || m->cols >= 8192 || (m->rows >= 16384 && m->cols >= 5120))
        && m->cols + m->cols / QK * 4 + 4 * Q8P_KC * 16 * (QK + 2) <= 48 * 1024) {
        const int rpc = m->rows >= 16384 ? 16 : m->rows >= 4096 ? 8 : 4;
        if (!seg_rows || seg_rows % rpc == 0) {
            if (rpc == 4) q8p_go<4>(m, xq, xd, xs, seg_rows, seg_cols, y);
            else if (rpc == 8) q8p_go<8>(m, xq, xd, xs, seg_rows, seg_cols, y);
            else q8p_go<16>(m, xq, xd, xs, seg_rows, seg_cols, y);
            return;
        }
    }
    /* verify batches, matrices of fewer than 2048 rows: k_q8t */
    if (T >= 2 && T <= 8 && use_t && m->rows < 2048 && (!seg_rows || seg_rows % 4 == 0)) {
        k_q8t<4><<<(m->rows + 3) / 4, dim3(32, T), 0, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs, seg_rows, seg_cols,
                                                              y, ys);
        return;
    }
    static int gemm = -1;
    if (gemm < 0) gemm = getenv("KINDI_DSV_GEMM") ? atoi(getenv("KINDI_DSV_GEMM")) : 1;
    if (gemm == 1 && T > GEMM_MIN_T)
        k_q8_mma<<<dim3(((m->rows + 15) / 16 + 3) / 4, (T + MMA_TOK - 1) / MMA_TOK), 128, 0, st>>>(m->q, m->s, m->rows,
            m->cols, xq, xd, xs, seg_rows, seg_cols, T, y, ys);
    else if (gemm && T > GEMM_MIN_T)
        k_q8_gemm<<<dim3((m->rows + 63) / 64, (T + 63) / 64), 256, 0, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs,
                                                                         seg_rows, seg_cols, T, y, ys);
    static int wsmem = -1;
    if (wsmem < 0) wsmem = getenv("KINDI_DSV_WSMEM") ? atoi(getenv("KINDI_DSV_WSMEM")) : 1;
    static int wide_min = -1;
    if (wide_min < 0) wide_min = getenv("KINDI_DSV_WIDE_MIN") ? atoi(getenv("KINDI_DSV_WIDE_MIN")) : 4096;
    if (gemm && T > GEMM_MIN_T) {}
    else if ((m->rows >= wide_min || T > TTN || (m->rows >= 2048 && T >= 2)) && T > 1 && wsmem && (T <= TTN ? TTN : 8) * (m->cols + m->cols / QK * 4) <= 99 * 1024
             && (seg_rows == 0 || seg_rows % (WARPS8 * 16) == 0)) {
        /* verify batches: one pass over the weights for up to 8 tokens */
        static int attr = 0;
        cudaStreamCaptureStatus cs_;
        if (!attr && cudaStreamIsCapturing(st, &cs_) == cudaSuccess && cs_ == cudaStreamCaptureStatusNone) {
            cudaFuncSetAttribute(k_q8ws<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            cudaFuncSetAttribute(k_q8ws<TTN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            attr = 1;
        }
        const int grid = ((m->rows + 15) / 16 + WARPS - 1) / WARPS, sm4 = TTN * (m->cols + m->cols / QK * 4);
        if (T <= TTN)
            k_q8ws<TTN><<<grid, WARPS * 32, sm4, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs, seg_rows, seg_cols, T, y, ys);
        else {  /* long rows: 8 warps per block (the 8 tokens' activations fill twice the shared memory; twice the warps
                   share them) */
            const int wp = m->cols >= 4096 ? WARPS8 : WARPS;
            k_q8ws<8><<<((m->rows + 15) / 16 + wp - 1) / wp, wp * 32, 2 * sm4, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs,
                                                                                   seg_rows, seg_cols, T, y, ys);
        }
    }
    else if (m->rows >= wide_min)
        { if (T == 1) k_q8w<1><<<((m->rows + 15) / 16 + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs,
                                                                             seg_rows, seg_cols, T, y, ys); else k_q8w<TTN><<<((m->rows + 15) / 16 + WARPS - 1) / WARPS, WARPS * 32, 0, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs,
                                                                             seg_rows, seg_cols, T, y, ys); }
    else
        { if (T == 1) k_q8<1><<<(m->rows + 15) / 16, 128, m->cols + m->cols / QK * 4, st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs, seg_rows, seg_cols, T, y, ys); else k_q8<TTN><<<(m->rows + 15) / 16, 128, TTN * (m->cols + m->cols / QK * 4), st>>>(m->q, m->s, m->rows, m->cols, xq, xd, xs, seg_rows, seg_cols, T, y, ys); }
}

extern "C" {

/* kernel time of one matrix job (device buffers, no transfers), microseconds per call */
double dg_bench_mat(int mat, int T, int iters) {
    const GMat* m = &GM[mat];
    int8_t* xq = NULL; float *xd = NULL, *xf = NULL, *y = NULL;
    cudaMalloc(&xq, (size_t)T * 32768); cudaMalloc(&xd, (size_t)T * 32768 / QK * 4); cudaMalloc(&xf, (size_t)T * 32768 * 4);
    cudaMalloc(&y, (size_t)T * m->rows * 4);
    cudaMemset(xq, 1, (size_t)T * 32768); cudaMemset(xd, 0, (size_t)T * 32768 / QK * 4); cudaMemset(xf, 0, (size_t)T * 32768 * 4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ms = 0;
    for (int it = -1; it < iters; it++) {
        if (it == 0) cudaEventRecord(a, st);
        if (m->type == 8) q8_launch(m, xq, xd, m->cols, 0, 0, T, y, m->rows);
        else { int grid = (m->rows + 15) / 16; if (T == 1) k_f32<1><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, xf, m->cols, T, y, m->rows);
               else if (T > TTN) k_f32<8><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, xf, m->cols, T, y, m->rows); else k_f32<TTN><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, xf, m->cols, T, y, m->rows); }
    }
    cudaEventRecord(b, st); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b);
    cudaFree(xq); cudaFree(xd); cudaFree(xf); cudaFree(y); cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1e3 / iters;
}

/* run nop jobs for T tokens: inputs to the card (each distinct input once), kernels, outputs back. Returns 0. */
int dg_gemv(const DgOp* ops, int nop, int T) {
    const void* seen[16]; size_t soff[16]; int ns = 0;
    size_t in_off = 0, out_off = 0;
    size_t ioff[16], doff[16], ooff[16];
    for (int o = 0; o < nop; o++) {
        const DgOp* op = &ops[o];
        const GMat* m = &GM[op->mat];
        const void* key = m->type == 8 ? (const void*)op->xq : (const void*)op->xf;
        int i;
        for (i = 0; i < ns && seen[i] != key; i++) {}
        if (i == ns) {
            seen[ns] = key; soff[ns] = in_off; ns++;
            size_t a = m->type == 8 ? (size_t)T * op->xs : (size_t)T * op->xs * 4;
            if (m->type == 8) {
                memcpy(h_in + in_off, op->xq, a);
                memcpy(h_in + in_off + ((a + 255) & ~(size_t)255), op->xd, (size_t)T * (op->xs / QK) * 4);
                a = ((a + 255) & ~(size_t)255) + (((size_t)T * (op->xs / QK) * 4 + 255) & ~(size_t)255);
            } else {
                memcpy(h_in + in_off, op->xf, a);
                a = (a + 255) & ~(size_t)255;
            }
            in_off += a;
            if (in_off > in_cap) { fprintf(stderr, "dsv41_gpu: input staging too small\n"); return -1; }
        }
        ioff[o] = soff[i];
        doff[o] = soff[i] + ((((size_t)T * op->xs) + 255) & ~(size_t)255);
        ooff[o] = out_off;
        out_off += (((size_t)T * m->rows * 4) + 255) & ~(size_t)255;
        if (out_off > out_cap) { fprintf(stderr, "dsv41_gpu: output staging too small\n"); return -1; }
    }
    CK(cudaMemcpyAsync(d_in, h_in, in_off, cudaMemcpyHostToDevice, st));
    for (int o = 0; o < nop; o++) {
        const DgOp* op = &ops[o];
        const GMat* m = &GM[op->mat];
        int grid = (m->rows + 15) / 16;
        float* y = (float*)(d_out + ooff[o]);
        if (m->type == 8)
            q8_launch(m, (const int8_t*)(d_in + ioff[o]), (const float*)(d_in + doff[o]), op->xs, op->seg_rows,
                      op->seg_cols, T, y, m->rows);
        else
            { if (T == 1) k_f32<1><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, (const float*)(d_in + ioff[o]), op->xs, T, y, m->rows); else if (T > TTN) k_f32<8><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, (const float*)(d_in + ioff[o]), op->xs, T, y, m->rows); else k_f32<TTN><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, (const float*)(d_in + ioff[o]), op->xs, T, y, m->rows); }
    }
    CK(cudaMemcpyAsync(h_out, d_out, out_off, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    for (int o = 0; o < nop; o++) {
        const DgOp* op = &ops[o];
        const GMat* m = &GM[op->mat];
        const float* src = (const float*)(h_out + ooff[o]);
        for (int t = 0; t < T; t++) memcpy(op->y + (size_t)t * op->ys, src + (size_t)t * m->rows, (size_t)m->rows * 4);
    }
    return 0;
}

}  /* extern "C" */

#include "dsv41_gpu_fwd.cuh"
