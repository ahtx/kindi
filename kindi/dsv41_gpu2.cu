/* DeepSeek-V4.1-Flash: a second card as a routed-expert cache (expert cards, KINDI_DSV_GPU2).

   Loaded by dsv41_cpu.c with dlopen, one copy per extra card (each copy has its own CUDA runtime state). A card holds
   a static set of routed experts in MXFP4 and computes the (token, slot) pairs whose expert it holds, while the host
   computes the other pairs. The arithmetic is the one of the hot-expert kernels in dsv41_gpu_fwd.cuh (moe_cpu.c's
   per-lane order: codes -> doubled int8 weights, 8 int32 lanes of 4 products per block, fma with scale/2 * activation
   scale, hsum8), so the values do not depend on which card computes a pair. Built for older cards too (sm_52 and
   up): without __dp4a the 4 products are summed in integers, which gives the same int32.
   The main card's library has its own CUDA runtime but shares the driver's per-thread current context, so every entry
   point here makes this card's context current and puts the caller's context back on return (CtxGuard): otherwise
   the main card's next calls on this thread would go to this card. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK 32
#define D 5120
#define FF 2304
#define NL 40
#define NE 384
#define NS 6
#define MAXT 8                    /* forwards of up to MAXT tokens (decode, verify), like HOTT */
#define MAXX 1024

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "dsv41_gpu2: %s at line %d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

typedef struct { const uint8_t *gq, *gs, *uq, *us, *dq, *ds; } XHot;

__device__ __forceinline__ float d_expf(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float f16r(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int dot4(int a, int b) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, 0);
#else
    return (int)(int8_t)a * (int)(int8_t)b + (int)(int8_t)(a >> 8) * (int)(int8_t)(b >> 8) +
           (int)(int8_t)(a >> 16) * (int)(int8_t)(b >> 16) + (int)(int8_t)(a >> 24) * (int)(int8_t)(b >> 24);
#endif
}
/* 4 FP4 codes of lane l (weights 4l .. 4l+3 of the block) -> 4 doubled int8 weights packed in an int. Two byte
   permutes over the magnitude and the negative table instead of a constant-memory table (its divergent reads
   serialize, which made this card 10x slower than its bandwidth); same values as c_fp4x2. */
__device__ __forceinline__ int fp4_lane(uint32_t v, int l) {
    if (l >= 4) v >>= 4;
    const uint32_t n = v & 0x0F0F0F0Fu, i = n & 0x07070707u;
    uint32_t x = (i | (i >> 4)) & 0x00FF00FFu;
    const uint32_t sel = (x & 0xFFu) | ((x >> 8) & 0xFF00u);
    const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, sel);     /* 0 1 2 3 4 6 8 12 */
    const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel);     /* 0 -1 -2 -3 -4 -6 -8 -12 */
    const uint32_t m = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (int)((pos & ~m) | (neg & m));
}
/* 2^(e - 128) exactly (the E8M0 scale / 2), without a table */
__device__ __forceinline__ float e8m0_half(int e) {
    return e >= 2 ? __int_as_float((e - 1) << 23) : (e == 1 ? 0x1p-127f : 0x1p-128f);
}

/* which: 0 gate, 1 up, 2 down; grid (rows / 128, NS, T); slots with hix < 0 are not ours. One thread per row keeps
   the 8 lane sums of moe_cpu.c (lane k = weights 4k .. 4k+3 of each block, blocks in order) and adds them in hsum8
   order at the end, so the value is the same as the 8-lane kernels'. A row's 16 code bytes per block are one
   16-byte load (16 rows of a group are 256 contiguous bytes), 8 blocks in flight. */
#define XR 64
__global__ void __launch_bounds__(XR) k_x_mx(int which_in, const int* hix, const XHot* hot, const int8_t* xq,
        const float* xd, const int8_t* hq, const float* hd, float* gu, float* vg) {
    __shared__ int s_x[D / 4];
    __shared__ float s_xd[D / QK];
    const int s = blockIdx.y, t = blockIdx.z;
    const int hi = hix[t * NS + s];
    if (hi < 0) return;
    /* which_in < 0: gate and up in one launch (the first FF / XR blocks gate, the rest up) */
    const int which = which_in >= 0 ? which_in : (int)(blockIdx.x >= FF / XR);
    const int bx = which_in >= 0 ? blockIdx.x : blockIdx.x - which * (FF / XR);
    const int rows = which == 2 ? D : FF, cols = which == 2 ? FF : D, nb = cols / QK;
    const uint8_t* W = which == 0 ? hot[hi].gq : which == 1 ? hot[hi].uq : hot[hi].dq;
    const uint8_t* S = which == 0 ? hot[hi].gs : which == 1 ? hot[hi].us : hot[hi].ds;
    const int8_t* X = which == 2 ? hq + ((size_t)t * NS + s) * FF : xq + (size_t)t * D;
    const float* XD = which == 2 ? hd + ((size_t)t * NS + s) * (FF / QK) : xd + (size_t)t * (D / QK);
    for (int k = threadIdx.x; k < cols / 4; k += blockDim.x) s_x[k] = __ldg((const int*)X + k);
    for (int k = threadIdx.x; k < nb; k += blockDim.x) s_xd[k] = __ldg(XD + k);
    __syncthreads();
    const int row = bx * XR + threadIdx.x;
    if (row >= rows) return;
    const int g = row >> 4, rr = row & 15;
    const uint4* w = (const uint4*)(W + ((size_t)g * nb * 16 + rr) * 16);
    const uint8_t* sc = S + (size_t)g * nb * 16 + rr;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int b0 = 0; b0 < nb; b0 += 8) {
        uint4 cw[8];
        int se[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            cw[j] = __ldg(w + (size_t)(b0 + j) * 16);
            se[j] = __ldg(sc + (size_t)(b0 + j) * 16);
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int b = b0 + j;
            const float sa = e8m0_half(se[j]) * s_xd[b];
            const uint32_t c4[4] = {cw[j].x, cw[j].y, cw[j].z, cw[j].w};
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const int wv = fp4_lane(c4[l & 3], l);
                const int xv = s_x[b * (QK / 4) + l];
                acc[l] = __fmaf_rn((float)dot4(wv, xv), sa, acc[l]);
            }
        }
    }
    const float v = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    if (which == 2) vg[((size_t)t * NS + s) * D + row] = v;
    else gu[((size_t)t * NS + s) * 2 * FF + which * FF + row] = v;
}
__global__ void k_x_act(const int* hix, const float* gu, int8_t* hq, float* hd) {
    const int s = blockIdx.x, t = blockIdx.y;
    if (hix[t * NS + s] < 0) return;
    const float* g = gu + ((size_t)t * NS + s) * 2 * FF;
    int8_t* q = hq + ((size_t)t * NS + s) * FF;
    float* d = hd + ((size_t)t * NS + s) * (FF / QK);
    const int lane = threadIdx.x & 31;
    for (int b = (threadIdx.x >> 5) * 32; b < FF; b += (blockDim.x >> 5) * 32) {
        float gt = g[b + lane], up = g[FF + b + lane];
        if (gt > 10.f) gt = 10.f;
        if (up > 10.f) up = 10.f;
        if (up < -10.f) up = -10.f;
        float v = gt / (1.f + d_expf(-gt)) * up;
        float am = warp_max(fabsf(v));
        float id = am != 0.f ? 127.f / am : 0.f;
        int i = __float2int_rn(v * id);
        q[b + lane] = (int8_t)(i > 127 ? 127 : (i < -128 ? -128 : i));
        if (lane == 0) d[b / QK] = f16r(am / 127.f);
    }
}

static struct {
    int dev, nhot, hotmap[NL][NE];
    XHot hot_h[MAXX];
    XHot* hot_d;
    int *hix_d, *hix_h;
    int8_t *xq_d, *xq_h, *hq;
    float *xd_d, *xd_h, *hd, *gu, *vg, *vg_h;
    size_t bytes;
    cudaStream_t st;
    int T;
    CUcontext ctx;
    cudaEvent_t ev0, ev1;
    double gpu_ms;
} X;

struct CtxGuard {
    CUcontext prev;
    CtxGuard() { prev = NULL; cuCtxGetCurrent(&prev); if (X.ctx) cuCtxSetCurrent(X.ctx); }
    ~CtxGuard() { cuCtxSetCurrent(prev); }
};

static int x_mat(const uint8_t* src, int rows, int cols, const uint8_t** dq, const uint8_t** ds) {
    int nb = cols / QK;
    size_t nq = (size_t)rows * nb * 16, ns = (size_t)rows * nb;
    uint8_t* hq = (uint8_t*)malloc(nq + ns);
    uint8_t* hs = hq + nq;
    for (size_t r = 0; r < (size_t)rows; r++)
        for (int b = 0; b < nb; b++) {
            const uint8_t* B = src + (r * nb + b) * 17;
            size_t gi = ((r / 16) * nb + b) * 16 + r % 16;
            hs[gi] = B[0];
            memcpy(hq + gi * 16, B + 1, 16);
        }
    uint8_t* d = NULL;
    if (cudaMalloc(&d, nq + ns) != cudaSuccess) { free(hq); return -1; }
    cudaMemcpy(d, hq, nq + ns, cudaMemcpyHostToDevice);
    free(hq);
    *dq = d; *ds = d + nq;
    X.bytes += nq + ns;
    return 0;
}

extern "C" {

/* card dev; returns 0 */
int dx_init(int dev) {
    X.dev = dev;
    CtxGuard guard;
    CUdevice cd;
    if (cuInit(0) || cuDeviceGet(&cd, dev) || cuDevicePrimaryCtxRetain(&X.ctx, cd) || cuCtxSetCurrent(X.ctx)) {
        fprintf(stderr, "dsv41_gpu2: cannot open the context of device %d\n", dev);
        return -1;
    }
    CK(cudaSetDevice(dev));
    CK(cudaStreamCreateWithFlags(&X.st, cudaStreamNonBlocking));
    CK(cudaEventCreate(&X.ev0)); CK(cudaEventCreate(&X.ev1));
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) X.hotmap[l][e] = -1;
    CK(cudaMalloc(&X.hot_d, sizeof(XHot) * MAXX));
    /* the inputs of a call in one buffer (one copy): slot map [MAXT][NS], scales [MAXT][D / QK], codes [MAXT][D] */
    const size_t in_bytes = (size_t)MAXT * NS * 4 + (size_t)MAXT * D / QK * 4 + (size_t)MAXT * D;
    uint8_t *ind, *inh;
    CK(cudaMalloc(&ind, in_bytes)); CK(cudaMallocHost(&inh, in_bytes));
    X.hix_d = (int*)ind; X.hix_h = (int*)inh;
    X.xd_d = (float*)(ind + MAXT * NS * 4); X.xd_h = (float*)(inh + MAXT * NS * 4);
    X.xq_d = (int8_t*)(ind + MAXT * NS * 4 + MAXT * D / QK * 4); X.xq_h = (int8_t*)(inh + MAXT * NS * 4 + MAXT * D / QK * 4);
    CK(cudaMalloc(&X.gu, (size_t)MAXT * NS * 2 * FF * 4));
    CK(cudaMalloc(&X.hq, (size_t)MAXT * NS * FF)); CK(cudaMalloc(&X.hd, (size_t)MAXT * NS * FF / QK * 4));
    CK(cudaMalloc(&X.vg, (size_t)MAXT * NS * D * 4)); CK(cudaMallocHost(&X.vg_h, (size_t)MAXT * NS * D * 4));
    return 0;
}
/* one routed expert (GGUF MXFP4 rows) into the card; -1 when full or fewer than reserve bytes would stay free */
int dx_add(int l, int e, const uint8_t* gate, const uint8_t* up, const uint8_t* down, size_t reserve) {
    if (X.nhot >= MAXX || X.hotmap[l][e] >= 0) return -1;
    CtxGuard guard;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    if (fr < (size_t)3 * FF * D / QK * 17 + reserve) return -1;
    XHot* h = &X.hot_h[X.nhot];
    if (x_mat(gate, FF, D, &h->gq, &h->gs) || x_mat(up, FF, D, &h->uq, &h->us) || x_mat(down, D, FF, &h->dq, &h->ds))
        return -1;
    X.hotmap[l][e] = X.nhot++;
    return 0;
}
int dx_commit(void) {
    CtxGuard guard;
    CK(cudaMemcpy(X.hot_d, X.hot_h, sizeof(XHot) * MAXX, cudaMemcpyHostToDevice));
    return X.nhot;
}
size_t dx_bytes(void) { return X.bytes; }
double dx_gpu_ms(void) { return X.gpu_ms; }
int dx_has(int l, int e) { return X.hotmap[l][e] >= 0; }

/* start the pairs of layer l whose expert is on the card and mine[i] != 0 (i = t * NS + s); xq / xd: the Q8 input
   [T][D] as moe_cpu.c quantizes it. Returns the number of pairs started (0: nothing to wait for). */
int dx_start(int l, int T, const int8_t* xq, const float* xd, const int* sel, const uint8_t* mine) {
    if (T > MAXT) return 0;
    int n = 0;
    for (int i = 0; i < T * NS; i++) {
        X.hix_h[i] = mine[i] ? X.hotmap[l][sel[i]] : -1;
        n += X.hix_h[i] >= 0;
    }
    if (!n) return 0;
    CtxGuard guard;
    cudaEventRecord(X.ev0, X.st);
    memcpy(X.xq_h, xq, (size_t)T * D);
    memcpy(X.xd_h, xd, (size_t)T * (D / QK) * 4);
    cudaMemcpyAsync(X.hix_d, X.hix_h, (size_t)MAXT * NS * 4 + (size_t)MAXT * D / QK * 4 + (size_t)T * D,
                    cudaMemcpyHostToDevice, X.st);
    k_x_mx<<<dim3(2 * FF / XR, NS, T), XR, 0, X.st>>>(-1, X.hix_d, X.hot_d, X.xq_d, X.xd_d, NULL, NULL, X.gu, NULL);
    k_x_act<<<dim3(NS, T), 256, 0, X.st>>>(X.hix_d, X.gu, X.hq, X.hd);
    k_x_mx<<<dim3(D / XR, NS, T), XR, 0, X.st>>>(2, X.hix_d, X.hot_d, NULL, NULL, X.hq, X.hd, NULL, X.vg);
    int i0 = T * NS, i1 = -1;               /* one copy of the rows from the first to the last computed pair */
    for (int i = 0; i < T * NS; i++)
        if (X.hix_h[i] >= 0) { if (i < i0) i0 = i; i1 = i; }
    cudaMemcpyAsync(X.vg_h + (size_t)i0 * D, X.vg + (size_t)i0 * D, (size_t)(i1 - i0 + 1) * D * 4, cudaMemcpyDeviceToHost, X.st);
    cudaEventRecord(X.ev1, X.st);
    X.T = T;
    return n;
}
/* wait for dx_start's pairs and write their values into parts [T][NS][D] */
int dx_finish(float* parts) {
    CtxGuard guard;
    CK(cudaStreamSynchronize(X.st));
    float ms = 0.f;
    cudaEventElapsedTime(&ms, X.ev0, X.ev1);
    X.gpu_ms += ms;
    for (int i = 0; i < X.T * NS; i++)
        if (X.hix_h[i] >= 0) memcpy(parts + (size_t)i * D, X.vg_h + (size_t)i * D, D * 4);
    return 0;
}

}
