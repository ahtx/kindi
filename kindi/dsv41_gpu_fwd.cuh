/* dsv41_gpu_fwd.cuh - the whole DeepSeek-V4.1-Flash layer on the GPU except the routed experts (included by
   dsv41_gpu.cu). Every kernel repeats the CPU engine's arithmetic (dsv41_cpu.c), operation for operation, with three
   deliberate exceptions that can move a result by one ulp in rare cases: expf / log1pf are computed in double and
   rounded (glibc's float functions are not correctly rounded in every case), and the fp64 sums of squares and of the
   engram gate are tree sums instead of the CPU's 4-lane sequential sums (the sums are fp64, so the float results are
   equal except in rare near-ties). The token gate is the test that counts; per-layer comparisons with the CPU
   engine (dg_debug) show how close the two are.

   Per layer: engram (layers 1, 14), the attention sub-layer, the FFN sub-layer up to the router and the shared
   expert; then the normalized FFN input and the routing go to the host, the host's routed experts (moe_cpu.c)
   come back, and the layer ends on the card. One round trip per layer. */

#define D 5120
#define NL 40
#define NH 64
#define HD 512
#define RD 64
#define QR 1280
#define OR 1024
#define OG 8
#define NE 384
#define NS 6
#define FF 2304
#define WIN 128
#define IH 32
#define IHD 128
#define ITOPK 512
#define HC 4
#define MIX 24
#define VOCAB 129280
#define CBLK 8
#define CTOPK 2048
#define CAND_SRC 20
#define ENG_COLS 24
#define ENG_HD 256
#define ACS 160
#define AGH 8
#define HKC 8
#define ANCH ((WIN + ITOPK + ACS - 1) / ACS)
#define MAXE (WIN + ITOPK)
#define RPARTS 8                  /* parts (blocks) of the fp64 sum of squares of the stream per token */
#define CRING 16                  /* compressor input ring (see dsv41_cpu.c) */

enum { F_FP8_KV = 1, F_FP8_O = 2, F_FP4_IDX = 4, F_FP4_CKV = 8, F_BF16_ENG = 16, F_F16_CACHE = 32, F_F16_Q = 64,
       F_LLAMA_FA = 128 };

/* ---------------------------------------------------------------- numerics */
__device__ __forceinline__ float d_expf(float x) { return (float)exp((double)x); }
__device__ __forceinline__ float d_log1pf(float x) { return (float)log1p((double)x); }
__device__ __forceinline__ float f16r(float x) { return __half2float(__float2half_rn(x)); }

/* hsum8 of 8 lane values in the CPU order */
__device__ __forceinline__ float hsum8v(const float* s) { return ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7])); }

__device__ __forceinline__ float v_rne_e4m3(float x) {      /* the AVX2 bit form in dsv41_cpu.c */
    float a = fabsf(x);
    uint32_t u = __float_as_uint(x);
    uint32_t r = u + 0x7ffffu + ((u >> 20) & 1u);
    float vn = __uint_as_float(r & 0xfff00000u);
    float vs = rintf(x * 512.f) * (1.f / 512.f);
    return a < 0x1p-6f ? vs : vn;
}
__device__ __forceinline__ float v_rne_e2m1(float x) {
    float a = fabsf(x);
    bool lt2 = a < 2.f, lt4 = a < 4.f;
    float iq = lt2 ? 2.f : (lt4 ? 1.f : 0.5f), q = lt2 ? 0.5f : (lt4 ? 1.f : 2.f);
    return rintf(x * iq) * q;
}
__device__ __forceinline__ float rne_e4m3_s(float x) {      /* scalar form (frexpf / ldexpf) */
    float a = fabsf(x);
    if (a == 0.f) return x;
    float q;
    if (a < 0x1p-6f) q = 0x1p-9f;
    else { int e; frexpf(a, &e); q = ldexpf(1.f, e - 1 - 3); }
    return rintf(x / q) * q;
}
__device__ __forceinline__ float pow2_ceil(float v) {
    uint32_t b = __float_as_uint(v);
    int e = (int)((b >> 23) & 0xff) - 127 + ((b & 0x7fffff) != 0);
    return ldexpf(1.f, e);
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ double warp_sumd(double v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
/* fp64 sum over a thread block (blockDim.x multiple of 32, <= 1024); all threads get the result */
__device__ double block_sumd(double v) {
    __shared__ double sh[32];
    v = warp_sumd(v);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    double r = l < nw ? sh[l] : 0.0;
    r = warp_sumd(r);
    return r;
}

/* ---------------------------------------------------------------- simple kernels */
/* in-place rounding of rows: n values per row (multiple of the block), rows = gridDim.x; one warp per 32-block
   (FP8 E4M3 with power-of-2 scales per 32, FP4 with power-of-2 scales per 32, FP4 with E4M3 scales per 16) */
__global__ void k_sim_fp8(float* x, int n, int stride) {      /* grid (n / blockDim.x, rows) */
    float* r = x + (size_t)blockIdx.y * stride + (size_t)blockIdx.x * blockDim.x;
    int lane = threadIdx.x & 31;
    for (int b = (threadIdx.x >> 5) * 32; b < (int)blockDim.x && blockIdx.x * blockDim.x + b < n; b += (blockDim.x >> 5) * 32) {
        float v = r[b + lane];
        float am = warp_max(fabsf(v));
        float s = pow2_ceil(fmaxf(am, 1e-4f) * (1.f / 448.f)), is = 1.f / s;
        float u = fminf(fmaxf(v * is, -448.f), 448.f);
        r[b + lane] = v_rne_e4m3(u) * s;
    }
}
__global__ void k_sim_fp4p2(float* x, int n, int stride) {
    float* r = x + (size_t)blockIdx.x * stride;
    int lane = threadIdx.x & 31;
    for (int b = (threadIdx.x >> 5) * 32; b < n; b += (blockDim.x >> 5) * 32) {
        float v = r[b + lane];
        float am = warp_max(fabsf(v));
        float s = pow2_ceil(fmaxf(am, 6.f * 0x1p-126f) * (1.f / 6.f)), is = 1.f / s;
        float u = fminf(fmaxf(v * is, -6.f), 6.f);
        r[b + lane] = v_rne_e2m1(u) * s;
    }
}
__device__ void sim_fp4e4_16(float* r, int n) {             /* one warp, n % 32 == 0: two 16-blocks per step */
    int lane = threadIdx.x & 31;
    for (int b = 0; b < n; b += 32) {
        float v = r[b + lane];
        float am = fabsf(v);
        for (int o = 8; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));   /* within 16 lanes */
        float s = rne_e4m3_s(fmaxf(am, 6.f * 0x1p-9f) / 6.f);
        float u = fminf(fmaxf(v / s, -6.f), 6.f);
        r[b + lane] = v_rne_e2m1(u) * s;
    }
}

/* Q8 quantization of rows (the llama.cpp rounding); n values per row, rows = gridDim.x; one warp per 32-block */
__global__ void k_quant(const float* x, int n, int xstride, int8_t* q, float* d) {
    const float* r = x + (size_t)blockIdx.x * xstride;
    int8_t* qr = q + (size_t)blockIdx.x * n;
    float* dr = d + (size_t)blockIdx.x * (n / QK);
    int lane = threadIdx.x & 31;
    for (int b = (threadIdx.x >> 5) * 32; b < n; b += (blockDim.x >> 5) * 32) {
        float v = r[b + lane];
        float am = warp_max(fabsf(v));
        float id = am != 0.f ? 127.f / am : 0.f;
        int i = __float2int_rn(v * id);
        i = i > 127 ? 127 : (i < -128 ? -128 : i);
        qr[b + lane] = (int8_t)i;
        if (lane == 0) dr[b / QK] = f16r(am / 127.f);
    }
}

/* k_quant with a warp per 32-block over many thread blocks (grid (n / 256, rows), 256 threads): the same arithmetic
   per block, so the same bits, without one thread block walking the whole row */
__global__ void k_quantm(const float* x, int n, int xstride, int8_t* q, float* d) {
    const int b = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 32, lane = threadIdx.x & 31;
    if (b >= n) return;
    const float* r = x + (size_t)blockIdx.y * xstride;
    float v = r[b + lane];
    float am = warp_max(fabsf(v));
    float id = am != 0.f ? 127.f / am : 0.f;
    int i = __float2int_rn(v * id);
    i = i > 127 ? 127 : (i < -128 ? -128 : i);
    q[(size_t)blockIdx.y * n + b + lane] = (int8_t)i;
    if (lane == 0) d[(size_t)blockIdx.y * (n / QK) + b / QK] = f16r(am / 127.f);
}

/* RMS norm of rows in place (or to y), weights w (may be NULL); rows = gridDim.x */
__global__ void k_rmsnorm(float* y, int ystride, const float* x, int xstride, const float* w, int n) {
    const float* r = x + (size_t)blockIdx.x * xstride;
    float* o = y + (size_t)blockIdx.x * ystride;
    double s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += (double)r[i] * (double)r[i];
    s = block_sumd(s);
    float rr = 1.f / sqrtf((float)(s / n) + 1e-20f);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = (w ? w[i] : 1.f) * (r[i] * rr);
}

/* k_rmsnorm + k_quant in two short kernels over many thread blocks, with k_rmsnorm's exact fp64 tree: k_rmsnorm's
   1024 threads become 1024 virtual threads in 8 blocks of 128 (thread v sums elements v, v + 1024, ... in order, then
   the butterfly of its warp); k_rmsq takes the 32 warp sums (block_sumd's second stage), normalizes its part of the
   row and quantizes it as k_quant (a warp per 32-block). wsum [rows][32]. */
__global__ void __launch_bounds__(128) k_rms_part(const float* x, int xstride, int n, double* wsum) {
    const float* r = x + (size_t)blockIdx.y * xstride;
    const int v = blockIdx.x * 128 + threadIdx.x;
    double s = 0;
    for (int i = v; i < n; i += 1024) s += (double)r[i] * (double)r[i];
    s = warp_sumd(s);
    if ((threadIdx.x & 31) == 0) wsum[(size_t)blockIdx.y * 32 + (v >> 5)] = s;
}
__global__ void __launch_bounds__(256) k_rmsq(float* y, int ystride, const float* x, int xstride, const float* w, int n,
                                              const double* wsum, int8_t* q, float* d) {
    __shared__ float rrs;
    const int lane = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        double r = warp_sumd(wsum[(size_t)blockIdx.y * 32 + lane]);
        if (lane == 0) rrs = 1.f / sqrtf((float)(r / n) + 1e-20f);
    }
    __syncthreads();
    const float rr = rrs;
    const int b = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 32;
    if (b >= n) return;
    const int i = b + lane;
    const float o = (w ? w[i] : 1.f) * (x[(size_t)blockIdx.y * xstride + i] * rr);
    y[(size_t)blockIdx.y * ystride + i] = o;
    float am = warp_max(fabsf(o));
    float id = am != 0.f ? 127.f / am : 0.f;
    int qi = __float2int_rn(o * id);
    qi = qi > 127 ? 127 : (qi < -128 ? -128 : qi);
    q[(size_t)blockIdx.y * n + i] = (int8_t)qi;
    if (lane == 0) d[(size_t)blockIdx.y * (n / QK) + b / QK] = f16r(am / 127.f);
}

/* RoPE on the last RD dims of nv vectors of length len per token; cs [T][cstride] (32 cos/sin pairs at csoff) */
__global__ void k_rope(float* x, int nv, int len, const float* cs, int cstride, int csoff, int inverse) {
    int t = blockIdx.x, v = blockIdx.y * blockDim.y + threadIdx.y, i = threadIdx.x;   /* i < RD / 2 */
    if (v >= nv) return;
    const float* c2 = cs + (size_t)t * cstride + csoff;
    float c = c2[2 * i], s = inverse ? -c2[2 * i + 1] : c2[2 * i + 1];
    float* p = x + ((size_t)t * nv + v) * len + len - RD;
    float x0 = p[2 * i], x1 = p[2 * i + 1];
    p[2 * i] = x0 * c - x1 * s;
    p[2 * i + 1] = x0 * s + x1 * c;
}

__global__ void k_round_f16(float* x, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] = f16r(x[i]);
}
__global__ void k_scale(float* x, size_t n, float s) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] *= s;
}
__global__ void k_add(float* y, const float* x, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] += x[i];
}

/* ---------------------------------------------------------------- hyper-connections */
/* the 24 x 20480 F32 mix projection: thread (t, row, chunk, lane 0..15), lane j sums w[k + j] * x[k + j] for
   k = 0, 16, ... of its chunk; the chunk value = hsum8 of (lane l + lane l + 8); part [HKC][T][MIX] */
__global__ void k_hcmix(const float* W, const float* h, int T, float* part) {
    int t = blockIdx.x, r = blockIdx.y, c = threadIdx.x >> 4, j = threadIdx.x & 15;   /* 128 threads */
    const int kc = HC * D / HKC;
    const float* w = W + (size_t)r * HC * D + (size_t)c * kc;
    const float* x = h + (size_t)t * HC * D + (size_t)c * kc;
    float a = 0.f;
#pragma unroll 8
    for (int k = 0; k < kc; k += 16) a = __fmaf_rn(w[k + j], x[k + j], a);
    float b = __shfl_down_sync(0xffffffffu, a, 8, 16);
    __shared__ float s[HKC][8];
    if (j < 8) s[c][j] = a + b;
    __syncthreads();
    if (j == 0) part[((size_t)c * T + t) * MIX + r] = hsum8v(s[c]);
}

/* per token: the mixes -> pre / post / comb (Sinkhorn), as mix_tok; one thread per token; rs = 1 / rms of h */
__global__ void k_mix(const float* part, const float* h, int T, const float* scale, const float* base, float* pre,
                      float* post, float* comb, const double* rsv) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float eps = 1e-6f;
    double ss = 0;
    for (int c = 0; c < RPARTS; c++) ss += rsv[t * RPARTS + c];
    float r = 1.f / sqrtf((float)(ss / (HC * D)) + 1e-20f);
    float m[MIX];
    for (int i = 0; i < MIX; i++) {
        float v = part[(size_t)t * MIX + i];
        for (int c = 1; c < HKC; c++) v += part[((size_t)c * T + t) * MIX + i];
        m[i] = v * r;
    }
    float *pr = pre + t * HC, *po = post + t * HC, *c = comb + t * 16;
    for (int j = 0; j < HC; j++) pr[j] = 1.f / (1.f + d_expf(-(m[j] * scale[0] + base[j]))) + eps;
    for (int j = 0; j < HC; j++) po[j] = 2.f / (1.f + d_expf(-(m[j + HC] * scale[1] + base[j + HC])));
    for (int j = 0; j < HC; j++)
        for (int k = 0; k < HC; k++) c[j * HC + k] = m[j * HC + k + 2 * HC] * scale[2] + base[j * HC + k + 2 * HC];
    for (int j = 0; j < HC; j++) {
        float mx = c[j * HC];
        for (int k = 1; k < HC; k++) mx = fmaxf(mx, c[j * HC + k]);
        float s2 = 0;
        for (int k = 0; k < HC; k++) { c[j * HC + k] = d_expf(c[j * HC + k] - mx); s2 += c[j * HC + k]; }
        for (int k = 0; k < HC; k++) c[j * HC + k] = c[j * HC + k] / s2 + eps;
    }
    for (int it = 0; it < 20; it++) {
        if (it > 0)
            for (int j = 0; j < HC; j++) {
                float s2 = 0;
                for (int k = 0; k < HC; k++) s2 += c[j * HC + k];
                for (int k = 0; k < HC; k++) c[j * HC + k] /= s2 + eps;
            }
        for (int k = 0; k < HC; k++) {
            float s2 = 0;
            for (int j = 0; j < HC; j++) s2 += c[j * HC + k];
            for (int j = 0; j < HC; j++) c[j * HC + k] /= s2 + eps;
        }
    }
}

/* the same as k_mix with a warp per token: lane i < 24 makes mix i; lanes 0-3 pre, 4-7 post, 8-23 the 4 x 4 comb
   entry (lane - 8) = (j, k); every sum is gathered by shuffles and added in the CPU's order (left to right) */
__global__ void k_mixw(const float* part, int T, const float* scale, const float* base, float* pre, float* post,
                       float* comb, const double* rsv, int which) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float eps = 1e-6f;
    double ss = 0;
    for (int c = 0; c < RPARTS; c++) ss += rsv[t * RPARTS + c];
    const float r = 1.f / sqrtf((float)(ss / (HC * D)) + 1e-20f);
    float m = 0.f;
    if (lane < MIX) {
        float v = part[(size_t)t * MIX + lane];
        for (int c = 1; c < HKC; c++) v += part[((size_t)c * T + t) * MIX + lane];
        m = v * r;
    }
    if (which != 2) {
        if (lane < HC) pre[t * HC + lane] = 1.f / (1.f + d_expf(-(m * scale[0] + base[lane]))) + eps;
        else if (lane < 2 * HC) post[t * HC + lane - HC] = 2.f / (1.f + d_expf(-(m * scale[1] + base[lane])));
        if (which == 1) return;
    }
    /* comb: lane 8 + j * 4 + k */
    const bool cl = lane >= 2 * HC && lane < MIX;
    const int ci = lane - 2 * HC, j = ci >> 2, k = ci & 3;
    float c = cl ? m * scale[2] + base[lane] : 0.f;
    /* row softmax: max over k, exp, sum in k order, then / sum + eps */
    const int rb = 2 * HC + j * 4;                 /* lane of (j, 0) */
    float mx = __shfl_sync(0xffffffffu, c, cl ? rb : 0);
    for (int q = 1; q < HC; q++) mx = fmaxf(mx, __shfl_sync(0xffffffffu, c, cl ? rb + q : 0));
    c = cl ? d_expf(c - mx) : 0.f;
    float s2 = 0.f;
    for (int q = 0; q < HC; q++) s2 += __shfl_sync(0xffffffffu, c, cl ? rb + q : 0);
    c = c / s2 + eps;
    for (int it = 0; it < 20; it++) {
        if (it > 0) {
            float a = 0.f;
            for (int q = 0; q < HC; q++) a += __shfl_sync(0xffffffffu, c, cl ? rb + q : 0);
            c /= a + eps;
        }
        float b = 0.f;
        for (int q = 0; q < HC; q++) b += __shfl_sync(0xffffffffu, c, cl ? 2 * HC + q * 4 + k : 0);
        c /= b + eps;
    }
    if (cl) comb[t * 16 + ci] = c;
}

/* fp64 sums of squares of HC * D values per token in RPARTS parts (blocks), for the mix */
__global__ void k_rinv(const float* x, int n, double* out) {
    const float* r = x + (size_t)blockIdx.x * n + (size_t)blockIdx.y * (n / RPARTS);
    double s = 0;
    for (int i = threadIdx.x; i < n / RPARTS; i += blockDim.x) s += (double)r[i] * (double)r[i];
    s = block_sumd(s);
    if (threadIdx.x == 0) out[blockIdx.x * RPARTS + blockIdx.y] = s;
}

/* x[t] = sum_c pre_in[t][c] * h[t][c] (v = p0 * h0, then v = v + pc * hc) */
__global__ void k_collapse(const float* h, const float* pin, float* x) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    const float* ht = h + (size_t)t * HC * D;
    const float* p = pin + t * HC;
    float v = p[0] * ht[d];
    for (int k = 1; k < HC; k++) v = v + p[k] * ht[k * D + d];
    x[(size_t)t * D + d] = v;
}
/* the final collapse before the head: v = 0, v += pc * hc */
__global__ void k_collapse0(const float* h, const float* pin, float* x) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    const float* ht = h + (size_t)t * HC * D;
    const float* p = pin + t * HC;
    float v = 0;
    for (int k = 0; k < HC; k++) v += p[k] * ht[k * D + d];
    x[(size_t)t * D + d] = v;
}

/* h[t][j] = post[j] * y + sum_i comb[i][j] * h[t][i] */
__global__ void k_post(float* h, const float* y, const float* post, const float* comb) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    float* ht = h + (size_t)t * HC * D;
    const float *po = post + t * HC, *c = comb + t * 16;
    float v = y[(size_t)t * D + d], r[HC], o[HC];
    for (int i = 0; i < HC; i++) r[i] = ht[i * D + d];
    for (int j = 0; j < HC; j++) {
        float s = c[j] * r[0];
        for (int i = 1; i < HC; i++) s = s + c[i * HC + j] * r[i];
        o[j] = po[j] * v + s;
    }
    for (int j = 0; j < HC; j++) ht[j * D + d] = o[j];
}

/* embeddings: h[t][c] = e[t] for the 4 copies, pre = (1, 0, 0, 0) */
__global__ void k_embed(const float* e, float* h, float* pre) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    float v = e[(size_t)t * D + d];
    for (int c = 0; c < HC; c++) h[((size_t)t * HC + c) * D + d] = v;
    if (d < HC) pre[t * HC + d] = d == 0;
}

/* engram gate: block per (token, copy c): dt from fp64 sums, then h[c] += g * val */
__global__ void k_eng_gate(float* h, const float* ekv, const float* gw) {
    int t = blockIdx.x, c = blockIdx.y;
    float* hc = h + ((size_t)t * HC + c) * D;
    const float* key = ekv + (size_t)t * (HC + 1) * D;
    const float *kc = key + (size_t)c * D, *val = key + (size_t)HC * D, *wc = gw + (size_t)c * D;
    double sh = 0, sk = 0, dot = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sh += (double)hc[d] * hc[d];
        sk += (double)kc[d] * kc[d];
        dot += (double)hc[d] * wc[d] * kc[d];
    }
    sh = block_sumd(sh); __syncthreads();
    sk = block_sumd(sk); __syncthreads();
    dot = block_sumd(dot);
    float rstd = (1.f / sqrtf((float)(sh / D) + 1e-20f)) * (1.f / sqrtf((float)(sk / D) + 1e-20f));
    float dt = (float)dot * rstd * (1.f / sqrtf((float)D));
    float g = sqrtf(fmaxf(fabsf(dt), 1e-6f));
    g = 1.f / (1.f + d_expf(-(dt < 0 ? -g : g)));
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += blockDim.x) hc[d] += g * val[d];
}

/* ---------------------------------------------------------------- attention pieces */
/* kv of the new tokens: rms norm, RoPE, FP8 rounding, fp16; into the window ring. Block per token (HD threads). */
__global__ void k_kv_win(const float* kvw, const float* kvnorm, const float* cs, int csoff, float* win, int W, const int* ctl,
                         int flags) {
    const int pos = ctl[0];
    int t = blockIdx.x, d = threadIdx.x;
    __shared__ float v[HD];
    const float* x = kvw + (size_t)t * HD;
    double s = (double)x[d] * (double)x[d];
    s = block_sumd(s);
    float rr = 1.f / sqrtf((float)(s / HD) + 1e-20f);
    v[d] = kvnorm[d] * (x[d] * rr);
    __syncthreads();
    if (d < RD / 2) {
        const float* c2 = cs + (size_t)t * 2 * RD + csoff;
        float c = c2[2 * d], sn = c2[2 * d + 1];
        float x0 = v[HD - RD + 2 * d], x1 = v[HD - RD + 2 * d + 1];
        v[HD - RD + 2 * d] = x0 * c - x1 * sn;
        v[HD - RD + 2 * d + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
    float y = v[d];
    if (flags & F_FP8_KV) {
        float am = warp_max(fabsf(y));
        float sc = pow2_ceil(fmaxf(am, 1e-4f) * (1.f / 448.f)), is = 1.f / sc;
        y = v_rne_e4m3(fminf(fmaxf(y * is, -448.f), 448.f)) * sc;
    }
    if (flags & F_F16_CACHE) y = f16r(y);
    win[(size_t)((pos + t) % W) * HD + d] = y;
}

/* compressor of a kv source layer: one block of HD threads, tokens in order (the pooling state carries over between
   calls); writes the latents of the completed groups, rms-normed, to lat [nl][HD] and quantized */
__global__ void k_compress(const float* ck, const float* cg, int T, const int* ctl, int r, float* cst_kv, float* cst_sc,
                           const float* cnorm, float* lat, int8_t* latq, float* latd) {
    const int pos = ctl[0];
    int d = threadIdx.x, nl = 0;
    for (int t = 0; t < T; t++) {
        int p = pos + t;
        float x;
        if (r == 1) x = ck[(size_t)t * HD + d];
        else {
            int slot = p % CRING, s0 = (p + CRING - 1) % CRING;
            cst_kv[slot * HD + d] = ck[(size_t)t * HD + d];
            cst_sc[slot * HD + d] = cg[(size_t)t * HD + d];
            if ((p + 1) % r) continue;
            float a = cst_sc[s0 * HD + d], b = cst_sc[slot * HD + d], m = fmaxf(a, b);
            float ea = d_expf(a - m), eb = d_expf(b - m), s = ea + eb;
            x = cst_kv[s0 * HD + d] * (ea / s) + cst_kv[slot * HD + d] * (eb / s);
        }
        double q = (double)x * (double)x;
        q = block_sumd(q);
        float rr = 1.f / sqrtf((float)(q / HD) + 1e-20f);
        float y = cnorm[d] * (x * rr);
        lat[(size_t)nl * HD + d] = y;
        float am = warp_max(fabsf(y));
        float id = am != 0.f ? 127.f / am : 0.f;
        int i = __float2int_rn(y * id);
        latq[(size_t)nl * HD + d] = (int8_t)(i > 127 ? 127 : (i < -128 ? -128 : i));
        if ((d & 31) == 0) latd[(size_t)nl * (HD / QK) + d / QK] = f16r(am / 127.f);
        nl++;
        __syncthreads();
    }
}

/* the latents -> index keys ik[g] and compressed kv ckv[g]; block per latent (HD threads); cs_lat [nl][RD] */
__global__ void k_lat(const float* lat, const float* latk, const int* latg, const float* cs_lat, const float* iknorm,
                      float* ik, float* ckv, int flags, const int* nlp) {
    if ((int)blockIdx.x >= *nlp) return;
    int i = blockIdx.x, d = threadIdx.x, g = latg[i];
    __shared__ float kn[IHD], lv[HD];
    __shared__ double ws[HD / 32];
    const float* cs = cs_lat + (size_t)i * RD;
    double s = d < IHD ? (double)latk[(size_t)i * IHD + d] * (double)latk[(size_t)i * IHD + d] : 0.0;
    s = warp_sumd(s);
    if ((d & 31) == 0) ws[d >> 5] = s;
    __syncthreads();
    s = (ws[0] + ws[1]) + (ws[2] + ws[3]);
    if (d < IHD) {
        float rr = 1.f / sqrtf((float)(s / IHD) + 1e-20f);
        kn[d] = iknorm[d] * (latk[(size_t)i * IHD + d] * rr);
    }
    lv[d] = lat[(size_t)i * HD + d];
    __syncthreads();
    if (d < RD / 2) {
        float c = cs[2 * d], sn = cs[2 * d + 1];
        float x0 = kn[IHD - RD + 2 * d], x1 = kn[IHD - RD + 2 * d + 1];
        kn[IHD - RD + 2 * d] = x0 * c - x1 * sn;
        kn[IHD - RD + 2 * d + 1] = x0 * sn + x1 * c;
        x0 = lv[HD - RD + 2 * d]; x1 = lv[HD - RD + 2 * d + 1];
        lv[HD - RD + 2 * d] = x0 * c - x1 * sn;
        lv[HD - RD + 2 * d + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
    if (d < IHD) {
        float y = kn[d];
        if (flags & F_FP4_IDX) {
            float am = warp_max(fabsf(y));
            float sc = pow2_ceil(fmaxf(am, 6.f * 0x1p-126f) * (1.f / 6.f)), is = 1.f / sc;
            y = v_rne_e2m1(fminf(fmaxf(y * is, -6.f), 6.f)) * sc;
        }
        if (flags & F_F16_CACHE) y = f16r(y);
        ik[(size_t)g * IHD + d] = y;
    }
    float y = lv[d];
    if (flags & F_FP4_CKV) {
        float am = fabsf(y);
        for (int o = 8; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        float sc = rne_e4m3_s(fmaxf(am, 6.f * 0x1p-9f) / 6.f);
        y = v_rne_e2m1(fminf(fmaxf(y / sc, -6.f), 6.f)) * sc;
    }
    if (flags & F_F16_CACHE) y = f16r(y);
    ckv[(size_t)g * HD + d] = y;
}

/* index query prep: RoPE on the 32 heads (compressed frequencies), FP4 rounding; weights scaled. 32 warps. */
__global__ void k_iq_prep(float* iq, float* iw, const float* cs, int flags) {
    int t = blockIdx.x, hh = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float* q = iq + ((size_t)t * IH + hh) * IHD;
    const float* c2 = cs + (size_t)t * 2 * RD + RD;
    float c = c2[2 * lane], sn = c2[2 * lane + 1];
    float x0 = q[IHD - RD + 2 * lane], x1 = q[IHD - RD + 2 * lane + 1];
    __syncwarp();
    q[IHD - RD + 2 * lane] = x0 * c - x1 * sn;
    q[IHD - RD + 2 * lane + 1] = x0 * sn + x1 * c;
    __syncwarp();
    if (flags & F_FP4_IDX)
        for (int b = 0; b < IHD; b += 32) {
            float v = q[b + lane];
            float am = warp_max(fabsf(v));
            float s = pow2_ceil(fmaxf(am, 6.f * 0x1p-126f) * (1.f / 6.f)), is = 1.f / s;
            q[b + lane] = v_rne_e2m1(fminf(fmaxf(v * is, -6.f), 6.f)) * s;
        }
    if (lane == 0) iw[t * IH + hh] *= (1.f / sqrtf((float)IHD)) * (1.f / sqrtf((float)IH));
}

/* indexer scores: thread per (token, group); sc [T][scs]; n_t = (pos + t + 1) / r; cand masks above CAND_SRC */
__global__ void k_idx_score(const float* iq, const float* iw, const float* ik, const int* ctl, int r, int use_cand,
                            const uint8_t* cand, int cands, float* sc, int scs) {
    const int pos = ctl[0];
    int t = blockIdx.y;
    int n = (pos + t + 1) / r;
    if (blockIdx.x * blockDim.x >= n) return;
    __shared__ float qt[IHD][IH];
    __shared__ float w[IH];
    for (int i = threadIdx.x; i < IH * IHD; i += blockDim.x) {
        int hh = i / IHD, d = i % IHD;
        qt[d][hh] = iq[(size_t)t * IH * IHD + i];
    }
    if (threadIdx.x < IH) w[threadIdx.x] = iw[t * IH + threadIdx.x];
    __syncthreads();
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n) return;
    float* out = sc + (size_t)t * scs;
    if (use_cand && !cand[(size_t)t * cands + g / CBLK]) { out[g] = -INFINITY; return; }
    const float* k = ik + (size_t)g * IHD;
    float acc[IH];
#pragma unroll
    for (int j = 0; j < IH; j++) acc[j] = 0.f;
    for (int d = 0; d < IHD; d++) {
        float kb = k[d];
#pragma unroll
        for (int j = 0; j < IH; j++) acc[j] = __fmaf_rn(qt[d][j], kb, acc[j]);
    }
    float s8[8];
#pragma unroll
    for (int l = 0; l < 8; l++) s8[l] = 0.f;
#pragma unroll
    for (int j = 0; j < IH / 8; j++)
#pragma unroll
        for (int l = 0; l < 8; l++) s8[l] = __fmaf_rn(fmaxf(acc[8 * j + l], 0.f), w[8 * j + l], s8[l]);
    out[g] = hsum8v(s8);
}

/* layer-20 block scores: max over each block of CBLK; the newest (partial) block +inf */
__global__ void k_blockmax(const float* sc, int scs, const int* ctl, int r, float* bs, int bss) {
    const int pos = ctl[0];
    int t = blockIdx.y, b = blockIdx.x * blockDim.x + threadIdx.x;
    int n = (pos + t + 1) / r, nb = (n + CBLK - 1) / CBLK;
    if (b >= nb) return;
    float m = -INFINITY;
    for (int g = b * CBLK; g < (b + 1) * CBLK && g < n; g++) m = fmaxf(m, sc[(size_t)t * scs + g]);
    if (b == (n - 1) / CBLK) m = INFINITY;
    bs[(size_t)t * bss + b] = m;
}

/* Top-k of a row (the k largest; ties at the threshold: the lowest indices; output ascending), as topk_idx. One
   block (1024 threads) per row: radix select on order-preserving keys, then an ordered compaction. n = (pos + t + 1)
   / r, or the number of CBLK blocks of that (blocks = 1, then cand[b] = 1 for the selected finite blocks). */
__device__ __forceinline__ uint32_t fkey(float f) {
    uint32_t u = __float_as_uint(f == 0.f ? 0.f : f);      /* -0 == +0 as in the float comparisons */
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void k_topk(const float* sc, int scs, const int* ctl, int r, int blocks, int k, int* out, int outs, int* nout,
                       uint8_t* cand, int cands) {
    const int pos = ctl[0];
    int t = blockIdx.x;
    int n = (pos + t + 1) / r;
    if (blocks) n = (n + CBLK - 1) / CBLK;
    const float* s = sc + (size_t)t * scs;
    int* o = out + (size_t)t * outs;
    __shared__ unsigned hist[256];
    __shared__ uint32_t sprefix, smask;
    __shared__ int sk, wsum[32];
    __shared__ unsigned sabove;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    int cnt;
    if (n <= k) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = i;
        cnt = n;
        if (cand) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) cand[(size_t)t * cands + i] = 1;
            if (threadIdx.x == 0 && nout) nout[t] = cnt;
            return;
        }
    } else {
        if (threadIdx.x == 0) { sprefix = 0; smask = 0; sk = k; sabove = 0; }
        __syncthreads();
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            uint32_t pre = sprefix, msk = smask;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                uint32_t kk = fkey(s[i]);
                if ((kk & msk) == pre) atomicAdd(&hist[(kk >> shift) & 255], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                int need = sk, b = 255;
                for (; b > 0; b--) { if ((int)hist[b] >= need) break; need -= hist[b]; }
                sk = need;
                sprefix = pre | ((uint32_t)b << shift);
                smask = msk | (255u << shift);
            }
            __syncthreads();
        }
        const uint32_t th = sprefix;
        unsigned above = 0;
        for (int i = threadIdx.x; i < n; i += blockDim.x) above += fkey(s[i]) > th;
        for (int off = 16; off; off >>= 1) above += __shfl_xor_sync(0xffffffffu, above, off);
        if (lane == 0) atomicAdd(&sabove, above);
        __syncthreads();
        const int neq = k - (int)sabove;
        int base = 0, eqbase = 0;
        const unsigned lm = (1u << lane) - 1;
        for (int i0 = 0; i0 < n; i0 += blockDim.x) {
            int i = i0 + threadIdx.x;
            uint32_t kk = i < n ? fkey(s[i]) : 0;
            int isgt = i < n && kk > th, iseq = i < n && kk == th;
            unsigned beq = __ballot_sync(0xffffffffu, iseq);
            if (lane == 0) wsum[wid] = __popc(beq);
            __syncthreads();
            int peq = 0, teq = 0;
            for (int w = 0; w < nw; w++) { if (w < wid) peq += wsum[w]; teq += wsum[w]; }
            int take = isgt || (iseq && eqbase + peq + __popc(beq & lm) < neq);
            __syncthreads();
            unsigned bt = __ballot_sync(0xffffffffu, take);
            if (lane == 0) wsum[wid] = __popc(bt);
            __syncthreads();
            int pt = 0, tt = 0;
            for (int w = 0; w < nw; w++) { if (w < wid) pt += wsum[w]; tt += wsum[w]; }
            if (take) o[base + pt + __popc(bt & lm)] = i;
            base += tt;
            eqbase += teq;
            __syncthreads();
        }
        cnt = base;
    }
    if (threadIdx.x == 0 && nout) nout[t] = cnt;
    if (cand) {
        uint8_t* cr = cand + (size_t)t * cands;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) cr[i] = 0;
        __syncthreads();
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) { int b = o[j]; if (s[b] > -INFINITY) cr[b] = 1; }
    }
}

/* attention entries of token t: window positions, then the selected compressed positions (ascending) */
__device__ __forceinline__ int attn_n(int p, int ratio, const int* ntop, int t) {
    int w0 = p - WIN + 1 < 0 ? 0 : p - WIN + 1;
    return (p - w0 + 1) + (ratio ? ntop[t] : 0);
}
__device__ __forceinline__ const float* attn_ent(int e, int p, int W, const float* win, const int* topk, int t,
                                                 const float* ckv) {
    int w0 = p - WIN + 1 < 0 ? 0 : p - WIN + 1, nw = p - w0 + 1;
    if (e < nw) return win + (size_t)((w0 + e) % W) * HD;
    return ckv + (size_t)topk[(size_t)t * ITOPK + e - nw] * HD;
}

/* scores q.k * scale for (token, head, entry): 8 lane chains over d (step 8), hsum8; thread per entry.
   qf16: the query rounded to fp16 first (llama.cpp mode) */
__global__ void k_att_scores(const float* q, int qf16, const float* win, int W, const int* ctl, int ratio, const int* topk,
                             const int* ntop, const float* ckv, float* S) {
    const int pos = ctl[0];
    int t = blockIdx.z, h = blockIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x, p = pos + t;
    int n = attn_n(p, ratio, ntop, t);
    if (e >= n) return;
    const float* k = attn_ent(e, p, W, win, topk, t, ckv);
    const float* qh = q + ((size_t)t * NH + h) * HD;
    float a[8];
#pragma unroll
    for (int l = 0; l < 8; l++) a[l] = 0.f;
    for (int d = 0; d < HD; d += 8)
#pragma unroll
        for (int l = 0; l < 8; l++) a[l] = __fmaf_rn(qf16 ? f16r(qh[d + l]) : qh[d + l], k[d + l], a[l]);
    S[((size_t)t * NH + h) * MAXE + e] = hsum8v(a) * (1.f / sqrtf((float)HD));
}

/* Attention with each entry read once for 8 heads (as the CPU's split items). Scores: block per (chunk of ACS
   entries, group of 8 heads, token); thread = (entry, lane); lane l sums q[h][d + l] * k[d + l] over d = 0, 8, ...
   for the 8 heads, hsum8 by shuffles (the same order per head as k_att_scores). */
__global__ void __launch_bounds__(256) k_att_scores8(const float* q, int qf16, const float* win, int W, const int* ctl,
        int ratio, const int* topk, const int* ntop, const float* ckv, float* S) {
    const int pos = ctl[0];
    const int c = blockIdx.x, h0 = blockIdx.y * 8, t = blockIdx.z, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t), e0 = c * ACS;
    if (e0 >= n) return;
    const int ne = n - e0 < ACS ? n - e0 : ACS;
    __shared__ float qs[8][HD];
    for (int i = threadIdx.x; i < 8 * HD; i += blockDim.x) {
        float v = q[((size_t)t * NH + h0) * HD + i];
        qs[i / HD][i % HD] = qf16 ? f16r(v) : v;
    }
    __syncthreads();
    const float scale = 1.f / sqrtf((float)HD);
    for (int i0 = 0; i0 < ne * 8; i0 += blockDim.x) {
        const int i = i0 + threadIdx.x, el = i >> 3, l = i & 7;
        const bool live = el < ne;
        const float* k = attn_ent(e0 + (live ? el : 0), p, W, win, topk, t, ckv);
        float a[8];
#pragma unroll
        for (int h = 0; h < 8; h++) a[h] = 0.f;
        for (int d = 0; d < HD; d += 8) {
            const float kv = k[d + l];
#pragma unroll
            for (int h = 0; h < 8; h++) a[h] = __fmaf_rn(qs[h][d + l], kv, a[h]);
        }
#pragma unroll
        for (int h = 0; h < 8; h++) {
            float v = red8l(a[h]);
            if (live && l == 0) S[((size_t)t * NH + h0 + h) * MAXE + e0 + el] = v * scale;
        }
    }
}
/* token count from which the attention uses the kernels with more work per thread (KINDI_ATT_X_MIN, default 1: verify batches too) */
static int att_x_min() {
    static int v = -1;
    if (v < 0) v = getenv("KINDI_ATT_X_MIN") ? atoi(getenv("KINDI_ATT_X_MIN")) : 1;
    return v;
}
/* the same scores for prompt chunks: each thread makes 4 entries x 8 heads (a query value from shared memory feeds 4
   FMAs instead of 1); per score the lane chain and the reduction are k_att_scores8's (the same bits) */
__global__ void __launch_bounds__(256) k_att_scores8x4(const float* q, int qf16, const float* win, int W, const int* ctl,
        int ratio, const int* topk, const int* ntop, const float* ckv, float* S) {
    const int pos = ctl[0];
    const int c = blockIdx.x, h0 = blockIdx.y * 8, t = blockIdx.z, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t), e0 = c * ACS;
    if (e0 >= n) return;
    const int ne = n - e0 < ACS ? n - e0 : ACS;
    __shared__ float qs[8][HD];
    for (int i = threadIdx.x; i < 8 * HD; i += blockDim.x) {
        float v = q[((size_t)t * NH + h0) * HD + i];
        qs[i / HD][i % HD] = qf16 ? f16r(v) : v;
    }
    __syncthreads();
    const float scale = 1.f / sqrtf((float)HD);
    const int eg = threadIdx.x >> 3, l = threadIdx.x & 7, ng = blockDim.x >> 3;   /* 32 entry groups */
    for (int b0 = 0; b0 < ne; b0 += 4 * ng) {
        const float* k[4]; bool live[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int el = b0 + eg + j * ng;
            live[j] = el < ne;
            k[j] = attn_ent(e0 + (live[j] ? el : 0), p, W, win, topk, t, ckv);
        }
        float a[4][8];
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int h = 0; h < 8; h++) a[j][h] = 0.f;
        for (int d = 0; d < HD; d += 8) {
            float kv[4];
#pragma unroll
            for (int j = 0; j < 4; j++) kv[j] = k[j][d + l];
#pragma unroll
            for (int h = 0; h < 8; h++) {
                const float qv = qs[h][d + l];
#pragma unroll
                for (int j = 0; j < 4; j++) a[j][h] = __fmaf_rn(qv, kv[j], a[j][h]);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int h = 0; h < 8; h++) {
                float v = red8l(a[j][h]);
                if (live[j] && l == 0) S[((size_t)t * NH + h0 + h) * MAXE + e0 + b0 + eg + j * ng] = v * scale;
            }
    }
}

/* split output: block per (chunk, group of 8 heads, token), thread per dim; ap[t][h][c][d] = sum_e p[h][e] * k_e[d] */
__global__ void __launch_bounds__(HD) k_att_split_out8(const float* S, const float* win, int W, const int* ctl, int ratio,
        const int* topk, const int* ntop, const float* ckv, float* ap) {
    const int pos = ctl[0];
    const int c = blockIdx.x, h0 = blockIdx.y * 8, t = blockIdx.z, d = threadIdx.x, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t), e0 = c * ACS;
    if (e0 >= n) return;
    const int ne = n - e0 < ACS ? n - e0 : ACS;
    __shared__ float ps[8][ACS];
    for (int i = threadIdx.x; i < 8 * ACS; i += blockDim.x)
        ps[i / ACS][i % ACS] = (i % ACS) < ne ? S[((size_t)t * NH + h0 + i / ACS) * MAXE + e0 + i % ACS] : 0.f;
    __syncthreads();
    float o[8];
#pragma unroll
    for (int h = 0; h < 8; h++) o[h] = 0.f;
    for (int e = 0; e < ne; e++) {
        const float kv = attn_ent(e0 + e, p, W, win, topk, t, ckv)[d];
#pragma unroll
        for (int h = 0; h < 8; h++) o[h] = __fmaf_rn(ps[h][e], kv, o[h]);
    }
#pragma unroll
    for (int h = 0; h < 8; h++) ap[(((size_t)t * NH + h0 + h) * ANCH + c) * HD + d] = o[h];
}
/* the split output for prompt chunks: 256 threads, dims d and d + 256 per thread (a probability from shared memory
   feeds 2 FMAs); per (head, dim) the entry order is k_att_split_out8's (the same bits) */
__global__ void __launch_bounds__(HD / 2) k_att_split_out8x2(const float* S, const float* win, int W, const int* ctl, int ratio,
        const int* topk, const int* ntop, const float* ckv, float* ap) {
    const int pos = ctl[0];
    const int c = blockIdx.x, h0 = blockIdx.y * 8, t = blockIdx.z, d = threadIdx.x, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t), e0 = c * ACS;
    if (e0 >= n) return;
    const int ne = n - e0 < ACS ? n - e0 : ACS;
    __shared__ float ps[8][ACS];
    for (int i = threadIdx.x; i < 8 * ACS; i += blockDim.x)
        ps[i / ACS][i % ACS] = (i % ACS) < ne ? S[((size_t)t * NH + h0 + i / ACS) * MAXE + e0 + i % ACS] : 0.f;
    __syncthreads();
    float o[8], o2[8];
#pragma unroll
    for (int h = 0; h < 8; h++) o[h] = o2[h] = 0.f;
    for (int e = 0; e < ne; e++) {
        const float* kr = attn_ent(e0 + e, p, W, win, topk, t, ckv);
        const float kv = kr[d], kv2 = kr[d + HD / 2];
#pragma unroll
        for (int h = 0; h < 8; h++) { o[h] = __fmaf_rn(ps[h][e], kv, o[h]); o2[h] = __fmaf_rn(ps[h][e], kv2, o2[h]); }
    }
#pragma unroll
    for (int h = 0; h < 8; h++) {
        ap[(((size_t)t * NH + h0 + h) * ANCH + c) * HD + d] = o[h];
        ap[(((size_t)t * NH + h0 + h) * ANCH + c) * HD + d + HD / 2] = o2[h];
    }
}
/* llama.cpp-mode output for prompt chunks: dims d and d + 256 per thread, the same per-element sequence */
__global__ void __launch_bounds__(HD / 2) k_att_fa_out8x2(const float* S, const float* MS, const float* den, const float* win,
        int W, const int* ctl, int ratio, const int* topk, const int* ntop, const float* ckv, float* o) {
    const int pos = ctl[0];
    const int h0 = blockIdx.x * 8, t = blockIdx.y, d = threadIdx.x, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t);
    float acc[8], acc2[8];
#pragma unroll
    for (int h = 0; h < 8; h++) acc[h] = acc2[h] = 0.f;
    for (int e = 0; e < n; e++) {
        const float* kr = attn_ent(e, p, W, win, topk, t, ckv);
        const float kv = kr[d], kv2 = kr[d + HD / 2];
#pragma unroll
        for (int h = 0; h < 8; h++) {
            const size_t ix = ((size_t)t * NH + h0 + h) * MAXE + e;
            const float ms = MS[ix], sv = S[ix];
            if (ms != 1.f) { acc[h] = f16r(acc[h] * ms); acc2[h] = f16r(acc2[h] * ms); }
            acc[h] = f16r(acc[h] + sv * kv); acc2[h] = f16r(acc2[h] + sv * kv2);
        }
    }
#pragma unroll
    for (int h = 0; h < 8; h++) {
        const float dn = den[(size_t)t * NH + h0 + h];
        o[((size_t)t * NH + h0 + h) * HD + d] = acc[h] / dn;
        o[((size_t)t * NH + h0 + h) * HD + d + HD / 2] = acc2[h] / dn;
    }
}

/* llama.cpp-mode output: block per (group of 8 heads, token), thread per dim, all entries in order */
__global__ void __launch_bounds__(HD) k_att_fa_out8(const float* S, const float* MS, const float* den, const float* win,
        int W, const int* ctl, int ratio, const int* topk, const int* ntop, const float* ckv, float* o) {
    const int pos = ctl[0];
    const int h0 = blockIdx.x * 8, t = blockIdx.y, d = threadIdx.x, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t);
    float acc[8];
#pragma unroll
    for (int h = 0; h < 8; h++) acc[h] = 0.f;
    for (int e = 0; e < n; e++) {
        const float kv = attn_ent(e, p, W, win, topk, t, ckv)[d];
#pragma unroll
        for (int h = 0; h < 8; h++) {
            const size_t ix = ((size_t)t * NH + h0 + h) * MAXE + e;
            const float ms = MS[ix];
            if (ms != 1.f) acc[h] = f16r(acc[h] * ms);
            acc[h] = f16r(acc[h] + S[ix] * kv);
        }
    }
#pragma unroll
    for (int h = 0; h < 8; h++) o[((size_t)t * NH + h0 + h) * HD + d] = acc[h] / den[(size_t)t * NH + h0 + h];
}

/* e^x for x <= 0 as exp256 in dsv41_cpu.c (Cephes polynomial), the same operations */
__device__ __forceinline__ float exp_poly(float x) {
    x = fmaxf(x, -87.f);
    float n = floorf(x * 1.44269504088896341f + 0.5f);
    x = x - n * 0.693359375f;
    x = x - n * -2.12194440e-4f;
    float y = 1.9875691500e-4f;
    y = y * x + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = ((y * x) * x + x) + 1.f;
    return y * __int_as_float(((int)n + 127) << 23);
}

/* Split attention (reference numerics), step 1: per (token, head, chunk of ACS entries): max, exp (polynomial in
   groups of 8 from the chunk start, expf for the tail), sum; probabilities in place of the scores */
__global__ void k_att_split_soft(float* S, const int* ctl, int ratio, const int* ntop, float* am, float* as) {
    const int pos = ctl[0];
    int t = blockIdx.y, h = blockIdx.x, c = threadIdx.x, p = pos + t;   /* threads = ANCH */
    int n = attn_n(p, ratio, ntop, t), nch = (n + ACS - 1) / ACS;
    if (c >= nch) return;
    float* sc = S + ((size_t)t * NH + h) * MAXE + c * ACS;
    int ne = n - c * ACS < ACS ? n - c * ACS : ACS;
    float mx = -1e30f;
    for (int j = 0; j < ne; j++) mx = fmaxf(mx, sc[j]);
    int j = 0, nv = ne & ~7;            /* the CPU's 8-wide vectors from the chunk start, then the scalar tail */
    for (; j < nv; j++) sc[j] = exp_poly(sc[j] - mx);
    for (; j < ne; j++) sc[j] = d_expf(sc[j] - mx);
    float sum = 0;
    for (j = 0; j < ne; j++) sum += sc[j];
    size_t pi = ((size_t)t * NH + h) * ANCH + c;
    am[pi] = mx;
    as[pi] = sum;
}
/* step 2: unnormalized outputs per chunk, ap[t][h][c][d] = sum_e p_e * k_e[d] (fma chain over e) */
__global__ void k_att_split_out(const float* S, const float* win, int W, const int* ctl, int ratio, const int* topk,
                                const int* ntop, const float* ckv, float* ap) {
    const int pos = ctl[0];
    int t = blockIdx.z, h = blockIdx.y, c = blockIdx.x, d = threadIdx.x, p = pos + t;   /* HD threads */
    int n = attn_n(p, ratio, ntop, t), nch = (n + ACS - 1) / ACS;
    if (c >= nch) return;
    int ne = n - c * ACS < ACS ? n - c * ACS : ACS;
    const float* sc = S + ((size_t)t * NH + h) * MAXE + c * ACS;
    float o = 0.f;
    for (int e = 0; e < ne; e++) o = __fmaf_rn(sc[e], attn_ent(c * ACS + e, p, W, win, topk, t, ckv)[d], o);
    ap[(((size_t)t * NH + h) * ANCH + c) * HD + d] = o;
}
/* step 3: merge the chunks of each head with the sink, into o[t][h] */
__global__ void k_att_combine(const float* ap, const float* am, const float* as, const float* sinks, const int* ctl, int ratio,
                              const int* ntop, float* o) {
    const int pos = ctl[0];
    int t = blockIdx.y, h = blockIdx.x, d = threadIdx.x, p = pos + t;
    int n = attn_n(p, ratio, ntop, t), nch = (n + ACS - 1) / ACS;
    size_t pi = ((size_t)t * NH + h) * ANCH;
    __shared__ float wS[ANCH];
    if (d == 0) {
        float mx = am[pi];
        for (int c = 1; c < nch; c++) mx = fmaxf(mx, am[pi + c]);
        float w[ANCH], S = 0;
        for (int c = 0; c < nch; c++) { w[c] = d_expf(am[pi + c] - mx); S += as[pi + c] * w[c]; }
        S += d_expf(sinks[h] - mx);
        for (int c = 0; c < nch; c++) wS[c] = w[c] / S;
    }
    __syncthreads();
    float v = wS[0] * ap[pi * HD + d];
    for (int c = 1; c < nch; c++) v = v + wS[c] * ap[(pi + c) * HD + d];
    o[((size_t)t * NH + h) * HD + d] = v;
}

/* Attention of up to KINDI_ATT_WIDE tokens (default 8; 0 off) with more thread blocks. Every score, probability and output
   element is computed by the same operation sequence as k_att_scores8 / k_att_split_soft / k_att_split_out8 (the same
   bits); only the work per thread block is smaller. */
#define ACS1 32                   /* entries per thread block of the scores */
__global__ void __launch_bounds__(256) k_att_scores8w(const float* q, int qf16, const float* win, int W, const int* ctl,
        int ratio, const int* topk, const int* ntop, const float* ckv, float* S) {
    const int pos = ctl[0];
    const int h0 = blockIdx.y * 8, t = blockIdx.z, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t), e0 = blockIdx.x * ACS1;
    if (e0 >= n) return;
    __shared__ float qs[8][HD];
    for (int i = threadIdx.x; i < 8 * HD; i += blockDim.x) {
        float v = q[((size_t)t * NH + h0) * HD + i];
        qs[i / HD][i % HD] = qf16 ? f16r(v) : v;
    }
    __syncthreads();
    const float scale = 1.f / sqrtf((float)HD);
    const int el = threadIdx.x >> 3, l = threadIdx.x & 7;      /* 32 entries x 8 lanes */
    const bool live = e0 + el < n;
    const float* k = attn_ent(live ? e0 + el : e0, p, W, win, topk, t, ckv);
    float a[8];
#pragma unroll
    for (int h = 0; h < 8; h++) a[h] = 0.f;
#pragma unroll 4
    for (int d = 0; d < HD; d += 8) {
        const float kv = k[d + l];
#pragma unroll
        for (int h = 0; h < 8; h++) a[h] = __fmaf_rn(qs[h][d + l], kv, a[h]);
    }
#pragma unroll
    for (int h = 0; h < 8; h++) {
        float v = red8l(a[h]);
        if (live && l == 0) S[((size_t)t * NH + h0 + h) * MAXE + e0 + el] = v * scale;
    }
}
/* the split softmax with a warp per (token, head, chunk): max and exps in parallel (each element's own operation),
   the sum in entry order by lane 0 */
__global__ void k_att_split_softw(float* S, const int* ctl, int ratio, const int* ntop, float* am, float* as) {
    const int pos = ctl[0];
    const int t = blockIdx.y, h = blockIdx.x, c = threadIdx.x >> 5, lane = threadIdx.x & 31, p = pos + t;
    const int n = attn_n(p, ratio, ntop, t), nch = (n + ACS - 1) / ACS;
    if (c >= nch) return;
    float* sc = S + ((size_t)t * NH + h) * MAXE + c * ACS;
    const int ne = n - c * ACS < ACS ? n - c * ACS : ACS;
    float mx = -1e30f;
    for (int j = lane; j < ne; j += 32) mx = fmaxf(mx, sc[j]);
    mx = warp_max(mx);
    const int nv = ne & ~7;
    for (int j = lane; j < ne; j += 32) sc[j] = j < nv ? exp_poly(sc[j] - mx) : d_expf(sc[j] - mx);
    __syncwarp();
    if (lane == 0) {
        float sum = 0;
        for (int j = 0; j < ne; j++) sum += sc[j];
        size_t pi = ((size_t)t * NH + h) * ANCH + c;
        am[pi] = mx;
        as[pi] = sum;
    }
}
/* the split output with the dims over ATT_DS thread blocks: block (chunk, group of 8 heads, token * ATT_DS + part),
   HD / ATT_DS threads; per (head, dim) the fma chain over the chunk's entries in order */
#define ATT_DS 4
__global__ void __launch_bounds__(HD / ATT_DS) k_att_split_out8w(const float* S, const float* win, int W, const int* ctl,
        int ratio, const int* topk, const int* ntop, const float* ckv, float* ap) {
    const int pos = ctl[0];
    const int c = blockIdx.x, h0 = blockIdx.y * 8, t = blockIdx.z / ATT_DS, p = pos + t;
    const int d = (blockIdx.z % ATT_DS) * (HD / ATT_DS) + threadIdx.x;
    const int n = attn_n(p, ratio, ntop, t), e0 = c * ACS;
    if (e0 >= n) return;
    const int ne = n - e0 < ACS ? n - e0 : ACS;
    __shared__ float ps[8][ACS];
    __shared__ const float* kp[ACS];
    for (int i = threadIdx.x; i < 8 * ACS; i += blockDim.x)
        ps[i / ACS][i % ACS] = (i % ACS) < ne ? S[((size_t)t * NH + h0 + i / ACS) * MAXE + e0 + i % ACS] : 0.f;
    for (int e = threadIdx.x; e < ne; e += blockDim.x) kp[e] = attn_ent(e0 + e, p, W, win, topk, t, ckv);
    __syncthreads();
    float o[8];
#pragma unroll
    for (int h = 0; h < 8; h++) o[h] = 0.f;
    int e = 0;
    for (; e + 4 <= ne; e += 4) {
        const float k0 = kp[e][d], k1 = kp[e + 1][d], k2 = kp[e + 2][d], k3 = kp[e + 3][d];
#pragma unroll
        for (int h = 0; h < 8; h++) {
            o[h] = __fmaf_rn(ps[h][e], k0, o[h]);
            o[h] = __fmaf_rn(ps[h][e + 1], k1, o[h]);
            o[h] = __fmaf_rn(ps[h][e + 2], k2, o[h]);
            o[h] = __fmaf_rn(ps[h][e + 3], k3, o[h]);
        }
    }
    for (; e < ne; e++) {
        const float kv = kp[e][d];
#pragma unroll
        for (int h = 0; h < 8; h++) o[h] = __fmaf_rn(ps[h][e], kv, o[h]);
    }
#pragma unroll
    for (int h = 0; h < 8; h++) ap[(((size_t)t * NH + h0 + h) * ANCH + c) * HD + d] = o[h];
}
static int att_wide(void) {
    static int v = -1;
    if (v < 0) v = getenv("KINDI_ATT_WIDE") ? atoi(getenv("KINDI_ATT_WIDE")) : 8;
    return v;
}

/* llama.cpp flash-attention numerics (flag 128): online softmax in entry order with the output accumulator rounded
   to fp16 after every rescale and every added entry. Step 1 (thread per (t, h)): running max; per entry the
   rescale factor ms and weight vs; S. Step 2 (thread per (t, h, d)): the accumulator. */
__global__ void k_att_fa_soft(float* S, float* MS, const int* ctl, int ratio, const int* ntop, const float* sinks, float* den) {
    const int pos = ctl[0];
    int t = blockIdx.y, h = blockIdx.x * blockDim.x + threadIdx.x, p = pos + t;
    if (h >= NH) return;
    int n = attn_n(p, ratio, ntop, t);
    float* sc = S + ((size_t)t * NH + h) * MAXE;
    float* ms = MS + ((size_t)t * NH + h) * MAXE;
    float mx = -INFINITY, s2 = 0.f;
    for (int e = 0; e < n; e++) {
        float s = sc[e], m = 1.f, v = 1.f;
        if (s > mx) { m = mx == -INFINITY ? 0.f : d_expf(mx - s); mx = s; }
        else v = d_expf(s - mx);
        ms[e] = m; sc[e] = v;
        s2 = s2 * m + v;
    }
    den[(size_t)t * NH + h] = s2 + d_expf(sinks[h] - mx);
}
__global__ void k_att_fa_out(const float* S, const float* MS, const float* den, const float* win, int W, const int* ctl,
                             int ratio, const int* topk, const int* ntop, const float* ckv, float* o) {
    const int pos = ctl[0];
    int t = blockIdx.y, h = blockIdx.x, d = threadIdx.x, p = pos + t;
    int n = attn_n(p, ratio, ntop, t);
    const float* vs = S + ((size_t)t * NH + h) * MAXE;
    const float* ms = MS + ((size_t)t * NH + h) * MAXE;
    float acc = 0.f;
    for (int e = 0; e < n; e++) {
        if (ms[e] != 1.f) acc = f16r(acc * ms[e]);
        acc = f16r(acc + vs[e] * attn_ent(e, p, W, win, topk, t, ckv)[d]);
    }
    o[((size_t)t * NH + h) * HD + d] = acc / den[(size_t)t * NH + h];
}

/* attention output: remove the query rotation (per head, inverse RoPE) — done with k_rope inverse=1 — then FP8
   rounding per group of 8 heads (32-blocks are inside a group, so per row of NH * HD is the same) */

/* the expert inputs of layer l are on the host: write (token sequence * 64 + l) to the host-mapped flag (after the
   copies in stream order); the host spins on it. Works inside CUDA graphs (an event inside a graph does not). */
__global__ void k_signal(volatile int* flag, const int* ctl, int l) {
    __threadfence_system();
    *flag = ctl[1 + NL] * 64 + l;
    __threadfence_system();
}

/* the expert inputs of a small forward straight into the mapped host buffers */
__global__ void k_pack_out(const float* x, const int* sel, const float* wsel, int T, float* hx, int* hsel, float* hwsel) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < T * D; i += gridDim.x * blockDim.x) hx[i] = x[i];
    if (blockIdx.x == 0 && threadIdx.x < T * NS) { hsel[threadIdx.x] = sel[threadIdx.x]; hwsel[threadIdx.x] = wsel[threadIdx.x]; }
}

/* ---------------------------------------------------------------- router and shared expert */
/* block per token (NE threads): sqrt(softplus) scores, biased top-6 (ties: the lowest expert) by six block argmax
   rounds, weights normalized x 1.5 (thread 0, in selection order) */
__global__ void k_route(const float* glog, const float* bias, int T, int* sel, float* wsel) {
    int t = blockIdx.x, e = threadIdx.x;
    __shared__ float bv[NE / 32];
    __shared__ int bi[NE / 32];
    __shared__ float ssel[NS];
    __shared__ int isel[NS];
    float v = glog[(size_t)t * NE + e];
    float sp = v > 20.f ? v : d_log1pf(d_expf(v));
    float se = sqrtf(sp), be = se + bias[e];
    bool used = false;
    for (int k = 0; k < NS; k++) {
        float mv = used ? -INFINITY : be; int mi = used ? 0x7fffffff : e;
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, mv, o); int oi = __shfl_xor_sync(0xffffffffu, mi, o);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if ((e & 31) == 0) { bv[e >> 5] = mv; bi[e >> 5] = mi; }
        __syncthreads();
        if (e == 0) {
            float m2 = bv[0]; int i2 = bi[0];
            for (int w = 1; w < NE / 32; w++) if (bv[w] > m2 || (bv[w] == m2 && bi[w] < i2)) { m2 = bv[w]; i2 = bi[w]; }
            isel[k] = i2;
        }
        __syncthreads();
        if (isel[k] == e) { used = true; ssel[k] = se; }
        __syncthreads();
    }
    if (e == 0) {
        int* sl = sel + t * NS;
        float* w = wsel + t * NS;
        float sum = 0;
        for (int k = 0; k < NS; k++) { sl[k] = isel[k]; w[k] = ssel[k]; sum += w[k]; }
        for (int k = 0; k < NS; k++) w[k] = w[k] / (sum + 1e-20f) * 1.5f;
    }
}

/* shared expert SwiGLU with the clamp, in place in sg */
__global__ void k_swiglu(float* sg, const float* su, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float gt = fminf(sg[i], 10.f), up = fminf(fmaxf(su[i], -10.f), 10.f);
        sg[i] = gt / (1.f + d_expf(-gt)) * up;
    }
}


/* ---------------------------------------------------------------- hot experts in VRAM (G4) */
/* A static set of routed experts lives on the card in MXFP4 (the GGUF codes and E8M0 scales, re-laid out in groups
   of 16 rows: codes [group][block][16 rows][16 bytes], scales [group][block][16 rows]). For single tokens and
   speculation batches the card computes the slots whose expert is cached, the host computes the others
   (moe_forward_parts) and sends their row values; one kernel then adds w * value over each token's slots in order.
   The cached slots use moe_cpu.c's arithmetic per lane (codes -> doubled int8 weights, 8 int32 lanes of 4 products
   per block, fma with scale/2 * activation scale, hsum8), so the result does not depend on which experts are cached
   (up to expf in the SwiGLU, double-rounded here). */
#define MAXHOT 1024
typedef struct { const uint8_t *gq, *gs, *uq, *us, *dq, *ds; } DHot;
__constant__ float c_e8m0_half[256];
__constant__ int8_t c_fp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/* 4 FP4 codes of lane l (weights 4l .. 4l+3 of the block) -> 4 doubled int8 weights packed in an int */
__device__ __forceinline__ int fp4_lane(const uint8_t* codes, int l) {
    uint32_t v = *(const uint32_t*)(codes + (l & 3) * 4);
    if (l >= 4) v >>= 4;
    int r = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) r |= ((int)(uint8_t)c_fp4x2[(v >> (8 * i)) & 15]) << (8 * i);
    return r;
}

/* which: 0 gate, 1 up (input x [T][K], output gu[t][s][0 .. F) or [F .. 2F)), 2 down (input hq [t][s][F], output
   vg[t][s][K]). Grid (row groups, NS, T); a block returns at once when the slot's expert is not cached. */
__global__ void __launch_bounds__(128) k_hot_mx(int which, const int* hotmap, const int* sel, const DHot* hot,
        const int8_t* xq, const float* xd, const int8_t* hq, const float* hd, float* gu, float* vg) {
    const int s = blockIdx.y, t = blockIdx.z;
    const int hi = hotmap[sel[t * NS + s]];
    if (hi < 0) return;
    const int rows = which == 2 ? D : FF, cols = which == 2 ? FF : D, nb = cols / QK;
    const uint8_t* W = which == 0 ? hot[hi].gq : which == 1 ? hot[hi].uq : hot[hi].dq;
    const uint8_t* S = which == 0 ? hot[hi].gs : which == 1 ? hot[hi].us : hot[hi].ds;
    const int8_t* X = which == 2 ? hq + ((size_t)t * NS + s) * FF : xq + (size_t)t * D;
    const float* XD = which == 2 ? hd + ((size_t)t * NS + s) * (FF / QK) : xd + (size_t)t * (D / QK);
    const int g = blockIdx.x, rr = threadIdx.x >> 3, l = threadIdx.x & 7, row = g * 16 + rr;
    const uint8_t* w = W + ((size_t)g * nb * 16 + rr) * 16;
    const uint8_t* sc = S + (size_t)g * nb * 16 + rr;
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const int wv = fp4_lane(w + (size_t)b * 16 * 16, l);
        const int xv = *(const int*)(X + (size_t)b * QK + l * 4);
        const float sa = c_e8m0_half[sc[(size_t)b * 16]] * XD[b];
        acc = __fmaf_rn((float)__dp4a(wv, xv, 0), sa, acc);
    }
    acc = red8l(acc);
    if (l == 0 && row < rows) {
        if (which == 2) vg[((size_t)t * NS + s) * D + row] = acc;
        else gu[((size_t)t * NS + s) * 2 * FF + which * FF + row] = acc;
    }
}
/* h = silu(min(gate, 10)) * clamp(up, -10, 10), quantized (Q8, the llama.cpp rounding), for cached slots */
__global__ void k_hot_act(const int* hotmap, const int* sel, const float* gu, int8_t* hq, float* hd) {
    const int s = blockIdx.x, t = blockIdx.y;
    if (hotmap[sel[t * NS + s]] < 0) return;
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
/* mo[t][d] = sum over slots in order of w * value (cached slots from vg, the others from the host's vc) */
__global__ void k_moe_combine(const int* hotmap, const int* sel, const float* wsel, const float* vg, const float* vc,
                              float* mo) {
    const int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.f;
    for (int s = 0; s < NS; s++) {
        const size_t ix = ((size_t)t * NS + s) * D + d;
        const float v = wsel[t * NS + s] * (hotmap[sel[t * NS + s]] >= 0 ? vg[ix] : vc[ix]);
        acc = acc + v;
    }
    mo[(size_t)t * D + d] = acc;
}


/* ---------------------------------------------------------------- prompt chunks: experts split CPU / GPU (G7) */
/* For a prompt chunk (T > HOTT) the card computes the routed experts that carry the most pairs: the cached hot
   experts, and up to KINDI_SPLIT_STREAM others streamed over PCIe (raw GGUF bytes into a ring of VRAM slots,
   re-laid out on the card), while the host computes the remaining pairs. Each pair's row values come back to the
   host, which adds them in slot order (moe_combine). The card's per-expert GEMM sums each block in one int32 (tensor
   cores), not the CPU's 8 lanes, so a prompt chunk can differ from single steps in the last bits (as the dense GEMM). */
#define SPLIT_SLOTS 6             /* VRAM ring of streamed experts */

/* one GGUF MXFP4 matrix (rows x cols, 17-byte blocks) into the 16-row layout of the hot experts */
__global__ void k_relayout_mx(const uint8_t* raw, int rows, int cols, uint8_t* oq, uint8_t* os) {
    const int nb = cols / QK;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)rows * nb; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / nb, b = i % nb;
        const uint8_t* B = raw + i * 17;
        const size_t gi = ((r / 16) * nb + b) * 16 + r % 16;
        os[gi] = B[0];
        uint32_t w[4];
#pragma unroll
        for (int q = 0; q < 4; q++)
            w[q] = (uint32_t)B[1 + 4 * q] | ((uint32_t)B[2 + 4 * q] << 8) | ((uint32_t)B[3 + 4 * q] << 16) | ((uint32_t)B[4 + 4 * q] << 24);
        *(uint4*)(oq + gi * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

/* doubled int8 weights of 4 codes from 4 bytes: low nibbles (hi = 0) or high nibbles (hi = 1) */
__device__ __forceinline__ int fp4_word(uint32_t v, int hi) {
    if (hi) v >>= 4;
    int r = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) r |= ((int)(uint8_t)c_fp4x2[(v >> (8 * i)) & 15]) << (8 * i);
    return r;
}

/* Y[p][ycol + row] = sum over blocks of fma(float(int32 block sum), w_scale/2 * x_scale) for the pairs p of one
   expert matrix (16-row MXFP4 layout), tensor cores (m16n8k32: one block per step). X rows by xrow[p] (gather);
   a warp makes 16 rows x 64 pairs, a thread block 64 rows x 64 pairs. */
__global__ void __launch_bounds__(128) k_mx_mma(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, int rows,
        int cols, const int8_t* __restrict__ X, const float* __restrict__ XD, int xs, const int* __restrict__ xrow,
        int np, float* __restrict__ Y, int ys, int ycol) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int g = blockIdx.x * 4 + warp;
    if (g * 16 >= rows) return;
    const int p0 = blockIdx.y * MMA_TOK, nb = cols / QK;
    const uint8_t* wg = W + (size_t)g * nb * 16 * 16;
    const uint8_t* sg = S + (size_t)g * nb * 16;
    int xr[MMA_TOK / 8];
#pragma unroll
    for (int n = 0; n < MMA_TOK / 8; n++) { int p = p0 + n * 8 + gid; xr[n] = p < np ? xrow[p] : -1; }
    float acc[MMA_TOK / 8][4];
#pragma unroll
    for (int n = 0; n < MMA_TOK / 8; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    for (int b = 0; b < nb; b++) {
        const uint8_t* wb = wg + (size_t)b * 16 * 16;
        const uint32_t v0 = *(const uint32_t*)(wb + gid * 16 + tig * 4), v1 = *(const uint32_t*)(wb + (gid + 8) * 16 + tig * 4);
        const int a0 = fp4_word(v0, 0), a2 = fp4_word(v0, 1), a1 = fp4_word(v1, 0), a3 = fp4_word(v1, 1);
        const float w0 = c_e8m0_half[sg[(size_t)b * 16 + gid]], w1 = c_e8m0_half[sg[(size_t)b * 16 + gid + 8]];
#pragma unroll
        for (int n = 0; n < MMA_TOK / 8; n++) {
            int b0 = 0, b1 = 0;
            if (xr[n] >= 0) {
                const int8_t* xp = X + (size_t)xr[n] * xs + (size_t)b * QK;
                b0 = *(const int*)(xp + tig * 4); b1 = *(const int*)(xp + 16 + tig * 4);
            }
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                         "{%0,%1,%2,%3};\n"
                         : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            /* the C columns are pairs p0 + n * 8 + tig * 2 + {0, 1}: their x scales */
            const int pa = p0 + n * 8 + tig * 2, pb = pa + 1;
            const float xa = pa < np ? XD[(size_t)xrow[pa] * (xs / QK) + b] : 0.f;
            const float xb = pb < np ? XD[(size_t)xrow[pb] * (xs / QK) + b] : 0.f;
            acc[n][0] = __fmaf_rn((float)c0, w0 * xa, acc[n][0]);
            acc[n][1] = __fmaf_rn((float)c1, w0 * xb, acc[n][1]);
            acc[n][2] = __fmaf_rn((float)c2, w1 * xa, acc[n][2]);
            acc[n][3] = __fmaf_rn((float)c3, w1 * xb, acc[n][3]);
        }
    }
#pragma unroll
    for (int n = 0; n < MMA_TOK / 8; n++) {
        const int pa = p0 + n * 8 + tig * 2, r0 = g * 16 + gid, r1 = r0 + 8;
        if (pa < np) { if (r0 < rows) Y[(size_t)pa * ys + ycol + r0] = acc[n][0]; if (r1 < rows) Y[(size_t)pa * ys + ycol + r1] = acc[n][2]; }
        if (pa + 1 < np) { if (r0 < rows) Y[(size_t)(pa + 1) * ys + ycol + r0] = acc[n][1]; if (r1 < rows) Y[(size_t)(pa + 1) * ys + ycol + r1] = acc[n][3]; }
    }
}

/* SwiGLU (clamp 10) and Q8 quantization of np pairs: gu [p][2F] -> hq [p][F], hd */
__global__ void k_act_pairs(const float* gu, int np, int8_t* hq, float* hd) {
    const int p = blockIdx.x;
    if (p >= np) return;
    const float* g = gu + (size_t)p * 2 * FF;
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
        hq[(size_t)p * FF + b + lane] = (int8_t)(i > 127 ? 127 : (i < -128 ? -128 : i));
        if (lane == 0) hd[(size_t)p * (FF / QK) + b / QK] = f16r(am / 127.f);
    }
}

/* mean over the 4 copies of h (the attention input) for layer j = l - 37: mainh[t][j][d] */
__global__ void k_hmean(const float* h, float* mainh, int j) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    const float* ht = h + (size_t)t * HC * D;
    mainh[((size_t)t * 3 + j) * D + d] = ((ht[d] + ht[D + d]) + (ht[2 * D + d] + ht[3 * D + d])) * 0.25f;
}

/* ---------------------------------------------------------------- the model on the card */
typedef struct {
    int ratio, kv_src, idx_src, ks, is, eng;    /* ks / is: the kv source and index source layers */
    int m_qa, m_qb, m_kv, m_oa, m_ob, m_shg, m_shu, m_shd, m_gate, m_ckv, m_cg, m_ik, m_iqb, m_ip, m_ekv;  /* -1 none */
    const float *hc_attn_fn, *hc_ffn_fn;        /* host F32 [MIX][HC * D] */
    const float *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias, *hc_attn_base, *hc_attn_scale, *hc_ffn_base,
        *hc_ffn_scale, *cmp_norm, *idx_k_norm, *eng_gate;   /* host vectors (eng_gate [HC][D]) */
} DgLayerDesc;

typedef struct {
    DgLayerDesc h;
    float *hc_attn_fn, *hc_ffn_fn, *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias, *hc_attn_base,
        *hc_attn_scale, *hc_ffn_base, *hc_ffn_scale, *cmp_norm, *idx_k_norm, *eng_gate;
    float* win;                                 /* [W][HD] */
    float *ckv, *ik, *cst_kv, *cst_sc;          /* kv sources */
    int *topk, *ntop;                           /* index sources: [tmax][ITOPK], [tmax] */
} DLayer;

typedef int (*dg_moe_fn)(int l, int T, const float* x, const int* sel, const float* wsel, float* mo);
typedef int (*dg_parts_fn)(int l, int T, const float* x, const int* sel, const uint8_t* skip, float* parts);
#define HOTT 8                    /* the hot-expert path serves forwards of up to HOTT tokens (decode, verify) */
typedef int (*dg_combine_fn)(int T, const float* wsel, const float* parts, float* out);
typedef int (*dg_combine_alt_fn)(int T, const float* wsel, const float* parts, const int* row, const float* alt, float* out);

static struct {
    int cap, tmax, W, m_out, cands, scs;
    DLayer L[NL];
    float* out_norm;
    /* scratch [tmax][...] */
    float *h, *x, *part, *rinv, *pre, *apre, *post, *comb, *qa, *q, *kvw, *ck, *cg, *iq, *iw, *o, *oa, *ao, *glog, *sg,
        *su, *mo, *fo, *ekv, *lat, *latk, *sc, *bs, *S, *MS, *den, *ap, *am, *as;
    int8_t *xq, *qaq, *oq, *oaq, *suq, *latq, *lq;
    float *xd, *qad, *od, *oad, *sud, *latd, *ld;
    int *sel, *latg;
    float* wsel;
    uint8_t* cand;
    /* per-forward inputs (one upload) */
    uint8_t* din; uint8_t* hin; size_t incap;
    /* host side of the expert round trip (pinned) */
    float *hx, *hmo, *hwsel; int* hsel;
    float *hlog, *lg;
    int* ctl;                     /* device: [0] = pos of the forward, [1 + l] = latents of layer l (upload) */
    int graphing;                 /* 1 while a decode graph is captured: grids at their maxima */
    cudaGraphExec_t seg[NL + 1];  /* decode (T = 1) graphs: [embed + A0], [B(l-1) + A(l)], [B(39) + head] */
    int seg_flags, seg_ok;
    cudaEvent_t ev;
    volatile int *hflag; int* dflag;   /* host-mapped completion flag (k_signal) */
    float *dhx, *dhmo, *dhwsel, *dhparts; int* dhsel;   /* device views of the mapped hand-off buffers */
    int prof;                     /* KINDI_DSV_PROF=1: phase times of single-token forwards (no graphs) */
    cudaEvent_t pev[NL][12];
    double pt[12], hwait, hmoe; long pn;
    int seq;
    int cur_t;
    float* mainh;                 /* [tmax][3][D]: mean of the stream copies at the attention input of layers 37-39 */
    float* ekv_e[2];              /* [HOTT][(HC + 1) * D]: engram slot e's projection, made early (see layer_a) */
    int eng_l[2];                 /* the layer of engram slot e (-1 none) */
    cudaStream_t sb;              /* side stream of the independent branches (KINDI_DSV_BRANCH) */
    cudaEvent_t evf[4], evj[4];
    double* wsum;                 /* [tmax][32] warp sums of k_rms_part */
    /* hot experts */
    int nhot, hotmap_h[NL][NE];
    int* hotmap_d;                /* [NL][NE] */
    DHot* hot_d; DHot hot_h[MAXHOT];
    float *gu, *vg, *vc, *hparts; int8_t* hq; float* hd; uint8_t* hskip;
    dg_parts_fn parts;
    size_t hot_bytes;
    long hhit, hslot;             /* cached slots / all slots served by the hot path */
    /* prompt-chunk expert split (G7) */
    int split, nstream, pmax, d2h_early;
    int pf_hold;                  /* pipelined chunk pair: the first chunk keeps the prefetch set (the second follows) */
    const uint8_t *src_g[NL], *src_u[NL], *src_d[NL];   /* host GGUF expert tensors (page cache) */
    uint8_t *raw_d[2], *raw_h[2];                         /* PCIe staging: pinned host, device */
    uint8_t *slot_q[SPLIT_SLOTS], *slot_s[SPLIT_SLOTS];  /* re-laid streamed experts: gate, up, down back to back */
    cudaEvent_t raw_ev[2];
    cudaStream_t st2;
    int* xrow_d; int* xrow_h; int* ident_d;
    float *gup, *vp, *hpart_big, *vp_h; int8_t* hqp; float* hdp;
    dg_combine_fn combine;
    dg_combine_alt_fn combine_alt;
    double sp_cpu, sp_gpu, sp_wait; long sp_n, sp_gpairs, sp_pairs;
} G;

static float* dvec(const float* h, size_t n) {
    float* d = NULL;
    if (!h) return NULL;
    if (cudaMalloc(&d, n * 4) != cudaSuccess) return NULL;
    cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
    vram_used += n * 4;
    return d;
}
static void* dz(size_t n) {
    void* d = NULL;
    if (cudaMalloc(&d, n ? n : 4) != cudaSuccess) { fprintf(stderr, "dsv41_gpu: out of VRAM (%zu bytes)\n", n); return NULL; }
    cudaMemset(d, 0, n ? n : 4);
    vram_used += n;
    return d;
}

/* internal GEMV on device buffers (the same kernels as dg_gemv) */
static void gemv_d(int mat, const int8_t* xq, const float* xd, const float* xf, int xs, int seg_rows, int seg_cols,
                   int T, float* y, int ys) {
    const GMat* m = &GM[mat];
    int grid = (m->rows + 15) / 16;
    if (m->type == 8) q8_launch(m, xq, xd, xs, seg_rows, seg_cols, T, y, ys);
    else if (T > GEMM_MIN_T && (!getenv("KINDI_DSV_GEMM") || atoi(getenv("KINDI_DSV_GEMM")) != 0)) k_f32_gemm<<<dim3((m->rows + 63) / 64, (T + 63) / 64), 256, 0, st>>>(m->f, m->rows, m->cols, xf, xs, T, y, ys);
    else if (T >= 2 && T <= 8 && env_on("KINDI_DSV_F32T"))
        k_f32t<<<(m->rows + 3) / 4, dim3(32, T), 0, st>>>(m->f, m->rows, m->cols, xf, xs, y, ys);
    else { if (T == 1) k_f32<1><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, xf, xs, T, y, ys); else if (T > TTN) k_f32<8><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, xf, xs, T, y, ys); else k_f32<TTN><<<grid, 128, 0, st>>>(m->f, m->rows, m->cols, xf, xs, T, y, ys); }
}

static const char* PNAME[11] = {"engram", "hc_attn", "q/kv proj", "compress+index", "attention", "o proj",
                                 "hc_ffn", "router+gate", "shared exp", "expert hand-off", "end (add+post)"};
#define PMARK(l, k) do { if (G.prof && G.cur_t == 1 && !G.graphing) cudaEventRecord(G.pev[l][k], st); } while (0)

/* same-input small matrices: one launch (decode); prompt chunks go one by one through the GEMM */
static void gemv_multi(const int* mats, float* const* ys_ptr, const int* ys, int n, const int8_t* xq, const float* xd,
                       int xs, int T) {
    static int multi = -1;
    if (multi < 0) multi = getenv("KINDI_DSV_MULTI") && atoi(getenv("KINDI_DSV_MULTI")) != 0;   /* off: slower than separate launches */
    if (T > GEMM_MIN_T || n == 1 || !multi) {
        for (int i = 0; i < n; i++) gemv_d(mats[i], xq, xd, NULL, xs, 0, 0, T, ys_ptr[i], ys[i]);
        return;
    }
    QMulti M;
    M.n = n; M.first[0] = 0;
    for (int i = 0; i < n; i++) {
        const GMat* m = &GM[mats[i]];
        M.W[i] = m->q; M.S[i] = m->s; M.rows[i] = m->rows; M.Y[i] = ys_ptr[i]; M.ys[i] = ys[i];
        M.first[i + 1] = M.first[i] + (m->rows + 15) / 16;
    }
    { if (T == 1) k_q8_multi<1><<<M.first[n], 128, GM[mats[0]].cols + GM[mats[0]].cols / QK * 4, st>>>(M, GM[mats[0]].cols, xq, xd, xs, T); else k_q8_multi<TTN><<<M.first[n], 128, TTN * (GM[mats[0]].cols + GM[mats[0]].cols / QK * 4), st>>>(M, GM[mats[0]].cols, xq, xd, xs, T); }
}

static inline int ew_grid(size_t n) { size_t g = (n + 255) / 256; return (int)(g > 4096 ? 4096 : g); }

extern "C" {

int dg_model_init(int cap, int tmax, int m_out, const float* out_norm, const DgLayerDesc* desc) {
    /* the verify-batch GEMV may need more than 48 KB of shared memory (set before any graph capture) */
    CK(cudaFuncSetAttribute(k_q8ws<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    CK(cudaFuncSetAttribute(k_q8ws<TTN>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024));
    G.cap = cap; G.tmax = tmax; G.W = WIN + tmax; G.m_out = m_out;
    G.cands = cap / CBLK + 1; G.scs = cap + 8;
    G.out_norm = dvec(out_norm, D);
    for (int l = 0; l < NL; l++) {
        DLayer* L = &G.L[l];
        L->h = desc[l];
        const DgLayerDesc* d = &desc[l];
        L->hc_attn_fn = dvec(d->hc_attn_fn, (size_t)MIX * HC * D); L->hc_ffn_fn = dvec(d->hc_ffn_fn, (size_t)MIX * HC * D);
        L->sinks = dvec(d->sinks, NH); L->kv_norm = dvec(d->kv_norm, HD); L->q_norm = dvec(d->q_norm, QR);
        L->attn_norm = dvec(d->attn_norm, D); L->ffn_norm = dvec(d->ffn_norm, D); L->bias = dvec(d->bias, NE);
        L->hc_attn_base = dvec(d->hc_attn_base, MIX); L->hc_attn_scale = dvec(d->hc_attn_scale, 3);
        L->hc_ffn_base = dvec(d->hc_ffn_base, MIX); L->hc_ffn_scale = dvec(d->hc_ffn_scale, 3);
        L->cmp_norm = dvec(d->cmp_norm, HD); L->idx_k_norm = dvec(d->idx_k_norm, IHD);
        L->eng_gate = dvec(d->eng_gate, (size_t)HC * D);
        L->win = (float*)dz((size_t)G.W * HD * 4);
        if (d->kv_src) {
            size_t ng = (size_t)cap / d->ratio + 1;
            L->ckv = (float*)dz(ng * HD * 4); L->ik = (float*)dz(ng * IHD * 4);
            L->cst_kv = (float*)dz(CRING * HD * 4); L->cst_sc = (float*)dz(CRING * HD * 4);
        }
        if (d->idx_src) { L->topk = (int*)dz((size_t)tmax * ITOPK * 4); L->ntop = (int*)dz((size_t)tmax * 4); }
        if (!L->win) return -1;
    }
    size_t T = tmax;
    G.h = (float*)dz(T * HC * D * 4); G.x = (float*)dz(T * D * 4); G.part = (float*)dz(T * HKC * MIX * 4);
    G.rinv = (float*)dz(T * RPARTS * 8);
    G.pre = (float*)dz(T * HC * 4); G.apre = (float*)dz(T * HC * 4); G.post = (float*)dz(T * HC * 4);
    G.comb = (float*)dz(T * 16 * 4);
    G.qa = (float*)dz(T * QR * 4); G.q = (float*)dz(T * NH * HD * 4); G.kvw = (float*)dz(T * HD * 4);
    G.ck = (float*)dz(T * HD * 4); G.cg = (float*)dz(T * HD * 4); G.iq = (float*)dz(T * IH * IHD * 4);
    G.iw = (float*)dz(T * IH * 4); G.o = (float*)dz(T * NH * HD * 4); G.oa = (float*)dz(T * OG * OR * 4);
    G.ao = (float*)dz(T * D * 4); G.glog = (float*)dz(T * NE * 4); G.sg = (float*)dz(T * FF * 4);
    G.su = (float*)dz(T * FF * 4); G.mo = (float*)dz(T * D * 4); G.fo = (float*)dz(T * D * 4);
    G.ekv = (float*)dz(T * (HC + 1) * D * 4);
    CK(cudaStreamCreateWithFlags(&G.sb, cudaStreamNonBlocking));
    for (int i = 0; i < 4; i++) {
        CK(cudaEventCreateWithFlags(&G.evf[i], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&G.evj[i], cudaEventDisableTiming));
    }
    G.wsum = (double*)dz((size_t)tmax * 32 * 8);
    G.eng_l[0] = G.eng_l[1] = -1;
    for (int l = 0; l < NL; l++) if (desc[l].eng >= 0 && desc[l].eng < 2) G.eng_l[desc[l].eng] = l;
    for (int e = 0; e < 2; e++) G.ekv_e[e] = (float*)dz((size_t)HOTT * (HC + 1) * D * 4);
    G.lat = (float*)dz(T * HD * 4); G.latk = (float*)dz(T * IHD * 4);
    G.sc = (float*)dz(T * G.scs * 4); G.bs = (float*)dz(T * G.cands * 4);
    G.S = (float*)dz(T * NH * MAXE * 4); G.MS = (float*)dz(T * NH * MAXE * 4); G.den = (float*)dz(T * NH * 4);
    G.ap = (float*)dz(T * NH * ANCH * HD * 4); G.am = (float*)dz(T * NH * ANCH * 4); G.as = (float*)dz(T * NH * ANCH * 4);
    G.xq = (int8_t*)dz(T * D); G.xd = (float*)dz(T * D / QK * 4);
    G.qaq = (int8_t*)dz(T * QR); G.qad = (float*)dz(T * QR / QK * 4);
    G.oq = (int8_t*)dz(T * NH * HD); G.od = (float*)dz(T * NH * HD / QK * 4);
    G.oaq = (int8_t*)dz(T * OG * OR); G.oad = (float*)dz(T * OG * OR / QK * 4);
    G.suq = (int8_t*)dz(T * FF); G.sud = (float*)dz(T * FF / QK * 4);
    G.latq = (int8_t*)dz(T * HD); G.latd = (float*)dz(T * HD / QK * 4);
    G.lq = (int8_t*)dz(T * D); G.ld = (float*)dz(T * D / QK * 4);
    G.sel = (int*)dz(T * NS * 4); G.wsel = (float*)dz(T * NS * 4); G.latg = (int*)dz(T * 4 * NL);
    G.cand = (uint8_t*)dz(T * G.cands);
    G.lg = (float*)dz((size_t)8 * VOCAB * 4);
    G.mainh = (float*)dz(T * 3 * D * 4);
    /* inputs of one forward: cs [T][2 RD], emb [T][D], engram q8 [2][T][6144] + scales, latents cs/g per layer */
    G.incap = T * 2 * RD * 4 + T * D * 4 + 2 * T * ENG_COLS * ENG_HD * 5 + (size_t)NL * T * (RD + 1) * 4 + 4096;
    G.din = (uint8_t*)dz(G.incap);
    CK(cudaMallocHost(&G.hin, G.incap));
    /* the expert hand-off buffers are host memory mapped into the card: for single tokens and speculation batches
       the kernels write the expert inputs and read the host's results there directly (no copies to launch) */
    CK(cudaHostAlloc((void**)&G.hx, T * D * 4, cudaHostAllocMapped)); CK(cudaHostAlloc((void**)&G.hmo, T * D * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**)&G.hsel, T * NS * 4, cudaHostAllocMapped)); CK(cudaHostAlloc((void**)&G.hwsel, T * NS * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&G.dhx, G.hx, 0)); CK(cudaHostGetDevicePointer((void**)&G.dhmo, G.hmo, 0));
    CK(cudaHostGetDevicePointer((void**)&G.dhsel, G.hsel, 0)); CK(cudaHostGetDevicePointer((void**)&G.dhwsel, G.hwsel, 0));
    CK(cudaMallocHost(&G.hlog, (size_t)8 * VOCAB * 4));
    CK(cudaEventCreateWithFlags(&G.ev, cudaEventDisableTiming));
    G.prof = getenv("KINDI_DSV_PROF") && atoi(getenv("KINDI_DSV_PROF"));
    if (G.prof)
        for (int l = 0; l < NL; l++)
            for (int k = 0; k < 12; k++) CK(cudaEventCreate(&G.pev[l][k]));
    CK(cudaHostAlloc((void**)&G.hflag, 64, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&G.dflag, (void*)G.hflag, 0));
    *G.hflag = -1;
    if (!G.h || !G.cand || !G.din || !G.lg) return -1;
    G.split = getenv("KINDI_SPLIT") && atoi(getenv("KINDI_SPLIT")) && tmax > HOTT;
    if (G.split) {
        G.nstream = getenv("KINDI_SPLIT_STREAM") ? atoi(getenv("KINDI_SPLIT_STREAM")) : 12;   /* best of 12/14/16/24 at 2K */
        G.pmax = tmax * NS * 2 / 3;              /* at most 2/3 of a chunk's pairs on the card */
        G.d2h_early = getenv("KINDI_SPLIT_D2H_EARLY") ? atoi(getenv("KINDI_SPLIT_D2H_EARLY")) : 1;
        const size_t eb = (size_t)3 * FF * D / QK * 17;   /* one expert, GGUF bytes */
        for (int i = 0; i < 2; i++) {
            G.raw_d[i] = (uint8_t*)dz(eb);
            CK(cudaMallocHost(&G.raw_h[i], eb));
            CK(cudaEventCreateWithFlags(&G.raw_ev[i], cudaEventDisableTiming));
        }
        for (int i = 0; i < SPLIT_SLOTS; i++) { G.slot_q[i] = (uint8_t*)dz(eb / 17 * 16); G.slot_s[i] = (uint8_t*)dz(eb / 17); }
        CK(cudaStreamCreateWithFlags(&G.st2, cudaStreamNonBlocking));
        G.xrow_d = (int*)dz((size_t)G.pmax * 4);
        {
            int* id = (int*)malloc((size_t)G.pmax * 4);
            for (int i = 0; i < G.pmax; i++) id[i] = i;
            G.ident_d = (int*)dz((size_t)G.pmax * 4);
            CK(cudaMemcpy(G.ident_d, id, (size_t)G.pmax * 4, cudaMemcpyHostToDevice));
            free(id);
        }
        CK(cudaMallocHost(&G.xrow_h, (size_t)G.pmax * 4));
        G.gup = (float*)dz((size_t)G.pmax * 2 * FF * 4); G.vp = (float*)dz((size_t)G.pmax * D * 4);
        G.hqp = (int8_t*)dz((size_t)G.pmax * FF); G.hdp = (float*)dz((size_t)G.pmax * FF / QK * 4);
        CK(cudaMallocHost(&G.hpart_big, (size_t)tmax * NS * D * 4));
        CK(cudaMallocHost(&G.vp_h, (size_t)G.pmax * D * 4));
        if (!G.vp || !G.gup) return -1;
    }
    G.hotmap_d = (int*)dz((size_t)NL * NE * 4);
    for (int l = 0; l < NL; l++) for (int e = 0; e < NE; e++) G.hotmap_h[l][e] = -1;
    CK(cudaMemcpy(G.hotmap_d, G.hotmap_h, sizeof G.hotmap_h, cudaMemcpyHostToDevice));
    G.hot_d = (DHot*)dz(sizeof(DHot) * MAXHOT);
    G.gu = (float*)dz((size_t)HOTT * NS * 2 * FF * 4); G.vg = (float*)dz((size_t)HOTT * NS * D * 4);
    G.vc = (float*)dz((size_t)HOTT * NS * D * 4);
    G.hq = (int8_t*)dz((size_t)HOTT * NS * FF); G.hd = (float*)dz((size_t)HOTT * NS * FF / QK * 4);
    CK(cudaHostAlloc((void**)&G.hparts, (size_t)HOTT * NS * D * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&G.dhparts, G.hparts, 0));
    CK(cudaMallocHost(&G.hskip, (size_t)HOTT * NS));
    {
        float tab[256];
        for (int e = 0; e < 256; e++) tab[e] = ldexpf(1.f, e - 128);
        CK(cudaMemcpyToSymbol(c_e8m0_half, tab, sizeof tab));
    }
    return 0;
}

/* one routed expert into the VRAM cache (GGUF MXFP4 rows: gate, up [FF][D / 32] blocks of 17 bytes, down [D][FF / 32]);
   returns 0, or -1 when fewer than reserve bytes would stay free */
static int hot_mat(const uint8_t* src, int rows, int cols, const uint8_t** dq, const uint8_t** ds) {
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
    G.hot_bytes += nq + ns; vram_used += nq + ns;
    return 0;
}

extern "C" {

int dg_hot_add(int l, int e, const uint8_t* gate, const uint8_t* up, const uint8_t* down, size_t reserve) {
    if (G.nhot >= MAXHOT || G.hotmap_h[l][e] >= 0) return -1;
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    size_t need = (size_t)3 * FF * D / QK * 17;
    if (fr < need + reserve) return -1;
    DHot* h = &G.hot_h[G.nhot];
    if (hot_mat(gate, FF, D, &h->gq, &h->gs) || hot_mat(up, FF, D, &h->uq, &h->us) || hot_mat(down, D, FF, &h->dq, &h->ds))
        return -1;
    G.hotmap_h[l][e] = G.nhot++;
    return 0;
}
int dg_hot_commit(dg_parts_fn fn) {
    G.parts = fn;
    CK(cudaMemcpy(G.hot_d, G.hot_h, sizeof(DHot) * MAXHOT, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G.hotmap_d, G.hotmap_h, sizeof G.hotmap_h, cudaMemcpyHostToDevice));
    G.seg_ok = 0;                 /* recapture the decode graphs */
    return G.nhot;
}
size_t dg_hot_bytes(void) { return G.hot_bytes; }
void dg_set_expert_src(int l, const uint8_t* gate, const uint8_t* up, const uint8_t* down) {
    G.src_g[l] = gate; G.src_u[l] = up; G.src_d[l] = down;
}
void dg_set_combine(dg_combine_fn fn, dg_parts_fn pf) { G.combine = fn; if (!G.parts) G.parts = pf; }
void dg_set_combine_alt(dg_combine_alt_fn fn) { G.combine_alt = fn; }

}  /* extern "C" */

}  /* extern "C" */

/* KINDI_DSV_BRANCH (default 1): independent kernels of a layer on a side stream (a fork and a join by events; in a
   graph capture they become parallel branches), so small latency-bound kernels overlap. The same kernels on the same
   inputs, and no two branches write the same buffer: the bits do not change. */
static int branch_on(void) {
    static int v = -1;
    if (v < 0) v = !getenv("KINDI_DSV_BRANCH") || atoi(getenv("KINDI_DSV_BRANCH")) != 0;
    return v;
}
static void br_fork(int i) { cudaEventRecord(G.evf[i], st); cudaStreamWaitEvent(G.sb, G.evf[i], 0); }
static void br_join(int i) { cudaEventRecord(G.evj[i], G.sb); cudaStreamWaitEvent(st, G.evj[i], 0); }
/* run the launches of the statement on the side stream (all launch helpers use the global st) */
#define ON_SIDE(br, ...) do { cudaStream_t keep_ = st; if (br) st = G.sb; __VA_ARGS__; st = keep_; } while (0)

static void br_mark(int i) { cudaEventRecord(G.evj[i], G.sb); }
static void br_wait(int i) { cudaStreamWaitEvent(st, G.evj[i], 0); }
/* KINDI_DSV_RMSQ (default 1): k_rms_part + k_rmsq instead of k_rmsnorm + k_quant (the same bits) */
static int rmsq_on(void) {
    static int v = -1;
    if (v < 0) v = !getenv("KINDI_DSV_RMSQ") || atoi(getenv("KINDI_DSV_RMSQ")) != 0;
    return v;
}
static void rms_quant(float* x, int n, const float* w, int T, int8_t* q, float* dq) {
    if (rmsq_on()) {
        k_rms_part<<<dim3(8, T), 128, 0, st>>>(x, n, n, G.wsum);
        k_rmsq<<<dim3((n + 255) / 256, T), 256, 0, st>>>(x, n, x, n, w, n, G.wsum, q, dq);
    } else {
        k_rmsnorm<<<T, 1024, 0, st>>>(x, n, x, n, w, n);
        k_quantm<<<dim3((n + 255) / 256, T), 256, 0, st>>>(x, n, n, q, dq);
    }
}
/* comb_ev >= 0 (with branches): the Sinkhorn mix (comb, used only by k_post) runs on the side stream and the caller
   waits for event comb_ev before k_post */
static void hc_in_d(const float* fn, const float* scale, const float* base, const float* pre_in, const float* norm_w,
                    float* pre, int T, int comb_ev = -1) {
    const int br = branch_on();
    if (br) br_fork(0);
    ON_SIDE(br, (k_rinv<<<dim3(T, RPARTS), 512, 0, st>>>(G.h, HC * D, (double*)G.rinv)));
    k_hcmix<<<dim3(T, MIX), 128, 0, st>>>(fn, G.h, T, G.part);
    if (br) br_join(0);
    if (br && comb_ev >= 0) {
        br_fork(comb_ev);
        ON_SIDE(br, (k_mixw<<<T, 32, 0, st>>>(G.part, T, scale, base, pre, G.post, G.comb, (const double*)G.rinv, 2)));
        br_mark(comb_ev);
        k_mixw<<<T, 32, 0, st>>>(G.part, T, scale, base, pre, G.post, G.comb, (const double*)G.rinv, 1);
    } else
        k_mixw<<<T, 32, 0, st>>>(G.part, T, scale, base, pre, G.post, G.comb, (const double*)G.rinv, 0);
    k_collapse<<<dim3(D / 256, T), 256, 0, st>>>(G.h, pre_in, G.x);
    rms_quant(G.x, D, norm_w, T, G.xq, G.xd);
}

static void attention_d(int l, int T, int pos, int flags, const float* cs, const float* cs_lat, const int* latg, int nl) {
    DLayer* L = &G.L[l];
    const DgLayerDesc* d = &L->h;
    const int br = branch_on();
    int csoff = d->ratio ? RD : 0;
    const DLayer *KS = d->ratio ? &G.L[d->ks] : NULL, *IS = d->ratio ? &G.L[d->is] : NULL;
    /* the kv path (window kv, compressor, indexer weights) beside the query path (q_a, norm, q_b, RoPE) */
    if (br) br_fork(1);
    ON_SIDE(br, {
        int mats[4], ys[4], n = 0;
        float* yp[4];
        mats[n] = d->m_kv; yp[n] = G.kvw; ys[n++] = HD;
        if (d->kv_src) {
            mats[n] = d->m_ckv; yp[n] = G.ck; ys[n++] = HD;
            if (d->ratio > 1) { mats[n] = d->m_cg; yp[n] = G.cg; ys[n++] = HD; }
        }
        if (d->idx_src) { mats[n] = d->m_ip; yp[n] = G.iw; ys[n++] = IH; }
        gemv_multi(mats, yp, ys, n, G.xq, G.xd, D, T);
        k_kv_win<<<T, HD, 0, st>>>(G.kvw, L->kv_norm, cs, csoff, L->win, G.W, G.ctl, flags);
        if (d->ratio && d->kv_src) {
            k_compress<<<1, HD, 0, st>>>(G.ck, G.cg, T, G.ctl, d->ratio, L->cst_kv, L->cst_sc, L->cmp_norm, G.lat, G.latq, G.latd);
            if (nl || G.graphing) {
                int n2 = G.graphing ? T : nl;     /* graphs: the most a chunk of T can complete; k_lat reads nl */
                gemv_d(d->m_ik, G.latq, G.latd, NULL, HD, 0, 0, n2, G.latk, IHD);
                k_lat<<<n2, HD, 0, st>>>(G.lat, G.latk, latg, cs_lat, L->idx_k_norm, L->ik, L->ckv, flags, G.ctl + 1 + l);
            }
        }
    });
    gemv_d(d->m_qa, G.xq, G.xd, NULL, D, 0, 0, T, G.qa, QR);
    rms_quant(G.qa, QR, L->q_norm, T, G.qaq, G.qad);
    gemv_d(d->m_qb, G.qaq, G.qad, NULL, QR, 0, 0, T, G.q, NH * HD);
    if (d->idx_src) gemv_d(d->m_iqb, G.qaq, G.qad, NULL, QR, 0, 0, T, G.iq, IH * IHD);
    PMARK(l, 3);
    k_rope<<<dim3(T, NH / 8), dim3(RD / 2, 8), 0, st>>>(G.q, NH, HD, cs, 2 * RD, csoff, 0);
    if (flags & F_F16_Q) k_round_f16<<<ew_grid((size_t)T * NH * HD), 256, 0, st>>>(G.q, (size_t)T * NH * HD);
    if (br) br_join(1);
    if (d->idx_src) k_iq_prep<<<T, IH * 32, 0, st>>>(G.iq, G.iw, cs, flags);
    if (d->ratio) {
        if (d->idx_src) {
            int r = d->ratio, nmax = G.graphing ? G.cap / r : (pos + T) / r;
            if (nmax > 0) {
                k_idx_score<<<dim3((nmax + 127) / 128, T), 128, 0, st>>>(G.iq, G.iw, KS->ik, G.ctl, r, l > CAND_SRC, G.cand,
                                                                         G.cands, G.sc, G.scs);
                if (l == CAND_SRC) {
                    int nbm = (nmax + CBLK - 1) / CBLK;
                    k_blockmax<<<dim3((nbm + 255) / 256, T), 256, 0, st>>>(G.sc, G.scs, G.ctl, r, G.bs, G.cands);
                    /* the block list goes to bs-sized scratch (G.ap is free at this point) */
                    k_topk<<<T, 1024, 0, st>>>(G.bs, G.cands, G.ctl, r, 1, CTOPK, (int*)G.ap, G.cands, NULL, G.cand, G.cands);
                }
            }
            k_topk<<<T, 1024, 0, st>>>(G.sc, G.scs, G.ctl, r, 0, ITOPK, L->topk, ITOPK, L->ntop, NULL, 0);
        }
    }
    PMARK(l, 4);
    const int* topk = IS ? IS->topk : NULL;
    const int* ntop = IS ? IS->ntop : NULL;
    const float* ckv = KS ? KS->ckv : NULL;
    int ratio = d->ratio;
    int nmaxe = G.graphing ? WIN + ITOPK : (pos + T < WIN ? pos + T : WIN) + (ratio ? ITOPK : 0);
    int fa = (flags & F_LLAMA_FA) != 0;
    const int wide = T <= att_wide();
    if (wide)
        k_att_scores8w<<<dim3((nmaxe + ACS1 - 1) / ACS1, NH / 8, T), 256, 0, st>>>(G.q, fa, L->win, G.W, G.ctl, ratio, topk,
                                                                                 ntop, ckv, G.S);
    else if (T > att_x_min())
        k_att_scores8x4<<<dim3((nmaxe + ACS - 1) / ACS, NH / 8, T), 256, 0, st>>>(G.q, fa, L->win, G.W, G.ctl, ratio, topk,
                                                                                ntop, ckv, G.S);
    else
        k_att_scores8<<<dim3((nmaxe + ACS - 1) / ACS, NH / 8, T), 256, 0, st>>>(G.q, fa, L->win, G.W, G.ctl, ratio, topk, ntop,
                                                                               ckv, G.S);
    if (fa) {
        k_att_fa_soft<<<dim3(1, T), NH, 0, st>>>(G.S, G.MS, G.ctl, ratio, ntop, L->sinks, G.den);
        if (T > att_x_min()) k_att_fa_out8x2<<<dim3(NH / 8, T), HD / 2, 0, st>>>(G.S, G.MS, G.den, L->win, G.W, G.ctl, ratio, topk, ntop, ckv, G.o);
        else k_att_fa_out8<<<dim3(NH / 8, T), HD, 0, st>>>(G.S, G.MS, G.den, L->win, G.W, G.ctl, ratio, topk, ntop, ckv, G.o);
    } else {
        if (wide) k_att_split_softw<<<dim3(NH, T), 32 * ANCH, 0, st>>>(G.S, G.ctl, ratio, ntop, G.am, G.as);
        else k_att_split_soft<<<dim3(NH, T), 32, 0, st>>>(G.S, G.ctl, ratio, ntop, G.am, G.as);
        if (wide) k_att_split_out8w<<<dim3(ANCH, NH / 8, T * ATT_DS), HD / ATT_DS, 0, st>>>(G.S, L->win, G.W, G.ctl, ratio, topk,
                                                                                          ntop, ckv, G.ap);
        else if (T > att_x_min()) k_att_split_out8x2<<<dim3(ANCH, NH / 8, T), HD / 2, 0, st>>>(G.S, L->win, G.W, G.ctl, ratio, topk, ntop, ckv, G.ap);
        else k_att_split_out8<<<dim3(ANCH, NH / 8, T), HD, 0, st>>>(G.S, L->win, G.W, G.ctl, ratio, topk, ntop, ckv, G.ap);
        k_att_combine<<<dim3(NH, T), HD, 0, st>>>(G.ap, G.am, G.as, L->sinks, G.ctl, ratio, ntop, G.o);
    }
    PMARK(l, 5);
    k_rope<<<dim3(T, NH / 8), dim3(RD / 2, 8), 0, st>>>(G.o, NH, HD, cs, 2 * RD, csoff, 1);
    if (flags & F_FP8_O) k_sim_fp8<<<dim3(NH * HD / 256, T), 256, 0, st>>>(G.o, NH * HD, NH * HD);
    k_quantm<<<dim3((NH * HD + 255) / 256, T), 256, 0, st>>>(G.o, NH * HD, NH * HD, G.oq, G.od);
    gemv_d(d->m_oa, G.oq, G.od, NULL, NH * HD, OR, NH * HD / OG, T, G.oa, OG * OR);
    k_quantm<<<dim3((OG * OR + 255) / 256, T), 256, 0, st>>>(G.oa, OG * OR, OG * OR, G.oaq, G.oad);
    gemv_d(d->m_ob, G.oaq, G.oad, NULL, OG * OR, 0, 0, T, G.ao, D);
}

extern "C" {

/* the first part of layer l: engram, attention, the FFN input, router; the expert inputs go to the host (event), the
   shared expert is queued behind them */
/* KINDI_DSV_EARLY (default 1), forwards of <= HOTT tokens: the engram projection of layer l + 1 (it reads only the
   token's engram rows) and the shared expert run after layer l's hand-off signal, while the host computes the routed
   experts, instead of on the path to the signal. The same kernels on the same inputs (the same bits). */
static int early_on(int T) {
    static int v = -1;
    if (v < 0) v = !getenv("KINDI_DSV_EARLY") || atoi(getenv("KINDI_DSV_EARLY")) != 0;
    return v && T <= HOTT;
}
static void eng_proj(int l, int T, size_t o_eq, size_t o_ed, float* y) {
    const DgLayerDesc* d = &G.L[l].h;
    const int8_t* q = (const int8_t*)(G.din + o_eq) + (size_t)d->eng * T * ENG_COLS * ENG_HD;
    const float* qd = (const float*)(G.din + o_ed) + (size_t)d->eng * T * ENG_COLS * ENG_HD / QK;
    gemv_d(d->m_ekv, q, qd, NULL, ENG_COLS * ENG_HD, 0, 0, T, y, (HC + 1) * D);
}

static int layer_a(int l, int T, int pos, int flags, size_t o_eq, size_t o_ed, size_t o_csl, size_t o_lg, int nl) {
    DLayer* L = &G.L[l];
    const DgLayerDesc* d = &L->h;
    G.cur_t = T;
    const int early = early_on(T);
    PMARK(l, 0);
    if (d->eng >= 0) {
        if (early && l > 0) k_eng_gate<<<dim3(T, HC), 512, 0, st>>>(G.h, G.ekv_e[d->eng], L->eng_gate);
        else {
            eng_proj(l, T, o_eq, o_ed, G.ekv);
            k_eng_gate<<<dim3(T, HC), 512, 0, st>>>(G.h, G.ekv, L->eng_gate);
        }
    }
    PMARK(l, 1);
    if (l >= NL - 3) k_hmean<<<dim3(D / 256, T), 256, 0, st>>>(G.h, G.mainh, l - (NL - 3));
    hc_in_d(L->hc_attn_fn, L->hc_attn_scale, L->hc_attn_base, G.pre, L->attn_norm, G.apre, T, 2);
    PMARK(l, 2);
    attention_d(l, T, pos, flags, (const float*)G.din, (const float*)(G.din + o_csl) + (size_t)l * T * RD,
                (const int*)(G.din + o_lg) + (size_t)l * T, nl);
    PMARK(l, 6);
    if (branch_on()) br_wait(2);
    k_post<<<dim3(D / 256, T), 256, 0, st>>>(G.h, G.ao, G.post, G.comb);
    hc_in_d(L->hc_ffn_fn, L->hc_ffn_scale, L->hc_ffn_base, G.apre, L->ffn_norm, G.pre, T, 3);
    PMARK(l, 7);
    gemv_d(d->m_gate, NULL, NULL, G.x, D, 0, 0, T, G.glog, NE);
    if (!early) {
        int mats[2] = {d->m_shg, d->m_shu}, ys[2] = {FF, FF};
        float* yp[2] = {G.sg, G.su};
        gemv_multi(mats, yp, ys, 2, G.xq, G.xd, D, T);
    }
    k_route<<<T, NE, 0, st>>>(G.glog, L->bias, T, G.sel, G.wsel);
    if (T <= HOTT) k_pack_out<<<20, 256, 0, st>>>(G.x, G.sel, G.wsel, T, G.dhx, G.dhsel, G.dhwsel);
    else {
        CK(cudaMemcpyAsync(G.hx, G.x, (size_t)T * D * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(G.hsel, G.sel, (size_t)T * NS * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(G.hwsel, G.wsel, (size_t)T * NS * 4, cudaMemcpyDeviceToHost, st));
    }
    k_signal<<<1, 1, 0, st>>>(G.dflag, G.ctl, l);
    if (branch_on()) br_wait(3);            /* the FFN's comb (used by layer_b's k_post) */
    PMARK(l, 8);
    if (early) {
        int mats[2] = {d->m_shg, d->m_shu}, ys[2] = {FF, FF};
        float* yp[2] = {G.sg, G.su};
        gemv_multi(mats, yp, ys, 2, G.xq, G.xd, D, T);
    }
    k_swiglu<<<ew_grid((size_t)T * FF), 256, 0, st>>>(G.sg, G.su, (size_t)T * FF);
    k_quantm<<<dim3((FF + 255) / 256, T), 256, 0, st>>>(G.sg, FF, FF, G.suq, G.sud);
    gemv_d(d->m_shd, G.suq, G.sud, NULL, FF, 0, 0, T, G.fo, D);
    if (G.nhot && T <= HOTT) {
        const int* hm = G.hotmap_d + (size_t)l * NE;
        k_hot_mx<<<dim3(FF / 16, NS, T), 128, 0, st>>>(0, hm, G.sel, G.hot_d, G.xq, G.xd, NULL, NULL, G.gu, NULL);
        k_hot_mx<<<dim3(FF / 16, NS, T), 128, 0, st>>>(1, hm, G.sel, G.hot_d, G.xq, G.xd, NULL, NULL, G.gu, NULL);
        k_hot_act<<<dim3(NS, T), 256, 0, st>>>(hm, G.sel, G.gu, G.hq, G.hd);
        k_hot_mx<<<dim3(D / 16, NS, T), 128, 0, st>>>(2, hm, G.sel, G.hot_d, NULL, NULL, G.hq, G.hd, NULL, G.vg);
    }
    if (early)
        for (int e = 0; e < 2; e++)
            if (G.eng_l[e] == l + 1) eng_proj(l + 1, T, o_eq, o_ed, G.ekv_e[e]);
    PMARK(l, 9);
    return 0;
}
/* the end of layer l: the host's routed-expert output back, added, hyper-connection post */
static int layer_b(int l, int T, float* dbg) {
    PMARK(l, 10);
    if (G.nhot && T <= HOTT) {               /* cached slots on the card + the host's values (mapped), in slot order */
        k_moe_combine<<<dim3(D / 256, T), 256, 0, st>>>(G.hotmap_d + (size_t)l * NE, G.sel, G.wsel, G.vg, G.dhparts, G.mo);
        k_add<<<ew_grid((size_t)T * D), 256, 0, st>>>(G.fo, G.mo, (size_t)T * D);
    } else if (T <= HOTT) {                  /* the host's sum read from mapped memory */
        k_add<<<ew_grid((size_t)T * D), 256, 0, st>>>(G.fo, G.dhmo, (size_t)T * D);
    } else {
        CK(cudaMemcpyAsync(G.mo, G.hmo, (size_t)T * D * 4, cudaMemcpyHostToDevice, st));
        k_add<<<ew_grid((size_t)T * D), 256, 0, st>>>(G.fo, G.mo, (size_t)T * D);
    }
    k_post<<<dim3(D / 256, T), 256, 0, st>>>(G.h, G.fo, G.post, G.comb);
    if (dbg) CK(cudaMemcpyAsync(dbg + (size_t)l * T * HC * D, G.h, (size_t)T * HC * D * 4, cudaMemcpyDeviceToHost, st));
    PMARK(l, 11);
    return 0;
}
static int head_d(int T, int nlog) {
    int t0 = T - nlog;
    k_collapse0<<<dim3(D / 256, nlog), 256, 0, st>>>(G.h + (size_t)t0 * HC * D, G.pre + t0 * HC, G.x);
    k_rmsnorm<<<nlog, 1024, 0, st>>>(G.x, D, G.x, D, G.out_norm, D);
    k_quant<<<nlog, 1024, 0, st>>>(G.x, D, D, G.lq, G.ld);
    return 0;
}
static int head_part(int a, int n) {
    gemv_d(G.m_out, G.lq + (size_t)a * D, G.ld + (size_t)a * D / QK, NULL, D, 0, 0, n, G.lg, VOCAB);
    CK(cudaMemcpyAsync(G.hlog, G.lg, (size_t)n * VOCAB * 4, cudaMemcpyDeviceToHost, st));
    return 0;
}

#include <time.h>
static double wall(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

#include <pthread.h>
/* one expert's three matrices (GGUF bytes, back to back) from the page cache into pinned staging k; the copier
   thread does this while the main thread computes experts with the CPU pool */
typedef struct { int l, n, e[64], j0, pairs; const int* off; cudaEvent_t hot_ev; } CopyJob2;
static CopyJob2 CJ2;
/* the streamed experts, all on st2 from this thread: wait until the staging buffer's previous DMA is done, copy the
   expert from the page cache into it, DMA, re-layout into a VRAM slot, compute its pairs */
/* the copy threads run on idle hyper-thread siblings (KINDI_SPLIT_COPY_CPU, default 59 = node 0's core 19), not
   on the expert pool's cores: there a copy thread pre-empts a pool thread and the whole job waits for it */
static void pin_copy_thread(int dflt) {
    const int cpu = getenv("KINDI_SPLIT_COPY_CPU") ? atoi(getenv("KINDI_SPLIT_COPY_CPU")) + (dflt - 59) : dflt;
    if (cpu < 0) return;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
}
static void* copier(void* a) {
    (void)a;
    pin_copy_thread(59);
    const size_t gb = (size_t)FF * (D / QK) * 17, db = (size_t)D * (FF / QK) * 17;
    const size_t gq = (size_t)FF * (D / QK) * 16, gs = (size_t)FF * (D / QK);
    for (int i = 0; i < CJ2.n; i++) {
        const int k = i & 1, e = CJ2.e[i], l = CJ2.l, q = i % SPLIT_SLOTS, j = CJ2.j0 + i;
        const int o = CJ2.off[j], n = CJ2.off[j + 1] - o;
        cudaEventSynchronize(G.raw_ev[k]);           /* recorded by this thread after the DMA of expert i - 2 */
        memcpy(G.raw_h[k], G.src_g[l] + e * gb, gb);
        memcpy(G.raw_h[k] + gb, G.src_u[l] + e * gb, gb);
        memcpy(G.raw_h[k] + 2 * gb, G.src_d[l] + e * db, db);
        cudaMemcpyAsync(G.raw_d[k], G.raw_h[k], 2 * gb + db, cudaMemcpyHostToDevice, G.st2);
        cudaEventRecord(G.raw_ev[k], G.st2);
        uint8_t *q0 = G.slot_q[q], *s0 = G.slot_s[q];
        k_relayout_mx<<<1440, 256, 0, G.st2>>>(G.raw_d[k], FF, D, q0, s0);
        k_relayout_mx<<<1440, 256, 0, G.st2>>>(G.raw_d[k] + gb, FF, D, q0 + gq, s0 + gs);
        k_relayout_mx<<<1440, 256, 0, G.st2>>>(G.raw_d[k] + 2 * gb, D, FF, q0 + 2 * gq, s0 + 2 * gs);
        const dim3 gg((FF / 16 + 3) / 4, (n + MMA_TOK - 1) / MMA_TOK), gd((D / 16 + 3) / 4, (n + MMA_TOK - 1) / MMA_TOK);
        k_mx_mma<<<gg, 128, 0, G.st2>>>(q0, s0, FF, D, G.xq, G.xd, D, G.xrow_d + o, n, G.gup + (size_t)o * 2 * FF, 2 * FF, 0);
        k_mx_mma<<<gg, 128, 0, G.st2>>>(q0 + gq, s0 + gs, FF, D, G.xq, G.xd, D, G.xrow_d + o, n, G.gup + (size_t)o * 2 * FF, 2 * FF, FF);
        k_act_pairs<<<n, 256, 0, G.st2>>>(G.gup + (size_t)o * 2 * FF, n, G.hqp + (size_t)o * FF, G.hdp + (size_t)o * (FF / QK));
        k_mx_mma<<<gd, 128, 0, G.st2>>>(q0 + 2 * gq, s0 + 2 * gs, D, FF, G.hqp + (size_t)o * FF, G.hdp + (size_t)o * (FF / QK), FF,
                                        G.ident_d, n, G.vp + (size_t)o * D, D, 0);
    }
    /* the card's values to the host as soon as its last expert is done (overlaps the host's experts) */
    if (CJ2.pairs) cudaStreamWaitEvent(G.st2, CJ2.hot_ev, 0);   /* and the hot experts on st */
    if (CJ2.pairs) cudaMemcpyAsync(G.vp_h, G.vp, (size_t)CJ2.pairs * D * 4, cudaMemcpyDeviceToHost, G.st2);
    return NULL;
}

/* Predicted-expert prefetch for prompt chunks (KINDI_SPLIT_PF = file of per-layer expert lists, most used first;
   KINDI_SPLIT_PF_N experts per layer). The experts that carry the most pairs of a prompt chunk are mostly the same
   ones in every prompt (a static list from routing traces covers ~90 % of the ideal top-12 pairs), so they need no
   routing result: a thread streams layer l + 2's list into one of two VRAM sets (page cache -> pinned staging -> DMA
   -> re-layout, on its own stream) while layers l and l + 1 run, and the card's experts of a chunk are the listed
   ones its tokens use. The PCIe time leaves the critical path; which experts the card computes changes nothing in
   the arithmetic of either side. */
#define PF_MAX 40
static struct {
    int n, list[NL][PF_MAX], nl[NL];   /* per layer: the list, its valid length */
    int adapt;                    /* the lists follow the previous chunk's routing (KINDI_SPLIT_PF=prev) */
    uint8_t *q[2][PF_MAX], *sc[2][PF_MAX];
    uint8_t *raw_h[2], *raw_d[2];
    cudaEvent_t raw_ev[2], ready[2], free_[2];
    cudaStream_t st3;
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int want[2], done[2];         /* per set: layer requested, layer loaded (-1: none) */
    int on;
} PF;
static void* pf_thread(void* a) {
    (void)a;
    pin_copy_thread(58);
    const size_t gb = (size_t)FF * (D / QK) * 17, db = (size_t)D * (FF / QK) * 17;
    const size_t gq = (size_t)FF * (D / QK) * 16, gs = (size_t)FF * (D / QK);
    int kk = 0;
    for (;;) {
        int s = -1, l = -1;
        pthread_mutex_lock(&PF.mu);
        while (s < 0) {
            for (int i = 0; i < 2; i++) if (PF.want[i] >= 0 && PF.done[i] != PF.want[i]) { s = i; l = PF.want[i]; break; }
            if (s < 0) pthread_cond_wait(&PF.cv, &PF.mu);
        }
        PF.done[s] = -1;
        pthread_mutex_unlock(&PF.mu);
        cudaEventSynchronize(PF.free_[s]);           /* the set's last layer is computed */
        const int ne = PF.nl[l] < PF.n ? PF.nl[l] : PF.n;
        for (int i = 0; i < ne; i++) {
            const int e = PF.list[l][i], k = kk++ & 1;
            cudaEventSynchronize(PF.raw_ev[k]);
            memcpy(PF.raw_h[k], G.src_g[l] + e * gb, gb);
            memcpy(PF.raw_h[k] + gb, G.src_u[l] + e * gb, gb);
            memcpy(PF.raw_h[k] + 2 * gb, G.src_d[l] + e * db, db);
            cudaMemcpyAsync(PF.raw_d[k], PF.raw_h[k], 2 * gb + db, cudaMemcpyHostToDevice, PF.st3);
            cudaEventRecord(PF.raw_ev[k], PF.st3);
            uint8_t *q0 = PF.q[s][i], *s0 = PF.sc[s][i];
            k_relayout_mx<<<1440, 256, 0, PF.st3>>>(PF.raw_d[k], FF, D, q0, s0);
            k_relayout_mx<<<1440, 256, 0, PF.st3>>>(PF.raw_d[k] + gb, FF, D, q0 + gq, s0 + gs);
            k_relayout_mx<<<1440, 256, 0, PF.st3>>>(PF.raw_d[k] + 2 * gb, D, FF, q0 + 2 * gq, s0 + 2 * gs);
        }
        cudaEventRecord(PF.ready[s], PF.st3);
        pthread_mutex_lock(&PF.mu);
        if (PF.want[s] == l) PF.done[s] = l;
        pthread_cond_broadcast(&PF.cv);
        pthread_mutex_unlock(&PF.mu);
    }
    return NULL;
}
static void pf_request(int l) {
    const int s = l & 1;
    pthread_mutex_lock(&PF.mu);
    if (PF.want[s] != l) { PF.want[s] = l; pthread_cond_broadcast(&PF.cv); }
    pthread_mutex_unlock(&PF.mu);
}
static int pf_init(void) {
    static int tried = 0;
    if (tried) return 0;
    tried = 1;
    const char* path = getenv("KINDI_SPLIT_PF") ? getenv("KINDI_SPLIT_PF") : "prev";   /* default: adaptive */
    if (!strcmp(path, "0") || !path[0]) return 0;
    PF.n = getenv("KINDI_SPLIT_PF_N") ? atoi(getenv("KINDI_SPLIT_PF_N")) : 16;
    if (PF.n > PF_MAX) PF.n = PF_MAX;
    PF.adapt = !strcmp(path, "prev");
    if (!PF.adapt) {                         /* a static list file: "layer e1 e2 ..." per line */
        FILE* f = fopen(path, "r");
        if (!f) { fprintf(stderr, "dsv41_gpu: cannot read %s\n", path); return 0; }
        int got = 0;
        for (int l; fscanf(f, "%d", &l) == 1;) {
            int k = 0;
            for (int e; k < PF_MAX && fscanf(f, "%d", &e) == 1; k++) if (l >= 0 && l < NL) PF.list[l][k] = e;
            if (l >= 0 && l < NL) { PF.nl[l] = k; got += k >= PF.n; }
            if (k == PF_MAX) { int c; while ((c = fgetc(f)) != '\n' && c != EOF) {} }
        }
        fclose(f);
        if (got < NL) { fprintf(stderr, "dsv41_gpu: %s: lists for %d of %d layers, prefetch off\n", path, got, NL); return 0; }
    }
    const size_t eb = (size_t)3 * FF * D / QK * 17;
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < PF.n; i++) {
            PF.q[s][i] = (uint8_t*)dz(eb / 17 * 16); PF.sc[s][i] = (uint8_t*)dz(eb / 17);
            if (!PF.q[s][i] || !PF.sc[s][i]) { fprintf(stderr, "dsv41_gpu: no VRAM for the prefetch sets\n"); return 0; }
        }
        PF.raw_d[s] = (uint8_t*)dz(eb);
        CK(cudaMallocHost(&PF.raw_h[s], eb));
        CK(cudaEventCreateWithFlags(&PF.raw_ev[s], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&PF.ready[s], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&PF.free_[s], cudaEventDisableTiming));
        PF.want[s] = PF.done[s] = -1;
    }
    CK(cudaStreamCreateWithFlags(&PF.st3, cudaStreamNonBlocking));
    pthread_mutex_init(&PF.mu, NULL); pthread_cond_init(&PF.cv, NULL);
    pthread_create(&PF.th, NULL, pf_thread, NULL);
    PF.on = 1;
    fprintf(stderr, "dsv41_gpu: predicted-expert prefetch on (%d experts per layer, %.0f MB of VRAM)\n", PF.n,
            2.0 * PF.n * eb / 1048576.0);
    return 0;
}

/* the experts of a prompt chunk, split: the card takes the hot experts, the prefetched experts its tokens use, and
   the nstream most used other experts of this layer (streamed now), the host the rest (moe_forward_parts); the host
   adds all row values in slot order (combine) */
static int host_experts_split(int l, int T) {
    pf_init();
    const double t0 = wall();
    const int np_all = T * NS;
    static int cnt[NE], gpu_e[NE], off[NE + 1], fill[NE];
    for (int e = 0; e < NE; e++) cnt[e] = 0;
    for (int i = 0; i < np_all; i++) cnt[G.hsel[i]]++;
    /* the card's experts: hot ones with pairs, then the most used others (stream order = most pairs first) */
    int ng = 0, nst = 0, pairs = 0;
    int stream_e[64];
    for (int e = 0; e < NE; e++) if (cnt[e] && G.hotmap_h[l][e] >= 0 && pairs + cnt[e] <= G.pmax) { gpu_e[ng++] = e; pairs += cnt[e]; }
    const int nhot_used = ng;
    /* the card's other experts: the ncard most used ones of this chunk (>= 8 pairs, ties to the lower index). The choice
       depends on this chunk alone, so the bits do not depend on earlier chunks; the prefetch only decides which of
       them are already on the card (the others are streamed now). */
    int npf = 0, pf_i[PF_MAX];
    const int pfs = l & 1;
    static int ncard_env = -2;
    if (ncard_env == -2) ncard_env = getenv("KINDI_SPLIT_CARD") ? atoi(getenv("KINDI_SPLIT_CARD")) : -1;
    const int ncard = ncard_env >= 0 ? ncard_env : G.nstream;   /* KINDI_SPLIT_CARD, default KINDI_SPLIT_STREAM (12) */
    int want[64], nw = 0;
    {
        static int taken[NE];
        for (int e = 0; e < NE; e++) taken[e] = G.hotmap_h[l][e] >= 0;
        while (nw < ncard && nw < 64) {
            int best = -1;
            for (int e = 0; e < NE; e++) if (cnt[e] >= 8 && !taken[e] && (best < 0 || cnt[e] > cnt[best])) best = e;
            if (best < 0 || pairs + cnt[best] > G.pmax) break;
            taken[best] = 1; want[nw++] = best; pairs += cnt[best];
        }
    }
    if (PF.on) {                             /* wait until the set holds this layer */
        pf_request(l);
        pthread_mutex_lock(&PF.mu);
        while (PF.done[pfs] != l) pthread_cond_wait(&PF.cv, &PF.mu);
        pthread_mutex_unlock(&PF.mu);
    }
    const int ne = PF.on ? (PF.nl[l] < PF.n ? PF.nl[l] : PF.n) : 0;
    for (int w = 0; w < nw; w++) {           /* prefetched first (hot, prefetched, streamed: the card order) */
        int i = 0;
        while (i < ne && PF.list[l][i] != want[w]) i++;
        if (i < ne) { pf_i[npf++] = i; gpu_e[ng++] = want[w]; want[w] = -1; }
    }
    for (int w = 0; w < nw; w++) if (want[w] >= 0) { stream_e[nst++] = want[w]; gpu_e[ng++] = want[w]; }
    /* pair lists of the card's experts, in expert order; the skip mask of the host */
    static int ismap[NE];
    for (int e = 0; e < NE; e++) ismap[e] = -1;
    for (int j = 0; j < ng; j++) ismap[gpu_e[j]] = j;
    off[0] = 0;
    for (int j = 0; j < ng; j++) { off[j + 1] = off[j] + cnt[gpu_e[j]]; fill[j] = off[j]; }
    static int pts[8192];                    /* (token, slot) of each card pair */
    for (int i = 0; i < np_all; i++) {
        const int j = ismap[G.hsel[i]];
        G.hskip[i] = 0;
        if (j < 0) continue;
        const int p = fill[j]++;
        G.xrow_h[p] = i / NS; pts[p] = i;
    }
    uint8_t* skip = (uint8_t*)G.vp_h;        /* reuse: T * NS bytes (the card's values come later) */
    static uint8_t skipbuf[8192];
    skip = skipbuf;
    for (int i = 0; i < np_all; i++) skip[i] = ismap[G.hsel[i]] >= 0;
    /* the pair lists go up on st2: the host has seen this layer's flag, so the chunk's inputs (xq) are on the card; st
       may already run the other pipelined chunk's layer */
    CK(cudaMemcpyAsync(G.xrow_d, G.xrow_h, (size_t)pairs * 4, cudaMemcpyHostToDevice, G.st2));
    cudaEvent_t xr_ev; cudaEventCreateWithFlags(&xr_ev, cudaEventDisableTiming);
    cudaEventRecord(xr_ev, G.st2);
    if (nhot_used) cudaStreamWaitEvent(st, xr_ev, 0);
    /* hot experts on st */
    for (int j = 0; j < nhot_used; j++) {
        const DHot* h = &G.hot_h[G.hotmap_h[l][gpu_e[j]]];
        const int n = off[j + 1] - off[j];
        const dim3 gg((FF / 16 + 3) / 4, (n + MMA_TOK - 1) / MMA_TOK), gd((D / 16 + 3) / 4, (n + MMA_TOK - 1) / MMA_TOK);
        k_mx_mma<<<gg, 128, 0, st>>>(h->gq, h->gs, FF, D, G.xq, G.xd, D, G.xrow_d + off[j], n, G.gup + (size_t)off[j] * 2 * FF, 2 * FF, 0);
        k_mx_mma<<<gg, 128, 0, st>>>(h->uq, h->us, FF, D, G.xq, G.xd, D, G.xrow_d + off[j], n, G.gup + (size_t)off[j] * 2 * FF, 2 * FF, FF);
        k_act_pairs<<<n, 256, 0, st>>>(G.gup + (size_t)off[j] * 2 * FF, n, G.hqp + (size_t)off[j] * FF, G.hdp + (size_t)off[j] * (FF / QK));
        /* down: the inputs are this expert's own pairs, rows 0 .. n of hq (identity gather) */
        k_mx_mma<<<gd, 128, 0, st>>>(h->dq, h->ds, D, FF, G.hqp + (size_t)off[j] * FF, G.hdp + (size_t)off[j] * (FF / QK), FF,
                                     G.ident_d, n, G.vp + (size_t)off[j] * D, D, 0);
    }
    /* prefetched experts on st2 (the set is loaded: wait for its re-layout), then the set is free for layer l + 2 */
    if (PF.on) {
        cudaStreamWaitEvent(G.st2, PF.ready[pfs], 0);
        const size_t gq = (size_t)FF * (D / QK) * 16, gs = (size_t)FF * (D / QK);
        for (int jj = 0; jj < npf; jj++) {
            const int j = nhot_used + jj, n = off[j + 1] - off[j];
            const uint8_t *q0 = PF.q[pfs][pf_i[jj]], *s0 = PF.sc[pfs][pf_i[jj]];
            const dim3 gg((FF / 16 + 3) / 4, (n + MMA_TOK - 1) / MMA_TOK), gd((D / 16 + 3) / 4, (n + MMA_TOK - 1) / MMA_TOK);
            k_mx_mma<<<gg, 128, 0, G.st2>>>(q0, s0, FF, D, G.xq, G.xd, D, G.xrow_d + off[j], n, G.gup + (size_t)off[j] * 2 * FF, 2 * FF, 0);
            k_mx_mma<<<gg, 128, 0, G.st2>>>(q0 + gq, s0 + gs, FF, D, G.xq, G.xd, D, G.xrow_d + off[j], n, G.gup + (size_t)off[j] * 2 * FF, 2 * FF, FF);
            k_act_pairs<<<n, 256, 0, G.st2>>>(G.gup + (size_t)off[j] * 2 * FF, n, G.hqp + (size_t)off[j] * FF, G.hdp + (size_t)off[j] * (FF / QK));
            k_mx_mma<<<gd, 128, 0, G.st2>>>(q0 + 2 * gq, s0 + 2 * gs, D, FF, G.hqp + (size_t)off[j] * FF, G.hdp + (size_t)off[j] * (FF / QK), FF,
                                            G.ident_d, n, G.vp + (size_t)off[j] * D, D, 0);
        }
        if (!G.pf_hold) cudaEventRecord(PF.free_[pfs], G.st2);
        if (PF.adapt && !G.pf_hold) {                      /* this chunk's most used experts: the list of layer l for the next chunk */
            static int used[NE];
            for (int e = 0; e < NE; e++) used[e] = 0;
            int k = 0;
            for (; k < PF.n; k++) {
                int best = -1;
                for (int e = 0; e < NE; e++) if (!used[e] && cnt[e] && (best < 0 || cnt[e] > cnt[best])) best = e;
                if (best < 0) break;
                used[best] = 1;
                PF.list[l][k] = best;
            }
            pthread_mutex_lock(&PF.mu); PF.nl[l] = k; pthread_mutex_unlock(&PF.mu);
        }
        if (!G.pf_hold) {
            pf_request((l + 2) % NL);
            pf_request((l + 1) % NL);                /* cold start: the next layer's set (no-op when requested) */
        }
    }
    /* streamed experts on st2: copier thread fills pinned staging, DMA, re-layout, compute */
    pthread_t cth;
    CJ2.l = l; CJ2.n = nst; CJ2.j0 = nhot_used + npf; CJ2.off = off; CJ2.pairs = 0;
    for (int i = 0; i < nst; i++) CJ2.e[i] = stream_e[i];
    if (pairs && G.d2h_early) {                      /* the D2H goes on st2 after the card's last expert */
        if (nst) {
            if (!CJ2.hot_ev) CK(cudaEventCreateWithFlags(&CJ2.hot_ev, cudaEventDisableTiming));
            cudaEventRecord(CJ2.hot_ev, st);
            CJ2.pairs = pairs;
        } else {
            cudaEvent_t eh; cudaEventCreateWithFlags(&eh, cudaEventDisableTiming);
            cudaEventRecord(eh, st); cudaStreamWaitEvent(G.st2, eh, 0); cudaEventDestroy(eh);
            CK(cudaMemcpyAsync(G.vp_h, G.vp, (size_t)pairs * D * 4, cudaMemcpyDeviceToHost, G.st2));
        }
    }
    if (nst) pthread_create(&cth, NULL, copier, NULL);
    /* the host's pairs meanwhile (the main thread is thread 0 of the expert pool) ... */
    const double t1 = wall();
    int r = 0;
    if (pairs < np_all) r = G.parts(l, T, G.hx, G.hsel, skip, G.hpart_big);
    const double t2 = wall();
    if (nst) pthread_join(cth, NULL);
    /* the card's values to the host, into their (token, slot) places; then the combine */
    cudaEvent_t e2; cudaEventCreateWithFlags(&e2, cudaEventDisableTiming);
    cudaEventRecord(e2, G.st2);
    cudaStreamWaitEvent(st, e2, 0);
    if (pairs) {
        if (G.d2h_early) CK(cudaStreamSynchronize(G.st2));
        else {
            CK(cudaMemcpyAsync(G.vp_h, G.vp, (size_t)pairs * D * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
        }
        if (!G.combine_alt)
            for (int p = 0; p < pairs; p++) memcpy(G.hpart_big + (size_t)pts[p] * D, G.vp_h + (size_t)p * D, D * 4);
    }
    static int crow[8192];                   /* (token, slot) -> card row, or -1 */
    if (G.combine_alt) {
        for (int i = 0; i < np_all; i++) crow[i] = -1;
        for (int p = 0; p < pairs; p++) crow[pts[p]] = p;
    }
    cudaEventDestroy(e2); cudaEventDestroy(xr_ev);
    const double t3 = wall();
    if (!r) r = G.combine_alt ? G.combine_alt(T, G.hwsel, G.hpart_big, crow, G.vp_h, G.hmo)
                              : G.combine(T, G.hwsel, G.hpart_big, G.hmo);
    G.sp_cpu += t2 - t1; G.sp_wait += t3 - t2; G.sp_gpu += t1 - t0; G.sp_n++; G.sp_gpairs += pairs; G.sp_pairs += np_all;
    if (getenv("KINDI_SPLIT_LOG") && G.sp_n % 40 == 0)
        fprintf(stderr, "dsv41_gpu split: %ld layer chunks, card pairs %.1f%%, host experts %.1f ms, wait for card %.1f ms, setup %.1f ms per layer chunk (last: %d hot + %d streamed)\n",
                G.sp_n, 100.0 * G.sp_gpairs / G.sp_pairs, G.sp_cpu * 1e3 / G.sp_n, G.sp_wait * 1e3 / G.sp_n, G.sp_gpu * 1e3 / G.sp_n, nhot_used, nst + npf);
    return r;
}

/* the host's part of layer l's routed experts: all of them, or the uncached slots when the hot path is on */
static int host_experts(int l, int T, dg_moe_fn moe) {
    if (G.split && T > HOTT && G.combine) return host_experts_split(l, T);
    if (G.nhot && T <= HOTT) {
        int any = 0;
        for (int i = 0; i < T * NS; i++) { G.hskip[i] = G.hotmap_h[l][G.hsel[i]] >= 0; any |= !G.hskip[i]; G.hhit += G.hskip[i]; }
        G.hslot += T * NS;
        return any ? G.parts(l, T, G.hx, G.hsel, G.hskip, G.hparts) : 0;
    }
    return moe(l, T, G.hx, G.hsel, G.hwsel, G.hmo);
}

static void wait_flag(int l) {
    const int want = G.seq * 64 + l;
    while (*G.hflag != want) {
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
    }
}

/* the decode graphs (T = 1, logits of the token), captured on first use and when the flags change */
static int capture_segments(int flags, size_t o_eq, size_t o_ed, size_t o_csl, size_t o_lg) {
    for (int i = 0; i <= NL; i++) {
        if (G.seg_ok) cudaGraphExecDestroy(G.seg[i]);
    }
    G.seg_ok = 0;
    G.graphing = 1;
    for (int i = 0; i <= NL; i++) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        if (i == 0) k_embed<<<dim3(D / 256, 1), 256, 0, st>>>((const float*)(G.din + (size_t)2 * RD * 4), G.h, G.pre);
        else layer_b(i - 1, 1, NULL);
        if (i < NL) layer_a(i, 1, 0, flags, o_eq, o_ed, o_csl, o_lg, 1);
        else { head_d(1, 1); head_part(0, 1); }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&G.seg[i], g, 0));
        cudaGraphDestroy(g);
    }
    G.graphing = 0;
    G.seg_ok = 1; G.seg_flags = flags;
    return 0;
}

/* One forward of T tokens at positions pos.. (the host keeps the token map, the engram rows and the RoPE tables):
   cs [T][2 RD] (window, compressed), emb [T][D] dequantized embeddings, eq/ed [2][T][6144] engram rows (Q8),
   cs_lat [NL][T][RD] and latg [NL][T] the latents each kv source layer completes (nlat [NL]). moe(l, ...) computes the
   routed experts on the host. logits: [nlog][VOCAB] for the last nlog tokens (nlog 0: none). dbg: when not NULL,
   h after every layer [NL][T][HC][D]. A single decode token (T = 1, nlog = 1, no dbg) replays CUDA graphs when
   KINDI_DSV_GRAPH is not 0. */
/* Two prompt chunks in a pipeline (dg_forward2): the card runs chunk B's layer l while the host computes chunk A's
   layer-l experts, and chunk A's layer l + 1 while the host computes B's. B's attention at layer l reads A's caches of
   layer l, written before on the same stream. Each chunk has its own activations and hand-off buffers (a context,
   swapped into G); the kernels and the experts are those of two dg_forward calls, so the bits are the same. */
typedef struct {
    float *h, *pre, *apre, *post, *comb, *x, *fo, *mo, *mainh, *wsel, *hx, *hmo, *hwsel, *dhx, *dhmo, *dhwsel;
    int8_t* xq; float* xd; int *sel, *hsel, *dhsel, *ctl, *dflag, seq;
    uint8_t *din, *hin; volatile int* hflag;
    float* hpart_big; uint8_t* hskip;
    uint8_t* cand;                /* the indexer's block candidates (made at one layer, used by the later ones) */
    float* ekv_e[2];              /* the early engram projections (layer_a of the previous layer makes them) */
    int *topk[NL], *ntop[NL];     /* the indexer source layers' selections (used by the layers that share them) */
} PipeCtx;
static PipeCtx PC[2];
static int pc_cur = 0;
static void pc_save(PipeCtx* c) {
    c->h = G.h; c->pre = G.pre; c->apre = G.apre; c->post = G.post; c->comb = G.comb; c->x = G.x; c->fo = G.fo;
    c->mo = G.mo; c->mainh = G.mainh; c->wsel = G.wsel; c->hx = G.hx; c->hmo = G.hmo; c->hwsel = G.hwsel;
    c->dhx = G.dhx; c->dhmo = G.dhmo; c->dhwsel = G.dhwsel; c->xq = G.xq; c->xd = G.xd; c->sel = G.sel;
    c->hsel = G.hsel; c->dhsel = G.dhsel; c->ctl = G.ctl; c->dflag = G.dflag; c->seq = G.seq; c->din = G.din;
    c->hin = G.hin; c->hflag = G.hflag; c->hpart_big = G.hpart_big; c->hskip = G.hskip; c->cand = G.cand;
    c->ekv_e[0] = G.ekv_e[0]; c->ekv_e[1] = G.ekv_e[1];
    for (int l = 0; l < NL; l++) { c->topk[l] = G.L[l].topk; c->ntop[l] = G.L[l].ntop; }
}
static void pc_load(const PipeCtx* c) {
    G.h = c->h; G.pre = c->pre; G.apre = c->apre; G.post = c->post; G.comb = c->comb; G.x = c->x; G.fo = c->fo;
    G.mo = c->mo; G.mainh = c->mainh; G.wsel = c->wsel; G.hx = c->hx; G.hmo = c->hmo; G.hwsel = c->hwsel;
    G.dhx = c->dhx; G.dhmo = c->dhmo; G.dhwsel = c->dhwsel; G.xq = c->xq; G.xd = c->xd; G.sel = c->sel;
    G.hsel = c->hsel; G.dhsel = c->dhsel; G.ctl = c->ctl; G.dflag = c->dflag; G.seq = c->seq; G.din = c->din;
    G.hin = c->hin; G.hflag = c->hflag; G.hpart_big = c->hpart_big; G.hskip = c->hskip; G.cand = c->cand;
    G.ekv_e[0] = c->ekv_e[0]; G.ekv_e[1] = c->ekv_e[1];
    for (int l = 0; l < NL; l++) { G.L[l].topk = c->topk[l]; G.L[l].ntop = c->ntop[l]; }
}
static void pc_use(int i) {
    if (i == pc_cur) return;
    pc_save(&PC[pc_cur]);
    pc_load(&PC[i]);
    pc_cur = i;
}
/* the second context's buffers (once) */
static int pc_init(void) {
    static int done = 0;
    if (done) return done > 0 ? 0 : -1;
    done = -1;
    const size_t T = G.tmax;
    pc_save(&PC[0]);
    PipeCtx* c = &PC[1];
    *c = PC[0];
    c->h = (float*)dz(T * HC * D * 4); c->pre = (float*)dz(T * HC * 4); c->apre = (float*)dz(T * HC * 4);
    c->post = (float*)dz(T * HC * 4); c->comb = (float*)dz(T * 16 * 4); c->x = (float*)dz(T * D * 4);
    c->fo = (float*)dz(T * D * 4); c->mo = (float*)dz(T * D * 4); c->mainh = (float*)dz(T * 3 * D * 4);
    c->wsel = (float*)dz(T * NS * 4); c->sel = (int*)dz(T * NS * 4);
    c->xq = (int8_t*)dz(T * D); c->xd = (float*)dz(T * D / QK * 4);
    c->din = (uint8_t*)dz(G.incap);
    c->cand = (uint8_t*)dz(T * G.cands);
    for (int e = 0; e < 2; e++) c->ekv_e[e] = (float*)dz((size_t)HOTT * (HC + 1) * D * 4);
    for (int l = 0; l < NL; l++)
        if (PC[0].topk[l]) { c->topk[l] = (int*)dz(T * ITOPK * 4); c->ntop[l] = (int*)dz(T * 4); }
    if (!c->h || !c->din || !c->mainh || !c->xd) return -1;
    CK(cudaMallocHost(&c->hin, G.incap));
    CK(cudaHostAlloc((void**)&c->hx, T * D * 4, cudaHostAllocMapped)); CK(cudaHostAlloc((void**)&c->hmo, T * D * 4, cudaHostAllocMapped));
    CK(cudaHostAlloc((void**)&c->hsel, T * NS * 4, cudaHostAllocMapped)); CK(cudaHostAlloc((void**)&c->hwsel, T * NS * 4, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&c->dhx, c->hx, 0)); CK(cudaHostGetDevicePointer((void**)&c->dhmo, c->hmo, 0));
    CK(cudaHostGetDevicePointer((void**)&c->dhsel, c->hsel, 0)); CK(cudaHostGetDevicePointer((void**)&c->dhwsel, c->hwsel, 0));
    CK(cudaHostAlloc((void**)&c->hflag, 64, cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer((void**)&c->dflag, (void*)c->hflag, 0));
    *c->hflag = -1;
    if (G.split) {
        CK(cudaMallocHost(&c->hpart_big, T * NS * D * 4));
        c->hskip = (uint8_t*)malloc(T * NS);
    }
    done = 1;
    return 0;
}
/* inputs of one chunk into its context: the upload layout of dg_forward */
typedef struct { const float *cs, *emb; const int8_t* eq; const float *ed, *cs_lat; const int *latg, *nlat; } DgIn;
static int pc_upload(int pos, int T, const DgIn* in, size_t* oo) {
    size_t o_cs = 0, o_emb = o_cs + (size_t)T * 2 * RD * 4, o_eq = o_emb + (size_t)T * D * 4;
    size_t neq = (size_t)2 * T * ENG_COLS * ENG_HD, o_ed = o_eq + ((neq + 255) & ~(size_t)255);
    size_t o_csl = o_ed + ((neq / QK * 4 + 255) & ~(size_t)255), o_lg = o_csl + (size_t)NL * T * RD * 4;
    size_t o_ctl = o_lg + (((size_t)NL * T * 4 + 255) & ~(size_t)255);
    size_t tot = o_ctl + 256;
    memcpy(G.hin + o_cs, in->cs, (size_t)T * 2 * RD * 4);
    memcpy(G.hin + o_emb, in->emb, (size_t)T * D * 4);
    memcpy(G.hin + o_eq, in->eq, neq);
    memcpy(G.hin + o_ed, in->ed, neq / QK * 4);
    memcpy(G.hin + o_csl, in->cs_lat, (size_t)NL * T * RD * 4);
    memcpy(G.hin + o_lg, in->latg, (size_t)NL * T * 4);
    int* hctl = (int*)(G.hin + o_ctl);
    hctl[0] = pos;
    for (int l = 0; l < NL; l++) hctl[1 + l] = in->nlat[l];
    G.seq = (G.seq + 2) & 0xffffff;          /* distinct from the other context's sequence */
    hctl[1 + NL] = G.seq;
    G.ctl = (int*)(G.din + o_ctl);
    CK(cudaMemcpyAsync(G.din, G.hin, tot, cudaMemcpyHostToDevice, st));
    oo[0] = o_eq; oo[1] = o_ed; oo[2] = o_csl; oo[3] = o_lg; oo[4] = o_emb;
    return 0;
}

/* KINDI_DSV_TPROF=1: host-clock split of every forward of <= 8 tokens (graph and non-graph): the card's segments
   (from the end of the previous host work to the layer's completion flag: launch + layer_b(l - 1) + layer_a(l)),
   the host experts, the head (last launch to logits), by T; printed every 256 forwards and by dg_tprof_print() */
static struct { int on; double seg[9], host[9], head[9], up[9]; long n[9]; long tot; } TP = {-1};
extern "C" void dg_tprof_print(void) {
    if (TP.on != 1) return;
    for (int t = 1; t <= 8; t++) {
        if (!TP.n[t]) continue;
        double n = (double)TP.n[t];
        fprintf(stderr, "dsv41_gpu tprof T=%d: %ld forwards, ms per forward: upload %.2f, card segments %.2f, host experts %.2f, "
                "head %.2f (sum %.2f)\n", t, TP.n[t], TP.up[t] * 1e3 / n, TP.seg[t] * 1e3 / n, TP.host[t] * 1e3 / n,
                TP.head[t] * 1e3 / n, (TP.up[t] + TP.seg[t] + TP.host[t] + TP.head[t]) * 1e3 / n);
    }
}

int dg_forward(int pos, int T, int flags, const float* cs, const float* emb, const int8_t* eq, const float* ed,
               const float* cs_lat, const int* latg, const int* nlat, dg_moe_fn moe, float* logits, int nlog, float* dbg) {
    if (TP.on < 0) { TP.on = getenv("KINDI_DSV_TPROF") && atoi(getenv("KINDI_DSV_TPROF")) != 0; if (TP.on) atexit(dg_tprof_print); }
    const int tp = TP.on == 1 && T <= 8;
    double tp0 = tp ? wall() : 0, tpa = 0, tpb = 0;
    pc_use(0);
    if (T < 1 || T > G.tmax || pos + T > G.cap) return -1;
    static int use_graph = -1;
    if (use_graph < 0) use_graph = !getenv("KINDI_DSV_GRAPH") || atoi(getenv("KINDI_DSV_GRAPH")) != 0;
    int graph = use_graph && T == 1 && nlog == 1 && !dbg;
    /* one upload of the inputs; for the graphs the layout is the one of T = 1 */
    size_t o_cs = 0, o_emb = o_cs + (size_t)T * 2 * RD * 4, o_eq = o_emb + (size_t)T * D * 4;
    size_t neq = (size_t)2 * T * ENG_COLS * ENG_HD, o_ed = o_eq + ((neq + 255) & ~(size_t)255);
    size_t o_csl = o_ed + ((neq / QK * 4 + 255) & ~(size_t)255), o_lg = o_csl + (size_t)NL * T * RD * 4;
    size_t o_ctl = o_lg + (((size_t)NL * T * 4 + 255) & ~(size_t)255);
    size_t tot = o_ctl + 256;
    memcpy(G.hin + o_cs, cs, (size_t)T * 2 * RD * 4);
    memcpy(G.hin + o_emb, emb, (size_t)T * D * 4);
    memcpy(G.hin + o_eq, eq, neq);
    memcpy(G.hin + o_ed, ed, neq / QK * 4);
    memcpy(G.hin + o_csl, cs_lat, (size_t)NL * T * RD * 4);
    memcpy(G.hin + o_lg, latg, (size_t)NL * T * 4);
    int* hctl = (int*)(G.hin + o_ctl);
    hctl[0] = pos;
    for (int l = 0; l < NL; l++) hctl[1 + l] = nlat[l];
    G.seq = (G.seq + 1) & 0xffffff;
    hctl[1 + NL] = G.seq;
    G.ctl = (int*)(G.din + o_ctl);
    CK(cudaMemcpyAsync(G.din, G.hin, tot, cudaMemcpyHostToDevice, st));
    if (tp) { tpa = wall(); TP.up[T] += tpa - tp0; }
    if (graph) {
        if (!G.seg_ok || G.seg_flags != flags) if (capture_segments(flags, o_eq, o_ed, o_csl, o_lg)) return -4;
        for (int i = 0; i <= NL; i++) {
            CK(cudaGraphLaunch(G.seg[i], st));
            if (i == NL) break;
            wait_flag(i);
            if (tp) { tpb = wall(); TP.seg[T] += tpb - tpa; }
            if (host_experts(i, 1, moe)) return -2;
            if (tp) { tpa = wall(); TP.host[T] += tpa - tpb; }
        }
        CK(cudaStreamSynchronize(st));
        memcpy(logits, G.hlog, (size_t)VOCAB * 4);
        if (tp) TP.head[T] += wall() - tpa;
    } else {
        k_embed<<<dim3(D / 256, T), 256, 0, st>>>((const float*)(G.din + o_emb), G.h, G.pre);
        for (int l = 0; l < NL; l++) {
            if (layer_a(l, T, pos, flags, o_eq, o_ed, o_csl, o_lg, nlat[l])) return -3;
            double h0 = G.prof ? wall() : 0;
            wait_flag(l);
            double h1 = G.prof ? wall() : 0;
            if (tp) { tpb = wall(); TP.seg[T] += tpb - tpa; }
            if (host_experts(l, T, moe)) return -2;
            if (tp) { tpa = wall(); TP.host[T] += tpa - tpb; }
            if (G.prof && T == 1) { G.hwait += h1 - h0; G.hmoe += wall() - h1; }
            if (layer_b(l, T, dbg)) return -3;
        }
        if (nlog > 0) {
            head_d(T, nlog);
            for (int a = 0; a < nlog; a += 8) {
                int n = nlog - a < 8 ? nlog - a : 8;
                head_part(a, n);
                CK(cudaStreamSynchronize(st));
                memcpy(logits + (size_t)a * VOCAB, G.hlog, (size_t)n * VOCAB * 4);
            }
        }
        CK(cudaStreamSynchronize(st));
        if (tp) TP.head[T] += wall() - tpa;
        if (G.prof && T == 1) {
            for (int l = 0; l < NL; l++)
                for (int k = 0; k < 11; k++) {
                    float ms = 0;
                    cudaEventElapsedTime(&ms, G.pev[l][k], G.pev[l][k + 1]);
                    G.pt[k] += ms;
                }
            if (++G.pn % 16 == 0) {
                fprintf(stderr, "dsv41_gpu prof (ms per token, %ld tokens): host wait %.2f, host experts %.2f, hot hits %.1f%% |",
                        G.pn, G.hwait * 1e3 / G.pn, G.hmoe * 1e3 / G.pn, G.hslot ? 100.0 * G.hhit / G.hslot : 0.0);
                for (int k = 0; k < 11; k++) fprintf(stderr, " %s %.2f", PNAME[k], G.pt[k] / G.pn);
                fprintf(stderr, "\n");
            }
        }
    }
    if (tp) { TP.n[T]++; if (++TP.tot % 256 == 0) dg_tprof_print(); }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "dsv41_gpu: %s\n", cudaGetErrorString(e)); return -3; }
    return 0;
}

/* chunks A (positions pos .. pos + T - 1) and B (pos + T .. pos + 2T - 1); logits of B's last token if nlog. After
   it, G holds B's activations (B runs in context 0, A in context 1; the DSpark push reads B's rows). */
int dg_forward2(int pos, int T, int flags, const DgIn* inA, const DgIn* inB, dg_moe_fn moe, float* logits, int nlog) {
    if (T <= HOTT || T > G.tmax || pos + 2 * T > G.cap || pc_init()) return -1;
    size_t oa[5], ob[5];
    pc_use(1); if (pc_upload(pos, T, inA, oa)) return -1;
    k_embed<<<dim3(D / 256, T), 256, 0, st>>>((const float*)(G.din + oa[4]), G.h, G.pre);
    pc_use(0); if (pc_upload(pos + T, T, inB, ob)) return -1;
    k_embed<<<dim3(D / 256, T), 256, 0, st>>>((const float*)(G.din + ob[4]), G.h, G.pre);
    pc_use(1); if (layer_a(0, T, pos, flags, oa[0], oa[1], oa[2], oa[3], inA->nlat[0])) return -3;
    pc_use(0); if (layer_a(0, T, pos + T, flags, ob[0], ob[1], ob[2], ob[3], inB->nlat[0])) return -3;
    for (int l = 0; l < NL; l++) {
        pc_use(1);
        wait_flag(l);
        G.pf_hold = 1;
        if (host_experts(l, T, moe)) return -2;
        G.pf_hold = 0;
        if (layer_b(l, T, NULL)) return -3;
        if (l + 1 < NL && layer_a(l + 1, T, pos, flags, oa[0], oa[1], oa[2], oa[3], inA->nlat[l + 1])) return -3;
        pc_use(0);
        wait_flag(l);
        if (host_experts(l, T, moe)) return -2;
        if (layer_b(l, T, NULL)) return -3;
        if (l + 1 < NL && layer_a(l + 1, T, pos + T, flags, ob[0], ob[1], ob[2], ob[3], inB->nlat[l + 1])) return -3;
    }
    if (nlog > 0) {
        head_d(T, 1);
        head_part(0, 1);
        CK(cudaStreamSynchronize(st));
        memcpy(logits, G.hlog, (size_t)VOCAB * 4);
    }
    CK(cudaStreamSynchronize(st));            /* B ran in context 0: G holds its state (DSpark push, next forward) */
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "dsv41_gpu: %s\n", cudaGetErrorString(e)); return -3; }
    return 0;
}

/* A verify batch as two halves A (ta tokens at pos) and B (tb tokens after them), pipelined like dg_forward2: the
   card runs B's layer l while the host computes A's layer-l experts, and A's layer l + 1 while the host computes B's.
   B's attention at layer l reads A's caches of layer l, written before on the same stream. logits [ta + tb][VOCAB].
   B runs in context 0 (G holds its state after), A in context 1; the mean rows for the DSpark push (mainh) are written
   at rows 0 .. ta - 1 (A) and ta .. (B) of context 0's buffer. */
int dg_forward_split(int pos, int ta, int tb, int flags, const DgIn* inA, const DgIn* inB, dg_moe_fn moe, float* logits) {
    if (ta < 1 || tb < 1 || ta + tb > HOTT || pos + ta + tb > G.cap || pc_init()) return -1;
    if (TP.on < 0) { TP.on = getenv("KINDI_DSV_TPROF") && atoi(getenv("KINDI_DSV_TPROF")) != 0; if (TP.on) atexit(dg_tprof_print); }
    const int T = ta + tb, tp = TP.on == 1;
    double tpa = tp ? wall() : 0, tpb = 0;
    size_t oa[5], ob[5];
    pc_use(0);
    float* mbase = G.mainh;
    float* m1 = PC[1].mainh;
    pc_use(1); G.mainh = mbase;
    if (pc_upload(pos, ta, inA, oa)) return -1;
    k_embed<<<dim3(D / 256, ta), 256, 0, st>>>((const float*)(G.din + oa[4]), G.h, G.pre);
    pc_use(0); G.mainh = mbase + (size_t)ta * 3 * D;
    if (pc_upload(pos + ta, tb, inB, ob)) return -1;
    k_embed<<<dim3(D / 256, tb), 256, 0, st>>>((const float*)(G.din + ob[4]), G.h, G.pre);
    pc_use(1); if (layer_a(0, ta, pos, flags, oa[0], oa[1], oa[2], oa[3], inA->nlat[0])) return -3;
    pc_use(0); if (layer_a(0, tb, pos + ta, flags, ob[0], ob[1], ob[2], ob[3], inB->nlat[0])) return -3;
    for (int l = 0; l < NL; l++) {
        pc_use(1);
        wait_flag(l);
        if (tp) { tpb = wall(); TP.seg[T] += tpb - tpa; }
        if (host_experts(l, ta, moe)) return -2;
        if (tp) { tpa = wall(); TP.host[T] += tpa - tpb; }
        if (layer_b(l, ta, NULL)) return -3;
        if (l + 1 < NL && layer_a(l + 1, ta, pos, flags, oa[0], oa[1], oa[2], oa[3], inA->nlat[l + 1])) return -3;
        pc_use(0);
        wait_flag(l);
        if (tp) { tpb = wall(); TP.seg[T] += tpb - tpa; }
        if (host_experts(l, tb, moe)) return -2;
        if (tp) { tpa = wall(); TP.host[T] += tpa - tpb; }
        if (layer_b(l, tb, NULL)) return -3;
        if (l + 1 < NL && layer_a(l + 1, tb, pos + ta, flags, ob[0], ob[1], ob[2], ob[3], inB->nlat[l + 1])) return -3;
    }
    pc_use(1);
    head_d(ta, ta);
    head_part(0, ta);
    CK(cudaStreamSynchronize(st));
    memcpy(logits, G.hlog, (size_t)ta * VOCAB * 4);
    G.mainh = m1;
    pc_use(0);
    head_d(tb, tb);
    head_part(0, tb);
    CK(cudaStreamSynchronize(st));
    memcpy(logits + (size_t)ta * VOCAB, G.hlog, (size_t)tb * VOCAB * 4);
    G.mainh = mbase;
    if (tp) { TP.head[T] += wall() - tpa; TP.n[T]++; if (++TP.tot % 256 == 0) dg_tprof_print(); }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "dsv41_gpu: %s\n", cudaGetErrorString(e)); return -3; }
    return 0;
}

}  /* extern "C" */

/* ---------------------------------------------------------------- DSpark drafter (G6) */
/* The drafter of the original checkpoint (mtp.0-2): it reads the mean of the 4 stream copies at the attention input of
   layers 37-39 (saved by the main forward, G.mainh), keeps a 128-slot ring of KV made from those of the accepted
   positions, and drafts a block of 5 tokens: [next token, 4 noise tokens] through 3 blocks (attention over the ring
   + the block, non-causal within the block; MoE 128 experts top-3 on the host + a shared expert), the main head, then
   token by token with a Markov bias from the previous token, and a confidence logit per position. Its output only
   proposes tokens: the main model verifies every one. */
#define DS_STAGES 3
#define DS_BLOCK 5
#define DS_NE 128
#define DS_NS 3
#define DS_WIN 128
#define DS_MR 256                 /* Markov rank */

typedef int (*dg_dsmoe_fn)(int stage, int T, const float* x, const int* sel, const float* wsel, float* mo);

typedef struct {
    int m_qa, m_qb, m_kv, m_oa, m_ob, m_shg, m_shu, m_shd, m_gate;
    float *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias, *hc_attn_fn, *hc_attn_base, *hc_attn_scale,
        *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
    float* ring;                  /* [DS_WIN][HD] */
} DSStage;

static struct {
    int on, m_proj;
    DSStage S[DS_STAGES];
    float *main_norm, *norm, *conf, *membed, *mhead;
    __nv_bfloat16* mhead16;      /* the Markov head as BF16 (exact), else mhead (F32) */
    float *mainx, *kvb, *xpre, *sc, *me; int* tok_d;
    int8_t* mq; float* md;
    float* cs_d; float* cs_h;     /* RoPE (window frequencies) of the pushed / block positions */
    int *posd, *posh;
    dg_dsmoe_fn moe;
    int* hout; float* hconf;
} DS;

/* kv rows -> rms norm, RoPE at position pos[t], FP8 rounding; into ring[pos % DS_WIN] (ring) or out[t] */
__global__ void k_ds_kv(const float* kvw, const float* kvnorm, const float* cs, const int* pos, float* ring, float* out,
                        int flags) {
    int t = blockIdx.x, d = threadIdx.x;
    __shared__ float v[HD];
    const float* x = kvw + (size_t)t * HD;
    double s = (double)x[d] * (double)x[d];
    s = block_sumd(s);
    float rr = 1.f / sqrtf((float)(s / HD) + 1e-20f);
    v[d] = kvnorm[d] * (x[d] * rr);
    __syncthreads();
    if (d < RD / 2) {
        const float* c2 = cs + (size_t)t * RD;
        float c = c2[2 * d], sn = c2[2 * d + 1];
        float x0 = v[HD - RD + 2 * d], x1 = v[HD - RD + 2 * d + 1];
        v[HD - RD + 2 * d] = x0 * c - x1 * sn;
        v[HD - RD + 2 * d + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
    float y = v[d];
    if (flags & F_FP8_KV) {
        float am = warp_max(fabsf(y));
        float sc = pow2_ceil(fmaxf(am, 1e-4f) * (1.f / 448.f)), is = 1.f / sc;
        y = v_rne_e4m3(fminf(fmaxf(y * is, -448.f), 448.f)) * sc;
    }
    if (ring) ring[(size_t)(pos[t] % DS_WIN) * HD + d] = y;
    else out[(size_t)t * HD + d] = y;
}

/* drafter attention: block per (block token, head): the nr ring slots (storage order) + the DS_BLOCK block entries,
   softmax with the head's sink; o[t][h] */
__global__ void k_ds_attn(const float* q, const float* ring, int nr, const float* kvb, const float* sinks, float* o) {
    const int t = blockIdx.y, h = blockIdx.x, n = nr + DS_BLOCK;
    __shared__ float sc[DS_WIN + DS_BLOCK];
    __shared__ float qs[HD];
    for (int d = threadIdx.x; d < HD; d += blockDim.x) qs[d] = q[((size_t)t * NH + h) * HD + d];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int e = warp; e < n; e += nw) {
        const float* k = e < nr ? ring + (size_t)e * HD : kvb + (size_t)(e - nr) * HD;
        float a = 0.f;
        for (int d = lane; d < HD; d += 32) a += qs[d] * k[d];
        for (int off = 16; off; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
        if (lane == 0) sc[e] = a * (1.f / sqrtf((float)HD));
    }
    __syncthreads();
    __shared__ float smx, ssum;
    if (threadIdx.x == 0) {
        float mx = sinks[h];
        for (int e = 0; e < n; e++) mx = fmaxf(mx, sc[e]);
        float s = expf(sinks[h] - mx);
        for (int e = 0; e < n; e++) { sc[e] = expf(sc[e] - mx); s += sc[e]; }
        smx = mx; ssum = s;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float a = 0.f;
        for (int e = 0; e < n; e++) a += sc[e] * (e < nr ? ring[(size_t)e * HD + d] : kvb[(size_t)(e - nr) * HD + d]);
        o[((size_t)t * NH + h) * HD + d] = a / ssum;
    }
}

/* drafter router: block per token (DS_NE threads), sqrt(softplus), biased top-3, weights normalized x 1.5 */
__global__ void k_ds_route(const float* glog, const float* bias, int* sel, float* wsel) {
    int t = blockIdx.x, e = threadIdx.x;
    __shared__ float bv[DS_NE / 32]; __shared__ int bi[DS_NE / 32];
    __shared__ float ssel[DS_NS]; __shared__ int isel[DS_NS];
    float v = glog[(size_t)t * DS_NE + e];
    float sp = v > 20.f ? v : log1pf(expf(v));
    float se = sqrtf(sp), be = se + bias[e];
    bool used = false;
    for (int k = 0; k < DS_NS; k++) {
        float mv = used ? -INFINITY : be; int mi = used ? 0x7fffffff : e;
        for (int o = 16; o; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, mv, o); int oi = __shfl_xor_sync(0xffffffffu, mi, o);
            if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
        }
        if ((e & 31) == 0) { bv[e >> 5] = mv; bi[e >> 5] = mi; }
        __syncthreads();
        if (e == 0) {
            float m2 = bv[0]; int i2 = bi[0];
            for (int w = 1; w < DS_NE / 32; w++) if (bv[w] > m2 || (bv[w] == m2 && bi[w] < i2)) { m2 = bv[w]; i2 = bi[w]; }
            isel[k] = i2;
        }
        __syncthreads();
        if (isel[k] == e) { used = true; ssel[k] = se; }
        __syncthreads();
    }
    if (e == 0) {
        float sum = 0;
        for (int k = 0; k < DS_NS; k++) { sel[t * DS_NS + k] = isel[k]; wsel[t * DS_NS + k] = ssel[k]; sum += ssel[k]; }
        for (int k = 0; k < DS_NS; k++) wsel[t * DS_NS + k] = wsel[t * DS_NS + k] / (sum + 1e-20f) * 1.5f;
    }
}

/* Markov step i: score[v] = logits[i][v] + mhead[v] . membed[tok[i]]; then the argmax -> tok[i + 1] */
template <typename TH>
__global__ void k_markov(const float* logits, const TH* mhead, const float* membed, const int* tok, int i, float* sc) {
    __shared__ float me[DS_MR];
    const int prev = tok[i];
    for (int k = threadIdx.x; k < DS_MR; k += blockDim.x) me[k] = membed[(size_t)prev * DS_MR + k];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int v = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   /* a warp per vocabulary row */
    if (v >= VOCAB) return;
    const TH* hr = mhead + (size_t)v * DS_MR;
    float a = 0.f;
    for (int k = lane; k < DS_MR; k += 32) a += (float)hr[k] * me[k];
    for (int off = 16; off; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
    (void)warp;
    if (lane == 0) sc[v] = logits[(size_t)i * VOCAB + v] + a;
}
__global__ void k_argmax(const float* sc, int n, int* tok, int i) {
    __shared__ float bv[32]; __shared__ int bi[32];
    float mv = -INFINITY; int mi = 0x7fffffff;
    for (int v = threadIdx.x; v < n; v += blockDim.x) if (sc[v] > mv) { mv = sc[v]; mi = v; }
    for (int o = 16; o; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, mv, o); int oi = __shfl_xor_sync(0xffffffffu, mi, o);
        if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; }
    }
    if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = mv; bi[threadIdx.x >> 5] = mi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) if (bv[w] > mv || (bv[w] == mv && bi[w] < mi)) { mv = bv[w]; mi = bi[w]; }
        mv = bv[0]; mi = bi[0];
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) if (bv[w] > mv || (bv[w] == mv && bi[w] < mi)) { mv = bv[w]; mi = bi[w]; }
        tok[i + 1] = mi;
    }
}
/* confidence logit of block position i: proj . [x_pre[i], membed[tok[i]]] */
__global__ void k_conf(const float* proj, const float* xpre, const float* membed, const int* tok, float* conf) {
    const int i = blockIdx.x;
    float a = 0.f;
    for (int k = threadIdx.x; k < D + DS_MR; k += blockDim.x)
        a += proj[k] * (k < D ? xpre[(size_t)i * D + k] : membed[(size_t)tok[i] * DS_MR + k - D]);
    a = (float)block_sumd((double)a);
    if (threadIdx.x == 0) conf[i] = a;
}

/* ---------------------------------------------------------------- DSpark: host side */
typedef struct {
    const uint8_t *qa, *qb, *kv, *oa, *ob, *shg, *shu, *shd;           /* GGUF Q8_0 rows */
    const float *gate, *sinks, *kv_norm, *q_norm, *attn_norm, *ffn_norm, *bias, *hc_attn_fn, *hc_attn_base, *hc_attn_scale,
        *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
} DSStageDesc;
typedef struct {
    const uint8_t* main_proj;
    const float *main_norm, *norm, *conf, *membed, *mhead;
    DSStageDesc s[DS_STAGES];
} DSDesc;

extern "C" {

int dg_dspark_init(const DSDesc* d, dg_dsmoe_fn moe) {
    DS.m_proj = dg_mat_q8(d->main_proj, D, 3 * D);
    DS.main_norm = dvec(d->main_norm, D); DS.norm = dvec(d->norm, D); DS.conf = dvec(d->conf, D + DS_MR);
    DS.membed = dvec(d->membed, (size_t)VOCAB * DS_MR);
    {   /* the Markov head is BF16 in the checkpoint: kept as BF16 on the card (the same values, half the bytes to read
           per draft position) when every value is exactly a BF16 */
        const size_t n = (size_t)VOCAB * DS_MR;
        int exact = 1;
        for (size_t i = 0; i < n && exact; i++) { uint32_t u; memcpy(&u, d->mhead + i, 4); exact = (u & 0xffff) == 0; }
        if (exact) {
            __nv_bfloat16* hb = (__nv_bfloat16*)malloc(n * 2);
            for (size_t i = 0; i < n; i++) { uint32_t u; memcpy(&u, d->mhead + i, 4); uint16_t h = (uint16_t)(u >> 16); memcpy(hb + i, &h, 2); }
            DS.mhead16 = (__nv_bfloat16*)dz(n * 2);
            if (DS.mhead16) CK(cudaMemcpy(DS.mhead16, hb, n * 2, cudaMemcpyHostToDevice));
            free(hb);
        }
        if (!DS.mhead16) DS.mhead = dvec(d->mhead, n);
    }
    for (int s = 0; s < DS_STAGES; s++) {
        const DSStageDesc* e = &d->s[s];
        DSStage* S = &DS.S[s];
        S->m_qa = dg_mat_q8(e->qa, QR, D); S->m_qb = dg_mat_q8(e->qb, NH * HD, QR); S->m_kv = dg_mat_q8(e->kv, HD, D);
        S->m_oa = dg_mat_q8(e->oa, OG * OR, NH * HD / OG); S->m_ob = dg_mat_q8(e->ob, D, OG * OR);
        S->m_shg = dg_mat_q8(e->shg, FF, D); S->m_shu = dg_mat_q8(e->shu, FF, D); S->m_shd = dg_mat_q8(e->shd, D, FF);
        S->m_gate = dg_mat_f32(e->gate, DS_NE, D);
        S->sinks = dvec(e->sinks, NH); S->kv_norm = dvec(e->kv_norm, HD); S->q_norm = dvec(e->q_norm, QR);
        S->attn_norm = dvec(e->attn_norm, D); S->ffn_norm = dvec(e->ffn_norm, D); S->bias = dvec(e->bias, DS_NE);
        S->hc_attn_fn = dvec(e->hc_attn_fn, (size_t)MIX * HC * D); S->hc_attn_base = dvec(e->hc_attn_base, MIX);
        S->hc_attn_scale = dvec(e->hc_attn_scale, 3);
        S->hc_ffn_fn = dvec(e->hc_ffn_fn, (size_t)MIX * HC * D); S->hc_ffn_base = dvec(e->hc_ffn_base, MIX);
        S->hc_ffn_scale = dvec(e->hc_ffn_scale, 3);
        S->ring = (float*)dz((size_t)DS_WIN * HD * 4);
        if (S->m_qa < 0 || S->m_gate < 0 || !S->ring) return -1;
    }
    DS.mainx = (float*)dz((size_t)DS_WIN * D * 4); DS.kvb = (float*)dz((size_t)DS_BLOCK * HD * 4);
    DS.xpre = (float*)dz((size_t)DS_BLOCK * D * 4); DS.sc = (float*)dz((size_t)VOCAB * 4);
    DS.tok_d = (int*)dz(64); DS.mq = (int8_t*)dz((size_t)DS_WIN * 3 * D); DS.md = (float*)dz((size_t)DS_WIN * 3 * D / QK * 4);
    DS.cs_d = (float*)dz((size_t)DS_WIN * RD * 4); DS.posd = (int*)dz((size_t)DS_WIN * 4);
    CK(cudaMallocHost(&DS.cs_h, (size_t)DS_WIN * RD * 4)); CK(cudaMallocHost(&DS.posh, (size_t)DS_WIN * 4));
    CK(cudaMallocHost(&DS.hout, 64)); CK(cudaMallocHost(&DS.hconf, 64));
    if (!(DS.mhead || DS.mhead16) || !DS.membed || !DS.sc) return -1;
    DS.moe = moe; DS.on = 1;
    return 0;
}

/* the accepted positions pos0 .. pos0 + n - 1 are rows row0 .. row0 + n - 1 of the last forward: their main_x into
   every stage's ring (cs: their window RoPE, [n][RD]) */
int dg_dspark_push(int row0, int n, int pos0, const float* cs, int flags) {
    if (!DS.on || n <= 0) return 0;
    if (n > DS_WIN) { row0 += n - DS_WIN; pos0 += n - DS_WIN; cs += (size_t)(n - DS_WIN) * RD; n = DS_WIN; }
    memcpy(DS.cs_h, cs, (size_t)n * RD * 4);
    for (int i = 0; i < n; i++) DS.posh[i] = pos0 + i;
    CK(cudaMemcpyAsync(DS.cs_d, DS.cs_h, (size_t)n * RD * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(DS.posd, DS.posh, (size_t)n * 4, cudaMemcpyHostToDevice, st));
    k_quant<<<n, 1024, 0, st>>>(G.mainh + (size_t)row0 * 3 * D, 3 * D, 3 * D, DS.mq, DS.md);
    gemv_d(DS.m_proj, DS.mq, DS.md, NULL, 3 * D, 0, 0, n, DS.mainx, D);
    k_rmsnorm<<<n, 1024, 0, st>>>(DS.mainx, D, DS.mainx, D, DS.main_norm, D);
    k_quant<<<n, 1024, 0, st>>>(DS.mainx, D, D, G.xq, G.xd);
    for (int s = 0; s < DS_STAGES; s++) {
        gemv_d(DS.S[s].m_kv, G.xq, G.xd, NULL, D, 0, 0, n, G.kvw, HD);
        k_ds_kv<<<n, HD, 0, st>>>(G.kvw, DS.S[s].kv_norm, DS.cs_d, DS.posd, DS.S[s].ring, NULL, flags);
    }
    CK(cudaStreamSynchronize(st));
    return 0;
}

/* Draft DS_BLOCK tokens after position pos_last (the last accepted position; its main_x is in the rings). y = the
   main model's next token (position pos_last + 1); emb [DS_BLOCK][D] = embeddings of [y, noise x 4]; cs [DS_BLOCK][RD]
   = window RoPE of positions pos_last + 1 .. + DS_BLOCK. out_tok [DS_BLOCK + 1] = y, drafts; conf [DS_BLOCK]. */
int dg_dspark_draft(int pos_last, int y, const float* emb, const float* cs, int flags, int* out_tok, float* conf) {
    if (!DS.on) return -1;
    const int T = DS_BLOCK;
    memcpy(G.hin, emb, (size_t)T * D * 4);
    memcpy(G.hin + (size_t)T * D * 4, cs, (size_t)T * RD * 4);
    DS.hout[0] = y;
    CK(cudaMemcpyAsync(G.din, G.hin, (size_t)T * (D + RD) * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(DS.tok_d, DS.hout, 4, cudaMemcpyHostToDevice, st));
    const float* csd = (const float*)(G.din + (size_t)T * D * 4);
    k_embed<<<dim3(D / 256, T), 256, 0, st>>>((const float*)G.din, G.h, G.pre);
    const int nr = pos_last + 1 < DS_WIN ? pos_last + 1 : DS_WIN;
    for (int s = 0; s < DS_STAGES; s++) {
        const DSStage* S = &DS.S[s];
        hc_in_d(S->hc_attn_fn, S->hc_attn_scale, S->hc_attn_base, G.pre, S->attn_norm, G.apre, T);
        gemv_d(S->m_qa, G.xq, G.xd, NULL, D, 0, 0, T, G.qa, QR);
        k_rmsnorm<<<T, 1024, 0, st>>>(G.qa, QR, G.qa, QR, S->q_norm, QR);
        k_quantm<<<dim3((QR + 255) / 256, T), 256, 0, st>>>(G.qa, QR, QR, G.qaq, G.qad);
        gemv_d(S->m_qb, G.qaq, G.qad, NULL, QR, 0, 0, T, G.q, NH * HD);
        k_rope<<<dim3(T, NH / 8), dim3(RD / 2, 8), 0, st>>>(G.q, NH, HD, csd, RD, 0, 0);
        gemv_d(S->m_kv, G.xq, G.xd, NULL, D, 0, 0, T, G.kvw, HD);
        k_ds_kv<<<T, HD, 0, st>>>(G.kvw, S->kv_norm, csd, NULL, NULL, DS.kvb, flags);
        k_ds_attn<<<dim3(NH, T), 256, 0, st>>>(G.q, S->ring, nr, DS.kvb, S->sinks, G.o);
        k_rope<<<dim3(T, NH / 8), dim3(RD / 2, 8), 0, st>>>(G.o, NH, HD, csd, RD, 0, 1);
        if (flags & F_FP8_O) k_sim_fp8<<<dim3(NH * HD / 256, T), 256, 0, st>>>(G.o, NH * HD, NH * HD);
        k_quantm<<<dim3((NH * HD + 255) / 256, T), 256, 0, st>>>(G.o, NH * HD, NH * HD, G.oq, G.od);
        gemv_d(S->m_oa, G.oq, G.od, NULL, NH * HD, OR, NH * HD / OG, T, G.oa, OG * OR);
        k_quantm<<<dim3((OG * OR + 255) / 256, T), 256, 0, st>>>(G.oa, OG * OR, OG * OR, G.oaq, G.oad);
        gemv_d(S->m_ob, G.oaq, G.oad, NULL, OG * OR, 0, 0, T, G.ao, D);
        k_post<<<dim3(D / 256, T), 256, 0, st>>>(G.h, G.ao, G.post, G.comb);
        hc_in_d(S->hc_ffn_fn, S->hc_ffn_scale, S->hc_ffn_base, G.apre, S->ffn_norm, G.pre, T);
        gemv_d(S->m_gate, NULL, NULL, G.x, D, 0, 0, T, G.glog, DS_NE);
        k_ds_route<<<T, DS_NE, 0, st>>>(G.glog, S->bias, G.sel, G.wsel);
        CK(cudaMemcpyAsync(G.hx, G.x, (size_t)T * D * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(G.hsel, G.sel, (size_t)T * DS_NS * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(G.hwsel, G.wsel, (size_t)T * DS_NS * 4, cudaMemcpyDeviceToHost, st));
        cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming); cudaEventRecord(ev, st);
        gemv_d(S->m_shg, G.xq, G.xd, NULL, D, 0, 0, T, G.sg, FF);
        gemv_d(S->m_shu, G.xq, G.xd, NULL, D, 0, 0, T, G.su, FF);
        k_swiglu<<<ew_grid((size_t)T * FF), 256, 0, st>>>(G.sg, G.su, (size_t)T * FF);
        k_quantm<<<dim3((FF + 255) / 256, T), 256, 0, st>>>(G.sg, FF, FF, G.suq, G.sud);
        gemv_d(S->m_shd, G.suq, G.sud, NULL, FF, 0, 0, T, G.fo, D);
        CK(cudaEventSynchronize(ev)); cudaEventDestroy(ev);
        if (DS.moe(s, T, G.hx, G.hsel, G.hwsel, G.hmo)) return -2;
        CK(cudaMemcpyAsync(G.mo, G.hmo, (size_t)T * D * 4, cudaMemcpyHostToDevice, st));
        k_add<<<ew_grid((size_t)T * D), 256, 0, st>>>(G.fo, G.mo, (size_t)T * D);
        k_post<<<dim3(D / 256, T), 256, 0, st>>>(G.h, G.fo, G.post, G.comb);
    }
    k_collapse0<<<dim3(D / 256, T), 256, 0, st>>>(G.h, G.pre, DS.xpre);
    k_rmsnorm<<<T, 1024, 0, st>>>(G.x, D, DS.xpre, D, DS.norm, D);
    k_quant<<<T, 1024, 0, st>>>(G.x, D, D, G.lq, G.ld);
    gemv_d(G.m_out, G.lq, G.ld, NULL, D, 0, 0, T, G.lg, VOCAB);
    for (int i = 0; i < T; i++) {
        if (DS.mhead16) k_markov<<<(VOCAB * 32 + 255) / 256, 256, 0, st>>>(G.lg, DS.mhead16, DS.membed, DS.tok_d, i, DS.sc);
        else k_markov<<<(VOCAB * 32 + 255) / 256, 256, 0, st>>>(G.lg, DS.mhead, DS.membed, DS.tok_d, i, DS.sc);
        k_argmax<<<1, 1024, 0, st>>>(DS.sc, VOCAB, DS.tok_d, i);
    }
    float* confd = DS.sc;                    /* reuse: T floats after the last argmax */
    k_conf<<<T, 256, 0, st>>>(DS.conf, DS.xpre, DS.membed, DS.tok_d, confd);
    CK(cudaMemcpyAsync(DS.hout, DS.tok_d, (T + 1) * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaMemcpyAsync(DS.hconf, confd, T * 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    memcpy(out_tok, DS.hout, (T + 1) * 4);
    memcpy(conf, DS.hconf, T * 4);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "dsv41_gpu dspark: %s\n", cudaGetErrorString(e)); return -3; }
    return 0;
}

}  /* extern "C" */

/* ---------------------------------------------------------------- prompt cache: state checkpoints (G8) */
/* The state that the positions 0 .. pos - 1 leave on the card: every layer's window ring, the compressed KV and index
   keys of the kv source layers up to pos / ratio, the compressor input rings and the drafter's rings. Saved to / loaded
   from host memory (the host writes it to disk with the token map); everything is indexed by position, so a restored
   state continues exactly as the original. */
extern "C" {

static size_t state_parts(int pos, void* host, int load) {
    size_t off = 0;
    uint8_t* h = (uint8_t*)host;
#define PART(dptr, bytes) do { size_t b_ = (bytes); if (h) { if (load) cudaMemcpy((dptr), h + off, b_, cudaMemcpyHostToDevice); \
                                                         else cudaMemcpy(h + off, (dptr), b_, cudaMemcpyDeviceToHost); } off += b_; } while (0)
    for (int l = 0; l < NL; l++) {
        DLayer* L = &G.L[l];
        PART(L->win, (size_t)G.W * HD * 4);
        if (L->h.kv_src) {
            const size_t ng = (size_t)pos / L->h.ratio + 1;
            PART(L->ckv, ng * HD * 4);
            PART(L->ik, ng * IHD * 4);
            PART(L->cst_kv, (size_t)CRING * HD * 4);
            PART(L->cst_sc, (size_t)CRING * HD * 4);
        }
    }
    if (DS.on)
        for (int s = 0; s < DS_STAGES; s++) PART(DS.S[s].ring, (size_t)DS_WIN * HD * 4);
#undef PART
    return off;
}

size_t dg_state_size(int pos) { return state_parts(pos, NULL, 0); }
int dg_state_save(int pos, void* host) { CK(cudaStreamSynchronize(st)); state_parts(pos, host, 0); return 0; }
int dg_state_load(int pos, const void* host) { CK(cudaStreamSynchronize(st)); state_parts(pos, (void*)host, 1); CK(cudaDeviceSynchronize()); return 0; }

}  /* extern "C" */
