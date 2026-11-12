/* lightning.cu - Kindi kernels for Nemotron-H MoE GGUF files (Nemotron 3.5 Lightning 30B-A3B: Mamba-2 mixers, MoE
   feed-forward layers with relu^2 experts, attention layers without positions; Q2_0 and Q8_0 weights).

   Numbers. Every matrix product with a quantized weight takes its input quantized to int8 in runs of 32 (scale =
   max |x| / 127, q = roundf(x / scale)), as nbq.cu does. A run gives an exact int32 dot product, which is scaled in
   fp32. Q2_0 weights are ternary (codes 0, 1, 2 = -1, 0, +1 times the fp16 block scale of 64 weights), so a run is
   sum(c * q) - sum(q). The router (f32 weights), the Mamba-2 scan, the norms and attention are fp32; the K/V cache
   is fp16.

   Layouts (lightning.py repacks the file blocks at load time):
   - Q2_0: codes as u32 words, row-major (word w of a row = weights 16w .. 16w+15, weight 16w+i at bits 2i), and
     the fp16 block scales in their own array [row][cols / 64].
   - Q8_0 (output head): the int8 weights in the "lane order" below, and the fp16 scales in their own array.
   - Quantized activations (xq): per run of 32, eight u32 words in lane order: in each group of 16 values, word k
     holds the values 4j + k (j = 0..3) as bytes j. Then (code word >> 2k) & 0x03030303 meets word k in one dp4a.
     dx = the run scales (f32), sx = the sum of q in each run (int).

   Cards before sm_61 have no dp4a: lt_dp4a is then four byte products (the same exact sums).
   The decode kernels read the position and the token from device memory, so one CUDA graph replays every step. */
#include <cuda_fp16.h>

typedef unsigned int u32;
typedef unsigned char u8;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int lt_dp4a(int a, int b, int c) {
    return c + (a << 24 >> 24) * (b << 24 >> 24) + (a << 16 >> 24) * (b << 16 >> 24) + (a << 8 >> 24) * (b << 8 >> 24)
             + (a >> 24) * (b >> 24);
}
#else
__device__ __forceinline__ int lt_dp4a(int a, int b, int c) {
    int r; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
#endif

#define LT_M 0x03030303u
#define INFINITY __int_as_float(0x7f800000)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int warp_isum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* sum(c * q) over one run: the two code words of 32 ternary weights, the eight lane-order words of the run */
__device__ __forceinline__ int lt_run_q2(u32 w0, u32 w1, const int* q) {
    int s = 0;
    s = lt_dp4a((int)(w0 & LT_M), q[0], s);
    s = lt_dp4a((int)((w0 >> 2) & LT_M), q[1], s);
    s = lt_dp4a((int)((w0 >> 4) & LT_M), q[2], s);
    s = lt_dp4a((int)((w0 >> 6) & LT_M), q[3], s);
    s = lt_dp4a((int)(w1 & LT_M), q[4], s);
    s = lt_dp4a((int)((w1 >> 2) & LT_M), q[5], s);
    s = lt_dp4a((int)((w1 >> 4) & LT_M), q[6], s);
    s = lt_dp4a((int)((w1 >> 6) & LT_M), q[7], s);
    return s;
}

/* one warp quantizes one run of 32 values: lane i holds value i. Writes the lane-order words, the scale, the sum. */
__device__ __forceinline__ void lt_quant_run(float v, int lane, int* xq, float* dx, int* sx) {
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : (int)roundf(v / d);
    const int s = warp_isum(q);
    /* word L (L < 8): group g = L / 4, k = L % 4; byte j = value g * 16 + 4j + k */
    const int g = (lane >> 2) & 1, k = lane & 3;
    u32 w = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int qj = __shfl_sync(0xffffffffu, q, g * 16 + 4 * j + k);
        w |= ((u32)qj & 255u) << (8 * j);
    }
    if (lane < 8) xq[lane] = (int)w;
    if (lane == 0) { *dx = d; *sx = s; }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* embedding, norms and quantization                                                                               */

/* x[t] = row tok[t] of the Q8_0 embedding (file blocks: fp16 d, 32 int8), for T tokens. The table may be in host
   memory (mapped): the row is read in 8-byte pieces into shared memory first (cols / 32 * 34 bytes, a multiple of 8
   when cols % 256 == 0; cols <= 4096) */
extern "C" __global__ void lt_embed(const u8* __restrict__ emb, const int* __restrict__ tok, int cols, float* x) {
    __shared__ uint2 rb[4096 / 32 * 34 / 8];
    const int t = blockIdx.x;
    const int nb = cols / 32, n8 = nb * 34 / 8;
    const uint2* row = reinterpret_cast<const uint2*>(emb + (long long)tok[t] * nb * 34);
    for (int i = threadIdx.x; i < n8; i += blockDim.x) rb[i] = row[i];
    __syncthreads();
    const u8* b0 = reinterpret_cast<const u8*>(rb);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const u8* b = b0 + (i >> 5) * 34;
        const float d = __half2float(__ushort_as_half((unsigned short)(b[0] | (b[1] << 8))));
        x[(long long)t * cols + i] = d * (float)(signed char)b[2 + (i & 31)];
    }
}

/* RMS norm of x[t] (eps) times w, quantized for the next matrix product; xn (if not NULL) gets the fp32 values.
   grid (T, parts), 256 threads: every block of a token sums the row (the same order), then quantizes its share of
   the runs (more blocks: less latency for one token). cols % 32 == 0 */
extern "C" __global__ void lt_rms_quant(const float* __restrict__ x, const float* __restrict__ w, int cols, float eps,
                                        float* xn, int* xq, float* dx, int* sx) {
    __shared__ float red[8];
    const int t = blockIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const float* xr = x + (long long)t * cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += 256) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    if (lane == 0) red[wp] = ss;
    __syncthreads();
    ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += red[i];
    const float inv = rsqrtf(ss / (float)cols + eps);
    const int runs = cols >> 5;
    for (int r = blockIdx.y * 8 + wp; r < runs; r += 8 * gridDim.y) {
        const int i = r * 32 + lane;
        const float v = xr[i] * inv * w[i];
        if (xn) xn[(long long)t * cols + i] = v;
        lt_quant_run(v, lane, xq + ((long long)t * runs + r) * 8, dx + (long long)t * runs + r, sx + (long long)t * runs + r);
    }
}

/* quantize rows of x (rows x cols, row stride ld floats); relu2 = 1: max(x, 0)^2 first (the expert activation).
   One warp per run */
extern "C" __global__ void lt_quant(const float* __restrict__ x, int rows, int cols, int ld, int relu2,
                                    int* xq, float* dx, int* sx) {
    const int lane = threadIdx.x & 31;
    const int runs = cols >> 5;
    const long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= (long long)rows * runs) return;
    const long long row = r / runs, run = r % runs;
    float v = x[row * ld + run * 32 + lane];
    if (relu2) { v = fmaxf(v, 0.0f); v = v * v; }
    lt_quant_run(v, lane, xq + r * 8, dx + r, sx + r);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* decode matrix-vector products (one token)                                                                        */

/* y = W x for Q2_0 W, for one or more slots (blockIdx.y). Each slot's matrix is codes[slot] / sc[slot] (pointer
   tables: the experts of a layer, device or mapped host memory), chosen by ids[slot] when ids is not NULL. The
   input of slot s is xq + s * xs (xs = 0: one input for all); the output is y + s * ys. acc = 1: y += W x. */
extern "C" __global__ void __launch_bounds__(128) lt_mv_q2(const u32* const* __restrict__ codes,
                                    const __half* const* __restrict__ sc, const int* __restrict__ ids, int rows,
                                    int cols, const int* __restrict__ xq, const float* __restrict__ dx,
                                    const int* __restrict__ sx, int xs, float* y, int ys, int acc, int rlast) {
    /* 4 warps, 2 rows per warp (8 rows per block): each lane reads its runs of the input once (through the cache,
       no shared-memory staging) for both rows; the sums per row are in the same order as before (lane-strided runs,
       then the warp tree). rlast > 0: the last slot has rlast rows (the shared expert after the routed ones). */
    const int slot = blockIdx.y, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    if (rlast > 0 && slot == gridDim.y - 1) rows = rlast;
    const int runs = cols >> 5, nb = cols >> 6;
    const int r0 = blockIdx.x * 8 + wp * 2;
    if (r0 >= rows) return;
    const bool two = r0 + 1 < rows;
    const int* xqs = xq + (long long)slot * xs * 8;
    const float* dxs = dx + (long long)slot * xs;
    const int* sxs = sx + (long long)slot * xs;
    const int e = ids ? ids[slot] : slot;
    const uint2* w0 = reinterpret_cast<const uint2*>(codes[e]) + (long long)r0 * runs;
    const uint2* w1 = two ? w0 + runs : w0;
    const __half* s0 = sc[e] + (long long)r0 * nb;
    const __half* s1 = two ? s0 + nb : s0;
    float a0 = 0.0f, a1 = 0.0f;
    for (int h = lane; h < runs; h += 32) {
        const uint2 c0 = __ldg(w0 + h), c1 = __ldg(w1 + h);
        const float f0 = __half2float(s0[h >> 1]), f1 = __half2float(s1[h >> 1]);
        const int4 qa = __ldg(reinterpret_cast<const int4*>(xqs + h * 8));
        const int4 qb = __ldg(reinterpret_cast<const int4*>(xqs + h * 8) + 1);
        const int q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        const float dd = __ldg(dxs + h);
        const int ss = __ldg(sxs + h);
        a0 += f0 * dd * (float)(lt_run_q2(c0.x, c0.y, q) - ss);
        a1 += f1 * dd * (float)(lt_run_q2(c1.x, c1.y, q) - ss);
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    if (lane == 0) {
        float* o = y + (long long)slot * ys + r0;
        o[0] = acc ? o[0] + a0 : a0;
        if (two) o[1] = acc ? o[1] + a1 : a1;
    }
}

/* MoE down projection and combine for one token: x[r] += sum_s gw[s] * (W_ids[s] h_s)[r] + (Wsh hsh)[r]. */
extern "C" __global__ void __launch_bounds__(128) lt_mv_q2_down(const u32* const* __restrict__ codes,
                                         const __half* const* __restrict__ sc, const int* __restrict__ ids,
                                         const float* __restrict__ gw, int nslot, int rows, int cols,
                                         const int* __restrict__ hq, const float* __restrict__ hd,
                                         const int* __restrict__ hs, const u32* __restrict__ csh,
                                         const __half* __restrict__ ssh, int colsh, const int* __restrict__ gq,
                                         const float* __restrict__ gd, const int* __restrict__ gs, float* x) {
    /* one warp per row, 4 rows per block; the inputs are read through the cache (no shared-memory staging), and the
       nslot experts' words of a lane's run are loaded together before they are used. nslot <= 8. */
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int r = blockIdx.x * 4 + wp;
    if (r >= rows) return;
    const int runs = cols >> 5, nb = cols >> 6, rsh = colsh >> 5;
    const uint2* wr[8];
    const __half* sr[8];
    float g[8], a[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
        if (k < nslot) {
            const int e = ids[k];
            wr[k] = reinterpret_cast<const uint2*>(codes[e]) + (long long)r * runs;
            sr[k] = sc[e] + (long long)r * nb;
            g[k] = gw[k];
        }
        a[k] = 0.0f;
    }
    for (int h = lane; h < runs; h += 32) {
        uint2 w[8];
        float f[8];
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (k < nslot) { w[k] = __ldg(wr[k] + h); f[k] = __half2float(sr[k][h >> 1]); }
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (k < nslot) {
                const int i = k * runs + h;
                const int4 qa = __ldg(reinterpret_cast<const int4*>(hq + i * 8));
                const int4 qb = __ldg(reinterpret_cast<const int4*>(hq + i * 8) + 1);
                const int q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                a[k] += f[k] * __ldg(hd + i) * (float)(lt_run_q2(w[k].x, w[k].y, q) - __ldg(hs + i));
            }
    }
    float b = 0.0f;
    const uint2* wsh = reinterpret_cast<const uint2*>(csh) + (long long)r * rsh;
    const __half* ssr = ssh + (long long)r * (colsh >> 6);
    for (int h = lane; h < rsh; h += 32) {
        const uint2 w = __ldg(wsh + h);
        const int4 qa = __ldg(reinterpret_cast<const int4*>(gq + h * 8));
        const int4 qb = __ldg(reinterpret_cast<const int4*>(gq + h * 8) + 1);
        const int q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        b += __half2float(ssr[h >> 1]) * __ldg(gd + h) * (float)(lt_run_q2(w.x, w.y, q) - __ldg(gs + h));
    }
    float tot = b;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (k < nslot) tot += g[k] * a[k];
    tot = warp_sum(tot);
    if (lane == 0) x[r] += tot;
}

/* logits = W x for the Q8_0 output head (lane-order int8 weights, fp16 scales per run). One warp per row. */
extern "C" __global__ void lt_mv_q8(const int* __restrict__ wq, const __half* __restrict__ sc, int rows, int cols,
                                    const int* __restrict__ xq, const float* __restrict__ dx, float* y) {
    extern __shared__ int smem[];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int runs = cols >> 5;
    int* q = smem;
    float* d = (float*)(q + runs * 8);
    for (int i = threadIdx.x; i < runs * 8; i += blockDim.x) q[i] = xq[i];
    for (int i = threadIdx.x; i < runs; i += blockDim.x) d[i] = dx[i];
    __syncthreads();
    const int r = blockIdx.x * 8 + wp;
    if (r >= rows) return;
    const int4* wr = reinterpret_cast<const int4*>(wq) + (long long)r * runs * 2;
    const __half* sr = sc + (long long)r * runs;
    float a = 0.0f;
    for (int h = lane; h < runs; h += 32) {
        const int4 w0 = wr[2 * h], w1 = wr[2 * h + 1];
        const int* qq = q + h * 8;
        int v = lt_dp4a(w0.x, qq[0], 0);
        v = lt_dp4a(w0.y, qq[1], v); v = lt_dp4a(w0.z, qq[2], v); v = lt_dp4a(w0.w, qq[3], v);
        v = lt_dp4a(w1.x, qq[4], v); v = lt_dp4a(w1.y, qq[5], v); v = lt_dp4a(w1.z, qq[6], v); v = lt_dp4a(w1.w, qq[7], v);
        a += __half2float(sr[h]) * d[h] * (float)v;
    }
    a = warp_sum(a);
    if (lane == 0) y[r] = a;
}

/* greedy token: the first index of the largest logit. Part 1: one block per 4096 logits */
extern "C" __global__ void lt_argmax_part(const float* __restrict__ y, int n, float* pv, int* pi) {
    __shared__ float sv[8];
    __shared__ int si[8];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = blockIdx.x * 4096 + threadIdx.x; i < min(n, (blockIdx.x + 1) * 4096); i += 256)
        if (y[i] > bv) { bv = y[i]; bi = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffffu, bv, o);
        const int i = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    if (lane == 0) { sv[wp] = bv; si[wp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        bv = sv[0]; bi = si[0];
        for (int k = 1; k < 8; k++)
            if (sv[k] > bv || (sv[k] == bv && si[k] < bi)) { bv = sv[k]; bi = si[k]; }
        pv[blockIdx.x] = bv; pi[blockIdx.x] = bi;
    }
}

/* part 2: one warp over the parts; the token goes to out[0] and, when next is not NULL, to next[0] (the input of
   the next step) */
extern "C" __global__ void lt_argmax_final(const float* __restrict__ pv, const int* __restrict__ pi, int n, int* out,
                                           int* next) {
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += 32)
        if (pv[i] > bv || (pv[i] == bv && pi[i] < bi)) { bv = pv[i]; bi = pi[i]; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v = __shfl_xor_sync(0xffffffffu, bv, o);
        const int i = __shfl_xor_sync(0xffffffffu, bi, o);
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    if (threadIdx.x == 0) { out[0] = bi; if (next) next[0] = bi; }
}

/* sampling candidates after a step: every logit within w of the largest (the argmax parts give it) is appended as
   (index, value) to out (at most cap; cnt[0] = how many qualified, which can exceed cap). The order of the list is
   not fixed; the host sorts it. One pass over the logits, 256 threads per block. */
extern "C" __global__ void lt_cand(const float* __restrict__ y, int n, const float* __restrict__ pv, int np, float w,
                                   int cap, int* cnt, int* oi, float* ov) {
    float m = -INFINITY;
    for (int i = 0; i < np; i++) m = fmaxf(m, pv[i]);
    const float lo = m - w;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (y[i] >= lo) {
            const int k = atomicAdd(cnt, 1);
            if (k < cap) { oi[k] = i; ov[k] = y[i]; }
        }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* MoE routing                                                                                                      */

/* router logits for T tokens: lg[t][e] = Wr[e] . xn[t] (f32). One warp per (expert, 8 tokens): each lane reads a
   weight piece once for the 8 tokens. Every logit sums the same terms in the same order for any T (lane-strided
   float4 pieces, then the warp tree), so a prompt and a step give the same bits. */
extern "C" __global__ void lt_router(const float* __restrict__ wr, const float* __restrict__ xn, int ne, int cols,
                                     int T, float* lg) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int ng = (T + 7) / 8;
    if (i >= ne * ng) return;
    const int e = i % ne, t0 = (i / ne) * 8, nt = min(8, T - t0);
    const float4* a = reinterpret_cast<const float4*>(wr + (long long)e * cols);
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = lane; k < cols / 4; k += 32) {
        const float4 u = a[k];
#pragma unroll
        for (int j = 0; j < 8; j++)
            if (j < nt) {
                const float4 v = reinterpret_cast<const float4*>(xn + (long long)(t0 + j) * cols)[k];
                s[j] += u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
            }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float v = warp_sum(s[j]);
        if (lane == 0 && j < nt) lg[(long long)(t0 + j) * ne + e] = v;
    }
}

/* top-k experts per token: score = sigmoid(logit); choice = score + bias (ties: the lower index); weights = the
   chosen scores, normalized to sum 1 and times scale. One warp per token (no block barriers): lane l holds experts
   l, l + 32, ... (ne <= 1024, a multiple of 32; k <= 16) */
extern "C" __global__ void lt_topk(const float* __restrict__ lg, const float* __restrict__ bias, int ne, int k,
                                   float scale, int* ids, float* gw) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const int per = ne >> 5;
    float sc[32], ch[32];
    for (int j = 0; j < per; j++) {
        const int e = lane + 32 * j;
        sc[j] = 1.0f / (1.0f + __expf(-lg[t * ne + e]));
        ch[j] = sc[j] + bias[e];
    }
    float pick[16];
    float tot = 0.0f;
    for (int r = 0; r < k; r++) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bj = 0;
        for (int j = 0; j < per; j++)
            if (ch[j] > bv) { bv = ch[j]; bi = lane + 32 * j; bj = j; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v = __shfl_xor_sync(0xffffffffu, bv, o);
            const int i = __shfl_xor_sync(0xffffffffu, bi, o);
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        float s = 0.0f;
        if ((bi & 31) == lane) { s = sc[bi >> 5]; ch[bi >> 5] = -INFINITY; }
        pick[r] = __shfl_sync(0xffffffffu, s, bi & 31);
        if (lane == 0) ids[t * k + r] = bi;
        (void)bj;
    }
    if (lane == 0) {
        for (int r = 0; r < k; r++) tot += pick[r];
        for (int r = 0; r < k; r++) gw[t * k + r] = pick[r] / tot * scale;
    }
}

/* decode, all experts on the card: the router logits, then top-k in the block that finishes last (a counter, reset by
   that block); ids gets the k experts and, at ids[k], ne (the shared expert's entry in the up table). 64 threads per
   block, one warp per expert, T = 1. */
extern "C" __global__ void lt_router_topk(const float* __restrict__ wr, const float* __restrict__ xn, int ne, int cols,
                                          float* lg, const float* __restrict__ bias, int k, float scale, int* ids,
                                          float* gw, int* counter) {
    __shared__ int last;
    const int lane = threadIdx.x & 31;
    const int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (e < ne) {
        const float4* a = reinterpret_cast<const float4*>(wr + (long long)e * cols);
        const float4* b = reinterpret_cast<const float4*>(xn);
        float s = 0.0f;
        for (int i = lane; i < cols / 4; i += 32) {
            const float4 u = a[i], v = b[i];
            s += u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
        }
        s = warp_sum(s);
        if (lane == 0) lg[e] = s;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter, 1) == gridDim.x - 1;
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    const int per = ne >> 5;
    float sc[32], ch[32], pick[16];
    for (int j = 0; j < per; j++) {
        const int x = lane + 32 * j;
        sc[j] = 1.0f / (1.0f + __expf(-((volatile float*)lg)[x]));
        ch[j] = sc[j] + bias[x];
    }
    for (int r = 0; r < k; r++) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int j = 0; j < per; j++)
            if (ch[j] > bv) { bv = ch[j]; bi = lane + 32 * j; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float v = __shfl_xor_sync(0xffffffffu, bv, o);
            const int i = __shfl_xor_sync(0xffffffffu, bi, o);
            if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
        }
        float s = 0.0f;
        if ((bi & 31) == lane) { s = sc[bi >> 5]; ch[bi >> 5] = -INFINITY; }
        pick[r] = __shfl_sync(0xffffffffu, s, bi & 31);
        if (lane == 0) ids[r] = bi;
    }
    if (lane == 0) {
        float tot = 0.0f;
        for (int r = 0; r < k; r++) tot += pick[r];
        for (int r = 0; r < k; r++) gw[r] = pick[r] / tot * scale;
        ids[k] = ne;
        *counter = 0;
    }
}

/* quantize two row sets in one launch (relu2 for both): rows1 x cols1 at x1, then rows2 x cols2 at x2 */
extern "C" __global__ void lt_quant2(const float* __restrict__ x1, int rows1, int cols1, int* q1, float* d1, int* s1,
                                     const float* __restrict__ x2, int rows2, int cols2, int* q2, float* d2, int* s2) {
    const int lane = threadIdx.x & 31;
    const long long r = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long n1 = (long long)rows1 * (cols1 >> 5), n2 = (long long)rows2 * (cols2 >> 5);
    if (r >= n1 + n2) return;
    const bool one = r < n1;
    const long long i = one ? r : r - n1;
    const int runs = (one ? cols1 : cols2) >> 5;
    float v = (one ? x1 : x2)[(i / runs) * (one ? cols1 : cols2) + (i % runs) * 32 + lane];
    v = fmaxf(v, 0.0f);
    v = v * v;
    lt_quant_run(v, lane, (one ? q1 : q2) + i * 8, (one ? d1 : d2) + i, (one ? s1 : s2) + i);
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* Mamba-2 mixer                                                                                                    */

/* causal depthwise conv (kernel 4) with bias and SiLU over T tokens for the xBC channels of zxbcdt (row stride ld,
   channel offset c0); cs = the last 3 inputs per channel (oldest first), updated. One thread per channel. */
extern "C" __global__ void lt_conv(const float* __restrict__ zx, int ld, int c0, int nch, int T,
                                   const float* __restrict__ w, const float* __restrict__ b, float* cs, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nch) return;
    float s0 = cs[c * 3], s1 = cs[c * 3 + 1], s2 = cs[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3], bb = b[c];
    for (int t = 0; t < T; t++) {
        const float v = zx[(long long)t * ld + c0 + c];
        const float y = bb + w0 * s0 + w1 * s1 + w2 * s2 + w3 * v;
        out[(long long)t * nch + c] = y / (1.0f + __expf(-y));
        s0 = s1; s1 = s2; s2 = v;
    }
    cs[c * 3] = s0; cs[c * 3 + 1] = s1; cs[c * 3 + 2] = s2;
}

/* the selective scan over T tokens; grid (heads, 64 / (blockDim / 4)): a block takes blockDim / 4 rows of a head (head
   dim P = 64, state N = 128, ngroup groups):
   dt = softplus(dt_raw + dt_bias), dA = exp(dt * A); S = S * dA + dt * x B^T; y = S C + D x; y *= silu(z).
   Thread (p = tid / 4, q = tid % 4) holds the 32 states n = q + 4 j of row p. xbc rows: x (H*P), B, C (ngroup*N
   each); zx rows (stride ld): z at 0, dt at dt0. y rows: H*P. */
extern "C" __global__ void lt_ssm(const float* __restrict__ xbc, const float* __restrict__ zx, int ld, int dt0,
                                  int T, int H, int ngroup, const float* __restrict__ A, const float* __restrict__ D,
                                  const float* __restrict__ dtb, float* st, float* y) {
    __shared__ float sb[128], scv[128];
    const int h = blockIdx.x, p = blockIdx.y * (blockDim.x >> 2) + (threadIdx.x >> 2), q = threadIdx.x & 3;
    const int g = h / (H / ngroup);
    const int xw = H * 64, bw = ngroup * 128;
    float* sp = st + ((long long)h * 64 + p) * 128;
    float s[32];
#pragma unroll
    for (int j = 0; j < 32; j++) s[j] = sp[q + 4 * j];
    const float a = A[h], dd = D[h], db = dtb[h];
    for (int t = 0; t < T; t++) {
        const float* row = xbc + (long long)t * (xw + 2 * bw);
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            if (i < 128) sb[i] = row[xw + g * 128 + i];
            else scv[i - 128] = row[xw + bw + g * 128 + i - 128];
        }
        __syncthreads();
        float dt = zx[(long long)t * ld + dt0 + h] + db;
        dt = dt > 20.0f ? dt : log1pf(__expf(dt));
        const float dA = __expf(dt * a);
        const float xv = row[h * 64 + p];
        const float dx = dt * xv;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; j++) {
            s[j] = s[j] * dA + dx * sb[q + 4 * j];
            acc += s[j] * scv[q + 4 * j];
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (q == 0) {
            const float z = zx[(long long)t * ld + h * 64 + p];
            const float v = acc + dd * xv;
            y[(long long)t * xw + h * 64 + p] = v * (z / (1.0f + __expf(-z)));
        }
    }
#pragma unroll
    for (int j = 0; j < 32; j++) sp[q + 4 * j] = s[j];
}

/* grouped RMS norm (groups of gs values; eps) times w, quantized for out_proj. grid (T, groups), 128 threads: a block
   takes one group of one token (gs % 32 == 0, gs <= 1024) */
extern "C" __global__ void lt_gnorm_quant(const float* __restrict__ y, const float* __restrict__ w, int cols, int gs,
                                          float eps, int* xq, float* dx, int* sx) {
    __shared__ float red[4];
    const int t = blockIdx.x, g = blockIdx.y, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const float* yr = y + (long long)t * cols + g * gs;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < gs; i += 128) ss += yr[i] * yr[i];
    ss = warp_sum(ss);
    if (lane == 0) red[wp] = ss;
    __syncthreads();
    const float inv = rsqrtf((red[0] + red[1] + red[2] + red[3]) / (float)gs + eps);
    const int runs = cols >> 5;
    for (int r = g * (gs >> 5) + wp; r < (g + 1) * (gs >> 5); r += 4) {
        const int i = r * 32 + lane;
        const float v = y[(long long)t * cols + i] * inv * w[i];
        lt_quant_run(v, lane, xq + ((long long)t * runs + r) * 8, dx + (long long)t * runs + r, sx + (long long)t * runs + r);
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* attention (no positions; GQA: 16 query heads per K/V head, head dim 128; fp16 K/V cache [pos][kvh * 128])       */

#define LT_HD 128
#define LT_KP 130                                            /* padded K/V tile rows (halves): no bank conflicts */

/* store the K and V rows of T tokens at positions pos0 .. pos0 + T - 1 (pos0 = *ppos + off). qkv rows: q (nh * 128),
   k (nkv * 128), v (nkv * 128) */
extern "C" __global__ void lt_kv_store(const float* __restrict__ qkv, int ld, int nh, int nkv, const int* __restrict__ ppos,
                                       int off, __half* kc, __half* vc) {
    const int t = blockIdx.x, kw = nkv * LT_HD;
    const long long p = (long long)(*ppos + off + t);
    const float* row = qkv + (long long)t * ld + nh * LT_HD;
    for (int i = threadIdx.x; i < kw; i += blockDim.x) {
        kc[p * kw + i] = __float2half(row[i]);
        vc[p * kw + i] = __float2half(row[kw + i]);
    }
}

/* one tile of up to 32 keys for one query row per warp (lane = key): online softmax update of (m, l, acc[4]) */
__device__ __forceinline__ void lt_att_tile(const float* qh, const __half* ks, const __half* vs, int nk, float scale,
                                            int lane, float& m, float& l, float* acc) {
    float sc = -INFINITY;
    if (lane < nk) {
        const __half* kr = ks + lane * LT_KP;
        float s = 0.0f;
#pragma unroll 8
        for (int d = 0; d < LT_HD; d += 4) {
            const float4 q4 = *reinterpret_cast<const float4*>(qh + d);
            const float2 k0 = __half22float2(*reinterpret_cast<const __half2*>(kr + d));
            const float2 k1 = __half22float2(*reinterpret_cast<const __half2*>(kr + d + 2));
            s += q4.x * k0.x + q4.y * k0.y + q4.z * k1.x + q4.w * k1.y;
        }
        sc = s * scale;
    }
    const float mt = warp_max(sc);
    const float mn = fmaxf(m, mt);
    const float corr = m == -INFINITY ? 0.0f : __expf(m - mn);
    const float pj = lane < nk ? __expf(sc - mn) : 0.0f;
    l = l * corr + warp_sum(pj);
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] *= corr;
    for (int j = 0; j < nk; j++) {
        const float pp = __shfl_sync(0xffffffffu, pj, j);
        const __half* vr = vs + j * LT_KP + lane * 4;
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(vr));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(vr + 2));
        acc[0] += pp * a.x; acc[1] += pp * a.y; acc[2] += pp * b.x; acc[3] += pp * b.y;
    }
    m = mn;
}

/* load keys [k0, k0 + nk) of K/V head kh into the shared tiles: 16-byte global reads, 4-byte shared stores (the
   padded row stride of 130 halves keeps the score loop free of bank conflicts) */
__device__ __forceinline__ void lt_att_load(const __half* kc, const __half* vc, int kw, int kh, long long k0, int nk,
                                            __half* ks, __half* vs) {
    for (int i = threadIdx.x; i < nk * (LT_HD / 8); i += blockDim.x) {
        const int j = i / (LT_HD / 8), d = (i % (LT_HD / 8)) * 8;
        const long long o = (k0 + j) * kw + kh * LT_HD + d;
        const uint4 a = *reinterpret_cast<const uint4*>(kc + o), b = *reinterpret_cast<const uint4*>(vc + o);
        u32* kd = reinterpret_cast<u32*>(ks + j * LT_KP + d);
        u32* vd = reinterpret_cast<u32*>(vs + j * LT_KP + d);
        kd[0] = a.x; kd[1] = a.y; kd[2] = a.z; kd[3] = a.w;
        vd[0] = b.x; vd[1] = b.y; vd[2] = b.z; vd[3] = b.w;
    }
}

/* decode: grid (nkv, nsplit), 16 warps (one query head each). Keys 0 .. *ppos split in nsplit ranges of whole tiles;
   each block writes (m, l, acc[128]) per head for its range. */
extern "C" __global__ void lt_attn_dec(const float* __restrict__ q, const __half* __restrict__ kc,
                                       const __half* __restrict__ vc, const int* __restrict__ ppos, int nkv, int gq,
                                       float scale, float* pm, float* pl, float* pacc) {
    __shared__ __align__(16) float qs[16 * LT_HD];
    __shared__ __align__(16) __half ks[32 * LT_KP];
    __shared__ __align__(16) __half vs[32 * LT_KP];
    const int kh = blockIdx.x, sp = blockIdx.y, ns = gridDim.y, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int kw = nkv * LT_HD;
    const int n = *ppos + 1;
    const int tiles = (n + 31) / 32, per = (tiles + ns - 1) / ns;
    const int t0 = sp * per, t1 = min(tiles, t0 + per);
    for (int i = threadIdx.x; i < gq * LT_HD; i += blockDim.x) qs[i] = q[kh * gq * LT_HD + i];
    float m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int tt = t0; tt < t1; tt++) {
        const int k0 = tt * 32, nk = min(32, n - k0);
        __syncthreads();
        lt_att_load(kc, vc, kw, kh, k0, nk, ks, vs);
        __syncthreads();
        if (wp < gq) lt_att_tile(qs + wp * LT_HD, ks, vs, nk, scale, lane, m, l, acc);
    }
    if (wp < gq) {
        const long long o = ((long long)(kh * gq + wp) * ns + sp);
        if (lane == 0) { pm[o] = m; pl[o] = l; }
#pragma unroll
        for (int i = 0; i < 4; i++) pacc[o * LT_HD + lane * 4 + i] = acc[i];
    }
}

/* merge the split parts: out[h * 128 + d]. One warp per head */
extern "C" __global__ void lt_attn_merge(const float* __restrict__ pm, const float* __restrict__ pl,
                                         const float* __restrict__ pacc, int nh, int ns, float* out) {
    const int h = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (h >= nh) return;
    float m = -INFINITY;
    for (int s = 0; s < ns; s++) m = fmaxf(m, pm[h * ns + s]);
    float l = 0.0f, a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int s = 0; s < ns; s++) {
        const float ms = pm[(long long)h * ns + s];
        if (ms == -INFINITY) continue;
        const float c = __expf(ms - m);
        l += c * pl[(long long)h * ns + s];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] += c * pacc[((long long)h * ns + s) * LT_HD + lane * 4 + i];
    }
#pragma unroll
    for (int i = 0; i < 4; i++) out[h * LT_HD + lane * 4 + i] = a[i] / l;
}

/* prompt: grid (nkv, ceil(T / 2)), 16 warps; two query tokens per block share each K/V tile. Token t sits at
   position *ppos + off + t and sees keys 0 .. that position. q rows: stride ld. out rows: nh * 128. */
extern "C" __global__ void lt_attn_pre(const float* __restrict__ q, int ld, const __half* __restrict__ kc,
                                       const __half* __restrict__ vc, const int* __restrict__ ppos, int off, int T,
                                       int nkv, int gq, float scale, float* out) {
    __shared__ __align__(16) float qs[2][16 * LT_HD];
    __shared__ __align__(16) __half ks[32 * LT_KP];
    __shared__ __align__(16) __half vs[32 * LT_KP];
    const int kh = blockIdx.x, lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int kw = nkv * LT_HD, nh = nkv * gq;
    const int ta = blockIdx.y * 2, nt = min(2, T - ta);
    const int base = *ppos + off;
    for (int u = 0; u < nt; u++)
        for (int i = threadIdx.x; i < gq * LT_HD; i += blockDim.x)
            qs[u][i] = q[(long long)(ta + u) * ld + kh * gq * LT_HD + i];
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    float acc[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    const int n = base + ta + nt;                            /* keys seen by the last token of the pair */
    for (int k0 = 0; k0 < n; k0 += 32) {
        const int nk = min(32, n - k0);
        __syncthreads();
        lt_att_load(kc, vc, kw, kh, k0, nk, ks, vs);
        __syncthreads();
        if (wp < gq)
            for (int u = 0; u < nt; u++) {
                const int lim = base + ta + u + 1 - k0;      /* keys of this tile that token u sees */
                if (lim > 0) lt_att_tile(qs[u] + wp * LT_HD, ks, vs, min(nk, lim), scale, lane, m[u], l[u], acc[u]);
            }
    }
    if (wp < gq)
        for (int u = 0; u < nt; u++)
#pragma unroll
            for (int i = 0; i < 4; i++)
                out[(long long)(ta + u) * nh * LT_HD + (kh * gq + wp) * LT_HD + lane * 4 + i] = acc[u][i] / l[u];
}

/* prompt attention for cards without tensor cores (fp32): grid (nkv, ceil(T / 8)), 256 threads. A block takes 8
   tokens x 16 query heads = 128 query rows against shared tiles of 32 keys (fp32 in shared memory). Thread t owns
   row t / 2 and half t % 2 of the 128 dims: its 64 query values and 64 output sums stay in registers, a score is
   the sum of the two halves (one shuffle), and both threads of a row keep the same running max and sum. The next
   tile comes in 16-byte loads to registers while the current one is used. */
extern "C" __global__ void __launch_bounds__(256) lt_attn_pre8(const float* __restrict__ q, int ld,
        const __half* __restrict__ kc, const __half* __restrict__ vc, const int* __restrict__ ppos, int off, int T,
        int nkv, int gq, float scale, float* out) {
    __shared__ __align__(16) float Ks[32][LT_HD];
    __shared__ __align__(16) float Vs[32][LT_HD];
    const int kh = blockIdx.x, tid = threadIdx.x;
    const int kw = nkv * LT_HD, nh = nkv * gq;
    const int ta = blockIdx.y * 8, nt = min(8, T - ta);
    const int row = tid >> 1, half = tid & 1, u = row >> 4, j = row & 15;
    const int base = *ppos + off;
    const bool live = u < nt && j < gq;
    const int last = base + ta + min(u, nt - 1);             /* the last key this row sees */
    float qv[64], o[64];
    {
        const float* qr = q + (long long)(ta + min(u, nt - 1)) * ld + (kh * gq + min(j, gq - 1)) * LT_HD + 64 * half;
#pragma unroll
        for (int d = 0; d < 64; d++) { qv[d] = qr[d] * scale; o[d] = 0.0f; }
    }
    float m = -1e30f, l = 0.0f;
    const int n = base + ta + nt;                            /* keys seen by the last token of the block */
    const int lk = tid >> 4, ldm = (tid & 15) * 8;          /* this thread loads keys lk, lk + 16, dims ldm .. + 7 */
    int4 rk[2], rv[2];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int key = k0 + lk + 16 * h;
            rk[h] = rv[h] = make_int4(0, 0, 0, 0);
            if (key < n) {
                const long long g = (long long)key * kw + kh * LT_HD + ldm;
                rk[h] = __ldg(reinterpret_cast<const int4*>(kc + g));
                rv[h] = __ldg(reinterpret_cast<const int4*>(vc + g));
            }
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < n; k0 += 32) {
        const int nk = min(32, n - k0);
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const __half2* a = reinterpret_cast<const __half2*>(&rk[h]);
            const __half2* b = reinterpret_cast<const __half2*>(&rv[h]);
            float4* kd = reinterpret_cast<float4*>(&Ks[lk + 16 * h][ldm]);
            float4* vd = reinterpret_cast<float4*>(&Vs[lk + 16 * h][ldm]);
#pragma unroll
            for (int x = 0; x < 2; x++) {
                const float2 a0 = __half22float2(a[2 * x]), a1 = __half22float2(a[2 * x + 1]);
                const float2 b0 = __half22float2(b[2 * x]), b1 = __half22float2(b[2 * x + 1]);
                kd[x] = make_float4(a0.x, a0.y, a1.x, a1.y);
                vd[x] = make_float4(b0.x, b0.y, b1.x, b1.y);
            }
        }
        __syncthreads();
        if (k0 + 32 < n) fetch(k0 + 32);
        float s[32];
        float mt = -1e30f;
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            const float4* kr = reinterpret_cast<const float4*>(&Ks[kk][64 * half]);
            float a = 0.0f;
#pragma unroll
            for (int d = 0; d < 16; d++) {
                const float4 k4 = kr[d];
                a += qv[4 * d] * k4.x + qv[4 * d + 1] * k4.y + qv[4 * d + 2] * k4.z + qv[4 * d + 3] * k4.w;
            }
            a += __shfl_xor_sync(0xffffffffu, a, 1);
            s[kk] = (kk < nk && k0 + kk <= last) ? a : -1e30f;
            mt = fmaxf(mt, s[kk]);
        }
        const float mn = fmaxf(m, mt);
        const float corr = __expf(m - mn);
        l *= corr;
#pragma unroll
        for (int d = 0; d < 64; d++) o[d] *= corr;
#pragma unroll
        for (int kk = 0; kk < 32; kk++) {
            const float p = s[kk] > -1e29f ? __expf(s[kk] - mn) : 0.0f;
            l += p;
            const float4* vr = reinterpret_cast<const float4*>(&Vs[kk][64 * half]);
#pragma unroll
            for (int d = 0; d < 16; d++) {
                const float4 v4 = vr[d];
                o[4 * d] += p * v4.x; o[4 * d + 1] += p * v4.y; o[4 * d + 2] += p * v4.z; o[4 * d + 3] += p * v4.w;
            }
        }
        m = mn;
    }
    if (live) {
        const float il = 1.0f / l;
        float* orow = out + (long long)(ta + u) * nh * LT_HD + (kh * gq + j) * LT_HD + 64 * half;
#pragma unroll
        for (int d = 0; d < 64; d++) orow[d] = o[d] * il;
    }
}

/* prompt attention for Pascal and Volta (fp32, no tensor cores): grid (nkv, ceil(T / 6)), 256 threads. A block takes
   6 tokens x 16 query heads. Thread t serves query head t / 8 % 16 of tokens t / 128, + 2, + 4 (three rows), with
   16 of the 128 dims (part t % 8): each value it reads from the shared tiles of 16 keys serves three rows. A
   score is the sum of the 8 parts (three shuffles). The tile rows are padded so that the 8 parts of a warp read
   different banks. Measured on a GTX 1080 at 32K: 2.2x lt_attn_pre8. */
extern "C" __global__ void __launch_bounds__(256) lt_attn_pre6(const float* __restrict__ q, int ld,
        const __half* __restrict__ kc, const __half* __restrict__ vc, const int* __restrict__ ppos, int off, int T,
        int nkv, int gq, float scale, float* out) {
    constexpr int NK = 16, DD = 16, R = 3, PADW = LT_HD + 32;  /* 8 parts of 16 dims, each 4 floats apart */
    __shared__ __align__(16) float Ks[NK][PADW];
    __shared__ __align__(16) float Vs[NK][PADW];
    const int kh = blockIdx.x, tid = threadIdx.x;
    const int kw = nkv * LT_HD, nh = nkv * gq;
    const int ta = blockIdx.y * 6, nt = min(6, T - ta);
    const int rs = tid >> 3, part = tid & 7, u0 = rs >> 4, j = rs & 15;
    const int base = *ppos + off;
    float qv[R][DD], o[R][DD], m[R], l[R];
    int last[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int u = min(u0 + 2 * r, nt - 1);
        last[r] = base + ta + u;
        const float* qr = q + (long long)(ta + u) * ld + (kh * gq + min(j, gq - 1)) * LT_HD + DD * part;
#pragma unroll
        for (int d = 0; d < DD; d++) { qv[r][d] = qr[d] * scale; o[r][d] = 0.0f; }
        m[r] = -1e30f; l[r] = 0.0f;
    }
    const int n = base + ta + nt;                            /* keys seen by the last token of the block */
    const int lk = tid >> 4, ldm = (tid & 15) * 8;          /* this thread loads key lk, dims ldm .. + 7 */
    const int sdm = ldm + 4 * (ldm / DD);
    int4 rk, rv;
    auto fetch = [&](int k0) {
        rk = rv = make_int4(0, 0, 0, 0);
        if (k0 + lk < n) {
            const long long g = (long long)(k0 + lk) * kw + kh * LT_HD + ldm;
            rk = __ldg(reinterpret_cast<const int4*>(kc + g));
            rv = __ldg(reinterpret_cast<const int4*>(vc + g));
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < n; k0 += NK) {
        const int nk = min(NK, n - k0);
        __syncthreads();
        {
            const __half2* a = reinterpret_cast<const __half2*>(&rk);
            const __half2* b = reinterpret_cast<const __half2*>(&rv);
            float4* kd = reinterpret_cast<float4*>(&Ks[lk][sdm]);
            float4* vd = reinterpret_cast<float4*>(&Vs[lk][sdm]);
#pragma unroll
            for (int x = 0; x < 2; x++) {
                const float2 a0 = __half22float2(a[2 * x]), a1 = __half22float2(a[2 * x + 1]);
                const float2 b0 = __half22float2(b[2 * x]), b1 = __half22float2(b[2 * x + 1]);
                kd[x] = make_float4(a0.x, a0.y, a1.x, a1.y);
                vd[x] = make_float4(b0.x, b0.y, b1.x, b1.y);
            }
        }
        __syncthreads();
        if (k0 + NK < n) fetch(k0 + NK);
        float s[R][NK], mt[R];
#pragma unroll
        for (int r = 0; r < R; r++) mt[r] = -1e30f;
#pragma unroll
        for (int kk = 0; kk < NK; kk++) {
            const float4* kr = reinterpret_cast<const float4*>(&Ks[kk][(DD + 4) * part]);
            float a[R];
#pragma unroll
            for (int r = 0; r < R; r++) a[r] = 0.0f;
#pragma unroll
            for (int d = 0; d < DD / 4; d++) {
                const float4 k4 = kr[d];
#pragma unroll
                for (int r = 0; r < R; r++)
                    a[r] += qv[r][4 * d] * k4.x + qv[r][4 * d + 1] * k4.y + qv[r][4 * d + 2] * k4.z + qv[r][4 * d + 3] * k4.w;
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
#pragma unroll
                for (int x = 1; x < 8; x <<= 1) a[r] += __shfl_xor_sync(0xffffffffu, a[r], x);
                s[r][kk] = (kk < nk && k0 + kk <= last[r]) ? a[r] : -1e30f;
                mt[r] = fmaxf(mt[r], s[r][kk]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float mn = fmaxf(m[r], mt[r]);
            const float corr = __expf(m[r] - mn);
            l[r] *= corr;
#pragma unroll
            for (int d = 0; d < DD; d++) o[r][d] *= corr;
            m[r] = mn;
#pragma unroll
            for (int kk = 0; kk < NK; kk++) {
                s[r][kk] = s[r][kk] > -1e29f ? __expf(s[r][kk] - mn) : 0.0f;
                l[r] += s[r][kk];
            }
        }
#pragma unroll
        for (int kk = 0; kk < NK; kk++) {
            const float4* vr = reinterpret_cast<const float4*>(&Vs[kk][(DD + 4) * part]);
#pragma unroll
            for (int d = 0; d < DD / 4; d++) {
                const float4 v4 = vr[d];
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const float p = s[r][kk];
                    o[r][4 * d] += p * v4.x; o[r][4 * d + 1] += p * v4.y; o[r][4 * d + 2] += p * v4.z;
                    o[r][4 * d + 3] += p * v4.w;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int u = u0 + 2 * r;
        if (u < nt && j < gq) {
            const float il = 1.0f / l[r];
            float* orow = out + (long long)(ta + u) * nh * LT_HD + (kh * gq + j) * LT_HD + DD * part;
#pragma unroll
            for (int d = 0; d < DD; d++) orow[d] = o[r][d] * il;
        }
    }
}

/* ---------------------------------------------------------------------------------------------------------------- */
/* prompt matrix products (T tokens)                                                                                */

/* Y = W X for Q2_0 W (rows x cols) and T quantized inputs. Tile: 64 rows x 32 tokens, 256 threads, 64 columns per
   step; thread (ty, tx) computes rows 4 ty .. 4 ty + 3 for tokens 2 tx, 2 tx + 1.
   Grouped (MoE) form: blockIdx.z = expert e; its tokens are the entries [off[e], off[e] + cnt[e]) of a list in
   expert order; input row i of the group is xrow[off[e] + i] (xrow NULL: off[e] + i) and output row off[e] + i.
   Plain form: cnt NULL, T tokens, input/output row = token. acc = 1: Y += W X. Output row stride ldy. */
extern "C" __global__ void lt_mm_q2(const u32* const* __restrict__ codes, const __half* const* __restrict__ sc,
                                    int rows, int cols, const int* __restrict__ cnt, const int* __restrict__ off,
                                    const int* __restrict__ xrow, int T, const int* __restrict__ xq,
                                    const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy, int acc) {
    __shared__ int wsm[64 * 17];                             /* 64 rows x 16 expanded words (+1 pad) */
    __shared__ float wd[64];
    __shared__ int xs[32 * 17];                              /* 32 tokens x 16 words (+1 pad) */
    __shared__ float xd[32 * 2];
    __shared__ int xsum[32 * 2];
    const int e = blockIdx.z;
    const int n = cnt ? cnt[e] : T;
    const int c0 = blockIdx.y * 32;
    if (c0 >= n) return;
    const int o = cnt ? off[e] : 0;
    const int r0 = blockIdx.x * 64;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int runs = cols >> 5;
    const u32* cw = codes[e];
    const __half* cs = sc[e];
    float a[4][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    for (int k = 0; k < cols; k += 64) {
        __syncthreads();
        /* weights: 64 rows x 4 code words -> 64 x 16 expanded words (code word w, shift k: index w * 4 + k) */
        {
            const int r = threadIdx.x >> 2, w = threadIdx.x & 3;
            const int gr = r0 + r;
            u32 c = 0;
            if (gr < rows) c = cw[(long long)gr * (cols >> 4) + (k >> 4) + w];
#pragma unroll
            for (int s = 0; s < 4; s++) wsm[r * 17 + w * 4 + s] = (int)((c >> (2 * s)) & LT_M);
            if (w == 0) wd[r] = gr < rows ? __half2float(cs[(long long)gr * (cols >> 6) + (k >> 6)]) : 0.0f;
        }
        /* inputs: 32 tokens x 2 runs x 8 words */
        for (int i = threadIdx.x; i < 32 * 16; i += 256) {
            const int t = i >> 4, wv = i & 15;
            const int c = c0 + t;
            int v = 0;
            if (c < n) {
                const long long xr = xrow ? xrow[o + c] : (long long)(o + c);
                v = xq[(xr * runs + (k >> 5)) * 8 + wv];
            }
            xs[t * 17 + wv] = v;
        }
        if (threadIdx.x < 64) {
            const int t = threadIdx.x >> 1, h = threadIdx.x & 1;
            const int c = c0 + t;
            float dv = 0.0f;
            int sv = 0;
            if (c < n) {
                const long long xr = xrow ? xrow[o + c] : (long long)(o + c);
                dv = dx[xr * runs + (k >> 5) + h];
                sv = sx[xr * runs + (k >> 5) + h];
            }
            xd[t * 2 + h] = dv;
            xsum[t * 2 + h] = sv;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = ty * 4 + i;
            const int* wr = wsm + r * 17;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int t = tx * 2 + j;
                const int* xr = xs + t * 17;
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int u = 0; u < 8; u++) s0 = lt_dp4a(wr[u], xr[u], s0);
#pragma unroll
                for (int u = 8; u < 16; u++) s1 = lt_dp4a(wr[u], xr[u], s1);
                a[i][j] += wd[r] * (xd[t * 2] * (float)(s0 - xsum[t * 2]) + xd[t * 2 + 1] * (float)(s1 - xsum[t * 2 + 1]));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int r = r0 + ty * 4 + i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int c = c0 + tx * 2 + j;
            if (c >= n) continue;
            float* p = y + (long long)(o + c) * ldy + r;
            *p = acc ? *p + a[i][j] : a[i][j];
        }
    }
}

/* prompt on dp4a cards without tensor cores (sm_61, sm_62: GTX 10xx, P40): Y = W X as lt_mm_q2 (the same exact
   run sums), tiled as lightning_k35.cu's lt_mmk_q2: 128 rows x 16 TN tokens, 256 threads, thread (ty, tx) = rows
   8 ty .. + 7 x tokens TN tx .. + TN - 1, one run per step with the next run's words in registers. The digit words
   and the input words stay int32 in shared memory (k-major: word k of a run = lane-order word k). */
template <int TN>
__device__ __forceinline__ void lt_mmd_t(const u32* const* __restrict__ codes, const __half* const* __restrict__ sc,
        int rows, int cols, const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow,
        int T, const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy,
        int acc) {
    const int NT = 16 * TN;
    __shared__ __align__(16) int As[8][128];
    __shared__ __align__(16) int Bs[8][NT];
    const int e = blockIdx.z;
    const int n = cnt ? cnt[e] : T;
    const int c0 = blockIdx.y * NT;
    if (c0 >= n) return;
    const int o = cnt ? off[e] : 0;
    const int r0 = blockIdx.x * 128;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int runs = cols >> 5;
    const u32* cw = codes[e];
    const __half* cs = sc[e];
    /* loads: code word aw of row ar (256 threads: 128 rows x 2 words); input word bw of tokens bt + 32 p */
    const int NP = NT / 32;
    const int ar = tid >> 1, aw = tid & 1, arow = r0 + ar;
    const int bt = tid >> 3, bw = tid & 7;
    int bxr[NP];
#pragma unroll
    for (int p = 0; p < NP; p++) {
        const int c = c0 + bt + 32 * p;
        bxr[p] = c < n ? (xrow ? xrow[o + c] : o + c) : -1;
    }
    int txr[TN];
#pragma unroll
    for (int j = 0; j < TN; j++) {
        const int c = c0 + TN * tx + j;
        txr[j] = c < n ? (xrow ? xrow[o + c] : o + c) : -1;
    }
    u32 wa = arow < rows ? cw[(long long)arow * (cols >> 4) + aw] : 0u;
    int wb[NP];
#pragma unroll
    for (int p = 0; p < NP; p++) wb[p] = bxr[p] >= 0 ? xq[(long long)bxr[p] * runs * 8 + bw] : 0;
    float a[8][TN];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) a[i][j] = 0.0f;
    for (int h = 0; h < runs; h++) {
#pragma unroll
        for (int i = 0; i < 4; i++) As[4 * aw + i][ar] = (int)((wa >> (2 * i)) & LT_M);
#pragma unroll
        for (int p = 0; p < NP; p++) Bs[bw][bt + 32 * p] = wb[p];
        __syncthreads();
        if (h + 1 < runs) {
            wa = arow < rows ? cw[(long long)arow * (cols >> 4) + 2 * (h + 1) + aw] : 0u;
#pragma unroll
            for (int p = 0; p < NP; p++) wb[p] = bxr[p] >= 0 ? xq[((long long)bxr[p] * runs + h + 1) * 8 + bw] : 0;
        }
        int s[8][TN];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) s[i][j] = 0;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int4 a0 = *reinterpret_cast<const int4*>(&As[k][8 * ty]);
            const int4 a1 = *reinterpret_cast<const int4*>(&As[k][8 * ty + 4]);
            const int av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            int bv[TN];
            if (TN == 4) {
                const int4 b = *reinterpret_cast<const int4*>(&Bs[k][4 * tx]);
                bv[0] = b.x; bv[1 % TN] = b.y; bv[2 % TN] = b.z; bv[3 % TN] = b.w;
            } else {
                const int2 b = *reinterpret_cast<const int2*>(&Bs[k][2 * tx]);
                bv[0] = b.x; bv[1 % TN] = b.y;
            }
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) s[i][j] = lt_dp4a(av[i], bv[j], s[i][j]);
        }
        float ad[8], bd[TN], bs[TN];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = r0 + 8 * ty + i;
            ad[i] = r < rows ? __half2float(cs[(long long)r * (cols >> 6) + (h >> 1)]) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < TN; j++) {
            bd[j] = txr[j] >= 0 ? dx[(long long)txr[j] * runs + h] : 0.0f;
            bs[j] = txr[j] >= 0 ? (float)sx[(long long)txr[j] * runs + h] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < TN; j++) a[i][j] += ad[i] * (bd[j] * ((float)s[i][j] - bs[j]));
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int r = r0 + 8 * ty + i;
        if (r >= rows) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            const int c = c0 + TN * tx + j;
            if (c >= n) continue;
            float* p = y + (long long)(o + c) * ldy + r;
            *p = acc ? *p + a[i][j] : a[i][j];
        }
    }
}

#define LT_MMD_ARGS const u32* const* __restrict__ codes, const __half* const* __restrict__ sc, int rows, int cols, \
    const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow, int T, \
    const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy, int acc
extern "C" __global__ void __launch_bounds__(256) lt_mmd_q2(LT_MMD_ARGS) {
    lt_mmd_t<4>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, sx, y, ldy, acc); }
extern "C" __global__ void __launch_bounds__(256) lt_mmd32_q2(LT_MMD_ARGS) {
    lt_mmd_t<2>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, sx, y, ldy, acc); }

/* MoE groups on dp4a cards without tensor cores (sm_61, sm_62) as a decode-style product over 4 tokens at a time,
   as lightning_k35.cu's lt_mvkg_q2: grid (ceil(rows / 16), ngroups), 128 threads, warp w = rows 4 w .. 4 w + 3 of
   the block; lane l owns run 32 c + l of chunk c, keeps the 4 tokens' lane-order words in registers, and each
   expanded code word meets the 4 tokens' words in 4 dp4a (exact run sums). Lanes added through shared memory. */
extern "C" __global__ void __launch_bounds__(128) lt_mvdg_q2(const u32* const* __restrict__ codes,
        const __half* const* __restrict__ sc, int rows, int cols, const int* __restrict__ glist,
        const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow,
        const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy) {
    __shared__ float red[4][16][33];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    const int e = glist[2 * blockIdx.y], j0 = glist[2 * blockIdx.y + 1];
    const int o = off[e] + j0, nt = min(4, cnt[e] - j0);
    const int runs = cols >> 5, nb = cols >> 6, nch = (runs + 31) >> 5;
    int xr[4];
#pragma unroll
    for (int t = 0; t < 4; t++) {
        const int p = o + min(t, nt - 1);
        xr[t] = xrow ? xrow[p] : p;
    }
    const uint2* wb = reinterpret_cast<const uint2*>(codes[e]);
    const unsigned short* sb = reinterpret_cast<const unsigned short*>(sc[e]);
    const int r0 = blockIdx.x * 16 + wp * 4;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int t = 0; t < 4; t++) acc[i][t] = 0.0f;
    for (int c = 0; c < nch; c++) {
        const int h = c * 32 + lane;
        const bool on = h < runs;
        const int hc = on ? h : runs - 1;
        int q[4][8];
        float dd[4];
        int sq[4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            const int xi = xr[t] * runs + hc;
            const int4 a = reinterpret_cast<const int4*>(xq + xi * 8)[0], b = reinterpret_cast<const int4*>(xq + xi * 8)[1];
            q[t][0] = a.x; q[t][1] = a.y; q[t][2] = a.z; q[t][3] = a.w;
            q[t][4] = b.x; q[t][5] = b.y; q[t][6] = b.z; q[t][7] = b.w;
            dd[t] = on && t < nt ? dx[xi] : 0.0f;
            sq[t] = sx[xi];
        }
        uint2 wn[4];
        unsigned short sn[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned r = (unsigned)min(r0 + i, rows - 1);
            wn[i] = __ldg(wb + r * (unsigned)runs + hc);
            sn[i] = __ldg(sb + r * (unsigned)nb + (hc >> 1));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int s[4] = {0, 0, 0, 0};
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int m0 = (int)((wn[i].x >> (2 * k)) & LT_M), m1 = (int)((wn[i].y >> (2 * k)) & LT_M);
#pragma unroll
                for (int t = 0; t < 4; t++) s[t] = lt_dp4a(m1, q[t][4 + k], lt_dp4a(m0, q[t][k], s[t]));
            }
            const float ds = __half2float(__ushort_as_half(sn[i]));
#pragma unroll
            for (int t = 0; t < 4; t++) acc[i][t] += ds * dd[t] * (float)(s[t] - sq[t]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int t = 0; t < 4; t++) red[wp][4 * i + t][lane] = acc[i][t];
    __syncwarp();
    if (lane < 16) {
        float v = 0.0f;
#pragma unroll 8
        for (int k = 0; k < 32; k++) v += red[wp][lane][k];
        const int r = r0 + (lane >> 2), t = lane & 3;
        if (r < rows && t < nt) y[(long long)(o + t) * ldy + r] = v;
    }
}

/* decode with host experts: after routing, copy each chosen expert that is in host memory (hidx[e] >= 0) into its
   slot of a device scratch buffer with wide coalesced reads (far faster over PCIe than the products' small reads),
   and write per-slot pointer tables (the scratch copy, or the expert's own device pointers) for the products.
   grid (slots, nb), 256 threads; block (k, b) copies 1 / nb of slot k. Sizes in bytes, multiples of 16. */
/* decode with host experts, one layer: the device slot of each chosen host expert (slot[k]; miss[k] = 1 when it
   must be read from host memory). The C slots keep the last used host experts: owner[c] = the expert in slot c,
   age[c] = the step that used it last (tick[0] counts the steps). One thread: there are only n_used choices. */
extern "C" __global__ void lt_cache_dec(const int* __restrict__ ids, const int* __restrict__ hidx, int* owner,
                                        int* age, int* tick, int C, int U, int* slot, int* miss) {
    if (threadIdx.x != 0) return;
    const int now = ++tick[0];
    for (int k = 0; k < U; k++) {                      /* the hits first, so that no miss takes a slot in use */
        const int e = ids[k];
        slot[k] = -1;
        miss[k] = 0;
        if (hidx[e] < 0) continue;
        for (int c = 0; c < C; c++)
            if (owner[c] == e) { slot[k] = c; age[c] = now; break; }
    }
    for (int k = 0; k < U; k++) {
        const int e = ids[k];
        if (hidx[e] < 0 || slot[k] >= 0) continue;
        int j = -1;
        for (int q = 0; q < k; q++)                     /* (the same expert twice in one step: one copy) */
            if (ids[q] == e && slot[q] >= 0) j = slot[q];
        if (j < 0) {
            int best = 0;
            for (int c = 1; c < C; c++)
                if (age[c] < age[best]) best = c;
            j = best;
            owner[j] = e;
            miss[k] = 1;
        }
        age[j] = now;
        slot[k] = j;
    }
}

extern "C" __global__ void lt_stage_dec(const int* __restrict__ ids, const int* __restrict__ hidx,
        const unsigned long long* __restrict__ uc, const unsigned long long* __restrict__ us,
        const unsigned long long* __restrict__ dc, const unsigned long long* __restrict__ ds,
        long long ucb, long long usb, long long dcb, long long dsb, unsigned char* scratch,
        const int* __restrict__ cslot, const int* __restrict__ cmiss,
        unsigned long long* suc, unsigned long long* sus, unsigned long long* sdc, unsigned long long* sds) {
    /* the pointer tables for the n_used choices; a chosen host expert that is not in its slot yet is copied there */
    const int k = blockIdx.x, b = blockIdx.y, nb = gridDim.y;
    const int e = ids[k];
    const long long slot = ucb + usb + dcb + dsb;
    const bool host = hidx[e] >= 0;
    unsigned char* base = scratch + (host ? cslot[k] : 0) * slot;
    if (b == 0 && threadIdx.x == 0) {
        suc[k] = host ? (unsigned long long)base : uc[e];
        sus[k] = host ? (unsigned long long)(base + ucb) : us[e];
        sdc[k] = host ? (unsigned long long)(base + ucb + usb) : dc[e];
        sds[k] = host ? (unsigned long long)(base + ucb + usb + dcb) : ds[e];
    }
    if (!host || !cmiss[k]) return;
    const unsigned long long src[4] = {uc[e], us[e], dc[e], ds[e]};
    const long long len[4] = {ucb, usb, dcb, dsb};
    long long o = 0;
    for (int p = 0; p < 4; p++) {
        const uint4* s = reinterpret_cast<const uint4*>(src[p]);
        uint4* d = reinterpret_cast<uint4*>(base + o);
        const long long n = len[p] / 16;
        for (long long i = (long long)b * blockDim.x + threadIdx.x; i < n; i += (long long)nb * blockDim.x) d[i] = s[i];
        o += len[p];
    }
}

/* MoE prompt routing lists: for T tokens with k experts each (ids [T][k]), count the tokens per expert, give each
   expert a range of the list (in expert order) and record for (t, j) its list entry. The order of the tokens inside
   an expert's range does not change any number (each output row of the grouped product is its own dot product).
   One block of 1024 threads; ne <= 1024. */
extern "C" __global__ void lt_moe_lists(const int* __restrict__ ids, int T, int k, int ne, int* cnt, int* off,
                                        int* xrow, int* where) {
    __shared__ int c[1024], o[1024];
    for (int e = threadIdx.x; e < ne; e += blockDim.x) c[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < T * k; i += blockDim.x) atomicAdd(&c[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x < 32) {                                  /* exclusive prefix sum by one warp */
        int base = 0;
        for (int e0 = 0; e0 < ne; e0 += 32) {
            const int e = e0 + threadIdx.x;
            const int v = e < ne ? c[e] : 0;
            int incl = v;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const int u = __shfl_up_sync(0xffffffffu, incl, d);
                if ((int)threadIdx.x >= d) incl += u;
            }
            if (e < ne) o[e] = base + incl - v;
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < ne; e += blockDim.x) { cnt[e] = c[e]; off[e] = o[e]; c[e] = 0; }
    __syncthreads();
    for (int i = threadIdx.x; i < T * k; i += blockDim.x) {
        const int e = ids[i];
        const int p = o[e] + atomicAdd(&c[e], 1);
        xrow[p] = i / k;
        where[i] = p;
    }
}

/* x[t] += sum_j gw[t][j] * yd[where[t][j]] + ysh[t]. One block per token */
extern "C" __global__ void lt_moe_combine(const float* __restrict__ yd, const float* __restrict__ ysh,
                                          const int* __restrict__ where, const float* __restrict__ gw, int k, int cols,
                                          float* x) {
    const int t = blockIdx.x;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float s = 0.0f;
        for (int j = 0; j < k; j++) s += gw[t * k + j] * yd[(long long)where[t * k + j] * cols + i];
        x[(long long)t * cols + i] += s + ysh[(long long)t * cols + i];
    }
}

/* pos[0] += n (the decode position advances inside the graph) */
extern "C" __global__ void lt_advance(int* pos, int n) { if (threadIdx.x == 0) pos[0] += n; }

/* p[0] = v, in stream order (a host buffer reused by queued copies would race) */
extern "C" __global__ void lt_set(int* p, int v) { if (threadIdx.x == 0) p[0] = v; }
