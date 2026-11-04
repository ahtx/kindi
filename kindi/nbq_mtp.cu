/* The Nanbeige MTP head kernels (nanbeige.py Engine.mtp_draft). */

__device__ __forceinline__ float mtp_warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float mtp_block_sum(float v, float* red) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = mtp_warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) { v = mtp_warp_sum(lane < (blockDim.x >> 5) ? red[lane] : 0.0f); if (lane == 0) red[0] = v; }
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

/* one block of 1024: y = [norm(e) * ne, norm(h + hadd) * nh] (hadd may be null) */
extern "C" __global__ void __launch_bounds__(1024) mtp_in(const float* e, const float* h, const float* hadd,
        const float* ne, const float* nh, int n, float eps, float* y) {
    __shared__ float red[32];
    float se = 0.0f, sh = 0.0f;
    for (int i = threadIdx.x; i < n; i += 1024) {
        float a = e[i], b = h[i] + (hadd ? hadd[i] : 0.0f);
        se = fmaf(a, a, se);
        sh = fmaf(b, b, sh);
    }
    se = rsqrtf(mtp_block_sum(se, red) / (float)n + eps);
    sh = rsqrtf(mtp_block_sum(sh, red) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) {
        y[i] = e[i] * se * ne[i];
        y[n + i] = (h[i] + (hadd ? hadd[i] : 0.0f)) * sh * nh[i];
    }
}

/* one block of 1024: y = norm(x) * w */
extern "C" __global__ void __launch_bounds__(1024) mtp_rms(const float* x, const float* w, int n, float eps, float* y) {
    __shared__ float red[32];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += 1024) s = fmaf(x[i], x[i], s);
    s = rsqrtf(mtp_block_sum(s, red) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += 1024) y[i] = x[i] * s * w[i];
}

/* y[r] = s[r] * (W[r, :] . x) + add[r]; one warp per row, 8 rows per block; cols a multiple of 512 */
extern "C" __global__ void mtp_gemv(const signed char* W, const float* s, const float* x, int rows, int cols,
        const float* add, float* y) {
    int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const int4* w = (const int4*)(W + (size_t)r * cols);
    const float4* x4 = (const float4*)x;
    float acc = 0.0f;
    for (int c = lane; c < cols / 16; c += 32) {
        int4 q = w[c];
        int qq[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float4 xv = x4[4 * c + j];
            int v = qq[j];
            acc = fmaf((float)(signed char)(v), xv.x, acc);
            acc = fmaf((float)(signed char)(v >> 8), xv.y, acc);
            acc = fmaf((float)(signed char)(v >> 16), xv.z, acc);
            acc = fmaf((float)(signed char)(v >> 24), xv.w, acc);
        }
    }
    acc = mtp_warp_sum(acc);
    if (lane == 0) y[r] = s[r] * acc + (add ? add[r] : 0.0f);
}

/* y = silu(g) * u */
extern "C" __global__ void mtp_swiglu(const float* g, const float* u, int n, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float a = g[i]; y[i] = a / (1.0f + __expf(-a)) * u[i]; }
}
