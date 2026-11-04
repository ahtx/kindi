/* nbq_mvn.cu - decode kernels for N tokens at once (N = 2 .. 8): the check pass of speculative decoding. Each
   weight run is decoded once and used for the N inputs, so a pass reads the weights once (appended to nbq.cu). */

template <int T, int N>
__device__ __forceinline__ void nbq_row_dotn(const u8* w, const __half* wd, int r, int nb, const signed char* xq,
        const float* xs, int lane, const typename Tab<T>::E* tab, float* out, int rows = 0) {
    const int cols = nb << 8;
    const u8* row = w + (size_t)r * nb * BT<T>::B;
    const __half* rd = wd + (size_t)r * nb;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) acc[t] = 0.0f;
    for (int g = lane; g < nb * 8; g += 32) {
        int blk = g >> 3, sub = g & 7;
        int v[8]; float s0, s1, mn;
#if defined(NBQ_K35) && !defined(NBQ_KROWMAJOR)
        /* Kepler cards keep the weights in the chunk-major layout of nbq_mvc (nbq_kreorder) */
        const size_t kb = ((size_t)(g >> 5) * rows + r) * 4 + ((g >> 3) & 3);
        nbq_run<T>(w + kb * BT<T>::B, wd + kb, sub, v, s0, s1, mn, tab);
#else
        nbq_run<T>(row + (size_t)blk * BT<T>::B, rd + blk, sub, v, s0, s1, mn, tab);
#endif
#pragma unroll
        for (int t = 0; t < N; t++) {
            const int4* xp = reinterpret_cast<const int4*>(xq + (size_t)t * cols + 32 * g);
            int4 a = NBQ_LDG(xp), c = NBQ_LDG(xp + 1);
            NBQ_XF(a, c, xf);
            NBQ_DOT(v, a, c, xf, lo, hi);
            float f = nbq_runf(s0, lo, s1, hi);
            if (T == T_Q4K || T == T_Q5K) f = __fmaf_rn(-mn, (float)NBQ_SX(a, c, xf), f);
            acc[t] = __fmaf_rn(NBQ_LDG(xs + (size_t)t * (cols >> 5) + g), f, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < N; t++) out[t] = warp_sum(acc[t]);
}

/* y[t][yo + r] (+= add[t][r] when add != 0) for N inputs xq [N][cols] */
/* the N-token inputs of a row go to shared memory when they are big (long rows: L1 does not keep them) but fit */
__device__ __forceinline__ bool stage_x(int n, int cols) { return false; }   /* measured slower (occupancy) */

template <int T, int N>
__device__ __forceinline__ void nbq_mvn(const u8* w, const __half* wd, const signed char* xq, const float* xs,
        float* y, const float* add, int rows, int cols, int ldy, int yo) {
    NBQ_TAB(T, tab);
    const int lane = threadIdx.x & 31;
    extern __shared__ __align__(16) unsigned char xsm[];
    if (stage_x(N, cols)) {                            /* the inputs in shared memory (launched with that much) */
        const int nq = N * cols / 16, ns = N * cols / 32;
        for (int i = threadIdx.x; i < nq; i += blockDim.x) reinterpret_cast<uint4*>(xsm)[i] = reinterpret_cast<const uint4*>(xq)[i];
        for (int i = threadIdx.x; i < ns; i += blockDim.x) reinterpret_cast<float*>(xsm + N * cols)[i] = xs[i];
        __syncthreads();
        xq = reinterpret_cast<const signed char*>(xsm);
        xs = reinterpret_cast<const float*>(xsm + N * cols);
    }
    for (int r = blockIdx.x * 8 + (threadIdx.x >> 5); r < rows; r += gridDim.x * 8) {
        float o[N];
        nbq_row_dotn<T, N>(w, wd, r, cols >> 8, xq, xs, lane, tab, o, rows);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < N; t++) y[(size_t)t * ldy + yo + r] = add ? add[(size_t)t * ldy + yo + r] + o[t] : o[t];
        }
    }
}
#define NBQ_MVN1(T, NAME, N) extern "C" __global__ void __launch_bounds__(256) NAME##_##N(const u8* w, const __half* wd, \
    const signed char* xq, const float* xs, float* y, const float* add, int rows, int cols, int ldy, int yo) { \
    nbq_mvn<T, N>(w, wd, xq, xs, y, add, rows, cols, ldy, yo); }
#define NBQ_MVN(T, NAME) NBQ_MVN1(T, NAME, 1) NBQ_MVN1(T, NAME, 2) NBQ_MVN1(T, NAME, 3) NBQ_MVN1(T, NAME, 4) NBQ_MVN1(T, NAME, 5) \
    NBQ_MVN1(T, NAME, 6) NBQ_MVN1(T, NAME, 7) NBQ_MVN1(T, NAME, 8)
NBQ_MVN(T_Q4K, nbq_mvn_q4k)
NBQ_MVN(T_Q5K, nbq_mvn_q5k)
NBQ_MVN(T_IQ3XXS, nbq_mvn_iq3xxs)
NBQ_MVN(T_IQ3S, nbq_mvn_iq3s)
NBQ_MVN(T_IQ2S, nbq_mvn_iq2s)

/* gate and up for N inputs in one launch: rows [0, rows) of y[t] from wg, [rows, 2 rows) from wu (ld 2 rows) */
template <int N>
__device__ __forceinline__ void nbq_mv2n(const u8* wg, const __half* dg, const u8* wu, const __half* du,
        const signed char* xq, const float* xs, float* y, int rows, int cols) {
    NBQ_TAB(T_IQ3XXS, tab);
    const int lane = threadIdx.x & 31;
    for (int r = blockIdx.x * 8 + (threadIdx.x >> 5); r < 2 * rows; r += gridDim.x * 8) {
        float o[N];
        if (r < rows) nbq_row_dotn<T_IQ3XXS, N>(wg, dg, r, cols >> 8, xq, xs, lane, tab, o);
        else nbq_row_dotn<T_IQ3XXS, N>(wu, du, r - rows, cols >> 8, xq, xs, lane, tab, o);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < N; t++) y[(size_t)t * 2 * rows + r] = o[t];
        }
    }
}

/* Q, K, V of N tokens (Q, K: IQ2_S; V: Q4_K) with RoPE and the K / V cache store at positions pos[t]; q rows
   [N][nh 128] get the rotated Q. A warp takes the two rows of one RoPE pair (as nbq_qkv_rope). */
template <int N>
__device__ __forceinline__ void nbq_qkvn(const u8* wq, const __half* dq, const u8* wk, const __half* dk, const u8* wv,
        const signed char* xq, const float* xs, int nh, int nkv, int cols, const int* pos, float theta_scale,
        float* q, int ldq, __half* kc, __half* vc, int ctx) {
    NBQ_TAB(T_IQ2S, tab);
    const int lane = threadIdx.x & 31, nb = cols >> 8;
    const int pq = nh * 64, pk = nkv * 64, np = pq + 2 * pk;
    for (int pr = blockIdx.x * 8 + (threadIdx.x >> 5); pr < np; pr += gridDim.x * 8) {
        float a[N], b[N];
        if (pr < pq + pk) {
            const int r = 2 * (pr < pq ? pr : pr - pq);
            if (pr < pq) { nbq_row_dotn<T_IQ2S, N>(wq, dq, r, nb, xq, xs, lane, tab, a); nbq_row_dotn<T_IQ2S, N>(wq, dq, r + 1, nb, xq, xs, lane, tab, b); }
            else { nbq_row_dotn<T_IQ2S, N>(wk, dk, r, nb, xq, xs, lane, tab, a); nbq_row_dotn<T_IQ2S, N>(wk, dk, r + 1, nb, xq, xs, lane, tab, b); }
            if (lane == 0) {
                const int i = (r & 127) >> 1;
#pragma unroll
                for (int t = 0; t < N; t++) {
                    const int p = pos[t];
                    const float theta = (float)p * powf(theta_scale, (float)i);
                    const float c = cosf(theta), s = sinf(theta);
                    float x0, x1;
                    nbq_rot(a[t], b[t], c, s, x0, x1);
                    if (pr < pq) { q[(size_t)t * ldq + r] = x0; q[(size_t)t * ldq + r + 1] = x1; }
                    else {
                        size_t ko = nbq_kofs(r >> 7, ctx, p, i >> 2) + 2 * (i & 3);
                        kc[ko] = __float2half(x0);
                        kc[ko + 1] = __float2half(x1);
                    }
                }
            }
        } else {
            const int r = 2 * (pr - pq - pk);
            const __half* dv = reinterpret_cast<const __half*>(wv);
            nbq_row_dotn<T_Q4K, N>(wv, dv, r, nb, xq, xs, lane, nullptr, a);
            nbq_row_dotn<T_Q4K, N>(wv, dv, r + 1, nb, xq, xs, lane, nullptr, b);
            if (lane == 0) {
#pragma unroll
                for (int t = 0; t < N; t++) {
                    size_t o = ((size_t)(r >> 7) * ctx + pos[t]) * 128 + (r & 127);
                    vc[o] = __float2half(a[t]);
                    vc[o + 1] = __float2half(b[t]);
                }
            }
        }
    }
}
#define NBQ_FUSEN(N) \
extern "C" __global__ void __launch_bounds__(256) nbq_mv2n_iq3xxs_##N(const u8* wg, const __half* dg, const u8* wu, \
    const __half* du, const signed char* xq, const float* xs, float* y, int rows, int cols) { \
    nbq_mv2n<N>(wg, dg, wu, du, xq, xs, y, rows, cols); } \
extern "C" __global__ void __launch_bounds__(256) nbq_qkvn_rope_##N(const u8* wq, const __half* dq, const u8* wk, \
    const __half* dk, const u8* wv, const signed char* xq, const float* xs, int nh, int nkv, int cols, const int* pos, \
    float theta_scale, float* q, int ldq, __half* kc, __half* vc, int ctx) { \
    nbq_qkvn<N>(wq, dq, wk, dk, wv, xq, xs, nh, nkv, cols, pos, theta_scale, q, ldq, kc, vc, ctx); }
NBQ_FUSEN(2) NBQ_FUSEN(3) NBQ_FUSEN(4)

/* merge the attention splits of token blockIdx.y and write int8 runs (row blockIdx.y of xq / xs) */
extern "C" __global__ void nbq_attn_merge_q8_rows(const float* part_o, const float* part_ms, int nsp, signed char* xq,
        float* xs) {
    int h = blockIdx.x, d = threadIdx.x, t = blockIdx.y, nh = gridDim.x;
    part_o += (size_t)t * nh * nsp * 128;
    part_ms += (size_t)t * nh * nsp * 2;
    float mx = -1e30f;
    for (int s = 0; s < nsp; s++) mx = fmaxf(mx, part_ms[(h * nsp + s) * 2]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < nsp; s++) {
        float f = expf(part_ms[(h * nsp + s) * 2] - mx);
        l = __fmaf_rn(f, part_ms[(h * nsp + s) * 2 + 1], l);
        o = __fmaf_rn(f, part_o[((size_t)h * nsp + s) * 128 + d], o);
    }
    nbq_q8_run(o / l, h * 4 + (d >> 5), d & 31, xq + (size_t)t * nh * 128, xs + (size_t)t * nh * 4);
}

/* argmax of each of the rows of x [n][ld] (one block per row) -> out[row] */
extern "C" __global__ void nbq_argmax_rows(const float* x, int n, int ld, int* out) {
    __shared__ float bv[32]; __shared__ int bi[32];
    const float* p = x + (size_t)blockIdx.x * ld;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float v = -3.4e38f; int idx = 0x7fffffff;
    for (int i = tid; i < n; i += blockDim.x) if (p[i] > v) { v = p[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, v, o); int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if (lane == 0) { bv[warp] = v; bi[warp] = idx; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) if (bv[w] > v || (bv[w] == v && bi[w] < idx)) { v = bv[w]; idx = bi[w]; }
        out[blockIdx.x] = idx;
    }
}
