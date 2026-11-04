/* nbq_tcv.cu - matrix x (up to 8 vectors) on int8 tensor cores, for the check pass of speculative decoding
   (appended to nbq.cu).

   y[t][yo + r] = W x_t for t < n (n <= 8); xq [n][cols] int8, xs [n][cols / 32]. Each warp works alone on a tile
   of 16 rows and a range of runs (gridDim.y ranges; with more than one, the sums go to part [range][t][row] and
   nbq_tcv_sum adds them in range order). Per step of 8 runs a warp decodes the 16 x 8 runs into its own shared
   tile (a lane takes 4: one load covers 4 rows x 8 runs, 96 contiguous bytes per row), then takes the A fragments
   by ldmatrix and the B fragments (the inputs; tokens past n are zero) from global memory: one m16n8k32 per run.
   The inputs are loaded once per 16 rows, so a pass with 2 .. 8 tokens costs about one single-token pass. Per run
   and token, the int32 dot is scaled by the weight scale and the input scale in fp32, as in the other kernels; the
   sum over runs is in another order, so the result is not bit-equal to nbq_mvn. */
#define TCV_LD (8 * 32 + 16)
#define TCV_W 4                   /* warps per block */

template <int T>
__device__ __forceinline__ void nbq_tcv(const u8* w, const __half* wd, const signed char* xq, const float* xs,
        float* y, int rows, int cols, int n, int ldy, int yo, float* part) {
    NBQ_TAB(T, tab);
    const bool HALF = (T == T_IQ2S), MINS = (T == T_Q4K || T == T_Q5K);
    __shared__ __align__(16) u8 As[TCV_W][16 * TCV_LD];
    __shared__ float Sc[TCV_W][16][8][2], Mn[MINS ? TCV_W : 1][16][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gid = lane >> 2, t4 = lane & 3;
    const int nb = cols >> 8, nrun = cols >> 5, ntile = (rows + 15) >> 4;
    const int ta = 2 * t4, tb = 2 * t4 + 1;                  /* the two tokens of the C elements of this thread */
    const bool live = gid < n;
    const signed char* xb = xq + (size_t)(live ? gid : 0) * cols + 4 * t4;
    const int l8 = lane & 7, lq = lane >> 3;
    const int per = ((nrun + gridDim.y - 1) / gridDim.y + 7) & ~7;
    const int g0 = blockIdx.y * per, g1 = min(nrun, g0 + per);
    u8* aw = As[warp];
    const int wm = MINS ? warp : 0;
    for (int tile = blockIdx.x * TCV_W + warp; tile < ntile; tile += gridDim.x * TCV_W) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k0 = g0; k0 < g1; k0 += 8) {
#pragma unroll
            for (int h = 0; h < 4; h++) {                    /* decode 16 rows x 8 runs */
                const int it = lane + 32 * h, r = it >> 3, u = it & 7, g = k0 + u;
                const int row = min(16 * tile + r, rows - 1);
                if (g < g1) {
                    int v[8]; float s0, s1, mn;
                    nbq_run<T>(w + ((size_t)row * nb + (g >> 3)) * BT<T>::B, wd + (size_t)row * nb + (g >> 3), g & 7, v, s0,
                               s1, mn, tab);
                    uint4* dst = reinterpret_cast<uint4*>(aw + r * TCV_LD + 32 * u);
                    dst[0] = make_uint4(v[0], v[1], v[2], v[3]);
                    dst[1] = make_uint4(v[4], v[5], v[6], v[7]);
                    Sc[warp][r][u][0] = s0; Sc[warp][r][u][1] = s1;
                    if (MINS) Mn[wm][r][u] = mn;
                }
            }
            __syncwarp();
#pragma unroll
            for (int u = 0; u < 8; u++) {                    /* MMA */
                const int gu = k0 + u;
                if (gu >= g1) break;
                u32 a0, a1, a2, a3;
                u32 ad = (u32)__cvta_generic_to_shared(aw + (l8 + 8 * (lq & 1)) * TCV_LD + 32 * u + 16 * (lq >> 1));
                NBQ_ASM("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(ad));
                u32 b0 = 0u, b1 = 0u;
                if (live) { b0 = *reinterpret_cast<const u32*>(xb + 32 * gu); b1 = *reinterpret_cast<const u32*>(xb + 32 * gu + 16); }
                const float xa = ta < n ? xs[(size_t)ta * nrun + gu] : 0.0f;
                const float xc = tb < n ? xs[(size_t)tb * nrun + gu] : 0.0f;
                const float sa0 = Sc[warp][gid][u][0], sa1 = Sc[warp][gid][u][1];
                const float sb0 = Sc[warp][gid + 8][u][0], sb1 = Sc[warp][gid + 8][u][1];
                if (HALF) {
                    int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
                    NBQ_MMA16(c0, a0, a1, b0);
                    NBQ_MMA16(c1, a2, a3, b1);
                    acc[0] += ((float)c0[0] * sa0 + (float)c1[0] * sa1) * xa;
                    acc[1] += ((float)c0[1] * sa0 + (float)c1[1] * sa1) * xc;
                    acc[2] += ((float)c0[2] * sb0 + (float)c1[2] * sb1) * xa;
                    acc[3] += ((float)c0[3] * sb0 + (float)c1[3] * sb1) * xc;
                } else {
                    int c[4] = {0, 0, 0, 0};
                    NBQ_MMA32(c, a0, a1, a2, a3, b0, b1);
                    float fa = (float)c[0] * sa0, fc = (float)c[1] * sa0, fe = (float)c[2] * sb0, ff = (float)c[3] * sb0;
                    if (MINS) {                              /* the sum of the inputs of this run, tokens ta and tb */
                        int sx = __dp4a((int)b0, 0x01010101, __dp4a((int)b1, 0x01010101, 0));
                        sx += __shfl_xor_sync(0xffffffffu, sx, 1);
                        sx += __shfl_xor_sync(0xffffffffu, sx, 2);  /* lanes 4 t .. 4 t + 3 hold token t */
                        const float sxa = (float)__shfl_sync(0xffffffffu, sx, 4 * ta);
                        const float sxb = (float)__shfl_sync(0xffffffffu, sx, 4 * tb);
                        const float ma = Mn[wm][gid][u], mb = Mn[wm][gid + 8][u];
                        fa -= ma * sxa; fc -= ma * sxb; fe -= mb * sxa; ff -= mb * sxb;
                    }
                    acc[0] += fa * xa; acc[1] += fc * xc; acc[2] += fe * xa; acc[3] += ff * xc;
                }
            }
            __syncwarp();
        }
        const int r0 = 16 * tile + gid, r1 = r0 + 8;
        float* out = gridDim.y > 1 ? part + (size_t)blockIdx.y * 8 * rows : y;
        const int ld = gridDim.y > 1 ? rows : ldy, off = gridDim.y > 1 ? 0 : yo;
        if (r0 < rows) {
            if (ta < n) out[(size_t)ta * ld + off + r0] = acc[0];
            if (tb < n) out[(size_t)tb * ld + off + r0] = acc[1];
        }
        if (r1 < rows) {
            if (ta < n) out[(size_t)ta * ld + off + r1] = acc[2];
            if (tb < n) out[(size_t)tb * ld + off + r1] = acc[3];
        }
    }
}
#define NBQ_TCV(T, NAME) extern "C" __global__ void __launch_bounds__(32 * TCV_W) NAME(const u8* w, const __half* wd, \
    const signed char* xq, const float* xs, float* y, int rows, int cols, int n, int ldy, int yo, float* part) { \
    nbq_tcv<T>(w, wd, xq, xs, y, rows, cols, n, ldy, yo, part); }
#if NBQ_TC
NBQ_TCV(T_Q4K, nbq_tcv_q4k)
NBQ_TCV(T_Q5K, nbq_tcv_q5k)
NBQ_TCV(T_IQ3XXS, nbq_tcv_iq3xxs)
NBQ_TCV(T_IQ3S, nbq_tcv_iq3s)
NBQ_TCV(T_IQ2S, nbq_tcv_iq2s)
#endif

/* y[t][yo + r] = the sum over the ranges of part[range][t][r], in range order */
extern "C" __global__ void nbq_tcv_sum(const float* part, int nrange, int rows, int n, float* y, int ldy, int yo) {
    int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (r >= rows || t >= n) return;
    float v = 0.0f;
    for (int k = 0; k < nrange; k++) v += part[((size_t)k * 8 + t) * rows + r];
    y[(size_t)t * ldy + yo + r] = v;
}
