/* nbq_q8kv.cu - the q8_0 K/V cache (appended to nbq.cu at compile time; config kv_cache = q8_0).

   Per layer, K and V are int8 [nkv][ctx][128] with an fp16 scale per 32 values, [nkv][ctx][4]. A block of 32
   values is stored as d = amax / 127 (rounded to fp16) and q = round-to-nearest-even(x * 127 / amax), the q8_0 of
   llama.cpp's -ctk q8_0 -ctv q8_0. Decode attention: the scaled query is also put in q8_0, and a score is, per
   block, the exact integer dot product (dp4a) times d_q * d_k, the 4 blocks added in order; V is q * d in fp32.
   Prompt attention: nbq_fa.cu with the K / V tiles made fp16 from q * d (one rounding). The splits and the merges
   are those of the fp16 cache. */

__device__ __forceinline__ void nbq_q8blk(float x, int lane, signed char* q, __half* d) {
    float a = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    const float id = a != 0.0f ? 127.0f / a : 0.0f;
    q[lane] = (signed char)__float2int_rn(x * id);
    if (lane == 0) d[0] = __float2half_rn(a / 127.0f);
}

/* RoPE + the q8_0 K/V store for n rows (grid (nh + nkv, n), 128 threads = one value each, a warp = one block).
   Row t: q, k, v at q + t ld, k + t ld, v + t ld (fp32), position pos[t]; q is rotated in place. */
extern "C" __global__ void __launch_bounds__(128) nbq_rope_store_q8(float* q, const float* k, const float* v, int ld,
        const int* pos, int nh, int nkv, float theta_scale, signed char* kq, __half* kd, signed char* vq, __half* vd,
        int ctx) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x, lane = i & 31, pr = i >> 1;
    const int p = pos[t];
    q += (size_t)t * ld; k += (size_t)t * ld; v += (size_t)t * ld;
    const float theta = (float)p * powf(theta_scale, (float)pr);
    const float c = cosf(theta), s = sinf(theta);
    const float* x = h < nh ? q + h * 128 : k + (h - nh) * 128;
    float y0, y1;
    nbq_rot(x[2 * pr], x[2 * pr + 1], c, s, y0, y1);
    const float y = (i & 1) ? y1 : y0;
    if (h < nh) {
        __syncthreads();                               /* both values of every pair are read before the writes */
        q[h * 128 + i] = y;
        return;
    }
    const int hk = h - nh;
    const size_t o = ((size_t)hk * ctx + p) * 128 + (i & ~31);
    nbq_q8blk(y, lane, kq + o, kd + o / 32);
    nbq_q8blk(v[hk * 128 + i], lane, vq + o, vd + o / 32);
}

/* decode attention on the q8_0 cache: nbq_attn_part_t with int8 K and V. grid (nkv, NSPLIT, rows), 256 threads;
   warp w takes the chunks of 32 positions w, w + 8, ... of its run (lane = position for the scores, lane = 4 dims
   for P V). */
template <int G>
__device__ __forceinline__ void nbq_attn_q8_t(const float* q, const signed char* kq, const __half* kd,
        const signed char* vq, const __half* vd, const int* pos, int ctx, float scale, float* part_o, float* part_ms,
        int qstride) {
    const int hk = blockIdx.x, sp = blockIdx.y, nsp = gridDim.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = pos[blockIdx.z] + 1;
    q += (size_t)blockIdx.z * qstride;
    part_o += (size_t)blockIdx.z * gridDim.x * G * nsp * 128;
    part_ms += (size_t)blockIdx.z * gridDim.x * G * nsp * 2;
    const int per = max(256, ((n + nsp - 1) / nsp + 255) & ~255);
    const int p0 = sp * per, p1 = min(n, p0 + per);
    __shared__ __align__(16) signed char qc[G][128];
    __shared__ float qd[G][4];
    __shared__ float wo[8][G][128];
    __shared__ float wm[8][G], wl[8][G];
    if (p0 >= p1) {
        for (int i = tid; i < G * 128; i += 256) {
            int j = i >> 7, d = i & 127, h = hk * G + j;
            part_o[((size_t)h * nsp + sp) * 128 + d] = 0.0f;
            if (d == 0) { part_ms[(h * nsp + sp) * 2] = -1e30f; part_ms[(h * nsp + sp) * 2 + 1] = 0.0f; }
        }
        return;
    }
    if (warp < G) {                                     /* warp j: head j of the group, its 4 blocks */
        __half dh[4];
#pragma unroll
        for (int b = 0; b < 4; b++)
            nbq_q8blk(q[(hk * G + warp) * 128 + 32 * b + lane] * scale, lane, &qc[warp][32 * b], &dh[b]);
        if (lane == 0) {
#pragma unroll
            for (int b = 0; b < 4; b++) qd[warp][b] = __half2float(dh[b]);
        }
    }
    __syncthreads();
    float m[G], l[G], o[G][4];
#pragma unroll
    for (int j = 0; j < G; j++) { m[j] = -1e30f; l[j] = 0.0f; o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
    const size_t hb = (size_t)hk * ctx;
    for (int c0 = p0 + 32 * warp; c0 < p1; c0 += 256) {
        const int p = c0 + lane;
        const bool live = p < p1;
        float s[G];
#pragma unroll
        for (int j = 0; j < G; j++) s[j] = 0.0f;
        unsigned int vr[32];                            /* V of the chunk: loaded with K, used after the softmax */
        float vs[32];
#pragma unroll
        for (int t = 0; t < 32; t++) {
            vr[t] = *reinterpret_cast<const unsigned int*>(vq + (hb + c0 + t) * 128 + 4 * lane);
            vs[t] = __half2float(vd[(hb + c0 + t) * 4 + (lane >> 3)]);
        }
        const int4* kr = reinterpret_cast<const int4*>(kq + (hb + p) * 128);
        const uint2 dk2 = *reinterpret_cast<const uint2*>(kd + (hb + p) * 4);
        const __half* dkh = reinterpret_cast<const __half*>(&dk2);
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const int4 a = kr[2 * b], c = kr[2 * b + 1];
            const float dk = __half2float(dkh[b]);
#pragma unroll
            for (int j = 0; j < G; j++) {
                const int4* qw = reinterpret_cast<const int4*>(&qc[j][32 * b]);
                const int4 qa = qw[0], qb = qw[1];
                int dot = __dp4a(a.x, qa.x, 0);
                dot = __dp4a(a.y, qa.y, dot); dot = __dp4a(a.z, qa.z, dot); dot = __dp4a(a.w, qa.w, dot);
                dot = __dp4a(c.x, qb.x, dot); dot = __dp4a(c.y, qb.y, dot); dot = __dp4a(c.z, qb.z, dot);
                dot = __dp4a(c.w, qb.w, dot);
                s[j] += (float)dot * (qd[j][b] * dk);
            }
        }
#pragma unroll
        for (int j = 0; j < G; j++) {
            float mx = warp_max(live ? s[j] : -1e30f);
            float mn = fmaxf(m[j], mx);
            float f = expf(m[j] - mn);
            float pr = live ? expf(s[j] - mn) : 0.0f;
            s[j] = pr;
            l[j] = l[j] * f + warp_sum(pr);
            m[j] = mn;
            o[j][0] *= f; o[j][1] *= f; o[j][2] *= f; o[j][3] *= f;
        }
        /* P V: rows past the end have p = 0 (the cache is allocated to a multiple of 256 positions) */
#pragma unroll
        for (int t = 0; t < 32; t++) {
            const unsigned int r = vr[t];
            const float d = vs[t];
            const float v0 = (float)(signed char)(r & 255u) * d, v1 = (float)(signed char)((r >> 8) & 255u) * d;
            const float v2 = (float)(signed char)((r >> 16) & 255u) * d, v3 = (float)(signed char)(r >> 24) * d;
#pragma unroll
            for (int j = 0; j < G; j++) {
                float pr = __shfl_sync(0xffffffffu, s[j], t);
                o[j][0] += pr * v0; o[j][1] += pr * v1; o[j][2] += pr * v2; o[j][3] += pr * v3;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < G; j++) {
        wo[warp][j][4 * lane] = o[j][0]; wo[warp][j][4 * lane + 1] = o[j][1];
        wo[warp][j][4 * lane + 2] = o[j][2]; wo[warp][j][4 * lane + 3] = o[j][3];
        if (lane == 0) { wm[warp][j] = m[j]; wl[warp][j] = l[j]; }
    }
    __syncthreads();
    for (int i = tid; i < G * 128; i += 256) {
        int j = i >> 7, d = i & 127, h = hk * G + j;
        float mx = -1e30f;
#pragma unroll
        for (int w = 0; w < 8; w++) mx = fmaxf(mx, wm[w][j]);
        float ov = 0.0f, lv = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; w++) { float f = expf(wm[w][j] - mx); ov += f * wo[w][j][d]; lv += f * wl[w][j]; }
        part_o[((size_t)h * nsp + sp) * 128 + d] = ov;
        if (d == 0) { part_ms[(h * nsp + sp) * 2] = mx; part_ms[(h * nsp + sp) * 2 + 1] = lv; }
    }
}
#define NBQ_ATTN_Q8(G, NAME) extern "C" __global__ void __launch_bounds__(256, NBQ_ATT_MINB) NAME(const float* q, \
    const signed char* kq, const __half* kd, const signed char* vq, const __half* vd, const int* pos, int ctx, \
    float scale, float* part_o, float* part_ms, int qstride) { \
    nbq_attn_q8_t<G>(q, kq, kd, vq, vd, pos, ctx, scale, part_o, part_ms, qstride); }
NBQ_ATTN_Q8(7, nbq_attn_q8)
NBQ_ATTN_Q8(5, nbq_attn_q8_g5)
