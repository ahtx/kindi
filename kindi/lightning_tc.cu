/* lightning_tc.cu - the prompt matrix product of lightning.cu on int8 tensor cores (mma.sync m16n8k32, sm_80+;
   compiled as compute_75 PTX marked PTX 7.1 under NVRTC 10.2, as kindi/tc.py does). With LT_SM75 (Turing) each
   m16n8k32 is four m8n8k16 and each fp16 m16n8k16 two m16n8k8, on the same fragments: the same numbers.

   Y[t][r] = sum over runs k of d_w[r][k / 2] * dx[t][k] * (sum(c * q) - sx[t][k]) for the Q2_0 rows r and the
   tokens t (the same numbers as lt_mm_q2: each run's integer sum is exact). Block: 256 threads, 128 rows x 64
   tokens, K in steps of 128 columns (4 runs). The codes of a step are expanded once into shared memory as int8
   digits 0 / 1 / 2 in the lane order of the inputs (both operands share the order inside a run, so the dot products
   are unchanged); 8 warps (4 x 2) each take 32 rows x 32 tokens: 2 x 4 m16n8k32 tiles per run. The accumulator
   starts at the integer 0x4B400000, so its bits read as a float are 12582912 + dot (|dot| < 2^22), as in nbq_mm.cu.
   Grouped (MoE) form as lt_mm_q2: blockIdx.z = expert, its tokens = entries [off[e], off[e] + cnt[e]) of a list. */
typedef unsigned int u32;
typedef unsigned char u8;

#define LT_M 0x03030303u
#define INFINITY __int_as_float(0x7f800000)
#define LT_MAGIC 0x4B400000
#define LT_BK 128
#define LT_LD (LT_BK + 16)                     /* shared row stride (bytes): the fragment loads hit 32 banks */
#ifdef LT_SM75
/* sm_75 (Turing) has no m16n8k32: four m8n8k16 on the same fragments (rows gid / gid + 8 x k 0..15 / 16..31) */
#define LT_MMA8(c0, c1, a, b) \
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" \
        : "+r"(c0), "+r"(c1) : "r"(a), "r"(b))
#define LT_MMA(c, a0, a1, a2, a3, b0, b1) do { \
        LT_MMA8(c[0], c[1], a0, b0); LT_MMA8(c[0], c[1], a2, b1); \
        LT_MMA8(c[2], c[3], a1, b0); LT_MMA8(c[2], c[3], a3, b1); } while (0)
#else
#define LT_MMA(c, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#endif

__device__ __forceinline__ u32 lt_sh(const void* p) {
    u32 r; asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(r) : "l"(p)); return r;
}
__device__ __forceinline__ float lt_dotf(int c) { return __int_as_float(c) - 12582912.0f; }
__device__ __forceinline__ float lt_h2f(unsigned short h) { float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f; }

/* TN = tokens per block: 64 (8 warps as 4 x 2, each 32 rows x 32 tokens) or 32 (8 warps as 8 x 1, each 16 rows x 32
   tokens: the MoE groups, which hold few tokens) */
template <int TN>
__device__ __forceinline__ void lt_mm_tc_t(
        const u32* const* __restrict__ codes, const unsigned short* const* __restrict__ sc, int rows, int cols,
        const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow, int T,
        const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, int ldy, int acc) {
    const int MI = TN == 64 ? 2 : 1;                         /* 16-row tiles per warp */
    __shared__ __align__(16) u8 As[128 * LT_LD];
    __shared__ __align__(16) u8 Bs[TN * LT_LD];
    __shared__ float Asc[128 * 4];
    __shared__ float Bsc[TN * 4];
    __shared__ float Bsx[TN * 4];
    __shared__ int Brow[TN];
    const int e = blockIdx.z;
    const int n = cnt ? cnt[e] : T;
    const int c0 = blockIdx.y * TN;
    if (c0 >= n) return;
    const int o = cnt ? off[e] : 0;
    const int r0 = blockIdx.x * 128;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, t4 = lane & 3;
    const int wm = TN == 64 ? (warp & 3) * 32 : warp * 16, wn = TN == 64 ? (warp >> 2) * 32 : 0;
    const int runs = cols >> 5;
    const u32* cw = codes[e];
    const unsigned short* cs = sc[e];
    if (tid < TN) {
        const int c = c0 + tid;
        Brow[tid] = c < n ? (xrow ? xrow[o + c] : o + c) : -1;
    }
    float a[MI][4][4];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) a[i][j][k] = 0.0f;
    __syncthreads();                                        /* Brow */
    /* each step's global words go to registers one step ahead, so the loads overlap the mma of the step before */
    uint2 pw[2];
    float ps[2];
    uint4 pv[TN / 32];
    float pd = 0.0f, px = 0.0f;
#define LT_FETCH(k0_) do { \
        for (int h_ = 0; h_ < 2; h_++) { \
            const int idx_ = tid + 256 * h_, r_ = idx_ >> 2, u_ = idx_ & 3; \
            const int row_ = min(r0 + r_, rows - 1), run_ = ((k0_) >> 5) + u_; \
            pw[h_] = run_ < runs ? *reinterpret_cast<const uint2*>(cw + (long long)row_ * (cols >> 4) + run_ * 2) \
                                 : make_uint2(0u, 0u); \
            ps[h_] = run_ < runs ? lt_h2f(cs[(long long)row_ * (cols >> 6) + (run_ >> 1)]) : 0.0f; \
        } \
        for (int h_ = 0; h_ < TN / 32; h_++) { \
            const int idx_ = tid + 256 * h_, t_ = idx_ >> 3, pc_ = idx_ & 7; \
            const int xr_ = Brow[t_], run_ = ((k0_) >> 5) + (pc_ >> 1); \
            pv[h_] = xr_ >= 0 && run_ < runs \
                ? *reinterpret_cast<const uint4*>(xq + ((long long)xr_ * runs + run_) * 8 + 4 * (pc_ & 1)) \
                : make_uint4(0u, 0u, 0u, 0u); \
        } \
        if (tid < TN * 4) { \
            const int t_ = tid >> 2, u_ = tid & 3; \
            const int xr_ = Brow[t_], run_ = ((k0_) >> 5) + u_; \
            const bool ok_ = xr_ >= 0 && run_ < runs; \
            pd = ok_ ? dx[(long long)xr_ * runs + run_] : 0.0f; \
            px = ok_ ? (float)sx[(long long)xr_ * runs + run_] : 0.0f; \
        } \
    } while (0)
    LT_FETCH(0);
    for (int k0 = 0; k0 < cols; k0 += LT_BK) {
        __syncthreads();
        /* the codes: 128 rows x 4 runs, two (row, run) pieces per thread -> 32 bytes of digits each */
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int idx = tid + 256 * h, r = idx >> 2, u = idx & 3;
            const uint2 w = pw[h];
            uint4* dst = reinterpret_cast<uint4*>(As + r * LT_LD + 32 * u);
            dst[0] = make_uint4(w.x & LT_M, (w.x >> 2) & LT_M, (w.x >> 4) & LT_M, (w.x >> 6) & LT_M);
            dst[1] = make_uint4(w.y & LT_M, (w.y >> 2) & LT_M, (w.y >> 4) & LT_M, (w.y >> 6) & LT_M);
            Asc[r * 4 + u] = ps[h];
        }
        /* the inputs: TN tokens x 4 runs x 32 bytes, 16-byte pieces */
#pragma unroll
        for (int h = 0; h < TN / 32; h++) {
            const int idx = tid + 256 * h, t = idx >> 3, pc = idx & 7;
            *reinterpret_cast<uint4*>(Bs + t * LT_LD + 16 * pc) = pv[h];
        }
        if (tid < TN * 4) {
            const int t = tid >> 2, u = tid & 3;
            Bsc[t * 4 + u] = pd;
            Bsx[t * 4 + u] = px;
        }
        __syncthreads();
        if (k0 + LT_BK < cols) LT_FETCH(k0 + LT_BK);
#pragma unroll 1
        for (int u = 0; u < 4; u++) {
            u32 fa[MI][4], fb[4][2];
            const int l8 = lane & 7, lq = lane >> 3;
#pragma unroll
            for (int i = 0; i < MI; i++) {
                const u32 ad = lt_sh(As + (wm + 16 * i + l8 + 8 * (lq & 1)) * LT_LD + 32 * u + 16 * (lq >> 1));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(fa[i][0]), "=r"(fa[i][1]), "=r"(fa[i][2]), "=r"(fa[i][3]) : "r"(ad));
            }
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const u32 ad = lt_sh(Bs + (wn + 8 * j + l8 + 8 * (lq >> 1)) * LT_LD + 32 * u + 16 * (lq & 1));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(fb[j][0]), "=r"(fb[j][1]), "=r"(fb[j + 1][0]), "=r"(fb[j + 1][1]) : "r"(ad));
            }
            float xa[4], xb[4], sa[4], sb[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ta = wn + 8 * j + 2 * t4;
                xa[j] = Bsc[ta * 4 + u]; xb[j] = Bsc[(ta + 1) * 4 + u];
                sa[j] = Bsx[ta * 4 + u]; sb[j] = Bsx[(ta + 1) * 4 + u];
            }
#pragma unroll
            for (int i = 0; i < MI; i++) {
                const int ra = wm + 16 * i + gid, rb = ra + 8;
                const float wa = Asc[ra * 4 + u], wb = Asc[rb * 4 + u];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int c[4] = {LT_MAGIC, LT_MAGIC, LT_MAGIC, LT_MAGIC};
                    LT_MMA(c, fa[i][0], fa[i][1], fa[i][2], fa[i][3], fb[j][0], fb[j][1]);
                    a[i][j][0] += wa * (xa[j] * (lt_dotf(c[0]) - sa[j]));
                    a[i][j][1] += wa * (xb[j] * (lt_dotf(c[1]) - sb[j]));
                    a[i][j][2] += wb * (xa[j] * (lt_dotf(c[2]) - sa[j]));
                    a[i][j][3] += wb * (xb[j] * (lt_dotf(c[3]) - sb[j]));
                }
            }
        }
    }
#undef LT_FETCH
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const int r = r0 + wm + 16 * i + gid + 8 * (k >> 1);
                const int c = c0 + wn + 8 * j + 2 * t4 + (k & 1);
                if (r < rows && c < n) {
                    float* p = y + (long long)(o + c) * ldy + r;
                    *p = acc ? *p + a[i][j][k] : a[i][j][k];
                }
            }
}

#define LT_MM_ARGS const u32* const* __restrict__ codes, const unsigned short* const* __restrict__ sc, \
    int rows, int cols, const int* __restrict__ cnt, const int* __restrict__ off, const int* __restrict__ xrow, \
    int T, const int* __restrict__ xq, const float* __restrict__ dx, const int* __restrict__ sx, float* y, \
    int ldy, int acc
extern "C" __global__ void __launch_bounds__(256) lt_mm_tc(LT_MM_ARGS) {
    lt_mm_tc_t<64>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, sx, y, ldy, acc); }
extern "C" __global__ void __launch_bounds__(256) lt_mm_tc32(LT_MM_ARGS) {
    lt_mm_tc_t<32>(codes, sc, rows, cols, cnt, off, xrow, T, xq, dx, sx, y, ldy, acc); }

/* prompt attention on fp16 tensor cores (nbq_fa.cu's FA2 kernel for 16 query heads per K/V head and the
   lightning.cu cache layout [pos][kvh * 128]). Query rows are (token, head of the K/V group): row r = 16 t + j.
   grid (ceil(16 T / 64), nkv), 128 threads; warp w takes rows 16 w .. +15 against tiles of 64 positions, with a
   running max and sum per row. Q is scaled, then rounded to fp16; P is rounded to fp16 for P V; the sums are fp32.
   Token t sits at position *ppos + off + t. */
#include <cuda_fp16.h>
#define LT_FLD 136                                      /* shared row stride in halves */
#ifdef LT_SM75
/* sm_75: two m16n8k8 (k 0..7: a0 a1 b0; k 8..15: a2 a3 b1) */
#define LT_MMAH8(c, a0, a1, b) \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b))
#define LT_MMAH(c, a0, a1, a2, a3, b0, b1) do { LT_MMAH8(c, a0, a1, b0); LT_MMAH8(c, a2, a3, b1); } while (0)
#else
#define LT_MMAH(c, a0, a1, a2, a3, b0, b1) \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#endif

__device__ __forceinline__ u32 lt_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<u32*>(&h);
}

extern "C" __global__ void __launch_bounds__(128) lt_fa(const float* __restrict__ qkv, int ld, int nt,
        const int* __restrict__ ppos, int off, const __half* __restrict__ kc, const __half* __restrict__ vc,
        float scale, float* out) {
    __shared__ __align__(16) __half Ks[64 * LT_FLD];
    __shared__ __align__(16) __half Vs[64 * LT_FLD];
    const int G = 16;
    const int hk = blockIdx.y, nkv = gridDim.y, kw = nkv * 128;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, t4 = lane & 3;
    const int pos0 = *ppos + off;
    const int R = G * nt, r0 = blockIdx.x * 64;
    const int ra = r0 + 16 * warp + gid, rb = ra + 8;
    const int ca = min(ra, R - 1), cb = min(rb, R - 1);
    const int ta = ca / G, tb = cb / G;
    const int la = pos0 + ta, lb = pos0 + tb;
    const int lmax = pos0 + min(r0 + 63, R - 1) / G;
    u32 qa[8][4];
    {
        const float* pa = qkv + (long long)ta * ld + (hk * G + ca % G) * 128;
        const float* pb = qkv + (long long)tb * ld + (hk * G + cb % G) * 128;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int d = 16 * k + 2 * t4;
            qa[k][0] = lt_h2(pa[d] * scale, pa[d + 1] * scale);
            qa[k][1] = lt_h2(pb[d] * scale, pb[d + 1] * scale);
            qa[k][2] = lt_h2(pa[d + 8] * scale, pa[d + 9] * scale);
            qa[k][3] = lt_h2(pb[d + 8] * scale, pb[d + 9] * scale);
        }
    }
    float o[16][4];
#pragma unroll
    for (int n = 0; n < 16; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float ma = -1e30f, mb = -1e30f, sa = 0.0f, sb = 0.0f;
    for (int p0 = 0; p0 <= lmax; p0 += 64) {
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 8; h++) {                   /* K and V tiles: 64 positions x 16 pieces of 8 halves */
            const int idx = tid + 128 * h, p = idx >> 4, pc = idx & 15;
            const long long g = (long long)(p0 + p) * kw + hk * 128 + 8 * pc;
            *reinterpret_cast<uint4*>(Ks + p * LT_FLD + 8 * pc) = *reinterpret_cast<const uint4*>(kc + g);
            *reinterpret_cast<uint4*>(Vs + p * LT_FLD + 8 * pc) = *reinterpret_cast<const uint4*>(vc + g);
        }
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
            const __half* kr = Ks + (8 * n + gid) * LT_FLD + 2 * t4;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u32 b0 = *reinterpret_cast<const u32*>(kr + 16 * k), b1 = *reinterpret_cast<const u32*>(kr + 16 * k + 8);
                LT_MMAH(s[n], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
            }
        }
        float mxa = -1e30f, mxb = -1e30f;
#pragma unroll
        for (int n = 0; n < 8; n++) {
            const int pc = p0 + 8 * n + 2 * t4;
            if (pc > la) s[n][0] = -1e30f;
            if (pc + 1 > la) s[n][1] = -1e30f;
            if (pc > lb) s[n][2] = -1e30f;
            if (pc + 1 > lb) s[n][3] = -1e30f;
            mxa = fmaxf(mxa, fmaxf(s[n][0], s[n][1]));
            mxb = fmaxf(mxb, fmaxf(s[n][2], s[n][3]));
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
        const float fa = __expf(ma - na), fb = __expf(mb - nb);
        ma = na; mb = nb;
        sa *= fa; sb *= fb;
#pragma unroll
        for (int n = 0; n < 16; n++) { o[n][0] *= fa; o[n][1] *= fa; o[n][2] *= fb; o[n][3] *= fb; }
        u32 pf[4][4];
#pragma unroll
        for (int n = 0; n < 8; n++) {
            const float e0 = __expf(s[n][0] - na), e1 = __expf(s[n][1] - na);
            const float e2 = __expf(s[n][2] - nb), e3 = __expf(s[n][3] - nb);
            sa += e0 + e1; sb += e2 + e3;
            pf[n >> 1][(n & 1) * 2] = lt_h2(e0, e1);
            pf[n >> 1][(n & 1) * 2 + 1] = lt_h2(e2, e3);
        }
        const int m8 = lane >> 3, r8 = lane & 7;
#pragma unroll
        for (int k = 0; k < 4; k++) {
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                const __half* addr = Vs + (16 * k + (m8 & 1) * 8 + r8) * LT_FLD + 8 * n + (m8 >> 1) * 8;
                u32 b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(lt_sh(addr)));
                LT_MMAH(o[n], pf[k][0], pf[k][1], pf[k][2], pf[k][3], b0, b1);
                LT_MMAH(o[n + 1], pf[k][0], pf[k][1], pf[k][2], pf[k][3], b2, b3);
            }
        }
    }
    sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
    sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
    const float ia = 1.0f / sa, ib = 1.0f / sb;
    const int lo = nkv * G * 128;
    float* oa = out + (long long)ta * lo + (hk * G + ca % G) * 128;
    float* ob = out + (long long)tb * lo + (hk * G + cb % G) * 128;
#pragma unroll
    for (int n = 0; n < 16; n++) {
        const int d = 8 * n + 2 * t4;
        if (ra < R) { oa[d] = o[n][0] * ia; oa[d + 1] = o[n][1] * ia; }
        if (rb < R) { ob[d] = o[n][2] * ib; ob[d + 1] = o[n][3] * ib; }
    }
}

/* decode attention on fp16 tensor cores: the 16 query heads of a K/V head are the 16 rows of one m16n8k16 tile.
   grid (nsplit, nkv), one warp per block: the keys 0 .. *ppos are cut in nsplit ranges of whole 64-key tiles; each
   block writes (m, l, acc[128]) per head for its range, unnormalized, in lt_attn_dec's layout (lt_attn_merge adds
   the ranges). Q is scaled, then rounded to fp16; P is rounded to fp16 for P V; the sums are fp32. */
extern "C" __global__ void __launch_bounds__(32) lt_fa_dec(const float* __restrict__ q, const __half* __restrict__ kc,
        const __half* __restrict__ vc, const int* __restrict__ ppos, float scale, float* pm, float* pl, float* pacc) {
    __shared__ __align__(16) __half Ks[64 * LT_FLD];
    __shared__ __align__(16) __half Vs[64 * LT_FLD];
    const int G = 16;
    const int sp = blockIdx.x, ns = gridDim.x, hk = blockIdx.y, nkv = gridDim.y, kw = nkv * 128;
    const int lane = threadIdx.x, gid = lane >> 2, t4 = lane & 3;
    const int n = *ppos + 1;
    const int tiles = (n + 63) / 64, per = (tiles + ns - 1) / ns;
    const int t0 = sp * per, t1 = min(tiles, t0 + per);
    const int ha = hk * G + gid, hb = ha + 8;
    if (t0 >= t1) {
        if (t4 == 0) {
            pm[(long long)ha * ns + sp] = -INFINITY; pl[(long long)ha * ns + sp] = 0.0f;
            pm[(long long)hb * ns + sp] = -INFINITY; pl[(long long)hb * ns + sp] = 0.0f;
        }
        for (int d = 2 * t4; d < 128; d += 8) {
            pacc[((long long)ha * ns + sp) * 128 + d] = 0.0f; pacc[((long long)ha * ns + sp) * 128 + d + 1] = 0.0f;
            pacc[((long long)hb * ns + sp) * 128 + d] = 0.0f; pacc[((long long)hb * ns + sp) * 128 + d + 1] = 0.0f;
        }
        return;
    }
    u32 qa[8][4];
    {
        const float* pa = q + ha * 128;
        const float* pb = q + hb * 128;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int d = 16 * k + 2 * t4;
            qa[k][0] = lt_h2(pa[d] * scale, pa[d + 1] * scale);
            qa[k][1] = lt_h2(pb[d] * scale, pb[d + 1] * scale);
            qa[k][2] = lt_h2(pa[d + 8] * scale, pa[d + 9] * scale);
            qa[k][3] = lt_h2(pb[d + 8] * scale, pb[d + 9] * scale);
        }
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float ma = -1e30f, mb = -1e30f, sa = 0.0f, sb = 0.0f;
    for (int tt = t0; tt < t1; tt++) {
        const int p0 = tt * 64;
        __syncwarp();
        for (int i = lane; i < 64 * 16; i += 32) {
            const int p = i >> 4, pc = i & 15;
            const long long g = (long long)(p0 + p) * kw + hk * 128 + 8 * pc;
            *reinterpret_cast<uint4*>(Ks + p * LT_FLD + 8 * pc) = *reinterpret_cast<const uint4*>(kc + g);
            *reinterpret_cast<uint4*>(Vs + p * LT_FLD + 8 * pc) = *reinterpret_cast<const uint4*>(vc + g);
        }
        __syncwarp();
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const __half* kr = Ks + (8 * j + gid) * LT_FLD + 2 * t4;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const u32 b0 = *reinterpret_cast<const u32*>(kr + 16 * k), b1 = *reinterpret_cast<const u32*>(kr + 16 * k + 8);
                LT_MMAH(s[j], qa[k][0], qa[k][1], qa[k][2], qa[k][3], b0, b1);
            }
        }
        float mxa = -1e30f, mxb = -1e30f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int pc = p0 + 8 * j + 2 * t4;
            if (pc >= n) { s[j][0] = -1e30f; s[j][2] = -1e30f; }
            if (pc + 1 >= n) { s[j][1] = -1e30f; s[j][3] = -1e30f; }
            mxa = fmaxf(mxa, fmaxf(s[j][0], s[j][1]));
            mxb = fmaxf(mxb, fmaxf(s[j][2], s[j][3]));
        }
        mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1)); mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
        mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1)); mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
        const float na = fmaxf(ma, mxa), nb = fmaxf(mb, mxb);
        const float fa = __expf(ma - na), fb = __expf(mb - nb);
        ma = na; mb = nb;
        sa *= fa; sb *= fb;
#pragma unroll
        for (int i = 0; i < 16; i++) { o[i][0] *= fa; o[i][1] *= fa; o[i][2] *= fb; o[i][3] *= fb; }
        u32 pf[4][4];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float e0 = __expf(s[j][0] - na), e1 = __expf(s[j][1] - na);
            const float e2 = __expf(s[j][2] - nb), e3 = __expf(s[j][3] - nb);
            sa += e0 + e1; sb += e2 + e3;
            pf[j >> 1][(j & 1) * 2] = lt_h2(e0, e1);
            pf[j >> 1][(j & 1) * 2 + 1] = lt_h2(e2, e3);
        }
        const int m8 = lane >> 3, r8 = lane & 7;
#pragma unroll
        for (int k = 0; k < 4; k++) {
#pragma unroll
            for (int i = 0; i < 16; i += 2) {
                const __half* addr = Vs + (16 * k + (m8 & 1) * 8 + r8) * LT_FLD + 8 * i + (m8 >> 1) * 8;
                u32 b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(lt_sh(addr)));
                LT_MMAH(o[i], pf[k][0], pf[k][1], pf[k][2], pf[k][3], b0, b1);
                LT_MMAH(o[i + 1], pf[k][0], pf[k][1], pf[k][2], pf[k][3], b2, b3);
            }
        }
    }
    sa += __shfl_xor_sync(0xffffffffu, sa, 1); sa += __shfl_xor_sync(0xffffffffu, sa, 2);
    sb += __shfl_xor_sync(0xffffffffu, sb, 1); sb += __shfl_xor_sync(0xffffffffu, sb, 2);
    if (t4 == 0) {
        pm[(long long)ha * ns + sp] = ma; pl[(long long)ha * ns + sp] = sa;
        pm[(long long)hb * ns + sp] = mb; pl[(long long)hb * ns + sp] = sb;
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const int d = 8 * i + 2 * t4;
        pacc[((long long)ha * ns + sp) * 128 + d] = o[i][0]; pacc[((long long)ha * ns + sp) * 128 + d + 1] = o[i][1];
        pacc[((long long)hb * ns + sp) * 128 + d] = o[i][2]; pacc[((long long)hb * ns + sp) * 128 + d + 1] = o[i][3];
    }
}
