/* nbq.cu - Kindi kernels for llama-architecture GGUF models with K-quant and I-quant weights (Nanbeige 4.3 33B
   IQ3_XXS: IQ3_XXS FFN, IQ2_S Q/K, Q4_K V, IQ3_S O and embedding, Q5_K head).

   Numbers follow the CUDA build of llama.cpp: the input of each matrix product is quantized to int8 in runs of 32
   (scale = max |x| / 127, q = round(x / scale)), each run gives an exact int32 dot product, and the int32 is scaled
   in fp32. The module is compiled with --use_fast_math, as llama.cpp is.

   Weight layouts. The I-quant types are repacked at load time: the fp16 block scale moves to its own array, so the
   rest of the block is 4-byte aligned (IQ3_XXS 96 bytes, IQ3_S 108, IQ2_S 80). Q4_K and Q5_K stay as in the file
   (144 and 176 bytes, 16-byte aligned).

   The decode kernels read the position and the token from device memory, so one CUDA graph replays every step. */
#include <cuda_fp16.h>
#if defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ < 11     /* NVRTC 10.2 (the release image) */
static __device__ __forceinline__ unsigned long long __cvta_generic_to_shared(const void* p) {
    unsigned long long r; asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p)); return r; }
/* NVRTC 10.2 does not define the __dp4a intrinsic (sm_61_intrinsics are not included): the same instruction
   (cards before sm_61: the byte products below) */
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
static __device__ __forceinline__ int nbq_dp4a(int a, int b, int c) {
    int r; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
#define __dp4a nbq_dp4a
#endif
#endif
#ifndef NBQ_TC                /* 0 before sm_80: no tensor-core kernels (the prompt then goes through Engine.check) */
#define NBQ_TC 1
#endif

/* Kepler (sm_35, sm_37) has no __dp4a and no tensor cores. __dp4a is then four byte products (exact); the
   tensor-core kernels compile to a trap and nanbeige.py does not launch them. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int nbq_dp4a(int a, int b, int c) {
    return c + (a << 24 >> 24) * (b << 24 >> 24) + (a << 16 >> 24) * (b << 16 >> 24) + (a << 8 >> 24) * (b << 8 >> 24)
             + (a >> 24) * (b >> 24);
}
#define __dp4a nbq_dp4a
#endif
/* the tensor-core asm only where the tensor-core kernels are built (NBQ_TC: sm_80+; in the CUDA 10.2 image they
   are compiled as compute_75 PTX for an sm_8x card, so __CUDA_ARCH__ does not tell) */
#if NBQ_TC
#define NBQ_ASM(...) asm volatile(__VA_ARGS__)
#else
#define NBQ_ASM(...) __trap()
#endif

typedef unsigned char u8;
typedef unsigned int u32;

#define T_Q4K 12
#define T_Q5K 13
#define T_IQ3XXS 18
#define T_IQ3S 21
#define T_IQ2S 22

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

/* 4 sign bits -> the 4 value bytes of g negated where the bit is set. No byte of a codebook entry is 0, so
   (g ^ 0xff) + 1 = 256 - g stays inside its byte: a plain 32-bit add does it. */
__device__ __forceinline__ int nbq_sign4(u32 g, u32 s) {
    u32 m1 = ((s & 15u) * 0x204081u) & 0x01010101u;
    return (int)((g ^ (m1 * 0xffu)) + m1);
}
__device__ __forceinline__ u32 nbq_ld32(const u8* p) { return *reinterpret_cast<const u32*>(p); }
/* the float ops shared by every decode path, with explicit rounding: one run's contribution to a dot, the RoPE pair */
__device__ __forceinline__ float nbq_runf(float s0, int lo, float s1, int hi) {
    return __fmaf_rn(s1, (float)hi, __fmul_rn(s0, (float)lo));
}
/* The int8 dot products of a run. Kepler (NBQ_K35): its integer multiply is slow (32 per clock and SMX against
   192 fp32 FMAs), so the bytes become exact floats and the products are fp32 FMAs. Every product and every sum of
   16 of them is an integer under 2^24, so the float sums equal the int32 dot products and the results are the same
   bits as on the dp4a cards. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610 && !defined(NBQ_K35_INT)   /* NBQ_K35_INT: tests */
#define NBQ_K35 1
/* 4 signed bytes -> 4 floats: bias each byte to 0..255, put it in the low mantissa of 2^23 (bytes 0, 1 at their
   own place), and one FMA removes the bias and the place value */
__device__ __forceinline__ void nbq_f4(int v, float* f) {
    const u32 u = (u32)v ^ 0x80808080u, w = u >> 16;
    f[0] = __int_as_float((u & 0xffu) | 0x4b000000u) - 8388736.0f;
    f[1] = __fmaf_rn(__int_as_float((u & 0xff00u) | 0x4b000000u), 0.00390625f, -32896.0f);
    f[2] = __int_as_float((w & 0xffu) | 0x4b000000u) - 8388736.0f;
    f[3] = __fmaf_rn(__int_as_float((w & 0xff00u) | 0x4b000000u), 0.00390625f, -32896.0f);
}
__device__ __forceinline__ void nbq_xf(int4 a, int4 c, float* x) {
    nbq_f4(a.x, x); nbq_f4(a.y, x + 4); nbq_f4(a.z, x + 8); nbq_f4(a.w, x + 12);
    nbq_f4(c.x, x + 16); nbq_f4(c.y, x + 20); nbq_f4(c.z, x + 24); nbq_f4(c.w, x + 28);
}
/* the dot of 16 weights (4 ints of v) with 16 floats */
__device__ __forceinline__ float nbq_fdot16(const int* v, const float* x) {
    float s = 0.0f, w[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        nbq_f4(v[i], w);
        s = __fmaf_rn(w[0], x[4 * i], s); s = __fmaf_rn(w[1], x[4 * i + 1], s);
        s = __fmaf_rn(w[2], x[4 * i + 2], s); s = __fmaf_rn(w[3], x[4 * i + 3], s);
    }
    return s;
}
__device__ __forceinline__ float nbq_fsum32(const float* x) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 32; i++) s += x[i];
    return s;
}
__device__ __forceinline__ float nbq_runf(float s0, float lo, float s1, float hi) {
    return __fmaf_rn(s1, hi, __fmul_rn(s0, lo));
}
/* GK110 keeps global loads in L2 only; the inputs, which every warp reads, go through the read-only cache */
#define NBQ_LDG(p) __ldg(p)
#define NBQ_XF(a, c, xf) float xf[32]; nbq_xf(a, c, xf)
#define NBQ_DOT(v, a, c, xf, lo, hi) float lo = nbq_fdot16(v, xf), hi = nbq_fdot16(v + 4, xf + 16)
#define NBQ_SX(a, c, xf) nbq_fsum32(xf)
#else
#define NBQ_LDG(p) (*(p))
#define NBQ_XF(a, c, xf)
#define NBQ_DOT(v, a, c, xf, lo, hi)     int lo = __dp4a(v[3], a.w, __dp4a(v[2], a.z, __dp4a(v[1], a.y, __dp4a(v[0], a.x, 0))));     int hi = __dp4a(v[7], c.w, __dp4a(v[6], c.z, __dp4a(v[5], c.y, __dp4a(v[4], c.x, 0))))
#define NBQ_SX(a, c, xf) (__dp4a(0x01010101, c.x, __dp4a(0x01010101, c.y, __dp4a(0x01010101, c.z, __dp4a(0x01010101, c.w,     __dp4a(0x01010101, a.x, __dp4a(0x01010101, a.y, __dp4a(0x01010101, a.z, __dp4a(0x01010101, a.w, 0)))))))))
#endif
__device__ __forceinline__ void nbq_rot(float x0, float x1, float c, float s, float& y0, float& y1) {
    y0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    y1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
}


/* body bytes of one block (the I-quant fp16 scales are in their own array) */
template <int T> struct BT;
template <> struct BT<T_Q4K>    { enum { B = 144 }; };
template <> struct BT<T_Q5K>    { enum { B = 176 }; };
template <> struct BT<T_IQ3XXS> { enum { B = 96 }; };
template <> struct BT<T_IQ3S>   { enum { B = 108 }; };
template <> struct BT<T_IQ2S>   { enum { B = 80 }; };

// 6-bit scale d and min m number j (0..7) of a Q4_K/Q5_K block from its 12 scale bytes. With k = j mod 4 and
// u = j / 4, the bytes x = sc[k], y = sc[k + 4], z = sc[k + 8] give: low 4 bits from x / y (u = 0) or from the low /
// high nibble of z (u = 1); bits 4-5 from bits 4-5 (u = 0) or 6-7 (u = 1) of x (scale) and y (min).
__device__ __forceinline__ void kq_scale_min(const u8* sc, int j, int& d, int& m) {
    const int k = j & 3, u = j >> 2;
    const unsigned x = sc[k], y = sc[k + 4], z = sc[k + 8], sh = 2 * u;
    d = (int)(((u ? z : x) & 15u) | ((x >> sh) & 48u));
    m = (int)(((u ? z >> 4 : y) & 15u) | ((y >> sh) & 48u));
}

/* Decode run `sub` (0..7, 32 weights) of one block: v = the 32 int8 values (4 per int), s0 the scale of values
   0..15, s1 of 16..31, mn the minimum (weight = s * v - mn; K-quants only). */
/* the codebook of type T in shared memory (IQ3_XXS 1 KB, IQ3_S 2 KB, IQ2_S 8 KB); every thread of the block calls it */
template <int T> struct Tab { typedef u32 E; enum { N = 1 }; };
template <> struct Tab<T_IQ3XXS> { typedef u32 E; enum { N = 256 }; };
template <> struct Tab<T_IQ3S> { typedef u32 E; enum { N = 512 }; };
template <> struct Tab<T_IQ2S> { typedef unsigned long long E; enum { N = 1024 }; };
template <int T>
__device__ __forceinline__ const typename Tab<T>::E* nbq_tab(typename Tab<T>::E* sh) {
    if (T == T_IQ3XXS) for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = nbq_iq3xxs_grid[i];
    if (T == T_IQ3S) for (int i = threadIdx.x; i < 512; i += blockDim.x) sh[i] = nbq_iq3s_grid[i];
    if (T == T_IQ2S) for (int i = threadIdx.x; i < 1024; i += blockDim.x) sh[i] = nbq_iq2s_grid[i];
    __syncthreads();
    return sh;
}
#define NBQ_TAB(T, name) __shared__ typename Tab<T>::E name##_s[Tab<T>::N]; const typename Tab<T>::E* name = nbq_tab<T>(name##_s)

/* NBQ_REP (nanbeige.py, sm_80+): the decode products of IQ3_XXS read the codebook from 32 copies, entry i of copy
   l at word 32 i + l, so the lanes of a warp never meet in a bank (the 32 KB table replaces the 1 KB one in nbq_mv
   and nbq_mv2 only; the other kernels keep theirs). The values are the same, so the products are bit-identical. */
#ifndef NBQ_REP
#define NBQ_REP 0
#endif
template <int T, bool REP = false>
__device__ __forceinline__ void nbq_run(const u8* b, const __half* dh, int sub, int v[8], float& s0, float& s1, float& mn,
                                        const typename Tab<T>::E* tab) {
    if (REP && T == T_IQ3XXS) tab += threadIdx.x & 31;
    if (T == T_IQ3XXS) {
        const float d = __half2float(*dh);
        const uint2 qs = *reinterpret_cast<const uint2*>(b + 8 * sub);
        const u32 aux = nbq_ld32(b + 64 + 4 * sub);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            u32 s7 = (aux >> (7 * l)) & 127u;
            u32 s8 = s7 | ((__popc(s7) & 1u) << 7);
            u32 w2 = l < 2 ? qs.x : qs.y;
            v[2 * l] = nbq_sign4(tab[((w2 >> (16 * (l & 1))) & 255u) << (REP ? 5 : 0)], s8);
            v[2 * l + 1] = nbq_sign4(tab[((w2 >> (16 * (l & 1) + 8)) & 255u) << (REP ? 5 : 0)], s8 >> 4);
        }
        s0 = s1 = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        mn = 0.0f;
    } else if (T == T_IQ3S) {
        const float d = __half2float(*dh);
        const u32 qa = nbq_ld32(b + 8 * sub), qb = nbq_ld32(b + 8 * sub + 4);
        const u32 qh = b[64 + sub];
        const u32 sg = nbq_ld32(b + 72 + 4 * sub);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            u32 s8 = (sg >> (8 * l)) & 255u;
            u32 w2 = l < 2 ? qa : qb;
            v[2 * l] = nbq_sign4(tab[((w2 >> (16 * (l & 1))) & 255u) | ((qh << (8 - 2 * l)) & 256u)], s8);
            v[2 * l + 1] = nbq_sign4(tab[((w2 >> (16 * (l & 1) + 8)) & 255u) | ((qh << (7 - 2 * l)) & 256u)], s8 >> 4);
        }
        s0 = s1 = d * (float)(1 + 2 * ((b[104 + (sub >> 1)] >> (4 * (sub & 1))) & 15));
        mn = 0.0f;
    } else if (T == T_IQ2S) {
        const float d = __half2float(*dh);
        const u32 qs = nbq_ld32(b + 4 * sub);
        const u32 sg = nbq_ld32(b + 32 + 4 * sub);
        const u32 qh = b[64 + sub];
        const u32 sc = b[72 + sub];
#pragma unroll
        for (int l = 0; l < 4; l++) {
            unsigned long long g = tab[((qs >> (8 * l)) & 255u) | ((qh << (8 - 2 * l)) & 0x300u)];
            u32 s8 = (sg >> (8 * l)) & 255u;
            v[2 * l] = nbq_sign4((u32)g, s8);
            v[2 * l + 1] = nbq_sign4((u32)(g >> 32), s8 >> 4);
        }
        s0 = d * (0.5f + (float)(sc & 15)) * 0.25f;
        s1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        mn = 0.0f;
    } else {                                    /* Q4_K, Q5_K */
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dm = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, m;
        kq_scale_min(b + 4, sub, sc, m);
        const int sh = 4 * (sub & 1);
        const u8* ql = b + (T == T_Q4K ? 16 : 48) + 32 * (sub >> 1);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            u32 q = (nbq_ld32(ql + 4 * k) >> sh) & 0x0f0f0f0fu;
            if (T == T_Q5K) q |= ((nbq_ld32(b + 16 + 4 * k) >> sub) & 0x01010101u) << 4;
            v[k] = (int)q;
        }
        s0 = s1 = d * (float)sc;
        mn = dm * (float)m;
    }
}

/* ---------------------------------------------------------------- decode: matrix x vector
   y[r] = sum over runs of the int32 dot of the run times the weight scale times the input scale.
   One warp per row (rows strided over the grid); lane l takes runs l, l + 32, ... */
template <int T, bool REP = false>
__device__ __forceinline__ float nbq_row_dot(const u8* w, const __half* wd, int r, int nb,
                                             const int* xq, const float* xs, int lane, const typename Tab<T>::E* tab) {
    const u8* row = w + (size_t)r * nb * BT<T>::B;
    const __half* rd = wd + (size_t)r * nb;
    float acc = 0.0f;
#pragma unroll 2
    for (int g = lane; g < nb * 8; g += 32) {
        int blk = g >> 3, sub = g & 7;
        int v[8]; float s0, s1, mn;
        nbq_run<T, REP>(row + (size_t)blk * BT<T>::B, rd + blk, sub, v, s0, s1, mn, tab);
        const int4* xp = reinterpret_cast<const int4*>(xq + 8 * g);
        int4 a = NBQ_LDG(xp), c = NBQ_LDG(xp + 1);
        NBQ_XF(a, c, xf);
        NBQ_DOT(v, a, c, xf, lo, hi);
        float f = nbq_runf(s0, lo, s1, hi);
        if (T == T_Q4K || T == T_Q5K) f = __fmaf_rn(-mn, (float)NBQ_SX(a, c, xf), f);
        acc = __fmaf_rn(NBQ_LDG(xs + g), f, acc);
    }
    return warp_sum(acc);
}

/* y = W x (+ add, when add != 0) */
template <int T>
__device__ __forceinline__ void nbq_mv(const u8* w, const __half* wd, const int* xq, const float* xs, float* y,
                                       const float* add, int rows, int cols) {
    NBQ_TAB(T, tab);
    const int lane = threadIdx.x & 31;
    for (int r = blockIdx.x * 8 + (threadIdx.x >> 5); r < rows; r += gridDim.x * 8) {
        float s = nbq_row_dot<T>(w, wd, r, cols >> 8, xq, xs, lane, tab);
        if (lane == 0) y[r] = add ? add[r] + s : s;
    }
}
#define NBQ_MV(T, NAME) extern "C" __global__ void __launch_bounds__(256) NAME(const u8* w, const __half* wd, \
    const int* xq, const float* xs, float* y, const float* add, int rows, int cols) { nbq_mv<T>(w, wd, xq, xs, y, add, rows, cols); }
NBQ_MV(T_Q4K, nbq_mv_q4k)
NBQ_MV(T_Q5K, nbq_mv_q5k)
#if NBQ_REP
extern "C" __global__ void __launch_bounds__(256) nbq_mv_iq3xxs(const u8* w, const __half* wd, const int* xq, const float* xs,
        float* y, const float* add, int rows, int cols) {
    __shared__ u32 tab[256 * 32];
    for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) tab[i] = nbq_iq3xxs_grid[i >> 5];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    for (int r = blockIdx.x * 8 + (threadIdx.x >> 5); r < rows; r += gridDim.x * 8) {
        float s = nbq_row_dot<T_IQ3XXS, true>(w, wd, r, cols >> 8, xq, xs, lane, tab);
        if (lane == 0) y[r] = add ? add[r] + s : s;
    }
}
#else
NBQ_MV(T_IQ3XXS, nbq_mv_iq3xxs)
#endif
NBQ_MV(T_IQ3S, nbq_mv_iq3s)
NBQ_MV(T_IQ2S, nbq_mv_iq2s)

/* gate and up in one launch: rows [0, rows) of y from wg, [rows, 2 rows) from wu */
extern "C" __global__ void __launch_bounds__(256) nbq_mv2_iq3xxs(const u8* wg, const __half* dg, const u8* wu,
        const __half* du, const int* xq, const float* xs, float* y, int rows, int cols) {
#if NBQ_REP
    __shared__ u32 tab[256 * 32];
    for (int i = threadIdx.x; i < 256 * 32; i += blockDim.x) tab[i] = nbq_iq3xxs_grid[i >> 5];
    __syncthreads();
#else
    NBQ_TAB(T_IQ3XXS, tab);
#endif
    const int lane = threadIdx.x & 31;
    for (int r = blockIdx.x * 8 + (threadIdx.x >> 5); r < 2 * rows; r += gridDim.x * 8) {
        float s = r < rows ? nbq_row_dot<T_IQ3XXS, NBQ_REP>(wg, dg, r, cols >> 8, xq, xs, lane, tab)
                           : nbq_row_dot<T_IQ3XXS, NBQ_REP>(wu, du, r - rows, cols >> 8, xq, xs, lane, tab);
        if (lane == 0) y[r] = s;
    }
}

/* two adjacent rows in one warp (they share the input loads); for the fused Q K V launch below */
template <int T>
__device__ __forceinline__ void nbq_row_dot2(const u8* w, const __half* wd, int r, int nb, const int* xq,
        const float* xs, int lane, const typename Tab<T>::E* tab, float& o0, float& o1) {
    const u8* row0 = w + (size_t)r * nb * BT<T>::B;
    const u8* row1 = row0 + (size_t)nb * BT<T>::B;
    const __half* rd0 = wd + (size_t)r * nb;
    const __half* rd1 = rd0 + nb;
    float a0 = 0.0f, a1 = 0.0f;
#pragma unroll 2
    for (int g = lane; g < nb * 8; g += 32) {
        int blk = g >> 3, sub = g & 7;
        int v[8], u[8]; float s0, s1, mn, t0, t1, tn;
        nbq_run<T>(row0 + (size_t)blk * BT<T>::B, rd0 + blk, sub, v, s0, s1, mn, tab);
        nbq_run<T>(row1 + (size_t)blk * BT<T>::B, rd1 + blk, sub, u, t0, t1, tn, tab);
        const int4* xp = reinterpret_cast<const int4*>(xq + 8 * g);
        int4 a = NBQ_LDG(xp), c = NBQ_LDG(xp + 1);
        NBQ_XF(a, c, xf);
        NBQ_DOT(v, a, c, xf, lo, hi);
        NBQ_DOT(u, a, c, xf, lo2, hi2);
        float f0 = nbq_runf(s0, lo, s1, hi), f1 = nbq_runf(t0, lo2, t1, hi2);
        if (T == T_Q4K || T == T_Q5K) {
            const float sx = (float)NBQ_SX(a, c, xf);
            f0 = __fmaf_rn(-mn, sx, f0); f1 = __fmaf_rn(-tn, sx, f1);
        }
        float xsg = NBQ_LDG(xs + g);
        a0 = __fmaf_rn(xsg, f0, a0); a1 = __fmaf_rn(xsg, f1, a1);
    }
    o0 = warp_sum(a0); o1 = warp_sum(a1);
}

__device__ __forceinline__ size_t nbq_kofs(int hk, int ctx, int p, int piece);

/* Q, K and V of one token in one launch (Q, K: IQ2_S; V: Q4_K), with RoPE and the K / V cache store: a warp takes
   a pair of rows (2 i, 2 i + 1), which is one RoPE pair. q gets the rotated Q (fp32). */
extern "C" __global__ void __launch_bounds__(256) nbq_qkv_rope(const u8* wq, const __half* dq, const u8* wk,
        const __half* dk, const u8* wv, const int* xq, const float* xs, int nh, int nkv, int cols, const int* pos,
        float theta_scale, float* q, __half* kc, __half* vc, int ctx) {
    NBQ_TAB(T_IQ2S, tab);
    const int lane = threadIdx.x & 31, nb = cols >> 8;
    const int pq = nh * 64, pk = nkv * 64, np = pq + 2 * pk;
    const int p = pos[0];
    for (int pr = blockIdx.x * 8 + (threadIdx.x >> 5); pr < np; pr += gridDim.x * 8) {
        float a, b;
        if (pr < pq + pk) {
            int r = 2 * (pr < pq ? pr : pr - pq);
            if (pr < pq) nbq_row_dot2<T_IQ2S>(wq, dq, r, nb, xq, xs, lane, tab, a, b);
            else nbq_row_dot2<T_IQ2S>(wk, dk, r, nb, xq, xs, lane, tab, a, b);
            if (lane == 0) {
                const int i = (r & 127) >> 1;
                const float theta = (float)p * powf(theta_scale, (float)i);
                const float c = cosf(theta), s = sinf(theta);
                float x0, x1;
                nbq_rot(a, b, c, s, x0, x1);
                if (pr < pq) { q[r] = x0; q[r + 1] = x1; }
                else {
                    const int hk = r >> 7;
                    size_t ko = nbq_kofs(hk, ctx, p, i >> 2) + 2 * (i & 3);
                    kc[ko] = __float2half(x0);
                    kc[ko + 1] = __float2half(x1);
                }
            }
        } else {
            int r = 2 * (pr - pq - pk);
            nbq_row_dot2<T_Q4K>(wv, reinterpret_cast<const __half*>(wv), r, nb, xq, xs, lane, nullptr, a, b);
            if (lane == 0) {
                const int hk = r >> 7, d = r & 127;
                size_t o = ((size_t)hk * ctx + p) * 128 + d;
                vc[o] = __float2half(a);
                vc[o + 1] = __float2half(b);
            }
        }
    }
}

/* ---------------------------------------------------------------- repack: I-quant block -> fp16 scale + body */
extern "C" __global__ void nbq_repack(const u8* src, u8* body, __half* d, long long nblk, int bsrc) {
    long long i = (long long)blockIdx.x * blockDim.y + threadIdx.y;
    if (i >= nblk) return;
    const u8* s = src + i * bsrc;
    u8* o = body + i * (bsrc - 2);
    for (int k = threadIdx.x; k < bsrc - 2; k += blockDim.x) o[k] = s[2 + k];
    if (threadIdx.x == 0) d[i] = __ushort_as_half((unsigned short)(s[0] | (s[1] << 8)));
}

/* ---------------------------------------------------------------- quantize, norms */
/* one warp quantizes one run of 32: v is the value of this lane */
__device__ __forceinline__ void nbq_q8_run(float v, int g, int lane, signed char* xq, float* xs) {
    float amax = warp_max(fabsf(v));
    float d = amax / 127.0f;
    signed char q = amax == 0.0f ? 0 : (signed char)roundf(v / d);
    xq[32 * g + lane] = q;
    if (lane == 0) xs[g] = d;
}

/* one block of 1024 threads: x (+ add when add != 0, written back to x) -> RMS norm * w -> int8 runs */
extern "C" __global__ void __launch_bounds__(1024) nbq_add_rms_q8(float* x, const float* add, const float* w, int n,
        float eps, signed char* xq, float* xs) {
    __shared__ float red[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float ss = 0.0f;
    for (int i = tid; i < n; i += 1024) {
        float v = x[i];
        if (add) { v += add[i]; x[i] = v; }
        ss = __fmaf_rn(v, v, ss);
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) { ss = warp_sum(red[lane]); if (lane == 0) red[0] = ss; }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float)n + eps);
    for (int g = warp; g < n / 32; g += 32) {
        int i = 32 * g + lane;
        nbq_q8_run(__fmul_rn(__fmul_rn(scale, x[i]), w[i]), g, lane, xq, xs);
    }
}

/* the final norm: y = RMS norm(x + add) * w (fp32 out) */
extern "C" __global__ void __launch_bounds__(1024) nbq_add_rms(const float* x, const float* add, const float* w, int n,
        float eps, float* y, signed char* xq, float* xs) {
    __shared__ float red[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float ss = 0.0f;
    for (int i = tid; i < n; i += 1024) { float v = x[i] + (add ? add[i] : 0.0f); ss = __fmaf_rn(v, v, ss); }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (warp == 0) { ss = warp_sum(red[lane]); if (lane == 0) red[0] = ss; }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float)n + eps);
    for (int g = warp; g < n / 32; g += 32) {
        int i = 32 * g + lane;
        float v = scale * (x[i] + (add ? add[i] : 0.0f)) * w[i];
        if (y) y[i] = v;
        if (xq) nbq_q8_run(v, g, lane, xq, xs);
    }
}

/* attention output (fp32) -> int8 runs; SwiGLU of [gate | up] -> int8 runs */
extern "C" __global__ void nbq_q8(const float* x, int n, signed char* xq, float* xs) {
    int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= n / 32) return;
    nbq_q8_run(x[32 * g + lane], g, lane, xq, xs);
}
extern "C" __global__ void nbq_swiglu_q8(const float* gu, int n, signed char* xq, float* xs) {
    int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (g >= n / 32) return;
    int i = 32 * g + lane;
    float a = gu[i], b = gu[n + i];
    nbq_q8_run(a / (1.0f + expf(-a)) * b, g, lane, xq, xs);
}

/* ---------------------------------------------------------------- embedding (IQ3_S row -> fp32) */
/* one embedding row (token tok[blockIdx.y]) -> fp32 row blockIdx.y of y; grid (runs / 8, tokens) */
template <int T>
__device__ __forceinline__ void nbq_embed_t(const u8* w, const __half* wd, const int* tok, int cols, float* y) {
    int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    int nb = cols >> 8;
    if (g >= nb * 8) return;
    size_t r = (size_t)tok[t];
    int blk = g >> 3, sub = g & 7;
    int v[8]; float s0, s1, mn;
    const typename Tab<T>::E* tab = (T == T_IQ3S) ? (const typename Tab<T>::E*)nbq_iq3s_grid : nullptr;
    nbq_run<T>(w + (r * nb + blk) * BT<T>::B, wd + r * nb + blk, sub, v, s0, s1, mn, tab);
    int q = (signed char)(v[lane >> 2] >> (8 * (lane & 3)));
    if (T == T_Q4K || T == T_Q5K) q &= 0xff;
    y[(size_t)t * cols + 32 * g + lane] = (lane < 16 ? s0 : s1) * (float)q - mn;
}
extern "C" __global__ void nbq_embed_t21(const u8* w, const __half* wd, const int* tok, int cols, float* y) { nbq_embed_t<T_IQ3S>(w, wd, tok, cols, y); }
extern "C" __global__ void nbq_embed_t12(const u8* w, const __half* wd, const int* tok, int cols, float* y) { nbq_embed_t<T_Q4K>(w, wd, tok, cols, y); }
extern "C" __global__ void nbq_embed_t13(const u8* w, const __half* wd, const int* tok, int cols, float* y) { nbq_embed_t<T_Q5K>(w, wd, tok, cols, y); }

/* The K cache is kept in tiles of 32 positions: [nkv][ctx / 32][16 pieces][32 positions][8 halves], so that 32 lanes
   that take 32 positions read 512 contiguous bytes per piece. V stays [nkv][ctx][128]. */
__device__ __forceinline__ size_t nbq_kofs(int hk, int ctx, int p, int piece) {
    return ((((size_t)hk * (ctx >> 5) + (p >> 5)) * 16 + piece) * 32 + (p & 31)) * 8;
}

/* ---------------------------------------------------------------- RoPE (adjacent pairs) + K/V store
   q: [nh][128], k, v: [nkv][128] fp32. K and V go to the fp16 cache [nkv][ctx][128] at position pos[0]. */
extern "C" __global__ void nbq_rope_store(float* q, const float* k, const float* v, const int* pos, int nh, int nkv,
        float theta_scale, __half* kc, __half* vc, int ctx) {
    int h = blockIdx.x, i = threadIdx.x;          /* 64 threads: pair i */
    const int p = pos[0];
    const float theta = (float)p * powf(theta_scale, (float)i);
    const float c = cosf(theta), s = sinf(theta);
    if (h < nh) {
        float* x = q + h * 128;
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        nbq_rot(x0, x1, c, s, x[2 * i], x[2 * i + 1]);
    } else {
        int hk = h - nh;
        const float* x = k + hk * 128;
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        size_t o = ((size_t)hk * ctx + p) * 128;
        size_t ko = nbq_kofs(hk, ctx, p, i >> 2) + 2 * (i & 3);
        float y0, y1;
        nbq_rot(x0, x1, c, s, y0, y1);
        kc[ko] = __float2half(y0);
        kc[ko + 1] = __float2half(y1);
        vc[o + 2 * i] = __float2half(v[hk * 128 + 2 * i]);
        vc[o + 2 * i + 1] = __float2half(v[hk * 128 + 2 * i + 1]);
    }
}

/* ---------------------------------------------------------------- decode attention (split over positions)
   grid (nkv, NSPLIT), 256 threads. Each block takes the 7 (= nh / nkv) query heads of one K/V head and the
   positions of its split; it writes per head the running max, the sum and the unnormalized output. */
#define NBQ_G 7
#ifndef NBQ_ATT_MINB
#define NBQ_ATT_MINB 2
#endif
/* grid (nkv, NSPLIT), 256 threads. A block takes the 7 query heads of one K/V head and a run of positions; warp w
   takes the chunks of 32 positions w, w + 8, ... of the run (lane = position for the scores, lane = 4 dims for P V)
   with its own running max and sum. The 8 warps are merged at the end. */
template <int G>
__device__ __forceinline__ void nbq_attn_part_t(const float* q, const __half* kc, const __half* vc,
        const int* pos, int ctx, float scale, float* part_o, float* part_ms, int qstride) {
    const int hk = blockIdx.x, sp = blockIdx.y, nsp = gridDim.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = pos[blockIdx.z] + 1;
    q += (size_t)blockIdx.z * qstride;
    part_o += (size_t)blockIdx.z * gridDim.x * G * nsp * 128;
    part_ms += (size_t)blockIdx.z * gridDim.x * G * nsp * 2;
    const int per = max(256, ((n + nsp - 1) / nsp + 255) & ~255);
    const int p0 = sp * per, p1 = min(n, p0 + per);
    __shared__ __align__(16) float qs[G][128];
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
    for (int i = tid; i < G * 128; i += 256) reinterpret_cast<float*>(qs)[i] = q[hk * G * 128 + i] * scale;
    __syncthreads();
    float m[G], l[G], o[G][4];
#pragma unroll
    for (int j = 0; j < G; j++) { m[j] = -1e30f; l[j] = 0.0f; o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
    const __half* vb = vc + (size_t)hk * ctx * 128 + 4 * lane;
    for (int c0 = p0 + 32 * warp; c0 < p1; c0 += 256) {
        const int p = c0 + lane;
        const bool live = p < p1;
        float s[G];
#pragma unroll
        for (int j = 0; j < G; j++) s[j] = 0.0f;
        const uint4* kp = reinterpret_cast<const uint4*>(kc + nbq_kofs(hk, ctx, c0, 0)) + lane;
        uint2 vr[32];                                   /* the V rows of the chunk: loaded with K, used after the softmax */
#pragma unroll
        for (int t = 0; t < 32; t++) vr[t] = *reinterpret_cast<const uint2*>(vb + (size_t)(c0 + t) * 128);
#pragma unroll
        for (int u0 = 0; u0 < 16; u0 += 8) {
            uint4 kr[8];
#pragma unroll
            for (int u = 0; u < 8; u++) kr[u] = kp[(u0 + u) * 32];
#pragma unroll
            for (int u = 0; u < 8; u++) {
                const __half2* h = reinterpret_cast<const __half2*>(&kr[u]);
                float2 k0 = __half22float2(h[0]), k1 = __half22float2(h[1]), k2 = __half22float2(h[2]), k3 = __half22float2(h[3]);
                const int c = 8 * (u0 + u);
#pragma unroll
                for (int j = 0; j < G; j++) {
                    float4 a = *reinterpret_cast<const float4*>(&qs[j][c]), b = *reinterpret_cast<const float4*>(&qs[j][c + 4]);
                    s[j] += a.x * k0.x + a.y * k0.y + a.z * k1.x + a.w * k1.y + b.x * k2.x + b.y * k2.y + b.z * k3.x + b.w * k3.y;
                }
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
            uint2 r = vr[t];
            float2 v0 = __half22float2(*reinterpret_cast<const __half2*>(&r.x)), v1 = __half22float2(*reinterpret_cast<const __half2*>(&r.y));
#pragma unroll
            for (int j = 0; j < G; j++) {
                float pr = __shfl_sync(0xffffffffu, s[j], t);
                o[j][0] += pr * v0.x; o[j][1] += pr * v0.y; o[j][2] += pr * v1.x; o[j][3] += pr * v1.y;
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
#define NBQ_ATTN(G, NAME) extern "C" __global__ void __launch_bounds__(256, NBQ_ATT_MINB) NAME(const float* q, const __half* kc, \
    const __half* vc, const int* pos, int ctx, float scale, float* part_o, float* part_ms, int qstride) { \
    nbq_attn_part_t<G>(q, kc, vc, pos, ctx, scale, part_o, part_ms, qstride); }
NBQ_ATTN(7, nbq_attn_part)
NBQ_ATTN(5, nbq_attn_part_g5)
/* merge the splits: grid nh, 128 threads -> out [nh][128] fp32 */
extern "C" __global__ void nbq_attn_merge(const float* part_o, const float* part_ms, int nsp, float* out) {
    int h = blockIdx.x, d = threadIdx.x;
    part_o += (size_t)blockIdx.y * gridDim.x * nsp * 128;
    part_ms += (size_t)blockIdx.y * gridDim.x * nsp * 2;
    out += (size_t)blockIdx.y * gridDim.x * 128;
    float mx = -1e30f;
    for (int s = 0; s < nsp; s++) mx = fmaxf(mx, part_ms[(h * nsp + s) * 2]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < nsp; s++) {
        float f = expf(part_ms[(h * nsp + s) * 2] - mx);
        l = __fmaf_rn(f, part_ms[(h * nsp + s) * 2 + 1], l);
        o = __fmaf_rn(f, part_o[((size_t)h * nsp + s) * 128 + d], o);
    }
    out[h * 128 + d] = o / l;
}
/* the same merge, written as int8 runs (the input of the O projection); 128 threads = 4 runs per head */
extern "C" __global__ void nbq_attn_merge_q8(const float* part_o, const float* part_ms, int nsp, signed char* xq, float* xs) {
    int h = blockIdx.x, d = threadIdx.x;
    float mx = -1e30f;
    for (int s = 0; s < nsp; s++) mx = fmaxf(mx, part_ms[(h * nsp + s) * 2]);
    float l = 0.0f, o = 0.0f;
    for (int s = 0; s < nsp; s++) {
        float f = expf(part_ms[(h * nsp + s) * 2] - mx);
        l = __fmaf_rn(f, part_ms[(h * nsp + s) * 2 + 1], l);
        o = __fmaf_rn(f, part_o[((size_t)h * nsp + s) * 128 + d], o);
    }
    nbq_q8_run(o / l, h * 4 + (d >> 5), d & 31, xq, xs);
}

extern "C" __global__ void nbq_add(float* x, const float* add, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += add[i];
}

/* ---------------------------------------------------------------- argmax over the logits (two stages) */
extern "C" __global__ void nbq_argmax_part(const float* x, int n, float* pv, int* pi) {
    __shared__ float bv[32]; __shared__ int bi[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float v = -3.4e38f; int idx = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffffu, v, o); int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
        if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    if (lane == 0) { bv[warp] = v; bi[warp] = idx; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) if (bv[w] > v || (bv[w] == v && bi[w] < idx)) { v = bv[w]; idx = bi[w]; }
        pv[blockIdx.x] = v; pi[blockIdx.x] = idx;
    }
}
extern "C" __global__ void nbq_argmax_final(const float* pv, const int* pi, int n, int* out, int* pos) {
    if (threadIdx.x) return;
    float v = pv[0]; int idx = pi[0];
    for (int i = 1; i < n; i++) if (pv[i] > v || (pv[i] == v && pi[i] < idx)) { v = pv[i]; idx = pi[i]; }
    out[0] = idx;
    if (pos) pos[0] += 1;
}
