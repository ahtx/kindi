/* nbq_mm.cu - the prompt kernels of nbq.cu (appended to it at compile time).

   Prompt GEMM on int8 tensor cores: Y[t][yo + r] = sum over runs k of ws[r][k] * xs[t][k] * (int32 dot of run k),
   for the rows r of W and the tokens t of X. Block: 256 threads, 128 weight rows x 128 tokens; K in steps of 128
   columns (half a quant block). The weights of a step are decoded once into shared memory as int8 with their run
   scales; then 8 warps (2 x 4) each take 64 rows x 32 tokens: 4 x 4 m16n8k32 tiles. The MMA accumulator starts at
   the integer 0x4B400000, so its bits read as a float are exactly 12582912 + dot (|dot| < 2^22): one FADD gives the
   dot in fp32 without an int-to-float conversion. IQ2_S has a scale per 16 weights: its tiles take two m16n8k16
   steps. Q4_K and Q5_K subtract min * xs * sum(x).

   NBQ_SIMT (Volta, sm_70: no int8 tensor cores): the same kernel, with each tile's four int32 dot products taken by
   dp4a from the same shared-memory rows (nbq_sdot) instead of the MMA. The dots are the same integers, added to the
   same magic start, so the output is bit for bit the output of the tensor-core kernel. */
#define NBQ_MAGIC 0x4B400000
/* c[0..3] += the int32 dots of bytes 4*k0 .. 4*k1-1: (row a, token a), (row a, token b), (row b, token a),
   (row b, token b); the order of nbq_gemm's accumulator (the m16n8 tile's c0..c3) */
__device__ __forceinline__ void nbq_sdot(int* c, const int* ra, const int* rb, const int* ta, const int* tb, int k0, int k1) {
#pragma unroll
    for (int k = k0; k < k1; k++) {
        const int a0 = ra[k], a1 = rb[k], b0 = ta[k], b1 = tb[k];
        c[0] = __dp4a(a0, b0, c[0]); c[1] = __dp4a(a0, b1, c[1]);
        c[2] = __dp4a(a1, b0, c[2]); c[3] = __dp4a(a1, b1, c[3]);
    }
}
#define NBQ_MMA32(c, a0, a1, a2, a3, b0, b1) \
  NBQ_ASM("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define NBQ_MMA16(c, a0, a1, b0) \
  NBQ_ASM("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0))
#define GBK 128
#define GLD (GBK + 16)            /* shared row stride in bytes: the fragment loads hit 32 banks */
#define NBQ_GEMM_SMEM (2 * 128 * GLD + 128 * 8 * 4 + 3 * 128 * 4 * 4)

__device__ __forceinline__ float nbq_dotf(int c) { return __int_as_float(c) - 12582912.0f; }

template <int T>
__device__ __forceinline__ void nbq_gemm(const u8* w, const __half* wd, const signed char* xq, const float* xs,
                                         float* y, int rows, int cols, int nt, int ldy, int yo, int rblk) {
    const bool HALF = (T == T_IQ2S), MINS = (T == T_Q4K || T == T_Q5K);
    extern __shared__ __align__(16) unsigned char gsm[];
    u8* As = gsm;                                            /* [128][GLD] */
    u8* Bs = As + 128 * GLD;                                 /* [128][GLD] */
    float* Asc = reinterpret_cast<float*>(Bs + 128 * GLD);   /* [128][8]: per 16 (IQ2_S) or per 32 (4 used) */
    float* Amn = Asc + 128 * 8;                              /* [128][4] */
    float* Bsc = Amn + 128 * 4;                              /* [128][4] */
    float* Bsm = Bsc + 128 * 4;                              /* [128][4]: xs * sum(x), for the mins */
    NBQ_TAB(T, tab);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, t4 = lane & 3;
    const int row0 = rblk * 128, tok0 = blockIdx.y * 128;
    const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
    const int nb = cols >> 8;
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    for (int k0 = 0; k0 < cols; k0 += GBK) {
        __syncthreads();
        /* decode the weights: 128 rows x 4 runs, two per thread */
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int idx = tid + 256 * h, r = idx >> 2, u = idx & 3;
            int row = min(row0 + r, rows - 1);
            int g = (k0 >> 5) + u, blk = g >> 3, sub = g & 7;
            int v[8]; float s0, s1, mn;
            nbq_run<T>(w + ((size_t)row * nb + blk) * BT<T>::B, wd + (size_t)row * nb + blk, sub, v, s0, s1, mn, tab);
            uint4* dst = reinterpret_cast<uint4*>(As + r * GLD + 32 * u);
            dst[0] = make_uint4(v[0], v[1], v[2], v[3]);
            dst[1] = make_uint4(v[4], v[5], v[6], v[7]);
            if (HALF) { Asc[r * 8 + 2 * u] = s0; Asc[r * 8 + 2 * u + 1] = s1; }
            else Asc[r * 8 + u] = s0;
            if (MINS) Amn[r * 4 + u] = mn;
        }
        /* the inputs: 128 tokens x 128 columns, four 16-byte pieces per thread */
#pragma unroll
        for (int h = 0; h < 4; h++) {
            int idx = tid + 256 * h, t = idx >> 3, pc = idx & 7, tok = tok0 + t;
            uint4 val = tok < nt ? *reinterpret_cast<const uint4*>(xq + (size_t)tok * cols + k0 + 16 * pc) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(Bs + t * GLD + 16 * pc) = val;
        }
        for (int idx = tid; idx < 512; idx += 256) {
            int t = idx >> 2, u = idx & 3, tok = tok0 + t;
            Bsc[t * 4 + u] = tok < nt ? xs[(size_t)tok * (cols >> 5) + (k0 >> 5) + u] : 0.0f;
        }
        __syncthreads();
        if (MINS) {
            for (int idx = tid; idx < 512; idx += 256) {
                int t = idx >> 2, u = idx & 3;
                const int* p = reinterpret_cast<const int*>(Bs + t * GLD + 32 * u);
                int sx = 0;
#pragma unroll
                for (int e = 0; e < 8; e++) sx = __dp4a(p[e], 0x01010101, sx);
                Bsm[t * 4 + u] = Bsc[t * 4 + u] * (float)sx;
            }
            __syncthreads();
        }
#pragma unroll 1
        for (int u = 0; u < 4; u++) {                        /* the four runs of 32 of this step */
#if NBQ_SIMT
            const int* sa = reinterpret_cast<const int*>(As + 32 * u);       /* the run's bytes of row r: sa + r * GLD / 4 */
            const int* sb = reinterpret_cast<const int*>(Bs + 32 * u);
#define NBQ_ROW(p, r) ((p) + (r) * (GLD / 4))
#else
            unsigned int a[4][4], b[4][2];
            {
                const int l8 = lane & 7, lq = lane >> 3;
#pragma unroll
                for (int i = 0; i < 4; i++) {                /* A: rows +0..7 / +8..15 (lq & 1), bytes +0 / +16 (lq >> 1) */
                    u32 ad = (u32)__cvta_generic_to_shared(As + (wm + 16 * i + l8 + 8 * (lq & 1)) * GLD + 32 * u + 16 * (lq >> 1));
                    NBQ_ASM("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(a[i][0]), "=r"(a[i][1]), "=r"(a[i][2]), "=r"(a[i][3]) : "r"(ad));
                }
#pragma unroll
                for (int j = 0; j < 4; j += 2) {             /* B: tokens +0..7 / +8..15 (lq >> 1), bytes +0 / +16 (lq & 1) */
                    u32 ad = (u32)__cvta_generic_to_shared(Bs + (wn + 8 * j + l8 + 8 * (lq >> 1)) * GLD + 32 * u + 16 * (lq & 1));
                    NBQ_ASM("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(b[j][0]), "=r"(b[j][1]), "=r"(b[j + 1][0]), "=r"(b[j + 1][1]) : "r"(ad));
                }
            }
#endif
            float xa[4], xb[4], ma[4], mb[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                int ta = wn + 8 * j + 2 * t4;
                xa[j] = Bsc[ta * 4 + u]; xb[j] = Bsc[(ta + 1) * 4 + u];
                if (MINS) { ma[j] = Bsm[ta * 4 + u]; mb[j] = Bsm[(ta + 1) * 4 + u]; }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int ra = wm + 16 * i + gid, rb = ra + 8;
                if (HALF) {
                    float wa0 = Asc[ra * 8 + 2 * u], wa1 = Asc[ra * 8 + 2 * u + 1];
                    float wb0 = Asc[rb * 8 + 2 * u], wb1 = Asc[rb * 8 + 2 * u + 1];
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int c0[4] = {NBQ_MAGIC, NBQ_MAGIC, NBQ_MAGIC, NBQ_MAGIC}, c1[4] = {NBQ_MAGIC, NBQ_MAGIC, NBQ_MAGIC, NBQ_MAGIC};
#if NBQ_SIMT
                        const int ta = wn + 8 * j + 2 * t4;
                        nbq_sdot(c0, NBQ_ROW(sa, ra), NBQ_ROW(sa, rb), NBQ_ROW(sb, ta), NBQ_ROW(sb, ta + 1), 0, 4);
                        nbq_sdot(c1, NBQ_ROW(sa, ra), NBQ_ROW(sa, rb), NBQ_ROW(sb, ta), NBQ_ROW(sb, ta + 1), 4, 8);
#else
                        NBQ_MMA16(c0, a[i][0], a[i][1], b[j][0]);
                        NBQ_MMA16(c1, a[i][2], a[i][3], b[j][1]);
#endif
                        acc[i][j][0] += (nbq_dotf(c0[0]) * wa0 + nbq_dotf(c1[0]) * wa1) * xa[j];
                        acc[i][j][1] += (nbq_dotf(c0[1]) * wa0 + nbq_dotf(c1[1]) * wa1) * xb[j];
                        acc[i][j][2] += (nbq_dotf(c0[2]) * wb0 + nbq_dotf(c1[2]) * wb1) * xa[j];
                        acc[i][j][3] += (nbq_dotf(c0[3]) * wb0 + nbq_dotf(c1[3]) * wb1) * xb[j];
                    }
                } else {
                    float wa = Asc[ra * 8 + u], wb = Asc[rb * 8 + u];
                    float na = MINS ? Amn[ra * 4 + u] : 0.0f, nbm = MINS ? Amn[rb * 4 + u] : 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int c[4] = {NBQ_MAGIC, NBQ_MAGIC, NBQ_MAGIC, NBQ_MAGIC};
#if NBQ_SIMT
                        const int ta = wn + 8 * j + 2 * t4;
                        nbq_sdot(c, NBQ_ROW(sa, ra), NBQ_ROW(sa, rb), NBQ_ROW(sb, ta), NBQ_ROW(sb, ta + 1), 0, 8);
#else
                        NBQ_MMA32(c, a[i][0], a[i][1], a[i][2], a[i][3], b[j][0], b[j][1]);
#endif
                        acc[i][j][0] += nbq_dotf(c[0]) * wa * xa[j];
                        acc[i][j][1] += nbq_dotf(c[1]) * wa * xb[j];
                        acc[i][j][2] += nbq_dotf(c[2]) * wb * xa[j];
                        acc[i][j][3] += nbq_dotf(c[3]) * wb * xb[j];
                        if (MINS) {
                            acc[i][j][0] -= na * ma[j]; acc[i][j][1] -= na * mb[j];
                            acc[i][j][2] -= nbm * ma[j]; acc[i][j][3] -= nbm * mb[j];
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int ra = row0 + wm + 16 * i + gid, rb = ra + 8, ta = tok0 + wn + 8 * j + 2 * t4;
            if (ra < rows && ta < nt) y[(size_t)ta * ldy + yo + ra] = acc[i][j][0];
            if (ra < rows && ta + 1 < nt) y[(size_t)(ta + 1) * ldy + yo + ra] = acc[i][j][1];
            if (rb < rows && ta < nt) y[(size_t)ta * ldy + yo + rb] = acc[i][j][2];
            if (rb < rows && ta + 1 < nt) y[(size_t)(ta + 1) * ldy + yo + rb] = acc[i][j][3];
        }
}
#if NBQ_HMMA
/* NBQ_HMMA (Volta, sm_70: fp16 tensor cores, no int8 MMA): the same GEMM as nbq_gemm, bit for bit. The decoded
   weights and the int8 inputs go to shared memory as fp16, which holds them exactly. Each run of 32 columns (or
   each half run, IQ2_S) is its own sum of m8n8k4 products (mma.sync, sm_70) from a zero accumulator: the products
   of integers are exact, and every partial sum is an integer below 2^24, so the fp32 result is the int32 dot. Each
   run's dots are scaled in fp32 in the order of nbq_gemm, from the accumulator layout that PTX documents for
   m8n8k4. Block: 128 rows x 128 tokens as nbq_gemm; warp: 64 rows x 32 tokens = 32 tiles of 8 x 8. */
#define HLD (GBK + 8)                                  /* fp16 row stride in shared memory (272 bytes) */
#define NBQ_HM_SMEM (2 * 128 * HLD * 2 + 128 * 8 * 4 + 3 * 128 * 4 * 4)
#define NBQ_HMMA884(d, a, b) asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])     : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y))

__device__ __forceinline__ __half2 nbq_i8x2(int x, int sh) {          /* bytes sh/8 and sh/8 + 1 of x as fp16 */
    return __halves2half2(__int2half_rn((x << (24 - sh)) >> 24), __int2half_rn((x << (16 - sh)) >> 24));
}

template <int T>
__device__ __forceinline__ void nbq_gemm_hm(const u8* w, const __half* wd, const signed char* xq, const float* xs,
                                            float* y, int rows, int cols, int nt, int ldy, int yo, int rblk) {
    const bool HALF = (T == T_IQ2S), MINS = (T == T_Q4K || T == T_Q5K);
    extern __shared__ __align__(16) unsigned char gsm[];
    __half* Ah = reinterpret_cast<__half*>(gsm);             /* [128][HLD] */
    __half* Bh = Ah + 128 * HLD;                             /* [128][HLD] */
    float* Asc = reinterpret_cast<float*>(Bh + 128 * HLD);   /* [128][8] */
    float* Amn = Asc + 128 * 8;                              /* [128][4] */
    float* Bsc = Amn + 128 * 4;                              /* [128][4] */
    float* Bsm = Bsc + 128 * 4;                              /* [128][4] */
    NBQ_TAB(T, tab);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row0 = rblk * 128, tok0 = blockIdx.y * 128;
    const int wm = (warp & 1) * 64, wn = (warp >> 1) * 32;
    /* m8n8k4: the warp is 4 quad pairs (lanes 0-3 + 16-19, 4-7 + 20-23, ...), each with its own 8 x 8 tiles; quad
       pair qp takes the row groups 2qp, 2qp + 1 (8 rows each) of the warp's 64 rows and the 4 token groups of its 32
       tokens. qr: this lane's row of A and token of B in a tile; ar0, tc0: its first accumulator row and token */
    const int qp = (lane >> 2) & 3, hi = lane >> 4, qr = (lane & 3) + 4 * hi;
    const int ar0 = wm + 16 * qp + (lane & 1) + 4 * hi, tc0 = lane & 2;
    const int nb = cols >> 8;
    float acc[2][4][8];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int e = 0; e < 8; e++) acc[a][b][e] = 0.0f;
    /* the next step's inputs are loaded into registers while this step computes (and its weight blocks are
       prefetched to L2), so the global reads overlap the tensor-core work */
    int4 bq[4]; float xr[2];
    auto fetch = [&](int k1) {
#pragma unroll
        for (int h = 0; h < 4; h++) {
            int idx = tid + 256 * h, t = idx >> 3, pc = idx & 7, tok = tok0 + t;
            bq[h] = tok < nt ? *reinterpret_cast<const int4*>(xq + (size_t)tok * cols + k1 + 16 * pc) : make_int4(0, 0, 0, 0);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int idx = tid + 256 * h, t = idx >> 2, u = idx & 3, tok = tok0 + t;
            xr[h] = tok < nt ? xs[(size_t)tok * (cols >> 5) + (k1 >> 5) + u] : 0.0f;
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < cols; k0 += GBK) {
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int idx = tid + 256 * h, r = idx >> 2, u = idx & 3;
            int row = min(row0 + r, rows - 1);
            int g = (k0 >> 5) + u, blk = g >> 3, sub = g & 7;
            int v[8]; float s0, s1, mn;
            nbq_run<T>(w + ((size_t)row * nb + blk) * BT<T>::B, wd + (size_t)row * nb + blk, sub, v, s0, s1, mn, tab);
            __half2* dst = reinterpret_cast<__half2*>(Ah + r * HLD + 32 * u);
#pragma unroll
            for (int e = 0; e < 8; e++) { dst[2 * e] = nbq_i8x2(v[e], 0); dst[2 * e + 1] = nbq_i8x2(v[e], 16); }
            if (HALF) { Asc[r * 8 + 2 * u] = s0; Asc[r * 8 + 2 * u + 1] = s1; }
            else Asc[r * 8 + u] = s0;
            if (MINS) Amn[r * 4 + u] = mn;
        }
#pragma unroll
        for (int h = 0; h < 4; h++) {
            int idx = tid + 256 * h, t = idx >> 3, pc = idx & 7;
            __half2* dst = reinterpret_cast<__half2*>(Bh + t * HLD + 16 * pc);
            const int q[4] = {bq[h].x, bq[h].y, bq[h].z, bq[h].w};
#pragma unroll
            for (int e = 0; e < 4; e++) { dst[2 * e] = nbq_i8x2(q[e], 0); dst[2 * e + 1] = nbq_i8x2(q[e], 16); }
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            int idx = tid + 256 * h, t = idx >> 2, u = idx & 3;
            Bsc[t * 4 + u] = xr[h];
        }
        __syncthreads();
        if (k0 + GBK < cols) {
            fetch(k0 + GBK);
#pragma unroll
            for (int h = 0; h < 2; h++) {                     /* this thread's two weight blocks of the next step */
                int idx = tid + 256 * h, r = idx >> 2, u = idx & 3;
                int row = min(row0 + r, rows - 1), g = ((k0 + GBK) >> 5) + u;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(w + ((size_t)row * nb + (g >> 3)) * BT<T>::B));
            }
        }
        if (MINS) {
            for (int idx = tid; idx < 512; idx += 256) {
                int t = idx >> 2, u = idx & 3;
                const __half* p = Bh + t * HLD + 32 * u;
                int sx = 0;
#pragma unroll
                for (int e = 0; e < 32; e++) sx += __half2int_rn(p[e]);
                Bsm[t * 4 + u] = Bsc[t * 4 + u] * (float)sx;
            }
            __syncthreads();
        }
#pragma unroll 1
        for (int u = 0; u < 4; u++) {                        /* the four runs of 32 of this step */
            float d[2][4][8];                                /* this run's dots: [row group][token group][element] */
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int b = 0; b < 4; b++)
#pragma unroll
                    for (int e = 0; e < 8; e++) d[a][b][e] = 0.0f;
            float h0[2][4][8];                               /* IQ2_S: the first half run's dot * its scale */
#pragma unroll
            for (int s = 0; s < 8; s++) {                    /* 8 steps of 4 columns */
                if (HALF && s == 4) {
#pragma unroll
                    for (int a = 0; a < 2; a++) {
                        const float w0a = Asc[(ar0 + 8 * a) * 8 + 2 * u], w0b = Asc[(ar0 + 8 * a + 2) * 8 + 2 * u];
#pragma unroll
                        for (int b = 0; b < 4; b++)
#pragma unroll
                            for (int e = 0; e < 8; e++) {
                                h0[a][b][e] = d[a][b][e] * ((e & 2) ? w0b : w0a);
                                d[a][b][e] = 0.0f;
                            }
                    }
                }
                uint2 fa[2], fb[4];
#pragma unroll
                for (int a = 0; a < 2; a++) fa[a] = *reinterpret_cast<const uint2*>(Ah + (wm + 8 * (2 * qp + a) + qr) * HLD + 32 * u + 4 * s);
#pragma unroll
                for (int b = 0; b < 4; b++) fb[b] = *reinterpret_cast<const uint2*>(Bh + (wn + 8 * b + qr) * HLD + 32 * u + 4 * s);
#pragma unroll
                for (int a = 0; a < 2; a++)
#pragma unroll
                    for (int b = 0; b < 4; b++) NBQ_HMMA884(d[a][b], fa[a], fb[b]);
            }
            /* scale this run's dots: element e of a tile is row (lane & 1) + (e & 2) (+ 4 in the upper half of the
               quad pair), token (e & 4) + (lane & 2) + (e & 1) */
            float xv[4][4], mv[4][4];
#pragma unroll
            for (int b = 0; b < 4; b++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = wn + 8 * b + tc0 + (c & 1) + 4 * (c >> 1);
                    xv[b][c] = Bsc[t * 4 + u];
                    mv[b][c] = MINS ? Bsm[t * 4 + u] : 0.0f;
                }
#pragma unroll
            for (int a = 0; a < 2; a++) {
                const int ra = ar0 + 8 * a, rb = ra + 2;      /* (e & 2) == 0: row ra, else rb */
                const float wa = Asc[ra * 8 + (HALF ? 2 * u + 1 : u)], wb = Asc[rb * 8 + (HALF ? 2 * u + 1 : u)];
                const float na = MINS ? Amn[ra * 4 + u] : 0.0f, nb_ = MINS ? Amn[rb * 4 + u] : 0.0f;
#pragma unroll
                for (int b = 0; b < 4; b++)
#pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const float ws = (e & 2) ? wb : wa, x = xv[b][(e & 1) + 2 * (e >> 2)];
                        if (HALF) acc[a][b][e] += (h0[a][b][e] + d[a][b][e] * ws) * x;
                        else {
                            acc[a][b][e] += d[a][b][e] * ws * x;
                            if (MINS) acc[a][b][e] -= ((e & 2) ? nb_ : na) * mv[b][(e & 1) + 2 * (e >> 2)];
                        }
                    }
            }
        }
    }
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int r = row0 + ar0 + 8 * a + (e & 2), t = tok0 + wn + 8 * b + tc0 + (e & 1) + 4 * (e >> 2);
                if (r < rows && t < nt) y[(size_t)t * ldy + yo + r] = acc[a][b][e];
            }
}
#define NBQ_GEMM_T nbq_gemm_hm
#else
#define NBQ_GEMM_T nbq_gemm
#endif
/* Up to three weight matrices in one launch (Q, K, V or gate, up): blockIdx.x walks the row blocks of segment 0,
   then 1, then 2. Segments 0 and 1 have type TA, segment 2 type TB. */
template <int TA, int TB>
__device__ __forceinline__ void nbq_gemm3(const u8* w0, const __half* d0, int r0, int y0, const u8* w1, const __half* d1,
        int r1, int y1, const u8* w2, const __half* d2, int r2, int y2, const signed char* xq, const float* xs, float* y,
        int cols, int nt, int ldy) {
    int b = blockIdx.x, n0 = (r0 + 127) >> 7, n1 = (r1 + 127) >> 7;
    if (b < n0) NBQ_GEMM_T<TA>(w0, d0, xq, xs, y, r0, cols, nt, ldy, y0, b);
    else if (b < n0 + n1) NBQ_GEMM_T<TA>(w1, d1, xq, xs, y, r1, cols, nt, ldy, y1, b - n0);
    else if (TB != TA) NBQ_GEMM_T<TB>(w2, d2, xq, xs, y, r2, cols, nt, ldy, y2, b - n0 - n1);
    else NBQ_GEMM_T<TA>(w2, d2, xq, xs, y, r2, cols, nt, ldy, y2, b - n0 - n1);
}
#if NBQ_HMMA
#define NBQ_GEMM_LB 1             /* (one block per SM: its shared memory is 78 KB) */
#else
#define NBQ_GEMM_LB 2
#endif
#define NBQ_GEMM3(TA, TB, NAME) extern "C" __global__ void __launch_bounds__(256, NBQ_GEMM_LB) NAME(const u8* w0, const __half* d0,     int r0, int y0, const u8* w1, const __half* d1, int r1, int y1, const u8* w2, const __half* d2, int r2, int y2,     const signed char* xq, const float* xs, float* y, int cols, int nt, int ldy) {     nbq_gemm3<TA, TB>(w0, d0, r0, y0, w1, d1, r1, y1, w2, d2, r2, y2, xq, xs, y, cols, nt, ldy); }
#if NBQ_TC || NBQ_SIMT || NBQ_HMMA   /* int8 tensor cores (sm_80+); Volta: dp4a or fp16 tensor cores */
NBQ_GEMM3(T_Q4K, T_Q4K, nbq_mm_q4k)
NBQ_GEMM3(T_Q5K, T_Q5K, nbq_mm_q5k)
NBQ_GEMM3(T_IQ3XXS, T_IQ3XXS, nbq_mm_iq3xxs)
NBQ_GEMM3(T_IQ3S, T_IQ3S, nbq_mm_iq3s)
NBQ_GEMM3(T_IQ2S, T_IQ2S, nbq_mm_iq2s)
NBQ_GEMM3(T_IQ2S, T_Q4K, nbq_mm_iq2s_q4k)
#endif

/* ---------------------------------------------------------------- prompt: per-token kernels */
/* x[t] (+= add[t]) -> RMS norm * w -> int8 runs; one block of 1024 threads per token */
extern "C" __global__ void __launch_bounds__(1024) nbq_add_rms_q8_rows(float* x, const float* add, const float* w, int n,
        float eps, signed char* xq, float* xs) {
    const size_t o = (size_t)blockIdx.x * n;
    __shared__ float red[32];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float ss = 0.0f;
    for (int i = tid; i < n; i += 1024) {
        float v = x[o + i];
        if (add) { v += add[o + i]; x[o + i] = v; }
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
        nbq_q8_run(__fmul_rn(__fmul_rn(scale, x[o + i]), w[i]), g, lane, xq + o, xs + o / 32);
    }
}
/* SwiGLU of gu [nt][2n] (gate | up) -> int8 runs [nt][n]; grid (n / 256, nt) */
extern "C" __global__ void nbq_swiglu_q8_rows(const float* gu, int n, signed char* xq, float* xs) {
    int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (g >= n / 32) return;
    const float* p = gu + (size_t)t * 2 * n;
    int i = 32 * g + lane;
    float a = p[i], b = p[n + i];
    nbq_q8_run(a / (1.0f + expf(-a)) * b, g, lane, xq + (size_t)t * n, xs + (size_t)t * (n / 32));
}
/* fp32 rows [nt][n] -> int8 runs; grid (n / 256, nt) */
extern "C" __global__ void nbq_q8_rows(const float* x, int n, signed char* xq, float* xs) {
    int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (g >= n / 32) return;
    nbq_q8_run(x[(size_t)t * n + 32 * g + lane], g, lane, xq + (size_t)t * n, xs + (size_t)t * (n / 32));
}
/* RoPE + K/V store for nt tokens at positions pos[t]; qkv rows [nt][(nh + 2 nkv) 128]; grid (nh + nkv, nt) */
extern "C" __global__ void nbq_rope_store_rows(float* qkv, const int* pos, int nh, int nkv, float theta_scale, __half* kc,
        __half* vc, int ctx) {
    int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const int p = pos[t];
    float* row = qkv + (size_t)t * (nh + 2 * nkv) * 128;
    const float theta = (float)p * powf(theta_scale, (float)i);
    const float c = cosf(theta), s = sinf(theta);
    if (h < nh) {
        float* x = row + h * 128;
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        nbq_rot(x0, x1, c, s, x[2 * i], x[2 * i + 1]);
    } else {
        int hk = h - nh;
        const float* x = row + nh * 128 + hk * 128;
        const float* v = row + (nh + nkv) * 128 + hk * 128;
        float x0 = x[2 * i], x1 = x[2 * i + 1];
        size_t ko = nbq_kofs(hk, ctx, p, i >> 2) + 2 * (i & 3);
        float y0, y1;
        nbq_rot(x0, x1, c, s, y0, y1);
        kc[ko] = __float2half(y0);
        kc[ko + 1] = __float2half(y1);
        size_t o = ((size_t)hk * ctx + p) * 128;
        vc[o + 2 * i] = __float2half(v[2 * i]);
        vc[o + 2 * i + 1] = __float2half(v[2 * i + 1]);
    }
}
