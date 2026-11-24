/* tca.cu - prompt-chunk attention on the tensor cores (KINDI_ATT_TC=1; sm_80+, the patched-PTX module).

   k_att_tc_T has k_att_flash_T's interface and job: all T tokens of a chunk, one K/V head per block row, 64 query rows
   (t, j) per block, positions in tiles, online softmax, the output gated (or, with Z > 1, partial results in PO for
   k_att_merge_T). Scores: int8 mma.sync m16n8k32 over each 32-value q8_0 block (an exact integer), times dq * dk,
   the 8 blocks added in order (fma), times 1/16 - the same float scores as k_attn_sq8t / k_att_flash_T. Values: the
   probabilities in fp16 against V dequantised to fp16 (q8_0 value x fp16 scale, rounded to fp16, as llama.cpp's
   flash attention does), mma.sync m16n8k16 with fp32 accumulation.
   Block: 256 threads = 8 warps; warp w takes query rows 16 (w >> 1) .. + 15 and features 128 (w & 1) .. + 127 (the
   two feature halves of a row group compute the same scores). Each warp keeps its Q fragments in registers; the K
   tiles (32 positions) are double-buffered with cp.async, and a tile's V bytes are loaded into registers before its
   scores are computed. */

#define ATC_TILE 32
#define ATC_QS 272            /* row stride (bytes) of the int8 K tile in shared memory (conflict-free) */
#define ATC_VS 40             /* row stride (halves) of the transposed fp16 V tile */

__device__ __forceinline__ unsigned int atc_h2(float lo, float hi){
  unsigned int r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

#ifdef KINDI_TURING
/* Turing (tc.cu KINDI_TURING): the int8 product as tc.cu's TC_MMA (exact); the fp16 one as two m16n8k8 (A a0 a1 =
   k 0..7, a2 a3 = k 8..15; B b0 = k 0..7, b1 = k 8..15) */
#define ATC_MMA_S8(c, a0, a1, a2, a3, b0, b1) TC_MMA(c, a0, a1, a2, a3, b0, b1)
#define ATC_M168(c, a0, a1, b) \
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b))
#define ATC_MMA_F16(c, a0, a1, a2, a3, b0, b1) do { ATC_M168(c, a0, a1, b0); ATC_M168(c, a2, a3, b1); } while(0)
__device__ __forceinline__ void tc_cp8(void* s, const void* g){ *(uint2*)s = __ldg((const uint2*)g); }
#else
#define ATC_MMA_S8(c, a0, a1, a2, a3, b0, b1) \
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define ATC_MMA_F16(c, a0, a1, a2, a3, b0, b1) \
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

__device__ __forceinline__ void tc_cp8(void* s, const void* g){
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(tc_smem(s)), "l"(g));
}
#endif

__device__ __forceinline__ void atc_stage_k(unsigned char* Ks, unsigned short* dk, const uint2* __restrict__ Kq,
                                            const unsigned short* __restrict__ Kd, int p0, int nblk, int n_kv, int kv){
  const int tid = threadIdx.x;
  for(int i = tid; i < ATC_TILE * 16; i += 256){
    const int pp = i >> 4, c16 = i & 15, p = p0 + pp;
    if(p < nblk) tc_cp16(Ks + pp * ATC_QS + c16 * 16, (const uint4*)(Kq + ((long long)p * n_kv + kv) * 32) + c16);
    else *(uint4*)(Ks + pp * ATC_QS + c16 * 16) = make_uint4(0u, 0u, 0u, 0u);
  }
  if(tid < ATC_TILE * 2){
    const int pp = tid >> 1, hlf = tid & 1, p = p0 + pp;
    if(p < nblk) tc_cp8(dk + pp * 8 + hlf * 4, Kd + ((long long)p * n_kv + kv) * 8 + hlf * 4);
    else *(uint2*)(dk + pp * 8 + hlf * 4) = make_uint2(0u, 0u);
  }
  tc_commit();
}

extern "C" __global__ void __launch_bounds__(256) k_att_tc_T(const uint2* __restrict__ Kq,
    const unsigned short* __restrict__ Kd, const uint4* __restrict__ Vq, const unsigned short* __restrict__ Vd,
    const signed char* __restrict__ qq, const float* __restrict__ qd, const float* __restrict__ gatev, int ldg,
    float* __restrict__ att, int lda, int pos, int T, int n_kv, int G, int NH, float* __restrict__ PO){
  __shared__ __align__(16) unsigned char Ks[2][ATC_TILE * ATC_QS];
  __shared__ __align__(16) unsigned short dks[2][ATC_TILE * 8];
  __shared__ __align__(16) unsigned short VT[256 * ATC_VS];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g8 = lane >> 2, t = lane & 3;
  const int kv = blockIdx.y, r0 = blockIdx.x * 64, nrow = T * G, QL = NH * 256;
  const int rlast = min(r0 + 63, nrow - 1), nblk = pos + rlast / G + 1;
  const int rg = (w >> 1) * 16, fh = (w & 1) * 128;
  /* this thread's two rows: their Q fragments (8 blocks x 4 words), block scales and position limits */
  const int ra = r0 + rg + g8, rb = ra + 8;
  unsigned int qa[8][4];
  float dqa[8], dqb[8];
  {
    const bool oka = ra < nrow, okb = rb < nrow;
    const int ta = ra / G, ja = ra - ta * G, tb = rb / G, jb = rb - tb * G;
    const signed char* qra = qq + (long long)ta * QL + (kv * G + ja) * 256;
    const signed char* qrb = qq + (long long)tb * QL + (kv * G + jb) * 256;
#pragma unroll
    for(int b = 0; b < 8; ++b){
      qa[b][0] = oka ? *(const unsigned int*)(qra + b * 32 + 4 * t) : 0u;
      qa[b][1] = okb ? *(const unsigned int*)(qrb + b * 32 + 4 * t) : 0u;
      qa[b][2] = oka ? *(const unsigned int*)(qra + b * 32 + 16 + 4 * t) : 0u;
      qa[b][3] = okb ? *(const unsigned int*)(qrb + b * 32 + 16 + 4 * t) : 0u;
      dqa[b] = oka ? qd[(long long)ta * (QL / 32) + (kv * G + ja) * 8 + b] : 0.f;
      dqb[b] = okb ? qd[(long long)tb * (QL / 32) + (kv * G + jb) * 8 + b] : 0.f;
    }
  }
  const int na = ra < nrow ? pos + ra / G + 1 : 0, nb = rb < nrow ? pos + rb / G + 1 : 0;
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for(int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  const int ntl = (nblk + ATC_TILE - 1) / ATC_TILE, z = blockIdx.z, Z = gridDim.z;
  const int tlo = (int)((long long)z * ntl / Z), thi = (int)((long long)(z + 1) * ntl / Z);
  const int f8 = (tid & 31) * 8;
  if(tlo < thi) atc_stage_k(Ks[0], dks[0], Kq, Kd, tlo * ATC_TILE, nblk, n_kv, kv);
  for(int it = tlo; it < thi; ++it){
    const int p0 = it * ATC_TILE, buf = (it - tlo) & 1;
    /* this tile's V bytes (2 passes x 2 positions x 8 features per thread) into registers, early */
    uint2 wv[2][2];
    float vdv[2][2];
#pragma unroll
    for(int pass = 0; pass < 2; ++pass){
      const int pp = 2 * ((tid >> 5) + 8 * pass);
#pragma unroll
      for(int e = 0; e < 2; ++e){
        const int p = p0 + pp + e;
        wv[pass][e] = make_uint2(0u, 0u);
        vdv[pass][e] = 0.f;
        if(p < nblk){
          wv[pass][e] = KLDG((const uint2*)(Vq + ((long long)p * n_kv + kv) * 16) + (f8 >> 3));
          vdv[pass][e] = h2f(KLDG(Vd + ((long long)p * n_kv + kv) * 8 + (f8 >> 5)));
        }
      }
    }
    if(it + 1 < thi){
      atc_stage_k(Ks[buf ^ 1], dks[buf ^ 1], Kq, Kd, p0 + ATC_TILE, nblk, n_kv, kv);
      tc_wait1();
    } else {
      tc_wait0();
    }
    __syncthreads();
    const unsigned char* K = Ks[buf];
    const unsigned short* dkt = dks[buf];
    /* scores: 16 rows x 32 positions per warp, the 8 blocks in order */
    float s[4][4];
#pragma unroll
    for(int nt = 0; nt < 4; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
    for(int b = 0; b < 8; ++b){
#pragma unroll
      for(int nt = 0; nt < 4; ++nt){
        const int pp = nt * 8 + g8;
        const unsigned int b0 = *(const unsigned int*)(K + pp * ATC_QS + b * 32 + 4 * t);
        const unsigned int b1 = *(const unsigned int*)(K + pp * ATC_QS + b * 32 + 16 + 4 * t);
        int c[4] = {0, 0, 0, 0};
        ATC_MMA_S8(c, qa[b][0], qa[b][1], qa[b][2], qa[b][3], b0, b1);
        const float dk0 = h2f(dkt[(nt * 8 + 2 * t) * 8 + b]), dk1 = h2f(dkt[(nt * 8 + 2 * t + 1) * 8 + b]);
        s[nt][0] = __fmaf_rn((float)c[0], dqa[b] * dk0, s[nt][0]);
        s[nt][1] = __fmaf_rn((float)c[1], dqa[b] * dk1, s[nt][1]);
        s[nt][2] = __fmaf_rn((float)c[2], dqb[b] * dk0, s[nt][2]);
        s[nt][3] = __fmaf_rn((float)c[3], dqb[b] * dk1, s[nt][3]);
      }
    }
    /* mask, online softmax per row (a quad of lanes shares a row) */
    float mt[2] = {-1e30f, -1e30f};
#pragma unroll
    for(int nt = 0; nt < 4; ++nt){
      const int p = p0 + nt * 8 + 2 * t;
      s[nt][0] = p < na ? s[nt][0] * 0.0625f : -1e30f;
      s[nt][1] = p + 1 < na ? s[nt][1] * 0.0625f : -1e30f;
      s[nt][2] = p < nb ? s[nt][2] * 0.0625f : -1e30f;
      s[nt][3] = p + 1 < nb ? s[nt][3] * 0.0625f : -1e30f;
      mt[0] = fmaxf(mt[0], fmaxf(s[nt][0], s[nt][1]));
      mt[1] = fmaxf(mt[1], fmaxf(s[nt][2], s[nt][3]));
    }
#pragma unroll
    for(int h = 0; h < 2; ++h){
      mt[h] = fmaxf(mt[h], __shfl_xor_sync(0xffffffffu, mt[h], 1));
      mt[h] = fmaxf(mt[h], __shfl_xor_sync(0xffffffffu, mt[h], 2));
    }
    float sc[2], rs[2] = {0.f, 0.f};
#pragma unroll
    for(int h = 0; h < 2; ++h){
      const float mn = fmaxf(m[h], mt[h]);
      sc[h] = m[h] <= -1e29f ? 0.f : expf(m[h] - mn);
      m[h] = mn;
    }
    unsigned int pa[4][2];                       /* fp16 probabilities: [n-tile][row half] (cols 2t, 2t + 1) */
#pragma unroll
    for(int nt = 0; nt < 4; ++nt){
      const float e0 = m[0] <= -1e29f ? 0.f : expf(s[nt][0] - m[0]), e1 = m[0] <= -1e29f ? 0.f : expf(s[nt][1] - m[0]);
      const float e2 = m[1] <= -1e29f ? 0.f : expf(s[nt][2] - m[1]), e3 = m[1] <= -1e29f ? 0.f : expf(s[nt][3] - m[1]);
      rs[0] += e0 + e1;
      rs[1] += e2 + e3;
      pa[nt][0] = atc_h2(e0, e1);
      pa[nt][1] = atc_h2(e2, e3);
    }
#pragma unroll
    for(int h = 0; h < 2; ++h){
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 1);
      rs[h] += __shfl_xor_sync(0xffffffffu, rs[h], 2);
      l[h] = l[h] * sc[h] + rs[h];
    }
#pragma unroll
    for(int i = 0; i < 16; ++i){ o[i][0] *= sc[0]; o[i][1] *= sc[0]; o[i][2] *= sc[1]; o[i][3] *= sc[1]; }
    /* the V tile, dequantised to fp16 and transposed: VT[f][p] (from the registers loaded above) */
#pragma unroll
    for(int pass = 0; pass < 2; ++pass){
      const int pp = 2 * ((tid >> 5) + 8 * pass);
#pragma unroll
      for(int k = 0; k < 8; ++k){
        const unsigned int b0 = (k < 4 ? wv[pass][0].x : wv[pass][0].y) >> (8 * (k & 3));
        const unsigned int b1 = (k < 4 ? wv[pass][1].x : wv[pass][1].y) >> (8 * (k & 3));
        *(unsigned int*)(VT + (f8 + k) * ATC_VS + pp) = atc_h2(sb2f(b0) * vdv[pass][0], sb2f(b1) * vdv[pass][1]);
      }
    }
    __syncthreads();
    /* values: o[16 rows x 128 features] += P[16 x 32] V[32 x 128] */
#pragma unroll
    for(int ks = 0; ks < 2; ++ks){
      const unsigned int a0 = pa[2 * ks][0], a1 = pa[2 * ks][1], a2 = pa[2 * ks + 1][0], a3 = pa[2 * ks + 1][1];
#pragma unroll
      for(int ft = 0; ft < 16; ++ft){
        const int f = fh + ft * 8 + g8;
        const unsigned int b0 = *(const unsigned int*)(VT + f * ATC_VS + ks * 16 + 2 * t);
        const unsigned int b1 = *(const unsigned int*)(VT + f * ATC_VS + ks * 16 + 8 + 2 * t);
        ATC_MMA_F16(o[ft], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();                             /* (VT and this K buffer are rewritten next round) */
  }
  /* out: rows ra (o[.][0..1]) and rb (o[.][2..3]), features fh + 8 ft + 2 t, + 1 */
#pragma unroll
  for(int h = 0; h < 2; ++h){
    const int r = h ? rb : ra;
    if(r >= nrow) continue;
    if(Z > 1){
      float* po = PO + (((long long)z * n_kv + kv) * nrow + r) * 260;
#pragma unroll
      for(int ft = 0; ft < 16; ++ft){
        const int f = fh + ft * 8 + 2 * t;
        po[f] = o[ft][2 * h];
        po[f + 1] = o[ft][2 * h + 1];
      }
      if(fh == 0 && t == 0){ po[256] = m[h] <= -1e29f ? __int_as_float(0xff800000) : m[h]; po[257] = l[h]; }
      continue;
    }
    const int tq = r / G, j = r - tq * G, hh = kv * G + j;
    const float il = 1.0f / l[h];
#pragma unroll
    for(int ft = 0; ft < 16; ++ft){
#pragma unroll
      for(int e = 0; e < 2; ++e){
        const int f = fh + ft * 8 + 2 * t + e;
        const float g = gatev[(long long)tq * ldg + hh * 512 + 256 + f];
        att[(long long)tq * lda + hh * 256 + f] = (o[ft][2 * h + e] * il) * (1.0f / (1.0f + expf(-g)));
      }
    }
  }
}
