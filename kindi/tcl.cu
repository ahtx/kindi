/* tcl.cu - the products of the tc plans (spec.py) on sm_60 (Tesla P100): no dp4a, so the weights stay in the 5-trit
   lane layout and each product is k_g5n4's lane-table product (fp32 activations, kindi.cu g5n_tables), for up to 8
   tokens of a pass. Compiled after kindi.cu (g5n_tables, h2f, KLDG, KSHFL, KSHFLX).

   k_g5t*: grid (ceil(R / (16 U nw)), sp), block 32 nw (nw = 9: one warp per table part of g5n_tables). Block (bx, by) takes the rows of its warps' U tiles and the
   column groups [by NG / sp, (by + 1) NG / sp). For each group the weights are loaded once; then, token after
   token, the block builds that token's table (g5n_tables) and adds the group's products to the token's sums. A
   token's arithmetic (its table, the order of the groups and of the additions, the lane reduction) is k_g5n4's and
   does not depend on U or on how many tokens share the pass, so decode (T = 1) and verify (T = k + 1) give the same
   bits. Token j's sums of split by go to y + (j sp + by) R (the tc plans' part layout [token][split][row]). */

template <int U, int NT>
__device__ __forceinline__ void g5t_core(const uint4* __restrict__ W, float* __restrict__ y, int R, int NG,
                                         const float* __restrict__ x, int ldx){
  __shared__ float T[256 * 32];        /* declared here (one instance per kernel): the lookups stay LDS, not generic loads */
  const int t = threadIdx.x, nt = blockDim.x, lane = t & 31, nw = nt >> 5;
  const int NT16 = R >> 4, sp = gridDim.y;
  const int g_lo = (int)((long long)blockIdx.y * NG / sp), g_hi = (int)((long long)(blockIdx.y + 1) * NG / sp);
  const int m0 = (blockIdx.x * nw + (t >> 5)) * U;
  const char* Tl = (const char*)(T + lane);
  for(int e = t; e < 256 * 32; e += nt) T[e] = 0.f;   /* entries 243 .. 255 and lanes 26 .. 31 stay zero */
  float acc[NT][U][16];
#pragma unroll
  for(int j = 0; j < NT; ++j)
#pragma unroll
    for(int u = 0; u < U; ++u)
#pragma unroll
      for(int k = 0; k < 16; ++k) acc[j][u][k] = 0.f;
  uint4 w[U], wn[U];                       /* the group's weights and the next group's (loaded one group ahead) */
#pragma unroll
  for(int u = 0; u < U; ++u){
    const int m = m0 + u;
    wn[u] = make_uint4(0u, 0u, 0u, 0u);
    if(g_lo < g_hi && m < NT16 && lane < 28) wn[u] = KLDG(W + ((long long)g_lo * NT16 + m) * 28 + lane);
  }
  for(int g = g_lo; g < g_hi; ++g){
#pragma unroll
    for(int u = 0; u < U; ++u){
      const int m = m0 + u;
      w[u] = wn[u];
      if(g + 1 < g_hi && m < NT16 && lane < 28) wn[u] = KLDG(W + ((long long)(g + 1) * NT16 + m) * 28 + lane);
    }
    /* the 16 row scales of each tile, once per group: lane k < 16 converts row k's (k = 4 i + q: byte q of word i
       of lanes 26 (low) and 27 (high)); a lookup then takes its row's scale by shuffle (no conversion per lookup) */
    float ds[U];
#pragma unroll
    for(int u = 0; u < U; ++u){
      const unsigned int l0 = KSHFL(w[u].x, 26), h0 = KSHFL(w[u].x, 27), l1 = KSHFL(w[u].y, 26), h1 = KSHFL(w[u].y, 27);
      const unsigned int l2 = KSHFL(w[u].z, 26), h2 = KSHFL(w[u].z, 27), l3 = KSHFL(w[u].w, 26), h3 = KSHFL(w[u].w, 27);
      const int i = (lane >> 2) & 3, q = lane & 3;
      const unsigned int lo = i == 0 ? l0 : i == 1 ? l1 : i == 2 ? l2 : l3, hi = i == 0 ? h0 : i == 1 ? h1 : i == 2 ? h2 : h3;
      ds[u] = h2f((unsigned short)__byte_perm(lo, hi, q | ((4 + q) << 4)));
    }
#pragma unroll
    for(int j = 0; j < NT; ++j){
      __syncthreads();
      g5n_tables(T, x + (long long)j * ldx, g, lane, t >> 5, nw);
      __syncthreads();
#pragma unroll
      for(int u = 0; u < U; ++u){
        const unsigned int wv[4] = {w[u].x, w[u].y, w[u].z, w[u].w};
#pragma unroll
        for(int i = 0; i < 4; ++i){
#pragma unroll
          for(int q = 0; q < 4; ++q){
            const unsigned int c = (wv[i] >> (8 * q)) & 255u;
            const float v = *(const float*)(Tl + (c << 7));
            const float d = KSHFL(ds[u], 4 * i + q);
            acc[j][u][4 * i + q] += v * d;
          }
        }
      }
    }
  }
#pragma unroll
  for(int j = 0; j < NT; ++j){
#pragma unroll
    for(int u = 0; u < U; ++u){
      float v[16];
#pragma unroll
      for(int k = 0; k < 16; ++k) v[k] = acc[j][u][k];
#pragma unroll
      for(int lv = 0; lv < 4; ++lv){
        const int h = 8 >> lv, msk = 16 >> lv;
        const bool up = (lane & msk) != 0;
#pragma unroll
        for(int k = 0; k < 8; ++k){
          if(k < h){
            const float send = up ? v[k] : v[k + h], keep = up ? v[k + h] : v[k];
            v[k] = keep + KSHFLX(send, msk);
          }
        }
      }
      const float s = v[0] + KSHFLX(v[0], 1);
      const int m = m0 + u;
      if(m < NT16 && !(lane & 1)) y[((long long)j * sp + blockIdx.y) * R + m * 16 + ((lane >> 1) & 15)] = s;
    }
  }
}

/* one kernel per (tiles per warp U, token count NT): k_g5t4 (U 4, one token: the decode step), k_g5t2_N (U 2, N = 2 .. 4
   tokens), k_g5t1_N (U 1, N = 5 .. 8) */
#define G5T(name, U, NT) extern "C" __global__ void __launch_bounds__(288) name(const uint4* __restrict__ W,     float* __restrict__ y, int R, int NG, const float* __restrict__ x, int ldx, int ntok){ g5t_core<U, NT>(W, y, R, NG, x, ldx); }
G5T(k_g5t4, 4, 1)
G5T(k_g5t2_2, 2, 2)
G5T(k_g5t2_3, 2, 3)
G5T(k_g5t2_4, 2, 4)
G5T(k_g5t1_5, 1, 5)
G5T(k_g5t1_6, 1, 6)
G5T(k_g5t1_7, 1, 7)
G5T(k_g5t1_8, 1, 8)
