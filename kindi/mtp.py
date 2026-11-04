"""Multi-token prediction: the MTP-head drafter (tensor-core path).

KINDI_SPEC=mtp (or auto), KINDI_MTP_HEAD=<head.npz>: after every pass of the main model the head reads that pass's
verified rows into its own K/V window and drafts KINDI_SPEC_K tokens. The main model verifies the draft in one
pass (the speculative verify path), so the output is exactly the main model's (greedy)."""
import json
import os

import numpy as np
import cupy as cp

i32 = np.int32


# a recorder hook for the development tools (None in normal use; the engine calls DUMP.rows when it is set)
DUMP = None


# ====================================================================================== the head's kernels
SRC = r"""
extern "C" __global__ void m_gemv8(const signed char* __restrict__ W, const float* __restrict__ sc,
                                   const float* __restrict__ x, float* __restrict__ y, int R, int C, int T){
  /* y[t][r] = sc[r] * sum_c W[r][c] x[t][c], T <= 8 rows. One warp per 4 output rows: each lane loads 16 weights
     of each of its 4 rows once and 16 inputs of each token row once, so every input load serves 4 rows */
  const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * 4, lane = threadIdx.x & 31;
  if(r0 >= R) return;
  float acc[4][8];
  #pragma unroll
  for(int r = 0; r < 4; ++r){
    #pragma unroll
    for(int t = 0; t < 8; ++t) acc[r][t] = 0.f;
  }
  for(int c = lane * 16; c < C; c += 512){
    int4 q[4];
    #pragma unroll
    for(int r = 0; r < 4; ++r) q[r] = (r0 + r < R) ? *(const int4*)(W + (long long)(r0 + r) * C + c) : make_int4(0, 0, 0, 0);
    #pragma unroll
    for(int t = 0; t < 8; ++t){
      if(t < T){
        const float4* xp = (const float4*)(x + (long long)t * C + c);
        float xv[16];
        #pragma unroll
        for(int i = 0; i < 4; ++i){ const float4 f = xp[i]; xv[4 * i] = f.x; xv[4 * i + 1] = f.y; xv[4 * i + 2] = f.z; xv[4 * i + 3] = f.w; }
        #pragma unroll
        for(int r = 0; r < 4; ++r){
          const signed char* qb = (const signed char*)&q[r];
          float s = 0.f;
          #pragma unroll
          for(int i = 0; i < 16; ++i) s += (float)qb[i] * xv[i];
          acc[r][t] += s;
        }
      }
    }
  }
  #pragma unroll
  for(int r = 0; r < 4; ++r){
    #pragma unroll
    for(int t = 0; t < 8; ++t){
      if(t < T){
        float v = acc[r][t];
        for(int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if(lane == 0 && r0 + r < R) y[(long long)t * R + r0 + r] = v * sc[r0 + r];
      }
    }
  }
}

extern "C" __global__ void m_rms(const float* __restrict__ x, const float* __restrict__ w, float* out, int n,
                                 int ldx, int ldo){
  /* one row per block (256 threads): out = x / rms(x) * w (w = 1 + the Qwen weight); in place is allowed */
  const float* xr = x + (long long)blockIdx.x * ldx;
  float* o = out + (long long)blockIdx.x * ldo;
  __shared__ float red[8];
  float s = 0.f;
  for(int e = threadIdx.x; e < n; e += 256){ const float v = xr[e]; s += v * v; }
  for(int k = 16; k; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
  if((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
  __syncthreads();
  if(threadIdx.x < 32){
    s = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    for(int k = 16; k; k >>= 1) s += __shfl_xor_sync(0xffffffffu, s, k);
    if(threadIdx.x == 0) red[0] = s;
  }
  __syncthreads();
  const float r = rsqrtf(red[0] / n + 1e-6f);
  for(int e = threadIdx.x; e < n; e += 256) o[e] = xr[e] * r * w[e];
}

extern "C" __global__ void m_rope(float* x, int ld, int nh, int p0, int rd, double theta){
  /* rotate-half RoPE on the first rd dims of row r = t * nh + h (position p0 + t); block rd / 2 threads */
  const int r = blockIdx.x, t = r / nh, j = threadIdx.x;
  float* v = x + (long long)r * ld;
  const double ang = (double)(p0 + t) * pow(theta, -2.0 * j / rd);
  const float c = (float)cos(ang), s = (float)sin(ang);
  const float a = v[j], b = v[j + rd / 2];
  v[j] = a * c - b * s;
  v[j + rd / 2] = a * s + b * c;
}

extern "C" __global__ void m_kvstore(const float* __restrict__ k, const float* __restrict__ v, float* Kr, float* Vr,
                                     int* spos, int p0, int W, int L){
  /* row t -> window slot (p0 + t) % W */
  const int t = blockIdx.x, slot = (p0 + t) % W;
  for(int e = threadIdx.x; e < L; e += 256){
    Kr[(long long)slot * L + e] = k[(long long)t * L + e];
    Vr[(long long)slot * L + e] = v[(long long)t * L + e];
  }
  if(threadIdx.x == 0) spos[slot] = p0 + t;
}

extern "C" __global__ void m_attn(const float* __restrict__ q, const float* __restrict__ Kr,
                                  const float* __restrict__ Vr, const int* __restrict__ spos, float* __restrict__ out,
                                  int p0, int start, int W, int NH, int NHKV){
  /* grid (NH, T), block 256 (= head dim): query (t, h) at position p = p0 + t sees the window slots whose position
     is in [max(start, p - W + 1), p]; q rows are [q | gate] (2 x 256); out = softmax(q.K / 16) V * sigmoid(gate) */
  const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x, p = p0 + t, kv = h / (NH / NHKV), L = NHKV * 256;
  __shared__ float sc[64];
  __shared__ float red[8];
  const float qd = q[((long long)t * NH + h) * 512 + d];
  for(int s = 0; s < W; ++s){
    const int ps = spos[s];
    const bool ok = ps >= start && ps <= p && ps > p - W;
    float v = ok ? qd * Kr[(long long)s * L + kv * 256 + d] : 0.f;
    for(int k = 16; k; k >>= 1) v += __shfl_xor_sync(0xffffffffu, v, k);
    if((d & 31) == 0) red[d >> 5] = v;
    __syncthreads();
    if(d == 0){
      float a = 0.f;
      for(int i = 0; i < 8; ++i) a += red[i];
      sc[s] = ok ? a * 0.0625f : -1e30f;
    }
    __syncthreads();
  }
  if(d == 0){
    float mx = -1e30f, tot = 0.f;
    for(int s = 0; s < W; ++s) mx = fmaxf(mx, sc[s]);
    for(int s = 0; s < W; ++s){ const float e = sc[s] > -1e29f ? expf(sc[s] - mx) : 0.f; sc[s] = e; tot += e; }
    for(int s = 0; s < W; ++s) sc[s] /= tot;
  }
  __syncthreads();
  float acc = 0.f;
  for(int s = 0; s < W; ++s) if(sc[s] > 0.f) acc += sc[s] * Vr[(long long)s * L + kv * 256 + d];
  const float g = q[((long long)t * NH + h) * 512 + 256 + d];
  out[((long long)t * NH + h) * 256 + d] = acc / (1.f + expf(-g));
}
"""


class _K(object):
    def __init__(self):
        mod = cp.RawModule(code=SRC)
        for k in ("m_gemv8", "m_rms", "m_rope", "m_kvstore", "m_attn"):
            setattr(self, k, mod.get_function(k))
        self.silumul = cp.ElementwiseKernel("float32 g, float32 u", "float32 f", "f = g / (1.f + expf(-g)) * u",
                                            "m_silumul")


# ====================================================================================== the drafter
class MTPDraft(object):
    """The MTP head on the main model's card. It drafts with the main model's own output head (its plan kernels)."""

    MATS = ("fc", "q", "k", "v", "o", "gate", "up", "down")
    NORMS = ("e_norm", "h_norm", "in_norm", "post_norm", "q_norm", "k_norm", "out_norm")

    def __init__(self, eng, path, log=print):
        from . import spec as _sp
        from . import tp as _tp
        self.e, s = eng, eng.cards[0]
        self.s = s
        z = np.load(path)
        m = self.meta = json.loads(bytes(z["meta"]).decode())
        D, NH, NHKV, HD, NFF, W = m["D"], m["NH"], m["NHKV"], m["HD"], m["NFF"], m["W"]
        assert D == s.D and HD == 256 and W <= 64, (D, s.D, HD, W)
        self.D, self.NH, self.NHKV, self.HD, self.NFF, self.W = D, NH, NHKV, HD, NFF, W
        T = _sp.TMAX
        with cp.cuda.Device(s.card):
            self.k = _K()
            self.w = {n: (cp.asarray(z[n + ".w"]), cp.asarray(z[n + ".s"].astype(np.float32))) for n in self.MATS}
            self.n = {n: cp.asarray(z[n].astype(np.float32)) for n in self.NORMS}
            f = lambda n: cp.zeros(int(n), dtype=cp.float32)  # noqa: E731
            L = NHKV * HD
            self.PH, self.HIN = f(T * D), f(D)
            self.EB, self.X2, self.U, self.A, self.Y = f(T * D), f(T * 2 * D), f(T * D), f(T * D), f(T * D)
            self.QG, self.KB, self.VB, self.AO = f(T * NH * 2 * HD), f(T * L), f(T * L), f(T * NH * HD)
            self.G, self.UU, self.F, self.HH = f(T * NFF), f(T * NFF), f(T * NFF), f(T * D)
            self.Kr, self.Vr = f(W * L), f(W * L)
            self.spos = cp.full(W, -(1 << 30), dtype=cp.int32)
            # the output head: the main model's norm + fold + int8 rows + head GEMV + argmax, on HH's last row
            self.src, self.ones = f(D), cp.ones(D, dtype=cp.float32)
            self.xn, self.h2, self.xp = f(D), f(D), f(s.tcb["work"])
            self.yh, self.aout = f(s.R_head), cp.zeros(1, dtype=cp.int32)
            self.headg = _sp.Graph(self._head_plan(_sp, _tp))
        self.start, self.pend = None, None
        self.stats = {"calls": 0, "drafts": 0}
        log("MTP head attached: %s (%.0f MB, window %d)" % (os.path.basename(path), sum(
            a.nbytes for a, _ in self.w.values()) / 1e6, W))

    def _head_plan(self, _sp, _tp):
        s = self.s
        SK, b = s.sk, s.tcb
        D, HB, work = s.D, s.HB, b["work"]
        TT = i32(1)
        P = _tp.Plan()
        P.add(SK.k_rmsfold_qT, (D // HB, 1), (512,), (self.src, b["Y"], self.h2, self.ones, s.sD, self.xn, self.xp,
                                                      i32(D), i32(0), i32(HB), TT, i32(D), i32(D), i32(D), i32(D),
                                                      i32(work), b["XQ"], b["XS"], b["XA"]))
        _, Wt, R, C = s.w[s.head_name]
        R, C = int(R), int(C)
        nw, sp = _sp.tune(s, R, C)
        gx = (R // 16 + nw - 1) // nw
        if sp == 1:
            P.add(SK.mv[nw], (gx, 1), (32 * nw,), (Wt, self.yh, i32(R), i32(C // 1024), b["XQ"], b["XS"], b["XA"], TT))
        else:
            P.add(SK.mv[nw], (gx, sp), (32 * nw,), (Wt, b["PART"], i32(R), i32(C // 1024), b["XQ"], b["XS"], b["XA"], TT))
            P.add(SK.k_tc_sum, ((R + 255) // 256,), (256,), (b["PART"], self.yh, i32(R), TT, i32(sp)))
        P.add(SK.k_argmax_T, (1, 1), (1024,), (self.yh, i32(s.R_head), self.aout, TT, i32(s.R_head), i32(1)))
        return P

    # ------------------------------------------------------------------ history
    def reset(self):
        self.start, self.pend = None, None
        self.spos.fill(-(1 << 30))

    def gap(self):
        """the main model read tokens without its head (a prompt): the head's history starts again"""
        self.start, self.pend = None, None
        self.spos.fill(-(1 << 30))

    def seen(self, pos, n, nexts):
        """after a main-model pass with the head: rows 0 .. n-1 (positions pos ..) are verified; nexts = the tokens
        that followed them"""
        D = self.D
        self.PH[:n * D] = self.s.tcb["XN"][:n * D]
        self.pend = (pos, n, [int(t) for t in nexts[:n]])
        if self.start is None:
            self.start = pos

    # ------------------------------------------------------------------ one call of the head
    def _gemv(self, name, x, y, T):
        W8, sc = self.w[name]
        R, C = W8.shape
        self.k.m_gemv8(((R + 31) // 32,), (256,), (W8, sc, x, y, i32(R), i32(C), i32(T)))

    def _call(self, hin, toks, p0):
        """rows (hin[t], toks[t]) at positions p0 + t: K/V into the window, HH = the head's output rows; returns
        the argmax of the last row"""
        s, D = self.s, self.D
        T = len(toks)
        emb, K = s.tp.emb, s.k
        for t, tok in enumerate(toks):
            K.k_deq_emb4((D // s.HB,), (512,), (np.uint64(emb["bc"]), np.uint64(emb["dp"]), s.sD, self.EB[t * D:],
                                                i32(tok), i32(D), i32(s.HB)))
        self._layer(hin, T, p0)
        self.src[:] = self.HH[(T - 1) * D:T * D]
        self.headg.launch()
        self.stats["calls"] += 1
        return int(self.aout.get()[0])

    def _layer(self, hin, T, p0):
        """EB (the embeddings) and hin -> HH (the head's output rows); K/V of rows p0 .. into the window"""
        k = self.k
        D, NH, NHKV, HD, NFF, W = self.D, self.NH, self.NHKV, self.HD, self.NFF, self.W
        n = self.n
        k.m_rms((T,), (256,), (self.EB, n["e_norm"], self.X2, i32(D), i32(D), i32(2 * D)))
        k.m_rms((T,), (256,), (hin, n["h_norm"], self.X2[D:], i32(D), i32(D), i32(2 * D)))
        self._gemv("fc", self.X2, self.U, T)
        k.m_rms((T,), (256,), (self.U, n["in_norm"], self.A, i32(D), i32(D), i32(D)))
        self._gemv("q", self.A, self.QG, T)
        self._gemv("k", self.A, self.KB, T)
        self._gemv("v", self.A, self.VB, T)
        k.m_rms((T * NH,), (256,), (self.QG, n["q_norm"], self.QG, i32(HD), i32(2 * HD), i32(2 * HD)))
        k.m_rms((T * NHKV,), (256,), (self.KB, n["k_norm"], self.KB, i32(HD), i32(HD), i32(HD)))
        rd = int(self.meta["rope_dim"])
        th = np.float64(self.meta["rope_theta"])
        k.m_rope((T * NH,), (rd // 2,), (self.QG, i32(2 * HD), i32(NH), i32(p0), i32(rd), th))
        k.m_rope((T * NHKV,), (rd // 2,), (self.KB, i32(HD), i32(NHKV), i32(p0), i32(rd), th))
        k.m_kvstore((T,), (256,), (self.KB, self.VB, self.Kr, self.Vr, self.spos, i32(p0), i32(W), i32(NHKV * HD)))
        k.m_attn((NH, T), (256,), (self.QG, self.Kr, self.Vr, self.spos, self.AO, i32(p0), i32(self.start), i32(W),
                                   i32(NH), i32(NHKV)))
        self._gemv("o", self.AO, self.Y, T)
        cp.add(self.U[:T * D], self.Y[:T * D], out=self.U[:T * D])
        k.m_rms((T,), (256,), (self.U, n["post_norm"], self.A, i32(D), i32(D), i32(D)))
        self._gemv("gate", self.A, self.G, T)
        self._gemv("up", self.A, self.UU, T)
        k.silumul(self.G[:T * NFF], self.UU[:T * NFF], self.F[:T * NFF])
        self._gemv("down", self.F, self.Y, T)
        cp.add(self.U[:T * D], self.Y[:T * D], out=self.U[:T * D])
        k.m_rms((T,), (256,), (self.U, n["out_norm"], self.HH, i32(D), i32(D), i32(D)))

    def draft(self, tok, pos, k):
        """k tokens after tok (at pos), or [] when the head has no verified row just before pos"""
        pd, self.pend = self.pend, None
        if pd is None or k < 1:
            return []
        p0, n, nx = pd
        if p0 + n != pos or nx[-1] != tok:
            self.start = None
            return []
        with cp.cuda.Device(self.s.card):
            d = [self._call(self.PH, nx, p0)]
            last = n - 1                                  # the row of HH the next step reads
            for i in range(1, k):
                self.HIN[:] = self.HH[last * self.D:(last + 1) * self.D]
                d.append(self._call(self.HIN, [d[-1]], pos + i - 1))
                last = 0
        self.stats["drafts"] += 1
        return d
