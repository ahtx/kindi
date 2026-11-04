"""kindi fast-head: build the cluster file the fast (shortlist) output head needs.

Spherical k-means of the 248,320 head rows (ternary x fp16 scales, in the folded input space the head GEMV uses):
rows are dequantised on the GPU chunk by chunk, normalised, assigned to the nearest unit centroid by cosine and the
centroids re-estimated with scatter-adds. Writes <pack>/head_clusters.npz (assign [R], centroids [K, D]).
About 2 minutes on a K420; needs ~300 MB of GPU memory (stop `kindi serve` first)."""
import json
import os
import time

import numpy as np

DEQ_SRC = r'''
extern "C" __global__ void deq(const unsigned char* __restrict__ b, const unsigned short* __restrict__ s,
                               float* __restrict__ out, int n, int C){
  const int r = blockIdx.x;
  /* one 4-column slot per iteration; the block has DEQ_THREADS threads, so C/4 may exceed 1,024 (the 27B: 1,280) */
  for (int t = threadIdx.x; t < (C >> 2); t += blockDim.x) {
    const unsigned int v = b[(long long)r * (C >> 2) + t];
    float sc; asm("cvt.f32.f16 %0, %1;" : "=f"(sc) : "h"(s[(long long)r * (C >> 7) + (t >> 5)]));
    float* o = out + (long long)r * C + 4 * t;
    o[0] = (float)((int)(v % 3u) - 1) * sc;
    o[1] = (float)((int)((v / 3u) % 3u) - 1) * sc;
    o[2] = (float)((int)((v / 9u) % 3u) - 1) * sc;
    o[3] = (float)((int)((v / 27u) % 3u) - 1) * sc;
  }
}'''
DEQ_THREADS = 256                                  # threads per block of the deq kernel (at most 1,024)


def build(pack, K=1024, iters=12, log=print):
    import cupy as cp
    import cupyx
    head = json.load(open(os.path.join(pack, "kindi_pack.json"))).get("head", "token_embd.weight")
    rec = json.load(open(os.path.join(pack, "bcodes_index.json")))["tensors"][head]
    R, C = rec["rows"], rec["ne0"]
    ns, P = C // 4, C // 128
    bc = np.fromfile(os.path.join(pack, "bcodes.bin"), dtype=np.uint8, count=ns * R,
                     offset=rec["bc_off"]).reshape(ns, R)
    dp = np.fromfile(os.path.join(pack, "dpm.bin"), dtype=np.uint16, count=P * R,
                     offset=rec["dpm_off"]).reshape(P, R)
    deq = cp.RawKernel(DEQ_SRC, "deq")
    CH = 16384

    def chunk(i0):
        n = min(CH, R - i0)
        out = cp.empty((n, C), dtype=cp.float32)
        deq((n,), (min(DEQ_THREADS, C // 4),), (cp.asarray(np.ascontiguousarray(bc[:, i0:i0 + n].T)),
                              cp.asarray(np.ascontiguousarray(dp[:, i0:i0 + n].T)), out, np.int32(n), np.int32(C)))
        out /= cp.maximum(cp.sqrt((out * out).sum(axis=1, keepdims=True)), 1e-12)
        return out

    t0 = time.time()
    rng = np.random.RandomState(0)
    seed = np.sort(rng.choice(R, K, replace=False))
    cent = cp.empty((K, C), dtype=cp.float32)
    for i0 in range(0, R, CH):
        sel = seed[(seed >= i0) & (seed < i0 + CH)] - i0
        if len(sel):
            j = np.searchsorted(seed, i0)
            cent[j:j + len(sel)] = chunk(i0)[cp.asarray(sel)]
    assign = np.zeros(R, dtype=np.int32)
    for it in range(iters):
        csum = cp.zeros((K, C), dtype=cp.float32)
        ccnt = cp.zeros(K, dtype=cp.float32)
        moved = 0
        for i0 in range(0, R, CH):
            X = chunk(i0)
            a = cp.argmax(X @ cent.T, axis=1).astype(cp.int32)
            ah = cp.asnumpy(a)
            moved += int((ah != assign[i0:i0 + len(ah)]).sum())
            assign[i0:i0 + len(ah)] = ah
            cupyx.scatter_add(csum, a, X)
            cupyx.scatter_add(ccnt, a, cp.ones(len(ah), dtype=cp.float32))
        cent = cp.where(ccnt[:, None] > 0, csum / cp.maximum(ccnt, 1)[:, None], cent)
        cent /= cp.maximum(cp.sqrt((cent * cent).sum(axis=1, keepdims=True)), 1e-12)
        log("  iteration %d/%d: %d rows moved" % (it + 1, iters, moved))
        if it > 2 and moved < R // 2000:
            break
    out = os.path.join(pack, "head_clusters.npz")
    np.savez(out, assign=assign, centroids=cp.asnumpy(cent))
    log("wrote %s (%d clusters, %.0fs)" % (out, K, time.time() - t0))
    return out
