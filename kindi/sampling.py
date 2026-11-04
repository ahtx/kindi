"""Sampling with the logits left on the GPU.

Only candidate tokens are copied to the host: for top-k sampling, a single-block kernel binary-searches the largest
threshold that keeps at least k tokens (k_topk); without top-k, every token within 30 * temperature of the best
(k_band - anything further has relative probability < e^-30). The candidates then go through the usual
temperature / top-k / top-p steps on the host. The chosen token is the same as sampling from the full logits.
With a repeat penalty (llama.cpp's repeat_penalty over the last repeat_last_n tokens), the full logits go to the
host, because the penalty can move the best token and the band."""
import os

import numpy as np
try:
    import cupy as cp
except ImportError:               # the CPU engine
    cp = None

BAND_T = 30.0
CAP = 8192
SAMPLER_ORDER = os.environ.get("KINDI_SAMPLER_ORDER", "llama").lower()     # llama | hf


class Sampler(object):
    def __init__(self, eng):
        self.eng = eng
        self.gpu = cp is not None and hasattr(eng, "k") and hasattr(eng, "aout")
        if not self.gpu:                 # the CPU engine: the logits are a NumPy array
            return
        self.ci = cp.zeros(CAP, dtype=cp.int32)
        self.cv = cp.zeros(CAP, dtype=cp.float32)
        self.cn = cp.zeros(1, dtype=cp.int32)

    def __call__(self, greedy_tok, temperature, top_p, top_k, rng, min_p=0.0, penalty=None):
        """penalty: None, or (repeat_penalty, the recent tokens)"""
        if penalty is not None and penalty[0] > 1.0 and temperature <= 0 and greedy_tok not in penalty[1]:
            return greedy_tok                # the penalty only lowers the recent tokens: the best token stays
        if penalty is not None:
            return self._penalized(temperature, top_p, top_k, rng, min_p, *penalty)
        if temperature <= 0:
            return greedy_tok
        e = self.eng
        yl, nl, ymap = e.ylog, int(e.ylog_n), e.ylog_map
        kk = int(top_k or 0)
        if not self.gpu:
            return self._pick(np.asarray(ymap[:nl]).astype(np.int64) if ymap is not None else np.arange(nl),
                              np.asarray(yl[:nl]).astype(np.float64), temperature, top_p, kk, rng, min_p)
        self.cn.fill(0)
        band = np.float32(BAND_T * temperature)
        if 0 < kk <= 1024:
            e.k.k_topk((1,), (1024,), (yl, e.aout, np.int32(nl), np.int32(kk), band, self.ci, self.cv, self.cn,
                                       np.int32(CAP)))
        else:
            e.k.k_band((32,), (256,), (yl, e.aout, np.int32(nl), band, self.ci, self.cv, self.cn, np.int32(CAP)))
        nc = int(self.cn.get()[0])
        if 0 < nc <= CAP:
            ci = self.ci[:nc]
            cand = cp.asnumpy(ymap[ci] if ymap is not None else ci).astype(np.int64)
            z = cp.asnumpy(self.cv[:nc]).astype(np.float64)
            o = np.argsort(cand)                         # atomics make the order arbitrary
            cand, z = cand[o], z[o]
        else:
            cand = cp.asnumpy(ymap[:nl]).astype(np.int64) if ymap is not None else np.arange(nl)
            z = cp.asnumpy(yl[:nl]).astype(np.float64)
            ok = cand >= 0
            cand, z = cand[ok], z[ok]
        return self._pick(cand, z, temperature, top_p, kk, rng, min_p)

    def _logits(self):
        """every candidate token and its logit, on the host"""
        e = self.eng
        yl, nl, ymap = e.ylog, int(e.ylog_n), e.ylog_map
        host = cp.asnumpy if self.gpu else np.asarray
        cand = host(ymap[:nl]).astype(np.int64) if ymap is not None else np.arange(nl)
        z = host(yl[:nl]).astype(np.float64)
        ok = cand >= 0
        return cand[ok], z[ok]

    def _penalized(self, temperature, top_p, top_k, rng, min_p, rp, recent):
        cand, z = self._logits()
        z = penalize(cand, z, rp, recent)
        if temperature <= 0:
            return int(cand[int(np.argmax(z))])
        band = z >= z.max() - BAND_T * temperature        # as k_band: the rest has probability < e^-30
        return self._pick(cand[band], z[band], temperature, top_p, int(top_k or 0), rng, min_p)

    @staticmethod
    def _pick(cand, z, temperature, top_p, kk, rng, min_p=0.0):
        """top-k, then top-p, then the temperature: the order of llama.cpp's default sampler chain, where top-p
        sees the probabilities at temperature 1. KINDI_SAMPLER_ORDER=hf applies the temperature first (the
        Hugging Face and vLLM order); at temperature < 1 that keeps fewer tokens and samples more greedily."""
        k = max(1, min(kk or len(cand), len(cand)))
        sel = np.argpartition(-z, k - 1)[:k] if k < len(cand) else np.arange(len(cand))
        idx, zs = cand[sel], z[sel]
        order = np.argsort(-zs)
        idx, zs = idx[order], zs[order]
        if top_p and top_p < 1.0:
            q = zs / temperature if SAMPLER_ORDER == "hf" else zs
            q = np.exp(q - q.max())
            q /= q.sum()
            keep = (np.cumsum(q) - q) < top_p
            idx, zs = idx[keep], zs[keep]
        if min_p and min_p > 0.0:
            # min-p after top-p, on the probabilities at temperature 1 (llama.cpp's order); the best token stays
            keep = zs >= zs.max() + np.log(min_p)
            idx, zs = idx[keep], zs[keep]
        p = np.exp((zs - zs.max()) / temperature)
        p /= p.sum()
        return int(rng.choice(idx, p=p))


def penalize(cand, z, rp, recent):
    """llama.cpp's repeat penalty: the logit of each recent token is divided by rp when it is positive and
    multiplied by rp when it is negative"""
    hit = np.isin(cand, np.asarray(list(recent), dtype=np.int64))
    z = z.copy()
    z[hit] = np.where(z[hit] > 0, z[hit] / rp, z[hit] * rp)
    return z
