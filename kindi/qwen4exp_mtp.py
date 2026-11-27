"""MTP drafting for Flash-Next on the card engine (kindi.qwen4exp_gpu.Model with mtp=...): the head proposes k tokens,
the main model checks them in one pass of k + 1 tokens, and the longest agreeing prefix is kept (greedy: the output is
the main model's own greedy output).

This first version runs on the plain (non-graph) path and keeps exactness simple: before each check the main state
is saved, and when a draft is refused the state is restored and the accepted tokens are run again. It measures the
acceptance of the head; the fast path (the graph decoder, per-token Gated DeltaNet states instead of the replay)
comes after.

The head's own attention cache holds one row per position of the sequence: row p is made from the main model's
streams at p and the token at p + 1. After each check, the accepted positions are added from the main model's
streams (the drafting rows, made from the head's own streams, are dropped)."""
import os
import time

import torch


def _clone(st):
    out = {}
    for k, v in st.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.clone()
        elif hasattr(v, "copy"):
            out[k] = v.copy()
        else:
            out[k] = v
    return out


def _trim_attn(st, keep):
    """keep the first `keep` positions of an attention-only state (the head's)"""
    for d in ("k", "v", "ik"):
        for l in list(st[d]):
            st[d][l] = st[d][l][:keep]
    st["pos"] = keep


class Drafter(object):
    def __init__(self, gm, k=3, offset=0):
        """gm: kindi.qwen4exp_gpu.Model built with mtp=...; k: drafts per step; offset: the head's position for row p
        is p + offset (0 or 1; chosen by acceptance)"""
        if not gm.mtp:
            raise ValueError("the model was loaded without the MTP head")
        self.g, self.k, self.offset = gm, k, offset

    def _main(self, ids, st, last_n=None):
        """the main model on ids (continuing st) -> (logits [T, V], or of the last last_n rows; streams after the
        last layer [T, 4H])"""
        box = {}

        def tap(l, h):
            if l == self.g.L:
                box["h"] = h
        lg = self.g.logits(ids, state=st, tap=tap, last_n=last_n)
        return lg, box["h"]

    def generate(self, ids, n, log=None):
        """greedy, n new tokens -> (tokens, stats)"""
        g, k = self.g, self.k
        ids = list(ids)
        st = g.new_state()
        t0 = time.time()
        lg, hs = self._main(ids, st)
        nxt = int(lg[-1].argmax())
        # the head's cache over the prompt: row p from the streams at p and the token at p + 1
        hst = g.new_state()
        hst["pos"] = self.offset
        g.mtp_logits(hs[:-1], ids[1:], hst) if len(ids) > 1 else None
        h_last, out = hs[-1:], [nxt]
        stats = {"passes": 0, "drafted": 0, "accepted": 0, "prompt_s": time.time() - t0}
        t1 = time.time()
        while len(out) < n:
            # drafts: (streams at p, token at p + 1) -> token at p + 2, then on the head's own streams
            base = hst["pos"]
            drafts, hc, tk = [], h_last, nxt
            for _ in range(k):
                dl, hc = g.mtp_logits(hc, [tk], hst)
                tk = int(dl[-1].argmax())
                drafts.append(tk)
            _trim_attn(hst, base + 1)                     # the first row (true streams, true token) stays
            # check nxt + drafts in one pass
            saved = _clone(st)
            seq = [nxt] + drafts
            lg, hs = self._main(seq, st)
            pred = lg.argmax(-1).tolist()                # pred[i]: the main model's token after seq[: i + 1]
            j = 0
            while j < k and pred[j] == drafts[j]:
                j += 1
            stats["passes"] += 1
            stats["drafted"] += k
            stats["accepted"] += j
            new = drafts[:j] + [pred[j]]                  # accepted drafts, then the main model's own next token
            if j < k:                                    # refused at draft j: back to the saved state, run nxt + accepted
                st = saved
                lg2, hs = self._main(seq[:j + 1], st)
            hs = hs[:j + 1]
            # the head's cache: the accepted drafts' places from the main model's streams (the last place's row is
            # the next pass's first draft step)
            if j:
                g.mtp_logits(hs[:j], seq[1:j + 1], hst)
            h_last = hs[-1:]
            out += new
            nxt = new[-1]
            if log:
                log("pass %d: %d of %d drafts accepted" % (stats["passes"], j, k))
        stats["decode_s"] = time.time() - t1
        stats["tokens"] = len(out)
        return out[:n], stats


class GraphDrafter(Drafter):
    """the drafter on the fast path: the head drafts k tokens (on the card), the main model checks the k + 1 tokens
    in one multi-token window graph (GraphDecoder.window: each token with the bits of a one-token step), and the
    window is cut back to the accepted tokens (GraphDecoder.keep). The output is the graph decoder's own greedy
    output, token for token. Needs the mailbox pass (KINDI_QXG_PASS=1: one card, in-process CPU experts).

    A draft whose probability is under p_min ends the chain (the first draft always goes). With the draft
    vocabulary loaded (Model._load_draft_vocab), the head's cache rows of the accepted places and the k drafts run
    as one graph on the card (GraphDecoder.draft_chain); KINDI_MTP_CHAIN=0 keeps the step-by-step head."""
    p_min = float(os.environ.get("KINDI_MTP_PMIN", "0.5"))
    # when drafting does not pay, a pass checks no drafts (a window of one token: the plain step's bits):
    # KINDI_MTP_PFIRST: the first draft goes only at this probability or more (0: always, the old rule);
    # KINDI_MTP_AMIN: when the running acceptance (accepted / drafted, decay 0.8 a pass) falls under this, the next
    # KINDI_MTP_BACKOFF passes (8) skip the head's drafts entirely (its cache rows are still written), then it tries
    # again (0: off)
    p_first = float(os.environ.get("KINDI_MTP_PFIRST", "0"))
    a_min = float(os.environ.get("KINDI_MTP_AMIN", "0"))
    backoff = int(os.environ.get("KINDI_MTP_BACKOFF", "8"))
    # KINDI_MTP_AUTO=1: drafting where it is measured to pay. The drafter keeps, per mode, the seconds
    # and the new tokens of its last passes (drafted: the last 32; plain, a window of one token with the plain step's
    # bits: the last 8) and compares seconds per token over those sums (not per pass: one refused draft makes a
    # pass look slow). Drafting, it measures 4 plain passes every KINDI_MTP_PROBE (64) passes and switches to plain
    # only if plain is at least KINDI_MTP_HYST (10%) faster; plain, it measures 4 drafted passes every 16 and goes
    # back as soon as drafting is faster. B26 (3060): drafted k=2 wins on chat and lost after a 4K prompt; B32: the
    # per-pass average of the first version fell back on chat (20.5 against 24.3 always drafting). The output is the
    # same either way. 0: always draft
    # B38 (3060, unseen chats, rc3 3505570): with the CUDA-core window kernels drafting k=2 also wins after a 4K prompt
    # (19.0 drafting always, 17.9 with AUTO), and on chat AUTO and always drafting are within noise (23.9 / 23.0):
    # so AUTO is off by default (KINDI_MTP_AUTO=1 turns it on)
    auto = os.environ.get("KINDI_MTP_AUTO", "0") == "1"
    probe = int(os.environ.get("KINDI_MTP_PROBE", "64"))
    probe_len = int(os.environ.get("KINDI_MTP_PROBE_LEN", "4"))
    hyst = float(os.environ.get("KINDI_MTP_HYST", "0.10"))

    def _prompt(self, ids, chunk, cache, retry=False):
        """Extend both prompt states at a complete chunk boundary."""
        g = self.g
        saved = cache.pop("prompt", None) if cache is not None else None
        carry, start = None, 0
        if saved is not None:
            prefix, st, hst, carry = saved
            if len(prefix) < len(ids) and ids[:len(prefix)] == prefix:
                start = len(prefix)
        else:
            st = hst = None
        if not start:
            saved, carry = None, None
            st, hst = g.new_state(), g.new_state()
            hst["pos"] = self.offset
        cur = {"carry": carry, "off": start}

        def head_rows(c, hs):
            """the head's cache rows of chunk c (tokens) from its streams hs after the last layer"""
            carry = cur["carry"]
            rows = hs[:-1] if carry is None else torch.cat([carry, hs[:-1]])
            toks = c[1:] if carry is None else c
            if len(toks):
                g.mtp_kv(rows, toks, hst)                 # row p of the head's cache: streams at p, token at p + 1
            cur["carry"] = hs[-1:].clone()
        # a long prompt goes through the layer-major path when the model has it and the chunks are the model's own:
        # its logits and streams have the chunked path's bits, chunk by chunk, and the head's rows are made from each
        # chunk's streams as the chunked loop below would (rc5b: the drafted reads were chunked, layer-major unused)
        def read(end):
            pos = cur["off"]
            pc = getattr(g, "prompt_chunk", 0)
            if (getattr(g, "layer_major", False) is True and end - pos > chunk and chunk == pc and len(g.devices) == 1
                    and os.environ.get("KINDI_MTP_LM_PROMPT", "1") == "1"):
                def tap(l, h):
                    if l == g.L:
                        n_ = h.shape[0]
                        head_rows(ids[cur["off"]:cur["off"] + n_], h)
                        cur["off"] += n_
                if retry:
                    return g.logits(ids[pos:end], state=st, tap=tap, last_n=1, draft_prefill_retry=True)
                return g.logits(ids[pos:end], state=st, tap=tap, last_n=1)
            for i in range(pos, end, chunk):
                c = ids[i:min(i + chunk, end)]
                lg, hs = self._main(c, st, last_n=1)
                head_rows(c, hs)
                cur["off"] += len(c)
            return lg

        boundary = len(ids) // chunk * chunk if cache is not None else 0
        checkpoint = None
        if 0 < boundary < len(ids):
            if boundary > start:
                read(boundary)
            # Recompute the short final chunk on the next turn. Its shape can
            # change when the reply and next user message extend this prompt.
            checkpoint = (ids[:boundary], _clone(st), _clone(hst), cur["carry"])
        lg = read(len(ids))
        if boundary == len(ids):
            checkpoint = (ids, st, hst, cur["carry"])
        return lg, st, hst, cur["carry"], start, checkpoint

    def _passes(self, ids, n, log=None, chunk=None, stats=None, sampler=None, prompt_cache=None):
        """the drafted decode as a generator: yields the first token as [tok], then each pass's new tokens.
        sampler: None for greedy (the main model's argmax); else an object with first(logits, ids) -> the first token
        and accept(window logits, drafts, ids, out) -> the pass's new tokens by speculative sampling
        (kindi.sampling.spec_accept: the kept drafts, then one drawn token), so the output is distributed as plain
        sampling. The drafts are the head's argmax either way."""
        g, k = self.g, self.k
        chunk = chunk or getattr(g, "prompt_chunk", 256)
        ids = list(ids)
        t0 = time.time()
        retried = False
        try:
            prompt = self._prompt(ids, chunk, prompt_cache)
        except torch.OutOfMemoryError:
            from kindi.qwen4exp_gpu import env
            if env("KINDI_MTP_STAGE_PROMPT") != "1":
                raise
            retried = True
        if retried:
            # The failed read may have extended both states. Rebuild them from the prompt using smaller batches.
            torch.cuda.empty_cache()
            prompt = self._prompt(ids, chunk, prompt_cache, retry=True)
        lg, st, hst, carry, reused, checkpoint = prompt
        nxt = int(lg[-1].argmax()) if sampler is None else int(sampler.first(lg[-1], ids))
        torch.cuda.synchronize()
        dec = g.decoder()
        dec.win_max = k + 1                           # window buffers for the longest chain
        dec.load(st)
        dec.load_head(hst)
        dec.step_log()                                # the head's cache into its one-token graphs
        # a window of 1 when passes may check no drafts
        lo = 0 if (self.p_first > 0 or self.a_min > 0 or self.auto) else 1
        for T in range(k + 1, lo, -1):                # every window size captured now, not in the middle of a reply
            if T not in (getattr(dec, "_wins", None) or {}):
                dec._capture_window(T)
        if checkpoint is not None:
            # The graph decoder copied both states into its own buffers. Decode cannot modify this prefix.
            prompt_cache["prompt"] = checkpoint
        del st, hst, prompt
        h_last, out = carry, [nxt]
        stats = {} if stats is None else stats
        stats.update({"passes": 0, "drafted": 0, "accepted": 0, "prompt_s": time.time() - t0, "head_s": 0.0,
                 "window_s": 0.0, "judge_s": 0.0, "reused": reused, "prefill_retried": retried})
        yield [nxt]
        dec.t_prep = dec.t_run = 0.0
        dec.t_cache = 0.0                             # the per-pass split (drafter log): head, window, cache, judge
        chain = g.draft_map is not None and os.environ.get("KINDI_MTP_CHAIN", "1") != "0"
        stats["chain"] = chain
        start, upd = dec.hp, (None, [])               # the chain's start and the accepted places it writes first
        t1 = time.time()
        acc_avg, skip_left = 1.0, 0
        stats["skipped"] = 0
        # KINDI_MTP_AUTO: seconds per new token, drafted and plain (None: not measured yet); the current mode and
        # how many passes of it are left before the other is measured again
        from collections import deque
        hist = {"draft": deque(maxlen=32), "plain": deque(maxlen=8)}     # (seconds, new tokens) of recent passes

        def rate(kind):                               # seconds per new token over the recent passes (None: unknown)
            h = hist[kind]
            return sum(a for a, _ in h) / sum(b for _, b in h) if h else None
        mode, left = "draft", self.probe
        stats["auto_plain"] = 0
        while len(out) < n:
            th = time.time()
            plain = False
            if self.auto:
                probing = False
                if left <= 0:                         # measure the other mode probe_len passes, keep the faster
                    other = "plain" if mode == "draft" else "draft"
                    if left > -self.probe_len:
                        plain, probing = other == "plain", True
                        left -= 1
                    else:
                        ro, rm = rate(other), rate(mode)
                        need = 1.0 - (self.hyst if other == "plain" else 0.0)
                        if ro is not None and rm is not None and ro < rm * need:
                            mode = other
                        left = self.probe if mode == "draft" else 16
                if not probing:
                    plain = mode == "plain"
                    left -= 1
                stats["auto_plain"] += plain
            if skip_left > 0 or plain:                # backoff or auto-plain: no drafts; the head's rows step by step
                skip_left = max(0, skip_left - 1)
                stats["skipped"] += 1
                dec.hp = start
                for r, tk in enumerate(upd[1]):
                    dec.head_step(upd[0][r:r + 1], tk, logits=False)
                base = dec.hp
                dec.head_step(h_last, nxt, logits=False)  # row base: the true streams and token
                drafts = []
            elif chain:
                toks, prs = dec.draft_chain(start, upd[0], upd[1], h_last, nxt, k)
                base = start + len(upd[1])            # the first draft step's row (true streams and token)
                drafts = toks[:1] if prs[0] >= self.p_first else []
                for tk, pr in zip(toks[1:], prs[1:]):
                    if not drafts:
                        break
                    if pr < self.p_min:
                        break
                    drafts.append(tk)
            else:
                dec.hp = start
                for r, tk in enumerate(upd[1]):           # the accepted places' rows, step by step
                    dec.head_step(upd[0][r:r + 1], tk, logits=False)
                base = dec.hp
                drafts, hc, tk = [], h_last, nxt
                for _ in range(k):
                    dl, hc = dec.head_step(hc, tk)
                    pr, ti = torch.softmax(dl[-1].float(), -1).max(-1)
                    if float(pr) < (self.p_min if drafts else self.p_first):   # the first: KINDI_MTP_PFIRST
                        break
                    tk = int(ti)
                    drafts.append(tk)
            stats["head_s"] += time.time() - th
            k_now = len(drafts)
            seq = [nxt] + drafts
            tw = time.time()
            lgT, hT = dec.window(seq)
            tj = time.time()
            stats["window_s"] += tj - tw
            if sampler is None:
                pred = lgT.argmax(-1).tolist()
                j = 0
                while j < k_now and pred[j] == drafts[j]:
                    j += 1
                pred = pred[:j + 1]
            else:                                     # speculative sampling: kept drafts, then one drawn token
                pred = sampler.accept(lgT, drafts, ids, out)
                j = len(pred) - 1
            stats["judge_s"] += time.time() - tj
            dec.keep(j + 1)
            stats["passes"] += 1
            stats["drafted"] += k_now
            stats["accepted"] += j
            if k_now and self.a_min > 0:
                acc_avg = 0.8 * acc_avg + 0.2 * (j / k_now)
                if acc_avg < self.a_min:
                    skip_left, acc_avg = self.backoff, self.a_min   # after the backoff: one more try at the floor
            new = drafts[:j] + [pred[j]]
            # the head's cache: rows base + 1 .. base + j from the main model's streams at the accepted drafts' places,
            # written first thing next pass (row base, from the true streams and token, was written by the first
            # draft step; the row of the last accepted place by the next pass's first draft step); the drafting rows
            # after are written again
            start, upd = base + 1, (hT[:j].clone(), seq[1:j + 1])
            h_last = hT[j:j + 1].clone()
            out += new
            if self.auto:                             # this pass's seconds and new tokens, into its mode's history
                hist["draft" if k_now else "plain"].append((time.time() - th, len(new)))
            t_y = time.time()
            yield new
            t1 += time.time() - t_y                   # the caller's time is not decode time
            nxt = new[-1]
            if log:
                log("pass %d: %d of %d drafts accepted" % (stats["passes"], j, k))
        stats["decode_s"] = time.time() - t1
        stats["cache_s"] = dec.t_cache                # inside window_s
        stats["prep_s"], stats["run_s"] = dec.t_prep, dec.t_run   # inside window_s: before the replay, replay + serve
        stats["steps"] = dec.step_summary()
        stats["tokens"] = len(out)
        return

    def generate(self, ids, n, log=None, chunk=None):
        stats, out = {}, []
        for new in self._passes(ids, n, log=log, chunk=chunk, stats=stats):
            out += new
        return out[:n], stats

    def stream(self, ids, n, cancel=None, stop=(), chunk=None, stats=None, sampler=None, prompt_cache=None):
        """token by token (greedy, the same tokens as generate; or sampled, see _passes): yields each accepted token;
        ends after n tokens, after a token in stop (yielded), or when cancel() is true (checked between passes; a
        pass in flight finishes first). stats: a dict filled as generate's."""
        k = 0
        stop = set(stop)
        gen = self._passes(ids, n, chunk=chunk, stats=stats, sampler=sampler, prompt_cache=prompt_cache)
        try:
            for new in gen:
                for t in new:
                    yield t
                    k += 1
                    if k >= n or t in stop:
                        return
                if cancel is not None and cancel():
                    return
        finally:
            gen.close()

