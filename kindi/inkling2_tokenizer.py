"""Tokenizer for Inkling-2-Mini GGUF files (tokenizer.ggml.model "gpt2", pre-tokenizer "inkling2" -> GPT-4o).

Text is split on the model's control tokens (token_type != 1, e.g. <|message_user|>, <|message_model|>,
<|content_thinking|>, <|content_text|>, <|end_message|>, <|content_model_end_sampling|>), then each plain
fragment goes through llama.cpp's LLAMA_VOCAB_PRE_TYPE_GPT4O regex alternation semantics (pure Python, no `regex`
dependency) and byte-level BPE using GGUF merges.
"""
import re
import unicodedata

from .tokenizer import _B2U, bpe, gpt2_bytes

# Unicode White_Space (\s in llama.cpp / Rust regex)
_WS = set("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
          "\u2028\u2029\u202f\u205f\u3000")
_CAT = {}


def _cat(ch):
    c = _CAT.get(ch)
    if c is None:
        c = unicodedata.category(ch)
        _CAT[ch] = c
    return c


def _is_lm(ch):
    return _cat(ch)[0] in ("L", "M")


def _up_like(ch):
    # ((?=[\p{L}\p{M}])([^a-z]))
    return _is_lm(ch) and not ("a" <= ch <= "z")


def _lo_like(ch):
    # ((?=[\p{L}\p{M}])([^A-Z]))
    return _is_lm(ch) and not ("A" <= ch <= "Z")


def _pre(ch):
    # [^\r\n\p{L}\p{N}]
    return ch not in "\r\n" and _cat(ch)[0] not in ("L", "N")


def _other(ch):
    # [^\s\p{L}\p{N}]
    return ch not in _WS and _cat(ch)[0] not in ("L", "N")


def _contraction(text, e):
    # (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
    n = len(text)
    if e >= n or text[e] != "'" or e + 1 >= n:
        return e
    c1 = text[e + 1].lower()
    if c1 in ("s", "t", "m", "d"):
        return e + 2
    if e + 2 < n:
        pair = c1 + text[e + 2].lower()
        if pair in ("re", "ve", "ll"):
            return e + 3
    return e


def _alt1(text, j):
    # ((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+ + contraction
    n = len(text)
    u = j
    while u < n and _up_like(text[u]):
        u += 1
    for b in range(u, j - 1, -1):
        if b < n and _lo_like(text[b]):
            e = b + 1
            while e < n and _lo_like(text[e]):
                e += 1
            return _contraction(text, e)
    return -1


def _alt2(text, j):
    # ((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))* + contraction
    n = len(text)
    u = j
    while u < n and _up_like(text[u]):
        u += 1
    if u == j:
        return -1
    while u < n and _lo_like(text[u]):
        u += 1
    return _contraction(text, u)


def pretoken(text):
    """GPT-4o / o200k pre-tokenizer matching llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT4O."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        starts = (i + 1, i) if _pre(c) else (i,)
        e = -1
        for alt in (_alt1, _alt2):
            for j in starts:
                e = alt(text, j)
                if e > 0:
                    break
            if e > 0:
                break
        if e > 0:
            out.append(text[i:e])
            i = e
            continue
        # \p{N}{1,3}
        if _cat(c)[0] == "N":
            k = i + 1
            while k < n and k - i < 3 and _cat(text[k])[0] == "N":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # ' '?[^\s\p{L}\p{N}]+[\r\n/]*
        j = i + 1 if (c == " " and i + 1 < n and _other(text[i + 1])) else i
        if j < n and _other(text[j]):
            k = j
            while k < n and _other(text[k]):
                k += 1
            while k < n and text[k] in "\r\n/":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # \s*[\r\n]+ | \s+(?!\S) | \s+
        if c in _WS:
            k = i
            while k < n and text[k] in _WS:
                k += 1
            last = -1
            for q in range(i, k):
                if text[q] in "\r\n":
                    last = q
            if last >= 0:
                e = last + 1
            elif k == n:
                e = k
            elif k - 1 > i:
                e = k - 1
            else:
                e = k
            out.append(text[i:e])
            i = e
            continue
        out.append(c)
        i += 1
    return out


class Inkling2Tokenizer(object):
    def __init__(self, kv):
        self.vocab = kv["tokenizer.ggml.tokens"]
        self.types = kv["tokenizer.ggml.token_type"]
        self.sid = {s: i for i, s in enumerate(self.vocab)}
        self.ranks = {}
        for r, m in enumerate(kv["tokenizer.ggml.merges"]):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = r
        self.bos = int(kv["tokenizer.ggml.bos_token_id"]) if "tokenizer.ggml.bos_token_id" in kv else None
        self.eos = int(kv.get("tokenizer.ggml.eos_token_id", 200006))
        self.eot = int(kv.get("tokenizer.ggml.eot_token_id", self.eos))
        self.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
        self.special = set(i for i, t in enumerate(self.types) if t != 1)
        sp = [self.vocab[i] for i in self.special]
        if sp:
            alts = "|".join(re.escape(s) for s in sorted(sp, key=len, reverse=True))
            self.split_re = re.compile("(" + alts + ")")
        else:
            self.split_re = None
        self.u2b = {v: k for k, v in _B2U.items()}
        self.cache = {}

    def encode(self, text, special=True):
        out = [self.bos] if (self.add_bos and self.bos is not None) else []
        parts = self.split_re.split(text) if (special and self.split_re is not None) else [text]
        for k, part in enumerate(parts):
            if not part:
                continue
            if special and self.split_re is not None and (k % 2):
                out.append(self.sid[part])
                continue
            for w in pretoken(part):
                ids = self.cache.get(w)
                if ids is None:
                    b = gpt2_bytes(w)
                    ids = [self.sid[p] for p in bpe(b, self.ranks)]
                    if len(self.cache) < 200000:
                        self.cache[w] = ids
                out.extend(ids)
        return out

    def decode_bytes(self, i):
        if i < 0 or i >= len(self.vocab):
            return b""
        s = self.vocab[i]
        if i in self.special:
            return s.encode("utf-8")
        if all(ch in self.u2b for ch in s):
            return bytes(self.u2b[ch] for ch in s)
        return s.encode("utf-8")

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8", errors="replace")
