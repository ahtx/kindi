"""Tokenizer for Nemotron-H MoE GGUF files (GGUF tokenizer model "gpt2", pre-tokenizer "pixtral"): byte-level BPE
with the merges of the model file.

Text is split on the control tokens first (token type 3: <|im_start|>, <|im_end|>, <SPECIAL_n>, ...), then each plain
fragment goes through the pre-tokenizer pattern of the model's tokenizer.json (pretoken(), regex alternation
semantics reproduced: no `regex` dependency). A piece that is itself in the vocabulary is one token (the BPE model
has ignore_merges); other pieces go through the merges."""
import re

from .tokenizer import _B2U, _cat, bpe, gpt2_bytes

_UPPER = ("Lu", "Lt", "Lm", "Lo")
_LOWER = ("Ll", "Lm", "Lo")
_FULL = {}


def _fcat(ch):
    c = _FULL.get(ch)
    if c is None:
        import unicodedata
        c = unicodedata.category(ch)
        _FULL[ch] = c
    return c


def _up(ch):                       # [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    c = _fcat(ch)
    return c in _UPPER or c[0] == "M"


def _lo(ch):                       # [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    c = _fcat(ch)
    return c in _LOWER or c[0] == "M"


def _pre(ch):                      # [^\r\n\p{L}\p{N}]
    return ch not in "\r\n" and _cat(ch) not in ("L", "N")


def _other(ch):                    # [^\s\p{L}\p{N}]
    return not ch.isspace() and _cat(ch) not in ("L", "N")


def _alt1(text, j):
    """[\\p{Lu}...]*[\\p{Ll}...]+ from j: the end of the match or -1 (the greedy upper run gives back characters
    until a lower-class character follows)"""
    n = len(text)
    u = j
    while u < n and _up(text[u]):
        u += 1
    for b in range(u, j - 1, -1):
        if b < n and _lo(text[b]):
            e = b + 1
            while e < n and _lo(text[e]):
                e += 1
            return e
    return -1


def _alt2(text, j):
    """[\\p{Lu}...]+[\\p{Ll}...]* from j: the end of the match or -1"""
    n = len(text)
    u = j
    while u < n and _up(text[u]):
        u += 1
    if u == j:
        return -1
    while u < n and _lo(text[u]):
        u += 1
    return u


def pretoken(text):
    """The pixtral pre-tokenizer pattern, first matching alternative at each position (greedy, with backtracking):
      1 [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+
      2 [^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*
      3 \\p{N}   4 ' '?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*   5 \\s*[\\r\\n]+   6 \\s+(?!\\S)   7 \\s+"""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        starts = (i + 1, i) if _pre(c) else (i,)
        e = -1
        for alt in (_alt1, _alt2):          # the optional prefix is tried first (greedy), then without it
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
        if _cat(c) == "N":
            out.append(c)
            i += 1
            continue
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
        if c.isspace():
            k = i
            while k < n and text[k].isspace():
                k += 1
            last = -1
            for q in range(i, k):
                if text[q] in "\r\n":
                    last = q
            if last >= 0:                    # 5: up to and including the last CR/LF of the run
                e = last + 1
            elif k == n:                     # 6: the whole run at the end of the text
                e = k
            elif k - 1 > i:                  # 6: the run minus its last character when a non-space follows
                e = k - 1
            else:                            # 7
                e = k
            out.append(text[i:e])
            i = e
            continue
        out.append(c)
        i += 1
    return out


class LightningTokenizer(object):
    def __init__(self, kv):
        self.vocab = kv["tokenizer.ggml.tokens"]
        self.types = kv["tokenizer.ggml.token_type"]
        self.sid = {s: i for i, s in enumerate(self.vocab)}
        self.ranks = {}
        for r, m in enumerate(kv["tokenizer.ggml.merges"]):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = r
        self.bos = int(kv.get("tokenizer.ggml.bos_token_id", 1))
        self.eos = int(kv.get("tokenizer.ggml.eos_token_id", 2))
        self.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
        self.control = [s for s, t in zip(self.vocab, self.types) if t == 3]
        alts = "|".join(re.escape(s) for s in sorted(self.control, key=len, reverse=True))
        self.split_re = re.compile("(" + alts + ")")
        self.u2b = {v: k for k, v in _B2U.items()}
        self.cache = {}

    def encode(self, text, special=True):
        out = [self.bos] if self.add_bos else []
        parts = self.split_re.split(text) if special else [text]
        for k, part in enumerate(parts):
            if not part:
                continue
            if k % 2:
                out.append(self.sid[part])
                continue
            for w in pretoken(part):
                ids = self.cache.get(w)
                if ids is None:
                    b = gpt2_bytes(w)
                    ids = [self.sid[b]] if b in self.sid else [self.sid.get(p, 0) for p in bpe(b, self.ranks)]
                    if len(self.cache) < 200000:
                        self.cache[w] = ids
                out.extend(ids)
        return out

    def decode_bytes(self, i):
        s = self.vocab[i]
        if self.types[i] == 3:
            return s.encode("utf-8")
        if all(ch in self.u2b for ch in s):
            return bytes(self.u2b[ch] for ch in s)
        return s.encode("utf-8")

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8", errors="replace")
