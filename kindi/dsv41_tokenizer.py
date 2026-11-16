"""Kindi tokenizer for the DeepSeek-V4.1 vocabulary (byte-level BPE), from the GGUF vocabulary and merges.

Text is split on every special token first (GGUF token types other than "normal"), then each plain fragment goes
through the DeepSeek-V4.1 pre-tokenizer: digit runs cut into groups of up to 3, CJK runs isolated, then the main
pattern (regex alternation semantics reproduced here, no `regex` dependency):
  A [!"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+
  B [^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+
  C ' '?[\\p{P}\\p{S}]+[\\r\\n]*
  D \\s*[\\r\\n]+
  E \\s+(?!\\S)
  F \\s+
Characters that no alternative matches stay together as one piece (split behavior "isolated")."""
import re
import unicodedata

from .tokenizer import _B2U, bpe

_ASCII_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
# Unicode White_Space (the \s of the Rust regex engine)
_WS = set("\t\n\x0b\x0c\r \x85\xa0            "
          "    　")
_CAT = {}


def _cat(ch):
    c = _CAT.get(ch)
    if c is None:
        c = unicodedata.category(ch)
        _CAT[ch] = c
    return c


def _is_lm(ch):
    return _cat(ch)[0] in "LM"


def _is_ps(ch):
    return _cat(ch)[0] in "PS"


def _is_cjk(ch):
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FA5 or 0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF


def _split_isolated(pieces, match):
    """apply one "isolated" split: match(text, i) -> end of a match at i, or -1"""
    out = []
    for s in pieces:
        i, n, un = 0, len(s), 0
        while i < n:
            j = match(s, i)
            if j > i:
                if un < i:
                    out.append(s[un:i])
                out.append(s[i:j])
                i = un = j
            else:
                i += 1
        if un < n:
            out.append(s[un:])
    return out


def _m_digits(s, i):
    j = i
    while j < len(s) and j - i < 3 and _cat(s[j])[0] == "N":
        j += 1
    return j


def _m_cjk(s, i):
    j = i
    while j < len(s) and _is_cjk(s[j]):
        j += 1
    return j


def _m_main(s, i):
    n, c = len(s), s[i]
    # A
    if c in _ASCII_PUNCT and i + 1 < n and s[i + 1].isascii() and s[i + 1].isalpha():
        j = i + 1
        while j < n and s[j].isascii() and s[j].isalpha():
            j += 1
        return j
    # B
    j = i
    if c not in "\r\n" and not _is_lm(c) and _cat(c)[0] not in "LPS" and i + 1 < n and _is_lm(s[i + 1]):
        j = i + 1
    if _is_lm(s[j]):
        while j < n and _is_lm(s[j]):
            j += 1
        return j
    # C
    j = i + 1 if (c == " " and i + 1 < n and _is_ps(s[i + 1])) else i
    if j < n and _is_ps(s[j]):
        while j < n and _is_ps(s[j]):
            j += 1
        while j < n and s[j] in "\r\n":
            j += 1
        return j
    if c in _WS:
        k = i
        while k < n and s[k] in _WS:
            k += 1
        # D: through the last CR/LF of the whitespace run
        last = -1
        for q in range(i, k):
            if s[q] in "\r\n":
                last = q
        if last >= 0:
            return last + 1
        # E: the run, minus its last character when a non-space follows
        if k == n:
            return k
        if k - 1 > i:
            return k - 1
        # F
        return k
    return -1


def pretokens(text):
    p = _split_isolated([text], _m_digits)
    p = _split_isolated(p, _m_cjk)
    return _split_isolated(p, _m_main)


def gpt2_bytes(s):
    return "".join(_B2U[b] for b in s.encode("utf-8"))


class Tok(object):
    def __init__(self, tokens, types, merges):
        self.vocab = tokens
        self.sid = {s: i for i, s in enumerate(tokens)}
        self.ranks = {}
        for r, m in enumerate(merges):
            a, _, b = m.partition(" ")
            self.ranks[(a, b)] = r
        self.u2b = {v: k for k, v in _B2U.items()}
        self.special = set(i for i, t in enumerate(types) if t != 1)
        sp = sorted((tokens[i] for i in self.special), key=len, reverse=True)
        self.split_re = re.compile("(" + "|".join(re.escape(s) for s in sp) + ")")
        self.cache = {}

    def encode(self, text, special=True):
        out = []
        parts = self.split_re.split(text) if special else [text]
        for k, part in enumerate(parts):
            if not part:
                continue
            if k % 2:
                out.append(self.sid[part])
                continue
            for w in pretokens(part):
                ids = self.cache.get(w)
                if ids is None:
                    ids = [self.sid[p] for p in bpe(gpt2_bytes(w), self.ranks)]
                    if len(self.cache) < 200000:
                        self.cache[w] = ids
                out.extend(ids)
        return out

    def decode_bytes(self, i):
        s = self.vocab[i]
        if i not in self.special and all(ch in self.u2b for ch in s):
            return bytes(self.u2b[ch] for ch in s)
        return s.encode("utf-8")

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8", errors="replace")
