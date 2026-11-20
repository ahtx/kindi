"""Kindi tokenizer for the Qwen3.5 vocabulary (byte-level BPE), token-for-token equal to llama.cpp.

Text is split on every added token first (<|im_start|>, <tool_call>, <think>, ...), then each plain fragment goes
through the Qwen3.5 pre-tokenizer pattern (pretoken(), regex alternation semantics reproduced exactly: no `regex`
dependency) and byte-level BPE with the merges from the model file."""
import re
import unicodedata

# GPT-2 bytes_to_unicode
_BS = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
_CS = _BS[:]
_n = 0
for _b in range(256):
    if _b not in _BS:
        _BS.append(_b)
        _CS.append(256 + _n)
        _n += 1
_B2U = dict(zip(_BS, [chr(c) for c in _CS]))


def gpt2_bytes(s):
    return "".join(_B2U[b] for b in s.encode("utf-8"))


def load_ranks(path):
    """merges.txt lines 'A B' -> {(A, B): rank} (rank = line number)."""
    ranks = {}
    with open(path, encoding="utf-8") as f:
        r = 0
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                ranks[(parts[0], parts[1])] = r
            r += 1
    return ranks


def bpe(word, ranks):
    """Merge the lowest-rank adjacent pair until none applies."""
    parts = list(word)
    while len(parts) > 1:
        best_r, best_k = None, -1
        for k in range(len(parts) - 1):
            r = ranks.get((parts[k], parts[k + 1]))
            if r is not None and (best_r is None or r < best_r):
                best_r, best_k = r, k
        if best_k < 0:
            break
        parts[best_k:best_k + 2] = [parts[best_k] + parts[best_k + 1]]
    return parts


_CAT = {}


def _cat(ch):
    c = _CAT.get(ch)
    if c is None:
        c = unicodedata.category(ch)[0]
        _CAT[ch] = c
    return c


def _is_l(ch):
    return _cat(ch) == "L"


def _is_lm(ch):
    return _cat(ch) in ("L", "M")


def _is_n(ch):
    return _cat(ch) == "N"


def pretoken(text):
    """The Qwen3.5 pre-tokenizer pattern with regex alternation semantics (first alternative that matches, greedy):
      1 (?i:'s|'t|'re|'ve|'m|'ll|'d)   2 [^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+   3 \\p{N}
      4 ' '?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*   5 \\s*[\\r\\n]+   6 \\s+(?!\\S)   7 \\s+
    Rule 6 matters: a whitespace run before a non-space keeps its last character for the next token, which is how
    llama.cpp splits "  Size" and code indentation."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # 1 contractions
        if c == "'" and i + 1 < n:
            t3 = text[i:i + 3].lower()
            if t3 in ("'re", "'ve", "'ll"):
                out.append(text[i:i + 3])
                i += 3
                continue
            if text[i:i + 2].lower() in ("'s", "'t", "'m", "'d"):
                out.append(text[i:i + 2])
                i += 2
                continue
        # 2 optional prefix + letter/mark run
        j = None
        if not (c in "\r\n" or _is_l(c) or _is_n(c)) and i + 1 < n and _is_lm(text[i + 1]):
            j = i + 1
        elif _is_lm(c):
            j = i
        if j is not None:
            k = j
            while k < n and _is_lm(text[k]):
                k += 1
            out.append(text[i:k])
            i = k
            continue
        # 3 one number
        if _is_n(c):
            out.append(c)
            i += 1
            continue
        # 4 optional space + run of "other" + trailing CR/LF

        def other(ch):
            return not (ch.isspace() or _is_lm(ch) or _is_n(ch))
        j = i + 1 if (c == " " and i + 1 < n and other(text[i + 1])) else i
        if j < n and other(text[j]):
            k = j
            while k < n and other(text[k]):
                k += 1
            while k < n and text[k] in "\r\n":
                k += 1
            out.append(text[i:k])
            i = k
            continue
        if c.isspace():
            k = i
            while k < n and text[k].isspace():
                k += 1
            # 5 \s*[\r\n]+ : up to and including the last CR/LF of the whitespace run
            last = -1
            for q in range(i, k):
                if text[q] in "\r\n":
                    last = q
            if last >= 0:
                out.append(text[i:last + 1])
                i = last + 1
                continue
            # 6 \s+(?!\S) : the run, minus its last char when a non-space follows
            if k == n:
                out.append(text[i:k])
                i = k
                continue
            if k - 1 > i:
                out.append(text[i:k - 1])
                i = k - 1
                continue
            # 7 \s+
            out.append(text[i:k])
            i = k
            continue
        out.append(c)
        i += 1
    return out




class Tok(object):
    def __init__(self, vocab, merges_path):
        self.vocab = vocab
        self.sid = {s: i for i, s in enumerate(vocab) if isinstance(s, str)}
        self.ranks = load_ranks(merges_path)
        self.u2b = {v: k for k, v in _B2U.items()}
        added = [s for s in (v[0] if isinstance(v, list) else v for v in vocab[248044:])
                 if isinstance(s, str) and s.startswith("<") and s.endswith(">")]
        self.added = added
        self.split_re = re.compile("(" + "|".join(re.escape(s) for s in sorted(added, key=len, reverse=True)) + ")")
        self.cache = {}

    def encode(self, text):
        out = []
        for k, part in enumerate(self.split_re.split(text)):
            if not part:
                continue
            if k % 2:
                out.append(self.sid[part])
                continue
            for w in pretoken(part):
                ids = self.cache.get(w)
                if ids is None:
                    pieces = bpe(gpt2_bytes(w), self.ranks)
                    try:
                        ids = [self.sid[p] for p in pieces]
                    except KeyError as e:
                        raise ValueError("tokenizer vocabulary has no token for %r; the pack's vocabulary and "
                                         "merges are inconsistent" % e.args[0])
                    if len(self.cache) < 200000:
                        self.cache[w] = ids
                out.extend(ids)
        return out

    def decode_bytes(self, i):
        item = self.vocab[i]
        s = item[0] if isinstance(item, list) else str(item)
        if all(ch in self.u2b for ch in s):
            return bytes(self.u2b[ch] for ch in s)
        return s.encode("utf-8")

    def decode(self, ids):
        return b"".join(self.decode_bytes(i) for i in ids).decode("utf-8", errors="replace")
