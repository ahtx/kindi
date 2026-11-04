"""Kindi tokenizer for SentencePiece (llama) vocabularies in GGUF files (Nanbeige 4.x), token for token equal to
llama.cpp: special tokens are split out first; each plain fragment gets a leading space when it is the first
fragment or follows a special token (add_space_prefix), spaces become U+2581, and the pieces are merged pair by pair,
highest score first (ties: the leftmost pair). A piece that is not in the vocabulary is written as byte tokens."""
import heapq

SP = "▁"


class SPM(object):
    def __init__(self, kv):
        self.tokens = kv["tokenizer.ggml.tokens"]
        self.scores = kv.get("tokenizer.ggml.scores") or [0.0] * len(self.tokens)
        self.types = kv.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)
        self.index = {}
        for i, t in enumerate(self.tokens):
            self.index.setdefault(t, i)
        self.bos = kv.get("tokenizer.ggml.bos_token_id")
        self.eos = kv.get("tokenizer.ggml.eos_token_id")
        self.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", True))
        self.space_prefix = bool(kv.get("tokenizer.ggml.add_space_prefix", True))
        # control (3) and user-defined (4) tokens are matched in the text, longest first
        self.special = sorted((t for i, t in enumerate(self.tokens) if self.types[i] in (3, 4) and t),
                              key=lambda t: -len(t))
        self.byte_tok = {}
        for b in range(256):
            i = self.index.get("<0x%02X>" % b)
            if i is not None:
                self.byte_tok[b] = i

    # ------------------------------------------------------------------ encode
    def _split_special(self, text):
        parts = [(False, text)]
        for st in self.special:
            out = []
            for is_tok, s in parts:
                if is_tok or st not in s:
                    out.append((is_tok, s))
                    continue
                pieces = s.split(st)
                for k, p in enumerate(pieces):
                    if p:
                        out.append((False, p))
                    if k + 1 < len(pieces):
                        out.append((True, st))
            parts = out
        return parts

    def _spm(self, text, out):
        """Append the ids of one fragment (spaces already U+2581) to out. The fragment starts as one span per
        character. end[a] is the end of the span that starts at a (-1 when a is inside a span) and start[b] is the
        start of the span that ends at b. The heap holds every adjacent span pair whose union is a vocabulary piece,
        as (-score, a, m, b) for the spans text[a:m] and text[m:b]; a pair is stale once either span has changed.
        Popping the best pair (ties: the smallest a) and joining it repeats until the heap is empty."""
        n = len(text)
        if not n:
            return
        index = self.index
        scores = self.scores
        end = list(range(1, n + 1))
        start = list(range(-1, n))
        heap = []
        for a in range(n - 1):
            i = index.get(text[a:a + 2])
            if i is not None:
                heap.append((-scores[i], a, a + 1, a + 2))
        heapq.heapify(heap)
        pop = heapq.heappop
        push = heapq.heappush
        while heap:
            _, a, m, b = pop(heap)
            if end[a] != m or end[m] != b:
                continue
            end[a] = b
            end[m] = -1
            start[b] = a
            if a:
                left = start[a]
                i = index.get(text[left:b])
                if i is not None:
                    push(heap, (-scores[i], left, a, b))
            if b < n:
                right = end[b]
                i = index.get(text[a:right])
                if i is not None:
                    push(heap, (-scores[i], a, b, right))
        a = 0
        byte_tok = self.byte_tok
        while a < n:
            b = end[a]
            i = index.get(text[a:b])
            if i is not None:
                out.append(i)
            else:
                out.extend(byte_tok[c] for c in text[a:b].encode("utf-8"))
            a = b

    def encode(self, text, add_bos=None):
        out = []
        if self.add_bos if add_bos is None else add_bos:
            out.append(self.bos)
        prev_special = True
        for is_tok, s in self._split_special(text):
            if is_tok:
                out.append(self.index[s])
                prev_special = True
            else:
                t = (" " if self.space_prefix and prev_special else "") + s
                self._spm(t.replace(" ", SP), out)
                prev_special = False
        return out

    # ------------------------------------------------------------------ decode
    def piece(self, tid):
        t = self.tokens[tid]
        ty = self.types[tid]
        if ty == 6 and t.startswith("<0x") and len(t) == 6:
            return bytes([int(t[3:5], 16)])
        if ty in (3, 5):                                       # control / unused: no text
            return b""
        return t.replace(SP, " ").encode("utf-8")

    def decode(self, ids, strip_first_space=True):
        b = b"".join(self.piece(t) for t in ids)
        s = b.decode("utf-8", errors="replace")
        if strip_first_space and self.space_prefix and s.startswith(" "):
            s = s[1:]
        return s
