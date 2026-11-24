# SPDX-License-Identifier: AGPL-3.0-only
"""Token-class table for constrained decoding (REQ-0027).

The old ``vocab_allowed`` advanced the automaton through every one of a pack's
token texts for every new automaton state.  This table keeps the per-state
operation exact without that full sweep:

* token ids are indexed by their first character and by the set of characters
  their decoded text contains;
* a state that can prove a universal character loop (for example the body of a
  JSON string) can add every token whose text is drawn only from that loop in
  one slice;
* a small character trie handles the remaining multi-character entries, and
  prunes a whole subtree only when the same proof covers every character below
  it.

The proof comes from the grammar, never from a shape heuristic.  A bulk set is
used only when a productive automaton loop accepts every character in the set;
otherwise the table falls back to the exact per-token advance.  The resulting
ids are sorted, like the old implementation.
"""
import collections
import threading

import numpy as np

# The large part of the table (the character buckets, safe-id sets and tries)
# depends only on the vocabulary.  Keep a few vocabularies' indexes alive so a
# new grammar in the same process does not rebuild them.  The cache is bounded:
# a server normally has one pack, while tests create many small vocabularies.
_INDEX_CACHE_LIMIT = 2
_INDEX_CACHE: "collections.OrderedDict[int, tuple[object, _VocabularyIndex]]" = collections.OrderedDict()
_INDEX_LOCK = threading.Lock()


class _VocabularyIndex(object):
    """The vocabulary-only part of a token-class table."""

    __slots__ = ("texts", "by_first", "by_char", "charsets", "alphabet",
                 "all_ids", "safe_ids", "outside_ids", "outside_tries", "root", "lock")

    def __init__(self, texts, by_first, by_char, charsets, alphabet):
        self.texts = texts
        self.by_first = by_first
        self.by_char = by_char
        self.charsets = charsets
        self.alphabet = alphabet
        self.all_ids = np.arange(len(texts), dtype=np.int64)
        self.safe_ids = {}
        self.outside_ids = {}
        self.outside_tries = {}
        self.root: _Node | None = None
        self.lock = threading.RLock()


def _build_index(texts):
    """Build the vocabulary-only indexes for ``texts``."""
    by_first = collections.defaultdict(list)
    by_char = collections.defaultdict(list)
    charsets = []
    alphabet = set()
    for token_id, text in enumerate(texts):
        first = text[:1]
        by_first[first].append(token_id)
        chars = frozenset(text)
        charsets.append(chars)
        alphabet.update(chars)
        for ch in chars:
            by_char[ch].append(token_id)
    by_first = {k: np.asarray(v, dtype=np.int64) for k, v in by_first.items()}
    by_char = {k: np.asarray(v, dtype=np.int64) for k, v in by_char.items()}
    return _VocabularyIndex(texts, by_first, by_char, charsets, frozenset(alphabet))


def _index_for(texts):
    """The shared index for one texts sequence (keyed by object identity)."""
    key = id(texts)
    with _INDEX_LOCK:
        entry = _INDEX_CACHE.get(key)
        if entry is not None and entry[0] is texts:
            _INDEX_CACHE.move_to_end(key)
            return entry[1]
        index = _build_index(texts)
        _INDEX_CACHE[key] = (texts, index)
        _INDEX_CACHE.move_to_end(key)
        while len(_INDEX_CACHE) > _INDEX_CACHE_LIMIT:
            _INDEX_CACHE.popitem(last=False)
        return index


class _Node(object):
    __slots__ = ("children", "ids", "all_ids", "chars")

    def __init__(self):
        self.children = {}
        self.ids = []
        self.all_ids = ()
        self.chars = frozenset()


class TokenClassTable(object):
    """A grammar-owned view over a process-shared vocabulary index.

    ``step(state, ch)`` advances one character and ``safe_sets(state, alphabet)``
    returns zero or more character sets for which the state has a productive
    universal loop.  Empty ``safe_sets`` is always safe: the exact path is used.

    The vocabulary-dependent indexes are reused across grammar instances that
    pass the same ``texts`` object, so a new structured request does not rebuild
    them.  Each grammar still owns its own step function, safe-set proof and
    per-state mask cache.  ``texts`` must not be mutated while the index is in
    use; the server and the backends already build it once and keep it.
    """

    def __init__(self, texts, step, safe_sets):
        self.texts = texts
        self._step = step
        self._safe_sets = safe_sets
        self._index = _index_for(texts)
        self._by_first = self._index.by_first
        self._by_char = self._index.by_char
        self._charsets = self._index.charsets
        self._alphabet = self._index.alphabet
        self._all_ids = self._index.all_ids
        self._safe_ids = self._index.safe_ids
        self._outside_ids = self._index.outside_ids
        self._outside_tries = self._index.outside_tries

    def allowed(self, state):
        """The exact ids allowed at ``state``; the caller owns the result cache."""
        safe = [s for s in self._safe_sets(state, self._alphabet) if s]
        if safe:
            # A broad class is the useful one; do not price every narrow
            # alternative just to select it.  The broad class makes its
            # complement small, so only tokens containing a non-loop character
            # need the exact check.  Narrow loops go to the trie.
            chosen = max(safe, key=len)
            ids = self._safe_ids_for(chosen)
            if len(ids) * 2 >= len(self.texts):
                outside = self._outside_ids_for(chosen)
                if outside.size:
                    kept = self._subset_allowed(state, outside, chosen)
                    ids = np.union1d(ids, kept)
                return ids
            return self._trie_allowed(state, safe)
        ids = self._first_char_allowed(state)
        if ids is not None:
            return ids
        return self._trie_allowed(state, safe)

    def _advance(self, state, text):
        for ch in text:
            state = self._step(state, ch)
            if state is None:
                return None
        return state

    def _safe_ids_for(self, chars):
        with self._index.lock:
            cached = self._safe_ids.get(chars)
            if cached is not None:
                return cached
            outside_chars = [c for c in self._alphabet if c not in chars]
            expected = sum(len(self._by_char[c]) for c in outside_chars)
            if expected <= len(self.texts) * 4:
                pieces = [self._by_char[c] for c in outside_chars if c in self._by_char]
                outside = (np.unique(np.concatenate(pieces)) if pieces else
                           np.empty(0, dtype=np.int64))
                self._outside_ids[chars] = outside
                ids = np.setdiff1d(self._all_ids, outside)
            else:
                # A narrow loop: scanning the token character sets is cheaper than
                # materializing almost the whole vocabulary as an outside union.
                ids = np.fromiter(
                    (i for i, token_chars in enumerate(self._charsets)
                     if token_chars <= chars),
                    dtype=np.int64)
            self._safe_ids[chars] = ids
            return ids

    def _outside_ids_for(self, chars):
        with self._index.lock:
            outside = self._outside_ids.get(chars)
            if outside is None:
                outside = np.setdiff1d(self._all_ids, self._safe_ids_for(chars))
                self._outside_ids[chars] = outside
            return outside

    def _exact_candidates(self, state, pieces):
        if not pieces:
            return np.empty(0, dtype=np.int64)
        candidates = np.concatenate(pieces)
        accepted = np.fromiter(
            (int(i) for i in candidates
             if self._advance(state, self.texts[int(i)]) is not None),
            dtype=np.int64)
        return np.unique(accepted)

    def _first_char_allowed(self, state):
        """Exact mask from the first-character buckets, when that is cheap."""
        pieces = []
        kept = 0
        for ch, ids in self._by_first.items():
            if ch == "" or self._step(state, ch) is not None:
                pieces.append(ids)
                kept += len(ids)
        if kept > 50000:
            return None
        return self._exact_candidates(state, pieces)


    def _subset_trie(self, ids, key):
        with self._index.lock:
            tree = self._outside_tries.get(key)
            if tree is not None:
                return tree
            root = _Node()
            for token_id in ids:
                node = root
                for ch in self.texts[int(token_id)]:
                    node = node.children.setdefault(ch, _Node())
                node.ids.append(int(token_id))
            self._outside_tries[key] = root
            return root

    def _subset_allowed(self, state, ids, key):
        root = self._subset_trie(ids, key)
        out = []
        stack = [(root, state)]
        while stack:
            node, node_state = stack.pop()
            if node_state is None:
                continue
            if node.ids:
                out.extend(node.ids)
            for ch, child in node.children.items():
                next_state = self._step(node_state, ch)
                if next_state is not None:
                    stack.append((child, next_state))
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(out), dtype=np.int64)

    def _build_trie(self):
        with self._index.lock:
            if self._index.root is not None:
                return self._index.root
            root = _Node()
            for token_id, text in enumerate(self.texts):
                node = root
                for ch in text:
                    node = node.children.setdefault(ch, _Node())
                node.ids.append(token_id)

            def finish(node):
                all_ids = list(node.ids)
                chars = set()
                for ch, child in node.children.items():
                    chars.add(ch)
                    child_ids, child_chars = finish(child)
                    all_ids.extend(child_ids)
                    chars.update(child_chars)
                node.all_ids = tuple(all_ids)
                node.chars = frozenset(chars)
                return node.all_ids, node.chars

            finish(root)
            self._index.root = root
            return root

    def _trie_allowed(self, state, safe):
        """Walk the character trie, pruning subtrees covered by ``safe``."""
        root = self._build_trie()
        out = []
        stack = [(root, state)]
        while stack:
            node, node_state = stack.pop()
            if node_state is None:
                continue
            if any(node.chars <= chars for chars in safe):
                out.extend(node.all_ids)
                continue
            if node.ids:
                out.extend(node.ids)
            for ch, child in node.children.items():
                next_state = self._step(node_state, ch)
                if next_state is not None:
                    stack.append((child, next_state))
        if not out:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(out), dtype=np.int64)


__all__ = ["TokenClassTable"]
