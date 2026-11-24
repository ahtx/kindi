# SPDX-License-Identifier: AGPL-3.0-only
"""GBNF grammars for constrained decoding (REQ-0018), as a character automaton.

``Grammar.from_string(text)`` parses llama.cpp's GBNF and exposes the same shape as
``jsongrammar.JsonGrammar`` — ``start()``, ``done(state)``, ``advance(state, text)`` and
``vocab_allowed(state, texts)`` — so the server's constrained sampler drives either one.
``vocab_allowed`` uses a token-class table (``kindi.token_classes``) and only
falls back to a per-token scan when the grammar cannot prove a universal
character loop.

Supported: rules ``name ::= body`` (``root`` is the start rule), alternation ``|``,
sequences, grouping ``(...)``, literals ``"..."`` with C-style escapes, character
classes ``[a-z]`` / ``[^...]`` with ranges and escapes, ``.`` (any character),
repetition ``* + ? {m} {m,} {m,n}``, rule references, comments ``#`` and blank lines.
A body may start on the line after ``::=`` and continue across newlines inside
parentheses or after ``|``, which is the form llama.cpp documents and writes its own
``grammars/json.gbnf`` in; a newline anywhere else ends the rule.

A recursive rule (one that reaches itself) is a pushdown language, which a finite
automaton cannot express. By default it is refused, naming the cycle. With
``KINDI_GBNF_MAX_DEPTH=N`` (N > 0) it is compiled by unrolling: every reference to a
recursive rule builds a fresh copy one level deeper, so nesting up to N levels is in
the automaton's language and a branch that would need level N+1 is pruned (that
nesting is refused, not silently truncated). Each reference gets its own copy - sharing
one would let a caller's successor leak into another use, which is why a plain NFA
cannot call and return. The default is 0 because the unrolled language is a *bounded*
subset of the real one: ``response_format: json_schema`` stays the exact, unbounded
path, and the state limit bounds the copies. Every refusal is a ``ValueError`` at
compile time, which the server turns into an enveloped 400 before any token is
generated.

The automaton is forward-complete: a token is only accepted when the state it reaches
can still reach the accepting state, so constrained decoding never paints itself into
a dead end that would leave the allowed set empty.
"""
import collections
import os

# A grammar is compiled into one state per literal character, so an unbounded repetition or a
# megabyte literal is a memory request from whoever sent the request. Both limits are refusals
# (ValueError -> an enveloped 400), not silent truncation.
MAX_REPEAT = 10000
DEFAULT_MAX_STATES = 20000
DEFAULT_MAX_DEPTH = 0                     # recursion unrolling is opt-in (0 = refuse)


def _state_limit():
    """KINDI_GBNF_MAX_STATES (read per grammar, so an operator can raise it without a restart of
    the module's constants). A bad value falls back to the default rather than failing a request."""
    try:
        return max(1, int(os.environ.get("KINDI_GBNF_MAX_STATES", str(DEFAULT_MAX_STATES))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_STATES


def _depth_limit():
    """KINDI_GBNF_MAX_DEPTH: how many times a recursive rule may be entered before the
    automaton prunes that branch. 0 (the default) keeps the strict refusal; read per
    grammar, so an operator can raise it without a restart."""
    try:
        return max(0, int(os.environ.get("KINDI_GBNF_MAX_DEPTH", str(DEFAULT_MAX_DEPTH))))
    except (TypeError, ValueError):
        return DEFAULT_MAX_DEPTH


# --- parsing ---------------------------------------------------------------

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'",
            "0": "\0", "[": "[", "]": "]", "-": "-", ".": "."}


class _Parser(object):
    """Recursive-descent parser for the GBNF subset above. Raises ValueError."""

    def __init__(self, text):
        self.s = text if isinstance(text, str) else ""
        self.i = 0
        self.n = len(self.s)
        self.rules = {}
        self.order = []
        self.depth = 0                    # parenthesis nesting: newlines are whitespace inside

    def error(self, msg):
        line = self.s.count("\n", 0, self.i) + 1
        raise ValueError("GBNF line %d: %s" % (line, msg))

    # whitespace and comments

    def skip_inline(self):
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r":
                self.i += 1
            elif c == "#":
                while self.i < self.n and self.s[self.i] != "\n":
                    self.i += 1
            else:
                return

    def skip_lines(self):
        while True:
            self.skip_inline()
            if self.i < self.n and self.s[self.i] == "\n":
                self.i += 1
            else:
                return

    # grammar

    def parse(self):
        while True:
            self.skip_lines()
            if self.i >= self.n:
                break
            name = self.name()
            self.skip_inline()
            if not self.s.startswith("::=", self.i):
                self.error("expected '::=' after rule name %r" % name)
            self.i += 3
            self.skip_lines()             # llama.cpp's json.gbnf starts a body on the next line
            body = self.alternation()
            self.skip_inline()
            if self.i < self.n and self.s[self.i] != "\n":
                self.error("unexpected %r after the body of %r" % (self.s[self.i], name))
            if name in self.rules:
                self.error("rule %r is defined twice" % name)
            self.rules[name] = body
            self.order.append(name)
        if not self.rules:
            raise ValueError("GBNF: no rules")
        if "root" not in self.rules:
            raise ValueError("GBNF: no 'root' rule (llama.cpp starts the grammar at 'root')")
        return self.rules

    def name(self):
        start = self.i
        while self.i < self.n and (self.s[self.i].isalnum() or self.s[self.i] == "_"):
            self.i += 1
        if start == self.i:
            self.error("expected a rule name or a reference")
        if self.s[start].isdigit():
            self.error("a name cannot start with a digit")
        return self.s[start:self.i]

    def alternation(self):
        branches = [self.sequence()]
        self.skip_inline()
        while self.i < self.n and self.s[self.i] == "|":
            self.i += 1
            self.skip_lines()             # a newline after '|' continues the same rule
            branches.append(self.sequence())
            self.skip_inline()
        return branches

    def sequence(self):
        items = []
        while True:
            self.skip_inline()
            if self.i >= self.n or self.s[self.i] in "|)":
                return items
            if self.s[self.i] == "\n":
                if not self.depth:
                    return items          # at the top level a newline ends the rule
                self.i += 1               # inside parentheses newlines are whitespace
                continue
            items.append(self.item())

    def item(self):
        atom = self.atom()
        while True:
            self.skip_inline()
            if self.i >= self.n:
                return atom
            c = self.s[self.i]
            if c == "*":
                self.i += 1
                atom = ("rep", atom, 0, None)
            elif c == "+":
                self.i += 1
                atom = ("rep", atom, 1, None)
            elif c == "?":
                self.i += 1
                atom = ("rep", atom, 0, 1)
            elif c == "{":
                lo, hi = self.braces()
                atom = ("rep", atom, lo, hi)
            else:
                return atom

    def atom(self):
        c = self.s[self.i]
        if c == "(":
            self.i += 1
            self.depth += 1
            try:
                alt = self.alternation()
            finally:
                self.depth -= 1
            self.skip_inline()
            if self.i >= self.n or self.s[self.i] != ")":
                self.error("expected ')'")
            self.i += 1
            return ("group", alt)
        if c == '"':
            return ("lit", self.literal())
        if c == "[":
            ranges, negated = self.charclass()
            return ("range", ranges, negated)
        if c == ".":
            self.i += 1
            return ("any",)
        if c.isalnum() or c == "_":
            return ("ref", self.name())
        self.error("unexpected %r" % c)

    def literal(self):
        self.i += 1                                   # opening quote
        out = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == '"':
                self.i += 1
                return "".join(out)
            if c == "\n":
                self.error("unterminated string literal")
            if c == "\\":
                out.append(self.escape())
            else:
                self.i += 1
                out.append(c)
        self.error("unterminated string literal")

    def escape(self):
        self.i += 1                                   # backslash
        if self.i >= self.n:
            self.error("trailing backslash")
        c = self.s[self.i]
        if c == "x":
            h = self.s[self.i + 1:self.i + 3]
            if len(h) != 2 or any(d not in "0123456789abcdefABCDEF" for d in h):
                self.error("\\x needs two hex digits")
            self.i += 3
            return chr(int(h, 16))
        if c == "u":
            h = self.s[self.i + 1:self.i + 5]
            if len(h) != 4 or any(d not in "0123456789abcdefABCDEF" for d in h):
                self.error("\\u needs four hex digits")
            self.i += 5
            return chr(int(h, 16))
        self.i += 1
        return _ESCAPES.get(c, c)

    def charclass(self):
        self.i += 1                                   # opening bracket
        negated = self.i < self.n and self.s[self.i] == "^"
        if negated:
            self.i += 1
        ranges = []
        while self.i < self.n:
            if self.s[self.i] == "]":
                self.i += 1
                if not ranges:
                    self.error("empty character class")
                return tuple(ranges), negated
            lo = self.class_char()
            if (self.i + 1 < self.n and self.s[self.i] == "-"
                    and self.s[self.i + 1] != "]"):
                self.i += 1
                hi = self.class_char()
                if ord(hi) < ord(lo):
                    self.error("character class range is reversed")
                ranges.append((ord(lo), ord(hi)))
            else:
                ranges.append((ord(lo), ord(lo)))
        self.error("unterminated character class")

    def class_char(self):
        if self.i >= self.n:
            self.error("unterminated character class")
        if self.s[self.i] == "\\":
            return self.escape()
        c = self.s[self.i]
        self.i += 1
        return c

    def braces(self):
        self.i += 1                                   # opening brace

        def number():
            start = self.i
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
            if start == self.i:
                self.error("expected a number inside {...}")
            return int(self.s[start:self.i])

        lo = number()
        hi = lo
        if self.i < self.n and self.s[self.i] == ",":
            self.i += 1
            hi = None if (self.i < self.n and self.s[self.i] == "}") else number()
        if self.i >= self.n or self.s[self.i] != "}":
            self.error("expected '}'")
        self.i += 1
        if hi is not None and hi < lo:
            self.error("{m,n} needs n >= m")
        if hi is not None and hi > MAX_REPEAT:
            self.error("{m,n} expands to at most %d copies (%d requested); narrower bounds or a "
                       "rule reference with '*' is cheaper" % (MAX_REPEAT, hi))
        if lo > MAX_REPEAT:
            self.error("{m} expands to at most %d copies (%d requested)" % (MAX_REPEAT, lo))
        return lo, hi


# --- the automaton ---------------------------------------------------------

class _TooDeep(Exception):
    """A branch would need more recursion levels than KINDI_GBNF_MAX_DEPTH allows."""

    def __init__(self, rule):
        Exception.__init__(self, rule)
        self.rule = rule


def _refs(node, out):
    kind = node[0]
    if kind == "ref":
        out.add(node[1])
    elif kind == "group":
        for seq in node[1]:
            for item in seq:
                _refs(item, out)
    elif kind == "rep":
        _refs(node[1], out)


def _recursive_rules(rules):
    """The rules that can reach themselves - the ones that are unrolled per level."""
    edges = {}
    for name, branches in rules.items():
        out = set()
        for seq in branches:
            for item in seq:
                _refs(item, out)
        edges[name] = out
    found = set()
    for start in rules:
        seen = set()
        stack = list(edges[start])
        while stack:
            name = stack.pop()
            if name == start:
                found.add(start)
                break
            if name in seen or name not in edges:
                continue
            seen.add(name)
            stack.extend(edges[name])
    return found


class _NFA(object):
    """Thompson construction over labels ("ch", char) / ("range", ...) / ("any",)."""

    def __init__(self, rules, max_states=None, max_depth=None):
        self.program = rules
        self.edges = {}
        self.count = 0
        self.path = []                 # rules being built (for the cycle message)
        self.limit_hit = None          # the rule whose nesting first exceeded max_depth
        self.recursive = _recursive_rules(rules)
        self.max_states = _state_limit() if max_states is None else int(max_states)
        self.max_depth = _depth_limit() if max_depth is None else int(max_depth)

    def state(self):
        if self.count >= self.max_states:
            raise ValueError(
                "GBNF: grammar too large (%d automaton states, limit %d); every reference is "
                "expanded once per use, so prefer `*`/`+` to a large `{n}`, shorten a literal, or "
                "use response_format json_schema (KINDI_GBNF_MAX_STATES raises the limit)"
                % (self.count, self.max_states))
        s = self.count
        self.count += 1
        self.edges[s] = []
        return s

    def edge(self, a, label, b):
        self.edges[a].append((label, b))

    def build_rule(self, name, level=0):
        """Compile one reference to a rule. Every reference gets a fresh copy.

        Sharing a completed sub-automaton between two uses does not work in a plain NFA: the
        entry and exit states would be shared too, so the epsilon closure after the first use
        would reach the second use's successor and the automaton would accept a string that
        skipped the intervening symbols (real grammars reuse rules constantly).

        ``level`` is the recursion budget: entering a rule that can reach itself costs one
        level, and past ``max_depth`` that branch is pruned (``_TooDeep``). Non-recursive
        rules never spend a level, so a flat grammar is unaffected.
        """
        if name not in self.program:
            raise ValueError("GBNF: unknown rule %r" % name)
        if name in self.path:
            # a rule already being built is reached again: these rules have no
            # representation without unrolling, and `path` already holds the cycle.
            if self.max_depth == 0:
                cycle = self.path[self.path.index(name):] + [name]
                raise ValueError(
                    "GBNF: recursive rule (%s); this slice compiles regular grammars "
                    "only - use response_format json_schema for recursive JSON, or set "
                    "KINDI_GBNF_MAX_DEPTH to unroll it" % " -> ".join(cycle))
            if level > self.max_depth:
                self.limit_hit = name
                raise _TooDeep(name)
        self.path.append(name)
        try:
            return self.build_alt(self.program[name], level)
        finally:
            self.path.pop()

    def build_alt(self, branches, level=0):
        start, end = self.state(), self.state()
        kept = 0
        for seq in branches:
            try:
                s, e = self.build_seq(seq, level)
            except _TooDeep:
                continue                     # this branch needs more nesting than allowed
            self.edge(start, ("eps",), s)
            self.edge(e, ("eps",), end)
            kept += 1
        if not kept:
            # nothing survived: report the rule that actually hit the limit, not the branch
            raise _TooDeep(self.limit_hit or "branch")
        return start, end

    def build_seq(self, items, level=0):
        start = end = self.state()
        for item in items:
            s, e = self.build_item(item, level)
            self.edge(end, ("eps",), s)
            end = e
        return start, end

    def build_item(self, item, level=0):
        kind = item[0]
        if kind == "lit":
            start = end = self.state()
            for ch in item[1]:
                nxt = self.state()
                self.edge(end, ("ch", ch), nxt)
                end = nxt
            return start, end
        if kind == "range":
            start, end = self.state(), self.state()
            self.edge(start, ("range", item[1], item[2]), end)
            return start, end
        if kind == "any":
            start, end = self.state(), self.state()
            self.edge(start, ("any",), end)
            return start, end
        if kind == "ref":
            name = item[1]
            # one level per *re-entry* of a rule already being built, not per use: two
            # sibling uses of a recursive rule are the same nesting level, and charging
            # them separately would both waste the budget and duplicate more states.
            return self.build_rule(name, level + 1 if name in self.path else level)
        if kind == "group":
            return self.build_alt(item[1], level)
        if kind == "rep":
            return self.build_rep(item[1], item[2], item[3], level)
        raise ValueError("GBNF: unknown node %r" % (kind,))

    def build_rep(self, sub, lo, hi, level=0):
        start = end = self.state()
        for _ in range(lo):
            s, e = self.build_item(sub, level)
            self.edge(end, ("eps",), s)
            end = e
        if hi is None:
            s, e = self.build_item(sub, level)
            nxt = self.state()
            self.edge(end, ("eps",), s)
            self.edge(e, ("eps",), s)                 # loop
            self.edge(end, ("eps",), nxt)
            self.edge(e, ("eps",), nxt)
            return start, nxt
        for _ in range(hi - lo):
            s, e = self.build_item(sub, level)
            nxt = self.state()
            self.edge(end, ("eps",), s)
            self.edge(e, ("eps",), nxt)
            self.edge(end, ("eps",), nxt)             # skip this copy
            end = nxt
        return start, end


def _accepts(label, ch):
    kind = label[0]
    if kind == "ch":
        return ch == label[1]
    if kind == "any":
        return True
    if kind == "range":
        cp = ord(ch)
        hit = any(lo <= cp <= hi for lo, hi in label[1])
        return (not hit) if label[2] else hit
    return False


class Grammar(object):
    """A parsed GBNF grammar with the same interface as ``jsongrammar.JsonGrammar``."""

    def __init__(self, rules, start_rule="root"):
        nfa = _NFA(rules)
        self._nfa = nfa
        try:
            self._start, self._accept = nfa.build_rule(start_rule)
        except _TooDeep as e:
            raise ValueError(
                "GBNF: recursion in %r needs more than KINDI_GBNF_MAX_DEPTH=%d levels; "
                "raise the limit, or use response_format json_schema for recursive JSON"
                % (e.rule, nfa.max_depth))
        self._productive = self._productive_states()
        self._closures = {}
        self._vocab_cache = {}
        self._mask_table = None
        self._mask_texts = None
        self._self_loop_chars = {}

    @property
    def states(self):
        """Automaton states in this grammar (a size diagnostic; the build itself is bounded by
        ``KINDI_GBNF_MAX_STATES``). Every reference is expanded once per use, so this grows with
        reuse, not with the number of rules."""
        return self._nfa.count

    @classmethod
    def from_string(cls, text):
        """Parse GBNF text. Raises ValueError on syntax errors and unknown rules.

        A recursive rule is refused too, unless KINDI_GBNF_MAX_DEPTH opts into the bounded
        unroll; that language is exact at the depths it allows but bounded, so
        ``response_format: json_schema`` stays the recursive-JSON path.
        """
        return cls(_Parser(text).parse())

    # --- states ---------------------------------------------------------

    def _closure(self, states):
        cached = self._closures.get(states)
        if cached is not None:
            return cached
        out = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for label, target in self._nfa.edges[s]:
                if label[0] == "eps" and target not in out:
                    out.add(target)
                    stack.append(target)
        closed = frozenset(out)
        self._closures[states] = closed
        return closed

    def _productive_states(self):
        """Every NFA state from which the accepting state is still reachable."""
        reverse = collections.defaultdict(list)
        for src, edges in self._nfa.edges.items():
            for _, target in edges:
                reverse[target].append(src)
        seen = set([self._accept])
        stack = [self._accept]
        while stack:
            s = stack.pop()
            for prev in reverse[s]:
                if prev not in seen:
                    seen.add(prev)
                    stack.append(prev)
        return seen

    def start(self):
        return self._closure(frozenset([self._start]))

    def done(self, state):
        return state is not None and self._accept in state

    def advance(self, state, text):
        """The state reached by ``text``, or None when the text leaves the language."""
        if state is None:
            return None
        for ch in (text or ""):
            state = self._step(state, ch)
            if state is None:
                return None
        return state

    def _step(self, state, ch):
        out = set()
        for s in state:
            for label, target in self._nfa.edges[s]:
                if label[0] != "eps" and _accepts(label, ch):
                    out.add(target)
        if not out:
            return None
        closed = self._closure(frozenset(out))
        if not (closed & self._productive):
            return None                                # dead end: the language cannot finish
        return closed

    def _token_safe_sets(self, state, alphabet):
        """Characters a productive self-loop in ``state`` accepts without end.

        A token whose text uses only one such set can be accepted from that
        loop for its whole length.  The table uses this only as a proof; it
        never treats a merely plausible JSON-shaped class as safe.
        """
        out = []
        for q in state:
            if q not in self._productive:
                continue
            chars = self._self_loop_chars.get(q)
            if chars is None:
                closed = frozenset([q])
                chars = set()
                for c in alphabet:
                    nxt = self._step(closed, c)
                    if nxt is not None and q in nxt:
                        chars.add(c)
                chars = frozenset(chars)
                self._self_loop_chars[q] = chars
            if chars:
                out.append(chars)
        return out

    def _vocab_table(self, texts):
        if self._mask_texts is not texts:
            from .token_classes import TokenClassTable

            self._mask_table = TokenClassTable(texts, self._step, self._token_safe_sets)
            self._mask_texts = texts
            self._vocab_cache.clear()
        table = self._mask_table
        if table is None:
            raise RuntimeError("token-class table was not initialized")
        return table

    def _vocab_allowed_scan(self, state, texts):
        """The pre-token-class implementation, kept as the set-equality oracle."""
        import numpy as np

        return np.fromiter((i for i, t in enumerate(texts) if self.advance(state, t) is not None),
                           dtype=np.int64)

    def vocab_allowed(self, state, texts):
        """Token ids whose decoded text keeps ``state`` alive (cached per state)."""
        cached = self._vocab_cache.get(state)
        if cached is not None:
            return cached
        ids = self._vocab_table(texts).allowed(state)
        self._vocab_cache[state] = ids
        return ids


def parse(text):
    """The rule table of a GBNF string (mostly useful in tests and error messages)."""
    return _Parser(text).parse()


__all__ = ["Grammar", "parse"]
