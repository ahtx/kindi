# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded recursion in GBNF (KINDI_GBNF_MAX_DEPTH, REQ-0018).

A recursive rule is a pushdown language, so no automaton represents it exactly. The opt-in
unroll compiles a fixed number of levels, one fresh copy per reference. Two properties make
that shippable, and this module checks both without a model:

  * at every allowed depth the automaton's language equals the grammar's, checked
    exhaustively against an independent AST matcher over all strings up to a small length,
    for several recursion shapes and every depth up to the limit;
  * nesting past the limit is *refused* rather than silently truncated - the branch is
    pruned, so the extra symbol is not in the allowed set.

The cost is real: with no sharing, every level re-expands every reused rule, so the
canonical JSON grammar reaches nesting 2 inside the default 20,000-state limit. That is
why the feature is opt-in, and why ``response_format: json_schema`` remains the exact,
unbounded path.
"""
import itertools

import pytest

from gbnf_fixtures import CANONICAL_JSON
from kindi.gbnf import Grammar, parse


def _accepts(grammar, text):
    return grammar.done(grammar.advance(grammar.start(), text))


def _set_depth(monkeypatch, depth, states=None):
    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", str(depth))
    if states is not None:
        monkeypatch.setenv("KINDI_GBNF_MAX_STATES", str(states))


# --- the reference matcher -------------------------------------------------
# An independent reading of the same grammar: backtracking over the parse tree, with the
# recursion budget counted the way the compiler counts it (one level per re-entry of a rule
# that is already being built, so a second sibling use costs nothing). Any disagreement
# with the automaton means the unroll does not mean what the grammar means.

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
    edges = {}
    for name, branches in rules.items():
        out = set()
        for seq in branches:
            for item in seq:
                _refs(item, out)
        edges[name] = out
    found = set()
    for start in rules:
        stack, seen = list(edges[start]), set()
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


def _in_class(ch, ranges, negated):
    return any(lo <= ord(ch) <= hi for lo, hi in ranges) != negated


def _apply(rules, recursive, node, text, positions, path, level, max_depth):
    out = set()
    for pos in positions:
        out |= _ends(rules, recursive, node, text, pos, path, level, max_depth)
    return out


def _seq(rules, recursive, items, text, pos, path, level, max_depth):
    positions = {pos}
    for item in items:
        positions = _apply(rules, recursive, item, text, positions, path, level, max_depth)
        if not positions:
            break
    return positions


def _alts(rules, recursive, branches, text, pos, path, level, max_depth):
    out = set()
    for seq in branches:
        out |= _seq(rules, recursive, seq, text, pos, path, level, max_depth)
    return out


def _ends(rules, recursive, node, text, pos, path, level, max_depth):
    kind = node[0]
    if kind == "lit":
        return {pos + len(node[1])} if text.startswith(node[1], pos) else set()
    if kind == "range":
        if pos < len(text) and _in_class(text[pos], node[1], node[2]):
            return {pos + 1}
        return set()
    if kind == "any":
        return {pos + 1} if pos < len(text) else set()
    if kind == "group":
        return _alts(rules, recursive, node[1], text, pos, path, level, max_depth)
    if kind == "ref":
        name = node[1]
        if name not in rules:
            return set()
        # the compiler's budget: every re-entry of a rule already being built costs a
        # level, so the depth of an occurrence is the number of re-entries above it, not
        # the number of times that one rule appears
        entered = level + 1 if name in path else level
        if name in recursive and entered > max_depth:
            return set()                   # this nesting level is not in the language
        return _alts(rules, recursive, rules[name], text, pos, path + [name], entered, max_depth)
    if kind == "rep":
        sub, lo, hi = node[1], node[2], node[3]
        limit = hi if hi is not None else len(text) + 1
        out = {pos} if lo == 0 else set()
        cur, copies = {pos}, 0
        while cur and copies < limit:
            cur = _apply(rules, recursive, sub, text, cur, path, level, max_depth)
            copies += 1
            if copies >= lo:
                out |= cur
        return out
    raise AssertionError("unknown node %r" % (kind,))


def _reference_accepts(rules, recursive, text, max_depth):
    ends = _alts(rules, recursive, rules["root"], text, 0, ["root"], 0, max_depth)
    return len(text) in ends


# --- the property the unroll has to have -----------------------------------

CASES = [
    # (grammar, alphabet, longest string enumerated)
    ('root ::= "(" root ")" | "x"', "()x", 8),
    ('root ::= "a" root | "b"', "ab", 9),
    ('root ::= "[" item "]" | "x"\nitem ::= root | "y"', "[]xy", 6),
    ('root ::= "p" root "s" | ""', "ps", 8),
]


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("grammar,alphabet,longest", CASES)
def test_the_unrolled_language_equals_the_grammar(monkeypatch, depth, grammar, alphabet, longest):
    _set_depth(monkeypatch, depth)
    automaton = Grammar.from_string(grammar)
    rules = parse(grammar)
    recursive = _recursive_rules(rules)
    wrong = []
    for length in range(longest + 1):
        for chars in itertools.product(alphabet, repeat=length):
            text = "".join(chars)
            got = _accepts(automaton, text)
            want = _reference_accepts(rules, recursive, text, depth)
            if got != want:
                wrong.append((text, got, want))
    assert not wrong, "automaton and grammar disagree at depth %d: %r" % (depth, wrong[:5])


# --- the opt-in itself -----------------------------------------------------

def test_recursion_stays_refused_without_the_opt_in(monkeypatch):
    monkeypatch.delenv("KINDI_GBNF_MAX_DEPTH", raising=False)
    with pytest.raises(ValueError, match="recursive rule"):
        Grammar.from_string('root ::= "(" root ")" | "x"')


@pytest.mark.parametrize("value", ["0", "-3", "not-a-number"])
def test_a_zero_or_unusable_limit_means_refuse(monkeypatch, value):
    monkeypatch.setenv("KINDI_GBNF_MAX_DEPTH", value)
    with pytest.raises(ValueError, match="recursive rule"):
        Grammar.from_string('root ::= "(" root ")" | "x"')


def test_the_opt_in_unrolls_to_the_limit_and_refuses_deeper(monkeypatch):
    _set_depth(monkeypatch, 4)
    g = Grammar.from_string('root ::= "(" root ")" | "x"')
    for depth in range(5):
        assert _accepts(g, "(" * depth + "x" + ")" * depth)
    assert not _accepts(g, "(" * 5 + "x" + ")" * 5)
    # refused, not truncated: the deepest allowed body still starts, a fifth level does not
    state = g.advance(g.start(), "(" * 4)
    assert state is not None and not g.done(state)
    assert g.advance(state, "x") is not None
    assert g.advance(state, "(") is None


def test_the_canonical_json_grammar_compiles_when_opted_in(monkeypatch):
    _set_depth(monkeypatch, 3, states=60000)
    g = Grammar.from_string(CANONICAL_JSON)
    assert _accepts(g, '{"a": 1}')
    assert _accepts(g, '{"a": [1, 2, {"b": null}]}')
    assert not _accepts(g, '{"a": }')              # the reused-rule bug stays fixed
    assert not _accepts(g, '{"a": 1} x')
    assert not _accepts(g, '{"a": "b\nc"}')        # a raw newline inside a string

    def nested(depth):
        return '{"a": ' + "[" * depth + "1" + "]" * depth + "}"

    deepest = max(depth for depth in range(1, 12) if _accepts(g, nested(depth)))
    assert deepest >= 1
    assert not _accepts(g, nested(deepest + 1))


def test_a_grammar_with_only_recursive_branches_names_the_limit(monkeypatch):
    _set_depth(monkeypatch, 1)
    with pytest.raises(ValueError, match="KINDI_GBNF_MAX_DEPTH=1"):
        Grammar.from_string('root ::= "[" root "]"')      # every branch nests one deeper


def test_a_deep_opt_in_still_hits_the_state_limit(monkeypatch):
    _set_depth(monkeypatch, 12)
    with pytest.raises(ValueError, match="too large"):
        Grammar.from_string(CANONICAL_JSON)
