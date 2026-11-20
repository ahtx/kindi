# SPDX-License-Identifier: AGPL-3.0-only
"""The token-class mask must equal the old per-token sweep (REQ-0027 A3).

Every case here compares the new ``vocab_allowed`` with
``_vocab_allowed_scan`` for every vocabulary entry, over states reached by a
short document and by a small reachable-state exploration.  The token list is
deliberately adversarial: quotes, backslashes, whitespace, control characters,
prefixes, duplicates and the empty piece all occur, so a class shortcut cannot
pass by looking JSON-shaped.
"""
import numpy as np

import kindi.token_classes as token_classes
from kindi.gbnf import Grammar
from kindi.jsongrammar import grammar_from_schema

TEXTS = [
    "",
    '"',
    "a",
    "ab",
    "abc",
    "a b",
    " ",
    "  ",
    "\t",
    "\n",
    "\x01",
    "\\",
    "\\n",
    '\\"',
    '"a',
    'a"',
    "0",
    "01",
    "12",
    "1.5",
    "-3",
    "true",
    "false",
    "null",
    "{",
    "}",
    "[",
    "]",
    ":",
    ",",
    '{"name": "a',
    '"name"',
    "\\u0041",
    "name",
    "tags",
]


def _same_mask(grammar, texts, states):
    for state in states:
        got = grammar.vocab_allowed(state, texts)
        want = grammar._vocab_allowed_scan(state, texts)
        assert got.tolist() == want.tolist(), (state, got, want)


def _reachable(grammar, alphabet, limit=250):
    seen = {grammar.start()}
    todo = [grammar.start()]
    while todo and len(seen) < limit:
        state = todo.pop()
        for ch in alphabet:
            nxt = grammar._step(state, ch)
            if nxt is not None and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def test_json_schema_table_matches_the_sweep_on_string_and_key_states():
    grammar = grammar_from_schema({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "n": {"type": "integer"},
        },
        "required": ["name", "n"],
    })
    states = [grammar.start()]
    for text in ['{"name": "a', '{"name": "a\\', '{"name": "abc"', '{"n": -',
                 '{"name": "x", "n": 12', '{"name": "x"}']:
        state = grammar.start()
        for ch in text:
            state = grammar._step(state, ch)
            if state is None:
                break
        if state is not None:
            states.append(state)
    _same_mask(grammar, TEXTS, states)


def test_json_schema_table_matches_on_every_small_reachable_state():
    grammar = grammar_from_schema({
        "type": "object",
        "properties": {"s": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["s", "n"],
    })
    alphabet = sorted(set('{}[],:"\\ 0123abtnf.-'))
    _same_mask(grammar, TEXTS, _reachable(grammar, alphabet))


LOOP_GBNF = r'''root ::= "\"" body "\""
body ::= [^"\\\x00-\x1F]+
'''


def test_gbnf_closure_loop_table_matches_the_sweep():
    grammar = Grammar.from_string(LOOP_GBNF)
    states = [grammar.start()]
    state = grammar.start()
    for ch in '"abc"':
        state = grammar._step(state, ch)
        if state is None:
            break
        states.append(state)
    _same_mask(grammar, TEXTS, states)


def test_gbnf_table_matches_on_every_small_reachable_state():
    grammar = Grammar.from_string(LOOP_GBNF)
    alphabet = sorted(set('"abc \\\n\x01'))
    _same_mask(grammar, TEXTS, _reachable(grammar, alphabet))


def test_a_bounded_loop_is_not_treated_as_a_universal_loop():
    grammar = Grammar.from_string(r'root ::= "\"" [a-z]{1,3} "\""')
    state = grammar._step(grammar.start(), '"')
    _same_mask(grammar, TEXTS, [grammar.start(), state])


def test_a_prefix_loop_does_not_accept_a_non_loop_suffix():
    grammar = Grammar.from_string('root ::= "a"+ "b"')
    state = grammar.start()
    for ch in "aa":
        state = grammar._step(state, ch)
    _same_mask(grammar, TEXTS, [grammar.start(), state])


def test_empty_piece_and_duplicate_texts_are_preserved():
    grammar = grammar_from_schema({"type": "string"})
    state = grammar._step(grammar.start(), '"')
    texts = ["", "a", "a", "\\", '"', "ab"]
    got = grammar.vocab_allowed(state, texts)
    want = grammar._vocab_allowed_scan(state, texts)
    assert np.array_equal(got, want)


def test_the_vocabulary_index_is_built_once_for_two_grammars(monkeypatch):
    """A new structured request must not rebuild the 248K-entry vocabulary index."""
    texts = list(TEXTS)
    builds = []
    real_build = token_classes._build_index

    def counted(texts_arg):
        builds.append(texts_arg)
        return real_build(texts_arg)

    monkeypatch.setattr(token_classes, "_build_index", counted)
    first = grammar_from_schema({"type": "object", "properties": {"name": {"type": "string"}},
                                 "required": ["name"]})
    second = grammar_from_schema({"type": "object", "properties": {"city": {"type": "string"}},
                                  "required": ["city"]})
    first_table = first._vocab_table(texts)
    second_table = second._vocab_table(texts)
    assert first_table._index is second_table._index
    assert first_table._safe_ids is second_table._safe_ids
    assert first_table._outside_tries is second_table._outside_tries
    assert len(builds) == 1


def test_a_shared_index_does_not_change_either_language():
    """The process cache shares only vocabulary data, never grammar proofs or state caches."""
    texts = list(TEXTS)
    first = grammar_from_schema({"type": "object", "properties": {"name": {"type": "string"}},
                                 "required": ["name"]})
    second = grammar_from_schema({"type": "object", "properties": {"city": {"type": "string"}},
                                  "required": ["city"]})
    first_state = first.start()
    second_state = second.start()
    assert first.vocab_allowed(first_state, texts).tolist() == \
        first._vocab_allowed_scan(first_state, texts).tolist()
    assert second.vocab_allowed(second_state, texts).tolist() == \
        second._vocab_allowed_scan(second_state, texts).tolist()
    # A string-body state reaches the bulk safe-set/trie path on the second grammar.
    for grammar, text in ((first, '{"name": "a'), (second, '{"city": "c')):
        state = grammar.start()
        for ch in text:
            state = grammar._step(state, ch)
        _same_mask(grammar, texts, [state])


def test_gbnf_instances_share_the_same_index_and_language():
    texts = list(TEXTS)
    first = Grammar.from_string(LOOP_GBNF)
    second = Grammar.from_string('root ::= "[" [a-z ]+ "]"')
    first_table = first._vocab_table(texts)
    second_table = second._vocab_table(texts)
    assert first_table._index is second_table._index
    states = [second.start()]
    state = second._step(second.start(), "[")
    assert state is not None
    states.append(state)
    _same_mask(second, texts, states)
