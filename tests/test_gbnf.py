# SPDX-License-Identifier: AGPL-3.0-only
"""GBNF grammars for constrained decoding (REQ-0018).

The automaton is tested on its own (no model, no GPU): the language it accepts, the
dead-end rule that keeps the sampler's allowed set non-empty, and the parse errors the
server turns into an enveloped 400. The last section runs the grammars people actually
write - llama.cpp's own ``json.gbnf`` layout, a tool call, an enum and an ISO date - so
the parser stays pinned to llama.cpp's documented GBNF rather than to a convenient
subset. That file is recursive, which is refused unless KINDI_GBNF_MAX_DEPTH opts into a
bounded unroll: the test pins the layout (it parses), the refusal (it names the
cycle), and test_gbnf_recursion.py covers the opt-in and its limits.
"""
import string

import pytest

from kindi import gbnf
from kindi.gbnf import Grammar, parse
from gbnf_fixtures import CANONICAL_JSON


def _accepts(grammar, text):
    return grammar.done(grammar.advance(grammar.start(), text))


def test_literal_alternation():
    g = Grammar.from_string('root ::= "yes" | "no"')
    assert _accepts(g, "yes") and _accepts(g, "no")
    assert not _accepts(g, "ye")
    assert not _accepts(g, "yess")


def test_sequence_group_and_repetition():
    g = Grammar.from_string('root ::= ("ab")+ "c"?')
    assert _accepts(g, "ab")
    assert _accepts(g, "ababc")
    assert _accepts(g, "abc")
    assert not _accepts(g, "a")


def test_character_classes_and_negation():
    g = Grammar.from_string("root ::= [a-c]+ [^0-9]")
    assert _accepts(g, "abcx")
    assert not _accepts(g, "abc1")


def test_bounded_and_open_counts():
    bounded = Grammar.from_string('root ::= "a"{2,3}')
    assert not _accepts(bounded, "a")
    assert _accepts(bounded, "aa") and _accepts(bounded, "aaa")
    assert not _accepts(bounded, "aaaa")
    open_ended = Grammar.from_string('root ::= "a"{2,}')
    assert _accepts(open_ended, "a" * 5)


def test_escapes_comments_and_hex():
    g = Grammar.from_string('# a comment\nroot ::= "\\n" [\\t] "\\x41"')
    assert _accepts(g, "\n\tA")


def test_rule_references():
    g = Grammar.from_string(
        'root ::= greeting " " name\ngreeting ::= "hi" | "hello"\nname ::= [^ ]+')
    assert _accepts(g, "hello Ada")
    assert not _accepts(g, "hi")


def test_dot_matches_any_character():
    g = Grammar.from_string('root ::= . "!"')
    assert _accepts(g, "x!")
    assert not _accepts(g, "!")


def test_empty_body_is_the_empty_string():
    g = Grammar.from_string('root ::= ""')
    assert _accepts(g, "")


def test_a_dead_end_is_refused_before_the_sampler_reaches_it():
    """A live state that can no longer accept must not keep the allowed set open:
    otherwise the sampler picks a token the grammar can never finish."""
    g = Grammar.from_string('root ::= "ab" | "ac"')
    after_a = g.advance(g.start(), "a")
    assert after_a is not None
    assert g.advance(after_a, "x") is None


def test_vocab_allowed_keeps_only_tokens_inside_the_language():
    g = Grammar.from_string('root ::= "yes" | "no"')
    texts = ["y", "ye", "yes", "n", "no", "nope", "x"]
    ids = set(int(i) for i in g.vocab_allowed(g.start(), texts).tolist())
    assert ids == {0, 1, 2, 3, 4}


def test_parse_returns_the_rule_table():
    rules = parse('root ::= a\na ::= "x"')
    assert set(rules) == {"root", "a"}


def test_errors_name_the_problem():
    with pytest.raises(ValueError, match="root"):
        Grammar.from_string('start ::= "a"')
    with pytest.raises(ValueError, match="unknown rule"):
        Grammar.from_string('root ::= missing')
    with pytest.raises(ValueError, match="recursive"):
        Grammar.from_string('root ::= "(" root ")" | "x"')
    with pytest.raises(ValueError, match="GBNF line 2"):
        Grammar.from_string('root ::= "a"\nbogus')
    with pytest.raises(ValueError, match="unterminated"):
        Grammar.from_string('root ::= "a')


def test_a_repetition_bound_that_would_explode_is_refused():
    """A grammar is compiled to one state per literal character, so an unbounded bound is a
    memory request from whoever sent it: refused at parse time, before the automaton is built."""
    with pytest.raises(ValueError, match="expands to at most 10000 copies"):
        Grammar.from_string('root ::= "a"{1,100000}')
    with pytest.raises(ValueError, match="expands to at most 10000 copies"):
        Grammar.from_string('root ::= "a"{20000}')


def test_the_compiled_size_is_bounded_and_raiseable(monkeypatch):
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "8")
    with pytest.raises(ValueError, match="grammar too large"):
        Grammar.from_string('root ::= "abcdefghij"')      # ten literal states, limit eight
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "5000")
    g = Grammar.from_string('root ::= "abcdefghij"')
    assert g.done(g.advance(g.start(), "abcdefghij"))
    # a bad override is not a request failure: the default limit still applies
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "not-a-number")
    g = Grammar.from_string('root ::= "ok"')
    assert g.done(g.advance(g.start(), "ok"))


def _key(i):
    """A key the grammar's [a-z]+ accepts: a, b, ..., z, aa, ab, ..."""
    return string.ascii_lowercase[i % 26] * (1 + i // 26)


def _enumerated_record(pairs):
    """A flat JSON record whose pair group is enumerated `pairs` times: `ws` is referenced twice
    per copy (plus one at each end) and every copy re-expands `pair`'s key/value subtree. This is
    the reference-heavy shape the rule cache used to (wrongly) collapse: 40 copies is 82 uses."""
    return "\n".join([
        'root ::= ws "{" ws pair (ws "," ws pair){%d} ws "}" ws' % (pairs - 1),
        'pair ::= key ws ":" ws value',
        'key ::= "\\"" [a-z]+ "\\""',
        'value ::= string | number | "true" | "false" | "null"',
        'string ::= "\\"" [a-z0-9 ]* "\\""',
        'number ::= "-"? [0-9]+ ("." [0-9]+)?',
        'ws ::= (" " | "\\t" | "\\n")*',
    ])


def _record_text(keys):
    return "{" + ",".join('"%s": 1' % k for k in keys) + "}"


def test_a_reference_heavy_regular_grammar_fits_the_default_budget(monkeypatch):
    """The rule cache that shipped before 2026-11-17 made this grammar 20x smaller by letting
    every use share one copy of a rule - which is what changed the language: an ``exactly 3``
    grammar accepted 0-5 pairs (the cache let the closure jump from one use to a later use's
    successor). Expanding a rule once per use is exact, so the price is states, and the price is
    pinned here: a 40-pair record (82 `ws` uses) compiles under the default limit and accepts
    exactly 40 pairs."""
    monkeypatch.delenv("KINDI_GBNF_MAX_STATES", raising=False)
    g = Grammar.from_string(_enumerated_record(40))
    assert g.states < 6500, g.states                 # measured 5,915 on 2026-11-17
    keys = [_key(i) for i in range(40)]
    assert _accepts(g, _record_text(keys))
    assert not _accepts(g, _record_text(keys[:-1]))
    assert not _accepts(g, _record_text(keys + [_key(40)]))


def test_the_loop_form_of_the_same_constraint_costs_an_order_of_less(monkeypatch):
    """`pair (" ," pair)*` says the same practical thing - a record of one or more pairs at any
    length - in ~330 states regardless of the document, against ~143 states per enumerated copy.
    This is the alternative the refusal message names; zero pairs needs the group wrapped in `?`."""
    monkeypatch.delenv("KINDI_GBNF_MAX_STATES", raising=False)
    text = _enumerated_record(2).replace('(ws "," ws pair){1}', '(ws "," ws pair)*')
    g = Grammar.from_string(text)
    assert g.states < 400, g.states
    for n in (1, 40):
        assert _accepts(g, _record_text([_key(i) for i in range(n)]))
    assert not _accepts(g, _record_text([]))
    zero_or_more = Grammar.from_string(text.replace('pair (ws "," ws pair)*',
                                                   '(pair (ws "," ws pair)*)?'))
    assert _accepts(zero_or_more, _record_text([]))


def test_an_enumeration_past_the_budget_is_refused_with_the_alternative(monkeypatch):
    """The limit is a contract, not a crash: past the default it stays a ValueError whose message
    names the compact alternatives, and the documented operator escape hatch compiles the same
    grammar."""
    monkeypatch.delenv("KINDI_GBNF_MAX_STATES", raising=False)
    with pytest.raises(ValueError, match="grammar too large") as exc:
        Grammar.from_string(_enumerated_record(200))
    message = str(exc.value)
    assert "`*`/`+`" in message and "KINDI_GBNF_MAX_STATES" in message
    monkeypatch.setenv("KINDI_GBNF_MAX_STATES", "60000")
    g = Grammar.from_string(_enumerated_record(200))
    assert g.states > gbnf.DEFAULT_MAX_STATES        # the raise was genuinely needed


def test_a_reused_rule_cannot_lend_its_successors_to_another_use():
    """A built rule used twice must not share its entry/exit states. Sharing them let the
    epsilon closure after the first use jump straight to the second use's successor, so the
    automaton accepted a string that skipped the symbols between the two uses. This shipped
    before 2026-11-17 and hit every grammar that reuses a rule - `ws` in every JSON grammar -
    which is why a realistic grammar accepted `{"a": }`."""
    g = Grammar.from_string('root ::= a ":" a' + chr(10) + 'a ::= "x"')
    assert _accepts(g, "x:x")
    assert not _accepts(g, "x:") and not _accepts(g, ":")
    triple = Grammar.from_string('root ::= ws "a" ws "b" ws' + chr(10) + 'ws ::= " "*')
    assert _accepts(triple, " a b ")
    assert not _accepts(triple, "b a")


# ---------- the layouts people actually write ----------

# llama.cpp/grammars/json.gbnf, verbatim (MIT License, the ggml authors; see NOTICE).
# Its bodies start on the line after '::=' and
# continue across newlines inside parentheses - the parser used to refuse both, so the
# canonical file failed with "expected a rule name or a reference" before any recursion
# check could run.


def test_llamacpp_multiline_bodies_parse():
    """The canonical file parses; it never got this far before - the layout used to be the
    first error. The grammar is recursive, so compiling it is refused (third test below)."""
    rules = parse(CANONICAL_JSON)
    assert set(rules) == {"root", "value", "object", "array", "string", "number", "ws"}
    assert [ref[1] for ref in rules["object"][0] if ref[0] == "ref"] == ["ws", "ws"]


def test_the_canonical_json_grammar_is_refused_for_its_recursion():
    with pytest.raises(ValueError, match="recursive rule") as exc:
        Grammar.from_string(CANONICAL_JSON)
    message = str(exc.value)
    assert "object" in message and "value" in message     # the cycle is named, not just "no"


def test_multiline_bodies_parse_the_way_llamacpp_documents_them():
    g = Grammar.from_string('root ::=\n  "{" "a" "}"\n')          # body on the next line
    assert _accepts(g, "{a}")
    g = Grammar.from_string('root ::= (\n  "a"\n  "b"\n)\n')       # newlines inside parentheses
    assert _accepts(g, "ab") and not _accepts(g, "a")
    g = Grammar.from_string('root ::= "x" |\n  "y"\n')             # '|' continues the rule
    assert _accepts(g, "x") and _accepts(g, "y")
    with pytest.raises(ValueError, match="expected a rule name"):
        Grammar.from_string('root ::= "a"\n  "b"\n')               # a bare continuation line


JSONISH = r'''root ::= object
object ::= "{" ws ( string ":" ws values ("," ws string ":" ws values)* )? "}" ws
values ::= scalar | array
array ::= "[" ws ( scalar ("," ws scalar)* )? "]" ws
scalar ::= ( string | number | "true" | "false" | "null" ) ws
string ::= "\"" ( [^"\\\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
number ::= "-"? ( "0" | [1-9] [0-9]* ) ("." [0-9]+)? ( [eE] [-+]? [0-9]+ )?
ws ::= [ \t\n]*
'''


def test_a_regular_jsonish_grammar_is_accepted_and_matches_its_language():
    g = Grammar.from_string(JSONISH)
    for text in ['{}', '{"a": 1}', '{"a": "x\\ny"}', '{"a": [1, 2, 3]}',
                 '{"a": -1.5e+3, "b": true}', '{\n  "a": null\n}', '{"a": {"b": 1}}']:
        assert not _accepts(g, text) if text == '{"a": {"b": 1}}' else _accepts(g, text), text
    for text in ['[]', '1', '{"a": }', '{"a": 1', '{"a": 1} junk', '{"a" 1}',
                 '{"a": "bad\nbreak"}', '{"a": "\\q"}', '{"a": [,]}']:
        assert not _accepts(g, text), text


TOOL_CALL = r'''root ::= "{\"name\": \"" tool "\", \"arguments\": \"" arg "\"}"
tool ::= "get_weather" | "search_docs"
arg ::= [a-zA-Z0-9 ,._-]{1,64}
'''


def test_a_tool_call_grammar():
    g = Grammar.from_string(TOOL_CALL)
    assert _accepts(g, '{"name": "get_weather", "arguments": "Paris, FR"}')
    assert _accepts(g, '{"name": "search_docs", "arguments": "a"}')
    assert not _accepts(g, '{"name": "rm -rf", "arguments": "x"}')
    assert not _accepts(g, '{"name": "get_weather", "arguments": "%s"}' % ("x" * 65))
    assert not _accepts(g, '{"name": "get_weather", "arguments": "a\\"b"}')


def test_enum_and_iso_date_grammars():
    enum = Grammar.from_string(r'root ::= "\"" ("red" | "green" | "blue") "\""')
    assert _accepts(enum, '"green"')
    for bad in ['"purple"', '"Green"', "green", '"green']:
        assert not _accepts(enum, bad), bad

    date = Grammar.from_string(
        'root ::= date "T" time "Z"\n'
        'date ::= [0-9]{4} "-" month "-" day\n'
        'month ::= "0" [1-9] | "1" [0-2]\n'
        'day ::= "0" [1-9] | [12] [0-9] | "3" [01]\n'
        'time ::= [01][0-9] ":" [0-5][0-9] ":" [0-5][0-9]\n')
    assert _accepts(date, "2026-11-17T09:05:59Z")
    for bad in ["2026-13-01T00:00:00Z", "2026-1-1T00:00:00Z", "2026-11-32T00:00:00Z",
                "2026-11-17T24:00:00Z", "2026-11-17"]:
        assert not _accepts(date, bad), bad
