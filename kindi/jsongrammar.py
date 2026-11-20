# SPDX-License-Identifier: AGPL-3.0-only
"""JSON schema constrained decoding (REQ-0018), as a character-level automaton.

``JsonGrammar`` compiles a JSON schema into a small deterministic automaton over the
reply text: ``start()``, ``done(state)`` and ``advance(state, text)`` — the same
shape as the Gemma 4 ``CallGrammar`` in ``kindi/gemma4_tools.py``. Object properties
are emitted in schema order (like llama.cpp's ``json-schema-to-grammar``); a
``required`` property must be emitted before ``}``, optional ones may be skipped.

This is decode-side constraint, not a post-hoc prompt: the sampler only ever picks
token ids whose decoded text the automaton accepts, so the finished reply parses
against the schema by construction. ``vocab_allowed(state, texts)`` uses the token-class table
(``kindi.token_classes``), bulk-accepting a state's universal character loops
and checking the remaining token texts exactly; the server's constrained
sampler intersects the result with the engine head logits.

A state is a small tuple, so a candidate token is checked on a copy:
``advance(state, text) -> new state or None``. The automaton is forward-complete:
every token it accepts keeps a completion possible, so a non-empty allowed set
exists at every live state.

What a schema may say (v1; anything else is a ``ValueError`` at compile time and
becomes an enveloped 400 in the server):
  - ``type``: "object", "array", "string", "number", "integer", "boolean", "null",
    or a list of those. A node with ``properties`` is an object even without a type.
  - ``properties``, ``required``, ``items`` for objects and arrays.
  - ``enum`` / ``const``: primitive (string / number / boolean / null) literals only.
  - ``$schema``, ``title``, ``description``: ignored metadata.
  - ``additionalProperties``: only schema properties are accepted (OpenAI strict-mode
    behaviour). Rejecting unlisted keys keeps every produced document schema-valid.

``integer`` is the canonical form ``-?(0|[1-9][0-9]*)``; ``1.0`` for an integer
property would be rejected (the model emits canonical JSON).
"""
import json
import re

_WS = " \t\n\r"
_NUM = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\Z")
_INT = re.compile(r"-?(0|[1-9][0-9]*)\Z")
# The prefix language of each: every number buffer is a prefix of some complete
# match, so the automaton never paints itself into a dead number ("1.2.3", "1e2.3").
_NUM_PRE = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]*)?([eE][+-]?[0-9]*)?\Z")
_INT_PRE = re.compile(r"-?(0|[1-9][0-9]*)\Z")
_META = ("$schema", "title", "description")
_SUPPORTED_KEYS = _META + ("type", "properties", "required", "items", "enum", "const",
                           "additionalProperties")
_TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")
_ANY_LEAVES = (("string",), ("number",), ("bool",), ("null",))


def _json_text(value):
    """The canonical JSON text of a primitive literal (`1` and `1.0` differ)."""
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError("enum/const only support primitive literals (string, number, boolean, null)")


def _node(schema, path):
    """Compile a JSON schema subtree to a grammar node or raise ValueError."""
    if not isinstance(schema, dict):
        raise ValueError("%s: a schema must be an object" % path)
    bad = sorted(set(schema) - set(_SUPPORTED_KEYS))
    if bad:
        raise ValueError("%s: unsupported schema keys %s" % (path, ", ".join(bad)))
    enum = schema.get("enum")
    enum = enum if enum is not None else ([schema["const"]] if "const" in schema else None)
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise ValueError("%s: enum must be a non-empty array" % path)
        texts = sorted({_json_text(v) for v in enum})
        return ("lit", tuple(texts))
    typ = schema.get("type")
    types = None
    if typ is not None:
        types = [typ] if isinstance(typ, str) else typ
        if not isinstance(types, list) or not types or any(t not in _TYPES for t in types):
            raise ValueError("%s: type must be one of %s" % (path, ", ".join(_TYPES)))
        types = tuple(dict.fromkeys(types))
    if schema.get("properties") is not None:
        types = types or ("object",)
    if types is None:
        return ("any",)
    if len(types) > 1:
        branches = []
        for t in types:
            merged = {k: schema[k] for k in schema if k != "type"}
            merged["type"] = t
            branches.append(_node(merged, path))
        return ("union", tuple(branches))
    kind = types[0]
    if kind == "object":
        return _object_node(schema, path)
    if kind == "array":
        items = schema.get("items")
        if items is None:
            return ("any_arr",)
        if not isinstance(items, dict):
            raise ValueError("%s: items must be a schema object" % path)
        return ("array", _node(items, path + ".items"))
    if kind == "string":
        return ("string",)
    if kind == "number":
        return ("number",)
    if kind == "integer":
        return ("int",)
    if kind == "boolean":
        return ("bool",)
    return ("null",)


def _object_node(schema, path):
    props = schema.get("properties")
    if props is None:
        return ("any_obj",)
    if not isinstance(props, dict) or not all(isinstance(k, str) for k in props):
        raise ValueError("%s: properties must be an object" % path)
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(r, str) for r in required):
        raise ValueError("%s: required must be an array of property names" % path)
    out = []
    seen = set()
    for name, sub in props.items():
        if name in seen:
            raise ValueError("%s: duplicate property %r" % (path, name))
        seen.add(name)
        out.append((name, _node(sub, path + "." + name), name in required))
    extra = sorted(set(required) - seen)
    if extra:
        raise ValueError("%s: required names undeclared properties: %s" % (path, ", ".join(extra)))
    return ("object", tuple(out))


class JsonGrammar(object):
    """A schema-ordered JSON automaton. States are small tuples; advance() copies."""

    def __init__(self, node):
        self.root = node
        self._vocab_cache = {}
        self._mask_table = None
        self._mask_texts = None

    def start(self):
        return ("val", (), "", None)

    def done(self, st):
        return st is not None and st[0] == "trail" and not st[1]

    def advance(self, st, text):
        for ch in (text or ""):
            st = self._step(st, ch)
            if st is None:
                return None
        return st

    def _token_safe_sets(self, state, alphabet):
        """Character sets this state accepts for any length.

        ``str`` and ``akey`` are the JSON string-body loops: every character
        except a closing quote, a backslash, or a control character returns the
        same state.  The whitespace prefixes in front of every structural
        token have the analogous, narrower loop.
        """
        mode = state[0]
        if mode in ("str", "akey"):
            safe = frozenset(ch for ch in alphabet
                             if ord(ch) >= 0x20 and ch not in '"' and ch != "\\")
            return [safe] if safe else []
        if mode in ("trail", "val", "aval", "oval", "aempty", "otail", "atail",
                    "objk", "objkb", "objkf", "objka", "ocolon"):
            safe = frozenset(ch for ch in _WS if ch in alphabet)
            return [safe] if safe else []
        return []

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

    # --- completion ---------------------------------------------------------

    def _after_value(self, stk):
        """A value just completed: continue the parent, or finish at the root."""
        if not stk:
            return ("trail", (), "", None)
        top = stk[-1]
        if top[0] == "obj":
            _, node, idx, seen = top
            if node[0] == "object":
                props = node[1]
                if idx < len(props) and props[idx][2]:
                    seen |= (1 << idx)
            return ("otail", stk[:-1] + (("obj", node, idx + 1, seen),), "", None)
        return ("atail", stk, "", None)

    def _close(self, stk, ch):
        """Close the container on top of the stack (an object or an array)."""
        if not stk:
            return None
        top = stk[-1]
        want = "}" if top[0] == "obj" else "]"
        if ch != want:
            return None
        if top[0] == "obj" and top[1][0] == "object":
            props = top[1][1]
            for i, (_, _, req) in enumerate(props):
                if req and not (top[3] & (1 << i)):
                    return None                    # a required property is unmet
        return self._after_value(stk[:-1])

    # --- the character step -------------------------------------------------

    def _step(self, st, ch):
        mode, stk, buf, aux = st

        if mode == "trail":
            return st if ch in _WS else None

        if mode in ("val", "aval", "oval"):
            if ch in _WS:
                return st
            if mode == "val":
                return self._value_start(stk, self.root, ch)
            if mode == "aval":
                node = stk[-1][1]
                if node[0] == "array":
                    return self._value_start(stk, node[1], ch)
                return self._value_start(stk, ("any",), ch)
            top = stk[-1]
            onode = top[1]
            sub = onode[1][top[2]][1] if onode[0] == "object" else ("any",)
            return self._value_start(stk, sub, ch)

        if mode == "aempty":
            if ch in _WS:
                return st
            if ch == "]":
                return self._close(stk, ch)
            node = stk[-1][1]
            if node[0] == "array":
                return self._value_start(stk, node[1], ch)
            return self._value_start(stk, ("any",), ch)

        if mode in ("otail", "atail"):
            if ch in _WS:
                return st
            if ch == ",":
                if mode == "otail":
                    node = stk[-1][1]
                    if node[0] == "object" and stk[-1][2] >= len(node[1]):
                        return None       # no further property can follow
                    return ("objka" if node[0] == "any_obj" else "objkb", stk, "", None)
                return ("aval", stk, "", None)
            return self._close(stk, ch)

        if mode in ("objk", "objkb", "objkf", "objka"):
            if ch in _WS:
                return st
            if mode in ("objk", "objkf") and ch == "}":
                return self._close(stk, ch)
            if ch != '"':
                return None
            return ("akey" if mode in ("objkf", "objka") else "okey", stk, "", None)

        if mode == "okey":
            return self._key_char(stk, buf, ch)

        if mode == "akey":
            if ch == '"':
                return ("ocolon", stk, "", None)
            if ch == "\\":
                return ("sx", stk, "", "akey")
            return None if ord(ch) < 0x20 else st

        if mode == "ocolon":
            if ch in _WS:
                return st
            return ("oval", stk, "", None) if ch == ":" else None

        if mode == "str":
            if ch == '"':
                return self._after_value(stk)
            if ch == "\\":
                return ("sx", stk, "", "str")
            return None if ord(ch) < 0x20 else st

        if mode == "sx":
            if ch in '"\\/bfnrt':
                return (aux or "str", stk, "", None)
            if ch == "u":
                return ("su", stk, "", aux or "str")
            return None

        if mode == "su":
            if ch in "0123456789abcdefABCDEF":
                if len(buf) < 3:
                    return ("su", stk, buf + ch, aux)
                return (aux or "str", stk, "", None)
            return None

        if mode in ("num", "int"):
            if ch in "0123456789.eE+-" and len(buf) < 40:
                pre = _NUM_PRE if mode == "num" else _INT_PRE
                if pre.match(buf + ch):
                    return (mode, stk, buf + ch, None)
                return None
            ok = (_NUM if mode == "num" else _INT).match(buf)
            if not ok:
                return None
            return self._step(self._after_value(stk), ch)

        if mode == "kw":
            cands = aux or ()
            buf2 = buf + ch
            hits = [c for c in cands if c.startswith(buf2)]
            if not hits:
                return None
            if buf2 in cands:
                return self._after_value(stk)
            return ("kw", stk, buf2, tuple(cands))

        return None

    # --- value dispatch -----------------------------------------------------

    def _value_start(self, stk, node, ch):
        kind = node[0]
        if kind == "object":
            if ch != "{":
                return None
            return ("objk", stk + (("obj", node, 0, 0),), "", None)
        if kind == "any_obj":
            if ch != "{":
                return None
            return ("objkf", stk + (("obj", node, 0, 0),), "", None)
        if kind in ("array", "any_arr"):
            if ch != "[":
                return None
            return ("aempty", stk + (("arr", node),), "", None)
        if kind == "string":
            return ("str", stk, "", None) if ch == '"' else None
        if kind == "number":
            return ("num", stk, ch, None) if ch in "-0123456789" else None
        if kind == "int":
            return ("int", stk, ch, None) if ch in "-0123456789" else None
        if kind == "bool":
            if ch == "t":
                return ("kw", stk, "t", ("true",))
            if ch == "f":
                return ("kw", stk, "f", ("false",))
            return None
        if kind == "null":
            return ("kw", stk, "n", ("null",)) if ch == "n" else None
        if kind == "lit":
            hits = [t for t in node[1] if t.startswith(ch)]
            if not hits:
                return None
            if ch in node[1] and len(hits) == 1:
                return self._after_value(stk)
            return ("kw", stk, ch, tuple(hits))
        if kind == "union":
            for sub in node[1]:
                st = self._value_start(stk, sub, ch)
                if st is not None:
                    return st
            return None
        # any: generic JSON value dispatch over the leaf constructors
        for node in _ANY_LEAVES + (("any_obj",), ("any_arr",)):
            st = self._value_start(stk, node, ch)
            if st is not None:
                return st
        return None

    # --- schema object keys --------------------------------------------------

    def _key_char(self, stk, buf, ch):
        """One char of a schema key; ``"` closes it."""
        top = stk[-1]
        _, node, idx, seen = top
        props = node[1]
        if ch == '"':
            first_unseen_req = None
            for i in range(idx, len(props)):
                if props[i][2] and not (seen & (1 << i)):
                    first_unseen_req = i
                    break
            for i in range(idx, len(props)):
                if props[i][0] == buf:
                    if first_unseen_req is not None and i > first_unseen_req:
                        return None              # in-order: the required one comes first
                    # remember the choice in the frame; "ocolon" dispatches its value
                    return ("ocolon", stk[:-1] + (("obj", node, i, seen),), "", None)
            # a key that is only a prefix of a longer name is rejected at the close quote
            return None
        if ord(ch) < 0x20:
            return None
        buf2 = buf + ch
        # the first unseen required property bounds what may start here: a key past
        # it is rejected at the close quote, so a buffer that only prefixes such a
        # key is already dead and is refused here instead
        limit = len(props)
        for i in range(idx, len(props)):
            if props[i][2] and not (seen & (1 << i)):
                limit = i + 1
                break
        for i in range(idx, limit):
            if props[i][0].startswith(buf2):
                return ("okey", stk, buf2, None)
        return None


def grammar_from_schema(schema):
    """Build a JsonGrammar from a JSON schema dict (raises ValueError when invalid)."""
    return JsonGrammar(_node(schema, "$"))


def json_object_grammar():
    """A grammar for any JSON value (the ``json_object`` response format)."""
    return JsonGrammar(("any",))


__all__ = ["JsonGrammar", "grammar_from_schema", "json_object_grammar"]
