# SPDX-License-Identifier: AGPL-3.0-only
# Third-party material: the grammar string below is copied verbatim from llama.cpp's
# grammars/json.gbnf (https://github.com/ggml-org/llama.cpp), MIT License,
# Copyright (c) 2023-2026 The ggml authors. The MIT text is in NOTICE.
"""The recursive json.gbnf from llama.cpp's own docs, shared by the GBNF tests."""

CANONICAL_JSON = r'''root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
'''
